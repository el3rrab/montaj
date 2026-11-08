//! Speech-to-text with Gemini: [`GeminiTranscriber`] implements FilmCraft's [`Transcriber`].
//!
//! Audio goes inline (base64 WAV) in chunks of at most [`CHUNK_SECONDS`], cut at quiet points.
//! Gemini returns short timed segments (a few words each) as JSON; each segment's words share
//! its time span by length, then FilmCraft's VAD moves every word boundary onto the audio and
//! keeps pauses out of words — so `transcript.removePauses` sees the real silences.

use filmcraft_project::{Transcript, Word};
use filmcraft_speech::{Options, ProgressFn, SAMPLE_RATE, SpeechError, Transcriber, seconds_tick};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AiError, Keys, http};

/// Longest chunk per request: 5 min of 16 kHz 16-bit mono is 9.6 MB, 12.8 MB as base64 —
/// under Gemini's 20 MB inline request limit.
pub const CHUNK_SECONDS: usize = 300;

const SYSTEM: &str = "You are a precise verbatim transcriber for Egyptian Arabic video recordings.";

const PROMPT: &str = "Transcribe this audio verbatim, in the language spoken (usually Egyptian Arabic, written in Arabic script; keep English words as spoken).\n\
Rules:\n\
- Write exactly what is said: keep fillers (اممم، آآ، إإ، يعني، بص، إيه), repeated words, stutters and false starts. Never correct, summarize or skip.\n\
- Split into short segments of 1 to 4 words, following the speech. Never more than 4 words or 2.5 seconds in one segment.\n\
- Give every disfluency its own segment and its kind:\n\
  - \"stutter\": a word or syllable said twice in a row, or a word cut off and restarted (أنا أنا، ال… الطيارة).\n\
  - \"filler\": a hesitation sound or filler word that carries no meaning here (اممم، آآ، إإ، يعني، بص used as a pause).\n\
  - \"false_start\": a sentence started and abandoned before the host restarts it.\n\
  - \"aside\": talking to someone in the room or about the recording itself, not to the audience (e.g. to a child: يا ابني، روحي عند أمك).\n\
  - \"noise\": laughter, coughing or background sounds with no words for the audience.\n\
  - \"speech\": everything else.\n\
- start and end are seconds from the beginning of this audio, with two decimals.\n\
- A segment never covers silence: end it where the voice stops; start the next where the voice starts again.\n\
- Leave out music and long silences.";

pub struct GeminiTranscriber {
    key: String,
    model: String,
}

/// Token use of one request (Gemini's `usageMetadata`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub thinking: u64,
}

#[derive(Clone, Debug, serde::Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// `speech`, `stutter`, `filler` or `false_start` (see the prompt).
    #[serde(default = "speech_kind")]
    pub kind: String,
}

fn speech_kind() -> String {
    "speech".into()
}

impl Segment {
    pub fn new(start: f64, end: f64, text: impl Into<String>) -> Self {
        Self { start, end, text: text.into(), kind: speech_kind() }
    }
    /// A stutter, filler, false start, aside or noise: cut by default.
    pub fn disfluent(&self) -> bool {
        matches!(self.kind.as_str(), "stutter" | "filler" | "false_start" | "aside" | "noise")
    }
}

const CLIP_PROMPT: &str = "The audio parts above are numbered clips cut in order from one raw recording of a YouTube host (usually Egyptian Arabic). Each clip is one stretch of voice between pauses.\n\
For every clip, write exactly what is said in it (Arabic script; English words as spoken). Keep fillers, repeated words, stutters and cut-off words. Never correct, summarize, merge or skip clips.\n\
Give each clip one kind:\n\
- \"speech\": normal content for the audience.\n\
- \"stutter\": only a repeated or cut-off word/syllable (أنا أنا، ال…).\n\
- \"filler\": only a hesitation or filler (اممم، آآ، إإ، يعني used as a pause).\n\
- \"false_start\": a sentence started and abandoned (the host restarts it in a later clip).\n\
- \"aside\": talking to someone in the room or about the recording itself, not to the audience (e.g. to a child: يا ابني، روحي عند أمك).\n\
- \"noise\": no voice, laughter, coughing or background sounds only.\n\
Answer with one entry per clip, in order.";

/// Stretches of voice in 16 kHz audio, as sample ranges: voiced runs (FilmCraft's VAD) joined
/// across gaps shorter than `join` seconds, at least `min_len` long, split above `max_len` at
/// the quietest frame, each padded by `pad` seconds into the surrounding silence.
pub fn voice_ranges(audio: &[f32], join: f64, min_len: f64, max_len: f64, pad: f64) -> Vec<std::ops::Range<usize>> {
    let db = filmcraft_speech::vad::frame_db(audio);
    if db.is_empty() {
        return Vec::new();
    }
    let th = filmcraft_speech::vad::threshold(&db);
    let f = |s: f64| (s * 100.0).round() as usize; // 10 ms frames
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < db.len() {
        if db[i] < th {
            i += 1;
            continue;
        }
        let a = i;
        while i < db.len() && db[i] >= th {
            i += 1;
        }
        match runs.last_mut() {
            Some(last) if a - last.1 < f(join) => last.1 = i,
            _ => runs.push((a, i)),
        }
    }
    let mut split = Vec::new();
    for (a, b) in runs.into_iter().filter(|(a, b)| b - a >= f(min_len)) {
        let mut a = a;
        while b - a > f(max_len) {
            // quietest frame in the middle half of the allowed span
            let lo = a + f(max_len) / 2;
            let hi = a + f(max_len);
            let cut = (lo..hi).min_by(|&x, &y| db[x].total_cmp(&db[y])).unwrap_or(hi);
            split.push((a, cut));
            a = cut;
        }
        split.push((a, b));
    }
    let n = audio.len();
    let mut out: Vec<std::ops::Range<usize>> = Vec::new();
    for (k, &(a, b)) in split.iter().enumerate() {
        let prev_end = if k > 0 { split[k - 1].1 * 160 } else { 0 };
        let next_start = split.get(k + 1).map_or(n, |s| s.0 * 160);
        let p = (pad * 16_000.0) as usize;
        let s = (a * 160).saturating_sub(p).max((prev_end + a * 160) / 2);
        let e = (b * 160 + p).min(n).min((next_start + b * 160).div_ceil(2));
        if e > s {
            out.push(s..e);
        }
    }
    out
}

#[derive(Deserialize)]
struct ClipAnswer {
    clip: usize,
    #[serde(default)]
    text: String,
    #[serde(default = "speech_kind")]
    kind: String,
}

impl GeminiTranscriber {
    pub fn new(key: impl Into<String>, model: Option<String>) -> Self {
        Self { key: key.into(), model: model.filter(|m| !m.is_empty()).unwrap_or_else(|| crate::llm::GEMINI_MODEL.into()) }
    }

    pub fn from_keys(k: &Keys) -> Option<Self> {
        k.gemini.clone().map(|key| Self::new(key, k.gemini_model.clone()))
    }

    /// Transcribe one chunk (16 kHz mono) into timed segments, relative to the chunk's start.
    pub fn chunk(&self, audio: &[f32]) -> Result<(Vec<Segment>, Usage), AiError> {
        let wav = crate::wav::encode(audio, SAMPLE_RATE);
        let body = json!({
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": base64(&wav)}},
                {"text": PROMPT},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 65536,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "start": {"type": "NUMBER"},
                            "end": {"type": "NUMBER"},
                            "text": {"type": "STRING"},
                            "kind": {"type": "STRING", "enum": ["speech", "stutter", "filler", "false_start", "aside", "noise"]},
                        },
                        "required": ["start", "end", "text", "kind"],
                    },
                },
            },
        });
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", self.model);
        let reply = http::post_json("Gemini", &url, &[("x-goog-api-key", &self.key)], &body)?;
        parse(&http::check("Gemini", reply)?)
    }

    /// Timed transcript of `audio` with reliable times: sent in pieces of at most
    /// [`TIMED_PIECE_SECONDS`] (Gemini's timestamps drift on longer audio), each piece's times
    /// stretched back to its real length if they overrun, then every boundary snapped onto the
    /// nearest edge of real voice (FilmCraft's VAD). Times are seconds from `audio`'s start.
    pub fn timed(&self, audio: &[f32], progress: &mut dyn FnMut(usize, usize) -> bool) -> Result<(Vec<Segment>, Usage), AiError> {
        let sr = SAMPLE_RATE as usize;
        let pieces = crate::wav::split_points(audio, TIMED_PIECE_SECONDS * sr, 10 * sr, sr / 2);
        let mut all = Vec::new();
        let mut total = Usage::default();
        let n = pieces.len();
        // up to PARALLEL requests at once: a 25-minute recording is ~25 pieces
        for (b, batch) in pieces.chunks(PARALLEL).enumerate() {
            if !progress(b * PARALLEL, n) {
                return Err(AiError::Cancelled);
            }
            let results: Vec<Result<(Vec<Segment>, Usage), AiError>> = std::thread::scope(|sc| {
                let handles: Vec<_> = batch.iter().map(|r| sc.spawn(|| self.chunk(&audio[r.clone()]))).collect();
                handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(AiError::Io("transcription thread failed".into())))).collect()
            });
            for (r, res) in batch.iter().zip(results) {
                let (mut segs, u) = res?;
                let offset = r.start as f64 / sr as f64;
                fit_to_length(&mut segs, r.len() as f64 / sr as f64);
                for s in &mut segs {
                    s.start += offset;
                    s.end += offset;
                }
                all.extend(segs);
                total.input += u.input;
                total.output += u.output;
                total.thinking += u.thinking;
            }
        }
        let voice = voice_ranges(audio, 0.12, 0.08, 3600.0, 0.0);
        snap_to_voice(&mut all, &voice, sr as f64, 0.6);
        Ok((all, total))
    }

    /// Clip mode: the times come from the audio (`ranges`, e.g. [`voice_ranges`]); Gemini only
    /// writes each clip's text and kind. Returns one segment per clip it answered for, in seconds
    /// relative to `audio`'s start.
    pub fn clips(&self, audio: &[f32], ranges: &[std::ops::Range<usize>]) -> Result<(Vec<Segment>, Usage), AiError> {
        if ranges.is_empty() {
            return Ok((Vec::new(), Usage::default()));
        }
        let mut parts = Vec::with_capacity(ranges.len() * 2 + 1);
        for (k, r) in ranges.iter().enumerate() {
            parts.push(json!({"text": format!("Clip {k}:")}));
            let wav = crate::wav::encode(&audio[r.clone()], SAMPLE_RATE);
            parts.push(json!({"inlineData": {"mimeType": "audio/wav", "data": base64(&wav)}}));
        }
        parts.push(json!({"text": CLIP_PROMPT}));
        let body = json!({
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 65536,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "clip": {"type": "INTEGER"},
                            "text": {"type": "STRING"},
                            "kind": {"type": "STRING", "enum": ["speech", "stutter", "filler", "false_start", "aside", "noise"]},
                        },
                        "required": ["clip", "text", "kind"],
                    },
                },
            },
        });
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", self.model);
        let reply = http::post_json("Gemini", &url, &[("x-goog-api-key", &self.key)], &body)?;
        let body = http::check("Gemini", reply)?;
        let (answers, usage) = parse_clips(&body)?;
        Ok((clip_segments(ranges, answers), usage))
    }
}

/// Longest piece sent for a timed transcript.
pub const TIMED_PIECE_SECONDS: usize = 60;

/// Requests sent to Gemini at the same time.
const PARALLEL: usize = 6;

/// Drop segments with bad times and, when the last end overruns the piece's real length,
/// stretch every time by `len / last_end` (Gemini's clock ran fast).
pub fn fit_to_length(segs: &mut Vec<Segment>, len: f64) {
    segs.retain(|s| s.start.is_finite() && s.end.is_finite() && s.end >= s.start && s.start >= 0.0);
    let last = segs.iter().map(|s| s.end).fold(0.0f64, f64::max);
    if last > len && last > 0.0 {
        let k = len / last;
        for s in segs.iter_mut() {
            s.start *= k;
            s.end *= k;
        }
    }
}

/// Move each segment's start to the nearest voice start and its end to the nearest voice end
/// within `reach` seconds, keeping segments in order and non-overlapping.
pub fn snap_to_voice(segs: &mut [Segment], voice: &[std::ops::Range<usize>], sr: f64, reach: f64) {
    let starts: Vec<f64> = voice.iter().map(|r| r.start as f64 / sr).collect();
    let ends: Vec<f64> = voice.iter().map(|r| r.end as f64 / sr).collect();
    let nearest = |edges: &[f64], t: f64| -> f64 { edges.iter().copied().filter(|e| (e - t).abs() <= reach).min_by(|a, b| (a - t).abs().total_cmp(&(b - t).abs())).unwrap_or(t) };
    let mut floor = 0.0f64;
    for s in segs.iter_mut() {
        let a = nearest(&starts, s.start).max(floor);
        let b = nearest(&ends, s.end).max(a);
        s.start = a;
        s.end = b;
        floor = b;
    }
}

fn parse_clips(body: &str) -> Result<(Vec<ClipAnswer>, Usage), AiError> {
    let v: Value = serde_json::from_str(body).map_err(|e| AiError::BadResponse("Gemini", e.to_string()))?;
    if let Some(r) = v.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Err(AiError::Refused("Gemini", r.to_string()));
    }
    let u = |k: &str| v.pointer(&format!("/usageMetadata/{k}")).and_then(Value::as_u64).unwrap_or(0);
    let usage = Usage { input: u("promptTokenCount"), output: u("candidatesTokenCount"), thinking: u("thoughtsTokenCount") };
    let text: String = v
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(|ps| ps.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect())
        .unwrap_or_default();
    if text.trim().is_empty() {
        return Ok((Vec::new(), usage));
    }
    let a: Vec<ClipAnswer> = serde_json::from_str(&text).map_err(|e| AiError::BadResponse("Gemini", format!("clips: {e}")))?;
    Ok((a, usage))
}

/// Segments from clip answers: times from the clip ranges; unknown clip numbers and repeats
/// are ignored; a clip Gemini skipped stays as speech with no text, so it is never cut blindly.
fn clip_segments(ranges: &[std::ops::Range<usize>], answers: Vec<ClipAnswer>) -> Vec<Segment> {
    let sr = SAMPLE_RATE as f64;
    let mut by_clip: Vec<Option<ClipAnswer>> = ranges.iter().map(|_| None).collect();
    for a in answers {
        if let Some(slot) = by_clip.get_mut(a.clip)
            && slot.is_none()
        {
            *slot = Some(a);
        }
    }
    ranges
        .iter()
        .zip(by_clip)
        .map(|(r, a)| {
            let (text, kind) = a.map(|a| (a.text, a.kind)).unwrap_or_else(|| (String::new(), speech_kind()));
            Segment { start: r.start as f64 / sr, end: r.end as f64 / sr, text, kind }
        })
        .collect()
}

fn parse(body: &str) -> Result<(Vec<Segment>, Usage), AiError> {
    let v: Value = serde_json::from_str(body).map_err(|e| AiError::BadResponse("Gemini", e.to_string()))?;
    if let Some(r) = v.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Err(AiError::Refused("Gemini", r.to_string()));
    }
    let u = |k: &str| v.pointer(&format!("/usageMetadata/{k}")).and_then(Value::as_u64).unwrap_or(0);
    let usage = Usage { input: u("promptTokenCount"), output: u("candidatesTokenCount"), thinking: u("thoughtsTokenCount") };
    let text: String = v
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(|ps| ps.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect())
        .unwrap_or_default();
    if text.trim().is_empty() {
        // nothing said in this chunk
        return Ok((Vec::new(), usage));
    }
    let segs: Vec<Segment> = serde_json::from_str(&text).map_err(|e| AiError::BadResponse("Gemini", format!("segments: {e}")))?;
    Ok((segs, usage))
}

/// Words of `segments` (shifted by `offset` seconds): each segment's span is shared among its
/// words by length. Segments out of order or overlapping are clamped so words stay in order.
pub fn words(segments: &[Segment], offset: f64, limit: f64) -> Vec<Word> {
    let mut out = Vec::new();
    let mut floor = 0.0f64;
    for s in segments {
        if !s.start.is_finite() || !s.end.is_finite() {
            continue;
        }
        let start = s.start.max(floor).min(limit);
        let end = s.end.max(start).min(limit);
        let parts: Vec<&str> = s.text.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let weight: f64 = parts.iter().map(|p| p.chars().count() as f64 + 1.0).sum();
        let mut t = start;
        for p in parts {
            let d = (end - start) * (p.chars().count() as f64 + 1.0) / weight;
            let mut w = Word::new(p, seconds_tick(offset + t), seconds_tick(offset + t + d));
            w.speaker = Some(0);
            out.push(w);
            t += d;
        }
        floor = end;
    }
    out
}

/// Standard base64 (RFC 4648, padded).
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let b = [c[0], c.get(1).copied().unwrap_or(0), c.get(2).copied().unwrap_or(0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

fn speech(e: AiError) -> SpeechError {
    match e {
        AiError::Cancelled => SpeechError::Cancelled,
        other => SpeechError::Model(other.to_string()),
    }
}

impl Transcriber for GeminiTranscriber {
    fn id(&self) -> String {
        format!("gemini-{}", self.model)
    }

    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        let len = audio.len() as f64 / SAMPLE_RATE as f64;
        let (segs, _) = self
            .timed(audio, &mut |i, n| progress(i as f32 / n.max(1) as f32 * 0.9, &format!("Sending audio to Gemini ({}/{n})", i + 1)))
            .map_err(speech)?;
        let all = words(&segs, 0.0, len);
        let language = opts.language.clone().filter(|l| !l.is_empty() && l != "auto").unwrap_or_else(|| "ar".into());
        let mut t = Transcript { language, source: self.id(), speakers: Vec::new(), words: all };
        t.normalize();
        filmcraft_speech::vad::tighten_words(audio, &mut t.words);
        if opts.diarize && !t.words.is_empty() {
            if !progress(0.96, "Labelling speakers") {
                return Err(SpeechError::Cancelled);
            }
            let p = filmcraft_speech::diarize::Params { max_speakers: opts.max_speakers, ..Default::default() };
            filmcraft_speech::diarize::diarize(audio, &mut t, &p);
        }
        progress(1.0, "Done");
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_ranges_follow_the_audio() {
        // voice at 1.0-2.0 s and 2.1-2.6 s (joined: gap < 0.2 s) and 4.0-5.0 s
        let mut a: Vec<f32> = (0..96_000).map(|i| if i % 2 == 0 { 0.0005 } else { -0.0005 }).collect();
        for (f, t) in [(1.0, 2.0), (2.1, 2.6), (4.0, 5.0)] {
            for i in (f * 16_000.0) as usize..(t * 16_000.0) as usize {
                a[i] = 0.3 * ((i as f32) * 0.2).sin();
            }
        }
        let r = voice_ranges(&a, 0.2, 0.15, 8.0, 0.05);
        assert_eq!(r.len(), 2, "{r:?}");
        let s = |x: usize| x as f64 / 16_000.0;
        assert!((s(r[0].start) - 0.95).abs() < 0.02 && (s(r[0].end) - 2.65).abs() < 0.02, "{r:?}");
        assert!((s(r[1].start) - 3.95).abs() < 0.02 && (s(r[1].end) - 5.05).abs() < 0.02, "{r:?}");
        // a long stretch is split
        assert!(voice_ranges(&a, 10.0, 0.15, 1.0, 0.0).len() >= 3);
        assert!(voice_ranges(&[], 0.2, 0.15, 8.0, 0.05).is_empty());
    }

    #[test]
    fn clip_answers_map_to_clip_times() {
        let ranges = vec![16_000..32_000, 48_000..64_000, 70_000..80_000];
        let answers = vec![
            ClipAnswer { clip: 1, text: "يا ابني".into(), kind: "aside".into() },
            ClipAnswer { clip: 0, text: "اهلا".into(), kind: "speech".into() },
            ClipAnswer { clip: 0, text: "dup".into(), kind: "noise".into() },
            ClipAnswer { clip: 9, text: "x".into(), kind: "speech".into() },
        ];
        let s = clip_segments(&ranges, answers);
        assert_eq!(s.len(), 3);
        assert_eq!((s[0].start, s[0].end, s[0].text.as_str(), s[0].kind.as_str()), (1.0, 2.0, "اهلا", "speech"));
        assert!(s[1].disfluent());
        // clip 2 was skipped by the model: kept as speech, never cut blindly
        assert_eq!((s[2].text.as_str(), s[2].kind.as_str()), ("", "speech"));
        let body = r#"{"candidates":[{"content":{"parts":[{"text":"[{\"clip\":0,\"text\":\"a\",\"kind\":\"filler\"}]"}]}}],"usageMetadata":{"promptTokenCount":5}}"#;
        let (a, u) = parse_clips(body).unwrap();
        assert_eq!((a[0].clip, a[0].kind.as_str(), u.input), (0, "filler", 5));
    }

    #[test]
    fn overrunning_times_are_stretched_back() {
        let mut s = vec![Segment::new(10.0, 20.0, "a"), Segment::new(70.0, 80.0, "b"), Segment::new(f64::NAN, 1.0, "x")];
        fit_to_length(&mut s, 60.0);
        assert_eq!(s.len(), 2);
        assert!((s[1].end - 60.0).abs() < 1e-9 && (s[0].start - 7.5).abs() < 1e-9);
        let mut ok = vec![Segment::new(1.0, 2.0, "a")];
        fit_to_length(&mut ok, 60.0);
        assert_eq!(ok[0].end, 2.0);
    }

    #[test]
    fn boundaries_snap_to_voice_edges() {
        let voice = vec![16_000..40_000, 48_000..64_000]; // 1.0-2.5 s and 3.0-4.0 s
        let mut s = vec![Segment::new(1.3, 2.2, "a"), Segment::new(2.8, 4.3, "b"), Segment::new(9.0, 9.5, "far")];
        snap_to_voice(&mut s, &voice, 16_000.0, 0.6);
        assert_eq!((s[0].start, s[0].end), (1.0, 2.5));
        assert_eq!((s[1].start, s[1].end), (3.0, 4.0));
        assert_eq!((s[2].start, s[2].end), (9.0, 9.5)); // nothing within reach
    }

    #[test]
    fn base64_matches_rfc_4648() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn parses_segments_and_usage() {
        let body = r#"{"candidates":[{"content":{"parts":[{"text":"[{\"start\":0.5,\"end\":1.5,\"text\":\"اممم يعني\"}]"}]}}],
            "usageMetadata":{"promptTokenCount":9600,"candidatesTokenCount":40}}"#;
        let (s, u) = parse(body).unwrap();
        assert_eq!(s, vec![Segment::new(0.5, 1.5, "اممم يعني")]);
        assert_eq!(u, Usage { input: 9600, output: 40, thinking: 0 });
        assert!(parse(r#"{"candidates":[{"content":{"parts":[{"text":"oops"}]}}]}"#).is_err());
        assert!(parse(r#"{"candidates":[]}"#).unwrap().0.is_empty());
    }

    #[test]
    fn words_share_the_segment_and_stay_in_order() {
        let segs = vec![
            Segment::new(1.0, 2.0, "ab cd"),
            // overlaps the first and runs past the chunk: clamped
            Segment::new(1.5, 400.0, "e"),
            Segment::new(3.0, f64::NAN, "skip"),
        ];
        let w = words(&segs, 300.0, 10.0);
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].start, seconds_tick(301.0));
        assert_eq!(w[1].end, seconds_tick(302.0));
        assert_eq!(w[2].start, seconds_tick(302.0));
        assert_eq!(w[2].end, seconds_tick(310.0));
        assert!(w.windows(2).all(|p| p[0].end <= p[1].start));
    }
}
