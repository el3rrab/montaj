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
- Split into short segments of 1 to 6 words, following the speech.\n\
- start and end are seconds from the beginning of this audio, with two decimals.\n\
- A segment never covers silence: end it where the voice stops; start the next where the voice starts again.\n\
- Leave out music, noises and long silences.";

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

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
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
                        "properties": {"start": {"type": "NUMBER"}, "end": {"type": "NUMBER"}, "text": {"type": "STRING"}},
                        "required": ["start", "end", "text"],
                    },
                },
            },
        });
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", self.model);
        let reply = http::post_json("Gemini", &url, &[("x-goog-api-key", &self.key)], &body)?;
        parse(&http::check("Gemini", reply)?)
    }
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
        let sr = SAMPLE_RATE as usize;
        let chunks = crate::wav::split_points(audio, CHUNK_SECONDS * sr, 20 * sr, sr / 2);
        let n = chunks.len().max(1);
        let mut all = Vec::new();
        for (i, r) in chunks.into_iter().enumerate() {
            if !progress(i as f32 / n as f32 * 0.9, &format!("Sending audio to Gemini ({}/{n})", i + 1)) {
                return Err(SpeechError::Cancelled);
            }
            let offset = r.start as f64 / sr as f64;
            let limit = r.len() as f64 / sr as f64;
            let (segs, _) = self.chunk(&audio[r]).map_err(speech)?;
            all.extend(words(&segs, offset, limit));
        }
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
        assert_eq!(s, vec![Segment { start: 0.5, end: 1.5, text: "اممم يعني".into() }]);
        assert_eq!(u, Usage { input: 9600, output: 40, thinking: 0 });
        assert!(parse(r#"{"candidates":[{"content":{"parts":[{"text":"oops"}]}}]}"#).is_err());
        assert!(parse(r#"{"candidates":[]}"#).unwrap().0.is_empty());
    }

    #[test]
    fn words_share_the_segment_and_stay_in_order() {
        let segs = vec![
            Segment { start: 1.0, end: 2.0, text: "ab cd".into() },
            // overlaps the first and runs past the chunk: clamped
            Segment { start: 1.5, end: 400.0, text: "e".into() },
            Segment { start: 3.0, end: f64::NAN, text: "skip".into() },
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
