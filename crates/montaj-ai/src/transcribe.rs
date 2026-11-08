//! Cloud speech-to-text: [`OpenAiTranscriber`] implements FilmCraft's [`Transcriber`].
//!
//! Long audio is cut into chunks of at most [`CHUNK_SECONDS`] at the quietest point near each
//! boundary (OpenAI takes files up to 25 MB; 10 minutes of 16 kHz 16-bit mono is 19 MB). Each
//! chunk is sent as WAV with `response_format=verbose_json` and word timestamps; word times are
//! shifted by the chunk's start, then tightened against the audio and labelled by speaker with
//! FilmCraft's own VAD and diarization, exactly like the built-in Whisper.

use filmcraft_project::{Transcript, Word};
use filmcraft_speech::{Options, ProgressFn, SAMPLE_RATE, SpeechError, Transcriber, seconds_tick};
use serde::Deserialize;

use crate::http::{self, Part};
use crate::{AiError, Keys};

/// Longest chunk sent in one request.
pub const CHUNK_SECONDS: usize = 600;
/// The default OpenAI speech model: it returns word-level timestamps.
pub const DEFAULT_MODEL: &str = "whisper-1";
const URL: &str = "https://api.openai.com/v1/audio/transcriptions";

/// Hint for Egyptian Arabic: write what was said, fillers included, so they can be cut later.
const ARABIC_PROMPT: &str = "تسجيل بالعامية المصرية. اكتب الكلام زي ما اتقال بالظبط، من غير تصحيح، وبالكلمات اللي بتتكرر وكلمات الحشو زي: اممم، آآ، يعني، إيه، بص.";

pub struct OpenAiTranscriber {
    key: String,
    model: String,
}

impl OpenAiTranscriber {
    pub fn new(key: impl Into<String>, model: Option<String>) -> Self {
        Self { key: key.into(), model: model.filter(|m| !m.is_empty()).unwrap_or_else(|| DEFAULT_MODEL.into()) }
    }

    /// From the loaded keys; `None` without an OpenAI key.
    pub fn from_keys(k: &Keys) -> Option<Self> {
        k.openai.clone().map(|key| Self::new(key, k.transcribe_model.clone()))
    }

    fn request(&self, wav: &[u8], language: Option<&str>) -> Result<String, AiError> {
        let auth = format!("Bearer {}", self.key);
        let mut parts = vec![
            Part::Text("model", &self.model),
            Part::Text("response_format", "verbose_json"),
            Part::Text("timestamp_granularities[]", "word"),
            Part::Text("timestamp_granularities[]", "segment"),
            Part::File { name: "file", filename: "audio.wav", content_type: "audio/wav", data: wav },
        ];
        if let Some(l) = language {
            parts.push(Part::Text("language", l));
        }
        if language.is_none_or(|l| l == "ar") {
            parts.push(Part::Text("prompt", ARABIC_PROMPT));
        }
        let reply = http::post_multipart("OpenAI", URL, &[("authorization", &auth)], &parts)?;
        http::check("OpenAI", reply)
    }
}

#[derive(Deserialize)]
struct Verbose {
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    words: Vec<VWord>,
}

#[derive(Deserialize)]
struct VWord {
    word: String,
    start: f64,
    end: f64,
}

/// ISO 639-1 code for the language names OpenAI reports (`"arabic"` → `"ar"`).
fn iso(lang: &str) -> String {
    match lang.to_lowercase().as_str() {
        "arabic" => "ar",
        "english" => "en",
        "french" => "fr",
        "german" => "de",
        "spanish" => "es",
        "turkish" => "tr",
        "italian" => "it",
        "portuguese" => "pt",
        "russian" => "ru",
        "japanese" => "ja",
        "chinese" => "zh",
        "korean" => "ko",
        "hindi" => "hi",
        "urdu" => "ur",
        "persian" => "fa",
        other if other.len() == 2 => other,
        _ => "und",
    }
    .to_string()
}

/// Parse one chunk's response into words shifted by `offset` seconds; also the language seen.
fn parse_chunk(body: &str, offset: f64) -> Result<(Vec<Word>, Option<String>), AiError> {
    let v: Verbose = serde_json::from_str(body).map_err(|e| AiError::BadResponse("OpenAI", e.to_string()))?;
    let words = v
        .words
        .into_iter()
        .filter(|w| w.start.is_finite() && w.end.is_finite() && !w.word.trim().is_empty())
        .map(|w| {
            let s = (w.start.max(0.0)) + offset;
            let e = (w.end.max(w.start)) + offset;
            let mut word = Word::new(w.word.trim(), seconds_tick(s), seconds_tick(e));
            word.speaker = Some(0);
            word
        })
        .collect();
    Ok((words, v.language.map(|l| iso(&l))))
}

/// The transcriber Montaj installs at start-up (`Session::transcriber`): OpenAI's when an OpenAI
/// key is set, else `None` (FilmCraft's own behaviour is unchanged).
pub fn installed(data_dir: Option<&std::path::Path>) -> Option<std::sync::Arc<dyn Transcriber>> {
    let keys = Keys::load(data_dir);
    if let Some(t) = OpenAiTranscriber::from_keys(&keys) {
        return Some(std::sync::Arc::new(t));
    }
    let t = crate::gemini_stt::GeminiTranscriber::from_keys(&keys)?;
    Some(std::sync::Arc::new(t))
}

fn speech(e: AiError) -> SpeechError {
    match e {
        AiError::Cancelled => SpeechError::Cancelled,
        AiError::MissingKey(..) => SpeechError::Unavailable(e.to_string()),
        other => SpeechError::Model(other.to_string()),
    }
}

impl Transcriber for OpenAiTranscriber {
    fn id(&self) -> String {
        format!("openai-{}", self.model)
    }

    fn transcribe(&self, audio: &[f32], opts: &Options, progress: ProgressFn) -> Result<Transcript, SpeechError> {
        let sr = SAMPLE_RATE as usize;
        let chunks = crate::wav::split_points(audio, CHUNK_SECONDS * sr, 20 * sr, sr / 2);
        let n = chunks.len().max(1);
        let language = opts.language.as_deref().filter(|l| !l.is_empty() && *l != "auto");
        let mut words: Vec<Word> = Vec::new();
        let mut seen_language = None;
        for (i, r) in chunks.into_iter().enumerate() {
            if !progress(i as f32 / n as f32 * 0.9, &format!("Sending audio to OpenAI ({}/{n})", i + 1)) {
                return Err(SpeechError::Cancelled);
            }
            let offset = r.start as f64 / sr as f64;
            let wav = crate::wav::encode(&audio[r], SAMPLE_RATE);
            let body = self.request(&wav, language).map_err(speech)?;
            let (w, l) = parse_chunk(&body, offset).map_err(speech)?;
            words.extend(w);
            seen_language = seen_language.or(l);
        }
        let mut t = Transcript {
            language: language.map(str::to_string).or(seen_language).unwrap_or_else(|| "und".into()),
            source: self.id(),
            speakers: Vec::new(),
            words,
        };
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
    fn parses_words_and_shifts_them_by_the_chunk_start() {
        let body = r#"{"task":"transcribe","language":"arabic","duration":3.2,"text":"اهلا يعني",
            "words":[{"word":"اهلا","start":0.5,"end":0.9},{"word":" يعني ","start":1.0,"end":1.4},
                     {"word":"","start":2.0,"end":2.1},{"word":"x","start":3.0,"end":2.0}]}"#;
        let (w, lang) = parse_chunk(body, 600.0).unwrap();
        assert_eq!(lang.as_deref(), Some("ar"));
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].text, "اهلا");
        assert_eq!(w[1].text, "يعني");
        assert_eq!(w[0].start, seconds_tick(600.5));
        assert_eq!(w[1].end, seconds_tick(601.4));
        // an end before the start is clamped to the start
        assert_eq!(w[2].start, w[2].end);
    }

    #[test]
    fn bad_json_is_an_error_not_a_panic() {
        assert!(matches!(parse_chunk("not json", 0.0), Err(AiError::BadResponse("OpenAI", _))));
        let (w, _) = parse_chunk(r#"{"text":"hi"}"#, 0.0).unwrap();
        assert!(w.is_empty());
    }

    #[test]
    fn language_names_map_to_iso() {
        assert_eq!(iso("Arabic"), "ar");
        assert_eq!(iso("en"), "en");
        assert_eq!(iso("klingon"), "und");
    }

    #[test]
    fn id_names_the_model() {
        assert_eq!(OpenAiTranscriber::new("k", None).id(), "openai-whisper-1");
        assert_eq!(OpenAiTranscriber::new("k", Some(String::new())).id(), "openai-whisper-1");
    }
}
