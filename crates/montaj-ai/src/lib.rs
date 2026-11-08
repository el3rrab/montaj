//! Montaj's cloud AI layer.
//!
//! - [`keys`]: API keys for Anthropic (Claude), OpenAI and Google (Gemini), from the environment
//!   or `<data dir>/montaj/keys.json`. Keys never go into projects, logs or errors.
//! - [`transcribe`]: [`transcribe::OpenAiTranscriber`], a [`filmcraft_speech::Transcriber`] that
//!   sends speech to OpenAI and returns words with times, so every `transcript.*` command
//!   (remove pauses and fillers, captions, text-based editing) works on it.
//! - [`llm`]: one text-model interface ([`llm::Llm`]) with Claude, OpenAI and Gemini behind it,
//!   for editing decisions (retakes, highlights for shorts, where to place assets).
//!
//! Everything here is Montaj's own code; FilmCraft's crates are used, never changed.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

mod http;
pub mod gemini_stt;
pub mod keys;
pub mod llm;
pub mod retakes;
pub mod transcribe;
pub mod wav;

pub use keys::Keys;

/// Errors from the cloud providers. Messages never contain an API key.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AiError {
    #[error("no {0} API key (set it in Montaj ▸ AI keys or the {1} environment variable)")]
    MissingKey(&'static str, &'static str),
    #[error("network: {0}")]
    Network(String),
    #[error("{provider} returned HTTP {status}: {message}")]
    Http { provider: &'static str, status: u16, message: String },
    #[error("{0} sent a response Montaj could not read: {1}")]
    BadResponse(&'static str, String),
    #[error("{0} declined the request: {1}")]
    Refused(&'static str, String),
    #[error("stopped")]
    Cancelled,
    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for AiError {
    fn from(e: std::io::Error) -> Self {
        AiError::Io(e.to_string())
    }
}
