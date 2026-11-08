//! One text-model interface for editing decisions, with Claude, OpenAI and Gemini behind it.
//!
//! Requests are single, non-streaming calls with a system prompt and one user message; answers
//! come back as text (ask for JSON in the prompt when a structure is needed).

use serde_json::{Value, json};

use crate::keys::{Keys, Provider};
use crate::{AiError, http};

/// Defaults are each provider.s cheapest model (the user.s choice: keep token spend low); any of
/// them can be changed in the keys file.
pub const CLAUDE_MODEL: &str = "claude-haiku-5-5";
pub const OPENAI_MODEL: &str = "gpt-5-mini";
pub const GEMINI_MODEL: &str = "gemini-3.5-flash-lite";

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2026-10-21";
const OPENAI_URL: &str = "https://api.openai.com/v1/chat/completions";

/// How hard the model should think.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effort {
    Low,
    Medium,
    High,
}

impl Effort {
    fn as_str(self) -> &'static str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub system: String,
    pub user: String,
    pub max_tokens: u32,
    pub effort: Effort,
}

impl Request {
    pub fn new(system: impl Into<String>, user: impl Into<String>) -> Self {
        Self { system: system.into(), user: user.into(), max_tokens: 16_000, effort: Effort::Low }
    }
}

/// A text model.
pub trait Llm: Send + Sync {
    /// `"claude:claude-opus-5-5"` and the like.
    fn id(&self) -> String;
    fn complete(&self, req: &Request) -> Result<String, AiError>;
}

/// The text model the keys pick (`keys.provider`), or an error naming the missing key.
pub fn from_keys(k: &Keys) -> Result<Box<dyn Llm>, AiError> {
    Ok(match k.provider {
        Provider::Claude => Box::new(Claude {
            key: k.anthropic.clone().ok_or(AiError::MissingKey("Anthropic", "ANTHROPIC_API_KEY"))?,
            model: k.claude_model.clone().unwrap_or_else(|| CLAUDE_MODEL.into()),
        }),
        Provider::OpenAi => Box::new(OpenAi {
            key: k.openai.clone().ok_or(AiError::MissingKey("OpenAI", "OPENAI_API_KEY"))?,
            model: k.openai_model.clone().unwrap_or_else(|| OPENAI_MODEL.into()),
        }),
        Provider::Gemini => Box::new(Gemini {
            key: k.gemini.clone().ok_or(AiError::MissingKey("Gemini", "GEMINI_API_KEY"))?,
            model: k.gemini_model.clone().unwrap_or_else(|| GEMINI_MODEL.into()),
        }),
    })
}

// ---------------------------------------------------------------- Claude

pub struct Claude {
    pub key: String,
    pub model: String,
}

impl Claude {
    /// Server-side refusal fallbacks exist for Opus and Sonnet, not Haiku.
    fn fallbacks(&self) -> bool {
        !self.model.contains("haiku")
    }

    fn body(&self, req: &Request) -> Value {
        let mut b = json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": [{"role": "user", "content": req.user}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": req.effort.as_str()},
        });
        if self.fallbacks() {
            // a declined request is retried on a suitable model instead of failing
            b["fallbacks"] = json!("default");
        }
        b
    }
}

/// The text of a Messages API response; a refusal is an error.
fn claude_text(body: &str) -> Result<String, AiError> {
    let v: Value = serde_json::from_str(body).map_err(|e| AiError::BadResponse("Claude", e.to_string()))?;
    if v.get("stop_reason").and_then(Value::as_str) == Some("refusal") {
        let why = v.pointer("/stop_details/explanation").and_then(Value::as_str).unwrap_or("no reason given");
        return Err(AiError::Refused("Claude", why.to_string()));
    }
    let content = v.get("content").and_then(Value::as_array).ok_or_else(|| AiError::BadResponse("Claude", "no content".into()))?;
    let text: String = content.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("text")).filter_map(|b| b.get("text").and_then(Value::as_str)).collect();
    Ok(text)
}

impl Llm for Claude {
    fn id(&self) -> String {
        format!("claude:{}", self.model)
    }
    fn complete(&self, req: &Request) -> Result<String, AiError> {
        let mut headers = vec![("x-api-key", self.key.as_str()), ("anthropic-version", ANTHROPIC_VERSION)];
        if self.fallbacks() {
            headers.push(("anthropic-beta", "server-side-fallback-2026-07-01"));
        }
        let reply = http::post_json("Claude", ANTHROPIC_URL, &headers, &self.body(req))?;
        claude_text(&http::check("Claude", reply)?)
    }
}

// ---------------------------------------------------------------- OpenAI

pub struct OpenAi {
    pub key: String,
    pub model: String,
}

fn openai_text(body: &str) -> Result<String, AiError> {
    let v: Value = serde_json::from_str(body).map_err(|e| AiError::BadResponse("OpenAI", e.to_string()))?;
    if let Some(r) = v.pointer("/choices/0/message/refusal").and_then(Value::as_str) {
        return Err(AiError::Refused("OpenAI", r.to_string()));
    }
    v.pointer("/choices/0/message/content").and_then(Value::as_str).map(str::to_string).ok_or_else(|| AiError::BadResponse("OpenAI", "no message content".into()))
}

impl Llm for OpenAi {
    fn id(&self) -> String {
        format!("openai:{}", self.model)
    }
    fn complete(&self, req: &Request) -> Result<String, AiError> {
        let auth = format!("Bearer {}", self.key);
        let body = json!({
            "model": self.model,
            "max_completion_tokens": req.max_tokens,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
        });
        let reply = http::post_json("OpenAI", OPENAI_URL, &[("authorization", &auth)], &body)?;
        openai_text(&http::check("OpenAI", reply)?)
    }
}

// ---------------------------------------------------------------- Gemini

pub struct Gemini {
    pub key: String,
    pub model: String,
}

fn gemini_text(body: &str) -> Result<String, AiError> {
    let v: Value = serde_json::from_str(body).map_err(|e| AiError::BadResponse("Gemini", e.to_string()))?;
    if let Some(r) = v.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Err(AiError::Refused("Gemini", r.to_string()));
    }
    let parts = v.pointer("/candidates/0/content/parts").and_then(Value::as_array).ok_or_else(|| AiError::BadResponse("Gemini", "no candidates".into()))?;
    Ok(parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect())
}

impl Llm for Gemini {
    fn id(&self) -> String {
        format!("gemini:{}", self.model)
    }
    fn complete(&self, req: &Request) -> Result<String, AiError> {
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent", self.model);
        let body = json!({
            "systemInstruction": {"parts": [{"text": req.system}]},
            "contents": [{"role": "user", "parts": [{"text": req.user}]}],
            "generationConfig": {"maxOutputTokens": req.max_tokens},
        });
        let reply = http::post_json("Gemini", &url, &[("x-goog-api-key", &self.key)], &body)?;
        gemini_text(&http::check("Gemini", reply)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_request_defaults_to_haiku_at_low_effort() {
        let c = Claude { key: "k".into(), model: CLAUDE_MODEL.into() };
        let b = c.body(&Request::new("sys", "hi"));
        assert_eq!(b["model"], "claude-haiku-5-5");
        assert_eq!(b["thinking"]["type"], "adaptive");
        assert_eq!(b["output_config"]["effort"], "low");
        assert!(b.get("fallbacks").is_none(), "Haiku has no server-side fallback");
        assert_eq!(b["messages"][0]["content"], "hi");
        let opus = Claude { key: "k".into(), model: "claude-opus-5-5".into() };
        assert_eq!(opus.body(&Request::new("s", "u"))["fallbacks"], "default");
    }

    #[test]
    fn claude_text_joins_text_blocks_and_skips_thinking() {
        let body = r#"{"content":[{"type":"thinking","thinking":""},{"type":"text","text":"اقص "},{"type":"text","text":"هنا"}],"stop_reason":"end_turn"}"#;
        assert_eq!(claude_text(body).unwrap(), "اقص هنا");
        let refused = r#"{"content":[],"stop_reason":"refusal","stop_details":{"type":"refusal","explanation":"nope"}}"#;
        assert_eq!(claude_text(refused).unwrap_err(), AiError::Refused("Claude", "nope".into()));
    }

    #[test]
    fn openai_and_gemini_text() {
        assert_eq!(openai_text(r#"{"choices":[{"message":{"content":"ok"}}]}"#).unwrap(), "ok");
        assert!(matches!(openai_text(r#"{"choices":[{"message":{"refusal":"no"}}]}"#), Err(AiError::Refused(..))));
        assert_eq!(gemini_text(r#"{"candidates":[{"content":{"parts":[{"text":"a"},{"text":"b"}]}}]}"#).unwrap(), "ab");
        assert!(matches!(gemini_text(r#"{"promptFeedback":{"blockReason":"SAFETY"}}"#), Err(AiError::Refused(..))));
    }

    #[test]
    fn missing_key_is_named() {
        let k = Keys::default();
        let Err(e) = from_keys(&k) else { panic!("expected an error") };
        assert_eq!(e, AiError::MissingKey("Anthropic", "ANTHROPIC_API_KEY"));
    }
}
