//! API keys and model choices.
//!
//! Read from `<data dir>/montaj/keys.json`, then overridden by environment variables
//! (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). The file is created with
//! owner-only permissions (0600) on Unix. Keys are never printed: `Debug` hides them.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Which text model makes editing decisions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Claude,
    OpenAi,
    Gemini,
}

#[derive(Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Keys {
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub gemini: Option<String>,
    /// The text model used for decisions.
    pub provider: Provider,
    /// Model overrides; `None` = the built-in default for that provider.
    pub claude_model: Option<String>,
    pub openai_model: Option<String>,
    pub gemini_model: Option<String>,
    /// Speech-to-text model at OpenAI (default `whisper-1`, which returns word times).
    pub transcribe_model: Option<String>,
}

impl std::fmt::Debug for Keys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let set = |k: &Option<String>| if k.as_deref().is_some_and(|s| !s.is_empty()) { "set" } else { "-" };
        f.debug_struct("Keys")
            .field("anthropic", &set(&self.anthropic))
            .field("openai", &set(&self.openai))
            .field("gemini", &set(&self.gemini))
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

/// `<data dir>/montaj/keys.json`.
pub fn path(data_dir: &Path) -> PathBuf {
    data_dir.join("montaj").join("keys.json")
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

fn clean(k: Option<String>) -> Option<String> {
    k.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

impl Keys {
    /// Load the keys file (missing or unreadable = empty) and apply the environment on top.
    pub fn load(data_dir: Option<&Path>) -> Self {
        let mut k: Keys = data_dir
            .and_then(|d| std::fs::read_to_string(path(d)).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        k.anthropic = env("ANTHROPIC_API_KEY").or(clean(k.anthropic));
        k.openai = env("OPENAI_API_KEY").or(clean(k.openai));
        k.gemini = env("GEMINI_API_KEY").or_else(|| env("GOOGLE_API_KEY")).or(clean(k.gemini));
        k
    }

    /// Write the keys file with owner-only permissions.
    pub fn save(&self, data_dir: &Path) -> std::io::Result<()> {
        let p = path(data_dir);
        if let Some(dir) = p.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        let tmp = p.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::rename(&tmp, &p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_never_shows_a_key() {
        let k = Keys { anthropic: Some("sk-ant-secret".into()), ..Default::default() };
        let s = format!("{k:?}");
        assert!(!s.contains("secret"), "{s}");
        assert!(s.contains("anthropic: \"set\""));
    }

    #[test]
    fn save_and_load_round_trip() {
        let d = tempfile::tempdir().unwrap();
        let k = Keys { gemini: Some(" g-key ".into()), provider: Provider::Gemini, ..Default::default() };
        k.save(d.path()).unwrap();
        let back = Keys::load(Some(d.path()));
        if std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err() {
            assert_eq!(back.gemini.as_deref(), Some("g-key"));
        }
        assert_eq!(back.provider, Provider::Gemini);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path(d.path())).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }
}
