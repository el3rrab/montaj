//! Montaj inside FilmCraft.
//!
//! [`install`] (call it first thing in `main`) registers Montaj's commands with the engine:
//!
//! - `montaj.autocut.analyze`: mix the active sequence's audio, transcribe it (Gemini), find
//!   retakes and asides (the text model), and return the suggested cuts — each with an id, its
//!   time, kind, words and reason. The paid result is kept for the sequence, so applying,
//!   re-planning or asking again costs nothing.
//! - `montaj.autocut.apply`: ripple-delete the suggested cuts (all, or all but `skip`) in ONE
//!   undo step, and log which suggestions were taken or refused (the start of learning the
//!   editor's taste).
//!
//! Both appear in Sequence ▸ Montaj AI, the CLI and the MCP server like any FilmCraft command
//! (FilmCraft's menu bar has a fixed set of top-level menus).

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::collections::HashMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use filmcraft_engine::commands::NamedKeys;
use filmcraft_engine::{CommandSpec, EngineError, Result, Session};
use filmcraft_project::ItemId;
use filmcraft_time::{Tick, TimeRange};
use montaj_ai::autocut::Analysis;
use montaj_ai::keys::Keys;
use serde_json::{Value, json};

/// Register Montaj's commands. Call before anything uses the command registry.
pub fn install() -> bool {
    filmcraft_engine::commands::register_extra(specs)
}

/// The cloud transcriber to put in `Session::transcriber` (OpenAI or Gemini, by the keys set).
pub fn transcriber(data_dir: Option<&std::path::Path>) -> Option<Arc<dyn filmcraft_speech::Transcriber>> {
    montaj_ai::transcribe::installed(data_dir)
}

fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "montaj.autocut.analyze",
            label: "Auto Cut: Analyze (AI)",
            menu: &["Sequence", "Montaj AI"],
            shortcut: None,
            params: r#"{"refresh":bool?}"#,
            enabled: has_sequence,
            run: analyze,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "montaj.autocut.apply",
            label: "Auto Cut: Apply",
            menu: &["Sequence", "Montaj AI"],
            shortcut: None,
            params: r#"{"skip":[int]?}"#,
            enabled: has_analysis,
            run: apply,
            journal: true,
            named: NamedKeys::NONE,
        },
    ]
}

/// Paid analyses kept per sequence for this run of the app.
fn cache() -> &'static Mutex<HashMap<ItemId, Analysis>> {
    static C: std::sync::OnceLock<Mutex<HashMap<ItemId, Analysis>>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn err(msg: impl std::fmt::Display) -> EngineError {
    EngineError::Other(msg.to_string())
}

fn has_sequence(s: &Session) -> std::result::Result<(), String> {
    if s.active_sequence().is_some() { Ok(()) } else { Err("open a sequence first".into()) }
}

fn has_analysis(s: &Session) -> std::result::Result<(), String> {
    has_sequence(s)?;
    let id = s.state.active_sequence.ok_or("open a sequence first")?;
    let ok = cache().lock().map(|c| c.contains_key(&id)).unwrap_or(false);
    if ok { Ok(()) } else { Err("run Auto Cut: Analyze first".into()) }
}

/// The active sequence's mix as 16 kHz mono, mixed in 10-second blocks.
fn sequence_audio_16k(s: &Session) -> Result<Vec<f32>> {
    let seq = s.active_sequence().ok_or_else(|| err("open a sequence first"))?;
    let rate = i64::from(seq.settings.sample_rate.max(1));
    let total = (seq.duration().seconds() * rate as f64).ceil() as i64;
    if total <= 0 {
        return Err(err("the sequence is empty"));
    }
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let step = rate as f64 / 16_000.0;
    let mut out = Vec::with_capacity((total as f64 / step) as usize + 1);
    let block = 10 * rate;
    let mut start = 0i64;
    let mut next = 0.0f64; // next output sample position, in input samples
    while start < total {
        let n = block.min(total - start) as usize;
        let buf = filmcraft_render::audio::mix_sequence(&s.project, seq, start, n, &provider);
        let chans = buf.channels.len().max(1) as f32;
        let mono: Vec<f32> = (0..n).map(|i| buf.channels.iter().map(|c| c.get(i).copied().unwrap_or(0.0)).sum::<f32>() / chans).collect();
        // box-average down to 16 kHz (a light low-pass for speech)
        while next < (start + n as i64) as f64 {
            let a = (next - start as f64).max(0.0) as usize;
            let b = ((next + step - start as f64) as usize).clamp(a + 1, n);
            let sum: f32 = mono[a.min(n - 1)..b].iter().sum();
            out.push(sum / (b - a.min(n - 1)).max(1) as f32);
            next += step;
        }
        start += n as i64;
    }
    Ok(out)
}

/// FNV-1a over the audio quantized to 16 bits: the same mix gives the same key.
fn fingerprint(audio: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &s in audio {
        let q = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        for b in q.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h ^ audio.len() as u64
}

fn data_dir() -> Option<std::path::PathBuf> {
    filmcraft_engine::autosave::default_data_dir()
}

fn suggestions(a: &Analysis) -> Value {
    let cut: f64 = a.cuts.iter().map(|c| c.end - c.start).sum();
    json!({
        "duration": a.duration,
        "cutSeconds": cut,
        "keptSeconds": a.duration - cut,
        "tokens": {"in": a.tokens_in, "out": a.tokens_out},
        "cuts": a.cuts,
    })
}

fn analyze(s: &mut Session, p: &Value) -> Result<Value> {
    let id = s.state.active_sequence.ok_or_else(|| err("open a sequence first"))?;
    let refresh = p.get("refresh").and_then(Value::as_bool).unwrap_or(false);
    if !refresh && let Some(a) = cache().lock().ok().and_then(|c| c.get(&id).cloned()) {
        return Ok(suggestions(&a));
    }
    let audio = sequence_audio_16k(s)?;
    // the same audio was analysed before (any project, any run): reuse it, no paid call
    let disk = data_dir().map(|d| d.join("montaj").join("analysis").join(format!("{:016x}.json", fingerprint(&audio))));
    let cached: Option<Analysis> = if refresh { None } else { disk.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str(&t).ok()) };
    let a = match cached {
        Some(a) => a,
        None => {
            let keys = Keys::load(data_dir().as_deref());
            let a = montaj_ai::autocut::analyze(&audio, &keys, &mut |_| true).map_err(err)?;
            if let Some(p) = &disk
                && let Some(dir) = p.parent()
                && std::fs::create_dir_all(dir).is_ok()
                && let Ok(t) = serde_json::to_string(&a)
            {
                let _ = std::fs::write(p, t);
            }
            a
        }
    };
    let out = suggestions(&a);
    if let Ok(mut c) = cache().lock() {
        c.insert(id, a);
    }
    Ok(out)
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    let id = s.state.active_sequence.ok_or_else(|| err("open a sequence first"))?;
    let a = cache().lock().ok().and_then(|c| c.get(&id).cloned()).ok_or_else(|| err("run Auto Cut: Analyze first"))?;
    let skip: Vec<usize> = p.get("skip").and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_u64).map(|n| n as usize).collect()).unwrap_or_default();
    let ranges: Vec<TimeRange> = a
        .cuts
        .iter()
        .filter(|c| !skip.contains(&c.id))
        .map(|c| TimeRange::from_bounds(Tick::from_seconds_f64(c.start), Tick::from_seconds_f64(c.end)))
        .filter(|r| r.duration > Tick::ZERO)
        .collect();
    let n = ranges.len();
    let removed = s.edit_sequence("Auto Cut", |q, ctx, _| Ok(filmcraft_edit::transcript::ripple_delete_ranges(q, ranges, ctx)))?;
    log_decisions(&a, &skip);
    // the analysis no longer matches the edited sequence
    if let Ok(mut c) = cache().lock() {
        c.remove(&id);
    }
    Ok(json!({"applied": n, "skipped": skip.len(), "removedSeconds": removed.seconds()}))
}

/// Append one line per suggestion to `<data dir>/montaj/decisions.jsonl`: what was suggested
/// and whether the editor kept the cut. Montaj learns the editor's taste from this later.
fn log_decisions(a: &Analysis, skip: &[usize]) {
    let Some(dir) = data_dir().map(|d| d.join("montaj")) else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("decisions.jsonl")) else { return };
    for c in &a.cuts {
        let line = json!({"kind": c.kind, "text": c.text, "reason": c.reason, "seconds": c.end - c.start, "accepted": !skip.contains(&c.id)});
        let _ = writeln!(f, "{line}");
    }
}
