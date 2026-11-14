//! Montaj inside FilmCraft.
//!
//! [`install`] (call it first thing in `main`) registers Montaj's commands and its per-frame
//! poll with the engine:
//!
//! - `montaj.autocut` — **Sequence ▸ Montaj AI ▸ Auto Cut (AI): Suggest Cuts**. Runs in the
//!   background as a job (progress bar, step and cancel button in the status bar): mixes the
//!   active sequence's audio, transcribes it (Gemini), finds retakes and asides (the text
//!   model), then puts one coloured duration marker per suggested cut on the timeline. Nothing is
//!   cut: the editor reviews the markers (the Markers panel is the list) and deletes the ones to
//!   keep. With `{"wait": true}` (CLI, MCP) it runs to the end before returning.
//! - `montaj.autocut.applyMarked` — **Apply Marked Cuts**: ripple-delete what the remaining
//!   suggestion markers cover, in ONE undo step, and log each suggestion as taken or refused.
//! - `montaj.autocut.clearMarks` — **Clear Suggested Cuts**.
//! - `montaj.autocut.analyze` / `montaj.autocut.apply`: analysis only / apply every suggestion
//!   (CLI, MCP).
//! - `montaj.marks.add` — put suggested-cut markers for given time ranges (e.g. the silences LiteRec
//!   finds on the mic track); Apply Marked Cuts then cuts them like any other suggestion.
//!
//! Paid analyses are cached on disk by an audio fingerprint, so the same footage is never paid
//! for twice. Every applied suggestion is logged (taken / refused) for learning the editor's
//! taste.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

pub mod link;

use filmcraft_engine::commands::NamedKeys;
use filmcraft_engine::export::{Progress, Report};
use filmcraft_engine::{CommandSpec, EngineError, Job, Result, Session};
use filmcraft_project::{ItemId, Label, Marker, MarkerId, MarkerKind, Project, Sequence};
use filmcraft_render::SourceProvider;
use filmcraft_time::{Tick, TimeRange};
use montaj_ai::autocut::Analysis;
use montaj_ai::keys::Keys;
use serde_json::{Value, json};

/// Register Montaj's commands and poll. Call before anything uses the command registry.
pub fn install() -> bool {
    let a = filmcraft_engine::commands::register_extra(specs);
    let b = filmcraft_engine::commands::register_poll(poll);
    a && b
}

/// The cloud transcriber to put in `Session::transcriber` (OpenAI or Gemini, by the keys set).
pub fn transcriber(data_dir: Option<&std::path::Path>) -> Option<Arc<dyn filmcraft_speech::Transcriber>> {
    montaj_ai::transcribe::installed(data_dir)
}

const MENU: &[&str] = &["Sequence", "Montaj AI"];

fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "montaj.autocut",
            label: "Auto Cut (AI): Suggest Cuts",
            menu: MENU,
            shortcut: None,
            params: r#"{"wait":bool?,"refresh":bool?}"#,
            enabled: can_start,
            run: autocut,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "montaj.autocut.applyMarked",
            label: "Apply Marked Cuts",
            menu: MENU,
            shortcut: None,
            params: "{}",
            enabled: has_marks,
            run: apply_marked,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "montaj.autocut.clearMarks",
            label: "Clear Suggested Cuts",
            menu: MENU,
            shortcut: None,
            params: "{}",
            enabled: has_marks,
            run: clear_marks,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "montaj.autocut.apply",
            label: "Auto Cut: Apply All Suggestions",
            menu: &[],
            shortcut: None,
            params: r#"{"skip":[int]?}"#,
            enabled: has_analysis,
            run: apply,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "montaj.marks.add",
            label: "Add Suggested Cuts",
            menu: &[],
            shortcut: None,
            params: r#"{"cuts":[{"start":seconds,"end":seconds,"kind":"silence"|"retake"|…?,"reason":str?}],"replace":bool?}"#,
            enabled: has_sequence,
            run: add_marks,
            journal: true,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "montaj.openLiteRec",
            label: "Open LiteRec",
            menu: MENU,
            shortcut: None,
            params: "{}",
            enabled: always,
            run: open_literec,
            journal: false,
            named: NamedKeys::NONE,
        },
        CommandSpec {
            id: "montaj.autocut.analyze",
            label: "Auto Cut: Analyze",
            menu: &[],
            shortcut: None,
            params: r#"{"refresh":bool?}"#,
            enabled: can_start,
            run: analyze,
            journal: false,
            named: NamedKeys::NONE,
        },
    ]
}

// ------------------------------------------------------------------ state

/// Analyses kept per sequence for this run of the app (for Apply Again).
fn cache() -> &'static Mutex<HashMap<ItemId, Analysis>> {
    static C: std::sync::OnceLock<Mutex<HashMap<ItemId, Analysis>>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The background auto-cut, if one is running.
struct Running {
    job: u64,
    sequence: ItemId,
    revision: u64,
    progress: Arc<Progress>,
    result: Arc<Mutex<Option<std::result::Result<Analysis, String>>>>,
    job_result: Arc<Mutex<Option<std::result::Result<Report, String>>>>,
}

fn running() -> &'static Mutex<Option<Running>> {
    static R: std::sync::OnceLock<Mutex<Option<Running>>> = std::sync::OnceLock::new();
    R.get_or_init(|| Mutex::new(None))
}

fn err(msg: impl std::fmt::Display) -> EngineError {
    EngineError::Other(msg.to_string())
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

/// LiteRec (the recorder that sends videos here): brings it up, or starts it.
fn open_literec(_: &mut Session, _: &Value) -> Result<Value> {
    for id in ["com.el3rrab.literec.rs", "com.el3rrab.literec"] {
        if std::process::Command::new("/usr/bin/open").args(["-b", id]).status().is_ok_and(|s| s.success()) {
            return Ok(serde_json::json!({"opened": id}));
        }
    }
    Err(EngineError::Other("LiteRec isn't installed".into()))
}

fn has_sequence(s: &Session) -> std::result::Result<(), String> {
    match s.active_sequence() {
        Some(q) if q.duration() > Tick::ZERO => Ok(()),
        Some(_) => Err("the sequence is empty".into()),
        None => Err("open a sequence first".into()),
    }
}

fn can_start(s: &Session) -> std::result::Result<(), String> {
    has_sequence(s)?;
    if running().lock().map(|r| r.is_some()).unwrap_or(false) { Err("Auto Cut is already running".into()) } else { Ok(()) }
}

fn has_analysis(s: &Session) -> std::result::Result<(), String> {
    has_sequence(s)?;
    let id = s.state.active_sequence.ok_or("open a sequence first")?;
    let ok = cache().lock().map(|c| c.contains_key(&id)).unwrap_or(false);
    if ok { Ok(()) } else { Err("run Auto Cut (AI) first".into()) }
}

fn data_dir() -> Option<std::path::PathBuf> {
    filmcraft_engine::autosave::default_data_dir()
}

/// A message in the app's status bar (and the Events panel for errors).
fn toast(s: &mut Session, message: String, error: bool) {
    s.events.push(filmcraft_engine::Event::Toast { message, error });
}

// ------------------------------------------------------------------ work (any thread)

/// The sequence's mix as 16 kHz mono, mixed in 10-second blocks; `progress(fraction)` per block.
fn mix_16k(project: &Project, seq: &Sequence, provider: &dyn SourceProvider, progress: &mut dyn FnMut(f32) -> bool) -> std::result::Result<Vec<f32>, String> {
    let rate = i64::from(seq.settings.sample_rate.max(1));
    let total = (seq.duration().seconds() * rate as f64).ceil() as i64;
    if total <= 0 {
        return Err("the sequence is empty".into());
    }
    let step = rate as f64 / 16_000.0;
    let mut out = Vec::with_capacity((total as f64 / step) as usize + 1);
    let block = 10 * rate;
    let mut start = 0i64;
    let mut next = 0.0f64; // next output sample position, in input samples
    while start < total {
        if !progress(start as f32 / total as f32) {
            return Err("stopped".into());
        }
        let n = block.min(total - start) as usize;
        let buf = filmcraft_render::audio::mix_sequence(project, seq, start, n, provider);
        let chans = buf.channels.len().max(1) as f32;
        let mono: Vec<f32> = (0..n).map(|i| buf.channels.iter().map(|c| c.get(i).copied().unwrap_or(0.0)).sum::<f32>() / chans).collect();
        // box-average down to 16 kHz (a light low-pass for speech)
        while next < (start + n as i64) as f64 {
            let a = ((next - start as f64).max(0.0) as usize).min(n - 1);
            let b = ((next + step - start as f64) as usize).clamp(a + 1, n);
            out.push(mono[a..b].iter().sum::<f32>() / (b - a) as f32);
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

/// The analysis of `audio`: from the disk cache when this audio was analysed before (free),
/// else paid (Gemini + the text model) and cached.
fn analysis_for(audio: &[f32], refresh: bool, progress: &mut dyn FnMut(f32, &str) -> bool) -> std::result::Result<Analysis, String> {
    let disk = data_dir().map(|d| d.join("montaj").join("analysis").join(format!("v2-{:016x}.json", fingerprint(audio))));
    if !refresh && let Some(a) = disk.as_ref().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|t| serde_json::from_str::<Analysis>(&t).ok()) {
        // re-plan with the current rules: older analyses get every improvement for free
        let usage = montaj_ai::gemini_stt::Usage { input: a.tokens_in, output: a.tokens_out, thinking: 0 };
        return Ok(montaj_ai::autocut::plan(audio, a.segments, a.drops, usage));
    }
    let keys = Keys::load(data_dir().as_deref());
    let a = montaj_ai::autocut::analyze(audio, &keys, progress).map_err(|e| e.to_string())?;
    if let Some(p) = &disk
        && let Some(dir) = p.parent()
        && std::fs::create_dir_all(dir).is_ok()
        && let Ok(t) = serde_json::to_string(&a)
    {
        let _ = std::fs::write(p, t);
    }
    Ok(a)
}

// ------------------------------------------------------------------ commands (main thread)

fn summary(a: &Analysis) -> String {
    let cut: f64 = a.cuts.iter().map(|c| c.end - c.start).sum();
    let talk = a.cuts.iter().filter(|c| c.kind != "pause").count();
    format!("{} cuts ({talk} retakes/asides/stutters, {} pauses), {:.0} s of {:.0} s", a.cuts.len(), a.cuts.len() - talk, cut, a.duration)
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

/// Ripple-delete the cuts of `a` (but `skip`) from the active sequence in one undo step.
fn apply_analysis(s: &mut Session, a: &Analysis, skip: &[usize]) -> Result<(usize, f64)> {
    let ranges: Vec<TimeRange> = a
        .cuts
        .iter()
        .filter(|c| !skip.contains(&c.id))
        .map(|c| TimeRange::from_bounds(Tick::from_seconds_f64(c.start), Tick::from_seconds_f64(c.end)))
        .filter(|r| r.duration > Tick::ZERO)
        .collect();
    let n = ranges.len();
    let removed = s.edit_sequence("Auto Cut", |q, ctx, _| Ok(filmcraft_edit::transcript::ripple_delete_ranges(q, ranges, ctx)))?;
    log_decisions(a, skip);
    Ok((n, removed.seconds()))
}

/// Synchronous analysis of the active sequence (CLI, MCP, `wait`).
fn analyze_now(s: &mut Session, refresh: bool) -> Result<Analysis> {
    let id = s.state.active_sequence.ok_or_else(|| err("open a sequence first"))?;
    let seq = s.active_sequence().cloned().ok_or_else(|| err("open a sequence first"))?;
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let audio = mix_16k(&s.project, &seq, &provider, &mut |_| true).map_err(err)?;
    let a = match analysis_for(&audio, refresh, &mut |_, _| true) {
        Ok(a) => a,
        Err(e) => {
            toast(s, format!("Montaj Auto Cut failed: {e}"), true);
            return Err(err(e));
        }
    };
    if let Ok(mut c) = cache().lock() {
        c.insert(id, a.clone());
    }
    Ok(a)
}

fn analyze(s: &mut Session, p: &Value) -> Result<Value> {
    let a = analyze_now(s, p.get("refresh").and_then(Value::as_bool).unwrap_or(false))?;
    toast(s, format!("Montaj Auto Cut: {} suggested", summary(&a)), false);
    Ok(suggestions(&a))
}

fn autocut(s: &mut Session, p: &Value) -> Result<Value> {
    let refresh = p.get("refresh").and_then(Value::as_bool).unwrap_or(false);
    if p.get("wait").and_then(Value::as_bool).unwrap_or(false) {
        let a = analyze_now(s, refresh)?;
        let n = mark_analysis(s, &a)?;
        toast(s, review_message(&a), false);
        return Ok(json!({"marked": n, "analysis": suggestions(&a)}));
    }
    start_background(s, refresh)
}

// ------------------------------------------------------------------ review on the timeline

/// Montaj's suggested-cut markers carry this author and a `[montaj #id]` note.
const AUTHOR: &str = "Montaj";

fn is_mark(m: &Marker) -> bool {
    m.author == AUTHOR && m.comment.starts_with("[montaj #")
}

/// The cut id in a marker's note (`[montaj #12] …`).
fn mark_cut_id(m: &Marker) -> Option<usize> {
    m.comment.strip_prefix("[montaj #")?.split(']').next()?.parse().ok()
}

fn kind_label(kind: &str) -> (&'static str, Label) {
    match kind {
        "retake" => ("✂ إعادة", Label::Rose),
        "aside" => ("✂ كلام جانبي", Label::Purple),
        "stutter" => ("✂ تلعثم", Label::Mango),
        "filler" => ("✂ حشو", Label::Mango),
        "false start" => ("✂ بداية مقطوعة", Label::Mango),
        "noise" => ("✂ صوت", Label::Tan),
        _ => ("✂ سكتة", Label::Yellow),
    }
}

fn review_message(a: &Analysis) -> String {
    format!(
        "Montaj: {} marked on the timeline. Review them in the Markers panel, delete the markers of cuts you want to keep, then Sequence ▸ Montaj AI ▸ Apply Marked Cuts",
        summary(a)
    )
}

/// Put one duration marker per suggested cut on the active sequence (replacing earlier ones).
fn mark_analysis(s: &mut Session, a: &Analysis) -> Result<usize> {
    let cuts = a.cuts.clone();
    s.edit_sequence("Auto Cut: Suggest Cuts", move |q, ctx, _| {
        q.markers.retain(|m| !is_mark(m));
        for c in &cuts {
            let (name, color) = kind_label(&c.kind);
            let words: String = c.text.chars().take(60).collect();
            q.markers.push(Marker {
                url: String::new(),
                author: AUTHOR.into(),
                custom_color: None,
                id: MarkerId(ctx.alloc()),
                start: Tick::from_seconds_f64(c.start),
                duration: Tick::from_seconds_f64(c.end) - Tick::from_seconds_f64(c.start),
                name: if words.is_empty() { name.to_string() } else { format!("{name}: {words}") },
                comment: format!("[montaj #{}] {}", c.id, c.reason),
                kind: MarkerKind::Comment,
                color,
            });
        }
        q.markers.sort_by_key(|m| m.start);
        Ok(cuts.len())
    })
}

/// Ids of cuts added from outside (LiteRec's silences) start here, clear of Auto Cut's own ids.
const OUTSIDE_IDS: usize = 100_000;

/// `montaj.marks.add`: suggested-cut markers for the given ranges (sequence seconds). With `replace`,
/// earlier suggestion markers go first. Review and apply them like Auto Cut's.
fn add_marks(s: &mut Session, p: &Value) -> Result<Value> {
    let cuts: Vec<(f64, f64, String, String)> = p
        .get("cuts")
        .and_then(Value::as_array)
        .ok_or_else(|| err("cuts: a list of {start, end} in seconds"))?
        .iter()
        .filter_map(|c| {
            let (a, b) = (c.get("start")?.as_f64()?, c.get("end")?.as_f64()?);
            let kind = c.get("kind").and_then(Value::as_str).unwrap_or("silence").to_string();
            let reason = c.get("reason").and_then(Value::as_str).unwrap_or("").to_string();
            (b > a && a >= 0.0).then_some((a, b, kind, reason))
        })
        .collect();
    let replace = p.get("replace").and_then(Value::as_bool).unwrap_or(false);
    let n = cuts.len();
    s.edit_sequence("Mark Silences", move |q, ctx, _| {
        if replace {
            q.markers.retain(|m| !is_mark(m));
        }
        for (i, (a, b, kind, reason)) in cuts.iter().enumerate() {
            let (name, color) = kind_label(kind);
            q.markers.push(Marker {
                url: String::new(),
                author: AUTHOR.into(),
                custom_color: None,
                id: MarkerId(ctx.alloc()),
                start: Tick::from_seconds_f64(*a),
                duration: Tick::from_seconds_f64(*b) - Tick::from_seconds_f64(*a),
                name: name.to_string(),
                comment: format!("[montaj #{}] {reason}", OUTSIDE_IDS + i),
                kind: MarkerKind::Comment,
                color,
            });
        }
        q.markers.sort_by_key(|m| m.start);
        Ok(())
    })?;
    Ok(json!({"marked": n}))
}

fn has_marks(s: &Session) -> std::result::Result<(), String> {
    has_sequence(s)?;
    let any = s.active_sequence().is_some_and(|q| q.markers.iter().any(is_mark));
    if any { Ok(()) } else { Err("no suggested cuts on this sequence (run Auto Cut (AI): Suggest Cuts)".into()) }
}

/// Cut what the marked suggestions cover, in one undo step; the editor deleted the markers of the
/// cuts to keep. Logs taken / refused against the last analysis of this sequence.
fn apply_marked(s: &mut Session, _: &Value) -> Result<Value> {
    let id = s.state.active_sequence.ok_or_else(|| err("open a sequence first"))?;
    let marks: Vec<(Option<usize>, TimeRange)> = s
        .active_sequence()
        .map(|q| q.markers.iter().filter(|m| is_mark(m)).map(|m| (mark_cut_id(m), TimeRange::from_bounds(m.start, m.start + m.duration))).collect())
        .unwrap_or_default();
    let kept_ids: Vec<usize> = marks.iter().filter_map(|(i, _)| *i).collect();
    let ranges: Vec<TimeRange> = marks.into_iter().map(|(_, r)| r).filter(|r| r.duration > Tick::ZERO).collect();
    let n = ranges.len();
    let removed = s.edit_sequence("Apply Marked Cuts", move |q, ctx, _| {
        q.markers.retain(|m| !is_mark(m));
        Ok(filmcraft_edit::transcript::ripple_delete_ranges(q, ranges, ctx))
    })?;
    if let Some(a) = cache().lock().ok().and_then(|mut c| c.remove(&id)) {
        let refused: Vec<usize> = a.cuts.iter().map(|c| c.id).filter(|i| !kept_ids.contains(i)).collect();
        log_decisions(&a, &refused);
    }
    toast(s, format!("Montaj: removed {n} cuts ({:.0} s). Cmd+Z undoes all of it.", removed.seconds()), false);
    Ok(json!({"applied": n, "removedSeconds": removed.seconds()}))
}

fn clear_marks(s: &mut Session, _: &Value) -> Result<Value> {
    let n = s.edit_sequence("Clear Suggested Cuts", |q, _, _| {
        let before = q.markers.len();
        q.markers.retain(|m| !is_mark(m));
        Ok(before - q.markers.len())
    })?;
    Ok(json!({"cleared": n}))
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    let id = s.state.active_sequence.ok_or_else(|| err("open a sequence first"))?;
    let a = cache().lock().ok().and_then(|c| c.get(&id).cloned()).ok_or_else(|| err("run Auto Cut (AI) first"))?;
    let skip: Vec<usize> = p.get("skip").and_then(Value::as_array).map(|v| v.iter().filter_map(Value::as_u64).map(|n| n as usize).collect()).unwrap_or_default();
    let (n, secs) = apply_analysis(s, &a, &skip)?;
    if let Ok(mut c) = cache().lock() {
        c.remove(&id);
    }
    toast(s, format!("Montaj Auto Cut: removed {n} cuts ({secs:.0} s). Cmd+Z undoes all of it."), false);
    Ok(json!({"applied": n, "skipped": skip.len(), "removedSeconds": secs}))
}

// ------------------------------------------------------------------ background job

const TOTAL: u64 = 1000;

fn set(progress: &Progress, fraction: f32, status: &str) {
    progress.done.store((fraction.clamp(0.0, 1.0) * TOTAL as f32) as u64, Ordering::Relaxed);
    if let Ok(mut st) = progress.status.lock() {
        *st = format!("Montaj Auto Cut: {status}");
    }
}

fn start_background(s: &mut Session, refresh: bool) -> Result<Value> {
    let sequence = s.state.active_sequence.ok_or_else(|| err("open a sequence first"))?;
    let seq = s.active_sequence().cloned().ok_or_else(|| err("open a sequence first"))?;
    let project = s.project.clone();
    let provider = s.media.provider(s.project.clone(), s.services.clone());
    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0) + 1;
    let job = Job { id, label: "Montaj Auto Cut".into(), progress: Default::default(), result: Default::default() };
    let progress = job.progress.clone();
    progress.total.store(TOTAL, Ordering::Relaxed);
    set(&progress, 0.0, "mixing the audio");
    let result: Arc<Mutex<Option<std::result::Result<Analysis, String>>>> = Arc::default();
    {
        let (progress, result) = (progress.clone(), result.clone());
        std::thread::spawn(move || {
            let cancelled = || progress.cancel.load(Ordering::Relaxed);
            let out = mix_16k(&project, &seq, &provider, &mut |f| {
                set(&progress, 0.15 * f, "mixing the audio");
                !cancelled()
            })
            .and_then(|audio| {
                analysis_for(&audio, refresh, &mut |f, step| {
                    set(&progress, f, step);
                    !cancelled()
                })
            });
            if let Ok(mut r) = result.lock() {
                *r = Some(out);
            }
        });
    }
    if let Ok(mut r) = running().lock() {
        *r = Some(Running { job: id, sequence, revision: s.revision, progress, result, job_result: job.result.clone() });
    }
    s.jobs.push(job);
    toast(s, "Montaj Auto Cut started — progress at the bottom right".into(), false);
    Ok(json!({"job": id}))
}

/// Per frame: when the background analysis is done, apply it on the main thread.
fn poll(s: &mut Session) {
    let finished = {
        let Ok(mut r) = running().lock() else { return };
        let done = r.as_ref().and_then(|run| run.result.lock().ok().and_then(|mut x| x.take()));
        match done {
            Some(out) => r.take().map(|run| (run, out)),
            None => None,
        }
    };
    let Some((run, out)) = finished else { return };
    let finish = |job: std::result::Result<Report, String>| {
        run.progress.done.store(TOTAL, Ordering::Relaxed);
        run.progress.finished.store(true, Ordering::Relaxed);
        if let Ok(mut r) = run.job_result.lock() {
            *r = Some(job);
        }
    };
    let a = match out {
        Ok(a) => a,
        Err(e) => {
            let msg = if run.progress.cancel.load(Ordering::Relaxed) { "Montaj Auto Cut cancelled".to_string() } else { format!("Montaj Auto Cut failed: {e}") };
            toast(s, msg.clone(), true);
            finish(Err(msg));
            return;
        }
    };
    if let Ok(mut c) = cache().lock() {
        c.insert(run.sequence, a.clone());
    }
    let report = |secs: f64| Report { path: String::new(), frames: a.cuts.len() as u64, seconds: secs, bytes: 0, render_fps: 0.0, extra_files: Vec::new() };
    // the sequence changed while we were analysing: its times no longer match, let the editor decide
    if s.state.active_sequence != Some(run.sequence) || s.revision != run.revision {
        toast(s, format!("Montaj Auto Cut: {} found, but the sequence changed meanwhile, so nothing was marked — run Auto Cut (AI) again", summary(&a)), false);
        finish(Ok(report(0.0)));
        return;
    }
    // suggestions, not cuts: the editor reviews the markers and applies what stays
    match mark_analysis(s, &a) {
        Ok(_) => {
            toast(s, review_message(&a), false);
            finish(Ok(report(0.0)));
        }
        Err(e) => {
            toast(s, format!("Montaj Auto Cut could not mark the cuts: {e}"), true);
            finish(Err(e.to_string()));
        }
    }
    let _ = run.job;
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
