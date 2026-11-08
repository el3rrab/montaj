//! `montaj-autocut <video> [--start S] [--dur D] [--out DIR] [--stt-model M] [--llm-model M] [--render]`:
//! run the app's auto-cut (same plan as the editor) on a video or an excerpt and write a
//! report of every cut; `--render` also writes an edited mp4. Needs ffmpeg.
//! `--stt-model` picks the Gemini model that listens, `--llm-model` the Claude model that
//! decides the retakes. Paid results are cached per model in DIR, so re-planning is free.

use std::path::{Path, PathBuf};
use std::process::Command;

use montaj_ai::autocut;
use montaj_ai::gemini_stt::{GeminiTranscriber, Segment, Usage};
use montaj_ai::keys::Keys;
use montaj_ai::llm::{Claude, Llm};
use montaj_ai::retakes::{self, Drop};

fn data_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("FILMCRAFT_DATA_DIR").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(d));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/FilmCraft"))
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

fn ffmpeg(args: &[&str]) {
    let ok = Command::new("ffmpeg").args(["-v", "error", "-y"]).args(args).status().map(|s| s.success()).unwrap_or(false);
    if !ok {
        die("ffmpeg failed");
    }
}

fn read_wav16(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| die(e));
    b.get(44..).unwrap_or(&[]).as_chunks::<2>().0.iter().map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(video) = args.first().cloned() else { die("usage: montaj-autocut <video> [--start S] [--dur D] [--out DIR] [--stt-model M] [--llm-model M] [--render]") };
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let start = opt("--start").unwrap_or_else(|| "0".into());
    let dur = opt("--dur").unwrap_or_else(|| "120".into());
    let out = PathBuf::from(opt("--out").unwrap_or_else(|| "autocut".into()));
    std::fs::create_dir_all(&out).unwrap_or_else(|e| die(e));
    let keys = Keys::load(data_dir().as_deref());
    let stt_model = opt("--stt-model").unwrap_or_else(|| montaj_ai::llm::GEMINI_MODEL.into());
    let llm_model = opt("--llm-model").unwrap_or_else(|| montaj_ai::llm::CLAUDE_MODEL.into());

    // 1. audio (free)
    let wav = out.join("audio.wav");
    if !wav.exists() {
        ffmpeg(&["-ss", &start, "-t", &dur, "-i", &video, "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav.to_str().unwrap_or_default()]);
    }
    let audio = read_wav16(&wav);

    // 2. transcript (paid once per model, cached)
    let piece: usize = opt("--piece").and_then(|p| p.parse().ok()).unwrap_or(montaj_ai::gemini_stt::TIMED_PIECE_SECONDS);
    let seg_cache = out.join(format!("timed-{stt_model}{}.json", if piece == montaj_ai::gemini_stt::TIMED_PIECE_SECONDS { String::new() } else { format!("-p{piece}") }));
    let mut usage = Usage::default();
    let segs: Vec<Segment> = match std::fs::read_to_string(&seg_cache).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        Some(s) => s,
        None => {
            let key = keys.gemini.clone().unwrap_or_else(|| die("no Gemini key (run montaj-keys)"));
            let mut t = GeminiTranscriber::new(key, Some(stt_model.clone()));
            t.piece_seconds = piece;
            let started = std::time::Instant::now();
            let (s, u) = t
                .timed(&audio, &mut |i, n| {
                    eprintln!("{stt_model}: piece {}/{n}", i + 1);
                    true
                })
                .unwrap_or_else(|e| die(e));
            eprintln!("{stt_model}: {} segments in {:.0} s; tokens in {} out {} thinking {}", s.len(), started.elapsed().as_secs_f64(), u.input, u.output, u.thinking);
            usage = u;
            let _ = std::fs::write(&seg_cache, serde_json::to_string_pretty(&s).unwrap_or_default());
            s
        }
    };

    // 3. retakes (paid once per model pair, cached)
    let rt_cache = out.join(format!("retakes-{stt_model}-{llm_model}.json"));
    let drops: Vec<Drop> = match std::fs::read_to_string(&rt_cache).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        Some(d) => d,
        None => {
            let key = keys.anthropic.clone().unwrap_or_else(|| die("no Anthropic key (run montaj-keys)"));
            let llm = Claude { key, model: llm_model.clone() };
            let started = std::time::Instant::now();
            let d = retakes::find(&llm as &dyn Llm, &segs).unwrap_or_else(|e| die(e));
            eprintln!("{}: {} lines marked in {:.0} s", llm.id(), d.len(), started.elapsed().as_secs_f64());
            let _ = std::fs::write(&rt_cache, serde_json::to_string_pretty(&d).unwrap_or_default());
            d
        }
    };

    // 4. the app's plan (free) and a report
    let a = autocut::plan(&audio, segs, drops, usage);
    let mut report = String::new();
    for c in &a.cuts {
        report.push_str(&format!("✂ {:<11} {:8.2}-{:8.2} ({:5.1}s)  {}  ← {}\n", c.kind, c.start, c.end, c.end - c.start, c.text, c.reason));
    }
    let cut: f64 = a.cuts.iter().map(|c| c.end - c.start).sum();
    report.push_str(&format!("\n{stt_model} + {llm_model}: {} cuts, {cut:.0} s of {:.0} s cut\n", a.cuts.len(), a.duration));
    let _ = std::fs::write(out.join(format!("report-{stt_model}-{llm_model}.txt")), &report);
    let _ = std::fs::write(out.join(format!("keep-{stt_model}-{llm_model}.json")), serde_json::to_string(&a.keep).unwrap_or_default());
    let _ = std::fs::write(out.join(format!("analysis-{stt_model}-{llm_model}.json")), serde_json::to_string(&a).unwrap_or_default());
    eprintln!("{}", report.lines().last().unwrap_or_default());

    // 5. render (free, optional)
    if args.iter().any(|a| a == "--render") {
        let mut f = String::new();
        let mut cat = String::new();
        for (i, (s, e)) in a.keep.iter().enumerate() {
            let fade = (e - s - 0.03).max(0.0);
            f.push_str(&format!("[0:v]trim={s:.3}:{e:.3},setpts=PTS-STARTPTS[v{i}];[0:a]atrim={s:.3}:{e:.3},asetpts=PTS-STARTPTS,afade=t=in:d=0.02,afade=t=out:st={fade:.3}:d=0.03[a{i}];"));
            cat.push_str(&format!("[v{i}][a{i}]"));
        }
        f.push_str(&format!("{cat}concat=n={}:v=1:a=1[v][a]", a.keep.len()));
        let filter = out.join("filter.txt");
        let _ = std::fs::write(&filter, f);
        let edited = out.join(format!("edited-{stt_model}-{llm_model}.mp4"));
        ffmpeg(&[
            "-ss", &start, "-t", &dur, "-i", &video,
            "-/filter_complex", filter.to_str().unwrap_or_default(),
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "22", "-preset", "veryfast", "-c:a", "aac", "-b:a", "160k",
            edited.to_str().unwrap_or_default(),
        ]);
        eprintln!("→ {}", edited.display());
    }
}
