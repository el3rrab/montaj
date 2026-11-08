//! `montaj-autocut <video> [--start S] [--dur D] [--out DIR]`: try the whole auto-cut on an
//! excerpt — Gemini transcript with disfluency kinds, retakes from the text model, pauses
//! shortened — and render `edited.mp4` beside a `report.txt` of every cut. Needs ffmpeg.
//! Paid results are cached in DIR (`segments.json`, `retakes.json`) and reused on later runs,
//! so tuning the cut costs nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

use montaj_ai::cutplan::{self, CutOptions};
use montaj_ai::gemini_stt::{GeminiTranscriber, Segment};
use montaj_ai::keys::Keys;
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
    let Some(video) = args.first().cloned() else { die("usage: montaj-autocut <video> [--start S] [--dur D] [--out DIR]") };
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let start = opt("--start").unwrap_or_else(|| "0".into());
    let dur = opt("--dur").unwrap_or_else(|| "120".into());
    let out = PathBuf::from(opt("--out").unwrap_or_else(|| "autocut".into()));
    std::fs::create_dir_all(&out).unwrap_or_else(|e| die(e));
    let keys = Keys::load(data_dir().as_deref());

    // 1. audio excerpt (free)
    let wav = out.join("audio.wav");
    if !wav.exists() {
        ffmpeg(&["-ss", &start, "-t", &dur, "-i", &video, "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav.to_str().unwrap_or_default()]);
    }
    let audio = read_wav16(&wav);

    // 2. transcript with disfluency kinds (paid once, cached)
    let seg_cache = out.join("timed.json");
    let segs: Vec<Segment> = match std::fs::read_to_string(&seg_cache).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        Some(s) => s,
        None => {
            let t = GeminiTranscriber::from_keys(&keys).unwrap_or_else(|| die("no Gemini key (run montaj-keys)"));
            let (s, u) = t.timed(&audio, &mut |i, n| { eprintln!("Gemini piece {}/{n}", i + 1); true }).unwrap_or_else(|e| die(e));
            eprintln!("Gemini: {} segments; tokens in {} out {} thinking {}", s.len(), u.input, u.output, u.thinking);
            let _ = std::fs::write(&seg_cache, serde_json::to_string_pretty(&s).unwrap_or_default());
            s
        }
    };

    // 3. retakes (paid once, cached)
    let rt_cache = out.join("timed_retakes.json");
    let drops: Vec<Drop> = match std::fs::read_to_string(&rt_cache).ok().and_then(|s| serde_json::from_str(&s).ok()) {
        Some(d) => d,
        None => {
            let llm = montaj_ai::llm::from_keys(&keys).unwrap_or_else(|e| die(e));
            let d = retakes::find(llm.as_ref(), &segs).unwrap_or_else(|e| die(e));
            eprintln!("{}: {} retake segments", llm.id(), d.len());
            let _ = std::fs::write(&rt_cache, serde_json::to_string_pretty(&d).unwrap_or_default());
            d
        }
    };
    // free local rule on top: an opening said again soon after is an abandoned attempt
    let mut drops = drops;
    let model_lines: Vec<usize> = drops.iter().map(|d| d.line).collect();
    drops.extend(retakes::repeated_openings(&segs, &model_lines, 3, 30.0));
    drops.sort_by_key(|d| d.line);
    // a line that only starts a retake midway keeps its head
    let (segs, drops) = retakes::split_partial_starts(&segs, &drops);
    let drop_lines: Vec<usize> = drops.iter().map(|d| d.line).collect();

    // 4. plan (free) and report
    let keep = cutplan::keep_ranges(&audio, &segs, &drop_lines, &CutOptions::default());
    let mut report = String::new();
    for (i, s) in segs.iter().enumerate() {
        let label = drops.iter().find(|d| d.line == i).map(|d| d.kind.as_str()).or_else(|| cutplan::cut_reason(i, s, &drop_lines));
        let mark = match label {
            Some(r) => format!("✂ {r:<11}"),
            None => "  ".to_string() + &" ".repeat(11),
        };
        let why = drops.iter().find(|d| d.line == i).map(|d| format!("  ← {}", d.reason)).unwrap_or_default();
        report.push_str(&format!("{mark} [{i:3}] {:7.2}-{:7.2}  {}{why}\n", s.start, s.end, s.text));
    }
    let kept: f64 = keep.iter().map(|(a, b)| b - a).sum();
    let total = audio.len() as f64 / 16_000.0;
    report.push_str(&format!("\nkept {kept:.1} s of {total:.1} s in {} pieces\n", keep.len()));
    let _ = std::fs::write(out.join("report.txt"), &report);
    print!("{report}");

    // 5. render (free)
    let mut f = String::new();
    let mut cat = String::new();
    for (i, (a, b)) in keep.iter().enumerate() {
        let fade = (b - a - 0.03).max(0.0);
        f.push_str(&format!(
            "[0:v]trim={a:.3}:{b:.3},setpts=PTS-STARTPTS[v{i}];[0:a]atrim={a:.3}:{b:.3},asetpts=PTS-STARTPTS,afade=t=in:d=0.02,afade=t=out:st={fade:.3}:d=0.03[a{i}];"
        ));
        cat.push_str(&format!("[v{i}][a{i}]"));
    }
    f.push_str(&format!("{cat}concat=n={}:v=1:a=1[v][a]", keep.len()));
    let filter = out.join("filter.txt");
    let _ = std::fs::write(&filter, f);
    let edited = out.join("edited.mp4");
    ffmpeg(&[
        "-ss", &start, "-t", &dur, "-i", &video,
        "-/filter_complex", filter.to_str().unwrap_or_default(),
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-c:a", "aac", "-b:a", "160k",
        edited.to_str().unwrap_or_default(),
    ]);
    eprintln!("→ {}", edited.display());
}
