//! `montaj-retakes <segments.txt>`: try retake detection on the output of `montaj-transcribe`
//! (lines "start end text") with the text model chosen in the keys file.

use std::path::PathBuf;

use montaj_ai::gemini_stt::Segment;
use montaj_ai::keys::Keys;

fn data_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("FILMCRAFT_DATA_DIR").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(d));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/FilmCraft"))
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: montaj-retakes <segments.txt>");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let segs: Vec<Segment> = text
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let start = it.next()?.parse().ok()?;
            let end = it.next()?.parse().ok()?;
            let rest: Vec<&str> = it.collect();
            Some(Segment { start, end, text: rest.join(" ") })
        })
        .collect();
    let llm = match montaj_ai::llm::from_keys(&Keys::load(data_dir().as_deref())) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    eprintln!("{} segments → {}", segs.len(), llm.id());
    match montaj_ai::retakes::find(llm.as_ref(), &segs) {
        Ok(drops) => {
            let mut cut = 0.0;
            for d in &drops {
                let s = &segs[d.line];
                cut += s.end - s.start;
                println!("✂ [{}] {:7.2}-{:7.2}  {}   ← {}", d.line, s.start, s.end, s.text, d.reason);
            }
            println!("\n{} of {} segments marked, {:.1} s of speech removed", drops.len(), segs.len(), cut);
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
