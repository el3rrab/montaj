//! `montaj-replan <analysis.json> <audio16k.wav>`: re-plan a saved analysis with different cut
//! settings (no API calls) and measure how much real speech each setting removes:
//! voiced audio inside segments that should stay, and cut edges that land inside voice.

use montaj_ai::autocut::Analysis;
use montaj_ai::cutplan::{self, CutOptions};

fn read_wav16(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_default();
    b.get(44..).unwrap_or(&[]).as_chunks::<2>().0.iter().map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(an), Some(wav)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: montaj-replan <analysis.json> <audio16k.wav>");
        std::process::exit(2);
    };
    let a: Analysis = match std::fs::read_to_string(an).ok().and_then(|t| serde_json::from_str(&t).ok()) {
        Some(a) => a,
        None => {
            eprintln!("cannot read {an}");
            std::process::exit(1);
        }
    };
    let audio = read_wav16(wav);
    let db = filmcraft_speech::vad::frame_db(&audio);
    let th = filmcraft_speech::vad::threshold(&db);
    let lines: Vec<usize> = a.drops.iter().map(|d| d.line).collect();
    // frames that belong to speech that should stay: inside a kept segment and voiced
    let mut must_keep = vec![false; db.len()];
    for (i, s) in a.segments.iter().enumerate() {
        if cutplan::cut_reason(i, s, &lines).is_some() {
            continue;
        }
        for f in ((s.start * 100.0) as usize)..((s.end * 100.0) as usize).min(db.len()) {
            must_keep[f] = db[f] >= th;
        }
    }
    let total_voice: usize = must_keep.iter().filter(|&&b| b).count();
    let settings = [
        ("current", CutOptions::default()),
        ("wider pads", CutOptions { pad_before: 0.15, pad_after: 0.25, ..CutOptions::default() }),
        ("pauses >0.8s", CutOptions { pad_before: 0.15, pad_after: 0.25, max_pause: 0.8, air: 0.25, ..CutOptions::default() }),
        ("pauses >0.8s, quiet -6dB", CutOptions { pad_before: 0.15, pad_after: 0.25, max_pause: 0.8, air: 0.25, quiet_db: 6.0, ..CutOptions::default() }),
        ("pauses >1.2s, quiet -8dB", CutOptions { pad_before: 0.2, pad_after: 0.3, max_pause: 1.2, air: 0.3, quiet_db: 8.0, ..CutOptions::default() }),
        ("+ keep missed voice 0.3s", CutOptions { pad_before: 0.15, pad_after: 0.25, max_pause: 0.8, air: 0.25, quiet_db: 6.0, min_voice: 0.3, ..CutOptions::default() }),
    ];
    let old: f64 = a.cuts.iter().map(|c| c.end - c.start).sum();
    let new = montaj_ai::autocut::plan(&audio, a.segments.clone(), a.drops.clone(), montaj_ai::gemini_stt::Usage::default());
    let newcut: f64 = new.cuts.iter().map(|c| c.end - c.start).sum();
    if let Some(p) = args.get(3) {
        let _ = std::fs::write(p, serde_json::to_string(&new.keep).unwrap_or_default());
    }
    let by = |x: &montaj_ai::autocut::Analysis| { let mut m = std::collections::BTreeMap::new(); for c in &x.cuts { let e = m.entry(c.kind.clone()).or_insert((0usize, 0.0f64)); e.0 += 1; e.1 += c.end - c.start; } m };
    println!("saved plan: cut {old:.0} s of {:.0} s  {:?}", a.duration, by(&a));
    println!("new plan:   cut {newcut:.0} s of {:.0} s  {:?}\n", new.duration, by(&new));
    println!("{:<28} {:>8} {:>8} {:>16} {:>14}", "setting", "kept s", "pieces", "speech lost s", "edges in voice");
    for (name, o) in settings {
        let keep = cutplan::keep_ranges(&audio, &a.segments, &lines, &o);
        let mut kept = vec![false; db.len()];
        for &(s, e) in &keep {
            for f in ((s * 100.0) as usize)..((e * 100.0) as usize).min(db.len()) {
                kept[f] = true;
            }
        }
        let lost = must_keep.iter().zip(&kept).filter(|(m, k)| **m && !**k).count();
        let edges = keep
            .iter()
            .flat_map(|&(s, e)| [s, e])
            .filter(|&t| {
                let f = (t * 100.0) as usize;
                (f.saturating_sub(2)..(f + 2).min(db.len())).any(|g| db[g] >= th)
            })
            .count();
        let secs: f64 = keep.iter().map(|(s, e)| e - s).sum();
        println!("{name:<28} {secs:>8.0} {:>8} {:>10.1} ({:>4.1}%) {edges:>8}/{}", keep.len(), lost as f64 / 100.0, 100.0 * lost as f64 / total_voice.max(1) as f64, keep.len() * 2);
    }
}
