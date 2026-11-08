//! From a timed transcript to the ranges to keep: drop retakes and disfluencies (stutters,
//! fillers, false starts), pad each kept segment a little without reaching into a dropped
//! neighbour, then cut long pauses inside what is kept down to a short breath.
//! Pure functions over 16 kHz audio; FilmCraft's VAD decides what is silence.

use crate::gemini_stt::Segment;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CutOptions {
    /// Seconds kept before / after each kept segment.
    pub pad_before: f64,
    pub pad_after: f64,
    /// A silence inside kept speech longer than this is shortened…
    pub max_pause: f64,
    /// …to this much air on each side.
    pub air: f64,
    /// Pieces shorter than this are dropped (clicks, breaths).
    pub min_piece: f64,
}

impl Default for CutOptions {
    fn default() -> Self {
        Self { pad_before: 0.08, pad_after: 0.15, max_pause: 0.35, air: 0.12, min_piece: 0.15 }
    }
}

/// Why a segment is cut, or `None` to keep it.
pub fn cut_reason(i: usize, s: &Segment, drop: &[usize]) -> Option<&'static str> {
    if drop.contains(&i) {
        Some("retake")
    } else if s.disfluent() {
        Some(match s.kind.as_str() {
            "stutter" => "stutter",
            "filler" => "filler",
            "aside" => "aside",
            "noise" => "noise",
            _ => "false start",
        })
    } else {
        None
    }
}

/// Seconds-ranges to keep, in order, non-overlapping.
pub fn keep_ranges(audio_16k: &[f32], segs: &[Segment], drop: &[usize], o: &CutOptions) -> Vec<(f64, f64)> {
    let dur = audio_16k.len() as f64 / 16_000.0;
    let ok = |s: &Segment| s.start.is_finite() && s.end.is_finite() && s.end > s.start;
    // every segment's span (kept or not), to keep padding out of dropped neighbours
    let mut spans: Vec<(f64, f64, bool)> =
        segs.iter().enumerate().filter(|(_, s)| ok(s)).map(|(i, s)| (s.start.max(0.0), s.end.min(dur), cut_reason(i, s, drop).is_none())).collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut ranges: Vec<(f64, f64)> = Vec::new();
    for (k, &(s, e, keep)) in spans.iter().enumerate() {
        if !keep {
            continue;
        }
        let prev_end = if k > 0 { spans[k - 1].1 } else { 0.0 };
        let next_start = spans.get(k + 1).map_or(dur, |n| n.0);
        let a = (s - o.pad_before).max(if k > 0 && !spans[k - 1].2 { prev_end } else { 0.0 }).max(0.0);
        let b = (e + o.pad_after).min(if spans.get(k + 1).is_some_and(|n| !n.2) { next_start } else { dur }).min(dur);
        if b <= a {
            continue;
        }
        match ranges.last_mut() {
            Some(last) if a <= last.1 + 0.05 => last.1 = last.1.max(b),
            _ => ranges.push((a, b)),
        }
    }
    let ranges = shorten_pauses(audio_16k, &ranges, o);
    ranges.into_iter().filter(|(a, b)| b - a >= o.min_piece).collect()
}

/// Split each range at silences longer than `max_pause`, leaving `air` on both sides.
fn shorten_pauses(audio_16k: &[f32], ranges: &[(f64, f64)], o: &CutOptions) -> Vec<(f64, f64)> {
    let db = filmcraft_speech::vad::frame_db(audio_16k);
    if db.is_empty() {
        return ranges.to_vec();
    }
    let th = filmcraft_speech::vad::threshold(&db);
    const FRAME: f64 = 0.01; // frame_db works in 10 ms frames
    let min_run = ((o.max_pause + 2.0 * o.air) / FRAME).ceil() as usize;
    let mut out = Vec::new();
    for &(a, b) in ranges {
        let fa = (a / FRAME) as usize;
        let fb = ((b / FRAME) as usize).min(db.len());
        let mut start = a;
        let mut f = fa;
        while f < fb {
            if db[f] >= th {
                f += 1;
                continue;
            }
            let run_start = f;
            while f < fb && db[f] < th {
                f += 1;
            }
            if f - run_start >= min_run && run_start > fa && f < fb {
                let cut_a = run_start as f64 * FRAME + o.air;
                let cut_b = f as f64 * FRAME - o.air;
                out.push((start, cut_a));
                start = cut_b;
            }
        }
        out.push((start, b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `seconds` of near-silence with a loud tone over each `(from, to)`.
    fn audio(seconds: f64, tones: &[(f64, f64)]) -> Vec<f32> {
        let n = (seconds * 16_000.0) as usize;
        let mut a: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 0.0005 } else { -0.0005 }).collect();
        for &(f, t) in tones {
            for i in (f * 16_000.0) as usize..((t * 16_000.0) as usize).min(n) {
                a[i] = 0.3 * ((i as f32) * 0.2).sin();
            }
        }
        a
    }

    fn seg(s: f64, e: f64, kind: &str) -> Segment {
        Segment { kind: kind.into(), ..Segment::new(s, e, "x") }
    }

    #[test]
    fn drops_retakes_and_disfluencies_and_keeps_padding_out_of_them() {
        let a = audio(10.0, &[(1.0, 2.0), (2.1, 2.5), (3.0, 4.0), (5.0, 6.0)]);
        let segs = vec![seg(1.0, 2.0, "speech"), seg(2.1, 2.5, "stutter"), seg(3.0, 4.0, "speech"), seg(5.0, 6.0, "speech")];
        let r = keep_ranges(&a, &segs, &[2], &CutOptions::default());
        assert_eq!(r.len(), 2, "{r:?}");
        // padding after segment 0 stops where the stutter starts
        assert!((r[0].0 - 0.92).abs() < 1e-9 && (r[0].1 - 2.1).abs() < 1e-9, "{r:?}");
        assert!((r[1].0 - 4.92).abs() < 1e-9 && (r[1].1 - 6.15).abs() < 1e-9, "{r:?}");
        assert_eq!(cut_reason(1, &segs[1], &[2]), Some("stutter"));
        assert_eq!(cut_reason(2, &segs[2], &[2]), Some("retake"));
        assert_eq!(cut_reason(0, &segs[0], &[2]), None);
    }

    #[test]
    fn long_pause_inside_a_kept_segment_is_shortened() {
        // one segment from 1 to 5 s, but silent from 2 to 4 s
        let a = audio(6.0, &[(1.0, 2.0), (4.0, 5.0)]);
        let r = keep_ranges(&a, &[seg(1.0, 5.0, "speech")], &[], &CutOptions::default());
        assert_eq!(r.len(), 2, "{r:?}");
        assert!((r[0].1 - 2.12).abs() < 0.03, "{r:?}");
        assert!((r[1].0 - 3.88).abs() < 0.03, "{r:?}");
        // a short breath stays
        let b = audio(4.0, &[(1.0, 2.0), (2.2, 3.0)]);
        assert_eq!(keep_ranges(&b, &[seg(1.0, 3.0, "speech")], &[], &CutOptions::default()).len(), 1);
    }

    #[test]
    fn bad_times_are_ignored() {
        let a = audio(2.0, &[(0.5, 1.0)]);
        let segs = vec![seg(f64::NAN, 1.0, "speech"), seg(1.5, 1.2, "speech"), seg(0.5, 1.0, "speech")];
        assert_eq!(keep_ranges(&a, &segs, &[], &CutOptions::default()).len(), 1);
        assert!(keep_ranges(&[], &segs, &[], &CutOptions::default()).is_empty());
    }
}
