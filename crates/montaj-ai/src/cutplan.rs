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
    /// Only audio this many dB under the voice threshold counts as silence when shortening
    /// pauses, so soft word endings and breathy syllables are never taken for a pause.
    pub quiet_db: f32,
    /// Voiced stretches at least this long that the transcript missed are kept when they sit
    /// between kept speech (0 = off).
    pub min_voice: f64,
    /// Missed voice must be this many dB above the voice threshold (breathing is not speech).
    pub voice_margin_db: f32,
    /// Sounds this short inside a silence count as silence (seconds).
    pub blip: f64,
}

impl Default for CutOptions {
    fn default() -> Self {
        // tuned on a 31-minute raw recording (montaj-replan): wider pads, only real pauses
        // shortened to a natural breath, and voice the transcript missed is kept
        Self { pad_before: 0.15, pad_after: 0.25, max_pause: 0.5, air: 0.2, min_piece: 0.15, quiet_db: 0.0, min_voice: 0.3, voice_margin_db: 8.0, blip: 0.15 }
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
    // voice the transcript missed, between two kept segments: not understood, so not cut
    for (a, b) in untranscribed_voice(audio_16k, &spans, o.min_voice, o.voice_margin_db) {
        ranges.push(((a - o.pad_before).max(0.0), (b + o.pad_after).min(dur)));
    }
    ranges.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (a, b) in ranges {
        match merged.last_mut() {
            Some(last) if a <= last.1 + 0.05 => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let ranges = shorten_pauses(audio_16k, &merged, o);
    ranges.into_iter().filter(|(a, b)| b - a >= o.min_piece).collect()
}

/// Voiced stretches (≥ `min_len` s) that no segment covers and that lie between two kept
/// segments: speech the transcript missed. Inside a cut passage they are left to be cut.
fn untranscribed_voice(audio_16k: &[f32], spans: &[(f64, f64, bool)], min_len: f64, margin_db: f32) -> Vec<(f64, f64)> {
    if min_len <= 0.0 {
        return Vec::new();
    }
    let db = filmcraft_speech::vad::frame_db(audio_16k);
    if db.is_empty() {
        return Vec::new();
    }
    let th = filmcraft_speech::vad::threshold(&db) + margin_db;
    let covered = |t: f64| spans.iter().any(|&(s, e, _)| t >= s && t < e);
    let mut out = Vec::new();
    let mut f = 0;
    while f < db.len() {
        let t = f as f64 / 100.0;
        if db[f] < th || covered(t) {
            f += 1;
            continue;
        }
        let a = f;
        while f < db.len() && db[f] >= th && !covered(f as f64 / 100.0) {
            f += 1;
        }
        let (s, e) = (a as f64 / 100.0, f as f64 / 100.0);
        if e - s < min_len {
            continue;
        }
        let prev = spans.iter().filter(|x| x.1 <= s + 1e-9).max_by(|x, y| x.1.total_cmp(&y.1));
        let next = spans.iter().filter(|x| x.0 >= e - 1e-9).min_by(|x, y| x.0.total_cmp(&y.0));
        if prev.is_some_and(|p| p.2) && next.is_some_and(|n| n.2) {
            out.push((s, e));
        }
    }
    out
}

/// Split each range at silences longer than `max_pause`, leaving `air` on both sides.
fn shorten_pauses(audio_16k: &[f32], ranges: &[(f64, f64)], o: &CutOptions) -> Vec<(f64, f64)> {
    let db = filmcraft_speech::vad::frame_db(audio_16k);
    if db.is_empty() {
        return ranges.to_vec();
    }
    let th = filmcraft_speech::vad::threshold(&db) - o.quiet_db;
    const FRAME: f64 = 0.01; // frame_db works in 10 ms frames
    let silent = silence_mask(&db, th, (o.blip / FRAME) as usize);
    let min_run = ((o.max_pause + 2.0 * o.air) / FRAME).ceil() as usize;
    let mut out = Vec::new();
    let air_frames = (o.air / FRAME).ceil() as usize;
    for &(a, b) in ranges {
        let fa = (a / FRAME) as usize;
        let fb = ((b / FRAME) as usize).min(db.len());
        let mut start = a;
        let mut end = b;
        let mut f = fa;
        let mut voiced = false;
        while f < fb {
            if !silent[f] {
                voiced = true;
                f += 1;
                continue;
            }
            let run_start = f;
            while f < fb && silent[f] {
                f += 1;
            }
            let len = f - run_start;
            if run_start == fa && f < fb {
                // silence at the start of the range (a segment timed too early): trim to a breath
                if len > air_frames {
                    start = start.max(f as f64 * FRAME - o.air);
                }
            } else if f >= fb && run_start > fa {
                // silence at the end (a segment timed too long): trim to a breath
                if len > air_frames {
                    end = end.min(run_start as f64 * FRAME + o.air);
                }
            } else if len >= min_run && run_start > fa && f < fb {
                let cut_a = run_start as f64 * FRAME + o.air;
                let cut_b = f as f64 * FRAME - o.air;
                out.push((start, cut_a));
                start = cut_b;
            }
        }
        // a range with no voice at all is dropped
        if voiced && end > start {
            out.push((start, end));
        }
    }
    out
}

/// Frames under `th`, with sounds of at most `blip` frames between two silent frames counted
/// as silence too (a breath or a click does not split a pause in two).
fn silence_mask(db: &[f32], th: f32, blip: usize) -> Vec<bool> {
    let mut s: Vec<bool> = db.iter().map(|&v| v < th).collect();
    let mut f = 0;
    while f < s.len() {
        if s[f] {
            f += 1;
            continue;
        }
        let a = f;
        while f < s.len() && !s[f] {
            f += 1;
        }
        if a > 0 && f < s.len() && f - a <= blip {
            for x in &mut s[a..f] {
                *x = true;
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tight settings these tests were written for.
    fn tight() -> CutOptions {
        CutOptions { pad_before: 0.08, pad_after: 0.15, max_pause: 0.35, air: 0.12, min_piece: 0.15, quiet_db: 0.0, min_voice: 0.0, voice_margin_db: 0.0, blip: 0.0 }
    }

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
        let r = keep_ranges(&a, &segs, &[2], &tight());
        assert_eq!(r.len(), 2, "{r:?}");
        // padding after segment 0 stops where the stutter starts
        assert!((r[0].0 - 0.92).abs() < 1e-9 && (r[0].1 - 2.1).abs() < 1e-9, "{r:?}");
        // the pad after the last word is silence, so it is trimmed to a breath (air)
        assert!((r[1].0 - 4.92).abs() < 1e-9 && (r[1].1 - 6.12).abs() < 1e-9, "{r:?}");
        assert_eq!(cut_reason(1, &segs[1], &[2]), Some("stutter"));
        assert_eq!(cut_reason(2, &segs[2], &[2]), Some("retake"));
        assert_eq!(cut_reason(0, &segs[0], &[2]), None);
    }

    #[test]
    fn long_pause_inside_a_kept_segment_is_shortened() {
        // one segment from 1 to 5 s, but silent from 2 to 4 s
        let a = audio(6.0, &[(1.0, 2.0), (4.0, 5.0)]);
        let r = keep_ranges(&a, &[seg(1.0, 5.0, "speech")], &[], &tight());
        assert_eq!(r.len(), 2, "{r:?}");
        assert!((r[0].1 - 2.12).abs() < 0.03, "{r:?}");
        assert!((r[1].0 - 3.88).abs() < 0.03, "{r:?}");
        // a short breath stays
        let b = audio(4.0, &[(1.0, 2.0), (2.2, 3.0)]);
        assert_eq!(keep_ranges(&b, &[seg(1.0, 3.0, "speech")], &[], &tight()).len(), 1);
    }

    #[test]
    fn voice_the_transcript_missed_between_kept_speech_stays() {
        // speech 1-2 s (segment), voice 2.6-3.4 s with no segment, speech 4-5 s (segment),
        // then voice 6.0-6.8 s with no segment before a dropped retake at 7-8 s
        let a = audio(9.0, &[(1.0, 2.0), (2.6, 3.4), (4.0, 5.0), (6.0, 6.8), (7.0, 8.0)]);
        let segs = vec![seg(1.0, 2.0, "speech"), seg(4.0, 5.0, "speech"), seg(7.0, 8.0, "speech")];
        let o = CutOptions { min_voice: 0.3, max_pause: 5.0, ..tight() };
        let r = keep_ranges(&a, &segs, &[2], &o);
        let kept = |t: f64| r.iter().any(|&(s, e)| t >= s && t < e);
        assert!(kept(3.0), "missed voice between kept speech is kept: {r:?}");
        assert!(!kept(6.4), "missed voice next to a dropped retake is left to be cut: {r:?}");
        assert!(!kept(7.5));
        // off: the missed voice is cut
        let r0 = keep_ranges(&a, &segs, &[2], &CutOptions { min_voice: 0.0, ..o });
        assert!(!r0.iter().any(|&(s, e)| 3.0 >= s && 3.0 < e));
    }

    #[test]
    fn a_breath_does_not_split_a_long_pause() {
        // one kept segment 1-6 s: speech, 3 s of silence with a 0.1 s breath in the middle, speech
        let a = audio(7.0, &[(1.0, 2.0), (3.45, 3.55), (5.0, 6.0)]);
        let s = vec![seg(1.0, 6.0, "speech")];
        let with_blip = keep_ranges(&a, &s, &[], &CutOptions { blip: 0.15, max_pause: 0.5, air: 0.2, ..tight() });
        assert_eq!(with_blip.len(), 2, "the pause is cut despite the breath: {with_blip:?}");
        let without = keep_ranges(&a, &s, &[], &CutOptions { blip: 0.0, max_pause: 1.2, air: 0.2, ..tight() });
        assert_eq!(without.len(), 1, "each half alone is too short to cut: {without:?}");
    }

    #[test]
    fn bad_times_are_ignored() {
        let a = audio(2.0, &[(0.5, 1.0)]);
        let segs = vec![seg(f64::NAN, 1.0, "speech"), seg(1.5, 1.2, "speech"), seg(0.5, 1.0, "speech")];
        assert_eq!(keep_ranges(&a, &segs, &[], &tight()).len(), 1);
        assert!(keep_ranges(&[], &segs, &[], &tight()).is_empty());
    }
}
