//! The whole auto-cut on 16 kHz mono audio: timed transcript (Gemini) → retakes and asides
//! (text model) → local rules → ranges to keep → the cuts, each with its reasons.

use crate::cutplan::{self, CutOptions};
use crate::gemini_stt::{GeminiTranscriber, Segment, Usage};
use crate::retakes::{self, Drop};
use crate::{AiError, Keys};

/// One suggested cut, in seconds from the start of the analysed audio.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Cut {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    /// `retake`, `aside`, `stutter`, `filler`, `false start`, `noise` or `pause`.
    pub kind: String,
    /// What was said in the cut (empty for a pause).
    pub text: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Analysis {
    pub duration: f64,
    pub segments: Vec<Segment>,
    pub drops: Vec<Drop>,
    pub keep: Vec<(f64, f64)>,
    pub cuts: Vec<Cut>,
    pub tokens_in: u64,
    pub tokens_out: u64,
}

/// Run the paid steps (transcript, retakes) and plan the cuts.
pub fn analyze(audio: &[f32], keys: &Keys, progress: &mut dyn FnMut(&str) -> bool) -> Result<Analysis, AiError> {
    let t = GeminiTranscriber::from_keys(keys).ok_or(AiError::MissingKey("Gemini", "GEMINI_API_KEY"))?;
    let (segs, usage) = t.timed(audio, &mut |i, n| progress(&format!("Transcribing with Gemini ({}/{n})", (i + 1).min(n))))?;
    if !progress("Finding retakes") {
        return Err(AiError::Cancelled);
    }
    let llm = crate::llm::from_keys(keys)?;
    let drops = retakes::find(llm.as_ref(), &segs)?;
    Ok(plan(audio, segs, drops, usage))
}

/// The free part: local rules, partial cuts, the ranges to keep and the cut list.
pub fn plan(audio: &[f32], segs: Vec<Segment>, mut drops: Vec<Drop>, usage: Usage) -> Analysis {
    let model_lines: Vec<usize> = drops.iter().map(|d| d.line).collect();
    drops.extend(retakes::repeated_openings(&segs, &model_lines, 3, 30.0));
    drops.sort_by_key(|d| d.line);
    drops.dedup_by_key(|d| d.line);
    let (segs, drops) = retakes::split_partial_starts(&segs, &drops);
    let lines: Vec<usize> = drops.iter().map(|d| d.line).collect();
    let keep = cutplan::keep_ranges(audio, &segs, &lines, &CutOptions::default());
    let duration = audio.len() as f64 / 16_000.0;
    let cuts = cuts_between(&keep, duration, &segs, &drops, &lines);
    Analysis { duration, segments: segs, drops, keep, cuts, tokens_in: usage.input, tokens_out: usage.output }
}

/// The gaps between kept ranges, each described by the cut segments it covers (most of its
/// time); a gap with no cut segment is a pause.
fn cuts_between(keep: &[(f64, f64)], duration: f64, segs: &[Segment], drops: &[Drop], lines: &[usize]) -> Vec<Cut> {
    let mut gaps = Vec::new();
    let mut t = 0.0;
    for &(a, b) in keep {
        if a - t > 0.01 {
            gaps.push((t, a));
        }
        t = b;
    }
    if duration - t > 0.01 {
        gaps.push((t, duration));
    }
    gaps.into_iter()
        .enumerate()
        .map(|(id, (a, b))| {
            let inside: Vec<(usize, &Segment)> =
                segs.iter().enumerate().filter(|(_, s)| s.end > a && s.start < b && (s.end.min(b) - s.start.max(a)) > 0.5 * (s.end - s.start)).collect();
            let mut kind = String::from("pause");
            let mut reason = String::new();
            let mut text = Vec::new();
            for (i, s) in &inside {
                let k = drops
                    .iter()
                    .find(|d| d.line == *i)
                    .map(|d| d.kind.clone())
                    .or_else(|| cutplan::cut_reason(*i, s, lines).map(str::to_string));
                if let Some(k) = k {
                    if kind == "pause" {
                        kind = k;
                    }
                    if reason.is_empty() {
                        reason = drops.iter().find(|d| d.line == *i).map(|d| d.reason.clone()).unwrap_or_default();
                    }
                    if !s.text.is_empty() {
                        text.push(s.text.clone());
                    }
                }
            }
            Cut { id, start: a, end: b, kind, text: text.join(" "), reason }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_are_the_gaps_with_their_reasons() {
        let segs = vec![Segment::new(1.0, 2.0, "قديم"), Segment::new(3.0, 4.0, "جديد")];
        let drops = vec![Drop { line: 0, kind: "retake".into(), reason: "محاولة قديمة".into() }];
        let keep = vec![(2.9, 4.15)];
        let c = cuts_between(&keep, 6.0, &segs, &drops, &[0]);
        assert_eq!(c.len(), 2);
        assert_eq!((c[0].start, c[0].end, c[0].kind.as_str(), c[0].text.as_str()), (0.0, 2.9, "retake", "قديم"));
        assert_eq!(c[0].reason, "محاولة قديمة");
        assert_eq!((c[1].kind.as_str(), c[1].text.as_str()), ("pause", ""));
    }
}
