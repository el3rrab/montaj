//! Find retakes: when the host stops and says a sentence again, every earlier attempt is marked
//! for removal and the last complete take stays. The text model reads the timed transcript
//! (one numbered line per segment) and answers with the line numbers to drop.

use serde::Deserialize;

use crate::AiError;
use crate::gemini_stt::Segment;
use crate::llm::{Llm, Request};

const SYSTEM: &str = "You edit raw YouTube recordings of one host speaking Egyptian Arabic. You find failed attempts that the host repeated, so they can be cut. You answer with JSON only.";

const TASK: &str = "Below is a raw recording, one numbered line per segment: [number] start-end text.\n\
The host often stops, then says the same sentence — or a whole passage of several lines — again, sometimes with different wording after the same opening.\n\
1. Find every passage the host said more than once. List its attempts in order as line ranges [first, last]. An attempt runs from the line where the host starts the passage up to the line before he starts it again. Set \"keep\" to the index of the attempt to keep: the LAST complete one. Passages said only once are not listed.\n\
2. List the asides: lines where the host talks to someone in the room (a child, family) instead of to the audience (e.g. \"ولكن يا بنتي بتعملي ايه؟\"). Talking to the audience is never an aside, even an apology or a greeting.\n\
When unsure, leave it out.\n\
Answer with JSON only, no prose:\n\
{\"passages\": [{\"attempts\": [[first, last], [first, last]], \"keep\": <index>, \"reason\": \"<few Arabic words>\"}], \"asides\": [{\"line\": <number>, \"reason\": \"<few Arabic words>\"}]}";

/// One line to remove.
#[derive(Clone, Debug, serde::Serialize, Deserialize, PartialEq)]
pub struct Drop {
    pub line: usize,
    /// `retake` or `aside`.
    #[serde(default = "retake_kind")]
    pub kind: String,
    #[serde(default)]
    pub reason: String,
}

fn retake_kind() -> String {
    "retake".into()
}

#[derive(Deserialize)]
struct Answer {
    #[serde(default)]
    passages: Vec<Passage>,
    #[serde(default)]
    asides: Vec<Aside>,
}

#[derive(Deserialize)]
struct Passage {
    #[serde(default)]
    attempts: Vec<[usize; 2]>,
    #[serde(default)]
    keep: Option<usize>,
    #[serde(default)]
    reason: String,
}

#[derive(Deserialize)]
struct Aside {
    line: usize,
    #[serde(default)]
    reason: String,
}

/// The prompt's transcript block.
pub fn numbered(segs: &[Segment]) -> String {
    segs.iter()
        .enumerate()
        .map(|(i, s)| if s.disfluent() { format!("[{i}] {:.2}-{:.2} ({}) {}\n", s.start, s.end, s.kind, s.text) } else { format!("[{i}] {:.2}-{:.2} {}\n", s.start, s.end, s.text) })
        .collect()
}

/// Parse the model's answer (the first `{...}` block) into drops: every line of every attempt
/// but the kept one (the last attempt when `keep` is missing or out of range), and the asides.
/// A line inside a kept attempt is never dropped; out-of-range lines are ignored.
pub fn parse(answer: &str, lines: usize) -> Result<Vec<Drop>, AiError> {
    let start = answer.find('{').ok_or_else(|| AiError::BadResponse("text model", "no JSON object".into()))?;
    let end = answer.rfind('}').filter(|&e| e > start).ok_or_else(|| AiError::BadResponse("text model", "unterminated JSON".into()))?;
    let a: Answer = serde_json::from_str(&answer[start..=end]).map_err(|e| AiError::BadResponse("text model", e.to_string()))?;
    let mut protected = vec![false; lines];
    let mut d: Vec<Drop> = Vec::new();
    for p in &a.passages {
        let valid: Vec<(usize, usize)> = p.attempts.iter().map(|r| (r[0].min(r[1]), r[0].max(r[1]))).filter(|(f, _)| *f < lines).map(|(f, l)| (f, l.min(lines - 1))).collect();
        if valid.len() < 2 {
            continue;
        }
        let keep = p.keep.filter(|&k| k < valid.len()).unwrap_or(valid.len() - 1);
        for (i, &(f, l)) in valid.iter().enumerate() {
            if i == keep {
                for slot in &mut protected[f..=l] {
                    *slot = true;
                }
            } else {
                d.extend((f..=l).map(|line| Drop { line, kind: retake_kind(), reason: format!("محاولة قديمة ({f}-{l}): {}", p.reason) }));
            }
        }
    }
    d.extend(a.asides.into_iter().filter(|x| x.line < lines).map(|x| Drop { line: x.line, kind: "aside".into(), reason: x.reason }));
    d.retain(|x| !protected[x.line]);
    d.sort_by_key(|d| d.line);
    d.dedup_by_key(|d| d.line);
    Ok(d)
}

/// Arabic-aware word normalization for comparing openings: punctuation and diacritics
/// removed, alef forms → ا, ة → ه, ى → ي, lower case.
pub fn norm_word(w: &str) -> String {
    w.chars()
        .filter(|c| c.is_alphanumeric())
        .filter(|c| !('\u{064B}'..='\u{0652}').contains(c) && *c != '\u{0670}' && *c != 'ـ')
        .map(|c| match c {
            'أ' | 'إ' | 'آ' | 'ٱ' => 'ا',
            'ة' => 'ه',
            'ى' => 'ي',
            other => other,
        })
        .flat_map(char::to_lowercase)
        .collect()
}

/// Free, local rule: a kept line whose first `n` words open a later line (within `window`
/// seconds) is an abandoned attempt of that later line. `skip` lines are already cut.
pub fn repeated_openings(segs: &[Segment], skip: &[usize], n: usize, window: f64) -> Vec<Drop> {
    let opening = |s: &Segment| -> Vec<String> { s.text.split_whitespace().map(norm_word).filter(|w| !w.is_empty()).take(n).collect() };
    let mut out = Vec::new();
    for (i, s) in segs.iter().enumerate() {
        if skip.contains(&i) || s.disfluent() {
            continue;
        }
        let a = opening(s);
        if a.len() < n {
            continue;
        }
        let later = segs.iter().enumerate().skip(i + 1).take_while(|(_, t)| t.start - s.end <= window).find(|(_, t)| opening(t) == a);
        if let Some((j, _)) = later {
            out.push(Drop { line: i, kind: retake_kind(), reason: format!("بداية اتعادت في السطر {j}") });
        }
    }
    out
}

/// Partial cuts: when an old attempt's first line starts with something said only once and
/// the restart's opening (its first 3 words) begins in the middle of that line, only the tail
/// from that word is a retake. Such lines are split in two (times shared by length) — the head
/// kept, the tail dropped — and the drops renumbered to the new lines.
pub fn split_partial_starts(segs: &[Segment], drops: &[Drop]) -> (Vec<Segment>, Vec<Drop>) {
    let dropped = |k: usize| drops.iter().any(|d| d.line == k && d.kind == "retake");
    let mut split_at: Vec<Option<usize>> = vec![None; segs.len()];
    let mut i = 0;
    while i < segs.len() {
        if !dropped(i) || (i > 0 && dropped(i - 1)) {
            i += 1;
            continue;
        }
        // run of retake lines starting at i; the restart is the next kept line
        let mut j = i;
        while j < segs.len() && dropped(j) {
            j += 1;
        }
        if let Some(restart) = segs.get(j) {
            let open: Vec<String> = restart.text.split_whitespace().map(norm_word).filter(|w| !w.is_empty()).take(3).collect();
            let words: Vec<String> = segs[i].text.split_whitespace().map(norm_word).collect();
            if open.len() == 3 {
                split_at[i] = (1..words.len()).find(|&m| words.len() >= m + 3 && words[m..m + 3] == open[..]);
            }
        }
        i = j;
    }
    let mut out_segs = Vec::new();
    let mut new_index = Vec::with_capacity(segs.len());
    for (k, s) in segs.iter().enumerate() {
        match split_at[k] {
            Some(m) => {
                let words: Vec<&str> = s.text.split_whitespace().collect();
                let head = words[..m].join(" ");
                let tail = words[m..].join(" ");
                let frac = (head.chars().count() as f64 + 0.5) / (s.text.chars().count() as f64 + 1.0);
                let cut = s.start + (s.end - s.start) * frac;
                out_segs.push(Segment { start: s.start, end: cut, text: head, kind: s.kind.clone() });
                new_index.push(out_segs.len());
                out_segs.push(Segment { start: cut, end: s.end, text: tail, kind: s.kind.clone() });
            }
            None => {
                new_index.push(out_segs.len());
                out_segs.push(s.clone());
            }
        }
    }
    let out_drops = drops.iter().filter_map(|d| new_index.get(d.line).map(|&line| Drop { line, ..d.clone() })).collect();
    (out_segs, out_drops)
}

/// A word's stem for loose matching: normalized, a leading و and ال removed.
fn stem(w: &str) -> String {
    let n = norm_word(w);
    let n = n.strip_prefix('و').filter(|r| r.chars().count() >= 2).map_or(n.clone(), str::to_string);
    n.strip_prefix("ال").filter(|r| r.chars().count() >= 2).map_or(n.clone(), str::to_string)
}

/// Safety check on retakes: a line is only cut as a retake when its first word (≥ 2 letters)
/// comes back in a later line within `window` seconds — otherwise nothing repeated it, and
/// cutting it could remove the start of the final take. Asides are not checked.
pub fn confirm(segs: &[Segment], drops: Vec<Drop>, window: f64) -> Vec<Drop> {
    drops
        .into_iter()
        .filter(|d| {
            if d.kind == "aside" {
                return true;
            }
            let Some(s) = segs.get(d.line) else { return false };
            let Some(first) = s.text.split_whitespace().map(stem).find(|w| w.chars().count() >= 3).or_else(|| s.text.split_whitespace().map(stem).find(|w| w.chars().count() >= 2)) else { return false };
            segs.iter().skip(d.line + 1).take_while(|t| t.start - s.end <= window).any(|t| {
                t.text.split_whitespace().map(stem).filter(|w| w.chars().count() >= 2).any(|w| w.starts_with(&first) || first.starts_with(&w))
            })
        })
        .collect()
}

/// A line's opening stem: its first word of ≥ 3 letters (else ≥ 2).
fn opening_stem(s: &Segment) -> Option<String> {
    let words: Vec<String> = s.text.split_whitespace().map(stem).collect();
    words.iter().find(|w| w.chars().count() >= 3).or_else(|| words.iter().find(|w| w.chars().count() >= 2)).cloned()
}

/// Free, local rule: when a retake line opens the same way as a later line that restarts it
/// (within `window` seconds), every line in between is the tail of the abandoned attempt and is
/// cut too — so no orphan half-passage stays behind. Returns the added drops.
pub fn extend_to_restart(segs: &[Segment], drops: &[Drop], window: f64) -> Vec<Drop> {
    let dropped = |k: usize| drops.iter().any(|d| d.line == k);
    let mut added: Vec<Drop> = Vec::new();
    for d in drops.iter().filter(|d| d.kind == "retake") {
        let Some(s) = segs.get(d.line) else { continue };
        let Some(open) = opening_stem(s) else { continue };
        let restart = segs.iter().enumerate().skip(d.line + 1).take_while(|(_, t)| t.start - s.start <= window).find(|(j, t)| {
            !dropped(*j) && opening_stem(t).is_some_and(|o| o.starts_with(&open) || open.starts_with(&o))
        });
        if let Some((j, _)) = restart {
            for k in d.line + 1..j {
                if !dropped(k) && !added.iter().any(|a| a.line == k) {
                    added.push(Drop { line: k, kind: retake_kind(), reason: format!("باقي محاولة اتعادت من السطر {j}") });
                }
            }
        }
    }
    added
}

/// Ask `llm` which segments are superseded retakes.
pub fn find(llm: &dyn Llm, segs: &[Segment]) -> Result<Vec<Drop>, AiError> {
    if segs.is_empty() {
        return Ok(Vec::new());
    }
    let mut r = Request::new(SYSTEM, format!("{TASK}\n\n{}", numbered(segs)));
    r.max_tokens = 8_000;
    let answer = llm.complete(&r)?;
    parse(&answer, segs.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arabic_words_normalize() {
        assert_eq!(norm_word("إزاي؟"), "ازاي");
        assert_eq!(norm_word("الغريبةُ"), "الغريبه");
        assert_eq!(norm_word("CIA,"), "cia");
    }

    #[test]
    fn rephrased_restart_is_caught_but_not_a_lone_sentence() {
        let segs = vec![
            Segment::new(70.9, 72.8, "وازاي الكلام ده يشتغل؟"),
            Segment::new(74.9, 78.4, "وإزاي الكلام ده بيتشكل دلوقت"), // already a retake
            Segment::new(84.1, 87.8, "وازاي الكلام ده بيتشكل دلوقت في التكنولوجيا"),
            Segment::new(89.6, 91.6, "من اول جوجل لحد شاشات موبايلك"),
        ];
        let d = repeated_openings(&segs, &[1], 3, 30.0);
        assert_eq!(d.iter().map(|d| d.line).collect::<Vec<_>>(), vec![0]);
        // too far apart: not a retake
        assert!(repeated_openings(&segs, &[1], 3, 1.0).is_empty());
    }

    #[test]
    fn numbered_lines() {
        let s = vec![Segment::new(1.0, 2.5, "اهلا")];
        assert_eq!(numbered(&s), "[0] 1.00-2.50 اهلا\n");
    }

    #[test]
    fn confirm_keeps_the_start_of_the_final_take() {
        let segs = vec![
            Segment::new(90.9, 91.7, "اربطوا"),                                 // repeated later: a retake
            Segment::new(104.4, 106.3, "اربطوا الأ… اربطوا الأحزمة كده عربين"), // final take: nothing repeats it
            Segment::new(106.3, 108.9, "الحكايات لأننا داخلين على منطقة الـ S4"),
            Segment::new(109.4, 113.9, "ودي منطقة كده غامضة جدا"),
            Segment::new(7.3, 7.5, "النفـ"),
        ];
        let d = |line, kind: &str| Drop { line, kind: kind.into(), reason: String::new() };
        let kept = confirm(&segs, vec![d(0, "retake"), d(1, "retake"), d(3, "aside")], 60.0);
        assert_eq!(kept.iter().map(|d| d.line).collect::<Vec<_>>(), vec![0, 3]);
        // a cut-off word whose stem starts a later word is a real retake
        let s2 = vec![Segment::new(7.3, 7.5, "النفـ"), Segment::new(7.8, 12.1, "من أول ما ركب محرك نفاث")];
        assert_eq!(confirm(&s2, vec![d(0, "retake")], 60.0).len(), 1);
    }

    #[test]
    fn tail_of_an_abandoned_attempt_is_cut_up_to_the_restart() {
        let segs = vec![
            Segment::new(90.9, 91.7, "اربطوا"),
            Segment::new(91.7, 95.2, "الأحزمة عربين حكايات"),
            Segment::new(95.5, 100.3, "الـ S4 دي منطقة خطيرة"),
            Segment::new(100.9, 102.8, "منطقة Area 25"),
            Segment::new(104.4, 106.3, "اربطوا الأحزمة كده"),
            Segment::new(109.4, 113.9, "ودي منطقة غامضة"),
            Segment::new(200.0, 201.0, "اربطوا تاني"), // too late to count as the restart
        ];
        let d = vec![Drop { line: 0, kind: "retake".into(), reason: String::new() }, Drop { line: 1, kind: "retake".into(), reason: String::new() }];
        let added = extend_to_restart(&segs, &d, 40.0);
        assert_eq!(added.iter().map(|a| a.line).collect::<Vec<_>>(), vec![2, 3]);
        // nothing restarts it within the window: nothing added
        assert!(extend_to_restart(&segs, &[Drop { line: 4, kind: "retake".into(), reason: String::new() }], 40.0).is_empty());
    }

    #[test]
    fn a_line_that_starts_a_retake_midway_keeps_its_head() {
        let segs = vec![
            Segment::new(2.2, 3.1, "في الحلقة دي"),
            Segment::new(3.3, 7.2, "هنمشي مع باب خطوة خطوة من أول ما ركب محركه"),
            Segment::new(7.3, 7.5, "النفـ"),
            Segment::new(7.8, 12.1, "من أول ما ركب محرك نفاث"),
        ];
        let d = |line| Drop { line, kind: "retake".into(), reason: String::new() };
        let (s, dr) = split_partial_starts(&segs, &[d(1), d(2)]);
        assert_eq!(s.len(), 5);
        assert_eq!(s[1].text, "هنمشي مع باب خطوة خطوة");
        assert_eq!(s[2].text, "من أول ما ركب محركه");
        assert!(s[1].end > s[1].start && s[1].end == s[2].start && s[2].end == 7.2);
        // the head (line 1) is no longer dropped; the tail (2) and النفـ (3) are
        assert_eq!(dr.iter().map(|d| d.line).collect::<Vec<_>>(), vec![2, 3]);
        // a line that already opens with the restart is dropped whole
        let segs2 = vec![Segment::new(0.0, 1.0, "لكن الطيارة ديت"), Segment::new(2.0, 3.0, "لكن الطيارة ديت للأسف")];
        let (s2, dr2) = split_partial_starts(&segs2, &[d(0)]);
        assert_eq!((s2.len(), dr2[0].line), (2, 0));
    }

    #[test]
    fn parse_drops_whole_old_attempts_and_protects_the_kept_one() {
        let a = "Here you go:\n```json\n{\"passages\":[{\"attempts\":[[0,5],[6,10]],\"keep\":1,\"reason\":\"الطيارة\"},\
                 {\"attempts\":[[12,16],[17,19]],\"reason\":\"الغريبة\"},{\"attempts\":[[30,31]],\"keep\":0}],\
                 \"asides\":[{\"line\":21,\"reason\":\"يا بنتي\"},{\"line\":7},{\"line\":99}]}\n```";
        let d = parse(a, 25).unwrap();
        let lines: Vec<usize> = d.iter().map(|d| d.line).collect();
        // attempts 0-5 and 12-16 go; 6-10 and 17-19 stay; the aside on kept line 7 is refused;
        // a passage with a single attempt (30-31) cuts nothing; line 99 is out of range
        assert_eq!(lines, vec![0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 21]);
        assert_eq!(d.last().map(|x| x.kind.as_str()), Some("aside"));
        assert!(parse("no json", 3).is_err());
        assert!(parse("{\"passages\":[],\"asides\":[]}", 3).unwrap().is_empty());
    }
}
