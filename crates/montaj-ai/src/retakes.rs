//! Find retakes: when the host stops and says a sentence again, every earlier attempt is marked
//! for removal and the last complete take stays. The text model reads the timed transcript
//! (one numbered line per segment) and answers with the line numbers to drop.

use serde::Deserialize;

use crate::AiError;
use crate::gemini_stt::Segment;
use crate::llm::{Llm, Request};

const SYSTEM: &str = "You edit raw YouTube recordings of one host speaking Egyptian Arabic. You find failed attempts that the host repeated, so they can be cut. You answer with JSON only.";

const TASK: &str = "Below is a raw recording, one numbered line per segment: [number] start-end text.\n\
The host often stops mid-sentence or says something wrong, then says the same sentence again.\n\
Mark for removal every line that is an earlier attempt of something said again later (a false start, an unfinished sentence, a version that the host then repeated). Keep the LAST complete take of each sentence.\n\
Do not mark lines whose content is said only once. When unsure, keep the line.\n\
Answer with JSON only, no prose: {\"drop\": [{\"line\": <number>, \"reason\": \"<few Arabic words>\"}]}";

/// One line to remove.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Drop {
    pub line: usize,
    #[serde(default)]
    pub reason: String,
}

#[derive(Deserialize)]
struct Answer {
    #[serde(default)]
    drop: Vec<Drop>,
}

/// The prompt's transcript block.
pub fn numbered(segs: &[Segment]) -> String {
    segs.iter().enumerate().map(|(i, s)| format!("[{i}] {:.2}-{:.2} {}\n", s.start, s.end, s.text)).collect()
}

/// Parse the model's answer: the first `{...}` block, lines out of range dropped, sorted, unique.
pub fn parse(answer: &str, lines: usize) -> Result<Vec<Drop>, AiError> {
    let start = answer.find('{').ok_or_else(|| AiError::BadResponse("text model", "no JSON object".into()))?;
    let end = answer.rfind('}').filter(|&e| e > start).ok_or_else(|| AiError::BadResponse("text model", "unterminated JSON".into()))?;
    let a: Answer = serde_json::from_str(&answer[start..=end]).map_err(|e| AiError::BadResponse("text model", e.to_string()))?;
    let mut d: Vec<Drop> = a.drop.into_iter().filter(|d| d.line < lines).collect();
    d.sort_by_key(|d| d.line);
    d.dedup_by_key(|d| d.line);
    Ok(d)
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
    fn numbered_lines() {
        let s = vec![Segment { start: 1.0, end: 2.5, text: "اهلا".into() }];
        assert_eq!(numbered(&s), "[0] 1.00-2.50 اهلا\n");
    }

    #[test]
    fn parse_tolerates_prose_and_bad_lines() {
        let a = "Here you go:\n```json\n{\"drop\":[{\"line\":3,\"reason\":\"إعادة\"},{\"line\":0},{\"line\":3},{\"line\":99}]}\n```";
        let d = parse(a, 10).unwrap();
        assert_eq!(d.iter().map(|d| d.line).collect::<Vec<_>>(), vec![0, 3]);
        assert_eq!(d[1].reason, "إعادة");
        assert!(parse("no json", 3).is_err());
        assert!(parse("{\"drop\":[]}", 3).unwrap().is_empty());
    }
}
