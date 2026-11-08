//! Staying inside provider limits: a per-minute request limiter (Gemini's free tier allows 15
//! requests a minute) and retries with back-off for "busy" and "slow down" answers.

use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::AiError;

/// At most `per_minute` requests in any rolling 60 seconds, across threads.
pub struct RateLimiter {
    per_minute: usize,
    sent: Mutex<VecDeque<Instant>>,
}

impl RateLimiter {
    pub fn new(per_minute: usize) -> Self {
        Self { per_minute: per_minute.max(1), sent: Mutex::new(VecDeque::new()) }
    }

    /// Wait until one more request fits, then count it.
    pub fn acquire(&self) {
        loop {
            let wait = {
                let Ok(mut q) = self.sent.lock() else { return };
                let now = Instant::now();
                while q.front().is_some_and(|t| now.duration_since(*t) >= Duration::from_secs(60)) {
                    q.pop_front();
                }
                if q.len() < self.per_minute {
                    q.push_back(now);
                    return;
                }
                q.front().map_or(Duration::from_secs(1), |t| Duration::from_secs(60).saturating_sub(now.duration_since(*t)) + Duration::from_millis(50))
            };
            std::thread::sleep(wait);
        }
    }
}

/// Gemini requests: 12 a minute keeps clear of the free tier's 15.
pub fn gemini() -> &'static RateLimiter {
    static L: OnceLock<RateLimiter> = OnceLock::new();
    L.get_or_init(|| RateLimiter::new(12))
}

/// How long to wait before retrying after `e` (attempt 0, 1, …), or `None` when retrying cannot
/// help (a bad key, a bad request, a refusal). A "retry in 44.6s" hint in the message wins.
pub fn retry_delay(e: &AiError, attempt: u32) -> Option<Duration> {
    let backoff = Duration::from_secs((5u64 << attempt.min(4)).min(60));
    match e {
        AiError::Http { status: 429 | 500 | 502 | 503 | 504 | 529, message, .. } => {
            Some(hinted(message).map_or(backoff, |d| d.min(Duration::from_secs(120))))
        }
        AiError::Network(_) => Some(backoff),
        _ => None,
    }
}

/// "Please retry in 44.68s" → 45 s.
fn hinted(msg: &str) -> Option<Duration> {
    let lower = msg.to_lowercase();
    let at = lower.find("retry in ")? + "retry in ".len();
    let num: String = lower[at..].chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let secs: f64 = num.parse().ok()?;
    secs.is_finite().then(|| Duration::from_secs_f64(secs.max(0.0) + 1.0))
}

/// Run `f`, retrying up to `attempts` times while [`retry_delay`] says it can help.
pub fn with_retry<T>(attempts: u32, mut f: impl FnMut() -> Result<T, AiError>) -> Result<T, AiError> {
    let mut attempt = 0;
    loop {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => match retry_delay(&e, attempt) {
                Some(d) if attempt + 1 < attempts => {
                    std::thread::sleep(d);
                    attempt += 1;
                }
                _ => return Err(e),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn http(status: u16, message: &str) -> AiError {
        AiError::Http { provider: "Gemini", status, message: message.into() }
    }

    #[test]
    fn retry_hints_and_backoff() {
        let quota = http(429, "Quota exceeded … Please retry in 44.68178818s.");
        assert_eq!(retry_delay(&quota, 0), Some(Duration::from_secs_f64(45.68178818)));
        assert_eq!(retry_delay(&http(503, "busy"), 0), Some(Duration::from_secs(5)));
        assert_eq!(retry_delay(&http(503, "busy"), 2), Some(Duration::from_secs(20)));
        assert_eq!(retry_delay(&http(503, "busy"), 9), Some(Duration::from_secs(60)));
        assert_eq!(retry_delay(&http(401, "bad key"), 0), None);
        assert_eq!(retry_delay(&AiError::Refused("Claude", "x".into()), 0), None);
    }

    #[test]
    fn with_retry_stops_on_success_or_hopeless_errors() {
        let mut n = 0;
        let r: Result<u8, AiError> = with_retry(3, || {
            n += 1;
            Err(http(400, "bad request"))
        });
        assert!(r.is_err());
        assert_eq!(n, 1);
    }

    #[test]
    fn limiter_counts_without_waiting_under_the_limit() {
        let l = RateLimiter::new(3);
        let t = Instant::now();
        for _ in 0..3 {
            l.acquire();
        }
        assert!(t.elapsed() < Duration::from_millis(100));
        assert_eq!(l.sent.lock().unwrap().len(), 3);
    }
}
