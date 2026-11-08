//! 16-bit mono WAV encoding and splitting long audio at quiet points.

/// Encode mono `samples` (-1..1) at `rate` Hz as a 16-bit PCM WAV file.
pub fn encode(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut b = Vec::with_capacity(44 + samples.len() * 2);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_len).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes()); // PCM
    b.extend_from_slice(&1u16.to_le_bytes()); // mono
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 2).to_le_bytes());
    b.extend_from_slice(&2u16.to_le_bytes());
    b.extend_from_slice(&16u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let v = if s.is_finite() { (s.clamp(-1.0, 1.0) * 32767.0).round() as i16 } else { 0 };
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Split `len` samples into chunks of at most `max` samples, cutting each chunk at the quietest
/// `window`-sample stretch within its last `search` samples, so no word is cut in half.
pub fn split_points(audio: &[f32], max: usize, search: usize, window: usize) -> Vec<std::ops::Range<usize>> {
    let len = audio.len();
    let max = max.max(1);
    let window = window.max(1);
    let mut out = Vec::new();
    let mut start = 0;
    while start < len {
        if len - start <= max {
            out.push(start..len);
            break;
        }
        let hard_end = start + max;
        let from = hard_end.saturating_sub(search).max(start + window);
        let mut best = hard_end;
        let mut best_energy = f64::MAX;
        let mut i = from;
        while i + window <= hard_end {
            let e: f64 = audio[i..i + window].iter().map(|&s| (s as f64) * (s as f64)).sum();
            if e < best_energy {
                best_energy = e;
                best = i + window / 2;
            }
            i += window / 2;
        }
        out.push(start..best);
        start = best;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_and_samples() {
        let w = encode(&[0.0, 1.0, -1.0, 2.0, f32::NAN], 16_000);
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(&w[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes([w[24], w[25], w[26], w[27]]), 16_000);
        assert_eq!(u32::from_le_bytes([w[40], w[41], w[42], w[43]]), 10);
        let s: Vec<i16> = w[44..].chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(s, vec![0, 32767, -32767, 32767, 0]);
    }

    #[test]
    fn splits_at_the_quiet_part_and_covers_everything() {
        // loud everywhere except a quiet gap around sample 900
        let mut a = vec![0.5f32; 2500];
        for s in &mut a[880..920] {
            *s = 0.0;
        }
        let parts = split_points(&a, 1000, 300, 20);
        assert_eq!(parts[0].start, 0);
        assert!((880..=920).contains(&parts[0].end), "{parts:?}");
        assert_eq!(parts.last().unwrap().end, 2500);
        for w in parts.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert!(parts.iter().all(|r| r.len() <= 1000));
    }

    #[test]
    fn short_audio_is_one_chunk() {
        assert_eq!(split_points(&[0.1; 10], 1000, 300, 20), vec![0..10]);
        assert!(split_points(&[], 1000, 300, 20).is_empty());
    }
}
