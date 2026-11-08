//! `montaj-transcribe <audio.wav>`: try Gemini speech-to-text on a short 16 kHz mono 16-bit WAV
//! excerpt and print the timed segments, the final words and the tokens used.
//! (Make an excerpt with: ffmpeg -ss 60 -t 120 -i video.mp4 -ac 1 -ar 16000 -c:a pcm_s16le x.wav)

use std::path::PathBuf;

use montaj_ai::gemini_stt::GeminiTranscriber;
use montaj_ai::keys::Keys;

fn data_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("FILMCRAFT_DATA_DIR").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(d));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/FilmCraft"))
}

/// Samples of a 16-bit PCM WAV (first channel).
fn read_wav(path: &str) -> Result<(Vec<f32>, u32), String> {
    let b = std::fs::read(path).map_err(|e| e.to_string())?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let (mut i, mut rate, mut channels) = (12, 0u32, 1usize);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let body = &b[i + 8..(i + 8 + len).min(b.len())];
        if id == b"fmt " && body.len() >= 16 {
            channels = u16::from_le_bytes([body[2], body[3]]).max(1) as usize;
            rate = u32::from_le_bytes([body[4], body[5], body[6], body[7]]);
        } else if id == b"data" {
            let s = body.chunks_exact(2 * channels).map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0).collect();
            return Ok((s, rate));
        }
        i += 8 + len + (len & 1);
    }
    Err("no audio data".into())
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: montaj-transcribe <16 kHz mono wav>");
        std::process::exit(2);
    };
    let keys = Keys::load(data_dir().as_deref());
    let Some(t) = GeminiTranscriber::from_keys(&keys) else {
        eprintln!("no Gemini key (run montaj-keys)");
        std::process::exit(2);
    };
    let (audio, rate) = match read_wav(&path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    if rate != 16_000 {
        eprintln!("need 16 kHz audio, got {rate} Hz");
        std::process::exit(1);
    }
    let started = std::time::Instant::now();
    let (segs, usage) = match t.chunk(&audio) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    for s in &segs {
        println!("{:7.2} {:7.2}  {}", s.start, s.end, s.text);
    }
    let mut words = montaj_ai::gemini_stt::words(&segs, 0.0, audio.len() as f64 / 16_000.0);
    filmcraft_speech::vad::tighten_words(&audio, &mut words);
    eprintln!(
        "\n{} segments, {} words, {:.1} s audio, {:.1} s wait; tokens: input {}, output {}, thinking {}",
        segs.len(),
        words.len(),
        audio.len() as f64 / 16_000.0,
        started.elapsed().as_secs_f64(),
        usage.input,
        usage.output,
        usage.thinking
    );
}
