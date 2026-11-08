//! `montaj-keys`: set and check Montaj's cloud AI keys without putting them anywhere else.
//!
//!   montaj-keys            ask for each key (Enter keeps the current one) and save them
//!   montaj-keys status     show which keys are set (never the keys)
//!   montaj-keys check      send a one-line test to every provider that has a key
//!   montaj-keys provider claude|openai|gemini   choose the text model for decisions
//!
//! Keys are saved to `<data dir>/montaj/keys.json` (owner-only); the data dir is FilmCraft's
//! (`FILMCRAFT_DATA_DIR`, else the platform default).

use std::io::{BufRead, Write};
use std::path::PathBuf;

use montaj_ai::keys::{Keys, Provider};
use montaj_ai::llm::{self, Claude, Gemini, Llm, OpenAi, Request};

fn data_dir() -> Option<PathBuf> {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(d) = env("FILMCRAFT_DATA_DIR") {
        return Some(d);
    }
    if cfg!(target_os = "macos") {
        return env("HOME").map(|h| h.join("Library/Application Support/FilmCraft"));
    }
    if cfg!(windows) {
        return env("APPDATA").map(|h| h.join("FilmCraft"));
    }
    env("XDG_DATA_HOME").map(|d| d.join("filmcraft")).or_else(|| env("HOME").map(|h| h.join(".local/share/filmcraft")))
}

fn ask(label: &str, current: &Option<String>) -> Option<String> {
    let state = if current.is_some() { "set — Enter keeps it" } else { "not set — Enter skips" };
    print!("{label} key ({state}): ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() {
        return current.clone();
    }
    let v = line.trim().to_string();
    if v.is_empty() { current.clone() } else { Some(v) }
}

fn status(k: &Keys) {
    let s = |v: &Option<String>| if v.is_some() { "set" } else { "not set" };
    println!("Anthropic (Claude): {}", s(&k.anthropic));
    println!("OpenAI:             {}", s(&k.openai));
    println!("Gemini:             {}", s(&k.gemini));
    println!("Decisions by:       {:?}", k.provider);
    println!("Speech-to-text:     {}", if k.openai.is_some() { "OpenAI (on)" } else { "off (needs an OpenAI key)" });
}

fn check(k: &Keys) {
    let mut models: Vec<Box<dyn Llm>> = Vec::new();
    if let Some(key) = &k.anthropic {
        models.push(Box::new(Claude { key: key.clone(), model: k.claude_model.clone().unwrap_or_else(|| llm::CLAUDE_MODEL.into()) }));
    }
    if let Some(key) = &k.openai {
        models.push(Box::new(OpenAi { key: key.clone(), model: k.openai_model.clone().unwrap_or_else(|| llm::OPENAI_MODEL.into()) }));
    }
    if let Some(key) = &k.gemini {
        models.push(Box::new(Gemini { key: key.clone(), model: k.gemini_model.clone().unwrap_or_else(|| llm::GEMINI_MODEL.into()) }));
    }
    if models.is_empty() {
        println!("no keys set — run montaj-keys first");
    }
    for m in models {
        let mut r = Request::new("Answer in one short line.", "قول: تمام");
        r.max_tokens = 2000;
        r.effort = llm::Effort::Low;
        match m.complete(&r) {
            Ok(t) => println!("{}: OK — {}", m.id(), t.trim()),
            Err(e) => println!("{}: FAILED — {e}", m.id()),
        }
    }
}

fn main() {
    let Some(dir) = data_dir() else {
        eprintln!("no data directory (set FILMCRAFT_DATA_DIR)");
        std::process::exit(2);
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut k = Keys::load(Some(&dir));
    match args.first().map(String::as_str) {
        Some("status") => status(&k),
        Some("check") => check(&k),
        Some("provider") => {
            k.provider = match args.get(1).map(String::as_str) {
                Some("claude") => Provider::Claude,
                Some("openai") => Provider::OpenAi,
                Some("gemini") => Provider::Gemini,
                _ => {
                    eprintln!("usage: montaj-keys provider claude|openai|gemini");
                    std::process::exit(2);
                }
            };
            if let Err(e) = k.save(&dir) {
                eprintln!("could not save: {e}");
                std::process::exit(1);
            }
            status(&k);
        }
        None => {
            println!("Montaj AI keys — saved to {}", montaj_ai::keys::path(&dir).display());
            k.anthropic = ask("Anthropic (Claude)", &k.anthropic);
            k.openai = ask("OpenAI", &k.openai);
            k.gemini = ask("Gemini", &k.gemini);
            if let Err(e) = k.save(&dir) {
                eprintln!("could not save: {e}");
                std::process::exit(1);
            }
            println!();
            status(&k);
        }
        Some(other) => {
            eprintln!("unknown command `{other}` (use: status, check, provider)");
            std::process::exit(2);
        }
    }
}
