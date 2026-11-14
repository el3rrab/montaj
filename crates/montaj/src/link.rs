//! LiteRec link: every Montaj window can be driven by LiteRec, also one opened from the Dock or
//! Finder (LiteRec's "Link to Montaj" button).
//!
//! FilmCraft's control server only starts with `--control <port>` + `FILMCRAFT_CONTROL_TOKEN`. When
//! Montaj starts without them, [`prepare`] picks a free loopback port and a fresh random token for
//! this process only (kept in memory, never on a command line or in the environment). [`publish`] then
//! writes `<FilmCraft data>/montaj/link-<pid>.json` (`{port, token, pid}`, readable by this user
//! only) so LiteRec can find the window; files of Montaj processes that are gone are removed.

use std::path::PathBuf;

fn dir() -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var_os("HOME")?);
    Some(if cfg!(target_os = "macos") { home.join("Library/Application Support/FilmCraft/montaj") } else { home.join(".local/share/FilmCraft/montaj") })
}

fn random_hex() -> Option<String> {
    use std::io::Read;
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom").ok()?.read_exact(&mut b).ok()?;
    Some(b.iter().map(|x| format!("{x:02x}")).collect())
}

static TOKEN: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// The private token [`prepare`] made (the control server takes it when the environment has none).
pub fn token() -> Option<String> {
    TOKEN.get().cloned()
}

/// Before the arguments are read: no control asked for → a private one for LiteRec. Returns the
/// port to listen on (the window then still opens in front, as for any normal start).
pub fn prepare(args: &[std::ffi::OsString]) -> Option<u16> {
    let asked = std::env::var_os("FILMCRAFT_CONTROL_PORT").is_some_and(|p| !p.is_empty()) || args.iter().any(|a| a == "--control" || a.to_string_lossy().starts_with("--control="));
    let helper = args.iter().any(|a| a.to_string_lossy().starts_with("--test-graphics-start") || a == "--finish-update" || a == "--help" || a == "-h" || a == "--version");
    if asked || helper || !cfg!(unix) {
        return None;
    }
    let token = random_hex()?;
    let port = std::net::TcpListener::bind("127.0.0.1:0").ok().and_then(|l| l.local_addr().ok()).map(|a| a.port())?;
    TOKEN.set(token).ok()?;
    Some(port)
}

/// After the arguments are read: tell LiteRec where this window listens (any control port, also
/// one LiteRec itself asked for, so it can find the window again after its panel was closed).
pub fn publish(port: Option<u16>) {
    let (Some(port), Some(token), Some(dir)) = (port, std::env::var("FILMCRAFT_CONTROL_TOKEN").ok().or_else(token), dir()) else { return };
    let _ = std::fs::create_dir_all(&dir);
    // files of Montaj windows that are gone
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(pid) = name.strip_prefix("link-").and_then(|n| n.strip_suffix(".json")).and_then(|n| n.parse::<i32>().ok()) {
                if !alive(pid) {
                    let _ = std::fs::remove_file(e.path());
                }
            }
        }
    }
    let pid = std::process::id();
    let path = dir.join(format!("link-{pid}.json"));
    let body = serde_json::json!({"port": port, "token": token, "pid": pid}).to_string();
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        if let Ok(mut f) = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path) {
            let _ = f.write_all(body.as_bytes());
        }
    }
}

/// The process exists (`kill -0`).
fn alive(pid: i32) -> bool {
    std::process::Command::new("/bin/kill").args(["-0", &pid.to_string()]).stderr(std::process::Stdio::null()).status().is_ok_and(|s| s.success())
}
