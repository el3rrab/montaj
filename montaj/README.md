# Montaj

Montaj is our own build of [FilmCraft](https://github.com/storytold/filmcraft). The code keeps
FilmCraft's internal names on purpose, so every upstream release merges in cleanly. Everything
that is ours alone lives in this folder or in files upstream does not have.

## Our layer

Our work sits on top of FilmCraft and never rewrites it, so each FilmCraft release adds to ours
instead of fighting it:

- **Our code lives only in our own places:** `crates/montaj` (commands, cloud AI hookup),
  `crates/montaj-ai` (tools), `montaj/` (scripts, docs), `sync.sh` and
  `.github/workflows/montaj-upstream-sync.yml`.
- **FilmCraft files get only short hook lines**, each marked with a `// Montaj:` comment:
  `apps/filmcraft/src/main.rs` and `apps/filmcraft-cli/src/main.rs` (call `montaj::install()`,
  set the transcriber, no FilmCraft update prompt), their `Cargo.toml` (the `montaj` dependency),
  and two small extension points in `crates/engine` (`register_extra`, `register_poll`).
  A new feature goes behind these hooks, never as an edit inside FilmCraft code.
- **Never change the version in `Cargo.toml`** and don't resolve `Cargo.lock` by hand: both always
  come from FilmCraft (the sync takes theirs and adds our crates back).
- To list everything we changed in FilmCraft's files:
  `git diff $(git describe --tags --abbrev=0 --match 'v*') HEAD -- . ':!montaj' ':!crates/montaj*'`

## Staying up to date

- **On GitHub:** `.github/workflows/montaj-upstream-sync.yml` checks `storytold/filmcraft` every
  6 hours. A new release tag is merged into `main` and pushed. A conflict with our changes opens
  an issue instead, and nothing is pushed.
  It needs a repository secret `SYNC_TOKEN`: a fine-grained token for `el3rrab/montaj` with
  Contents, Workflows and Issues set to read and write.
- **On the Mac:** `update-local.sh` fast-forwards the checkout from `origin/main` and rebuilds,
  only when it is on `main` with no uncommitted changes. The launchd agent runs it every 6 hours:

  ```sh
  cp montaj/com.el3rrab.montaj.update.plist ~/Library/LaunchAgents/
  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.el3rrab.montaj.update.plist
  ```

  Log: `~/Library/Logs/montaj-update.log`.

## Remotes

- `origin`: `el3rrab/montaj` (ours; push here)
- `upstream`: `storytold/filmcraft` (fetch only)

FilmCraft is MIT OR Apache-2.0; keep `LICENSE-MIT`, `LICENSE-APACHE` and `NOTICE`.
