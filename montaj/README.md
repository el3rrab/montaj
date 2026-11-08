# Montaj

Montaj is our own build of [FilmCraft](https://github.com/storytold/filmcraft). The code keeps
FilmCraft's internal names on purpose, so every upstream release merges in cleanly. Everything
that is ours alone lives in this folder or in files upstream does not have.

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
