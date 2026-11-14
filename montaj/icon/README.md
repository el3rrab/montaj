# Montaj icon

The owner's logo, exactly as provided (MONTAGE included): black on a white tile in Light mode,
white on a black tile in Dark mode.

- `Montaj.icon`: the Icon Composer icon (`icon.json` + `Assets/light.png`, `Assets/dark.png`).
  `montaj/install-app.sh` compiles it with `actool` into the bundle's `Assets.car` and sets
  `CFBundleIconName`, so Finder and the Dock follow the system appearance.
- `montaj-light.svg`, `montaj-dark.svg`: the sources of the two layers (1024×1024, transparent
  around the logo; the system draws the tile).
- `montaj.icns`: fallback for systems that can't read the `.icon` (made by `actool`).
- `apps/filmcraft/src/main.rs` gives eframe no icon on macOS, so the Dock keeps the bundle's.
- The logo file was `magnific_cinemalogovectorrollfilmvectorwhitebackground…svg`; its licence is
  the owner's to confirm before any public use. FilmCraft's icon in `assets/app-icon/` is untouched.
