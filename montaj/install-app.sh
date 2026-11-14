#!/bin/zsh
# Montaj: put /Applications/Montaj.app together from this checkout's release build
# (target/release/filmcraft + filmcraft-cli). Ad-hoc signed, for this Mac only.
# The CLI sits next to the app binary, so other apps (LiteRec) can find it at
# /Applications/Montaj.app/Contents/MacOS/filmcraft-cli.
set -eu
ROOT="${0:A:h:h}"
APP="${MONTAJ_APP:-/Applications/Montaj.app}"
BIN="$ROOT/target/release"
[[ -x "$BIN/filmcraft" && -x "$BIN/filmcraft-cli" ]] || { echo "build first: cargo build --release -p filmcraft -p filmcraft-cli" >&2; exit 1; }
VERSION=$(awk '/^\[/{p=($0=="[workspace.package]");next} p&&$1=="version"{gsub(/[" ]/,"",$3);print $3;exit}' "$ROOT/Cargo.toml")
TMP="$(mktemp -d)/Montaj.app"
mkdir -p "$TMP/Contents/MacOS" "$TMP/Contents/Resources"
cp "$BIN/filmcraft" "$TMP/Contents/MacOS/FilmCraft"
cp "$BIN/filmcraft-cli" "$TMP/Contents/MacOS/filmcraft-cli"
# Montaj's own icon: white in Light mode, black in Dark mode (Icon Composer .icon → Assets.car);
# montaj.icns is the fallback for systems that can't read it
cp "$ROOT/montaj/icon/montaj.icns" "$TMP/Contents/Resources/FilmCraft.icns"
xcrun actool "$ROOT/montaj/icon/Montaj.icon" --compile "$TMP/Contents/Resources" --platform macosx \
    --minimum-deployment-target 11.0 --app-icon Montaj --output-partial-info-plist "$TMP/icon.plist" >/dev/null
rm -f "$TMP/icon.plist"
cp "$ROOT/NOTICE" "$TMP/Contents/Resources/"
sed -e "s/@VERSION@/$VERSION/g" -e "s/@SHORT_VERSION@/${VERSION%%-*}/g" \
    -e "s/@BUILD_SHA@/$(git -C "$ROOT" rev-parse --short HEAD)/g" \
    "$ROOT/packaging/macos/Info.plist.in" > "$TMP/Contents/Info.plist"
plutil -replace CFBundleName -string Montaj "$TMP/Contents/Info.plist"
plutil -replace CFBundleDisplayName -string Montaj "$TMP/Contents/Info.plist" 2>/dev/null || plutil -insert CFBundleDisplayName -string Montaj "$TMP/Contents/Info.plist"
plutil -replace CFBundleIdentifier -string com.el3rrab.montaj "$TMP/Contents/Info.plist"
plutil -replace CFBundleIconName -string Montaj "$TMP/Contents/Info.plist"
printf 'APPL????' > "$TMP/Contents/PkgInfo"
codesign --force --sign - --timestamp=none "$TMP/Contents/MacOS/filmcraft-cli"
codesign --force --sign - --timestamp=none --entitlements "$ROOT/packaging/macos/entitlements.plist" "$TMP/Contents/MacOS/FilmCraft"
codesign --force --sign - --timestamp=none "$TMP"
rm -rf "$APP"
mv "$TMP" "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP"
echo "installed $APP ($VERSION)"
