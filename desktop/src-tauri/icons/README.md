# App icons

Tauri requires platform icon files here. Generate them once from a single square
PNG (≥1024×1024) — the command writes every size/format Tauri needs
(`32x32.png`, `128x128.png`, `128x128@2x.png`, `icon.icns`, `icon.ico`, …):

```bash
# from desktop/src-tauri/
cargo tauri icon path/to/logo.png
# or, if using the npm CLI:
npm exec tauri icon path/to/logo.png
```

The generated icons are git-ignored (see ../.gitignore); only this README is
tracked. The bundle will not build until icons exist.
