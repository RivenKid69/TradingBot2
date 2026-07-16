# Sidecar binaries

This directory holds the bundled Python backend, produced by PyInstaller and
named with the Rust **target triple** that Tauri's `externalBin` mechanism
requires. The build scripts populate it automatically:

| Platform | Expected file |
|----------|---------------|
| Windows x64 | `riven-backend-x86_64-pc-windows-msvc.exe` |
| macOS Apple Silicon | `riven-backend-aarch64-apple-darwin` |
| macOS Intel | `riven-backend-x86_64-apple-darwin` |

Generate with:

```bash
# Windows (PowerShell)
pwsh desktop/scripts/build_sidecar.ps1 -Profile trader

# macOS / Linux
desktop/scripts/build_sidecar.sh trader
```

These artifacts are build outputs and are **git-ignored** (see ../.gitignore).
Find your exact triple with `rustc -Vv | grep host`.
