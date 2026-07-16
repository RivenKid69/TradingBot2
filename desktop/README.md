# RivenQuant Desktop 2.6.0 (Tauri v2 + Python sidecar)

Production desktop packaging of the RivenQuant platform for **Windows 11** and
**macOS**. The desktop app reproduces the current MVP 1:1 — it renders the exact
same `index.html` served by the existing FastAPI backend (`app:api`); only the
window chrome changes.

This layout is also the correct home for a **fully real (non-demo) CCEA**: the
desktop is, by definition, the client's machine = the CCEA *Agent zone*. Secrets
live in the OS keychain on the device and orders are created locally — no servers.

> **CCEA is implemented and runs in the desktop.** On launch the app starts a real
> local CCEA stack (cloud control plane on SQLite + Agent daemon with keychain
> vault, policy firewall, kill switch, paper broker) over loopback, the agent
> enrolls/heartbeats, and the UI shows live status (`GET /api/ccea/status`, Home
> card). Controlled by `RIVEN_ENABLE_CCEA` (default `1`). Full design + verification:
> [docs/CCEA_DESKTOP.md](../docs/CCEA_DESKTOP.md).

---

## Architecture

```
┌──────────────── RivenQuant.exe / .app (Tauri shell) ────────────────┐
│  System WebView (WebView2 on Win11 / WKWebView on macOS)             │
│        renders  http://127.0.0.1:<port>/   (the same index.html)     │
│           │                                                          │
│           │  fetch /api/*   (same-origin, window.RIVEN_API_BASE='')  │
│           ▼                                                          │
│  Python sidecar  ── desktop_backend.py ──► uvicorn app:api          │
│     • provisions a writable runtime root (app-data dir)              │
│     • picks a loopback port, prints RIVEN_PORT=<n> on stdout         │
│     • (CCEA) vault on OS keychain, policy firewall, OMS, brokers     │
└─────────────────────────────────────────────────────────────────────┘
```

**Boundary & lifecycle**
- The Rust shell owns no business logic. It spawns the sidecar, reads the
  `RIVEN_PORT=` handshake, waits for the port to accept connections, opens the
  main window, and asks `/api/desktop/shutdown` to flush the Agent/SQLite stores
  before terminating the sidecar on exit.
- `window.RIVEN_API_BASE=''` is injected *before* page scripts run, so every
  `/api/*` call is same-origin against the sidecar on whatever port it chose.
- Auth stays in `loopback` mode (default) — identical behaviour to the MVP.

**Key files**
| Path | Role |
|------|------|
| `../desktop_backend.py` | Python sidecar entry (runtime-root, port handshake, uvicorn) |
| `../packaging/riven_backend.spec` | PyInstaller spec (profiles: `trader` / `research`) |
| `src-tauri/src/lib.rs` | Shell logic (spawn sidecar, port wait, window, lifecycle) |
| `src-tauri/tauri.conf.json` | Tauri v2 config (externalBin, bundle, entitlements) |
| `src-tauri/capabilities/default.json` | Webview capabilities (no IPC exposed to page) |
| `src-tauri/Entitlements.plist` | macOS hardened-runtime entitlements (Python/JIT) |
| `app-dist/index.html` | Splash/loading screen (Tauri `frontendDist`) |
| `../web_assets/` | Bundled Tailwind, Chart.js, fonts, Font Awesome and Monaco (offline) |
| `scripts/` | Build + dev runners (Win `.ps1`, macOS/Linux `.sh`) |

The only changes to the existing app were minimal and backward-compatible:
1. `app.py` — the Streamlit wrapper is now guarded so plain import (sidecar /
   `uvicorn app:api`) does not run Streamlit; `streamlit run app.py` still works.
2. `index.html` — `getApiBase()` honours an injected `window.RIVEN_API_BASE`
   first; behaviour is unchanged when it is unset.

---

## Prerequisites

Install on the build machine (per OS):

- **Rust** (stable): https://rustup.rs
- **Tauri CLI v2**: `cargo install tauri-cli --version "^2"` (gives `cargo tauri`)
- **Python 3.12** with the project deps installed, plus **PyInstaller**:
  `pip install -r ../requirements.txt pyinstaller` (uvicorn must be available)
- **Windows**: WebView2 runtime (preinstalled on Win11), MSVC build tools.
- **macOS**: Xcode command-line tools; for distribution an **Apple Developer**
  account (Developer ID + notarization).

> macOS apps can only be built/signed on macOS (use a Mac or a macOS CI runner).

---

## Develop (fast loop, no packaging)

Runs the backend from source and a hot Tauri shell that connects to it:

```bash
# Windows
pwsh desktop/scripts/dev.ps1

# macOS / Linux
desktop/scripts/dev.sh
```

Mechanism: the script prefers the project's `.venv` Python, starts
`desktop_backend.py --port 8002`, and runs
`cargo tauri dev` with `RIVEN_DEV_URL=http://127.0.0.1:8002`, so the shell skips
the bundled sidecar and points at your live backend. Edit `index.html`/`app.py`,
restart the script.

---

## Build a release

### 1. Build the Python sidecar

```bash
# Windows  (-> src-tauri/binaries/riven-backend-x86_64-pc-windows-msvc.exe)
pwsh desktop/scripts/build_sidecar.ps1 -Profile trader

# macOS    (-> src-tauri/binaries/riven-backend-<arch>-apple-darwin)
desktop/scripts/build_sidecar.sh trader
```

Profiles: `trader` (slim — excludes torch/RL training) or `research` (full).

### 2. Generate icons (once)

```bash
cd desktop/src-tauri && cargo tauri icon path/to/logo.png
```

### 3. Bundle the desktop app

```bash
cd desktop/src-tauri
cargo tauri build
```

Outputs: Windows → NSIS/MSI installer; macOS → `.app` + `.dmg`.

---

## Code signing & notarization

### Windows
Sign the installer with your Authenticode certificate (configure
`bundle.windows.certificateThumbprint` / sign the produced installer), or use
`signtool`. Unsigned installers trigger SmartScreen warnings.

### macOS (required for distribution)
Gatekeeper rejects unsigned/un-notarized apps. The `.app` **and the embedded
sidecar** must be signed (hardened runtime) and notarized. The provided
`Entitlements.plist` grants the JIT / executable-memory entitlements Python
needs. Typical env for `cargo tauri build`:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Co (TEAMID)"
export APPLE_ID="you@example.com"
export APPLE_PASSWORD="app-specific-password"
export APPLE_TEAM_ID="TEAMID"
cargo tauri build            # Tauri signs + notarizes when these are set
```

Distribute the notarized `.dmg` (Developer ID), **not** via the Mac App Store —
the App Store sandbox forbids spawning a bundled executable (the sidecar).

---

## Cross-platform notes

- **Rendering**: identical to the MVP on Win11 (WebView2 = Chromium). On macOS
  the engine is WebKit (WKWebView) — verify the UI once and tweak CSS if needed.
- **Offline UI**: all rendering/editor dependencies are served from
  `/assets/*`; no CDN is required for the application shell.
- **Sidecar arch**: ship `aarch64-apple-darwin` (Apple Silicon) and/or
  `x86_64-apple-darwin` (Intel); build on each arch or in CI.
- **Writable data**: runtime files (logs/state/reports/configs) live under the
  per-user app-data dir (`%APPDATA%\RivenQuant`, `~/Library/Application
  Support/RivenQuant`). Override with `RIVEN_DATA_DIR`.
