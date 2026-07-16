#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RivenQuant desktop backend (Python sidecar).

This is the entrypoint that the Tauri desktop shell launches. It boots the
existing FastAPI application (``app:api``) on a local loopback port and prints a
small handshake to stdout so the Rust shell knows where to point the webview:

    RIVEN_HOST=127.0.0.1
    RIVEN_PORT=<chosen port>
    RIVEN_READY=1            (emitted once Uvicorn has started serving)

Design notes (why this is more than a thin wrapper):

* ``app.py`` uses *relative* paths for both reading (``index.html``, ``configs/``)
  and writing (``logs/``, ``state/``, ``reports/`` ...). A packaged desktop app
  is installed read-only (e.g. Program Files / *.app), so we cannot write next to
  the executable. This module therefore provisions a writable **runtime root**
  under the per-user app-data directory, syncs the bundled read-only assets into
  it, ``chdir``s there, and only then imports ``app``. All of ``app.py``'s
  relative paths then resolve under a writable location with zero edits to it.

* In development (not frozen by PyInstaller) the repository directory is already
  writable, so we use it directly for a fast edit/reload loop.

* ``SEASONALITY_API_TOKEN`` is required by ``app.py`` at import time; we generate
  an ephemeral one if absent. Auth stays in the default ``loopback`` mode, so the
  webview (a 127.0.0.1 client) works exactly like the current MVP.
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import socket
import sys
from pathlib import Path


APP_NAME = "RivenQuant"

# Read-only assets that must exist next to the working directory at runtime.
# Files are copied if missing; index.html is always refreshed to match the build.
_ASSET_FILES_REFRESH = ("index.html",)
# Directories whose *missing* files are seeded from the bundle (user edits kept).
_ASSET_DIRS_SEED = ("configs", "web_assets")
# Writable directories the backend expects to exist.
_WRITABLE_DIRS = ("logs", "state", ".run", "models", "reports", "data")


def _is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def _resource_root() -> Path:
    """Directory containing the bundled read-only assets."""
    if _is_frozen():
        # PyInstaller unpacks data files here (onefile) or alongside (onedir).
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent


def _default_data_dir() -> Path:
    """Per-OS writable application-data directory."""
    override = os.environ.get("RIVEN_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def _provision_runtime_root() -> Path:
    """Choose the working directory and ensure assets + writable dirs exist."""
    resource_root = _resource_root()

    # Dev (not frozen): use the repo directly unless an explicit data dir is set.
    if not _is_frozen() and not os.environ.get("RIVEN_DATA_DIR"):
        runtime_root = resource_root
        for d in _WRITABLE_DIRS:
            (runtime_root / d).mkdir(parents=True, exist_ok=True)
        return runtime_root

    runtime_root = _default_data_dir()
    runtime_root.mkdir(parents=True, exist_ok=True)

    for d in _WRITABLE_DIRS:
        (runtime_root / d).mkdir(parents=True, exist_ok=True)

    # Always refresh shell assets (UI only, no user data).
    for name in _ASSET_FILES_REFRESH:
        src = resource_root / name
        if src.is_file():
            shutil.copy2(src, runtime_root / name)

    # Seed default config files without clobbering user-modified ones.
    for d in _ASSET_DIRS_SEED:
        src_dir = resource_root / d
        if not src_dir.is_dir():
            continue
        for src in src_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(src_dir)
            dst = runtime_root / d / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    return runtime_root


def _pick_port(preferred: int | None) -> int:
    """Return an available loopback port, preferring ``preferred`` then ephemeral."""
    candidates = [preferred] if preferred else []
    candidates.append(0)  # 0 => OS-assigned ephemeral port
    for cand in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", cand))
                return int(s.getsockname()[1])
        except OSError:
            continue
    raise RuntimeError("No available loopback port")


def _emit(line: str) -> None:
    """Write a handshake line to stdout and flush so the shell sees it promptly."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main() -> int:
    # A frozen sidecar is also the Python runtime for bundled background jobs.
    # Dispatch those jobs before parsing the HTTP-server arguments.
    from desktop_job_runtime import dispatch_worker

    worker_result = dispatch_worker()
    if worker_result is not None:
        return worker_result

    parser = argparse.ArgumentParser(description="RivenQuant desktop backend sidecar")
    parser.add_argument("--host", default=os.environ.get("RIVEN_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ["RIVEN_PORT"]) if os.environ.get("RIVEN_PORT") else None,
        help="Preferred port (default: 8002 if free, else ephemeral).",
    )
    args = parser.parse_args()

    # 1) Writable working directory must be set BEFORE importing app.
    runtime_root = _provision_runtime_root()
    os.chdir(runtime_root)

    # 2) Satisfy app.py's import-time auth requirement; keep default loopback mode.
    os.environ.setdefault("SEASONALITY_API_TOKEN", secrets.token_urlsafe(32))
    os.environ.setdefault("RIVEN_API_AUTH_MODE", "loopback")
    # The desktop IS the client environment (CCEA Agent zone): run the real CCEA
    # stack (local control-plane + Agent daemon) by default. Set to "0" to disable.
    os.environ.setdefault("RIVEN_ENABLE_CCEA", "1")
    os.environ.setdefault("RIVEN_DATA_DIR", str(runtime_root))

    # 3) Pick the port and hand it to the shell before the (blocking) server starts.
    host = args.host
    port = _pick_port(args.port if args.port else 8002)
    _emit(f"RIVEN_HOST={host}")
    _emit(f"RIVEN_PORT={port}")

    # 4) Import the FastAPI app (Streamlit wrapper stays dormant on import) and run.
    import uvicorn  # local import keeps startup errors close to use
    from app import api  # noqa: WPS433 — intentional late import after chdir/env

    @api.on_event("startup")
    async def _announce_ready() -> None:  # pragma: no cover - runtime hook
        _emit("RIVEN_READY=1")

    config = uvicorn.Config(api, host=host, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    # The loopback desktop shutdown endpoint flips ``should_exit`` after it has
    # flushed CCEA.  This lets PyInstaller one-file parent/child processes exit
    # naturally instead of leaving the child alive after a forced parent kill.
    api.state.desktop_server = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
