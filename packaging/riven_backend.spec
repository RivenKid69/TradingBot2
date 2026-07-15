# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the RivenQuant desktop backend (Python sidecar).

Builds a single-file executable `riven-backend` that the Tauri shell ships as a
bundled sidecar. Run from the repository root:

    pyinstaller packaging/riven_backend.spec --noconfirm

Profiles (env RIVEN_BUILD_PROFILE):
    trader   (default) — excludes torch / RL training stack → small, fast binary
    research            — includes everything (large; needed for in-app training)

Output: dist/riven-backend[.exe]; the build scripts then copy it into
desktop/src-tauri/binaries/ with the Tauri target-triple suffix.
"""

import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

from desktop_job_runtime import WORKER_MODULES

# SPECPATH is injected by PyInstaller = absolute dir of this spec (packaging/).
REPO_ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))
PROFILE = os.environ.get("RIVEN_BUILD_PROFILE", "trader").strip().lower()


def _repo(*parts):
    return os.path.join(REPO_ROOT, *parts)


# --- Project packages that must be fully collected (dynamic imports abound) ---
_PROJECT_PKGS = [
    "packages", "services", "adapters", "signals", "loaders",
    "strategies", "lob", "research", "wrappers", "optimizers", "ccea",
]
hiddenimports = []
for _pkg in _PROJECT_PKGS:
    try:
        hiddenimports += collect_submodules(_pkg)
    except Exception:
        pass

# Web/server stack often needs explicit hints.
hiddenimports += [
    "uvicorn", "uvicorn.logging", "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on", "fastapi", "starlette", "pydantic", "anyio",
    "httpx", "yaml",
]
hiddenimports += sorted(WORKER_MODULES)
# CCEA runtime: local control-plane (async SQLite) + Agent daemon + cloud client.
hiddenimports += [
    "ccea",
    "aiosqlite", "greenlet", "email_validator",
    "sqlalchemy", "sqlalchemy.ext.asyncio", "sqlalchemy.dialects.sqlite",
    "cryptography", "cryptography.hazmat.primitives.asymmetric.ed25519",
]
try:
    hiddenimports += collect_submodules("ccea")
    hiddenimports += collect_submodules("sqlalchemy.dialects")
except Exception:
    pass

# --- Bundled read-only assets (provisioned into the writable runtime root) ---
datas = [
    (_repo("index.html"), "."),
    (_repo("configs"), "configs"),
    (_repo("web_assets"), "web_assets"),
]
for _opt in ("data/universe",):
    if os.path.isdir(_repo(_opt)):
        datas.append((_repo(_opt), _opt))

# --- Always excluded: Streamlit (legacy wrapper only; UI is served by FastAPI) ---
excludes = ["streamlit", "altair", "pydeck"]

# --- Heavy/optional excludes for the slim "trader" profile ---
if PROFILE != "research":
    excludes += [
        "torch", "torchvision", "torchaudio",
        "stable_baselines3", "sb3_contrib", "optuna",
        "matplotlib", "tensorboard", "gymnasium",
    ]
else:
    # Research profile bundles the RL training stack. Python modules alone are
    # not enough: stable_baselines3 reads its `version.txt` data file at import
    # time, and gymnasium ships non-code package data too. Missing these made
    # packaged `run_train` crash before training started (audit L2-005).
    for _data_pkg in ("stable_baselines3", "sb3_contrib", "gymnasium"):
        try:
            datas += collect_data_files(_data_pkg)
        except Exception:
            pass

block_cipher = None

a = Analysis(
    [_repo("desktop_backend.py")],
    pathex=[REPO_ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="riven-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,            # console=True so the shell can read the stdout handshake
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,        # set via --target-arch on macOS for arm64/x86_64
    codesign_identity=None,  # signing handled by build scripts / notarization
    entitlements_file=None,
)
