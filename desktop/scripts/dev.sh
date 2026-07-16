#!/usr/bin/env bash
#
# Fast dev loop (macOS/Linux): run the Python backend directly + Tauri dev shell.
# Starts the sidecar entry on a fixed port (no PyInstaller), then runs
# `cargo tauri dev` with RIVEN_DEV_URL so the shell connects to it.
#
# Usage: desktop/scripts/dev.sh [PORT]
set -euo pipefail

PORT="${1:-8002}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python3)"
fi

echo "==> Starting Python backend on 127.0.0.1:${PORT}"
"${PYTHON}" desktop_backend.py --port "${PORT}" &
BACKEND_PID=$!

cleanup() {
  echo "==> Stopping backend (PID ${BACKEND_PID})"
  kill "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export RIVEN_DEV_URL="http://127.0.0.1:${PORT}"
cd "${REPO_ROOT}/desktop/src-tauri"
echo "==> Launching Tauri dev shell (RIVEN_DEV_URL=${RIVEN_DEV_URL})"
cargo tauri dev
