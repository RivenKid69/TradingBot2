#!/usr/bin/env bash
#
# Build the RivenQuant Python sidecar (macOS/Linux) and place it where Tauri
# expects. Runs PyInstaller against packaging/riven_backend.spec, then copies the
# produced binary into desktop/src-tauri/binaries/ with the Rust target-triple
# suffix required by Tauri's sidecar (externalBin) mechanism.
#
# Usage:
#   desktop/scripts/build_sidecar.sh [trader|research]
#
# On macOS the triple is derived from the current arch (Apple Silicon ->
# aarch64-apple-darwin, Intel -> x86_64-apple-darwin). Build on each arch (or in
# CI) to ship both.
set -euo pipefail

PROFILE="${1:-trader}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    PYTHON="$(command -v python3)"
fi

export RIVEN_BUILD_PROFILE="${PROFILE}"
echo "==> Building sidecar (profile=${PROFILE}) in ${REPO_ROOT}"

"${PYTHON}" -m PyInstaller packaging/riven_backend.spec --noconfirm \
    --distpath dist --workpath build/pyinstaller

# Determine the Rust host target triple (fallback by uname).
TRIPLE=""
if command -v rustc >/dev/null 2>&1; then
    TRIPLE="$(rustc -Vv | awk '/^host:/ {print $2}')"
fi
if [ -z "${TRIPLE}" ]; then
    case "$(uname -s)-$(uname -m)" in
        Darwin-arm64)  TRIPLE="aarch64-apple-darwin" ;;
        Darwin-x86_64) TRIPLE="x86_64-apple-darwin" ;;
        Linux-x86_64)  TRIPLE="x86_64-unknown-linux-gnu" ;;
        Linux-aarch64) TRIPLE="aarch64-unknown-linux-gnu" ;;
        *) echo "Unknown platform; set TRIPLE manually" >&2; exit 1 ;;
    esac
fi

SRC="${REPO_ROOT}/dist/riven-backend"
[ -f "${SRC}" ] || { echo "Expected build output not found: ${SRC}" >&2; exit 1; }

BIN_DIR="${REPO_ROOT}/desktop/src-tauri/binaries"
mkdir -p "${BIN_DIR}"
DST="${BIN_DIR}/riven-backend-${TRIPLE}"
cp -f "${SRC}" "${DST}"
chmod +x "${DST}"

echo "==> Sidecar ready: ${DST}"
