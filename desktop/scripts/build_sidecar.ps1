<#
.SYNOPSIS
    Build the RivenQuant Python sidecar (Windows) and place it where Tauri expects.

.DESCRIPTION
    Runs PyInstaller against packaging/riven_backend.spec, then copies the produced
    executable into desktop/src-tauri/binaries/ with the Rust target-triple suffix
    that Tauri's `externalBin` / sidecar mechanism requires.

.PARAMETER Profile
    'trader' (default, slim) or 'research' (full, includes torch/training).

.EXAMPLE
    pwsh desktop/scripts/build_sidecar.ps1 -Profile trader
#>
param(
    [ValidateSet('trader', 'research')]
    [string]$Profile = 'trader'
)

$ErrorActionPreference = 'Stop'

# Repo root = two levels up from this script (desktop/scripts/).
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$env:RIVEN_BUILD_PROFILE = $Profile
Write-Host "==> Building sidecar (profile=$Profile) in $RepoRoot"

& $Python -m PyInstaller packaging/riven_backend.spec --noconfirm --distpath dist --workpath build/pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Determine the Rust host target triple (fallback to the common x64 MSVC triple).
$triple = 'x86_64-pc-windows-msvc'
try {
    $rv = (rustc -Vv 2>$null) | Select-String '^host:'
    if ($rv) { $triple = ($rv -replace 'host:\s*', '').Trim() }
} catch { }

$srcExe = Join-Path $RepoRoot 'dist/riven-backend.exe'
if (-not (Test-Path $srcExe)) { throw "Expected build output not found: $srcExe" }

$binDir = Join-Path $RepoRoot 'desktop/src-tauri/binaries'
New-Item -ItemType Directory -Force -Path $binDir | Out-Null
$dstExe = Join-Path $binDir "riven-backend-$triple.exe"
Copy-Item -Force $srcExe $dstExe

Write-Host "==> Sidecar ready: $dstExe"
