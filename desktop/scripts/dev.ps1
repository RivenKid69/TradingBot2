<#
.SYNOPSIS
    Fast dev loop (Windows): run the Python backend directly + Tauri dev shell.

.DESCRIPTION
    Starts the Python sidecar entry on a fixed port (no PyInstaller needed), then
    launches `cargo tauri dev` with RIVEN_DEV_URL so the shell connects to it
    instead of spawning a bundled binary. Edit index.html / app.py and just
    restart this script.

.EXAMPLE
    pwsh desktop/scripts/dev.ps1
#>
param([int]$Port = 8002)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

Write-Host "==> Starting Python backend on 127.0.0.1:$Port"
$backend = Start-Process -PassThru -NoNewWindow -FilePath $Python `
    -ArgumentList 'desktop_backend.py', '--port', "$Port"

try {
    $env:RIVEN_DEV_URL = "http://127.0.0.1:$Port"
    Set-Location (Join-Path $RepoRoot 'desktop/src-tauri')
    Write-Host "==> Launching Tauri dev shell (RIVEN_DEV_URL=$($env:RIVEN_DEV_URL))"
    cargo tauri dev
}
finally {
    if ($backend -and -not $backend.HasExited) {
        Write-Host "==> Stopping backend (PID $($backend.Id))"
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}
