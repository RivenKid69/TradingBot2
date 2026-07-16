@echo off
REM Install the Tauri v2 CLI (cargo subcommand) inside the MSVC build environment.
call "%~dp0_msvc_env.bat" || exit /b 1
cargo install tauri-cli --locked --version "^2"
