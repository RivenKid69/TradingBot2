@echo off
REM Run cargo inside the MSVC build environment. Usage: cargo_msvc.bat <cargo args...>
call "%~dp0_msvc_env.bat" || exit /b 1
cargo %*
