@echo off
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build-and-run.ps1"
set "build_exit=%errorlevel%"
if not "%build_exit%"=="0" pause
exit /b %build_exit%
