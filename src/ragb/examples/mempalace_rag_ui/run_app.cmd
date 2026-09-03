@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Could not find the project virtual environment. Run setup.cmd first.
    exit /b 1
)

if not exist "%~dp0.env" (
    copy "%~dp0.env.example" "%~dp0.env" >nul
    echo Created .env from .env.example. Add a provider key and run this again.
    exit /b 1
)

if not exist "%~dp0frontend\dist\index.html" (
    echo React frontend build not found.
    echo Run build_frontend.cmd first, then run_app.cmd again.
    exit /b 1
)

cd /d "%~dp0"
"%PY%" app.py --no-browser %*
endlocal
