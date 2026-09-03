@echo off
setlocal

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install it with: winget install --id=astral-sh.uv -e
    exit /b 1
)

if not exist "%~dp0dist" (
    echo No dist folder found. Run build_package.cmd first.
    exit /b 1
)

cd /d "%~dp0."
echo PyPI upload requires UV_PUBLISH_TOKEN.
echo Set it with: set UV_PUBLISH_TOKEN=pypi-your-token
uv publish
endlocal
