@echo off
setlocal

set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "BASE_DIR=%REPO_ROOT%\..\..\examples\mempalace_quivr"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install it with: winget install --id=astral-sh.uv -e
    exit /b 1
)

if "%UV_PUBLISH_TOKEN%"=="" (
    echo Set UV_PUBLISH_TOKEN in this CMD window before publishing.
    exit /b 1
)

if exist "%BASE_DIR%\pyproject.toml" (
    echo Publishing quivr-mempalace first...
    cd /d "%BASE_DIR%"
    uv publish
    if errorlevel 1 exit /b 1
) else (
    echo Local quivr-mempalace source was not found; publishing the UI package only.
)

echo Publishing quivr-mempalace-rag-ui...
cd /d "%~dp0."
uv publish
if errorlevel 1 exit /b 1

echo Both packages were published.
endlocal
