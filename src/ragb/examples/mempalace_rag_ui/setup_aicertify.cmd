@echo off
setlocal

set "PROJECT_ROOT=%~dp0..\..\..\.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "AICERTIFY_ENV=%PROJECT_ROOT%\.aicertify-venv"
set "REQ=%PROJECT_ROOT%\requirements-aicertify.txt"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install it with: winget install --id=astral-sh.uv -e
    exit /b 1
)

if not exist "%AICERTIFY_ENV%\Scripts\python.exe" (
    echo Creating the isolated AICertify environment...
    uv venv "%AICERTIFY_ENV%" --python 3.12
    if errorlevel 1 exit /b 1
)

echo Installing AICertify and its compatible dependency stack...
uv pip install --python "%AICERTIFY_ENV%\Scripts\python.exe" --upgrade -r "%REQ%"
if errorlevel 1 exit /b 1

echo.
echo AICertify setup complete.
echo Add this to src\ragb\examples\mempalace_rag_ui\.env:
echo AICERTIFY_PYTHON=%AICERTIFY_ENV%\Scripts\python.exe
echo AICERTIFY_CAPTURE_INTERACTIONS=true
endlocal
