@echo off
setlocal

set "PROJECT_ROOT=%~dp0..\..\..\.."
for %%I in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~fI"
set "SOURCE_ROOT=%PROJECT_ROOT%\src\ragb"
set "PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv was not found. Install it with: winget install --id=astral-sh.uv -e
    exit /b 1
)

if not exist "%PY%" (
    echo Creating the project virtual environment with uv...
    uv venv "%PROJECT_ROOT%\.venv" --python 3.13
    if errorlevel 1 exit /b 1
)

for /f "tokens=2" %%V in ('"%PY%" --version 2^>nul') do set "PY_VERSION=%%V"
if not "%PY_VERSION:~0,4%"=="3.13" (
    echo This project requires Python 3.13. Found Python %PY_VERSION%.
    echo Remove the existing .venv and run setup.cmd again:
    echo rmdir /s /q "%PROJECT_ROOT%\.venv"
    exit /b 1
)

set "LOCAL_BASE=%PROJECT_ROOT%\examples\mempalace_quivr"
if exist "%LOCAL_BASE%\pyproject.toml" (
    echo Installing the local MemPalace package...
    uv pip install --python "%PY%" --upgrade -e "%LOCAL_BASE%[nvidia]"
    if errorlevel 1 exit /b 1
) else (
    echo Installing the published MemPalace package...
    uv pip install --python "%PY%" --upgrade "quivr-mempalace[nvidia]"
    if errorlevel 1 exit /b 1
)

echo Installing the lightweight Groq and NVIDIA provider dependencies...
uv pip install --python "%PY%" --upgrade "langchain-groq>=0.3.5,<1" "langchain-nvidia-ai-endpoints>=0.3.19,<1" "python-dotenv>=1.0"
if errorlevel 1 exit /b 1

echo Installing the local Quivr Core fork...
uv pip install --python "%PY%" --upgrade -e "%SOURCE_ROOT%\core"
if errorlevel 1 exit /b 1

echo Installing the React API server...
uv pip install --python "%PY%" --upgrade "fastapi>=0.115,<1" "python-multipart>=0.0.20,<1" "uvicorn[standard]>=0.34,<1"
if errorlevel 1 exit /b 1

echo Installing local document parsing and TTS dependencies...
uv pip install --python "%PY%" --upgrade "langchain-text-splitters>=0.3,<0.4" "pillow>=10,<13" "pymupdf>=1.24,<2" "pypdf>=5,<7" "pytesseract>=0.3.13,<1" "python-docx>=1.1,<2" "kokoro-onnx>=0.3,<1" "soundfile>=0.12,<1" "numpy>=1.26,<3"
if errorlevel 1 exit /b 1

echo Applying compatible dependency pins...
uv pip install --python "%PY%" --upgrade "langchain-core>=0.3.85,<0.4" "packaging>=23.2,<25"
if errorlevel 1 exit /b 1

echo Installing the combined UI package in editable mode...
uv pip install --python "%PY%" --upgrade --editable "%~dp0." --no-deps
if errorlevel 1 exit /b 1

if not exist "%~dp0.env" copy "%~dp0.env.example" "%~dp0.env" >nul

where npm >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Install Node.js 20 or newer, then run setup.cmd again.
    exit /b 1
)

echo Installing React frontend dependencies...
pushd "%~dp0frontend"
npm install
if errorlevel 1 (
    popd
    exit /b 1
)
popd

echo.
echo Setup complete.
echo Edit this file with your provider key:
echo %~dp0.env
echo Build the React UI with build_frontend.cmd before starting the app.
endlocal
