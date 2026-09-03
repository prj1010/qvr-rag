@echo off
setlocal

where npm >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Install Node.js 20 or newer, then run setup.cmd.
    exit /b 1
)

if not exist "%~dp0frontend\node_modules" (
    echo Frontend dependencies are missing. Run setup.cmd first.
    exit /b 1
)

echo Starting the React development server at http://127.0.0.1:5173
pushd "%~dp0frontend"
npm run dev
popd
endlocal
