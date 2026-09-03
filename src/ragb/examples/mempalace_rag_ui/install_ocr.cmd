@echo off
setlocal

where winget >nul 2>nul
if errorlevel 1 (
    echo winget was not found. Install Tesseract OCR manually, then rerun setup.cmd.
    exit /b 1
)

echo Installing Tesseract OCR for Windows...
winget install --id UB-Mannheim.TesseractOCR -e
if errorlevel 1 exit /b 1

echo.
echo Tesseract installation complete. Close and reopen CMD, then run setup.cmd.
endlocal
