@echo off
REM ============================================================
REM  PaleoVox Windows Build Script
REM  Run this on a Windows machine with Python 3.8+ installed
REM ============================================================

echo.
echo [1/4] Installing PyInstaller...
pip install pyinstaller
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PyInstaller
    exit /b 1
)

echo.
echo [2/4] Installing project dependencies...
pip install open3d numpy scipy matplotlib seaborn plotly scikit-learn PyQt5
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo [3/4] Building PaleoVox.exe...
REM Build from the parent directory so project files resolve correctly
cd /d "%~dp0..\..\"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to navigate to project root
    exit /b 1
)

pyinstaller --distpath "portable_versions\windows" ^
    --workpath "%TEMP%\paleovox_build" ^
    --clean ^
    "portable_versions\PaleoVox.spec"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed
    exit /b 1
)

echo.
echo [4/4] Post-build cleanup...
REM Remove CUDA library to save space
rmdir /s /q "portable_versions\windows\PaleoVox\_internal\open3d\cuda" 2>nul
REM Remove Dash (not needed)
rmdir /s /q "portable_versions\windows\PaleoVox\_internal\dash" 2>nul
rmdir /s /q "portable_versions\windows\PaleoVox\_internal\dash_html_components" 2>nul
rmdir /s /q "portable_versions\windows\PaleoVox\_internal\dash_core_components" 2>nul
rmdir /s /q "portable_versions\windows\PaleoVox\_internal\dash_table" 2>nul

echo.
echo ========================================
echo  Build complete!
echo  Output: portable_versions\windows\PaleoVox\
echo  Launch: PaleoVox\PaleoVox.bin
echo ========================================

REM Create a launcher batch file
echo @echo off > "portable_versions\windows\run_paleovox.bat"
echo start "" "%%~dp0PaleoVox\PaleoVox.bin" %%* >> "portable_versions\windows\run_paleovox.bat"

echo Launcher created: portable_versions\windows\run_paleovox.bat
pause
