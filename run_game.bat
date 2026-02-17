@echo off
setlocal
echo ==========================================
echo      Be & An Game Launcher v4
echo ==========================================

:: 1. Check Python Version strictness
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)

for /f "tokens=2" %%I in ('python --version') do set "PY_VER=%%I"
echo [INFO] Detected Python Version: %PY_VER%

:: Simple string check for 3.12 or 3.13 (Risk)
echo %PY_VER% | findstr /b "3.12" >nul
if %ERRORLEVEL% EQU 0 goto :VersionWarning
echo %PY_VER% | findstr /b "3.13" >nul
if %ERRORLEVEL% EQU 0 goto :VersionWarning

goto :Setup

:VersionWarning
echo.
echo [WARNING] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo You are using Python %PY_VER%.
echo Libraries like MediaPipe often FAIL to install on Python 3.12+.
echo.
echo RECOMMENDATION:
echo 1. Uninstall Python %PY_VER%.
echo 2. Install Python 3.10 or 3.11 from python.org.
echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
echo.
echo Press any key to try installing anyway (likely to fail)...
pause >nul

:Setup
echo.
echo [STEP 1] Setting up Virtual Environment...
if not exist ".venv" (
    echo Creating .venv...
    python -m venv .venv
) else (
    echo .venv already exists. Checking for pip...
)

:: Ensure pip is installed
.venv\Scripts\python.exe -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Pip not found in .venv. Attempting to install pip...
    .venv\Scripts\python.exe -m ensurepip --default-pip
)

echo.
echo [STEP 2] Installing Dependencies...
echo This version limits versions to ensure compatibility.
.venv\Scripts\python.exe -m pip install --upgrade pip
:: Enforcing compatible versions for mediapipe if possible, but standard install usually works on 3.10/3.11
.venv\Scripts\python.exe -m pip install "streamlit" "streamlit-webrtc" "opencv-python-headless" "mediapipe==0.10.9" "protobuf==3.20.3" "numpy<2.0.0" "av" "streamlit-autorefresh"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FATAL ERROR] Installation Failed.
    echo Most likely cause: Incompatible Python Version.
    echo Please install Python 3.10 or 3.11.
    pause
    exit /b 1
)

echo.
echo [STEP 2.5] Verifying MediaPipe...
.venv\Scripts\python.exe test_mp.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] MediaPipe verification failed.
    pause
    exit /b 1
)

echo.
echo [STEP 3] Launching Game...
.venv\Scripts\python.exe -m streamlit run app.py
pause
