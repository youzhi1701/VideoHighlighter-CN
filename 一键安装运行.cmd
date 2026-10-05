@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title VideoHighlighter-CN Setup and Run

if not exist "main.py" (
  echo.
  echo [ERROR] main.py was not found.
  echo Please keep this file in the VideoHighlighter-CN project root.
  echo.
  pause
  exit /b 1
)

set "VENV=%CD%\.venv"
set "VPY=%VENV%\Scripts\python.exe"

if exist "%VPY%" goto RUN_APP

echo.
echo ========================================
echo   VideoHighlighter-CN first-time setup
echo ========================================
echo.

set "PY_CMD="

py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if %errorlevel%==0 set "PY_CMD=py -3.12"

if not defined PY_CMD (
  python -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
  if %errorlevel%==0 set "PY_CMD=python"
)

if not defined PY_CMD (
  echo [INFO] Python 3.12 was not found.
  where winget >nul 2>&1
  if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.12 is required, and winget is not available.
    echo Install Python 3.12 from python.org, then run this file again.
    echo.
    pause
    exit /b 2
  )

  echo [INFO] Installing Python 3.12 with winget...
  winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo.
    echo [ERROR] Python 3.12 installation failed.
    echo.
    pause
    exit /b 3
  )

  set "PY312=%LocalAppData%\Programs\Python\Python312\python.exe"
  if exist "%PY312%" (
    set "PY_CMD="%PY312%""
  ) else (
    echo.
    echo [INFO] Python was installed, but this terminal cannot see it yet.
    echo Close this window and double-click this file again.
    echo.
    pause
    exit /b 0
  )
)

echo [1/4] Creating isolated Python environment...
%PY_CMD% -m venv "%VENV%"
if errorlevel 1 goto SETUP_FAILED

echo [2/4] Updating pip...
"%VPY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto SETUP_FAILED

echo [3/4] Installing VideoHighlighter dependencies...
"%VPY%" -m pip install -r "requirements.txt"
if errorlevel 1 goto SETUP_FAILED

echo [4/4] Installing YOLOX runtime package...
"%VPY%" -m pip install yolox==0.3.0 --no-deps
if errorlevel 1 (
  echo [WARN] YOLOX package installation failed.
  echo [WARN] The main app can still be started; some detector training features may be unavailable.
)

:RUN_APP
echo.
echo [INFO] Starting VideoHighlighter-CN...
start "" "%VPY%" "%CD%\main.py"
exit /b 0

:SETUP_FAILED
echo.
echo ========================================
echo [ERROR] Installation did not complete.
echo No system Python files were modified.
echo Delete the .venv folder and run this file again if needed.
echo ========================================
echo.
pause
exit /b 10
