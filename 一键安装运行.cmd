@echo off
setlocal EnableExtensions EnableDelayedExpansion
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
set "READY=%VENV%\.videohighlighter-cn-ready"

if exist "%READY%" if exist "%VPY%" goto RUN_APP

echo.
echo ========================================
echo   VideoHighlighter-CN first-time setup
echo ========================================
echo.

call :FIND_PYTHON

if not defined PYEXE (
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

  echo [INFO] Python installation finished. Detecting the new installation...

  set /a RETRIES=0
  :PY_RETRY
  set /a RETRIES+=1
  call :FIND_PYTHON
  if defined PYEXE goto PY_READY

  if !RETRIES! LSS 10 (
    timeout /t 2 /nobreak >nul
    goto PY_RETRY
  )

  echo.
  echo [ERROR] Python 3.12 was installed, but its executable could not be located.
  echo Common install locations were checked automatically.
  echo.
  echo You can close this window and run this file again.
  echo If the problem repeats, verify Python 3.12 in Windows Settings ^> Apps.
  echo.
  pause
  exit /b 4
)

:PY_READY
echo [INFO] Using Python: "%PYEXE%"

if not exist "%VPY%" (
  echo [1/4] Creating isolated Python environment...
  "%PYEXE%" -m venv "%VENV%"
  if errorlevel 1 goto SETUP_FAILED
) else (
  echo [1/4] Reusing incomplete Python environment...
)

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

> "%READY%" echo ready

:RUN_APP
echo.
echo [INFO] Starting VideoHighlighter-CN...
start "" "%VPY%" "%CD%\main.py"
exit /b 0

:FIND_PYTHON
set "PYEXE="

for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys; print(sys.executable)" 2^>nul`) do (
  if exist "%%~P" set "PYEXE=%%~P"
)
if defined PYEXE goto :eof

for /f "usebackq delims=" %%P in (`python -c "import sys; assert sys.version_info[:2] == (3,12); print(sys.executable)" 2^>nul`) do (
  if exist "%%~P" set "PYEXE=%%~P"
)
if defined PYEXE goto :eof

for %%P in (
  "%LocalAppData%\Programs\Python\Python312\python.exe"
  "%ProgramFiles%\Python312\python.exe"
  "%ProgramFiles(x86)%\Python312\python.exe"
) do (
  if exist "%%~P" (
    "%%~P" -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=%%~P"
  )
)
if defined PYEXE goto :eof

for /d %%D in ("%LocalAppData%\Programs\Python\Python312*") do (
  if exist "%%~fD\python.exe" (
    "%%~fD\python.exe" -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=%%~fD\python.exe"
  )
)
if defined PYEXE goto :eof

for /d %%D in ("%ProgramFiles%\Python312*") do (
  if exist "%%~fD\python.exe" (
    "%%~fD\python.exe" -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
    if not errorlevel 1 set "PYEXE=%%~fD\python.exe"
  )
)

goto :eof

:SETUP_FAILED
echo.
echo ========================================
echo [ERROR] Installation did not complete.
echo The project source was not modified.
echo The incomplete environment is kept in:
echo %VENV%
echo.
echo Run this file again to retry. If the same dependency keeps failing,
echo delete the .venv folder and try again.
echo ========================================
echo.
pause
exit /b 10
