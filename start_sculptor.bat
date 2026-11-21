@echo off
rem ===========================================================================
rem  SCULPTOR v1.0.0  -  one-click launcher
rem
rem  Double-click this file.  It will
rem    1. build the environment if it is missing (Python 3.11.9 + packages),
rem    2. repair it if an earlier install was interrupted,
rem    3. open the control panel - the native window, or the browser when the
rem       native window cannot be used on this machine.
rem
rem  Optional first argument:
rem    start_sculptor.bat            - control panel, native window
rem    start_sculptor.bat web        - control panel in the browser
rem    start_sculptor.bat test       - run the unit tests
rem    start_sculptor.bat cli <args> - raw command line
rem ===========================================================================
setlocal EnableExtensions
set "HERE=%~dp0"
cd /d "%HERE%"
set "PY=%HERE%.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo [SCULPTOR] No environment found.
    echo [SCULPTOR] Downloading Python 3.11.9 into runtime\python311 and
    echo [SCULPTOR] building .venv from it.  This needs internet access and
    echo [SCULPTOR] takes a few minutes; it only happens once.
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%setup_env.ps1"
)

if not exist "%PY%" (
    echo.
    echo [SCULPTOR] Could not create the environment.
    echo            Run this manually and read the output:
    echo              powershell -ExecutionPolicy Bypass -File setup_env.ps1
    echo.
    pause
    exit /b 1
)

rem ------------------------------------------------------------------ core ---
rem The pipeline needs these; without them nothing runs.
"%PY%" -c "import torch, rdkit, torch_geometric, numpy, pandas, sklearn, e3nn, xgboost, h5py" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [SCULPTOR] The environment is incomplete - finishing the installation.
    echo [SCULPTOR] Already downloaded files are reused, nothing is thrown away.
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%setup_env.ps1"
)

"%PY%" -c "import torch, rdkit, torch_geometric, numpy, pandas, sklearn, e3nn, xgboost, h5py" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [SCULPTOR] The environment is still incomplete after the repair.
    echo            Run this to see what is missing:
    echo              powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Check
    echo.
    pause
    exit /b 1
)

set "PYTHONNOUSERSITE=1"
set "PYTHONPATH=%HERE%"

if /i "%~1"=="cli"  goto cli
if /i "%~1"=="test" goto test
if /i "%~1"=="web"  goto web
goto panel


:panel
rem ---------------------------------------------------------- native window ---
rem pywebview is optional, but it is the nicer experience.
"%PY%" -c "import webview, proxy_tools, bottle" >nul 2>&1
if not errorlevel 1 goto native

rem proxy_tools is shipped inside the package because some mirrors serve a
rem wheel that contains only its metadata.  If that is what is missing, the
rem repair works offline; anything else means pip cannot help either and the
rem browser is the right answer straight away.
"%PY%" -c "import proxy_tools" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [SCULPTOR] The native window needs pywebview, which is not usable on
    echo            this machine.  Opening the panel in your browser instead.
    echo.
    goto web
)

echo.
echo [SCULPTOR] Preparing the native window ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%setup_env.ps1" >nul 2>&1
"%PY%" -c "import webview, proxy_tools, bottle" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [SCULPTOR] The native window could not be prepared, so the panel will
    echo            open in your browser instead.
    echo.
    goto web
)

:native
echo [SCULPTOR] opening the control panel ...
"%PY%" "%HERE%tools\control_panel\desktop_panel.pyw"
goto end


:web
rem --------------------------------------------------------- browser panel ---
netstat -an | findstr /r /c:":8765 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [SCULPTOR] Port 8765 is already in use - a panel server is probably
    echo            still running.  Open it directly:
    echo              http://127.0.0.1:8765/
    echo            or close the other window and start this again.
    echo.
    start "" "http://127.0.0.1:8765/"
    pause
    goto end
)

echo.
echo [SCULPTOR] Starting the panel server at http://127.0.0.1:8765/
echo [SCULPTOR] The browser opens in a few seconds.
echo.
echo [SCULPTOR] KEEP THIS WINDOW OPEN while you use the panel - the server
echo [SCULPTOR] runs inside it.  Closing it stops the panel, and the browser
echo [SCULPTOR] page will then report "Failed to fetch".
echo.
start "" /b cmd /c "timeout /t 4 >nul & start "" http://127.0.0.1:8765/"
"%PY%" -m tools.control_panel.server

echo.
echo [SCULPTOR] The panel server has stopped.
echo [SCULPTOR] If the browser page is still open it can no longer reach the
echo [SCULPTOR] server, so its buttons will report "Failed to fetch".
echo.
pause
goto end


:test
"%PY%" -m pytest tests -q
goto end


:cli
shift
"%PY%" -m fuel_design.cli %1 %2 %3 %4 %5 %6 %7 %8 %9
goto end


:end
echo.
echo [SCULPTOR] finished.
pause
