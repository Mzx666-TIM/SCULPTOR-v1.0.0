@echo off
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\.."
start "" "%ROOT%\.venv\Scripts\pythonw.exe" "%SCRIPT_DIR%desktop_panel.pyw"
