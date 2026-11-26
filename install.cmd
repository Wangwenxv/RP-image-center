@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\install_native.py all --workers 12
if errorlevel 1 goto failed
echo Installed. Use start.cmd.
exit /b 0
:failed
echo Installation failed. Inspect the error above.
pause
exit /b 1
