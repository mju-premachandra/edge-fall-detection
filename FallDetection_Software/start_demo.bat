@echo off
setlocal
cd /d "%~dp0"
title Fall Detection Software Demo
echo Fall Detection - starting your local software demo
echo.
set "DEMO_PYTHON="
if exist "..\.venv313\Scripts\python.exe" set "DEMO_PYTHON=%~dp0..\.venv313\Scripts\python.exe"
if not defined DEMO_PYTHON if exist ".venv313\Scripts\python.exe" set "DEMO_PYTHON=%~dp0.venv313\Scripts\python.exe"
if not defined DEMO_PYTHON if exist ".venv\Scripts\python.exe" set "DEMO_PYTHON=%~dp0.venv\Scripts\python.exe"
if defined DEMO_PYTHON goto install
echo Creating a Python 3.13 environment for this project...
py -3.13 -m venv .venv
if errorlevel 1 goto python_error
set "DEMO_PYTHON=%~dp0.venv\Scripts\python.exe"
:install
"%DEMO_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto install_error
echo.
echo Open http://127.0.0.1:5050 if the browser does not open automatically.
echo Keep this window open while using the app. Press Ctrl+C to stop.
echo.
"%DEMO_PYTHON%" app.py
if errorlevel 1 goto app_error
goto end
:python_error
echo.
echo Python 3.13 could not be found. See START_HERE.txt for the manual commands.
goto end
:install_error
echo.
echo Package installation failed. Check your internet connection and read the error above.
goto end
:app_error
echo.
echo The app stopped with an error. Keep this window open and share the error text.
:end
pause
endlocal
