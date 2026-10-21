@echo off
setlocal
cd /d "%~dp0"
title Random Forest - Second Model Experiment
set "RF_PYTHON="
if exist "..\.venv313\Scripts\python.exe" set "RF_PYTHON=%~dp0..\.venv313\Scripts\python.exe"
if not defined RF_PYTHON if exist ".venv313\Scripts\python.exe" set "RF_PYTHON=%~dp0.venv313\Scripts\python.exe"
if not defined RF_PYTHON if exist ".venv\Scripts\python.exe" set "RF_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined RF_PYTHON goto missing_python
"%RF_PYTHON%" -m pip install -r requirements-training.txt
if errorlevel 1 goto finished
echo.
echo Training the separate Random Forest. Original app and tree will be preserved.
echo Only Random Forest model and report files will be replaced on repeat runs.
echo.
"%RF_PYTHON%" train_random_forest.py
goto finished
:missing_python
echo Could not find the existing Python environment.
echo See RANDOM_FOREST_START_HERE.txt for manual commands.
:finished
pause
endlocal
