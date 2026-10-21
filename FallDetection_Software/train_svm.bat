@echo off
setlocal
cd /d "%~dp0"
title SVM - Third Model Experiment
set "SVM_PYTHON="
if exist "..\.venv313\Scripts\python.exe" set "SVM_PYTHON=%~dp0..\.venv313\Scripts\python.exe"
if not defined SVM_PYTHON if exist ".venv313\Scripts\python.exe" set "SVM_PYTHON=%~dp0.venv313\Scripts\python.exe"
if not defined SVM_PYTHON if exist ".venv\Scripts\python.exe" set "SVM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined SVM_PYTHON goto missing_python
"%SVM_PYTHON%" -m pip install -r requirements-training.txt
if errorlevel 1 goto failed
echo.
echo Training SVM using subjects 1-3; choosing settings using subject 4.
echo Subjects 5-6 are a reused comparison set, not a new independent test.
echo.
"%SVM_PYTHON%" train_svm.py
if errorlevel 1 goto failed
echo.
echo Training finished. Restart start_demo.bat and select Support Vector Machine (SVM).
goto finished
:missing_python
echo Could not find your existing Python environment.
echo Read SVM_START_HERE.txt for manual commands.
goto finished
:failed
echo.
echo The command failed. Read the error above or send a screenshot for help.
:finished
pause
endlocal
