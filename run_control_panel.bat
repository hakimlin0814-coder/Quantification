@echo off
cd /d "%~dp0"
python control_panel.py
if errorlevel 1 (
  echo.
  echo Quant control panel exited with an error.
  pause
)
