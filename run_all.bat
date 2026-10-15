@echo off
title Launch Centrifugal Pump Digital Twin
echo Starting Digital Twin Suite...
start "Digital Twin Backend (FastAPI)" cmd /c "%~dp0start_backend.bat"
timeout /t 2 /nobreak >nul
start "Digital Twin Frontend (SCADA UI)" cmd /c "%~dp0start_frontend.bat"
echo Done! Both backend and frontend windows have been launched.
