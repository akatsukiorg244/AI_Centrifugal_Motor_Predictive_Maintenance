@echo off
title Centrifugal Pump Digital Twin - SCADA Frontend
cd /d "%~dp0frontend"
echo ======================================================================
echo Starting Centrifugal Pump Industrial SCADA Frontend on port 3000...
echo URL: http://localhost:3000
echo ======================================================================
start http://localhost:3000
python -m http.server 3000
pause
