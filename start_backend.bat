@echo off
title Centrifugal Pump Digital Twin - FastAPI Backend
cd /d "%~dp0backend"
echo ======================================================================
echo Starting Centrifugal Pump Multi-Model AI Backend (FastAPI)...
echo Models: Random Forest, Ridge Regression, XGBoost
echo ======================================================================
python -m uvicorn main:app --reload --port 8000
pause
