# 🔧 Centrifugal Pump Digital Twin — Industrial Predictive Maintenance Suite
> **ISO 10816-3 Class II Standard Compliant** | **3 Multi-Model ML Ensemble** | **21,429 Telemetry Records** | **Interactive SCADA Visualizer**

---

## 🏗️ 3-Model Machine Learning Architecture

This industrial digital twin benchmarks 3 distinct machine learning architectures on **21,429 rows of real sensor data** across 13 industrial features:

1. **Model 1: Random Forest Ensemble (Bagging)**
   - 120 Estimators with RobustScaler normalization.
   - Non-parametric decision boundary modeling.
   - $R^2 = 0.999999$, $\text{MAE} = 0.12\text{ hours}$ (~7 minutes).
2. **Model 2: Ridge Regularized Regression (Analytical Linear)**
   - L2 regularized multivariable linear model.
   - Closed-form analytical optimization.
   - $R^2 = 0.983445$, $\text{MAE} = 36.71\text{ hours}$ (~1.5 days).
3. **Model 3: XGBoost (Gradient Boosted Decision Trees — PHM Industry Benchmark)**
   - Second-order Taylor gradient optimization.
   - Captures non-linear saturation thresholds with ultra-low latency.
   - $R^2 = 0.999989$, $\text{MAE} = 0.80\text{ hours}$ (~48 minutes), **Inference Latency = 0.003 ms (15× faster than Random Forest)**.

---

## 📊 Benchmark Comparison (4,286 Real Test Samples)

| Metric | Random Forest | Ridge Regression | XGBoost (Recommended) |
| :--- | :--- | :--- | :--- |
| **Architecture** | Bagged Trees (120 Estimators) | L2 Linear Hyperplane | Gradient Boosted Trees |
| **RUL $R^2$ Score** | **0.999999** | **0.983445** | **0.999989** |
| **RUL MAE (Error)** | **0.12 hrs** (~7 min) | **36.71 hrs** (~1.5 days) | **0.80 hrs** (~48 min) |
| **Anomaly Accuracy** | **100.00%** | **96.94%** | **99.91%** |
| **Anomaly F1-Score** | **1.0000** | **0.8927** | **0.9968** |
| **Latency per Point** | 0.0460 ms | **0.0019 ms** | **0.0030 ms** |

---

## 🚀 Quick Start Guide

### 1. Train the Models
```bash
python ml/train_all_models.py
# Outputs: backend/pump_models_all.pkl & frontend/data/models_benchmark.json
```

### 2. Start the Backend API (FastAPI)
```bash
cd backend
uvicorn main:app --reload --port 8000
# API Docs available at http://localhost:8000/docs
```

### 3. Open the Industrial SCADA Frontend
Open `frontend/index.html` in any modern web browser or serve via:
```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```
*Note: The frontend is built with a hybrid client/server architecture. When FastAPI is running, it communicates live via REST/WebSocket. When hosted statically (e.g., on Vercel), it executes embedded ML weights directly in JavaScript with zero downtime!*

---

## 🎛️ Interactive Seminar Features
- **3 Synchronized Industrial Charts**: RUL Trajectory, Anomaly Risk & Vibration Boundary, and Component Health Degradation.
- **Physical Twin Schematic**: Animated impeller RPM, cavitation bubble burst effects, and clickable thermal hotspots (Bearings, Impeller, Mechanical Seal, Stator Windings).
- **1-Click Failure Injections**: Nominal, Cavitation, Bearing Spall, Motor Overheat, Seal Leak, Rotor Unbalance.
- **Timeline Scrubber**: Scrub through 150 real verified data points from the test dataset.
- **Official Work Order Generator**: One-click printable PDF maintenance audit report.
