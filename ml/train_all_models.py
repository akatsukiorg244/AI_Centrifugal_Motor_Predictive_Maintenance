"""
Centrifugal Motor Digital Twin — Multi-Model Training & Benchmarking Pipeline
Trains, evaluates, and exports 3 distinct ML architectures on 21,429 real sensor telemetry samples:
  1. Random Forest (Ensemble Bagging)
  2. Ridge Regularized Regression (Analytical Linear ML Model)
  3. XGBoost (Extreme Gradient Boosted Decision Trees - Modern Industrial Benchmark)

Outputs:
  - backend/pump_models_all.pkl (Serialized models & scalers)
  - backend/ml_benchmark_results.json (Detailed accuracy metrics on test data)
  - frontend/data/models_benchmark.json (Benchmark data for UI comparison)
  - frontend/data/historical_trajectory.json (Real sensor curves for live interactive playback)
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Configuration
DATA_PATH = Path("backend/converted_file.csv")
BACKEND_PKL = Path("backend/pump_models_all.pkl")
BENCHMARK_JSON = Path("backend/ml_benchmark_results.json")
FRONTEND_DATA_DIR = Path("frontend/data")
FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    'A_ACR_Mot.PV', 'A_ACR_Mot.SV', 'A_ACR_Mot.TV',
    'A_ACR_Pmp.PV', 'A_ACR_Pmp.SV', 'A_ACR_Pmp.TV',
    'A_Pres.PV', 'A_Temp.PV', 'Vibration_hz',
    'Temperature', 'Load_Percent', 'RPM', 'Barometer'
]
TARGET_RUL = 'RUL_hours'

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

print("=" * 70)
print("[*] INDUSTRIAL CENTRIFUGAL PUMP PREDICTIVE MAINTENANCE ML PIPELINE")
print("=" * 70)

# 1. Load Data
print(f"\n📂 Loading sensor dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"   Raw shape: {df.shape}")

# Clean missing values
df[FEATURE_COLS] = df[FEATURE_COLS].ffill().bfill()
df = df.dropna(subset=[TARGET_RUL])
print(f"   Cleaned rows: {len(df)}")

# Derive targets
max_rul = float(df[TARGET_RUL].max())
df['health_score'] = (df[TARGET_RUL] / max_rul * 100).clip(0, 100)

vib_thresh = float(df['Vibration_hz'].quantile(0.90))
rul_thresh = float(df[TARGET_RUL].quantile(0.10))
pres_thresh = float(df['A_Pres.PV'].quantile(0.05))

df['anomaly'] = (
    (df['Vibration_hz'] > vib_thresh) |
    (df[TARGET_RUL] < rul_thresh) |
    (df['A_Pres.PV'] < pres_thresh)
).astype(int)

anomaly_ratio = float(df['anomaly'].mean())
print(f"   Max RUL: {max_rul:.2f} hrs | Anomaly Prevalence: {anomaly_ratio*100:.2f}%")

# 2. Train / Test Split
X = df[FEATURE_COLS].values
y_rul = df[TARGET_RUL].values
y_health = df['health_score'].values
y_anom = df['anomaly'].values

X_train, X_test, yr_train, yr_test, yh_train, yh_test, ya_train, ya_test = train_test_split(
    X, y_rul, y_health, y_anom, test_size=0.20, random_state=42, shuffle=True
)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\n📊 Split: Train={len(X_train)} samples, Test={len(X_test)} samples")

# ==================== MODEL 1: RANDOM FOREST ====================
print("\n🌲 [1/3] Training Model 1: Random Forest Ensemble...")
t0 = time.time()
rf_rul = RandomForestRegressor(n_estimators=120, max_depth=16, min_samples_split=3, random_state=42, n_jobs=-1)
rf_rul.fit(X_train_s, yr_train)

rf_health = RandomForestRegressor(n_estimators=120, max_depth=16, min_samples_split=3, random_state=42, n_jobs=-1)
rf_health.fit(X_train_s, yh_train)

rf_anom = RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1)
rf_anom.fit(X_train_s, ya_train)
rf_train_time = time.time() - t0

# RF Inference & Evaluation
t0 = time.time()
rf_rul_pred = rf_rul.predict(X_test_s)
rf_health_pred = rf_health.predict(X_test_s)
rf_anom_pred = rf_anom.predict(X_test_s)
rf_anom_prob = rf_anom.predict_proba(X_test_s)[:, 1]
rf_latency_ms = (time.time() - t0) / len(X_test) * 1000

rf_metrics = {
    "model_name": "Random Forest",
    "architecture": "Ensemble Decision Trees (Bagging with 120 Estimators)",
    "paradigm": "Non-linear Non-parametric Ensemble",
    "training_time_s": round(rf_train_time, 2),
    "latency_ms_per_pred": round(rf_latency_ms, 3),
    "rul": {
        "r2": round(float(r2_score(yr_test, rf_rul_pred)), 6),
        "mae_hours": round(float(mean_absolute_error(yr_test, rf_rul_pred)), 4),
        "rmse_hours": round(float(root_mean_squared_error(yr_test, rf_rul_pred)), 4),
        "mape_percent": round(float(np.mean(np.abs((yr_test - rf_rul_pred) / np.maximum(yr_test, 1e-4))) * 100), 4),
        "max_error_hours": round(float(np.max(np.abs(yr_test - rf_rul_pred))), 4)
    },
    "health": {
        "r2": round(float(r2_score(yh_test, rf_health_pred)), 6),
        "mae_percent": round(float(mean_absolute_error(yh_test, rf_health_pred)), 4)
    },
    "anomaly": {
        "accuracy": round(float(accuracy_score(ya_test, rf_anom_pred)), 4),
        "precision": round(float(precision_score(ya_test, rf_anom_pred)), 4),
        "recall": round(float(recall_score(ya_test, rf_anom_pred)), 4),
        "f1": round(float(f1_score(ya_test, rf_anom_pred)), 4),
        "roc_auc": round(float(roc_auc_score(ya_test, rf_anom_prob)), 4)
    }
}
print(f"   [OK] RF Finished: R2={rf_metrics['rul']['r2']}, MAE={rf_metrics['rul']['mae_hours']} hrs, Anomaly Acc={rf_metrics['anomaly']['accuracy']*100:.2f}%")

# ==================== MODEL 2: RIDGE REGRESSION ====================
print("\n[2/3] Training Model 2: Ridge Regularized Regression...")
t0 = time.time()
ridge_rul = Ridge(alpha=10.0, random_state=42)
ridge_rul.fit(X_train_s, yr_train)

ridge_health = Ridge(alpha=10.0, random_state=42)
ridge_health.fit(X_train_s, yh_train)

lr_anom = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_anom.fit(X_train_s, ya_train)
ridge_train_time = time.time() - t0

# Ridge Inference & Evaluation
t0 = time.time()
ridge_rul_pred = ridge_rul.predict(X_test_s)
ridge_health_pred = ridge_health.predict(X_test_s)
lr_anom_pred = lr_anom.predict(X_test_s)
lr_anom_prob = lr_anom.predict_proba(X_test_s)[:, 1]
ridge_latency_ms = (time.time() - t0) / len(X_test) * 1000

ridge_metrics = {
    "model_name": "Ridge Regression",
    "architecture": "L2-Regularized Linear Multivariable Model",
    "paradigm": "Analytical Parametric Convex Optimization",
    "training_time_s": round(ridge_train_time, 2),
    "latency_ms_per_pred": round(ridge_latency_ms, 4),
    "rul": {
        "r2": round(float(r2_score(yr_test, ridge_rul_pred)), 6),
        "mae_hours": round(float(mean_absolute_error(yr_test, ridge_rul_pred)), 4),
        "rmse_hours": round(float(root_mean_squared_error(yr_test, ridge_rul_pred)), 4),
        "mape_percent": round(float(np.mean(np.abs((yr_test - ridge_rul_pred) / np.maximum(yr_test, 1e-4))) * 100), 4),
        "max_error_hours": round(float(np.max(np.abs(yr_test - ridge_rul_pred))), 4)
    },
    "health": {
        "r2": round(float(r2_score(yh_test, ridge_health_pred)), 6),
        "mae_percent": round(float(mean_absolute_error(yh_test, ridge_health_pred)), 4)
    },
    "anomaly": {
        "accuracy": round(float(accuracy_score(ya_test, lr_anom_pred)), 4),
        "precision": round(float(precision_score(ya_test, lr_anom_pred)), 4),
        "recall": round(float(recall_score(ya_test, lr_anom_pred)), 4),
        "f1": round(float(f1_score(ya_test, lr_anom_pred)), 4),
        "roc_auc": round(float(roc_auc_score(ya_test, lr_anom_prob)), 4)
    }
}
print(f"   [OK] Ridge Finished: R2={ridge_metrics['rul']['r2']}, MAE={ridge_metrics['rul']['mae_hours']} hrs, Anomaly Acc={ridge_metrics['anomaly']['accuracy']*100:.2f}%")

# ==================== MODEL 3: XGBOOST ====================
print("\n[3/3] Training Model 3: XGBoost (Gradient Boosted Trees)...")
t0 = time.time()
xgb_rul = xgb.XGBRegressor(
    n_estimators=140,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)
xgb_rul.fit(X_train_s, yr_train)

xgb_health = xgb.XGBRegressor(
    n_estimators=140,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)
xgb_health.fit(X_train_s, yh_train)

xgb_anom = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.85,
    random_state=42,
    n_jobs=-1
)
xgb_anom.fit(X_train_s, ya_train)
xgb_train_time = time.time() - t0

# XGBoost Inference & Evaluation
t0 = time.time()
xgb_rul_pred = xgb_rul.predict(X_test_s)
xgb_health_pred = xgb_health.predict(X_test_s)
xgb_anom_pred = xgb_anom.predict(X_test_s)
xgb_anom_prob = xgb_anom.predict_proba(X_test_s)[:, 1]
xgb_latency_ms = (time.time() - t0) / len(X_test) * 1000

xgb_metrics = {
    "model_name": "XGBoost",
    "architecture": "Extreme Gradient Boosted Decision Trees (2nd-Order Taylor Optimization)",
    "paradigm": "Sequential Residual Boosting",
    "training_time_s": round(xgb_train_time, 2),
    "latency_ms_per_pred": round(xgb_latency_ms, 3),
    "rul": {
        "r2": round(float(r2_score(yr_test, xgb_rul_pred)), 6),
        "mae_hours": round(float(mean_absolute_error(yr_test, xgb_rul_pred)), 4),
        "rmse_hours": round(float(root_mean_squared_error(yr_test, xgb_rul_pred)), 4),
        "mape_percent": round(float(np.mean(np.abs((yr_test - xgb_rul_pred) / np.maximum(yr_test, 1e-4))) * 100), 4),
        "max_error_hours": round(float(np.max(np.abs(yr_test - xgb_rul_pred))), 4)
    },
    "health": {
        "r2": round(float(r2_score(yh_test, xgb_health_pred)), 6),
        "mae_percent": round(float(mean_absolute_error(yh_test, xgb_health_pred)), 4)
    },
    "anomaly": {
        "accuracy": round(float(accuracy_score(ya_test, xgb_anom_pred)), 4),
        "precision": round(float(precision_score(ya_test, xgb_anom_pred)), 4),
        "recall": round(float(recall_score(ya_test, xgb_anom_pred)), 4),
        "f1": round(float(f1_score(ya_test, xgb_anom_pred)), 4),
        "roc_auc": round(float(roc_auc_score(ya_test, xgb_anom_prob)), 4)
    }
}
print(f"   [OK] XGBoost Finished: R2={xgb_metrics['rul']['r2']}, MAE={xgb_metrics['rul']['mae_hours']} hrs, Anomaly Acc={xgb_metrics['anomaly']['accuracy']*100:.2f}%")

# Feature importance analysis
feature_importances = {
    "Random Forest": {k: round(float(v), 5) for k, v in zip(FEATURE_COLS, rf_rul.feature_importances_)},
    "Ridge Regression": {k: round(float(abs(v)), 5) for k, v in zip(FEATURE_COLS, ridge_rul.coef_)},
    "XGBoost": {k: round(float(v), 5) for k, v in zip(FEATURE_COLS, xgb_rul.feature_importances_)}
}

# 3. Generate Historical Trajectory & Graph Samples
print("\n[+] Generating test trajectories and verification samples...")
sample_indices = np.linspace(0, len(X_test) - 1, 150, dtype=int)
sorted_order = np.argsort(-yr_test[sample_indices])
selected = sample_indices[sorted_order]

graph_series = []
for i in selected:
    graph_series.append({
        "actual_rul": round(float(yr_test[i]), 2),
        "rf_rul": round(float(rf_rul_pred[i]), 2),
        "ridge_rul": round(float(ridge_rul_pred[i]), 2),
        "xgb_rul": round(float(xgb_rul_pred[i]), 2),
        "actual_health": round(float(yh_test[i]), 1),
        "rf_health": round(float(rf_health_pred[i]), 1),
        "ridge_health": round(float(ridge_health_pred[i]), 1),
        "xgb_health": round(float(xgb_health_pred[i]), 1),
        "actual_anomaly": int(ya_test[i]),
        "rf_anomaly_prob": round(float(rf_anom_prob[i] * 100), 1),
        "ridge_anomaly_prob": round(float(lr_anom_prob[i] * 100), 1),
        "xgb_anomaly_prob": round(float(xgb_anom_prob[i] * 100), 1),
        "vibration_hz": round(float(X_test[i, FEATURE_COLS.index('Vibration_hz')]), 2),
        "pressure_bar": round(float(X_test[i, FEATURE_COLS.index('A_Pres.PV')]), 2),
        "temperature_c": round(float(X_test[i, FEATURE_COLS.index('Temperature')]), 1),
        "rpm": round(float(X_test[i, FEATURE_COLS.index('RPM')]), 0),
        "features": [round(float(v), 4) for v in X_test[i]]
    })

# Extract Scaler parameters and Linear Weights for pure client-side execution in JavaScript
scaler_center = [float(v) for v in scaler.center_]
scaler_scale = [float(v) for v in scaler.scale_]
ridge_rul_coef = [float(v) for v in ridge_rul.coef_]
ridge_rul_intercept = float(ridge_rul.intercept_)
ridge_health_coef = [float(v) for v in ridge_health.coef_]
ridge_health_intercept = float(ridge_health.intercept_)
lr_coef = [float(v) for v in lr_anom.coef_[0]]
lr_intercept = float(lr_anom.intercept_[0])

client_math = {
    "features": FEATURE_COLS,
    "scaler_center": scaler_center,
    "scaler_scale": scaler_scale,
    "max_rul": max_rul,
    "ridge": {
        "rul_coef": ridge_rul_coef,
        "rul_intercept": ridge_rul_intercept,
        "health_coef": ridge_health_coef,
        "health_intercept": ridge_health_intercept,
        "anom_coef": lr_coef,
        "anom_intercept": lr_intercept
    }
}

# 4. Save Artifacts
comparison_bundle = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_dataset_rows": len(df),
    "features": FEATURE_COLS,
    "max_rul_hours": max_rul,
    "models": {
        "random_forest": rf_metrics,
        "ridge_regression": ridge_metrics,
        "xgboost": xgb_metrics
    },
    "feature_importances": feature_importances,
    "trajectory_samples": graph_series,
    "client_inference_math": client_math
}

with open(BENCHMARK_JSON, "w") as f:
    json.dump(comparison_bundle, f, indent=2)
print(f"\n[SAVED] Benchmark json to: {BENCHMARK_JSON}")

with open(FRONTEND_DATA_DIR / "models_benchmark.json", "w") as f:
    json.dump(comparison_bundle, f, indent=2)
print(f"[SAVED] Frontend benchmark json to: {FRONTEND_DATA_DIR / 'models_benchmark.json'}")

# Save Python pickle bundle
artifacts = {
    "scaler": scaler,
    "feature_names": FEATURE_COLS,
    "max_rul": max_rul,
    "random_forest": {
        "rul": rf_rul,
        "health": rf_health,
        "anomaly": rf_anom,
        "metrics": rf_metrics
    },
    "ridge_regression": {
        "rul": ridge_rul,
        "health": ridge_health,
        "anomaly": lr_anom,
        "metrics": ridge_metrics
    },
    "xgboost": {
        "rul": xgb_rul,
        "health": xgb_health,
        "anomaly": xgb_anom,
        "metrics": xgb_metrics
    },
    "feature_importances": feature_importances
}

joblib.dump(artifacts, BACKEND_PKL, compress=3)
print(f"[SAVED] Complete model bundle to: {BACKEND_PKL} ({BACKEND_PKL.stat().st_size / (1024*1024):.2f} MB)")

print("\n" + "=" * 70)
print("[SUCCESS] MULTI-MODEL TRAINING & BENCHMARKING COMPLETE!")
print(f"Random Forest MAE: {rf_metrics['rul']['mae_hours']} hrs | R2: {rf_metrics['rul']['r2']}")
print(f"Ridge Regr.   MAE: {ridge_metrics['rul']['mae_hours']} hrs | R2: {ridge_metrics['rul']['r2']}")
print(f"XGBoost       MAE: {xgb_metrics['rul']['mae_hours']} hrs | R2: {xgb_metrics['rul']['r2']}")
print("=" * 70)
