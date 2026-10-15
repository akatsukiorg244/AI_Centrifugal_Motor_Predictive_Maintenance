"""
Multi-Model Machine Learning Engine — Centrifugal Pump Digital Twin
Serves 3 concurrent models:
  1. Random Forest (Ensemble Bagging)
  2. Ridge Regression (Analytical Linear Regularized ML Model)
  3. XGBoost (Gradient Boosted Trees - PHM Industry Benchmark)
"""

import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class MultiModelPumpEngine:
    FEATURE_NAMES = [
        'A_ACR_Mot.PV', 'A_ACR_Mot.SV', 'A_ACR_Mot.TV',
        'A_ACR_Pmp.PV', 'A_ACR_Pmp.SV', 'A_ACR_Pmp.TV',
        'A_Pres.PV', 'A_Temp.PV', 'Vibration_hz',
        'Temperature', 'Load_Percent', 'RPM', 'Barometer'
    ]

    def __init__(self, model_path: str = "pump_models_all.pkl"):
        p = Path(model_path)
        if not p.exists():
            # Try searching parent or backend dir
            search_paths = [
                Path("backend") / model_path,
                Path("../backend") / model_path,
                Path("pump_models_all.pkl")
            ]
            for sp in search_paths:
                if sp.exists():
                    p = sp
                    break

        print(f"[*] Loading multi-model artifacts from {p}...")
        self.bundle = joblib.load(p)
        self.scaler = self.bundle['scaler']
        self.max_rul = self.bundle['max_rul']
        self.rf = self.bundle['random_forest']
        self.ridge = self.bundle['ridge_regression']
        self.xgb = self.bundle['xgboost']
        self.feature_importances = self.bundle['feature_importances']
        print("[OK] All 3 ML Models (Random Forest, Ridge Regression, XGBoost) successfully initialized.")

    def _determine_status(self, health: float, anomaly_flag: int) -> Dict[str, str]:
        if health >= 80 and anomaly_flag == 0:
            return {"status": "HEALTHY", "color": "#10b981", "badge_class": "status-healthy"}
        elif health >= 65:
            return {"status": "GOOD", "color": "#06b6d4", "badge_class": "status-good"}
        elif health >= 45:
            return {"status": "NEUTRAL", "color": "#f59e0b", "badge_class": "status-neutral"}
        elif health >= 25:
            return {"status": "NEEDS ATTENTION", "color": "#f97316", "badge_class": "status-warning"}
        else:
            return {"status": "CRITICAL / FAILURE", "color": "#ef4444", "badge_class": "status-critical"}

    def predict_model(self, model_key: str, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Inference for a single selected model."""
        if model_key == "random_forest":
            rul = float(self.rf['rul'].predict(X_scaled)[0])
            health = float(self.rf['health'].predict(X_scaled)[0])
            anom_flag = int(self.rf['anomaly'].predict(X_scaled)[0])
            anom_prob = float(self.rf['anomaly'].predict_proba(X_scaled)[0][1])
            name = "Random Forest"
        elif model_key == "ridge_regression":
            rul = float(self.ridge['rul'].predict(X_scaled)[0])
            health = float(self.ridge['health'].predict(X_scaled)[0])
            anom_flag = int(self.ridge['anomaly'].predict(X_scaled)[0])
            anom_prob = float(self.ridge['anomaly'].predict_proba(X_scaled)[0][1])
            name = "Ridge Regression"
        elif model_key == "xgboost":
            rul = float(self.xgb['rul'].predict(X_scaled)[0])
            health = float(self.xgb['health'].predict(X_scaled)[0])
            anom_flag = int(self.xgb['anomaly'].predict(X_scaled)[0])
            anom_prob = float(self.xgb['anomaly'].predict_proba(X_scaled)[0][1])
            name = "XGBoost"
        else:
            raise ValueError(f"Unknown model_key: {model_key}")

        # Physical boundary clipping
        rul = max(0.0, min(self.max_rul, rul))
        health = max(0.0, min(100.0, health))
        anom_prob = max(0.0, min(1.0, anom_prob))

        status_info = self._determine_status(health, anom_flag)

        return {
            "model_key": model_key,
            "model_name": name,
            "rul_hours": round(rul, 2),
            "rul_days": round(rul / 24.0, 1),
            "health_score": round(health, 1),
            "anomaly_flag": anom_flag,
            "anomaly_probability": round(anom_prob * 100, 2),
            **status_info
        }

    def predict_all(self, features: List[float]) -> Dict[str, Any]:
        """Runs concurrent prediction through all 3 models and computes ensemble consensus."""
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        rf_res = self.predict_model("random_forest", X_scaled)
        ridge_res = self.predict_model("ridge_regression", X_scaled)
        xgb_res = self.predict_model("xgboost", X_scaled)

        rul_list = [rf_res['rul_hours'], ridge_res['rul_hours'], xgb_res['rul_hours']]
        health_list = [rf_res['health_score'], ridge_res['health_score'], xgb_res['health_score']]
        anom_prob_list = [rf_res['anomaly_probability'], ridge_res['anomaly_probability'], xgb_res['anomaly_probability']]

        avg_rul = round(float(np.mean(rul_list)), 2)
        std_rul = round(float(np.std(rul_list)), 2)
        avg_health = round(float(np.mean(health_list)), 1)
        max_discrepancy_hours = round(float(max(rul_list) - min(rul_list)), 2)

        # Consensus confidence (100% when all agree closely, lower when models diverge)
        consensus_confidence = max(0.0, min(100.0, 100.0 - (std_rul / max(avg_rul, 1.0) * 100.0 * 2.0)))

        return {
            "timestamp": None,
            "models": {
                "random_forest": rf_res,
                "ridge_regression": ridge_res,
                "xgboost": xgb_res
            },
            "consensus": {
                "ensemble_rul_hours": avg_rul,
                "ensemble_health_score": avg_health,
                "discrepancy_hours": max_discrepancy_hours,
                "consensus_confidence_pct": round(consensus_confidence, 1),
                "model_agreement": "HIGH" if max_discrepancy_hours < 25 else ("MODERATE" if max_discrepancy_hours < 60 else "DIVERGENT")
            },
            "features_used": dict(zip(self.FEATURE_NAMES, features))
        }

    def get_benchmark_report(self) -> Dict[str, Any]:
        return {
            "random_forest": self.rf['metrics'],
            "ridge_regression": self.ridge['metrics'],
            "xgboost": self.xgb['metrics'],
            "feature_importances": self.feature_importances
        }