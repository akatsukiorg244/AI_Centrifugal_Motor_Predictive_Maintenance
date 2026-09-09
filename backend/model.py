"""
ML Model Wrapper — Loads the pre-trained pump_model.pkl
Handles 13 real sensor features from your CSV data.
"""

import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore")

class PumpMLModel:
    FEATURE_NAMES = [
        'A_ACR_Mot.PV', 'A_ACR_Mot.SV', 'A_ACR_Mot.TV',
        'A_ACR_Pmp.PV', 'A_ACR_Pmp.SV', 'A_ACR_Pmp.TV',
        'A_Pres.PV', 'A_Temp.PV', 'Vibration_hz',
        'Temperature', 'Load_Percent', 'RPM', 'Barometer'
    ]

    def __init__(self, model_path: str = "pump_model.pkl"):
        print(f"🔧 Loading model from {model_path}...")
        self.artifacts = joblib.load(model_path)
        self.rul_model = self.artifacts['rul_model']
        self.health_model = self.artifacts['health_model']
        self.anomaly_model = self.artifacts['anomaly_model']
        self.scaler = self.artifacts['scaler']
        self.metadata = self.artifacts['metadata']
        print(f"✅ Model loaded! R²={self.metadata['rul_r2']:.3f}, Trained on {self.metadata['training_samples']} samples")

    def predict(self, features: List[float]) -> Dict:
        """Run inference on a single 13-feature vector."""
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        rul = float(self.rul_model.predict(X_scaled)[0])
        health = float(self.health_model.predict(X_scaled)[0])
        anomaly_prob = float(self.anomaly_model.predict_proba(X_scaled)[0][1])
        anomaly_flag = int(self.anomaly_model.predict(X_scaled)[0])

        # Determine status
        if health >= 80 and anomaly_flag == 0:
            status = "Healthy"
            color = "#00e676"
        elif health >= 50:
            status = "Warning"
            color = "#ff9100"
        else:
            status = "Critical"
            color = "#ff1744"

        return {
            "rul_hours": round(rul, 2),
            "health_score": round(health, 1),
            "anomaly_probability": round(anomaly_prob * 100, 2),
            "anomaly_flag": anomaly_flag,
            "status": status,
            "status_color": color,
            "features_used": self.FEATURE_NAMES
        }

    def get_feature_importance(self) -> Dict:
        """Return top features for explainability."""
        return self.metadata['feature_importance']

    def get_model_stats(self) -> Dict:
        """Return training metadata for judge display."""
        return {
            "training_samples": self.metadata['training_samples'],
            "test_samples": self.metadata['test_samples'],
            "rul_mae": self.metadata['rul_mae'],
            "rul_r2": self.metadata['rul_r2'],
            "health_r2": self.metadata['health_r2'],
            "anomaly_accuracy": self.metadata['anomaly_acc'],
            "top_features": dict(sorted(
                self.metadata['feature_importance'].items(),
                key=lambda x: -x[1]
            )[:5])
        }