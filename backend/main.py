"""
FastAPI Industrial Backend — Centrifugal Pump Digital Twin
Supports multi-model inference (Random Forest, Ridge Regression, XGBoost),
real-time telemetry playback, and physical failure scenario simulation.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from model import MultiModelPumpEngine
from data_streamer import PumpDataStreamer

app = FastAPI(
    title="Centrifugal Pump Industrial Digital Twin API",
    description="Multi-Model Predictive Maintenance Suite (Random Forest, Ridge Regression, XGBoost)",
    version="3.0.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load engine
print("[*] Initializing Multi-Model Engine...")
ml_engine = MultiModelPumpEngine("pump_models_all.pkl")

# Load historical telemetry cache
benchmark_file = Path("backend/ml_benchmark_results.json")
if not benchmark_file.exists():
    benchmark_file = Path("ml_benchmark_results.json")

trajectory_cache = []
benchmark_cache = {}
if benchmark_file.exists():
    with open(benchmark_file, "r") as f:
        data = json.load(f)
        trajectory_cache = data.get("trajectory_samples", [])
        benchmark_cache = data.get("models", {})

# Try loading streamer
streamer = None
csv_path = Path("backend/converted_file.csv")
if not csv_path.exists():
    csv_path = Path("converted_file.csv")
if csv_path.exists():
    try:
        streamer = PumpDataStreamer(str(csv_path))
        print("[OK] Telemetry streamer initialized with real pump data.")
    except Exception as e:
        print(f"[!] Streamer warning: {e}")

class SensorInput(BaseModel):
    features: List[float] = Field(..., description="13 telemetry features in exact order")

class FaultSimulationRequest(BaseModel):
    scenario: str = Field(..., description="cavitation, bearing_spall, thermal_overload, seal_leak, or unbalance")
    severity: float = Field(default=1.0, ge=0.1, le=3.0, description="Fault severity multiplier")

@app.get("/api/system")
def system_info():
    return {
        "system": "Centrifugal Motor Digital Twin - Industrial Edition",
        "version": "3.0.0",
        "models_active": ["Random Forest", "Ridge Regression", "XGBoost"],
        "endpoints": [
            "/api/predict/all",
            "/api/benchmark",
            "/api/telemetry/history",
            "/api/simulate/fault"
        ]
    }

@app.get("/api/benchmark")
def get_benchmark():
    """Returns training accuracy, MAE, R², latency, and ROC-AUC for all 3 models."""
    return {
        "status": "success",
        "models": benchmark_cache,
        "feature_importances": ml_engine.feature_importances
    }

@app.get("/api/telemetry/history")
def get_telemetry_history():
    """Returns real continuous sensor trajectories with predictions from all 3 models."""
    return {
        "status": "success",
        "count": len(trajectory_cache),
        "trajectory": trajectory_cache
    }

@app.post("/api/predict/all")
def predict_all_models(input_data: SensorInput):
    """Executes simultaneous inference across Random Forest, Ridge Regression, and XGBoost."""
    if len(input_data.features) != 13:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 13 feature inputs, received {len(input_data.features)}"
        )
    return ml_engine.predict_all(input_data.features)

@app.post("/api/predict")
def legacy_predict(input_data: SensorInput):
    """Backwards-compatible prediction endpoint matching frontend v1/v2 format."""
    if len(input_data.features) != 13:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 13 feature inputs, received {len(input_data.features)}"
        )
    result = ml_engine.predict_all(input_data.features)
    # Default to XGBoost as primary high-performance model with consensus attached
    primary = result["models"]["xgboost"]
    return {
        "rul_hours": primary["rul_hours"],
        "health_score": primary["health_score"],
        "anomaly_probability": primary["anomaly_probability"],
        "anomaly_flag": primary["anomaly_flag"],
        "status": primary["status"],
        "status_color": primary["color"],
        "multi_model": result["models"],
        "consensus": result["consensus"]
    }

@app.post("/api/simulate/fault")
def simulate_fault(req: FaultSimulationRequest):
    """Injects industrial fault profiles into nominal baseline and calculates response across all 3 models."""
    # Nominal baseline [Mot.PV, Mot.SV, Mot.TV, Pmp.PV, Pmp.SV, Pmp.TV, Pres, PmpTemp, Vib, Temp, Load, RPM, Baro]
    baseline = [0.0018, 3.0, 30.0, 0.0012, 6.7, 32.0, 1.2, 32.0, 2.45, 25.0, 55.0, 1448.0, 1012.0]
    f = list(baseline)
    s = req.severity

    fault_description = ""
    affected_components = []

    if req.scenario == "cavitation":
        f[6] = max(0.2, f[6] - 7.5 * s)     # Sharp suction pressure collapse
        f[8] = f[8] + 2.8 * s               # High-frequency vibration spike
        f[10] = min(100.0, f[10] + 15.0 * s)# Load fluctuation
        fault_description = f"Impeller Cavitation: Severe vapor bubble collapse detected at suction stage. Pressure dropped to {f[6]:.2f} bar with violent vibration harmonics."
        affected_components = ["Impeller Vanes", "Suction Volute", "Mechanical Seal"]

    elif req.scenario == "bearing_spall":
        f[8] = f[8] + 3.9 * s               # Severe vibration spike
        f[7] = f[7] + 8.5 * s               # Bearing friction heating
        f[0] = f[0] * (1.0 + 0.4 * s)       # Motor current chatter
        fault_description = f"Bearing Outer Race Spall: High-frequency acceleration impact spikes (Vibration: {f[8]:.2f} Hz) and localized heat buildup ({f[7]:.1f}°C)."
        affected_components = ["Drive End Bearings", "Shaft Journals"]

    elif req.scenario == "thermal_overload":
        f[7] = f[7] + 18.0 * s              # Pump casing temp
        f[9] = f[9] + 14.0 * s              # Ambient/Oil temp
        f[0] = f[0] * (1.0 + 0.8 * s)       # Stator overcurrent
        f[10] = min(100.0, f[10] + 25.0 * s)# High continuous load
        fault_description = f"Motor Stator Thermal Overload: Excessive continuous load ({f[10]:.1f}%) resulting in winding thermal degradation."
        affected_components = ["Motor Stator Windings", "Insulation Class H"]

    elif req.scenario == "seal_leak":
        f[6] = max(0.5, f[6] - 3.5 * s)     # Pressure loss
        f[7] = f[7] + 6.0 * s               # Dry friction on seal face
        fault_description = f"Mechanical Seal Face Degradation: Fluid bypass resulting in barrier pressure drop ({f[6]:.2f} bar) and dry-run face heating."
        affected_components = ["Silicon Carbide Seal Face", "O-Ring Elastomers"]

    elif req.scenario == "unbalance":
        f[8] = f[8] + 2.4 * s               # 1X RPM radial vibration
        f[11] = f[11] + 18.0 * s            # Speed hunting
        fault_description = f"Dynamic Rotor Unbalance: Mass eccentricity causing 1X rotational synchronous vibration ({f[8]:.2f} Hz)."
        affected_components = ["Rotor Assembly", "Flexible Coupling"]
    else:
        fault_description = "Nominal operation restored."
        affected_components = []

    prediction_result = ml_engine.predict_all(f)
    return {
        "status": "success",
        "scenario": req.scenario,
        "severity": s,
        "description": fault_description,
        "affected_components": affected_components,
        "features": f,
        "prediction": prediction_result
    }

# ============ WEBSOCKET STREAMING ============
class StreamManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

manager = StreamManager()

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Streams live sensor telemetry + concurrent predictions from all 3 models."""
    await manager.connect(websocket)
    try:
        idx = 0
        while True:
            if trajectory_cache:
                sample = trajectory_cache[idx % len(trajectory_cache)]
                features = sample["features"]
                pred = ml_engine.predict_all(features)
                payload = {
                    "type": "telemetry_tick",
                    "index": idx,
                    "sample": sample,
                    "prediction": pred
                }
                await websocket.send_json(payload)
                idx += 1
            await asyncio.sleep(0.8)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

# ============ MOUNT INDUSTRIAL SCADA FRONTEND ============
# Serves the upgraded UI directly on http://localhost:8000/
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if not frontend_dir.exists():
    frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    print(f"[OK] Mounted industrial SCADA frontend from {frontend_dir}")