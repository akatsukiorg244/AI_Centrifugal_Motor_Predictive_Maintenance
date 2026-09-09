"""
FastAPI Backend — REST API + WebSocket for Real-Time Pump Digital Twin
Serves ML predictions and streams live sensor data to frontend.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import asyncio
from typing import List

from model import PumpMLModel
from data_streamer import PumpDataStreamer

app = FastAPI(title="Pump Digital Twin API", version="2.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML model & data streamer
print("🚀 Initializing Pump Digital Twin Backend...")
ml_model = PumpMLModel("pump_model.pkl")
streamer = PumpDataStreamer("converted_file.csv")
print("✅ Backend ready!")

# ============ REST API ============

class SensorInput(BaseModel):
    features: List[float]

@app.get("/")
def root():
    return {
        "project": "Centrifugal Pump Digital Twin",
        "layers": {"ml": "RandomForest 3-model ensemble", "backend": "FastAPI + WebSocket", "frontend": "Three.js + MATLAB UI"},
        "status": "running"
    }

@app.get("/api/model/stats")
def model_stats():
    """Judge endpoint: Show model training performance."""
    return ml_model.get_model_stats()

@app.get("/api/model/importance")
def feature_importance():
    """Judge endpoint: Explainable AI — top features."""
    return ml_model.get_feature_importance()

@app.post("/api/predict")
def predict(input_data: SensorInput):
    """REST endpoint for single prediction."""
    if len(input_data.features) != 13:
        return {"error": f"Expected 13 features, got {len(input_data.features)}"}
    return ml_model.predict(input_data.features)

@app.get("/api/stream/current")
def current_reading():
    """Get current sensor row without WebSocket."""
    row = streamer.get_current_row()
    prediction = ml_model.predict(list(row["features"].values()))
    return {"sensor": row, "prediction": prediction}

# ============ WEBSOCKET ============

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket: Streams real sensor data + ML predictions every 500ms.
    Frontend receives: {sensor_data, prediction, feature_importance}
    """
    await manager.connect(websocket)
    try:
        while True:
            row = streamer.next()
            features = list(row["features"].values())
            prediction = ml_model.predict(features)

            payload = {
                "type": "live_update",
                "timestamp": row["timestamp"],
                "sensor": row["features"],
                "prediction": prediction,
                "model_stats": ml_model.get_model_stats(),
                "progress": {
                    "current": row["index"],
                    "total": row["total"],
                    "percent": round((row["index"] / row["total"]) * 100, 1)
                }
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)  # 500ms update rate

    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)