from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class SensorReading(BaseModel):
    timestamp: datetime
    # 13 features for the centrifugal pump
    flow_rate: float              # m³/h
    suction_pressure: float       # bar
    discharge_pressure: float     # bar
    motor_current: float          # A
    motor_voltage: float          # V
    motor_temperature: float      # °C
    bearing_temperature: float      # °C
    vibration_x: float            # mm/s
    vibration_y: float            # mm/s
    vibration_z: float            # mm/s
    pump_speed: float             # RPM
    oil_temperature: float          # °C
    seal_pressure: float          # bar


class PredictionResponse(BaseModel):
    timestamp: datetime
    health_score: float           # 0-100
    predicted_rul: float          # Remaining Useful Life (hours)
    anomaly_probability: float    # 0-1
    health_status: str            # Healthy, Warning, Critical
    feature_importance: Optional[dict] = None


class RealTimeData(BaseModel):
    readings: List[SensorReading]
    latest_prediction: Optional[PredictionResponse] = None


class PumpState(BaseModel):
    health_status: str
    rpm: float
    flow_rate: float
    temperature: float
    vibration_level: float
    color_hex: str  # For 3D visualizationS