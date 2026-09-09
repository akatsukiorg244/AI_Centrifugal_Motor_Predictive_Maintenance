"""
Data Streamer — Reads your real CSV and streams rows via WebSocket
Simulates live sensor feed for the digital twin.
"""

import pandas as pd
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator

class PumpDataStreamer:
    FEATURE_COLS = [
        'A_ACR_Mot.PV', 'A_ACR_Mot.SV', 'A_ACR_Mot.TV',
        'A_ACR_Pmp.PV', 'A_ACR_Pmp.SV', 'A_ACR_Pmp.TV',
        'A_Pres.PV', 'A_Temp.PV', 'Vibration_hz',
        'Temperature', 'Load_Percent', 'RPM', 'Barometer'
    ]

    def __init__(self, csv_path: str = "converted_file (1).csv"):
        self.df = pd.read_csv(csv_path)
        self.df[self.FEATURE_COLS] = self.df[self.FEATURE_COLS].ffill().bfill()
        self.index = 0
        self.total = len(self.df)

    def get_current_row(self) -> dict:
        """Get current sensor row as dict."""
        row = self.df.iloc[self.index]
        return {
            "timestamp": str(row.get('timestamp', f"t-{self.index}")),
            "features": {col: float(row[col]) for col in self.FEATURE_COLS},
            "index": self.index,
            "total": self.total
        }

    def next(self) -> dict:
        """Advance to next row (loops for demo)."""
        data = self.get_current_row()
        self.index = (self.index + 1) % self.total
        return data

    def reset(self):
        self.index = 0

    async def stream(self, delay_ms: float = 500) -> AsyncGenerator[dict, None]:
        """Async generator for WebSocket streaming."""
        while True:
            yield self.next()
            await asyncio.sleep(delay_ms / 1000)