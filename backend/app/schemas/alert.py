from typing import Optional
from pydantic import BaseModel


class AlertOut(BaseModel):
    field_id: str
    field_name: str
    severity: str  # "info" | "warning" | "critical"
    title: str
    message: str
    water_deficit_mm: float
    stress_level: str
    confidence: float
    data_source: str = "earth_engine_live"
    created_at: str
