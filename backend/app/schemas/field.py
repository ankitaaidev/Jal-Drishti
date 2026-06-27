from typing import Optional, List
from pydantic import BaseModel, Field


class FieldCreate(BaseModel):
    """Payload to register a new field polygon."""
    name: str
    geometry_geojson: dict = Field(
        ..., description="GeoJSON Polygon/MultiPolygon for the field boundary"
    )
    farmer_name: Optional[str] = None
    declared_crop: Optional[str] = Field(
        None, description="Optional farmer-declared crop, used as a prior/sanity check"
    )
    sowing_date: Optional[str] = Field(
        None, description="ISO date, e.g. 2026-06-01. Improves stage estimation."
    )
    soil_type: Optional[str] = None


class FieldOut(BaseModel):
    field_id: str
    name: str
    geometry_geojson: dict
    area_hectares: float
    farmer_name: Optional[str]
    declared_crop: Optional[str]
    sowing_date: Optional[str]
    soil_type: Optional[str]
    created_at: str
