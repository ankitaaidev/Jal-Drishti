"""
SQLAlchemy models.

Geometry is stored as a JSON-encoded GeoJSON string (`geometry_geojson`)
for portability across SQLite (demo) and Postgres (production). For a
production PostGIS deployment, replace this column with:
    from geoalchemy2 import Geometry
    geometry = Column(Geometry("POLYGON", srid=4326))
and add spatial indexes - left as-is here to keep the demo zero-setup.
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Float, DateTime, Text
from app.db.session import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Field(Base):
    __tablename__ = "fields"

    field_id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    geometry_geojson = Column(Text, nullable=False)  # JSON string
    area_hectares = Column(Float, nullable=True)
    farmer_name = Column(String, nullable=True)
    declared_crop = Column(String, nullable=True)
    sowing_date = Column(String, nullable=True)
    soil_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class InferenceRecord(Base):
    """Cached inference results, one row per field per inference run.
    Keeps history for the timeline chart (/fields/{id}/timeline)."""
    __tablename__ = "inference_records"

    id = Column(String, primary_key=True, default=gen_uuid)
    field_id = Column(String, nullable=False, index=True)
    date = Column(String, nullable=False)
    data_source = Column(String, default="earth_engine_live")

    crop_type = Column(String)
    crop_type_confidence = Column(Float)
    growth_stage = Column(String)
    growth_stage_confidence = Column(Float)
    stress_level = Column(String)
    stress_score = Column(Float)
    water_deficit_mm = Column(Float)
    etc_mm = Column(Float)
    effective_rainfall_mm = Column(Float)
    overall_confidence = Column(Float)

    ndvi = Column(Float)
    evi = Column(Float)
    ndwi = Column(Float)
    s1_vv_db = Column(Float)
    s1_vh_db = Column(Float)
    rainfall_7d_mm = Column(Float)

    explanation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
