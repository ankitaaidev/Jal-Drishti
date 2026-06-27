import json
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import Field, InferenceRecord
from app.schemas.field import FieldCreate
from app.schemas.inference import InferenceResult
from app.utils.geo import polygon_area_hectares


def create_field(db: Session, payload: FieldCreate) -> Field:
    area = polygon_area_hectares(payload.geometry_geojson)
    field = Field(
        name=payload.name,
        geometry_geojson=json.dumps(payload.geometry_geojson),
        area_hectares=area,
        farmer_name=payload.farmer_name,
        declared_crop=payload.declared_crop,
        sowing_date=payload.sowing_date,
        soil_type=payload.soil_type,
    )
    db.add(field)
    db.commit()
    db.refresh(field)
    return field


def get_field(db: Session, field_id: str) -> Optional[Field]:
    return db.query(Field).filter(Field.field_id == field_id).first()


def list_fields(db: Session) -> List[Field]:
    return db.query(Field).all()


def save_inference_record(db: Session, result: InferenceResult) -> InferenceRecord:
    record = InferenceRecord(
        field_id=result.field_id,
        date=result.date,
        data_source=result.data_source,
        crop_type=result.crop_type,
        crop_type_confidence=result.crop_type_confidence,
        growth_stage=result.growth_stage,
        growth_stage_confidence=result.growth_stage_confidence,
        stress_level=result.stress_level,
        stress_score=result.stress_score,
        water_deficit_mm=result.water_deficit_mm,
        etc_mm=result.etc_mm,
        effective_rainfall_mm=result.effective_rainfall_mm,
        overall_confidence=result.overall_confidence,
        ndvi=result.satellite_features.ndvi,
        evi=result.satellite_features.evi,
        ndwi=result.satellite_features.ndwi,
        s1_vv_db=result.satellite_features.s1_vv_db,
        s1_vh_db=result.satellite_features.s1_vh_db,
        rainfall_7d_mm=result.weather_features.rainfall_7d_mm,
        explanation=result.explanation,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_latest_inference(db: Session, field_id: str) -> Optional[InferenceRecord]:
    return (
        db.query(InferenceRecord)
        .filter(InferenceRecord.field_id == field_id)
        .order_by(InferenceRecord.created_at.desc())
        .first()
    )


def get_inference_timeline(db: Session, field_id: str) -> List[InferenceRecord]:
    return (
        db.query(InferenceRecord)
        .filter(InferenceRecord.field_id == field_id)
        .order_by(InferenceRecord.date.asc())
        .all()
    )


def get_all_latest_inferences(db: Session) -> List[InferenceRecord]:
    """One latest record per field - used for /priority-list and /alerts."""
    fields = list_fields(db)
    out = []
    for f in fields:
        latest = get_latest_inference(db, f.field_id)
        if latest:
            out.append(latest)
    return out
