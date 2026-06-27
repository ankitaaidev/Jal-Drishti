import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.field import FieldCreate, FieldOut
from app.db import repository

router = APIRouter(tags=["fields"])


@router.post("/fields", response_model=FieldOut)
def create_field(payload: FieldCreate, db: Session = Depends(get_db)):
    field = repository.create_field(db, payload)
    return FieldOut(
        field_id=field.field_id,
        name=field.name,
        geometry_geojson=json.loads(field.geometry_geojson),
        area_hectares=field.area_hectares,
        farmer_name=field.farmer_name,
        declared_crop=field.declared_crop,
        sowing_date=field.sowing_date,
        soil_type=field.soil_type,
        created_at=field.created_at.isoformat(),
    )


@router.get("/fields", response_model=List[FieldOut])
def list_fields(db: Session = Depends(get_db)):
    fields = repository.list_fields(db)
    return [
        FieldOut(
            field_id=f.field_id,
            name=f.name,
            geometry_geojson=json.loads(f.geometry_geojson),
            area_hectares=f.area_hectares,
            farmer_name=f.farmer_name,
            declared_crop=f.declared_crop,
            sowing_date=f.sowing_date,
            soil_type=f.soil_type,
            created_at=f.created_at.isoformat(),
        )
        for f in fields
    ]


@router.get("/fields/{field_id}", response_model=FieldOut)
def get_field(field_id: str, db: Session = Depends(get_db)):
    field = repository.get_field(db, field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return FieldOut(
        field_id=field.field_id,
        name=field.name,
        geometry_geojson=json.loads(field.geometry_geojson),
        area_hectares=field.area_hectares,
        farmer_name=field.farmer_name,
        declared_crop=field.declared_crop,
        sowing_date=field.sowing_date,
        soil_type=field.soil_type,
        created_at=field.created_at.isoformat(),
    )


@router.get("/fields/{field_id}/timeline")
def get_field_timeline(field_id: str, db: Session = Depends(get_db)):
    field = repository.get_field(db, field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    records = repository.get_inference_timeline(db, field_id)
    return [
        {
            "date": r.date,
            "ndvi": r.ndvi,
            "ndwi": r.ndwi,
            "s1_vv_db": r.s1_vv_db,
            "rainfall_7d_mm": r.rainfall_7d_mm,
            "stress_score": r.stress_score,
            "water_deficit_mm": r.water_deficit_mm,
            "growth_stage": r.growth_stage,
        }
        for r in records
    ]
