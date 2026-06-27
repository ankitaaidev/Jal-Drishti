import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db import repository
from app.services.ingest import fetch_field_satellite_features

router = APIRouter(tags=["satellite"])


@router.post("/satellite/{field_id}/refresh")
def refresh_satellite_features(field_id: str, db: Session = Depends(get_db)):
    """
    Pulls fresh Sentinel-1/2 features for a field directly from Earth
    Engine, without running the full inference pipeline. Useful for
    debugging GEE connectivity/credentials independent of the model logic.
    """
    field = repository.get_field(db, field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")

    geometry = json.loads(field.geometry_geojson)
    try:
        features = fetch_field_satellite_features(geometry)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Earth Engine request failed: {exc}. Check GEE credentials in .env "
                    f"(see docs/GEE_SETUP.md).",
        )
    return features
