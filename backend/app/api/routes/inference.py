import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import get_settings
from app.db import repository
from app.schemas.inference import InferenceResult
from app.services.ingest import fetch_field_satellite_features
from app.services.feature_engineering import fetch_weather_features
from app.services.demo_data import synthetic_satellite_features, synthetic_weather_features
from app.services.inference_orchestrator import run_inference

router = APIRouter(tags=["inference"])
settings = get_settings()


@router.post("/infer/{field_id}", response_model=InferenceResult)
async def infer_field(field_id: str, db: Session = Depends(get_db)):
    """
    Full pipeline for one field:
      1. Pull satellite features (real, from Earth Engine - unless
         DEMO_MODE=true in .env, in which case clearly-labeled synthetic
         values are used instead so the API can be demoed pre-GEE-setup)
      2. Pull weather features (real, from Open-Meteo, unless DEMO_MODE)
      3. Run crop/stage/stress/deficit models
      4. Persist the result (so /fields/{id}/timeline has history)
      5. Return the InferenceResult JSON, with data_source flagging which
         path was used
    """
    field = repository.get_field(db, field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")

    geometry = json.loads(field.geometry_geojson)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    if settings.DEMO_MODE:
        satellite_features = synthetic_satellite_features(geometry, today)
        weather_features = synthetic_weather_features(geometry)
        data_source = "SYNTHETIC_DEMO_DATA"
    else:
        try:
            satellite_features = fetch_field_satellite_features(geometry, today)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Earth Engine request failed: {exc}. Check GEE credentials "
                        f"(see docs/GEE_SETUP.md), or set DEMO_MODE=true in .env "
                        f"to use synthetic data while you finish setup.",
            )
        weather_features = await fetch_weather_features(geometry, today)
        data_source = "earth_engine_live"

    result = run_inference(
        field_id=field_id,
        satellite_features=satellite_features,
        weather_features=weather_features,
        declared_crop=field.declared_crop,
        sowing_date=field.sowing_date,
        data_source=data_source,
    )

    repository.save_inference_record(db, result)
    return result
