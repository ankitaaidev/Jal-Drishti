from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db import repository
from app.schemas.inference import InferenceResult, SatelliteFeatures, WeatherFeatures, PriorityListItem
from app.services.ranking import rank_fields

router = APIRouter(tags=["irrigation"])


@router.get("/priority-list", response_model=List[PriorityListItem])
def get_priority_list(db: Session = Depends(get_db)):
    """
    Returns all fields' latest cached inference results, ranked by
    irrigation priority. Run POST /infer/{field_id} for each field first
    (or via a batch refresh script) so there's data to rank.
    """
    latest_records = repository.get_all_latest_inferences(db)
    fields = {f.field_id: f.name for f in repository.list_fields(db)}

    results = []
    for r in latest_records:
        results.append(
            InferenceResult(
                field_id=r.field_id,
                date=r.date,
                data_source=r.data_source or "earth_engine_live",
                crop_type=r.crop_type,
                crop_type_confidence=r.crop_type_confidence,
                growth_stage=r.growth_stage,
                growth_stage_confidence=r.growth_stage_confidence,
                stress_level=r.stress_level,
                stress_score=r.stress_score,
                water_deficit_mm=r.water_deficit_mm,
                etc_mm=r.etc_mm,
                effective_rainfall_mm=r.effective_rainfall_mm,
                overall_confidence=r.overall_confidence,
                satellite_features=SatelliteFeatures(
                    date=r.date, ndvi=r.ndvi, evi=r.evi, ndwi=r.ndwi,
                    s1_vv_db=r.s1_vv_db, s1_vh_db=r.s1_vh_db,
                ),
                weather_features=WeatherFeatures(rainfall_7d_mm=r.rainfall_7d_mm),
                explanation=r.explanation,
            )
        )

    return rank_fields(results, fields)
