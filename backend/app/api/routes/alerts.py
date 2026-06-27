from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.db import repository
from app.schemas.alert import AlertOut

router = APIRouter(tags=["alerts"])


def _severity_for(stress_level: str, water_deficit_mm: float) -> str:
    if stress_level == "high" or water_deficit_mm > 20:
        return "critical"
    if stress_level == "medium" or water_deficit_mm > 0:
        return "warning"
    return "info"


@router.get("/alerts", response_model=List[AlertOut])
def get_alerts(db: Session = Depends(get_db)):
    """
    Derives alerts directly from the latest inference per field -
    no separate alert-storage table needed since alerts are just a
    severity-filtered, human-readable view of the inference results.
    Only returns fields with stress_level in {medium, high} or a
    positive water deficit (i.e. fields that actually need attention).
    """
    fields = {f.field_id: f.name for f in repository.list_fields(db)}
    latest_records = repository.get_all_latest_inferences(db)

    alerts = []
    for r in latest_records:
        if r.stress_level not in ("medium", "high") and (r.water_deficit_mm or 0) <= 0:
            continue

        severity = _severity_for(r.stress_level, r.water_deficit_mm or 0)
        alerts.append(
            AlertOut(
                field_id=r.field_id,
                field_name=fields.get(r.field_id, r.field_id),
                severity=severity,
                title=f"{r.stress_level.title()} moisture stress - {fields.get(r.field_id, r.field_id)}",
                message=r.explanation,
                water_deficit_mm=r.water_deficit_mm or 0.0,
                stress_level=r.stress_level,
                confidence=r.overall_confidence,
                data_source=r.data_source or "earth_engine_live",
                created_at=r.created_at.isoformat() if r.created_at else datetime.utcnow().isoformat(),
            )
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 3))
    return alerts
