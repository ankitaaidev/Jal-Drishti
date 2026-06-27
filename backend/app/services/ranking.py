"""
Irrigation priority ranking.

priority = W_STRESS*stress_score
         + W_DEFICIT*normalized_deficit
         + W_STAGE*stage_weight
         + W_CONFIDENCE*confidence

Weights are configured in core/config.py (sum to 1.0) and were chosen to
match the brief's example formula (0.35/0.30/0.20/0.15), with one
deliberate change: confidence is ADDED (not used as a discount), so a
high-need-but-uncertain field still surfaces for human review rather than
being silently buried. This is a defensible product decision worth
stating out loud in a demo: "we'd rather a human double-check an uncertain
high-stress field than have the algorithm hide it."
"""
from typing import List

from app.core.config import get_settings
from app.schemas.inference import InferenceResult, PriorityListItem

settings = get_settings()


def _normalize_deficit(deficit_mm: float, all_deficits: List[float]) -> float:
    """Min-max normalize water deficit across the current field set to 0..1.
    Deficits below 0 (surplus) are clamped to 0 contribution."""
    if not all_deficits:
        return 0.0
    lo, hi = min(all_deficits), max(all_deficits)
    if hi == lo:
        return 0.0 if deficit_mm <= 0 else 1.0
    clamped = max(deficit_mm, 0.0)
    return round((clamped - max(lo, 0.0)) / (hi - max(lo, 0.0)), 3) if hi > 0 else 0.0


def rank_fields(
    results: List[InferenceResult],
    field_names: dict,  # field_id -> name
) -> List[PriorityListItem]:
    valid_results = [r for r in results if r.water_deficit_mm is not None]
    all_deficits = [r.water_deficit_mm for r in valid_results]

    scored: List[PriorityListItem] = []
    for r in results:
        if r.water_deficit_mm is None:
            # Missing weather data - still show the field, ranked last,
            # rather than silently dropping it.
            priority = -1.0
        else:
            stage_weight = settings.STAGE_WEIGHT.get(r.growth_stage, 0.5)
            norm_deficit = _normalize_deficit(r.water_deficit_mm, all_deficits)

            priority = round(
                settings.W_STRESS * r.stress_score
                + settings.W_DEFICIT * norm_deficit
                + settings.W_STAGE * stage_weight
                + settings.W_CONFIDENCE * r.overall_confidence,
                4,
            )

        scored.append(
            PriorityListItem(
                field_id=r.field_id,
                field_name=field_names.get(r.field_id, r.field_id),
                priority_rank=0,  # set after sort
                priority_score=priority,
                stress_level=r.stress_level,
                water_deficit_mm=r.water_deficit_mm or 0.0,
                growth_stage=r.growth_stage,
                crop_type=r.crop_type,
                confidence=r.overall_confidence,
                explanation=r.explanation,
            )
        )

    scored.sort(key=lambda x: x.priority_score, reverse=True)
    for i, item in enumerate(scored, start=1):
        item.priority_rank = i

    return scored
