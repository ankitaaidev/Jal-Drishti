"""
Inference orchestrator. This is what api/routes/inference.py calls.
It does NOT talk to Earth Engine or Open-Meteo directly - it receives
already-fetched SatelliteFeatures/WeatherFeatures and runs the model
services in order, then assembles the final, explainable result.
"""
from app.schemas.inference import InferenceResult, SatelliteFeatures, WeatherFeatures
from app.services import crop_model, deficit_model
from app.services.stress_model import estimate_growth_stage, estimate_moisture_stress


def _overall_confidence(
    crop_conf: float, stage_conf: float, stress_score_present: bool,
    pixel_count: int = None,
) -> float:
    """
    Blends the per-component confidences into one number for the dashboard
    badge. Weighted toward stage/stress since those drive the irrigation
    decision; crop type is informative but not safety-critical here.
    Low pixel_count (small/cloud-obscured field) drags confidence down -
    a 3-pixel field mean is noisier than a 300-pixel one.
    """
    base = 0.3 * crop_conf + 0.4 * stage_conf + 0.3 * (1.0 if stress_score_present else 0.0)
    if pixel_count is not None and pixel_count < 10:
        base *= 0.7  # penalize tiny/noisy field samples
    return round(min(base, 1.0), 3)


def _build_explanation(
    crop_type: str, stage: str, stress_level: str,
    water_deficit_mm: float, ndvi: float, ndwi: float,
) -> str:
    """Human-readable reasoning string shown on the dashboard/alert card.
    Every clause references a real computed number - no templated filler
    that doesn't trace back to data."""
    parts = []
    parts.append(f"Crop identified as {crop_type.replace('_', ' ')} at {stage} stage")
    if ndvi is not None:
        parts.append(f"(NDVI={ndvi:.2f})")
    if ndwi is not None:
        parts.append(f"(NDWI={ndwi:.2f})")
    parts.append(". Moisture stress: " + stress_level + ".")
    if water_deficit_mm is not None:
        if water_deficit_mm > 0:
            parts.append(f" Estimated water deficit of {water_deficit_mm}mm over the period — irrigation recommended.")
        else:
            parts.append(f" Estimated water surplus of {abs(water_deficit_mm)}mm — no irrigation needed now.")
    else:
        parts.append(" Water balance could not be computed (missing weather data).")
    return " ".join(parts)


def run_inference(
    field_id: str,
    satellite_features: SatelliteFeatures,
    weather_features: WeatherFeatures,
    declared_crop: str = None,
    sowing_date: str = None,
    data_source: str = "earth_engine_live",
) -> InferenceResult:
    current_date = satellite_features.date

    crop_type, crop_conf, crop_method = crop_model.classify_crop(
        satellite_features, declared_crop
    )

    growth_stage, stage_conf = estimate_growth_stage(
        satellite_features, sowing_date, current_date
    )

    stress_level, stress_score = estimate_moisture_stress(
        satellite_features, weather_features
    )

    deficit_result = deficit_model.estimate_water_deficit(
        satellite_features, weather_features, growth_stage
    )

    overall_conf = _overall_confidence(
        crop_conf, stage_conf,
        stress_score_present=(stress_level != "unknown"),
        pixel_count=satellite_features.pixel_count,
    )

    explanation = _build_explanation(
        crop_type, growth_stage, stress_level,
        deficit_result["water_deficit_mm"],
        satellite_features.ndvi, satellite_features.ndwi,
    )

    return InferenceResult(
        field_id=field_id,
        date=current_date,
        data_source=data_source,
        crop_type=crop_type,
        crop_type_confidence=crop_conf,
        crop_type_method=crop_method,
        growth_stage=growth_stage,
        growth_stage_confidence=stage_conf,
        stress_level=stress_level,
        stress_score=stress_score,
        water_deficit_mm=deficit_result["water_deficit_mm"] or 0.0,
        etc_mm=deficit_result["etc_mm"] or 0.0,
        effective_rainfall_mm=deficit_result["effective_rainfall_mm"] or 0.0,
        overall_confidence=overall_conf,
        satellite_features=satellite_features,
        weather_features=weather_features,
        explanation=explanation,
    )
