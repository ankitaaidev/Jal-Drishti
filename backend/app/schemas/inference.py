from typing import Optional, Dict
from pydantic import BaseModel, Field


class SatelliteFeatures(BaseModel):
    """Raw indices pulled from Earth Engine for one field, one date window."""
    date: str
    s2_image_date: Optional[str] = Field(
        None, description="Actual acquisition date of the Sentinel-2 scene used"
    )
    s1_image_date: Optional[str] = Field(
        None, description="Actual acquisition date of the Sentinel-1 scene used"
    )
    ndvi: Optional[float] = None
    evi: Optional[float] = None
    ndwi: Optional[float] = None
    s1_vv_db: Optional[float] = Field(None, description="Sentinel-1 VV backscatter, dB")
    s1_vh_db: Optional[float] = Field(None, description="Sentinel-1 VH backscatter, dB")
    s1_vv_vh_diff_db: Optional[float] = Field(
        None,
        description="VV minus VH, in dB. This is the log-domain equivalent of a "
                    "linear VV/VH ratio (since dB is already a log scale, "
                    "ratio-in-linear-space = subtraction-in-dB-space). "
                    "Higher values = more vertical/structural scattering "
                    "(e.g. cereal stalks); lower/negative = more volume "
                    "scattering (e.g. dense canopy or flooded paddy)."
    )
    cloud_pct_in_window: Optional[float] = None
    pixel_count: Optional[int] = Field(
        None, description="Valid (non-masked) pixels used in the field-mean. "
                           "Low counts -> lower confidence."
    )


class WeatherFeatures(BaseModel):
    rainfall_7d_mm: Optional[float] = None
    rainfall_14d_mm: Optional[float] = None
    temp_avg_7d_c: Optional[float] = None
    source: str = "open-meteo"  # see services/feature_engineering.py for swap-in note


class InferenceResult(BaseModel):
    field_id: str
    date: str

    data_source: str = Field(
        "earth_engine_live",
        description="'earth_engine_live' for real Sentinel data, "
                    "'SYNTHETIC_DEMO_DATA' when DEMO_MODE is enabled. "
                    "Always check this before treating numbers as real."
    )

    crop_type: str
    crop_type_confidence: float = Field(..., ge=0, le=1)
    crop_type_method: str = "rule-based-fallback"

    growth_stage: str
    growth_stage_confidence: float = Field(..., ge=0, le=1)

    stress_level: str  # "low" | "medium" | "high"
    stress_score: float = Field(..., ge=0, le=1)

    water_deficit_mm: float = Field(
        ..., description="Estimated crop water deficit over the period, mm. "
                          "Positive = crop needs more water than it received."
    )
    etc_mm: float = Field(..., description="Estimated crop water requirement (ETc), mm")
    effective_rainfall_mm: float

    overall_confidence: float = Field(..., ge=0, le=1)
    priority_score: Optional[float] = Field(
        None, description="Set by ranking.py once compared across fields"
    )

    satellite_features: SatelliteFeatures
    weather_features: WeatherFeatures

    explanation: str = Field(
        ..., description="Human-readable reason string for the dashboard/alert"
    )


class PriorityListItem(BaseModel):
    field_id: str
    field_name: str
    priority_rank: int
    priority_score: float
    stress_level: str
    water_deficit_mm: float
    growth_stage: str
    crop_type: str
    confidence: float
    explanation: str

