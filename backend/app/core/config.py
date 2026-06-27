"""
Central configuration for Jal-Drishti backend.
All tunable constants live here so the math in services/ stays readable
and so judges/teammates can see every assumption in one place.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Demo / fallback mode ---
    # If true, services/ingest.py returns clearly-labeled SYNTHETIC satellite
    # values instead of calling Earth Engine, so the API/frontend can be
    # demoed before GEE credentials are set up. Real deployments MUST set
    # this to false - see services/demo_data.py for why this exists and how
    # it's flagged everywhere it touches the response.
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() == "true"

    # --- Google Earth Engine ---
    # GCP project that has the Earth Engine API enabled.
    GEE_PROJECT_ID: str = os.getenv("GEE_PROJECT_ID", "")
    # Path to the service-account JSON key (see docs/GEE_SETUP.md)
    GEE_SERVICE_ACCOUNT_EMAIL: str = os.getenv("GEE_SERVICE_ACCOUNT_EMAIL", "")
    GEE_SERVICE_ACCOUNT_KEY_PATH: str = os.getenv(
        "GEE_SERVICE_ACCOUNT_KEY_PATH", "./gee-key.json"
    )

    # --- Database ---
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/jaldrishti"
    )

    # --- Satellite query windows ---
    # Sentinel-2 revisit is ~5 days; Sentinel-1 ~6-12 days over India.
    # We pull a rolling window and take the least-cloudy / most recent
    # valid composite inside it.
    S2_LOOKBACK_DAYS: int = 20      # widen if cloud cover is heavy (monsoon)
    S1_LOOKBACK_DAYS: int = 24
    CLOUD_PROB_THRESHOLD: int = 40  # % - pixels above this are masked out

    # --- Water balance model constants (FAO-56 style, simplified) ---
    # Reference crop coefficient ranges by stage (Kc), FAO-56 ch.6 (Allen et al., 1998)
    KC_BY_STAGE: dict = {
        "sowing": 0.35,
        "vegetative": 0.75,
        "flowering": 1.15,
        "maturity": 0.60,
    }
    # Stage weight in the priority formula - flowering is most water-sensitive
    STAGE_WEIGHT: dict = {
        "sowing": 0.4,
        "vegetative": 0.6,
        "flowering": 1.0,
        "maturity": 0.3,
    }

    # NDVI thresholds used for rule-based growth-stage heuristic
    NDVI_SOWING_MAX: float = 0.3
    NDVI_VEGETATIVE_MAX: float = 0.55
    NDVI_FLOWERING_MAX: float = 0.8
    # above NDVI_FLOWERING_MAX or declining-after-peak => maturity

    # NDWI / soil-moisture-proxy thresholds for stress classification
    STRESS_LOW_NDWI: float = 0.0     # NDWI above this -> low stress
    STRESS_HIGH_NDWI: float = -0.25  # NDWI below this -> high stress

    # Priority formula weights (must sum to 1.0) - see services/ranking.py
    W_STRESS: float = 0.35
    W_DEFICIT: float = 0.30
    W_STAGE: float = 0.20
    W_CONFIDENCE: float = 0.15

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()