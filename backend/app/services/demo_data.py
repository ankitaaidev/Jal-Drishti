"""
DEMO MODE fallback data.

This module exists for ONE reason: so you can run and click through the
whole API/frontend before Google Earth Engine credentials are set up,
without the demo being blocked on a 15-minute one-time GCP setup.

EVERY value this returns is synthetic. It is never silently substituted -
the API response always includes "data_source": "SYNTHETIC_DEMO_DATA" so
nobody (including future-you, mid-demo) mistakes it for a real satellite
reading. Do not present numbers from this module as real measurements.

Values are deterministic per field_id (seeded hash) so repeated calls
during a demo show a believable, stable dashboard rather than random
flicker, while still varying sensibly across different fields/polygons.
"""
import hashlib
from datetime import datetime

from app.schemas.inference import SatelliteFeatures, WeatherFeatures


def _seed_from(geometry_geojson: dict) -> float:
    """Deterministic [0,1) value derived from the polygon's coordinates,
    so the same field always gets the same synthetic reading."""
    raw = str(geometry_geojson.get("coordinates", ""))
    digest = hashlib.md5(raw.encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def synthetic_satellite_features(
    geometry_geojson: dict, as_of_date: str = None
) -> SatelliteFeatures:
    seed = _seed_from(geometry_geojson)
    date = as_of_date or datetime.utcnow().strftime("%Y-%m-%d")

    # Spread the seed across a plausible NDVI range (0.15 - 0.75) so
    # different demo fields land in different stress/stage buckets.
    ndvi = round(0.15 + seed * 0.6, 3)
    ndwi = round(-0.30 + seed * 0.45, 3)  # roughly correlated with ndvi via shared seed
    vv_db = round(-12.0 - seed * 6, 2)
    vh_db = round(-18.0 - seed * 4, 2)

    return SatelliteFeatures(
        date=date,
        s2_image_date=date,
        s1_image_date=date,
        ndvi=ndvi,
        evi=round(ndvi * 0.85, 3),
        ndwi=ndwi,
        s1_vv_db=vv_db,
        s1_vh_db=vh_db,
        s1_vv_vh_diff_db=round(vv_db - vh_db, 2),
        cloud_pct_in_window=round(seed * 30, 1),
        pixel_count=150,
    )


def synthetic_weather_features(geometry_geojson: dict) -> WeatherFeatures:
    seed = _seed_from(geometry_geojson)
    rainfall_7d = round(seed * 60, 1)        # 0-60mm spread
    temp_avg = round(26 + seed * 9, 1)       # 26-35C spread
    return WeatherFeatures(
        rainfall_7d_mm=rainfall_7d,
        rainfall_14d_mm=round(rainfall_7d * 1.8, 1),
        temp_avg_7d_c=temp_avg,
        source="SYNTHETIC_DEMO_DATA",
    )
