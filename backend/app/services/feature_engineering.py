"""
Feature engineering service.

Combines:
  - Satellite features from services/ingest.py (real, GEE-sourced)
  - Weather features from Open-Meteo (real, free, no API key required:
    https://open-meteo.com/en/docs - past_days parameter gives recent
    observed rainfall/temperature for any lat/lon globally, sourced from
    ECMWF/national weather service reanalysis blends)

NOTE ON IMD: India Meteorological Department station data (via data.gov.in)
is the more India-specific source, but requires a data.gov.in API key and
is station-point data (sparser spatially than a gridded product). Open-Meteo
is used here for the prototype because it's zero-setup; swapping in IMD
later only requires changing fetch_weather_features() - the rest of the
pipeline is source-agnostic.
"""
from datetime import datetime
from typing import Tuple

import httpx

from app.core.logging import get_logger
from app.schemas.inference import WeatherFeatures

logger = get_logger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _polygon_centroid(geometry_geojson: dict) -> Tuple[float, float]:
    """Rough centroid (lon, lat) for a Polygon/MultiPolygon - good enough
    for pulling a representative weather grid cell (Open-Meteo cells are
    ~9-25km, far larger than a single field)."""
    geom_type = geometry_geojson["type"]
    coords = geometry_geojson["coordinates"]

    def flatten_ring(ring):
        return ring  # list of [lon, lat]

    points = []
    if geom_type == "Polygon":
        points = flatten_ring(coords[0])
    elif geom_type == "MultiPolygon":
        for poly in coords:
            points.extend(flatten_ring(poly[0]))
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return sum(lons) / len(lons), sum(lats) / len(lats)


async def fetch_weather_features(
    geometry_geojson: dict,
    as_of_date: str = None,
) -> WeatherFeatures:
    """Pulls real recent rainfall + temperature for the field's location."""
    lon, lat = _polygon_centroid(geometry_geojson)

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_mean",
        "past_days": 14,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.error("Open-Meteo fetch failed: %s", exc)
        return WeatherFeatures(
            rainfall_7d_mm=None, rainfall_14d_mm=None,
            temp_avg_7d_c=None, source="open-meteo (failed)",
        )

    daily = data.get("daily", {})
    precip = daily.get("precipitation_sum", [])
    temps = daily.get("temperature_2m_mean", [])

    # past_days=14 + forecast_days=1 -> last element is "today"/forecast,
    # the 14 before it are the observed history we want.
    rainfall_14d = sum(v for v in precip[-15:-1] if v is not None)
    rainfall_7d = sum(v for v in precip[-8:-1] if v is not None)
    temp_vals = [t for t in temps[-8:-1] if t is not None]
    temp_avg_7d = sum(temp_vals) / len(temp_vals) if temp_vals else None

    return WeatherFeatures(
        rainfall_7d_mm=round(rainfall_7d, 1),
        rainfall_14d_mm=round(rainfall_14d, 1),
        temp_avg_7d_c=round(temp_avg_7d, 1) if temp_avg_7d is not None else None,
        source="open-meteo",
    )
