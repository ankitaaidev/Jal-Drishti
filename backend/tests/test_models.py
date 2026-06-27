"""
Unit tests for math-only services (no GEE/network credentials required).
Run with: pytest backend/tests/test_models.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.deficit_model import estimate_et0_hargreaves, estimate_water_deficit
from app.services.stress_model import estimate_moisture_stress
from app.schemas.inference import SatelliteFeatures, WeatherFeatures


def test_et0_increases_with_temperature():
    cool = estimate_et0_hargreaves(20.0)
    hot = estimate_et0_hargreaves(35.0)
    assert hot > cool


def test_water_deficit_positive_when_dry():
    features = SatelliteFeatures(date="2026-06-27")
    weather = WeatherFeatures(rainfall_7d_mm=2.0, temp_avg_7d_c=34.0)
    result = estimate_water_deficit(features, weather, growth_stage="flowering")
    assert result["water_deficit_mm"] > 0


def test_water_deficit_negative_when_wet():
    features = SatelliteFeatures(date="2026-06-27")
    weather = WeatherFeatures(rainfall_7d_mm=90.0, temp_avg_7d_c=27.0)
    result = estimate_water_deficit(features, weather, growth_stage="vegetative")
    assert result["water_deficit_mm"] < 0


def test_water_deficit_none_when_missing_weather():
    features = SatelliteFeatures(date="2026-06-27")
    weather = WeatherFeatures(rainfall_7d_mm=None, temp_avg_7d_c=None)
    result = estimate_water_deficit(features, weather, growth_stage="flowering")
    assert result["water_deficit_mm"] is None


def test_flowering_has_highest_etc():
    weather = WeatherFeatures(rainfall_7d_mm=10.0, temp_avg_7d_c=30.0)
    features = SatelliteFeatures(date="2026-06-27")
    etc_values = {}
    for stage in ["sowing", "vegetative", "flowering", "maturity"]:
        result = estimate_water_deficit(features, weather, growth_stage=stage)
        etc_values[stage] = result["etc_mm"]
    assert etc_values["flowering"] == max(etc_values.values())


def test_stress_high_when_dry_and_low_ndwi():
    features = SatelliteFeatures(date="2026-06-27", ndwi=-0.30)
    weather = WeatherFeatures(rainfall_7d_mm=1.0)
    level, score = estimate_moisture_stress(features, weather)
    assert level == "high"
    assert score > 0.66


def test_stress_low_when_wet_and_high_ndwi():
    features = SatelliteFeatures(date="2026-06-27", ndwi=0.15)
    weather = WeatherFeatures(rainfall_7d_mm=80.0)
    level, score = estimate_moisture_stress(features, weather)
    assert level == "low"
    assert score < 0.33


def test_stress_unknown_when_no_ndwi():
    features = SatelliteFeatures(date="2026-06-27", ndwi=None)
    weather = WeatherFeatures(rainfall_7d_mm=10.0)
    level, score = estimate_moisture_stress(features, weather)
    assert level == "unknown"
    assert score == 0.0
