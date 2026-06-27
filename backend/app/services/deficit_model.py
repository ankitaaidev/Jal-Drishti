"""
Crop water deficit estimation.

Uses a SIMPLIFIED version of the FAO-56 crop water balance method
(Allen, Pereira, Raes & Smith, 1998, "Crop evapotranspiration -
Guidelines for computing crop water requirements", FAO Irrigation and
Drainage Paper 56). This is the standard reference method used globally
for irrigation scheduling, including by India's own agromet advisory
systems.

Real FAO-56 needs daily reference evapotranspiration (ET0) from a
Penman-Monteith calculation requiring solar radiation, wind speed,
humidity, and temperature. For a hackathon prototype without a full
weather station feed, we substitute a TEMPERATURE-BASED ET0 proxy
(Hargreaves-Samani, 1985) - a well-established simplification that only
needs min/max/mean temperature, which Open-Meteo provides for free.

This means: the *formula* is the real, citable FAO-56/Hargreaves method.
The *inputs* are a reasonable proxy, not full meteorological-station-grade
data. That distinction should be stated plainly in any demo or report.
"""
from typing import Optional

from app.core.config import get_settings
from app.schemas.inference import SatelliteFeatures, WeatherFeatures

settings = get_settings()


def estimate_et0_hargreaves(
    temp_avg_c: float,
    temp_range_c: float = 10.0,  # placeholder daily range if min/max unavailable
    extraterrestrial_radiation_proxy: float = 15.0,  # MJ/m2/day, rough tropical mid-latitude value
) -> float:
    """
    Hargreaves-Samani (1985) reference ET0, mm/day:
        ET0 = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)

    Ra (extraterrestrial radiation) properly depends on latitude and
    day-of-year; we use a fixed mid-range value here as a v1 simplification
    appropriate for most of India's latitude band (8-35°N) in growing
    season. This is the single biggest accuracy compromise in the model -
    flagged here and in the README for anyone improving v2.
    """
    et0 = 0.0023 * extraterrestrial_radiation_proxy * (temp_avg_c + 17.8) * (temp_range_c ** 0.5)
    return max(0.0, round(et0, 2))


def estimate_water_deficit(
    features: SatelliteFeatures,
    weather: WeatherFeatures,
    growth_stage: str,
    period_days: int = 7,
) -> dict:
    """
    Returns dict with etc_mm, effective_rainfall_mm, water_deficit_mm.

    ETc (crop water requirement) = ET0 * Kc
      where Kc (crop coefficient) varies by growth stage per FAO-56 Table 12
      (generic row-crop values; configured in core/config.KC_BY_STAGE).

    Effective rainfall: not all rainfall is usable by the crop (runoff,
    deep percolation). We apply the commonly used 80% effectiveness factor
    for the kind of moderate rainfall events typical outside extreme
    monsoon bursts (FAO Irrigation & Drainage Paper 25 guidance range).

    Water deficit = ETc - effective_rainfall
      Positive => crop needed more water than it received => irrigate.
      Negative => surplus, no irrigation need this period.
    """
    temp_avg = weather.temp_avg_7d_c
    rainfall = weather.rainfall_7d_mm or 0.0

    if temp_avg is None:
        # Can't compute ET0 without temperature - return None deficit,
        # caller should treat this field as low-confidence / needs-data.
        return {
            "etc_mm": None,
            "effective_rainfall_mm": None,
            "water_deficit_mm": None,
        }

    et0_daily = estimate_et0_hargreaves(temp_avg)
    et0_period = et0_daily * period_days

    kc = settings.KC_BY_STAGE.get(growth_stage, 0.7)  # 0.7 = generic mid-season fallback
    etc_period = round(et0_period * kc, 1)

    EFFECTIVE_RAINFALL_FACTOR = 0.8
    effective_rainfall = round(rainfall * EFFECTIVE_RAINFALL_FACTOR, 1)

    water_deficit = round(etc_period - effective_rainfall, 1)

    return {
        "etc_mm": etc_period,
        "effective_rainfall_mm": effective_rainfall,
        "water_deficit_mm": water_deficit,
    }
