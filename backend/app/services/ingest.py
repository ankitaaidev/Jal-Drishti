"""
Google Earth Engine connector.

This is the only file that talks to GEE. Everything downstream
(feature_engineering, crop_model, stress_model, deficit_model) consumes
plain Python floats, so the rest of the codebase has zero GEE dependency
and is easy to test or swap a different imagery source into later.

DATA SOURCES (all real, all free, all public):
  - Sentinel-2 SR Harmonized (COPERNICUS/S2_SR_HARMONIZED)
      -> NDVI, EVI, NDWI
  - Sentinel-1 GRD (COPERNICUS/S1_GRD)
      -> VV, VH backscatter, VV/VH ratio (radar - works through cloud cover)
  - Cloud probability (COPERNICUS/S2_CLOUD_PROBABILITY) for masking

SETUP REQUIRED (see docs/GEE_SETUP.md):
  1. Free Earth Engine account (signup.earthengine.google.com)
  2. A Google Cloud project with the Earth Engine API enabled
  3. A service account with the "Earth Engine Resource Viewer" role,
     registered for EE access, with a downloaded JSON key
  4. Set GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT_EMAIL,
     GEE_SERVICE_ACCOUNT_KEY_PATH in .env
"""
from datetime import datetime, timedelta
from typing import Optional

import ee

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.inference import SatelliteFeatures

logger = get_logger(__name__)
settings = get_settings()

_initialized = False


def init_earth_engine() -> None:
    """Authenticate once per process using a service account.
    Safe to call multiple times - it's a no-op after the first success.
    """
    global _initialized
    if _initialized:
        return

    if not settings.GEE_SERVICE_ACCOUNT_EMAIL or not settings.GEE_PROJECT_ID:
        raise RuntimeError(
            "GEE_SERVICE_ACCOUNT_EMAIL / GEE_PROJECT_ID not set. "
            "See docs/GEE_SETUP.md - the app cannot fetch real satellite "
            "data without these."
        )

    credentials = ee.ServiceAccountCredentials(
        settings.GEE_SERVICE_ACCOUNT_EMAIL,
        settings.GEE_SERVICE_ACCOUNT_KEY_PATH,
    )
    ee.Initialize(credentials, project=settings.GEE_PROJECT_ID)
    _initialized = True
    logger.info("Earth Engine initialized for project %s", settings.GEE_PROJECT_ID)


def _geojson_to_ee_geometry(geometry_geojson: dict) -> ee.Geometry:
    geom_type = geometry_geojson.get("type")
    coords = geometry_geojson.get("coordinates")
    if geom_type == "Polygon":
        return ee.Geometry.Polygon(coords)
    if geom_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon(coords)
    raise ValueError(f"Unsupported geometry type for a field boundary: {geom_type}")


def _mask_s2_clouds(image: ee.Image, cloud_prob: ee.Image) -> ee.Image:
    """Mask out pixels above the configured cloud-probability threshold."""
    mask = cloud_prob.select("probability").lt(settings.CLOUD_PROB_THRESHOLD)
    return image.updateMask(mask)


def fetch_sentinel2_indices(
    geometry_geojson: dict,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Returns field-mean NDVI, EVI, NDWI for the most recent cloud-filtered
    Sentinel-2 composite within S2_LOOKBACK_DAYS of as_of_date.

    NDVI = (NIR - RED) / (NIR + RED)            -> vegetation vigor
    EVI  = 2.5*(NIR-RED) / (NIR+6*RED-7.5*BLUE+1) -> vigor, less soil-noise
    NDWI = (NIR - SWIR1) / (NIR + SWIR1)         -> canopy/leaf water content
           (Gao, 1996 formulation; SWIR-based, distinct from McFeeters' open-water NDWI)
    """
    init_earth_engine()

    region = _geojson_to_ee_geometry(geometry_geojson)
    end = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.utcnow()
    start = end - timedelta(days=settings.S2_LOOKBACK_DAYS)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_str, end_str)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    cloud_prob_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(region)
        .filterDate(start_str, end_str)
    )

    count = s2.size().getInfo()
    if count == 0:
        logger.warning("No Sentinel-2 scenes found in window %s..%s", start_str, end_str)
        return {
            "ndvi": None, "evi": None, "ndwi": None,
            "s2_image_date": None, "cloud_pct_in_window": None, "pixel_count": 0,
        }

    image = s2.first()
    image_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    cloud_pct = image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    cloud_prob_img = cloud_prob_col.filter(
        ee.Filter.eq("system:index", image.get("system:index"))
    ).first()
    if cloud_prob_img is not None:
        image = _mask_s2_clouds(image, cloud_prob_img)

    nir = image.select("B8")
    red = image.select("B4")
    blue = image.select("B2")
    swir1 = image.select("B11")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).rename("EVI")
    ndwi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDWI")

    composite = ndvi.addBands(evi).addBands(ndwi)

    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=region,
        scale=10,
        maxPixels=1e9,
    ).getInfo()

    return {
        "ndvi": stats.get("NDVI_mean"),
        "evi": stats.get("EVI_mean"),
        "ndwi": stats.get("NDWI_mean"),
        "s2_image_date": image_date,
        "cloud_pct_in_window": cloud_pct,
        "pixel_count": stats.get("NDVI_count"),
    }


def fetch_sentinel1_backscatter(
    geometry_geojson: dict,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Returns field-mean VV, VH (dB) and VV/VH ratio from the most recent
    Sentinel-1 GRD scene within S1_LOOKBACK_DAYS. SAR penetrates cloud cover,
    which is why we keep it alongside optical - it's the fallback signal
    during monsoon when Sentinel-2 is unusable for weeks at a time.

    VV/VH ratio is sensitive to canopy structure and is a known proxy for
    crop biomass/moisture in SAR literature (e.g. McNairn & Brisco, 2004).
    """
    init_earth_engine()

    region = _geojson_to_ee_geometry(geometry_geojson)
    end = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.utcnow()
    start = end - timedelta(days=settings.S1_LOOKBACK_DAYS)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_str, end_str)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .sort("system:time_start", False)  # most recent first
    )

    count = s1.size().getInfo()
    if count == 0:
        logger.warning("No Sentinel-1 scenes found in window %s..%s", start_str, end_str)
        return {
            "s1_vv_db": None, "s1_vh_db": None,
            "s1_vv_vh_diff_db": None, "s1_image_date": None,
        }

    image = s1.first()
    image_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()

    vv = image.select("VV")
    vh = image.select("VH")

    stats = vv.addBands(vh).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=10,
        maxPixels=1e9,
    ).getInfo()

    vv_db, vh_db = stats.get("VV"), stats.get("VH")
    diff_db = (vv_db - vh_db) if (vv_db is not None and vh_db is not None) else None

    return {
        "s1_vv_db": vv_db,
        "s1_vh_db": vh_db,
        "s1_vv_vh_diff_db": diff_db,  # VV-VH in dB; see SatelliteFeatures docstring for why this isn't a linear ratio
        "s1_image_date": image_date,
    }


def fetch_field_satellite_features(
    geometry_geojson: dict,
    as_of_date: Optional[str] = None,
) -> SatelliteFeatures:
    """Combines S2 + S1 pulls into one SatelliteFeatures record."""
    s2 = fetch_sentinel2_indices(geometry_geojson, as_of_date)
    s1 = fetch_sentinel1_backscatter(geometry_geojson, as_of_date)

    return SatelliteFeatures(
        date=as_of_date or datetime.utcnow().strftime("%Y-%m-%d"),
        s2_image_date=s2.get("s2_image_date"),
        s1_image_date=s1.get("s1_image_date"),
        ndvi=s2.get("ndvi"),
        evi=s2.get("evi"),
        ndwi=s2.get("ndwi"),
        s1_vv_db=s1.get("s1_vv_db"),
        s1_vh_db=s1.get("s1_vh_db"),
        s1_vv_vh_diff_db=s1.get("s1_vv_vh_diff_db"),
        cloud_pct_in_window=s2.get("cloud_pct_in_window"),
        pixel_count=s2.get("pixel_count"),
    )
