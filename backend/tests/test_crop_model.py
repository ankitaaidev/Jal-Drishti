"""
Tests for the crop classifier's ML path + rule-based fallback.
Run with: pytest backend/tests/test_crop_model.py

Note: these tests require models/crop_classifier.joblib to exist for the
ML-path assertions to run (skipped otherwise). Run
scripts/train_crop_model.py first if those tests are being skipped.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.services.crop_model import classify_crop, MODEL_PATH
from app.schemas.inference import SatelliteFeatures

MODEL_EXISTS = os.path.exists(MODEL_PATH)


@pytest.mark.skipif(not MODEL_EXISTS, reason="model not trained yet - run scripts/train_crop_model.py")
def test_ml_path_used_when_model_present():
    features = SatelliteFeatures(
        date="2026-06-27", ndvi=0.55, ndwi=-0.05, s1_vh_db=-15.0, s1_vv_vh_diff_db=4.0
    )
    crop, conf, method = classify_crop(features)
    assert "ml-logistic-regression" in method
    assert conf > 0


@pytest.mark.skipif(not MODEL_EXISTS, reason="model not trained yet - run scripts/train_crop_model.py")
def test_ml_correctly_separates_paddy_and_wheat_signatures():
    paddy_like = SatelliteFeatures(
        date="2026-06-27", ndvi=0.52, ndwi=-0.05, s1_vh_db=-15.5, s1_vv_vh_diff_db=4.0
    )
    wheat_like = SatelliteFeatures(
        date="2026-06-27", ndvi=0.48, ndwi=-0.22, s1_vh_db=-19.0, s1_vv_vh_diff_db=10.5
    )
    paddy_crop, _, _ = classify_crop(paddy_like)
    wheat_crop, _, _ = classify_crop(wheat_like)
    assert paddy_crop == "paddy_rice"
    assert wheat_crop == "wheat_cereal"


def test_falls_back_to_rules_when_required_feature_missing():
    # s1_vv_vh_diff_db missing -> ML path can't run even if a model is loaded,
    # should fall back to rule-based without crashing
    features = SatelliteFeatures(
        date="2026-06-27", ndvi=0.55, ndwi=-0.05, s1_vh_db=-15.0, s1_vv_vh_diff_db=None
    )
    crop, conf, method = classify_crop(features)
    assert crop is not None
    assert "rule-based" in method  # either pure fallback or "ml model loaded but missing feature"


def test_no_ndvi_returns_unknown_without_crashing():
    features = SatelliteFeatures(date="2026-06-27", ndvi=None)
    crop, conf, method = classify_crop(features)
    assert crop == "unknown"
    assert conf == 0.0


def test_farmer_declared_crop_overrides_with_lower_confidence_on_disagreement():
    # Wheat-like signal but farmer says paddy -> should trust farmer, flag low confidence
    features = SatelliteFeatures(
        date="2026-06-27", ndvi=0.48, ndwi=-0.22, s1_vh_db=-19.0, s1_vv_vh_diff_db=10.5
    )
    crop, conf, method = classify_crop(features, declared_crop="paddy_rice")
    assert crop == "paddy_rice"
    assert conf <= 0.4
    assert "flagged for review" in method or "mismatch" in method
