"""
Crop-type identification.

TWO PATHS, IN ORDER OF PREFERENCE:

1. ML PATH (used when models/crop_classifier.joblib exists): a Logistic
   Regression model trained by scripts/train_crop_model.py.

   The model file self-reports exactly what it was trained on via its
   saved "training_data_source" metadata - check
   model_data["training_data_source"] (also embedded in the
   crop_type_method string on every /infer response, e.g.
   "ml-logistic-regression (cv_acc=0.91)") before stating an accuracy
   number anywhere. It will say either:
     - "real_labels.csv (N confirmed field samples)" - trained on YOUR
       real fields with confirmed ground-truth crop types. This is a
       genuinely validated accuracy number.
     - "SYNTHETIC bootstrap dataset" - trained on literature-grounded
       synthetic feature distributions (see
       scripts/generate_training_data.py), used only when fewer than
       MIN_REAL_SAMPLES real labels exist yet. In this case the accuracy
       number describes separating synthetic distributions, NOT validated
       real-field accuracy - say so if asked.

   Either way: NEVER state a higher accuracy number than what the most
   recent run of scripts/train_crop_model.py actually printed.

2. RULE-BASED FALLBACK (used if the model file is missing, corrupted, or
   fails to load for any reason): the original transparent threshold
   rules. This guarantees the API never crashes or returns nothing just
   because a model file wasn't shipped/trained yet.

Either path's output goes through the same farmer-declared-crop
reconciliation logic at the bottom of classify_crop().

To train on your own real fields instead of synthetic data: see
docs/REAL_FIELD_LABELING.md and scripts/batch_register_real_fields.py.
"""
import os
from typing import Tuple, Optional

from app.core.logging import get_logger
from app.schemas.inference import SatelliteFeatures

logger = get_logger(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "models", "crop_classifier.joblib"
)

# --- Load the trained model once at import time, if present ---
_ml_model_data: Optional[dict] = None
_ml_load_attempted = False


def _try_load_ml_model() -> Optional[dict]:
    global _ml_model_data, _ml_load_attempted
    if _ml_load_attempted:
        return _ml_model_data
    _ml_load_attempted = True

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "No trained crop model found at %s - using rule-based fallback. "
            "Run scripts/train_crop_model.py to train one.", MODEL_PATH
        )
        return None

    try:
        import joblib  # local import - keeps joblib optional if model is never trained
        data = joblib.load(MODEL_PATH)
        logger.info(
            "Loaded trained crop classifier (%s, cv_accuracy=%.3f, n_samples=%d)",
            data.get("training_data_source"), data.get("cv_accuracy_mean", 0),
            data.get("n_training_samples", 0),
        )
        _ml_model_data = data
        return data
    except Exception as exc:
        logger.error("Failed to load crop model from %s: %s - using rule-based fallback.", MODEL_PATH, exc)
        return None


# --- Rule-based fallback (unchanged from the original heuristic version) ---
# Single-date NDVI/NDWI/SAR ranges are NOT crop-specific on their own;
# this is intentionally a coarse 3-way split using thresholds adapted from
# published Sentinel-1/2 crop discrimination work (e.g. paddy's high VH
# backscatter and high NDWI due to standing water; wheat/cereal's high
# VV/VH separation from vertical stalk structure).
CROP_RULES = [
    (
        "paddy_rice",
        lambda f: (f.ndwi is not None and f.ndwi > -0.1)
        and (f.s1_vh_db is not None and f.s1_vh_db > -18),
        0.55,
    ),
    (
        "wheat_cereal",
        lambda f: (f.s1_vv_vh_diff_db is not None and f.s1_vv_vh_diff_db > 9)
        and (f.ndvi is not None and f.ndvi > 0.3),
        0.50,
    ),
    (
        "other_row_crop",
        lambda f: f.ndvi is not None and f.ndvi > 0.2,
        0.40,
    ),
]


def _classify_crop_rule_based(features: SatelliteFeatures) -> Tuple[str, float]:
    for name, condition, base_conf in CROP_RULES:
        try:
            if condition(features):
                return name, base_conf
        except TypeError:
            continue
    return "unclassified", 0.25


def _classify_crop_ml(features: SatelliteFeatures, model_data: dict) -> Optional[Tuple[str, float]]:
    """Returns (crop_type, confidence) from the trained model, or None if a
    required feature is missing (model can't make a meaningful prediction)."""
    required = model_data["feature_columns"]
    values = []
    for col in required:
        val = getattr(features, col, None)
        if val is None:
            return None
        values.append(val)

    proba = model_data["pipeline"].predict_proba([values])[0]
    classes = model_data["pipeline"].classes_
    best_idx = proba.argmax()
    return str(classes[best_idx]), float(proba[best_idx])


def classify_crop(
    features: SatelliteFeatures,
    declared_crop: str = None,
) -> Tuple[str, float, str]:
    """
    Returns (crop_type, confidence, method_string).

    Tries the trained ML model first; falls back to rule-based thresholds
    if no model is available or a required feature is missing. Either way,
    a farmer-declared crop is then reconciled against the satellite-derived
    guess (declared crop wins, but disagreement lowers reported confidence
    so a "verify this field" flag can be shown).
    """
    if features.ndvi is None:
        return "unknown", 0.0, "no-data (no cloud-free Sentinel-2 scene in window)"

    model_data = _try_load_ml_model()
    method = "rule-based-fallback"
    matched_crop, matched_conf = None, 0.0

    if model_data is not None:
        ml_result = _classify_crop_ml(features, model_data)
        if ml_result is not None:
            matched_crop, matched_conf = ml_result
            method = f"ml-logistic-regression (cv_acc={model_data.get('cv_accuracy_mean', 0):.2f})"

    if matched_crop is None:
        matched_crop, matched_conf = _classify_crop_rule_based(features)
        if method != "rule-based-fallback":
            method = "rule-based-fallback (ml model loaded but missing required feature)"

    if declared_crop:
        declared_norm = declared_crop.strip().lower().replace(" ", "_")
        if declared_norm == matched_crop:
            return declared_crop, min(matched_conf + 0.2, 0.95), f"{method}+farmer-confirmed"
        else:
            return declared_crop, 0.35, f"farmer-declared ({method} predicted '{matched_crop}' instead - flagged for review)"

    return matched_crop, matched_conf, method