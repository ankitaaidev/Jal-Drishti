"""
Crop-type identification.

HONESTY NOTE: a real ML crop classifier (e.g. XGBoost on multi-temporal
NDVI/SAR time series, as in published India crop-mapping studies) needs a
labeled training set of field polygons with confirmed crop type across a
full season. That dataset does not exist in this repo and can't be
fabricated for a demo without misrepresenting the model's accuracy.

So v1 uses a transparent, literature-grounded RULE-BASED classifier on
single-date spectral signatures + the farmer's declared crop as a prior.
This is weaker than a trained classifier but every number is explainable
on stage, and it correctly reports LOW confidence rather than a fake
accuracy figure.

Upgrade path (documented, not done here): collect 1 season of confirmed
field labels -> train services/crop_model_ml.py -> swap classify_crop()
to call it. Until then this function is the single integration point.
"""
from typing import Tuple

from app.schemas.inference import SatelliteFeatures

# Single-date NDVI/NDWI/SAR ranges are NOT crop-specific on their own;
# this is intentionally a coarse 3-way split (the most common BAH-style
# prototype scope: 2-3 crop types) using thresholds adapted from published
# Sentinel-1/2 crop discrimination work (e.g. paddy's high VH backscatter
# and high NDWI due to standing water; wheat/cereal's high VV/VH ratio
# from vertical stalk structure). This is a *heuristic prior*, not a
# calibrated classifier - confidence is capped accordingly.
CROP_RULES = [
    # (name, condition_fn, base_confidence)
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
        "other_row_crop",  # cotton/sugarcane/vegetables bucket for v1
        lambda f: f.ndvi is not None and f.ndvi > 0.2,
        0.40,
    ),
]


def classify_crop(
    features: SatelliteFeatures,
    declared_crop: str = None,
) -> Tuple[str, float, str]:
    """
    Returns (crop_type, confidence, method_string).

    If the farmer declared a crop, we trust it but still report the
    satellite signal's agreement/disagreement as a confidence modifier -
    this surfaces a useful real-world flag (mislabeled fields, wrong
    polygon, etc.) instead of silently overriding the farmer.
    """
    if features.ndvi is None:
        return "unknown", 0.0, "no-data (no cloud-free Sentinel-2 scene in window)"

    matched_crop = None
    matched_conf = 0.0
    for name, condition, base_conf in CROP_RULES:
        try:
            if condition(features):
                matched_crop, matched_conf = name, base_conf
                break
        except TypeError:
            continue  # missing band value, skip this rule

    if matched_crop is None:
        matched_crop, matched_conf = "unclassified", 0.25

    if declared_crop:
        declared_norm = declared_crop.strip().lower().replace(" ", "_")
        if declared_norm == matched_crop:
            # satellite signal agrees with farmer's declaration -> boost confidence
            return declared_crop, min(matched_conf + 0.3, 0.9), "spectral-rule-based+farmer-confirmed"
        else:
            # disagreement: trust farmer's declaration (ground truth beats
            # a coarse spectral rule) but flag low confidence so the
            # dashboard can show a "verify this field" warning.
            return declared_crop, 0.35, "farmer-declared (spectral signal mismatch - flagged for review)"

    return matched_crop, matched_conf, "spectral-rule-based"
