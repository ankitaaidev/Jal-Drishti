"""
Generates a labeled training dataset for the crop classifier.

HONESTY NOTE (read this before trusting the model's accuracy number):
This dataset is SYNTHETIC. It is NOT real Sentinel-1/2 pixels from real
labeled Indian fields - no such public dataset was available to download
in this project's timeframe. Instead, each crop class's NDVI/NDWI/SAR
values are sampled from distributions centered on values reported in
published Sentinel-1/2 crop-discrimination studies (cited in comments
below), with realistic noise added.

This means: the model trained on this data learns a real, generalizable
PATTERN (e.g. "paddy has higher NDWI and VH backscatter than wheat at a
similar NDVI"), and its accuracy number is a real, honestly-measured
number - but it is accuracy at separating these literature-grounded
synthetic distributions, NOT accuracy on real field labels. Treat it as
a credible prior / starting point, not a validated production model.

REAL UPGRADE PATH (do this once you have real fields):
1. For each of your real registered fields, confirm the actual crop type
   on the ground (ask the farmer, visit, or use known crop calendars).
2. Call POST /satellite/{field_id}/refresh to get real NDVI/NDWI/SAR
   features for that field on a given date.
3. Append (features, confirmed_crop_type) to data/labels/real_labels.csv
   using the same column format as below.
4. Re-run scripts/train_crop_model.py - it will automatically prefer
   data/labels/real_labels.csv over this synthetic generator once that
   file has enough rows (see MIN_REAL_SAMPLES in train_crop_model.py).

Sources used for the per-crop centers (single-date, mid-season snapshot):
- Paddy rice: NDVI ~0.45-0.60 mid-season, elevated VH backscatter and
  NDWI during flooding/tillering (Bouvet & Le Toan 2011; Ranjan et al.,
  Sentinel-1 rice biophysical parameters study, Gujarat; Chen et al. 2026
  multi-temporal S1/S2 rice classification study)
- Wheat/cereal: NDVI ~0.35-0.65 depending on stage, higher VV/VH
  separation (dB difference) than paddy due to vertical stalk structure,
  lower NDWI than paddy (no standing water)
- Other row crop (cotton/sugarcane/vegetables bucket): broad NDVI range,
  intermediate SAR signature - used as the catch-all v1 third class
"""
import csv
import os
import random

random.seed(42)  # reproducible dataset

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "labels", "synthetic_crop_training.csv"
)

# (mean, std) per feature per crop class - grounded in the ranges cited above
CROP_DISTRIBUTIONS = {
    "paddy_rice": {
        "ndvi": (0.52, 0.08),
        "ndwi": (-0.05, 0.10),       # elevated vs wheat - standing water / high canopy water content
        "s1_vh_db": (-15.5, 2.0),    # elevated VH backscatter (less negative = stronger return)
        "s1_vv_vh_diff_db": (4.0, 2.0),
    },
    "wheat_cereal": {
        "ndvi": (0.48, 0.12),
        "ndwi": (-0.22, 0.09),
        "s1_vh_db": (-19.0, 2.2),
        "s1_vv_vh_diff_db": (10.5, 2.3),  # higher VV/VH separation - vertical stalk structure
    },
    "other_row_crop": {
        "ndvi": (0.40, 0.15),
        "ndwi": (-0.15, 0.12),
        "s1_vh_db": (-17.5, 2.5),
        "s1_vv_vh_diff_db": (7.0, 2.8),   # intermediate - broad catch-all bucket
    },
}

SAMPLES_PER_CLASS = 150  # small but workable for a simple model; documented as small-N


def generate():
    rows = []
    for crop, dist in CROP_DISTRIBUTIONS.items():
        for _ in range(SAMPLES_PER_CLASS):
            ndvi = round(random.gauss(*dist["ndvi"]), 4)
            ndwi = round(random.gauss(*dist["ndwi"]), 4)
            vh = round(random.gauss(*dist["s1_vh_db"]), 3)
            diff_db = round(random.gauss(*dist["s1_vv_vh_diff_db"]), 3)
            # clip to physically plausible ranges
            ndvi = max(-0.1, min(0.95, ndvi))
            ndwi = max(-0.6, min(0.5, ndwi))
            rows.append({
                "ndvi": ndvi,
                "ndwi": ndwi,
                "s1_vh_db": vh,
                "s1_vv_vh_diff_db": diff_db,
                "crop_type": crop,
            })

    random.shuffle(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ndvi", "ndwi", "s1_vh_db", "s1_vv_vh_diff_db", "crop_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic labeled rows to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate()
