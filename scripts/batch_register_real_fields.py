"""
Batch-registers real fields from a CSV and builds real_labels.csv from
real Earth Engine satellite features - no synthetic data involved.

Usage:
    1. Make sure your backend is running with DEMO_MODE=false and real
       GEE credentials configured (see docs/GEE_SETUP.md) - this script
       calls your live API, so whatever DEMO_MODE is set to there is
       what you'll actually get.
    2. Fill out data/raw/real_fields.csv (see docs/REAL_FIELD_LABELING.md
       for exactly how - copy data/raw/real_fields_template.csv to start)
    3. Run: python3 scripts/batch_register_real_fields.py

For each row, this:
    a) Registers the field via POST /fields
    b) Pulls real satellite features via POST /satellite/{field_id}/refresh
    c) Appends one row to data/labels/real_labels.csv with the REAL
       ndvi/ndwi/s1_vh_db/s1_vv_vh_diff_db values + your confirmed crop_type

It is safe to re-run: fields already present in real_labels.csv (matched
by field_name) are skipped rather than duplicated, so you can add new
rows to the input CSV over time and just re-run this script.
"""
import csv
import json
import os
import sys
import time

import httpx

THIS_DIR = os.path.dirname(__file__)
INPUT_CSV = os.path.join(THIS_DIR, "..", "data", "raw", "real_fields.csv")
GEOJSON_DIR = os.path.join(THIS_DIR, "..", "data", "raw", "fields")
OUTPUT_CSV = os.path.join(THIS_DIR, "..", "data", "labels", "real_labels.csv")
API_BASE = os.getenv("JAL_DRISHTI_API", "http://localhost:8000")

REQUIRED_FEATURE_COLUMNS = ["ndvi", "ndwi", "s1_vh_db", "s1_vv_vh_diff_db"]


def build_geojson_from_corners(row: dict) -> dict:
    """Builds a closed Polygon from 4 lat/lon corner pairs in the CSV row."""
    coords = []
    for i in range(1, 5):
        lat = float(row[f"corner{i}_lat"])
        lon = float(row[f"corner{i}_lon"])
        coords.append([lon, lat])  # GeoJSON order is [lon, lat]
    coords.append(coords[0])  # close the ring
    return {"type": "Polygon", "coordinates": [coords]}


def load_geojson_file(filename: str) -> dict:
    path = os.path.join(GEOJSON_DIR, filename)
    with open(path) as f:
        data = json.load(f)
    if data.get("type") == "Feature":
        return data["geometry"]
    return data


def load_already_done() -> set:
    if not os.path.exists(OUTPUT_CSV):
        return set()
    with open(OUTPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r.get("field_name", "") for r in rows if r.get("field_name")}


def main():
    if not os.path.exists(INPUT_CSV):
        print(
            f"No input file at {INPUT_CSV}.\n"
            f"Copy data/raw/real_fields_template.csv to data/raw/real_fields.csv "
            f"and fill it out first - see docs/REAL_FIELD_LABELING.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(INPUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("Input CSV is empty.", file=sys.stderr)
        sys.exit(1)

    already_done = load_already_done()
    print(f"{len(already_done)} fields already in real_labels.csv - will skip those.\n")

    client = httpx.Client(timeout=60.0)
    new_rows = []
    failures = []

    for i, row in enumerate(rows, start=1):
        field_name = row["field_name"].strip()
        crop_type = row["crop_type"].strip()

        if field_name in already_done:
            print(f"[{i}/{len(rows)}] SKIP (already labeled): {field_name}")
            continue

        try:
            if row.get("geojson_file"):
                geometry = load_geojson_file(row["geojson_file"].strip())
            else:
                geometry = build_geojson_from_corners(row)
        except Exception as exc:
            print(f"[{i}/{len(rows)}] FAILED to build geometry for {field_name}: {exc}")
            failures.append((field_name, f"geometry error: {exc}"))
            continue

        payload = {
            "name": field_name,
            "geometry_geojson": geometry,
            "farmer_name": row.get("farmer_name") or None,
            "declared_crop": crop_type,
            "sowing_date": row.get("sowing_date") or None,
            "soil_type": row.get("soil_type") or None,
        }
        try:
            resp = client.post(f"{API_BASE}/fields", json=payload)
            resp.raise_for_status()
            field = resp.json()
            field_id = field["field_id"]
        except Exception as exc:
            print(f"[{i}/{len(rows)}] FAILED to register {field_name}: {exc}")
            failures.append((field_name, f"registration error: {exc}"))
            continue

        try:
            resp = client.post(f"{API_BASE}/satellite/{field_id}/refresh")
            resp.raise_for_status()
            features = resp.json()
        except Exception as exc:
            print(f"[{i}/{len(rows)}] FAILED to fetch satellite features for {field_name}: {exc}")
            failures.append((field_name, f"satellite fetch error: {exc}"))
            continue

        missing = [c for c in REQUIRED_FEATURE_COLUMNS if features.get(c) is None]
        if missing:
            print(
                f"[{i}/{len(rows)}] SKIPPED {field_name}: missing {missing} "
                f"(likely no cloud-free scene in the lookback window for this field/date)"
            )
            failures.append((field_name, f"missing features: {missing}"))
            continue

        new_rows.append({
            "field_name": field_name,
            "ndvi": features["ndvi"],
            "ndwi": features["ndwi"],
            "s1_vh_db": features["s1_vh_db"],
            "s1_vv_vh_diff_db": features["s1_vv_vh_diff_db"],
            "crop_type": crop_type,
        })
        print(f"[{i}/{len(rows)}] OK: {field_name} -> ndvi={features['ndvi']}, "
              f"ndwi={features['ndwi']}, crop={crop_type}")

        time.sleep(1)  # be polite to the Earth Engine API across many calls

    if new_rows:
        file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        fieldnames = ["field_name"] + REQUIRED_FEATURE_COLUMNS + ["crop_type"]
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in new_rows:
                writer.writerow(r)

    print(f"\nDone. {len(new_rows)} new real-labeled rows added to {OUTPUT_CSV}.")
    total_real = len(already_done) + len(new_rows)
    print(f"Total real-labeled fields so far: {total_real}")
    if total_real >= 30:
        print("You're at/above the 30-sample minimum - scripts/train_crop_model.py "
              "will now train on REAL data automatically.")
    else:
        print(f"Need {30 - total_real} more to hit the real-data training threshold "
              f"(MIN_REAL_SAMPLES in scripts/train_crop_model.py).")

    if failures:
        print(f"\n{len(failures)} fields failed or were skipped:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
