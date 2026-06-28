"""
Seeds the sample fields into a running Jal-Drishti backend via its own
HTTP API, then triggers inference for each so /priority-list and /alerts
have data immediately.

Usage:
    cd backend && uvicorn app.main:app --reload &
    python scripts/seed_demo.py
"""
import json
import time
import sys
import os

import httpx

API_BASE = os.getenv("JAL_DRISHTI_API", "http://localhost:8000")
SAMPLE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample", "sample_fields.json")


def main():
    with open(SAMPLE_FILE) as f:
        data = json.load(f)

    client = httpx.Client(timeout=60.0)
    field_ids = []

    for field_def in data["fields"]:
        payload = {
            "name": field_def["name"],
            "geometry_geojson": field_def["geometry_geojson"],
            "farmer_name": field_def.get("farmer_name"),
            "declared_crop": field_def.get("declared_crop"),
            "sowing_date": field_def.get("sowing_date"),
            "soil_type": field_def.get("soil_type"),
        }
        resp = client.post(f"{API_BASE}/fields", json=payload)
        resp.raise_for_status()
        field = resp.json()
        field_ids.append(field["field_id"])
        print(f"Created field: {field['name']} -> {field['field_id']} "
              f"({field['area_hectares']} ha)")

    print("\nRunning inference (calls Earth Engine + Open-Meteo for each field)...")
    for fid in field_ids:
        try:
            resp = client.post(f"{API_BASE}/infer/{fid}")
            resp.raise_for_status()
            result = resp.json()
            print(f"  {fid}: {result['crop_type']} / {result['growth_stage']} / "
                  f"stress={result['stress_level']} / deficit={result['water_deficit_mm']}mm")
        except httpx.HTTPStatusError as e:
            print(f"  {fid}: FAILED - {e.response.text}")
        time.sleep(1)  # be polite to the GEE API

    print("\nDone. Try: GET /priority-list and GET /alerts")


if __name__ == "__main__":
    main()
