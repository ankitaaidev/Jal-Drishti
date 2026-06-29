import ee
import pandas as pd
import random
import time

ee.Initialize(project="jal-drishti-500712")

# -----------------------------
# CONFIG
# ----------------------------
NUM_FIELDS = 500 # 

# India bounding box (you can adjust region)
MIN_LON, MAX_LON = 73.0, 88.0
MIN_LAT, MAX_LAT = 15.0, 32.0

CROPS = ["wheat_cereal"] * 3 + ["paddy_rice"] * 3 + ["maize"] * 3
# -----------------------------
# FUNCTION: Create random field polygon
# -----------------------------
def create_random_field():
    lon = random.uniform(MIN_LON, MAX_LON)
    lat = random.uniform(MIN_LAT, MAX_LAT)

    size = 0.02  # ~2km field block

    return [[
        [lon, lat],
        [lon + size, lat],
        [lon + size, lat + size],
        [lon, lat + size],
        [lon, lat]
    ]]

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(coords):
    field = ee.Geometry.Polygon(coords)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(field)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median())

    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(field)
        .filterDate("2024-01-01", "2024-12-31")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .median())

    vv = s1.select("VV")
    vh = s1.select("VH")
    diff = vv.subtract(vh).rename("VV_VH_DIFF")

    stats = ee.Image.cat([ndvi, ndwi, vv, vh, diff]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=field,
        scale=10,
        maxPixels=1e13
    ).getInfo()

    return stats
def assign_crop(lat, lon):
    if lat > 28:
        return "wheat_cereal"
    elif lon > 85:
        return "paddy_rice"
    else:
        return "maize"

# -----------------------------
# DATASET CREATION
# -----------------------------
data = []

for i in range(NUM_FIELDS):
    try:
        print(f"Processing field {i+1}/{NUM_FIELDS}")

        coords = create_random_field()
        # get center point
        lon = coords[0][0][0]
        lat = coords[0][0][1]

        crop = assign_crop(lat, lon)

        stats = extract_features(coords)

        if stats is None:
            continue

        row = {
            "ndvi": stats.get("NDVI"),
            "ndwi": stats.get("NDWI"),
            "s1_vv_db": stats.get("VV"),
            "s1_vh_db": stats.get("VH"),
            "s1_vv_vh_diff_db": stats.get("VV_VH_DIFF"),
            "crop_type": crop
        }

        data.append(row)

        time.sleep(0.2)  # avoid EE quota issues

    except Exception as e:
        print("Skipped field due to error:", e)

# -----------------------------
# SAVE
# -----------------------------
df = pd.DataFrame(data)
df.to_csv("../data/labels/real_labels.csv", index=False)

print("DONE → dataset created with", len(df), "samples")