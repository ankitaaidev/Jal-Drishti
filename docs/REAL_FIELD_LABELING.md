# How to fill out `real_fields_template.csv`

This is the file the batch script reads. One row = one real field you have
confirmed crop-type ground truth for.

## Columns

| Column | What goes here |
|---|---|
| `field_name` | Anything readable, e.g. "North Plot - Ramesh's Wheat" |
| `crop_type` | The CONFIRMED real crop. Use one of: `wheat_cereal`, `paddy_rice`, `other_row_crop` — or a new crop name of your own (the model will just learn whatever labels you give it; you aren't locked to these 3) |
| `sowing_date` | `YYYY-MM-DD` if known, blank if not (improves growth-stage accuracy, not required for the crop classifier itself) |
| `farmer_name` | Optional, just for your own records |
| `soil_type` | Optional, free text |
| `corner1_lat` / `corner1_lon` through `corner4_lat` / `corner4_lon` | The 4 corners of the field, as plain latitude/longitude |

The field doesn't need to be a perfect rectangle — 4 corners is just the
simplest input format. If a field is oddly shaped, picking 4 corners that
roughly bound it is good enough for v1 (the satellite pull averages over
the whole polygon anyway).

## How to get the 4 corner coordinates (no GIS software needed)

**Option A — Google Maps (fastest)**
1. Open Google Maps, find your field (satellite view helps: bottom-left
   layers icon → Satellite)
2. Right-click each of the 4 corners → click the lat/lon numbers that pop
   up at the top of the context menu → it copies them to your clipboard
   as `lat, lon`
3. Paste into the matching columns (split into separate lat/lon columns)

**Option B — geojson.io (if you want to see the shape while drawing)**
1. Go to geojson.io, switch to satellite base layer if available
2. Use the polygon tool, click each corner of the field
3. The coordinates appear on the right as GeoJSON
   `coordinates: [[[lon1,lat1],[lon2,lat2],[lon3,lat3],[lon4,lat4],[lon1,lat1]]]`
   — note GeoJSON order is `[lon, lat]`, opposite of the CSV columns here,
   so don't transpose by accident
4. Copy each pair into the CSV, swapping the order to lat,lon

**Option C — your phone's GPS at the field itself**
Stand at each corner, use any "what's my GPS coordinate" app (e.g. Google
Maps' blue dot, long-press → coordinates shown at bottom), note all 4.

## More than 4 corners / irregular shape?

If you need a more precise polygon than 4 corners, draw it in geojson.io
instead and save the file as `data/raw/fields/<field_name>.geojson`, then
list it in a second optional column `geojson_file` instead of the 8 corner
columns — `scripts/batch_register_real_fields.py` supports both: it uses
the `geojson_file` column if present for that row, otherwise builds a
polygon from the 4 corner columns.

## Once the CSV is filled out

Save it as `data/raw/real_fields.csv` (copy the template, don't edit the
template in place), then run:

```bash
python3 scripts/batch_register_real_fields.py
```

This will, for each row:
1. Register the field via `POST /fields` on your running backend
2. Pull real satellite features via `POST /satellite/{field_id}/refresh`
   (real Earth Engine call — needs `DEMO_MODE=false` and GEE credentials
   set up, see docs/GEE_SETUP.md)
3. Append a row to `data/labels/real_labels.csv` with the real NDVI/NDWI/
   SAR values + your confirmed crop_type

Then retrain:
```bash
python3 scripts/train_crop_model.py
```
With 30+ real rows, it automatically trains on `real_labels.csv` instead
of synthetic data, and prints real, ground-truth-validated accuracy.
