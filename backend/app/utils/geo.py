"""
Geo utilities that don't need Earth Engine - kept dependency-light so
they can be unit tested without GEE credentials.
"""
import math


def _haversine_ring_area_m2(ring: list) -> float:
    """
    Approximate planar-equivalent area (m^2) of a lon/lat ring using an
    equirectangular projection centered on the ring's mean latitude.
    Good enough for field-sized polygons (a few hectares); not intended
    for very large or polar-region polygons.
    """
    if len(ring) < 3:
        return 0.0

    lats = [p[1] for p in ring]
    mean_lat = sum(lats) / len(lats)
    lat_rad = math.radians(mean_lat)

    R = 6378137.0  # WGS84 equatorial radius, m
    m_per_deg_lat = (math.pi / 180) * R
    m_per_deg_lon = (math.pi / 180) * R * math.cos(lat_rad)

    xy = [(lon * m_per_deg_lon, lat * m_per_deg_lat) for lon, lat in ring]

    area = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_area_hectares(geometry_geojson: dict) -> float:
    geom_type = geometry_geojson["type"]
    coords = geometry_geojson["coordinates"]

    total_m2 = 0.0
    if geom_type == "Polygon":
        total_m2 = _haversine_ring_area_m2(coords[0])
    elif geom_type == "MultiPolygon":
        for poly in coords:
            total_m2 += _haversine_ring_area_m2(poly[0])
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    return round(total_m2 / 10_000, 3)  # m^2 -> hectares
