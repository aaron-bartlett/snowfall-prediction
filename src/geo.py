"""Great-circle distance, bearing, and a local tangent-plane projection."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2, degrees clockwise from north in [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def local_xy_km(lat, lon, lat0, lon0):
    """Equirectangular projection: km east (x) and north (y) of (lat0, lon0)."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_RADIUS_KM * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_RADIUS_KM
    return x, y
