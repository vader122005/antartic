"""
mock_data.py
============
Pure dummy-data generators. Nothing in this file talks to a real API,
database, or model — it only exists so the UI is fully visual/runnable
before the real backend is connected.

You will NOT need to edit this file to plug in real data — instead, go to
backend_stubs.py and change what those functions return / call. This file
is only imported by backend_stubs.py as a temporary source of fake data.
"""

import random
import numpy as np

from config import LAT_MIN, LAT_MAX, LON_SEGMENTS, GRID_STEP_DEG


def _lon_range(step):
    """Yield (west, east) cell edges across both Ross Sea longitude segments."""
    for seg_west, seg_east in LON_SEGMENTS:
        lon = seg_west
        while lon < seg_east:
            yield (lon, min(lon + step, seg_east))
            lon += step


def generate_ice_grid(step=GRID_STEP_DEG, seed=None):
    """
    Dummy sea-ice-cover grid.

    Returns a list of dicts, one per grid cell:
        {
          "polygon": [[lon, lat], [lon, lat], ...]  # closed ring
          "thickness": float in [0, 1]               # normalized 0=thin, 1=thick
        }

    Real replacement (see backend_stubs.load_ice_cover_data) should return
    the same shape: a list of {"polygon": [[lon,lat], ...], "thickness": 0..1}.
    """
    rng = random.Random(seed)
    cells = []
    lat = LAT_MIN
    while lat < LAT_MAX:
        lat_next = min(lat + step, LAT_MAX)
        for lon_west, lon_east in _lon_range(step):
            # bias thickness so ice is generally heavier further south / near shelf
            lat_bias = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN)
            thickness = min(1.0, max(0.0, lat_bias * 0.7 + rng.uniform(-0.2, 0.3)))
            polygon = [
                [lon_west, lat],
                [lon_east, lat],
                [lon_east, lat_next],
                [lon_west, lat_next],
                [lon_west, lat],
            ]
            cells.append({"polygon": polygon, "thickness": round(thickness, 3)})
        lat = lat_next
    return cells


def generate_icebergs(n=14, seed=None):
    """
    Dummy iceberg point observations.

    Returns a list of dicts:
        {
          "id": str,
          "lat": float,
          "lon": float,
          "size_km2": float,     # relative size
          "risk": float,         # 0..1 normalized risk score
        }

    Real replacement (see backend_stubs.load_current_iceberg_positions) should
    return the same shape, parsed from the user's uploaded CSV/JSON.
    """
    rng = random.Random(seed)
    icebergs = []
    for i in range(n):
        lat = rng.uniform(LAT_MIN + 1, LAT_MAX - 1)
        seg_west, seg_east = rng.choice(LON_SEGMENTS)
        lon = rng.uniform(seg_west + 1, seg_east - 1)
        icebergs.append(
            {
                "id": f"BRG-{1000 + i}",
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "size_km2": round(rng.uniform(1, 120), 1),
                "risk": round(rng.uniform(0, 1), 2),
            }
        )
    return icebergs


def generate_dummy_trajectory(iceberg_data, steps=8, seed=None):
    """
    Dummy forecasted trajectory for each iceberg: a short polyline of
    lat/lon points drifting from its current position.

    Returns:
        {
          "<iceberg_id>": [[lon, lat], [lon, lat], ...],   # forecast path
          ...
        }

    Real replacement (see backend_stubs.run_forecast) should return the
    same shape: a dict keyed by iceberg id, mapping to an ordered list of
    [lon, lat] forecast points.
    """
    rng = random.Random(seed)
    paths = {}
    for berg in iceberg_data:
        lat, lon = berg["lat"], berg["lon"]
        drift_lat = rng.uniform(-0.15, 0.05)  # icebergs tend to drift north
        drift_lon = rng.uniform(-0.2, 0.2)
        path = [[lon, lat]]
        for _ in range(steps):
            lat = lat + drift_lat + rng.uniform(-0.03, 0.03)
            lon = lon + drift_lon + rng.uniform(-0.05, 0.05)
            path.append([lon, lat])
        paths[berg["id"]] = path
    return paths


def generate_dummy_route(ship_pos, destination, seed=None):
    """
    Dummy "suggested safe route" for the ship: a simple set of waypoints
    between current position and destination with a bit of jitter so it
    doesn't look like a perfectly straight line.

    ship_pos / destination: dicts like {"lat": float, "lon": float}

    Returns:
        [[lon, lat], [lon, lat], ...]   # ordered waypoints, start -> end

    Real replacement (see backend_stubs.run_forecast) should return the
    same shape: an ordered list of [lon, lat] waypoints.
    """
    rng = random.Random(seed)
    n_points = 5
    route = []
    for i in range(n_points + 1):
        t = i / n_points
        lat = ship_pos["lat"] + (destination["lat"] - ship_pos["lat"]) * t
        lon = ship_pos["lon"] + (destination["lon"] - ship_pos["lon"]) * t
        if 0 < i < n_points:
            lat += rng.uniform(-0.3, 0.3)
            lon += rng.uniform(-0.3, 0.3)
        route.append([lon, lat])
    return route


def generate_summary_metrics(icebergs, seed=None):
    """
    Dummy right-panel summary numbers, styled after the reference image's
    "Miles Required / Completed / % Complete" cards.
    """
    rng = random.Random(seed)
    high_risk = sum(1 for b in icebergs if b["risk"] > 0.66)
    return {
        "tracked_icebergs": len(icebergs),
        "high_risk_icebergs": high_risk,
        "avg_size_km2": round(np.mean([b["size_km2"] for b in icebergs]), 1) if icebergs else 0,
        "forecast_confidence_pct": rng.randint(55, 92),
    }
