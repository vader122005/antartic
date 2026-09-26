"""
route_optimiser.py - Pareto multi-objective route optimiser for the Ross Sea.

WHAT IT DOES
------------
Given a cost surface (from `cost_surface.py`), a start and an end position, and
the currently known ice hazards, it finds safe, fuel-efficient routes for a
vessel of a given IACS polar class.

GRAPH
-----
Nodes  : all passable cells (`passable == True` and finite cost)
Edges  : 8-connected (N, NE, E, SE, S, SW, W, NW)
Weight : the DESTINATION cell's cost, x sqrt(2) on diagonals to account for the
         longer traverse. Impassable cells are never added as nodes - a hard
         wall, not an expensive detour.
Search : A* with a haversine heuristic. The heuristic is scaled by the minimum
         per-km cost present in the grid, which keeps it ADMISSIBLE (it can
         never overestimate the true remaining cost) so A* stays optimal.
         Falls back to Dijkstra (heuristic = 0) if A* finds no path.

PARETO SET
----------
The same search is run at three alpha/beta weightings, trading fuel against
time, and all three are returned:

    (alpha=1.0, beta=0.2)  fuel_priority
    (alpha=0.5, beta=0.5)  balanced          <- recommended
    (alpha=0.2, beta=1.0)  time_priority

HAZARDS
-------
Three sources are burned into the cost grid BEFORE the search, so the graph
search itself needs no hazard logic:

  1. tracked BYU/NIC bergs      - from the iceberg LSTM, exclusion radius
                                  scaled by the model's own uncertainty_km
  2. uncharted captain-reported - forecast_uncharted_berg(), physics-only,
     bergs                        wider radius (no LSTM correction available)
  3. growlers / bergy bits      - log_static_hazard(), fixed 5 km, no trajectory

KNOWN GAPS surfaced in the output rather than silently ignored:
  * ocean currents (CMEMS) unavailable -> iceberg trajectory uncertainty
    elevated; hazard radii are correspondingly wider
  * bathymetry (GEBCO) unavailable -> no grounding check
"""

from __future__ import annotations

import heapq
import math
import os

import numpy as np

import grid_utils as gu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARETO_WEIGHTS = [
    (1.0, 0.2, "fuel_priority"),
    (0.5, 0.5, "balanced"),
    (0.2, 1.0, "time_priority"),
]
RECOMMENDED = "balanced"

# Hazard exclusion for a TRACKED berg: base radius, widened by the LSTM's own
# 1-sigma so a less certain forecast clears a wider berth.
TRACKED_BASE_RADIUS_KM = 50.0
TRACKED_SIGMA_MULTIPLIER = 2.0
HAZARD_PENALTY_FACTOR = 5.0

SQRT2 = math.sqrt(2.0)

# 8-connected neighbourhood
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]

ICE_SEVERITY_BANDS = ((0.20, "low"), (0.55, "moderate"))


# ---------------------------------------------------------------------------
# Cost surface loading
# ---------------------------------------------------------------------------

def load_cost_surface(path):
    """Load a cost_surface_*.npz into a plain dict."""
    d = np.load(path, allow_pickle=True)
    return {
        "cost": np.asarray(d["cost"], dtype=np.float64),
        "cost_raw": np.asarray(d["cost_raw"], dtype=np.float64),
        "speed_ms": np.asarray(d["speed_ms"], dtype=np.float64),
        "fuel_rate_kgs": np.asarray(d["fuel_rate_kgs"], dtype=np.float64),
        "passable": np.asarray(d["passable"]).astype(bool),
        "ice_thickness_m": np.asarray(d["ice_thickness_m"], dtype=np.float64),
        "lat": np.asarray(d["lat"], dtype=np.float64),
        "lon": np.asarray(d["lon"], dtype=np.float64),
        "metadata": d["metadata"] if "metadata" in d.files else np.array([]),
    }


def build_cost_surface_for(polar_class, date_str,
                          sic_npz="ross_sea_concentration.npz",
                          csv_path="pc_ice_speed_model.csv",
                          alpha=1.0, beta=1.0, out_path=None):
    """Build a cost surface on the fly when no pre-built file is supplied."""
    import cost_surface as cs

    dates = list(np.load(sic_npz)["dates"])
    key = str(date_str).replace("-", "")
    if key not in dates:
        raise ValueError(f"date {date_str} not present in {sic_npz}")
    idx = dates.index(key)

    if out_path is None:
        out_path = f"cost_surface_{polar_class}_{key}.npz"
    cs.build_cost_surface(
        npz_path=sic_npz, csv_path=csv_path, polar_class=polar_class,
        date_index=idx, alpha=alpha, beta=beta, output_path=out_path,
    )
    return load_cost_surface(out_path)


# ---------------------------------------------------------------------------
# Hazard assembly
# ---------------------------------------------------------------------------

def tracked_berg_hazards(iceberg_positions):
    """
    Convert LSTM-predicted tracked berg positions into hazard dicts.

    The exclusion radius is the 50 km base widened by twice the model's
    reported 1-sigma, so a more uncertain forecast produces a wider berth -
    which is the point of propagating uncertainty at all.
    """
    out = []
    for k, b in enumerate(iceberg_positions or []):
        sigma = float(b.get("uncertainty_km", 0.0) or 0.0)
        out.append({
            "hazard_id": b.get("berg_id", f"TRACKED_{k}"),
            "lat": float(b["lat"]),
            "lon": float(b["lon"]),
            "exclusion_km": TRACKED_BASE_RADIUS_KM + TRACKED_SIGMA_MULTIPLIER * sigma,
            "hazard_type": "tracked_berg",
            "is_static": False,
            "uncertainty_km": sigma,
        })
    return out


def apply_all_hazards(cost, lat, lon, iceberg_positions=None,
                      uncharted_bergs=None, static_hazards=None,
                      date_str=None, penalty_factor=HAZARD_PENALTY_FACTOR):
    """
    Burn every hazard source into the cost grid before the graph search.
    Returns (modified_cost, hazard_records).
    """
    from uncharted_berg_handler import (get_active_hazard_zones,
                                        apply_hazard_zones_to_cost_grid)

    grid = np.array(cost, dtype=np.float64, copy=True)
    hazards = tracked_berg_hazards(iceberg_positions)

    if hazards:
        grid = apply_hazard_zones_to_cost_grid(grid, lat, lon, hazards,
                                               penalty_factor=penalty_factor)

    if uncharted_bergs or static_hazards:
        query_time = None
        if date_str:
            k = str(date_str).replace("-", "")
            query_time = f"{k[:4]}-{k[4:6]}-{k[6:]} 06:00"
        active = get_active_hazard_zones(uncharted_bergs or [],
                                         static_hazards or [],
                                         query_time=query_time)
        if active:
            grid = apply_hazard_zones_to_cost_grid(grid, lat, lon, active,
                                                   penalty_factor=penalty_factor)
            hazards.extend(active)

    return grid, hazards


# ---------------------------------------------------------------------------
# Graph search
# ---------------------------------------------------------------------------

def _cell_km(lat, lon, a, b):
    """Great-circle distance between two grid cells, in km."""
    return float(gu.haversine_km(lat[a], lon[a], lat[b], lon[b]))


def astar(cost, passable, lat, lon, start, goal, use_heuristic=True):
    """
    A* over the 8-connected grid. Edge weight is the destination cell's cost,
    scaled by sqrt(2) on diagonals.

    The heuristic is `min_cost_per_km * haversine(node, goal)`. Scaling by the
    grid's MINIMUM per-km cost guarantees admissibility: no remaining path can
    ever be cheaper than travelling the straight-line distance at the cheapest
    rate anywhere on the grid, so A* still returns an optimal path.
    """
    rows, cols = cost.shape
    open_ok = passable & np.isfinite(cost)

    if not open_ok[start] or not open_ok[goal]:
        return None

    if use_heuristic:
        # Cheapest cost-per-km anywhere on the grid (cells are ~25 km across)
        finite = cost[open_ok]
        cell_km = max(_cell_km(lat, lon, (0, 0), (0, 1)), 1e-6)
        min_rate = float(np.min(finite)) / cell_km if len(finite) else 0.0
    else:
        min_rate = 0.0

    goal_lat, goal_lon = lat[goal], lon[goal]

    def h(node):
        if min_rate <= 0.0:
            return 0.0
        return min_rate * float(gu.haversine_km(lat[node], lon[node],
                                                goal_lat, goal_lon))

    g = {start: 0.0}
    came = {}
    pq = [(h(start), 0.0, start)]
    closed = set()

    while pq:
        _, gc, node = heapq.heappop(pq)
        if node in closed:
            continue
        closed.add(node)
        if node == goal:
            path = [node]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]

        i, j = node
        for di, dj, mult in _NEIGHBOURS:
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            nxt = (ni, nj)
            if nxt in closed or not open_ok[nxt]:
                continue
            step = cost[nxt] * mult
            ng = gc + step
            if ng < g.get(nxt, math.inf):
                g[nxt] = ng
                came[nxt] = node
                heapq.heappush(pq, (ng + h(nxt), ng, nxt))
    return None


# ---------------------------------------------------------------------------
# Route summarisation
# ---------------------------------------------------------------------------

def _severity(mean_thickness):
    for limit, label in ICE_SEVERITY_BANDS:
        if mean_thickness <= limit:
            return label
    return "high"


def summarise_route(path, surf, cost_used, cost_type):
    """Turn a cell path into the route dict the frontend consumes."""
    lat, lon = surf["lat"], surf["lon"]
    speed, fuel = surf["speed_ms"], surf["fuel_rate_kgs"]
    thick = surf["ice_thickness_m"]

    waypoints, speed_profile = [], []
    total_km = 0.0
    total_h = 0.0
    total_fuel = 0.0
    thicknesses = []

    for k, cell in enumerate(path):
        la, lo = float(lat[cell]), float(lon[cell])
        waypoints.append((la, lo))
        v = float(speed[cell]) if np.isfinite(speed[cell]) else 0.0
        speed_profile.append((la, lo, v))
        if np.isfinite(thick[cell]):
            thicknesses.append(float(thick[cell]))

        if k > 0:
            seg_km = _cell_km(lat, lon, path[k - 1], cell)
            total_km += seg_km
            # Traverse the segment at the destination cell's speed
            if v > 1e-6:
                seg_h = (seg_km * 1000.0) / v / 3600.0
                total_h += seg_h
                fr = float(fuel[cell]) if np.isfinite(fuel[cell]) else 0.0
                total_fuel += fr * seg_h * 3600.0

    mean_thick = float(np.mean(thicknesses)) if thicknesses else 0.0
    return {
        "waypoints": waypoints,
        "total_distance_km": round(total_km, 2),
        "estimated_time_h": round(total_h, 2),
        "estimated_fuel_kg": round(total_fuel, 1),
        "ice_severity": _severity(mean_thick),
        "mean_ice_thickness_m": round(mean_thick, 3),
        "speed_profile": speed_profile,
        "cost_type": cost_type,
        "n_waypoints": len(waypoints),
        "path_cost": round(float(sum(cost_used[c] for c in path[1:])), 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calculate_routes(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float,
    polar_class: str,
    date_str: str,
    iceberg_positions: list = None,
    uncharted_bergs: list = None,
    static_hazards: list = None,
    cost_surface_path: str = None,
) -> dict:
    """
    Compute the Pareto set of routes.

    Parameters
    ----------
    start_lat/lon, end_lat/lon : endpoints in degrees (-180/180 longitude)
    polar_class                : 'PC1'..'PC7'
    date_str                   : 'YYYYMMDD'
    iceberg_positions          : [{'lat','lon','uncertainty_km'}, ...] from the
                                 iceberg LSTM (tracked BYU/NIC bergs)
    uncharted_bergs            : list of forecast_uncharted_berg() outputs
    static_hazards             : list of log_static_hazard() outputs
    cost_surface_path          : pre-built surface; built on the fly if None

    Returns
    -------
    {'routes': [...], 'recommended': {...}, 'impossible': bool, 'reason': str,
     'hazards': [...], 'data_gaps': {...}, ...}
    """
    key = str(date_str).replace("-", "")

    base_reply = {
        "routes": [],
        "recommended": None,
        "impossible": True,
        "reason": "",
        "polar_class": polar_class,
        "date": key,
        "hazards": [],
        "data_gaps": {
            "ocean_current_data": "unavailable - trajectory uncertainty elevated",
            "bathymetry": "GEBCO unavailable - no grounding check performed",
            "air_temperature": "ERA5 t2m unavailable - sea-ice physics uses "
                               "Ross Sea seasonal climatology",
        },
    }

    # --- 1. cost surface ---------------------------------------------------
    if cost_surface_path is None:
        cost_surface_path = f"cost_surface_{polar_class}_{key}.npz"
    if os.path.exists(cost_surface_path):
        surf = load_cost_surface(cost_surface_path)
    else:
        try:
            surf = build_cost_surface_for(polar_class, key)
        except Exception as e:
            base_reply["reason"] = f"could not obtain a cost surface: {e}"
            return base_reply

    lat, lon = surf["lat"], surf["lon"]
    passable = surf["passable"] & np.isfinite(surf["cost"])

    # --- 2. endpoints ------------------------------------------------------
    if not gu.in_ross_box(start_lat, start_lon):
        base_reply["reason"] = (f"start ({start_lat}, {start_lon}) is outside "
                                "the Ross Sea bounding box")
        return base_reply
    if not gu.in_ross_box(end_lat, end_lon):
        base_reply["reason"] = (f"end ({end_lat}, {end_lon}) is outside "
                                "the Ross Sea bounding box")
        return base_reply

    locator = gu.GridLocator(lat, lon, passable)
    start = locator.nearest(start_lat, start_lon)
    goal = locator.nearest(end_lat, end_lon)

    base_reply["start_cell"] = list(start)
    base_reply["end_cell"] = list(goal)
    base_reply["start_snap_km"] = round(locator.distance_km(start_lat, start_lon), 2)
    base_reply["end_snap_km"] = round(locator.distance_km(end_lat, end_lon), 2)

    if start == goal:
        base_reply["reason"] = "start and end snap to the same grid cell"
        return base_reply

    # --- 3. hazards --------------------------------------------------------
    cost_haz, hazards = apply_all_hazards(
        surf["cost"], lat, lon,
        iceberg_positions=iceberg_positions,
        uncharted_bergs=uncharted_bergs,
        static_hazards=static_hazards,
        date_str=key,
    )
    base_reply["hazards"] = [
        {k: v for k, v in h.items() if k != "trajectory"} for h in hazards
    ]

    passable_haz = passable & np.isfinite(cost_haz)
    if not passable_haz[start]:
        base_reply["reason"] = "start cell lies inside a hazard exclusion zone"
        return base_reply
    if not passable_haz[goal]:
        base_reply["reason"] = "end cell lies inside a hazard exclusion zone"
        return base_reply

    # --- 4. Pareto search --------------------------------------------------
    # alpha weights fuel, beta weights time. cost_surface.py already blends the
    # two at its own alpha/beta; we re-blend from the normalised speed and fuel
    # fields so the three Pareto routes genuinely differ.
    # Both objectives must be expressed PER UNIT DISTANCE, because the search
    # accumulates one weight per cell entered and every cell spans the same
    # ~25 km (x sqrt(2) diagonally).
    #   time per km = 1 / speed
    #   fuel per km = fuel_rate_per_second / speed
    # Using the raw per-SECOND fuel rate instead rewards slow cells - burning
    # less per second while taking far longer - which inverted the Pareto set
    # (the "time priority" route came out slower than the fuel-priority one).
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = surf["speed_ms"]
        time_term = np.where(speed > 1e-6, 1.0 / speed, np.inf)
        fuel_term = np.where(speed > 1e-6, surf["fuel_rate_kgs"] / speed, np.inf)

    def _norm(a):
        """
        Scale an objective to O(1) WITHOUT subtracting its minimum.

        Min-max normalisation is wrong here. Subtracting `lo` from every cell
        makes the path total (sum(a) - n*lo)/range depend on the NUMBER of
        cells n, so a longer detour is rewarded purely for visiting more
        cells - which is why the "time priority" route came out longer and
        slower than the fuel-priority one. Dividing by a scale keeps the
        weights strictly proportional to the real objective, so minimising the
        blended sum genuinely minimises time and fuel.
        """
        finite = a[np.isfinite(a) & passable_haz]
        if finite.size == 0:
            return np.zeros_like(a)
        scale = float(np.median(finite))
        if scale < 1e-12:
            scale = float(finite.max()) or 1.0
        return a / scale

    n_time, n_fuel = _norm(time_term), _norm(fuel_term)
    # Hazard penalty ratio, preserved across every weighting
    with np.errstate(divide="ignore", invalid="ignore"):
        haz_mult = np.where(np.isfinite(surf["cost"]) & (surf["cost"] > 1e-12),
                            cost_haz / surf["cost"], 1.0)
    haz_mult = np.where(np.isfinite(haz_mult), haz_mult, 1.0)
    haz_mult[~np.isfinite(cost_haz)] = np.inf

    routes = []
    for alpha, beta, label in PARETO_WEIGHTS:
        blended = (alpha * n_fuel + beta * n_time) / max(alpha + beta, 1e-9)
        blended = np.clip(blended, 1e-4, None) * haz_mult
        blended[~passable_haz] = np.inf

        path = astar(blended, passable_haz, lat, lon, start, goal,
                     use_heuristic=True)
        if path is None:
            # Disconnected under A*'s assumptions -> retry as plain Dijkstra
            path = astar(blended, passable_haz, lat, lon, start, goal,
                         use_heuristic=False)
        if path is None:
            continue

        r = summarise_route(path, surf, blended, label)
        r["alpha"] = alpha
        r["beta"] = beta
        routes.append(r)

    if not routes:
        base_reply["reason"] = (
            "no navigable path exists between the start and end cells for "
            f"{polar_class} on {key} - the passable graph is disconnected "
            "(ice and/or hazard exclusion zones block every route)")
        return base_reply

    recommended = next((r for r in routes if r["cost_type"] == RECOMMENDED),
                       routes[0])

    base_reply.update({
        "routes": routes,
        "recommended": recommended,
        "impossible": False,
        "reason": "",
    })
    return base_reply


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Pareto route optimiser (demo run).")
    ap.add_argument("--start_lat", type=float, default=-75.0)
    ap.add_argument("--start_lon", type=float, default=165.0)
    ap.add_argument("--end_lat", type=float, default=-73.0)
    ap.add_argument("--end_lon", type=float, default=-155.0)
    ap.add_argument("--pc", default="PC4")
    ap.add_argument("--date", default="20231115")
    args = ap.parse_args()

    res = calculate_routes(args.start_lat, args.start_lon,
                           args.end_lat, args.end_lon,
                           args.pc, args.date)
    if res["impossible"]:
        print("IMPOSSIBLE:", res["reason"])
    else:
        print(f"start snap {res['start_snap_km']} km, "
              f"end snap {res['end_snap_km']} km")
        for r in res["routes"]:
            print(f"  {r['cost_type']:15s} {r['total_distance_km']:8.1f} km  "
                  f"{r['estimated_time_h']:7.1f} h  "
                  f"{r['estimated_fuel_kg']:11.1f} kg  "
                  f"ice={r['ice_severity']}  wp={r['n_waypoints']}")
        print(f"  recommended -> {res['recommended']['cost_type']}")
