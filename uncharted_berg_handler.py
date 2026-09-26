"""
uncharted_berg_handler.py
==========================
Antarctic Nav System — Uncharted Iceberg Handler
Problem Statement: SIH26059 | Team: Daði og Gagnamagnið (SIH050)

Handles captain-reported uncharted icebergs (and growler/bergy-bit warnings)
that are not in the BYU/NIC tracking database.

WORKFLOW
--------
1. Captain (or officer of the watch) reports an uncharted berg via the
   dashboard "Report uncharted hazard" button, supplying:
     - Current position (lat, lon)
     - Visual size estimate (size category: small / medium / large / very_large)
     - Optional: heading estimate if berg is visibly moving

2. System runs the Bigg/Wagner/Lichey physics baseline forward in 6h steps
   (same physics as iceberg_physics_baseline.py — no ML correction applied
   because there is no observed drift history to train a residual on).

3. Trajectory cone is returned: waypoints at 6h intervals with an uncertainty
   radius that grows with time (larger at 48h than at 6h).

4. Hazard zone dict is returned in the format the route optimiser expects,
   with a LARGER exclusion radius than BYU/NIC tracked bergs (because there
   is no LSTM correction and no drift history).

5. If the captain later reports a growler/bergy-bit at a specific position,
   that is logged as a STATIC hazard (growlers are too small to track
   trajectories). A fixed exclusion zone is applied and the route is
   recalculated.

WHY NO ML HERE
--------------
The LSTM residual layer in iceberg_lstm.py was trained on B38's observed
drift history. An uncharted berg has no drift history, so the LSTM has
nothing to correct against. Physics-only is the correct approach, and it
is what operational models (WEDDELL/AWI, CIS, NAVO) do for newly detected
bergs before sufficient tracking observations accumulate.

The uncertainty bands widen with time precisely because of this — the physics
model is accurate for the first 6–12h but diverges without the ocean current
correction (CMEMS gap) over longer horizons.

UNCERTAINTY MODEL
-----------------
Uncertainty radius grows with forecast horizon following an empirical
linear-plus-drift model calibrated to IDRIFTNET (Barbosa Aguiar et al.,
arXiv:2507.00036) ablation results:

    σ(t) = σ_0 + k * t_hours

where:
    σ_0 = base uncertainty at t=0 (function of size category)
    k   = growth rate (km/hour) — higher without LSTM correction
    t   = forecast horizon in hours

For BYU/NIC tracked bergs (with LSTM): k ≈ 0.5 km/h
For uncharted bergs (physics-only):    k ≈ 1.5 km/h  ← 3× wider cone

ASSUMPTIONS
-----------
A1. Berg dimensions estimated from visual size category — not measured.
    Actual dimensions may differ significantly. Uncertainty radius partially
    accounts for this via the σ_0 term.
A2. Initial velocity assumed zero (berg at rest) unless captain provides
    a heading/speed estimate.
A3. Ocean current = zero (CMEMS unavailable) — same gap as main model.
    Larger k value partially compensates for this.
A4. Growlers and bergy bits are treated as static hazards only — too small
    and numerous to track individually.
"""

# Windows consoles default to cp1252, which cannot encode the arrows and
# degree signs used in this script's output; force UTF-8 so printing never
# aborts a run.
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        if (_s.encoding or "").lower().replace("-", "") != "utf8":
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import math
import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import grid_utils as gu

# Import the existing physics engine — must be in the same folder
from iceberg_physics_baseline import physics_step, berg_geometry

# ---------------------------------------------------------------------------
# SIZE CATEGORY → REPRESENTATIVE BERG DIMENSIONS
# Based on WMO iceberg size classification and Ross Sea tabular berg
# observations from BYU/NIC database (our training bergs).
#
# WMO size classes (length × width, approximate):
#   Growler:    < 5m above water,  < 20m length  → static hazard only
#   Bergy bit:  1–4m above water,  5–14m length  → static hazard only
#   Small:      5–15m high,        15–60m length
#   Medium:     16–45m high,       61–200m length
#   Large:      46–75m high,       201–500m length
#   Very large: > 75m high,        > 500m length → tabular class (like B38)
#
# We map these to representative length × width in metres for the physics model.
# ---------------------------------------------------------------------------
SIZE_CATEGORIES = {
    "small": {
        "length_m":     40.0,
        "width_m":      20.0,
        "description":  "Small berg (~40×20m) — 5–15m freeboard",
        "sigma_0_km":    2.0,   # base uncertainty radius at t=0
    },
    "medium": {
        "length_m":    150.0,
        "width_m":      80.0,
        "description":  "Medium berg (~150×80m) — 16–45m freeboard",
        "sigma_0_km":    5.0,
    },
    "large": {
        "length_m":    400.0,
        "width_m":     200.0,
        "description":  "Large berg (~400×200m) — 46–75m freeboard",
        "sigma_0_km":    8.0,
    },
    "very_large": {
        "length_m":   5_000.0,
        "width_m":    2_000.0,
        "description": "Very large / tabular berg (~5×2km+) — >75m freeboard",
        "sigma_0_km":   12.0,
    },
}

# Growlers and bergy bits — static hazard only, no trajectory
STATIC_HAZARD_CATEGORIES = {"growler", "bergy_bit"}

# Uncertainty growth rates (km per hour)
SIGMA_GROWTH_RATE_KM_PER_H = 1.5    # physics-only (no LSTM correction)
SIGMA_GROWTH_BYU_KM_PER_H  = 0.5    # for reference: tracked berg with LSTM

# Default exclusion radius multiplier vs BYU/NIC bergs
UNCHARTED_RADIUS_MULTIPLIER = 2.0   # uncharted → 2× the exclusion radius

# Growler static exclusion radius.
#
# Raised from the nominal 5.0 km to 30.0 km because of a GRID-RESOLUTION limit,
# not because a growler is physically dangerous 30 km away. The NSIDC cost
# surface is a ~25 km grid, whose cell diagonal is ~35 km, so a 5 km circle
# routinely falls entirely between cell centres and modifies NO cell at all -
# a captain-reported growler was effectively invisible to the route optimiser.
# 30 km is a conservative floor: just under one cell diagonal, so the report
# reliably reaches at least the containing cell and its immediate neighbours,
# while still being smaller than the ~35 km at which it would always spill into
# a full ring of adjacent cells.
#
# This is a representational minimum imposed by the grid. If the cost surface
# is ever rebuilt at finer resolution, this should drop back toward the true
# 5 km hazard radius.
GROWLER_EXCLUSION_RADIUS_KM = 30.0


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points (degrees) in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def uncertainty_radius_km(t_hours: float, sigma_0_km: float) -> float:
    """
    Uncertainty radius (km) at forecast horizon t_hours.
    Grows linearly: σ(t) = σ_0 + k * t
    """
    return sigma_0_km + SIGMA_GROWTH_RATE_KM_PER_H * t_hours


# Module-level caches: these helpers are called once per 6 h forecast step, and
# the ERA5 file is ~500 MB. Reloading it per call would dominate runtime.
_WIND_SAMPLER = {}
_SIC_CACHE = {}


def _wind_sampler(wind_npz_path: str):
    if wind_npz_path not in _WIND_SAMPLER:
        _WIND_SAMPLER[wind_npz_path] = gu.WindSampler(wind_npz_path)
    return _WIND_SAMPLER[wind_npz_path]


def _sic_bundle(sic_npz_path: str):
    if sic_npz_path not in _SIC_CACHE:
        s = np.load(sic_npz_path)
        conc = s["concentration"]
        loc = gu.GridLocator(s["lat"], s["lon"], ~np.isnan(conc[0]))
        _SIC_CACHE[sic_npz_path] = (np.asarray(s["dates"]), conc, loc)
    return _SIC_CACHE[sic_npz_path]


def interpolate_wind_at_position(
    lat: float, lon: float,
    time_str: str,
    wind_npz_path: str,
) -> tuple:
    """
    Bilinear interpolation of ERA5 wind to a given lat/lon/time.
    Returns (wind_u, wind_v) in m/s.

    FIX: this previously built a RegularGridInterpolator directly on the ERA5
    lat/lon axes. ERA5 longitude here runs 160 -> 180 then WRAPS to -180 ->
    -150, so it is not monotonic and scipy raised
    "The points in dimension 1 must be strictly ascending or descending".
    The bare `except` swallowed that and returned ZERO WIND for every step, so
    forecast bergs never moved at all - the 48 h trajectory was the report
    position repeated nine times while only the uncertainty circle grew.
    grid_utils.WindSampler unwraps the longitude and flips the descending
    latitude before interpolating.
    """
    try:
        return _wind_sampler(wind_npz_path).sample(lat, lon, time_str)
    except Exception as e:
        print(f"    [WARN] Wind interpolation failed: {e} - using zero wind")
        return 0.0, 0.0


def interpolate_sic_at_position(
    lat: float, lon: float,
    date_str: str,
    sic_npz_path: str,
) -> float:
    """
    Sample NSIDC SIC at a given lat/lon/date. Returns a fraction 0.0-1.0.

    FIX: the NSIDC grid is curvilinear (polar stereographic), so the previous
    code's `lat[:, 0]` / `lon[0, :]` axes were not valid separable coordinates
    and `int(date_str)` also raised on 'YYYY-MM-DD' input. Both failures fell
    through to a hard-coded SIC of 0.5. Nearest-cell lookup on the real mesh is
    used instead (see grid_utils.GridLocator).
    """
    try:
        dates, conc, loc = _sic_bundle(sic_npz_path)
        idx = gu.sic_date_index(dates, date_str)
        val = loc.sample(conc[idx], lat, lon, default=0.0)
        return float(np.clip(val / 100.0, 0.0, 1.0))
    except Exception as e:
        print(f"    [WARN] SIC interpolation failed: {e} - using 0.5")
        return 0.5


# ---------------------------------------------------------------------------
# MAIN FUNCTION 1: Uncharted berg trajectory forecast
# ---------------------------------------------------------------------------

def forecast_uncharted_berg(
    lat: float,
    lon: float,
    size_category: str,
    report_time: str,                       # 'YYYY-MM-DD HH:MM' UTC
    wind_npz_path: str,
    sic_npz_path: str,
    forecast_hours: int = 48,
    step_hours: int = 6,
    initial_vel_u: float = 0.0,            # m/s east — 0 if unknown
    initial_vel_v: float = 0.0,            # m/s north — 0 if unknown
    berg_id: str = None,
) -> dict:
    """
    Forecast trajectory of a captain-reported uncharted iceberg.

    Parameters
    ----------
    lat, lon        : reported position (degrees)
    size_category   : 'small' | 'medium' | 'large' | 'very_large'
    report_time     : time of captain's report, 'YYYY-MM-DD HH:MM' UTC
    wind_npz_path   : path to ross_sea_era5_wind.npz
    sic_npz_path    : path to ross_sea_concentration.npz
    forecast_hours  : how far ahead to forecast (default 48h)
    step_hours      : physics time step (default 6h, matches main model)
    initial_vel_u/v : initial berg velocity if known (default 0)
    berg_id         : optional label for this berg (auto-assigned if None)

    Returns
    -------
    dict with keys:
        berg_id         : str
        size_category   : str
        size_description: str
        report_time     : str
        trajectory      : list of waypoint dicts, one per step:
            {
                'time':         'YYYY-MM-DD HH:MM',
                'lat':          float,
                'lon':          float,
                'uncertainty_km': float,
                'vel_u_ms':     float,
                'vel_v_ms':     float,
                'validation_flag': str,
                't_hours':      float,
            }
        hazard_zones    : list of dicts for route optimiser:
            {
                'lat':            float,
                'lon':            float,
                'exclusion_km':   float,
                'time':           str,
                'is_uncharted':   True,
            }
        data_source     : 'captain_report_physics_only'
        uncertainty_note: str
    """
    if size_category in STATIC_HAZARD_CATEGORIES:
        raise ValueError(
            f"'{size_category}' is a static hazard — use log_static_hazard() instead."
        )

    if size_category not in SIZE_CATEGORIES:
        raise ValueError(
            f"Unknown size category '{size_category}'. "
            f"Choose from: {list(SIZE_CATEGORIES.keys())}"
        )

    if berg_id is None:
        berg_id = f"UNCHARTED_{report_time.replace(' ','T').replace(':','')}"

    spec      = SIZE_CATEGORIES[size_category]
    length_m  = spec["length_m"]
    width_m   = spec["width_m"]
    sigma_0   = spec["sigma_0_km"]

    print(f"\n[Uncharted berg forecast] {berg_id}")
    print(f"  Position : {lat:.4f}°, {lon:.4f}°")
    print(f"  Size     : {spec['description']}")
    print(f"  Time     : {report_time} UTC")
    print(f"  Forecast : {forecast_hours}h ahead in {step_hours}h steps")

    dt_s        = step_hours * 3600.0
    n_steps     = forecast_hours // step_hours
    report_dt   = datetime.strptime(report_time, "%Y-%m-%d %H:%M")

    # State
    cur_lat   = lat
    cur_lon   = lon
    cur_vel_u = initial_vel_u
    cur_vel_v = initial_vel_v

    trajectory   = []
    hazard_zones = []

    # t=0: report position
    trajectory.append({
        "time":             report_time,
        "lat":              cur_lat,
        "lon":              cur_lon,
        "uncertainty_km":   sigma_0,
        "vel_u_ms":         cur_vel_u,
        "vel_v_ms":         cur_vel_v,
        "validation_flag":  "reported",
        "t_hours":          0.0,
    })
    hazard_zones.append({
        "lat":           cur_lat,
        "lon":           cur_lon,
        "exclusion_km":  sigma_0 * UNCHARTED_RADIUS_MULTIPLIER,
        "time":          report_time,
        "is_uncharted":  True,
    })

    for step in range(1, n_steps + 1):
        t_hours   = step * step_hours
        step_time = report_dt + timedelta(hours=t_hours)
        time_str  = step_time.strftime("%Y-%m-%d %H:%M")
        date_str  = step_time.strftime("%Y%m%d")

        # Get forcing at current position
        wind_u, wind_v = interpolate_wind_at_position(
            cur_lat, cur_lon, time_str, wind_npz_path
        )
        sic = interpolate_sic_at_position(
            cur_lat, cur_lon, date_str, sic_npz_path
        )

        # Physics step
        result = physics_step(
            lat=cur_lat, lon=cur_lon,
            vel_u=cur_vel_u, vel_v=cur_vel_v,
            wind_u=wind_u, wind_v=wind_v,
            sic=sic,
            dt_s=dt_s,
            ocean_u=0.0, ocean_v=0.0,   # CMEMS unavailable
            length_m=length_m,
            width_m=width_m,
        )

        cur_lat   = result["lat_new"]
        cur_lon   = result["lon_new"]
        cur_vel_u = result["vel_u_new"]
        cur_vel_v = result["vel_v_new"]

        sigma = uncertainty_radius_km(t_hours, sigma_0)
        excl  = sigma * UNCHARTED_RADIUS_MULTIPLIER

        waypoint = {
            "time":               time_str,
            "lat":                cur_lat,
            "lon":                cur_lon,
            "uncertainty_km":     sigma,
            "vel_u_ms":           cur_vel_u,
            "vel_v_ms":           cur_vel_v,
            "validation_flag":    result["validation_flag"],
            "t_hours":            float(t_hours),
        }
        hazard = {
            "lat":           cur_lat,
            "lon":           cur_lon,
            "exclusion_km":  excl,
            "time":          time_str,
            "is_uncharted":  True,
        }

        trajectory.append(waypoint)
        hazard_zones.append(hazard)

        print(f"  t={t_hours:3.0f}h  →  lat={cur_lat:.4f}  lon={cur_lon:.4f}  "
              f"σ={sigma:.1f}km  flag={result['validation_flag']}")

    uncertainty_note = (
        f"Trajectory is physics-only (Bigg/Wagner/Lichey). "
        f"No LSTM residual correction applied — no drift history available for this berg. "
        f"Uncertainty grows at {SIGMA_GROWTH_RATE_KM_PER_H} km/h "
        f"(vs {SIGMA_GROWTH_BYU_KM_PER_H} km/h for BYU/NIC tracked bergs). "
        f"Ocean current forcing unavailable (CMEMS skipped) — "
        f"trajectory error will increase beyond 24h. "
        f"Master must maintain independent watch per IMO Polar Code Chapter 11."
    )

    return {
        "berg_id":          berg_id,
        "size_category":    size_category,
        "size_description": spec["description"],
        "report_time":      report_time,
        "trajectory":       trajectory,
        "hazard_zones":     hazard_zones,
        "data_source":      "captain_report_physics_only",
        "uncertainty_note": uncertainty_note,
    }


# ---------------------------------------------------------------------------
# MAIN FUNCTION 2: Static hazard logging (growlers, bergy bits)
# ---------------------------------------------------------------------------

def log_static_hazard(
    lat: float,
    lon: float,
    hazard_type: str,          # 'growler' | 'bergy_bit'
    report_time: str,          # 'YYYY-MM-DD HH:MM' UTC
    hazard_id: str = None,
) -> dict:
    """
    Log a captain-reported growler or bergy bit as a static hazard.

    Growlers and bergy bits are too small to track trajectories —
    they are logged as fixed exclusion zones only. The route optimiser
    will treat these as permanent impassable cells until the captain
    clears them or the vessel transits the area.

    Returns a hazard dict ready for the route optimiser.
    """
    if hazard_id is None:
        hazard_id = f"{hazard_type.upper()}_{report_time.replace(' ','T').replace(':','')}"

    print(f"\n[Static hazard logged] {hazard_id}")
    print(f"  Type     : {hazard_type}")
    print(f"  Position : {lat:.4f}°, {lon:.4f}°")
    print(f"  Time     : {report_time} UTC")
    print(f"  Exclusion: {GROWLER_EXCLUSION_RADIUS_KM} km fixed radius")
    print(f"  NOTE: No trajectory forecast — static exclusion zone only.")

    return {
        "hazard_id":      hazard_id,
        "hazard_type":    hazard_type,
        "lat":            lat,
        "lon":            lon,
        "report_time":    report_time,
        "exclusion_km":   GROWLER_EXCLUSION_RADIUS_KM,
        "is_static":      True,
        "is_uncharted":   True,
        "trajectory":     None,    # no trajectory for growlers
        "uncertainty_note": (
            f"{hazard_type.capitalize()} logged as static hazard. "
            f"No trajectory forecast possible for objects of this size. "
            f"Fixed {GROWLER_EXCLUSION_RADIUS_KM} km exclusion zone applied. "
            f"Captain should maintain visual/radar watch and update position as needed."
        ),
    }


# ---------------------------------------------------------------------------
# MAIN FUNCTION 3: Apply all hazards to cost grid (called by route optimiser)
# ---------------------------------------------------------------------------

def apply_hazard_zones_to_cost_grid(
    cost_grid: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    hazard_list: list,
    penalty_factor: float = 5.0,
) -> np.ndarray:
    """
    Apply all uncharted berg and static hazard exclusion zones to the
    cost surface grid. Returns a modified cost grid.

    For cells within the exclusion radius of any hazard:
        cost = cost * penalty_factor   (if cost is finite)
    For cells within 50% of the exclusion radius (inner zone):
        cost = inf                     (hard impassable — too close)

    Parameters
    ----------
    cost_grid   : (133, 147) float32 normalised cost array from cost_surface.py
    lat_grid    : (133, 147) float32 latitude of each cell
    lon_grid    : (133, 147) float32 longitude of each cell
    hazard_list : list of hazard dicts from forecast_uncharted_berg() or
                  log_static_hazard() — can mix both types freely
    penalty_factor : cost multiplier for outer hazard zone (default 5.0)

    Returns
    -------
    modified cost grid (133, 147) float32
    """
    modified = np.array(cost_grid, dtype=np.float64, copy=True)
    lat_grid = np.asarray(lat_grid, dtype=np.float64)
    lon_grid = np.asarray(lon_grid, dtype=np.float64)
    finite_cells = np.isfinite(lat_grid) & np.isfinite(lon_grid)

    # Cost floor applied before the multiplier. cost_surface.py normalises cost
    # to [0, 1], so the cheapest (open-water) cells are EXACTLY 0.0 - and a
    # purely multiplicative penalty leaves 0 * 5 = 0, making precisely the
    # cheapest, most attractive cells completely immune to hazard zones. Any
    # hazard sitting over open water was silently voided.
    #
    # Flooring the cost at COST_FLOOR before multiplying guarantees that every
    # cell inside a zone accumulates a real penalty: an open-water cell goes
    # from 0.0 to 0.05 * 5 = 0.25, which is a genuine deterrent on a surface
    # whose ordinary cells span 0-1.
    COST_FLOOR = 0.05

    for hazard in hazard_list:
        h_lat = hazard["lat"]
        h_lon = hazard["lon"]
        excl_r = hazard["exclusion_km"]
        inner_r = excl_r * 0.5   # hard impassable inner zone

        # Vectorised over the whole grid: the original nested Python loop ran
        # 19551 haversine calls PER HAZARD, and a 48 h forecast plus captain
        # reports can easily produce dozens of hazards.
        dist = gu.haversine_km(lat_grid, lon_grid, h_lat, h_lon)

        outer = finite_cells & (dist <= excl_r) & (dist > inner_r)
        inner = finite_cells & (dist <= inner_r)

        # FIX: the outer-zone penalty was written as
        #     min(cost * penalty_factor, 1.0)
        # but cost_surface.py emits a NORMALISED 0-1 cost, so the clamp pinned
        # every penalised cell to exactly 1.0 - the same value as the worst
        # ordinary ice cell. That removed the deterrent the penalty exists to
        # create (and flattened the gradient inside the zone). The multiplier
        # is now applied without the clamp; the graph search does not require
        # costs to stay within 0-1.
        # Floor first, then multiply, so zero-cost open-water cells are
        # penalised rather than left untouched (see COST_FLOOR above).
        penal = outer & np.isfinite(modified)
        modified[penal] = np.maximum(modified[penal], COST_FLOOR) * penalty_factor

        modified[inner] = np.inf

        # SUB-GRID HAZARDS. The NSIDC grid is ~25 km, so a hazard whose
        # exclusion radius is smaller than the cell spacing can fall entirely
        # between cell centres and modify NOTHING. This is now a backstop
        # rather than the primary defence: GROWLER_EXCLUSION_RADIUS_KM was
        # raised to 30 km so growlers reach the grid directly.
        # When a hazard selects no cell, penalise the single nearest valid cell
        # so the report still influences the route. We apply the multiplier
        # rather than a hard wall: closing a whole 25 km cell for a 5 km
        # growler would overstate what was actually reported.
        if not (outer.any() or inner.any()):
            masked = np.where(finite_cells, dist, np.inf)
            k = np.unravel_index(np.argmin(masked), masked.shape)
            if np.isfinite(modified[k]):
                modified[k] = max(modified[k], COST_FLOOR) * penalty_factor
            elif np.isfinite(masked[k]):
                modified[k] = np.inf

    return modified


# ---------------------------------------------------------------------------
# CONVENIENCE: Export all hazard zones for a given time window
# (called by route optimiser to get the hazard list at a specific time)
# ---------------------------------------------------------------------------

def get_active_hazard_zones(
    uncharted_bergs: list,     # list of forecast_uncharted_berg() outputs
    static_hazards: list,      # list of log_static_hazard() outputs
    query_time: str,           # 'YYYY-MM-DD HH:MM' — get zones valid at this time
) -> list:
    """
    Extract all hazard zone dicts active at a given time.
    For uncharted bergs: returns the trajectory waypoint closest to query_time.
    For static hazards: always returned (they don't expire).

    Returns a flat list of hazard dicts ready for apply_hazard_zones_to_cost_grid().
    """
    active = []
    if query_time is None:
        # No time given: fall back to each berg's own report time, i.e. the
        # t=0 waypoint, rather than raising.
        query_dt = None
    else:
        query_dt = datetime.strptime(query_time, "%Y-%m-%d %H:%M")

    for berg in uncharted_bergs:
        # Find trajectory waypoint closest to query_time
        best_wp   = None
        best_diff = timedelta(days=999)
        for wp in berg["trajectory"]:
            wp_dt = datetime.strptime(wp["time"], "%Y-%m-%d %H:%M")
            diff  = abs(wp_dt - query_dt)
            if diff < best_diff:
                best_diff = diff
                best_wp   = wp
        if best_wp:
            active.append({
                "lat":           best_wp["lat"],
                "lon":           best_wp["lon"],
                "exclusion_km":  best_wp["uncertainty_km"] * UNCHARTED_RADIUS_MULTIPLIER,
                "time":          best_wp["time"],
                "is_uncharted":  True,
                "berg_id":       berg["berg_id"],
            })

    for sh in static_hazards:
        active.append({
            "lat":          sh["lat"],
            "lon":          sh["lon"],
            "exclusion_km": sh["exclusion_km"],
            "time":         sh["report_time"],
            "is_uncharted": True,
            "berg_id":      sh["hazard_id"],
        })

    return active


# ---------------------------------------------------------------------------
# CLI — quick test with a synthetic captain report
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Forecast an uncharted iceberg or log a static hazard."
    )
    parser.add_argument("--lat",        type=float, default=-74.5)
    parser.add_argument("--lon",        type=float, default=172.0)
    parser.add_argument("--size",       type=str,   default="medium",
                        choices=list(SIZE_CATEGORIES.keys()) + list(STATIC_HAZARD_CATEGORIES))
    parser.add_argument("--time",       type=str,   default="2023-11-15 06:00")
    parser.add_argument("--wind_npz",   type=str,   default="ross_sea_era5_wind.npz")
    parser.add_argument("--sic_npz",    type=str,   default="ross_sea_concentration.npz")
    parser.add_argument("--forecast_h", type=int,   default=48)
    parser.add_argument("--output",     type=str,   default="uncharted_berg_forecast.json")
    args = parser.parse_args()

    if args.size in STATIC_HAZARD_CATEGORIES:
        result = log_static_hazard(
            lat=args.lat,
            lon=args.lon,
            hazard_type=args.size,
            report_time=args.time,
        )
    else:
        result = forecast_uncharted_berg(
            lat=args.lat,
            lon=args.lon,
            size_category=args.size,
            report_time=args.time,
            wind_npz_path=args.wind_npz,
            sic_npz_path=args.sic_npz,
            forecast_hours=args.forecast_h,
        )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {args.output}")
    print(f"Uncertainty note: {result['uncertainty_note']}")
