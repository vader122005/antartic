"""
cost_surface.py
===============
Antarctic Nav System — Route Cost Surface Builder
Problem Statement: SIH26059 | Team: Daði og Gagnamagnið (SIH050)

Builds a per-grid-cell cost array for the Pareto route optimiser by combining:
  1. CSV lookup table  (pc_ice_speed_model.csv)   — passability gate + speed tier
  2. Lindqvist (1989) physics model               — continuous speed & ice resistance
  3. Generalised PC engine power + SFOC           — fuel burn estimate
  4. Pareto cost function                         — α·fuel + β·time, user-adjustable

INPUT
-----
- ross_sea_concentration.npz  : (605, 133, 147) SIC grid, float32 0–100 %
- pc_ice_speed_model.csv      : IMO/IACS lookup table
- lindqvist_model.py          : must be in the same folder (imported directly)

OUTPUT
------
- cost_surface.npz : arrays keyed by 'cost', 'speed_ms', 'fuel_rate_kgs',
                     'passable', 'ice_thickness_m', 'lat', 'lon'
  Shape of each 2-D array: (133, 147) — matching the NSIDC Ross Sea grid.

ASSUMPTIONS & LIMITATIONS (document these in your dashboard)
-------------------------------------------------------------
A1. SIC → ice thickness conversion uses a simple empirical linear proxy
    (NSIDC pixel SIC % → ice type category → representative thickness).
    Real thickness requires dedicated data (e.g. ICESat-2, SMOS). Flag as
    approximation in uncertainty output.

A2. Engine power is generalised per PC class (midpoint of literature range).
    Real vessels within a class vary significantly. Flagged as assumption.

A3. SFOC = 185 g/kWh, engine efficiency η = 0.85 — representative for
    modern diesel-electric polar vessels. Constant across load levels.

A4. Lindqvist bow geometry uses DEFAULT_HULL from lindqvist_model.py.
    Swap in real hull values once a candidate vessel is chosen.

A5. Ice resistance model is for level (undeformed) ice only. Ridged ice /
    pressure ice adds resistance not captured here — covered partially by
    the RIV < 0 impassable gate.

A6. Cost weights α and β are equal by default (α=β=1.0, normalised units).
    Adjust via --alpha / --beta CLI args for different mission priorities.
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

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import Lindqvist model — must be in the same directory
# ---------------------------------------------------------------------------
try:
    from lindqvist_model import total_resistance, total_resistance_with_open_water, DEFAULT_HULL
except ImportError:
    sys.exit(
        "ERROR: lindqvist_model.py not found in the same folder as cost_surface.py.\n"
        "Place both files in the same directory and retry."
    )

# ---------------------------------------------------------------------------
# 1. GENERALISED ENGINE POWER PER POLAR CLASS
#    Source: midpoint of literature ranges (IACS, Lloyd's Register polar class
#    design studies, Lindqvist 1989, Kämäräinen 2007).
#    Units: Watts (converted from kW midpoints below)
#    ASSUMPTION A2 — replace with actual vessel spec when available.
# ---------------------------------------------------------------------------
PC_ENGINE_POWER_KW = {
    "PC1": 87_500,   # midpoint 75,000–100,000 kW
    "PC2": 60_000,   # midpoint 45,000–75,000 kW
    "PC3": 35_000,   # midpoint 25,000–45,000 kW
    "PC4": 20_000,   # midpoint 15,000–25,000 kW
    "PC5": 12_500,   # midpoint 10,000–15,000 kW
    "PC6":  8_000,   # midpoint 6,000–10,000 kW
    "PC7":  4_500,   # midpoint 3,000–6,000 kW
}

# Fraction of total shaft power available as net propulsive thrust force.
# Accounts for drivetrain losses; 0.75 is a conservative mid-value.
PROPULSIVE_EFFICIENCY = 0.75

# ---------------------------------------------------------------------------
# 2. FUEL BURN PARAMETERS
#    ASSUMPTION A3
# ---------------------------------------------------------------------------
SFOC_G_PER_KWH = 185.0          # specific fuel oil consumption, g/kWh
ENGINE_EFF = 0.85                # diesel-electric overall efficiency

# ---------------------------------------------------------------------------
# 3. SIC → ICE THICKNESS PROXY
#    Maps NSIDC sea-ice concentration (%) to a representative ice thickness
#    in metres, based on WMO ice nomenclature and typical growth rates for
#    Ross Sea first-year ice during the Nov–Mar shipping season.
#    ASSUMPTION A1 — coarse proxy only; flag in uncertainty output.
#
#    Thresholds: (SIC_lower%, SIC_upper%, thickness_m_representative)
#    Using midpoints of WMO thickness ranges per ice type.
# ---------------------------------------------------------------------------
# Each tuple: (sic_min_exclusive, sic_max_inclusive, thickness_m)
SIC_TO_THICKNESS = [
    (0.0,  15.0,  0.00),   # open water / frazil — negligible
    (15.0, 40.0,  0.05),   # new / grease ice — ~5 cm representative
    (40.0, 55.0,  0.12),   # grey ice — 10–15 cm, mid = 12 cm
    (55.0, 70.0,  0.22),   # grey-white ice — 15–30 cm, mid = 22 cm
    (70.0, 80.0,  0.40),   # thin FY 1st stage — 30–50 cm, mid = 40 cm
    (80.0, 88.0,  0.60),   # thin FY 2nd stage — 50–70 cm, mid = 60 cm
    (88.0, 93.0,  0.83),   # medium FY 1st stage — 70–95 cm, mid = 83 cm
    (93.0, 97.0,  1.08),   # medium FY 2nd stage — 95–120 cm, mid = 108 cm
    (97.0, 100.0, 1.60),   # thick FY ice — 120–200 cm, mid = 160 cm
]

def sic_to_thickness_m(sic: float) -> float:
    """
    Convert sea-ice concentration (0–100 %) to representative thickness (m).
    Returns 0.0 for open water, NaN propagated for NaN SIC.
    """
    if np.isnan(sic):
        return np.nan
    for sic_lo, sic_hi, h in SIC_TO_THICKNESS:
        if sic_lo < sic <= sic_hi:
            return h
    if sic <= 0.0:
        return 0.0
    return 1.60  # anything above 97 % → thick FY ice (conservative)


# ---------------------------------------------------------------------------
# 4. CSV LOOKUP — passability gate
#    Maps (polar_class, ice_thickness_m) → RIV value.
#    RIV < 0: impassable (treat as infinite cost).
#    RIV >= 0: passable, proceed to Lindqvist for continuous speed.
# ---------------------------------------------------------------------------
def load_csv_lookup(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def thickness_to_csv_row(df: pd.DataFrame, polar_class: str, thickness_m: float):
    """
    Find the matching CSV row for a given PC and ice thickness.
    Returns the row as a Series, or None if no match (should not happen).
    """
    pc_rows = df[df["Polar_Class"] == polar_class]
    if pc_rows.empty:
        return None

    h_cm = thickness_m * 100.0

    for _, row in pc_rows.iterrows():
        raw = str(row["Ice_Thickness_cm"]).strip()
        if raw == "0":
            if h_cm == 0.0:
                return row
        elif "-" in raw:
            lo, hi = raw.split("-")
            if float(lo) <= h_cm < float(hi):
                return row
        elif raw.endswith("+"):
            lo = float(raw[:-1])
            if h_cm >= lo:
                return row
    # Fallback: return last row for this PC (heaviest ice)
    return pc_rows.iloc[-1]


# ---------------------------------------------------------------------------
# 5. LINDQVIST CONTINUOUS SPEED SOLVER
#    Given ice thickness and available thrust, solve for actual ship speed.
# ---------------------------------------------------------------------------
def compute_continuous_speed(thrust_n: float, h_m: float) -> float:
    """
    Use Lindqvist bisection solver to find steady-state speed (m/s).
    Returns 0.0 if ship cannot move (resistance > thrust at V=0).
    """
    # The previous version returned a hard-coded 7.0 m/s whenever h <= 0.01 m
    # while letting the ice solver reach its 15 m/s ceiling in thin ice. That
    # made THIN ICE FASTER THAN OPEN WATER: on 2023-02-01, 8097 of 8599 cells
    # sat at exactly 7.0 m/s while lightly iced cells ran at 13.7-15.0, and a
    # high-ice November date produced a FASTER transit than an ice-free
    # February one over the identical path.
    #
    # The solver now always includes calm-water hull resistance, so it is well
    # posed at h = 0 and speed decreases monotonically with thickness. The
    # ceiling is the vessel's service speed rather than an arbitrary 15 m/s.
    from lindqvist_model import max_speed_for_thrust
    return max_speed_for_thrust(thrust_n, max(h_m, 0.0),
                                v_max=SERVICE_SPEED_MS,
                                include_open_water=True)


# ---------------------------------------------------------------------------
# 6. FUEL BURN CALCULATION
#    Propulsive power = R_ice × V  (Watts)
#    Fuel burn rate   = P × SFOC / η  (g/s → kg/s for output)
#    ASSUMPTION A3
# ---------------------------------------------------------------------------
def fuel_burn_kg_per_s(resistance_n: float, speed_ms: float) -> float:
    if speed_ms <= 0.0:
        return 0.0
    power_w = resistance_n * speed_ms          # propulsive power, W
    power_kw = power_w / 1000.0
    # g/s = kW × (g/kWh) / 3600
    fuel_g_per_s = power_kw * SFOC_G_PER_KWH / 3600.0 / ENGINE_EFF
    return fuel_g_per_s / 1000.0              # convert to kg/s


# ---------------------------------------------------------------------------
# 7. CELL COST FUNCTION
#    cost = α × (fuel_kg per unit distance) + β × (time per unit distance)
#    "per unit distance" normalises to a common basis (per 25 km grid cell).
#
#    time_per_cell  = CELL_SIZE_M / speed_ms          (seconds)
#    fuel_per_cell  = fuel_burn_kg_per_s × time_per_cell (kg)
#
#    Both terms are in physically meaningful units before weighting —
#    normalise each to [0,1] range across the full grid before summing
#    so α and β are truly comparable weights.
# ---------------------------------------------------------------------------
CELL_SIZE_M = 25_000.0  # 25 km — matches NSIDC pixel resolution

# Design service speed ceiling (m/s). ~15.5 knots is representative of a
# polar-class research vessel; ice can only reduce the achievable speed below
# this, never raise it.
SERVICE_SPEED_MS = 8.0


def cell_cost(speed_ms: float, fuel_rate_kgs: float,
              alpha: float = 1.0, beta: float = 1.0) -> float:
    """
    Raw (un-normalised) cost for one 25 km grid cell.
    Returns np.inf for impassable cells.
    """
    if speed_ms <= 0.0:
        return np.inf
    time_s = CELL_SIZE_M / speed_ms
    fuel_kg = fuel_rate_kgs * time_s
    return alpha * fuel_kg + beta * time_s


# ---------------------------------------------------------------------------
# 8. MAIN BUILDER — processes one date slice from the SIC .npz
# ---------------------------------------------------------------------------
def build_cost_surface(
    npz_path: Path,
    csv_path: Path,
    polar_class: str,
    date_index: int = -1,       # default: last (most recent) date in array
    alpha: float = 1.0,
    beta: float = 1.0,
    output_path: Path = Path("cost_surface.npz"),
):
    """
    Build and save the cost surface array for one date slice.

    Parameters
    ----------
    npz_path     : path to ross_sea_concentration.npz
    csv_path     : path to pc_ice_speed_model.csv
    polar_class  : e.g. 'PC4'
    date_index   : index into the (605,) dates array; -1 = latest
    alpha        : weight on fuel cost
    beta         : weight on transit time cost
    output_path  : where to save the output .npz
    """

    # Accept plain strings as well as Path objects: callers (including
    # route_optimiser's on-the-fly build) naturally pass str paths, and this
    # function calls output_path.resolve() when it finishes.
    npz_path = Path(npz_path)
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    # --- Load SIC grid ---
    print(f"[1/5] Loading SIC grid from {npz_path} ...")
    d = np.load(npz_path)
    sic_all  = d["concentration"]   # (605, 133, 147) float32
    dates    = d["dates"]           # (605,) YYYYMMDD strings
    lat      = d["lat"]             # (133, 147)
    lon      = d["lon"]             # (133, 147)

    sic_slice = sic_all[date_index]  # (133, 147)
    chosen_date = dates[date_index]
    print(f"    Date slice: {chosen_date}  |  PC: {polar_class}")

    # --- Load CSV ---
    print("[2/5] Loading CSV lookup table ...")
    df_csv = load_csv_lookup(csv_path)

    # --- Derive thrust from representative engine power ---
    power_kw   = PC_ENGINE_POWER_KW[polar_class]
    power_w    = power_kw * 1000.0
    # Net thrust (N) = (shaft power × propulsive efficiency) / assumed reference speed
    # Reference speed for thrust derivation: 2 m/s (low-speed icebreaking regime)
    REF_SPEED  = 2.0   # m/s
    thrust_n   = (power_w * PROPULSIVE_EFFICIENCY) / REF_SPEED
    print(f"    Representative engine power: {power_kw:,} kW  →  net thrust: {thrust_n/1000:.0f} kN")

    # --- Build per-cell arrays ---
    print("[3/5] Computing per-cell thickness, speed, fuel, cost ...")
    rows, cols = sic_slice.shape
    out_passable    = np.zeros((rows, cols), dtype=bool)
    out_thickness   = np.full((rows, cols), np.nan, dtype=np.float32)
    out_speed_ms    = np.zeros((rows, cols), dtype=np.float32)
    out_fuel_kgs    = np.zeros((rows, cols), dtype=np.float32)
    out_cost        = np.full((rows, cols), np.inf, dtype=np.float32)

    sic_to_h = np.vectorize(sic_to_thickness_m)
    thickness_grid = sic_to_h(sic_slice).astype(np.float32)

    for i in range(rows):
        for j in range(cols):
            sic = float(sic_slice[i, j])

            # NaN = land or outside bounding box → impassable
            if np.isnan(sic):
                continue

            h = float(thickness_grid[i, j])
            out_thickness[i, j] = h

            # CSV gate: check RIV
            csv_row = thickness_to_csv_row(df_csv, polar_class, h)
            if csv_row is None:
                continue
            riv = int(csv_row["RIV_Official_IMO"])
            if riv < 0:
                # Impassable for this PC at this ice thickness
                continue

            # Lindqvist: continuous speed
            v = compute_continuous_speed(thrust_n, h)
            if v <= 0.0:
                continue

            # Ice + calm-water resistance at that speed (for fuel calc).
            # Ice resistance alone is exactly zero in open water, which made
            # 97% of cells report zero fuel burn; see lindqvist_model.
            r = total_resistance_with_open_water(v, h, DEFAULT_HULL)

            # Fuel burn
            fb = fuel_burn_kg_per_s(r, v)

            # Cell cost
            c = cell_cost(v, fb, alpha, beta)

            out_passable[i, j]  = True
            out_speed_ms[i, j]  = v
            out_fuel_kgs[i, j]  = fb
            out_cost[i, j]      = c

    # --- Normalise cost to [0, 1] across passable cells (for Pareto optimiser) ---
    print("[4/5] Normalising cost surface ...")
    finite_mask = np.isfinite(out_cost)
    if finite_mask.any():
        c_min = out_cost[finite_mask].min()
        c_max = out_cost[finite_mask].max()
        out_cost_norm = np.full_like(out_cost, np.inf)
        if c_max > c_min:
            out_cost_norm[finite_mask] = (out_cost[finite_mask] - c_min) / (c_max - c_min)
        else:
            out_cost_norm[finite_mask] = 0.0
    else:
        out_cost_norm = out_cost.copy()
        print("    WARNING: no passable cells found for this PC / date combination.")

    # --- Save ---
    print(f"[5/5] Saving to {output_path} ...")
    np.savez_compressed(
        output_path,
        cost=out_cost_norm,           # (133,147) float32, normalised [0,1]; inf = impassable
        cost_raw=out_cost,            # (133,147) float32, raw α·fuel + β·time units
        speed_ms=out_speed_ms,        # (133,147) float32, m/s
        fuel_rate_kgs=out_fuel_kgs,   # (133,147) float32, kg/s
        passable=out_passable,        # (133,147) bool
        ice_thickness_m=thickness_grid, # (133,147) float32
        lat=lat,
        lon=lon,
        metadata=np.array([
            f"date={chosen_date}",
            f"polar_class={polar_class}",
            f"alpha={alpha}",
            f"beta={beta}",
            f"engine_kw={power_kw}",
            f"thrust_kn={thrust_n/1000:.0f}",
            f"sfoc_g_per_kwh={SFOC_G_PER_KWH}",
            f"engine_eff={ENGINE_EFF}",
            f"cell_size_m={CELL_SIZE_M}",
            "ASSUMPTION_A1=SIC_to_thickness_is_proxy_only",
            "ASSUMPTION_A2=engine_power_is_class_generalisation",
            "ASSUMPTION_A3=SFOC_and_eta_are_representative_constants",
        ])
    )

    passable_count = out_passable.sum()
    total_cells = rows * cols
    print(f"\n    Done. {passable_count}/{total_cells} cells passable "
          f"({100*passable_count/total_cells:.1f}%)")
    print(f"    Cost range (raw): {out_cost[finite_mask].min():.2f} – "
          f"{out_cost[finite_mask].max():.2f}")
    print(f"    Output: {output_path.resolve()}")
    return output_path


# ---------------------------------------------------------------------------
# 9. HOW TO LOAD THE OUTPUT (for the route optimiser)
# ---------------------------------------------------------------------------
LOAD_EXAMPLE = """
# In your route optimiser, load like this:
#
#   import numpy as np
#   d = np.load('cost_surface.npz')
#
#   cost      = d['cost']          # (133,147) — use this as edge weights in graph search
#   passable  = d['passable']      # (133,147) bool — False = wall node, skip entirely
#   speed_ms  = d['speed_ms']      # (133,147) — for waypoint speed annotations
#   fuel_rate = d['fuel_rate_kgs'] # (133,147) — for fuel summary panel
#   lat, lon  = d['lat'], d['lon'] # (133,147) — for Mapbox coordinate mapping
#
# Graph search tip: build an 8-connected grid graph. For each edge (i,j)→(i',j'),
# use the DESTINATION cell's cost as the edge weight (standard for grid Dijkstra).
# Impassable cells (cost == inf or passable == False) are simply not added as nodes.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Antarctic nav cost surface from SIC grid + Lindqvist model."
    )
    parser.add_argument(
        "--sic_npz",
        type=Path,
        default=Path("processed/ross_sea_concentration.npz"),
        help="Path to ross_sea_concentration.npz (default: processed/ross_sea_concentration.npz)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("pc_ice_speed_model.csv"),
        help="Path to pc_ice_speed_model.csv",
    )
    parser.add_argument(
        "--pc",
        type=str,
        default="PC4",
        choices=list(PC_ENGINE_POWER_KW.keys()),
        help="Polar class (default: PC4)",
    )
    parser.add_argument(
        "--date_index",
        type=int,
        default=-1,
        help="Index into dates array (-1 = latest date, default)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight on fuel cost in Pareto objective (default: 1.0)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight on transit time in Pareto objective (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed/cost_surface.npz"),
        help="Output .npz path (default: processed/cost_surface.npz)",
    )
    args = parser.parse_args()

    print(LOAD_EXAMPLE)

    build_cost_surface(
        npz_path=args.sic_npz,
        csv_path=args.csv,
        polar_class=args.pc,
        date_index=args.date_index,
        alpha=args.alpha,
        beta=args.beta,
        output_path=args.output,
    )
