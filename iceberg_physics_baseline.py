"""
iceberg_physics_baseline.py
============================
Antarctic Nav System — Iceberg Drift Physics Baseline
Problem Statement: SIH26059 | Team: Daði og Gagnamagnið (SIH050)

Implements the physics-based iceberg drift model as the baseline component of
the residual-learning hybrid architecture. The ML residual layer (built separately
by Claude Code) will learn Δ(observed position − physics position) on top of this.

PHYSICS CHAIN
-------------
1. Bigg et al. (1997) — force balance momentum equation (structure)
2. Wagner et al. (2017) — analytical solution to the momentum balance
   (avoids expensive numerical ODE integration for hackathon timescales)
3. Lichey & Hellmer (2001) — sea-ice drag regime switcher
   (modifies ocean drag coefficient based on local SIC)

REFERENCES
----------
- Bigg, G.R., Wadley, M.R., Stevens, D.P., Johnson, J.A. (1997).
  "Modelling the dynamics and thermodynamics of icebergs."
  Cold Regions Science and Technology, 26(2), 113–135.

- Wagner, T.J.W., Dell, R.W., Eisenman, I. (2017).
  "An analytical model of iceberg drift."
  Journal of Physical Oceanography, 47(7), 1605–1616.
  https://doi.org/10.1175/JPO-D-16-0262.1

- Lichey, C., Hellmer, H.H. (2001).
  "Modeling giant-iceberg drift under the influence of sea ice in the
  Weddell Sea, Antarctica."
  Journal of Glaciology, 47(158), 452–460.

INPUTS (per time step)
----------------------
- berg state : lat, lon, vel_u (m/s east), vel_v (m/s north)
- wind       : u10, v10 (m/s) — ERA5 10m wind at berg location
- ocean curr : uo, vo (m/s) — NOT available (CMEMS skipped for hackathon scope)
               → set to zero; flagged as Assumption A1, increases residual
- SIC        : local sea-ice concentration (0–1) — from NSIDC .npz

OUTPUT (per time step)
----------------------
- new lat, lon after dt seconds
- new vel_u, vel_v (m/s)
These are the "physics prediction" that the ML residual layer corrects.

ASSUMPTIONS & LIMITATIONS
--------------------------
A1. Ocean current set to zero (CMEMS skipped). This is the largest source of
    error for large tabular icebergs where Stokes drift and geostrophic flow
    dominate. Surface explicitly in uncertainty output.
A2. Iceberg treated as a uniform rectangular prism (tabular berg geometry).
    Size estimated from BYU size_1, size_2 columns where available;
    falls back to B38 representative dimensions.
A3. Coriolis parameter computed at each berg latitude — included.
A4. Sea-surface tilt (pressure gradient) term omitted — minor for short forecasts.
A5. Melt / size change omitted within a single forecast step (valid for 6–12h).
A6. Wagner et al. (2017) analytical solution assumes slowly varying forcing —
    valid for 6h steps, less so for rapidly changing wind events.
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
import numpy as np

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------
OMEGA   = 7.2921e-5   # Earth rotation rate, rad/s
RHO_W   = 1025.0      # seawater density, kg/m³
RHO_I   = 917.0       # glacier ice density, kg/m³  (tabular berg; denser than sea ice)
RHO_A   = 1.225       # air density, kg/m³
G       = 9.81        # gravity, m/s²
R_EARTH = 6.371e6     # Earth radius, m

# ---------------------------------------------------------------------------
# DRAG & ADDED MASS COEFFICIENTS
# Standard values from Bigg et al. (1997) Table 1 and Wagner et al. (2017)
# ---------------------------------------------------------------------------
C_W  = 0.9    # ocean (water) drag coefficient on berg keel
C_A  = 1.3    # air drag coefficient on berg freeboard
C_SI = 0.0    # sea-ice skin drag on waterline — overridden by Lichey & Hellmer below
C_AM = 0.5    # added mass coefficient (fluid inertia around berg)

# ---------------------------------------------------------------------------
# REPRESENTATIVE BERG GEOMETRY — B38 (primary training berg)
# From BYU size_1 / size_2 columns (km). B38 is a large tabular berg.
# These are used as fallback when actual size data is unavailable.
# ---------------------------------------------------------------------------
B38_LENGTH_M = 40_000.0   # ~40 km along-track (representative mid-life size)
B38_WIDTH_M  = 25_000.0   # ~25 km cross-track
B38_DRAFT_M  = 220.0      # typical tabular berg draft: ~220 m (90% submerged)
B38_HEIGHT_M =  24.0      # freeboard above waterline: ~24 m

# Tabular berg thickness bounds (m). Antarctic calving fronts are typically
# 150-400 m thick; thickness saturates rather than growing with berg area.
MIN_BERG_THICKNESS_M = 10.0
MAX_BERG_THICKNESS_M = 300.0


# ---------------------------------------------------------------------------
# 1. BERG GEOMETRY HELPERS
# ---------------------------------------------------------------------------

def berg_geometry(length_m: float, width_m: float):
    """
    Derive berg cross-sectional areas and mass from length × width footprint.
    Assumes tabular berg (rectangular prism), RHO_I / RHO_W freeboard ratio.

    Returns
    -------
    draft_m   : keel depth (m)
    height_m  : freeboard (m)
    mass_kg   : berg mass (kg)
    A_keel    : horizontal keel area (m²) — ocean drag acts on this
    A_side_w  : vertical side area in water (m²) — ocean side drag
    A_side_a  : vertical side area in air (m²)   — wind drag
    """
    # FIX: the original heuristic set thickness = sqrt(length*width), which
    # for B38 (40 km x 25 km) gives a 31.6 KM thick berg - two orders of
    # magnitude too deep, and flatly contradicted by this module's own
    # B38_DRAFT_M = 220 m / B38_HEIGHT_M = 24 m constants (which were never
    # used). The inflated mass made the Coriolis term swamp the drag balance,
    # so large bergs barely drifted at all.
    #
    # Tabular berg thickness scales with horizontal size only up to a point and
    # then saturates: calving fronts are a few hundred metres thick regardless
    # of how wide the berg is. Cap it accordingly, then apply isostasy for the
    # draft/freeboard split.
    thickness_m = float(np.clip(0.5 * (length_m * width_m) ** 0.5,
                                MIN_BERG_THICKNESS_M, MAX_BERG_THICKNESS_M))
    draft_m     = (RHO_I / RHO_W) * thickness_m
    height_m    = thickness_m - draft_m

    mass_kg  = RHO_I * length_m * width_m * thickness_m
    A_keel   = length_m * width_m          # bottom face
    A_side_w = (length_m + width_m) * draft_m   # submerged sides
    A_side_a = (length_m + width_m) * height_m  # above-water sides

    return draft_m, height_m, mass_kg, A_keel, A_side_w, A_side_a


# ---------------------------------------------------------------------------
# 2. LICHEY & HELLMER (2001) SEA-ICE DRAG REGIME
# ---------------------------------------------------------------------------

def sea_ice_drag_coefficient(sic: float) -> float:
    """
    Return effective ocean drag scale factor for sea-ice drag on iceberg,
    following Lichey & Hellmer (2001) Table 2 regime classification.

    sic : local sea-ice concentration, 0.0–1.0

    The paper identifies three regimes based on SIC:
      - Open water  (SIC < 0.15) : no sea-ice drag, C_SI = 0
      - Partial ice (0.15–0.80)  : form drag from floe collisions,
                                    C_SI scales linearly with SIC
      - Dense pack  (SIC > 0.80) : berg partly locked; effective C_SI peaks
                                    then berg motion suppressed
    Returns a multiplier applied to the ocean water drag coefficient C_W.
    (Lichey & Hellmer quote C_SI values 0.0–0.15 for their Weddell bergs;
    we express it as an additive term on top of C_W.)
    """
    if sic < 0.15:
        return 0.0
    elif sic <= 0.80:
        # Linear ramp: 0 at SIC=0.15 → 0.15 at SIC=0.80
        return 0.15 * (sic - 0.15) / (0.80 - 0.15)
    else:
        # Dense pack: C_SI peaks at 0.15, then berg motion strongly suppressed
        # We cap at 0.15 and let the momentum balance handle reduced motion
        return 0.15


# ---------------------------------------------------------------------------
# 3. CORIOLIS PARAMETER
# ---------------------------------------------------------------------------

def coriolis(lat_deg: float) -> float:
    """Coriolis parameter f (rad/s) at given latitude."""
    return 2.0 * OMEGA * math.sin(math.radians(lat_deg))


# ---------------------------------------------------------------------------
# 4. WAGNER et al. (2017) ANALYTICAL DRIFT SOLUTION
#
# Wagner et al. derive a closed-form solution to the Bigg momentum balance
# by treating the forcing (wind + current) as slowly varying over one time step.
# This avoids full ODE integration, making it tractable for a 6–12h forecast
# on a 25 km grid.
#
# The momentum balance (Bigg eq. 1, simplified, ocean current = 0 here):
#
#   (1 + C_AM) * M * dV/dt = F_wind + F_ocean + F_coriolis + F_si
#
# where V = (u_berg, v_berg) is the berg velocity vector.
#
# Wagner et al. (2017) eq. (7–9) give the steady-state (equilibrium) drift
# velocity V_eq as a function of wind and ocean velocity. For short time steps,
# we use V_eq as the target and step toward it with a relaxation time τ.
#
# Relaxation time τ (Wagner eq. 10):
#   τ = (1 + C_AM) * M / (k_w * A_keel)
# where k_w = 0.5 * RHO_W * C_W (effective ocean drag per unit area).
# ---------------------------------------------------------------------------

def wagner_equilibrium_velocity(
    wind_u: float, wind_v: float,
    ocean_u: float, ocean_v: float,
    lat_deg: float,
    mass_kg: float,
    A_keel: float, A_side_w: float, A_side_a: float,
    sic: float,
    n_iter: int = 40,
    tol: float = 1e-7,
) -> tuple:
    """
    Compute Wagner et al. (2017) steady-state (equilibrium) berg velocity.

    The equilibrium solution balances wind drag, ocean drag (+ sea-ice drag),
    and Coriolis. Derivation: set dV/dt = 0 in the momentum balance, solve
    the resulting 2×2 linear system for (u_eq, v_eq).

    Parameters
    ----------
    wind_u, wind_v   : 10m wind components (m/s), ERA5
    ocean_u, ocean_v : surface ocean current (m/s) — zero if CMEMS unavailable
    lat_deg          : berg latitude
    mass_kg          : berg mass (kg)
    A_keel           : keel plan area (m²)
    A_side_w         : submerged side area (m²)
    A_side_a         : freeboard side area (m²)
    sic              : local SIC, 0–1

    Returns
    -------
    u_eq, v_eq : equilibrium velocity (m/s east, north)
    tau        : relaxation timescale (s)
    """
    f = coriolis(lat_deg)

    # Effective drag coefficients
    c_si_add = sea_ice_drag_coefficient(sic)
    C_w_eff  = C_W + c_si_add      # ocean + sea-ice combined

    M_eff = (1.0 + C_AM) * mass_kg

    # -----------------------------------------------------------------------
    # TWO FIXES relative to the original implementation, which together made
    # bergs drift at ~0.04% of wind speed instead of the well-established ~2%:
    #
    # 1. WRONG AREA. Ocean drag was applied to A_keel, the HORIZONTAL bottom
    #    face (length x width, 12000 m2 for a medium berg), while wind drag
    #    used the vertical freeboard area. Horizontal drift is resisted by the
    #    SUBMERGED VERTICAL cross-section, A_side_w - which the function
    #    already received but never used. Using the plan area overstated water
    #    drag by ~3 orders of magnitude. (Keel skin friction is neglected: it
    #    is small next to side form drag.)
    #
    # 2. WRONG DRAG LAW. Drag is quadratic, F = 0.5*rho*C*A*|dV|*dV, and the
    #    original linearised it with constant coefficients. That is what
    #    destroys the classic result: balancing the two quadratic drags gives
    #        |V| / |W| = sqrt( rho_a*C_a*A_a / (rho_w*C_w*A_w) ) ~= 2%,
    #    and the square root cannot survive linearisation with fixed k's.
    #
    # We keep Wagner's analytical 2x2 Coriolis solve but recompute the drag
    # coefficients from the current relative speeds each iteration, so the
    # converged answer satisfies the quadratic balance.
    # -----------------------------------------------------------------------
    # Solve the quadratic balance by DAMPED fixed-point iteration, seeded with
    # the exact Coriolis-free solution. An undamped iteration oscillates
    # divergently (the drag coefficients depend on the very speed being
    # solved for), alternating between ~29% and ~0.06% of wind speed, so the
    # answer would depend only on where the loop happened to stop.
    #
    # Coriolis-free closed form: balancing the two quadratic drags along the
    # wind direction with gamma = sqrt(rho_a*C_a*A_a / (rho_w*C_w*A_w)) gives
    #     |V| = gamma / (1 + gamma) * |W|,
    # i.e. the classic ~2%-of-wind iceberg drift rule.
    gamma = math.sqrt((RHO_A * C_A * A_side_a)
                      / max(RHO_W * C_w_eff * A_side_w, 1e-12))
    frac = gamma / (1.0 + gamma)
    u_eq = ocean_u + frac * (wind_u - ocean_u)
    v_eq = ocean_v + frac * (wind_v - ocean_v)

    k_w = 0.5 * RHO_W * C_w_eff * A_side_w * max(
        math.hypot(u_eq - ocean_u, v_eq - ocean_v), 1e-3)

    damping = 0.5
    for _ in range(n_iter):
        d_air   = max(math.hypot(wind_u - u_eq, wind_v - v_eq), 1e-3)
        d_water = max(math.hypot(u_eq - ocean_u, v_eq - ocean_v), 1e-3)

        k_a = 0.5 * RHO_A * C_A     * A_side_a * d_air
        k_w = 0.5 * RHO_W * C_w_eff * A_side_w * d_water

        k_tot = k_w + k_a
        fM    = f * M_eff

        rhs_u = k_w * ocean_u + k_a * wind_u
        rhs_v = k_w * ocean_v + k_a * wind_v

        det = k_tot ** 2 + fM ** 2
        if abs(det) < 1e-12:
            return 0.0, 0.0, M_eff / max(k_w, 1e-9)

        u_sol = ( k_tot * rhs_u + fM * rhs_v) / det
        v_sol = (-fM    * rhs_u + k_tot * rhs_v) / det

        u_new = (1.0 - damping) * u_eq + damping * u_sol
        v_new = (1.0 - damping) * v_eq + damping * v_sol

        converged = math.hypot(u_new - u_eq, v_new - v_eq) < tol
        u_eq, v_eq = u_new, v_new
        if converged:
            break

    # Relaxation timescale (Wagner eq. 10), using the converged ocean drag
    tau = M_eff / max(k_w, 1e-9)

    return u_eq, v_eq, tau


# ---------------------------------------------------------------------------
# 5. PHYSICS VALIDATION GATE
# (Applied after each step — rejects physically impossible states)
# ---------------------------------------------------------------------------

MAX_BERG_SPEED_MS = 0.5   # tabular berg hard speed cap (m/s) — ~1 knot
SIC_LOCK_THRESHOLD = 0.90 # SIC ≥ 90%: berg effectively locked in pack ice

def apply_validation_gate(
    u_new: float, v_new: float,
    sic: float,
    lat_new: float, lon_new: float,
    lat_prev: float = None, lon_prev: float = None,
) -> tuple:
    """
    Physics validation gate (Handover Section 2).
    Returns corrected (u, v, lat, lon) and a flag string.

    Gate 1: SIC ≥ 90% → berg locked, velocity zeroed.
    Gate 2: Speed > 0.5 m/s → cap to 0.5 m/s, preserve direction.
    Gate 3: Bathymetry / grounding — NOT implemented here (requires GEBCO grid).
            Claude Code should add this once GEBCO is available.
    """
    flag = "ok"

    # Gate 1: SIC lock — a berg held fast in >=90% pack does not drift, so
    # freeze POSITION as well as velocity. (The original returned the already
    # advanced lat/lon, letting a "locked" berg keep moving.)
    if sic >= SIC_LOCK_THRESHOLD:
        if lat_prev is None:
            lat_prev, lon_prev = lat_new, lon_new
        return 0.0, 0.0, lat_prev, lon_prev, "sic_locked"

    # Gate 2: Speed cap
    speed = math.sqrt(u_new**2 + v_new**2)
    if speed > MAX_BERG_SPEED_MS:
        scale = MAX_BERG_SPEED_MS / speed
        u_new *= scale
        v_new *= scale
        flag = "speed_capped"

    return u_new, v_new, lat_new, lon_new, flag


# ---------------------------------------------------------------------------
# 6. COORDINATE UPDATE
# (Flat-Earth approximation valid over short 6–12h steps in high latitudes)
# ---------------------------------------------------------------------------

def update_position(lat_deg: float, lon_deg: float,
                    u_ms: float, v_ms: float,
                    dt_s: float) -> tuple:
    """
    Move berg by (u, v) over dt seconds. Returns new (lat, lon) in degrees.
    Uses spherical Earth for accuracy at high southern latitudes.
    """
    d_lat = (v_ms * dt_s) / R_EARTH
    d_lon = (u_ms * dt_s) / (R_EARTH * math.cos(math.radians(lat_deg)))
    new_lat = lat_deg + math.degrees(d_lat)
    new_lon = lon_deg + math.degrees(d_lon)
    # Wrap longitude to -180/180
    new_lon = (new_lon + 180.0) % 360.0 - 180.0
    return new_lat, new_lon


# ---------------------------------------------------------------------------
# 7. MAIN STEP FUNCTION — called once per time step by the ML pipeline
# ---------------------------------------------------------------------------

def physics_step(
    lat: float, lon: float,
    vel_u: float, vel_v: float,
    wind_u: float, wind_v: float,
    sic: float,
    dt_s: float = 21600.0,          # default: 6-hour step (6 × 3600)
    ocean_u: float = 0.0,           # zero until CMEMS available (Assumption A1)
    ocean_v: float = 0.0,
    length_m: float = B38_LENGTH_M,
    width_m:  float = B38_WIDTH_M,
) -> dict:
    """
    Advance iceberg state by one time step using Wagner et al. (2017).

    Parameters
    ----------
    lat, lon         : current position (degrees)
    vel_u, vel_v     : current berg velocity (m/s, east/north)
    wind_u, wind_v   : ERA5 10m wind at berg location (m/s)
    sic              : NSIDC SIC at berg location (0.0–1.0)
    dt_s             : time step in seconds (default 6h)
    ocean_u, ocean_v : surface current (m/s) — zero if CMEMS unavailable
    length_m, width_m: berg footprint dimensions (m)

    Returns
    -------
    dict with keys:
      lat_new, lon_new       : predicted position (degrees)
      vel_u_new, vel_v_new   : predicted velocity (m/s)
      u_eq, v_eq             : Wagner equilibrium velocity (m/s)
      tau_s                  : relaxation timescale (s)
      validation_flag        : 'ok' | 'sic_locked' | 'speed_capped'
    """
    # Geometry
    draft, height, mass, A_keel, A_side_w, A_side_a = berg_geometry(length_m, width_m)

    # Wagner equilibrium velocity
    u_eq, v_eq, tau = wagner_equilibrium_velocity(
        wind_u, wind_v, ocean_u, ocean_v,
        lat, mass, A_keel, A_side_w, A_side_a, sic
    )

    # Relax current velocity toward equilibrium over dt
    # V(t+dt) = V_eq + (V(t) - V_eq) * exp(-dt/tau)
    decay   = math.exp(-dt_s / max(tau, 1.0))
    u_new   = u_eq + (vel_u - u_eq) * decay
    v_new   = v_eq + (vel_v - v_eq) * decay

    # Update position
    lat_new, lon_new = update_position(lat, lon, u_new, v_new, dt_s)

    # Physics validation gate
    u_new, v_new, lat_new, lon_new, flag = apply_validation_gate(
        u_new, v_new, sic, lat_new, lon_new, lat_prev=lat, lon_prev=lon
    )

    return {
        "lat_new":          lat_new,
        "lon_new":          lon_new,
        "vel_u_new":        u_new,
        "vel_v_new":        v_new,
        "u_eq":             u_eq,
        "v_eq":             v_eq,
        "tau_s":            tau,
        "validation_flag":  flag,
    }


# ---------------------------------------------------------------------------
# 8. BATCH RUNNER — generate physics baseline predictions for all B38 obs
#    This is what the ML residual layer trains on:
#    residual = observed_position − physics_predicted_position
# ---------------------------------------------------------------------------

def run_baseline_on_csv(
    iceberg_csv_path: str,
    wind_npz_path: str,
    sic_npz_path: str,
    iceberg_id: str = "B38",
    dt_s: float = 21600.0,
    output_csv_path: str = "iceberg_physics_residuals.csv",
    shipping_season_only: bool = True,
    max_gap_hours: float = 72.0,
    dedupe_static_fixes: bool = True,
):
    """
    Run the physics baseline across consecutive observation pairs for one
    iceberg, sampling ERA5 wind and NSIDC SIC at each berg position/time.

    Saves a CSV with columns:
      DATE, LAT_obs, LON_obs, LAT_phys, LON_phys,
      RESID_LAT, RESID_LON,   <- what the LSTM trains on
      VEL_U, VEL_V, WIND_U, WIND_V, SIC, TAU_S, FLAG
    plus ICEBERG_ID, SEASON, GAP_H, N_SUBSTEPS, DIST_ERR_KM for diagnostics.

    Four corrections to the original implementation, each of which produced
    silently wrong numbers rather than an error:

    1. ID matching is case-insensitive. The CSV stores 'b38', not 'B38', so
       the original exact match selected ZERO rows.
    2. Dates are 'YYYY-MM-DD', not 'YYYYMMDD'. The original strptime('%Y%m%d')
       raised, was swallowed by a bare `except`, and pinned EVERY wind lookup
       to time index 0; the SIC lookup's int() cast would then crash.
    3. Wind and SIC are sampled through grid_utils, which handles the
       curvilinear SIC mesh and the non-monotonic wind longitude. The original
       fed both to RegularGridInterpolator as if they were regular ascending
       lat/lon axes, which they are not.
    4. Observations are ~24 h apart (median), not 6 h. The original advanced
       the physics by a single 6 h step and differenced against an observation
       a full day later, so the "residual" was mostly just un-integrated
       drift. We now integrate in dt_s sub-steps across the ACTUAL observation
       gap, re-sampling wind at every sub-step.

    Only Nov-Mar observations are used (the stated training window), and pairs
    straddling a season break or separated by more than `max_gap_hours` are
    skipped rather than integrated across a data void.
    """
    import pandas as pd
    import datetime as dt
    import grid_utils as gu

    print(f"[1/4] Loading iceberg observations for {iceberg_id} ...")
    df = pd.read_csv(iceberg_csv_path)
    want = str(iceberg_id).strip().lower()
    df = df[df["ICEBERG_ID"].astype(str).str.strip().str.lower() == want].copy()
    if df.empty:
        raise ValueError(
            f"No observations for iceberg_id={iceberg_id!r}. "
            f"Available: {sorted(pd.read_csv(iceberg_csv_path)['ICEBERG_ID'].unique())}"
        )

    df["_dt"] = df["DATE"].map(gu.parse_date)
    n_all = len(df)
    if shipping_season_only:
        df = df[df["_dt"].map(lambda d: d.month in gu.SHIPPING_MONTHS)].copy()
    df = df.sort_values("_dt").reset_index(drop=True)
    n_window = len(df)

    if dedupe_static_fixes and len(df) > 1:
        # BYU/NIC CARRIES FORWARD the last known fix when a berg is not
        # re-imaged: 66% of consecutive B38 rows repeat the previous position
        # byte-for-byte, and 371 records contain only 40 distinct latitudes.
        # Differencing those repeats yields a "residual" that is an artefact of
        # the reporting cadence (zero displacement, then one large catch-up
        # jump) rather than a physical drift error - it even has NEGATIVE lag-1
        # autocorrelation. Collapse each run of identical positions to its
        # first report so pairs span genuinely distinct fixes.
        moved = (df["LAT"].diff().abs() + df["LON"].diff().abs()) > 1e-9
        moved.iloc[0] = True
        df = df[moved].reset_index(drop=True)

    print(f"      {len(df)} observations in window "
          f"(of {n_all} total, {n_window} in season"
          + (f", {n_window - len(df)} repeated fixes dropped)" if dedupe_static_fixes
             else ")"))

    print("[2/4] Loading ERA5 wind ...")
    wind = gu.WindSampler(wind_npz_path)

    print("[3/4] Loading SIC grid ...")
    s = np.load(sic_npz_path)
    sic_dates = np.asarray(s["dates"])
    sic_all = s["concentration"]
    locator = gu.GridLocator(s["lat"], s["lon"], ~np.isnan(sic_all[0]))

    # Cache the date-index lookup: one entry per distinct observation date
    sic_secs = np.array([gu.parse_date(str(x)).timestamp() for x in sic_dates])

    def sic_index(when):
        return int(np.argmin(np.abs(sic_secs - when.timestamp())))

    # Berg footprint: BYU size_1/size_2 are nautical miles and are 0 when not
    # reported, so fall back to the representative B38 dimensions.
    def berg_dims(row):
        s1 = float(row.get("size_1", 0) or 0.0)
        s2 = float(row.get("size_2", 0) or 0.0)
        if s1 > 0 and s2 > 0:
            return s1 * 1852.0, s2 * 1852.0
        return B38_LENGTH_M, B38_WIDTH_M

    print("[4/4] Running physics baseline step-by-step ...")
    records = []
    vel_u = vel_v = 0.0
    prev_end = None   # datetime of the previous pair's end, for velocity carry

    for i in range(len(df) - 1):
        row_now, row_next = df.iloc[i], df.iloc[i + 1]
        t0, t1 = row_now["_dt"], row_next["_dt"]
        gap_h = (t1 - t0).total_seconds() / 3600.0

        # Skip season breaks and long data voids
        if gap_h <= 0 or gap_h > max_gap_hours:
            continue
        if gu.season_label(t0) != gu.season_label(t1):
            continue

        lat0, lon0 = float(row_now["LAT"]), float(row_now["LON"])
        lat_obs, lon_obs = float(row_next["LAT"]), float(row_next["LON"])
        length_m, width_m = berg_dims(row_now)

        # Seed velocity: carry over if this pair continues the previous one,
        # otherwise estimate from the PREVIOUS interval's observed displacement.
        #
        # It must never be seeded from this interval's own end point: that is
        # the target being predicted. Doing so leaks the answer into the
        # initial condition, and because a large berg's relaxation timescale
        # is enormous (tau ~ 5000 h for B38) the berg simply coasts at the
        # seeded velocity, so the "physics baseline" was largely replaying the
        # observation it was meant to forecast.
        if prev_end is None or prev_end != t0:
            if i > 0:
                row_prev = df.iloc[i - 1]
                gap_prev = (t0 - row_prev["_dt"]).total_seconds()
                if gap_prev > 0:
                    lat_p0, lon_p0 = float(row_prev["LAT"]), float(row_prev["LON"])
                    dist_e = gu.haversine_km(lat_p0, lon_p0, lat_p0, lon0) * 1000.0
                    dist_n = gu.haversine_km(lat_p0, lon_p0, lat0, lon_p0) * 1000.0
                    dlon = ((lon0 - lon_p0 + 180.0) % 360.0) - 180.0
                    vel_u = math.copysign(dist_e, dlon) / gap_prev
                    vel_v = math.copysign(dist_n, lat0 - lat_p0) / gap_prev
                else:
                    vel_u = vel_v = 0.0
            else:
                vel_u = vel_v = 0.0

        # Wind/SIC recorded at the START of the interval (LSTM input features)
        wind_u0, wind_v0 = wind.sample(lat0, lon0, t0)
        sic0 = locator.sample(sic_all[sic_index(t0)], lat0, lon0, default=0.0) / 100.0

        # --- Integrate physics across the real gap in dt_s sub-steps --------
        n_sub = max(1, int(round(gap_h * 3600.0 / dt_s)))
        sub_dt = gap_h * 3600.0 / n_sub
        lat_p, lon_p = lat0, lon0
        flag = "ok"
        tau_s = float("nan")

        for k in range(n_sub):
            t_k = t0 + dt.timedelta(seconds=k * sub_dt)
            wu, wv = wind.sample(lat_p, lon_p, t_k)
            sic_k = locator.sample(sic_all[sic_index(t_k)], lat_p, lon_p,
                                   default=0.0) / 100.0
            res = physics_step(
                lat=lat_p, lon=lon_p,
                vel_u=vel_u, vel_v=vel_v,
                wind_u=wu, wind_v=wv,
                sic=sic_k,
                dt_s=sub_dt,
                length_m=length_m, width_m=width_m,
            )
            lat_p, lon_p = res["lat_new"], res["lon_new"]
            vel_u, vel_v = res["vel_u_new"], res["vel_v_new"]
            tau_s = res["tau_s"]
            if res["validation_flag"] != "ok":
                flag = res["validation_flag"]

        prev_end = t1

        resid_lat = lat_obs - lat_p
        resid_lon = ((lon_obs - lon_p + 180.0) % 360.0) - 180.0

        records.append({
            "DATE":       t0.strftime("%Y-%m-%d"),
            "ICEBERG_ID": want,
            "SEASON":     gu.season_label(t0),
            "LAT_obs":    lat_obs,
            "LON_obs":    lon_obs,
            "LAT_phys":   lat_p,
            "LON_phys":   lon_p,
            "RESID_LAT":  resid_lat,
            "RESID_LON":  resid_lon,
            "VEL_U":      vel_u,
            "VEL_V":      vel_v,
            "WIND_U":     wind_u0,
            "WIND_V":     wind_v0,
            "SIC":        sic0,
            "TAU_S":      tau_s,
            "FLAG":       flag,
            "GAP_H":      gap_h,
            "N_SUBSTEPS": n_sub,
            "DIST_ERR_KM": float(gu.haversine_km(lat_obs, lon_obs, lat_p, lon_p)),
        })

    out_df = pd.DataFrame(records)
    if out_df.empty:
        raise ValueError("No usable observation pairs — check the date window.")

    out_df.to_csv(output_csv_path, index=False)
    print(f"\n    Saved {len(out_df)} residual records to {output_csv_path}")
    print(f"    Mean |RESID_LAT|: {out_df['RESID_LAT'].abs().mean():.4f} deg")
    print(f"    Mean |RESID_LON|: {out_df['RESID_LON'].abs().mean():.4f} deg")
    print(f"    Mean position error: {out_df['DIST_ERR_KM'].mean():.2f} km "
          f"(median {out_df['DIST_ERR_KM'].median():.2f} km)")
    return out_df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run iceberg physics baseline (Bigg/Wagner/Lichey) and output residuals for ML training."
    )
    parser.add_argument("--iceberg_csv",  default="ross_sea_icebergs_2021_2025.csv")
    parser.add_argument("--wind_npz",     default="ross_sea_era5_wind.npz")
    parser.add_argument("--sic_npz",      default="ross_sea_concentration.npz")
    parser.add_argument("--iceberg_id",   default="B38")
    parser.add_argument("--dt_hours",     type=float, default=6.0,
                        help="Physics time step in hours (default: 6)")
    parser.add_argument("--output",       default="iceberg_physics_residuals.csv")
    parser.add_argument("--all_months", action="store_true",
                        help="Use all months instead of the Nov-Mar window.")
    parser.add_argument("--max_gap_hours", type=float, default=336.0,
                        help="Skip observation pairs separated by more than this "
                             "(default 14 days; deduped fixes are days apart).")
    parser.add_argument("--keep_repeat_fixes", action="store_true",
                        help="Keep BYU carried-forward duplicate positions "
                             "(NOT recommended - see run_baseline_on_csv docs).")
    args = parser.parse_args()

    run_baseline_on_csv(
        iceberg_csv_path=args.iceberg_csv,
        wind_npz_path=args.wind_npz,
        sic_npz_path=args.sic_npz,
        iceberg_id=args.iceberg_id,
        dt_s=args.dt_hours * 3600.0,
        output_csv_path=args.output,
        shipping_season_only=not args.all_months,
        max_gap_hours=args.max_gap_hours,
        dedupe_static_fixes=not args.keep_repeat_fixes,
    )
