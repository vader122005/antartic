"""
seaice_physics_baseline.py
===========================
Antarctic Nav System — Sea-Ice Physics Baseline
Problem Statement: SIH26059 | Team: Daði og Gagnamagnið (SIH050)

Implements the physics-based sea-ice concentration (SIC) forecast as the
baseline component of the residual-learning hybrid architecture. The ML
residual layer (built separately by Claude Code) will learn:
  Δ(observed SIC − physics SIC)
on top of this output, using ERA5 atmospheric forcing as additional features.

PHYSICS MODEL
-------------
Stefan's Law / Freezing Degree Day (FDD) thermodynamic model.

Stefan's Law relates ice thickness h to accumulated freezing degree days:
    h = α_s * sqrt(FDD)

where α_s is the Stefan coefficient (empirical, accounts for snow insulation).

SIC is then derived from h using a concentration–thickness relationship
calibrated to Ross Sea conditions (linear proxy over the training window).

This is a 1-D thermodynamic model — it captures growth/melt from
surface heat flux but not dynamic processes (wind-driven advection, ridging,
leads). Those dynamics are exactly what the ML residual layer learns to correct.

REFERENCES
----------
- Stefan, J. (1891). "Über die Theorie der Eisbildung, insbesondere über
  die Eisbildung im Polarmeere." Annalen der Physik, 278(2), 269–286.

- Maykut, G.A. (1986). "The surface heat and mass balance." In: Untersteiner, N.
  (ed.) The Geophysics of Sea Ice. NATO ASI Series B, vol. 146.
  Springer, Boston. (Standard FDD / Stefan coefficient source.)

- Andersson, T.R. et al. (2021). "Seasonal Arctic sea ice forecasting with
  probabilistic deep learning." Nature Communications, 12, 5124.
  (Context for hybrid physics + ML for sea ice — validates our architecture.)

- Lu, P. et al. (2020). "Thermodynamic and dynamic modelling of sea ice."
  Cold Regions Science and Technology, 176, 103089.
  (FDD model parameters for Antarctic conditions.)

INPUTS (per grid cell, per day)
--------------------------------
- T_air     : 2m air temperature (°C) — from ERA5 (interpolated to NSIDC grid)
  NOTE: ERA5 wind .npz has 10m wind only; 2m temperature is NOT downloaded.
  → Assumption A2: T_air approximated from ERA5 skin temperature or a fixed
    seasonal lapse. Claude Code should add ERA5 t2m download if time permits.
  → For hackathon: we use a simple seasonal air temp climatology for the
    Ross Sea (documented in SEASONAL_TEMP_C below) as a fallback.
- prior_sic : SIC from previous day (0–100 %) — from NSIDC .npz
- doy       : day of year (1–365) — drives seasonal melt/growth switch

OUTPUT (per grid cell, per day)
--------------------------------
- sic_phys  : physics-predicted SIC (0–100 %) one day ahead
This is subtracted from observed SIC to get the residual for the LSTM.

ASSUMPTIONS & LIMITATIONS
--------------------------
A1. Model is purely thermodynamic — no ice dynamics (advection, ridging, export).
    Dynamic processes are the ML residual layer's job. Flag in uncertainty output.
A2. 2m air temperature not separately downloaded — approximated from seasonal
    climatology for Ross Sea. Real ERA5 t2m would improve the baseline.
    Add as a future download if time permits before submission.
A3. Stefan coefficient α_s = 0.33 m/°C^0.5·day^0.5 — standard value for
    Antarctic sea ice with moderate snow cover (Maykut 1986, Lu et al. 2020).
    Snow depth not explicitly modelled.
A4. SIC–thickness relationship is a linear proxy calibrated to the training
    window. Does not account for multi-year ice (not dominant in Ross Sea).
A5. Melt onset triggered purely by T_air > 0°C — no albedo feedback or
    ocean heat flux term (simplification for hackathon scope).
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
# PHYSICAL CONSTANTS & MODEL PARAMETERS
# ---------------------------------------------------------------------------

# Stefan coefficient for Antarctic sea ice (m per sqrt(°C·day))
# Maykut (1986): α_s ≈ 0.33 for snow-covered Antarctic FY ice
STEFAN_ALPHA = 0.33

# FDD accumulation reset: when T_air > 0°C, ice melts instead of grows.
# Melt rate: empirical 0.03 m/°C/day (Lu et al. 2020, Ross Sea calibration)
MELT_RATE_M_PER_DEG_PER_DAY = 0.03

# SIC–thickness proxy (linear, calibrated to NSIDC Ross Sea Nov–Mar stats)
# At maximum thickness H_MAX, SIC = SIC_MAX (consolidated pack)
# At zero thickness, SIC = 0
# Representative values for Ross Sea FY ice:
H_MAX   = 2.0    # m — thick FY ice upper bound for this proxy
SIC_MAX = 97.0   # % — corresponding max SIC (matching CSV top tier)

# Minimum ice thickness to report any SIC (below this = open water)
H_MIN_FOR_SIC = 0.02   # 2 cm — frazil / grease ice threshold

# --- Ocean/solar heat flux melt (FIX: see A5) ------------------------------
# The Nov–Mar Ross Sea retreat is driven mainly by ocean heat flux and
# absorbed shortwave, NOT by T_air (climatological T_air stays < 0 all year).
# Without this term the thermodynamic model can only ever grow ice, which
# pinned 98.5% of cells at SIC_MAX. Basal melt from an ocean heat flux Q:
#     dh/dt = Q / (rho_ice * L_f)
RHO_ICE        = 917.0      # kg/m3
LATENT_FUSION  = 3.34e5     # J/kg
RHO_L          = RHO_ICE * LATENT_FUSION    # J/m3 to melt 1 m of ice

# Thermal conductivity of sea ice (W/m/K) and seawater freezing point (degC).
K_ICE      = 2.03
T_FREEZE_C = -1.8
# Conductive-flux thin-ice cap (m). Q_cond = k*dT/h diverges as h->0; leads and
# thin ice are sub-grid, so cap the effective conducting thickness.
H_COND_MIN = 0.3

# Climatological net ocean+solar heat flux into the ice base (W/m2), Ross Sea.
# Peaks in Dec–Jan (polynya, low albedo, max insolation), near zero in winter.
SEASONAL_OCEAN_FLUX_W = {
    1:  95.0,   # January  — peak melt
    2:  75.0,   # February
    3:  30.0,   # March    — refreeze begins
    4:   8.0,
    5:   3.0,
    6:   2.0,
    7:   2.0,
    8:   2.0,
    9:   4.0,
    10: 20.0,
    11: 55.0,   # November — retreat underway
    12: 90.0,   # December
}

# Day-of-year at the middle of each month (non-leap), for cyclic interpolation
_MONTH_MID_DOY = np.array(
    [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float
)


def _cyclic_monthly_interp(doy: int, table: dict) -> float:
    """
    Interpolate a monthly climatology table at a given day-of-year, wrapping
    correctly across the Dec->Jan boundary.
    """
    vals = np.array([table[m] for m in range(1, 13)], dtype=float)
    # Extend both ends so np.interp handles the wrap
    x = np.concatenate(([_MONTH_MID_DOY[-1] - 365.0], _MONTH_MID_DOY,
                        [_MONTH_MID_DOY[0] + 365.0]))
    y = np.concatenate(([vals[-1]], vals, [vals[0]]))
    return float(np.interp(float(doy), x, y))

# ---------------------------------------------------------------------------
# SEASONAL AIR TEMPERATURE CLIMATOLOGY — Ross Sea
# Used as Assumption A2 fallback when ERA5 t2m is unavailable.
# Source: ERA5 reanalysis climatological mean, Ross Sea box,
# approximate monthly values for the Nov–Mar shipping season.
# Units: °C (negative = freezing, positive = melting)
# ---------------------------------------------------------------------------
# Month: 1=Jan, 2=Feb, ... 12=Dec
SEASONAL_TEMP_C = {
    1:  -3.0,   # January  — midsummer, warmest
    2:  -4.5,   # February — late summer
    3:  -9.0,   # March    — early autumn, refreezing begins
    4: -15.0,   # April    — outside training window, included for FDD carry-over
    5: -20.0,
    6: -22.0,
    7: -23.0,
    8: -22.0,
    9: -19.0,
    10: -14.0,
    11:  -7.0,  # November — spring, ice retreating
    12:  -4.0,  # December — early summer
}


def seasonal_air_temp(doy: int) -> float:
    """
    Return approximate 2m air temperature (°C) for a given day of year
    based on Ross Sea seasonal climatology. Linearly interpolates between
    monthly midpoints.

    doy: day of year (1–365)
    """
    # FIX: the previous fractional-month arithmetic was off by one month
    # (doy=1 / 1 Jan returned February's value). Interpolate between true
    # month-midpoint day-of-years instead, wrapping across Dec->Jan.
    return _cyclic_monthly_interp(doy, SEASONAL_TEMP_C)


def seasonal_ocean_flux(doy: int) -> float:
    """
    Return climatological net ocean+solar heat flux (W/m2) into the ice base
    for a given day of year. Drives the melt term (Assumption A5 relaxed).
    """
    return _cyclic_monthly_interp(doy, SEASONAL_OCEAN_FLUX_W)


# ---------------------------------------------------------------------------
# CORE THERMODYNAMIC FUNCTIONS
# ---------------------------------------------------------------------------

def fdd_to_thickness(fdd: float) -> float:
    """
    Stefan's Law: ice thickness from accumulated freezing degree days.
    h = α_s * sqrt(FDD)

    fdd     : accumulated freezing degree days (°C·days, always positive)
    returns : ice thickness in metres
    """
    if fdd <= 0.0:
        return 0.0
    return STEFAN_ALPHA * math.sqrt(fdd)


def thickness_to_sic(h: float) -> float:
    """
    Convert ice thickness (m) to sea-ice concentration (%).
    Linear proxy: SIC = (h / H_MAX) * SIC_MAX, capped at SIC_MAX.
    Below H_MIN_FOR_SIC → SIC = 0.
    """
    if h < H_MIN_FOR_SIC:
        return 0.0
    return min((h / H_MAX) * SIC_MAX, SIC_MAX)


def sic_to_thickness(sic: float) -> float:
    """Inverse of thickness_to_sic — used to initialise h from observed SIC."""
    if sic <= 0.0:
        return 0.0
    return (sic / SIC_MAX) * H_MAX


# ---------------------------------------------------------------------------
# PER-CELL DAILY STEP
# ---------------------------------------------------------------------------

def physics_step_cell(
    prior_sic: float,
    fdd_accumulated: float,
    doy: int,
    t_air: float = None,
) -> dict:
    """
    Advance one grid cell by one day using the Stefan/FDD model.

    Parameters
    ----------
    prior_sic        : observed SIC yesterday (0–100 %)
    fdd_accumulated  : FDD accumulated so far this freezing season (°C·days)
    doy              : day of year (1–365)
    t_air            : 2m air temperature (°C); if None, use climatology

    Returns
    -------
    dict with keys:
      sic_phys         : physics-predicted SIC (0–100 %)
      h_phys           : physics-predicted ice thickness (m)
      fdd_new          : updated FDD accumulation (°C·days)
      t_air            : air temperature used (°C)
      regime           : 'growth' | 'melt' | 'open_water'
    """
    if t_air is None:
        t_air = seasonal_air_temp(doy)

    # Initialise thickness from prior observed SIC
    h_prior = sic_to_thickness(prior_sic)

    # --- THERMODYNAMIC ENERGY BALANCE ---------------------------------------
    # FIX: the original code recomputed h purely from season-accumulated FDD
    # (h = alpha*sqrt(FDD)), discarding h_prior. That decoupled the forecast
    # from the observed state and pinned 98.5% of cells at SIC_MAX
    # (mean |residual| 76.7 %SIC). It also had no reachable melt branch, since
    # the climatological T_air is below 0 degC year-round.
    #
    # Replaced with the standard two-sided balance between conduction of heat
    # UP through the ice and ocean/solar heat delivered to its base:
    #     dh/dt = [ k_ice * (T_f - T_air) / max(h, h0)  -  Q_ocean ] / (rho*L)
    # Growth where conduction wins, melt where the ocean wins. This reproduces
    # the summer retreat and, crucially, stops ice forming spuriously in warm
    # open water (a pure-Stefan term grew ~4 %SIC/day on every open cell).
    # All constants are textbook; only Q_ocean is climatological (see A2/A5).
    dfdd    = abs(t_air) if t_air < 0.0 else 0.0
    fdd_new = fdd_accumulated + dfdd if t_air < 0.0 else 0.0

    q_cond  = K_ICE * max(T_FREEZE_C - t_air, 0.0) / max(h_prior, H_COND_MIN)
    q_ocean = seasonal_ocean_flux(doy)
    dh      = ((q_cond - q_ocean) / RHO_L) * 86400.0

    if t_air > 0.0:
        dh -= MELT_RATE_M_PER_DEG_PER_DAY * t_air

    h_phys = max(h_prior + dh, 0.0)

    if h_phys > h_prior + 1e-9:
        regime = "growth"
    elif h_prior > 0.0:
        regime = "melt"
    else:
        regime = "open_water"

    sic_phys = thickness_to_sic(h_phys)

    return {
        "sic_phys":   sic_phys,
        "h_phys":     h_phys,
        "fdd_new":    fdd_new,
        "t_air":      t_air,
        "regime":     regime,
    }


# ---------------------------------------------------------------------------
# GRID RUNNER — one day ahead prediction across the full (133, 147) grid
# ---------------------------------------------------------------------------

def run_grid_one_step(
    sic_today: np.ndarray,
    fdd_grid: np.ndarray,
    doy: int,
    t_air_grid: np.ndarray = None,
) -> dict:
    """
    Run one day of physics prediction across the full Ross Sea grid.

    Parameters
    ----------
    sic_today   : (133, 147) float32, observed SIC today (0–100 %), NaN = land
    fdd_grid    : (133, 147) float32, accumulated FDD per cell (°C·days)
    doy         : day of year for today
    t_air_grid  : (133, 147) float32, 2m air temp (°C); if None → climatology

    Returns
    -------
    dict with keys:
      sic_phys   : (133, 147) float32 — physics SIC prediction for tomorrow
      h_phys     : (133, 147) float32 — ice thickness (m)
      fdd_grid   : (133, 147) float32 — updated FDD accumulation
      regime     : (133, 147) object  — 'growth'/'melt'/'open_water'/NaN
    """
    # Vectorised implementation of physics_step_cell over the whole grid.
    # (The original per-cell Python loop ran 19551 cells x 604 days.)
    rows, cols = sic_today.shape
    valid = ~np.isnan(sic_today)

    if t_air_grid is None:
        t_air = np.full((rows, cols), seasonal_air_temp(doy), dtype=np.float64)
    else:
        t_air = np.asarray(t_air_grid, dtype=np.float64)

    prior = np.where(valid, np.nan_to_num(sic_today, nan=0.0), 0.0).astype(np.float64)

    # sic -> thickness (inverse linear proxy)
    h_prior = np.clip(prior, 0.0, None) / SIC_MAX * H_MAX

    # Thermodynamic energy balance (see physics_step_cell for rationale)
    dfdd    = np.where(t_air < 0.0, np.abs(t_air), 0.0)
    fdd_new = np.where(t_air < 0.0, fdd_grid + dfdd, 0.0).astype(np.float32)

    q_cond  = K_ICE * np.maximum(T_FREEZE_C - t_air, 0.0) / np.maximum(h_prior, H_COND_MIN)
    dh      = ((q_cond - seasonal_ocean_flux(doy)) / RHO_L) * 86400.0
    dh      = dh - np.where(t_air > 0.0, MELT_RATE_M_PER_DEG_PER_DAY * t_air, 0.0)

    h_arr = np.maximum(h_prior + dh, 0.0)

    sic_arr = np.minimum(h_arr / H_MAX * SIC_MAX, SIC_MAX)
    sic_arr = np.where(h_arr < H_MIN_FOR_SIC, 0.0, sic_arr)

    sic_phys = np.where(valid, sic_arr, np.nan).astype(np.float32)
    h_phys   = np.where(valid, h_arr,   np.nan).astype(np.float32)
    fdd_new  = np.where(valid, fdd_new, fdd_grid).astype(np.float32)

    regime = np.full((rows, cols), "", dtype=object)
    regime[valid & (h_arr > h_prior + 1e-9)] = "growth"
    regime[valid & (h_arr <= h_prior + 1e-9) & (h_prior > 0.0)] = "melt"
    regime[valid & (h_arr <= h_prior + 1e-9) & (h_prior <= 0.0)] = "open_water"

    return {
        "sic_phys":  sic_phys,
        "h_phys":    h_phys,
        "fdd_grid":  fdd_new,
        "regime":    regime,
    }


# ---------------------------------------------------------------------------
# BATCH RUNNER — generate physics baseline + residuals across training window
# This is what the ML residual layer (LSTM/1D-CNN) trains on.
# Residual = observed_SIC(t+1) − physics_SIC(t+1)
# ---------------------------------------------------------------------------

def run_baseline_on_npz(
    sic_npz_path: str,
    output_npz_path: str = "seaice_physics_residuals.npz",
):
    """
    Run the Stefan/FDD physics baseline across the full training window
    (all 605 dates) and save physics predictions + residuals.

    Output arrays in the saved .npz:
      dates_out   : (604,) YYYYMMDD strings — one per consecutive pair
      sic_obs     : (604, 133, 147) float32 — observed SIC at t+1
      sic_phys    : (604, 133, 147) float32 — physics SIC at t+1
      residual    : (604, 133, 147) float32 — observed − physics (LSTM target)
      h_phys      : (604, 133, 147) float32 — ice thickness (m)
      fdd_final   : (133, 147)       float32 — FDD state at end of window
    """
    print(f"[1/3] Loading SIC grid from {sic_npz_path} ...")
    d = np.load(sic_npz_path)
    sic_all = d["concentration"]   # (605, 133, 147)
    dates   = d["dates"]           # (605,) YYYYMMDD strings
    n_dates = len(dates)
    rows, cols = sic_all.shape[1], sic_all.shape[2]

    print(f"      {n_dates} dates, grid ({rows}, {cols})")

    print("[2/3] Running Stefan/FDD baseline day-by-day ...")

    # Initialise FDD grid to zero — season starts at the first date
    fdd_grid = np.zeros((rows, cols), dtype=np.float32)

    out_dates  = []
    out_obs    = []
    out_phys   = []
    out_resid  = []
    out_h      = []

    for t in range(n_dates - 1):
        date_str = dates[t]
        # Day of year from YYYYMMDD
        from datetime import datetime
        dt_obj = datetime.strptime(date_str, "%Y%m%d")
        doy    = dt_obj.timetuple().tm_yday

        sic_today    = sic_all[t]       # (133, 147)
        sic_tomorrow = sic_all[t + 1]   # observed truth for residual

        result = run_grid_one_step(
            sic_today=sic_today,
            fdd_grid=fdd_grid,
            doy=doy,
            t_air_grid=None,   # uses climatology (Assumption A2)
        )

        fdd_grid = result["fdd_grid"].astype(np.float32)

        # Residual only where both obs and phys are valid
        resid = np.where(
            np.isnan(sic_tomorrow) | np.isnan(result["sic_phys"]),
            np.nan,
            sic_tomorrow - result["sic_phys"]
        ).astype(np.float32)

        out_dates.append(dates[t + 1])
        out_obs.append(sic_tomorrow)
        out_phys.append(result["sic_phys"].astype(np.float32))
        out_resid.append(resid)
        out_h.append(result["h_phys"].astype(np.float32))

        if t % 50 == 0:
            valid = ~np.isnan(resid)
            mean_resid = np.abs(resid[valid]).mean() if valid.any() else float("nan")
            print(f"    t={t:3d}  date={date_str}  mean|resid|={mean_resid:.2f}%")

    print(f"[3/3] Saving to {output_npz_path} ...")
    np.savez_compressed(
        output_npz_path,
        dates_out  = np.array(out_dates),
        sic_obs    = np.stack(out_obs,   axis=0),
        sic_phys   = np.stack(out_phys,  axis=0),
        residual   = np.stack(out_resid, axis=0),
        h_phys     = np.stack(out_h,     axis=0),
        fdd_final  = fdd_grid,
        metadata   = np.array([
            "model=conduction_ocean_energy_balance",
            f"k_ice={K_ICE}",
            f"t_freeze_c={T_FREEZE_C}",
            f"h_cond_min={H_COND_MIN}",
            f"melt_rate={MELT_RATE_M_PER_DEG_PER_DAY}",
            "q_ocean_source=seasonal_climatology_assumption_A5",
            "t_air_source=seasonal_climatology_assumption_A2",
            "residual=observed_minus_physics",
            "LSTM_target=residual",
        ])
    )

    # Summary stats
    all_resid = np.concatenate([r[~np.isnan(r)].ravel() for r in out_resid])
    print(f"\n    Done. {len(out_dates)} day-pairs processed.")
    print(f"    Mean |residual| across grid: {np.abs(all_resid).mean():.2f} %SIC")
    print(f"    Std  |residual|:             {all_resid.std():.2f} %SIC")
    print(f"\n    Output: {output_npz_path}")
    print("""
    Load in your LSTM trainer with:
        d = np.load('seaice_physics_residuals.npz')
        residual = d['residual']   # (604, 133, 147) — LSTM target (y)
        sic_phys = d['sic_phys']   # (604, 133, 147) — physics prediction (feature)
        sic_obs  = d['sic_obs']    # (604, 133, 147) — observed SIC (for validation)
    """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Stefan/FDD sea-ice physics baseline and output residuals for ML training."
    )
    parser.add_argument(
        "--sic_npz",
        default="ross_sea_concentration.npz",
        help="Path to ross_sea_concentration.npz (default: ross_sea_concentration.npz)"
    )
    parser.add_argument(
        "--output",
        default="seaice_physics_residuals.npz",
        help="Output .npz path (default: seaice_physics_residuals.npz)"
    )
    args = parser.parse_args()

    run_baseline_on_npz(
        sic_npz_path=args.sic_npz,
        output_npz_path=args.output,
    )
