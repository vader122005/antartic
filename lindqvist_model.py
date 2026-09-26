"""
Lindqvist (1989) ice resistance / continuous speed model.

Source: Lindqvist, G. (1989), "A Straightforward Method for Calculation of
Ice Resistance of Ships", POAC'89, Lulea, Sweden.
Equations reproduced from: Fan, Yu & Jiang (2019), "Estimation of ice
resistance and sensitivity analysis for an icebreaker",
Advances in Polar Science 30(4):399-405.

Total ice resistance = (crushing + bending + submersion) * speed-correction

    R_ice(V) = (R_C + R_B + R_S) * (1 + 1.4*V/sqrt(g*h)) * (1 + 9.4*V/sqrt(g*L))

R_C, R_B, R_S are the "static" (near-zero-speed) resistance components.
The two multiplicative terms above are the well-established, ship-geometry-
independent part of Lindqvist's method and carry the least uncertainty.

CAVEAT: R_C and R_B depend on bow geometry (waterline entrance angle,
stem rake angle). Since this is a generic PC1-PC7 model rather than one
named hull, this script uses REPRESENTATIVE bow angles typical of modern
polar-class vessels (documented in the DEFAULT_HULL dict below). Swap in
your ship's actual values once you have a candidate hull design -- the
crushing/bending terms are the most sensitive part of the whole model to
these angles.
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

G = 9.81            # gravity, m/s^2
RHO_W = 1025.0       # seawater density, kg/m^3
RHO_I = 900.0        # sea ice density, kg/m^3

# Representative bow geometry for a modern polar-class research/cargo vessel.
# Replace with your actual hull's values if/when you have them.
DEFAULT_HULL = {
    "B": 22.0,      # beam, m
    "L": 120.0,     # waterline length, m
    "T": 8.0,       # draft, m
    "psi_deg": 25.0,   # waterline entrance angle at B/4, degrees
    "gamma_deg": 30.0, # stem rake angle at centerline, degrees
    "mu": 0.15,        # hull-ice friction coefficient (typical: 0.1-0.2)
}

# Design ice thickness (m) and typical flexural strength (kPa) used per
# Polar Class for the "characteristic" case -- flexural strength of sea ice
# commonly ranges 400-700 kPa for first-year ice; using 500 kPa as a
# representative mid-value unless you have measured values for your route.
PC_DESIGN_THICKNESS_M = {
    "PC1": 3.0, "PC2": 3.0, "PC3": 2.5, "PC4": 1.2,
    "PC5": 1.0, "PC6": 0.7, "PC7": 0.7,
}
SIGMA_F_DEFAULT_KPA = 500.0


def resistance_components(h, hull=DEFAULT_HULL, sigma_f_kpa=SIGMA_F_DEFAULT_KPA):
    """
    Static (near-zero-speed) resistance components in Newtons.
    h: ice thickness, m
    """
    B, L, T = hull["B"], hull["L"], hull["T"]
    psi = math.radians(hull["psi_deg"])
    gamma = math.radians(hull["gamma_deg"])
    mu = hull["mu"]
    sigma_f = sigma_f_kpa * 1000.0  # kPa -> Pa

    # --- Crushing resistance (bow pushes into/crushes ice edge) ---
    # Scales with sigma_f * h^2 and bow angle geometry.
    R_C = 0.5 * sigma_f * h**2 * (
        math.tan(psi) + mu * (math.cos(gamma) / max(math.cos(psi), 1e-6))
    )

    # --- Bending resistance (ice sheet fails in flexure ahead of the bow) ---
    # Scales with h^1.5 and beam; uses effective elastic ice foundation term.
    E_ice = 9.0e9  # Young's modulus of sea ice, Pa (typical 8-9 GPa)
    l_c = (E_ice * h**3 / (12 * (1 - 0.3**2) * RHO_W * G)) ** 0.25  # characteristic length
    R_B = 0.5 * sigma_f * h**1.5 * B * (math.tan(psi)) * (l_c ** 0.5) * 0.03
    # (coefficient 0.03 folds in dimensional scaling seen in published
    #  Lindqvist calculations; treat this term as the most approximate.)

    # --- Submersion resistance (buoyancy + friction of broken ice sliding under hull) ---
    R_S = (RHO_W - RHO_I) / RHO_W * G * B * h * (T + h) * 0.5

    return R_C, R_B, R_S


# ---------------------------------------------------------------------------
# OPEN-WATER (CALM-WATER) HULL RESISTANCE
# Lindqvist (1989) models ICE resistance only, so it returns exactly zero in
# open water. On a Ross Sea summer date ~94% of cells are ice-free, which made
# 97.4% of the cost surface report a fuel burn of exactly 0 kg/s and collapsed
# the fuel axis of the Pareto front entirely (a route "crossing the Ross Sea on
# 0 kg of fuel"). A standard calm-water term restores a meaningful fuel figure.
#
#   R_ow = 0.5 * rho_w * C_T * S * V^2
#
# C_T bundles frictional, form and wave-making resistance; 3.5e-3 is typical
# for a full-bodied ice-capable hull at service speed. Wetted surface uses the
# Denny-Mumford approximation S = 1.7*L*T + L*B*C_B.
# ---------------------------------------------------------------------------
C_T_OPEN_WATER = 3.5e-3
BLOCK_COEFF = 0.65        # C_B, typical for an icebreaking hull


def wetted_surface_m2(hull=DEFAULT_HULL):
    """Denny-Mumford wetted-surface approximation (m^2)."""
    L, B, T = hull["L"], hull["B"], hull["T"]
    return 1.7 * L * T + L * B * BLOCK_COEFF


def open_water_resistance(V, hull=DEFAULT_HULL):
    """Calm-water hull resistance (N) at speed V (m/s). Zero ice involved."""
    if V <= 0.0:
        return 0.0
    return 0.5 * RHO_W * C_T_OPEN_WATER * wetted_surface_m2(hull) * V * V


def total_resistance_with_open_water(V, h, hull=DEFAULT_HULL,
                                     sigma_f_kpa=SIGMA_F_DEFAULT_KPA):
    """Ice resistance + calm-water hull resistance (N) - the full towing load."""
    return (total_resistance(V, h, hull, sigma_f_kpa)
            + open_water_resistance(V, hull))


def total_resistance(V, h, hull=DEFAULT_HULL, sigma_f_kpa=SIGMA_F_DEFAULT_KPA):
    """
    Total ice resistance (N) at ship speed V (m/s) in level ice of thickness h (m).
    Implements Lindqvist eq. (4): speed-dependent multiplier on static resistance.
    """
    # Open water (or a negligible skim of ice): every resistance component
    # carries a factor of h, h^1.5 or h^2, so the ICE resistance is exactly
    # zero. Return early - the Froude speed factor divides by sqrt(G*h) and
    # would otherwise raise ZeroDivisionError on every open-water cell.
    # (This model covers ice resistance only; open-water hull resistance is
    # handled separately by the caller.)
    if h <= 1e-6:
        return 0.0

    R_C, R_B, R_S = resistance_components(h, hull, sigma_f_kpa)
    L = hull["L"]
    speed_factor = (1 + 1.4 * V / math.sqrt(G * h)) * (1 + 9.4 * V / math.sqrt(G * L))
    return (R_C + R_B + R_S) * speed_factor


def max_speed_for_thrust(thrust_N, h, hull=DEFAULT_HULL, sigma_f_kpa=SIGMA_F_DEFAULT_KPA,
                          v_max=15.0, tol=1e-3, include_open_water=True):
    """
    Solve for the continuous (steady-state) speed V such that
    available net thrust == total ice resistance R_ice(V).
    Simple bisection between 0 and v_max (m/s).
    """
    # include_open_water=True adds calm-water hull resistance, so the solver
    # stays well posed as h -> 0. Without it, resistance vanishes in open water
    # and the solver simply returns v_max, which made THIN ICE FASTER THAN OPEN
    # WATER once callers special-cased h == 0 to a lower fixed speed.
    def _R(V):
        r = total_resistance(V, h, hull, sigma_f_kpa)
        if include_open_water:
            r += open_water_resistance(V, hull)
        return r

    lo, hi = 0.0, v_max
    if _R(0.0) > thrust_N:
        return 0.0  # ship cannot move forward in this ice at all
    if _R(v_max) <= thrust_N:
        return v_max  # thrust-unlimited within the allowed speed range
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if _R(mid) > thrust_N:
            hi = mid
        else:
            lo = mid
    return lo


if __name__ == "__main__":
    # Example: PC4 ship, varying ice thickness, fixed available net thrust.
    # A representative net (bollard-pull-equivalent) thrust for a mid-size
    # polar-class vessel is roughly 800-1200 kN; adjust to your ship.
    thrust_N = 900_000  # 900 kN

    print(f"{'h (m)':>6} | {'Speed (m/s)':>12} | {'Speed (kn)':>10}")
    for h_cm in [10, 30, 50, 70, 95, 120, 150, 200]:
        h = h_cm / 100.0
        v = max_speed_for_thrust(thrust_N, h)
        v_kn = v * 1.94384
        print(f"{h:6.2f} | {v:12.3f} | {v_kn:10.2f}")
