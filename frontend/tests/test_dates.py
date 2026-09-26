"""
test_dates.py
=============
Regression test for the on-the-fly cost-surface path.

Before the fix, only 2023-11-15 worked. Every other valid Nov-Mar date failed
with:

    could not obtain a cost surface:
    [Errno 2] No such file or directory: 'ross_sea_concentration.npz'

because route_optimiser.build_cost_surface_for() resolves its inputs relative
to the process CWD, which under Streamlit is the frontend folder.

Checks, per date: no missing-file error, three Pareto routes with real
geometry, the working directory is left untouched, and the on-the-fly build
finishes well inside the 10 s budget.
"""

import os
import sys
import time

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)
os.chdir(_FRONTEND)

import shutil
import warnings

warnings.filterwarnings("ignore")

import backend_stubs as backend

DATES = [
    ("20211216", "2021-22 season"),
    ("20221101", "2022-23 season"),
    ("20231115", "demo date (pre-built)"),
    ("20240215", "2023-24 season"),
]

START = {"lat": -75.0, "lon": 165.0}
END = {"lat": -73.0, "lon": -155.0}

FAIL = []


def chk(name, cond, detail=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def run_one(date_str):
    bergs = backend.load_current_iceberg_positions(date_str=date_str)
    weather = backend.parse_weather_input(
        "manual", manual_params={"wind_speed_kt": 15, "wind_direction_deg": 225})
    t0 = time.time()
    traj, pareto = backend.run_forecast(bergs, weather, START, END,
                                        polar_class="PC4", date_str=date_str)
    elapsed = time.time() - t0
    return traj, pareto, elapsed, backend.drain_warnings()


def main():
    # Force a genuine first-time build for the non-demo dates.
    shutil.rmtree(backend.CACHE_DIR, ignore_errors=True)

    # app.py calls prewarm() at startup, so a real user never pays the one-time
    # ~8.5 s ERA5 decompression inside their first forecast. Calling
    # run_forecast() directly would bill it to whichever date happens to run
    # first, making the per-date timing load-sensitive and measuring something
    # no user experiences. The cost-surface build - the thing the 10 s budget
    # is actually about - is still timed separately below.
    backend.prewarm(background=False)

    print("=== cost surface build cost (the thing the 10 s budget is about) ===")
    for date_str, label in DATES:
        t0 = time.time()
        path, how = backend.cost_surface_for("PC4", date_str)
        build_s = time.time() - t0
        backend.drain_warnings()
        chk(f"{date_str} surface ready ({how}) in {build_s:.1f}s",
            os.path.exists(path) and build_s < 10.0)

    print()
    print("=== full forecast per date ===")
    cwd_before = os.getcwd()
    for date_str, label in DATES:
        print(f"  --- {date_str}  ({label}) ---")
        traj, pareto, elapsed, warns = run_one(date_str)

        missing_file = [w for w in warns
                        if "Errno 2" in w or "No such file" in w]
        chk("no missing-file error", not missing_file, str(missing_file[:1]))
        chk("not impossible", pareto.get("impossible") is False,
            str(pareto.get("reason", ""))[:90])

        metrics = pareto.get("metrics", {})
        chk("three Pareto routes",
            set(metrics) == {"fuel_priority", "balanced", "time_priority"},
            str(sorted(metrics)))
        chk("all three have geometry",
            all(pareto.get(k) for k in
                ("fuel_priority", "balanced", "time_priority")),
            str({k: len(pareto.get(k) or []) for k in
                 ("fuel_priority", "balanced", "time_priority")}))
        chk("waypoints are [lon, lat]",
            bool(pareto.get("balanced")) and -180 <= pareto["balanced"][0][0] <= 180
            and -90 <= pareto["balanced"][0][1] <= -55,
            str(pareto.get("balanced", [[None, None]])[0]))
        chk("iceberg drift produced", len(traj) > 0, f"{len(traj)} bergs")
        chk("working directory restored", os.getcwd() == cwd_before)
        chk(f"warm forecast under 10s ({elapsed:.1f}s)", elapsed < 10.0)

        if metrics:
            for k in ("fuel_priority", "balanced", "time_priority"):
                if k in metrics:
                    m = metrics[k]
                    print(f"       {k:15s} {m['distance']:8.1f} km "
                          f"{m['time_h']:6.1f} h {m['fuel_t']:8.1f} t  "
                          f"ice={m['ice_severity']}")
        print()

    print("=== second pass: cached surfaces must be fast ===")
    for date_str, _ in DATES:
        _, pareto, elapsed, _ = run_one(date_str)
        chk(f"{date_str} cached re-run under 3s ({elapsed:.1f}s)", elapsed < 3.0)

    print()
    print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
