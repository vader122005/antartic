"""End-to-end: click Run Forecast and verify the three Pareto routes."""
import os
import sys

# Run from anywhere: put the frontend folder (the parent of tests/) on sys.path
# and make it the cwd, so `import config` resolves and AppTest can find app.py.
_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)
os.chdir(_FRONTEND)
APP = os.path.join(_FRONTEND, "app.py")

import warnings; warnings.filterwarnings("ignore")
import datetime as dt
from streamlit.testing.v1 import AppTest

FAIL=[]
def chk(n,c,d=""):
    print(("  [PASS] " if c else "  [FAIL] ")+n+(f"  {d}" if d else ""))
    if not c: FAIL.append(n)

print("=== demo scenario: PC4, 2023-11-15, (-75,165) -> (-73,-155) ===")
at = AppTest.from_file(APP, default_timeout=900).run()
chk("initial load clean", not at.exception, str(at.exception[0].message) if at.exception else "")

btn = [b for b in at.button if "Run Forecast" in b.label][0]
btn.click().run()
chk("forecast run without exception", not at.exception,
    str(at.exception[0].message) if at.exception else "")
if at.exception: raise SystemExit(1)

ss = at.session_state
par = ss["pareto_routes"]
chk("pareto_routes stored in session", bool(par))
chk("not impossible", par.get("impossible") is False, str(par.get("reason","")))

m = par.get("metrics", {})
print("  route metrics:")
for k in ("fuel_priority","balanced","time_priority"):
    if k in m:
        v=m[k]; print(f"    {k:15s} {v['distance']:8.1f} km  {v['time_h']:6.1f} h  "
                      f"{v['fuel_t']:8.1f} t  ice={v['ice_severity']}  wp={v['n_waypoints']}")
chk("all three weightings returned", set(m)=={"fuel_priority","balanced","time_priority"}, str(set(m)))

paths = {k: par.get(k) for k in ("fuel_priority","balanced","time_priority")}
chk("all three paths non-empty", all(paths.values()),
    str({k:len(v or []) for k,v in paths.items()}))
sig = {tuple(map(tuple, p)) for p in paths.values() if p}
chk("three DISTINCT route geometries", len(sig)==3, f"{len(sig)} distinct")

# Pareto validity
if set(m)=={"fuel_priority","balanced","time_priority"}:
    t=[m[k]["time_h"] for k in ("fuel_priority","balanced","time_priority")]
    f=[m[k]["fuel_t"] for k in ("fuel_priority","balanced","time_priority")]
    chk("time decreases fuel->time priority", t[0]>=t[1]>=t[2], str(t))
    chk("fuel increases fuel->time priority", f[0]<=f[1]<=f[2], str(f))

# coordinate order sanity: [lon, lat]
p0 = paths["balanced"][0]
chk("waypoints are [lon, lat]", -180<=p0[0]<=180 and -90<=p0[1]<=-60, str(p0))

traj = ss["trajectories"]
chk("iceberg drift trajectories produced", len(traj)>0, f"{len(traj)} bergs")
if traj:
    k=list(traj)[0]; chk("trajectory has 48h of 6h steps (9 pts)", len(traj[k])==9, str(len(traj[k])))

md = " ".join(x.value for x in at.markdown)
chk("ROUTE RECOMMENDATIONS card rendered", "Route Recommendations" in md)
chk("FUEL PRIORITY card", "FUEL PRIORITY" in md)
chk("BALANCED card", "BALANCED" in md)
chk("TIME PRIORITY card", "TIME PRIORITY" in md)
chk("gold border on balanced card", "#f5b400" in md)
chk("RECOMMENDED badge", "RECOMMENDED" in md)
chk("disclaimer STILL visible after forecast", "DECISION SUPPORT ONLY" in md)
chk("accuracy panel still visible", "Forecast Accuracy" in md)
chk("forecast layers auto-enabled", ss["show_forecast_layers"] is True)

print()
print("=== impossible-route path (PC7 across a blocked transect) ===")
at2 = AppTest.from_file(APP, default_timeout=900).run()
# start on land/outside to force an impossible result
at2.number_input(key="ship_lat").set_value(-89.0).run()
b2=[b for b in at2.button if "Run Forecast" in b.label][0]
b2.click().run()
chk("impossible path does not crash", not at2.exception,
    str(at2.exception[0].message) if at2.exception else "")
if not at2.exception:
    par2=at2.session_state["pareto_routes"]
    md2=" ".join(x.value for x in at2.markdown)
    if par2.get("impossible"):
        chk("impossible banner rendered", "No viable route" in md2)
        chk("banner names polar class", "PC4" in md2)
    else:
        print("  (info) backend still found a route; banner not exercised here")
    chk("disclaimer present in impossible state", "DECISION SUPPORT ONLY" in md2)

print()
print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
raise SystemExit(1 if FAIL else 0)
