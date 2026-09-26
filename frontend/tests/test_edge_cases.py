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
from streamlit.testing.v1 import AppTest
FAIL=[]
def chk(n,c,d=""):
    print(("  [PASS] " if c else "  [FAIL] ")+n+(f"  {d}" if d else ""))
    if not c: FAIL.append(n)

print("=== impossible route: start outside the Ross Sea box (lon 0) ===")
at = AppTest.from_file(APP, default_timeout=900).run()
at.number_input(key="ship_lon").set_value(0.0).run()
[b for b in at.button if "Run Forecast" in b.label][0].click().run()
chk("no crash", not at.exception, str(at.exception[0].message) if at.exception else "")
if not at.exception:
    par=at.session_state["pareto_routes"]; md=" ".join(x.value for x in at.markdown)
    chk("impossible flag set", par.get("impossible") is True, str(par.get("reason",""))[:90])
    chk("red 'No viable route' banner shown", "No viable route" in md)
    chk("banner names the polar class", "PC4" in md)
    chk("banner suggests upgrade/destination", "upgrading Polar Class" in md)
    chk("backend reason surfaced", "bounding box" in md.lower())
    chk("no route cards when impossible", "FUEL PRIORITY" not in md)
    chk("disclaimer still present", "DECISION SUPPORT ONLY" in md)

print()
print("=== backend unavailable -> graceful mock fallback ===")
import os
import shutil
import tempfile
import importlib.util

# Load a copy of the resolver from an ISOLATED temp directory. Placing it under
# tests/ would let `yield FRONTEND_DIR.parent` land on the frontend folder and
# pass for the wrong reason; from a temp dir, no candidate can possibly hit the
# real backend, so this genuinely exercises the not-found path.
_tmpdir = tempfile.mkdtemp(prefix="bp_isolated_")
_tmp = os.path.join(_tmpdir, "_bp_tmp.py")
shutil.copyfile(os.path.join(_FRONTEND, "backend_paths.py"), _tmp)
_spec = importlib.util.spec_from_file_location("_bp_tmp", _tmp)
_bp_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bp_test)

chk("isolated resolver finds no backend", _bp_test.find_backend_dir() is None,
    str(_bp_test.find_backend_dir()))
chk("BACKEND_AVAILABLE is False", _bp_test.BACKEND_AVAILABLE is False)
chk("describe() says not found", "NOT found" in _bp_test.describe())
chk("has_data() returns False, does not raise",
    _bp_test.has_data("ross_sea_concentration.npz") is False)
chk("cost_surface_path() returns None", _bp_test.cost_surface_path("PC4", "20231115") is None)
chk("ensure_on_path() returns False", _bp_test.ensure_on_path() is False)
try:
    _bp_test.data_path("x.npz")
    chk("data_path raises when missing", False)
except FileNotFoundError as e:
    chk("data_path raises a clear FileNotFoundError", "ANTARCTIC_NAV_BACKEND" in str(e))

# An env override pointing at a non-existent path must also be rejected.
os.environ["ANTARCTIC_NAV_BACKEND"] = os.path.join(_tmpdir, "does_not_exist")
_spec.loader.exec_module(_bp_test)
chk("bogus ANTARCTIC_NAV_BACKEND rejected", _bp_test.find_backend_dir() is None)
os.environ.pop("ANTARCTIC_NAV_BACKEND", None)
shutil.rmtree(_tmpdir, ignore_errors=True)

# And the UI layer must degrade to mock data rather than crash.
import backend_stubs as _bs
_real = _bs.bp.BACKEND_AVAILABLE
try:
    _bs.bp.BACKEND_AVAILABLE = False
    _bs.LAST_WARNINGS.clear()
    cells = _bs.load_ice_cover_data("20231115")
    bergs = _bs.load_current_iceberg_positions(date_str="20231115")
    traj, par = _bs.run_forecast(bergs, {"mode": "manual", "manual_params": {}},
                                 {"lat": -75.0, "lon": 165.0},
                                 {"lat": -73.0, "lon": -155.0}, "PC4", "20231115")
    warns = " ".join(_bs.drain_warnings())
    chk("mock ice cover returned, no crash", len(cells) > 0, f"{len(cells)} cells")
    chk("mock icebergs returned, no crash", len(bergs) > 0, f"{len(bergs)} bergs")
    chk("mock forecast returns a route", bool(par.get("balanced")))
    chk("user is warned backend is unavailable", "Backend unavailable" in warns)
finally:
    _bs.bp.BACKEND_AVAILABLE = _real

print()
print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
raise SystemExit(1 if FAIL else 0)
