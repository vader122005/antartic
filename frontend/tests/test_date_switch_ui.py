"""UI-level proof: pick a non-demo date in the app and run a forecast."""
import os
import sys
import datetime as dt

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)
os.chdir(_FRONTEND)
APP = os.path.join(_FRONTEND, "app.py")

import warnings
warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest

FAIL = []
def chk(n, c, d=""):
    print(("  [PASS] " if c else "  [FAIL] ") + n + (f"  {d}" if d else ""))
    if not c: FAIL.append(n)

for date, label in [(dt.date(2021,12,16), "2021-22"),
                    (dt.date(2022,11,1),  "2022-23"),
                    (dt.date(2024,2,15),  "2023-24")]:
    print(f"=== app: switch to {date} ({label}) and Run Forecast ===")
    at = AppTest.from_file(APP, default_timeout=900).run()
    at.date_input[0].set_value(date).run()
    chk("date accepted (no error)", len(at.error) == 0,
        at.error[0].value[:100] if at.error else "")
    btn = [b for b in at.button if "Run Forecast" in b.label][0]
    chk("Run Forecast enabled", not btn.disabled)
    btn.click().run()
    chk("no exception", not at.exception,
        str(at.exception[0].message)[:120] if at.exception else "")
    if at.exception:
        print(); continue

    par = at.session_state["pareto_routes"]
    chk("not impossible", par.get("impossible") is False,
        str(par.get("reason",""))[:100])
    m = par.get("metrics", {})
    chk("three route cards' data present",
        set(m) == {"fuel_priority","balanced","time_priority"}, str(sorted(m)))
    chk("three route geometries",
        all(par.get(k) for k in ("fuel_priority","balanced","time_priority")))
    md = " ".join(x.value for x in at.markdown)
    chk("Route Recommendations rendered", "Route Recommendations" in md)
    chk("map layers enabled", at.session_state["show_forecast_layers"] is True)
    chk("no FileNotFoundError warning",
        not any("Errno 2" in w.value or "No such file" in w.value for w in at.warning))
    chk("disclaimer present", "DECISION SUPPORT ONLY" in md)
    print()

print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
raise SystemExit(1 if FAIL else 0)
