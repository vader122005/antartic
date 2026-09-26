"""
test_date_visibility.py
=======================
The Forecast date picker is hidden while ice cover comes from an upload, and
Polar Class takes the full width. Validation is skipped in that state so the
run is never blocked by a control the user cannot see.

Also pins the behaviour that makes hiding it safe: an uploaded ice field is
DISPLAY ONLY, so routing still runs against the model archive at the retained
date and must still produce routes.
"""

import os
import sys

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)
os.chdir(_FRONTEND)
APP = os.path.join(_FRONTEND, "app.py")

import io
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from streamlit.testing.v1 import AppTest
import backend_stubs as backend

FAIL = []


def chk(name, cond, detail=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


class FakeUpload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def ice_csv():
    lat = np.repeat(np.linspace(-76.0, -72.0, 6), 6)
    lon = np.tile(np.linspace(165.0, 175.0, 6), 6)
    df = pd.DataFrame({"lat": lat, "lon": lon,
                       "concentration": np.linspace(0.0, 100.0, lat.size)})
    return FakeUpload(df.to_csv(index=False).encode(), "ice.csv")


def date_labels(app):
    return [d.label for d in app.date_input]


def main():
    print("=== model mode: date picker visible ===")
    at = AppTest.from_file(APP, default_timeout=900).run()
    chk("app loads", not at.exception,
        str(at.exception[0].message)[:110] if at.exception else "")
    if at.exception:
        print("SUMMARY: FAILED")
        return 1
    chk("Forecast date shown", "Forecast date" in date_labels(at), str(date_labels(at)))
    chk("Polar Class shown", any(s.label == "Polar Class" for s in at.selectbox))
    chk("Run Forecast enabled",
        not [b for b in at.button if "Run Forecast" in b.label][0].disabled)

    print()
    print("=== upload mode: date picker hidden ===")
    at2 = AppTest.from_file(APP, default_timeout=900).run()
    at2.radio(key="ice_mode_radio").set_value("Upload file").run()
    at2.session_state["ice_upload"] = ice_csv()
    at2.run()
    chk("no exception", not at2.exception,
        str(at2.exception[0].message)[:110] if at2.exception else "")
    chk("Forecast date HIDDEN", "Forecast date" not in date_labels(at2),
        str(date_labels(at2)))
    chk("Polar Class still shown",
        any(s.label == "Polar Class" for s in at2.selectbox))
    chk("no season-validation error", len(at2.error) == 0,
        at2.error[0].value[:90] if at2.error else "")
    chk("Run Forecast still enabled",
        not [b for b in at2.button if "Run Forecast" in b.label][0].disabled)
    caps = " ".join(c.value for c in at2.caption)
    chk("routing-date note removed from the panel",
        "Routing still uses the model archive" not in caps, caps[-120:])
    chk("uploaded ice on the map", len(at2.session_state["ice_cells"]) == 36,
        f"{len(at2.session_state['ice_cells'])} cells")

    print()
    print("=== upload mode: forecast still produces routes ===")
    # AppTest does not model file_uploader (it is an UnknownElement), so an
    # injected session_state["ice_upload"] is reset to None on the next rerun.
    # Real Streamlit keeps the file attached - verified in a live browser:
    # after Run Forecast the file chip, the "Showing <file> instead of the
    # model field" caption and the uploaded field all persist. Re-inject to
    # reproduce the real widget's behaviour.
    _uploaded = ice_csv()
    at2.session_state["ice_upload"] = _uploaded
    [b for b in at2.button if "Run Forecast" in b.label][0].click()
    at2.session_state["ice_upload"] = _uploaded
    at2.run()
    chk("no exception on run", not at2.exception,
        str(at2.exception[0].message)[:110] if at2.exception else "")
    if not at2.exception:
        par = at2.session_state["pareto_routes"]
        chk("not impossible", par.get("impossible") is False,
            str(par.get("reason", ""))[:90])
        chk("three Pareto routes",
            set(par.get("metrics", {})) ==
            {"fuel_priority", "balanced", "time_priority"},
            str(sorted(par.get("metrics", {}))))
        chk("uploaded ice survives the run",
            len(at2.session_state["ice_cells"]) == 36,
            f"{len(at2.session_state['ice_cells'])} cells")
        chk("routing used the retained archive date, not today",
            not any("not present in ross_sea_concentration" in w.value
                    for w in at2.warning))

    print()
    print("=== switching back restores the picker and the model field ===")
    at2.radio(key="ice_mode_radio").set_value("Use model data").run()
    chk("Forecast date visible again", "Forecast date" in date_labels(at2),
        str(date_labels(at2)))
    chk("model ice field restored",
        len(at2.session_state["ice_cells"]) > 1000,
        f"{len(at2.session_state['ice_cells'])} cells")

    print()
    print("=== backend: date_str=None must not fall back to today ===")
    bergs = backend.load_current_iceberg_positions(date_str="20231115")
    backend.drain_warnings()
    weather = backend.parse_weather_input("manual", manual_params={})
    _, par = backend.run_forecast(bergs, weather, {"lat": -75.0, "lon": 165.0},
                                  {"lat": -73.0, "lon": -155.0}, "PC4", None)
    backend.drain_warnings(); backend.drain_errors()
    chk("None date still routes", par.get("impossible") is False,
        str(par.get("reason", ""))[:90])
    chk("three routes from a None date",
        set(par.get("metrics", {})) ==
        {"fuel_priority", "balanced", "time_priority"})

    print()
    print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
