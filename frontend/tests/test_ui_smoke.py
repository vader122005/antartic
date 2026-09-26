"""Headless checks for the Streamlit app using streamlit.testing AppTest."""
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

FAIL = []
def chk(name, cond, detail=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))
    if not cond: FAIL.append(name)

print("=== 1. first load ===")
at = AppTest.from_file(APP, default_timeout=600).run()
chk("app runs with no exception", not at.exception,
    str(at.exception[0].message) if at.exception else "")

if at.exception:
    raise SystemExit(1)

md = " ".join(m.value for m in at.markdown)
chk("IMO disclaimer visible on first load", "DECISION SUPPORT ONLY" in md)
chk("disclaimer cites SOLAS V/34", "SOLAS regulation V/34" in md)
chk("disclaimer cites Polar Code Ch.11", "Polar Code Chapter 11" in md)
chk("red border on disclaimer", "#c0392b" in md)
chk("Forecast Accuracy panel present", "Forecast Accuracy" in md)
chk("accuracy: sea-ice coverage figure", "21.5%" in md)
chk("accuracy: sea-ice skill figures", "+2.7%" in md and "+1.4%" in md)
chk("accuracy: iceberg B38 skill", "+5.0%" in md and "69 genuine drift pairs" in md)
chk("accuracy: CMEMS gap surfaced", "CMEMS skipped" in md)
chk("accuracy: Lindqvist cited", "Lindqvist (1989)" in md)

sels = {s.label: s for s in at.selectbox}
chk("Polar Class selector present", "Polar Class" in sels)
if "Polar Class" in sels:
    chk("Polar Class default PC4", sels["Polar Class"].value == "PC4",
        f"got {sels['Polar Class'].value}")
    chk("Polar Class has PC1-PC7", list(sels["Polar Class"].options) ==
        ["PC1","PC2","PC3","PC4","PC5","PC6","PC7"])

dates = {d.label: d for d in at.date_input}
chk("Forecast date picker present", "Forecast date" in dates)
if "Forecast date" in dates:
    chk("date default = 2023-11-15", str(dates["Forecast date"].value) == "2023-11-15",
        str(dates["Forecast date"].value))

cbs = {c.label: c for c in at.checkbox}
chk("Uncertainty rings toggle present", "Uncertainty" in cbs)
if "Uncertainty" in cbs:
    chk("Uncertainty rings default ON", cbs["Uncertainty"].value is True)

chk("no blocking error on first load", len(at.error) == 0,
    at.error[0].value if at.error else "")
print(f"  (info) markdown blocks={len(at.markdown)} warnings={len(at.warning)}")
for w in at.warning:
    print("   warn:", w.value[:120])

print()
print("=== 2. invalid date rejected (2023-07-15, outside Nov-Mar) ===")
at2 = AppTest.from_file(APP, default_timeout=600).run()
import datetime as dt
at2.date_input[0].set_value(dt.date(2023, 7, 15)).run()
errs = " ".join(e.value for e in at2.error)
chk("error shown for out-of-season date", "Nov-Mar shipping season" in errs, errs[:120])
btns = [b for b in at2.button if "Run Forecast" in b.label]
chk("Run Forecast disabled on invalid date", bool(btns) and btns[0].disabled)

print()
print("=== 3. valid in-season date accepted (2022-11-15) ===")
at3 = AppTest.from_file(APP, default_timeout=600).run()
at3.date_input[0].set_value(dt.date(2022, 11, 15)).run()
chk("no error for valid date", len(at3.error) == 0,
    at3.error[0].value[:120] if at3.error else "")

print()
print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
raise SystemExit(1 if FAIL else 0)
