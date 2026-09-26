"""
test_ice_upload.py
==================
Covers the "Ice Cover Data" source selector: model archive vs user upload,
every accepted file shape, and the failure paths.

The rule being protected: a bad or missing upload must never blank the map.
It falls back to the model field and says why.
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
    """Stands in for a Streamlit UploadedFile (needs .name and .size)."""

    def __init__(self, data, name):
        super().__init__(data)
        self.name = name
        self.size = len(data)


def sample_frame(n=8):
    lat = np.repeat(np.linspace(-76.0, -72.0, n), n)
    lon = np.tile(np.linspace(165.0, 175.0, n), n)
    conc = np.linspace(0.0, 100.0, lat.size)
    return pd.DataFrame({"lat": lat, "lon": lon, "concentration": conc})


def csv_upload(n=8):
    return FakeUpload(sample_frame(n).to_csv(index=False).encode(), "ice.csv")


def json_upload(n=6):
    return FakeUpload(sample_frame(n).to_json(orient="records").encode(), "ice.json")


def npz_upload(rows=12, cols=15):
    bio = io.BytesIO()
    np.savez(bio,
             lat=np.linspace(-78.0, -70.0, rows),
             lon=np.linspace(160.0, 179.0, cols),
             concentration=np.random.uniform(0.0, 100.0, (rows, cols)))
    return FakeUpload(bio.getvalue(), "ice.npz")


def main():
    print("=== UI: card present and defaults ===")
    at = AppTest.from_file(APP, default_timeout=900).run()
    chk("app loads without exception", not at.exception,
        str(at.exception[0].message)[:120] if at.exception else "")
    if at.exception:
        print("SUMMARY: FAILED (app did not load)")
        return 1

    md = " ".join(m.value for m in at.markdown)
    chk("'Ice Cover Data' card rendered", "Ice Cover Data" in md)

    radios = {r.label: r for r in at.radio}
    chk("ice source radio present", "Ice cover source" in radios)
    if "Ice cover source" in radios:
        r = radios["Ice cover source"]
        chk("default is 'Use model data'", r.value == "Use model data", str(r.value))
        chk("both options offered",
            list(r.options) == ["Use model data", "Upload file"], str(list(r.options)))

    caps = " ".join(c.value for c in at.caption)
    chk("model-mode info line shown",
        "Using ross_sea_concentration.npz" in caps, caps[:90])
    chk("model mode recorded in session",
        at.session_state["ice_input_mode"] == "model")
    n_model = len(at.session_state["ice_cells"])
    chk("model ice cover loaded", n_model > 0, f"{n_model} cells")

    print()
    print("=== UI: switch to 'Upload file' with NO file ===")
    at2 = AppTest.from_file(APP, default_timeout=900).run()
    at2.radio(key="ice_mode_radio").set_value("Upload file").run()
    chk("no exception", not at2.exception,
        str(at2.exception[0].message)[:120] if at2.exception else "")
    # AppTest models file_uploader as UnknownElement, whose .key is never
    # populated, so match on the proto label instead.
    def _uploader_labels(app):
        return [str(getattr(u.proto, "label", "")) for u in app.get("file_uploader")]

    chk("ice uploader is shown in upload mode",
        "Ice concentration file" in _uploader_labels(at2),
        str(_uploader_labels(at2)))
    chk("ice uploader hidden in model mode",
        "Ice concentration file" not in _uploader_labels(at),
        str(_uploader_labels(at)))
    hint = " ".join(c.value for c in at2.caption)
    chk("format hint shown",
        "lat, lon, concentration" in hint and "'lat', 'lon', 'concentration'" in hint)
    chk("falls back to model data, map not blank",
        len(at2.session_state["ice_cells"]) == n_model,
        f"{len(at2.session_state['ice_cells'])} cells")
    chk("mode recorded as upload", at2.session_state["ice_input_mode"] == "upload")
    chk("no error for merely choosing upload", len(at2.error) == 0)

    print()
    print("=== backend: each accepted upload shape renders ===")
    for label, f, expect in [("CSV", csv_upload(), 64),
                             ("JSON", json_upload(), 36),
                             ("NPZ", npz_upload(), None)]:
        cells = backend.load_ice_cover_data(date_str="20231115",
                                            uploaded_file=f, mode="upload")
        warns = backend.drain_warnings()
        errs = backend.drain_errors()
        ok = len(cells) > 0 and not errs
        if expect is not None:
            ok = ok and len(cells) == expect
        chk(f"{label} upload renders", ok, f"{len(cells)} cells, errs={len(errs)}")
        chk(f"{label} flagged as non-model data",
            any("uploaded file" in w for w in warns))
        chk(f"{label} polygons are [lon, lat] in range",
            all(-180 <= p[0] <= 180 and -90 <= p[1] <= 90
                for c in cells[:20] for p in c["polygon"]))
        chk(f"{label} thickness normalised 0-1",
            all(0.0 <= c["thickness"] <= 1.0 for c in cells))

    print()
    print("=== backend: bad uploads fall back with an error ===")
    bad = [
        ("wrong columns", FakeUpload(b"a,b,c\n1,2,3\n", "bad.csv")),
        ("binary junk", FakeUpload(b"\x00\x01\x02not-a-table", "junk.csv")),
        ("unsupported type", FakeUpload(b"x", "ice.txt")),
    ]
    for label, f in bad:
        cells = backend.load_ice_cover_data(date_str="20231115",
                                            uploaded_file=f, mode="upload")
        errs = backend.drain_errors()
        backend.drain_warnings()
        chk(f"{label}: model data still shown", len(cells) == n_model,
            f"{len(cells)} cells")
        chk(f"{label}: error names the expected fields",
            len(errs) == 1 and "lat, lon, concentration" in errs[0])

    print()
    print("=== UI: a bad upload surfaces st.error and keeps the map ===")
    at3 = AppTest.from_file(APP, default_timeout=900).run()
    at3.radio(key="ice_mode_radio").set_value("Upload file").run()
    at3.session_state["ice_upload"] = FakeUpload(b"a,b,c\n1,2,3\n", "bad.csv")
    at3.run()
    chk("no exception on bad upload", not at3.exception,
        str(at3.exception[0].message)[:120] if at3.exception else "")
    if not at3.exception:
        errs = " ".join(e.value for e in at3.error)
        chk("st.error shown in the UI",
            "Could not parse ice cover file" in errs, errs[:100])
        chk("map still shows model data",
            len(at3.session_state["ice_cells"]) == n_model,
            f"{len(at3.session_state['ice_cells'])} cells")
        md3 = " ".join(m.value for m in at3.markdown)
        chk("disclaimer still present", "DECISION SUPPORT ONLY" in md3)

    print()
    print("=== UI: a good CSV upload replaces the field on the map ===")
    at4 = AppTest.from_file(APP, default_timeout=900).run()
    at4.radio(key="ice_mode_radio").set_value("Upload file").run()
    at4.session_state["ice_upload"] = csv_upload()
    at4.run()
    chk("no exception", not at4.exception,
        str(at4.exception[0].message)[:120] if at4.exception else "")
    if not at4.exception:
        n_up = len(at4.session_state["ice_cells"])
        chk("uploaded field is on the map", n_up == 64, f"{n_up} cells")
        chk("differs from the model field", n_up != n_model,
            f"upload {n_up} vs model {n_model}")
        chk("no error for a good file", len(at4.error) == 0)
        chk("ice_upload_file recorded in session",
            getattr(at4.session_state["ice_upload_file"], "name", None) == "ice.csv")

        print()
        print("=== UI: switching back to model restores the archive field ===")
        at4.radio(key="ice_mode_radio").set_value("Use model data").run()
        chk("model field restored",
            len(at4.session_state["ice_cells"]) == n_model,
            f"{len(at4.session_state['ice_cells'])} cells")
        chk("upload cleared from session",
            at4.session_state["ice_upload_file"] is None)

    print()
    print("=== delimiter handling (regression: semicolon/tab CSV) ===")
    # A non-comma separator collapsed the header into ONE column named
    # "lat;lon;concentration", so all three required columns read as missing -
    # the reported "missing column(s): lat, lon, concentration" on a file whose
    # header plainly contained them. Excel emits ';' under a semicolon-list
    # locale, which is the usual way to hit this.
    EXACT = b"lat,lon,concentration\n-70.0,160.0,15.2\n-72.0,160.0,28.4\n"
    variants = [
        ("comma", EXACT),
        ("semicolon", EXACT.replace(b",", b";")),
        ("tab", EXACT.replace(b",", b"\t")),
        ("pipe", EXACT.replace(b",", b"|")),
        ("semicolon + CRLF + BOM",
         b"\xef\xbb\xbf" + EXACT.replace(b",", b";").replace(b"\n", b"\r\n")),
        ("uppercase + semicolon",
         b"LAT;LON;CONCENTRATION\n-70.0;160.0;15.2\n-72.0;160.0;28.4\n"),
    ]
    for label, data in variants:
        cells = backend.load_ice_cover_data(
            date_str="20231115", uploaded_file=FakeUpload(data, "ice.csv"),
            mode="upload")
        errs = backend.drain_errors()
        backend.drain_warnings()
        chk(f"{label} separator parses", len(cells) == 2 and not errs,
            f"{len(cells)} cells, errs={len(errs)}")
        if cells:
            chk(f"{label}: 15.2 percent -> thickness 0.152",
                abs(cells[0]["thickness"] - 0.152) < 1e-6,
                str(round(cells[0]["thickness"], 4)))

    print()
    print("=== stream already consumed (Streamlit re-serves the same object) ===")
    reused = FakeUpload(EXACT, "ice.csv")
    reused.read()          # exhaust it, as a previous parse would have
    cells = backend.load_ice_cover_data(date_str="20231115",
                                        uploaded_file=reused, mode="upload")
    errs = backend.drain_errors()
    backend.drain_warnings()
    chk("parses after EOF", len(cells) == 2 and not errs, f"{len(cells)} cells")

    print()
    print("=== parse diagnostics exposed for the debug expander ===")
    backend.load_ice_cover_data(
        date_str="20231115",
        uploaded_file=FakeUpload(b"a,b,c\n1,2,3\n", "bad.csv"),
        mode="upload")
    backend.drain_errors()
    backend.drain_warnings()
    chk("columns recorded",
        backend.LAST_PARSE_DEBUG.get("columns") == ["a", "b", "c"],
        str(backend.LAST_PARSE_DEBUG.get("columns")))
    chk("delimiter recorded", "delimiter" in backend.LAST_PARSE_DEBUG,
        str(backend.LAST_PARSE_DEBUG.get("delimiter")))

    print()
    print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
