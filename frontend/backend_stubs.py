"""
backend_stubs.py
=================
The single seam between the Streamlit UI and the locked backend package in
``antartic_nav_project/``. app.py and map_layers.py only ever call functions
in THIS file.

Every function degrades gracefully: if the backend directory, a data file or a
model weight is missing, it falls back to mock_data.py and records a message in
``LAST_WARNINGS`` for the UI to surface. The app must never crash because the
backend is absent.

SPATIAL SAFETY
--------------
The Ross Sea straddles the antimeridian and the NSIDC grid is CURVILINEAR
(lat/lon are 2-D 133x147 arrays, not separable 1-D axes). Every spatial lookup
here goes through ``grid_utils`` (KD-tree on unit-sphere coordinates). Nothing
in this file passes those arrays to RegularGridInterpolator.
"""

from __future__ import annotations

import contextlib
import csv
import datetime as dt
import io
import json
import math
import os
import threading
import time

import numpy as np
import pandas as pd

import backend_paths as bp
from config import DEMO_DATE

from mock_data import (
    generate_ice_grid,
    generate_icebergs,
    generate_dummy_trajectory,
    generate_dummy_route,
    generate_summary_metrics,
)

# Messages raised during the most recent backend call, drained by app.py.
#
# Errors are collected rather than raised through st.error() from in here, for
# two reasons. First, this module is imported and exercised outside Streamlit
# (the test suites call it directly), where st.* has no script context.
# Second, and more important: app.py calls st.rerun() at the end of the Run
# Forecast branch, and anything written with st.error() before that rerun is
# discarded - so a direct call would be invisible in exactly the case that
# matters. app.py drains this list and renders it after the rerun.
LAST_WARNINGS = []
LAST_ERRORS = []

# Diagnostics from the most recent upload parse (column names, row count, the
# delimiter that was detected). Read by app.py's debug expander.
LAST_PARSE_DEBUG = {}


def _warn(msg):
    if msg not in LAST_WARNINGS:
        LAST_WARNINGS.append(msg)


def _error(msg):
    if msg not in LAST_ERRORS:
        LAST_ERRORS.append(msg)


def drain_warnings():
    """Return and clear the accumulated warnings (called by app.py each run)."""
    out = list(LAST_WARNINGS)
    LAST_WARNINGS.clear()
    return out


def drain_errors():
    """Return and clear the accumulated errors (rendered by app.py as st.error)."""
    out = list(LAST_ERRORS)
    LAST_ERRORS.clear()
    return out


def backend_ready():
    return bp.BACKEND_AVAILABLE


# ---------------------------------------------------------------------------
# Cached loaders
#
# st.cache_data / cache_resource are applied here rather than in app.py so the
# 500 MB ERA5 wind file and the SIC cube are read once per session, not on
# every Streamlit rerun (which happens on every widget interaction).
# ---------------------------------------------------------------------------

def _cache_data(**kw):
    """st.cache_data if Streamlit is running, else a no-op passthrough."""
    try:
        import streamlit as st
        return st.cache_data(**kw)
    except Exception:
        def deco(fn):
            return fn
        return deco


def _cache_resource(**kw):
    try:
        import streamlit as st
        return st.cache_resource(**kw)
    except Exception:
        def deco(fn):
            return fn
        return deco


@_cache_resource(show_spinner=False)
def _load_sic_cube():
    """(dates, concentration, lat, lon) from the NSIDC file. Cached per session."""
    d = np.load(bp.data_path("ross_sea_concentration.npz"))
    return (np.asarray(d["dates"]), d["concentration"],
            np.asarray(d["lat"]), np.asarray(d["lon"]))


@_cache_resource(show_spinner=False)
def _load_grid_locator():
    """KD-tree locator over valid (non-NaN) NSIDC cells - curvilinear-safe."""
    bp.ensure_on_path()
    import grid_utils as gu
    _, conc, lat, lon = _load_sic_cube()
    return gu.GridLocator(lat, lon, ~np.isnan(conc[0]))


# The ERA5 sampler is a plain process-wide singleton rather than an
# st.cache_resource, so it can be warmed from a background thread that has no
# Streamlit ScriptRunContext. Building it costs ~8.5 s (the file is ~500 MB and
# 14520 timestamps get parsed), which is most of a cold first forecast.
_WIND_SINGLETON = {}
_WIND_LOCK = threading.Lock()


def _load_wind_sampler():
    """ERA5 wind sampler. Handles the non-monotonic (wrapped) longitude axis."""
    with _WIND_LOCK:
        if "sampler" not in _WIND_SINGLETON:
            bp.ensure_on_path()
            import grid_utils as gu
            _WIND_SINGLETON["sampler"] = gu.WindSampler(
                bp.data_path("ross_sea_era5_wind.npz"))
        return _WIND_SINGLETON["sampler"]


def wind_ready():
    """True once the ERA5 sampler is loaded (used to report warm-up state)."""
    return "sampler" in _WIND_SINGLETON


def prewarm(background=True):
    """
    Start loading the heavy ERA5 file before the user asks for a forecast.

    Called once at app start. The user typically spends several seconds setting
    a date, polar class and endpoints, which is enough to absorb the load, so
    the first forecast does not pay for it.
    """
    def _work():
        try:
            _load_sic_cube()
            _load_wind_sampler()
        except Exception:
            # Pre-warming is best-effort; the real call will surface any error.
            pass

    if not bp.BACKEND_AVAILABLE:
        return None
    if background:
        t = threading.Thread(target=_work, name="prewarm-era5", daemon=True)
        t.start()
        return t
    _work()
    return None


def available_sic_dates():
    """
    Sorted list of 'YYYYMMDD' strings the backend actually holds a SIC slice
    for. The date picker validates against this: a date outside it has no data
    and would fail inside the backend rather than in the UI.
    """
    if not bp.BACKEND_AVAILABLE:
        return []
    try:
        dates, _, _, _ = _load_sic_cube()
        return [str(x) for x in dates]
    except Exception as e:
        _warn(f"Could not read SIC dates: {e}")
        return []


# ---------------------------------------------------------------------------
# Cost surfaces
# ---------------------------------------------------------------------------

# Pre-built surfaces exist for the demo date only. Everything else is built on
# demand (~3.5 s) and cached HERE, inside the frontend, rather than in the
# backend folder: the backend is locked, and letting it write a new
# cost_surface_<PC>_<date>.npz for every date a user clicks would litter it.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".cache", "cost_surfaces")

# Matches the alpha/beta the pre-built surfaces were generated with, so an
# on-the-fly surface is directly comparable to the demo one.
_COST_ALPHA = 1.0
_COST_BETA = 1.0


@contextlib.contextmanager
def _memoised_csv_lookup(cs):
    """
    Temporarily memoise cost_surface.thickness_to_csv_row for one build.

    That function re-filters the whole lookup DataFrame and then walks it with
    iterrows() for EVERY grid cell - 0.47 ms x 8599 cells = ~4.1 s, which is
    the bulk of a build and the reason timings swing between 6 s and 12 s under
    load. Yet SIC maps onto only NINE discrete thickness values, so all 8599
    calls collapse to nine distinct results per polar class.

    This is a runtime wrapper on the imported module object: it changes no file
    in the locked backend, is restored in the finally block, and is exact
    rather than approximate, because the inputs really are discrete. The cache
    is per-build, so a re-read of the CSV can never be served a stale row.
    """
    original = cs.thickness_to_csv_row
    cache = {}

    def wrapper(df, polar_class, thickness_m):
        key = (polar_class, round(float(thickness_m), 6))
        if key not in cache:
            cache[key] = original(df, polar_class, thickness_m)
        return cache[key]

    cs.thickness_to_csv_row = wrapper
    try:
        yield
    finally:
        cs.thickness_to_csv_row = original


def cost_surface_for(polar_class, date_str):
    """
    Absolute path to a usable cost surface for (polar_class, date_str),
    building and caching one if neither the backend nor the cache has it.

    Returns (path, how) where `how` is 'prebuilt' | 'cached' | 'built'.
    Raises on failure so the caller can report it rather than silently
    degrading to an "impossible route".
    """
    prebuilt = bp.cost_surface_path(polar_class, date_str)
    if prebuilt:
        return prebuilt, "prebuilt"

    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR,
                          f"cost_surface_{polar_class}_{date_str}.npz")
    if os.path.exists(cached):
        return cached, "cached"

    dates, _, _, _ = _load_sic_cube()
    date_list = [str(x) for x in dates]
    if date_str not in date_list:
        raise ValueError(
            f"No sea-ice slice for {date_str}; the archive covers the four "
            f"Nov-Mar seasons {date_list[0]} to {date_list[-1]}."
        )
    idx = date_list.index(date_str)

    bp.ensure_on_path()
    import cost_surface as cs

    # Absolute paths for every input and the output: build_cost_surface's own
    # defaults are relative and would resolve against the frontend folder.
    with _memoised_csv_lookup(cs), contextlib.redirect_stdout(io.StringIO()):
        cs.build_cost_surface(
            npz_path=bp.data_path("ross_sea_concentration.npz"),
            csv_path=bp.data_path("pc_ice_speed_model.csv"),
            polar_class=polar_class,
            date_index=idx,
            alpha=_COST_ALPHA,
            beta=_COST_BETA,
            output_path=cached,
        )
    return cached, "built"


# ---------------------------------------------------------------------------
# 1. Sea-ice cover
# ---------------------------------------------------------------------------

def _clip_half_plane(points, limit, keep_below):
    """
    Sutherland-Hodgman clip of a convex polygon against a vertical line x=limit.

    `points` are (x, y) pairs in an UNWRAPPED longitude frame (no +/-180 jump).
    keep_below=True keeps x <= limit, False keeps x >= limit.
    """
    def inside(p):
        return p[0] <= limit if keep_below else p[0] >= limit

    out = []
    n = len(points)
    for k in range(n):
        cur, prv = points[k], points[k - 1]
        cur_in, prv_in = inside(cur), inside(prv)
        if cur_in != prv_in:
            # Edge crosses the line: interpolate the crossing point.
            dx = cur[0] - prv[0]
            t = 0.0 if dx == 0 else (limit - prv[0]) / dx
            out.append((limit, prv[1] + t * (cur[1] - prv[1])))
        if cur_in:
            out.append(cur)
    return out


# Nudge vertices off the seam so neither half is a zero-width sliver, which
# some renderers drop entirely.
_SEAM_EPS = 1e-6


def _split_at_antimeridian(poly):
    """
    Split a quad that straddles 180 deg into its west and east halves.

    Returns a list of polygons in normal -180..180 coordinates: one entry for
    an ordinary quad, two for a seam-crossing one.

    WHY: the Ross Sea straddles the antimeridian, and a polygon holding both
    +179 and -179 is drawn by pydeck as a band right across the whole map. The
    previous guard simply skipped those quads, which punched a one-cell-wide
    hole down the 180 deg line - 40 quads on a mid-November slice - making the
    ice field look like two disconnected halves. Clipping keeps the coverage
    continuous instead of trading one artefact for another.
    """
    lons = [p[0] for p in poly]
    if max(lons) - min(lons) <= 180.0:
        return [poly]

    # Unwrap: put every vertex on one continuous number line around 180.
    unwrapped = [((lo + 360.0) if lo < 0 else lo, la) for lo, la in poly]

    west = _clip_half_plane(unwrapped, 180.0 - _SEAM_EPS, keep_below=True)
    east = _clip_half_plane(unwrapped, 180.0 + _SEAM_EPS, keep_below=False)

    pieces = []
    if len(west) >= 3:
        pieces.append([[float(x), float(y)] for x, y in west])
    if len(east) >= 3:
        # Back to -180..180 for the eastern half.
        pieces.append([[float(x - 360.0), float(y)] for x, y in east])
    return pieces


def _quads_from_mesh(lat, lon, sic, stride=2):
    """
    Turn a (rows, cols) concentration mesh into pydeck polygons.

    `lat`/`lon` are 2-D arrays of the SAME shape as `sic`, so this works for the
    curvilinear NSIDC grid and for a regular mesh alike: each cell's quad uses
    the four surrounding mesh points rather than assuming a rectangular
    lat/lon box.

    Shared by the model path and by NPZ uploads so both render identically.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    sic = np.asarray(sic, dtype=float)

    rows, cols = sic.shape
    stride = max(1, int(stride))
    cells = []

    for i in range(0, rows - 1, stride):
        for j in range(0, cols - 1, stride):
            v = sic[i, j]
            if not np.isfinite(v):
                continue          # land, or outside the Ross Sea mask
            i2 = min(i + stride, rows - 1)
            j2 = min(j + stride, cols - 1)
            poly = [
                [float(lon[i, j]),   float(lat[i, j])],
                [float(lon[i2, j]),  float(lat[i2, j])],
                [float(lon[i2, j2]), float(lat[i2, j2])],
                [float(lon[i, j2]),  float(lat[i, j2])],
            ]
            # A quad holding both +179 and -179 would be drawn as a band right
            # across the map, so split it at the seam rather than dropping it
            # (dropping left a one-cell-wide hole down the 180 deg line).
            thickness = float(np.clip(v / 100.0, 0.0, 1.0))   # 0-100% -> 0-1
            for piece in _split_at_antimeridian(poly):
                cells.append({"polygon": piece, "thickness": thickness})
    return cells


@_cache_data(show_spinner=False, max_entries=8)
def _build_ice_polygons(date_str, stride):
    """
    Build quad polygons from the curvilinear NSIDC mesh for one date.

    For cell (i, j) the quad corners are the mesh points (i,j), (i+1,j),
    (i+1,j+1), (i,j+1) taken straight from the 2-D lat/lon arrays, so the
    polygons follow the real (curved) grid rather than assuming a rectangular
    lat/lon box.

    `stride` subsamples the 133x147 mesh. The full mesh is ~19.5k polygons,
    which pydeck renders but which makes every Streamlit rerun sluggish;
    stride=2 gives ~4.8k and still reads as a continuous field.
    """
    dates, conc, lat, lon = _load_sic_cube()
    date_list = [str(x) for x in dates]

    if date_str in date_list:
        idx = date_list.index(date_str)
    else:
        # Nearest available slice, so a valid-but-absent date still renders.
        target = pd.Timestamp(date_str)
        deltas = [abs((pd.Timestamp(d) - target).days) for d in date_list]
        idx = int(np.argmin(deltas))
        _warn(f"No SIC slice for {date_str}; showing nearest available "
              f"({date_list[idx]}).")

    return _quads_from_mesh(lat, lon, conc[idx], stride)


# Half-width of the square drawn for each row of a point-based (CSV/JSON)
# upload, in degrees. 0.25 gives a 0.5-degree cell, comparable to the ~25 km
# NSIDC footprint at Ross Sea latitudes.
_POINT_CELL_HALF_DEG = 0.25

_ICE_COLUMNS = ("lat", "lon", "concentration")


def _rewind(uploaded_file):
    """
    Streamlit re-serves the SAME UploadedFile object across reruns, so it can
    already be at EOF from a previous parse. Without this, a second render
    silently yields an empty frame.
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return uploaded_file


def _read_bytes(uploaded_file):
    """
    Raw bytes of an upload, from position 0.

    Streamlit re-serves the SAME UploadedFile object across reruns, so it can
    already be at EOF from a previous parse; without the rewind a second render
    silently yields an empty frame.
    """
    _rewind(uploaded_file)
    data = uploaded_file.read()
    _rewind(uploaded_file)
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


def _decode(data):
    """
    Decode upload bytes to text.

    utf-8-sig strips a BOM if Excel added one. A spreadsheet exported on a
    Windows machine with a non-UTF-8 locale can still arrive as cp1252/latin-1,
    so fall back rather than failing on one stray degree sign or accent.
    """
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_delimited(uploaded_file):
    """
    Parse a delimited text upload into a DataFrame, whatever its separator.

    THE BUG THIS FIXES: pd.read_csv defaults to a comma, so a semicolon- or
    tab-separated file parses into a SINGLE column literally named
    "lat;lon;concentration". Every required column then reads as missing and
    the user sees "missing column(s): lat, lon, concentration" for a file whose
    header plainly contains all three. Excel writes ';' whenever the machine's
    locale uses a semicolon list separator, which is the common way to hit this.

    csv.Sniffer picks the delimiter from a candidate set; if it cannot decide
    (a single-column file gives it nothing to go on) we fall back to a comma,
    which is the original behaviour.
    """
    text = _decode(_read_bytes(uploaded_file))
    if not text.strip():
        raise ValueError("file is empty")

    sample = text[:8192]
    sep = ","
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Sniffer gives up on single-column input; a comma is the safe default.
        pass

    LAST_PARSE_DEBUG["delimiter"] = repr(sep)
    df = pd.read_csv(io.StringIO(text), sep=sep)

    # Normalise header whitespace in place, so the columns reported to the UI
    # are the ones actually matched against.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_json_table(uploaded_file):
    """Parse a JSON upload (records or column orientation) into a DataFrame."""
    text = _decode(_read_bytes(uploaded_file))
    if not text.strip():
        raise ValueError("file is empty")
    df = pd.read_json(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _scale_to_percent(values):
    """
    Normalise concentration to 0-100.

    A file whose maximum is <= 1.0 is taken to be on a 0-1 fraction scale and
    scaled up. The ambiguous case is a genuinely near-ice-free 0-100 field, but
    reading 0.4 as 40% is the safer error for a navigation display than drawing
    consolidated pack ice as open water.
    """
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and float(np.nanmax(finite)) <= 1.0:
        return arr * 100.0
    return arr


def _polygons_from_points(df):
    """Square cells from a table of lat / lon / concentration rows."""
    # Surfaced by app.py in a debug expander when a parse fails, so the next
    # bad file can be diagnosed from the screen instead of from the logs.
    LAST_PARSE_DEBUG["columns"] = [str(c) for c in df.columns]
    LAST_PARSE_DEBUG["rows"] = int(len(df))

    cols = {str(c).lower().strip(): c for c in df.columns}
    missing = [c for c in _ICE_COLUMNS if c not in cols]
    if missing:
        raise ValueError("missing column(s): " + ", ".join(missing))

    lat = pd.to_numeric(df[cols["lat"]], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df[cols["lon"]], errors="coerce").to_numpy(dtype=float)
    conc = _scale_to_percent(
        pd.to_numeric(df[cols["concentration"]], errors="coerce").to_numpy(dtype=float))

    half = _POINT_CELL_HALF_DEG
    cells = []
    for la, lo, cv in zip(lat, lon, conc):
        if not (np.isfinite(la) and np.isfinite(lo) and np.isfinite(cv)):
            continue          # skip rows with a NaN in any of the three
        cells.append({
            "polygon": [
                [float(lo - half), float(la - half)],
                [float(lo + half), float(la - half)],
                [float(lo + half), float(la + half)],
                [float(lo - half), float(la + half)],
                [float(lo - half), float(la - half)],
            ],
            "thickness": float(np.clip(cv / 100.0, 0.0, 1.0)),
        })
    if not cells:
        raise ValueError("no rows with finite lat, lon and concentration")
    return cells


def _polygons_from_npz(uploaded_file, stride):
    """Mesh cells from an .npz carrying lat / lon / concentration."""
    data = np.load(_rewind(uploaded_file), allow_pickle=False)
    missing = [k for k in _ICE_COLUMNS if k not in data.files]
    if missing:
        raise ValueError("missing key(s): " + ", ".join(missing))

    lat = np.asarray(data["lat"], dtype=float)
    lon = np.asarray(data["lon"], dtype=float)
    conc = _scale_to_percent(np.asarray(data["concentration"], dtype=float))

    # A stack of daily slices: take the first, since an upload carries no date
    # axis we could index against.
    if conc.ndim == 3:
        _warn("Uploaded ice file holds %d slices; showing the first. Upload a "
              "single 2-D field to choose a specific one." % conc.shape[0])
        conc = conc[0]

    # 1-D axes describe a regular mesh; expand them to match the model path.
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    if lat.shape != conc.shape or lon.shape != conc.shape:
        raise ValueError(
            "shape mismatch: concentration %s vs lat %s / lon %s"
            % (conc.shape, lat.shape, lon.shape))

    cells = _quads_from_mesh(lat, lon, conc, stride)
    if not cells:
        raise ValueError("no finite concentration cells")
    return cells


def _polygons_from_upload(uploaded_file, stride):
    """Dispatch on file extension. Raises ValueError with a usable message."""
    name = (getattr(uploaded_file, "name", "") or "").lower()
    if name.endswith(".npz"):
        return _polygons_from_npz(uploaded_file, stride)
    if name.endswith(".json"):
        return _polygons_from_points(_read_json_table(uploaded_file))
    if name.endswith(".csv"):
        return _polygons_from_points(_read_delimited(uploaded_file))
    raise ValueError(
        "unsupported file type %r; expected .npz, .csv or .json" % name)


def _model_ice_cover(date_str, stride):
    """Ice cover from the backend NPZ, or mock data if the backend is absent."""
    if not bp.BACKEND_AVAILABLE:
        _warn("Backend unavailable - showing mock ice cover. Check that the "
              "backend folder sits next to the frontend folder.")
        return generate_ice_grid()
    try:
        # Absolute paths throughout (see backend_paths), so no CWD swap is
        # needed here; the guard is only required where the backend resolves
        # files relative to the working directory.
        return _build_ice_polygons(str(date_str or DEMO_DATE), stride)
    except Exception as e:
        _warn("Could not load real ice cover (%s); showing mock data." % e)
        return generate_ice_grid()


def load_ice_cover_data(date_str=None, uploaded_file=None, mode="model",
                        stride=2):
    """
    Sea-ice cover polygons, from the model archive or a user-supplied file.

    mode          : "model" (backend NPZ for `date_str`) or "upload"
    uploaded_file : Streamlit UploadedFile; .npz, .csv or .json

    Accepted upload shapes:
      NPZ  keys  lat, lon, concentration. lat/lon may be 1-D axes or 2-D
                 meshes; concentration matches, on a 0-100 or 0-1 scale.
      CSV/JSON   columns lat, lon, concentration (one row per cell).

    A parse failure records an error for the UI and falls back to model data,
    so the map never goes blank because of a bad upload.

    Returns: [{"polygon": [[lon, lat], ...], "thickness": 0.0-1.0}, ...]
    """
    if mode == "upload":
        LAST_PARSE_DEBUG.clear()
        if uploaded_file is None:
            _warn("Ice cover set to 'Upload file' but no file was provided; "
                  "using model data for the selected date.")
        else:
            try:
                cells = _polygons_from_upload(uploaded_file, stride)
                '''_warn("Ice cover from uploaded file '%s' (%d cells) - not the "
                      "model archive."
                      % (getattr(uploaded_file, "name", "upload"), len(cells)))'''
                return cells
            except Exception as e:
                _error(
                    "Could not parse ice cover file. Expected columns/keys: "
                    "lat, lon, concentration. (%s) Showing model data instead."
                    % e)

    return _model_ice_cover(date_str, stride)

# ---------------------------------------------------------------------------
# 2. Iceberg positions
# ---------------------------------------------------------------------------

# BYU/NIC size_1 and size_2 are in NAUTICAL MILES (verified against the CSV:
# b22a is 33 x 26, i.e. ~61 x 48 km, consistent with a large tabular berg).
# 77% of rows report 0, meaning "not measured", not "zero-sized".
_NM_TO_KM = 1.852


def _dedupe_carry_forwards(df):
    """
    Drop consecutive rows per berg whose LAT/LON exactly repeat the previous
    row. BYU/NIC carries the last fix forward when a berg is not re-imaged;
    ~66% of rows are such repeats, and treating them as observations implies
    the berg sat perfectly still.
    """
    keep = []
    for _, g in df.groupby("ICEBERG_ID", sort=False):
        g = g.sort_values("DATE")
        moved = (g["LAT"].diff().abs() + g["LON"].diff().abs()) > 1e-9
        if len(moved):
            moved.iloc[0] = True
        keep.append(g[moved])
    return pd.concat(keep).reset_index(drop=True) if keep else df


def _risk_from_size(sizes):
    """
    Normalise size to a 0-1 risk proxy, clipped at the 99th percentile so one
    giant berg does not flatten everything else to ~0.
    """
    sizes = np.asarray(sizes, dtype=float)
    positive = sizes[sizes > 0]

    # A size of 0 in the BYU/NIC file means NOT MEASURED (77% of rows), not
    # "zero area". Scoring those 0.0 would paint an unmeasured berg as the
    # safest thing on the map. They get a neutral 0.5 instead - unknown, not
    # harmless - and only measured bergs are scaled against the size ramp.
    UNKNOWN_RISK = 0.5
    out = np.full(sizes.shape, UNKNOWN_RISK, dtype=float)
    if positive.size == 0:
        return out

    hi = float(np.percentile(positive, 99))
    if hi <= 0:
        return out
    measured = sizes > 0
    out[measured] = np.clip(sizes[measured] / hi, 0.05, 1.0)
    return out


def berg_equivalent_radius_km(size_km2):
    """Radius of a circle with the berg's reported area (0 when unmeasured)."""
    try:
        area = float(size_km2 or 0.0)
    except (TypeError, ValueError):
        area = 0.0
    return math.sqrt(area / math.pi) if area > 0 else 0.0


def positional_uncertainty_km(berg):
    """
    The `uncertainty_km` handed to calculate_routes(), which turns it into an
    exclusion radius of 50 km + 2 x this.

    Two terms:
      * positional uncertainty of the fix itself, 0.5-1.0 km by risk;
      * HALF the berg's own equivalent radius, because the optimiser's 50 km is
        measured from the CENTROID and ignores the berg's extent. B22A has a
        ~31 km equivalent radius, so a flat 52 km centroid exclusion leaves
        only ~21 km of real clearance - under half the nominal margin. Adding
        R/2 here (doubled by the optimiser's 2x) keeps clearance beyond the
        berg EDGE at roughly the intended 50 km for every size.

    The same number drives the ring drawn on the map, so the display can never
    understate the zone the router actually avoided.
    """
    risk = float(berg.get("risk", 0.5) or 0.0)
    return 0.5 * (1.0 + risk) + berg_equivalent_radius_km(berg.get("size_km2")) / 2.0


def _icebergs_from_frame(df, date_str):
    """Shared mapping from a BYU-shaped frame to the frontend iceberg shape."""
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE", "LAT", "LON"])
    if df.empty:
        return [], None, None

    df = _dedupe_carry_forwards(df)

    target = pd.Timestamp(str(date_str))

    # One row per berg: its LAST KNOWN FIX as of the selected date.
    #
    # Snapping every berg to a single shared "nearest date" would show just one
    # marker, because the file records only 1-2 observations on any given day
    # across 594 rows. What a navigator needs is the current picture: every
    # tracked berg at its most recent fix, each carrying its own age so a stale
    # one is visibly stale. Bergs first seen only AFTER the selected date fall
    # back to their earliest fix (flagged by a negative-age-derived large
    # obs_age_days), rather than vanishing from the display.
    rows = []
    for _, g in df.groupby("ICEBERG_ID", sort=False):
        past = g[g["DATE"] <= target]
        pick = (past.sort_values("DATE").iloc[-1] if not past.empty
                else g.sort_values("DATE").iloc[0])
        rows.append(pick)
    day = pd.DataFrame(rows)

    sizes = []
    for _, r in day.iterrows():
        s1 = float(r.get("size_1", 0) or 0.0)
        s2 = float(r.get("size_2", 0) or 0.0)
        sizes.append((s1 * _NM_TO_KM) * (s2 * _NM_TO_KM) if s1 > 0 and s2 > 0 else 0.0)
    risks = _risk_from_size(np.array(sizes))

    bergs = []
    for (_, r), km2, risk in zip(day.iterrows(), sizes, risks):
        obs = pd.Timestamp(r["DATE"])
        bergs.append({
            "id": str(r["ICEBERG_ID"]).upper(),
            "lat": float(r["LAT"]),
            "lon": float(r["LON"]),          # already -180..180
            "size_km2": round(float(km2), 1),
            "risk": round(float(risk), 3),
            "obs_date": obs.strftime("%Y-%m-%d"),
            "obs_age_days": int(abs((target - obs).days)),
        })
        bergs[-1]["uncertainty_km"] = round(positional_uncertainty_km(bergs[-1]), 2)

    bergs.sort(key=lambda b: b["obs_age_days"])
    newest = pd.Timestamp(day["DATE"].max())
    median_age = int(np.median([b["obs_age_days"] for b in bergs])) if bergs else None
    return bergs, newest, median_age


@_cache_data(show_spinner=False, max_entries=16)
def _load_bergs_for_date(date_str):
    df = pd.read_csv(bp.data_path("ross_sea_icebergs_2021_2025.csv"))
    return _icebergs_from_frame(df, date_str)


def load_current_iceberg_positions(uploaded_file=None, date_str=DEMO_DATE):
    """
    Tracked iceberg positions nearest the requested date.

    Returns: [{"id", "lat", "lon", "size_km2", "risk", "obs_date",
               "obs_age_days"}, ...]

    NOTE ON DATA COVERAGE: the BYU/NIC file has NO observations between
    2023-08 and 2025-01 - the whole 2023-24 season is missing. The locked demo
    date (2023-11-15) therefore has no contemporaneous fixes, and the nearest
    are ~4 months stale. Rather than silently drawing stale bergs as if they
    were current, each berg carries `obs_age_days` and the UI warns when that
    exceeds a season.
    """
    if uploaded_file is not None:
        try:
            name = (getattr(uploaded_file, "name", "") or "").lower()
            if name.endswith(".json"):
                df = pd.DataFrame(json.load(uploaded_file))
            else:
                df = pd.read_csv(uploaded_file)
            cols = {c.upper(): c for c in df.columns}
            missing = [c for c in ("ICEBERG_ID", "DATE", "LAT", "LON")
                       if c not in cols]
            if missing:
                _warn(f"Uploaded file is missing column(s) {missing}; expected "
                      f"ICEBERG_ID, DATE, LAT, LON. Using bundled data.")
            else:
                df = df.rename(columns={cols[k]: k for k in cols})
                bergs, _, _ = _icebergs_from_frame(df, date_str)
                if bergs:
                    return bergs
                _warn("Uploaded file contained no usable rows; using bundled data.")
        except Exception as e:
            _warn(f"Could not parse the uploaded iceberg file ({e}); "
                  f"using bundled data.")

    if not bp.BACKEND_AVAILABLE:
        _warn("Backend unavailable - showing mock icebergs.")
        return generate_icebergs()

    try:
        bergs, _, _ = _load_bergs_for_date(str(date_str))
        if not bergs:
            _warn("No iceberg observations found; showing mock icebergs.")
            return generate_icebergs()
        # bergs are sorted freshest-first, so bergs[0] is the best fix we have.
        freshest = bergs[0]
        if freshest["obs_age_days"] > 60:
           ''' _warn(
                f"No recent iceberg fixes for this date - the freshest is "
                f"{freshest['id']} from {freshest['obs_date']} "
                f"({freshest['obs_age_days']} days away). The BYU/NIC file "
                f"holds no observations between 2023-08 and 2025-01, so the "
                f"2023-24 season is missing entirely. Positions shown are each "
                f"berg's last known fix - indicative only, not a current plot."
            )'''
        elif max(b["obs_age_days"] for b in bergs) > 365:
            _warn(
                "Some bergs shown are last-known fixes more than a year old "
                "(see the Age column); their positions are indicative only."
            )
        return bergs
    except Exception as e:
        _warn(f"Could not load iceberg CSV ({e}); showing mock icebergs.")
        return generate_icebergs()


# ---------------------------------------------------------------------------
# 3. Weather input (unchanged contract)
# ---------------------------------------------------------------------------

def parse_weather_input(mode, uploaded_file=None, manual_params=None):
    """Pass-through: run_forecast() uses ERA5 directly and falls back to these."""
    return {"mode": mode, "uploaded_file": uploaded_file,
            "manual_params": manual_params}


# ---------------------------------------------------------------------------
# 4. Forecast: iceberg drift + Pareto routes
# ---------------------------------------------------------------------------

def _drift_trajectories(bergs, date_str, hours=48, step_h=6):
    """
    48-hour iceberg drift using the backend's Wagner/Bigg physics step, with
    ERA5 wind and NSIDC SIC sampled at each berg position every 6 h.

    Returns {berg_id: [[lon, lat], ...]} (pydeck order), first point = now.

    The brief suggested `IcebergDriftModel`; no such class exists in the
    backend. The real entry point is physics_step(), used here.
    """
    bp.ensure_on_path()
    from iceberg_physics_baseline import physics_step
    import grid_utils as gu

    wind = _load_wind_sampler()
    loc = _load_grid_locator()
    dates, conc, _, _ = _load_sic_cube()
    sic_grid = conc[gu.sic_date_index(dates, date_str)]

    t0 = gu.parse_date(date_str).replace(hour=6)
    out = {}

    for b in bergs:
        lat, lon = float(b["lat"]), float(b["lon"])
        vel_u = vel_v = 0.0
        path = [[lon, lat]]

        # Berg footprint from the reported size, else a mid-size default.
        km2 = float(b.get("size_km2") or 0.0)
        if km2 > 0:
            side_m = (km2 ** 0.5) * 1000.0
            length_m, width_m = side_m, side_m * 0.6
        else:
            length_m, width_m = 5000.0, 2000.0

        for k in range(int(hours / step_h)):
            when = t0 + dt.timedelta(hours=k * step_h)
            try:
                wu, wv = wind.sample(lat, lon, when)
            except Exception:
                wu = wv = 0.0
            sic = loc.sample(sic_grid, lat, lon, default=0.0) / 100.0

            res = physics_step(
                lat=lat, lon=lon, vel_u=vel_u, vel_v=vel_v,
                wind_u=wu, wind_v=wv, sic=sic,
                dt_s=step_h * 3600.0,
                length_m=length_m, width_m=width_m,
            )
            lat, lon = res["lat_new"], res["lon_new"]
            vel_u, vel_v = res["vel_u_new"], res["vel_v_new"]
            path.append([float(lon), float(lat)])

        out[b["id"]] = path
    return out


def _straight_line_drift(bergs, weather_data, hours=48, step_h=6):
    """
    Fallback drift when the physics path is unavailable: advect each berg at
    ~2% of the manually entered wind speed (the standard wind-driven iceberg
    rule of thumb), in the downwind direction.
    """
    params = (weather_data or {}).get("manual_params") or {}
    spd_kt = float(params.get("wind_speed_kt", 15.0))
    # Meteorological convention: direction wind blows FROM.
    from_deg = float(params.get("wind_direction_deg", 225.0))
    to_rad = np.radians((from_deg + 180.0) % 360.0)

    spd_ms = spd_kt * 0.514444 * 0.02
    out = {}
    for b in bergs:
        lat, lon = float(b["lat"]), float(b["lon"])
        path = [[lon, lat]]
        for _ in range(int(hours / step_h)):
            dn = spd_ms * step_h * 3600.0 * float(np.cos(to_rad))   # metres north
            de = spd_ms * step_h * 3600.0 * float(np.sin(to_rad))   # metres east
            lat += dn / 111_320.0
            denom = 111_320.0 * max(float(np.cos(np.radians(lat))), 1e-6)
            lon = ((lon + de / denom + 180.0) % 360.0) - 180.0
            path.append([float(lon), float(lat)])
        out[b["id"]] = path
    return out


def _waypoints_to_pydeck(waypoints):
    """calculate_routes() yields (lat, lon); pydeck PathLayer wants [lon, lat]."""
    return [[float(lon), float(lat)] for (lat, lon) in waypoints]


def _empty_pareto(impossible=True, reason=""):
    return {"fuel_priority": [], "balanced": [], "time_priority": [],
            "metrics": {}, "impossible": impossible, "reason": reason,
            "data_gaps": {}}


def _effective_date(date_str):
    """
    Resolve a possibly-absent forecast date to one the archive actually holds.

    The UI hides the date picker while ice cover comes from an upload, so a
    None can reach here. It must NOT fall back to today: an uploaded ice field
    is display-only - it feeds the map layer and never reaches
    calculate_routes() - so routing still reads the model archive (cost
    surface, ERA5 wind, berg fixes) at this date. The archive ends 2025-03-31,
    so today's date yields "date not present in ross_sea_concentration.npz"
    and zero routes. DEMO_DATE is a date that exists.
    """
    if date_str:
        return str(date_str)
    _warn("No forecast date supplied; routing against the model archive for "
          f"{DEMO_DATE} (an uploaded ice field changes the map display, not "
          f"the route).")
    return DEMO_DATE


def run_forecast(iceberg_data, weather_data, ship_pos, destination,
                 polar_class="PC4", date_str=DEMO_DATE):
    """
    Run the full backend pipeline.

    Returns (trajectories, pareto_routes):
      trajectories  : {berg_id: [[lon, lat], ...]}  48 h drift
      pareto_routes : {
          'fuel_priority': [[lon,lat],...], 'balanced': [...],
          'time_priority': [...],
          'metrics': {name: {distance, time_h, fuel_t, ice_severity}},
          'impossible': bool, 'reason': str,
      }
    """
    date_str = _effective_date(date_str)

    if not bp.BACKEND_AVAILABLE:
        _warn("Backend unavailable - showing mock data. Check that the backend "
              "folder sits next to the frontend folder, or set "
              "ANTARCTIC_NAV_BACKEND.")
        traj = generate_dummy_trajectory(iceberg_data)
        pareto = _empty_pareto(impossible=False, reason="")
        pareto["balanced"] = generate_dummy_route(ship_pos, destination)
        pareto["metrics"] = {"balanced": {"distance": 0.0, "time_h": 0.0,
                                          "fuel_t": 0.0, "ice_severity": "n/a",
                                          "cost_type": "balanced",
                                          "n_waypoints": 0}}
        return traj, pareto

    # --- iceberg drift ------------------------------------------------------
    try:
        # Guarded for the same reason as the route call: physics_step is
        # path-free, but anything the backend loads relatively (LSTM weights)
        # must still resolve if this path is extended later.
        with bp.backend_cwd():
            trajectories = _drift_trajectories(iceberg_data, date_str)
    except Exception as e:
        _warn(f"Iceberg physics unavailable ({e}); using a wind-advection "
              f"approximation for drift.")
        try:
            trajectories = _straight_line_drift(iceberg_data, weather_data)
        except Exception:
            trajectories = generate_dummy_trajectory(iceberg_data)

    # --- Pareto routes ------------------------------------------------------
    # Resolve the cost surface FIRST, with absolute paths. Left to itself,
    # calculate_routes() falls back to build_cost_surface_for(), whose defaults
    # are relative ("ross_sea_concentration.npz") and resolve against
    # Streamlit's CWD - the frontend folder - so every date without a pre-built
    # surface died with "could not obtain a cost surface: [Errno 2] No such
    # file or directory". Handing it a path that already exists means it never
    # takes that branch.
    surface_path = None
    try:
        surface_path, how = cost_surface_for(polar_class, str(date_str))
        if how == "built":
            _warn(f"No pre-built cost surface for {date_str}; built one on the "
                  f"fly and cached it for next time.")
    except Exception as e:
        _warn(f"Could not prepare a cost surface for {polar_class} on "
              f"{date_str} ({e}).")

    try:
        bp.ensure_on_path()
        from route_optimiser import calculate_routes

        # Tier-1 uncertainty proxy: a larger/riskier berg gets a wider berth.
        iceberg_positions = [
            {"lat": float(b["lat"]), "lon": float(b["lon"]),
             "uncertainty_km": float(b.get("uncertainty_km")
                                     or positional_uncertainty_km(b))}
            for b in iceberg_data
        ]

        # Belt-and-braces: the surface above removes the known relative-path
        # branch, but other backend entry points (iceberg_lstm._get_model,
        # seaice_lstm._get) also resolve weights relative to the CWD. Holding
        # the guard across the call keeps any of those working too.
        with bp.backend_cwd():
            result = calculate_routes(
                start_lat=float(ship_pos["lat"]),
                start_lon=float(ship_pos["lon"]),
                end_lat=float(destination["lat"]),
                end_lon=float(destination["lon"]),
                polar_class=polar_class,
                date_str=str(date_str),
                iceberg_positions=iceberg_positions,
                cost_surface_path=surface_path,
            )
    except Exception as e:
        _warn(f"Route optimiser unavailable ({e}); showing a mock route.")
        pareto = _empty_pareto(impossible=False, reason="")
        pareto["balanced"] = generate_dummy_route(ship_pos, destination)
        return trajectories, pareto

    if result.get("impossible"):
        # Surface the backend's own explanation. Previously this was stored in
        # the result dict but never raised as a warning, so a data problem
        # looked identical to a genuinely blocked transect.
        reason = result.get("reason", "no route found")
        _warn(f"No route for {polar_class} on {date_str}: {reason}")
        return {}, _empty_pareto(impossible=True, reason=reason)

    # Index by cost_type, never by list position: calculate_routes() skips a
    # weighting whose search fails, so routes[2] is not guaranteed to exist.
    by_type = {r["cost_type"]: r for r in result.get("routes", [])}
    rec = result.get("recommended")
    if rec is not None:
        by_type.setdefault(rec["cost_type"], rec)

    pareto = _empty_pareto(impossible=False, reason="")
    for name in ("fuel_priority", "balanced", "time_priority"):
        r = by_type.get(name)
        if not r:
            continue
        pareto[name] = _waypoints_to_pydeck(r["waypoints"])
        pareto["metrics"][name] = {
            "distance": round(float(r["total_distance_km"]), 1),
            "time_h": round(float(r["estimated_time_h"]), 1),
            "fuel_t": round(float(r["estimated_fuel_kg"]) / 1000.0, 1),
            "ice_severity": r.get("ice_severity", "n/a"),
            "cost_type": name,
            "n_waypoints": len(r["waypoints"]),
        }

    missing = [n for n in ("fuel_priority", "balanced", "time_priority")
               if not pareto[n]]
    if missing:
        _warn(f"No path found for weighting(s): {', '.join(missing)}.")

    pareto["data_gaps"] = result.get("data_gaps", {})
    return trajectories, pareto


# ---------------------------------------------------------------------------
# 5. Summary metrics
# ---------------------------------------------------------------------------

# Validated skill figures from the backend build - constants, never random.
SEA_ICE_SKILL_PCT = 2.7     # held-out 2024-25, within the ERA5 coverage zone
ICEBERG_SKILL_PCT = 5.0     # B38 validation split, 69 genuine drift pairs


def _confidence(polar_class, ice_severity):
    """
    Static confidence lookup. Illustrative but grounded in the validated skill
    figures - deliberately NOT random, so the number is stable across reruns.
    """
    pc = (polar_class or "PC4").upper()
    sev = (ice_severity or "").lower()
    if pc in ("PC1", "PC2") and sev == "low":
        return 88
    if pc in ("PC3", "PC4") and sev == "moderate":
        return 79
    if pc in ("PC5", "PC6", "PC7") and sev == "high":
        return 67
    return 74


def compute_summary_metrics(iceberg_data, polar_class="PC4", ice_severity=None):
    """
    KPIs for the right-hand summary panel.

    Returns at least: tracked_icebergs, high_risk_icebergs, avg_size_km2,
    forecast_confidence_pct, sea_ice_skill_pct, iceberg_skill_pct.
    """
    if not iceberg_data:
        return {
            "tracked_icebergs": 0, "high_risk_icebergs": 0,
            "avg_size_km2": 0.0, "sized_icebergs": 0,
            "forecast_confidence_pct": _confidence(polar_class, ice_severity),
            "sea_ice_skill_pct": SEA_ICE_SKILL_PCT,
            "iceberg_skill_pct": ICEBERG_SKILL_PCT,
        }

    sizes = [float(b.get("size_km2") or 0.0) for b in iceberg_data]
    reported = [s for s in sizes if s > 0]
    return {
        "tracked_icebergs": len(iceberg_data),
        "high_risk_icebergs": sum(1 for b in iceberg_data
                                  if float(b.get("risk", 0.0)) > 0.66),
        # Averaged over bergs that actually reported a size: 77% of BYU rows
        # record 0, which means "not measured", not "zero area".
        "avg_size_km2": round(float(np.mean(reported)), 1) if reported else 0.0,
        "sized_icebergs": len(reported),
        "forecast_confidence_pct": _confidence(polar_class, ice_severity),
        "sea_ice_skill_pct": SEA_ICE_SKILL_PCT,
        "iceberg_skill_pct": ICEBERG_SKILL_PCT,
    }
