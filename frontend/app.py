"""
app.py
======
Streamlit frontend ONLY - no backend logic and no data loading live in this
file. Everything that needs real data calls a function in backend_stubs.py,
which owns the seam to the locked backend package.

Run with:
    streamlit run app.py
"""

import datetime as dt

import streamlit as st
import pydeck as pdk

from config import (
    APP_TITLE,
    DEFAULT_VIEW_STATE,
    POLAR_CLASSES,
    DEFAULT_POLAR_CLASS,
    SHIPPING_MONTHS,
    SEASON_YEAR_MIN,
    SEASON_YEAR_MAX,
    DEMO_DATE,
    DEMO_START,
    DEMO_END,
)
import backend_stubs as backend
from map_layers import (
    build_ice_layer,
    build_iceberg_layer,
    build_trajectory_layer,
    build_trajectory_head_layer,
    build_pareto_route_layers,
    build_iceberg_uncertainty_layer,
    build_ship_markers_layer,
    build_graticule_layer,
)
from route_cards import render_route_cards, render_impossible_banner
from uncertainty_panel import render_accuracy_panel, render_disclaimer

# ---------------------------------------------------------------------------
# Page config + dark "common operating picture" theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        .stApp { background-color: #12161d; color: #e6e9ef; }
        section[data-testid="stSidebar"] { display: none; }

        .top-bar {
            display: flex; align-items: center; justify-content: space-between;
            background-color: #1a2029; padding: 0.75rem 1.25rem; border-radius: 6px;
            margin-bottom: 0.75rem; border: 1px solid #262d3a;
        }
        .top-bar h1 { font-size: 1.05rem; margin: 0; color: #f2f4f7; font-weight: 600; }
        .top-bar .nav-item { color: #9fb3c8; font-size: 0.85rem; margin-left: 1.25rem; }

        .panel-card {
            background-color: #1a2029; border: 1px solid #262d3a; border-radius: 8px;
            padding: 0.9rem 1rem; margin-bottom: 0.75rem;
        }
        .panel-card h4 { margin: 0 0 0.6rem 0; font-size: 0.8rem; letter-spacing: 0.04em;
            color: #8fa2b8; text-transform: uppercase; }

        .metric-value { font-size: 1.6rem; font-weight: 700; color: #f2f4f7; line-height: 1.2; }
        .metric-label { font-size: 0.78rem; color: #8fa2b8; margin-bottom: 0.5rem; }
        .metric-pct-warn { color: #e8845c; font-weight: 700; }
        .metric-pct-ok { color: #56c288; font-weight: 700; }

        div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #262d3a !important; }
        button[kind="primary"] { background-color: #2f6feb !important; border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Date-window validation
# ---------------------------------------------------------------------------
def validate_forecast_date(d):
    """
    The backend is trained and validated on the Nov-Mar shipping season only,
    and the SIC cube physically contains no slice outside it.

    Returns (ok, message). Streamlit's date_input has no month whitelist, so
    the check has to happen after selection.
    """
    if d.month not in SHIPPING_MONTHS:
        return False, "Select a date in the Nov-Mar shipping season, 2021-2025."
    if not (SEASON_YEAR_MIN <= d.year <= SEASON_YEAR_MAX):
        return False, "Select a date in the Nov-Mar shipping season, 2021-2025."

    # Stronger check: the date must actually exist in the SIC cube. The window
    # is not fully continuous, so month/year alone does not guarantee data.
    available = backend.available_sic_dates()
    if available and d.strftime("%Y%m%d") not in available:
        return False, (
            f"No sea-ice data for {d:%Y-%m-%d}. The archive covers the four "
            f"Nov-Mar seasons from {available[0][:4]}-{available[0][4:6]} to "
            f"{available[-1][:4]}-{available[-1][4:6]}."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_demo_date = dt.date(int(DEMO_DATE[:4]), int(DEMO_DATE[4:6]), int(DEMO_DATE[6:]))

_DEFAULTS = {
    "forecast_date": _demo_date,
    "polar_class": DEFAULT_POLAR_CLASS,
    "ice_date_loaded": None,
    "ice_input_mode": "model",
    "ice_upload_file": None,
    "pending_errors": [],
    "pending_parse_debug": {},
    "ice_cells": [],
    "icebergs": [],
    "trajectories": {},
    "pareto_routes": {},
    "show_ice_layer": True,
    "show_iceberg_layer": True,
    "show_graticule": True,
    "show_forecast_layers": False,
    "show_uncertainty_rings": True,
    "pending_warnings": [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Kick off the ~8.5 s ERA5 load in the background on first render. The user
# spends several seconds choosing a date and endpoints, which absorbs it, so
# the first forecast does not stall on a cold read.
if "_prewarmed" not in st.session_state:
    backend.prewarm()
    st.session_state["_prewarmed"] = True

date_key = st.session_state.forecast_date.strftime("%Y%m%d")

# Reload the date-dependent layers when the date changes, and also when the ice
# source changes - keying the reload on the date alone would leave a stale
# model field on screen after switching to an upload (or back).
# Read the ice source from the WIDGET keys, not from the mirrored values the
# card writes. The card renders below the map, so on the rerun that follows a
# radio change the mirrored value is still last run's - the map would lag one
# interaction behind. Streamlit populates widget keys before the script runs,
# so these are already current here.
_ice_mode = ("upload"
             if st.session_state.get("ice_mode_radio") == "Upload file"
             else "model")
_ice_file = st.session_state.get("ice_upload") if _ice_mode == "upload" else None

# Keep the documented session_state names in step with the widgets.
st.session_state.ice_input_mode = _ice_mode
st.session_state.ice_upload_file = _ice_file

_ice_sig = (date_key, _ice_mode,
            getattr(_ice_file, "name", None),
            getattr(_ice_file, "size", None))

if st.session_state.ice_date_loaded != _ice_sig:
    st.session_state.ice_cells = backend.load_ice_cover_data(
        # The date is irrelevant to an uploaded field, so don't pretend it
        # applies; load_ice_cover_data() handles None.
        date_str=None if _ice_mode == "upload" else date_key,
        uploaded_file=_ice_file,
        mode=_ice_mode,
    )
    st.session_state.icebergs = backend.load_current_iceberg_positions(
        date_str=date_key)
    st.session_state.ice_date_loaded = _ice_sig
    # A date or source change invalidates any previously computed forecast.
    st.session_state.trajectories = {}
    st.session_state.pareto_routes = {}
    st.session_state.pending_warnings = backend.drain_warnings()
    st.session_state.pending_errors = backend.drain_errors()
    st.session_state.pending_parse_debug = dict(backend.LAST_PARSE_DEBUG)

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="top-bar">
        <h1>&#129482; {APP_TITLE}</h1>
        <div>
            <span class="nav-item">&#128200; KPI Tracking</span>
            <span class="nav-item">&#128197; Work Planning</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Layout: [layer rail] [map + legend] [input / summary panel]
# ---------------------------------------------------------------------------
rail_col, map_col, panel_col = st.columns([0.6, 3.4, 1.6], gap="medium")

# A completed forecast should switch its own layers on. The flag is consumed
# HERE, before the checkbox widget is created: a keyed widget reads
# st.session_state[key] only at instantiation, and Streamlit raises if you
# assign to that key after the widget exists. Writing the checkbox value from
# the post-run block (as the original code did) therefore had no effect at all
# - the widget kept its own stored False and the routes stayed hidden.
if st.session_state.pop("_enable_forecast_layers", False):
    st.session_state["show_forecast_layers"] = True

# --- Left rail: layer visibility toggles -----------------------------------
with rail_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Layers</h4>", unsafe_allow_html=True)
    st.checkbox("Ice cover", key="show_ice_layer")
    st.checkbox("Icebergs", key="show_iceberg_layer")
    st.checkbox(
        "Uncertainty", key="show_uncertainty_rings",
        help="Hazard exclusion footprint the optimiser routes around "
             "(50 km + 2 sigma per berg)",
    )
    st.checkbox("Grid", key="show_graticule")
    st.checkbox(
        "Forecast", key="show_forecast_layers",
        help="Iceberg drift + the three Pareto routes (after a forecast run)",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- Center: legend + map ---------------------------------------------------
with map_col:
    with st.container(border=True):
        leg1, leg2, leg3 = st.columns([1, 1, 1.4])
        with leg1:
            st.caption("Sea-ice concentration")
            st.markdown(
                "<div style='background:linear-gradient(90deg,#fffaf5,#b21818);"
                "height:10px;border-radius:4px;'></div><div style='display:flex;"
                "justify-content:space-between;font-size:0.7rem;color:#8fa2b8;'>"
                "<span>0%</span><span>100%</span></div>",
                unsafe_allow_html=True,
            )
        with leg2:
            st.caption("Iceberg risk")
            st.markdown(
                "<div style='background:linear-gradient(90deg,#28a85a,#d62728);"
                "height:10px;border-radius:4px;'></div><div style='display:flex;"
                "justify-content:space-between;font-size:0.7rem;color:#8fa2b8;'>"
                "<span>low</span><span>high</span></div>",
                unsafe_allow_html=True,
            )
        with leg3:
            st.caption("Pareto routes")
            st.markdown(
                "<div style='display:flex;gap:0.75rem;font-size:0.7rem;"
                "color:#b8c6d6;align-items:center;'>"
                "<span><span style='display:inline-block;width:14px;height:3px;"
                "background:#e6783c;vertical-align:middle;'></span> fuel</span>"
                "<span><span style='display:inline-block;width:14px;height:3px;"
                "background:#f5b400;vertical-align:middle;'></span> balanced</span>"
                "<span><span style='display:inline-block;width:14px;height:3px;"
                "background:#3cbec8;vertical-align:middle;'></span> time</span>"
                "</div>",
                unsafe_allow_html=True,
            )

    layers = [build_graticule_layer(visible=st.session_state.show_graticule)]
    layers.append(build_ice_layer(st.session_state.ice_cells,
                                  visible=st.session_state.show_ice_layer))

    # Rings first so the berg markers draw on top of their own footprints.
    ring_layer = build_iceberg_uncertainty_layer(
        st.session_state.icebergs,
        visible=st.session_state.show_uncertainty_rings)
    if ring_layer:
        layers.append(ring_layer)

    layers.append(build_iceberg_layer(st.session_state.icebergs,
                                      visible=st.session_state.show_iceberg_layer))

    if st.session_state.trajectories:
        layers.append(build_trajectory_layer(
            st.session_state.trajectories,
            visible=st.session_state.show_forecast_layers))
        # Head dot at +48 h, so the direction of drift is readable.
        head = build_trajectory_head_layer(
            st.session_state.trajectories,
            visible=st.session_state.show_forecast_layers)
        if head:
            layers.append(head)

    # All three Pareto routes, not just one.
    layers.extend(build_pareto_route_layers(
        st.session_state.pareto_routes,
        visible=st.session_state.show_forecast_layers))

    markers_layer = build_ship_markers_layer(
        st.session_state.get("ship_pos_last"),
        st.session_state.get("destination_last"),
        visible=st.session_state.show_forecast_layers,
    )
    if markers_layer:
        layers.append(markers_layer)

    deck = pdk.Deck(
        map_provider="carto",
        map_style="dark",
        initial_view_state=pdk.ViewState(**DEFAULT_VIEW_STATE),
        layers=layers,
        # One always-present key. A combined template naming {id}, {size_km2},
        # {thickness} etc. leaves LITERAL braces on any layer whose row dict
        # lacks that key - which is why the tooltip rendered as
        # "{id}{label} size: {size_km2} km2". Each pickable layer now renders
        # its own `tooltip` string (see map_layers._tip).
        tooltip={
            "html": "<div style='background:#1a2029;color:#e6e9ef;padding:8px;"
                    "border-radius:6px;font-size:12px;line-height:1.45;"
                    "border:1px solid #2b3442;'>{tooltip}</div>",
            "style": {"backgroundColor": "transparent", "color": "white"},
        },
    )
    st.pydeck_chart(deck, width="stretch", height=650)

# --- Right: input / summary panel ------------------------------------------
with panel_col:
    # The legal notice renders FIRST so it is present in every screen state,
    # including the very first load before any forecast has been run.
    render_disclaimer()

    if not backend.backend_ready():
        st.warning(
            "Backend package not found - the app is running on mock data. "
            "Place the frontend folder next to the backend folder, or set the "
            "ANTARCTIC_NAV_BACKEND environment variable."
        )

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Iceberg Position Data</h4>", unsafe_allow_html=True)
    iceberg_upload = st.file_uploader(
        "Last week's positions (CSV/JSON)", type=["csv", "json"],
        key="iceberg_upload", label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Weather Forecast Input</h4>", unsafe_allow_html=True)
    weather_mode = st.radio("Input mode", ["Manual parameters", "Upload file"],
                            horizontal=True, label_visibility="collapsed")
    weather_upload = None
    manual_params = None
    if weather_mode == "Upload file":
        weather_upload = st.file_uploader("Weather data file",
                                          type=["csv", "json"],
                                          key="weather_upload",
                                          label_visibility="collapsed")
    else:
        wc1, wc2 = st.columns(2)
        with wc1:
            wind_speed = st.number_input("Wind speed (kt)", min_value=0.0,
                                         value=15.0, step=1.0)
            current_speed = st.number_input("Current speed (kt)", min_value=0.0,
                                            value=0.8, step=0.1)
        with wc2:
            wind_dir = st.number_input("Wind dir (deg)", min_value=0.0,
                                       max_value=360.0, value=225.0, step=5.0)
            current_dir = st.number_input("Current dir (deg)", min_value=0.0,
                                          max_value=360.0, value=90.0, step=5.0)
        temperature = st.number_input("Air temp (C)", value=-15.0, step=1.0)
        manual_params = {
            "wind_speed_kt": wind_speed,
            "wind_direction_deg": wind_dir,
            "current_speed_kt": current_speed,
            "current_direction_deg": current_dir,
            "temperature_c": temperature,
        }
        st.caption(
            "Drift uses ERA5 reanalysis at the selected date; these values are "
            "the fallback if ERA5 is unavailable."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Ice Cover Data</h4>", unsafe_allow_html=True)
    st.radio(
        "Ice cover source", ["Use model data", "Upload file"],
        horizontal=True, label_visibility="collapsed",
        key="ice_mode_radio",
    )

    if _ice_mode == "upload":
        st.file_uploader(
            "Ice concentration file", type=["npz", "csv", "json"],
            key="ice_upload", label_visibility="collapsed",
        )
        st.caption(
            "CSV/JSON: columns lat, lon, concentration (0-100). "
            "NPZ: must have keys 'lat', 'lon', 'concentration'."
        )
        if _ice_file is not None:
            st.caption(f"Showing **{_ice_file.name}** instead of the model field.")
    else:
        st.caption(
            "Using ross_sea_concentration.npz for "
            f"{st.session_state.forecast_date:%Y-%m-%d}."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Ship Position &amp; Destination</h4>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        ship_lat = st.number_input("Ship lat", value=float(DEMO_START[0]),
                                   format="%.3f", key="ship_lat")
        dest_mode = st.selectbox("Destination as", ["Lat/Lon", "Named location"],
                                 key="dest_mode")
    with sc2:
        ship_lon = st.number_input("Ship lon", value=float(DEMO_START[1]),
                                   format="%.3f", key="ship_lon")
    if dest_mode == "Lat/Lon":
        dc1, dc2 = st.columns(2)
        with dc1:
            dest_lat = st.number_input("Dest lat", value=float(DEMO_END[0]),
                                       format="%.3f", key="dest_lat")
        with dc2:
            dest_lon = st.number_input("Dest lon", value=float(DEMO_END[1]),
                                       format="%.3f", key="dest_lon")
        dest_name = None
    else:
        dest_name = st.text_input("Destination name", value="McMurdo Station")
        # Placeholder coords until a real geocoding lookup is wired in.
        dest_lat, dest_lon = -77.85, 166.7

    # --- Polar class + forecast date ---------------------------------------
    # The date picker is hidden while ice cover comes from an upload. Polar
    # Class then takes the full width; otherwise the two share a row to save
    # vertical space in the panel.
    if _ice_mode == "model":
        pc_col, date_col = st.columns(2)
    else:
        pc_col, date_col = st.container(), None

    with pc_col:
        st.session_state.polar_class = st.selectbox(
            "Polar Class", POLAR_CLASSES,
            index=POLAR_CLASSES.index(st.session_state.polar_class),
            key="polar_class_select",
            help="IACS polar class of the vessel. Lower number = heavier ice "
                 "capability.",
        )

    if date_col is not None:
        with date_col:
            picked = st.date_input(
                "Forecast date",
                value=st.session_state.forecast_date,
                min_value=dt.date(SEASON_YEAR_MIN, 11, 1),
                max_value=dt.date(SEASON_YEAR_MAX, 3, 31),
                key="forecast_date_input",
                help="Nov-Mar shipping seasons only (2021-22 to 2024-25).",
            )

        date_ok, date_msg = validate_forecast_date(picked)
        if not date_ok:
            st.error(date_msg)
        elif picked != st.session_state.forecast_date:
            st.session_state.forecast_date = picked
            st.rerun()

        if date_ok and picked.strftime("%Y%m%d") != DEMO_DATE:
            st.caption(
                "No pre-built cost surface for this date - it will be built on "
                "the fly (a few seconds longer)."
            )
    else:
        # Upload mode: no picker to validate, so nothing blocks the run.
        # NOTE: routing still runs against the model archive at this retained
        # date - an uploaded ice field feeds the map layer only and never
        # reaches calculate_routes().
        picked = st.session_state.forecast_date
        date_ok = True

    run_clicked = st.button("Run Forecast", width="stretch",
                            type="primary", disabled=not date_ok)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Warnings raised by the most recent backend call --------------------
    for msg in st.session_state.pending_errors:
        st.error(msg)

    # Only shown when a parse actually failed: what the file really looked like
    # after decoding, so the next bad upload is diagnosable from the screen.
    _dbg = st.session_state.pending_parse_debug
    if st.session_state.pending_errors and _dbg:
        with st.expander("Debug: column names found", expanded=False):
            st.write({
                "columns found": _dbg.get("columns", "[file could not be read as a table]"),
                "delimiter detected": _dbg.get("delimiter", "n/a"),
                "rows parsed": _dbg.get("rows", "n/a"),
                "columns required": ["lat", "lon", "concentration"],
            })
            st.caption(
                "Column matching ignores case and surrounding whitespace. "
                "Comma, semicolon, tab and pipe separators are all detected "
                "automatically."
            )
    for msg in st.session_state.pending_warnings:
        st.warning(msg)

    # --- Route recommendations ---------------------------------------------
    pareto = st.session_state.pareto_routes
    if pareto:
        if pareto.get("impossible"):
            render_impossible_banner(st.session_state.polar_class,
                                     pareto.get("reason", ""))
        else:
            render_route_cards(pareto)

    # --- Summary -----------------------------------------------------------
    rec_metrics = ((pareto.get("metrics") or {}).get("balanced")
                   if pareto else None) or {}
    metrics = backend.compute_summary_metrics(
        st.session_state.icebergs,
        polar_class=st.session_state.polar_class,
        ice_severity=rec_metrics.get("ice_severity"),
    )

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Summary</h4>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f'<div class="metric-value">{metrics["tracked_icebergs"]}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Icebergs Tracked</div>',
                    unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-value">{metrics["high_risk_icebergs"]}</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="metric-label">High Risk</div>',
                    unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{metrics["avg_size_km2"]} km&sup2;</div>',
                unsafe_allow_html=True)
    sized = metrics.get("sized_icebergs")
    sized_note = f" ({sized} of {metrics['tracked_icebergs']} sized)" if sized is not None else ""
    st.markdown(f'<div class="metric-label">Avg. Iceberg Size{sized_note}</div>',
                unsafe_allow_html=True)
    conf = metrics["forecast_confidence_pct"]
    conf_class = "metric-pct-ok" if conf >= 70 else "metric-pct-warn"
    st.markdown(
        f'<span class="{conf_class}">{conf}%</span> '
        f'<span class="metric-label">Forecast Confidence</span>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Forecast accuracy / model confidence ------------------------------
    render_accuracy_panel()

    # Repeated at the foot of the panel so the notice is visible whether the
    # user is looking at the controls or at the results.
    render_disclaimer()

# ---------------------------------------------------------------------------
# Run Forecast -> backend_stubs.run_forecast()
# ---------------------------------------------------------------------------
if run_clicked and date_ok:
    date_str = st.session_state.forecast_date.strftime("%Y%m%d")

    with st.spinner("Running drift physics and Pareto route search..."):
        # Re-read the ice field with the source selected right now, so a file
        # chosen in the same interaction as the click is honoured.
        _mode_now = st.session_state.get("ice_input_mode", "model")
        st.session_state.ice_cells = backend.load_ice_cover_data(
            date_str=None if _mode_now == "upload" else date_str,
            uploaded_file=st.session_state.get("ice_upload_file"),
            mode=_mode_now,
        )
        st.session_state.icebergs = backend.load_current_iceberg_positions(
            iceberg_upload, date_str=date_str)

        weather_data = backend.parse_weather_input(
            mode="upload" if weather_mode == "Upload file" else "manual",
            uploaded_file=weather_upload,
            manual_params=manual_params,
        )

        ship_pos = {"lat": ship_lat, "lon": ship_lon}
        destination = {"lat": dest_lat, "lon": dest_lon, "name": dest_name}

        trajectories, pareto_routes = backend.run_forecast(
            st.session_state.icebergs, weather_data, ship_pos, destination,
            polar_class=st.session_state.polar_class,
            date_str=date_str,
        )

    st.session_state.trajectories = trajectories
    st.session_state.pareto_routes = pareto_routes
    st.session_state.ship_pos_last = ship_pos
    st.session_state.destination_last = destination
    # Non-widget flag, consumed at the top of the next run (see above).
    st.session_state["_enable_forecast_layers"] = True
    st.session_state.pending_warnings = backend.drain_warnings()
    st.session_state.pending_errors = backend.drain_errors()
    st.session_state.pending_parse_debug = dict(backend.LAST_PARSE_DEBUG)

    st.rerun()
