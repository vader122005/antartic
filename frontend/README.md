# Ross Sea Iceberg Forecast — Frontend Shell

A Streamlit-only UI shell for an AI-assisted iceberg trajectory forecasting
tool, styled after a dark GIS "common operating picture" (map left, legend
overlay, summary/input panel right — same visual language as a fire-risk /
transmission-line ops dashboard).

**This app has zero real backend logic.** Every number, position, and path
you see on first run is randomly generated mock data, purely so the UI is
complete and clickable. Your job is to edit **one file** — `backend_stubs.py`
— to swap the mock data for real data and a real forecasting model.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File structure

| File               | What it's for                                                          | Do you edit it? |
|--------------------|--------------------------------------------------------------------------|:---:|
| `app.py`           | Layout, widgets, session state, wiring the "Run Forecast" button        | Only for UI/layout changes |
| `map_layers.py`    | Turns data (real or mock) into pydeck layers (ice, icebergs, paths)     | Only for map styling changes |
| `config.py`        | Ross Sea bounds, color ramps, default map view                          | Tweak freely |
| `backend_stubs.py` | **Placeholder functions with real-data contracts — edit this file**     | **Yes — this is the whole point** |
| `mock_data.py`     | Random dummy-data generators used only by `backend_stubs.py` for now    | No — delete usages here as you fill in `backend_stubs.py` |

## What to do next

Open `backend_stubs.py`. Each function has a docstring describing:
- what arguments it receives from the UI,
- the exact shape of data it must return so `app.py` / `map_layers.py`
  keep working unmodified.

Suggested order:
1. `load_ice_cover_data()` — plug in your real ice-cover raster/data source.
2. `load_current_iceberg_positions(uploaded_file)` — parse the uploaded
   CSV/JSON of last week's iceberg positions.
3. `parse_weather_input(mode, uploaded_file, manual_params)` — parse the
   uploaded weather file or manual wind/current/temperature inputs.
4. `run_forecast(iceberg_data, weather_data, ship_pos, destination)` —
   your real trajectory model. Must return `(trajectories, ship_route)`.
5. `compute_summary_metrics(iceberg_data)` — real KPI numbers for the
   right-hand summary cards.

You should not need to touch `app.py` or `map_layers.py` unless you want
to change layout/styling or add new map layers.

## Notes / known simplifications

- The Ross Sea straddles the antimeridian (180°). `config.py` models the
  region as two longitude segments (`160°→180°` and `-180°→-150°`) rather
  than doing full geodesic unwrapping — good enough for a mock UI, but
  double-check date-line handling in your real forecasting logic.
- The map uses CARTO's free dark basemap (`map_provider="carto"`) so no
  Mapbox API token is required. Swap this in `app.py` if you'd rather use
  your own Mapbox style.
- "Destination as Named location" currently just falls back to a fixed
  placeholder lat/lon (McMurdo Station) instead of geocoding — see the
  comment in `app.py` right above `dest_lat, dest_lon = -77.85, 166.7`.
- The top-left "Map Layers" panel in the reference image floats directly
  on top of the map. Streamlit renders `st.pydeck_chart` inside an
  isolated component frame, so a true floating overlay isn't reliably
  achievable without custom JS components — this app instead uses an
  adjacent icon rail + a legend card directly above the map, styled to
  match the same dark/card look.

---

## Wired to the real backend (SIH26059)

The UI now reads the locked backend package instead of `mock_data.py`.

### Layout

Either layout works. Current layout is **nested**:

```
antartic_nav_project/          <- the locked backend (note the spelling)
├── route_optimiser.py
├── ross_sea_concentration.npz
└── frontend/                  <- this folder
```

A sibling layout is also supported:

```
<parent>/
├── frontend/
└── antartic_nav_project/
```

`backend_paths.py` locates the backend at import time, trying in order:
the `$ANTARCTIC_NAV_BACKEND` override, the **parent directory itself**
(the nested case), then named sibling / child / grandparent directories under
**both** spellings (`antarctic_` and `antartic_`). Every candidate must contain
`route_optimiser.py` and `ross_sea_concentration.npz`, so a wrong guess is
rejected rather than silently accepted. All returned paths are absolute, so the
app does not depend on the directory you launch `streamlit` from.

`ensure_on_path()` *appends* the backend to `sys.path` rather than prepending,
so backend modules can never shadow a frontend module of the same name.

### Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the backend cannot be found the app still starts, falls back to mock data
and shows a warning. It never crashes on a missing backend.

### Tests

```bash
python tests/test_ui_smoke.py       # first load, date validation, PC selector
python tests/test_forecast_e2e.py   # demo scenario -> 3 Pareto routes
python tests/test_edge_cases.py     # impossible route, missing backend
```

### Demo scenario

Date `2023-11-15`, `PC4`, start `(-75.0, 165.0)` -> end `(-73.0, -155.0)`.

### Map rendering notes

**Antimeridian.** The Ross Sea straddles 180 deg, so quads holding both +179
and -179 would be drawn as a band across the whole map. They are clipped into
a west and an east half (`backend_stubs._split_at_antimeridian`) rather than
skipped - skipping them punched a one-cell-wide hole down the 180 deg line
(40 quads on a mid-November slice) and made the ice look like two disconnected
halves. The default view is centred at 160W so both sides frame together.

**Berg markers vs hazard rings.** These are deliberately different things.
The MARKER is a symbol sized on a square-root scale
(`8000 + sqrt(area)*800` m) so an 82 km2 berg and a 2943 km2 tabular giant are
distinguishable; it intentionally exaggerates small bergs for legibility. The
RING is the hazard zone and must stay truthful: it always equals the exclusion
radius `calculate_routes()` actually applied, so the map can never show a
smaller keep-out than the router used.

Rings scale with berg size because the berg's own extent now feeds the
`uncertainty_km` passed to the optimiser. The optimiser's 50 km is measured
from the CENTROID: for B22A (equivalent radius ~31 km) a flat 52 km ring left
only ~21 km of real clearance. Adding half the berg radius to the uncertainty
- doubled by the optimiser's 2x - keeps clearance beyond the berg EDGE at
~50 km for every size, and rings now span 51.5-82.6 km instead of a uniform
51.5 km.

**Iceberg size units.** BYU/NIC `size_1`/`size_2` are NAUTICAL MILES, not
metres. B22A reads 33 x 26, which as nautical miles is 2943 km2 - matching the
published ~60 x 40 km - but as metres would be 0.000858 km2, smaller than a
house. A metre reading collapses every berg onto the minimum marker radius.

**Drift tracks.** Bright white-yellow at width 6, with a head dot at the +48 h
end for direction. deck.gl has no ArrowLayer, and PathLayer's `getDashArray`
is only honoured with the `PathStyleExtension`, which pydeck 0.9.1 does not
emit - it would serialise and silently do nothing, so it is not used.

**Tooltips.** Every pickable layer carries its own pre-rendered `tooltip`
string and the deck template is just `{tooltip}`. A single combined template
naming `{id}`, `{size_km2}`, `{thickness}` and so on cannot work: pydeck
substitutes only the keys present in the hovered row and leaves the rest as
LITERAL braces, which is why the tooltip used to read
`{id}{label} size: {size_km2} km2`. One always-present key removes that
failure mode and lets each layer word its own tooltip. Values are HTML-escaped
because iceberg ids can come from a user-uploaded CSV.

### Ice cover source

The "Ice Cover Data" card selects where the sea-ice field comes from:

* **Use model data** (default) - `ross_sea_concentration.npz` for the selected date.
* **Upload file** - your own field, in one of:
  * **CSV / JSON** - columns `lat`, `lon`, `concentration`. One row per cell;
    each becomes a 0.5-degree square. Rows with a NaN in any of the three are skipped.
    The separator is detected automatically (comma, semicolon, tab or pipe), so a
    spreadsheet exported under a semicolon-list locale works unchanged. Column
    matching ignores case and surrounding whitespace, and UTF-8 (with or without
    BOM), cp1252 and latin-1 are all decoded.
  * **NPZ** - keys `lat`, `lon`, `concentration`. `lat`/`lon` may be 1-D axes
    (meshgridded automatically) or 2-D meshes matching `concentration`. A 3-D
    stack renders its first slice, with a warning.

Concentration may be 0-100 or 0-1; a file whose maximum is <= 1.0 is treated as
a fraction and scaled up. Reading 0.4 as 40% is the safer failure for a
navigation display than drawing consolidated pack ice as open water.

The **Forecast date picker is hidden** while ice cover comes from an upload,
and Polar Class takes the full width. Season validation is skipped in that
state so a control the user cannot see never blocks the run.

Note what the uploaded field does and does not do: it is **display only**. It
feeds the map's ice layer and never reaches `calculate_routes()`. Routing still
reads the model archive - cost surface, ERA5 wind and berg fixes - at the
retained date, which is why the panel states which date that is. For the same
reason a missing date resolves to `DEMO_DATE` rather than today: the archive
ends 2025-03-31, so today's date produces "date not present in
ross_sea_concentration.npz" and zero routes.

A malformed upload shows an error naming the expected fields and **falls back to
the model field** - the map never goes blank because of a bad file. Errors are
collected and rendered by `app.py` rather than raised with `st.error()` inside
`backend_stubs`, because the Run Forecast branch ends in `st.rerun()`, which
discards anything written before it.

### Data caveats surfaced in the UI

* **No iceberg fixes for the demo date.** The BYU/NIC file holds no
  observations between 2023-08 and 2025-01, so the whole 2023-24 season is
  missing. The map shows each berg's *last known fix* with its age in days,
  and warns when the freshest is over 60 days old. Choose a 2022-23 or
  2024-25 date for contemporaneous positions.
* **77% of berg size values are 0**, meaning "not measured", not "zero area".
  Those bergs get a neutral 0.5 risk rather than 0.0, and the average-size KPI
  is computed only over bergs that reported a size.
* **Cost surfaces are pre-built for 2023-11-15 only.** Other dates are built
  on demand in well under a second and cached in `frontend/.cache/` (not in the
  locked backend folder). The UI says so before you run.

### Why the frontend controls the working directory

Several backend entry points resolve data files relative to the process CWD
(`route_optimiser.build_cost_surface_for`, `iceberg_lstm._get_model`,
`seaice_lstm._get`). Streamlit's CWD is the frontend folder, so those all
missed, and every date without a pre-built surface failed with
`[Errno 2] No such file or directory: 'ross_sea_concentration.npz'`.

`backend_paths.backend_cwd()` is a re-entrant, lock-guarded context manager
that supplies the CWD the backend expects. It is locked because `os.chdir` is
process-global while Streamlit runs each session in its own thread, so an
unguarded swap in one session would relocate every other session's relative
paths.

`backend_stubs.cost_surface_for()` additionally resolves the surface with
absolute paths *before* calling `calculate_routes()`, so the relative-path
branch is never reached and build artifacts stay out of the backend folder.

### A note on build speed

`cost_surface.thickness_to_csv_row()` re-filters the lookup DataFrame and walks
it with `iterrows()` for every grid cell: 0.47 ms x 8599 cells = ~4.1 s, most
of a build. SIC maps onto only nine discrete thickness values, so
`backend_stubs._memoised_csv_lookup()` wraps that function for the duration of
a build, collapsing 8599 calls to nine. Builds drop from 6-12 s to ~0.4 s.

This is a runtime wrapper on the imported module object - it modifies no file
in the locked backend and is restored afterwards - and it is **exact**, not an
approximation: a memoised rebuild of the demo date is bit-identical to the
backend's own pre-built surface across every array.
