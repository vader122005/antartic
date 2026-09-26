"""
config.py
=========
Static configuration: map bounds, color ramps, default view state.

Nothing here talks to a backend — it's just numbers/strings used by the
UI layer. Tweak freely to match your real-world coverage area.
"""

# ---------------------------------------------------------------------------
# Ross Sea geographic bounds
# ---------------------------------------------------------------------------
# The Ross Sea straddles the antimeridian (180 degrees), so instead of a
# single min/max longitude we describe it as two longitude segments that
# together make up the region of interest. Each segment is a normal
# (west, east) pair in the standard -180..180 range.
# Widened to the real extent of the NSIDC data rather than a nominal box: the
# valid (non-NaN) cells reach lat -60 and run from 160E east to about 120W, so
# the previous -78..-70 / 160E..150W graticule stopped well short of the ice.
LAT_MIN = -78.5
LAT_MAX = -60.0

LON_SEGMENTS = [
    (160.0, 180.0),    # eastern side of the date line (~Victoria Land / McMurdo)
    (-180.0, -130.0),  # western side of the date line (~Ross Ice Shelf front)
]

# Default pydeck view. The Ross Sea spans 160E -> 180 -> 130W, whose midpoint
# is ~165W, NOT the antimeridian: centring on 180 pushed most of the ice field
# into the left half of the map and cut off the eastern sector entirely.
DEFAULT_VIEW_STATE = {
    "latitude": -71.5,
    "longitude": -160.0,
    "zoom": 3.8,
    "pitch": 0,
    "bearing": 0,
}

# ---------------------------------------------------------------------------
# Color ramps (RGBA, 0-255) — mirrors the "white -> red" fire-risk look
# ---------------------------------------------------------------------------
ICE_COLOR_LOW = [255, 250, 245, 160]   # thin ice -> near white
ICE_COLOR_HIGH = [178, 24, 24, 190]    # thick ice -> dark red

# Iceberg risk color ramp (green -> red), same idea as the reference image's
# "Transmission Line Risk" legend.
ICEBERG_RISK_LOW = [40, 168, 90, 220]   # low risk -> green
ICEBERG_RISK_HIGH = [214, 39, 40, 220]  # high risk -> red

# Bright green: the drift track has to read against BOTH the dark ocean and the
# deep-red high-concentration ice. Green sits furthest from the white->red ice
# ramp, so it stays legible across the whole concentration range.
TRAJECTORY_COLOR = [0, 255, 120, 255]       # predicted iceberg path
SHIP_ROUTE_COLOR = [64, 156, 255, 255]      # suggested ship route -> blue
SHIP_POS_COLOR = [64, 156, 255, 255]
DESTINATION_COLOR = [80, 220, 120, 255]

GRID_LINE_COLOR = [120, 140, 160, 90]

# ---------------------------------------------------------------------------
# Pareto route colors (RGBA) - one per weighting returned by
# route_optimiser.calculate_routes(). The balanced route is the recommended
# one, so it gets the brightest/warmest treatment (gold).
# ---------------------------------------------------------------------------
ROUTE_FUEL_COLOR       = [230, 120, 60, 200]    # muted orange - fuel priority
ROUTE_BALANCED_COLOR   = [245, 180, 0, 220]     # gold        - balanced (recommended)
ROUTE_TIME_COLOR       = [60, 190, 200, 200]    # teal        - time priority

# Hex equivalents, for the HTML route cards in the right-hand panel
ROUTE_FUEL_HEX     = "#e6783c"
ROUTE_BALANCED_HEX = "#f5b400"
ROUTE_TIME_HEX     = "#3cbec8"

# Maps a route's cost_type -> (map color, card hex, display label)
ROUTE_STYLES = {
    "fuel_priority": (ROUTE_FUEL_COLOR,     ROUTE_FUEL_HEX,     "FUEL PRIORITY"),
    "balanced":      (ROUTE_BALANCED_COLOR, ROUTE_BALANCED_HEX, "BALANCED"),
    "time_priority": (ROUTE_TIME_COLOR,     ROUTE_TIME_HEX,     "TIME PRIORITY"),
}
ROUTE_ORDER = ("fuel_priority", "balanced", "time_priority")

# Iceberg positional-uncertainty footprint
UNCERTAINTY_RING_COLOR = [255, 140, 0, 30]      # transparent amber

# ---------------------------------------------------------------------------
# Forecast date window
# ---------------------------------------------------------------------------
# The backend is trained and validated on the Nov-Mar southern-hemisphere
# shipping season only, across the 2021-22 .. 2024-25 seasons. Dates outside
# this are not merely untested - the SIC file has no slice for them at all.
SHIPPING_MONTHS = {11, 12, 1, 2, 3}
SEASON_YEAR_MIN = 2021
SEASON_YEAR_MAX = 2025

# The locked demo date: the only one with pre-built cost surfaces for all
# 7 polar classes, and a genuinely ice-affected date (mean SIC 49.6%).
DEMO_DATE = "20231115"

POLAR_CLASSES = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]
DEFAULT_POLAR_CLASS = "PC4"

# Demo transect used by the backend integration test
DEMO_START = (-75.0, 165.0)
DEMO_END = (-73.0, -155.0)

# ---------------------------------------------------------------------------
# Misc UI constants
# ---------------------------------------------------------------------------
APP_TITLE = "Ross Sea Iceberg Forecast — Common Operating Picture"
GRID_STEP_DEG = 2.0   # spacing of the dummy ice-cover grid cells
GRATICULE_STEP_DEG = 2.0  # spacing of the lat/lon reference grid lines
