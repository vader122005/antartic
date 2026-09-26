"""
grid_utils.py - shared geospatial helpers for the Antarctic Nav backend.

Why this module exists
----------------------
Two properties of the supplied data break the naive interpolation that the
original baseline scripts used, and both fail SILENTLY (wrong numbers, no
exception):

1. The NSIDC SIC grid is CURVILINEAR (polar stereographic), not a regular
   lat/lon mesh. Latitude varies by up to 23 deg ACROSS a single row and
   longitude by up to 83 deg DOWN a single column, so treating lat[:, 0] and
   lon[0, :] as separable 1-D axes (as RegularGridInterpolator requires) is
   invalid. We use a KD-tree nearest-neighbour lookup on 3-D unit-sphere
   coordinates instead, which is correct for any grid topology and needs no
   antimeridian special case.

2. ERA5 wind longitude is NON-MONOTONIC: it runs 160 -> 180 then wraps to
   -180 -> -150. Feeding that straight to RegularGridInterpolator is either an
   error or garbage. We unwrap to a continuous ascending frame before
   interpolating. Wind latitude is also DESCENDING (-70 -> -78), so we flip it.

Region convention: Ross Sea, lat <= -60, lon 160E..130W (straddles the
antimeridian). Longitudes are -180/180 throughout, matching the iceberg CSV
and the already-converted ERA5 file. Nothing here re-converts them.
"""

from __future__ import annotations

import datetime as dt
import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088

# Ross Sea bounding box (the project's hard spatial constraint)
ROSS_LAT_MAX = -60.0
ROSS_LON_EAST = 160.0    # eastern edge, going east from 160E
ROSS_LON_WEST = -130.0   # western edge, i.e. 130W


# ---------------------------------------------------------------------------
# Longitude handling
# ---------------------------------------------------------------------------

def unwrap_lon(lon, origin: float = 160.0):
    """
    Map longitude(s) from the -180/180 convention onto a continuous ascending
    frame starting at `origin`, so the Ross Sea sector (160E -> 180 -> -180 ->
    -130) becomes a monotonic interval. Works on scalars or arrays.
    """
    lon = np.asarray(lon, dtype=np.float64)
    return origin + np.mod(lon - origin, 360.0)


def in_ross_box(lat, lon):
    """Boolean mask: is (lat, lon) inside the Ross Sea bounding box?"""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return (lat <= ROSS_LAT_MAX) & (
        (lon >= ROSS_LON_EAST) | (lon <= ROSS_LON_WEST)
    )


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Scalar or array, antimeridian-safe."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64))
                              for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _to_unit_sphere(lat_deg, lon_deg):
    """Lat/lon in degrees -> 3-D Cartesian on the unit sphere."""
    la = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lo = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cla = np.cos(la)
    return np.stack([cla * np.cos(lo), cla * np.sin(lo), np.sin(la)], axis=-1)


# ---------------------------------------------------------------------------
# Curvilinear SIC grid lookup
# ---------------------------------------------------------------------------

class GridLocator:
    """
    Nearest-cell lookup for the curvilinear NSIDC grid.

    Builds a KD-tree over 3-D unit-sphere positions of the grid cells, which is
    exact for curvilinear meshes. By default only cells flagged valid are
    indexed, so queries never snap to a land / out-of-region cell.
    """

    def __init__(self, lat_grid, lon_grid, valid_mask=None):
        self.lat = np.asarray(lat_grid, dtype=np.float64)
        self.lon = np.asarray(lon_grid, dtype=np.float64)
        self.shape = self.lat.shape

        if valid_mask is None:
            valid_mask = np.ones(self.shape, dtype=bool)
        self.valid_mask = np.asarray(valid_mask, dtype=bool)

        idx = np.argwhere(self.valid_mask)
        if len(idx) == 0:
            raise ValueError("GridLocator: no valid cells to index")
        self._idx = idx
        pts = _to_unit_sphere(self.lat[self.valid_mask],
                              self.lon[self.valid_mask])
        self._tree = cKDTree(pts)

    def nearest(self, lat, lon):
        """Return (i, j) of the nearest valid grid cell to (lat, lon)."""
        q = _to_unit_sphere(np.atleast_1d(lat), np.atleast_1d(lon))
        _, k = self._tree.query(q, k=1)
        i, j = self._idx[int(np.atleast_1d(k)[0])]
        return int(i), int(j)

    def nearest_many(self, lat, lon):
        """Vectorised nearest lookup -> (i_array, j_array)."""
        q = _to_unit_sphere(np.asarray(lat), np.asarray(lon))
        _, k = self._tree.query(q, k=1)
        sel = self._idx[np.asarray(k).ravel()]
        return sel[:, 0], sel[:, 1]

    def distance_km(self, lat, lon):
        """Great-circle distance from (lat, lon) to its nearest valid cell."""
        i, j = self.nearest(lat, lon)
        return float(haversine_km(lat, lon, self.lat[i, j], self.lon[i, j]))

    def sample(self, field, lat, lon, default=np.nan):
        """Sample a (rows, cols) field at (lat, lon) via nearest valid cell."""
        i, j = self.nearest(lat, lon)
        v = field[i, j]
        return default if (v is None or np.isnan(v)) else float(v)


# ---------------------------------------------------------------------------
# ERA5 wind sampling
# ---------------------------------------------------------------------------

class WindSampler:
    """
    Bilinear sampling of ERA5 10 m wind, correcting the two layout problems
    described in the module docstring. Times are hourly but NON-CONTIGUOUS:
    the file holds only the Nov-Mar seasons, so it jumps from 31 Mar to 1 Nov.
    Lookups are by nearest timestamp via binary search over epoch seconds.
    """

    def __init__(self, npz_path, mmap=True):
        z = np.load(npz_path, mmap_mode="r" if mmap else None)
        self.times = np.asarray(z["time"])
        self.u = z["wind_u"]
        self.v = z["wind_v"]

        lat = np.asarray(z["lat"], dtype=np.float64)
        lon = np.asarray(z["lon"], dtype=np.float64)

        # Flip latitude to ascending if needed
        self._lat_flip = bool(lat[0] > lat[-1])
        self.lat = lat[::-1] if self._lat_flip else lat

        # Unwrap longitude to a continuous ascending frame
        lon_u = unwrap_lon(lon)
        self._lon_order = np.argsort(lon_u)
        self.lon = lon_u[self._lon_order]

        secs = np.array([
            dt.datetime.strptime(str(t), "%Y-%m-%d %H:%M").timestamp()
            for t in self.times
        ])
        self._order = np.argsort(secs)
        self._secs = secs[self._order]

        self._cache_idx = None
        self._cache = None

    def time_index(self, when):
        """Index of the nearest available wind timestamp to `when`."""
        if not isinstance(when, dt.datetime):
            when = parse_date(when)
        target = when.timestamp()
        p = int(np.searchsorted(self._secs, target))
        if p <= 0:
            k = 0
        elif p >= len(self._secs):
            k = len(self._secs) - 1
        else:
            k = p if (self._secs[p] - target) < (target - self._secs[p - 1]) else p - 1
        return int(self._order[k])

    def hours_from_nearest(self, when):
        """
        Gap in hours between `when` and the nearest available wind timestamp.
        Large values mean the request falls in the Apr-Oct data gap.
        """
        if not isinstance(when, dt.datetime):
            when = parse_date(when)
        t_idx = self.time_index(when)
        actual = dt.datetime.strptime(str(self.times[t_idx]), "%Y-%m-%d %H:%M")
        return abs((actual - when).total_seconds()) / 3600.0

    def _slices(self, t_idx):
        if self._cache_idx == t_idx:
            return self._cache
        us = np.asarray(self.u[t_idx], dtype=np.float64)
        vs = np.asarray(self.v[t_idx], dtype=np.float64)
        if self._lat_flip:
            us = us[::-1, :]
            vs = vs[::-1, :]
        us = us[:, self._lon_order]
        vs = vs[:, self._lon_order]
        self._cache_idx, self._cache = t_idx, (us, vs)
        return self._cache

    def sample(self, lat, lon, when):
        """
        Bilinear wind (u, v) in m/s at a position and time. Positions outside
        the wind domain clamp to the domain edge rather than returning zero, so
        a berg near the boundary still feels a realistic wind.
        """
        us, vs = self._slices(self.time_index(when))
        y = float(np.clip(float(lat), self.lat[0], self.lat[-1]))
        x = float(np.clip(float(unwrap_lon(lon)), self.lon[0], self.lon[-1]))
        return (_bilinear(us, self.lat, self.lon, y, x),
                _bilinear(vs, self.lat, self.lon, y, x))

    def grid_at(self, when):
        """Full (u, v) field for the nearest timestamp, on the corrected axes."""
        return self._slices(self.time_index(when))


def _bilinear(field, ax_y, ax_x, y, x):
    """Bilinear interpolation on a regular ascending (ax_y, ax_x) grid."""
    i = int(np.clip(np.searchsorted(ax_y, y) - 1, 0, len(ax_y) - 2))
    j = int(np.clip(np.searchsorted(ax_x, x) - 1, 0, len(ax_x) - 2))
    y0, y1 = ax_y[i], ax_y[i + 1]
    x0, x1 = ax_x[j], ax_x[j + 1]
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    f = field[i:i + 2, j:j + 2]
    return float(
        f[0, 0] * (1 - ty) * (1 - tx) + f[0, 1] * (1 - ty) * tx
        + f[1, 0] * ty * (1 - tx) + f[1, 1] * ty * tx
    )


# ---------------------------------------------------------------------------
# Time / date helpers
# ---------------------------------------------------------------------------

SHIPPING_MONTHS = (11, 12, 1, 2, 3)   # Nov-Mar Southern Hemisphere summer


def parse_date(s):
    """
    Parse the several date/time spellings used across these datasets:
    'YYYYMMDD' (SIC), 'YYYY-MM-DD' (iceberg CSV), 'YYYY-MM-DD HH:MM' (ERA5).
    """
    if isinstance(s, dt.datetime):
        return s
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError("Unrecognised date/time string: %r" % (s,))


def is_shipping_season(d):
    """True if the date falls in the Nov-Mar training / shipping window."""
    return parse_date(d).month in SHIPPING_MONTHS


def season_label(d):
    """
    Label the Nov-Mar season a date belongs to, e.g. '2022-23'. Nov and Dec
    open the season named for that year; Jan-Mar close the previous year's.
    Returns None outside the shipping window.
    """
    d = parse_date(d)
    if d.month in (11, 12):
        return "%d-%02d" % (d.year, (d.year + 1) % 100)
    if d.month in (1, 2, 3):
        return "%d-%02d" % (d.year - 1, d.year % 100)
    return None


def sic_date_index(dates_array, when):
    """Index of the nearest SIC date ('YYYYMMDD' strings) to `when`."""
    target = parse_date(when).timestamp()
    secs = np.array([parse_date(str(x)).timestamp() for x in dates_array])
    return int(np.argmin(np.abs(secs - target)))
