"""
backend_paths.py
================
Locates the locked backend package and exposes absolute paths to its data
files. Import this before anything that touches the backend.

WHY THIS EXISTS
---------------
Two things make a hardcoded 'antarctic_nav_project/...' path unreliable:

1. SPELLING. The backend directory on this machine is actually named
   ``antartic_nav_project`` (no 'c' in "antartic"), not the
   ``antarctic_nav_project`` used throughout the brief. A literal path would
   fail with FileNotFoundError on first load.

2. WORKING DIRECTORY. ``np.load('antarctic_nav_project/...')`` resolves
   relative to the process CWD, which for Streamlit is wherever the user ran
   ``streamlit run`` from - not necessarily this file's folder. Every path
   here is absolute and derived from ``__file__``.

Resolution order (first hit wins):
    1. $ANTARCTIC_NAV_BACKEND environment variable
    2. the PARENT directory itself (frontend nested inside the backend)
    3. a named sibling directory next to this frontend folder
    4. a named child directory inside this frontend folder
    5. a named sibling of the parent (one level further up)

Every candidate must contain the marker files below, so a wrong guess is
rejected rather than silently accepted.

Accepted directory names: both spellings, so this keeps working if the
backend folder is ever renamed to the spelling used in the docs.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent

# Both spellings - the real folder uses "antartic", the docs say "antarctic".
_CANDIDATE_NAMES = (
    "antarctic_nav_project",
    "antartic_nav_project",
)

# A directory only counts as the backend if it actually contains these.
_REQUIRED_MARKERS = (
    "route_optimiser.py",
    "ross_sea_concentration.npz",
)


def _looks_like_backend(path: Path) -> bool:
    return path.is_dir() and all((path / m).exists() for m in _REQUIRED_MARKERS)


def _candidate_dirs():
    env = os.environ.get("ANTARCTIC_NAV_BACKEND")
    if env:
        yield Path(env).expanduser().resolve()
    # The frontend may live INSIDE the backend folder, in which case the parent
    # directory IS the backend. Checked before the named searches, which would
    # otherwise look for <backend>/antartic_nav_project/ - a path that does not
    # exist. _looks_like_backend() still gates this, so a parent that is not
    # the backend is skipped harmlessly.
    yield FRONTEND_DIR.parent
    for name in _CANDIDATE_NAMES:
        yield FRONTEND_DIR.parent / name      # sibling
        yield FRONTEND_DIR / name             # child
        yield FRONTEND_DIR.parent.parent / name   # one level further up


def find_backend_dir():
    """Return the backend directory as a Path, or None if it cannot be found."""
    for cand in _candidate_dirs():
        try:
            if _looks_like_backend(cand):
                return cand
        except OSError:
            continue
    return None


BACKEND_DIR = find_backend_dir()
BACKEND_AVAILABLE = BACKEND_DIR is not None


def ensure_on_path():
    """Put the backend directory on sys.path so its modules are importable."""
    if BACKEND_DIR is None:
        return False
    p = str(BACKEND_DIR)
    if p not in sys.path:
        # append, not insert(0): the backend must never shadow a frontend
        # module of the same name (e.g. config.py exists in both worlds).
        sys.path.append(p)
    return True


# ---------------------------------------------------------------------------
# Working-directory guard
# ---------------------------------------------------------------------------

# Several backend entry points resolve data files RELATIVE to the process CWD:
#   route_optimiser.build_cost_surface_for -> "ross_sea_concentration.npz",
#                                             "pc_ice_speed_model.csv"
#   iceberg_lstm._get_model                -> "iceberg_lstm.pt"
#   seaice_lstm._get / build_wind_on_sic_grid -> "seaice_model.pt", "ross_sea_*.npz"
# Streamlit's CWD is the frontend folder, so those all miss. Since the backend
# is locked, the frontend has to supply the CWD the backend expects.
#
# os.chdir is PROCESS-global while Streamlit runs each session's script in its
# own thread, so an unguarded chdir in one session would silently relocate
# every other session's relative paths. The lock serialises the swap, and the
# depth counter makes it re-entrant (nested use must not restore early).
_CWD_LOCK = threading.RLock()
_CWD_DEPTH = 0


@contextlib.contextmanager
def backend_cwd():
    """
    Temporarily run with the backend directory as the working directory.

    Keep the guarded region as short as possible: while it is held, relative
    paths anywhere else in the process resolve against the backend folder.
    A no-op (but still safe) if the backend was not found.
    """
    global _CWD_DEPTH
    if BACKEND_DIR is None:
        yield None
        return
    with _CWD_LOCK:
        previous = os.getcwd()
        outermost = _CWD_DEPTH == 0
        try:
            if outermost:
                os.chdir(BACKEND_DIR)
            _CWD_DEPTH += 1
            yield BACKEND_DIR
        finally:
            _CWD_DEPTH -= 1
            if outermost:
                os.chdir(previous)


def data_path(filename: str) -> str:
    """Absolute path to a file inside the backend directory."""
    if BACKEND_DIR is None:
        raise FileNotFoundError(
            "Backend directory not found. Set ANTARCTIC_NAV_BACKEND to the "
            "folder containing route_optimiser.py, or place the frontend "
            "folder next to it."
        )
    return str(BACKEND_DIR / filename)


def has_data(filename: str) -> bool:
    """True if a backend data file exists (does not raise when the dir is gone)."""
    if BACKEND_DIR is None:
        return False
    return (BACKEND_DIR / filename).exists()


def cost_surface_path(polar_class: str, date_str: str):
    """
    Absolute path to a PRE-BUILT cost surface, or None if that combination was
    never generated.

    Only 2023-11-15 has pre-built surfaces (all 7 PC classes). Returning None
    tells calculate_routes() to build the surface on the fly (~3.5 s), which is
    correct, just slower.
    """
    if BACKEND_DIR is None:
        return None
    name = f"cost_surface_{polar_class}_{date_str}.npz"
    return str(BACKEND_DIR / name) if (BACKEND_DIR / name).exists() else None


def describe():
    """One-line human-readable status, used in the UI's diagnostics."""
    if BACKEND_DIR is None:
        return "backend NOT found"
    return f"backend at {BACKEND_DIR}"
