"""
test_map_visuals.py
===================
Two rendering regressions:

1. ANTIMERIDIAN SEAM. The Ross Sea straddles 180 deg. Quads holding both
   +179 and -179 used to be dropped, punching a one-cell-wide hole down the
   180 deg line so the ice field looked like two disconnected halves. They are
   now clipped into a west and an east half instead.

2. LITERAL TOOLTIP BRACES. pydeck substitutes only the keys present in the
   hovered row, leaving the rest as literal text - so a combined template
   rendered as "{id}{label} size: {size_km2} km2". Every pickable layer now
   carries its own rendered `tooltip` string and the deck template is just
   "{tooltip}".
"""

import os
import re
import sys

_FRONTEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)
os.chdir(_FRONTEND)
APP = os.path.join(_FRONTEND, "app.py")

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import backend_stubs as backend
from config import DEFAULT_VIEW_STATE, LAT_MIN, LAT_MAX
from map_layers import (build_ice_layer, build_iceberg_layer,
                        build_trajectory_layer, build_pareto_route_layers,
                        build_iceberg_uncertainty_layer,
                        build_ship_markers_layer, build_graticule_layer)

FAIL = []


def chk(name, cond, detail=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    print("=== 1. antimeridian seam ===")
    q = [[179.5, -75.0], [179.5, -74.0], [-179.5, -74.0], [-179.5, -75.0]]
    pieces = backend._split_at_antimeridian(q)
    chk("seam quad splits into two", len(pieces) == 2, f"{len(pieces)} pieces")
    chk("each piece stays on one side of 180",
        all(max(p[0] for p in pc) - min(p[0] for p in pc) < 180 for pc in pieces))
    chk("ordinary quad untouched",
        len(backend._split_at_antimeridian(
            [[170., -75.], [171., -75.], [171., -74.], [170., -74.]])) == 1)

    cells = backend.load_ice_cover_data("20231115")
    backend.drain_warnings()
    lons = np.array([c["polygon"][0][0] for c in cells])
    near_seam = int(((lons > 178) | (lons < -178)).sum())
    chk("ice rendered right up to the seam", near_seam > 0,
        f"{near_seam} polygons within 2 deg of 180")
    chk("both sides of the antimeridian populated",
        (lons > 0).sum() > 0 and (lons < 0).sum() > 0,
        f"east={(lons > 0).sum()} west={(lons < 0).sum()}")
    widest = max(max(p[0] for p in c["polygon"]) - min(p[0] for p in c["polygon"])
                 for c in cells)
    chk("no polygon spans the globe", widest < 180.0, f"widest {widest:.2f} deg")

    print()
    print("=== 2. view state frames the whole Ross Sea ===")
    chk("centred across the antimeridian, not on it",
        DEFAULT_VIEW_STATE["longitude"] == -160.0,
        str(DEFAULT_VIEW_STATE["longitude"]))
    chk("zoom fits the region", abs(DEFAULT_VIEW_STATE["zoom"] - 3.8) < 1e-9,
        str(DEFAULT_VIEW_STATE["zoom"]))
    lat = DEFAULT_VIEW_STATE["latitude"]
    chk("centre latitude inside the data bounds", LAT_MIN <= lat <= LAT_MAX,
        f"{lat} in [{LAT_MIN}, {LAT_MAX}]")

    print()
    print("=== 3. CSV upload cell size matches a 0.5 deg grid ===")
    chk("half-cell is 0.25 deg", backend._POINT_CELL_HALF_DEG == 0.25,
        str(backend._POINT_CELL_HALF_DEG))

    print()
    print("=== 4. every pickable layer carries a tooltip ===")
    bergs = backend.load_current_iceberg_positions(date_str="20231115")
    backend.drain_warnings()
    weather = backend.parse_weather_input("manual", manual_params={})
    traj, par = backend.run_forecast(bergs, weather, {"lat": -75.0, "lon": 165.0},
                                     {"lat": -73.0, "lon": -155.0}, "PC4", "20231115")
    backend.drain_warnings(); backend.drain_errors()

    layers = [build_graticule_layer(), build_ice_layer(cells),
              build_iceberg_uncertainty_layer(bergs), build_iceberg_layer(bergs),
              build_trajectory_layer(traj)]
    layers += build_pareto_route_layers(par)
    layers.append(build_ship_markers_layer(
        {"lat": -75.0, "lon": 165.0},
        {"lat": -73.0, "lon": -155.0, "name": "McMurdo"}))

    missing = []
    for layer in layers:
        if layer is None or not getattr(layer, "pickable", False):
            continue
        rows = layer.data if isinstance(layer.data, list) else []
        if rows and not all("tooltip" in r for r in rows):
            missing.append(layer.id)
    chk("no pickable layer missing 'tooltip'", not missing, str(missing))

    # No tooltip string may itself contain an unsubstituted placeholder.
    leftovers = []
    for layer in layers:
        rows = layer.data if isinstance(layer.data, list) else []
        for r in rows:
            tip = r.get("tooltip") if isinstance(r, dict) else None
            if tip and re.search(r"\{\w+\}", tip):
                leftovers.append((layer.id, tip))
    chk("no literal {placeholders} inside any tooltip", not leftovers,
        str(leftovers[:2]))

    print()
    print("=== 5. deck template references only 'tooltip' ===")
    app_src = open(APP, encoding="utf-8").read()
    block = app_src[app_src.index("tooltip={"):]
    block = block[:block.index("},\n") + 3]
    keys = set(re.findall(r"\{(\w+)\}", block))
    chk("template uses exactly one key", keys == {"tooltip"}, str(sorted(keys)))
    chk("uses the html tooltip form", '"html"' in block)
    for stale in ("{id}", "{size_km2}", "{risk}", "{thickness}", "{label}",
                  "{distance_km}", "{time_h}", "{fuel_t}"):
        chk(f"stale placeholder {stale} removed", stale not in block)

    print()
    print("=== 6. berg marker radius is size-discriminating ===")
    import map_layers as ml
    radii = {a: ml._berg_radius_m(a) / 1000.0
             for a in (0, 82, 309, 1132, 2943, 10000, 100000)}
    chk("radius increases with area",
        all(radii[a] < radii[c] for a, c in
            [(82, 309), (309, 1132), (1132, 2943), (2943, 10000), (10000, 100000)]),
        str({k: round(v) for k, v in radii.items()}))
    chk("small and large bergs are visually distinct",
        radii[2943] / radii[82] > 3.0, f"{radii[2943]/radii[82]:.1f}x")
    chk("unmeasured berg still visible", radii[0] > 5.0, f"{radii[0]:.1f} km")
    chk("largest berg does not swallow the sea", radii[100000] < 300.0,
        f"{radii[100000]:.0f} km")

    print()
    print("=== 7. size_km2 uses nautical miles, not metres ===")
    # B22A is a real NIC berg of roughly 60 x 40 km. size_1/size_2 are 33 x 26;
    # as metres that is 0.000858 km2 (smaller than a house), as nautical miles
    # 2943 km2. Every berg would collapse onto the minimum marker radius under
    # the metre reading.
    chk("nm conversion constant", abs(backend._NM_TO_KM - 1.852) < 1e-9,
        str(backend._NM_TO_KM))
    all_bergs = backend.load_current_iceberg_positions(date_str="20250215")
    backend.drain_warnings()
    big = max(all_bergs, key=lambda z: z["size_km2"])
    print(f"         sample: {big['id']} -> size_km2={big['size_km2']}")
    chk("largest berg is of tabular scale (1000-6000 km2)",
        1000 < big["size_km2"] < 6000, f"{big['size_km2']} km2")

    print()
    print("=== 8. hazard ring never understates the optimiser's exclusion ===")
    ring = build_iceberg_uncertainty_layer(all_bergs)
    by_id = {d["id"]: d for d in ring.data}
    understated = []
    for berg in all_bergs:
        drawn = by_id[berg["id"]]["radius_km"]
        optimiser = 50.0 + 2.0 * float(berg["uncertainty_km"])
        if drawn < optimiser - 0.05:
            understated.append((berg["id"], drawn, optimiser))
    chk("drawn ring >= routed exclusion for every berg", not understated,
        str(understated[:2]))
    spread = [by_id[b["id"]]["radius_km"] for b in all_bergs]
    chk("rings scale with berg size", max(spread) > min(spread) + 10.0,
        f"{min(spread):.1f}-{max(spread):.1f} km")
    for berg in all_bergs:
        if berg["size_km2"] > 0:
            clearance = (by_id[berg["id"]]["radius_km"]
                         - backend.berg_equivalent_radius_km(berg["size_km2"]))
            chk(f"{berg['id']}: ~50 km clearance beyond the berg edge",
                45.0 <= clearance <= 60.0, f"{clearance:.1f} km")

    print()
    print("=== 9. trajectory is legible and shows direction ===")
    from config import TRAJECTORY_COLOR
    chk("trajectory colour is bright green",
        TRAJECTORY_COLOR == [0, 255, 120, 255], str(TRAJECTORY_COLOR))
    tl = build_trajectory_layer(traj)
    chk("path is at least 4 px wide", tl.width_min_pixels == 4,
        str(tl.width_min_pixels))
    chk("still a plain PathLayer", tl.type == "PathLayer", str(tl.type))
    head = ml.build_trajectory_head_layer(traj)
    chk("arrowhead layer built", head is not None)
    if head:
        chk("one head per track", len(head.data) == len(traj),
            f"{len(head.data)} vs {len(traj)} tracks")
        chk("head sits at the +48 h end",
            all(h["position"] == [float(traj[h["id"]][-1][0]),
                                  float(traj[h["id"]][-1][1])] for h in head.data))
        chk("head is pickable with a tooltip",
            head.pickable and all("tooltip" in h for h in head.data))

    print()
    print("SUMMARY:", "ALL PASS" if not FAIL else f"{len(FAIL)} FAILED -> {FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
