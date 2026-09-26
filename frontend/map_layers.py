"""
map_layers.py
=============
Builds pydeck Layer objects from already-loaded data. This file is pure
presentation — it never fetches or invents data itself, it just turns
data (real or mock, doesn't care) into map layers.

Nothing here needs to change when you plug in your real backend, as long
as the data shapes documented in backend_stubs.py stay the same.
"""

import html
import math

import pydeck as pdk

from config import (
    ROUTE_STYLES,
    ROUTE_ORDER,
    UNCERTAINTY_RING_COLOR,
    LAT_MIN,
    LAT_MAX,
    LON_SEGMENTS,
    ICE_COLOR_LOW,
    ICE_COLOR_HIGH,
    ICEBERG_RISK_LOW,
    ICEBERG_RISK_HIGH,
    TRAJECTORY_COLOR,
    SHIP_ROUTE_COLOR,
    SHIP_POS_COLOR,
    DESTINATION_COLOR,
    GRID_LINE_COLOR,
    GRATICULE_STEP_DEG,
)


def _tip(*lines):
    """
    Build one row's tooltip HTML.

    Every pickable layer carries its own rendered `tooltip` string, and the
    deck-level template is just "{tooltip}". A single combined template
    referencing {id}, {size_km2}, {thickness} and so on cannot work: pydeck
    substitutes only the keys present in the hovered row's dict and leaves the
    rest as LITERAL braces, which is why the tooltip read "{id}{label}
    size: {size_km2} km2". One always-present key removes that failure mode
    entirely, and lets each layer word its own tooltip.

    Values are escaped because some of them (iceberg ids) can originate in a
    user-uploaded CSV.
    """
    return "<br/>".join(l for l in lines if l)


def _esc(value):
    return html.escape(str(value), quote=True)


def _lerp_color(low, high, t):
    t = max(0.0, min(1.0, t))
    return [low[i] + (high[i] - low[i]) * t for i in range(4)]


def build_ice_layer(ice_cells, visible=True):
    """PolygonLayer: sea-ice cover, colored white -> red by thickness."""
    data = []
    for cell in ice_cells:
        color = _lerp_color(ICE_COLOR_LOW, ICE_COLOR_HIGH, cell["thickness"])
        pct = cell["thickness"] * 100.0
        data.append({
            "polygon": cell["polygon"],
            "fill_color": color,
            "thickness": cell["thickness"],
            "tooltip": _tip("<b>Sea ice</b>",
                            f"Concentration: {pct:.0f}%"),
        })

    return pdk.Layer(
        "PolygonLayer",
        data=data,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255, 20],
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        visible=visible,
        id="ice-cover-layer",
    )


# Marker radius for a berg of a given area.
#
# This is a SYMBOL, not a footprint. The area-equivalent radius (sqrt(A/pi))
# is geometrically honest but compresses the real data badly: the Ross Sea
# bergs span 82-2943 km2, which maps to 5-31 km, and everything under ~250 km2
# lands on the minimum radius and looks identical. A square-root scale with a
# floor spreads the same range over 15-51 km, so a growler-sized berg and a
# tabular giant are told apart at a glance.
#
# Deliberately kept separate from the HAZARD ring, which must stay truthful -
# see build_iceberg_uncertainty_layer.
_BERG_BASE_RADIUS_M = 8000.0
_BERG_AREA_SCALE = 800.0


def _berg_radius_m(size_km2):
    try:
        area = float(size_km2 or 0.0)
    except (TypeError, ValueError):
        area = 0.0
    return _BERG_BASE_RADIUS_M + math.sqrt(max(area, 1.0)) * _BERG_AREA_SCALE


def build_iceberg_layer(icebergs, visible=True):
    """ScatterplotLayer: iceberg point markers, sized by size, colored by risk."""
    data = []
    for b in icebergs:
        color = _lerp_color(ICEBERG_RISK_LOW, ICEBERG_RISK_HIGH, b["risk"])
        data.append(
            {
                "position": [b["lon"], b["lat"]],
                "radius": _berg_radius_m(b.get("size_km2")),
                "fill_color": color,
                "id": b["id"],
                "size_km2": b["size_km2"],
                "risk": b["risk"],
                "tooltip": _tip(
                    f"<b>{_esc(b['id'])}</b>",
                    (f"Size: {b['size_km2']:.0f} km&sup2;"
                     if b.get("size_km2") else "Size: not measured"),
                    f"Risk: {b['risk']:.2f}",
                    (f"Last fix: {_esc(b['obs_date'])} "
                     f"({b['obs_age_days']} d ago)"
                     if b.get("obs_date") else ""),
                ),
            }
        )

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="position",
        get_radius="radius",
        get_fill_color="fill_color",
        get_line_color=[20, 20, 20, 200],
        line_width_min_pixels=1,
        stroked=True,
        pickable=True,
        visible=visible,
        id="iceberg-layer",
    )


def build_trajectory_layer(trajectories, visible=True):
    """PathLayer: predicted iceberg drift paths."""
    data = [
        {
            "path": path,
            "id": berg_id,
            "tooltip": _tip(f"<b>{_esc(berg_id)}</b>", "48 h drift forecast"),
        }
        for berg_id, path in trajectories.items()
    ]
    return pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_color=TRAJECTORY_COLOR,
        get_width=6,
        width_min_pixels=4,
        pickable=True,
        visible=visible,
        id="trajectory-layer",
    )


def build_trajectory_head_layer(trajectories, visible=True):
    """
    ScatterplotLayer marking the END of each drift track - the +48 h position.

    Acts as an arrowhead so the direction of drift is readable. deck.gl has no
    ArrowLayer, and PathLayer's getDashArray is only honoured with the
    PathStyleExtension (absent from pydeck 0.9.1's layer JSON), so a dashed
    line would serialise but never render. A head dot conveys direction without
    depending on either.
    """
    data = []
    for berg_id, path in (trajectories or {}).items():
        if not path or len(path) < 2:
            continue
        end = path[-1]
        data.append({
            "position": [float(end[0]), float(end[1])],
            "id": berg_id,
            "tooltip": _tip(f"<b>{_esc(berg_id)}</b>",
                            "Forecast position at +48 h"),
        })
    if not data:
        return None

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="position",
        get_radius=5000,
        radius_min_pixels=3,
        get_fill_color=TRAJECTORY_COLOR,
        get_line_color=[20, 20, 20, 200],
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=True,
        visible=visible,
        id="trajectory-head-layer",
    )


def build_ship_route_layer(route, visible=True):
    """
    PathLayer: a single ship route.

    Kept for backward compatibility. The app now renders the full Pareto set
    via build_pareto_route_layers().
    """
    if not route:
        return None
    return pdk.Layer(
        "PathLayer",
        data=[{"path": route}],
        get_path="path",
        get_color=SHIP_ROUTE_COLOR,
        get_width=4,
        width_min_pixels=2,
        pickable=False,
        visible=visible,
        id="ship-route-layer",
    )


def build_pareto_route_layers(pareto_routes, visible=True):
    """
    One PathLayer per Pareto weighting, colored from config.ROUTE_STYLES.

    Returns a LIST of layers (possibly empty). The balanced route is drawn
    LAST and thickest so it sits on top where the three overlap - on many
    transects two weightings share long stretches of path, and whichever is
    drawn last is the one the eye follows.

    `pareto_routes` is the dict from backend_stubs.run_forecast():
        {'fuel_priority': [[lon,lat],...], 'balanced': [...],
         'time_priority': [...], 'metrics': {...}, ...}
    """
    if not pareto_routes:
        return []

    layers = []
    # balanced last => on top
    draw_order = [n for n in ROUTE_ORDER if n != "balanced"] + ["balanced"]

    for name in draw_order:
        path = pareto_routes.get(name)
        if not path:
            continue
        color, _hex, label = ROUTE_STYLES[name]
        is_balanced = name == "balanced"
        metrics = (pareto_routes.get("metrics") or {}).get(name, {})
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{
                    "path": path,
                    "label": label,
                    "distance_km": metrics.get("distance"),
                    "time_h": metrics.get("time_h"),
                    "fuel_t": metrics.get("fuel_t"),
                    "ice_severity": metrics.get("ice_severity"),
                    "tooltip": _tip(
                        f"<b>{_esc(label)}</b>"
                        + (" &middot; recommended" if is_balanced else ""),
                        (f"{metrics['distance']:.0f} km &middot; "
                         f"{metrics['time_h']:.1f} h &middot; "
                         f"{metrics['fuel_t']:.1f} t"
                         if metrics else ""),
                        (f"Ice: {_esc(metrics['ice_severity'])}"
                         if metrics.get("ice_severity") else ""),
                    ),
                }],
                get_path="path",
                get_color=color,
                get_width=6 if is_balanced else 4,
                width_min_pixels=3 if is_balanced else 2,
                pickable=True,
                visible=visible,
                id=f"route-{name}-layer",
            )
        )
    return layers


# Exclusion radius the route optimiser actually applies around a tracked berg:
# a 50 km base widened by twice the model's 1-sigma. Mirrored here so the ring
# drawn on the map is the same footprint the search avoided, rather than a
# decorative circle.
_TRACKED_BASE_RADIUS_KM = 50.0
_SIGMA_MULTIPLIER = 2.0


def build_iceberg_uncertainty_layer(icebergs, visible=True):
    """
    ScatterplotLayer: the positional-uncertainty footprint around each tracked
    berg - the area the route optimiser treats as hazardous.

    Radius = 50 km base + 2 x uncertainty_km, matching
    route_optimiser.tracked_berg_hazards(). (The brief suggested a flat 500 km
    placeholder; at this zoom that is wider than the Ross Sea itself and would
    hide the very routes it is meant to contextualise, so the real exclusion
    radius is used instead.)
    """
    if not icebergs:
        return None

    data = []
    for b in icebergs:
        sigma = b.get("uncertainty_km")
        if sigma is None:
            # Same Tier-1 proxy backend_stubs passes to calculate_routes()
            sigma = 0.5 * (1.0 + float(b.get("risk", 0.5)))
        radius_km = _TRACKED_BASE_RADIUS_KM + _SIGMA_MULTIPLIER * float(sigma)
        data.append({
            "position": [b["lon"], b["lat"]],
            "radius": radius_km * 1000.0,     # pydeck wants metres
            "id": b.get("id", "?"),
            "radius_km": round(radius_km, 1),
            "obs_age_days": b.get("obs_age_days"),
        })

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="position",
        get_radius="radius",
        get_fill_color=UNCERTAINTY_RING_COLOR,
        get_line_color=[255, 140, 0, 120],
        line_width_min_pixels=1,
        stroked=True,
        filled=True,
        pickable=False,
        visible=visible,
        id="iceberg-uncertainty-layer",
    )


def build_ship_markers_layer(ship_pos, destination, visible=True):
    """ScatterplotLayer: ship current position + destination markers."""
    data = []
    if ship_pos:
        data.append(
            {
                "position": [ship_pos["lon"], ship_pos["lat"]],
                "fill_color": SHIP_POS_COLOR,
                "radius": 6000,
                "label": "Ship (current)",
                "tooltip": _tip("<b>Ship (current)</b>",
                                f"{ship_pos['lat']:.3f}, {ship_pos['lon']:.3f}"),
            }
        )
    if destination:
        data.append(
            {
                "position": [destination["lon"], destination["lat"]],
                "fill_color": DESTINATION_COLOR,
                "radius": 6000,
                "label": destination.get("name") or "Destination",
                "tooltip": _tip(
                    f"<b>{_esc(destination.get('name') or 'Destination')}</b>",
                    f"{destination['lat']:.3f}, {destination['lon']:.3f}"),
            }
        )
    if not data:
        return None
    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="position",
        get_radius="radius",
        get_fill_color="fill_color",
        get_line_color=[20, 20, 20, 220],
        line_width_min_pixels=2,
        stroked=True,
        pickable=True,
        visible=visible,
        id="ship-markers-layer",
    )


def build_graticule_layer(step=GRATICULE_STEP_DEG, visible=True):
    """LineLayer: simple lat/lon reference grid over the Ross Sea bounds."""
    lines = []

    # latitude lines (horizontal), spanning each longitude segment
    lat = LAT_MIN
    while lat <= LAT_MAX:
        for seg_west, seg_east in LON_SEGMENTS:
            lines.append({"start": [seg_west, lat], "end": [seg_east, lat]})
        lat += step

    # longitude lines (vertical), spanning the full lat range
    for seg_west, seg_east in LON_SEGMENTS:
        lon = seg_west
        while lon <= seg_east:
            lines.append({"start": [lon, LAT_MIN], "end": [lon, LAT_MAX]})
            lon += step

    return pdk.Layer(
        "LineLayer",
        data=lines,
        get_source_position="start",
        get_target_position="end",
        get_color=GRID_LINE_COLOR,
        get_width=1,
        pickable=False,
        visible=visible,
        id="graticule-layer",
    )
