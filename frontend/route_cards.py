"""
route_cards.py
==============
Renders the three-up Pareto route recommendation cards, and the red banner
shown when no route exists.

Presentation only - it reads the dict produced by
backend_stubs.run_forecast() and never computes anything itself.
"""

import streamlit as st

from config import ROUTE_STYLES, ROUTE_ORDER

_SEVERITY_COLOR = {
    "low": "#56c288",
    "moderate": "#e8a87c",
    "high": "#e74c3c",
}


def _card_html(name, metrics, is_recommended):
    color, hex_color, label = ROUTE_STYLES[name]
    border = hex_color if is_recommended else "#2b3442"
    width = "2px" if is_recommended else "1px"

    sev = str(metrics.get("ice_severity", "n/a")).lower()
    sev_col = _SEVERITY_COLOR.get(sev, "#8fa2b8")

    badge = (
        f"<div style='font-size:0.58rem;color:#12161d;background:{hex_color};"
        f"display:inline-block;padding:0.05rem 0.35rem;border-radius:3px;"
        f"font-weight:700;letter-spacing:0.04em;margin-bottom:0.3rem;'>"
        f"RECOMMENDED</div>"
        if is_recommended else
        "<div style='height:0.95rem;'></div>"
    )

    def row(value, unit, sub):
        return (
            f"<div style='margin-bottom:0.28rem;'>"
            f"<span style='font-size:1.02rem;font-weight:700;color:#f2f4f7;'>{value}</span>"
            f"<span style='font-size:0.64rem;color:#8fa2b8;'> {unit}</span><br>"
            f"<span style='font-size:0.6rem;color:#7c8ca0;letter-spacing:0.03em;'>{sub}</span>"
            f"</div>"
        )

    return f"""
    <div style="
        border:{width} solid {border};
        border-radius:7px;
        padding:0.5rem 0.55rem;
        background:#161c25;
        height:100%;
    ">
      {badge}
      <div style="font-size:0.64rem;font-weight:700;letter-spacing:0.05em;
                  color:{hex_color};margin-bottom:0.4rem;">{label}</div>
      {row(metrics.get('distance', '-'), 'km', 'DISTANCE')}
      {row(metrics.get('time_h', '-'), 'h', 'TIME')}
      {row(metrics.get('fuel_t', '-'), 't', 'FUEL')}
      <div style="margin-top:0.35rem;">
        <span style="font-size:0.6rem;color:#7c8ca0;letter-spacing:0.03em;">ICE</span><br>
        <span style="font-size:0.76rem;font-weight:700;color:{sev_col};
                     text-transform:uppercase;">{sev}</span>
      </div>
    </div>
    """


def render_impossible_banner(polar_class, reason=""):
    """Red banner shown when calculate_routes() reports no viable route."""
    detail = (
        f"<div style='font-size:0.68rem;color:#c98b86;margin-top:0.35rem;"
        f"line-height:1.35;'>Backend reason: {reason}</div>"
        if reason else ""
    )
    st.markdown(
        f"""
        <div style="
            border:1px solid #c0392b;
            border-left:4px solid #c0392b;
            background-color:rgba(192,57,43,0.14);
            color:#e74c3c;
            border-radius:6px;
            padding:0.65rem 0.8rem;
            margin-bottom:0.75rem;
            font-size:0.78rem;
            line-height:1.4;
        ">
          &#9888; <b>No viable route for {polar_class} on this date.</b><br>
          Consider upgrading Polar Class or selecting a different destination.
          {detail}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_route_cards(pareto_routes):
    """
    Three-up route recommendation cards. The balanced route carries a gold
    border and a RECOMMENDED badge.
    """
    metrics = (pareto_routes or {}).get("metrics") or {}
    if not metrics:
        return

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Route Recommendations</h4>", unsafe_allow_html=True)

    present = [n for n in ROUTE_ORDER if n in metrics]
    cols = st.columns(len(present), gap="small")
    for col, name in zip(cols, present):
        with col:
            st.markdown(
                _card_html(name, metrics[name], is_recommended=(name == "balanced")),
                unsafe_allow_html=True,
            )

    missing = [n for n in ROUTE_ORDER if n not in metrics]
    if missing:
        pretty = ", ".join(ROUTE_STYLES[m][2].title() for m in missing)
        st.markdown(
            f"<div style='font-size:0.66rem;color:#e8a87c;margin-top:0.5rem;'>"
            f"No path found for: {pretty}.</div>",
            unsafe_allow_html=True,
        )

    # When several weightings return the same path the Pareto front has
    # collapsed - one route dominates on every objective. Say so, rather than
    # letting three identical cards imply a trade-off that does not exist.
    sigs = {}
    for n in present:
        m = metrics[n]
        sigs.setdefault((m.get("distance"), m.get("time_h"), m.get("fuel_t")),
                        []).append(n)
    dupes = [v for v in sigs.values() if len(v) > 1]
    if dupes and len(present) > 1:
        st.markdown(
            "<div style='font-size:0.66rem;color:#8fa2b8;margin-top:0.5rem;"
            "line-height:1.35;'>Identical figures mean one route is optimal "
            "under every weighting on this transect &mdash; the Pareto front "
            "has collapsed to a single point, not a display error.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
