"""
uncertainty_panel.py
====================
Two presentation components for the right-hand panel:

  render_accuracy_panel()  - validated model-skill figures
  render_disclaimer()      - the mandatory IMO/SOLAS decision-support notice

Every number in the accuracy panel is a FIXED, VALIDATED figure measured
during backend development. None of it is random or recomputed per run: a
confidence figure that changes when you click something is worse than no
figure at all.

The disclaimer is mandatory on every screen state per the project spec
(SOLAS V/34, IMO Polar Code Ch. 11). It is deliberately not collapsible and
not behind a toggle.
"""

import streamlit as st

# --- Validated skill figures (see seaice_validation.json /
#     iceberg_lstm_generalisation.csv in the backend folder) -----------------
SEA_ICE_COVERAGE_PCT = 21.5     # share of grid cells inside the ERA5 wind box
SEA_ICE_SKILL_IN_ZONE = 2.7     # % RMSE improvement vs physics, within zone
SEA_ICE_SKILL_NET = 1.4         # % improvement across the full grid
ICEBERG_SKILL_B38 = 5.0         # % improvement on the B38 validation split
ICEBERG_DRIFT_PAIRS = 69        # genuine (deduplicated) B38 drift pairs

_ITEM = (
    "<div style='margin:0.15rem 0 0.15rem 0.15rem;font-size:0.74rem;"
    "color:#b8c6d6;line-height:1.35;'>&bull; {}</div>"
)
_GROUP = (
    "<div style='font-size:0.72rem;font-weight:700;letter-spacing:0.05em;"
    "color:{color};text-transform:uppercase;margin:0.6rem 0 0.2rem 0;'>{title}</div>"
)


def _group(title, color, items):
    html = _GROUP.format(title=title, color=color)
    html += "".join(_ITEM.format(i) for i in items)
    return html


def render_accuracy_panel():
    """Render the 'Forecast Accuracy' card. Static validated figures only."""
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown("<h4>Forecast Accuracy</h4>", unsafe_allow_html=True)

    body = ""

    body += _group(
        "Sea-ice LSTM", "#6fa8dc",
        [
            f"Coverage: <b>{SEA_ICE_COVERAGE_PCT}%</b> of grid cells "
            f"(ERA5 wind coverage zone)",
            f"Skill vs physics-only: <b>+{SEA_ICE_SKILL_IN_ZONE}%</b> within "
            f"coverage zone, <b>+{SEA_ICE_SKILL_NET}%</b> net across full grid",
            "Validation period: held-out <b>2024&ndash;25</b> season",
            "Outside coverage: physics-only "
            "(conduction / ocean heat-flux thermodynamics)",
        ],
    )

    body += _group(
        "Iceberg trajectory", "#e8a87c",
        [
            f"Primary berg (B38): <b>+{ICEBERG_SKILL_B38}%</b> skill vs "
            f"physics-only ({ICEBERG_DRIFT_PAIRS} genuine drift pairs)",
            "Transfer bergs (7 bergs): approximately <b>break-even</b> "
            "(limited training data)",
            "Validation gate: LSTM applied only when it outperforms physics "
            "on the validation set",
            "Known gap: ocean currents set to zero (CMEMS skipped) "
            "&mdash; largest error source",
        ],
    )

    body += _group(
        "Route optimiser", "#8fd6a0",
        [
            "Lindqvist (1989) ice-resistance physics",
            "Engine power generalised per PC class (assumption flagged)",
            "<b>6&ndash;12 h</b> dynamic recalculation recommended in operations",
        ],
    )

    st.markdown(body, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


DISCLAIMER_TEXT = (
    "&#9888; <b>DECISION SUPPORT ONLY</b> &mdash; This system provides "
    "navigational decision support only. The Master retains full authority and "
    "responsibility for all navigation decisions under SOLAS regulation V/34 "
    "and IMO Polar Code Chapter 11. Model output does not constitute a safe "
    "passage guarantee."
)


def render_disclaimer():
    """
    Mandatory legal notice. Must render in EVERY screen state - before any
    forecast, after a successful forecast, and when no route exists.
    """
    st.markdown(
        f"""
        <div style="
            border: 1px solid #c0392b;
            border-left: 4px solid #c0392b;
            background-color: rgba(192, 57, 43, 0.08);
            color: #e74c3c;
            font-size: 0.75rem;
            line-height: 1.4;
            border-radius: 6px;
            padding: 0.6rem 0.75rem;
            margin-top: 0.25rem;
            margin-bottom: 0.75rem;
        ">{DISCLAIMER_TEXT}</div>
        """,
        unsafe_allow_html=True,
    )
