# AI-Enabled Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System

**Smart India Hackathon 2026 — SIH26059**
**Team:** Dadi & Gagnamagnid · **Team ID:** SIH050
**Theme:** Transportation & Logistics · **Category:** Software

---

## Overview

This project is an AI/ML-enabled decision-support platform for Antarctic research vessels.
It uses **satellite, oceanographic, and meteorological datasets** to:

1. **Forecast sea-ice concentration**
2. **Predict iceberg trajectories**
3. **Recommend safe, fuel-efficient navigation routes**

The system combines **physics-based baselines** with **AI residual correction** — physics
provides a reliable starting estimate, and machine learning corrects the gap between that
estimate and real observed conditions. Every output carries a confidence/uncertainty level,
and a physics validation gate rejects any prediction that isn't physically plausible before
it reaches the routing engine.

> Physics provides the baseline • AI learns the gap • Routing converts forecasts into decisions

---

## How It Works

```
New Satellite / Oceanographic / Weather Data
        │
        ▼
Sea-Ice + Iceberg Forecast (with confidence)
        │
        ▼
Physics Validation Gate  →  rejects physically impossible predictions
        │
        ▼
Route Optimiser (2–3 options: fastest / safest / most fuel-efficient)
        │
        ▼
Crew Reviews & Approves  →  human-in-the-loop, explainable alternatives
        │
        └──── repeats every 6–12 hours as new data arrives ────┘
```

### 1. Sea-Ice Forecasting
- **Input:** NSIDC satellite data (sea-ice concentration) + ERA5 meteorological data (wind)
- **Method:** Stefan's Law (`seaice_physics_baseline.py`) estimates ice growth from
  temperature; an LSTM model (`seaice_lstm.py`) learns the gap between this physics
  estimate and real observations, then corrects it.
- **Output:** Forecasted ice concentration + prediction confidence (see
  `seaice_validation.json` / `seaice_training.png` for validation results)

### 2. Iceberg Trajectory Prediction
- **Input:** BYU/NIC tracked iceberg positions + CMEMS oceanographic data (ocean currents) +
  ERA5 meteorological data (wind)
- **Method:** Bigg's momentum-balance model (`iceberg_physics_baseline.py`) predicts drift
  from currents and wind; an LSTM correction (`iceberg_lstm.py`) is trained primarily on our
  best-tracked iceberg (B38) for reliability. Untracked/uncharted bergs are handled
  separately (`uncharted_berg_handler.py`).
- **Output:** Predicted iceberg path + confidence range, checked against real-world physical
  limits (e.g. maximum plausible drift speed, grounding/bathymetry conflicts)

### 3. Safe & Fuel-Efficient Routing
- **Input:** Forecasts from the sea-ice and iceberg modules + ship class and IMO Polar Code
  constraints
- **Method:** A Pareto multi-objective optimiser (`route_optimiser.py`) — using the
  **Lindqvist ice-resistance model** (`lindqvist_model.py`) and a Polar Class ↔ ice-speed
  lookup table (`pc_ice_speed_model.csv`) to build a fuel/time/risk cost surface
  (`cost_surface.py`) — balances fuel, time, and safety, recalculating every 6–12 hours as
  new data arrives.
- **Output:** 2–3 route options (fastest / safest / most fuel-efficient) for the crew to
  choose from — not a single forced answer

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Backend** | Python, PyTorch, NumPy, SciPy, Pandas |
| **Frontend** | Streamlit, pydeck, CARTO |

<!-- TODO: add versions once pinned, e.g. Python 3.x, PyTorch x.x -->

---

## Datasets Used

| Dataset | Source | In repo as | Notes |
|---|---|---|---|
| Sea-ice concentration | NSIDC Sea Ice Index v4.0 | *(loaded by `seaice_physics_baseline.py`)* | Passive microwave, 25 km resolution |
| Iceberg tracking | BYU/NIC Antarctic Iceberg Tracking Database v8.0 | `ross_sea_icebergs_2021_2025.csv` | Ross Sea filtered, 2021–2025 |
| Meteorological (wind) | ERA5 surface analysis (NCAR GDEX) | *(loaded by physics baseline scripts)* | Dataset d633000, hourly, 10m wind (U+V) |
| Ocean currents | CMEMS — Ocean Surface Current Analyses Real-time (OSCAR) | *(routing/oceanographic forcing)* | — |

Per-iceberg physics residuals (used for LSTM training/generalisation testing) are broken
out individually: `iceberg_physics_residuals_{b22a,b22f,b22g,b38,b42,b46,b47,b50}.csv`,
plus a combined `iceberg_physics_residuals_all.csv`. B38 is the primary training berg
(highest observation count); the rest are used to test generalisation.

<!-- TODO: add exact date ranges / geographic bounds for the NSIDC, ERA5, and CMEMS
     datasets, and confirm whether raw files are fetched at runtime or expected locally -->

---

## Project Structure

```
.
├── frontend/                             # Streamlit + pydeck dashboard
│
├── seaice_physics_baseline.py            # Stefan's Law / freezing-degree-day baseline
├── seaice_lstm.py                        # Sea-ice residual LSTM (training + inference)
├── seaice_model.pt                       # Trained sea-ice residual model weights
├── seaice_training.png                   # Training curve
├── seaice_validation.json                # Validation / skill-score results
│
├── iceberg_physics_baseline.py           # Bigg's momentum-balance drift model
├── iceberg_lstm.py                       # Iceberg residual LSTM (training + inference)
├── iceberg_lstm.pt                       # Trained iceberg residual model weights
├── iceberg_lstm_training.png             # Training curve
├── iceberg_lstm_generalisation.csv       # Transfer-berg generalisation results
├── iceberg_physics_residuals*.csv        # Per-berg residuals (B22A, B22F, B22G, B38,
│                                          # B42, B46, B47, B50) + a combined "_all" file
├── uncharted_berg_handler.py             # Handling for untracked/uncharted icebergs
├── ross_sea_icebergs_2021_2025.csv       # BYU/NIC iceberg tracking data, Ross Sea filter
│
├── lindqvist_model.py                    # Lindqvist ice-resistance model (routing input)
├── pc_ice_speed_model.csv                # Polar Class ↔ safe ice-speed lookup table
├── cost_surface.py                       # Builds the fuel/time/risk cost surface
├── route_optimiser.py                    # Pareto multi-objective route optimiser
│
├── grid_utils.py                         # Shared geospatial grid utilities (incl.
│                                          # antimeridian-safe handling)
├── test_backend.py                       # Backend test suite
│
├── .gitattributes
├── .gitignore
└── README.md
```

> Note: model weights (`*.pt`) and result files (`*.csv`, `*.json`, `*.png`) are committed
> directly to the repo for reproducibility. If the repo grows, consider moving large
> binaries (`.pt` files) to [Git LFS](https://git-lfs.github.com/).

---

## Getting Started

### Prerequisites
- Python 3.x
- pip / virtualenv

### Installation
```bash
git clone https://github.com/vader122005/antartic.git
cd antartic
pip install -r requirements.txt
```

<!-- TODO: add a requirements.txt to the repo if not already present -->

### Running the Backend Pipeline

```bash
# Sea-ice: physics baseline → LSTM residual correction
python seaice_physics_baseline.py
python seaice_lstm.py

# Iceberg: physics baseline → LSTM residual correction
python iceberg_physics_baseline.py
python iceberg_lstm.py

# Routing: build cost surface, then optimise
python cost_surface.py
python route_optimiser.py
```

<!-- TODO: confirm exact CLI args / entry-point functions for each script above -->

### Running the Dashboard
```bash
cd frontend
streamlit run app.py
```

<!-- TODO: confirm frontend entry-point filename and add any required env vars,
     API keys, or data download steps -->

### Running Tests
```bash
python test_backend.py
```

---

## Key Design Principles

- **Physics-anchored hybrid forecasting** — AI corrects what physics gets wrong; physics
  constrains what AI predicts (e.g. rejecting grounding/bathymetry conflicts, implausible
  drift speeds, or ice-concentration contradictions before a forecast reaches routing).
  Removing the physics link has been shown (independently, by IDRIFTNET) to increase
  error 5–30×.
- **Decision support, not autonomy** — the system produces explainable route alternatives
  for a human crew to review and approve, not a single automated decision.
- **Calibrated uncertainty everywhere** — every forecast and prediction carries a confidence
  level, not just a point estimate.
- **Modular pipeline** — Sea-Ice Forecasting → Iceberg Trajectory Prediction → Route
  Optimisation are independently built, tested, and improved.

---

## Research Basis & References

- Bigg et al. (1997) — Classical iceberg momentum-balance model (force-balance baseline)
- Wagner et al. (2017) — Analytical solution to the Bigg momentum model
- Lichey & Hellmer (2001) — Sea-ice drag regimes for iceberg drift
- Barbosa Aguiar et al. (2025) — IDRIFTNET: physics-anchored residual ML, Antarctic
  validation ([arXiv:2507.00036](https://arxiv.org/abs/2507.00036))
- Andersson et al. (2021) — Probabilistic ice forecasting
- Lu et al. (2020) — Multi-objective polar routing
- IMO (2017) — Polar Code (hard operational constraint)
- IACS (2016) — Polar Class (vessel capability definitions)



Built for Smart India Hackathon 2026, Problem Statement SIH26059, under the Ministry of
Earth Sciences (MoES) — National Centre for Polar and Ocean Research (NCPOR).
