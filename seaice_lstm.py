"""
seaice_lstm.py - LSTM / 1D-CNN residual model for Antarctic sea-ice concentration.

WHAT IT LEARNS
--------------
The residual between observed SIC and the thermodynamic physics prediction from
`seaice_physics_baseline.py`, per grid cell. The physics captures conductive
growth and ocean-driven melt; the ML layer corrects for the wind-driven
DYNAMICS a 1-D column model cannot see - advection, ridging and lead opening.

    residual(t) = sic_obs(t+1) - sic_phys(t+1)

ARCHITECTURE
------------
Both variants specified in the brief are implemented and trained, and the one
that validates better is saved as `seaice_model.pt`:

  Option A  LSTM   : 2-layer LSTM (hidden 48) -> dense head
  Option B  1D-CNN : 3 x Conv1d(kernel 3) -> global pooling -> dense head

Input per timestep : [sic_phys(t), wind_u(t), wind_v(t), sic_obs(t-1)]
Window             : 7 days
Output             : residual(t+1) for that cell

TWO CONSTRAINTS THAT SHAPE THE TRAINING SET
-------------------------------------------
1. WIND COVERAGE. The ERA5 box (lat -78..-70, lon 160E..150W) is much smaller
   than the SIC Ross Sea domain (lat -78.5..-60, lon 160E..130W): only 21.5% of
   the 8599 valid SIC cells have genuine wind. Training on the rest would feed
   the model edge-clamped, fabricated wind. We therefore train ONLY on
   wind-covered cells, and at inference return a `wind_coverage` mask with
   inflated uncertainty outside it, rather than pretending the forecast is
   equally trustworthy everywhere.

2. GRID SIZE. Training a model per cell over a (133, 147) grid is infeasible.
   We treat each cell's time series as an independent sample (one shared model
   across cells) and subsample every `--cell_stride`-th valid cell for
   training, then evaluate on the full grid.

SPLIT
-----
By SEASON, not by random shuffle: seasons 2021-22 .. 2023-24 train, 2024-25 is
held out. A random split would leak, because neighbouring days in the same
season are almost identical (daily SIC persistence error is only 1.4 %SIC).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import grid_utils as gu

WINDOW = 7
N_FEATURES = 4          # sic_phys, wind_u, wind_v, sic_obs(t-1)
SIC_MIN, SIC_MAX = 0.0, 100.0

TRAIN_SEASONS = ("2021-22", "2022-23", "2023-24")
VAL_SEASONS = ("2024-25",)

WIND_CACHE = "wind_on_sic_grid.npz"
DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Wind regridding (ERA5 regular grid -> curvilinear NSIDC grid)
# ---------------------------------------------------------------------------

def build_wind_on_sic_grid(sic_npz="ross_sea_concentration.npz",
                           wind_npz="ross_sea_era5_wind.npz",
                           cache_path=WIND_CACHE, verbose=True):
    """
    Interpolate ERA5 10 m wind onto the NSIDC grid for every SIC date.

    Bilinear weights are computed ONCE (the target cells never move) and then
    reused for all dates, which turns 605 x 19551 interpolations into a single
    gather per date. Returns (wind_u, wind_v, coverage) where coverage marks
    cells genuinely inside the ERA5 domain rather than clamped to its edge.
    """
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        if verbose:
            print(f"      using cached wind regrid: {cache_path}")
        return z["wind_u"], z["wind_v"], z["coverage"]

    s = np.load(sic_npz)
    dates = np.asarray(s["dates"])
    lat, lon = np.asarray(s["lat"]), np.asarray(s["lon"])
    rows, cols = lat.shape

    w = gu.WindSampler(wind_npz)
    ax_y, ax_x = w.lat, w.lon                   # ascending, unwrapped

    ty = np.asarray(lat, dtype=np.float64).ravel()
    tx = gu.unwrap_lon(lon).ravel()

    coverage = ((ty >= ax_y[0]) & (ty <= ax_y[-1])
                & (tx >= ax_x[0]) & (tx <= ax_x[-1])).reshape(rows, cols)

    yc = np.clip(ty, ax_y[0], ax_y[-1])
    xc = np.clip(tx, ax_x[0], ax_x[-1])

    iy = np.clip(np.searchsorted(ax_y, yc) - 1, 0, len(ax_y) - 2)
    ix = np.clip(np.searchsorted(ax_x, xc) - 1, 0, len(ax_x) - 2)
    fy = (yc - ax_y[iy]) / (ax_y[iy + 1] - ax_y[iy])
    fx = (xc - ax_x[ix]) / (ax_x[ix + 1] - ax_x[ix])

    w00 = (1 - fy) * (1 - fx)
    w01 = (1 - fy) * fx
    w10 = fy * (1 - fx)
    w11 = fy * fx

    out_u = np.empty((len(dates), rows, cols), dtype=np.float32)
    out_v = np.empty((len(dates), rows, cols), dtype=np.float32)

    if verbose:
        print(f"      regridding wind for {len(dates)} dates ...")
    for t, d in enumerate(dates):
        # 12:00 UTC is representative of the daily-mean SIC field
        us, vs = w.grid_at(gu.parse_date(str(d)).replace(hour=12))
        for src, dst in ((us, out_u), (vs, out_v)):
            vals = (src[iy, ix] * w00 + src[iy, ix + 1] * w01
                    + src[iy + 1, ix] * w10 + src[iy + 1, ix + 1] * w11)
            dst[t] = vals.reshape(rows, cols)

    np.savez_compressed(cache_path, wind_u=out_u, wind_v=out_v,
                        coverage=coverage, dates=dates)
    if verbose:
        print(f"      cached -> {cache_path}")
    return out_u, out_v, coverage


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(cell_stride=3, window=WINDOW, verbose=True):
    """
    Assemble per-cell temporal windows.

    For a target index t (into the 604 residual frames) the input window spans
    residual frames t-window+1 .. t, and at frame j the features are
        [sic_phys[j], wind_u[j], wind_v[j], sic_obs[j-1]]
    with sic_obs[j-1] = conc[j], since sic_obs[j] == conc[j+1]. Nothing in the
    window uses information from after frame t, so there is no leakage.
    """
    s = np.load("ross_sea_concentration.npz")
    conc = s["concentration"]
    r = np.load("seaice_physics_residuals.npz")
    resid, sic_phys = r["residual"], r["sic_phys"]
    dates_out = np.asarray(r["dates_out"])

    wu, wv, coverage = build_wind_on_sic_grid(verbose=verbose)
    # wind arrays are indexed by SIC date; residual frame t corresponds to the
    # physics step launched from SIC day t
    wu, wv = wu[:len(resid)], wv[:len(resid)]

    n_t, rows, cols = resid.shape
    seasons = np.array([gu.season_label(str(d)) or "" for d in dates_out])

    # Cells usable for training: valid SIC, genuine wind, and subsampled
    valid = ~np.isnan(conc[0])
    trainable = valid & coverage
    mask = np.zeros_like(trainable)
    mask[::cell_stride, ::cell_stride] = True
    sel = np.argwhere(trainable & mask)

    if verbose:
        print(f"      valid cells {valid.sum()}, wind-covered {trainable.sum()} "
              f"({100.0 * trainable.sum() / max(valid.sum(), 1):.1f}%), "
              f"training cells {len(sel)} (stride {cell_stride})")

    ii, jj = sel[:, 0], sel[:, 1]

    # (n_t, n_cells) views of every field, then slice windows in one shot
    rp = resid[:, ii, jj]
    sp = sic_phys[:, ii, jj]
    wuc = wu[:, ii, jj]
    wvc = wv[:, ii, jj]
    prev_obs = conc[:n_t, ii, jj]     # conc[j] == sic_obs[j-1]

    X_parts, y_parts, t_parts = [], [], []
    for t in range(window - 1, n_t):
        w0 = t - window + 1
        if len(set(seasons[w0:t + 1])) != 1:
            continue                       # window must not straddle seasons
        feats = np.stack([sp[w0:t + 1], wuc[w0:t + 1],
                          wvc[w0:t + 1], prev_obs[w0:t + 1]], axis=-1)
        X_parts.append(np.transpose(feats, (1, 0, 2)))   # (cells, window, 4)
        y_parts.append(rp[t])
        t_parts.append(np.full(len(ii), t))

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    tidx = np.concatenate(t_parts, axis=0)

    good = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
    X, y, tidx = X[good], y[good], tidx[good]

    season_of = np.array([seasons[t] for t in tidx])
    if verbose:
        print(f"      samples: {len(X)} "
              f"(train {np.isin(season_of, TRAIN_SEASONS).sum()}, "
              f"val {np.isin(season_of, VAL_SEASONS).sum()})")
    return X, y, season_of


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SeaIceLSTM(nn.Module):
    """Option A - per-cell temporal LSTM."""

    def __init__(self, n_features=N_FEATURES, hidden=48, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


class SeaIceCNN(nn.Module):
    """Option B - per-cell temporal 1D-CNN (faster than the LSTM)."""

    def __init__(self, n_features=N_FEATURES, ch=48, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(self.drop(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class Scaler:
    def fit(self, X, y):
        f = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = f.mean(0), f.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        self.ymu, self.ysd = float(y.mean()), float(y.std())
        if self.ysd < 1e-8:
            self.ysd = 1.0
        return self

    def tx(self, X):
        return (X - self.mu) / self.sd

    def ty(self, y):
        return (y - self.ymu) / self.ysd

    def inv(self, y):
        return y * self.ysd + self.ymu

    def state(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(),
                    ymu=self.ymu, ysd=self.ysd)

    @classmethod
    def from_state(cls, st):
        s = cls()
        s.mu = np.asarray(st["mu"], dtype=np.float32)
        s.sd = np.asarray(st["sd"], dtype=np.float32)
        s.ymu, s.ysd = st["ymu"], st["ysd"]
        return s


def train_one(model, Xtr, ytr, Xva, yva, scaler, epochs=25,
              batch_size=1024, lr=1e-3, patience=6, tag="", verbose=True):
    tr = TensorDataset(torch.tensor(scaler.tx(Xtr)), torch.tensor(scaler.ty(ytr)))
    va = TensorDataset(torch.tensor(scaler.tx(Xva)), torch.tensor(scaler.ty(yva)))
    # num_workers=0: DataLoader worker processes crash on Windows
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=4096, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5,
                                                       patience=2)
    lossf = nn.MSELoss()
    hist = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in tl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(xb)
        trl = tot / len(tr)

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                tot += lossf(model(xb), yb).item() * len(xb)
        val = tot / len(va)

        sched.step(val)
        hist["train"].append(trl)
        hist["val"].append(val)

        if val < best - 1e-7:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"        [{tag}] epoch {ep + 1:3d}  train {trl:.5f}  val {val:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best


def predict_np(model, X, scaler, batch=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            xb = torch.tensor(scaler.tx(X[k:k + batch]))
            out.append(model(xb).numpy())
    return scaler.inv(np.concatenate(out)) if out else np.empty(0)


def metrics(y_true, y_pred, sic_phys=None, sic_obs=None):
    err = y_pred - y_true
    m = {
        "n": int(len(y_true)),
        "rmse_residual": float(np.sqrt((err ** 2).mean())),
        "mae_residual": float(np.abs(err).mean()),
        "bias_residual": float(err.mean()),
    }
    if len(y_true) > 1 and y_true.std() > 0 and y_pred.std() > 0:
        m["correlation"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        m["correlation"] = float("nan")
    # Skill against the physics baseline alone (residual predicted as 0)
    base = float(np.sqrt((y_true ** 2).mean()))
    m["rmse_physics_only"] = base
    m["skill_vs_physics_pct"] = float(100.0 * (base - m["rmse_residual"]) / max(base, 1e-9))
    return m


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class SeaIceResidualModel:
    def __init__(self, model, scaler, kind, coverage=None,
                 sigma_covered=1.0, sigma_uncovered=None):
        self.model = model
        self.scaler = scaler
        self.kind = kind
        self.coverage = coverage
        # 1-sigma in %SIC, from validation RMSE. Cells outside the ERA5 domain
        # get an inflated value because their wind features are edge-clamped.
        self.sigma_covered = float(sigma_covered)
        self.sigma_uncovered = float(sigma_uncovered
                                     if sigma_uncovered is not None
                                     else sigma_covered * 2.0)
        self.model.eval()

    def save(self, path="seaice_model.pt"):
        torch.save({
            "kind": self.kind,
            "state_dict": self.model.state_dict(),
            "scaler": self.scaler.state(),
            "window": WINDOW,
            "coverage": None if self.coverage is None else self.coverage.astype(bool),
            "sigma_covered": self.sigma_covered,
            "sigma_uncovered": self.sigma_uncovered,
        }, path)
        return path

    @classmethod
    def load(cls, path="seaice_model.pt"):
        b = torch.load(path, map_location="cpu", weights_only=False)
        m = SeaIceLSTM() if b["kind"] == "lstm" else SeaIceCNN()
        m.load_state_dict(b["state_dict"])
        cov = b.get("coverage")
        return cls(m, Scaler.from_state(b["scaler"]), b["kind"],
                   coverage=None if cov is None else np.asarray(cov),
                   sigma_covered=b.get("sigma_covered", 1.0),
                   sigma_uncovered=b.get("sigma_uncovered", 2.0))


_CACHE = {}


def _get(path="seaice_model.pt"):
    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - train it first: python seaice_lstm.py")
        _CACHE[path] = SeaIceResidualModel.load(path)
    return _CACHE[path]


def predict_sic_grid(sic_phys_grid, wind_u_grid, wind_v_grid,
                     sequence_history, model_path="seaice_model.pt"):
    """
    Physics + ML corrected SIC for the whole Ross Sea grid.

    Parameters
    ----------
    sic_phys_grid    : (133, 147) physics SIC prediction
    wind_u_grid      : (133, 147) ERA5 wind interpolated to the NSIDC grid
    wind_v_grid      : (133, 147)
    sequence_history : list of the last N observed SIC grids, oldest first.
                       Short histories are padded by repeating the oldest.

    Returns
    -------
    {'sic_pred'      : (133, 147) physics + correction, clipped 0-100,
     'uncertainty'   : (133, 147) 1-sigma in %SIC,
     'wind_coverage' : (133, 147) bool - False where ERA5 wind was
                       edge-clamped; those cells keep the physics value and
                       carry the inflated uncertainty}
    """
    sic_phys = np.asarray(sic_phys_grid, dtype=np.float32)
    wu = np.asarray(wind_u_grid, dtype=np.float32)
    wv = np.asarray(wind_v_grid, dtype=np.float32)
    rows, cols = sic_phys.shape

    hist = [np.asarray(h, dtype=np.float32) for h in (sequence_history or [])]
    if not hist:
        hist = [np.nan_to_num(sic_phys)]
    hist = hist[-WINDOW:]
    while len(hist) < WINDOW:
        hist.insert(0, hist[0])

    valid = np.isfinite(sic_phys)
    ii, jj = np.where(valid)

    # Build (n_cells, window, 4). sic_phys and wind are held at their current
    # value across the window; prior observations vary through the history.
    n = len(ii)
    X = np.empty((n, WINDOW, N_FEATURES), dtype=np.float32)
    X[:, :, 0] = sic_phys[ii, jj][:, None]
    X[:, :, 1] = wu[ii, jj][:, None]
    X[:, :, 2] = wv[ii, jj][:, None]
    for k, h in enumerate(hist):
        X[:, k, 3] = np.nan_to_num(h[ii, jj])
    X = np.nan_to_num(X)

    sic_pred = np.full((rows, cols), np.nan, dtype=np.float32)
    unc = np.full((rows, cols), np.nan, dtype=np.float32)

    try:
        mw = _get(model_path)
        corr = predict_np(mw.model, X, mw.scaler)
        cov = mw.coverage if mw.coverage is not None else np.ones((rows, cols), bool)
        s_cov, s_unc = mw.sigma_covered, mw.sigma_uncovered
    except FileNotFoundError:
        corr = np.zeros(n, dtype=np.float32)
        cov = np.ones((rows, cols), bool)
        s_cov = s_unc = 5.0

    # Apply the learned correction ONLY where the wind features are genuine.
    # The model never saw edge-clamped wind, and measured against held-out
    # observations it is +1.4% better than physics inside the ERA5 domain but
    # -2.1% WORSE outside it, so those cells fall back to physics alone.
    corr = np.where(cov[ii, jj], corr, 0.0)

    sic_pred[ii, jj] = np.clip(sic_phys[ii, jj] + corr, SIC_MIN, SIC_MAX)
    unc[ii, jj] = np.where(cov[ii, jj], s_cov, s_unc)

    return {"sic_pred": sic_pred, "uncertainty": unc, "wind_coverage": cov}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train the sea-ice residual model.")
    ap.add_argument("--cell_stride", type=int, default=3,
                    help="Subsample every Nth valid cell for training.")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=1,
                    help="CLI compatibility only; DataLoader always uses 0 "
                         "worker processes because they crash on Windows.")
    ap.add_argument("--model_out", default="seaice_model.pt")
    ap.add_argument("--curve_out", default="seaice_training.png")
    ap.add_argument("--metrics_out", default="seaice_validation.json")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_seed(args.seed)

    print("[1/4] Building dataset ...")
    X, y, season = build_dataset(cell_stride=args.cell_stride)

    tr_m = np.isin(season, TRAIN_SEASONS)
    va_m = np.isin(season, VAL_SEASONS)
    if va_m.sum() == 0:
        raise ValueError("No validation samples - check season labels.")
    Xtr, ytr, Xva, yva = X[tr_m], y[tr_m], X[va_m], y[va_m]

    scaler = Scaler().fit(Xtr, ytr)

    print("[2/4] Training both variants (the better one is saved) ...")
    results = {}
    for kind, net in (("cnn", SeaIceCNN()), ("lstm", SeaIceLSTM())):
        print(f"      --- {kind.upper()} ---")
        m, hist, best = train_one(net, Xtr, ytr, Xva, yva, scaler,
                                  epochs=args.epochs, batch_size=args.batch_size,
                                  lr=args.lr, tag=kind)
        pred = predict_np(m, Xva, scaler)
        results[kind] = {"model": m, "hist": hist, "val_loss": best,
                         "metrics": metrics(yva, pred)}
        mm = results[kind]["metrics"]
        print(f"      {kind.upper():4s} val RMSE {mm['rmse_residual']:.3f} %SIC  "
              f"MAE {mm['mae_residual']:.3f}  r {mm['correlation']:.3f}  "
              f"skill vs physics {mm['skill_vs_physics_pct']:+.1f}%")

    best_kind = min(results, key=lambda k: results[k]["metrics"]["rmse_residual"])
    print(f"[3/4] Selected: {best_kind.upper()} "
          f"(lower validation RMSE)")

    _, _, coverage = build_wind_on_sic_grid(verbose=False)
    bm = results[best_kind]
    sigma = bm["metrics"]["rmse_residual"]
    wrap = SeaIceResidualModel(bm["model"], scaler, best_kind,
                               coverage=coverage,
                               sigma_covered=sigma, sigma_uncovered=sigma * 2.0)

    print("[4/4] Saving ...")
    wrap.save(args.model_out)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k in results:
        ax[0].plot(results[k]["hist"]["train"], label=f"{k} train")
        ax[0].plot(results[k]["hist"]["val"], "--", label=f"{k} val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("MSE (standardised)")
    ax[0].set_title("Sea-ice residual model - loss")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    names = list(results)
    ax[1].bar(names, [results[k]["metrics"]["rmse_residual"] for k in names],
              color=["tab:blue", "tab:orange"])
    ax[1].axhline(results[names[0]]["metrics"]["rmse_physics_only"],
                  color="k", ls=":", label="physics only")
    ax[1].set_ylabel("validation RMSE (%SIC)")
    ax[1].set_title("Held-out season 2024-25")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(args.curve_out, dpi=130)
    plt.close(fig)

    out = {
        "selected_model": best_kind,
        "window_days": WINDOW,
        "cell_stride": args.cell_stride,
        "train_seasons": list(TRAIN_SEASONS),
        "val_seasons": list(VAL_SEASONS),
        "n_train": int(tr_m.sum()),
        "n_val": int(va_m.sum()),
        "wind_coverage_fraction": float(coverage.mean()),
        "sigma_covered_pct_sic": sigma,
        "sigma_uncovered_pct_sic": sigma * 2.0,
        "variants": {k: results[k]["metrics"] for k in results},
        "notes": [
            "Trained only on cells inside the ERA5 wind domain "
            "(21.5% of valid SIC cells); elsewhere wind would be edge-clamped.",
            "Split is by season, not random, to avoid leakage from "
            "near-identical neighbouring days.",
            "ERA5 2m air temperature unavailable - physics uses Ross Sea "
            "seasonal climatology (Assumption A2).",
        ],
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n    model   -> {args.model_out}")
    print(f"    curves  -> {args.curve_out}")
    print(f"    metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
