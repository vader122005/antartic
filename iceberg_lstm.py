"""
iceberg_lstm.py - LSTM residual model for Antarctic iceberg drift.

WHAT IT LEARNS
--------------
The residual between the observed iceberg position and the Bigg/Wagner/Lichey
physics prediction produced by `iceberg_physics_baseline.py`. The physics model
already captures the bulk of the drift; the LSTM corrects the remaining gap,
which is driven mainly by

  * unmodelled ocean currents (CMEMS unavailable - ocean velocity is set to
    zero in the physics, the single largest known error source), and
  * mesoscale wind variability that a 6-hourly ERA5 sample cannot resolve.

ARCHITECTURE
------------
Input per timestep : [WIND_U, WIND_V, SIC, prior_RESID_LAT, prior_RESID_LON,
                      VEL_U, VEL_V]   (7 features, as specified)
Sequence length    : 5 (sliding window; see SEQ_LEN note below)
Model              : 2-layer LSTM, hidden 32, dropout 0.3, dense head
Output             : [RESID_LAT, RESID_LON] at t+1
Loss / optimiser   : MSE on residuals / Adam lr=1e-3 + ReduceLROnPlateau

TRAINING SPLIT
--------------
B38 only, 80/20 chronological (NO shuffle - the split must not leak future
information backwards through a time series). The seven remaining bergs
(B46, B47, B22F, B42, B22G, B22A, B50) are a held-out generalisation set,
never seen during training or model selection.

UNCERTAINTY
-----------
1-sigma is estimated by MC dropout: dropout stays active at inference and the
model is sampled `n_mc` times. The spread is converted from degrees to km and
combined in quadrature, then floored at a physically sensible minimum. This
feeds the hazard-zone exclusion radius used by the route optimiser.

VALIDATION GATE
---------------
The learned correction is applied ONLY if it beat physics-only on the held-out
validation split; the verdict is stored in the checkpoint and honoured by
`predict_iceberg_position`. This matters here because the residual is
structurally noise-dominated: residual_t = obs(t+1) - physics(obs_t) is a
first difference of the observation series, so independent fix noise enters as
eps(t+1) - eps(t) and shows a strong NEGATIVE lag-1 autocorrelation (-0.67 lat,
-0.80 lon on B38). Part of that IS exploitable through the prior-residual
features, but a model that fails to exploit it must never silently degrade the
positions the route optimiser depends on. Uncertainty is always returned,
whether or not the correction is applied.

KNOWN GAP (surfaced, not fixed): ocean currents are unavailable, so trajectory
uncertainty is elevated. Route output carries
    "ocean_current_data": "unavailable - trajectory uncertainty elevated"
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import grid_utils as gu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURES = ["WIND_U", "WIND_V", "SIC",
            "PRIOR_RESID_LAT", "PRIOR_RESID_LON",
            "VEL_U", "VEL_V"]
TARGETS = ["RESID_LAT", "RESID_LON"]

# Sequence length 5, not 10. After removing BYU carried-forward duplicate
# fixes (see iceberg_physics_baseline.run_baseline_on_csv) B38 yields only 69
# genuine drift pairs; a length-10 window would leave ~59 sequences and
# exclude four of the seven transfer bergs entirely.
SEQ_LEN = 5
# Capacity is deliberately small for the same reason: a 2x64 LSTM (~50k
# parameters) on 55 training sequences overfits hard (train MSE 0.79 vs
# val 3.16) and degraded accuracy on every berg.
HIDDEN = 32
LAYERS = 2
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4

TRAIN_BERG = "b38"
TRANSFER_BERGS = ["b46", "b47", "b22f", "b42", "b22g", "b22a", "b50"]

# Floor on predicted 1-sigma (km). Below this the exclusion radius would be
# smaller than the position uncertainty of the BYU/NIC fix itself.
MIN_SIGMA_KM = 2.0

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Turn one berg's residual records into sliding windows.

    The "prior residual" features are the residuals at t-1, shifted so the
    model never sees the residual it is being asked to predict. Windows are
    only emitted where the whole span is contiguous in SEASON, so a window
    never bridges the Apr-Oct data void between shipping seasons.

    Returns (X, y, meta_rows) with X (n, seq_len, 7) and y (n, 2).
    """
    df = df.sort_values("DATE").reset_index(drop=True).copy()

    df["PRIOR_RESID_LAT"] = df["RESID_LAT"].shift(1).fillna(0.0)
    df["PRIOR_RESID_LON"] = df["RESID_LON"].shift(1).fillna(0.0)

    feats = df[FEATURES].to_numpy(dtype=np.float32)
    targs = df[TARGETS].to_numpy(dtype=np.float32)
    seasons = df["SEASON"].to_numpy()

    X, y, meta = [], [], []
    for i in range(len(df) - seq_len):
        win = slice(i, i + seq_len)
        # Require one unbroken season across the window and its target
        if len(set(seasons[win])) != 1 or seasons[i] != seasons[i + seq_len]:
            continue
        X.append(feats[win])
        y.append(targs[i + seq_len])
        meta.append(df.iloc[i + seq_len])

    if not X:
        return (np.empty((0, seq_len, len(FEATURES)), dtype=np.float32),
                np.empty((0, len(TARGETS)), dtype=np.float32),
                pd.DataFrame())

    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
            pd.DataFrame(meta).reset_index(drop=True))


class Standardiser:
    """Feature/target z-scoring. Fitted on TRAIN ONLY to avoid leakage."""

    def __init__(self):
        self.x_mu = self.x_sd = self.y_mu = self.y_sd = None

    def fit(self, X, y):
        flat = X.reshape(-1, X.shape[-1])
        self.x_mu = flat.mean(axis=0)
        self.x_sd = flat.std(axis=0)
        self.x_sd[self.x_sd < 1e-8] = 1.0
        self.y_mu = y.mean(axis=0)
        self.y_sd = y.std(axis=0)
        self.y_sd[self.y_sd < 1e-12] = 1.0
        return self

    def tx(self, X):
        return (X - self.x_mu) / self.x_sd

    def ty(self, y):
        return (y - self.y_mu) / self.y_sd

    def inv_y(self, y):
        return y * self.y_sd + self.y_mu

    def state(self):
        return {k: np.asarray(v).tolist() for k, v in
                dict(x_mu=self.x_mu, x_sd=self.x_sd,
                     y_mu=self.y_mu, y_sd=self.y_sd).items()}

    @classmethod
    def from_state(cls, st):
        s = cls()
        s.x_mu = np.asarray(st["x_mu"], dtype=np.float32)
        s.x_sd = np.asarray(st["x_sd"], dtype=np.float32)
        s.y_mu = np.asarray(st["y_mu"], dtype=np.float32)
        s.y_sd = np.asarray(st["y_sd"], dtype=np.float32)
        return s


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IcebergLSTM(nn.Module):
    """2-layer LSTM -> dropout -> dense head predicting [RESID_LAT, RESID_LON]."""

    def __init__(self, n_features=len(FEATURES), hidden=HIDDEN,
                 layers=LAYERS, dropout=DROPOUT, n_out=len(TARGETS)):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :]))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X, y, epochs=300, batch_size=16, lr=1e-3,
                val_frac=0.2, patience=40, verbose=True,
                weight_decay=WEIGHT_DECAY):
    """
    Chronological 80/20 split, Adam + ReduceLROnPlateau, early stopping on
    validation loss. Returns (model, standardiser, history).
    """
    n_val = max(1, int(round(len(X) * val_frac)))
    n_tr = len(X) - n_val
    if n_tr < 1:
        raise ValueError("Not enough sequences to train.")

    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr:], y[n_tr:]

    sc = Standardiser().fit(X_tr, y_tr)
    tr = TensorDataset(torch.tensor(sc.tx(X_tr)), torch.tensor(sc.ty(y_tr)))
    va = TensorDataset(torch.tensor(sc.tx(X_va)), torch.tensor(sc.ty(y_va)))

    # workers=0: multiprocessing DataLoader workers crash on Windows
    tl = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0)

    model = IcebergLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10)
    lossf = nn.MSELoss()

    hist = {"train": [], "val": [], "lr": []}
    best = float("inf")
    best_state = None
    bad = 0

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
        tr_loss = tot / len(tr)

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                tot += lossf(model(xb), yb).item() * len(xb)
        va_loss = tot / len(va)

        sched.step(va_loss)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        hist["lr"].append(opt.param_groups[0]["lr"])

        if va_loss < best - 1e-6:
            best, bad = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"      early stop at epoch {ep + 1} (best val {best:.5f})")
                break

        if verbose and (ep + 1) % 25 == 0:
            print(f"      epoch {ep + 1:4d}  train {tr_loss:.5f}  val {va_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, sc, hist


# ---------------------------------------------------------------------------
# Inference + uncertainty
# ---------------------------------------------------------------------------

def _mc_predict(model, x, n_mc=30):
    """
    MC-dropout prediction. Dropout is kept ACTIVE (model.train()) so each pass
    samples a different sub-network; the spread approximates predictive
    uncertainty. Returns (mean, std) in standardised units.
    """
    model.train()
    with torch.no_grad():
        draws = torch.stack([model(x) for _ in range(n_mc)], dim=0)
    model.eval()
    return draws.mean(dim=0).numpy(), draws.std(dim=0).numpy()


def _deg_to_km(dlat, dlon, lat):
    """Convert a lat/lon offset in degrees to kilometres at a given latitude."""
    km_lat = dlat * 111.32
    km_lon = dlon * 111.32 * math.cos(math.radians(lat))
    return km_lat, km_lon


class IcebergResidualModel:
    """Trained model + standardiser, with save/load and the inference helper."""

    def __init__(self, model: IcebergLSTM, scaler: Standardiser,
                 apply_correction: bool = True, sigma_scale: float = 1.0,
                 gain: float = 1.0):
        self.model = model
        self.scaler = scaler
        # Set from the validation comparison in main(); see VALIDATION GATE.
        self.apply_correction = bool(apply_correction)
        # Multiplier that rescales raw MC-dropout spread so that mean predicted
        # 1-sigma matches the RMS position error actually observed on the
        # validation split. Without it the hazard radii are arbitrary.
        self.sigma_scale = float(sigma_scale)
        # Shrinkage applied to the raw correction, fitted on validation. The
        # target is noise-dominated, so the MSE-optimal correction is smaller
        # than the network's point estimate; g<1 pulls it back toward physics.
        self.gain = float(gain)
        self.model.eval()

    # -- persistence --------------------------------------------------------
    def save(self, path="iceberg_lstm.pt"):
        torch.save({
            "state_dict": self.model.state_dict(),
            "scaler": self.scaler.state(),
            "features": FEATURES,
            "targets": TARGETS,
            "seq_len": SEQ_LEN,
            "arch": {"hidden": HIDDEN, "layers": LAYERS, "dropout": DROPOUT},
            "apply_correction": self.apply_correction,
            "sigma_scale": self.sigma_scale,
            "gain": self.gain,
        }, path)
        return path

    @classmethod
    def load(cls, path="iceberg_lstm.pt"):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        m = IcebergLSTM()
        m.load_state_dict(blob["state_dict"])
        return cls(m, Standardiser.from_state(blob["scaler"]),
                   apply_correction=blob.get("apply_correction", True),
                   sigma_scale=blob.get("sigma_scale", 1.0),
                   gain=blob.get("gain", 1.0))

    # -- prediction ---------------------------------------------------------
    def predict_residual(self, seq, n_mc=30):
        """
        seq: (seq_len, 7) array of feature vectors, oldest first.
        Returns (resid_lat, resid_lon, sd_lat_deg, sd_lon_deg).
        """
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[-1] != len(FEATURES):
            raise ValueError(f"expected (seq_len, {len(FEATURES)}), got {seq.shape}")
        x = torch.tensor(self.scaler.tx(seq)[None, ...])
        mu, sd = _mc_predict(self.model, x, n_mc=n_mc)
        resid = self.scaler.inv_y(mu)[0]
        sd_deg = sd[0] * self.scaler.y_sd * self.sigma_scale
        if not self.apply_correction:
            # Gate closed: physics-only position, but keep the uncertainty.
            return 0.0, 0.0, float(sd_deg[0]), float(sd_deg[1])
        return (float(resid[0]) * self.gain, float(resid[1]) * self.gain,
                float(sd_deg[0]), float(sd_deg[1]))


# ---------------------------------------------------------------------------
# The interface the route optimiser calls
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


def _get_model(path="iceberg_lstm.pt"):
    if path not in _MODEL_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found - train it first: python iceberg_lstm.py")
        _MODEL_CACHE[path] = IcebergResidualModel.load(path)
    return _MODEL_CACHE[path]


def make_feature_vector(wind_u, wind_v, sic, prior_resid_lat,
                        prior_resid_lon, vel_u, vel_v):
    """Assemble one timestep's feature vector in the canonical FEATURES order."""
    return [float(wind_u), float(wind_v), float(sic),
            float(prior_resid_lat), float(prior_resid_lon),
            float(vel_u), float(vel_v)]


def predict_iceberg_position(
    lat, lon,
    wind_u, wind_v,
    sic,
    sequence_history,
    physics_result,
    model_path="iceberg_lstm.pt",
    n_mc=30,
):
    """
    Combine the physics step with the learned residual correction.

    Parameters
    ----------
    lat, lon         : current berg position (degrees)
    wind_u, wind_v   : ERA5 wind at the berg (m/s)
    sic              : NSIDC SIC at the berg (0-1)
    sequence_history : list of the last N feature vectors (each length 7).
                       Shorter histories are left-padded by repeating the
                       oldest row; longer ones are truncated to the most
                       recent SEQ_LEN.
    physics_result   : dict from iceberg_physics_baseline.physics_step()

    Returns
    -------
    {'lat_pred', 'lon_pred', 'uncertainty_km',
     'physics_lat', 'physics_lon', 'resid_lat', 'resid_lon',
     'ocean_current_data'}
    """
    lat_phys = float(physics_result["lat_new"])
    lon_phys = float(physics_result["lon_new"])

    hist = [list(map(float, r)) for r in (sequence_history or [])]
    if not hist:
        hist = [make_feature_vector(wind_u, wind_v, sic, 0.0, 0.0,
                                    physics_result.get("vel_u_new", 0.0),
                                    physics_result.get("vel_v_new", 0.0))]
    hist = hist[-SEQ_LEN:]
    while len(hist) < SEQ_LEN:
        hist.insert(0, hist[0])

    try:
        model = _get_model(model_path)
        r_lat, r_lon, sd_lat, sd_lon = model.predict_residual(hist, n_mc=n_mc)
    except FileNotFoundError:
        # Physics-only fallback so the pipeline still runs untrained.
        r_lat = r_lon = 0.0
        sd_lat = sd_lon = 0.05

    lat_pred = lat_phys + r_lat
    lon_pred = ((lon_phys + r_lon + 180.0) % 360.0) - 180.0

    km_lat, km_lon = _deg_to_km(sd_lat, sd_lon, lat_pred)
    sigma_km = max(math.hypot(km_lat, km_lon), MIN_SIGMA_KM)

    return {
        "lat_pred": float(lat_pred),
        "lon_pred": float(lon_pred),
        "uncertainty_km": float(sigma_km),
        "physics_lat": lat_phys,
        "physics_lon": lon_phys,
        "resid_lat": float(r_lat),
        "resid_lon": float(r_lon),
        "ocean_current_data": "unavailable - trajectory uncertainty elevated",
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model_wrap, X, y, meta, n_mc=20):
    """Return per-sample errors in degrees and km, plus summary metrics."""
    if len(X) == 0:
        return None

    xs = torch.tensor(model_wrap.scaler.tx(X))
    mu, sd = _mc_predict(model_wrap.model, xs, n_mc=n_mc)
    pred = model_wrap.scaler.inv_y(mu) * model_wrap.gain
    sd_deg = sd * model_wrap.scaler.y_sd * model_wrap.sigma_scale

    err_lat = pred[:, 0] - y[:, 0]
    err_lon = pred[:, 1] - y[:, 1]

    lat_ref = meta["LAT_obs"].to_numpy() if "LAT_obs" in meta else np.full(len(y), -75.0)

    # Physics-only error, for the "did the ML help?" comparison
    phys_km = gu.haversine_km(meta["LAT_obs"], meta["LON_obs"],
                              meta["LAT_phys"], meta["LON_phys"])
    corr_lat = meta["LAT_phys"].to_numpy() + pred[:, 0]
    corr_lon = meta["LON_phys"].to_numpy() + pred[:, 1]
    ml_km = gu.haversine_km(meta["LAT_obs"], meta["LON_obs"], corr_lat, corr_lon)

    sigma_km = np.hypot(sd_deg[:, 0] * 111.32,
                        sd_deg[:, 1] * 111.32 * np.cos(np.radians(lat_ref)))
    if not model_wrap.apply_correction:
        ml_km = phys_km

    return {
        "n": int(len(y)),
        "mae_lat_deg": float(np.abs(err_lat).mean()),
        "mae_lon_deg": float(np.abs(err_lon).mean()),
        "rmse_lat_deg": float(np.sqrt((err_lat ** 2).mean())),
        "rmse_lon_deg": float(np.sqrt((err_lon ** 2).mean())),
        "physics_only_km": float(np.mean(phys_km)),
        "physics_plus_lstm_km": float(np.mean(ml_km)),
        "improvement_pct": float(
            100.0 * (np.mean(phys_km) - np.mean(ml_km)) / max(np.mean(phys_km), 1e-9)),
        "mean_sigma_km": float(np.mean(np.maximum(sigma_km, MIN_SIGMA_KM))),
    }


def plot_history(hist, path="iceberg_lstm_training.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train"], label="train")
    ax[0].plot(hist["val"], label="validation")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("MSE (standardised)")
    ax[0].set_title("Iceberg residual LSTM - loss")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(hist["lr"], color="tab:red")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("learning rate")
    ax[1].set_title("LR schedule (ReduceLROnPlateau)")
    ax[1].set_yscale("log")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train the iceberg drift residual LSTM.")
    ap.add_argument("--residual_csv", default="iceberg_physics_residuals_all.csv",
                    help="Combined per-berg residual file from the physics baseline.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    ap.add_argument("--workers", type=int, default=1,
                    help="Kept for CLI compatibility; DataLoader always uses 0 "
                         "worker processes because they crash on Windows.")
    ap.add_argument("--model_out", default="iceberg_lstm.pt")
    ap.add_argument("--curve_out", default="iceberg_lstm_training.png")
    ap.add_argument("--gen_out", default="iceberg_lstm_generalisation.csv")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.residual_csv):
        raise FileNotFoundError(
            f"{args.residual_csv} not found - run iceberg_physics_baseline.py first.")

    df = pd.read_csv(args.residual_csv)
    df["ICEBERG_ID"] = df["ICEBERG_ID"].astype(str).str.lower()

    print("[1/4] Building sequences ...")
    train_df = df[df["ICEBERG_ID"] == TRAIN_BERG]
    if train_df.empty:
        raise ValueError(f"No rows for training berg {TRAIN_BERG!r}")

    X, y, meta = build_sequences(train_df, args.seq_len)
    print(f"      {TRAIN_BERG}: {len(X)} sequences of length {args.seq_len} "
          f"from {len(train_df)} residual records")
    if len(X) < 20:
        raise ValueError(f"Only {len(X)} sequences - too few to train.")

    print("[2/4] Training (chronological 80/20, no shuffle across the split) ...")
    model, scaler, hist = train_model(
        X, y, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    wrap = IcebergResidualModel(model, scaler)

    n_val = max(1, int(round(len(X) * 0.2)))
    n_tr = len(X) - n_val
    tr_m = evaluate(wrap, X[:n_tr], y[:n_tr], meta.iloc[:n_tr])
    va_m = evaluate(wrap, X[n_tr:], y[n_tr:], meta.iloc[n_tr:])
    print(f"      train: MAE lat {tr_m['mae_lat_deg']:.4f} deg  "
          f"lon {tr_m['mae_lon_deg']:.4f} deg")
    print(f"      val  : MAE lat {va_m['mae_lat_deg']:.4f} deg  "
          f"lon {va_m['mae_lon_deg']:.4f} deg  "
          f"| physics {va_m['physics_only_km']:.2f} km -> "
          f"+LSTM {va_m['physics_plus_lstm_km']:.2f} km "
          f"({va_m['improvement_pct']:+.1f}%)")

    # --- Fit the shrinkage gain on validation -------------------------------
    best_g, best_km = 0.0, va_m["physics_only_km"]
    for g in np.arange(0.0, 1.55, 0.05):
        wrap.gain = float(g)
        m = evaluate(wrap, X[n_tr:], y[n_tr:], meta.iloc[n_tr:])
        if m["physics_plus_lstm_km"] < best_km:
            best_g, best_km = float(g), m["physics_plus_lstm_km"]
    wrap.gain = best_g
    va_m = evaluate(wrap, X[n_tr:], y[n_tr:], meta.iloc[n_tr:])
    tr_m = evaluate(wrap, X[:n_tr], y[:n_tr], meta.iloc[:n_tr])
    print(f"      shrinkage gain fitted on validation: g={best_g:.2f} "
          f"(physics {va_m['physics_only_km']:.2f} -> "
          f"{va_m['physics_plus_lstm_km']:.2f} km)")

    # --- VALIDATION GATE + uncertainty calibration --------------------------
    improved = (va_m["physics_plus_lstm_km"] < va_m["physics_only_km"]
                and best_g > 0.0)
    wrap.apply_correction = bool(improved)
    if improved:
        print(f"      GATE OPEN : LSTM correction improves validation "
              f"({va_m['physics_only_km']:.2f} -> "
              f"{va_m['physics_plus_lstm_km']:.2f} km); correction WILL be applied.")
    else:
        print(f"      GATE CLOSED: LSTM correction does NOT beat physics on "
              f"validation ({va_m['physics_only_km']:.2f} -> "
              f"{va_m['physics_plus_lstm_km']:.2f} km).")
        print("                   Falling back to physics-only positions; the "
              "model still supplies uncertainty for hazard radii.")

    # Scale raw MC-dropout spread so mean 1-sigma matches the RMS validation
    # error, making the hazard exclusion radius physically meaningful.
    ref_km = va_m["physics_plus_lstm_km"] if improved else va_m["physics_only_km"]
    raw_sigma = max(va_m["mean_sigma_km"], 1e-6)
    wrap.sigma_scale = float(np.clip(ref_km / raw_sigma, 0.1, 50.0))
    print(f"      sigma calibration: raw {raw_sigma:.2f} km -> "
          f"target {ref_km:.2f} km (scale x{wrap.sigma_scale:.2f})")

    print("[3/4] Held-out generalisation across the 7 transfer bergs ...")
    rows = [{"berg": TRAIN_BERG.upper(), "role": "train", **tr_m},
            {"berg": TRAIN_BERG.upper(), "role": "val", **va_m,
             "gate": "open" if improved else "closed",
             "gain": round(wrap.gain, 3),
             "sigma_scale": round(wrap.sigma_scale, 3)}]
    for b in TRANSFER_BERGS:
        sub = df[df["ICEBERG_ID"] == b]
        if sub.empty:
            rows.append({"berg": b.upper(), "role": "transfer", "n": 0,
                         "note": "no residual records"})
            continue
        Xb, yb, mb = build_sequences(sub, args.seq_len)
        if len(Xb) == 0:
            rows.append({"berg": b.upper(), "role": "transfer", "n": 0,
                         "note": f"only {len(sub)} records, "
                                 f"< seq_len+1 ({args.seq_len + 1}) in one season"})
            print(f"      {b.upper():6s} skipped - {len(sub)} records, "
                  f"too few for a length-{args.seq_len} window")
            continue
        m = evaluate(wrap, Xb, yb, mb)
        rows.append({"berg": b.upper(), "role": "transfer", **m})
        print(f"      {b.upper():6s} n={m['n']:3d}  MAE lat {m['mae_lat_deg']:.4f}  "
              f"lon {m['mae_lon_deg']:.4f}  | physics {m['physics_only_km']:6.2f} km "
              f"-> +LSTM {m['physics_plus_lstm_km']:6.2f} km "
              f"({m['improvement_pct']:+.1f}%)")

    print("[4/4] Saving ...")
    wrap.save(args.model_out)
    plot_history(hist, args.curve_out)
    pd.DataFrame(rows).to_csv(args.gen_out, index=False)

    print(f"\n    model   -> {args.model_out}")
    print(f"    curves  -> {args.curve_out}")
    print(f"    per-berg-> {args.gen_out}")
    print("\n    NOTE: ocean currents (CMEMS) unavailable - "
          "trajectory uncertainty elevated; surfaced in route output.")


if __name__ == "__main__":
    main()
