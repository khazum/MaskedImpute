#!/usr/bin/env python3
"""
experiment2.py
---------------

Runs the imputation experiment (Autoencoder) with fixed settings, then predicts
biological zeros among observed zeros using multiple approaches (baseline + SPLAT
variants). For each approach, two prediction modes are evaluated:
  1) Predictions from observed counts.
  2) Predictions from imputed values (in count space)

Predicted biological-zero positions are set to 0 in the imputed logcounts, and
MSEs vs. logTrueCounts are reported (overall + NonZero/BioZero/Dropout masks).

Thresholds are tuned *globally across all datasets* (single threshold per
approach/variant), not per-dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from DenseLayerPack import DenseLayer
from DenseLayerPack.const import DENSE_LAYER_CONST
from predict_dropouts_new import (
    baseline_gene_mean_heuristic_counts,
    splatter_bio_posterior_from_counts,
    _choose_thresh_for_metric,
)
from rds2py import read_rds

# ------------------------
# Helpers from experiment.py
# ------------------------

class RobustZThenMinMaxToNeg1Pos1:
    def __init__(self, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-8):
        assert 0.0 <= p_low < p_high <= 100.0
        self.p_low = p_low
        self.p_high = p_high
        self.eps = eps
        self.lo_ = None
        self.hi_ = None
        self.mean_ = None
        self.std_ = None
        self.zmin_ = None
        self.zmax_ = None
        self.zspan_ = None

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lo_, self.hi_)

    def fit(self, X: np.ndarray):
        self.lo_ = np.percentile(X, self.p_low, axis=0)
        self.hi_ = np.percentile(X, self.p_high, axis=0)
        Xc = self._clip(X)
        self.mean_ = Xc.mean(axis=0)
        self.std_ = Xc.std(axis=0)
        self.std_[self.std_ < self.eps] = 1.0
        Z = (Xc - self.mean_) / self.std_
        self.zmin_ = Z.min(axis=0)
        self.zmax_ = Z.max(axis=0)
        self.zspan_ = self.zmax_ - self.zmin_
        self.zspan_[self.zspan_ < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = self._clip(X)
        Z = (Xc - self.mean_) / self.std_
        X01 = (Z - self.zmin_) / self.zspan_
        Xscaled = X01 * 2.0 - 1.0
        return Xscaled.astype(np.float32)

    def inverse_transform(self, Xscaled: np.ndarray) -> np.ndarray:
        X01 = (Xscaled + 1.0) / 2.0
        Z = X01 * self.zspan_ + self.zmin_
        X_unz = Z * self.std_ + self.mean_
        return X_unz.astype(np.float32)


class IdentityScaler:
    def fit(self, X: np.ndarray):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)

    def inverse_transform(self, Xscaled: np.ndarray) -> np.ndarray:
        return Xscaled.astype(np.float32)


class AE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        bottleneck: int,
        layer_type: str,
    ):
        super().__init__()
        sizes_enc = [input_dim] + list(hidden) + [bottleneck]
        sizes_dec = [bottleneck] + list(reversed(hidden)) + [input_dim]

        enc_layers = []
        for i in range(len(sizes_enc) - 1):
            enc_layers.append(DenseLayer(sizes_enc[i], sizes_enc[i + 1], layer_type=layer_type))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for i in range(len(sizes_dec) - 1):
            dec_layers.append(DenseLayer(sizes_dec[i], sizes_dec[i + 1], layer_type=layer_type))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def mse_from_residual(residual: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        residual = residual * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(residual.numel(), device=residual.device, dtype=residual.dtype)
    return residual.pow(2).sum() / denom


def load_dataset(path: str, need_labels: bool, need_truth: bool):
    sce = read_rds(path)
    logcounts = sce.assay("logcounts").T.astype("float32")
    keep = np.sum(logcounts != 0, axis=0) >= 2
    logcounts = logcounts[:, keep]

    log_true_counts = None
    if need_truth:
        for assay_name in ("logTrueCounts", "perfect_logcounts"):
            try:
                log_true_counts = sce.assay(assay_name).T[:, keep].astype("float32")
                break
            except Exception:
                continue

    return logcounts, log_true_counts


def logcounts_to_counts(logcts: np.ndarray, base: float = 2.0) -> np.ndarray:
    return np.expm1(logcts * np.log(base))

APPROACHES = [
    {"name": "baseline", "kind": "baseline"},
    {"name": "splat_cellaware", "kind": "splat", "disp_mode": "estimate", "use_cell_factor": True},
    {"name": "splat_mom", "kind": "splat", "disp_mode": "estimate", "use_cell_factor": False},
    {"name": "splat_fixed", "kind": "splat", "disp_mode": "fixed", "disp_const": 0.1, "use_cell_factor": False},
]

def _sanitize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(p, 0.0, 1.0)


def _splat_bio_posterior_at_zeros(approach: Dict[str, object], counts: np.ndarray, zeros_obs: np.ndarray) -> np.ndarray:
    disp_mode = str(approach.get("disp_mode", "estimate"))
    disp_const = float(approach.get("disp_const", 0.1))
    use_cell_factor = bool(approach.get("use_cell_factor", False))
    bio_post = splatter_bio_posterior_from_counts(
        counts,
        disp_mode=disp_mode,
        disp_const=disp_const,
        use_cell_factor=use_cell_factor,
        groups=None,
    )
    return _sanitize_prob(np.asarray(bio_post, dtype=np.float64)[zeros_obs]).astype(np.float32)


def _baseline_pred_dropout_at_zeros(counts: np.ndarray, zeros_obs: np.ndarray) -> np.ndarray:
    pred_dropout = baseline_gene_mean_heuristic_counts(counts, quantile=0.2).astype(bool)
    return pred_dropout[zeros_obs].astype(bool)


def _choose_global_f1_thresh(p_drop_list: List[np.ndarray], bio_true_list: List[np.ndarray], drop_true_list: List[np.ndarray]) -> float:
    if not p_drop_list:
        return float("nan")
    p_all = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in p_drop_list], axis=0)
    bio_all = np.concatenate([np.asarray(b, dtype=bool).reshape(-1) for b in bio_true_list], axis=0)
    drop_all = np.concatenate([np.asarray(d, dtype=bool).reshape(-1) for d in drop_true_list], axis=0)
    zeros_all = np.ones_like(bio_all, dtype=bool)
    return float(_choose_thresh_for_metric(p_all, zeros_all, drop_all, bio_all, metric="f1"))


def _choose_global_thr_bio_for_min_arith_mean_bz_do_mse(
    p_bio_list: List[np.ndarray],
    delta_list: List[np.ndarray],
    bio_true_list: List[np.ndarray],
    drop_true_list: List[np.ndarray],
    base_sum_bz: float,
    base_sum_do: float,
    n_bz: int,
    n_do: int,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Global thrBio (single threshold for all datasets) that minimizes:
      0.5 * (BioZero_MSE + Dropout_MSE)
    after hard-zeroing predicted biological zeros among observed zeros, using
    P(Bio|0) (computed from imputed counts).

    Returns (thrBio, best_obj). thrBio=NA means "no adjustment".
    """
    if n_bz <= 0 or n_do <= 0:
        return float("nan"), float("nan")

    base_sum_bz = float(base_sum_bz)
    base_sum_do = float(base_sum_do)
    base_obj = 0.5 * ((base_sum_bz / float(n_bz)) + (base_sum_do / float(n_do)))

    if not p_bio_list:
        return float("nan"), float(base_obj)

    p_all = np.concatenate([_sanitize_prob(p).reshape(-1) for p in p_bio_list], axis=0)
    delta_all = np.concatenate([np.asarray(d, dtype=np.float64).reshape(-1) for d in delta_list], axis=0)
    bz_all = np.concatenate([np.asarray(b, dtype=bool).reshape(-1) for b in bio_true_list], axis=0)
    do_all = np.concatenate([np.asarray(d, dtype=bool).reshape(-1) for d in drop_true_list], axis=0)

    finite = np.isfinite(p_all) & np.isfinite(delta_all)
    if int(finite.sum()) == 0:
        return float("nan"), float(base_obj)

    p_all = p_all[finite]
    delta_all = delta_all[finite]
    bz_all = bz_all[finite]
    do_all = do_all[finite]

    order = np.argsort(-p_all, kind="mergesort")  # descending p_bio
    p_s = p_all[order]
    delta_s = delta_all[order]
    bz_s = bz_all[order].astype(np.float64)
    do_s = do_all[order].astype(np.float64)

    cum_delta_bz = np.cumsum(delta_s * bz_s)
    cum_delta_do = np.cumsum(delta_s * do_s)

    group_ends = np.flatnonzero(np.r_[p_s[1:] != p_s[:-1], True])

    best_obj = float(base_obj)
    best_thr = 1.0 + 1e-6  # >1 => no adjustment

    for idx in group_ends:
        d_bz = float(cum_delta_bz[idx])
        d_do = float(cum_delta_do[idx])

        sse_bz = max(0.0, base_sum_bz + d_bz)
        sse_do = max(0.0, base_sum_do + d_do)
        obj = 0.5 * ((sse_bz / float(n_bz)) + (sse_do / float(n_do)))

        thr = float(p_s[idx])
        if (obj < best_obj - eps) or (abs(obj - best_obj) <= eps and thr > best_thr):
            best_obj = obj
            best_thr = thr

    thr_out = float("nan") if best_thr > 1.0 else float(best_thr)
    return thr_out, best_obj


def tune_hp_threshold_min_mse(
    repeats: List[Dict[str, object]],
    forced_masks: List[np.ndarray],
    n_total: int,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    High-pass thresholding on observed-zero positions in log space:
      - Start from the "final" matrix after biological-zero zeroing (forced_masks).
      - Further set to 0 any remaining observed-zero positions with value < x.

    Threshold x is chosen by scanning candidate values (ascending) from the final
    matrices' observed-zero positions (pooled across repeats) and stopping once
    the averaged overall MSE increases vs. the previous candidate.

    Returns (best_x, best_mse). If no x lowers MSE, best_x is NA and best_mse is
    the baseline (no HP) MSE.
    """
    if not repeats or n_total <= 0:
        return float("nan"), float("nan")
    if len(forced_masks) != len(repeats):
        raise ValueError("forced_masks length must match repeats length")

    n_rep = len(repeats)
    denom = float(n_rep * int(n_total))

    forced_any = False
    base_sse_sum = 0.0
    values_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []

    for rep, forced in zip(repeats, forced_masks):
        forced = np.asarray(forced, dtype=bool).reshape(-1)
        forced_any = forced_any or bool(forced.any())

        delta_z = np.asarray(rep["delta_z"], dtype=np.float64).reshape(-1)
        pred_z = np.asarray(rep["pred_z"], dtype=np.float64).reshape(-1)
        if delta_z.shape != pred_z.shape or delta_z.shape != forced.shape:
            raise ValueError("delta_z/pred_z/forced mask shapes must match")

        base_sse_sum += float(rep["raw_sse_total"]) + float(np.sum(delta_z[forced]))

        keep = ~forced
        if int(keep.sum()) > 0:
            values_list.append(pred_z[keep])
            delta_list.append(delta_z[keep])

    base_mse = base_sse_sum / denom

    if not values_list:
        return float("nan"), float(base_mse)

    values = np.concatenate(values_list, axis=0).astype(np.float64, copy=False)
    deltas = np.concatenate(delta_list, axis=0).astype(np.float64, copy=False)

    finite = np.isfinite(values) & np.isfinite(deltas)
    if int(finite.sum()) == 0:
        return float("nan"), float(base_mse)

    values = values[finite]
    deltas = deltas[finite]

    order = np.argsort(values, kind="mergesort")
    values_s = values[order]
    deltas_s = deltas[order]

    thr_candidates = np.unique(values_s)
    if forced_any:
        thr_candidates = np.unique(np.concatenate([thr_candidates, np.array([0.0], dtype=np.float64)]))

    cum_delta = 0.0
    idx = 0

    best_mse = float(base_mse)
    best_thr = float("nan")
    prev_mse = float(base_mse)

    for x in thr_candidates:
        while idx < values_s.size and values_s[idx] < x:
            cum_delta += float(deltas_s[idx])
            idx += 1

        mse = (base_sse_sum + cum_delta) / denom

        # Stop once MSE increases (vs. previous candidate).
        if mse > prev_mse + eps:
            break

        if (mse < best_mse - eps) or (abs(mse - best_mse) <= eps and np.isfinite(best_thr) and x > best_thr):
            best_mse = float(mse)
            best_thr = float(x)
        elif (abs(mse - best_mse) <= eps) and (not np.isfinite(best_thr)):
            best_thr = float(x)

        prev_mse = float(mse)

    if best_mse < base_mse - eps and np.isfinite(best_thr):
        return float(best_thr), float(best_mse)

    return float("nan"), float(base_mse)


def bio_zero_metrics(
    zeros: np.ndarray,
    bio_true: np.ndarray,
    drop_true: np.ndarray,
    pred_dropout: np.ndarray,
) -> Dict[str, float]:
    zeros = zeros.astype(bool)
    bio_true = bio_true.astype(bool)
    drop_true = drop_true.astype(bool)
    pred_dropout = pred_dropout.astype(bool)

    pred_drop_on_zeros = pred_dropout & zeros
    pred_bio_on_zeros = zeros & (~pred_dropout)

    total_zeros = float(zeros.sum())
    true_bio_zeros = float(bio_true.sum())
    true_dropouts = float(drop_true.sum())
    predicted_dropouts = float(pred_drop_on_zeros.sum())
    predicted_bio = float(pred_bio_on_zeros.sum())

    tp = float((bio_true & pred_bio_on_zeros).sum())
    tn = float((drop_true & pred_drop_on_zeros).sum())

    rec = tp / true_bio_zeros if true_bio_zeros > 0 else float("nan")
    if predicted_bio > 0:
        prec = tp / predicted_bio
    else:
        prec = 0.0 if np.isfinite(rec) else float("nan")

    if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = float("nan")

    acc = (tp + tn) / total_zeros if total_zeros > 0 else float("nan")

    return {
        "total_zeros": total_zeros,
        "true_bio_zeros": true_bio_zeros,
        "true_dropouts": true_dropouts,
        "predicted_bio": predicted_bio,
        "predicted_dropouts": predicted_dropouts,
        "Precision_Bio": 100.0 * prec if np.isfinite(prec) else float("nan"),
        "Recall_Bio": 100.0 * rec if np.isfinite(rec) else float("nan"),
        "F1_Score_Bio": 100.0 * f1 if np.isfinite(f1) else float("nan"),
        "Accuracy_Bio": 100.0 * acc if np.isfinite(acc) else float("nan"),
    }


def _masked_mse(diff: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.mean((diff[mask]) ** 2))


def mse_breakdown(log_true: np.ndarray, pred: np.ndarray, log_obs: np.ndarray) -> Dict[str, float]:
    diff = log_true - pred
    mask_nonzero = log_true > 0.0
    mask_biozero = log_true == 0.0
    mask_dropout = (log_true > 0.0) & (log_obs <= 0.0)
    return {
        "mse": float(np.mean(diff ** 2)),
        "mse_nonzero": _masked_mse(diff, mask_nonzero),
        "mse_biozero": _masked_mse(diff, mask_biozero),
        "mse_dropout": _masked_mse(diff, mask_dropout),
    }


def nanmean_safe(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def train_autoencoder_reconstruct(
    logcounts: np.ndarray,
    device: torch.device,
    hidden: List[int],
    bottleneck: int,
) -> np.ndarray:
    scale_on = True
    scaler = RobustZThenMinMaxToNeg1Pos1().fit(logcounts) if scale_on else IdentityScaler().fit(logcounts)
    Xs = scaler.transform(logcounts).astype(np.float32)

    zeros_indicator = (logcounts <= 0.0).astype(np.float32)

    Xtr = torch.tensor(Xs, dtype=torch.float32)
    Ztr = torch.tensor(zeros_indicator, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, Ztr), batch_size=32, shuffle=True, drop_last=False)

    model = AE(input_dim=logcounts.shape[1], hidden=hidden, bottleneck=bottleneck, layer_type=DENSE_LAYER_CONST.SILU_LAYER).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    # Train
    model.train()
    for _ in range(100):
        for xb, zb in loader:
            xb = xb.to(device)
            zb = zb.to(device)
            # Masked-denoising objective (fixed settings)
            p_zero = 0.01
            p_nz = 0.30
            probs = torch.where(
                zb > 0.5,
                torch.full_like(xb, p_zero),
                torch.full_like(xb, p_nz),
            )
            mask = torch.bernoulli(probs)  # 1 = hide
            fill = torch.zeros_like(xb)  # fill=zero
            mask = mask.to(xb.dtype)
            x_in = (1.0 - mask) * xb + mask * fill
            x_tgt = xb

            opt.zero_grad()
            recon = model(x_in)
            residual = recon - x_tgt
            loss = mse_from_residual(residual, mask=mask)
            loss.backward()
            opt.step()

    # Reconstructions
    model.eval()
    recon_list = []
    with torch.no_grad():
        for i in range(0, Xtr.size(0), 32):
            xb = Xtr[i : i + 32].to(device)
            recon = model(xb)
            recon_np = recon.cpu().numpy()
            recon_orig = scaler.inverse_transform(recon_np)
            recon_list.append(recon_orig)
    recon_all = np.vstack(recon_list)
    return recon_all.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="experiment2: fixed settings with SPLAT cell-aware post-processing.")
    parser.add_argument("data_dir", type=str, help="Directory containing .rds files (searched recursively).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=== Settings ===")
    print(" Mode     : MSE")
    print(" Layers   : SILU")
    print(" Model    : hidden=64  bottleneck=32")
    print(" Train    : epochs=100  batch=32  lr=0.001  wd=0.0")
    print(" Repeats  : 10 (seed base=42)")
    print(f" Device   : {args.device}")
    print(" Scale    : on")
    print(" MaskDeno : ON  p_zero=0.01  p_nonzero=0.3  fill=zero  noise_std=0.3")
    print("================")

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    files = sorted(str(p) for p in data_dir.rglob("*.rds"))
    if not files:
        raise FileNotFoundError(f"No .rds files found under: {data_dir}")

    hidden = [64]
    bottleneck = 32
    repeats = 10
    seed_base = 42

    dataset_runs: List[Dict[str, object]] = []
    for path in files:
        ds_name = Path(path).stem
        logcounts, log_true = load_dataset(path, need_labels=False, need_truth=True)
        if log_true is None:
            print(f"[WARN] Dataset '{ds_name}' lacks 'logTrueCounts'; skipping.")
            continue

        # --- Static masks / labels (log2(1+norm) scale) ---
        mask_nonzero = log_true > 0.0
        mask_biozero = log_true == 0.0
        mask_dropout = (log_true > 0.0) & (logcounts <= 0.0)

        n_total = int(log_true.size)
        n_nonzero = int(mask_nonzero.sum())
        n_biozero = int(mask_biozero.sum())
        n_dropout = int(mask_dropout.sum())

        # --- Baseline MSE (true vs observed), in the same log space ---
        diff_base = (log_true - logcounts).astype(np.float64)
        baseline_sse_total = float(np.sum(diff_base**2))
        baseline_sse_nonzero = float(np.sum((diff_base[mask_nonzero]) ** 2))
        baseline_sse_biozero = float(np.sum((diff_base[mask_biozero]) ** 2))
        baseline_sse_dropout = float(np.sum((diff_base[mask_dropout]) ** 2))

        # --- Count-space matrices for SPLAT (reverse of log2(1+·)) ---
        counts_obs = np.clip(logcounts_to_counts(logcounts), 0.0, None)
        counts_true = np.clip(logcounts_to_counts(log_true), 0.0, None)
        zeros_obs = counts_obs <= 0.0
        bio_true_obs = zeros_obs & (counts_true <= 0.0)
        drop_true_obs = zeros_obs & (counts_true > 0.0)

        bio_true_z = bio_true_obs[zeros_obs].astype(bool)
        drop_true_z = drop_true_obs[zeros_obs].astype(bool)

        # Cache observed-counts posteriors/predictions at observed-zeros for global threshold tuning.
        obs_cache: Dict[str, Dict[str, object]] = {}
        for approach in APPROACHES:
            name = str(approach["name"])
            if str(approach["kind"]) == "baseline":
                obs_cache[name] = {"pred_dropout_z": _baseline_pred_dropout_at_zeros(counts_obs, zeros_obs)}
            else:
                obs_cache[name] = {"p_bio_z": _splat_bio_posterior_at_zeros(approach, counts_obs, zeros_obs)}

        # Precompute log_true at observed-zero positions for delta computation.
        log_true_z = np.asarray(log_true, dtype=np.float64)[zeros_obs].reshape(-1)

        repeat_runs: List[Dict[str, object]] = []
        for rep in range(repeats):
            seed = seed_base + rep
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            recon_all = train_autoencoder_reconstruct(logcounts, device, hidden=hidden, bottleneck=bottleneck)

            diff_raw = (log_true - recon_all).astype(np.float64)
            raw_sse_total = float(np.sum(diff_raw**2))
            raw_sse_nonzero = float(np.sum((diff_raw[mask_nonzero]) ** 2))
            raw_sse_biozero = float(np.sum((diff_raw[mask_biozero]) ** 2))
            raw_sse_dropout = float(np.sum((diff_raw[mask_dropout]) ** 2))

            # Delta in SSE if we hard-zero imputed logcounts at observed-zero positions.
            pred_z = np.asarray(recon_all, dtype=np.float64)[zeros_obs].reshape(-1)
            old_err = (log_true_z - pred_z) ** 2
            new_err = log_true_z**2
            delta_z = (new_err - old_err).astype(np.float64)

            # Build imputed-counts matrix preserving observed-zero mask.
            counts_imputed = np.clip(logcounts_to_counts(recon_all), 0.0, None)
            counts_for_imp = counts_imputed.copy()
            counts_for_imp[zeros_obs] = 0.0

            imp_cache: Dict[str, Dict[str, object]] = {}
            for approach in APPROACHES:
                name = str(approach["name"])
                if str(approach["kind"]) == "baseline":
                    imp_cache[name] = {"pred_dropout_z": _baseline_pred_dropout_at_zeros(counts_for_imp, zeros_obs)}
                else:
                    imp_cache[name] = {"p_bio_z": _splat_bio_posterior_at_zeros(approach, counts_for_imp, zeros_obs)}

            repeat_runs.append(
                {
                    "raw_sse_total": raw_sse_total,
                    "raw_sse_nonzero": raw_sse_nonzero,
                    "raw_sse_biozero": raw_sse_biozero,
                    "raw_sse_dropout": raw_sse_dropout,
                    "delta_z": delta_z,
                    "pred_z": pred_z.astype(np.float32, copy=False),
                    "imp": imp_cache,
                }
            )

        if not repeat_runs:
            continue

        dataset_runs.append(
            {
                "dataset": ds_name,
                "n_total": n_total,
                "n_nonzero": n_nonzero,
                "n_biozero": n_biozero,
                "n_dropout": n_dropout,
                "baseline_sse_total": baseline_sse_total,
                "baseline_sse_nonzero": baseline_sse_nonzero,
                "baseline_sse_biozero": baseline_sse_biozero,
                "baseline_sse_dropout": baseline_sse_dropout,
                "bio_true_z": bio_true_z,
                "drop_true_z": drop_true_z,
                "obs": obs_cache,
                "repeats": repeat_runs,
            }
        )

    if dataset_runs:
        # ---- Global thresholds (single value shared across datasets) ----
        thr_obs_global: Dict[str, float] = {}
        thr_imp_global: Dict[str, float] = {}
        thr_bio_global: Dict[str, float] = {}

        # Precompute global denominators and raw SSEs for the thrBio objective.
        global_n_bz = 0
        global_n_do = 0
        global_raw_sse_bz = 0.0
        global_raw_sse_do = 0.0
        for ds in dataset_runs:
            n_rep = len(ds["repeats"])  # type: ignore[arg-type]
            global_n_bz += int(ds["n_biozero"]) * int(n_rep)
            global_n_do += int(ds["n_dropout"]) * int(n_rep)
            for rep in ds["repeats"]:  # type: ignore[assignment]
                global_raw_sse_bz += float(rep["raw_sse_biozero"])
                global_raw_sse_do += float(rep["raw_sse_dropout"])

        for approach in APPROACHES:
            name = str(approach["name"])
            kind = str(approach["kind"])

            if kind == "baseline":
                thr_obs_global[name] = float("nan")
                thr_imp_global[name] = float("nan")
            else:
                pdrop_obs_list: List[np.ndarray] = []
                bio_obs_list: List[np.ndarray] = []
                drop_obs_list: List[np.ndarray] = []
                for ds in dataset_runs:
                    p_bio_z = np.asarray(ds["obs"][name]["p_bio_z"], dtype=np.float64)  # type: ignore[index]
                    pdrop_obs_list.append(1.0 - p_bio_z)
                    bio_obs_list.append(np.asarray(ds["bio_true_z"], dtype=bool))  # type: ignore[list-item]
                    drop_obs_list.append(np.asarray(ds["drop_true_z"], dtype=bool))  # type: ignore[list-item]
                thr_obs_global[name] = _choose_global_f1_thresh(pdrop_obs_list, bio_obs_list, drop_obs_list)

                pdrop_imp_list: List[np.ndarray] = []
                bio_imp_list: List[np.ndarray] = []
                drop_imp_list: List[np.ndarray] = []
                for ds in dataset_runs:
                    bz = np.asarray(ds["bio_true_z"], dtype=bool)
                    do = np.asarray(ds["drop_true_z"], dtype=bool)
                    for rep in ds["repeats"]:  # type: ignore[assignment]
                        p_bio_z = np.asarray(rep["imp"][name]["p_bio_z"], dtype=np.float64)  # type: ignore[index]
                        pdrop_imp_list.append(1.0 - p_bio_z)
                        bio_imp_list.append(bz)
                        drop_imp_list.append(do)
                thr_imp_global[name] = _choose_global_f1_thresh(pdrop_imp_list, bio_imp_list, drop_imp_list)

            # thrBio (imputed-values table) is chosen globally for every approach (including baseline).
            p_bio_list: List[np.ndarray] = []
            delta_list: List[np.ndarray] = []
            bz_list: List[np.ndarray] = []
            do_list: List[np.ndarray] = []
            for ds in dataset_runs:
                bz = np.asarray(ds["bio_true_z"], dtype=bool)
                do = np.asarray(ds["drop_true_z"], dtype=bool)
                for rep in ds["repeats"]:  # type: ignore[assignment]
                    delta_list.append(np.asarray(rep["delta_z"], dtype=np.float64))
                    bz_list.append(bz)
                    do_list.append(do)
                    if kind == "baseline":
                        pred_drop_z = np.asarray(rep["imp"][name]["pred_dropout_z"], dtype=bool)  # type: ignore[index]
                        p_bio_list.append((~pred_drop_z).astype(np.float64))
                    else:
                        p_bio_list.append(np.asarray(rep["imp"][name]["p_bio_z"], dtype=np.float64))  # type: ignore[index]

            thr_bio, _best_obj = _choose_global_thr_bio_for_min_arith_mean_bz_do_mse(
                p_bio_list,
                delta_list,
                bz_list,
                do_list,
                base_sum_bz=global_raw_sse_bz,
                base_sum_do=global_raw_sse_do,
                n_bz=global_n_bz,
                n_do=global_n_do,
            )
            thr_bio_global[name] = float(thr_bio)

        # ---- Build per-dataset result rows (averaged over repeats) ----
        def _mse(sse: float, denom: int) -> float:
            if denom <= 0 or not np.isfinite(sse):
                return float("nan")
            return float(sse) / float(denom)

        def _adj_mse_from_delta(
            base_sse_total: float,
            base_sse_nonzero: float,
            base_sse_biozero: float,
            base_sse_dropout: float,
            delta_z: np.ndarray,
            pred_bio_z: np.ndarray,
            bio_true_z: np.ndarray,
            drop_true_z: np.ndarray,
            n_total: int,
            n_nonzero: int,
            n_biozero: int,
            n_dropout: int,
        ) -> Dict[str, float]:
            pred_bio_z = np.asarray(pred_bio_z, dtype=bool).reshape(-1)
            bio_true_z = np.asarray(bio_true_z, dtype=bool).reshape(-1)
            drop_true_z = np.asarray(drop_true_z, dtype=bool).reshape(-1)
            delta_z = np.asarray(delta_z, dtype=np.float64).reshape(-1)

            d_total = float(np.sum(delta_z[pred_bio_z])) if int(pred_bio_z.sum()) > 0 else 0.0
            d_bz = float(np.sum(delta_z[pred_bio_z & bio_true_z])) if int((pred_bio_z & bio_true_z).sum()) > 0 else 0.0
            d_do = float(np.sum(delta_z[pred_bio_z & drop_true_z])) if int((pred_bio_z & drop_true_z).sum()) > 0 else 0.0

            sse_total = max(0.0, float(base_sse_total) + d_total)
            sse_nonzero = max(0.0, float(base_sse_nonzero) + d_do)
            sse_biozero = max(0.0, float(base_sse_biozero) + d_bz)
            sse_dropout = max(0.0, float(base_sse_dropout) + d_do)

            return {
                "mse": _mse(sse_total, n_total),
                "mse_nonzero": _mse(sse_nonzero, n_nonzero),
                "mse_biozero": _mse(sse_biozero, n_biozero),
                "mse_dropout": _mse(sse_dropout, n_dropout),
            }

        results: List[Dict[str, object]] = []
        for ds in dataset_runs:
            ds_name = str(ds["dataset"])
            n_total = int(ds["n_total"])
            n_nonzero = int(ds["n_nonzero"])
            n_biozero = int(ds["n_biozero"])
            n_dropout = int(ds["n_dropout"])
            bio_true_z = np.asarray(ds["bio_true_z"], dtype=bool)
            drop_true_z = np.asarray(ds["drop_true_z"], dtype=bool)

            repeats_list = ds["repeats"]  # type: ignore[assignment]

            row: Dict[str, object] = {"dataset": ds_name}

            row["mse_raw"] = nanmean_safe([_mse(float(r["raw_sse_total"]), n_total) for r in repeats_list])
            row["mse_raw_nonzero"] = nanmean_safe([_mse(float(r["raw_sse_nonzero"]), n_nonzero) for r in repeats_list])
            row["mse_raw_biozero"] = nanmean_safe([_mse(float(r["raw_sse_biozero"]), n_biozero) for r in repeats_list])
            row["mse_raw_dropout"] = nanmean_safe([_mse(float(r["raw_sse_dropout"]), n_dropout) for r in repeats_list])

            row["baseline_mse"] = _mse(float(ds["baseline_sse_total"]), n_total)
            row["baseline_mse_nonzero"] = _mse(float(ds["baseline_sse_nonzero"]), n_nonzero)
            row["baseline_mse_biozero"] = _mse(float(ds["baseline_sse_biozero"]), n_biozero)
            row["baseline_mse_dropout"] = _mse(float(ds["baseline_sse_dropout"]), n_dropout)

            for approach in APPROACHES:
                name = str(approach["name"])
                kind = str(approach["kind"])

                # (A) Observed-count predictions (global thr_obs)
                if kind == "baseline":
                    pred_dropout_z_obs = np.asarray(ds["obs"][name]["pred_dropout_z"], dtype=bool)  # type: ignore[index]
                    thr_obs = float("nan")
                else:
                    p_bio_z_obs = np.asarray(ds["obs"][name]["p_bio_z"], dtype=np.float64)  # type: ignore[index]
                    p_drop_z_obs = 1.0 - p_bio_z_obs
                    thr_obs = float(thr_obs_global[name])
                    pred_dropout_z_obs = p_drop_z_obs >= thr_obs

                pred_bio_z_obs = ~pred_dropout_z_obs
                row[f"{name}_thr_obs"] = float(thr_obs)
                met_obs = bio_zero_metrics(
                    np.ones_like(bio_true_z, dtype=bool),
                    bio_true_z,
                    drop_true_z,
                    pred_dropout_z_obs,
                )
                for k, v in met_obs.items():
                    row[f"{name}_obs_{k}"] = float(v)

                mse_obs_list = []
                mse_obs_nz_list = []
                mse_obs_bz_list = []
                mse_obs_do_list = []
                for r in repeats_list:
                    adj = _adj_mse_from_delta(
                        base_sse_total=float(r["raw_sse_total"]),
                        base_sse_nonzero=float(r["raw_sse_nonzero"]),
                        base_sse_biozero=float(r["raw_sse_biozero"]),
                        base_sse_dropout=float(r["raw_sse_dropout"]),
                        delta_z=np.asarray(r["delta_z"], dtype=np.float64),
                        pred_bio_z=pred_bio_z_obs,
                        bio_true_z=bio_true_z,
                        drop_true_z=drop_true_z,
                        n_total=n_total,
                        n_nonzero=n_nonzero,
                        n_biozero=n_biozero,
                        n_dropout=n_dropout,
                    )
                    mse_obs_list.append(adj["mse"])
                    mse_obs_nz_list.append(adj["mse_nonzero"])
                    mse_obs_bz_list.append(adj["mse_biozero"])
                    mse_obs_do_list.append(adj["mse_dropout"])

                row[f"{name}_mse_obs"] = nanmean_safe(mse_obs_list)
                row[f"{name}_mse_obs_nonzero"] = nanmean_safe(mse_obs_nz_list)
                row[f"{name}_mse_obs_biozero"] = nanmean_safe(mse_obs_bz_list)
                row[f"{name}_mse_obs_dropout"] = nanmean_safe(mse_obs_do_list)

                hp_thr_obs, hp_mse_obs = tune_hp_threshold_min_mse(
                    repeats_list,
                    forced_masks=[pred_bio_z_obs for _ in repeats_list],
                    n_total=n_total,
                )
                row[f"{name}_hp_thr_obs"] = float(hp_thr_obs)
                row[f"{name}_hp_mse_obs"] = float(hp_mse_obs)

                # (B) Imputed-values predictions (global thr_imp)
                if kind == "baseline":
                    thr_imp = float("nan")
                else:
                    thr_imp = float(thr_imp_global[name])
                row[f"{name}_thr_imp"] = float(thr_imp)

                met_imp_list = []
                mse_imp_list = []
                mse_imp_nz_list = []
                mse_imp_bz_list = []
                mse_imp_do_list = []
                forced_imp_masks: List[np.ndarray] = []
                for r in repeats_list:
                    if kind == "baseline":
                        pred_dropout_z_imp = np.asarray(r["imp"][name]["pred_dropout_z"], dtype=bool)  # type: ignore[index]
                    else:
                        p_bio_z_imp = np.asarray(r["imp"][name]["p_bio_z"], dtype=np.float64)  # type: ignore[index]
                        p_drop_z_imp = 1.0 - p_bio_z_imp
                        pred_dropout_z_imp = p_drop_z_imp >= float(thr_imp)

                    pred_bio_z_imp = ~pred_dropout_z_imp
                    forced_imp_masks.append(pred_bio_z_imp)
                    met_imp_list.append(
                        bio_zero_metrics(
                            np.ones_like(bio_true_z, dtype=bool),
                            bio_true_z,
                            drop_true_z,
                            pred_dropout_z_imp,
                        )
                    )
                    adj = _adj_mse_from_delta(
                        base_sse_total=float(r["raw_sse_total"]),
                        base_sse_nonzero=float(r["raw_sse_nonzero"]),
                        base_sse_biozero=float(r["raw_sse_biozero"]),
                        base_sse_dropout=float(r["raw_sse_dropout"]),
                        delta_z=np.asarray(r["delta_z"], dtype=np.float64),
                        pred_bio_z=pred_bio_z_imp,
                        bio_true_z=bio_true_z,
                        drop_true_z=drop_true_z,
                        n_total=n_total,
                        n_nonzero=n_nonzero,
                        n_biozero=n_biozero,
                        n_dropout=n_dropout,
                    )
                    mse_imp_list.append(adj["mse"])
                    mse_imp_nz_list.append(adj["mse_nonzero"])
                    mse_imp_bz_list.append(adj["mse_biozero"])
                    mse_imp_do_list.append(adj["mse_dropout"])

                row[f"{name}_mse_imp"] = nanmean_safe(mse_imp_list)
                row[f"{name}_mse_imp_nonzero"] = nanmean_safe(mse_imp_nz_list)
                row[f"{name}_mse_imp_biozero"] = nanmean_safe(mse_imp_bz_list)
                row[f"{name}_mse_imp_dropout"] = nanmean_safe(mse_imp_do_list)

                for k in [
                    "total_zeros",
                    "true_bio_zeros",
                    "true_dropouts",
                    "predicted_bio",
                    "predicted_dropouts",
                    "Precision_Bio",
                    "Recall_Bio",
                    "F1_Score_Bio",
                    "Accuracy_Bio",
                ]:
                    row[f"{name}_imp_{k}"] = nanmean_safe([float(m.get(k, float("nan"))) for m in met_imp_list])

                hp_thr_imp, hp_mse_imp = tune_hp_threshold_min_mse(
                    repeats_list,
                    forced_masks=forced_imp_masks,
                    n_total=n_total,
                )
                row[f"{name}_hp_thr_imp"] = float(hp_thr_imp)
                row[f"{name}_hp_mse_imp"] = float(hp_mse_imp)

                # (C) Imputed-values adjustment with global thrBio
                thr_bio = float(thr_bio_global[name])
                row[f"{name}_thr_imp_hm"] = thr_bio

                mse_hm_list = []
                mse_hm_nz_list = []
                mse_hm_bz_list = []
                mse_hm_do_list = []
                forced_hm_masks: List[np.ndarray] = []
                for r in repeats_list:
                    if kind == "baseline":
                        pred_drop_z = np.asarray(r["imp"][name]["pred_dropout_z"], dtype=bool)  # type: ignore[index]
                        p_bio_z = (~pred_drop_z).astype(np.float64)
                    else:
                        p_bio_z = np.asarray(r["imp"][name]["p_bio_z"], dtype=np.float64)  # type: ignore[index]

                    pred_bio_z = p_bio_z >= thr_bio
                    forced_hm_masks.append(pred_bio_z)
                    adj = _adj_mse_from_delta(
                        base_sse_total=float(r["raw_sse_total"]),
                        base_sse_nonzero=float(r["raw_sse_nonzero"]),
                        base_sse_biozero=float(r["raw_sse_biozero"]),
                        base_sse_dropout=float(r["raw_sse_dropout"]),
                        delta_z=np.asarray(r["delta_z"], dtype=np.float64),
                        pred_bio_z=pred_bio_z,
                        bio_true_z=bio_true_z,
                        drop_true_z=drop_true_z,
                        n_total=n_total,
                        n_nonzero=n_nonzero,
                        n_biozero=n_biozero,
                        n_dropout=n_dropout,
                    )
                    mse_hm_list.append(adj["mse"])
                    mse_hm_nz_list.append(adj["mse_nonzero"])
                    mse_hm_bz_list.append(adj["mse_biozero"])
                    mse_hm_do_list.append(adj["mse_dropout"])

                row[f"{name}_mse_imp_hm"] = nanmean_safe(mse_hm_list)
                row[f"{name}_mse_imp_hm_nonzero"] = nanmean_safe(mse_hm_nz_list)
                row[f"{name}_mse_imp_hm_biozero"] = nanmean_safe(mse_hm_bz_list)
                row[f"{name}_mse_imp_hm_dropout"] = nanmean_safe(mse_hm_do_list)

                hp_thr_hm, hp_mse_hm = tune_hp_threshold_min_mse(
                    repeats_list,
                    forced_masks=forced_hm_masks,
                    n_total=n_total,
                )
                row[f"{name}_hp_thr_imp_hm"] = float(hp_thr_hm)
                row[f"{name}_hp_mse_imp_hm"] = float(hp_mse_hm)

            results.append(row)

        # ---- Reporting ----
        DATASET_W = 44

        def _fmt(x: float) -> str:
            return "NA" if not np.isfinite(x) else f"{x:.3f}"

        print("\n=== Per-dataset MSE (raw vs baseline; averaged over repeats) ===")
        header_raw = (
            f"{'Dataset':<{DATASET_W}}"
            f"{'MSE':>8} {'NonZero':>8} {'BioZero':>8} {'Dropout':>8}  "
            f"{'BASE_MSE':>9} {'BASE_NZ':>8} {'BASE_BZ':>8} {'BASE_DO':>8}"
        )
        print(header_raw)
        for row in results:
            print(
                f"{row['dataset']:<{DATASET_W}}"
                f"{_fmt(float(row['mse_raw'])):>8} {_fmt(float(row['mse_raw_nonzero'])):>8} {_fmt(float(row['mse_raw_biozero'])):>8} {_fmt(float(row['mse_raw_dropout'])):>8}  "
                f"{_fmt(float(row['baseline_mse'])):>9} {_fmt(float(row['baseline_mse_nonzero'])):>8} {_fmt(float(row['baseline_mse_biozero'])):>8} {_fmt(float(row['baseline_mse_dropout'])):>8}"
            )

        avg_raw = {
            k: nanmean_safe([float(r.get(k, float("nan"))) for r in results])
            for k in [
                "mse_raw",
                "mse_raw_nonzero",
                "mse_raw_biozero",
                "mse_raw_dropout",
                "baseline_mse",
                "baseline_mse_nonzero",
                "baseline_mse_biozero",
                "baseline_mse_dropout",
            ]
        }
        print(
            f"{'AVG':<{DATASET_W}}"
            f"{_fmt(avg_raw['mse_raw']):>8} {_fmt(avg_raw['mse_raw_nonzero']):>8} {_fmt(avg_raw['mse_raw_biozero']):>8} {_fmt(avg_raw['mse_raw_dropout']):>8}  "
            f"{_fmt(avg_raw['baseline_mse']):>9} {_fmt(avg_raw['baseline_mse_nonzero']):>8} {_fmt(avg_raw['baseline_mse_biozero']):>8} {_fmt(avg_raw['baseline_mse_dropout']):>8}"
        )

        def _fmt_cnt(x: float) -> str:
            return "NA" if not np.isfinite(x) else f"{x:.1f}"

        def _fmt_pct(x: float) -> str:
            return "NA" if not np.isfinite(x) else f"{x:.2f}%"

        def _print_bio_summary(variant: str):
            print(f"\n=== Bio-zero prediction summary ({variant}) ===")
            print(
                f"{'approach':>20} {'n_datasets':>10} {'total_zeros':>11} {'true_bio_zeros':>13} {'true_dropouts':>12} "
                f"{'predicted_bio':>13} {'predicted_dropouts':>17} {'Precision_Bio':>14} {'Recall_Bio':>10} "
                f"{'F1_Score_Bio':>12} {'Accuracy_Bio':>12}"
            )
            for approach in APPROACHES:
                name = str(approach["name"])
                prefix = f"{name}_{variant}_"
                keys = [
                    "total_zeros",
                    "true_bio_zeros",
                    "true_dropouts",
                    "predicted_bio",
                    "predicted_dropouts",
                    "Precision_Bio",
                    "Recall_Bio",
                    "F1_Score_Bio",
                    "Accuracy_Bio",
                ]
                vals = {k: nanmean_safe([float(r.get(prefix + k, float("nan"))) for r in results]) for k in keys}
                n_datasets = sum(1 for r in results if np.isfinite(r.get(prefix + "total_zeros", float("nan"))))
                print(
                    f"{name:>20} {n_datasets:>10d} { _fmt_cnt(vals['total_zeros']):>11} { _fmt_cnt(vals['true_bio_zeros']):>13} { _fmt_cnt(vals['true_dropouts']):>12} "
                    f"{ _fmt_cnt(vals['predicted_bio']):>13} { _fmt_cnt(vals['predicted_dropouts']):>17} { _fmt_pct(vals['Precision_Bio']):>14} { _fmt_pct(vals['Recall_Bio']):>10} "
                    f"{ _fmt_pct(vals['F1_Score_Bio']):>12} { _fmt_pct(vals['Accuracy_Bio']):>12}"
                )

        _print_bio_summary("obs")
        _print_bio_summary("imp")

        def _print_adjusted_mse_table(name: str, variant: str, source_label: str, thr_header: str = "thr") -> None:
            thr_key = f"{name}_thr_{variant}"
            mse_key = f"{name}_mse_{variant}"
            nz_key = f"{name}_mse_{variant}_nonzero"
            bz_key = f"{name}_mse_{variant}_biozero"
            do_key = f"{name}_mse_{variant}_dropout"
            hp_thr_key = f"{name}_hp_thr_{variant}"
            hp_mse_key = f"{name}_hp_mse_{variant}"

            print(f"\n=== Adjusted MSE ({name}; predictions from {source_label}) ===")
            print(
                f"{'Dataset':<{DATASET_W}}{thr_header:>8}{'MSE':>8} {'NonZero':>8} {'BioZero':>8} {'Dropout':>8} "
                f"{'HP_thr':>8} {'HP_MSE':>8}"
            )
            for row in results:
                print(
                    f"{row['dataset']:<{DATASET_W}}{_fmt(float(row.get(thr_key, float('nan')))):>8}"
                    f"{_fmt(float(row.get(mse_key, float('nan')))):>8} {_fmt(float(row.get(nz_key, float('nan')))):>8} {_fmt(float(row.get(bz_key, float('nan')))):>8} {_fmt(float(row.get(do_key, float('nan')))):>8} "
                    f"{_fmt(float(row.get(hp_thr_key, float('nan')))):>8} {_fmt(float(row.get(hp_mse_key, float('nan')))):>8}"
                )

            avg = {
                "thr": nanmean_safe([float(r.get(thr_key, float("nan"))) for r in results]),
                "mse": nanmean_safe([float(r.get(mse_key, float("nan"))) for r in results]),
                "nz": nanmean_safe([float(r.get(nz_key, float("nan"))) for r in results]),
                "bz": nanmean_safe([float(r.get(bz_key, float("nan"))) for r in results]),
                "do": nanmean_safe([float(r.get(do_key, float("nan"))) for r in results]),
                "hp_thr": nanmean_safe([float(r.get(hp_thr_key, float("nan"))) for r in results]),
                "hp_mse": nanmean_safe([float(r.get(hp_mse_key, float("nan"))) for r in results]),
            }
            print(
                f"{'AVG':<{DATASET_W}}{_fmt(avg['thr']):>8}"
                f"{_fmt(avg['mse']):>8} {_fmt(avg['nz']):>8} {_fmt(avg['bz']):>8} {_fmt(avg['do']):>8} "
                f"{_fmt(avg['hp_thr']):>8} {_fmt(avg['hp_mse']):>8}"
            )

        for approach in APPROACHES:
            name = str(approach["name"])
            _print_adjusted_mse_table(name, "obs", "observed counts")
            _print_adjusted_mse_table(name, "imp", "imputed values")
            _print_adjusted_mse_table(name, "imp_hm", "imputed values (thrBio@HM-impr)", thr_header="thrBio")


if __name__ == "__main__":
    main()
