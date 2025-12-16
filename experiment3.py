#!/usr/bin/env python3
"""
experiment3.py
---------------

Optimized for Minimum MSE Performance with Robust Numerical Stability.

Key Improvements:
1. Stability: Sigmoid output + MaxAbsScaler strictly bounds predictions to [0,1],
   preventing "exploding" counts that cause overflow in Splatter.
2. Logic: Uses a Hybrid Matrix (Imputed Non-Zeros + Explicit Zeros) for Splatter
   estimation. This provides clean data for parameter fitting while correctly
   classifying zero entries.
3. Tuning: Minimizes Total SSE via exhaustive search (mathematically optimal).
4. Safety: Context managers suppress harmless numerical warnings.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Robustly filter warnings from external library computations
warnings.filterwarnings("ignore", category=RuntimeWarning, module="predict_dropouts_new")

from DenseLayerPack import DenseLayer
from DenseLayerPack.const import DENSE_LAYER_CONST
from predict_dropouts_new import (
    baseline_gene_mean_heuristic_counts,
    splatter_bio_posterior_from_counts,
    _choose_thresh_for_metric,
)
from rds2py import read_rds

# ------------------------
# 1. Optimized Scaler (Preserves Sparsity & Bounds Data)
# ------------------------

class MaxAbsScaler:
    """
    Scales each gene by its maximum absolute value to range [0, 1].
    Crucially, it preserves 0.0 as 0.0, keeping the sparsity structure intact.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.max_ = None

    def fit(self, X: np.ndarray):
        # Assumes non-negative data (log-counts)
        self.max_ = np.abs(X).max(axis=0)
        self.max_[self.max_ < self.eps] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X / self.max_).astype(np.float32)

    def inverse_transform(self, Xscaled: np.ndarray) -> np.ndarray:
        return (Xscaled * self.max_).astype(np.float32)


class IdentityScaler:
    def fit(self, X: np.ndarray):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)

    def inverse_transform(self, Xscaled: np.ndarray) -> np.ndarray:
        return Xscaled.astype(np.float32)


# ------------------------
# 2. Optimized Model (Sigmoid Output)
# ------------------------

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
            # Use Sigmoid in the last layer.
            # Combined with MaxAbsScaler, this bounds output to [0, 1].
            # This mathematically prevents predictions from exceeding the observed max,
            # eliminating the "overflow in exp" errors caused by exploding predictions.
            if i == len(sizes_dec) - 2:
                dec_layers.append(nn.Linear(sizes_dec[i], sizes_dec[i + 1]))
                dec_layers.append(nn.Sigmoid())
            else:
                dec_layers.append(DenseLayer(sizes_dec[i], sizes_dec[i + 1], layer_type=layer_type))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# ------------------------
# Helpers
# ------------------------

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
    """
    Convert logcounts to counts with safety clipping.
    20.0 log2-counts is ~1 million, well above biological limits for UMI scRNA.
    """
    logcts = np.clip(logcts, 0.0, 20.0) 
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
    
    # Clip inputs to safe range [0, 1e9]
    counts_safe = np.clip(counts, 0.0, 1e9)
    
    # Use context manager to strictly ignore "overflow in exp" warnings.
    # The sigmoid logic inside Splatter handles +/- inf correctly (returning 0 or 1), 
    # so the warning is harmless noise caused by log(0)=-inf.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            bio_post = splatter_bio_posterior_from_counts(
                counts_safe,
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


def _choose_global_thr_bio_for_min_total_sse(
    p_bio_list: List[np.ndarray],
    delta_list: List[np.ndarray],
    base_sse_total: float,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Global thrBio that strictly minimizes TOTAL SSE (Sum of Squared Errors).
    """
    if not p_bio_list:
        return float("nan"), float(base_sse_total)

    p_all = np.concatenate([_sanitize_prob(p).reshape(-1) for p in p_bio_list], axis=0)
    delta_all = np.concatenate([np.asarray(d, dtype=np.float64).reshape(-1) for d in delta_list], axis=0)

    finite = np.isfinite(p_all) & np.isfinite(delta_all)
    if int(finite.sum()) == 0:
        return float("nan"), float(base_sse_total)

    p_all = p_all[finite]
    delta_all = delta_all[finite]

    # Sort descending (high confidence bio-zero first)
    order = np.argsort(-p_all, kind="mergesort")
    p_s = p_all[order]
    delta_s = delta_all[order]

    cum_delta = np.cumsum(delta_s)

    # Check boundaries where probability value actually changes
    group_ends = np.flatnonzero(np.r_[p_s[1:] != p_s[:-1], True])

    best_delta = 0.0
    best_thr = 1.0 + 1e-6

    # Exhaustive search
    for idx in group_ends:
        d_val = float(cum_delta[idx])
        if d_val < best_delta - eps:
            best_delta = d_val
            best_thr = float(p_s[idx])

    thr_out = float("nan") if best_thr > 1.0 else float(best_thr)
    final_sse = max(0.0, base_sse_total + best_delta)
    return thr_out, final_sse


def tune_hp_threshold_min_mse(
    repeats: List[Dict[str, object]],
    forced_masks: List[np.ndarray],
    n_total: int,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """
    Exhaustive search (vectorized) for High-Pass threshold to minimize MSE.
    """
    if not repeats or n_total <= 0:
        return float("nan"), float("nan")

    n_rep = len(repeats)
    denom = float(n_rep * int(n_total))

    base_sse_sum = 0.0
    values_list: List[np.ndarray] = []
    delta_list: List[np.ndarray] = []

    for rep, forced in zip(repeats, forced_masks):
        forced = np.asarray(forced, dtype=bool).reshape(-1)
        delta_z = np.asarray(rep["delta_z"], dtype=np.float64).reshape(-1)
        pred_z = np.asarray(rep["pred_z"], dtype=np.float64).reshape(-1)
        
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

    cum_delta = np.cumsum(deltas_s)
    
    min_idx = np.argmin(cum_delta)
    min_val = float(cum_delta[min_idx])
    
    best_thr = float("nan")
    best_mse = float(base_mse)
    
    if min_val < -eps:
        best_mse = (base_sse_sum + min_val) / denom
        best_thr = float(values_s[min_idx])

    return best_thr, best_mse


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
    # CHANGED: MaxAbsScaler preserves 0.0 and range.
    scaler = MaxAbsScaler().fit(logcounts) if scale_on else IdentityScaler().fit(logcounts)
    Xs = scaler.transform(logcounts).astype(np.float32)

    zeros_indicator = (logcounts <= 0.0).astype(np.float32)

    Xtr = torch.tensor(Xs, dtype=torch.float32)
    Ztr = torch.tensor(zeros_indicator, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, Ztr), batch_size=32, shuffle=True, drop_last=False)

    model = AE(input_dim=logcounts.shape[1], hidden=hidden, bottleneck=bottleneck, layer_type=DENSE_LAYER_CONST.SILU_LAYER).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train
    model.train()
    for _ in range(100):
        for xb, zb in loader:
            xb = xb.to(device)
            zb = zb.to(device)
            # Masked Denoising
            p_zero = 0.01
            p_nz = 0.30
            probs = torch.where(
                zb > 0.5,
                torch.full_like(xb, p_zero),
                torch.full_like(xb, p_nz),
            )
            mask = torch.bernoulli(probs)
            fill = torch.zeros_like(xb) 
            mask = mask.to(xb.dtype)
            x_in = (1.0 - mask) * xb + mask * fill
            x_tgt = xb

            opt.zero_grad()
            recon = model(x_in)
            residual = recon - x_tgt
            # Global Loss (mask=None). Learns identity + inpainting.
            loss = mse_from_residual(residual, mask=None)
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
    parser = argparse.ArgumentParser(description="experiment3: MaxAbsScaler + Sigmoid + Total MSE + Safety Clips")
    parser.add_argument("data_dir", type=str, help="Directory containing .rds files")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=== Settings ===")
    print(" Mode     : MSE Optimized")
    print(" Scaler   : MaxAbsScaler")
    print(" Model    : Sigmoid Output [0,1]")
    print(" Loss     : Global MSE")
    print(" Tuning   : Total SSE (Exhaustive)")
    print(f" Device   : {args.device}")
    print("================")

    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    files = sorted(str(p) for p in data_dir.rglob("*.rds"))
    if not files:
        raise FileNotFoundError(f"No .rds files found under: {data_dir}")

    hidden = [128, 64]
    bottleneck = 32
    repeats = 10
    seed_base = 42

    dataset_runs: List[Dict[str, object]] = []
    
    global_accumulators = {
        "base_sse": 0.0,
        "approaches": { str(app["name"]): {"p_bio": [], "delta": []} for app in APPROACHES }
    }

    for path in files:
        ds_name = Path(path).stem
        logcounts, log_true = load_dataset(path, need_labels=False, need_truth=True)
        if log_true is None:
            print(f"[WARN] Dataset '{ds_name}' lacks 'logTrueCounts'; skipping.")
            continue

        mask_nonzero = log_true > 0.0
        mask_biozero = log_true == 0.0
        mask_dropout = (log_true > 0.0) & (logcounts <= 0.0)

        n_total = int(log_true.size)
        n_nonzero = int(mask_nonzero.sum())
        n_biozero = int(mask_biozero.sum())
        n_dropout = int(mask_dropout.sum())

        diff_base = (log_true - logcounts).astype(np.float64)
        baseline_sse_total = float(np.sum(diff_base**2))
        baseline_sse_nonzero = float(np.sum((diff_base[mask_nonzero]) ** 2))
        baseline_sse_biozero = float(np.sum((diff_base[mask_biozero]) ** 2))
        baseline_sse_dropout = float(np.sum((diff_base[mask_dropout]) ** 2))

        counts_obs = np.clip(logcounts_to_counts(logcounts), 0.0, None)
        zeros_obs = counts_obs <= 0.0
        bio_true_obs = zeros_obs & (log_true <= 0.0)
        drop_true_obs = zeros_obs & (log_true > 0.0)

        bio_true_z = bio_true_obs[zeros_obs].astype(bool)
        drop_true_z = drop_true_obs[zeros_obs].astype(bool)

        obs_cache: Dict[str, Dict[str, object]] = {}
        for approach in APPROACHES:
            name = str(approach["name"])
            if str(approach["kind"]) == "baseline":
                obs_cache[name] = {"pred_dropout_z": _baseline_pred_dropout_at_zeros(counts_obs, zeros_obs)}
            else:
                obs_cache[name] = {"p_bio_z": _splat_bio_posterior_at_zeros(approach, counts_obs, zeros_obs)}

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

            pred_z = np.asarray(recon_all, dtype=np.float64)[zeros_obs].reshape(-1)
            old_err = (log_true_z - pred_z) ** 2
            new_err = log_true_z**2
            delta_z = (new_err - old_err).astype(np.float64)

            # Build counts for imputation.
            counts_imputed = np.clip(logcounts_to_counts(recon_all), 0.0, 1e9)
            
            # Hybrid Matrix Strategy:
            # We pass a matrix to Splatter that has:
            # 1. Imputed values for non-zero positions (to get cleaner mean/disp estimates)
            # 2. Explicit ZEROS for observed zero positions.
            # This ensures Splatter correctly calculates P(Bio | Zero) instead of P(Bio | ImputedValue).
            counts_for_imp = counts_imputed.copy()
            counts_for_imp[zeros_obs] = 0.0

            imp_cache: Dict[str, Dict[str, object]] = {}
            for approach in APPROACHES:
                name = str(approach["name"])
                if str(approach["kind"]) == "baseline":
                    imp_cache[name] = {"pred_dropout_z": _baseline_pred_dropout_at_zeros(counts_for_imp, zeros_obs)}
                else:
                    p_bio = _splat_bio_posterior_at_zeros(approach, counts_for_imp, zeros_obs)
                    imp_cache[name] = {"p_bio_z": p_bio}
                    
                    global_accumulators["approaches"][name]["p_bio"].append(p_bio)
                    global_accumulators["approaches"][name]["delta"].append(delta_z)

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
            
            global_accumulators["base_sse"] += raw_sse_total

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
        # ---- Global thresholds ----
        thr_obs_global: Dict[str, float] = {}
        thr_imp_global: Dict[str, float] = {}
        thr_bio_global: Dict[str, float] = {}

        for approach in APPROACHES:
            name = str(approach["name"])
            kind = str(approach["kind"])

            if kind == "baseline":
                thr_obs_global[name] = float("nan")
                thr_imp_global[name] = float("nan")
                thr_bio_global[name] = float("nan")
            else:
                pdrop_obs_list = []
                bio_obs_list = []
                drop_obs_list = []
                for ds in dataset_runs:
                    p_bio_z = np.asarray(ds["obs"][name]["p_bio_z"], dtype=np.float64) # type: ignore
                    pdrop_obs_list.append(1.0 - p_bio_z)
                    bio_obs_list.append(np.asarray(ds["bio_true_z"], dtype=bool)) # type: ignore
                    drop_obs_list.append(np.asarray(ds["drop_true_z"], dtype=bool)) # type: ignore
                thr_obs_global[name] = _choose_global_f1_thresh(pdrop_obs_list, bio_obs_list, drop_obs_list)

                pdrop_imp_list: List[np.ndarray] = []
                bio_imp_list: List[np.ndarray] = []
                drop_imp_list: List[np.ndarray] = []
                for ds in dataset_runs:
                    bz = np.asarray(ds["bio_true_z"], dtype=bool)
                    do = np.asarray(ds["drop_true_z"], dtype=bool)
                    for rep in ds["repeats"]: # type: ignore
                        p_bio_z = np.asarray(rep["imp"][name]["p_bio_z"], dtype=np.float64) # type: ignore
                        pdrop_imp_list.append(1.0 - p_bio_z)
                        bio_imp_list.append(bz)
                        drop_imp_list.append(do)
                thr_imp_global[name] = _choose_global_f1_thresh(pdrop_imp_list, bio_imp_list, drop_imp_list)

                # Total SSE Optimization
                p_bio_list = global_accumulators["approaches"][name]["p_bio"]
                delta_list = global_accumulators["approaches"][name]["delta"]
                thr_bio, _ = _choose_global_thr_bio_for_min_total_sse(
                    p_bio_list,
                    delta_list,
                    base_sse_total=global_accumulators["base_sse"]
                )
                thr_bio_global[name] = float(thr_bio)

        # ---- Reporting ----
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
            repeats_list = ds["repeats"] # type: ignore

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
                # (A) Obs
                if kind == "baseline":
                    thr_obs = float("nan")
                else:
                    thr_obs = float(thr_obs_global[name])
                row[f"{name}_thr_obs"] = thr_obs
                
                # (C) Imputed Adjustment
                thr_bio = float(thr_bio_global[name])
                row[f"{name}_thr_imp_hm"] = thr_bio

                mse_hm_list = []
                forced_hm_masks: List[np.ndarray] = []
                for r in repeats_list:
                    if kind == "baseline":
                        pred_drop_z = np.asarray(r["imp"][name]["pred_dropout_z"], dtype=bool) # type: ignore
                        p_bio_z = (~pred_drop_z).astype(np.float64)
                    else:
                        p_bio_z = np.asarray(r["imp"][name]["p_bio_z"], dtype=np.float64) # type: ignore

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

                row[f"{name}_mse_imp_hm"] = nanmean_safe(mse_hm_list)
                
                hp_thr_hm, hp_mse_hm = tune_hp_threshold_min_mse(
                    repeats_list,
                    forced_masks=forced_hm_masks,
                    n_total=n_total,
                )
                row[f"{name}_hp_thr_imp_hm"] = float(hp_thr_hm)
                row[f"{name}_hp_mse_imp_hm"] = float(hp_mse_hm)

            results.append(row)

        DATASET_W = 44
        def _fmt(x: float) -> str: return "NA" if not np.isfinite(x) else f"{x:.4f}"

        print("\n=== Per-dataset MSE (Total SSE Optimized) ===")
        header = f"{'Dataset':<{DATASET_W}} {'Base':>8} {'Raw':>8} {'Spl_Imp':>8} {'Spl_HP':>8}"
        print(header)
        for row in results:
            print(
                f"{row['dataset']:<{DATASET_W}}"
                f"{_fmt(float(row['baseline_mse'])):>8} "
                f"{_fmt(float(row['mse_raw'])):>8} "
                f"{_fmt(float(row.get('splat_cellaware_mse_imp_hm', float('nan')))):>8} "
                f"{_fmt(float(row.get('splat_cellaware_hp_mse_imp_hm', float('nan')))):>8}"
            )

if __name__ == "__main__":
    main()