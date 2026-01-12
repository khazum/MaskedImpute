#!/usr/bin/env python3
"""
mask_impute15.py

Mask imputation using MSE on log-normalized counts with proxy calibration.
Builds count-derived proxy targets and pseudo-labels (KNN mean + gene quantiles)
for per-gene linear calibration and biozero supervision (no TrueCounts for calibration).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predict_dropouts_new import splatter_bio_posterior_from_counts

_REPO_ROOT = Path(__file__).resolve().parent
_SYS_PATH = list(sys.path)
try:
    sys.path = [
        p
        for p in sys.path
        if str(_REPO_ROOT) not in p and "MaskedImpute/rds2py" not in p
    ]
    from rds2py import read_rds
finally:
    sys.path = _SYS_PATH

EPSILON = 1e-6
GENE_NORM_LOW = 5.0
GENE_NORM_HIGH = 95.0
LAMBDA_MSE = 0.5
NORM_FACTOR = 10000.0

try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

CONFIG = {
    "thr_drop": 0.9,
    "disp_mode": "estimate",
    "disp_const": 0.05,
    "use_cell_factor": True,
    "tau_dispersion": 20.0,
    "tau_group_dispersion": 20.0,
    "tau_dropout": 50.0,
    "bio_model": "splat",
    "bio_mix_weight": 0.5,
    "zinb_phi_min": 1e-3,
    "zinb_pi_max": 0.99,
    "zinb_cluster_k": 4,
    "zinb_cluster_pca": 20,
    "zinb_cluster_iters": 2,
    "zinb_cluster_min_cells": 10,
    "zinb_cluster_seed": 42,
    "zinb_cluster_update_phi": True,
    "zinb_cluster_pi_tau": 50.0,
    "zinb_cluster_phi_tau": 50.0,
    "p_zero": 0.0,
    "p_nz": 0.2,
    "noise_min": 0.0,
    "noise_max": 0.2,
    "hidden": [128, 64],
    "bottleneck": 64,
    "dropout": 0.0,
    "use_residual": False,
    "epochs": 100,
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "loss_bio_weight": 2.0,
    "loss_nz_weight": 1.0,
    "bio_reg_weight": 1.0,
    "recon_weight": 0.1,
    "p_low": 2.0,
    "p_high": 99.5,
    "keep_positive": True,
    "renorm_imputed": False,
    "hard_zero_bio": True,
    "p_bio_temp": 1.55,
    "p_bio_bias": 0.45,
    "zero_iso_weight": 0.0,
    "zero_iso_bins": 12,
    "zero_iso_gamma": 1.0,
    "zero_iso_bio_weight": 20.0,
    "zero_iso_min_scale": 0.0,
    "zero_iso_max_scale": 2.0,
    "dropout_iso_weight": 0.0,
    "dropout_iso_bins": 12,
    "dropout_iso_gamma": 1.0,
    "dropout_iso_min_scale": 1.0,
    "dropout_iso_max_scale": 2.0,
    "dropout_iso_pmax": 0.15,
    "constrained_zero_scale": False,
    "constrained_zero_max_mse_inc": 0.1,
    "constrained_zero_lambda_max": 1000.0,
    "constrained_zero_iters": 30,
    "cell_zero_weight": 0.6,
    "cell_scale_alpha": 0.0,
    "cell_scale_max": 3.0,
    "sf_dropout_alpha": 0.0,
    "sf_dropout_max_scale": 3.0,
    "proxy_bio_weight": 1.0,
    "proxy_drop_weight": 0.5,
    "proxy_impute_alpha": 1.0,
    "proxy_impute_gamma": 2.0,
    "proxy_mean_mode": "knn",
    "proxy_calib_min_points": 20,
    "proxy_pu_weight": 0.5,
    "proxy_pu_lr": 0.1,
    "proxy_pu_epochs": 300,
    "proxy_pu_l2": 0.0,
    "bio_soft_gamma": 0.0,
    "proxy_knn_k": 15,
    "proxy_knn_pca": 20,
    "proxy_knn_min_nz": 3,
    "proxy_knn_q_low": 0.2,
    "proxy_knn_q_high": 0.8,
    "proxy_knn_label_min_points": 20,
    "proxy_knn_label": True,
    "proxy_gene_bio_max": 0.02,
    "proxy_gene_drop_min": 0.2,
    "proxy_calib_mode": "labeled",
    "proxy_knn_ignore_zeros": True,
    "knn_bio_mix_weight": 0.0,
    "knn_bio_mix_iters": 8,
    "knn_bio_mix_min_points": 20,
    "knn_bio_mix_q_low": 0.2,
    "knn_bio_mix_q_high": 0.8,
    "knn_bio_mix_sigma_floor": 0.1,
    "impute_mode": "gene",
    "impute_blend_alpha": 0.5,
}


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


class ImprovedAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        bottleneck: int,
        dropout: float,
        use_residual: bool = True,
    ):
        super().__init__()
        sizes_enc = [input_dim] + list(hidden) + [bottleneck]
        sizes_dec = [bottleneck] + list(reversed(hidden)) + [input_dim]
        self.use_residual = use_residual

        enc_layers = []
        for i in range(len(sizes_enc) - 1):
            enc_layers.append(self._block(sizes_enc[i], sizes_enc[i + 1], dropout))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for i in range(len(sizes_dec) - 2):
            dec_layers.append(self._block(sizes_dec[i], sizes_dec[i + 1], dropout))
        dec_layers.append(nn.Linear(sizes_dec[-2], sizes_dec[-1]))
        self.decoder = nn.Sequential(*dec_layers)

    @staticmethod
    def _block(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
        layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        if self.use_residual:
            recon = recon + x
        return recon


def mse_from_residual(residual: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        residual = residual * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(residual.numel(), device=residual.device, dtype=residual.dtype)
    return residual.pow(2).sum() / denom


def weighted_masked_mse(
    residual: torch.Tensor,
    mask_bio: torch.Tensor,
    mask_nz: torch.Tensor,
    weight_bio: float,
    weight_nz: float,
) -> torch.Tensor:
    weight_bio_t = float(weight_bio)
    weight_nz_t = float(weight_nz)
    term_bio = residual.pow(2) * mask_bio * weight_bio_t
    term_nz = residual.pow(2) * mask_nz * weight_nz_t
    denom = (mask_bio * weight_bio_t).sum() + (mask_nz * weight_nz_t).sum()
    denom = denom.clamp_min(1.0)
    return (term_bio.sum() + term_nz.sum()) / denom


def load_dataset(path: str) -> Dict[str, np.ndarray] | None:
    sce = read_rds(path)
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object (expected SingleCellExperiment): {type(sce)}")

    logcounts = sce.assay("logcounts").T.astype("float32")
    keep = np.sum(logcounts != 0, axis=0) >= 2
    logcounts = logcounts[:, keep]

    log_true = None
    for assay_name in ("logTrueCounts", "perfect_logcounts"):
        try:
            log_true = sce.assay(assay_name).T[:, keep].astype("float32")
            break
        except Exception:
            continue

    if log_true is None:
        return None

    counts = None
    try:
        counts = sce.assay("counts").T.astype("float32")
        counts = counts[:, keep]
    except Exception:
        counts = None

    scale_factor = None
    lib_true = None
    try:
        col_data = getattr(sce, "col_data", None)
        if col_data is not None and hasattr(col_data, "column_names"):
            names = list(col_data.column_names)
            if "scaleFactorTrueCounts" in names:
                sf_vals = np.array(col_data.column("scaleFactorTrueCounts"), dtype=float)
                sf_vals = sf_vals[np.isfinite(sf_vals) & (sf_vals > 0)]
                if sf_vals.size:
                    scale_factor = float(np.median(sf_vals))
            if "libSizeTrueCounts" in names:
                lib_vals = np.array(col_data.column("libSizeTrueCounts"), dtype=float)
                lib_true = lib_vals.astype(np.float32, copy=False)
    except Exception:
        pass

    return {
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts,
        "scale_factor": scale_factor,
        "lib_true": lib_true,
    }


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def logcounts_to_counts(logcts: np.ndarray, base: float = 2.0) -> np.ndarray:
    return np.expm1(logcts * np.log(base))


def _size_factors(counts: np.ndarray) -> np.ndarray:
    lib_sizes = counts.sum(axis=1).astype(np.float64)
    lib_med = float(np.median(lib_sizes)) if lib_sizes.size else 0.0
    if lib_med <= EPSILON:
        factors = np.ones_like(lib_sizes, dtype=np.float32)
    else:
        factors = (lib_sizes / lib_med).astype(np.float32, copy=False)
        factors[factors <= 0.0] = 1.0
    return factors


def _normalize_counts(counts: np.ndarray, size_factors: np.ndarray) -> np.ndarray:
    factors = np.maximum(size_factors, EPSILON).astype(np.float32, copy=False)
    return (counts / factors[:, None]).astype(np.float32, copy=False)


def _log_normalize_counts(counts: np.ndarray, norm_factor: float = NORM_FACTOR) -> np.ndarray:
    lib_sizes = counts.sum(axis=1).astype(np.float64)
    denom = np.where(lib_sizes > 0.0, lib_sizes, 1.0)
    norm = counts / denom[:, None] * float(norm_factor)
    log_norm = np.log1p(norm) / np.log(2.0)
    return np.nan_to_num(log_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def get_norm_factor(ds: Dict[str, object]) -> float:
    sf = ds.get("scale_factor")
    if sf is None:
        return float(NORM_FACTOR)
    try:
        sf_val = float(sf)
    except Exception:
        return float(NORM_FACTOR)
    if not np.isfinite(sf_val) or sf_val <= 0.0:
        return float(NORM_FACTOR)
    return sf_val


def renorm_logcounts(log_imputed: np.ndarray, norm_factor: float) -> np.ndarray:
    counts = np.clip(logcounts_to_counts(log_imputed), 0.0, None)
    return _log_normalize_counts(counts, norm_factor=float(norm_factor))


def estimate_dropout_aware_size_factors(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    p_bio: Optional[np.ndarray],
    alpha: float,
    max_scale: float,
    min_adj: float = 0.05,
) -> np.ndarray:
    lib = counts.sum(axis=1).astype(np.float64)
    if float(alpha) <= 0.0:
        return np.maximum(lib, 1.0)
    n_genes = max(counts.shape[1], 1)
    obs_zero_frac = zeros_obs.mean(axis=1).astype(np.float64)
    if p_bio is None:
        bio_zero_frac = np.zeros_like(obs_zero_frac)
    else:
        bio_zero_frac = (p_bio * zeros_obs).sum(axis=1).astype(np.float64) / float(n_genes)
    drop_frac = np.clip(obs_zero_frac - bio_zero_frac, 0.0, 0.95)
    adj = np.clip(1.0 - float(alpha) * drop_frac, float(min_adj), 1.0)
    sf = lib / adj
    max_scale_f = float(max_scale)
    if max_scale_f > 0.0:
        sf = np.minimum(sf, lib * max_scale_f)
    return np.maximum(sf, 1.0)


def _mse_from_diff(diff: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff ** 2))


def compute_mse_metrics(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    counts_obs: np.ndarray,
) -> Dict[str, float]:
    diff = true_log - pred_log
    mask_biozero = true_log <= EPSILON
    mask_dropout = (true_log > EPSILON) & (counts_obs <= EPSILON)
    mask_non_zero = (true_log > EPSILON) & (counts_obs > EPSILON)
    return {
        "mse": _mse_from_diff(diff),
        "mse_dropout": _mse_from_diff(diff, mask_dropout),
        "mse_biozero": _mse_from_diff(diff, mask_biozero),
        "mse_non_zero": _mse_from_diff(diff, mask_non_zero),
        "n_total": int(diff.size),
        "n_dropout": int(mask_dropout.sum()),
        "n_biozero": int(mask_biozero.sum()),
        "n_non_zero": int(mask_non_zero.sum()),
    }




def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit_scale_probs(p: np.ndarray, temp: float, bias: float) -> np.ndarray:
    if float(temp) == 1.0 and float(bias) == 0.0:
        return p.astype(np.float32, copy=False)
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(p_clip / (1.0 - p_clip))
    logit = logit * float(temp) + float(bias)
    return _sigmoid(logit).astype(np.float32, copy=False)


def calibrate_p_bio_with_proxy(
    p_bio: np.ndarray,
    zeros_obs: np.ndarray,
    proxy_bio_mask: Optional[np.ndarray],
    proxy_bio_label: Optional[np.ndarray],
    weight_bio: float,
    weight_drop: float,
) -> np.ndarray:
    if proxy_bio_mask is None or proxy_bio_label is None:
        return p_bio
    weight_bio = float(np.clip(weight_bio, 0.0, 1.0))
    weight_drop = float(np.clip(weight_drop, 0.0, 1.0))
    if weight_bio <= 0.0 and weight_drop <= 0.0:
        return p_bio
    p_out = p_bio.astype(np.float32, copy=True)
    boost_mask = proxy_bio_mask & (proxy_bio_label > 0.5) & zeros_obs
    drop_mask = proxy_bio_mask & (proxy_bio_label <= 0.5) & zeros_obs
    if weight_bio > 0.0 and np.any(boost_mask):
        p_sel = p_out[boost_mask]
        p_out[boost_mask] = p_sel + weight_bio * (1.0 - p_sel)
    if weight_drop > 0.0 and np.any(drop_mask):
        p_out[drop_mask] = p_out[drop_mask] * (1.0 - weight_drop)
    return p_out


def fit_proxy_bio_from_counts(
    p_bio: np.ndarray,
    ds: Dict[str, object],
    lr: float,
    epochs: int,
    l2: float,
) -> Optional[np.ndarray]:
    proxy_mask = ds.get("proxy_bio_mask")
    proxy_label = ds.get("proxy_bio_label")
    knn_log = ds.get("knn_log_mean")
    if proxy_mask is None or proxy_label is None:
        return None
    zeros_obs = ds["zeros_obs"]
    if proxy_mask.sum() < 50:
        return None

    p_clip = np.clip(p_bio, 1e-6, 1.0 - 1e-6)
    x1_all = np.log(p_clip / (1.0 - p_clip)).astype(np.float32, copy=False)
    if knn_log is None:
        knn_log = ds["gene_log_mean_nz"][None, :].astype(np.float32, copy=False)
    gene_log_mean = ds["gene_log_mean_nz"].astype(np.float32, copy=False)
    gene_nz_frac = ds["gene_nz_frac"].astype(np.float32, copy=False)
    cell_zero_norm = ds["cell_zero_norm"].astype(np.float32, copy=False)
    x2_all = knn_log.astype(np.float32, copy=False)
    x3_all = np.broadcast_to(gene_log_mean, p_bio.shape).astype(np.float32, copy=False)
    x4_all = np.broadcast_to(gene_nz_frac, p_bio.shape).astype(np.float32, copy=False)
    x5_all = np.broadcast_to(cell_zero_norm[:, None], p_bio.shape).astype(
        np.float32, copy=False
    )
    x6_all = (x2_all - x3_all).astype(np.float32, copy=False)

    idx_r, idx_c = np.where(proxy_mask)
    if idx_r.size < 50:
        return None
    max_train = 200000
    if idx_r.size > max_train:
        sel = np.random.choice(idx_r.size, size=max_train, replace=False)
        idx_r = idx_r[sel]
        idx_c = idx_c[sel]

    X = np.stack(
        [
            x1_all[idx_r, idx_c],
            x2_all[idx_r, idx_c],
            x3_all[idx_r, idx_c],
            x4_all[idx_r, idx_c],
            x5_all[idx_r, idx_c],
            x6_all[idx_r, idx_c],
        ],
        axis=1,
    )
    y = proxy_label[idx_r, idx_c].astype(np.float32, copy=False)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std
    if X.shape[0] < 50:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    w = torch.zeros((Xt.shape[1],), device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), device=device, dtype=torch.float32, requires_grad=True)
    opt = optim.Adam([w, b], lr=float(lr), weight_decay=float(l2))
    pos = float(y.sum())
    neg = float(y.size - pos)
    if pos > 0.0 and neg > 0.0:
        pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = Xt @ w + b
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        x1_all = (x1_all - mean[0, 0]) / std[0, 0]
        x2_all = (x2_all - mean[0, 1]) / std[0, 1]
        x3_all = (x3_all - mean[0, 2]) / std[0, 2]
        x4_all = (x4_all - mean[0, 3]) / std[0, 3]
        x5_all = (x5_all - mean[0, 4]) / std[0, 4]
        x6_all = (x6_all - mean[0, 5]) / std[0, 5]
        X_all = np.stack(
            [x1_all, x2_all, x3_all, x4_all, x5_all, x6_all], axis=-1
        ).astype(np.float32, copy=False)
        Xt_all = torch.from_numpy(X_all.reshape(-1, X_all.shape[-1])).to(device)
        logits_all = Xt_all @ w + b
        probs = torch.sigmoid(logits_all).cpu().numpy().reshape(p_bio.shape)
    p_full = np.zeros_like(p_bio, dtype=np.float32)
    p_full[zeros_obs] = probs[zeros_obs]
    return np.clip(p_full, 0.0, 1.0).astype(np.float32, copy=False)


def _fit_logistic_regression(x: np.ndarray, y: np.ndarray, x0_approx: float) -> Tuple[float, float]:
    def _sigmoid_curve(x_vals, k, x0):
        return 1.0 / (1.0 + np.exp(-k * (x_vals - x0)))

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if x_clean.size < 5:
        return -1.0, x0_approx

    if curve_fit is not None:
        try:
            popt, _ = curve_fit(
                _sigmoid_curve,
                x_clean,
                y_clean,
                p0=[-1.0, x0_approx],
                bounds=([-np.inf, -np.inf], [0.0, np.inf]),
                method="trf",
                maxfev=2000,
            )
            k_fit, x0_fit = popt
            if k_fit > -1e-4:
                k_fit = -1.0
            return float(k_fit), float(x0_fit)
        except Exception:
            pass

    y_clipped = np.clip(y_clean, 1e-6, 1.0 - 1e-6)
    logit_y = np.log(y_clipped / (1.0 - y_clipped))
    try:
        k, b = np.polyfit(x_clean, logit_y, 1)
        if k > -1e-3:
            k, b = -1.0, x0_approx
    except Exception:
        k, b = -1.0, x0_approx
    x0 = float(-b / k) if abs(k) > 1e-12 else x0_approx
    return float(k), float(x0)


def estimate_dropout_curve(norm_counts: np.ndarray) -> Tuple[float, float, np.ndarray]:
    means = norm_counts.mean(axis=0).astype(np.float64)
    zeros_frac = (norm_counts <= 0.0).mean(axis=0).astype(np.float64)
    x = np.log(np.maximum(means, EPSILON))
    y = np.clip(zeros_frac, 1e-6, 1.0 - 1e-6)

    mid_mask = (y > 0.2) & (y < 0.8) & np.isfinite(x)
    if np.any(mid_mask):
        x0_approx = float(np.median(x[mid_mask]))
    elif np.isfinite(x).any():
        x0_approx = float(np.median(x[np.isfinite(x)]))
    else:
        x0_approx = 0.0

    k, x0 = _fit_logistic_regression(x, y, x0_approx)
    p_drop = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    p_drop = np.clip(p_drop, 0.0, 1.0).astype(np.float32, copy=False)
    return float(k), float(x0), p_drop


def _parse_float_list(raw: Optional[str], default: List[float]) -> List[float]:
    if raw is None or raw.strip() == "":
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip() != ""]


def _parse_bool_list(raw: Optional[str], default: List[bool]) -> List[bool]:
    if raw is None or raw.strip() == "":
        return list(default)
    out: List[bool] = []
    for item in raw.split(","):
        val = item.strip().lower()
        if val in ("true", "1", "yes", "y"):
            out.append(True)
        elif val in ("false", "0", "no", "n"):
            out.append(False)
    return out if out else list(default)


def _parse_str_list(raw: Optional[str], default: List[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        return list(default)
    return [val.strip() for val in raw.split(",") if val.strip() != ""]


def compute_knn_log_mean(
    logcounts: np.ndarray, k: int, pca_dim: int, ignore_zeros: bool
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    n_cells, n_genes = logcounts.shape
    k = int(k)
    if k <= 0 or n_cells <= 1:
        return None, None
    k = min(k, n_cells - 1)

    X = logcounts.astype(np.float32, copy=False)
    Xc = X - X.mean(axis=0, keepdims=True)
    n_comp = min(int(pca_dim), n_cells, n_genes)
    if n_comp > 0 and n_comp < min(n_cells, n_genes):
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        Z = U[:, :n_comp] * S[:n_comp]
    else:
        Z = Xc

    dists = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(dists, np.inf)
    nn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
    knn_mean = np.empty_like(X, dtype=np.float32)
    knn_nz_count = np.zeros_like(X, dtype=np.int32) if ignore_zeros else None
    for i in range(n_cells):
        neigh = X[nn_idx[i]]
        if ignore_zeros:
            nz = neigh > 0.0
            denom = np.maximum(nz.sum(axis=0), 1)
            knn_mean[i] = (neigh * nz).sum(axis=0) / denom
            knn_nz_count[i] = nz.sum(axis=0).astype(np.int32, copy=False)
        else:
            knn_mean[i] = neigh.mean(axis=0)
    return knn_mean.astype(np.float32, copy=False), knn_nz_count


def _pca_embed(X: np.ndarray, n_components: int) -> np.ndarray:
    n_cells, n_genes = X.shape
    n_comp = min(int(n_components), n_cells, n_genes)
    Xc = X - X.mean(axis=0, keepdims=True)
    if n_comp <= 0 or n_comp >= min(n_cells, n_genes):
        return Xc.astype(np.float32, copy=False)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :n_comp] * S[:n_comp]
    return Z.astype(np.float32, copy=False)


def _kmeans(X: np.ndarray, k: int, iters: int, seed: int) -> Tuple[np.ndarray, int]:
    n = X.shape[0]
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(int(seed))
    centroids = X[rng.choice(n, size=k, replace=False)].astype(np.float32, copy=True)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(int(iters)):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centroids[j] = X[rng.integers(0, n)]
            else:
                centroids[j] = X[mask].mean(axis=0)
    return labels, k


def estimate_bio_prob_from_knn(
    knn_log: Optional[np.ndarray],
    zeros_obs: np.ndarray,
    q_low: float,
    q_high: float,
    min_points: int,
    iters: int,
    sigma_floor: float,
) -> Optional[np.ndarray]:
    if knn_log is None:
        return None
    if not (0.0 < float(q_low) < float(q_high) < 1.0):
        return None
    n_cells, n_genes = knn_log.shape
    p_bio = np.zeros((n_cells, n_genes), dtype=np.float32)
    sqrt2pi = float(np.sqrt(2.0 * np.pi))
    sigma_floor_f = float(max(sigma_floor, 1e-3))
    for j in range(n_genes):
        zmask = zeros_obs[:, j]
        vals = knn_log[zmask, j]
        if vals.size < int(min_points):
            continue
        mu1 = float(np.quantile(vals, q_low))
        mu2 = float(np.quantile(vals, q_high))
        if not np.isfinite(mu1) or not np.isfinite(mu2):
            continue
        if mu1 > mu2:
            mu1, mu2 = mu2, mu1
        sigma = float(np.std(vals))
        sigma1 = max(sigma, sigma_floor_f)
        sigma2 = max(sigma, sigma_floor_f)
        pi = 0.5
        r1 = None
        for _ in range(int(iters)):
            z1 = (vals - mu1) / sigma1
            z2 = (vals - mu2) / sigma2
            p1 = pi * np.exp(-0.5 * z1 * z1) / (sigma1 * sqrt2pi)
            p2 = (1.0 - pi) * np.exp(-0.5 * z2 * z2) / (sigma2 * sqrt2pi)
            denom = p1 + p2 + EPSILON
            r1 = p1 / denom
            pi = float(np.clip(r1.mean(), 0.05, 0.95))
            sum_r1 = float(r1.sum()) + EPSILON
            sum_r2 = float((1.0 - r1).sum()) + EPSILON
            mu1 = float(np.sum(r1 * vals) / sum_r1)
            mu2 = float(np.sum((1.0 - r1) * vals) / sum_r2)
            sigma1 = float(np.sqrt(np.sum(r1 * (vals - mu1) ** 2) / sum_r1))
            sigma2 = float(np.sqrt(np.sum((1.0 - r1) * (vals - mu2) ** 2) / sum_r2))
            sigma1 = max(sigma1, sigma_floor_f)
            sigma2 = max(sigma2, sigma_floor_f)
        if r1 is None:
            continue
        if mu1 > mu2:
            r1 = 1.0 - r1
        p_bio[zmask, j] = r1.astype(np.float32, copy=False)
    return p_bio


def _nb_zero_prob(mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    mu = np.maximum(mu, 1e-8)
    phi = np.maximum(phi, 1e-8)
    return np.exp(-(1.0 / phi) * np.log1p(phi * mu))


def estimate_bio_prob_zinb(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    phi_min: float,
    pi_max: float,
) -> np.ndarray:
    size_factors = _size_factors(counts)
    norm_counts = counts / np.maximum(size_factors[:, None], EPSILON)
    mu_ref = norm_counts.mean(axis=0)
    var_ref = norm_counts.var(axis=0)
    phi = (var_ref - mu_ref) / (mu_ref ** 2 + EPSILON)
    phi = np.clip(phi, float(phi_min), None)
    mu_ij = mu_ref[None, :] * size_factors[:, None]
    p0_nb = _nb_zero_prob(mu_ij, phi[None, :])
    z_obs = zeros_obs.mean(axis=0)
    p0_nb_gene = p0_nb.mean(axis=0)
    pi = (z_obs - p0_nb_gene) / np.maximum(1.0 - p0_nb_gene, EPSILON)
    pi = np.clip(pi, 0.0, float(pi_max))
    p_bio = np.zeros_like(counts, dtype=np.float32)
    num = (1.0 - pi[None, :]) * p0_nb
    denom = pi[None, :] + num
    if np.any(zeros_obs):
        p_bio[zeros_obs] = num[zeros_obs] / np.maximum(denom[zeros_obs], EPSILON)
    return np.clip(p_bio, 0.0, 1.0).astype(np.float32, copy=False)


def estimate_bio_prob_zinb_cluster(
    counts: np.ndarray,
    logcounts: np.ndarray,
    zeros_obs: np.ndarray,
    phi_min: float,
    pi_max: float,
    k: int,
    pca_dim: int,
    em_iters: int,
    min_cells: int,
    seed: int,
    update_phi: bool,
    pi_tau: float,
    phi_tau: float,
) -> np.ndarray:
    n_cells, n_genes = counts.shape
    size_factors = _size_factors(counts)
    norm_counts = counts / np.maximum(size_factors[:, None], EPSILON)
    mu_ref = norm_counts.mean(axis=0)
    var_ref = norm_counts.var(axis=0)
    phi_global = (var_ref - mu_ref) / (mu_ref ** 2 + EPSILON)
    phi_global = np.clip(phi_global, float(phi_min), None)
    mu_ij_global = mu_ref[None, :] * size_factors[:, None]
    p0_nb_global = _nb_zero_prob(mu_ij_global, phi_global[None, :])
    z_obs_global = zeros_obs.mean(axis=0)
    p0_nb_gene_global = p0_nb_global.mean(axis=0)
    pi_global = (z_obs_global - p0_nb_gene_global) / np.maximum(
        1.0 - p0_nb_gene_global, EPSILON
    )
    pi_global = np.clip(pi_global, 0.0, float(pi_max))

    Z = _pca_embed(logcounts, pca_dim)
    labels, k_eff = _kmeans(Z, k, iters=25, seed=seed)
    p_bio_global = estimate_bio_prob_zinb(
        counts=counts,
        zeros_obs=zeros_obs,
        phi_min=phi_min,
        pi_max=pi_max,
    )

    p_bio = np.zeros_like(counts, dtype=np.float32)
    for _ in range(max(1, int(em_iters))):
        for cl in range(k_eff):
            mask = labels == cl
            if mask.sum() < int(min_cells):
                p_bio[mask] = p_bio_global[mask]
                continue
            weights = np.ones_like(norm_counts[mask], dtype=np.float32)
            if np.any(zeros_obs[mask]):
                weights[zeros_obs[mask]] = 1.0 - p_bio[mask][zeros_obs[mask]]
            denom_w = np.maximum(weights.sum(axis=0), EPSILON)
            mu_ref_cl = (norm_counts[mask] * weights).sum(axis=0) / denom_w
            mu_ij = mu_ref_cl[None, :] * size_factors[mask][:, None]
            if bool(update_phi):
                diff = norm_counts[mask] - mu_ref_cl[None, :]
                var_ref_cl = (weights * diff * diff).sum(axis=0) / denom_w
                phi_cl = (var_ref_cl - mu_ref_cl) / (mu_ref_cl ** 2 + EPSILON)
                phi_cl = np.clip(phi_cl, float(phi_min), None)
                w_phi = 1.0 / (1.0 + float(mask.sum()) / max(float(phi_tau), EPSILON))
                phi_cl = w_phi * phi_global + (1.0 - w_phi) * phi_cl
            else:
                phi_cl = phi_global
            p0_nb = _nb_zero_prob(mu_ij, phi_cl[None, :])
            z_obs = zeros_obs[mask].mean(axis=0)
            p0_nb_gene = p0_nb.mean(axis=0)
            pi = (z_obs - p0_nb_gene) / np.maximum(1.0 - p0_nb_gene, EPSILON)
            pi = np.clip(pi, 0.0, float(pi_max))
            w_pi = 1.0 / (1.0 + float(mask.sum()) / max(float(pi_tau), EPSILON))
            pi = w_pi * pi_global + (1.0 - w_pi) * pi
            denom = pi[None, :] + (1.0 - pi[None, :]) * p0_nb
            p_bio_cl = (pi[None, :] / np.maximum(denom, EPSILON)).astype(
                np.float32, copy=False
            )
            p_bio_cl[~zeros_obs[mask]] = 0.0
            p_bio[mask] = p_bio_cl

    p_bio[~zeros_obs] = 0.0
    return np.clip(p_bio, 0.0, 1.0).astype(np.float32, copy=False)


def _get_gene_log_threshold(ds: Dict[str, object], quantile: float) -> np.ndarray:
    cache: Dict[float, np.ndarray] = ds.setdefault("log_thresh_cache", {})
    key = float(quantile)
    if key not in cache:
        logcounts = ds["logcounts"]
        thresh = np.zeros((logcounts.shape[1],), dtype=np.float32)
        for j in range(logcounts.shape[1]):
            vals = logcounts[:, j]
            nz = vals[vals > 0.0]
            if nz.size == 0:
                thresh[j] = 0.0
            else:
                thresh[j] = float(np.percentile(nz, key))
        cache[key] = thresh.astype(np.float32, copy=False)
    return cache[key]


def _get_gene_log_threshold_from_mat(logcounts: np.ndarray, quantile: float) -> np.ndarray:
    thresh = np.zeros((logcounts.shape[1],), dtype=np.float32)
    for j in range(logcounts.shape[1]):
        vals = logcounts[:, j]
        nz = vals[vals > 0.0]
        if nz.size == 0:
            thresh[j] = 0.0
        else:
            thresh[j] = float(np.percentile(nz, quantile))
    return thresh


def calibrate_log_to_proxy(
    pred_log: np.ndarray,
    proxy_log: np.ndarray,
    shrink: float,
    mask: Optional[np.ndarray] = None,
    min_points: int = 20,
) -> np.ndarray:
    shrink = float(np.clip(shrink, 0.0, 1.0))
    if mask is None:
        mask = np.ones_like(pred_log, dtype=bool)
    min_points = int(min_points)
    n_genes = pred_log.shape[1]
    a = np.ones((n_genes,), dtype=np.float32)
    b = np.zeros((n_genes,), dtype=np.float32)
    for j in range(n_genes):
        m = mask[:, j]
        if m.sum() < min_points:
            continue
        x = pred_log[m, j]
        y = proxy_log[m, j]
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        dx = x - mean_x
        dy = y - mean_y
        var_x = float(np.mean(dx * dx))
        if var_x < EPSILON:
            continue
        cov_xy = float(np.mean(dx * dy))
        a[j] = float(cov_xy / (var_x + EPSILON))
        b[j] = float(mean_y - a[j] * mean_x)
    if shrink > 0.0:
        a = (1.0 - shrink) * a + shrink * 1.0
        b = (1.0 - shrink) * b
    pred_cal = pred_log * a[None, :] + b[None, :]
    pred_cal = np.clip(pred_cal, 0.0, None)
    return pred_cal.astype(np.float32, copy=False)


def prepare_dataset(
    path: Path,
    knn_k: int,
    knn_pca: int,
    proxy_gene_bio_max: float,
    proxy_gene_drop_min: float,
    proxy_knn_q_low: float,
    proxy_knn_q_high: float,
    proxy_knn_label_min_points: int,
    proxy_knn_label: bool,
    proxy_knn_ignore_zeros: bool,
    proxy_knn_min_nz: int,
) -> Dict[str, object] | None:
    dataset = load_dataset(str(path))
    if dataset is None:
        return None
    logcounts = dataset["logcounts"]
    log_true = dataset["log_true"]
    counts = dataset["counts"]

    if counts is None:
        counts_obs = np.clip(logcounts_to_counts(logcounts), 0.0, None)
    else:
        counts_obs = np.clip(counts, 0.0, None)
    zeros_obs = counts_obs <= 0.0
    counts_max = counts_obs.max(axis=0)
    lib_size = counts_obs.sum(axis=1).astype(np.float32)

    norm_counts_raw = np.clip(logcounts_to_counts(logcounts), 0.0, None)
    gene_scale = np.ones((norm_counts_raw.shape[1],), dtype=np.float32)
    norm_counts = norm_counts_raw.astype(np.float32, copy=False)

    nz_mask = norm_counts > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_mean = norm_counts.mean(axis=0)
        gene_mean_nz = np.sum(norm_counts * nz_mask, axis=0) / np.maximum(nz_mask.sum(axis=0), 1)
    gene_mean = np.nan_to_num(gene_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    gene_mean_nz = np.nan_to_num(gene_mean_nz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    drop_k, drop_x0, p_drop_gene = estimate_dropout_curve(norm_counts_raw)

    log_nz_mask = logcounts > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_log_mean = logcounts.mean(axis=0)
        gene_log_mean_nz = np.sum(logcounts * log_nz_mask, axis=0) / np.maximum(
            log_nz_mask.sum(axis=0), 1
        )
    gene_log_mean = np.nan_to_num(gene_log_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    gene_log_mean_nz = np.nan_to_num(
        gene_log_mean_nz, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    gene_nz_frac = log_nz_mask.mean(axis=0).astype(np.float32)

    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, GENE_NORM_LOW))
    cz_hi = float(np.percentile(cell_zero_frac, GENE_NORM_HIGH))
    cz_span = max(cz_hi - cz_lo, EPSILON)
    cell_zero_norm = np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)

    knn_log_mean, knn_nz_count = compute_knn_log_mean(
        logcounts,
        k=int(knn_k),
        pca_dim=int(knn_pca),
        ignore_zeros=bool(proxy_knn_ignore_zeros),
    )
    knn_valid_mask = None
    if knn_nz_count is not None:
        knn_valid_mask = knn_nz_count >= int(proxy_knn_min_nz)
    proxy_bio_label = np.zeros_like(logcounts, dtype=np.float32)
    proxy_bio_mask = np.zeros_like(logcounts, dtype=bool)
    bio_genes = gene_nz_frac <= float(proxy_gene_bio_max)
    drop_genes = gene_nz_frac >= float(proxy_gene_drop_min)
    if np.any(bio_genes):
        bio_mask = zeros_obs & bio_genes[None, :]
        proxy_bio_label[bio_mask] = 1.0
        proxy_bio_mask[bio_mask] = True
    if np.any(drop_genes):
        drop_mask = zeros_obs & drop_genes[None, :]
        proxy_bio_label[drop_mask] = 0.0
        proxy_bio_mask[drop_mask] = True
    if (
        knn_log_mean is not None
        and bool(proxy_knn_label)
        and 0.0 < float(proxy_knn_q_low) < float(proxy_knn_q_high) < 1.0
    ):
        q_low = float(proxy_knn_q_low)
        q_high = float(proxy_knn_q_high)
        min_points = int(proxy_knn_label_min_points)
        for j in range(logcounts.shape[1]):
            zmask = zeros_obs[:, j]
            if knn_valid_mask is not None:
                zmask = zmask & knn_valid_mask[:, j]
            if np.sum(zmask) < min_points:
                continue
            vals = knn_log_mean[zmask, j]
            if vals.size < min_points:
                continue
            ql = float(np.quantile(vals, q_low))
            qh = float(np.quantile(vals, q_high))
            unlabeled = zmask & (~proxy_bio_mask[:, j])
            if not np.any(unlabeled):
                continue
            v = knn_log_mean[:, j]
            bio_mask = unlabeled & (v <= ql)
            drop_mask = unlabeled & (v >= qh)
            if np.any(bio_mask):
                proxy_bio_label[bio_mask, j] = 1.0
                proxy_bio_mask[bio_mask, j] = True
            if np.any(drop_mask):
                proxy_bio_label[drop_mask, j] = 0.0
                proxy_bio_mask[drop_mask, j] = True

    log_imputed_gene = logcounts.copy()
    if zeros_obs.any():
        rr, cc = np.where(zeros_obs)
        log_imputed_gene[rr, cc] = gene_log_mean_nz[cc]
    log_imputed_knn = None
    if knn_log_mean is not None:
        proxy_mean = knn_log_mean
        if knn_valid_mask is not None:
            proxy_mean = np.where(knn_valid_mask, proxy_mean, gene_log_mean_nz[None, :])
        log_imputed_knn = logcounts.copy()
        log_imputed_knn[zeros_obs] = proxy_mean[zeros_obs]

    return {
        "dataset": path.stem,
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts_obs,
        "scale_factor": dataset.get("scale_factor"),
        "lib_true": dataset.get("lib_true"),
        "zeros_obs": zeros_obs,
        "counts_max": counts_max,
        "lib_size": lib_size,
        "norm_counts": norm_counts,
        "norm_counts_raw": norm_counts_raw,
        "gene_scale": gene_scale,
        "gene_mean": gene_mean,
        "gene_mean_nz": gene_mean_nz,
        "gene_log_mean": gene_log_mean,
        "gene_log_mean_nz": gene_log_mean_nz,
        "gene_nz_frac": gene_nz_frac,
        "drop_k": drop_k,
        "drop_x0": drop_x0,
        "p_drop_gene": p_drop_gene,
        "cell_zero_norm": cell_zero_norm,
        "knn_log_mean": knn_log_mean,
        "knn_nz_count": knn_nz_count,
        "knn_valid_mask": knn_valid_mask,
        "proxy_bio_mask": proxy_bio_mask,
        "proxy_bio_label": proxy_bio_label,
        "log_imputed_gene": log_imputed_gene,
        "log_imputed_knn": log_imputed_knn,
    }


def splat_cellaware_bio_prob(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    disp_mode: str,
    disp_const: float,
    use_cell_factor: bool,
    tau_dispersion: float,
    tau_group_dispersion: float,
    tau_dropout: float,
) -> np.ndarray:
    bio_post = splatter_bio_posterior_from_counts(
        counts,
        disp_mode=disp_mode,
        disp_const=float(disp_const),
        use_cell_factor=bool(use_cell_factor),
        groups=None,
        tau_dispersion=float(tau_dispersion),
        tau_group_dispersion=float(tau_group_dispersion),
        tau_dropout=float(tau_dropout),
    )
    p_bio = np.asarray(bio_post, dtype=np.float64)
    p_bio = np.nan_to_num(p_bio, nan=0.0, posinf=0.0, neginf=0.0)
    p_bio = np.clip(p_bio, 0.0, 1.0)
    p_bio[~zeros_obs] = 0.0
    return p_bio.astype(np.float32)


def compute_p_bio(ds: Dict[str, object], cfg: Dict[str, object]) -> np.ndarray:
    zeros_obs = ds["zeros_obs"]
    bio_model = str(cfg.get("bio_model", "splat")).lower()
    p_bio_splat = None
    if bio_model in ("splat", "mix", "mix_cluster"):
        p_bio_splat = splat_cellaware_bio_prob(
            counts=ds["counts"],
            zeros_obs=zeros_obs,
            disp_mode=str(cfg["disp_mode"]),
            disp_const=float(cfg["disp_const"]),
            use_cell_factor=bool(cfg["use_cell_factor"]),
            tau_dispersion=float(cfg["tau_dispersion"]),
            tau_group_dispersion=float(cfg["tau_group_dispersion"]),
            tau_dropout=float(cfg["tau_dropout"]),
        )
    p_bio_zinb = None
    if bio_model in ("zinb", "mix"):
        p_bio_zinb = estimate_bio_prob_zinb(
            counts=ds["counts"],
            zeros_obs=zeros_obs,
            phi_min=float(cfg.get("zinb_phi_min", 1e-3)),
            pi_max=float(cfg.get("zinb_pi_max", 0.99)),
        )
    p_bio_zinb_cluster = None
    if bio_model in ("zinb_cluster", "mix_cluster"):
        p_bio_zinb_cluster = estimate_bio_prob_zinb_cluster(
            counts=ds["counts"],
            logcounts=ds["logcounts"],
            zeros_obs=zeros_obs,
            phi_min=float(cfg.get("zinb_phi_min", 1e-3)),
            pi_max=float(cfg.get("zinb_pi_max", 0.99)),
            k=int(cfg.get("zinb_cluster_k", 4)),
            pca_dim=int(cfg.get("zinb_cluster_pca", 20)),
            em_iters=int(cfg.get("zinb_cluster_iters", 2)),
            min_cells=int(cfg.get("zinb_cluster_min_cells", 10)),
            seed=int(cfg.get("zinb_cluster_seed", 42)),
            update_phi=bool(cfg.get("zinb_cluster_update_phi", True)),
            pi_tau=float(cfg.get("zinb_cluster_pi_tau", 50.0)),
            phi_tau=float(cfg.get("zinb_cluster_phi_tau", 50.0)),
        )

    if bio_model == "zinb" and p_bio_zinb is not None:
        p_bio = p_bio_zinb
    elif bio_model == "zinb_cluster" and p_bio_zinb_cluster is not None:
        p_bio = p_bio_zinb_cluster
    elif bio_model == "mix" and p_bio_splat is not None and p_bio_zinb is not None:
        w = float(np.clip(cfg.get("bio_mix_weight", 0.5), 0.0, 1.0))
        p_bio = (1.0 - w) * p_bio_splat + w * p_bio_zinb
    elif (
        bio_model == "mix_cluster"
        and p_bio_splat is not None
        and p_bio_zinb_cluster is not None
    ):
        w = float(np.clip(cfg.get("bio_mix_weight", 0.5), 0.0, 1.0))
        p_bio = (1.0 - w) * p_bio_splat + w * p_bio_zinb_cluster
    elif p_bio_splat is not None:
        p_bio = p_bio_splat
    else:
        p_bio = np.zeros_like(zeros_obs, dtype=np.float32)

    if float(cfg["cell_zero_weight"]) > 0.0:
        cell_w = np.clip(float(cfg["cell_zero_weight"]) * ds["cell_zero_norm"], 0.0, 1.0)
        p_bio = p_bio * (1.0 - cell_w[:, None])
    mix_weight = float(cfg.get("knn_bio_mix_weight", 0.0))
    if mix_weight > 0.0:
        p_knn = estimate_bio_prob_from_knn(
            ds.get("knn_log_mean"),
            zeros_obs=zeros_obs,
            q_low=float(cfg.get("knn_bio_mix_q_low", 0.2)),
            q_high=float(cfg.get("knn_bio_mix_q_high", 0.8)),
            min_points=int(cfg.get("knn_bio_mix_min_points", 20)),
            iters=int(cfg.get("knn_bio_mix_iters", 8)),
            sigma_floor=float(cfg.get("knn_bio_mix_sigma_floor", 0.1)),
        )
        if p_knn is not None:
            w = float(np.clip(mix_weight, 0.0, 1.0))
            p_bio = (1.0 - w) * p_bio + w * p_knn
            ds["p_bio_knn"] = p_knn
    p_bio = np.clip(p_bio, 0.0, 1.0).astype(np.float32, copy=False)
    p_bio[~zeros_obs] = 0.0
    return p_bio


def train_autoencoder_reconstruct(
    logcounts: np.ndarray,
    counts_max: np.ndarray,
    p_bio: np.ndarray,
    device: torch.device,
    hidden: Sequence[int],
    bottleneck: int,
    p_zero: float,
    p_nz: float,
    noise_min_frac: float,
    noise_max_frac: float,
    dropout: float,
    use_residual: bool,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    loss_bio_weight: float,
    loss_nz_weight: float,
    bio_reg_weight: float,
    recon_weight: float,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    scaler = RobustZThenMinMaxToNeg1Pos1(p_low=float(p_low), p_high=float(p_high)).fit(logcounts)
    Xs = scaler.transform(logcounts).astype(np.float32)

    bio_prob = p_bio.astype(np.float32)
    nonzero_mask = logcounts > 0.0

    Xtr = torch.tensor(Xs, dtype=torch.float32)
    bio_mask = torch.tensor(bio_prob, dtype=torch.float32)
    nz_mask = torch.tensor(nonzero_mask.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, bio_mask, nz_mask), batch_size=batch_size, shuffle=True, drop_last=False)

    lo = torch.tensor(scaler.lo_, dtype=torch.float32, device=device)
    hi = torch.tensor(scaler.hi_, dtype=torch.float32, device=device)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
    zmin = torch.tensor(scaler.zmin_, dtype=torch.float32, device=device)
    zspan = torch.tensor(scaler.zspan_, dtype=torch.float32, device=device)
    zero_scaled = ((0.0 - scaler.mean_) / scaler.std_ - scaler.zmin_) / scaler.zspan_
    zero_scaled = zero_scaled * 2.0 - 1.0
    zero_scaled_t = torch.tensor(zero_scaled, dtype=torch.float32, device=device)
    log2_base = float(np.log(2.0))
    counts_max_t = torch.tensor(np.maximum(counts_max, 1.0), dtype=torch.float32, device=device)

    model = ImprovedAE(
        input_dim=logcounts.shape[1],
        hidden=hidden,
        bottleneck=bottleneck,
        dropout=dropout,
        use_residual=use_residual,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    model.train()
    for _ in range(int(epochs)):
        for xb, bio_b, nz_b in loader:
            xb = xb.to(device)
            bio_b = bio_b.to(device)
            nz_b = nz_b.to(device)

            mask_bio = torch.bernoulli(bio_b * float(p_zero))
            mask_nz = torch.bernoulli(nz_b * float(p_nz))

            x_in = xb.clone()
            if mask_nz.any():
                x_in = torch.where(mask_nz.bool(), torch.zeros_like(x_in), x_in)
            if mask_bio.any():
                noise_lo = float(noise_min_frac)
                noise_hi = float(noise_max_frac)
                noise_scale = torch.rand_like(xb) * (noise_hi - noise_lo) + noise_lo
                noise_counts = noise_scale * counts_max_t
                log_noise = torch.log1p(noise_counts) / log2_base
                log_noise = torch.minimum(torch.maximum(log_noise, lo), hi)
                z = (log_noise - mean) / std
                x01 = (z - zmin) / zspan
                noise_scaled = x01 * 2.0 - 1.0
                x_in = torch.where(mask_bio.bool(), noise_scaled, x_in)

            opt.zero_grad()
            recon = model(x_in)
            residual = recon - xb
            masked_loss = weighted_masked_mse(
                residual,
                mask_bio=mask_bio,
                mask_nz=mask_nz,
                weight_bio=loss_bio_weight,
                weight_nz=loss_nz_weight,
            )
            full_loss = mse_from_residual(residual)
            bio_reg = ((recon - zero_scaled_t) ** 2 * bio_b).sum() / bio_b.sum().clamp_min(1.0)
            rw = float(recon_weight)
            loss = (1.0 - rw) * masked_loss + rw * full_loss + float(bio_reg_weight) * bio_reg
            loss.backward()
            opt.step()

    model.eval()
    recon_list = []
    with torch.no_grad():
        for i in range(0, Xtr.size(0), batch_size):
            xb = Xtr[i : i + batch_size].to(device)
            recon = model(xb)
            recon_np = recon.cpu().numpy()
            recon_orig = scaler.inverse_transform(recon_np)
            recon_list.append(recon_orig)
    recon_all = np.vstack(recon_list)
    return recon_all.astype(np.float32)


def get_count_ae_recon(
    ds: Dict[str, object],
    sf_dropout_alpha: float,
    device: torch.device,
    norm_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cache: Dict[float, Dict[str, object]] = ds.setdefault("count_ae_cache", {})
    key = float(sf_dropout_alpha)
    if key in cache:
        cached = cache[key]
        return cached["recon"], cached["sf"]

    counts_obs = ds["counts"]
    zeros_obs = ds["zeros_obs"]
    p_bio_base = ds.get("p_bio_base")
    sf = estimate_dropout_aware_size_factors(
        counts=counts_obs,
        zeros_obs=zeros_obs,
        p_bio=p_bio_base,
        alpha=float(sf_dropout_alpha),
        max_scale=float(CONFIG["sf_dropout_max_scale"]),
    )
    counts_norm = counts_obs / sf[:, None] * float(norm_factor)
    counts_norm = np.nan_to_num(
        counts_norm, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
    logcounts_norm = np.log1p(counts_norm) / np.log(2.0)
    counts_max_norm = counts_norm.max(axis=0)
    recon = train_autoencoder_reconstruct(
        logcounts=logcounts_norm,
        counts_max=counts_max_norm,
        p_bio=p_bio_base,
        device=device,
        hidden=CONFIG["hidden"],
        bottleneck=int(CONFIG["bottleneck"]),
        p_zero=float(CONFIG["p_zero"]),
        p_nz=float(CONFIG["p_nz"]),
        noise_min_frac=float(CONFIG["noise_min"]),
        noise_max_frac=float(CONFIG["noise_max"]),
        dropout=float(CONFIG["dropout"]),
        use_residual=bool(CONFIG["use_residual"]),
        epochs=int(CONFIG["epochs"]),
        batch_size=int(CONFIG["batch_size"]),
        lr=float(CONFIG["lr"]),
        weight_decay=float(CONFIG["weight_decay"]),
        loss_bio_weight=float(CONFIG["loss_bio_weight"]),
        loss_nz_weight=float(CONFIG["loss_nz_weight"]),
        bio_reg_weight=float(CONFIG["bio_reg_weight"]),
        recon_weight=float(CONFIG["recon_weight"]),
        p_low=float(CONFIG["p_low"]),
        p_high=float(CONFIG["p_high"]),
    )
    cache[key] = {"recon": recon, "sf": sf}
    return recon, sf


def _postprocess_imputation(
    norm_imputed_raw: np.ndarray,
    p_bio_post: np.ndarray,
    ds: Dict[str, object],
    thr_drop: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeros_obs = ds["zeros_obs"]
    p_bio_use = np.array(p_bio_post, copy=True)
    thr_bio = 1.0 - float(thr_drop)
    pred_bio_mask = p_bio_use >= float(thr_bio)

    norm_imputed_final = norm_imputed_raw.copy()
    if bool(CONFIG["keep_positive"]):
        norm_imputed_final[~zeros_obs] = ds["norm_counts"][~zeros_obs]

    if bool(CONFIG["hard_zero_bio"]):
        norm_imputed_final[pred_bio_mask] = 0.0

    return norm_imputed_final, p_bio_use, pred_bio_mask


def _build_proxy_target(
    ds: Dict[str, object],
    proxy_mean_mode: str,
) -> np.ndarray:
    zeros_obs = ds["zeros_obs"]
    proxy_target = ds["norm_counts"].copy()
    mode = str(proxy_mean_mode).lower()
    if mode == "gene":
        proxy_mean = ds["gene_mean_nz"]
        if zeros_obs.any():
            rr, cc = np.where(zeros_obs)
            proxy_target[rr, cc] = proxy_mean[cc]
    else:
        knn_log = ds.get("knn_log_mean")
        if knn_log is not None:
            proxy_norm = logcounts_to_counts(knn_log)
            proxy_target[zeros_obs] = proxy_norm[zeros_obs]

    proxy_bio_mask = ds.get("proxy_bio_mask")
    proxy_bio_label = ds.get("proxy_bio_label")
    if proxy_bio_mask is not None and proxy_bio_label is not None:
        bio_mask = proxy_bio_mask & (proxy_bio_label > 0.5)
        if np.any(bio_mask):
            proxy_target[bio_mask] = 0.0

    proxy_target = np.nan_to_num(proxy_target, nan=0.0, posinf=0.0, neginf=0.0)
    proxy_target = np.clip(proxy_target, 0.0, None)
    return proxy_target.astype(np.float32, copy=False)


def _pava_nonincreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32, copy=True)
    vals: List[float] = []
    wts: List[float] = []
    sizes: List[int] = []
    for v, w in zip(values, weights):
        vals.append(float(v))
        wts.append(float(w))
        sizes.append(1)
        while len(vals) >= 2 and (vals[-2] < vals[-1] - 1e-12):
            w_sum = wts[-2] + wts[-1]
            if w_sum <= EPSILON:
                merged = 0.5 * (vals[-2] + vals[-1])
            else:
                merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w_sum
            sizes[-2] += sizes[-1]
            vals[-2] = merged
            wts[-2] = w_sum
            vals.pop()
            wts.pop()
            sizes.pop()
    out = np.empty(len(values), dtype=np.float32)
    idx = 0
    for v, sz in zip(vals, sizes):
        out[idx : idx + sz] = v
        idx += sz
    return out


def _assign_quantile_bins(values: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = int(bins)
    if n_bins <= 1 or values.size == 0:
        edges = np.array([0.0, 1.0], dtype=np.float32)
        return np.zeros_like(values, dtype=np.int32), edges
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0
    uniq = [float(edges[0])]
    for val in edges[1:]:
        if val > uniq[-1] + 1e-8:
            uniq.append(float(val))
    if len(uniq) < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.array(uniq, dtype=np.float32)
    n_bins = max(int(edges.size - 1), 1)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)
    return idx.astype(np.int32, copy=False), edges


def _apply_zero_iso_scale(
    pred: np.ndarray,
    target: np.ndarray,
    zeros_obs: np.ndarray,
    p_bio: np.ndarray,
    weight: float,
    bins: int,
    gamma: float,
    bio_weight: float,
    min_scale: float,
    max_scale: float,
) -> Tuple[np.ndarray, float]:
    weight = float(weight)
    if weight <= 0.0:
        return pred, float("nan")
    p_use = np.clip(p_bio, 0.0, 1.0) ** float(gamma)
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if max_scale <= 0.0:
        return pred, float("nan")
    log_adj = pred.copy()

    def _fit_bins(z_all: np.ndarray, x_all: np.ndarray, y_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        if z_all.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32), float("nan")
        bin_idx, edges = _assign_quantile_bins(z_all, bins)
        x_fit = x_all
        y_fit = y_all
        bin_idx_fit = bin_idx
        weight_mult = np.ones_like(x_fit, dtype=np.float64)
        if float(bio_weight) > 0.0:
            weight_mult = weight_mult + float(bio_weight) * (y_fit <= EPSILON)
        sum_x2 = np.bincount(
            bin_idx_fit, weights=weight_mult * x_fit * x_fit, minlength=int(edges.size - 1)
        )
        sum_xy = np.bincount(
            bin_idx_fit, weights=weight_mult * x_fit * y_fit, minlength=int(edges.size - 1)
        )
        scales = np.ones((int(edges.size - 1),), dtype=np.float32)
        for i in range(scales.size):
            if sum_x2[i] > EPSILON:
                scales[i] = float(sum_xy[i] / sum_x2[i])
        scales = _pava_nonincreasing(scales, sum_x2)
        scales = np.clip(scales, min_scale, max_scale)
        mean_scale = float(np.nanmean(scales)) if scales.size else float("nan")
        return scales.astype(np.float32, copy=False), bin_idx, mean_scale

    n_genes = pred.shape[1]
    scales_mean: List[float] = []
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        z = p_use[mask, j]
        x = pred[mask, j]
        y = target[mask, j]
        scales, bin_idx, mean_scale = _fit_bins(z, x, y)
        if scales.size == 0:
            continue
        scale = scales[bin_idx]
        scale_blend = (1.0 - weight) + weight * scale
        log_adj[mask, j] = log_adj[mask, j] * scale_blend
        scales_mean.append(mean_scale)
    mean_scale = float(np.nanmean(scales_mean)) if scales_mean else float("nan")
    return log_adj, mean_scale


def _apply_dropout_iso_scale(
    pred: np.ndarray,
    target: np.ndarray,
    zeros_obs: np.ndarray,
    p_bio: np.ndarray,
    weight: float,
    bins: int,
    gamma: float,
    min_scale: float,
    max_scale: float,
    p_max: float,
) -> Tuple[np.ndarray, float]:
    weight = float(weight)
    if weight <= 0.0:
        return pred, float("nan")
    p_max = float(p_max)
    if p_max <= 0.0:
        return pred, float("nan")
    p_use = np.clip(p_bio, 0.0, 1.0) ** float(gamma)
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if max_scale <= 0.0:
        return pred, float("nan")
    log_adj = pred.copy()

    def _fit_bins(z_all: np.ndarray, x_all: np.ndarray, y_all: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        if z_all.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int32), float("nan")
        bin_idx, edges = _assign_quantile_bins(z_all, bins)
        drop_mask = y_all > EPSILON
        if not np.any(drop_mask):
            scales = np.ones((int(edges.size - 1),), dtype=np.float32)
            return scales, bin_idx, float(np.nanmean(scales))
        x_fit = x_all[drop_mask]
        y_fit = y_all[drop_mask]
        bin_idx_fit = bin_idx[drop_mask]
        sum_x2 = np.bincount(bin_idx_fit, weights=x_fit * x_fit, minlength=int(edges.size - 1))
        sum_xy = np.bincount(bin_idx_fit, weights=x_fit * y_fit, minlength=int(edges.size - 1))
        scales = np.ones((int(edges.size - 1),), dtype=np.float32)
        for i in range(scales.size):
            if sum_x2[i] > EPSILON:
                scales[i] = float(sum_xy[i] / sum_x2[i])
        scales = _pava_nonincreasing(scales, sum_x2)
        scales = np.clip(scales, min_scale, max_scale)
        mean_scale = float(np.nanmean(scales)) if scales.size else float("nan")
        return scales.astype(np.float32, copy=False), bin_idx, mean_scale

    n_genes = pred.shape[1]
    scales_mean: List[float] = []
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        z = p_use[mask, j]
        x = pred[mask, j]
        y = target[mask, j]
        scales, bin_idx, mean_scale = _fit_bins(z, x, y)
        if scales.size == 0:
            continue
        scale = scales[bin_idx]
        apply_mask = z <= p_max
        if not np.any(apply_mask):
            continue
        scale_adj = np.ones_like(scale, dtype=np.float32)
        scale_adj[apply_mask] = scale[apply_mask]
        scale_blend = (1.0 - weight) + weight * scale_adj
        log_adj[mask, j] = log_adj[mask, j] * scale_blend
        scales_mean.append(mean_scale)
    mean_scale = float(np.nanmean(scales_mean)) if scales_mean else float("nan")
    return log_adj, mean_scale


def _apply_constrained_zero_scale(
    pred: np.ndarray,
    target: np.ndarray,
    zeros_obs: np.ndarray,
    max_mse_inc: float,
    lambda_max: float,
    iters: int,
) -> Tuple[np.ndarray, float]:
    diff_base = target - pred
    mse_base = _mse_from_diff(diff_base)
    mse_target = mse_base * (1.0 + float(max_mse_inc))
    if mse_target <= 0.0:
        return pred, float("nan")
    n_genes = pred.shape[1]
    sum_x2 = np.zeros(n_genes, dtype=np.float64)
    sum_xy = np.zeros(n_genes, dtype=np.float64)
    sum_x2_bio = np.zeros(n_genes, dtype=np.float64)
    sum_xy_bio = np.zeros(n_genes, dtype=np.float64)
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        x = pred[mask, j].astype(np.float64, copy=False)
        y = target[mask, j].astype(np.float64, copy=False)
        sum_x2[j] = float(np.sum(x * x))
        sum_xy[j] = float(np.sum(x * y))
        bio_mask = y <= EPSILON
        if np.any(bio_mask):
            xb = x[bio_mask]
            yb = y[bio_mask]
            sum_x2_bio[j] = float(np.sum(xb * xb))
            sum_xy_bio[j] = float(np.sum(xb * yb))

    def _scales_for_lambda(lam: float) -> np.ndarray:
        scales = np.ones(n_genes, dtype=np.float64)
        lam_f = float(lam)
        for j in range(n_genes):
            if sum_x2[j] <= EPSILON:
                continue
            if sum_x2_bio[j] <= EPSILON:
                s = sum_xy[j] / max(sum_x2[j], EPSILON)
            else:
                denom = sum_x2_bio[j] + lam_f * sum_x2[j]
                if denom <= EPSILON:
                    s = 1.0
                else:
                    s = (sum_xy_bio[j] + lam_f * sum_xy[j]) / denom
            scales[j] = float(np.clip(s, 0.0, 1.0))
        return scales

    def _eval_lambda(lam: float) -> Tuple[float, np.ndarray]:
        scales = _scales_for_lambda(lam)
        log_adj = pred.copy()
        for j in range(n_genes):
            s = scales[j]
            if abs(s - 1.0) < 1e-6:
                continue
            mask = zeros_obs[:, j]
            if not np.any(mask):
                continue
            log_adj[mask, j] = log_adj[mask, j] * s
        mse = _mse_from_diff(target - log_adj)
        return mse, log_adj

    mse_low, log_low = _eval_lambda(0.0)
    if mse_low <= mse_target:
        return log_low, 0.0

    lam_high = float(lambda_max)
    mse_high, log_high = _eval_lambda(lam_high)
    if mse_high > mse_target:
        return log_high, lam_high

    lo = 0.0
    hi = lam_high
    best_log = log_high
    best_lam = lam_high
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        mse_mid, log_mid = _eval_lambda(mid)
        if mse_mid <= mse_target:
            hi = mid
            best_log = log_mid
            best_lam = mid
        else:
            lo = mid
    return best_log, best_lam


def _score(avg_mse: float, avg_bz_mse: float) -> float:
    return float(avg_bz_mse) + float(LAMBDA_MSE) * float(avg_mse)


def run_pipeline(
    ds: Dict[str, object],
    zero_iso_weight: float,
    dropout_iso_weight: float,
    constrained_zero_scale: bool,
    renorm_imputed: bool,
    p_bio_temp: float,
    p_bio_bias: float,
    thr_drop: float,
    p_bio_nz_scale: float,
    post_log_quantile: float,
    cell_scale_alpha: float,
    sf_dropout_alpha: float,
    proxy_calib_shrink: float,
    proxy_bio_weight: float,
    proxy_drop_weight: float,
    proxy_impute_alpha: float,
    proxy_impute_gamma: float,
    proxy_mean_mode: str,
    proxy_calib_min_points: int,
    proxy_pu_weight: float,
    bio_soft_gamma: float,
    proxy_calib_mode: str,
    impute_mode: str,
    impute_blend_alpha: float,
) -> Dict[str, float]:
    p_bio_use = _logit_scale_probs(ds["p_bio_base"], temp=p_bio_temp, bias=p_bio_bias)
    p_bio_proxy = ds.get("p_bio_proxy")
    if p_bio_proxy is not None and float(proxy_pu_weight) > 0.0:
        w_proxy = float(np.clip(proxy_pu_weight, 0.0, 1.0))
        p_bio_use = (1.0 - w_proxy) * p_bio_use + w_proxy * p_bio_proxy
    p_bio_use = calibrate_p_bio_with_proxy(
        p_bio_use,
        zeros_obs=ds["zeros_obs"],
        proxy_bio_mask=ds.get("proxy_bio_mask"),
        proxy_bio_label=ds.get("proxy_bio_label"),
        weight_bio=proxy_bio_weight,
        weight_drop=proxy_drop_weight,
    )
    nz_scale = float(np.clip(p_bio_nz_scale, 0.0, 1.0))
    if nz_scale < 1.0:
        nz_mask = ~ds["zeros_obs"]
        if np.any(nz_mask):
            p_bio_use = p_bio_use.copy()
            p_bio_use[nz_mask] = p_bio_use[nz_mask] * nz_scale
    proxy_target = _build_proxy_target(
        ds=ds,
        proxy_mean_mode=proxy_mean_mode,
    )

    impute_mode_l = str(impute_mode).lower()
    zeros_obs = ds["zeros_obs"]
    pred_bio_mask = None

    if impute_mode_l == "count_ae":
        device = ds.get("device", torch.device("cpu"))
        norm_factor = get_norm_factor(ds)
        recon_log, sf = get_count_ae_recon(
            ds,
            sf_dropout_alpha=sf_dropout_alpha,
            device=device,
            norm_factor=norm_factor,
        )
        counts_obs = ds["counts"]
        counts_norm = counts_obs / sf[:, None] * float(norm_factor)
        counts_norm = np.nan_to_num(
            counts_norm, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)
        logcounts_norm = np.log1p(counts_norm) / np.log(2.0)
        norm_imputed_raw = logcounts_to_counts(recon_log)
        norm_imputed_final = norm_imputed_raw.copy()
        if bool(CONFIG["keep_positive"]):
            norm_imputed_final[~zeros_obs] = counts_norm[~zeros_obs]

        thr_bio = 1.0 - float(thr_drop)
        pred_bio_mask = p_bio_use >= float(thr_bio)
        if bool(CONFIG["hard_zero_bio"]):
            norm_imputed_final[pred_bio_mask] = 0.0

        log_imputed = np.log1p(np.clip(norm_imputed_final, 0.0, None)) / np.log(2.0)
        if float(post_log_quantile) > 0.0:
            thresh = _get_gene_log_threshold_from_mat(logcounts_norm, float(post_log_quantile))
            log_imputed = log_imputed.copy()
            log_imputed[log_imputed < thresh[None, :]] = 0.0
    elif impute_mode_l == "count":
        counts_obs = ds["counts"]
        norm_factor = get_norm_factor(ds)
        lib_obs = counts_obs.sum(axis=1).astype(np.float32)
        counts_norm_ae = ds.get("norm_imputed_raw")
        if counts_norm_ae is None:
            counts_norm_ae = _log_normalize_counts(counts_obs, norm_factor=norm_factor)
        counts_ae_raw = counts_norm_ae / float(norm_factor) * lib_obs[:, None]
        counts_ae_raw = np.nan_to_num(
            counts_ae_raw, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)

        if bool(CONFIG["keep_positive"]):
            counts_ae_raw[~zeros_obs] = counts_obs[~zeros_obs]

        thr_bio = 1.0 - float(thr_drop)
        pred_bio_mask = p_bio_use >= float(thr_bio)
        if bool(CONFIG["hard_zero_bio"]):
            counts_ae_raw[pred_bio_mask] = 0.0

        proxy_bio_mask = ds.get("proxy_bio_mask")
        proxy_bio_label = ds.get("proxy_bio_label")
        if proxy_bio_mask is not None and proxy_bio_label is not None:
            bio_mask = proxy_bio_mask & (proxy_bio_label > 0.5)
            if np.any(bio_mask):
                counts_ae_raw[bio_mask] = 0.0

        if float(sf_dropout_alpha) <= 0.0:
            sf = np.maximum(lib_obs.astype(np.float64), 1.0)
        else:
            sf = estimate_dropout_aware_size_factors(
                counts=counts_obs,
                zeros_obs=zeros_obs,
                p_bio=p_bio_use,
                alpha=float(sf_dropout_alpha),
                max_scale=float(CONFIG["sf_dropout_max_scale"]),
            )
        counts_norm = counts_ae_raw / sf[:, None] * float(norm_factor)
        counts_norm = np.nan_to_num(counts_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
        logcounts_norm = np.log1p(counts_norm) / np.log(2.0)

        knn_log_mean = None
        knn_valid = None
        if str(proxy_mean_mode).lower() != "gene":
            knn_log_mean, knn_nz_count = compute_knn_log_mean(
                logcounts_norm,
                k=int(CONFIG["proxy_knn_k"]),
                pca_dim=int(CONFIG["proxy_knn_pca"]),
                ignore_zeros=bool(CONFIG["proxy_knn_ignore_zeros"]),
            )
            if knn_nz_count is not None:
                knn_valid = knn_nz_count >= int(CONFIG["proxy_knn_min_nz"])

        nz_mask = counts_norm > 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            gene_mean_nz = np.sum(counts_norm * nz_mask, axis=0) / np.maximum(nz_mask.sum(axis=0), 1)
        gene_mean_nz = np.nan_to_num(
            gene_mean_nz, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        proxy_mean = None
        if str(proxy_mean_mode).lower() == "gene" or knn_log_mean is None:
            proxy_mean = np.broadcast_to(gene_mean_nz, counts_norm.shape)
        else:
            proxy_mean = logcounts_to_counts(knn_log_mean)
            if knn_valid is not None:
                gene_mean_mat = np.broadcast_to(gene_mean_nz, counts_norm.shape)
                proxy_mean = np.where(knn_valid, proxy_mean, gene_mean_mat)

        if proxy_bio_mask is not None and proxy_bio_label is not None and proxy_mean is not None:
            drop_mask = proxy_bio_mask & (proxy_bio_label <= 0.5)
            if np.any(drop_mask):
                counts_norm[drop_mask] = proxy_mean[drop_mask]

        if float(proxy_impute_alpha) > 0.0 and proxy_mean is not None:
            alpha = float(np.clip(proxy_impute_alpha, 0.0, 1.0))
            gamma = float(max(proxy_impute_gamma, 0.0))
            p_scale = np.clip(1.0 - p_bio_use, 0.0, 1.0)
            if gamma != 1.0:
                p_scale = p_scale ** gamma
            apply_mask = zeros_obs
            if pred_bio_mask is not None:
                apply_mask = zeros_obs & (~pred_bio_mask)
            counts_norm[apply_mask] = (1.0 - alpha) * counts_norm[apply_mask] + alpha * (
                proxy_mean[apply_mask] * p_scale[apply_mask]
            )

        if float(bio_soft_gamma) > 0.0:
            scale = np.clip(1.0 - p_bio_use, 0.0, 1.0) ** float(bio_soft_gamma)
            counts_norm = counts_norm.copy()
            counts_norm[zeros_obs] = counts_norm[zeros_obs] * scale[zeros_obs]

        log_imputed = np.log1p(np.clip(counts_norm, 0.0, None)) / np.log(2.0)
        if float(proxy_calib_shrink) > 0.0:
            proxy_log = logcounts_norm.copy()
            if proxy_bio_mask is not None and proxy_bio_label is not None and knn_log_mean is not None:
                proxy_log = proxy_log.astype(np.float32, copy=True)
                bio_mask = proxy_bio_mask & (proxy_bio_label > 0.5)
                drop_mask = proxy_bio_mask & (proxy_bio_label <= 0.5)
                proxy_log[bio_mask] = 0.0
                proxy_log[drop_mask] = knn_log_mean[drop_mask]
            log_imputed = calibrate_log_to_proxy(
                log_imputed,
                proxy_log,
                shrink=proxy_calib_shrink,
                mask=(~zeros_obs) | (proxy_bio_mask if proxy_bio_mask is not None else False),
                min_points=proxy_calib_min_points,
            )
        if float(post_log_quantile) > 0.0:
            thresh = _get_gene_log_threshold_from_mat(logcounts_norm, float(post_log_quantile))
            log_imputed = log_imputed.copy()
            log_imputed[log_imputed < thresh[None, :]] = 0.0
    elif impute_mode_l == "ae":
        norm_imputed_final, p_bio_use, pred_bio_mask = _postprocess_imputation(
            norm_imputed_raw=ds["norm_imputed_raw"],
            p_bio_post=p_bio_use,
            ds=ds,
            thr_drop=thr_drop,
        )

        p_bio_mix = np.clip(p_bio_use, 0.0, 1.0)
        if pred_bio_mask is not None:
            p_bio_mix = p_bio_mix.copy()
            p_bio_mix[pred_bio_mask] = 1.0
        p_bio_mix[~zeros_obs] = 0.0

        if float(zero_iso_weight) > 0.0:
            norm_imputed_final, _ = _apply_zero_iso_scale(
                pred=norm_imputed_final,
                target=proxy_target,
                zeros_obs=zeros_obs,
                p_bio=p_bio_mix,
                weight=float(zero_iso_weight),
                bins=int(CONFIG["zero_iso_bins"]),
                gamma=float(CONFIG["zero_iso_gamma"]),
                bio_weight=float(CONFIG["zero_iso_bio_weight"]),
                min_scale=float(CONFIG["zero_iso_min_scale"]),
                max_scale=float(CONFIG["zero_iso_max_scale"]),
            )

        if float(dropout_iso_weight) > 0.0:
            norm_imputed_final, _ = _apply_dropout_iso_scale(
                pred=norm_imputed_final,
                target=proxy_target,
                zeros_obs=zeros_obs,
                p_bio=p_bio_mix,
                weight=float(dropout_iso_weight),
                bins=int(CONFIG["dropout_iso_bins"]),
                gamma=float(CONFIG["dropout_iso_gamma"]),
                min_scale=float(CONFIG["dropout_iso_min_scale"]),
                max_scale=float(CONFIG["dropout_iso_max_scale"]),
                p_max=float(CONFIG["dropout_iso_pmax"]),
            )

        if bool(constrained_zero_scale):
            norm_imputed_final, _ = _apply_constrained_zero_scale(
                pred=norm_imputed_final,
                target=proxy_target,
                zeros_obs=zeros_obs,
                max_mse_inc=float(CONFIG["constrained_zero_max_mse_inc"]),
                lambda_max=float(CONFIG["constrained_zero_lambda_max"]),
                iters=int(CONFIG["constrained_zero_iters"]),
            )

        norm10k_imputed = norm_imputed_final * ds["gene_scale"][None, :]
        norm10k_imputed = np.clip(norm10k_imputed, 0.0, None)
        log_imputed = np.log1p(norm10k_imputed) / np.log(2.0)
    else:
        if impute_mode_l == "gene":
            log_imputed = ds["log_imputed_gene"].copy()
        elif impute_mode_l == "knn" and ds.get("log_imputed_knn") is not None:
            log_imputed = ds["log_imputed_knn"].copy()
        elif impute_mode_l == "blend" and ds.get("log_imputed_ae") is not None:
            alpha = float(np.clip(impute_blend_alpha, 0.0, 1.0))
            base = ds.get("log_imputed_gene")
            if base is None:
                base = ds["logcounts"]
            log_imputed = (1.0 - alpha) * ds["log_imputed_ae"] + alpha * base
        else:
            log_imputed = ds.get("log_imputed_ae", ds["logcounts"]).copy()

        if bool(CONFIG["keep_positive"]):
            log_imputed[~zeros_obs] = ds["logcounts"][~zeros_obs]

        proxy_bio_mask = ds.get("proxy_bio_mask")
        proxy_bio_label = ds.get("proxy_bio_label")
        if proxy_bio_mask is not None and proxy_bio_label is not None:
            bio_mask = proxy_bio_mask & (proxy_bio_label > 0.5)
            drop_mask = proxy_bio_mask & (proxy_bio_label <= 0.5)
            if np.any(bio_mask):
                log_imputed[bio_mask] = 0.0
            if np.any(drop_mask):
                mode = str(proxy_mean_mode).lower()
                gene_mean_log = ds["gene_log_mean_nz"]
                gene_mean_log_mat = np.broadcast_to(
                    gene_mean_log, ds["logcounts"].shape
                )
                if mode == "gene" or ds.get("knn_log_mean") is None:
                    proxy_mean = gene_mean_log_mat
                else:
                    proxy_mean = ds["knn_log_mean"]
                    knn_valid = ds.get("knn_valid_mask")
                    if knn_valid is not None:
                        proxy_mean = np.where(knn_valid, proxy_mean, gene_mean_log_mat)
                log_imputed[drop_mask] = proxy_mean[drop_mask]

        thr_bio = 1.0 - float(thr_drop)
        pred_bio_mask = p_bio_use >= float(thr_bio)
        if bool(CONFIG["hard_zero_bio"]):
            if proxy_bio_mask is None:
                log_imputed[pred_bio_mask] = 0.0
            else:
                unlabeled = zeros_obs & (~proxy_bio_mask)
                log_imputed[pred_bio_mask & unlabeled] = 0.0
    zeros_obs = ds["zeros_obs"]
    knn_log = ds.get("knn_log_mean")
    if float(proxy_impute_alpha) > 0.0:
        mode = str(proxy_mean_mode).lower()
        gene_mean_log = ds["gene_log_mean_nz"]
        gene_mean_log_mat = np.broadcast_to(
            gene_mean_log, ds["logcounts"].shape
        )
        if mode == "gene" or knn_log is None:
            proxy_mean = gene_mean_log_mat
        else:
            proxy_mean = knn_log
            knn_valid = ds.get("knn_valid_mask")
            if knn_valid is not None:
                proxy_mean = np.where(knn_valid, proxy_mean, gene_mean_log_mat)
        alpha = float(np.clip(proxy_impute_alpha, 0.0, 1.0))
        gamma = float(max(proxy_impute_gamma, 0.0))
        p_scale = np.clip(1.0 - p_bio_use, 0.0, 1.0)
        if gamma != 1.0:
            p_scale = p_scale ** gamma
        apply_mask = zeros_obs
        if pred_bio_mask is not None:
            apply_mask = zeros_obs & (~pred_bio_mask)
        log_imputed = log_imputed.copy()
        log_imputed[apply_mask] = (1.0 - alpha) * log_imputed[apply_mask] + alpha * (
            proxy_mean[apply_mask] * p_scale[apply_mask]
        )
    if float(proxy_calib_shrink) > 0.0:
        proxy_log = ds["logcounts"].copy()
        proxy_bio_mask = ds.get("proxy_bio_mask")
        proxy_bio_label = ds.get("proxy_bio_label")
        if knn_log is not None:
            if proxy_bio_mask is not None and proxy_bio_label is not None:
                proxy_log = proxy_log.astype(np.float32, copy=True)
                bio_mask = proxy_bio_mask & (proxy_bio_label > 0.5)
                drop_mask = proxy_bio_mask & (proxy_bio_label <= 0.5)
                proxy_log[bio_mask] = 0.0
                proxy_log[drop_mask] = knn_log[drop_mask]
            if str(proxy_calib_mode).lower() == "all":
                knn_valid = ds.get("knn_valid_mask")
                if knn_valid is not None:
                    proxy_log[zeros_obs & knn_valid] = knn_log[zeros_obs & knn_valid]
                else:
                    proxy_log[zeros_obs] = knn_log[zeros_obs]
        mode = str(proxy_calib_mode).lower()
        if mode == "nonzero":
            proxy_mask = ~zeros_obs
        elif mode == "all":
            proxy_mask = np.ones_like(zeros_obs, dtype=bool)
        else:
            if proxy_bio_mask is None:
                proxy_mask = ~zeros_obs
            else:
                proxy_mask = (~zeros_obs) | proxy_bio_mask
        log_imputed = calibrate_log_to_proxy(
            log_imputed,
            proxy_log,
            shrink=proxy_calib_shrink,
            mask=proxy_mask,
            min_points=proxy_calib_min_points,
        )
    if float(bio_soft_gamma) > 0.0:
        zeros_obs = ds["zeros_obs"]
        scale = np.clip(1.0 - p_bio_use, 0.0, 1.0) ** float(bio_soft_gamma)
        log_imputed = log_imputed.copy()
        log_imputed[zeros_obs] = log_imputed[zeros_obs] * scale[zeros_obs]
    if float(post_log_quantile) > 0.0:
        thresh = _get_gene_log_threshold(ds, float(post_log_quantile))
        log_imputed = log_imputed.copy()
        log_imputed[log_imputed < thresh[None, :]] = 0.0

    if float(cell_scale_alpha) > 0.0:
        alpha = float(cell_scale_alpha)
        zeros_obs = ds["zeros_obs"]
        n_genes = max(zeros_obs.shape[1], 1)
        obs_zero_frac = zeros_obs.mean(axis=1)
        bio_zero_frac = np.zeros_like(obs_zero_frac)
        if p_bio_use is not None:
            bio_zero_frac = (p_bio_use * zeros_obs).sum(axis=1) / float(n_genes)
        drop_zero_frac = np.clip(obs_zero_frac - bio_zero_frac, 0.0, 0.95)
        denom = np.maximum(1.0 - drop_zero_frac, 1e-3)
        scale = np.clip(denom ** (-alpha), 1.0, float(CONFIG["cell_scale_max"]))
        counts = np.clip(logcounts_to_counts(log_imputed), 0.0, None)
        counts = counts * scale[:, None]
        log_imputed = np.log1p(counts) / np.log(2.0)

    if bool(renorm_imputed):
        log_imputed = renorm_logcounts(log_imputed, norm_factor=get_norm_factor(ds))

    return compute_mse_metrics(
        log_imputed,
        ds["log_true"],
        ds["counts"],
    )


def evaluate_proxy_grid(
    datasets: List[Dict[str, object]],
    zero_iso_weight: float,
    dropout_iso_weight: float,
    constrained_zero_scale: bool,
    renorm_imputed: bool,
    p_bio_temp: float,
    p_bio_bias: float,
    thr_drop: float,
    p_bio_nz_scale: float,
    post_log_quantile: float,
    cell_scale_alpha: float,
    sf_dropout_alpha: float,
    proxy_calib_shrink: float,
    proxy_bio_weight: float,
    proxy_drop_weight: float,
    proxy_impute_alpha: float,
    proxy_impute_gamma: float,
    proxy_mean_mode: str,
    proxy_calib_min_points: int,
    proxy_pu_weight: float,
    bio_soft_gamma: float,
    proxy_calib_mode: str,
    impute_mode: str,
    impute_blend_alpha: float,
) -> Tuple[float, float, float]:
    mse_list: List[float] = []
    bz_list: List[float] = []
    for ds in datasets:
        metrics = run_pipeline(
            ds=ds,
            zero_iso_weight=zero_iso_weight,
            dropout_iso_weight=dropout_iso_weight,
            constrained_zero_scale=constrained_zero_scale,
            renorm_imputed=renorm_imputed,
            p_bio_temp=p_bio_temp,
            p_bio_bias=p_bio_bias,
            thr_drop=thr_drop,
            p_bio_nz_scale=p_bio_nz_scale,
            post_log_quantile=post_log_quantile,
            cell_scale_alpha=cell_scale_alpha,
            sf_dropout_alpha=sf_dropout_alpha,
            proxy_calib_shrink=proxy_calib_shrink,
            proxy_bio_weight=proxy_bio_weight,
            proxy_drop_weight=proxy_drop_weight,
            proxy_impute_alpha=proxy_impute_alpha,
            proxy_impute_gamma=proxy_impute_gamma,
            proxy_mean_mode=proxy_mean_mode,
            proxy_calib_min_points=proxy_calib_min_points,
            proxy_pu_weight=proxy_pu_weight,
            bio_soft_gamma=bio_soft_gamma,
            proxy_calib_mode=proxy_calib_mode,
            impute_mode=impute_mode,
            impute_blend_alpha=impute_blend_alpha,
        )
        mse_list.append(float(metrics["mse"]))
        bz_list.append(float(metrics["mse_biozero"]))
    avg_mse = float(np.nanmean(mse_list)) if mse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = _score(avg_mse, avg_bz)
    return avg_mse, avg_bz, score


def evaluate_datasets(
    datasets: List[Dict[str, object]],
    zero_iso_weight: float,
    dropout_iso_weight: float,
    constrained_zero_scale: bool,
    renorm_imputed: bool,
    p_bio_temp: float,
    p_bio_bias: float,
    thr_drop: float,
    p_bio_nz_scale: float,
    post_log_quantile: float,
    cell_scale_alpha: float,
    sf_dropout_alpha: float,
    proxy_calib_shrink: float,
    proxy_bio_weight: float,
    proxy_drop_weight: float,
    proxy_impute_alpha: float,
    proxy_impute_gamma: float,
    proxy_mean_mode: str,
    proxy_calib_min_points: int,
    proxy_pu_weight: float,
    bio_soft_gamma: float,
    proxy_calib_mode: str,
    impute_mode: str,
    impute_blend_alpha: float,
) -> Tuple[List[Dict[str, float]], float, float, float]:
    metrics_all: List[Dict[str, float]] = []
    mse_list: List[float] = []
    bz_list: List[float] = []
    for ds in datasets:
        metrics = run_pipeline(
            ds=ds,
            zero_iso_weight=zero_iso_weight,
            dropout_iso_weight=dropout_iso_weight,
            constrained_zero_scale=constrained_zero_scale,
            renorm_imputed=renorm_imputed,
            p_bio_temp=p_bio_temp,
            p_bio_bias=p_bio_bias,
            thr_drop=thr_drop,
            p_bio_nz_scale=p_bio_nz_scale,
            post_log_quantile=post_log_quantile,
            cell_scale_alpha=cell_scale_alpha,
            sf_dropout_alpha=sf_dropout_alpha,
            proxy_calib_shrink=proxy_calib_shrink,
            proxy_bio_weight=proxy_bio_weight,
            proxy_drop_weight=proxy_drop_weight,
            proxy_impute_alpha=proxy_impute_alpha,
            proxy_impute_gamma=proxy_impute_gamma,
            proxy_mean_mode=proxy_mean_mode,
            proxy_calib_min_points=proxy_calib_min_points,
            proxy_pu_weight=proxy_pu_weight,
            bio_soft_gamma=bio_soft_gamma,
            proxy_calib_mode=proxy_calib_mode,
            impute_mode=impute_mode,
            impute_blend_alpha=impute_blend_alpha,
        )
        metrics = dict(metrics)
        metrics["dataset"] = str(ds["dataset"])
        metrics_all.append(metrics)
        mse_list.append(float(metrics["mse"]))
        bz_list.append(float(metrics["mse_biozero"]))
    avg_mse = float(np.nanmean(mse_list)) if mse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = _score(avg_mse, avg_bz)
    return metrics_all, avg_mse, avg_bz, score


def write_mse_table(path: Path, metrics: List[Dict[str, float]], avg_mse: float, avg_bz: float) -> None:
    header = [
        "dataset",
        "mse",
        "mse_dropout",
        "mse_biozero",
        "mse_non_zero",
        "n_total",
        "n_dropout",
        "n_biozero",
        "n_non_zero",
    ]
    lines = ["\t".join(header)]
    for row in metrics:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    lines.append("\t".join(["AVG", f"{avg_mse}", "", f"{avg_bz}", "", "", "", "", ""]))
    path.write_text("\n".join(lines) + "\n")


def write_tuning_table(path: Path, rows: List[Dict[str, object]]) -> None:
    header = [
        "config_id",
        "zero_iso_weight",
        "dropout_iso_weight",
        "constrained_zero_scale",
        "renorm_imputed",
        "p_bio_temp",
        "p_bio_bias",
        "thr_drop",
        "p_bio_nz_scale",
        "post_log_quantile",
        "cell_scale_alpha",
        "sf_dropout_alpha",
        "proxy_calib_shrink",
        "proxy_bio_weight",
        "proxy_drop_weight",
        "proxy_impute_alpha",
        "proxy_impute_gamma",
        "proxy_mean_mode",
        "proxy_calib_min_points",
        "proxy_pu_weight",
        "bio_soft_gamma",
        "proxy_calib_mode",
        "impute_mode",
        "impute_blend_alpha",
        "avg_mse",
        "avg_bz_mse",
        "score",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune proxy calibration parameters for MSE on normalized counts "
            "using count-derived proxy calibration (no TrueCounts for calibration)."
        )
    )
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proxy-knn-k", type=int, default=CONFIG["proxy_knn_k"])
    parser.add_argument("--proxy-knn-pca", type=int, default=CONFIG["proxy_knn_pca"])
    parser.add_argument("--proxy-knn-min-nz", type=int, default=CONFIG["proxy_knn_min_nz"])
    parser.add_argument("--proxy-knn-q-low", type=float, default=CONFIG["proxy_knn_q_low"])
    parser.add_argument("--proxy-knn-q-high", type=float, default=CONFIG["proxy_knn_q_high"])
    parser.add_argument(
        "--proxy-knn-label-min-points",
        type=int,
        default=CONFIG["proxy_knn_label_min_points"],
    )
    parser.add_argument(
        "--proxy-knn-label",
        action="store_true",
        default=CONFIG["proxy_knn_label"],
    )
    parser.add_argument(
        "--no-proxy-knn-label",
        dest="proxy_knn_label",
        action="store_false",
    )
    parser.add_argument(
        "--proxy-knn-ignore-zeros",
        action="store_true",
        default=CONFIG["proxy_knn_ignore_zeros"],
    )
    parser.add_argument(
        "--proxy-knn-use-zeros",
        action="store_false",
        dest="proxy_knn_ignore_zeros",
    )
    parser.add_argument("--bio-model", default=CONFIG["bio_model"])
    parser.add_argument("--bio-mix-weight", type=float, default=CONFIG["bio_mix_weight"])
    parser.add_argument("--zinb-phi-min", type=float, default=CONFIG["zinb_phi_min"])
    parser.add_argument("--zinb-pi-max", type=float, default=CONFIG["zinb_pi_max"])
    parser.add_argument("--zinb-cluster-k", type=int, default=CONFIG["zinb_cluster_k"])
    parser.add_argument("--zinb-cluster-pca", type=int, default=CONFIG["zinb_cluster_pca"])
    parser.add_argument("--zinb-cluster-iters", type=int, default=CONFIG["zinb_cluster_iters"])
    parser.add_argument("--zinb-cluster-min-cells", type=int, default=CONFIG["zinb_cluster_min_cells"])
    parser.add_argument("--zinb-cluster-seed", type=int, default=CONFIG["zinb_cluster_seed"])
    parser.add_argument("--zinb-cluster-pi-tau", type=float, default=CONFIG["zinb_cluster_pi_tau"])
    parser.add_argument("--zinb-cluster-phi-tau", type=float, default=CONFIG["zinb_cluster_phi_tau"])
    parser.add_argument(
        "--zinb-cluster-update-phi",
        action="store_true",
        default=CONFIG["zinb_cluster_update_phi"],
    )
    parser.add_argument(
        "--no-zinb-cluster-update-phi",
        dest="zinb_cluster_update_phi",
        action="store_false",
    )
    parser.add_argument(
        "--knn-bio-mix-weight",
        type=float,
        default=CONFIG["knn_bio_mix_weight"],
    )
    parser.add_argument(
        "--knn-bio-mix-iters",
        type=int,
        default=CONFIG["knn_bio_mix_iters"],
    )
    parser.add_argument(
        "--knn-bio-mix-min-points",
        type=int,
        default=CONFIG["knn_bio_mix_min_points"],
    )
    parser.add_argument(
        "--knn-bio-mix-q-low",
        type=float,
        default=CONFIG["knn_bio_mix_q_low"],
    )
    parser.add_argument(
        "--knn-bio-mix-q-high",
        type=float,
        default=CONFIG["knn_bio_mix_q_high"],
    )
    parser.add_argument(
        "--knn-bio-mix-sigma-floor",
        type=float,
        default=CONFIG["knn_bio_mix_sigma_floor"],
    )
    parser.add_argument("--proxy-gene-bio-max", type=float, default=CONFIG["proxy_gene_bio_max"])
    parser.add_argument("--proxy-gene-drop-min", type=float, default=CONFIG["proxy_gene_drop_min"])
    parser.add_argument("--proxy-calib-mode", default=CONFIG["proxy_calib_mode"])
    parser.add_argument("--zero-iso-weight-grid", default="0.0")
    parser.add_argument("--dropout-iso-weight-grid", default="0.0")
    parser.add_argument("--constrained-zero-scale-grid", default="false")
    parser.add_argument("--renorm-imputed-grid", default="false")
    parser.add_argument("--p-bio-temp-grid", default="1.0")
    parser.add_argument("--p-bio-bias-grid", default="0.0")
    parser.add_argument("--thr-drop-grid", default="0.9")
    parser.add_argument("--p-bio-nz-scale-grid", default="0.0")
    parser.add_argument("--post-log-quantile-grid", default="0.0")
    parser.add_argument("--cell-scale-alpha-grid", default="0.0")
    parser.add_argument("--cell-scale-max", type=float, default=CONFIG["cell_scale_max"])
    parser.add_argument("--sf-dropout-alpha-grid", default="0.0")
    parser.add_argument("--sf-dropout-max-scale", type=float, default=CONFIG["sf_dropout_max_scale"])
    parser.add_argument("--proxy-calib-shrink-grid", default="0.0,0.3,0.6")
    parser.add_argument("--proxy-bio-weight-grid", default="0.0,0.5,1.0")
    parser.add_argument("--proxy-drop-weight-grid", default="0.0,0.5,1.0")
    parser.add_argument("--proxy-impute-alpha-grid", default="0.0,0.5,1.0")
    parser.add_argument("--proxy-impute-gamma-grid", default="1.0,2.0,4.0")
    parser.add_argument("--proxy-mean-mode-grid", default="knn,gene")
    parser.add_argument("--proxy-calib-min-points", type=int, default=20)
    parser.add_argument("--proxy-pu-weight-grid", default="0.0,0.5,1.0")
    parser.add_argument("--bio-soft-gamma-grid", default="0.0,2.0,4.0")
    parser.add_argument("--impute-mode-grid", default="gene,knn,ae")
    parser.add_argument("--impute-blend-alpha-grid", default="0.0,0.5,1.0")
    args = parser.parse_args()

    CONFIG["bio_model"] = str(args.bio_model)
    CONFIG["bio_mix_weight"] = float(args.bio_mix_weight)
    CONFIG["zinb_phi_min"] = float(args.zinb_phi_min)
    CONFIG["zinb_pi_max"] = float(args.zinb_pi_max)
    CONFIG["zinb_cluster_k"] = int(args.zinb_cluster_k)
    CONFIG["zinb_cluster_pca"] = int(args.zinb_cluster_pca)
    CONFIG["zinb_cluster_iters"] = int(args.zinb_cluster_iters)
    CONFIG["zinb_cluster_min_cells"] = int(args.zinb_cluster_min_cells)
    CONFIG["zinb_cluster_seed"] = int(args.zinb_cluster_seed)
    CONFIG["zinb_cluster_pi_tau"] = float(args.zinb_cluster_pi_tau)
    CONFIG["zinb_cluster_phi_tau"] = float(args.zinb_cluster_phi_tau)
    CONFIG["zinb_cluster_update_phi"] = bool(args.zinb_cluster_update_phi)
    CONFIG["knn_bio_mix_weight"] = float(args.knn_bio_mix_weight)
    CONFIG["knn_bio_mix_iters"] = int(args.knn_bio_mix_iters)
    CONFIG["knn_bio_mix_min_points"] = int(args.knn_bio_mix_min_points)
    CONFIG["knn_bio_mix_q_low"] = float(args.knn_bio_mix_q_low)
    CONFIG["knn_bio_mix_q_high"] = float(args.knn_bio_mix_q_high)
    CONFIG["knn_bio_mix_sigma_floor"] = float(args.knn_bio_mix_sigma_floor)
    CONFIG["cell_scale_max"] = float(args.cell_scale_max)
    CONFIG["sf_dropout_max_scale"] = float(args.sf_dropout_max_scale)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    seed = int(args.seed)
    zero_iso_weight_list = _parse_float_list(
        args.zero_iso_weight_grid, [float(CONFIG["zero_iso_weight"])]
    )
    dropout_iso_weight_list = _parse_float_list(
        args.dropout_iso_weight_grid, [float(CONFIG["dropout_iso_weight"])]
    )
    constrained_zero_scale_list = _parse_bool_list(
        args.constrained_zero_scale_grid, [bool(CONFIG["constrained_zero_scale"])]
    )
    renorm_imputed_list = _parse_bool_list(
        args.renorm_imputed_grid, [bool(CONFIG["renorm_imputed"])]
    )
    p_bio_temp_list = _parse_float_list(args.p_bio_temp_grid, [float(CONFIG["p_bio_temp"])])
    p_bio_bias_list = _parse_float_list(args.p_bio_bias_grid, [float(CONFIG["p_bio_bias"])])
    thr_drop_list = _parse_float_list(args.thr_drop_grid, [float(CONFIG["thr_drop"])])
    p_bio_nz_scale_list = _parse_float_list(args.p_bio_nz_scale_grid, [0.0])
    post_log_quantile_list = _parse_float_list(args.post_log_quantile_grid, [0.0])
    cell_scale_alpha_list = _parse_float_list(
        args.cell_scale_alpha_grid, [float(CONFIG["cell_scale_alpha"])]
    )
    sf_dropout_alpha_list = _parse_float_list(
        args.sf_dropout_alpha_grid, [float(CONFIG["sf_dropout_alpha"])]
    )
    proxy_calib_shrink_list = _parse_float_list(args.proxy_calib_shrink_grid, [0.0])
    proxy_bio_weight_list = _parse_float_list(
        args.proxy_bio_weight_grid, [float(CONFIG["proxy_bio_weight"])]
    )
    proxy_drop_weight_list = _parse_float_list(
        args.proxy_drop_weight_grid, [float(CONFIG["proxy_drop_weight"])]
    )
    proxy_impute_alpha_list = _parse_float_list(
        args.proxy_impute_alpha_grid, [float(CONFIG["proxy_impute_alpha"])]
    )
    proxy_impute_gamma_list = _parse_float_list(
        args.proxy_impute_gamma_grid, [float(CONFIG["proxy_impute_gamma"])]
    )
    proxy_mean_mode_list = _parse_str_list(
        args.proxy_mean_mode_grid, [str(CONFIG["proxy_mean_mode"])]
    )
    proxy_calib_min_points = int(args.proxy_calib_min_points)
    proxy_pu_weight_list = _parse_float_list(
        args.proxy_pu_weight_grid, [float(CONFIG["proxy_pu_weight"])]
    )
    bio_soft_gamma_list = _parse_float_list(
        args.bio_soft_gamma_grid, [float(CONFIG["bio_soft_gamma"])]
    )
    impute_mode_list = _parse_str_list(
        args.impute_mode_grid, [str(CONFIG["impute_mode"])]
    )
    impute_blend_alpha_list = _parse_float_list(
        args.impute_blend_alpha_grid, [float(CONFIG["impute_blend_alpha"])]
    )

    datasets: List[Dict[str, object]] = []
    for path in collect_rds_files(args.input_path):
        ds = prepare_dataset(
            path,
            knn_k=int(args.proxy_knn_k),
            knn_pca=int(args.proxy_knn_pca),
            proxy_gene_bio_max=float(args.proxy_gene_bio_max),
            proxy_gene_drop_min=float(args.proxy_gene_drop_min),
            proxy_knn_q_low=float(args.proxy_knn_q_low),
            proxy_knn_q_high=float(args.proxy_knn_q_high),
            proxy_knn_label_min_points=int(args.proxy_knn_label_min_points),
            proxy_knn_label=bool(args.proxy_knn_label),
            proxy_knn_ignore_zeros=bool(args.proxy_knn_ignore_zeros),
            proxy_knn_min_nz=int(args.proxy_knn_min_nz),
        )
        if ds is None:
            print(f"[WARN] {path.stem}: missing logTrueCounts; skipping.")
            continue
        datasets.append(ds)

    if not datasets:
        raise SystemExit("No datasets processed.")

    for ds in datasets:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        ds["device"] = device

        p_bio = compute_p_bio(ds, CONFIG)
        p_bio_proxy = fit_proxy_bio_from_counts(
            p_bio,
            ds,
            lr=float(CONFIG["proxy_pu_lr"]),
            epochs=int(CONFIG["proxy_pu_epochs"]),
            l2=float(CONFIG["proxy_pu_l2"]),
        )
        recon = train_autoencoder_reconstruct(
            logcounts=ds["logcounts"],
            counts_max=ds["counts_max"],
            p_bio=p_bio,
            device=device,
            hidden=CONFIG["hidden"],
            bottleneck=int(CONFIG["bottleneck"]),
            p_zero=float(CONFIG["p_zero"]),
            p_nz=float(CONFIG["p_nz"]),
            noise_min_frac=float(CONFIG["noise_min"]),
            noise_max_frac=float(CONFIG["noise_max"]),
            dropout=float(CONFIG["dropout"]),
            use_residual=bool(CONFIG["use_residual"]),
            epochs=int(CONFIG["epochs"]),
            batch_size=int(CONFIG["batch_size"]),
            lr=float(CONFIG["lr"]),
            weight_decay=float(CONFIG["weight_decay"]),
            loss_bio_weight=float(CONFIG["loss_bio_weight"]),
            loss_nz_weight=float(CONFIG["loss_nz_weight"]),
            bio_reg_weight=float(CONFIG["bio_reg_weight"]),
            recon_weight=float(CONFIG["recon_weight"]),
            p_low=float(CONFIG["p_low"]),
            p_high=float(CONFIG["p_high"]),
        )
        ds["p_bio_base"] = p_bio
        ds["p_bio_proxy"] = p_bio_proxy
        norm_counts_imputed = np.clip(logcounts_to_counts(recon), 0.0, None)
        norm_imputed = norm_counts_imputed / ds["gene_scale"][None, :]
        ds["norm_imputed_raw"] = np.nan_to_num(norm_imputed, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )
        ds["log_imputed_ae"] = (
            np.log1p(norm_counts_imputed) / np.log(2.0)
        ).astype(np.float32, copy=False)

    rows: List[Dict[str, object]] = []
    idx = 1
    for (
        zero_iso_weight,
        dropout_iso_weight,
        constrained_zero_scale,
        renorm_imputed,
        p_bio_temp,
        p_bio_bias,
        thr_drop,
        p_bio_nz_scale,
        post_log_quantile,
        cell_scale_alpha,
        sf_dropout_alpha,
        proxy_calib_shrink,
        proxy_bio_weight,
        proxy_drop_weight,
        proxy_impute_alpha,
        proxy_impute_gamma,
        proxy_mean_mode,
        proxy_pu_weight,
        bio_soft_gamma,
        impute_mode,
        impute_blend_alpha,
    ) in itertools.product(
        zero_iso_weight_list,
        dropout_iso_weight_list,
        constrained_zero_scale_list,
        renorm_imputed_list,
        p_bio_temp_list,
        p_bio_bias_list,
        thr_drop_list,
        p_bio_nz_scale_list,
        post_log_quantile_list,
        cell_scale_alpha_list,
        sf_dropout_alpha_list,
        proxy_calib_shrink_list,
        proxy_bio_weight_list,
        proxy_drop_weight_list,
        proxy_impute_alpha_list,
        proxy_impute_gamma_list,
        proxy_mean_mode_list,
        proxy_pu_weight_list,
        bio_soft_gamma_list,
        impute_mode_list,
        impute_blend_alpha_list,
    ):
        avg_mse, avg_bz, score = evaluate_proxy_grid(
            datasets,
            zero_iso_weight=zero_iso_weight,
            dropout_iso_weight=dropout_iso_weight,
            constrained_zero_scale=constrained_zero_scale,
            renorm_imputed=renorm_imputed,
            p_bio_temp=p_bio_temp,
            p_bio_bias=p_bio_bias,
            thr_drop=thr_drop,
            p_bio_nz_scale=p_bio_nz_scale,
            post_log_quantile=post_log_quantile,
            cell_scale_alpha=cell_scale_alpha,
            sf_dropout_alpha=sf_dropout_alpha,
            proxy_calib_shrink=proxy_calib_shrink,
            proxy_bio_weight=proxy_bio_weight,
            proxy_drop_weight=proxy_drop_weight,
            proxy_impute_alpha=proxy_impute_alpha,
            proxy_impute_gamma=proxy_impute_gamma,
            proxy_mean_mode=proxy_mean_mode,
            proxy_calib_min_points=proxy_calib_min_points,
            proxy_pu_weight=proxy_pu_weight,
            bio_soft_gamma=bio_soft_gamma,
            proxy_calib_mode=str(args.proxy_calib_mode),
            impute_mode=str(impute_mode),
            impute_blend_alpha=float(impute_blend_alpha),
        )
        rows.append(
            {
                "config_id": idx,
                "zero_iso_weight": zero_iso_weight,
                "dropout_iso_weight": dropout_iso_weight,
                "constrained_zero_scale": constrained_zero_scale,
                "renorm_imputed": renorm_imputed,
                "p_bio_temp": p_bio_temp,
                "p_bio_bias": p_bio_bias,
                "thr_drop": thr_drop,
                "p_bio_nz_scale": p_bio_nz_scale,
                "post_log_quantile": post_log_quantile,
                "cell_scale_alpha": cell_scale_alpha,
                "sf_dropout_alpha": sf_dropout_alpha,
                "proxy_calib_shrink": proxy_calib_shrink,
                "proxy_bio_weight": proxy_bio_weight,
                "proxy_drop_weight": proxy_drop_weight,
                "proxy_impute_alpha": proxy_impute_alpha,
                "proxy_impute_gamma": proxy_impute_gamma,
                "proxy_mean_mode": proxy_mean_mode,
                "proxy_calib_min_points": proxy_calib_min_points,
                "proxy_pu_weight": proxy_pu_weight,
                "bio_soft_gamma": bio_soft_gamma,
                "proxy_calib_mode": str(args.proxy_calib_mode),
                "impute_mode": str(impute_mode),
                "impute_blend_alpha": impute_blend_alpha,
                "avg_mse": avg_mse,
                "avg_bz_mse": avg_bz,
                "score": score,
            }
        )
        idx += 1

    rows.sort(key=lambda r: float(r["score"]))
    write_tuning_table(output_dir / "mask_impute15_tuning.tsv", rows)

    best = rows[0] if rows else None
    print("\n=== mask_impute15 ===")
    print("Fixed components: keep_positive=True, hard_zero_bio=True, zero_iso+dropout_iso+constrained_zero")
    if best is not None:
        metrics, avg_mse, avg_bz, score = evaluate_datasets(
            datasets,
            zero_iso_weight=float(best["zero_iso_weight"]),
            dropout_iso_weight=float(best["dropout_iso_weight"]),
            constrained_zero_scale=bool(best["constrained_zero_scale"]),
            renorm_imputed=bool(best.get("renorm_imputed", False)),
            p_bio_temp=float(best["p_bio_temp"]),
            p_bio_bias=float(best["p_bio_bias"]),
            thr_drop=float(best["thr_drop"]),
            p_bio_nz_scale=float(best["p_bio_nz_scale"]),
            post_log_quantile=float(best["post_log_quantile"]),
            cell_scale_alpha=float(best.get("cell_scale_alpha", 0.0)),
            sf_dropout_alpha=float(best.get("sf_dropout_alpha", 0.0)),
            proxy_calib_shrink=float(best["proxy_calib_shrink"]),
            proxy_bio_weight=float(best.get("proxy_bio_weight", 0.0)),
            proxy_drop_weight=float(best.get("proxy_drop_weight", 0.0)),
            proxy_impute_alpha=float(best.get("proxy_impute_alpha", 0.0)),
            proxy_impute_gamma=float(best.get("proxy_impute_gamma", 1.0)),
            proxy_mean_mode=str(best.get("proxy_mean_mode", "all")),
            proxy_calib_min_points=int(best.get("proxy_calib_min_points", 20)),
            proxy_pu_weight=float(best.get("proxy_pu_weight", 0.0)),
            bio_soft_gamma=float(best.get("bio_soft_gamma", 0.0)),
            proxy_calib_mode=str(best.get("proxy_calib_mode", "labeled")),
            impute_mode=str(best.get("impute_mode", CONFIG["impute_mode"])),
            impute_blend_alpha=float(best.get("impute_blend_alpha", CONFIG["impute_blend_alpha"])),
        )
        write_mse_table(output_dir / "mask_impute15_mse_table.tsv", metrics, avg_mse, avg_bz)
        print(
            "Best proxy config: "
            f"zero_iso={best['zero_iso_weight']} "
            f"dropout_iso={best['dropout_iso_weight']} constrained={best['constrained_zero_scale']} "
            f"renorm={best.get('renorm_imputed', False)} "
            f"p_bio={best['p_bio_temp']}/{best['p_bio_bias']} "
            f"thr_drop={best['thr_drop']} nz_scale={best['p_bio_nz_scale']} "
            f"q={best['post_log_quantile']} cell_scale={best.get('cell_scale_alpha', 0.0)} "
            f"sf_drop={best.get('sf_dropout_alpha', 0.0)} "
            f"cal_shrink={best['proxy_calib_shrink']} "
            f"proxy_bio={best.get('proxy_bio_weight', 0.0)} "
            f"proxy_drop={best.get('proxy_drop_weight', 0.0)} "
            f"impute_alpha={best.get('proxy_impute_alpha', 0.0)} "
            f"impute_gamma={best.get('proxy_impute_gamma', 1.0)} "
            f"mean_mode={best.get('proxy_mean_mode', 'all')} "
            f"pu_weight={best.get('proxy_pu_weight', 0.0)} "
            f"bio_soft={best.get('bio_soft_gamma', 0.0)} "
            f"calib_mode={best.get('proxy_calib_mode', 'labeled')} | "
            f"impute_mode={best.get('impute_mode', CONFIG['impute_mode'])} "
            f"blend_alpha={best.get('impute_blend_alpha', CONFIG['impute_blend_alpha'])} | "
            f"avg_mse={avg_mse:.6f} "
            f"avg_bz={avg_bz:.6f} score={score:.6f}"
        )


if __name__ == "__main__":
    main()
