#!/usr/bin/env python3
"""
mask_impute15.py

Mask imputation using RMSE on log-normalized counts with proxy calibration.
Uses the SPLAT Dropout/DropProb assays for supervised biozero calibration and
per-gene linear calibration on non-dropout entries (no TrueCounts for calibration).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predict_dropouts_new import splatter_bio_posterior_from_counts
from rds2py import read_rds

EPSILON = 1e-6
GENE_NORM_LOW = 5.0
GENE_NORM_HIGH = 95.0
LAMBDA_RMSE = 0.5
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
    "constrained_zero_max_rmse_inc": 0.1,
    "constrained_zero_lambda_max": 1000.0,
    "constrained_zero_iters": 30,
    "cell_zero_weight": 0.6,
    "proxy_mix": 0.5,
    "proxy_drop_gamma": 1.0,
    "proxy_drop_scale": 1.0,
    "proxy_bio_weight": 1.0,
    "proxy_impute_alpha": 1.0,
    "proxy_impute_gamma": 2.0,
    "proxy_mean_mode": "all",
    "proxy_calib_min_points": 20,
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

    dropout = None
    try:
        dropout = sce.assay("Dropout").T.astype("float32")
        dropout = dropout[:, keep]
    except Exception:
        dropout = None

    drop_prob = None
    try:
        drop_prob = sce.assay("DropProb").T.astype("float32")
        drop_prob = drop_prob[:, keep]
    except Exception:
        drop_prob = None

    return {
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts,
        "dropout": dropout,
        "drop_prob": drop_prob,
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


def _rmse_from_diff(diff: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_rmse_metrics(
    pred_log: np.ndarray,
    true_log: np.ndarray,
    counts_obs: np.ndarray,
) -> Dict[str, float]:
    diff = true_log - pred_log
    mask_biozero = true_log <= EPSILON
    mask_dropout = (true_log > EPSILON) & (counts_obs <= EPSILON)
    mask_non_zero = (true_log > EPSILON) & (counts_obs > EPSILON)
    return {
        "rmse": _rmse_from_diff(diff),
        "rmse_dropout": _rmse_from_diff(diff, mask_dropout),
        "rmse_biozero": _rmse_from_diff(diff, mask_biozero),
        "rmse_non_zero": _rmse_from_diff(diff, mask_non_zero),
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
    weight: float,
) -> np.ndarray:
    if proxy_bio_mask is None:
        return p_bio
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return p_bio
    p_out = p_bio.astype(np.float32, copy=True)
    boost_mask = proxy_bio_mask & zeros_obs
    if np.any(boost_mask):
        p_sel = p_out[boost_mask]
        p_out[boost_mask] = p_sel + weight * (1.0 - p_sel)
    return p_out


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


def prepare_dataset(path: Path) -> Dict[str, object] | None:
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

    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, GENE_NORM_LOW))
    cz_hi = float(np.percentile(cell_zero_frac, GENE_NORM_HIGH))
    cz_span = max(cz_hi - cz_lo, EPSILON)
    cell_zero_norm = np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)

    drop_arr = dataset.get("dropout")
    drop_prob = dataset.get("drop_prob")
    dropout_mask = None
    proxy_non_dropout = None
    proxy_dropout_mask = None
    proxy_bio_mask = None
    proxy_log_mean_all = None
    proxy_log_mean_nz = None
    proxy_norm_mean_all = None
    proxy_norm_mean_nz = None
    if drop_arr is not None:
        dropout_mask = drop_arr > 0.5
        proxy_non_dropout = ~dropout_mask
        proxy_dropout_mask = zeros_obs & dropout_mask
        proxy_bio_mask = zeros_obs & (~dropout_mask)
        n_genes = logcounts.shape[1]
        proxy_log_mean_all = np.zeros((n_genes,), dtype=np.float32)
        proxy_log_mean_nz = np.zeros((n_genes,), dtype=np.float32)
        proxy_norm_mean_all = np.zeros((n_genes,), dtype=np.float32)
        proxy_norm_mean_nz = np.zeros((n_genes,), dtype=np.float32)
        for j in range(n_genes):
            nd = proxy_non_dropout[:, j]
            if np.any(nd):
                vals_log = logcounts[nd, j]
                vals_norm = norm_counts[nd, j]
                proxy_log_mean_all[j] = float(np.mean(vals_log))
                proxy_norm_mean_all[j] = float(np.mean(vals_norm))
                nd_nz = nd & (logcounts[:, j] > 0.0)
                if np.any(nd_nz):
                    proxy_log_mean_nz[j] = float(np.mean(logcounts[nd_nz, j]))
                    proxy_norm_mean_nz[j] = float(np.mean(norm_counts[nd_nz, j]))
                else:
                    proxy_log_mean_nz[j] = proxy_log_mean_all[j]
                    proxy_norm_mean_nz[j] = proxy_norm_mean_all[j]

    return {
        "dataset": path.stem,
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts_obs,
        "zeros_obs": zeros_obs,
        "counts_max": counts_max,
        "lib_size": lib_size,
        "norm_counts": norm_counts,
        "norm_counts_raw": norm_counts_raw,
        "gene_scale": gene_scale,
        "gene_mean": gene_mean,
        "gene_mean_nz": gene_mean_nz,
        "drop_k": drop_k,
        "drop_x0": drop_x0,
        "p_drop_gene": p_drop_gene,
        "cell_zero_norm": cell_zero_norm,
        "dropout": dropout_mask,
        "drop_prob": drop_prob,
        "proxy_non_dropout": proxy_non_dropout,
        "proxy_dropout_mask": proxy_dropout_mask,
        "proxy_bio_mask": proxy_bio_mask,
        "proxy_log_mean_all": proxy_log_mean_all,
        "proxy_log_mean_nz": proxy_log_mean_nz,
        "proxy_norm_mean_all": proxy_norm_mean_all,
        "proxy_norm_mean_nz": proxy_norm_mean_nz,
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
    p_bio = splat_cellaware_bio_prob(
        counts=ds["counts"],
        zeros_obs=zeros_obs,
        disp_mode=str(cfg["disp_mode"]),
        disp_const=float(cfg["disp_const"]),
        use_cell_factor=bool(cfg["use_cell_factor"]),
        tau_dispersion=float(cfg["tau_dispersion"]),
        tau_group_dispersion=float(cfg["tau_group_dispersion"]),
        tau_dropout=float(cfg["tau_dropout"]),
    )
    if float(cfg["cell_zero_weight"]) > 0.0:
        cell_w = np.clip(float(cfg["cell_zero_weight"]) * ds["cell_zero_norm"], 0.0, 1.0)
        p_bio = p_bio * (1.0 - cell_w[:, None])
    return np.clip(p_bio, 0.0, 1.0).astype(np.float32, copy=False)


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
    p_bio: np.ndarray,
    proxy_mix: float,
    proxy_drop_gamma: float,
    proxy_drop_scale: float,
    proxy_mean_mode: str,
) -> np.ndarray:
    zeros_obs = ds["zeros_obs"]
    proxy_non_dropout = ds.get("proxy_non_dropout")
    proxy_dropout_mask = ds.get("proxy_dropout_mask")
    proxy_bio_mask = ds.get("proxy_bio_mask")
    if proxy_non_dropout is not None and proxy_dropout_mask is not None:
        proxy_target = ds["norm_counts"].copy()
        if str(proxy_mean_mode).lower() == "nz":
            proxy_mean = ds.get("proxy_norm_mean_nz")
        else:
            proxy_mean = ds.get("proxy_norm_mean_all")
        if proxy_mean is not None:
            for j in range(proxy_target.shape[1]):
                mask = proxy_dropout_mask[:, j]
                if np.any(mask):
                    proxy_target[mask, j] = float(proxy_mean[j])
        if proxy_bio_mask is not None and np.any(proxy_bio_mask):
            proxy_target[proxy_bio_mask] = 0.0
        return np.nan_to_num(proxy_target, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )

    mean_all = ds["gene_mean"]
    mean_nz = ds["gene_mean_nz"]
    mix = float(proxy_mix)
    mix = np.clip(mix, 0.0, 1.0)
    proxy_mean = (1.0 - mix) * mean_all + mix * mean_nz
    p_drop = np.clip(ds["p_drop_gene"], 0.0, 1.0) ** float(proxy_drop_gamma)
    p_drop = np.clip(p_drop * float(proxy_drop_scale), 0.0, 1.0)
    proxy_zero = proxy_mean[None, :] * p_drop[None, :] * (1.0 - p_bio)
    proxy_target = ds["norm_counts"].copy()
    proxy_target[zeros_obs] = proxy_zero[zeros_obs]
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
    max_rmse_inc: float,
    lambda_max: float,
    iters: int,
) -> Tuple[np.ndarray, float]:
    diff_base = target - pred
    rmse_base = _rmse_from_diff(diff_base)
    rmse_target = rmse_base * (1.0 + float(max_rmse_inc))
    if rmse_target <= 0.0:
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
        rmse = _rmse_from_diff(target - log_adj)
        return rmse, log_adj

    rmse_low, log_low = _eval_lambda(0.0)
    if rmse_low <= rmse_target:
        return log_low, 0.0

    lam_high = float(lambda_max)
    rmse_high, log_high = _eval_lambda(lam_high)
    if rmse_high > rmse_target:
        return log_high, lam_high

    lo = 0.0
    hi = lam_high
    best_log = log_high
    best_lam = lam_high
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        rmse_mid, log_mid = _eval_lambda(mid)
        if rmse_mid <= rmse_target:
            hi = mid
            best_log = log_mid
            best_lam = mid
        else:
            lo = mid
    return best_log, best_lam


def _score(avg_rmse: float, avg_bz_rmse: float) -> float:
    return float(avg_bz_rmse) + float(LAMBDA_RMSE) * float(avg_rmse)


def run_pipeline(
    ds: Dict[str, object],
    proxy_mix: float,
    proxy_drop_gamma: float,
    proxy_drop_scale: float,
    zero_iso_weight: float,
    dropout_iso_weight: float,
    constrained_zero_scale: bool,
    p_bio_temp: float,
    p_bio_bias: float,
    thr_drop: float,
    p_bio_nz_scale: float,
    post_log_quantile: float,
    proxy_calib_shrink: float,
    proxy_bio_weight: float,
    proxy_impute_alpha: float,
    proxy_impute_gamma: float,
    proxy_mean_mode: str,
    proxy_calib_min_points: int,
) -> Dict[str, float]:
    p_bio_use = _logit_scale_probs(ds["p_bio_base"], temp=p_bio_temp, bias=p_bio_bias)
    p_bio_use = calibrate_p_bio_with_proxy(
        p_bio_use,
        zeros_obs=ds["zeros_obs"],
        proxy_bio_mask=ds.get("proxy_bio_mask"),
        weight=proxy_bio_weight,
    )
    nz_scale = float(np.clip(p_bio_nz_scale, 0.0, 1.0))
    if nz_scale < 1.0:
        nz_mask = ~ds["zeros_obs"]
        if np.any(nz_mask):
            p_bio_use = p_bio_use.copy()
            p_bio_use[nz_mask] = p_bio_use[nz_mask] * nz_scale
    proxy_target = _build_proxy_target(
        ds=ds,
        p_bio=p_bio_use,
        proxy_mix=proxy_mix,
        proxy_drop_gamma=proxy_drop_gamma,
        proxy_drop_scale=proxy_drop_scale,
        proxy_mean_mode=proxy_mean_mode,
    )

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
    p_bio_mix[~ds["zeros_obs"]] = 0.0

    if float(zero_iso_weight) > 0.0:
        norm_imputed_final, _ = _apply_zero_iso_scale(
            pred=norm_imputed_final,
            target=proxy_target,
            zeros_obs=ds["zeros_obs"],
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
            zeros_obs=ds["zeros_obs"],
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
            zeros_obs=ds["zeros_obs"],
            max_rmse_inc=float(CONFIG["constrained_zero_max_rmse_inc"]),
            lambda_max=float(CONFIG["constrained_zero_lambda_max"]),
            iters=int(CONFIG["constrained_zero_iters"]),
        )

    norm10k_imputed = norm_imputed_final * ds["gene_scale"][None, :]
    norm10k_imputed = np.clip(norm10k_imputed, 0.0, None)
    log_imputed = np.log1p(norm10k_imputed) / np.log(2.0)
    proxy_dropout_mask = ds.get("proxy_dropout_mask")
    if proxy_dropout_mask is not None and float(proxy_impute_alpha) > 0.0:
        if str(proxy_mean_mode).lower() == "nz":
            proxy_mean = ds.get("proxy_log_mean_nz")
        else:
            proxy_mean = ds.get("proxy_log_mean_all")
        if proxy_mean is not None:
            alpha = float(np.clip(proxy_impute_alpha, 0.0, 1.0))
            gamma = float(max(proxy_impute_gamma, 0.0))
            p_scale = np.clip(1.0 - p_bio_use, 0.0, 1.0)
            if gamma != 1.0:
                p_scale = p_scale ** gamma
            log_imputed = log_imputed.copy()
            for j in range(log_imputed.shape[1]):
                mask = proxy_dropout_mask[:, j]
                if np.any(mask):
                    mean_val = float(proxy_mean[j])
                    log_imputed[mask, j] = (1.0 - alpha) * log_imputed[mask, j] + alpha * (
                        mean_val * p_scale[mask, j]
                    )
    if float(proxy_calib_shrink) > 0.0:
        proxy_log = ds["logcounts"]
        proxy_mask = ds.get("proxy_non_dropout")
        log_imputed = calibrate_log_to_proxy(
            log_imputed,
            proxy_log,
            shrink=proxy_calib_shrink,
            mask=proxy_mask,
            min_points=proxy_calib_min_points,
        )
    if float(post_log_quantile) > 0.0:
        thresh = _get_gene_log_threshold(ds, float(post_log_quantile))
        log_imputed = log_imputed.copy()
        log_imputed[log_imputed < thresh[None, :]] = 0.0

    return compute_rmse_metrics(
        log_imputed,
        ds["log_true"],
        ds["counts"],
    )


def evaluate_proxy_grid(
    datasets: List[Dict[str, object]],
    proxy_mix: float,
    proxy_drop_gamma: float,
    proxy_drop_scale: float,
    zero_iso_weight: float,
    dropout_iso_weight: float,
    constrained_zero_scale: bool,
    p_bio_temp: float,
    p_bio_bias: float,
    thr_drop: float,
    p_bio_nz_scale: float,
    post_log_quantile: float,
    proxy_calib_shrink: float,
    proxy_bio_weight: float,
    proxy_impute_alpha: float,
    proxy_impute_gamma: float,
    proxy_mean_mode: str,
    proxy_calib_min_points: int,
) -> Tuple[float, float, float]:
    rmse_list: List[float] = []
    bz_list: List[float] = []
    for ds in datasets:
        metrics = run_pipeline(
            ds=ds,
            proxy_mix=proxy_mix,
            proxy_drop_gamma=proxy_drop_gamma,
            proxy_drop_scale=proxy_drop_scale,
            zero_iso_weight=zero_iso_weight,
            dropout_iso_weight=dropout_iso_weight,
            constrained_zero_scale=constrained_zero_scale,
            p_bio_temp=p_bio_temp,
            p_bio_bias=p_bio_bias,
            thr_drop=thr_drop,
            p_bio_nz_scale=p_bio_nz_scale,
            post_log_quantile=post_log_quantile,
            proxy_calib_shrink=proxy_calib_shrink,
            proxy_bio_weight=proxy_bio_weight,
            proxy_impute_alpha=proxy_impute_alpha,
            proxy_impute_gamma=proxy_impute_gamma,
            proxy_mean_mode=proxy_mean_mode,
            proxy_calib_min_points=proxy_calib_min_points,
        )
        rmse_list.append(float(metrics["rmse"]))
        bz_list.append(float(metrics["rmse_biozero"]))
    avg_rmse = float(np.nanmean(rmse_list)) if rmse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = _score(avg_rmse, avg_bz)
    return avg_rmse, avg_bz, score


def evaluate_datasets(
    datasets: List[Dict[str, object]],
    proxy_mix: float,
    proxy_drop_gamma: float,
    proxy_drop_scale: float,
    zero_iso_weight: float,
    dropout_iso_weight: float,
    constrained_zero_scale: bool,
    p_bio_temp: float,
    p_bio_bias: float,
    thr_drop: float,
    p_bio_nz_scale: float,
    post_log_quantile: float,
    proxy_calib_shrink: float,
    proxy_bio_weight: float,
    proxy_impute_alpha: float,
    proxy_impute_gamma: float,
    proxy_mean_mode: str,
    proxy_calib_min_points: int,
) -> Tuple[List[Dict[str, float]], float, float, float]:
    metrics_all: List[Dict[str, float]] = []
    rmse_list: List[float] = []
    bz_list: List[float] = []
    for ds in datasets:
        metrics = run_pipeline(
            ds=ds,
            proxy_mix=proxy_mix,
            proxy_drop_gamma=proxy_drop_gamma,
            proxy_drop_scale=proxy_drop_scale,
            zero_iso_weight=zero_iso_weight,
            dropout_iso_weight=dropout_iso_weight,
            constrained_zero_scale=constrained_zero_scale,
            p_bio_temp=p_bio_temp,
            p_bio_bias=p_bio_bias,
            thr_drop=thr_drop,
            p_bio_nz_scale=p_bio_nz_scale,
            post_log_quantile=post_log_quantile,
            proxy_calib_shrink=proxy_calib_shrink,
            proxy_bio_weight=proxy_bio_weight,
            proxy_impute_alpha=proxy_impute_alpha,
            proxy_impute_gamma=proxy_impute_gamma,
            proxy_mean_mode=proxy_mean_mode,
            proxy_calib_min_points=proxy_calib_min_points,
        )
        metrics = dict(metrics)
        metrics["dataset"] = str(ds["dataset"])
        metrics_all.append(metrics)
        rmse_list.append(float(metrics["rmse"]))
        bz_list.append(float(metrics["rmse_biozero"]))
    avg_rmse = float(np.nanmean(rmse_list)) if rmse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = _score(avg_rmse, avg_bz)
    return metrics_all, avg_rmse, avg_bz, score


def write_rmse_table(path: Path, metrics: List[Dict[str, float]], avg_rmse: float, avg_bz: float) -> None:
    header = [
        "dataset",
        "rmse",
        "rmse_dropout",
        "rmse_biozero",
        "rmse_non_zero",
        "n_total",
        "n_dropout",
        "n_biozero",
        "n_non_zero",
    ]
    lines = ["\t".join(header)]
    for row in metrics:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    lines.append("\t".join(["AVG", f"{avg_rmse}", "", f"{avg_bz}", "", "", "", "", ""]))
    path.write_text("\n".join(lines) + "\n")


def write_tuning_table(path: Path, rows: List[Dict[str, object]]) -> None:
    header = [
        "config_id",
        "proxy_mix",
        "proxy_drop_gamma",
        "proxy_drop_scale",
        "zero_iso_weight",
        "dropout_iso_weight",
        "constrained_zero_scale",
        "p_bio_temp",
        "p_bio_bias",
        "thr_drop",
        "p_bio_nz_scale",
        "post_log_quantile",
        "proxy_calib_shrink",
        "proxy_bio_weight",
        "proxy_impute_alpha",
        "proxy_impute_gamma",
        "proxy_mean_mode",
        "proxy_calib_min_points",
        "avg_rmse",
        "avg_bz_rmse",
        "score",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune proxy calibration parameters for RMSE on normalized counts "
            "using SPLAT-style dropout estimates (no TrueCounts for calibration)."
        )
    )
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proxy-mix-grid", default="0.5")
    parser.add_argument("--proxy-drop-gamma-grid", default="1.0")
    parser.add_argument("--proxy-drop-scale-grid", default="1.0")
    parser.add_argument("--zero-iso-weight-grid", default="0.0")
    parser.add_argument("--dropout-iso-weight-grid", default="0.0")
    parser.add_argument("--constrained-zero-scale-grid", default="false")
    parser.add_argument("--p-bio-temp-grid", default="1.0")
    parser.add_argument("--p-bio-bias-grid", default="0.0")
    parser.add_argument("--thr-drop-grid", default="0.9")
    parser.add_argument("--p-bio-nz-scale-grid", default="0.0")
    parser.add_argument("--post-log-quantile-grid", default="0.0")
    parser.add_argument("--proxy-calib-shrink-grid", default="0.0,0.3,0.6")
    parser.add_argument("--proxy-bio-weight-grid", default="0.0,0.5,1.0")
    parser.add_argument("--proxy-impute-alpha-grid", default="0.0,0.5,1.0")
    parser.add_argument("--proxy-impute-gamma-grid", default="1.0,2.0,4.0")
    parser.add_argument("--proxy-mean-mode-grid", default="all,nz")
    parser.add_argument("--proxy-calib-min-points", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    seed = int(args.seed)
    proxy_mix_list = _parse_float_list(args.proxy_mix_grid, [float(CONFIG["proxy_mix"])])
    proxy_drop_gamma_list = _parse_float_list(
        args.proxy_drop_gamma_grid, [float(CONFIG["proxy_drop_gamma"])]
    )
    proxy_drop_scale_list = _parse_float_list(
        args.proxy_drop_scale_grid, [float(CONFIG["proxy_drop_scale"])]
    )
    zero_iso_weight_list = _parse_float_list(
        args.zero_iso_weight_grid, [float(CONFIG["zero_iso_weight"])]
    )
    dropout_iso_weight_list = _parse_float_list(
        args.dropout_iso_weight_grid, [float(CONFIG["dropout_iso_weight"])]
    )
    constrained_zero_scale_list = _parse_bool_list(
        args.constrained_zero_scale_grid, [bool(CONFIG["constrained_zero_scale"])]
    )
    p_bio_temp_list = _parse_float_list(args.p_bio_temp_grid, [float(CONFIG["p_bio_temp"])])
    p_bio_bias_list = _parse_float_list(args.p_bio_bias_grid, [float(CONFIG["p_bio_bias"])])
    thr_drop_list = _parse_float_list(args.thr_drop_grid, [float(CONFIG["thr_drop"])])
    p_bio_nz_scale_list = _parse_float_list(args.p_bio_nz_scale_grid, [0.0])
    post_log_quantile_list = _parse_float_list(args.post_log_quantile_grid, [0.0])
    proxy_calib_shrink_list = _parse_float_list(args.proxy_calib_shrink_grid, [0.0])
    proxy_bio_weight_list = _parse_float_list(
        args.proxy_bio_weight_grid, [float(CONFIG["proxy_bio_weight"])]
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

    datasets: List[Dict[str, object]] = []
    for path in collect_rds_files(args.input_path):
        ds = prepare_dataset(path)
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

        p_bio = compute_p_bio(ds, CONFIG)
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
        norm_counts_imputed = np.clip(logcounts_to_counts(recon), 0.0, None)
        norm_imputed = norm_counts_imputed / ds["gene_scale"][None, :]
        ds["norm_imputed_raw"] = np.nan_to_num(norm_imputed, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )

    rows: List[Dict[str, object]] = []
    idx = 1
    for (
        proxy_mix,
        proxy_drop_gamma,
        proxy_drop_scale,
        zero_iso_weight,
        dropout_iso_weight,
        constrained_zero_scale,
        p_bio_temp,
        p_bio_bias,
        thr_drop,
        p_bio_nz_scale,
        post_log_quantile,
        proxy_calib_shrink,
        proxy_bio_weight,
        proxy_impute_alpha,
        proxy_impute_gamma,
        proxy_mean_mode,
    ) in itertools.product(
        proxy_mix_list,
        proxy_drop_gamma_list,
        proxy_drop_scale_list,
        zero_iso_weight_list,
        dropout_iso_weight_list,
        constrained_zero_scale_list,
        p_bio_temp_list,
        p_bio_bias_list,
        thr_drop_list,
        p_bio_nz_scale_list,
        post_log_quantile_list,
        proxy_calib_shrink_list,
        proxy_bio_weight_list,
        proxy_impute_alpha_list,
        proxy_impute_gamma_list,
        proxy_mean_mode_list,
    ):
        avg_rmse, avg_bz, score = evaluate_proxy_grid(
            datasets,
            proxy_mix=proxy_mix,
            proxy_drop_gamma=proxy_drop_gamma,
            proxy_drop_scale=proxy_drop_scale,
            zero_iso_weight=zero_iso_weight,
            dropout_iso_weight=dropout_iso_weight,
            constrained_zero_scale=constrained_zero_scale,
            p_bio_temp=p_bio_temp,
            p_bio_bias=p_bio_bias,
            thr_drop=thr_drop,
            p_bio_nz_scale=p_bio_nz_scale,
            post_log_quantile=post_log_quantile,
            proxy_calib_shrink=proxy_calib_shrink,
            proxy_bio_weight=proxy_bio_weight,
            proxy_impute_alpha=proxy_impute_alpha,
            proxy_impute_gamma=proxy_impute_gamma,
            proxy_mean_mode=proxy_mean_mode,
            proxy_calib_min_points=proxy_calib_min_points,
        )
        rows.append(
            {
                "config_id": idx,
                "proxy_mix": proxy_mix,
                "proxy_drop_gamma": proxy_drop_gamma,
                "proxy_drop_scale": proxy_drop_scale,
                "zero_iso_weight": zero_iso_weight,
                "dropout_iso_weight": dropout_iso_weight,
                "constrained_zero_scale": constrained_zero_scale,
                "p_bio_temp": p_bio_temp,
                "p_bio_bias": p_bio_bias,
                "thr_drop": thr_drop,
                "p_bio_nz_scale": p_bio_nz_scale,
                "post_log_quantile": post_log_quantile,
                "proxy_calib_shrink": proxy_calib_shrink,
                "proxy_bio_weight": proxy_bio_weight,
                "proxy_impute_alpha": proxy_impute_alpha,
                "proxy_impute_gamma": proxy_impute_gamma,
                "proxy_mean_mode": proxy_mean_mode,
                "proxy_calib_min_points": proxy_calib_min_points,
                "avg_rmse": avg_rmse,
                "avg_bz_rmse": avg_bz,
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
        metrics, avg_rmse, avg_bz, score = evaluate_datasets(
            datasets,
            proxy_mix=float(best["proxy_mix"]),
            proxy_drop_gamma=float(best["proxy_drop_gamma"]),
            proxy_drop_scale=float(best["proxy_drop_scale"]),
            zero_iso_weight=float(best["zero_iso_weight"]),
            dropout_iso_weight=float(best["dropout_iso_weight"]),
            constrained_zero_scale=bool(best["constrained_zero_scale"]),
            p_bio_temp=float(best["p_bio_temp"]),
            p_bio_bias=float(best["p_bio_bias"]),
            thr_drop=float(best["thr_drop"]),
            p_bio_nz_scale=float(best["p_bio_nz_scale"]),
            post_log_quantile=float(best["post_log_quantile"]),
            proxy_calib_shrink=float(best["proxy_calib_shrink"]),
            proxy_bio_weight=float(best.get("proxy_bio_weight", 0.0)),
            proxy_impute_alpha=float(best.get("proxy_impute_alpha", 0.0)),
            proxy_impute_gamma=float(best.get("proxy_impute_gamma", 1.0)),
            proxy_mean_mode=str(best.get("proxy_mean_mode", "all")),
            proxy_calib_min_points=int(best.get("proxy_calib_min_points", 20)),
        )
        write_rmse_table(output_dir / "mask_impute15_rmse_table.tsv", metrics, avg_rmse, avg_bz)
        print(
            "Best proxy config: "
            f"mix={best['proxy_mix']} drop_gamma={best['proxy_drop_gamma']} "
            f"drop_scale={best['proxy_drop_scale']} zero_iso={best['zero_iso_weight']} "
            f"dropout_iso={best['dropout_iso_weight']} constrained={best['constrained_zero_scale']} "
            f"p_bio={best['p_bio_temp']}/{best['p_bio_bias']} "
            f"thr_drop={best['thr_drop']} nz_scale={best['p_bio_nz_scale']} "
            f"q={best['post_log_quantile']} cal_shrink={best['proxy_calib_shrink']} "
            f"proxy_bio={best.get('proxy_bio_weight', 0.0)} "
            f"impute_alpha={best.get('proxy_impute_alpha', 0.0)} "
            f"impute_gamma={best.get('proxy_impute_gamma', 1.0)} "
            f"mean_mode={best.get('proxy_mean_mode', 'all')} | "
            f"avg_rmse={avg_rmse:.6f} "
            f"avg_bz={avg_bz:.6f} score={score:.6f}"
        )


if __name__ == "__main__":
    main()
