#!/usr/bin/env python3
"""
Shared implementation for masked imputation option experiments.

Options implemented:
1. Dropout head with synthetic supervision.
2. EM-style zero posterior update (Poisson/NB proxy).
3. Two-expert decoder with learned dropout gate.
4. Neighbor-informed posterior correction.
5. Gene-specific posterior calibration.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predict_dropouts_new import splatter_bio_posterior_from_counts

_REPO_ROOT = Path(__file__).resolve().parent
_SYS_PATH = list(sys.path)
try:
    sys.path = [
        p for p in sys.path if str(_REPO_ROOT) not in p and "MaskedImpute/rds2py" not in p
    ]
    from rds2py import read_rds
finally:
    sys.path = _SYS_PATH

EPSILON = 1e-6

BIO_PARAMS_BASE = {
    "disp_mode": "estimate",
    "use_cell_factor": True,
    "cell_zero_weight": 0.3,
}

MODEL_PARAMS_BASE = {
    "hidden": [64],
    "bottleneck": 32,
    "batch_size": 32,
    "weight_decay": 0.0,
}

AE_PARAMS_BASE = {
    "epochs": 220,
    "lr": 0.0001,
    "p_zero": 0.01,
    "p_nz": 0.35,
    "noise_max": 0.2,
    "loss_bio_weight": 2.0,
    "loss_nz_weight": 1.0,
    "bio_reg_weight": 1.0,
}

REFINE_PARAMS_BASE = {
    "enabled": True,
    "start_epoch": 80,
    "every_n_epochs": 20,
    "ema_alpha": 0.2,
    "prior_blend": 0.8,
    "dropout_midpoint": 0.08,
    "dropout_scale": 0.30,
    "update_min_dropout_prob": 0.95,
    "force_dropout_prob": 0.999,
    "prior_floor_ratio": 0.2,
    "p_bio_min": 1e-4,
    "p_bio_max": 0.999,
}

SCALER_PARAMS = {
    "p_low": 2.0,
    "p_high": 99.5,
}

OPTION_DEFAULTS = {
    1: {
        "ae": {
            "p_nz": 0.45,
            "noise_max": 0.25,
            "loss_bio_weight": 2.0,
            "bio_reg_weight": 1.0,
        },
        "refine": {
            "start_epoch": 60,
            "every_n_epochs": 15,
            "ema_alpha": 0.25,
            "prior_blend": 0.65,
            "update_min_dropout_prob": 0.85,
        },
        "extra": {
            "gate_loss_weight": 0.6,
            "gate_zero_soft_weight": 0.3,
        },
    },
    2: {
        "ae": {
            "p_nz": 0.4,
            "noise_max": 0.2,
            "loss_bio_weight": 2.5,
            "bio_reg_weight": 1.2,
        },
        "refine": {
            "start_epoch": 80,
            "every_n_epochs": 20,
            "ema_alpha": 0.2,
            "prior_blend": 0.7,
            "update_min_dropout_prob": 0.9,
        },
        "extra": {},
    },
    3: {
        "ae": {
            "p_nz": 0.45,
            "noise_max": 0.25,
            "loss_bio_weight": 2.0,
            "bio_reg_weight": 1.0,
        },
        "refine": {
            "start_epoch": 70,
            "every_n_epochs": 15,
            "ema_alpha": 0.25,
            "prior_blend": 0.7,
            "update_min_dropout_prob": 0.85,
        },
        "extra": {
            "gate_loss_weight": 0.5,
            "expert_bio_weight": 0.4,
            "expert_drop_weight": 0.2,
            "gate_zero_soft_weight": 0.3,
        },
    },
    4: {
        "ae": {
            "p_nz": 0.5,
            "noise_max": 0.25,
            "loss_bio_weight": 1.8,
            "bio_reg_weight": 0.9,
        },
        "refine": {
            "start_epoch": 70,
            "every_n_epochs": 15,
            "ema_alpha": 0.3,
            "prior_blend": 0.6,
            "update_min_dropout_prob": 0.85,
        },
        "extra": {
            "knn_k": 8,
            "knn_recon_weight": 0.55,
            "knn_neighbor_weight": 0.45,
        },
    },
    5: {
        "ae": {
            "p_nz": 0.45,
            "noise_max": 0.2,
            "loss_bio_weight": 2.0,
            "bio_reg_weight": 1.0,
        },
        "refine": {
            "start_epoch": 70,
            "every_n_epochs": 15,
            "ema_alpha": 0.25,
            "prior_blend": 0.6,
            "update_min_dropout_prob": 0.85,
        },
        "extra": {
            "gene_shrink": 0.4,
        },
    },
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

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lo_, self.hi_)

    def fit(self, x: np.ndarray):
        self.lo_ = np.percentile(x, self.p_low, axis=0)
        self.hi_ = np.percentile(x, self.p_high, axis=0)
        xc = self._clip(x)
        self.mean_ = xc.mean(axis=0)
        self.std_ = xc.std(axis=0)
        self.std_[self.std_ < self.eps] = 1.0
        z = (xc - self.mean_) / self.std_
        self.zmin_ = z.min(axis=0)
        self.zmax_ = z.max(axis=0)
        self.zspan_ = self.zmax_ - self.zmin_
        self.zspan_[self.zspan_ < self.eps] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        xc = self._clip(x)
        z = (xc - self.mean_) / self.std_
        x01 = (z - self.zmin_) / self.zspan_
        xscaled = x01 * 2.0 - 1.0
        return xscaled.astype(np.float32)

    def inverse_transform(self, xscaled: np.ndarray) -> np.ndarray:
        x01 = (xscaled + 1.0) / 2.0
        z = x01 * self.zspan_ + self.zmin_
        out = z * self.std_ + self.mean_
        return out.astype(np.float32)


def _make_block(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU())


class FlexAE(nn.Module):
    def __init__(self, input_dim: int, hidden: Sequence[int], bottleneck: int, option: int):
        super().__init__()
        self.option = int(option)
        sizes_enc = [input_dim] + list(hidden) + [bottleneck]

        enc_layers = []
        for i in range(len(sizes_enc) - 1):
            enc_layers.append(_make_block(sizes_enc[i], sizes_enc[i + 1]))
        self.encoder = nn.Sequential(*enc_layers)

        def build_decoder() -> nn.Sequential:
            sizes_dec = [bottleneck] + list(reversed(hidden)) + [input_dim]
            dec_layers = []
            for i in range(len(sizes_dec) - 2):
                dec_layers.append(_make_block(sizes_dec[i], sizes_dec[i + 1]))
            dec_layers.append(nn.Linear(sizes_dec[-2], sizes_dec[-1]))
            return nn.Sequential(*dec_layers)

        if self.option == 3:
            self.decoder_drop = build_decoder()
            self.decoder_bio = build_decoder()
            self.gate_head = nn.Linear(bottleneck, input_dim)
        else:
            self.decoder = build_decoder()
            # Keep a dropout head available for all non-expert options; it can be
            # supervised explicitly when enabled via extra params.
            self.gate_head = nn.Linear(bottleneck, input_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor | None]:
        z = self.encoder(x)
        if self.option == 3:
            recon_drop = self.decoder_drop(z)
            recon_bio = self.decoder_bio(z)
            gate_logits = self.gate_head(z)
            gate = torch.sigmoid(gate_logits)
            recon = gate * recon_drop + (1.0 - gate) * recon_bio
            return {
                "recon": recon,
                "gate_logits": gate_logits,
                "recon_bio": recon_bio,
                "recon_drop": recon_drop,
                "z": z,
            }

        recon = self.decoder(z)
        gate_logits = self.gate_head(z) if self.gate_head is not None else None
        return {
            "recon": recon,
            "gate_logits": gate_logits,
            "recon_bio": None,
            "recon_drop": None,
            "z": z,
        }


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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _dropout_from_recon_global(
    recon_log: np.ndarray,
    zeros_obs: np.ndarray,
    midpoint: float,
    scale: float,
) -> np.ndarray:
    p_dropout = np.ones_like(recon_log, dtype=np.float32)
    if np.any(zeros_obs):
        score = (recon_log - float(midpoint)) / max(float(scale), EPSILON)
        p_dropout_zero = _sigmoid_np(score)
        p_dropout[zeros_obs] = p_dropout_zero[zeros_obs]
    return np.clip(p_dropout, 0.0, 1.0)


def _dropout_from_recon_em(
    recon_log: np.ndarray,
    zeros_obs: np.ndarray,
    midpoint: float,
    scale: float,
) -> np.ndarray:
    pi = _dropout_from_recon_global(recon_log, zeros_obs, midpoint, scale)
    mu = np.clip(np.expm1(recon_log * np.log(2.0)), 1e-8, None)
    # Poisson approximation to NB zero-probability for fast EM-style posterior updates.
    p0_nb = np.exp(-mu)
    numer = pi
    denom = pi + (1.0 - pi) * p0_nb + 1e-8
    q_dropout = numer / denom
    q_dropout[~zeros_obs] = 0.0
    return np.clip(q_dropout.astype(np.float32), 0.0, 1.0)


def _dropout_from_recon_neighbor(
    recon_log: np.ndarray,
    zeros_obs: np.ndarray,
    midpoint: float,
    scale: float,
    k: int,
    recon_w: float,
    nbr_w: float,
) -> np.ndarray:
    n_cells = recon_log.shape[0]
    if n_cells <= 2:
        return _dropout_from_recon_global(recon_log, zeros_obs, midpoint, scale)

    x = recon_log.astype(np.float64)
    xc = x - x.mean(axis=0, keepdims=True)
    try:
        u, s, _ = np.linalg.svd(xc, full_matrices=False)
        d = min(8, u.shape[1])
        emb = u[:, :d] * s[:d]
    except Exception:
        emb = xc[:, : min(8, xc.shape[1])]

    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    kk = int(max(2, min(max(k, 2), n_cells - 1)))
    nbr_idx = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    nbr_mean = np.mean(recon_log[nbr_idx], axis=1)

    blend = float(recon_w) * recon_log + float(nbr_w) * nbr_mean
    return _dropout_from_recon_global(blend, zeros_obs, midpoint, scale)


def _dropout_from_recon_gene_specific(
    recon_log: np.ndarray,
    zeros_obs: np.ndarray,
    nonzero_obs: np.ndarray,
    midpoint: float,
    scale: float,
    shrink: float,
) -> np.ndarray:
    p_dropout = np.ones_like(recon_log, dtype=np.float32)
    g = recon_log.shape[1]
    global_mid = float(midpoint)
    global_scale = max(float(scale), EPSILON)
    shrink = float(np.clip(shrink, 0.0, 1.0))

    mids = np.full(g, global_mid, dtype=np.float32)
    scales = np.full(g, global_scale, dtype=np.float32)

    for j in range(g):
        vals = recon_log[nonzero_obs[:, j], j]
        if vals.size >= 8:
            q25 = float(np.percentile(vals, 25.0))
            q75 = float(np.percentile(vals, 75.0))
            iqr = max(q75 - q25, 1e-3)
            s = max(iqr / 1.349, 1e-3)
            mids[j] = (1.0 - shrink) * q25 + shrink * global_mid
            scales[j] = (1.0 - shrink) * s + shrink * global_scale

    score = (recon_log - mids[None, :]) / scales[None, :]
    p_dropout_zero = _sigmoid_np(score)
    p_dropout[zeros_obs] = p_dropout_zero[zeros_obs]
    return np.clip(p_dropout, 0.0, 1.0)


def _logcounts_from_counts_median_norm(counts: np.ndarray) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    lib = x.sum(axis=1)
    med = float(np.median(lib[lib > 0.0])) if np.any(lib > 0.0) else 1.0
    med = med if np.isfinite(med) and med > 0.0 else 1.0
    sf = lib / med
    sf = np.where(np.isfinite(sf) & (sf > 0.0), sf, 1.0)
    norm = x / sf[:, None]
    return (np.log1p(norm) / np.log(2.0)).astype(np.float32)


def _logcounts_from_counts_cpm(counts: np.ndarray, target_sum: float = 1e6) -> np.ndarray:
    x = np.asarray(counts, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    lib = x.sum(axis=1)
    denom = np.where(np.isfinite(lib) & (lib > 0.0), lib, 1.0)
    cpm = (x / denom[:, None]) * float(target_sum)
    return (np.log1p(cpm) / np.log(2.0)).astype(np.float32)


def _pca_embed_np(x: np.ndarray, n_components: int = 16) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    d = max(1, min(int(n_components), u.shape[1]))
    return (u[:, :d] * s[:d]).astype(np.float32)


def _kmeans_simple(x: np.ndarray, k: int, seed: int = 42, max_iter: int = 50) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    k = int(max(2, min(k, n)))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(int(max_iter)):
        dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist2.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = x[mask].mean(axis=0)
            else:
                centers[j] = x[rng.integers(0, n)]
    return labels


def _silhouette_score_np(x: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    n = x.shape[0]
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= n:
        return float("nan")
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (x @ x.T)
    dist = np.sqrt(np.maximum(dist2, 0.0))
    sil = np.zeros(n, dtype=np.float32)
    for i in range(n):
        same = labels == labels[i]
        if same.sum() <= 1:
            a = 0.0
        else:
            a = dist[i, same].sum() / max(int(same.sum()) - 1, 1)
        b = np.inf
        for cl in uniq:
            if cl == labels[i]:
                continue
            mask = labels == cl
            if not np.any(mask):
                continue
            b = min(b, float(dist[i, mask].mean()))
        if np.isfinite(b):
            den = max(a, b)
            sil[i] = 0.0 if den <= 0.0 else (b - a) / den
    return float(np.mean(sil))


def _graph_smooth_cells(log_x: np.ndarray, k: int, blend: float, iters: int) -> np.ndarray:
    if blend <= 0.0:
        return log_x
    x = np.asarray(log_x, dtype=np.float32)
    n = x.shape[0]
    if n <= 2:
        return x
    emb = _pca_embed_np(x, n_components=16)
    diff = emb[:, None, :] - emb[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    kk = int(max(2, min(int(k), n - 1)))
    nbr_idx = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    out = x.copy()
    a = float(np.clip(blend, 0.0, 1.0))
    for _ in range(int(max(1, iters))):
        nbr_mean = np.mean(out[nbr_idx], axis=1)
        out = (1.0 - a) * out + a * nbr_mean
    return out.astype(np.float32, copy=False)


def _cluster_centroid_blend(
    log_x: np.ndarray,
    blend: float,
    k_min: int,
    k_max: int,
    seed: int = 42,
) -> np.ndarray:
    if blend <= 0.0:
        return log_x
    x = np.asarray(log_x, dtype=np.float32)
    n = x.shape[0]
    if n <= 3:
        return x
    emb = _pca_embed_np(x, n_components=16)
    lo = int(max(2, k_min))
    hi = int(max(lo, min(k_max, n - 1)))
    best_k = lo
    best_score = -np.inf
    best_labels = None
    for k in range(lo, hi + 1):
        labels = _kmeans_simple(emb, k, seed=seed, max_iter=60)
        sil = _silhouette_score_np(emb, labels)
        if np.isfinite(sil) and sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels
    if best_labels is None:
        best_labels = _kmeans_simple(emb, best_k, seed=seed, max_iter=60)
    centroids = np.zeros((int(best_labels.max()) + 1, x.shape[1]), dtype=np.float32)
    for j in range(centroids.shape[0]):
        mask = best_labels == j
        if np.any(mask):
            centroids[j] = x[mask].mean(axis=0)
    a = float(np.clip(blend, 0.0, 1.0))
    out = (1.0 - a) * x + a * centroids[best_labels]
    return out.astype(np.float32, copy=False)


def _postprocess_for_clustering(
    log_recon: np.ndarray,
    extra_params: Dict[str, float | int],
    log_obs: Optional[np.ndarray] = None,
) -> np.ndarray:
    out = np.asarray(log_recon, dtype=np.float32)
    graph_blend = float(extra_params.get("graph_blend", 0.0))
    if graph_blend > 0.0:
        out = _graph_smooth_cells(
            out,
            k=int(extra_params.get("graph_k", 8)),
            blend=graph_blend,
            iters=int(extra_params.get("graph_iters", 1)),
        )
    cluster_blend = float(extra_params.get("cluster_blend", 0.0))
    if cluster_blend > 0.0:
        out = _cluster_centroid_blend(
            out,
            blend=cluster_blend,
            k_min=int(extra_params.get("cluster_k_min", 2)),
            k_max=int(extra_params.get("cluster_k_max", 10)),
            seed=int(extra_params.get("cluster_seed", 42)),
        )
    orig_blend = float(extra_params.get("orig_blend", 0.0))
    if (log_obs is not None) and (orig_blend > 0.0):
        b = float(np.clip(orig_blend, 0.0, 1.0))
        obs = np.asarray(log_obs, dtype=np.float32)
        out = (1.0 - b) * out + b * obs
    return out.astype(np.float32, copy=False)


def _refine_bio_probabilities(
    p_bio_curr: np.ndarray,
    p_bio_prior: np.ndarray,
    p_dropout: np.ndarray,
    zeros_obs: np.ndarray,
    refine_params: Dict[str, float | int | bool],
) -> np.ndarray:
    target_bio = 1.0 - p_dropout
    prior_blend = float(refine_params["prior_blend"])
    if prior_blend > 0.0:
        target_bio = prior_blend * p_bio_prior + (1.0 - prior_blend) * target_bio

    refined = p_bio_curr.copy()
    zmask = zeros_obs
    update_mask = zmask & (p_dropout >= float(refine_params["update_min_dropout_prob"]))
    if not np.any(update_mask):
        return refined.astype(np.float32, copy=False)

    alpha = float(refine_params["ema_alpha"])
    refined[update_mask] = (1.0 - alpha) * p_bio_curr[update_mask] + alpha * target_bio[update_mask]

    prior_floor_ratio = float(refine_params["prior_floor_ratio"])
    if prior_floor_ratio > 0.0:
        floor = prior_floor_ratio * p_bio_prior
        refined[zmask] = np.maximum(refined[zmask], floor[zmask])

    force_thr = float(refine_params["force_dropout_prob"])
    if force_thr > 0.0:
        force_dropout = zmask & (p_dropout >= force_thr)
        refined[force_dropout] = float(refine_params["p_bio_min"])

    refined[~zmask] = 0.0
    refined[zmask] = np.clip(
        refined[zmask],
        float(refine_params["p_bio_min"]),
        float(refine_params["p_bio_max"]),
    )
    return refined.astype(np.float32, copy=False)


def splat_cellaware_bio_prob(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    disp_mode: str,
    use_cell_factor: bool,
) -> np.ndarray:
    bio_post = splatter_bio_posterior_from_counts(
        counts,
        disp_mode=disp_mode,
        use_cell_factor=bool(use_cell_factor),
        groups=None,
    )
    p_bio = np.asarray(bio_post, dtype=np.float64)
    p_bio = np.nan_to_num(p_bio, nan=0.0, posinf=0.0, neginf=0.0)
    p_bio = np.clip(p_bio, 0.0, 1.0)
    p_bio[~zeros_obs] = 0.0
    return p_bio.astype(np.float32)


def _to_dense_float32(x: object) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    elif hasattr(x, "A"):
        x = x.A
    return np.asarray(x, dtype=np.float32)


def load_dataset(path: str) -> Dict[str, np.ndarray] | None:
    sce = read_rds(path)
    if not hasattr(sce, "assay"):
        raise TypeError(f"Unsupported RDS object (expected SingleCellExperiment): {type(sce)}")

    # Real datasets may be sparse matrices; convert to dense once here.
    logcounts = _to_dense_float32(sce.assay("logcounts").T)
    keep = np.sum(logcounts != 0, axis=0) >= 2
    keep = np.asarray(keep).reshape(-1)
    logcounts = logcounts[:, keep]

    log_true = None
    for assay_name in ("logTrueCounts", "perfect_logcounts"):
        try:
            log_true = _to_dense_float32(sce.assay(assay_name).T)
            log_true = log_true[:, keep]
            break
        except Exception:
            continue

    counts = None
    try:
        counts = _to_dense_float32(sce.assay("counts").T)
        counts = counts[:, keep]
    except Exception:
        counts = None

    return {
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts,
    }


def prepare_dataset(path: Path, real_norm_mode: str = "median") -> Dict[str, object] | None:
    dataset = load_dataset(str(path))
    if dataset is None:
        return None
    logcounts = dataset["logcounts"]
    log_true = dataset["log_true"]
    counts_raw = dataset["counts"]
    is_real_dataset = (log_true is None) and (counts_raw is not None)
    if is_real_dataset:
        mode = str(real_norm_mode).strip().lower()
        # Allow explicit normalization choice for real-data clustering sweeps.
        if mode in ("median", "median_libsize", "med"):
            logcounts = _logcounts_from_counts_median_norm(counts_raw)
        elif mode in ("cpm", "cpm1e6"):
            logcounts = _logcounts_from_counts_cpm(counts_raw)
        elif mode in ("native", "logcounts", "as_is"):
            pass
        else:
            logcounts = _logcounts_from_counts_median_norm(counts_raw)

    # For synthetic datasets (log_true available), keep mask construction aligned
    # with previous runs (log-space observed zeros). For real datasets, prefer raw counts.
    if (log_true is not None):
        counts_obs = np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    elif counts_raw is not None:
        counts_obs = np.clip(counts_raw, 0.0, None).astype(np.float32)
    else:
        counts_obs = np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    zeros_obs = counts_obs <= 0.0
    counts_max = counts_obs.max(axis=0)

    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, 5.0))
    cz_hi = float(np.percentile(cell_zero_frac, 95.0))
    cz_span = max(cz_hi - cz_lo, EPSILON)
    cell_zero_norm = np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)

    return {
        "dataset": path.stem,
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts_obs,
        "zeros_obs": zeros_obs,
        "nonzero_obs": ~zeros_obs,
        "counts_max": counts_max,
        "cell_zero_norm": cell_zero_norm,
        "is_real_dataset": bool(is_real_dataset),
    }


def _run_full_prediction(
    model: FlexAE,
    xtr: torch.Tensor,
    scaler: RobustZThenMinMaxToNeg1Pos1,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    model.eval()
    recon_list: List[np.ndarray] = []
    gate_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, xtr.shape[0], batch_size):
            xb = xtr[i : i + batch_size].to(device)
            out = model(xb)
            recon_np = out["recon"].detach().cpu().numpy().astype(np.float32)
            recon_list.append(scaler.inverse_transform(recon_np))
            if out["gate_logits"] is not None:
                gate_np = torch.sigmoid(out["gate_logits"]).detach().cpu().numpy().astype(np.float32)
                gate_list.append(gate_np)

    recon_all = np.vstack(recon_list).astype(np.float32)
    gate_all = np.vstack(gate_list).astype(np.float32) if gate_list else None
    return recon_all, gate_all


def train_autoencoder_reconstruct(
    logcounts: np.ndarray,
    counts_max: np.ndarray,
    p_bio: np.ndarray,
    device: torch.device,
    option: int,
    model_params: Dict[str, object],
    ae_params: Dict[str, float],
    refine_params: Dict[str, float | int | bool],
    extra_params: Dict[str, float | int],
    *,
    refine_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = RobustZThenMinMaxToNeg1Pos1(
        p_low=float(SCALER_PARAMS["p_low"]), p_high=float(SCALER_PARAMS["p_high"])
    ).fit(logcounts)

    xs = scaler.transform(logcounts).astype(np.float32)
    zeros_obs = logcounts <= EPSILON
    nonzero_obs = ~zeros_obs

    bio_prob = np.clip(p_bio.astype(np.float32), 0.0, 1.0)
    bio_prob[~zeros_obs] = 0.0
    bio_prob_prior = bio_prob.copy()

    xtr = torch.tensor(xs, dtype=torch.float32)
    bio_mask = torch.tensor(bio_prob, dtype=torch.float32)
    nz_mask = torch.tensor(nonzero_obs.astype(np.float32), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(xtr, bio_mask, nz_mask),
        batch_size=int(model_params["batch_size"]),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    zero_scaled = ((0.0 - scaler.mean_) / scaler.std_ - scaler.zmin_) / scaler.zspan_
    zero_scaled = zero_scaled * 2.0 - 1.0
    zero_scaled_t = torch.tensor(zero_scaled, dtype=torch.float32, device=device)
    counts_max_t = torch.tensor(np.maximum(counts_max, 1.0), dtype=torch.float32, device=device)

    lo = torch.tensor(scaler.lo_, dtype=torch.float32, device=device)
    hi = torch.tensor(scaler.hi_, dtype=torch.float32, device=device)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
    zmin = torch.tensor(scaler.zmin_, dtype=torch.float32, device=device)
    zspan = torch.tensor(scaler.zspan_, dtype=torch.float32, device=device)

    model = FlexAE(
        input_dim=logcounts.shape[1],
        hidden=model_params["hidden"],
        bottleneck=int(model_params["bottleneck"]),
        option=option,
    ).to(device)

    opt = optim.Adam(
        model.parameters(),
        lr=float(ae_params["lr"]),
        weight_decay=float(model_params["weight_decay"]),
    )

    refine_active = bool(refine_enabled) and bool(refine_params["enabled"])
    refine_start = int(max(1, int(refine_params["start_epoch"])))
    refine_every = int(max(1, int(refine_params["every_n_epochs"])))

    for epoch_idx in range(1, int(ae_params["epochs"]) + 1):
        model.train()
        for xb, bio_b, nz_b in loader:
            xb = xb.to(device)
            bio_b = bio_b.to(device)
            nz_b = nz_b.to(device)

            mask_bio = torch.bernoulli(bio_b * float(ae_params["p_zero"]))
            mask_nz = torch.bernoulli(nz_b * float(ae_params["p_nz"]))

            x_in = xb.clone()
            if mask_nz.any():
                x_in = torch.where(mask_nz.bool(), torch.zeros_like(x_in), x_in)
            if mask_bio.any():
                noise_scale = torch.rand_like(xb) * float(ae_params["noise_max"])
                noise_counts = noise_scale * counts_max_t
                noise_vals = torch.log1p(noise_counts) / float(np.log(2.0))
                noise_vals = torch.minimum(torch.maximum(noise_vals, lo), hi)
                z = (noise_vals - mean) / std
                x01 = (z - zmin) / zspan
                noise_scaled = x01 * 2.0 - 1.0
                x_in = torch.where(mask_bio.bool(), noise_scaled, x_in)

            opt.zero_grad()
            out = model(x_in)
            recon = out["recon"]
            residual = recon - xb
            masked_loss = weighted_masked_mse(
                residual,
                mask_bio=mask_bio,
                mask_nz=mask_nz,
                weight_bio=float(ae_params["loss_bio_weight"]),
                weight_nz=float(ae_params["loss_nz_weight"]),
            )
            bio_reg = ((recon - zero_scaled_t) ** 2 * bio_b).sum() / bio_b.sum().clamp_min(1.0)
            loss = masked_loss + float(ae_params["bio_reg_weight"]) * bio_reg

            full_recon_w = float(extra_params.get("full_recon_weight", 0.0))
            if full_recon_w > 0.0:
                loss = loss + full_recon_w * F.mse_loss(recon, xb)

            cell_sim_w = float(extra_params.get("cell_sim_weight", 0.0))
            if cell_sim_w > 0.0 and xb.shape[0] > 2:
                with torch.no_grad():
                    xb_norm = F.normalize(xb, p=2.0, dim=1)
                    sim_target = xb_norm @ xb_norm.T
                recon_norm = F.normalize(recon, p=2.0, dim=1)
                sim_pred = recon_norm @ recon_norm.T
                loss = loss + cell_sim_w * F.mse_loss(sim_pred, sim_target)

            gate_logits = out["gate_logits"]
            if gate_logits is not None:
                if option in (1, 3):
                    gate_w = float(extra_params.get("gate_loss_weight", 0.5))
                    zero_soft_w = float(extra_params.get("gate_zero_soft_weight", 0.3))
                else:
                    gate_w = float(
                        extra_params.get(
                            "gate_loss_weight",
                            extra_params.get("shared_gate_loss_weight", 0.0),
                        )
                    )
                    zero_soft_w = float(
                        extra_params.get(
                            "gate_zero_soft_weight",
                            extra_params.get("shared_gate_zero_soft_weight", 0.3),
                        )
                    )

                if gate_w > 0.0:
                    pos = mask_nz
                    neg = nz_b * (1.0 - mask_nz)
                    syn_weights = pos + neg
                    syn_labels = pos
                    bce_syn = F.binary_cross_entropy_with_logits(
                        gate_logits,
                        syn_labels,
                        reduction="none",
                    )
                    bce_syn = (bce_syn * syn_weights).sum() / syn_weights.sum().clamp_min(1.0)

                    zero_label = 1.0 - bio_b
                    zero_weights = (bio_b > 0).float()
                    bce_zero = F.binary_cross_entropy_with_logits(
                        gate_logits,
                        zero_label,
                        reduction="none",
                    )
                    bce_zero = (bce_zero * zero_weights).sum() / zero_weights.sum().clamp_min(1.0)

                    loss = loss + gate_w * ((1.0 - zero_soft_w) * bce_syn + zero_soft_w * bce_zero)

                    # Encourage clear separation between synthetic-dropout positives and negatives.
                    margin = float(extra_params.get("gate_margin", 0.0))
                    margin_w = float(extra_params.get("gate_margin_weight", 0.0))
                    if margin > 0.0 and margin_w > 0.0:
                        pos_logits = gate_logits[pos.bool()]
                        neg_logits = gate_logits[neg.bool()]
                        if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                            sep = pos_logits.mean() - neg_logits.mean()
                            margin_loss = F.relu(torch.tensor(margin, device=sep.device) - sep)
                            loss = loss + margin_w * margin_loss

            if option == 3:
                recon_bio = out["recon_bio"]
                recon_drop = out["recon_drop"]
                zmask = (bio_b > 0).float()
                bio_expert = ((recon_bio - zero_scaled_t) ** 2 * zmask).sum() / zmask.sum().clamp_min(1.0)
                drop_expert = ((recon_drop - xb) ** 2 * nz_b).sum() / nz_b.sum().clamp_min(1.0)
                loss = loss + float(extra_params.get("expert_bio_weight", 0.4)) * bio_expert
                loss = loss + float(extra_params.get("expert_drop_weight", 0.2)) * drop_expert

            # Latent consistency under additional input corruption helps clustering stability.
            lat_w = float(extra_params.get("latent_consistency_weight", 0.0))
            if lat_w > 0.0:
                aug_keep = torch.bernoulli(nz_b * float(extra_params.get("latent_aug_drop_prob", 0.15)))
                x_aug = torch.where(aug_keep.bool(), torch.zeros_like(xb), xb)
                out_aug = model(x_aug)
                z_main = out["z"]
                z_aug = out_aug["z"]
                lat_cons = F.mse_loss(z_main, z_aug)
                loss = loss + lat_w * lat_cons

            loss.backward()
            opt.step()

        if refine_active and epoch_idx >= refine_start:
            if ((epoch_idx - refine_start) % refine_every == 0) or (epoch_idx == int(ae_params["epochs"])):
                recon_probe, gate_probe = _run_full_prediction(
                    model=model,
                    xtr=xtr,
                    scaler=scaler,
                    batch_size=int(model_params["batch_size"]),
                    device=device,
                )

                if option in (1, 3) and gate_probe is not None:
                    p_dropout = np.zeros_like(recon_probe, dtype=np.float32)
                    p_dropout[zeros_obs] = gate_probe[zeros_obs]
                elif option == 2:
                    p_dropout = _dropout_from_recon_em(
                        recon_probe,
                        zeros_obs,
                        midpoint=float(refine_params["dropout_midpoint"]),
                        scale=float(refine_params["dropout_scale"]),
                    )
                elif option == 4:
                    p_dropout = _dropout_from_recon_neighbor(
                        recon_probe,
                        zeros_obs,
                        midpoint=float(refine_params["dropout_midpoint"]),
                        scale=float(refine_params["dropout_scale"]),
                        k=int(extra_params.get("knn_k", 8)),
                        recon_w=float(extra_params.get("knn_recon_weight", 0.55)),
                        nbr_w=float(extra_params.get("knn_neighbor_weight", 0.45)),
                    )
                elif option == 5:
                    p_dropout = _dropout_from_recon_gene_specific(
                        recon_probe,
                        zeros_obs,
                        nonzero_obs,
                        midpoint=float(refine_params["dropout_midpoint"]),
                        scale=float(refine_params["dropout_scale"]),
                        shrink=float(extra_params.get("gene_shrink", 0.4)),
                    )
                else:
                    p_dropout = _dropout_from_recon_global(
                        recon_probe,
                        zeros_obs,
                        midpoint=float(refine_params["dropout_midpoint"]),
                        scale=float(refine_params["dropout_scale"]),
                    )

                if gate_probe is not None:
                    gate_blend = float(extra_params.get("refine_gate_blend", 0.0))
                    if gate_blend > 0.0:
                        b = float(np.clip(gate_blend, 0.0, 1.0))
                        p_dropout[zeros_obs] = (1.0 - b) * p_dropout[zeros_obs] + b * gate_probe[zeros_obs]

                bio_prob = _refine_bio_probabilities(
                    p_bio_curr=bio_prob,
                    p_bio_prior=bio_prob_prior,
                    p_dropout=p_dropout,
                    zeros_obs=zeros_obs,
                    refine_params=refine_params,
                )
                bio_mask.copy_(torch.from_numpy(bio_prob.astype(np.float32)))

    recon_all, _ = _run_full_prediction(
        model=model,
        xtr=xtr,
        scaler=scaler,
        batch_size=int(model_params["batch_size"]),
        device=device,
    )
    return recon_all.astype(np.float32), bio_prob.astype(np.float32)


def _mse_from_diff(diff: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff ** 2))


def compute_mse_metrics(pred_log: np.ndarray, true_log: Optional[np.ndarray], counts_obs: np.ndarray) -> Dict[str, float]:
    if true_log is None:
        return {
            "mse": float("nan"),
            "mse_dropout": float("nan"),
            "mse_biozero": float("nan"),
            "mse_non_zero": float("nan"),
        }
    diff = true_log - pred_log
    mask_biozero = true_log <= EPSILON
    mask_dropout = (true_log > EPSILON) & (counts_obs <= EPSILON)
    mask_non_zero = (true_log > EPSILON) & (counts_obs > EPSILON)
    return {
        "mse": _mse_from_diff(diff),
        "mse_dropout": _mse_from_diff(diff, mask_dropout),
        "mse_biozero": _mse_from_diff(diff, mask_biozero),
        "mse_non_zero": _mse_from_diff(diff, mask_non_zero),
    }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_table(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


def _parse_hidden(raw: str) -> List[int]:
    s = str(raw).strip()
    if not s:
        return [64]
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else [64]


def run_option(
    option: int,
    input_path: str,
    output_dir: Path,
    device: torch.device,
    seed: int,
    keep_positive: bool,
    save_imputed: bool,
    model_params: Dict[str, object],
    ae_params: Dict[str, float],
    bio_params: Dict[str, object],
    refine_params: Dict[str, float | int | bool],
    extra_params: Dict[str, float | int],
    real_norm_mode: str,
    refine_enabled: bool,
) -> Dict[str, float]:
    datasets: List[Dict[str, object]] = []
    for path in sorted(Path(input_path).rglob("*.rds")):
        ds = prepare_dataset(path, real_norm_mode=real_norm_mode)
        if ds is None:
            continue
        datasets.append(ds)

    if not datasets:
        raise SystemExit("No datasets processed.")

    rows: List[Dict[str, object]] = []
    mse_list: List[float] = []
    bz_list: List[float] = []

    start_time = time.perf_counter()
    for ds in datasets:
        set_seed(seed)
        p_bio = splat_cellaware_bio_prob(
            counts=ds["counts"],
            zeros_obs=ds["zeros_obs"],
            disp_mode=str(bio_params["disp_mode"]),
            use_cell_factor=bool(bio_params["use_cell_factor"]),
        )
        if float(bio_params["cell_zero_weight"]) > 0.0:
            cell_w = np.clip(
                float(bio_params["cell_zero_weight"]) * ds["cell_zero_norm"],
                0.0,
                1.0,
            )
            p_bio = p_bio * (1.0 - cell_w[:, None])

        recon, p_bio_final = train_autoencoder_reconstruct(
            logcounts=ds["logcounts"],
            counts_max=ds["counts_max"],
            p_bio=p_bio,
            device=device,
            option=option,
            model_params=model_params,
            ae_params=ae_params,
            refine_params=refine_params,
            extra_params=extra_params,
            refine_enabled=refine_enabled,
        )

        log_recon = recon
        if keep_positive:
            log_recon[~ds["zeros_obs"]] = ds["logcounts"][~ds["zeros_obs"]]
        elif bool(ds.get("is_real_dataset", False)):
            # For real datasets (no logTrueCounts), enable optional clustering-oriented refinement.
            log_recon = _postprocess_for_clustering(log_recon, extra_params, log_obs=ds["logcounts"])

        metrics = compute_mse_metrics(log_recon, ds["log_true"], ds["counts"])
        rows.append(
            {
                "dataset": ds["dataset"],
                "mse": metrics["mse"],
                "mse_biozero": metrics["mse_biozero"],
                "mse_dropout": metrics["mse_dropout"],
                "mse_non_zero": metrics["mse_non_zero"],
            }
        )

        if not np.isnan(metrics["mse"]):
            mse_list.append(float(metrics["mse"]))
        if not np.isnan(metrics["mse_biozero"]):
            bz_list.append(float(metrics["mse_biozero"]))

        if save_imputed:
            np.savez_compressed(
                output_dir / f"{ds['dataset']}_imputed.npz",
                logcounts=ds["logcounts"],
                log_imputed=log_recon,
                p_bio=p_bio_final,
            )

    avg_mse = float(np.nanmean(mse_list)) if mse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    runtime_sec = float(time.perf_counter() - start_time)

    metrics_name = f"masked_imputation_option{option}_metrics.tsv"
    summary_name = f"masked_imputation_option{option}_summary.tsv"

    _write_table(
        output_dir / metrics_name,
        ["dataset", "mse", "mse_biozero", "mse_dropout", "mse_non_zero"],
        rows,
    )
    _write_table(
        output_dir / summary_name,
        ["avg_mse", "avg_biozero", "runtime_sec", "refine_enabled"],
        [
            {
                "avg_mse": avg_mse,
                "avg_biozero": avg_bz,
                "runtime_sec": runtime_sec,
                "refine_enabled": refine_enabled,
            }
        ],
    )

    print(f"\n=== masked_imputation_option{option} ===")
    print("AE params:", ae_params)
    print("Model params:", model_params)
    print("Refine params:", refine_params, "enabled:", refine_enabled)
    print("Extra params:", extra_params)
    print("avg_mse:", avg_mse, "avg_biozero:", avg_bz)
    print("runtime_sec:", runtime_sec)
    print("Metrics written to", metrics_name)
    print("Summary written to", summary_name)

    return {
        "avg_mse": avg_mse,
        "avg_biozero": avg_bz,
        "runtime_sec": runtime_sec,
    }


def main_for_option(option: int) -> None:
    if option not in OPTION_DEFAULTS:
        raise SystemExit(f"Unsupported option: {option}")

    parser = argparse.ArgumentParser(description=f"masked imputation option {option}")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-positive", default="true")
    parser.add_argument("--save-imputed", default="false")
    parser.add_argument("--no-refine", action="store_true")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--p-zero", type=float, default=None)
    parser.add_argument("--p-nz", type=float, default=None)
    parser.add_argument("--noise-max", type=float, default=None)
    parser.add_argument("--loss-bio-weight", type=float, default=None)
    parser.add_argument("--loss-nz-weight", type=float, default=None)
    parser.add_argument("--bio-reg-weight", type=float, default=None)

    parser.add_argument("--hidden", default=None, help="Comma-separated hidden widths, e.g. 128,64")
    parser.add_argument("--bottleneck", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument("--refine-start", type=int, default=None)
    parser.add_argument("--refine-every", type=int, default=None)
    parser.add_argument("--refine-alpha", type=float, default=None)
    parser.add_argument("--prior-blend", type=float, default=None)
    parser.add_argument("--update-min-dropout", type=float, default=None)

    parser.add_argument("--gate-loss-weight", type=float, default=None)
    parser.add_argument("--gate-margin", type=float, default=None)
    parser.add_argument("--gate-margin-weight", type=float, default=None)
    parser.add_argument("--expert-bio-weight", type=float, default=None)
    parser.add_argument("--expert-drop-weight", type=float, default=None)
    parser.add_argument("--latent-consistency-weight", type=float, default=None)
    parser.add_argument("--latent-aug-drop-prob", type=float, default=None)
    parser.add_argument("--full-recon-weight", type=float, default=None)
    parser.add_argument("--cell-sim-weight", type=float, default=None)
    parser.add_argument("--shared-gate-loss-weight", type=float, default=None)
    parser.add_argument("--shared-gate-zero-soft-weight", type=float, default=None)
    parser.add_argument("--refine-gate-blend", type=float, default=None)
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--gene-shrink", type=float, default=None)
    parser.add_argument("--graph-blend", type=float, default=None)
    parser.add_argument("--graph-k", type=int, default=None)
    parser.add_argument("--graph-iters", type=int, default=None)
    parser.add_argument("--cluster-blend", type=float, default=None)
    parser.add_argument("--cluster-k-min", type=int, default=None)
    parser.add_argument("--cluster-k-max", type=int, default=None)
    parser.add_argument("--orig-blend", type=float, default=None)
    parser.add_argument("--real-norm-mode", default="median", help="Real-data normalization: median|cpm|native")

    args = parser.parse_args()

    keep_positive = str(args.keep_positive).strip().lower() in ("1", "true", "yes", "y")
    save_imputed = str(args.save_imputed).strip().lower() in ("1", "true", "yes", "y")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bio_params = copy.deepcopy(BIO_PARAMS_BASE)
    model_params = copy.deepcopy(MODEL_PARAMS_BASE)
    ae_params = copy.deepcopy(AE_PARAMS_BASE)
    refine_params = copy.deepcopy(REFINE_PARAMS_BASE)
    extra_params: Dict[str, float | int] = copy.deepcopy(OPTION_DEFAULTS[option]["extra"])

    # Option defaults override base params.
    ae_params.update(OPTION_DEFAULTS[option]["ae"])
    refine_params.update(OPTION_DEFAULTS[option]["refine"])

    # CLI overrides.
    if args.epochs is not None:
        ae_params["epochs"] = int(args.epochs)
    if args.lr is not None:
        ae_params["lr"] = float(args.lr)
    if args.p_zero is not None:
        ae_params["p_zero"] = float(args.p_zero)
    if args.p_nz is not None:
        ae_params["p_nz"] = float(args.p_nz)
    if args.noise_max is not None:
        ae_params["noise_max"] = float(args.noise_max)
    if args.loss_bio_weight is not None:
        ae_params["loss_bio_weight"] = float(args.loss_bio_weight)
    if args.loss_nz_weight is not None:
        ae_params["loss_nz_weight"] = float(args.loss_nz_weight)
    if args.bio_reg_weight is not None:
        ae_params["bio_reg_weight"] = float(args.bio_reg_weight)

    if args.hidden is not None:
        model_params["hidden"] = _parse_hidden(args.hidden)
    if args.bottleneck is not None:
        model_params["bottleneck"] = int(args.bottleneck)
    if args.batch_size is not None:
        model_params["batch_size"] = int(args.batch_size)
    if args.weight_decay is not None:
        model_params["weight_decay"] = float(args.weight_decay)

    if args.refine_start is not None:
        refine_params["start_epoch"] = int(args.refine_start)
    if args.refine_every is not None:
        refine_params["every_n_epochs"] = int(args.refine_every)
    if args.refine_alpha is not None:
        refine_params["ema_alpha"] = float(args.refine_alpha)
    if args.prior_blend is not None:
        refine_params["prior_blend"] = float(args.prior_blend)
    if args.update_min_dropout is not None:
        refine_params["update_min_dropout_prob"] = float(args.update_min_dropout)

    if args.gate_loss_weight is not None:
        extra_params["gate_loss_weight"] = float(args.gate_loss_weight)
    if args.gate_margin is not None:
        extra_params["gate_margin"] = float(args.gate_margin)
    if args.gate_margin_weight is not None:
        extra_params["gate_margin_weight"] = float(args.gate_margin_weight)
    if args.expert_bio_weight is not None:
        extra_params["expert_bio_weight"] = float(args.expert_bio_weight)
    if args.expert_drop_weight is not None:
        extra_params["expert_drop_weight"] = float(args.expert_drop_weight)
    if args.latent_consistency_weight is not None:
        extra_params["latent_consistency_weight"] = float(args.latent_consistency_weight)
    if args.latent_aug_drop_prob is not None:
        extra_params["latent_aug_drop_prob"] = float(args.latent_aug_drop_prob)
    if args.full_recon_weight is not None:
        extra_params["full_recon_weight"] = float(args.full_recon_weight)
    if args.cell_sim_weight is not None:
        extra_params["cell_sim_weight"] = float(args.cell_sim_weight)
    if args.shared_gate_loss_weight is not None:
        extra_params["shared_gate_loss_weight"] = float(args.shared_gate_loss_weight)
    if args.shared_gate_zero_soft_weight is not None:
        extra_params["shared_gate_zero_soft_weight"] = float(args.shared_gate_zero_soft_weight)
    if args.refine_gate_blend is not None:
        extra_params["refine_gate_blend"] = float(args.refine_gate_blend)
    if args.knn_k is not None:
        extra_params["knn_k"] = int(args.knn_k)
    if args.gene_shrink is not None:
        extra_params["gene_shrink"] = float(args.gene_shrink)
    if args.graph_blend is not None:
        extra_params["graph_blend"] = float(args.graph_blend)
    if args.graph_k is not None:
        extra_params["graph_k"] = int(args.graph_k)
    if args.graph_iters is not None:
        extra_params["graph_iters"] = int(args.graph_iters)
    if args.cluster_blend is not None:
        extra_params["cluster_blend"] = float(args.cluster_blend)
    if args.cluster_k_min is not None:
        extra_params["cluster_k_min"] = int(args.cluster_k_min)
    if args.cluster_k_max is not None:
        extra_params["cluster_k_max"] = int(args.cluster_k_max)
    if args.orig_blend is not None:
        extra_params["orig_blend"] = float(args.orig_blend)

    set_seed(int(args.seed))
    run_option(
        option=option,
        input_path=args.input_path,
        output_dir=output_dir,
        device=device,
        seed=int(args.seed),
        keep_positive=keep_positive,
        save_imputed=save_imputed,
        model_params=model_params,
        ae_params=ae_params,
        bio_params=bio_params,
        refine_params=refine_params,
        extra_params=extra_params,
        real_norm_mode=str(args.real_norm_mode),
        refine_enabled=(not bool(args.no_refine)),
    )


if __name__ == "__main__":
    raise SystemExit("Use masked_imputation_option{1..5}.py wrappers.")
