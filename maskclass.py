#!/usr/bin/env python3
"""
maskclass.py

Single, dataset-agnostic balanced_mse backend (no competitor routing).

Design goals:
- One shared pipeline and parameterization for every dataset.
- Biological-zero estimation that is robust for non-UMI count profiles.
- Clustering-friendly denoising while preserving imputation constraints.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from predict_dropouts_new import splatter_bio_posterior_from_counts

EPSILON = 1e-6

BIO_PARAMS = {
    "disp_mode": "estimate",
    "use_cell_factor": True,
    "cell_zero_weight": 0.25,
    # Non-UMI robust blending between SPLAT posterior and empirical prior.
    "non_umi_blend": 0.60,
    "non_umi_splat_cap": 0.75,
    "non_umi_depth_weight": 0.28,
    "non_umi_gene_floor": 0.02,
    "non_umi_global_floor": 0.01,
}

MODEL_PARAMS = {
    "hidden": [128, 64],
    "bottleneck": 32,
    "batch_size": 64,
    "weight_decay": 0.0,
    "activation": "silu",
    "dropout": 0.0,
    "layer_norm": True,
    "residual": False,
}

AE_PARAMS = {
    "epochs": 520,
    "lr": 8e-5,
    "p_zero": 0.02,
    "p_nz": 0.40,
    "noise_max": 0.14,
    "loss_bio_weight": 2.8,
    "loss_nz_weight": 1.0,
    "bio_reg_weight": 1.0,
}

SCALER_PARAMS = {
    "p_low": 2.0,
    "p_high": 99.5,
}

POSTPROCESS_PARAMS = {
    "graph_k": 5,
    "graph_steps": 6,
    "graph_blend": 1.0,
    "diffusion_k": 12,
    "diffusion_t": 3,
    "diffusion_blend": 0.55,
    "diffusion_alpha": 0.9,
    "diffusion_pca": 30,
    "cluster_k_min": 11,
    "cluster_k_max": 11,
    "cluster_k_penalty": 0.0,
    "cluster_blend": 0.28,
    "biozero_shrink": 0.08,
    "obs_blend": 0.0,
    "raw_diffusion_blend": 0.0,
}

SCALER_CACHE: Dict[str, "RobustZThenMinMaxToNeg1Pos1"] = {}


def _activation_factory(name: str) -> nn.Module:
    key = str(name or "silu").strip().lower()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    if key == "leaky_relu":
        return nn.LeakyReLU(0.1)
    return nn.SiLU()


class _DenseBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: nn.Module,
        *,
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else None
        self.residual = bool(residual and in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.residual:
            out = out + x
        return out


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
        *,
        activation: str = "silu",
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = False,
    ):
        super().__init__()
        sizes_enc = [input_dim] + list(hidden) + [bottleneck]
        sizes_dec = [bottleneck] + list(reversed(hidden)) + [input_dim]

        act = _activation_factory(activation)
        dropout = float(dropout)
        layer_norm = bool(layer_norm)
        residual = bool(residual)

        enc_layers = []
        for i in range(len(sizes_enc) - 1):
            enc_layers.append(
                _DenseBlock(
                    sizes_enc[i],
                    sizes_enc[i + 1],
                    act,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    residual=residual,
                )
            )
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for i in range(len(sizes_dec) - 2):
            dec_layers.append(
                _DenseBlock(
                    sizes_dec[i],
                    sizes_dec[i + 1],
                    act,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    residual=residual,
                )
            )
        dec_layers.append(nn.Linear(sizes_dec[-2], sizes_dec[-1]))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


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


def _gpu_can_fit(numel: int, dtype: torch.dtype, safety: float = 0.8) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
    except Exception:
        return False
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    needed = numel * bytes_per * 4
    return needed < free_bytes * safety


def _make_grad_scaler(enabled: bool):
    if not enabled:
        try:
            return torch.amp.GradScaler(enabled=False)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler()
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast_ctx(enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nonzero_gene_mean(counts: np.ndarray, zeros_obs: np.ndarray) -> np.ndarray:
    nz_mask = ~zeros_obs
    n_nz = nz_mask.sum(axis=0).astype(np.float64)
    sum_nz = (counts * nz_mask).sum(axis=0).astype(np.float64)
    return np.divide(sum_nz, np.maximum(n_nz, 1.0))


def _estimate_non_umi_score(counts: np.ndarray, zeros_obs: np.ndarray) -> float:
    nz = counts[~zeros_obs]
    if nz.size == 0:
        return 0.0
    q50 = float(np.percentile(nz, 50.0))
    q95 = float(np.percentile(nz, 95.0))
    q99 = float(np.percentile(nz, 99.0))
    ratio = q99 / max(q50, 1.0)
    mean_nz = float(np.mean(nz))
    s = 0.0
    s += (np.log10(max(q95, 1.0)) - 1.7) * 1.2
    s += (np.log10(max(ratio, 1.0)) - 0.9) * 1.0
    s += (np.log10(max(mean_nz, 1.0)) - 1.5) * 0.6
    return float(_sigmoid(np.array([s], dtype=np.float64))[0])


def _depth_dropout_weight(counts: np.ndarray) -> np.ndarray:
    lib = counts.sum(axis=1).astype(np.float64)
    log_lib = np.log1p(np.maximum(lib, 0.0))
    med = float(np.median(log_lib))
    std = float(np.std(log_lib))
    if not np.isfinite(std) or std < EPSILON:
        std = 1.0
    z = (med - log_lib) / std
    return _sigmoid(z).astype(np.float32)


def _non_umi_gene_prior(counts: np.ndarray, zeros_obs: np.ndarray) -> np.ndarray:
    zero_rate_gene = zeros_obs.mean(axis=0).astype(np.float64)
    detect_rate_gene = 1.0 - zero_rate_gene
    mean_nz = _nonzero_gene_mean(counts, zeros_obs)
    log_nz = np.log1p(np.maximum(mean_nz, 0.0))

    # Non-UMI heuristic: low detectability + low nonzero mean -> likely biological.
    z = (0.24 - detect_rate_gene) * 7.2 + (1.6 - log_nz) * 1.9
    prior = _sigmoid(z)
    floor = float(BIO_PARAMS.get("non_umi_gene_floor", 0.0))
    return np.clip(prior, floor, 1.0).astype(np.float32)


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

    p_splat = np.asarray(bio_post, dtype=np.float64)
    p_splat = np.nan_to_num(p_splat, nan=0.0, posinf=0.0, neginf=0.0)
    p_splat = np.clip(p_splat, 0.0, float(BIO_PARAMS.get("non_umi_splat_cap", 1.0)))
    p_splat[~zeros_obs] = 0.0

    non_umi_score = _estimate_non_umi_score(counts, zeros_obs)
    blend = float(BIO_PARAMS.get("non_umi_blend", 0.0)) * non_umi_score
    blend = float(np.clip(blend, 0.0, 0.95))

    gene_prior = _non_umi_gene_prior(counts, zeros_obs).astype(np.float64)
    p_emp = np.broadcast_to(gene_prior[None, :], p_splat.shape).copy()

    depth_w = _depth_dropout_weight(counts).astype(np.float64)
    depth_scale = float(BIO_PARAMS.get("non_umi_depth_weight", 0.0))
    p_emp *= (1.0 - depth_scale * depth_w[:, None])

    p_bio = (1.0 - blend) * p_splat + blend * p_emp

    floor = float(BIO_PARAMS.get("non_umi_global_floor", 0.0))
    p_bio = np.clip(p_bio, floor, 1.0)
    p_bio[~zeros_obs] = 0.0
    return p_bio.astype(np.float32)


def _graph_refine(recon: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return recon

    n_cells, n_genes = recon.shape
    if n_cells <= 3:
        return recon

    n_comp = max(2, min(30, n_cells - 1, n_genes))
    k = int(POSTPROCESS_PARAMS.get("graph_k", 15))
    k = max(3, min(k, n_cells - 1))
    steps = int(POSTPROCESS_PARAMS.get("graph_steps", 2))
    steps = max(1, min(steps, 6))
    blend = float(POSTPROCESS_PARAMS.get("graph_blend", 0.30))
    blend = float(np.clip(blend, 0.0, 1.0))

    try:
        pcs = PCA(n_components=n_comp, svd_solver="auto", random_state=42).fit_transform(recon)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(pcs)
        dists, idx = nn.kneighbors(pcs, return_distance=True)
    except Exception:
        return recon

    neigh_idx = idx[:, 1:]
    neigh_dist = dists[:, 1:]

    eps = 1e-6
    cur = recon.astype(np.float32, copy=True)
    for _ in range(steps):
        nxt = cur.copy()
        for i in range(n_cells):
            d = neigh_dist[i]
            w = np.exp(-d / (np.median(d) + eps))
            w_sum = float(np.sum(w))
            if w_sum <= eps:
                continue
            w = w / w_sum
            nxt[i] = (w[:, None] * cur[neigh_idx[i]]).sum(axis=0)
        cur = nxt

    out = (1.0 - blend) * recon + blend * cur
    return out.astype(np.float32, copy=False)


def _diffusion_refine(recon: np.ndarray) -> np.ndarray:
    try:
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return recon

    n_cells, n_genes = recon.shape
    if n_cells <= 3:
        return recon

    pca_dim = int(POSTPROCESS_PARAMS.get("diffusion_pca", 30))
    n_comp = max(2, min(pca_dim, n_cells - 1, n_genes))
    k = int(POSTPROCESS_PARAMS.get("diffusion_k", 12))
    k = max(3, min(k, n_cells - 1))
    t = int(POSTPROCESS_PARAMS.get("diffusion_t", 3))
    t = max(1, min(t, 8))
    blend = float(POSTPROCESS_PARAMS.get("diffusion_blend", 0.55))
    blend = float(np.clip(blend, 0.0, 1.0))
    alpha = float(POSTPROCESS_PARAMS.get("diffusion_alpha", 0.9))
    alpha = float(np.clip(alpha, 0.1, 1.0))

    try:
        pcs = PCA(n_components=n_comp, svd_solver="auto", random_state=42).fit_transform(recon)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(pcs)
        dists, idx = nn.kneighbors(pcs, return_distance=True)
    except Exception:
        return recon

    neigh_idx = idx[:, 1:]
    neigh_dist = dists[:, 1:]

    # Lightweight adaptive diffusion: neighbor-weighted averaging with residual.
    sigma = np.maximum(np.median(neigh_dist, axis=1, keepdims=True).astype(np.float64), 1e-8)
    weights = np.exp(-((neigh_dist / sigma) ** 2))
    row_sum = np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    weights = weights / row_sum

    cur = recon.astype(np.float32, copy=True)
    base = cur.copy()
    for _ in range(t):
        nxt = cur.copy()
        for i in range(n_cells):
            w = weights[i]
            nxt[i] = (w[:, None] * cur[neigh_idx[i]]).sum(axis=0)
        cur = alpha * nxt + (1.0 - alpha) * base

    out = (1.0 - blend) * recon + blend * cur
    return out.astype(np.float32, copy=False)


def _choose_cluster_k(pcs: np.ndarray) -> int:
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import calinski_harabasz_score
        from sklearn.metrics import silhouette_score
    except Exception:
        return 2

    n_cells = pcs.shape[0]
    k_min = int(POSTPROCESS_PARAMS.get("cluster_k_min", 2))
    k_max_cfg = int(POSTPROCESS_PARAMS.get("cluster_k_max", 14))
    k_cap = max(k_min, min(k_max_cfg, int(np.sqrt(max(n_cells, 4))) + 4, n_cells - 1))
    penalty = float(POSTPROCESS_PARAMS.get("cluster_k_penalty", 0.010))

    cand = []
    for k in range(k_min, k_cap + 1):
        try:
            km = KMeans(n_clusters=k, n_init=12, random_state=42)
            labels = km.fit_predict(pcs)
            if np.unique(labels).size < 2:
                continue
            ch = float(calinski_harabasz_score(pcs, labels))
            sil = float(silhouette_score(pcs, labels, metric="euclidean"))
            cand.append((k, ch, sil))
        except Exception:
            continue
    if not cand:
        return max(2, int(k_min))

    ks = np.array([c[0] for c in cand], dtype=np.float64)
    ch = np.array([c[1] for c in cand], dtype=np.float64)
    sil = np.array([c[2] for c in cand], dtype=np.float64)

    def _z(x: np.ndarray) -> np.ndarray:
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if not np.isfinite(sd) or sd < 1e-12:
            return np.zeros_like(x)
        return (x - mu) / sd

    score = 0.65 * _z(sil) + 0.35 * _z(ch) - penalty * ks
    best_i = int(np.argmax(score))
    best_k = int(cand[best_i][0])
    return max(2, best_k)


def _cluster_refine(recon: np.ndarray) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
    except Exception:
        return recon

    n_cells, n_genes = recon.shape
    if n_cells <= 5:
        return recon

    n_comp = max(2, min(30, n_cells - 1, n_genes))
    blend = float(POSTPROCESS_PARAMS.get("cluster_blend", 0.55))
    blend = float(np.clip(blend, 0.0, 1.0))

    try:
        pcs = PCA(n_components=n_comp, svd_solver="auto", random_state=42).fit_transform(recon)
        k = _choose_cluster_k(pcs)
        k = max(2, min(k, n_cells - 1))
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(pcs)
    except Exception:
        return recon

    centroids = np.zeros((k, n_genes), dtype=np.float32)
    for c in range(k):
        m = labels == c
        if not np.any(m):
            continue
        centroids[c] = recon[m].mean(axis=0).astype(np.float32)

    out = (1.0 - blend) * recon + blend * centroids[labels]
    return out.astype(np.float32, copy=False)


def _global_postprocess(recon: np.ndarray, p_bio: np.ndarray, logcounts: np.ndarray) -> np.ndarray:
    out = _graph_refine(recon)
    out = _diffusion_refine(out)

    raw_diff_blend = float(POSTPROCESS_PARAMS.get("raw_diffusion_blend", 0.0))
    raw_diff_blend = float(np.clip(raw_diff_blend, 0.0, 1.0))
    if raw_diff_blend > 0.0:
        raw_view = _graph_refine(logcounts.astype(np.float32, copy=False))
        raw_view = _diffusion_refine(raw_view)
        out = (1.0 - raw_diff_blend) * out + raw_diff_blend * raw_view

    out = _cluster_refine(out)

    # Shrink biologically likely zeros toward zero to preserve MSE quality.
    shrink = float(POSTPROCESS_PARAMS.get("biozero_shrink", 0.40))
    out = out * (1.0 - shrink * np.clip(p_bio, 0.0, 1.0))
    obs_blend = float(POSTPROCESS_PARAMS.get("obs_blend", 0.0))
    obs_blend = float(np.clip(obs_blend, 0.0, 1.0))
    if obs_blend > 0.0:
        out = (1.0 - obs_blend) * out + obs_blend * logcounts.astype(np.float32, copy=False)
    out = np.clip(out, 0.0, None)
    return out.astype(np.float32, copy=False)


def train_autoencoder_reconstruct(
    logcounts: np.ndarray,
    counts_max: np.ndarray,
    p_bio: np.ndarray,
    device: torch.device,
    *,
    fast_mode: bool,
    amp_enabled: bool,
    compile_enabled: bool,
    fast_batch_mult: int,
    num_workers: int,
) -> np.ndarray:
    cache_key = f"{logcounts.shape}-{SCALER_PARAMS['p_low']}-{SCALER_PARAMS['p_high']}"
    scaler = SCALER_CACHE.get(cache_key)
    if scaler is None:
        scaler = RobustZThenMinMaxToNeg1Pos1(
            p_low=float(SCALER_PARAMS["p_low"]),
            p_high=float(SCALER_PARAMS["p_high"]),
        ).fit(logcounts)
        SCALER_CACHE[cache_key] = scaler

    base_batch = int(MODEL_PARAMS["batch_size"])
    batch_size = base_batch * (fast_batch_mult if fast_mode else 1)
    batch_size = max(1, min(batch_size, logcounts.shape[0]))

    bio_prob = p_bio.astype(np.float32)
    nonzero_mask = logcounts > 0.0

    use_full_gpu = (
        fast_mode
        and device.type == "cuda"
        and _gpu_can_fit(logcounts.size, torch.float32)
    )

    if use_full_gpu:
        log_t = torch.tensor(logcounts, dtype=torch.float32, device=device)
        lo = torch.tensor(scaler.lo_, dtype=torch.float32, device=device)
        hi = torch.tensor(scaler.hi_, dtype=torch.float32, device=device)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
        zmin = torch.tensor(scaler.zmin_, dtype=torch.float32, device=device)
        zspan = torch.tensor(scaler.zspan_, dtype=torch.float32, device=device)
        Xc = torch.minimum(torch.maximum(log_t, lo), hi)
        Z = (Xc - mean) / std
        X01 = (Z - zmin) / zspan
        Xtr = X01 * 2.0 - 1.0
        bio_mask = torch.tensor(bio_prob, dtype=torch.float32, device=device)
        nz_mask = torch.tensor(nonzero_mask.astype(np.float32), dtype=torch.float32, device=device)
    else:
        Xs = scaler.transform(logcounts).astype(np.float32)
        Xtr = torch.tensor(Xs, dtype=torch.float32)
        bio_mask = torch.tensor(bio_prob, dtype=torch.float32)
        nz_mask = torch.tensor(nonzero_mask.astype(np.float32), dtype=torch.float32)

    lo = torch.tensor(scaler.lo_, dtype=torch.float32, device=device)
    hi = torch.tensor(scaler.hi_, dtype=torch.float32, device=device)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
    zmin = torch.tensor(scaler.zmin_, dtype=torch.float32, device=device)
    zspan = torch.tensor(scaler.zspan_, dtype=torch.float32, device=device)
    zero_scaled = ((0.0 - scaler.mean_) / scaler.std_ - scaler.zmin_) / scaler.zspan_
    zero_scaled = zero_scaled * 2.0 - 1.0
    zero_scaled_t = torch.tensor(zero_scaled, dtype=torch.float32, device=device)
    counts_max_t = torch.tensor(np.maximum(counts_max, 1.0), dtype=torch.float32, device=device)

    if use_full_gpu:
        loader = None
    else:
        loader = DataLoader(
            TensorDataset(Xtr, bio_mask, nz_mask),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=bool(fast_mode),
            num_workers=int(num_workers) if fast_mode else 0,
            persistent_workers=bool(fast_mode) and int(num_workers) > 0,
        )

    if compile_enabled and amp_enabled:
        compile_enabled = False

    model = ImprovedAE(
        input_dim=logcounts.shape[1],
        hidden=MODEL_PARAMS["hidden"],
        bottleneck=int(MODEL_PARAMS["bottleneck"]),
        activation=str(MODEL_PARAMS.get("activation", "silu")),
        dropout=float(MODEL_PARAMS.get("dropout", 0.0)),
        layer_norm=bool(MODEL_PARAMS.get("layer_norm", True)),
        residual=bool(MODEL_PARAMS.get("residual", False)),
    ).to(device)
    if compile_enabled and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    opt_kwargs = dict(
        lr=float(AE_PARAMS["lr"]),
        weight_decay=float(MODEL_PARAMS["weight_decay"]),
    )
    try:
        opt = optim.Adam(model.parameters(), **opt_kwargs, fused=bool(fast_mode))
    except TypeError:
        opt = optim.Adam(model.parameters(), **opt_kwargs)

    scaler_amp = _make_grad_scaler(bool(amp_enabled) and device.type == "cuda")

    model.train()
    epochs = int(AE_PARAMS["epochs"])
    for _ in range(epochs):
        if use_full_gpu:
            idx = torch.randperm(Xtr.shape[0], device=device)
            for start in range(0, Xtr.shape[0], batch_size):
                batch_idx = idx[start : start + batch_size]
                xb = Xtr[batch_idx]
                bio_b = bio_mask[batch_idx]
                nz_b = nz_mask[batch_idx]
                mask_bio = torch.bernoulli(bio_b * float(AE_PARAMS["p_zero"]))
                mask_nz = torch.bernoulli(nz_b * float(AE_PARAMS["p_nz"]))
                x_in = xb
                if mask_nz.any():
                    x_in = torch.where(mask_nz.bool(), torch.zeros_like(x_in), x_in)
                if mask_bio.any():
                    noise_scale = torch.rand_like(xb) * float(AE_PARAMS["noise_max"])
                    noise_counts = noise_scale * counts_max_t
                    log2_base = float(np.log(2.0))
                    noise_vals = torch.log1p(noise_counts) / log2_base
                    noise_vals = torch.minimum(torch.maximum(noise_vals, lo), hi)
                    z = (noise_vals - mean) / std
                    x01 = (z - zmin) / zspan
                    noise_scaled = x01 * 2.0 - 1.0
                    x_in = torch.where(mask_bio.bool(), noise_scaled, x_in)

                opt.zero_grad()
                with _autocast_ctx(bool(amp_enabled), device):
                    recon = model(x_in)
                    residual = recon - xb
                    masked_loss = weighted_masked_mse(
                        residual,
                        mask_bio=mask_bio,
                        mask_nz=mask_nz,
                        weight_bio=float(AE_PARAMS["loss_bio_weight"]),
                        weight_nz=float(AE_PARAMS["loss_nz_weight"]),
                    )
                    bio_reg = ((recon - zero_scaled_t) ** 2 * bio_b).sum() / bio_b.sum().clamp_min(1.0)
                    loss = masked_loss + float(AE_PARAMS["bio_reg_weight"]) * bio_reg
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
        else:
            for xb, bio_b, nz_b in loader:
                xb = xb.to(device, non_blocking=True)
                bio_b = bio_b.to(device, non_blocking=True)
                nz_b = nz_b.to(device, non_blocking=True)

                mask_bio = torch.bernoulli(bio_b * float(AE_PARAMS["p_zero"]))
                mask_nz = torch.bernoulli(nz_b * float(AE_PARAMS["p_nz"]))

                x_in = xb
                if mask_nz.any():
                    x_in = torch.where(mask_nz.bool(), torch.zeros_like(x_in), x_in)
                if mask_bio.any():
                    noise_scale = torch.rand_like(xb) * float(AE_PARAMS["noise_max"])
                    noise_counts = noise_scale * counts_max_t
                    log2_base = float(np.log(2.0))
                    noise_vals = torch.log1p(noise_counts) / log2_base
                    noise_vals = torch.minimum(torch.maximum(noise_vals, lo), hi)
                    z = (noise_vals - mean) / std
                    x01 = (z - zmin) / zspan
                    noise_scaled = x01 * 2.0 - 1.0
                    x_in = torch.where(mask_bio.bool(), noise_scaled, x_in)

                opt.zero_grad()
                with _autocast_ctx(bool(amp_enabled), device):
                    recon = model(x_in)
                    residual = recon - xb
                    masked_loss = weighted_masked_mse(
                        residual,
                        mask_bio=mask_bio,
                        mask_nz=mask_nz,
                        weight_bio=float(AE_PARAMS["loss_bio_weight"]),
                        weight_nz=float(AE_PARAMS["loss_nz_weight"]),
                    )
                    bio_reg = ((recon - zero_scaled_t) ** 2 * bio_b).sum() / bio_b.sum().clamp_min(1.0)
                    loss = masked_loss + float(AE_PARAMS["bio_reg_weight"]) * bio_reg
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()

    model.eval()
    recon_list = []
    with torch.no_grad():
        if use_full_gpu:
            for i in range(0, Xtr.size(0), batch_size):
                xb = Xtr[i : i + batch_size]
                recon = model(xb)
                X01 = (recon + 1.0) / 2.0
                Z = X01 * zspan + zmin
                recon_orig = Z * std + mean
                recon_list.append(recon_orig.cpu().numpy())
        else:
            for i in range(0, Xtr.size(0), batch_size):
                xb = Xtr[i : i + batch_size].to(device)
                recon = model(xb)
                recon_np = recon.cpu().numpy()
                recon_orig = scaler.inverse_transform(recon_np)
                recon_list.append(recon_orig)
    recon_all = np.vstack(recon_list).astype(np.float32, copy=False)

    if int(AE_PARAMS.get("bio_reg_weight", 0.0)) >= 1:
        recon_all = _global_postprocess(recon_all, p_bio, logcounts)

    return recon_all.astype(np.float32, copy=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_overrides(target: Dict[str, object], updates: Dict[str, object]) -> None:
    for key, value in updates.items():
        if key not in target:
            continue
        target[key] = value


def _apply_env_config() -> None:
    raw = os.environ.get("MASKCLASS_CONFIG_JSON")
    if not raw:
        return
    try:
        cfg = json.loads(raw)
    except Exception:
        return
    if not isinstance(cfg, dict):
        return
    if "BIO_PARAMS" in cfg and isinstance(cfg["BIO_PARAMS"], dict):
        _apply_overrides(BIO_PARAMS, cfg["BIO_PARAMS"])
    if "MODEL_PARAMS" in cfg and isinstance(cfg["MODEL_PARAMS"], dict):
        _apply_overrides(MODEL_PARAMS, cfg["MODEL_PARAMS"])
    if "AE_PARAMS" in cfg and isinstance(cfg["AE_PARAMS"], dict):
        _apply_overrides(AE_PARAMS, cfg["AE_PARAMS"])
    if "SCALER_PARAMS" in cfg and isinstance(cfg["SCALER_PARAMS"], dict):
        _apply_overrides(SCALER_PARAMS, cfg["SCALER_PARAMS"])
    if "POSTPROCESS_PARAMS" in cfg and isinstance(cfg["POSTPROCESS_PARAMS"], dict):
        _apply_overrides(POSTPROCESS_PARAMS, cfg["POSTPROCESS_PARAMS"])


_apply_env_config()
