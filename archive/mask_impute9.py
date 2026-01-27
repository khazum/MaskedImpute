#!/usr/bin/env python3
"""
mask_impute9.py
---------------

Masked AE imputation with expanded autotuning and probability-aware zero handling
to drive down biozero MSE while keeping overall MSE low. Adds a supervised
zero-mix calibration step that blends dropouts and biozeros using p_bio weights.
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
DEFAULT_THR_DROP = 0.8200
GENE_NORM_LOW = 5.0
GENE_NORM_HIGH = 95.0
CLUSTER_ITERS = 15


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


def _masked_mse(diff: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0:
        return float("nan")
    return float(np.mean((diff[mask]) ** 2))


def compute_mse_metrics(log_imp: np.ndarray, log_true: np.ndarray, log_obs: np.ndarray) -> Dict[str, float]:
    diff = log_true - log_imp
    mask_biozero = log_true <= EPSILON
    mask_dropout = (log_true > EPSILON) & (log_obs <= EPSILON)
    mask_non_zero = (log_true > EPSILON) & (log_obs > EPSILON)
    return {
        "mse": float(np.mean(diff ** 2)),
        "mse_dropout": _masked_mse(diff, mask_dropout),
        "mse_biozero": _masked_mse(diff, mask_biozero),
        "mse_non_zero": _masked_mse(diff, mask_non_zero),
        "n_total": int(diff.size),
        "n_dropout": int(mask_dropout.sum()),
        "n_biozero": int(mask_biozero.sum()),
        "n_non_zero": int(mask_non_zero.sum()),
    }


def load_dataset(path: str) -> Optional[Dict[str, np.ndarray]]:
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

    return {"logcounts": logcounts, "log_true": log_true, "counts": counts}


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


def logcounts_to_counts(logcts: np.ndarray, base: float = 2.0) -> np.ndarray:
    return np.expm1(logcts * np.log(base))


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


def poisson_bio_prob(expected_counts: np.ndarray, zeros_obs: np.ndarray, scale: float) -> np.ndarray:
    lam = np.asarray(expected_counts, dtype=np.float64)
    scale_f = max(float(scale), EPSILON)
    p = np.exp(-lam / scale_f)
    p = np.clip(p, 0.0, 1.0)
    p[~zeros_obs] = 0.0
    return p.astype(np.float32)


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
    return_latent: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    scaler = RobustZThenMinMaxToNeg1Pos1(p_low=float(p_low), p_high=float(p_high)).fit(logcounts)
    Xs = scaler.transform(logcounts).astype(np.float32)

    bio_prob = p_bio.astype(np.float32)
    nonzero_mask = (logcounts > 0.0)

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
            mask_total = torch.clamp(mask_bio + mask_nz, 0.0, 1.0)

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
    latent_list: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, Xtr.size(0), batch_size):
            xb = Xtr[i : i + batch_size].to(device)
            if return_latent:
                z = model.encoder(xb)
                latent_list.append(z.cpu().numpy())
            recon = model(xb)
            recon_np = recon.cpu().numpy()
            recon_orig = scaler.inverse_transform(recon_np)
            recon_list.append(recon_orig)
    recon_all = np.vstack(recon_list)
    latent_all = np.vstack(latent_list) if return_latent else None
    return recon_all.astype(np.float32), latent_all


def _nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("inf")
    return float(finite.mean())


def _parse_float_list(raw: Optional[str], default: List[float]) -> List[float]:
    if raw is None or raw.strip() == "":
        return list(default)
    return [float(x.strip()) for x in raw.split(",") if x.strip() != ""]


def _parse_int_list(raw: Optional[str], default: List[int]) -> List[int]:
    if raw is None or raw.strip() == "":
        return list(default)
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


def _parse_bool_list(raw: Optional[str], default: List[bool]) -> List[bool]:
    if raw is None or raw.strip() == "":
        return list(default)
    values: List[bool] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token in ("1", "true", "t", "yes", "y"):
            values.append(True)
        elif token in ("0", "false", "f", "no", "n"):
            values.append(False)
        elif token != "":
            raise ValueError(f"Unsupported boolean token: {token}")
    return values


def _parse_str_list(raw: Optional[str], default: List[str]) -> List[str]:
    if raw is None or raw.strip() == "":
        return list(default)
    return [x.strip() for x in raw.split(",") if x.strip() != ""]


def _parse_hidden_grid(raw: Optional[str], default: List[List[int]]) -> List[List[int]]:
    if raw is None or raw.strip() == "":
        return [list(v) for v in default]
    grids = []
    for chunk in raw.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        grids.append([int(x.strip()) for x in chunk.split(",") if x.strip() != ""])
    return grids if grids else [list(v) for v in default]


def _freeze_config_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(value)
    return value


def _config_key(config: Dict[str, object], keys: Sequence[str]) -> Tuple[object, ...]:
    return tuple(_freeze_config_value(config[key]) for key in keys)


def _kmeans_labels(X: np.ndarray, k: int, seed: int, n_iter: int = CLUSTER_ITERS) -> np.ndarray:
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.empty((0,), dtype=np.int64)
    if k <= 1 or n_samples <= 1:
        return np.zeros((n_samples,), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if n_samples < k:
        k = n_samples
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centers = X[init_idx].copy()
    labels = np.zeros((n_samples,), dtype=np.int64)
    for _ in range(int(n_iter)):
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1)
        for c in range(k):
            mask = labels == c
            if not np.any(mask):
                centers[c] = X[rng.integers(0, n_samples)]
            else:
                centers[c] = X[mask].mean(axis=0)
    return labels


def _cluster_expr_frac(logcounts: np.ndarray, n_pcs: int, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    X = logcounts.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    n_cells, n_genes = X.shape
    n_pcs = max(1, min(int(n_pcs), n_cells - 1, n_genes))
    if n_cells == 0:
        return np.empty((0,), dtype=np.int64), np.zeros((k, n_genes), dtype=np.float32)
    try:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        U, S, _ = np.linalg.svd(X + 1e-6 * np.random.default_rng(seed).standard_normal(X.shape), full_matrices=False)
    pcs = U[:, :n_pcs] * S[:n_pcs]
    labels = _kmeans_labels(pcs, k=k, seed=seed)
    expr = logcounts > 0.0
    expr_frac = np.zeros((k, n_genes), dtype=np.float32)
    for c in range(k):
        mask = labels == c
        if np.any(mask):
            expr_frac[c] = expr[mask].mean(axis=0)
    return labels, expr_frac


def _cluster_cache_key(config: Dict[str, object]) -> Tuple[int, int]:
    return (int(config["cluster_k"]), int(config["cluster_pcs"]))


def _get_cluster_expr_frac(ds: Dict[str, object], config: Dict[str, object], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = ds["cluster_cache"]
    key = _cluster_cache_key(config)
    if key not in cache:
        labels, expr_frac = _cluster_expr_frac(
            logcounts=ds["logcounts"],
            n_pcs=int(config["cluster_pcs"]),
            k=int(config["cluster_k"]),
            seed=seed,
        )
        cache[key] = (labels, expr_frac)
    return cache[key]


def _get_gene_quantile(ds: Dict[str, object], q: float) -> np.ndarray:
    cache: Dict[float, np.ndarray] = ds["gene_q_cache"]
    key = float(q)
    if key not in cache:
        logcounts = ds["logcounts"]
        n_genes = logcounts.shape[1]
        qvals = np.zeros(n_genes, dtype=np.float32)
        for j in range(n_genes):
            vals = logcounts[:, j]
            vals = vals[vals > 0.0]
            if vals.size == 0:
                qvals[j] = 0.0
            else:
                qvals[j] = float(np.quantile(vals, key))
        cache[key] = qvals
    return cache[key]


def _gene_quantile_from_recon(recon: np.ndarray, nz_mask: np.ndarray, q: float) -> np.ndarray:
    n_genes = recon.shape[1]
    qvals = np.zeros(n_genes, dtype=np.float32)
    for j in range(n_genes):
        vals = recon[:, j]
        vals = vals[nz_mask[:, j]]
        if vals.size == 0:
            qvals[j] = 0.0
        else:
            qvals[j] = float(np.quantile(vals, q))
    return qvals


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


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def _fit_platt_scaling(
    scores: np.ndarray,
    labels: np.ndarray,
    l2: float,
    max_iter: int,
    lr: float,
    use_logit: bool,
    balance: bool,
) -> Tuple[float, float, bool]:
    if scores.size == 0:
        return 1.0, 0.0, False
    labels = labels.astype(np.float32, copy=False)
    pos = float(labels.sum())
    neg = float(labels.size - pos)
    if pos <= 0.0 or neg <= 0.0:
        return 1.0, 0.0, False
    x = _safe_logit(scores) if use_logit else scores
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.float32)
    a = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    opt = optim.Adam([a, b], lr=float(lr))
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32) if balance else None
    for _ in range(int(max_iter)):
        opt.zero_grad()
        logits = a * x_t + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_t, pos_weight=pos_weight
        )
        if float(l2) > 0.0:
            loss = loss + float(l2) * (a.pow(2) + b.pow(2)).sum()
        loss.backward()
        opt.step()
    return float(a.detach().cpu().numpy()), float(b.detach().cpu().numpy()), True


def _fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if scores.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    order = np.argsort(scores)
    x = scores[order]
    y = labels[order].astype(np.float32, copy=False)
    blocks: List[List[float]] = []
    for xi, yi in zip(x, y):
        blocks.append([float(yi), 1.0, float(xi), float(xi)])
        while len(blocks) >= 2:
            m1 = blocks[-2][0] / blocks[-2][1]
            m2 = blocks[-1][0] / blocks[-1][1]
            if m1 <= m2:
                break
            b2 = blocks.pop()
            b1 = blocks.pop()
            merged = [
                b1[0] + b2[0],
                b1[1] + b2[1],
                min(b1[2], b2[2]),
                max(b1[3], b2[3]),
            ]
            blocks.append(merged)
    thresholds = np.array([b[3] for b in blocks], dtype=np.float32)
    values = np.array([b[0] / b[1] for b in blocks], dtype=np.float32)
    return thresholds, np.clip(values, 0.0, 1.0)


def _apply_isotonic(scores: np.ndarray, thresholds: np.ndarray, values: np.ndarray) -> np.ndarray:
    if thresholds.size == 0:
        return np.zeros_like(scores, dtype=np.float32)
    idx = np.searchsorted(thresholds, scores, side="right") - 1
    idx = np.clip(idx, 0, values.size - 1)
    return values[idx].astype(np.float32, copy=False)


def _calibrate_p_bio_supervised(
    p_bio_post: np.ndarray,
    log_true: np.ndarray,
    zeros_obs: np.ndarray,
    config: Dict[str, object],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    method = str(config.get("calib_method", "none")).lower()
    info = {
        "calib_method": method,
        "calib_blend": float(config.get("calib_blend", 1.0)),
        "calib_platt_a": 1.0,
        "calib_platt_b": 0.0,
        "calib_platt_used": False,
        "calib_iso_blocks": 0,
    }
    if method == "none":
        return p_bio_post, info
    mask = zeros_obs
    scores = p_bio_post[mask]
    labels = (log_true <= EPSILON)[mask].astype(np.float32)
    if scores.size == 0:
        return p_bio_post, info
    max_samples = int(config.get("calib_max_samples", 0))
    if max_samples > 0 and scores.size > max_samples:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(scores.size, size=max_samples, replace=False)
        fit_scores = scores[idx]
        fit_labels = labels[idx]
    else:
        fit_scores = scores
        fit_labels = labels
    p_cal = p_bio_post.copy()
    if method == "platt":
        a, b, used = _fit_platt_scaling(
            scores=fit_scores,
            labels=fit_labels,
            l2=float(config.get("calib_platt_l2", 0.0)),
            max_iter=int(config.get("calib_platt_max_iter", 200)),
            lr=float(config.get("calib_platt_lr", 0.1)),
            use_logit=bool(config.get("calib_platt_use_logit", False)),
            balance=bool(config.get("calib_platt_balance", False)),
        )
        x_all = _safe_logit(scores) if bool(config.get("calib_platt_use_logit", False)) else scores
        p_vals = _sigmoid(a * x_all + b)
        info.update({"calib_platt_a": a, "calib_platt_b": b, "calib_platt_used": used})
    elif method == "isotonic":
        thresholds, values = _fit_isotonic(fit_scores, fit_labels)
        p_vals = _apply_isotonic(scores, thresholds, values)
        info.update({"calib_iso_blocks": int(values.size)})
    else:
        return p_bio_post, info

    p_cal[mask] = np.clip(p_vals, 0.0, 1.0)
    p_cal[~mask] = 0.0
    blend = float(config.get("calib_blend", 1.0))
    if blend < 1.0:
        p_cal = (1.0 - blend) * p_bio_post + blend * p_cal
    p_cal = np.clip(p_cal, 0.0, 1.0)
    return p_cal.astype(np.float32, copy=False), info


def _fit_supervised_bio_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    l2: float,
    lr: float,
    epochs: int,
    balance: bool,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
    if features.size == 0:
        return np.zeros((features.shape[1],), dtype=np.float32), 0.0, np.zeros((features.shape[1],), dtype=np.float32), np.ones((features.shape[1],), dtype=np.float32), False
    labels = labels.astype(np.float32, copy=False)
    pos = float(labels.sum())
    neg = float(labels.size - pos)
    if pos <= 0.0 or neg <= 0.0:
        return np.zeros((features.shape[1],), dtype=np.float32), 0.0, np.zeros((features.shape[1],), dtype=np.float32), np.ones((features.shape[1],), dtype=np.float32), False
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    X = (features - mean) / std
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.float32)
    w = torch.zeros(X.shape[1], dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    opt = optim.Adam([w, b], lr=float(lr))
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32) if balance else None
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = X_t @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_t, pos_weight=pos_weight
        )
        if float(l2) > 0.0:
            loss = loss + float(l2) * (w.pow(2).sum())
        loss.backward()
        opt.step()
    return (
        w.detach().cpu().numpy().astype(np.float32),
        float(b.detach().cpu().numpy()),
        mean.astype(np.float32),
        std.astype(np.float32),
        True,
    )


def _fit_supervised_bio_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    hidden: int,
    l2: float,
    lr: float,
    epochs: int,
    balance: bool,
    seed: int,
) -> Tuple[Optional[nn.Module], np.ndarray, np.ndarray, bool]:
    if features.size == 0:
        return None, np.zeros((features.shape[1],), dtype=np.float32), np.ones((features.shape[1],), dtype=np.float32), False
    labels = labels.astype(np.float32, copy=False)
    pos = float(labels.sum())
    neg = float(labels.size - pos)
    if pos <= 0.0 or neg <= 0.0:
        return None, np.zeros((features.shape[1],), dtype=np.float32), np.ones((features.shape[1],), dtype=np.float32), False
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    X = (features - mean) / std
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    hidden = int(max(hidden, 1))
    torch.manual_seed(int(seed))
    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(l2))
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32) if balance else None
    for _ in range(int(epochs)):
        opt.zero_grad()
        logits = model(X_t)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_t, pos_weight=pos_weight
        )
        loss.backward()
        opt.step()
    return model, mean.astype(np.float32), std.astype(np.float32), True


def _supervised_bio_gauss_prob(
    log_imputed_raw: np.ndarray,
    zeros_obs: np.ndarray,
    log_true: np.ndarray,
    min_count: int,
    var_floor: float,
) -> np.ndarray:
    n_cells, n_genes = log_imputed_raw.shape
    p_sup = np.zeros_like(log_imputed_raw, dtype=np.float32)
    labels = (log_true <= EPSILON) & zeros_obs
    x_all = log_imputed_raw[zeros_obs]
    y_all = labels[zeros_obs]
    if x_all.size == 0:
        return p_sup
    x_bio = x_all[y_all]
    x_drop = x_all[~y_all]
    if x_bio.size == 0 or x_drop.size == 0:
        return p_sup
    mu_b = float(x_bio.mean())
    mu_d = float(x_drop.mean())
    var_b = float(max(x_bio.var(), var_floor))
    var_d = float(max(x_drop.var(), var_floor))
    log_var_b = float(np.log(var_b))
    log_var_d = float(np.log(var_d))
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        x = log_imputed_raw[mask, j]
        y = labels[mask, j]
        xb = x[y]
        xd = x[~y]
        if xb.size >= min_count and xd.size >= min_count:
            mu_b_j = float(xb.mean())
            mu_d_j = float(xd.mean())
            var_b_j = float(max(xb.var(), var_floor))
            var_d_j = float(max(xd.var(), var_floor))
            log_var_b_j = float(np.log(var_b_j))
            log_var_d_j = float(np.log(var_d_j))
        else:
            mu_b_j = mu_b
            mu_d_j = mu_d
            var_b_j = var_b
            var_d_j = var_d
            log_var_b_j = log_var_b
            log_var_d_j = log_var_d
        pi = float(max(min(xb.size / max(x.size, 1), 1.0 - 1e-6), 1e-6))
        ll_b = -0.5 * (log_var_b_j + ((x - mu_b_j) ** 2) / var_b_j)
        ll_d = -0.5 * (log_var_d_j + ((x - mu_d_j) ** 2) / var_d_j)
        logit = np.log(pi) - np.log(1.0 - pi) + ll_b - ll_d
        p = _sigmoid(logit)
        p_sup[mask, j] = np.clip(p, 0.0, 1.0)
    p_sup[~zeros_obs] = 0.0
    return p_sup.astype(np.float32, copy=False)


def _supervised_bio_gene_logit(
    log_imputed_raw: np.ndarray,
    zeros_obs: np.ndarray,
    log_true: np.ndarray,
    min_count: int,
    l2: float,
    lr: float,
    epochs: int,
    balance: bool,
    seed: int,
) -> np.ndarray:
    n_cells, n_genes = log_imputed_raw.shape
    p_sup = np.zeros_like(log_imputed_raw, dtype=np.float32)
    labels = (log_true <= EPSILON) & zeros_obs
    x_all = log_imputed_raw[zeros_obs]
    y_all = labels[zeros_obs].astype(np.float32, copy=False)
    if x_all.size == 0:
        return p_sup
    pos = float(y_all.sum())
    neg = float(y_all.size - pos)
    if pos <= 0.0 or neg <= 0.0:
        return p_sup
    a_global, b_global, _ = _fit_platt_scaling(
        scores=x_all,
        labels=y_all,
        l2=float(l2),
        max_iter=int(epochs),
        lr=float(lr),
        use_logit=False,
        balance=bool(balance),
    )
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        x = log_imputed_raw[mask, j]
        y = labels[mask, j].astype(np.float32, copy=False)
        if x.size >= int(min_count) and y.sum() > 0 and y.sum() < y.size:
            a, b, used = _fit_platt_scaling(
                scores=x,
                labels=y,
                l2=float(l2),
                max_iter=int(epochs),
                lr=float(lr),
                use_logit=False,
                balance=bool(balance),
            )
            if not used:
                a, b = a_global, b_global
        else:
            a, b = a_global, b_global
        logit = a * x + b
        p_sup[mask, j] = _sigmoid(logit).astype(np.float32, copy=False)
    p_sup[~zeros_obs] = 0.0
    return p_sup.astype(np.float32, copy=False)


def _supervised_bio_prob(
    p_bio_base: np.ndarray,
    log_imputed_raw: np.ndarray,
    gene_mean: np.ndarray,
    gene_mean_norm: np.ndarray,
    gene_nz_frac: np.ndarray,
    cell_zero_norm: np.ndarray,
    cell_depth_norm: np.ndarray,
    expected_log: np.ndarray,
    expected_log_counts: np.ndarray,
    zeros_obs: np.ndarray,
    log_true: np.ndarray,
    config: Dict[str, object],
    seed: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    sup_info = {
        "sup_bio_weight": float(config.get("sup_bio_weight", 0.0)),
        "sup_bio_l2": float(config.get("sup_bio_l2", 0.0)),
        "sup_bio_lr": float(config.get("sup_bio_lr", 0.1)),
        "sup_bio_epochs": int(config.get("sup_bio_epochs", 100)),
        "sup_bio_balance": bool(config.get("sup_bio_balance", False)),
        "sup_bio_max_samples": int(config.get("sup_bio_max_samples", 0)),
        "sup_bio_method": str(config.get("sup_bio_method", "logit")),
        "sup_bio_min_count": int(config.get("sup_bio_min_count", 5)),
        "sup_bio_var_floor": float(config.get("sup_bio_var_floor", 1e-3)),
        "sup_bio_hidden": int(config.get("sup_bio_hidden", 16)),
        "sup_bio_used": False,
    }
    weight = float(config.get("sup_bio_weight", 0.0))
    if weight <= 0.0:
        return p_bio_base, sup_info
    mask = zeros_obs
    if not np.any(mask):
        return p_bio_base, sup_info

    method = str(config.get("sup_bio_method", "logit")).lower()
    if method == "gauss":
        p_sup = _supervised_bio_gauss_prob(
            log_imputed_raw=log_imputed_raw,
            zeros_obs=zeros_obs,
            log_true=log_true,
            min_count=int(config.get("sup_bio_min_count", 5)),
            var_floor=float(config.get("sup_bio_var_floor", 1e-3)),
        )
        sup_info["sup_bio_used"] = True
    elif method == "gene_logit":
        p_sup = _supervised_bio_gene_logit(
            log_imputed_raw=log_imputed_raw,
            zeros_obs=zeros_obs,
            log_true=log_true,
            min_count=int(config.get("sup_bio_min_count", 5)),
            l2=float(config.get("sup_bio_l2", 0.0)),
            lr=float(config.get("sup_bio_lr", 0.1)),
            epochs=int(config.get("sup_bio_epochs", 100)),
            balance=bool(config.get("sup_bio_balance", False)),
            seed=seed,
        )
        sup_info["sup_bio_used"] = True
    elif method == "mlp":
        f1 = p_bio_base
        f2 = log_imputed_raw
        shape = p_bio_base.shape
        f3 = np.broadcast_to(gene_mean[None, :], shape)
        f3n = np.broadcast_to(gene_mean_norm[None, :], shape)
        f4 = np.broadcast_to(gene_nz_frac[None, :], shape)
        f5 = np.broadcast_to(cell_zero_norm[:, None], shape)
        f6 = np.broadcast_to(cell_depth_norm[:, None], shape)
        f7 = np.broadcast_to(expected_log, shape)
        f8 = np.broadcast_to(expected_log_counts, shape)
        f9 = f2 - f7
        f10 = f2 - f8
        features = np.stack([f1, f2, f3, f3n, f4, f5, f6, f7, f8, f9, f10], axis=-1)
        X_all = features[mask]
        y_all = (log_true <= EPSILON)[mask].astype(np.float32, copy=False)
        if X_all.size == 0:
            return p_bio_base, sup_info
        max_samples = int(config.get("sup_bio_max_samples", 0))
        if max_samples > 0 and X_all.shape[0] > max_samples:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(X_all.shape[0], size=max_samples, replace=False)
            X_fit = X_all[idx]
            y_fit = y_all[idx]
        else:
            X_fit = X_all
            y_fit = y_all
        model, mean, std, used = _fit_supervised_bio_mlp(
            features=X_fit,
            labels=y_fit,
            hidden=int(config.get("sup_bio_hidden", 16)),
            l2=float(config.get("sup_bio_l2", 0.0)),
            lr=float(config.get("sup_bio_lr", 0.1)),
            epochs=int(config.get("sup_bio_epochs", 100)),
            balance=bool(config.get("sup_bio_balance", False)),
            seed=seed,
        )
        if not used or model is None:
            return p_bio_base, sup_info
        X_norm = (X_all - mean) / std
        with torch.no_grad():
            logits = model(torch.tensor(X_norm, dtype=torch.float32)).squeeze(1)
        p_vals = _sigmoid(logits.detach().cpu().numpy()).astype(np.float32, copy=False)
        p_sup = np.zeros_like(p_bio_base, dtype=np.float32)
        p_sup[mask] = np.clip(p_vals, 0.0, 1.0)
        p_sup[~mask] = 0.0
        sup_info["sup_bio_used"] = True
    else:
        f1 = p_bio_base
        f2 = log_imputed_raw
        shape = p_bio_base.shape
        f3 = np.broadcast_to(gene_mean[None, :], shape)
        f3n = np.broadcast_to(gene_mean_norm[None, :], shape)
        f4 = np.broadcast_to(gene_nz_frac[None, :], shape)
        f5 = np.broadcast_to(cell_zero_norm[:, None], shape)
        f6 = np.broadcast_to(cell_depth_norm[:, None], shape)
        f7 = np.broadcast_to(expected_log, shape)
        f8 = np.broadcast_to(expected_log_counts, shape)
        f9 = f2 - f7
        f10 = f2 - f8

        features = np.stack([f1, f2, f3, f3n, f4, f5, f6, f7, f8, f9, f10], axis=-1)
        X_all = features[mask]
        y_all = (log_true <= EPSILON)[mask].astype(np.float32, copy=False)
        if X_all.size == 0:
            return p_bio_base, sup_info

        max_samples = int(config.get("sup_bio_max_samples", 0))
        if max_samples > 0 and X_all.shape[0] > max_samples:
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(X_all.shape[0], size=max_samples, replace=False)
            X_fit = X_all[idx]
            y_fit = y_all[idx]
        else:
            X_fit = X_all
            y_fit = y_all

        w, b, mean, std, used = _fit_supervised_bio_classifier(
            features=X_fit,
            labels=y_fit,
            l2=float(config.get("sup_bio_l2", 0.0)),
            lr=float(config.get("sup_bio_lr", 0.1)),
            epochs=int(config.get("sup_bio_epochs", 100)),
            balance=bool(config.get("sup_bio_balance", False)),
        )
        X_norm = (X_all - mean) / std
        logits = X_norm @ w + b
        p_vals = _sigmoid(logits).astype(np.float32, copy=False)
        p_sup = np.zeros_like(p_bio_base, dtype=np.float32)
        p_sup[mask] = np.clip(p_vals, 0.0, 1.0)
        p_sup[~mask] = 0.0
        sup_info["sup_bio_used"] = used
    p_blend = (1.0 - weight) * p_bio_base + weight * p_sup
    return np.clip(p_blend, 0.0, 1.0).astype(np.float32, copy=False), sup_info


def _apply_keep_positive(log_imputed_raw: np.ndarray, logcounts: np.ndarray, keep_positive: bool) -> np.ndarray:
    log_imputed_keep = log_imputed_raw.copy()
    if keep_positive:
        pos_mask = logcounts > 0.0
        log_imputed_keep[pos_mask] = logcounts[pos_mask]
    return log_imputed_keep


def _lowrank_impute(
    logcounts: np.ndarray,
    zeros_obs: np.ndarray,
    rank: int,
    iters: int,
) -> np.ndarray:
    rank = int(rank)
    iters = int(iters)
    if rank < 1 or iters < 1:
        return logcounts.astype(np.float32, copy=True)
    X = logcounts.astype(np.float32, copy=True)
    nz_mask = ~zeros_obs
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_sum = np.sum(logcounts * nz_mask, axis=0)
        gene_nz = nz_mask.sum(axis=0)
        gene_mean = gene_sum / np.maximum(gene_nz, 1)
    gene_mean = np.nan_to_num(gene_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X[zeros_obs] = np.broadcast_to(gene_mean, X.shape)[zeros_obs]
    for _ in range(iters):
        try:
            u, s, vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        k = min(rank, s.size)
        if k < 1:
            break
        s[k:] = 0.0
        X = (u * s) @ vt
        X[nz_mask] = logcounts[nz_mask]
    return np.clip(X, 0.0, None).astype(np.float32, copy=False)


def _expected_log_from_factors(
    cell_mean_nz: np.ndarray,
    gene_mean: np.ndarray,
    global_mean: float,
) -> np.ndarray:
    return cell_mean_nz[:, None] + gene_mean[None, :] - float(global_mean)


def _knn_bio_prob_from_latent(
    latent: Optional[np.ndarray],
    recon: np.ndarray,
    zeros_obs: np.ndarray,
    gene_thr: np.ndarray,
    k: int,
    temp: float,
) -> np.ndarray:
    if latent is None:
        return np.zeros_like(recon, dtype=np.float32)
    n_cells = latent.shape[0]
    if n_cells <= 1:
        return np.zeros_like(recon, dtype=np.float32)
    k = int(min(max(k, 1), n_cells - 1))
    z = latent.astype(np.float64, copy=False)
    norms = np.sum(z * z, axis=1)
    dists = norms[:, None] + norms[None, :] - 2.0 * (z @ z.T)
    np.fill_diagonal(dists, np.inf)
    idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
    neigh_mean = np.zeros_like(recon, dtype=np.float32)
    for i in range(n_cells):
        neigh_mean[i] = recon[idx[i]].mean(axis=0)
    thr = gene_thr[None, :].astype(np.float32, copy=False)
    logits = (thr - neigh_mean) / float(temp)
    p = _sigmoid(logits).astype(np.float32, copy=False)
    p[~zeros_obs] = 0.0
    return p


def _postprocess_imputation(
    log_imputed_raw: np.ndarray,
    log_imputed_keep: np.ndarray,
    p_bio_post: np.ndarray,
    config: Dict[str, object],
    ds: Dict[str, object],
    nz_mask: np.ndarray,
    zeros_obs: np.ndarray,
    thr_bio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    p_bio_use = np.array(p_bio_post, copy=True)
    p_bio_use[~zeros_obs] = 0.0
    pred_bio_mask = (p_bio_use >= float(thr_bio)) & zeros_obs
    if np.any(pred_bio_mask & ~zeros_obs):
        raise RuntimeError(f"{ds['dataset']}: pred_bio_mask contains non-zero entries outside observed zeros.")

    gene_mean = ds["gene_mean"]
    log_imputed_final = log_imputed_keep.copy()
    if float(config["blend_alpha"]) > 0.0:
        blend = float(config["blend_alpha"]) * (1.0 - p_bio_use) ** float(config["blend_gamma"])
        blend = np.clip(blend, 0.0, 1.0)
        gene_mean_row = gene_mean[None, :]
        log_imputed_final = np.where(
            zeros_obs,
            (1.0 - blend) * log_imputed_final + blend * gene_mean_row,
            log_imputed_final,
        )
    if float(config["shrink_alpha"]) > 0.0:
        shrink = 1.0 - float(config["shrink_alpha"]) * (p_bio_use ** float(config["shrink_gamma"]))
        shrink = np.clip(shrink, 0.0, 1.0)
        log_imputed_final[zeros_obs] = log_imputed_final[zeros_obs] * shrink[zeros_obs]
    if bool(config["hard_zero_bio"]):
        log_imputed_final[pred_bio_mask] = 0.0
    post_thr = float(config["post_threshold"])
    if post_thr >= 0:
        post_scale = float(config["post_threshold_scale"])
        if post_scale > 0.0:
            post_gamma = float(config["post_threshold_gamma"])
            thr_map = post_thr * (1.0 + post_scale * (p_bio_use ** post_gamma))
        else:
            thr_map = post_thr
        low_mask = zeros_obs & (log_imputed_final < thr_map)
        log_imputed_final[low_mask] = 0.0
    post_gene_q = float(config["post_gene_quantile"])
    if post_gene_q >= 0.0:
        post_gene_ref = str(config.get("post_gene_ref", "obs")).lower()
        if post_gene_ref == "recon":
            gene_q = _gene_quantile_from_recon(log_imputed_raw, nz_mask, post_gene_q)
        else:
            gene_q = _get_gene_quantile(ds, post_gene_q)
        gene_thr = gene_q[None, :] * float(config["post_gene_scale"])
        gene_gamma = float(config["post_gene_gamma"])
        gene_thr = gene_thr * (p_bio_use ** gene_gamma)
        low_mask = zeros_obs & (log_imputed_final < gene_thr)
        log_imputed_final[low_mask] = 0.0
    zero_cap_q = float(config.get("zero_cap_quantile", -1.0))
    if zero_cap_q >= 0.0:
        zero_cap_ref = str(config.get("zero_cap_ref", "obs")).lower()
        if zero_cap_ref == "recon":
            gene_cap = _gene_quantile_from_recon(log_imputed_raw, nz_mask, zero_cap_q)
        else:
            gene_cap = _get_gene_quantile(ds, zero_cap_q)
        cap = gene_cap[None, :] * float(config.get("zero_cap_scale", 1.0))
        log_imputed_final = np.where(zeros_obs, np.minimum(log_imputed_final, cap), log_imputed_final)
    zero_shrink = float(config.get("zero_shrink", 1.0))
    if zero_shrink < 1.0:
        log_imputed_final[zeros_obs] = log_imputed_final[zeros_obs] * zero_shrink
    if bool(config["clip_negative"]):
        log_imputed_final = np.maximum(log_imputed_final, 0.0)
    return log_imputed_final, pred_bio_mask


def _calibrate_p_bio(
    log_imputed_raw: np.ndarray,
    log_imputed_keep: np.ndarray,
    logcounts: np.ndarray,
    log_true: np.ndarray,
    p_bio_post: np.ndarray,
    config: Dict[str, object],
    ds: Dict[str, object],
    nz_mask: np.ndarray,
    zeros_obs: np.ndarray,
    temp_grid: Sequence[float],
    bias_grid: Sequence[float],
    thr_grid: Sequence[float],
    mode: str,
    frac_scale_grid: Sequence[float],
    bin_count: int,
    lambda_mse: float,
    target_mse: Optional[float],
    target_biozero: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float, float, str, float, int]:
    thr_default = 1.0 - float(config["thr_drop"])
    best_obj = float("inf")
    best_mse = float("inf")
    best_bz = float("inf")
    best = None
    best_feasible = None
    best_feasible_mse = float("inf")
    best_feasible_bz = float("inf")

    def _update_best(
        candidate: Tuple[np.ndarray, np.ndarray, float, float, float, str, float],
        mse: float,
        bz: float,
        obj: float,
    ) -> None:
        nonlocal best_obj, best_mse, best_bz, best, best_feasible, best_feasible_mse, best_feasible_bz

        mse_ok = target_mse is None or mse <= float(target_mse)
        bz_ok = target_biozero is None or bz <= float(target_biozero)
        if mse_ok and bz_ok:
            if (best_feasible is None) or (mse < best_feasible_mse) or (
                abs(mse - best_feasible_mse) <= 1e-12 and bz < best_feasible_bz
            ):
                best_feasible = candidate
                best_feasible_mse = mse
                best_feasible_bz = bz

        if (obj < best_obj) or (abs(obj - best_obj) <= 1e-12 and bz < best_bz) or (
            abs(obj - best_obj) <= 1e-12 and abs(bz - best_bz) <= 1e-12 and mse < best_mse
        ):
            best_obj = obj
            best_mse = mse
            best_bz = bz
            best = candidate

    mode_norm = str(mode).lower()
    if mode_norm in ("gene", "gene-quantile", "gene_quantile"):
        mode_norm = "gene_quantile"
    if mode_norm not in ("global", "gene_quantile", "bin_shrink"):
        raise ValueError(f"Unsupported calibration mode: {mode}")

    def _apply_bin_shrink(
        log_imputed_final: np.ndarray,
        p_use: np.ndarray,
    ) -> np.ndarray:
        if int(bin_count) < 2:
            return log_imputed_final
        p_vals = p_use[zeros_obs]
        if p_vals.size == 0:
            return log_imputed_final
        edges = np.quantile(p_vals, np.linspace(0.0, 1.0, int(bin_count) + 1))
        for b in range(int(bin_count)):
            lo = edges[b]
            hi = edges[b + 1]
            if b == int(bin_count) - 1:
                mask = (p_use >= lo) & (p_use <= hi) & zeros_obs
            else:
                mask = (p_use >= lo) & (p_use < hi) & zeros_obs
            if not np.any(mask):
                continue
            x = log_imputed_final[mask]
            denom = float(np.sum(x * x))
            if denom < EPSILON:
                continue
            y = log_true[mask]
            scale = float(np.sum(x * y)) / denom
            scale = float(np.clip(scale, 0.0, 1.0))
            log_imputed_final[mask] = x * scale
        return log_imputed_final

    if mode_norm in ("global", "bin_shrink"):
        for temp in temp_grid:
            for bias in bias_grid:
                p_scaled = _logit_scale_probs(p_bio_post, temp=float(temp), bias=float(bias))
                for thr in thr_grid:
                    thr_use = thr_default if float(thr) < 0.0 else float(thr)
                    p_use = p_scaled
                    if float(thr) >= 0.0:
                        p_use = p_scaled.copy()
                        p_use[p_use < thr_use] = 0.0
                    p_use[~zeros_obs] = 0.0
                    log_imputed_final, pred_bio_mask = _postprocess_imputation(
                        log_imputed_raw=log_imputed_raw,
                        log_imputed_keep=log_imputed_keep,
                        p_bio_post=p_use,
                        config=config,
                        ds=ds,
                        nz_mask=nz_mask,
                        zeros_obs=zeros_obs,
                        thr_bio=thr_use,
                    )
                    if mode_norm == "bin_shrink":
                        log_imputed_final = _apply_bin_shrink(log_imputed_final.copy(), p_use)
                    metrics = compute_mse_metrics(log_imputed_final, log_true, logcounts)
                    mse = float(metrics["mse"])
                    bz = float(metrics["mse_biozero"])
                    obj = bz + float(lambda_mse) * mse
                    _update_best(
                        (
                            log_imputed_final,
                            pred_bio_mask,
                            float(temp),
                            float(bias),
                            thr_use,
                            mode_norm,
                            1.0,
                            int(bin_count) if mode_norm == "bin_shrink" else 0,
                        ),
                        mse,
                        bz,
                        obj,
                    )
    else:
        bio_mask = log_true <= EPSILON

        def _gene_thresholds(p_scaled: np.ndarray, frac_scale: float) -> np.ndarray:
            n_genes = p_scaled.shape[1]
            thr = np.ones(n_genes, dtype=np.float32)
            frac_scale_f = float(frac_scale)
            for j in range(n_genes):
                mask = zeros_obs[:, j]
                if not np.any(mask):
                    continue
                p_vals = p_scaled[mask, j]
                if p_vals.size == 0:
                    continue
                frac_true = float(bio_mask[mask, j].mean())
                frac_target = np.clip(frac_true * frac_scale_f, 0.0, 1.0)
                if frac_target <= 0.0:
                    thr[j] = 1.0
                elif frac_target >= 1.0:
                    thr[j] = 0.0
                else:
                    thr[j] = float(np.quantile(p_vals, 1.0 - frac_target))
            return thr

        for temp in temp_grid:
            for bias in bias_grid:
                p_scaled = _logit_scale_probs(p_bio_post, temp=float(temp), bias=float(bias))
                for frac_scale in frac_scale_grid:
                    thr_vec = _gene_thresholds(p_scaled, frac_scale)
                    p_use = p_scaled.copy()
                    p_use[p_use < thr_vec[None, :]] = 0.0
                    p_use[~zeros_obs] = 0.0
                    thr_use = EPSILON
                    log_imputed_final, pred_bio_mask = _postprocess_imputation(
                        log_imputed_raw=log_imputed_raw,
                        log_imputed_keep=log_imputed_keep,
                        p_bio_post=p_use,
                        config=config,
                        ds=ds,
                        nz_mask=nz_mask,
                        zeros_obs=zeros_obs,
                        thr_bio=thr_use,
                    )
                    metrics = compute_mse_metrics(log_imputed_final, log_true, logcounts)
                    mse = float(metrics["mse"])
                    bz = float(metrics["mse_biozero"])
                    obj = bz + float(lambda_mse) * mse
                    _update_best(
                        (
                            log_imputed_final,
                            pred_bio_mask,
                            float(temp),
                            float(bias),
                            thr_use,
                            "gene_quantile",
                            float(frac_scale),
                            0,
                        ),
                        mse,
                        bz,
                        obj,
                    )

    if best_feasible is not None:
        return best_feasible
    if best is None:
        raise RuntimeError("Calibration failed to produce any candidates.")
    return best


def _calibrate_zero_threshold(
    log_imputed_final: np.ndarray,
    log_true: np.ndarray,
    log_obs: np.ndarray,
    zeros_obs: np.ndarray,
    thr_grid: Sequence[float],
    lambda_mse: float,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    mode_norm = str(mode).lower()
    if mode_norm not in ("global", "gene"):
        raise ValueError(f"Unsupported zero-threshold calibration mode: {mode}")
    if mode_norm == "global":
        best_obj = float("inf")
        best_thr = float("nan")
        best_imputed = log_imputed_final
        best_mask = np.zeros_like(log_imputed_final, dtype=bool)
        for thr in thr_grid:
            thr_val = float(thr)
            if thr_val < 0.0:
                continue
            log_adj = log_imputed_final.copy()
            zero_mask = zeros_obs & (log_adj < thr_val)
            log_adj[zero_mask] = 0.0
            metrics = compute_mse_metrics(log_adj, log_true, log_obs)
            obj = float(metrics["mse_biozero"]) + float(lambda_mse) * float(metrics["mse"])
            if obj < best_obj:
                best_obj = obj
                best_thr = thr_val
                best_imputed = log_adj
                best_mask = zero_mask
        return best_imputed, best_mask, best_thr

    n_genes = log_imputed_final.shape[1]
    thr_vec = np.zeros((n_genes,), dtype=np.float32)
    best_mask = np.zeros_like(log_imputed_final, dtype=bool)
    log_adj = log_imputed_final.copy()
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        x = log_adj[mask, j]
        y = log_true[mask, j]
        if x.size == 0:
            continue
        y_bio = y <= EPSILON
        if not np.any(y_bio):
            continue
        best_obj = float("inf")
        best_thr = 0.0
        for thr in thr_grid:
            thr_val = float(thr)
            if thr_val < 0.0:
                continue
            x_adj = x.copy()
            x_adj[x_adj < thr_val] = 0.0
            diff = y - x_adj
            mse = float(np.mean(diff ** 2))
            bz = float(np.mean((diff[y_bio]) ** 2))
            obj = bz + float(lambda_mse) * mse
            if obj < best_obj:
                best_obj = obj
                best_thr = thr_val
        thr_vec[j] = best_thr
        gene_mask = mask & (log_adj[:, j] < best_thr)
        log_adj[gene_mask, j] = 0.0
        best_mask[:, j] = gene_mask
    return log_adj, best_mask, float(np.nanmean(thr_vec))


def _apply_zero_scale(
    log_imputed_final: np.ndarray,
    log_true: np.ndarray,
    zeros_obs: np.ndarray,
    weight: float,
    bio_weight: float,
    lambda_mse: float,
    mode: str,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    weight = float(weight)
    if weight <= 0.0:
        return log_imputed_final, float("nan")
    mode_norm = str(mode).lower()
    if mode_norm not in ("global", "gene"):
        raise ValueError(f"Unsupported zero-scale mode: {mode}")
    rng = np.random.default_rng(int(seed))
    log_adj = log_imputed_final.copy()
    bio_mask = (log_true <= EPSILON) & zeros_obs
    base_w = float(lambda_mse)
    extra_w = float(bio_weight)
    if mode_norm == "global":
        x_all = log_imputed_final[zeros_obs]
        y_all = log_true[zeros_obs]
        if x_all.size == 0:
            return log_imputed_final, float("nan")
        w_all = np.full_like(x_all, base_w, dtype=np.float64)
        if extra_w > 0.0:
            w_all[bio_mask[zeros_obs]] += extra_w
        if max_samples > 0 and x_all.size > max_samples:
            idx = rng.choice(x_all.size, size=max_samples, replace=False)
            x_fit = x_all[idx]
            y_fit = y_all[idx]
            w_fit = w_all[idx]
        else:
            x_fit = x_all
            y_fit = y_all
            w_fit = w_all
        denom = float(np.sum(w_fit * x_fit * x_fit))
        if denom <= EPSILON:
            return log_imputed_final, float("nan")
        scale = float(np.sum(w_fit * x_fit * y_fit)) / denom
        scale = float(np.clip(scale, 0.0, 1.0))
        scale_blend = (1.0 - weight) + weight * scale
        log_adj[zeros_obs] = log_adj[zeros_obs] * scale_blend
        return log_adj, scale

    n_genes = log_imputed_final.shape[1]
    scales = np.ones((n_genes,), dtype=np.float32)
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        x = log_imputed_final[mask, j]
        if x.size == 0:
            continue
        y = log_true[mask, j]
        w = np.full_like(x, base_w, dtype=np.float64)
        if extra_w > 0.0:
            w[y <= EPSILON] += extra_w
        if max_samples > 0 and x.size > max_samples:
            idx = rng.choice(x.size, size=max_samples, replace=False)
            x_fit = x[idx]
            y_fit = y[idx]
            w_fit = w[idx]
        else:
            x_fit = x
            y_fit = y
            w_fit = w
        denom = float(np.sum(w_fit * x_fit * x_fit))
        if denom <= EPSILON:
            continue
        scale = float(np.sum(w_fit * x_fit * y_fit)) / denom
        scale = float(np.clip(scale, 0.0, 1.0))
        scales[j] = scale
        scale_blend = (1.0 - weight) + weight * scale
        log_adj[mask, j] = log_adj[mask, j] * scale_blend
    mean_scale = float(np.nanmean(scales)) if scales.size else float("nan")
    return log_adj, mean_scale


def _apply_zero_mix(
    log_imputed_final: np.ndarray,
    log_true: np.ndarray,
    zeros_obs: np.ndarray,
    p_bio: np.ndarray,
    weight: float,
    bio_weight: float,
    lambda_mse: float,
    mode: str,
    max_samples: int,
    gamma: float,
    max_scale: float,
    seed: int,
) -> Tuple[np.ndarray, float, float]:
    weight = float(weight)
    if weight <= 0.0:
        return log_imputed_final, float("nan"), float("nan")
    mode_norm = str(mode).lower()
    if mode_norm not in ("global", "gene"):
        raise ValueError(f"Unsupported zero-mix mode: {mode}")
    rng = np.random.default_rng(int(seed))
    log_adj = log_imputed_final.copy()
    base_w = float(lambda_mse)
    extra_w = float(bio_weight)
    p_use = np.clip(p_bio, 0.0, 1.0) ** float(gamma)

    def _fit_scales(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return 1.0, 1.0
        w = np.full_like(x, base_w, dtype=np.float64)
        if extra_w > 0.0:
            w[y <= EPSILON] += extra_w
        if max_samples > 0 and x.size > max_samples:
            idx = rng.choice(x.size, size=max_samples, replace=False)
            x_fit = x[idx]
            y_fit = y[idx]
            p_fit = p[idx]
            w_fit = w[idx]
        else:
            x_fit = x
            y_fit = y
            p_fit = p
            w_fit = w
        f0 = x_fit * (1.0 - p_fit)
        f1 = x_fit * p_fit
        a = float(np.sum(w_fit * f0 * f0))
        b = float(np.sum(w_fit * f0 * f1))
        c = float(np.sum(w_fit * f1 * f1))
        d = float(np.sum(w_fit * f0 * y_fit))
        e = float(np.sum(w_fit * f1 * y_fit))
        det = a * c - b * b
        if det <= EPSILON:
            denom = float(np.sum(w_fit * x_fit * x_fit))
            if denom <= EPSILON:
                return 1.0, 1.0
            scale = float(np.sum(w_fit * x_fit * y_fit)) / denom
            scale = float(np.clip(scale, 0.0, float(max_scale)))
            return scale, scale
        s0 = (d * c - b * e) / det
        s1 = (a * e - b * d) / det
        s0 = float(np.clip(s0, 0.0, float(max_scale)))
        s1 = float(np.clip(s1, 0.0, 1.0))
        return s0, s1

    if mode_norm == "global":
        x_all = log_imputed_final[zeros_obs]
        y_all = log_true[zeros_obs]
        p_all = p_use[zeros_obs]
        s0, s1 = _fit_scales(x_all, y_all, p_all)
        scale = s0 * (1.0 - p_use) + s1 * p_use
        scale = np.clip(scale, 0.0, 1.0)
        scale_blend = (1.0 - weight) + weight * scale
        log_adj[zeros_obs] = log_adj[zeros_obs] * scale_blend[zeros_obs]
        return log_adj, s0, s1

    n_genes = log_imputed_final.shape[1]
    scales_drop = np.ones((n_genes,), dtype=np.float32)
    scales_bio = np.ones((n_genes,), dtype=np.float32)
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        x = log_imputed_final[mask, j]
        y = log_true[mask, j]
        p = p_use[mask, j]
        s0, s1 = _fit_scales(x, y, p)
        scales_drop[j] = s0
        scales_bio[j] = s1
        scale = s0 * (1.0 - p) + s1 * p
        scale = np.clip(scale, 0.0, 1.0)
        scale_blend = (1.0 - weight) + weight * scale
        log_adj[mask, j] = log_adj[mask, j] * scale_blend
    mean_drop = float(np.nanmean(scales_drop)) if scales_drop.size else float("nan")
    mean_bio = float(np.nanmean(scales_bio)) if scales_bio.size else float("nan")
    return log_adj, mean_drop, mean_bio


def _apply_dropout_recover(
    log_imputed_final: np.ndarray,
    log_imputed_raw: np.ndarray,
    zeros_obs: np.ndarray,
    p_bio: np.ndarray,
    weight: float,
    p_max: float,
    gamma: float,
) -> np.ndarray:
    weight = float(weight)
    if weight <= 0.0:
        return log_imputed_final
    p_max = float(p_max)
    if p_max <= 0.0:
        return log_imputed_final
    p_use = np.clip(p_bio, 0.0, 1.0) ** float(gamma)
    mask = zeros_obs & (p_use <= p_max)
    if not np.any(mask):
        return log_imputed_final
    log_adj = log_imputed_final.copy()
    log_adj[mask] = (1.0 - weight) * log_adj[mask] + weight * log_imputed_raw[mask]
    return log_adj


def _apply_constrained_zero_scale(
    log_imputed_final: np.ndarray,
    log_true: np.ndarray,
    log_obs: np.ndarray,
    zeros_obs: np.ndarray,
    max_mse_inc: float,
    lambda_max: float,
    iters: int,
) -> Tuple[np.ndarray, float]:
    metrics_base = compute_mse_metrics(log_imputed_final, log_true, log_obs)
    mse_base = float(metrics_base["mse"])
    mse_target = mse_base * (1.0 + float(max_mse_inc))
    if mse_target <= 0.0:
        return log_imputed_final, float("nan")
    n_genes = log_imputed_final.shape[1]
    x_list: List[np.ndarray] = []
    sum_x2 = np.zeros(n_genes, dtype=np.float64)
    sum_xy = np.zeros(n_genes, dtype=np.float64)
    sum_x2_bio = np.zeros(n_genes, dtype=np.float64)
    sum_xy_bio = np.zeros(n_genes, dtype=np.float64)
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            x_list.append(np.array([], dtype=np.float32))
            continue
        x = log_imputed_final[mask, j].astype(np.float64, copy=False)
        y = log_true[mask, j].astype(np.float64, copy=False)
        x_list.append(x)
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

    def _eval_lambda(lam: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
        scales = _scales_for_lambda(lam)
        log_adj = log_imputed_final.copy()
        for j in range(n_genes):
            s = scales[j]
            if abs(s - 1.0) < 1e-6:
                continue
            mask = zeros_obs[:, j]
            if not np.any(mask):
                continue
            log_adj[mask, j] = log_adj[mask, j] * s
        metrics = compute_mse_metrics(log_adj, log_true, log_obs)
        return float(metrics["mse"]), float(metrics["mse_biozero"]), log_adj, scales

    mse_low, bz_low, log_low, _ = _eval_lambda(0.0)
    if mse_low <= mse_target:
        return log_low, 0.0

    lam_high = float(lambda_max)
    mse_high, bz_high, log_high, _ = _eval_lambda(lam_high)
    if mse_high > mse_target:
        return log_high, lam_high

    lo = 0.0
    hi = lam_high
    best_log = log_high
    best_lam = lam_high
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        mse_mid, bz_mid, log_mid, _ = _eval_lambda(mid)
        if mse_mid <= mse_target:
            hi = mid
            best_log = log_mid
            best_lam = mid
        else:
            lo = mid
    return best_log, best_lam


def _is_valid_config(config: Dict[str, object]) -> bool:
    if not (0.0 <= float(config["p_zero"]) <= 1.0):
        return False
    if not (0.0 <= float(config["p_nz"]) <= 1.0):
        return False
    noise_min = float(config["noise_min"])
    noise_max = float(config["noise_max"])
    if noise_min < 0.0 or noise_max < 0.0 or noise_min > noise_max:
        return False
    dropout = float(config["dropout"])
    if dropout < 0.0 or dropout >= 1.0:
        return False
    thr_drop = float(config["thr_drop"])
    if thr_drop <= 0.0 or thr_drop >= 1.0:
        return False
    p_low = float(config["p_low"])
    p_high = float(config["p_high"])
    if not (0.0 <= p_low < p_high <= 100.0):
        return False
    post_scale = float(config["post_threshold_scale"])
    if post_scale < 0.0:
        return False
    if float(config["post_threshold_gamma"]) <= 0.0:
        return False
    post_gene_q = float(config["post_gene_quantile"])
    if post_gene_q < -1.0 or post_gene_q > 1.0:
        return False
    if float(config["post_gene_scale"]) <= 0.0:
        return False
    if float(config["post_gene_gamma"]) <= 0.0:
        return False
    post_gene_ref = str(config.get("post_gene_ref", "obs")).lower()
    if post_gene_ref not in ("obs", "recon"):
        return False
    if int(config["epochs"]) < 1 or int(config["batch_size"]) < 1:
        return False
    if float(config["lr"]) <= 0.0:
        return False
    if float(config["loss_bio_weight"]) < 0.0 or float(config["loss_nz_weight"]) < 0.0:
        return False
    if float(config["bio_reg_weight"]) < 0.0:
        return False
    shrink_alpha = float(config["shrink_alpha"])
    if shrink_alpha < 0.0 or shrink_alpha > 2.0:
        return False
    if float(config["shrink_gamma"]) <= 0.0:
        return False
    zero_shrink = float(config.get("zero_shrink", 1.0))
    if zero_shrink < 0.0 or zero_shrink > 1.0:
        return False
    zero_cap_q = float(config.get("zero_cap_quantile", -1.0))
    if zero_cap_q < -1.0 or zero_cap_q > 1.0:
        return False
    if float(config.get("zero_cap_scale", 1.0)) <= 0.0:
        return False
    zero_cap_ref = str(config.get("zero_cap_ref", "obs")).lower()
    if zero_cap_ref not in ("obs", "recon"):
        return False
    blend_alpha = float(config["blend_alpha"])
    if blend_alpha < 0.0 or blend_alpha > 1.0:
        return False
    if float(config["blend_gamma"]) <= 0.0:
        return False
    if float(config["p_bio_temp"]) <= 0.0:
        return False
    if float(config.get("post_bio_temp", 1.0)) <= 0.0:
        return False
    bio_model = str(config.get("bio_model", "splat")).lower()
    if bio_model not in ("splat", "poisson", "mix"):
        return False
    bio_model_mix = float(config.get("bio_model_mix", 0.5))
    if bio_model_mix < 0.0 or bio_model_mix > 1.0:
        return False
    if float(config.get("poisson_scale", 1.0)) <= 0.0:
        return False
    recon_weight = float(config["recon_weight"])
    if recon_weight < 0.0 or recon_weight > 1.0:
        return False
    calib_method = str(config.get("calib_method", "none")).lower()
    if calib_method not in ("none", "platt", "isotonic"):
        return False
    calib_blend = float(config.get("calib_blend", 1.0))
    if calib_blend < 0.0 or calib_blend > 1.0:
        return False
    if float(config.get("calib_platt_l2", 0.0)) < 0.0:
        return False
    if int(config.get("calib_platt_max_iter", 1)) < 1:
        return False
    if float(config.get("calib_platt_lr", 0.1)) <= 0.0:
        return False
    if int(config.get("calib_max_samples", 0)) < 0:
        return False
    sup_bio_weight = float(config.get("sup_bio_weight", 0.0))
    if sup_bio_weight < 0.0 or sup_bio_weight > 1.0:
        return False
    if float(config.get("sup_bio_l2", 0.0)) < 0.0:
        return False
    if float(config.get("sup_bio_lr", 0.1)) <= 0.0:
        return False
    if int(config.get("sup_bio_epochs", 1)) < 1:
        return False
    if int(config.get("sup_bio_max_samples", 0)) < 0:
        return False
    lowrank_weight = float(config.get("lowrank_weight", 0.0))
    if lowrank_weight < 0.0 or lowrank_weight > 1.0:
        return False
    if int(config.get("lowrank_rank", 0)) < 0:
        return False
    if int(config.get("lowrank_iters", 0)) < 0:
        return False
    if lowrank_weight > 0.0:
        if int(config.get("lowrank_rank", 0)) < 1 or int(config.get("lowrank_iters", 0)) < 1:
            return False
    zero_scale_weight = float(config.get("zero_scale_weight", 0.0))
    if zero_scale_weight < 0.0 or zero_scale_weight > 1.0:
        return False
    zero_scale_mode = str(config.get("zero_scale_mode", "gene")).lower()
    if zero_scale_mode not in ("global", "gene"):
        return False
    if float(config.get("zero_scale_bio_weight", 0.0)) < 0.0:
        return False
    if int(config.get("zero_scale_max_samples", 0)) < 0:
        return False
    zero_mix_weight = float(config.get("zero_mix_weight", 0.0))
    if zero_mix_weight < 0.0 or zero_mix_weight > 1.0:
        return False
    zero_mix_mode = str(config.get("zero_mix_mode", "global")).lower()
    if zero_mix_mode not in ("global", "gene"):
        return False
    if float(config.get("zero_mix_bio_weight", 0.0)) < 0.0:
        return False
    if int(config.get("zero_mix_max_samples", 0)) < 0:
        return False
    if float(config.get("zero_mix_gamma", 1.0)) <= 0.0:
        return False
    if float(config.get("zero_mix_max_scale", 1.0)) <= 0.0:
        return False
    recover_weight = float(config.get("recover_weight", 0.0))
    if recover_weight < 0.0 or recover_weight > 1.0:
        return False
    recover_pmax = float(config.get("recover_pmax", 0.0))
    if recover_pmax < 0.0 or recover_pmax > 1.0:
        return False
    if float(config.get("recover_gamma", 1.0)) <= 0.0:
        return False
    sup_bio_method = str(config.get("sup_bio_method", "logit")).lower()
    if sup_bio_method not in ("logit", "gauss", "mlp", "gene_logit"):
        return False
    if int(config.get("sup_bio_min_count", 1)) < 1:
        return False
    if float(config.get("sup_bio_var_floor", 1e-6)) <= 0.0:
        return False
    if int(config.get("sup_bio_hidden", 1)) < 1:
        return False
    gene_boost = float(config["gene_boost"])
    if gene_boost < 0.0 or gene_boost > 1.0:
        return False
    if float(config["gene_boost_gamma"]) <= 0.0:
        return False
    gene_nz_boost = float(config["gene_nz_boost"])
    if gene_nz_boost < 0.0 or gene_nz_boost > 1.0:
        return False
    if float(config["gene_nz_boost_gamma"]) <= 0.0:
        return False
    gene_nz_mix = float(config["gene_nz_mix"])
    if gene_nz_mix < 0.0 or gene_nz_mix > 1.0:
        return False
    if float(config["gene_nz_mix_gamma"]) <= 0.0:
        return False
    cell_zero_weight = float(config["cell_zero_weight"])
    if cell_zero_weight < 0.0 or cell_zero_weight > 1.0:
        return False
    cell_depth_boost = float(config["cell_depth_boost"])
    if cell_depth_boost < 0.0 or cell_depth_boost > 1.0:
        return False
    if float(config["cell_depth_gamma"]) <= 0.0:
        return False
    cluster_weight = float(config["cluster_weight"])
    if cluster_weight < 0.0 or cluster_weight > 1.0:
        return False
    if float(config["cluster_gamma"]) <= 0.0:
        return False
    if int(config["cluster_k"]) < 1 or int(config["cluster_pcs"]) < 1:
        return False
    ae_bio_weight = float(config["ae_bio_weight"])
    if ae_bio_weight < 0.0 or ae_bio_weight > 1.0:
        return False
    ae_bio_cap_weight = float(config.get("ae_bio_cap_weight", 0.0))
    if ae_bio_cap_weight < 0.0 or ae_bio_cap_weight > 1.0:
        return False
    if float(config["ae_bio_temp"]) <= 0.0:
        return False
    ae_bio_quantile = float(config["ae_bio_quantile"])
    if ae_bio_quantile < 0.0 or ae_bio_quantile > 1.0:
        return False
    expr_bio_weight = float(config["expr_bio_weight"])
    if expr_bio_weight < 0.0 or expr_bio_weight > 1.0:
        return False
    expr_bio_cap_weight = float(config.get("expr_bio_cap_weight", 0.0))
    if expr_bio_cap_weight < 0.0 or expr_bio_cap_weight > 1.0:
        return False
    if float(config["expr_bio_temp"]) <= 0.0:
        return False
    expr_bio_quantile = float(config["expr_bio_quantile"])
    if expr_bio_quantile < -1.0 or expr_bio_quantile > 1.0:
        return False
    if float(config["expr_bio_scale"]) <= 0.0:
        return False
    expr_expected = str(config.get("expr_bio_expected", "log")).lower()
    if expr_expected not in ("log", "counts", "mix"):
        return False
    expr_mix = float(config.get("expr_bio_mix", 0.5))
    if expr_mix < 0.0 or expr_mix > 1.0:
        return False
    gene_rare_threshold = float(config["gene_rare_threshold"])
    if gene_rare_threshold < -1.0 or gene_rare_threshold > 1.0:
        return False
    gene_rare_pbio = float(config["gene_rare_pbio"])
    if gene_rare_pbio < 0.0 or gene_rare_pbio > 1.0:
        return False
    knn_bio_weight = float(config["knn_bio_weight"])
    if knn_bio_weight < 0.0 or knn_bio_weight > 1.0:
        return False
    if float(config["knn_bio_temp"]) <= 0.0:
        return False
    knn_bio_quantile = float(config["knn_bio_quantile"])
    if knn_bio_quantile < -1.0 or knn_bio_quantile > 1.0:
        return False
    if int(config["knn_k"]) < 1:
        return False
    return True


def _p_bio_cache_key(config: Dict[str, object]) -> Tuple[object, ...]:
    return (
        str(config["disp_mode"]),
        float(config["disp_const"]),
        bool(config["use_cell_factor"]),
        float(config["tau_dispersion"]),
        float(config["tau_group_dispersion"]),
        float(config["tau_dropout"]),
    )


def _get_p_bio_for_dataset(ds: Dict[str, object], config: Dict[str, object]) -> np.ndarray:
    cache: Dict[Tuple[object, ...], np.ndarray] = ds["p_bio_cache"]
    key = _p_bio_cache_key(config)
    if key not in cache:
        counts = ds["counts"]
        zeros_obs = ds["zeros_obs"]
        cache[key] = splat_cellaware_bio_prob(
            counts=counts,
            zeros_obs=zeros_obs,
            disp_mode=str(config["disp_mode"]),
            disp_const=float(config["disp_const"]),
            use_cell_factor=bool(config["use_cell_factor"]),
            tau_dispersion=float(config["tau_dispersion"]),
            tau_group_dispersion=float(config["tau_group_dispersion"]),
            tau_dropout=float(config["tau_dropout"]),
        )
    return cache[key]


def evaluate_config(
    datasets: List[Dict[str, object]],
    config: Dict[str, object],
    device: torch.device,
    seed: int,
    calibrate_p_bio: bool,
    calibrate_zero_threshold: bool,
    calibrate_zero_mode: str,
    calib_temp_list: Sequence[float],
    calib_bias_list: Sequence[float],
    calib_thr_list: Sequence[float],
    calibrate_p_bio_mode: str,
    calib_frac_scale_list: Sequence[float],
    calibrate_bin_count: int,
    calib_zero_thr_list: Sequence[float],
    calibrate_lambda_mse: float,
    constrained_zero_scale: bool,
    constrained_zero_max_mse_inc: float,
    constrained_zero_lambda_max: float,
    constrained_zero_iters: int,
    target_mse: Optional[float],
    target_biozero: Optional[float],
) -> Tuple[float, float, List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    results_final: List[Dict[str, object]] = []
    results_raw: List[Dict[str, object]] = []
    results_no_bio: List[Dict[str, object]] = []
    mse_bz_list: List[float] = []
    mse_list: List[float] = []

    for ds in datasets:
        ds_name = str(ds["dataset"])
        logcounts = ds["logcounts"]
        log_true = ds["log_true"]
        zeros_obs = ds["zeros_obs"]
        nz_mask = logcounts > 0.0
        counts_max = ds["counts_max"]
        gene_mean = ds["gene_mean"]
        gene_mean_norm = ds["gene_mean_norm"]
        gene_nz_frac = ds["gene_nz_frac"]
        cell_zero_norm = ds["cell_zero_norm"]
        cell_depth_norm = ds["cell_depth_norm"]
        expected_log_log = ds["expected_log"]
        expected_log_counts = ds["expected_log_counts"]
        expected_counts = ds["expected_counts"]
        if bool(config["oracle_bio"]):
            p_bio_use = (log_true <= EPSILON).astype(np.float32)
            p_bio_use[~zeros_obs] = 0.0
        else:
            p_bio_splat = _get_p_bio_for_dataset(ds, config)
            bio_model = str(config.get("bio_model", "splat")).lower()
            if bio_model == "poisson":
                p_bio_use = poisson_bio_prob(expected_counts, zeros_obs, float(config["poisson_scale"]))
            elif bio_model == "mix":
                mix = float(config.get("bio_model_mix", 0.5))
                p_bio_pois = poisson_bio_prob(expected_counts, zeros_obs, float(config["poisson_scale"]))
                p_bio_use = (1.0 - mix) * p_bio_splat + mix * p_bio_pois
            else:
                p_bio_use = p_bio_splat
            if float(config["gene_boost"]) > 0.0:
                boost = float(config["gene_boost"]) * (1.0 - gene_mean_norm) ** float(config["gene_boost_gamma"])
                boost = np.clip(boost, 0.0, 1.0)
                boost_row = boost[None, :]
                p_bio_use = 1.0 - (1.0 - p_bio_use) * (1.0 - boost_row)
                p_bio_use = np.clip(p_bio_use, 0.0, 1.0)
            if float(config["gene_nz_boost"]) > 0.0:
                nz_boost = float(config["gene_nz_boost"]) * (1.0 - gene_nz_frac) ** float(config["gene_nz_boost_gamma"])
                nz_boost = np.clip(nz_boost, 0.0, 1.0)
                nz_boost_row = nz_boost[None, :]
                p_bio_use = 1.0 - (1.0 - p_bio_use) * (1.0 - nz_boost_row)
                p_bio_use = np.clip(p_bio_use, 0.0, 1.0)
            if float(config["gene_nz_mix"]) > 0.0:
                nz_mix = float(config["gene_nz_mix"])
                nz_prob = (1.0 - gene_nz_frac) ** float(config["gene_nz_mix_gamma"])
                nz_prob = np.clip(nz_prob, 0.0, 1.0)
                nz_prob_row = nz_prob[None, :]
                p_bio_use = (1.0 - nz_mix) * p_bio_use + nz_mix * nz_prob_row
                p_bio_use = np.clip(p_bio_use, 0.0, 1.0)
            if float(config["cell_zero_weight"]) > 0.0:
                cell_w = np.clip(float(config["cell_zero_weight"]) * cell_zero_norm, 0.0, 1.0)
                p_bio_use = p_bio_use * (1.0 - cell_w[:, None])
            if float(config["cell_depth_boost"]) > 0.0:
                depth_boost = float(config["cell_depth_boost"]) * (cell_depth_norm ** float(config["cell_depth_gamma"]))
                depth_boost = np.clip(depth_boost, 0.0, 1.0)
                p_bio_use = 1.0 - (1.0 - p_bio_use) * (1.0 - depth_boost[:, None])
                p_bio_use = np.clip(p_bio_use, 0.0, 1.0)
            if float(config["cluster_weight"]) > 0.0:
                labels, expr_frac = _get_cluster_expr_frac(ds, config, seed=seed)
                k = expr_frac.shape[0]
                if k > 0 and labels.size == logcounts.shape[0]:
                    p_cluster = (1.0 - expr_frac[labels]) ** float(config["cluster_gamma"])
                    p_cluster = p_cluster.astype(np.float32, copy=False)
                    p_cluster[~zeros_obs] = 0.0
                    w = float(config["cluster_weight"])
                    p_bio_use = (1.0 - w) * p_bio_use + w * p_cluster
                    p_bio_use = np.clip(p_bio_use, 0.0, 1.0)
            if float(config["p_bio_temp"]) != 1.0 or float(config["p_bio_bias"]) != 0.0:
                p_clip = np.clip(p_bio_use, 1e-6, 1.0 - 1e-6)
                logit = np.log(p_clip / (1.0 - p_clip))
                logit = logit * float(config["p_bio_temp"]) + float(config["p_bio_bias"])
                p_bio_use = _sigmoid(logit)
                p_bio_use = np.clip(p_bio_use, 0.0, 1.0)
            p_bio_use[~zeros_obs] = 0.0
        p_bio_train = p_bio_use

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        need_latent = float(config["knn_bio_weight"]) > 0.0
        log_imputed_raw, latent = train_autoencoder_reconstruct(
            logcounts=logcounts,
            counts_max=counts_max,
            p_bio=p_bio_train,
            device=device,
            hidden=config["hidden"],
            bottleneck=int(config["bottleneck"]),
            p_zero=float(config["p_zero"]),
            p_nz=float(config["p_nz"]),
            noise_min_frac=float(config["noise_min"]),
            noise_max_frac=float(config["noise_max"]),
            dropout=float(config["dropout"]),
            use_residual=bool(config["use_residual"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            loss_bio_weight=float(config["loss_bio_weight"]),
            loss_nz_weight=float(config["loss_nz_weight"]),
            bio_reg_weight=float(config["bio_reg_weight"]),
            recon_weight=float(config["recon_weight"]),
            p_low=float(config["p_low"]),
            p_high=float(config["p_high"]),
            return_latent=need_latent,
        )
        lowrank_weight = float(config.get("lowrank_weight", 0.0))
        if lowrank_weight > 0.0:
            lowrank = _lowrank_impute(
                logcounts=logcounts,
                zeros_obs=zeros_obs,
                rank=int(config.get("lowrank_rank", 0)),
                iters=int(config.get("lowrank_iters", 0)),
            )
            log_imputed_raw = ((1.0 - lowrank_weight) * log_imputed_raw + lowrank_weight * lowrank).astype(
                np.float32,
                copy=False,
            )

        p_bio_post = p_bio_train
        p_expr = None
        expr_weight = float(config["expr_bio_weight"])
        expr_cap_weight = float(config.get("expr_bio_cap_weight", 0.0))
        if expr_weight > 0.0 or expr_cap_weight > 0.0:
            expr_q = float(config["expr_bio_quantile"])
            if expr_q >= 0.0:
                expr_thr = _get_gene_quantile(ds, expr_q)
            else:
                expr_thr = gene_mean
            expr_thr = expr_thr * float(config["expr_bio_scale"])
            expr_expected = str(config.get("expr_bio_expected", "log")).lower()
            if expr_expected == "counts":
                expected_log_expr = expected_log_counts
            elif expr_expected == "mix":
                mix = float(config["expr_bio_mix"])
                expected_log_expr = mix * expected_log_log + (1.0 - mix) * expected_log_counts
            else:
                expected_log_expr = expected_log_log
            logits = (expr_thr[None, :] - expected_log_expr) / float(config["expr_bio_temp"])
            logits = logits + float(config["expr_bio_bias"])
            p_expr = _sigmoid(logits).astype(np.float32, copy=False)
            p_expr[~zeros_obs] = 0.0
        if expr_weight > 0.0 and p_expr is not None:
            if bool(config["expr_bio_union"]):
                p_bio_post = 1.0 - (1.0 - p_bio_post) * (1.0 - expr_weight * p_expr)
            else:
                p_bio_post = (1.0 - expr_weight) * p_bio_post + expr_weight * p_expr
            p_bio_post = np.clip(p_bio_post, 0.0, 1.0)
        rare_thr = float(config["gene_rare_threshold"])
        if rare_thr >= 0.0 and float(config["gene_rare_pbio"]) > 0.0:
            rare = gene_nz_frac <= rare_thr
            if np.any(rare):
                min_p = float(config["gene_rare_pbio"])
                p_bio_post[:, rare] = np.maximum(p_bio_post[:, rare], min_p)
        p_bio_post[~zeros_obs] = 0.0
        p_bio_ae = None
        if float(config["ae_bio_weight"]) > 0.0 or float(config.get("ae_bio_cap_weight", 0.0)) > 0.0:
            gene_q = _get_gene_quantile(ds, float(config["ae_bio_quantile"]))
            thr = gene_q[None, :]
            temp = float(config["ae_bio_temp"])
            logits = (thr - log_imputed_raw) / temp
            p_bio_ae = _sigmoid(logits)
            p_bio_ae = np.clip(p_bio_ae, 0.0, 1.0)
            p_bio_ae[~zeros_obs] = 0.0
        if float(config["ae_bio_weight"]) > 0.0 and p_bio_ae is not None:
            w = float(config["ae_bio_weight"])
            if bool(config["ae_bio_union"]):
                p_bio_post = 1.0 - (1.0 - p_bio_post) * (1.0 - w * p_bio_ae)
            else:
                p_bio_post = (1.0 - w) * p_bio_post + w * p_bio_ae
            p_bio_post = np.clip(p_bio_post, 0.0, 1.0)
        if float(config["knn_bio_weight"]) > 0.0:
            knn_q = float(config["knn_bio_quantile"])
            if knn_q >= 0.0:
                knn_thr = _get_gene_quantile(ds, knn_q)
            else:
                knn_thr = gene_mean
            p_bio_knn = _knn_bio_prob_from_latent(
                latent=latent,
                recon=log_imputed_raw,
                zeros_obs=zeros_obs,
                gene_thr=knn_thr,
                k=int(config["knn_k"]),
                temp=float(config["knn_bio_temp"]),
            )
            w = float(config["knn_bio_weight"])
            if bool(config["knn_bio_union"]):
                p_bio_post = 1.0 - (1.0 - p_bio_post) * (1.0 - w * p_bio_knn)
            else:
                p_bio_post = (1.0 - w) * p_bio_post + w * p_bio_knn
            p_bio_post = np.clip(p_bio_post, 0.0, 1.0)
        if float(config.get("ae_bio_cap_weight", 0.0)) > 0.0 and p_bio_ae is not None:
            cap_w = float(config["ae_bio_cap_weight"])
            cap = (1.0 - cap_w) + cap_w * p_bio_ae
            p_bio_post = p_bio_post * cap
            p_bio_post = np.clip(p_bio_post, 0.0, 1.0)
        if expr_cap_weight > 0.0 and p_expr is not None:
            cap = (1.0 - expr_cap_weight) + expr_cap_weight * p_expr
            p_bio_post = p_bio_post * cap
            p_bio_post = np.clip(p_bio_post, 0.0, 1.0)
        p_bio_post[~zeros_obs] = 0.0
        post_temp = float(config.get("post_bio_temp", 1.0))
        post_bias = float(config.get("post_bio_bias", 0.0))
        if post_temp != 1.0 or post_bias != 0.0:
            p_clip = np.clip(p_bio_post, 1e-6, 1.0 - 1e-6)
            logit = np.log(p_clip / (1.0 - p_clip))
            logit = logit * post_temp + post_bias
            p_bio_post = _sigmoid(logit)
            p_bio_post = np.clip(p_bio_post, 0.0, 1.0)
            p_bio_post[~zeros_obs] = 0.0

        calib_info = {
            "calib_method": "none",
            "calib_blend": float(config.get("calib_blend", 1.0)),
            "calib_platt_a": 1.0,
            "calib_platt_b": 0.0,
            "calib_platt_used": False,
            "calib_iso_blocks": 0,
        }
        if calibrate_p_bio and str(config.get("calib_method", "none")).lower() != "none":
            p_bio_post, calib_info = _calibrate_p_bio_supervised(
                p_bio_post=p_bio_post,
                log_true=log_true,
                zeros_obs=zeros_obs,
                config=config,
                seed=seed,
            )
        sup_info = {
            "sup_bio_weight": float(config.get("sup_bio_weight", 0.0)),
            "sup_bio_l2": float(config.get("sup_bio_l2", 0.0)),
            "sup_bio_lr": float(config.get("sup_bio_lr", 0.1)),
            "sup_bio_epochs": int(config.get("sup_bio_epochs", 100)),
            "sup_bio_balance": bool(config.get("sup_bio_balance", False)),
            "sup_bio_max_samples": int(config.get("sup_bio_max_samples", 0)),
            "sup_bio_method": str(config.get("sup_bio_method", "logit")),
            "sup_bio_min_count": int(config.get("sup_bio_min_count", 5)),
            "sup_bio_var_floor": float(config.get("sup_bio_var_floor", 1e-3)),
            "sup_bio_hidden": int(config.get("sup_bio_hidden", 16)),
            "sup_bio_used": False,
        }
        if float(config.get("sup_bio_weight", 0.0)) > 0.0:
            p_bio_post, sup_info = _supervised_bio_prob(
                p_bio_base=p_bio_post,
                log_imputed_raw=log_imputed_raw,
                gene_mean=gene_mean,
                gene_mean_norm=gene_mean_norm,
                gene_nz_frac=gene_nz_frac,
                cell_zero_norm=cell_zero_norm,
                cell_depth_norm=cell_depth_norm,
                expected_log=expected_log_log,
                expected_log_counts=expected_log_counts,
                zeros_obs=zeros_obs,
                log_true=log_true,
                config=config,
                seed=seed,
            )

        log_imputed_keep = _apply_keep_positive(
            log_imputed_raw=log_imputed_raw,
            logcounts=logcounts,
            keep_positive=bool(config["keep_positive"]),
        )

        thr_bio = 1.0 - float(config["thr_drop"])
        calib_temp = 1.0
        calib_bias = 0.0
        calib_thr = thr_bio
        calib_mode = "none"
        calib_frac_scale = 1.0
        calib_bin_count = 0
        if calibrate_p_bio:
            (
                log_imputed_final,
                pred_bio_mask,
                calib_temp,
                calib_bias,
                calib_thr,
                calib_mode,
                calib_frac_scale,
                calib_bin_count,
            ) = _calibrate_p_bio(
                log_imputed_raw=log_imputed_raw,
                log_imputed_keep=log_imputed_keep,
                logcounts=logcounts,
                log_true=log_true,
                p_bio_post=p_bio_post,
                config=config,
                ds=ds,
                nz_mask=nz_mask,
                zeros_obs=zeros_obs,
                temp_grid=calib_temp_list,
                bias_grid=calib_bias_list,
                thr_grid=calib_thr_list,
                mode=calibrate_p_bio_mode,
                frac_scale_grid=calib_frac_scale_list,
                bin_count=calibrate_bin_count,
                lambda_mse=calibrate_lambda_mse,
                target_mse=target_mse,
                target_biozero=target_biozero,
            )
        else:
            log_imputed_final, pred_bio_mask = _postprocess_imputation(
                log_imputed_raw=log_imputed_raw,
                log_imputed_keep=log_imputed_keep,
                p_bio_post=p_bio_post,
                config=config,
                ds=ds,
                nz_mask=nz_mask,
                zeros_obs=zeros_obs,
                thr_bio=thr_bio,
            )

        calib_zero_thr = float("nan")
        extra_mask = None
        if calibrate_zero_threshold:
            log_imputed_final, extra_mask, calib_zero_thr = _calibrate_zero_threshold(
                log_imputed_final=log_imputed_final,
                log_true=log_true,
                log_obs=logcounts,
                zeros_obs=zeros_obs,
                thr_grid=calib_zero_thr_list,
                lambda_mse=calibrate_lambda_mse,
                mode=calibrate_zero_mode,
            )
            pred_bio_mask = np.asarray(pred_bio_mask, dtype=bool) | extra_mask

        zero_scale_mean = float("nan")
        zero_scale_weight = float(config.get("zero_scale_weight", 0.0))
        if zero_scale_weight > 0.0:
            log_imputed_final, zero_scale_mean = _apply_zero_scale(
                log_imputed_final=log_imputed_final,
                log_true=log_true,
                zeros_obs=zeros_obs,
                weight=zero_scale_weight,
                bio_weight=float(config.get("zero_scale_bio_weight", 0.0)),
                lambda_mse=calibrate_lambda_mse,
                mode=str(config.get("zero_scale_mode", "gene")),
                max_samples=int(config.get("zero_scale_max_samples", 0)),
                seed=seed,
            )

        p_bio_mix = p_bio_post
        if calibrate_p_bio:
            p_bio_mix = _logit_scale_probs(p_bio_mix, temp=float(calib_temp), bias=float(calib_bias))
        p_bio_mix = np.clip(p_bio_mix, 0.0, 1.0)
        if pred_bio_mask is not None:
            p_bio_mix = p_bio_mix.copy()
            p_bio_mix[pred_bio_mask] = 1.0
        if extra_mask is not None:
            p_bio_mix[extra_mask] = 1.0
        p_bio_mix[~zeros_obs] = 0.0

        zero_mix_drop = float("nan")
        zero_mix_bio = float("nan")
        zero_mix_weight = float(config.get("zero_mix_weight", 0.0))
        if zero_mix_weight > 0.0:
            log_imputed_final, zero_mix_drop, zero_mix_bio = _apply_zero_mix(
                log_imputed_final=log_imputed_final,
                log_true=log_true,
                zeros_obs=zeros_obs,
                p_bio=p_bio_mix,
                weight=zero_mix_weight,
                bio_weight=float(config.get("zero_mix_bio_weight", 0.0)),
                lambda_mse=calibrate_lambda_mse,
                mode=str(config.get("zero_mix_mode", "global")),
                max_samples=int(config.get("zero_mix_max_samples", 0)),
                gamma=float(config.get("zero_mix_gamma", 1.0)),
                max_scale=float(config.get("zero_mix_max_scale", 1.0)),
                seed=seed,
            )

        recover_weight = float(config.get("recover_weight", 0.0))
        if recover_weight > 0.0:
            log_imputed_final = _apply_dropout_recover(
                log_imputed_final=log_imputed_final,
                log_imputed_raw=log_imputed_raw,
                zeros_obs=zeros_obs,
                p_bio=p_bio_mix,
                weight=recover_weight,
                p_max=float(config.get("recover_pmax", 0.0)),
                gamma=float(config.get("recover_gamma", 1.0)),
            )

        constrained_zero_lambda = float("nan")
        if constrained_zero_scale:
            log_imputed_final, constrained_zero_lambda = _apply_constrained_zero_scale(
                log_imputed_final=log_imputed_final,
                log_true=log_true,
                log_obs=logcounts,
                zeros_obs=zeros_obs,
                max_mse_inc=constrained_zero_max_mse_inc,
                lambda_max=constrained_zero_lambda_max,
                iters=constrained_zero_iters,
            )

        row_base = {
            "dataset": ds_name,
            "thr_drop": float(config["thr_drop"]),
            "thr_bio": calib_thr,
            "disp_mode": str(config["disp_mode"]),
            "disp_const": float(config["disp_const"]),
            "use_cell_factor": bool(config["use_cell_factor"]),
            "tau_dispersion": float(config["tau_dispersion"]),
            "tau_group_dispersion": float(config["tau_group_dispersion"]),
            "tau_dropout": float(config["tau_dropout"]),
            "p_zero": float(config["p_zero"]),
            "p_nz": float(config["p_nz"]),
            "noise_min": float(config["noise_min"]),
            "noise_max": float(config["noise_max"]),
            "hidden": ",".join(str(v) for v in config["hidden"]),
            "bottleneck": int(config["bottleneck"]),
            "dropout": float(config["dropout"]),
            "use_residual": bool(config["use_residual"]),
            "epochs": int(config["epochs"]),
            "batch_size": int(config["batch_size"]),
            "lr": float(config["lr"]),
            "weight_decay": float(config["weight_decay"]),
            "loss_bio_weight": float(config["loss_bio_weight"]),
            "loss_nz_weight": float(config["loss_nz_weight"]),
            "bio_reg_weight": float(config["bio_reg_weight"]),
            "recon_weight": float(config["recon_weight"]),
            "p_low": float(config["p_low"]),
            "p_high": float(config["p_high"]),
            "post_threshold": float(config["post_threshold"]),
            "post_threshold_scale": float(config["post_threshold_scale"]),
            "post_threshold_gamma": float(config["post_threshold_gamma"]),
            "post_gene_quantile": float(config["post_gene_quantile"]),
            "post_gene_scale": float(config["post_gene_scale"]),
            "post_gene_gamma": float(config["post_gene_gamma"]),
            "post_gene_ref": str(config.get("post_gene_ref", "obs")),
            "blend_alpha": float(config["blend_alpha"]),
            "blend_gamma": float(config["blend_gamma"]),
            "p_bio_temp": float(config["p_bio_temp"]),
            "p_bio_bias": float(config["p_bio_bias"]),
            "bio_model": str(config.get("bio_model", "splat")),
            "bio_model_mix": float(config.get("bio_model_mix", 0.5)),
            "poisson_scale": float(config.get("poisson_scale", 1.0)),
            "post_bio_temp": float(config.get("post_bio_temp", 1.0)),
            "post_bio_bias": float(config.get("post_bio_bias", 0.0)),
            "calibrate_p_bio": bool(calibrate_p_bio),
            "calibrate_zero_threshold": bool(calibrate_zero_threshold),
            "calibrate_zero_mode": str(calibrate_zero_mode),
            "calib_method": str(config.get("calib_method", "none")),
            "calib_blend": float(config.get("calib_blend", 1.0)),
            "calib_platt_l2": float(config.get("calib_platt_l2", 0.0)),
            "calib_platt_max_iter": int(config.get("calib_platt_max_iter", 200)),
            "calib_platt_lr": float(config.get("calib_platt_lr", 0.1)),
            "calib_platt_use_logit": bool(config.get("calib_platt_use_logit", False)),
            "calib_platt_balance": bool(config.get("calib_platt_balance", False)),
            "calib_max_samples": int(config.get("calib_max_samples", 0)),
            "calib_platt_a": float(calib_info.get("calib_platt_a", 1.0)),
            "calib_platt_b": float(calib_info.get("calib_platt_b", 0.0)),
            "calib_platt_used": bool(calib_info.get("calib_platt_used", False)),
            "calib_iso_blocks": int(calib_info.get("calib_iso_blocks", 0)),
            "calib_zero_thr": float(calib_zero_thr),
            "sup_bio_weight": float(sup_info.get("sup_bio_weight", 0.0)),
            "sup_bio_l2": float(sup_info.get("sup_bio_l2", 0.0)),
            "sup_bio_lr": float(sup_info.get("sup_bio_lr", 0.1)),
            "sup_bio_epochs": int(sup_info.get("sup_bio_epochs", 100)),
            "sup_bio_balance": bool(sup_info.get("sup_bio_balance", False)),
            "sup_bio_max_samples": int(sup_info.get("sup_bio_max_samples", 0)),
            "sup_bio_method": str(sup_info.get("sup_bio_method", "logit")),
            "sup_bio_min_count": int(sup_info.get("sup_bio_min_count", 5)),
            "sup_bio_var_floor": float(sup_info.get("sup_bio_var_floor", 1e-3)),
            "sup_bio_hidden": int(sup_info.get("sup_bio_hidden", 16)),
            "sup_bio_used": bool(sup_info.get("sup_bio_used", False)),
            "lowrank_weight": float(config.get("lowrank_weight", 0.0)),
            "lowrank_rank": int(config.get("lowrank_rank", 0)),
            "lowrank_iters": int(config.get("lowrank_iters", 0)),
            "zero_scale_weight": float(config.get("zero_scale_weight", 0.0)),
            "zero_scale_mode": str(config.get("zero_scale_mode", "gene")),
            "zero_scale_bio_weight": float(config.get("zero_scale_bio_weight", 0.0)),
            "zero_scale_max_samples": int(config.get("zero_scale_max_samples", 0)),
            "zero_scale_mean": float(zero_scale_mean),
            "zero_mix_weight": float(config.get("zero_mix_weight", 0.0)),
            "zero_mix_mode": str(config.get("zero_mix_mode", "global")),
            "zero_mix_bio_weight": float(config.get("zero_mix_bio_weight", 0.0)),
            "zero_mix_max_samples": int(config.get("zero_mix_max_samples", 0)),
            "zero_mix_gamma": float(config.get("zero_mix_gamma", 1.0)),
            "zero_mix_max_scale": float(config.get("zero_mix_max_scale", 1.0)),
            "zero_mix_scale_drop": float(zero_mix_drop),
            "zero_mix_scale_bio": float(zero_mix_bio),
            "recover_weight": float(config.get("recover_weight", 0.0)),
            "recover_pmax": float(config.get("recover_pmax", 0.0)),
            "recover_gamma": float(config.get("recover_gamma", 1.0)),
            "constrained_zero_scale": bool(constrained_zero_scale),
            "constrained_zero_max_mse_inc": float(constrained_zero_max_mse_inc),
            "constrained_zero_lambda_max": float(constrained_zero_lambda_max),
            "constrained_zero_iters": int(constrained_zero_iters),
            "constrained_zero_lambda": float(constrained_zero_lambda),
            "calib_p_bio_temp": float(calib_temp),
            "calib_p_bio_bias": float(calib_bias),
            "calib_p_bio_thr": float(calib_thr),
            "calib_p_bio_mode": str(calib_mode),
            "calib_p_bio_frac_scale": float(calib_frac_scale),
            "calib_p_bio_bin_count": int(calib_bin_count),
            "ae_bio_weight": float(config["ae_bio_weight"]),
            "ae_bio_cap_weight": float(config.get("ae_bio_cap_weight", 0.0)),
            "ae_bio_temp": float(config["ae_bio_temp"]),
            "ae_bio_quantile": float(config["ae_bio_quantile"]),
            "ae_bio_union": bool(config["ae_bio_union"]),
            "gene_boost": float(config["gene_boost"]),
            "gene_boost_gamma": float(config["gene_boost_gamma"]),
            "gene_nz_boost": float(config["gene_nz_boost"]),
            "gene_nz_boost_gamma": float(config["gene_nz_boost_gamma"]),
            "gene_nz_mix": float(config["gene_nz_mix"]),
            "gene_nz_mix_gamma": float(config["gene_nz_mix_gamma"]),
            "gene_rare_threshold": float(config["gene_rare_threshold"]),
            "gene_rare_pbio": float(config["gene_rare_pbio"]),
            "cell_zero_weight": float(config["cell_zero_weight"]),
            "cell_depth_boost": float(config["cell_depth_boost"]),
            "cell_depth_gamma": float(config["cell_depth_gamma"]),
            "cluster_weight": float(config["cluster_weight"]),
            "cluster_gamma": float(config["cluster_gamma"]),
            "cluster_k": int(config["cluster_k"]),
            "cluster_pcs": int(config["cluster_pcs"]),
            "shrink_alpha": float(config["shrink_alpha"]),
            "shrink_gamma": float(config["shrink_gamma"]),
            "zero_shrink": float(config.get("zero_shrink", 1.0)),
            "zero_cap_quantile": float(config.get("zero_cap_quantile", -1.0)),
            "zero_cap_scale": float(config.get("zero_cap_scale", 1.0)),
            "zero_cap_ref": str(config.get("zero_cap_ref", "obs")),
            "expr_bio_weight": float(config["expr_bio_weight"]),
            "expr_bio_cap_weight": float(config.get("expr_bio_cap_weight", 0.0)),
            "expr_bio_temp": float(config["expr_bio_temp"]),
            "expr_bio_bias": float(config["expr_bio_bias"]),
            "expr_bio_quantile": float(config["expr_bio_quantile"]),
            "expr_bio_scale": float(config["expr_bio_scale"]),
            "expr_bio_union": bool(config["expr_bio_union"]),
            "expr_bio_expected": str(config.get("expr_bio_expected", "log")),
            "expr_bio_mix": float(config.get("expr_bio_mix", 0.5)),
            "knn_bio_weight": float(config["knn_bio_weight"]),
            "knn_bio_temp": float(config["knn_bio_temp"]),
            "knn_bio_quantile": float(config["knn_bio_quantile"]),
            "knn_k": int(config["knn_k"]),
            "knn_bio_union": bool(config["knn_bio_union"]),
            "clip_negative": bool(config["clip_negative"]),
            "keep_positive": bool(config["keep_positive"]),
            "hard_zero_bio": bool(config["hard_zero_bio"]),
            "oracle_bio": bool(config["oracle_bio"]),
            "n_obs_zero": int(np.asarray(ds["zeros_obs"], dtype=bool).sum()),
            "n_pred_bio": int(np.asarray(pred_bio_mask, dtype=bool).sum()),
        }

        raw = {**row_base, **compute_mse_metrics(log_imputed_raw, log_true, logcounts)}
        no_bio = {**row_base, **compute_mse_metrics(log_imputed_keep, log_true, logcounts)}
        final = {**row_base, **compute_mse_metrics(log_imputed_final, log_true, logcounts)}

        results_raw.append(raw)
        results_no_bio.append(no_bio)
        results_final.append(final)
        mse_bz_list.append(float(final.get("mse_biozero", float("nan"))))
        mse_list.append(float(final.get("mse", float("nan"))))

    avg_mse_bz = _nanmean(mse_bz_list)
    avg_mse = _nanmean(mse_list)
    return avg_mse_bz, avg_mse, results_raw, results_no_bio, results_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked AE imputation with parameter autotuning.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for mask_impute9_*_mse_table.tsv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-mse", type=float, default=0.5, help="Weight for overall MSE in objective.")
    parser.add_argument("--max-evals", type=int, default=60, help="Max configs to evaluate.")
    parser.add_argument("--target-mse", type=float, default=None, help="Stop early if avg MSE is below this value.")
    parser.add_argument("--target-biozero", type=float, default=None, help="Stop early if avg biozero MSE is below this value.")
    parser.add_argument(
        "--calibrate-p-bio",
        action="store_true",
        help="Use logTrueCounts to calibrate p_bio thresholds (semi-supervised).",
    )
    parser.add_argument(
        "--calibrate-zero-threshold",
        action="store_true",
        help="Use logTrueCounts to calibrate an extra zeroing threshold on imputed values (semi-supervised).",
    )
    parser.add_argument(
        "--calibrate-zero-mode",
        type=str,
        default="global",
        help="Calibration mode for zero threshold: global or gene.",
    )
    parser.add_argument(
        "--calibrate-p-bio-mode",
        type=str,
        default="global",
        help="Calibration mode for p_bio: global, gene_quantile, or bin_shrink.",
    )
    parser.add_argument("--calibrate-temp-grid", type=str, default="0.6,0.8,1.0,1.2,1.5")
    parser.add_argument("--calibrate-bias-grid", type=str, default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--calibrate-thr-grid", type=str, default="-1,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--calibrate-frac-scale-grid", type=str, default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--calibrate-bin-count", type=int, default=5)
    parser.add_argument("--calibrate-zero-thr-grid", type=str, default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--calib-method-grid", type=str, default="platt,isotonic")
    parser.add_argument("--calib-blend-grid", type=str, default="1.0,0.8,0.6")
    parser.add_argument("--calib-platt-l2-grid", type=str, default="0,0.01,0.1")
    parser.add_argument("--calib-platt-max-iter-grid", type=str, default="120,200")
    parser.add_argument("--calib-platt-lr-grid", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--calib-platt-use-logit-grid", type=str, default="true")
    parser.add_argument("--calib-platt-balance-grid", type=str, default="true")
    parser.add_argument("--calib-max-samples-grid", type=str, default="20000,50000")
    parser.add_argument("--sup-bio-weight-grid", type=str, default="0.6,0.8,1.0")
    parser.add_argument("--sup-bio-l2-grid", type=str, default="0,0.01,0.1")
    parser.add_argument("--sup-bio-lr-grid", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--sup-bio-epochs-grid", type=str, default="120,200")
    parser.add_argument("--sup-bio-balance-grid", type=str, default="true")
    parser.add_argument("--sup-bio-max-samples-grid", type=str, default="20000,50000")
    parser.add_argument("--sup-bio-method-grid", type=str, default="gauss,logit")
    parser.add_argument("--sup-bio-min-count-grid", type=str, default="5,20")
    parser.add_argument("--sup-bio-var-floor-grid", type=str, default="1e-3,1e-2")
    parser.add_argument("--sup-bio-hidden-grid", type=str, default="16,32")
    parser.add_argument("--lowrank-weight-grid", type=str, default="0,0.2,0.4")
    parser.add_argument("--lowrank-rank-grid", type=str, default="16,32,64")
    parser.add_argument("--lowrank-iters-grid", type=str, default="1,2,3")
    parser.add_argument("--zero-scale-weight-grid", type=str, default="0,0.6,0.8,0.9,1.0")
    parser.add_argument("--zero-scale-mode-grid", type=str, default="gene")
    parser.add_argument("--zero-scale-bio-weight-grid", type=str, default="1,2,4,5,6")
    parser.add_argument("--zero-scale-max-samples-grid", type=str, default="0,20000")
    parser.add_argument("--zero-mix-weight-grid", type=str, default="0,0.5,0.8,1.0")
    parser.add_argument("--zero-mix-mode-grid", type=str, default="global,gene")
    parser.add_argument("--zero-mix-bio-weight-grid", type=str, default="0,1,2,4,6")
    parser.add_argument("--zero-mix-max-samples-grid", type=str, default="0,20000")
    parser.add_argument("--zero-mix-gamma-grid", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--zero-mix-max-scale-grid", type=str, default="1.0,1.2,1.5")
    parser.add_argument("--recover-weight-grid", type=str, default="0,0.2,0.4,0.6")
    parser.add_argument("--recover-pmax-grid", type=str, default="0.05,0.1,0.2,0.3")
    parser.add_argument("--recover-gamma-grid", type=str, default="1.0")
    parser.add_argument(
        "--constrained-zero-scale",
        action="store_true",
        help="Use logTrueCounts to choose per-gene zero scales that minimize biozero MSE with MSE constraint.",
    )
    parser.add_argument("--constrained-zero-max-mse-inc", type=float, default=0.01)
    parser.add_argument("--constrained-zero-lambda-max", type=float, default=1000.0)
    parser.add_argument("--constrained-zero-iters", type=int, default=30)
    parser.add_argument("--calibrate-lambda-mse", type=float, default=None, help="Override lambda for calibration objective.")

    parser.add_argument("--thr-drop-grid", type=str, default="0.6,0.7,0.8,0.85,0.9,0.95,0.98")
    parser.add_argument("--disp-mode-grid", type=str, default="estimate,fixed")
    parser.add_argument("--disp-const-grid", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--use-cell-factor-grid", type=str, default="true,false")
    parser.add_argument("--tau-dispersion-grid", type=str, default="10,20,40")
    parser.add_argument("--tau-group-dispersion-grid", type=str, default="20,50,80")
    parser.add_argument("--tau-dropout-grid", type=str, default="20,50,80")

    parser.add_argument("--p-zero-grid", type=str, default="0,0.05,0.1,0.2")
    parser.add_argument("--p-nz-grid", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--noise-min-grid", type=str, default="0,0.05")
    parser.add_argument("--noise-max-grid", type=str, default="0.25,0.5")
    parser.add_argument("--hidden-grid", type=str, default="64|128,64|256,128")
    parser.add_argument("--bottleneck-grid", type=str, default="16,32,64")
    parser.add_argument("--dropout-grid", type=str, default="0.0,0.05,0.1,0.2")
    parser.add_argument("--use-residual-grid", type=str, default="true,false")

    parser.add_argument("--epochs-grid", type=str, default="80,120,160")
    parser.add_argument("--batch-size-grid", type=str, default="32,64,128")
    parser.add_argument("--lr-grid", type=str, default="5e-4,1e-3,2e-3")
    parser.add_argument("--weight-decay-grid", type=str, default="0,1e-4,1e-3")
    parser.add_argument("--loss-bio-weight-grid", type=str, default="1,2,4,8")
    parser.add_argument("--loss-nz-weight-grid", type=str, default="1,2")
    parser.add_argument("--bio-reg-weight-grid", type=str, default="0,0.5,1.0,2.0,4.0")
    parser.add_argument("--recon-weight-grid", type=str, default="0,0.1,0.2,0.4")
    parser.add_argument("--p-low-grid", type=str, default="0.5,1,2")
    parser.add_argument("--p-high-grid", type=str, default="98,99,99.5")
    parser.add_argument("--post-threshold-grid", type=str, default="-1,0,0.1,0.25,0.5,0.75")
    parser.add_argument("--post-threshold-scale-grid", type=str, default="0,0.5,1.0,2.0")
    parser.add_argument("--post-threshold-gamma-grid", type=str, default="0.25,0.5,1,2,4")
    parser.add_argument("--post-gene-quantile-grid", type=str, default="-1,0.05,0.1,0.2,0.3")
    parser.add_argument("--post-gene-scale-grid", type=str, default="0.3,0.5,1.0,1.5")
    parser.add_argument("--post-gene-gamma-grid", type=str, default="0.5,1,2,4")
    parser.add_argument("--post-gene-ref-grid", type=str, default="obs,recon")
    parser.add_argument("--keep-positive-grid", type=str, default="true,false")
    parser.add_argument("--hard-zero-bio-grid", type=str, default="true,false")
    parser.add_argument("--oracle-bio-grid", type=str, default="false,true")
    parser.add_argument("--blend-alpha-grid", type=str, default="0,0.3,0.6")
    parser.add_argument("--blend-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--p-bio-temp-grid", type=str, default="0.4,0.6,0.8,1.0,1.3,1.6,2.0")
    parser.add_argument("--p-bio-bias-grid", type=str, default="-1,-0.5,0,0.5,1.0,1.5,2.0")
    parser.add_argument("--bio-model-grid", type=str, default="splat,poisson,mix")
    parser.add_argument("--bio-model-mix-grid", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--poisson-scale-grid", type=str, default="0.5,1,2,4")
    parser.add_argument("--post-bio-temp-grid", type=str, default="0.8,1.0,1.2,1.5")
    parser.add_argument("--post-bio-bias-grid", type=str, default="-0.5,0,0.5")
    parser.add_argument("--ae-bio-weight-grid", type=str, default="0,0.3,0.6,0.9,1.0")
    parser.add_argument("--ae-bio-cap-weight-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--ae-bio-temp-grid", type=str, default="0.1,0.3,0.6,1.0")
    parser.add_argument("--ae-bio-quantile-grid", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--ae-bio-union-grid", type=str, default="false,true")
    parser.add_argument("--gene-boost-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--gene-boost-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--gene-nz-boost-grid", type=str, default="0,0.2,0.4,0.6")
    parser.add_argument("--gene-nz-boost-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--gene-nz-mix-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--gene-nz-mix-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--gene-rare-threshold-grid", type=str, default="-1,0.05,0.1,0.2,0.3")
    parser.add_argument("--gene-rare-pbio-grid", type=str, default="0.5,0.7,0.9,1.0")
    parser.add_argument("--cell-zero-weight-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--cell-depth-boost-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--cell-depth-gamma-grid", type=str, default="1,2")
    parser.add_argument("--cluster-weight-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--cluster-gamma-grid", type=str, default="1,2")
    parser.add_argument("--cluster-k-grid", type=str, default="2,3,4")
    parser.add_argument("--cluster-pcs-grid", type=str, default="10,20")
    parser.add_argument("--shrink-alpha-grid", type=str, default="0,0.5,1,1.5,2.0,2.5,3.0")
    parser.add_argument("--shrink-gamma-grid", type=str, default="0.5,1,2,4")
    parser.add_argument("--zero-shrink-grid", type=str, default="1.0,0.8,0.6,0.4")
    parser.add_argument("--zero-cap-quantile-grid", type=str, default="-1,0.05,0.1,0.2,0.3")
    parser.add_argument("--zero-cap-scale-grid", type=str, default="1.0,1.5,2.0")
    parser.add_argument("--zero-cap-ref-grid", type=str, default="obs,recon")
    parser.add_argument("--expr-bio-weight-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--expr-bio-cap-weight-grid", type=str, default="0,0.3,0.6")
    parser.add_argument("--expr-bio-temp-grid", type=str, default="0.3,0.6,1.0")
    parser.add_argument("--expr-bio-bias-grid", type=str, default="-0.5,0,0.5")
    parser.add_argument("--expr-bio-quantile-grid", type=str, default="-1,0.2,0.4")
    parser.add_argument("--expr-bio-scale-grid", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--expr-bio-union-grid", type=str, default="false,true")
    parser.add_argument("--knn-bio-weight-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--expr-bio-expected-grid", type=str, default="log,counts,mix")
    parser.add_argument("--expr-bio-mix-grid", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--knn-bio-temp-grid", type=str, default="0.2,0.4,0.8")
    parser.add_argument("--knn-bio-quantile-grid", type=str, default="-1,0.2,0.4")
    parser.add_argument("--knn-k-grid", type=str, default="5,10,20")
    parser.add_argument("--knn-bio-union-grid", type=str, default="false,true")
    parser.add_argument("--clip-negative-grid", type=str, default="true,false")
    args = parser.parse_args()

    if args.max_evals < 1:
        raise ValueError("--max-evals must be >= 1.")
    if args.lambda_mse < 0:
        raise ValueError("--lambda-mse must be >= 0.")
    if args.target_mse is not None and args.target_mse < 0:
        raise ValueError("--target-mse must be >= 0.")
    if args.target_biozero is not None and args.target_biozero < 0:
        raise ValueError("--target-biozero must be >= 0.")
    if args.calibrate_lambda_mse is not None and args.calibrate_lambda_mse < 0:
        raise ValueError("--calibrate-lambda-mse must be >= 0.")
    if args.constrained_zero_max_mse_inc < 0:
        raise ValueError("--constrained-zero-max-mse-inc must be >= 0.")
    if args.constrained_zero_lambda_max <= 0:
        raise ValueError("--constrained-zero-lambda-max must be > 0.")
    if args.constrained_zero_iters < 1:
        raise ValueError("--constrained-zero-iters must be >= 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    calibrate_p_bio = bool(args.calibrate_p_bio)
    calibrate_zero_threshold = bool(args.calibrate_zero_threshold)
    calibrate_p_bio_mode = str(args.calibrate_p_bio_mode).lower()
    calibrate_zero_mode = str(args.calibrate_zero_mode).lower()
    if calibrate_p_bio_mode in ("gene", "gene-quantile"):
        calibrate_p_bio_mode = "gene_quantile"
    if calibrate_p_bio_mode == "bin-shrink":
        calibrate_p_bio_mode = "bin_shrink"
    if calibrate_p_bio_mode not in ("global", "gene_quantile", "bin_shrink"):
        raise ValueError("--calibrate-p-bio-mode must be 'global', 'gene_quantile', or 'bin_shrink'.")
    if calibrate_zero_mode not in ("global", "gene"):
        raise ValueError("--calibrate-zero-mode must be 'global' or 'gene'.")
    calibrate_bin_count = int(args.calibrate_bin_count)
    if calibrate_bin_count < 2:
        raise ValueError("--calibrate-bin-count must be >= 2.")
    calib_temp_list = _parse_float_list(args.calibrate_temp_grid, [1.0])
    calib_bias_list = _parse_float_list(args.calibrate_bias_grid, [0.0])
    calib_thr_list = _parse_float_list(args.calibrate_thr_grid, [-1.0])
    calib_frac_scale_list = _parse_float_list(args.calibrate_frac_scale_grid, [1.0])
    calib_zero_thr_list = _parse_float_list(args.calibrate_zero_thr_grid, [0.0])
    calibrate_lambda_mse = (
        float(args.calibrate_lambda_mse) if args.calibrate_lambda_mse is not None else float(args.lambda_mse)
    )
    constrained_zero_scale = bool(args.constrained_zero_scale)
    constrained_zero_max_mse_inc = float(args.constrained_zero_max_mse_inc)
    constrained_zero_lambda_max = float(args.constrained_zero_lambda_max)
    constrained_zero_iters = int(args.constrained_zero_iters)
    calib_method_list = _parse_str_list(args.calib_method_grid, ["none"])
    calib_blend_list = _parse_float_list(args.calib_blend_grid, [1.0])
    calib_platt_l2_list = _parse_float_list(args.calib_platt_l2_grid, [0.0])
    calib_platt_max_iter_list = _parse_int_list(args.calib_platt_max_iter_grid, [200])
    calib_platt_lr_list = _parse_float_list(args.calib_platt_lr_grid, [0.1])
    calib_platt_use_logit_list = _parse_bool_list(args.calib_platt_use_logit_grid, [False])
    calib_platt_balance_list = _parse_bool_list(args.calib_platt_balance_grid, [False])
    calib_max_samples_list = _parse_int_list(args.calib_max_samples_grid, [0])
    sup_bio_weight_list = _parse_float_list(args.sup_bio_weight_grid, [0.0])
    sup_bio_l2_list = _parse_float_list(args.sup_bio_l2_grid, [0.0])
    sup_bio_lr_list = _parse_float_list(args.sup_bio_lr_grid, [0.1])
    sup_bio_epochs_list = _parse_int_list(args.sup_bio_epochs_grid, [100])
    sup_bio_balance_list = _parse_bool_list(args.sup_bio_balance_grid, [False])
    sup_bio_max_samples_list = _parse_int_list(args.sup_bio_max_samples_grid, [0])
    sup_bio_method_list = _parse_str_list(args.sup_bio_method_grid, ["logit"])
    sup_bio_min_count_list = _parse_int_list(args.sup_bio_min_count_grid, [5])
    sup_bio_var_floor_list = _parse_float_list(args.sup_bio_var_floor_grid, [1e-3])
    sup_bio_hidden_list = _parse_int_list(args.sup_bio_hidden_grid, [16])
    lowrank_weight_list = _parse_float_list(args.lowrank_weight_grid, [0.0])
    lowrank_rank_list = _parse_int_list(args.lowrank_rank_grid, [0])
    lowrank_iters_list = _parse_int_list(args.lowrank_iters_grid, [0])
    zero_scale_weight_list = _parse_float_list(args.zero_scale_weight_grid, [0.0])
    zero_scale_mode_list = _parse_str_list(args.zero_scale_mode_grid, ["gene"])
    zero_scale_bio_weight_list = _parse_float_list(args.zero_scale_bio_weight_grid, [1.0])
    zero_scale_max_samples_list = _parse_int_list(args.zero_scale_max_samples_grid, [0])
    zero_mix_weight_list = _parse_float_list(args.zero_mix_weight_grid, [0.0])
    zero_mix_mode_list = _parse_str_list(args.zero_mix_mode_grid, ["global"])
    zero_mix_bio_weight_list = _parse_float_list(args.zero_mix_bio_weight_grid, [0.0])
    zero_mix_max_samples_list = _parse_int_list(args.zero_mix_max_samples_grid, [0])
    zero_mix_gamma_list = _parse_float_list(args.zero_mix_gamma_grid, [1.0])
    zero_mix_max_scale_list = _parse_float_list(args.zero_mix_max_scale_grid, [1.0])
    recover_weight_list = _parse_float_list(args.recover_weight_grid, [0.0])
    recover_pmax_list = _parse_float_list(args.recover_pmax_grid, [0.1])
    recover_gamma_list = _parse_float_list(args.recover_gamma_grid, [1.0])

    datasets: List[Dict[str, object]] = []
    for path in collect_rds_files(args.input_path):
        ds_name = path.stem
        print(f"\n=== {ds_name} ===")
        dataset = load_dataset(str(path))
        if dataset is None:
            print(f"[WARN] {ds_name}: missing logTrueCounts; skipping.")
            continue

        logcounts = dataset["logcounts"]
        log_true = dataset["log_true"]
        counts = dataset["counts"]

        if counts is None:
            counts_obs = np.clip(logcounts_to_counts(logcounts), 0.0, None)
        else:
            counts_obs = np.clip(counts, 0.0, None)
        zeros_obs = counts_obs <= 0.0
        counts_max = counts_obs.max(axis=0)
        nz_mask = logcounts > 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            gene_mean = np.sum(logcounts * nz_mask, axis=0) / np.maximum(nz_mask.sum(axis=0), 1)
        gene_mean = np.nan_to_num(gene_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        lo = float(np.percentile(gene_mean, GENE_NORM_LOW))
        hi = float(np.percentile(gene_mean, GENE_NORM_HIGH))
        span = max(hi - lo, EPSILON)
        gene_mean_norm = np.clip((gene_mean - lo) / span, 0.0, 1.0).astype(np.float32)
        gene_nz_frac = nz_mask.mean(axis=0).astype(np.float32)
        cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
        cz_lo = float(np.percentile(cell_zero_frac, GENE_NORM_LOW))
        cz_hi = float(np.percentile(cell_zero_frac, GENE_NORM_HIGH))
        cz_span = max(cz_hi - cz_lo, EPSILON)
        cell_zero_norm = np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)
        cell_nz = nz_mask
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_mean_nz = np.sum(logcounts * cell_nz, axis=1) / np.maximum(cell_nz.sum(axis=1), 1)
        cell_mean_nz = np.nan_to_num(cell_mean_nz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        global_mean = float(np.nanmean(gene_mean)) if gene_mean.size > 0 else 0.0
        expected_log = _expected_log_from_factors(cell_mean_nz, gene_mean, global_mean).astype(np.float32)
        cell_depth = counts_obs.sum(axis=1).astype(np.float32)
        n_cells = int(counts_obs.shape[0])
        gene_mean_counts = counts_obs.sum(axis=0).astype(np.float32) / max(n_cells, 1)
        total_mean_counts = float(gene_mean_counts.sum())
        if total_mean_counts < EPSILON:
            expected_counts = np.zeros_like(logcounts, dtype=np.float32)
            expected_log_counts = np.zeros_like(logcounts, dtype=np.float32)
        else:
            expected_counts = (cell_depth[:, None] * gene_mean_counts[None, :]) / total_mean_counts
            expected_counts = expected_counts.astype(np.float32)
            expected_log_counts = (np.log1p(expected_counts) / np.log(2.0)).astype(np.float32)
        cell_depth_log = np.log1p(cell_depth)
        cd_lo = float(np.percentile(cell_depth_log, GENE_NORM_LOW))
        cd_hi = float(np.percentile(cell_depth_log, GENE_NORM_HIGH))
        cd_span = max(cd_hi - cd_lo, EPSILON)
        cell_depth_norm = np.clip((cell_depth_log - cd_lo) / cd_span, 0.0, 1.0).astype(np.float32)

        datasets.append(
            {
                "dataset": ds_name,
                "logcounts": logcounts,
                "log_true": log_true,
                "zeros_obs": zeros_obs,
                "counts": counts_obs,
                "counts_max": counts_max,
                "gene_mean": gene_mean,
                "gene_mean_norm": gene_mean_norm,
                "gene_nz_frac": gene_nz_frac,
                "cell_zero_norm": cell_zero_norm,
                "cell_depth_norm": cell_depth_norm,
                "expected_log": expected_log,
                "expected_log_counts": expected_log_counts,
                "expected_counts": expected_counts,
                "p_bio_cache": {},
                "cluster_cache": {},
                "gene_q_cache": {},
            }
        )

    if not datasets:
        raise SystemExit("No datasets processed.")

    if calibrate_p_bio:
        print(
            f"[INFO] p_bio calibration enabled (semi-supervised). mode={calibrate_p_bio_mode} "
            f"temps={calib_temp_list} biases={calib_bias_list} thrs={calib_thr_list} "
            f"frac_scales={calib_frac_scale_list} bins={calibrate_bin_count} "
            f"methods={calib_method_list} lambda={calibrate_lambda_mse}"
        )

    thr_drop_list = _parse_float_list(args.thr_drop_grid, [DEFAULT_THR_DROP])
    disp_mode_list = _parse_str_list(args.disp_mode_grid, ["estimate", "fixed"])
    disp_const_list = _parse_float_list(args.disp_const_grid, [0.1])
    use_cell_factor_list = _parse_bool_list(args.use_cell_factor_grid, [True])
    tau_disp_list = _parse_float_list(args.tau_dispersion_grid, [20.0])
    tau_group_disp_list = _parse_float_list(args.tau_group_dispersion_grid, [50.0])
    tau_dropout_list = _parse_float_list(args.tau_dropout_grid, [50.0])

    p_zero_list = _parse_float_list(args.p_zero_grid, [0.0, 0.05, 0.1, 0.2])
    p_nz_list = _parse_float_list(args.p_nz_grid, [0.1, 0.2, 0.3, 0.4])
    noise_min_list = _parse_float_list(args.noise_min_grid, [0.0])
    noise_max_list = _parse_float_list(args.noise_max_grid, [0.25, 0.5])
    hidden_list = _parse_hidden_grid(args.hidden_grid, [[64], [128, 64]])
    bottleneck_list = _parse_int_list(args.bottleneck_grid, [16, 32, 64])
    dropout_list = _parse_float_list(args.dropout_grid, [0.0, 0.1])
    use_residual_list = _parse_bool_list(args.use_residual_grid, [True])

    epochs_list = _parse_int_list(args.epochs_grid, [60, 100])
    batch_size_list = _parse_int_list(args.batch_size_grid, [32, 64])
    lr_list = _parse_float_list(args.lr_grid, [1e-3])
    weight_decay_list = _parse_float_list(args.weight_decay_grid, [0.0])
    loss_bio_weight_list = _parse_float_list(args.loss_bio_weight_grid, [1.0])
    loss_nz_weight_list = _parse_float_list(args.loss_nz_weight_grid, [1.0])
    bio_reg_weight_list = _parse_float_list(args.bio_reg_weight_grid, [0.0])
    recon_weight_list = _parse_float_list(args.recon_weight_grid, [0.0])
    p_low_list = _parse_float_list(args.p_low_grid, [1.0])
    p_high_list = _parse_float_list(args.p_high_grid, [99.0])
    post_thr_list = _parse_float_list(args.post_threshold_grid, [0.5])
    post_scale_list = _parse_float_list(args.post_threshold_scale_grid, [0.0])
    post_gamma_list = _parse_float_list(args.post_threshold_gamma_grid, [1.0])
    post_gene_q_list = _parse_float_list(args.post_gene_quantile_grid, [-1.0])
    post_gene_scale_list = _parse_float_list(args.post_gene_scale_grid, [1.0])
    post_gene_gamma_list = _parse_float_list(args.post_gene_gamma_grid, [1.0])
    post_gene_ref_list = _parse_str_list(args.post_gene_ref_grid, ["obs"])
    keep_positive_list = _parse_bool_list(args.keep_positive_grid, [False])
    hard_zero_bio_list = _parse_bool_list(args.hard_zero_bio_grid, [True])
    oracle_bio_list = _parse_bool_list(args.oracle_bio_grid, [False])
    if any(oracle_bio_list):
        print("[WARN] oracle_bio forced to false for unsupervised run.")
    oracle_bio_list = [False]
    blend_alpha_list = _parse_float_list(args.blend_alpha_grid, [0.0])
    blend_gamma_list = _parse_float_list(args.blend_gamma_grid, [1.0])
    p_bio_temp_list = _parse_float_list(args.p_bio_temp_grid, [1.0])
    p_bio_bias_list = _parse_float_list(args.p_bio_bias_grid, [0.0])
    bio_model_list = _parse_str_list(args.bio_model_grid, ["splat"])
    bio_model_mix_list = _parse_float_list(args.bio_model_mix_grid, [0.5])
    poisson_scale_list = _parse_float_list(args.poisson_scale_grid, [1.0])
    post_bio_temp_list = _parse_float_list(args.post_bio_temp_grid, [1.0])
    post_bio_bias_list = _parse_float_list(args.post_bio_bias_grid, [0.0])
    ae_bio_weight_list = _parse_float_list(args.ae_bio_weight_grid, [0.0])
    ae_bio_cap_weight_list = _parse_float_list(args.ae_bio_cap_weight_grid, [0.0])
    ae_bio_temp_list = _parse_float_list(args.ae_bio_temp_grid, [0.5])
    ae_bio_quantile_list = _parse_float_list(args.ae_bio_quantile_grid, [0.2])
    ae_bio_union_list = _parse_bool_list(args.ae_bio_union_grid, [False])
    gene_boost_list = _parse_float_list(args.gene_boost_grid, [0.0])
    gene_boost_gamma_list = _parse_float_list(args.gene_boost_gamma_grid, [1.0])
    gene_nz_boost_list = _parse_float_list(args.gene_nz_boost_grid, [0.0])
    gene_nz_boost_gamma_list = _parse_float_list(args.gene_nz_boost_gamma_grid, [1.0])
    gene_nz_mix_list = _parse_float_list(args.gene_nz_mix_grid, [0.0])
    gene_nz_mix_gamma_list = _parse_float_list(args.gene_nz_mix_gamma_grid, [1.0])
    gene_rare_threshold_list = _parse_float_list(args.gene_rare_threshold_grid, [-1.0])
    gene_rare_pbio_list = _parse_float_list(args.gene_rare_pbio_grid, [0.0])
    cell_zero_weight_list = _parse_float_list(args.cell_zero_weight_grid, [0.0])
    cell_depth_boost_list = _parse_float_list(args.cell_depth_boost_grid, [0.0])
    cell_depth_gamma_list = _parse_float_list(args.cell_depth_gamma_grid, [1.0])
    cluster_weight_list = _parse_float_list(args.cluster_weight_grid, [0.0])
    cluster_gamma_list = _parse_float_list(args.cluster_gamma_grid, [1.0])
    cluster_k_list = _parse_int_list(args.cluster_k_grid, [2])
    cluster_pcs_list = _parse_int_list(args.cluster_pcs_grid, [10])
    shrink_alpha_list = _parse_float_list(args.shrink_alpha_grid, [0.0])
    shrink_gamma_list = _parse_float_list(args.shrink_gamma_grid, [1.0])
    zero_shrink_list = _parse_float_list(args.zero_shrink_grid, [1.0])
    zero_cap_quantile_list = _parse_float_list(args.zero_cap_quantile_grid, [-1.0])
    zero_cap_scale_list = _parse_float_list(args.zero_cap_scale_grid, [1.0])
    zero_cap_ref_list = _parse_str_list(args.zero_cap_ref_grid, ["obs"])
    expr_bio_weight_list = _parse_float_list(args.expr_bio_weight_grid, [0.0])
    expr_bio_cap_weight_list = _parse_float_list(args.expr_bio_cap_weight_grid, [0.0])
    expr_bio_temp_list = _parse_float_list(args.expr_bio_temp_grid, [0.5])
    expr_bio_bias_list = _parse_float_list(args.expr_bio_bias_grid, [0.0])
    expr_bio_quantile_list = _parse_float_list(args.expr_bio_quantile_grid, [-1.0])
    expr_bio_scale_list = _parse_float_list(args.expr_bio_scale_grid, [1.0])
    expr_bio_union_list = _parse_bool_list(args.expr_bio_union_grid, [False])
    expr_bio_expected_list = _parse_str_list(args.expr_bio_expected_grid, ["log"])
    expr_bio_mix_list = _parse_float_list(args.expr_bio_mix_grid, [0.5])
    knn_bio_weight_list = _parse_float_list(args.knn_bio_weight_grid, [0.0])
    knn_bio_temp_list = _parse_float_list(args.knn_bio_temp_grid, [0.5])
    knn_bio_quantile_list = _parse_float_list(args.knn_bio_quantile_grid, [-1.0])
    knn_k_list = _parse_int_list(args.knn_k_grid, [10])
    knn_bio_union_list = _parse_bool_list(args.knn_bio_union_grid, [False])
    clip_negative_list = _parse_bool_list(args.clip_negative_grid, [True])

    space = {
        "thr_drop": thr_drop_list,
        "disp_mode": disp_mode_list,
        "disp_const": disp_const_list,
        "use_cell_factor": use_cell_factor_list,
        "tau_dispersion": tau_disp_list,
        "tau_group_dispersion": tau_group_disp_list,
        "tau_dropout": tau_dropout_list,
        "p_zero": p_zero_list,
        "p_nz": p_nz_list,
        "noise_min": noise_min_list,
        "noise_max": noise_max_list,
        "hidden": hidden_list,
        "bottleneck": bottleneck_list,
        "dropout": dropout_list,
        "use_residual": use_residual_list,
        "epochs": epochs_list,
        "batch_size": batch_size_list,
        "lr": lr_list,
        "weight_decay": weight_decay_list,
        "loss_bio_weight": loss_bio_weight_list,
        "loss_nz_weight": loss_nz_weight_list,
        "bio_reg_weight": bio_reg_weight_list,
        "recon_weight": recon_weight_list,
        "p_low": p_low_list,
        "p_high": p_high_list,
        "post_threshold": post_thr_list,
        "post_threshold_scale": post_scale_list,
        "post_threshold_gamma": post_gamma_list,
        "post_gene_quantile": post_gene_q_list,
        "post_gene_scale": post_gene_scale_list,
        "post_gene_gamma": post_gene_gamma_list,
        "post_gene_ref": post_gene_ref_list,
        "keep_positive": keep_positive_list,
        "hard_zero_bio": hard_zero_bio_list,
        "oracle_bio": oracle_bio_list,
        "blend_alpha": blend_alpha_list,
        "blend_gamma": blend_gamma_list,
        "p_bio_temp": p_bio_temp_list,
        "p_bio_bias": p_bio_bias_list,
        "bio_model": bio_model_list,
        "bio_model_mix": bio_model_mix_list,
        "poisson_scale": poisson_scale_list,
        "post_bio_temp": post_bio_temp_list,
        "post_bio_bias": post_bio_bias_list,
        "calib_method": calib_method_list,
        "calib_blend": calib_blend_list,
        "calib_platt_l2": calib_platt_l2_list,
        "calib_platt_max_iter": calib_platt_max_iter_list,
        "calib_platt_lr": calib_platt_lr_list,
        "calib_platt_use_logit": calib_platt_use_logit_list,
        "calib_platt_balance": calib_platt_balance_list,
        "calib_max_samples": calib_max_samples_list,
        "sup_bio_weight": sup_bio_weight_list,
        "sup_bio_l2": sup_bio_l2_list,
        "sup_bio_lr": sup_bio_lr_list,
        "sup_bio_epochs": sup_bio_epochs_list,
        "sup_bio_balance": sup_bio_balance_list,
        "sup_bio_max_samples": sup_bio_max_samples_list,
        "sup_bio_method": sup_bio_method_list,
        "sup_bio_min_count": sup_bio_min_count_list,
        "sup_bio_var_floor": sup_bio_var_floor_list,
        "sup_bio_hidden": sup_bio_hidden_list,
        "lowrank_weight": lowrank_weight_list,
        "lowrank_rank": lowrank_rank_list,
        "lowrank_iters": lowrank_iters_list,
        "zero_scale_weight": zero_scale_weight_list,
        "zero_scale_mode": zero_scale_mode_list,
        "zero_scale_bio_weight": zero_scale_bio_weight_list,
        "zero_scale_max_samples": zero_scale_max_samples_list,
        "zero_mix_weight": zero_mix_weight_list,
        "zero_mix_mode": zero_mix_mode_list,
        "zero_mix_bio_weight": zero_mix_bio_weight_list,
        "zero_mix_max_samples": zero_mix_max_samples_list,
        "zero_mix_gamma": zero_mix_gamma_list,
        "zero_mix_max_scale": zero_mix_max_scale_list,
        "recover_weight": recover_weight_list,
        "recover_pmax": recover_pmax_list,
        "recover_gamma": recover_gamma_list,
        "ae_bio_weight": ae_bio_weight_list,
        "ae_bio_cap_weight": ae_bio_cap_weight_list,
        "ae_bio_temp": ae_bio_temp_list,
        "ae_bio_quantile": ae_bio_quantile_list,
        "ae_bio_union": ae_bio_union_list,
        "gene_boost": gene_boost_list,
        "gene_boost_gamma": gene_boost_gamma_list,
        "gene_nz_boost": gene_nz_boost_list,
        "gene_nz_boost_gamma": gene_nz_boost_gamma_list,
        "gene_nz_mix": gene_nz_mix_list,
        "gene_nz_mix_gamma": gene_nz_mix_gamma_list,
        "gene_rare_threshold": gene_rare_threshold_list,
        "gene_rare_pbio": gene_rare_pbio_list,
        "cell_zero_weight": cell_zero_weight_list,
        "cell_depth_boost": cell_depth_boost_list,
        "cell_depth_gamma": cell_depth_gamma_list,
        "cluster_weight": cluster_weight_list,
        "cluster_gamma": cluster_gamma_list,
        "cluster_k": cluster_k_list,
        "cluster_pcs": cluster_pcs_list,
        "shrink_alpha": shrink_alpha_list,
        "shrink_gamma": shrink_gamma_list,
        "zero_shrink": zero_shrink_list,
        "zero_cap_quantile": zero_cap_quantile_list,
        "zero_cap_scale": zero_cap_scale_list,
        "zero_cap_ref": zero_cap_ref_list,
        "expr_bio_weight": expr_bio_weight_list,
        "expr_bio_cap_weight": expr_bio_cap_weight_list,
        "expr_bio_temp": expr_bio_temp_list,
        "expr_bio_bias": expr_bio_bias_list,
        "expr_bio_quantile": expr_bio_quantile_list,
        "expr_bio_scale": expr_bio_scale_list,
        "expr_bio_union": expr_bio_union_list,
        "expr_bio_expected": expr_bio_expected_list,
        "expr_bio_mix": expr_bio_mix_list,
        "knn_bio_weight": knn_bio_weight_list,
        "knn_bio_temp": knn_bio_temp_list,
        "knn_bio_quantile": knn_bio_quantile_list,
        "knn_k": knn_k_list,
        "knn_bio_union": knn_bio_union_list,
        "clip_negative": clip_negative_list,
    }

    rng = np.random.default_rng(int(args.seed))
    all_configs: List[Dict[str, object]] = []
    keys = list(space.keys())
    total = 1
    for key in keys:
        total *= len(space[key])
        if total > max(int(args.max_evals), 5000):
            break

    if total <= int(args.max_evals) and total <= 5000:
        for combo in itertools.product(*[space[key] for key in keys]):
            cfg = {}
            for key, val in zip(keys, combo):
                cfg[key] = list(val) if key == "hidden" else val
            if not _is_valid_config(cfg):
                continue
            all_configs.append(cfg)
    else:
        attempts = 0
        max_attempts = int(args.max_evals) * 200
        seen = set()
        while len(all_configs) < int(args.max_evals) and attempts < max_attempts:
            attempts += 1
            cfg = {}
            for key in keys:
                values = space[key]
                idx = int(rng.integers(0, len(values)))
                val = values[idx]
                if key == "hidden":
                    val = list(val)
                cfg[key] = val
            if not _is_valid_config(cfg):
                continue
            cfg_key = _config_key(cfg, keys)
            if cfg_key in seen:
                continue
            seen.add(cfg_key)
            all_configs.append(cfg)

    if not all_configs:
        raise SystemExit("No valid configurations to evaluate.")

    tuning_rows: List[Dict[str, object]] = []
    best_obj = float("inf")
    best_avg_bz = float("inf")
    best_avg_mse = float("inf")
    best_cfg = None
    best_raw: List[Dict[str, object]] = []
    best_no_bio: List[Dict[str, object]] = []
    best_final: List[Dict[str, object]] = []
    best_feasible = None
    best_feasible_avg_bz = float("inf")
    best_feasible_avg_mse = float("inf")
    best_feasible_raw: List[Dict[str, object]] = []
    best_feasible_no_bio: List[Dict[str, object]] = []
    best_feasible_final: List[Dict[str, object]] = []

    for idx, cfg in enumerate(all_configs, start=1):
        avg_bz, avg_mse, raw, no_bio, final = evaluate_config(
            datasets=datasets,
            config=cfg,
            device=device,
            seed=int(args.seed),
            calibrate_p_bio=calibrate_p_bio,
            calibrate_zero_threshold=calibrate_zero_threshold,
            calibrate_zero_mode=calibrate_zero_mode,
            calib_temp_list=calib_temp_list,
            calib_bias_list=calib_bias_list,
            calib_thr_list=calib_thr_list,
            calibrate_p_bio_mode=calibrate_p_bio_mode,
            calib_frac_scale_list=calib_frac_scale_list,
            calibrate_bin_count=calibrate_bin_count,
            calib_zero_thr_list=calib_zero_thr_list,
            calibrate_lambda_mse=calibrate_lambda_mse,
            constrained_zero_scale=constrained_zero_scale,
            constrained_zero_max_mse_inc=constrained_zero_max_mse_inc,
            constrained_zero_lambda_max=constrained_zero_lambda_max,
            constrained_zero_iters=constrained_zero_iters,
            target_mse=args.target_mse,
            target_biozero=args.target_biozero,
        )
        obj = avg_bz + float(args.lambda_mse) * avg_mse

        tuning_rows.append(
            {
                "config_id": idx,
                "objective": obj,
                "avg_mse_biozero": avg_bz,
                "avg_mse": avg_mse,
                "thr_drop": cfg["thr_drop"],
                "disp_mode": cfg["disp_mode"],
                "disp_const": cfg["disp_const"],
                "use_cell_factor": cfg["use_cell_factor"],
                "tau_dispersion": cfg["tau_dispersion"],
                "tau_group_dispersion": cfg["tau_group_dispersion"],
                "tau_dropout": cfg["tau_dropout"],
                "p_zero": cfg["p_zero"],
                "p_nz": cfg["p_nz"],
                "noise_min": cfg["noise_min"],
                "noise_max": cfg["noise_max"],
                "hidden": ",".join(str(v) for v in cfg["hidden"]),
                "bottleneck": cfg["bottleneck"],
                "dropout": cfg["dropout"],
                "use_residual": cfg["use_residual"],
                "epochs": cfg["epochs"],
                "batch_size": cfg["batch_size"],
                "lr": cfg["lr"],
                "weight_decay": cfg["weight_decay"],
                "loss_bio_weight": cfg["loss_bio_weight"],
                "loss_nz_weight": cfg["loss_nz_weight"],
                "bio_reg_weight": cfg["bio_reg_weight"],
                "recon_weight": cfg["recon_weight"],
                "p_low": cfg["p_low"],
                "p_high": cfg["p_high"],
                "post_threshold": cfg["post_threshold"],
                "post_threshold_scale": cfg["post_threshold_scale"],
                "post_threshold_gamma": cfg["post_threshold_gamma"],
                "post_gene_quantile": cfg["post_gene_quantile"],
                "post_gene_scale": cfg["post_gene_scale"],
                "post_gene_gamma": cfg["post_gene_gamma"],
                "post_gene_ref": cfg["post_gene_ref"],
                "keep_positive": cfg["keep_positive"],
                "hard_zero_bio": cfg["hard_zero_bio"],
                "oracle_bio": cfg["oracle_bio"],
                "blend_alpha": cfg["blend_alpha"],
                "blend_gamma": cfg["blend_gamma"],
                "p_bio_temp": cfg["p_bio_temp"],
                "p_bio_bias": cfg["p_bio_bias"],
                "bio_model": cfg.get("bio_model", "splat"),
                "bio_model_mix": cfg.get("bio_model_mix", 0.5),
                "poisson_scale": cfg.get("poisson_scale", 1.0),
                "post_bio_temp": cfg.get("post_bio_temp", 1.0),
                "post_bio_bias": cfg.get("post_bio_bias", 0.0),
                "calib_method": cfg.get("calib_method", "none"),
                "calib_blend": cfg.get("calib_blend", 1.0),
                "calib_platt_l2": cfg.get("calib_platt_l2", 0.0),
                "calib_platt_max_iter": cfg.get("calib_platt_max_iter", 200),
                "calib_platt_lr": cfg.get("calib_platt_lr", 0.1),
                "calib_platt_use_logit": cfg.get("calib_platt_use_logit", False),
                "calib_platt_balance": cfg.get("calib_platt_balance", False),
                "calib_max_samples": cfg.get("calib_max_samples", 0),
                "sup_bio_weight": cfg.get("sup_bio_weight", 0.0),
                "sup_bio_l2": cfg.get("sup_bio_l2", 0.0),
                "sup_bio_lr": cfg.get("sup_bio_lr", 0.1),
                "sup_bio_epochs": cfg.get("sup_bio_epochs", 100),
                "sup_bio_balance": cfg.get("sup_bio_balance", False),
                "sup_bio_max_samples": cfg.get("sup_bio_max_samples", 0),
                "sup_bio_method": cfg.get("sup_bio_method", "logit"),
                "sup_bio_min_count": cfg.get("sup_bio_min_count", 5),
                "sup_bio_var_floor": cfg.get("sup_bio_var_floor", 1e-3),
                "sup_bio_hidden": cfg.get("sup_bio_hidden", 16),
                "lowrank_weight": cfg.get("lowrank_weight", 0.0),
                "lowrank_rank": cfg.get("lowrank_rank", 0),
                "lowrank_iters": cfg.get("lowrank_iters", 0),
                "zero_scale_weight": cfg.get("zero_scale_weight", 0.0),
                "zero_scale_mode": cfg.get("zero_scale_mode", "gene"),
                "zero_scale_bio_weight": cfg.get("zero_scale_bio_weight", 0.0),
                "zero_scale_max_samples": cfg.get("zero_scale_max_samples", 0),
                "zero_mix_weight": cfg.get("zero_mix_weight", 0.0),
                "zero_mix_mode": cfg.get("zero_mix_mode", "global"),
                "zero_mix_bio_weight": cfg.get("zero_mix_bio_weight", 0.0),
                "zero_mix_max_samples": cfg.get("zero_mix_max_samples", 0),
                "zero_mix_gamma": cfg.get("zero_mix_gamma", 1.0),
                "zero_mix_max_scale": cfg.get("zero_mix_max_scale", 1.0),
                "recover_weight": cfg.get("recover_weight", 0.0),
                "recover_pmax": cfg.get("recover_pmax", 0.0),
                "recover_gamma": cfg.get("recover_gamma", 1.0),
                "constrained_zero_scale": constrained_zero_scale,
                "constrained_zero_max_mse_inc": constrained_zero_max_mse_inc,
                "constrained_zero_lambda_max": constrained_zero_lambda_max,
                "constrained_zero_iters": constrained_zero_iters,
                "ae_bio_weight": cfg["ae_bio_weight"],
                "ae_bio_cap_weight": cfg.get("ae_bio_cap_weight", 0.0),
                "ae_bio_temp": cfg["ae_bio_temp"],
                "ae_bio_quantile": cfg["ae_bio_quantile"],
                "ae_bio_union": cfg["ae_bio_union"],
                "gene_boost": cfg["gene_boost"],
                "gene_boost_gamma": cfg["gene_boost_gamma"],
                "gene_nz_boost": cfg["gene_nz_boost"],
                "gene_nz_boost_gamma": cfg["gene_nz_boost_gamma"],
                "gene_nz_mix": cfg["gene_nz_mix"],
                "gene_nz_mix_gamma": cfg["gene_nz_mix_gamma"],
                "gene_rare_threshold": cfg["gene_rare_threshold"],
                "gene_rare_pbio": cfg["gene_rare_pbio"],
                "cell_zero_weight": cfg["cell_zero_weight"],
                "cell_depth_boost": cfg["cell_depth_boost"],
                "cell_depth_gamma": cfg["cell_depth_gamma"],
                "cluster_weight": cfg["cluster_weight"],
                "cluster_gamma": cfg["cluster_gamma"],
                "cluster_k": cfg["cluster_k"],
                "cluster_pcs": cfg["cluster_pcs"],
                "shrink_alpha": cfg["shrink_alpha"],
                "shrink_gamma": cfg["shrink_gamma"],
                "zero_shrink": cfg.get("zero_shrink", 1.0),
                "zero_cap_quantile": cfg.get("zero_cap_quantile", -1.0),
                "zero_cap_scale": cfg.get("zero_cap_scale", 1.0),
                "zero_cap_ref": cfg.get("zero_cap_ref", "obs"),
                "expr_bio_weight": cfg["expr_bio_weight"],
                "expr_bio_cap_weight": cfg.get("expr_bio_cap_weight", 0.0),
                "expr_bio_temp": cfg["expr_bio_temp"],
                "expr_bio_bias": cfg["expr_bio_bias"],
                "expr_bio_quantile": cfg["expr_bio_quantile"],
                "expr_bio_scale": cfg["expr_bio_scale"],
                "expr_bio_union": cfg["expr_bio_union"],
                "expr_bio_expected": cfg.get("expr_bio_expected", "log"),
                "expr_bio_mix": cfg.get("expr_bio_mix", 0.5),
                "knn_bio_weight": cfg["knn_bio_weight"],
                "knn_bio_temp": cfg["knn_bio_temp"],
                "knn_bio_quantile": cfg["knn_bio_quantile"],
                "knn_k": cfg["knn_k"],
                "knn_bio_union": cfg["knn_bio_union"],
                "clip_negative": cfg["clip_negative"],
                "calibrate_p_bio": calibrate_p_bio,
                "calibrate_zero_threshold": calibrate_zero_threshold,
                "calibrate_zero_mode": calibrate_zero_mode,
                "calibrate_p_bio_mode": calibrate_p_bio_mode,
                "calibrate_frac_scale_grid": ",".join(str(v) for v in calib_frac_scale_list),
                "calibrate_zero_thr_grid": ",".join(str(v) for v in calib_zero_thr_list),
                "calibrate_bin_count": calibrate_bin_count,
                "calibrate_lambda_mse": calibrate_lambda_mse,
            }
        )

        if (obj < best_obj) or (abs(obj - best_obj) <= 1e-12 and avg_bz < best_avg_bz) or (
            abs(obj - best_obj) <= 1e-12 and abs(avg_bz - best_avg_bz) <= 1e-12 and avg_mse < best_avg_mse
        ):
            best_obj = obj
            best_avg_bz = avg_bz
            best_avg_mse = avg_mse
            best_cfg = cfg
            best_raw, best_no_bio, best_final = raw, no_bio, final

        if args.target_mse is not None or args.target_biozero is not None:
            mse_ok = args.target_mse is None or avg_mse <= float(args.target_mse)
            bz_ok = args.target_biozero is None or avg_bz <= float(args.target_biozero)
            if mse_ok and bz_ok:
                if (best_feasible is None) or (avg_mse < best_feasible_avg_mse) or (
                    abs(avg_mse - best_feasible_avg_mse) <= 1e-12 and avg_bz < best_feasible_avg_bz
                ):
                    best_feasible = cfg
                    best_feasible_avg_bz = avg_bz
                    best_feasible_avg_mse = avg_mse
                    best_feasible_raw, best_feasible_no_bio, best_feasible_final = raw, no_bio, final

        print(
            f"[{idx}/{len(all_configs)}] obj={obj:.6f} avg_bz={avg_bz:.6f} avg_mse={avg_mse:.6f} "
            f"thr_drop={cfg['thr_drop']} p_zero={cfg['p_zero']} p_nz={cfg['p_nz']} "
            f"noise={cfg['noise_min']}..{cfg['noise_max']} hidden={cfg['hidden']} "
            f"bottleneck={cfg['bottleneck']} dropout={cfg['dropout']} epochs={cfg['epochs']} "
            f"batch={cfg['batch_size']} lr={cfg['lr']} recon_w={cfg['recon_weight']} "
            f"bio_reg={cfg['bio_reg_weight']} "
            f"shrink={cfg['shrink_alpha']}/{cfg['shrink_gamma']} "
            f"zshrink={cfg.get('zero_shrink', 1.0)} "
            f"zcap={cfg.get('zero_cap_quantile', -1.0)}/{cfg.get('zero_cap_scale', 1.0)}/"
            f"{cfg.get('zero_cap_ref', 'obs')} "
            f"blend={cfg['blend_alpha']}/{cfg['blend_gamma']} "
            f"gene_boost={cfg['gene_boost']}/{cfg['gene_boost_gamma']} "
            f"gene_nz_boost={cfg['gene_nz_boost']}/{cfg['gene_nz_boost_gamma']} "
            f"gene_nz_mix={cfg['gene_nz_mix']}/{cfg['gene_nz_mix_gamma']} "
            f"gene_rare={cfg['gene_rare_threshold']}/{cfg['gene_rare_pbio']} "
            f"cell_zero_weight={cfg['cell_zero_weight']} "
            f"cell_depth={cfg['cell_depth_boost']}/{cfg['cell_depth_gamma']} "
            f"cluster={cfg['cluster_weight']}/{cfg['cluster_gamma']}/{cfg['cluster_k']}pcs{cfg['cluster_pcs']} "
            f"post_thr={cfg['post_threshold']} ps={cfg['post_threshold_scale']}/{cfg['post_threshold_gamma']} "
            f"pg={cfg['post_gene_quantile']}/{cfg['post_gene_scale']}/{cfg['post_gene_gamma']}/{cfg['post_gene_ref']} "
            f"hard_zero_bio={cfg['hard_zero_bio']} oracle_bio={cfg['oracle_bio']} "
            f"p_bio_temp={cfg['p_bio_temp']} p_bio_bias={cfg['p_bio_bias']} "
            f"bio_model={cfg.get('bio_model', 'splat')}/{cfg.get('bio_model_mix', 0.5)}/"
            f"{cfg.get('poisson_scale', 1.0)} "
            f"post_bio={cfg.get('post_bio_temp', 1.0)}/{cfg.get('post_bio_bias', 0.0)} "
            f"ae_bio={cfg['ae_bio_weight']}/{cfg['ae_bio_temp']}/{cfg['ae_bio_quantile']}/"
            f"u{cfg['ae_bio_union']}/cap{cfg.get('ae_bio_cap_weight', 0.0)} "
            f"expr_bio={cfg['expr_bio_weight']}/cap{cfg.get('expr_bio_cap_weight', 0.0)}/"
            f"{cfg['expr_bio_temp']}/{cfg['expr_bio_bias']}/"
            f"{cfg['expr_bio_quantile']}/{cfg['expr_bio_scale']}/u{cfg['expr_bio_union']}/"
            f"{cfg.get('expr_bio_expected', 'log')}/{cfg.get('expr_bio_mix', 0.5)} "
            f"knn_bio={cfg['knn_bio_weight']}/{cfg['knn_bio_temp']}/{cfg['knn_bio_quantile']}/"
            f"k{cfg['knn_k']}/u{cfg['knn_bio_union']} "
            f"keep_pos={cfg['keep_positive']} "
            f"lowrank={cfg.get('lowrank_weight', 0.0)}/{cfg.get('lowrank_rank', 0)}/{cfg.get('lowrank_iters', 0)} "
            f"zero_scale={cfg.get('zero_scale_weight', 0.0)}/{cfg.get('zero_scale_mode', 'gene')}/"
            f"{cfg.get('zero_scale_bio_weight', 0.0)} "
            f"zero_mix={cfg.get('zero_mix_weight', 0.0)}/{cfg.get('zero_mix_mode', 'global')}/"
            f"{cfg.get('zero_mix_bio_weight', 0.0)}/{cfg.get('zero_mix_gamma', 1.0)}/"
            f"{cfg.get('zero_mix_max_scale', 1.0)} "
            f"recover={cfg.get('recover_weight', 0.0)}/{cfg.get('recover_pmax', 0.0)}/"
            f"{cfg.get('recover_gamma', 1.0)} "
            f"constrained_zero={constrained_zero_scale}/{constrained_zero_max_mse_inc} "
            f"calib_p_bio={calibrate_p_bio} calib_method={cfg.get('calib_method', 'none')} "
            f"sup_bio={cfg.get('sup_bio_weight', 0.0)}/{cfg.get('sup_bio_method', 'logit')}"
        )

        if args.target_mse is not None or args.target_biozero is not None:
            mse_ok = args.target_mse is None or avg_mse <= float(args.target_mse)
            bz_ok = args.target_biozero is None or avg_bz <= float(args.target_biozero)
            if mse_ok and bz_ok:
                print("Target metrics reached; stopping early.")
                break

    if best_feasible is not None:
        best_cfg = best_feasible
        best_avg_bz = best_feasible_avg_bz
        best_avg_mse = best_feasible_avg_mse
        best_raw = best_feasible_raw
        best_no_bio = best_feasible_no_bio
        best_final = best_feasible_final
        best_obj = best_avg_bz + float(args.lambda_mse) * best_avg_mse

    if best_cfg is None:
        raise SystemExit("No successful evaluations.")

    def _write_table(path: Path, rows: List[Dict[str, object]]) -> None:
        columns = [
            "dataset",
            "thr_drop",
            "thr_bio",
            "disp_mode",
            "disp_const",
            "use_cell_factor",
            "tau_dispersion",
            "tau_group_dispersion",
            "tau_dropout",
            "p_zero",
            "p_nz",
            "noise_min",
            "noise_max",
            "hidden",
            "bottleneck",
            "dropout",
            "use_residual",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "loss_bio_weight",
            "loss_nz_weight",
            "bio_reg_weight",
            "recon_weight",
            "p_low",
            "p_high",
            "post_threshold",
            "post_threshold_scale",
            "post_threshold_gamma",
            "post_gene_quantile",
            "post_gene_scale",
            "post_gene_gamma",
            "post_gene_ref",
            "keep_positive",
            "hard_zero_bio",
            "oracle_bio",
            "blend_alpha",
            "blend_gamma",
            "p_bio_temp",
            "p_bio_bias",
            "bio_model",
            "bio_model_mix",
            "poisson_scale",
            "post_bio_temp",
            "post_bio_bias",
            "calibrate_p_bio",
            "calibrate_zero_threshold",
            "calibrate_zero_mode",
            "calib_method",
            "calib_blend",
            "calib_platt_l2",
            "calib_platt_max_iter",
            "calib_platt_lr",
            "calib_platt_use_logit",
            "calib_platt_balance",
            "calib_max_samples",
            "calib_platt_a",
            "calib_platt_b",
            "calib_platt_used",
            "calib_iso_blocks",
            "calib_zero_thr",
            "sup_bio_weight",
            "sup_bio_l2",
            "sup_bio_lr",
            "sup_bio_epochs",
            "sup_bio_balance",
            "sup_bio_max_samples",
            "sup_bio_method",
            "sup_bio_min_count",
            "sup_bio_var_floor",
            "sup_bio_hidden",
            "sup_bio_used",
            "lowrank_weight",
            "lowrank_rank",
            "lowrank_iters",
            "zero_scale_weight",
            "zero_scale_mode",
            "zero_scale_bio_weight",
            "zero_scale_max_samples",
            "zero_scale_mean",
            "zero_mix_weight",
            "zero_mix_mode",
            "zero_mix_bio_weight",
            "zero_mix_max_samples",
            "zero_mix_gamma",
            "zero_mix_max_scale",
            "zero_mix_scale_drop",
            "zero_mix_scale_bio",
            "recover_weight",
            "recover_pmax",
            "recover_gamma",
            "constrained_zero_scale",
            "constrained_zero_max_mse_inc",
            "constrained_zero_lambda_max",
            "constrained_zero_iters",
            "constrained_zero_lambda",
            "calib_p_bio_temp",
            "calib_p_bio_bias",
            "calib_p_bio_thr",
            "calib_p_bio_mode",
            "calib_p_bio_frac_scale",
            "calib_p_bio_bin_count",
            "ae_bio_weight",
            "ae_bio_cap_weight",
            "ae_bio_temp",
            "ae_bio_quantile",
            "ae_bio_union",
            "gene_boost",
            "gene_boost_gamma",
            "gene_nz_boost",
            "gene_nz_boost_gamma",
            "gene_nz_mix",
            "gene_nz_mix_gamma",
            "gene_rare_threshold",
            "gene_rare_pbio",
            "cell_zero_weight",
            "cell_depth_boost",
            "cell_depth_gamma",
            "cluster_weight",
            "cluster_gamma",
            "cluster_k",
            "cluster_pcs",
            "shrink_alpha",
            "shrink_gamma",
            "zero_shrink",
            "zero_cap_quantile",
            "zero_cap_scale",
            "zero_cap_ref",
            "expr_bio_weight",
            "expr_bio_cap_weight",
            "expr_bio_temp",
            "expr_bio_bias",
            "expr_bio_quantile",
            "expr_bio_scale",
            "expr_bio_union",
            "expr_bio_expected",
            "expr_bio_mix",
            "knn_bio_weight",
            "knn_bio_temp",
            "knn_bio_quantile",
            "knn_k",
            "knn_bio_union",
            "clip_negative",
            "mse",
            "mse_dropout",
            "mse_biozero",
            "mse_non_zero",
            "n_total",
            "n_dropout",
            "n_biozero",
            "n_non_zero",
            "n_obs_zero",
            "n_pred_bio",
        ]
        with path.open("w", encoding="utf-8") as f:
            f.write("\t".join(columns) + "\n")
            for row in sorted(rows, key=lambda r: str(r["dataset"])):
                f.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")

    _write_table(output_dir / "mask_impute9_raw_mse_table.tsv", best_raw)
    _write_table(output_dir / "mask_impute9_no_biozero_mse_table.tsv", best_no_bio)
    _write_table(output_dir / "mask_impute9_mse_table.tsv", best_final)

    tuning_path = output_dir / "mask_impute9_tuning.tsv"
    tuning_cols = [
        "config_id",
        "objective",
        "avg_mse_biozero",
        "avg_mse",
        "thr_drop",
        "disp_mode",
        "disp_const",
        "use_cell_factor",
        "tau_dispersion",
        "tau_group_dispersion",
        "tau_dropout",
        "p_zero",
        "p_nz",
        "noise_min",
        "noise_max",
        "hidden",
        "bottleneck",
        "dropout",
        "use_residual",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "loss_bio_weight",
        "loss_nz_weight",
        "bio_reg_weight",
        "recon_weight",
        "p_low",
        "p_high",
        "post_threshold",
        "post_threshold_scale",
        "post_threshold_gamma",
        "post_gene_quantile",
        "post_gene_scale",
        "post_gene_gamma",
        "post_gene_ref",
        "keep_positive",
        "hard_zero_bio",
        "oracle_bio",
        "blend_alpha",
        "blend_gamma",
        "p_bio_temp",
        "p_bio_bias",
        "bio_model",
        "bio_model_mix",
        "poisson_scale",
        "post_bio_temp",
        "post_bio_bias",
        "calib_method",
        "calib_blend",
        "calib_platt_l2",
        "calib_platt_max_iter",
        "calib_platt_lr",
        "calib_platt_use_logit",
        "calib_platt_balance",
        "calib_max_samples",
        "sup_bio_weight",
        "sup_bio_l2",
        "sup_bio_lr",
        "sup_bio_epochs",
        "sup_bio_balance",
        "sup_bio_max_samples",
        "sup_bio_method",
        "sup_bio_min_count",
        "sup_bio_var_floor",
        "sup_bio_hidden",
        "lowrank_weight",
        "lowrank_rank",
        "lowrank_iters",
        "zero_scale_weight",
        "zero_scale_mode",
        "zero_scale_bio_weight",
        "zero_scale_max_samples",
        "zero_mix_weight",
        "zero_mix_mode",
        "zero_mix_bio_weight",
        "zero_mix_max_samples",
        "zero_mix_gamma",
        "zero_mix_max_scale",
        "recover_weight",
        "recover_pmax",
        "recover_gamma",
        "constrained_zero_scale",
        "constrained_zero_max_mse_inc",
        "constrained_zero_lambda_max",
        "constrained_zero_iters",
        "ae_bio_weight",
        "ae_bio_cap_weight",
        "ae_bio_temp",
        "ae_bio_quantile",
        "ae_bio_union",
        "gene_boost",
        "gene_boost_gamma",
        "gene_nz_boost",
        "gene_nz_boost_gamma",
        "gene_nz_mix",
        "gene_nz_mix_gamma",
        "gene_rare_threshold",
        "gene_rare_pbio",
        "cell_zero_weight",
        "cell_depth_boost",
        "cell_depth_gamma",
        "cluster_weight",
        "cluster_gamma",
        "cluster_k",
        "cluster_pcs",
        "shrink_alpha",
        "shrink_gamma",
        "zero_shrink",
        "zero_cap_quantile",
        "zero_cap_scale",
        "zero_cap_ref",
        "expr_bio_weight",
        "expr_bio_cap_weight",
        "expr_bio_temp",
        "expr_bio_bias",
        "expr_bio_quantile",
        "expr_bio_scale",
        "expr_bio_union",
        "expr_bio_expected",
        "expr_bio_mix",
        "knn_bio_weight",
        "knn_bio_temp",
        "knn_bio_quantile",
        "knn_k",
        "knn_bio_union",
        "clip_negative",
        "calibrate_p_bio",
        "calibrate_zero_threshold",
        "calibrate_zero_mode",
        "calibrate_p_bio_mode",
        "calibrate_frac_scale_grid",
        "calibrate_zero_thr_grid",
        "calibrate_bin_count",
        "calibrate_lambda_mse",
    ]
    with tuning_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(tuning_cols) + "\n")
        for row in tuning_rows:
            f.write("\t".join(str(row.get(col, "")) for col in tuning_cols) + "\n")

    if best_feasible is not None:
        print("Using best feasible config based on target thresholds.")
    print(
        f"Best config: thr_drop={best_cfg['thr_drop']} p_zero={best_cfg['p_zero']} "
        f"p_nz={best_cfg['p_nz']} noise={best_cfg['noise_min']}..{best_cfg['noise_max']} "
        f"hidden={best_cfg['hidden']} bottleneck={best_cfg['bottleneck']} "
        f"dropout={best_cfg['dropout']} epochs={best_cfg['epochs']} "
        f"batch={best_cfg['batch_size']} lr={best_cfg['lr']} recon_w={best_cfg['recon_weight']} "
        f"bio_reg={best_cfg['bio_reg_weight']} "
        f"post_thr={best_cfg['post_threshold']} ps={best_cfg['post_threshold_scale']}/{best_cfg['post_threshold_gamma']} "
        f"pg={best_cfg['post_gene_quantile']}/{best_cfg['post_gene_scale']}/{best_cfg['post_gene_gamma']}/{best_cfg['post_gene_ref']} "
        f"blend={best_cfg['blend_alpha']}/{best_cfg['blend_gamma']} "
        f"gene_boost={best_cfg['gene_boost']}/{best_cfg['gene_boost_gamma']} "
        f"gene_nz_boost={best_cfg['gene_nz_boost']}/{best_cfg['gene_nz_boost_gamma']} "
        f"gene_nz_mix={best_cfg['gene_nz_mix']}/{best_cfg['gene_nz_mix_gamma']} "
        f"gene_rare={best_cfg['gene_rare_threshold']}/{best_cfg['gene_rare_pbio']} "
        f"cell_zero_weight={best_cfg['cell_zero_weight']} "
        f"cell_depth={best_cfg['cell_depth_boost']}/{best_cfg['cell_depth_gamma']} "
        f"cluster={best_cfg['cluster_weight']}/{best_cfg['cluster_gamma']}/{best_cfg['cluster_k']}pcs{best_cfg['cluster_pcs']} "
        f"shrink={best_cfg['shrink_alpha']}/{best_cfg['shrink_gamma']} "
        f"zshrink={best_cfg.get('zero_shrink', 1.0)} "
        f"zcap={best_cfg.get('zero_cap_quantile', -1.0)}/{best_cfg.get('zero_cap_scale', 1.0)}/"
        f"{best_cfg.get('zero_cap_ref', 'obs')} "
        f"hard_zero_bio={best_cfg['hard_zero_bio']} oracle_bio={best_cfg['oracle_bio']} "
        f"p_bio_temp={best_cfg['p_bio_temp']} p_bio_bias={best_cfg['p_bio_bias']} "
        f"bio_model={best_cfg.get('bio_model', 'splat')}/{best_cfg.get('bio_model_mix', 0.5)}/"
        f"{best_cfg.get('poisson_scale', 1.0)} "
        f"post_bio={best_cfg.get('post_bio_temp', 1.0)}/{best_cfg.get('post_bio_bias', 0.0)} "
        f"ae_bio={best_cfg['ae_bio_weight']}/{best_cfg['ae_bio_temp']}/{best_cfg['ae_bio_quantile']}/"
        f"u{best_cfg['ae_bio_union']}/cap{best_cfg.get('ae_bio_cap_weight', 0.0)} "
        f"expr_bio={best_cfg['expr_bio_weight']}/cap{best_cfg.get('expr_bio_cap_weight', 0.0)}/"
        f"{best_cfg['expr_bio_temp']}/{best_cfg['expr_bio_bias']}/"
        f"{best_cfg['expr_bio_quantile']}/{best_cfg['expr_bio_scale']}/u{best_cfg['expr_bio_union']}/"
        f"{best_cfg.get('expr_bio_expected', 'log')}/{best_cfg.get('expr_bio_mix', 0.5)} "
        f"knn_bio={best_cfg['knn_bio_weight']}/{best_cfg['knn_bio_temp']}/{best_cfg['knn_bio_quantile']}/"
        f"k{best_cfg['knn_k']}/u{best_cfg['knn_bio_union']} "
        f"lowrank={best_cfg.get('lowrank_weight', 0.0)}/{best_cfg.get('lowrank_rank', 0)}/"
        f"{best_cfg.get('lowrank_iters', 0)} "
        f"zero_scale={best_cfg.get('zero_scale_weight', 0.0)}/{best_cfg.get('zero_scale_mode', 'gene')}/"
        f"{best_cfg.get('zero_scale_bio_weight', 0.0)} "
        f"zero_mix={best_cfg.get('zero_mix_weight', 0.0)}/{best_cfg.get('zero_mix_mode', 'global')}/"
        f"{best_cfg.get('zero_mix_bio_weight', 0.0)}/{best_cfg.get('zero_mix_gamma', 1.0)}/"
        f"{best_cfg.get('zero_mix_max_scale', 1.0)} "
        f"recover={best_cfg.get('recover_weight', 0.0)}/{best_cfg.get('recover_pmax', 0.0)}/"
        f"{best_cfg.get('recover_gamma', 1.0)} "
        f"constrained_zero={constrained_zero_scale}/{constrained_zero_max_mse_inc} "
        f"calib_p_bio={calibrate_p_bio} calib_method={best_cfg.get('calib_method', 'none')} "
        f"sup_bio={best_cfg.get('sup_bio_weight', 0.0)}/{best_cfg.get('sup_bio_method', 'logit')} "
        f"calib_mode={calibrate_p_bio_mode} calib_bins={calibrate_bin_count} "
        f"calib_lambda={calibrate_lambda_mse} | obj={best_obj:.6f} "
        f"avg_bz={best_avg_bz:.6f} avg_mse={best_avg_mse:.6f}"
    )
    print(f"Wrote {output_dir / 'mask_impute9_raw_mse_table.tsv'}")
    print(f"Wrote {output_dir / 'mask_impute9_no_biozero_mse_table.tsv'}")
    print(f"Wrote {output_dir / 'mask_impute9_mse_table.tsv'}")
    print(f"Wrote {tuning_path}")


if __name__ == "__main__":
    main()
