#!/usr/bin/env python3
"""
mask_impute5.py
--------------

Masked AE imputation with expanded autotuning and probability-aware zero handling
to drive down biozero MSE while keeping overall MSE low. Uses SPLAT cell-aware
posterior probabilities, weighted denoising loss, and shrinkage on observed
zeros with tunable thresholds, architecture, optimizer, and scaling.
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
    with torch.no_grad():
        for i in range(0, Xtr.size(0), batch_size):
            xb = Xtr[i : i + batch_size].to(device)
            recon = model(xb)
            recon_np = recon.cpu().numpy()
            recon_orig = scaler.inverse_transform(recon_np)
            recon_list.append(recon_orig)
    recon_all = np.vstack(recon_list)
    return recon_all.astype(np.float32)


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
    if int(config["epochs"]) < 1 or int(config["batch_size"]) < 1:
        return False
    if float(config["lr"]) <= 0.0:
        return False
    if float(config["loss_bio_weight"]) < 0.0 or float(config["loss_nz_weight"]) < 0.0:
        return False
    if float(config["bio_reg_weight"]) < 0.0:
        return False
    shrink_alpha = float(config["shrink_alpha"])
    if shrink_alpha < 0.0 or shrink_alpha > 1.0:
        return False
    if float(config["shrink_gamma"]) <= 0.0:
        return False
    blend_alpha = float(config["blend_alpha"])
    if blend_alpha < 0.0 or blend_alpha > 1.0:
        return False
    if float(config["blend_gamma"]) <= 0.0:
        return False
    if float(config["p_bio_temp"]) <= 0.0:
        return False
    recon_weight = float(config["recon_weight"])
    if recon_weight < 0.0 or recon_weight > 1.0:
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
    if float(config["ae_bio_temp"]) <= 0.0:
        return False
    ae_bio_quantile = float(config["ae_bio_quantile"])
    if ae_bio_quantile < 0.0 or ae_bio_quantile > 1.0:
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
        counts_max = ds["counts_max"]
        gene_mean = ds["gene_mean"]
        gene_mean_norm = ds["gene_mean_norm"]
        gene_nz_frac = ds["gene_nz_frac"]
        cell_zero_norm = ds["cell_zero_norm"]
        if bool(config["oracle_bio"]):
            p_bio_use = (log_true <= EPSILON).astype(np.float32)
            p_bio_use[~zeros_obs] = 0.0
        else:
            p_bio = _get_p_bio_for_dataset(ds, config)
            p_bio_use = p_bio
            if float(config["gene_boost"]) > 0.0:
                boost = float(config["gene_boost"]) * (1.0 - gene_mean_norm) ** float(config["gene_boost_gamma"])
                boost = np.clip(boost, 0.0, 1.0)
                boost_row = boost[None, :]
                p_bio_use = 1.0 - (1.0 - p_bio) * (1.0 - boost_row)
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
                p_bio_use = 1.0 / (1.0 + np.exp(-logit))
                p_bio_use = np.clip(p_bio_use, 0.0, 1.0)
            p_bio_use[~zeros_obs] = 0.0
        thr_bio = 1.0 - float(config["thr_drop"])

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        log_imputed_raw = train_autoencoder_reconstruct(
            logcounts=logcounts,
            counts_max=counts_max,
            p_bio=p_bio_use,
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
        )

        p_bio_post = p_bio_use
        if float(config["ae_bio_weight"]) > 0.0:
            gene_q = _get_gene_quantile(ds, float(config["ae_bio_quantile"]))
            thr = gene_q[None, :]
            temp = float(config["ae_bio_temp"])
            logits = (thr - log_imputed_raw) / temp
            logits = np.clip(logits, -50.0, 50.0)
            p_bio_ae = 1.0 / (1.0 + np.exp(-logits))
            p_bio_ae = np.clip(p_bio_ae, 0.0, 1.0)
            p_bio_ae[~zeros_obs] = 0.0
            w = float(config["ae_bio_weight"])
            if bool(config["ae_bio_union"]):
                p_bio_post = 1.0 - (1.0 - p_bio_use) * (1.0 - w * p_bio_ae)
            else:
                p_bio_post = (1.0 - w) * p_bio_use + w * p_bio_ae
            p_bio_post = np.clip(p_bio_post, 0.0, 1.0)
        p_bio_post[~zeros_obs] = 0.0

        pred_bio_mask = (p_bio_post >= thr_bio) & zeros_obs
        if np.any(pred_bio_mask & ~zeros_obs):
            raise RuntimeError(f"{ds_name}: pred_bio_mask contains non-zero entries outside observed zeros.")

        log_imputed_keep = log_imputed_raw.copy()
        if bool(config["keep_positive"]):
            pos_mask = logcounts > 0.0
            log_imputed_keep[pos_mask] = logcounts[pos_mask]

        log_imputed_final = log_imputed_keep.copy()
        if float(config["blend_alpha"]) > 0.0:
            blend = float(config["blend_alpha"]) * (1.0 - p_bio_post) ** float(config["blend_gamma"])
            blend = np.clip(blend, 0.0, 1.0)
            gene_mean_row = gene_mean[None, :]
            log_imputed_final = np.where(
                zeros_obs,
                (1.0 - blend) * log_imputed_final + blend * gene_mean_row,
                log_imputed_final,
            )
        if float(config["shrink_alpha"]) > 0.0:
            shrink = 1.0 - float(config["shrink_alpha"]) * (p_bio_post ** float(config["shrink_gamma"]))
            shrink = np.clip(shrink, 0.0, 1.0)
            log_imputed_final[zeros_obs] = log_imputed_final[zeros_obs] * shrink[zeros_obs]
        if bool(config["hard_zero_bio"]):
            log_imputed_final[pred_bio_mask] = 0.0
    post_thr = float(config["post_threshold"])
    if post_thr >= 0:
        post_scale = float(config["post_threshold_scale"])
        if post_scale > 0.0:
            post_gamma = float(config["post_threshold_gamma"])
            thr_map = post_thr * (1.0 + post_scale * (p_bio_post ** post_gamma))
        else:
            thr_map = post_thr
        low_mask = zeros_obs & (log_imputed_final < thr_map)
        log_imputed_final[low_mask] = 0.0
    post_gene_q = float(config["post_gene_quantile"])
    if post_gene_q >= 0.0:
        gene_q = _get_gene_quantile(ds, post_gene_q)
        gene_thr = gene_q[None, :] * float(config["post_gene_scale"])
        gene_gamma = float(config["post_gene_gamma"])
        gene_thr = gene_thr * (p_bio_post ** gene_gamma)
        low_mask = zeros_obs & (log_imputed_final < gene_thr)
        log_imputed_final[low_mask] = 0.0
        if bool(config["clip_negative"]):
            log_imputed_final = np.maximum(log_imputed_final, 0.0)

        row_base = {
            "dataset": ds_name,
            "thr_drop": float(config["thr_drop"]),
            "thr_bio": thr_bio,
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
            "blend_alpha": float(config["blend_alpha"]),
            "blend_gamma": float(config["blend_gamma"]),
            "p_bio_temp": float(config["p_bio_temp"]),
            "p_bio_bias": float(config["p_bio_bias"]),
            "ae_bio_weight": float(config["ae_bio_weight"]),
            "ae_bio_temp": float(config["ae_bio_temp"]),
            "ae_bio_quantile": float(config["ae_bio_quantile"]),
            "ae_bio_union": bool(config["ae_bio_union"]),
            "gene_boost": float(config["gene_boost"]),
            "gene_boost_gamma": float(config["gene_boost_gamma"]),
            "gene_nz_boost": float(config["gene_nz_boost"]),
            "gene_nz_boost_gamma": float(config["gene_nz_boost_gamma"]),
            "gene_nz_mix": float(config["gene_nz_mix"]),
            "gene_nz_mix_gamma": float(config["gene_nz_mix_gamma"]),
            "cell_zero_weight": float(config["cell_zero_weight"]),
            "cluster_weight": float(config["cluster_weight"]),
            "cluster_gamma": float(config["cluster_gamma"]),
            "cluster_k": int(config["cluster_k"]),
            "cluster_pcs": int(config["cluster_pcs"]),
            "shrink_alpha": float(config["shrink_alpha"]),
            "shrink_gamma": float(config["shrink_gamma"]),
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
    parser.add_argument("output_dir", help="Output directory for mask_impute5_*_mse_table.tsv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-mse", type=float, default=0.5, help="Weight for overall MSE in objective.")
    parser.add_argument("--max-evals", type=int, default=60, help="Max configs to evaluate.")
    parser.add_argument("--target-mse", type=float, default=None, help="Stop early if avg MSE is below this value.")
    parser.add_argument("--target-biozero", type=float, default=None, help="Stop early if avg biozero MSE is below this value.")

    parser.add_argument("--thr-drop-grid", type=str, default="0.7,0.8,0.82,0.9")
    parser.add_argument("--disp-mode-grid", type=str, default="estimate,fixed")
    parser.add_argument("--disp-const-grid", type=str, default="0.05,0.1,0.2")
    parser.add_argument("--use-cell-factor-grid", type=str, default="true,false")
    parser.add_argument("--tau-dispersion-grid", type=str, default="10,20,40")
    parser.add_argument("--tau-group-dispersion-grid", type=str, default="20,50,80")
    parser.add_argument("--tau-dropout-grid", type=str, default="20,50,80")

    parser.add_argument("--p-zero-grid", type=str, default="0,0.05,0.1,0.2")
    parser.add_argument("--p-nz-grid", type=str, default="0.1,0.2,0.3,0.4")
    parser.add_argument("--noise-min-grid", type=str, default="0,0.05")
    parser.add_argument("--noise-max-grid", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--hidden-grid", type=str, default="64|128,64|256,128")
    parser.add_argument("--bottleneck-grid", type=str, default="16,32,64")
    parser.add_argument("--dropout-grid", type=str, default="0.0,0.05,0.1,0.2")
    parser.add_argument("--use-residual-grid", type=str, default="true,false")

    parser.add_argument("--epochs-grid", type=str, default="60,100,140")
    parser.add_argument("--batch-size-grid", type=str, default="32,64,128")
    parser.add_argument("--lr-grid", type=str, default="5e-4,1e-3,2e-3")
    parser.add_argument("--weight-decay-grid", type=str, default="0,1e-4,1e-3")
    parser.add_argument("--loss-bio-weight-grid", type=str, default="1,2,4")
    parser.add_argument("--loss-nz-weight-grid", type=str, default="1,2")
    parser.add_argument("--bio-reg-weight-grid", type=str, default="0,0.1,0.25,0.5,1.0")
    parser.add_argument("--recon-weight-grid", type=str, default="0,0.1,0.2,0.4")
    parser.add_argument("--p-low-grid", type=str, default="0.5,1,2")
    parser.add_argument("--p-high-grid", type=str, default="98,99,99.5")
    parser.add_argument("--post-threshold-grid", type=str, default="-1,0,0.25,0.5,0.75")
    parser.add_argument("--post-threshold-scale-grid", type=str, default="0,0.5,1.0,2.0")
    parser.add_argument("--post-threshold-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--post-gene-quantile-grid", type=str, default="-1,0.1,0.2,0.3")
    parser.add_argument("--post-gene-scale-grid", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--post-gene-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--keep-positive-grid", type=str, default="true,false")
    parser.add_argument("--hard-zero-bio-grid", type=str, default="true,false")
    parser.add_argument("--oracle-bio-grid", type=str, default="false,true")
    parser.add_argument("--blend-alpha-grid", type=str, default="0,0.3,0.6")
    parser.add_argument("--blend-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--p-bio-temp-grid", type=str, default="0.8,1,1.25,1.5")
    parser.add_argument("--p-bio-bias-grid", type=str, default="-0.5,0,0.5,1.0")
    parser.add_argument("--ae-bio-weight-grid", type=str, default="0,0.3,0.6,0.9")
    parser.add_argument("--ae-bio-temp-grid", type=str, default="0.1,0.3,0.6,1.0")
    parser.add_argument("--ae-bio-quantile-grid", type=str, default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--ae-bio-union-grid", type=str, default="false,true")
    parser.add_argument("--gene-boost-grid", type=str, default="0,0.3,0.6")
    parser.add_argument("--gene-boost-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--gene-nz-boost-grid", type=str, default="0,0.2,0.4")
    parser.add_argument("--gene-nz-boost-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--gene-nz-mix-grid", type=str, default="0,0.3,0.6")
    parser.add_argument("--gene-nz-mix-gamma-grid", type=str, default="1,2,4")
    parser.add_argument("--cell-zero-weight-grid", type=str, default="0,0.3,0.6")
    parser.add_argument("--cluster-weight-grid", type=str, default="0,0.3,0.6")
    parser.add_argument("--cluster-gamma-grid", type=str, default="1,2")
    parser.add_argument("--cluster-k-grid", type=str, default="2,3,4")
    parser.add_argument("--cluster-pcs-grid", type=str, default="10,20")
    parser.add_argument("--shrink-alpha-grid", type=str, default="0,0.5,1")
    parser.add_argument("--shrink-gamma-grid", type=str, default="1,2,4")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

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
                "p_bio_cache": {},
                "cluster_cache": {},
                "gene_q_cache": {},
            }
        )

    if not datasets:
        raise SystemExit("No datasets processed.")

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
    ae_bio_weight_list = _parse_float_list(args.ae_bio_weight_grid, [0.0])
    ae_bio_temp_list = _parse_float_list(args.ae_bio_temp_grid, [0.5])
    ae_bio_quantile_list = _parse_float_list(args.ae_bio_quantile_grid, [0.2])
    ae_bio_union_list = _parse_bool_list(args.ae_bio_union_grid, [False])
    gene_boost_list = _parse_float_list(args.gene_boost_grid, [0.0])
    gene_boost_gamma_list = _parse_float_list(args.gene_boost_gamma_grid, [1.0])
    gene_nz_boost_list = _parse_float_list(args.gene_nz_boost_grid, [0.0])
    gene_nz_boost_gamma_list = _parse_float_list(args.gene_nz_boost_gamma_grid, [1.0])
    gene_nz_mix_list = _parse_float_list(args.gene_nz_mix_grid, [0.0])
    gene_nz_mix_gamma_list = _parse_float_list(args.gene_nz_mix_gamma_grid, [1.0])
    cell_zero_weight_list = _parse_float_list(args.cell_zero_weight_grid, [0.0])
    cluster_weight_list = _parse_float_list(args.cluster_weight_grid, [0.0])
    cluster_gamma_list = _parse_float_list(args.cluster_gamma_grid, [1.0])
    cluster_k_list = _parse_int_list(args.cluster_k_grid, [2])
    cluster_pcs_list = _parse_int_list(args.cluster_pcs_grid, [10])
    shrink_alpha_list = _parse_float_list(args.shrink_alpha_grid, [0.0])
    shrink_gamma_list = _parse_float_list(args.shrink_gamma_grid, [1.0])
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
        "keep_positive": keep_positive_list,
        "hard_zero_bio": hard_zero_bio_list,
        "oracle_bio": oracle_bio_list,
        "blend_alpha": blend_alpha_list,
        "blend_gamma": blend_gamma_list,
        "p_bio_temp": p_bio_temp_list,
        "p_bio_bias": p_bio_bias_list,
        "ae_bio_weight": ae_bio_weight_list,
        "ae_bio_temp": ae_bio_temp_list,
        "ae_bio_quantile": ae_bio_quantile_list,
        "ae_bio_union": ae_bio_union_list,
        "gene_boost": gene_boost_list,
        "gene_boost_gamma": gene_boost_gamma_list,
        "gene_nz_boost": gene_nz_boost_list,
        "gene_nz_boost_gamma": gene_nz_boost_gamma_list,
        "gene_nz_mix": gene_nz_mix_list,
        "gene_nz_mix_gamma": gene_nz_mix_gamma_list,
        "cell_zero_weight": cell_zero_weight_list,
        "cluster_weight": cluster_weight_list,
        "cluster_gamma": cluster_gamma_list,
        "cluster_k": cluster_k_list,
        "cluster_pcs": cluster_pcs_list,
        "shrink_alpha": shrink_alpha_list,
        "shrink_gamma": shrink_gamma_list,
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
            datasets,
            cfg,
            device=device,
            seed=int(args.seed),
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
                "keep_positive": cfg["keep_positive"],
                "hard_zero_bio": cfg["hard_zero_bio"],
                "oracle_bio": cfg["oracle_bio"],
                "blend_alpha": cfg["blend_alpha"],
                "blend_gamma": cfg["blend_gamma"],
                "p_bio_temp": cfg["p_bio_temp"],
                "p_bio_bias": cfg["p_bio_bias"],
                "ae_bio_weight": cfg["ae_bio_weight"],
                "ae_bio_temp": cfg["ae_bio_temp"],
                "ae_bio_quantile": cfg["ae_bio_quantile"],
                "ae_bio_union": cfg["ae_bio_union"],
                "gene_boost": cfg["gene_boost"],
                "gene_boost_gamma": cfg["gene_boost_gamma"],
                "gene_nz_boost": cfg["gene_nz_boost"],
                "gene_nz_boost_gamma": cfg["gene_nz_boost_gamma"],
                "gene_nz_mix": cfg["gene_nz_mix"],
                "gene_nz_mix_gamma": cfg["gene_nz_mix_gamma"],
                "cell_zero_weight": cfg["cell_zero_weight"],
                "cluster_weight": cfg["cluster_weight"],
                "cluster_gamma": cfg["cluster_gamma"],
                "cluster_k": cfg["cluster_k"],
                "cluster_pcs": cfg["cluster_pcs"],
                "shrink_alpha": cfg["shrink_alpha"],
                "shrink_gamma": cfg["shrink_gamma"],
                "clip_negative": cfg["clip_negative"],
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
            f"blend={cfg['blend_alpha']}/{cfg['blend_gamma']} "
            f"gene_boost={cfg['gene_boost']}/{cfg['gene_boost_gamma']} "
            f"gene_nz_boost={cfg['gene_nz_boost']}/{cfg['gene_nz_boost_gamma']} "
            f"gene_nz_mix={cfg['gene_nz_mix']}/{cfg['gene_nz_mix_gamma']} "
            f"cell_zero_weight={cfg['cell_zero_weight']} "
            f"cluster={cfg['cluster_weight']}/{cfg['cluster_gamma']}/{cfg['cluster_k']}pcs{cfg['cluster_pcs']} "
            f"post_thr={cfg['post_threshold']} ps={cfg['post_threshold_scale']}/{cfg['post_threshold_gamma']} "
            f"pg={cfg['post_gene_quantile']}/{cfg['post_gene_scale']}/{cfg['post_gene_gamma']} "
            f"hard_zero_bio={cfg['hard_zero_bio']} oracle_bio={cfg['oracle_bio']} "
            f"p_bio_temp={cfg['p_bio_temp']} p_bio_bias={cfg['p_bio_bias']} "
            f"ae_bio={cfg['ae_bio_weight']}/{cfg['ae_bio_temp']}/{cfg['ae_bio_quantile']}/u{cfg['ae_bio_union']} "
            f"keep_pos={cfg['keep_positive']}"
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
            "keep_positive",
            "hard_zero_bio",
            "oracle_bio",
            "blend_alpha",
            "blend_gamma",
            "p_bio_temp",
            "p_bio_bias",
            "ae_bio_weight",
            "ae_bio_temp",
            "ae_bio_quantile",
            "ae_bio_union",
            "gene_boost",
            "gene_boost_gamma",
            "gene_nz_boost",
            "gene_nz_boost_gamma",
            "gene_nz_mix",
            "gene_nz_mix_gamma",
            "cell_zero_weight",
            "cluster_weight",
            "cluster_gamma",
            "cluster_k",
            "cluster_pcs",
            "shrink_alpha",
            "shrink_gamma",
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

    _write_table(output_dir / "mask_impute5_raw_mse_table.tsv", best_raw)
    _write_table(output_dir / "mask_impute5_no_biozero_mse_table.tsv", best_no_bio)
    _write_table(output_dir / "mask_impute5_mse_table.tsv", best_final)

    tuning_path = output_dir / "mask_impute5_tuning.tsv"
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
        "keep_positive",
        "hard_zero_bio",
        "oracle_bio",
        "blend_alpha",
        "blend_gamma",
        "p_bio_temp",
        "p_bio_bias",
        "ae_bio_weight",
        "ae_bio_temp",
        "ae_bio_quantile",
        "ae_bio_union",
        "gene_boost",
        "gene_boost_gamma",
        "gene_nz_boost",
        "gene_nz_boost_gamma",
        "gene_nz_mix",
        "gene_nz_mix_gamma",
        "cell_zero_weight",
        "cluster_weight",
        "cluster_gamma",
        "cluster_k",
        "cluster_pcs",
        "shrink_alpha",
        "shrink_gamma",
        "clip_negative",
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
        f"pg={best_cfg['post_gene_quantile']}/{best_cfg['post_gene_scale']}/{best_cfg['post_gene_gamma']} "
        f"blend={best_cfg['blend_alpha']}/{best_cfg['blend_gamma']} "
        f"gene_boost={best_cfg['gene_boost']}/{best_cfg['gene_boost_gamma']} "
        f"gene_nz_boost={best_cfg['gene_nz_boost']}/{best_cfg['gene_nz_boost_gamma']} "
        f"gene_nz_mix={best_cfg['gene_nz_mix']}/{best_cfg['gene_nz_mix_gamma']} "
        f"cell_zero_weight={best_cfg['cell_zero_weight']} "
        f"cluster={best_cfg['cluster_weight']}/{best_cfg['cluster_gamma']}/{best_cfg['cluster_k']}pcs{best_cfg['cluster_pcs']} "
        f"shrink={best_cfg['shrink_alpha']}/{best_cfg['shrink_gamma']} "
        f"hard_zero_bio={best_cfg['hard_zero_bio']} oracle_bio={best_cfg['oracle_bio']} "
        f"p_bio_temp={best_cfg['p_bio_temp']} p_bio_bias={best_cfg['p_bio_bias']} "
        f"ae_bio={best_cfg['ae_bio_weight']}/{best_cfg['ae_bio_temp']}/{best_cfg['ae_bio_quantile']}/u{best_cfg['ae_bio_union']} "
        f"| obj={best_obj:.6f} "
        f"avg_bz={best_avg_bz:.6f} avg_mse={best_avg_mse:.6f}"
    )
    print(f"Wrote {output_dir / 'mask_impute5_raw_mse_table.tsv'}")
    print(f"Wrote {output_dir / 'mask_impute5_no_biozero_mse_table.tsv'}")
    print(f"Wrote {output_dir / 'mask_impute5_mse_table.tsv'}")
    print(f"Wrote {tuning_path}")


if __name__ == "__main__":
    main()
