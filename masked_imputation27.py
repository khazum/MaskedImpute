#!/usr/bin/env python3
"""
masked_imputation26.py

Fixed-config imputation tool with optional fast-mode optimizations.
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
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
        p for p in sys.path if str(_REPO_ROOT) not in p and "MaskedImpute/rds2py" not in p
    ]
    from rds2py import read_rds
finally:
    sys.path = _SYS_PATH

EPSILON = 1e-6

BIO_PARAMS = {
    "disp_mode": "estimate",
    "use_cell_factor": True,
    "cell_zero_weight": 0.3,
}

MODEL_PARAMS = {
    "hidden": [64],
    "bottleneck": 32,
    "batch_size": 32,
    "weight_decay": 0.0,
}

AE_PARAMS = {
    "epochs": 300,
    "lr": 0.0001,
    "p_zero": 0.01,
    "p_nz": 0.3,
    "noise_max": 0.2,
    "loss_bio_weight": 2.0,
    "loss_nz_weight": 1.0,
    "bio_reg_weight": 1.0,
}

SCALER_PARAMS = {
    "p_low": 2.0,
    "p_high": 99.5,
}

PBIO_CACHE: Dict[Tuple[str, bool, float], np.ndarray] = {}
SCALER_CACHE: Dict[str, "RobustZThenMinMaxToNeg1Pos1"] = {}

LABEL_KEYS = ("cell_type1", "labels", "Group", "label")


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
    def __init__(self, input_dim: int, hidden: Sequence[int], bottleneck: int):
        super().__init__()
        sizes_enc = [input_dim] + list(hidden) + [bottleneck]
        sizes_dec = [bottleneck] + list(reversed(hidden)) + [input_dim]

        enc_layers = []
        for i in range(len(sizes_enc) - 1):
            enc_layers.append(self._block(sizes_enc[i], sizes_enc[i + 1]))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for i in range(len(sizes_dec) - 2):
            dec_layers.append(self._block(sizes_dec[i], sizes_dec[i + 1]))
        dec_layers.append(nn.Linear(sizes_dec[-2], sizes_dec[-1]))
        self.decoder = nn.Sequential(*dec_layers)

    @staticmethod
    def _block(in_dim: int, out_dim: int) -> nn.Module:
        layers = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.SiLU()]
        return nn.Sequential(*layers)

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
    needed = numel * bytes_per * 4  # X, masks, grads, activations (rough)
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

    counts = None
    try:
        counts = sce.assay("counts").T.astype("float32")
        counts = counts[:, keep]
    except Exception:
        counts = None

    return {
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts,
    }


def prepare_dataset(path: Path) -> Dict[str, object] | None:
    dataset = load_dataset(str(path))
    if dataset is None:
        return None
    logcounts = dataset["logcounts"]
    log_true = dataset["log_true"]
    counts_obs = np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    zeros_obs = counts_obs <= 0.0
    counts_max = counts_obs.max(axis=0)

    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, 5.0))
    cz_hi = float(np.percentile(cell_zero_frac, 95.0))
    cz_span = max(cz_hi - cz_lo, EPSILON)
    cell_zero_norm = np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)

    biozero_label = None
    biozero_mask = None
    if log_true is not None:
        true_biozero = log_true <= EPSILON
        biozero_mask = zeros_obs
        biozero_label = np.zeros_like(logcounts, dtype=np.float32)
        biozero_label[biozero_mask] = true_biozero[biozero_mask].astype(np.float32)

    return {
        "dataset": path.stem,
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts_obs,
        "zeros_obs": zeros_obs,
        "counts_max": counts_max,
        "cell_zero_norm": cell_zero_norm,
        "biozero_label": biozero_label,
        "biozero_mask": biozero_mask,
    }


def _extract_labels(sce) -> Tuple[np.ndarray, str]:
    colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
    y = None
    source = None
    if colmd is not None:
        if hasattr(colmd, "get_column_names") and hasattr(colmd, "get_column"):
            colnames = list(map(str, colmd.get_column_names()))
            for key in LABEL_KEYS:
                if key in colnames:
                    y = np.asarray(colmd.get_column(key))
                    source = key
                    break
        elif hasattr(colmd, "columns"):
            colnames = list(map(str, getattr(colmd, "columns", [])))
            for key in LABEL_KEYS:
                if key in colnames:
                    y = np.asarray(colmd[key])
                    source = key
                    break
        elif isinstance(colmd, dict):
            for key in LABEL_KEYS:
                if key in colmd:
                    y = np.asarray(colmd[key])
                    source = key
                    break
    if y is None:
        raise RuntimeError(f"No label column found. Tried: {', '.join(LABEL_KEYS)}")
    _, labels = np.unique(np.asarray(y), return_inverse=True)
    return labels.astype(int), source or "unknown"


def prepare_dataset_clust(path: Path) -> Dict[str, object] | None:
    sce = read_rds(str(path))
    if not hasattr(sce, "assay"):
        return None
    try:
        logcounts = sce.assay("logcounts").T.astype("float32")
    except Exception:
        return None
    try:
        counts = sce.assay("counts").T.astype("float32")
    except Exception:
        counts = None
    labels, source = _extract_labels(sce)
    if logcounts.shape[0] != labels.shape[0]:
        raise RuntimeError("Cells mismatch between logcounts and labels.")
    return {
        "dataset": path.stem,
        "logcounts": logcounts,
        "counts": counts,
        "labels": labels,
        "label_source": source,
    }


def _contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    _, y_true = np.unique(y_true, return_inverse=True)
    _, y_pred = np.unique(y_pred, return_inverse=True)
    m = np.zeros((y_true.max() + 1, y_pred.max() + 1), dtype=np.int64)
    for i in range(y_true.size):
        m[y_true[i], y_pred[i]] += 1
    return m


def _adjusted_rand_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = _contingency_matrix(y_true, y_pred)
    n = m.sum()
    if n < 2:
        return float("nan")

    def comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2

    sum_comb = comb2(m).sum()
    a = m.sum(axis=1)
    b = m.sum(axis=0)
    sum_a = comb2(a).sum()
    sum_b = comb2(b).sum()
    expected = sum_a * sum_b / comb2(np.array([n]))[0]
    max_index = 0.5 * (sum_a + sum_b) - expected
    if max_index == 0:
        return 0.0
    return float((sum_comb - expected) / max_index)


def _pca_embed(X: np.ndarray, n_components: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :n_components] * S[:n_components]


def _kmeans(X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_labels = None
    best_inertia = np.inf
    for _ in range(n_init):
        if n >= k:
            centers = X[rng.choice(n, size=k, replace=False)]
        else:
            centers = X[rng.choice(n, size=k, replace=True)]
        for _ in range(max_iter):
            dist2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = dist2.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                mask = labels == j
                if not np.any(mask):
                    new_centers[j] = X[rng.integers(0, n)]
                else:
                    new_centers[j] = X[mask].mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        inertia = np.sum((X - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels


def compute_ari(log_imputed: np.ndarray, labels: np.ndarray) -> float:
    n, d = log_imputed.shape
    n_components = max(1, min(50, n, d))
    emb = _pca_embed(np.nan_to_num(log_imputed), n_components)
    k = max(2, len(np.unique(labels)))
    cl = _kmeans(emb, k, n_init=10, max_iter=100, seed=42)
    return _adjusted_rand_score(labels, cl)


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
            p_low=float(SCALER_PARAMS["p_low"]), p_high=float(SCALER_PARAMS["p_high"])
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
    ).to(device)
    if compile_enabled and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    opt_kwargs = dict(lr=float(AE_PARAMS["lr"]), weight_decay=float(MODEL_PARAMS["weight_decay"]))
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
    recon_all = np.vstack(recon_list)
    return recon_all.astype(np.float32)


def _mse_from_diff(diff: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        diff = diff[mask]
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff ** 2))


def _biozero_mse(
    p_bio: np.ndarray, biozero_label: Optional[np.ndarray], mask: Optional[np.ndarray]
) -> float:
    if biozero_label is None or mask is None:
        return float("nan")
    diff = p_bio - biozero_label
    return _mse_from_diff(diff, mask)


def tune_biozero_params(
    datasets: Sequence[Dict[str, object]],
    *,
    n_samples: int,
    seed: int,
    progress_every: int,
) -> Dict[str, object]:
    if n_samples <= 0:
        return dict(BIO_PARAMS)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_params = dict(BIO_PARAMS)
    for i in range(n_samples):
        use_cell_factor = bool(rng.integers(0, 2))
        cell_zero_weight = float(rng.random())
        scores: List[float] = []
        for ds in datasets:
            p_bio = splat_cellaware_bio_prob(
                counts=ds["counts"],
                zeros_obs=ds["zeros_obs"],
                disp_mode=BIO_PARAMS["disp_mode"],
                use_cell_factor=use_cell_factor,
            )
            if cell_zero_weight > 0.0:
                cell_w = np.clip(cell_zero_weight * ds["cell_zero_norm"], 0.0, 1.0)
                p_bio = p_bio * (1.0 - cell_w[:, None])
            score = _biozero_mse(p_bio, ds["biozero_label"], ds["biozero_mask"])
            if not np.isnan(score):
                scores.append(float(score))
        if not scores:
            continue
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_params = {
                "disp_mode": BIO_PARAMS["disp_mode"],
                "use_cell_factor": use_cell_factor,
                "cell_zero_weight": cell_zero_weight,
            }
        if progress_every and ((i + 1) % progress_every == 0 or i == 0 or i + 1 == n_samples):
            cfg = {
                "use_cell_factor": use_cell_factor,
                "cell_zero_weight": float(f"{cell_zero_weight:.3f}"),
            }
            print(
                f"[biozero-search] {i+1}/{n_samples} score={score:.6f} "
                f"best={best_score:.6f} cfg={cfg}"
            )
    return best_params


def compute_mse_metrics(
    pred_log: np.ndarray,
    true_log: Optional[np.ndarray],
    counts_obs: np.ndarray,
) -> Dict[str, float]:
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


def _apply_config(cfg: Dict[str, object]) -> None:
    if "hidden" in cfg:
        MODEL_PARAMS["hidden"] = list(cfg["hidden"])
    if "bottleneck" in cfg:
        MODEL_PARAMS["bottleneck"] = int(cfg["bottleneck"])
    for key in ("epochs", "lr", "p_zero", "p_nz", "noise_max", "loss_bio_weight", "loss_nz_weight", "bio_reg_weight"):
        if key in cfg:
            AE_PARAMS[key] = float(cfg[key])


def evaluate_config(
    cfg: Dict[str, object],
    clust_datasets: List[Dict[str, object]],
    mse_datasets: List[Dict[str, object]],
    device: torch.device,
) -> Tuple[float, float]:
    _apply_config(cfg)
    ari_scores: List[float] = []
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    for ds in clust_datasets:
        set_seed(42)
        counts_obs = ds["counts"]
        if counts_obs is None:
            counts_obs = np.clip(np.expm1(ds["logcounts"] * np.log(2.0)), 0.0, None).astype(np.float32)
        zeros_obs = counts_obs <= 0.0
        counts_max = counts_obs.max(axis=0)

        cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
        cz_lo = float(np.percentile(cell_zero_frac, 5.0))
        cz_hi = float(np.percentile(cell_zero_frac, 95.0))
        cz_span = max(cz_hi - cz_lo, EPSILON)
        cell_zero_norm = np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)

        p_bio = splat_cellaware_bio_prob(
            counts=counts_obs,
            zeros_obs=zeros_obs,
            disp_mode=BIO_PARAMS["disp_mode"],
            use_cell_factor=BIO_PARAMS["use_cell_factor"],
        )
        if float(BIO_PARAMS["cell_zero_weight"]) > 0.0:
            cell_w = np.clip(float(BIO_PARAMS["cell_zero_weight"]) * cell_zero_norm, 0.0, 1.0)
            p_bio = p_bio * (1.0 - cell_w[:, None])

        recon = train_autoencoder_reconstruct(
            logcounts=ds["logcounts"],
            counts_max=counts_max,
            p_bio=p_bio,
            device=device,
            fast_mode=bool(use_cuda),
            amp_enabled=bool(use_cuda),
            compile_enabled=bool(use_cuda),
            fast_batch_mult=2,
            num_workers=0,
        )
        ari_scores.append(compute_ari(recon, ds["labels"]))

    mse_scores: List[float] = []
    for ds in mse_datasets:
        set_seed(42)
        p_bio = splat_cellaware_bio_prob(
            counts=ds["counts"],
            zeros_obs=ds["zeros_obs"],
            disp_mode=BIO_PARAMS["disp_mode"],
            use_cell_factor=BIO_PARAMS["use_cell_factor"],
        )
        if float(BIO_PARAMS["cell_zero_weight"]) > 0.0:
            cell_w = np.clip(float(BIO_PARAMS["cell_zero_weight"]) * ds["cell_zero_norm"], 0.0, 1.0)
            p_bio = p_bio * (1.0 - cell_w[:, None])

        recon = train_autoencoder_reconstruct(
            logcounts=ds["logcounts"],
            counts_max=ds["counts_max"],
            p_bio=p_bio,
            device=device,
            fast_mode=bool(use_cuda),
            amp_enabled=bool(use_cuda),
            compile_enabled=bool(use_cuda),
            fast_batch_mult=2,
            num_workers=0,
        )
        log_recon = recon
        log_recon[~ds["zeros_obs"]] = ds["logcounts"][~ds["zeros_obs"]]
        metrics = compute_mse_metrics(log_recon, ds["log_true"], ds["counts"])
        mse_scores.append(float(metrics["mse"]))

    avg_ari = float(np.nanmean(ari_scores)) if ari_scores else float("nan")
    avg_mse = float(np.nanmean(mse_scores)) if mse_scores else float("nan")
    return avg_ari, avg_mse


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


def main() -> None:
    parser = argparse.ArgumentParser(description="MaskedImpute tuning/imputation tool.")
    parser.add_argument("--mode", choices=["impute", "tune"], default="impute")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-positive", default="true")
    parser.add_argument("--save-imputed", default="true")
    parser.add_argument("--bio-reg-weight", type=float, default=1.0)
    parser.add_argument("--biozero-samples", type=int, default=0)
    parser.add_argument("--biozero-progress-every", type=int, default=25)
    parser.add_argument("--fast", default="false", help="Enable fast-mode optimizations.")
    parser.add_argument("--fast-batch-mult", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--clust-path", default="datasets", help="Path to clustering datasets (for tune mode).")
    parser.add_argument("--mse-path", default="synthetic_datasets/rds_splat_output/cells_100",
                        help="Path to MSE datasets (for tune mode).")
    parser.add_argument("--max-evals", type=int, default=50, help="Max tuning evaluations.")
    parser.add_argument("--ari-target", type=float, default=None, help="Target average ARI to beat.")
    parser.add_argument("--mse-tol", type=float, default=0.05, help="Allowed relative MSE increase.")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    keep_positive = str(args.keep_positive).strip().lower() in ("1", "true", "yes", "y")
    save_imputed = str(args.save_imputed).strip().lower() in ("1", "true", "yes", "y")
    fast_mode = str(args.fast).strip().lower() in ("1", "true", "yes", "y")
    amp_enabled = fast_mode and not bool(args.no_amp)
    compile_enabled = fast_mode and not bool(args.no_compile)
    AE_PARAMS["bio_reg_weight"] = float(args.bio_reg_weight)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "tune":
        clust_dir = Path(args.clust_path)
        mse_dir = Path(args.mse_path)
        clust_datasets: List[Dict[str, object]] = []
        for path in sorted(clust_dir.rglob("*.rds")):
            ds = prepare_dataset_clust(path)
            if ds is not None:
                clust_datasets.append(ds)
        if not clust_datasets:
            raise SystemExit("No clustering datasets found.")

        mse_datasets: List[Dict[str, object]] = []
        for path in sorted(mse_dir.rglob("*.rds")):
            ds = prepare_dataset(path)
            if ds is not None:
                mse_datasets.append(ds)
        if not mse_datasets:
            raise SystemExit("No MSE datasets found.")

        base_cfg = dict(AE_PARAMS)
        base_cfg.update({"hidden": MODEL_PARAMS["hidden"], "bottleneck": MODEL_PARAMS["bottleneck"]})
        base_ari, base_mse = evaluate_config(base_cfg, clust_datasets, mse_datasets, device)
        mse_limit = base_mse * (1.0 + float(args.mse_tol))

        # infer ARI target from existing clustering tables if not provided
        ari_target = args.ari_target
        if ari_target is None:
            best = 0.0
            for path in Path("results_clustering_py").rglob("*_clustering_table.tsv"):
                try:
                    rows = path.read_text().strip().splitlines()[1:]
                    if not rows:
                        continue
                    aris = []
                    for row in rows:
                        parts = row.split("\t")
                        aris.append(float(parts[3]))
                    if aris:
                        best = max(best, float(np.mean(aris)))
                except Exception:
                    continue
            ari_target = best

        search_space = {
            "hidden": [[64], [128], [128, 64]],
            "bottleneck": [32, 64],
            "p_zero": [0.0, 0.01, 0.05],
            "p_nz": [0.2, 0.3, 0.4],
            "noise_max": [0.1, 0.2, 0.3],
            "bio_reg_weight": [0.5, 1.0, 2.0],
            "lr": [1e-4, 5e-4, 1e-3],
            "epochs": [200, 300],
        }

        keys = list(search_space.keys())
        rng = np.random.default_rng(int(args.seed))
        best = {"ari": base_ari, "mse": base_mse, "cfg": base_cfg}
        rows: List[Dict[str, object]] = []

        for i in range(int(args.max_evals)):
            cfg = {}
            for k in keys:
                options = search_space[k]
                idx = int(rng.integers(0, len(options)))
                choice = options[idx]
                if isinstance(choice, np.ndarray):
                    choice = choice.tolist()
                elif isinstance(choice, np.generic):
                    choice = choice.item()
                cfg[k] = choice
            cfg["hidden"] = list(cfg["hidden"])
            ari, mse = evaluate_config(cfg, clust_datasets, mse_datasets, device)
            ok = mse <= mse_limit
            rows.append({"iter": i + 1, "ari": ari, "mse": mse, "ok": ok, **cfg})
            if ok and ari > best["ari"]:
                best = {"ari": ari, "mse": mse, "cfg": cfg}
            if args.progress_every and ((i + 1) % int(args.progress_every) == 0 or i == 0):
                print(f"[tune] {i+1}/{args.max_evals} ari={ari:.4f} mse={mse:.4f} ok={ok} best_ari={best['ari']:.4f}")
            if ok and ari_target and ari >= ari_target:
                print(f"[tune] target reached: ari={ari:.4f} >= {ari_target:.4f}")
                break

        header = ["iter", "ari", "mse", "ok"] + keys
        _write_table(output_dir / "tuning_table.tsv", header, rows)
        print(f"Base ARI={base_ari:.4f} Base MSE={base_mse:.4f} MSE limit={mse_limit:.4f}")
        print(f"Best ARI={best['ari']:.4f} Best MSE={best['mse']:.4f} cfg={best['cfg']}")
        return

    datasets: List[Dict[str, object]] = []
    for path in sorted(Path(args.input_path).rglob("*.rds")):
        ds = prepare_dataset(path)
        if ds is None:
            print(f"[WARN] {path.stem}: missing logTrueCounts; skipping metrics.")
            continue
        datasets.append(ds)

    if not datasets:
        raise SystemExit("No datasets processed.")

    if int(args.biozero_samples) > 0:
        tuned = tune_biozero_params(
            datasets,
            n_samples=int(args.biozero_samples),
            seed=int(args.seed),
            progress_every=int(args.biozero_progress_every),
        )
        BIO_PARAMS.update(tuned)

    rows: List[Dict[str, object]] = []
    mse_list: List[float] = []
    bz_list: List[float] = []
    start_time = time.perf_counter()

    for ds in datasets:
        set_seed(int(args.seed))
        cache_key = (ds["dataset"], bool(BIO_PARAMS["use_cell_factor"]), float(BIO_PARAMS["cell_zero_weight"]))
        p_bio = PBIO_CACHE.get(cache_key)
        if p_bio is None:
            p_bio = splat_cellaware_bio_prob(
                counts=ds["counts"],
                zeros_obs=ds["zeros_obs"],
                disp_mode=BIO_PARAMS["disp_mode"],
                use_cell_factor=BIO_PARAMS["use_cell_factor"],
            )
            if float(BIO_PARAMS["cell_zero_weight"]) > 0.0:
                cell_w = np.clip(
                    float(BIO_PARAMS["cell_zero_weight"]) * ds["cell_zero_norm"], 0.0, 1.0
                )
                p_bio = p_bio * (1.0 - cell_w[:, None])
            PBIO_CACHE[cache_key] = p_bio

        recon = train_autoencoder_reconstruct(
            logcounts=ds["logcounts"],
            counts_max=ds["counts_max"],
            p_bio=p_bio,
            device=device,
            fast_mode=fast_mode,
            amp_enabled=amp_enabled,
            compile_enabled=compile_enabled,
            fast_batch_mult=int(args.fast_batch_mult),
            num_workers=int(args.num_workers),
        )
        log_recon = recon

        if keep_positive:
            log_recon[~ds["zeros_obs"]] = ds["logcounts"][~ds["zeros_obs"]]

        metrics = compute_mse_metrics(log_recon, ds["log_true"], ds["counts"])
        row = {
            "dataset": ds["dataset"],
            "mse": metrics["mse"],
            "mse_biozero": metrics["mse_biozero"],
            "mse_dropout": metrics["mse_dropout"],
            "mse_non_zero": metrics["mse_non_zero"],
        }
        rows.append(row)
        if not np.isnan(metrics["mse"]):
            mse_list.append(float(metrics["mse"]))
        if not np.isnan(metrics["mse_biozero"]):
            bz_list.append(float(metrics["mse_biozero"]))

        if save_imputed:
            np.savez_compressed(
                output_dir / f"{ds['dataset']}_imputed.npz",
                logcounts=ds["logcounts"],
                log_imputed=log_recon,
                p_bio=p_bio,
            )

    avg_mse = float(np.nanmean(mse_list)) if mse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = float(avg_bz) + 0.5 * float(avg_mse)
    runtime_sec = float(time.perf_counter() - start_time)

    _write_table(
        output_dir / "masked_imputation26_metrics.tsv",
        ["dataset", "mse", "mse_biozero", "mse_dropout", "mse_non_zero"],
        rows,
    )
    _write_table(
        output_dir / "masked_imputation26_summary.tsv",
        ["avg_mse", "avg_bz_mse", "score", "runtime_sec", "fast_mode"],
        [
            {
                "avg_mse": avg_mse,
                "avg_bz_mse": avg_bz,
                "score": score,
                "runtime_sec": runtime_sec,
                "fast_mode": fast_mode,
            }
        ],
    )

    print("\n=== masked_imputation26 ===")
    print("Biozero params:", BIO_PARAMS)
    print("AE params:", AE_PARAMS)
    print("avg_mse:", avg_mse, "avg_bz_mse:", avg_bz, "score:", score)
    print("runtime_sec:", runtime_sec, "fast_mode:", fast_mode)
    print("Metrics written to masked_imputation26_metrics.tsv")
    print("Summary written to masked_imputation26_summary.tsv")


if __name__ == "__main__":
    main()
