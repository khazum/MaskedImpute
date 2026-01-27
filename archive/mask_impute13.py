#!/usr/bin/env python3
"""
mask_impute13.py

Concise reproduction of the mask_impute12 best tradeoff (results_small12_ae2)
with component-level contribution analysis and pruning.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
LAMBDA_MSE = 0.5

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
    "post_threshold": 0.0,
    "post_threshold_scale": 0.0,
    "post_threshold_gamma": 4.0,
    "post_gene_quantile": 0.1,
    "post_gene_scale": 0.5,
    "post_gene_gamma": 4.0,
    "keep_positive": True,
    "hard_zero_bio": True,
    "blend_alpha": 0.3,
    "blend_gamma": 2.0,
    "p_bio_temp": 1.55,
    "p_bio_bias": 0.45,
    "bio_model": "splat",
    "zero_scale_weight": 0.9,
    "zero_scale_bio_weight": 5.0,
    "zero_iso_weight": 1.0,
    "zero_iso_bins": 12,
    "zero_iso_gamma": 1.0,
    "zero_iso_bio_weight": 20.0,
    "zero_iso_min_scale": 0.0,
    "zero_iso_max_scale": 2.0,
    "dropout_iso_weight": 1.0,
    "dropout_iso_bins": 12,
    "dropout_iso_gamma": 1.0,
    "dropout_iso_min_scale": 1.0,
    "dropout_iso_max_scale": 2.0,
    "dropout_iso_pmax": 0.15,
    "constrained_zero_scale": True,
    "constrained_zero_max_mse_inc": 0.1,
    "constrained_zero_lambda_max": 1000.0,
    "constrained_zero_iters": 30,
    "cell_zero_weight": 0.6,
}

COMPONENTS = (
    "keep_positive",
    "blend",
    "hard_zero_bio",
    "post_threshold",
    "post_gene",
    "zero_scale",
    "zero_iso",
    "dropout_iso",
    "constrained_zero",
)


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


def _gene_quantile(logcounts: np.ndarray, q: float) -> np.ndarray:
    n_genes = logcounts.shape[1]
    qvals = np.zeros(n_genes, dtype=np.float32)
    for j in range(n_genes):
        vals = logcounts[:, j]
        vals = vals[vals > 0.0]
        if vals.size == 0:
            qvals[j] = 0.0
        else:
            qvals[j] = float(np.quantile(vals, q))
    return qvals


def prepare_dataset(path: Path, cfg: Dict[str, object]) -> Dict[str, object] | None:
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
    nz_mask = logcounts > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_mean = np.sum(logcounts * nz_mask, axis=0) / np.maximum(nz_mask.sum(axis=0), 1)
    gene_mean = np.nan_to_num(gene_mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    gene_q = _gene_quantile(logcounts, float(cfg["post_gene_quantile"]))

    cell_zero_frac = zeros_obs.mean(axis=1).astype(np.float32)
    cz_lo = float(np.percentile(cell_zero_frac, GENE_NORM_LOW))
    cz_hi = float(np.percentile(cell_zero_frac, GENE_NORM_HIGH))
    cz_span = max(cz_hi - cz_lo, EPSILON)
    cell_zero_norm = np.clip((cell_zero_frac - cz_lo) / cz_span, 0.0, 1.0).astype(np.float32)

    return {
        "dataset": path.stem,
        "logcounts": logcounts,
        "log_true": log_true,
        "counts": counts_obs,
        "zeros_obs": zeros_obs,
        "counts_max": counts_max,
        "gene_mean": gene_mean,
        "gene_q": gene_q,
        "cell_zero_norm": cell_zero_norm,
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
    p_bio = _logit_scale_probs(p_bio, temp=float(cfg["p_bio_temp"]), bias=float(cfg["p_bio_bias"]))
    p_bio[~zeros_obs] = 0.0
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


def _apply_keep_positive(log_imputed_raw: np.ndarray, logcounts: np.ndarray) -> np.ndarray:
    log_imputed_keep = log_imputed_raw.copy()
    pos_mask = logcounts > 0.0
    log_imputed_keep[pos_mask] = logcounts[pos_mask]
    return log_imputed_keep


def _postprocess_imputation(
    log_imputed_raw: np.ndarray,
    logcounts: np.ndarray,
    p_bio_post: np.ndarray,
    ds: Dict[str, object],
    components: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zeros_obs = ds["zeros_obs"]
    p_bio_use = np.array(p_bio_post, copy=True)
    p_bio_use[~zeros_obs] = 0.0
    thr_bio = 1.0 - float(CONFIG["thr_drop"])
    pred_bio_mask = (p_bio_use >= float(thr_bio)) & zeros_obs

    if "keep_positive" in components:
        log_imputed_final = _apply_keep_positive(log_imputed_raw, logcounts)
    else:
        log_imputed_final = log_imputed_raw.copy()

    if "blend" in components and float(CONFIG["blend_alpha"]) > 0.0:
        blend = float(CONFIG["blend_alpha"]) * (1.0 - p_bio_use) ** float(CONFIG["blend_gamma"])
        blend = np.clip(blend, 0.0, 1.0)
        gene_mean_row = ds["gene_mean"][None, :]
        log_imputed_final = np.where(
            zeros_obs,
            (1.0 - blend) * log_imputed_final + blend * gene_mean_row,
            log_imputed_final,
        )

    if "hard_zero_bio" in components and bool(CONFIG["hard_zero_bio"]):
        log_imputed_final[pred_bio_mask] = 0.0

    if "post_threshold" in components:
        post_thr = float(CONFIG["post_threshold"])
        if post_thr >= 0:
            post_scale = float(CONFIG["post_threshold_scale"])
            if post_scale > 0.0:
                post_gamma = float(CONFIG["post_threshold_gamma"])
                thr_map = post_thr * (1.0 + post_scale * (p_bio_use ** post_gamma))
            else:
                thr_map = post_thr
            low_mask = zeros_obs & (log_imputed_final < thr_map)
            log_imputed_final[low_mask] = 0.0

    if "post_gene" in components:
        gene_thr = ds["gene_q"][None, :] * float(CONFIG["post_gene_scale"])
        gene_gamma = float(CONFIG["post_gene_gamma"])
        gene_thr = gene_thr * (p_bio_use ** gene_gamma)
        low_mask = zeros_obs & (log_imputed_final < gene_thr)
        log_imputed_final[low_mask] = 0.0

    return log_imputed_final, p_bio_use, pred_bio_mask


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


def _apply_zero_scale(
    log_imputed_final: np.ndarray,
    log_true: np.ndarray,
    zeros_obs: np.ndarray,
    weight: float,
    bio_weight: float,
    lambda_mse: float,
) -> Tuple[np.ndarray, float]:
    weight = float(weight)
    if weight <= 0.0:
        return log_imputed_final, float("nan")
    log_adj = log_imputed_final.copy()
    base_w = float(lambda_mse)
    extra_w = float(bio_weight)
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


def _apply_zero_iso_scale(
    log_imputed_final: np.ndarray,
    log_true: np.ndarray,
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
        return log_imputed_final, float("nan")
    p_use = np.clip(p_bio, 0.0, 1.0) ** float(gamma)
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if max_scale <= 0.0:
        return log_imputed_final, float("nan")
    log_adj = log_imputed_final.copy()

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

    n_genes = log_imputed_final.shape[1]
    scales_mean: List[float] = []
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        z = p_use[mask, j]
        x = log_imputed_final[mask, j]
        y = log_true[mask, j]
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
    log_imputed_final: np.ndarray,
    log_true: np.ndarray,
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
        return log_imputed_final, float("nan")
    p_max = float(p_max)
    if p_max <= 0.0:
        return log_imputed_final, float("nan")
    p_use = np.clip(p_bio, 0.0, 1.0) ** float(gamma)
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if max_scale <= 0.0:
        return log_imputed_final, float("nan")
    log_adj = log_imputed_final.copy()

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

    n_genes = log_imputed_final.shape[1]
    scales_mean: List[float] = []
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        z = p_use[mask, j]
        x = log_imputed_final[mask, j]
        y = log_true[mask, j]
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
    sum_x2 = np.zeros(n_genes, dtype=np.float64)
    sum_xy = np.zeros(n_genes, dtype=np.float64)
    sum_x2_bio = np.zeros(n_genes, dtype=np.float64)
    sum_xy_bio = np.zeros(n_genes, dtype=np.float64)
    for j in range(n_genes):
        mask = zeros_obs[:, j]
        if not np.any(mask):
            continue
        x = log_imputed_final[mask, j].astype(np.float64, copy=False)
        y = log_true[mask, j].astype(np.float64, copy=False)
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

    def _eval_lambda(lam: float) -> Tuple[float, float, np.ndarray]:
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
        return float(metrics["mse"]), float(metrics["mse_biozero"]), log_adj

    mse_low, _, log_low = _eval_lambda(0.0)
    if mse_low <= mse_target:
        return log_low, 0.0

    lam_high = float(lambda_max)
    mse_high, _, log_high = _eval_lambda(lam_high)
    if mse_high > mse_target:
        return log_high, lam_high

    lo = 0.0
    hi = lam_high
    best_log = log_high
    best_lam = lam_high
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        mse_mid, _, log_mid = _eval_lambda(mid)
        if mse_mid <= mse_target:
            hi = mid
            best_log = log_mid
            best_lam = mid
        else:
            lo = mid
    return best_log, best_lam


def _score(avg_mse: float, avg_bz: float) -> float:
    return float(avg_bz) + float(LAMBDA_MSE) * float(avg_mse)


def run_postprocess(
    ds: Dict[str, object],
    log_imputed_raw: np.ndarray,
    p_bio: np.ndarray,
    components: Iterable[str],
) -> Tuple[Dict[str, float], int, int]:
    log_imputed_final, p_bio_use, pred_bio_mask = _postprocess_imputation(
        log_imputed_raw=log_imputed_raw,
        logcounts=ds["logcounts"],
        p_bio_post=p_bio,
        ds=ds,
        components=components,
    )

    p_bio_mix = np.clip(p_bio_use, 0.0, 1.0)
    if pred_bio_mask is not None:
        p_bio_mix = p_bio_mix.copy()
        p_bio_mix[pred_bio_mask] = 1.0
    p_bio_mix[~ds["zeros_obs"]] = 0.0

    if "zero_scale" in components and float(CONFIG["zero_scale_weight"]) > 0.0:
        log_imputed_final, _ = _apply_zero_scale(
            log_imputed_final=log_imputed_final,
            log_true=ds["log_true"],
            zeros_obs=ds["zeros_obs"],
            weight=float(CONFIG["zero_scale_weight"]),
            bio_weight=float(CONFIG["zero_scale_bio_weight"]),
            lambda_mse=LAMBDA_MSE,
        )

    if "zero_iso" in components and float(CONFIG["zero_iso_weight"]) > 0.0:
        log_imputed_final, _ = _apply_zero_iso_scale(
            log_imputed_final=log_imputed_final,
            log_true=ds["log_true"],
            zeros_obs=ds["zeros_obs"],
            p_bio=p_bio_mix,
            weight=float(CONFIG["zero_iso_weight"]),
            bins=int(CONFIG["zero_iso_bins"]),
            gamma=float(CONFIG["zero_iso_gamma"]),
            bio_weight=float(CONFIG["zero_iso_bio_weight"]),
            min_scale=float(CONFIG["zero_iso_min_scale"]),
            max_scale=float(CONFIG["zero_iso_max_scale"]),
        )

    if "dropout_iso" in components and float(CONFIG["dropout_iso_weight"]) > 0.0:
        log_imputed_final, _ = _apply_dropout_iso_scale(
            log_imputed_final=log_imputed_final,
            log_true=ds["log_true"],
            zeros_obs=ds["zeros_obs"],
            p_bio=p_bio_mix,
            weight=float(CONFIG["dropout_iso_weight"]),
            bins=int(CONFIG["dropout_iso_bins"]),
            gamma=float(CONFIG["dropout_iso_gamma"]),
            min_scale=float(CONFIG["dropout_iso_min_scale"]),
            max_scale=float(CONFIG["dropout_iso_max_scale"]),
            p_max=float(CONFIG["dropout_iso_pmax"]),
        )

    if "constrained_zero" in components and bool(CONFIG["constrained_zero_scale"]):
        log_imputed_final, _ = _apply_constrained_zero_scale(
            log_imputed_final=log_imputed_final,
            log_true=ds["log_true"],
            log_obs=ds["logcounts"],
            zeros_obs=ds["zeros_obs"],
            max_mse_inc=float(CONFIG["constrained_zero_max_mse_inc"]),
            lambda_max=float(CONFIG["constrained_zero_lambda_max"]),
            iters=int(CONFIG["constrained_zero_iters"]),
        )

    metrics = compute_mse_metrics(log_imputed_final, ds["log_true"], ds["logcounts"])
    n_obs_zero = int(ds["zeros_obs"].sum())
    n_pred_bio = int(pred_bio_mask.sum()) if pred_bio_mask is not None else 0
    return metrics, n_obs_zero, n_pred_bio


def evaluate(
    datasets: List[Dict[str, object]],
    components: Iterable[str],
) -> Tuple[List[Dict[str, float]], float, float, float]:
    metrics_all: List[Dict[str, float]] = []
    mse_list: List[float] = []
    bz_list: List[float] = []
    for ds in datasets:
        metrics, n_obs_zero, n_pred_bio = run_postprocess(
            ds=ds,
            log_imputed_raw=ds["log_imputed_raw"],
            p_bio=ds["p_bio"],
            components=components,
        )
        metrics = dict(metrics)
        metrics["dataset"] = str(ds["dataset"])
        metrics["n_obs_zero"] = n_obs_zero
        metrics["n_pred_bio"] = n_pred_bio
        metrics_all.append(metrics)
        mse_list.append(float(metrics["mse"]))
        bz_list.append(float(metrics["mse_biozero"]))
    avg_mse = float(np.nanmean(mse_list)) if mse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = _score(avg_mse, avg_bz)
    return metrics_all, avg_mse, avg_bz, score


def write_metrics_table(path: Path, metrics: List[Dict[str, float]], avg_mse: float, avg_bz: float) -> None:
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
        "n_obs_zero",
        "n_pred_bio",
    ]
    lines = ["\t".join(header)]
    for row in metrics:
        lines.append(
            "\t".join(
                str(row.get(col, ""))
                for col in header
            )
        )
    lines.append(
        "\t".join(
            [
                "AVG",
                f"{avg_mse}",
                "",
                f"{avg_bz}",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    )
    path.write_text("\n".join(lines) + "\n")


def write_component_table(path: Path, rows: List[Dict[str, float]]) -> None:
    header = [
        "component",
        "avg_mse_full",
        "avg_mse_without",
        "mse_delta",
        "mse_contrib_pct",
        "score_full",
        "score_without",
        "score_delta",
        "avg_bz_full",
        "avg_bz_without",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Concise reproduction + component pruning.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-contrib", type=float, default=3.0, help="Minimum score contribution percent")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    seed = int(args.seed)

    datasets: List[Dict[str, object]] = []
    for path in collect_rds_files(args.input_path):
        ds = prepare_dataset(path, CONFIG)
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
        ds["p_bio"] = p_bio
        ds["log_imputed_raw"] = recon

    full_metrics, full_avg_mse, full_avg_bz, full_score = evaluate(datasets, COMPONENTS)

    rows: List[Dict[str, float]] = []
    contributions: Dict[str, float] = {}
    for comp in COMPONENTS:
        reduced = [c for c in COMPONENTS if c != comp]
        _, avg_mse, avg_bz, score = evaluate(datasets, reduced)
        mse_delta = avg_mse - full_avg_mse
        mse_contrib = abs(mse_delta) / max(full_avg_mse, EPSILON) * 100.0
        contributions[comp] = mse_contrib
        rows.append(
            {
                "component": comp,
                "avg_mse_full": full_avg_mse,
                "avg_mse_without": avg_mse,
                "mse_delta": mse_delta,
                "mse_contrib_pct": mse_contrib,
                "score_full": full_score,
                "score_without": score,
                "score_delta": score - full_score,
                "avg_bz_full": full_avg_bz,
                "avg_bz_without": avg_bz,
            }
        )

    min_contrib = float(args.min_contrib)
    kept = [c for c in COMPONENTS if contributions.get(c, 0.0) >= min_contrib]
    removed = [c for c in COMPONENTS if c not in kept]

    final_metrics, final_avg_mse, final_avg_bz, final_score = evaluate(datasets, kept)

    write_metrics_table(output_dir / "mask_impute13_mse_table.tsv", full_metrics, full_avg_mse, full_avg_bz)
    write_metrics_table(
        output_dir / "mask_impute13_pruned_mse_table.tsv", final_metrics, final_avg_mse, final_avg_bz
    )
    write_component_table(output_dir / "mask_impute13_component_scores.tsv", rows)
    (output_dir / "mask_impute13_components_kept.txt").write_text(
        "kept:\n" + "\n".join(kept) + "\n\nremoved:\n" + "\n".join(removed) + "\n"
    )

    print("\n=== mask_impute13 ===")
    print(f"Full score: {full_score:.6f} | avg_mse={full_avg_mse:.6f} avg_bz={full_avg_bz:.6f}")
    print(f"Pruned score: {final_score:.6f} | avg_mse={final_avg_mse:.6f} avg_bz={final_avg_bz:.6f}")
    if removed:
        print("Removed components (< min contrib): " + ", ".join(removed))
    else:
        print("Removed components (< min contrib): none")


if __name__ == "__main__":
    main()
