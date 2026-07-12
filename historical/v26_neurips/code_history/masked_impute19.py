#!/usr/bin/env python3
"""
masked_impute19.py

Supervised tuning + pruning for a masked-loss autoencoder.

Strategy:
1) Tune the biozero probability model against ground-truth biozero labels.
2) Tune masked denoising AE parameters (no proxy labeling/post-process).
3) Tune proxy labeling + post-process parameters.

Parameters that improve MSE by <3% are dropped along with their logic.
"""

from __future__ import annotations

import argparse
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
        p for p in sys.path if str(_REPO_ROOT) not in p and "MaskedImpute/rds2py" not in p
    ]
    from rds2py import read_rds
finally:
    sys.path = _SYS_PATH

EPSILON = 1e-6
GENE_NORM_LOW = 5.0
GENE_NORM_HIGH = 95.0
LAMBDA_MSE = 0.5

PROXY_IMPUTE_ALPHA_DEFAULT = 0.5
AUTO_BIO_THRESHOLD_DEFAULT = 0.04

CONFIG = {
    "disp_mode": "estimate",
    "disp_const": 0.05,
    "use_cell_factor": True,
    "tau_dispersion": 20.0,
    "tau_group_dispersion": 20.0,
    "tau_dropout": 50.0,
    "cell_zero_weight": 0.6,
    "hidden": [128, 64],
    "bottleneck": 64,
    "dropout": 0.0,
    "batch_size": 32,
    "weight_decay": 0.0,
    "p_low": 2.0,
    "p_high": 99.5,
    "proxy_gene_bio_max": 0.02,
    "proxy_gene_drop_min": 0.2,
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
        use_residual: bool,
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


def mse_from_residual(residual: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    counts = dataset["counts"]
    if counts is None:
        counts_obs = np.clip(np.expm1(logcounts * np.log(2.0)), 0.0, None).astype(np.float32)
    else:
        counts_obs = np.clip(counts, 0.0, None)
    zeros_obs = counts_obs <= 0.0
    counts_max = counts_obs.max(axis=0)

    log_nz_mask = logcounts > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_log_mean_nz = np.sum(logcounts * log_nz_mask, axis=0) / np.maximum(
            log_nz_mask.sum(axis=0), 1
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

    proxy_bio_label_gene = np.zeros_like(logcounts, dtype=np.float32)
    proxy_bio_mask_gene = np.zeros_like(logcounts, dtype=bool)
    bio_genes = gene_nz_frac <= float(CONFIG["proxy_gene_bio_max"])
    drop_genes = gene_nz_frac >= float(CONFIG["proxy_gene_drop_min"])
    if np.any(bio_genes):
        bio_mask = zeros_obs & bio_genes[None, :]
        proxy_bio_label_gene[bio_mask] = 1.0
        proxy_bio_mask_gene[bio_mask] = True
    if np.any(drop_genes):
        drop_mask = zeros_obs & drop_genes[None, :]
        proxy_bio_label_gene[drop_mask] = 0.0
        proxy_bio_mask_gene[drop_mask] = True

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
        "gene_log_mean_nz": gene_log_mean_nz,
        "cell_zero_norm": cell_zero_norm,
        "proxy_bio_mask_gene": proxy_bio_mask_gene,
        "proxy_bio_label_gene": proxy_bio_label_gene,
        "biozero_label": biozero_label,
        "biozero_mask": biozero_mask,
    }


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
    loader = DataLoader(
        TensorDataset(Xtr, bio_mask, nz_mask),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

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


def _logit_scale_probs(p: np.ndarray, temp: float, bias: float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    logit = logit * float(temp) + float(bias)
    return 1.0 / (1.0 + np.exp(-logit))


def _select_proxy_mean(ds: Dict[str, object]) -> np.ndarray:
    gene_mean = ds["gene_log_mean_nz"][None, :].astype(np.float32, copy=False)
    gene_mean_full = np.broadcast_to(gene_mean, ds["logcounts"].shape)
    return gene_mean_full.astype(np.float32, copy=False)


def _combine_proxy_labels(ds: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    mask = ds["proxy_bio_mask_gene"].copy()
    label = ds["proxy_bio_label_gene"].copy()
    return mask, label


def _apply_proxy_drop(
    p_bio: np.ndarray,
    zeros_obs: np.ndarray,
    proxy_bio_mask: np.ndarray,
    proxy_bio_label: np.ndarray,
    weight_drop: float,
) -> np.ndarray:
    p_out = p_bio.astype(np.float32, copy=True)
    drop_mask = proxy_bio_mask & (proxy_bio_label <= 0.5) & zeros_obs
    weight_drop = float(np.clip(weight_drop, 0.0, 1.0))
    if weight_drop > 0.0 and np.any(drop_mask):
        p_out[drop_mask] = p_out[drop_mask] * (1.0 - weight_drop)
    return p_out


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
    }


def run_pipeline(
    ds: Dict[str, object],
    log_recon: np.ndarray,
    p_bio_base: np.ndarray,
    *,
    p_bio_temp: float,
    p_bio_bias: float,
    keep_positive: bool,
    use_postprocess: bool,
    bio_threshold: float,
    use_proxy_labels: bool,
    proxy_drop_weight: float,
    proxy_impute_alpha: float,
) -> Dict[str, float]:
    zeros_obs = ds["zeros_obs"]
    p_bio = _logit_scale_probs(p_bio_base, temp=p_bio_temp, bias=p_bio_bias)

    proxy_bio_mask = None
    proxy_bio_label = None
    if use_proxy_labels:
        proxy_bio_mask, proxy_bio_label = _combine_proxy_labels(ds)
        if float(proxy_drop_weight) > 0.0:
            p_bio = _apply_proxy_drop(
                p_bio,
                zeros_obs=zeros_obs,
                proxy_bio_mask=proxy_bio_mask,
                proxy_bio_label=proxy_bio_label,
                weight_drop=float(proxy_drop_weight),
            )

    log_imputed = log_recon.copy()
    if bool(keep_positive):
        log_imputed[~zeros_obs] = ds["logcounts"][~zeros_obs]

    if use_postprocess:
        bio_mask = None
        if use_proxy_labels and proxy_bio_mask is not None:
            bio_mask = proxy_bio_mask & (proxy_bio_label > 0.5)
            if np.any(bio_mask):
                log_imputed[bio_mask] = 0.0

        pred_bio_mask = zeros_obs & (p_bio >= float(bio_threshold))
        if np.any(pred_bio_mask):
            log_imputed[pred_bio_mask] = 0.0

        if float(proxy_impute_alpha) > 0.0:
            proxy_mean = _select_proxy_mean(ds)
            p_scale = np.clip(1.0 - p_bio, 0.0, 1.0)
            apply_mask = zeros_obs & (~pred_bio_mask)
            if bio_mask is not None:
                apply_mask = apply_mask & (~bio_mask)
            log_imputed[apply_mask] = (1.0 - proxy_impute_alpha) * log_imputed[apply_mask] + (
                proxy_impute_alpha * proxy_mean[apply_mask] * p_scale[apply_mask]
            )

    return compute_mse_metrics(log_imputed, ds["log_true"], ds["counts"])


def evaluate_datasets(
    datasets: List[Dict[str, object]],
    recons: List[np.ndarray],
    p_bio_list: List[np.ndarray],
    cfg: Dict[str, object],
) -> Tuple[float, float, float]:
    mse_list: List[float] = []
    bz_list: List[float] = []
    for ds, recon, p_bio in zip(datasets, recons, p_bio_list):
        metrics = run_pipeline(ds, recon, p_bio, **cfg)
        mse_list.append(float(metrics["mse"]))
        bz_list.append(float(metrics["mse_biozero"]))
    avg_mse = float(np.nanmean(mse_list)) if mse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = float(avg_bz) + float(LAMBDA_MSE) * float(avg_mse)
    return avg_mse, avg_bz, score


def write_table(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


def _parse_float_list(text: str) -> List[float]:
    if text is None:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    if text is None:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def improvement_ratio(prev: float, new: float) -> float:
    denom = max(prev, EPSILON)
    return max(0.0, (prev - new) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised tuning + pruning.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-positive", default="true")

    parser.add_argument("--bio-disp-const-grid", default="0.02,0.05,0.1")
    parser.add_argument("--bio-use-cell-factor-grid", default="true,false")
    parser.add_argument("--bio-tau-dispersion-grid", default="10,20,40")
    parser.add_argument("--bio-tau-group-dispersion-grid", default="10,20,40")
    parser.add_argument("--bio-tau-dropout-grid", default="20,50,80")
    parser.add_argument("--bio-cell-zero-weight-grid", default="0.0,0.3,0.6,0.9")

    parser.add_argument("--epochs-grid", default="50,100,150")
    parser.add_argument("--lr-grid", default="0.0001,0.0005,0.001")
    parser.add_argument("--p-zero-grid", default="0.0,0.05")
    parser.add_argument("--p-nz-grid", default="0.1,0.2,0.3")
    parser.add_argument("--noise-max-grid", default="0.0,0.2,0.5")
    parser.add_argument("--dropout-grid", default="0.0,0.05")
    parser.add_argument("--loss-bio-weight-grid", default="1.0,2.0,3.0")
    parser.add_argument("--loss-nz-weight-grid", default="1.0")
    parser.add_argument("--bio-reg-weight-grid", default="0.0,0.5,1.0")
    parser.add_argument("--recon-weight-grid", default="0.0,0.1,0.2")
    parser.add_argument("--residual-grid", default="false,true")

    parser.add_argument("--p-bio-temp-grid", default="1.0,1.5,2.0")
    parser.add_argument("--p-bio-bias-grid", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--bio-threshold-grid", default="0.02,0.04,0.06,0.08")
    parser.add_argument("--proxy-drop-weight-grid", default="0.0,0.3,0.5,0.7")
    parser.add_argument("--proxy-impute-alpha-grid", default="0.0,0.5")
    args = parser.parse_args()

    keep_positive = str(args.keep_positive).strip().lower() in ("1", "true", "yes", "y")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: List[Dict[str, object]] = []
    for path in sorted(Path(args.input_path).rglob("*.rds")):
        ds = prepare_dataset(path)
        if ds is None:
            print(f"[WARN] {path.stem}: missing logTrueCounts; skipping.")
            continue
        datasets.append(ds)

    if not datasets:
        raise SystemExit("No datasets processed.")

    def set_seed() -> None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    # Step 1: supervised tuning of biozero model
    bio_params = {
        "disp_mode": CONFIG["disp_mode"],
        "disp_const": CONFIG["disp_const"],
        "use_cell_factor": CONFIG["use_cell_factor"],
        "tau_dispersion": CONFIG["tau_dispersion"],
        "tau_group_dispersion": CONFIG["tau_group_dispersion"],
        "tau_dropout": CONFIG["tau_dropout"],
        "cell_zero_weight": CONFIG["cell_zero_weight"],
    }

    bio_param_candidates = [
        ("disp_const", _parse_float_list(args.bio_disp_const_grid)),
        ("use_cell_factor", [s.strip().lower() == "true" for s in args.bio_use_cell_factor_grid.split(",")]),
        ("tau_dispersion", _parse_float_list(args.bio_tau_dispersion_grid)),
        ("tau_group_dispersion", _parse_float_list(args.bio_tau_group_dispersion_grid)),
        ("tau_dropout", _parse_float_list(args.bio_tau_dropout_grid)),
        ("cell_zero_weight", _parse_float_list(args.bio_cell_zero_weight_grid)),
    ]

    def eval_biozero_params(params: Dict[str, object]) -> float:
        mse_list: List[float] = []
        for ds in datasets:
            p_bio = splat_cellaware_bio_prob(
                counts=ds["counts"],
                zeros_obs=ds["zeros_obs"],
                disp_mode=str(params["disp_mode"]),
                disp_const=float(params["disp_const"]),
                use_cell_factor=bool(params["use_cell_factor"]),
                tau_dispersion=float(params["tau_dispersion"]),
                tau_group_dispersion=float(params["tau_group_dispersion"]),
                tau_dropout=float(params["tau_dropout"]),
            )
            if float(params["cell_zero_weight"]) > 0.0:
                cell_w = np.clip(float(params["cell_zero_weight"]) * ds["cell_zero_norm"], 0.0, 1.0)
                p_bio = p_bio * (1.0 - cell_w[:, None])
            mask = ds["biozero_mask"]
            label = ds["biozero_label"]
            mse = _mse_from_diff(p_bio - label, mask)
            mse_list.append(float(mse))
        return float(np.nanmean(mse_list))

    bio_steps: List[Dict[str, object]] = []
    bio_dropped: List[str] = []

    current_score = eval_biozero_params(bio_params)
    for param, candidates in bio_param_candidates:
        best_val = bio_params[param]
        best_score = current_score
        for val in candidates:
            trial = dict(bio_params)
            trial[param] = val
            score = eval_biozero_params(trial)
            if score < best_score:
                best_score = score
                best_val = val
        improvement = improvement_ratio(current_score, best_score)
        kept = improvement >= 0.03
        if kept:
            bio_params[param] = best_val
            current_score = best_score
        else:
            bio_dropped.append(param)
            if param == "cell_zero_weight":
                bio_params[param] = 0.0
            if param == "use_cell_factor":
                bio_params[param] = False
        bio_steps.append(
            {
                "param": param,
                "best_value": best_val,
                "prev_mse": current_score if kept else best_score,
                "best_mse": best_score,
                "improvement": improvement,
                "kept": kept,
            }
        )

    # Precompute tuned p_bio for AE training
    p_bio_list: List[np.ndarray] = []
    for ds in datasets:
        p_bio = splat_cellaware_bio_prob(
            counts=ds["counts"],
            zeros_obs=ds["zeros_obs"],
            disp_mode=str(bio_params["disp_mode"]),
            disp_const=float(bio_params["disp_const"]),
            use_cell_factor=bool(bio_params["use_cell_factor"]),
            tau_dispersion=float(bio_params["tau_dispersion"]),
            tau_group_dispersion=float(bio_params["tau_group_dispersion"]),
            tau_dropout=float(bio_params["tau_dropout"]),
        )
        if float(bio_params["cell_zero_weight"]) > 0.0:
            cell_w = np.clip(float(bio_params["cell_zero_weight"]) * ds["cell_zero_norm"], 0.0, 1.0)
            p_bio = p_bio * (1.0 - cell_w[:, None])
        p_bio_list.append(p_bio)

    write_table(
        output_dir / "masked_impute19_biozero_steps.tsv",
        ["param", "best_value", "best_mse", "improvement", "kept"],
        bio_steps,
    )

    # Step 2: hill-climb AE parameters (masked loss only)
    epochs_list = _parse_int_list(args.epochs_grid)
    lr_list = _parse_float_list(args.lr_grid)
    p_zero_list = _parse_float_list(args.p_zero_grid)
    p_nz_list = _parse_float_list(args.p_nz_grid)
    noise_max_list = _parse_float_list(args.noise_max_grid)
    dropout_list = _parse_float_list(args.dropout_grid)
    loss_bio_weight_list = _parse_float_list(args.loss_bio_weight_grid)
    loss_nz_weight_list = _parse_float_list(args.loss_nz_weight_grid)
    bio_reg_weight_list = _parse_float_list(args.bio_reg_weight_grid)
    recon_weight_list = _parse_float_list(args.recon_weight_grid)
    residual_list = [s.strip().lower() == "true" for s in args.residual_grid.split(",")]

    ae_cfg = {
        "epochs": 100,
        "lr": 5e-4,
        "p_zero": 0.0,
        "p_nz": 0.2,
        "noise_max": 0.2,
        "dropout": 0.0,
        "loss_bio_weight": 2.0,
        "loss_nz_weight": 1.0,
        "bio_reg_weight": 0.0,
        "recon_weight": 0.0,
        "use_residual": False,
    }

    def train_recons(cfg: Dict[str, object]) -> List[np.ndarray]:
        recons: List[np.ndarray] = []
        for ds, p_bio in zip(datasets, p_bio_list):
            set_seed()
            recon = train_autoencoder_reconstruct(
                logcounts=ds["logcounts"],
                counts_max=ds["counts_max"],
                p_bio=p_bio,
                device=device,
                hidden=CONFIG["hidden"],
                bottleneck=int(CONFIG["bottleneck"]),
                p_zero=float(cfg["p_zero"]),
                p_nz=float(cfg["p_nz"]),
                noise_min_frac=0.0,
                noise_max_frac=float(cfg["noise_max"]),
                dropout=float(cfg["dropout"]),
                use_residual=bool(cfg["use_residual"]),
                epochs=int(cfg["epochs"]),
                batch_size=int(CONFIG["batch_size"]),
                lr=float(cfg["lr"]),
                weight_decay=float(CONFIG["weight_decay"]),
                loss_bio_weight=float(cfg["loss_bio_weight"]),
                loss_nz_weight=float(cfg["loss_nz_weight"]),
                bio_reg_weight=float(cfg["bio_reg_weight"]),
                recon_weight=float(cfg["recon_weight"]),
                p_low=float(CONFIG["p_low"]),
                p_high=float(CONFIG["p_high"]),
            )
            recons.append(recon)
        return recons

    def eval_recons(recons: List[np.ndarray]) -> Tuple[float, float, float]:
        return evaluate_datasets(
            datasets,
            recons,
            p_bio_list,
            {
                "p_bio_temp": 1.0,
                "p_bio_bias": 0.0,
                "keep_positive": bool(keep_positive),
                "use_postprocess": False,
                "bio_threshold": 0.0,
                "use_proxy_labels": False,
                "proxy_drop_weight": 0.0,
                "proxy_impute_alpha": 0.0,
            },
        )

    ae_steps: List[Dict[str, object]] = []

    def tune_param(param: str, candidates: List[object]) -> None:
        nonlocal ae_cfg
        current_recons = train_recons(ae_cfg)
        current_mse, current_bz, current_score = eval_recons(current_recons)
        best_val = ae_cfg[param]
        best_mse = current_mse
        for val in candidates:
            trial = dict(ae_cfg)
            trial[param] = val
            recons = train_recons(trial)
            avg_mse, avg_bz, score = eval_recons(recons)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_val = val
        improvement = improvement_ratio(current_mse, best_mse)
        kept = improvement >= 0.03
        if kept:
            ae_cfg[param] = best_val
        else:
            if param in ("bio_reg_weight", "recon_weight"):
                ae_cfg[param] = 0.0
            if param == "use_residual":
                ae_cfg[param] = False
        recons = train_recons(ae_cfg)
        avg_mse, avg_bz, score = eval_recons(recons)
        ae_steps.append(
            {
                "param": param,
                "best_value": best_val,
                "avg_mse": avg_mse,
                "avg_bz_mse": avg_bz,
                "score": score,
                "improvement": improvement,
                "kept": kept,
            }
        )

    tune_param("lr", lr_list)
    tune_param("epochs", epochs_list)
    tune_param("p_zero", p_zero_list)
    tune_param("p_nz", p_nz_list)
    tune_param("noise_max", noise_max_list)
    tune_param("dropout", dropout_list)
    tune_param("loss_bio_weight", loss_bio_weight_list)
    tune_param("loss_nz_weight", loss_nz_weight_list)
    tune_param("bio_reg_weight", bio_reg_weight_list)
    tune_param("recon_weight", recon_weight_list)
    tune_param("use_residual", residual_list)

    write_table(
        output_dir / "masked_impute19_ae_steps.tsv",
        ["param", "best_value", "avg_mse", "avg_bz_mse", "score", "improvement", "kept"],
        ae_steps,
    )

    # Train final AE with tuned params
    final_recons = train_recons(ae_cfg)
    base_mse, base_bz, base_score = eval_recons(final_recons)

    # Step 3: tune proxy labeling + post-process
    p_bio_temp_list = _parse_float_list(args.p_bio_temp_grid)
    p_bio_bias_list = _parse_float_list(args.p_bio_bias_grid)
    bio_threshold_list = _parse_float_list(args.bio_threshold_grid)
    proxy_drop_weight_list = _parse_float_list(args.proxy_drop_weight_grid)
    proxy_impute_alpha_list = _parse_float_list(args.proxy_impute_alpha_grid)

    post_cfg = {
        "p_bio_temp": 1.0,
        "p_bio_bias": 0.0,
        "bio_threshold": AUTO_BIO_THRESHOLD_DEFAULT,
        "proxy_drop_weight": 0.0,
        "proxy_impute_alpha": 0.0,
        "use_postprocess": True,
        "use_proxy_labels": True,
    }

    def eval_post(cfg: Dict[str, object]) -> Tuple[float, float, float]:
        return evaluate_datasets(
            datasets,
            final_recons,
            p_bio_list,
            {
                "p_bio_temp": float(cfg["p_bio_temp"]),
                "p_bio_bias": float(cfg["p_bio_bias"]),
                "keep_positive": bool(keep_positive),
                "use_postprocess": bool(cfg["use_postprocess"]),
                "bio_threshold": float(cfg["bio_threshold"]),
                "use_proxy_labels": bool(cfg["use_proxy_labels"]),
                "proxy_drop_weight": float(cfg["proxy_drop_weight"]),
                "proxy_impute_alpha": float(cfg["proxy_impute_alpha"]),
            },
        )

    post_steps: List[Dict[str, object]] = []
    current_mse, current_bz, current_score = base_mse, base_bz, base_score

    def tune_post_param(param: str, candidates: List[float]) -> None:
        nonlocal current_mse, current_bz, current_score
        best_val = post_cfg[param]
        best_mse = current_mse
        best_bz = current_bz
        best_score = current_score
        for val in candidates:
            trial = dict(post_cfg)
            trial[param] = val
            avg_mse, avg_bz, score = eval_post(trial)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_bz = avg_bz
                best_score = score
                best_val = val
        improvement = improvement_ratio(current_mse, best_mse)
        kept = improvement >= 0.03
        if kept:
            post_cfg[param] = best_val
            current_mse = best_mse
            current_bz = best_bz
            current_score = best_score
        else:
            if param in ("proxy_drop_weight", "proxy_impute_alpha"):
                post_cfg[param] = 0.0
        post_steps.append(
            {
                "param": param,
                "best_value": best_val,
                "avg_mse": current_mse,
                "avg_bz_mse": current_bz,
                "score": current_score,
                "improvement": improvement,
                "kept": kept,
            }
        )

    tune_post_param("p_bio_temp", p_bio_temp_list)
    tune_post_param("p_bio_bias", p_bio_bias_list)
    tune_post_param("bio_threshold", bio_threshold_list)
    tune_post_param("proxy_drop_weight", proxy_drop_weight_list)
    tune_post_param("proxy_impute_alpha", proxy_impute_alpha_list)

    post_improvement = improvement_ratio(base_mse, current_mse)
    if post_improvement < 0.03:
        post_cfg["use_postprocess"] = False
        post_cfg["use_proxy_labels"] = False
        post_cfg["proxy_drop_weight"] = 0.0
        post_cfg["proxy_impute_alpha"] = 0.0
        current_mse, current_bz, current_score = base_mse, base_bz, base_score

    write_table(
        output_dir / "masked_impute19_post_steps.tsv",
        ["param", "best_value", "avg_mse", "avg_bz_mse", "score", "improvement", "kept"],
        post_steps,
    )

    final_summary = [
        {
            "stage": "biozero_tuned",
            "avg_mse": "",
            "avg_bz_mse": "",
            "score": "",
            "details": str(bio_params),
        },
        {
            "stage": "ae_tuned",
            "avg_mse": base_mse,
            "avg_bz_mse": base_bz,
            "score": base_score,
            "details": str(ae_cfg),
        },
        {
            "stage": "postprocess_tuned",
            "avg_mse": current_mse,
            "avg_bz_mse": current_bz,
            "score": current_score,
            "details": str(post_cfg),
        },
    ]

    write_table(
        output_dir / "masked_impute19_summary.tsv",
        ["stage", "avg_mse", "avg_bz_mse", "score", "details"],
        final_summary,
    )

    print("\n=== masked_impute19 ===")
    print("Biozero params:", bio_params)
    print("Dropped biozero params:", bio_dropped)
    print("AE params:", ae_cfg)
    print("Postprocess params:", post_cfg)
    print("Baseline AE MSE:", base_mse, "avg_bz_mse:", base_bz)
    print("Final MSE:", current_mse, "avg_bz_mse:", current_bz)
    print("Biozero steps written to masked_impute19_biozero_steps.tsv")
    print("AE steps written to masked_impute19_ae_steps.tsv")
    print("Postprocess steps written to masked_impute19_post_steps.tsv")
    print("Summary written to masked_impute19_summary.tsv")


if __name__ == "__main__":
    main()
