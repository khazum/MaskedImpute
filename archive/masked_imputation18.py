#!/usr/bin/env python3
"""
masked_imputation18.py

Ablation study for masked_imputation17.py. Measures the impact of:
- Proxy Labels
- Post-processing Imputation

Runs both balanced (best biozero MSE) and non-balanced (best overall MSE)
baselines before ablation.
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

PROXY_IMPUTE_ALPHA = 0.5
AUTO_BIO_THRESHOLD = 0.04

CONFIG = {
    "disp_mode": "estimate",
    "disp_const": 0.05,
    "use_cell_factor": True,
    "tau_dispersion": 20.0,
    "tau_group_dispersion": 20.0,
    "tau_dropout": 50.0,
    "cell_zero_weight": 0.6,
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


def collect_rds_files(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        return sorted(path.rglob("*.rds"))
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Input path not found: {input_path}")


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
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(p_clip / (1.0 - p_clip))
    logit = logit * float(temp) + float(bias)
    return _sigmoid(logit).astype(np.float32, copy=False)


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


def run_pipeline(
    ds: Dict[str, object],
    p_bio_temp: float,
    p_bio_bias: float,
    proxy_drop_weight: float,
    keep_positive: bool,
    use_proxy_labels: bool,
    use_postprocess: bool,
) -> Dict[str, float]:
    zeros_obs = ds["zeros_obs"]
    p_bio = ds["p_bio_base"]
    p_bio = _logit_scale_probs(p_bio, temp=p_bio_temp, bias=p_bio_bias)

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

    log_imputed = ds["log_recon"].copy()
    if bool(keep_positive):
        log_imputed[~zeros_obs] = ds["logcounts"][~zeros_obs]

    if use_postprocess:
        bio_mask = None
        if use_proxy_labels and proxy_bio_mask is not None:
            bio_mask = proxy_bio_mask & (proxy_bio_label > 0.5)
            if np.any(bio_mask):
                log_imputed[bio_mask] = 0.0

        pred_bio_mask = zeros_obs & (p_bio >= float(AUTO_BIO_THRESHOLD))
        if np.any(pred_bio_mask):
            log_imputed[pred_bio_mask] = 0.0

        if float(PROXY_IMPUTE_ALPHA) > 0.0:
            proxy_mean = _select_proxy_mean(ds)
            p_scale = np.clip(1.0 - p_bio, 0.0, 1.0)
            apply_mask = zeros_obs & (~pred_bio_mask)
            if bio_mask is not None:
                apply_mask = apply_mask & (~bio_mask)
            log_imputed[apply_mask] = (1.0 - PROXY_IMPUTE_ALPHA) * log_imputed[apply_mask] + (
                PROXY_IMPUTE_ALPHA * proxy_mean[apply_mask] * p_scale[apply_mask]
            )

    return compute_mse_metrics(
        log_imputed,
        ds["log_true"],
        ds["counts"],
    )


def evaluate_datasets(
    datasets: List[Dict[str, object]],
    cfg: Dict[str, object],
) -> Tuple[List[Dict[str, float]], float, float, float]:
    metrics_all: List[Dict[str, float]] = []
    mse_list: List[float] = []
    bz_list: List[float] = []
    for ds in datasets:
        metrics = run_pipeline(ds=ds, **cfg)
        metrics = dict(metrics)
        metrics["dataset"] = str(ds["dataset"])
        metrics_all.append(metrics)
        mse_list.append(float(metrics["mse"]))
        bz_list.append(float(metrics["mse_biozero"]))
    avg_mse = float(np.nanmean(mse_list)) if mse_list else float("nan")
    avg_bz = float(np.nanmean(bz_list)) if bz_list else float("nan")
    score = float(avg_bz) + float(LAMBDA_MSE) * float(avg_mse)
    return metrics_all, avg_mse, avg_bz, score


def write_table(path: Path, header: List[str], rows: List[Dict[str, object]]) -> None:
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n")


def _parse_float_list(text: str) -> List[float]:
    if text is None:
        return []
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study for masked_imputation17.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-positive", default="true")
    parser.add_argument("--p-bio-temp-grid", default="1.2,1.5,1.8,2.0")
    parser.add_argument("--p-bio-bias-grid", default="-1.0,-0.5,0.0,0.5")
    parser.add_argument("--proxy-drop-weight-grid", default="0.0,0.3,0.5,0.7")
    parser.add_argument("--balanced-p-bio-bias", type=float, default=0.5)
    parser.add_argument("--fixed-proxy-drop-weight", type=float, default=0.5)
    args = parser.parse_args()

    keep_positive = str(args.keep_positive).strip().lower() in ("1", "true", "yes", "y")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

        p_bio = splat_cellaware_bio_prob(
            counts=ds["counts"],
            zeros_obs=ds["zeros_obs"],
            disp_mode=str(CONFIG["disp_mode"]),
            disp_const=float(CONFIG["disp_const"]),
            use_cell_factor=bool(CONFIG["use_cell_factor"]),
            tau_dispersion=float(CONFIG["tau_dispersion"]),
            tau_group_dispersion=float(CONFIG["tau_group_dispersion"]),
            tau_dropout=float(CONFIG["tau_dropout"]),
        )
        cell_zero_weight = float(CONFIG["cell_zero_weight"])
        if cell_zero_weight > 0.0:
            cell_w = np.clip(cell_zero_weight * ds["cell_zero_norm"], 0.0, 1.0)
            p_bio = p_bio * (1.0 - cell_w[:, None])

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
        ds["log_recon"] = recon

    p_bio_temp_list = _parse_float_list(args.p_bio_temp_grid)
    p_bio_bias_list = _parse_float_list(args.p_bio_bias_grid)
    proxy_drop_weight_list = _parse_float_list(args.proxy_drop_weight_grid)

    def _eval_full(cfg: Dict[str, object]) -> Tuple[float, float, float]:
        cfg_full = dict(cfg)
        cfg_full.update(
            {
                "use_proxy_labels": True,
                "use_postprocess": True,
            }
        )
        _, avg_mse, avg_bz, score = evaluate_datasets(datasets, cfg_full)
        return avg_mse, avg_bz, score

    # Balanced baseline: minimize score (avg_bz_mse + 0.5 * avg_mse).
    balanced_candidates: List[Dict[str, object]] = []
    for p_bio_temp in p_bio_temp_list:
        for proxy_drop_weight in proxy_drop_weight_list:
            cfg = {
                "p_bio_temp": p_bio_temp,
                "p_bio_bias": float(args.balanced_p_bio_bias),
                "proxy_drop_weight": proxy_drop_weight,
                "keep_positive": bool(keep_positive),
            }
            avg_mse, avg_bz, score = _eval_full(cfg)
            cfg = dict(cfg)
            cfg.update({"avg_mse": avg_mse, "avg_bz_mse": avg_bz, "score": score})
            balanced_candidates.append(cfg)
    balanced_candidates.sort(key=lambda r: float(r["score"]))
    balanced_cfg = balanced_candidates[0]

    # Non-balanced baseline: minimize overall MSE.
    mse_candidates: List[Dict[str, object]] = []
    for p_bio_temp in p_bio_temp_list:
        for p_bio_bias in p_bio_bias_list:
            cfg = {
                "p_bio_temp": p_bio_temp,
                "p_bio_bias": p_bio_bias,
                "proxy_drop_weight": float(args.fixed_proxy_drop_weight),
                "keep_positive": bool(keep_positive),
            }
            avg_mse, avg_bz, score = _eval_full(cfg)
            cfg = dict(cfg)
            cfg.update({"avg_mse": avg_mse, "avg_bz_mse": avg_bz, "score": score})
            mse_candidates.append(cfg)
    mse_candidates.sort(key=lambda r: float(r["avg_mse"]))
    mse_cfg = mse_candidates[0]

    scenarios = [
        {
            "name": "full",
            "use_proxy_labels": True,
            "use_postprocess": True,
        },
        {
            "name": "no_proxy_labels",
            "use_proxy_labels": False,
            "use_postprocess": True,
        },
        {
            "name": "no_postprocess",
            "use_proxy_labels": True,
            "use_postprocess": False,
        },
    ]

    rows: List[Dict[str, object]] = []
    for label, base_cfg in (("balanced", balanced_cfg), ("non_balanced", mse_cfg)):
        base_cfg = {
            "p_bio_temp": float(base_cfg["p_bio_temp"]),
            "p_bio_bias": float(base_cfg["p_bio_bias"]),
            "proxy_drop_weight": float(base_cfg["proxy_drop_weight"]),
            "keep_positive": bool(keep_positive),
        }
        for scenario in scenarios:
            cfg = dict(base_cfg)
            cfg.update(
                {
                    "use_proxy_labels": scenario["use_proxy_labels"],
                    "use_postprocess": scenario["use_postprocess"],
                }
            )
            _, avg_mse, avg_bz, score = evaluate_datasets(datasets, cfg)
            rows.append(
                {
                    "baseline": label,
                    "scenario": scenario["name"],
                    "p_bio_temp": base_cfg["p_bio_temp"],
                    "p_bio_bias": base_cfg["p_bio_bias"],
                    "proxy_drop_weight": base_cfg["proxy_drop_weight"],
                    "avg_mse": avg_mse,
                    "avg_bz_mse": avg_bz,
                    "score": score,
                }
            )

    write_table(
        output_dir / "masked_imputation18_ablation.tsv",
        [
            "baseline",
            "scenario",
            "p_bio_temp",
            "p_bio_bias",
            "proxy_drop_weight",
            "avg_mse",
            "avg_bz_mse",
            "score",
        ],
        rows,
    )

    print("\n=== masked_imputation18 ===")
    print("Balanced baseline:", balanced_cfg)
    print("Non-balanced baseline:", mse_cfg)
    for row in rows:
        print(row)
    print("Ablation table written to masked_imputation18_ablation.tsv")


if __name__ == "__main__":
    main()
