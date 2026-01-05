#!/usr/bin/env python3
"""
mask_impute4.py
--------------

Masked AE imputation using SPLAT cell-aware (mode none, thr_drop=0.8200) with
fixed p_zero=0.10 and p_nz=0.30. Non-zero masked entries are zeroed, while
predicted biological-zero masked entries are corrupted with Uniform(0,0.5)
noise in raw counts, then mapped to the training scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from DenseLayerPack import DenseLayer
from DenseLayerPack.const import DENSE_LAYER_CONST
from predict_dropouts_new import splatter_bio_posterior_from_counts
from rds2py import read_rds

EPSILON = 1e-6
THR_DROP = 0.8200
P_ZERO = 0.10
P_NZ = 0.30


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


class AE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: List[int],
        bottleneck: int,
        layer_type: str,
    ):
        super().__init__()
        sizes_enc = [input_dim] + list(hidden) + [bottleneck]
        sizes_dec = [bottleneck] + list(reversed(hidden)) + [input_dim]

        enc_layers = []
        for i in range(len(sizes_enc) - 1):
            enc_layers.append(DenseLayer(sizes_enc[i], sizes_enc[i + 1], layer_type=layer_type))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for i in range(len(sizes_dec) - 1):
            dec_layers.append(DenseLayer(sizes_dec[i], sizes_dec[i + 1], layer_type=layer_type))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def mse_from_residual(residual: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        residual = residual * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(residual.numel(), device=residual.device, dtype=residual.dtype)
    return residual.pow(2).sum() / denom


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


def splat_cellaware_bio_mask(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    thr_drop: float,
) -> Tuple[np.ndarray, np.ndarray]:
    bio_post = splatter_bio_posterior_from_counts(
        counts,
        disp_mode="estimate",
        disp_const=0.1,
        use_cell_factor=True,
        groups=None,
    )
    p_bio_z = np.asarray(bio_post, dtype=np.float64)[zeros_obs]
    p_bio_z = np.nan_to_num(p_bio_z, nan=0.0, posinf=0.0, neginf=0.0)
    p_bio_z = np.clip(p_bio_z, 0.0, 1.0)
    thr_bio = 1.0 - float(thr_drop)
    pred_bio_z = p_bio_z >= thr_bio

    pred_mask = np.zeros_like(zeros_obs, dtype=bool)
    pred_mask[zeros_obs] = pred_bio_z.astype(bool)
    return pred_mask, p_bio_z.astype(np.float32)


def train_autoencoder_reconstruct(
    logcounts: np.ndarray,
    counts_obs: np.ndarray,
    pred_bio_mask: np.ndarray,
    device: torch.device,
    hidden: List[int],
    bottleneck: int,
    epochs: int = 100,
    batch_size: int = 32,
) -> np.ndarray:
    scaler = RobustZThenMinMaxToNeg1Pos1().fit(logcounts)
    Xs = scaler.transform(logcounts).astype(np.float32)

    bio_zero_mask = pred_bio_mask.astype(bool)
    nonzero_mask = (logcounts > 0.0)

    Xtr = torch.tensor(Xs, dtype=torch.float32)
    bio_mask = torch.tensor(bio_zero_mask.astype(np.float32), dtype=torch.float32)
    nz_mask = torch.tensor(nonzero_mask.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xtr, bio_mask, nz_mask), batch_size=batch_size, shuffle=True, drop_last=False)

    lo = torch.tensor(scaler.lo_, dtype=torch.float32, device=device)
    hi = torch.tensor(scaler.hi_, dtype=torch.float32, device=device)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    std = torch.tensor(scaler.std_, dtype=torch.float32, device=device)
    zmin = torch.tensor(scaler.zmin_, dtype=torch.float32, device=device)
    zspan = torch.tensor(scaler.zspan_, dtype=torch.float32, device=device)
    log2_base = float(np.log(2.0))
    counts_max = torch.tensor(np.maximum(counts_obs.max(axis=0), 1.0), dtype=torch.float32, device=device)

    model = AE(
        input_dim=logcounts.shape[1],
        hidden=hidden,
        bottleneck=bottleneck,
        layer_type=DENSE_LAYER_CONST.SILU_LAYER,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    model.train()
    for _ in range(int(epochs)):
        for xb, bio_b, nz_b in loader:
            xb = xb.to(device)
            bio_b = bio_b.to(device)
            nz_b = nz_b.to(device)

            mask_bio = torch.bernoulli(bio_b * float(P_ZERO))
            mask_nz = torch.bernoulli(nz_b * float(P_NZ))
            mask_total = torch.clamp(mask_bio + mask_nz, 0.0, 1.0)

            x_in = xb.clone()
            if mask_nz.any():
                x_in = torch.where(mask_nz.bool(), torch.zeros_like(x_in), x_in)
            if mask_bio.any():
                noise_counts = torch.rand_like(xb) * 0.5 * counts_max
                log_noise = torch.log1p(noise_counts) / log2_base
                log_noise = torch.minimum(torch.maximum(log_noise, lo), hi)
                z = (log_noise - mean) / std
                x01 = (z - zmin) / zspan
                noise_scaled = x01 * 2.0 - 1.0
                x_in = torch.where(mask_bio.bool(), noise_scaled, x_in)

            opt.zero_grad()
            recon = model(x_in)
            residual = recon - xb
            loss = mse_from_residual(residual, mask=mask_total)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked AE imputation with noise-corrupted bio-zero masking.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory for mask_impute4_mse_table.tsv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-positive", action="store_true", help="Keep observed positive entries after imputation.")
    parser.add_argument(
        "--post-threshold",
        type=float,
        default=0.75,
        help="After imputation, set log_imputed to 0 at observed zeros when below this threshold.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    hidden = [64]
    bottleneck = 32

    results_final: List[Dict[str, object]] = []
    results_raw: List[Dict[str, object]] = []
    results_no_bio: List[Dict[str, object]] = []

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

        pred_bio_mask, _p_bio_z = splat_cellaware_bio_mask(
            counts_obs, zeros_obs, thr_drop=float(THR_DROP)
        )
        if np.any(pred_bio_mask & ~zeros_obs):
            raise RuntimeError(f"{ds_name}: pred_bio_mask contains non-zero entries outside observed zeros.")

        log_imputed_raw = train_autoencoder_reconstruct(
            logcounts,
            counts_obs,
            pred_bio_mask,
            device=device,
            hidden=hidden,
            bottleneck=bottleneck,
        )

        log_imputed_keep = log_imputed_raw.copy()
        if args.keep_positive:
            pos_mask = logcounts > 0.0
            log_imputed_keep[pos_mask] = logcounts[pos_mask]

        log_imputed_final = log_imputed_keep.copy()
        log_imputed_final[pred_bio_mask] = 0.0
        low_mask = (logcounts <= 0.0) & (log_imputed_final < float(args.post_threshold))
        log_imputed_final[low_mask] = 0.0

        row_base = {
            "dataset": ds_name,
            "thr_drop": THR_DROP,
            "thr_bio": 1.0 - THR_DROP,
            "p_zero": P_ZERO,
            "p_nz": P_NZ,
            "keep_positive": bool(args.keep_positive),
            "n_obs_zero": int(zeros_obs.sum()),
            "n_pred_bio": int(pred_bio_mask.sum()),
        }
        results_raw.append({**row_base, **compute_mse_metrics(log_imputed_raw, log_true, logcounts)})
        results_no_bio.append({**row_base, **compute_mse_metrics(log_imputed_keep, log_true, logcounts)})
        results_final.append({**row_base, **compute_mse_metrics(log_imputed_final, log_true, logcounts)})

    if not results_final:
        raise SystemExit("No datasets processed.")

    def _write_table(path: Path, rows: List[Dict[str, object]]) -> None:
        columns = [
            "dataset",
            "thr_drop",
            "thr_bio",
            "p_zero",
            "p_nz",
            "keep_positive",
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

    _write_table(output_dir / "mask_impute4_raw_mse_table.tsv", results_raw)
    _write_table(output_dir / "mask_impute4_no_biozero_mse_table.tsv", results_no_bio)
    _write_table(output_dir / "mask_impute4_mse_table.tsv", results_final)

    print(f"Wrote {output_dir / 'mask_impute4_raw_mse_table.tsv'}")
    print(f"Wrote {output_dir / 'mask_impute4_no_biozero_mse_table.tsv'}")
    print(f"Wrote {output_dir / 'mask_impute4_mse_table.tsv'}")


if __name__ == "__main__":
    main()
