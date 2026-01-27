#!/usr/bin/env python3
"""
masked_imputation25.py

Fixed-config imputation tool (no tuning).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def train_autoencoder_reconstruct(
    logcounts: np.ndarray,
    counts_max: np.ndarray,
    p_bio: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    scaler = RobustZThenMinMaxToNeg1Pos1(
        p_low=float(SCALER_PARAMS["p_low"]), p_high=float(SCALER_PARAMS["p_high"])
    ).fit(logcounts)
    Xs = scaler.transform(logcounts).astype(np.float32)

    bio_prob = p_bio.astype(np.float32)
    nonzero_mask = logcounts > 0.0

    Xtr = torch.tensor(Xs, dtype=torch.float32)
    bio_mask = torch.tensor(bio_prob, dtype=torch.float32)
    nz_mask = torch.tensor(nonzero_mask.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(Xtr, bio_mask, nz_mask),
        batch_size=int(MODEL_PARAMS["batch_size"]),
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
    counts_max_t = torch.tensor(np.maximum(counts_max, 1.0), dtype=torch.float32, device=device)

    model = ImprovedAE(
        input_dim=logcounts.shape[1],
        hidden=MODEL_PARAMS["hidden"],
        bottleneck=int(MODEL_PARAMS["bottleneck"]),
    ).to(device)
    opt = optim.Adam(
        model.parameters(),
        lr=float(AE_PARAMS["lr"]),
        weight_decay=float(MODEL_PARAMS["weight_decay"]),
    )

    model.train()
    for _ in range(int(AE_PARAMS["epochs"])):
        for xb, bio_b, nz_b in loader:
            xb = xb.to(device)
            bio_b = bio_b.to(device)
            nz_b = nz_b.to(device)

            mask_bio = torch.bernoulli(bio_b * float(AE_PARAMS["p_zero"]))
            mask_nz = torch.bernoulli(nz_b * float(AE_PARAMS["p_nz"]))

            x_in = xb.clone()
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
            loss.backward()
            opt.step()

    model.eval()
    recon_list = []
    with torch.no_grad():
        for i in range(0, Xtr.size(0), int(MODEL_PARAMS["batch_size"])):
            xb = Xtr[i : i + int(MODEL_PARAMS["batch_size"])].to(device)
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
    parser = argparse.ArgumentParser(description="Fixed-config imputation tool.")
    parser.add_argument("input_path", help="Path to .rds file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-positive", default="true")
    parser.add_argument("--save-imputed", default="true")
    parser.add_argument("--bio-reg-weight", type=float, default=1.0)
    parser.add_argument("--biozero-samples", type=int, default=0)
    parser.add_argument("--biozero-progress-every", type=int, default=25)
    args = parser.parse_args()

    keep_positive = str(args.keep_positive).strip().lower() in ("1", "true", "yes", "y")
    save_imputed = str(args.save_imputed).strip().lower() in ("1", "true", "yes", "y")
    AE_PARAMS["bio_reg_weight"] = float(args.bio_reg_weight)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    for ds in datasets:
        set_seed(int(args.seed))
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

        recon = train_autoencoder_reconstruct(
            logcounts=ds["logcounts"],
            counts_max=ds["counts_max"],
            p_bio=p_bio,
            device=device,
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

    _write_table(
        output_dir / "masked_imputation25_metrics.tsv",
        ["dataset", "mse", "mse_biozero", "mse_dropout", "mse_non_zero"],
        rows,
    )
    _write_table(
        output_dir / "masked_imputation25_summary.tsv",
        ["avg_mse", "avg_bz_mse", "score"],
        [{"avg_mse": avg_mse, "avg_bz_mse": avg_bz, "score": score}],
    )

    print("\n=== masked_imputation25 ===")
    print("Biozero params:", BIO_PARAMS)
    print("AE params:", AE_PARAMS)
    print("avg_mse:", avg_mse, "avg_bz_mse:", avg_bz, "score:", score)
    print("Metrics written to masked_imputation25_metrics.tsv")
    print("Summary written to masked_imputation25_summary.tsv")


if __name__ == "__main__":
    main()
