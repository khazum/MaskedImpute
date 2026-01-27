# run_from_rds.py
# Measure MSE or run CLUSTERING on .rds datasets using a plain Autoencoder (AE).
# - MSE: compares reconstructions to "logTrueCounts" when present.
# - CLUST: evaluates clustering (PCA -> KMeans) on AE reconstructions against labels.
# Supports optional masked-denoising and a switch to turn per-gene input scaling on/off.

import argparse
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Require the repo layout so DenseLayerPack is importable ---
try:
    from DenseLayerPack import DenseLayer
    from DenseLayerPack.const import DENSE_LAYER_CONST
except Exception as e:
    raise ImportError(
        "Import failed. Place this script at the repository root (so 'DenseLayerPack' is on PYTHONPATH)."
    ) from e

from rds2py import read_rds  # must expose read_rds(path) → SCE-like object

# ----- Try to use your repo's evaluate_clustering for perfect metric parity -----
try:
    from utils.evaluation import evaluate_clustering  # ASW, ARI, NMI, PS (Purity)
    _HAVE_UTILS_EVAL = True
except Exception:
    _HAVE_UTILS_EVAL = False
    # Fallback reproducing your utils/evaluation.py (PCA + KMeans → ASW/ARI/NMI/PS).
    # This matches your implementation so metrics remain comparable.  # see repo
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.metrics.cluster import contingency_matrix
    import pandas as pd

    def _purity_score(y_true, y_pred) -> float:
        mat = contingency_matrix(y_true, y_pred)
        return float(np.sum(np.max(mat, axis=0)) / np.sum(mat))

    def evaluate_clustering(imputed_data: np.ndarray, true_labels: np.ndarray) -> "pd.DataFrame":
        n_samples, n_features = imputed_data.shape
        n_components = max(2, min(50, n_samples, n_features))
        emb = PCA(n_components=n_components).fit_transform(np.nan_to_num(imputed_data))
        k = len(np.unique(true_labels))
        kmeans = KMeans(n_clusters=max(2, k), init="k-means++", n_init=10, random_state=42).fit(emb)
        cluster_labels = kmeans.labels_
        asw = silhouette_score(emb, cluster_labels) if 2 <= len(np.unique(cluster_labels)) < n_samples else np.nan
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        ps = _purity_score(true_labels, cluster_labels)
        return pd.DataFrame({"ASW": [round(asw, 4)], "ARI": [round(ari, 4)], "NMI": [round(nmi, 4)], "PS": [round(ps, 4)]})

# --------- Robust per-gene scaler: clip(p1,p99) → z-score → min-max to [-1, 1] ---------
class RobustZThenMinMaxToNeg1Pos1:
    """
    Steps per feature (gene):
      1) Compute low/high percentiles (p1/p99 by default) and clip values to [lo, hi].
      2) Z-score on the clipped values.
      3) Min-max the z-scored values to [-1, 1].
    Inverse transform reverses steps 3 and 2 (clipping is not reversible).
    """
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


# --------- Identity scaler for --scale-input off ---------
class IdentityScaler:
    def fit(self, X: np.ndarray):
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)


# --------- Data loading for .rds with flexible structure ---------
def load_dataset(path: str, need_labels: bool, need_truth: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns:
        logcounts           : (n_cells × n_genes_kept) float32
        log_true_counts     : (n_cells × n_genes_kept) float32 or None
        label_ids           : (n_cells,) int (factorized labels) or None
    """
    sce = read_rds(path)  # must return an object with .assay(), .column_data/colData

    # Assays
    logcounts = sce.assay("logcounts").T.astype("float32")
    keep = np.sum(logcounts != 0, axis=0) >= 2
    logcounts = logcounts[:, keep]

    log_true_counts = None
    if need_truth:
        # Prefer the renamed assay, but allow older files with perfect_logcounts.
        for assay_name in ("logTrueCounts", "perfect_logcounts"):
            try:
                log_true_counts = sce.assay(assay_name).T[:, keep].astype("float32")
                break
            except Exception:
                continue

    labels = None
    if need_labels:
        # Try BiocFrame-like then pandas-like access (matches your verify.py).  # see repo
        colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
        y = None
        if colmd is not None:
            if hasattr(colmd, "get_column_names") and hasattr(colmd, "get_column"):
                colnames = list(map(str, colmd.get_column_names()))
                for key in ("cell_type1", "labels", "Group", "label"):
                    if key in colnames:
                        y = np.asarray(colmd.get_column(key))
                        break
            elif hasattr(colmd, "columns"):
                colnames = list(map(str, getattr(colmd, "columns", [])))
                for key in ("cell_type1", "labels", "Group", "label"):
                    if key in colnames:
                        y = np.asarray(colmd[key])
                        break
        if y is None:
            y = np.zeros(logcounts.shape[0], dtype=int)
        # Factorize to 0..K-1 (same as repo).  # see repo
        import pandas as pd
        labels, _ = pd.factorize(np.asarray(y))

    return logcounts, log_true_counts, (labels.astype(int) if labels is not None else None)


# --------- Model: plain AE (no classifier) ---------
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

        # Encoder
        enc_layers = []
        for i in range(len(sizes_enc) - 1):
            enc_layers.append(DenseLayer(sizes_enc[i], sizes_enc[i + 1], layer_type=layer_type))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        for i in range(len(sizes_dec) - 1):
            dec_layers.append(DenseLayer(sizes_dec[i], sizes_dec[i + 1], layer_type=layer_type))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# --------- Utility: masked MSE ---------
def mse_from_residual(
    residual: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        residual = residual * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(residual.numel(), device=residual.device, dtype=residual.dtype)
    return residual.pow(2).sum() / denom


# --------- Training / Evaluation for a single dataset/layer ---------
def train_and_eval_single(
    X: np.ndarray,
    X_gt: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    args,
    device: torch.device,
    dataset_name: str,
    layer_type: str,
) -> Dict[str, float]:
    # Choose scaler based on flag
    scale_on = (args.scale_input == "on")
    scaler = (RobustZThenMinMaxToNeg1Pos1().fit(X) if scale_on else IdentityScaler().fit(X))
    Xs = scaler.transform(X).astype(np.float32)

    # Precompute per-feature mean in the space fed to the model (scaled or raw)
    feat_mean_for_fill = Xs.mean(axis=0).astype(np.float32)

    # Zero-indicator from original logcounts (0 means zero after log2(·+1))
    zeros_indicator = (X <= 0.0).astype(np.float32)

    # Tensors / loader
    Xtr = torch.tensor(Xs, dtype=torch.float32)
    Ztr = torch.tensor(zeros_indicator, dtype=torch.float32)  # for masked-denoise sampling

    tr_ds = TensorDataset(Xtr, Ztr)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Model / Optim
    model = AE(
        input_dim=X.shape[1],
        hidden=[int(h) for h in args.hidden.split(",")] if args.hidden else [],
        bottleneck=int(args.bottleneck),
        layer_type=layer_type,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Helper tensors
    feat_mean_t = torch.tensor(feat_mean_for_fill, device=device, dtype=torch.float32).unsqueeze(0)

    # ----- Training loop -----
    model.train()
    for _ in range(args.epochs):
        for xb, zb in tr_loader:
            xb = xb.to(device)      # model-input space (scaled or raw)
            zb = zb.to(device)      # zero indicator from original logcounts

            # Masked-denoising (optional)
            if args.masked_denoise:
                p_zero = args.md_p_zero
                p_nz = args.md_p_nonzero
                probs = torch.where(zb > 0.5,
                                    torch.full_like(xb, p_zero),
                                    torch.full_like(xb, p_nz))
                mask = torch.bernoulli(probs)               # 1 = hide
                if args.md_fill == "mean":
                    fill = feat_mean_t.expand_as(xb)
                elif args.md_fill == "zero":
                    fill = torch.zeros_like(xb)
                else:  # "noise"
                    fill = torch.normal(mean=0.0, std=args.md_noise_std,
                                        size=xb.shape, device=xb.device)
                mask = mask.to(xb.dtype)
                x_in  = (1.0 - mask) * xb + mask * fill
                x_tgt = xb
            else:
                mask = None
                x_in = xb
                x_tgt = xb

            # Forward / loss / step
            opt.zero_grad()
            recon = model(x_in)
            residual = recon - x_tgt
            loss = mse_from_residual(residual, mask=mask)
            loss.backward()
            opt.step()

    # ----- Evaluation -----
    model.eval()
    recon_list = []
    with torch.no_grad():
        for i in range(0, Xtr.size(0), args.batch_size):
            xb = Xtr[i : i + args.batch_size].to(device)
            recon = model(xb)  # full reconstruction, no masking in eval
            recon_np = recon.cpu().numpy()
            recon_orig = scaler.inverse_transform(recon_np)  # if scaling OFF this is identity
            recon_list.append(recon_orig)

    recon_all = np.vstack(recon_list)

    # ---- Metrics by mode ----
    if args.mode == "MSE":
        if X_gt is None:
            return {
                "dataset": dataset_name,
                "layer": layer_type,
                "test_mse": float("nan"),
                "mse_nonzero": float("nan"),
                "mse_biozero": float("nan"),
                "mse_dropout": float("nan"),
                "baseline_mse": float("nan"),
                "baseline_mse_nonzero": float("nan"),
                "baseline_mse_biozero": float("nan"),
                "baseline_mse_dropout": float("nan"),
            }
        diff = X_gt - recon_all
        mse_all = float(np.mean(diff ** 2))

        def _masked_mse(mask: np.ndarray) -> float:
            count = mask.sum()
            if count == 0:
                return float("nan")
            return float(np.mean((diff[mask]) ** 2))

        mask_nonzero = X_gt > 0.0
        mask_biozero = X_gt == 0.0
        mask_dropout = (X_gt > 0.0) & (X <= 0.0)

        mse_nonzero = _masked_mse(mask_nonzero)
        mse_biozero = _masked_mse(mask_biozero)
        mse_dropout = _masked_mse(mask_dropout)

        # Baseline: raw observed counts vs true counts.
        baseline_diff = X_gt - X
        baseline_mse_all = float(np.mean(baseline_diff ** 2))

        def _baseline_masked_mse(mask: np.ndarray) -> float:
            count = mask.sum()
            if count == 0:
                return float("nan")
            return float(np.mean((baseline_diff[mask]) ** 2))

        baseline_mse_nonzero = _baseline_masked_mse(mask_nonzero)
        baseline_mse_biozero = _baseline_masked_mse(mask_biozero)
        baseline_mse_dropout = _baseline_masked_mse(mask_dropout)

        return {
            "dataset": dataset_name,
            "layer": layer_type,
            "test_mse": mse_all,
            "mse_nonzero": mse_nonzero,
            "mse_biozero": mse_biozero,
            "mse_dropout": mse_dropout,
            "baseline_mse": baseline_mse_all,
            "baseline_mse_nonzero": baseline_mse_nonzero,
            "baseline_mse_biozero": baseline_mse_biozero,
            "baseline_mse_dropout": baseline_mse_dropout,
        }
    else:  # CLUST
        if labels is None:
            raise RuntimeError("CLUST mode requires labels (could not load from colData).")
        # Matches your utils.evaluation.evaluate_clustering. :contentReference[oaicite:3]{index=3}
        df = evaluate_clustering(recon_all, labels)
        row = df.iloc[0].to_dict()
        # Ensure plain floats
        return {"dataset": dataset_name, "layer": layer_type,
                "ASW": float(row.get("ASW", np.nan)),
                "ARI": float(row.get("ARI", np.nan)),
                "NMI": float(row.get("NMI", np.nan)),
                "PS":  float(row.get("PS", np.nan))}


def _layer_options(selected: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """[(pretty_name, layer_value)] for selected DenseLayerPack options."""
    available = {
        "SILU":     DENSE_LAYER_CONST.SILU_LAYER,
        "LINEAR":   DENSE_LAYER_CONST.LINEAR_LAYER,
        # Add other layer types if desired:
        # "KAN":       DENSE_LAYER_CONST.KAN_LAYER,
        # "WAVELET":   DENSE_LAYER_CONST.WAVELET_KAN_LAYER,
        # "FOURIER":   DENSE_LAYER_CONST.FOURIER_KAN_LAYER,
        # "KAE":       DENSE_LAYER_CONST.KAE_LAYER,
    }
    if selected is None:
        selected = ["SILU"]
    layers: List[Tuple[str, str]] = []
    for name in selected:
        key = name.upper()
        if key not in available:
            raise ValueError(f"Unknown layer option '{name}'. Available: {', '.join(available.keys())}")
        layers.append((key, available[key]))
    return layers


def _discover_rds_files(data_dir: str) -> List[str]:
    p = Path(data_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Data dir not found or not a directory: {data_dir}")
    files = sorted(str(fp) for fp in p.rglob("*.rds"))
    if not files:
        raise FileNotFoundError(f"No .rds files found under: {data_dir}")
    return files


def main():
    p = argparse.ArgumentParser(
        description="Measure MSE or run CLUSTERING over all .rds in a directory with an Autoencoder; runs every DenseLayerPack layer, repeats each experiment, and averages results."
    )
    p.add_argument("data_dir", type=str, help="Directory containing .rds files (searched recursively).")
    p.add_argument("--mode", type=str, choices=["MSE", "CLUST"], default="MSE",
                   help="Choose evaluation mode: 'MSE' compares to logTrueCounts; 'CLUST' runs PCA→KMeans clustering metrics.")
    p.add_argument("--hidden", type=str, default="64",
                   help="Comma-separated encoder hidden sizes (decoder is symmetric).")
    p.add_argument("--bottleneck", type=int, default=32, help="Bottleneck width.")
    p.add_argument("--layers", type=str, default="SILU",
                   help='Comma-separated DenseLayerPack layer types to evaluate (default: "SILU"). '
                        'Choices include SILU, LINEAR.')
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42, help="Base random seed (repeat k uses seed+k).")
    p.add_argument("--repeats", type=int, default=5, help="Number of repeats per dataset×layer.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Per-gene input scaling on/off
    p.add_argument("--scale-input", type=str, choices=["on", "off"], default="on",
                   help='Apply robust per-gene scaling to [-1,1] before training; "off" uses raw logcounts throughout.')

    # Masked-denoising (uniform by zero/non-zero)
    p.add_argument("--masked-denoise", action="store_true",
                   help="Enable masked-denoising objective: corrupt inputs and compute reconstruction loss only on masked entries.")
    p.add_argument("--md-p-zero", type=float, default=0.0,
                   help="Mask probability for entries that are zero in original logcounts.")
    p.add_argument("--md-p-nonzero", type=float, default=0.30,
                   help="Mask probability for entries that are non-zero in original logcounts.")
    p.add_argument("--md-fill", type=str, default="zero",
                   choices=["mean", "zero", "noise"],
                   help='Mask-fill strategy (no weighted/w_zero option).')
    p.add_argument("--md-noise-std", type=float, default=0.3,
                   help="Std of Gaussian noise when --md-fill noise is selected.")

    args = p.parse_args()
    device = torch.device(args.device)

    print("=== Settings ===")
    print(f" Mode     : {args.mode}")
    print(f" Layers   : {args.layers}")
    print(f" Model    : hidden={args.hidden}  bottleneck={args.bottleneck}")
    print(f" Train    : epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}  wd={args.weight_decay}")
    print(f" Repeats  : {args.repeats} (seed base={args.seed})")
    print(f" Device   : {args.device}")
    print(f" Scale    : {args.scale_input}")
    if args.masked_denoise:
        print(f" MaskDeno : ON  p_zero={args.md_p_zero}  p_nonzero={args.md_p_nonzero}  fill={args.md_fill}  noise_std={args.md_noise_std}")
    else:
        print(" MaskDeno : OFF")
    print("================")

    rds_files = _discover_rds_files(args.data_dir)
    layer_names = [s.strip() for s in args.layers.split(",") if s.strip()]
    if not layer_names:
        layer_names = ["SILU"]
    layers = _layer_options(layer_names)

    # Collect per-layer aggregated (over datasets) results
    results_by_layer: Dict[str, List[Dict[str, float]]] = {name: [] for name, _ in layers}
    dataset_rows_mse: List[Dict[str, float]] = []
    dataset_rows_clust: List[Dict[str, float]] = []

    for rds_path in rds_files:
        ds_name = Path(rds_path).stem
        need_labels = (args.mode == "CLUST")
        need_truth = (args.mode == "MSE")
        X, X_gt, labels = load_dataset(rds_path, need_labels=need_labels, need_truth=need_truth)
        X = np.asarray(X, dtype=np.float32)

        if args.mode == "MSE" and X_gt is None:
            print(f"[WARN] Dataset '{ds_name}' lacks 'logTrueCounts'; skipping in MSE mode.")
            continue

        dataset_rows: List[Dict[str, float]] = []
        for layer_name, layer_value in layers:
            rep_records: List[Dict[str, float]] = []

            for rep in range(args.repeats):
                # different seed per repeat
                seed = args.seed + rep
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                res = train_and_eval_single(X, X_gt, labels, args, device, ds_name, layer_value)
                rep_records.append(res)

            # Aggregate over repeats for this dataset×layer
            if args.mode == "MSE":
                mse_keys = ["test_mse", "mse_nonzero", "mse_biozero", "mse_dropout"]
                baseline_keys = ["baseline_mse", "baseline_mse_nonzero", "baseline_mse_biozero", "baseline_mse_dropout"]
                all_keys = mse_keys + baseline_keys
                means = {k: float(np.nanmean([r[k] for r in rep_records])) for k in mse_keys}
                base_means = {k: float(np.nanmean([r[k] for r in rep_records])) for k in baseline_keys}
                stds = {
                    f"{k}_std": float(np.nanstd([r[k] for r in rep_records], ddof=1)) if args.repeats > 1 else 0.0
                    for k in all_keys
                }
                res_avg = {"dataset": ds_name, "layer": layer_name, **means, **base_means, **stds}
                results_by_layer[layer_name].append(res_avg)
                dataset_rows_mse.append(res_avg)
            else:
                # Average each clustering metric
                keys = ["ASW", "ARI", "NMI", "PS"]
                mean_std = {k: float(np.mean([r[k] for r in rep_records])) for k in keys}
                stds = {f"{k}_std": float(np.std([r[k] for r in rep_records], ddof=1)) if args.repeats > 1 else 0.0
                        for k in keys}
                res_avg = {"dataset": ds_name, "layer": layer_name, **mean_std, **stds}
                results_by_layer[layer_name].append(res_avg)
                dataset_rows_clust.append(res_avg)

    # --- Averages across datasets (using per-dataset repeat-averaged stats) ---
    if args.mode == "MSE" and dataset_rows_mse:
        def _fmt3(x: float) -> str:
            return "nan" if np.isnan(x) else f"{x:.3f}"

        print("\n=== Per-dataset MSE (averaged over repeats) ===")
        header = (
            f"{'Dataset':<32} "
            f"{'MSE':>8} {'NonZero':>8} {'BioZero':>8} {'Dropout':>8}  "
            f"{'BASE_MSE':>9} {'BASE_NZ':>8} {'BASE_BZ':>8} {'BASE_DO':>8}"
        )
        print(header)
        for row in dataset_rows_mse:
            line = (
                f"{row['dataset']:<32}"
                f"{_fmt3(row['test_mse']):>8} {_fmt3(row['mse_nonzero']):>8} {_fmt3(row['mse_biozero']):>8} {_fmt3(row['mse_dropout']):>8}  "
                f"{_fmt3(row['baseline_mse']):>9} {_fmt3(row['baseline_mse_nonzero']):>8} {_fmt3(row['baseline_mse_biozero']):>8} {_fmt3(row['baseline_mse_dropout']):>8}"
            )
            print(line)

        print("\n=== Averages across datasets (per layer) ===")
        header_avg = (
            f"{'avg_MSE':>9} {'avg_NZ':>8} {'avg_BZ':>8} {'avg_DO':>8}  "
            f"{'avg_BASE_MSE':>12} {'avg_BASE_NZ':>11} {'avg_BASE_BZ':>11} {'avg_BASE_DO':>11}"
        )
        print(header_avg)
        if results_by_layer:
            mse_keys = ["test_mse", "mse_nonzero", "mse_biozero", "mse_dropout"]
            baseline_keys = ["baseline_mse", "baseline_mse_nonzero", "baseline_mse_biozero", "baseline_mse_dropout"]
            agg_means = {k: float(np.nanmean([r[k] for layer_res in results_by_layer.values() for r in layer_res])) for k in mse_keys + baseline_keys}
            line = (
                f"{_fmt3(agg_means['test_mse']):>9} {_fmt3(agg_means['mse_nonzero']):>8} {_fmt3(agg_means['mse_biozero']):>8} {_fmt3(agg_means['mse_dropout']):>8}  "
                f"{_fmt3(agg_means['baseline_mse']):>12} {_fmt3(agg_means['baseline_mse_nonzero']):>11} {_fmt3(agg_means['baseline_mse_biozero']):>11} {_fmt3(agg_means['baseline_mse_dropout']):>11}"
            )
            print(line)

    if args.mode == "CLUST" and dataset_rows_clust:
        def _fmt3c(x: float) -> str:
            return "nan" if np.isnan(x) else f"{x:.3f}"

        print("\n=== Per-dataset CLUST (averaged over repeats) ===")
        header = f"{'Dataset':<32} {'ASW':>8} {'ARI':>8} {'NMI':>8} {'PS':>8}"
        print(header)
        for row in dataset_rows_clust:
            line = (
                f"{row['dataset']:<32}"
                f"{_fmt3c(row['ASW']):>8} {_fmt3c(row['ARI']):>8} {_fmt3c(row['NMI']):>8} {_fmt3c(row['PS']):>8}"
            )
            print(line)

        print("\n=== Averages across datasets (per layer) ===")
        header_avg = f"{'avg_ASW':>9} {'avg_ARI':>9} {'avg_NMI':>9} {'avg_PS':>9}"
        print(header_avg)
        if results_by_layer:
            keys = ["ASW", "ARI", "NMI", "PS"]
            agg_means = {k: float(np.mean([r[k] for layer_res in results_by_layer.values() for r in layer_res])) for k in keys}
            line = f"{_fmt3c(agg_means['ASW']):>9} {_fmt3c(agg_means['ARI']):>9} {_fmt3c(agg_means['NMI']):>9} {_fmt3c(agg_means['PS']):>9}"
            print(line)


if __name__ == "__main__":
    main()
