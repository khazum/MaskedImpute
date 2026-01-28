# run_from_rds_oracle_biozeros_simplified.py
# Autoencoder on .rds datasets for MSE (incl. zero-aware metrics) or clustering metrics.

import argparse
from typing import Tuple, List, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from rds2py import read_rds  # must expose read_rds(path) -> SCE-like object

# ---- Clustering metrics (prefer repo function; else reproducible fallback)
try:
    from utils.evaluation import evaluate_clustering  # ASW, ARI, NMI, PS (Purity)
except Exception:  # lightweight, reproducible fallback
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.metrics.cluster import contingency_matrix
    import pandas as pd

    def _purity_score(y_true, y_pred) -> float:
        m = contingency_matrix(y_true, y_pred)
        return float(np.sum(np.max(m, axis=0)) / np.sum(m))

    def evaluate_clustering(imputed_data: np.ndarray, true_labels: np.ndarray) -> "pd.DataFrame":
        n, d = imputed_data.shape
        # Safe for very small n/d
        n_components = max(1, min(50, n, d))
        emb = PCA(n_components=n_components).fit_transform(np.nan_to_num(imputed_data))
        k = max(2, len(np.unique(true_labels)))
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(emb)
        cl = km.labels_
        asw = silhouette_score(emb, cl) if 2 <= len(np.unique(cl)) < n else np.nan
        ari = adjusted_rand_score(true_labels, cl)
        nmi = normalized_mutual_info_score(true_labels, cl)
        ps = _purity_score(true_labels, cl)
        return pd.DataFrame({"ASW": [round(asw, 4)], "ARI": [round(ari, 4)], "NMI": [round(nmi, 4)], "PS": [round(ps, 4)]})


# ---- Scalers
class RobustZThenMinMaxToNeg1Pos1:
    def __init__(self, p_low: float = 0.0, p_high: float = 99.0, eps: float = 1e-8):
        assert 0.0 <= p_low < p_high <= 100.0
        self.p_low, self.p_high, self.eps = p_low, p_high, eps
        self.lo_ = self.hi_ = self.mean_ = self.std_ = self.zmin_ = self.zmax_ = self.zspan_ = None

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
        return (X01 * 2.0 - 1.0).astype(np.float32)

    def inverse_transform(self, Xscaled: np.ndarray) -> np.ndarray:
        X01 = (Xscaled + 1.0) / 2.0
        Z = X01 * self.zspan_ + self.zmin_
        return (Z * self.std_ + self.mean_).astype(np.float32)


class IdentityScaler:
    def fit(self, X: np.ndarray): return self
    def transform(self, X: np.ndarray) -> np.ndarray: return X.astype(np.float32)
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: return X.astype(np.float32)


# ---- .rds loading
def load_dataset(path: str, need_labels: bool, need_perfect: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    sce = read_rds(path)
    logcounts = sce.assay("logcounts").T.astype("float32")
    keep = np.sum(logcounts != 0, axis=0) >= 2
    logcounts = logcounts[:, keep]

    perfect_logcounts = None
    if need_perfect:
        try:
            perfect_logcounts = sce.assay("logTrueCounts").T[:, keep].astype("float32")
        except Exception:
            perfect_logcounts = None

    labels = None
    if need_labels:
        colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
        y = None
        if colmd is not None:
            if hasattr(colmd, "get_column_names") and hasattr(colmd, "get_column"):
                colnames = list(map(str, colmd.get_column_names()))
                for key in ("cell_type1", "labels", "Group", "label"):
                    if key in colnames: y = np.asarray(colmd.get_column(key)); break
            elif hasattr(colmd, "columns"):
                colnames = list(map(str, getattr(colmd, "columns", [])))
                for key in ("cell_type1", "labels", "Group", "label"):
                    if key in colnames: y = np.asarray(colmd[key]); break
        if y is None:
            y = np.zeros(logcounts.shape[0], dtype=int)
        import pandas as pd
        labels, _ = pd.factorize(np.asarray(y))
        labels = labels.astype(int)

    return logcounts, perfect_logcounts, labels


# ---- Simple MLP AE (SiLU activations)
def _build_mlp(sizes: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class AE(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], bottleneck: int):
        super().__init__()
        self.encoder = _build_mlp([input_dim] + list(hidden) + [bottleneck])
        self.decoder = _build_mlp([bottleneck] + list(reversed(hidden)) + [input_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def mse_from_residual(residual: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        residual = residual * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(residual.numel(), device=residual.device, dtype=residual.dtype)
    return residual.pow(2).sum() / denom


# ---- Splatter-guided bio-zero estimator (for non-oracle modes)
def _to_norm_counts_from_logcounts(X_log: np.ndarray, base: float) -> np.ndarray:
    base = float(base) if base and base > 0 else 2.0
    C = np.power(base, X_log, dtype=np.float64) - 1.0
    C[C < 0] = 0.0
    return C

def _fit_splatter_dropout_from_logcounts(X_log: np.ndarray, zero_thr: float, log_base: float, eps: float = 1e-8):
    Z = (X_log <= zero_thr)
    C = _to_norm_counts_from_logcounts(X_log, log_base)
    mu = C.mean(axis=0)
    x = np.log(mu + eps)
    y = Z.mean(axis=0).clip(1e-6, 1.0 - 1e-6)
    logit_y = np.log(y / (1 - y))
    mask = np.isfinite(x) & np.isfinite(logit_y)
    if mask.sum() < 2:
        a, b = -1.0, 0.0
    else:
        a, b = np.polyfit(x[mask], logit_y[mask], 1)
        if a > 0: a, b = -abs(a), -b
    k = float(a)
    x0 = float(-b / a) if abs(a) > 1e-12 else (float(np.median(x[mask])) if mask.any() else 0.0)
    p_drop = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    return k, x0, p_drop.astype(np.float64), mu.astype(np.float64)

def _nb_zero_prob(mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    phi_b = np.broadcast_to(phi, mu.shape) if phi.shape != mu.shape else phi
    p0 = np.empty_like(mu, dtype=np.float64)
    small = (phi_b <= 1e-12)
    p0[small] = np.exp(-mu[small])
    p0[~small] = np.power(1.0 + mu[~small] * phi_b[~small], -1.0 / phi_b[~small])
    return p0

def _estimate_phi_moments(C: np.ndarray, mu: np.ndarray) -> np.ndarray:
    var = C.var(axis=0, ddof=1).astype(np.float64)
    phi = (var - mu) / np.maximum(mu * mu, 1e-12)
    phi[phi < 0] = 0.0
    return phi

def estimate_biozero_posterior_matrix(
    X_log: np.ndarray, zero_thr: float, log_base: float,
    disp_mode: str = "estimate", disp_const: float = 0.1, use_cell_factor: bool = False,
) -> np.ndarray:
    _, _, p_drop_g, mu_g = _fit_splatter_dropout_from_logcounts(X_log, zero_thr, log_base)
    C = _to_norm_counts_from_logcounts(X_log, log_base)
    phi_g = (np.full(mu_g.shape, float(disp_const), dtype=np.float64)
             if disp_mode == "fixed" else _estimate_phi_moments(C, mu_g))
    if use_cell_factor:
        s_i = C.mean(axis=1).astype(np.float64); s_i /= (s_i.mean() + 1e-12)
        mu_ij = mu_g[None, :] * s_i[:, None]
        p0_nb_ij = _nb_zero_prob(mu_ij, phi_g[None, :])
        p_drop_ij = np.broadcast_to(p_drop_g[None, :], X_log.shape).astype(np.float64)
        p_zero_obs = p_drop_ij + (1.0 - p_drop_ij) * p0_nb_ij
        p_bio_ij = ((1.0 - p_drop_ij) * p0_nb_ij) / np.maximum(p_zero_obs, 1e-12)
    else:
        p0_nb_g = _nb_zero_prob(mu_g, phi_g)
        p_zero_obs_g = p_drop_g + (1.0 - p_drop_g) * p0_nb_g
        p_bio_g = ((1.0 - p_drop_g) * p0_nb_g) / np.maximum(p_zero_obs_g, 1e-12)
        p_bio_ij = np.broadcast_to(p_bio_g[None, :], X_log.shape).astype(np.float64)

    bio_post = np.zeros_like(X_log, dtype=np.float32)
    is_zero = (X_log <= zero_thr)
    bio_post[is_zero] = p_bio_ij[is_zero].astype(np.float32)
    return bio_post


# ---- Helpers
def _resolve_fill_modes(args):
    base, z, nz = args.md_fill, args.md_fill_zeroes, args.md_fill_nonzero
    return (base if z is None else z, base if nz is None else nz)

def _make_fill_tensor(mode: str, xb: torch.Tensor, feat_mean_t: torch.Tensor, noise_std: float) -> torch.Tensor:
    if mode == "mean":  return feat_mean_t.expand_as(xb)
    if mode == "zero":  return torch.zeros_like(xb)
    if mode == "noise": return torch.normal(mean=0.0, std=noise_std, size=xb.shape, device=xb.device)
    raise ValueError(f"Unknown md-fill mode '{mode}' (expected one of: mean, zero, noise)")

def _mse_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    n = int(mask.sum())
    if n == 0: return float("nan")
    d = (a - b)[mask]
    return float(np.mean(d * d))

def _frac(mask_true: np.ndarray, mask_cond: np.ndarray) -> float:
    n = int(mask_true.sum())
    if n == 0: return float("nan")
    return float(100.0 * (mask_true & mask_cond).sum() / n)


# ---- Train & evaluate single dataset
def train_and_eval_single(
    X: np.ndarray, X_gt: Optional[np.ndarray], labels: Optional[np.ndarray], args, device: torch.device, dataset_name: str
) -> Dict[str, float]:
    scaler = (RobustZThenMinMaxToNeg1Pos1().fit(X) if args.scale_input == "on" else IdentityScaler().fit(X))
    Xs = scaler.transform(X).astype(np.float32)
    feat_mean_for_fill = Xs.mean(axis=0).astype(np.float32)

    thr = float(args.zero_threshold)
    zeros_total = (X <= thr).astype(np.float32)

    if args.biozero_mode == "oracle":
        if X_gt is None:
            raise RuntimeError("--biozero-mode oracle requires 'perfect_logcounts'.")
        likely_bio_zero = (zeros_total * (X_gt <= thr)).astype(np.float32)
        print("[BioZero] ORACLE: using perfect_logcounts.")
    else:
        bio_post = estimate_biozero_posterior_matrix(
            X_log=X, zero_thr=thr, log_base=float(args.log_base),
            disp_mode=args.biozero_disp, disp_const=float(args.biozero_disp_const),
            use_cell_factor=bool(args.biozero_cell_factor),
        )
        if args.biozero_mode == "off":
            likely_bio_zero = zeros_total
        elif args.biozero_mode == "soft":
            likely_bio_zero = (zeros_total * (np.random.rand(*bio_post.shape) < bio_post)).astype(np.float32)
        else:  # hard
            likely_bio_zero = (zeros_total * (bio_post >= float(args.biozero_thresh))).astype(np.float32)

    Xtr = torch.tensor(Xs, dtype=torch.float32)
    Zbio = torch.tensor(likely_bio_zero, dtype=torch.float32)
    Zall = torch.tensor(zeros_total, dtype=torch.float32)
    tr_loader = DataLoader(TensorDataset(Xtr, Zbio, Zall), batch_size=args.batch_size, shuffle=True, drop_last=False)

    print(f"Dataset '{dataset_name}': {X.shape[0]} cells × {X.shape[1]} genes.")
    print("Scaled to [-1,1]." if args.scale_input == "on" else "[Input scaling OFF] raw logcounts to model.")

    kept, allz = float(likely_bio_zero.sum()), float(zeros_total.sum())
    pct = (100.0 * kept / max(allz, 1.0))
    print(f"[BioZero] {kept:.0f}/{allz:.0f} zeros ({pct:.1f}%) flagged as biological (mode={args.biozero_mode}).")

    hidden = [int(h) for h in args.hidden.split(",")] if args.hidden else []
    model = AE(input_dim=X.shape[1], hidden=hidden, bottleneck=int(args.bottleneck)).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    feat_mean_t = torch.tensor(feat_mean_for_fill, device=device, dtype=torch.float32).unsqueeze(0)
    fill_bio_mode, fill_non_mode = _resolve_fill_modes(args)

    model.train()
    for _ in range(args.epochs):
        for xb, zb_bio, z_all in tr_loader:
            xb, zb_bio, z_all = xb.to(device), zb_bio.to(device), z_all.to(device)

            if args.masked_denoise:
                probs = torch.full_like(xb, args.md_p_nonzero)
                is_dropout_zero = (z_all > 0.5) & (zb_bio <= 0.5)
                probs = torch.where(is_dropout_zero, torch.zeros_like(probs), probs)
                probs = torch.where(zb_bio > 0.5, torch.full_like(probs, args.md_p_zero), probs)
                mask = torch.bernoulli(probs).to(xb.dtype)

                fill_bio = _make_fill_tensor(fill_bio_mode, xb, feat_mean_t, args.md_noise_std)
                fill_non = _make_fill_tensor(fill_non_mode, xb, feat_mean_t, args.md_noise_std)
                fill = torch.where(zb_bio > 0.5, fill_bio, fill_non)

                x_in  = (1.0 - mask) * xb + mask * fill
                x_tgt = xb
            else:
                mask = None
                x_in = x_tgt = xb

            opt.zero_grad()
            residual = model(x_in) - x_tgt
            mse_from_residual(residual, mask=mask).backward()
            opt.step()

    model.eval()
    recon_list = []
    with torch.no_grad():
        for i in range(0, Xtr.size(0), args.batch_size):
            xb = Xtr[i : i + args.batch_size].to(device)
            recon_np = model(xb).cpu().numpy()
            recon_list.append(scaler.inverse_transform(recon_np))
    recon_all = np.vstack(recon_list)  # reconstruction in logcounts space

    if args.mode == "MSE":
        if X_gt is None:
            raise RuntimeError("MSE mode requires 'perfect_logcounts' in the dataset.")
        # Masks
        gt_bio = (X_gt <= thr)            # GT biological zeros
        gt_non = ~gt_bio                  # GT nonzeros
        obs_zero = (X <= thr)             # observed zeros (input)
        obs_non = ~obs_zero               # observed nonzeros (input)
        dropout_mask = obs_zero & gt_non  # dropouts (obs zero but GT nonzero)
        recon_zero = (recon_all <= thr)   # zeros in reconstruction

        # Imputed MSEs: (X_gt - recon_all)^2
        MSE_total_imputed            = float(np.mean((X_gt - recon_all) ** 2))
        MSE_nonzero_gt_imputed       = _mse_mask(X_gt, recon_all, gt_non)
        MSE_nonzero_input_imputed    = _mse_mask(X_gt, recon_all, obs_non)
        MSE_biological_zero_imputed  = _mse_mask(X_gt, recon_all, gt_bio)
        MSE_dropout_imputed          = _mse_mask(X_gt, recon_all, dropout_mask)

        # Raw MSEs: (X - X_gt)^2
        MSE_total_raw            = float(np.mean((X_gt - X) ** 2))
        MSE_nonzero_gt_raw       = _mse_mask(X_gt, X, gt_non)
        MSE_nonzero_input_raw    = _mse_mask(X_gt, X, obs_non)
        MSE_biological_zero_raw  = _mse_mask(X_gt, X, gt_bio)
        MSE_dropout_raw          = _mse_mask(X_gt, X, dropout_mask)

        # Percentages
        bio_preserved_imputed   = _frac(gt_bio, recon_zero)   # % GT zeros reconstructed as zero
        dropout_mod_imputed     = _frac(gt_non, ~recon_zero)  # % GT nonzeros reconstructed as nonzero
        bio_preserved_raw       = _frac(gt_bio, obs_zero)     # % GT zeros present in observed
        dropout_mod_raw         = _frac(gt_non, ~obs_zero)    # % GT nonzeros present in observed

        return {
            "dataset": dataset_name,
            "MSE_total_imputed": MSE_total_imputed,
            "MSE_nonzero_gt_imputed": MSE_nonzero_gt_imputed,
            "MSE_nonzero_input_imputed": MSE_nonzero_input_imputed,
            "MSE_biological_zero_imputed": MSE_biological_zero_imputed,
            "MSE_dropout_imputed": MSE_dropout_imputed,
            "biological_zero_preserved_imputed": bio_preserved_imputed,
            "dropout_zero_modified_imputed": dropout_mod_imputed,
            "MSE_total_raw": MSE_total_raw,
            "MSE_nonzero_gt_raw": MSE_nonzero_gt_raw,
            "MSE_nonzero_input_raw": MSE_nonzero_input_raw,
            "MSE_biological_zero_raw": MSE_biological_zero_raw,
            "MSE_dropout_raw": MSE_dropout_raw,
            "biological_zero_preserved_raw": bio_preserved_raw,
            "dropout_zero_modified_raw": dropout_mod_raw,
        }
    else:  # CLUST
        if labels is None:
            raise RuntimeError("CLUST mode requires labels.")
        df = evaluate_clustering(recon_all, labels)
        row = df.iloc[0].to_dict()
        return {k: float(row.get(k, np.nan)) for k in ("ASW", "ARI", "NMI", "PS")} | {"dataset": dataset_name}


# ---- File discovery
def _discover_rds_files(data_dir: str) -> List[str]:
    p = Path(data_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Data dir not found or not a directory: {data_dir}")
    files = sorted(str(fp) for fp in p.rglob("*.rds"))
    if not files:
        raise FileNotFoundError(f"No .rds files found under: {data_dir}")
    return files


# ---- Main
def main():
    p = argparse.ArgumentParser(
        description="Autoencoder over .rds: MSE (imputed vs GT AND raw vs GT) or CLUST. SiLU MLP; masked-denoise optional."
    )
    p.add_argument("data_dir", type=str, help="Directory containing .rds files (searched recursively).")
    p.add_argument("--mode", type=str, choices=["MSE", "CLUST"], default="MSE")
    p.add_argument("--hidden", type=str, default="64", help="Comma-separated encoder hidden sizes; decoder symmetric.")
    p.add_argument("--bottleneck", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42, help="Base random seed (repeat k uses seed+k).")
    p.add_argument("--repeats", type=int, default=5, help="Number of repeats per dataset.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--scale-input", type=str, choices=["on", "off"], default="on",
                   help='Robust per-gene scaling to [-1,1]; "off" uses raw logcounts.')

    p.add_argument("--masked-denoise", action="store_true",
                   help="Corrupt inputs; compute reconstruction loss only on masked entries.")
    p.add_argument("--md-p-zero", type=float, default=0.10, dest="md_p_zero")
    p.add_argument("--md-p-nonzero", type=float, default=0.30, dest="md_p_nonzero")
    p.add_argument("--md-fill", type=str, default="mean", choices=["mean", "zero", "noise"])
    p.add_argument("--md-fill-zeroes", type=str, default=None)
    p.add_argument("--md-fill-nonzero", type=str, default=None)
    p.add_argument("--md-noise-std", type=float, default=0.3)

    p.add_argument("--zero-threshold", type=float, default=1e-4, help="Logcounts ≤ thr treated as zero.")

    p.add_argument("--log-base", type=float, default=2.0)
    p.add_argument("--biozero-mode", type=str, choices=["off", "hard", "soft", "oracle"], default="hard")
    p.add_argument("--biozero-thresh", type=float, default=0.5)
    p.add_argument("--biozero-disp", type=str, choices=["fixed", "estimate"], default="estimate", dest="biozero_disp")
    p.add_argument("--biozero-disp-const", type=float, default=0.1, dest="biozero_disp_const")
    p.add_argument("--biozero-cell-factor", action="store_true")

    args = p.parse_args()

    valid_fills = {"mean", "zero", "noise"}
    if args.md_fill_zeroes is not None and args.md_fill_zeroes not in valid_fills:
        raise ValueError("--md-fill-zeroes must be one of {mean, zero, noise}")
    if args.md_fill_nonzero is not None and args.md_fill_nonzero not in valid_fills:
        raise ValueError("--md-fill-nonzero must be one of {mean, zero, noise}")

    device = torch.device(args.device)
    rds_files = _discover_rds_files(args.data_dir)

    all_results = []
    for rds_path in rds_files:
        ds_name = Path(rds_path).stem
        need_labels = (args.mode == "CLUST")
        need_perfect = (args.mode == "MSE") or (args.biozero_mode == "oracle")
        X, X_gt, labels = load_dataset(rds_path, need_labels=need_labels, need_perfect=need_perfect)
        X = np.asarray(X, dtype=np.float32)

        if args.biozero_mode == "oracle" and X_gt is None:
            print(f"[ERROR] '{ds_name}' lacks perfect_logcounts; skipping (oracle mode requested).")
            continue
        if args.mode == "MSE" and X_gt is None:
            print(f"[WARN] '{ds_name}' lacks perfect_logcounts; skipping in MSE mode.")
            continue

        rep_records: List[Dict[str, float]] = []
        for rep in range(args.repeats):
            seed = args.seed + rep
            torch.manual_seed(seed); np.random.seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

            res = train_and_eval_single(X, X_gt, labels, args, device, ds_name)
            rep_records.append(res)

            if args.mode == "MSE":
                def _fmt(v, p=6, pct=False):
                    if v is None or not np.isfinite(v): return "NaN"
                    return (f"{v:.{p}f}%" if pct else f"{v:.{p}f}")
                print(
                    f"{ds_name} (rep {rep+1}/{args.repeats}): "
                    f"[imputed vs GT] total={_fmt(res['MSE_total_imputed'])}, nonzero_gt={_fmt(res['MSE_nonzero_gt_imputed'])}, "
                    f"nonzero_input={_fmt(res['MSE_nonzero_input_imputed'])}, bio={_fmt(res['MSE_biological_zero_imputed'])}, "
                    f"dropout={_fmt(res['MSE_dropout_imputed'])};  "
                    f"bio_preserved={_fmt(res['biological_zero_preserved_imputed'],2,True)}, "
                    f"dropout_modified={_fmt(res['dropout_zero_modified_imputed'],2,True)}  ||  "
                    f"[raw vs GT] total={_fmt(res['MSE_total_raw'])}, nonzero_gt={_fmt(res['MSE_nonzero_gt_raw'])}, "
                    f"nonzero_input={_fmt(res['MSE_nonzero_input_raw'])}, bio={_fmt(res['MSE_biological_zero_raw'])}, "
                    f"dropout={_fmt(res['MSE_dropout_raw'])};  "
                    f"bio_preserved_raw={_fmt(res['biological_zero_preserved_raw'],2,True)}, "
                    f"dropout_modified_raw={_fmt(res['dropout_zero_modified_raw'],2,True)}"
                )
            else:
                print(f"{ds_name} (rep {rep+1}/{args.repeats}): "
                      f"ASW={res['ASW']:.4f} ARI={res['ARI']:.4f} NMI={res['NMI']:.4f} PS={res['PS']:.4f}")

        # Aggregate per dataset
        if args.mode == "MSE":
            keys = [
                "MSE_total_imputed","MSE_nonzero_gt_imputed","MSE_nonzero_input_imputed",
                "MSE_biological_zero_imputed","MSE_dropout_imputed",
                "biological_zero_preserved_imputed","dropout_zero_modified_imputed",
                "MSE_total_raw","MSE_nonzero_gt_raw","MSE_nonzero_input_raw",
                "MSE_biological_zero_raw","MSE_dropout_raw",
                "biological_zero_preserved_raw","dropout_zero_modified_raw",
            ]
            means = {k: float(np.nanmean([r[k] for r in rep_records])) for k in keys}
            stds  = {f"{k}_std": float(np.nanstd([r[k] for r in rep_records], ddof=1)) if args.repeats > 1 else 0.0 for k in keys}
            agg = {"dataset": ds_name, **means, **stds}
            all_results.append(agg)

            def _fmtm(k, pct=False, p=6):
                v, s = agg[k], agg[f"{k}_std"]
                if not np.isfinite(v): return "NaN (±NaN)"
                suf = "%" if pct else ""
                return f"{v:.{2 if pct else p}f}{suf} (±{s:.{2 if pct else p}f}{suf})"

            print(
                f"{ds_name} AVG over {args.repeats} reps:\n"
                f"  [imputed vs GT]  "
                f"total={_fmtm('MSE_total_imputed')}  nonzero_gt={_fmtm('MSE_nonzero_gt_imputed')}  "
                f"nonzero_input={_fmtm('MSE_nonzero_input_imputed')}  bio={_fmtm('MSE_biological_zero_imputed')}  "
                f"dropout={_fmtm('MSE_dropout_imputed')}  bio_preserved={_fmtm('biological_zero_preserved_imputed', pct=True)}  "
                f"dropout_modified={_fmtm('dropout_zero_modified_imputed', pct=True)}\n"
                f"  [raw vs GT]      "
                f"total={_fmtm('MSE_total_raw')}  nonzero_gt={_fmtm('MSE_nonzero_gt_raw')}  "
                f"nonzero_input={_fmtm('MSE_nonzero_input_raw')}  bio={_fmtm('MSE_biological_zero_raw')}  "
                f"dropout={_fmtm('MSE_dropout_raw')}  bio_preserved={_fmtm('biological_zero_preserved_raw', pct=True)}  "
                f"dropout_modified={_fmtm('dropout_zero_modified_raw', pct=True)}"
            )
        else:
            keys = ["ASW", "ARI", "NMI", "PS"]
            means = {k: float(np.mean([r[k] for r in rep_records])) for k in keys}
            stds  = {f"{k}_std": float(np.std([r[k] for r in rep_records], ddof=1)) if args.repeats > 1 else 0.0 for k in keys}
            all_results.append({"dataset": ds_name, **means, **stds})
            print(f"{ds_name} AVG over {args.repeats} reps: " + " ".join([f"{k}={means[k]:.4f} (±{stds[k+'_std']:.4f})" for k in keys]))

    # ---- Averages across datasets (exact names requested)
    print("\n=== Averages across datasets ===")
    if args.mode == "MSE" and all_results:
        def _avg(key): return float(np.nanmean([r[key] for r in all_results]))
        # Raw vs GT
        avg_MSE_total_raw                 = _avg("MSE_total_raw")
        avg_MSE_nonzero_gt_raw            = _avg("MSE_nonzero_gt_raw")
        avg_MSE_nonzero_input_raw         = _avg("MSE_nonzero_input_raw")
        avg_MSE_biological_zero_raw       = _avg("MSE_biological_zero_raw")
        avg_MSE_dropout_raw               = _avg("MSE_dropout_raw")
        avg_biological_zero_preserved_raw = _avg("biological_zero_preserved_raw")
        avg_dropout_zero_modified_raw     = _avg("dropout_zero_modified_raw")
        # Imputed vs GT
        avg_MSE_total_imputed                 = _avg("MSE_total_imputed")
        avg_MSE_nonzero_gt_imputed            = _avg("MSE_nonzero_gt_imputed")
        avg_MSE_nonzero_input_imputed         = _avg("MSE_nonzero_input_imputed")
        avg_MSE_biological_zero_imputed       = _avg("MSE_biological_zero_imputed")
        avg_MSE_dropout_imputed               = _avg("MSE_dropout_imputed")
        avg_biological_zero_preserved_imputed = _avg("biological_zero_preserved_imputed")
        avg_dropout_zero_modified_imputed     = _avg("dropout_zero_modified_imputed")

        n = len(all_results)
        print(
            f"avg_MSE_total_raw={avg_MSE_total_raw:.6f}  "
            f"avg_MSE_nonzero_gt_raw={avg_MSE_nonzero_gt_raw:.6f}  "
            f"avg_MSE_nonzero_input_raw={avg_MSE_nonzero_input_raw:.6f}  "
            f"avg_MSE_biological_zero_raw={avg_MSE_biological_zero_raw:.6f}  "
            f"avg_MSE_dropout_raw={avg_MSE_dropout_raw:.6f}  "
            f"avg_biological_zero_preserved_raw={(f'{avg_biological_zero_preserved_raw:.2f}%' if np.isfinite(avg_biological_zero_preserved_raw) else 'NaN')}  "
            f"avg_dropout_zero_modified_raw={(f'{avg_dropout_zero_modified_raw:.2f}%' if np.isfinite(avg_dropout_zero_modified_raw) else 'NaN')}"
        )
        print(
            f"avg_MSE_total_imputed={avg_MSE_total_imputed:.6f}  "
            f"avg_MSE_nonzero_gt_imputed={avg_MSE_nonzero_gt_imputed:.6f}  "
            f"avg_MSE_nonzero_input_imputed={avg_MSE_nonzero_input_imputed:.6f}  "
            f"avg_MSE_biological_zero_imputed={avg_MSE_biological_zero_imputed:.6f}  "
            f"avg_MSE_dropout_imputed={avg_MSE_dropout_imputed:.6f}  "
            f"avg_biological_zero_preserved_imputed={(f'{avg_biological_zero_preserved_imputed:.2f}%' if np.isfinite(avg_biological_zero_preserved_imputed) else 'NaN')}  "
            f"avg_dropout_zero_modified_imputed={(f'{avg_dropout_zero_modified_imputed:.2f}%' if np.isfinite(avg_dropout_zero_modified_imputed) else 'NaN')}  "
            f"over {n} dataset(s)"
        )
    elif args.mode == "CLUST":
        keys = ["ASW", "ARI", "NMI", "PS"]
        n = len(all_results)
        if n:
            means = {k: float(np.mean([r[k] for r in all_results])) for k in keys}
            print(" ".join([f"avg_{k}={means[k]:.4f}" for k in keys]) + f" over {n} dataset(s)")

if __name__ == "__main__":
    main()
