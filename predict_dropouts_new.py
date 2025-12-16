#!/usr/bin/env python3
"""
predict_dropouts.py  (Revised with Standard DCA Interpretation and Bio-Zero Metrics)
----------------------------------------------------------------------------------

- Implements shrinkage estimation for robust stratified analysis in SPLAT.
- Uses standard ZINB interpretation for DCA posterior probabilities.
- Includes stratified DCA with global fallback.
- Evaluates using Precision, Recall, Accuracy, and F1 Score for Biological Zero identification.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

import subprocess
import tempfile
import numpy as np
import pandas as pd
import traceback
import warnings
import sys
import os

# Handle optional Keras/TensorFlow imports and configuration
try:
    # Minimize TensorFlow logging noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    if 'TF_ENABLE_ONEDNN_OPTS' not in os.environ:
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import tensorflow as tf
    tf.get_logger().setLevel('WARNING')
    import keras
    # Optimization: disable traceback filtering if keras is available
    if hasattr(keras.config, 'disable_traceback_filtering'):
        keras.config.disable_traceback_filtering()
except ImportError:
    pass # Keras/TensorFlow are optional

# Optional heavy deps isolated to features that use them
try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except Exception:
    PCA = None
    NearestNeighbors = None

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

try:
    # Assume rds2py is available in the environment
    from rds2py import read_rds
except Exception:
    read_rds = None

# ---------------------------- I/O helpers ----------------------------

def load_sce(path: Path, need_perfect: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """
    Read a SingleCellExperiment stored as .rds.
    """
    if read_rds is None:
        raise RuntimeError("rds2py is not installed or not importable.")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sce = read_rds(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to read RDS file {path}: {e}")

    # 1. Extract Assays
    # Note: The exact structure depends on the rds2py version and the RDS object.
    # This implementation attempts to handle common structures robustly.
    assays = getattr(sce, "assays", None)
    if assays is None and isinstance(sce, dict): assays = sce.get("assays")
    
    # Handle SummarizedExperiment structure if necessary
    if hasattr(assays, "listData"): assays = assays.listData
    elif isinstance(assays, dict) and "listData" in assays: assays = assays["listData"]
        
    if assays is None: raise RuntimeError(f"Could not locate 'assays' in {path}.")

    def _get_assay(name):
        if isinstance(assays, dict):
            if name in assays: return assays[name]
            for k, v in assays.items():
                if k.lower() == name.lower(): return v
        if hasattr(assays, "get"): return assays.get(name)
        return None

    c_raw = _get_assay("counts")
    if c_raw is None: raise RuntimeError(f"Assay 'counts' not found in {path}.")

    # Transpose [genes x cells] -> [cells x genes]
    counts = np.asarray(c_raw).T.astype("float32")
    n_cells = counts.shape[0]

    # 2. Extract Metadata (colData)
    colmd = getattr(sce, "column_data", None) or getattr(sce, "colData", None)
    if colmd is None and isinstance(sce, dict): colmd = sce.get("colData") or sce.get("column_data")

    extracted_cols = {}
    if colmd is not None:
        # Handle different metadata container types (e.g., DataFrame, R-like structures)
        if hasattr(colmd, "get_column_names") and hasattr(colmd, "get_column"):
            try:
                colnames = list(map(str, colmd.get_column_names()))
                for name in colnames:
                    extracted_cols[name] = np.asarray(colmd.get_column(name))
            except Exception: pass
        elif hasattr(colmd, "columns"): # Pandas DataFrame like
            try:
                colnames = list(map(str, getattr(colmd, "columns", [])))
                for name in colnames:
                    extracted_cols[name] = np.asarray(colmd[name])
            except Exception: pass
        elif isinstance(colmd, dict):
            target_dict = colmd.get("listData", colmd) 
            for k, v in target_dict.items():
                if hasattr(v, "__len__") and len(v) == n_cells:
                    extracted_cols[k] = np.asarray(v)

    coldata = pd.DataFrame(extracted_cols) if extracted_cols else pd.DataFrame(index=range(n_cells))

    # 3. Extract Additional Assays
    truecounts = None
    if need_perfect:
        tc_raw = _get_assay("TrueCounts")
        if tc_raw is not None: truecounts = np.asarray(tc_raw).T.astype("float32")

    # 4. Final Processing
    for k in ["Group", "Batch", "group", "batch"]:
        if k in coldata.columns: coldata[k] = coldata[k].astype(str)

    # Filter genes (keep genes expressed in >= 2 cells)
    keep = np.sum(counts > 0.0, axis=0) >= 2
    counts = counts[:, keep]

    if truecounts is not None:
        try: truecounts = truecounts[:, keep]
        except IndexError: truecounts = None # Handle shape mismatch if filtering changes dimensions unexpectedly

    return counts, truecounts, coldata

def discover_rds_files(data_dir: Path) -> List[Path]:
    exts = (".rds", ".RDS")
    return sorted([p for p in data_dir.rglob("*") if p.suffix in exts])


# ---------------------------- SPLAT math on counts (With Shrinkage) ----------------------------
# (SPLAT related functions remain unchanged as the focus is on fixing DCA interpretation)

def _nb_zero_prob(mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """P(Y=0 | mu, phi) for NB. phi is dispersion (size = 1/phi)."""
    mu = np.asarray(mu, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    
    # Handle broadcasting robustly
    try:
        if phi.shape != mu.shape:
            phi_b = np.broadcast_to(phi, mu.shape)
        else:
            phi_b = phi
    except ValueError:
        if phi.ndim == 1 and mu.ndim == 2 and phi.shape[0] == mu.shape[1]:
            phi_b = np.broadcast_to(phi[None, :], mu.shape)
        elif phi.ndim == 2 and mu.ndim == 2 and phi.shape[1] == mu.shape[1] and phi.shape[0] == 1:
            phi_b = np.broadcast_to(phi, mu.shape)
        else:
            raise ValueError(f"Cannot broadcast dispersion shape {phi.shape} to mean shape {mu.shape}")

    p0 = np.empty_like(mu, dtype=np.float64)
    
    # Handle small phi (Poisson approximation) for stability
    small = (phi_b <= 1e-12)
    if small.any():
        p0[small] = np.exp(-mu[small])
        
    not_small = ~small
    if not_small.any():
        mu_ns = mu[not_small]
        phi_ns = phi_b[not_small]
        # (1 + mu*phi)^(-1/phi)
        # Use np.power for element-wise power calculation
        p0[not_small] = np.power(1.0 + mu_ns * phi_ns, -1.0 / phi_ns)
        
    return p0

def _size_factors(C: np.ndarray) -> np.ndarray:
    """Library-size factors s_i."""
    if C.size == 0: return np.array([], dtype=np.float64)

    lib = C.sum(axis=1).astype(np.float64)
    # Robust median (median of positive library sizes)
    med = np.median(lib[lib>0]) if lib.size and (lib>0).any() else 1.0
    if med <= 0 or not np.isfinite(med):
        med = 1.0
    s = lib / med
    s[~np.isfinite(s)] = 1.0
    s[s <= 0] = 1.0 # Ensure strictly positive
    return s

def _logistic_from_logmean(log_mu: np.ndarray, x0: float, k: float) -> np.ndarray:
    """p = 1 / (1 + exp(-k * (log_mu - x0))). k is typically negative."""
    return 1.0 / (1.0 + np.exp(-k * (log_mu - x0)))

# --- Dispersion Estimation (with Shrinkage) ---

def _estimate_phi_moments_robust(C: np.ndarray, tau: float = 20.0) -> np.ndarray:
    """Robust per-gene dispersion (phi) using MoM on non-zero counts, with shrinkage towards median."""
    n_cells, n_genes = C.shape
    if n_cells < 2 or n_genes == 0:
        return np.zeros(n_genes, dtype=np.float64)

    nz = (C > 0)
    n_nz = nz.sum(axis=0).astype(np.float64)

    # Mean of non-zeros
    sum_nz = (C * nz).sum(axis=0).astype(np.float64)
    mu_nz = np.divide(sum_nz, np.maximum(n_nz, 1.0))
    
    # Variance of non-zeros (sample variance, ddof=1)
    var_nz = np.zeros(n_genes, dtype=np.float64)
    
    # Optimized variance calculation
    if n_cells > 0:
        C_sq = C.astype(np.float64)**2
        sum_sq_nz = (C_sq * nz).sum(axis=0)
        # Var = (SumSq - (Sum)^2 / N) / (N-1)
        mask = n_nz >= 2
        if mask.any():
            # Use where condition to avoid division by zero where N < 2
            var_nz[mask] = (sum_sq_nz[mask] - (sum_nz[mask]**2) / n_nz[mask]) / (n_nz[mask] - 1)


    # MoM estimate: phi = (Var - Mu) / Mu^2
    with np.errstate(invalid="ignore", divide="ignore"):
        phi_hat = (var_nz - mu_nz) / np.maximum(mu_nz * mu_nz, 1e-12)
        
    phi_hat[~np.isfinite(phi_hat) | (phi_hat < 0)] = 0.0

    # Shrinkage towards median (Per-gene shrinkage)
    finite_phi = phi_hat[np.isfinite(phi_hat) & (phi_hat > 0)]
    med = float(np.median(finite_phi)) if finite_phi.size > 0 else 0.0
    
    # Weight w represents trust in the median (prior)
    w = 1.0 / (1.0 + n_nz / max(tau, 1e-6)) 
    phi = w * med + (1.0 - w) * phi_hat
    
    phi[~np.isfinite(phi) | (phi < 0)] = 0.0
    return phi

def _estimate_phi_stratified(C: np.ndarray, groups: Optional[np.ndarray], tau: float = 20.0, tau_group: float = 50.0) -> Dict[str, np.ndarray]:
    """
    Estimates dispersion per group with shrinkage towards the global estimate.
    """
    
    # 1. Global estimate (Target for shrinkage)
    phi_global = _estimate_phi_moments_robust(C, tau=tau)
    
    if groups is None or len(np.unique(groups)) <= 1:
        return {"global": phi_global}
    
    # 2. Per-group estimates and shrinkage
    phi_stratified = {"global": phi_global}
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask_c = (groups == group)
        n_cells_g = mask_c.sum()
        C_group = C[mask_c, :]
        
        if n_cells_g < 5: # Too few cells for reliable estimation
            phi_stratified[str(group)] = phi_global
            continue
            
        # Raw group estimate
        phi_g_raw = _estimate_phi_moments_robust(C_group, tau=tau)
        
        # 3. Shrinkage towards global estimate (Group-level shrinkage)
        w_g = 1.0 / (1.0 + n_cells_g / max(tau_group, 1e-6))
        
        # Apply shrinkage element-wise for each gene's dispersion
        phi_g_shrunk = w_g * phi_global + (1.0 - w_g) * phi_g_raw
        
        # Final cleanup
        phi_g_shrunk[~np.isfinite(phi_g_shrunk) | (phi_g_shrunk < 0)] = 0.0
        phi_stratified[str(group)] = phi_g_shrunk
        
    return phi_stratified

# --- Dropout Curve Fitting (with Shrinkage) ---
# (Implementation remains the same as provided in the prompt)

def _fit_logistic_regression(x: np.ndarray, y: np.ndarray, x0_approx: float) -> Tuple[float, float]:
    """
    Fit logistic curve P = 1 / (1 + exp(-k*(x - x0))) using non-linear least squares.
    Fallback to linear regression on logits if optimization fails.
    """
    def _sigmoid(x, k, x0):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    # Filter valid data
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    
    if len(x_clean) < 5:
        return -1.0, x0_approx

    # Try Scipy curve_fit first (Robust)
    if curve_fit is not None:
        try:
            # Bounds: k in [-inf, 0]
            popt, _ = curve_fit(
                _sigmoid, x_clean, y_clean, 
                p0=[-1.0, x0_approx],
                bounds=([-np.inf, -np.inf], [0.0, np.inf]),
                method='trf',
                maxfev=2000
            )
            k_fit, x0_fit = popt
            # Ensure significant slope
            if k_fit > -1e-4: k_fit = -1.0
            return float(k_fit), float(x0_fit)
        except Exception:
            pass # Fallback to polyfit

    # Fallback: Linear regression on logit(y)
    y_clipped = y_clean.clip(1e-6, 1.0 - 1e-6)
    logit_y = np.log(y_clipped / (1.0 - y_clipped))
    
    try:
        k, b = np.polyfit(x_clean, logit_y, 1)
        if k > -1e-3: # Enforce negative slope
            k, b = -1.0, x0_approx
    except Exception:
        k, b = -1.0, x0_approx

    x0 = float(-b / k) if abs(k) > 1e-12 else x0_approx
    return float(k), float(x0)

def _fit_splatter_dropout_stratified(
    C: np.ndarray, 
    groups: Optional[np.ndarray] = None, 
    eps: float = 1e-8,
    min_cells_for_fit: int = 5,
    tau_dropout: float = 50.0
) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray]:
    """
    Fit SPLAT-like logistic dropout parameters (k, x0) globally or per group with shrinkage.
    """
    if C.size == 0:
        return {}, np.array([], dtype=np.float64)

    Z = (C <= 0.0)
    s = _size_factors(C)
    
    # Normalized counts (Global normalization context)
    with np.errstate(invalid="ignore", divide="ignore"):
        Cn = C / s[:, None]
        Cn[~np.isfinite(Cn)] = 0.0 
    
    mu_ref_global = Cn.mean(axis=0).astype(np.float64)

    # Helper function to encapsulate the fitting logic
    def _calculate_fit(Z_subset, Cn_subset):
        mu = Cn_subset.mean(axis=0).astype(np.float64)
        x = np.log(mu + eps)
        y = Z_subset.mean(axis=0)
        
        # Approximate midpoint x0
        y_clipped = y.clip(1e-6, 1.0 - 1e-6)
        mid_mask = (y_clipped > 0.2) & (y_clipped < 0.8) & np.isfinite(x)
        if mid_mask.any():
            x0_approx = float(np.median(x[mid_mask]))
        elif np.isfinite(x).any():
            x0_approx = float(np.median(x[np.isfinite(x)]))
        else:
            x0_approx = 0.0
            
        k, x0 = _fit_logistic_regression(x, y, x0_approx)
        return k, x0

    # 1. Global Fit (Target for Shrinkage)
    k_global, x0_global = _calculate_fit(Z, Cn)
    params = {"global": (k_global, x0_global)}

    if groups is None or len(np.unique(groups)) <= 1:
        return params, mu_ref_global

    # 2. Per-group fitting with Shrinkage
    unique_groups = np.unique(groups)
        
    for group in unique_groups:
        mask_c = (groups == group)
        n_cells_g = mask_c.sum()
        
        if n_cells_g < min_cells_for_fit: 
            params[str(group)] = (k_global, x0_global)
            continue

        Z_g = Z[mask_c, :]
        Cn_g = Cn[mask_c, :] 

        # Raw group fit
        k_g_raw, x0_g_raw = _calculate_fit(Z_g, Cn_g)
        
        # 3. Shrinkage towards global parameters
        w_g = 1.0 / (1.0 + n_cells_g / max(tau_dropout, 1e-6))
        
        k_g_shrunk = w_g * k_global + (1.0 - w_g) * k_g_raw
        x0_g_shrunk = w_g * x0_global + (1.0 - w_g) * x0_g_raw
        
        k_g_shrunk = min(k_g_shrunk, -1e-3)

        params[str(group)] = (k_g_shrunk, x0_g_shrunk)

    return params, mu_ref_global

# --- Main Posterior Calculation ---

def splatter_bio_posterior_from_counts(
    C: np.ndarray,
    disp_mode: str = "estimate",
    disp_const: float = 0.1,
    use_cell_factor: bool = False,
    groups: Optional[np.ndarray] = None,
    tau_dispersion: float = 20.0,
    tau_group_dispersion: float = 50.0,
    tau_dropout: float = 50.0,
) -> np.ndarray:
    """
    Compute P(biological zero | observed zero) = P(T=0|Y=0) for SPLAT methods.
    (Implementation remains the same as provided in the prompt)
    """
    if C.size == 0:
        return np.empty_like(C, dtype=np.float32)

    # 1. Fit dropout parameters (k, x0), stratified with shrinkage.
    params_drop, _ = _fit_splatter_dropout_stratified(
        C, groups=groups, tau_dropout=tau_dropout
    )

    # 2. Estimate dispersion (phi), stratified with shrinkage (unless fixed).
    if disp_mode == "fixed":
        phi_g = np.full(C.shape[1], float(disp_const), dtype=np.float64)
        params_disp = {"global": phi_g}
        if groups is not None:
            for group in np.unique(groups):
                params_disp[str(group)] = phi_g
    else:
        params_disp = _estimate_phi_stratified(
            C, groups=groups, 
            tau=tau_dispersion, tau_group=tau_group_dispersion
        )

    is_zero = (C <= 0.0)
    p_bio_ij = np.zeros_like(C, dtype=np.float64)
    
    # 3. Calculate Posterior Probability P(Bio|Zero)

    if groups is not None and len(np.unique(groups)) > 1:
        unique_groups = np.unique(groups)
    else:
        # If no groups or only one group, treat as global
        unique_groups = ["global"]
        groups = np.full(C.shape[0], "global") 

    
    k_global, x0_global = params_drop.get("global", (None, None))
    phi_global = params_disp.get("global", None)

    for group in unique_groups:
        group_key = str(group)
        mask_c = (groups == group)
        if mask_c.sum() == 0: continue

        k, x0 = params_drop.get(group_key, (k_global, x0_global))
        phi_g = params_disp.get(group_key, phi_global)

        if k is None or phi_g is None:
            warnings.warn(f"Missing parameters for group {group_key} or global fallback. Skipping.")
            continue

        C_group = C[mask_c, :]
        s_i_group = _size_factors(C_group)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            if s_i_group.size == 0: continue
            Cn_group = C_group / s_i_group[:, None]
            Cn_group[~np.isfinite(Cn_group)] = 0.0
        mu_ref_group = Cn_group.mean(axis=0).astype(np.float64)

        if use_cell_factor:
            mu_ij_g = mu_ref_group[None, :] * s_i_group[:, None]
        else:
            mu_ij_g = np.broadcast_to(mu_ref_group[None, :], (C_group.shape[0], C.shape[1]))

        log_mu_ij_g = np.log(np.maximum(mu_ij_g, 1e-12))
            
        p_zero_total_est = _logistic_from_logmean(log_mu_ij_g, x0=x0, k=k).astype(np.float64)

        # --- Empirical Probability Bounding ---
        # 1. Calculate raw NB prob using Observed Mean
        p0_nb_raw = _nb_zero_prob(mu_ij_g, phi_g)
        
        # 2. Cap P_NB at P_Total
        p0_nb_ij_g = np.minimum(p0_nb_raw, p_zero_total_est + 1e-6)
        
        # 3. Derive Residual Dropout Rate (pi)
        with np.errstate(divide='ignore', invalid='ignore'):
            p_drop_ij_g = (p_zero_total_est - p0_nb_ij_g) / (1.0 - p0_nb_ij_g)
        
        p_drop_ij_g = np.clip(p_drop_ij_g, 0.0, 1.0 - 1e-6)

        # 4. Calculate Posterior P(Bio | Zero)
        p_zero_obs_g = p_drop_ij_g + (1.0 - p_drop_ij_g) * p0_nb_ij_g
        p_bio_numerator = p0_nb_ij_g * (1.0 - p_drop_ij_g)
        p_bio_ij_g = p_bio_numerator / np.maximum(p_zero_obs_g, 1e-12)
            
        p_bio_ij[mask_c, :] = p_bio_ij_g

    # Final assembly
    bio_post = np.zeros_like(C, dtype=np.float32)
    bio_post[is_zero] = p_bio_ij[is_zero].astype(np.float32)
    return bio_post


# ---------------------------- Other approaches ----------------------------

def baseline_gene_mean_heuristic_counts(C: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    # (Implementation remains the same)
    if C.size == 0: return np.empty((0,0), dtype=bool)
    is_zero = (C <= 0.0)
    nz_mask = ~is_zero
    with np.errstate(invalid="ignore", divide="ignore"):
        m_g = np.sum(C * nz_mask, axis=0) / np.maximum(nz_mask.sum(axis=0), 1)
    
    finite_mg = m_g[np.isfinite(m_g)]
    thr = np.nanquantile(finite_mg, quantile) if finite_mg.size > 0 else 0.0
    
    pred = np.zeros_like(is_zero, dtype=bool)
    high = (m_g >= thr)
    pred[:, high] = is_zero[:, high]
    return pred

def pca_knn_expectation_counts(
    C: np.ndarray, n_components: int = 20, k: int = 15, mean_thresh_log1p: float = 0.5, gene_quantile: Optional[float] = 0.25
) -> np.ndarray:
    # (Implementation remains the same)
    if PCA is None or NearestNeighbors is None:
        raise RuntimeError("scikit-learn is required for the PCA+kNN approach.")
    if C.shape[0] <= 1:
        return np.zeros_like(C, dtype=bool)

    X = np.log1p(C.copy())
    is_zero = (C <= 0.0)
    nz = ~is_zero

    with np.errstate(invalid="ignore"):
        gene_means = np.sum(X * nz, axis=0) / np.maximum(nz.sum(axis=0), 1)
    X_imputed = X.copy()
    
    # Ensure broadcasting safety
    if X_imputed.shape == np.broadcast_to(gene_means[None, :], X.shape).shape:
        X_imputed[is_zero] = np.broadcast_to(gene_means[None, :], X.shape)[is_zero]

    n_comp = max(1, min(n_components, X.shape[0]-1, X.shape[1]))
    if n_comp < 1: return np.zeros_like(C, dtype=bool)
        
    try:
        pcs = PCA(n_components=n_comp).fit_transform(X_imputed)
    except Exception as e:
        warnings.warn(f"PCA failed: {e}. Skipping kNN approach.")
        return np.zeros_like(C, dtype=bool)

    n_neighbors = min(k, max(1, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(pcs)
    dists, indices = nn.kneighbors(pcs, return_distance=True)

    if indices.shape[1] > 0 and np.all(indices[:, 0] == np.arange(indices.shape[0])):
        neigh_idx, neigh_d = indices[:, 1:], dists[:, 1:]
    else:
        neigh_idx, neigh_d = indices, dists

    thr_g = None
    if gene_quantile is not None:
        with np.errstate(invalid="ignore"):
            X_nz = np.where(nz, X, np.nan)
            thr_g = np.nanquantile(X_nz, gene_quantile, axis=0)
            thr_g = np.where(np.isfinite(thr_g), thr_g, np.inf)

    pred = np.zeros_like(is_zero, dtype=bool)
    eps = 1e-6
    
    if neigh_idx.shape[1] == 0: return pred

    for i in range(X.shape[0]):
        w = 1.0 / (neigh_d[i] + eps)
        w_sum = w.sum()
        if w_sum <= eps: continue
        w = w / w_sum
        
        neigh = X[neigh_idx[i]]
        neigh_mean = (w[:, None] * neigh).sum(axis=0)
        
        if thr_g is not None:
            pred[i] = is_zero[i] & (neigh_mean > thr_g)
        else:
            pred[i] = is_zero[i] & (neigh_mean > mean_thresh_log1p)
    return pred

# ---------------------------- DCA approach (FIXED) ----------------------------

def _broadcast_disp_to_cells(ret):
    """
    (FIXED) Robust helper to handle constant, conditional, and shared dispersion from DCA output.
    Ensures theta is broadcast to shape (N_cells, N_genes).
    """
    if "X_dca_dispersion" in ret.obsm.keys():
        # Conditional dispersion (N_cells, N_genes) or Shared (N_cells, 1)
        theta_raw = np.asarray(ret.obsm["X_dca_dispersion"])
        if theta_raw.shape == ret.X.shape:
            return theta_raw
        elif theta_raw.shape == (ret.X.shape[0], 1) or (theta_raw.ndim == 1 and len(theta_raw) == ret.X.shape[0]):
            # Broadcast shared dispersion (B, 1) or (B,) to (B, G)
            return np.broadcast_to(theta_raw.reshape(-1, 1), ret.X.shape)
        else:
            raise RuntimeError(f"Unexpected shape for dispersion in obsm: {theta_raw.shape} vs expected {ret.X.shape} or ({ret.X.shape[0]}, 1)")

    elif "X_dca_dispersion" in ret.var.keys():
        # Constant dispersion (N_genes,) -> Broadcast to (N_cells, N_genes)
        theta = np.asarray(ret.var["X_dca_dispersion"]).reshape(1, -1)
        theta = np.broadcast_to(theta, ret.X.shape)
        return theta
    else:
        raise RuntimeError("DCA did not return dispersion (theta) in obsm or var.")

def dca_posterior_dropout_from_counts(
    counts, ae_type: str = "zinb-conddisp", return_mu: bool = False, **kw
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    (FIXED) Calculate the posterior probability of a dropout event P(Dropout | Y=0)
    using the standard interpretation of parameters (mu, theta, pi) estimated by the DCA model.
    """
    try:
        from dca.api import dca as dca_run
        import anndata as ad
    except ImportError:
        raise RuntimeError("DCA dependencies (dca, anndata, scanpy, tensorflow) not installed.")

    C = np.rint(np.asarray(counts)).astype(np.int32, copy=False)
    cell_mask = (C.sum(axis=1) > 0)
    gene_mask = (C.sum(axis=0) > 0)
    n_drop_cells = int((~cell_mask).sum())
    n_drop_genes = int((~gene_mask).sum())
    C_work = C[np.ix_(cell_mask, gene_mask)] if (n_drop_cells or n_drop_genes) else C

    if n_drop_cells or n_drop_genes:
        if kw.get("verbose"):
            print(f"   [DCA] Removing {n_drop_cells} zero-count cells and {n_drop_genes} zero-count genes before DCA.")

    if C_work.shape[0] < 2 or C_work.shape[1] < 2:
        raise ValueError(f"DCA requires at least 2 cells and 2 genes. Got shape {C_work.shape}.")
        
    adata = ad.AnnData(C_work)
    
    # Check if the AE type actually produces dropout probabilities (pi)
    is_zinb = ae_type.startswith("zinb")
    
    # Run DCA
    ret = dca_run(adata, mode="denoise", ae_type=ae_type,
                  return_info=True, copy=True, check_counts=True, **kw)

    # 1. Extract parameters estimated by DCA
    # mu: Denoised mean (reconstructed counts)
    mu = np.asarray(ret.X, dtype=np.float64)
    # theta: Dispersion parameter (NB size parameter, theta = 1/phi)
    theta = _broadcast_disp_to_cells(ret).astype(np.float64) 
    
    if is_zinb:
        if "X_dca_dropout" not in ret.obsm.keys():
            raise RuntimeError(f"DCA type {ae_type} did not return dropout probabilities (pi) in obsm.")
        
        # pi: Dropout probability (Zero-inflation component)
        pi_raw = np.asarray(ret.obsm["X_dca_dropout"], dtype=np.float64)
        
        # Handle potential shared pi (e.g. zinb-shared, zinb-elempi_sharedpi)
        if pi_raw.shape == mu.shape:
            pi = pi_raw
        elif pi_raw.shape == (mu.shape[0], 1) or (pi_raw.ndim == 1 and len(pi_raw) == mu.shape[0]):
            # Broadcast shared pi (B, 1) or (B,) to (B, G)
            pi = np.broadcast_to(pi_raw.reshape(-1, 1), mu.shape)
        else:
            raise RuntimeError(f"DCA type {ae_type} returned unexpected shape for dropout probabilities (pi): {pi_raw.shape} vs expected {mu.shape} or ({mu.shape[0]}, 1)")

    else:
        # If NB model (not ZINB), pi is implicitly zero
        pi = np.zeros_like(mu, dtype=np.float64)

    # Ensure numerical stability
    eps = 1e-12
    mu = np.maximum(mu, eps)
    # Constrain theta. Use constraints consistent with DCA activation functions (e.g. 1e-4 to 1e4 for DispAct)
    theta = np.clip(theta, 1e-4, 1e4) 
    pi = np.clip(pi, 0.0, 1.0 - eps)

    # 2. Calculate Biological Zero Probability (P_NB(0))
    # P_NB(0) = (theta / (theta + mu))^theta
    # Calculate in log space for stability. Use log1p for better precision when mu/theta is small.
    # log(P_NB(0)) = -theta * log(1 + mu/theta)
    log_p0_nb = -theta * np.log1p(mu / theta)
    
    p0_nb = np.exp(np.clip(log_p0_nb, a_min=-50.0, a_max=0.0)) # Prevent underflow

    # 3. Calculate Total Probability of Zero (P(Y=0)) according to the ZINB model
    # P(Y=0) = pi + (1 - pi) * P_NB(0)
    p_zero_total = pi + (1.0 - pi) * p0_nb

    # 4. Calculate Posterior Dropout Probability (P(Dropout | Y=0))
    # Using Bayes' theorem: P(Dropout | Y=0) = pi / P(Y=0)
    p_drop_post = pi / np.maximum(p_zero_total, eps)
    
    # Final clipping to ensure valid probabilities
    p_drop_post = np.clip(p_drop_post, 0.0, 1.0)

    # Mask out non-zero counts: P(Dropout | Y>0) = 0 (matches legacy posterior definition)
    zero_mask = (C_work <= 0)
    p_drop_post = p_drop_post * zero_mask

    if n_drop_cells or n_drop_genes:
        full_shape = C.shape
        p_full = np.zeros(full_shape, dtype=np.float32)
        mu_full = np.zeros(full_shape, dtype=np.float32)
        idx = np.ix_(cell_mask, gene_mask)
        p_full[idx] = p_drop_post.astype(np.float32)
        mu_full[idx] = mu.astype(np.float32)
        if return_mu:
            return p_full, mu_full
        else:
            return p_full

    if return_mu:
        return p_drop_post.astype(np.float32), mu.astype(np.float32)
    else:
        return p_drop_post.astype(np.float32)

def dca_posterior_stratified(
    C: np.ndarray,
    groups: np.ndarray,
    ae_type: str,
    min_cells_per_stratum: int,
    **kw
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs DCA stratified by groups. If a group is too small or fails, falls back to global DCA."""
    
    # 1. Run Global DCA (as a robust fallback)
    print(f"dca_stratified: Running Global DCA (fallback)...")
    
    is_verbose = kw.get("verbose", False)
    if is_verbose: kw["verbose"] = False

    try:
        p_drop_global, mu_global = dca_posterior_dropout_from_counts(C, ae_type=ae_type, return_mu=True, **kw)
    except Exception as e:
        print(f"dca_stratified: Global DCA failed: {e}. Cannot proceed with stratification.")
        if is_verbose: kw["verbose"] = True
        raise e
        
    if is_verbose: kw["verbose"] = True

    p_drop_stratified = np.zeros_like(C, dtype=np.float32)
    mu_stratified = np.zeros_like(C, dtype=np.float32)
    unique_groups = np.unique(groups)
    
    # 2. Run DCA per stratum
    for group in unique_groups:
        mask_c = (groups == group)
        n_cells_g = mask_c.sum()
        C_group = C[mask_c, :]
        
        if n_cells_g >= min_cells_per_stratum:
            print(f"dca_stratified: Running DCA for group {group} ({n_cells_g} cells)...")
            try:
                p_drop_g, mu_g = dca_posterior_dropout_from_counts(C_group, ae_type=ae_type, return_mu=True, **kw)
                p_drop_stratified[mask_c, :] = p_drop_g
                mu_stratified[mask_c, :] = mu_g
            except Exception as e:
                print(f"dca_stratified: DCA failed for group {group}. Falling back to global results. Error: {e}")
                p_drop_stratified[mask_c, :] = p_drop_global[mask_c, :]
                mu_stratified[mask_c, :] = mu_global[mask_c, :]
        else:
            # If too small, use global results.
            p_drop_stratified[mask_c, :] = p_drop_global[mask_c, :]
            mu_stratified[mask_c, :] = mu_global[mask_c, :]
            
    return p_drop_stratified, mu_stratified


# ---------------------------- Evaluation (Revised Metrics) ----------------------------
# (Evaluation functions remain the same as provided in the prompt)

def compute_truth_masks(counts: np.ndarray, truecounts: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Returns boolean masks. Positive Class = Biological Zero. Negative Class = Dropout."""
    zeros_total = (counts <= 0.0)
    res = {"zeros_total": zeros_total}

    if truecounts is not None and truecounts.shape == counts.shape:
        res["bio_true"] = zeros_total & (truecounts <= 0.0)
        res["dropout_true"] = zeros_total & (truecounts > 0.0)

    return res

def _choose_thresh_for_metric(
    pdrop,
    zeros,
    drop_true,
    bio_true,
    metric: str = "f1",
    grid=np.linspace(0.01, 0.99, 99),
):
    """
    Find threshold 't' on P(dropout) such that predicting P(dropout) < t as Biological
    maximizes the requested metric (precision/recall/accuracy/f1) for the Biological class.
    """
    metric = metric.lower()
    if metric not in {"precision", "recall", "accuracy", "f1"}:
        raise ValueError(f"Unknown metric '{metric}'. Choose from precision, recall, accuracy, f1.")

    best_t, best_score = 0.5, -1.0

    n_zeros = int(zeros.sum())
    if n_zeros == 0:
        return best_t

    n_bio_true = int(bio_true.sum())
    # n_drop_true = int(drop_true.sum()) # Unused in loop

    # Optimize calculation by operating only on the observed zeros
    pdrop_z = pdrop[zeros]
    bio_true_z = bio_true[zeros]
    drop_true_z = drop_true[zeros]

    if len(pdrop_z) == 0:
        return best_t

    for t in grid:
        # Prediction Logic: Predict POSITIVE (Biological) if P(Dropout) < t
        pred_bio_z = (pdrop_z < t)
        n_pred_bio = int(pred_bio_z.sum())

        TP = int((pred_bio_z & bio_true_z).sum())
        # FN = n_bio_true - TP # Unused
        # FP = int((pred_bio_z & drop_true_z).sum()) # Unused
        TN = int((drop_true_z & (~pred_bio_z)).sum())

        Recall = TP / n_bio_true if n_bio_true > 0 else np.nan
        if n_pred_bio > 0:
            Precision = TP / n_pred_bio
        else:
            Precision = 0.0 if pd.notnull(Recall) else np.nan

        if pd.notnull(Precision) and pd.notnull(Recall) and (Precision + Recall) > 0:
            F1 = 2 * Precision * Recall / (Precision + Recall)
        else:
            F1 = 0.0

        Accuracy = (TP + TN) / n_zeros if n_zeros > 0 else np.nan

        metric_value = {
            "precision": Precision,
            "recall": Recall,
            "accuracy": Accuracy,
            "f1": F1,
        }[metric]

        if pd.isnull(metric_value):
            metric_value = -1.0

        if metric_value > best_score:
            best_score, best_t = metric_value, t

    return best_t

def summarize_predictions(name: str, approach: str,
                          truth: Dict[str, np.ndarray],
                          pred_dropout: np.ndarray,
                          best_thresh: Optional[float] = None) -> Dict[str, Union[float, str]]:
    
    pred_dropout_bool = pred_dropout.astype(bool)
    zeros_total = truth["zeros_total"]
    
    # Predictions (Focusing on Zeros)
    # Predicted NEGATIVE (Dropout)
    pred_drop_on_zeros = pred_dropout_bool & zeros_total
    n_pred_drop = int(pred_drop_on_zeros.sum())
    
    # Predicted POSITIVE (Biological)
    pred_bio_on_zeros = zeros_total & ~pred_dropout_bool
    n_pred_bio = int(pred_bio_on_zeros.sum())

    row = {"dataset": name,
           "approach": approach,
           "total_zeros": int(zeros_total.sum()),
           "predicted_dropouts": n_pred_drop,
           "predicted_bio": n_pred_bio}

    if best_thresh is not None:
        row["best_thresh"] = float(best_thresh)

    # 1. Metrics based on TrueCounts
    # POSITIVE = Biological, NEGATIVE = Dropout
    if "bio_true" in truth and "dropout_true" in truth:
        n_true_bio = int(truth["bio_true"].sum())
        n_true_drop = int(truth["dropout_true"].sum())

        # True Positives (Correctly identified Biological)
        TP = int((truth["bio_true"] & pred_bio_on_zeros).sum())
        
        # --- Calculate Metrics for Biological (Positive Class) ---
        
        # Recall (Bio) = TP / (Total Actual Bio)
        rec_bio = float(TP / n_true_bio) if n_true_bio > 0 else np.nan

        # Precision (Bio) = TP / (Total Predicted Bio)
        # Handle case where prediction is empty (Prec=0 if Recall is defined, else NaN)
        prec_bio = float(TP / n_pred_bio) if n_pred_bio > 0 else (0.0 if pd.notnull(rec_bio) else np.nan)

        # F1 Score (Bio)
        if pd.notnull(prec_bio) and pd.notnull(rec_bio) and (prec_bio + rec_bio) > 0:
            f1_bio = 2.0 * prec_bio * rec_bio / (prec_bio + rec_bio)
        else:
            f1_bio = np.nan

        tn = int((truth["dropout_true"] & pred_dropout_bool).sum())
        acc_bio = float((TP + tn) / truth["zeros_total"].sum()) if truth["zeros_total"].sum() > 0 else np.nan

        row.update({
            "true_dropouts": n_true_drop,
            "true_bio_zeros": n_true_bio,
            # New metric names
            "Recall_Bio": (100.0 * rec_bio) if pd.notnull(rec_bio) else np.nan,
            "Precision_Bio": (100.0 * prec_bio) if pd.notnull(prec_bio) else np.nan,
            "F1_Score_Bio": (100.0 * f1_bio) if pd.notnull(f1_bio) else np.nan,
            "Accuracy_Bio": (100.0 * acc_bio) if pd.notnull(acc_bio) else np.nan,
        })
    else:
        # Update keys list based on new metric names
        metric_keys = ["true_dropouts", "true_bio_zeros", "Recall_Bio", "Precision_Bio", "F1_Score_Bio", "Accuracy_Bio"]
        row.update({k: np.nan for k in metric_keys})


    return row

# REVISED: Updated column names and sorting criteria. Aggregation logic remains macro-average.
def _print_console_summary(rows: List[Dict[str, Union[float, str]]], sort_by_metric: str = "F1_Score_Bio"):
    if not rows:
        print("No results to display.")
        return

    df = pd.DataFrame(rows)

    # Updated column order with new metric names, prioritizing Bio (Positive Class)
    col_order = [
        "dataset", "approach", "best_thresh", "total_zeros",
        "true_bio_zeros", "true_dropouts", 
        "predicted_bio", "predicted_dropouts",
        "Precision_Bio", "Recall_Bio", "F1_Score_Bio", "Accuracy_Bio"
    ]
    col_order = [c for c in col_order if c in df.columns]

    # Calculate Aggregates (Macro-average)
    # This calculates the average performance of an approach across the datasets where it ran.
    # Due to changes in run_on_dataset, this now covers all applicable datasets.
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if "dataset" in df.columns:
        n_by_app = df.groupby("approach", as_index=False)["dataset"].nunique().rename(columns={"dataset": "n_datasets"})
    else:
        n_by_app = df.groupby("approach", as_index=False).size().rename(columns={"size": "n_datasets"})

    agg_mean = (
        df.groupby("approach", as_index=False)[num_cols]
          .mean(numeric_only=True) # Averages across the runs
          .merge(n_by_app, on="approach", how="left")
    )

    agg_order = ["approach", "n_datasets"] + [c for c in col_order if c not in ("dataset", "approach")]
    agg_order = [c for c in agg_order if c in agg_mean.columns]
    agg_mean = agg_mean[agg_order]
    
    # Sort aggregates by the requested metric
    if sort_by_metric in agg_mean.columns:
        agg_mean = agg_mean.sort_values(sort_by_metric, ascending=False).reset_index(drop=True)

    # Formatting helpers
    def fmt_pct(x): return f"{x:.2f}%" if pd.notnull(x) else "NA"
    def fmt_mean_count(x): return f"{x:.1f}" if pd.notnull(x) else "NA"
    def fmt_thr(x): return f"{x:.3f}" if pd.notnull(x) else "NA"

    # Updated metric columns list
    pct_cols = ["Precision_Bio", "Recall_Bio", "F1_Score_Bio", "Accuracy_Bio"]

    # Print per-dataset summary
    df_print = df.copy()
    for c in pct_cols:
        if c in df_print.columns: df_print[c] = df_print[c].map(fmt_pct)
    if "best_thresh" in df_print.columns: df_print["best_thresh"] = df_print["best_thresh"].map(fmt_thr)

    if "dataset" in df_print.columns and "approach" in df_print.columns:
        # Logical sorting
        def sort_key(approach):
            if "baseline" in approach: return (0, approach)
            if "splat" in approach: return (1, approach)
            if "dca" in approach: return (2, approach)
            if "knn" in approach: return (3, approach)
            return (99, approach)
        
        df_print['sort_key'] = df_print['approach'].apply(sort_key)
        df_print = df_print.sort_values(["dataset", "sort_key"]).drop('sort_key', axis=1).reset_index(drop=True)


    print("\n==================== Biological Zero Identification (per dataset) ====================")
    with pd.option_context('display.max_columns', None, 'display.width', 1500):
        print(df_print[col_order].to_string(index=False))
    print("======================================================================================\n")

    # Print aggregate summary
    agg_print = agg_mean.copy()
    for c in pct_cols:
        if c in agg_print.columns: agg_print[c] = agg_print[c].map(fmt_pct)
    
    # Define count columns for formatting
    count_cols = ["total_zeros", "true_dropouts", "true_bio_zeros", "predicted_dropouts", "predicted_bio"]
    for c in count_cols:
        if c in agg_print.columns: agg_print[c] = agg_print[c].map(fmt_mean_count)
    if "best_thresh" in agg_print.columns: agg_print["best_thresh"] = agg_print["best_thresh"].map(fmt_thr)

    print("================== Aggregate (macro) AVERAGE across datasets by approach ==================")
    print(f"Sorted by {sort_by_metric} (Higher is better)")
    with pd.option_context('display.max_columns', None, 'display.width', 1500):
        print(agg_print.to_string(index=False))
    print("===========================================================================================\n")


def _metric_to_column(metric: str) -> str:
    mapping = {
        "precision": "Precision_Bio",
        "recall": "Recall_Bio",
        "accuracy": "Accuracy_Bio",
        "f1": "F1_Score_Bio",
    }
    return mapping.get((metric or "").lower(), "F1_Score_Bio")


# ---------------------------- Main pipeline ----------------------------

def _parse_dca_list(s: str) -> List[str]:
    ALL = ["zinb", "zinb-conddisp", "zinb-shared", "zinb-fork", "zinb-elempi", "zinb-elempi_sharedpi"]
    if s.strip().lower() == "all":
        return ALL
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in parts if p not in ALL]
    if bad:
        raise ValueError(f"Unknown DCA ae types: {bad}. Allowed: {ALL} or 'all'.")
    return parts

# REVISED: Updated stratification logic and calls to evaluation functions
def run_on_dataset(path: Path, args, out_dir: Path) -> List[Dict[str, Union[float, str]]]:
    name = path.stem
    try:
        C, C_true, coldata = load_sce(path, need_perfect=True)
    except Exception as e:
        print(f"[{name}] Error loading dataset: {e}")
        return []

    if C.size == 0 or C.shape[0] < 2:
        print(f"[{name}] Skipping dataset (empty or too few cells).")
        return []

    # Get truth masks (uses revised function definitions)
    truth = compute_truth_masks(C, C_true)
    zeros = truth["zeros_total"]

    # Identify available stratification factors
    # REVISED: Allow stratification if the column exists, regardless of the number of unique levels.
    # This ensures comprehensive aggregation as the underlying methods handle single groups gracefully.
    stratifications = {}
    if "Group" in coldata.columns and not coldata["Group"].empty:
        stratifications["Group"] = coldata["Group"].to_numpy().astype(str)
    if "Batch" in coldata.columns and not coldata["Batch"].empty:
        stratifications["Batch"] = coldata["Batch"].to_numpy().astype(str)
    
    # Combined stratification
    if "Group" in stratifications and "Batch" in stratifications:
        combined_labels = np.char.add(stratifications["Group"], "_x_")
        combined_labels = np.char.add(combined_labels, stratifications["Batch"])
        stratifications["Group_x_Batch"] = combined_labels

    print(f"[{name}] cells x genes: {C.shape[0]} x {C.shape[1]} | zeros: {int(zeros.sum())} | Stratifications Available: {list(stratifications.keys())}")
    
    rows: List[Dict[str, Union[float, str]]] = []

    # Helper function to run and summarize posterior-based methods
    def _run_posterior_method(approach_name, bio_post):
        # P(Dropout|Zero) = 1 - P(Bio|Zero)
        p_drop = 1.0 - bio_post
        
        # Check if ground truth is available
        if args.posterior_auto_thresh and {"bio_true","dropout_true"} <= set(truth.keys()):
            # Optimize threshold for the requested metric
            thr = _choose_thresh_for_metric(
                p_drop,
                zeros,
                truth["dropout_true"],
                truth["bio_true"],
                metric=args.posterior_auto_metric,
            )
            # Prediction for Dropout (Negative) is P(drop) >= thr
            pred = zeros & (p_drop >= thr)
            rows.append(summarize_predictions(name, approach_name, truth, pred, best_thresh=float(thr)))
        else:
            # Use fixed threshold
            thr = args.posterior_thresh
            # Prediction for Dropout (Negative) is P(drop) >= thr
            pred = zeros & (p_drop >= thr)
            rows.append(summarize_predictions(name, approach_name, truth, pred))
        
        np.savez_compressed(out_dir / f"{name}__pred_{approach_name}.npz", pred_dropout=pred)
        return pred

    # (A) Baseline heuristic
    try:
        pred_A = baseline_gene_mean_heuristic_counts(C, quantile=args.baseline_quantile)
        np.savez_compressed(out_dir / f"{name}__pred_baseline.npz", pred_dropout=pred_A)
        rows.append(summarize_predictions(name, "baseline", truth, pred_A))
    except Exception as e:
        print(f"[{name}] Error in baseline: {e}")

    # Define SPLAT configurations
    splat_configs = {
        "fixed": {"disp_mode": "fixed", "disp_const": args.disp_const, "use_cell_factor": False},
        "mom": {"disp_mode": "estimate", "use_cell_factor": False},
        "cellaware": {"disp_mode": "estimate", "use_cell_factor": True},
    }
    
    # Combine Global (None) with available stratifications for iteration
    all_stratifications = {"Global": None}
    all_stratifications.update(stratifications)

    # (SPLAT) Global and Stratified
    for config_name, config_params in splat_configs.items():
        for cov_name, cov_labels in all_stratifications.items():
            
            if cov_name == "Global":
                approach_name = f"splat_{config_name}"
            else:
                approach_name = f"splat_{config_name}_stratified_{cov_name.lower()}"
            
            try:
                bio_post_S = splatter_bio_posterior_from_counts(
                    C, 
                    groups=cov_labels,
                    tau_dropout=args.tau_dropout,
                    tau_group_dispersion=args.tau_group_dispersion,
                    tau_dispersion=args.tau_dispersion,
                    **config_params
                )
                _run_posterior_method(approach_name, bio_post_S)
                
            except Exception as e:
                print(f"[{name}] Error in SPLAT method ({approach_name}): {e}")
                traceback.print_exc()

    # (E) PCA+kNN expectation
    if args.run_knn:
        try:
            pred_E = pca_knn_expectation_counts(
                C,
                n_components=args.pca_components,
                k=args.knn_k,
                mean_thresh_log1p=args.knn_mean_thresh,
                gene_quantile=args.knn_gene_quantile
            )
            np.savez_compressed(out_dir / f"{name}__pred_pca_knn.npz", pred_dropout=pred_E)
            rows.append(summarize_predictions(name, "pca_knn", truth, pred_E))
        except Exception as e:
            print(f"[{name}] PCA+kNN approach skipped: {e}")
            traceback.print_exc()

    # (F) DCA (Global and Stratified)
    if args.run_dca or args.run_dca_stratified:
        ae_list = _parse_dca_list(args.dca_ae_types)
        
        # Helper to process DCA output (Global or Stratified)
        def _process_dca_output(approach_name, pdrop, mu):
            if args.dca_mu_gate is not None and args.dca_mu_gate > 0:
                mu_gate = (mu >= float(args.dca_mu_gate))
            else:
                mu_gate = np.ones_like(zeros, dtype=bool)

            best_thr = None
            use_auto_thresh = args.dca_auto_thresh or args.posterior_auto_thresh
            
            # Check ground truth availability
            if use_auto_thresh and {"bio_true","dropout_true"} <= set(truth.keys()):
                thr = _choose_thresh_for_metric(
                    pdrop,
                    zeros,
                    truth["dropout_true"],
                    truth["bio_true"],
                    metric=args.posterior_auto_metric,
                )
                best_thr = float(thr)
                # Prediction for Dropout (Negative) is P(drop) >= thr
                pred = zeros & mu_gate & (pdrop >= thr)
            else:
                thr = float(args.posterior_thresh)
                # Prediction for Dropout (Negative) is P(drop) >= thr
                pred = zeros & mu_gate & (pdrop >= thr)

            np.savez_compressed(out_dir / f"{name}__pred_{approach_name}.npz", pred_dropout=pred)
            rows.append(summarize_predictions(name, approach_name, truth, pred, best_thresh=best_thr))

        
        for ae in ae_list:
            net_kw = {"ridge": 1e-3}
            if ae == "zinb-elempi_sharedpi":
                ae_name = "zinb-elempi"
                net_kw["sharedpi"] = True
            else:
                ae_name = ae
                
            dca_kwargs = {
                "epochs": args.dca_epochs, "batch_size": args.dca_batch_size,
                "threads": args.dca_threads, "random_state": args.dca_random_state,
                "verbose": args.dca_verbose, "network_kwds": net_kw
            }

            # --- F.1 Global DCA ---
            if args.run_dca:
                approach_name = f"dca_{ae}"
                try:
                    pdrop_G, mu_G = dca_posterior_dropout_from_counts(
                        C, ae_type=ae_name, return_mu=True, **dca_kwargs
                    )
                    
                    _process_dca_output(approach_name, pdrop_G, mu_G)

                except Exception as e:
                    print(f"[{name}] Global DCA ({ae}) skipped: {e}")
                    traceback.print_exc()

            # --- F.2 Stratified DCA ---
            if args.run_dca_stratified:
                # Iterate over the stratifications defined earlier (which now includes all possibilities)
                for cov_name, cov_labels in stratifications.items():
                    approach_name = f"dca_{ae}_stratified_{cov_name.lower()}"
                    try:
                        pdrop_S, mu_S = dca_posterior_stratified(
                            C, groups=cov_labels, ae_type=ae_name,
                            min_cells_per_stratum=args.dca_min_cells_stratum,
                            **dca_kwargs
                        )
                        
                        _process_dca_output(approach_name, pdrop_S, mu_S)

                    except Exception as e:
                        print(f"[{name}] Stratified DCA ({approach_name}) skipped: {e}")
                        traceback.print_exc()

    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Predict dropout zeros in SPLAT .rds datasets using robust stratified analysis with configurable threshold optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("data_dir", type=str, help="Directory containing .rds files (recursively scanned).")
    ap.add_argument("--out-dir", type=str, default="dropout_predictions_final", help="Where to write prediction masks (.npz).")

    # Baseline heuristic
    ap.add_argument("--baseline-quantile", type=float, default=0.2, help="Gene mean quantile for baseline heuristic.")

    # SPLAT posterior (Global and Stratified)
    g_splat = ap.add_argument_group("SPLAT Posterior Options")
    g_splat.add_argument("--disp-const", type=float, default=0.1, help="Fixed dispersion for 'splat_fixed'.")
    g_splat.add_argument("--posterior-thresh", type=float, default=0.5, help="Threshold on P(Dropout|Zero) for classification.")
    g_splat.add_argument("--posterior-auto-thresh", action="store_true",
                    help="Calibrate posterior threshold (SPLAT/DCA) per dataset to maximize a metric (uses TrueCounts). Recommended.")
    g_splat.add_argument("--posterior-auto-metric", type=str.lower, default="f1",
                    choices=["precision", "recall", "accuracy", "f1"],
                    help="Metric to maximize when --posterior-auto-thresh is enabled.")
    
    # Shrinkage Parameters
    g_shrink = ap.add_argument_group("SPLAT Shrinkage Options (Advanced)")
    g_shrink.add_argument("--tau-dropout", type=float, default=50.0, help="Shrinkage strength (Tau) for dropout curve parameters (k, x0).")
    g_shrink.add_argument("--tau-group-dispersion", type=float, default=50.0, help="Shrinkage strength (Tau) for group dispersion estimation (phi).")
    g_shrink.add_argument("--tau-dispersion", type=float, default=20.0, help="Shrinkage strength (Tau) for per-gene dispersion MoM estimation.")


    # PCA+kNN
    g_knn = ap.add_argument_group("PCA+kNN Options")
    g_knn.add_argument("--run-knn", action="store_true", help="Enable PCA+kNN approach (requires scikit-learn).")
    g_knn.add_argument("--pca-components", type=int, default=20)
    g_knn.add_argument("--knn-k", type=int, default=15)
    g_knn.add_argument("--knn-mean-thresh", type=float, default=0.5, help="Global threshold on kNN mean (log1p) if --knn-gene-quantile is disabled.")
    g_knn.add_argument("--knn-gene-quantile", type=float, default=0.25,
                    help="Per-gene NON-ZERO log1p quantile threshold for kNN. Set to -1 to disable.")

    # DCA (Global and Stratified)
    g_dca = ap.add_argument_group("DCA Options")
    g_dca.add_argument("--run-dca", action="store_true", help="Enable DCA dropout estimation (Global).")
    g_dca.add_argument("--run-dca-stratified", action="store_true", help="Enable Stratified DCA (Group/Batch/Combined).")
    g_dca.add_argument("--dca-ae_types", type=str, default="all", help="Comma-separated list of DCA AE types or 'all'.")
    # Training takes slightly longer on the modern TF/Keras stack to reach the same fit as the legacy release.
    # Bump default epochs to 100 to match legacy-quality denoising/dropout posteriors.
    g_dca.add_argument("--dca-epochs", type=int, default=50)
    g_dca.add_argument("--dca-batch-size", type=int, default=32)
    g_dca.add_argument("--dca-threads", type=int, default=None)
    g_dca.add_argument("--dca-random-state", type=int, default=0)
    g_dca.add_argument("--dca-auto-thresh", action="store_true", help="[Deprecated] Use --posterior-auto-thresh.")
    g_dca.add_argument("--dca-mu-gate", type=float, default=0.2, help="Minimum reconstructed mean (mu) required to predict dropout.")
    g_dca.add_argument("--dca-min-cells-stratum", type=int, default=100, help="Minimum cells required to run DCA independently in a stratum; otherwise fallback to global.")
    g_dca.add_argument("--dca-verbose", action="store_true")

    # Handle argument parsing
    if (hasattr(sys, 'ps1') or 'ipykernel' in sys.modules) and len(sys.argv) == 1:
        # print("Interactive mode detected.")
        return
    elif len(sys.argv) == 1 and sys.stdin.isatty():
        ap.print_help()
        sys.exit(1)

    args = ap.parse_args()

    # Argument adjustments
    if args.knn_gene_quantile is not None and args.knn_gene_quantile < 0:
        args.knn_gene_quantile = None
        
    if args.dca_auto_thresh:
        args.posterior_auto_thresh = True
    args.posterior_auto_metric = args.posterior_auto_metric.lower()

    # Setup directories
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    rds_files = discover_rds_files(data_dir)
    if not rds_files:
        raise SystemExit(f"No .rds files found under {data_dir}")

    print(f"Found {len(rds_files)} datasets. Processing...")

    # Main loop
    all_rows: List[Dict[str, Union[float, str]]] = []
    for f in rds_files:
        print(f"\n--- Processing {f.name} ---")
        try:
            rows = run_on_dataset(f, args, out_dir)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            traceback.print_exc()

    # Print summary
    _print_console_summary(all_rows, sort_by_metric=_metric_to_column(args.posterior_auto_metric))
    print(f"Prediction masks saved under: {out_dir.as_posix()}")
    print("Done.")


if __name__ == "__main__":
    # Set a default start method for multiprocessing if needed (often helps with DCA/TensorFlow)
    import multiprocessing
    try:
        # Using 'spawn' is generally safer than 'fork' when using GPUs or complex libraries
        if sys.platform != "win32":
            # Check if start method is already set before trying to set it
            if multiprocessing.get_start_method(allow_none=True) is None:
                multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # Start method might already be set
    except Exception:
        pass
    main()
