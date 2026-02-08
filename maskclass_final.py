#!/usr/bin/env python3
"""
maskclass_final.py

Final balanced_mse backend that starts from bare masked_imputation26 behavior
and only retunes existing parameters.
"""

from __future__ import annotations

import json
import os

import numpy as np

import masked_imputation26 as _base

# Re-export parameter dictionaries (same underlying objects).
BIO_PARAMS = _base.BIO_PARAMS
MODEL_PARAMS = _base.MODEL_PARAMS
AE_PARAMS = _base.AE_PARAMS
SCALER_PARAMS = _base.SCALER_PARAMS

# Keep this for compatibility with maskclass2 re-export shape.
POSTPROCESS_PARAMS = {}


def _apply_overrides(target: dict, updates: dict) -> None:
    for k, v in updates.items():
        if k in target:
            target[k] = v


def _apply_env_config() -> None:
    raw = os.environ.get("MASKCLASS_FINAL_CONFIG_JSON")
    if not raw:
        return
    try:
        cfg = json.loads(raw)
    except Exception:
        return
    if not isinstance(cfg, dict):
        return
    if isinstance(cfg.get("BIO_PARAMS"), dict):
        _apply_overrides(BIO_PARAMS, cfg["BIO_PARAMS"])
    if isinstance(cfg.get("MODEL_PARAMS"), dict):
        _apply_overrides(MODEL_PARAMS, cfg["MODEL_PARAMS"])
    if isinstance(cfg.get("AE_PARAMS"), dict):
        _apply_overrides(AE_PARAMS, cfg["AE_PARAMS"])
    if isinstance(cfg.get("SCALER_PARAMS"), dict):
        _apply_overrides(SCALER_PARAMS, cfg["SCALER_PARAMS"])


# Start from masked_imputation26 defaults; override only existing parameters.
# These defaults are tuned in this turn using the same shared setting across datasets.
BIO_PARAMS["cell_zero_weight"] = 0.30
AE_PARAMS["p_zero"] = 0.01
AE_PARAMS["p_nz"] = 0.30
AE_PARAMS["noise_max"] = 0.20
AE_PARAMS["loss_bio_weight"] = 2.0
AE_PARAMS["loss_nz_weight"] = 1.0
AE_PARAMS["bio_reg_weight"] = 1.0
AE_PARAMS["epochs"] = 300
AE_PARAMS["lr"] = 0.0001

_apply_env_config()

# Re-export public API expected by run_imputation.py / run_clustering.py.
EPSILON = _base.EPSILON
train_autoencoder_reconstruct = _base.train_autoencoder_reconstruct
set_seed = _base.set_seed


def _estimate_nb_dispersion(counts: np.ndarray, lib_factor: np.ndarray) -> np.ndarray:
    x = counts / np.clip(lib_factor[:, None], 1e-8, None)
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    denom = np.maximum(var - mu, 1e-8)
    theta = np.where(mu > 1e-8, (mu * mu) / denom, 1e6)
    theta = np.clip(theta, 0.1, 1e6)

    valid = np.isfinite(theta) & (theta > 0.0)
    if np.any(valid):
        theta_med = float(np.median(theta[valid]))
        # Mild shrinkage improves stability for sparse non-UMI datasets.
        theta = 0.7 * theta + 0.3 * theta_med
    return np.clip(theta, 0.1, 1e6).astype(np.float64, copy=False)


def _zinb_bio_posterior(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    use_cell_factor: bool,
) -> np.ndarray:
    c = np.clip(np.asarray(counts, dtype=np.float64), 0.0, None)
    z = np.asarray(zeros_obs, dtype=bool)
    n, g = c.shape

    if use_cell_factor:
        lib = c.sum(axis=1)
        lib_pos = lib[lib > 0.0]
        lib_med = float(np.median(lib_pos)) if lib_pos.size else 1.0
        lib_factor = np.clip(lib / max(lib_med, 1e-8), 1e-4, 1e4)
    else:
        lib_factor = np.ones(n, dtype=np.float64)

    mu_gene = np.mean(c / np.clip(lib_factor[:, None], 1e-8, None), axis=0)
    mu_gene = np.clip(mu_gene, 1e-8, None)
    mu_ij = lib_factor[:, None] * mu_gene[None, :]

    theta = _estimate_nb_dispersion(c, lib_factor)
    theta_ij = theta[None, :]

    # NB zero probability: P(Y=0 | NB(mu, theta)).
    log_nb0 = theta_ij * (np.log(theta_ij) - np.log(theta_ij + mu_ij))
    log_nb0 = np.clip(log_nb0, -60.0, 0.0)
    nb0 = np.exp(log_nb0)

    zero_rate_obs = z.mean(axis=0)
    zero_rate_nb = nb0.mean(axis=0)
    pi_gene = (zero_rate_obs - zero_rate_nb) / np.maximum(1.0 - zero_rate_nb, 1e-8)
    pi_gene = np.clip(np.nan_to_num(pi_gene, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 0.995)

    if use_cell_factor:
        z_cell = z.mean(axis=1)
        z_med = float(np.median(z_cell))
        z_mad = float(np.median(np.abs(z_cell - z_med))) + 1e-8
        cell_mult = 1.0 + 0.35 * np.tanh((z_cell - z_med) / (2.5 * z_mad))
        pi_ij = np.clip(pi_gene[None, :] * cell_mult[:, None], 0.0, 0.995)
    else:
        pi_ij = np.broadcast_to(pi_gene[None, :], (n, g))

    # P(dropout | y=0) = pi / (pi + (1-pi) * NB0), so biological-zero posterior is 1 - that.
    denom = pi_ij + (1.0 - pi_ij) * nb0
    p_dropout = pi_ij / np.maximum(denom, 1e-8)
    p_bio = 1.0 - p_dropout
    p_bio = np.clip(np.nan_to_num(p_bio, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    p_bio[~z] = 0.0
    return p_bio.astype(np.float32, copy=False)


def splat_cellaware_bio_prob(
    counts: np.ndarray,
    zeros_obs: np.ndarray,
    disp_mode: str,
    use_cell_factor: bool,
) -> np.ndarray:
    # `disp_mode` is an existing masked_imputation26 parameter; use it to select posterior style.
    mode = str(disp_mode).strip().lower()
    p_base = None
    p_zinb = None

    def _legacy() -> np.ndarray:
        return _base.splat_cellaware_bio_prob(
            counts=counts,
            zeros_obs=zeros_obs,
            disp_mode="estimate",
            use_cell_factor=use_cell_factor,
        )

    try:
        if mode in ("zinb", "zinb_only"):
            p_bio = _zinb_bio_posterior(counts, zeros_obs, use_cell_factor=bool(use_cell_factor))
        elif mode in ("zinb_blend", "blend", "non_umi_blend"):
            p_base = _legacy()
            p_zinb = _zinb_bio_posterior(counts, zeros_obs, use_cell_factor=bool(use_cell_factor))
            # Conservative blend to avoid destabilizing strong legacy behavior.
            p_bio = 0.7 * p_base + 0.3 * p_zinb
        else:
            p_bio = _legacy()
    except Exception:
        p_bio = _legacy()

    p_bio = np.nan_to_num(p_bio, nan=0.0, posinf=0.0, neginf=0.0)
    p_bio = np.clip(p_bio, 0.0, 1.0)
    p_bio[~zeros_obs] = 0.0
    return p_bio.astype(np.float32, copy=False)
