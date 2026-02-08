#!/usr/bin/env python3
"""
maskclass3.py

Balanced_mse backend variant that explicitly keeps the non-ablated settings for:
- cell_zero_weight (not old_cell_zero_weight)
- p_nz (not old_p_nz)
- noise_max (not old_noise_max)
- graph_blend (not no_graph_refine)
- diffusion_blend (not no_diffusion_refine)
- cluster_blend (not no_cluster_refine)
- biozero_shrink (not no_biozero_shrink)
"""

from __future__ import annotations

import json
import os

import maskclass as _base

# Re-export parameter dictionaries (same underlying objects).
BIO_PARAMS = _base.BIO_PARAMS
MODEL_PARAMS = _base.MODEL_PARAMS
AE_PARAMS = _base.AE_PARAMS
SCALER_PARAMS = _base.SCALER_PARAMS
POSTPROCESS_PARAMS = _base.POSTPROCESS_PARAMS

# Best shared configuration found from GPU sweeps under the current constraints:
# - single pipeline/settings for all datasets
# - no competitor method routing
# - preserves cells_100 MSE targets while improving ARI wins+ties
BIO_PARAMS["non_umi_blend"] = 0.64
BIO_PARAMS["cell_zero_weight"] = 0.20
AE_PARAMS["p_nz"] = 0.40
AE_PARAMS["noise_max"] = 0.12
POSTPROCESS_PARAMS["graph_blend"] = 1.0
POSTPROCESS_PARAMS["diffusion_blend"] = 0.55
POSTPROCESS_PARAMS["cluster_blend"] = 0.28
POSTPROCESS_PARAMS["cluster_k_min"] = 9
POSTPROCESS_PARAMS["cluster_k_max"] = 9
POSTPROCESS_PARAMS["cluster_k_penalty"] = 0.0
POSTPROCESS_PARAMS["biozero_shrink"] = 0.08


def _apply_overrides(target: dict, updates: dict) -> None:
    for k, v in updates.items():
        if k in target:
            target[k] = v


def _apply_maskclass3_env() -> None:
    raw = os.environ.get("MASKCLASS3_CONFIG_JSON")
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
    if isinstance(cfg.get("POSTPROCESS_PARAMS"), dict):
        _apply_overrides(POSTPROCESS_PARAMS, cfg["POSTPROCESS_PARAMS"])


_apply_maskclass3_env()

# Re-export public API expected by run_imputation.py / run_clustering.py.
EPSILON = _base.EPSILON
splat_cellaware_bio_prob = _base.splat_cellaware_bio_prob
train_autoencoder_reconstruct = _base.train_autoencoder_reconstruct
set_seed = _base.set_seed
