#!/usr/bin/env python3
"""
maskclass2.py

Second-generation balanced_mse backend.
This module re-exports the tuned maskclass implementation so benchmarking code
can select it explicitly without changing evaluation logic.
"""

from __future__ import annotations

import maskclass_final as _base

# Re-export shared parameter dictionaries (same underlying objects).
BIO_PARAMS = _base.BIO_PARAMS
MODEL_PARAMS = _base.MODEL_PARAMS
AE_PARAMS = _base.AE_PARAMS
SCALER_PARAMS = _base.SCALER_PARAMS
POSTPROCESS_PARAMS = _base.POSTPROCESS_PARAMS

# Re-export public API expected by run_imputation.py / run_clustering.py.
EPSILON = _base.EPSILON
splat_cellaware_bio_prob = _base.splat_cellaware_bio_prob
train_autoencoder_reconstruct = _base.train_autoencoder_reconstruct
set_seed = _base.set_seed
