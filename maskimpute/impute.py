"""Public selective imputation entry point for MaskImpute v27."""

from __future__ import annotations

import math

import numpy as np
import torch

from maskimpute.config import MaskImputeConfig
from maskimpute.count_model import PreZeroCountModelScore
from maskimpute.result import ImputationResult
from maskimpute.train import (
    _numeric_matrix_to_dense,
    invert_observed_normalization,
    train_v27,
    validate_observed_counts,
    validate_p_pre_zero,
)


def inference_availability_mask(observed_counts: object) -> np.ndarray:
    """Mark observed positives available and natural observed zeros unavailable."""

    counts = validate_observed_counts(observed_counts)
    return counts > 0


def apply_zero_gate(
    candidates: object,
    p_pre_zero: object,
    *,
    gamma: float,
) -> np.ndarray:
    """Apply the monotone power-complement gate on count scale."""

    predictions, _ = _numeric_matrix_to_dense(candidates, "candidates")
    probability, _ = _numeric_matrix_to_dense(p_pre_zero, "p_pre_zero")
    if predictions.ndim != 2 or probability.shape != predictions.shape:
        raise ValueError("candidate and p_pre_zero shapes must match in two dimensions")
    if not np.all(np.isfinite(predictions)) or np.any(predictions < 0):
        raise ValueError("candidates must be finite and nonnegative")
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("p_pre_zero must be finite and lie in [0, 1]")
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float, np.number)):
        raise TypeError("gamma must be a real number")
    exponent = float(gamma)
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError("gamma must be finite and nonnegative")
    return predictions * np.power(1.0 - probability, exponent)


def impute_counts(
    observed_counts: object,
    p_pre_zero: object,
    config: MaskImputeConfig = MaskImputeConfig(),
    device: str | torch.device = "cpu",
    *,
    cell_ids: object | None = None,
) -> ImputationResult:
    """Fit v27 and selectively impute observed zeros.

    Parameters are intentionally limited to raw observed counts and an external
    count-model score.  Exact verified score artifacts additionally require the
    same external ``cell_ids`` used during score fitting.  No evaluator truth,
    labels, annotations, or reconstruction-derived score can enter this interface.
    """

    if not isinstance(config, MaskImputeConfig):
        raise TypeError("config must be a MaskImputeConfig")
    counts = validate_observed_counts(observed_counts)
    if type(p_pre_zero) is PreZeroCountModelScore:
        probability = validate_p_pre_zero(
            p_pre_zero.score_for_counts(counts, cell_ids),
            counts,
        )
        score_manifest = p_pre_zero.manifest
        cell_identity = score_manifest["cell_identity"]
        score_diagnostics = {
            "score_source": "maskimpute_cross_fitted_count_only_p_pre_zero",
            "score_provenance_verified": True,
            "score_provenance": {
                "artifact_type": "maskimpute_count_model_score",
                "cell_ids_sha256": cell_identity["digest_sha256"],
                "cell_id_source": cell_identity["source"],
                "config_sha256": score_manifest["config_sha256"],
                "cross_fitting": cell_identity["assignment"],
                "effective_folds": score_manifest["cross_fitting"]["effective_folds"],
                "fit_inputs": ("observed_counts", "cell_ids"),
                "input_sha256": score_manifest["input_sha256"],
                "score_sha256": score_manifest["score_sha256"],
            },
        }
    else:
        probability = validate_p_pre_zero(p_pre_zero, counts)
        score_diagnostics = {
            "score_source": "caller_supplied_count_model_p_pre_zero_unverified",
            "score_provenance_verified": False,
        }
    outcome = train_v27(counts, probability, config, device)

    selected_device = next(outcome.model.parameters()).device
    expression = torch.as_tensor(
        outcome.normalized_expression,
        dtype=torch.float32,
        device=selected_device,
    )
    availability = torch.as_tensor(
        counts > 0,
        dtype=torch.bool,
        device=selected_device,
    )
    outcome.model.eval()
    with torch.no_grad():
        normalized_prediction, latent = outcome.model(expression, availability)

    normalized_dense = normalized_prediction.detach().cpu().numpy().astype(np.float64)
    latent_dense = latent.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(normalized_dense)) or np.any(normalized_dense < 0):
        raise FloatingPointError("v27 decoder produced invalid normalized predictions")
    if not np.all(np.isfinite(latent_dense)):
        raise FloatingPointError("v27 encoder produced invalid latent values")

    with np.errstate(over="ignore", invalid="ignore"):
        denoised_counts = invert_observed_normalization(
            normalized_dense,
            outcome.library_sizes,
            target=config.normalization_target,
        )
    if not np.all(np.isfinite(denoised_counts)) or np.any(denoised_counts < 0):
        raise FloatingPointError("v27 inverse normalization produced invalid counts")

    selective_counts = apply_zero_gate(
        denoised_counts,
        probability,
        gamma=config.gate_gamma,
    )
    observed_positive = counts > 0
    selective_counts[observed_positive] = counts[observed_positive]

    diagnostics = {
        "method_version": "v27",
        **score_diagnostics,
        "normalization": {
            "target_formula": (
                "log1p(observed_count / full_observed_library_size * target)"
            ),
            "encoder_formula": (
                "log1p(available_observed_count / corrupted_available_library_size "
                "* target)"
            ),
            "inverse": "expm1(value) * observed_library_size / target",
            "target": config.normalization_target,
            "zero_library_policy": "preserve_all_zero",
            "zero_library_cells": int(np.count_nonzero(outcome.library_sizes == 0)),
        },
        "masks": {
            "inference_unavailable": "observed_count_equals_zero",
            "inference_unavailable_entries": int(np.count_nonzero(~observed_positive)),
            "fixed_validation_positive_entries": int(
                np.count_nonzero(outcome.validation_mask)
            ),
            "fixed_validation_mask_sha256": outcome.validation_mask_hashes[0],
            "epoch_training_mask_sha256": outcome.epoch_training_mask_hashes,
            "epoch_mask_strata": tuple(config.log_count_bin_edges),
        },
        "losses": {
            "primary": "artificially_masked_observed_positive_mse",
            "natural_zero_penalty": "mean(p_pre_zero * normalized_prediction_squared)",
            "natural_zero_penalty_weight": config.pre_zero_regularization,
            "training": outcome.training_loss_history,
            "validation": outcome.validation_loss_history,
            "validation_criterion": "fixed_artificial_positive_mse",
        },
        "early_stopping": {
            "best_epoch": outcome.best_epoch,
            "stopped_epoch": outcome.stopped_epoch,
            "patience": config.patience,
            "minimum_improvement": config.early_stopping_min_delta,
        },
        "randomness": {
            "model_seed": config.seed,
            "validation_mask_seed": outcome.validation_seed,
            "epoch_mask_seed": outcome.training_seed,
            "deterministic_algorithms": outcome.deterministic_algorithms,
            "caller_rng_state_restored": outcome.caller_rng_state_restored,
            "cublas_workspace_config": outcome.cublas_workspace_config,
        },
        "device": outcome.device,
        "gate": {
            "family": "power_complement",
            "formula": "prediction * (1 - p_pre_zero) ** gamma",
            "gamma": config.gate_gamma,
        },
    }
    return ImputationResult(
        selective_counts=selective_counts,
        denoised_counts=denoised_counts,
        p_pre_zero=probability,
        latent=latent_dense,
        diagnostics=diagnostics,
    )
