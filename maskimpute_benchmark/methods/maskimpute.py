"""In-tree MaskImpute and capacity-control benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
import platform

import numpy as np
import scipy
import torch

from maskimpute import (
    MaskImputeConfig,
    PreZeroCountModelConfig,
    fit_p_pre_zero_count_model,
)
from maskimpute.ablations import (
    _TRACKED_ABLATION_REGISTRY,
    AblationRunResult,
    _fit_ablation_once,
    load_ablation_registry,
)
from maskimpute.calibration import CalibrationArtifact

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .observed import (
    AdapterExecution,
    CompatibilityEvent,
    raw_output_to_count_equivalent,
    require_method_spec,
)


@dataclass(frozen=True, slots=True)
class MaskImputeAdapterExecution(AdapterExecution):
    """Bound primary output plus immutable score/denoising diagnostics."""

    ablation_result: AblationRunResult


def maskimpute_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Declare MaskImpute's primary raw output as count equivalents."""

    return raw_output_to_count_equivalent(method_input, native_output)


def finalize_maskimpute_output(
    spec: MethodSpec,
    method_input: MethodInput,
    primary_counts: object,
) -> MethodOutputSnapshot:
    """Bind one in-tree primary output to the evaluator dataset and IDs."""

    require_method_spec(
        spec,
        spec.id,
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    if spec.id not in {"maskimpute", "capacity-matched-ae"}:
        raise ValueError(
            "in-tree adapter accepts only MaskImpute or its matched control"
        )
    return snapshot_method_output(
        spec,
        method_input,
        primary_counts,
        source_dataset_sha256=method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def _run_in_tree(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    variant_id: str,
    calibration_artifact: CalibrationArtifact,
    seed: int,
    config: MaskImputeConfig,
    count_model_config: PreZeroCountModelConfig,
    device: str | torch.device,
    development_mechanism: str,
    development_biological_id: str,
) -> MaskImputeAdapterExecution:
    require_method_spec(
        spec,
        spec.id,
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    expected_method = (
        "maskimpute" if variant_id == "maskimpute-reference" else "capacity-matched-ae"
    )
    if spec.id != expected_method:
        raise ValueError("method specification and ablation variant differ")
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if type(calibration_artifact) is not CalibrationArtifact:
        raise TypeError("calibration_artifact must be an exact CalibrationArtifact")
    if type(config) is not MaskImputeConfig:
        raise TypeError("config must be an exact MaskImputeConfig")
    config = replace(config)
    if isinstance(seed, bool) or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2^63)")
    if seed != config.seed:
        raise ValueError("seed must equal the bound MaskImpute config seed")
    if type(count_model_config) is not PreZeroCountModelConfig:
        raise TypeError("count_model_config must be an exact PreZeroCountModelConfig")

    registry = load_ablation_registry(_TRACKED_ABLATION_REGISTRY)
    specifications = {registry.reference.id: registry.reference, **registry.by_id}
    try:
        ablation_spec = specifications[variant_id]
    except KeyError as error:  # pragma: no cover - internal closed dispatch
        raise RuntimeError("tracked in-tree ablation variant is unavailable") from error

    counts = method_input.counts
    score = fit_p_pre_zero_count_model(
        counts,
        method_input.obs_ids,
        count_model_config,
    )
    result = _fit_ablation_once(
        counts,
        score,
        calibration_artifact,
        ablation_spec,
        config,
        device,
        cell_ids=method_input.obs_ids,
        development_mechanism=development_mechanism,
        development_biological_id=development_biological_id,
    )
    snapshot = finalize_maskimpute_output(
        spec,
        method_input,
        result.primary_counts,
    )
    selected_device = str(torch.device(device))
    compatibility = (
        CompatibilityEvent(
            "truth_free_score",
            "cross-fitted count-only p_pre_zero was rederived from MethodInput counts and external cell IDs",
        ),
        CompatibilityEvent(
            "primary_output_policy",
            f"tracked ablation policy={result.output_policy}; denoised output remains separate",
        ),
        CompatibilityEvent(
            "development_authority_boundary",
            "adapter validates artifact integrity and applies the held-out draw calibrator for SymSim; the evaluator runner separately authorizes dataset/config/score/calibration hashes and the complete run grid",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "primary raw-count output is a count equivalent and receives the shared log2(CP10k+1) evaluator transform",
        ),
    )
    environment_receipt = tuple(
        sorted(
            {
                "device": selected_device,
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "scipy_version": scipy.__version__,
                "torch_version": torch.__version__,
            }.items()
        )
    )
    return MaskImputeAdapterExecution(
        snapshot=snapshot,
        compatibility_log=compatibility,
        environment_receipt=environment_receipt,
        stdout=b"",
        stderr=b"",
        command=None,
        ablation_result=result,
    )


def run_maskimpute(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    calibration_artifact: CalibrationArtifact,
    seed: int,
    config: MaskImputeConfig,
    count_model_config: PreZeroCountModelConfig,
    device: str | torch.device,
    development_mechanism: str,
    development_biological_id: str,
) -> MaskImputeAdapterExecution:
    """Run the retained-calibrator selective v27 candidate on a truth-free input."""

    return _run_in_tree(
        spec,
        method_input,
        variant_id="maskimpute-reference",
        calibration_artifact=calibration_artifact,
        seed=seed,
        config=config,
        count_model_config=count_model_config,
        device=device,
        development_mechanism=development_mechanism,
        development_biological_id=development_biological_id,
    )


def run_capacity_matched_ae(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    calibration_artifact: CalibrationArtifact,
    seed: int,
    config: MaskImputeConfig,
    count_model_config: PreZeroCountModelConfig,
    device: str | torch.device,
    development_mechanism: str,
    development_biological_id: str,
) -> MaskImputeAdapterExecution:
    """Run the prespecified full-output capacity-matched masked-AE control."""

    return _run_in_tree(
        spec,
        method_input,
        variant_id="capacity-matched-ae",
        calibration_artifact=calibration_artifact,
        seed=seed,
        config=config,
        count_model_config=count_model_config,
        device=device,
        development_mechanism=development_mechanism,
        development_biological_id=development_biological_id,
    )


__all__ = [
    "MaskImputeAdapterExecution",
    "finalize_maskimpute_output",
    "maskimpute_to_evaluator_counts",
    "run_capacity_matched_ae",
    "run_maskimpute",
]
