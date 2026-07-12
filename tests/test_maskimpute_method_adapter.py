from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_benchmark_method_registry_import_does_not_require_torch():
    script = r"""
import importlib.abc
import sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch import blocked by test")
        return None
sys.meta_path.insert(0, BlockTorch())
import maskimpute_benchmark.methods as methods
assert "maskimpute" in methods.CORE_EVALUATOR_COUNT_CONVERTERS
assert "torch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _method_input(counts: np.ndarray):
    from maskimpute_benchmark.methods.base import MethodInput

    return MethodInput(
        source_dataset_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        obs_ids=tuple(f"cell-{index}" for index in range(counts.shape[0])),
        var_ids=tuple(f"gene-{index}" for index in range(counts.shape[1])),
        shape=counts.shape,
        obs_covariates=(),
        var_covariates=(),
        _count_bytes=np.asarray(counts, dtype="<f8", order="C").tobytes(order="C"),
        _normalization_bytes=b"{}",
    )


def _identity_calibration_artifact():
    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
        fit_development_calibration,
    )

    records = []
    for index, (draw, view) in enumerate(
        (
            ("draw-01", "moderate"),
            ("draw-01", "severe"),
            ("draw-02", "moderate"),
            ("draw-02", "severe"),
        ),
        start=1,
    ):
        dataset_sha = hashlib.sha256(f"{draw}:{view}".encode()).hexdigest()
        records.append(
            CalibrationRecord(
                p_pre_zero=(0.1, 0.25, 0.7, 0.9),
                target=(0, 0, 1, 1),
                mechanism="symsim",
                biological_id=draw,
                manifest_sha256=f"{index:x}" * 64,
                truth_kind="exact_pre_capture",
                namespace="dev",
                data_role="development",
                technical_view=view,
                dataset_id=f"dataset-{dataset_sha[:24]}",
                dataset_sha256=dataset_sha,
                protocol_sha256=DEVELOPMENT_PROTOCOL_SHA256,
            )
        )
    return fit_development_calibration(records)


def test_maskimpute_and_capacity_control_adapters_execute_bound_outputs():
    from maskimpute import MaskImputeConfig, PreZeroCountModelConfig
    from maskimpute_benchmark.methods.maskimpute import (
        MaskImputeAdapterExecution,
        maskimpute_to_evaluator_counts,
        run_capacity_matched_ae,
        run_maskimpute,
    )
    from maskimpute_benchmark.methods import core_output_to_evaluator_counts
    from maskimpute_benchmark.methods.registry import load_method_registry

    counts = np.array(
        [
            [5, 0, 1, 0],
            [2, 3, 0, 1],
            [0, 4, 2, 1],
            [1, 0, 3, 2],
            [4, 1, 0, 2],
            [3, 2, 1, 0],
            [2, 0, 4, 1],
            [1, 3, 2, 0],
        ],
        dtype=np.int64,
    )
    method_input = _method_input(counts)
    methods = load_method_registry(Path("study/methods.json"))
    config = MaskImputeConfig(
        hidden_dims=(7, 5),
        latent_dim=3,
        batch_size=4,
        max_epochs=2,
        patience=2,
        seed=42,
    )
    score_config = PreZeroCountModelConfig(n_folds=2, link_max_iter=25)
    calibration = _identity_calibration_artifact()

    candidate = run_maskimpute(
        methods.by_id("maskimpute"),
        method_input,
        calibration_artifact=calibration,
        seed=42,
        config=config,
        count_model_config=score_config,
        device="cpu",
        development_mechanism="symsim",
        development_biological_id="draw-01",
    )
    control = run_capacity_matched_ae(
        methods.by_id("capacity-matched-ae"),
        method_input,
        calibration_artifact=calibration,
        seed=42,
        config=config,
        count_model_config=score_config,
        device="cpu",
        development_mechanism="symsim",
        development_biological_id="draw-01",
    )

    assert isinstance(candidate, MaskImputeAdapterExecution)
    assert candidate.snapshot.method_id == "maskimpute"
    assert (
        candidate.snapshot.source_dataset_sha256 == method_input.source_dataset_sha256
    )
    np.testing.assert_array_equal(
        candidate.snapshot.matrix[counts > 0], counts[counts > 0]
    )
    np.testing.assert_array_equal(
        maskimpute_to_evaluator_counts(method_input, candidate.snapshot.matrix),
        candidate.snapshot.matrix,
    )
    np.testing.assert_array_equal(
        core_output_to_evaluator_counts(method_input, candidate.snapshot),
        candidate.snapshot.matrix,
    )
    assert candidate.ablation_result.output_policy == "selective"
    assert candidate.ablation_result.p_pre_zero.shape == counts.shape
    assert candidate.ablation_result.diagnostics["score"]["artifact_integrity_verified"]
    assert candidate.ablation_result.diagnostics["score"]["calibration_scope"] == (
        "leave_one_biological_draw_out"
    )
    assert not candidate.ablation_result.diagnostics["score"][
        "source_authorized_by_panel"
    ]
    assert control.snapshot.method_id == "capacity-matched-ae"
    assert control.ablation_result.output_policy == "full_ungated"
    assert control.snapshot.matrix.shape == counts.shape
    np.testing.assert_array_equal(
        core_output_to_evaluator_counts(method_input, control.snapshot),
        control.snapshot.matrix,
    )
    assert candidate.command is None and control.command is None
    assert dict(candidate.environment_receipt)["device"] == "cpu"


def test_maskimpute_adapter_rejects_seed_or_artifact_drift():
    from maskimpute import MaskImputeConfig, PreZeroCountModelConfig
    from maskimpute_benchmark.methods.maskimpute import run_maskimpute
    from maskimpute_benchmark.methods.registry import load_method_registry

    counts = np.array([[2, 0], [1, 3]], dtype=np.int64)
    method_input = _method_input(counts)
    spec = load_method_registry(Path("study/methods.json")).by_id("maskimpute")
    config = MaskImputeConfig(max_epochs=1, patience=1, seed=42)

    with pytest.raises(ValueError, match="seed.*config"):
        run_maskimpute(
            spec,
            method_input,
            calibration_artifact=_identity_calibration_artifact(),
            seed=43,
            config=config,
            count_model_config=PreZeroCountModelConfig(n_folds=2),
            device="cpu",
            development_mechanism="symsim",
            development_biological_id="draw-01",
        )
    with pytest.raises(TypeError, match="CalibrationArtifact"):
        run_maskimpute(
            spec,
            method_input,
            calibration_artifact=object(),
            seed=42,
            config=config,
            count_model_config=PreZeroCountModelConfig(n_folds=2),
            device="cpu",
            development_mechanism="symsim",
            development_biological_id="draw-01",
        )
