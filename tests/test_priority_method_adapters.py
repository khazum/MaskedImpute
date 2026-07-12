from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import boxcox

import maskimpute_benchmark.methods as benchmark_methods
from maskimpute_benchmark.methods import load_method_registry, prepare_method_input
from maskimpute_benchmark.methods.observed import AdapterUnavailableError


def _adapter_module(name: str):
    try:
        return importlib.import_module(f"maskimpute_benchmark.methods.{name}")
    except ModuleNotFoundError:
        return SimpleNamespace()


afmf_adapter = _adapter_module("afmf")
biaeimpute_adapter = _adapter_module("biaeimpute")
d3impute_adapter = _adapter_module("d3impute")
scziva_adapter = _adapter_module("scziva")


METHODS_PATH = Path("study/methods.json")
SOURCE_ROOT = Path("artifacts/method-sources")
SOURCE_SHA = "d" * 64


def _method_input(*, cells: int = 30, genes: int = 20):
    counts = np.empty((cells, genes), dtype=np.int64)
    for cell in range(cells):
        for gene in range(genes):
            value = 1 + ((cell * 3 + gene * 5) % 11)
            counts[cell, gene] = 0 if (cell + gene * 2) % 7 == 0 else value
    view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(cells)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(genes)]),
    )
    view.uns["source_dataset_sha256"] = SOURCE_SHA
    view.uns["allowed_covariates"] = {"obs": [], "var": []}
    view.uns["normalization"] = {
        "input": "counts",
        "target_sum": None,
        "log_base": None,
        "size_factor": None,
    }
    return prepare_method_input(view)


def _bulk_reference(method_input, *, samples: int = 5):
    matrix = np.empty((method_input.shape[1], samples), dtype=np.float64)
    for gene in range(method_input.shape[1]):
        for sample in range(samples):
            matrix[gene, sample] = 5.5 + ((gene * 7 + sample * 3) % 19)
    return d3impute_adapter.prepare_matched_bulk_reference(
        reference_id="matched-bulk-1",
        source_sha256="e" * 64,
        matrix=matrix,
        var_ids=method_input.var_ids,
        sample_ids=tuple(f"bulk-{index}" for index in range(samples)),
    )


def _registry():
    return load_method_registry(METHODS_PATH)


def _cached_source(source_name: str) -> Path:
    source_dir = SOURCE_ROOT / source_name
    if not source_dir.is_dir():
        pytest.skip(f"ignored pinned-source cache is absent: {source_name}")
    return source_dir


def _assert_snapshot(snapshot, method_id: str, method_input, scale: str) -> None:
    assert snapshot.method_id == method_id
    assert snapshot.source_dataset_sha256 == method_input.source_dataset_sha256
    assert snapshot.obs_ids == method_input.obs_ids
    assert snapshot.var_ids == method_input.var_ids
    assert snapshot.shape == method_input.shape
    assert snapshot.output_scale == scale
    assert snapshot.matrix.flags.writeable is False


def _assert_evaluator_scales(snapshot, method_input) -> None:
    counts = benchmark_methods.recent_output_to_evaluator_counts(method_input, snapshot)
    common_log = benchmark_methods.recent_output_to_evaluator_log2_cp10k(
        method_input, snapshot
    )
    assert counts.shape == method_input.shape
    assert common_log.shape == method_input.shape
    assert np.isfinite(counts).all()
    assert np.isfinite(common_log).all()
    assert counts.flags.writeable is False
    assert common_log.flags.writeable is False


def test_priority_configs_match_pinned_defaults_and_required_adapter_policy() -> None:
    assert scziva_adapter.SCZivaConfig() == scziva_adapter.SCZivaConfig(
        num_epochs=200,
        learning_rate=1e-3,
        hidden_dim=128,
        latent_dim=64,
        use_cnn=True,
        tau=0.001,
        auxiliary_weight_min=0.5,
        auxiliary_weight_max=1.5,
        auxiliary_regularization=1e-3,
        reorder_genes=True,
        device=None,
    )
    assert afmf_adapter.AFMFConfig() == afmf_adapter.AFMFConfig(
        iterations=10_000,
        tolerance=1e-4,
        lambda_p=0.0,
        lambda_q=0.0,
        sigma=3.0,
    )
    assert biaeimpute_adapter.BiAEImputeConfig() == (
        biaeimpute_adapter.BiAEImputeConfig(
            epochs=500,
            latent_size=128,
            learning_rate=0.0002,
            beta1=0.9,
            beta2=0.999,
            row_batch_size=31,
            column_batch_size=200,
            mask_ratio=0.0,
            device=None,
        )
    )
    assert d3impute_adapter.D3ImputeConfig() == d3impute_adapter.D3ImputeConfig(
        neighbors=23,
        latent_dimension=10,
        iterations=100,
        sparsity=0.001,
        cell_regularization=0.1,
        gene_regularization=0.1,
        fixed_seed=42,
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: scziva_adapter.SCZivaConfig(num_epochs=0), "num_epochs"),
        (lambda: scziva_adapter.SCZivaConfig(use_cnn=False), "use_cnn"),
        (lambda: afmf_adapter.AFMFConfig(iterations=0), "iterations"),
        (lambda: afmf_adapter.AFMFConfig(lambda_p=-1), "lambda_p"),
        (lambda: biaeimpute_adapter.BiAEImputeConfig(mask_ratio=0.4), "mask_ratio"),
        (lambda: biaeimpute_adapter.BiAEImputeConfig(epochs=0), "epochs"),
        (lambda: d3impute_adapter.D3ImputeConfig(neighbors=0), "neighbors"),
        (lambda: d3impute_adapter.D3ImputeConfig(fixed_seed=7), "fixed_seed"),
    ],
)
def test_priority_configs_reject_unsupported_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_native_finalizers_retain_declared_scales_and_do_not_copy_entries() -> None:
    method_input = _method_input(cells=8, genes=6)
    registry = _registry()
    raw = np.full(method_input.shape, 0.75)
    normalized = np.full(method_input.shape, 1.25)
    snapshots = [
        scziva_adapter.finalize_scziva_output(
            registry.by_id("scziva"), method_input, raw
        ),
        afmf_adapter.finalize_afmf_output(
            registry.by_id("afmf"), method_input, normalized
        ),
        biaeimpute_adapter.finalize_biaeimpute_output(
            registry.by_id("biaeimpute"), method_input, raw
        ),
        d3impute_adapter.finalize_d3impute_output(
            registry.by_id("d3impute"), method_input, normalized
        ),
    ]
    expected = (
        ("scziva", "raw_counts"),
        ("afmf", "method_native_normalized"),
        ("biaeimpute", "raw_counts"),
        ("d3impute", "external_reference_adjusted"),
    )
    for snapshot, (method_id, scale) in zip(snapshots, expected, strict=True):
        _assert_snapshot(snapshot, method_id, method_input, scale)
    assert snapshots[0].matrix[0, 0] != method_input.counts[0, 0]


def test_recent_converters_are_explicit_and_use_one_shared_log_scale() -> None:
    method_input = _method_input(cells=8, genes=6)
    registry = _registry()
    expected_counts = np.array(method_input.counts, dtype=np.float64, copy=True)
    libraries = method_input.counts.sum(axis=1, keepdims=True)
    afmf_native = np.log1p(expected_counts / libraries * 10_000.0)
    d3_native = np.empty_like(expected_counts)
    for gene in range(expected_counts.shape[1]):
        shifted = expected_counts[:, gene] + 1.0
        d3_native[:, gene] = boxcox(shifted)[0]
    snapshots = {
        "scziva": scziva_adapter.finalize_scziva_output(
            registry.by_id("scziva"), method_input, expected_counts
        ),
        "afmf": afmf_adapter.finalize_afmf_output(
            registry.by_id("afmf"), method_input, afmf_native
        ),
        "biaeimpute": biaeimpute_adapter.finalize_biaeimpute_output(
            registry.by_id("biaeimpute"), method_input, expected_counts
        ),
        "d3impute": d3impute_adapter.finalize_d3impute_output(
            registry.by_id("d3impute"), method_input, d3_native
        ),
    }

    assert set(benchmark_methods.RECENT_EVALUATOR_COUNT_CONVERTERS) == set(snapshots)
    for snapshot in snapshots.values():
        evaluator_counts = benchmark_methods.recent_output_to_evaluator_counts(
            method_input, snapshot
        )
        np.testing.assert_allclose(
            evaluator_counts, expected_counts, rtol=1e-11, atol=1e-11
        )
        _assert_evaluator_scales(snapshot, method_input)


def test_d3impute_count_inverse_rejects_noninvertible_constant_gene() -> None:
    method_input = _method_input(cells=8, genes=6)
    constant_counts = np.array(method_input.counts, copy=True)
    constant_counts[:, 0] = 2
    view = ad.AnnData(
        X=constant_counts,
        obs=pd.DataFrame(index=method_input.obs_ids),
        var=pd.DataFrame(index=method_input.var_ids),
    )
    view.uns["source_dataset_sha256"] = SOURCE_SHA
    view.uns["allowed_covariates"] = {"obs": [], "var": []}
    view.uns["normalization"] = {"input": "counts"}
    constant_input = prepare_method_input(view)

    with pytest.raises(AdapterUnavailableError, match="constant gene") as captured:
        d3impute_adapter.d3impute_to_evaluator_counts(
            constant_input, np.zeros(constant_input.shape)
        )
    assert captured.value.reason_code == "noninvertible_native_scale"


def test_matched_bulk_reference_is_immutable_and_gene_aligned() -> None:
    method_input = _method_input(cells=8, genes=6)
    reference = _bulk_reference(method_input, samples=4)

    assert reference.reference_id == "matched-bulk-1"
    assert reference.source_sha256 == "e" * 64
    assert reference.var_ids == method_input.var_ids
    assert reference.sample_ids == ("bulk-0", "bulk-1", "bulk-2", "bulk-3")
    assert reference.shape == (6, 4)
    assert reference.matrix.flags.writeable is False

    bad_ids = ("wrong", *method_input.var_ids[1:])
    with pytest.raises(ValueError, match="gene IDs"):
        d3impute_adapter.validate_matched_bulk_reference(
            method_input, replace(reference, var_ids=bad_ids)
        )


@pytest.mark.parametrize(
    ("method_id", "source_name"),
    [
        ("scziva", "scziva"),
        ("afmf", "afmf"),
        ("biaeimpute", "biaeimpute"),
        ("d3impute", "d3impute"),
    ],
)
def test_priority_source_boundaries_match_frozen_pins(
    method_id: str, source_name: str
) -> None:
    source_dir = _cached_source(source_name)
    before = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    receipt = benchmark_methods.verify_pinned_source(
        _registry().by_id(method_id), source_dir
    )

    assert receipt.revision == _registry().by_id(method_id).source.revision
    after = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert before == after == ""


def test_missing_priority_environment_fails_before_source_execution(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    missing = tmp_path / "missing-python"

    with pytest.raises(AdapterUnavailableError) as captured:
        scziva_adapter.run_scziva(
            _registry().by_id("scziva"),
            method_input,
            source_dir=SOURCE_ROOT / "scziva",
            python_executable=missing,
            seed=42,
            work_root=tmp_path,
        )
    assert captured.value.reason_code == "environment_executable_missing"
    assert captured.value.command is None


def test_real_pinned_afmf_tiny_smoke(tmp_path: Path) -> None:
    method_input = _method_input(cells=105, genes=105)
    execution = afmf_adapter.run_afmf(
        _registry().by_id("afmf"),
        method_input,
        source_dir=_cached_source("afmf"),
        python_executable=Path(shutil.which("python") or "/missing-python"),
        seed=42,
        config=afmf_adapter.AFMFConfig(iterations=2),
        work_root=tmp_path,
    )

    _assert_snapshot(
        execution.snapshot, "afmf", method_input, "method_native_normalized"
    )
    _assert_evaluator_scales(execution.snapshot, method_input)
    assert execution.command is not None
    assert "-B" in execution.command
    assert "upstream_iteration_override" in {
        event.code for event in execution.compatibility_log
    }


def test_real_pinned_d3impute_tiny_smoke_is_fixed_seed_reproducible(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=30, genes=20)
    reference = _bulk_reference(method_input)
    exact_python = Path("/tmp/maskimpute-d3-conda/bin/python")
    python = (
        exact_python
        if exact_python.is_file()
        else Path(shutil.which("python") or "/missing-python")
    )
    runs = [
        d3impute_adapter.run_d3impute(
            _registry().by_id("d3impute"),
            method_input,
            bulk_reference=reference,
            source_dir=_cached_source("d3impute"),
            python_executable=python,
            config=d3impute_adapter.D3ImputeConfig(iterations=2),
            work_root=tmp_path,
        )
        for _ in range(2)
    ]

    _assert_snapshot(
        runs[0].snapshot,
        "d3impute",
        method_input,
        "external_reference_adjusted",
    )
    _assert_evaluator_scales(runs[0].snapshot, method_input)
    assert runs[0].snapshot.matrix_sha256 == runs[1].snapshot.matrix_sha256
    np.testing.assert_array_equal(runs[0].snapshot.matrix, runs[1].snapshot.matrix)
    assert dict(runs[0].environment_receipt)["bulk_reference_sha256"] == "e" * 64
    if python == exact_python:
        receipt = dict(runs[0].environment_receipt)
        assert receipt["numpy_version"].startswith("2.3.")
        assert receipt["pandas_version"] == "2.3.3"
        assert receipt["scipy_version"].startswith("1.16.")
        assert receipt["sklearn_version"] == "1.7.2"
    assert "fixed_rng_compatibility" in {
        event.code for event in runs[0].compatibility_log
    }


def test_real_pinned_scziva_tiny_smoke_when_torch_environment_exists(
    tmp_path: Path,
) -> None:
    candidates = (
        Path("/tmp/maskimpute-scziva312/bin/python"),
        Path("/home/marcinmaleclocal/miniconda3/envs/magic311/bin/python"),
    )
    python = next((value for value in candidates if value.is_file()), None)
    if python is None:
        pytest.skip("temporary torch environment is unavailable")
    method_input = _method_input(cells=12, genes=8)

    runs = [
        scziva_adapter.run_scziva(
            _registry().by_id("scziva"),
            method_input,
            source_dir=_cached_source("scziva"),
            python_executable=python,
            seed=42,
            config=scziva_adapter.SCZivaConfig(num_epochs=1, device="cpu"),
            work_root=tmp_path,
        )
        for _ in range(2)
    ]
    execution = runs[0]

    _assert_snapshot(execution.snapshot, "scziva", method_input, "raw_counts")
    _assert_evaluator_scales(execution.snapshot, method_input)
    positive = method_input.counts > 0
    np.testing.assert_array_equal(
        execution.snapshot.matrix[positive], method_input.counts[positive]
    )
    assert runs[0].snapshot.matrix_sha256 == runs[1].snapshot.matrix_sha256
    if python == candidates[0]:
        receipt = dict(execution.environment_receipt)
        assert receipt["python_version"] == "3.12.7"
        assert receipt["torch_version"].startswith("2.4.1")
        assert receipt["numpy_version"] == "1.26.4"
    assert "upstream_training_override" in {
        event.code for event in execution.compatibility_log
    }
    assert "unused_preprocessing_dependency_shim" in {
        event.code for event in execution.compatibility_log
    }


def test_real_pinned_biaeimpute_tiny_smoke_when_torch_environment_exists(
    tmp_path: Path,
) -> None:
    python = Path("/home/marcinmaleclocal/miniconda3/envs/magic311/bin/python")
    if not python.is_file():
        pytest.skip("temporary torch environment is unavailable")
    method_input = _method_input(cells=12, genes=8)

    runs = [
        biaeimpute_adapter.run_biaeimpute(
            _registry().by_id("biaeimpute"),
            method_input,
            source_dir=_cached_source("biaeimpute"),
            python_executable=python,
            seed=42,
            config=biaeimpute_adapter.BiAEImputeConfig(epochs=1, device="cpu"),
            work_root=tmp_path,
        )
        for _ in range(2)
    ]
    execution = runs[0]

    _assert_snapshot(execution.snapshot, "biaeimpute", method_input, "raw_counts")
    _assert_evaluator_scales(execution.snapshot, method_input)
    positive = method_input.counts > 0
    np.testing.assert_array_equal(
        execution.snapshot.matrix[positive], method_input.counts[positive]
    )
    assert runs[0].snapshot.matrix_sha256 == runs[1].snapshot.matrix_sha256
    codes = {event.code for event in execution.compatibility_log}
    assert "broken_cli_compatibility" in codes
    assert "upstream_training_override" in codes
