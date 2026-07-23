from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.methods as core_methods
import maskimpute_benchmark.methods.alra as alra_adapter
import maskimpute_benchmark.methods.dca as dca_adapter
import maskimpute_benchmark.methods.magic as magic_adapter
import maskimpute_benchmark.methods.saver as saver_adapter
from maskimpute_benchmark.methods import (
    EnvironmentSpec,
    SourceSpec,
    load_method_registry,
    prepare_method_input,
)
from maskimpute_benchmark.methods.alra import (
    ALRAConfig,
    finalize_alra_output,
    run_alra,
)
from maskimpute_benchmark.methods.dca import (
    DCAConfig,
    finalize_dca_output,
    run_dca,
)
from maskimpute_benchmark.methods.magic import (
    MAGICConfig,
    finalize_magic_output,
    run_magic,
)
from maskimpute_benchmark.methods.observed import (
    AdapterUnavailableError,
    count_equivalent_to_log2_cp10k,
    execute_pinned_command,
    log1p_cp10k,
    log1p_cp10k_to_count_equivalent,
    observed_library_sizes,
    raw_output_to_count_equivalent,
    require_method_spec,
    run_observed,
    verify_pinned_source,
    require_executable,
)
from maskimpute_benchmark.methods.saver import (
    SAVERConfig,
    finalize_saver_output,
    run_saver,
)
from maskimpute_benchmark.methods.scvi import (
    SCVIConfig,
    finalize_scvi_output,
    frequencies_to_observed_library_counts,
    run_scvi,
)


METHODS_PATH = Path("study/methods.json")
SOURCE_ROOT = Path("artifacts/method-sources")
SOURCE_SHA = "d" * 64
SAVER_LOCK_PATH = Path("environments/saver-r.lock.json")
SAVER_BUILD_RECEIPT_PATH = Path("environments/saver-r.build-receipt.json")
SAVER_QUALIFICATION_PATH = Path("environments/saver-r.qualification.json")
DEVELOPMENT_RUNTIME_LOCK_PATH = Path("environments/development-runtime.lock.json")
SAVER_LIBRARY_PATH = Path("/tmp/maskimpute-saver-r461/library")
SAVER_PACKAGE_VERSIONS = {
    "Matrix": "1.7-5",
    "Rcpp": "1.1.2",
    "RcppEigen": "0.3.4.0.2",
    "codetools": "0.2-20",
    "doParallel": "1.0.17",
    "foreach": "1.5.2",
    "glmnet": "5.0",
    "iterators": "1.0.14",
    "lattice": "0.22-9",
    "shape": "1.4.6.1",
    "survival": "3.8-9",
    "SAVER": "1.1.3",
}


def test_saver_registry_runtime_lock_is_independent_from_package_qualification() -> (
    None
):
    runtime_bound_spec = replace(
        _registry().by_id("saver"),
        environment=EnvironmentSpec(
            id="saver-r",
            status="ready",
            lock_sha256="a" * 64,
        ),
    )

    manifest_sha256, *_rest = saver_adapter._load_saver_environment_lock(
        runtime_bound_spec,
        SAVER_LOCK_PATH,
    )

    assert manifest_sha256 == hashlib.sha256(SAVER_LOCK_PATH.read_bytes()).hexdigest()


def test_saver_rejects_package_lock_not_bound_by_tracked_qualification(
    tmp_path: Path,
) -> None:
    environment_dir = tmp_path / "environments"
    environment_dir.mkdir()
    for source in (
        SAVER_LOCK_PATH,
        SAVER_BUILD_RECEIPT_PATH,
        SAVER_QUALIFICATION_PATH,
    ):
        shutil.copyfile(source, environment_dir / source.name)
    qualification_path = environment_dir / SAVER_QUALIFICATION_PATH.name
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["package_lock"]["sha256"] = "0" * 64
    qualification_path.write_text(json.dumps(qualification), encoding="utf-8")
    runtime_bound_spec = replace(
        _registry().by_id("saver"),
        environment=EnvironmentSpec(
            id="saver-r",
            status="ready",
            lock_sha256="a" * 64,
        ),
    )

    with pytest.raises(AdapterUnavailableError) as captured:
        saver_adapter._load_saver_environment_lock(
            runtime_bound_spec,
            environment_dir / SAVER_LOCK_PATH.name,
        )

    assert captured.value.reason_code == "environment_qualification_mismatch"


def _method_input(
    *,
    cells: int = 12,
    genes: int = 8,
    counts: np.ndarray | None = None,
):
    if counts is None:
        rows = []
        for cell in range(cells):
            rows.append(
                [
                    1 + ((cell + gene * 3) % 7) if (cell + gene) % 4 else 0
                    for gene in range(genes)
                ]
            )
        counts = np.asarray(rows, dtype=np.int64)
    else:
        if type(counts) is not np.ndarray or counts.ndim != 2:
            raise TypeError("fixture counts must be a two-dimensional ndarray")
        cells, genes = counts.shape
    view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"batch": pd.Categorical([f"b{cell % 2}" for cell in range(cells)])},
            index=[f"cell-{cell}" for cell in range(cells)],
        ),
        var=pd.DataFrame(index=[f"gene-{gene}" for gene in range(genes)]),
    )
    view.uns["source_dataset_sha256"] = SOURCE_SHA
    view.uns["allowed_covariates"] = {"obs": ["batch"], "var": []}
    view.uns["normalization"] = {
        "input": "counts",
        "target_sum": None,
        "log_base": None,
        "size_factor": None,
    }
    return prepare_method_input(view)


def _registry():
    return load_method_registry(METHODS_PATH)


def _cached_source(source_name: str) -> Path:
    source_dir = SOURCE_ROOT / source_name
    if not source_dir.is_dir():
        pytest.skip(f"ignored pinned-source cache is absent: {source_name}")
    return source_dir


def _assert_snapshot_bound(snapshot, method_id: str, method_input) -> None:
    assert snapshot.method_id == method_id
    assert snapshot.source_dataset_sha256 == method_input.source_dataset_sha256
    assert snapshot.obs_ids == method_input.obs_ids
    assert snapshot.var_ids == method_input.var_ids
    assert snapshot.shape == method_input.shape
    assert snapshot.matrix.flags.writeable is False


def _assert_evaluator_scales(execution, method_input) -> None:
    counts = core_methods.core_output_to_evaluator_counts(
        method_input,
        execution.snapshot,
    )
    common_log = core_methods.core_output_to_evaluator_log2_cp10k(
        method_input,
        execution.snapshot,
    )
    assert counts.shape == method_input.shape
    assert common_log.shape == method_input.shape
    assert np.isfinite(counts).all()
    assert np.isfinite(common_log).all()
    assert counts.flags.writeable is False
    assert common_log.flags.writeable is False


def test_observed_adapter_returns_exact_immutable_counts_without_subprocess() -> None:
    method_input = _method_input()
    execution = run_observed(_registry().by_id("observed"), method_input)

    _assert_snapshot_bound(execution.snapshot, "observed", method_input)
    _assert_evaluator_scales(execution, method_input)
    np.testing.assert_array_equal(execution.snapshot.matrix, method_input.counts)
    assert execution.command is None
    assert execution.stdout == b""
    assert execution.stderr == b""
    assert execution.environment_receipt == ()
    assert [event.code for event in execution.compatibility_log] == [
        "identity_control",
        "observed_positive_policy",
        "evaluator_scale_conversion",
    ]


@pytest.mark.parametrize(
    ("method_id", "stochastic", "seed_policy"),
    (
        ("observed", True, "required"),
        ("observed", 0, "not_applicable"),
        ("alra", False, "not_applicable"),
        ("alra", 1, "required"),
    ),
)
def test_same_input_adapter_guard_rejects_exact_seed_contract_drift(
    method_id: str,
    stochastic: object,
    seed_policy: str,
) -> None:
    canonical = _registry().by_id(method_id)
    drifted = replace(
        canonical,
        stochastic=stochastic,
        seed_policy=seed_policy,
    )

    with pytest.raises(ValueError, match="stochastic/seed contract"):
        require_method_spec(
            drifted,
            method_id,
            input_scale=canonical.input_scale,
            output_scale=canonical.output_scale,
        )


def test_log1p_cp10k_conversion_is_exact_and_rejects_zero_library_cells() -> None:
    counts = np.array([[2.0, 0.0, 3.0], [0.0, 4.0, 1.0]])
    expected = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10_000.0)

    np.testing.assert_allclose(log1p_cp10k(counts), expected, rtol=0, atol=0)
    with pytest.raises(AdapterUnavailableError, match="zero-library") as captured:
        log1p_cp10k(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert captured.value.reason_code == "zero_library_cell"


def test_alra_and_magic_inverse_log1p_cp10k_using_observed_libraries() -> None:
    method_input = _method_input(cells=4, genes=3)
    observed_libraries = method_input.counts.sum(axis=1, keepdims=True)
    expected_counts = np.array(
        [
            [0.25, 2.5, 1.75],
            [3.0, 0.5, 4.25],
            [1.125, 2.25, 0.375],
            [5.5, 1.0, 0.75],
        ],
        dtype=np.float64,
    )
    native_log1p_cp10k = np.log1p(expected_counts / observed_libraries * 10_000.0)

    for converter in (
        alra_adapter.alra_to_evaluator_counts,
        magic_adapter.magic_to_evaluator_counts,
    ):
        converted = converter(method_input, native_log1p_cp10k)
        np.testing.assert_allclose(converted, expected_counts, rtol=1e-12, atol=1e-12)
        assert converted.flags.writeable is False
        assert not np.shares_memory(converted, native_log1p_cp10k)


def test_saver_inverse_matches_pinned_null_size_factor_and_posterior_scale() -> None:
    source_dir = _cached_source("saver")
    utils_source = (source_dir / "R" / "utils.R").read_text(encoding="utf-8")
    posterior_source = (source_dir / "R" / "calc_posterior.R").read_text(
        encoding="utf-8"
    )
    assert "sf <- Matrix::colSums(x)/mean(Matrix::colSums(x))" in utils_source
    assert "scale.sf <- 1" in utils_source
    assert "lambda.hat*1000*scale.sf" in posterior_source

    method_input = _method_input(cells=4, genes=3)
    observed_libraries = method_input.counts.sum(axis=1, keepdims=True)
    size_factors = observed_libraries / observed_libraries.mean()
    expected_counts = np.array(
        [
            [0.25, 2.5, 1.75],
            [3.0, 0.5, 4.25],
            [1.125, 2.25, 0.375],
            [5.5, 1.0, 0.75],
        ],
        dtype=np.float64,
    )
    native_normalized = expected_counts / size_factors

    converted = saver_adapter.saver_to_evaluator_counts(method_input, native_normalized)

    np.testing.assert_allclose(converted, expected_counts, rtol=0, atol=0)
    assert converted.flags.writeable is False
    assert not np.shares_memory(converted, native_normalized)


@pytest.mark.parametrize("method_id", ["alra", "magic", "saver"])
def test_observed_library_bound_inverses_reject_zero_library_cells(
    method_id: str,
) -> None:
    method_input = _method_input(
        counts=np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int64)
    )

    with pytest.raises(AdapterUnavailableError, match="zero-library") as captured:
        converters = core_methods.CORE_EVALUATOR_COUNT_CONVERTERS
        converters[method_id](
            method_input, np.zeros(method_input.shape, dtype=np.float64)
        )

    assert captured.value.reason_code == "zero_library_cell"


def test_every_core_method_uses_explicit_count_inverse_and_one_common_log_scale() -> (
    None
):
    method_input = _method_input(cells=4, genes=3)
    observed_counts = np.array(method_input.counts, dtype=np.float64, copy=True)
    observed_libraries = observed_counts.sum(axis=1, keepdims=True)
    native_log1p_cp10k = np.log1p(observed_counts / observed_libraries * 10_000.0)
    size_factors = observed_libraries / observed_libraries.mean()
    registry = _registry()
    snapshots = {
        "observed": run_observed(registry.by_id("observed"), method_input).snapshot,
        "alra": finalize_alra_output(
            registry.by_id("alra"), method_input, native_log1p_cp10k
        ),
        "magic": finalize_magic_output(
            registry.by_id("magic"), method_input, native_log1p_cp10k
        ),
        "dca": finalize_dca_output(
            registry.by_id("dca"), method_input, observed_counts
        ),
        "scvi": finalize_scvi_output(
            registry.by_id("scvi"),
            method_input,
            observed_counts / observed_libraries,
        ),
        "saver": finalize_saver_output(
            registry.by_id("saver"), method_input, observed_counts / size_factors
        ),
    }
    expected_log = np.log2(1.0 + observed_counts / observed_libraries * 10_000.0)

    assert set(snapshots).issubset(core_methods.CORE_EVALUATOR_COUNT_CONVERTERS)
    for method_id, snapshot in snapshots.items():
        evaluator_counts = core_methods.core_output_to_evaluator_counts(
            method_input, snapshot
        )
        evaluator_log = core_methods.core_output_to_evaluator_log2_cp10k(
            method_input, snapshot
        )
        np.testing.assert_allclose(
            evaluator_counts, observed_counts, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(evaluator_log, expected_log, rtol=1e-12, atol=1e-12)
        assert evaluator_counts.flags.writeable is False
        assert evaluator_log.flags.writeable is False

    with pytest.raises(ValueError, match="no evaluator count converter"):
        core_methods.core_output_to_evaluator_counts(
            method_input, replace(snapshots["observed"], method_id="unknown")
        )

    with pytest.raises(AdapterUnavailableError, match="zero-library") as captured:
        core_methods.count_equivalent_to_log2_cp10k(np.array([[0.0, 0.0], [1.0, 2.0]]))
    assert captured.value.reason_code == "zero_library_cell"


def test_common_scale_normalization_handles_unrepresentable_raw_row_totals() -> None:
    maximum = np.finfo(np.float64).max
    counts = np.array([[maximum, maximum], [2.0, 3.0]])
    safe_log1p = np.log1p(counts[1] / np.sum(counts[1]) * 10_000.0)
    safe_log2 = np.log2(1.0 + counts[1] / np.sum(counts[1]) * 10_000.0)

    with np.errstate(all="raise"):
        log1p_result = log1p_cp10k(counts)
        log2_result = count_equivalent_to_log2_cp10k(counts)

    np.testing.assert_array_equal(
        log1p_result[0],
        np.full(2, np.log1p(5_000.0)),
    )
    np.testing.assert_array_equal(
        log2_result[0],
        np.full(2, np.log2(5_001.0)),
    )
    np.testing.assert_array_equal(log1p_result[1], safe_log1p)
    np.testing.assert_array_equal(log2_result[1], safe_log2)


def test_common_scale_normalization_retains_tiny_cp10k_terms_until_conversion() -> None:
    counts = np.array(
        [
            [
                np.finfo(np.float64).max,
                np.finfo(np.float64).max,
                2.0**-51,
            ]
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            log1p_result = log1p_cp10k(counts)
            log2_result = count_equivalent_to_log2_cp10k(counts)

    assert log1p_result[0, 2].hex() == "0x0.00000000009c4p-1022"
    assert log2_result[0, 2].hex() == "0x0.0000000000e17p-1022"


def test_observed_library_sizes_reject_unrepresentable_totals_without_fp_errors() -> (
    None
):
    maximum = np.finfo(np.float64).max
    method_input = _method_input(counts=np.array([[maximum, maximum]]))

    with np.errstate(all="raise"):
        with pytest.raises(AdapterUnavailableError, match="representable") as captured:
            observed_library_sizes(method_input)

    assert captured.value.reason_code == "unrepresentable_library_size"


@pytest.mark.parametrize(
    "conversion",
    [raw_output_to_count_equivalent, log1p_cp10k_to_count_equivalent],
)
def test_native_output_conversion_rejects_unrepresentable_longdouble_without_warning(
    conversion,
) -> None:
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("longdouble has no wider finite range on this platform")
    method_input = _method_input()
    native_output = np.ones(method_input.shape, dtype=np.longdouble)
    native_output[0, 0] = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            with pytest.raises(ValueError, match="representable as float64"):
                conversion(method_input, native_output)


def test_common_scale_zero_row_reason_precedes_another_row_overflow() -> None:
    maximum = np.finfo(np.float64).max
    counts = np.array([[maximum, maximum], [0.0, 0.0]])
    method_input = _method_input(counts=counts)

    for conversion in (log1p_cp10k, count_equivalent_to_log2_cp10k):
        with np.errstate(all="raise"):
            with pytest.raises(AdapterUnavailableError) as captured:
                conversion(counts)
        assert captured.value.reason_code == "zero_library_cell"

    with np.errstate(all="raise"):
        with pytest.raises(AdapterUnavailableError) as captured:
            observed_library_sizes(method_input)
    assert captured.value.reason_code == "zero_library_cell"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_dataset_sha256", "e" * 64, "source dataset"),
        ("output_scale", "log1p_cp10k", "native output scale"),
        ("obs_ids", ("wrong-1", "wrong-2", "wrong-3", "wrong-4"), "cell IDs"),
        ("var_ids", ("wrong-1", "wrong-2", "wrong-3"), "gene IDs"),
        ("shape", (2, 6), "shape"),
        ("matrix_sha256", "0" * 64, "matrix hash"),
    ],
)
def test_evaluator_dispatch_rejects_snapshot_binding_or_hash_tampering(
    field: str,
    value: object,
    message: str,
) -> None:
    method_input = _method_input(cells=4, genes=3)
    snapshot = run_observed(_registry().by_id("observed"), method_input).snapshot
    tampered = replace(snapshot, **{field: value})

    with pytest.raises(ValueError, match=message):
        core_methods.core_output_to_evaluator_counts(method_input, tampered)


def test_configs_match_exact_pinned_upstream_defaults_and_seed_overrides() -> None:
    assert ALRAConfig() == ALRAConfig(
        k=0,
        q=10,
        quantile_probability=0.001,
        use_mkl=False,
    )
    assert MAGICConfig() == MAGICConfig(
        knn=5,
        knn_max=None,
        decay=1,
        diffusion_time=3,
        n_pca=100,
        solver="exact",
        distance="euclidean",
        n_jobs=1,
    )
    assert DCAConfig() == DCAConfig(
        ae_type="zinb-conddisp",
        normalize_per_cell=True,
        scale=True,
        log1p=True,
        hidden_size=(64, 32, 64),
        hidden_dropout=0.0,
        batchnorm=True,
        activation="relu",
        initializer="glorot_uniform",
        epochs=300,
        reduce_lr=10,
        early_stop=15,
        batch_size=32,
        optimizer="RMSprop",
    )
    assert SCVIConfig() == SCVIConfig(
        n_hidden=128,
        n_latent=10,
        n_layers=1,
        dropout_rate=0.1,
        dispersion="gene",
        gene_likelihood="zinb",
        use_observed_lib_size=True,
        latent_distribution="normal",
        max_epochs=None,
        batch_size=128,
        batch_key="batch",
    )
    assert SAVERConfig() == SAVERConfig(
        do_fast=True,
        ncores=1,
        size_factor=None,
        estimates_only=True,
    )
    assert _registry().by_id("saver").seed_policy == "required"


def test_alra_binds_gnu_mkl_threading_layer_and_receipts_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_environment = None

    def fake_execute(spec, source_dir, command, **kwargs):
        nonlocal captured_environment
        captured_environment = kwargs.get("environment")
        rows, columns = int(command[8]), int(command[9])
        Path(command[6]).write_bytes(
            np.ones((rows, columns), dtype="<f8").tobytes(order="C")
        )
        Path(command[7]).write_text(
            "mkl_threading_layer\tGNU\n"
            "r_version\ttest-r\n"
            "rsvd_version\ttest-rsvd\n"
            f"upstream_source_file\t{command[4]}\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(alra_adapter, "execute_pinned_command", fake_execute)
    execution = run_alra(
        _registry().by_id("alra"),
        _method_input(),
        source_dir=tmp_path,
        rscript=Path(sys.executable),
        seed=42,
        work_root=tmp_path,
    )

    assert captured_environment == {"MKL_THREADING_LAYER": "GNU"}
    assert dict(execution.environment_receipt)["mkl_threading_layer"] == "GNU"
    compatibility = {event.code: event.detail for event in execution.compatibility_log}
    assert "MKL_THREADING_LAYER=GNU" in compatibility["numerical_stability_policy"]


def test_dca_binds_tensorflow_gpu_growth_and_receipts_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_environment = None

    def fake_execute(spec, source_dir, command, **kwargs):
        nonlocal captured_environment
        captured_environment = kwargs.get("environment")
        np.save(command[6], np.load(command[5], allow_pickle=False), allow_pickle=False)
        Path(command[7]).write_text(
            "anndata_version\ttest-anndata\n"
            f"dca_module\t{source_dir}/dca/__init__.py\n"
            "dca_version\ttest-dca\n"
            "numpy_version\ttest-numpy\n"
            "python_version\ttest-python\n"
            "scanpy_version\ttest-scanpy\n"
            "tensorflow_force_gpu_allow_growth\ttrue\n"
            "tensorflow_memory_growth\ttrue\n"
            "tensorflow_version\ttest-tensorflow\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(dca_adapter, "execute_pinned_command", fake_execute)
    execution = run_dca(
        _registry().by_id("dca"),
        _method_input(),
        source_dir=tmp_path,
        python_executable=Path(sys.executable),
        seed=42,
        work_root=tmp_path,
    )

    assert captured_environment == {"TF_FORCE_GPU_ALLOW_GROWTH": "true"}
    assert (
        dict(execution.environment_receipt)["tensorflow_force_gpu_allow_growth"]
        == "true"
    )
    assert dict(execution.environment_receipt)["tensorflow_memory_growth"] == "true"
    compatibility = {event.code: event.detail for event in execution.compatibility_log}
    assert "TF_FORCE_GPU_ALLOW_GROWTH=true" in compatibility["allocator_policy"]


def test_core_adapter_registry_metadata_binds_every_runtime_qualified_adapter() -> None:
    registry = _registry()
    runtime_sha256 = hashlib.sha256(
        DEVELOPMENT_RUNTIME_LOCK_PATH.read_bytes()
    ).hexdigest()
    for method_id in ("alra", "magic", "dca", "scvi", "saver"):
        spec = registry.by_id(method_id)
        assert spec.integration_status == "implemented"
        assert spec.integration_reason == "runtime_locked_adapter_smoke_passed"
        assert spec.environment.status == "ready"
        assert spec.environment.lock_sha256 == runtime_sha256


def test_saver_lock_manifest_and_build_receipt_cover_complete_dependency_closure() -> (
    None
):
    manifest = json.loads(SAVER_LOCK_PATH.read_text(encoding="utf-8"))
    packages = {item["package"]: item for item in manifest["packages"]}
    receipt = json.loads(SAVER_BUILD_RECEIPT_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["environment_id"] == "saver-r"
    assert manifest["r_version"] == "4.6.1"
    assert len(manifest["installed_library_sha256"]) == 64
    assert (
        manifest["build_receipt_sha256"]
        == hashlib.sha256(SAVER_BUILD_RECEIPT_PATH.read_bytes()).hexdigest()
    )
    assert {name: item["version"] for name, item in packages.items()} == {
        name: version
        for name, version in SAVER_PACKAGE_VERSIONS.items()
        if name != "SAVER"
    }
    assert all(
        item["url"].startswith("https://github.com/cran/") for item in packages.values()
    )
    assert all(len(item["sha256"]) == 64 for item in packages.values())
    assert manifest["upstream_saver"] == {
        "package": "SAVER",
        "version": "1.1.3",
        "url": "https://github.com/mohuangx/SAVER.git",
        "revision": "ad9bde51bffaa1413e57d88f15d2b452c6331253",
        "tree": "76884afc63ef27d78ce929bd608b29dad2b0a0be",
    }
    assert receipt["status"] == "real_tiny_smoke_passed"
    assert receipt["installed_library_sha256"] == manifest["installed_library_sha256"]
    assert receipt["package_versions"] == SAVER_PACKAGE_VERSIONS


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ALRAConfig(k=-1), "k"),
        (lambda: ALRAConfig(q=0), "q"),
        (lambda: MAGICConfig(knn=0), "knn"),
        (lambda: MAGICConfig(solver="approximate"), "exact"),
        (lambda: DCAConfig(epochs=0), "epochs"),
        (lambda: DCAConfig(ae_type="gaussian"), "ae_type"),
        (lambda: SCVIConfig(gene_likelihood="normal"), "gene_likelihood"),
        (lambda: SCVIConfig(batch_size=True), "batch_size"),
        (lambda: SAVERConfig(ncores=0), "ncores"),
    ],
)
def test_adapter_configs_reject_unprespecified_or_malformed_settings(
    factory,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("method_id", "source_name"),
    [
        ("alra", "alra"),
        ("magic", "core-magic"),
        ("dca", "core-dca"),
        ("scvi", "scvi"),
        ("saver", "saver"),
    ],
)
def test_adapter_source_boundaries_verify_exact_pins_without_writes(
    method_id: str,
    source_name: str,
) -> None:
    source_dir = _cached_source(source_name)
    before = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    receipt = verify_pinned_source(_registry().by_id(method_id), source_dir)

    assert receipt.revision == _registry().by_id(method_id).source.revision
    assert receipt.tree == _registry().by_id(method_id).source.tree
    assert receipt.url == _registry().by_id(method_id).source.url
    after = subprocess.run(
        ["git", "-C", str(source_dir), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert before == after == ""


def test_all_learned_finalizers_bind_native_scale_ids_and_source_hash() -> None:
    registry = _registry()
    method_input = _method_input()
    positive_output = np.full(method_input.shape, 0.75)
    normalized_output = log1p_cp10k(method_input.counts) + 0.25

    snapshots = [
        finalize_alra_output(registry.by_id("alra"), method_input, normalized_output),
        finalize_magic_output(registry.by_id("magic"), method_input, normalized_output),
        finalize_dca_output(registry.by_id("dca"), method_input, positive_output),
        finalize_saver_output(registry.by_id("saver"), method_input, positive_output),
    ]

    for method_id, snapshot in zip(
        ("alra", "magic", "dca", "saver"), snapshots, strict=True
    ):
        _assert_snapshot_bound(snapshot, method_id, method_input)
    assert [snapshot.output_scale for snapshot in snapshots] == [
        "log1p_cp10k",
        "log1p_cp10k",
        "raw_counts",
        "method_native_normalized",
    ]
    # Comparator adapters must not silently copy observed positives back.
    assert snapshots[0].matrix[0, 1] != method_input.counts[0, 1]
    assert snapshots[2].matrix[0, 1] != method_input.counts[0, 1]


def test_finalizers_reject_wrong_method_identity_and_nonfinite_upstream_output() -> (
    None
):
    registry = _registry()
    method_input = _method_input()
    with pytest.raises(ValueError, match="expected method alra"):
        finalize_alra_output(
            registry.by_id("magic"),
            method_input,
            np.ones(method_input.shape),
        )
    malformed = np.ones(method_input.shape)
    malformed[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        finalize_magic_output(registry.by_id("magic"), method_input, malformed)


def test_scvi_frequency_to_count_conversion_preserves_observed_library_sizes() -> None:
    method_input = _method_input(cells=4, genes=3)
    frequencies = np.array(
        [
            [0.2, 0.3, 0.5],
            [0.1, 0.1, 0.8],
            [0.25, 0.25, 0.5],
            [0.6, 0.2, 0.2],
        ]
    )

    output = frequencies_to_observed_library_counts(method_input, frequencies)
    np.testing.assert_allclose(
        output.sum(axis=1), method_input.counts.sum(axis=1), rtol=1e-12
    )
    snapshot = finalize_scvi_output(
        _registry().by_id("scvi"), method_input, frequencies
    )
    _assert_snapshot_bound(snapshot, "scvi", method_input)
    np.testing.assert_allclose(snapshot.matrix, output)


@pytest.mark.parametrize(
    "frequencies",
    [
        np.array([[0.2, 0.2], [0.5, 0.5]]),
        np.array([[0.5, -0.5], [0.5, 0.5]]),
        np.array([[np.nan, 1.0], [0.5, 0.5]]),
    ],
)
def test_scvi_rejects_invalid_decoded_frequencies(frequencies: np.ndarray) -> None:
    method_input = _method_input(cells=2, genes=2)
    with pytest.raises(ValueError, match="frequenc"):
        frequencies_to_observed_library_counts(method_input, frequencies)


def test_missing_environment_is_reported_with_reproducible_reason(
    tmp_path: Path,
) -> None:
    method_input = _method_input()
    missing = tmp_path / "missing-python"

    with pytest.raises(AdapterUnavailableError) as captured:
        run_scvi(
            _registry().by_id("scvi"),
            method_input,
            source_dir=SOURCE_ROOT / "scvi",
            python_executable=missing,
            seed=42,
            work_root=tmp_path,
        )

    assert captured.value.reason_code == "environment_executable_missing"
    assert str(missing) in captured.value.detail
    assert captured.value.stdout == b""
    assert captured.value.stderr == b""


def test_require_executable_preserves_absolute_venv_launcher_and_packages(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
        capture_output=True,
    )
    launcher = environment / "bin" / "python"
    assert launcher.is_symlink()
    site_packages = subprocess.run(
        [
            str(launcher),
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    Path(site_packages, "maskimpute_env_sentinel.py").write_text(
        "VALUE = 'selected-environment'\n", encoding="utf-8"
    )

    selected = require_executable(launcher)

    assert selected == launcher.absolute()
    assert selected.is_symlink()
    receipt = subprocess.run(
        [
            str(selected),
            "-I",
            "-c",
            "import maskimpute_env_sentinel,sys; "
            "print(sys.prefix); print(maskimpute_env_sentinel.VALUE)",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert receipt == [str(environment), "selected-environment"]


@pytest.mark.parametrize("kind", ["relative", "traversal", "directory"])
def test_require_executable_rejects_path_escape_and_nonregular_identity(
    tmp_path: Path,
    kind: str,
) -> None:
    executable = tmp_path / "tool"
    if kind == "relative":
        selected = Path("relative/tool")
    elif kind == "traversal":
        selected = tmp_path / "nested" / ".." / "tool"
    else:
        executable.mkdir()
        selected = executable

    with pytest.raises(AdapterUnavailableError) as captured:
        require_executable(selected)

    assert captured.value.reason_code == "environment_executable_unsafe"


def test_require_executable_rejects_broken_symlink(tmp_path: Path) -> None:
    launcher = tmp_path / "python"
    launcher.symlink_to(tmp_path / "missing-target")

    with pytest.raises(AdapterUnavailableError) as captured:
        require_executable(launcher)

    assert captured.value.reason_code == "environment_executable_missing"


def test_nonzero_upstream_exit_retains_logs_and_does_not_publish_output(
    tmp_path: Path,
) -> None:
    false = Path(shutil.which("false") or "/bin/false")
    method_input = _method_input()

    with pytest.raises(AdapterUnavailableError) as captured:
        run_magic(
            _registry().by_id("magic"),
            method_input,
            source_dir=_cached_source("core-magic"),
            python_executable=false,
            seed=42,
            work_root=tmp_path,
        )

    assert captured.value.reason_code == "upstream_nonzero_exit"
    assert captured.value.command is not None
    assert captured.value.stdout == b""
    assert captured.value.stderr == b""
    assert not list(tmp_path.glob("**/output.npy"))


def test_source_mutation_during_execution_fails_with_retained_logs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"], check=True
    )
    tracked = source / "method.py"
    tracked.write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "method.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "pin"], check=True)
    remote = "https://example.org/pinned.git"
    subprocess.run(
        ["git", "-C", str(source), "remote", "add", "origin", remote], check=True
    )
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    spec = replace(
        _registry().by_id("magic"),
        source=SourceSpec(
            kind="git",
            url=remote,
            revision=revision,
            tree=tree,
            cache_path="unused",
            freeze_binding=None,
        ),
    )
    command = (
        "/bin/sh",
        "-c",
        f"printf execution-log; printf changed >> {tracked}",
    )

    with pytest.raises(AdapterUnavailableError) as captured:
        execute_pinned_command(
            spec,
            source,
            command,
            cwd=tmp_path,
            timeout_seconds=10,
        )

    assert captured.value.reason_code == "source_mutated_during_execution"
    assert captured.value.stdout == b"execution-log"
    assert captured.value.command == command


def _has_rsvd(rscript: str) -> bool:
    result = subprocess.run(
        [
            rscript,
            "--vanilla",
            "-e",
            'quit(status=!requireNamespace("rsvd", quietly=TRUE))',
        ],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def test_real_pinned_alra_tiny_smoke(tmp_path: Path) -> None:
    rscript_value = shutil.which("Rscript")
    if rscript_value is None or not _has_rsvd(rscript_value):
        pytest.skip("Rscript with rsvd is unavailable")
    method_input = _method_input(cells=16, genes=10)

    execution = run_alra(
        _registry().by_id("alra"),
        method_input,
        source_dir=_cached_source("alra"),
        rscript=Path(rscript_value),
        seed=42,
        config=ALRAConfig(k=2),
        work_root=tmp_path,
    )

    _assert_snapshot_bound(execution.snapshot, "alra", method_input)
    _assert_evaluator_scales(execution, method_input)
    assert np.isfinite(execution.snapshot.matrix).all()
    assert (execution.snapshot.matrix >= 0).all()
    assert dict(execution.environment_receipt)["upstream_source_file"].endswith(
        "/alra.R"
    )
    assert dict(execution.environment_receipt)["mkl_threading_layer"] == "GNU"
    assert "upstream_rank_override" in [
        event.code for event in execution.compatibility_log
    ]
    assert "evaluator_scale_conversion" in [
        event.code for event in execution.compatibility_log
    ]


def test_real_pinned_alra_automatic_rank_medium_matrix_smoke(
    tmp_path: Path,
) -> None:
    rscript_value = shutil.which("Rscript")
    if rscript_value is None or not _has_rsvd(rscript_value):
        pytest.skip("Rscript with rsvd is unavailable")
    # Pinned choose_k computes its fixed 100-value tail, so both dimensions
    # must exceed 100 for the upstream automatic-rank default to be defined.
    method_input = _method_input(cells=160, genes=128)

    execution = run_alra(
        _registry().by_id("alra"),
        method_input,
        source_dir=_cached_source("alra"),
        rscript=Path(rscript_value),
        seed=42,
        config=ALRAConfig(),
        work_root=tmp_path,
    )

    _assert_snapshot_bound(execution.snapshot, "alra", method_input)
    _assert_evaluator_scales(execution, method_input)
    assert np.isfinite(execution.snapshot.matrix).all()
    assert np.isfinite(
        core_methods.core_output_to_evaluator_log2_cp10k(
            method_input, execution.snapshot
        )
    ).all()
    compatibility = {event.code: event.detail for event in execution.compatibility_log}
    assert compatibility["upstream_rank_selection"] == "k=0 automatic choice"
    assert "q=10" in compatibility["upstream_parameters"]


def test_real_pinned_magic_tiny_smoke_when_declared_dependency_env_exists(
    tmp_path: Path,
) -> None:
    python = Path("/home/marcinmaleclocal/miniconda3/envs/magic311/bin/python")
    if not python.is_file():
        pytest.skip("MAGIC dependency environment is unavailable")
    method_input = _method_input(cells=16, genes=10)

    execution = run_magic(
        _registry().by_id("magic"),
        method_input,
        source_dir=_cached_source("core-magic"),
        python_executable=python,
        seed=42,
        work_root=tmp_path,
    )

    _assert_snapshot_bound(execution.snapshot, "magic", method_input)
    _assert_evaluator_scales(execution, method_input)
    assert np.isfinite(execution.snapshot.matrix).all()
    assert (execution.snapshot.matrix >= 0).all()
    assert dict(execution.environment_receipt)["magic_module"].endswith(
        "/python/magic/__init__.py"
    )
    assert "evaluator_scale_conversion" in [
        event.code for event in execution.compatibility_log
    ]


def test_real_pinned_dca_tiny_smoke_when_legacy_environment_exists(
    tmp_path: Path,
) -> None:
    python = Path("/home/marcinmaleclocal/miniconda3/envs/dca_env/bin/python")
    if not python.is_file():
        pytest.skip("DCA dependency environment is unavailable")
    method_input = _method_input(cells=16, genes=10)

    execution = run_dca(
        _registry().by_id("dca"),
        method_input,
        source_dir=_cached_source("core-dca"),
        python_executable=python,
        seed=42,
        config=DCAConfig(epochs=1, reduce_lr=1, early_stop=1, batch_size=4),
        work_root=tmp_path,
    )

    _assert_snapshot_bound(execution.snapshot, "dca", method_input)
    _assert_evaluator_scales(execution, method_input)
    assert np.isfinite(execution.snapshot.matrix).all()
    assert (execution.snapshot.matrix >= 0).all()
    assert dict(execution.environment_receipt)["dca_module"].endswith(
        "/dca/__init__.py"
    )
    assert (
        dict(execution.environment_receipt)["tensorflow_force_gpu_allow_growth"]
        == "true"
    )
    assert dict(execution.environment_receipt)["tensorflow_memory_growth"] == "true"
    assert "upstream_training_override" in [
        event.code for event in execution.compatibility_log
    ]
    assert "evaluator_scale_conversion" in [
        event.code for event in execution.compatibility_log
    ]


def test_saver_missing_dependencies_are_a_failed_attempt_not_silent_replacement(
    tmp_path: Path,
) -> None:
    rscript_value = shutil.which("Rscript")
    if rscript_value is None:
        pytest.skip("Rscript is unavailable")
    method_input = _method_input(cells=8, genes=6)

    with pytest.raises(AdapterUnavailableError) as captured:
        run_saver(
            _registry().by_id("saver"),
            method_input,
            source_dir=_cached_source("saver"),
            rscript=Path(rscript_value),
            seed=42,
            library_dir=tmp_path / "missing-saver-library",
            lock_manifest=SAVER_LOCK_PATH,
            build_receipt=SAVER_BUILD_RECEIPT_PATH,
            work_root=tmp_path,
        )
    assert captured.value.reason_code == "environment_library_missing"
    assert captured.value.command is None


def test_saver_rejects_a_mutated_installed_library_before_execution(
    tmp_path: Path,
) -> None:
    rscript_value = shutil.which("Rscript")
    if rscript_value is None:
        pytest.skip("Rscript is unavailable")
    library_dir = tmp_path / "mutated-saver-library"
    for package in SAVER_PACKAGE_VERSIONS:
        (library_dir / package).mkdir(parents=True)
    (library_dir / "SAVER" / "modified-code.R").write_text(
        "stop('mutated')\n", encoding="utf-8"
    )

    with pytest.raises(AdapterUnavailableError) as captured:
        run_saver(
            _registry().by_id("saver"),
            _method_input(cells=8, genes=6),
            source_dir=_cached_source("saver"),
            rscript=Path(rscript_value),
            seed=42,
            library_dir=library_dir,
            lock_manifest=SAVER_LOCK_PATH,
            build_receipt=SAVER_BUILD_RECEIPT_PATH,
            work_root=tmp_path,
        )
    assert captured.value.reason_code == "environment_library_digest_mismatch"
    assert captured.value.command is None


def _has_saver_dependencies(rscript: str, library_dir: Path) -> bool:
    expression = (
        f".libPaths(c({str(library_dir)!r}, .Library));"
        f"packages<-c({','.join(repr(name) for name in SAVER_PACKAGE_VERSIONS)});"
        "quit(status=!all(vapply(packages,requireNamespace,logical(1),quietly=TRUE)))"
    )
    result = subprocess.run(
        [rscript, "--vanilla", "-e", expression],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def test_real_pinned_saver_fixed_seed_is_reproducible_when_dependencies_exist(
    tmp_path: Path,
) -> None:
    rscript_value = shutil.which("Rscript")
    if (
        rscript_value is None
        or not SAVER_LIBRARY_PATH.is_dir()
        or not _has_saver_dependencies(rscript_value, SAVER_LIBRARY_PATH)
    ):
        pytest.skip("SAVER dependency environment is unavailable")
    method_input = _method_input(cells=8, genes=6)
    manifest_sha256 = hashlib.sha256(SAVER_LOCK_PATH.read_bytes()).hexdigest()
    runtime_bound_spec = replace(
        _registry().by_id("saver"),
        environment=EnvironmentSpec(
            id="saver-r",
            status="ready",
            lock_sha256="a" * 64,
        ),
    )
    runs = [
        run_saver(
            runtime_bound_spec,
            method_input,
            source_dir=_cached_source("saver"),
            rscript=Path(rscript_value),
            seed=42,
            library_dir=SAVER_LIBRARY_PATH,
            lock_manifest=SAVER_LOCK_PATH,
            build_receipt=SAVER_BUILD_RECEIPT_PATH,
            work_root=tmp_path,
        )
        for _ in range(2)
    ]

    assert runs[0].snapshot.matrix_sha256 == runs[1].snapshot.matrix_sha256
    np.testing.assert_array_equal(runs[0].snapshot.matrix, runs[1].snapshot.matrix)
    _assert_evaluator_scales(runs[0], method_input)
    receipt = dict(runs[0].environment_receipt)
    assert receipt["manifest_sha256"] == manifest_sha256
    assert (
        receipt["qualification_sha256"]
        == hashlib.sha256(SAVER_QUALIFICATION_PATH.read_bytes()).hexdigest()
    )
    assert (
        receipt["installed_library_sha256"]
        == json.loads(SAVER_LOCK_PATH.read_text(encoding="utf-8"))[
            "installed_library_sha256"
        ]
    )
    assert (
        receipt["build_receipt_sha256"]
        == hashlib.sha256(SAVER_BUILD_RECEIPT_PATH.read_bytes()).hexdigest()
    )
    assert receipt["saver_library_dir"] == str(SAVER_LIBRARY_PATH.resolve())
    for package, version in SAVER_PACKAGE_VERSIONS.items():
        key = package.casefold() + "_version"
        assert receipt[key] == version
    assert runs[0].command is not None
    assert all("install.packages" not in argument for argument in runs[0].command)
    assert "evaluator_scale_conversion" in [
        event.code for event in runs[0].compatibility_log
    ]
