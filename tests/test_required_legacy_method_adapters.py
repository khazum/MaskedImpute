from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.methods as benchmark_methods
from maskimpute_benchmark.methods import (
    SourceSpec,
    load_method_registry,
    prepare_method_input,
)
from maskimpute_benchmark.methods.observed import AdapterUnavailableError
from maskimpute_benchmark.methods.sccr import (
    _SCCR_GRAPH_RECONSTRUCTION,
    SCCR_GRAPH_CONTRACT_REVISION,
    SCCR_GRAPH_CONTRACT_SHA256,
    SCCRConfig,
    finalize_sccr_output,
    reconstructed_sccr_knn_dense,
    run_sccr,
)
from maskimpute_benchmark.methods.scsdae import (
    _SCSDAE_PROBE_DRIVER,
    SCSDaeConfig,
    SCSDaeUnavailableError,
    finalize_scsdae_output,
    run_scsdae,
)


METHODS_PATH = Path("study/methods.json")
SOURCE_ROOT = Path("artifacts/method-sources")
SOURCE_SHA = "d" * 64


def _method_input(*, cells: int = 12, genes: int = 8):
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


def _registry():
    return load_method_registry(METHODS_PATH)


def _cached_source(name: str) -> Path:
    source = SOURCE_ROOT / name
    if not source.is_dir():
        pytest.skip(f"ignored pinned-source cache is absent: {name}")
    return source


def _fake_pinned_spec(tmp_path: Path, method_id: str):
    source = tmp_path / f"{method_id}-source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"], check=True
    )
    (source / "method.py").write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "method.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "pin"], check=True)
    remote = "https://example.org/pinned.git"
    subprocess.run(
        ["git", "-C", str(source), "remote", "add", "origin", remote],
        check=True,
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
        _registry().by_id(method_id),
        source=SourceSpec(
            kind="git",
            url=remote,
            revision=revision,
            tree=tree,
            cache_path="unused",
            freeze_binding=None,
        ),
    )
    return source, spec


def _fake_cpu_sccr_launcher(tmp_path: Path) -> Path:
    launcher = tmp_path / "fake-cpu-sccr"
    launcher.write_text(
        f"""#!{sys.executable}
from pathlib import Path
import sys
import numpy as np

requested = sys.argv[17]
device = "cpu" if requested == "auto" else requested
values = np.load(sys.argv[6], allow_pickle=False)
np.save(Path(sys.argv[7]), values, allow_pickle=False)
receipt = {{
    "device": device,
    "graph_contract_revision": {SCCR_GRAPH_CONTRACT_REVISION!r},
    "graph_contract_sha256": {SCCR_GRAPH_CONTRACT_SHA256!r},
    "graph_contract_url": "https://github.com/Junseok0207/scFP.git",
    "numpy_version": str(np.__version__),
    "python_version": sys.version.split()[0],
    "sccr_module": str(Path(sys.argv[5]) / "scCR.py"),
    "torch_num_threads": "3",
    "torch_version": "2.4.1",
}}
Path(sys.argv[8]).write_text(
    "".join(f"{{key}}\\t{{receipt[key]}}\\n" for key in sorted(receipt)),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def _fake_scsdae_launcher(tmp_path: Path, mode: str) -> Path:
    launcher = tmp_path / f"fake-scsdae-{mode}"
    launcher.write_text(
        f"""#!{sys.executable}
from pathlib import Path
import sys
import numpy as np

mode = {mode!r}
driver = sys.argv[4]
if "adapter probe expected" in driver:
    if mode == "kernel_probe_failure":
        print("MASKIMPUTE_LEGACY_GPU_KERNEL_INCOMPATIBLE gpu=/gpu:0")
        print("first-gpu0-kernel-failed", file=sys.stderr)
        raise SystemExit(9)
    print("probe-ok")
    raise SystemExit(0)
if mode == "run_failure":
    print("run-stdout")
    print("run-stderr", file=sys.stderr)
    raise SystemExit(7)
np.save(Path(sys.argv[7]), np.ones((1, 1), dtype=np.float64), allow_pickle=False)
receipt = {{
    "gpu_available": "true",
    "gpu_index": "0",
    "keras_version": "2.2.4",
    "numpy_version": str(np.__version__),
    "pandas_version": "0.24.2",
    "python_version": "3.6.13",
    "source_script": str(Path(sys.argv[5])),
    "tensorflow_memory_growth": "true",
    "tensorflow_version": "1.12.0",
}}
Path(sys.argv[9]).write_text(
    "".join(f"{{key}}\\t{{receipt[key]}}\\n" for key in sorted(receipt)),
    encoding="utf-8",
)
print("run-output-written")
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def _assert_snapshot(snapshot, method_id: str, method_input) -> None:
    assert snapshot.method_id == method_id
    assert snapshot.source_dataset_sha256 == method_input.source_dataset_sha256
    assert snapshot.obs_ids == method_input.obs_ids
    assert snapshot.var_ids == method_input.var_ids
    assert snapshot.shape == method_input.shape
    assert snapshot.output_scale == "method_native_normalized"
    assert snapshot.matrix.flags.writeable is False


def test_required_legacy_configs_match_pinned_defaults() -> None:
    assert SCCRConfig() == SCCRConfig(
        neighbors=15,
        gene_neighbors=2,
        symmetric_final_graph=True,
        iterations=40,
        complete_relation_weight=0.05,
        soft_propagation_weight=0.99,
        final_blend_weight=0.01,
        device=None,
    )
    assert SCSDaeConfig() == SCSDaeConfig(
        batch_size=256,
        autoencoder_iterations=2000,
        pretrain_iterations=1000,
        zero_loss_weight=1.0,
        observed_loss_weight=1.0,
        dropout_rate=0.2,
        l1_regularization=0.0,
        l2_regularization=0.0,
        gene_scale=False,
        gpu_index=0,
    )


def test_scsdae_gpu_discovery_uses_the_growth_enabled_session_config() -> None:
    assert "tf.test.is_gpu_available" not in _SCSDAE_PROBE_DRIVER
    assert (
        "device_lib.list_local_devices(session_config=probe_config)"
        in _SCSDAE_PROBE_DRIVER
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SCCRConfig(neighbors=0), "neighbors"),
        (lambda: SCCRConfig(gene_neighbors=0), "gene_neighbors"),
        (lambda: SCCRConfig(iterations=0), "iterations"),
        (lambda: SCCRConfig(complete_relation_weight=1.1), "complete_relation"),
        (lambda: SCCRConfig(device="cuda"), "device"),
        (lambda: SCSDaeConfig(batch_size=0), "batch_size"),
        (lambda: SCSDaeConfig(autoencoder_iterations=0), "autoencoder_iterations"),
        (lambda: SCSDaeConfig(dropout_rate=1.0), "dropout_rate"),
        (lambda: SCSDaeConfig(gene_scale=True), "gene_scale"),
        (lambda: SCSDaeConfig(gpu_index=-1), "gpu_index"),
    ],
)
def test_required_legacy_configs_reject_unsupported_values(
    factory, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_native_finalizers_and_scale_converters_are_explicit() -> None:
    method_input = _method_input(cells=8, genes=6)
    registry = _registry()
    counts = np.asarray(method_input.counts, dtype=np.float64)
    libraries = counts.sum(axis=1, keepdims=True)
    sccr_native = np.log1p(counts / libraries * 10_000.0)
    scsdae_native = np.log1p(counts / libraries * 1_000_000.0)
    snapshots = {
        "sccr": finalize_sccr_output(registry.by_id("sccr"), method_input, sccr_native),
        "scsdae": finalize_scsdae_output(
            registry.by_id("scsdae"), method_input, scsdae_native
        ),
    }

    assert set(benchmark_methods.LEGACY_EVALUATOR_COUNT_CONVERTERS) == set(snapshots)
    assert benchmark_methods.LEGACY_EVALUATOR_NATIVE_SCALES == {
        "sccr": "method_native_normalized",
        "scsdae": "method_native_normalized",
    }
    for method_id, snapshot in snapshots.items():
        _assert_snapshot(snapshot, method_id, method_input)
        converted = benchmark_methods.legacy_output_to_evaluator_counts(
            method_input, snapshot
        )
        common = benchmark_methods.legacy_output_to_evaluator_log2_cp10k(
            method_input, snapshot
        )
        np.testing.assert_allclose(converted, counts, rtol=1e-11, atol=1e-11)
        expected_common = np.log2(1.0 + counts / libraries * 10_000.0)
        np.testing.assert_allclose(common, expected_common, rtol=1e-11, atol=1e-11)
        assert converted.flags.writeable is False
        assert common.flags.writeable is False


def test_scsdae_count_inverse_does_not_assume_cp10k() -> None:
    method_input = _method_input(cells=8, genes=6)
    counts = np.asarray(method_input.counts, dtype=np.float64)
    libraries = counts.sum(axis=1, keepdims=True)
    native = np.log1p(counts / libraries * 1_000_000.0)

    converted = benchmark_methods.scsdae_to_evaluator_counts(method_input, native)

    np.testing.assert_allclose(converted, counts, rtol=1e-11, atol=1e-11)


def test_sccr_reconstructed_graph_matches_two_frozen_source_fixtures() -> None:
    assert SCCR_GRAPH_CONTRACT_REVISION == ("de372f99aa33a7cc4214bd99e0fa4a253652e505")
    assert SCCR_GRAPH_CONTRACT_SHA256 == (
        "fb90fd2409337fb39247fb11ed2076f532566f946bf38103b5f4c6fe9a50cda3"
    )
    embeddings_one = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-0.6, -0.8]],
        dtype=np.float64,
    )
    expected_one = np.array(
        [
            [5 / 7, 2 / 7, 0.0, 0.0],
            [8 / 31, 20 / 31, 3 / 31, 0.0],
            [0.0, 3 / 23, 20 / 23, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    embeddings_two = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 0.6, 0.8],
            [-0.8, 0.0, 0.6],
        ],
        dtype=np.float64,
    )
    expected_two = np.array(
        [
            [10 / 13, 3 / 13, 0.0, 0.0],
            [15 / 77, 50 / 77, 12 / 77, 0.0],
            [0.0, 6 / 37, 25 / 37, 6 / 37],
            [0.0, 0.0, 6 / 31, 25 / 31],
        ],
        dtype=np.float64,
    )

    actual_one = reconstructed_sccr_knn_dense(embeddings_one, 1, symmetric=True)
    actual_two = reconstructed_sccr_knn_dense(embeddings_two, 2, symmetric=False)

    np.testing.assert_allclose(actual_one, expected_one, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(actual_two, expected_two, rtol=1e-7, atol=1e-7)

    torch_python = Path("/tmp/maskimpute-scziva312/bin/python")
    if not torch_python.is_file():
        pytest.skip(
            "Torch adapter environment is absent; runtime graph equivalence was not run"
        )
    driver = r"""
import json
import sys
import torch
namespace = {}
exec(sys.argv[1], namespace)
embeddings = torch.tensor(json.loads(sys.argv[2]), dtype=torch.float32)
edges, weights = namespace["knn_graph"](
    embeddings, int(sys.argv[3]), sym=sys.argv[4] == "true"
)
dense = torch.sparse_coo_tensor(
    edges, weights, (embeddings.shape[0], embeddings.shape[0])
).to_dense()
print(json.dumps(dense.tolist()))
"""
    for embeddings, neighbors, symmetric, expected in (
        (embeddings_one, 1, True, expected_one),
        (embeddings_two, 2, False, expected_two),
    ):
        result = subprocess.run(
            (
                str(torch_python),
                "-B",
                "-I",
                "-c",
                driver,
                _SCCR_GRAPH_RECONSTRUCTION,
                json.dumps(embeddings.tolist()),
                str(neighbors),
                "true" if symmetric else "false",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        actual_runtime = np.asarray(json.loads(result.stdout), dtype=np.float64)
        np.testing.assert_allclose(actual_runtime, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method_id", ["sccr", "scsdae"])
def test_required_legacy_source_boundaries_match_frozen_pins(method_id: str) -> None:
    source = _cached_source(method_id)
    before = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    receipt = benchmark_methods.verify_pinned_source(
        _registry().by_id(method_id), source
    )

    assert receipt.revision == _registry().by_id(method_id).source.revision
    after = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert before == after == ""


@pytest.mark.parametrize("method_id", ["sccr", "scsdae"])
def test_required_legacy_adapters_reject_work_roots_inside_pinned_source(
    method_id: str,
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=10, genes=7)
    source = _cached_source(method_id)
    nested = source / "adapter-work-must-not-be-created"
    assert not nested.exists()

    with pytest.raises(AdapterUnavailableError) as captured:
        if method_id == "sccr":
            run_sccr(
                _registry().by_id(method_id),
                method_input,
                source_dir=source,
                python_executable=tmp_path / "missing-python",
                seed=42,
                config=SCCRConfig(neighbors=3, gene_neighbors=1),
                work_root=nested,
            )
        else:
            run_scsdae(
                _registry().by_id(method_id),
                method_input,
                source_dir=source,
                python_executable=tmp_path / "missing-python",
                seed=42,
                work_root=nested,
            )

    assert captured.value.reason_code == "unsafe_work_root"
    assert not nested.exists()
    assert (
        subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_sccr_rejects_dimensions_smaller_than_its_declared_graphs() -> None:
    method_input = _method_input(cells=8, genes=6)

    with pytest.raises(AdapterUnavailableError) as captured:
        run_sccr(
            _registry().by_id("sccr"),
            method_input,
            source_dir=_cached_source("sccr"),
            python_executable=Path(shutil.which("python") or "/missing-python"),
            seed=42,
            work_root=Path("/tmp"),
        )
    assert captured.value.reason_code == "upstream_minimum_dimension"
    assert captured.value.command is None


def test_sccr_real_pinned_tiny_smoke_when_torch_environment_is_available(
    tmp_path: Path,
) -> None:
    torch_python = Path("/tmp/maskimpute-scziva312/bin/python")
    if not torch_python.is_file():
        pytest.skip("exact cached PyTorch smoke environment is absent")
    method_input = _method_input(cells=10, genes=7)
    config = SCCRConfig(
        neighbors=3,
        gene_neighbors=1,
        iterations=2,
        device="cpu",
    )

    first = run_sccr(
        _registry().by_id("sccr"),
        method_input,
        source_dir=_cached_source("sccr"),
        python_executable=torch_python,
        seed=42,
        config=config,
        work_root=tmp_path,
    )
    second = run_sccr(
        _registry().by_id("sccr"),
        method_input,
        source_dir=_cached_source("sccr"),
        python_executable=torch_python,
        seed=42,
        config=config,
        work_root=tmp_path,
    )

    _assert_snapshot(first.snapshot, "sccr", method_input)
    np.testing.assert_array_equal(first.snapshot.matrix, second.snapshot.matrix)
    receipt = dict(first.environment_receipt)
    assert receipt["sccr_module"].endswith("/scCR.py")
    assert receipt["graph_contract_revision"] == SCCR_GRAPH_CONTRACT_REVISION
    assert receipt["graph_contract_sha256"] == SCCR_GRAPH_CONTRACT_SHA256
    assert receipt["torch_num_threads"] == "3"
    codes = {event.code for event in first.compatibility_log}
    assert "missing_graph_utility_reconstruction" in codes
    assert "truth_free_entrypoint" in codes
    assert "resource_behavior" in codes
    assert "reconstruction_license_limitation" in codes
    assert "upstream_parameter_override" in codes


def test_sccr_default_device_uses_cpu_when_selected_executable_has_no_cuda(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=10, genes=7)
    source, spec = _fake_pinned_spec(tmp_path, "sccr")
    launcher = _fake_cpu_sccr_launcher(tmp_path)

    execution = run_sccr(
        spec,
        method_input,
        source_dir=source,
        python_executable=launcher,
        seed=42,
        config=SCCRConfig(neighbors=3, gene_neighbors=1, iterations=1),
        work_root=tmp_path,
    )

    assert execution.command is not None
    assert execution.command[-1] == "auto"
    assert dict(execution.environment_receipt)["device"] == "cpu"
    assert spec.resources.gpu_mode == "required"


def test_sccr_default_device_uses_cuda_in_supported_selected_executable(
    tmp_path: Path,
) -> None:
    python = Path("/tmp/maskimpute-supported/bin/python")
    if not python.is_file():
        pytest.skip("supported CUDA executable is absent")
    source = _cached_source("sccr")
    method_input = _method_input(cells=10, genes=7)

    execution = run_sccr(
        _registry().by_id("sccr"),
        method_input,
        source_dir=source,
        python_executable=python,
        seed=42,
        config=SCCRConfig(neighbors=3, gene_neighbors=1, iterations=1),
        work_root=tmp_path,
    )

    assert dict(execution.environment_receipt)["device"] == "cuda:0"
    assert execution.command is not None
    assert execution.command[-1] == "auto"


def test_scsdae_missing_environment_returns_full_source_attempt_receipt(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    missing = tmp_path / "missing-python"

    with pytest.raises(SCSDaeUnavailableError) as captured:
        run_scsdae(
            _registry().by_id("scsdae"),
            method_input,
            source_dir=_cached_source("scsdae"),
            python_executable=missing,
            seed=42,
            work_root=tmp_path,
        )

    error = captured.value
    assert error.reason_code == "environment_executable_missing"
    assert error.command is None
    assert error.attempt_receipt.source_revision == (
        "fa7ded1080695e38e6193ef137dc8d635ae64ec9"
    )
    assert error.attempt_receipt.source_tree == (
        "b6f13e3ef513ad6c1fe1afb702681f249c7cbca8"
    )
    assert error.attempt_receipt.environment_id == "scsdae-tensorflow1"
    assert error.attempt_receipt.environment_registry_status == "pending"
    assert error.attempt_receipt.executable == str(missing)
    assert error.attempt_receipt.outcome == "unavailable"
    assert error.attempt_receipt.reason_code == error.reason_code
    assert error.attempt_receipt.stdout_sha256 == error.stdout_sha256
    assert error.attempt_receipt.stderr_sha256 == error.stderr_sha256
    assert error.attempt_receipt.probe_command is None
    assert error.attempt_receipt.run_command is None


def test_scsdae_real_pinned_gpu0_tiny_smoke_parses_upstream_header(
    tmp_path: Path,
) -> None:
    python = Path("/tmp/maskimpute-scsdae-conda/bin/python")
    if not python.is_file():
        pytest.skip("exact legacy scSDAE environment is absent")
    method_input = _method_input(cells=16, genes=10)

    execution = run_scsdae(
        _registry().by_id("scsdae"),
        method_input,
        source_dir=_cached_source("scsdae"),
        python_executable=python,
        seed=42,
        config=SCSDaeConfig(
            batch_size=16,
            # One epoch can leave the pinned linear output layer negative on
            # current CUDA kernels; five remains a tiny disclosed override
            # while exercising a valid native-output/evaluator path.
            autoencoder_iterations=5,
            pretrain_iterations=5,
            gpu_index=0,
        ),
        work_root=tmp_path,
    )

    _assert_snapshot(execution.snapshot, "scsdae", method_input)
    assert execution.snapshot.shape == (16, 10)
    receipt = dict(execution.environment_receipt)
    assert receipt["gpu_available"] == "true"
    assert receipt["gpu_index"] == "0"
    assert receipt["tensorflow_version"] == "1.12.0"
    assert receipt["keras_version"] == "2.2.4"
    assert receipt["tensorflow_memory_growth"] == "true"
    assert (
        b"MASKIMPUTE_SCSDAE_PREFLIGHT tensorflow_memory_growth=true"
        in execution.stdout
    )
    assert "gpu_device_binding" in {
        event.code for event in execution.compatibility_log
    }
    assert "allocator_policy" in {
        event.code for event in execution.compatibility_log
    }


def test_scsdae_failed_run_receipt_retains_probe_and_run_evidence(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    launcher = _fake_scsdae_launcher(tmp_path, "run_failure")

    with pytest.raises(SCSDaeUnavailableError) as captured:
        run_scsdae(
            _registry().by_id("scsdae"),
            method_input,
            source_dir=_cached_source("scsdae"),
            python_executable=launcher,
            seed=42,
            config=replace(
                SCSDaeConfig(),
                autoencoder_iterations=1,
                pretrain_iterations=1,
                gpu_index=0,
            ),
            work_root=tmp_path,
        )

    error = captured.value
    attempt = error.attempt_receipt
    assert error.reason_code == "upstream_nonzero_exit"
    assert attempt.probe_command is not None
    assert attempt.run_command == error.command
    assert attempt.probe_command != attempt.run_command
    assert attempt.probe_stdout_sha256 == hashlib.sha256(b"probe-ok\n").hexdigest()
    assert attempt.run_stdout_sha256 == hashlib.sha256(b"run-stdout\n").hexdigest()
    assert attempt.run_stderr_sha256 == hashlib.sha256(b"run-stderr\n").hexdigest()
    assert b"probe-ok" in error.stdout
    assert b"run-stdout" in error.stdout
    assert b"run-stderr" in error.stderr
    assert attempt.stdout_sha256 == error.stdout_sha256
    assert attempt.stderr_sha256 == error.stderr_sha256


def test_scsdae_gpu0_kernel_probe_failure_uses_canonical_retained_receipt(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    source, spec = _fake_pinned_spec(tmp_path, "scsdae")
    launcher = _fake_scsdae_launcher(tmp_path, "kernel_probe_failure")

    with pytest.raises(SCSDaeUnavailableError) as captured:
        run_scsdae(
            spec,
            method_input,
            source_dir=source,
            python_executable=launcher,
            seed=42,
            config=replace(
                SCSDaeConfig(),
                autoencoder_iterations=1,
                pretrain_iterations=1,
                gpu_index=0,
            ),
            work_root=tmp_path,
        )

    error = captured.value
    attempt = error.attempt_receipt
    assert error.reason_code == "legacy_gpu_kernel_incompatible"
    assert attempt.reason_code == "legacy_gpu_kernel_incompatible"
    assert attempt.probe_command == error.command
    assert attempt.run_command is None
    assert attempt.probe_stdout_sha256 == hashlib.sha256(
        b"MASKIMPUTE_LEGACY_GPU_KERNEL_INCOMPATIBLE gpu=/gpu:0\n"
    ).hexdigest()
    assert attempt.probe_stderr_sha256 == hashlib.sha256(
        b"first-gpu0-kernel-failed\n"
    ).hexdigest()
    assert attempt.run_stdout_sha256 == hashlib.sha256(b"").hexdigest()
    assert attempt.run_stderr_sha256 == hashlib.sha256(b"").hexdigest()


def test_scsdae_malformed_shape_is_wrapped_with_complete_attempt_receipt(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    launcher = _fake_scsdae_launcher(tmp_path, "malformed_shape")

    with pytest.raises(SCSDaeUnavailableError) as captured:
        run_scsdae(
            _registry().by_id("scsdae"),
            method_input,
            source_dir=_cached_source("scsdae"),
            python_executable=launcher,
            seed=42,
            config=replace(
                SCSDaeConfig(),
                autoencoder_iterations=1,
                pretrain_iterations=1,
                gpu_index=0,
            ),
            work_root=tmp_path,
        )

    error = captured.value
    attempt = error.attempt_receipt
    assert error.reason_code == "malformed_upstream_output"
    assert "shape" in error.detail
    assert attempt.probe_command is not None
    assert attempt.run_command == error.command
    assert attempt.probe_stdout_sha256 == hashlib.sha256(b"probe-ok\n").hexdigest()
    assert (
        attempt.run_stdout_sha256 == hashlib.sha256(b"run-output-written\n").hexdigest()
    )
    assert attempt.stdout_sha256 == error.stdout_sha256
    assert attempt.stderr_sha256 == error.stderr_sha256


def test_scsdae_modern_environment_is_not_silently_used(tmp_path: Path) -> None:
    method_input = _method_input(cells=8, genes=6)
    executable = Path(shutil.which("python") or "/missing-python")
    if not executable.is_file():
        pytest.skip("host Python executable is absent")

    with pytest.raises(SCSDaeUnavailableError) as captured:
        run_scsdae(
            _registry().by_id("scsdae"),
            method_input,
            source_dir=_cached_source("scsdae"),
            python_executable=executable,
            seed=42,
            config=replace(
                SCSDaeConfig(),
                autoencoder_iterations=1,
                pretrain_iterations=1,
                gpu_index=0,
            ),
            work_root=tmp_path,
        )

    error = captured.value
    assert error.command is not None
    assert error.attempt_receipt.command == error.command
    assert (
        error.attempt_receipt.source_revision
        == _registry().by_id("scsdae").source.revision
    )
    assert error.attempt_receipt.stdout_sha256 == error.stdout_sha256
    assert error.attempt_receipt.stderr_sha256 == error.stderr_sha256
    assert error.reason_code in {
        "legacy_environment_mismatch",
        "legacy_gpu_kernel_incompatible",
        "upstream_dependency_missing",
        "upstream_nonzero_exit",
    }
