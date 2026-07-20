from __future__ import annotations

from dataclasses import fields, replace
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import boxcox

import maskimpute_benchmark.methods as benchmark_methods
from maskimpute_benchmark.methods import (
    SourceSpec,
    load_method_registry,
    prepare_method_input,
)
from maskimpute_benchmark.methods.observed import AdapterUnavailableError
from maskimpute_benchmark.runner import (
    ExecutionEnvironmentRegistry,
    RepositoryAdapterDispatcher,
)


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


def _fake_pinned_source(tmp_path: Path, method_id: str):
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
    return source, replace(
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


def _fake_negative_afmf_launcher(tmp_path: Path) -> Path:
    launcher = tmp_path / "fake-negative-afmf"
    launcher.write_text(
        f"""#!{sys.executable}
from pathlib import Path
import sys
import numpy as np

values = np.load(sys.argv[6], allow_pickle=False)
output = np.ones_like(values, dtype=np.float64)
output[0, 0] = -0.25
output[1, 1] = -2.5
np.save(Path(sys.argv[7]), output, allow_pickle=False)
receipt = {{
    "afmf_module": str(Path(sys.argv[5]) / "runafMF.py"),
    "numpy_version": str(np.__version__),
    "pandas_version": "2.2.1",
    "python_version": sys.version.split()[0],
    "sklearn_version": "1.5.0",
}}
Path(sys.argv[8]).write_text(
    "".join(f"{{key}}\\t{{receipt[key]}}\\n" for key in sorted(receipt)),
    encoding="utf-8",
)
print("negative-native-output")
print("negative-native-stderr", file=sys.stderr)
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


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


def test_direct_repository_mapping_dispatches_all_ten_typed_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = _registry()
    from maskimpute_benchmark.comparator_tuning import (
        load_comparator_tuning_authority,
    )

    authority = load_comparator_tuning_authority(
        Path.cwd(),
        registry=registry,
        require_clean=False,
    )
    for method_id in authority.method_order:
        source = registry.by_id(method_id).source.cache_path
        assert source is not None
        (tmp_path / source).mkdir(parents=True, exist_ok=True)
    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        {method_id: Path(sys.executable) for method_id in authority.method_order},
    )
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "revalidate_for",
        lambda _self, _method_id: None,
    )
    dispatcher = RepositoryAdapterDispatcher(tmp_path, environments)
    adapters = dispatcher.direct_comparator_adapters()
    method_input = _method_input(cells=8, genes=6)
    received: dict[str, object] = {}

    def forbidden_legacy_adapter(*_args, **_kwargs):
        raise AssertionError("direct production mapping called a legacy adapter")

    for method_id in authority.method_order:
        spec = registry.by_id(method_id)

        direct_name = f"run_{method_id}_direct"
        direct_adapter = getattr(benchmark_methods, direct_name, None)
        assert direct_adapter is not None, f"{direct_name} is absent"

        def spy(*_args, _method_id=method_id, **kwargs):
            from maskimpute_benchmark.methods.direct import (
                DirectAdapterExecution,
                finalize_direct_method_output,
            )

            received[_method_id] = kwargs["config"]
            return DirectAdapterExecution(
                output=finalize_direct_method_output(
                    spec,
                    method_input,
                    method_input.counts,
                    output_scale=spec.output_scale,
                    obs_ids=method_input.obs_ids,
                    var_ids=method_input.var_ids,
                ),
                stdout=b"",
                stderr=b"",
            )

        monkeypatch.setattr(
            benchmark_methods,
            f"run_{method_id}",
            forbidden_legacy_adapter,
        )
        monkeypatch.setattr(benchmark_methods, direct_name, spy)
        row = authority.configurations_for(method_id)[0]
        outcome = adapters[method_id](
            spec,
            method_input,
            seed=42,
            config=row.decode(),
        )
        assert outcome.status == "completed"
        assert outcome.execution is not None
        assert tuple(field.name for field in fields(outcome.execution)) == (
            "output",
            "stdout",
            "stderr",
        )
        assert received[method_id] == row.decode()

    assert tuple(adapters) == authority.method_order


def test_direct_finalizers_never_call_content_summary_helpers_and_are_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.methods import base as base_module

    registry = _registry()
    method_input = _method_input(cells=120, genes=120)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct finalization called a content-summary helper")

    monkeypatch.setattr(base_module, "_output_digest", forbidden)
    monkeypatch.setattr(base_module, "snapshot_method_output", forbidden)
    forbidden_tokens = ("hash", "digest", "checksum", "fingerprint", "sha")
    for method_id in (
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "scziva",
        "afmf",
        "biaeimpute",
        "sccr",
        "scsdae",
    ):
        module = _adapter_module(method_id)
        finalizer = getattr(module, f"finalize_{method_id}_direct_output", None)
        assert finalizer is not None, f"direct {method_id} finalizer is absent"
        monkeypatch.setattr(module, "snapshot_method_output", forbidden)
        native = np.array(method_input.counts, dtype=np.float64, copy=True)
        if method_id == "scvi":
            native /= native.sum(axis=1, keepdims=True)
        output = finalizer(registry.by_id(method_id), method_input, native)
        names = tuple(field.name for field in fields(output))
        assert not any(
            token in name.casefold()
            for name in names
            for token in forbidden_tokens
            if name != "shape"
        )
        assert output.matrix.shape == method_input.shape
        assert np.isfinite(output.matrix).all()


def test_direct_repository_mapping_reports_missing_executable_without_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = _registry()
    spec = registry.by_id("magic")
    source = spec.source.cache_path
    assert source is not None
    (tmp_path / source).mkdir(parents=True)
    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        {"magic": tmp_path / "missing-python"},
    )
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "revalidate_for",
        lambda _self, _method_id: None,
    )
    attempted: list[bool] = []
    direct_adapter = getattr(benchmark_methods, "run_magic_direct", None)
    assert direct_adapter is not None, "run_magic_direct is absent"

    def spy(*_args, **_kwargs):
        attempted.append(True)
        raise AssertionError("missing executable reached the adapter")

    monkeypatch.setattr(benchmark_methods, "run_magic_direct", spy)
    dispatcher = RepositoryAdapterDispatcher(tmp_path, environments)
    from maskimpute_benchmark.comparator_tuning import (
        load_comparator_tuning_authority,
    )

    row = load_comparator_tuning_authority(
        Path.cwd(), registry=registry, require_clean=False
    ).configurations_for("magic")[0]

    outcome = dispatcher.direct_comparator_adapters()["magic"](
        spec,
        _method_input(),
        seed=42,
        config=row.decode(),
    )

    assert outcome.status == "unavailable"
    assert outcome.reason == "environment_executable_unavailable_magic"
    assert attempted == []


def test_saver_direct_wrapper_bypasses_all_legacy_content_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _adapter_module("saver")
    spec = _registry().by_id("saver")
    method_input = _method_input()
    source_dir = tmp_path / "source"
    library_dir = tmp_path / "library"
    source_dir.mkdir()
    library_dir.mkdir()
    for package in module._SAVER_PACKAGE_KEYS:
        (library_dir / package).mkdir()
    lock_manifest = tmp_path / "saver.lock.json"
    build_receipt = tmp_path / "saver.build.json"
    lock_manifest.write_text("{}", encoding="utf-8")
    build_receipt.write_text("{}", encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct SAVER called a content-summary helper")

    for helper in (
        "_load_saver_environment_lock",
        "_load_saver_build_receipt",
        "_saver_library_sha256",
        "_validate_saver_library",
        "snapshot_method_output",
    ):
        monkeypatch.setattr(module, helper, forbidden)
    executable = tmp_path / "Rscript"
    monkeypatch.setattr(module, "require_executable", lambda _path: executable)
    monkeypatch.setattr(
        module,
        "execute_pinned_command",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"out", stderr=b"err"),
    )
    monkeypatch.setattr(
        module,
        "read_raw_output",
        lambda _path, _shape: np.array(method_input.counts, copy=True),
    )

    execution = module.run_saver_direct(
        spec,
        method_input,
        source_dir=source_dir,
        rscript=executable,
        seed=42,
        library_dir=library_dir,
        lock_manifest=lock_manifest,
        build_receipt=build_receipt,
    )

    assert tuple(field.name for field in fields(execution)) == (
        "output",
        "stdout",
        "stderr",
    )


@pytest.mark.parametrize(
    "method_id",
    (
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "scziva",
        "afmf",
        "biaeimpute",
        "sccr",
        "scsdae",
    ),
)
def test_real_direct_wrappers_do_not_construct_legacy_summaries_on_missing_executable(
    method_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _adapter_module(method_id)
    method_input = _method_input(cells=120, genes=120)
    source_dir = tmp_path / method_id
    source_dir.mkdir()
    missing = tmp_path / f"missing-{method_id}"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct wrapper constructed a legacy content summary")

    monkeypatch.setattr(module, "snapshot_method_output", forbidden)
    if method_id == "scsdae":
        monkeypatch.setattr(
            module,
            "verify_pinned_source",
            lambda *_args, **_kwargs: SimpleNamespace(
                revision="revision", tree="tree", url="url"
            ),
        )
        monkeypatch.setattr(module, "SCSDaeUnavailableError", forbidden)
        monkeypatch.setattr(module, "SCSDaeAttemptReceipt", forbidden)
        monkeypatch.setattr(module.hashlib, "sha256", forbidden)
    monkeypatch.setattr(
        module,
        "require_executable",
        lambda _path: (_ for _ in ()).throw(
            AdapterUnavailableError(
                "environment_executable_missing",
                "synthetic missing executable",
                stdout=b"missing-out",
                stderr=b"missing-err",
            )
        ),
    )
    kwargs: dict[str, object] = {
        "source_dir": source_dir,
        "seed": 42,
    }
    if method_id in {"alra", "saver"}:
        kwargs["rscript"] = missing
    else:
        kwargs["python_executable"] = missing
    if method_id == "saver":
        kwargs.update(
            library_dir=tmp_path / "library",
            lock_manifest=tmp_path / "lock.json",
            build_receipt=tmp_path / "build.json",
        )

    with pytest.raises(AdapterUnavailableError) as captured:
        getattr(module, f"run_{method_id}_direct")(
            _registry().by_id(method_id),
            method_input,
            **kwargs,
        )

    assert type(captured.value) is AdapterUnavailableError
    assert captured.value.reason_code == "environment_executable_missing"
    assert captured.value.stdout == b"missing-out"
    assert captured.value.stderr == b"missing-err"
    assert not hasattr(captured.value, "attempt_receipt")


@pytest.mark.parametrize(
    ("failure_stage", "expected_reason", "expected_stdout", "expected_stderr"),
    (
        (
            "probe",
            "legacy_gpu_kernel_incompatible",
            b"MASKIMPUTE_LEGACY_GPU_KERNEL_INCOMPATIBLE",
            b"probe-err",
        ),
        (
            "run",
            "upstream_process_failed",
            b"probe-out\nrun-out",
            b"probe-err\nrun-err",
        ),
        (
            "malformed_output",
            "malformed_upstream_output",
            b"probe-out\nrun-out",
            b"probe-err\nrun-err",
        ),
    ),
)
def test_scsdae_direct_failure_paths_never_construct_attempt_receipts(
    failure_stage: str,
    expected_reason: str,
    expected_stdout: bytes,
    expected_stderr: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _adapter_module("scsdae")
    method_input = _method_input(cells=8, genes=6)
    source_dir = tmp_path / "scsdae"
    source_dir.mkdir()
    (source_dir / "pure_ae_new.py").write_text("# synthetic\n", encoding="utf-8")
    executable = tmp_path / "python"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct scSDAE constructed a legacy content summary")

    monkeypatch.setattr(
        module,
        "verify_pinned_source",
        lambda *_args, **_kwargs: SimpleNamespace(
            revision="revision", tree="tree", url="url"
        ),
    )
    monkeypatch.setattr(module, "require_executable", lambda _path: executable)
    monkeypatch.setattr(module, "SCSDaeUnavailableError", forbidden)
    monkeypatch.setattr(module, "SCSDaeAttemptReceipt", forbidden)
    monkeypatch.setattr(module.hashlib, "sha256", forbidden)
    calls = 0

    def fake_execute(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if failure_stage == "probe" and calls == 1:
            raise AdapterUnavailableError(
                "upstream_process_failed",
                "synthetic probe failure",
                command=(str(executable), "probe"),
                stdout=b"MASKIMPUTE_LEGACY_GPU_KERNEL_INCOMPATIBLE",
                stderr=b"probe-err",
            )
        if failure_stage == "run" and calls == 2:
            raise AdapterUnavailableError(
                "upstream_process_failed",
                "synthetic run failure",
                command=(str(executable), "run"),
                stdout=b"run-out",
                stderr=b"run-err",
            )
        if calls == 1:
            return SimpleNamespace(stdout=b"probe-out", stderr=b"probe-err")
        return SimpleNamespace(stdout=b"run-out", stderr=b"run-err")

    monkeypatch.setattr(module, "execute_pinned_command", fake_execute)
    monkeypatch.setattr(
        module,
        "read_npy_output",
        lambda _path: np.ones((1, 1), dtype=np.float64),
    )
    monkeypatch.setattr(module, "read_environment_receipt", lambda *_a, **_k: ())

    with pytest.raises(AdapterUnavailableError) as captured:
        module.run_scsdae_direct(
            _registry().by_id("scsdae"),
            method_input,
            source_dir=source_dir,
            python_executable=executable,
            seed=42,
        )

    assert type(captured.value) is AdapterUnavailableError
    assert captured.value.reason_code == expected_reason
    assert captured.value.stdout == expected_stdout
    assert captured.value.stderr == expected_stderr
    assert not hasattr(captured.value, "attempt_receipt")
    assert calls == (1 if failure_stage == "probe" else 2)


def test_sccr_direct_wrapper_uses_identity_free_driver_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _adapter_module("sccr")
    spec = _registry().by_id("sccr")
    method_input = _method_input(cells=20, genes=10)
    source_dir = tmp_path / "sccr"
    source_dir.mkdir()
    executable = tmp_path / "python"
    captured: dict[str, object] = {}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("direct scCR constructed a legacy content summary")

    def fake_execute(_spec, _source_dir, command, **_kwargs):
        driver = command[4]
        assert "graph_contract_sha256" not in driver
        assert "graph_contract_revision" not in driver
        assert "graph_contract_url" not in driver
        output_path = Path(command[7])
        receipt_path = Path(command[8])
        np.save(
            output_path,
            module.log1p_cp10k(method_input.counts),
            allow_pickle=False,
        )
        receipt = {
            "device": "cpu",
            "numpy_version": "2.0.0",
            "python_version": "3.12.0",
            "sccr_module": str(source_dir / "scCR.py"),
            "torch_num_threads": "3",
            "torch_version": "2.4.1",
        }
        receipt_path.write_text(
            "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
            encoding="utf-8",
        )
        captured["receipt"] = receipt
        captured["driver"] = driver
        return SimpleNamespace(stdout=b"out", stderr=b"err")

    monkeypatch.setattr(module, "snapshot_method_output", forbidden)
    monkeypatch.setattr(module, "require_executable", lambda _path: executable)
    monkeypatch.setattr(module, "execute_pinned_command", fake_execute)

    execution = module.run_sccr_direct(
        spec,
        method_input,
        source_dir=source_dir,
        python_executable=executable,
        seed=42,
        config=module.SCCRConfig(neighbors=3, gene_neighbors=1),
    )

    assert tuple(field.name for field in fields(execution)) == (
        "output",
        "stdout",
        "stderr",
    )
    assert set(captured["receipt"]) == {
        "device",
        "numpy_version",
        "python_version",
        "sccr_module",
        "torch_num_threads",
        "torch_version",
    }


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


def test_afmf_negative_native_output_fails_closed_with_bound_diagnostics(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=100, genes=100)
    source, spec = _fake_pinned_source(tmp_path, "afmf")
    launcher = _fake_negative_afmf_launcher(tmp_path)

    with pytest.raises(AdapterUnavailableError) as captured:
        afmf_adapter.run_afmf(
            spec,
            method_input,
            source_dir=source,
            python_executable=launcher,
            seed=42,
            config=afmf_adapter.AFMFConfig(iterations=2),
            work_root=tmp_path,
        )

    error = captured.value
    assert error.reason_code == "upstream_negative_native_output"
    assert error.detail == "afMF native output negative_count=2 minimum=-2.5"
    assert error.command is not None
    assert error.stdout == b"negative-native-output\n"
    assert error.stderr == (
        b"negative-native-stderr\n"
        b"MASKIMPUTE_AFMF_NATIVE_OUTPUT negative_count=2 minimum=-2.5\n"
    )


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
