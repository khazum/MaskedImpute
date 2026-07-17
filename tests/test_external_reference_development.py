from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
import warnings

import numpy as np
import pytest
from scipy.stats import boxcox

from maskimpute_benchmark.development_evaluation import (
    BaronSource,
    CiteSeqSource,
    OrthogonalInput,
    PreparedRealOrthogonalPanel,
    RealSourceEvidence,
    SourceArtifactBinding,
    SourceReceiptBinding,
    TungSource,
)
from maskimpute_benchmark.methods import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    MethodInput,
    SCTSIAttemptReceipt,
    SourceReceipt,
    finalize_d3impute_output,
    finalize_sctsi_output,
)
from maskimpute_benchmark.methods.d3impute import (
    D3ImputeConfig,
    _D3IMPUTE_DRIVER,
)
from maskimpute_benchmark.methods.sctsi import SCTSIConfig, _SCTSI_DRIVER
from maskimpute_benchmark.protocol import canonical_sha256
from maskimpute_benchmark.runtime_environments import (
    RuntimeEnvironmentEntry,
    RuntimeEnvironmentError,
    RuntimeEnvironmentLock,
)


REPOSITORY = Path(__file__).resolve().parents[1]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


@pytest.fixture(autouse=True)
def _restore_temporary_permissions(tmp_path: Path):
    yield
    for path in sorted(
        tmp_path.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        if path.is_symlink():
            continue
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except FileNotFoundError:
            pass


@dataclass
class _Fixture:
    repository: Path
    environments: dict[str, Path]
    r_library: Path
    runtime_state: dict[str, bool]
    adapter_inputs: list[tuple[str, str, str]]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _method_input(counts: np.ndarray) -> MethodInput:
    cells, genes = counts.shape
    return MethodInput(
        source_dataset_sha256=_sha("tung-single-cell-method-input"),
        obs_ids=tuple(f"cell-{index:03d}" for index in range(cells)),
        var_ids=tuple(f"gene-{index:03d}" for index in range(genes)),
        shape=(cells, genes),
        obs_covariates=(),
        var_covariates=(),
        _count_bytes=np.asarray(counts, dtype="<f8", order="C").tobytes(order="C"),
        _normalization_bytes=b"null",
    )


def _panel() -> PreparedRealOrthogonalPanel:
    cells = 30
    genes = 30
    rows = np.arange(cells, dtype=np.int64)[:, None]
    columns = np.arange(genes, dtype=np.int64)[None, :]
    counts = ((rows * 3 + columns * 5 + (rows % 4) * columns) % 17).astype(np.float64)
    counts[(rows + columns) % 7 == 0] = 0.0
    method_input = _method_input(counts)
    sample_names = ("sample-a", "sample-b", "sample-c")
    sample_ids = tuple(sample_names[index // 10] for index in range(cells))
    individual_ids = tuple(f"individual-{index // 10}" for index in range(cells))
    replicate_ids = tuple(f"lane-{index % 2}" for index in range(cells))
    bulk_profiles: dict[str, np.ndarray] = {}
    lane_profiles: dict[str, np.ndarray] = {}
    sample_array = np.asarray(sample_ids)
    gene_offset = np.arange(genes, dtype=np.float64) + 1.0
    for sample_index, sample in enumerate(sample_names):
        sample_profile = counts[sample_array == sample].sum(axis=0)
        bulk_profiles[sample] = sample_profile + gene_offset * (sample_index + 1)
        lane_profiles[f"{sample}:lane-a"] = sample_profile + gene_offset * (
            sample_index + 2
        )
    tung = TungSource(
        cell_ids=method_input.obs_ids,
        sample_ids=sample_ids,
        individual_ids=individual_ids,
        replicate_ids=replicate_ids,
        gene_ids=method_input.var_ids,
        counts=np.array(counts, copy=True),
        ercc_mask=np.asarray([index < 6 for index in range(genes)], dtype=np.bool_),
        bulk_profiles=MappingProxyType(bulk_profiles),
        lane_profiles=MappingProxyType(lane_profiles),
        single_sample_file_sha256=_sha("tung-single-sample"),
        bulk_sample_file_sha256=_sha("tung-bulk-sample"),
        single_lane_file_sha256=_sha("tung-single-lane"),
    )
    evidence = RealSourceEvidence(
        ledger_path="artifacts/external/source-ledger.json",
        ledger_file_sha256=_sha("source-ledger-file"),
        ledger_sha256=_sha("source-ledger-payload"),
        receipts=(
            SourceReceiptBinding(
                "tung-ipsc-ercc-bulk-replicates",
                "artifacts/external/receipts/tung.json",
                _sha("tung-receipt"),
            ),
        ),
        artifacts=(
            SourceArtifactBinding(
                "tung-ipsc-ercc-bulk-replicates",
                (
                    "artifacts/external/data/tung-ipsc-ercc-bulk-replicates/"
                    "GSE77288_reads-raw-bulk-per-sample.txt.gz"
                ),
                tung.bulk_sample_file_sha256,
                123,
            ),
        ),
    )
    return PreparedRealOrthogonalPanel(
        evidence,
        BaronSource(("human",), (3,), (3,), _sha("baron")),
        CiteSeqSource(
            ("cite-cell",),
            ("cite-gene",),
            ("cite-gene",),
            np.ones((1, 1), dtype=np.float64),
            ("protein",),
            np.ones((1, 1), dtype=np.float64),
            _sha("cite-rna"),
            _sha("cite-protein"),
        ),
        tung,
        (OrthogonalInput("tung-ipsc-ercc-bulk-replicates", method_input),),
    )


def _d3_native(counts: np.ndarray) -> np.ndarray:
    native = np.empty_like(counts, dtype=np.float64)
    for index in range(counts.shape[1]):
        observed = counts[:, index]
        shift = abs(float(observed.min())) + 1.0 if observed.min() <= 0 else 0.0
        native[:, index] = boxcox(observed + shift)[0]
    return native


def _make_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    unavailable: bool = False,
    structured_reason: bool = False,
    commandless: bool = False,
) -> _Fixture:
    import maskimpute_benchmark.external_reference_development as module

    repository = tmp_path / "repository"
    (repository / "study").mkdir(parents=True)
    shutil.copyfile(
        REPOSITORY / "study/methods.json", repository / "study/methods.json"
    )
    runtime_path = repository / "environments/development-runtime.lock.json"
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_bytes(b"{}\n")
    runtime_file_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    methods_path = repository / "study/methods.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    for row in methods["methods"]:
        if row["id"] in {"d3impute", "sctsi"}:
            row["environment"] = {
                "id": row["environment"]["id"],
                "status": "ready",
                "lock_sha256": runtime_file_sha,
            }
    _canonical_write(methods_path, methods)
    for method_id in ("d3impute", "sctsi"):
        (repository / f"artifacts/method-sources/{method_id}").mkdir(parents=True)
    executable_root = tmp_path / "executables"
    executable_root.mkdir()
    environments = {
        "d3impute": executable_root / "python",
        "sctsi": executable_root / "Rscript",
    }
    for path in environments.values():
        path.write_bytes(b"#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    r_library = tmp_path / "sctsi-library"
    r_library.mkdir()

    panel = _panel()
    entries = tuple(
        RuntimeEnvironmentEntry(
            method_id,
            "python" if method_id == "d3impute" else "r",
            b"{}",
            _sha(f"{method_id}-inventory"),
        )
        for method_id in ("d3impute", "sctsi")
    )
    lock = RuntimeEnvironmentLock(runtime_path, runtime_file_sha, entries)
    runtime_state = {"valid": True}
    adapter_inputs: list[tuple[str, str, str]] = []

    monkeypatch.setattr(module, "prepare_real_orthogonal_panel", lambda _root: panel)
    monkeypatch.setattr(module, "load_runtime_environment_lock", lambda _path: lock)

    def validate_runtime(
        selected_lock: RuntimeEnvironmentLock,
        environment_id: str,
        kind: str,
        executable: Path,
        *,
        r_library_paths: tuple[Path, ...] = (),
    ) -> str:
        assert selected_lock is lock
        assert executable == environments[environment_id]
        assert kind == ("python" if environment_id == "d3impute" else "r")
        assert r_library_paths == (() if kind == "python" else (r_library,))
        if not runtime_state["valid"]:
            raise RuntimeEnvironmentError("runtime inventory mismatch for test")
        return lock.by_id(environment_id).inventory_sha256

    monkeypatch.setattr(module, "validate_runtime_environment_entry", validate_runtime)

    def verify_source(spec, source_dir: Path) -> SourceReceipt:
        assert source_dir == repository / f"artifacts/method-sources/{spec.id}"
        return SourceReceipt(spec.source.revision, spec.source.tree, spec.source.url)

    monkeypatch.setattr(module, "verify_pinned_source", verify_source)

    def run_d3(spec, method_input, *, bulk_reference, **_kwargs):
        adapter_inputs.append(
            (spec.id, method_input.source_dataset_sha256, bulk_reference.source_sha256)
        )
        config = D3ImputeConfig()
        work_dir = _kwargs["work_root"] / "maskimpute-d3impute-test"
        command = (
            str(_kwargs["python_executable"]),
            "-B",
            "-I",
            "-c",
            _D3IMPUTE_DRIVER,
            str((_kwargs["source_dir"] / "PYTHON.zip").resolve()),
            str(work_dir / "input.npy"),
            str(work_dir / "bulk.npy"),
            str(work_dir / "output.npy"),
            str(work_dir / "receipt.tsv"),
            str(config.fixed_seed),
            str(config.neighbors),
            str(config.latent_dimension),
            str(config.iterations),
            repr(float(config.sparsity)),
            repr(float(config.cell_regularization)),
            repr(float(config.gene_regularization)),
            bulk_reference.reference_id,
            bulk_reference.source_sha256,
        )
        if unavailable:
            reason = (
                "adapter_exception:d3impute:ValueError"
                if structured_reason
                else "dependency_missing"
            )
            raise AdapterUnavailableError(
                reason,
                "D3 development attempt did not start",
                command=(
                    None if commandless else command
                ),
                stdout=b"d3-out",
                stderr=b"d3-err",
            )
        native = _d3_native(np.asarray(method_input.counts, dtype=np.float64))
        return AdapterExecution(
            finalize_d3impute_output(spec, method_input, native),
            tuple(
                CompatibilityEvent(code, f"fixed D3 disclosure for {code}")
                for code in (
                    "external_reference_binding",
                    "source_archive_execution",
                    "upstream_parameters",
                    "fixed_rng_compatibility",
                    "evaluation_label_exclusion",
                    "evaluator_scale_conversion",
                )
            ),
            (
                ("bulk_reference_id", bulk_reference.reference_id),
                ("bulk_reference_sha256", bulk_reference.source_sha256),
                (
                    "inference_module",
                    str(
                        (_kwargs["source_dir"] / "PYTHON.zip").resolve()
                    )
                    + "/PYTHON/Function/Inference.py",
                ),
                ("numpy_version", "test"),
                ("pandas_version", "test"),
                ("python_version", "test"),
                ("scipy_version", "test"),
                ("sklearn_version", "test"),
                (
                    "source_archive",
                    str((_kwargs["source_dir"] / "PYTHON.zip").resolve()),
                ),
            ),
            b"d3-out",
            b"d3-err",
            command,
        )

    def run_sctsi_adapter(spec, method_input, *, bulk_reference, **_kwargs):
        adapter_inputs.append(
            (spec.id, method_input.source_dataset_sha256, bulk_reference.source_sha256)
        )
        config = SCTSIConfig()
        work_dir = _kwargs["work_root"] / "maskimpute-sctsi-test"
        cells, genes = method_input.shape
        command = (
            str(_kwargs["rscript"]),
            "--vanilla",
            "-e",
            _SCTSI_DRIVER,
            str((_kwargs["source_dir"] / "code/scTsI.R").resolve()),
            str(work_dir / "input.bin"),
            str(work_dir / "bulk-average.bin"),
            str(work_dir / "output.bin"),
            str(work_dir / "receipt.tsv"),
            str(_kwargs["r_library"]),
            str(genes),
            str(cells),
            repr(float(config.threshold)),
            str(config.cell_neighbors),
            str(config.gene_neighbors),
            bulk_reference.reference_id,
            bulk_reference.source_sha256,
            bulk_reference.matrix_sha256,
        )
        if unavailable:
            error = AdapterUnavailableError(
                "dependency_missing",
                "scTsI development attempt did not start",
                command=(
                    None if commandless else command
                ),
                stdout=b"sctsi-out",
                stderr=b"sctsi-err",
            )
            error.attempt_receipt = SCTSIAttemptReceipt(
                source_revision=spec.source.revision,
                source_tree=spec.source.tree,
                source_url=spec.source.url,
                environment_id=spec.environment.id,
                environment_registry_status=spec.environment.status,
                executable=str(environments["sctsi"]),
                r_library=str(r_library),
                reference_id=bulk_reference.reference_id,
                reference_source_sha256=bulk_reference.source_sha256,
                reference_matrix_sha256=bulk_reference.matrix_sha256,
                outcome="unavailable",
                reason_code=error.reason_code,
                command=error.command,
                stdout_sha256=error.stdout_sha256,
                stderr_sha256=error.stderr_sha256,
            )
            raise error
        counts = np.asarray(method_input.counts, dtype=np.float64)
        native = counts / counts.sum(axis=1, keepdims=True) * 1_000_000.0
        return AdapterExecution(
            finalize_sctsi_output(spec, method_input, native),
            tuple(
                CompatibilityEvent(code, f"fixed scTsI disclosure for {code}")
                for code in (
                    "input_scale_conversion",
                    "input_orientation",
                    "bulk_average_contract",
                    "published_demo_truth_exclusion",
                    "upstream_defaults",
                    "deterministic_execution",
                    "upstream_selective_policy",
                    "evaluator_scale_conversion",
                    "source_policy",
                )
            ),
            (
                ("bulk_constraint_scale", "cpm"),
                ("bulk_reference_id", bulk_reference.reference_id),
                ("bulk_reference_input_scale", "raw_counts"),
                (
                    "bulk_reference_matrix_sha256",
                    bulk_reference.matrix_sha256,
                ),
                (
                    "bulk_reference_source_sha256",
                    bulk_reference.source_sha256,
                ),
                ("cpm_target", "1000000"),
                ("devtools_version", "test"),
                ("fnn_version", "test"),
                ("fpc_version", "test"),
                ("glmnet_version", "test"),
                ("matrix_version", "test"),
                ("mclust_version", "test"),
                ("metrics_version", "test"),
                ("ngram_version", "test"),
                ("r_library", str(_kwargs["r_library"])),
                (
                    "r_library_paths",
                    f"{_kwargs['r_library']};/fixed/base-r-library",
                ),
                ("r_version", "test"),
                ("sctsi_native_output_scale", "cpm"),
                ("single_cell_input_scale", "cpm"),
                (
                    "sctsi_source_file",
                    str((_kwargs["source_dir"] / "code/scTsI.R").resolve()),
                ),
            ),
            b"sctsi-out",
            b"sctsi-err",
            command,
        )

    monkeypatch.setattr(module, "run_d3impute", run_d3)
    monkeypatch.setattr(module, "run_sctsi", run_sctsi_adapter)
    return _Fixture(repository, environments, r_library, runtime_state, adapter_inputs)


def _make_tree_writable(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.exists():
            mode = candidate.stat().st_mode
            needed = 0o300 if candidate.is_dir() else 0o200
            if candidate != path and mode & needed == needed:
                break
            candidate.chmod(mode | needed)


def _flip_byte(path: Path) -> None:
    _make_tree_writable(path)
    raw = bytearray(path.read_bytes())
    assert raw
    raw[0] ^= 1
    path.write_bytes(raw)


def _canonical_write(path: Path, value: object) -> None:
    _make_tree_writable(path)
    path.write_bytes(
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )


def test_producer_writes_reopenable_semantic_tung_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        OUTPUT_RELATIVE_PATH,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    produced = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    reopened = load_external_reference_evidence(fixture.repository)

    assert produced.checkpoint == reopened.checkpoint
    assert produced.output_directory == fixture.repository / OUTPUT_RELATIVE_PATH
    assert produced.dataset_id == "tung-ipsc-ercc-bulk-replicates"
    assert produced.method_ids == ("d3impute", "sctsi")
    assert fixture.adapter_inputs == [
        (
            method_id,
            _sha("tung-single-cell-method-input"),
            _sha("tung-bulk-sample"),
        )
        for method_id in ("d3impute", "sctsi")
    ]
    plan = json.loads((reopened.output_directory / "plan.json").read_text())
    assert plan["scientific_design"]["independent_endpoint_ids"] == []
    assert plan["scientific_design"]["non_matched_bulk_endpoint_ids"] == [
        "technical_replicate_concordance"
    ]
    assert (
        plan["scientific_design"]["independence_disclosure"]
        == "no_endpoint_is_an_independent_validation_cohort"
    )
    for record in reopened.checkpoint["records"]:
        assert record["run"]["status"] == "completed"
        metrics_path = (
            reopened.output_directory / record["artifacts"]["metrics"]["path"]
        )
        metrics = json.loads(metrics_path.read_text())
        assert [row["endpoint"] for row in metrics["endpoints"]] == [
            "bulk_pseudobulk_concordance",
            "ercc_recovery",
            "technical_replicate_concordance",
        ]
        assert all(
            row["status"] == "completed" and row["units"]
            for row in metrics["endpoints"]
        )
        assert {
            row["endpoint"]: row["reference_overlap"]
            for row in metrics["endpoints"]
        } == {
            "bulk_pseudobulk_concordance": "adapter_input_matched_bulk",
            "ercc_recovery": "adapter_input_matched_bulk",
            "technical_replicate_concordance": "same_experiment_technical_lane",
        }


def test_producer_rejects_nonready_registry_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    methods_path = fixture.repository / "study/methods.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    for row in methods["methods"]:
        if row["id"] in {"d3impute", "sctsi"}:
            row["environment"] = {
                "id": row["environment"]["id"],
                "status": "pending",
                "lock_sha256": None,
            }
    _canonical_write(methods_path, methods)

    with pytest.raises(ExternalReferenceDevelopmentError, match="ready.*runtime lock"):
        run_external_reference_development(
            fixture.repository,
            environments=fixture.environments,
            sctsi_library=fixture.r_library,
        )


def test_producer_rejects_symlinked_output_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    outside = tmp_path / "outside-study"
    outside.mkdir()
    os.symlink(outside, fixture.repository / "artifacts/study")

    with pytest.raises(ExternalReferenceDevelopmentError, match="symlink|unsafe"):
        run_external_reference_development(
            fixture.repository,
            environments=fixture.environments,
            sctsi_library=fixture.r_library,
        )

    assert not (outside / "development").exists()


def test_producer_refuses_to_overwrite_existing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )

    with pytest.raises(ExternalReferenceDevelopmentError, match="already exists"):
        run_external_reference_development(
            fixture.repository,
            environments=fixture.environments,
            sctsi_library=fixture.r_library,
        )


def test_producer_atomically_refuses_output_created_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.external_reference_development as module

    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        OUTPUT_RELATIVE_PATH,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    final_output = fixture.repository / OUTPUT_RELATIVE_PATH
    original_publish = module._rename_directory_noreplace
    injected_inode: int | None = None

    def inject_empty_output_before_publish(source: Path, destination: Path) -> None:
        nonlocal injected_inode
        assert destination == final_output
        final_output.mkdir()
        injected_inode = final_output.stat().st_ino
        original_publish(source, destination)

    monkeypatch.setattr(
        module, "_rename_directory_noreplace", inject_empty_output_before_publish
    )

    with pytest.raises(ExternalReferenceDevelopmentError, match="appeared|overwrite"):
        run_external_reference_development(
            fixture.repository,
            environments=fixture.environments,
            sctsi_library=fixture.r_library,
        )

    assert injected_inode is not None
    assert final_output.stat().st_ino == injected_inode
    assert list(final_output.iterdir()) == []


@pytest.mark.parametrize(
    "artifact_name",
    ["input_counts", "bulk_reference", "stdout", "native_output"],
)
def test_loader_rejects_bound_byte_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    if artifact_name in {"input_counts", "bulk_reference"}:
        binding = evidence.checkpoint["artifacts"][artifact_name]
    else:
        binding = evidence.checkpoint["records"][0]["artifacts"][artifact_name]
    _flip_byte(evidence.output_directory / binding["path"])

    with pytest.raises(ExternalReferenceDevelopmentError, match="checksum|bytes"):
        load_external_reference_evidence(fixture.repository)


def test_loader_semantically_rejects_empty_metrics_even_when_rehashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    checkpoint = json.loads((evidence.output_directory / "checkpoint.json").read_text())
    binding = checkpoint["records"][0]["artifacts"]["metrics"]
    metrics_path = evidence.output_directory / binding["path"]
    _canonical_write(
        metrics_path,
        {
            "schema_version": 1,
            "dataset_id": evidence.dataset_id,
            "method_id": "d3impute",
            "status": "completed",
            "endpoints": [],
        },
    )
    raw = metrics_path.read_bytes()
    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    binding["size_bytes"] = len(raw)
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(unsigned)
    _canonical_write(evidence.output_directory / "checkpoint.json", checkpoint)

    with pytest.raises(ExternalReferenceDevelopmentError, match="endpoint|metrics"):
        load_external_reference_evidence(fixture.repository)


def test_loader_rejects_rehashed_relocation_of_fixed_input_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    output = evidence.output_directory
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    plan_path = output / checkpoint["artifacts"]["plan"]["path"]
    plan = json.loads(plan_path.read_text())
    old_path = output / checkpoint["artifacts"]["input_counts"]["path"]
    new_relative = "inputs/renamed-tung-counts.f64"
    new_path = output / new_relative
    _make_tree_writable(old_path)
    old_path.rename(new_path)
    checkpoint["artifacts"]["input_counts"]["path"] = new_relative
    plan["artifacts"]["input_counts"]["path"] = new_relative
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    _canonical_write(plan_path, plan)
    plan_raw = plan_path.read_bytes()
    checkpoint["artifacts"]["plan"].update(
        {
            "sha256": hashlib.sha256(plan_raw).hexdigest(),
            "size_bytes": len(plan_raw),
        }
    )
    checkpoint["plan_sha256"] = plan["plan_sha256"]
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    _canonical_write(checkpoint_path, checkpoint)

    with pytest.raises(ExternalReferenceDevelopmentError, match="fixed|path"):
        load_external_reference_evidence(fixture.repository)


def test_loader_rejects_runtime_drift_after_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    fixture.runtime_state["valid"] = False

    with pytest.raises(ExternalReferenceDevelopmentError, match="runtime"):
        load_external_reference_evidence(fixture.repository)


def test_loader_rejects_loss_of_immutable_file_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    evidence.checkpoint_path.chmod(0o644)

    with pytest.raises(ExternalReferenceDevelopmentError, match="immutable"):
        load_external_reference_evidence(fixture.repository)


def test_loader_rejects_handwritten_checkpoint_without_producer_artifacts(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        OUTPUT_RELATIVE_PATH,
        load_external_reference_evidence,
    )

    repository = tmp_path / "repository"
    output = repository / OUTPUT_RELATIVE_PATH
    output.mkdir(parents=True)
    body = {
        "schema_version": 1,
        "track": "external_reference",
        "status": "completed",
        "method_ids": ["d3impute", "sctsi"],
        "eligible_dataset_ids": ["tung-ipsc-ercc-bulk-replicates"],
        "planned_run_count": 2,
        "records": [],
    }
    _canonical_write(
        output / "checkpoint.json", body | {"checkpoint_sha256": canonical_sha256(body)}
    )

    with pytest.raises(ExternalReferenceDevelopmentError):
        load_external_reference_evidence(repository)


def test_unavailable_attempts_remain_terminal_and_auditable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(
        tmp_path, monkeypatch, unavailable=True, structured_reason=True
    )
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    reopened = load_external_reference_evidence(fixture.repository)

    assert reopened.checkpoint == evidence.checkpoint
    for record in evidence.checkpoint["records"]:
        run = record["run"]
        assert run["status"] == "unavailable"
        assert run["reason"].replace("_", "").isalnum()
        assert ":" not in run["reason"]
        assert len(run["reason_detail_sha256"]) == 64
        assert "native_output" not in record["artifacts"]
        metrics = json.loads(
            (
                evidence.output_directory / record["artifacts"]["metrics"]["path"]
            ).read_text()
        )
        assert len(metrics["endpoints"]) == 3
        assert all(
            endpoint["status"] == "unavailable"
            and endpoint["reason"] == run["reason"]
            and endpoint["reason_detail_sha256"] == run["reason_detail_sha256"]
            for endpoint in metrics["endpoints"]
        )


def test_loader_rejects_rehashed_sctsi_attempt_receipt_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.external_reference_development as module

    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch, unavailable=True)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    output = evidence.output_directory
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint["records"][1]
    binding = record["artifacts"]["environment"]
    environment_path = output / binding["path"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["adapter_attempt_receipt"]["reference_source_sha256"] = "0" * 64
    _canonical_write(environment_path, environment)
    raw = environment_path.read_bytes()
    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    binding["size_bytes"] = len(raw)
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    _canonical_write(checkpoint_path, checkpoint)
    module._make_read_only(output)

    with pytest.raises(ExternalReferenceDevelopmentError, match="attempt receipt"):
        load_external_reference_evidence(fixture.repository)


def test_loader_rejects_rehashed_nondefault_adapter_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.external_reference_development as module

    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    output = evidence.output_directory
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    binding = checkpoint["records"][0]["artifacts"]["environment"]
    environment_path = output / binding["path"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["command"][1] = "--tampered"
    _canonical_write(environment_path, environment)
    raw = environment_path.read_bytes()
    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    binding["size_bytes"] = len(raw)
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    _canonical_write(checkpoint_path, checkpoint)
    module._make_read_only(output)

    with pytest.raises(ExternalReferenceDevelopmentError, match="command"):
        load_external_reference_evidence(fixture.repository)


def test_loader_rejects_rehashed_completed_reference_receipt_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.external_reference_development as module

    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    output = evidence.output_directory
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    binding = checkpoint["records"][0]["artifacts"]["environment"]
    environment_path = output / binding["path"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    receipt = dict(environment["adapter_environment_receipt"])
    receipt["bulk_reference_sha256"] = "0" * 64
    environment["adapter_environment_receipt"] = [
        [key, receipt[key]] for key in sorted(receipt)
    ]
    _canonical_write(environment_path, environment)
    raw = environment_path.read_bytes()
    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    binding["size_bytes"] = len(raw)
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    _canonical_write(checkpoint_path, checkpoint)
    module._make_read_only(output)

    with pytest.raises(ExternalReferenceDevelopmentError, match="environment receipt"):
        load_external_reference_evidence(fixture.repository)


def test_loader_rejects_rehashed_incomplete_compatibility_disclosure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.external_reference_development as module

    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    evidence = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )
    output = evidence.output_directory
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    binding = checkpoint["records"][1]["artifacts"]["environment"]
    environment_path = output / binding["path"]
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["compatibility_log"].pop()
    _canonical_write(environment_path, environment)
    raw = environment_path.read_bytes()
    binding["sha256"] = hashlib.sha256(raw).hexdigest()
    binding["size_bytes"] = len(raw)
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    _canonical_write(checkpoint_path, checkpoint)
    module._make_read_only(output)

    with pytest.raises(ExternalReferenceDevelopmentError, match="compatibility"):
        load_external_reference_evidence(fixture.repository)


def test_pre_command_unavailability_is_persisted_as_a_terminal_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        load_external_reference_evidence,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch, unavailable=True, commandless=True)
    produced = run_external_reference_development(
        fixture.repository,
        environments=fixture.environments,
        sctsi_library=fixture.r_library,
    )

    assert load_external_reference_evidence(fixture.repository).checkpoint == (
        produced.checkpoint
    )
    for record in produced.checkpoint["records"]:
        environment = json.loads(
            (
                produced.output_directory / record["artifacts"]["environment"]["path"]
            ).read_text()
        )
        assert environment["command"] is None


def test_operational_locators_are_fixed_absolute_and_non_symlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
        run_external_reference_development,
    )

    fixture = _make_fixture(tmp_path, monkeypatch)
    with pytest.raises(ExternalReferenceDevelopmentError, match="exactly"):
        run_external_reference_development(
            fixture.repository,
            environments={"d3impute": fixture.environments["d3impute"]},
            sctsi_library=fixture.r_library,
        )
    link = tmp_path / "python-link"
    os.symlink(fixture.environments["d3impute"], link)
    with pytest.raises(ExternalReferenceDevelopmentError, match="symlink"):
        run_external_reference_development(
            fixture.repository,
            environments=fixture.environments | {"d3impute": link},
            sctsi_library=fixture.r_library,
        )


def _load_entrypoint():
    path = REPOSITORY / "scripts/run_external_reference_development.py"
    spec = importlib.util.spec_from_file_location(
        "run_external_reference_development_entrypoint", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_only_approved_operational_locators() -> None:
    entrypoint = _load_entrypoint()
    parser = entrypoint._parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "--help" and option != "-h"
    }

    assert option_strings == {"--environment", "--sctsi-library"}
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-dir", "/tmp/forbidden"])


def test_cli_rejects_duplicate_or_unknown_environment_locators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = _load_entrypoint()
    called = False

    def forbidden_call(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid CLI locators reached the producer")

    monkeypatch.setattr(
        entrypoint, "run_external_reference_development", forbidden_call
    )
    python = tmp_path / "python"
    rscript = tmp_path / "Rscript"
    library = tmp_path / "library"
    for path in (python, rscript):
        path.write_text("executable")
    library.mkdir()

    for argv in (
        [
            "--environment",
            f"d3impute={python}",
            "--environment",
            f"d3impute={python}",
            "--sctsi-library",
            str(library),
        ],
        [
            "--environment",
            f"d3impute={python}",
            "--environment",
            f"unknown={rscript}",
            "--sctsi-library",
            str(library),
        ],
    ):
        with pytest.raises(SystemExit):
            entrypoint.main(argv)
    assert called is False


@integration
@pytest.mark.skipif(
    os.environ.get("MASKIMPUTE_RUN_EXTERNAL_REFERENCE_INTEGRATION") != "1",
    reason=(
        "set MASKIMPUTE_RUN_EXTERNAL_REFERENCE_INTEGRATION=1 with the three "
        "fixed locator variables to run the production Tung track"
    ),
)
def test_real_external_reference_producer_and_loader_when_locked_assets_exist() -> (
    None
):
    from maskimpute_benchmark.external_reference_development import (
        OUTPUT_RELATIVE_PATH,
        load_external_reference_evidence,
        run_external_reference_development,
    )

    raw_locators = {
        "d3impute": os.environ.get("MASKIMPUTE_D3IMPUTE_PYTHON"),
        "sctsi": os.environ.get("MASKIMPUTE_SCTSI_RSCRIPT"),
    }
    raw_library = os.environ.get("MASKIMPUTE_SCTSI_LIBRARY")
    if any(value is None for value in raw_locators.values()) or raw_library is None:
        pytest.skip("fixed external-reference integration locators are not supplied")
    environments = {
        method_id: Path(str(value)) for method_id, value in raw_locators.items()
    }
    output = REPOSITORY / OUTPUT_RELATIVE_PATH
    evidence = (
        load_external_reference_evidence(REPOSITORY)
        if output.exists()
        else run_external_reference_development(
            REPOSITORY,
            environments=environments,
            sctsi_library=Path(raw_library),
        )
    )

    assert evidence.method_ids == ("d3impute", "sctsi")
    assert evidence.dataset_id == "tung-ipsc-ercc-bulk-replicates"
    assert evidence.checkpoint["status"] == "completed"
