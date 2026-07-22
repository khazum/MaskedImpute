from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import zlib

import anndata as ad
import numpy as np
import pytest

from maskimpute import ImputationResult
from maskimpute.ablations import AblationRunResult
from maskimpute_benchmark.methods import (
    MethodInput,
    load_method_registry,
    run_observed,
    snapshot_method_output,
)
from maskimpute_benchmark.methods.maskimpute import MaskImputeAdapterExecution
from maskimpute_benchmark.runner import (
    AdapterOutcome,
    CalibrationFoldReceipt,
    CheckpointStore,
    CompetitionPlan,
    DatasetBinding,
    DatasetQCAudit,
    DatasetQCPolicy,
    DevelopmentBudget,
    PreparedDataset,
    RunPlanEntry,
    RunnerContractError,
    evaluate_adapter_outcome,
    implementation_source_sha256,
    replay_development_budget,
)
from maskimpute_benchmark.protocol import canonical_sha256


METHODS = Path("study/methods.json")
SHA_A = "a" * 64


def _score_input_sha256(prepared: PreparedDataset) -> str:
    from maskimpute.count_model import _counts_sha256

    return _counts_sha256(prepared.method_input.counts)


def _prepared_datasets(
    prepared: PreparedDataset,
) -> dict[str, PreparedDataset]:
    return {prepared.binding.dataset_id: prepared}


def _budget_for_attempt(plan: CompetitionPlan, attempt) -> DevelopmentBudget:
    return replay_development_budget(
        load_method_registry(METHODS),
        plan.entries,
        (
            {
                "run": {
                    "status": attempt.run.status,
                    "runtime_seconds": attempt.run.runtime_seconds,
                }
            },
        ),
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


def _write_real_score_authority(
    repository: Path,
    prepared: PreparedDataset,
):
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model
    from maskimpute.calibration import save_calibration_artifact
    from maskimpute_benchmark.development_scores import (
        _manifest_payload,
        _score_entry,
        _score_filename,
        save_count_score_artifact,
    )
    from maskimpute_benchmark.runner import ExecutionAuthorityContext

    count_config = PreZeroCountModelConfig(n_folds=2, link_max_iter=25)
    ablation_path = repository / "study/ablations.json"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_bytes(Path("study/ablations.json").read_bytes())
    score = fit_p_pre_zero_count_model(
        prepared.method_input.counts,
        prepared.method_input.obs_ids,
        count_config,
    )
    score_directory = repository / "scores"
    score_directory.mkdir(parents=True)
    save_count_score_artifact(score_directory / _score_filename(prepared), score)
    manifest = _manifest_payload(
        [_score_entry(prepared, score)],
        dataset_manifest_sha256=prepared.binding.manifest_sha256,
        count_model_config_sha256=score.config_sha256,
        dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
    )
    manifest_path = score_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    calibration = _identity_calibration_artifact()
    calibration_path = repository / "calibration.json"
    save_calibration_artifact(calibration_path, calibration)
    base = {}
    count_payload = asdict(count_config)
    context = ExecutionAuthorityContext(
        authority_sha256="9" * 64,
        base_configuration_json=json.dumps(base, separators=(",", ":"), sort_keys=True),
        base_configuration_sha256=canonical_sha256(base),
        count_model_config_json=json.dumps(
            count_payload, separators=(",", ":"), sort_keys=True
        ),
        count_model_config_sha256=canonical_sha256(count_payload),
        count_score_manifest_path="scores/manifest.json",
        count_score_manifest_sha256=hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        retained_calibration_path="calibration.json",
        retained_calibration_sha256=hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
    )
    return context, score, calibration


def _prepared(mechanism: str = "symsim") -> PreparedDataset:
    from maskimpute_benchmark.development_scores import canonical_cell_ids_sha256

    counts = np.array([[0.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]], dtype="<f8")
    cells = ("cell-1", "cell-2")
    genes = ("gene-1", "gene-2", "gene-3", "gene-4")
    truth_kind = {
        "symsim": "exact_pre_capture",
        "sergio": "exact_continuous",
        "sparsim": "exact_continuous",
        "semisynthetic": "proxy_high_depth",
    }[mechanism]
    truth_layer = {
        "exact_pre_capture": "pre_capture_counts",
        "exact_continuous": "latent_expression",
        "proxy_high_depth": "reference_counts",
    }[truth_kind]
    truth = np.array([[0.0, 1.0, 2.0, 5.0], [0.0, 3.0, 1.0, 0.0]], dtype="<f8")
    dataset = ad.AnnData(X=counts.copy())
    dataset.obs_names = list(cells)
    dataset.var_names = list(genes)
    dataset.layers[truth_layer] = truth
    dataset.uns["truth_kind"] = truth_kind
    dataset.uns["primary_truth_layer"] = truth_layer
    method_input = MethodInput(
        source_dataset_sha256=SHA_A,
        obs_ids=cells,
        var_ids=genes,
        shape=counts.shape,
        obs_covariates=(),
        var_covariates=(),
        _count_bytes=counts.tobytes(order="C"),
        _normalization_bytes=b"{}",
    )
    empty_ids_sha = hashlib.sha256(
        b"maskimpute-external-cell-ids-v1\0\x00\x00\x00\x00\x00\x00\x00\x00"
    ).hexdigest()
    audit = DatasetQCAudit(
        excluded_cell_count=0,
        excluded_cell_ids_sha256=empty_ids_sha,
        retained_cell_count=2,
        retained_cell_ids_sha256=canonical_cell_ids_sha256(cells),
        excluded_cell_ids=(),
        retained_cell_ids=cells,
    )
    binding = DatasetBinding(
        mechanism=mechanism,
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id=f"dataset-{mechanism}",
        dataset_sha256=SHA_A,
        output_file_sha256="c" * 64,
        truth_sha256="d" * 64,
        output_path=f"{mechanism}.h5ad",
        independent_unit_id=f"unit-{mechanism}",
        cells=2,
        genes=4,
        manifest_sha256="e" * 64,
        protocol_sha256="f" * 64,
        design_sha256="1" * 64,
        seed_source_sha256="2" * 64,
    )
    return PreparedDataset(
        binding=binding,
        audit=audit,
        method_input=method_input,
        evaluator_dataset=dataset,
    )


def _entry(prepared: PreparedDataset, method_id: str = "maskimpute") -> RunPlanEntry:
    return RunPlanEntry(
        ordinal=1,
        run_id=f"run-{method_id}-{prepared.binding.mechanism}",
        method_id=method_id,
        dataset_id=prepared.binding.dataset_id,
        source_dataset_sha256=prepared.binding.dataset_sha256,
        mechanism=prepared.binding.mechanism,
        biological_id=prepared.binding.biological_id,
        technical_view=prepared.binding.technical_view,
        model_seed=42 if method_id == "maskimpute" else None,
        configuration_id="v27-reference"
        if method_id == "maskimpute"
        else "registry-default",
        configuration_sha256="3" * 64,
        preflight_status="planned",
        preflight_reason=None,
        configuration_kind="candidate_search"
        if method_id == "maskimpute"
        else "registry",
        requires_count_score=method_id == "maskimpute",
        requires_calibration=method_id == "maskimpute",
    )


def _maskimpute_execution(
    prepared: PreparedDataset,
    probability: np.ndarray | None = None,
    score_diagnostics: dict[str, object] | None = None,
) -> MaskImputeAdapterExecution:
    if probability is None:
        probability = np.array(
            [[0.8, 0.2, 0.0, 0.4], [0.7, 0.0, 0.3, 0.9]], dtype="<f8"
        )
    if score_diagnostics is None:
        score_diagnostics = {
            "source": "retained_calibrator",
            "score_artifact_sha256": "4" * 64,
            "score_input_sha256": _score_input_sha256(prepared),
            "score_config_sha256": "6" * 64,
            "calibration_file_sha256": "7" * 64,
            "calibration_payload_sha256": "8" * 64,
            "retained_calibrator": "identity",
            "calibration_scope": "leave_one_biological_draw_out",
            "equivalence_reason": "retained_identity_calibrator_equals_direct_score",
        }
    result = ImputationResult(
        selective_counts=prepared.method_input.counts,
        denoised_counts=prepared.method_input.counts,
        p_pre_zero=probability,
        latent=np.ones((2, 1), dtype="<f8"),
        diagnostics={"score": score_diagnostics},
    )
    spec = load_method_registry(METHODS).by_id("maskimpute")
    snapshot = snapshot_method_output(
        spec,
        prepared.method_input,
        result.selective_counts,
        source_dataset_sha256=prepared.method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=prepared.method_input.obs_ids,
        var_ids=prepared.method_input.var_ids,
    )
    return MaskImputeAdapterExecution(
        snapshot=snapshot,
        compatibility_log=(),
        environment_receipt=(),
        stdout=b"",
        stderr=b"",
        command=None,
        ablation_result=AblationRunResult(output_policy="selective", _result=result),
    )


def _completed_maskimpute(
    prepared: PreparedDataset,
    *,
    execution: MaskImputeAdapterExecution | None = None,
    calibration_file_sha256: str = "7" * 64,
    output_converter=None,
):
    arguments = {}
    if output_converter is not None:
        arguments["output_converter"] = output_converter
    return evaluate_adapter_outcome(
        _entry(prepared),
        prepared,
        AdapterOutcome.completed(
            _maskimpute_execution(prepared) if execution is None else execution,
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=1,
            calibration_fold_receipt=CalibrationFoldReceipt(
                calibration_artifact_sha256=calibration_file_sha256,
                calibration_context_sha256="8" * 64,
                mechanism=prepared.binding.mechanism,
                biological_id=prepared.binding.biological_id,
                training_manifest_sha256s=("9" * 64,),
                held_out_manifest_sha256s=("a" * 64,),
                fold_calibrator_sha256="b" * 64,
            ),
        ),
        **arguments,
    )


def _plan(prepared: PreparedDataset) -> CompetitionPlan:
    return CompetitionPlan(
        schema_version=1,
        input_hashes={
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "implementation_source_sha256": implementation_source_sha256(),
            "count_model_config_sha256": "6" * 64,
            "retained_calibration_sha256": "7" * 64,
        },
        entries=(_entry(prepared),),
        plan_sha256="c" * 64,
    )


def _real_development_checkpoint_case(
    repository: Path,
    prepared: PreparedDataset,
) -> tuple[CompetitionPlan, object, CheckpointStore]:
    from maskimpute_benchmark.runner import _derive_prezero_execution_authority

    context, _score, _calibration = _write_real_score_authority(repository, prepared)
    probability, policy = _derive_prezero_execution_authority(
        prepared,
        _entry(prepared),
        context,
        calibration_usage="development_holdout",
        repository=repository,
    )
    score_diagnostics = {
        "source": policy["score_source"],
        "score_artifact_sha256": policy["score_artifact_sha256"],
        "score_input_sha256": policy["score_input_sha256"],
        "score_config_sha256": policy["score_config_sha256"],
        "calibration_file_sha256": policy["calibration_file_sha256"],
        "calibration_payload_sha256": policy["calibration_payload_sha256"],
        "retained_calibrator": policy["calibration_algorithm"],
        "calibration_scope": policy["calibration_scope"],
        "equivalence_reason": policy["calibration_equivalence_reason"],
    }
    execution = _maskimpute_execution(
        prepared,
        probability,
        score_diagnostics,
    )
    attempt = _completed_maskimpute(
        prepared,
        execution=execution,
        calibration_file_sha256=context.retained_calibration_sha256,
    )
    plan = replace(
        _plan(prepared),
        input_hashes={
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "implementation_source_sha256": implementation_source_sha256(),
            "count_model_config_sha256": context.count_model_config_sha256,
            "count_score_manifest_sha256": context.count_score_manifest_sha256,
            "retained_calibration_sha256": context.retained_calibration_sha256,
        },
        execution_context=context,
    )
    return (
        plan,
        attempt,
        CheckpointStore(
            repository / "competition",
            authority_repository=repository,
        ),
    )


def _diagnostics_from_persisted_policy(
    policy: dict[str, object],
) -> dict[str, object]:
    return {
        "source": policy["score_source"],
        "score_artifact_sha256": policy["score_artifact_sha256"],
        "score_input_sha256": policy["score_input_sha256"],
        "score_config_sha256": policy["score_config_sha256"],
        "calibration_file_sha256": policy["calibration_file_sha256"],
        "calibration_payload_sha256": policy["calibration_payload_sha256"],
        "retained_calibrator": policy["calibration_algorithm"],
        "calibration_scope": policy["calibration_scope"],
        "equivalence_reason": policy["calibration_equivalence_reason"],
    }


def _rewrite_checkpoint(store: CheckpointStore, mutate) -> dict[str, object]:
    payload = json.loads(store.checkpoint_path.read_text(encoding="utf-8"))
    mutate(payload)
    body = {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    payload["checkpoint_sha256"] = canonical_sha256(body)
    store.checkpoint_path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return payload


def _leave_staging_temporary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    canonical_path: Path,
    temporary_prefix: str,
    phase: str,
    publish,
) -> Path:
    """Model process death around mkstemp/link without actually killing pytest."""

    original_unlink = Path.unlink
    original_link = os.link
    observed: list[Path] = []

    def preserve_temporary(path: Path, *args, **kwargs):
        if (
            path.parent == canonical_path.parent
            and path.name.startswith(temporary_prefix)
            and path.name.endswith(".tmp")
        ):
            observed.append(path)
            return None
        return original_unlink(path, *args, **kwargs)

    def interrupt_before_link(source, destination, *args, **kwargs):
        if Path(destination) == canonical_path:
            raise RuntimeError("simulated crash before immutable link")
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", preserve_temporary)
    if phase == "pre_link":
        monkeypatch.setattr(os, "link", interrupt_before_link)
        with pytest.raises(RuntimeError, match="before immutable link"):
            publish()
        monkeypatch.setattr(os, "link", original_link)
    elif phase == "post_link":
        publish()
    else:  # pragma: no cover - parametrization is fixed below
        raise AssertionError(phase)
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert len(observed) == 1
    temporary = observed[0]
    assert temporary.is_file()
    if phase == "pre_link":
        assert not canonical_path.exists()
        assert temporary.stat().st_nlink == 1
    else:
        assert canonical_path.is_file()
        assert temporary.stat().st_ino == canonical_path.stat().st_ino
        assert temporary.stat().st_nlink == 2
    return temporary


def _development_artifact_payload(
    attempt,
    artifact_suffix: str,
) -> tuple[str, bytes]:
    from maskimpute_benchmark.prezero_evidence import encode_prezero_evidence

    base = f"runs/{attempt.run.run_id}"
    if artifact_suffix == ".stdout":
        return f"{base}.stdout", attempt.stdout
    if artifact_suffix == ".stderr":
        return f"{base}.stderr", attempt.stderr
    if artifact_suffix == ".native-f64":
        assert attempt.native_output is not None
        return (
            f"{base}.native-f64",
            np.asarray(attempt.native_output, dtype="<f8").tobytes(order="C"),
        )
    if artifact_suffix == ".log2-cp10k-f64":
        assert attempt.evaluator_output is not None
        return (
            f"{base}.log2-cp10k-f64",
            np.asarray(attempt.evaluator_output, dtype="<f8").tobytes(order="C"),
        )
    if artifact_suffix == ".p-pre-zero-f64.zlib":
        _record, compressed = encode_prezero_evidence(attempt.p_pre_zero_evidence)
        assert compressed is not None
        return f"{base}.p-pre-zero-f64.zlib", compressed
    raise AssertionError(artifact_suffix)  # pragma: no cover - fixed parametrization


def _scientifically_equivalent_retry(attempt):
    retry_stdout = b"scientifically equivalent retry stdout\n"
    retry_stderr = b"scientifically equivalent retry stderr\n"
    return replace(
        attempt,
        run=replace(
            attempt.run,
            stdout_sha256=hashlib.sha256(retry_stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(retry_stderr).hexdigest(),
        ),
        stdout=retry_stdout,
        stderr=retry_stderr,
    )


def _assert_clean_development_retry(
    store: CheckpointStore,
    report,
    retry,
) -> None:
    assert report.status == "completed"
    assert not (store.output_dir / "transactions").exists()
    record = report.records[0]
    run = record["run"]
    assert (store.output_dir / run["stdout_path"]).read_bytes() == retry.stdout
    assert (store.output_dir / run["stderr_path"]).read_bytes() == retry.stderr
    expected_files = {
        "checkpoint.json",
        run["stdout_path"],
        run["stderr_path"],
        run["native_output_path"],
        run["evaluator_output_path"],
        record["p_pre_zero_evidence"]["storage"]["path"],
    }
    observed_files = {
        path.relative_to(store.output_dir).as_posix()
        for path in store.output_dir.rglob("*")
        if path.is_file()
    }
    assert observed_files == expected_files


def _rebind_score_payload(
    payload: dict[str, object], *, score_raw: bytes | None = None
) -> None:
    evidence = payload["records"][0]["p_pre_zero_evidence"]
    if score_raw is not None:
        evidence["policy_sha256"] = canonical_sha256(evidence["policy"])
        semantic = hashlib.sha256()
        semantic.update(b"maskimpute-realized-p-pre-zero-v1\0")
        semantic.update(
            json.dumps(
                {
                    "identity": evidence["identity"],
                    "shape": evidence["matrix"]["shape"],
                    "dtype": evidence["matrix"]["dtype"],
                    "policy_sha256": evidence["policy_sha256"],
                },
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        semantic.update(score_raw)
        evidence["matrix"]["semantic_sha256"] = semantic.hexdigest()
    body = {
        key: value
        for key, value in evidence.items()
        if key not in {"evidence_sha256", "storage"}
    }
    evidence["evidence_sha256"] = canonical_sha256(body)


def _validate_stored_completed_score(
    prepared: PreparedDataset,
    record: dict[str, object],
    compressed: bytes,
    *,
    expected_probability: np.ndarray | None = None,
    expected_policy: dict[str, object] | None = None,
    bind_exact_authority: bool = True,
) -> dict[str, object]:
    from maskimpute_benchmark.prezero_evidence import (
        validate_stored_prezero_evidence,
    )

    truth_layer = prepared.evaluator_dataset.uns["primary_truth_layer"]
    if bind_exact_authority:
        if expected_probability is None:
            expected_probability = np.frombuffer(
                zlib.decompress(compressed), dtype="<f8"
            ).reshape(prepared.method_input.shape)
        if expected_policy is None:
            expected_policy = dict(record["policy"])
    return validate_stored_prezero_evidence(
        record,
        expected_identity=record["identity"],
        run_status="completed",
        run_reason=None,
        observed_zero_count=6,
        expected_shape=prepared.method_input.shape,
        requires_count_score=True,
        requires_calibration=True,
        expected_calibration_file_sha256="7" * 64,
        compressed=compressed,
        observed=np.asarray(prepared.evaluator_dataset.X, dtype=np.float64),
        truth=np.asarray(
            prepared.evaluator_dataset.layers[truth_layer], dtype=np.float64
        ),
        truth_kind="exact_pre_capture",
        expected_score_input_sha256=_score_input_sha256(prepared),
        expected_score_config_sha256="6" * 64,
        expected_matrix_present=True,
        expected_probability=expected_probability,
        expected_policy=expected_policy,
    )


def _encoded_completed_score(
    prepared: PreparedDataset,
) -> tuple[dict[str, object], bytes]:
    from maskimpute_benchmark.prezero_evidence import encode_prezero_evidence

    record, compressed = encode_prezero_evidence(
        _completed_maskimpute(prepared).p_pre_zero_evidence
    )
    assert compressed is not None
    record["storage"]["path"] = "runs/test.p-pre-zero-f64.zlib"
    return record, compressed


def test_stored_score_validation_rejects_rehashed_metric_drift() -> None:
    from maskimpute_benchmark.prezero_evidence import PreZeroEvidenceError

    prepared = _prepared()
    record, compressed = _encoded_completed_score(prepared)
    record["overall"]["metrics"]["brier"]["value"] = 0.123456
    _rebind_score_payload(
        {"records": [{"p_pre_zero_evidence": record}]},
    )

    with pytest.raises(PreZeroEvidenceError, match="report differs"):
        _validate_stored_completed_score(prepared, record, compressed)


def test_stored_score_validation_rejects_rehashed_policy_drift() -> None:
    from maskimpute_benchmark.prezero_evidence import PreZeroEvidenceError

    prepared = _prepared()
    record, compressed = _encoded_completed_score(prepared)
    record["policy"]["score_input_sha256"] = "f" * 64
    _rebind_score_payload(
        {"records": [{"p_pre_zero_evidence": record}]},
        score_raw=zlib.decompress(compressed),
    )

    with pytest.raises(PreZeroEvidenceError, match="score input"):
        _validate_stored_completed_score(prepared, record, compressed)


def test_stored_score_validation_rejects_coordinated_matrix_report_replacement() -> (
    None
):
    from maskimpute_benchmark.prezero_evidence import (
        PreZeroEvidenceError,
        _score_report,
    )

    prepared = _prepared()
    record, compressed = _encoded_completed_score(prepared)
    expected_probability = np.frombuffer(
        zlib.decompress(compressed), dtype="<f8"
    ).reshape(prepared.method_input.shape)
    expected_policy = dict(record["policy"])
    replacement = np.array(
        [[0.05, 0.95, 0.0, 0.85], [0.15, 0.0, 0.75, 0.25]],
        dtype="<f8",
    )
    replacement_raw = replacement.tobytes(order="C")
    replacement_compressed = zlib.compress(replacement_raw, level=6)
    record["matrix"]["content_sha256"] = hashlib.sha256(replacement_raw).hexdigest()
    record["storage"].update(
        {
            "compressed_sha256": hashlib.sha256(replacement_compressed).hexdigest(),
            "compressed_nbytes": len(replacement_compressed),
            "uncompressed_sha256": hashlib.sha256(replacement_raw).hexdigest(),
            "uncompressed_nbytes": len(replacement_raw),
        }
    )
    observed = np.asarray(prepared.evaluator_dataset.X, dtype=np.float64)
    truth = np.asarray(
        prepared.evaluator_dataset.layers["pre_capture_counts"], dtype=np.float64
    )
    record["overall"], record["strata"] = _score_report(
        replacement,
        observed,
        truth,
        truth_kind="exact_pre_capture",
    )
    _rebind_score_payload(
        {"records": [{"p_pre_zero_evidence": record}]},
        score_raw=replacement_raw,
    )

    with pytest.raises(PreZeroEvidenceError, match="matrix differs"):
        _validate_stored_completed_score(
            prepared,
            record,
            replacement_compressed,
            expected_probability=expected_probability,
            expected_policy=expected_policy,
        )


def test_stored_score_validation_requires_exact_execution_authority() -> None:
    from maskimpute_benchmark.prezero_evidence import PreZeroEvidenceError

    prepared = _prepared()
    record, compressed = _encoded_completed_score(prepared)

    with pytest.raises(PreZeroEvidenceError, match="exact execution authority"):
        _validate_stored_completed_score(
            prepared,
            record,
            compressed,
            bind_exact_authority=False,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("score_artifact_sha256", "d" * 64),
        ("calibration_payload_sha256", "e" * 64),
        ("calibration_algorithm", "isotonic"),
        ("calibration_scope", "forged_scope"),
        ("calibration_equivalence_reason", "forged_equivalence"),
    ),
)
def test_stored_score_validation_rejects_rehashed_valid_policy_drift(
    field: str,
    replacement: str,
) -> None:
    from maskimpute_benchmark.prezero_evidence import PreZeroEvidenceError

    prepared = _prepared()
    record, compressed = _encoded_completed_score(prepared)
    expected_policy = dict(record["policy"])
    record["policy"][field] = replacement
    raw = zlib.decompress(compressed)
    _rebind_score_payload(
        {"records": [{"p_pre_zero_evidence": record}]},
        score_raw=raw,
    )

    with pytest.raises(PreZeroEvidenceError, match="policy differs"):
        _validate_stored_completed_score(
            prepared,
            record,
            compressed,
            expected_probability=np.frombuffer(raw, dtype="<f8").reshape(
                prepared.method_input.shape
            ),
            expected_policy=expected_policy,
        )


def test_development_score_authority_uses_exact_ready_artifacts_and_digest_domains(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.runner import _derive_prezero_execution_authority

    prepared = _prepared()
    context, score, calibration = _write_real_score_authority(tmp_path, prepared)

    probability, policy = _derive_prezero_execution_authority(
        prepared,
        _entry(prepared),
        context,
        calibration_usage="development_holdout",
        repository=tmp_path,
    )

    np.testing.assert_array_equal(probability, score.p_pre_zero)
    assert policy["schema_version"] == 2
    assert policy["score_artifact_sha256"] == score.score_sha256
    assert policy["score_input_sha256"] == score.input_sha256
    assert policy["score_config_sha256"] == score.config_sha256
    assert policy["calibration_file_sha256"] == (context.retained_calibration_sha256)
    assert (
        policy["calibration_payload_sha256"]
        == (calibration.to_dict()["payload_sha256"])
    )
    assert policy["calibration_file_sha256"] != policy["calibration_payload_sha256"]
    assert policy["calibration_algorithm"] == "identity"
    assert policy["calibration_scope"] == "leave_one_biological_draw_out"
    assert policy["calibration_equivalence_reason"] == (
        "retained_identity_calibrator_equals_direct_score"
    )


def test_adapter_exposes_realized_p_pre_zero_as_a_defensive_matrix() -> None:
    prepared = _prepared()
    execution = _maskimpute_execution(prepared)

    first = execution.realized_p_pre_zero
    expected = first.copy()
    with pytest.raises(ValueError, match="read-only"):
        first[0, 0] = 0.0

    np.testing.assert_array_equal(execution.realized_p_pre_zero, expected)


def test_evaluator_binds_realized_matrix_policy_and_machine_readable_metrics() -> None:
    prepared = _prepared()
    attempt = _completed_maskimpute(prepared)
    evidence = attempt.p_pre_zero_evidence
    record = evidence.to_record()

    assert evidence.status == "completed"
    assert evidence.reason is None
    np.testing.assert_array_equal(
        evidence.matrix, _maskimpute_execution(prepared).realized_p_pre_zero
    )
    assert record["identity"] == {
        "run_id": "run-maskimpute-symsim",
        "method_id": "maskimpute",
        "dataset_id": "dataset-symsim",
        "source_dataset_sha256": SHA_A,
        "mechanism": "symsim",
        "biological_id": "draw-01",
        "technical_view": "moderate",
        "model_seed": 42,
        "configuration_id": "v27-reference",
        "configuration_sha256": "3" * 64,
        "method_input_sha256": attempt.run.method_input_sha256,
        "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
    }
    assert record["matrix"]["shape"] == [2, 4]
    assert record["matrix"]["dtype"] == "<f8"
    assert len(record["matrix"]["semantic_sha256"]) == 64
    assert record["policy"]["schema_version"] == 2
    assert record["policy"]["score_source"] == "retained_calibrator"
    assert record["policy"]["calibration_file_sha256"] == "7" * 64
    assert record["policy"]["calibration_payload_sha256"] == "8" * 64
    assert (
        record["policy"]["calibration_file_sha256"]
        != record["policy"]["calibration_payload_sha256"]
    )
    assert record["policy_sha256"]
    assert record["overall"]["metrics"] == {
        name: record["overall"]["metrics"][name]
        for name in (
            "auroc",
            "auprc",
            "brier",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
            "ece",
        )
    }
    assert record["overall"]["reliability_bins"]
    assert len(record["strata"]["library_size_quartiles"]) == 4
    assert len(record["strata"]["truth_expression_bins"]) == 4
    assert record["evidence_sha256"]


def test_non_maskimpute_and_noncompleted_attempts_have_explicit_score_rows() -> None:
    prepared = _prepared()
    observed_execution = run_observed(
        load_method_registry(METHODS).by_id("observed"), prepared.method_input
    )
    observed = evaluate_adapter_outcome(
        _entry(prepared, "observed"),
        prepared,
        AdapterOutcome.completed(
            observed_execution,
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=0,
        ),
    ).p_pre_zero_evidence
    timeout = evaluate_adapter_outcome(
        _entry(prepared), prepared, AdapterOutcome.timeout()
    ).p_pre_zero_evidence

    assert observed.status == "not_applicable"
    assert observed.reason == "method_does_not_emit_p_pre_zero"
    assert observed.matrix is None
    assert observed.to_record()["overall"]["metrics"]["brier"] == {
        "value": None,
        "n": 6,
        "status": "not_applicable",
        "reason": "method_does_not_emit_p_pre_zero",
    }
    assert timeout.status == "timeout"
    assert timeout.reason == "timeout"
    assert timeout.matrix is None
    assert timeout.to_record()["overall"]["metrics"]["auroc"]["status"] == "timeout"


def test_unavailable_library_strata_use_the_canonical_tie_aware_partition() -> None:
    from maskimpute_benchmark.metrics import stratified_zero_score_metrics
    from maskimpute_benchmark.prezero_evidence import evaluate_prezero_evidence

    library_groups = np.array([1.0] + [2.0] * 8 + [3.0, 4.0, 5.0])
    observed = np.column_stack((library_groups, np.zeros(library_groups.size)))
    expected = stratified_zero_score_metrics(
        np.zeros_like(observed),
        observed,
        None,
        truth_kind="orthogonal_only",
    )["library_size_quartiles"]

    record = evaluate_prezero_evidence(
        identity={
            "run_id": "run-observed-orthogonal",
            "method_id": "observed",
            "dataset_id": "dataset-orthogonal",
            "source_dataset_sha256": "1" * 64,
            "mechanism": "orthogonal",
            "biological_id": "draw-01",
            "technical_view": "replicate-a",
            "model_seed": None,
            "configuration_id": "registry-default",
            "configuration_sha256": "2" * 64,
            "method_input_sha256": "3" * 64,
            "retained_cell_ids_sha256": "4" * 64,
        },
        method_shape=observed.shape,
        method_id="observed",
        execution=None,
        run_status="completed",
        run_reason=None,
        observed=observed,
        truth=None,
        truth_kind="orthogonal_only",
    ).to_record()

    assert [
        (item["label"], item["lower"], item["upper"], item["n"])
        for item in record["strata"]["library_size_quartiles"]
    ] == [(item["label"], item["lower"], item["upper"], item["n"]) for item in expected]


def test_all_four_mechanisms_have_exact_or_reason_coded_score_evidence() -> None:
    expected = {
        "symsim": None,
        "sergio": "undefined_for_continuous_truth",
        "sparsim": "undefined_for_continuous_truth",
        "semisynthetic": "proxy_truth_not_exact",
    }

    for mechanism, reason in expected.items():
        record = _completed_maskimpute(
            _prepared(mechanism)
        ).p_pre_zero_evidence.to_record()
        metrics = record["overall"]["metrics"]
        if reason is None:
            assert metrics["brier"]["value"] is not None
            assert metrics["brier"]["reason"] is None
        else:
            assert all(value["value"] is None for value in metrics.values())
            assert {value["reason"] for value in metrics.values()} == {reason}
            assert record["overall"]["reliability_bins"] == []
        for stratum in (
            *record["strata"]["library_size_quartiles"],
            *record["strata"]["truth_expression_bins"],
        ):
            if reason is not None:
                assert {value["reason"] for value in stratum["metrics"].values()} == {
                    reason
                }


def test_development_checkpoint_compresses_and_resumes_realized_score_evidence(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan, attempt, _store = _real_development_checkpoint_case(tmp_path, prepared)
    first = CheckpointStore(tmp_path / "first", authority_repository=tmp_path)
    second = CheckpointStore(tmp_path / "second", authority_repository=tmp_path)

    first_report = first.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )
    second_report = second.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )
    evidence = first_report.records[0]["p_pre_zero_evidence"]
    storage = evidence["storage"]
    first_path = first.output_dir / storage["path"]
    second_path = (
        second.output_dir
        / second_report.records[0]["p_pre_zero_evidence"]["storage"]["path"]
    )

    assert storage["encoding"] == "zlib_raw_f64_v1"
    assert storage["compression_level"] == 6
    assert (
        storage["compressed_sha256"]
        == hashlib.sha256(first_path.read_bytes()).hexdigest()
    )
    expected = attempt.p_pre_zero_evidence.matrix.astype("<f8").tobytes(order="C")
    assert zlib.decompress(first_path.read_bytes()) == expected
    assert storage["uncompressed_sha256"] == hashlib.sha256(expected).hexdigest()
    assert storage["uncompressed_nbytes"] == len(expected)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert (
        CheckpointStore(first.output_dir, authority_repository=tmp_path).load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=_prepared_datasets(prepared),
        )
        == first_report
    )


def test_development_checkpoint_rejects_coordinated_score_replacement(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.prezero_evidence import _score_report

    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    prepared_datasets = _prepared_datasets(prepared)
    store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )

    def mutate(payload: dict[str, object]) -> None:
        evidence = payload["records"][0]["p_pre_zero_evidence"]
        replacement = np.where(
            prepared.method_input.counts == 0.0,
            0.125,
            0.0,
        ).astype("<f8")
        raw = replacement.tobytes(order="C")
        compressed = zlib.compress(raw, level=6)
        path = store.output_dir / evidence["storage"]["path"]
        path.write_bytes(compressed)
        evidence["storage"].update(
            {
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "compressed_nbytes": len(compressed),
                "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                "uncompressed_nbytes": len(raw),
            }
        )
        evidence["matrix"]["content_sha256"] = hashlib.sha256(raw).hexdigest()
        evidence["overall"], evidence["strata"] = _score_report(
            replacement,
            np.asarray(prepared.evaluator_dataset.X, dtype=np.float64),
            np.asarray(
                prepared.evaluator_dataset.layers["pre_capture_counts"],
                dtype=np.float64,
            ),
            truth_kind="exact_pre_capture",
        )
        _rebind_score_payload(payload, score_raw=raw)

    _rewrite_checkpoint(store, mutate)
    with pytest.raises(RunnerContractError, match="matrix differs"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )


def test_score_authority_cache_returns_detached_matrix_and_policy(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    prepared_datasets = _prepared_datasets(prepared)
    store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )
    assert plan.execution_context is not None

    probability, policy = store._expected_prezero_authority(
        plan.entries[0],
        prepared,
        plan.execution_context,
        calibration_usage="development_holdout",
        expected_matrix_present=True,
    )
    assert probability is not None
    assert policy is not None
    probability.setflags(write=True)
    probability[0, 0] = 0.125
    policy["nested_mutation"] = {"values": []}
    policy["nested_mutation"]["values"].append("forged")

    repeated_probability, repeated_policy = store._expected_prezero_authority(
        plan.entries[0],
        prepared,
        plan.execution_context,
        calibration_usage="development_holdout",
        expected_matrix_present=True,
    )
    np.testing.assert_array_equal(
        repeated_probability,
        attempt.p_pre_zero_evidence.matrix,
    )
    assert repeated_probability.flags.writeable is False
    assert repeated_policy == attempt.p_pre_zero_evidence.to_record()["policy"]

    def mutate(payload: dict[str, object]) -> None:
        evidence = payload["records"][0]["p_pre_zero_evidence"]
        evidence["policy"]["calibration_scope"] = "forged_scope"
        raw = zlib.decompress(
            (store.output_dir / evidence["storage"]["path"]).read_bytes()
        )
        _rebind_score_payload(payload, score_raw=raw)

    _rewrite_checkpoint(store, mutate)
    with pytest.raises(RunnerContractError, match="policy differs"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )


def test_development_append_validates_before_publishing_and_allows_retry(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan, valid_attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    policy = valid_attempt.p_pre_zero_evidence.to_record()["policy"]
    assert isinstance(policy, dict)
    invalid_probability = np.where(
        prepared.method_input.counts == 0.0,
        0.125,
        0.0,
    ).astype("<f8")
    invalid_attempt = _completed_maskimpute(
        prepared,
        execution=_maskimpute_execution(
            prepared,
            invalid_probability,
            _diagnostics_from_persisted_policy(policy),
        ),
        calibration_file_sha256=plan.input_hashes["retained_calibration_sha256"],
    )
    prepared_datasets = _prepared_datasets(prepared)

    with pytest.raises(RunnerContractError, match="matrix differs"):
        store.append(
            plan,
            None,
            invalid_attempt,
            _budget_for_attempt(plan, invalid_attempt),
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )
    assert not store.output_dir.exists()

    report = store.append(
        plan,
        None,
        valid_attempt,
        _budget_for_attempt(plan, valid_attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )
    assert report.status == "completed"
    assert len(report.records) == 1


@pytest.mark.parametrize("phase", ("pre_link", "post_link"))
def test_development_restart_cleans_interrupted_intent_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    canonical = store.output_dir / "transactions/00000001.json"
    temporary = _leave_staging_temporary(
        monkeypatch,
        canonical_path=canonical,
        temporary_prefix=".00000001.json.",
        phase=phase,
        publish=lambda: store._publish_transaction_intent(
            plan,
            0,
            plan.entries[0],
            attempt,
        ),
    )

    restarted = CheckpointStore(store.output_dir, authority_repository=tmp_path)
    retry = _scientifically_equivalent_retry(attempt)
    report = restarted.append(
        plan,
        None,
        retry,
        _budget_for_attempt(plan, retry),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )

    assert not temporary.exists()
    _assert_clean_development_retry(restarted, report, retry)


@pytest.mark.parametrize("phase", ("pre_link", "post_link"))
@pytest.mark.parametrize(
    "artifact_suffix",
    (
        ".stdout",
        ".stderr",
        ".native-f64",
        ".log2-cp10k-f64",
        ".p-pre-zero-f64.zlib",
    ),
)
def test_development_restart_cleans_interrupted_run_artifact_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    artifact_suffix: str,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    store._publish_transaction_intent(plan, 0, plan.entries[0], attempt)
    relative, data = _development_artifact_payload(attempt, artifact_suffix)
    canonical = store.output_dir / relative
    temporary = _leave_staging_temporary(
        monkeypatch,
        canonical_path=canonical,
        temporary_prefix=f".{canonical.name}.",
        phase=phase,
        publish=lambda: store._publish_immutable(relative, data),
    )

    restarted = CheckpointStore(store.output_dir, authority_repository=tmp_path)
    retry = _scientifically_equivalent_retry(attempt)
    report = restarted.append(
        plan,
        None,
        retry,
        _budget_for_attempt(plan, retry),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )

    assert not temporary.exists()
    _assert_clean_development_retry(restarted, report, retry)


def test_development_restart_cleans_interrupted_checkpoint_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    original_replace = os.replace
    original_unlink = Path.unlink
    observed: list[Path] = []

    def interrupt_before_replace(source, destination, *args, **kwargs):
        if Path(destination) == store.checkpoint_path:
            raise RuntimeError("simulated crash before checkpoint replace")
        return original_replace(source, destination, *args, **kwargs)

    def preserve_checkpoint_temporary(path: Path, *args, **kwargs):
        if (
            path.parent == store.output_dir
            and path.name.startswith(".checkpoint.")
            and path.name.endswith(".tmp")
        ):
            observed.append(path)
            return None
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "replace", interrupt_before_replace)
    monkeypatch.setattr(Path, "unlink", preserve_checkpoint_temporary)
    with pytest.raises(RuntimeError, match="before checkpoint replace"):
        store._publish_checkpoint({"interrupted": True})
    monkeypatch.setattr(os, "replace", original_replace)
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert len(observed) == 1
    temporary = observed[0]
    assert temporary.is_file()
    assert temporary.stat().st_nlink == 1

    restarted = CheckpointStore(store.output_dir, authority_repository=tmp_path)
    retry = _scientifically_equivalent_retry(attempt)
    report = restarted.append(
        plan,
        None,
        retry,
        _budget_for_attempt(plan, retry),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )

    assert not temporary.exists()
    _assert_clean_development_retry(restarted, report, retry)


@pytest.mark.parametrize(
    "artifact_suffix",
    (
        ".stdout",
        ".stderr",
        ".native-f64",
        ".log2-cp10k-f64",
        ".p-pre-zero-f64.zlib",
    ),
)
def test_development_restart_recovers_every_interrupted_artifact_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_suffix: str,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    assert attempt.native_output is not None
    assert attempt.evaluator_output is not None
    assert attempt.p_pre_zero_evidence.raw_matrix_bytes is not None
    prepared_datasets = _prepared_datasets(prepared)
    original_publish = store._publish_immutable

    def interrupt_after_closed_artifact(relative: str, data: bytes):
        receipt = original_publish(relative, data)
        if relative.endswith(artifact_suffix):
            raise RuntimeError(f"interrupted after {artifact_suffix}")
        return receipt

    monkeypatch.setattr(store, "_publish_immutable", interrupt_after_closed_artifact)
    with pytest.raises(RuntimeError, match="interrupted after"):
        store.append(
            plan,
            None,
            attempt,
            _budget_for_attempt(plan, attempt),
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )
    assert not store.checkpoint_path.exists()
    assert (store.output_dir / "transactions/00000001.json").is_file()

    retry = _scientifically_equivalent_retry(attempt)
    restarted = CheckpointStore(
        store.output_dir,
        authority_repository=tmp_path,
    )
    report = restarted.append(
        plan,
        None,
        retry,
        _budget_for_attempt(plan, retry),
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )

    _assert_clean_development_retry(restarted, report, retry)


def test_development_restart_retains_committed_artifacts_and_closes_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    prepared_datasets = _prepared_datasets(prepared)
    original_publish_checkpoint = store._publish_checkpoint

    def interrupt_after_checkpoint(payload):
        original_publish_checkpoint(payload)
        raise RuntimeError("interrupted after checkpoint")

    monkeypatch.setattr(store, "_publish_checkpoint", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="interrupted after checkpoint"):
        store.append(
            plan,
            None,
            attempt,
            _budget_for_attempt(plan, attempt),
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )
    assert store.checkpoint_path.is_file()
    assert (store.output_dir / "transactions/00000001.json").is_file()
    before = {
        path.relative_to(store.output_dir).as_posix(): path.read_bytes()
        for path in (store.output_dir / "runs").iterdir()
        if path.is_file()
    }

    restarted = CheckpointStore(store.output_dir, authority_repository=tmp_path)
    report = restarted.load(
        plan,
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )

    assert report.status == "completed"
    assert not (store.output_dir / "transactions").exists()
    after = {
        path.relative_to(store.output_dir).as_posix(): path.read_bytes()
        for path in (store.output_dir / "runs").iterdir()
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("replacement", ("hardlink", "symlink"))
def test_development_recovery_refuses_nonunique_or_linked_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    prepared_datasets = _prepared_datasets(prepared)
    original_publish = store._publish_immutable

    def interrupt_after_stdout(relative: str, data: bytes):
        receipt = original_publish(relative, data)
        if relative.endswith(".stdout"):
            raise RuntimeError("interrupted after stdout")
        return receipt

    monkeypatch.setattr(store, "_publish_immutable", interrupt_after_stdout)
    with pytest.raises(RuntimeError, match="interrupted after stdout"):
        store.append(
            plan,
            None,
            attempt,
            _budget_for_attempt(plan, attempt),
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )
    artifact = store.output_dir / f"runs/{attempt.run.run_id}.stdout"
    external = tmp_path / f"external-{replacement}"
    if replacement == "hardlink":
        external.hardlink_to(artifact)
    else:
        artifact.unlink()
        external.write_bytes(b"must survive recovery")
        artifact.symlink_to(external)

    restarted = CheckpointStore(store.output_dir, authority_repository=tmp_path)
    with pytest.raises(RunnerContractError, match="unique|symlink|unsafe"):
        restarted.append(
            plan,
            None,
            attempt,
            _budget_for_attempt(plan, attempt),
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )

    assert external.exists()
    assert (store.output_dir / "transactions/00000001.json").is_file()


@pytest.mark.parametrize(
    "relative",
    (
        ".checkpoint.abcdefgh.tmp",
        "transactions/.00000001.json.abcdefgh.tmp",
        "runs/.run-maskimpute-symsim.stdout.abcdefgh.tmp",
    ),
)
def test_development_staging_recovery_rejects_unrelated_hardlink(
    tmp_path: Path,
    relative: str,
) -> None:
    prepared = _prepared()
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    temporary = store.output_dir / relative
    temporary.parent.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external-hardlink"
    external.write_bytes(b"must survive")
    temporary.hardlink_to(external)

    with pytest.raises(RunnerContractError, match="staging|hardlink|sibling"):
        store._recover_interrupted_transactions(plan)

    assert temporary.exists()
    assert external.read_bytes() == b"must survive"
    assert external.stat().st_nlink == 2


def test_development_staging_recovery_rejects_symlink(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    runs = store.output_dir / "runs"
    runs.mkdir(parents=True)
    canonical = runs / f"{plan.entries[0].run_id}.stdout"
    temporary = runs / f".{canonical.name}.abcdefgh.tmp"
    external = tmp_path / "external-symlink"
    external.write_bytes(b"must survive")
    temporary.symlink_to(external)

    with pytest.raises(RunnerContractError, match="staging|regular|symlink"):
        store._recover_interrupted_transactions(plan)

    assert temporary.is_symlink()
    assert external.read_bytes() == b"must survive"


@pytest.mark.parametrize(
    ("relative", "payload"),
    (
        ("unexpected", b"unexpected root file"),
        ("runs/unexpected", b"unexpected run file"),
        ("runs/.malformed.short.tmp", b"malformed staging file"),
        ("transactions/.00000001.json.short.tmp", b"malformed intent staging"),
    ),
)
def test_development_staging_recovery_rejects_unexpected_files(
    tmp_path: Path,
    relative: str,
    payload: bytes,
) -> None:
    prepared = _prepared()
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    unexpected = store.output_dir / relative
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_bytes(payload)

    with pytest.raises(RunnerContractError, match="unexpected|transaction name"):
        store._recover_interrupted_transactions(plan)

    assert unexpected.read_bytes() == payload


def test_development_staging_recovery_rejects_symlinked_owned_directory(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    store.output_dir.mkdir()
    external = tmp_path / "external-runs"
    external.mkdir()
    (store.output_dir / "runs").symlink_to(external, target_is_directory=True)

    with pytest.raises(RunnerContractError, match="directory|symlink|canonical"):
        store._recover_interrupted_transactions(plan)

    assert (store.output_dir / "runs").is_symlink()


@pytest.mark.parametrize("tamper", ("matrix", "policy", "report"))
def test_development_conversion_terminal_score_remains_exactly_authorized(
    tmp_path: Path,
    tamper: str,
) -> None:
    from maskimpute_benchmark.prezero_evidence import _score_report

    prepared = _prepared()
    plan, valid_attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    policy = valid_attempt.p_pre_zero_evidence.to_record()["policy"]
    assert isinstance(policy, dict)

    def reject_conversion(_method_input, _execution):
        raise ValueError("deliberate evaluator conversion rejection")

    conversion_attempt = _completed_maskimpute(
        prepared,
        execution=_maskimpute_execution(
            prepared,
            valid_attempt.p_pre_zero_evidence.matrix,
            _diagnostics_from_persisted_policy(policy),
        ),
        calibration_file_sha256=plan.input_hashes["retained_calibration_sha256"],
        output_converter=reject_conversion,
    )
    assert conversion_attempt.run.status == "unavailable"
    assert conversion_attempt.p_pre_zero_evidence.raw_matrix_bytes is not None
    prepared_datasets = _prepared_datasets(prepared)
    report = store.append(
        plan,
        None,
        conversion_attempt,
        _budget_for_attempt(plan, conversion_attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )
    assert report.records[0]["run"]["status"] == "unavailable"
    assert report.records[0]["p_pre_zero_evidence"]["status"] == "unavailable"

    def mutate(payload: dict[str, object]) -> None:
        evidence = payload["records"][0]["p_pre_zero_evidence"]
        path = store.output_dir / evidence["storage"]["path"]
        raw = zlib.decompress(path.read_bytes())
        if tamper == "matrix":
            replacement = np.where(
                prepared.method_input.counts == 0.0,
                0.125,
                0.0,
            ).astype("<f8")
            raw = replacement.tobytes(order="C")
            compressed = zlib.compress(raw, level=6)
            path.write_bytes(compressed)
            evidence["storage"].update(
                {
                    "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                    "compressed_nbytes": len(compressed),
                    "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                    "uncompressed_nbytes": len(raw),
                }
            )
            evidence["matrix"]["content_sha256"] = hashlib.sha256(raw).hexdigest()
            evidence["overall"], evidence["strata"] = _score_report(
                replacement,
                np.asarray(prepared.evaluator_dataset.X, dtype=np.float64),
                np.asarray(
                    prepared.evaluator_dataset.layers["pre_capture_counts"],
                    dtype=np.float64,
                ),
                truth_kind="exact_pre_capture",
                unavailable_status="unavailable",
                unavailable_reason=evidence["reason"],
            )
        elif tamper == "policy":
            evidence["policy"]["calibration_scope"] = "forged_scope"
        elif tamper == "report":
            evidence["overall"]["metrics"]["brier"]["reason"] = "forged_reason"
        else:  # pragma: no cover - parametrization is fixed above
            raise AssertionError(tamper)
        _rebind_score_payload(payload, score_raw=raw)

    _rewrite_checkpoint(store, mutate)
    with pytest.raises(RunnerContractError, match="p_pre_zero|matrix|policy|report"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )


def test_development_conversion_terminal_score_cannot_be_coordinately_removed(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.prezero_evidence import _score_report

    prepared = _prepared()
    plan, valid_attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    policy = valid_attempt.p_pre_zero_evidence.to_record()["policy"]
    assert isinstance(policy, dict)

    def reject_conversion(_method_input, _execution):
        raise ValueError("deliberate evaluator conversion rejection")

    conversion_attempt = _completed_maskimpute(
        prepared,
        execution=_maskimpute_execution(
            prepared,
            valid_attempt.p_pre_zero_evidence.matrix,
            _diagnostics_from_persisted_policy(policy),
        ),
        calibration_file_sha256=plan.input_hashes["retained_calibration_sha256"],
        output_converter=reject_conversion,
    )
    prepared_datasets = _prepared_datasets(prepared)
    report = store.append(
        plan,
        None,
        conversion_attempt,
        _budget_for_attempt(plan, conversion_attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )
    stored_evidence = report.records[0]["p_pre_zero_evidence"]
    (store.output_dir / stored_evidence["storage"]["path"]).unlink()

    def remove_score(payload: dict[str, object]) -> None:
        evidence = payload["records"][0]["p_pre_zero_evidence"]
        evidence["matrix"] = {
            "shape": None,
            "dtype": None,
            "content_sha256": None,
            "semantic_sha256": None,
        }
        evidence["policy"] = None
        evidence["policy_sha256"] = None
        evidence["storage"] = {
            "encoding": None,
            "compression_level": None,
            "path": None,
            "compressed_sha256": None,
            "compressed_nbytes": None,
            "uncompressed_sha256": None,
            "uncompressed_nbytes": None,
        }
        evidence["overall"], evidence["strata"] = _score_report(
            None,
            np.asarray(prepared.evaluator_dataset.X, dtype=np.float64),
            np.asarray(
                prepared.evaluator_dataset.layers["pre_capture_counts"],
                dtype=np.float64,
            ),
            truth_kind="exact_pre_capture",
            unavailable_status="unavailable",
            unavailable_reason=evidence["reason"],
        )
        _rebind_score_payload(payload)

    _rewrite_checkpoint(store, remove_score)
    with pytest.raises(RunnerContractError, match="p_pre_zero|matrix|authority"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )


def test_development_checkpoint_rejects_rehashed_metric_drift(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    prepared_datasets = {prepared.binding.dataset_id: prepared}
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=prepared_datasets,
    )

    def mutate(payload: dict[str, object]) -> None:
        evidence = payload["records"][0]["p_pre_zero_evidence"]
        evidence["overall"]["metrics"]["brier"]["value"] = 0.123456
        _rebind_score_payload(payload)

    _rewrite_checkpoint(store, mutate)
    with pytest.raises(RunnerContractError, match="report differs"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=prepared_datasets,
        )


def test_development_checkpoint_rejects_score_tamper_and_partial_receipts(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    report = store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )
    evidence = report.records[0]["p_pre_zero_evidence"]
    path = store.output_dir / evidence["storage"]["path"]
    original = path.read_bytes()

    path.write_bytes(original + b"tamper")
    with pytest.raises(RunnerContractError, match="p_pre_zero|score.*checksum"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=_prepared_datasets(prepared),
        )

    path.write_bytes(original)
    _rewrite_checkpoint(
        store,
        lambda payload: payload["records"][0]["p_pre_zero_evidence"]["storage"].update(
            {"compression_level": None}
        ),
    )
    with pytest.raises(RunnerContractError, match="p_pre_zero|score.*partial"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=_prepared_datasets(prepared),
        )


@pytest.mark.parametrize(
    "case",
    (
        "library_label",
        "truth_bounds",
        "library_denominator",
        "truth_denominator",
        "bounded_metric",
        "negative_log_loss",
        "reliability_probability",
        "reliability_count",
        "reliability_interval",
        "too_many_reliability_bins",
        "policy_extra_field",
        "policy_semantics",
        "policy_artifact_checksum",
        "policy_empty_algorithm",
    ),
)
def test_development_checkpoint_rejects_semantically_invalid_score_payload(
    tmp_path: Path, case: str
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )

    def mutate(payload: dict[str, object]) -> None:
        evidence = payload["records"][0]["p_pre_zero_evidence"]
        library = evidence["strata"]["library_size_quartiles"]
        truth = evidence["strata"]["truth_expression_bins"]
        overall = evidence["overall"]
        if case == "library_label":
            library[0]["label"] = "Q9"
        elif case == "truth_bounds":
            truth[0]["lower"] = 0.5
        elif case in {"library_denominator", "truth_denominator"}:
            record = library[0] if case == "library_denominator" else truth[0]
            record["n"] += 1
            for metric in record["metrics"].values():
                metric["n"] = record["n"]
        elif case == "bounded_metric":
            overall["metrics"]["brier"]["value"] = 1.25
        elif case == "negative_log_loss":
            overall["metrics"]["log_loss"]["value"] = -0.1
        elif case == "reliability_probability":
            overall["reliability_bins"][0]["mean_prediction"] = 1.1
        elif case == "reliability_count":
            overall["reliability_bins"][0]["n"] = 0
        elif case == "reliability_interval":
            overall["reliability_bins"][0]["wilson_lower"] = 0.9
            overall["reliability_bins"][0]["wilson_upper"] = 0.1
        elif case == "too_many_reliability_bins":
            template = dict(overall["reliability_bins"][0])
            overall["reliability_bins"] = [
                {**template, "bin": index} for index in range(1, 12)
            ]
        elif case == "policy_extra_field":
            evidence["policy"]["unbound"] = "value"
        elif case == "policy_semantics":
            evidence["policy"]["probability_semantics"] = "post_capture_zero"
        elif case == "policy_artifact_checksum":
            evidence["policy"]["score_artifact_sha256"] = "not-a-checksum"
        elif case == "policy_empty_algorithm":
            evidence["policy"]["calibration_algorithm"] = ""
        else:  # pragma: no cover - parametrization is fixed above
            raise AssertionError(case)
        raw = zlib.decompress(
            (store.output_dir / evidence["storage"]["path"]).read_bytes()
        )
        _rebind_score_payload(payload, score_raw=raw)

    _rewrite_checkpoint(store, mutate)
    with pytest.raises(RunnerContractError, match="p_pre_zero|score"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=_prepared_datasets(prepared),
        )


def test_checkpoint_rejects_observed_zero_count_larger_than_retained_matrix(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    entry = _entry(prepared, "observed")
    plan = CompetitionPlan(
        schema_version=1,
        input_hashes={
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "implementation_source_sha256": implementation_source_sha256(),
        },
        entries=(entry,),
        plan_sha256="c" * 64,
    )
    execution = run_observed(
        load_method_registry(METHODS).by_id("observed"), prepared.method_input
    )
    attempt = evaluate_adapter_outcome(
        entry,
        prepared,
        AdapterOutcome.completed(
            execution,
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=0,
        ),
    )
    store = CheckpointStore(tmp_path / "impossible-zero-count")
    store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )

    def mutate(payload: dict[str, object]) -> None:
        record = payload["records"][0]
        record["run"]["observed_zero_count"] = 9
        evidence = record["p_pre_zero_evidence"]
        evidence["overall"]["n"] = 9
        for metric in evidence["overall"]["metrics"].values():
            metric["n"] = 9
        for stratum_type in (
            "library_size_quartiles",
            "truth_expression_bins",
        ):
            first = evidence["strata"][stratum_type][0]
            first["n"] += 3
            for metric in first["metrics"].values():
                metric["n"] = first["n"]
        _rebind_score_payload(payload)

    _rewrite_checkpoint(store, mutate)
    with pytest.raises(RunnerContractError, match="observed_zero_count"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=_prepared_datasets(prepared),
        )


def test_development_checkpoint_bounded_decompression_rejects_zip_bomb(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    report = store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )
    evidence = report.records[0]["p_pre_zero_evidence"]
    storage = evidence["storage"]
    path = store.output_dir / storage["path"]
    oversized = zlib.compress(b"x" * (int(storage["uncompressed_nbytes"]) + 1), level=6)
    path.write_bytes(oversized)

    def bind_oversized(payload):
        receipt = payload["records"][0]["p_pre_zero_evidence"]["storage"]
        receipt["compressed_sha256"] = hashlib.sha256(oversized).hexdigest()
        receipt["compressed_nbytes"] = len(oversized)

    _rewrite_checkpoint(store, bind_oversized)
    with pytest.raises(RunnerContractError, match="p_pre_zero|score.*compressed"):
        store.load(
            plan,
            registry=load_method_registry(METHODS),
            prepared_datasets=_prepared_datasets(prepared),
        )


def test_development_evidence_manifest_includes_realized_score_artifact(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        load_completed_reconstruction_checkpoint,
    )

    prepared = _prepared()
    plan, attempt, store = _real_development_checkpoint_case(tmp_path, prepared)
    store.append(
        plan,
        None,
        attempt,
        _budget_for_attempt(plan, attempt),
        registry=load_method_registry(METHODS),
        prepared_datasets=_prepared_datasets(prepared),
    )

    evidence = load_completed_reconstruction_checkpoint(
        store.output_dir,
        plan,
        prepared_datasets=_prepared_datasets(prepared),
        authority_repository=tmp_path,
    )
    score = next(item for item in evidence.raw_artifacts if item.kind == "p_pre_zero")
    assert score.run_id == "run-maskimpute-symsim"
    assert score.path.endswith(".p-pre-zero-f64.zlib")
    assert (
        score.file_sha256
        == hashlib.sha256((store.output_dir / score.path).read_bytes()).hexdigest()
    )
