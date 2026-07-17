from __future__ import annotations

import hashlib
import json
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
)
from maskimpute_benchmark.protocol import canonical_sha256


METHODS = Path("study/methods.json")
SHA_A = "a" * 64
SHA_B = "b" * 64


def _prepared(mechanism: str = "symsim") -> PreparedDataset:
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
        retained_cell_ids_sha256=SHA_B,
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
    prepared: PreparedDataset, probability: np.ndarray | None = None
) -> MaskImputeAdapterExecution:
    if probability is None:
        probability = np.array(
            [[0.8, 0.2, 0.0, 0.4], [0.7, 0.0, 0.3, 0.9]], dtype="<f8"
        )
    result = ImputationResult(
        selective_counts=prepared.method_input.counts,
        denoised_counts=prepared.method_input.counts,
        p_pre_zero=probability,
        latent=np.ones((2, 1), dtype="<f8"),
        diagnostics={
            "score": {
                "source": "retained_calibrator",
                "score_artifact_sha256": "4" * 64,
                "score_input_sha256": "5" * 64,
                "score_config_sha256": "6" * 64,
                "calibration_artifact_sha256": "7" * 64,
                "retained_calibrator": "identity",
                "calibration_scope": "leave_one_biological_draw_out",
                "equivalence_reason": "retained_identity_calibrator_equals_direct_score",
            }
        },
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


def _completed_maskimpute(prepared: PreparedDataset):
    return evaluate_adapter_outcome(
        _entry(prepared),
        prepared,
        AdapterOutcome.completed(
            _maskimpute_execution(prepared),
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=1,
            calibration_fold_receipt=CalibrationFoldReceipt(
                calibration_artifact_sha256="7" * 64,
                calibration_context_sha256="8" * 64,
                mechanism=prepared.binding.mechanism,
                biological_id=prepared.binding.biological_id,
                training_manifest_sha256s=("9" * 64,),
                held_out_manifest_sha256s=("a" * 64,),
                fold_calibrator_sha256="b" * 64,
            ),
        ),
    )


def _plan(prepared: PreparedDataset) -> CompetitionPlan:
    return CompetitionPlan(
        schema_version=1,
        input_hashes={
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "implementation_source_sha256": implementation_source_sha256(),
        },
        entries=(_entry(prepared),),
        plan_sha256="c" * 64,
    )


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
        "retained_cell_ids_sha256": SHA_B,
    }
    assert record["matrix"]["shape"] == [2, 4]
    assert record["matrix"]["dtype"] == "<f8"
    assert len(record["matrix"]["semantic_sha256"]) == 64
    assert record["policy"]["score_source"] == "retained_calibrator"
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
    plan = _plan(prepared)
    attempt = _completed_maskimpute(prepared)
    first = CheckpointStore(tmp_path / "first")
    second = CheckpointStore(tmp_path / "second")

    first_report = first.append(plan, None, attempt, DevelopmentBudget())
    second_report = second.append(plan, None, attempt, DevelopmentBudget())
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
    assert CheckpointStore(first.output_dir).load(plan) == first_report


def test_development_checkpoint_rejects_score_tamper_and_partial_receipts(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    report = store.append(
        plan, None, _completed_maskimpute(prepared), DevelopmentBudget()
    )
    evidence = report.records[0]["p_pre_zero_evidence"]
    path = store.output_dir / evidence["storage"]["path"]
    original = path.read_bytes()

    path.write_bytes(original + b"tamper")
    with pytest.raises(RunnerContractError, match="p_pre_zero|score.*checksum"):
        store.load(plan)

    path.write_bytes(original)
    _rewrite_checkpoint(
        store,
        lambda payload: payload["records"][0]["p_pre_zero_evidence"]["storage"].update(
            {"compression_level": None}
        ),
    )
    with pytest.raises(RunnerContractError, match="p_pre_zero|score.*partial"):
        store.load(plan)


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
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / case)
    store.append(plan, None, _completed_maskimpute(prepared), DevelopmentBudget())

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
        store.load(plan)


def test_development_checkpoint_bounded_decompression_rejects_zip_bomb(
    tmp_path: Path,
) -> None:
    prepared = _prepared()
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    report = store.append(
        plan, None, _completed_maskimpute(prepared), DevelopmentBudget()
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
        store.load(plan)


def test_development_evidence_manifest_includes_realized_score_artifact(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        load_completed_reconstruction_checkpoint,
    )

    prepared = _prepared()
    plan = _plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    store.append(plan, None, _completed_maskimpute(prepared), DevelopmentBudget())

    evidence = load_completed_reconstruction_checkpoint(store.output_dir, plan)
    score = next(item for item in evidence.raw_artifacts if item.kind == "p_pre_zero")
    assert score.run_id == "run-maskimpute-symsim"
    assert score.path.endswith(".p-pre-zero-f64.zlib")
    assert (
        score.file_sha256
        == hashlib.sha256((store.output_dir / score.path).read_bytes()).hexdigest()
    )
