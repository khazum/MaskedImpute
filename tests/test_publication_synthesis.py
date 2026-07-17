from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import pytest

import maskimpute_benchmark.publication_synthesis as publication_synthesis
from maskimpute_benchmark.downstream_evaluation import DOWNSTREAM_ENDPOINT_NAMES
from maskimpute_benchmark.downstream_evidence import (
    DatasetEvidenceBinding,
    DownstreamEvidenceManifest,
    DownstreamEvidencePlan,
    DownstreamPlanEntry,
    EvaluatedRoundBinding,
)
from maskimpute_benchmark.final_null_de import (
    FINAL_NULL_DE_ALGORITHM,
    FinalNullDEManifest,
    FinalNullDEPlan,
)
from maskimpute_benchmark.protocol import canonical_sha256
from maskimpute_benchmark.publication_synthesis import (
    PublicationSynthesisError,
    _LoadedPublicationEvidence,
    _build_final_null_de_gate,
    _build_publication_synthesis,
    generate_publication_synthesis,
)
from maskimpute_benchmark.runner import AuthorizedConfiguration
from maskimpute_benchmark.scaling import (
    ScalingCheckpoint,
    scaling_checkpoint_payload,
)


CANDIDATE = "maskimpute"
COMPARATORS = ("observed", "dca")
PRIMARY_METRICS = ("mse", "mse_dropout")
PAIRWISE_EXCLUSION_NAMES = (
    "failed_rows",
    "nonfinite_rows",
    "duplicate_rows",
    "missing_method_pairs",
    "missing_comparator_pairs",
    "zero_comparator_pairs",
    "nonrepresentable_effect_pairs",
    "biological_draws_without_pairs",
    "bootstrap_zero_comparator_pairs",
    "bootstrap_nonrepresentable_effect_pairs",
    "bootstrap_empty_replicates",
)
VIEWS = (
    ("symsim", "draw-1", "moderate", "symsim-draw-1-moderate"),
    ("symsim", "draw-1", "severe", "symsim-draw-1-severe"),
    ("sergio", "draw-2", "moderate", "sergio-draw-2-moderate"),
    ("sergio", "draw-2", "severe", "sergio-draw-2-severe"),
)


def _digest(label: str) -> str:
    return canonical_sha256({"test_binding": label})


def _sealed(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    body = dict(value)
    return MappingProxyType({**body, field: canonical_sha256(body)})


def _frozen_method() -> Mapping[str, object]:
    assessment = {
        "configuration_id": "candidate-01",
        "version": "v29",
        "eligible": True,
        "efficacy_pass": True,
        "safety_pass": True,
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "candidate_method_id": CANDIDATE,
        "selected_configuration_id": "candidate-01",
        "selected_version": "v29",
        "selection_gate_table": [assessment],
        "selected_assessment": assessment,
        "required_comparator_ids": list(COMPARATORS),
    }
    return _sealed(body, "payload_sha256")


def _scaling_checkpoint(frozen_sha256: str) -> ScalingCheckpoint:
    body = {
        "schema_version": 1,
        "plan_sha256": _digest("scaling-plan"),
        "input_hashes": {
            "frozen_method_sha256": frozen_sha256,
            "protocol_file_sha256": _digest("protocol-file"),
        },
        "planned_run_count": 2,
        "status": "completed",
        "datasets": [{"cells": 1_000, "dataset_sha256": _digest("scaling-dataset")}],
        "records": [
            {"run_id": "scaling-maskimpute-1000", "status": "completed"},
            {"run_id": "scaling-dca-1000", "status": "completed"},
        ],
    }
    return ScalingCheckpoint(
        schema_version=1,
        plan_sha256=str(body["plan_sha256"]),
        input_hashes=MappingProxyType(dict(body["input_hashes"])),
        planned_run_count=2,
        status="completed",
        datasets=tuple(MappingProxyType(row) for row in body["datasets"]),
        records=tuple(MappingProxyType(row) for row in body["records"]),
        checkpoint_sha256=canonical_sha256(body),
    )


def _evaluated_binding(
    scaling: ScalingCheckpoint,
) -> EvaluatedRoundBinding:
    return EvaluatedRoundBinding(
        repository_root="/study",
        round_root="/study/final/round-01",
        round_id="round-01",
        evaluation_receipt_path="evaluation_receipt.json",
        evaluation_receipt_file_sha256=_digest("receipt-file"),
        evaluation_receipt_payload_sha256=_digest("receipt-payload"),
        result_manifest_sha256=_digest("result-manifest"),
        final_plan_sha256=_digest("final-plan"),
        final_execution_manifest_path="results/final/execution/manifest.json",
        final_execution_manifest_file_sha256=_digest("execution-manifest-file"),
        final_execution_manifest_payload_sha256=_digest(
            "execution-manifest-payload"
        ),
        execution_validation_sha256=_digest("execution-validation"),
        storage_preflight_sha256=_digest("storage-preflight"),
        scaling_evidence_sha256=_digest("scaling-evidence"),
        scaling_plan_sha256=scaling.plan_sha256,
        scaling_checkpoint_path="results/scaling/checkpoints/00000003.json",
        scaling_checkpoint_file_sha256=_digest("scaling-checkpoint-file"),
        scaling_checkpoint_payload_sha256=canonical_sha256(
            scaling_checkpoint_payload(scaling)
        ),
        scaling_checkpoint_history_sha256=_digest("scaling-history"),
        scaling_checkpoint_history_count=3,
        scaling_result_files_sha256=_digest("scaling-result-files"),
        scaling_result_file_count=3,
    )


def _dataset_bindings() -> tuple[DatasetEvidenceBinding, ...]:
    return tuple(
        DatasetEvidenceBinding(
            dataset_id=dataset_id,
            path=f"/study-data/{dataset_id}.h5ad",
            file_sha256=_digest(f"{dataset_id}-file"),
            dataset_sha256=_digest(f"{dataset_id}-semantic"),
            mechanism=mechanism,
            biological_id=biological_id,
            technical_view=technical_view,
            method_input_sha256=_digest(f"{dataset_id}-method-input"),
            dataset_qc_policy_sha256=_digest(f"{dataset_id}-qc"),
            excluded_cell_count=0,
            excluded_cell_ids_sha256=_digest(f"{dataset_id}-excluded"),
            retained_cell_count=2,
            retained_cell_ids_sha256=_digest(f"{dataset_id}-cells"),
            retained_gene_count=2,
            observed_zero_count=1,
            retained_cell_ids=("cell-1", "cell-2"),
            gene_ids=("gene-1", "gene-2"),
        )
        for mechanism, biological_id, technical_view, dataset_id in VIEWS
    )


def _configurations() -> tuple[AuthorizedConfiguration, ...]:
    return tuple(
        AuthorizedConfiguration.create(
            method_id=method,
            configuration_id=("candidate-01" if method == CANDIDATE else "registry-default"),
            kind=("candidate_search" if method == CANDIDATE else "registry"),
            payload={"method_id": method},
            requires_count_score=method == CANDIDATE,
            requires_calibration=method == CANDIDATE,
        )
        for method in (CANDIDATE, *COMPARATORS)
    )


def _entries(
    datasets: tuple[DatasetEvidenceBinding, ...],
    configurations: tuple[AuthorizedConfiguration, ...],
) -> tuple[DownstreamPlanEntry, ...]:
    by_config = {value.method_id: value for value in configurations}
    values: list[DownstreamPlanEntry] = []
    ordinal = 0
    for method, seeds in (
        (CANDIDATE, (42, 43, 44)),
        ("observed", (None,)),
        ("dca", (None,)),
    ):
        config = by_config[method]
        for binding in datasets:
            for seed in seeds:
                ordinal += 1
                seed_id = "deterministic" if seed is None else f"seed-{seed}"
                run_id = f"{method}-{binding.dataset_id}-{seed_id}"
                values.append(
                    DownstreamPlanEntry(
                        ordinal=ordinal,
                        source_record_path=f"records/{ordinal:08d}.json",
                        source_record_sha256=_digest(f"{run_id}-source-record"),
                        run_id=run_id,
                        method_id=method,
                        dataset_id=binding.dataset_id,
                        source_dataset_sha256=binding.dataset_sha256,
                        mechanism=binding.mechanism,
                        biological_id=binding.biological_id,
                        technical_view=binding.technical_view,
                        model_seed=seed,
                        configuration_id=config.configuration_id,
                        configuration_sha256=config.configuration_sha256,
                        configuration_kind=config.kind,
                        method_artifact_sha256=_digest(f"{method}-artifact"),
                        method_input_sha256=binding.method_input_sha256,
                        retained_cell_ids_sha256=binding.retained_cell_ids_sha256,
                        status="completed",
                        reason=None,
                        evaluator_output_sha256=_digest(f"{run_id}-output"),
                        evaluator_output_path=f"outputs/{run_id}.bin",
                        evaluator_output_file_sha256=_digest(f"{run_id}-output-file"),
                        evaluator_output_shape=(2, 2),
                        evaluator_output_encoding="zlib_raw_f64_v1",
                        evaluator_output_uncompressed_nbytes=32,
                        evaluator_output_uncompressed_sha256=_digest(
                            f"{run_id}-uncompressed"
                        ),
                    )
                )
    return tuple(values)


def _downstream_plan(
    binding: EvaluatedRoundBinding,
) -> DownstreamEvidencePlan:
    datasets = _dataset_bindings()
    configurations = _configurations()
    entries = _entries(datasets, configurations)
    provisional = DownstreamEvidencePlan(
        source_root=binding.repository_root,
        source_kind="final",
        evidence_scope="all",
        evaluator_source_sha256=_digest("downstream-evaluator"),
        source_manifest_path=binding.final_execution_manifest_path,
        source_manifest_file_sha256=binding.final_execution_manifest_file_sha256,
        source_manifest_payload_sha256=(
            binding.final_execution_manifest_payload_sha256
        ),
        source_plan_sha256=binding.final_plan_sha256,
        source_input_hashes_sha256=_digest("source-input-hashes"),
        source_statuses_sha256=_digest("source-statuses"),
        source_plan_authority="independent",
        evaluated_round_binding=binding,
        development_revision_versions=(),
        development_sources=(),
        datasets=datasets,
        configurations=configurations,
        entries=entries,
        plan_sha256=_digest("temporary-downstream-plan"),
    )
    return replace(provisional, plan_sha256=canonical_sha256(provisional.body()))


def _downstream_record(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
) -> Mapping[str, object]:
    endpoint_rows = [
        {
            "endpoint": endpoint,
            "status": "completed",
            "reason_code": None,
            "value": 0.5,
        }
        for endpoint in DOWNSTREAM_ENDPOINT_NAMES
    ]
    body = {
        "schema_version": 1,
        "ordinal": entry.ordinal,
        "source_kind": plan.source_kind,
        "run_id": entry.run_id,
        "runner_method_id": entry.method_id,
        "method": entry.method_id,
        "dataset_id": entry.dataset_id,
        "mechanism": entry.mechanism,
        "biological_id": entry.biological_id,
        "technical_view": entry.technical_view,
        "model_seed": entry.model_seed,
        "run_status": entry.status,
        "run_reason": entry.reason,
        "endpoints": endpoint_rows,
    }
    return _sealed(body, "record_sha256")


def _downstream_manifest(
    plan: DownstreamEvidencePlan,
) -> DownstreamEvidenceManifest:
    records = tuple(_downstream_record(plan, entry) for entry in plan.entries)
    references = [
        {
            "ordinal": entry.ordinal,
            "run_id": entry.run_id,
            "path": f"records/{entry.ordinal:08d}.json",
            "sha256": _digest(f"downstream-record-file-{entry.ordinal}"),
            "record_sha256": records[entry.ordinal - 1]["record_sha256"],
        }
        for entry in plan.entries
    ]
    body = {
        "schema_version": 3,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": _digest("downstream-plan-file"),
        "source_kind": plan.source_kind,
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "source_manifest_path": plan.source_manifest_path,
        "source_manifest_file_sha256": plan.source_manifest_file_sha256,
        "source_manifest_payload_sha256": plan.source_manifest_payload_sha256,
        "source_plan_sha256": plan.source_plan_sha256,
        "source_input_hashes_sha256": plan.source_input_hashes_sha256,
        "source_statuses_sha256": plan.source_statuses_sha256,
        "source_plan_authority": plan.source_plan_authority,
        "evaluated_round_binding_sha256": plan.evaluated_round_binding.binding_sha256,
        "development_revision_versions": [],
        "development_sources": [],
        "planned_denominator_count": len(plan.entries),
        "recorded_denominator_count": len(records),
        "endpoint_row_count": len(records) * len(DOWNSTREAM_ENDPOINT_NAMES),
        "records": references,
    }
    payload = _sealed(body, "manifest_sha256")
    return DownstreamEvidenceManifest(
        plan_sha256=plan.plan_sha256,
        manifest_sha256=str(payload["manifest_sha256"]),
        planned_denominator_count=len(plan.entries),
        endpoint_row_count=len(records) * len(DOWNSTREAM_ENDPOINT_NAMES),
        records=records,
        payload=payload,
    )


def _null_plan(
    plan: DownstreamEvidencePlan,
    downstream: DownstreamEvidenceManifest,
) -> FinalNullDEPlan:
    binding = plan.evaluated_round_binding
    assert binding is not None
    provisional = FinalNullDEPlan(
        source_plan=plan,
        downstream_directory=(
            "/study-final-analysis/downstream/"
            f"{binding.round_id}/{binding.evaluation_receipt_payload_sha256}"
        ),
        downstream_manifest_file_sha256=_digest("downstream-manifest-file"),
        downstream_manifest_payload_sha256=downstream.manifest_sha256,
        evaluator_source_sha256=plan.evaluator_source_sha256,
        plan_sha256=_digest("temporary-null-plan"),
    )
    return replace(provisional, plan_sha256=canonical_sha256(provisional.body()))


def _null_record(
    plan: FinalNullDEPlan,
    entry: DownstreamPlanEntry,
    *,
    fpr: float,
) -> Mapping[str, object]:
    binding = plan.source_plan.evaluated_round_binding
    assert binding is not None
    body = {
        "schema_version": 1,
        "algorithm": FINAL_NULL_DE_ALGORITHM,
        "ordinal": entry.ordinal,
        "source_plan_sha256": plan.source_plan.plan_sha256,
        "evaluated_round_binding_sha256": binding.binding_sha256,
        "evaluation_receipt_payload_sha256": (
            binding.evaluation_receipt_payload_sha256
        ),
        "run_id": entry.run_id,
        "method_id": entry.method_id,
        "dataset_id": entry.dataset_id,
        "dataset_sha256": entry.source_dataset_sha256,
        "mechanism": entry.mechanism,
        "biological_id": entry.biological_id,
        "technical_view": entry.technical_view,
        "model_seed": entry.model_seed,
        "status": "completed",
        "reason_code": None,
        "fpr": fpr,
        "n_tested_genes": 100,
    }
    return _sealed(body, "record_sha256")


def _null_records(plan: FinalNullDEPlan) -> tuple[Mapping[str, object], ...]:
    fprs = {CANDIDATE: 0.06, "observed": 0.05, "dca": 0.055}
    return tuple(
        _null_record(plan, entry, fpr=fprs[entry.method_id])
        for entry in plan.source_plan.entries
    )


def _null_manifest(
    plan: FinalNullDEPlan,
    records: tuple[Mapping[str, object], ...] | None = None,
) -> FinalNullDEManifest:
    values = _null_records(plan) if records is None else records
    binding = plan.source_plan.evaluated_round_binding
    assert binding is not None
    references = [
        {
            "ordinal": index,
            "run_id": record["run_id"],
            "path": f"records/{index:08d}.json",
            "sha256": _digest(f"null-record-file-{index}"),
            "record_sha256": record["record_sha256"],
        }
        for index, record in enumerate(values, start=1)
    ]
    body = {
        "schema_version": 1,
        "algorithm": FINAL_NULL_DE_ALGORITHM,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": _digest("null-plan-file"),
        "source_plan_sha256": plan.source_plan.plan_sha256,
        "evaluated_round_binding_sha256": binding.binding_sha256,
        "downstream_manifest_file_sha256": (
            plan.downstream_manifest_file_sha256
        ),
        "downstream_manifest_payload_sha256": (
            plan.downstream_manifest_payload_sha256
        ),
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "planned_denominator_count": len(plan.source_plan.entries),
        "recorded_denominator_count": len(values),
        "records": references,
    }
    payload = _sealed(body, "manifest_sha256")
    return FinalNullDEManifest(
        plan_sha256=plan.plan_sha256,
        manifest_sha256=str(payload["manifest_sha256"]),
        manifest_file_sha256=_digest("null-manifest-file"),
        planned_denominator_count=len(plan.source_plan.entries),
        records=values,
        payload=payload,
    )


def _comparison(
    metric: str,
    comparator: str,
    *,
    ci_upper: float = -0.01,
    holm_status: str = "ok",
    adjusted_p: float | None = 0.04,
) -> dict[str, object]:
    return {
        "candidate_method_id": CANDIDATE,
        "comparator_method_id": comparator,
        "metric": metric,
        "status": "ok",
        "reason": None,
        "favorable_direction": "lower",
        "direction_source": "validated_frozen_metric_direction_contract",
        "ci_95_lower": -0.2,
        "ci_95_upper": ci_upper,
        "n_independent_biological_draws": 2,
        "n_paired_dataset_views": 4,
        "exclusions": {name: 0 for name in PAIRWISE_EXCLUSION_NAMES},
        "holm_family_id": "protocol_primary_metrics",
        "holm_hypothesis_count": len(PRIMARY_METRICS),
        "holm_status": holm_status,
        "holm_reason": None if holm_status == "ok" else "raw_p_value_unavailable",
        "holm_adjusted_p_value": adjusted_p,
    }


def _primary_report(
    frozen: Mapping[str, object],
    binding: EvaluatedRoundBinding,
    *,
    reconstruction_status: str = "passed",
    strongest_method: str | None = "dca",
    tied_methods: tuple[str, ...] = ("dca",),
    pairwise: tuple[Mapping[str, object], ...] | None = None,
    trajectory_sha256: str | None = None,
) -> Mapping[str, object]:
    strongest_status = "ok" if strongest_method is not None else "unavailable"
    strongest = [
        {
            "metric": metric,
            "status": strongest_status,
            "reason": None if strongest_method is not None else "incomplete_comparator",
            "method_id": strongest_method,
            "tied_method_ids": list(tied_methods),
            "median": 1.1 if strongest_method is not None else None,
            "n_biological_draws": 2 if strongest_method is not None else 0,
        }
        for metric in PRIMARY_METRICS
    ]
    summaries = [
        {
            "status": "complete",
            "method_id": method,
            "metric": metric,
            "median": median,
            "n_biological_draws": 2,
        }
        for metric in PRIMARY_METRICS
        for method, median in ((CANDIDATE, 1.0), ("dca", 1.1), ("observed", 1.2))
    ]
    comparisons = pairwise or tuple(
        _comparison(metric, comparator)
        for comparator in COMPARATORS
        for metric in PRIMARY_METRICS
    )
    inputs: dict[str, object] = {
        "round_id": binding.round_id,
        "evaluation_receipt_payload_sha256": (
            binding.evaluation_receipt_payload_sha256
        ),
        "result_manifest_sha256": binding.result_manifest_sha256,
        "final_plan_sha256": binding.final_plan_sha256,
        "final_execution_manifest_path": binding.final_execution_manifest_path,
        "final_execution_manifest_sha256": (
            binding.final_execution_manifest_file_sha256
        ),
        "final_execution_payload_sha256": (
            binding.final_execution_manifest_payload_sha256
        ),
        "execution_validation_sha256": binding.execution_validation_sha256,
        "storage_preflight_sha256": binding.storage_preflight_sha256,
        "scaling_evidence_sha256": binding.scaling_evidence_sha256,
        "frozen_method_payload_sha256": frozen["payload_sha256"],
        "planned_run_count": 20,
    }
    inputs["trajectory_evidence_sha256"] = (
        _digest("trajectory-evidence")
        if trajectory_sha256 is None
        else trajectory_sha256
    )
    body = {
        "schema_version": 1,
        "status": "completed",
        "candidate_method_id": CANDIDATE,
        "input_bindings": inputs,
        "analysis_policy": {
            "declared_metric_family": {
                "id": "protocol_primary_metrics",
                "metrics": list(PRIMARY_METRICS),
            },
            "metric_direction_contract": {
                "status": "validated",
                "favorable_direction": "lower",
                "metrics": list(PRIMARY_METRICS),
            },
        },
        "reconstruction_claim_gate": {
            "status": reconstruction_status,
            "reason": (
                None
                if reconstruction_status == "passed"
                else f"reconstruction_{reconstruction_status}"
            ),
            "candidate_method_id": CANDIDATE,
            "required_comparator_ids": list(COMPARATORS),
            "draw_collapsed_method_summaries": summaries,
            "strongest_applicable_comparators": strongest,
        },
        "paired_comparisons": list(comparisons),
    }
    return _sealed(body, "analysis_sha256")


def _loaded() -> _LoadedPublicationEvidence:
    frozen = _frozen_method()
    scaling = _scaling_checkpoint(str(frozen["payload_sha256"]))
    binding = _evaluated_binding(scaling)
    plan = _downstream_plan(binding)
    downstream = _downstream_manifest(plan)
    null_plan = _null_plan(plan, downstream)
    null_manifest = _null_manifest(null_plan)
    return _LoadedPublicationEvidence(
        primary_report=_primary_report(frozen, binding),
        frozen_method=frozen,
        downstream_plan=plan,
        downstream_manifest=downstream,
        null_de_plan=null_plan,
        null_de_manifest=null_manifest,
        scaling_checkpoint=scaling,
    )


def _with_null_records(
    loaded: _LoadedPublicationEvidence,
    transform: Callable[[list[dict[str, object]]], list[dict[str, object]]],
) -> _LoadedPublicationEvidence:
    records = [dict(row) for row in loaded.null_de_manifest.records]
    changed = transform(records)
    resealed = tuple(
        _sealed(
            {key: value for key, value in row.items() if key != "record_sha256"},
            "record_sha256",
        )
        for row in changed
    )
    return replace(
        loaded,
        null_de_manifest=_null_manifest(loaded.null_de_plan, resealed),
    )


def _with_report(
    loaded: _LoadedPublicationEvidence,
    **changes: object,
) -> _LoadedPublicationEvidence:
    report = dict(loaded.primary_report)
    report.pop("analysis_sha256")
    report.update(changes)
    return replace(
        loaded,
        primary_report=_sealed(report, "analysis_sha256"),
    )


def _with_report_input(
    loaded: _LoadedPublicationEvidence,
    key: str,
    value: object,
    *,
    remove: bool = False,
) -> _LoadedPublicationEvidence:
    report = dict(loaded.primary_report)
    report.pop("analysis_sha256")
    inputs = dict(report["input_bindings"])
    if remove:
        inputs.pop(key)
    else:
        inputs[key] = value
    report["input_bindings"] = inputs
    return replace(loaded, primary_report=_sealed(report, "analysis_sha256"))


def _with_frozen(
    loaded: _LoadedPublicationEvidence,
    transform: Callable[[dict[str, object]], None],
) -> _LoadedPublicationEvidence:
    frozen = dict(loaded.frozen_method)
    frozen.pop("payload_sha256")
    transform(frozen)
    resealed = _sealed(frozen, "payload_sha256")
    changed = replace(loaded, frozen_method=resealed)
    return _with_report_input(
        changed,
        "frozen_method_payload_sha256",
        resealed["payload_sha256"],
    )


def _install_authoritative_loaders(
    monkeypatch: pytest.MonkeyPatch,
    loaded: _LoadedPublicationEvidence,
    *,
    persisted_downstream_plan: DownstreamEvidencePlan | None = None,
) -> None:
    binding = loaded.downstream_plan.evaluated_round_binding
    assert binding is not None
    repository = Path(binding.repository_root)
    round_dir = Path(binding.round_root)
    downstream_directory = (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "downstream"
        / binding.round_id
        / binding.evaluation_receipt_payload_sha256
    ).absolute()
    null_directory = (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "null-de"
        / binding.round_id
        / binding.evaluation_receipt_payload_sha256
    ).absolute()

    def primary(observed_repository: Path, observed_round: Path) -> Mapping[str, object]:
        assert observed_repository == repository
        assert observed_round == round_dir
        return loaded.primary_report

    def frozen(observed_repository: Path) -> Mapping[str, object]:
        assert observed_repository == repository
        return loaded.frozen_method

    def downstream_plan(
        observed_repository: Path,
        observed_round: Path,
    ) -> DownstreamEvidencePlan:
        assert observed_repository == repository
        assert observed_round == round_dir
        return loaded.downstream_plan

    def persisted_plan(observed_directory: Path) -> DownstreamEvidencePlan:
        assert observed_directory == downstream_directory
        return persisted_downstream_plan or loaded.downstream_plan

    def downstream_manifest(
        observed_directory: Path,
    ) -> DownstreamEvidenceManifest:
        assert observed_directory == downstream_directory
        return loaded.downstream_manifest

    def null_plan(
        observed_repository: Path,
        observed_round: Path,
    ) -> FinalNullDEPlan:
        assert observed_repository == repository
        assert observed_round == round_dir
        return loaded.null_de_plan

    def expected_null_directory(observed_plan: FinalNullDEPlan) -> Path:
        assert observed_plan is loaded.null_de_plan
        return null_directory

    def null_manifest(observed_directory: Path) -> FinalNullDEManifest:
        assert observed_directory == null_directory
        return loaded.null_de_manifest

    def scaling(
        observed_repository: Path,
        observed_round: Path,
    ) -> ScalingCheckpoint:
        assert observed_repository == repository
        assert observed_round == round_dir
        return loaded.scaling_checkpoint

    monkeypatch.setattr(
        publication_synthesis, "generate_final_analysis", primary, raising=False
    )
    monkeypatch.setattr(
        publication_synthesis, "validate_frozen_method", frozen, raising=False
    )
    monkeypatch.setattr(
        publication_synthesis,
        "build_final_downstream_evidence_plan",
        downstream_plan,
        raising=False,
    )
    monkeypatch.setattr(
        publication_synthesis,
        "load_downstream_evidence_plan",
        persisted_plan,
        raising=False,
    )
    monkeypatch.setattr(
        publication_synthesis,
        "load_downstream_evidence_manifest",
        downstream_manifest,
        raising=False,
    )
    monkeypatch.setattr(
        publication_synthesis, "build_final_null_de_plan", null_plan, raising=False
    )
    monkeypatch.setattr(
        publication_synthesis,
        "expected_final_null_de_output_directory",
        expected_null_directory,
        raising=False,
    )
    monkeypatch.setattr(
        publication_synthesis,
        "load_final_null_de_manifest",
        null_manifest,
        raising=False,
    )
    monkeypatch.setattr(
        publication_synthesis,
        "load_publication_scaling_evidence",
        scaling,
        raising=False,
    )


def _superiority_row(
    synthesis: Mapping[str, object], metric: str
) -> Mapping[str, object]:
    permissions = synthesis["claim_permissions"]
    assert isinstance(permissions, Mapping)
    rows = permissions["superiority"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["metric"] == metric)


def test_final_null_de_boundary_and_synthesis_permissions_pass() -> None:
    loaded = _loaded()

    gate = _build_final_null_de_gate(loaded)
    synthesis = _build_publication_synthesis(loaded)

    assert gate["status"] == "passed"
    assert gate["maximum_fpr"] == pytest.approx(0.06)
    assert gate["maximum_above_observed"] == pytest.approx(0.01)
    assert gate["n_biological_draws"] == 2
    assert synthesis["gates"]["final_null_de"] == gate
    assert synthesis["gates"]["competitive"]["status"] == "passed"
    assert synthesis["claim_permissions"]["competitive"] is True
    freeze = synthesis["freeze_prerequisite"]
    assert freeze["gate_flags"] == {
        "efficacy_pass": True,
        "eligible": True,
        "safety_pass": True,
    }
    assert "selected_assessment" not in freeze
    assert synthesis["trajectory"]["role"] == "descriptive_only"
    assert synthesis["trajectory"]["gate_influence"] == "none"
    assert synthesis["downstream"]["numerical_gate_status"] == "not_prespecified"
    assert synthesis["scaling"]["numerical_gate_status"] == "not_prespecified"
    body = {
        key: value for key, value in synthesis.items() if key != "synthesis_sha256"
    }
    assert synthesis["synthesis_sha256"] == canonical_sha256(body)


def test_final_null_de_uses_seed_mean_then_paired_view_mean_not_raw_maximum() -> None:
    candidate_fprs = {
        ("draw-1", "moderate", 42): 0.08,
        ("draw-1", "moderate", 43): 0.04,
        ("draw-1", "moderate", 44): 0.04,
        ("draw-1", "severe", 42): 0.07,
        ("draw-1", "severe", 43): 0.06,
        ("draw-1", "severe", 44): 0.05,
    }

    def change(records: list[dict[str, object]]) -> list[dict[str, object]]:
        for row in records:
            identity = (
                row["biological_id"],
                row["technical_view"],
                row["model_seed"],
            )
            if row["method_id"] == CANDIDATE and identity in candidate_fprs:
                row["fpr"] = candidate_fprs[identity]
        return records

    loaded = _with_null_records(_loaded(), change)

    gate = _build_final_null_de_gate(loaded)

    assert max(
        float(row["fpr"])
        for row in loaded.null_de_manifest.records
        if row["method_id"] == CANDIDATE
    ) == pytest.approx(0.08)
    assert gate["status"] == "passed"
    assert gate["maximum_fpr"] == pytest.approx(0.06)
    assert gate["maximum_above_observed"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("method", "value", "expected_reason"),
    [
        (CANDIDATE, 0.0600001, "maximum_fpr_exceeds_limit"),
        ("observed", 0.0499998, "maximum_above_observed_exceeds_limit"),
    ],
)
def test_final_null_de_complete_threshold_violations_fail(
    method: str,
    value: float,
    expected_reason: str,
) -> None:
    def change(records: list[dict[str, object]]) -> list[dict[str, object]]:
        for row in records:
            if row["method_id"] == method:
                row["fpr"] = value
        return records

    loaded = _with_null_records(_loaded(), change)

    gate = _build_final_null_de_gate(loaded)
    synthesis = _build_publication_synthesis(loaded)

    assert gate["status"] == "failed"
    assert expected_reason in gate["failed_conditions"]
    assert synthesis["gates"]["competitive"]["status"] == "failed"
    assert synthesis["claim_permissions"]["competitive"] is False


@pytest.mark.parametrize("case", ["missing", "noncompleted", "mismatched_view"])
def test_final_null_de_incomplete_denominators_are_unavailable(case: str) -> None:
    def change(records: list[dict[str, object]]) -> list[dict[str, object]]:
        index = next(
            index
            for index, row in enumerate(records)
            if row["method_id"] == CANDIDATE
        )
        if case == "missing":
            records.pop(index)
        elif case == "noncompleted":
            records[index]["status"] = "unavailable"
            records[index]["reason_code"] = "gene_denominator_unavailable"
            records[index]["fpr"] = None
        else:
            records[index]["technical_view"] = "nearby-view"
        return records

    loaded = _with_null_records(_loaded(), change)

    gate = _build_final_null_de_gate(loaded)
    synthesis = _build_publication_synthesis(loaded)

    assert gate["status"] == "unavailable"
    assert gate["reason"] == "incomplete_final_null_de_denominator"
    assert synthesis["gates"]["competitive"]["status"] == "unavailable"
    assert synthesis["claim_permissions"]["competitive"] is False


@pytest.mark.parametrize(
    ("reconstruction_status", "competitive_status"),
    [("failed", "failed"), ("unavailable", "unavailable")],
)
def test_trajectory_evidence_never_rescues_reconstruction(
    reconstruction_status: str,
    competitive_status: str,
) -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    reconstruction = dict(report["reconstruction_claim_gate"])
    reconstruction["status"] = reconstruction_status
    reconstruction["reason"] = f"reconstruction_{reconstruction_status}"
    loaded = _with_report(
        loaded,
        reconstruction_claim_gate=reconstruction,
    )

    synthesis = _build_publication_synthesis(loaded)

    assert synthesis["trajectory"]["evidence_sha256"] == _digest(
        "trajectory-evidence"
    )
    assert synthesis["trajectory"]["gate_influence"] == "none"
    assert synthesis["gates"]["competitive"]["status"] == competitive_status
    assert synthesis["claim_permissions"]["competitive"] is False


def test_superiority_uses_deterministic_strongest_comparator_and_discloses_ties() -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    reconstruction = dict(report["reconstruction_claim_gate"])
    reconstruction["draw_collapsed_method_summaries"] = [
        (
            {**dict(row), "median": 1.1}
            if row["method_id"] == "observed"
            else dict(row)
        )
        for row in reconstruction["draw_collapsed_method_summaries"]
    ]
    reconstruction["strongest_applicable_comparators"] = [
        {
            **dict(row),
            "tied_method_ids": ["dca", "observed"],
        }
        for row in reconstruction["strongest_applicable_comparators"]
    ]
    loaded = _with_report(loaded, reconstruction_claim_gate=reconstruction)

    synthesis = _build_publication_synthesis(loaded)
    permission = _superiority_row(synthesis, "mse")

    assert permission["permitted"] is True
    assert permission["comparator_method_id"] == "dca"
    assert permission["tied_method_ids"] == ["dca", "observed"]
    assert permission["reason"] is None


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("stronger_different", "confidence_interval_not_strictly_favorable"),
        ("zero_crossing", "confidence_interval_not_strictly_favorable"),
        ("missing_holm", "multiplicity_adjustment_unavailable"),
        ("partial_holm_family", "multiplicity_adjustment_unavailable"),
        ("no_complete_comparator", "complete_strongest_comparator_unavailable"),
    ],
)
def test_superiority_fail_closed_conditions(
    case: str,
    expected_reason: str,
) -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    reconstruction = dict(report["reconstruction_claim_gate"])
    comparisons = [dict(row) for row in report["paired_comparisons"]]
    if case == "stronger_different":
        reconstruction["draw_collapsed_method_summaries"] = [
            (
                {**dict(row), "median": 1.0}
                if row["method_id"] == "observed"
                else dict(row)
            )
            for row in reconstruction["draw_collapsed_method_summaries"]
        ]
        reconstruction["strongest_applicable_comparators"] = [
            {
                **dict(row),
                "median": 1.0,
                "method_id": "observed",
                "tied_method_ids": ["observed"],
            }
            for row in reconstruction["strongest_applicable_comparators"]
        ]
        comparisons = [
            (
                {**row, "ci_95_upper": 0.0}
                if row["comparator_method_id"] == "observed"
                else row
            )
            for row in comparisons
        ]
    elif case == "zero_crossing":
        comparisons = [
            (
                {**row, "ci_95_upper": 0.0}
                if row["comparator_method_id"] == "dca"
                else row
            )
            for row in comparisons
        ]
    elif case == "missing_holm":
        comparisons = [
            (
                {
                    **row,
                    "holm_status": "unavailable",
                    "holm_adjusted_p_value": None,
                }
                if row["comparator_method_id"] == "dca"
                else row
            )
            for row in comparisons
        ]
    elif case == "partial_holm_family":
        comparisons = [
            (
                {**row, "holm_hypothesis_count": 1}
                if row["comparator_method_id"] == "dca"
                else row
            )
            for row in comparisons
        ]
    else:
        reconstruction["status"] = "unavailable"
        reconstruction["reason"] = "incomplete_final_claim_denominator"
        reconstruction["draw_collapsed_method_summaries"] = []
        reconstruction["strongest_applicable_comparators"] = [
            {
                **dict(row),
                "status": "unavailable",
                "reason": "incomplete_comparator",
                "method_id": None,
                "tied_method_ids": [],
                "median": None,
                "n_biological_draws": 0,
            }
            for row in reconstruction["strongest_applicable_comparators"]
        ]
    loaded = _with_report(
        loaded,
        reconstruction_claim_gate=reconstruction,
        paired_comparisons=comparisons,
    )

    synthesis = _build_publication_synthesis(loaded)
    permission = _superiority_row(synthesis, "mse")

    assert permission["permitted"] is False
    assert permission["reason"] == expected_reason
    if case == "stronger_different":
        assert permission["comparator_method_id"] == "observed"


def test_superiority_rejects_favorable_subset_with_excluded_draw() -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    comparisons = [dict(row) for row in report["paired_comparisons"]]
    for row in comparisons:
        if row["metric"] == "mse" and row["comparator_method_id"] == "dca":
            exclusions = dict(row["exclusions"])
            exclusions["zero_comparator_pairs"] = 1
            row["exclusions"] = exclusions
            row["n_independent_biological_draws"] = 1
            row["n_paired_dataset_views"] = 3
    loaded = _with_report(loaded, paired_comparisons=comparisons)

    permission = _superiority_row(_build_publication_synthesis(loaded), "mse")

    assert permission["permitted"] is False
    assert permission["reason"] == "complete_pairwise_denominator_unavailable"


def test_superiority_accepts_prespecified_out_of_domain_rows() -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    comparisons = [dict(row) for row in report["paired_comparisons"]]
    for row in comparisons:
        if row["metric"] == "mse" and row["comparator_method_id"] == "dca":
            exclusions = dict(row["exclusions"])
            exclusions.update(
                {
                    "biological_draws_without_pairs": 1,
                    "failed_rows": 2,
                    "missing_comparator_pairs": 1,
                    "missing_method_pairs": 1,
                }
            )
            row["exclusions"] = exclusions
    loaded = _with_report(loaded, paired_comparisons=comparisons)

    permission = _superiority_row(_build_publication_synthesis(loaded), "mse")

    assert permission["permitted"] is True
    assert permission["reason"] is None


def test_superiority_rejects_tied_comparator_outside_required_denominator() -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    reconstruction = dict(report["reconstruction_claim_gate"])
    reconstruction["strongest_applicable_comparators"] = [
        {
            **dict(row),
            "tied_method_ids": ["dca", "magic"],
        }
        for row in reconstruction["strongest_applicable_comparators"]
    ]
    loaded = _with_report(loaded, reconstruction_claim_gate=reconstruction)

    with pytest.raises(PublicationSynthesisError, match="tie authority"):
        _build_publication_synthesis(loaded)


def test_superiority_rejects_omitted_equal_strongest_comparator() -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    reconstruction = dict(report["reconstruction_claim_gate"])
    reconstruction["draw_collapsed_method_summaries"] = [
        (
            {**dict(row), "median": 1.1}
            if row["method_id"] == "observed"
            else dict(row)
        )
        for row in reconstruction["draw_collapsed_method_summaries"]
    ]
    loaded = _with_report(loaded, reconstruction_claim_gate=reconstruction)

    with pytest.raises(PublicationSynthesisError, match="tie authority"):
        _build_publication_synthesis(loaded)


def test_superiority_rejects_incomplete_comparator_holm_family() -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    report["paired_comparisons"] = [
        row
        for row in report["paired_comparisons"]
        if not (
            row["comparator_method_id"] == "dca"
            and row["metric"] == "mse_dropout"
        )
    ]
    loaded = _with_report(loaded, paired_comparisons=report["paired_comparisons"])

    permission = _superiority_row(_build_publication_synthesis(loaded), "mse")

    assert permission["permitted"] is False
    assert permission["reason"] == "complete_pairwise_family_unavailable"


def test_superiority_requires_lower_direction_for_every_primary_metric() -> None:
    loaded = _loaded()
    report = dict(loaded.primary_report)
    policy = dict(report["analysis_policy"])
    direction = dict(policy["metric_direction_contract"])
    direction["metrics"] = ["mse"]
    policy["metric_direction_contract"] = direction
    loaded = _with_report(loaded, analysis_policy=policy)

    with pytest.raises(PublicationSynthesisError, match="lower-direction"):
        _build_publication_synthesis(loaded)


def test_public_api_signature_has_no_evidence_or_threshold_overrides() -> None:
    signature = inspect.signature(generate_publication_synthesis)

    assert tuple(signature.parameters) == ("repository", "round_dir")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )
    assert signature.parameters["repository"].annotation == "Path"
    assert signature.parameters["round_dir"].annotation == "Path"
    assert issubclass(PublicationSynthesisError, RuntimeError)


def test_production_replays_only_authoritative_final_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    _install_authoritative_loaders(monkeypatch, loaded)

    observed = generate_publication_synthesis(
        Path("/study"), Path("/study/final/round-01")
    )

    assert observed == _build_publication_synthesis(loaded)


@pytest.mark.parametrize(
    ("field", "value"),
    [("source_kind", "development"), ("evidence_scope", "selection_primary")],
)
def test_production_rejects_nonfinal_source_kind_or_scope(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    loaded = _loaded()
    changed_plan = replace(loaded.downstream_plan, **{field: value})
    loaded = replace(loaded, downstream_plan=changed_plan)
    _install_authoritative_loaders(monkeypatch, loaded)

    with pytest.raises(PublicationSynthesisError, match="complete frozen final"):
        generate_publication_synthesis(
            Path("/study"), Path("/study/final/round-01")
        )


def test_production_rejects_rebuilt_and_persisted_downstream_plan_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    persisted = replace(
        loaded.downstream_plan,
        plan_sha256=_digest("different-persisted-downstream-plan"),
    )
    _install_authoritative_loaders(
        monkeypatch,
        loaded,
        persisted_downstream_plan=persisted,
    )

    with pytest.raises(PublicationSynthesisError, match="persisted downstream plan"):
        generate_publication_synthesis(
            Path("/study"), Path("/study/final/round-01")
        )


def test_production_rejects_null_de_source_plan_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    different_source = replace(
        loaded.downstream_plan,
        source_statuses_sha256=_digest("different-source-statuses"),
    )
    loaded = replace(
        loaded,
        null_de_plan=replace(loaded.null_de_plan, source_plan=different_source),
    )
    _install_authoritative_loaders(monkeypatch, loaded)

    with pytest.raises(PublicationSynthesisError, match="null-DE source plan"):
        generate_publication_synthesis(
            Path("/study"), Path("/study/final/round-01")
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("round_id", "round-02"),
        ("evaluation_receipt_payload_sha256", _digest("wrong-receipt")),
        ("result_manifest_sha256", _digest("wrong-result-manifest")),
        ("final_plan_sha256", _digest("wrong-final-plan")),
        (
            "final_execution_manifest_sha256",
            _digest("wrong-execution-manifest-file"),
        ),
        (
            "final_execution_payload_sha256",
            _digest("wrong-execution-manifest-payload"),
        ),
        ("scaling_evidence_sha256", _digest("wrong-scaling-evidence")),
        ("planned_run_count", 19),
    ],
)
def test_production_rejects_primary_evaluated_round_binding_differences(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    loaded = _with_report_input(_loaded(), field, value)
    _install_authoritative_loaders(monkeypatch, loaded)

    with pytest.raises(PublicationSynthesisError, match="binding|denominator"):
        generate_publication_synthesis(
            Path("/study"), Path("/study/final/round-01")
        )


def test_production_rejects_missing_trajectory_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _with_report_input(
        _loaded(),
        "trajectory_evidence_sha256",
        None,
        remove=True,
    )
    _install_authoritative_loaders(monkeypatch, loaded)

    with pytest.raises(PublicationSynthesisError, match="trajectory evidence"):
        generate_publication_synthesis(
            Path("/study"), Path("/study/final/round-01")
        )


@pytest.mark.parametrize(
    "case",
    [
        "selected_absent",
        "selected_differs",
        "eligible_false",
        "efficacy_false",
        "safety_false",
    ],
)
def test_production_rejects_invalid_frozen_selected_assessment(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    def change(frozen: dict[str, object]) -> None:
        if case == "selected_absent":
            frozen.pop("selected_assessment")
            return
        selected = dict(frozen["selected_assessment"])
        if case == "selected_differs":
            selected["review_only_field"] = "different"
        else:
            selected[case.removesuffix("_false")] = False
        frozen["selected_assessment"] = selected

    loaded = _with_frozen(_loaded(), change)
    _install_authoritative_loaders(monkeypatch, loaded)

    with pytest.raises(PublicationSynthesisError, match="freeze prerequisite"):
        generate_publication_synthesis(
            Path("/study"), Path("/study/final/round-01")
        )


def test_production_has_no_locally_rehashed_evidence_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    _install_authoritative_loaders(monkeypatch, loaded)

    with pytest.raises(TypeError):
        generate_publication_synthesis(
            Path("/study"),
            Path("/study/final/round-01"),
            primary_report=loaded.primary_report,  # type: ignore[call-arg]
        )
