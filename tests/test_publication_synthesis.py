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
    DownstreamEvidenceError,
    DownstreamEvidenceManifest,
    DownstreamEvidencePlan,
    DownstreamPlanEntry,
    EvaluatedRoundBinding,
    expected_final_downstream_output_directory,
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
        final_execution_manifest_path=(
            "results/final/execution/execution_manifest.json"
        ),
        final_execution_manifest_file_sha256=_digest("execution-manifest-file"),
        final_execution_manifest_payload_sha256=_digest("execution-manifest-payload"),
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
        trajectory_evidence_sha256=_digest("trajectory-evidence"),
        trajectory_plan_sha256=_digest("trajectory-plan"),
        trajectory_execution_claim_sha256=_digest("trajectory-claim"),
        trajectory_execution_environment_sha256=_digest("trajectory-environment"),
        trajectory_dataset_id="trajectory-exact-latent-01",
        trajectory_dataset_sha256=_digest("trajectory-dataset"),
        trajectory_dataset_file_sha256=_digest("trajectory-dataset-file"),
        trajectory_dataset_receipt_file_sha256=_digest(
            "trajectory-dataset-receipt-file"
        ),
        trajectory_dataset_receipt_payload_sha256=_digest(
            "trajectory-dataset-receipt-payload"
        ),
        trajectory_source_id="registered-synthetic-trajectory-v1",
        trajectory_root_cell_id="trajectory-cell-000001",
        trajectory_registered_authority_sha256=_digest(
            "trajectory-registered-authority"
        ),
        trajectory_registered_binding_sha256=_digest("trajectory-registered-binding"),
        trajectory_authority_sha256=_digest("trajectory-execution-authority"),
        trajectory_authority_file_sha256=_digest("trajectory-execution-authority-file"),
        trajectory_execution_manifest_path=(
            "results/trajectory/execution/execution_manifest.json"
        ),
        trajectory_execution_manifest_file_sha256=_digest(
            "trajectory-execution-manifest-file"
        ),
        trajectory_execution_manifest_payload_sha256=_digest(
            "trajectory-execution-manifest-payload"
        ),
        trajectory_execution_validation_sha256=_digest(
            "trajectory-execution-validation"
        ),
        trajectory_record_payload_sha256s_sha256=_digest("trajectory-record-payloads"),
        trajectory_status_counts_sha256=canonical_sha256(
            {
                "executed_status_counts": {"completed": 8},
                "not_applicable_count": 0,
            }
        ),
        trajectory_planned_run_count=8,
        trajectory_result_files_sha256=_digest("trajectory-result-files"),
        trajectory_result_file_count=30,
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
            configuration_id=(
                "candidate-01" if method == CANDIDATE else "registry-default"
            ),
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
        source_root=str(Path(binding.round_root) / "results/final/execution"),
        source_kind="final",
        evidence_scope="all",
        evaluator_source_sha256=_digest("downstream-evaluator"),
        source_manifest_path="execution_manifest.json",
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


def _trajectory_dataset(
    binding: EvaluatedRoundBinding,
) -> DatasetEvidenceBinding:
    return DatasetEvidenceBinding(
        dataset_id=binding.trajectory_dataset_id,
        path=str(
            Path(binding.round_root) / "results/trajectory/dataset/evaluator.h5ad"
        ),
        file_sha256=binding.trajectory_dataset_file_sha256,
        dataset_sha256=binding.trajectory_dataset_sha256,
        mechanism="synthetic_trajectory",
        biological_id="trajectory-draw-01",
        technical_view="exact_latent",
        method_input_sha256=_digest("trajectory-method-input"),
        dataset_qc_policy_sha256=_digest("trajectory-qc"),
        excluded_cell_count=0,
        excluded_cell_ids_sha256=_digest("trajectory-excluded"),
        retained_cell_count=2,
        retained_cell_ids_sha256=_digest("trajectory-cells"),
        retained_gene_count=2,
        observed_zero_count=1,
        retained_cell_ids=("trajectory-cell-000001", "trajectory-cell-000002"),
        gene_ids=("gene-1", "gene-2"),
        trajectory_root_cell_id=binding.trajectory_root_cell_id,
        trajectory_source_id=binding.trajectory_source_id,
        trajectory_authority_sha256=(binding.trajectory_registered_authority_sha256),
        trajectory_binding_sha256=(binding.trajectory_registered_binding_sha256),
    )


def _trajectory_plan(
    binding: EvaluatedRoundBinding,
) -> DownstreamEvidencePlan:
    dataset = _trajectory_dataset(binding)
    configurations = _configurations()
    by_method = {value.method_id: value for value in configurations}
    run_authority = (
        (CANDIDATE, 42),
        (CANDIDATE, 43),
        (CANDIDATE, 44),
        ("observed", None),
        ("dca", 42),
        ("dca", 43),
        ("dca", 44),
        ("dca", 45),
    )
    entries: list[DownstreamPlanEntry] = []
    for ordinal, (method, seed) in enumerate(run_authority, start=1):
        configuration = by_method[method]
        seed_id = "deterministic" if seed is None else f"seed-{seed}"
        run_id = f"trajectory-{method}-{seed_id}"
        entries.append(
            DownstreamPlanEntry(
                ordinal=ordinal,
                source_record_path=f"records/{ordinal:08d}.json",
                source_record_sha256=_digest(f"{run_id}-source-record"),
                run_id=run_id,
                method_id=method,
                dataset_id=dataset.dataset_id,
                source_dataset_sha256=dataset.dataset_sha256,
                mechanism=dataset.mechanism,
                biological_id=dataset.biological_id,
                technical_view=dataset.technical_view,
                model_seed=seed,
                configuration_id=configuration.configuration_id,
                configuration_sha256=configuration.configuration_sha256,
                configuration_kind=configuration.kind,
                method_artifact_sha256=_digest(f"{method}-trajectory-artifact"),
                method_input_sha256=dataset.method_input_sha256,
                retained_cell_ids_sha256=dataset.retained_cell_ids_sha256,
                status="completed",
                reason=None,
                evaluator_output_sha256=_digest(f"{run_id}-output"),
                evaluator_output_path=f"outputs/{run_id}.bin",
                evaluator_output_file_sha256=_digest(f"{run_id}-output-file"),
                evaluator_output_shape=(2, 2),
                evaluator_output_encoding="zlib_raw_f64_v1",
                evaluator_output_uncompressed_nbytes=32,
                evaluator_output_uncompressed_sha256=_digest(f"{run_id}-uncompressed"),
            )
        )
    provisional = DownstreamEvidencePlan(
        source_root=str(Path(binding.round_root) / "results/trajectory/execution"),
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        evaluator_source_sha256=_digest("downstream-evaluator"),
        source_manifest_path="execution_manifest.json",
        source_manifest_file_sha256=(binding.trajectory_execution_manifest_file_sha256),
        source_manifest_payload_sha256=(
            binding.trajectory_execution_manifest_payload_sha256
        ),
        source_plan_sha256=binding.trajectory_plan_sha256,
        source_input_hashes_sha256=_digest("trajectory-source-input-hashes"),
        source_statuses_sha256=canonical_sha256(
            [
                {
                    "run_id": entry.run_id,
                    "status": entry.status,
                    "reason": entry.reason,
                }
                for entry in entries
            ]
        ),
        source_plan_authority="independent",
        evaluated_round_binding=binding,
        development_revision_versions=(),
        development_sources=(),
        datasets=(dataset,),
        configurations=configurations,
        entries=tuple(entries),
        plan_sha256=_digest("temporary-trajectory-downstream-plan"),
    )
    return replace(provisional, plan_sha256=canonical_sha256(provisional.body()))


def _trajectory_record(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    *,
    value: float = 0.25,
) -> Mapping[str, object]:
    endpoint = {
        "endpoint": "trajectory_pseudotime_rank_loss",
        "status": "completed",
        "reason_code": None,
        "value": value,
    }
    if entry.status != "completed":
        endpoint = {
            "endpoint": "trajectory_pseudotime_rank_loss",
            "status": entry.status,
            "reason_code": "upstream_run_not_completed",
            "upstream_reason": entry.reason,
            "value": None,
        }
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
        "endpoints": [endpoint],
    }
    return _sealed(body, "record_sha256")


def _trajectory_manifest(
    plan: DownstreamEvidencePlan,
    *,
    values: tuple[float, ...] | None = None,
) -> DownstreamEvidenceManifest:
    selected_values = values or tuple(0.25 for _entry in plan.entries)
    assert len(selected_values) == len(plan.entries)
    records = tuple(
        _trajectory_record(plan, entry, value=value)
        for entry, value in zip(plan.entries, selected_values, strict=True)
    )
    references = [
        {
            "ordinal": entry.ordinal,
            "run_id": entry.run_id,
            "path": f"records/{entry.ordinal:08d}.json",
            "sha256": _digest(f"trajectory-record-file-{entry.ordinal}"),
            "record_sha256": records[entry.ordinal - 1]["record_sha256"],
        }
        for entry in plan.entries
    ]
    body = {
        "schema_version": 3,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": _digest("trajectory-downstream-plan-file"),
        "source_kind": plan.source_kind,
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "source_manifest_path": plan.source_manifest_path,
        "source_manifest_file_sha256": plan.source_manifest_file_sha256,
        "source_manifest_payload_sha256": plan.source_manifest_payload_sha256,
        "source_plan_sha256": plan.source_plan_sha256,
        "source_input_hashes_sha256": plan.source_input_hashes_sha256,
        "source_statuses_sha256": plan.source_statuses_sha256,
        "source_plan_authority": plan.source_plan_authority,
        "evaluated_round_binding_sha256": (plan.evaluated_round_binding.binding_sha256),
        "development_revision_versions": [],
        "development_sources": [],
        "planned_denominator_count": len(plan.entries),
        "recorded_denominator_count": len(records),
        "endpoint_row_count": len(records),
        "records": references,
    }
    payload = _sealed(body, "manifest_sha256")
    return DownstreamEvidenceManifest(
        plan_sha256=plan.plan_sha256,
        manifest_sha256=str(payload["manifest_sha256"]),
        planned_denominator_count=len(plan.entries),
        endpoint_row_count=len(records),
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
        "downstream_manifest_file_sha256": (plan.downstream_manifest_file_sha256),
        "downstream_manifest_payload_sha256": (plan.downstream_manifest_payload_sha256),
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
    trajectory_plan = _trajectory_plan(binding)
    trajectory_downstream = _trajectory_manifest(trajectory_plan)
    null_plan = _null_plan(plan, downstream)
    null_manifest = _null_manifest(null_plan)
    return _LoadedPublicationEvidence(
        primary_report=_primary_report(frozen, binding),
        frozen_method=frozen,
        downstream_plan=plan,
        downstream_manifest=downstream,
        trajectory_downstream_plan=trajectory_plan,
        trajectory_downstream_manifest=trajectory_downstream,
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


def _with_terminal_trajectory_status(
    loaded: _LoadedPublicationEvidence,
) -> _LoadedPublicationEvidence:
    entries = list(loaded.trajectory_downstream_plan.entries)
    entries[-1] = replace(
        entries[-1],
        status="resource_exceeded",
        reason="peak_gpu_memory_limit_exceeded",
        evaluator_output_sha256=None,
        evaluator_output_path=None,
        evaluator_output_file_sha256=None,
        evaluator_output_shape=None,
        evaluator_output_encoding=None,
        evaluator_output_uncompressed_nbytes=None,
        evaluator_output_uncompressed_sha256=None,
    )
    status_counts_sha256 = canonical_sha256(
        {
            "executed_status_counts": {
                "completed": 7,
                "resource_exceeded": 1,
            },
            "not_applicable_count": 0,
        }
    )
    old_binding = loaded.downstream_plan.evaluated_round_binding
    assert old_binding is not None
    binding = replace(
        old_binding,
        trajectory_status_counts_sha256=status_counts_sha256,
    )
    primary_provisional = replace(
        loaded.downstream_plan,
        evaluated_round_binding=binding,
        plan_sha256=_digest("temporary-primary-plan"),
    )
    primary = replace(
        primary_provisional,
        plan_sha256=canonical_sha256(primary_provisional.body()),
    )
    primary_manifest = _downstream_manifest(primary)
    trajectory_provisional = replace(
        loaded.trajectory_downstream_plan,
        evaluated_round_binding=binding,
        entries=tuple(entries),
        source_statuses_sha256=canonical_sha256(
            [
                {
                    "run_id": entry.run_id,
                    "status": entry.status,
                    "reason": entry.reason,
                }
                for entry in entries
            ]
        ),
        plan_sha256=_digest("temporary-trajectory-plan"),
    )
    trajectory = replace(
        trajectory_provisional,
        plan_sha256=canonical_sha256(trajectory_provisional.body()),
    )
    trajectory_manifest = _trajectory_manifest(trajectory)
    null_plan = _null_plan(primary, primary_manifest)
    null_manifest = _null_manifest(null_plan)
    return _LoadedPublicationEvidence(
        primary_report=_primary_report(loaded.frozen_method, binding),
        frozen_method=loaded.frozen_method,
        downstream_plan=primary,
        downstream_manifest=primary_manifest,
        trajectory_downstream_plan=trajectory,
        trajectory_downstream_manifest=trajectory_manifest,
        null_de_plan=null_plan,
        null_de_manifest=null_manifest,
        scaling_checkpoint=loaded.scaling_checkpoint,
    )


def _with_alternate_trajectory_counts(
    loaded: _LoadedPublicationEvidence,
    *,
    execution_run_count: int,
    receipt_result_file_count: int,
) -> _LoadedPublicationEvidence:
    entries = loaded.trajectory_downstream_plan.entries[:execution_run_count]
    old_binding = loaded.downstream_plan.evaluated_round_binding
    assert old_binding is not None
    binding = replace(
        old_binding,
        trajectory_evidence_sha256=_digest("alternate-trajectory-evidence"),
        trajectory_plan_sha256=_digest("alternate-trajectory-source-plan"),
        trajectory_execution_manifest_file_sha256=_digest(
            "alternate-trajectory-execution-manifest-file"
        ),
        trajectory_execution_manifest_payload_sha256=_digest(
            "alternate-trajectory-execution-manifest-payload"
        ),
        trajectory_execution_validation_sha256=_digest(
            "alternate-trajectory-execution-validation"
        ),
        trajectory_record_payload_sha256s_sha256=_digest(
            "alternate-trajectory-record-payloads"
        ),
        trajectory_status_counts_sha256=canonical_sha256(
            {
                "executed_status_counts": {"completed": execution_run_count},
                "not_applicable_count": 0,
            }
        ),
        trajectory_planned_run_count=execution_run_count,
        trajectory_result_files_sha256=_digest("alternate-trajectory-result-files"),
        trajectory_result_file_count=receipt_result_file_count,
    )
    primary_provisional = replace(
        loaded.downstream_plan,
        evaluated_round_binding=binding,
        plan_sha256=_digest("temporary-primary-plan"),
    )
    primary = replace(
        primary_provisional,
        plan_sha256=canonical_sha256(primary_provisional.body()),
    )
    primary_manifest = _downstream_manifest(primary)
    trajectory_provisional = replace(
        loaded.trajectory_downstream_plan,
        source_manifest_file_sha256=(binding.trajectory_execution_manifest_file_sha256),
        source_manifest_payload_sha256=(
            binding.trajectory_execution_manifest_payload_sha256
        ),
        source_plan_sha256=binding.trajectory_plan_sha256,
        source_statuses_sha256=canonical_sha256(
            [
                {
                    "run_id": entry.run_id,
                    "status": entry.status,
                    "reason": entry.reason,
                }
                for entry in entries
            ]
        ),
        evaluated_round_binding=binding,
        entries=entries,
        plan_sha256=_digest("temporary-trajectory-plan"),
    )
    trajectory = replace(
        trajectory_provisional,
        plan_sha256=canonical_sha256(trajectory_provisional.body()),
    )
    trajectory_manifest = _trajectory_manifest(trajectory)
    null_plan = _null_plan(primary, primary_manifest)
    null_manifest = _null_manifest(null_plan)
    return _LoadedPublicationEvidence(
        primary_report=_primary_report(
            loaded.frozen_method,
            binding,
            trajectory_sha256=binding.trajectory_evidence_sha256,
        ),
        frozen_method=loaded.frozen_method,
        downstream_plan=primary,
        downstream_manifest=primary_manifest,
        trajectory_downstream_plan=trajectory,
        trajectory_downstream_manifest=trajectory_manifest,
        null_de_plan=null_plan,
        null_de_manifest=null_manifest,
        scaling_checkpoint=loaded.scaling_checkpoint,
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
    persisted_trajectory_downstream_plan: DownstreamEvidencePlan | None = None,
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
    trajectory_downstream_directory = downstream_directory / "trajectory"
    null_directory = (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "null-de"
        / binding.round_id
        / binding.evaluation_receipt_payload_sha256
    ).absolute()

    def primary(
        observed_repository: Path, observed_round: Path
    ) -> Mapping[str, object]:
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

    def trajectory_downstream_plan(
        observed_repository: Path,
        observed_round: Path,
    ) -> DownstreamEvidencePlan:
        assert observed_repository == repository
        assert observed_round == round_dir
        return loaded.trajectory_downstream_plan

    def expected_downstream_directory(
        observed_plan: DownstreamEvidencePlan,
    ) -> Path:
        if observed_plan is loaded.downstream_plan:
            return downstream_directory
        assert observed_plan is loaded.trajectory_downstream_plan
        return trajectory_downstream_directory

    def persisted_plan(observed_directory: Path) -> DownstreamEvidencePlan:
        if observed_directory == downstream_directory:
            return persisted_downstream_plan or loaded.downstream_plan
        assert observed_directory == trajectory_downstream_directory
        return persisted_trajectory_downstream_plan or loaded.trajectory_downstream_plan

    def downstream_manifest(
        observed_directory: Path,
    ) -> DownstreamEvidenceManifest:
        if observed_directory == downstream_directory:
            return loaded.downstream_manifest
        assert observed_directory == trajectory_downstream_directory
        return loaded.trajectory_downstream_manifest

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
        "build_final_trajectory_downstream_evidence_plan",
        trajectory_downstream_plan,
        raising=False,
    )
    monkeypatch.setattr(
        publication_synthesis,
        "expected_final_downstream_output_directory",
        expected_downstream_directory,
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
    body = {key: value for key, value in synthesis.items() if key != "synthesis_sha256"}
    assert synthesis["synthesis_sha256"] == canonical_sha256(body)


def test_synthesis_loads_current_primary_and_separate_trajectory_evidence() -> None:
    loaded = _loaded()

    synthesis = _build_publication_synthesis(loaded)

    trajectory = synthesis["trajectory"]
    assert trajectory["role"] == "descriptive_only"
    assert trajectory["gate_influence"] == "none"
    assert trajectory["plan_sha256"] == loaded.trajectory_downstream_plan.plan_sha256
    assert trajectory["manifest_sha256"] == (
        loaded.trajectory_downstream_manifest.manifest_sha256
    )
    assert trajectory["planned_execution_run_count"] == 8
    assert trajectory["receipt_result_file_count"] == 30
    assert trajectory["external_endpoint_row_count"] == 8
    assert trajectory["run_status_counts"] == {"completed": 8}
    assert trajectory["endpoint_status_counts"] == {"completed": 8}


def test_trajectory_manifest_record_replacement_cannot_bypass_receipt_replay() -> None:
    loaded = _loaded()
    records = list(loaded.trajectory_downstream_manifest.records)
    first = dict(records[0])
    first.pop("record_sha256")
    endpoints = [dict(row) for row in first["endpoints"]]
    endpoints[0]["value"] = 0.99
    first["endpoints"] = endpoints
    records[0] = _sealed(first, "record_sha256")
    changed = replace(
        loaded,
        trajectory_downstream_manifest=replace(
            loaded.trajectory_downstream_manifest,
            records=tuple(records),
        ),
    )

    with pytest.raises(PublicationSynthesisError, match="manifest record binding"):
        _build_publication_synthesis(changed)


def test_trajectory_receipt_planned_count_must_match_rebuilt_plan() -> None:
    loaded = _loaded()
    old_binding = loaded.downstream_plan.evaluated_round_binding
    assert old_binding is not None
    binding = replace(old_binding, trajectory_planned_run_count=7)
    primary = replace(loaded.downstream_plan, evaluated_round_binding=binding)
    trajectory_provisional = replace(
        loaded.trajectory_downstream_plan,
        evaluated_round_binding=binding,
        plan_sha256=_digest("temporary-trajectory-plan"),
    )
    trajectory = replace(
        trajectory_provisional,
        plan_sha256=canonical_sha256(trajectory_provisional.body()),
    )
    changed = replace(
        loaded,
        downstream_plan=primary,
        trajectory_downstream_plan=trajectory,
    )

    with pytest.raises(PublicationSynthesisError, match="receipt denominator"):
        publication_synthesis._validate_trajectory_downstream_bindings(changed)


def test_coherent_alternate_trajectory_counts_are_receipt_derived() -> None:
    loaded = _with_alternate_trajectory_counts(
        _loaded(),
        execution_run_count=7,
        receipt_result_file_count=29,
    )

    synthesis = _build_publication_synthesis(loaded)

    assert synthesis["trajectory"]["planned_execution_run_count"] == 7
    assert synthesis["trajectory"]["receipt_result_file_count"] == 29
    assert synthesis["trajectory"]["external_endpoint_row_count"] == 7
    assert synthesis["trajectory"]["run_status_counts"] == {"completed": 7}
    assert synthesis["trajectory"]["endpoint_status_counts"] == {"completed": 7}


def test_trajectory_registered_dataset_authority_must_match_receipt() -> None:
    loaded = _loaded()
    dataset = replace(
        loaded.trajectory_downstream_plan.datasets[0],
        trajectory_binding_sha256=_digest("different-registered-binding"),
    )
    provisional = replace(
        loaded.trajectory_downstream_plan,
        datasets=(dataset,),
        plan_sha256=_digest("temporary-trajectory-plan"),
    )
    changed_plan = replace(
        provisional,
        plan_sha256=canonical_sha256(provisional.body()),
    )

    with pytest.raises(PublicationSynthesisError, match="registered dataset"):
        publication_synthesis._validate_trajectory_downstream_bindings(
            replace(loaded, trajectory_downstream_plan=changed_plan)
        )


def test_trajectory_binding_must_equal_primary_evaluated_receipt() -> None:
    loaded = _loaded()
    binding = loaded.trajectory_downstream_plan.evaluated_round_binding
    assert binding is not None
    changed_binding = replace(
        binding,
        trajectory_result_files_sha256=_digest("different-result-inventory"),
    )
    provisional = replace(
        loaded.trajectory_downstream_plan,
        evaluated_round_binding=changed_binding,
        plan_sha256=_digest("temporary-trajectory-plan"),
    )
    changed_plan = replace(
        provisional,
        plan_sha256=canonical_sha256(provisional.body()),
    )

    with pytest.raises(PublicationSynthesisError, match="evaluated-round binding"):
        publication_synthesis._validate_trajectory_downstream_bindings(
            replace(loaded, trajectory_downstream_plan=changed_plan)
        )


def test_trajectory_manifest_requires_one_exact_endpoint_per_execution_run() -> None:
    loaded = _loaded()
    records = list(loaded.trajectory_downstream_manifest.records)
    first = dict(records[0])
    first.pop("record_sha256")
    endpoints = [dict(row) for row in first["endpoints"]]
    endpoints[0]["endpoint"] = "trajectory_invented_endpoint"
    first["endpoints"] = endpoints
    records[0] = _sealed(first, "record_sha256")
    payload_body = dict(loaded.trajectory_downstream_manifest.payload)
    payload_body.pop("manifest_sha256")
    references = [dict(row) for row in payload_body["records"]]
    references[0]["record_sha256"] = records[0]["record_sha256"]
    payload_body["records"] = references
    payload = _sealed(payload_body, "manifest_sha256")
    changed = replace(
        loaded,
        trajectory_downstream_manifest=replace(
            loaded.trajectory_downstream_manifest,
            records=tuple(records),
            manifest_sha256=str(payload["manifest_sha256"]),
            payload=payload,
        ),
    )

    with pytest.raises(PublicationSynthesisError, match="record denominator"):
        _build_publication_synthesis(changed)


def test_trajectory_external_endpoint_denominator_is_not_receipt_file_count() -> None:
    loaded = _loaded()
    changed = replace(
        loaded,
        trajectory_downstream_manifest=replace(
            loaded.trajectory_downstream_manifest,
            endpoint_row_count=7,
        ),
    )

    with pytest.raises(PublicationSynthesisError, match="manifest denominator"):
        _build_publication_synthesis(changed)


@pytest.mark.parametrize("case", ["planned_count", "record_count"])
def test_trajectory_manifest_run_denominators_must_match_rebuilt_plan(
    case: str,
) -> None:
    loaded = _loaded()
    manifest = loaded.trajectory_downstream_manifest
    changed_manifest = (
        replace(manifest, planned_denominator_count=7)
        if case == "planned_count"
        else replace(manifest, records=manifest.records[:-1])
    )

    with pytest.raises(PublicationSynthesisError, match="manifest denominator"):
        _build_publication_synthesis(
            replace(
                loaded,
                trajectory_downstream_manifest=changed_manifest,
            )
        )


def test_trajectory_digest_in_primary_report_must_equal_evaluated_receipt() -> None:
    loaded = _with_report_input(
        _loaded(),
        "trajectory_evidence_sha256",
        _digest("different-trajectory-evidence"),
    )

    with pytest.raises(PublicationSynthesisError, match="trajectory evidence binding"):
        _build_publication_synthesis(loaded)


def test_trajectory_values_and_terminal_statuses_are_gate_inert() -> None:
    loaded = _loaded()
    base = _build_publication_synthesis(loaded)
    value_changed = replace(
        loaded,
        trajectory_downstream_manifest=_trajectory_manifest(
            loaded.trajectory_downstream_plan,
            values=tuple(index / 10.0 for index in range(8)),
        ),
    )
    status_changed = _with_terminal_trajectory_status(loaded)

    value_synthesis = _build_publication_synthesis(value_changed)
    status_synthesis = _build_publication_synthesis(status_changed)

    for changed in (value_synthesis, status_synthesis):
        assert canonical_sha256(changed["gates"]) == canonical_sha256(base["gates"])
        assert canonical_sha256(changed["claim_permissions"]) == canonical_sha256(
            base["claim_permissions"]
        )
    assert status_synthesis["trajectory"]["run_status_counts"] == {
        "completed": 7,
        "resource_exceeded": 1,
    }
    assert status_synthesis["trajectory"]["run_reason_counts"] == {
        "peak_gpu_memory_limit_exceeded": 1
    }
    assert status_synthesis["trajectory"]["endpoint_reason_counts"] == {
        "upstream_run_not_completed": 1
    }


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
            index for index, row in enumerate(records) if row["method_id"] == CANDIDATE
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

    assert synthesis["trajectory"]["evidence_sha256"] == _digest("trajectory-evidence")
    assert synthesis["trajectory"]["gate_influence"] == "none"
    assert synthesis["gates"]["competitive"]["status"] == competitive_status
    assert synthesis["claim_permissions"]["competitive"] is False


def test_superiority_uses_deterministic_strongest_comparator_and_discloses_ties() -> (
    None
):
    loaded = _loaded()
    report = dict(loaded.primary_report)
    reconstruction = dict(report["reconstruction_claim_gate"])
    reconstruction["draw_collapsed_method_summaries"] = [
        ({**dict(row), "median": 1.1} if row["method_id"] == "observed" else dict(row))
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
        ({**dict(row), "median": 1.1} if row["method_id"] == "observed" else dict(row))
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
        if not (row["comparator_method_id"] == "dca" and row["metric"] == "mse_dropout")
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
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


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
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


def test_production_rejects_rebuilt_and_persisted_trajectory_plan_difference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    persisted = replace(
        loaded.trajectory_downstream_plan,
        plan_sha256=_digest("different-persisted-trajectory-plan"),
    )
    _install_authoritative_loaders(
        monkeypatch,
        loaded,
        persisted_trajectory_downstream_plan=persisted,
    )

    with pytest.raises(
        PublicationSynthesisError,
        match="persisted trajectory downstream plan",
    ):
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


def test_production_rejects_missing_trajectory_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    _install_authoritative_loaders(monkeypatch, loaded)
    binding = loaded.downstream_plan.evaluated_round_binding
    assert binding is not None
    primary_directory = expected_final_downstream_output_directory(
        loaded.downstream_plan
    )

    def manifest(directory: Path) -> DownstreamEvidenceManifest:
        if directory == primary_directory:
            return loaded.downstream_manifest
        raise DownstreamEvidenceError("trajectory archive is absent")

    monkeypatch.setattr(
        publication_synthesis,
        "load_downstream_evidence_manifest",
        manifest,
    )

    with pytest.raises(
        PublicationSynthesisError,
        match="trajectory downstream manifest replay failed",
    ):
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


def test_production_rejects_trajectory_external_namespace_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _loaded()
    _install_authoritative_loaders(monkeypatch, loaded)
    primary_directory = expected_final_downstream_output_directory(
        loaded.downstream_plan
    )
    monkeypatch.setattr(
        publication_synthesis,
        "expected_final_downstream_output_directory",
        lambda _plan: primary_directory,
    )

    with pytest.raises(PublicationSynthesisError, match="namespace differs"):
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


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
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


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
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


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
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


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
        generate_publication_synthesis(Path("/study"), Path("/study/final/round-01"))


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
