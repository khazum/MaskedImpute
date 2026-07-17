"""Fail-closed publication claim permissions from frozen final evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re

from .downstream_evidence import (
    DownstreamEvidenceError,
    DownstreamEvidenceManifest,
    DownstreamEvidencePlan,
    build_final_downstream_evidence_plan,
    load_downstream_evidence_manifest,
    load_downstream_evidence_plan,
)
from .final_analysis import (
    FinalAnalysisContractError,
    generate_final_analysis,
)
from .final_null_de import (
    FinalNullDEError,
    FinalNullDEManifest,
    FinalNullDEPlan,
    build_final_null_de_plan,
    expected_final_null_de_output_directory,
    load_final_null_de_manifest,
)
from .protocol import canonical_sha256
from .publication_freeze import (
    PublicationFreezeError,
    validate_frozen_method,
)
from .scaling import (
    ScalingCheckpoint,
    ScalingContractError,
    load_publication_scaling_evidence,
    scaling_checkpoint_payload,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NULL_DE_MAXIMUM_FPR = 0.06
_NULL_DE_MAXIMUM_ABOVE_OBSERVED = 0.01
_SCIENTIFIC_GATE_STATUSES = frozenset({"passed", "failed", "unavailable"})
_PAIRWISE_EXCLUSION_NAMES = frozenset(
    {
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
    }
)
_FORBIDDEN_PAIRWISE_EXCLUSIONS = frozenset(
    {
        "bootstrap_empty_replicates",
        "bootstrap_nonrepresentable_effect_pairs",
        "bootstrap_zero_comparator_pairs",
        "duplicate_rows",
        "nonrepresentable_effect_pairs",
        "zero_comparator_pairs",
    }
)


class PublicationSynthesisError(RuntimeError):
    """Raised when publication evidence is not exact frozen-final authority."""


@dataclass(frozen=True, slots=True)
class _LoadedPublicationEvidence:
    primary_report: Mapping[str, object]
    frozen_method: Mapping[str, object]
    downstream_plan: DownstreamEvidencePlan
    downstream_manifest: DownstreamEvidenceManifest
    null_de_plan: FinalNullDEPlan
    null_de_manifest: FinalNullDEManifest
    scaling_checkpoint: ScalingCheckpoint


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicationSynthesisError(f"{name} is not a SHA-256 digest")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationSynthesisError(f"{name} is not a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationSynthesisError(f"{name} is not finite numerical evidence")
    result = float(value)
    if not math.isfinite(result):
        raise PublicationSynthesisError(f"{name} is not finite numerical evidence")
    return result


def _canonical_mapping(
    value: object,
    *,
    seal: str,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PublicationSynthesisError(f"{name} is not an object")
    result = dict(value)
    observed = _digest(result.get(seal), f"{name} checksum")
    body = {key: nested for key, nested in result.items() if key != seal}
    if canonical_sha256(body) != observed:
        raise PublicationSynthesisError(f"{name} checksum differs")
    return result


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PublicationSynthesisError(f"{name} is not an object")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicationSynthesisError(f"{name} is not a sequence")
    return value


def _validated_primary_report(
    loaded: _LoadedPublicationEvidence,
) -> tuple[dict[str, object], Mapping[str, object], str]:
    report = _canonical_mapping(
        loaded.primary_report,
        seal="analysis_sha256",
        name="primary final analysis",
    )
    if report.get("schema_version") != 1 or report.get("status") != "completed":
        raise PublicationSynthesisError("primary final analysis is not completed")
    candidate = _text(report.get("candidate_method_id"), "candidate method")
    inputs = _mapping(report.get("input_bindings"), "primary input bindings")
    return report, inputs, candidate


def _validate_freeze_prerequisite(
    frozen_value: Mapping[str, object],
    *,
    report_inputs: Mapping[str, object],
    candidate: str,
    reconstruction: Mapping[str, object],
) -> dict[str, object]:
    frozen = _canonical_mapping(
        frozen_value,
        seal="payload_sha256",
        name="frozen method",
    )
    frozen_sha256 = _digest(frozen.get("payload_sha256"), "frozen method checksum")
    if (
        frozen.get("candidate_method_id") != candidate
        or report_inputs.get("frozen_method_payload_sha256") != frozen_sha256
    ):
        raise PublicationSynthesisError(
            "primary candidate or frozen-method binding differs"
        )
    configuration_id = _text(
        frozen.get("selected_configuration_id"),
        "frozen selected configuration",
    )
    selected_version = _text(
        frozen.get("selected_version"), "frozen selected version"
    )
    raw_table = _sequence(
        frozen.get("selection_gate_table"), "frozen selection gate table"
    )
    table = [
        row
        for row in raw_table
        if isinstance(row, Mapping)
        and row.get("configuration_id") == configuration_id
    ]
    selected = frozen.get("selected_assessment")
    if (
        len(table) != 1
        or not isinstance(selected, Mapping)
        or dict(table[0]) != dict(selected)
        or selected.get("version") != selected_version
        or selected.get("eligible") is not True
        or selected.get("efficacy_pass") is not True
        or selected.get("safety_pass") is not True
    ):
        raise PublicationSynthesisError(
            "frozen selected assessment does not pass every freeze prerequisite"
        )
    required_value = _sequence(
        frozen.get("required_comparator_ids"), "frozen required comparators"
    )
    required = tuple(required_value)
    if (
        not required
        or any(not isinstance(value, str) or not value for value in required)
        or len(set(required)) != len(required)
        or list(required) != reconstruction.get("required_comparator_ids")
    ):
        raise PublicationSynthesisError(
            "frozen required comparator denominator differs"
        )
    return {
        "status": "passed",
        "candidate_method_id": candidate,
        "selected_configuration_id": configuration_id,
        "selected_version": selected_version,
        "gate_flags": {
            "efficacy_pass": True,
            "eligible": True,
            "safety_pass": True,
        },
        "frozen_method_payload_sha256": frozen_sha256,
        "numerical_use": "freeze_validity_only",
    }


def _validate_manifest_payload(
    payload: Mapping[str, object],
    *,
    expected_sha256: str,
    name: str,
) -> dict[str, object]:
    observed = _canonical_mapping(payload, seal="manifest_sha256", name=name)
    if observed["manifest_sha256"] != expected_sha256:
        raise PublicationSynthesisError(f"{name} binding differs")
    return observed


def _validate_downstream_bindings(
    loaded: _LoadedPublicationEvidence,
    *,
    report_inputs: Mapping[str, object],
) -> None:
    plan = loaded.downstream_plan
    manifest = loaded.downstream_manifest
    if not isinstance(plan, DownstreamEvidencePlan) or not isinstance(
        manifest, DownstreamEvidenceManifest
    ):
        raise PublicationSynthesisError("downstream loader result type differs")
    binding = plan.evaluated_round_binding
    if (
        plan.source_kind != "final"
        or plan.evidence_scope != "all"
        or binding is None
    ):
        raise PublicationSynthesisError(
            "downstream source must be complete frozen final evidence"
        )
    if canonical_sha256(plan.body()) != plan.plan_sha256:
        raise PublicationSynthesisError("downstream plan checksum differs")
    expected_plan_fields = {
        "source_root": binding.repository_root,
        "source_manifest_path": binding.final_execution_manifest_path,
        "source_manifest_file_sha256": (
            binding.final_execution_manifest_file_sha256
        ),
        "source_manifest_payload_sha256": (
            binding.final_execution_manifest_payload_sha256
        ),
        "source_plan_sha256": binding.final_plan_sha256,
    }
    if any(getattr(plan, key) != value for key, value in expected_plan_fields.items()):
        raise PublicationSynthesisError("downstream evaluated source binding differs")
    expected_report_fields = {
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
    }
    if any(report_inputs.get(key) != value for key, value in expected_report_fields.items()):
        raise PublicationSynthesisError("primary evaluated-round binding differs")
    if report_inputs.get("planned_run_count") != len(plan.entries):
        raise PublicationSynthesisError("primary final denominator differs")
    if (
        manifest.plan_sha256 != plan.plan_sha256
        or manifest.planned_denominator_count != len(plan.entries)
        or len(manifest.records) != len(plan.entries)
    ):
        raise PublicationSynthesisError("downstream manifest denominator differs")
    payload = _validate_manifest_payload(
        manifest.payload,
        expected_sha256=manifest.manifest_sha256,
        name="downstream manifest",
    )
    expected_payload = {
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "source_kind": "final",
        "planned_denominator_count": len(plan.entries),
        "recorded_denominator_count": len(plan.entries),
        "endpoint_row_count": manifest.endpoint_row_count,
        "evaluated_round_binding_sha256": binding.binding_sha256,
    }
    if any(payload.get(key) != value for key, value in expected_payload.items()):
        raise PublicationSynthesisError("downstream manifest binding differs")
    endpoint_count = 0
    for entry, record in zip(plan.entries, manifest.records, strict=True):
        endpoints = record.get("endpoints")
        if (
            record.get("ordinal") != entry.ordinal
            or record.get("run_id") != entry.run_id
            or record.get("runner_method_id") != entry.method_id
            or record.get("run_status") != entry.status
            or not isinstance(endpoints, list)
        ):
            raise PublicationSynthesisError("downstream record denominator differs")
        endpoint_count += len(endpoints)
    if endpoint_count != manifest.endpoint_row_count:
        raise PublicationSynthesisError("downstream endpoint denominator differs")


def _validate_null_de_bindings(loaded: _LoadedPublicationEvidence) -> None:
    plan = loaded.null_de_plan
    manifest = loaded.null_de_manifest
    source = loaded.downstream_plan
    downstream = loaded.downstream_manifest
    if not isinstance(plan, FinalNullDEPlan) or not isinstance(
        manifest, FinalNullDEManifest
    ):
        raise PublicationSynthesisError("final null-DE loader result type differs")
    binding = source.evaluated_round_binding
    if binding is None:
        raise PublicationSynthesisError("final null-DE receipt binding is absent")
    if (
        plan.source_plan.to_dict() != source.to_dict()
        or plan.downstream_manifest_payload_sha256 != downstream.manifest_sha256
        or plan.evaluator_source_sha256 != source.evaluator_source_sha256
        or canonical_sha256(plan.body()) != plan.plan_sha256
    ):
        raise PublicationSynthesisError("final null-DE source binding differs")
    expected_directory = (
        Path(binding.repository_root).parent
        / f"{Path(binding.repository_root).name}-final-analysis"
        / "downstream"
        / binding.round_id
        / binding.evaluation_receipt_payload_sha256
    ).absolute()
    if Path(plan.downstream_directory) != expected_directory:
        raise PublicationSynthesisError("final null-DE downstream location differs")
    if (
        manifest.plan_sha256 != plan.plan_sha256
        or manifest.planned_denominator_count != len(source.entries)
    ):
        raise PublicationSynthesisError("final null-DE manifest binding differs")
    payload = _validate_manifest_payload(
        manifest.payload,
        expected_sha256=manifest.manifest_sha256,
        name="final null-DE manifest",
    )
    expected_payload = {
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "source_plan_sha256": source.plan_sha256,
        "evaluated_round_binding_sha256": binding.binding_sha256,
        "downstream_manifest_file_sha256": (
            plan.downstream_manifest_file_sha256
        ),
        "downstream_manifest_payload_sha256": downstream.manifest_sha256,
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "planned_denominator_count": len(source.entries),
        "recorded_denominator_count": len(manifest.records),
    }
    if any(payload.get(key) != value for key, value in expected_payload.items()):
        raise PublicationSynthesisError("final null-DE manifest source differs")
    for record in manifest.records:
        _canonical_mapping(
            record,
            seal="record_sha256",
            name="final null-DE record",
        )


def _validate_scaling_binding(
    loaded: _LoadedPublicationEvidence,
    *,
    frozen_sha256: str,
) -> None:
    checkpoint = loaded.scaling_checkpoint
    if not isinstance(checkpoint, ScalingCheckpoint):
        raise PublicationSynthesisError("scaling loader result type differs")
    binding = loaded.downstream_plan.evaluated_round_binding
    if binding is None:
        raise PublicationSynthesisError("scaling receipt binding is absent")
    try:
        payload = scaling_checkpoint_payload(checkpoint)
    except (TypeError, ValueError, RuntimeError) as error:
        raise PublicationSynthesisError("scaling checkpoint is invalid") from error
    if (
        checkpoint.status != "completed"
        or len(checkpoint.records) != checkpoint.planned_run_count
        or checkpoint.plan_sha256 != binding.scaling_plan_sha256
        or canonical_sha256(payload) != binding.scaling_checkpoint_payload_sha256
        or checkpoint.input_hashes.get("frozen_method_sha256") != frozen_sha256
    ):
        raise PublicationSynthesisError("scaling evaluated-round binding differs")


def _validate_loaded_bindings(
    loaded: _LoadedPublicationEvidence,
) -> tuple[
    dict[str, object],
    Mapping[str, object],
    str,
    Mapping[str, object],
    dict[str, object],
    str,
]:
    if not isinstance(loaded, _LoadedPublicationEvidence):
        raise TypeError("loaded must be _LoadedPublicationEvidence")
    report, inputs, candidate = _validated_primary_report(loaded)
    reconstruction = _mapping(
        report.get("reconstruction_claim_gate"), "reconstruction claim gate"
    )
    reconstruction_status = reconstruction.get("status")
    if (
        reconstruction_status not in _SCIENTIFIC_GATE_STATUSES
        or reconstruction.get("candidate_method_id") != candidate
    ):
        raise PublicationSynthesisError("reconstruction claim gate is invalid")
    freeze = _validate_freeze_prerequisite(
        loaded.frozen_method,
        report_inputs=inputs,
        candidate=candidate,
        reconstruction=reconstruction,
    )
    _validate_downstream_bindings(loaded, report_inputs=inputs)
    _validate_null_de_bindings(loaded)
    frozen_sha256 = str(freeze["frozen_method_payload_sha256"])
    _validate_scaling_binding(loaded, frozen_sha256=frozen_sha256)
    trajectory_sha256 = _digest(
        inputs.get("trajectory_evidence_sha256"),
        "trajectory evidence checksum",
    )
    return report, inputs, candidate, reconstruction, freeze, trajectory_sha256


def _null_de_unavailable(
    *,
    candidate: str,
    expected_count: int,
    recorded_count: int,
) -> dict[str, object]:
    return {
        "status": "unavailable",
        "reason": "incomplete_final_null_de_denominator",
        "candidate_method_id": candidate,
        "comparator_method_id": "observed",
        "limits": {
            "maximum_fpr": _NULL_DE_MAXIMUM_FPR,
            "maximum_above_observed": _NULL_DE_MAXIMUM_ABOVE_OBSERVED,
        },
        "maximum_fpr": None,
        "maximum_above_observed": None,
        "failed_conditions": [],
        "n_biological_draws": 0,
        "denominator": {
            "status": "incomplete",
            "expected_record_count": expected_count,
            "recorded_record_count": recorded_count,
            "dataset_view_count": 0,
            "collapse": "seed_mean_then_paired_view_mean",
        },
        "biological_draws": [],
    }


def _null_identity(value: Mapping[str, object]) -> tuple[object, ...]:
    return (
        value.get("method_id"),
        value.get("mechanism"),
        value.get("biological_id"),
        value.get("technical_view"),
        value.get("dataset_id"),
        value.get("model_seed"),
    )


def _view_identity(value: Mapping[str, object]) -> tuple[str, str, str, str] | None:
    parts = (
        value.get("mechanism"),
        value.get("biological_id"),
        value.get("technical_view"),
        value.get("dataset_id"),
    )
    if any(not isinstance(part, str) or not part for part in parts):
        return None
    return parts  # type: ignore[return-value]


def _build_final_null_de_gate(
    loaded: _LoadedPublicationEvidence,
) -> dict[str, object]:
    report = _mapping(loaded.primary_report, "primary final analysis")
    candidate = _text(report.get("candidate_method_id"), "candidate method")
    methods = frozenset({candidate, "observed"})
    expected_entries = tuple(
        entry for entry in loaded.null_de_plan.source_plan.entries if entry.method_id in methods
    )
    records = tuple(
        record for record in loaded.null_de_manifest.records if record.get("method_id") in methods
    )
    def unavailable() -> dict[str, object]:
        return _null_de_unavailable(
            candidate=candidate,
            expected_count=len(expected_entries),
            recorded_count=len(records),
        )
    expected: dict[tuple[object, ...], object] = {}
    for entry in expected_entries:
        identity = (
            entry.method_id,
            entry.mechanism,
            entry.biological_id,
            entry.technical_view,
            entry.dataset_id,
            entry.model_seed,
        )
        if identity in expected:
            return unavailable()
        expected[identity] = entry
    observed: dict[tuple[object, ...], Mapping[str, object]] = {}
    for record in records:
        identity = _null_identity(record)
        if identity in observed:
            return unavailable()
        observed[identity] = record
    if not expected or set(observed) != set(expected):
        return unavailable()

    view_values: dict[tuple[str, str, str, str, str], list[float]] = {}
    view_sets: dict[str, set[tuple[str, str, str, str]]] = {
        candidate: set(),
        "observed": set(),
    }
    for identity, record in observed.items():
        method = identity[0]
        if not isinstance(method, str):
            return unavailable()
        view = _view_identity(record)
        if view is None or record.get("status") != "completed":
            return unavailable()
        try:
            fpr = _finite(record.get("fpr"), "final null-DE FPR")
        except PublicationSynthesisError:
            return unavailable()
        if not 0.0 <= fpr <= 1.0:
            return unavailable()
        view_sets[method].add(view)
        view_values.setdefault((method, *view), []).append(fpr)
    if not view_sets[candidate] or view_sets[candidate] != view_sets["observed"]:
        return unavailable()

    collapsed_views = {
        key: sum(values) / len(values) for key, values in view_values.items()
    }
    by_draw: dict[tuple[str, str, str], dict[str, float]] = {}
    for (method, mechanism, biological_id, technical_view, _dataset_id), value in (
        collapsed_views.items()
    ):
        key = (method, mechanism, biological_id)
        views = by_draw.setdefault(key, {})
        if technical_view in views:
            return unavailable()
        views[technical_view] = value
    draw_ids = {(view[0], view[1]) for view in view_sets[candidate]}
    if draw_ids != {(view[0], view[1]) for view in view_sets["observed"]}:
        return unavailable()
    draws: list[dict[str, object]] = []
    for mechanism, biological_id in sorted(draw_ids):
        candidate_views = by_draw.get((candidate, mechanism, biological_id), {})
        observed_views = by_draw.get(("observed", mechanism, biological_id), {})
        if (
            set(candidate_views) != {"moderate", "severe"}
            or set(observed_views) != {"moderate", "severe"}
        ):
            return unavailable()
        candidate_fpr = sum(candidate_views.values()) / 2.0
        observed_fpr = sum(observed_views.values()) / 2.0
        draws.append(
            {
                "mechanism": mechanism,
                "biological_id": biological_id,
                "candidate_fpr": candidate_fpr,
                "observed_fpr": observed_fpr,
                "candidate_minus_observed": candidate_fpr - observed_fpr,
            }
        )
    if not draws:
        return unavailable()
    maximum_fpr = max(float(row["candidate_fpr"]) for row in draws)
    maximum_above = max(
        float(row["candidate_minus_observed"]) for row in draws
    )
    failed_conditions: list[str] = []
    if maximum_fpr > _NULL_DE_MAXIMUM_FPR:
        failed_conditions.append("maximum_fpr_exceeds_limit")
    if maximum_above > _NULL_DE_MAXIMUM_ABOVE_OBSERVED:
        failed_conditions.append("maximum_above_observed_exceeds_limit")
    status = "failed" if failed_conditions else "passed"
    return {
        "status": status,
        "reason": "prespecified_final_null_de_limit_failed" if failed_conditions else None,
        "candidate_method_id": candidate,
        "comparator_method_id": "observed",
        "limits": {
            "maximum_fpr": _NULL_DE_MAXIMUM_FPR,
            "maximum_above_observed": _NULL_DE_MAXIMUM_ABOVE_OBSERVED,
        },
        "maximum_fpr": maximum_fpr,
        "maximum_above_observed": maximum_above,
        "failed_conditions": failed_conditions,
        "n_biological_draws": len(draws),
        "denominator": {
            "status": "complete",
            "expected_record_count": len(expected_entries),
            "recorded_record_count": len(records),
            "dataset_view_count": len(view_sets[candidate]),
            "collapse": "seed_mean_then_paired_view_mean",
        },
        "biological_draws": draws,
    }


def _downstream_summary(loaded: _LoadedPublicationEvidence) -> dict[str, object]:
    run_statuses: Counter[str] = Counter()
    endpoint_statuses: Counter[str] = Counter()
    endpoint_reasons: Counter[str] = Counter()
    for record in loaded.downstream_manifest.records:
        run_status = record.get("run_status")
        if isinstance(run_status, str):
            run_statuses[run_status] += 1
        endpoints = record.get("endpoints")
        if not isinstance(endpoints, list):
            continue
        for row in endpoints:
            if not isinstance(row, Mapping):
                continue
            status = row.get("status")
            reason = row.get("reason_code")
            if isinstance(status, str):
                endpoint_statuses[status] += 1
            if isinstance(reason, str):
                endpoint_reasons[reason] += 1
    return {
        "status": "completed",
        "numerical_gate_status": "not_prespecified",
        "plan_sha256": loaded.downstream_plan.plan_sha256,
        "manifest_sha256": loaded.downstream_manifest.manifest_sha256,
        "planned_denominator_count": loaded.downstream_manifest.planned_denominator_count,
        "endpoint_row_count": loaded.downstream_manifest.endpoint_row_count,
        "run_status_counts": dict(sorted(run_statuses.items())),
        "endpoint_status_counts": dict(sorted(endpoint_statuses.items())),
        "endpoint_reason_counts": dict(sorted(endpoint_reasons.items())),
    }


def _scaling_summary(loaded: _LoadedPublicationEvidence) -> dict[str, object]:
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for record in loaded.scaling_checkpoint.records:
        run = record.get("run")
        source = run if isinstance(run, Mapping) else record
        status = source.get("status")
        reason = source.get("reason")
        if isinstance(status, str):
            statuses[status] += 1
        if isinstance(reason, str):
            reasons[reason] += 1
    return {
        "status": loaded.scaling_checkpoint.status,
        "numerical_gate_status": "not_prespecified",
        "plan_sha256": loaded.scaling_checkpoint.plan_sha256,
        "checkpoint_sha256": loaded.scaling_checkpoint.checkpoint_sha256,
        "planned_run_count": loaded.scaling_checkpoint.planned_run_count,
        "recorded_run_count": len(loaded.scaling_checkpoint.records),
        "dataset_count": len(loaded.scaling_checkpoint.datasets),
        "run_status_counts": dict(sorted(statuses.items())),
        "run_reason_counts": dict(sorted(reasons.items())),
    }


def _competitive_gate(
    reconstruction: Mapping[str, object],
    null_de: Mapping[str, object],
) -> dict[str, object]:
    component_statuses = {
        "reconstruction": reconstruction.get("status"),
        "final_null_de": null_de.get("status"),
    }
    if any(value not in _SCIENTIFIC_GATE_STATUSES for value in component_statuses.values()):
        raise PublicationSynthesisError("scientific gate status is invalid")
    if "failed" in component_statuses.values():
        status = "failed"
        reason = "prespecified_scientific_gate_failed"
    elif "unavailable" in component_statuses.values():
        status = "unavailable"
        reason = "prespecified_scientific_gate_unavailable"
    else:
        status = "passed"
        reason = None
    return {
        "status": status,
        "reason": reason,
        "component_statuses": component_statuses,
    }


def _primary_metrics(report: Mapping[str, object]) -> tuple[str, ...]:
    policy = _mapping(report.get("analysis_policy"), "primary analysis policy")
    family = _mapping(
        policy.get("declared_metric_family"), "declared primary metric family"
    )
    values = _sequence(family.get("metrics"), "protocol primary metrics")
    metrics = tuple(values)
    if (
        family.get("id") != "protocol_primary_metrics"
        or not metrics
        or any(not isinstance(metric, str) or not metric for metric in metrics)
        or len(set(metrics)) != len(metrics)
    ):
        raise PublicationSynthesisError("protocol primary metric family differs")
    direction = _mapping(
        policy.get("metric_direction_contract"),
        "primary metric direction contract",
    )
    direction_metrics = _sequence(
        direction.get("metrics"), "lower-direction metric authority"
    )
    if (
        direction.get("status") != "validated"
        or direction.get("favorable_direction") != "lower"
        or any(metric not in direction_metrics for metric in metrics)
    ):
        raise PublicationSynthesisError(
            "protocol primary metric lacks lower-direction authority"
        )
    return metrics  # type: ignore[return-value]


def _recomputed_strongest_comparators(
    reconstruction: Mapping[str, object],
    *,
    metrics: Sequence[str],
    required_ids: Sequence[str],
) -> dict[str, dict[str, object] | None]:
    raw_summaries = _sequence(
        reconstruction.get("draw_collapsed_method_summaries"),
        "draw-collapsed method summaries",
    )
    lookup: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in raw_summaries:
        if not isinstance(row, Mapping):
            raise PublicationSynthesisError(
                "draw-collapsed method summary is invalid"
            )
        method = row.get("method_id")
        metric = row.get("metric")
        if method not in required_ids or metric not in metrics:
            continue
        assert isinstance(method, str) and isinstance(metric, str)
        key = (method, metric)
        if key in lookup:
            raise PublicationSynthesisError(
                "draw-collapsed method summaries are duplicated"
            )
        lookup[key] = row

    result: dict[str, dict[str, object] | None] = {}
    for metric in metrics:
        candidates: list[tuple[float, str, int]] = []
        for method in required_ids:
            row = lookup.get((method, metric))
            if row is None or row.get("status") != "complete":
                candidates = []
                break
            median = _finite(row.get("median"), "strongest comparator median")
            count = row.get("n_biological_draws")
            if type(count) is not int or count <= 0:
                raise PublicationSynthesisError(
                    "strongest comparator draw denominator is invalid"
                )
            candidates.append((median, method, count))
        if not candidates:
            result[metric] = None
            continue
        best_median = min(value[0] for value in candidates)
        tied = sorted(
            method for median, method, _count in candidates if median == best_median
        )
        selected = tied[0]
        count = next(
            observed_count
            for median, method, observed_count in candidates
            if median == best_median and method == selected
        )
        result[metric] = {
            "median": best_median,
            "method_id": selected,
            "n_biological_draws": count,
            "tied_method_ids": tied,
        }
    return result


def _superiority_permissions(
    report: Mapping[str, object],
    reconstruction: Mapping[str, object],
    competitive: Mapping[str, object],
) -> list[dict[str, object]]:
    metrics = _primary_metrics(report)
    strongest_values = _sequence(
        reconstruction.get("strongest_applicable_comparators"),
        "strongest comparator rows",
    )
    strongest: dict[str, Mapping[str, object]] = {}
    for row in strongest_values:
        if not isinstance(row, Mapping) or not isinstance(row.get("metric"), str):
            raise PublicationSynthesisError("strongest comparator row is invalid")
        metric = str(row["metric"])
        if metric in strongest:
            raise PublicationSynthesisError("strongest comparator rows are duplicated")
        strongest[metric] = row
    if set(strongest) != set(metrics):
        raise PublicationSynthesisError("strongest comparator metric family differs")
    pairwise_values = _sequence(
        report.get("paired_comparisons"), "primary pairwise comparisons"
    )
    pairwise = [
        row for row in pairwise_values if isinstance(row, Mapping)
    ]
    required = reconstruction.get("required_comparator_ids")
    required_ids = tuple(_sequence(required, "required comparator denominator"))
    recomputed = _recomputed_strongest_comparators(
        reconstruction,
        metrics=metrics,
        required_ids=required_ids,  # type: ignore[arg-type]
    )
    result: list[dict[str, object]] = []
    for metric in metrics:
        strongest_row = strongest[metric]
        expected_strongest = recomputed[metric]
        comparator = strongest_row.get("method_id")
        tied_value = strongest_row.get("tied_method_ids")
        tied = list(tied_value) if isinstance(tied_value, list) else []
        common = {
            "metric": metric,
            "comparator_method_id": comparator,
            "tied_method_ids": tied,
        }
        if strongest_row.get("status") != "ok" or not isinstance(comparator, str):
            if expected_strongest is not None:
                raise PublicationSynthesisError(
                    "strongest comparator tie authority differs"
                )
            result.append(
                {
                    **common,
                    "status": "unavailable",
                    "permitted": False,
                    "reason": "complete_strongest_comparator_unavailable",
                }
            )
            continue
        if expected_strongest is None:
            raise PublicationSynthesisError(
                "strongest comparator tie authority differs"
            )
        supplied_median = _finite(
            strongest_row.get("median"), "supplied strongest comparator median"
        )
        if (
            comparator not in required_ids
            or not tied
            or any(
                not isinstance(method_id, str) or method_id not in required_ids
                for method_id in tied
            )
            or comparator not in tied
            or comparator != min(tied)
            or len(tied) != len(set(tied))
            or comparator != expected_strongest["method_id"]
            or tied != expected_strongest["tied_method_ids"]
            or supplied_median != expected_strongest["median"]
            or strongest_row.get("n_biological_draws")
            != expected_strongest["n_biological_draws"]
        ):
            raise PublicationSynthesisError(
                "strongest comparator tie authority differs"
            )
        family_rows = [
            row
            for row in pairwise
            if row.get("candidate_method_id") == report.get("candidate_method_id")
            and row.get("comparator_method_id") == comparator
        ]
        family_by_metric = {
            row.get("metric"): row
            for row in family_rows
            if isinstance(row.get("metric"), str)
        }
        if (
            len(family_rows) != len(metrics)
            or len(family_by_metric) != len(metrics)
            or set(family_by_metric) != set(metrics)
        ):
            result.append(
                {
                    **common,
                    "status": "unavailable",
                    "permitted": False,
                    "reason": "complete_pairwise_family_unavailable",
                }
            )
            continue
        if any(
            row.get("holm_family_id") != "protocol_primary_metrics"
            or row.get("holm_hypothesis_count") != len(metrics)
            or row.get("holm_status") != "ok"
            for row in family_by_metric.values()
        ):
            result.append(
                {
                    **common,
                    "status": "not_permitted",
                    "permitted": False,
                    "reason": "multiplicity_adjustment_unavailable",
                }
            )
            continue
        comparison = family_by_metric[metric]
        reason: str | None = None
        if competitive.get("status") != "passed":
            reason = "competitive_gate_not_passed"
        elif (
            comparison.get("status") != "ok"
            or comparison.get("favorable_direction") != "lower"
        ):
            reason = "selected_comparison_unavailable"
        else:
            complete_draws = strongest_row.get("n_biological_draws")
            comparison_draws = comparison.get("n_independent_biological_draws")
            paired_views = comparison.get("n_paired_dataset_views")
            exclusions = comparison.get("exclusions")
            complete_pairwise = (
                type(complete_draws) is int
                and complete_draws > 0
                and comparison_draws == complete_draws
                and paired_views == 2 * complete_draws
                and comparison.get("direction_source")
                == "validated_frozen_metric_direction_contract"
                and isinstance(exclusions, Mapping)
                and set(exclusions) == _PAIRWISE_EXCLUSION_NAMES
                and all(
                    type(value) is int and value >= 0
                    for value in exclusions.values()
                )
                and all(
                    exclusions[name] == 0
                    for name in _FORBIDDEN_PAIRWISE_EXCLUSIONS
                )
            )
            if not complete_pairwise:
                reason = "complete_pairwise_denominator_unavailable"
        if reason is None:
            try:
                lower = _finite(comparison.get("ci_95_lower"), "95% CI lower")
                upper = _finite(comparison.get("ci_95_upper"), "95% CI upper")
            except PublicationSynthesisError:
                reason = "confidence_interval_unavailable"
            else:
                if lower > upper or upper >= 0.0:
                    reason = "confidence_interval_not_strictly_favorable"
            if reason is None:
                try:
                    adjusted = _finite(
                        comparison.get("holm_adjusted_p_value"),
                        "Holm-adjusted p-value",
                    )
                except PublicationSynthesisError:
                    reason = "multiplicity_adjustment_unavailable"
                else:
                    if not 0.0 <= adjusted <= 0.05:
                        reason = "multiplicity_adjustment_not_significant"
        result.append(
            {
                **common,
                "status": "permitted" if reason is None else "not_permitted",
                "permitted": reason is None,
                "reason": reason,
            }
        )
    return result


def _evidence_bindings(
    loaded: _LoadedPublicationEvidence,
    *,
    report: Mapping[str, object],
    inputs: Mapping[str, object],
    trajectory_sha256: str,
) -> dict[str, object]:
    binding = loaded.downstream_plan.evaluated_round_binding
    assert binding is not None
    return {
        "round_id": binding.round_id,
        "evaluation_receipt_payload_sha256": (
            binding.evaluation_receipt_payload_sha256
        ),
        "result_manifest_sha256": binding.result_manifest_sha256,
        "final_plan_sha256": binding.final_plan_sha256,
        "final_execution_manifest_file_sha256": (
            binding.final_execution_manifest_file_sha256
        ),
        "final_execution_manifest_payload_sha256": (
            binding.final_execution_manifest_payload_sha256
        ),
        "primary_analysis_sha256": report["analysis_sha256"],
        "frozen_method_payload_sha256": inputs["frozen_method_payload_sha256"],
        "downstream_plan_sha256": loaded.downstream_plan.plan_sha256,
        "downstream_manifest_sha256": loaded.downstream_manifest.manifest_sha256,
        "final_null_de_plan_sha256": loaded.null_de_plan.plan_sha256,
        "final_null_de_manifest_sha256": loaded.null_de_manifest.manifest_sha256,
        "scaling_evidence_sha256": binding.scaling_evidence_sha256,
        "scaling_plan_sha256": binding.scaling_plan_sha256,
        "scaling_checkpoint_sha256": loaded.scaling_checkpoint.checkpoint_sha256,
        "trajectory_evidence_sha256": trajectory_sha256,
    }


def _build_publication_synthesis(
    loaded: _LoadedPublicationEvidence,
) -> dict[str, object]:
    (
        report,
        inputs,
        candidate,
        reconstruction,
        freeze,
        trajectory_sha256,
    ) = _validate_loaded_bindings(loaded)
    null_de = _build_final_null_de_gate(loaded)
    competitive = _competitive_gate(reconstruction, null_de)
    superiority = _superiority_permissions(report, reconstruction, competitive)
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "candidate_method_id": candidate,
        "evidence_bindings": _evidence_bindings(
            loaded,
            report=report,
            inputs=inputs,
            trajectory_sha256=trajectory_sha256,
        ),
        "freeze_prerequisite": freeze,
        "downstream": _downstream_summary(loaded),
        "scaling": _scaling_summary(loaded),
        "trajectory": {
            "role": "descriptive_only",
            "gate_influence": "none",
            "evidence_sha256": trajectory_sha256,
        },
        "gates": {
            "reconstruction": dict(reconstruction),
            "final_null_de": null_de,
            "competitive": competitive,
        },
        "claim_permissions": {
            "competitive": competitive["status"] == "passed",
            "superiority": superiority,
        },
    }
    return {**body, "synthesis_sha256": canonical_sha256(body)}


def _load_publication_evidence(
    repository: Path,
    round_dir: Path,
) -> _LoadedPublicationEvidence:
    def replay(name: str, action, *arguments):
        try:
            return action(*arguments)
        except PublicationSynthesisError:
            raise
        except (
            DownstreamEvidenceError,
            FinalAnalysisContractError,
            FinalNullDEError,
            PublicationFreezeError,
            ScalingContractError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise PublicationSynthesisError(f"{name} replay failed") from error

    primary_report = replay(
        "primary final analysis", generate_final_analysis, repository, round_dir
    )
    frozen_method = replay("frozen method", validate_frozen_method, repository)
    downstream_plan = replay(
        "final downstream plan",
        build_final_downstream_evidence_plan,
        repository,
        round_dir,
    )
    if (
        not isinstance(downstream_plan, DownstreamEvidencePlan)
        or downstream_plan.source_kind != "final"
        or downstream_plan.evidence_scope != "all"
        or downstream_plan.evaluated_round_binding is None
    ):
        raise PublicationSynthesisError(
            "downstream source must be complete frozen final evidence"
        )
    binding = downstream_plan.evaluated_round_binding
    bound_repository = Path(binding.repository_root)
    downstream_directory = (
        bound_repository.parent
        / f"{bound_repository.name}-final-analysis"
        / "downstream"
        / binding.round_id
        / binding.evaluation_receipt_payload_sha256
    ).absolute()
    persisted_downstream_plan = replay(
        "persisted downstream plan",
        load_downstream_evidence_plan,
        downstream_directory,
    )
    if (
        not isinstance(persisted_downstream_plan, DownstreamEvidencePlan)
        or persisted_downstream_plan.to_dict() != downstream_plan.to_dict()
    ):
        raise PublicationSynthesisError(
            "persisted downstream plan differs from independently rebuilt plan"
        )
    downstream_manifest = replay(
        "downstream manifest",
        load_downstream_evidence_manifest,
        downstream_directory,
    )
    null_de_plan = replay(
        "final null-DE plan",
        build_final_null_de_plan,
        repository,
        round_dir,
    )
    if (
        not isinstance(null_de_plan, FinalNullDEPlan)
        or null_de_plan.source_plan.to_dict() != downstream_plan.to_dict()
    ):
        raise PublicationSynthesisError(
            "final null-DE source plan differs from final downstream plan"
        )
    null_de_directory = replay(
        "final null-DE output location",
        expected_final_null_de_output_directory,
        null_de_plan,
    )
    null_de_manifest = replay(
        "final null-DE manifest",
        load_final_null_de_manifest,
        null_de_directory,
    )
    scaling_checkpoint = replay(
        "publication scaling evidence",
        load_publication_scaling_evidence,
        repository,
        round_dir,
    )
    return _LoadedPublicationEvidence(
        primary_report=primary_report,
        frozen_method=frozen_method,
        downstream_plan=downstream_plan,
        downstream_manifest=downstream_manifest,
        null_de_plan=null_de_plan,
        null_de_manifest=null_de_manifest,
        scaling_checkpoint=scaling_checkpoint,
    )


def generate_publication_synthesis(
    repository: Path,
    round_dir: Path,
) -> dict[str, object]:
    """Return claim permissions for the sole authoritative evaluated round."""

    if not isinstance(repository, Path) or not isinstance(round_dir, Path):
        raise TypeError("repository and round_dir must be pathlib.Path values")
    return _build_publication_synthesis(
        _load_publication_evidence(repository, round_dir)
    )


__all__ = ["PublicationSynthesisError", "generate_publication_synthesis"]
