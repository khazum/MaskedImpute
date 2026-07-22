"""Direct checkpoint and budget replay for the fair-comparator plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any, Literal

from .comparator_tuning import (
    DEVELOPMENT_MAX_CHECKPOINT_BYTES,
    DEVELOPMENT_MAX_RECORD_BYTES,
    comparator_method_binding,
)
from .direct_values import direct_equal, direct_json_value
from .fair_comparator_execution import (
    DirectEvaluatedAttempt,
    DirectLogReceipt,
    DirectMetricRow,
    DirectPreZeroEvidence,
    DirectRunResult,
    _executor_receipt_for_run,
    validate_direct_evidence_semantics,
)
from .fair_comparator_plan import (
    DirectCompetitionPlan,
    DirectPlanEntry,
    PreparedInputDescriptor,
    _validate_direct_competition_plan_structure,
    describe_prepared_input,
    direct_run_id,
    validate_direct_competition_plan,
)
from .methods import MethodRegistry, MethodSpec
from .runner import (
    COMPARATOR_SELECTION_BLOCKING_STATUSES,
    INTRINSIC_TERMINAL_STATUSES,
    MAX_CPU_BUDGET_SECONDS,
    MAX_DEVELOPMENT_CONFIGURATIONS,
    MAX_GPU_BUDGET_SECONDS,
    SELECTION_COMPLETENESS_BLOCKERS,
    BudgetDecision,
    DatasetBinding,
    PreparedDataset,
    RunnerAuthority,
    RunnerContractError,
    development_storage_preflight,
)


_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "identity_mode",
        "authority_revision",
        "plan_snapshot",
        "input_descriptors",
        "planned_run_count",
        "status",
        "evaluation_scope",
        "comparator_selection_status",
        "selection_complete",
        "selection_blockers",
        "records",
        "budget",
        "storage_preflight",
        "remaining_storage_preflight",
    }
)
_INTENT_KEYS = frozenset(
    {
        "schema_version",
        "identity_mode",
        "authority_revision",
        "plan_snapshot",
        "position",
        "entry_identity",
        "record",
    }
)
_RUN_KEYS = frozenset(
    {
        "run_id",
        "identity",
        "status",
        "reason",
        "runtime_seconds",
        "peak_rss_bytes",
        "peak_gpu_bytes",
        "rss_measurement",
        "gpu_measurement",
        "excluded_cell_count",
        "excluded_cell_ids",
        "retained_cell_count",
        "retained_cell_ids",
        "retained_gene_count",
        "observed_zero_count",
        "stdout",
        "stderr",
    }
)
_METRIC_KEYS = frozenset({"identity", "metric", "value", "n", "status", "reason"})
_PREZERO_KEYS = frozenset(
    {
        "applicable",
        "status",
        "reason",
        "shape",
        "dtype",
        "encoding",
        "path",
        "compressed_byte_count",
    }
)
_LOG_KEYS = frozenset(
    {"stream", "original_byte_count", "capture_policy", "terminal_reason"}
)
_RUN_STATUSES = frozenset(
    {
        "completed",
        "unavailable",
        "failed",
        "timeout",
        "resource_exceeded",
        "infrastructure_error",
        "blocked_authority",
        "budget_exhausted",
    }
)


_direct_equal = direct_equal


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RunnerContractError("direct checkpoint is not canonical JSON") from error


def _reject_json_constant(value: str) -> None:
    raise RunnerContractError(f"nonfinite direct checkpoint JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunnerContractError(f"duplicate direct checkpoint JSON key: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerContractError(f"{name} is not canonical JSON") from error
    if not isinstance(value, dict) or _canonical_bytes(value) + b"\n" != raw:
        raise RunnerContractError(f"{name} is not canonical JSON")
    return value


def _require_nonnegative_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (value == 0 and math.copysign(1.0, float(value)) < 0.0)
    ):
        raise RunnerContractError(f"{name} must be a finite nonnegative number")
    return float(value)


def _configuration_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerContractError("budget configuration ID must be a nonempty string")
    return value


def _scope(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RunnerContractError("budget scope must be a nonempty string")
    return value


def configuration_budget_key(entry: DirectPlanEntry) -> str:
    if not isinstance(entry, DirectPlanEntry):
        raise TypeError("entry must be a DirectPlanEntry")
    return _configuration_key(entry.identity.configuration_id)


def budget_scope(entry: DirectPlanEntry) -> str:
    if not isinstance(entry, DirectPlanEntry):
        raise TypeError("entry must be a DirectPlanEntry")
    identity = entry.identity
    return (
        f"{identity.method.method_id}:{identity.configuration_kind}"
        if identity.method.method_id == "maskimpute"
        else identity.method.method_id
    )


def counts_toward_configuration_limit(entry: DirectPlanEntry) -> bool:
    if not isinstance(entry, DirectPlanEntry):
        raise TypeError("entry must be a DirectPlanEntry")
    return entry.identity.configuration_kind in {
        "candidate_search",
        "comparator_tuning",
    }


class DirectDevelopmentBudget:
    """Task 7's single ledger keyed by readable configuration IDs."""

    def __init__(self) -> None:
        self._configurations: dict[str, set[str]] = {}
        self._consumed_seconds: dict[str, float] = {}

    def authorize(
        self,
        spec: MethodSpec,
        configuration_id: str,
        *,
        counts_toward_configuration_limit: bool = True,
        budget_scope: str | None = None,
    ) -> BudgetDecision:
        if not isinstance(spec, MethodSpec):
            raise TypeError("spec must be a MethodSpec")
        key = _configuration_key(configuration_id)
        scope = _scope(spec.id if budget_scope is None else budget_scope)
        configurations = self._configurations.get(scope, set())
        limit = (
            float(MAX_GPU_BUDGET_SECONDS)
            if spec.resources.gpu_required
            else float(MAX_CPU_BUDGET_SECONDS)
        )
        consumed = self._consumed_seconds.get(scope, 0.0)
        remaining = max(0.0, limit - consumed)
        if (
            counts_toward_configuration_limit
            and key not in configurations
            and len(configurations) >= MAX_DEVELOPMENT_CONFIGURATIONS
        ):
            return BudgetDecision(
                authorized=False,
                reason="configuration_budget_exhausted",
                remaining_seconds=remaining,
                timeout_seconds=0.0,
            )
        if remaining <= 0:
            return BudgetDecision(
                authorized=False,
                reason=(
                    "gpu_time_budget_exhausted"
                    if spec.resources.gpu_required
                    else "cpu_time_budget_exhausted"
                ),
                remaining_seconds=0.0,
                timeout_seconds=0.0,
            )
        return BudgetDecision(
            authorized=True,
            reason=None,
            remaining_seconds=remaining,
            timeout_seconds=min(float(spec.resources.timeout_seconds), remaining),
        )

    def record(
        self,
        spec: MethodSpec,
        configuration_id: str,
        outcome: DirectEvaluatedAttempt | DirectRunResult,
        *,
        counts_toward_configuration_limit: bool = True,
        budget_scope: str | None = None,
    ) -> None:
        if not isinstance(spec, MethodSpec):
            raise TypeError("spec must be a MethodSpec")
        key = _configuration_key(configuration_id)
        if isinstance(outcome, DirectEvaluatedAttempt):
            run = outcome.run
        elif isinstance(outcome, DirectRunResult):
            run = outcome
        else:
            raise TypeError("outcome must be a direct run or evaluated attempt")
        if run.status in COMPARATOR_SELECTION_BLOCKING_STATUSES:
            return
        self._restore_consuming(
            spec,
            key,
            run.runtime_seconds,
            counts_toward_configuration_limit=counts_toward_configuration_limit,
            budget_scope=budget_scope,
            overage_message="configuration budget was exceeded before recording",
        )

    def restore(
        self,
        spec: MethodSpec,
        configuration_id: str,
        status: str,
        runtime_seconds: int | float,
        *,
        counts_toward_configuration_limit: bool = True,
        budget_scope: str | None = None,
    ) -> None:
        if not isinstance(spec, MethodSpec):
            raise TypeError("spec must be a MethodSpec")
        key = _configuration_key(configuration_id)
        if status not in _RUN_STATUSES:
            raise RunnerContractError("restored direct run status is invalid")
        if status in COMPARATOR_SELECTION_BLOCKING_STATUSES:
            return
        self._restore_consuming(
            spec,
            key,
            runtime_seconds,
            counts_toward_configuration_limit=counts_toward_configuration_limit,
            budget_scope=budget_scope,
            overage_message="checkpoint exceeds the configuration budget",
        )

    def _restore_consuming(
        self,
        spec: MethodSpec,
        configuration_id: str,
        runtime_seconds: int | float,
        *,
        counts_toward_configuration_limit: bool,
        budget_scope: str | None,
        overage_message: str,
    ) -> None:
        runtime = _require_nonnegative_number(runtime_seconds, "restored runtime")
        scope = _scope(spec.id if budget_scope is None else budget_scope)
        limit = (
            float(MAX_GPU_BUDGET_SECONDS)
            if spec.resources.gpu_required
            else float(MAX_CPU_BUDGET_SECONDS)
        )
        consumed = self._consumed_seconds.get(scope, 0.0)
        if consumed + runtime > limit:
            raise RunnerContractError(
                f"{overage_message}: direct time budget ceiling exceeded"
            )
        if counts_toward_configuration_limit:
            configurations = self._configurations.setdefault(scope, set())
            configurations.add(configuration_id)
            if len(configurations) > MAX_DEVELOPMENT_CONFIGURATIONS:
                raise RunnerContractError(overage_message)
        self._consumed_seconds[scope] = consumed + runtime

    def to_dict(self) -> dict[str, object]:
        return {
            scope: {
                "configuration_ids": sorted(self._configurations.get(scope, set())),
                "consumed_seconds": self._consumed_seconds.get(scope, 0.0),
            }
            for scope in sorted(set(self._configurations) | set(self._consumed_seconds))
        }


def _expected_identity(entry: DirectPlanEntry) -> Mapping[str, object]:
    encoded = direct_json_value(entry.identity)
    if not isinstance(encoded, Mapping):  # pragma: no cover - dataclass invariant
        raise AssertionError("direct entry identity must encode as an object")
    return encoded


def _validate_log(value: object, stream: str, reason: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _LOG_KEYS:
        raise RunnerContractError("direct checkpoint log receipt is invalid")
    if (
        value.get("stream") != stream
        or value.get("capture_policy") != "discard_content"
        or value.get("terminal_reason") != reason
        or type(value.get("original_byte_count")) is not int
        or int(value["original_byte_count"]) < 0
    ):
        raise RunnerContractError("direct checkpoint log receipt is invalid")


def _validate_prezero_evidence(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PREZERO_KEYS:
        raise RunnerContractError("direct checkpoint p_pre_zero evidence is invalid")
    if type(value.get("applicable")) is not bool:
        raise RunnerContractError(
            "direct checkpoint p_pre_zero applicability is invalid"
        )
    if not isinstance(value.get("status"), str):
        raise RunnerContractError("direct checkpoint p_pre_zero status is invalid")
    shape_value = value.get("shape")
    shape = tuple(shape_value) if isinstance(shape_value, list) else shape_value
    try:
        evidence = DirectPreZeroEvidence(
            applicable=value.get("applicable"),
            status=value.get("status"),
            reason=value.get("reason"),
            shape=shape,
            dtype=value.get("dtype"),
            encoding=value.get("encoding"),
            path=value.get("path"),
            compressed_byte_count=value.get("compressed_byte_count"),
        )
    except RunnerContractError:
        raise
    except (TypeError, ValueError) as error:
        raise RunnerContractError(
            "direct checkpoint p_pre_zero evidence is invalid"
        ) from error
    if evidence.path is not None:
        relative = PurePosixPath(evidence.path)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative.as_posix() != evidence.path
        ):
            raise RunnerContractError(
                "direct checkpoint p_pre_zero evidence path is unsafe"
            )
    encoded = direct_json_value(evidence)
    if not isinstance(encoded, dict) or not _direct_equal(encoded, value):
        raise RunnerContractError("direct checkpoint p_pre_zero evidence is invalid")
    return encoded


def _validate_record(
    value: object,
    entry: DirectPlanEntry,
    *,
    prepared: PreparedDataset | None = None,
    evidence_repository: Path | None = None,
    expected_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "run",
        "metrics",
        "p_pre_zero_evidence",
    }:
        raise RunnerContractError("direct checkpoint record has wrong schema")
    record = dict(value)
    run = record.get("run")
    if not isinstance(run, Mapping) or set(run) != _RUN_KEYS:
        raise RunnerContractError("direct checkpoint run has wrong schema")
    identity_snapshot = (
        _expected_identity(entry) if expected_identity is None else expected_identity
    )
    if (
        not _direct_equal(run.get("identity"), identity_snapshot)
        or run.get("run_id") != entry.run_id
    ):
        raise RunnerContractError("direct checkpoint record identity differs from plan")
    status = run.get("status")
    reason = run.get("reason")
    if status not in _RUN_STATUSES or (status == "completed") != (reason is None):
        raise RunnerContractError("direct checkpoint run status is invalid")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise RunnerContractError("direct checkpoint run reason is invalid")
    _require_nonnegative_number(run.get("runtime_seconds"), "direct run runtime")
    for name in (
        "peak_rss_bytes",
        "peak_gpu_bytes",
        "excluded_cell_count",
        "retained_cell_count",
        "retained_gene_count",
        "observed_zero_count",
    ):
        observed = run.get(name)
        if isinstance(observed, bool) or type(observed) is not int or observed < 0:
            raise RunnerContractError(f"direct checkpoint {name} is invalid")
    for name in ("excluded_cell_ids", "retained_cell_ids"):
        observed = run.get(name)
        if not isinstance(observed, list) or not all(
            isinstance(item, str) for item in observed
        ):
            raise RunnerContractError(f"direct checkpoint {name} is invalid")
    if (
        len(run["excluded_cell_ids"]) != run["excluded_cell_count"]
        or len(run["retained_cell_ids"]) != run["retained_cell_count"]
        or not isinstance(run.get("rss_measurement"), str)
        or not run.get("rss_measurement")
        or not isinstance(run.get("gpu_measurement"), str)
        or not run.get("gpu_measurement")
    ):
        raise RunnerContractError("direct checkpoint run audit is invalid")
    if prepared is not None:
        expected_qc = {
            "excluded_cell_count": prepared.audit.excluded_cell_count,
            "excluded_cell_ids": list(prepared.audit.excluded_cell_ids),
            "retained_cell_count": prepared.audit.retained_cell_count,
            "retained_cell_ids": list(prepared.audit.retained_cell_ids),
            "retained_gene_count": prepared.method_input.shape[1],
            "observed_zero_count": int((prepared.method_input.counts == 0).sum()),
        }
        if any(
            not _direct_equal(run.get(name), expected)
            for name, expected in expected_qc.items()
        ):
            raise RunnerContractError(
                "direct checkpoint run audit differs from prepared input"
            )
    _validate_log(run.get("stdout"), "stdout", reason)
    _validate_log(run.get("stderr"), "stderr", reason)
    metrics = record.get("metrics")
    if not isinstance(metrics, list):
        raise RunnerContractError("direct checkpoint metrics are invalid")
    metric_rows: list[DirectMetricRow] = []
    for metric in metrics:
        if not isinstance(metric, Mapping) or set(metric) != _METRIC_KEYS:
            raise RunnerContractError("direct checkpoint metric has wrong schema")
        if not _direct_equal(metric.get("identity"), identity_snapshot):
            raise RunnerContractError(
                "direct checkpoint metric identity differs from plan"
            )
        if (
            not isinstance(metric.get("metric"), str)
            or not metric.get("metric")
            or metric.get("status") not in _RUN_STATUSES
            or isinstance(metric.get("n"), bool)
            or type(metric.get("n")) is not int
            or int(metric["n"]) < 0
        ):
            raise RunnerContractError("direct checkpoint metric is invalid")
        metric_value = metric.get("value")
        if metric_value is not None and (
            type(metric_value) is not float
            or not math.isfinite(metric_value)
            or (metric_value == 0.0 and math.copysign(1.0, metric_value) < 0.0)
        ):
            raise RunnerContractError("direct checkpoint metric value is invalid")
        metric_reason = metric.get("reason")
        if metric_value is None:
            if (
                not isinstance(metric_reason, str)
                or not metric_reason
                or metric.get("status") == "completed"
            ):
                raise RunnerContractError("direct checkpoint metric is inconsistent")
        elif metric.get("status") != "completed" or metric_reason is not None:
            raise RunnerContractError("direct checkpoint metric is inconsistent")
        metric_rows.append(
            DirectMetricRow(
                identity=entry.identity,
                metric=metric["metric"],
                value=metric_value,
                n=metric["n"],
                status=metric["status"],
                reason=metric_reason,
            )
        )
    encoded_evidence = _validate_prezero_evidence(record.get("p_pre_zero_evidence"))
    record["p_pre_zero_evidence"] = encoded_evidence
    shape_value = encoded_evidence["shape"]
    evidence = DirectPreZeroEvidence(
        applicable=encoded_evidence["applicable"],
        status=encoded_evidence["status"],
        reason=encoded_evidence["reason"],
        shape=(None if shape_value is None else tuple(shape_value)),
        dtype=encoded_evidence["dtype"],
        encoding=encoded_evidence["encoding"],
        path=encoded_evidence["path"],
        compressed_byte_count=encoded_evidence["compressed_byte_count"],
    )
    run_result = DirectRunResult(
        run_id=run["run_id"],
        identity=entry.identity,
        status=status,
        reason=reason,
        runtime_seconds=run["runtime_seconds"],
        peak_rss_bytes=run["peak_rss_bytes"],
        peak_gpu_bytes=run["peak_gpu_bytes"],
        rss_measurement=run["rss_measurement"],
        gpu_measurement=run["gpu_measurement"],
        excluded_cell_count=run["excluded_cell_count"],
        excluded_cell_ids=tuple(run["excluded_cell_ids"]),
        retained_cell_count=run["retained_cell_count"],
        retained_cell_ids=tuple(run["retained_cell_ids"]),
        retained_gene_count=run["retained_gene_count"],
        observed_zero_count=run["observed_zero_count"],
        stdout=DirectLogReceipt(**run["stdout"]),
        stderr=DirectLogReceipt(**run["stderr"]),
    )
    _executor_receipt_for_run(run_result)
    validate_direct_evidence_semantics(run_result, tuple(metric_rows), evidence)
    if (
        evidence.applicable
        and evidence.status == "completed"
        and evidence_repository is not None
    ):
        matrix = evidence.reopen(evidence_repository)
        if (
            matrix is None
            or prepared is None
            or matrix.shape != prepared.method_input.shape
        ):
            raise RunnerContractError(
                "direct checkpoint p_pre_zero evidence differs from prepared input"
            )
    record_raw = _canonical_bytes(record) + b"\n"
    if len(record_raw) > DEVELOPMENT_MAX_RECORD_BYTES:
        raise RunnerContractError("development record exceeds its byte bound")
    return json.loads(record_raw[:-1].decode("utf-8"))


def _resolve_direct_method_specs(
    registry: MethodRegistry,
    entries: Sequence[DirectPlanEntry],
) -> dict[str, MethodSpec]:
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    entry_values = tuple(entries)
    if not all(isinstance(entry, DirectPlanEntry) for entry in entry_values):
        raise TypeError("entries must contain DirectPlanEntry values")
    if not all(isinstance(spec, MethodSpec) for spec in registry.methods):
        raise RunnerContractError("direct method registry is invalid")

    method_ids = tuple(
        dict.fromkeys(entry.identity.method.method_id for entry in entry_values)
    )
    resolved: dict[str, MethodSpec] = {}
    for method_id in method_ids:
        matches = tuple(spec for spec in registry.methods if spec.id == method_id)
        if len(matches) != 1:
            raise RunnerContractError(
                "direct method registry must resolve each referenced method exactly once"
            )
        resolved[method_id] = matches[0]

    for entry in entry_values:
        method_id = entry.identity.method.method_id
        if not _direct_equal(
            comparator_method_binding(resolved[method_id]),
            entry.identity.method,
        ):
            raise RunnerContractError(
                "direct method projection differs from the method registry"
            )
    return resolved


def replay_direct_development_budget(
    registry: MethodRegistry,
    entries: Sequence[DirectPlanEntry],
    records: Sequence[Mapping[str, object]],
) -> DirectDevelopmentBudget:
    entry_values = tuple(entries)
    record_values = tuple(records)
    if len(record_values) > len(entry_values):
        raise RunnerContractError("direct checkpoint records are not a plan prefix")
    resolved = _resolve_direct_method_specs(registry, entry_values)
    budget = DirectDevelopmentBudget()
    for entry, stored in zip(entry_values, record_values, strict=False):
        record = _validate_record(stored, entry)
        run = record["run"]
        assert isinstance(run, dict)
        budget.restore(
            resolved[entry.identity.method.method_id],
            configuration_budget_key(entry),
            str(run.get("status")),
            run.get("runtime_seconds"),
            counts_toward_configuration_limit=counts_toward_configuration_limit(entry),
            budget_scope=budget_scope(entry),
        )
    return budget


def direct_comparator_selection_status(
    entries: Sequence[DirectPlanEntry],
    records: Sequence[Mapping[str, object]],
) -> Literal[
    "complete_terminal_denominator",
    "blocked_incomplete_denominator",
]:
    entry_values = tuple(entries)
    record_values = tuple(records)
    if len(record_values) > len(entry_values):
        raise RunnerContractError("direct checkpoint records are not a plan prefix")
    for position, entry in enumerate(entry_values):
        if entry.identity.configuration_kind != "comparator_tuning":
            continue
        if position >= len(record_values):
            return "blocked_incomplete_denominator"
        record = _validate_record(record_values[position], entry)
        run = record["run"]
        assert isinstance(run, dict)
        status = run.get("status")
        if status in COMPARATOR_SELECTION_BLOCKING_STATUSES:
            return "blocked_incomplete_denominator"
        if status != "completed" and status not in INTRINSIC_TERMINAL_STATUSES:
            raise RunnerContractError(
                "direct checkpoint comparator-selection status is invalid"
            )
    return "complete_terminal_denominator"


@dataclass(frozen=True, slots=True)
class DirectCheckpointReport:
    schema_version: int
    identity_mode: Literal["direct-v1"]
    authority_revision: str
    plan_snapshot: Mapping[str, object]
    input_descriptors: tuple[PreparedInputDescriptor, ...]
    planned_run_count: int
    status: Literal["running", "completed"]
    evaluation_scope: Literal["reconstruction_only"]
    comparator_selection_status: Literal[
        "complete_terminal_denominator",
        "blocked_incomplete_denominator",
    ]
    selection_complete: bool
    selection_blockers: tuple[str, ...]
    records: tuple[Mapping[str, object], ...]
    budget: Mapping[str, object]
    storage_preflight: Mapping[str, object]
    remaining_storage_preflight: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        encoded = direct_json_value(self)
        if not isinstance(encoded, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError("direct checkpoint must encode as an object")
        return encoded


def _descriptor_from_json(value: object) -> PreparedInputDescriptor:
    names = {item.name for item in fields(PreparedInputDescriptor)}
    if not isinstance(value, Mapping) or set(value) != names:
        raise RunnerContractError("direct checkpoint input descriptors are invalid")
    observed = dict(value)
    for name in ("shape", "cell_ids", "gene_ids", "batch_labels"):
        if not isinstance(observed[name], list):
            raise RunnerContractError("direct checkpoint input descriptors are invalid")
        observed[name] = tuple(observed[name])
    try:
        return PreparedInputDescriptor(**observed)
    except TypeError as error:
        raise RunnerContractError(
            "direct checkpoint input descriptors are invalid"
        ) from error


def _prepared_descriptors(
    plan: DirectCompetitionPlan,
    prepared_datasets: Mapping[str, PreparedDataset],
) -> tuple[PreparedInputDescriptor, ...]:
    if not isinstance(prepared_datasets, Mapping):
        raise TypeError("prepared_datasets must be a mapping")
    expected_ids = tuple(descriptor.dataset_id for descriptor in plan.inputs)
    if len(expected_ids) != len(set(expected_ids)) or set(prepared_datasets) != set(
        expected_ids
    ):
        raise RunnerContractError(
            "prepared dataset authority does not exactly cover the direct plan"
        )
    descriptors: list[PreparedInputDescriptor] = []
    for expected in plan.inputs:
        prepared = prepared_datasets.get(expected.dataset_id)
        if not isinstance(prepared, PreparedDataset):
            raise RunnerContractError("direct prepared dataset authority is invalid")
        observed = describe_prepared_input(prepared)
        if not _direct_equal(observed, expected):
            raise RunnerContractError("direct prepared input descriptor differs")
        descriptors.append(observed)
    return tuple(descriptors)


def _validate_plan(plan: DirectCompetitionPlan) -> dict[str, object]:
    if not isinstance(plan, DirectCompetitionPlan):
        raise TypeError("plan must be a DirectCompetitionPlan")
    if (
        type(plan.schema_version) is not int
        or plan.schema_version != 1
        or plan.identity_mode != "direct-v1"
        or not isinstance(plan.authority_revision, str)
        or not plan.authority_revision
    ):
        raise RunnerContractError("direct checkpoint plan mode is invalid")
    if any(
        entry.identity.authority_revision != plan.authority_revision
        for entry in plan.entries
    ):
        raise RunnerContractError("direct checkpoint plan authority differs")
    ordinals = tuple(entry.identity.ordinal for entry in plan.entries)
    if any(type(ordinal) is not int for ordinal in ordinals) or ordinals != tuple(
        range(1, len(plan.entries) + 1)
    ):
        raise RunnerContractError("direct checkpoint plan ordinals are not contiguous")
    if any(entry.run_id != direct_run_id(entry.identity) for entry in plan.entries):
        raise RunnerContractError("direct checkpoint plan run ID differs")
    snapshot = plan.to_dict()
    if set(snapshot) != {
        "schema_version",
        "identity_mode",
        "authority_revision",
        "inputs",
        "entries",
        "configurations",
        "comparator_smoke_receipt",
        "comparator_smoke_receipt_bytes",
    }:
        raise RunnerContractError("direct checkpoint plan snapshot is invalid")
    return snapshot


def _snapshot_identity(
    snapshot: Mapping[str, object],
    position: int,
) -> Mapping[str, object]:
    entries = snapshot.get("entries")
    if not isinstance(entries, list) or position >= len(entries):
        raise RunnerContractError("direct checkpoint plan snapshot is invalid")
    entry = entries[position]
    identity = entry.get("identity") if isinstance(entry, Mapping) else None
    if not isinstance(identity, Mapping):
        raise RunnerContractError("direct checkpoint plan snapshot is invalid")
    return identity


class DirectCheckpointStore:
    """Atomic direct checkpoint with one recoverable provisional record."""

    def __init__(self, path: Path, *, repository_root: Path | None = None) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a pathlib.Path")
        if repository_root is not None and not isinstance(repository_root, Path):
            raise TypeError("repository_root must be a pathlib.Path")
        self.path = path.absolute()
        self.repository_root = (
            self.path.parent if repository_root is None else repository_root.absolute()
        )
        self.intent_path = self.path.with_name(f".{self.path.name}.transaction.json")

    def _read_owned(self, path: Path, name: str) -> bytes:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RunnerContractError(f"{name} is unavailable") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise RunnerContractError(f"{name} must be an owned regular file")
        if metadata.st_size > DEVELOPMENT_MAX_CHECKPOINT_BYTES:
            raise RunnerContractError(f"{name} exceeds its byte bound")
        try:
            return path.read_bytes()
        except OSError as error:
            raise RunnerContractError(f"{name} is unavailable") from error

    def _publish(self, path: Path, payload: Mapping[str, object]) -> None:
        data = _canonical_bytes(payload) + b"\n"
        name = "direct checkpoint" if path == self.path else "direct transaction intent"
        if len(data) > DEVELOPMENT_MAX_CHECKPOINT_BYTES:
            raise RunnerContractError(f"{name} exceeds its byte bound")
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(path):
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
            ):
                raise RunnerContractError(
                    "direct checkpoint destination must be an owned regular file"
                )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _body(
        plan: DirectCompetitionPlan,
        descriptors: Sequence[PreparedInputDescriptor],
        records: Sequence[Mapping[str, object]],
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> dict[str, object]:
        record_values = tuple(records)
        status = "completed" if len(record_values) == len(plan.entries) else "running"
        storage_preflight = development_storage_preflight(
            plan,
            prepared_datasets,
            completed_records=0,
        ).to_dict()
        remaining_storage_preflight = development_storage_preflight(
            plan,
            prepared_datasets,
            completed_records=len(record_values),
        ).to_dict()
        return {
            "schema_version": 1,
            "identity_mode": "direct-v1",
            "authority_revision": plan.authority_revision,
            "plan_snapshot": plan.to_dict(),
            "input_descriptors": [direct_json_value(value) for value in descriptors],
            "planned_run_count": len(plan.entries),
            "status": status,
            "evaluation_scope": "reconstruction_only",
            "comparator_selection_status": direct_comparator_selection_status(
                plan.entries, record_values
            ),
            "selection_complete": False,
            "selection_blockers": list(SELECTION_COMPLETENESS_BLOCKERS),
            "records": list(record_values),
            "budget": replay_direct_development_budget(
                registry, plan.entries, record_values
            ).to_dict(),
            "storage_preflight": storage_preflight,
            "remaining_storage_preflight": remaining_storage_preflight,
        }

    def write(
        self,
        plan: DirectCompetitionPlan,
        records: Sequence[Mapping[str, object]],
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
        authority: RunnerAuthority,
        datasets: Sequence[DatasetBinding],
    ) -> DirectCheckpointReport:
        validate_direct_competition_plan(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
            authority=authority,
            datasets=datasets,
        )
        return self._write_validated(
            plan,
            records,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _write_structural(
        self,
        plan: DirectCompetitionPlan,
        records: Sequence[Mapping[str, object]],
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> DirectCheckpointReport:
        _validate_direct_competition_plan_structure(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        return self._write_validated(
            plan,
            records,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _write_validated(
        self,
        plan: DirectCompetitionPlan,
        records: Sequence[Mapping[str, object]],
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> DirectCheckpointReport:
        snapshot = _validate_plan(plan)
        descriptors = _prepared_descriptors(plan, prepared_datasets)
        record_values = tuple(
            _validate_record(
                value,
                entry,
                prepared=prepared_datasets[entry.identity.dataset_id],
                expected_identity=_snapshot_identity(snapshot, position),
            )
            for position, (value, entry) in enumerate(
                zip(records, plan.entries, strict=False)
            )
        )
        if len(records) > len(plan.entries):
            raise RunnerContractError("direct checkpoint records are not a plan prefix")
        self._publish(
            self.path,
            self._body(
                plan,
                descriptors,
                record_values,
                registry,
                prepared_datasets,
            ),
        )
        return self._load_validated(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def append(
        self,
        plan: DirectCompetitionPlan,
        report: DirectCheckpointReport | None,
        attempt: DirectEvaluatedAttempt,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
        authority: RunnerAuthority,
        datasets: Sequence[DatasetBinding],
    ) -> DirectCheckpointReport:
        validate_direct_competition_plan(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
            authority=authority,
            datasets=datasets,
        )
        if not isinstance(attempt, DirectEvaluatedAttempt):
            raise TypeError("attempt must be a DirectEvaluatedAttempt")
        return self._append_validated(
            plan,
            report,
            attempt,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _append_structural(
        self,
        plan: DirectCompetitionPlan,
        report: DirectCheckpointReport | None,
        attempt: DirectEvaluatedAttempt,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> DirectCheckpointReport:
        if not isinstance(attempt, DirectEvaluatedAttempt):
            raise TypeError("attempt must be a DirectEvaluatedAttempt")
        _validate_direct_competition_plan_structure(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        return self._append_validated(
            plan,
            report,
            attempt,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _append_validated(
        self,
        plan: DirectCompetitionPlan,
        report: DirectCheckpointReport | None,
        attempt: DirectEvaluatedAttempt,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> DirectCheckpointReport:
        if os.path.lexists(self.path):
            current = self._load_validated(
                plan,
                registry=registry,
                prepared_datasets=prepared_datasets,
            )
            if report is not None and not _direct_equal(
                report.to_dict(), current.to_dict()
            ):
                raise RunnerContractError(
                    "direct checkpoint report differs from storage"
                )
        elif report is not None:
            raise RunnerContractError(
                "direct checkpoint report has no durable checkpoint"
            )
        else:
            current = None
        records = [] if current is None else list(current.records)
        if len(records) >= len(plan.entries):
            raise RunnerContractError(
                "direct checkpoint already contains its full plan"
            )
        position = len(records)
        entry = plan.entries[position]
        record = _validate_record(
            attempt.to_dict(),
            entry,
            prepared=prepared_datasets[entry.identity.dataset_id],
            expected_identity=_snapshot_identity(_validate_plan(plan), position),
        )
        self._publish_transaction_intent(plan, position, entry, attempt)
        descriptors = _prepared_descriptors(plan, prepared_datasets)
        self._publish(
            self.path,
            self._body(
                plan,
                descriptors,
                (*records, record),
                registry,
                prepared_datasets,
            ),
        )
        return self._load_validated(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _publish_transaction_intent(
        self,
        plan: DirectCompetitionPlan,
        position: int,
        entry: DirectPlanEntry,
        attempt: DirectEvaluatedAttempt,
    ) -> None:
        snapshot = _validate_plan(plan)
        if (
            isinstance(position, bool)
            or type(position) is not int
            or position < 0
            or position >= len(plan.entries)
            or not _direct_equal(entry, plan.entries[position])
        ):
            raise RunnerContractError("direct transaction position is invalid")
        identity = _snapshot_identity(snapshot, position)
        record = _validate_record(
            attempt.to_dict(),
            entry,
            expected_identity=identity,
        )
        body = {
            "schema_version": 1,
            "identity_mode": "direct-v1",
            "authority_revision": plan.authority_revision,
            "plan_snapshot": snapshot,
            "position": position,
            "entry_identity": identity,
            "record": record,
        }
        if os.path.lexists(self.intent_path):
            existing = self._read_owned(
                self.intent_path,
                "direct transaction intent",
            )
            if existing != _canonical_bytes(body) + b"\n":
                raise RunnerContractError("direct transaction intent already differs")
            return
        self._publish(self.intent_path, body)

    def _recover_interrupted_transaction(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> None:
        if not os.path.lexists(self.intent_path):
            return
        if os.path.lexists(self.path):
            report = self._load_current(
                plan,
                registry=registry,
                prepared_datasets=prepared_datasets,
            )
            records = list(report.records)
        else:
            records = []
        inspected = self._inspect_transaction_intent(
            plan,
            records=records,
            prepared_datasets=prepared_datasets,
        )
        if inspected is None:  # pragma: no cover - guarded above
            return
        position, record = inspected
        descriptors = _prepared_descriptors(plan, prepared_datasets)
        if position == len(records):
            self._publish(
                self.path,
                self._body(
                    plan,
                    descriptors,
                    (*records, record),
                    registry,
                    prepared_datasets,
                ),
            )
        self.intent_path.unlink()

    def _inspect_transaction_intent(
        self,
        plan: DirectCompetitionPlan,
        *,
        records: Sequence[Mapping[str, object]],
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> tuple[int, dict[str, object]] | None:
        if not os.path.lexists(self.intent_path):
            return None
        intent = _parse_json(
            self._read_owned(self.intent_path, "direct transaction intent"),
            "direct transaction intent",
        )
        if set(intent) != _INTENT_KEYS:
            raise RunnerContractError("direct transaction intent has wrong schema")
        snapshot = _validate_plan(plan)
        if (
            type(intent.get("schema_version")) is not int
            or intent.get("schema_version") != 1
            or intent.get("identity_mode") != "direct-v1"
            or intent.get("authority_revision") != plan.authority_revision
            or not _direct_equal(intent.get("plan_snapshot"), snapshot)
        ):
            raise RunnerContractError("direct transaction plan snapshot differs")
        position = intent.get("position")
        if (
            isinstance(position, bool)
            or type(position) is not int
            or position < 0
            or position >= len(plan.entries)
        ):
            raise RunnerContractError("direct transaction position is invalid")
        entry = plan.entries[position]
        identity = _snapshot_identity(snapshot, position)
        if not _direct_equal(intent.get("entry_identity"), identity):
            raise RunnerContractError("direct transaction entry identity differs")
        record = _validate_record(
            intent.get("record"),
            entry,
            prepared=prepared_datasets[entry.identity.dataset_id],
            expected_identity=identity,
        )
        if position == len(records):
            pass
        elif records and position == len(records) - 1:
            if not _direct_equal(records[position], record):
                raise RunnerContractError(
                    "durable direct transaction differs from its record"
                )
        else:
            raise RunnerContractError(
                "direct transaction position is stale or is not the next position"
            )
        return position, record

    def _load_current(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> DirectCheckpointReport:
        payload = _parse_json(
            self._read_owned(self.path, "direct checkpoint"),
            "direct checkpoint",
        )
        if set(payload) != _REPORT_KEYS:
            raise RunnerContractError("direct checkpoint has wrong schema")
        snapshot = _validate_plan(plan)
        descriptors = _prepared_descriptors(plan, prepared_datasets)
        if (
            type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
            or payload.get("identity_mode") != "direct-v1"
            or payload.get("authority_revision") != plan.authority_revision
        ):
            raise RunnerContractError(
                "direct checkpoint schema or identity mode is invalid"
            )
        if not _direct_equal(payload.get("plan_snapshot"), snapshot):
            raise RunnerContractError("direct checkpoint plan snapshot differs")
        stored_descriptors = payload.get("input_descriptors")
        if not isinstance(stored_descriptors, list):
            raise RunnerContractError("direct checkpoint input descriptors are invalid")
        decoded_descriptors = tuple(
            _descriptor_from_json(value) for value in stored_descriptors
        )
        if not _direct_equal(decoded_descriptors, descriptors):
            raise RunnerContractError("direct checkpoint input descriptors differ")
        if type(payload.get("planned_run_count")) is not int or payload.get(
            "planned_run_count"
        ) != len(plan.entries):
            raise RunnerContractError("direct checkpoint planned denominator changed")
        values = payload.get("records")
        if not isinstance(values, list) or len(values) > len(plan.entries):
            raise RunnerContractError("direct checkpoint records are not a plan prefix")
        records = tuple(
            _validate_record(
                value,
                entry,
                prepared=prepared_datasets[entry.identity.dataset_id],
                evidence_repository=self.repository_root,
                expected_identity=_snapshot_identity(snapshot, position),
            )
            for position, (value, entry) in enumerate(
                zip(values, plan.entries, strict=False)
            )
        )
        expected_status = (
            "completed" if len(records) == len(plan.entries) else "running"
        )
        if payload.get("status") != expected_status:
            raise RunnerContractError("direct checkpoint status contradicts its prefix")
        selection_status = direct_comparator_selection_status(plan.entries, records)
        if payload.get("comparator_selection_status") != selection_status:
            raise RunnerContractError(
                "direct checkpoint comparator-selection status differs from records"
            )
        if (
            payload.get("evaluation_scope") != "reconstruction_only"
            or payload.get("selection_complete") is not False
            or payload.get("selection_blockers")
            != list(SELECTION_COMPLETENESS_BLOCKERS)
        ):
            raise RunnerContractError(
                "direct checkpoint reconstruction completeness is invalid"
            )
        budget = payload.get("budget")
        replayed = replay_direct_development_budget(registry, plan.entries, records)
        if not isinstance(budget, dict) or not _direct_equal(
            budget, replayed.to_dict()
        ):
            raise RunnerContractError(
                "direct checkpoint budget ledger differs from replay"
            )
        storage_preflight = payload.get("storage_preflight")
        expected_storage_preflight = development_storage_preflight(
            plan,
            prepared_datasets,
            completed_records=0,
        ).to_dict()
        if not isinstance(storage_preflight, Mapping) or not _direct_equal(
            storage_preflight,
            expected_storage_preflight,
        ):
            raise RunnerContractError(
                "direct checkpoint storage preflight differs from its full plan"
            )
        remaining_storage_preflight = payload.get("remaining_storage_preflight")
        expected_remaining_storage_preflight = development_storage_preflight(
            plan,
            prepared_datasets,
            completed_records=len(records),
        ).to_dict()
        if not isinstance(
            remaining_storage_preflight,
            Mapping,
        ) or not _direct_equal(
            remaining_storage_preflight,
            expected_remaining_storage_preflight,
        ):
            raise RunnerContractError(
                "direct checkpoint remaining storage preflight differs from its prefix"
            )
        return DirectCheckpointReport(
            schema_version=1,
            identity_mode="direct-v1",
            authority_revision=plan.authority_revision,
            plan_snapshot=snapshot,
            input_descriptors=descriptors,
            planned_run_count=len(plan.entries),
            status=expected_status,
            evaluation_scope="reconstruction_only",
            comparator_selection_status=selection_status,
            selection_complete=False,
            selection_blockers=SELECTION_COMPLETENESS_BLOCKERS,
            records=records,
            budget=budget,
            storage_preflight=storage_preflight,
            remaining_storage_preflight=remaining_storage_preflight,
        )

    def inspect_prefix(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
        authority: RunnerAuthority,
        datasets: Sequence[DatasetBinding],
    ) -> int:
        """Validate the durable prefix and intent without recovery or writes."""

        validate_direct_competition_plan(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
            authority=authority,
            datasets=datasets,
        )
        return self._inspect_prefix_validated(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _inspect_prefix_structural(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> int:
        _validate_direct_competition_plan_structure(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        return self._inspect_prefix_validated(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _inspect_prefix_validated(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> int:
        if os.path.lexists(self.path):
            report = self._load_current(
                plan,
                registry=registry,
                prepared_datasets=prepared_datasets,
            )
            records = report.records
        else:
            records = ()
        self._inspect_transaction_intent(
            plan,
            records=records,
            prepared_datasets=prepared_datasets,
        )
        return len(records)

    def load(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
        authority: RunnerAuthority,
        datasets: Sequence[DatasetBinding],
    ) -> DirectCheckpointReport:
        validate_direct_competition_plan(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
            authority=authority,
            datasets=datasets,
        )
        return self._load_validated(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _load_structural(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> DirectCheckpointReport:
        _validate_direct_competition_plan_structure(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        return self._load_validated(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _load_validated(
        self,
        plan: DirectCompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> DirectCheckpointReport:
        self._recover_interrupted_transaction(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        return self._load_current(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )


__all__ = [
    "DirectCheckpointReport",
    "DirectCheckpointStore",
    "DirectDevelopmentBudget",
    "budget_scope",
    "configuration_budget_key",
    "counts_toward_configuration_limit",
    "direct_comparator_selection_status",
    "replay_direct_development_budget",
]
