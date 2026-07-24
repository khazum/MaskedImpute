"""Canonical analysis of immutable frozen-final benchmark evidence.

The analysis policy is intentionally not parameterized.  Model seeds and
technical views are repeated measurements; biological simulation draws are the
only independent inference units.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

from .protocol import canonical_sha256
from .statistics import (
    _finite_mean,
    _finite_median,
    _finite_quantile,
    hierarchical_paired_bootstrap,
    holm_adjust,
    summarize_seed_variance,
)


_TERMINAL_STATUSES = (
    "completed",
    "failed",
    "timeout",
    "resource_exceeded",
    "unavailable",
)
_ANALYTIC_STATUSES = ("ok", "failed", "timeout", "resource_exceeded", "unavailable")
_BOOTSTRAP_REPLICATES = 10_000
_BOOTSTRAP_SEED = 20_260_712
_DETERMINISTIC_SEED_SENTINEL = 0
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SCORE_METRICS = (
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "ece",
)
_SCORE_GROUPS = {
    "overall": ("all_observed_zeros",),
    "library_size_quartiles": ("Q1", "Q2", "Q3", "Q4"),
    "truth_expression_bins": ("[0,1)", "[1,2)", "[2,4)", "[4,inf)"),
}
_TRUTH_KINDS = frozenset(
    {
        "exact_pre_capture",
        "exact_continuous",
        "proxy_high_depth",
        "orthogonal_only",
    }
)
_STRUCTURAL_METRIC_UNAVAILABILITY = {
    "mse_pre_dropout_zero": {
        "exact_continuous": "undefined_for_continuous_truth",
        "proxy_high_depth": "proxy_truth_not_exact",
        "orthogonal_only": "truth_unavailable",
    }
}
_CLAIM_RANK_METRICS = ("mse", "mse_dropout", "gnrmse")
_CLAIM_PARETO_METRICS = (
    "mse_dropout",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
)
_SCORE_IDENTITY_FIELDS = frozenset(
    {
        "run_id",
        "method_id",
        "dataset_id",
        "source_dataset_sha256",
        "mechanism",
        "biological_id",
        "technical_view",
        "model_seed",
        "configuration_id",
        "configuration_sha256",
        "method_input_sha256",
        "retained_cell_ids_sha256",
    }
)
_SCORE_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "probability_semantics",
        "evaluation_domain",
        "score_source",
        "score_artifact_sha256",
        "score_input_sha256",
        "score_config_sha256",
        "calibration_file_sha256",
        "calibration_payload_sha256",
        "calibration_algorithm",
        "calibration_scope",
        "calibration_equivalence_reason",
    }
)
_EXPECTED_FINAL_STORAGE_POLICY = {
    "evaluator_output_compression_level": 6,
    "evaluator_output_encoding": "zlib_raw_f64_v1",
    "native_output_retention": "omitted_redundant_final_output",
    "p_pre_zero_compression_level": 6,
    "p_pre_zero_encoding": "zlib_raw_f64_v1",
}
_STORAGE_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        "completed_record_count",
        "remaining_entry_count",
        "remaining_execution_count",
        "remaining_p_pre_zero_execution_count",
        "per_execution_compressed_bound_bytes",
        "per_p_pre_zero_compressed_bound_bytes",
        "required_free_bytes",
        "observed_free_bytes",
    }
)
_EXECUTION_STORAGE_COMPONENT_KEYS = frozenset(
    {
        "completed_record_count",
        "remaining_entry_count",
        "remaining_execution_count",
        "remaining_p_pre_zero_execution_count",
        "cells",
        "genes",
        "per_execution_compressed_bound_bytes",
        "per_p_pre_zero_compressed_bound_bytes",
        "required_free_bytes",
    }
)
_COMBINED_STORAGE_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        "primary",
        "trajectory",
        "scaling",
        "reserve_bytes",
        "required_free_bytes",
        "observed_free_bytes",
    }
)


class FinalAnalysisContractError(RuntimeError):
    """Raised when final evidence or analysis authority is not canonical."""


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
        raise FinalAnalysisContractError(
            "final analysis evidence is not canonical JSON"
        ) from error


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FinalAnalysisContractError(f"{name} is not a SHA-256 digest")
    return value


def _unavailable_metric_direction_contract(reason: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "unavailable",
        "reason": reason,
        "favorable_direction": None,
        "metrics": [],
        "authority": None,
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def _validated_metric_direction_contract(
    input_bindings: Mapping[str, object],
) -> tuple[dict[str, object], frozenset[str]]:
    value = input_bindings.get("metric_direction_contract")
    if value is None:
        contract = _unavailable_metric_direction_contract(
            "frozen_metric_direction_authority_absent"
        )
        return contract, frozenset()
    contract = _exact_mapping(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "reason",
                "favorable_direction",
                "metrics",
                "authority",
                "contract_sha256",
            }
        ),
        "metric direction contract",
    )
    body = {key: nested for key, nested in contract.items() if key != "contract_sha256"}
    if (
        type(contract.get("schema_version")) is not int
        or contract.get("schema_version") != 1
        or contract.get("contract_sha256") != canonical_sha256(body)
    ):
        raise FinalAnalysisContractError("metric direction contract checksum differs")
    status = contract.get("status")
    metrics = contract.get("metrics")
    if status == "unavailable":
        if (
            not isinstance(contract.get("reason"), str)
            or not contract["reason"]
            or contract.get("favorable_direction") is not None
            or metrics != []
            or contract.get("authority") is not None
        ):
            raise FinalAnalysisContractError(
                "unavailable metric direction contract is invalid"
            )
        return dict(contract), frozenset()
    if (
        status != "validated"
        or contract.get("reason") is not None
        or contract.get("favorable_direction") != "lower"
        or not isinstance(metrics, list)
        or not metrics
        or any(not isinstance(metric, str) or not metric for metric in metrics)
        or not isinstance(contract.get("authority"), Mapping)
        or not contract["authority"]
    ):
        raise FinalAnalysisContractError(
            "validated metric direction contract is invalid"
        )
    assert isinstance(metrics, list)
    if metrics != sorted(set(metrics)):
        raise FinalAnalysisContractError(
            "validated metric direction contract is invalid"
        )
    return dict(contract), frozenset(metrics)


@dataclass(frozen=True, slots=True)
class _NormalizedMetric:
    mechanism: str
    biological_id: str
    technical_view: str
    dataset_id: str
    method: str
    model_seed: int
    source_model_seed: int | None
    truth_kind: str | None
    metric: str
    value: float | None
    status: str
    reason: str | None

    def statistics_row(self) -> dict[str, object]:
        return {
            "mechanism": self.mechanism,
            "biological_id": self.biological_id,
            "technical_view": self.technical_view,
            "dataset_id": self.dataset_id,
            "method": self.method,
            "model_seed": self.model_seed,
            "metric": self.metric,
            "value": self.value,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class _NormalizedEvidence:
    rows: tuple[_NormalizedMetric, ...]
    metric_names: tuple[str, ...]
    methods: tuple[str, ...]
    run_statuses: tuple[str, ...]
    run_reasons: tuple[str | None, ...]


@dataclass(frozen=True, slots=True)
class _NormalizedScoreMetric:
    mechanism: str
    biological_id: str
    technical_view: str
    dataset_id: str
    method: str
    model_seed: int
    stratum_type: str
    label: str
    metric: str
    value: float | None
    n: int
    status: str
    reason: str | None


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalAnalysisContractError(f"{name} must be a nonempty string")
    return value


def _model_seed(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value < 2**63:
        raise FinalAnalysisContractError(
            f"{name} must be null or an integer in [0, 2**63)"
        )
    return value


def _finite_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalAnalysisContractError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FinalAnalysisContractError(f"{name} must be a finite number")
    return numeric


def _reason(value: object, name: str) -> str:
    return _nonempty_string(value, name)


def _normalize_records(
    records: Sequence[Mapping[str, object]],
) -> _NormalizedEvidence:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    if not records:
        raise FinalAnalysisContractError("final analysis requires at least one record")

    rows: list[_NormalizedMetric] = []
    metric_names: tuple[str, ...] | None = None
    methods: set[str] = set()
    seeds_by_method: dict[str, set[str]] = {}
    run_statuses: list[str] = []
    run_reasons: list[str | None] = []
    seen_run_ids: set[str] = set()
    seen_metric_identities: set[tuple[object, ...]] = set()

    identity_fields = (
        "mechanism",
        "biological_id",
        "technical_view",
        "dataset_id",
    )
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FinalAnalysisContractError(f"record {record_index} must be a mapping")
        run = record.get("run")
        metrics = record.get("metrics")
        if not isinstance(run, Mapping) or not isinstance(metrics, list) or not metrics:
            raise FinalAnalysisContractError(
                f"record {record_index} must contain a run and nonempty metrics"
            )
        run_id = _nonempty_string(run.get("run_id"), f"record {record_index} run_id")
        if run_id in seen_run_ids:
            raise FinalAnalysisContractError("final records contain duplicate run_id")
        seen_run_ids.add(run_id)
        method = _nonempty_string(
            run.get("method_id"), f"record {record_index} method_id"
        )
        identity = {
            field: _nonempty_string(run.get(field), f"record {record_index} {field}")
            for field in identity_fields
        }
        seed = _model_seed(run.get("model_seed"), f"record {record_index} model_seed")
        truth_kind = run.get("truth_kind")
        score_evidence = record.get("p_pre_zero_evidence")
        if isinstance(score_evidence, Mapping):
            evidence_truth_kind = score_evidence.get("truth_kind")
            if truth_kind is not None and evidence_truth_kind != truth_kind:
                raise FinalAnalysisContractError(
                    f"record {record_index} truth kind differs from score evidence"
                )
            truth_kind = evidence_truth_kind
        if truth_kind is not None and truth_kind not in _TRUTH_KINDS:
            raise FinalAnalysisContractError(
                f"record {record_index} truth kind is invalid"
            )
        methods.add(method)
        seeds_by_method.setdefault(method, set()).add(
            "deterministic" if seed is None else "stochastic"
        )
        status = run.get("status")
        if status not in _TERMINAL_STATUSES:
            raise FinalAnalysisContractError(
                f"record {record_index} has a nonterminal run status"
            )
        run_reason = run.get("reason")
        if status == "completed":
            if run_reason is not None:
                raise FinalAnalysisContractError(
                    f"record {record_index} completed run has a reason"
                )
        else:
            run_reason = _reason(
                run_reason, f"record {record_index} terminal run reason"
            )
        run_statuses.append(str(status))
        run_reasons.append(run_reason if isinstance(run_reason, str) else None)

        observed_names: list[str] = []
        for metric_index, metric in enumerate(metrics):
            if not isinstance(metric, Mapping):
                raise FinalAnalysisContractError(
                    f"record {record_index} metric {metric_index} must be a mapping"
                )
            metric_name = _nonempty_string(
                metric.get("metric"),
                f"record {record_index} metric {metric_index} name",
            )
            observed_names.append(metric_name)
            if metric.get("method") != method or any(
                metric.get(field) != value for field, value in identity.items()
            ):
                raise FinalAnalysisContractError(
                    f"record {record_index} metric identity differs from its run"
                )
            metric_seed = _model_seed(
                metric.get("model_seed"),
                f"record {record_index} metric {metric_index} model_seed",
            )
            if metric_seed != seed:
                raise FinalAnalysisContractError(
                    f"record {record_index} metric seed differs from its run"
                )
            metric_identity = (
                identity["mechanism"],
                identity["biological_id"],
                identity["technical_view"],
                identity["dataset_id"],
                method,
                seed,
                metric_name,
            )
            if metric_identity in seen_metric_identities:
                raise FinalAnalysisContractError(
                    "final records contain a duplicate metric result identity"
                )
            seen_metric_identities.add(metric_identity)
            n = metric.get("n")
            if type(n) is not int or n < 0:
                raise FinalAnalysisContractError(
                    f"record {record_index} metric {metric_name} n is invalid"
                )
            metric_status = metric.get("status")
            metric_reason = metric.get("reason")
            metric_value = metric.get("value")
            if status == "completed" and metric_status == "completed":
                value = _finite_value(
                    metric_value, f"record {record_index} metric {metric_name} value"
                )
                if metric_reason is not None:
                    raise FinalAnalysisContractError(
                        f"record {record_index} completed metric has a reason"
                    )
                analytic_status = "ok"
                normalized_reason = None
            elif status == "completed" and metric_status == "unavailable":
                if metric_value is not None:
                    raise FinalAnalysisContractError(
                        f"record {record_index} unavailable metric has a value"
                    )
                value = None
                analytic_status = "unavailable"
                normalized_reason = _reason(
                    metric_reason,
                    f"record {record_index} unavailable metric reason",
                )
            elif metric_status == status and status != "completed":
                if metric_value is not None or n != 0 or metric_reason != run_reason:
                    raise FinalAnalysisContractError(
                        f"record {record_index} terminal metric differs from its run"
                    )
                value = None
                analytic_status = str(status)
                normalized_reason = str(run_reason)
            else:
                raise FinalAnalysisContractError(
                    f"record {record_index} metric status differs from its run"
                )
            rows.append(
                _NormalizedMetric(
                    mechanism=identity["mechanism"],
                    biological_id=identity["biological_id"],
                    technical_view=identity["technical_view"],
                    dataset_id=identity["dataset_id"],
                    method=method,
                    model_seed=(_DETERMINISTIC_SEED_SENTINEL if seed is None else seed),
                    source_model_seed=seed,
                    truth_kind=(str(truth_kind) if truth_kind is not None else None),
                    metric=metric_name,
                    value=value,
                    status=analytic_status,
                    reason=normalized_reason,
                )
            )
        names = tuple(observed_names)
        if len(set(names)) != len(names):
            raise FinalAnalysisContractError(
                f"record {record_index} contains duplicate metric names"
            )
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise FinalAnalysisContractError(
                "final records do not share one complete ordered metric denominator"
            )

    mixed = sorted(
        method for method, kinds in seeds_by_method.items() if len(kinds) > 1
    )
    if mixed:
        raise FinalAnalysisContractError(
            "methods mix deterministic and stochastic seed encodings: "
            + ", ".join(mixed)
        )
    assert metric_names is not None
    return _NormalizedEvidence(
        rows=tuple(rows),
        metric_names=metric_names,
        methods=tuple(sorted(methods)),
        run_statuses=tuple(run_statuses),
        run_reasons=tuple(run_reasons),
    )


def _score_metric_roles() -> dict[str, dict[str, str | None]]:
    return {
        "auprc": {"favorable_direction": "higher", "role": "efficacy"},
        "auroc": {"favorable_direction": "higher", "role": "efficacy"},
        "brier": {"favorable_direction": "lower", "role": "efficacy"},
        "calibration_intercept": {
            "favorable_direction": None,
            "role": "descriptive_calibration",
        },
        "calibration_slope": {
            "favorable_direction": None,
            "role": "descriptive_calibration",
        },
        "ece": {"favorable_direction": "lower", "role": "efficacy"},
        "log_loss": {"favorable_direction": "lower", "role": "efficacy"},
    }


def _metric_applicability_contract() -> dict[str, object]:
    body: dict[str, object] = {
        "source": "validated_record_truth_kind_contract",
        "rules": {
            metric: {
                "applicable_truth_kinds": ["exact_pre_capture"],
                "structural_unavailability_reasons": {
                    truth_kind: reasons[truth_kind] for truth_kind in sorted(reasons)
                },
            }
            for metric, reasons in sorted(_STRUCTURAL_METRIC_UNAVAILABILITY.items())
        },
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def _normalize_score_group(
    group: object,
    *,
    record_index: int,
    run: Mapping[str, object],
    evidence_status: str,
    evidence_reason: str | None,
    truth_kind: str,
    expected_stratum_type: str,
    expected_label: str,
) -> list[_NormalizedScoreMetric]:
    value = _exact_mapping(
        group,
        frozenset(
            {
                "stratum_type",
                "label",
                "lower",
                "upper",
                "n",
                "metrics",
                "reliability_bins",
            }
        ),
        f"record {record_index} p_pre_zero group",
    )
    if (
        value.get("stratum_type") != expected_stratum_type
        or value.get("label") != expected_label
    ):
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero group identity is invalid"
        )
    for bound in ("lower", "upper"):
        nested = value.get(bound)
        if nested is not None:
            _finite_value(
                nested,
                f"record {record_index} p_pre_zero group {bound}",
            )
    n = value.get("n")
    if type(n) is not int or n < 0:
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero group denominator is invalid"
        )
    metrics = _exact_mapping(
        value.get("metrics"),
        frozenset(_SCORE_METRICS),
        f"record {record_index} p_pre_zero metric denominator",
    )
    truth_reason = {
        "exact_continuous": "undefined_for_continuous_truth",
        "proxy_high_depth": "proxy_truth_not_exact",
        "orthogonal_only": "truth_unavailable",
    }.get(truth_kind)
    result: list[_NormalizedScoreMetric] = []
    source_seed = _model_seed(
        run.get("model_seed"), f"record {record_index} score model_seed"
    )
    for metric_name in _SCORE_METRICS:
        metric = _exact_mapping(
            metrics[metric_name],
            frozenset({"value", "n", "status", "reason"}),
            f"record {record_index} p_pre_zero metric {metric_name}",
        )
        if metric.get("n") != n:
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero metric denominator differs"
            )
        metric_value = metric.get("value")
        metric_status = metric.get("status")
        metric_reason = metric.get("reason")
        if evidence_status != "completed":
            if (
                metric_value is not None
                or metric_status != evidence_status
                or metric_reason != evidence_reason
            ):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero terminal metric differs"
                )
            normalized_value = None
            normalized_status = evidence_status
            normalized_reason = evidence_reason
        elif truth_reason is not None:
            if (
                metric_value is not None
                or metric_status != "unavailable"
                or metric_reason != truth_reason
            ):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero truth reason differs"
                )
            normalized_value = None
            normalized_status = "unavailable"
            normalized_reason = truth_reason
        elif metric_value is None:
            if metric_status != "unavailable":
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero unavailable metric status differs"
                )
            normalized_value = None
            normalized_status = "unavailable"
            normalized_reason = _reason(
                metric_reason,
                f"record {record_index} p_pre_zero unavailable metric reason",
            )
        else:
            normalized_value = _finite_value(
                metric_value,
                f"record {record_index} p_pre_zero metric {metric_name} value",
            )
            if metric_status != "completed" or metric_reason is not None:
                raise FinalAnalysisContractError(
                    f"record {record_index} completed p_pre_zero metric differs"
                )
            normalized_status = "ok"
            normalized_reason = None
        result.append(
            _NormalizedScoreMetric(
                mechanism=str(run["mechanism"]),
                biological_id=str(run["biological_id"]),
                technical_view=str(run["technical_view"]),
                dataset_id=str(run["dataset_id"]),
                method=str(run["method_id"]),
                model_seed=(
                    _DETERMINISTIC_SEED_SENTINEL if source_seed is None else source_seed
                ),
                stratum_type=expected_stratum_type,
                label=expected_label,
                metric=metric_name,
                value=normalized_value,
                n=n,
                status=normalized_status,
                reason=normalized_reason,
            )
        )
    bins = value.get("reliability_bins")
    if not isinstance(bins, list):
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero reliability bins are invalid"
        )
    for bin_index, bin_value in enumerate(bins, start=1):
        bin_row = _exact_mapping(
            bin_value,
            frozenset(
                {
                    "bin",
                    "n",
                    "mean_prediction",
                    "observed_fraction",
                    "wilson_lower",
                    "wilson_upper",
                }
            ),
            f"record {record_index} p_pre_zero reliability bin",
        )
        if (
            bin_row.get("bin") != bin_index
            or type(bin_row.get("n")) is not int
            or bin_row["n"] < 0
        ):
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero reliability bin order differs"
            )
        for field in (
            "mean_prediction",
            "observed_fraction",
            "wilson_lower",
            "wilson_upper",
        ):
            _finite_value(
                bin_row.get(field),
                f"record {record_index} p_pre_zero reliability bin {field}",
            )
    truth_reason = {
        "exact_continuous": "undefined_for_continuous_truth",
        "proxy_high_depth": "proxy_truth_not_exact",
        "orthogonal_only": "truth_unavailable",
    }.get(truth_kind)
    if (evidence_status != "completed" or truth_reason is not None) and bins:
        raise FinalAnalysisContractError(
            f"record {record_index} unavailable p_pre_zero evidence has bins"
        )
    return result


def _normalize_score_evidence(
    records: Sequence[Mapping[str, object]],
) -> tuple[_NormalizedScoreMetric, ...]:
    present = ["p_pre_zero_evidence" in record for record in records]
    if not any(present):
        return ()
    if not all(present):
        raise FinalAnalysisContractError(
            "final records have a partial p_pre_zero evidence denominator"
        )
    normalized: list[_NormalizedScoreMetric] = []
    for record_index, record in enumerate(records):
        run = record.get("run")
        if not isinstance(run, Mapping):
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero run is invalid"
            )
        evidence = _exact_mapping(
            record.get("p_pre_zero_evidence"),
            frozenset(
                {
                    "schema_version",
                    "status",
                    "reason",
                    "identity",
                    "truth_kind",
                    "matrix",
                    "policy",
                    "policy_sha256",
                    "overall",
                    "strata",
                    "evidence_sha256",
                    "storage",
                }
            ),
            f"record {record_index} p_pre_zero evidence",
        )
        if (
            type(evidence.get("schema_version")) is not int
            or evidence.get("schema_version") != 1
        ):
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero schema_version is invalid"
            )
        identity = _exact_mapping(
            evidence.get("identity"),
            _SCORE_IDENTITY_FIELDS,
            f"record {record_index} p_pre_zero identity",
        )
        if any(
            identity.get(field) != run.get(field) for field in _SCORE_IDENTITY_FIELDS
        ):
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero identity differs from its run"
            )
        body = {
            key: value
            for key, value in evidence.items()
            if key not in {"evidence_sha256", "storage"}
        }
        if evidence.get("evidence_sha256") != canonical_sha256(body):
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero payload checksum differs"
            )
        method = run.get("method_id")
        expected_status = (
            run.get("status") if method == "maskimpute" else "not_applicable"
        )
        expected_reason = (
            run.get("reason")
            if method == "maskimpute"
            else "method_does_not_emit_p_pre_zero"
        )
        evidence_status = evidence.get("status")
        evidence_reason = evidence.get("reason")
        if evidence_status != expected_status or evidence_reason != expected_reason:
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero status differs from its run"
            )
        if not isinstance(evidence_status, str) or not evidence_status:
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero status is invalid"
            )
        truth_kind = evidence.get("truth_kind")
        if truth_kind not in _TRUTH_KINDS:
            raise FinalAnalysisContractError(
                f"record {record_index} p_pre_zero truth kind is invalid"
            )
        matrix = _exact_mapping(
            evidence.get("matrix"),
            frozenset({"shape", "dtype", "content_sha256", "semantic_sha256"}),
            f"record {record_index} p_pre_zero matrix",
        )
        storage = _exact_mapping(
            evidence.get("storage"),
            frozenset(
                {
                    "encoding",
                    "compression_level",
                    "path",
                    "compressed_sha256",
                    "compressed_nbytes",
                    "uncompressed_sha256",
                    "uncompressed_nbytes",
                }
            ),
            f"record {record_index} p_pre_zero storage",
        )
        matrix_present = matrix.get("shape") is not None
        if matrix_present:
            shape = matrix.get("shape")
            policy = _exact_mapping(
                evidence.get("policy"),
                _SCORE_POLICY_FIELDS,
                f"record {record_index} p_pre_zero policy",
            )
            if (
                method != "maskimpute"
                or evidence_status not in {"completed", "unavailable"}
                or not isinstance(shape, list)
                or len(shape) != 2
                or any(type(item) is not int or item <= 0 for item in shape)
                or matrix.get("dtype") != "<f8"
                or evidence.get("policy_sha256") != canonical_sha256(dict(policy))
                or storage.get("encoding") != "zlib_raw_f64_v1"
                or storage.get("compression_level") != 6
                or not isinstance(storage.get("path"), str)
                or not storage.get("path")
            ):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero matrix receipt is invalid"
                )
            if (
                type(policy.get("schema_version")) is not int
                or policy.get("schema_version") != 2
                or policy.get("probability_semantics")
                != "pre_capture_count_is_zero_given_observed_counts"
                or policy.get("evaluation_domain") != "observed_zero_entries_only"
            ):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero policy semantics differ"
                )
            for field in (
                "score_artifact_sha256",
                "score_input_sha256",
                "score_config_sha256",
                "calibration_file_sha256",
                "calibration_payload_sha256",
            ):
                _sha256(
                    policy.get(field),
                    f"record {record_index} p_pre_zero policy {field}",
                )
            if policy.get("calibration_file_sha256") == policy.get(
                "calibration_payload_sha256"
            ):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero calibration digest domains coincide"
                )
            for field in (
                "score_source",
                "calibration_algorithm",
                "calibration_scope",
                "calibration_equivalence_reason",
            ):
                _nonempty_string(
                    policy.get(field),
                    f"record {record_index} p_pre_zero policy {field}",
                )
            for field in (
                "content_sha256",
                "semantic_sha256",
            ):
                _sha256(
                    matrix.get(field),
                    f"record {record_index} p_pre_zero matrix {field}",
                )
            for field in ("compressed_sha256", "uncompressed_sha256"):
                _sha256(
                    storage.get(field),
                    f"record {record_index} p_pre_zero storage {field}",
                )
            if any(
                type(storage.get(field)) is not int or storage.get(field) < 0
                for field in ("compressed_nbytes", "uncompressed_nbytes")
            ):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero storage size is invalid"
                )
            assert isinstance(shape, list)
            if storage.get("uncompressed_nbytes") != shape[0] * shape[
                1
            ] * 8 or storage.get("uncompressed_sha256") != matrix.get("content_sha256"):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero storage content binding differs"
                )
        elif (
            any(matrix.get(field) is not None for field in matrix)
            or evidence.get("policy") is not None
            or evidence.get("policy_sha256") is not None
            or any(storage.get(field) is not None for field in storage)
        ):
            raise FinalAnalysisContractError(
                f"record {record_index} absent p_pre_zero matrix is partial"
            )
        elif method == "maskimpute" and evidence_status == "completed":
            raise FinalAnalysisContractError(
                f"record {record_index} completed MaskImpute lacks p_pre_zero matrix"
            )

        normalized.extend(
            _normalize_score_group(
                evidence.get("overall"),
                record_index=record_index,
                run=run,
                evidence_status=evidence_status,
                evidence_reason=(
                    evidence_reason if isinstance(evidence_reason, str) else None
                ),
                truth_kind=str(truth_kind),
                expected_stratum_type="overall",
                expected_label="all_observed_zeros",
            )
        )
        strata = _exact_mapping(
            evidence.get("strata"),
            frozenset({"library_size_quartiles", "truth_expression_bins"}),
            f"record {record_index} p_pre_zero strata",
        )
        for stratum_type in ("library_size_quartiles", "truth_expression_bins"):
            groups = strata.get(stratum_type)
            labels = _SCORE_GROUPS[stratum_type]
            if not isinstance(groups, list) or len(groups) != len(labels):
                raise FinalAnalysisContractError(
                    f"record {record_index} p_pre_zero strata are incomplete"
                )
            for group, label in zip(groups, labels, strict=True):
                normalized.extend(
                    _normalize_score_group(
                        group,
                        record_index=record_index,
                        run=run,
                        evidence_status=evidence_status,
                        evidence_reason=(
                            evidence_reason
                            if isinstance(evidence_reason, str)
                            else None
                        ),
                        truth_kind=str(truth_kind),
                        expected_stratum_type=stratum_type,
                        expected_label=label,
                    )
                )
    return tuple(normalized)


def _score_evidence_report(
    evidence: _NormalizedEvidence,
    score_rows: Sequence[_NormalizedScoreMetric],
) -> dict[str, object]:
    common: dict[str, object] = {
        "expected_groups": {
            name: list(labels) for name, labels in _SCORE_GROUPS.items()
        },
        "family_id": "p_pre_zero_score_metrics",
        "metric_roles": _score_metric_roles(),
        "multiplicity": {
            "family_id": "p_pre_zero_score_metrics",
            "reason": "no_prespecified_pairwise_score_hypotheses",
            "status": "not_applicable",
        },
        "paired_inference": {
            "inference_unit": "biological_draw",
            "reason": "comparator_methods_do_not_emit_p_pre_zero",
            "status": "unavailable",
        },
        "reliability_bins": {
            "reason": "retained_as_record_level_calibration_evidence",
            "status": "not_aggregated",
        },
        "separate_from_metric_family": "protocol_primary_metrics",
    }
    if not score_rows:
        return {
            **common,
            "analytic_status_counts": {},
            "evidence_record_count": 0,
            "group_summaries": [],
            "reason": "p_pre_zero_evidence_not_present",
            "status": "unavailable",
        }

    raw_counts = Counter(
        (row.method, row.metric, row.stratum_type, row.label) for row in score_rows
    )
    reason_counts: dict[tuple[str, str, str, str], Counter[str]] = {}
    values_by_dataset: dict[
        tuple[str, str, str, str, str, str, str, str], list[float]
    ] = {}
    n_by_dataset: dict[tuple[str, str, str, str, str, str, str, str], set[int]] = {}
    for row in score_rows:
        summary_key = (row.method, row.metric, row.stratum_type, row.label)
        if row.status != "ok" and row.reason is not None:
            reason_counts.setdefault(summary_key, Counter())[row.reason] += 1
        dataset_key = (
            row.method,
            row.metric,
            row.stratum_type,
            row.label,
            row.mechanism,
            row.biological_id,
            row.technical_view,
            row.dataset_id,
        )
        n_by_dataset.setdefault(dataset_key, set()).add(row.n)
        if row.status == "ok":
            assert row.value is not None
            values_by_dataset.setdefault(dataset_key, []).append(row.value)
    if any(len(values) != 1 for values in n_by_dataset.values()):
        raise FinalAnalysisContractError(
            "p_pre_zero seed rows disagree on their dataset-view denominator"
        )

    values_by_draw: dict[tuple[str, str, str, str, str, str], list[float]] = {}
    dataset_counts = Counter()
    for key, values in sorted(values_by_dataset.items()):
        (
            method,
            metric,
            stratum_type,
            label,
            mechanism,
            biological_id,
            _view,
            _dataset,
        ) = key
        draw_key = (
            method,
            metric,
            stratum_type,
            label,
            mechanism,
            biological_id,
        )
        values_by_draw.setdefault(draw_key, []).append(_finite_mean(values))
        dataset_counts[(method, metric, stratum_type, label)] += 1
    draw_values: dict[tuple[str, str, str, str], list[float]] = {}
    mechanisms: dict[tuple[str, str, str, str], set[str]] = {}
    for key, values in sorted(values_by_draw.items()):
        method, metric, stratum_type, label, mechanism, _biological_id = key
        summary_key = (method, metric, stratum_type, label)
        draw_values.setdefault(summary_key, []).append(_finite_mean(values))
        mechanisms.setdefault(summary_key, set()).add(mechanism)

    roles = _score_metric_roles()
    summaries: list[dict[str, object]] = []
    for method in evidence.methods:
        for stratum_type, labels in _SCORE_GROUPS.items():
            for label in labels:
                for metric in _SCORE_METRICS:
                    summary_key = (method, metric, stratum_type, label)
                    values = draw_values.get(summary_key, [])
                    denominator_values = [
                        next(iter(nested))
                        for key, nested in n_by_dataset.items()
                        if key[:4] == summary_key
                    ]
                    entry_denominator = {
                        "maximum": (
                            max(denominator_values) if denominator_values else None
                        ),
                        "median": (
                            _finite_median(denominator_values)
                            if denominator_values
                            else None
                        ),
                        "minimum": (
                            min(denominator_values) if denominator_values else None
                        ),
                        "unit": "observed_zero_entries_per_dataset_view",
                    }
                    unavailable = reason_counts.get(summary_key, Counter())
                    base = {
                        "entry_denominator": entry_denominator,
                        "favorable_direction": roles[metric]["favorable_direction"],
                        "label": label,
                        "method": method,
                        "metric": metric,
                        "n_biological_draws": len(values),
                        "n_dataset_views": dataset_counts[summary_key],
                        "n_mechanisms": len(mechanisms.get(summary_key, set())),
                        "n_raw_metric_rows": raw_counts[summary_key],
                        "role": roles[metric]["role"],
                        "stratum_type": stratum_type,
                        "unavailable_reason_counts": {
                            reason: unavailable[reason]
                            for reason in sorted(unavailable)
                        },
                        "unit": "biological_draw",
                    }
                    if values:
                        first = _finite_quantile(values, 0.25)
                        third = _finite_quantile(values, 0.75)
                        summaries.append(
                            {
                                **base,
                                "first_quartile": first,
                                "interquartile_range": third - first,
                                "median": _finite_median(values),
                                "reason": None,
                                "status": "ok",
                                "third_quartile": third,
                            }
                        )
                    else:
                        summaries.append(
                            {
                                **base,
                                "first_quartile": None,
                                "interquartile_range": None,
                                "median": None,
                                "reason": "no_finite_ok_biological_draws",
                                "status": "unavailable",
                                "third_quartile": None,
                            }
                        )
    status_counts = Counter(row.status for row in score_rows)
    return {
        **common,
        "analytic_status_counts": {
            status: status_counts[status] for status in sorted(status_counts)
        },
        "evidence_record_count": len(score_rows) // (9 * len(_SCORE_METRICS)),
        "group_summaries": summaries,
        "reason": None,
        "status": "ok",
    }


def _denominator(
    evidence: _NormalizedEvidence,
    *,
    planned_run_count: int,
    execution_action_denominator: object = None,
) -> dict[str, object]:
    recorded_run_count = len(evidence.run_statuses)
    if (
        type(planned_run_count) is not int
        or planned_run_count <= 0
        or planned_run_count != recorded_run_count
    ):
        raise FinalAnalysisContractError(
            "validated final record denominator is incomplete"
        )
    run_counts = Counter(evidence.run_statuses)
    metric_counts = Counter(row.status for row in evidence.rows)
    run_reasons = Counter(
        reason for reason in evidence.run_reasons if reason is not None
    )
    metric_reasons = Counter(
        row.reason for row in evidence.rows if row.status != "ok" and row.reason
    )
    result: dict[str, object] = {
        "completeness": {
            "complete": True,
            "expected_metric_names": list(evidence.metric_names),
            "metric_row_count": len(evidence.rows),
            "metric_rows_per_run": len(evidence.metric_names),
            "planned_run_count": planned_run_count,
            "recorded_run_count": recorded_run_count,
        },
        "metric_analytic_status_counts": {
            status: metric_counts.get(status, 0)
            for status in _ANALYTIC_STATUSES
            if metric_counts.get(status, 0)
        },
        "metric_unavailable_reason_counts": {
            reason: metric_reasons[reason] for reason in sorted(metric_reasons)
        },
        "run_terminal_status_counts": {
            status: run_counts.get(status, 0) for status in _TERMINAL_STATUSES
        },
        "run_unavailable_reason_counts": {
            reason: run_reasons[reason] for reason in sorted(run_reasons)
        },
    }
    if execution_action_denominator is not None:
        expected = {
            "executed_algorithmic_failure_count",
            "executed_completed_count",
            "executed_run_count",
            "executed_terminal_status_counts",
            "not_applicable_count",
            "status",
        }
        if (
            not isinstance(execution_action_denominator, Mapping)
            or set(execution_action_denominator) != expected
            or execution_action_denominator.get("status") != "validated"
        ):
            raise FinalAnalysisContractError(
                "execution action denominator binding is invalid"
            )
        result["execution_action_denominator"] = dict(execution_action_denominator)
    return result


def _descriptive_summaries(
    evidence: _NormalizedEvidence,
) -> list[dict[str, object]]:
    ok_by_dataset: dict[tuple[str, str, str, str, str, str], list[float]] = {}
    raw_counts = Counter((row.method, row.metric) for row in evidence.rows)
    for row in evidence.rows:
        if row.status != "ok":
            continue
        assert row.value is not None
        key = (
            row.method,
            row.metric,
            row.mechanism,
            row.biological_id,
            row.technical_view,
            row.dataset_id,
        )
        ok_by_dataset.setdefault(key, []).append(row.value)

    by_draw: dict[tuple[str, str, str, str], list[float]] = {}
    dataset_counts = Counter()
    for key, values in sorted(ok_by_dataset.items()):
        method, metric, mechanism, biological_id, _view, _dataset = key
        draw_key = (method, metric, mechanism, biological_id)
        by_draw.setdefault(draw_key, []).append(_finite_mean(values))
        dataset_counts[(method, metric)] += 1

    draw_values: dict[tuple[str, str], list[float]] = {}
    mechanisms: dict[tuple[str, str], set[str]] = {}
    for (method, metric, mechanism, _biological_id), values in sorted(by_draw.items()):
        method_metric = (method, metric)
        draw_values.setdefault(method_metric, []).append(_finite_mean(values))
        mechanisms.setdefault(method_metric, set()).add(mechanism)

    summaries: list[dict[str, object]] = []
    for method in evidence.methods:
        for metric in evidence.metric_names:
            key = (method, metric)
            values = draw_values.get(key, [])
            if values:
                first = _finite_quantile(values, 0.25)
                third = _finite_quantile(values, 0.75)
                summaries.append(
                    {
                        "first_quartile": first,
                        "interquartile_range": third - first,
                        "median": _finite_median(values),
                        "method": method,
                        "metric": metric,
                        "n_biological_draws": len(values),
                        "n_dataset_views": dataset_counts[key],
                        "n_mechanisms": len(mechanisms[key]),
                        "n_raw_metric_rows": raw_counts[key],
                        "reason": None,
                        "status": "ok",
                        "third_quartile": third,
                        "unit": "biological_draw",
                    }
                )
            else:
                summaries.append(
                    {
                        "first_quartile": None,
                        "interquartile_range": None,
                        "median": None,
                        "method": method,
                        "metric": metric,
                        "n_biological_draws": 0,
                        "n_dataset_views": 0,
                        "n_mechanisms": 0,
                        "n_raw_metric_rows": raw_counts[key],
                        "reason": "no_finite_ok_biological_draws",
                        "status": "unavailable",
                        "third_quartile": None,
                        "unit": "biological_draw",
                    }
                )
    return summaries


def _empty_comparison(
    *,
    candidate: str,
    comparator: str,
    metric: str,
    direction_available: bool,
    reason: str,
) -> dict[str, object]:
    return {
        "biological_draw_losses": 0,
        "biological_draw_ties": 0,
        "biological_draw_wins": 0,
        "bootstrap": {
            "distribution_sha256": None,
            "replicates_available": 0,
            "replicates_requested": _BOOTSTRAP_REPLICATES,
            "seed": _BOOTSTRAP_SEED,
        },
        "candidate_method_id": candidate,
        "ci_95_lower": None,
        "ci_95_upper": None,
        "comparator_method_id": comparator,
        "direction_source": (
            "validated_frozen_metric_direction_contract"
            if direction_available
            else None
        ),
        "exclusions": {},
        "favorable_direction": "lower" if direction_available else None,
        "holm_adjusted_p_value": None,
        "holm_family_id": "protocol_primary_metrics",
        "holm_hypothesis_count": 0,
        "holm_reason": "raw_p_value_unavailable",
        "holm_status": "unavailable",
        "median_relative_effect": None,
        "metric": metric,
        "n_independent_biological_draws": 0,
        "n_paired_dataset_views": 0,
        "n_raw_metric_rows": 0,
        "probability_of_improvement": None,
        "reason": reason,
        "status": "unavailable",
        "two_sided_sign_probability": None,
    }


def _paired_comparisons(
    evidence: _NormalizedEvidence,
    *,
    candidate: str,
    primary_metrics: Sequence[str],
    lower_better_metrics: frozenset[str],
) -> list[dict[str, object]]:
    statistics_rows = [row.statistics_row() for row in evidence.rows]
    comparisons: list[dict[str, object]] = []
    for comparator in (method for method in evidence.methods if method != candidate):
        family: list[dict[str, object]] = []
        for metric in primary_metrics:
            direction_available = metric in lower_better_metrics
            if not direction_available:
                family.append(
                    _empty_comparison(
                        candidate=candidate,
                        comparator=comparator,
                        metric=metric,
                        direction_available=False,
                        reason="metric_direction_not_declared_lower",
                    )
                )
                continue
            result = hierarchical_paired_bootstrap(
                statistics_rows,
                candidate,
                comparator,
                metric,
                n_boot=_BOOTSTRAP_REPLICATES,
                seed=_BOOTSTRAP_SEED,
            )
            if result.n_independent_draws == 0 or result.median_effect is None:
                item = _empty_comparison(
                    candidate=candidate,
                    comparator=comparator,
                    metric=metric,
                    direction_available=True,
                    reason="no_paired_biological_draws",
                )
                item["bootstrap"] = {
                    "distribution_sha256": result.bootstrap_checksum,
                    "replicates_available": len(result.bootstrap_distribution),
                    "replicates_requested": _BOOTSTRAP_REPLICATES,
                    "seed": _BOOTSTRAP_SEED,
                }
                item["exclusions"] = {
                    name: result.exclusions[name] for name in sorted(result.exclusions)
                }
                item["n_raw_metric_rows"] = result.n_raw_rows
            else:
                bootstrap_available = len(result.bootstrap_distribution)
                interval_available = (
                    result.ci_lower is not None
                    and result.ci_upper is not None
                    and result.probability_effect_lt_zero is not None
                    and result.two_sided_sign_probability is not None
                )
                item = {
                    "biological_draw_losses": result.n_losses,
                    "biological_draw_ties": result.n_ties,
                    "biological_draw_wins": result.n_wins,
                    "bootstrap": {
                        "distribution_sha256": result.bootstrap_checksum,
                        "replicates_available": bootstrap_available,
                        "replicates_requested": _BOOTSTRAP_REPLICATES,
                        "seed": _BOOTSTRAP_SEED,
                    },
                    "candidate_method_id": candidate,
                    "ci_95_lower": result.ci_lower,
                    "ci_95_upper": result.ci_upper,
                    "comparator_method_id": comparator,
                    "direction_source": ("validated_frozen_metric_direction_contract"),
                    "exclusions": {
                        name: result.exclusions[name]
                        for name in sorted(result.exclusions)
                    },
                    "favorable_direction": "lower",
                    "holm_adjusted_p_value": None,
                    "holm_family_id": "protocol_primary_metrics",
                    "holm_hypothesis_count": 0,
                    "holm_reason": "raw_p_value_unavailable",
                    "holm_status": "unavailable",
                    "median_relative_effect": result.median_effect,
                    "metric": metric,
                    "n_independent_biological_draws": result.n_independent_draws,
                    "n_paired_dataset_views": result.n_paired_views,
                    "n_raw_metric_rows": result.n_raw_rows,
                    "probability_of_improvement": (result.probability_effect_lt_zero),
                    "reason": (
                        None
                        if interval_available
                        else "bootstrap_distribution_unavailable"
                    ),
                    "status": "ok" if interval_available else "unavailable",
                    "two_sided_sign_probability": (result.two_sided_sign_probability),
                }
            family.append(item)

        raw = [
            value
            if isinstance(value := item["two_sided_sign_probability"], float)
            else None
            for item in family
        ]
        adjusted = holm_adjust(raw)
        hypothesis_count = sum(value is not None for value in raw)
        for item, adjusted_value in zip(family, adjusted, strict=True):
            item["holm_hypothesis_count"] = hypothesis_count
            if adjusted_value is not None:
                item["holm_adjusted_p_value"] = adjusted_value
                item["holm_reason"] = None
                item["holm_status"] = "ok"
        comparisons.extend(family)
    return comparisons


def _variance_component(
    value: float | None,
    *,
    identifiable_groups: int,
    nonrepresentable_groups: int,
    unavailable_reason: str,
    no_finite_rows: bool,
) -> dict[str, object]:
    if value is not None:
        return {
            "estimate": value,
            "n_identifiable_groups": identifiable_groups,
            "reason": None,
            "status": "ok",
        }
    if no_finite_rows:
        reason = "no_finite_ok_rows"
    elif nonrepresentable_groups:
        reason = "nonrepresentable_float_variance"
    else:
        reason = unavailable_reason
    return {
        "estimate": None,
        "n_identifiable_groups": identifiable_groups,
        "reason": reason,
        "status": "unavailable",
    }


def _variance_components(
    evidence: _NormalizedEvidence,
) -> list[dict[str, object]]:
    report = summarize_seed_variance([row.statistics_row() for row in evidence.rows])
    rows: list[dict[str, object]] = []
    for summary in report.summaries:
        exclusions = {
            name: summary.exclusions[name] for name in sorted(summary.exclusions)
        }
        no_finite_rows = summary.n_seed_groups == 0
        rows.append(
            {
                "between_biological_draw_variance": _variance_component(
                    summary.between_biological_draw_variance,
                    identifiable_groups=summary.n_between_draw_mechanisms,
                    nonrepresentable_groups=exclusions[
                        "nonrepresentable_between_draw_variances"
                    ],
                    unavailable_reason=(
                        "fewer_than_two_biological_draws_per_mechanism"
                    ),
                    no_finite_rows=no_finite_rows,
                ),
                "between_technical_view_variance": _variance_component(
                    summary.between_view_variance,
                    identifiable_groups=summary.n_view_variance_draws,
                    nonrepresentable_groups=exclusions[
                        "nonrepresentable_between_view_variances"
                    ],
                    unavailable_reason=(
                        "fewer_than_two_technical_views_per_biological_draw"
                    ),
                    no_finite_rows=no_finite_rows,
                ),
                "denominators": {
                    "n_between_draw_mechanisms": (summary.n_between_draw_mechanisms),
                    "n_biological_draws": summary.n_biological_draws,
                    "n_mechanisms": summary.n_mechanisms,
                    "n_seed_groups": summary.n_seed_groups,
                    "n_seed_variance_groups": summary.n_seed_variance_groups,
                    "n_view_variance_draws": summary.n_view_variance_draws,
                },
                "exclusions": exclusions,
                "inference_unit": "biological_draw",
                "method": summary.method,
                "metric": summary.metric,
                "within_dataset_view_seed_variance": _variance_component(
                    summary.within_draw_seed_variance,
                    identifiable_groups=summary.n_seed_variance_groups,
                    nonrepresentable_groups=exclusions[
                        "nonrepresentable_within_seed_variances"
                    ],
                    unavailable_reason=("fewer_than_two_seed_levels_per_dataset_view"),
                    no_finite_rows=no_finite_rows,
                ),
            }
        )
    return rows


def _pareto_report(
    evidence: _NormalizedEvidence,
    summaries: Sequence[Mapping[str, object]],
    *,
    primary_metrics: Sequence[str],
    lower_better_metrics: frozenset[str],
) -> dict[str, object]:
    core = [metric for metric in primary_metrics if metric in lower_better_metrics]
    excluded = [
        {
            "metric": metric,
            "reason": "not_prespecified_lower_better_reconstruction_metric",
        }
        for metric in primary_metrics
        if metric not in lower_better_metrics
    ]
    base: dict[str, object] = {
        "complete_method_count": 0,
        "core_metrics": core,
        "direction_source": (
            "validated_frozen_metric_direction_contract"
            if lower_better_metrics
            else None
        ),
        "excluded_primary_metrics": excluded,
        "methods": [],
    }
    if not core:
        return {
            **base,
            "reason": "no_explicit_lower_better_core_metrics",
            "status": "unavailable",
        }

    summary_by_key = {(row.get("method"), row.get("metric")): row for row in summaries}
    evidence_by_key: dict[tuple[str, str], list[_NormalizedMetric]] = {}
    expected_views_by_metric: dict[str, set[tuple[str, str, str, str]]] = {}
    expected_draws_by_metric: dict[str, set[tuple[str, str]]] = {}

    def structural_reason(row: _NormalizedMetric) -> str | None:
        reasons = _STRUCTURAL_METRIC_UNAVAILABILITY.get(row.metric, {})
        return reasons.get(row.truth_kind)

    for row in evidence.rows:
        if row.metric not in core:
            continue
        evidence_by_key.setdefault((row.method, row.metric), []).append(row)
        if structural_reason(row) is not None:
            continue
        expected_views_by_metric.setdefault(row.metric, set()).add(
            (
                row.mechanism,
                row.biological_id,
                row.technical_view,
                row.dataset_id,
            )
        )
        expected_draws_by_metric.setdefault(row.metric, set()).add(
            (row.mechanism, row.biological_id)
        )

    def complete_metric_denominator(method: str, metric: str) -> bool:
        rows = evidence_by_key.get((method, metric), [])
        summary = summary_by_key[(method, metric)]
        applicable_rows = [row for row in rows if structural_reason(row) is None]
        structural_rows = [
            (row, reason)
            for row in rows
            if (reason := structural_reason(row)) is not None
        ]
        observed_views = {
            (
                row.mechanism,
                row.biological_id,
                row.technical_view,
                row.dataset_id,
            )
            for row in applicable_rows
        }
        observed_draws = {(row.mechanism, row.biological_id) for row in applicable_rows}
        return bool(applicable_rows) and (
            all(row.status == "ok" for row in applicable_rows)
            and all(
                row.status == "unavailable" and row.reason == reason
                for row, reason in structural_rows
            )
            and observed_views == expected_views_by_metric.get(metric, set())
            and observed_draws == expected_draws_by_metric.get(metric, set())
            and summary.get("status") == "ok"
            and summary.get("n_raw_metric_rows") == len(rows)
            and summary.get("n_dataset_views") == len(observed_views)
            and summary.get("n_biological_draws") == len(observed_draws)
        )

    complete_values: dict[str, tuple[float, ...]] = {}
    method_rows: list[dict[str, object]] = []
    for method in evidence.methods:
        missing = [
            metric for metric in core if not complete_metric_denominator(method, metric)
        ]
        if missing:
            method_rows.append(
                {
                    "dominated_by": [],
                    "method": method,
                    "missing_metrics": missing,
                    "non_dominated": None,
                    "reason": "incomplete_core_metric_denominator",
                    "status": "unavailable",
                }
            )
            continue
        values = tuple(
            float(summary_by_key[(method, metric)]["median"]) for metric in core
        )
        complete_values[method] = values

    for method, values in sorted(complete_values.items()):
        dominators = []
        for other, other_values in sorted(complete_values.items()):
            if other == method:
                continue
            if all(
                left <= right for left, right in zip(other_values, values, strict=True)
            ) and any(
                left < right for left, right in zip(other_values, values, strict=True)
            ):
                dominators.append(other)
        method_rows.append(
            {
                "dominated_by": dominators,
                "method": method,
                "missing_metrics": [],
                "non_dominated": not dominators,
                "reason": None,
                "status": "ok",
            }
        )
    method_rows.sort(key=lambda row: str(row["method"]))
    base["complete_method_count"] = len(complete_values)
    base["methods"] = method_rows
    if len(complete_values) < 2:
        return {
            **base,
            "reason": "fewer_than_two_methods_have_complete_core_metrics",
            "status": "unavailable",
        }
    return {**base, "reason": None, "status": "ok"}


def _unavailable_reconstruction_claim_gate(
    *,
    candidate: str,
    primary_metrics: Sequence[str],
    scheduled_methods: Sequence[str],
    numerical_methods: Sequence[str],
    reason: str,
) -> dict[str, object]:
    return {
        "status": "unavailable",
        "reason": reason,
        "candidate_method_id": candidate,
        "scheduled_same_input_ids": list(scheduled_methods),
        "numerical_comparison_population_ids": list(numerical_methods),
        "denominator": {
            "status": "unavailable",
            "reason": reason,
            "method_ids": [candidate, *numerical_methods],
            "metric_ids": list(
                dict.fromkeys((*_CLAIM_RANK_METRICS, *_CLAIM_PARETO_METRICS))
            ),
            "view_collapse": "seed_mean_then_paired_view_mean",
        },
        "rank_gates": [
            {
                "metric": metric,
                "status": "unavailable",
                "reason": reason,
                "threshold": 2.0,
                "median_biological_draw_rank": None,
                "n_biological_draws": 0,
                "mechanism_median_ranks": [],
            }
            for metric in _CLAIM_RANK_METRICS
        ],
        "pareto_gate": {
            "status": "unavailable",
            "reason": "incomplete_metric_denominator",
            "dimensions": list(_CLAIM_PARETO_METRICS),
            "complete_method_ids": [],
            "non_dominated": None,
            "dominated_by": [],
        },
        "draw_collapsed_method_summaries": [],
        "strongest_applicable_comparators": [
            {
                "metric": metric,
                "status": "unavailable",
                "reason": reason,
                "method_id": None,
                "tied_method_ids": [],
                "median": None,
                "n_biological_draws": 0,
            }
            for metric in primary_metrics
        ],
    }


def _claim_structural_reason(row: _NormalizedMetric) -> str | None:
    return _STRUCTURAL_METRIC_UNAVAILABILITY.get(row.metric, {}).get(row.truth_kind)


def _collapse_final_claim_draws(
    evidence: _NormalizedEvidence,
    *,
    method_ids: Sequence[str],
    metric_ids: Sequence[str],
) -> (
    tuple[
        dict[tuple[str, str, str, str], float],
        list[dict[str, object]],
    ]
    | None
):
    methods = tuple(method_ids)
    metrics = tuple(metric_ids)
    expected_seeds: dict[str, frozenset[int | None]] = {}
    for method in methods:
        seeds = frozenset(
            row.source_model_seed for row in evidence.rows if row.method == method
        )
        if not seeds:
            return None
        expected_seeds[method] = seeds

    grouped: dict[tuple[str, str, str, str, str, str], list[_NormalizedMetric]] = {}
    for row in evidence.rows:
        if row.method not in methods or row.metric not in metrics:
            continue
        key = (
            row.method,
            row.metric,
            row.mechanism,
            row.biological_id,
            row.technical_view,
            row.dataset_id,
        )
        grouped.setdefault(key, []).append(row)

    draw_values: dict[tuple[str, str, str, str], float] = {}
    for metric in metrics:
        expected_all_views: set[tuple[str, str, str, str]] | None = None
        expected_applicable_views: set[tuple[str, str, str, str]] | None = None
        expected_draws: set[tuple[str, str]] | None = None
        for method in methods:
            method_groups = {
                (mechanism, biological_id, technical_view, dataset_id): rows
                for (
                    grouped_method,
                    grouped_metric,
                    mechanism,
                    biological_id,
                    technical_view,
                    dataset_id,
                ), rows in grouped.items()
                if grouped_method == method and grouped_metric == metric
            }
            if not method_groups:
                return None
            all_views = set(method_groups)
            if expected_all_views is None:
                expected_all_views = all_views
            elif all_views != expected_all_views:
                return None

            view_values: dict[tuple[str, str, str, str], float] = {}
            for view_key, rows in method_groups.items():
                if (
                    frozenset(row.source_model_seed for row in rows)
                    != (expected_seeds[method])
                ):
                    return None
                structural_reasons = tuple(
                    _claim_structural_reason(row) for row in rows
                )
                if all(reason is not None for reason in structural_reasons):
                    if any(
                        row.status != "unavailable" or row.reason != structural_reason
                        for row, structural_reason in zip(
                            rows, structural_reasons, strict=True
                        )
                    ):
                        return None
                    continue
                if any(reason is not None for reason in structural_reasons):
                    return None
                if any(row.status != "ok" or row.value is None for row in rows):
                    return None
                view_values[view_key] = _finite_mean(
                    [float(row.value) for row in rows if row.value is not None]
                )

            applicable_views = set(view_values)
            if not applicable_views:
                return None
            if expected_applicable_views is None:
                expected_applicable_views = applicable_views
            elif applicable_views != expected_applicable_views:
                return None

            views_by_draw: dict[tuple[str, str], dict[str, float]] = {}
            for (
                mechanism,
                biological_id,
                technical_view,
                _dataset_id,
            ), value in sorted(view_values.items()):
                draw = (mechanism, biological_id)
                by_view = views_by_draw.setdefault(draw, {})
                if technical_view in by_view:
                    return None
                by_view[technical_view] = value
            if any(
                set(values) != {"moderate", "severe"}
                for values in views_by_draw.values()
            ):
                return None
            draw_keys = set(views_by_draw)
            if expected_draws is None:
                expected_draws = draw_keys
            elif draw_keys != expected_draws:
                return None
            for (mechanism, biological_id), values in views_by_draw.items():
                draw_values[(method, metric, mechanism, biological_id)] = _finite_mean(
                    [values["moderate"], values["severe"]]
                )

    summaries: list[dict[str, object]] = []
    for method in methods:
        for metric in metrics:
            values = [
                value
                for (
                    observed_method,
                    observed_metric,
                    _mechanism,
                    _biological_id,
                ), value in draw_values.items()
                if observed_method == method and observed_metric == metric
            ]
            if not values:
                return None
            summaries.append(
                {
                    "status": "complete",
                    "method_id": method,
                    "metric": metric,
                    "median": _finite_median(values),
                    "n_biological_draws": len(values),
                }
            )
    return draw_values, summaries


def _claim_average_rank(target: float, comparators: Sequence[float]) -> float:
    below = sum(value < target for value in comparators)
    tied = sum(value == target for value in comparators)
    return 1.0 + below + tied / 2.0


def _reconstruction_claim_gate(
    evidence: _NormalizedEvidence,
    *,
    candidate: str,
    primary_metrics: Sequence[str],
    selection_contract: Mapping[str, object],
    metric_direction_contract: Mapping[str, object],
) -> dict[str, object]:
    raw_scheduled = selection_contract.get("scheduled_same_input_ids")
    raw_comparators = selection_contract.get("numerical_comparison_population_ids")
    if raw_scheduled is None or raw_comparators is None:
        return _unavailable_reconstruction_claim_gate(
            candidate=candidate,
            primary_metrics=primary_metrics,
            scheduled_methods=(),
            numerical_methods=(),
            reason="comparison_population_authority_unavailable",
        )
    if (
        not isinstance(raw_scheduled, list)
        or not raw_scheduled
        or any(not isinstance(value, str) or not value for value in raw_scheduled)
        or len(set(raw_scheduled)) != len(raw_scheduled)
        or not isinstance(raw_comparators, list)
        or not raw_comparators
        or any(not isinstance(value, str) or not value for value in raw_comparators)
        or len(set(raw_comparators)) != len(raw_comparators)
        or candidate in raw_comparators
        or any(value not in raw_scheduled for value in raw_comparators)
    ):
        raise FinalAnalysisContractError(
            "selection comparison population authority is invalid"
        )
    scheduled = tuple(raw_scheduled)
    comparators = tuple(raw_comparators)

    authority = metric_direction_contract.get("authority")
    if (
        metric_direction_contract.get("status") != "validated"
        or not isinstance(authority, Mapping)
        or authority.get("rank_metrics") is None
        or authority.get("pareto_metrics") is None
    ):
        return _unavailable_reconstruction_claim_gate(
            candidate=candidate,
            primary_metrics=primary_metrics,
            scheduled_methods=scheduled,
            numerical_methods=comparators,
            reason="frozen_claim_metric_authority_unavailable",
        )
    if (
        tuple(authority.get("rank_metrics", ())) != _CLAIM_RANK_METRICS
        or tuple(authority.get("pareto_metrics", ())) != _CLAIM_PARETO_METRICS
    ):
        raise FinalAnalysisContractError(
            "frozen final claim metrics differ from selection authority"
        )

    methods = (candidate, *comparators)
    required_metrics = tuple(
        dict.fromkeys((*_CLAIM_RANK_METRICS, *_CLAIM_PARETO_METRICS, *primary_metrics))
    )
    if any(method not in evidence.methods for method in methods) or any(
        metric not in evidence.metric_names for metric in required_metrics
    ):
        return _unavailable_reconstruction_claim_gate(
            candidate=candidate,
            primary_metrics=primary_metrics,
            scheduled_methods=scheduled,
            numerical_methods=comparators,
            reason="incomplete_final_claim_denominator",
        )
    collapsed = _collapse_final_claim_draws(
        evidence,
        method_ids=methods,
        metric_ids=required_metrics,
    )
    if collapsed is None:
        return _unavailable_reconstruction_claim_gate(
            candidate=candidate,
            primary_metrics=primary_metrics,
            scheduled_methods=scheduled,
            numerical_methods=comparators,
            reason="incomplete_final_claim_denominator",
        )
    draw_values, summaries = collapsed

    rank_gates: list[dict[str, object]] = []
    for metric in _CLAIM_RANK_METRICS:
        candidate_draws = sorted(
            (
                mechanism,
                biological_id,
                value,
            )
            for (
                method,
                observed_metric,
                mechanism,
                biological_id,
            ), value in draw_values.items()
            if method == candidate and observed_metric == metric
        )
        ranks: list[float] = []
        by_mechanism: dict[str, list[float]] = {}
        for mechanism, biological_id, value in candidate_draws:
            comparator_values = [
                draw_values[(method, metric, mechanism, biological_id)]
                for method in comparators
            ]
            rank = _claim_average_rank(value, comparator_values)
            ranks.append(rank)
            by_mechanism.setdefault(mechanism, []).append(rank)
        median_rank = _finite_median(ranks)
        passed = median_rank <= 2.0
        rank_gates.append(
            {
                "metric": metric,
                "status": "passed" if passed else "failed",
                "reason": None if passed else "median_rank_exceeds_two",
                "threshold": 2.0,
                "median_biological_draw_rank": median_rank,
                "n_biological_draws": len(ranks),
                "mechanism_median_ranks": [
                    {
                        "mechanism": mechanism,
                        "median_biological_draw_rank": _finite_median(values),
                        "n_biological_draws": len(values),
                    }
                    for mechanism, values in sorted(by_mechanism.items())
                ],
            }
        )

    summary_lookup = {
        (str(row["method_id"]), str(row["metric"])): row for row in summaries
    }
    candidate_vector = tuple(
        float(summary_lookup[(candidate, metric)]["median"])
        for metric in _CLAIM_PARETO_METRICS
    )
    dominators = []
    for comparator in comparators:
        vector = tuple(
            float(summary_lookup[(comparator, metric)]["median"])
            for metric in _CLAIM_PARETO_METRICS
        )
        if all(
            left <= right for left, right in zip(vector, candidate_vector, strict=True)
        ) and any(
            left < right for left, right in zip(vector, candidate_vector, strict=True)
        ):
            dominators.append(comparator)
    dominators.sort()
    pareto_passed = not dominators
    pareto_gate: dict[str, object] = {
        "status": "passed" if pareto_passed else "failed",
        "reason": None if pareto_passed else "candidate_is_pareto_dominated",
        "dimensions": list(_CLAIM_PARETO_METRICS),
        "complete_method_ids": list(methods),
        "non_dominated": pareto_passed,
        "dominated_by": dominators,
    }

    strongest: list[dict[str, object]] = []
    for metric in primary_metrics:
        candidates = sorted(
            (
                float(summary_lookup[(method, metric)]["median"]),
                method,
                int(summary_lookup[(method, metric)]["n_biological_draws"]),
            )
            for method in comparators
        )
        best_median = candidates[0][0]
        tied = sorted(
            method for median, method, _count in candidates if median == best_median
        )
        selected = tied[0]
        selected_count = next(
            count
            for median, method, count in candidates
            if median == best_median and method == selected
        )
        strongest.append(
            {
                "metric": metric,
                "status": "ok",
                "reason": None,
                "method_id": selected,
                "tied_method_ids": tied,
                "median": best_median,
                "n_biological_draws": selected_count,
            }
        )

    failed = any(row["status"] == "failed" for row in rank_gates) or not (pareto_passed)
    return {
        "status": "failed" if failed else "passed",
        "reason": "final_reconstruction_gate_failed" if failed else None,
        "candidate_method_id": candidate,
        "scheduled_same_input_ids": list(scheduled),
        "numerical_comparison_population_ids": list(comparators),
        "denominator": {
            "status": "complete",
            "reason": None,
            "method_ids": list(methods),
            "metric_ids": list(required_metrics),
            "view_collapse": "seed_mean_then_paired_view_mean",
        },
        "rank_gates": rank_gates,
        "pareto_gate": pareto_gate,
        "draw_collapsed_method_summaries": summaries,
        "strongest_applicable_comparators": strongest,
    }


def build_final_analysis(
    records: Sequence[Mapping[str, object]],
    *,
    protocol: Mapping[str, object],
    selection_contract: Mapping[str, object],
    input_bindings: Mapping[str, object],
) -> dict[str, object]:
    """Build the fixed-policy report from already validated final records."""

    for name, value in (
        ("protocol", protocol),
        ("selection_contract", selection_contract),
        ("input_bindings", input_bindings),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
    primary = protocol.get("primary_metrics")
    if (
        not isinstance(primary, list)
        or not primary
        or any(not isinstance(metric, str) or not metric for metric in primary)
        or len(set(primary)) != len(primary)
    ):
        raise FinalAnalysisContractError(
            "protocol primary_metrics must be a nonempty unique string list"
        )
    candidate = _nonempty_string(
        selection_contract.get("candidate_method_id"),
        "selection contract candidate_method_id",
    )
    evidence = _normalize_records(records)
    score_rows = _normalize_score_evidence(records)
    metric_direction_contract, lower_better_metrics = (
        _validated_metric_direction_contract(input_bindings)
    )
    if candidate not in evidence.methods:
        raise FinalAnalysisContractError(
            "candidate method is absent from final records"
        )
    if any(metric not in evidence.metric_names for metric in primary):
        raise FinalAnalysisContractError(
            "protocol primary metrics are absent from the final metric denominator"
        )
    planned = input_bindings.get("planned_run_count")
    if type(planned) is not int or planned <= 0:
        raise FinalAnalysisContractError(
            "input bindings planned_run_count must be a positive integer"
        )

    summaries = _descriptive_summaries(evidence)
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "candidate_method_id": candidate,
        "input_bindings": dict(input_bindings),
        "analysis_policy": {
            "analytic_status_normalization": {
                "completed": "ok",
                "preserved_terminal_statuses": [
                    "failed",
                    "timeout",
                    "resource_exceeded",
                    "unavailable",
                ],
            },
            "bootstrap_replicates": _BOOTSTRAP_REPLICATES,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "declared_metric_family": {
                "id": "protocol_primary_metrics",
                "metrics": list(primary),
            },
            "descriptive_unit": "biological_draw",
            "deterministic_seed_encoding": {
                "analytic_sentinel": _DETERMINISTIC_SEED_SENTINEL,
                "meaning": "single_repeated_measurement_level",
                "source_value": None,
            },
            "inference_unit": "biological_draw",
            "metric_direction_contract": metric_direction_contract,
            "metric_applicability_contract": _metric_applicability_contract(),
            "technical_views_are_repeated_measurements": True,
            "model_seeds_are_repeated_measurements": True,
        },
        "denominator": _denominator(
            evidence,
            planned_run_count=planned,
            execution_action_denominator=input_bindings.get(
                "execution_action_denominator"
            ),
        ),
        "reconstruction_claim_gate": _reconstruction_claim_gate(
            evidence,
            candidate=candidate,
            primary_metrics=primary,
            selection_contract=selection_contract,
            metric_direction_contract=metric_direction_contract,
        ),
        "descriptive_summaries": summaries,
        "paired_comparisons": _paired_comparisons(
            evidence,
            candidate=candidate,
            primary_metrics=primary,
            lower_better_metrics=lower_better_metrics,
        ),
        "variance_components": _variance_components(evidence),
        "pareto": _pareto_report(
            evidence,
            summaries,
            primary_metrics=primary,
            lower_better_metrics=lower_better_metrics,
        ),
        "score_evidence": _score_evidence_report(evidence, score_rows),
    }
    return {**body, "analysis_sha256": canonical_sha256(body)}


def _read_json_file(
    path: Path,
    name: str,
    *,
    require_canonical: bool,
) -> tuple[dict[str, object], bytes]:
    from .final_runner import FinalRunnerContractError, _read_unique_file, _strict_json

    try:
        raw = _read_unique_file(path, name)
        value = _strict_json(raw, name)
    except FinalRunnerContractError as error:
        raise FinalAnalysisContractError(str(error)) from error
    if require_canonical and raw != _canonical_bytes(value) + b"\n":
        raise FinalAnalysisContractError(f"{name} is not canonical JSON")
    return value, raw


def _authority_file(
    repository: Path,
    relative_value: object,
    name: str,
) -> Path:
    if (
        not isinstance(relative_value, str)
        or not relative_value
        or "\\" in relative_value
    ):
        raise FinalAnalysisContractError(f"{name} path is invalid")
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != relative_value
    ):
        raise FinalAnalysisContractError(f"{name} path is invalid")
    candidate = repository.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository)
    except (OSError, ValueError) as error:
        raise FinalAnalysisContractError(f"{name} path is invalid") from error
    if candidate.is_symlink() or resolved != candidate.absolute():
        raise FinalAnalysisContractError(f"{name} must be a direct regular file")
    return candidate


def _git_object_bytes(repository: Path, method_commit: str) -> tuple[str, bytes]:
    from .study import _git_environment

    source_path = "maskimpute_benchmark/selection.py"
    environment = _git_environment()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", f"{method_commit}:{source_path}"],
            cwd=repository,
            check=False,
            capture_output=True,
            env=environment,
        )
        blob_oid = revision.stdout.decode("ascii").strip()
        if revision.returncode != 0 or _GIT_OID.fullmatch(blob_oid) is None:
            raise FinalAnalysisContractError(
                "method-commit selection source is unavailable"
            )
        content = subprocess.run(
            ["git", "cat-file", "blob", blob_oid],
            cwd=repository,
            check=False,
            capture_output=True,
            env=environment,
        )
        if content.returncode != 0:
            raise FinalAnalysisContractError(
                "method-commit selection source is unavailable"
            )
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        raise FinalAnalysisContractError(
            "method-commit selection source is unavailable"
        ) from error
    return blob_oid, content.stdout


def _source_metric_tuple(tree: ast.Module, name: str) -> tuple[str, ...]:
    matches: list[tuple[str, ...]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as error:
            raise FinalAnalysisContractError(
                "method-commit metric direction declaration is not literal"
            ) from error
        if (
            not isinstance(value, tuple)
            or not value
            or any(not isinstance(metric, str) or not metric for metric in value)
            or len(set(value)) != len(value)
        ):
            raise FinalAnalysisContractError(
                "method-commit metric direction declaration is invalid"
            )
        matches.append(value)
    if len(matches) != 1:
        raise FinalAnalysisContractError(
            "method-commit metric direction declaration is ambiguous"
        )
    return matches[0]


def _source_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise FinalAnalysisContractError(
            "method-commit metric direction implementation is ambiguous"
        )
    return matches[0]


def _contains_named_comparison(
    function: ast.FunctionDef,
    left: str,
    operator: type[ast.cmpop],
    right: str,
) -> bool:
    return any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == left
        and len(node.ops) == 1
        and isinstance(node.ops[0], operator)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Name)
        and node.comparators[0].id == right
        for node in ast.walk(function)
    )


def _frozen_metric_direction_contract(
    repository: Path,
    frozen_method: Mapping[str, object],
    freeze: Mapping[str, object],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    selected_assessment = frozen_method.get("selected_assessment")
    gate_table = frozen_method.get("selection_gate_table")
    if selected_assessment is None and gate_table is None:
        return _unavailable_metric_direction_contract(
            "frozen_selection_gate_declaration_absent"
        )
    if not isinstance(selected_assessment, Mapping) or not isinstance(gate_table, list):
        raise FinalAnalysisContractError(
            "frozen metric direction gate declaration is malformed"
        )
    selected_id = _nonempty_string(
        frozen_method.get("selected_configuration_id"),
        "frozen selected configuration_id",
    )
    if selected_assessment.get("configuration_id") != selected_id:
        raise FinalAnalysisContractError("frozen selected assessment identity differs")
    matching_assessments = [
        row
        for row in gate_table
        if isinstance(row, Mapping) and row.get("configuration_id") == selected_id
    ]
    if len(matching_assessments) != 1 or dict(matching_assessments[0]) != dict(
        selected_assessment
    ):
        raise FinalAnalysisContractError(
            "frozen selected assessment differs from its gate table"
        )
    method_commit = freeze.get("method_commit")
    if not isinstance(method_commit, str) or _GIT_OID.fullmatch(method_commit) is None:
        raise FinalAnalysisContractError(
            "frozen metric direction authority lacks a method commit"
        )
    blob_oid, source_raw = _git_object_bytes(repository, method_commit)
    try:
        tree = ast.parse(source_raw.decode("utf-8"))
    except (SyntaxError, UnicodeError) as error:
        raise FinalAnalysisContractError(
            "method-commit selection source cannot be parsed"
        ) from error
    rank_metrics = _source_metric_tuple(tree, "_RANK_METRICS")
    pareto_metrics = _source_metric_tuple(tree, "_PARETO_METRICS")
    average_rank = _source_function(tree, "_average_rank")
    pareto_function = _source_function(tree, "_pareto_dominators")
    if not _contains_named_comparison(average_rank, "value", ast.Lt, "target") or not (
        _contains_named_comparison(pareto_function, "left", ast.LtE, "right")
        and _contains_named_comparison(pareto_function, "left", ast.Lt, "right")
    ):
        raise FinalAnalysisContractError(
            "method-commit selection implementation does not declare lower preference"
        )

    gates = selected_assessment.get("gates")
    if not isinstance(gates, Mapping):
        raise FinalAnalysisContractError(
            "frozen selected assessment gates are malformed"
        )
    declared_rank_metrics = tuple(
        sorted(
            key.removeprefix("rank_")
            for key in gates
            if isinstance(key, str) and key.startswith("rank_")
        )
    )
    if declared_rank_metrics != tuple(sorted(rank_metrics)):
        raise FinalAnalysisContractError(
            "frozen rank gates differ from method-commit metric declaration"
        )
    for metric in rank_metrics:
        gate = gates.get(f"rank_{metric}")
        if not isinstance(gate, Mapping) or gate.get("threshold") != (
            "median biological-draw rank <= 2"
        ):
            raise FinalAnalysisContractError(
                "frozen rank gate does not preserve lower-metric semantics"
            )
    pareto_gate = gates.get("pareto_non_dominated")
    if (
        not isinstance(pareto_gate, Mapping)
        or pareto_gate.get("threshold")
        != (
            "no same-input method weakly better on all four dimensions and "
            "strictly better on one"
        )
        or not isinstance(pareto_gate.get("details"), Mapping)
        or pareto_gate["details"].get("dimensions") != list(pareto_metrics)
    ):
        raise FinalAnalysisContractError(
            "frozen Pareto gate differs from method-commit metric declaration"
        )
    lower_metrics = sorted(set((*rank_metrics, *pareto_metrics)))
    primary_metrics = protocol.get("primary_metrics")
    if not isinstance(primary_metrics, list) or any(
        metric not in lower_metrics for metric in primary_metrics
    ):
        raise FinalAnalysisContractError(
            "protocol primary metric lacks frozen lower-direction authority"
        )
    authority: dict[str, object] = {
        "source": "frozen_selection_gates_validated_against_method_commit",
        "method_commit": method_commit,
        "selection_source_path": "maskimpute_benchmark/selection.py",
        "selection_source_blob_oid": blob_oid,
        "selection_source_raw_sha256": hashlib.sha256(source_raw).hexdigest(),
        "frozen_method_payload_sha256": frozen_method.get("payload_sha256"),
        "selected_assessment_sha256": canonical_sha256(dict(selected_assessment)),
        "protocol_payload_sha256": canonical_sha256(dict(protocol)),
        "rank_metrics": list(rank_metrics),
        "pareto_metrics": list(pareto_metrics),
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "validated",
        "reason": None,
        "favorable_direction": "lower",
        "metrics": lower_metrics,
        "authority": authority,
    }
    return {**body, "contract_sha256": canonical_sha256(body)}


def _exact_mapping(
    value: object,
    expected_keys: frozenset[str],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise FinalAnalysisContractError(f"{name} schema is invalid")
    return value


def _result_file_bindings(
    manifest: Mapping[str, object],
) -> dict[str, str]:
    rows = manifest.get("result_files")
    if not isinstance(rows, list):
        raise FinalAnalysisContractError(
            "evaluation manifest result_files schema is invalid"
        )
    result: dict[str, str] = {}
    for index, row in enumerate(rows):
        mapping = _exact_mapping(
            row,
            frozenset({"path", "sha256"}),
            f"evaluation result file {index}",
        )
        path = mapping.get("path")
        if not isinstance(path, str) or not path or path in result:
            raise FinalAnalysisContractError(
                "evaluation manifest contains an invalid result path"
            )
        result[path] = _sha256(
            mapping.get("sha256"), f"evaluation result file {path} hash"
        )
    return result


def _validate_embedded_scaling_evidence(
    value: object,
    result_bindings: Mapping[str, str],
) -> Mapping[str, object]:
    evidence = _exact_mapping(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "plan",
                "checkpoint_path",
                "checkpoint_file_sha256",
                "checkpoint_payload",
                "result_files",
                "evidence_sha256",
            }
        ),
        "embedded scaling evidence",
    )
    body = {key: nested for key, nested in evidence.items() if key != "evidence_sha256"}
    checkpoint_path = evidence.get("checkpoint_path")
    checkpoint_file_sha256 = _sha256(
        evidence.get("checkpoint_file_sha256"),
        "embedded scaling checkpoint file",
    )
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
        or evidence.get("status") != "completed"
        or not isinstance(checkpoint_path, str)
        or re.fullmatch(
            r"results/scaling/checkpoints/[0-9]{8}\.json",
            checkpoint_path,
        )
        is None
        or evidence.get("evidence_sha256") != canonical_sha256(body)
        or result_bindings.get(str(checkpoint_path)) != checkpoint_file_sha256
    ):
        raise FinalAnalysisContractError("embedded scaling evidence binding is invalid")
    plan = _exact_mapping(
        evidence.get("plan"),
        frozenset(
            {
                "schema_version",
                "input_hashes",
                "entries",
                "configurations",
                "plan_sha256",
            }
        ),
        "embedded scaling plan",
    )
    plan_body = {key: nested for key, nested in plan.items() if key != "plan_sha256"}
    if (
        type(plan.get("schema_version")) is not int
        or plan.get("schema_version") != 1
        or plan.get("plan_sha256") != canonical_sha256(plan_body)
    ):
        raise FinalAnalysisContractError("embedded scaling plan binding is invalid")
    checkpoint = _exact_mapping(
        evidence.get("checkpoint_payload"),
        frozenset(
            {
                "schema_version",
                "plan_sha256",
                "input_hashes",
                "planned_run_count",
                "status",
                "datasets",
                "records",
                "checkpoint_sha256",
            }
        ),
        "embedded scaling checkpoint",
    )
    checkpoint_body = {
        key: nested for key, nested in checkpoint.items() if key != "checkpoint_sha256"
    }
    records = checkpoint.get("records")
    datasets = checkpoint.get("datasets")
    planned = checkpoint.get("planned_run_count")
    if (
        type(checkpoint.get("schema_version")) is not int
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("status") != "completed"
        or checkpoint.get("plan_sha256") != plan.get("plan_sha256")
        or checkpoint.get("input_hashes") != plan.get("input_hashes")
        or type(planned) is not int
        or planned <= 0
        or not isinstance(records, list)
        or len(records) != planned
        or not isinstance(datasets, list)
        or not datasets
        or checkpoint.get("checkpoint_sha256") != canonical_sha256(checkpoint_body)
        or checkpoint_path
        != (f"results/scaling/checkpoints/{len(datasets) + len(records):08d}.json")
    ):
        raise FinalAnalysisContractError(
            "embedded scaling checkpoint binding is invalid"
        )
    nested = _result_file_bindings({"result_files": evidence.get("result_files")})
    if (
        not nested
        or set(nested)
        != {path for path in result_bindings if path.startswith("results/scaling/")}
        or any(result_bindings.get(path) != digest for path, digest in nested.items())
        or nested.get(str(checkpoint_path)) != checkpoint_file_sha256
    ):
        raise FinalAnalysisContractError("embedded scaling result inventory is invalid")
    return evidence


def _validate_embedded_trajectory_evidence(
    value: object,
    result_bindings: Mapping[str, str],
    *,
    repository: Path,
    round_dir: Path,
    primary_final_plan_sha256: str,
    result_files: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Replay the receipt-local trajectory denominator from explicit context."""

    from .final_runner import (
        FinalPlanEntry,
        FinalRunnerContractError,
        FrozenPlanMethodAuthority,
        TrajectoryExecutionPlan,
        _canonical_round,
        _rederive_trajectory_evidence_before_receipt,
        _trajectory_run_id,
        _validate_frozen_execution_for_evaluation,
        _validate_trajectory_primary_authority_chain,
        trajectory_execution_plan_payload,
    )
    from .runner import (
        DEVELOPMENT_MODEL_SEEDS,
        AuthorizedConfiguration,
        RunPlanEntry,
        RunnerContractError,
        decode_direct_bound_comparator_value,
    )
    from .direct_values import direct_equal

    try:
        selected_repository, destination = _canonical_round(repository, round_dir)
    except (FinalRunnerContractError, OSError, TypeError, ValueError) as error:
        raise FinalAnalysisContractError(
            "trajectory evidence repository or round context is invalid"
        ) from error
    primary_plan_sha256 = _sha256(
        primary_final_plan_sha256,
        "primary final plan",
    )

    evidence = _exact_mapping(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "scope",
                "plan",
                "dataset",
                "execution_authority",
                "execution_manifest",
                "execution_validation",
                "result_files",
                "evidence_sha256",
            }
        ),
        "embedded trajectory evidence",
    )
    evidence_body = {
        key: nested for key, nested in evidence.items() if key != "evidence_sha256"
    }
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
        or evidence.get("status") != "completed"
        or evidence.get("scope") != "supplementary_trajectory"
        or evidence.get("evidence_sha256") != canonical_sha256(evidence_body)
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory evidence binding is invalid"
        )
    plan = _exact_mapping(
        evidence.get("plan"),
        frozenset(
            {
                "schema_version",
                "scope",
                "input_hashes",
                "entries",
                "configurations",
                "model_seed_policy",
                "plan_sha256",
            }
        ),
        "embedded trajectory plan",
    )
    plan_body = {key: nested for key, nested in plan.items() if key != "plan_sha256"}
    entry_rows = plan.get("entries")
    configuration_rows = plan.get("configurations")
    expected_input_hash_keys = frozenset(
        {
            "frozen_method_sha256",
            "method_registry_sha256",
            "runtime_lock_sha256",
            "primary_final_plan_sha256",
            "trajectory_authority_file_sha256",
            "trajectory_authority_sha256",
            "trajectory_binding_sha256",
            "trajectory_dataset_sha256",
            "trajectory_dataset_file_sha256",
            "trajectory_dataset_receipt_sha256",
            "trajectory_dataset_receipt_file_sha256",
            "trajectory_method_input_sha256",
            "trajectory_retained_cell_ids_sha256",
            "dataset_qc_policy_sha256",
            "execution_claim_sha256",
            "execution_environment_sha256",
            "execution_authority_sha256",
        }
    )
    input_hashes = _exact_mapping(
        plan.get("input_hashes"),
        expected_input_hash_keys,
        "embedded trajectory plan input hashes",
    )
    for name, digest in input_hashes.items():
        _sha256(digest, f"trajectory plan input {name}")
    if (
        type(plan.get("schema_version")) is not int
        or plan.get("schema_version") != 1
        or plan.get("scope") != "supplementary_trajectory"
        or plan.get("model_seed_policy") != list(DEVELOPMENT_MODEL_SEEDS)
        or not isinstance(entry_rows, list)
        or not entry_rows
        or not isinstance(configuration_rows, list)
        or not configuration_rows
        or input_hashes.get("primary_final_plan_sha256") != primary_plan_sha256
        or plan.get("plan_sha256") != canonical_sha256(plan_body)
    ):
        raise FinalAnalysisContractError("embedded trajectory plan binding is invalid")
    try:
        authority_chain = _validate_trajectory_primary_authority_chain(
            selected_repository,
            destination,
            primary_final_plan_sha256=primary_plan_sha256,
        )
    except FinalRunnerContractError as error:
        raise FinalAnalysisContractError(str(error)) from error
    if (
        input_hashes.get("execution_claim_sha256")
        != authority_chain["execution_claim_sha256"]
        or input_hashes.get("execution_environment_sha256")
        != authority_chain["execution_environment_sha256"]
        or input_hashes.get("execution_authority_sha256")
        != authority_chain["trajectory_execution_authority_sha256"]
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory plan differs from the primary authority chain"
        )

    configuration_fields = frozenset(
        {
            "method_id",
            "legacy_configuration",
            "comparator_configuration",
            "comparator_nonexecution_identity",
            "action",
            "reason",
            "seeds",
        }
    )
    configurations: list[FrozenPlanMethodAuthority] = []
    configuration_by_method: dict[str, FrozenPlanMethodAuthority] = {}
    for index, row_value in enumerate(configuration_rows):
        row = _exact_mapping(
            row_value,
            configuration_fields,
            f"embedded trajectory configuration {index}",
        )
        legacy_value = row.get("legacy_configuration")
        comparator_value = row.get("comparator_configuration")
        nonexecution_value = row.get("comparator_nonexecution_identity")
        seeds = row.get("seeds")
        try:
            legacy = None
            comparator = None
            if legacy_value is not None:
                if not isinstance(legacy_value, Mapping):
                    raise TypeError("legacy configuration is not an object")
                payload = legacy_value.get("payload")
                if not isinstance(payload, Mapping):
                    raise TypeError("legacy payload is not an object")
                legacy = AuthorizedConfiguration.create(
                    method_id=legacy_value.get("method_id"),
                    configuration_id=legacy_value.get("configuration_id"),
                    kind=legacy_value.get("kind"),
                    payload=payload,
                    requires_count_score=legacy_value.get("requires_count_score"),
                    requires_calibration=legacy_value.get("requires_calibration"),
                    configuration_sha256=legacy_value.get("configuration_sha256"),
                    registry_method_sha256=legacy_value.get("registry_method_sha256"),
                    tuning_authority_file_sha256=legacy_value.get(
                        "tuning_authority_file_sha256"
                    ),
                    tuning_authority_payload_sha256=legacy_value.get(
                        "tuning_authority_payload_sha256"
                    ),
                    source_authority_sha256=legacy_value.get("source_authority_sha256"),
                    runtime_lock_sha256=legacy_value.get("runtime_lock_sha256"),
                    environment_registry_sha256=legacy_value.get(
                        "environment_registry_sha256"
                    ),
                    configuration_method_identity_sha256=legacy_value.get(
                        "configuration_method_identity_sha256"
                    ),
                    nonexecution_identity_sha256=legacy_value.get(
                        "nonexecution_identity_sha256"
                    ),
                )
                if legacy.to_dict() != dict(legacy_value):
                    raise ValueError("legacy configuration changed during decoding")
            elif comparator_value is not None:
                comparator = decode_direct_bound_comparator_value(comparator_value)
            configuration = FrozenPlanMethodAuthority(
                method_id=row.get("method_id"),
                legacy_configuration=legacy,
                comparator_configuration=comparator,
                comparator_nonexecution_identity=(
                    nonexecution_value
                    if isinstance(nonexecution_value, Mapping)
                    else None
                ),
                action=row.get("action"),
                reason=row.get("reason"),
                seeds=tuple(seeds) if isinstance(seeds, list) else (),
            )
        except (
            FinalRunnerContractError,
            RunnerContractError,
            TypeError,
            ValueError,
        ) as error:
            raise FinalAnalysisContractError(
                "embedded trajectory configuration is invalid"
            ) from error
        if _canonical_bytes(configuration.to_dict()) != _canonical_bytes(dict(row)):
            raise FinalAnalysisContractError(
                "embedded trajectory configuration binding differs"
            )
        if configuration.method_id in configuration_by_method:
            raise FinalAnalysisContractError(
                "embedded trajectory configurations are duplicated"
            )
        configuration_by_method[configuration.method_id] = configuration
        configurations.append(configuration)

    dataset = _exact_mapping(
        evidence.get("dataset"),
        frozenset(
            {
                "binding",
                "dataset_path",
                "dataset_file_sha256",
                "dataset_sha256",
                "receipt_path",
                "receipt_file_sha256",
                "receipt_payload_sha256",
            }
        ),
        "embedded trajectory dataset",
    )
    from .trajectory_dataset import RegisteredTrajectoryBinding

    binding_value = dataset.get("binding")
    if not isinstance(binding_value, Mapping):
        raise FinalAnalysisContractError(
            "embedded trajectory dataset binding is invalid"
        )
    try:
        binding = RegisteredTrajectoryBinding(**dict(binding_value))
    except (TypeError, ValueError) as error:
        raise FinalAnalysisContractError(
            "embedded trajectory dataset binding is invalid"
        ) from error
    if (
        dataset.get("dataset_path") != binding.dataset_file_path
        or dataset.get("dataset_file_sha256") != binding.dataset_file_sha256
        or dataset.get("dataset_sha256") != binding.dataset_sha256
        or dataset.get("receipt_path")
        != "results/trajectory/dataset/dataset_receipt.json"
        or result_bindings.get(binding.dataset_file_path) != binding.dataset_file_sha256
        or result_bindings.get(str(dataset.get("receipt_path")))
        != _sha256(
            dataset.get("receipt_file_sha256"),
            "trajectory dataset receipt file",
        )
        or input_hashes.get("trajectory_binding_sha256")
        != binding.registered_binding_sha256
        or input_hashes.get("trajectory_dataset_sha256") != binding.dataset_sha256
        or input_hashes.get("trajectory_dataset_file_sha256")
        != binding.dataset_file_sha256
        or input_hashes.get("trajectory_dataset_receipt_sha256")
        != dataset.get("receipt_payload_sha256")
        or input_hashes.get("trajectory_dataset_receipt_file_sha256")
        != dataset.get("receipt_file_sha256")
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory dataset receipt binding is invalid"
        )

    legacy_run_fields = frozenset(
        {
            "ordinal",
            "run_id",
            "method_id",
            "dataset_id",
            "source_dataset_sha256",
            "mechanism",
            "biological_id",
            "technical_view",
            "model_seed",
            "configuration_id",
            "configuration_sha256",
            "configuration_payload_sha256",
            "configuration_method_identity_sha256",
            "nonexecution_identity_sha256",
            "preflight_status",
            "preflight_reason",
            "configuration_kind",
            "requires_count_score",
            "requires_calibration",
        }
    )
    direct_run_fields = (
        legacy_run_fields
        - {
            "configuration_sha256",
            "configuration_payload_sha256",
            "configuration_method_identity_sha256",
            "nonexecution_identity_sha256",
        }
    ) | {"comparator_configuration", "comparator_nonexecution_identity"}
    entries: list[FinalPlanEntry] = []
    run_ids: set[str] = set()
    used_methods: set[str] = set()
    for ordinal, entry_value in enumerate(entry_rows, start=1):
        entry_row = _exact_mapping(
            entry_value,
            frozenset({"run", "action", "reason"}),
            f"embedded trajectory plan entry {ordinal}",
        )
        raw_run_row = entry_row.get("run")
        method_id = (
            raw_run_row.get("method_id") if isinstance(raw_run_row, Mapping) else None
        )
        configuration = configuration_by_method.get(method_id)
        if configuration is None:
            raise FinalAnalysisContractError(
                "embedded trajectory run lacks its configuration"
            )
        direct_comparator = configuration.kind in {
            "comparator_tuning",
            "comparator_nonexecution",
        }
        run_row = _exact_mapping(
            raw_run_row,
            direct_run_fields if direct_comparator else legacy_run_fields,
            f"embedded trajectory run {ordinal}",
        )
        action = entry_row.get("action")
        reason = entry_row.get("reason")
        if (
            type(run_row.get("ordinal")) is not int
            or run_row["ordinal"] <= 0
            or any(
                not isinstance(run_row.get(name), str) or not run_row[name]
                for name in (
                    "run_id",
                    "method_id",
                    "dataset_id",
                    "source_dataset_sha256",
                    "mechanism",
                    "biological_id",
                    "technical_view",
                    "configuration_id",
                    "preflight_status",
                    "configuration_kind",
                )
            )
            or (
                run_row.get("model_seed") is not None
                and type(run_row.get("model_seed")) is not int
            )
            or type(run_row.get("requires_count_score")) is not bool
            or type(run_row.get("requires_calibration")) is not bool
            or not isinstance(action, str)
            or action not in {"execute", "not_applicable"}
            or (action == "execute" and reason is not None)
            or (
                action == "not_applicable"
                and (not isinstance(reason, str) or not reason)
            )
        ):
            raise FinalAnalysisContractError(
                "embedded trajectory plan entry has invalid exact types"
            )
        _sha256(
            run_row.get("source_dataset_sha256"),
            "trajectory run source dataset",
        )
        if not direct_comparator:
            _sha256(
                run_row.get("configuration_sha256"),
                "trajectory run configuration",
            )
        try:
            decoded_run = dict(run_row)
            if (
                direct_comparator
                and run_row.get("comparator_configuration") is not None
            ):
                decoded_run["comparator_configuration"] = (
                    decode_direct_bound_comparator_value(
                        run_row.get("comparator_configuration")
                    )
                )
            run = RunPlanEntry(**decoded_run)
            entry = FinalPlanEntry(
                run=run,
                action=action,
                reason=reason,
            )
        except (FinalRunnerContractError, TypeError, ValueError) as error:
            raise FinalAnalysisContractError(
                "embedded trajectory plan entry is invalid"
            ) from error
        try:
            expected_run_id = _trajectory_run_id(
                ordinal,
                run.method_id,
                run.dataset_id,
                run.model_seed,
                configuration,
            )
        except (TypeError, ValueError) as error:
            raise FinalAnalysisContractError(
                "embedded trajectory run identity is invalid"
            ) from error
        if (
            run.ordinal != ordinal
            or run.run_id != expected_run_id
            or run.run_id in run_ids
            or run.dataset_id != binding.dataset_id
            or run.source_dataset_sha256 != binding.dataset_sha256
            or run.mechanism != binding.mechanism
            or run.biological_id != binding.biological_id
            or run.technical_view != binding.technical_view
            or run.model_seed not in (None, *DEVELOPMENT_MODEL_SEEDS)
            or run.configuration_id != configuration.configuration_id
            or (
                run.configuration_sha256
                != configuration.legacy_configuration.configuration_sha256
                if configuration.legacy_configuration is not None
                else not direct_equal(
                    run.comparator_configuration,
                    configuration.comparator_configuration,
                )
                or not direct_equal(
                    run.comparator_nonexecution_identity,
                    configuration.comparator_nonexecution_identity,
                )
            )
            or run.preflight_status != "planned"
            or run.preflight_reason is not None
            or run.configuration_kind != configuration.kind
            or run.requires_count_score != configuration.requires_count_score
            or run.requires_calibration != configuration.requires_calibration
        ):
            raise FinalAnalysisContractError(
                "embedded trajectory run differs from its exact plan authority"
            )
        run_ids.add(run.run_id)
        used_methods.add(run.method_id)
        entries.append(entry)
    if used_methods != set(configuration_by_method):
        raise FinalAnalysisContractError(
            "embedded trajectory configuration denominator is incomplete"
        )
    typed_plan = TrajectoryExecutionPlan(
        schema_version=1,
        scope="supplementary_trajectory",
        input_hashes=dict(input_hashes),
        entries=tuple(entries),
        configurations=tuple(configurations),
        plan_sha256=str(plan["plan_sha256"]),
    )
    try:
        if _canonical_bytes(trajectory_execution_plan_payload(typed_plan)) != (
            _canonical_bytes(dict(plan))
        ):
            raise FinalAnalysisContractError("embedded trajectory plan replay differs")
    except (FinalRunnerContractError, TypeError, ValueError) as error:
        raise FinalAnalysisContractError(
            "embedded trajectory plan replay is invalid"
        ) from error

    authority = _exact_mapping(
        evidence.get("execution_authority"),
        frozenset(
            {
                "authority_path",
                "authority_file_sha256",
                "authority_sha256",
                "count_score_authority_path",
                "count_score_authority_file_sha256",
                "retained_calibration_path",
                "retained_calibration_file_sha256",
                "files",
            }
        ),
        "embedded trajectory execution authority",
    )
    authority_path = authority.get("authority_path")
    if (
        authority_path != "results/trajectory/execution_authority/authority.json"
        or result_bindings.get(str(authority_path))
        != _sha256(
            authority.get("authority_file_sha256"),
            "trajectory execution authority file",
        )
        or input_hashes.get("execution_authority_sha256")
        != _sha256(
            authority.get("authority_sha256"),
            "trajectory execution authority",
        )
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory execution authority binding is invalid"
        )
    authority_files = _result_file_bindings({"result_files": authority.get("files")})
    expected_authority_files = {
        path: digest
        for path, digest in result_bindings.items()
        if path.startswith("results/trajectory/execution_authority/")
    }
    if authority_files != expected_authority_files or len(authority_files) != 3:
        raise FinalAnalysisContractError(
            "embedded trajectory authority inventory is invalid"
        )

    manifest = _exact_mapping(
        evidence.get("execution_manifest"),
        frozenset({"path", "file_sha256", "payload_sha256"}),
        "embedded trajectory execution manifest",
    )
    if (
        manifest.get("path") != "results/trajectory/execution/execution_manifest.json"
        or result_bindings.get(str(manifest.get("path")))
        != _sha256(
            manifest.get("file_sha256"),
            "trajectory execution manifest file",
        )
        or _sha256(
            manifest.get("payload_sha256"),
            "trajectory execution manifest payload",
        )
        != manifest.get("payload_sha256")
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory execution manifest binding is invalid"
        )

    manifest_path = destination / str(manifest["path"])
    try:
        manifest_payload, manifest_raw = _read_json_file(
            manifest_path,
            "trajectory execution manifest",
            require_canonical=True,
        )
    except (FinalAnalysisContractError, OSError) as error:
        raise FinalAnalysisContractError(
            "embedded trajectory execution manifest cannot be read"
        ) from error
    expected_manifest_fields = frozenset(
        {
            "schema_version",
            "status",
            "plan_sha256",
            "input_hashes",
            "planned_run_count",
            "recorded_run_count",
            "records",
            "artifact_storage",
            "scope",
            "plan_entries",
            "configurations",
            "model_seed_policy",
            "manifest_sha256",
        }
    )
    manifest_payload = _exact_mapping(
        manifest_payload,
        expected_manifest_fields,
        "trajectory execution manifest payload",
    )
    manifest_body = {
        key: nested
        for key, nested in manifest_payload.items()
        if key != "manifest_sha256"
    }
    manifest_records = manifest_payload.get("records")
    if (
        hashlib.sha256(manifest_raw).hexdigest() != manifest.get("file_sha256")
        or type(manifest_payload.get("schema_version")) is not int
        or manifest_payload.get("schema_version") != 1
        or manifest_payload.get("status") != "completed"
        or manifest_payload.get("scope") != "supplementary_trajectory"
        or manifest_payload.get("plan_sha256") != typed_plan.plan_sha256
        or _canonical_bytes(manifest_payload.get("input_hashes"))
        != _canonical_bytes(dict(typed_plan.input_hashes))
        or type(manifest_payload.get("planned_run_count")) is not int
        or manifest_payload.get("planned_run_count") != len(entries)
        or type(manifest_payload.get("recorded_run_count")) is not int
        or manifest_payload.get("recorded_run_count") != len(entries)
        or _canonical_bytes(manifest_payload.get("plan_entries"))
        != _canonical_bytes([entry.to_dict() for entry in entries])
        or _canonical_bytes(manifest_payload.get("configurations"))
        != _canonical_bytes(
            [configuration.to_dict() for configuration in configurations]
        )
        or _canonical_bytes(manifest_payload.get("model_seed_policy"))
        != _canonical_bytes(list(DEVELOPMENT_MODEL_SEEDS))
        or _canonical_bytes(manifest_payload.get("artifact_storage"))
        != _canonical_bytes(_EXPECTED_FINAL_STORAGE_POLICY)
        or manifest_payload.get("manifest_sha256") != canonical_sha256(manifest_body)
        or manifest.get("payload_sha256") != manifest_payload.get("manifest_sha256")
        or not isinstance(manifest_records, list)
        or len(manifest_records) != len(entries)
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory execution manifest differs from its plan"
        )

    records: list[Mapping[str, object]] = []
    record_result_paths: set[str] = set()
    for entry, reference_value in zip(entries, manifest_records, strict=True):
        reference = _exact_mapping(
            reference_value,
            frozenset({"ordinal", "run_id", "path", "sha256"}),
            f"trajectory manifest record {entry.run.ordinal}",
        )
        relative = f"records/{entry.run.ordinal:08d}.json"
        full_relative = f"results/trajectory/execution/{relative}"
        if (
            type(reference.get("ordinal")) is not int
            or reference.get("ordinal") != entry.run.ordinal
            or reference.get("run_id") != entry.run.run_id
            or reference.get("path") != relative
        ):
            raise FinalAnalysisContractError(
                "trajectory manifest record binding differs"
            )
        record_path = destination / full_relative
        record, record_raw = _read_json_file(
            record_path,
            f"trajectory execution record {entry.run.ordinal}",
            require_canonical=True,
        )
        record = _exact_mapping(
            record,
            frozenset({"run", "metrics", "p_pre_zero_evidence", "execution_request"}),
            f"trajectory execution record {entry.run.ordinal}",
        )
        record_run = record.get("run")
        record_metrics = record.get("metrics")
        if (
            not isinstance(record_run, Mapping)
            or not isinstance(record_metrics, list)
            or any(
                not isinstance(record_run.get(name), str)
                for name in (
                    "run_id",
                    "method_id",
                    "dataset_id",
                    "configuration_sha256",
                    "status",
                )
            )
            or _canonical_bytes(record_run.get("model_seed"))
            != _canonical_bytes(entry.run.model_seed)
            or any(
                not isinstance(metric, Mapping)
                or not isinstance(metric.get("metric"), str)
                or not isinstance(metric.get("status"), str)
                or type(metric.get("n")) is not int
                or metric["n"] < 0
                or isinstance(metric.get("value"), bool)
                or (
                    metric.get("value") is not None
                    and not isinstance(metric.get("value"), (int, float))
                )
                or (
                    metric.get("reason") is not None
                    and not isinstance(metric.get("reason"), str)
                )
                for metric in record_metrics
            )
        ):
            raise FinalAnalysisContractError(
                "trajectory execution record has invalid exact types"
            )
        record_file_sha256 = hashlib.sha256(record_raw).hexdigest()
        if (
            record_file_sha256
            != _sha256(reference.get("sha256"), "trajectory record file")
            or result_bindings.get(full_relative) != record_file_sha256
        ):
            raise FinalAnalysisContractError(
                "trajectory execution record file binding differs"
            )
        records.append(record)
        record_result_paths.add(full_relative)
    validation = _exact_mapping(
        evidence.get("execution_validation"),
        frozenset(
            {
                "schema_version",
                "status",
                "scope",
                "trajectory_plan_sha256",
                "planned_run_count",
                "executed_completed_count",
                "executed_algorithmic_failure_count",
                "executed_status_counts",
                "not_applicable_count",
                "record_payload_sha256s",
                "validation_sha256",
            }
        ),
        "embedded trajectory execution validation",
    )
    validation_body = {
        key: nested for key, nested in validation.items() if key != "validation_sha256"
    }
    count_names = (
        "planned_run_count",
        "executed_completed_count",
        "executed_algorithmic_failure_count",
        "not_applicable_count",
    )
    status_counts = validation.get("executed_status_counts")
    payload_hashes = validation.get("record_payload_sha256s")
    if (
        type(validation.get("schema_version")) is not int
        or validation.get("schema_version") != 1
        or validation.get("status")
        != "eligible_for_final_evaluation_complete_terminal_denominator"
        or validation.get("scope") != "supplementary_trajectory"
        or validation.get("trajectory_plan_sha256") != plan.get("plan_sha256")
        or any(
            type(validation.get(name)) is not int or validation[name] < 0
            for name in count_names
        )
        or validation.get("planned_run_count") != len(entries)
        or not isinstance(status_counts, dict)
        or any(
            status
            not in {
                "completed",
                "failed",
                "timeout",
                "resource_exceeded",
                "unavailable",
            }
            or type(count) is not int
            or count <= 0
            for status, count in (
                status_counts.items() if isinstance(status_counts, dict) else ()
            )
        )
        or sum(status_counts.values())
        != validation.get("executed_completed_count")
        + validation.get("executed_algorithmic_failure_count")
        or sum(status_counts.values()) + validation.get("not_applicable_count")
        != len(entries)
        or status_counts.get("completed", 0)
        != validation.get("executed_completed_count")
        or sum(
            status_counts.get(status, 0)
            for status in (
                "failed",
                "timeout",
                "resource_exceeded",
                "unavailable",
            )
        )
        != validation.get("executed_algorithmic_failure_count")
        or not isinstance(payload_hashes, list)
        or len(payload_hashes) != len(entries)
        or validation.get("validation_sha256") != canonical_sha256(validation_body)
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory execution validation binding is invalid"
        )
    for index, digest in enumerate(payload_hashes):
        _sha256(digest, f"trajectory validation record {index} payload")
    try:
        replayed_validation = _validate_frozen_execution_for_evaluation(
            typed_plan,
            records,
        )
    except (FinalRunnerContractError, TypeError, ValueError) as error:
        raise FinalAnalysisContractError(
            "embedded trajectory terminal denominator is invalid"
        ) from error
    if _canonical_bytes(replayed_validation) != _canonical_bytes(dict(validation)):
        raise FinalAnalysisContractError(
            "embedded trajectory validation differs from record replay"
        )
    nested = _result_file_bindings({"result_files": evidence.get("result_files")})
    expected_nested = {
        path: digest
        for path, digest in result_bindings.items()
        if path.startswith("results/trajectory/")
    }
    nested_record_paths = {
        path
        for path in nested
        if path.startswith("results/trajectory/execution/records/")
    }
    if (
        not nested
        or nested != expected_nested
        or nested_record_paths != record_result_paths
    ):
        raise FinalAnalysisContractError(
            "embedded trajectory result inventory is invalid"
        )
    try:
        rederived = _rederive_trajectory_evidence_before_receipt(
            selected_repository,
            destination,
            evidence,
            result_files,
            primary_final_plan_sha256=primary_plan_sha256,
        )
    except FinalRunnerContractError as error:
        raise FinalAnalysisContractError(str(error)) from error
    if _canonical_bytes(rederived) != _canonical_bytes(dict(evidence)):
        raise FinalAnalysisContractError(
            "embedded trajectory evidence differs from frozen rederivation"
        )
    return evidence


def _validate_storage_preflight(
    value: object,
    *,
    planned_run_count: int,
    require_combined: bool = False,
) -> Mapping[str, object]:
    if type(planned_run_count) is not int or planned_run_count <= 0:
        raise FinalAnalysisContractError(
            "storage preflight planned run count is invalid"
        )
    if isinstance(value, Mapping) and value.get("schema") == (
        "maskimpute-combined-final-storage-preflight-v1"
    ):
        preflight = _exact_mapping(
            value,
            _COMBINED_STORAGE_PREFLIGHT_KEYS,
            "combined final storage preflight",
        )

        def component(
            nested_value: object,
            name: str,
            expected_planned: int | None,
        ) -> Mapping[str, object]:
            nested = _exact_mapping(
                nested_value,
                _EXECUTION_STORAGE_COMPONENT_KEYS,
                f"{name} storage component",
            )
            if any(
                type(nested.get(field)) is not int or nested[field] < 0
                for field in _EXECUTION_STORAGE_COMPONENT_KEYS
            ):
                raise FinalAnalysisContractError(
                    f"{name} storage component value is invalid"
                )
            completed = int(nested["completed_record_count"])
            remaining = int(nested["remaining_entry_count"])
            if (
                (
                    expected_planned is not None
                    and completed + remaining != expected_planned
                )
                or nested["remaining_execution_count"] > remaining
                or nested["remaining_p_pre_zero_execution_count"]
                > nested["remaining_execution_count"]
                or nested["cells"] <= 0
                or nested["genes"] <= 0
                or nested["per_execution_compressed_bound_bytes"] <= 0
                or nested["per_p_pre_zero_compressed_bound_bytes"] <= 0
            ):
                raise FinalAnalysisContractError(
                    f"{name} storage component denominator is invalid"
                )
            return nested

        primary = component(preflight.get("primary"), "primary", planned_run_count)
        trajectory = component(preflight.get("trajectory"), "trajectory", None)
        scaling = preflight.get("scaling")
        if not isinstance(scaling, Mapping):
            raise FinalAnalysisContractError(
                "combined scaling storage component is invalid"
            )
        scaling_body = {
            key: nested for key, nested in scaling.items() if key != "receipt_sha256"
        }
        scaling_required = scaling.get("required_free_bytes")
        if (
            scaling.get("schema") != "maskimpute-scaling-storage-preflight-v1"
            or type(scaling_required) is not int
            or scaling_required < 0
            or scaling.get("receipt_sha256") != canonical_sha256(scaling_body)
        ):
            raise FinalAnalysisContractError(
                "combined scaling storage component binding is invalid"
            )
        for field in ("reserve_bytes", "required_free_bytes", "observed_free_bytes"):
            if type(preflight.get(field)) is not int or preflight[field] < 0:
                raise FinalAnalysisContractError(
                    "combined final storage preflight value is invalid"
                )
        if (
            preflight["reserve_bytes"] != 1024**3
            or preflight["required_free_bytes"]
            != primary["required_free_bytes"]
            + trajectory["required_free_bytes"]
            + scaling_required
            + preflight["reserve_bytes"]
            or preflight["observed_free_bytes"] < preflight["required_free_bytes"]
        ):
            raise FinalAnalysisContractError(
                "combined final storage preflight denominator is invalid"
            )
        return preflight
    if require_combined:
        raise FinalAnalysisContractError(
            "trajectory-bound evaluation requires combined storage preflight"
        )
    preflight = _exact_mapping(
        value,
        _STORAGE_PREFLIGHT_KEYS,
        "final storage preflight",
    )
    if preflight.get("schema") != "maskimpute-final-storage-preflight-v1":
        raise FinalAnalysisContractError("final storage preflight schema is invalid")
    integer_fields = _STORAGE_PREFLIGHT_KEYS - {"schema"}
    if any(
        type(preflight.get(field)) is not int or preflight[field] < 0
        for field in integer_fields
    ):
        raise FinalAnalysisContractError("final storage preflight value is invalid")
    completed = int(preflight["completed_record_count"])
    remaining = int(preflight["remaining_entry_count"])
    remaining_execution = int(preflight["remaining_execution_count"])
    remaining_score = int(preflight["remaining_p_pre_zero_execution_count"])
    if (
        completed + remaining != planned_run_count
        or remaining_execution > remaining
        or remaining_score > remaining_execution
        or preflight["per_execution_compressed_bound_bytes"] <= 0
        or preflight["per_p_pre_zero_compressed_bound_bytes"] <= 0
        or preflight["observed_free_bytes"] < preflight["required_free_bytes"]
    ):
        raise FinalAnalysisContractError(
            "final storage preflight denominator is invalid"
        )
    return preflight


def _zlib_compress_bound(uncompressed_nbytes: int) -> int:
    return (
        uncompressed_nbytes
        + (uncompressed_nbytes >> 12)
        + (uncompressed_nbytes >> 14)
        + (uncompressed_nbytes >> 25)
        + 13
    )


def _score_artifact_binding(
    record: Mapping[str, object],
    *,
    record_index: int,
    execution_dir: Path,
    destination: Path,
    result_bindings: Mapping[str, str],
) -> tuple[dict[str, object] | None, tuple[Path, str] | None]:
    from .final_runner import FinalRunnerContractError, _read_unique_file

    evidence = record["p_pre_zero_evidence"]
    assert isinstance(evidence, Mapping)
    storage = evidence["storage"]
    assert isinstance(storage, Mapping)
    relative_value = storage["path"]
    if relative_value is None:
        return None, None
    if not isinstance(relative_value, str) or "\\" in relative_value:
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero artifact path is invalid"
        )
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "runs"
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != relative_value
    ):
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero artifact path is invalid"
        )
    compressed_sha256 = _sha256(
        storage["compressed_sha256"],
        f"record {record_index} p_pre_zero artifact hash",
    )
    compressed_nbytes = storage["compressed_nbytes"]
    uncompressed_nbytes = storage["uncompressed_nbytes"]
    assert isinstance(compressed_nbytes, int)
    assert isinstance(uncompressed_nbytes, int)
    if compressed_nbytes > _zlib_compress_bound(uncompressed_nbytes):
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero artifact exceeds its zlib bound"
        )
    path = execution_dir.joinpath(*relative.parts)
    declared_path = path.relative_to(destination).as_posix()
    if result_bindings.get(declared_path) != compressed_sha256:
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero artifact is not identically bound "
            "as an evaluated result file"
        )
    try:
        raw = _read_unique_file(
            path,
            f"record {record_index} p_pre_zero artifact",
            max_bytes=compressed_nbytes,
        )
    except FinalRunnerContractError as error:
        raise FinalAnalysisContractError(str(error)) from error
    if (
        len(raw) != compressed_nbytes
        or hashlib.sha256(raw).hexdigest() != compressed_sha256
    ):
        raise FinalAnalysisContractError(
            f"record {record_index} p_pre_zero artifact differs from its receipt"
        )
    return (
        {
            "path": relative_value,
            "raw_sha256": compressed_sha256,
            "raw_nbytes": compressed_nbytes,
        },
        (path, compressed_sha256),
    )


def _validate_execution_validation(
    value: object,
    *,
    final_plan_sha256: str,
    record_count: int,
) -> Mapping[str, object]:
    expected = frozenset(
        {
            "schema_version",
            "status",
            "final_plan_sha256",
            "planned_run_count",
            "executed_completed_count",
            "executed_algorithmic_failure_count",
            "executed_status_counts",
            "not_applicable_count",
            "record_payload_sha256s",
            "validation_sha256",
        }
    )
    validation = _exact_mapping(value, expected, "execution validation")
    body = {
        key: nested for key, nested in validation.items() if key != "validation_sha256"
    }
    if (
        type(validation.get("schema_version")) is not int
        or validation.get("schema_version") != 1
        or validation.get("status")
        != "eligible_for_final_evaluation_complete_terminal_denominator"
        or validation.get("final_plan_sha256") != final_plan_sha256
        or type(validation.get("planned_run_count")) is not int
        or validation.get("planned_run_count") != record_count
        or validation.get("validation_sha256") != canonical_sha256(body)
    ):
        raise FinalAnalysisContractError("execution validation binding is invalid")
    counts = validation.get("executed_status_counts")
    if not isinstance(counts, Mapping) or any(
        status not in _TERMINAL_STATUSES
        or status == "unavailable_not_applicable"
        or type(count) is not int
        or count < 0
        for status, count in counts.items()
    ):
        raise FinalAnalysisContractError("execution terminal status counts are invalid")
    not_applicable = validation.get("not_applicable_count")
    completed = validation.get("executed_completed_count")
    failures = validation.get("executed_algorithmic_failure_count")
    if any(
        type(item) is not int or item < 0
        for item in (not_applicable, completed, failures)
    ):
        raise FinalAnalysisContractError("execution denominator counts are invalid")
    assert isinstance(not_applicable, int)
    assert isinstance(completed, int)
    assert isinstance(failures, int)
    if (
        sum(int(count) for count in counts.values()) + not_applicable != record_count
        or counts.get("completed", 0) != completed
        or sum(
            int(counts.get(status, 0))
            for status in ("failed", "timeout", "resource_exceeded", "unavailable")
        )
        != failures
    ):
        raise FinalAnalysisContractError(
            "execution denominator counts do not reconcile"
        )
    payload_hashes = validation.get("record_payload_sha256s")
    if not isinstance(payload_hashes, list) or len(payload_hashes) != record_count:
        raise FinalAnalysisContractError(
            "execution validation record denominator is invalid"
        )
    for index, digest in enumerate(payload_hashes):
        _sha256(digest, f"execution validation record {index} payload hash")
    return validation


def _evaluated_inputs(
    repository: Path,
    round_dir: Path,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    from .final_runner import FinalRunnerContractError, _canonical_round
    from .study import (
        StudyStateError,
        _validate_freeze,
        _validate_registry,
        _validate_result_files,
        _validate_state_record_chain,
        _verify_frozen_repository,
    )

    try:
        selected_repository, destination = _canonical_round(repository, round_dir)
        freeze = _validate_freeze(destination, selected_repository)
        _validate_registry(
            selected_repository,
            destination,
            freeze,
            expected_state="evaluated",
        )
        _materialization, _claim, receipt = _validate_state_record_chain(
            destination,
            freeze,
            expected_state="evaluated",
        )
        if not isinstance(receipt, Mapping):
            raise FinalAnalysisContractError(
                "evaluated lifecycle lacks an evaluation receipt"
            )
        evaluation = receipt.get("result_manifest")
        expected_evaluation_keys = frozenset(
            {
                "schema_version",
                "status",
                "final_plan_sha256",
                "final_execution_manifest_path",
                "final_execution_manifest_sha256",
                "final_execution_payload_sha256",
                "execution_validation",
                "storage_preflight",
                "scaling_evidence",
                "trajectory_evidence",
                "result_files",
            }
        )
        evaluation = _exact_mapping(
            evaluation,
            expected_evaluation_keys,
            "evaluation manifest",
        )
        receipt_manifest_hash = _sha256(
            receipt.get("result_manifest_sha256"), "result manifest hash"
        )
        if canonical_sha256(dict(evaluation)) != receipt_manifest_hash:
            raise FinalAnalysisContractError(
                "evaluation receipt result manifest hash does not match"
            )
        if (
            type(evaluation.get("schema_version")) is not int
            or evaluation.get("schema_version") != 1
            or evaluation.get("status") != "completed"
        ):
            raise FinalAnalysisContractError("evaluation manifest status is invalid")
        allowed_paths = _validate_result_files(
            selected_repository, destination, evaluation
        )
        _verify_frozen_repository(
            selected_repository,
            destination,
            allowed_result_paths=allowed_paths,
        )
    except (FinalRunnerContractError, StudyStateError) as error:
        raise FinalAnalysisContractError(str(error)) from error

    result_bindings = _result_file_bindings(evaluation)
    scaling_evidence = _validate_embedded_scaling_evidence(
        evaluation.get("scaling_evidence"),
        result_bindings,
    )
    trajectory_evidence = _validate_embedded_trajectory_evidence(
        evaluation.get("trajectory_evidence"),
        result_bindings,
        repository=selected_repository,
        round_dir=destination,
        primary_final_plan_sha256=_sha256(
            evaluation.get("final_plan_sha256"),
            "primary final plan",
        ),
        result_files=evaluation["result_files"],
    )
    manifest_relative = evaluation.get("final_execution_manifest_path")
    expected_manifest_relative = "results/final/execution/execution_manifest.json"
    if manifest_relative != expected_manifest_relative:
        raise FinalAnalysisContractError(
            "final execution manifest path is not the canonical final path"
        )
    manifest_path = destination / expected_manifest_relative
    manifest, manifest_raw = _read_json_file(
        manifest_path,
        "final execution manifest",
        require_canonical=True,
    )
    manifest_raw_hash = hashlib.sha256(manifest_raw).hexdigest()
    if (
        result_bindings.get(expected_manifest_relative) != manifest_raw_hash
        or evaluation.get("final_execution_manifest_sha256") != manifest_raw_hash
    ):
        raise FinalAnalysisContractError(
            "final execution manifest raw hash does not match evaluated receipt"
        )
    expected_manifest_keys = frozenset(
        {
            "schema_version",
            "status",
            "plan_sha256",
            "input_hashes",
            "planned_run_count",
            "recorded_run_count",
            "records",
            "artifact_storage",
            "manifest_sha256",
        }
    )
    _exact_mapping(manifest, expected_manifest_keys, "final execution manifest")
    expected_input_hash_keys = frozenset(
        {
            "frozen_method_sha256",
            "method_registry_sha256",
            "runtime_lock_sha256",
            "dataset_manifest_sha256",
            "dataset_design_sha256",
            "dataset_seed_source_sha256",
            "protocol_sha256",
            "execution_claim_sha256",
            "execution_environment_sha256",
            "execution_authority_sha256",
        }
    )
    execution_input_hashes = _exact_mapping(
        manifest.get("input_hashes"),
        expected_input_hash_keys,
        "execution input hashes",
    )
    try:
        execution_input_hashes = {
            key: _sha256(execution_input_hashes[key], f"execution input hash {key}")
            for key in sorted(execution_input_hashes)
        }
    except FinalAnalysisContractError as error:
        raise FinalAnalysisContractError(
            "execution input hashes contain an invalid digest"
        ) from error
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    final_plan_sha256 = _sha256(evaluation.get("final_plan_sha256"), "final plan hash")
    references = manifest.get("records")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("plan_sha256") != final_plan_sha256
        or manifest.get("artifact_storage") != _EXPECTED_FINAL_STORAGE_POLICY
        or manifest.get("manifest_sha256") != canonical_sha256(manifest_body)
        or evaluation.get("final_execution_payload_sha256")
        != manifest.get("manifest_sha256")
        or not isinstance(references, list)
        or type(manifest.get("planned_run_count")) is not int
        or manifest.get("planned_run_count") != len(references)
        or type(manifest.get("recorded_run_count")) is not int
        or manifest.get("recorded_run_count") != len(references)
    ):
        raise FinalAnalysisContractError("final execution manifest binding is invalid")
    storage_preflight = _validate_storage_preflight(
        evaluation.get("storage_preflight"),
        planned_run_count=len(references),
        require_combined=True,
    )
    validation = _validate_execution_validation(
        evaluation.get("execution_validation"),
        final_plan_sha256=final_plan_sha256,
        record_count=len(references),
    )
    payload_hashes = validation["record_payload_sha256s"]
    assert isinstance(payload_hashes, list)

    execution_dir = manifest_path.parent
    records: list[dict[str, object]] = []
    record_bindings: list[dict[str, object]] = []
    raw_snapshots: list[tuple[Path, str]] = []
    score_artifact_snapshots: list[tuple[Path, str]] = []
    seen_run_ids: set[str] = set()
    for index, (reference_value, payload_hash) in enumerate(
        zip(references, payload_hashes, strict=True),
        start=1,
    ):
        reference = _exact_mapping(
            reference_value,
            frozenset({"ordinal", "run_id", "path", "sha256"}),
            f"final execution record reference {index}",
        )
        expected_path = f"records/{index:08d}.json"
        if (
            type(reference.get("ordinal")) is not int
            or reference.get("ordinal") != index
            or reference.get("path") != expected_path
        ):
            raise FinalAnalysisContractError("final execution record path is invalid")
        run_id = _nonempty_string(
            reference.get("run_id"), f"final execution record {index} run_id"
        )
        if run_id in seen_run_ids:
            raise FinalAnalysisContractError(
                "final execution record run_id is duplicated"
            )
        seen_run_ids.add(run_id)
        path = execution_dir / expected_path
        declared_path = path.relative_to(destination).as_posix()
        reference_hash = _sha256(
            reference.get("sha256"), f"final execution record {index} raw hash"
        )
        if result_bindings.get(declared_path) != reference_hash:
            raise FinalAnalysisContractError(
                "final execution record is not identically bound as a result file"
            )
        record, raw = _read_json_file(
            path,
            f"final execution record {index}",
            require_canonical=True,
        )
        raw_hash = hashlib.sha256(raw).hexdigest()
        if raw_hash != reference_hash:
            raise FinalAnalysisContractError(
                "final execution record raw hash does not match manifest"
            )
        _exact_mapping(
            record,
            frozenset({"run", "metrics", "p_pre_zero_evidence", "execution_request"}),
            f"final execution record {index}",
        )
        run = record.get("run")
        if not isinstance(run, Mapping) or run.get("run_id") != run_id:
            raise FinalAnalysisContractError(
                "final execution record run_id differs from manifest"
            )
        observed_payload_hash = canonical_sha256(record)
        if observed_payload_hash != payload_hash:
            raise FinalAnalysisContractError(
                "final execution record payload hash differs from validation"
            )
        _normalize_score_evidence((record,))
        _score_artifact, score_snapshot = _score_artifact_binding(
            record,
            record_index=index,
            execution_dir=execution_dir,
            destination=destination,
            result_bindings=result_bindings,
        )
        if score_snapshot is not None:
            score_artifact_snapshots.append(score_snapshot)
        records.append(record)
        raw_snapshots.append((path, raw_hash))
        record_bindings.append(
            {
                "ordinal": index,
                "path": expected_path,
                "payload_sha256": observed_payload_hash,
                "raw_sha256": raw_hash,
                "run_id": run_id,
            }
        )

    actual_counts = Counter(str(record["run"]["status"]) for record in records)
    executed_counts = validation["executed_status_counts"]
    assert isinstance(executed_counts, Mapping)
    not_applicable = validation["not_applicable_count"]
    assert isinstance(not_applicable, int)
    if (
        any(
            actual_counts.get(status, 0) != int(executed_counts.get(status, 0))
            for status in ("completed", "failed", "timeout", "resource_exceeded")
        )
        or actual_counts.get("unavailable", 0)
        != int(executed_counts.get("unavailable", 0)) + not_applicable
    ):
        raise FinalAnalysisContractError(
            "record terminal statuses differ from execution validation"
        )

    config_path = _authority_file(
        selected_repository, freeze.get("config_path"), "frozen method"
    )
    frozen_method, config_raw = _read_json_file(
        config_path,
        "frozen method",
        require_canonical=False,
    )
    if hashlib.sha256(config_raw).hexdigest() != freeze.get("config_sha256"):
        raise FinalAnalysisContractError("frozen method hash differs from freeze")
    frozen_payload_hash = _sha256(
        frozen_method.get("payload_sha256"), "frozen method payload hash"
    )
    frozen_body = {
        key: value for key, value in frozen_method.items() if key != "payload_sha256"
    }
    if canonical_sha256(frozen_body) != frozen_payload_hash:
        raise FinalAnalysisContractError("frozen method payload hash does not match")
    if execution_input_hashes["frozen_method_sha256"] != frozen_payload_hash:
        raise FinalAnalysisContractError(
            "execution input hashes differ from frozen method"
        )
    selection_binding = frozen_method.get("artifact_bindings")
    if not isinstance(selection_binding, Mapping):
        raise FinalAnalysisContractError(
            "frozen method selection contract binding is absent"
        )
    selection_binding = selection_binding.get("selection_contract")
    selection_binding = _exact_mapping(
        selection_binding,
        frozenset({"path", "sha256"}),
        "selection contract binding",
    )
    selection_path = _authority_file(
        selected_repository,
        selection_binding.get("path"),
        "selection contract",
    )
    selection_contract, selection_raw = _read_json_file(
        selection_path,
        "selection contract",
        require_canonical=False,
    )
    selection_raw_hash = hashlib.sha256(selection_raw).hexdigest()
    if selection_raw_hash != selection_binding.get("sha256"):
        raise FinalAnalysisContractError(
            "selection contract hash differs from frozen method"
        )
    candidate = selection_contract.get("candidate_method_id")
    if candidate != frozen_method.get("candidate_method_id"):
        raise FinalAnalysisContractError(
            "selection contract candidate differs from frozen method"
        )
    scheduled_methods = frozen_method.get("scheduled_same_input_ids")
    numerical_methods = frozen_method.get("ready_comparison_population_ids")
    if (
        not isinstance(scheduled_methods, list)
        or selection_contract.get("scheduled_same_input_ids") != scheduled_methods
        or not isinstance(numerical_methods, list)
        or any(value not in scheduled_methods for value in numerical_methods)
    ):
        raise FinalAnalysisContractError(
            "frozen comparison populations differ from selection authority"
        )
    analysis_selection_contract = dict(selection_contract)
    analysis_selection_contract["numerical_comparison_population_ids"] = list(
        numerical_methods
    )

    protocol_path = _authority_file(
        selected_repository, freeze.get("protocol_path"), "protocol"
    )
    protocol, protocol_raw = _read_json_file(
        protocol_path,
        "protocol",
        require_canonical=False,
    )
    protocol_raw_hash = hashlib.sha256(protocol_raw).hexdigest()
    if protocol_raw_hash != freeze.get("protocol_sha256"):
        raise FinalAnalysisContractError("protocol hash differs from freeze")
    if execution_input_hashes["protocol_sha256"] != protocol_raw_hash:
        raise FinalAnalysisContractError("execution input hashes differ from protocol")
    metric_direction_contract = _frozen_metric_direction_contract(
        selected_repository,
        frozen_method,
        freeze,
        protocol,
    )

    input_bindings: dict[str, object] = {
        "evaluation_receipt_payload_sha256": canonical_sha256(dict(receipt)),
        "execution_action_denominator": {
            "executed_algorithmic_failure_count": validation[
                "executed_algorithmic_failure_count"
            ],
            "executed_completed_count": validation["executed_completed_count"],
            "executed_run_count": sum(int(count) for count in executed_counts.values()),
            "executed_terminal_status_counts": {
                str(status): int(executed_counts[status])
                for status in sorted(executed_counts)
            },
            "not_applicable_count": not_applicable,
            "status": "validated",
        },
        "execution_input_hashes": execution_input_hashes,
        "execution_validation_sha256": validation["validation_sha256"],
        "final_execution_manifest_path": expected_manifest_relative,
        "final_execution_manifest_sha256": manifest_raw_hash,
        "final_execution_payload_sha256": manifest["manifest_sha256"],
        "final_plan_sha256": final_plan_sha256,
        "frozen_method_path": config_path.relative_to(selected_repository).as_posix(),
        "frozen_method_raw_sha256": hashlib.sha256(config_raw).hexdigest(),
        "frozen_method_payload_sha256": frozen_payload_hash,
        "method_commit": freeze.get("method_commit"),
        "metric_direction_contract": metric_direction_contract,
        "planned_run_count": len(records),
        "protocol_path": protocol_path.relative_to(selected_repository).as_posix(),
        "protocol_raw_sha256": protocol_raw_hash,
        "protocol_payload_sha256": canonical_sha256(protocol),
        "record_bindings": record_bindings,
        "result_manifest_sha256": receipt_manifest_hash,
        "round_id": destination.name,
        "selection_contract_path": selection_path.relative_to(
            selected_repository
        ).as_posix(),
        "selection_contract_raw_sha256": selection_raw_hash,
        "selection_contract_payload_sha256": canonical_sha256(selection_contract),
        "scaling_evidence_sha256": scaling_evidence["evidence_sha256"],
        "storage_preflight_sha256": canonical_sha256(dict(storage_preflight)),
    }
    input_bindings["trajectory_evidence_sha256"] = trajectory_evidence[
        "evidence_sha256"
    ]
    snapshots: dict[str, object] = {
        "allowed_paths": allowed_paths,
        "authority_raw_snapshots": [
            (
                config_path,
                hashlib.sha256(config_raw).hexdigest(),
                "frozen method",
            ),
            (selection_path, selection_raw_hash, "selection contract"),
            (protocol_path, protocol_raw_hash, "protocol"),
        ],
        "destination": destination,
        "evaluation": dict(evaluation),
        "freeze": dict(freeze),
        "manifest_path": manifest_path,
        "manifest_raw_hash": manifest_raw_hash,
        "raw_snapshots": raw_snapshots,
        "score_artifact_snapshots": score_artifact_snapshots,
        "receipt_hash": canonical_sha256(dict(receipt)),
        "repository": selected_repository,
    }
    return (
        records,
        protocol,
        analysis_selection_contract,
        {
            "input_bindings": input_bindings,
            "snapshots": snapshots,
        },
    )


def _revalidate_evaluated_inputs(snapshots: Mapping[str, object]) -> None:
    from .final_runner import FinalRunnerContractError, _read_unique_file
    from .study import (
        StudyStateError,
        _validate_freeze,
        _validate_registry,
        _validate_result_files,
        _validate_state_record_chain,
        _verify_frozen_repository,
    )

    repository = snapshots["repository"]
    destination = snapshots["destination"]
    freeze_snapshot = snapshots["freeze"]
    if not isinstance(repository, Path) or not isinstance(destination, Path):
        raise FinalAnalysisContractError("analysis snapshot paths are invalid")
    try:
        manifest_raw = _read_unique_file(
            snapshots["manifest_path"], "final execution manifest"
        )
        if hashlib.sha256(manifest_raw).hexdigest() != snapshots["manifest_raw_hash"]:
            raise FinalAnalysisContractError(
                "final execution manifest changed during analysis"
            )
        raw_snapshots = snapshots["raw_snapshots"]
        assert isinstance(raw_snapshots, list)
        for path, expected_hash in raw_snapshots:
            raw = _read_unique_file(path, "final execution record")
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise FinalAnalysisContractError(
                    "final execution record changed during analysis"
                )
        score_artifact_snapshots = snapshots["score_artifact_snapshots"]
        assert isinstance(score_artifact_snapshots, list)
        for path, expected_hash in score_artifact_snapshots:
            raw = _read_unique_file(path, "p_pre_zero score artifact")
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise FinalAnalysisContractError(
                    "p_pre_zero score artifact changed during analysis"
                )
        authority_raw_snapshots = snapshots["authority_raw_snapshots"]
        assert isinstance(authority_raw_snapshots, list)
        for path, expected_hash, name in authority_raw_snapshots:
            raw = _read_unique_file(path, name)
            if hashlib.sha256(raw).hexdigest() != expected_hash:
                raise FinalAnalysisContractError(f"{name} changed during analysis")
        freeze = _validate_freeze(destination, repository)
        if dict(freeze) != freeze_snapshot:
            raise FinalAnalysisContractError("freeze changed during analysis")
        _validate_registry(
            repository,
            destination,
            freeze,
            expected_state="evaluated",
        )
        _materialization, _claim, receipt = _validate_state_record_chain(
            destination,
            freeze,
            expected_state="evaluated",
        )
        if (
            not isinstance(receipt, Mapping)
            or canonical_sha256(dict(receipt)) != snapshots["receipt_hash"]
        ):
            raise FinalAnalysisContractError(
                "evaluation receipt changed during analysis"
            )
        evaluation = receipt.get("result_manifest")
        assert isinstance(evaluation, Mapping)
        allowed_paths = _validate_result_files(repository, destination, evaluation)
        if allowed_paths != snapshots["allowed_paths"]:
            raise FinalAnalysisContractError(
                "evaluated result allowlist changed during analysis"
            )
        _verify_frozen_repository(
            repository,
            destination,
            allowed_result_paths=allowed_paths,
        )
    except (FinalRunnerContractError, StudyStateError) as error:
        raise FinalAnalysisContractError(str(error)) from error


def generate_final_analysis(
    repository: Path,
    round_dir: Path,
) -> dict[str, object]:
    """Generate a canonical report from one validated evaluated final round."""

    records, protocol, selection_contract, loaded = _evaluated_inputs(
        repository, round_dir
    )
    input_bindings = loaded["input_bindings"]
    snapshots = loaded["snapshots"]
    assert isinstance(input_bindings, Mapping)
    assert isinstance(snapshots, Mapping)
    report = build_final_analysis(
        records,
        protocol=protocol,
        selection_contract=selection_contract,
        input_bindings=input_bindings,
    )
    _revalidate_evaluated_inputs(snapshots)
    return report


__all__ = [
    "FinalAnalysisContractError",
    "build_final_analysis",
    "generate_final_analysis",
]
