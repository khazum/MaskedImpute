"""Prespecified, non-compensatory development candidate selection.

Every gate is reported separately.  Model seeds and paired technical views are
collapsed before biological-draw ranks or mechanism effects are calculated.
No weighted efficacy score is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import median
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any


_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_ENDPOINT_ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_VERSIONS = ("v27", "v28", "v29")
_ROLES = {
    "candidate",
    "learned_comparator",
    "learned_control",
    "observed_control",
}
_TRACKS = {"same_input", "external_reference"}
_METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
    "null_de_fpr",
)
_RANK_METRICS = ("mse", "mse_dropout", "gnrmse")
_PARETO_METRICS = (
    "mse_dropout",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
)
_RECORD_FIELDS = {
    "mechanism",
    "biological_id",
    "technical_view",
    "dataset_id",
    "dataset_sha256",
    "method",
    "method_sha256",
    "model_seed",
    "metric",
    "value",
    "status",
}
_INTERVAL_FIELDS = {
    "configuration",
    "endpoint",
    "comparison",
    "estimate",
    "ci_lower",
    "ci_upper",
    "status",
}
_FAILED_STATUSES = {"failed", "timeout", "unavailable", "resource_exceeded"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_AUTHORITY_PATHS = (
    "study/protocol.json",
    "study/development_panel.json",
    "study/methods.json",
    "study/ablations.json",
    "study/calibration_contract.json",
    "study/selection_contract.json",
    "study/development_search.json",
)


class SelectionAuthorityError(RuntimeError):
    """Raised when tracked authority or bound development artifacts are invalid."""


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a canonical lowercase identifier")
    return value


def _canonical_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an ordered sequence of strings")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class MethodDeclaration:
    """Selection-relevant role and execution policy for one denominator row."""

    id: str
    role: str
    track: str
    stochastic: bool
    required_for_claim: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _safe_id(self.id, "method id"))
        if self.role not in _ROLES:
            raise ValueError("method role is invalid")
        if self.track not in _TRACKS:
            raise ValueError("method track is invalid")
        if type(self.stochastic) is not bool:
            raise TypeError("stochastic must be boolean")
        if type(self.required_for_claim) is not bool:
            raise TypeError("required_for_claim must be boolean")


@dataclass(frozen=True, slots=True)
class CandidateAttempt:
    """One attempted candidate configuration retained in the selection table."""

    configuration_id: str
    version: str
    parent_configuration_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration_id",
            _safe_id(self.configuration_id, "configuration_id"),
        )
        if self.version not in _VERSIONS:
            raise ValueError("candidate version must be v27, v28, or v29")
        if self.parent_configuration_id is not None:
            object.__setattr__(
                self,
                "parent_configuration_id",
                _safe_id(self.parent_configuration_id, "parent_configuration_id"),
            )
        if self.version == "v27" and self.parent_configuration_id is not None:
            raise ValueError("v27 configurations must not have a parent")
        if self.version != "v27" and self.parent_configuration_id is None:
            raise ValueError("revision configurations require a parent")


@dataclass(frozen=True, slots=True)
class EndpointPolicy:
    """Repository-owned interpretation and materiality rule for one endpoint."""

    id: str
    comparison: str
    favorable_direction: str
    materiality_margin: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ENDPOINT_ID.fullmatch(self.id):
            raise ValueError("endpoint id must be canonical lowercase snake_case")
        if self.comparison != "candidate_minus_observed":
            raise ValueError("endpoint comparison must be candidate_minus_observed")
        if self.favorable_direction not in {"higher", "lower"}:
            raise ValueError("endpoint favorable_direction must be higher or lower")
        object.__setattr__(
            self,
            "materiality_margin",
            _finite_nonnegative(self.materiality_margin, "materiality_margin"),
        )


@dataclass(frozen=True, slots=True)
class RevisionPolicy:
    """Repository-owned retention margin for conditional revisions."""

    v29_max_dropout_mse_loss: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "v29_max_dropout_mse_loss",
            _finite_nonnegative(
                self.v29_max_dropout_mse_loss,
                "v29_max_dropout_mse_loss",
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchExclusion:
    """A prespecified search configuration excluded as an exact equivalent."""

    configuration_id: str
    version: str
    equivalent_to: str | None
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration_id",
            _safe_id(self.configuration_id, "excluded configuration_id"),
        )
        if self.equivalent_to is not None:
            object.__setattr__(
                self,
                "equivalent_to",
                _safe_id(self.equivalent_to, "equivalent_to"),
            )
        if self.version not in _VERSIONS:
            raise ValueError("excluded candidate version is invalid")
        if self.reason_code not in {
            "retained_identity_calibrator_equals_direct_score",
            "exploratory_budget_overrun_not_selection_eligible",
        }:
            raise ValueError("search exclusion reason code is not prespecified")
        if (
            self.reason_code == "exploratory_budget_overrun_not_selection_eligible"
            and self.equivalent_to is not None
        ):
            raise ValueError("budget-overrun exclusions must not claim equivalence")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "configuration_id": self.configuration_id,
            "version": self.version,
            "equivalent_to": self.equivalent_to,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class RetainedCalibrationBinding:
    """Exact development calibration artifact named by the tracked search ledger."""

    status: str
    path: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class SelectionAuthority:
    """Immutable authority assembled only from canonical repository artifacts."""

    mechanisms: tuple[str, ...]
    biological_ids: tuple[str, ...]
    technical_views: tuple[str, ...]
    model_seeds: tuple[int, ...]
    required_comparator_ids: tuple[str, ...]
    attempts: tuple[CandidateAttempt, ...]
    declarations: tuple[MethodDeclaration, ...]
    endpoint_policies: tuple[EndpointPolicy, ...]
    revision_policy: RevisionPolicy
    exclusions: tuple[SearchExclusion, ...]
    method_bindings: Mapping[str, str]
    base_maskimpute_config: Mapping[str, Any]
    base_maskimpute_config_sha256: str
    count_model_config: Mapping[str, Any]
    count_model_config_sha256: str
    dataset_qc_policy: Mapping[str, Any]
    dataset_qc_policy_sha256: str
    ablation_specs: tuple[Mapping[str, Any], ...]
    ablation_spec_ids: tuple[str, ...]
    ablation_run_keys: tuple[tuple[str, int], ...]
    calibration_equivalence_reason: str | None
    calibration_effect_status: str
    retained_calibration: RetainedCalibrationBinding
    count_score_manifest: RetainedCalibrationBinding
    file_sha256: Mapping[str, str]


def _freeze_detail_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain nonfinite values")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_detail_value(nested, name) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_detail_value(item, name) for item in value)
    raise TypeError(f"{name} must contain JSON-compatible values")


def _freeze_details(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            key: _freeze_detail_value(nested, "gate details")
            for key, nested in value.items()
        }
    )


def _thaw_detail_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_detail_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_detail_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class GateResult:
    """One transparent pass/fail decision and its observed value."""

    passed: bool
    value: float | int | None
    threshold: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise TypeError("gate passed value must be boolean")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise TypeError("gate value must be numeric or None")
            if not math.isfinite(float(self.value)):
                raise ValueError("gate value must be finite")
        if not isinstance(self.threshold, str) or not self.threshold:
            raise ValueError("gate threshold must be a nonempty string")
        if not isinstance(self.details, Mapping):
            raise TypeError("gate details must be a mapping")
        object.__setattr__(self, "details", _freeze_details(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "details": _thaw_detail_value(self.details),
        }


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    """All non-compensatory gates for one attempted configuration."""

    configuration_id: str
    version: str
    gate_items: tuple[tuple[str, GateResult], ...]
    efficacy_pass: bool
    safety_pass: bool
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    independent_draws: int

    @property
    def gates(self) -> dict[str, GateResult]:
        return dict(self.gate_items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "version": self.version,
            "gates": {name: gate.to_dict() for name, gate in self.gate_items},
            "efficacy_pass": self.efficacy_pass,
            "safety_pass": self.safety_pass,
            "eligible": self.eligible,
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "independent_draws": self.independent_draws,
        }


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """Deterministic Pareto/trigger report retaining every attempted candidate."""

    assessments: tuple[CandidateAssessment, ...]
    pareto_set: tuple[str, ...]
    selected_configuration: str | None
    trigger: str
    excluded_configurations: tuple[SearchExclusion, ...] = ()
    authority_bindings: Mapping[str, str] | None = None

    @property
    def by_configuration(self) -> dict[str, CandidateAssessment]:
        return {item.configuration_id: item for item in self.assessments}

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessments": [item.to_dict() for item in self.assessments],
            "pareto_set": list(self.pareto_set),
            "selected_configuration": self.selected_configuration,
            "trigger": self.trigger,
            "excluded_configurations": [
                item.to_dict() for item in self.excluded_configurations
            ],
            "authority_bindings": (
                None
                if self.authority_bindings is None
                else dict(self.authority_bindings)
            ),
            "selection_rule": (
                "all_hard_gates_then_lowest_version_then_configuration_id"
            ),
            "combined_score": None,
        }


@dataclass(frozen=True, slots=True)
class _ValidatedRecord:
    mechanism: str
    biological_id: str
    technical_view: str
    dataset_id: str
    dataset_sha256: str
    method: str
    method_sha256: str
    model_seed: int | None
    metric: str
    value: float | None
    status: str

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.mechanism,
            self.biological_id,
            self.technical_view,
            self.dataset_id,
            self.method,
            self.model_seed,
            self.metric,
        )


@dataclass(frozen=True, slots=True)
class _Interval:
    configuration: str
    endpoint: str
    estimate: float | None
    ci_lower: float | None
    ci_upper: float | None
    status: str


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_declarations(
    declarations: object,
    attempts: tuple[CandidateAttempt, ...],
) -> tuple[MethodDeclaration, ...]:
    if isinstance(declarations, (str, bytes)) or not isinstance(declarations, Sequence):
        raise TypeError("method declarations must be an ordered sequence")
    values = tuple(declarations)
    if not values or any(type(item) is not MethodDeclaration for item in values):
        raise TypeError(
            "method declarations must contain exact MethodDeclaration values"
        )
    identifiers = tuple(item.id for item in values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("method declarations contain duplicate identifiers")
    candidates = {item.id for item in values if item.role == "candidate"}
    attempted = {item.configuration_id for item in attempts}
    if candidates != attempted:
        raise ValueError(
            "candidate declarations must exactly match attempted configurations"
        )
    observed = [item for item in values if item.role == "observed_control"]
    if len(observed) != 1 or observed[0].id != "observed":
        raise ValueError("exactly one observed control named observed is required")
    return values


def _validate_attempts(attempts: object) -> tuple[CandidateAttempt, ...]:
    if isinstance(attempts, (str, bytes)) or not isinstance(attempts, Sequence):
        raise TypeError("attempts must be an ordered sequence")
    values = tuple(attempts)
    if not values or any(type(item) is not CandidateAttempt for item in values):
        raise TypeError("attempts must contain exact CandidateAttempt values")
    identifiers = tuple(item.configuration_id for item in values)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("attempts contain duplicate configuration identifiers")
    observed_versions = {item.version for item in values}
    if "v27" not in observed_versions:
        raise ValueError("candidate search must begin with v27")
    if "v29" in observed_versions and "v28" not in observed_versions:
        raise ValueError("v29 cannot be attempted before v28 was assessed")
    by_id = {item.configuration_id: item for item in values}
    for item in values:
        if item.version == "v27":
            continue
        parent = by_id.get(item.parent_configuration_id)
        expected_parent_version = _VERSIONS[_VERSIONS.index(item.version) - 1]
        if parent is None or parent.version != expected_parent_version:
            raise ValueError(
                f"{item.version} parent must be an assessed {expected_parent_version} configuration"
            )
    return tuple(
        sorted(
            values,
            key=lambda item: (_VERSIONS.index(item.version), item.configuration_id),
        )
    )


def _validate_records(
    records: object,
    declarations: tuple[MethodDeclaration, ...],
    *,
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
    technical_views: tuple[str, ...],
    model_seeds: tuple[int, ...],
    dataset_bindings: Mapping[tuple[str, str, str], tuple[str, str]],
    method_bindings: Mapping[str, str],
) -> tuple[_ValidatedRecord, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    by_method = {item.id: item for item in declarations}
    unique: dict[tuple[object, ...], _ValidatedRecord] = {}
    dataset_ids: dict[tuple[str, str, str], str] = {}
    dataset_units: dict[str, tuple[str, str, str]] = {}
    for index, raw in enumerate(records):
        if type(raw) is not dict or set(raw) != _RECORD_FIELDS:
            raise ValueError(f"record {index} has missing or extra fields")
        method = raw["method"]
        if method not in by_method:
            raise ValueError(f"record {index} method is undeclared")
        method_sha256 = raw["method_sha256"]
        if not isinstance(method_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", method_sha256
        ):
            raise ValueError(f"record {index} method_sha256 is invalid")
        if method_bindings.get(method) != method_sha256:
            raise ValueError(
                "record method checksum mismatches the tracked method or configuration"
            )
        mechanism = raw["mechanism"]
        biological_id = raw["biological_id"]
        technical_view = raw["technical_view"]
        if mechanism not in mechanisms:
            raise ValueError(f"record {index} mechanism is not prespecified")
        if biological_id not in biological_ids:
            raise ValueError(f"record {index} biological_id is not prespecified")
        if technical_view not in technical_views:
            raise ValueError(f"record {index} technical_view is not prespecified")
        dataset_id = raw["dataset_id"]
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError(f"record {index} dataset_id must be nonempty")
        dataset_key = (mechanism, biological_id, technical_view)
        expected_dataset = dataset_bindings.get(dataset_key)
        if expected_dataset is None:
            raise ValueError("record dataset is absent from the validated manifest")
        dataset_sha256 = raw["dataset_sha256"]
        if not isinstance(dataset_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", dataset_sha256
        ):
            raise ValueError(f"record {index} dataset_sha256 is invalid")
        if (dataset_id, dataset_sha256) != expected_dataset:
            raise ValueError(
                "record dataset identity or checksum mismatches the validated manifest"
            )
        previous_dataset = dataset_ids.setdefault(dataset_key, dataset_id)
        if previous_dataset != dataset_id:
            raise ValueError("methods are not paired on the same dataset_id")
        previous_unit = dataset_units.setdefault(dataset_id, dataset_key)
        if previous_unit != dataset_key:
            raise ValueError(
                "dataset_id is reused across purported independent dataset units"
            )
        metric = raw["metric"]
        if metric not in _METRICS:
            raise ValueError(f"record {index} metric is not prespecified")
        if metric == "mse_pre_dropout_zero" and mechanism != "symsim":
            raise ValueError(
                "mse_pre_dropout_zero is applicable only to exact SymSim truth"
            )
        seed = raw["model_seed"]
        declaration = by_method[method]
        if declaration.stochastic:
            if (
                isinstance(seed, bool)
                or type(seed) is not int
                or seed not in model_seeds
            ):
                raise TypeError(f"record {index} model_seed is not prespecified")
        elif seed is not None:
            raise ValueError(f"record {index} deterministic method seed must be null")
        status = raw["status"]
        if status not in {"completed", *_FAILED_STATUSES}:
            raise ValueError(f"record {index} status is invalid")
        if status == "completed":
            value = _finite_nonnegative(raw["value"], f"record {index} value")
        else:
            if raw["value"] is not None:
                raise ValueError("non-completed result rows must have null value")
            value = None
        validated = _ValidatedRecord(
            mechanism=mechanism,
            biological_id=biological_id,
            technical_view=technical_view,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            method=method,
            method_sha256=method_sha256,
            model_seed=seed,
            metric=metric,
            value=value,
            status=status,
        )
        if validated.identity in unique and unique[validated.identity] != validated:
            raise ValueError("conflicting duplicate result identity")
        unique[validated.identity] = validated
    return tuple(unique[key] for key in sorted(unique, key=repr))


def _expected_mechanisms(metric: str, mechanisms: tuple[str, ...]) -> tuple[str, ...]:
    return ("symsim",) if metric == "mse_pre_dropout_zero" else mechanisms


def _record_lookup(
    records: tuple[_ValidatedRecord, ...],
) -> dict[tuple[object, ...], _ValidatedRecord]:
    return {
        (
            item.method,
            item.metric,
            item.mechanism,
            item.biological_id,
            item.technical_view,
            item.model_seed,
        ): item
        for item in records
    }


def _expected_seeds(
    declaration: MethodDeclaration,
    model_seeds: tuple[int, ...],
) -> tuple[int | None, ...]:
    return model_seeds if declaration.stochastic else (None,)


def _method_complete(
    method: MethodDeclaration,
    lookup: Mapping[tuple[object, ...], _ValidatedRecord],
    *,
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
    technical_views: tuple[str, ...],
    model_seeds: tuple[int, ...],
) -> bool:
    for metric in _METRICS:
        for mechanism in _expected_mechanisms(metric, mechanisms):
            for biological_id in biological_ids:
                for technical_view in technical_views:
                    for seed in _expected_seeds(method, model_seeds):
                        record = lookup.get(
                            (
                                method.id,
                                metric,
                                mechanism,
                                biological_id,
                                technical_view,
                                seed,
                            )
                        )
                        if (
                            record is None
                            or record.status != "completed"
                            or record.value is None
                        ):
                            return False
    return True


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _collapse_draws(
    declarations: tuple[MethodDeclaration, ...],
    lookup: Mapping[tuple[object, ...], _ValidatedRecord],
    *,
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
    technical_views: tuple[str, ...],
    model_seeds: tuple[int, ...],
) -> dict[tuple[str, str, str, str], float]:
    collapsed: dict[tuple[str, str, str, str], float] = {}
    for method in declarations:
        seeds = _expected_seeds(method, model_seeds)
        for metric in _METRICS:
            for mechanism in _expected_mechanisms(metric, mechanisms):
                for biological_id in biological_ids:
                    view_values = []
                    complete = True
                    for technical_view in technical_views:
                        seed_values = []
                        for seed in seeds:
                            record = lookup.get(
                                (
                                    method.id,
                                    metric,
                                    mechanism,
                                    biological_id,
                                    technical_view,
                                    seed,
                                )
                            )
                            if (
                                record is None
                                or record.status != "completed"
                                or record.value is None
                            ):
                                complete = False
                                break
                            seed_values.append(record.value)
                        if not complete:
                            break
                        view_values.append(_mean(seed_values))
                    if complete:
                        collapsed[(method.id, metric, mechanism, biological_id)] = (
                            _mean(view_values)
                        )
    return collapsed


def _mechanism_value(
    collapsed: Mapping[tuple[str, str, str, str], float],
    method: str,
    metric: str,
    mechanism: str,
    biological_ids: tuple[str, ...],
) -> float | None:
    values = [
        collapsed[(method, metric, mechanism, biological_id)]
        for biological_id in biological_ids
        if (method, metric, mechanism, biological_id) in collapsed
    ]
    return float(median(values)) if len(values) == len(biological_ids) else None


def _overall_value(
    collapsed: Mapping[tuple[str, str, str, str], float],
    method: str,
    metric: str,
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
) -> float | None:
    values = [
        collapsed[(method, metric, mechanism, biological_id)]
        for mechanism in _expected_mechanisms(metric, mechanisms)
        for biological_id in biological_ids
        if (method, metric, mechanism, biological_id) in collapsed
    ]
    expected = len(_expected_mechanisms(metric, mechanisms)) * len(biological_ids)
    return float(median(values)) if len(values) == expected else None


def _average_rank(target: float, values: Sequence[float]) -> float:
    below = sum(value < target for value in values)
    tied = sum(value == target for value in values)
    return 1.0 + below + tied / 2.0


def _rank_summary(
    collapsed: Mapping[tuple[str, str, str, str], float],
    candidate: str,
    metric: str,
    declarations: tuple[MethodDeclaration, ...],
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
) -> tuple[float | None, tuple[tuple[str, float | None], ...]]:
    same_input = tuple(
        item.id
        for item in declarations
        if item.track == "same_input" and item.role != "candidate"
    )
    ranks = []
    mechanism_ranks: list[tuple[str, float | None]] = []
    for mechanism in _expected_mechanisms(metric, mechanisms):
        local_ranks = []
        for biological_id in biological_ids:
            key = (candidate, metric, mechanism, biological_id)
            if key not in collapsed:
                local_ranks = []
                break
            values = [
                collapsed[(method, metric, mechanism, biological_id)]
                for method in same_input
                if (method, metric, mechanism, biological_id) in collapsed
            ]
            rank = _average_rank(collapsed[key], values)
            local_ranks.append(rank)
            ranks.append(rank)
        mechanism_ranks.append(
            (
                mechanism,
                float(median(local_ranks))
                if len(local_ranks) == len(biological_ids)
                else None,
            )
        )
    expected = len(_expected_mechanisms(metric, mechanisms)) * len(biological_ids)
    overall = float(median(ranks)) if len(ranks) == expected else None
    return overall, tuple(mechanism_ranks)


def _paired_mechanism_effect(
    collapsed: Mapping[tuple[str, str, str, str], float],
    candidate: str,
    comparators: tuple[str, ...],
    metric: str,
    mechanism: str,
    biological_ids: tuple[str, ...],
    *,
    positive_is_improvement: bool,
) -> tuple[float | None, str | None]:
    """Return the median paired percentage against the strongest comparator."""

    complete_comparators = []
    for comparator in comparators:
        values = [
            collapsed.get((comparator, metric, mechanism, biological_id))
            for biological_id in biological_ids
        ]
        if all(value is not None for value in values):
            complete_comparators.append(
                (
                    float(median(value for value in values if value is not None)),
                    comparator,
                )
            )
    if not complete_comparators:
        return None, None
    _aggregate, strongest = min(
        complete_comparators, key=lambda item: (item[0], item[1])
    )
    effects = []
    for biological_id in biological_ids:
        candidate_value = collapsed.get((candidate, metric, mechanism, biological_id))
        comparator_value = collapsed.get((strongest, metric, mechanism, biological_id))
        if candidate_value is None or comparator_value is None:
            return None, strongest
        if comparator_value == 0:
            if candidate_value != 0:
                return None, strongest
            effects.append(0.0)
            continue
        difference = (
            comparator_value - candidate_value
            if positive_is_improvement
            else candidate_value - comparator_value
        )
        effects.append(difference / abs(comparator_value))
    return float(median(effects)), strongest


def _pareto_dominators(
    collapsed: Mapping[tuple[str, str, str, str], float],
    candidate: str,
    declarations: tuple[MethodDeclaration, ...],
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
    *,
    mechanism: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dimensions = tuple(
        metric
        for metric in _PARETO_METRICS
        if not (metric == "mse_pre_dropout_zero" and mechanism not in {None, "symsim"})
    )

    def scoped_value(method: str, metric: str) -> float | None:
        if mechanism is None:
            return _overall_value(collapsed, method, metric, mechanisms, biological_ids)
        return _mechanism_value(collapsed, method, metric, mechanism, biological_ids)

    candidate_vector = tuple(scoped_value(candidate, metric) for metric in dimensions)
    if any(value is None for value in candidate_vector):
        return ("incomplete_candidate_vector",), dimensions
    target = tuple(float(value) for value in candidate_vector if value is not None)
    dominators = []
    for declaration in declarations:
        if declaration.track != "same_input" or declaration.id == candidate:
            continue
        vector = tuple(scoped_value(declaration.id, metric) for metric in dimensions)
        if any(value is None for value in vector):
            continue
        comparison = tuple(float(value) for value in vector if value is not None)
        if all(
            left <= right for left, right in zip(comparison, target, strict=True)
        ) and any(left < right for left, right in zip(comparison, target, strict=True)):
            dominators.append(declaration.id)
    return tuple(sorted(dominators)), dimensions


def _validate_intervals(
    intervals: object,
    attempts: tuple[CandidateAttempt, ...],
    endpoints: tuple[str, ...],
    endpoint_policies: Mapping[str, EndpointPolicy],
) -> dict[tuple[str, str], _Interval]:
    if isinstance(intervals, (str, bytes)) or not isinstance(intervals, Sequence):
        raise TypeError("orthogonal intervals must be a sequence")
    candidates = {item.configuration_id for item in attempts}
    unique: dict[tuple[str, str], _Interval] = {}
    for index, raw in enumerate(intervals):
        if type(raw) is not dict or set(raw) != _INTERVAL_FIELDS:
            raise ValueError(f"orthogonal interval {index} has missing or extra fields")
        configuration = raw["configuration"]
        endpoint = raw["endpoint"]
        if configuration not in candidates:
            raise ValueError("orthogonal interval configuration is not attempted")
        if endpoint not in endpoints:
            raise ValueError("orthogonal endpoint is not prespecified")
        if raw["comparison"] != "observed":
            raise ValueError("orthogonal intervals must compare against observed")
        status = raw["status"]
        if status not in {"completed", *_FAILED_STATUSES}:
            raise ValueError("orthogonal interval status is invalid")
        if status == "completed":
            estimate = _finite_number(raw["estimate"], "orthogonal estimate")
            lower = _finite_number(raw["ci_lower"], "orthogonal ci_lower")
            upper = _finite_number(raw["ci_upper"], "orthogonal ci_upper")
            if not lower <= estimate <= upper:
                raise ValueError("orthogonal interval must contain its estimate")
        else:
            if any(
                raw[field] is not None
                for field in (
                    "estimate",
                    "ci_lower",
                    "ci_upper",
                )
            ):
                raise ValueError("failed orthogonal intervals must have null values")
            estimate = lower = upper = None
        value = _Interval(
            configuration=configuration,
            endpoint=endpoint,
            estimate=estimate,
            ci_lower=lower,
            ci_upper=upper,
            status=status,
        )
        key = (configuration, endpoint)
        if key in unique and unique[key] != value:
            raise ValueError("conflicting duplicate orthogonal interval")
        unique[key] = value
    return unique


def _gate(
    passed: bool,
    value: float | int | None,
    threshold: str,
    **details: Any,
) -> GateResult:
    return GateResult(passed, value, threshold, details)


def _assessment(
    attempt: CandidateAttempt,
    *,
    declarations: tuple[MethodDeclaration, ...],
    complete: Mapping[str, bool],
    collapsed: Mapping[tuple[str, str, str, str], float],
    interval_lookup: Mapping[tuple[str, str], _Interval],
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
    endpoints: tuple[str, ...],
    endpoint_policies: Mapping[str, EndpointPolicy],
) -> CandidateAssessment:
    candidate = attempt.configuration_id
    gates: dict[str, GateResult] = {}
    gates["candidate_completeness"] = _gate(
        complete[candidate],
        int(complete[candidate]),
        "all prespecified candidate metric rows completed",
    )
    incomplete_required = tuple(
        sorted(
            item.id
            for item in declarations
            if item.role != "candidate"
            and item.required_for_claim
            and not complete[item.id]
        )
    )
    gates["required_comparator_completeness"] = _gate(
        not incomplete_required,
        len(incomplete_required),
        "zero incomplete required comparators",
        incomplete_methods=incomplete_required,
    )
    for metric in _RANK_METRICS:
        rank, mechanism_ranks = _rank_summary(
            collapsed,
            candidate,
            metric,
            declarations,
            mechanisms,
            biological_ids,
        )
        gates[f"rank_{metric}"] = _gate(
            rank is not None and rank <= 2.0,
            rank,
            "median biological-draw rank <= 2",
            overall_rank=rank,
            mechanism_ranks=mechanism_ranks,
            mechanism_pass=tuple(
                (
                    mechanism,
                    value is not None and value <= 2.0,
                )
                for mechanism, value in mechanism_ranks
            ),
        )
    dominators, pareto_dimensions = _pareto_dominators(
        collapsed,
        candidate,
        declarations,
        mechanisms,
        biological_ids,
        mechanism=None,
    )
    mechanism_pareto = []
    for mechanism in mechanisms:
        local_dominators, local_dimensions = _pareto_dominators(
            collapsed,
            candidate,
            declarations,
            mechanisms,
            biological_ids,
            mechanism=mechanism,
        )
        mechanism_pareto.append(
            (
                mechanism,
                {
                    "passed": not local_dominators,
                    "dominated_by": local_dominators,
                    "dimensions": local_dimensions,
                },
            )
        )
    gates["pareto_non_dominated"] = _gate(
        not dominators,
        len(dominators),
        "no same-input method weakly better on all four dimensions and strictly better on one",
        dominated_by=dominators,
        dimensions=pareto_dimensions,
        mechanism_results=tuple(mechanism_pareto),
    )
    learned = tuple(
        item.id
        for item in declarations
        if item.track == "same_input"
        and item.role in {"learned_comparator", "learned_control"}
    )
    improvements: dict[str, float | None] = {}
    improvement_comparators: dict[str, str | None] = {}
    for mechanism in mechanisms:
        effect, strongest = _paired_mechanism_effect(
            collapsed,
            candidate,
            learned,
            "mse_dropout",
            mechanism,
            biological_ids,
            positive_is_improvement=True,
        )
        improvements[mechanism] = effect
        improvement_comparators[mechanism] = strongest
    improved = tuple(
        mechanism
        for mechanism in mechanisms
        if improvements[mechanism] is not None and improvements[mechanism] >= 0.05
    )
    gates["dropout_improvement"] = _gate(
        len(improved) >= 3,
        len(improved),
        ">=5% MSE-dropout improvement over strongest learned comparator in >=3 mechanisms",
        improved_mechanisms=improved,
        mechanism_improvements=tuple(sorted(improvements.items())),
        strongest_comparators=tuple(sorted(improvement_comparators.items())),
    )

    def degradation_gate(metric: str) -> GateResult:
        values: dict[str, float | None] = {}
        comparators: dict[str, str | None] = {}
        for mechanism in _expected_mechanisms(metric, mechanisms):
            effect, strongest = _paired_mechanism_effect(
                collapsed,
                candidate,
                learned,
                metric,
                mechanism,
                biological_ids,
                positive_is_improvement=False,
            )
            values[mechanism] = effect
            comparators[mechanism] = strongest
        finite_values = [value for value in values.values() if value is not None]
        maximum = max(finite_values) if len(finite_values) == len(values) else None
        return _gate(
            maximum is not None and maximum <= 0.10,
            maximum,
            "maximum mechanism degradation <=10% versus best learned comparator",
            mechanism_degradation=tuple(sorted(values.items())),
            strongest_comparators=tuple(sorted(comparators.items())),
            undefined=maximum is None,
            metric=metric,
        )

    gates["prezero_degradation"] = degradation_gate("mse_pre_dropout_zero")
    gates["corr_err_degradation"] = degradation_gate("corr_err")
    fpr_values = []
    fpr_deltas = []
    mechanism_fpr = []
    for mechanism in mechanisms:
        candidate_fpr = _mechanism_value(
            collapsed, candidate, "null_de_fpr", mechanism, biological_ids
        )
        observed_fpr = _mechanism_value(
            collapsed, "observed", "null_de_fpr", mechanism, biological_ids
        )
        if candidate_fpr is not None and observed_fpr is not None:
            fpr_values.append(candidate_fpr)
            fpr_deltas.append(candidate_fpr - observed_fpr)
            mechanism_fpr.append(
                (
                    mechanism,
                    {
                        "candidate_fpr": candidate_fpr,
                        "observed_fpr": observed_fpr,
                        "above_observed": candidate_fpr - observed_fpr,
                        "passed": candidate_fpr <= 0.06
                        and candidate_fpr - observed_fpr <= 0.01,
                    },
                )
            )
        else:
            mechanism_fpr.append(
                (
                    mechanism,
                    {
                        "candidate_fpr": candidate_fpr,
                        "observed_fpr": observed_fpr,
                        "above_observed": None,
                        "passed": False,
                    },
                )
            )
    maximum_fpr = max(fpr_values) if len(fpr_values) == len(mechanisms) else None
    maximum_delta = max(fpr_deltas) if len(fpr_deltas) == len(mechanisms) else None
    gates["null_de_safety"] = _gate(
        maximum_fpr is not None
        and maximum_delta is not None
        and maximum_fpr <= 0.06
        and maximum_delta <= 0.01,
        maximum_fpr,
        "FPR <=0.06 and <=0.01 above observed in every mechanism",
        maximum_fpr=maximum_fpr,
        maximum_above_observed=maximum_delta,
        mechanism_results=tuple(mechanism_fpr),
    )
    missing_endpoints = []
    failed_endpoints = []
    unsafe_endpoints = []
    for endpoint in endpoints:
        interval = interval_lookup.get((candidate, endpoint))
        if interval is None:
            missing_endpoints.append(endpoint)
        elif interval.status != "completed":
            failed_endpoints.append(endpoint)
        elif interval.ci_lower is not None and interval.ci_upper is not None:
            policy = endpoint_policies[endpoint]
            if (
                policy.favorable_direction == "higher"
                and interval.ci_upper < -policy.materiality_margin
            ) or (
                policy.favorable_direction == "lower"
                and interval.ci_lower > policy.materiality_margin
            ):
                unsafe_endpoints.append(endpoint)
    gates["orthogonal_safety"] = _gate(
        not missing_endpoints and not failed_endpoints and not unsafe_endpoints,
        len(unsafe_endpoints),
        "no hierarchical interval establishes degradation beyond its tracked margin",
        missing_endpoints=tuple(missing_endpoints),
        failed_endpoints=tuple(failed_endpoints),
        unsafe_endpoints=tuple(unsafe_endpoints),
        endpoint_policies=tuple(
            (
                endpoint,
                {
                    "comparison": endpoint_policies[endpoint].comparison,
                    "favorable_direction": endpoint_policies[
                        endpoint
                    ].favorable_direction,
                    "materiality_margin": endpoint_policies[
                        endpoint
                    ].materiality_margin,
                },
            )
            for endpoint in endpoints
        ),
    )
    efficacy_names = (
        "candidate_completeness",
        "required_comparator_completeness",
        "rank_mse",
        "rank_mse_dropout",
        "rank_gnrmse",
        "pareto_non_dominated",
        "dropout_improvement",
    )
    safety_names = (
        "prezero_degradation",
        "corr_err_degradation",
        "null_de_safety",
        "orthogonal_safety",
    )
    efficacy_pass = all(gates[name].passed for name in efficacy_names)
    safety_pass = all(gates[name].passed for name in safety_names)
    reasons = []
    if not gates["candidate_completeness"].passed:
        reasons.append("incomplete_candidate_metrics")
    reasons.extend(
        f"required_comparator_incomplete:{method}" for method in incomplete_required
    )
    for metric in _RANK_METRICS:
        if not gates[f"rank_{metric}"].passed:
            reasons.append(f"median_rank_exceeds_two:{metric}")
    if not gates["pareto_non_dominated"].passed:
        reasons.append("pareto_dominated")
    if not gates["dropout_improvement"].passed:
        reasons.append("insufficient_dropout_improvement")
    for name in safety_names:
        if not gates[name].passed:
            reasons.append(name)
    gate_order = (
        "candidate_completeness",
        "required_comparator_completeness",
        "rank_mse",
        "rank_mse_dropout",
        "rank_gnrmse",
        "pareto_non_dominated",
        "dropout_improvement",
        "prezero_degradation",
        "corr_err_degradation",
        "null_de_safety",
        "orthogonal_safety",
    )
    independent_draws = sum(
        (candidate, "mse", mechanism, biological_id) in collapsed
        for mechanism in mechanisms
        for biological_id in biological_ids
    )
    return CandidateAssessment(
        configuration_id=candidate,
        version=attempt.version,
        gate_items=tuple((name, gates[name]) for name in gate_order),
        efficacy_pass=efficacy_pass,
        safety_pass=safety_pass,
        eligible=efficacy_pass and safety_pass,
        ineligibility_reasons=tuple(reasons),
        independent_draws=independent_draws,
    )


def _revision_retention_gate(
    attempt: CandidateAttempt,
    assessment: CandidateAssessment,
    *,
    by_assessment: Mapping[str, CandidateAssessment],
    collapsed: Mapping[tuple[str, str, str, str], float],
    interval_lookup: Mapping[tuple[str, str], _Interval],
    mechanisms: tuple[str, ...],
    biological_ids: tuple[str, ...],
    endpoints: tuple[str, ...],
    endpoint_policies: Mapping[str, EndpointPolicy],
    revision_policy: RevisionPolicy,
) -> GateResult:
    if attempt.version == "v27":
        return _gate(
            True,
            1,
            "not applicable to the initial v27 configuration",
            reason_code="initial_configuration_not_a_revision",
        )
    assert attempt.parent_configuration_id is not None
    parent_id = attempt.parent_configuration_id
    parent = by_assessment[parent_id]
    candidate_id = attempt.configuration_id
    if attempt.version == "v28":
        candidate_vector = tuple(
            _overall_value(
                collapsed,
                candidate_id,
                metric,
                mechanisms,
                biological_ids,
            )
            for metric in _PARETO_METRICS
        )
        parent_vector = tuple(
            _overall_value(
                collapsed,
                parent_id,
                metric,
                mechanisms,
                biological_ids,
            )
            for metric in _PARETO_METRICS
        )
        complete = not any(
            value is None for value in (*candidate_vector, *parent_vector)
        )
        pareto_improved = (
            complete
            and all(
                float(candidate) <= float(parent_value)
                for candidate, parent_value in zip(
                    candidate_vector, parent_vector, strict=True
                )
            )
            and any(
                float(candidate) < float(parent_value)
                for candidate, parent_value in zip(
                    candidate_vector, parent_vector, strict=True
                )
            )
        )
        zero_de_safe = (
            assessment.gates["prezero_degradation"].passed
            and assessment.gates["null_de_safety"].passed
        )
        if not zero_de_safe:
            reason = "v28_zero_or_de_safety_violation"
        elif not pareto_improved:
            reason = "v28_no_strict_pareto_improvement"
        else:
            reason = "v28_pareto_improved_with_zero_de_safety"
        return _gate(
            pareto_improved and zero_de_safe,
            int(pareto_improved and zero_de_safe),
            "strict Pareto improvement over v27 with zero-preservation and null-DE safety",
            reason_code=reason,
            parent_configuration_id=parent_id,
            dimensions=_PARETO_METRICS,
            candidate_vector=candidate_vector,
            parent_vector=parent_vector,
            strict_pareto_improvement=pareto_improved,
            zero_preservation_safe=assessment.gates["prezero_degradation"].passed,
            null_de_safe=assessment.gates["null_de_safety"].passed,
        )

    candidate_corr = _overall_value(
        collapsed, candidate_id, "corr_err", mechanisms, biological_ids
    )
    parent_corr = _overall_value(
        collapsed, parent_id, "corr_err", mechanisms, biological_ids
    )
    corr_improved = (
        candidate_corr is not None
        and parent_corr is not None
        and candidate_corr < parent_corr
    )
    endpoint_improvements: list[str] = []
    for endpoint in endpoints:
        candidate_interval = interval_lookup.get((candidate_id, endpoint))
        parent_interval = interval_lookup.get((parent_id, endpoint))
        if (
            candidate_interval is None
            or parent_interval is None
            or candidate_interval.status != "completed"
            or parent_interval.status != "completed"
            or candidate_interval.estimate is None
            or parent_interval.estimate is None
        ):
            continue
        policy = endpoint_policies[endpoint]
        if (
            policy.favorable_direction == "higher"
            and candidate_interval.estimate > parent_interval.estimate
        ) or (
            policy.favorable_direction == "lower"
            and candidate_interval.estimate < parent_interval.estimate
        ):
            endpoint_improvements.append(endpoint)
    dropout_losses: dict[str, float | None] = {}
    for mechanism in mechanisms:
        loss, _parent = _paired_mechanism_effect(
            collapsed,
            candidate_id,
            (parent_id,),
            "mse_dropout",
            mechanism,
            biological_ids,
            positive_is_improvement=False,
        )
        dropout_losses[mechanism] = loss
    finite_losses = [value for value in dropout_losses.values() if value is not None]
    maximum_loss = (
        max(finite_losses) if len(finite_losses) == len(dropout_losses) else None
    )
    structure_improved = corr_improved or bool(endpoint_improvements)
    dropout_safe = (
        maximum_loss is not None
        and maximum_loss <= revision_policy.v29_max_dropout_mse_loss
    )
    if not structure_improved:
        reason = "v29_no_structure_or_downstream_improvement"
    elif not dropout_safe:
        reason = "v29_material_dropout_mse_loss"
    else:
        reason = "v29_structure_improved_without_material_dropout_loss"
    return _gate(
        structure_improved and dropout_safe,
        maximum_loss,
        "structure or downstream improvement with no material dropout-MSE loss",
        reason_code=reason,
        parent_configuration_id=parent.configuration_id,
        corr_err_improved=corr_improved,
        candidate_corr_err=candidate_corr,
        parent_corr_err=parent_corr,
        improved_orthogonal_endpoints=tuple(endpoint_improvements),
        mechanism_dropout_mse_loss=tuple(sorted(dropout_losses.items())),
        maximum_dropout_mse_loss=maximum_loss,
        materiality_margin=revision_policy.v29_max_dropout_mse_loss,
    )


def _attach_revision_gates(
    attempts: tuple[CandidateAttempt, ...],
    assessments: tuple[CandidateAssessment, ...],
    **context: Any,
) -> tuple[CandidateAssessment, ...]:
    by_assessment = {item.configuration_id: item for item in assessments}
    revised = []
    for attempt, assessment in zip(attempts, assessments, strict=True):
        gate = _revision_retention_gate(
            attempt,
            assessment,
            by_assessment=by_assessment,
            **context,
        )
        gate_items = (*assessment.gate_items, ("revision_retention", gate))
        reasons = assessment.ineligibility_reasons
        if not gate.passed:
            reasons = (*reasons, "revision_not_retained")
        revised.append(
            CandidateAssessment(
                configuration_id=assessment.configuration_id,
                version=assessment.version,
                gate_items=gate_items,
                efficacy_pass=assessment.efficacy_pass,
                safety_pass=assessment.safety_pass,
                eligible=assessment.eligible and gate.passed,
                ineligibility_reasons=reasons,
                independent_draws=assessment.independent_draws,
            )
        )
    return tuple(revised)


def _evaluate_development_candidates(
    records: object,
    attempts: object,
    declarations: object,
    orthogonal_intervals: object,
    *,
    mechanisms: object,
    biological_ids: object,
    technical_views: object,
    model_seeds: object,
    required_orthogonal_endpoints: object,
    dataset_bindings: Mapping[tuple[str, str, str], tuple[str, str]],
    method_bindings: Mapping[str, str],
    endpoint_policies: object,
    revision_policy: RevisionPolicy,
    exclusions: object,
) -> SelectionReport:
    """Apply the prespecified hard gates and conditional revision trigger."""

    attempt_values = _validate_attempts(attempts)
    declaration_values = _validate_declarations(declarations, attempt_values)
    mechanism_values = _canonical_string_tuple(mechanisms, "mechanisms")
    if mechanism_values != ("symsim", "sergio", "sparsim", "semisynthetic"):
        raise ValueError("mechanisms must equal the four-mechanism publication panel")
    biological_values = _canonical_string_tuple(biological_ids, "biological_ids")
    technical_values = _canonical_string_tuple(technical_views, "technical_views")
    if technical_values != ("moderate", "severe"):
        raise ValueError("technical_views must equal moderate and severe")
    if isinstance(model_seeds, (str, bytes)) or not isinstance(model_seeds, Sequence):
        raise TypeError("model_seeds must be an ordered integer sequence")
    seed_values = tuple(model_seeds)
    if seed_values != (42, 43, 44) or any(
        type(seed) is not int for seed in seed_values
    ):
        raise ValueError("model_seeds must equal 42, 43, and 44")
    endpoints = tuple(
        sorted(
            _canonical_string_tuple(
                required_orthogonal_endpoints,
                "required_orthogonal_endpoints",
            )
        )
    )
    if isinstance(endpoint_policies, (str, bytes)) or not isinstance(
        endpoint_policies, Sequence
    ):
        raise TypeError("endpoint_policies must be an ordered sequence")
    policy_values = tuple(endpoint_policies)
    if any(type(item) is not EndpointPolicy for item in policy_values):
        raise TypeError("endpoint_policies must contain exact EndpointPolicy values")
    if len({item.id for item in policy_values}) != len(policy_values):
        raise ValueError("endpoint_policies contain duplicate identifiers")
    policy_lookup = {item.id: item for item in policy_values}
    if set(policy_lookup) != set(endpoints):
        raise ValueError("endpoint_policies must exactly cover required endpoints")
    if type(revision_policy) is not RevisionPolicy:
        raise TypeError("revision_policy must be an exact RevisionPolicy")
    if isinstance(exclusions, (str, bytes)) or not isinstance(exclusions, Sequence):
        raise TypeError("exclusions must be an ordered sequence")
    exclusion_values = tuple(exclusions)
    if any(type(item) is not SearchExclusion for item in exclusion_values):
        raise TypeError("exclusions must contain exact SearchExclusion values")
    excluded_ids = tuple(item.configuration_id for item in exclusion_values)
    if len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError(
            "search exclusions contain duplicate configuration identifiers"
        )
    attempted_ids = {item.configuration_id for item in attempt_values}
    if attempted_ids.intersection(excluded_ids):
        raise ValueError("a configuration cannot be both attempted and excluded")
    if any(
        item.equivalent_to is not None and item.equivalent_to not in attempted_ids
        for item in exclusion_values
    ):
        raise ValueError(
            "excluded equivalence target must be an attempted configuration"
        )
    exclusion_values = tuple(
        sorted(
            exclusion_values,
            key=lambda item: (
                _VERSIONS.index(item.version),
                item.configuration_id,
            ),
        )
    )
    validated_records = _validate_records(
        records,
        declaration_values,
        mechanisms=mechanism_values,
        biological_ids=biological_values,
        technical_views=technical_values,
        model_seeds=seed_values,
        dataset_bindings=dataset_bindings,
        method_bindings=method_bindings,
    )
    lookup = _record_lookup(validated_records)
    complete = {
        declaration.id: _method_complete(
            declaration,
            lookup,
            mechanisms=mechanism_values,
            biological_ids=biological_values,
            technical_views=technical_values,
            model_seeds=seed_values,
        )
        for declaration in declaration_values
    }
    collapsed = _collapse_draws(
        declaration_values,
        lookup,
        mechanisms=mechanism_values,
        biological_ids=biological_values,
        technical_views=technical_values,
        model_seeds=seed_values,
    )
    interval_lookup = _validate_intervals(
        orthogonal_intervals,
        attempt_values,
        endpoints,
        policy_lookup,
    )
    assessments = tuple(
        _assessment(
            attempt,
            declarations=declaration_values,
            complete=complete,
            collapsed=collapsed,
            interval_lookup=interval_lookup,
            mechanisms=mechanism_values,
            biological_ids=biological_values,
            endpoints=endpoints,
            endpoint_policies=policy_lookup,
        )
        for attempt in attempt_values
    )
    assessments = _attach_revision_gates(
        attempt_values,
        assessments,
        collapsed=collapsed,
        interval_lookup=interval_lookup,
        mechanisms=mechanism_values,
        biological_ids=biological_values,
        endpoints=endpoints,
        endpoint_policies=policy_lookup,
        revision_policy=revision_policy,
    )
    pareto_set = tuple(
        item.configuration_id
        for item in assessments
        if item.gates["pareto_non_dominated"].passed
    )
    eligible = [item for item in assessments if item.eligible]
    selected = (
        min(
            eligible,
            key=lambda item: (_VERSIONS.index(item.version), item.configuration_id),
        ).configuration_id
        if eligible
        else None
    )
    if selected is not None:
        trigger = "freeze_candidate"
    else:
        attempted_versions = {item.version for item in assessments}
        if "v29" in attempted_versions:
            trigger = "downgrade_claim"
        elif "v28" in attempted_versions:
            structure_failure = any(
                item.efficacy_pass
                and (
                    not item.gates["corr_err_degradation"].passed
                    or not item.gates["orthogonal_safety"].passed
                )
                for item in assessments
            )
            trigger = "v29" if structure_failure else "downgrade_claim"
        else:
            trigger = "v28"
    return SelectionReport(
        assessments=assessments,
        pareto_set=pareto_set,
        selected_configuration=selected,
        trigger=trigger,
        excluded_configurations=exclusion_values,
    )


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelectionAuthorityError("authority contains noncanonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SelectionAuthorityError(
            f"could not read authority file {path}"
        ) from error
    return digest.hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectionAuthorityError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SelectionAuthorityError(f"nonfinite JSON constant {value}")


def _read_authority_json(repository: Path, relative: str) -> dict[str, Any]:
    path = repository / relative
    try:
        if path.is_symlink() or not path.is_file():
            raise SelectionAuthorityError(
                f"authority path is not a regular file: {relative}"
            )
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelectionAuthorityError(
            f"could not parse authority file {relative}"
        ) from error
    if type(payload) is not dict:
        raise SelectionAuthorityError(f"authority file {relative} must be an object")
    return payload


def _exact_authority_mapping(
    value: object,
    fields: set[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise SelectionAuthorityError(f"{name} has missing or extra fields")
    return value


def _authority_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SelectionAuthorityError(f"{name} must be a lowercase SHA-256")
    return value


def _assert_tracked_clean(repository: Path, paths: Sequence[str]) -> None:
    try:
        root = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        if Path(root).resolve() != repository.resolve():
            raise SelectionAuthorityError("repository is not the canonical git root")
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-files",
                "--error-unmatch",
                "--",
                *paths,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--",
                *paths,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise SelectionAuthorityError(
            "selection authority must be tracked in the canonical git repository"
        ) from error
    if dirty:
        raise SelectionAuthorityError(
            "selection authority differs from the checked-out git commit"
        )


def _load_selection_authority(
    repository: Path,
    *,
    require_clean: bool = True,
) -> SelectionAuthority:
    """Load design, denominator, revisions, and margins from fixed tracked paths."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    repository = repository.resolve(strict=True)
    if require_clean:
        _assert_tracked_clean(repository, _AUTHORITY_PATHS)
    payloads = {
        relative: _read_authority_json(repository, relative)
        for relative in _AUTHORITY_PATHS
    }
    file_hashes = {
        relative: _file_sha256(repository / relative) for relative in _AUTHORITY_PATHS
    }

    protocol = _exact_authority_mapping(
        payloads["study/protocol.json"],
        {
            "schema_version",
            "legacy_data_role",
            "development",
            "final",
            "primary_metrics",
            "final_timeout_seconds",
            "max_rss_gib",
            "max_gpu_gib",
        },
        "study protocol",
    )
    if protocol["schema_version"] != 1 or type(protocol["schema_version"]) is not int:
        raise SelectionAuthorityError("study protocol schema_version must equal 1")
    development = _exact_authority_mapping(
        protocol["development"],
        {"namespace", "draws_per_condition", "cells", "genes"},
        "development protocol",
    )
    if development != {
        "namespace": "dev",
        "draws_per_condition": 2,
        "cells": 900,
        "genes": 500,
    }:
        raise SelectionAuthorityError("development protocol is not the closed design")

    panel = _exact_authority_mapping(
        payloads["study/development_panel.json"],
        {
            "schema_version",
            "role",
            "namespace",
            "mechanisms",
            "technical_views",
            "draws_per_mechanism",
            "cells",
            "genes",
            "seed_derivation",
        },
        "development panel",
    )
    mechanisms = tuple(panel["mechanisms"]) if type(panel["mechanisms"]) is list else ()
    technical_views = (
        tuple(panel["technical_views"])
        if type(panel["technical_views"]) is list
        else ()
    )
    if (
        panel["schema_version"] != 1
        or panel["role"] != "development_only"
        or panel["namespace"] != development["namespace"]
        or mechanisms != ("symsim", "sergio", "sparsim", "semisynthetic")
        or technical_views != ("moderate", "severe")
        or panel["draws_per_mechanism"] != development["draws_per_condition"]
        or panel["cells"] != development["cells"]
        or panel["genes"] != development["genes"]
    ):
        raise SelectionAuthorityError("development panel mismatches the protocol")
    seed_derivation = _exact_authority_mapping(
        panel["seed_derivation"],
        {"algorithm", "master_seed"},
        "development seed derivation",
    )
    if (
        seed_derivation["algorithm"] != "sha256-domain-separated-63bit-v1"
        or type(seed_derivation["master_seed"]) is not int
        or not 0 <= seed_derivation["master_seed"] < 2**63
    ):
        raise SelectionAuthorityError("development seed derivation is invalid")
    biological_ids = tuple(
        f"draw-{index:02d}" for index in range(1, panel["draws_per_mechanism"] + 1)
    )

    methods_root = _exact_authority_mapping(
        payloads["study/methods.json"],
        {"schema_version", "methods"},
        "method registry",
    )
    if methods_root["schema_version"] != 1 or type(methods_root["methods"]) is not list:
        raise SelectionAuthorityError("method registry schema is invalid")
    method_fields = {
        "id",
        "display_name",
        "role",
        "track",
        "input_scale",
        "output_scale",
        "stochastic",
        "seed_policy",
        "source",
        "license",
        "citation",
        "environment",
        "resources",
        "preserves_observed_positives",
        "source_policy",
        "integration_status",
        "integration_reason",
        "execution_scope",
        "applicability_reason",
    }
    method_rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(methods_root["methods"]):
        row = _exact_authority_mapping(raw, method_fields, f"method {index}")
        method_id = row["id"]
        if not isinstance(method_id, str) or not _SAFE_ID.fullmatch(method_id):
            raise SelectionAuthorityError(f"method {index} id is invalid")
        if method_id in method_rows:
            raise SelectionAuthorityError("method registry contains duplicate ids")
        if row["role"] not in {"control", "candidate", "competitor"}:
            raise SelectionAuthorityError(f"method {method_id} role is invalid")
        if row["track"] not in _TRACKS or type(row["stochastic"]) is not bool:
            raise SelectionAuthorityError(
                f"method {method_id} track or stochasticity is invalid"
            )
        execution_scope = row["execution_scope"]
        applicability_reason = row["applicability_reason"]
        if execution_scope not in {
            "same_input_required",
            "external_reference_only",
            "historical_not_run",
            "not_applicable",
        }:
            raise SelectionAuthorityError(
                f"method {method_id} execution scope is invalid"
            )
        if execution_scope == "same_input_required" and (
            row["track"] != "same_input" or applicability_reason is not None
        ):
            raise SelectionAuthorityError(
                f"method {method_id} same-input execution scope is inconsistent"
            )
        if execution_scope == "external_reference_only" and (
            row["track"] != "external_reference" or applicability_reason is not None
        ):
            raise SelectionAuthorityError(
                f"method {method_id} external-reference execution scope is inconsistent"
            )
        if execution_scope == "historical_not_run" and applicability_reason is not None:
            raise SelectionAuthorityError(
                f"method {method_id} historical execution scope is inconsistent"
            )
        if execution_scope == "not_applicable" and (
            not isinstance(applicability_reason, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", applicability_reason) is None
        ):
            raise SelectionAuthorityError(
                f"method {method_id} applicability reason is invalid"
            )
        expected_seed_policy = "required" if row["stochastic"] else "not_applicable"
        if row["seed_policy"] != expected_seed_policy:
            raise SelectionAuthorityError(
                f"method {method_id} seed policy contradicts stochasticity"
            )
        method_rows[method_id] = row

    contract = _exact_authority_mapping(
        payloads["study/selection_contract.json"],
        {
            "schema_version",
            "candidate_method_id",
            "model_seeds",
            "base_maskimpute_config",
            "base_maskimpute_config_sha256",
            "count_model_config",
            "count_model_config_sha256",
            "ablation_registry_path",
            "calibration_contract_path",
            "calibration_contract_sha256",
            "dataset_qc_policy",
            "dataset_qc_policy_sha256",
            "required_comparator_ids",
            "orthogonal_endpoints",
            "revision_policy",
            "equivalence_policy",
            "retained_calibration_artifact_path",
            "count_score_manifest_path",
        },
        "selection contract",
    )
    if contract["schema_version"] != 1:
        raise SelectionAuthorityError("selection contract schema_version must equal 1")
    candidate_method_id = contract["candidate_method_id"]
    candidate_method = method_rows.get(candidate_method_id)
    if (
        candidate_method is None
        or candidate_method["role"] != "candidate"
        or candidate_method["track"] != "same_input"
        or not candidate_method["stochastic"]
    ):
        raise SelectionAuthorityError("candidate method registry entry is invalid")
    model_seeds = (
        tuple(contract["model_seeds"]) if type(contract["model_seeds"]) is list else ()
    )
    if model_seeds != (42, 43, 44) or any(
        type(seed) is not int for seed in model_seeds
    ):
        raise SelectionAuthorityError("selection model seeds must equal 42, 43, and 44")
    expected_base_config = {
        "hidden_dims": [128, 64],
        "latent_dim": 24,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 300,
        "patience": 30,
        "artificial_mask_fraction": 0.2,
        "validation_fraction": 0.1,
        "log_count_bin_edges": [
            math.log1p(2.0),
            math.log1p(8.0),
            math.log1p(32.0),
        ],
        "early_stopping_min_delta": 0.0,
        "pre_zero_regularization": 1.0,
        "gate_gamma": 1.0,
        "normalization_target": 10_000.0,
    }
    base_config = _exact_authority_mapping(
        contract["base_maskimpute_config"],
        set(expected_base_config),
        "base MaskImpute configuration",
    )
    if base_config != expected_base_config:
        raise SelectionAuthorityError(
            "base MaskImpute configuration differs from publication defaults"
        )
    base_config_sha = _authority_sha(
        contract["base_maskimpute_config_sha256"],
        "base MaskImpute configuration checksum",
    )
    if _canonical_sha256(base_config) != base_config_sha:
        raise SelectionAuthorityError("base MaskImpute configuration checksum mismatch")
    expected_count_config = {
        "n_folds": 5,
        "use_library_size_exposure": True,
        "mean_prior_strength": 1.0,
        "mean_floor": 1e-8,
        "dispersion_prior_strength": 10.0,
        "link_bins": 64,
        "link_max_iter": 200,
        "link_tolerance": 1e-10,
        "link_bound": 30.0,
    }
    count_config = _exact_authority_mapping(
        contract["count_model_config"],
        set(expected_count_config),
        "count-model configuration",
    )
    if count_config != expected_count_config:
        raise SelectionAuthorityError(
            "count-model configuration differs from publication defaults"
        )
    count_config_sha = _authority_sha(
        contract["count_model_config_sha256"],
        "count-model configuration checksum",
    )
    if _canonical_sha256(count_config) != count_config_sha:
        raise SelectionAuthorityError("count-model configuration checksum mismatch")
    if contract["ablation_registry_path"] != "study/ablations.json":
        raise SelectionAuthorityError("ablation registry path is not canonical")
    expected_qc_policy = {
        "cell_exclusion_rule": "observed_library_size_equals_zero",
        "minimum_retained_cells": 2,
        "application": (
            "pre_dispatch_pair_union_zero_library_identical_cell_subset_all_methods"
        ),
        "additional_cell_filtering": "forbidden",
        "gene_filtering": "forbidden",
        "required_audit_fields": [
            "excluded_cell_count",
            "excluded_cell_ids_sha256",
            "retained_cell_count",
            "retained_cell_ids_sha256",
        ],
    }
    qc_policy = _exact_authority_mapping(
        contract["dataset_qc_policy"],
        set(expected_qc_policy),
        "dataset QC policy",
    )
    if qc_policy != expected_qc_policy:
        raise SelectionAuthorityError("dataset QC policy is invalid")
    qc_policy_sha = _authority_sha(
        contract["dataset_qc_policy_sha256"], "dataset QC policy checksum"
    )
    if _canonical_sha256(qc_policy) != qc_policy_sha:
        raise SelectionAuthorityError("dataset QC policy checksum mismatch")
    if contract["calibration_contract_path"] != "study/calibration_contract.json":
        raise SelectionAuthorityError("calibration contract path is not canonical")
    calibration_contract_sha = _authority_sha(
        contract["calibration_contract_sha256"],
        "calibration contract checksum",
    )
    if calibration_contract_sha != file_hashes["study/calibration_contract.json"]:
        raise SelectionAuthorityError("calibration contract checksum mismatch")
    expected_calibration_contract = {
        "schema_version": 1,
        "contract_id": "prezero-calibration-retention-binding-v1",
        "artifact_schema_version": 3,
        "status": "adopted",
        "timing": {
            "data_scope": "development_only",
            "adopted_before": "final_seed_execution",
            "final_data_used": False,
        },
        "truth_scope": {
            "eligible_exact_mechanisms": ["symsim"],
            "reason": "only_symsim_has_exact_discrete_pre_capture_zero_truth",
            "proxy_truth_relabelled": False,
            "panel_limitations": {
                "semisynthetic": "proxy_truth_not_exact",
                "sergio": "undefined_for_continuous_truth",
                "sparsim": "undefined_for_continuous_truth",
            },
        },
        "cross_validation": {
            "scheme": "leave_one_mechanism_biological_draw_out",
            "independent_unit": "biological_draw",
            "nested_technical_unit": "draw_technical_view_record",
            "development_inference": "held_out_fold_calibrator_only",
            "final_inference": "all_development_fitted_calibrator",
        },
        "retention_rules": {
            "minimum_exact_mechanisms_improved": 3,
            "minimum_biological_draws_improved": 2,
            "minimum_technical_records_improved": 4,
            "brier_improvement_epsilon": 1e-6,
            "log_loss_worsening_tolerance": 1e-3,
            "calibration_slope_lower": 0.8,
            "calibration_slope_upper": 1.2,
            "require_all_eligible_exact_mechanisms_improved": True,
            "require_all_biological_draws_improved": True,
            "require_all_technical_records_improved": True,
            "require_no_fit_failures": True,
            "log_loss_gated_levels": [
                "aggregate",
                "mechanism",
                "biological_draw",
                "technical_record",
            ],
            "calibration_slope_gated_levels": [
                "aggregate",
                "mechanism",
                "biological_draw",
            ],
            "technical_record_slope_policy": (
                "reported_not_gated_nested_technical_observation"
            ),
        },
    }
    if payloads["study/calibration_contract.json"] != expected_calibration_contract:
        raise SelectionAuthorityError("calibration retention contract is invalid")
    required_ids = (
        tuple(contract["required_comparator_ids"])
        if type(contract["required_comparator_ids"]) is list
        else ()
    )
    if not required_ids or len(required_ids) != len(set(required_ids)):
        raise SelectionAuthorityError("required comparator ids are invalid")
    for method_id in required_ids:
        row = method_rows.get(method_id)
        if row is None or row["track"] != "same_input" or row["role"] == "candidate":
            raise SelectionAuthorityError(
                f"required comparator {method_id!r} is not eligible"
            )
    if "observed" not in required_ids or "capacity-matched-ae" not in required_ids:
        raise SelectionAuthorityError(
            "observed and capacity-matched controls are required"
        )

    endpoint_rows = contract["orthogonal_endpoints"]
    if type(endpoint_rows) is not list or not endpoint_rows:
        raise SelectionAuthorityError("orthogonal endpoint contract is empty")
    endpoint_policies: list[EndpointPolicy] = []
    endpoint_ids: set[str] = set()
    for index, raw in enumerate(endpoint_rows):
        row = _exact_authority_mapping(
            raw,
            {
                "id",
                "source_id",
                "comparison",
                "favorable_direction",
                "materiality_margin",
            },
            f"orthogonal endpoint {index}",
        )
        if row["id"] in endpoint_ids:
            raise SelectionAuthorityError("orthogonal endpoint ids are duplicated")
        if not isinstance(row["source_id"], str) or not _SAFE_ID.fullmatch(
            row["source_id"]
        ):
            raise SelectionAuthorityError("orthogonal endpoint source id is invalid")
        try:
            policy = EndpointPolicy(
                id=row["id"],
                comparison=row["comparison"],
                favorable_direction=row["favorable_direction"],
                materiality_margin=row["materiality_margin"],
            )
        except (TypeError, ValueError) as error:
            raise SelectionAuthorityError(
                "orthogonal endpoint policy is invalid"
            ) from error
        endpoint_ids.add(policy.id)
        endpoint_policies.append(policy)

    revision = _exact_authority_mapping(
        contract["revision_policy"],
        {"v28_retention", "v29_retention", "v29_max_dropout_mse_loss"},
        "revision policy",
    )
    if (
        revision["v28_retention"]
        != "strict_pareto_improvement_with_zero_and_null_de_safety"
        or revision["v29_retention"]
        != "structure_or_downstream_improvement_without_material_dropout_mse_loss"
    ):
        raise SelectionAuthorityError("revision retention rules are invalid")
    try:
        revision_policy = RevisionPolicy(revision["v29_max_dropout_mse_loss"])
    except (TypeError, ValueError) as error:
        raise SelectionAuthorityError(
            "revision materiality margin is invalid"
        ) from error
    equivalence = _exact_authority_mapping(
        contract["equivalence_policy"],
        {
            "identity_calibration_reason_code",
            "nonidentity_calibration_reason_code",
            "pending_effect_status",
            "identity_effect_status",
            "nonidentity_effect_status",
            "calibrated_spec_id",
            "direct_spec_id",
        },
        "equivalence policy",
    )
    identity_reason = equivalence["identity_calibration_reason_code"]
    if (
        identity_reason != "retained_identity_calibrator_equals_direct_score"
        or equivalence["nonidentity_calibration_reason_code"]
        != "retained_nonidentity_calibrator_transformed_score"
        or equivalence["pending_effect_status"] != "pending_retained_artifact"
        or equivalence["identity_effect_status"] != "not_applicable_for_efficacy"
        or equivalence["nonidentity_effect_status"]
        != "retained_calibration_effect_estimable"
        or equivalence["calibrated_spec_id"] != "maskimpute-reference"
        or equivalence["direct_spec_id"] != "direct-score"
    ):
        raise SelectionAuthorityError(
            "dynamic calibration interpretation policy is invalid"
        )
    calibration_path = contract["retained_calibration_artifact_path"]
    if (
        calibration_path
        != "artifacts/study/development/calibration/retained_calibration.json"
        or PurePosixPath(calibration_path).is_absolute()
        or ".." in PurePosixPath(calibration_path).parts
    ):
        raise SelectionAuthorityError("retained calibration artifact path is invalid")
    count_score_path = contract["count_score_manifest_path"]
    if (
        count_score_path != "artifacts/study/development/count_scores/manifest.json"
        or PurePosixPath(count_score_path).is_absolute()
        or ".." in PurePosixPath(count_score_path).parts
    ):
        raise SelectionAuthorityError("count-score manifest path is invalid")

    ablations = _exact_authority_mapping(
        payloads["study/ablations.json"],
        {
            "schema_version",
            "model_seeds",
            "parameter_budget",
            "optimizer_budget",
            "preprocessing_budget",
            "reference",
            "variants",
        },
        "ablation registry",
    )
    if (
        ablations["schema_version"] != 1
        or ablations["model_seeds"] != list(model_seeds)
        or ablations["parameter_budget"] != "exact_nominal_match"
        or ablations["optimizer_budget"] != "shared_frozen_candidate_budget"
        or ablations["preprocessing_budget"] != "shared_except_named_component"
        or type(ablations["variants"]) is not list
    ):
        raise SelectionAuthorityError("ablation registry policy is invalid")
    ablation_fields = {
        "id",
        "changed_component",
        "positive_masking",
        "pre_zero_regularizer",
        "encoder_mode",
        "gate",
        "output_policy",
        "score_source",
    }
    reference_spec = _exact_authority_mapping(
        ablations["reference"], ablation_fields, "ablation reference"
    )
    raw_ablation_specs = [reference_spec]
    raw_ablation_specs.extend(
        _exact_authority_mapping(
            value,
            ablation_fields,
            f"ablation variant {index}",
        )
        for index, value in enumerate(ablations["variants"])
    )
    ablation_ids = tuple(value["id"] for value in raw_ablation_specs)
    expected_ablation_ids = (
        "maskimpute-reference",
        "capacity-matched-ae",
        "no-gate",
        "no-pre-zero-regularizer",
        "no-explicit-mask",
        "full-denoising",
        "direct-score",
    )
    if ablation_ids != expected_ablation_ids:
        raise SelectionAuthorityError("ablation panel is incomplete or reordered")
    for spec in raw_ablation_specs:
        if type(spec["pre_zero_regularizer"]) is not bool:
            raise SelectionAuthorityError("ablation boolean policy is invalid")
        expected_score_source = (
            "direct" if spec["id"] == "direct-score" else "retained_calibrator"
        )
        if spec["score_source"] != expected_score_source:
            raise SelectionAuthorityError(
                "ablation score source differs from the tracked retained/direct contrast"
            )
    if reference_spec["changed_component"] != "reference":
        raise SelectionAuthorityError("ablation reference semantics are invalid")
    ablation_specs = tuple(_freeze_details(spec) for spec in raw_ablation_specs)
    ablation_run_keys = tuple(
        (spec_id, seed) for spec_id in ablation_ids for seed in model_seeds
    )

    ledger = _exact_authority_mapping(
        payloads["study/development_search.json"],
        {
            "schema_version",
            "authority",
            "retained_calibration_artifact",
            "count_score_manifest",
            "configurations",
            "exclusions",
        },
        "development search ledger",
    )
    if ledger["schema_version"] != 1:
        raise SelectionAuthorityError("development search schema_version must equal 1")
    ledger_authority = _exact_authority_mapping(
        ledger["authority"],
        {
            "protocol_sha256",
            "development_panel_sha256",
            "methods_sha256",
            "selection_contract_sha256",
            "ablations_sha256",
            "calibration_contract_sha256",
        },
        "development search authority",
    )
    expected_ledger_hashes = {
        "protocol_sha256": file_hashes["study/protocol.json"],
        "development_panel_sha256": file_hashes["study/development_panel.json"],
        "methods_sha256": file_hashes["study/methods.json"],
        "selection_contract_sha256": file_hashes["study/selection_contract.json"],
        "ablations_sha256": file_hashes["study/ablations.json"],
        "calibration_contract_sha256": file_hashes["study/calibration_contract.json"],
    }
    if ledger_authority != expected_ledger_hashes:
        raise SelectionAuthorityError(
            "development search ledger authority hashes do not match tracked files"
        )

    calibration = _exact_authority_mapping(
        ledger["retained_calibration_artifact"],
        {"status", "path", "sha256"},
        "retained calibration binding",
    )
    if calibration["path"] != calibration_path:
        raise SelectionAuthorityError("retained calibration path mismatches contract")
    if calibration["status"] == "pending":
        if calibration["sha256"] is not None:
            raise SelectionAuthorityError(
                "pending retained calibration must not claim a checksum"
            )
    elif calibration["status"] == "ready":
        _authority_sha(calibration["sha256"], "retained calibration checksum")
    else:
        raise SelectionAuthorityError("retained calibration status is invalid")
    retained_calibration = RetainedCalibrationBinding(
        status=calibration["status"],
        path=calibration_path,
        sha256=calibration["sha256"],
    )
    count_score = _exact_authority_mapping(
        ledger["count_score_manifest"],
        {"status", "path", "sha256"},
        "count-score manifest binding",
    )
    if count_score["path"] != count_score_path:
        raise SelectionAuthorityError("count-score manifest path mismatches contract")
    if count_score["status"] == "pending":
        if count_score["sha256"] is not None:
            raise SelectionAuthorityError(
                "pending count-score manifest must not claim a checksum"
            )
    elif count_score["status"] == "ready":
        _authority_sha(count_score["sha256"], "count-score manifest checksum")
    else:
        raise SelectionAuthorityError("count-score manifest status is invalid")
    count_score_manifest = RetainedCalibrationBinding(
        status=count_score["status"],
        path=count_score_path,
        sha256=count_score["sha256"],
    )

    configuration_rows = ledger["configurations"]
    schedule = (
        ("v27-c01-direct-r1-g1", "direct_cross_fitted_count_score", 1.0, 1.0, True),
        (
            "v27-c02-calibrated-r1-g0p5",
            "retained_development_calibrator",
            1.0,
            0.5,
            True,
        ),
        ("v27-c03-calibrated-r1-g1", "retained_development_calibrator", 1.0, 1.0, True),
        (
            "v27-c04-calibrated-r1-g1p5",
            "retained_development_calibrator",
            1.0,
            1.5,
            True,
        ),
        ("v27-c05-calibrated-r1-g2", "retained_development_calibrator", 1.0, 2.0, True),
        ("v27-c06-calibrated-r1-g3", "retained_development_calibrator", 1.0, 3.0, True),
        ("v27-c07-calibrated-r1-g4", "retained_development_calibrator", 1.0, 4.0, True),
        ("v27-c08-calibrated-r1-g6", "retained_development_calibrator", 1.0, 6.0, True),
        ("v27-c09-calibrated-r1-g8", "retained_development_calibrator", 1.0, 8.0, True),
        (
            "v27-c10-calibrated-r1-g12",
            "retained_development_calibrator",
            1.0,
            12.0,
            True,
        ),
        ("v27-c11-calibrated-r2-g2", "retained_development_calibrator", 2.0, 2.0, True),
        ("v27-c12-calibrated-r2-g3", "retained_development_calibrator", 2.0, 3.0, True),
        ("v27-c13-calibrated-r2-g4", "retained_development_calibrator", 2.0, 4.0, True),
        ("v27-c14-calibrated-r2-g6", "retained_development_calibrator", 2.0, 6.0, True),
        ("v27-c15-calibrated-r5-g2", "retained_development_calibrator", 5.0, 2.0, True),
        ("v27-c16-calibrated-r5-g3", "retained_development_calibrator", 5.0, 3.0, True),
        ("v27-c17-calibrated-r5-g4", "retained_development_calibrator", 5.0, 4.0, True),
        ("v27-c18-calibrated-r5-g6", "retained_development_calibrator", 5.0, 6.0, True),
        (
            "v27-c19-calibrated-r10-g2",
            "retained_development_calibrator",
            10.0,
            2.0,
            True,
        ),
        (
            "v27-c20-calibrated-r10-g3",
            "retained_development_calibrator",
            10.0,
            3.0,
            True,
        ),
        (
            "v27-c21-calibrated-r10-g4",
            "retained_development_calibrator",
            10.0,
            4.0,
            False,
        ),
        (
            "v27-c22-calibrated-r10-g6",
            "retained_development_calibrator",
            10.0,
            6.0,
            False,
        ),
    )
    if type(configuration_rows) is not list or len(configuration_rows) != len(schedule):
        raise SelectionAuthorityError(
            "development search must contain the exact 22-row chronology"
        )
    attempts: list[CandidateAttempt] = []
    method_bindings = {
        method_id: _canonical_sha256(row)
        for method_id, row in method_rows.items()
        if row["track"] == "same_input" and row["role"] != "candidate"
    }
    method_bindings.update(
        {spec["id"]: _canonical_sha256(spec) for spec in raw_ablation_specs}
    )
    overrun_exclusions: list[SearchExclusion] = []
    for index, (raw, expected) in enumerate(
        zip(configuration_rows, schedule, strict=True), start=1
    ):
        row = _exact_authority_mapping(
            raw,
            {
                "configuration_id",
                "version",
                "disposition",
                "parent_configuration_id",
                "configuration",
                "configuration_sha256",
                "reason_code",
            },
            f"search configuration {index}",
        )
        expected_id, score_policy, regularization, gamma, selectable = expected
        expected_disposition = (
            "authorized"
            if selectable
            else "exploratory_budget_overrun_not_selection_eligible"
        )
        expected_reason = (
            "chronological_development_selection_eligible"
            if selectable
            else "exploratory_budget_overrun_not_selection_eligible"
        )
        if (
            row["configuration_id"] != expected_id
            or row["disposition"] != expected_disposition
            or row["reason_code"] != expected_reason
            or row["version"] != "v27"
            or row["parent_configuration_id"] is not None
        ):
            raise SelectionAuthorityError(
                "search chronology or budget disposition is invalid"
            )
        observed_configuration_hash = _canonical_sha256(row["configuration"])
        if row["configuration_sha256"] != observed_configuration_hash:
            raise SelectionAuthorityError("search configuration checksum mismatch")
        configuration = _exact_authority_mapping(
            row["configuration"],
            {
                "method_version",
                "decoder",
                "encoder_mode",
                "output_policy",
                "score_policy",
                "hyperparameters",
            },
            f"search configuration {index} payload",
        )
        expected_hyperparameters = dict(base_config)
        expected_hyperparameters["pre_zero_regularization"] = regularization
        expected_hyperparameters["gate_gamma"] = gamma
        if configuration != {
            "method_version": "v27",
            "decoder": "scaled_gaussian",
            "encoder_mode": "explicit_mask",
            "output_policy": "selective",
            "score_policy": score_policy,
            "hyperparameters": expected_hyperparameters,
        }:
            raise SelectionAuthorityError(
                "search configuration differs from its exact chronological authority"
            )
        try:
            attempt = CandidateAttempt(
                configuration_id=row["configuration_id"],
                version=row["version"],
                parent_configuration_id=row["parent_configuration_id"],
            )
        except (TypeError, ValueError) as error:
            raise SelectionAuthorityError(
                "search configuration identity is invalid"
            ) from error
        if attempt.configuration_id in method_bindings:
            raise SelectionAuthorityError(
                "search configuration collides with method id"
            )
        method_bindings[attempt.configuration_id] = observed_configuration_hash
        if selectable:
            attempts.append(attempt)
        else:
            overrun_exclusions.append(
                SearchExclusion(
                    configuration_id=attempt.configuration_id,
                    version="v27",
                    equivalent_to=None,
                    reason_code="exploratory_budget_overrun_not_selection_eligible",
                )
            )
    try:
        attempt_values = _validate_attempts(tuple(attempts))
    except (TypeError, ValueError) as error:
        raise SelectionAuthorityError("search revision ordering is invalid") from error

    exclusion_rows = ledger["exclusions"]
    if type(exclusion_rows) is not list:
        raise SelectionAuthorityError("search exclusions must be an array")
    exclusions: list[SearchExclusion] = list(overrun_exclusions)
    for index, raw in enumerate(exclusion_rows):
        row = _exact_authority_mapping(
            raw,
            {"configuration_id", "version", "equivalent_to", "reason_code"},
            f"search exclusion {index}",
        )
        if row["reason_code"] != identity_reason:
            raise SelectionAuthorityError("search exclusion reason is not authorized")
        try:
            exclusions.append(SearchExclusion(**row))
        except (TypeError, ValueError) as error:
            raise SelectionAuthorityError("search exclusion is invalid") from error

    declarations: list[MethodDeclaration] = []
    for method_id, row in method_rows.items():
        if row["track"] != "same_input" or row["role"] == "candidate":
            continue
        if method_id == "observed":
            role = "observed_control"
        elif row["role"] == "control":
            role = "learned_control"
        else:
            role = "learned_comparator"
        declarations.append(
            MethodDeclaration(
                id=method_id,
                role=role,
                track="same_input",
                stochastic=row["stochastic"],
                required_for_claim=method_id in required_ids,
            )
        )
    declarations.extend(
        MethodDeclaration(
            id=attempt.configuration_id,
            role="candidate",
            track="same_input",
            stochastic=True,
            required_for_claim=True,
        )
        for attempt in attempt_values
    )
    try:
        declaration_values = _validate_declarations(tuple(declarations), attempt_values)
    except (TypeError, ValueError) as error:
        raise SelectionAuthorityError(
            "derived method declarations are invalid"
        ) from error

    return SelectionAuthority(
        mechanisms=mechanisms,
        biological_ids=biological_ids,
        technical_views=technical_views,
        model_seeds=model_seeds,
        required_comparator_ids=required_ids,
        attempts=attempt_values,
        declarations=declaration_values,
        endpoint_policies=tuple(endpoint_policies),
        revision_policy=revision_policy,
        exclusions=tuple(exclusions),
        method_bindings=MappingProxyType(method_bindings),
        base_maskimpute_config=_freeze_details(base_config),
        base_maskimpute_config_sha256=base_config_sha,
        count_model_config=_freeze_details(count_config),
        count_model_config_sha256=count_config_sha,
        dataset_qc_policy=_freeze_details(qc_policy),
        dataset_qc_policy_sha256=qc_policy_sha,
        ablation_specs=ablation_specs,
        ablation_spec_ids=ablation_ids,
        ablation_run_keys=ablation_run_keys,
        calibration_equivalence_reason=None,
        calibration_effect_status=equivalence["pending_effect_status"],
        retained_calibration=retained_calibration,
        count_score_manifest=count_score_manifest,
        file_sha256=MappingProxyType(file_hashes),
    )


_DOWNSTREAM_SELECTION_BINDING_FIELDS = frozenset(
    {
        "path",
        "manifest_file_sha256",
        "manifest_sha256",
        "plan_sha256",
        "planned_denominator_count",
        "endpoint_row_count",
        "source_checkpoint_path",
        "source_checkpoint_file_sha256",
        "source_checkpoint_payload_sha256",
        "source_plan_sha256",
        "source_input_hashes_sha256",
        "source_statuses_sha256",
        "source_plan_authority",
        "revision_versions",
        "sources",
    }
)


def _selection_downstream_denominators(
    records: object,
) -> tuple[tuple[object, ...], ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SelectionAuthorityError("selection records are invalid")
    result: set[tuple[object, ...]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SelectionAuthorityError(f"selection record {index} is invalid")
        key = (
            record.get("mechanism"),
            record.get("biological_id"),
            record.get("technical_view"),
            record.get("dataset_id"),
            record.get("dataset_sha256"),
            record.get("method"),
            record.get("method_sha256"),
            record.get("model_seed"),
        )
        if (
            any(not isinstance(value, str) or not value for value in key[:7])
            or not _SHA256.fullmatch(str(key[4]))
            or not _SHA256.fullmatch(str(key[6]))
            or (
                key[7] is not None
                and (isinstance(key[7], bool) or type(key[7]) is not int)
            )
        ):
            raise SelectionAuthorityError(
                f"selection record {index} denominator identity is invalid"
            )
        result.add(key)
    if not result:
        raise SelectionAuthorityError("selection denominator set is empty")
    return tuple(sorted(result, key=lambda value: tuple(str(item) for item in value)))


def _selection_downstream_statuses(
    records: object,
) -> Mapping[tuple[object, ...], frozenset[str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise SelectionAuthorityError("selection records are invalid")
    result: dict[tuple[object, ...], set[str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SelectionAuthorityError(f"selection record {index} is invalid")
        key = (
            record.get("mechanism"),
            record.get("biological_id"),
            record.get("technical_view"),
            record.get("dataset_id"),
            record.get("dataset_sha256"),
            record.get("method"),
            record.get("method_sha256"),
            record.get("model_seed"),
        )
        status = record.get("status")
        if not isinstance(status, str) or not status:
            raise SelectionAuthorityError(f"selection record {index} status is invalid")
        result.setdefault(key, set()).add(status)
    return MappingProxyType(
        {key: frozenset(statuses) for key, statuses in result.items()}
    )


def _downstream_selection_status(value: object) -> str:
    if value in {"infrastructure_error", "blocked_authority", "budget_exhausted"}:
        return "failed"
    if value in {"completed", "failed", "timeout", "unavailable", "resource_exceeded"}:
        return str(value)
    raise SelectionAuthorityError("downstream source status is invalid")


def _downstream_output_directory(repository: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise SelectionAuthorityError("downstream evidence path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SelectionAuthorityError("downstream evidence path is unsafe")
    path = repository.joinpath(*relative.parts)
    for component in (path, *path.parents):
        if component == repository.parent:
            break
        if component.is_symlink():
            raise SelectionAuthorityError("downstream evidence path contains a symlink")
    if not path.is_dir():
        raise SelectionAuthorityError("downstream evidence directory is absent")
    return path


def validate_downstream_selection_completeness(
    repository: Path,
    records: object,
    binding: object,
) -> Mapping[str, str]:
    """Require bound eight-row downstream evidence for every selection denominator."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    row = _exact_authority_mapping(
        binding,
        _DOWNSTREAM_SELECTION_BINDING_FIELDS,
        "downstream evidence binding",
    )
    directory = _downstream_output_directory(repository, row["path"])
    manifest_path = directory / "downstream_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SelectionAuthorityError("downstream evidence manifest is absent")
    manifest_file_sha = _authority_sha(
        row["manifest_file_sha256"], "downstream manifest file checksum"
    )
    if _file_sha256(manifest_path) != manifest_file_sha:
        raise SelectionAuthorityError("downstream manifest file checksum mismatch")
    try:
        from .downstream_evidence import (
            DownstreamEvidenceError,
            downstream_source_statuses,
            validate_downstream_evidence_completeness,
        )

        evidence = validate_downstream_evidence_completeness(
            directory,
            expected_denominators=_selection_downstream_denominators(records),
        )
        source_statuses = downstream_source_statuses(directory)
    except DownstreamEvidenceError as error:
        raise SelectionAuthorityError(
            f"downstream evidence completeness failed: {error}"
        ) from error
    if evidence.payload.get("source_kind") != "development":
        raise SelectionAuthorityError("selection downstream source is not development")
    downstream_statuses = {
        (
            record.get("mechanism"),
            record.get("biological_id"),
            record.get("technical_view"),
            record.get("dataset_id"),
            record.get("dataset_sha256"),
            record.get("method"),
            record.get("method_artifact_sha256"),
            record.get("model_seed"),
        ): _downstream_selection_status(record.get("run_status"))
        for record in evidence.records
    }
    normalized_source_statuses = {
        key: _downstream_selection_status(status)
        for key, status in source_statuses.items()
    }
    if normalized_source_statuses != downstream_statuses:
        raise SelectionAuthorityError(
            "downstream run status differs from independently validated source"
        )
    selection_statuses = _selection_downstream_statuses(records)
    if set(selection_statuses) != set(normalized_source_statuses):
        raise SelectionAuthorityError("downstream source status differs from selection")
    for key, statuses in selection_statuses.items():
        source_status = normalized_source_statuses[key]
        allowed = (
            frozenset({"completed", "unavailable"})
            if source_status == "completed"
            else frozenset({source_status})
        )
        if not statuses or not statuses <= allowed:
            raise SelectionAuthorityError(
                "downstream source status differs from selection"
            )
    expected_count = row["planned_denominator_count"]
    expected_rows = row["endpoint_row_count"]
    revision_versions = evidence.payload.get("development_revision_versions")
    sources = evidence.payload.get("development_sources")
    if (
        row["manifest_sha256"] != evidence.manifest_sha256
        or row["plan_sha256"] != evidence.plan_sha256
        or isinstance(expected_count, bool)
        or type(expected_count) is not int
        or expected_count != evidence.planned_denominator_count
        or isinstance(expected_rows, bool)
        or type(expected_rows) is not int
        or expected_rows != evidence.endpoint_row_count
        or row["source_checkpoint_path"] != evidence.payload.get("source_manifest_path")
        or row["source_checkpoint_file_sha256"]
        != evidence.payload.get("source_manifest_file_sha256")
        or row["source_checkpoint_payload_sha256"]
        != evidence.payload.get("source_manifest_payload_sha256")
        or row["source_plan_sha256"] != evidence.payload.get("source_plan_sha256")
        or row["source_input_hashes_sha256"]
        != evidence.payload.get("source_input_hashes_sha256")
        or row["source_statuses_sha256"]
        != evidence.payload.get("source_statuses_sha256")
        or row["source_plan_authority"] != "independent"
        or evidence.payload.get("source_plan_authority") != "independent"
        or not isinstance(revision_versions, list)
        or not isinstance(sources, list)
        or row["revision_versions"] != revision_versions
        or row["sources"] != sources
    ):
        raise SelectionAuthorityError("downstream evidence binding differs")
    result = {
        "downstream_manifest_file_sha256": manifest_file_sha,
        "downstream_manifest_sha256": evidence.manifest_sha256,
        "downstream_plan_sha256": evidence.plan_sha256,
        "downstream_source_checkpoint_path": row["source_checkpoint_path"],
        "downstream_source_checkpoint_file_sha256": row[
            "source_checkpoint_file_sha256"
        ],
        "downstream_source_checkpoint_payload_sha256": row[
            "source_checkpoint_payload_sha256"
        ],
        "downstream_source_plan_sha256": row["source_plan_sha256"],
        "downstream_source_input_hashes_sha256": row["source_input_hashes_sha256"],
        "downstream_source_statuses_sha256": row["source_statuses_sha256"],
    }
    for source in sources:
        if not isinstance(source, Mapping):
            raise SelectionAuthorityError(
                "downstream development source binding is invalid"
            )
        source_id = source.get("source_id")
        if source_id not in {"base", "v28", "v29"}:
            raise SelectionAuthorityError(
                "downstream development source identity is invalid"
            )
        prefix = f"downstream_{source_id}"
        for output_name, source_name in (
            ("checkpoint_path", "manifest_path"),
            ("checkpoint_file_sha256", "manifest_file_sha256"),
            ("checkpoint_payload_sha256", "manifest_payload_sha256"),
            ("plan_sha256", "plan_sha256"),
            ("input_hashes_sha256", "input_hashes_sha256"),
            ("statuses_sha256", "statuses_sha256"),
            ("denominator_sha256", "denominator_sha256"),
            ("evaluation_manifest_path", "evaluation_manifest_path"),
            (
                "evaluation_manifest_file_sha256",
                "evaluation_manifest_file_sha256",
            ),
            (
                "evaluation_manifest_payload_sha256",
                "evaluation_manifest_payload_sha256",
            ),
            ("evaluation_source_sha256", "evaluation_source_sha256"),
        ):
            value = source.get(source_name)
            if not isinstance(value, str) or not value:
                raise SelectionAuthorityError(
                    "downstream development source binding is invalid"
                )
            result[f"{prefix}_{output_name}"] = value
    return MappingProxyType(result)


def attach_downstream_evidence_to_selection_result(
    payload: object,
    repository: Path,
    relative_directory: str,
) -> dict[str, object]:
    """Upgrade a schema-2/3 result to selection-complete schema 4."""

    if type(payload) is not dict or payload.get("schema_version") not in {2, 3}:
        raise SelectionAuthorityError("downstream attachment requires schema 2 or 3")
    original_sha = _authority_sha(
        payload.get("result_sha256"), "development result checksum"
    )
    original_core = {
        key: value for key, value in payload.items() if key != "result_sha256"
    }
    if _canonical_sha256(original_core) != original_sha:
        raise SelectionAuthorityError("development result checksum mismatch")
    directory = _downstream_output_directory(repository, relative_directory)
    try:
        from .downstream_evidence import load_downstream_evidence_manifest

        evidence = load_downstream_evidence_manifest(directory)
    except Exception as error:
        raise SelectionAuthorityError(
            "downstream evidence cannot be attached"
        ) from error
    manifest_path = directory / "downstream_manifest.json"
    binding: dict[str, object] = {
        "path": relative_directory,
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": evidence.manifest_sha256,
        "plan_sha256": evidence.plan_sha256,
        "planned_denominator_count": evidence.planned_denominator_count,
        "endpoint_row_count": evidence.endpoint_row_count,
        "source_checkpoint_path": evidence.payload["source_manifest_path"],
        "source_checkpoint_file_sha256": evidence.payload[
            "source_manifest_file_sha256"
        ],
        "source_checkpoint_payload_sha256": evidence.payload[
            "source_manifest_payload_sha256"
        ],
        "source_plan_sha256": evidence.payload["source_plan_sha256"],
        "source_input_hashes_sha256": evidence.payload["source_input_hashes_sha256"],
        "source_statuses_sha256": evidence.payload["source_statuses_sha256"],
        "source_plan_authority": evidence.payload["source_plan_authority"],
        "revision_versions": list(evidence.payload["development_revision_versions"]),
        "sources": list(evidence.payload["development_sources"]),
    }
    revision_versions = (
        [] if payload["schema_version"] == 2 else list(payload["revision_versions"])
    )
    if revision_versions != binding["revision_versions"]:
        raise SelectionAuthorityError(
            "downstream revision sources differ from selection input"
        )
    upgraded_core = {
        **original_core,
        "schema_version": 4,
        "revision_versions": revision_versions,
        "downstream_evidence": binding,
    }
    upgraded = {
        **upgraded_core,
        "result_sha256": _canonical_sha256(upgraded_core),
    }
    validate_downstream_selection_completeness(repository, upgraded["records"], binding)
    return upgraded


def _validate_revision_downstream_source_bindings(
    downstream: Mapping[str, str],
    evaluation: Mapping[str, str],
    revision_versions: Sequence[str],
) -> None:
    """Cross-bind every schema-4 source to independently rebuilt evaluation evidence."""

    versions = tuple(revision_versions)
    if versions not in {("v28",), ("v28", "v29")}:
        raise SelectionAuthorityError(
            "revision downstream source versions are incomplete or reordered"
        )
    names = (
        ("checkpoint_path", "reconstruction_checkpoint_path"),
        ("checkpoint_file_sha256", "reconstruction_checkpoint_file_sha256"),
        (
            "checkpoint_payload_sha256",
            "reconstruction_checkpoint_payload_sha256",
        ),
        ("plan_sha256", "reconstruction_plan_sha256"),
        ("input_hashes_sha256", "reconstruction_input_hashes_sha256"),
        ("statuses_sha256", "reconstruction_statuses_sha256"),
        ("evaluation_manifest_path", "evaluation_manifest_path"),
        (
            "evaluation_manifest_file_sha256",
            "evaluation_manifest_file_sha256",
        ),
        (
            "evaluation_manifest_payload_sha256",
            "evaluation_manifest_payload_sha256",
        ),
        ("evaluation_source_sha256", "evaluation_source_sha256"),
    )
    for source_id in ("base", *versions):
        if any(
            downstream.get(f"downstream_{source_id}_{downstream_name}")
            != evaluation.get(f"{source_id}_{evaluation_name}")
            for downstream_name, evaluation_name in names
        ):
            raise SelectionAuthorityError(
                f"{source_id} downstream source differs from revision evaluation authority"
            )


def _select_for_repository(
    payload: object,
    repository: Path,
    *,
    require_clean: bool = True,
) -> SelectionReport:
    if type(payload) is not dict:
        raise SelectionAuthorityError(
            "development result payload has missing or extra fields"
        )
    schema_version = payload.get("schema_version")
    base_fields = {
        "schema_version",
        "dataset_manifest_sha256",
        "count_score_manifest_sha256",
        "retained_calibration_artifact_sha256",
        "evaluation_manifest_sha256",
        "records",
        "orthogonal_intervals",
        "result_sha256",
    }
    expected_fields = (
        base_fields
        if schema_version == 2
        else {*base_fields, "revision_versions"}
        if schema_version == 3
        else {*base_fields, "revision_versions", "downstream_evidence"}
        if schema_version == 4
        else base_fields
    )
    data = _exact_authority_mapping(
        payload,
        expected_fields,
        "development result payload",
    )
    if schema_version not in {2, 3, 4} or type(schema_version) is not int:
        raise SelectionAuthorityError(
            "development result schema_version must equal 2, 3, or 4"
        )
    if schema_version in {3, 4}:
        revision_versions = data["revision_versions"]
        allowed_versions = (
            ([], ["v28"], ["v28", "v29"])
            if schema_version == 4
            else (["v28"], ["v28", "v29"])
        )
        if revision_versions not in allowed_versions:
            raise SelectionAuthorityError(
                "development revision versions are incomplete or reordered"
            )
    authority = _load_selection_authority(repository, require_clean=require_clean)
    if authority.retained_calibration.status != "ready":
        raise SelectionAuthorityError(
            "retained calibration artifact is pending and blocks development selection"
        )
    if authority.count_score_manifest.status != "ready":
        raise SelectionAuthorityError(
            "count-score manifest is pending and blocks development selection"
        )
    calibration_sha = authority.retained_calibration.sha256
    assert calibration_sha is not None
    calibration_path = repository / authority.retained_calibration.path
    if calibration_path.is_symlink() or not calibration_path.is_file():
        raise SelectionAuthorityError(
            "retained calibration artifact is absent or not a regular file"
        )
    if _file_sha256(calibration_path) != calibration_sha:
        raise SelectionAuthorityError("retained calibration artifact checksum mismatch")
    count_score_sha = authority.count_score_manifest.sha256
    assert count_score_sha is not None
    count_score_path = repository / authority.count_score_manifest.path
    if count_score_path.is_symlink() or not count_score_path.is_file():
        raise SelectionAuthorityError(
            "count-score manifest is absent or not a regular file"
        )
    if _file_sha256(count_score_path) != count_score_sha:
        raise SelectionAuthorityError("count-score manifest checksum mismatch")
    result_sha = _authority_sha(data["result_sha256"], "development result checksum")
    result_core = {key: value for key, value in data.items() if key != "result_sha256"}
    if _canonical_sha256(result_core) != result_sha:
        raise SelectionAuthorityError("development result checksum mismatch")

    status = _validate_development_dataset_status(repository)
    if not isinstance(status, Mapping):
        raise SelectionAuthorityError("validated development status is not a mapping")
    manifest_sha = _authority_sha(
        status.get("manifest_sha256"), "dataset manifest checksum"
    )
    if data["dataset_manifest_sha256"] != manifest_sha:
        raise SelectionAuthorityError(
            "development results do not bind the validated dataset manifest"
        )
    if data["count_score_manifest_sha256"] != count_score_sha:
        raise SelectionAuthorityError(
            "development results do not bind the tracked count-score manifest"
        )
    if data["retained_calibration_artifact_sha256"] != calibration_sha:
        raise SelectionAuthorityError(
            "development results do not bind the tracked calibration artifact"
        )
    if (
        status.get("namespace") != "dev"
        or status.get("status") != "completed"
        or status.get("protocol_sha256") != authority.file_sha256["study/protocol.json"]
    ):
        raise SelectionAuthorityError(
            "validated development dataset status mismatches selection authority"
        )
    rows = status.get("rows")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise SelectionAuthorityError("validated development rows are invalid")
    dataset_bindings: dict[tuple[str, str, str], tuple[str, str]] = {}
    dataset_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SelectionAuthorityError(f"validated dataset row {index} is invalid")
        mechanism = row.get("mechanism")
        biological_id = row.get("biological_id")
        technical_view = row.get("technical_view")
        key = (mechanism, biological_id, technical_view)
        if (
            mechanism not in authority.mechanisms
            or biological_id not in authority.biological_ids
            or technical_view not in authority.technical_views
            or row.get("status") != "completed"
        ):
            raise SelectionAuthorityError(
                f"validated dataset row {index} is outside the complete panel"
            )
        dataset_id = row.get("dataset_id")
        dataset_sha = row.get("dataset_sha256")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise SelectionAuthorityError("validated dataset id is invalid")
        _authority_sha(dataset_sha, "validated dataset checksum")
        if key in dataset_bindings or dataset_id in dataset_ids:
            raise SelectionAuthorityError(
                "validated development dataset identities are duplicated"
            )
        dataset_bindings[key] = (dataset_id, dataset_sha)
        dataset_ids.add(dataset_id)
    expected_units = {
        (mechanism, biological_id, technical_view)
        for mechanism in authority.mechanisms
        for biological_id in authority.biological_ids
        for technical_view in authority.technical_views
    }
    if set(dataset_bindings) != expected_units:
        raise SelectionAuthorityError(
            "validated development dataset manifest is incomplete"
        )

    artifact_bindings = _validate_bound_development_artifacts(
        repository,
        authority,
        status,
        MappingProxyType(dataset_bindings),
    )
    downstream_bindings: Mapping[str, str] = MappingProxyType({})
    if schema_version == 4:
        downstream_bindings = validate_downstream_selection_completeness(
            repository,
            data["records"],
            data["downstream_evidence"],
        )
    if schema_version == 2 or (schema_version == 4 and data["revision_versions"] == []):
        try:
            from .evaluation_manifest import (
                EvaluationManifestError,
                validate_selection_evaluation_manifest,
            )

            evaluation_data = data
            if schema_version == 4:
                evaluation_core = {
                    key: value
                    for key, value in data.items()
                    if key
                    not in {
                        "downstream_evidence",
                        "revision_versions",
                        "result_sha256",
                    }
                }
                evaluation_core["schema_version"] = 2
                evaluation_data = {
                    **evaluation_core,
                    "result_sha256": _canonical_sha256(evaluation_core),
                }
            evaluation_evidence = validate_selection_evaluation_manifest(
                repository, evaluation_data, authority, status
            )
        except EvaluationManifestError as error:
            raise SelectionAuthorityError(
                f"development evaluation manifest failed validation: {error}"
            ) from error
        selection_authority = authority
    else:
        try:
            from .revision_evaluation import (
                RevisionEvaluationError,
                validate_revision_selection_evaluation,
            )

            revision_data = data
            if schema_version == 4:
                revision_core = {
                    key: value
                    for key, value in data.items()
                    if key not in {"downstream_evidence", "result_sha256"}
                }
                revision_core["schema_version"] = 3
                revision_data = {
                    **revision_core,
                    "result_sha256": _canonical_sha256(revision_core),
                }
            evaluation_evidence = validate_revision_selection_evaluation(
                repository,
                revision_data,
                data["revision_versions"][-1],
                require_clean=require_clean,
            )
        except RevisionEvaluationError as error:
            raise SelectionAuthorityError(
                f"development revision evaluation failed validation: {error}"
            ) from error
        selection_authority = evaluation_evidence.authority
        if not isinstance(selection_authority, SelectionAuthority):
            raise SelectionAuthorityError(
                "development revision evaluation returned invalid authority"
            )

    if schema_version == 4 and data["revision_versions"] == []:
        expected_checkpoint_binding = {
            "downstream_source_checkpoint_path": "reconstruction_checkpoint_path",
            "downstream_source_checkpoint_file_sha256": (
                "reconstruction_checkpoint_file_sha256"
            ),
            "downstream_source_checkpoint_payload_sha256": (
                "reconstruction_checkpoint_payload_sha256"
            ),
            "downstream_source_plan_sha256": "reconstruction_plan_sha256",
            "downstream_source_input_hashes_sha256": (
                "reconstruction_input_hashes_sha256"
            ),
        }
        if any(
            downstream_bindings.get(downstream_name)
            != evaluation_evidence.bindings.get(evaluation_name)
            for downstream_name, evaluation_name in expected_checkpoint_binding.items()
        ):
            raise SelectionAuthorityError(
                "downstream source checkpoint differs from evaluation authority"
            )
    elif schema_version == 4:
        _validate_revision_downstream_source_bindings(
            downstream_bindings,
            evaluation_evidence.bindings,
            tuple(data["revision_versions"]),
        )

    report = _evaluate_development_candidates(
        data["records"],
        selection_authority.attempts,
        selection_authority.declarations,
        data["orthogonal_intervals"],
        mechanisms=selection_authority.mechanisms,
        biological_ids=selection_authority.biological_ids,
        technical_views=selection_authority.technical_views,
        model_seeds=selection_authority.model_seeds,
        required_orthogonal_endpoints=tuple(
            policy.id for policy in selection_authority.endpoint_policies
        ),
        dataset_bindings=MappingProxyType(dataset_bindings),
        method_bindings=selection_authority.method_bindings,
        endpoint_policies=selection_authority.endpoint_policies,
        revision_policy=selection_authority.revision_policy,
        exclusions=selection_authority.exclusions,
    )
    bindings = {
        relative.replace("/", "_").replace(".", "_") + "_sha256": digest
        for relative, digest in selection_authority.file_sha256.items()
    }
    bindings.update(
        {
            "dataset_manifest_sha256": manifest_sha,
            "development_result_sha256": result_sha,
            "retained_calibration_artifact_sha256": calibration_sha,
            "count_score_manifest_sha256": count_score_sha,
            **artifact_bindings,
            **dict(downstream_bindings),
            **dict(evaluation_evidence.bindings),
        }
    )
    return SelectionReport(
        assessments=report.assessments,
        pareto_set=report.pareto_set,
        selected_configuration=report.selected_configuration,
        trigger=report.trigger,
        excluded_configurations=report.excluded_configurations,
        authority_bindings=MappingProxyType(bindings),
    )


def _validate_development_dataset_status(repository: Path) -> Mapping[str, Any]:
    """Invoke the dataset registry's read-only byte-level status validator."""

    try:
        from maskimpute_benchmark.datasets import validate_dataset_status

        return validate_dataset_status(
            repository / "artifacts/study/development/results/dataset_status.json",
            repo=repository,
        )
    except Exception as error:
        raise SelectionAuthorityError(
            "development dataset status failed read-only revalidation"
        ) from error


def _validate_bound_development_artifacts(
    repository: Path,
    authority: SelectionAuthority,
    status: Mapping[str, Any],
    dataset_bindings: Mapping[tuple[str, str, str], tuple[str, str]],
) -> Mapping[str, str]:
    """Validate the ready count-score and calibration artifacts against datasets."""

    score_payload = _read_authority_json(
        repository, authority.count_score_manifest.path
    )
    score_root = _exact_authority_mapping(
        score_payload,
        {
            "schema_version",
            "artifact_type",
            "dataset_manifest_sha256",
            "count_model_config_sha256",
            "dataset_qc_policy_sha256",
            "entries",
            "manifest_sha256",
        },
        "count-score manifest",
    )
    if (
        score_root["schema_version"] != 1
        or score_root["artifact_type"] != "maskimpute_development_count_score_manifest"
        or score_root["dataset_manifest_sha256"] != status.get("manifest_sha256")
        or score_root["count_model_config_sha256"]
        != authority.count_model_config_sha256
        or score_root["dataset_qc_policy_sha256"] != authority.dataset_qc_policy_sha256
    ):
        raise SelectionAuthorityError(
            "count-score manifest authority bindings are invalid"
        )
    manifest_digest = _authority_sha(
        score_root["manifest_sha256"], "count-score payload checksum"
    )
    unsigned_score = {
        key: value for key, value in score_root.items() if key != "manifest_sha256"
    }
    if _canonical_sha256(unsigned_score) != manifest_digest:
        raise SelectionAuthorityError("count-score manifest payload checksum mismatch")
    entries = score_root["entries"]
    if type(entries) is not list:
        raise SelectionAuthorityError("count-score entries must be an array")
    expected_keys = tuple(
        (mechanism, biological_id, technical_view)
        for mechanism in authority.mechanisms
        for biological_id in authority.biological_ids
        for technical_view in authority.technical_views
    )
    entry_fields = {
        "mechanism",
        "biological_id",
        "technical_view",
        "dataset_id",
        "dataset_sha256",
        "input_sha256",
        "cell_ids_sha256",
        "excluded_cell_count",
        "excluded_cell_ids_sha256",
        "retained_cell_count",
        "retained_cell_ids_sha256",
        "score_sha256",
        "config_sha256",
    }
    score_by_unit: dict[tuple[str, str, str], dict[str, Any]] = {}
    score_hashes: set[str] = set()
    observed_order = []
    for index, raw in enumerate(entries):
        entry = _exact_authority_mapping(
            raw, entry_fields, f"count-score entry {index}"
        )
        key = (
            entry["mechanism"],
            entry["biological_id"],
            entry["technical_view"],
        )
        observed_order.append(key)
        if key in score_by_unit or key not in dataset_bindings:
            raise SelectionAuthorityError(
                "count-score entries are duplicated or outside the dataset panel"
            )
        if (entry["dataset_id"], entry["dataset_sha256"]) != dataset_bindings[key]:
            raise SelectionAuthorityError(
                "count-score entry mismatches the validated dataset identity"
            )
        if entry["config_sha256"] != authority.count_model_config_sha256:
            raise SelectionAuthorityError(
                "count-score entry configuration differs from tracked authority"
            )
        if (
            type(entry["excluded_cell_count"]) is not int
            or entry["excluded_cell_count"] < 0
            or type(entry["retained_cell_count"]) is not int
            or entry["retained_cell_count"] <= 0
            or entry["retained_cell_count"]
            < authority.dataset_qc_policy["minimum_retained_cells"]
            or entry["excluded_cell_count"] + entry["retained_cell_count"] != 900
            or entry["cell_ids_sha256"] != entry["retained_cell_ids_sha256"]
        ):
            raise SelectionAuthorityError(
                "count-score entry violates the tracked zero-library QC policy"
            )
        for field in (
            "dataset_sha256",
            "input_sha256",
            "cell_ids_sha256",
            "excluded_cell_ids_sha256",
            "retained_cell_ids_sha256",
            "score_sha256",
            "config_sha256",
        ):
            _authority_sha(entry[field], f"count-score entry {field}")
        if entry["score_sha256"] in score_hashes:
            raise SelectionAuthorityError("count-score hashes are not unique")
        score_hashes.add(entry["score_sha256"])
        score_by_unit[key] = entry
    if tuple(observed_order) != expected_keys:
        raise SelectionAuthorityError(
            "count-score entries do not cover the canonical complete panel"
        )
    for mechanism in authority.mechanisms:
        for biological_id in authority.biological_ids:
            pair = tuple(
                score_by_unit[(mechanism, biological_id, technical_view)]
                for technical_view in authority.technical_views
            )
            audit_fields = (
                "excluded_cell_count",
                "excluded_cell_ids_sha256",
                "retained_cell_count",
                "retained_cell_ids_sha256",
                "cell_ids_sha256",
            )
            if any(
                first[field] != second[field]
                for field in audit_fields
                for first, second in (pair,)
            ):
                raise SelectionAuthorityError(
                    "paired count-score entries violate pair-union QC identity"
                )

    calibration_payload = _read_authority_json(
        repository, authority.retained_calibration.path
    )
    try:
        from maskimpute.calibration import CalibrationArtifact

        calibration = CalibrationArtifact(calibration_payload)
    except Exception as error:
        raise SelectionAuthorityError(
            "retained calibration artifact failed semantic validation"
        ) from error
    verified_calibration = calibration.to_dict()
    if verified_calibration.get("schema_version") != 3:
        raise SelectionAuthorityError("retained calibration must use artifact schema 3")
    if verified_calibration.get("retention_contract") != {
        "contract_id": "prezero-calibration-retention-binding-v1",
        "path": "study/calibration_contract.json",
        "sha256": authority.file_sha256["study/calibration_contract.json"],
    }:
        raise SelectionAuthorityError(
            "retained calibration does not bind the adopted retention contract"
        )
    holdout_calibrators = verified_calibration.get("development_holdout_calibrators")
    if not isinstance(holdout_calibrators, list) or [
        (item.get("mechanism"), item.get("biological_id"))
        for item in holdout_calibrators
        if isinstance(item, Mapping)
    ] != [("symsim", "draw-01"), ("symsim", "draw-02")]:
        raise SelectionAuthorityError(
            "retained calibration lacks the two canonical LODO development folds"
        )
    calibration_payload_sha = _authority_sha(
        verified_calibration.get("payload_sha256"),
        "retained calibration payload checksum",
    )
    training = verified_calibration.get("training")
    if not isinstance(training, Mapping):
        raise SelectionAuthorityError("retained calibration training is invalid")
    record_bindings = training.get("record_bindings")
    if type(record_bindings) is not list:
        raise SelectionAuthorityError(
            "retained calibration record bindings are invalid"
        )
    expected_calibration_keys = tuple(
        ("symsim", biological_id, technical_view)
        for biological_id in authority.biological_ids
        for technical_view in authority.technical_views
    )
    observed_calibration_keys = []
    for index, binding in enumerate(record_bindings):
        if not isinstance(binding, Mapping):
            raise SelectionAuthorityError(
                f"retained calibration binding {index} is invalid"
            )
        key = (
            binding.get("mechanism"),
            binding.get("biological_id"),
            binding.get("technical_view"),
        )
        observed_calibration_keys.append(key)
        dataset = dataset_bindings.get(key)
        score = score_by_unit.get(key)
        if (
            dataset is None
            or score is None
            or (binding.get("dataset_id"), binding.get("dataset_sha256")) != dataset
            or binding.get("manifest_sha256") != score["score_sha256"]
            or binding.get("protocol_sha256")
            != authority.file_sha256["study/protocol.json"]
            or binding.get("namespace") != "dev"
            or binding.get("data_role") != "development"
        ):
            raise SelectionAuthorityError(
                "retained calibration does not bind the validated score/dataset panel"
            )
    if tuple(observed_calibration_keys) != expected_calibration_keys:
        raise SelectionAuthorityError(
            "retained calibration does not cover the complete exact-truth panel"
        )
    selected_algorithm = calibration.selected_algorithm
    if selected_algorithm == "identity":
        effect_status = "not_applicable_for_efficacy"
        equivalence_reason = "retained_identity_calibrator_equals_direct_score"
    else:
        effect_status = "retained_calibration_effect_estimable"
        equivalence_reason = "retained_nonidentity_calibrator_transformed_score"
    return MappingProxyType(
        {
            "count_score_manifest_payload_sha256": manifest_digest,
            "calibration_payload_sha256": calibration_payload_sha,
            "retained_calibration_algorithm": selected_algorithm,
            "calibration_effect_status": effect_status,
            "calibration_equivalence_reason": equivalence_reason,
        }
    )


def _dataset_bindings_for_finalization(
    authority: SelectionAuthority,
    status: Mapping[str, Any],
) -> Mapping[tuple[str, str, str], tuple[str, str]]:
    if (
        status.get("namespace") != "dev"
        or status.get("status") != "completed"
        or status.get("protocol_sha256") != authority.file_sha256["study/protocol.json"]
    ):
        raise SelectionAuthorityError(
            "validated development dataset status mismatches selection authority"
        )
    rows = status.get("rows")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise SelectionAuthorityError("validated development rows are invalid")
    expected = tuple(
        (mechanism, biological_id, technical_view)
        for mechanism in authority.mechanisms
        for biological_id in authority.biological_ids
        for technical_view in authority.technical_views
    )
    bindings: dict[tuple[str, str, str], tuple[str, str]] = {}
    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "completed":
            raise SelectionAuthorityError("development dataset row is not completed")
        key = (
            row.get("mechanism"),
            row.get("biological_id"),
            row.get("technical_view"),
        )
        dataset_id = row.get("dataset_id")
        dataset_sha = row.get("dataset_sha256")
        if (
            key not in expected
            or key in bindings
            or not isinstance(dataset_id, str)
            or not dataset_id
            or dataset_id in ids
        ):
            raise SelectionAuthorityError(
                "development dataset identities are incomplete or duplicated"
            )
        _authority_sha(dataset_sha, "development dataset checksum")
        bindings[key] = (dataset_id, dataset_sha)
        ids.add(dataset_id)
    if tuple(bindings) != expected:
        raise SelectionAuthorityError(
            "development dataset rows are not the canonical complete panel"
        )
    return MappingProxyType(bindings)


def _atomic_write_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_development_artifact_bindings_for_repository(
    repository: Path,
    *,
    require_clean: bool = True,
) -> Mapping[str, str]:
    """Validate both pending artifacts, then atomically bind both exact files."""

    authority = _load_selection_authority(repository, require_clean=require_clean)
    if (
        authority.count_score_manifest.status != "pending"
        or authority.retained_calibration.status != "pending"
    ):
        raise SelectionAuthorityError(
            "both development artifact bindings must be pending before finalization"
        )
    status = _validate_development_dataset_status(repository)
    bindings = _dataset_bindings_for_finalization(authority, status)
    paths = {
        "count_score_manifest": repository / authority.count_score_manifest.path,
        "retained_calibration_artifact": (
            repository / authority.retained_calibration.path
        ),
    }
    observed_hashes: dict[str, str] = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise SelectionAuthorityError(
                f"pending {name.replace('_', ' ')} is absent or not a regular file"
            )
        observed_hashes[name] = _file_sha256(path)
    preparation = _revalidate_development_score_preparation(repository)
    if (
        preparation.get("status") != "reused"
        or preparation.get("count_score_manifest_file_sha256")
        != observed_hashes["count_score_manifest"]
        or preparation.get("calibration_file_sha256")
        != observed_hashes["retained_calibration_artifact"]
    ):
        raise SelectionAuthorityError(
            "development score preparation did not revalidate the exact pending files"
        )
    semantic = _validate_bound_development_artifacts(
        repository,
        authority,
        status,
        bindings,
    )
    for name, path in paths.items():
        if _file_sha256(path) != observed_hashes[name]:
            raise SelectionAuthorityError(
                "development artifact changed during finalization validation"
            )
    ledger_path = repository / "study/development_search.json"
    if (
        _file_sha256(ledger_path)
        != authority.file_sha256["study/development_search.json"]
    ):
        raise SelectionAuthorityError(
            "development search ledger changed during finalization"
        )
    ledger = _read_authority_json(repository, "study/development_search.json")
    for name in paths:
        binding = ledger[name]
        if binding["status"] != "pending" or binding["sha256"] is not None:
            raise SelectionAuthorityError(
                "development artifact binding changed during finalization"
            )
        binding["status"] = "ready"
        binding["sha256"] = observed_hashes[name]
    _atomic_write_json(ledger_path, ledger)
    return MappingProxyType(
        {
            "count_score_manifest_sha256": observed_hashes["count_score_manifest"],
            "retained_calibration_artifact_sha256": observed_hashes[
                "retained_calibration_artifact"
            ],
            "dataset_manifest_sha256": status["manifest_sha256"],
            **semantic,
            "next_required_action": "commit_development_search_ledger",
        }
    )


def finalize_development_artifact_bindings() -> Mapping[str, str]:
    """Finalize fixed development artifacts; the resulting ledger must be committed."""

    return _finalize_development_artifact_bindings_for_repository(_REPOSITORY_ROOT)


def _revalidate_development_score_preparation(
    repository: Path,
) -> Mapping[str, object]:
    try:
        from maskimpute_benchmark.development_scores import (
            prepare_development_scores,
        )

        result = prepare_development_scores(repository)
    except Exception as error:
        raise SelectionAuthorityError(
            "development score/calibration artifacts failed full byte revalidation"
        ) from error
    if not isinstance(result, Mapping):
        raise SelectionAuthorityError(
            "development score preparation returned an invalid receipt"
        )
    return result


def select_development_candidate(payload: object) -> SelectionReport:
    """Select from results using only clean, repository-owned study authority."""

    return _select_for_repository(payload, _REPOSITORY_ROOT)


def load_publication_execution_authority() -> SelectionAuthority:
    """Load the clean tracked config, score/calibration bindings, and run grid."""

    return _load_selection_authority(_REPOSITORY_ROOT)


__all__ = [
    "CandidateAssessment",
    "CandidateAttempt",
    "EndpointPolicy",
    "GateResult",
    "MethodDeclaration",
    "RevisionPolicy",
    "SearchExclusion",
    "SelectionAuthority",
    "SelectionAuthorityError",
    "SelectionReport",
    "attach_downstream_evidence_to_selection_result",
    "finalize_development_artifact_bindings",
    "load_publication_execution_authority",
    "select_development_candidate",
    "validate_downstream_selection_completeness",
]
