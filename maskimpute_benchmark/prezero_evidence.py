"""Audit-ready evidence for MaskImpute's realized pre-capture-zero score."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import numpy as np

from .metrics import MetricValue, stratified_zero_score_metrics, zero_score_metrics
from .protocol import canonical_sha256


PREZERO_STORAGE_ENCODING = "zlib_raw_f64_v1"
PREZERO_STORAGE_COMPRESSION_LEVEL = 6
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_METRIC_NAMES = (
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "ece",
)
_POLICY_SOURCE_FIELDS = {
    "source": "score_source",
    "score_artifact_sha256": "score_artifact_sha256",
    "score_input_sha256": "score_input_sha256",
    "score_config_sha256": "score_config_sha256",
    "calibration_artifact_sha256": "calibration_artifact_sha256",
    "retained_calibrator": "calibration_algorithm",
    "calibration_scope": "calibration_scope",
    "equivalence_reason": "calibration_equivalence_reason",
}


class PreZeroEvidenceError(ValueError):
    """Raised when realized score evidence is absent, ambiguous, or malformed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_copy(value: object) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PreZeroEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _policy_from_execution(execution: object) -> dict[str, object]:
    policy_source = getattr(execution, "realized_p_pre_zero_policy", None)
    if not isinstance(policy_source, Mapping):
        raise PreZeroEvidenceError(
            "completed MaskImpute execution lacks realized score policy evidence"
        )
    policy: dict[str, object] = {
        "schema_version": 1,
        "probability_semantics": "pre_capture_count_is_zero_given_observed_counts",
        "evaluation_domain": "observed_zero_entries_only",
    }
    for source_name, destination_name in _POLICY_SOURCE_FIELDS.items():
        value = policy_source.get(source_name)
        if source_name.endswith("sha256"):
            policy[destination_name] = _require_sha256(
                value, f"score policy {source_name}"
            )
        elif not isinstance(value, str) or not value:
            raise PreZeroEvidenceError(
                f"score policy {source_name} must be a nonempty string"
            )
        else:
            policy[destination_name] = value
    return policy


def _probability_matrix(value: object, shape: tuple[int, int]) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise PreZeroEvidenceError(
            "realized p_pre_zero must be an exact NumPy ndarray"
        )
    try:
        probability = np.array(value, dtype="<f8", copy=True, order="C", subok=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise PreZeroEvidenceError(
            "realized p_pre_zero cannot be represented as float64"
        ) from error
    if probability.shape != shape:
        raise PreZeroEvidenceError(
            "realized p_pre_zero shape differs from the evaluator method input"
        )
    if not np.isfinite(probability).all() or bool(
        ((probability < 0.0) | (probability > 1.0)).any()
    ):
        raise PreZeroEvidenceError(
            "realized p_pre_zero must contain finite probabilities in [0, 1]"
        )
    return probability


def _metric_record(metric: MetricValue, *, status: str | None = None) -> dict[str, object]:
    return {
        "value": None if metric.value is None else float(metric.value),
        "n": int(metric.n),
        "status": (
            status
            if status is not None
            else ("unavailable" if metric.value is None else "completed")
        ),
        "reason": metric.reason,
    }


def _normalized_metric_group(value: Mapping[str, object]) -> dict[str, object]:
    source_names = {
        "auroc": "auroc",
        "auprc": "average_precision",
        "brier": "brier",
        "log_loss": "log_loss",
        "calibration_intercept": "calibration_intercept",
        "calibration_slope": "calibration_slope",
        "ece": "ece",
    }
    metrics: dict[str, object] = {}
    for destination, source in source_names.items():
        metric = value.get(source)
        if not isinstance(metric, MetricValue):
            raise PreZeroEvidenceError(
                f"score metric {source} is not a canonical MetricValue"
            )
        metrics[destination] = _metric_record(metric)
    bins = value.get("reliability_bins")
    if not isinstance(bins, list):
        raise PreZeroEvidenceError("score reliability bins are malformed")
    return {
        "metrics": metrics,
        "reliability_bins": _json_copy(bins),
    }


def _unavailable_metric_group(n: int, status: str, reason: str) -> dict[str, object]:
    metrics = {
        name: {
            "value": None,
            "n": int(n),
            "status": status,
            "reason": reason,
        }
        for name in _METRIC_NAMES
    }
    return {"metrics": metrics, "reliability_bins": []}


def _score_report(
    probability: np.ndarray | None,
    observed: np.ndarray,
    truth: np.ndarray | None,
    *,
    truth_kind: str,
    unavailable_status: str | None = None,
    unavailable_reason: str | None = None,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    placeholder = (
        probability
        if probability is not None
        else np.zeros(observed.shape, dtype="<f8")
    )
    overall_source = zero_score_metrics(
        placeholder, observed, truth, truth_kind=truth_kind
    )
    strata_source = stratified_zero_score_metrics(
        placeholder, observed, truth, truth_kind=truth_kind
    )
    observed_zero_count = int((observed == 0).sum())
    if unavailable_status is None:
        overall_group = _normalized_metric_group(overall_source)
    else:
        if not unavailable_reason:
            raise PreZeroEvidenceError("unavailable score evidence requires a reason")
        overall_group = _unavailable_metric_group(
            observed_zero_count, unavailable_status, unavailable_reason
        )
    overall = {
        "stratum_type": "overall",
        "label": "all_observed_zeros",
        "lower": None,
        "upper": None,
        "n": observed_zero_count,
        **overall_group,
    }

    strata: dict[str, list[dict[str, object]]] = {}
    for stratum_type in ("library_size_quartiles", "truth_expression_bins"):
        records: list[dict[str, object]] = []
        for source in strata_source[stratum_type]:
            metric_source = source.get("metrics")
            if not isinstance(metric_source, Mapping):
                raise PreZeroEvidenceError("stratified score metrics are malformed")
            count = source.get("n")
            if type(count) is not int or count < 0:
                raise PreZeroEvidenceError("stratified score denominator is invalid")
            group = (
                _normalized_metric_group(metric_source)
                if unavailable_status is None
                else _unavailable_metric_group(
                    count, unavailable_status, str(unavailable_reason)
                )
            )
            records.append(
                {
                    "stratum_type": stratum_type,
                    "label": source.get("label"),
                    "lower": source.get("lower"),
                    "upper": source.get("upper"),
                    "n": count,
                    **group,
                }
            )
        strata[stratum_type] = records
    return overall, strata


@dataclass(frozen=True, slots=True)
class PreZeroEvidence:
    """Immutable matrix plus canonical identity, policy, and score report."""

    status: str
    reason: str | None
    _record_bytes: bytes
    _matrix_bytes: bytes | None
    _matrix_shape: tuple[int, int] | None

    @property
    def matrix(self) -> np.ndarray | None:
        """Return a defensive copy of the realized probability matrix."""

        if self._matrix_bytes is None or self._matrix_shape is None:
            return None
        return (
            np.frombuffer(self._matrix_bytes, dtype="<f8")
            .reshape(self._matrix_shape)
            .copy()
        )

    @property
    def raw_matrix_bytes(self) -> bytes | None:
        """Return immutable canonical bytes for artifact persistence."""

        return self._matrix_bytes

    def to_record(self) -> dict[str, object]:
        """Return a defensive JSON-compatible evidence record."""

        return json.loads(self._record_bytes.decode("utf-8"))


def evaluate_prezero_evidence(
    *,
    identity: Mapping[str, object],
    method_shape: tuple[int, int],
    method_id: str,
    execution: object | None,
    run_status: str,
    run_reason: str | None,
    observed: np.ndarray,
    truth: np.ndarray | None,
    truth_kind: str,
) -> PreZeroEvidence:
    """Build one explicit score-evidence row without receipt reconstruction."""

    canonical_identity = _json_copy(dict(identity))
    matrix: np.ndarray | None = None
    policy: dict[str, object] | None = None
    policy_sha256: str | None = None
    evidence_status = run_status
    evidence_reason = run_reason

    if method_id != "maskimpute":
        evidence_status = "not_applicable"
        evidence_reason = "method_does_not_emit_p_pre_zero"
    elif execution is not None:
        realized = getattr(execution, "realized_p_pre_zero", None)
        matrix = _probability_matrix(realized, method_shape)
        policy = _policy_from_execution(execution)
        policy_sha256 = canonical_sha256(policy)
    elif run_status == "completed":
        raise PreZeroEvidenceError(
            "completed MaskImpute attempt lacks AdapterExecution evidence"
        )

    if evidence_status == "completed":
        if matrix is None or policy is None or policy_sha256 is None:
            raise PreZeroEvidenceError(
                "completed MaskImpute attempt lacks realized p_pre_zero evidence"
            )
        overall, strata = _score_report(
            matrix, observed, truth, truth_kind=truth_kind
        )
    else:
        if not evidence_reason:
            raise PreZeroEvidenceError("noncompleted score evidence requires a reason")
        overall, strata = _score_report(
            matrix,
            observed,
            truth,
            truth_kind=truth_kind,
            unavailable_status=evidence_status,
            unavailable_reason=evidence_reason,
        )

    matrix_bytes: bytes | None = None
    matrix_record: dict[str, object]
    if matrix is None:
        matrix_record = {
            "shape": None,
            "dtype": None,
            "content_sha256": None,
            "semantic_sha256": None,
        }
    else:
        matrix_bytes = matrix.tobytes(order="C")
        content_sha256 = hashlib.sha256(matrix_bytes).hexdigest()
        semantic = hashlib.sha256()
        semantic.update(b"maskimpute-realized-p-pre-zero-v1\0")
        semantic.update(
            _canonical_bytes(
                {
                    "identity": canonical_identity,
                    "shape": list(matrix.shape),
                    "dtype": "<f8",
                    "policy_sha256": policy_sha256,
                }
            )
        )
        semantic.update(matrix_bytes)
        matrix_record = {
            "shape": list(matrix.shape),
            "dtype": "<f8",
            "content_sha256": content_sha256,
            "semantic_sha256": semantic.hexdigest(),
        }

    body: dict[str, object] = {
        "schema_version": 1,
        "status": evidence_status,
        "reason": evidence_reason,
        "identity": canonical_identity,
        "truth_kind": truth_kind,
        "matrix": matrix_record,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "overall": overall,
        "strata": strata,
    }
    evidence_sha256 = canonical_sha256(body)
    record = {
        **body,
        "evidence_sha256": evidence_sha256,
        "storage": {
            "encoding": None,
            "compression_level": None,
            "path": None,
            "compressed_sha256": None,
            "compressed_nbytes": None,
            "uncompressed_sha256": None,
            "uncompressed_nbytes": None,
        },
    }
    return PreZeroEvidence(
        status=evidence_status,
        reason=evidence_reason,
        _record_bytes=_canonical_bytes(record),
        _matrix_bytes=matrix_bytes,
        _matrix_shape=None if matrix is None else matrix.shape,
    )


__all__ = [
    "PREZERO_STORAGE_COMPRESSION_LEVEL",
    "PREZERO_STORAGE_ENCODING",
    "PreZeroEvidence",
    "PreZeroEvidenceError",
    "evaluate_prezero_evidence",
]
