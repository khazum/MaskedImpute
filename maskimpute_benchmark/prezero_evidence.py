"""Audit-ready evidence for MaskImpute's realized pre-capture-zero score."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
import zlib

import numpy as np

from .metrics import (
    MetricValue,
    stratified_zero_score_metrics,
    tie_aware_groups,
    zero_score_metrics,
)
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
_BOUNDED_METRIC_NAMES = frozenset({"auroc", "auprc", "brier", "ece"})
_POLICY_FIELDS = {
    "schema_version",
    "probability_semantics",
    "evaluation_domain",
    "score_source",
    "score_artifact_sha256",
    "score_input_sha256",
    "score_config_sha256",
    "calibration_artifact_sha256",
    "calibration_algorithm",
    "calibration_scope",
    "calibration_equivalence_reason",
}
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
_IDENTITY_FIELDS = {
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
_STORAGE_FIELDS = {
    "encoding",
    "compression_level",
    "path",
    "compressed_sha256",
    "compressed_nbytes",
    "uncompressed_sha256",
    "uncompressed_nbytes",
}


class PreZeroEvidenceError(ValueError):
    """Raised when realized score evidence is absent, ambiguous, or malformed."""


def zlib_compress_bound(uncompressed_nbytes: int) -> int:
    """Return zlib's documented single-call compression upper bound."""

    if (
        isinstance(uncompressed_nbytes, bool)
        or type(uncompressed_nbytes) is not int
        or uncompressed_nbytes < 0
    ):
        raise ValueError("uncompressed_nbytes must be a nonnegative integer")
    return (
        uncompressed_nbytes
        + (uncompressed_nbytes >> 12)
        + (uncompressed_nbytes >> 14)
        + (uncompressed_nbytes >> 25)
        + 13
    )


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
        raise PreZeroEvidenceError("realized p_pre_zero must be an exact NumPy ndarray")
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


def _metric_record(
    metric: MetricValue, *, status: str | None = None
) -> dict[str, object]:
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
    observed_zero_count = int((observed == 0).sum())
    if unavailable_status is not None and truth is None:
        if not unavailable_reason:
            raise PreZeroEvidenceError("unavailable score evidence requires a reason")
        overall = {
            "stratum_type": "overall",
            "label": "all_observed_zeros",
            "lower": None,
            "upper": None,
            "n": observed_zero_count,
            **_unavailable_metric_group(
                observed_zero_count, unavailable_status, unavailable_reason
            ),
        }
        library_size = np.sum(observed, axis=1)
        cell_chunks = tie_aware_groups(library_size, 4)
        cell_chunks.extend(np.array([], dtype=int) for _ in range(4 - len(cell_chunks)))
        library_records: list[dict[str, object]] = []
        for quartile, cells in enumerate(cell_chunks, start=1):
            n = int((observed[cells] == 0).sum()) if len(cells) else 0
            library_records.append(
                {
                    "stratum_type": "library_size_quartiles",
                    "label": f"Q{quartile}",
                    "lower": (
                        None if not len(cells) else float(np.min(library_size[cells]))
                    ),
                    "upper": (
                        None if not len(cells) else float(np.max(library_size[cells]))
                    ),
                    "n": n,
                    **_unavailable_metric_group(
                        n, unavailable_status, unavailable_reason
                    ),
                }
            )
        truth_records = [
            {
                "stratum_type": "truth_expression_bins",
                "label": label,
                "lower": lower,
                "upper": upper,
                "n": 0,
                **_unavailable_metric_group(0, unavailable_status, unavailable_reason),
            }
            for label, lower, upper in (
                ("[0,1)", 0.0, 1.0),
                ("[1,2)", 1.0, 2.0),
                ("[2,4)", 2.0, 4.0),
                ("[4,inf)", 4.0, None),
            )
        ]
        return overall, {
            "library_size_quartiles": library_records,
            "truth_expression_bins": truth_records,
        }
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


def encode_prezero_evidence(
    evidence: PreZeroEvidence,
) -> tuple[dict[str, object], bytes | None]:
    """Return a canonical record and deterministic compressed matrix bytes."""

    if type(evidence) is not PreZeroEvidence:
        raise TypeError("evidence must be an exact PreZeroEvidence")
    record = evidence.to_record()
    raw = evidence.raw_matrix_bytes
    if raw is None:
        return record, None
    compressed = zlib.compress(raw, level=PREZERO_STORAGE_COMPRESSION_LEVEL)
    record["storage"] = {
        "encoding": PREZERO_STORAGE_ENCODING,
        "compression_level": PREZERO_STORAGE_COMPRESSION_LEVEL,
        "path": None,
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_nbytes": len(compressed),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "uncompressed_nbytes": len(raw),
    }
    return record, compressed


def _bounded_decompress(compressed: bytes, expected_nbytes: int) -> bytes:
    if type(compressed) is not bytes:
        raise TypeError("compressed p_pre_zero evidence must be exact bytes")
    if len(compressed) > zlib_compress_bound(expected_nbytes):
        raise PreZeroEvidenceError(
            "compressed p_pre_zero evidence exceeds its zlib bound"
        )
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_nbytes + 1)
        raw += decompressor.flush(max(1, expected_nbytes + 1 - len(raw)))
    except zlib.error as error:
        raise PreZeroEvidenceError(
            "compressed p_pre_zero evidence is invalid"
        ) from error
    if (
        len(raw) != expected_nbytes
        or not decompressor.eof
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        raise PreZeroEvidenceError(
            "compressed p_pre_zero evidence differs from its bounded receipt"
        )
    return raw


def _validate_metric_group(
    value: object,
    *,
    expected_n: int,
    evidence_status: str,
    evidence_reason: str | None,
    truth_kind: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "metrics",
        "reliability_bins",
    }:
        raise PreZeroEvidenceError("p_pre_zero metric group has the wrong schema")
    metrics = value.get("metrics")
    bins = value.get("reliability_bins")
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METRIC_NAMES):
        raise PreZeroEvidenceError("p_pre_zero metric denominator is incomplete")
    truth_reason = {
        "exact_continuous": "undefined_for_continuous_truth",
        "proxy_high_depth": "proxy_truth_not_exact",
        "orthogonal_only": "truth_unavailable",
    }.get(truth_kind)
    for name in _METRIC_NAMES:
        metric = metrics.get(name)
        if not isinstance(metric, Mapping) or set(metric) != {
            "value",
            "n",
            "status",
            "reason",
        }:
            raise PreZeroEvidenceError(f"p_pre_zero metric {name} is malformed")
        if metric.get("n") != expected_n:
            raise PreZeroEvidenceError(f"p_pre_zero metric {name} denominator differs")
        metric_value = metric.get("value")
        metric_status = metric.get("status")
        metric_reason = metric.get("reason")
        if evidence_status != "completed":
            if (
                metric_value is not None
                or metric_status != evidence_status
                or metric_reason != evidence_reason
            ):
                raise PreZeroEvidenceError(
                    "noncompleted p_pre_zero metric does not preserve its reason"
                )
        elif truth_reason is not None:
            if (
                metric_value is not None
                or metric_status != "unavailable"
                or metric_reason != truth_reason
            ):
                raise PreZeroEvidenceError(
                    "p_pre_zero metric truth-kind reason differs"
                )
        elif metric_value is None:
            if metric_status != "unavailable" or not isinstance(metric_reason, str):
                raise PreZeroEvidenceError(
                    "unavailable exact-truth p_pre_zero metric lacks a reason"
                )
        elif (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, (int, float))
            or not np.isfinite(float(metric_value))
            or metric_status != "completed"
            or metric_reason is not None
        ):
            raise PreZeroEvidenceError(
                "completed exact-truth p_pre_zero metric is invalid"
            )
        else:
            numeric = float(metric_value)
            if name in _BOUNDED_METRIC_NAMES and not 0.0 <= numeric <= 1.0:
                raise PreZeroEvidenceError(
                    f"p_pre_zero metric {name} must be in [0, 1]"
                )
            if name == "log_loss" and numeric < 0.0:
                raise PreZeroEvidenceError(
                    "p_pre_zero metric log_loss must be nonnegative"
                )
    if not isinstance(bins, list):
        raise PreZeroEvidenceError("p_pre_zero reliability bins must be a list")
    if evidence_status != "completed" or truth_reason is not None:
        if bins:
            raise PreZeroEvidenceError(
                "unavailable p_pre_zero evidence cannot have reliability bins"
            )
        return
    if expected_n == 0:
        if bins:
            raise PreZeroEvidenceError(
                "zero-denominator p_pre_zero metrics cannot have reliability bins"
            )
        return
    if not 1 <= len(bins) <= 10:
        raise PreZeroEvidenceError(
            "p_pre_zero reliability bin count must be between one and ten"
        )
    total_bin_n = 0
    for index, item in enumerate(bins, start=1):
        if not isinstance(item, Mapping) or set(item) != {
            "bin",
            "n",
            "mean_prediction",
            "observed_fraction",
            "wilson_lower",
            "wilson_upper",
        }:
            raise PreZeroEvidenceError("p_pre_zero reliability bin is malformed")
        if (
            item.get("bin") != index
            or type(item.get("n")) is not int
            or item.get("n") <= 0
        ):
            raise PreZeroEvidenceError("p_pre_zero reliability bin order differs")
        total_bin_n += item["n"]
        for name in (
            "mean_prediction",
            "observed_fraction",
            "wilson_lower",
            "wilson_upper",
        ):
            nested = item.get(name)
            if (
                isinstance(nested, bool)
                or not isinstance(nested, (int, float))
                or not np.isfinite(float(nested))
            ):
                raise PreZeroEvidenceError(
                    "p_pre_zero reliability bin contains an invalid number"
                )
            if not 0.0 <= float(nested) <= 1.0:
                raise PreZeroEvidenceError(
                    "p_pre_zero reliability bin probability is outside [0, 1]"
                )
        observed_fraction = float(item["observed_fraction"])
        wilson_lower = float(item["wilson_lower"])
        wilson_upper = float(item["wilson_upper"])
        if not wilson_lower <= observed_fraction <= wilson_upper:
            raise PreZeroEvidenceError(
                "p_pre_zero reliability interval does not contain its observation"
            )
    if total_bin_n != expected_n:
        raise PreZeroEvidenceError(
            "p_pre_zero reliability denominator differs from its metric group"
        )


def validate_stored_prezero_evidence(
    value: object,
    *,
    expected_identity: Mapping[str, object],
    run_status: str,
    run_reason: str | None,
    observed_zero_count: int,
    expected_shape: tuple[int, int],
    requires_count_score: bool,
    requires_calibration: bool,
    expected_calibration_artifact_sha256: str | None,
    compressed: bytes | None,
    observed: np.ndarray,
    truth: np.ndarray | None,
    truth_kind: str,
    expected_score_input_sha256: str | None,
    expected_score_config_sha256: str | None,
) -> dict[str, object]:
    """Validate one persisted evidence record and its bounded matrix bytes."""

    try:
        observed_array = np.array(
            observed, dtype="<f8", copy=True, order="C", subok=False
        )
        truth_array = (
            None
            if truth is None
            else np.array(truth, dtype="<f8", copy=True, order="C", subok=False)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise PreZeroEvidenceError(
            "authoritative p_pre_zero score targets are invalid"
        ) from error
    if (
        observed_array.shape != expected_shape
        or observed_array.ndim != 2
        or not np.isfinite(observed_array).all()
        or bool((observed_array < 0.0).any())
        or int((observed_array == 0.0).sum()) != observed_zero_count
    ):
        raise PreZeroEvidenceError(
            "authoritative p_pre_zero observed targets differ from the run"
        )
    if truth_kind not in {
        "exact_pre_capture",
        "exact_continuous",
        "proxy_high_depth",
        "orthogonal_only",
    }:
        raise PreZeroEvidenceError("authoritative p_pre_zero truth kind is invalid")
    if truth_kind == "orthogonal_only":
        if truth_array is not None:
            raise PreZeroEvidenceError(
                "orthogonal p_pre_zero authority cannot expose evaluator truth"
            )
    elif (
        truth_array is None
        or truth_array.shape != expected_shape
        or not np.isfinite(truth_array).all()
        or bool((truth_array < 0.0).any())
    ):
        raise PreZeroEvidenceError("authoritative p_pre_zero truth targets are invalid")

    if not isinstance(value, dict) or set(value) != {
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
    }:
        raise PreZeroEvidenceError("p_pre_zero evidence has the wrong schema")
    if value.get("schema_version") != 1:
        raise PreZeroEvidenceError("p_pre_zero evidence schema_version must be 1")
    identity = value.get("identity")
    canonical_identity = _json_copy(dict(expected_identity))
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _IDENTITY_FIELDS
        or dict(identity) != canonical_identity
    ):
        raise PreZeroEvidenceError("p_pre_zero evidence identity differs from its run")
    method_id = identity.get("method_id")
    evidence_status = value.get("status")
    evidence_reason = value.get("reason")
    expected_status = run_status if method_id == "maskimpute" else "not_applicable"
    expected_reason = (
        run_reason if method_id == "maskimpute" else "method_does_not_emit_p_pre_zero"
    )
    if evidence_status != expected_status or evidence_reason != expected_reason:
        raise PreZeroEvidenceError(
            "p_pre_zero evidence status or reason differs from its run"
        )
    stored_truth_kind = value.get("truth_kind")
    if stored_truth_kind != truth_kind:
        raise PreZeroEvidenceError(
            "p_pre_zero evidence truth kind differs from evaluator authority"
        )
    body = {
        key: nested
        for key, nested in value.items()
        if key not in {"evidence_sha256", "storage"}
    }
    if value.get("evidence_sha256") != canonical_sha256(body):
        raise PreZeroEvidenceError("p_pre_zero evidence payload checksum differs")

    matrix = value.get("matrix")
    policy = value.get("policy")
    policy_sha256 = value.get("policy_sha256")
    storage = value.get("storage")
    if not isinstance(matrix, Mapping) or set(matrix) != {
        "shape",
        "dtype",
        "content_sha256",
        "semantic_sha256",
    }:
        raise PreZeroEvidenceError("p_pre_zero matrix receipt is malformed")
    if not isinstance(storage, Mapping) or set(storage) != _STORAGE_FIELDS:
        raise PreZeroEvidenceError("p_pre_zero storage receipt is malformed")
    matrix_present = matrix.get("shape") is not None
    probability: np.ndarray | None = None
    if matrix_present:
        if method_id != "maskimpute" or not requires_count_score:
            raise PreZeroEvidenceError(
                "p_pre_zero matrix is not authorized for this configuration"
            )
        if matrix.get("shape") != list(expected_shape) or matrix.get("dtype") != "<f8":
            raise PreZeroEvidenceError("p_pre_zero matrix shape or dtype differs")
        content_sha256 = _require_sha256(
            matrix.get("content_sha256"), "p_pre_zero matrix content"
        )
        _require_sha256(matrix.get("semantic_sha256"), "p_pre_zero semantic matrix")
        if not isinstance(policy, Mapping) or set(policy) != _POLICY_FIELDS:
            raise PreZeroEvidenceError("p_pre_zero score policy is unavailable")
        policy_value = _json_copy(dict(policy))
        if policy_sha256 != canonical_sha256(policy_value):
            raise PreZeroEvidenceError("p_pre_zero policy checksum differs")
        if (
            policy.get("schema_version") != 1
            or policy.get("probability_semantics")
            != "pre_capture_count_is_zero_given_observed_counts"
            or policy.get("evaluation_domain") != "observed_zero_entries_only"
            or policy.get("score_source") not in {"direct", "retained_calibrator"}
        ):
            raise PreZeroEvidenceError("p_pre_zero score policy semantics differ")
        for name in (
            "score_artifact_sha256",
            "score_input_sha256",
            "score_config_sha256",
            "calibration_artifact_sha256",
        ):
            _require_sha256(policy.get(name), f"p_pre_zero policy {name}")
        expected_score_input = _require_sha256(
            expected_score_input_sha256,
            "authoritative p_pre_zero score input",
        )
        expected_score_config = _require_sha256(
            expected_score_config_sha256,
            "authoritative p_pre_zero score configuration",
        )
        expected_calibration = _require_sha256(
            expected_calibration_artifact_sha256,
            "authoritative p_pre_zero calibration artifact",
        )
        if policy.get("score_input_sha256") != expected_score_input:
            raise PreZeroEvidenceError(
                "p_pre_zero score input differs from execution authority"
            )
        if policy.get("score_config_sha256") != expected_score_config:
            raise PreZeroEvidenceError(
                "p_pre_zero score configuration differs from execution authority"
            )
        if policy.get("calibration_artifact_sha256") != expected_calibration:
            raise PreZeroEvidenceError(
                "p_pre_zero calibration policy differs from execution authority"
            )
        for name in (
            "calibration_algorithm",
            "calibration_scope",
            "calibration_equivalence_reason",
        ):
            if not isinstance(policy.get(name), str) or not policy.get(name):
                raise PreZeroEvidenceError(
                    f"p_pre_zero policy {name} must be a nonempty string"
                )
        expected_score_source = (
            "retained_calibrator" if requires_calibration else "direct"
        )
        if policy.get("score_source") != expected_score_source:
            raise PreZeroEvidenceError(
                "p_pre_zero score source differs from execution authority"
            )
        if (
            storage.get("encoding") != PREZERO_STORAGE_ENCODING
            or storage.get("compression_level") != PREZERO_STORAGE_COMPRESSION_LEVEL
            or not isinstance(storage.get("path"), str)
            or not storage.get("path")
        ):
            raise PreZeroEvidenceError("p_pre_zero storage binding is partial")
        expected_compressed_sha256 = _require_sha256(
            storage.get("compressed_sha256"), "compressed p_pre_zero matrix"
        )
        expected_nbytes = expected_shape[0] * expected_shape[1] * 8
        if (
            storage.get("uncompressed_nbytes") != expected_nbytes
            or storage.get("uncompressed_sha256") != content_sha256
            or type(storage.get("compressed_nbytes")) is not int
            or storage.get("compressed_nbytes") < 0
            or compressed is None
            or len(compressed) != storage.get("compressed_nbytes")
            or hashlib.sha256(compressed).hexdigest() != expected_compressed_sha256
        ):
            raise PreZeroEvidenceError("p_pre_zero compressed receipt differs")
        raw = _bounded_decompress(compressed, expected_nbytes)
        if hashlib.sha256(raw).hexdigest() != content_sha256:
            raise PreZeroEvidenceError("p_pre_zero uncompressed checksum differs")
        probability = np.frombuffer(raw, dtype="<f8").reshape(expected_shape)
        if not np.isfinite(probability).all() or bool(
            ((probability < 0.0) | (probability > 1.0)).any()
        ):
            raise PreZeroEvidenceError(
                "stored p_pre_zero matrix contains invalid probabilities"
            )
        semantic = hashlib.sha256()
        semantic.update(b"maskimpute-realized-p-pre-zero-v1\0")
        semantic.update(
            _canonical_bytes(
                {
                    "identity": canonical_identity,
                    "shape": list(expected_shape),
                    "dtype": "<f8",
                    "policy_sha256": policy_sha256,
                }
            )
        )
        semantic.update(raw)
        if semantic.hexdigest() != matrix.get("semantic_sha256"):
            raise PreZeroEvidenceError("p_pre_zero semantic matrix checksum differs")
    else:
        if any(matrix.get(name) is not None for name in matrix):
            raise PreZeroEvidenceError("p_pre_zero matrix receipt is partial")
        if policy is not None or policy_sha256 is not None:
            raise PreZeroEvidenceError("absent p_pre_zero matrix has a score policy")
        if (
            any(storage.get(name) is not None for name in storage)
            or compressed is not None
        ):
            raise PreZeroEvidenceError("absent p_pre_zero matrix has storage fields")
        if method_id == "maskimpute" and run_status == "completed":
            raise PreZeroEvidenceError("completed MaskImpute evidence lacks its matrix")

    overall = value.get("overall")
    if (
        not isinstance(overall, Mapping)
        or set(overall)
        != {
            "stratum_type",
            "label",
            "lower",
            "upper",
            "n",
            "metrics",
            "reliability_bins",
        }
        or overall.get("stratum_type") != "overall"
        or overall.get("label") != "all_observed_zeros"
        or overall.get("lower") is not None
        or overall.get("upper") is not None
        or overall.get("n") != observed_zero_count
    ):
        raise PreZeroEvidenceError("p_pre_zero overall denominator differs")
    _validate_metric_group(
        {
            "metrics": overall.get("metrics"),
            "reliability_bins": overall.get("reliability_bins"),
        },
        expected_n=observed_zero_count,
        evidence_status=str(evidence_status),
        evidence_reason=None if evidence_reason is None else str(evidence_reason),
        truth_kind=str(truth_kind),
    )
    strata = value.get("strata")
    if not isinstance(strata, Mapping) or set(strata) != {
        "library_size_quartiles",
        "truth_expression_bins",
    }:
        raise PreZeroEvidenceError("p_pre_zero strata are incomplete")
    library_records = strata.get("library_size_quartiles")
    truth_records = strata.get("truth_expression_bins")
    if not isinstance(library_records, list) or not isinstance(truth_records, list):
        raise PreZeroEvidenceError("p_pre_zero strata are incomplete")
    previous_library_upper: float | None = None
    padded_library_strata = False
    for stratum_type in ("library_size_quartiles", "truth_expression_bins"):
        records = strata.get(stratum_type)
        if not isinstance(records, list) or len(records) != 4:
            raise PreZeroEvidenceError("p_pre_zero strata cardinality differs")
        for index, record in enumerate(records):
            if (
                not isinstance(record, Mapping)
                or set(record)
                != {
                    "stratum_type",
                    "label",
                    "lower",
                    "upper",
                    "n",
                    "metrics",
                    "reliability_bins",
                }
                or record.get("stratum_type") != stratum_type
                or type(record.get("n")) is not int
                or record.get("n") < 0
            ):
                raise PreZeroEvidenceError("p_pre_zero stratum is malformed")
            if stratum_type == "library_size_quartiles":
                if record.get("label") != f"Q{index + 1}":
                    raise PreZeroEvidenceError(
                        "p_pre_zero library stratum label differs"
                    )
                lower = record.get("lower")
                upper = record.get("upper")
                if lower is None or upper is None:
                    if lower is not None or upper is not None:
                        raise PreZeroEvidenceError(
                            "p_pre_zero library stratum bounds are partial"
                        )
                    padded_library_strata = True
                else:
                    if (
                        padded_library_strata
                        or type(lower) is not float
                        or type(upper) is not float
                        or not np.isfinite(lower)
                        or not np.isfinite(upper)
                        or lower < 0.0
                        or lower > upper
                        or (
                            previous_library_upper is not None
                            and lower <= previous_library_upper
                        )
                    ):
                        raise PreZeroEvidenceError(
                            "p_pre_zero library stratum bounds differ"
                        )
                    previous_library_upper = upper
            else:
                expected_truth = (
                    ("[0,1)", 0.0, 1.0),
                    ("[1,2)", 1.0, 2.0),
                    ("[2,4)", 2.0, 4.0),
                    ("[4,inf)", 4.0, None),
                )[index]
                if (
                    record.get("label"),
                    record.get("lower"),
                    record.get("upper"),
                ) != expected_truth or type(record.get("lower")) is not float:
                    raise PreZeroEvidenceError(
                        "p_pre_zero truth-expression stratum bounds differ"
                    )
            _validate_metric_group(
                {
                    "metrics": record.get("metrics"),
                    "reliability_bins": record.get("reliability_bins"),
                },
                expected_n=record["n"],
                evidence_status=str(evidence_status),
                evidence_reason=(
                    None if evidence_reason is None else str(evidence_reason)
                ),
                truth_kind=str(truth_kind),
            )
    if sum(record["n"] for record in library_records) != observed_zero_count:
        raise PreZeroEvidenceError(
            "p_pre_zero library strata do not partition observed zeros"
        )
    truth_total = sum(record["n"] for record in truth_records)
    if truth_kind == "orthogonal_only":
        allowed_truth_totals = {0}
    elif evidence_status == "completed":
        allowed_truth_totals = {observed_zero_count}
    else:
        allowed_truth_totals = {0, observed_zero_count}
    if truth_total not in allowed_truth_totals:
        raise PreZeroEvidenceError(
            "p_pre_zero truth strata do not partition their declared truth"
        )
    unavailable_status = (
        None if evidence_status == "completed" else str(evidence_status)
    )
    unavailable_reason = (
        None if unavailable_status is None else str(evidence_reason)
    )
    expected_overall, expected_strata = _score_report(
        probability,
        observed_array,
        truth_array,
        truth_kind=truth_kind,
        unavailable_status=unavailable_status,
        unavailable_reason=unavailable_reason,
    )
    if _canonical_bytes(overall) != _canonical_bytes(expected_overall):
        raise PreZeroEvidenceError(
            "p_pre_zero overall report differs from evaluator authority"
        )
    if _canonical_bytes(strata) != _canonical_bytes(expected_strata):
        raise PreZeroEvidenceError(
            "p_pre_zero stratified report differs from evaluator authority"
        )
    return _json_copy(value)


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
        overall, strata = _score_report(matrix, observed, truth, truth_kind=truth_kind)
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
    "encode_prezero_evidence",
    "evaluate_prezero_evidence",
    "validate_stored_prezero_evidence",
    "zlib_compress_bound",
]
