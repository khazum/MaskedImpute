"""Direct-identity requests and records for fair-comparator execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Literal
import zlib

import numpy as np

from .comparator_tuning import (
    COMPARATOR_METHOD_IDS,
    DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
    DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
    ComparatorConfiguration,
    ComparatorTuningAuthority,
    ComparatorTuningError,
    comparator_method_binding,
    encode_comparator_configuration,
    validate_comparator_tuning_authority,
)
from .direct_values import direct_equal, direct_json_value, freeze_direct_mapping
from .fair_comparator_plan import (
    ComparatorRunIdentity,
    DirectPlanEntry,
    PreparedInputDescriptor,
    describe_prepared_input,
    direct_run_id,
)
from .methods import (
    DirectAdapterExecution,
    CORE_EVALUATOR_COUNT_CONVERTERS,
    CORE_EVALUATOR_NATIVE_SCALES,
    LEGACY_EVALUATOR_COUNT_CONVERTERS,
    LEGACY_EVALUATOR_NATIVE_SCALES,
    RECENT_EVALUATOR_COUNT_CONVERTERS,
    RECENT_EVALUATOR_NATIVE_SCALES,
    MethodInput,
    MethodSpec,
    count_equivalent_to_log2_cp10k,
)
from .runner import AdapterOutcome, PreparedDataset, RunnerContractError
from .runtime_environments import RuntimeEnvironmentError


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

_DIRECT_TERMINAL_REASON_FALLBACK = "noncanonical_adapter_reason"
_DIRECT_TERMINAL_REASONS = frozenset(
    {
        _DIRECT_TERMINAL_REASON_FALLBACK,
        "adapter_exception",
        "adapter_not_registered",
        "adapter_returned_noncanonical_outcome",
        "comparator_payload_changed_during_adapter_attempt",
        "configuration_budget_exhausted",
        "count_score_authority_pending",
        "count_score_or_calibration_authority_pending",
        "cpu_time_budget_exhausted",
        "environment_build_receipt_malformed",
        "environment_build_receipt_mismatch",
        "environment_build_receipt_missing",
        "environment_executable_missing",
        "environment_executable_unsafe",
        "environment_execution_failed",
        "environment_library_digest_mismatch",
        "environment_library_incomplete",
        "environment_library_malformed",
        "environment_library_missing",
        "environment_lock_malformed",
        "environment_lock_mismatch",
        "environment_lock_missing",
        "environment_qualification_malformed",
        "environment_qualification_mismatch",
        "environment_qualification_missing",
        "evaluator_conversion_invalid",
        "evaluator_conversion_unavailable",
        "gpu_time_budget_exhausted",
        "legacy_environment_mismatch",
        "legacy_gpu_initialization_timeout",
        "legacy_gpu_kernel_incompatible",
        "legacy_gpu_unavailable",
        "malformed_adapter_process_message",
        "malformed_environment_receipt",
        "malformed_upstream_output",
        "memory_limit_exceeded",
        "noncanonical_adapter_process_outcome",
        "peak_gpu_exceeded",
        "peak_rss_exceeded",
        "resource_telemetry_unavailable",
        "runtime_environment_invalid",
        "source_checkout_missing",
        "source_checkout_not_pristine",
        "source_identity_changed",
        "source_revision_mismatch",
        "source_tree_mismatch",
        "source_url_mismatch",
        "source_verification_failed",
        "timeout",
        "unsafe_upstream_output",
        "unsafe_work_root",
        "upstream_minimum_dimension",
        "upstream_negative_native_output",
        "upstream_output_missing",
        "upstream_timeout",
        "zero_library_cell",
        *(
            f"environment_executable_unavailable_{method_id}"
            for method_id in COMPARATOR_METHOD_IDS
        ),
        *(
            f"pinned_source_path_unavailable_{method_id}"
            for method_id in COMPARATOR_METHOD_IDS
        ),
        *(
            f"stochastic_seed_missing_{method_id}"
            for method_id in COMPARATOR_METHOD_IDS
        ),
    }
)
_DIRECT_MEASUREMENT_CODES = frozenset(
    {
        "executor_reported_unverified",
        "gpu_measurement_unavailable",
        "linux_proc_process_tree_rss",
        "not_applicable_cpu_only_method",
        "nvidia_smi_measurement_unavailable",
        "nvidia_smi_process_tree_used_memory",
        "rss_measurement_unavailable",
    }
)


def _canonical_terminal_reason(status: str, reason: str | None) -> str | None:
    if status == "completed":
        return None
    if reason in _DIRECT_TERMINAL_REASONS:
        return reason
    return _DIRECT_TERMINAL_REASON_FALLBACK


def _canonical_measurement_code(value: object) -> str:
    if value in _DIRECT_MEASUREMENT_CODES:
        return value
    return "executor_reported_unverified"


DirectAdapter = Callable[..., AdapterOutcome]


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
        raise RunnerContractError("direct execution receipt is not canonical JSON") from error


def _identity_dict(identity: ComparatorRunIdentity) -> dict[str, object]:
    encoded = direct_json_value(identity)
    if not isinstance(encoded, dict):  # pragma: no cover - dataclass invariant
        raise AssertionError("direct run identity must encode as an object")
    return encoded


def _require_nonnegative_number(value: object, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        or (value == 0 and math.copysign(1.0, float(value)) < 0.0)
    ):
        raise RunnerContractError(f"{name} must be a finite nonnegative number")
    return value


def _require_nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise RunnerContractError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class DirectExecutionRequest:
    identity: ComparatorRunIdentity
    method_spec: MethodSpec
    method_input: MethodInput
    timeout_seconds: float
    max_rss_bytes: int
    max_gpu_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ComparatorRunIdentity):
            raise TypeError("identity must be a ComparatorRunIdentity")
        if not isinstance(self.method_spec, MethodSpec):
            raise TypeError("method_spec must be a MethodSpec")
        if not isinstance(self.method_input, MethodInput):
            raise TypeError("method_input must be a MethodInput")
        timeout = _require_nonnegative_number(self.timeout_seconds, "timeout_seconds")
        if timeout == 0:
            raise RunnerContractError("timeout_seconds must be positive")
        _require_nonnegative_integer(self.max_rss_bytes, "max_rss_bytes")
        _require_nonnegative_integer(self.max_gpu_bytes, "max_gpu_bytes")

    def to_dict(self) -> dict[str, object]:
        """Serialize only direct identity and non-content execution bounds."""

        return {
            "identity": _identity_dict(self.identity),
            "method": direct_json_value(self.identity.method),
            "input": {
                "shape": list(self.method_input.shape),
                "dtype": self.method_input.counts.dtype.str,
                "cell_ids": list(self.method_input.obs_ids),
                "gene_ids": list(self.method_input.var_ids),
            },
            "timeout_seconds": self.timeout_seconds,
            "max_rss_bytes": self.max_rss_bytes,
            "max_gpu_bytes": self.max_gpu_bytes,
        }


@dataclass(frozen=True, slots=True)
class DirectLogReceipt:
    stream: Literal["stdout", "stderr"]
    original_byte_count: int
    capture_policy: Literal["discard_content"]
    terminal_reason: str | None

    def __post_init__(self) -> None:
        if self.stream not in {"stdout", "stderr"}:
            raise RunnerContractError("direct log stream is invalid")
        _require_nonnegative_integer(
            self.original_byte_count, "direct log original byte count"
        )
        if self.capture_policy != "discard_content":
            raise RunnerContractError("direct log content must be discarded")
        if self.terminal_reason is not None and (
            not isinstance(self.terminal_reason, str)
            or self.terminal_reason not in _DIRECT_TERMINAL_REASONS
        ):
            raise RunnerContractError("direct log terminal reason is invalid")
        if len(_canonical_bytes(self.to_dict()) + b"\n") > (
            DEVELOPMENT_MAX_LOG_RECEIPT_BYTES
        ):
            raise RunnerContractError("execution stream receipt exceeds its bound")

    def to_dict(self) -> dict[str, object]:
        return {
            "stream": self.stream,
            "original_byte_count": self.original_byte_count,
            "capture_policy": self.capture_policy,
            "terminal_reason": self.terminal_reason,
        }


def _stream_receipt(
    stream: str,
    raw: bytes,
    terminal_reason: str | None,
) -> DirectLogReceipt:
    if stream not in {"stdout", "stderr"} or type(raw) is not bytes:
        raise RunnerContractError("execution stream receipt input is invalid")
    return DirectLogReceipt(
        stream=stream,
        original_byte_count=len(raw),
        capture_policy="discard_content",
        terminal_reason=terminal_reason,
    )


@dataclass(frozen=True, slots=True)
class DirectRunResult:
    run_id: str
    identity: ComparatorRunIdentity
    status: str
    reason: str | None
    runtime_seconds: int | float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    rss_measurement: str
    gpu_measurement: str
    excluded_cell_count: int
    excluded_cell_ids: tuple[str, ...]
    retained_cell_count: int
    retained_cell_ids: tuple[str, ...]
    retained_gene_count: int
    observed_zero_count: int
    stdout: DirectLogReceipt
    stderr: DirectLogReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise RunnerContractError("direct run ID is invalid")
        if not isinstance(self.identity, ComparatorRunIdentity):
            raise TypeError("identity must be a ComparatorRunIdentity")
        if self.status not in _RUN_STATUSES:
            raise RunnerContractError("direct run status is invalid")
        if (self.status == "completed") != (self.reason is None):
            raise RunnerContractError("direct run status and reason disagree")
        if self.reason is not None and (
            not isinstance(self.reason, str)
            or self.reason not in _DIRECT_TERMINAL_REASONS
        ):
            raise RunnerContractError("direct run reason is invalid")
        _require_nonnegative_number(self.runtime_seconds, "direct run runtime")
        for name in (
            "peak_rss_bytes",
            "peak_gpu_bytes",
            "excluded_cell_count",
            "retained_cell_count",
            "retained_gene_count",
            "observed_zero_count",
        ):
            _require_nonnegative_integer(getattr(self, name), name)
        if (
            len(self.excluded_cell_ids) != self.excluded_cell_count
            or len(self.retained_cell_ids) != self.retained_cell_count
            or len(set(self.excluded_cell_ids)) != len(self.excluded_cell_ids)
            or len(set(self.retained_cell_ids)) != len(self.retained_cell_ids)
            or set(self.excluded_cell_ids) & set(self.retained_cell_ids)
        ):
            raise RunnerContractError("direct run cell audit is inconsistent")
        if (
            self.rss_measurement not in _DIRECT_MEASUREMENT_CODES
            or self.gpu_measurement not in _DIRECT_MEASUREMENT_CODES
        ):
            raise RunnerContractError("direct run measurement provenance is invalid")
        if (
            not isinstance(self.stdout, DirectLogReceipt)
            or self.stdout.stream != "stdout"
            or not isinstance(self.stderr, DirectLogReceipt)
            or self.stderr.stream != "stderr"
            or self.stdout.terminal_reason != self.reason
            or self.stderr.terminal_reason != self.reason
        ):
            raise RunnerContractError("direct run log receipts are inconsistent")

    def to_dict(self) -> dict[str, object]:
        encoded = direct_json_value(self)
        if not isinstance(encoded, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError("direct run result must encode as an object")
        return encoded


@dataclass(frozen=True, slots=True)
class DirectMetricRow:
    identity: ComparatorRunIdentity
    metric: str
    value: float | None
    n: int
    status: str
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ComparatorRunIdentity):
            raise TypeError("identity must be a ComparatorRunIdentity")
        if not isinstance(self.metric, str) or not self.metric:
            raise RunnerContractError("direct metric name is invalid")
        _require_nonnegative_integer(self.n, "direct metric denominator")
        if self.value is not None and (
            isinstance(self.value, bool)
            or not isinstance(self.value, float)
            or not math.isfinite(self.value)
            or (self.value == 0.0 and math.copysign(1.0, self.value) < 0.0)
        ):
            raise RunnerContractError("direct metric value is invalid")
        if self.status not in _RUN_STATUSES:
            raise RunnerContractError("direct metric status is invalid")
        if self.value is None:
            if not isinstance(self.reason, str) or not self.reason:
                raise RunnerContractError("unavailable direct metric lacks a reason")
            if self.status == "completed":
                raise RunnerContractError("unavailable direct metric status is invalid")
        elif self.status != "completed" or self.reason is not None:
            raise RunnerContractError("completed direct metric is inconsistent")

    def to_dict(self) -> dict[str, object]:
        encoded = direct_json_value(self)
        if not isinstance(encoded, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError("direct metric must encode as an object")
        return encoded


@dataclass(frozen=True, slots=True)
class DirectPreZeroEvidence:
    applicable: bool
    status: str
    reason: str | None
    shape: tuple[int, int] | None
    dtype: Literal["<f8"] | None
    encoding: Literal["zlib"] | None
    path: str | None
    compressed_byte_count: int

    def __post_init__(self) -> None:
        if type(self.applicable) is not bool:
            raise RunnerContractError("direct p_pre_zero applicability is invalid")
        _require_nonnegative_integer(
            self.compressed_byte_count,
            "direct p_pre_zero compressed byte count",
        )
        if not self.applicable:
            if (
                self.status != "not_applicable"
                or not isinstance(self.reason, str)
                or not self.reason
                or self.shape is not None
                or self.dtype is not None
                or self.encoding is not None
                or self.path is not None
                or self.compressed_byte_count != 0
            ):
                raise RunnerContractError(
                    "non-applicable direct p_pre_zero evidence is inconsistent"
                )
            return
        if self.status == "completed":
            if (
                self.reason is not None
                or type(self.shape) is not tuple
                or len(self.shape) != 2
                or any(type(value) is not int or value <= 0 for value in self.shape)
                or self.dtype != "<f8"
                or self.encoding != "zlib"
                or not isinstance(self.path, str)
                or not self.path
                or self.compressed_byte_count <= 0
            ):
                raise RunnerContractError(
                    "completed direct p_pre_zero evidence is inconsistent"
                )
        elif (
            self.status not in _RUN_STATUSES
            or self.status == "completed"
            or not isinstance(self.reason, str)
            or not self.reason
            or self.shape is not None
            or self.dtype is not None
            or self.encoding is not None
            or self.path is not None
            or self.compressed_byte_count != 0
        ):
            raise RunnerContractError(
                "direct p_pre_zero evidence status is inconsistent"
            )

    def to_dict(self) -> dict[str, object]:
        encoded = direct_json_value(self)
        if not isinstance(encoded, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError("direct p_pre_zero evidence must encode as an object")
        return encoded

    def reopen(self, repository: Path) -> np.ndarray | None:
        """Reopen a stored probability matrix using only its direct receipt."""

        if not self.applicable or self.status != "completed":
            return None
        if not isinstance(repository, Path):
            raise TypeError("repository must be a pathlib.Path")
        root = repository.resolve(strict=True)
        assert self.path is not None and self.shape is not None
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RunnerContractError("direct p_pre_zero path is unsafe")
        path = root.joinpath(*relative.parts)
        current = root
        for component in relative.parts:
            current = current / component
            if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
                raise RunnerContractError("direct p_pre_zero path is not owned")
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RunnerContractError(
                "direct p_pre_zero path is unavailable"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RunnerContractError("direct p_pre_zero path must be a regular file")
        if metadata.st_uid != os.getuid():
            raise RunnerContractError("direct p_pre_zero file owner differs")
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise RunnerContractError("direct p_pre_zero path is not owned")
        try:
            compressed = path.read_bytes()
        except OSError as error:
            raise RunnerContractError(
                "direct p_pre_zero data is unavailable"
            ) from error
        if len(compressed) != self.compressed_byte_count:
            raise RunnerContractError("direct p_pre_zero byte count differs")
        expected_bytes = self.shape[0] * self.shape[1] * 8
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed, expected_bytes + 1)
            raw += decoder.flush(max(1, expected_bytes + 1 - len(raw)))
        except zlib.error as error:
            raise RunnerContractError("direct p_pre_zero data is invalid") from error
        if (
            len(raw) != expected_bytes
            or not decoder.eof
            or decoder.unconsumed_tail
            or decoder.unused_data
        ):
            raise RunnerContractError(
                "direct p_pre_zero decoded shape or dtype differs"
            )
        matrix = np.frombuffer(raw, dtype="<f8").reshape(self.shape).copy()
        if not np.isfinite(matrix).all() or bool(
            ((matrix < 0.0) | (matrix > 1.0)).any()
        ):
            raise RunnerContractError(
                "direct p_pre_zero data must contain probabilities"
            )
        return matrix


@dataclass(frozen=True, slots=True)
class DirectEvaluatedAttempt:
    run: DirectRunResult
    metrics: tuple[DirectMetricRow, ...]
    native_output: np.ndarray | None
    native_output_scale: str | None
    evaluator_output: np.ndarray | None
    p_pre_zero_evidence: DirectPreZeroEvidence

    def __post_init__(self) -> None:
        validate_direct_evidence_semantics(
            self.run,
            self.metrics,
            self.p_pre_zero_evidence,
        )
        matrices = (self.native_output, self.evaluator_output)
        if self.run.status == "completed":
            if (
                any(type(value) is not np.ndarray for value in matrices)
                or not isinstance(self.native_output_scale, str)
                or not self.native_output_scale
                or self.native_output_scale != self.run.identity.method.output_scale
            ):
                raise RunnerContractError(
                    "completed direct attempt matrix/output scale is inconsistent"
                )
            assert self.native_output is not None and self.evaluator_output is not None
            if (
                self.native_output.shape != self.evaluator_output.shape
                or self.native_output.ndim != 2
                or not np.isfinite(self.native_output).all()
                or not np.isfinite(self.evaluator_output).all()
            ):
                raise RunnerContractError("direct attempt output matrices differ")
        elif (
            self.native_output is not None
            or self.native_output_scale is not None
            or self.evaluator_output is not None
        ):
            raise RunnerContractError(
                "noncompleted direct attempt retains an output matrix"
            )
        _executor_receipt(self)

    def to_dict(self) -> dict[str, object]:
        """Return the checkpoint-safe record without in-memory matrices."""

        return {
            "run": self.run.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "p_pre_zero_evidence": self.p_pre_zero_evidence.to_dict(),
        }


def _executor_receipt_for_run(run: DirectRunResult) -> bytes:
    if not isinstance(run, DirectRunResult):
        raise TypeError("run must be a DirectRunResult")
    value = {
        "schema": "maskimpute-development-executor-receipt-v1",
        "run_id": run.run_id,
        "status": run.status,
        "reason": run.reason,
        "runtime_seconds": run.runtime_seconds,
        "peak_rss_bytes": run.peak_rss_bytes,
        "peak_gpu_bytes": run.peak_gpu_bytes,
        "rss_measurement": run.rss_measurement,
        "gpu_measurement": run.gpu_measurement,
        "stdout": run.stdout.to_dict(),
        "stderr": run.stderr.to_dict(),
    }
    encoded = _canonical_bytes(value) + b"\n"
    if len(encoded) > DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES:
        raise RunnerContractError("executor receipt exceeds its bound")
    return encoded


def _executor_receipt(attempt: DirectEvaluatedAttempt) -> bytes:
    if not isinstance(attempt, DirectEvaluatedAttempt):
        raise TypeError("attempt must be a DirectEvaluatedAttempt")
    return _executor_receipt_for_run(attempt.run)


DIRECT_RECONSTRUCTION_METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
)


def validate_direct_evidence_semantics(
    run: DirectRunResult,
    metrics: tuple[DirectMetricRow, ...],
    evidence: DirectPreZeroEvidence,
) -> None:
    """Close run, metric, and pre-zero evidence cross-field semantics."""

    if not isinstance(run, DirectRunResult):
        raise TypeError("run must be a DirectRunResult")
    if type(metrics) is not tuple or not all(
        isinstance(metric, DirectMetricRow) for metric in metrics
    ):
        raise TypeError("metrics must contain DirectMetricRow values")
    if not isinstance(evidence, DirectPreZeroEvidence):
        raise TypeError("evidence must be a DirectPreZeroEvidence")
    if run.run_id != direct_run_id(run.identity):
        raise RunnerContractError("direct attempt run identity differs")
    if tuple(metric.metric for metric in metrics) != DIRECT_RECONSTRUCTION_METRICS:
        raise RunnerContractError("direct attempt metric denominator/order differs")
    if any(not direct_equal(metric.identity, run.identity) for metric in metrics):
        raise RunnerContractError("direct attempt metric identity differs")
    if run.status == "completed":
        if any(metric.status not in {"completed", "unavailable"} for metric in metrics):
            raise RunnerContractError("completed direct attempt metric status differs")
    elif any(
        metric.value is not None
        or metric.n != 0
        or metric.status != run.status
        or metric.reason != run.reason
        for metric in metrics
    ):
        raise RunnerContractError(
            "noncompleted direct attempt metric status/reason differs"
        )
    if run.identity.method.method_id == "maskimpute":
        if (
            not evidence.applicable
            or evidence.status != run.status
            or evidence.reason != run.reason
        ):
            raise RunnerContractError("direct p_pre_zero applicability/receipt differs")
    else:
        expected = DirectPreZeroEvidence(
            applicable=False,
            status="not_applicable",
            reason="method_does_not_emit_p_pre_zero",
            shape=None,
            dtype=None,
            encoding=None,
            path=None,
            compressed_byte_count=0,
        )
        if not direct_equal(evidence, expected):
            raise RunnerContractError("direct p_pre_zero applicability/receipt differs")


def _request_matches_prepared(
    request: DirectExecutionRequest,
    prepared: PreparedDataset,
) -> bool:
    identity = request.identity
    descriptor = describe_prepared_input(prepared)
    return (
        _direct_equal(request.method_input, prepared.method_input)
        and _direct_equal(identity.dataset_id, descriptor.dataset_id)
        and _direct_equal(identity.mechanism, descriptor.mechanism)
        and _direct_equal(identity.biological_id, prepared.binding.biological_id)
        and _direct_equal(identity.technical_view, descriptor.technical_view)
        and _direct_equal(identity.mask_seed, descriptor.mask_seed)
        and _direct_equal(
            identity.draw_index,
            int(prepared.evaluator_dataset.obs["draw"].tolist()[0]),
        )
        and _direct_equal(request.method_input.shape, descriptor.shape)
        and _direct_equal(request.method_input.obs_ids, descriptor.cell_ids)
        and _direct_equal(request.method_input.var_ids, descriptor.gene_ids)
    )


def _row_payload(row: ComparatorConfiguration) -> tuple[tuple[str, object], ...]:
    try:
        return freeze_direct_mapping(row.payload)
    except ValueError as error:
        raise RunnerContractError("direct request authority row differs") from error


def _validate_direct_identity(
    identity: ComparatorRunIdentity,
    method_spec: MethodSpec,
    row: ComparatorConfiguration,
) -> None:
    if not _direct_equal(identity.workflow_schema, "maskimpute-fair-comparator-run-v1"):
        raise RunnerContractError("direct request workflow schema differs")
    if not _direct_equal(identity.configuration_kind, "comparator_tuning"):
        raise RunnerContractError("direct request is not comparator tuning")
    if (
        isinstance(identity.ordinal, bool)
        or type(identity.ordinal) is not int
        or identity.ordinal <= 0
    ):
        raise RunnerContractError("direct request ordinal is invalid")
    if (
        isinstance(identity.mask_seed, bool)
        or type(identity.mask_seed) is not int
        or identity.mask_seed < 0
    ):
        raise RunnerContractError("direct request mask seed is invalid")
    if (
        isinstance(identity.draw_index, bool)
        or type(identity.draw_index) is not int
        or identity.draw_index <= 0
    ):
        raise RunnerContractError("direct request draw index is invalid")
    if _direct_equal(identity.configuration_id, "registry-default"):
        raise RunnerContractError("direct comparator cannot use registry-default")
    if (
        not _direct_equal(identity.method, comparator_method_binding(method_spec))
        or identity.method.method_id != method_spec.id
    ):
        raise RunnerContractError("direct request method projection differs")
    if (
        not _direct_equal(row.method_id, identity.method.method_id)
        or not _direct_equal(row.configuration_id, identity.configuration_id)
        or not _direct_equal(_row_payload(row), identity.configuration_payload)
    ):
        raise RunnerContractError("direct request authority row differs")
    try:
        decoded = row.decode()
        encoded = encode_comparator_configuration(decoded)
    except (ComparatorTuningError, TypeError, ValueError) as error:
        raise RunnerContractError("direct request typed payload differs") from error
    if not _direct_equal(encoded, row.payload):
        raise RunnerContractError("direct request typed payload differs")


def create_direct_request(
    entry: DirectPlanEntry,
    prepared: PreparedDataset,
    descriptor: PreparedInputDescriptor,
    method_spec: MethodSpec,
    authority: ComparatorTuningAuthority,
    *,
    timeout_seconds: int | float,
) -> DirectExecutionRequest:
    """Create one request after complete direct equality checks."""

    if not isinstance(entry, DirectPlanEntry):
        raise TypeError("entry must be a DirectPlanEntry")
    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    if not isinstance(descriptor, PreparedInputDescriptor):
        raise TypeError("descriptor must be a PreparedInputDescriptor")
    if not isinstance(method_spec, MethodSpec):
        raise TypeError("method_spec must be a MethodSpec")
    _require_complete_authority(authority)
    timeout = _require_nonnegative_number(timeout_seconds, "timeout_seconds")
    if timeout == 0:
        raise RunnerContractError("timeout_seconds must be positive")
    if (
        not _direct_equal(entry.preflight_status, "planned")
        or entry.preflight_reason is not None
    ):
        raise RunnerContractError("direct request entry is not executable")
    if not _direct_equal(entry.requires_count_score, False) or not _direct_equal(
        entry.requires_calibration, False
    ):
        raise RunnerContractError("direct comparator has inapplicable score authority")
    if not _direct_equal(entry.run_id, direct_run_id(entry.identity)):
        raise RunnerContractError("direct request run ID differs")
    if not _direct_equal(describe_prepared_input(prepared), descriptor):
        raise RunnerContractError("direct prepared input descriptor differs")
    _resolve_identity_row(entry.identity, method_spec, authority)
    model_seed = entry.identity.model_seed
    if method_spec.stochastic:
        if (
            isinstance(model_seed, bool)
            or type(model_seed) is not int
            or model_seed not in {42, 43, 44}
        ):
            raise RunnerContractError(
                "direct stochastic comparator model seed is invalid"
            )
    elif model_seed is not None:
        raise RunnerContractError(
            "direct deterministic comparator model seed must be null"
        )
    if not _request_matches_prepared(
        DirectExecutionRequest(
            identity=entry.identity,
            method_spec=method_spec,
            method_input=prepared.method_input,
            timeout_seconds=float(timeout),
            max_rss_bytes=int(method_spec.resources.max_rss_gib * 1024**3),
            max_gpu_bytes=int(method_spec.resources.max_gpu_gib * 1024**3),
        ),
        prepared,
    ):
        raise RunnerContractError("direct plan identity differs from prepared input")
    normalized_timeout = float(timeout)
    if normalized_timeout > float(method_spec.resources.timeout_seconds):
        raise RunnerContractError("direct request timeout exceeds method limit")
    return DirectExecutionRequest(
        identity=entry.identity,
        method_spec=method_spec,
        method_input=prepared.method_input,
        timeout_seconds=normalized_timeout,
        max_rss_bytes=int(method_spec.resources.max_rss_gib * 1024**3),
        max_gpu_bytes=int(method_spec.resources.max_gpu_gib * 1024**3),
    )


def _resolve_row(
    request: DirectExecutionRequest,
    authority: ComparatorTuningAuthority,
) -> ComparatorConfiguration:
    _require_complete_authority(authority)
    return _resolve_identity_row(request.identity, request.method_spec, authority)


def _require_complete_authority(authority: ComparatorTuningAuthority) -> None:
    if not isinstance(authority, ComparatorTuningAuthority):
        raise TypeError("authority must be a ComparatorTuningAuthority")
    try:
        validate_comparator_tuning_authority(authority)
    except ComparatorTuningError as error:
        raise RunnerContractError(
            "direct comparator authority is not the complete fixed authority"
        ) from error


def _resolve_identity_row(
    identity: ComparatorRunIdentity,
    method_spec: MethodSpec,
    authority: ComparatorTuningAuthority,
) -> ComparatorConfiguration:
    if not _direct_equal(identity.authority_revision, authority.authority_revision):
        raise RunnerContractError("direct request authority revision differs")
    matches = tuple(
        row
        for row in authority.configurations
        if _direct_equal(row.method_id, identity.method.method_id)
        and _direct_equal(row.configuration_id, identity.configuration_id)
        and _direct_equal(_row_payload(row), identity.configuration_payload)
    )
    if len(matches) != 1:
        raise RunnerContractError(
            "direct request does not resolve to exactly one authority row"
        )
    row = matches[0]
    _validate_direct_identity(identity, method_spec, row)
    return row


def _outcome_with_status(
    status: str,
    reason: str,
    outcome: AdapterOutcome | None = None,
) -> AdapterOutcome:
    values = {
        "stdout": b"" if outcome is None else outcome.stdout,
        "stderr": b"" if outcome is None else outcome.stderr,
        "runtime_seconds": 0 if outcome is None else outcome.runtime_seconds,
        "peak_rss_bytes": 0 if outcome is None else outcome.peak_rss_bytes,
        "peak_gpu_bytes": 0 if outcome is None else outcome.peak_gpu_bytes,
        "rss_measurement": (
            "executor_reported_unverified"
            if outcome is None
            else outcome.rss_measurement
        ),
        "gpu_measurement": (
            "executor_reported_unverified"
            if outcome is None
            else outcome.gpu_measurement
        ),
    }
    factory = {
        "failed": AdapterOutcome.failed,
        "timeout": AdapterOutcome.timeout,
        "resource_exceeded": AdapterOutcome.resource_exceeded,
        "unavailable": AdapterOutcome.unavailable,
        "infrastructure_error": AdapterOutcome.infrastructure_error,
        "blocked_authority": AdapterOutcome.blocked_authority,
        "budget_exhausted": AdapterOutcome.budget_exhausted,
    }[status]
    return factory(reason, **values)


def _dispatch(
    request: DirectExecutionRequest,
    row: ComparatorConfiguration,
    adapters: Mapping[str, DirectAdapter],
) -> AdapterOutcome:
    from .methods import AdapterUnavailableError

    if not isinstance(adapters, Mapping):
        raise TypeError("adapters must be a mapping")
    adapter = adapters.get(request.identity.method.method_id)
    if adapter is None or not callable(adapter):
        return _outcome_with_status("unavailable", "adapter_not_registered")
    config = row.decode()
    outcome: AdapterOutcome | None = None
    try:
        try:
            outcome = adapter(
                request.method_spec,
                request.method_input,
                seed=request.identity.model_seed,
                config=config,
            )
            if not isinstance(outcome, AdapterOutcome):
                return _outcome_with_status(
                    "infrastructure_error",
                    "adapter_returned_noncanonical_outcome",
                )
        except AdapterUnavailableError as error:
            outcome = AdapterOutcome.unavailable(
                error.reason_code,
                stdout=error.stdout,
                stderr=error.stderr,
            )
        except TimeoutError:
            outcome = AdapterOutcome.timeout("timeout")
        except MemoryError:
            outcome = AdapterOutcome.resource_exceeded("memory_limit_exceeded")
        except RuntimeEnvironmentError:
            outcome = AdapterOutcome.infrastructure_error("runtime_environment_invalid")
        except Exception:
            outcome = AdapterOutcome.failed("adapter_exception")
    finally:
        try:
            unchanged = _direct_equal(
                encode_comparator_configuration(config),
                row.payload,
            )
        except (TypeError, ValueError):
            unchanged = False
        if not unchanged:
            outcome = _outcome_with_status(
                "failed",
                "comparator_payload_changed_during_adapter_attempt",
                outcome,
            )
    if outcome is None:  # pragma: no cover - all branches assign an outcome
        raise AssertionError("direct comparator dispatch produced no outcome")
    if outcome.peak_rss_bytes > request.max_rss_bytes:
        return _outcome_with_status("resource_exceeded", "peak_rss_exceeded", outcome)
    if outcome.peak_gpu_bytes > request.max_gpu_bytes:
        return _outcome_with_status("resource_exceeded", "peak_gpu_exceeded", outcome)
    return outcome


_RECONSTRUCTION_METRICS = DIRECT_RECONSTRUCTION_METRICS


def _dense_matrix(value: object, name: str) -> np.ndarray:
    from scipy import sparse

    try:
        array = value.toarray() if sparse.issparse(value) else np.asarray(value)
        dense = np.array(array, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise RunnerContractError(f"{name} cannot be represented as float64") from error
    if dense.ndim != 2 or not np.isfinite(dense).all() or bool((dense < 0).any()):
        raise RunnerContractError(
            f"{name} must be finite nonnegative two-dimensional data"
        )
    return dense


def _evaluator_targets(
    prepared: PreparedDataset,
) -> tuple[np.ndarray, np.ndarray | None, str, np.ndarray | None]:
    observed = count_equivalent_to_log2_cp10k(
        _dense_matrix(prepared.evaluator_dataset.X, "observed counts")
    )
    truth_kind = prepared.evaluator_dataset.uns.get("truth_kind")
    if not isinstance(truth_kind, str):
        raise RunnerContractError("evaluator dataset truth_kind is invalid")
    if truth_kind == "orthogonal_only":
        truth = None
    else:
        truth_layer = prepared.evaluator_dataset.uns.get("primary_truth_layer")
        if (
            not isinstance(truth_layer, str)
            or truth_layer not in prepared.evaluator_dataset.layers
        ):
            raise RunnerContractError("evaluator primary truth layer is unavailable")
        truth = count_equivalent_to_log2_cp10k(
            _dense_matrix(
                prepared.evaluator_dataset.layers[truth_layer],
                "primary evaluator truth",
            )
        )
    marker_columns = tuple(
        name
        for name in prepared.evaluator_dataset.var.columns
        if isinstance(name, str) and name.casefold().startswith("marker")
    )
    masks = tuple(
        np.asarray(
            prepared.evaluator_dataset.var[name].to_numpy(copy=True),
            dtype=bool,
        )
        for name in marker_columns
        if prepared.evaluator_dataset.var[name].dtype.kind == "b"
    )
    marker_mask = None if not masks else np.logical_or.reduce(masks)
    return observed, truth, truth_kind, marker_mask


def _convert_output(
    request: DirectExecutionRequest,
    execution: DirectAdapterExecution,
) -> tuple[np.ndarray, str, np.ndarray]:
    if not isinstance(execution, DirectAdapterExecution):
        raise TypeError("direct evaluation requires a DirectAdapterExecution")
    snapshot = execution.output
    identity = request.identity
    if (
        snapshot.method_id != identity.method.method_id
        or snapshot.output_scale != request.method_spec.output_scale
        or snapshot.obs_ids != request.method_input.obs_ids
        or snapshot.var_ids != request.method_input.var_ids
        or snapshot.shape != request.method_input.shape
    ):
        raise ValueError("adapter output differs from direct request")
    try:
        native = np.array(
            snapshot.matrix,
            dtype=np.float64,
            copy=True,
            order="C",
            subok=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("adapter output cannot be represented as float64") from error
    if (
        native.shape != request.method_input.shape
        or not np.isfinite(native).all()
        or bool((native < 0).any())
    ):
        raise ValueError("adapter output matrix is invalid")
    converters = {
        **CORE_EVALUATOR_COUNT_CONVERTERS,
        **RECENT_EVALUATOR_COUNT_CONVERTERS,
        **LEGACY_EVALUATOR_COUNT_CONVERTERS,
    }
    scales = {
        **CORE_EVALUATOR_NATIVE_SCALES,
        **RECENT_EVALUATOR_NATIVE_SCALES,
        **LEGACY_EVALUATOR_NATIVE_SCALES,
    }
    method_id = identity.method.method_id
    if (
        method_id not in COMPARATOR_METHOD_IDS
        or scales.get(method_id) != snapshot.output_scale
    ):
        raise ValueError("adapter output scale is invalid")
    counts = converters[method_id](request.method_input, native)
    evaluator = np.array(
        count_equivalent_to_log2_cp10k(counts),
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    if (
        evaluator.shape != request.method_input.shape
        or not np.isfinite(evaluator).all()
        or bool((evaluator < 0).any())
    ):
        raise ValueError("common-scale adapter output is invalid")
    return native, snapshot.output_scale, evaluator


def _evaluate(
    request: DirectExecutionRequest,
    prepared: PreparedDataset,
    outcome: AdapterOutcome,
) -> DirectEvaluatedAttempt:
    from .methods import AdapterUnavailableError
    from .metrics import reconstruction_metrics

    status = outcome.status
    reason = outcome.reason
    native: np.ndarray | None = None
    native_scale: str | None = None
    evaluator: np.ndarray | None = None
    if status == "completed":
        assert outcome.execution is not None
        try:
            native, native_scale, evaluator = _convert_output(
                request, outcome.execution
            )
        except AdapterUnavailableError:
            status = "unavailable"
            reason = "evaluator_conversion_unavailable"
        except (TypeError, ValueError, OverflowError):
            status = "unavailable"
            reason = "evaluator_conversion_invalid"
    reason = _canonical_terminal_reason(status, reason)
    rss_measurement = _canonical_measurement_code(outcome.rss_measurement)
    gpu_measurement = _canonical_measurement_code(outcome.gpu_measurement)
    if evaluator is None:
        if reason is None:  # pragma: no cover - guarded by outcome contract
            raise AssertionError("noncompleted direct attempt lacks a reason")
        metric_rows = tuple(
            DirectMetricRow(
                identity=request.identity,
                metric=name,
                value=None,
                n=0,
                status=status,
                reason=reason,
            )
            for name in _RECONSTRUCTION_METRICS
        )
    else:
        observed, truth, truth_kind, marker_mask = _evaluator_targets(prepared)
        metrics = reconstruction_metrics(
            evaluator,
            observed,
            truth,
            marker_genes=marker_mask,
            truth_kind=truth_kind,
        )
        metric_rows = tuple(
            DirectMetricRow(
                identity=request.identity,
                metric=name,
                value=(
                    None if metrics[name].value is None else float(metrics[name].value)
                ),
                n=int(metrics[name].n),
                status=("unavailable" if metrics[name].value is None else "completed"),
                reason=metrics[name].reason,
            )
            for name in DIRECT_RECONSTRUCTION_METRICS
        )
    stdout = _stream_receipt("stdout", outcome.stdout, reason)
    stderr = _stream_receipt("stderr", outcome.stderr, reason)
    run = DirectRunResult(
        run_id=direct_run_id(request.identity),
        identity=request.identity,
        status=status,
        reason=reason,
        runtime_seconds=outcome.runtime_seconds,
        peak_rss_bytes=outcome.peak_rss_bytes,
        peak_gpu_bytes=outcome.peak_gpu_bytes,
        rss_measurement=rss_measurement,
        gpu_measurement=gpu_measurement,
        excluded_cell_count=prepared.audit.excluded_cell_count,
        excluded_cell_ids=prepared.audit.excluded_cell_ids,
        retained_cell_count=prepared.audit.retained_cell_count,
        retained_cell_ids=prepared.audit.retained_cell_ids,
        retained_gene_count=prepared.method_input.shape[1],
        observed_zero_count=int((prepared.method_input.counts == 0).sum()),
        stdout=stdout,
        stderr=stderr,
    )
    return DirectEvaluatedAttempt(
        run=run,
        metrics=metric_rows,
        native_output=native,
        native_output_scale=native_scale,
        evaluator_output=evaluator,
        p_pre_zero_evidence=DirectPreZeroEvidence(
            applicable=False,
            status="not_applicable",
            reason="method_does_not_emit_p_pre_zero",
            shape=None,
            dtype=None,
            encoding=None,
            path=None,
            compressed_byte_count=0,
        ),
    )


def validate_direct_request(
    request: DirectExecutionRequest,
    prepared: PreparedDataset,
    authority: ComparatorTuningAuthority,
) -> ComparatorConfiguration:
    """Validate all parent-owned request, dataset, and authority bindings."""

    if not isinstance(request, DirectExecutionRequest):
        raise TypeError("request must be a DirectExecutionRequest")
    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    expected_rss = int(request.method_spec.resources.max_rss_gib * 1024**3)
    expected_gpu = int(request.method_spec.resources.max_gpu_gib * 1024**3)
    if (
        request.max_rss_bytes != expected_rss
        or request.max_gpu_bytes != expected_gpu
        or request.timeout_seconds > request.method_spec.resources.timeout_seconds
    ):
        raise RunnerContractError("direct request resource limits differ")
    if not _request_matches_prepared(request, prepared):
        raise RunnerContractError("direct request differs from prepared input")
    return _resolve_row(request, authority)


def dispatch_direct_request(
    request: DirectExecutionRequest,
    authority: ComparatorTuningAuthority,
    adapters: Mapping[str, DirectAdapter],
) -> AdapterOutcome:
    """Resolve and dispatch inside a child without legacy request construction."""

    if not isinstance(request, DirectExecutionRequest):
        raise TypeError("request must be a DirectExecutionRequest")
    return _dispatch(request, _resolve_row(request, authority), adapters)


def evaluate_direct_outcome(
    request: DirectExecutionRequest,
    prepared: PreparedDataset,
    authority: ComparatorTuningAuthority,
    outcome: AdapterOutcome,
) -> DirectEvaluatedAttempt:
    """Revalidate direct authority after an attempt, then compute metrics."""

    validate_direct_request(request, prepared, authority)
    if not isinstance(outcome, AdapterOutcome):
        raise TypeError("outcome must be an AdapterOutcome")
    return _evaluate(request, prepared, outcome)


def execute_direct_request(
    request: DirectExecutionRequest,
    prepared: PreparedDataset,
    authority: ComparatorTuningAuthority,
    adapters: Mapping[str, DirectAdapter],
) -> DirectEvaluatedAttempt:
    """Resolve, dispatch, revalidate, and evaluate one direct comparator request."""

    validate_direct_request(request, prepared, authority)
    outcome = dispatch_direct_request(request, authority, adapters)
    return evaluate_direct_outcome(request, prepared, authority, outcome)


__all__ = [
    "DirectEvaluatedAttempt",
    "DirectExecutionRequest",
    "DirectLogReceipt",
    "DirectMetricRow",
    "DirectPreZeroEvidence",
    "DirectRunResult",
    "create_direct_request",
    "dispatch_direct_request",
    "evaluate_direct_outcome",
    "execute_direct_request",
    "validate_direct_request",
]
