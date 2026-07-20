"""Hash-bound resumable downstream evidence over persisted runner outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tempfile
from types import MappingProxyType
from typing import Any
import zlib

import anndata as ad
import numpy as np
from scipy import sparse

from .downstream_evaluation import (
    DOWNSTREAM_ENDPOINT_NAMES,
    EndpointRecord,
    EvaluatorTargets,
    MethodOutput,
    evaluate_downstream_endpoints,
    evaluate_trajectory_endpoint,
    evaluator_targets_from_dataset,
    terminal_downstream_endpoints,
)
from .protocol import canonical_sha256
from .runner import AuthorizedConfiguration
from .schema import benchmark_dataset_sha256, validate_benchmark_dataset


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_OUTPUT_ENCODING = "zlib_raw_f64_v1"
_TRAJECTORY_ENDPOINT_NAMES = ("trajectory_pseudotime_rank_loss",)
_ENDPOINT_VALUE_RANGES = MappingProxyType(
    {
        "marker_rank_loss": (0.0, 1.0),
        "clustering_ari_loss": (0.0, 2.0),
        "clustering_nmi_loss": (0.0, 1.0),
        "positive_de_marker_recall": (0.0, 1.0),
        "positive_de_false_discovery_rate": (0.0, 1.0),
        "heldout_gene_profile_rank_loss": (0.0, 2.0),
        "heldout_cell_profile_rank_loss": (0.0, 2.0),
        "trajectory_pseudotime_rank_loss": (0.0, 2.0),
    }
)
_CLUSTERING_PROCEDURE_PREFIX = (
    "log2_cp10k_plus_1_full_svd_pca_kmeans_fixed_k_grid=2:10_"
    "minimum_davies_bouldin_seed=20260716_n_init=20"
)
_ENDPOINT_PROCEDURES = MappingProxyType(
    {
        "marker_rank_loss": frozenset(
            {
                "group_macro_mean_normalized_true_marker_rank_log2_cp10k_plus_1",
            }
        ),
        "clustering_ari_loss": frozenset({_CLUSTERING_PROCEDURE_PREFIX}),
        "clustering_nmi_loss": frozenset({_CLUSTERING_PROCEDURE_PREFIX}),
        "positive_de_marker_recall": frozenset(
            {"one_sided_welch_log2_cp10k_plus_1_global_bh"}
        ),
        "positive_de_false_discovery_rate": frozenset(
            {"one_sided_welch_log2_cp10k_plus_1_global_bh"}
        ),
        "heldout_gene_profile_rank_loss": frozenset(
            {"mean_profile_spearman_log2_cp10k_plus_1_independent_count_split"}
        ),
        "heldout_cell_profile_rank_loss": frozenset(
            {"mean_profile_spearman_log2_cp10k_plus_1_independent_count_split"}
        ),
        "trajectory_pseudotime_rank_loss": frozenset(
            {
                "root_oriented_multiscale_diffusion_log2_cp10k_plus_1_full_svd_"
                "blockwise_exact_knn=floor_sqrt_n_capped_15_sparse_eigsh_modes=15"
            }
        ),
    }
)
_TERMINAL_PROCEDURES = frozenset(
    {
        "terminal_expected_numeric_failure",
        "terminal_upstream_run_not_completed",
    }
)
_SOURCE_KINDS = frozenset({"development", "final"})
_EVIDENCE_SCOPES = frozenset(
    {
        "all",
        "selection_primary",
        "supplementary_nonselection",
        "supplementary_trajectory",
    }
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
_DEVELOPMENT_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "input_hashes",
        "planned_run_count",
        "status",
        "evaluation_scope",
        "comparator_selection_status",
        "selection_complete",
        "selection_blockers",
        "records",
        "budget",
        "checkpoint_sha256",
    }
)
_FINAL_MANIFEST_FIELDS = frozenset(
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
_TRAJECTORY_MANIFEST_FIELDS = _FINAL_MANIFEST_FIELDS | frozenset(
    {"scope", "plan_entries", "configurations", "model_seed_policy"}
)
_EVALUATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "round_id",
        "state",
        "evaluated_at",
        "execution_claim_id",
        "result_manifest",
        "result_manifest_sha256",
        "seed_manifest_sha256",
        "round_path",
        "round_token",
        "repository_instance_id",
        "worktree_path_sha256",
        "git_common_dir_device",
        "git_common_dir_inode",
        "study_state_root_device",
        "study_state_root_inode",
        "registry_dir_device",
        "registry_dir_inode",
        "method_commit",
        "config_sha256",
        "protocol_sha256",
        "environment_sha256",
        "operational_artifact_roots_sha256",
    }
)
_FINAL_RESULT_MANIFEST_FIELDS = frozenset(
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
_TRAJECTORY_EVIDENCE_FIELDS = frozenset(
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
)
_TRAJECTORY_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "input_hashes",
        "entries",
        "configurations",
        "model_seed_policy",
        "plan_sha256",
    }
)
_TRAJECTORY_PLAN_INPUT_FIELDS = frozenset(
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
_TRAJECTORY_DATASET_EVIDENCE_FIELDS = frozenset(
    {
        "binding",
        "dataset_path",
        "dataset_file_sha256",
        "dataset_sha256",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_payload_sha256",
    }
)
_TRAJECTORY_AUTHORITY_EVIDENCE_FIELDS = frozenset(
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
)
_TRAJECTORY_MANIFEST_EVIDENCE_FIELDS = frozenset(
    {"path", "file_sha256", "payload_sha256"}
)
_SCALING_EVIDENCE_FIELDS = frozenset(
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
)
_SCALING_PLAN_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "input_hashes",
        "entries",
        "configurations",
        "plan_sha256",
    }
)
_SCALING_CHECKPOINT_PAYLOAD_FIELDS = frozenset(
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
)
_SCALING_CHECKPOINT_PATH = re.compile(r"results/scaling/checkpoints/([0-9]{8})\.json\Z")
_FINAL_EXECUTION_VALIDATION_FIELDS = frozenset(
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
_TRAJECTORY_EXECUTION_VALIDATION_FIELDS = frozenset(
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
)
_RAW_RUN_FIELDS = frozenset(
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
        "configuration_kind",
        "requires_count_score",
        "requires_calibration",
        "method_input_sha256",
        "dataset_qc_policy_sha256",
        "excluded_cell_count",
        "excluded_cell_ids_sha256",
        "retained_cell_count",
        "retained_cell_ids_sha256",
        "retained_gene_count",
        "observed_zero_count",
        "status",
        "reason",
        "runtime_seconds",
        "peak_rss_bytes",
        "peak_gpu_bytes",
        "rss_measurement",
        "gpu_measurement",
        "calibration_artifact_sha256",
        "calibration_context_sha256",
        "calibration_training_manifest_sha256s",
        "calibration_held_out_manifest_sha256s",
        "calibration_fold_calibrator_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "native_output_sha256",
        "evaluator_output_sha256",
    }
)
_STORED_OUTPUT_FIELDS = frozenset(
    {
        "stdout_path",
        "stdout_file_sha256",
        "stderr_path",
        "stderr_file_sha256",
        "native_output_path",
        "native_output_file_sha256",
        "native_output_shape",
        "native_output_dtype",
        "native_output_scale",
        "evaluator_output_path",
        "evaluator_output_file_sha256",
        "evaluator_output_shape",
        "evaluator_output_dtype",
        "evaluator_scale",
    }
)
_DEVELOPMENT_RUN_FIELDS = _RAW_RUN_FIELDS | _STORED_OUTPUT_FIELDS
_FINAL_RUN_FIELDS = _DEVELOPMENT_RUN_FIELDS | frozenset(
    {
        "native_output_retention",
        "evaluator_output_encoding",
        "evaluator_output_uncompressed_nbytes",
        "evaluator_output_uncompressed_sha256",
    }
)
_METRIC_FIELDS = frozenset(
    {
        "mechanism",
        "biological_id",
        "technical_view",
        "dataset_id",
        "method",
        "model_seed",
        "configuration_id",
        "configuration_sha256",
        "metric",
        "value",
        "n",
        "status",
        "reason",
    }
)
_PREZERO_EVIDENCE_FIELDS = frozenset(
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
)
_PREZERO_IDENTITY_FIELDS = frozenset(
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
_PREZERO_MATRIX_FIELDS = frozenset(
    {"shape", "dtype", "content_sha256", "semantic_sha256"}
)
_PREZERO_STORAGE_FIELDS = frozenset(
    {
        "encoding",
        "compression_level",
        "path",
        "compressed_sha256",
        "compressed_nbytes",
        "uncompressed_sha256",
        "uncompressed_nbytes",
    }
)
_PREZERO_POLICY_FIELDS = frozenset(
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
_FINAL_EXECUTION_REQUEST_FIELDS = frozenset(
    {
        "calibration_usage",
        "configuration_sha256",
        "count_score_manifest_sha256",
        "dataset_id",
        "execution_authority_sha256",
        "method_input_sha256",
        "model_seed",
        "request_sha256",
        "retained_calibration_sha256",
    }
)
_FINAL_STORAGE_POLICY = MappingProxyType(
    {
        "evaluator_output_encoding": _FINAL_OUTPUT_ENCODING,
        "evaluator_output_compression_level": 6,
        "native_output_retention": "omitted_redundant_final_output",
        "p_pre_zero_encoding": "zlib_raw_f64_v1",
        "p_pre_zero_compression_level": 6,
    }
)
_DOWNSTREAM_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "plan_sha256",
        "plan_file_sha256",
        "source_kind",
        "evaluator_source_sha256",
        "source_manifest_path",
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "source_plan_sha256",
        "source_input_hashes_sha256",
        "source_statuses_sha256",
        "source_plan_authority",
        "evaluated_round_binding_sha256",
        "development_revision_versions",
        "development_sources",
        "planned_denominator_count",
        "recorded_denominator_count",
        "endpoint_row_count",
        "records",
        "manifest_sha256",
    }
)


class DownstreamEvidenceError(ValueError):
    """Raised when source, dataset, resume, or output evidence is invalid."""


def _direct_projection_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return set(left) == set(right) and all(
            _direct_projection_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(
            _direct_projection_equal(first, second)
            for first, second in zip(left, right, strict=True)
        )
    if type(left) is float and type(right) is float:
        return left.hex() == right.hex()
    return type(left) is type(right) and left == right


def validate_direct_comparator_projection(
    projection: object,
    checkpoint_path: Path,
    plan: object,
    *,
    registry: object,
    prepared_datasets: Mapping[str, object],
    comparator_reference: object,
    comparator_authority: object,
    selected_rows: Sequence[object],
) -> dict[str, object]:
    """Revalidate the complete direct source before downstream handoff."""

    if getattr(plan, "identity_mode", None) != "direct-v1":
        raise DownstreamEvidenceError("direct comparator identity mode differs")
    if not isinstance(projection, Mapping) or set(projection) != {
        "comparator_authority",
        "selected_comparators",
    }:
        raise DownstreamEvidenceError("direct comparator projection schema differs")
    reference = projection.get("comparator_authority")
    selected = projection.get("selected_comparators")
    if not isinstance(reference, Mapping) or set(reference) != {
        "path",
        "schema_version",
        "authority_revision",
    }:
        raise DownstreamEvidenceError("direct comparator authority schema differs")
    if not isinstance(selected, Mapping) or any(
        not isinstance(method_id, str)
        or not method_id
        or not isinstance(row, Mapping)
        or set(row) != {"configuration_id", "payload"}
        or not isinstance(row.get("configuration_id"), str)
        or not row.get("configuration_id")
        or type(row.get("payload")) is not dict
        for method_id, row in selected.items()
    ):
        raise DownstreamEvidenceError(
            "direct selected comparator projection schema differs"
        )
    from .development_evaluation import project_direct_comparator_evidence

    expected = project_direct_comparator_evidence(
        checkpoint_path,
        plan,
        registry=registry,
        prepared_datasets=prepared_datasets,
        comparator_reference=comparator_reference,
        comparator_authority=comparator_authority,
        selected_rows=selected_rows,
    )
    if not _direct_projection_equal(projection, expected):
        raise DownstreamEvidenceError("direct comparator projection differs")
    return expected


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError("evidence is not canonical JSON") from error


def _reject_constant(value: str) -> None:
    raise DownstreamEvidenceError(f"nonfinite JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DownstreamEvidenceError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise DownstreamEvidenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DownstreamEvidenceError(f"{name} must be a nonempty string")
    return value


def _reject_symlink_chain(path: Path, name: str) -> None:
    for component in (path, *path.parents):
        try:
            metadata = component.lstat()
        except OSError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise DownstreamEvidenceError(f"{name} path contains a symlink")


def _existing_directory(value: str | Path, name: str) -> Path:
    path = Path(value).absolute()
    _reject_symlink_chain(path, name)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DownstreamEvidenceError(f"{name} is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DownstreamEvidenceError(f"{name} must be a non-symlink directory")
    return path


def _regular_file(path: Path, name: str) -> os.stat_result:
    _reject_symlink_chain(path, name)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise DownstreamEvidenceError(f"{name} is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DownstreamEvidenceError(f"{name} must be a regular non-symlink file")
    return metadata


def _stable_file_bytes(
    path: Path, name: str, *, max_bytes: int | None = None
) -> tuple[bytes, str]:
    before = _regular_file(path, name)
    if max_bytes is not None and before.st_size > max_bytes:
        raise DownstreamEvidenceError(f"{name} exceeds its bounded size")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DownstreamEvidenceError(f"{name} cannot be read") from error
    after = _regular_file(path, name)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise DownstreamEvidenceError(f"{name} changed while being read")
    return raw, hashlib.sha256(raw).hexdigest()


def _stable_file_sha256(path: Path, name: str) -> str:
    before = _regular_file(path, name)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DownstreamEvidenceError(f"{name} cannot be read") from error
    after = _regular_file(path, name)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise DownstreamEvidenceError(f"{name} changed while being hashed")
    return digest.hexdigest()


def _evaluator_source_sha256() -> str:
    """Hash every local source file that constructs or evaluates endpoint truth."""

    package = Path(__file__).absolute().parent
    digest = hashlib.sha256(b"maskimpute-downstream-evaluator-source-v1\0")
    for filename in (
        "downstream_evaluation.py",
        "downstream_evidence.py",
        "schema.py",
    ):
        raw, _file_sha256 = _stable_file_bytes(
            package / filename,
            f"downstream evaluator source {filename}",
            max_bytes=16 * 1024 * 1024,
        )
        encoded = filename.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _strict_json(path: Path, name: str) -> tuple[dict[str, object], bytes, str]:
    raw, file_sha256 = _stable_file_bytes(path, name, max_bytes=512 * 1024 * 1024)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except DownstreamEvidenceError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise DownstreamEvidenceError(f"{name} is not canonical JSON") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise DownstreamEvidenceError(f"{name} is not canonical JSON")
    return value, raw, file_sha256


def _safe_relative(root: Path, value: object, name: str) -> Path:
    relative = PurePosixPath(_text(value, f"{name} path"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise DownstreamEvidenceError(f"{name} path is unsafe")
    path = root.joinpath(*relative.parts)
    for component in (path, *path.parents):
        if component == root.parent:
            break
        if os.path.lexists(component) and stat.S_ISLNK(component.lstat().st_mode):
            raise DownstreamEvidenceError(f"{name} path contains a symlink")
    return path


def _cell_id_sha256(cell_ids: Sequence[str]) -> str:
    payload = bytearray(b"maskimpute-external-cell-ids-v1\0")
    payload.extend(struct.pack("<Q", len(cell_ids)))
    for cell_id in cell_ids:
        encoded = cell_id.encode("utf-8")
        payload.extend(struct.pack("<Q", len(encoded)))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def _stable_ids(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique")
    return result


@dataclass(frozen=True, slots=True)
class DatasetEvidenceBinding:
    """Raw and semantic dataset binding plus exact runner-retained IDs."""

    dataset_id: str
    path: str
    file_sha256: str
    dataset_sha256: str
    mechanism: str
    biological_id: str
    technical_view: str
    method_input_sha256: str
    dataset_qc_policy_sha256: str
    excluded_cell_count: int
    excluded_cell_ids_sha256: str
    retained_cell_count: int
    retained_cell_ids_sha256: str
    retained_gene_count: int
    observed_zero_count: int
    retained_cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    trajectory_root_cell_id: str | None = None
    trajectory_source_id: str | None = None
    trajectory_authority_sha256: str | None = None
    trajectory_binding_sha256: str | None = None

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset_id")
        if not Path(self.path).is_absolute():
            raise ValueError("dataset binding path must be absolute")
        _digest(self.file_sha256, "dataset file checksum")
        _digest(self.dataset_sha256, "dataset semantic checksum")
        for name in ("mechanism", "biological_id", "technical_view"):
            _text(getattr(self, name), f"dataset {name}")
        for name in (
            "method_input_sha256",
            "dataset_qc_policy_sha256",
            "excluded_cell_ids_sha256",
            "retained_cell_ids_sha256",
        ):
            _digest(getattr(self, name), name)
        cells = _stable_ids(self.retained_cell_ids, "retained_cell_ids")
        genes = _stable_ids(self.gene_ids, "gene_ids")
        for name in (
            "excluded_cell_count",
            "retained_cell_count",
            "retained_gene_count",
            "observed_zero_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            self.retained_cell_count != len(cells)
            or self.retained_gene_count != len(genes)
            or self.observed_zero_count
            > self.retained_cell_count * self.retained_gene_count
        ):
            raise ValueError("retained dataset dimension authority differs")
        trajectory_values = (
            self.trajectory_root_cell_id,
            self.trajectory_source_id,
            self.trajectory_authority_sha256,
            self.trajectory_binding_sha256,
        )
        if any(value is not None for value in trajectory_values) and not all(
            value is not None for value in trajectory_values
        ):
            raise ValueError("trajectory authority binding must be complete")
        if self.trajectory_root_cell_id is not None:
            if self.trajectory_root_cell_id not in cells:
                raise ValueError("trajectory root must be retained")
            _text(self.trajectory_source_id, "trajectory_source_id")
            _digest(
                self.trajectory_authority_sha256,
                "trajectory authority checksum",
            )
            _digest(self.trajectory_binding_sha256, "trajectory binding checksum")
        object.__setattr__(self, "retained_cell_ids", cells)
        object.__setattr__(self, "gene_ids", genes)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "path": self.path,
            "file_sha256": self.file_sha256,
            "dataset_sha256": self.dataset_sha256,
            "mechanism": self.mechanism,
            "biological_id": self.biological_id,
            "technical_view": self.technical_view,
            "method_input_sha256": self.method_input_sha256,
            "dataset_qc_policy_sha256": self.dataset_qc_policy_sha256,
            "excluded_cell_count": self.excluded_cell_count,
            "excluded_cell_ids_sha256": self.excluded_cell_ids_sha256,
            "retained_cell_count": self.retained_cell_count,
            "retained_cell_ids_sha256": self.retained_cell_ids_sha256,
            "retained_gene_count": self.retained_gene_count,
            "observed_zero_count": self.observed_zero_count,
            "retained_cell_ids": list(self.retained_cell_ids),
            "gene_ids": list(self.gene_ids),
            "trajectory_root_cell_id": self.trajectory_root_cell_id,
            "trajectory_source_id": self.trajectory_source_id,
            "trajectory_authority_sha256": self.trajectory_authority_sha256,
            "trajectory_binding_sha256": self.trajectory_binding_sha256,
        }


def _trajectory_authority_binding(
    dataset: ad.AnnData,
    dataset_sha256: str,
    *,
    trajectory_root_cell_id: str | None,
    trajectory_source_id: str | None,
    trajectory_authority_sha256: str | None,
    trajectory_binding_sha256: str | None,
    populate_authority_digests: bool,
) -> tuple[str | None, str | None]:
    """Validate the one registered trajectory contract as a closed identity."""

    from .trajectory_dataset import (
        FOUR_RECONSTRUCTION_MECHANISMS,
        REGISTERED_TRAJECTORY_DATASET_ID,
        TrajectoryAuthorityError,
        load_trajectory_authority,
    )

    mechanisms = dataset.obs["mechanism"].astype(str).unique().tolist()
    dataset_ids = dataset.obs["dataset_id"].astype(str).unique().tolist()
    if len(mechanisms) != 1 or len(dataset_ids) != 1:
        raise DownstreamEvidenceError(
            "evaluator trajectory dataset identity is not constant"
        )
    mechanism = mechanisms[0]
    dataset_id = dataset_ids[0]
    supplied = (
        trajectory_root_cell_id,
        trajectory_source_id,
        trajectory_authority_sha256,
        trajectory_binding_sha256,
    )
    any_supplied = any(value is not None for value in supplied)
    has_pseudotime = "pseudotime" in dataset.obs
    if mechanism in FOUR_RECONSTRUCTION_MECHANISMS:
        if any_supplied or has_pseudotime:
            raise DownstreamEvidenceError(
                "reconstruction mechanism cannot carry trajectory authority"
            )
        return None, None

    is_registered_trajectory = (
        mechanism == "synthetic_trajectory"
        or dataset_id == REGISTERED_TRAJECTORY_DATASET_ID
        or has_pseudotime
        or any_supplied
    )
    if not is_registered_trajectory:
        return None, None
    if trajectory_root_cell_id is None or trajectory_source_id is None:
        raise DownstreamEvidenceError("registered trajectory authority is required")
    try:
        authority = load_trajectory_authority()
    except TrajectoryAuthorityError as error:
        raise DownstreamEvidenceError(
            "registered trajectory authority cannot be validated"
        ) from error
    if populate_authority_digests:
        if (
            trajectory_authority_sha256 is not None
            or trajectory_binding_sha256 is not None
        ):
            raise DownstreamEvidenceError(
                "trajectory authority digests are evaluator-owned"
            )
        trajectory_authority_sha256 = authority.authority_sha256
        trajectory_binding_sha256 = authority.binding_sha256
    if trajectory_authority_sha256 != authority.authority_sha256:
        raise DownstreamEvidenceError("trajectory authority checksum differs")
    if trajectory_binding_sha256 != authority.binding_sha256:
        raise DownstreamEvidenceError("trajectory binding checksum differs")
    if dataset_sha256 != authority.expected_dataset_sha256:
        raise DownstreamEvidenceError("trajectory dataset checksum differs")
    if mechanism != authority.mechanism:
        raise DownstreamEvidenceError("trajectory mechanism differs")
    if dataset_id != REGISTERED_TRAJECTORY_DATASET_ID:
        raise DownstreamEvidenceError("trajectory dataset identity differs")
    if trajectory_root_cell_id != authority.root_cell_id:
        raise DownstreamEvidenceError("trajectory root differs")
    if trajectory_source_id != authority.source_id:
        raise DownstreamEvidenceError("trajectory source differs")
    return trajectory_authority_sha256, trajectory_binding_sha256


def _read_bound_dataset(binding: DatasetEvidenceBinding) -> ad.AnnData:
    path = Path(binding.path)
    observed_file_sha = _stable_file_sha256(path, "bound evaluator dataset")
    if observed_file_sha != binding.file_sha256:
        raise DownstreamEvidenceError("evaluator dataset raw file checksum differs")
    try:
        dataset = ad.read_h5ad(path)
        validate_benchmark_dataset(dataset)
    except (OSError, TypeError, ValueError) as error:
        raise DownstreamEvidenceError("evaluator dataset validation failed") from error
    dataset_sha256 = benchmark_dataset_sha256(dataset)
    if dataset_sha256 != binding.dataset_sha256:
        raise DownstreamEvidenceError("evaluator dataset semantic checksum differs")
    dataset_ids = dataset.obs["dataset_id"].astype(str).unique().tolist()
    if dataset_ids != [binding.dataset_id]:
        raise DownstreamEvidenceError("evaluator dataset identity differs")
    observed_cells = tuple(dataset.obs_names.astype(str))
    positions = {value: index for index, value in enumerate(observed_cells)}
    if any(value not in positions for value in binding.retained_cell_ids):
        raise DownstreamEvidenceError("retained cell IDs are absent from dataset")
    indices = [positions[value] for value in binding.retained_cell_ids]
    if indices != sorted(indices):
        raise DownstreamEvidenceError("retained cell IDs changed source order")
    if tuple(dataset.var_names.astype(str)) != binding.gene_ids:
        raise DownstreamEvidenceError("evaluator gene IDs differ")
    _trajectory_authority_binding(
        dataset,
        dataset_sha256,
        trajectory_root_cell_id=binding.trajectory_root_cell_id,
        trajectory_source_id=binding.trajectory_source_id,
        trajectory_authority_sha256=binding.trajectory_authority_sha256,
        trajectory_binding_sha256=binding.trajectory_binding_sha256,
        populate_authority_digests=False,
    )
    return dataset


def bind_evaluator_dataset(
    path: str | Path,
    *,
    retained_cell_ids: Sequence[str],
    trajectory_root_cell_id: str | None = None,
    trajectory_source_id: str | None = None,
) -> DatasetEvidenceBinding:
    """Create a fully revalidated binding for one persisted evaluator dataset."""

    dataset_path = Path(path).absolute()
    file_sha256 = _stable_file_sha256(dataset_path, "evaluator dataset")
    try:
        dataset = ad.read_h5ad(dataset_path)
        validate_benchmark_dataset(dataset)
    except (OSError, TypeError, ValueError) as error:
        raise DownstreamEvidenceError("evaluator dataset validation failed") from error
    cell_ids = _stable_ids(retained_cell_ids, "retained_cell_ids")
    source_cells = tuple(dataset.obs_names.astype(str))
    positions = {value: index for index, value in enumerate(source_cells)}
    if any(value not in positions for value in cell_ids):
        raise DownstreamEvidenceError("retained cell IDs are absent from dataset")
    if [positions[value] for value in cell_ids] != sorted(
        positions[value] for value in cell_ids
    ):
        raise DownstreamEvidenceError("retained cell IDs must preserve dataset order")
    if (trajectory_root_cell_id is None) != (trajectory_source_id is None):
        raise DownstreamEvidenceError("trajectory root and source must be paired")
    if trajectory_root_cell_id is not None and trajectory_root_cell_id not in cell_ids:
        raise DownstreamEvidenceError("trajectory root is not a retained cell")
    dataset_ids = dataset.obs["dataset_id"].astype(str).unique().tolist()
    if len(dataset_ids) != 1:
        raise DownstreamEvidenceError("evaluator dataset ID is not constant")
    identities: dict[str, str] = {}
    for name in ("mechanism", "biological_id", "technical_view"):
        values = dataset.obs[name].astype(str).unique().tolist()
        if len(values) != 1 or not values[0]:
            raise DownstreamEvidenceError(f"evaluator dataset {name} is not constant")
        identities[name] = values[0]
    excluded_cell_ids = tuple(
        value for value in source_cells if value not in set(cell_ids)
    )
    try:
        from .methods import prepare_method_input
        from .runner import DatasetQCPolicy, method_input_sha256
        from .schema import make_inference_view

        inference = make_inference_view(dataset)
        retained = inference[list(cell_ids), :].copy()
        method_input = prepare_method_input(retained)
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "evaluator method-input authority could not be derived"
        ) from error
    dataset_sha256 = benchmark_dataset_sha256(dataset)
    trajectory_authority_sha256, trajectory_binding_sha256 = (
        _trajectory_authority_binding(
            dataset,
            dataset_sha256,
            trajectory_root_cell_id=trajectory_root_cell_id,
            trajectory_source_id=trajectory_source_id,
            trajectory_authority_sha256=None,
            trajectory_binding_sha256=None,
            populate_authority_digests=True,
        )
    )
    return DatasetEvidenceBinding(
        dataset_id=dataset_ids[0],
        path=str(dataset_path),
        file_sha256=file_sha256,
        dataset_sha256=dataset_sha256,
        mechanism=identities["mechanism"],
        biological_id=identities["biological_id"],
        technical_view=identities["technical_view"],
        method_input_sha256=method_input_sha256(method_input),
        dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
        excluded_cell_count=len(excluded_cell_ids),
        excluded_cell_ids_sha256=_cell_id_sha256(excluded_cell_ids),
        retained_cell_count=len(cell_ids),
        retained_cell_ids_sha256=_cell_id_sha256(cell_ids),
        retained_gene_count=method_input.shape[1],
        observed_zero_count=int((method_input.counts == 0.0).sum()),
        retained_cell_ids=cell_ids,
        gene_ids=tuple(dataset.var_names.astype(str)),
        trajectory_root_cell_id=trajectory_root_cell_id,
        trajectory_source_id=trajectory_source_id,
        trajectory_authority_sha256=trajectory_authority_sha256,
        trajectory_binding_sha256=trajectory_binding_sha256,
    )


def bind_prepared_evaluator_panel(
    dataset_bindings: Sequence[object],
    prepared_datasets: Mapping[str, object],
    *,
    dataset_root: str | Path,
) -> tuple[DatasetEvidenceBinding, ...]:
    """Bridge runner-prepared QC identities to persisted evaluator H5AD files."""

    if isinstance(dataset_bindings, (str, bytes)) or not isinstance(
        dataset_bindings, Sequence
    ):
        raise TypeError("dataset_bindings must be a sequence")
    if not isinstance(prepared_datasets, Mapping):
        raise TypeError("prepared_datasets must be a mapping")
    root = _existing_directory(dataset_root, "evaluator dataset root")
    result: list[DatasetEvidenceBinding] = []
    seen: set[str] = set()
    for index, runner_binding in enumerate(dataset_bindings):
        dataset_id = getattr(runner_binding, "dataset_id", None)
        output_path = getattr(runner_binding, "output_path", None)
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or dataset_id in seen
            or not isinstance(output_path, str)
            or not output_path
        ):
            raise DownstreamEvidenceError(f"runner dataset binding {index} is invalid")
        relative = PurePosixPath(output_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DownstreamEvidenceError("runner dataset output path is unsafe")
        prepared = prepared_datasets.get(dataset_id)
        audit = getattr(prepared, "audit", None)
        retained_cell_ids = getattr(audit, "retained_cell_ids", None)
        if retained_cell_ids is None:
            raise DownstreamEvidenceError(
                f"prepared evaluator identity is absent for {dataset_id}"
            )
        bound = bind_evaluator_dataset(
            root.joinpath(*relative.parts),
            retained_cell_ids=retained_cell_ids,
        )
        if bound.dataset_id != dataset_id:
            raise DownstreamEvidenceError("runner and evaluator dataset IDs differ")
        prepared_binding = getattr(prepared, "binding", None)
        method_input = getattr(prepared, "method_input", None)
        try:
            from .runner import method_input_sha256

            prepared_method_input_sha256 = method_input_sha256(method_input)
        except (TypeError, ValueError) as error:
            raise DownstreamEvidenceError(
                f"prepared method-input authority is invalid for {dataset_id}"
            ) from error
        expected_prepared_authority = {
            "mechanism": getattr(prepared_binding, "mechanism", None),
            "biological_id": getattr(prepared_binding, "biological_id", None),
            "technical_view": getattr(prepared_binding, "technical_view", None),
            "method_input_sha256": prepared_method_input_sha256,
            "excluded_cell_count": getattr(audit, "excluded_cell_count", None),
            "excluded_cell_ids_sha256": getattr(
                audit, "excluded_cell_ids_sha256", None
            ),
            "retained_cell_count": getattr(audit, "retained_cell_count", None),
            "retained_cell_ids_sha256": getattr(
                audit, "retained_cell_ids_sha256", None
            ),
            "retained_gene_count": getattr(method_input, "shape", (None, None))[1],
            "observed_zero_count": int((method_input.counts == 0.0).sum()),
        }
        if any(
            getattr(bound, name) != expected
            for name, expected in expected_prepared_authority.items()
        ):
            raise DownstreamEvidenceError(
                f"prepared evaluator authority differs for {dataset_id}"
            )
        result.append(bound)
        seen.add(dataset_id)
    if not result or tuple(value.dataset_id for value in result) != tuple(
        getattr(value, "dataset_id", None) for value in dataset_bindings
    ):
        raise DownstreamEvidenceError("prepared evaluator panel order differs")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DownstreamPlanEntry:
    ordinal: int
    source_record_path: str
    source_record_sha256: str
    run_id: str
    method_id: str
    dataset_id: str
    source_dataset_sha256: str
    mechanism: str
    biological_id: str
    technical_view: str
    model_seed: int | None
    configuration_id: str
    configuration_sha256: str
    configuration_kind: str
    method_artifact_sha256: str
    method_input_sha256: str
    retained_cell_ids_sha256: str
    status: str
    reason: str | None
    evaluator_output_sha256: str | None
    evaluator_output_path: str | None
    evaluator_output_file_sha256: str | None
    evaluator_output_shape: tuple[int, int] | None
    evaluator_output_encoding: str | None
    evaluator_output_uncompressed_nbytes: int | None
    evaluator_output_uncompressed_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "source_record_path": self.source_record_path,
            "source_record_sha256": self.source_record_sha256,
            "run_id": self.run_id,
            "method_id": self.method_id,
            "dataset_id": self.dataset_id,
            "source_dataset_sha256": self.source_dataset_sha256,
            "mechanism": self.mechanism,
            "biological_id": self.biological_id,
            "technical_view": self.technical_view,
            "model_seed": self.model_seed,
            "configuration_id": self.configuration_id,
            "configuration_sha256": self.configuration_sha256,
            "configuration_kind": self.configuration_kind,
            "method_artifact_sha256": self.method_artifact_sha256,
            "method_input_sha256": self.method_input_sha256,
            "retained_cell_ids_sha256": self.retained_cell_ids_sha256,
            "status": self.status,
            "reason": self.reason,
            "evaluator_output_sha256": self.evaluator_output_sha256,
            "evaluator_output_path": self.evaluator_output_path,
            "evaluator_output_file_sha256": self.evaluator_output_file_sha256,
            "evaluator_output_shape": (
                None
                if self.evaluator_output_shape is None
                else list(self.evaluator_output_shape)
            ),
            "evaluator_output_encoding": self.evaluator_output_encoding,
            "evaluator_output_uncompressed_nbytes": (
                self.evaluator_output_uncompressed_nbytes
            ),
            "evaluator_output_uncompressed_sha256": (
                self.evaluator_output_uncompressed_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class EvaluatedRoundBinding:
    """Exact immutable lifecycle receipt binding for a frozen final source."""

    repository_root: str
    round_root: str
    round_id: str
    evaluation_receipt_path: str
    evaluation_receipt_file_sha256: str
    evaluation_receipt_payload_sha256: str
    result_manifest_sha256: str
    final_plan_sha256: str
    final_execution_manifest_path: str
    final_execution_manifest_file_sha256: str
    final_execution_manifest_payload_sha256: str
    execution_validation_sha256: str
    storage_preflight_sha256: str
    scaling_evidence_sha256: str
    scaling_plan_sha256: str
    scaling_checkpoint_path: str
    scaling_checkpoint_file_sha256: str
    scaling_checkpoint_payload_sha256: str
    scaling_checkpoint_history_sha256: str
    scaling_checkpoint_history_count: int
    scaling_result_files_sha256: str
    scaling_result_file_count: int
    trajectory_evidence_sha256: str
    trajectory_plan_sha256: str
    trajectory_execution_claim_sha256: str
    trajectory_execution_environment_sha256: str
    trajectory_dataset_id: str
    trajectory_dataset_sha256: str
    trajectory_dataset_file_sha256: str
    trajectory_dataset_receipt_file_sha256: str
    trajectory_dataset_receipt_payload_sha256: str
    trajectory_source_id: str
    trajectory_root_cell_id: str
    trajectory_registered_authority_sha256: str
    trajectory_registered_binding_sha256: str
    trajectory_authority_sha256: str
    trajectory_authority_file_sha256: str
    trajectory_execution_manifest_path: str
    trajectory_execution_manifest_file_sha256: str
    trajectory_execution_manifest_payload_sha256: str
    trajectory_execution_validation_sha256: str
    trajectory_record_payload_sha256s_sha256: str
    trajectory_status_counts_sha256: str
    trajectory_planned_run_count: int
    trajectory_result_files_sha256: str
    trajectory_result_file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": self.repository_root,
            "round_root": self.round_root,
            "round_id": self.round_id,
            "evaluation_receipt_path": self.evaluation_receipt_path,
            "evaluation_receipt_file_sha256": (self.evaluation_receipt_file_sha256),
            "evaluation_receipt_payload_sha256": (
                self.evaluation_receipt_payload_sha256
            ),
            "result_manifest_sha256": self.result_manifest_sha256,
            "final_plan_sha256": self.final_plan_sha256,
            "final_execution_manifest_path": self.final_execution_manifest_path,
            "final_execution_manifest_file_sha256": (
                self.final_execution_manifest_file_sha256
            ),
            "final_execution_manifest_payload_sha256": (
                self.final_execution_manifest_payload_sha256
            ),
            "execution_validation_sha256": self.execution_validation_sha256,
            "storage_preflight_sha256": self.storage_preflight_sha256,
            "scaling_evidence_sha256": self.scaling_evidence_sha256,
            "scaling_plan_sha256": self.scaling_plan_sha256,
            "scaling_checkpoint_path": self.scaling_checkpoint_path,
            "scaling_checkpoint_file_sha256": (self.scaling_checkpoint_file_sha256),
            "scaling_checkpoint_payload_sha256": (
                self.scaling_checkpoint_payload_sha256
            ),
            "scaling_checkpoint_history_sha256": (
                self.scaling_checkpoint_history_sha256
            ),
            "scaling_checkpoint_history_count": (self.scaling_checkpoint_history_count),
            "scaling_result_files_sha256": self.scaling_result_files_sha256,
            "scaling_result_file_count": self.scaling_result_file_count,
            "trajectory_evidence_sha256": self.trajectory_evidence_sha256,
            "trajectory_plan_sha256": self.trajectory_plan_sha256,
            "trajectory_execution_claim_sha256": (
                self.trajectory_execution_claim_sha256
            ),
            "trajectory_execution_environment_sha256": (
                self.trajectory_execution_environment_sha256
            ),
            "trajectory_dataset_id": self.trajectory_dataset_id,
            "trajectory_dataset_sha256": self.trajectory_dataset_sha256,
            "trajectory_dataset_file_sha256": self.trajectory_dataset_file_sha256,
            "trajectory_dataset_receipt_file_sha256": (
                self.trajectory_dataset_receipt_file_sha256
            ),
            "trajectory_dataset_receipt_payload_sha256": (
                self.trajectory_dataset_receipt_payload_sha256
            ),
            "trajectory_source_id": self.trajectory_source_id,
            "trajectory_root_cell_id": self.trajectory_root_cell_id,
            "trajectory_registered_authority_sha256": (
                self.trajectory_registered_authority_sha256
            ),
            "trajectory_registered_binding_sha256": (
                self.trajectory_registered_binding_sha256
            ),
            "trajectory_authority_sha256": self.trajectory_authority_sha256,
            "trajectory_authority_file_sha256": (self.trajectory_authority_file_sha256),
            "trajectory_execution_manifest_path": (
                self.trajectory_execution_manifest_path
            ),
            "trajectory_execution_manifest_file_sha256": (
                self.trajectory_execution_manifest_file_sha256
            ),
            "trajectory_execution_manifest_payload_sha256": (
                self.trajectory_execution_manifest_payload_sha256
            ),
            "trajectory_execution_validation_sha256": (
                self.trajectory_execution_validation_sha256
            ),
            "trajectory_record_payload_sha256s_sha256": (
                self.trajectory_record_payload_sha256s_sha256
            ),
            "trajectory_status_counts_sha256": (self.trajectory_status_counts_sha256),
            "trajectory_planned_run_count": self.trajectory_planned_run_count,
            "trajectory_result_files_sha256": self.trajectory_result_files_sha256,
            "trajectory_result_file_count": self.trajectory_result_file_count,
        }

    @property
    def binding_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


_EVALUATED_ROUND_BINDING_FIELDS = frozenset(
    {
        "repository_root",
        "round_root",
        "round_id",
        "evaluation_receipt_path",
        "evaluation_receipt_file_sha256",
        "evaluation_receipt_payload_sha256",
        "result_manifest_sha256",
        "final_plan_sha256",
        "final_execution_manifest_path",
        "final_execution_manifest_file_sha256",
        "final_execution_manifest_payload_sha256",
        "execution_validation_sha256",
        "storage_preflight_sha256",
        "scaling_evidence_sha256",
        "scaling_plan_sha256",
        "scaling_checkpoint_path",
        "scaling_checkpoint_file_sha256",
        "scaling_checkpoint_payload_sha256",
        "scaling_checkpoint_history_sha256",
        "scaling_checkpoint_history_count",
        "scaling_result_files_sha256",
        "scaling_result_file_count",
        "trajectory_evidence_sha256",
        "trajectory_plan_sha256",
        "trajectory_execution_claim_sha256",
        "trajectory_execution_environment_sha256",
        "trajectory_dataset_id",
        "trajectory_dataset_sha256",
        "trajectory_dataset_file_sha256",
        "trajectory_dataset_receipt_file_sha256",
        "trajectory_dataset_receipt_payload_sha256",
        "trajectory_source_id",
        "trajectory_root_cell_id",
        "trajectory_registered_authority_sha256",
        "trajectory_registered_binding_sha256",
        "trajectory_authority_sha256",
        "trajectory_authority_file_sha256",
        "trajectory_execution_manifest_path",
        "trajectory_execution_manifest_file_sha256",
        "trajectory_execution_manifest_payload_sha256",
        "trajectory_execution_validation_sha256",
        "trajectory_record_payload_sha256s_sha256",
        "trajectory_status_counts_sha256",
        "trajectory_planned_run_count",
        "trajectory_result_files_sha256",
        "trajectory_result_file_count",
    }
)


@dataclass(frozen=True, slots=True)
class DevelopmentSourceBinding:
    """One separately sealed checkpoint inside a revision-aware development plan."""

    source_id: str
    source_root: str
    selected_methods: tuple[str, ...]
    manifest_path: str
    manifest_file_sha256: str
    manifest_payload_sha256: str
    plan_sha256: str
    input_hashes_sha256: str
    statuses_sha256: str
    denominator_sha256: str
    planned_denominator_count: int
    evaluation_manifest_path: str
    evaluation_manifest_file_sha256: str
    evaluation_manifest_payload_sha256: str
    evaluation_source_pointer: str
    evaluation_source_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_root": self.source_root,
            "selected_methods": list(self.selected_methods),
            "manifest_path": self.manifest_path,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_payload_sha256": self.manifest_payload_sha256,
            "plan_sha256": self.plan_sha256,
            "input_hashes_sha256": self.input_hashes_sha256,
            "statuses_sha256": self.statuses_sha256,
            "denominator_sha256": self.denominator_sha256,
            "planned_denominator_count": self.planned_denominator_count,
            "evaluation_manifest_path": self.evaluation_manifest_path,
            "evaluation_manifest_file_sha256": (self.evaluation_manifest_file_sha256),
            "evaluation_manifest_payload_sha256": (
                self.evaluation_manifest_payload_sha256
            ),
            "evaluation_source_pointer": self.evaluation_source_pointer,
            "evaluation_source_sha256": self.evaluation_source_sha256,
        }


_DEVELOPMENT_SOURCE_BINDING_FIELDS = frozenset(
    {
        "source_id",
        "source_root",
        "selected_methods",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "plan_sha256",
        "input_hashes_sha256",
        "statuses_sha256",
        "denominator_sha256",
        "planned_denominator_count",
        "evaluation_manifest_path",
        "evaluation_manifest_file_sha256",
        "evaluation_manifest_payload_sha256",
        "evaluation_source_pointer",
        "evaluation_source_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class DownstreamEvidencePlan:
    source_root: str
    source_kind: str
    evidence_scope: str
    evaluator_source_sha256: str
    source_manifest_path: str
    source_manifest_file_sha256: str
    source_manifest_payload_sha256: str
    source_plan_sha256: str
    source_input_hashes_sha256: str
    source_statuses_sha256: str
    source_plan_authority: str
    evaluated_round_binding: EvaluatedRoundBinding | None
    development_revision_versions: tuple[str, ...]
    development_sources: tuple[DevelopmentSourceBinding, ...]
    datasets: tuple[DatasetEvidenceBinding, ...]
    configurations: tuple[AuthorizedConfiguration, ...]
    entries: tuple[DownstreamPlanEntry, ...]
    plan_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": 3,
            "source_root": self.source_root,
            "source_kind": self.source_kind,
            "evidence_scope": self.evidence_scope,
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "source_manifest_path": self.source_manifest_path,
            "source_manifest_file_sha256": self.source_manifest_file_sha256,
            "source_manifest_payload_sha256": self.source_manifest_payload_sha256,
            "source_plan_sha256": self.source_plan_sha256,
            "source_input_hashes_sha256": self.source_input_hashes_sha256,
            "source_statuses_sha256": self.source_statuses_sha256,
            "source_plan_authority": self.source_plan_authority,
            "evaluated_round_binding": (
                None
                if self.evaluated_round_binding is None
                else self.evaluated_round_binding.to_dict()
            ),
            "development_revision_versions": list(self.development_revision_versions),
            "development_sources": [
                value.to_dict() for value in self.development_sources
            ],
            "datasets": [value.to_dict() for value in self.datasets],
            "configurations": [
                _legacy_configuration_payload(value) for value in self.configurations
            ],
            "entries": [value.to_dict() for value in self.entries],
            "planned_denominator_count": len(self.entries),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "plan_sha256": self.plan_sha256}


@dataclass(frozen=True, slots=True)
class DevelopmentSourcePlan:
    """Transient independently authorized checkpoint used to assemble a bundle."""

    source_id: str
    plan: DownstreamEvidencePlan
    selected_methods: tuple[str, ...]
    evaluation_manifest_path: str
    evaluation_manifest_file_sha256: str
    evaluation_manifest_payload_sha256: str
    evaluation_source_pointer: str
    evaluation_source_sha256: str


_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "source_root",
        "source_kind",
        "evidence_scope",
        "evaluator_source_sha256",
        "source_manifest_path",
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "source_plan_sha256",
        "source_input_hashes_sha256",
        "source_statuses_sha256",
        "source_plan_authority",
        "evaluated_round_binding",
        "development_revision_versions",
        "development_sources",
        "datasets",
        "configurations",
        "entries",
        "planned_denominator_count",
        "plan_sha256",
    }
)
_DATASET_BINDING_FIELDS = frozenset(
    {
        "dataset_id",
        "path",
        "file_sha256",
        "dataset_sha256",
        "mechanism",
        "biological_id",
        "technical_view",
        "method_input_sha256",
        "dataset_qc_policy_sha256",
        "excluded_cell_count",
        "excluded_cell_ids_sha256",
        "retained_cell_count",
        "retained_cell_ids_sha256",
        "retained_gene_count",
        "observed_zero_count",
        "retained_cell_ids",
        "gene_ids",
        "trajectory_root_cell_id",
        "trajectory_source_id",
        "trajectory_authority_sha256",
        "trajectory_binding_sha256",
    }
)
_CONFIGURATION_FIELDS = frozenset(
    {
        "method_id",
        "configuration_id",
        "kind",
        "configuration_sha256",
        "payload",
        "requires_count_score",
        "requires_calibration",
    }
)


def _legacy_configuration_payload(
    configuration: AuthorizedConfiguration,
) -> dict[str, object]:
    """Preserve the established non-comparator downstream configuration schema."""

    if not isinstance(configuration, AuthorizedConfiguration):
        raise TypeError("configuration must be an AuthorizedConfiguration")
    encoded = configuration.to_dict()
    return {name: encoded[name] for name in _CONFIGURATION_FIELDS}


def _method_artifact_sha256(configuration: AuthorizedConfiguration) -> str:
    """Derive the selection-authority artifact from one sealed configuration."""

    if not isinstance(configuration, AuthorizedConfiguration):
        raise TypeError("configuration must be an AuthorizedConfiguration")
    payload = dict(configuration.payload)
    if configuration.kind != "registry":
        return configuration.configuration_sha256
    method = payload.get("method")
    if (
        configuration.configuration_id != "registry-default"
        or set(payload) != {"schema", "method"}
        or payload.get("schema") != "maskimpute-registry-default-configuration-v1"
        or not isinstance(method, Mapping)
        or method.get("id") != configuration.method_id
    ):
        raise DownstreamEvidenceError("registry configuration method payload differs")
    return canonical_sha256(dict(method))


def _validated_evaluated_round_receipt(
    repository: Path, round_root: Path
) -> dict[str, object]:
    """Reuse the frozen lifecycle's read-only evaluated-round validation."""

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
        selected_repository, destination = _canonical_round(repository, round_root)
        if selected_repository != repository or destination != round_root:
            raise DownstreamEvidenceError(
                "evaluated final round path differs from its lexical authority"
            )
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
            raise DownstreamEvidenceError("evaluated final round receipt is absent")
        result_manifest = receipt.get("result_manifest")
        if not isinstance(result_manifest, Mapping):
            raise DownstreamEvidenceError("evaluated final result manifest is absent")
        allowed_paths = _validate_result_files(
            selected_repository, destination, result_manifest
        )
        _verify_frozen_repository(
            selected_repository,
            destination,
            allowed_result_paths=allowed_paths,
        )
    except DownstreamEvidenceError:
        raise
    except (FinalRunnerContractError, StudyStateError) as error:
        raise DownstreamEvidenceError(
            f"evaluated final round failed lifecycle validation: {error}"
        ) from error
    return dict(receipt)


def _scaling_result_file_rows(
    value: object,
    name: str,
    *,
    scaling_only: bool,
) -> tuple[dict[str, str], ...]:
    """Return one exact, ordered scaling inventory without accepting aliases."""

    if not isinstance(value, list):
        raise DownstreamEvidenceError(f"{name} must be an array")
    rows: list[dict[str, str]] = []
    observed: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise DownstreamEvidenceError(f"{name} row schema differs")
        path = _text(item.get("path"), f"{name} path")
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or path != relative.as_posix()
            or ".." in relative.parts
            or not relative.parts
        ):
            raise DownstreamEvidenceError(f"{name} path is unsafe")
        if scaling_only and not path.startswith("results/scaling/"):
            continue
        if not path.startswith("results/scaling/") or path in observed:
            raise DownstreamEvidenceError(f"{name} path set differs")
        observed.add(path)
        rows.append(
            {
                "path": path,
                "sha256": _digest(item.get("sha256"), f"{name} file checksum"),
            }
        )
    if not rows or rows != sorted(rows, key=lambda item: item["path"]):
        raise DownstreamEvidenceError(f"{name} ordering differs")
    return tuple(rows)


def _validated_scaling_binding_fields(
    repository_root: Path,
    round_root: Path,
    result_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate and independently replay the receipt's exact scaling evidence."""

    evidence_value = result_manifest.get("scaling_evidence")
    if (
        not isinstance(evidence_value, Mapping)
        or set(evidence_value) != _SCALING_EVIDENCE_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated scaling evidence schema differs")
    evidence = dict(evidence_value)
    evidence_sha256 = _digest(
        evidence.get("evidence_sha256"), "evaluated scaling evidence checksum"
    )
    evidence_body = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != "completed"
        or canonical_sha256(evidence_body) != evidence_sha256
    ):
        raise DownstreamEvidenceError("evaluated scaling evidence binding differs")

    plan_value = evidence.get("plan")
    if (
        not isinstance(plan_value, Mapping)
        or set(plan_value) != _SCALING_PLAN_PAYLOAD_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated scaling plan schema differs")
    plan = dict(plan_value)
    plan_sha256 = _digest(plan.get("plan_sha256"), "evaluated scaling plan checksum")
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema_version") != 1
        or not isinstance(plan.get("input_hashes"), Mapping)
        or not isinstance(plan.get("entries"), list)
        or not isinstance(plan.get("configurations"), list)
        or canonical_sha256(plan_body) != plan_sha256
    ):
        raise DownstreamEvidenceError("evaluated scaling plan payload differs")

    checkpoint_value = evidence.get("checkpoint_payload")
    if (
        not isinstance(checkpoint_value, Mapping)
        or set(checkpoint_value) != _SCALING_CHECKPOINT_PAYLOAD_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated scaling checkpoint schema differs")
    checkpoint = dict(checkpoint_value)
    checkpoint_sha256 = _digest(
        checkpoint.get("checkpoint_sha256"),
        "evaluated scaling checkpoint checksum",
    )
    checkpoint_body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    planned = checkpoint.get("planned_run_count")
    datasets = checkpoint.get("datasets")
    records = checkpoint.get("records")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("status") != "completed"
        or checkpoint.get("plan_sha256") != plan_sha256
        or checkpoint.get("input_hashes") != plan.get("input_hashes")
        or type(planned) is not int
        or planned <= 0
        or not isinstance(datasets, list)
        or not isinstance(records, list)
        or len(records) != planned
        or canonical_sha256(checkpoint_body) != checkpoint_sha256
    ):
        raise DownstreamEvidenceError("evaluated scaling checkpoint payload differs")
    checkpoint_payload_sha256 = canonical_sha256(checkpoint)

    checkpoint_relative = _text(
        evidence.get("checkpoint_path"), "evaluated scaling checkpoint"
    )
    if _SCALING_CHECKPOINT_PATH.fullmatch(checkpoint_relative) is None:
        raise DownstreamEvidenceError("evaluated scaling checkpoint path differs")
    checkpoint_path = _safe_relative(
        round_root,
        checkpoint_relative,
        "evaluated scaling checkpoint",
    )
    checkpoint_file, _checkpoint_raw, checkpoint_file_sha256 = _strict_json(
        checkpoint_path,
        "evaluated scaling checkpoint",
    )
    if checkpoint_file != checkpoint or checkpoint_file_sha256 != _digest(
        evidence.get("checkpoint_file_sha256"),
        "evaluated scaling checkpoint file checksum",
    ):
        raise DownstreamEvidenceError("evaluated scaling checkpoint file differs")

    evidence_rows = _scaling_result_file_rows(
        evidence.get("result_files"),
        "evaluated scaling result inventory",
        scaling_only=False,
    )
    cumulative_rows = _scaling_result_file_rows(
        result_manifest.get("result_files"),
        "evaluated cumulative scaling result inventory",
        scaling_only=True,
    )
    if evidence_rows != cumulative_rows:
        raise DownstreamEvidenceError(
            "evaluated scaling result inventory differs from cumulative receipt"
        )
    checkpoint_history: list[dict[str, str]] = []
    for row in evidence_rows:
        path = row["path"]
        if path.startswith("results/scaling/checkpoints/"):
            if _SCALING_CHECKPOINT_PATH.fullmatch(path) is None:
                raise DownstreamEvidenceError(
                    "evaluated scaling checkpoint history path differs"
                )
            checkpoint_history.append(row)
    expected_history = [
        f"results/scaling/checkpoints/{index:08d}.json"
        for index in range(1, len(checkpoint_history) + 1)
    ]
    if (
        not checkpoint_history
        or [row["path"] for row in checkpoint_history] != expected_history
        or checkpoint_history[-1]["path"] != checkpoint_relative
        or checkpoint_history[-1]["sha256"] != checkpoint_file_sha256
    ):
        raise DownstreamEvidenceError("evaluated scaling checkpoint history differs")

    try:
        from .scaling import (
            ScalingContractError,
            load_publication_scaling_evidence,
            scaling_checkpoint_payload,
        )

        replayed = load_publication_scaling_evidence(repository_root, round_root)
        replayed_payload = scaling_checkpoint_payload(replayed)
    except (ScalingContractError, TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "evaluated scaling publication replay failed"
        ) from error
    if replayed_payload != checkpoint:
        raise DownstreamEvidenceError(
            "evaluated scaling publication replay differs from receipt"
        )
    checkpoint_after, _checkpoint_raw_after, checkpoint_file_sha256_after = (
        _strict_json(
            checkpoint_path,
            "evaluated scaling checkpoint after publication replay",
        )
    )
    if (
        checkpoint_after != checkpoint
        or checkpoint_file_sha256_after != checkpoint_file_sha256
    ):
        raise DownstreamEvidenceError(
            "evaluated scaling checkpoint changed during publication replay"
        )

    return {
        "scaling_evidence_sha256": evidence_sha256,
        "scaling_plan_sha256": plan_sha256,
        "scaling_checkpoint_path": checkpoint_relative,
        "scaling_checkpoint_file_sha256": checkpoint_file_sha256,
        "scaling_checkpoint_payload_sha256": checkpoint_payload_sha256,
        "scaling_checkpoint_history_sha256": canonical_sha256(checkpoint_history),
        "scaling_checkpoint_history_count": len(checkpoint_history),
        "scaling_result_files_sha256": canonical_sha256(evidence_rows),
        "scaling_result_file_count": len(evidence_rows),
    }


def _trajectory_result_file_rows(
    value: object,
    name: str,
    *,
    trajectory_only: bool,
) -> tuple[dict[str, str], ...]:
    """Return one exact, ordered trajectory inventory without aliases."""

    if not isinstance(value, list):
        raise DownstreamEvidenceError(f"{name} must be an array")
    rows: list[dict[str, str]] = []
    observed: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise DownstreamEvidenceError(f"{name} row schema differs")
        path = _text(item.get("path"), f"{name} path")
        relative = PurePosixPath(path)
        if (
            relative.is_absolute()
            or path != relative.as_posix()
            or ".." in relative.parts
            or not relative.parts
        ):
            raise DownstreamEvidenceError(f"{name} path is unsafe")
        if trajectory_only and not path.startswith("results/trajectory/"):
            continue
        if not path.startswith("results/trajectory/") or path in observed:
            raise DownstreamEvidenceError(f"{name} path set differs")
        observed.add(path)
        rows.append(
            {
                "path": path,
                "sha256": _digest(item.get("sha256"), f"{name} file checksum"),
            }
        )
    if not rows or rows != sorted(rows, key=lambda item: item["path"]):
        raise DownstreamEvidenceError(f"{name} ordering differs")
    return tuple(rows)


def _validated_trajectory_binding_fields(
    repository_root: Path,
    round_root: Path,
    result_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Validate and read-only replay the receipt's trajectory evidence."""

    evidence_value = result_manifest.get("trajectory_evidence")
    if (
        not isinstance(evidence_value, Mapping)
        or set(evidence_value) != _TRAJECTORY_EVIDENCE_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated trajectory evidence schema differs")
    evidence = dict(evidence_value)
    evidence_sha256 = _digest(
        evidence.get("evidence_sha256"), "evaluated trajectory evidence checksum"
    )
    evidence_body = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != "completed"
        or evidence.get("scope") != "supplementary_trajectory"
        or canonical_sha256(evidence_body) != evidence_sha256
    ):
        raise DownstreamEvidenceError("evaluated trajectory evidence binding differs")

    plan_value = evidence.get("plan")
    if (
        not isinstance(plan_value, Mapping)
        or set(plan_value) != _TRAJECTORY_PLAN_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated trajectory plan schema differs")
    plan = dict(plan_value)
    plan_sha256 = _digest(plan.get("plan_sha256"), "evaluated trajectory plan checksum")
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    input_hashes = plan.get("input_hashes")
    entries = plan.get("entries")
    configurations = plan.get("configurations")
    from .runner import DEVELOPMENT_MODEL_SEEDS

    if (
        plan.get("schema_version") != 1
        or plan.get("scope") != "supplementary_trajectory"
        or not isinstance(input_hashes, Mapping)
        or set(input_hashes) != _TRAJECTORY_PLAN_INPUT_FIELDS
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in input_hashes.values()
        )
        or not isinstance(entries, list)
        or not entries
        or not isinstance(configurations, list)
        or not configurations
        or plan.get("model_seed_policy") != list(DEVELOPMENT_MODEL_SEEDS)
        or canonical_sha256(plan_body) != plan_sha256
        or input_hashes.get("primary_final_plan_sha256")
        != result_manifest.get("final_plan_sha256")
    ):
        raise DownstreamEvidenceError("evaluated trajectory plan binding differs")

    dataset_value = evidence.get("dataset")
    if (
        not isinstance(dataset_value, Mapping)
        or set(dataset_value) != _TRAJECTORY_DATASET_EVIDENCE_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated trajectory dataset schema differs")
    dataset = dict(dataset_value)
    binding_value = dataset.get("binding")
    try:
        from .trajectory_dataset import RegisteredTrajectoryBinding

        if not isinstance(binding_value, Mapping):
            raise TypeError("binding is not a mapping")
        binding = RegisteredTrajectoryBinding(**dict(binding_value))
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "evaluated trajectory dataset binding differs"
        ) from error
    dataset_file_sha256 = _digest(
        dataset.get("dataset_file_sha256"),
        "evaluated trajectory dataset file checksum",
    )
    dataset_sha256 = _digest(
        dataset.get("dataset_sha256"), "evaluated trajectory dataset checksum"
    )
    receipt_file_sha256 = _digest(
        dataset.get("receipt_file_sha256"),
        "evaluated trajectory dataset receipt file checksum",
    )
    receipt_payload_sha256 = _digest(
        dataset.get("receipt_payload_sha256"),
        "evaluated trajectory dataset receipt payload checksum",
    )
    if (
        dataset.get("dataset_path") != binding.dataset_file_path
        or dataset.get("receipt_path")
        != "results/trajectory/dataset/dataset_receipt.json"
        or dataset_file_sha256 != binding.dataset_file_sha256
        or dataset_sha256 != binding.dataset_sha256
        or input_hashes.get("trajectory_authority_file_sha256")
        != binding.authority_file_sha256
        or input_hashes.get("trajectory_authority_sha256") != binding.authority_sha256
        or input_hashes.get("trajectory_binding_sha256")
        != binding.registered_binding_sha256
        or input_hashes.get("trajectory_dataset_sha256") != dataset_sha256
        or input_hashes.get("trajectory_dataset_file_sha256") != dataset_file_sha256
        or input_hashes.get("trajectory_dataset_receipt_sha256")
        != receipt_payload_sha256
        or input_hashes.get("trajectory_dataset_receipt_file_sha256")
        != receipt_file_sha256
    ):
        raise DownstreamEvidenceError("evaluated trajectory dataset binding differs")

    authority_value = evidence.get("execution_authority")
    if (
        not isinstance(authority_value, Mapping)
        or set(authority_value) != _TRAJECTORY_AUTHORITY_EVIDENCE_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated trajectory authority schema differs")
    authority = dict(authority_value)
    authority_file_sha256 = _digest(
        authority.get("authority_file_sha256"),
        "evaluated trajectory authority file checksum",
    )
    authority_sha256 = _digest(
        authority.get("authority_sha256"),
        "evaluated trajectory authority checksum",
    )
    authority_root = round_root / "results/trajectory/execution_authority"
    try:
        expected_count_score_path = (
            (authority_root / "count_score_authority.json")
            .relative_to(repository_root)
            .as_posix()
        )
        expected_calibration_path = (
            (authority_root / "retained_calibration.json")
            .relative_to(repository_root)
            .as_posix()
        )
    except ValueError as error:
        raise DownstreamEvidenceError(
            "evaluated trajectory authority path escapes its repository"
        ) from error
    if (
        authority.get("authority_path")
        != "results/trajectory/execution_authority/authority.json"
        or authority.get("count_score_authority_path") != expected_count_score_path
        or authority.get("retained_calibration_path") != expected_calibration_path
        or input_hashes.get("execution_authority_sha256") != authority_sha256
    ):
        raise DownstreamEvidenceError("evaluated trajectory authority binding differs")
    count_score_authority_file_sha256 = _digest(
        authority.get("count_score_authority_file_sha256"),
        "evaluated trajectory count-score authority file checksum",
    )
    retained_calibration_file_sha256 = _digest(
        authority.get("retained_calibration_file_sha256"),
        "evaluated trajectory retained calibration file checksum",
    )

    manifest_value = evidence.get("execution_manifest")
    if (
        not isinstance(manifest_value, Mapping)
        or set(manifest_value) != _TRAJECTORY_MANIFEST_EVIDENCE_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated trajectory manifest schema differs")
    manifest = dict(manifest_value)
    manifest_path = _text(
        manifest.get("path"), "evaluated trajectory execution manifest path"
    )
    manifest_file_sha256 = _digest(
        manifest.get("file_sha256"),
        "evaluated trajectory execution manifest file checksum",
    )
    manifest_payload_sha256 = _digest(
        manifest.get("payload_sha256"),
        "evaluated trajectory execution manifest payload checksum",
    )
    if manifest_path != "results/trajectory/execution/execution_manifest.json":
        raise DownstreamEvidenceError("evaluated trajectory manifest path differs")

    validation_value = evidence.get("execution_validation")
    if (
        not isinstance(validation_value, Mapping)
        or set(validation_value) != _TRAJECTORY_EXECUTION_VALIDATION_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated trajectory validation schema differs")
    validation = dict(validation_value)
    validation_sha256 = _digest(
        validation.get("validation_sha256"),
        "evaluated trajectory validation checksum",
    )
    validation_body = {
        key: value for key, value in validation.items() if key != "validation_sha256"
    }
    planned = validation.get("planned_run_count")
    completed = validation.get("executed_completed_count")
    algorithmic_failures = validation.get("executed_algorithmic_failure_count")
    not_applicable = validation.get("not_applicable_count")
    status_counts = validation.get("executed_status_counts")
    record_payload_sha256s = validation.get("record_payload_sha256s")
    terminal_statuses = frozenset(
        {"completed", "failed", "timeout", "resource_exceeded", "unavailable"}
    )
    if (
        validation.get("schema_version") != 1
        or validation.get("status")
        != "eligible_for_final_evaluation_complete_terminal_denominator"
        or validation.get("scope") != "supplementary_trajectory"
        or validation.get("trajectory_plan_sha256") != plan_sha256
        or canonical_sha256(validation_body) != validation_sha256
        or any(
            isinstance(value, bool) or type(value) is not int or value < 0
            for value in (planned, completed, algorithmic_failures, not_applicable)
        )
        or not isinstance(status_counts, Mapping)
        or any(
            status not in terminal_statuses
            or isinstance(count, bool)
            or type(count) is not int
            or count < 0
            for status, count in status_counts.items()
        )
        or not isinstance(record_payload_sha256s, list)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in record_payload_sha256s
        )
    ):
        raise DownstreamEvidenceError("evaluated trajectory validation differs")
    assert isinstance(planned, int)
    assert isinstance(completed, int)
    assert isinstance(algorithmic_failures, int)
    assert isinstance(not_applicable, int)
    assert isinstance(status_counts, Mapping)
    assert isinstance(record_payload_sha256s, list)
    if (
        planned != len(entries)
        or planned != len(record_payload_sha256s)
        or sum(int(count) for count in status_counts.values()) + not_applicable
        != planned
        or int(status_counts.get("completed", 0)) != completed
        or sum(
            int(status_counts.get(status, 0))
            for status in terminal_statuses - {"completed"}
        )
        != algorithmic_failures
    ):
        raise DownstreamEvidenceError(
            "evaluated trajectory validation denominator differs"
        )

    evidence_rows = _trajectory_result_file_rows(
        evidence.get("result_files"),
        "evaluated trajectory result inventory",
        trajectory_only=False,
    )
    cumulative_rows = _trajectory_result_file_rows(
        result_manifest.get("result_files"),
        "evaluated cumulative trajectory result inventory",
        trajectory_only=True,
    )
    evidence_lookup = {row["path"]: row["sha256"] for row in evidence_rows}
    authority_rows = _trajectory_result_file_rows(
        authority.get("files"),
        "evaluated trajectory authority inventory",
        trajectory_only=False,
    )
    if (
        evidence_rows != cumulative_rows
        or tuple(
            row
            for row in evidence_rows
            if row["path"].startswith("results/trajectory/execution_authority/")
        )
        != authority_rows
        or evidence_lookup.get(binding.dataset_file_path) != dataset_file_sha256
        or evidence_lookup.get("results/trajectory/dataset/dataset_receipt.json")
        != receipt_file_sha256
        or evidence_lookup.get(str(authority["authority_path"]))
        != authority_file_sha256
        or evidence_lookup.get(
            "results/trajectory/execution_authority/count_score_authority.json"
        )
        != count_score_authority_file_sha256
        or evidence_lookup.get(
            "results/trajectory/execution_authority/retained_calibration.json"
        )
        != retained_calibration_file_sha256
        or evidence_lookup.get(manifest_path) != manifest_file_sha256
    ):
        raise DownstreamEvidenceError(
            "evaluated trajectory result inventory differs from cumulative receipt"
        )

    try:
        from .final_runner import (
            FinalRunnerContractError,
            _rederive_trajectory_evidence_before_receipt,
        )

        replayed = _rederive_trajectory_evidence_before_receipt(
            repository_root,
            round_root,
            evidence,
            result_manifest.get("result_files"),
            primary_final_plan_sha256=str(result_manifest["final_plan_sha256"]),
        )
    except (FinalRunnerContractError, TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "evaluated trajectory publication replay failed"
        ) from error
    if replayed != evidence:
        raise DownstreamEvidenceError(
            "evaluated trajectory publication replay differs from receipt"
        )

    return {
        "trajectory_evidence_sha256": evidence_sha256,
        "trajectory_plan_sha256": plan_sha256,
        "trajectory_execution_claim_sha256": str(
            input_hashes["execution_claim_sha256"]
        ),
        "trajectory_execution_environment_sha256": str(
            input_hashes["execution_environment_sha256"]
        ),
        "trajectory_dataset_id": binding.dataset_id,
        "trajectory_dataset_sha256": dataset_sha256,
        "trajectory_dataset_file_sha256": dataset_file_sha256,
        "trajectory_dataset_receipt_file_sha256": receipt_file_sha256,
        "trajectory_dataset_receipt_payload_sha256": receipt_payload_sha256,
        "trajectory_source_id": binding.source_id,
        "trajectory_root_cell_id": binding.root_cell_id,
        "trajectory_registered_authority_sha256": binding.authority_sha256,
        "trajectory_registered_binding_sha256": (binding.registered_binding_sha256),
        "trajectory_authority_sha256": authority_sha256,
        "trajectory_authority_file_sha256": authority_file_sha256,
        "trajectory_execution_manifest_path": manifest_path,
        "trajectory_execution_manifest_file_sha256": manifest_file_sha256,
        "trajectory_execution_manifest_payload_sha256": manifest_payload_sha256,
        "trajectory_execution_validation_sha256": validation_sha256,
        "trajectory_record_payload_sha256s_sha256": canonical_sha256(
            record_payload_sha256s
        ),
        "trajectory_status_counts_sha256": canonical_sha256(
            {
                "executed_status_counts": dict(status_counts),
                "not_applicable_count": not_applicable,
            }
        ),
        "trajectory_planned_run_count": planned,
        "trajectory_result_files_sha256": canonical_sha256(evidence_rows),
        "trajectory_result_file_count": len(evidence_rows),
    }


def _read_verified_evaluated_round_binding(
    repository: str | Path,
    round_directory: str | Path,
) -> EvaluatedRoundBinding:
    repository_root = _existing_directory(repository, "evaluated final repository")
    round_root = _existing_directory(round_directory, "evaluated final round")
    try:
        round_root.relative_to(repository_root)
    except ValueError as error:
        raise DownstreamEvidenceError(
            "evaluated final round must be inside its repository"
        ) from error
    validated_receipt = _validated_evaluated_round_receipt(repository_root, round_root)
    receipt_path = round_root / "evaluation_receipt.json"
    receipt, _receipt_raw, receipt_file_sha256 = _strict_json(
        receipt_path, "evaluated final receipt"
    )
    if receipt != validated_receipt:
        raise DownstreamEvidenceError(
            "evaluated final receipt differs from lifecycle validation"
        )
    if set(receipt) != _EVALUATION_RECEIPT_FIELDS:
        raise DownstreamEvidenceError("evaluated final receipt schema differs")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("state") != "evaluated"
        or receipt.get("round_id") != round_root.name
    ):
        raise DownstreamEvidenceError("evaluated final receipt identity differs")
    _text(receipt.get("evaluated_at"), "final evaluation time")
    _text(receipt.get("execution_claim_id"), "final execution claim")
    _digest(receipt.get("seed_manifest_sha256"), "final seed manifest checksum")

    result_manifest = receipt.get("result_manifest")
    if (
        not isinstance(result_manifest, Mapping)
        or set(result_manifest) != _FINAL_RESULT_MANIFEST_FIELDS
    ):
        raise DownstreamEvidenceError("evaluated final result manifest schema differs")
    result_manifest = dict(result_manifest)
    result_manifest_sha256 = _digest(
        receipt.get("result_manifest_sha256"), "final result manifest checksum"
    )
    if canonical_sha256(result_manifest) != result_manifest_sha256:
        raise DownstreamEvidenceError("final result manifest checksum differs")
    if (
        result_manifest.get("schema_version") != 1
        or result_manifest.get("status") != "completed"
    ):
        raise DownstreamEvidenceError("final result manifest is incomplete")
    final_plan_sha256 = _digest(
        result_manifest.get("final_plan_sha256"), "final execution plan checksum"
    )

    validation = result_manifest.get("execution_validation")
    if (
        not isinstance(validation, Mapping)
        or set(validation) != _FINAL_EXECUTION_VALIDATION_FIELDS
    ):
        raise DownstreamEvidenceError("final execution validation schema differs")
    validation = dict(validation)
    validation_sha256 = _digest(
        validation.get("validation_sha256"), "final execution validation checksum"
    )
    validation_body = {
        key: value for key, value in validation.items() if key != "validation_sha256"
    }
    if (
        canonical_sha256(validation_body) != validation_sha256
        or validation.get("schema_version") != 1
        or validation.get("status")
        != "eligible_for_final_evaluation_complete_terminal_denominator"
        or validation.get("final_plan_sha256") != final_plan_sha256
    ):
        raise DownstreamEvidenceError("final execution validation differs")
    validation_planned = validation.get("planned_run_count")
    completed_count = validation.get("executed_completed_count")
    algorithmic_failure_count = validation.get("executed_algorithmic_failure_count")
    not_applicable_count = validation.get("not_applicable_count")
    status_counts = validation.get("executed_status_counts")
    terminal_statuses = frozenset(
        {"completed", "failed", "timeout", "resource_exceeded", "unavailable"}
    )
    if (
        any(
            isinstance(value, bool) or type(value) is not int or value < 0
            for value in (
                validation_planned,
                completed_count,
                algorithmic_failure_count,
                not_applicable_count,
            )
        )
        or not isinstance(status_counts, Mapping)
        or any(
            status not in terminal_statuses
            or isinstance(count, bool)
            or type(count) is not int
            or count < 0
            for status, count in status_counts.items()
        )
    ):
        raise DownstreamEvidenceError("final execution validation denominator differs")
    assert isinstance(validation_planned, int)
    assert isinstance(completed_count, int)
    assert isinstance(algorithmic_failure_count, int)
    assert isinstance(not_applicable_count, int)
    assert isinstance(status_counts, Mapping)
    if (
        sum(int(count) for count in status_counts.values()) + not_applicable_count
        != validation_planned
        or int(status_counts.get("completed", 0)) != completed_count
        or sum(
            int(status_counts.get(status, 0))
            for status in terminal_statuses - {"completed"}
        )
        != algorithmic_failure_count
    ):
        raise DownstreamEvidenceError("final execution validation denominator differs")

    final_manifest_relative = result_manifest.get("final_execution_manifest_path")
    if final_manifest_relative != ("results/final/execution/execution_manifest.json"):
        raise DownstreamEvidenceError("final execution manifest path differs")
    final_manifest_path = _safe_relative(
        round_root, final_manifest_relative, "final execution manifest"
    )
    final_manifest, _manifest_raw, final_manifest_file_sha256 = _strict_json(
        final_manifest_path, "final execution manifest"
    )
    if set(final_manifest) != _FINAL_MANIFEST_FIELDS:
        raise DownstreamEvidenceError("final execution manifest schema differs")
    final_manifest_payload_sha256 = _digest(
        final_manifest.get("manifest_sha256"),
        "final execution manifest payload checksum",
    )
    final_manifest_body = {
        key: value for key, value in final_manifest.items() if key != "manifest_sha256"
    }
    if canonical_sha256(final_manifest_body) != final_manifest_payload_sha256:
        raise DownstreamEvidenceError("final execution manifest checksum differs")
    if (
        result_manifest.get("final_execution_manifest_sha256")
        != final_manifest_file_sha256
        or result_manifest.get("final_execution_payload_sha256")
        != final_manifest_payload_sha256
        or final_manifest.get("plan_sha256") != final_plan_sha256
    ):
        raise DownstreamEvidenceError(
            "final execution manifest differs from evaluation receipt"
        )
    references = final_manifest.get("records")
    payload_sha256s = validation.get("record_payload_sha256s")
    planned = validation_planned
    if (
        not isinstance(references, list)
        or not isinstance(payload_sha256s, list)
        or isinstance(planned, bool)
        or type(planned) is not int
        or planned != len(references)
        or len(payload_sha256s) != len(references)
        or final_manifest.get("planned_run_count") != planned
        or final_manifest.get("recorded_run_count") != planned
    ):
        raise DownstreamEvidenceError("final execution validation denominator differs")
    execution_root = final_manifest_path.parent
    observed_status_counts: dict[str, int] = {}
    observed_not_applicable_count = 0
    for index, (reference, payload_sha256) in enumerate(
        zip(references, payload_sha256s, strict=True), start=1
    ):
        if (
            not isinstance(reference, Mapping)
            or set(reference) != {"ordinal", "run_id", "path", "sha256"}
            or reference.get("ordinal") != index
        ):
            raise DownstreamEvidenceError(
                "final execution validation reference differs"
            )
        record_path = _safe_relative(
            execution_root, reference.get("path"), "final execution record"
        )
        record, _record_raw, record_file_sha256 = _strict_json(
            record_path, "final execution record"
        )
        if record_file_sha256 != _digest(
            reference.get("sha256"), "final record checksum"
        ) or canonical_sha256(record) != _digest(
            payload_sha256, "final record payload checksum"
        ):
            raise DownstreamEvidenceError(
                "final execution validation record binding differs"
            )
        run = record.get("run")
        if not isinstance(run, Mapping) or run.get("status") not in terminal_statuses:
            raise DownstreamEvidenceError(
                "final execution validation record status differs"
            )
        status = str(run["status"])
        if record.get("execution_request") is None:
            if status != "unavailable":
                raise DownstreamEvidenceError(
                    "final non-applicable record status differs"
                )
            observed_not_applicable_count += 1
        else:
            observed_status_counts[status] = observed_status_counts.get(status, 0) + 1
    if (
        observed_not_applicable_count != not_applicable_count
        or observed_status_counts != dict(status_counts)
    ):
        raise DownstreamEvidenceError(
            "final execution validation record denominator differs"
        )

    storage_preflight = result_manifest.get("storage_preflight")
    if not isinstance(storage_preflight, Mapping):
        raise DownstreamEvidenceError("final storage preflight is invalid")
    scaling_binding_fields = _validated_scaling_binding_fields(
        repository_root,
        round_root,
        result_manifest,
    )
    trajectory_binding_fields = _validated_trajectory_binding_fields(
        repository_root,
        round_root,
        result_manifest,
    )
    receipt_after, _receipt_raw_after, receipt_file_sha256_after = _strict_json(
        receipt_path,
        "evaluated final receipt after publication replay",
    )
    if receipt_after != receipt or receipt_file_sha256_after != receipt_file_sha256:
        raise DownstreamEvidenceError(
            "evaluated final receipt changed during publication replay"
        )
    return EvaluatedRoundBinding(
        repository_root=str(repository_root),
        round_root=str(round_root),
        round_id=round_root.name,
        evaluation_receipt_path="evaluation_receipt.json",
        evaluation_receipt_file_sha256=receipt_file_sha256,
        evaluation_receipt_payload_sha256=canonical_sha256(receipt),
        result_manifest_sha256=result_manifest_sha256,
        final_plan_sha256=final_plan_sha256,
        final_execution_manifest_path=str(final_manifest_relative),
        final_execution_manifest_file_sha256=final_manifest_file_sha256,
        final_execution_manifest_payload_sha256=final_manifest_payload_sha256,
        execution_validation_sha256=validation_sha256,
        storage_preflight_sha256=canonical_sha256(dict(storage_preflight)),
        scaling_evidence_sha256=str(scaling_binding_fields["scaling_evidence_sha256"]),
        scaling_plan_sha256=str(scaling_binding_fields["scaling_plan_sha256"]),
        scaling_checkpoint_path=str(scaling_binding_fields["scaling_checkpoint_path"]),
        scaling_checkpoint_file_sha256=str(
            scaling_binding_fields["scaling_checkpoint_file_sha256"]
        ),
        scaling_checkpoint_payload_sha256=str(
            scaling_binding_fields["scaling_checkpoint_payload_sha256"]
        ),
        scaling_checkpoint_history_sha256=str(
            scaling_binding_fields["scaling_checkpoint_history_sha256"]
        ),
        scaling_checkpoint_history_count=int(
            scaling_binding_fields["scaling_checkpoint_history_count"]
        ),
        scaling_result_files_sha256=str(
            scaling_binding_fields["scaling_result_files_sha256"]
        ),
        scaling_result_file_count=int(
            scaling_binding_fields["scaling_result_file_count"]
        ),
        trajectory_evidence_sha256=str(
            trajectory_binding_fields["trajectory_evidence_sha256"]
        ),
        trajectory_plan_sha256=str(trajectory_binding_fields["trajectory_plan_sha256"]),
        trajectory_execution_claim_sha256=str(
            trajectory_binding_fields["trajectory_execution_claim_sha256"]
        ),
        trajectory_execution_environment_sha256=str(
            trajectory_binding_fields["trajectory_execution_environment_sha256"]
        ),
        trajectory_dataset_id=str(trajectory_binding_fields["trajectory_dataset_id"]),
        trajectory_dataset_sha256=str(
            trajectory_binding_fields["trajectory_dataset_sha256"]
        ),
        trajectory_dataset_file_sha256=str(
            trajectory_binding_fields["trajectory_dataset_file_sha256"]
        ),
        trajectory_dataset_receipt_file_sha256=str(
            trajectory_binding_fields["trajectory_dataset_receipt_file_sha256"]
        ),
        trajectory_dataset_receipt_payload_sha256=str(
            trajectory_binding_fields["trajectory_dataset_receipt_payload_sha256"]
        ),
        trajectory_source_id=str(trajectory_binding_fields["trajectory_source_id"]),
        trajectory_root_cell_id=str(
            trajectory_binding_fields["trajectory_root_cell_id"]
        ),
        trajectory_registered_authority_sha256=str(
            trajectory_binding_fields["trajectory_registered_authority_sha256"]
        ),
        trajectory_registered_binding_sha256=str(
            trajectory_binding_fields["trajectory_registered_binding_sha256"]
        ),
        trajectory_authority_sha256=str(
            trajectory_binding_fields["trajectory_authority_sha256"]
        ),
        trajectory_authority_file_sha256=str(
            trajectory_binding_fields["trajectory_authority_file_sha256"]
        ),
        trajectory_execution_manifest_path=str(
            trajectory_binding_fields["trajectory_execution_manifest_path"]
        ),
        trajectory_execution_manifest_file_sha256=str(
            trajectory_binding_fields["trajectory_execution_manifest_file_sha256"]
        ),
        trajectory_execution_manifest_payload_sha256=str(
            trajectory_binding_fields["trajectory_execution_manifest_payload_sha256"]
        ),
        trajectory_execution_validation_sha256=str(
            trajectory_binding_fields["trajectory_execution_validation_sha256"]
        ),
        trajectory_record_payload_sha256s_sha256=str(
            trajectory_binding_fields["trajectory_record_payload_sha256s_sha256"]
        ),
        trajectory_status_counts_sha256=str(
            trajectory_binding_fields["trajectory_status_counts_sha256"]
        ),
        trajectory_planned_run_count=int(
            trajectory_binding_fields["trajectory_planned_run_count"]
        ),
        trajectory_result_files_sha256=str(
            trajectory_binding_fields["trajectory_result_files_sha256"]
        ),
        trajectory_result_file_count=int(
            trajectory_binding_fields["trajectory_result_file_count"]
        ),
    )


def _validate_evaluated_round_binding(binding: EvaluatedRoundBinding) -> None:
    if not isinstance(binding, EvaluatedRoundBinding):
        raise TypeError("binding must be an EvaluatedRoundBinding")
    observed = _read_verified_evaluated_round_binding(
        binding.repository_root, binding.round_root
    )
    if observed != binding:
        raise DownstreamEvidenceError("evaluated final round binding changed")


def _evaluated_round_binding_from_payload(
    value: object,
) -> EvaluatedRoundBinding | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _EVALUATED_ROUND_BINDING_FIELDS:
        raise DownstreamEvidenceError(
            "persisted evaluated-round binding schema differs"
        )
    repository_root = _text(
        value.get("repository_root"), "evaluated-round repository root"
    )
    round_root = _text(value.get("round_root"), "evaluated-round root")
    if not Path(repository_root).is_absolute() or not Path(round_root).is_absolute():
        raise DownstreamEvidenceError(
            "persisted evaluated-round paths are not absolute"
        )
    binding = EvaluatedRoundBinding(
        repository_root=repository_root,
        round_root=round_root,
        round_id=_text(value.get("round_id"), "evaluated-round ID"),
        evaluation_receipt_path=_text(
            value.get("evaluation_receipt_path"), "evaluation receipt path"
        ),
        evaluation_receipt_file_sha256=_digest(
            value.get("evaluation_receipt_file_sha256"),
            "evaluation receipt file checksum",
        ),
        evaluation_receipt_payload_sha256=_digest(
            value.get("evaluation_receipt_payload_sha256"),
            "evaluation receipt payload checksum",
        ),
        result_manifest_sha256=_digest(
            value.get("result_manifest_sha256"), "result manifest checksum"
        ),
        final_plan_sha256=_digest(
            value.get("final_plan_sha256"), "final plan checksum"
        ),
        final_execution_manifest_path=_text(
            value.get("final_execution_manifest_path"),
            "final execution manifest path",
        ),
        final_execution_manifest_file_sha256=_digest(
            value.get("final_execution_manifest_file_sha256"),
            "final execution manifest file checksum",
        ),
        final_execution_manifest_payload_sha256=_digest(
            value.get("final_execution_manifest_payload_sha256"),
            "final execution manifest payload checksum",
        ),
        execution_validation_sha256=_digest(
            value.get("execution_validation_sha256"),
            "execution validation checksum",
        ),
        storage_preflight_sha256=_digest(
            value.get("storage_preflight_sha256"),
            "storage preflight checksum",
        ),
        scaling_evidence_sha256=_digest(
            value.get("scaling_evidence_sha256"),
            "scaling evidence checksum",
        ),
        scaling_plan_sha256=_digest(
            value.get("scaling_plan_sha256"),
            "scaling plan checksum",
        ),
        scaling_checkpoint_path=_text(
            value.get("scaling_checkpoint_path"),
            "scaling checkpoint path",
        ),
        scaling_checkpoint_file_sha256=_digest(
            value.get("scaling_checkpoint_file_sha256"),
            "scaling checkpoint file checksum",
        ),
        scaling_checkpoint_payload_sha256=_digest(
            value.get("scaling_checkpoint_payload_sha256"),
            "scaling checkpoint payload checksum",
        ),
        scaling_checkpoint_history_sha256=_digest(
            value.get("scaling_checkpoint_history_sha256"),
            "scaling checkpoint history checksum",
        ),
        scaling_checkpoint_history_count=value.get("scaling_checkpoint_history_count"),
        scaling_result_files_sha256=_digest(
            value.get("scaling_result_files_sha256"),
            "scaling result inventory checksum",
        ),
        scaling_result_file_count=value.get("scaling_result_file_count"),
        trajectory_evidence_sha256=_digest(
            value.get("trajectory_evidence_sha256"),
            "trajectory evidence checksum",
        ),
        trajectory_plan_sha256=_digest(
            value.get("trajectory_plan_sha256"),
            "trajectory plan checksum",
        ),
        trajectory_execution_claim_sha256=_digest(
            value.get("trajectory_execution_claim_sha256"),
            "trajectory execution claim checksum",
        ),
        trajectory_execution_environment_sha256=_digest(
            value.get("trajectory_execution_environment_sha256"),
            "trajectory execution environment checksum",
        ),
        trajectory_dataset_id=_text(
            value.get("trajectory_dataset_id"),
            "trajectory dataset ID",
        ),
        trajectory_dataset_sha256=_digest(
            value.get("trajectory_dataset_sha256"),
            "trajectory dataset checksum",
        ),
        trajectory_dataset_file_sha256=_digest(
            value.get("trajectory_dataset_file_sha256"),
            "trajectory dataset file checksum",
        ),
        trajectory_dataset_receipt_file_sha256=_digest(
            value.get("trajectory_dataset_receipt_file_sha256"),
            "trajectory dataset receipt file checksum",
        ),
        trajectory_dataset_receipt_payload_sha256=_digest(
            value.get("trajectory_dataset_receipt_payload_sha256"),
            "trajectory dataset receipt payload checksum",
        ),
        trajectory_source_id=_text(
            value.get("trajectory_source_id"),
            "trajectory source ID",
        ),
        trajectory_root_cell_id=_text(
            value.get("trajectory_root_cell_id"),
            "trajectory root cell ID",
        ),
        trajectory_registered_authority_sha256=_digest(
            value.get("trajectory_registered_authority_sha256"),
            "trajectory registered authority checksum",
        ),
        trajectory_registered_binding_sha256=_digest(
            value.get("trajectory_registered_binding_sha256"),
            "trajectory registered binding checksum",
        ),
        trajectory_authority_sha256=_digest(
            value.get("trajectory_authority_sha256"),
            "trajectory authority checksum",
        ),
        trajectory_authority_file_sha256=_digest(
            value.get("trajectory_authority_file_sha256"),
            "trajectory authority file checksum",
        ),
        trajectory_execution_manifest_path=_text(
            value.get("trajectory_execution_manifest_path"),
            "trajectory execution manifest path",
        ),
        trajectory_execution_manifest_file_sha256=_digest(
            value.get("trajectory_execution_manifest_file_sha256"),
            "trajectory execution manifest file checksum",
        ),
        trajectory_execution_manifest_payload_sha256=_digest(
            value.get("trajectory_execution_manifest_payload_sha256"),
            "trajectory execution manifest payload checksum",
        ),
        trajectory_execution_validation_sha256=_digest(
            value.get("trajectory_execution_validation_sha256"),
            "trajectory execution validation checksum",
        ),
        trajectory_record_payload_sha256s_sha256=_digest(
            value.get("trajectory_record_payload_sha256s_sha256"),
            "trajectory record payload inventory checksum",
        ),
        trajectory_status_counts_sha256=_digest(
            value.get("trajectory_status_counts_sha256"),
            "trajectory status counts checksum",
        ),
        trajectory_planned_run_count=value.get("trajectory_planned_run_count"),
        trajectory_result_files_sha256=_digest(
            value.get("trajectory_result_files_sha256"),
            "trajectory result inventory checksum",
        ),
        trajectory_result_file_count=value.get("trajectory_result_file_count"),
    )
    if (
        binding.evaluation_receipt_path != "evaluation_receipt.json"
        or binding.final_execution_manifest_path
        != "results/final/execution/execution_manifest.json"
        or Path(binding.round_root).name != binding.round_id
        or _SCALING_CHECKPOINT_PATH.fullmatch(binding.scaling_checkpoint_path) is None
        or type(binding.scaling_checkpoint_history_count) is not int
        or binding.scaling_checkpoint_history_count <= 0
        or type(binding.scaling_result_file_count) is not int
        or binding.scaling_result_file_count < binding.scaling_checkpoint_history_count
        or binding.trajectory_execution_manifest_path
        != "results/trajectory/execution/execution_manifest.json"
        or type(binding.trajectory_planned_run_count) is not int
        or binding.trajectory_planned_run_count <= 0
        or type(binding.trajectory_result_file_count) is not int
        or binding.trajectory_result_file_count < binding.trajectory_planned_run_count
    ):
        raise DownstreamEvidenceError(
            "persisted evaluated-round binding identity differs"
        )
    return binding


def _configuration_from_payload(value: object) -> AuthorizedConfiguration:
    if not isinstance(value, Mapping) or set(value) != _CONFIGURATION_FIELDS:
        raise DownstreamEvidenceError("persisted configuration schema differs")
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise DownstreamEvidenceError("persisted configuration payload is invalid")
    try:
        configuration = AuthorizedConfiguration.create(
            method_id=value.get("method_id"),
            configuration_id=value.get("configuration_id"),
            kind=value.get("kind"),
            payload=dict(payload),
            requires_count_score=value.get("requires_count_score"),
            requires_calibration=value.get("requires_calibration"),
            configuration_sha256=value.get("configuration_sha256"),
        )
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "persisted configuration authority is invalid"
        ) from error
    if _legacy_configuration_payload(configuration) != dict(value):
        raise DownstreamEvidenceError("persisted configuration authority differs")
    _method_artifact_sha256(configuration)
    return configuration


def _dataset_binding_from_payload(value: object) -> DatasetEvidenceBinding:
    if not isinstance(value, Mapping) or set(value) != _DATASET_BINDING_FIELDS:
        raise DownstreamEvidenceError("persisted dataset binding schema differs")
    retained = value.get("retained_cell_ids")
    genes = value.get("gene_ids")
    string_fields = (
        "dataset_id",
        "path",
        "file_sha256",
        "dataset_sha256",
        "mechanism",
        "biological_id",
        "technical_view",
        "method_input_sha256",
        "dataset_qc_policy_sha256",
        "excluded_cell_ids_sha256",
        "retained_cell_ids_sha256",
    )
    if (
        not isinstance(retained, list)
        or not isinstance(genes, list)
        or any(not isinstance(value.get(field), str) for field in string_fields)
        or (
            value.get("trajectory_root_cell_id") is not None
            and not isinstance(value.get("trajectory_root_cell_id"), str)
        )
        or (
            value.get("trajectory_source_id") is not None
            and not isinstance(value.get("trajectory_source_id"), str)
        )
        or (
            value.get("trajectory_authority_sha256") is not None
            and not isinstance(value.get("trajectory_authority_sha256"), str)
        )
        or (
            value.get("trajectory_binding_sha256") is not None
            and not isinstance(value.get("trajectory_binding_sha256"), str)
        )
    ):
        raise DownstreamEvidenceError("persisted dataset IDs are invalid")
    try:
        return DatasetEvidenceBinding(
            dataset_id=value["dataset_id"],
            path=value["path"],
            file_sha256=value["file_sha256"],
            dataset_sha256=value["dataset_sha256"],
            mechanism=value["mechanism"],
            biological_id=value["biological_id"],
            technical_view=value["technical_view"],
            method_input_sha256=value["method_input_sha256"],
            dataset_qc_policy_sha256=value["dataset_qc_policy_sha256"],
            excluded_cell_count=value["excluded_cell_count"],
            excluded_cell_ids_sha256=value["excluded_cell_ids_sha256"],
            retained_cell_count=value["retained_cell_count"],
            retained_cell_ids_sha256=value["retained_cell_ids_sha256"],
            retained_gene_count=value["retained_gene_count"],
            observed_zero_count=value["observed_zero_count"],
            retained_cell_ids=tuple(retained),
            gene_ids=tuple(genes),
            trajectory_root_cell_id=(
                None
                if value["trajectory_root_cell_id"] is None
                else value["trajectory_root_cell_id"]
            ),
            trajectory_source_id=(
                None
                if value["trajectory_source_id"] is None
                else value["trajectory_source_id"]
            ),
            trajectory_authority_sha256=(
                None
                if value["trajectory_authority_sha256"] is None
                else value["trajectory_authority_sha256"]
            ),
            trajectory_binding_sha256=(
                None
                if value["trajectory_binding_sha256"] is None
                else value["trajectory_binding_sha256"]
            ),
        )
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError("persisted dataset binding is invalid") from error


def _development_source_binding_from_payload(
    value: object,
) -> DevelopmentSourceBinding:
    if (
        not isinstance(value, Mapping)
        or set(value) != _DEVELOPMENT_SOURCE_BINDING_FIELDS
    ):
        raise DownstreamEvidenceError(
            "persisted development source binding schema differs"
        )
    methods = value.get("selected_methods")
    count = value.get("planned_denominator_count")
    if (
        not isinstance(methods, list)
        or not methods
        or methods != sorted(set(methods))
        or any(not isinstance(item, str) or not item for item in methods)
        or isinstance(count, bool)
        or type(count) is not int
        or count <= 0
    ):
        raise DownstreamEvidenceError(
            "persisted development source denominator is invalid"
        )
    for name in (
        "manifest_file_sha256",
        "manifest_payload_sha256",
        "plan_sha256",
        "input_hashes_sha256",
        "statuses_sha256",
        "denominator_sha256",
        "evaluation_manifest_file_sha256",
        "evaluation_manifest_payload_sha256",
        "evaluation_source_sha256",
    ):
        _digest(value.get(name), f"persisted development source {name}")
    for name in (
        "source_id",
        "source_root",
        "manifest_path",
        "evaluation_manifest_path",
        "evaluation_source_pointer",
    ):
        _text(value.get(name), f"persisted development source {name}")
    return DevelopmentSourceBinding(
        source_id=value["source_id"],
        source_root=value["source_root"],
        selected_methods=tuple(methods),
        manifest_path=value["manifest_path"],
        manifest_file_sha256=value["manifest_file_sha256"],
        manifest_payload_sha256=value["manifest_payload_sha256"],
        plan_sha256=value["plan_sha256"],
        input_hashes_sha256=value["input_hashes_sha256"],
        statuses_sha256=value["statuses_sha256"],
        denominator_sha256=value["denominator_sha256"],
        planned_denominator_count=count,
        evaluation_manifest_path=value["evaluation_manifest_path"],
        evaluation_manifest_file_sha256=value["evaluation_manifest_file_sha256"],
        evaluation_manifest_payload_sha256=value["evaluation_manifest_payload_sha256"],
        evaluation_source_pointer=value["evaluation_source_pointer"],
        evaluation_source_sha256=value["evaluation_source_sha256"],
    )


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    ordinal: int
    path: str
    sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _SourceBundle:
    manifest_path: str
    manifest_file_sha256: str
    manifest_payload_sha256: str
    source_plan_sha256: str
    source_input_hashes_sha256: str
    records: tuple[_SourceRecord, ...]
    execution_validation: Mapping[str, object] | None = None


def _source_input_hashes_sha256(payload: Mapping[str, object]) -> str:
    values = payload.get("input_hashes")
    if not isinstance(values, Mapping) or not values:
        raise DownstreamEvidenceError("source input authority is absent")
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise DownstreamEvidenceError("source input authority name is invalid")
        _digest(value, f"source input {name}")
    return canonical_sha256(dict(values))


def _validate_source_record_schema(
    record: Mapping[str, object], *, source_kind: str
) -> None:
    # Single compatibility seam with CheckpointStore/FinalResultStore.  Keep this
    # exact adapter synchronized with their current persisted-record contract.
    final_source = source_kind == "final"
    expected_record_fields = (
        {"run", "metrics", "p_pre_zero_evidence", "execution_request"}
        if final_source
        else {"run", "metrics", "p_pre_zero_evidence"}
    )
    if set(record) != expected_record_fields:
        raise DownstreamEvidenceError(f"{source_kind} source record schema differs")
    run = record.get("run")
    expected_run_fields = _FINAL_RUN_FIELDS if final_source else _DEVELOPMENT_RUN_FIELDS
    if not isinstance(run, Mapping) or set(run) != expected_run_fields:
        raise DownstreamEvidenceError("source run schema differs")
    metrics = record.get("metrics")
    if not isinstance(metrics, list) or any(
        not isinstance(metric, Mapping) or set(metric) != _METRIC_FIELDS
        for metric in metrics
    ):
        raise DownstreamEvidenceError("source metric schema differs")
    _validate_prezero_source_schema(record.get("p_pre_zero_evidence"), run=run)
    if final_source:
        request = record.get("execution_request")
        if request is not None and (
            not isinstance(request, Mapping)
            or set(request) != _FINAL_EXECUTION_REQUEST_FIELDS
        ):
            raise DownstreamEvidenceError("final execution request schema differs")
        native_retention = run.get("native_output_retention")
        expected_native_retention = (
            "omitted_redundant_final_output"
            if run.get("native_output_sha256") is not None
            else "not_available"
        )
        if native_retention != expected_native_retention or any(
            run.get(f"native_output_{suffix}") is not None
            for suffix in ("path", "file_sha256", "shape", "dtype")
        ):
            raise DownstreamEvidenceError("final native output storage policy differs")


def _validate_prezero_source_schema(
    value: object, *, run: Mapping[str, object]
) -> None:
    """Validate the exact persisted envelope; semantic revalidation stays upstream."""

    if (
        not isinstance(value, Mapping)
        or set(value) != _PREZERO_EVIDENCE_FIELDS
        or value.get("schema_version") != 1
    ):
        raise DownstreamEvidenceError("source p_pre_zero evidence schema differs")
    identity = value.get("identity")
    expected_identity = {name: run.get(name) for name in _PREZERO_IDENTITY_FIELDS}
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _PREZERO_IDENTITY_FIELDS
        or dict(identity) != expected_identity
    ):
        raise DownstreamEvidenceError("source p_pre_zero identity differs from run")
    matrix = value.get("matrix")
    if not isinstance(matrix, Mapping) or set(matrix) != _PREZERO_MATRIX_FIELDS:
        raise DownstreamEvidenceError("source p_pre_zero matrix schema differs")
    storage = value.get("storage")
    if not isinstance(storage, Mapping) or set(storage) != _PREZERO_STORAGE_FIELDS:
        raise DownstreamEvidenceError("source p_pre_zero storage schema differs")
    policy = value.get("policy")
    policy_sha256 = value.get("policy_sha256")
    if policy is None:
        if policy_sha256 is not None:
            raise DownstreamEvidenceError("source p_pre_zero policy receipt is partial")
    else:
        if (
            not isinstance(policy, Mapping)
            or set(policy) != _PREZERO_POLICY_FIELDS
            or policy.get("schema_version") != 2
        ):
            raise DownstreamEvidenceError(
                "source p_pre_zero score policy schema differs"
            )
        if policy_sha256 != canonical_sha256(dict(policy)):
            raise DownstreamEvidenceError("source p_pre_zero policy checksum differs")
        for name in (
            "score_artifact_sha256",
            "score_input_sha256",
            "score_config_sha256",
            "calibration_file_sha256",
            "calibration_payload_sha256",
        ):
            _digest(policy.get(name), f"source p_pre_zero policy {name}")
    body = {
        name: nested
        for name, nested in value.items()
        if name not in {"evidence_sha256", "storage"}
    }
    if value.get("evidence_sha256") != canonical_sha256(body):
        raise DownstreamEvidenceError("source p_pre_zero evidence checksum differs")


def _development_source(
    root: Path,
) -> _SourceBundle:
    path = root / "checkpoint.json"
    payload, _raw, file_sha = _strict_json(path, "development checkpoint")
    if set(payload) != _DEVELOPMENT_CHECKPOINT_FIELDS:
        raise DownstreamEvidenceError("development checkpoint schema differs")
    checksum = _digest(payload.get("checkpoint_sha256"), "checkpoint checksum")
    unsigned = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    if canonical_sha256(unsigned) != checksum:
        raise DownstreamEvidenceError("development checkpoint checksum differs")
    records = payload.get("records")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "completed"
        or not isinstance(records, list)
        or payload.get("planned_run_count") != len(records)
    ):
        raise DownstreamEvidenceError("development checkpoint is incomplete")
    result: list[_SourceRecord] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise DownstreamEvidenceError("development source record is malformed")
        _validate_source_record_schema(record, source_kind="development")
        result.append(
            _SourceRecord(
                ordinal=index,
                path=f"checkpoint.json#records/{index - 1}",
                sha256=canonical_sha256(record),
                payload=MappingProxyType(record),
            )
        )
    return _SourceBundle(
        manifest_path="checkpoint.json",
        manifest_file_sha256=file_sha,
        manifest_payload_sha256=checksum,
        source_plan_sha256=_digest(
            payload.get("plan_sha256"), "development source plan checksum"
        ),
        source_input_hashes_sha256=_source_input_hashes_sha256(payload),
        records=tuple(result),
    )


def _final_source(root: Path) -> _SourceBundle:
    path = root / "execution_manifest.json"
    payload, _raw, file_sha = _strict_json(path, "final execution manifest")
    if set(payload) != _FINAL_MANIFEST_FIELDS:
        raise DownstreamEvidenceError("final execution manifest schema differs")
    checksum = _digest(payload.get("manifest_sha256"), "final manifest checksum")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    if canonical_sha256(unsigned) != checksum:
        raise DownstreamEvidenceError("final execution manifest checksum differs")
    references = payload.get("records")
    storage = payload.get("artifact_storage")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "completed"
        or not isinstance(references, list)
        or payload.get("planned_run_count") != len(references)
        or payload.get("recorded_run_count") != len(references)
        or not isinstance(storage, Mapping)
        or dict(storage) != dict(_FINAL_STORAGE_POLICY)
    ):
        if isinstance(storage, Mapping) and dict(storage) != dict(
            _FINAL_STORAGE_POLICY
        ):
            raise DownstreamEvidenceError("final artifact storage policy differs")
        raise DownstreamEvidenceError("final execution manifest is incomplete")
    result: list[_SourceRecord] = []
    for expected_ordinal, reference in enumerate(references, start=1):
        if not isinstance(reference, Mapping) or set(reference) != {
            "ordinal",
            "run_id",
            "path",
            "sha256",
        }:
            raise DownstreamEvidenceError("final source reference is malformed")
        ordinal = reference.get("ordinal")
        if ordinal != expected_ordinal:
            raise DownstreamEvidenceError("final source records are not ordered")
        record_path = _safe_relative(root, reference.get("path"), "final record")
        record, _record_raw, record_file_sha = _strict_json(
            record_path, "final execution record"
        )
        _validate_source_record_schema(record, source_kind="final")
        if record_file_sha != _digest(reference.get("sha256"), "final record checksum"):
            raise DownstreamEvidenceError("final record raw checksum differs")
        run = record.get("run")
        if not isinstance(run, Mapping) or run.get("run_id") != reference.get("run_id"):
            raise DownstreamEvidenceError("final record identity differs")
        result.append(
            _SourceRecord(
                ordinal=expected_ordinal,
                path=str(reference["path"]),
                sha256=record_file_sha,
                payload=MappingProxyType(record),
            )
        )
    return _SourceBundle(
        manifest_path="execution_manifest.json",
        manifest_file_sha256=file_sha,
        manifest_payload_sha256=checksum,
        source_plan_sha256=_digest(
            payload.get("plan_sha256"), "final source plan checksum"
        ),
        source_input_hashes_sha256=_source_input_hashes_sha256(payload),
        records=tuple(result),
    )


def _trajectory_source(root: Path, source_plan: object) -> _SourceBundle:
    """Read one exact trajectory manifest against its rebuilt typed plan."""

    from .final_runner import (
        FinalRunnerContractError,
        TrajectoryExecutionPlan,
        trajectory_execution_plan_payload,
        validate_trajectory_execution_for_evaluation,
    )
    from .runner import DEVELOPMENT_MODEL_SEEDS

    if not isinstance(source_plan, TrajectoryExecutionPlan):
        raise DownstreamEvidenceError(
            "trajectory source requires a rebuilt TrajectoryExecutionPlan"
        )
    path = root / "execution_manifest.json"
    payload, _raw, file_sha = _strict_json(path, "trajectory execution manifest")
    if set(payload) != _TRAJECTORY_MANIFEST_FIELDS:
        raise DownstreamEvidenceError("trajectory execution manifest schema differs")
    checksum = _digest(payload.get("manifest_sha256"), "trajectory manifest checksum")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    expected_plan = trajectory_execution_plan_payload(source_plan)
    references = payload.get("records")
    storage = payload.get("artifact_storage")
    if (
        canonical_sha256(unsigned) != checksum
        or payload.get("schema_version") != 1
        or payload.get("status") != "completed"
        or payload.get("scope") != "supplementary_trajectory"
        or payload.get("plan_sha256") != source_plan.plan_sha256
        or payload.get("input_hashes") != dict(source_plan.input_hashes)
        or payload.get("plan_entries") != expected_plan["entries"]
        or payload.get("configurations") != expected_plan["configurations"]
        or payload.get("model_seed_policy") != list(DEVELOPMENT_MODEL_SEEDS)
        or not isinstance(references, list)
        or payload.get("planned_run_count") != len(source_plan.entries)
        or payload.get("recorded_run_count") != len(source_plan.entries)
        or len(references) != len(source_plan.entries)
        or not isinstance(storage, Mapping)
        or dict(storage) != dict(_FINAL_STORAGE_POLICY)
    ):
        if isinstance(storage, Mapping) and dict(storage) != dict(
            _FINAL_STORAGE_POLICY
        ):
            raise DownstreamEvidenceError("trajectory artifact storage policy differs")
        if (
            payload.get("plan_entries") != expected_plan["entries"]
            or payload.get("configurations") != expected_plan["configurations"]
            or payload.get("model_seed_policy") != list(DEVELOPMENT_MODEL_SEEDS)
        ):
            raise DownstreamEvidenceError("trajectory execution manifest plan differs")
        raise DownstreamEvidenceError("trajectory execution manifest is incomplete")

    result: list[_SourceRecord] = []
    record_payloads: list[Mapping[str, object]] = []
    for expected_ordinal, reference in enumerate(references, start=1):
        if not isinstance(reference, Mapping) or set(reference) != {
            "ordinal",
            "run_id",
            "path",
            "sha256",
        }:
            raise DownstreamEvidenceError("trajectory source reference is malformed")
        if reference.get("ordinal") != expected_ordinal:
            raise DownstreamEvidenceError("trajectory source records are not ordered")
        record_path = _safe_relative(root, reference.get("path"), "trajectory record")
        record, _record_raw, record_file_sha = _strict_json(
            record_path, "trajectory execution record"
        )
        _validate_source_record_schema(record, source_kind="final")
        if record_file_sha != _digest(
            reference.get("sha256"), "trajectory record checksum"
        ):
            raise DownstreamEvidenceError("trajectory record raw checksum differs")
        run = record.get("run")
        if not isinstance(run, Mapping) or run.get("run_id") != reference.get("run_id"):
            raise DownstreamEvidenceError("trajectory record identity differs")
        result.append(
            _SourceRecord(
                ordinal=expected_ordinal,
                path=str(reference["path"]),
                sha256=record_file_sha,
                payload=MappingProxyType(record),
            )
        )
        record_payloads.append(record)
    bundle = _SourceBundle(
        manifest_path="execution_manifest.json",
        manifest_file_sha256=file_sha,
        manifest_payload_sha256=checksum,
        source_plan_sha256=source_plan.plan_sha256,
        source_input_hashes_sha256=_source_input_hashes_sha256(payload),
        records=tuple(result),
    )
    _validate_independent_source_plan(source_plan, bundle)
    try:
        validation = validate_trajectory_execution_for_evaluation(
            source_plan, tuple(record_payloads)
        )
    except (FinalRunnerContractError, TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "trajectory source terminal denominator differs"
        ) from error
    return replace(
        bundle,
        execution_validation=MappingProxyType(validation),
    )


_SOURCE_PLAN_RUN_FIELDS = (
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
    "configuration_kind",
    "requires_count_score",
    "requires_calibration",
)


def _validate_independent_source_plan(
    source_plan: object, bundle: _SourceBundle
) -> None:
    plan_sha256 = getattr(source_plan, "plan_sha256", None)
    input_hashes = getattr(source_plan, "input_hashes", None)
    raw_entries = getattr(source_plan, "entries", None)
    if (
        plan_sha256 != bundle.source_plan_sha256
        or not isinstance(input_hashes, Mapping)
        or canonical_sha256(dict(input_hashes)) != bundle.source_input_hashes_sha256
        or isinstance(raw_entries, (str, bytes))
        or not isinstance(raw_entries, Sequence)
        or len(raw_entries) != len(bundle.records)
    ):
        raise DownstreamEvidenceError("source plan/input authority differs")
    for ordinal, (expected, source) in enumerate(
        zip(raw_entries, bundle.records, strict=True), start=1
    ):
        expected_run = getattr(expected, "run", expected)
        if hasattr(expected_run, "to_dict"):
            expected_payload = expected_run.to_dict()
        elif isinstance(expected_run, Mapping):
            expected_payload = dict(expected_run)
        else:
            raise DownstreamEvidenceError("source plan entry authority is invalid")
        observed = source.payload.get("run")
        if (
            not isinstance(expected_payload, Mapping)
            or not isinstance(observed, Mapping)
            or expected_payload.get("ordinal") != ordinal
            or any(
                observed.get(name) != expected_payload.get(name)
                for name in _SOURCE_PLAN_RUN_FIELDS
            )
        ):
            raise DownstreamEvidenceError(
                "source run differs from source plan authority"
            )


def _optional_model_seed(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise DownstreamEvidenceError(
            "model_seed must be a nonnegative integer or null"
        )
    return value


def _output_shape(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or item <= 0 for item in value)
    ):
        raise DownstreamEvidenceError("evaluator output shape is invalid")
    return int(value[0]), int(value[1])


def _zlib_compress_bound(uncompressed_nbytes: int) -> int:
    return (
        uncompressed_nbytes
        + (uncompressed_nbytes >> 12)
        + (uncompressed_nbytes >> 14)
        + (uncompressed_nbytes >> 25)
        + 13
    )


def _validated_plan_entry(
    source: _SourceRecord,
    *,
    source_kind: str,
    source_root: Path,
    datasets: Mapping[str, DatasetEvidenceBinding],
    configurations: Mapping[tuple[str, str, str, str], AuthorizedConfiguration],
) -> DownstreamPlanEntry:
    run = source.payload.get("run")
    if not isinstance(run, Mapping):
        raise DownstreamEvidenceError("source record run is malformed")
    method_id = _text(run.get("method_id"), "method_id")
    dataset_id = _text(run.get("dataset_id"), "run dataset_id")
    binding = datasets.get(dataset_id)
    if binding is None:
        raise DownstreamEvidenceError(
            f"evaluator dataset binding is absent for {dataset_id}"
        )
    source_dataset_sha256 = _digest(
        run.get("source_dataset_sha256"), "run source dataset checksum"
    )
    if source_dataset_sha256 != binding.dataset_sha256:
        raise DownstreamEvidenceError("run dataset semantic checksum differs")
    expected_dataset_authority = {
        "mechanism": binding.mechanism,
        "biological_id": binding.biological_id,
        "technical_view": binding.technical_view,
        "method_input_sha256": binding.method_input_sha256,
        "dataset_qc_policy_sha256": binding.dataset_qc_policy_sha256,
        "excluded_cell_count": binding.excluded_cell_count,
        "excluded_cell_ids_sha256": binding.excluded_cell_ids_sha256,
        "retained_cell_count": binding.retained_cell_count,
        "retained_cell_ids_sha256": binding.retained_cell_ids_sha256,
        "retained_gene_count": binding.retained_gene_count,
        "observed_zero_count": binding.observed_zero_count,
    }
    if any(
        run.get(name) != expected
        for name, expected in expected_dataset_authority.items()
    ):
        raise DownstreamEvidenceError("run dataset authority differs")
    retained_cell_ids_sha256 = _digest(
        run.get("retained_cell_ids_sha256"), "retained cell checksum"
    )
    if retained_cell_ids_sha256 != binding.retained_cell_ids_sha256:
        raise DownstreamEvidenceError("run retained cell identity differs")
    configuration_id = _text(run.get("configuration_id"), "configuration_id")
    configuration_sha256 = _digest(
        run.get("configuration_sha256"), "configuration checksum"
    )
    configuration_kind = _text(run.get("configuration_kind"), "configuration_kind")
    configuration = configurations.get(
        (
            method_id,
            configuration_id,
            configuration_kind,
            configuration_sha256,
        )
    )
    if configuration is None:
        raise DownstreamEvidenceError("run configuration authority differs")
    method_artifact_sha256 = _method_artifact_sha256(configuration)

    status = _text(run.get("status"), "run status")
    if status not in _RUN_STATUSES:
        raise DownstreamEvidenceError("run status is not recognized")
    reason = run.get("reason")
    if status == "completed":
        if reason is not None:
            raise DownstreamEvidenceError("completed run cannot have a reason")
    elif not isinstance(reason, str) or not reason:
        raise DownstreamEvidenceError("noncompleted run must preserve a reason")

    output_path: str | None
    output_file_sha: str | None
    output_sha: str | None
    shape: tuple[int, int] | None
    encoding: str | None
    uncompressed_nbytes: int | None
    uncompressed_sha: str | None
    if status == "completed":
        output_path = _text(run.get("evaluator_output_path"), "evaluator output path")
        output_file_sha = _digest(
            run.get("evaluator_output_file_sha256"), "evaluator output file checksum"
        )
        output_sha = _digest(
            run.get("evaluator_output_sha256"), "evaluator output content checksum"
        )
        shape = _output_shape(run.get("evaluator_output_shape"))
        if shape != (len(binding.retained_cell_ids), len(binding.gene_ids)):
            raise DownstreamEvidenceError("evaluator output shape differs from dataset")
        if run.get("evaluator_output_dtype") != "<f8":
            raise DownstreamEvidenceError("evaluator output dtype differs")
        if run.get("evaluator_scale") != "log2_cp10k_plus_1":
            raise DownstreamEvidenceError("evaluator output scale differs")
        artifact = _safe_relative(source_root, output_path, "evaluator output")
        observed_file_sha = _stable_file_sha256(artifact, "evaluator output")
        if observed_file_sha != output_file_sha:
            raise DownstreamEvidenceError("evaluator output file checksum differs")
        if source_kind == "development":
            encoding = None
            uncompressed_nbytes = None
            uncompressed_sha = None
            if (
                _regular_file(artifact, "evaluator output").st_size
                != shape[0] * shape[1] * 8
            ):
                raise DownstreamEvidenceError(
                    "development evaluator output byte size differs"
                )
        else:
            encoding = _text(
                run.get("evaluator_output_encoding"), "evaluator output encoding"
            )
            if encoding != _FINAL_OUTPUT_ENCODING or not output_path.endswith(
                ".log2-cp10k-f64.zlib"
            ):
                raise DownstreamEvidenceError("final evaluator output encoding differs")
            uncompressed_nbytes = run.get("evaluator_output_uncompressed_nbytes")
            if (
                type(uncompressed_nbytes) is not int
                or uncompressed_nbytes != shape[0] * shape[1] * 8
            ):
                raise DownstreamEvidenceError(
                    "final evaluator output byte count differs"
                )
            uncompressed_sha = _digest(
                run.get("evaluator_output_uncompressed_sha256"),
                "final evaluator uncompressed checksum",
            )
            if _regular_file(
                artifact, "evaluator output"
            ).st_size > _zlib_compress_bound(uncompressed_nbytes):
                raise DownstreamEvidenceError("final evaluator output exceeds bound")
    else:
        output_path = None
        output_file_sha = None
        output_sha = None
        shape = None
        encoding = None
        uncompressed_nbytes = None
        uncompressed_sha = None
        required_null = (
            "evaluator_output_path",
            "evaluator_output_file_sha256",
            "evaluator_output_shape",
            "evaluator_output_dtype",
            "evaluator_scale",
            "evaluator_output_sha256",
        )
        if any(run.get(name) is not None for name in required_null):
            raise DownstreamEvidenceError(
                "noncompleted evaluator output binding is partial"
            )
        if source_kind == "final" and any(
            run.get(name) is not None
            for name in (
                "evaluator_output_encoding",
                "evaluator_output_uncompressed_nbytes",
                "evaluator_output_uncompressed_sha256",
            )
        ):
            raise DownstreamEvidenceError(
                "noncompleted final output receipt is partial"
            )

    return DownstreamPlanEntry(
        ordinal=source.ordinal,
        source_record_path=source.path,
        source_record_sha256=source.sha256,
        run_id=_text(run.get("run_id"), "run_id"),
        method_id=method_id,
        dataset_id=dataset_id,
        source_dataset_sha256=source_dataset_sha256,
        mechanism=_text(run.get("mechanism"), "run mechanism"),
        biological_id=_text(run.get("biological_id"), "run biological_id"),
        technical_view=_text(run.get("technical_view"), "run technical_view"),
        model_seed=_optional_model_seed(run.get("model_seed")),
        configuration_id=configuration_id,
        configuration_sha256=configuration_sha256,
        configuration_kind=configuration_kind,
        method_artifact_sha256=method_artifact_sha256,
        method_input_sha256=_digest(
            run.get("method_input_sha256"), "method input checksum"
        ),
        retained_cell_ids_sha256=retained_cell_ids_sha256,
        status=status,
        reason=None if reason is None else str(reason),
        evaluator_output_sha256=output_sha,
        evaluator_output_path=output_path,
        evaluator_output_file_sha256=output_file_sha,
        evaluator_output_shape=shape,
        evaluator_output_encoding=encoding,
        evaluator_output_uncompressed_nbytes=uncompressed_nbytes,
        evaluator_output_uncompressed_sha256=uncompressed_sha,
    )


def _build_downstream_evidence_plan(
    source_root: str | Path,
    *,
    source_kind: str,
    evidence_scope: str = "all",
    datasets: Sequence[DatasetEvidenceBinding],
    configurations: Sequence[AuthorizedConfiguration],
    evaluated_round_binding: EvaluatedRoundBinding | None = None,
    source_plan: object | None = None,
    _source_plan_authority: str | None = None,
) -> DownstreamEvidencePlan:
    """Bind a sealed development checkpoint or final execution manifest."""

    if source_kind not in _SOURCE_KINDS:
        raise ValueError("source_kind must be development or final")
    if evidence_scope not in _EVIDENCE_SCOPES:
        raise ValueError("evidence_scope is invalid")
    if source_kind == "development" and evidence_scope == "supplementary_trajectory":
        raise DownstreamEvidenceError(
            "development source cannot use the supplementary trajectory scope"
        )
    if source_kind == "final" and evidence_scope not in {
        "all",
        "supplementary_trajectory",
    }:
        raise DownstreamEvidenceError("final source evidence scope is invalid")
    if source_kind == "final" and evaluated_round_binding is None:
        raise DownstreamEvidenceError("final evaluated-round binding is required")
    root = _existing_directory(source_root, "source root")
    if evaluated_round_binding is not None:
        if not isinstance(evaluated_round_binding, EvaluatedRoundBinding):
            raise TypeError(
                "evaluated_round_binding must be an EvaluatedRoundBinding or null"
            )
        if source_kind != "final":
            raise DownstreamEvidenceError(
                "development source cannot carry an evaluated-round binding"
            )
        expected_source_root = Path(evaluated_round_binding.round_root) / (
            "results/trajectory/execution"
            if evidence_scope == "supplementary_trajectory"
            else "results/final/execution"
        )
        if root != expected_source_root:
            raise DownstreamEvidenceError(
                "final source root differs from evaluated-round binding"
            )
        _validate_evaluated_round_binding(evaluated_round_binding)
    dataset_values = tuple(datasets)
    if not dataset_values or any(
        not isinstance(value, DatasetEvidenceBinding) for value in dataset_values
    ):
        raise TypeError("datasets must contain DatasetEvidenceBinding values")
    dataset_lookup = {value.dataset_id: value for value in dataset_values}
    if len(dataset_lookup) != len(dataset_values):
        raise DownstreamEvidenceError("dataset bindings contain duplicate IDs")
    if evidence_scope == "supplementary_trajectory":
        from .trajectory_dataset import REGISTERED_TRAJECTORY_DATASET_ID

        assert evaluated_round_binding is not None
        expected_dataset_path = (
            Path(evaluated_round_binding.round_root)
            / "results/trajectory/dataset/evaluator.h5ad"
        )
        if (
            len(dataset_values) != 1
            or dataset_values[0].dataset_id != REGISTERED_TRAJECTORY_DATASET_ID
            or Path(dataset_values[0].path) != expected_dataset_path
            or dataset_values[0].mechanism != "synthetic_trajectory"
            or dataset_values[0].trajectory_root_cell_id is None
            or dataset_values[0].trajectory_source_id is None
            or dataset_values[0].trajectory_authority_sha256 is None
            or dataset_values[0].trajectory_binding_sha256 is None
        ):
            raise DownstreamEvidenceError(
                "supplementary trajectory dataset authority differs"
            )
    for value in dataset_values:
        _read_bound_dataset(value)
    configuration_values = tuple(configurations)
    if not configuration_values or any(
        not isinstance(value, AuthorizedConfiguration) for value in configuration_values
    ):
        raise TypeError("configurations must contain AuthorizedConfiguration values")
    configuration_lookup = {
        (
            value.method_id,
            value.configuration_id,
            value.kind,
            value.configuration_sha256,
        ): value
        for value in configuration_values
    }
    configuration_identities = {
        (value.method_id, value.configuration_id, value.kind)
        for value in configuration_values
    }
    if len(configuration_lookup) != len(configuration_values) or len(
        configuration_identities
    ) != len(configuration_values):
        raise DownstreamEvidenceError(
            "configuration authority contains duplicate identities"
        )
    for value in configuration_values:
        _method_artifact_sha256(value)
    if source_kind == "development":
        source_bundle = _development_source(root)
    elif evidence_scope == "supplementary_trajectory":
        source_bundle = _trajectory_source(root, source_plan)
    else:
        source_bundle = _final_source(root)
    if evidence_scope == "supplementary_trajectory":
        assert evaluated_round_binding is not None
        source_input_hashes = getattr(source_plan, "input_hashes", None)
        source_configurations = getattr(source_plan, "configurations", None)
        trajectory_dataset = dataset_values[0]
        if (
            not isinstance(source_input_hashes, Mapping)
            or set(source_input_hashes) != _TRAJECTORY_PLAN_INPUT_FIELDS
            or tuple(source_configurations or ()) != configuration_values
            or source_input_hashes.get("primary_final_plan_sha256")
            != evaluated_round_binding.final_plan_sha256
            or source_input_hashes.get("execution_claim_sha256")
            != evaluated_round_binding.trajectory_execution_claim_sha256
            or source_input_hashes.get("execution_environment_sha256")
            != evaluated_round_binding.trajectory_execution_environment_sha256
            or source_input_hashes.get("execution_authority_sha256")
            != evaluated_round_binding.trajectory_authority_sha256
            or source_input_hashes.get("trajectory_authority_sha256")
            != trajectory_dataset.trajectory_authority_sha256
            or source_input_hashes.get("trajectory_binding_sha256")
            != trajectory_dataset.trajectory_binding_sha256
            or source_input_hashes.get("trajectory_dataset_sha256")
            != trajectory_dataset.dataset_sha256
            or source_input_hashes.get("trajectory_dataset_file_sha256")
            != trajectory_dataset.file_sha256
            or source_input_hashes.get("trajectory_dataset_receipt_sha256")
            != (evaluated_round_binding.trajectory_dataset_receipt_payload_sha256)
            or source_input_hashes.get("trajectory_dataset_receipt_file_sha256")
            != evaluated_round_binding.trajectory_dataset_receipt_file_sha256
            or source_input_hashes.get("trajectory_method_input_sha256")
            != trajectory_dataset.method_input_sha256
            or source_input_hashes.get("trajectory_retained_cell_ids_sha256")
            != trajectory_dataset.retained_cell_ids_sha256
            or source_input_hashes.get("dataset_qc_policy_sha256")
            != trajectory_dataset.dataset_qc_policy_sha256
        ):
            raise DownstreamEvidenceError("trajectory source plan authority differs")
    if source_plan is not None:
        _validate_independent_source_plan(source_plan, source_bundle)
        source_plan_authority = "independent"
    elif _source_plan_authority == "independent":
        source_plan_authority = "independent"
    else:
        source_plan_authority = "manifest_only"
    if (
        source_kind == "final" or evidence_scope == "selection_primary"
    ) and source_plan_authority != "independent":
        raise DownstreamEvidenceError("independent source plan authority is required")
    manifest_path = source_bundle.manifest_path
    manifest_file_sha = source_bundle.manifest_file_sha256
    manifest_payload_sha = source_bundle.manifest_payload_sha256
    source_records = source_bundle.records
    if evaluated_round_binding is not None and (
        manifest_path != "execution_manifest.json"
        or (
            evidence_scope == "supplementary_trajectory"
            and (
                manifest_file_sha
                != evaluated_round_binding.trajectory_execution_manifest_file_sha256
                or manifest_payload_sha
                != evaluated_round_binding.trajectory_execution_manifest_payload_sha256
            )
        )
        or (
            evidence_scope != "supplementary_trajectory"
            and (
                manifest_file_sha
                != evaluated_round_binding.final_execution_manifest_file_sha256
                or manifest_payload_sha
                != evaluated_round_binding.final_execution_manifest_payload_sha256
            )
        )
    ):
        raise DownstreamEvidenceError(
            "final source manifest differs from evaluated-round binding"
        )
    if evidence_scope == "supplementary_trajectory":
        assert evaluated_round_binding is not None
        trajectory_dataset = dataset_values[0]
        validation = source_bundle.execution_validation
        if not isinstance(validation, Mapping):
            raise DownstreamEvidenceError("trajectory source validation is absent")
        if (
            source_bundle.source_plan_sha256
            != evaluated_round_binding.trajectory_plan_sha256
            or trajectory_dataset.dataset_sha256
            != evaluated_round_binding.trajectory_dataset_sha256
            or trajectory_dataset.dataset_id
            != evaluated_round_binding.trajectory_dataset_id
            or trajectory_dataset.file_sha256
            != evaluated_round_binding.trajectory_dataset_file_sha256
            or trajectory_dataset.trajectory_source_id
            != evaluated_round_binding.trajectory_source_id
            or trajectory_dataset.trajectory_root_cell_id
            != evaluated_round_binding.trajectory_root_cell_id
            or trajectory_dataset.trajectory_authority_sha256
            != evaluated_round_binding.trajectory_registered_authority_sha256
            or trajectory_dataset.trajectory_binding_sha256
            != evaluated_round_binding.trajectory_registered_binding_sha256
            or validation.get("validation_sha256")
            != evaluated_round_binding.trajectory_execution_validation_sha256
            or validation.get("planned_run_count")
            != evaluated_round_binding.trajectory_planned_run_count
            or canonical_sha256(validation.get("record_payload_sha256s"))
            != evaluated_round_binding.trajectory_record_payload_sha256s_sha256
            or canonical_sha256(
                {
                    "executed_status_counts": validation.get("executed_status_counts"),
                    "not_applicable_count": validation.get("not_applicable_count"),
                }
            )
            != evaluated_round_binding.trajectory_status_counts_sha256
        ):
            raise DownstreamEvidenceError(
                "trajectory dataset differs from evaluated receipt"
            )
    all_entries = tuple(
        _validated_plan_entry(
            record,
            source_kind=source_kind,
            source_root=root,
            datasets=dataset_lookup,
            configurations=configuration_lookup,
        )
        for record in source_records
    )
    selection_declared = {
        value.configuration_id
        for value in configuration_values
        if value.kind == "candidate_search"
    } | {
        value.method_id
        for value in configuration_values
        if value.kind == "registry" or value.method_id == "capacity-matched-ae"
    }

    def selection_applicable(entry: DownstreamPlanEntry) -> bool:
        from .development_evaluation import reconstruction_selection_method

        return (
            reconstruction_selection_method(
                {
                    "configuration_kind": entry.configuration_kind,
                    "configuration_id": entry.configuration_id,
                    "method_id": entry.method_id,
                },
                selection_declared,
            )
            is not None
        )

    if evidence_scope == "selection_primary":
        selected_entries = tuple(
            entry for entry in all_entries if selection_applicable(entry)
        )
    elif evidence_scope == "supplementary_nonselection":
        selected_entries = tuple(
            entry for entry in all_entries if not selection_applicable(entry)
        )
    else:
        selected_entries = all_entries
    entries = tuple(
        replace(entry, ordinal=ordinal)
        for ordinal, entry in enumerate(selected_entries, start=1)
    )
    if not entries:
        raise DownstreamEvidenceError("source has no downstream denominators")
    source_statuses_sha256 = canonical_sha256(
        [
            {
                "run_id": entry.run_id,
                "status": entry.status,
                "reason": entry.reason,
            }
            for entry in entries
        ]
    )
    persisted_configurations = tuple(
        sorted(
            configuration_values,
            key=lambda value: (
                value.method_id,
                value.configuration_id,
                value.kind,
                value.configuration_sha256,
            ),
        )
    )
    provisional = DownstreamEvidencePlan(
        source_root=str(root),
        source_kind=source_kind,
        evidence_scope=evidence_scope,
        evaluator_source_sha256=_evaluator_source_sha256(),
        source_manifest_path=manifest_path,
        source_manifest_file_sha256=manifest_file_sha,
        source_manifest_payload_sha256=manifest_payload_sha,
        source_plan_sha256=source_bundle.source_plan_sha256,
        source_input_hashes_sha256=source_bundle.source_input_hashes_sha256,
        source_statuses_sha256=source_statuses_sha256,
        source_plan_authority=source_plan_authority,
        evaluated_round_binding=evaluated_round_binding,
        development_revision_versions=(),
        development_sources=(),
        datasets=tuple(sorted(dataset_values, key=lambda value: value.dataset_id)),
        configurations=persisted_configurations,
        entries=entries,
        plan_sha256="0" * 64,
    )
    return DownstreamEvidencePlan(
        source_root=provisional.source_root,
        source_kind=provisional.source_kind,
        evidence_scope=provisional.evidence_scope,
        evaluator_source_sha256=provisional.evaluator_source_sha256,
        source_manifest_path=provisional.source_manifest_path,
        source_manifest_file_sha256=provisional.source_manifest_file_sha256,
        source_manifest_payload_sha256=provisional.source_manifest_payload_sha256,
        source_plan_sha256=provisional.source_plan_sha256,
        source_input_hashes_sha256=provisional.source_input_hashes_sha256,
        source_statuses_sha256=provisional.source_statuses_sha256,
        source_plan_authority=provisional.source_plan_authority,
        evaluated_round_binding=provisional.evaluated_round_binding,
        development_revision_versions=provisional.development_revision_versions,
        development_sources=provisional.development_sources,
        datasets=provisional.datasets,
        configurations=provisional.configurations,
        entries=provisional.entries,
        plan_sha256=canonical_sha256(provisional.body()),
    )


def build_downstream_evidence_plan(
    source_root: str | Path,
    *,
    source_kind: str,
    evidence_scope: str = "all",
    datasets: Sequence[DatasetEvidenceBinding],
    configurations: Sequence[AuthorizedConfiguration],
    evaluated_round_binding: EvaluatedRoundBinding | None = None,
    source_plan: object | None = None,
    _source_plan_authority: str | None = None,
) -> DownstreamEvidencePlan:
    """Bind a development checkpoint or the fixed primary final denominator."""

    if source_kind == "final" and evidence_scope == "supplementary_trajectory":
        raise DownstreamEvidenceError(
            "supplementary trajectory plans require the fixed production builder"
        )
    return _build_downstream_evidence_plan(
        source_root,
        source_kind=source_kind,
        evidence_scope=evidence_scope,
        datasets=datasets,
        configurations=configurations,
        evaluated_round_binding=evaluated_round_binding,
        source_plan=source_plan,
        _source_plan_authority=_source_plan_authority,
    )


def _json_pointer_value(value: object, pointer: str) -> object:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise DownstreamEvidenceError("development evaluation pointer is invalid")
    current = value
    for token in pointer[1:].split("/"):
        if not token or "~" in token:
            raise DownstreamEvidenceError("development evaluation pointer is invalid")
        if isinstance(current, Mapping):
            if token not in current:
                raise DownstreamEvidenceError(
                    "development evaluation source binding is absent"
                )
            current = current[token]
        elif isinstance(current, list) and token.isdigit():
            index = int(token)
            if index >= len(current):
                raise DownstreamEvidenceError(
                    "development evaluation source binding is absent"
                )
            current = current[index]
        else:
            raise DownstreamEvidenceError(
                "development evaluation source binding is invalid"
            )
    return current


def _relative_source_root(repository: Path, source_root: str) -> PurePosixPath:
    source = _existing_directory(source_root, "development component source")
    try:
        relative = source.relative_to(repository)
    except ValueError as error:
        raise DownstreamEvidenceError(
            "development component source must be inside its repository"
        ) from error
    if not relative.parts:
        raise DownstreamEvidenceError(
            "development component source cannot equal its repository"
        )
    return PurePosixPath(*relative.parts)


def _development_denominator_payload(
    plan: DownstreamEvidencePlan, entry: DownstreamPlanEntry
) -> dict[str, object]:
    return {
        "mechanism": entry.mechanism,
        "biological_id": entry.biological_id,
        "technical_view": entry.technical_view,
        "dataset_id": entry.dataset_id,
        "dataset_sha256": entry.source_dataset_sha256,
        "method": _analysis_method(plan, entry),
        "method_artifact_sha256": entry.method_artifact_sha256,
        "model_seed": entry.model_seed,
    }


def _validated_development_source_binding(
    repository: Path,
    source: DevelopmentSourcePlan,
    selected_entries: tuple[DownstreamPlanEntry, ...],
) -> DevelopmentSourceBinding:
    if not isinstance(source, DevelopmentSourcePlan):
        raise TypeError("development sources must contain DevelopmentSourcePlan values")
    plan = source.plan
    if (
        not isinstance(plan, DownstreamEvidencePlan)
        or plan.plan_sha256 != canonical_sha256(plan.body())
        or plan.source_kind != "development"
        or plan.evaluated_round_binding is not None
        or plan.development_sources
        or plan.development_revision_versions
        or plan.source_plan_authority != "independent"
    ):
        raise DownstreamEvidenceError(
            "development component must be a single independently planned checkpoint"
        )
    if source.source_id not in {"base", "v28", "v29"}:
        raise DownstreamEvidenceError("development source identity is invalid")
    methods = tuple(source.selected_methods)
    if (
        not methods
        or methods != tuple(sorted(set(methods)))
        or any(not isinstance(value, str) or not value for value in methods)
    ):
        raise DownstreamEvidenceError("development source selected methods are invalid")
    relative_root = _relative_source_root(repository, plan.source_root)
    checkpoint_relative = str(relative_root / plan.source_manifest_path)
    evaluation_path = _safe_relative(
        repository,
        source.evaluation_manifest_path,
        "development evaluation manifest",
    )
    evaluation, _evaluation_raw, evaluation_file_sha = _strict_json(
        evaluation_path, "development evaluation manifest"
    )
    evaluation_payload_sha = _digest(
        evaluation.get("manifest_sha256"),
        "development evaluation manifest payload checksum",
    )
    evaluation_body = {
        key: value for key, value in evaluation.items() if key != "manifest_sha256"
    }
    evaluation_source = _json_pointer_value(
        evaluation, source.evaluation_source_pointer
    )
    if not isinstance(evaluation_source, Mapping):
        raise DownstreamEvidenceError(
            "development evaluation source binding must be an object"
        )
    source_sha = canonical_sha256(dict(evaluation_source))
    input_hashes = evaluation_source.get("input_hashes")
    if (
        canonical_sha256(evaluation_body) != evaluation_payload_sha
        or evaluation_file_sha
        != _digest(
            source.evaluation_manifest_file_sha256,
            "development evaluation manifest file checksum",
        )
        or evaluation_payload_sha
        != _digest(
            source.evaluation_manifest_payload_sha256,
            "development evaluation manifest payload checksum",
        )
        or source_sha
        != _digest(
            source.evaluation_source_sha256,
            "development evaluation source checksum",
        )
        or evaluation_source.get("checkpoint_path") != checkpoint_relative
        or evaluation_source.get("checkpoint_file_sha256")
        != plan.source_manifest_file_sha256
        or evaluation_source.get("checkpoint_sha256")
        != plan.source_manifest_payload_sha256
        or evaluation_source.get("plan_sha256") != plan.source_plan_sha256
        or not isinstance(input_hashes, Mapping)
        or canonical_sha256(dict(input_hashes)) != plan.source_input_hashes_sha256
    ):
        raise DownstreamEvidenceError(
            "development checkpoint differs from evaluation authority"
        )
    statuses = [
        {"run_id": entry.run_id, "status": entry.status, "reason": entry.reason}
        for entry in selected_entries
    ]
    denominators = [
        _development_denominator_payload(plan, entry) for entry in selected_entries
    ]
    return DevelopmentSourceBinding(
        source_id=source.source_id,
        source_root=str(relative_root),
        selected_methods=methods,
        manifest_path=checkpoint_relative,
        manifest_file_sha256=plan.source_manifest_file_sha256,
        manifest_payload_sha256=plan.source_manifest_payload_sha256,
        plan_sha256=plan.source_plan_sha256,
        input_hashes_sha256=plan.source_input_hashes_sha256,
        statuses_sha256=canonical_sha256(statuses),
        denominator_sha256=canonical_sha256(denominators),
        planned_denominator_count=len(selected_entries),
        evaluation_manifest_path=source.evaluation_manifest_path,
        evaluation_manifest_file_sha256=evaluation_file_sha,
        evaluation_manifest_payload_sha256=evaluation_payload_sha,
        evaluation_source_pointer=source.evaluation_source_pointer,
        evaluation_source_sha256=source_sha,
    )


def combine_development_downstream_evidence_plans(
    repository: str | Path,
    sources: Sequence[DevelopmentSourcePlan],
    *,
    revision_versions: Sequence[str],
) -> DownstreamEvidencePlan:
    """Combine base plus activated revision checkpoints without duplicate controls."""

    root = _existing_directory(repository, "development bundle repository")
    source_values = tuple(sources)
    versions = tuple(revision_versions)
    expected_versions = (
        ("v28",)
        if versions == ("v28",)
        else ("v28", "v29")
        if versions == ("v28", "v29")
        else ()
    )
    if versions != expected_versions:
        raise DownstreamEvidenceError(
            "development revision versions are incomplete or reordered"
        )
    if tuple(value.source_id for value in source_values) != ("base", *versions):
        raise DownstreamEvidenceError(
            "development checkpoint bundle is incomplete or reordered"
        )
    base_datasets = source_values[0].plan.datasets
    evaluator_sha = source_values[0].plan.evaluator_source_sha256
    entries: list[DownstreamPlanEntry] = []
    bindings: list[DevelopmentSourceBinding] = []
    configurations: dict[tuple[str, str, str, str], AuthorizedConfiguration] = {}
    for source in source_values:
        component = source.plan
        if (
            component.datasets != base_datasets
            or component.evaluator_source_sha256 != evaluator_sha
        ):
            raise DownstreamEvidenceError(
                "development checkpoint evaluator or dataset authority differs"
            )
        for configuration in component.configurations:
            key = (
                configuration.method_id,
                configuration.configuration_id,
                configuration.kind,
                configuration.configuration_sha256,
            )
            previous = configurations.setdefault(key, configuration)
            if previous.to_dict() != configuration.to_dict():
                raise DownstreamEvidenceError(
                    "development configuration authority conflicts across checkpoints"
                )
        selected = tuple(
            entry
            for entry in component.entries
            if _analysis_method(component, entry) in source.selected_methods
        )
        if not selected or {
            _analysis_method(component, entry) for entry in selected
        } != set(source.selected_methods):
            raise DownstreamEvidenceError(
                "development source selection denominator is incomplete"
            )
        binding = _validated_development_source_binding(root, source, selected)
        relative_root = PurePosixPath(binding.source_root)
        for entry in selected:
            rebased_output = (
                None
                if entry.evaluator_output_path is None
                else str(relative_root / entry.evaluator_output_path)
            )
            entries.append(
                replace(
                    entry,
                    ordinal=len(entries) + 1,
                    source_record_path=str(relative_root / entry.source_record_path),
                    evaluator_output_path=rebased_output,
                )
            )
        bindings.append(binding)
    denominator_payloads = [
        _development_denominator_payload(
            source_values[0].plan,
            entry,
        )
        for entry in entries
    ]
    denominator_keys = {tuple(value.values()) for value in denominator_payloads}
    if len(denominator_keys) != len(entries):
        raise DownstreamEvidenceError(
            "development checkpoint bundle contains duplicate denominators"
        )
    latest = bindings[-1]
    provisional = DownstreamEvidencePlan(
        source_root=str(root),
        source_kind="development",
        evidence_scope="selection_primary",
        evaluator_source_sha256=evaluator_sha,
        source_manifest_path=latest.evaluation_manifest_path,
        source_manifest_file_sha256=latest.evaluation_manifest_file_sha256,
        source_manifest_payload_sha256=latest.evaluation_manifest_payload_sha256,
        source_plan_sha256=canonical_sha256(
            [
                {"source_id": value.source_id, "plan_sha256": value.plan_sha256}
                for value in bindings
            ]
        ),
        source_input_hashes_sha256=canonical_sha256(
            [
                {
                    "source_id": value.source_id,
                    "input_hashes_sha256": value.input_hashes_sha256,
                }
                for value in bindings
            ]
        ),
        source_statuses_sha256=canonical_sha256(
            [
                {
                    "source_id": value.source_id,
                    "statuses_sha256": value.statuses_sha256,
                }
                for value in bindings
            ]
        ),
        source_plan_authority="independent",
        evaluated_round_binding=None,
        development_revision_versions=versions,
        development_sources=tuple(bindings),
        datasets=base_datasets,
        configurations=tuple(
            sorted(
                configurations.values(),
                key=lambda value: (
                    value.method_id,
                    value.configuration_id,
                    value.kind,
                    value.configuration_sha256,
                ),
            )
        ),
        entries=tuple(entries),
        plan_sha256="0" * 64,
    )
    return replace(provisional, plan_sha256=canonical_sha256(provisional.body()))


def build_development_downstream_evidence_plan(
    repository: str | Path,
    *,
    checkpoint_directory: str | Path | None = None,
    through_version: str | None = None,
) -> DownstreamEvidencePlan:
    """Build base-only or revision-aware production development evidence."""

    root = _existing_directory(repository, "development repository")
    active_repository = Path(__file__).absolute().parents[1]
    _reject_symlink_chain(active_repository, "active repository")
    if root != active_repository:
        raise DownstreamEvidenceError(
            "development downstream stage must use the active repository"
        )
    if through_version not in {None, "v28", "v29"}:
        raise ValueError("through_version must be null, v28, or v29")
    if through_version is not None and checkpoint_directory is not None:
        raise DownstreamEvidenceError(
            "revision-aware development evidence uses only fixed checkpoint paths"
        )
    from .methods import load_method_registry
    from .runner import (
        build_competition_plan,
        load_activated_v28_revision_authority,
        load_activated_v29_revision_authority,
        load_prepared_development_panel,
        load_runner_authority,
    )

    registry = load_method_registry(root / "study/methods.json")

    def component(
        authority: object,
        source: Path,
        *,
        evidence_scope: str,
    ) -> DownstreamEvidencePlan:
        configured_method_ids = {value.method_id for value in authority.configurations}
        configurations = tuple(authority.configurations) + tuple(
            AuthorizedConfiguration.registry_default(spec)
            for spec in registry.methods
            if spec.execution_scope == "same_input_required"
            and spec.id not in configured_method_ids
        )
        runner_bindings, prepared = load_prepared_development_panel(authority)
        datasets = bind_prepared_evaluator_panel(
            runner_bindings,
            prepared,
            dataset_root=root / "artifacts/study/development/results",
        )
        checkpoint, _checkpoint_raw, _checkpoint_file_sha = _strict_json(
            source / "checkpoint.json", "development checkpoint"
        )
        checkpoint_inputs = checkpoint.get("input_hashes")
        if not isinstance(checkpoint_inputs, Mapping):
            raise DownstreamEvidenceError(
                "development source input authority is absent"
            )
        source_plan = build_competition_plan(
            registry,
            runner_bindings,
            authority,
            execution_environment_sha256=_digest(
                checkpoint_inputs.get("execution_environment_sha256"),
                "development execution environment checksum",
            ),
        )
        return build_downstream_evidence_plan(
            source,
            source_kind="development",
            evidence_scope=evidence_scope,
            datasets=datasets,
            configurations=configurations,
            source_plan=source_plan,
        )

    base_authority = load_runner_authority()
    base_source = (
        root / "artifacts/study/development/competition-reconstruction"
        if checkpoint_directory is None
        else Path(checkpoint_directory).absolute()
    )
    base = component(
        base_authority,
        base_source,
        evidence_scope="selection_primary",
    )
    if through_version is None:
        return base

    from .revision_evaluation import (
        assemble_revision_evaluation,
        validate_revision_artifact_payloads,
    )
    from .revisions import revision_stage_paths

    assembled = assemble_revision_evaluation(
        root,
        through_version,
        execute_missing_orthogonal=False,
        require_clean=True,
    )
    paths = revision_stage_paths(through_version)
    revision_selection, _selection_raw, _selection_file_sha = _strict_json(
        root / paths.selection_input,
        f"{through_version} revision selection input",
    )
    try:
        validate_revision_artifact_payloads(root, revision_selection, assembled)
    except Exception as error:
        raise DownstreamEvidenceError(
            "revision evaluation authority could not be validated"
        ) from error

    def evaluation_source(
        source_id: str,
        plan: DownstreamEvidencePlan,
        methods: tuple[str, ...],
        evaluation_path: str,
        pointer: str,
    ) -> DevelopmentSourcePlan:
        evaluation, _raw, file_sha = _strict_json(
            root / evaluation_path,
            f"{source_id} development evaluation manifest",
        )
        payload_sha = _digest(
            evaluation.get("manifest_sha256"),
            f"{source_id} evaluation payload checksum",
        )
        source_value = _json_pointer_value(evaluation, pointer)
        if not isinstance(source_value, Mapping):
            raise DownstreamEvidenceError(
                f"{source_id} evaluation source binding is invalid"
            )
        return DevelopmentSourcePlan(
            source_id=source_id,
            plan=plan,
            selected_methods=methods,
            evaluation_manifest_path=evaluation_path,
            evaluation_manifest_file_sha256=file_sha,
            evaluation_manifest_payload_sha256=payload_sha,
            evaluation_source_pointer=pointer,
            evaluation_source_sha256=canonical_sha256(dict(source_value)),
        )

    sources = [
        evaluation_source(
            "base",
            base,
            tuple(sorted({_analysis_method(base, entry) for entry in base.entries})),
            assembled.base_evaluation_manifest_path,
            "/reconstruction",
        )
    ]
    revision_evaluation_path = paths.evaluation_manifest
    for index, stage in enumerate(assembled.stages):
        authority = (
            load_activated_v28_revision_authority()
            if stage.spec.version == "v28"
            else load_activated_v29_revision_authority()
        )
        stage_plan = component(
            authority,
            root / revision_stage_paths(stage.spec.version).reconstruction_directory,
            evidence_scope="all",
        )
        sources.append(
            evaluation_source(
                stage.spec.version,
                stage_plan,
                (stage.spec.configuration_id,),
                revision_evaluation_path,
                f"/revisions/{index}/reconstruction",
            )
        )
    return combine_development_downstream_evidence_plans(
        root,
        tuple(sources),
        revision_versions=tuple(stage.spec.version for stage in assembled.stages),
    )


def development_downstream_revision_version(
    repository: str | Path,
) -> str | None:
    """Select the latest fixed combined revision input without silent fallback."""

    root = _existing_directory(repository, "development repository")
    from .revisions import revision_stage_paths

    for version, expected_versions in (
        ("v29", ["v28", "v29"]),
        ("v28", ["v28"]),
    ):
        path = root / revision_stage_paths(version).selection_input
        if not os.path.lexists(path):
            continue
        payload, _raw, _file_sha = _strict_json(
            path, f"{version} revision selection input"
        )
        if (
            payload.get("schema_version") != 3
            or payload.get("revision_versions") != expected_versions
        ):
            raise DownstreamEvidenceError(
                f"{version} revision selection input identity differs"
            )
        return version
    return None


def build_final_downstream_evidence_plan(
    repository: str | Path,
    round_directory: str | Path,
) -> DownstreamEvidencePlan:
    """Build the production final plan from the frozen round's persisted data."""

    root = _existing_directory(repository, "final repository")
    active_repository = Path(__file__).absolute().parents[1]
    _reject_symlink_chain(active_repository, "active repository")
    if root != active_repository:
        raise DownstreamEvidenceError(
            "final downstream stage must use the active repository"
        )
    round_root = _existing_directory(round_directory, "final round")
    from .final_runner import (
        _configuration_for_method,
        build_final_execution_plan,
        load_prepared_final_panel,
    )
    from .methods import load_method_registry
    from .publication_freeze import validate_frozen_method

    evaluated_round_binding = _read_verified_evaluated_round_binding(root, round_root)
    frozen_method = validate_frozen_method(root)
    registry = load_method_registry(root / "study/methods.json")
    configurations = tuple(
        _configuration_for_method(spec.id, spec, frozen_method)
        for spec in registry.methods
    )
    runner_bindings, prepared = load_prepared_final_panel(
        root,
        round_root,
        allow_evaluated=True,
    )
    datasets = bind_prepared_evaluator_panel(
        runner_bindings,
        prepared,
        dataset_root=round_root / "results",
    )
    execution_manifest, _execution_raw, _execution_file_sha = _strict_json(
        round_root / "results/final/execution/execution_manifest.json",
        "final execution manifest",
    )
    execution_inputs = execution_manifest.get("input_hashes")
    if not isinstance(execution_inputs, Mapping):
        raise DownstreamEvidenceError("final source input authority is absent")
    source_plan = build_final_execution_plan(
        frozen_method,
        registry,
        runner_bindings,
        execution_claim_sha256=_digest(
            execution_inputs.get("execution_claim_sha256"),
            "final execution claim checksum",
        ),
        execution_environment_sha256=_digest(
            execution_inputs.get("execution_environment_sha256"),
            "final execution environment checksum",
        ),
        execution_authority_sha256=_digest(
            execution_inputs.get("execution_authority_sha256"),
            "final execution authority checksum",
        ),
    )
    return build_downstream_evidence_plan(
        round_root / "results/final/execution",
        source_kind="final",
        datasets=datasets,
        configurations=configurations,
        evaluated_round_binding=evaluated_round_binding,
        source_plan=source_plan,
    )


def _bind_registered_trajectory_dataset(
    round_root: Path,
    registered: object,
) -> DatasetEvidenceBinding:
    """Bridge a strictly loaded registered trajectory dataset into evidence."""

    from .runner import method_input_sha256
    from .trajectory_dataset import (
        RegisteredTrajectoryBinding,
        TrajectoryPreparedDataset,
    )

    if not isinstance(round_root, Path):
        raise TypeError("round_root must be a pathlib.Path")
    if not isinstance(registered, TrajectoryPreparedDataset):
        raise TypeError("registered must be a TrajectoryPreparedDataset")
    binding = registered.binding
    prepared = registered.prepared
    audit = getattr(prepared, "audit", None)
    method_input = getattr(prepared, "method_input", None)
    if not isinstance(binding, RegisteredTrajectoryBinding) or audit is None:
        raise DownstreamEvidenceError(
            "registered trajectory prepared authority is invalid"
        )
    dataset_path = _safe_relative(
        round_root,
        binding.dataset_file_path,
        "registered trajectory evaluator dataset",
    )
    try:
        retained_cell_ids = tuple(audit.retained_cell_ids)
        prepared_method_input_sha256 = method_input_sha256(method_input)
    except (AttributeError, TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "registered trajectory prepared authority is invalid"
        ) from error
    bound = bind_evaluator_dataset(
        dataset_path,
        retained_cell_ids=retained_cell_ids,
        trajectory_root_cell_id=binding.root_cell_id,
        trajectory_source_id=binding.source_id,
    )
    if (
        bound.dataset_id != binding.dataset_id
        or bound.file_sha256 != binding.dataset_file_sha256
        or bound.dataset_sha256 != binding.dataset_sha256
        or bound.mechanism != binding.mechanism
        or bound.biological_id != binding.biological_id
        or bound.technical_view != binding.technical_view
        or bound.method_input_sha256 != prepared_method_input_sha256
        or bound.excluded_cell_count != audit.excluded_cell_count
        or bound.excluded_cell_ids_sha256 != audit.excluded_cell_ids_sha256
        or bound.retained_cell_count != audit.retained_cell_count
        or bound.retained_cell_ids_sha256 != audit.retained_cell_ids_sha256
        or bound.trajectory_root_cell_id != binding.root_cell_id
        or bound.trajectory_source_id != binding.source_id
        or bound.trajectory_authority_sha256 != binding.authority_sha256
        or bound.trajectory_binding_sha256 != binding.registered_binding_sha256
    ):
        raise DownstreamEvidenceError(
            "registered trajectory dataset differs from its prepared authority"
        )
    return bound


def build_final_trajectory_downstream_evidence_plan(
    repository: str | Path,
    round_directory: str | Path,
) -> DownstreamEvidencePlan:
    """Build the fixed receipt-bound supplementary trajectory plan."""

    root = _existing_directory(repository, "final repository")
    active_repository = Path(__file__).absolute().parents[1]
    _reject_symlink_chain(active_repository, "active repository")
    if root != active_repository:
        raise DownstreamEvidenceError(
            "final trajectory downstream stage must use the active repository"
        )
    round_root = _existing_directory(round_directory, "final round")
    from .final_runner import (
        build_trajectory_execution_plan,
        load_prepared_trajectory_dataset,
    )
    from .methods import load_method_registry
    from .publication_freeze import validate_frozen_method

    evaluated_round_binding = _read_verified_evaluated_round_binding(root, round_root)
    frozen_method = validate_frozen_method(root)
    registry = load_method_registry(root / "study/methods.json")
    registered = load_prepared_trajectory_dataset(root, round_root)
    dataset = _bind_registered_trajectory_dataset(round_root, registered)
    source_plan = build_trajectory_execution_plan(
        frozen_method,
        registry,
        registered,
        execution_claim_sha256=(
            evaluated_round_binding.trajectory_execution_claim_sha256
        ),
        execution_environment_sha256=(
            evaluated_round_binding.trajectory_execution_environment_sha256
        ),
        execution_authority_sha256=(
            evaluated_round_binding.trajectory_authority_sha256
        ),
        primary_final_plan_sha256=evaluated_round_binding.final_plan_sha256,
    )
    return _build_downstream_evidence_plan(
        round_root / "results/trajectory/execution",
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=source_plan.configurations,
        evaluated_round_binding=evaluated_round_binding,
        source_plan=source_plan,
    )


def _decode_output(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    binding: DatasetEvidenceBinding,
) -> MethodOutput:
    if entry.evaluator_output_path is None or entry.evaluator_output_shape is None:
        raise DownstreamEvidenceError("completed evaluator output binding is absent")
    path = _safe_relative(
        Path(plan.source_root), entry.evaluator_output_path, "evaluator output"
    )
    if plan.source_kind == "development":
        raw, observed_file_sha = _stable_file_bytes(path, "evaluator output")
    else:
        expected_nbytes = entry.evaluator_output_uncompressed_nbytes
        if expected_nbytes is None:
            raise DownstreamEvidenceError(
                "final evaluator output byte receipt is absent"
            )
        compressed, observed_file_sha = _stable_file_bytes(
            path,
            "evaluator output",
            max_bytes=_zlib_compress_bound(expected_nbytes),
        )
        try:
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(compressed, expected_nbytes + 1)
            raw += decompressor.flush(max(1, expected_nbytes + 1 - len(raw)))
        except zlib.error as error:
            raise DownstreamEvidenceError(
                "final evaluator output compression is invalid"
            ) from error
        if (
            len(raw) != expected_nbytes
            or not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
            or hashlib.sha256(raw).hexdigest()
            != entry.evaluator_output_uncompressed_sha256
        ):
            raise DownstreamEvidenceError("final evaluator output receipt differs")
    if observed_file_sha != entry.evaluator_output_file_sha256:
        raise DownstreamEvidenceError("evaluator output file checksum differs")
    expected_size = (
        entry.evaluator_output_shape[0] * entry.evaluator_output_shape[1] * 8
    )
    if len(raw) != expected_size:
        raise DownstreamEvidenceError("evaluator output byte size differs")
    content_digest = hashlib.sha256()
    content_digest.update(b"maskimpute-evaluator-log2-cp10k-output-v1\0")
    content_digest.update(
        _canonical_bytes(
            {
                "run_id": entry.run_id,
                "method_input_sha256": entry.method_input_sha256,
                "retained_cell_ids_sha256": entry.retained_cell_ids_sha256,
                "shape": list(entry.evaluator_output_shape),
                "dtype": "<f8",
                "scale": "log2_cp10k_plus_1",
            }
        )
    )
    content_digest.update(raw)
    if content_digest.hexdigest() != entry.evaluator_output_sha256:
        raise DownstreamEvidenceError("evaluator output content checksum differs")
    values = (
        np.frombuffer(raw, dtype="<f8").reshape(entry.evaluator_output_shape).copy()
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise DownstreamEvidenceError("evaluator output contains invalid values")
    return MethodOutput(
        values=values,
        cell_ids=binding.retained_cell_ids,
        gene_ids=binding.gene_ids,
    )


_ENDPOINT_ROW_FIELDS = frozenset(
    {
        "source_kind",
        "run_id",
        "runner_method_id",
        "method",
        "dataset_id",
        "dataset_sha256",
        "mechanism",
        "biological_id",
        "technical_view",
        "model_seed",
        "configuration_id",
        "configuration_sha256",
        "method_artifact_sha256",
        "endpoint",
        "value",
        "status",
        "reason_code",
        "upstream_status",
        "upstream_reason",
        "direction",
        "independent_unit",
        "independent_n",
        "descriptive_n",
        "descriptive_unit",
        "procedure",
        "family_id",
        "family_size",
        "alpha",
    }
)
_RECORD_BODY_FIELDS = frozenset(
    {
        "schema_version",
        "ordinal",
        "source_kind",
        "source_record_path",
        "source_record_sha256",
        "run_id",
        "runner_method_id",
        "method",
        "dataset_id",
        "dataset_sha256",
        "dataset_file_sha256",
        "mechanism",
        "biological_id",
        "technical_view",
        "model_seed",
        "configuration_id",
        "configuration_sha256",
        "method_artifact_sha256",
        "run_status",
        "run_reason",
        "endpoints",
    }
)


def _analysis_method(plan: DownstreamEvidencePlan, entry: DownstreamPlanEntry) -> str:
    if (
        plan.source_kind == "development"
        and entry.configuration_kind == "candidate_search"
    ):
        return entry.configuration_id
    return entry.method_id


def _endpoint_names(plan: DownstreamEvidencePlan) -> tuple[str, ...]:
    """Return the closed endpoint schema for one immutable evidence scope."""

    if not isinstance(plan, DownstreamEvidencePlan):
        raise TypeError("plan must be a DownstreamEvidencePlan")
    if plan.evidence_scope == "supplementary_trajectory":
        return _TRAJECTORY_ENDPOINT_NAMES
    return DOWNSTREAM_ENDPOINT_NAMES


def _endpoint_row(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    endpoint: EndpointRecord,
) -> dict[str, object]:
    upstream_completed = entry.status == "completed"
    return {
        "source_kind": plan.source_kind,
        "run_id": entry.run_id,
        "runner_method_id": entry.method_id,
        "method": _analysis_method(plan, entry),
        "dataset_id": entry.dataset_id,
        "dataset_sha256": entry.source_dataset_sha256,
        "mechanism": entry.mechanism,
        "biological_id": entry.biological_id,
        "technical_view": entry.technical_view,
        "model_seed": entry.model_seed,
        "configuration_id": entry.configuration_id,
        "configuration_sha256": entry.configuration_sha256,
        "method_artifact_sha256": entry.method_artifact_sha256,
        "endpoint": endpoint.endpoint,
        "value": endpoint.value if upstream_completed else None,
        "status": endpoint.status if upstream_completed else entry.status,
        "reason_code": (
            endpoint.reason if upstream_completed else "upstream_run_not_completed"
        ),
        "upstream_status": entry.status,
        "upstream_reason": entry.reason,
        "direction": endpoint.direction,
        "independent_unit": endpoint.independent_unit,
        "independent_n": endpoint.independent_n,
        "descriptive_n": endpoint.descriptive_n if upstream_completed else 0,
        "descriptive_unit": endpoint.descriptive_unit,
        "procedure": endpoint.procedure,
        "family_id": endpoint.family_id,
        "family_size": endpoint.family_size,
        "alpha": endpoint.alpha,
    }


def _evaluate_entry(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    binding: DatasetEvidenceBinding,
    targets: EvaluatorTargets | None,
) -> dict[str, object]:
    if entry.status == "completed":
        output = _decode_output(plan, entry, binding)
        if targets is None:
            raise AssertionError("completed denominator targets were not cached")
        if plan.evidence_scope == "supplementary_trajectory":
            try:
                endpoints = (evaluate_trajectory_endpoint(output, targets),)
            except (
                np.linalg.LinAlgError,
                sparse.linalg.ArpackError,
                FloatingPointError,
                OverflowError,
            ):
                terminal = terminal_downstream_endpoints(
                    "numeric_evaluation_failed",
                    procedure="terminal_expected_numeric_failure",
                )
                endpoints = tuple(
                    value
                    for value in terminal
                    if value.endpoint in _TRAJECTORY_ENDPOINT_NAMES
                )
        else:
            endpoints = evaluate_downstream_endpoints(output, targets)
    else:
        terminal = terminal_downstream_endpoints(
            "upstream_run_not_completed",
            procedure="terminal_upstream_run_not_completed",
        )
        names = _endpoint_names(plan)
        endpoints = tuple(value for value in terminal if value.endpoint in names)
    if tuple(value.endpoint for value in endpoints) != _endpoint_names(plan):
        raise AssertionError("downstream evaluator did not emit its fixed schema")
    body: dict[str, object] = {
        "schema_version": 1,
        "ordinal": entry.ordinal,
        "source_kind": plan.source_kind,
        "source_record_path": entry.source_record_path,
        "source_record_sha256": entry.source_record_sha256,
        "run_id": entry.run_id,
        "runner_method_id": entry.method_id,
        "method": _analysis_method(plan, entry),
        "dataset_id": entry.dataset_id,
        "dataset_sha256": entry.source_dataset_sha256,
        "dataset_file_sha256": binding.file_sha256,
        "mechanism": entry.mechanism,
        "biological_id": entry.biological_id,
        "technical_view": entry.technical_view,
        "model_seed": entry.model_seed,
        "configuration_id": entry.configuration_id,
        "configuration_sha256": entry.configuration_sha256,
        "method_artifact_sha256": entry.method_artifact_sha256,
        "run_status": entry.status,
        "run_reason": entry.reason,
        "endpoints": [_endpoint_row(plan, entry, endpoint) for endpoint in endpoints],
    }
    return {**body, "record_sha256": canonical_sha256(body)}


def _load_targets(binding: DatasetEvidenceBinding) -> EvaluatorTargets:
    dataset = _read_bound_dataset(binding)
    evaluator = dataset[list(binding.retained_cell_ids), list(binding.gene_ids)].copy()
    return evaluator_targets_from_dataset(
        evaluator,
        trajectory_root_cell_id=binding.trajectory_root_cell_id,
        trajectory_source_id=binding.trajectory_source_id,
    )


def _ensure_directory(path: Path, name: str) -> None:
    _reject_symlink_chain(path, name)
    try:
        path.mkdir(parents=True, exist_ok=True)
        _reject_symlink_chain(path, name)
        metadata = path.lstat()
    except OSError as error:
        raise DownstreamEvidenceError(f"{name} cannot be created") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DownstreamEvidenceError(f"{name} must be a non-symlink directory")


def _publish_immutable(path: Path, raw: bytes, name: str) -> str:
    _ensure_directory(path.parent, f"{name} parent")
    if os.path.lexists(path):
        observed, digest = _stable_file_bytes(path, name)
        if observed != raw:
            raise DownstreamEvidenceError(f"refusing to replace immutable {name}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed, _digest_value = _stable_file_bytes(path, name)
            if observed != raw:
                raise DownstreamEvidenceError(f"conflicting immutable {name}")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _expected_record_common(
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    binding: DatasetEvidenceBinding,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "ordinal": entry.ordinal,
        "source_kind": plan.source_kind,
        "source_record_path": entry.source_record_path,
        "source_record_sha256": entry.source_record_sha256,
        "run_id": entry.run_id,
        "runner_method_id": entry.method_id,
        "method": _analysis_method(plan, entry),
        "dataset_id": entry.dataset_id,
        "dataset_sha256": entry.source_dataset_sha256,
        "dataset_file_sha256": binding.file_sha256,
        "mechanism": entry.mechanism,
        "biological_id": entry.biological_id,
        "technical_view": entry.technical_view,
        "model_seed": entry.model_seed,
        "configuration_id": entry.configuration_id,
        "configuration_sha256": entry.configuration_sha256,
        "method_artifact_sha256": entry.method_artifact_sha256,
        "run_status": entry.status,
        "run_reason": entry.reason,
    }


def _endpoint_record_from_row(
    row: Mapping[str, object], entry: DownstreamPlanEntry
) -> EndpointRecord:
    status = row.get("status") if entry.status == "completed" else "unavailable"
    reason = (
        row.get("reason_code")
        if entry.status == "completed"
        else "upstream_run_not_completed"
    )
    try:
        endpoint = EndpointRecord(
            endpoint=row.get("endpoint"),
            value=row.get("value"),
            status=status,
            reason=reason,
            direction=row.get("direction"),
            independent_unit=row.get("independent_unit"),
            independent_n=row.get("independent_n"),
            descriptive_n=row.get("descriptive_n"),
            descriptive_unit=row.get("descriptive_unit"),
            procedure=row.get("procedure"),
            family_id=row.get("family_id"),
            family_size=row.get("family_size"),
            alpha=row.get("alpha"),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise DownstreamEvidenceError("downstream endpoint contract differs") from error
    if endpoint.status == "completed":
        lower, upper = _ENDPOINT_VALUE_RANGES[endpoint.endpoint]
        if not lower <= float(endpoint.value) <= upper:
            raise DownstreamEvidenceError("downstream endpoint value is out of range")
    procedure_allowed = (
        endpoint.procedure in _TERMINAL_PROCEDURES
        or endpoint.procedure in _ENDPOINT_PROCEDURES[endpoint.endpoint]
        or (
            endpoint.endpoint in {"clustering_ari_loss", "clustering_nmi_loss"}
            and endpoint.procedure.startswith(
                f"{_CLUSTERING_PROCEDURE_PREFIX}_selected_k="
            )
        )
    )
    if not procedure_allowed:
        raise DownstreamEvidenceError("downstream endpoint procedure differs")
    family = (endpoint.family_id, endpoint.family_size, endpoint.alpha)
    if endpoint.endpoint in {
        "positive_de_marker_recall",
        "positive_de_false_discovery_rate",
    }:
        if endpoint.family_id is not None and (
            endpoint.family_id != "one_vs_rest_all_groups_all_genes"
            or endpoint.alpha != 0.05
        ):
            raise DownstreamEvidenceError(
                "downstream endpoint family authority differs"
            )
    elif any(value is not None for value in family):
        raise DownstreamEvidenceError("downstream endpoint family is unexpected")
    return endpoint


def _validate_endpoint_rows(
    rows: object,
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
) -> None:
    endpoint_names = _endpoint_names(plan)
    if not isinstance(rows, list) or len(rows) != len(endpoint_names):
        raise DownstreamEvidenceError("downstream record endpoint count differs")
    for expected_endpoint, row in zip(endpoint_names, rows, strict=True):
        if not isinstance(row, dict) or set(row) != _ENDPOINT_ROW_FIELDS:
            raise DownstreamEvidenceError("downstream endpoint row schema differs")
        expected_metadata = {
            "source_kind": plan.source_kind,
            "run_id": entry.run_id,
            "runner_method_id": entry.method_id,
            "method": _analysis_method(plan, entry),
            "dataset_id": entry.dataset_id,
            "dataset_sha256": entry.source_dataset_sha256,
            "mechanism": entry.mechanism,
            "biological_id": entry.biological_id,
            "technical_view": entry.technical_view,
            "model_seed": entry.model_seed,
            "configuration_id": entry.configuration_id,
            "configuration_sha256": entry.configuration_sha256,
            "method_artifact_sha256": entry.method_artifact_sha256,
            "endpoint": expected_endpoint,
            "upstream_status": entry.status,
            "upstream_reason": entry.reason,
        }
        if any(row.get(key) != value for key, value in expected_metadata.items()):
            raise DownstreamEvidenceError("downstream endpoint identity differs")
        if (
            row.get("independent_unit") != "biological_draw"
            or row.get("independent_n") != 1
        ):
            raise DownstreamEvidenceError("downstream independent unit differs")
        value = row.get("value")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
        ):
            raise DownstreamEvidenceError("downstream endpoint value is invalid")
        _endpoint_record_from_row(row, entry)
        if entry.status == "completed":
            if row.get("status") not in {"completed", "unavailable"}:
                raise DownstreamEvidenceError("completed downstream status differs")
            if row.get("status") == "completed" and (
                value is None or row.get("reason_code") is not None
            ):
                raise DownstreamEvidenceError("completed endpoint reason differs")
            if row.get("status") == "unavailable" and (
                value is not None or not isinstance(row.get("reason_code"), str)
            ):
                raise DownstreamEvidenceError("unavailable endpoint reason differs")
        elif (
            row.get("status") != entry.status
            or value is not None
            or row.get("reason_code") != "upstream_run_not_completed"
            or row.get("descriptive_n") != 0
        ):
            raise DownstreamEvidenceError("upstream failure was not preserved")


def _load_record(
    path: Path,
    plan: DownstreamEvidencePlan,
    entry: DownstreamPlanEntry,
    binding: DatasetEvidenceBinding,
    targets: EvaluatorTargets | None = None,
) -> dict[str, object]:
    value, _raw, _file_sha = _strict_json(path, "downstream record")
    if set(value) != {*_RECORD_BODY_FIELDS, "record_sha256"}:
        raise DownstreamEvidenceError("downstream record schema differs")
    body = {key: nested for key, nested in value.items() if key != "record_sha256"}
    if value.get("record_sha256") != canonical_sha256(body):
        raise DownstreamEvidenceError("downstream record checksum differs")
    common = _expected_record_common(plan, entry, binding)
    if any(value.get(key) != nested for key, nested in common.items()):
        raise DownstreamEvidenceError("downstream record identity differs")
    _validate_endpoint_rows(value.get("endpoints"), plan, entry)
    if entry.status == "completed" and targets is None:
        targets = _load_targets(binding)
    expected = _evaluate_entry(plan, entry, binding, targets)
    if _canonical_bytes(value) != _canonical_bytes(expected):
        raise DownstreamEvidenceError("downstream endpoint re-evaluation differs")
    return value


def _record_names(output_root: Path, planned: int) -> tuple[str, ...]:
    records_root = output_root / "records"
    if not os.path.lexists(records_root):
        return ()
    try:
        metadata = records_root.lstat()
        names = tuple(sorted(path.name for path in records_root.iterdir()))
    except OSError as error:
        raise DownstreamEvidenceError(
            "downstream record directory is invalid"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise DownstreamEvidenceError("downstream record directory is invalid")
    expected = tuple(f"{index:08d}.json" for index in range(1, len(names) + 1))
    if names != expected or len(names) > planned:
        raise DownstreamEvidenceError("downstream records are not a canonical prefix")
    return names


def _load_record_prefix(
    output_root: Path, plan: DownstreamEvidencePlan
) -> tuple[dict[str, object], ...]:
    bindings = {value.dataset_id: value for value in plan.datasets}
    names = _record_names(output_root, len(plan.entries))
    targets: dict[str, EvaluatorTargets] = {}
    records: list[dict[str, object]] = []
    for index, name in enumerate(names):
        entry = plan.entries[index]
        binding = bindings[entry.dataset_id]
        target = None
        if entry.status == "completed":
            target = targets.get(entry.dataset_id)
            if target is None:
                target = _load_targets(binding)
                targets[entry.dataset_id] = target
        records.append(
            _load_record(
                output_root / "records" / name,
                plan,
                entry,
                binding,
                target,
            )
        )
    return tuple(records)


def _rebuild_development_bundle(
    plan: DownstreamEvidencePlan,
) -> DownstreamEvidencePlan:
    repository = _existing_directory(plan.source_root, "development bundle repository")
    sources: list[DevelopmentSourcePlan] = []
    for binding in plan.development_sources:
        source_root = _safe_relative(
            repository, binding.source_root, "development component source"
        )
        expected_manifest = str(PurePosixPath(binding.source_root) / "checkpoint.json")
        if binding.manifest_path != expected_manifest:
            raise DownstreamEvidenceError(
                "development component checkpoint path is not fixed"
            )
        component = build_downstream_evidence_plan(
            source_root,
            source_kind="development",
            evidence_scope="all",
            datasets=plan.datasets,
            configurations=plan.configurations,
            _source_plan_authority="independent",
        )
        sources.append(
            DevelopmentSourcePlan(
                source_id=binding.source_id,
                plan=component,
                selected_methods=binding.selected_methods,
                evaluation_manifest_path=binding.evaluation_manifest_path,
                evaluation_manifest_file_sha256=(
                    binding.evaluation_manifest_file_sha256
                ),
                evaluation_manifest_payload_sha256=(
                    binding.evaluation_manifest_payload_sha256
                ),
                evaluation_source_pointer=binding.evaluation_source_pointer,
                evaluation_source_sha256=binding.evaluation_source_sha256,
            )
        )
    return combine_development_downstream_evidence_plans(
        repository,
        tuple(sources),
        revision_versions=plan.development_revision_versions,
    )


def _revalidate_plan(plan: DownstreamEvidencePlan) -> None:
    if not isinstance(plan, DownstreamEvidencePlan):
        raise TypeError("plan must be a DownstreamEvidencePlan")
    if plan.plan_sha256 != canonical_sha256(plan.body()):
        raise DownstreamEvidenceError("downstream plan checksum differs")
    if plan.source_kind == "final":
        binding = plan.evaluated_round_binding
        if binding is None:
            raise DownstreamEvidenceError("final evaluated-round binding is required")
        if plan.evidence_scope == "all":
            rebuilt = build_final_downstream_evidence_plan(
                binding.repository_root,
                binding.round_root,
            )
        elif plan.evidence_scope == "supplementary_trajectory":
            rebuilt = build_final_trajectory_downstream_evidence_plan(
                binding.repository_root,
                binding.round_root,
            )
        else:
            raise DownstreamEvidenceError("persisted final downstream scope differs")
    elif plan.development_sources:
        rebuilt = _rebuild_development_bundle(plan)
    else:
        rebuilt = build_downstream_evidence_plan(
            plan.source_root,
            source_kind=plan.source_kind,
            evidence_scope=plan.evidence_scope,
            datasets=plan.datasets,
            configurations=plan.configurations,
            evaluated_round_binding=plan.evaluated_round_binding,
            _source_plan_authority=plan.source_plan_authority,
        )
    if rebuilt.to_dict() != plan.to_dict():
        raise DownstreamEvidenceError("downstream plan sources changed")


def _manifest_payload(
    output_root: Path,
    plan: DownstreamEvidencePlan,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    references = []
    for entry in plan.entries:
        path = output_root / "records" / f"{entry.ordinal:08d}.json"
        raw, file_sha = _stable_file_bytes(path, "downstream record")
        references.append(
            {
                "ordinal": entry.ordinal,
                "run_id": entry.run_id,
                "path": f"records/{entry.ordinal:08d}.json",
                "sha256": file_sha,
                "record_sha256": json.loads(raw.decode("utf-8"))["record_sha256"],
            }
        )
    plan_raw, plan_file_sha = _stable_file_bytes(
        output_root / "plan.json", "downstream plan"
    )
    plan_payload = json.loads(plan_raw.decode("utf-8"))
    body: dict[str, object] = {
        "schema_version": 3,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": plan_file_sha,
        "source_kind": plan.source_kind,
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "source_manifest_path": plan.source_manifest_path,
        "source_manifest_file_sha256": plan.source_manifest_file_sha256,
        "source_manifest_payload_sha256": plan.source_manifest_payload_sha256,
        "source_plan_sha256": plan.source_plan_sha256,
        "source_input_hashes_sha256": plan.source_input_hashes_sha256,
        "source_statuses_sha256": plan.source_statuses_sha256,
        "source_plan_authority": plan.source_plan_authority,
        "evaluated_round_binding_sha256": (
            None
            if plan.evaluated_round_binding is None
            else plan.evaluated_round_binding.binding_sha256
        ),
        "development_revision_versions": list(plan.development_revision_versions),
        "development_sources": [value.to_dict() for value in plan.development_sources],
        "planned_denominator_count": len(plan.entries),
        "recorded_denominator_count": len(records),
        "endpoint_row_count": len(records) * len(_endpoint_names(plan)),
        "records": references,
    }
    if plan_payload.get("plan_sha256") != plan.plan_sha256:
        raise DownstreamEvidenceError("persisted downstream plan differs")
    return {**body, "manifest_sha256": canonical_sha256(body)}


def expected_final_downstream_output_directory(
    plan: DownstreamEvidencePlan,
) -> Path:
    """Return the sole receipt-bound external namespace for a final plan."""

    if not isinstance(plan, DownstreamEvidencePlan):
        raise TypeError("plan must be a DownstreamEvidencePlan")
    binding = plan.evaluated_round_binding
    if plan.source_kind != "final" or binding is None:
        raise DownstreamEvidenceError(
            "final downstream plan lacks an evaluated-round receipt binding"
        )
    if plan.evidence_scope not in {"all", "supplementary_trajectory"}:
        raise DownstreamEvidenceError("final downstream evidence scope differs")
    repository_root = Path(binding.repository_root)
    if (
        not repository_root.is_absolute()
        or not repository_root.name
        or Path(binding.round_root).name != binding.round_id
        or _SHA256.fullmatch(binding.evaluation_receipt_payload_sha256) is None
    ):
        raise DownstreamEvidenceError(
            "final downstream receipt namespace binding differs"
        )
    base = (
        repository_root.parent
        / f"{repository_root.name}-final-analysis"
        / "downstream"
        / binding.round_id
        / binding.evaluation_receipt_payload_sha256
    )
    return (
        base / "trajectory"
        if plan.evidence_scope == "supplementary_trajectory"
        else base
    )


def _validate_downstream_output_location(
    plan: DownstreamEvidencePlan, output_root: Path
) -> None:
    binding = plan.evaluated_round_binding
    if binding is None:
        if plan.source_kind == "final":
            raise DownstreamEvidenceError("final evaluated-round binding is required")
        return
    expected = expected_final_downstream_output_directory(plan)
    if output_root != expected:
        raise DownstreamEvidenceError(
            "final downstream output differs from its receipt-bound namespace"
        )
    repository_root = Path(binding.repository_root)
    try:
        output_root.relative_to(repository_root)
    except ValueError:
        return
    raise DownstreamEvidenceError(
        "receipt-bound final downstream output must be outside the frozen repository"
    )


def run_downstream_evidence(
    plan: DownstreamEvidencePlan,
    output_directory: str | Path,
    *,
    max_denominators: int | None = None,
) -> dict[str, object]:
    """Resume the plan prefix and emit its closed endpoint set for each run."""

    if max_denominators is not None and (
        isinstance(max_denominators, bool)
        or type(max_denominators) is not int
        or max_denominators < 0
    ):
        raise ValueError("max_denominators must be a nonnegative integer or null")
    _revalidate_plan(plan)
    output_root = Path(output_directory).absolute()
    _validate_downstream_output_location(plan, output_root)
    _ensure_directory(output_root, "downstream output directory")
    manifest_path = output_root / "downstream_manifest.json"
    if os.path.lexists(manifest_path):
        persisted_plan = load_downstream_evidence_plan(output_root)
        if persisted_plan.to_dict() != plan.to_dict():
            raise DownstreamEvidenceError(
                "completed downstream plan differs from requested plan"
            )
        completed = load_downstream_evidence_manifest(output_root)
        return dict(completed.payload)
    plan_raw = _canonical_bytes(plan.to_dict()) + b"\n"
    _publish_immutable(output_root / "plan.json", plan_raw, "downstream plan")
    records = list(_load_record_prefix(output_root, plan))
    remaining = len(plan.entries) - len(records)
    count = remaining if max_denominators is None else min(remaining, max_denominators)
    bindings = {value.dataset_id: value for value in plan.datasets}
    targets: dict[str, EvaluatorTargets] = {}
    for entry in plan.entries[len(records) : len(records) + count]:
        binding = bindings[entry.dataset_id]
        target = None
        if entry.status == "completed":
            target = targets.get(entry.dataset_id)
            if target is None:
                target = _load_targets(binding)
                targets[entry.dataset_id] = target
        record = _evaluate_entry(plan, entry, binding, target)
        raw = _canonical_bytes(record) + b"\n"
        _publish_immutable(
            output_root / "records" / f"{entry.ordinal:08d}.json",
            raw,
            "downstream record",
        )
        records.append(record)
    if len(records) != len(plan.entries):
        return {
            "schema_version": 1,
            "status": "running",
            "plan_sha256": plan.plan_sha256,
            "planned_denominator_count": len(plan.entries),
            "recorded_denominator_count": len(records),
            "endpoint_row_count": len(records) * len(_endpoint_names(plan)),
        }
    _revalidate_plan(plan)
    manifest = _manifest_payload(output_root, plan, records)
    _publish_immutable(
        manifest_path,
        _canonical_bytes(manifest) + b"\n",
        "downstream manifest",
    )
    return dict(manifest)


@dataclass(frozen=True, slots=True)
class DownstreamEvidenceManifest:
    plan_sha256: str
    manifest_sha256: str
    planned_denominator_count: int
    endpoint_row_count: int
    records: tuple[Mapping[str, object], ...]
    payload: Mapping[str, object]


def _load_persisted_plan(
    output_root: Path,
) -> tuple[DownstreamEvidencePlan, dict[str, object], str]:
    plan_payload, _plan_raw, plan_file_sha = _strict_json(
        output_root / "plan.json", "downstream plan"
    )
    if set(plan_payload) != _PLAN_FIELDS or plan_payload.get("schema_version") != 3:
        raise DownstreamEvidenceError("persisted downstream plan schema differs")
    plan_sha = _digest(plan_payload.get("plan_sha256"), "downstream plan checksum")
    plan_body = {
        key: value for key, value in plan_payload.items() if key != "plan_sha256"
    }
    if canonical_sha256(plan_body) != plan_sha:
        raise DownstreamEvidenceError("downstream plan checksum differs")
    raw_datasets = plan_payload.get("datasets")
    if not isinstance(raw_datasets, list):
        raise DownstreamEvidenceError("persisted downstream datasets are invalid")
    persisted_datasets = tuple(
        _dataset_binding_from_payload(value) for value in raw_datasets
    )
    raw_configurations = plan_payload.get("configurations")
    if not isinstance(raw_configurations, list):
        raise DownstreamEvidenceError("persisted downstream configurations are invalid")
    persisted_configurations = tuple(
        _configuration_from_payload(value) for value in raw_configurations
    )
    evaluated_round_binding = _evaluated_round_binding_from_payload(
        plan_payload.get("evaluated_round_binding")
    )
    raw_versions = plan_payload.get("development_revision_versions")
    raw_sources = plan_payload.get("development_sources")
    if not isinstance(raw_versions, list) or not isinstance(raw_sources, list):
        raise DownstreamEvidenceError("persisted development source bundle is invalid")
    versions = tuple(raw_versions)
    sources = tuple(
        _development_source_binding_from_payload(value) for value in raw_sources
    )
    if bool(versions) != bool(sources):
        raise DownstreamEvidenceError(
            "persisted development source bundle is incomplete"
        )
    try:
        if sources:
            skeleton = DownstreamEvidencePlan(
                source_root=_text(
                    plan_payload.get("source_root"), "persisted source root"
                ),
                source_kind=_text(
                    plan_payload.get("source_kind"), "persisted source kind"
                ),
                evidence_scope=_text(
                    plan_payload.get("evidence_scope"), "persisted evidence scope"
                ),
                evaluator_source_sha256=_digest(
                    plan_payload.get("evaluator_source_sha256"),
                    "persisted evaluator source checksum",
                ),
                source_manifest_path=_text(
                    plan_payload.get("source_manifest_path"),
                    "persisted source manifest path",
                ),
                source_manifest_file_sha256=_digest(
                    plan_payload.get("source_manifest_file_sha256"),
                    "persisted source manifest file checksum",
                ),
                source_manifest_payload_sha256=_digest(
                    plan_payload.get("source_manifest_payload_sha256"),
                    "persisted source manifest payload checksum",
                ),
                source_plan_sha256=_digest(
                    plan_payload.get("source_plan_sha256"),
                    "persisted source plan checksum",
                ),
                source_input_hashes_sha256=_digest(
                    plan_payload.get("source_input_hashes_sha256"),
                    "persisted source input checksum",
                ),
                source_statuses_sha256=_digest(
                    plan_payload.get("source_statuses_sha256"),
                    "persisted source statuses checksum",
                ),
                source_plan_authority=_text(
                    plan_payload.get("source_plan_authority"),
                    "persisted source plan authority",
                ),
                evaluated_round_binding=evaluated_round_binding,
                development_revision_versions=versions,
                development_sources=sources,
                datasets=persisted_datasets,
                configurations=persisted_configurations,
                entries=(),
                plan_sha256=plan_sha,
            )
            if (
                skeleton.source_kind != "development"
                or skeleton.evidence_scope != "selection_primary"
                or evaluated_round_binding is not None
            ):
                raise DownstreamEvidenceError(
                    "persisted development source bundle identity differs"
                )
            rebuilt = _rebuild_development_bundle(skeleton)
        else:
            source_kind = _text(
                plan_payload.get("source_kind"), "persisted source kind"
            )
            evidence_scope = _text(
                plan_payload.get("evidence_scope"), "persisted evidence scope"
            )
            if source_kind == "final":
                if evaluated_round_binding is None:
                    raise DownstreamEvidenceError(
                        "persisted final evaluated-round binding is absent"
                    )
                if evidence_scope == "all":
                    rebuilt = build_final_downstream_evidence_plan(
                        evaluated_round_binding.repository_root,
                        evaluated_round_binding.round_root,
                    )
                elif evidence_scope == "supplementary_trajectory":
                    rebuilt = build_final_trajectory_downstream_evidence_plan(
                        evaluated_round_binding.repository_root,
                        evaluated_round_binding.round_root,
                    )
                else:
                    raise DownstreamEvidenceError(
                        "persisted final downstream scope differs"
                    )
            else:
                rebuilt = build_downstream_evidence_plan(
                    _text(plan_payload.get("source_root"), "persisted source root"),
                    source_kind=source_kind,
                    evidence_scope=evidence_scope,
                    datasets=persisted_datasets,
                    configurations=persisted_configurations,
                    evaluated_round_binding=evaluated_round_binding,
                    _source_plan_authority=_text(
                        plan_payload.get("source_plan_authority"),
                        "persisted source plan authority",
                    ),
                )
    except DownstreamEvidenceError:
        raise
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "persisted downstream plan cannot be revalidated"
        ) from error
    if rebuilt.to_dict() != plan_payload:
        raise DownstreamEvidenceError("persisted downstream plan sources changed")
    _validate_downstream_output_location(rebuilt, output_root)
    return rebuilt, plan_payload, plan_file_sha


def load_downstream_evidence_plan(
    output_directory: str | Path,
) -> DownstreamEvidencePlan:
    """Reload a persisted plan after revalidating all source and dataset bindings."""

    output_root = Path(output_directory).absolute()
    _reject_symlink_chain(output_root, "downstream output directory")
    plan, _payload, _file_sha = _load_persisted_plan(output_root)
    return plan


def load_downstream_evidence_manifest(
    output_directory: str | Path,
) -> DownstreamEvidenceManifest:
    """Validate a complete downstream manifest and every referenced row file."""

    output_root = Path(output_directory).absolute()
    _reject_symlink_chain(output_root, "downstream output directory")
    rebuilt, plan_payload, plan_file_sha = _load_persisted_plan(output_root)
    plan_sha = rebuilt.plan_sha256
    manifest, _manifest_raw, _manifest_file_sha = _strict_json(
        output_root / "downstream_manifest.json", "downstream manifest"
    )
    if set(manifest) != _DOWNSTREAM_MANIFEST_FIELDS:
        raise DownstreamEvidenceError("downstream manifest schema differs")
    manifest_sha = _digest(
        manifest.get("manifest_sha256"), "downstream manifest checksum"
    )
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    references = manifest.get("records")
    planned = manifest.get("planned_denominator_count")
    endpoint_rows = manifest.get("endpoint_row_count")
    if (
        canonical_sha256(manifest_body) != manifest_sha
        or manifest.get("schema_version") != 3
        or manifest.get("status") != "completed"
        or manifest.get("plan_sha256") != plan_sha
        or manifest.get("plan_file_sha256") != plan_file_sha
        or manifest.get("source_kind") != rebuilt.source_kind
        or manifest.get("evaluator_source_sha256") != rebuilt.evaluator_source_sha256
        or manifest.get("source_manifest_path") != rebuilt.source_manifest_path
        or manifest.get("source_manifest_file_sha256")
        != rebuilt.source_manifest_file_sha256
        or manifest.get("source_manifest_payload_sha256")
        != rebuilt.source_manifest_payload_sha256
        or manifest.get("source_plan_sha256") != rebuilt.source_plan_sha256
        or manifest.get("source_input_hashes_sha256")
        != rebuilt.source_input_hashes_sha256
        or manifest.get("source_statuses_sha256") != rebuilt.source_statuses_sha256
        or manifest.get("source_plan_authority") != rebuilt.source_plan_authority
        or manifest.get("evaluated_round_binding_sha256")
        != (
            None
            if rebuilt.evaluated_round_binding is None
            else rebuilt.evaluated_round_binding.binding_sha256
        )
        or manifest.get("development_revision_versions")
        != list(rebuilt.development_revision_versions)
        or manifest.get("development_sources")
        != [value.to_dict() for value in rebuilt.development_sources]
        or type(planned) is not int
        or planned <= 0
        or manifest.get("recorded_denominator_count") != planned
        or endpoint_rows != planned * len(_endpoint_names(rebuilt))
        or not isinstance(references, list)
        or len(references) != planned
        or plan_payload.get("planned_denominator_count") != planned
    ):
        raise DownstreamEvidenceError("downstream manifest completeness differs")
    records: list[Mapping[str, object]] = []
    bindings = {value.dataset_id: value for value in rebuilt.datasets}
    targets: dict[str, EvaluatorTargets] = {}
    for ordinal, reference in enumerate(references, start=1):
        if (
            not isinstance(reference, Mapping)
            or set(reference)
            != {"ordinal", "run_id", "path", "sha256", "record_sha256"}
            or reference.get("ordinal") != ordinal
        ):
            raise DownstreamEvidenceError(
                "downstream manifest references are unordered"
            )
        entry = rebuilt.entries[ordinal - 1]
        binding = bindings[entry.dataset_id]
        target = None
        if entry.status == "completed":
            target = targets.get(entry.dataset_id)
            if target is None:
                target = _load_targets(binding)
                targets[entry.dataset_id] = target
        path = _safe_relative(output_root, reference.get("path"), "downstream record")
        record = _load_record(
            path,
            rebuilt,
            entry,
            binding,
            target,
        )
        _record_raw, record_file_sha = _stable_file_bytes(path, "downstream record")
        if (
            record_file_sha
            != _digest(reference.get("sha256"), "downstream record file checksum")
            or record.get("record_sha256")
            != _digest(
                reference.get("record_sha256"), "downstream record payload checksum"
            )
            or record.get("run_id") != reference.get("run_id")
            or record.get("ordinal") != ordinal
        ):
            raise DownstreamEvidenceError("downstream manifest record binding differs")
        records.append(MappingProxyType(record))
    return DownstreamEvidenceManifest(
        plan_sha256=plan_sha,
        manifest_sha256=manifest_sha,
        planned_denominator_count=planned,
        endpoint_row_count=endpoint_rows,
        records=tuple(records),
        payload=MappingProxyType(manifest),
    )


def downstream_denominator_key(record: Mapping[str, object]) -> tuple[object, ...]:
    """Return the fixed selection-completeness identity for one denominator."""

    return (
        record.get("mechanism"),
        record.get("biological_id"),
        record.get("technical_view"),
        record.get("dataset_id"),
        record.get("dataset_sha256"),
        record.get("method"),
        record.get("method_artifact_sha256"),
        record.get("model_seed"),
    )


def downstream_source_statuses(
    output_directory: str | Path,
) -> Mapping[tuple[object, ...], str]:
    """Return run statuses rederived from the independently loaded source plan."""

    plan = load_downstream_evidence_plan(output_directory)
    statuses: dict[tuple[object, ...], str] = {}
    for entry in plan.entries:
        key = (
            entry.mechanism,
            entry.biological_id,
            entry.technical_view,
            entry.dataset_id,
            entry.source_dataset_sha256,
            _analysis_method(plan, entry),
            entry.method_artifact_sha256,
            entry.model_seed,
        )
        if key in statuses:
            raise DownstreamEvidenceError(
                "source plan denominator identities are duplicated"
            )
        statuses[key] = entry.status
    return MappingProxyType(statuses)


def validate_downstream_evidence_completeness(
    output_directory: str | Path,
    *,
    expected_denominators: Sequence[tuple[object, ...]] | None = None,
) -> DownstreamEvidenceManifest:
    """Require complete closed-scope evidence and an optional denominator set."""

    manifest = load_downstream_evidence_manifest(output_directory)
    observed = tuple(downstream_denominator_key(record) for record in manifest.records)
    if len(set(observed)) != len(observed):
        raise DownstreamEvidenceError(
            "downstream denominator identities are duplicated"
        )
    if expected_denominators is not None:
        expected = tuple(expected_denominators)
        if len(set(expected)) != len(expected) or set(observed) != set(expected):
            raise DownstreamEvidenceError("downstream denominator completeness differs")
    return manifest


__all__ = [
    "DatasetEvidenceBinding",
    "DevelopmentSourceBinding",
    "DevelopmentSourcePlan",
    "DownstreamEvidenceError",
    "DownstreamEvidenceManifest",
    "DownstreamEvidencePlan",
    "EvaluatedRoundBinding",
    "bind_evaluator_dataset",
    "bind_prepared_evaluator_panel",
    "build_development_downstream_evidence_plan",
    "build_downstream_evidence_plan",
    "build_final_downstream_evidence_plan",
    "build_final_trajectory_downstream_evidence_plan",
    "combine_development_downstream_evidence_plans",
    "development_downstream_revision_version",
    "downstream_denominator_key",
    "downstream_source_statuses",
    "expected_final_downstream_output_directory",
    "load_downstream_evidence_manifest",
    "load_downstream_evidence_plan",
    "run_downstream_evidence",
    "validate_direct_comparator_projection",
    "validate_downstream_evidence_completeness",
]
