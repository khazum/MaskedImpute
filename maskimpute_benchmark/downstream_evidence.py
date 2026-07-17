"""Hash-bound resumable downstream evidence over persisted runner outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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

from .downstream_evaluation import (
    DOWNSTREAM_ENDPOINT_NAMES,
    EndpointRecord,
    EvaluatorTargets,
    MethodOutput,
    evaluate_downstream_endpoints,
    evaluator_targets_from_dataset,
    terminal_downstream_endpoints,
)
from .protocol import canonical_sha256
from .runner import AuthorizedConfiguration
from .schema import benchmark_dataset_sha256, validate_benchmark_dataset


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_OUTPUT_ENCODING = "zlib_raw_f64_v1"
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
        "result_files",
    }
)
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
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "evaluated_round_binding_sha256",
        "planned_denominator_count",
        "recorded_denominator_count",
        "endpoint_row_count",
        "records",
        "manifest_sha256",
    }
)


class DownstreamEvidenceError(ValueError):
    """Raised when source, dataset, resume, or output evidence is invalid."""


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
        cells = _stable_ids(self.retained_cell_ids, "retained_cell_ids")
        genes = _stable_ids(self.gene_ids, "gene_ids")
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

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_root": self.repository_root,
            "round_root": self.round_root,
            "round_id": self.round_id,
            "evaluation_receipt_path": self.evaluation_receipt_path,
            "evaluation_receipt_file_sha256": (
                self.evaluation_receipt_file_sha256
            ),
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
    }
)


@dataclass(frozen=True, slots=True)
class DownstreamEvidencePlan:
    source_root: str
    source_kind: str
    evaluator_source_sha256: str
    source_manifest_path: str
    source_manifest_file_sha256: str
    source_manifest_payload_sha256: str
    evaluated_round_binding: EvaluatedRoundBinding | None
    datasets: tuple[DatasetEvidenceBinding, ...]
    configurations: tuple[AuthorizedConfiguration, ...]
    entries: tuple[DownstreamPlanEntry, ...]
    plan_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "source_root": self.source_root,
            "source_kind": self.source_kind,
            "evaluator_source_sha256": self.evaluator_source_sha256,
            "source_manifest_path": self.source_manifest_path,
            "source_manifest_file_sha256": self.source_manifest_file_sha256,
            "source_manifest_payload_sha256": self.source_manifest_payload_sha256,
            "evaluated_round_binding": (
                None
                if self.evaluated_round_binding is None
                else self.evaluated_round_binding.to_dict()
            ),
            "datasets": [value.to_dict() for value in self.datasets],
            "configurations": [value.to_dict() for value in self.configurations],
            "entries": [value.to_dict() for value in self.entries],
            "planned_denominator_count": len(self.entries),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.body(), "plan_sha256": self.plan_sha256}


_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "source_root",
        "source_kind",
        "evaluator_source_sha256",
        "source_manifest_path",
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "evaluated_round_binding",
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
        or payload.get("schema")
        != "maskimpute-registry-default-configuration-v1"
        or not isinstance(method, Mapping)
        or method.get("id") != configuration.method_id
    ):
        raise DownstreamEvidenceError(
            "registry configuration method payload differs"
        )
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
            raise DownstreamEvidenceError(
                "evaluated final round receipt is absent"
            )
        result_manifest = receipt.get("result_manifest")
        if not isinstance(result_manifest, Mapping):
            raise DownstreamEvidenceError(
                "evaluated final result manifest is absent"
            )
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
    validated_receipt = _validated_evaluated_round_receipt(
        repository_root, round_root
    )
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
    algorithmic_failure_count = validation.get(
        "executed_algorithmic_failure_count"
    )
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
        raise DownstreamEvidenceError(
            "final execution validation denominator differs"
        )
    assert isinstance(validation_planned, int)
    assert isinstance(completed_count, int)
    assert isinstance(algorithmic_failure_count, int)
    assert isinstance(not_applicable_count, int)
    assert isinstance(status_counts, Mapping)
    if (
        sum(int(count) for count in status_counts.values())
        + not_applicable_count
        != validation_planned
        or int(status_counts.get("completed", 0)) != completed_count
        or sum(
            int(status_counts.get(status, 0))
            for status in terminal_statuses - {"completed"}
        )
        != algorithmic_failure_count
    ):
        raise DownstreamEvidenceError(
            "final execution validation denominator differs"
        )

    final_manifest_relative = result_manifest.get(
        "final_execution_manifest_path"
    )
    if final_manifest_relative != (
        "results/final/execution/execution_manifest.json"
    ):
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
        key: value
        for key, value in final_manifest.items()
        if key != "manifest_sha256"
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
        raise DownstreamEvidenceError(
            "final execution validation denominator differs"
        )
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
        if (
            record_file_sha256
            != _digest(reference.get("sha256"), "final record checksum")
            or canonical_sha256(record)
            != _digest(payload_sha256, "final record payload checksum")
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
            observed_status_counts[status] = (
                observed_status_counts.get(status, 0) + 1
            )
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
    )
    if (
        binding.evaluation_receipt_path != "evaluation_receipt.json"
        or binding.final_execution_manifest_path
        != "results/final/execution/execution_manifest.json"
        or Path(binding.round_root).name != binding.round_id
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
    if configuration.to_dict() != dict(value):
        raise DownstreamEvidenceError("persisted configuration authority differs")
    _method_artifact_sha256(configuration)
    return configuration


def _dataset_binding_from_payload(value: object) -> DatasetEvidenceBinding:
    if not isinstance(value, Mapping) or set(value) != _DATASET_BINDING_FIELDS:
        raise DownstreamEvidenceError("persisted dataset binding schema differs")
    retained = value.get("retained_cell_ids")
    genes = value.get("gene_ids")
    string_fields = ("dataset_id", "path", "file_sha256", "dataset_sha256")
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


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    ordinal: int
    path: str
    sha256: str
    payload: Mapping[str, object]


def _validate_source_record_schema(
    record: Mapping[str, object], *, source_kind: str
) -> None:
    expected_record_fields = (
        {"run", "metrics"}
        if source_kind == "development"
        else {"run", "metrics", "execution_request"}
    )
    if set(record) != expected_record_fields:
        raise DownstreamEvidenceError(f"{source_kind} source record schema differs")
    run = record.get("run")
    expected_run_fields = (
        _DEVELOPMENT_RUN_FIELDS
        if source_kind == "development"
        else _FINAL_RUN_FIELDS
    )
    if not isinstance(run, Mapping) or set(run) != expected_run_fields:
        raise DownstreamEvidenceError("source run schema differs")
    metrics = record.get("metrics")
    if not isinstance(metrics, list) or any(
        not isinstance(metric, Mapping) or set(metric) != _METRIC_FIELDS
        for metric in metrics
    ):
        raise DownstreamEvidenceError("source metric schema differs")
    if source_kind == "final":
        request = record.get("execution_request")
        if request is not None and (
            not isinstance(request, Mapping)
            or set(request) != _FINAL_EXECUTION_REQUEST_FIELDS
        ):
            raise DownstreamEvidenceError(
                "final execution request schema differs"
            )
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
            raise DownstreamEvidenceError(
                "final native output storage policy differs"
            )


def _development_source(
    root: Path,
) -> tuple[str, str, str, tuple[_SourceRecord, ...]]:
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
    return "checkpoint.json", file_sha, checksum, tuple(result)


def _final_source(root: Path) -> tuple[str, str, str, tuple[_SourceRecord, ...]]:
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
            raise DownstreamEvidenceError(
                "final artifact storage policy differs"
            )
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
    return "execution_manifest.json", file_sha, checksum, tuple(result)


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
    configurations: Mapping[
        tuple[str, str, str, str], AuthorizedConfiguration
    ],
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
    if run.get("retained_cell_count") != len(binding.retained_cell_ids):
        raise DownstreamEvidenceError("run retained cell count differs")
    retained_cell_ids_sha256 = _digest(
        run.get("retained_cell_ids_sha256"), "retained cell checksum"
    )
    if retained_cell_ids_sha256 != _cell_id_sha256(binding.retained_cell_ids):
        raise DownstreamEvidenceError("run retained cell identity differs")
    configuration_id = _text(run.get("configuration_id"), "configuration_id")
    configuration_sha256 = _digest(
        run.get("configuration_sha256"), "configuration checksum"
    )
    configuration_kind = _text(
        run.get("configuration_kind"), "configuration_kind"
    )
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
            if (
                encoding != _FINAL_OUTPUT_ENCODING
                or not output_path.endswith(".log2-cp10k-f64.zlib")
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


def build_downstream_evidence_plan(
    source_root: str | Path,
    *,
    source_kind: str,
    datasets: Sequence[DatasetEvidenceBinding],
    configurations: Sequence[AuthorizedConfiguration],
    evaluated_round_binding: EvaluatedRoundBinding | None = None,
) -> DownstreamEvidencePlan:
    """Bind a sealed development checkpoint or final execution manifest."""

    if source_kind not in _SOURCE_KINDS:
        raise ValueError("source_kind must be development or final")
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
        expected_source_root = (
            Path(evaluated_round_binding.round_root)
            / "results/final/execution"
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
    for value in dataset_values:
        _read_bound_dataset(value)
    configuration_values = tuple(configurations)
    if not configuration_values or any(
        not isinstance(value, AuthorizedConfiguration)
        for value in configuration_values
    ):
        raise TypeError(
            "configurations must contain AuthorizedConfiguration values"
        )
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
    if (
        len(configuration_lookup) != len(configuration_values)
        or len(configuration_identities) != len(configuration_values)
    ):
        raise DownstreamEvidenceError(
            "configuration authority contains duplicate identities"
        )
    for value in configuration_values:
        _method_artifact_sha256(value)
    manifest_path, manifest_file_sha, manifest_payload_sha, source_records = (
        _development_source(root)
        if source_kind == "development"
        else _final_source(root)
    )
    if evaluated_round_binding is not None and (
        manifest_path != "execution_manifest.json"
        or manifest_file_sha
        != evaluated_round_binding.final_execution_manifest_file_sha256
        or manifest_payload_sha
        != evaluated_round_binding.final_execution_manifest_payload_sha256
    ):
        raise DownstreamEvidenceError(
            "final source manifest differs from evaluated-round binding"
        )
    entries = tuple(
        _validated_plan_entry(
            record,
            source_kind=source_kind,
            source_root=root,
            datasets=dataset_lookup,
            configurations=configuration_lookup,
        )
        for record in source_records
    )
    if not entries:
        raise DownstreamEvidenceError("source has no downstream denominators")
    referenced_configuration_keys = {
        (
            entry.method_id,
            entry.configuration_id,
            entry.configuration_kind,
            entry.configuration_sha256,
        )
        for entry in entries
    }
    referenced_configurations = tuple(
        sorted(
            (
                configuration_lookup[key]
                for key in referenced_configuration_keys
            ),
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
        evaluator_source_sha256=_evaluator_source_sha256(),
        source_manifest_path=manifest_path,
        source_manifest_file_sha256=manifest_file_sha,
        source_manifest_payload_sha256=manifest_payload_sha,
        evaluated_round_binding=evaluated_round_binding,
        datasets=tuple(sorted(dataset_values, key=lambda value: value.dataset_id)),
        configurations=referenced_configurations,
        entries=entries,
        plan_sha256="0" * 64,
    )
    return DownstreamEvidencePlan(
        source_root=provisional.source_root,
        source_kind=provisional.source_kind,
        evaluator_source_sha256=provisional.evaluator_source_sha256,
        source_manifest_path=provisional.source_manifest_path,
        source_manifest_file_sha256=provisional.source_manifest_file_sha256,
        source_manifest_payload_sha256=provisional.source_manifest_payload_sha256,
        evaluated_round_binding=provisional.evaluated_round_binding,
        datasets=provisional.datasets,
        configurations=provisional.configurations,
        entries=provisional.entries,
        plan_sha256=canonical_sha256(provisional.body()),
    )


def build_development_downstream_evidence_plan(
    repository: str | Path,
    *,
    checkpoint_directory: str | Path | None = None,
) -> DownstreamEvidencePlan:
    """Build the production development plan from runner-prepared persisted data."""

    root = _existing_directory(repository, "development repository")
    active_repository = Path(__file__).absolute().parents[1]
    _reject_symlink_chain(active_repository, "active repository")
    if root != active_repository:
        raise DownstreamEvidenceError(
            "development downstream stage must use the active repository"
        )
    from .methods import load_method_registry
    from .runner import load_prepared_development_panel, load_runner_authority

    authority = load_runner_authority()
    registry = load_method_registry(root / "study/methods.json")
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
    source = (
        root / "artifacts/study/development/competition-reconstruction"
        if checkpoint_directory is None
        else Path(checkpoint_directory).absolute()
    )
    return build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=datasets,
        configurations=configurations,
    )


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
    from .final_runner import _configuration_for_method, load_prepared_final_panel
    from .methods import load_method_registry
    from .publication_freeze import validate_frozen_method

    evaluated_round_binding = _read_verified_evaluated_round_binding(
        root, round_root
    )
    frozen_method = validate_frozen_method(root)
    registry = load_method_registry(root / "study/methods.json")
    configurations = tuple(
        _configuration_for_method(spec.id, spec, frozen_method)
        for spec in registry.methods
    )
    runner_bindings, prepared = load_prepared_final_panel(root, round_root)
    datasets = bind_prepared_evaluator_panel(
        runner_bindings,
        prepared,
        dataset_root=round_root / "results",
    )
    return build_downstream_evidence_plan(
        round_root / "results/final/execution",
        source_kind="final",
        datasets=datasets,
        configurations=configurations,
        evaluated_round_binding=evaluated_round_binding,
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
        endpoints = evaluate_downstream_endpoints(output, targets)
    else:
        endpoints = terminal_downstream_endpoints(
            "upstream_run_not_completed",
            procedure="terminal_upstream_run_not_completed",
        )
    if tuple(value.endpoint for value in endpoints) != DOWNSTREAM_ENDPOINT_NAMES:
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
        raise DownstreamEvidenceError(
            "downstream endpoint contract differs"
        ) from error
    if endpoint.status == "completed":
        lower, upper = _ENDPOINT_VALUE_RANGES[endpoint.endpoint]
        if not lower <= float(endpoint.value) <= upper:
            raise DownstreamEvidenceError("downstream endpoint value is out of range")
    procedure_allowed = (
        endpoint.procedure in _TERMINAL_PROCEDURES
        or endpoint.procedure in _ENDPOINT_PROCEDURES[endpoint.endpoint]
        or (
            endpoint.endpoint
            in {"clustering_ari_loss", "clustering_nmi_loss"}
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
    if not isinstance(rows, list) or len(rows) != len(DOWNSTREAM_ENDPOINT_NAMES):
        raise DownstreamEvidenceError("downstream record endpoint count differs")
    for expected_endpoint, row in zip(DOWNSTREAM_ENDPOINT_NAMES, rows, strict=True):
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
        raise DownstreamEvidenceError(
            "downstream endpoint re-evaluation differs"
        )
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


def _revalidate_plan(plan: DownstreamEvidencePlan) -> None:
    if not isinstance(plan, DownstreamEvidencePlan):
        raise TypeError("plan must be a DownstreamEvidencePlan")
    if plan.plan_sha256 != canonical_sha256(plan.body()):
        raise DownstreamEvidenceError("downstream plan checksum differs")
    rebuilt = build_downstream_evidence_plan(
        plan.source_root,
        source_kind=plan.source_kind,
        datasets=plan.datasets,
        configurations=plan.configurations,
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
        "schema_version": 2,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": plan_file_sha,
        "source_kind": plan.source_kind,
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "source_manifest_file_sha256": plan.source_manifest_file_sha256,
        "source_manifest_payload_sha256": plan.source_manifest_payload_sha256,
        "evaluated_round_binding_sha256": (
            None
            if plan.evaluated_round_binding is None
            else plan.evaluated_round_binding.binding_sha256
        ),
        "planned_denominator_count": len(plan.entries),
        "recorded_denominator_count": len(records),
        "endpoint_row_count": len(records) * len(DOWNSTREAM_ENDPOINT_NAMES),
        "records": references,
    }
    if plan_payload.get("plan_sha256") != plan.plan_sha256:
        raise DownstreamEvidenceError("persisted downstream plan differs")
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _validate_downstream_output_location(
    plan: DownstreamEvidencePlan, output_root: Path
) -> None:
    binding = plan.evaluated_round_binding
    if binding is None:
        return
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
    """Resume the plan prefix and emit one immutable eight-row record per run."""

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
            "endpoint_row_count": len(records) * len(DOWNSTREAM_ENDPOINT_NAMES),
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
    if set(plan_payload) != _PLAN_FIELDS or plan_payload.get("schema_version") != 2:
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
        raise DownstreamEvidenceError(
            "persisted downstream configurations are invalid"
        )
    persisted_configurations = tuple(
        _configuration_from_payload(value) for value in raw_configurations
    )
    evaluated_round_binding = _evaluated_round_binding_from_payload(
        plan_payload.get("evaluated_round_binding")
    )
    try:
        rebuilt = build_downstream_evidence_plan(
            _text(plan_payload.get("source_root"), "persisted source root"),
            source_kind=_text(plan_payload.get("source_kind"), "persisted source kind"),
            datasets=persisted_datasets,
            configurations=persisted_configurations,
            evaluated_round_binding=evaluated_round_binding,
        )
    except DownstreamEvidenceError:
        raise
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError(
            "persisted downstream plan cannot be revalidated"
        ) from error
    if rebuilt.to_dict() != plan_payload:
        raise DownstreamEvidenceError("persisted downstream plan sources changed")
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
        or manifest.get("schema_version") != 2
        or manifest.get("status") != "completed"
        or manifest.get("plan_sha256") != plan_sha
        or manifest.get("plan_file_sha256") != plan_file_sha
        or manifest.get("source_kind") != rebuilt.source_kind
        or manifest.get("evaluator_source_sha256")
        != rebuilt.evaluator_source_sha256
        or manifest.get("source_manifest_file_sha256")
        != rebuilt.source_manifest_file_sha256
        or manifest.get("source_manifest_payload_sha256")
        != rebuilt.source_manifest_payload_sha256
        or manifest.get("evaluated_round_binding_sha256")
        != (
            None
            if rebuilt.evaluated_round_binding is None
            else rebuilt.evaluated_round_binding.binding_sha256
        )
        or type(planned) is not int
        or planned <= 0
        or manifest.get("recorded_denominator_count") != planned
        or endpoint_rows != planned * len(DOWNSTREAM_ENDPOINT_NAMES)
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


def validate_downstream_evidence_completeness(
    output_directory: str | Path,
    *,
    expected_denominators: Sequence[tuple[object, ...]] | None = None,
) -> DownstreamEvidenceManifest:
    """Require complete eight-row evidence and, optionally, an exact denominator set."""

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
    "DownstreamEvidenceError",
    "DownstreamEvidenceManifest",
    "DownstreamEvidencePlan",
    "EvaluatedRoundBinding",
    "bind_evaluator_dataset",
    "bind_prepared_evaluator_panel",
    "build_development_downstream_evidence_plan",
    "build_downstream_evidence_plan",
    "build_final_downstream_evidence_plan",
    "downstream_denominator_key",
    "load_downstream_evidence_manifest",
    "load_downstream_evidence_plan",
    "run_downstream_evidence",
    "validate_downstream_evidence_completeness",
]
