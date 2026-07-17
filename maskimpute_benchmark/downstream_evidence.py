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
from .schema import benchmark_dataset_sha256, validate_benchmark_dataset


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_OUTPUT_ENCODING = "zlib_raw_f64_v1"
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

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset_id")
        if not Path(self.path).is_absolute():
            raise ValueError("dataset binding path must be absolute")
        _digest(self.file_sha256, "dataset file checksum")
        _digest(self.dataset_sha256, "dataset semantic checksum")
        cells = _stable_ids(self.retained_cell_ids, "retained_cell_ids")
        genes = _stable_ids(self.gene_ids, "gene_ids")
        if (self.trajectory_root_cell_id is None) != (
            self.trajectory_source_id is None
        ):
            raise ValueError("trajectory root and source must be paired")
        if self.trajectory_root_cell_id is not None:
            if self.trajectory_root_cell_id not in cells:
                raise ValueError("trajectory root must be retained")
            _text(self.trajectory_source_id, "trajectory_source_id")
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
        }


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
    if benchmark_dataset_sha256(dataset) != binding.dataset_sha256:
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
    return DatasetEvidenceBinding(
        dataset_id=dataset_ids[0],
        path=str(dataset_path),
        file_sha256=file_sha256,
        dataset_sha256=benchmark_dataset_sha256(dataset),
        retained_cell_ids=cell_ids,
        gene_ids=tuple(dataset.var_names.astype(str)),
        trajectory_root_cell_id=trajectory_root_cell_id,
        trajectory_source_id=trajectory_source_id,
    )


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
class DownstreamEvidencePlan:
    source_root: str
    source_kind: str
    source_manifest_path: str
    source_manifest_file_sha256: str
    source_manifest_payload_sha256: str
    datasets: tuple[DatasetEvidenceBinding, ...]
    entries: tuple[DownstreamPlanEntry, ...]
    plan_sha256: str

    def body(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_root": self.source_root,
            "source_kind": self.source_kind,
            "source_manifest_path": self.source_manifest_path,
            "source_manifest_file_sha256": self.source_manifest_file_sha256,
            "source_manifest_payload_sha256": self.source_manifest_payload_sha256,
            "datasets": [value.to_dict() for value in self.datasets],
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
        "source_manifest_path",
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "datasets",
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
    }
)


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
        )
    except (TypeError, ValueError) as error:
        raise DownstreamEvidenceError("persisted dataset binding is invalid") from error


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    ordinal: int
    path: str
    sha256: str
    payload: Mapping[str, object]


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
        if not isinstance(record, dict) or set(record) != {
            "run",
            "metrics",
            "p_pre_zero_evidence",
        }:
            raise DownstreamEvidenceError("development source record is malformed")
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
        or storage.get("evaluator_output_encoding") != _FINAL_OUTPUT_ENCODING
    ):
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
        if set(record) != {
            "run",
            "metrics",
            "p_pre_zero_evidence",
            "execution_request",
        }:
            raise DownstreamEvidenceError("final source record schema differs")
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
) -> DownstreamPlanEntry:
    run = source.payload.get("run")
    if not isinstance(run, Mapping):
        raise DownstreamEvidenceError("source record run is malformed")
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
    if run.get("retained_gene_count") != len(binding.gene_ids):
        raise DownstreamEvidenceError("run retained gene count differs")

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
            if encoding != _FINAL_OUTPUT_ENCODING:
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
        method_id=_text(run.get("method_id"), "method_id"),
        dataset_id=dataset_id,
        source_dataset_sha256=source_dataset_sha256,
        mechanism=_text(run.get("mechanism"), "run mechanism"),
        biological_id=_text(run.get("biological_id"), "run biological_id"),
        technical_view=_text(run.get("technical_view"), "run technical_view"),
        model_seed=_optional_model_seed(run.get("model_seed")),
        configuration_id=_text(run.get("configuration_id"), "configuration_id"),
        configuration_sha256=_digest(
            run.get("configuration_sha256"), "configuration checksum"
        ),
        configuration_kind=_text(run.get("configuration_kind"), "configuration_kind"),
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
) -> DownstreamEvidencePlan:
    """Bind a sealed development checkpoint or final execution manifest."""

    if source_kind not in _SOURCE_KINDS:
        raise ValueError("source_kind must be development or final")
    root = Path(source_root).absolute()
    _reject_symlink_chain(root, "source root")
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise DownstreamEvidenceError("source root is unavailable") from error
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        raise DownstreamEvidenceError("source root must be a non-symlink directory")
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
    manifest_path, manifest_file_sha, manifest_payload_sha, source_records = (
        _development_source(root)
        if source_kind == "development"
        else _final_source(root)
    )
    entries = tuple(
        _validated_plan_entry(
            record,
            source_kind=source_kind,
            source_root=root,
            datasets=dataset_lookup,
        )
        for record in source_records
    )
    if not entries:
        raise DownstreamEvidenceError("source has no downstream denominators")
    provisional = DownstreamEvidencePlan(
        source_root=str(root),
        source_kind=source_kind,
        source_manifest_path=manifest_path,
        source_manifest_file_sha256=manifest_file_sha,
        source_manifest_payload_sha256=manifest_payload_sha,
        datasets=tuple(sorted(dataset_values, key=lambda value: value.dataset_id)),
        entries=entries,
        plan_sha256="0" * 64,
    )
    return DownstreamEvidencePlan(
        source_root=provisional.source_root,
        source_kind=provisional.source_kind,
        source_manifest_path=provisional.source_manifest_path,
        source_manifest_file_sha256=provisional.source_manifest_file_sha256,
        source_manifest_payload_sha256=provisional.source_manifest_payload_sha256,
        datasets=provisional.datasets,
        entries=provisional.entries,
        plan_sha256=canonical_sha256(provisional.body()),
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
        "run_status": entry.status,
        "run_reason": entry.reason,
    }


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
    return tuple(
        _load_record(
            output_root / "records" / name,
            plan,
            plan.entries[index],
            bindings[plan.entries[index].dataset_id],
        )
        for index, name in enumerate(names)
    )


def _revalidate_plan(plan: DownstreamEvidencePlan) -> None:
    if not isinstance(plan, DownstreamEvidencePlan):
        raise TypeError("plan must be a DownstreamEvidencePlan")
    if plan.plan_sha256 != canonical_sha256(plan.body()):
        raise DownstreamEvidenceError("downstream plan checksum differs")
    rebuilt = build_downstream_evidence_plan(
        plan.source_root,
        source_kind=plan.source_kind,
        datasets=plan.datasets,
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
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": plan_file_sha,
        "source_kind": plan.source_kind,
        "source_manifest_file_sha256": plan.source_manifest_file_sha256,
        "source_manifest_payload_sha256": plan.source_manifest_payload_sha256,
        "planned_denominator_count": len(plan.entries),
        "recorded_denominator_count": len(records),
        "endpoint_row_count": len(records) * len(DOWNSTREAM_ENDPOINT_NAMES),
        "records": references,
    }
    if plan_payload.get("plan_sha256") != plan.plan_sha256:
        raise DownstreamEvidenceError("persisted downstream plan differs")
    return {**body, "manifest_sha256": canonical_sha256(body)}


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
    _ensure_directory(output_root, "downstream output directory")
    plan_raw = _canonical_bytes(plan.to_dict()) + b"\n"
    _publish_immutable(output_root / "plan.json", plan_raw, "downstream plan")
    records = list(_load_record_prefix(output_root, plan))
    if os.path.lexists(output_root / "downstream_manifest.json") and len(
        records
    ) != len(plan.entries):
        raise DownstreamEvidenceError(
            "downstream manifest exists before its denominator is complete"
        )
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
    manifest = _manifest_payload(output_root, plan, records)
    _publish_immutable(
        output_root / "downstream_manifest.json",
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


def load_downstream_evidence_manifest(
    output_directory: str | Path,
) -> DownstreamEvidenceManifest:
    """Validate a complete downstream manifest and every referenced row file."""

    output_root = Path(output_directory).absolute()
    plan_payload, _plan_raw, plan_file_sha = _strict_json(
        output_root / "plan.json", "downstream plan"
    )
    if set(plan_payload) != _PLAN_FIELDS or plan_payload.get("schema_version") != 1:
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
    rebuilt = build_downstream_evidence_plan(
        _text(plan_payload.get("source_root"), "persisted source root"),
        source_kind=_text(plan_payload.get("source_kind"), "persisted source kind"),
        datasets=persisted_datasets,
    )
    if rebuilt.to_dict() != plan_payload:
        raise DownstreamEvidenceError("persisted downstream plan sources changed")
    manifest, _manifest_raw, _manifest_file_sha = _strict_json(
        output_root / "downstream_manifest.json", "downstream manifest"
    )
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
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("plan_sha256") != plan_sha
        or manifest.get("plan_file_sha256") != plan_file_sha
        or manifest.get("source_kind") != rebuilt.source_kind
        or manifest.get("source_manifest_file_sha256")
        != rebuilt.source_manifest_file_sha256
        or manifest.get("source_manifest_payload_sha256")
        != rebuilt.source_manifest_payload_sha256
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
        path = _safe_relative(output_root, reference.get("path"), "downstream record")
        record = _load_record(
            path,
            rebuilt,
            entry,
            bindings[entry.dataset_id],
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
        record.get("method"),
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
    "bind_evaluator_dataset",
    "build_downstream_evidence_plan",
    "downstream_denominator_key",
    "load_downstream_evidence_manifest",
    "run_downstream_evidence",
    "validate_downstream_evidence_completeness",
]
