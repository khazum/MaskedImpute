"""Frozen, bounded resource-scaling panel for the publication study."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Literal
import zlib

import numpy as np

from .comparator_tuning import (
    BoundComparatorConfiguration,
    ComparatorTuningError,
    _validate_bound_selection_configuration,
    comparator_method_binding,
)
from .direct_values import direct_equal, direct_json_value
from .final_runner import FrozenPlanMethodAuthority
from .methods.registry import MethodRegistry, load_method_registry
from .protocol import DevelopmentProtocol, Protocol, canonical_sha256, load_protocol
from .runner import (
    AdapterOutcome,
    DatasetBinding,
    ExecutionEnvironmentRegistry,
    ExecutionRequest,
    FinalComparatorExecutionRequest,
    LongFormMetric,
    PreparedDataset,
    RawRunResult,
    RepositoryAdapterDispatcher,
    RunPlanEntry,
    RunnerAuthority,
    SpawnedRepositoryExecutor,
    direct_bound_comparator_value,
    derive_lock_only_environment_ids,
    enforce_calibration_fold_receipt,
    implementation_source_sha256,
    load_runner_authority,
    method_input_sha256,
    prepare_dataset_for_execution,
)
from .simulators import SimulationArtifact, SimulationRequest, run_symsim_pair


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SCALING_CHECKPOINT_FILE = re.compile(r"[0-9]{8}\.json\Z")
_CELL_COUNTS = (10_000, 25_000, 50_000, 100_000)
_METHOD_IDS = ("observed", "maskimpute", "dca", "scvi", "magic")
_ARTIFACT_POLICY = {
    "evaluator_output_retention": "bounded_zlib_raw_f64_v1_with_metric_replay",
    "native_method_output_retention": (
        "bounded_zlib_raw_f64_v1_with_conversion_replay"
    ),
    "generated_dataset_retention": "moderate_h5ad",
    "paired_control_retention": "discard_after_semantic_and_file_hashing",
    "native_simulator_output_retention": "discard_after_manifest_hashing",
}
_CONTRACT_KEYS = {
    "schema_version",
    "role",
    "mechanism",
    "technical_view",
    "cell_counts",
    "accuracy_cell_counts",
    "accuracy_metrics",
    "excluded_metric_families",
    "genes",
    "method_ids",
    "model_seed",
    "seed_derivation",
    "artifact_policy",
}
_MAX_LOG_BYTES = 64 * 1024 * 1024
_MAX_EXECUTOR_RECEIPT_BYTES = 2 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
_EVALUATOR_OUTPUT_ENCODING = "zlib_raw_f64_v1"
_EVALUATOR_OUTPUT_COMPRESSION_LEVEL = 6
_EVALUATOR_OUTPUT_RETENTION = "compressed_zlib_raw_f64_v1"
_EVALUATOR_OUTPUT_SCALE = "log2_cp10k_plus_1"
_MAX_EVALUATOR_OUTPUT_BYTES = max(_CELL_COUNTS) * 500 * 8
_NATIVE_OUTPUT_ENCODING = "zlib_raw_f64_v1"
_NATIVE_OUTPUT_RETENTION = "compressed_zlib_raw_f64_v1"
_MAX_NATIVE_OUTPUT_BYTES = _MAX_EVALUATOR_OUTPUT_BYTES
_EXECUTOR_RECEIPT_COMMON_KEYS = {
    "schema_version",
    "run_id",
    "method_id",
    "dataset_id",
    "source_dataset_sha256",
    "model_seed",
    "method_input_sha256",
    "retained_cell_ids_sha256",
    "status",
    "reason",
    "runtime_seconds",
    "peak_rss_bytes",
    "peak_gpu_bytes",
    "rss_measurement",
    "gpu_measurement",
    "stdout_sha256",
    "stdout_size_bytes",
    "stderr_sha256",
    "stderr_size_bytes",
    "native_snapshot",
    "receipt_sha256",
}
_EXECUTOR_RECEIPT_LEGACY_KEYS = _EXECUTOR_RECEIPT_COMMON_KEYS | {
    "configuration_sha256",
}
_EXECUTOR_RECEIPT_DIRECT_KEYS = _EXECUTOR_RECEIPT_COMMON_KEYS | {
    "comparator_configuration",
    "comparator_nonexecution_identity",
}
_NATIVE_SNAPSHOT_KEYS = {
    "method_id",
    "source_dataset_sha256",
    "output_scale",
    "shape",
    "matrix_sha256",
}
_UNLOADED = object()
_CHECKPOINT_KEYS = {
    "schema_version",
    "plan_sha256",
    "input_hashes",
    "planned_run_count",
    "status",
    "datasets",
    "records",
    "checkpoint_sha256",
}
_SCALING_ACCURACY_METRICS = (
    "mse",
    "mse_dropout",
    "mse_pre_dropout_zero",
    "mse_nonzero",
    "gnrmse",
    "mean_distortion",
    "variance_distortion",
    "mean_gene_wasserstein_distance",
    "corr_err",
    "n_corr_genes",
)
_DATASET_RECEIPT_KEYS = {
    "schema_version",
    "cells",
    "genes",
    "namespace",
    "mechanism",
    "technical_view",
    "dataset_id",
    "independent_unit_id",
    "dataset_sha256",
    "truth_sha256",
    "moderate_output_path",
    "moderate_output_file_sha256",
    "moderate_output_size_bytes",
    "severe_dataset_sha256",
    "severe_output_file_sha256",
    "severe_output_size_bytes",
    "moderate_native_manifest_sha256",
    "severe_native_manifest_sha256",
    "native_files_sha256",
    "protocol_sha256",
    "design_sha256",
    "seed_source_sha256",
    "seeds",
    "severe_retention",
    "native_retention",
    "receipt_sha256",
}
_GENERATOR_RECEIPT_FIELDS = (
    "dataset_sha256",
    "truth_sha256",
    "severe_dataset_sha256",
    "moderate_native_manifest_sha256",
    "severe_native_manifest_sha256",
    "native_files_sha256",
)


class ScalingContractError(ValueError):
    """Raised when the scaling authority, plan, or evidence is not closed."""


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScalingContractError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScalingContractError(f"{name} must be a finite real number") from error
    if not np.isfinite(result):
        raise ScalingContractError(f"{name} must be a finite real number")
    return result


@dataclass(frozen=True, slots=True)
class ScalingEvaluatedAttempt:
    """Bounded scaling result without main-study score-matrix evidence."""

    run: RawRunResult
    metrics: tuple[LongFormMetric, ...]
    stdout: bytes
    stderr: bytes
    native_output: np.ndarray | None
    native_output_scale: str | None
    evaluator_output: np.ndarray | None
    executor_receipt: bytes | None = None


@dataclass(frozen=True, slots=True)
class ScalingContract:
    """Exact tracked design of the post-freeze resource-scaling panel."""

    schema_version: int
    role: str
    mechanism: str
    technical_view: str
    cell_counts: tuple[int, ...]
    accuracy_cell_counts: tuple[int, ...]
    accuracy_metrics: tuple[str, ...]
    excluded_metric_families: Mapping[str, str]
    genes: int
    method_ids: tuple[str, ...]
    model_seed: int
    seed_algorithm: str
    seed_master: int
    artifact_policy: Mapping[str, str]
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ScalingSeeds:
    """One independent truth seed and two paired measurement seeds."""

    biological: int
    moderate: int
    severe: int


@dataclass(frozen=True, slots=True)
class ScalingPlanEntry:
    """One required method-by-size scaling attempt."""

    ordinal: int
    run_id: str
    cells: int
    genes: int
    method_id: str
    model_seed: int | None
    configuration_id: str
    configuration_sha256: str | None
    configuration_kind: str
    requires_count_score: bool
    requires_calibration: bool
    comparator_configuration: BoundComparatorConfiguration | None
    comparator_nonexecution_identity: Mapping[str, object] | None
    accuracy_enabled: bool
    native_output_scale: str
    timeout_seconds: int
    max_rss_bytes: int
    max_gpu_bytes: int
    rss_measurement: str
    gpu_measurement: str

    def __post_init__(self) -> None:
        if self.configuration_kind == "comparator_tuning":
            if (
                self.configuration_sha256 is not None
                or not isinstance(
                    self.comparator_configuration, BoundComparatorConfiguration
                )
                or self.comparator_nonexecution_identity is not None
                or self.configuration_id == "registry-default"
                or self.comparator_configuration.configuration.configuration_id
                != self.configuration_id
                or self.comparator_configuration.configuration.method_id
                != self.method_id
                or self.comparator_configuration.method.method_id != self.method_id
                or self.requires_count_score
                or self.requires_calibration
            ):
                raise ScalingContractError(
                    "scaling selected comparator identity is invalid"
                )
            try:
                _validate_bound_selection_configuration(self.comparator_configuration)
            except (ComparatorTuningError, TypeError, ValueError) as error:
                raise ScalingContractError(
                    "scaling selected comparator identity is invalid"
                ) from error
            return
        if (
            not isinstance(self.configuration_sha256, str)
            or _SHA256.fullmatch(self.configuration_sha256) is None
            or self.comparator_configuration is not None
            or self.comparator_nonexecution_identity is not None
        ):
            raise ScalingContractError("scaling legacy configuration is invalid")

    def to_dict(self) -> dict[str, object]:
        value = direct_json_value(self)
        if not isinstance(value, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError("scaling plan entry must encode as an object")
        if self.configuration_kind == "comparator_tuning":
            value.pop("configuration_sha256")
            value["comparator_configuration"] = direct_bound_comparator_value(
                self.comparator_configuration
            )
            value["comparator_nonexecution_identity"] = None
        else:
            value.pop("comparator_configuration")
            value.pop("comparator_nonexecution_identity")
        return value


@dataclass(frozen=True, slots=True)
class ScalingPlan:
    """Hash-bound complete denominator for the resource-scaling panel."""

    schema_version: int
    input_hashes: Mapping[str, str]
    entries: tuple[ScalingPlanEntry, ...]
    configurations: tuple[FrozenPlanMethodAuthority, ...]
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class ScalingCheckpoint:
    """Validated resumable prefix of the fixed scaling denominator."""

    schema_version: int
    plan_sha256: str
    input_hashes: Mapping[str, str]
    planned_run_count: int
    status: Literal["running", "completed"]
    datasets: tuple[Mapping[str, object], ...]
    records: tuple[Mapping[str, object], ...]
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ScalingExecutionAuthority:
    """All validated production authority needed to execute the scaling plan."""

    repository: Path
    contract: ScalingContract
    protocol: Protocol
    frozen_method: Mapping[str, object]
    registry: MethodRegistry
    runner_authority: RunnerAuthority
    environments: ExecutionEnvironmentRegistry
    plan: ScalingPlan


def scaling_plan_payload(plan: ScalingPlan) -> dict[str, object]:
    """Return the complete canonical plan payload bound by ``plan_sha256``."""

    if not isinstance(plan, ScalingPlan):
        raise TypeError("plan must be a ScalingPlan")
    body: dict[str, object] = {
        "schema_version": plan.schema_version,
        "input_hashes": dict(plan.input_hashes),
        "entries": [entry.to_dict() for entry in plan.entries],
        "configurations": [value.to_dict() for value in plan.configurations],
    }
    if plan.schema_version != 1 or canonical_sha256(body) != plan.plan_sha256:
        raise ScalingContractError("scaling plan payload binding is invalid")
    return {**body, "plan_sha256": plan.plan_sha256}


def scaling_checkpoint_payload(checkpoint: ScalingCheckpoint) -> dict[str, object]:
    """Return the complete canonical checkpoint payload, including its seal."""

    if not isinstance(checkpoint, ScalingCheckpoint):
        raise TypeError("checkpoint must be a ScalingCheckpoint")
    body: dict[str, object] = {
        "schema_version": checkpoint.schema_version,
        "plan_sha256": checkpoint.plan_sha256,
        "input_hashes": dict(checkpoint.input_hashes),
        "planned_run_count": checkpoint.planned_run_count,
        "status": checkpoint.status,
        "datasets": [dict(value) for value in checkpoint.datasets],
        "records": [dict(value) for value in checkpoint.records],
    }
    if (
        checkpoint.schema_version != 1
        or canonical_sha256(body) != checkpoint.checkpoint_sha256
    ):
        raise ScalingContractError("scaling checkpoint payload binding is invalid")
    return {**body, "checkpoint_sha256": checkpoint.checkpoint_sha256}


def scaling_storage_preflight(authority: object) -> dict[str, object]:
    """Derive the complete scaling storage bound without executing the panel."""

    plan = getattr(authority, "plan", None)
    if not isinstance(plan, ScalingPlan):
        raise TypeError("authority must expose a ScalingPlan as plan")
    cells = tuple(dict.fromkeys(entry.cells for entry in plan.entries))
    genes = {entry.genes for entry in plan.entries}
    if (
        not cells
        or len(genes) != 1
        or any(entry.cells not in cells for entry in plan.entries)
    ):
        raise ScalingContractError("scaling storage plan dimensions are invalid")
    gene_count = next(iter(genes))
    retained = 0
    retained_h5ad = 0
    retained_runs = 0
    retained_checkpoints = 0
    peak_materialization = 0
    for cell_count in cells:
        materialization = cell_count * gene_count * 96 + 2 * 1024**3
        peak_materialization = max(
            peak_materialization,
            retained + materialization,
        )
        h5ad = _scaling_h5ad_size_ceiling(cell_count, gene_count)
        retained += h5ad + _MAX_CHECKPOINT_BYTES
        retained_h5ad += h5ad
        retained_checkpoints += _MAX_CHECKPOINT_BYTES
        for entry in (value for value in plan.entries if value.cells == cell_count):
            uncompressed_matrix_bytes = entry.cells * entry.genes * 8
            run_bound = (
                2 * _zlib_compress_bound(uncompressed_matrix_bytes)
                + 2 * _MAX_LOG_BYTES
                + _MAX_EXECUTOR_RECEIPT_BYTES
            )
            retained += run_bound + _MAX_CHECKPOINT_BYTES
            retained_runs += run_bound
            retained_checkpoints += _MAX_CHECKPOINT_BYTES
    required = max(peak_materialization, retained)
    body: dict[str, object] = {
        "schema": "maskimpute-scaling-storage-preflight-v1",
        "plan_sha256": plan.plan_sha256,
        "planned_run_count": len(plan.entries),
        "cell_counts": list(cells),
        "genes": gene_count,
        "retained_h5ad_bound_bytes": retained_h5ad,
        "retained_run_artifact_bound_bytes": retained_runs,
        "retained_checkpoint_history_bound_bytes": retained_checkpoints,
        "peak_materialization_bound_bytes": peak_materialization,
        "retained_completion_bound_bytes": retained,
        "required_free_bytes": required,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _reject_constant(value: str) -> None:
    raise ScalingContractError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScalingContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScalingContractError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaling_h5ad_size_ceiling(cells: int, genes: int) -> int:
    """Return the closed storage bound for one retained scaling H5AD."""

    if type(cells) is not int or cells <= 0 or type(genes) is not int or genes <= 0:
        raise ScalingContractError("scaling H5AD dimensions are invalid")
    matrix_bytes = cells * genes * 32
    observation_bytes = cells * 4_096
    annotation_bytes = genes * 65_536
    return 64 * 1024 * 1024 + matrix_bytes + observation_bytes + annotation_bytes


def _regular_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_scaling_h5ad_structure(
    path: Path,
    cells: int,
    genes: int,
    *,
    profile: Literal["scaling", "trajectory"] = "scaling",
) -> None:
    """Inspect closed HDF5 metadata before AnnData materializes an H5AD."""

    if profile not in {"scaling", "trajectory"}:
        raise ScalingContractError("H5AD HDF5 validation profile is invalid")

    try:
        import h5py

        def text(value: object, name: str) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if isinstance(value, (str, np.str_)):
                return str(value)
            raise ScalingContractError(f"H5AD HDF5 {name} is not text")

        def attrs(
            element: object,
            expected: Mapping[str, object],
            name: str,
        ) -> None:
            attributes = element.attrs  # type: ignore[attr-defined]
            if len(attributes) != len(expected) or set(attributes) != set(expected):
                raise ScalingContractError(
                    f"H5AD HDF5 {name} metadata structure is not closed"
                )
            for key, expected_value in expected.items():
                actual = attributes[key]
                if isinstance(expected_value, str):
                    actual = text(actual, f"{name} {key}")
                if actual != expected_value:
                    raise ScalingContractError(
                        f"H5AD HDF5 {name} metadata structure is invalid"
                    )

        def names(group: object, expected_maximum: int, name: str) -> tuple[str, ...]:
            if not isinstance(group, h5py.Group):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            if len(group) > expected_maximum:
                raise ScalingContractError(
                    f"H5AD HDF5 {name} structure exceeds its bound"
                )
            values = tuple(group.keys())
            if any(not value or len(value) > 128 for value in values):
                raise ScalingContractError(
                    f"H5AD HDF5 {name} contains an invalid field name"
                )
            return values

        def child(group: object, name: str, label: str) -> object:
            if not isinstance(group, h5py.Group):
                raise ScalingContractError(f"H5AD HDF5 {label} structure is invalid")
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ScalingContractError(
                    f"H5AD HDF5 {label} must use an internal hard link"
                )
            return group[name]

        def dataset_layout(element: object, name: str) -> object:
            if not isinstance(element, h5py.Dataset):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            if element.is_virtual:
                raise ScalingContractError(
                    f"H5AD HDF5 {name} virtual dataset layout is forbidden"
                )
            if element.external is not None:
                raise ScalingContractError(
                    f"H5AD HDF5 {name} external dataset layout is forbidden"
                )
            return element

        def data_dtype(dataset: object, name: str, kinds: str) -> None:
            dataset = dataset_layout(dataset, name)
            dtype = dataset.dtype
            if dtype.kind not in kinds or (
                dtype.kind in "biufc" and dtype.itemsize > 8
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} dtype is invalid")

        def array(
            dataset: object,
            shape: tuple[int, ...],
            name: str,
            kinds: str,
        ) -> None:
            data_dtype(dataset, name, kinds)
            if dataset.shape != shape:  # type: ignore[attr-defined]
                raise ScalingContractError(
                    f"H5AD HDF5 {name} shape structure is invalid"
                )
            attrs(
                dataset,
                {"encoding-type": "array", "encoding-version": "0.2.0"},
                name,
            )

        def string_array(dataset: object, shape: tuple[int, ...], name: str) -> None:
            dataset = dataset_layout(dataset, name)
            if dataset.shape != shape or h5py.check_string_dtype(dataset.dtype) is None:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            attrs(
                dataset,
                {
                    "encoding-type": "string-array",
                    "encoding-version": "0.2.0",
                },
                name,
            )

        def matrix(element: object, shape: tuple[int, int], name: str) -> None:
            if isinstance(element, h5py.Dataset):
                array(element, shape, name, "iu")
                return
            if not isinstance(element, h5py.Group):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            matrix_names = names(element, 3, name)
            if set(matrix_names) != {"data", "indices", "indptr"}:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            attribute_names = set(element.attrs)
            if attribute_names != {"encoding-type", "encoding-version", "shape"}:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            encoding_type = text(element.attrs["encoding-type"], f"{name} encoding")
            if (
                encoding_type not in {"csr_matrix", "csc_matrix"}
                or text(element.attrs["encoding-version"], f"{name} version") != "0.1.0"
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            encoded_shape = np.asarray(element.attrs["shape"])
            if (
                encoded_shape.shape != (2,)
                or encoded_shape.dtype.kind not in "iu"
                or tuple(int(value) for value in encoded_shape) != shape
            ):
                raise ScalingContractError(
                    f"H5AD HDF5 {name} shape structure is invalid"
                )
            data = child(element, "data", f"{name}/data")
            indices = child(element, "indices", f"{name}/indices")
            indptr = child(element, "indptr", f"{name}/indptr")
            data_dtype(data, f"{name}/data", "iu")
            data_dtype(indices, f"{name}/indices", "iu")
            data_dtype(indptr, f"{name}/indptr", "iu")
            nnz = data.shape[0]  # type: ignore[attr-defined]
            pointer_count = (
                shape[0] + 1 if encoding_type == "csr_matrix" else shape[1] + 1
            )
            if (
                data.ndim != 1  # type: ignore[attr-defined]
                or nnz > shape[0] * shape[1]
                or indices.shape != (nnz,)  # type: ignore[attr-defined]
                or indptr.shape != (pointer_count,)  # type: ignore[attr-defined]
                or len(data.attrs) != 0  # type: ignore[attr-defined]
                or len(indices.attrs) != 0  # type: ignore[attr-defined]
                or len(indptr.attrs) != 0  # type: ignore[attr-defined]
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")

        def categorical(
            group: object, length: int, maximum_categories: int, name: str
        ) -> None:
            if set(names(group, 2, name)) != {"categories", "codes"}:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            attributes = group.attrs  # type: ignore[attr-defined]
            if set(attributes) != {"encoding-type", "encoding-version", "ordered"}:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            if (
                text(attributes["encoding-type"], f"{name} encoding") != "categorical"
                or text(attributes["encoding-version"], f"{name} version") != "0.2.0"
                or bool(attributes["ordered"])
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            categories = child(group, "categories", f"{name}/categories")
            if (
                not isinstance(categories, h5py.Dataset)
                or categories.ndim != 1
                or categories.shape[0] > maximum_categories
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            string_array(categories, categories.shape, f"{name}/categories")
            array(
                child(group, "codes", f"{name}/codes"), (length,), f"{name}/codes", "iu"
            )

        def dataframe(
            group: object,
            length: int,
            columns: tuple[str, ...],
            name: str,
        ) -> None:
            expected_names = {"_index", *columns}
            if set(names(group, len(expected_names), name)) != expected_names:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            attributes = group.attrs  # type: ignore[attr-defined]
            if set(attributes) != {
                "_index",
                "column-order",
                "encoding-type",
                "encoding-version",
            }:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            order_value = np.asarray(attributes["column-order"])
            if order_value.ndim != 1 or order_value.size != len(columns):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            order = tuple(text(value, f"{name} column") for value in order_value)
            if (
                text(attributes["_index"], f"{name} index") != "_index"
                or text(attributes["encoding-type"], f"{name} encoding") != "dataframe"
                or text(attributes["encoding-version"], f"{name} version") != "0.2.0"
                or order != columns
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            string_array(
                child(group, "_index", f"{name}/_index"), (length,), f"{name}/_index"
            )

        metadata_nodes = 0

        def bounded_metadata(element: object, depth: int, name: str) -> None:
            nonlocal metadata_nodes
            metadata_nodes += 1
            if metadata_nodes > 1_024 or depth > 8:
                raise ScalingContractError(
                    "H5AD HDF5 supplementary metadata structure exceeds its bound"
                )
            if isinstance(element, h5py.Group):
                attrs(
                    element,
                    {"encoding-type": "dict", "encoding-version": "0.1.0"},
                    name,
                )
                for field in names(element, 64, name):
                    bounded_metadata(
                        child(element, field, f"{name}/{field}"),
                        depth + 1,
                        f"{name}/{field}",
                    )
                return
            element = dataset_layout(element, name)
            if element.ndim > 2:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            element_count = (
                int(np.prod(element.shape, dtype=np.int64)) if element.shape else 1
            )
            if element_count > 4_096:
                raise ScalingContractError(
                    f"H5AD HDF5 {name} structure exceeds its bound"
                )
            dtype = element.dtype
            if dtype.kind == "O":
                if h5py.check_string_dtype(dtype) is None:
                    raise ScalingContractError(f"H5AD HDF5 {name} dtype is invalid")
            elif dtype.kind not in "SUbiuf" or (
                dtype.kind in "biuf" and dtype.itemsize > 8
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} dtype is invalid")
            attribute_names = set(element.attrs)
            if attribute_names != {"encoding-type", "encoding-version"}:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            encoding = text(element.attrs["encoding-type"], f"{name} encoding")
            if (
                encoding not in {"array", "numeric-scalar", "string", "string-array"}
                or text(element.attrs["encoding-version"], f"{name} version") != "0.2.0"
            ):
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")

        def dictionary(group: object, expected_names: set[str], name: str) -> None:
            if set(names(group, len(expected_names), name)) != expected_names:
                raise ScalingContractError(f"H5AD HDF5 {name} structure is invalid")
            attrs(
                group,
                {"encoding-type": "dict", "encoding-version": "0.1.0"},
                name,
            )

        with h5py.File(path, "r") as handle:
            if len(handle) != 9 or set(handle.keys()) != {
                "X",
                "layers",
                "obs",
                "obsm",
                "obsp",
                "uns",
                "var",
                "varm",
                "varp",
            }:
                raise ScalingContractError("H5AD HDF5 root structure is not closed")
            attrs(
                handle,
                {"encoding-type": "anndata", "encoding-version": "0.1.0"},
                "root",
            )
            matrix(child(handle, "X", "X"), (cells, genes), "X")
            layers = child(handle, "layers", "layers")
            if profile == "scaling":
                dictionary(layers, {"pre_capture_counts"}, "layers")
                matrix(
                    child(layers, "pre_capture_counts", "layers/pre_capture_counts"),
                    (cells, genes),
                    "layers/pre_capture_counts",
                )
            else:
                dictionary(layers, set(), "layers")

            obs_columns = (
                "dataset_id",
                "mechanism",
                "condition",
                "biological_id",
                "technical_view",
                "draw",
                "library_size",
                "group",
                *(("pseudotime",) if profile == "trajectory" else ()),
            )
            obs = child(handle, "obs", "obs")
            dataframe(obs, cells, obs_columns, "obs")
            for field, maximum in {
                "dataset_id": 1,
                "mechanism": 1,
                "condition": 1,
                "biological_id": 1,
                "technical_view": 1,
                "group": 5 if profile == "scaling" else 3,
            }.items():
                categorical(
                    child(obs, field, f"obs/{field}"), cells, maximum, f"obs/{field}"
                )
            for field in ("draw", "library_size"):
                array(child(obs, field, f"obs/{field}"), (cells,), f"obs/{field}", "iu")
            if profile == "trajectory":
                array(
                    child(obs, "pseudotime", "obs/pseudotime"),
                    (cells,),
                    "obs/pseudotime",
                    "f",
                )

            marker_columns = tuple(
                field
                for group_number in range(1, 6)
                for field in (
                    f"theoretical_log2fc_group_{group_number}",
                    f"marker_group_{group_number}",
                )
            )
            var = child(handle, "var", "var")
            var_attributes = var.attrs  # type: ignore[attr-defined]
            raw_order = np.asarray(var_attributes.get("column-order", []))
            var_columns = tuple(text(value, "var column") for value in raw_order)
            allowed_var_columns = (
                {()} if profile == "trajectory" else {(), marker_columns}
            )
            if var_columns not in allowed_var_columns:
                raise ScalingContractError("H5AD HDF5 var structure is invalid")
            dataframe(var, genes, var_columns, "var")
            for field in var_columns:
                kinds = "fiu" if field.startswith("theoretical_") else "biu"
                array(
                    child(var, field, f"var/{field}"), (genes,), f"var/{field}", kinds
                )

            for field in ("obsm", "obsp", "varm", "varp"):
                dictionary(child(handle, field, field), set(), field)

            uns = child(handle, "uns", "uns")
            uns_names = set(names(uns, 5, "uns"))
            required_uns = (
                {"truth_kind", "primary_truth_layer", "provenance"}
                if profile == "scaling"
                else {"truth_kind", "provenance"}
            )
            if not required_uns <= uns_names or not uns_names <= required_uns | {
                "normalization",
                "allowed_covariates",
            }:
                raise ScalingContractError("H5AD HDF5 uns structure is invalid")
            attrs(
                uns,
                {"encoding-type": "dict", "encoding-version": "0.1.0"},
                "uns",
            )
            for field in uns_names:
                bounded_metadata(child(uns, field, f"uns/{field}"), 0, f"uns/{field}")
            provenance = child(uns, "provenance", "uns/provenance")
            if set(names(provenance, 6, "uns/provenance")) != {
                "source",
                "source_sha256",
                "software",
                "software_version",
                "parameters",
                "seeds",
            }:
                raise ScalingContractError("H5AD HDF5 provenance structure is invalid")
    except ScalingContractError:
        raise
    except Exception as error:
        raise ScalingContractError("H5AD HDF5 structure validation failed") from error


def _zlib_compress_bound(uncompressed_nbytes: int) -> int:
    """Return zlib's documented single-call upper bound."""

    return (
        uncompressed_nbytes
        + (uncompressed_nbytes >> 12)
        + (uncompressed_nbytes >> 14)
        + (uncompressed_nbytes >> 25)
        + 13
    )


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
        raise ScalingContractError("scaling evidence is not canonical JSON") from error


def _safe_relative_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScalingContractError(f"{name} is not a safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ScalingContractError(f"{name} is not a safe relative path")
    return path


def _reject_symlink_components(root: Path, path: Path, name: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ScalingContractError(f"{name} escaped the output root") from error
    current = root
    for part in relative.parts:
        current /= part
        if not os.path.lexists(current):
            continue
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ScalingContractError(f"{name} cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ScalingContractError(f"{name} contains a symlink component")


def _ensure_directory(root: Path, relative: Path, name: str) -> Path:
    current = root
    for part in relative.parts:
        current /= part
        if os.path.lexists(current):
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ScalingContractError(f"{name} is not a canonical directory")
            continue
        try:
            current.mkdir(mode=0o755)
        except OSError as error:
            raise ScalingContractError(f"{name} cannot be created") from error
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ScalingContractError(f"{name} is not a canonical directory")
    return current


def _atomic_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise ScalingContractError("immutable scaling checkpoint already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if os.path.lexists(path):
            raise ScalingContractError("immutable scaling checkpoint already exists")
        os.rename(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def load_scaling_contract(path: Path) -> ScalingContract:
    """Load the sole tracked scaling design without accepting design overrides."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except ScalingContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScalingContractError(
            f"scaling contract cannot be read: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _CONTRACT_KEYS:
        raise ScalingContractError("scaling contract fields are not closed")
    if payload.get("schema_version") != 1 or type(payload["schema_version"]) is not int:
        raise ScalingContractError("scaling contract schema_version must be 1")
    if payload.get("role") != "post_freeze_resource_scaling":
        raise ScalingContractError("scaling role is invalid")
    if (
        payload.get("mechanism") != "symsim"
        or payload.get("technical_view") != "moderate"
    ):
        raise ScalingContractError("scaling mechanism/view is not prespecified")
    raw_cells = payload.get("cell_counts")
    raw_accuracy = payload.get("accuracy_cell_counts")
    if raw_cells != list(_CELL_COUNTS):
        raise ScalingContractError("scaling cell-count grid is invalid")
    if raw_accuracy != list(_CELL_COUNTS):
        raise ScalingContractError("scaling accuracy cell-count grid is invalid")
    if payload.get("accuracy_metrics") != list(_SCALING_ACCURACY_METRICS):
        raise ScalingContractError("scaling accuracy metric set is invalid")
    excluded = payload.get("excluded_metric_families")
    if excluded != {
        "cell_cell_correlation_and_distance": (
            "quadratic_cell_pair_metric_not_scalable"
        ),
        "p_pre_zero_score_evidence": (
            "evaluated_in_main_final_panel_not_retained_in_scaling_panel"
        ),
    }:
        raise ScalingContractError("scaling excluded metric families are invalid")
    if payload.get("genes") != 500 or type(payload["genes"]) is not int:
        raise ScalingContractError("scaling gene count must be exactly 500")
    if payload.get("method_ids") != list(_METHOD_IDS):
        raise ScalingContractError("scaling method denominator is invalid")
    if payload.get("model_seed") != 42 or type(payload["model_seed"]) is not int:
        raise ScalingContractError("scaling model seed must be 42")
    seed = payload.get("seed_derivation")
    if (
        not isinstance(seed, dict)
        or set(seed) != {"algorithm", "master_seed"}
        or seed.get("algorithm") != "sha256-domain-separated-63bit-v1"
        or type(seed.get("master_seed")) is not int
        or not 0 < seed["master_seed"] < 2**63
    ):
        raise ScalingContractError("scaling seed derivation is invalid")
    policy = payload.get("artifact_policy")
    if policy != _ARTIFACT_POLICY:
        raise ScalingContractError("scaling artifact policy is invalid")
    contract = ScalingContract(
        schema_version=1,
        role="post_freeze_resource_scaling",
        mechanism="symsim",
        technical_view="moderate",
        cell_counts=_CELL_COUNTS,
        accuracy_cell_counts=_CELL_COUNTS,
        accuracy_metrics=_SCALING_ACCURACY_METRICS,
        excluded_metric_families=MappingProxyType(dict(excluded)),
        genes=500,
        method_ids=_METHOD_IDS,
        model_seed=42,
        seed_algorithm="sha256-domain-separated-63bit-v1",
        seed_master=seed["master_seed"],
        artifact_policy=MappingProxyType(dict(_ARTIFACT_POLICY)),
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    derived = [
        value
        for cells in contract.cell_counts
        for value in asdict(derive_scaling_seeds(contract, cells)).values()
    ]
    if len(derived) != len(set(derived)):
        raise ScalingContractError("scaling seed derivation contains a collision")
    return contract


def _derived_seed(contract: ScalingContract, cells: int, role: str) -> int:
    payload = json.dumps(
        {
            "schema": "maskimpute-scaling-seed-v1",
            "algorithm": contract.seed_algorithm,
            "master_seed": contract.seed_master,
            "mechanism": contract.mechanism,
            "cells": cells,
            "role": role,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer % (2**63 - 1) + 1


def derive_scaling_seeds(contract: ScalingContract, cells: int) -> ScalingSeeds:
    """Derive the fixed, domain-separated seeds for one panel size."""

    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if cells not in contract.cell_counts or type(cells) is not int:
        raise ScalingContractError("cells is outside the scaling grid")
    return ScalingSeeds(
        biological=_derived_seed(contract, cells, "biological"),
        moderate=_derived_seed(contract, cells, "measurement-moderate"),
        severe=_derived_seed(contract, cells, "measurement-severe"),
    )


def scaling_protocol(base: Protocol, contract: ScalingContract, cells: int) -> Protocol:
    """Create an ephemeral development namespace accepted by the SymSim adapter."""

    if not isinstance(base, Protocol):
        raise TypeError("base must be a Protocol")
    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if cells not in contract.cell_counts or type(cells) is not int:
        raise ScalingContractError("cells is outside the scaling grid")
    development = DevelopmentProtocol(
        namespace=f"scaling-{cells}",
        draws_per_condition=1,
        cells=cells,
        genes=contract.genes,
    )
    return replace(base, development=development)


def scaling_requests(
    contract: ScalingContract, protocol: Protocol, output_root: Path
) -> tuple[SimulationRequest, SimulationRequest]:
    """Build the exact moderate/severe pair needed for one scaling truth draw."""

    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    cells = protocol.development.cells
    if (
        cells not in contract.cell_counts
        or protocol.development.namespace != f"scaling-{cells}"
        or protocol.development.draws_per_condition != 1
        or protocol.development.genes != contract.genes
    ):
        raise ScalingContractError("ephemeral scaling protocol is invalid")
    seeds = derive_scaling_seeds(contract, cells)
    parent = output_root / protocol.development.namespace / "dataset"
    values = tuple(
        SimulationRequest(
            mechanism=contract.mechanism,
            namespace=protocol.development.namespace,
            biological_id="draw-01",
            biological_seed=seeds.biological,
            measurement_seed=getattr(seeds, view),
            technical_view=view,
            cells=cells,
            genes=contract.genes,
            output_path=parent / f"{view}.h5ad",
        )
        for view in ("moderate", "severe")
    )
    return values[0], values[1]


def _scaling_dataset_design_sha256(
    contract: ScalingContract,
    protocol: Protocol,
    request: SimulationRequest,
    seeds: ScalingSeeds,
) -> str:
    return canonical_sha256(
        {
            "schema": "maskimpute-scaling-dataset-design-v1",
            "scaling_contract_sha256": contract.file_sha256,
            "protocol": asdict(protocol),
            "request": {
                "mechanism": request.mechanism,
                "namespace": request.namespace,
                "biological_id": request.biological_id,
                "cells": request.cells,
                "genes": request.genes,
                "biological_seed": request.biological_seed,
                "moderate_measurement_seed": seeds.moderate,
                "severe_measurement_seed": seeds.severe,
            },
        }
    )


def _expected_scaling_dataset_authority(
    contract: ScalingContract,
    base_protocol: Protocol,
    output_dir: Path,
    cells: int,
) -> dict[str, object]:
    protocol = scaling_protocol(base_protocol, contract, cells)
    moderate, _severe = scaling_requests(contract, protocol, output_dir / "generated")
    seeds = derive_scaling_seeds(contract, cells)
    try:
        relative = moderate.output_path.absolute().relative_to(output_dir.absolute())
    except ValueError as error:  # pragma: no cover - constructed from output_dir
        raise ScalingContractError(
            "derived scaling dataset path escaped its result root"
        ) from error
    return {
        "cells": cells,
        "genes": contract.genes,
        "namespace": protocol.development.namespace,
        "mechanism": contract.mechanism,
        "technical_view": contract.technical_view,
        "dataset_id": moderate.dataset_id,
        "independent_unit_id": moderate.independent_unit_id,
        "moderate_output_path": relative.as_posix(),
        "protocol_sha256": canonical_sha256(asdict(protocol)),
        "design_sha256": _scaling_dataset_design_sha256(
            contract, protocol, moderate, seeds
        ),
        "seed_source_sha256": contract.file_sha256,
        "seeds": asdict(seeds),
        "severe_retention": "discarded_after_receipt",
        "native_retention": "discarded_after_receipt",
    }


def build_scaling_plan(
    contract: ScalingContract,
    registry: MethodRegistry,
    configurations: Sequence[FrozenPlanMethodAuthority],
    *,
    frozen_method_sha256: str,
    method_registry_file_sha256: str,
    protocol_file_sha256: str,
    execution_authority_sha256: str,
    execution_environment_sha256: str,
    implementation_source_sha256: str,
) -> ScalingPlan:
    """Close the fixed five-method by four-size scaling denominator."""

    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    values = tuple(configurations)
    if (
        len(values) != len(contract.method_ids)
        or not all(isinstance(value, FrozenPlanMethodAuthority) for value in values)
        or tuple(value.method_id for value in values) != contract.method_ids
    ):
        raise ScalingContractError("scaling configurations differ from method order")
    for method_id in contract.method_ids:
        try:
            spec = registry.by_id(method_id)
        except KeyError as error:
            raise ScalingContractError(
                f"scaling method {method_id} is absent from the registry"
            ) from error
        if not spec.executable:
            raise ScalingContractError(f"scaling method {method_id} is not executable")
    for value in values:
        if (
            value.action != "execute"
            or value.reason is not None
            or value.comparator_nonexecution_identity is not None
        ):
            raise ScalingContractError(
                f"scaling method {value.method_id} is not frozen executable authority"
            )
        if value.method_id in {"dca", "scvi", "magic"}:
            if (
                value.comparator_configuration is None
                or value.legacy_configuration is not None
                or value.configuration_id == "registry-default"
                or not direct_equal(
                    value.comparator_configuration.method,
                    comparator_method_binding(registry.by_id(value.method_id)),
                )
            ):
                raise ScalingContractError(
                    f"scaling frozen comparator {value.method_id} is invalid"
                )
        elif value.comparator_configuration is not None:
            raise ScalingContractError(
                f"scaling legacy method {value.method_id} has comparator authority"
            )
    candidate = values[contract.method_ids.index("maskimpute")]
    candidate_legacy = candidate.legacy_configuration
    if (
        candidate_legacy is None
        or candidate.kind != "candidate_search"
        or candidate.configuration_id == "registry-default"
        or not candidate.requires_count_score
    ):
        raise ScalingContractError("scaling MaskImpute is not the frozen candidate")
    input_hashes = {
        "scaling_contract_sha256": _sha256(
            contract.file_sha256, "scaling contract checksum"
        ),
        "frozen_method_sha256": _sha256(frozen_method_sha256, "frozen method checksum"),
        "method_registry_file_sha256": _sha256(
            method_registry_file_sha256, "method registry file checksum"
        ),
        "protocol_file_sha256": _sha256(
            protocol_file_sha256, "study protocol file checksum"
        ),
        "execution_authority_sha256": _sha256(
            execution_authority_sha256, "execution authority checksum"
        ),
        "execution_environment_sha256": _sha256(
            execution_environment_sha256, "execution environment checksum"
        ),
        "implementation_source_sha256": _sha256(
            implementation_source_sha256, "implementation source checksum"
        ),
    }
    entries: list[ScalingPlanEntry] = []
    ordinal = 0
    by_method = {value.method_id: value for value in values}
    for cells in contract.cell_counts:
        for method_id in contract.method_ids:
            ordinal += 1
            spec = registry.by_id(method_id)
            configuration = by_method[method_id]
            legacy = configuration.legacy_configuration
            comparator = configuration.comparator_configuration
            seed = contract.model_seed if spec.stochastic else None
            seed_token = "deterministic" if seed is None else f"seed-{seed}"
            identity_token = (
                configuration.configuration_id
                if comparator is not None
                else str(legacy.configuration_sha256)[:12]
            )
            entries.append(
                ScalingPlanEntry(
                    ordinal=ordinal,
                    run_id=(
                        f"scaling-{method_id}-{cells}-{seed_token}-{identity_token}"
                    ),
                    cells=cells,
                    genes=contract.genes,
                    method_id=method_id,
                    model_seed=seed,
                    configuration_id=configuration.configuration_id,
                    configuration_sha256=(
                        None if legacy is None else legacy.configuration_sha256
                    ),
                    configuration_kind=configuration.kind,
                    requires_count_score=configuration.requires_count_score,
                    requires_calibration=configuration.requires_calibration,
                    comparator_configuration=comparator,
                    comparator_nonexecution_identity=None,
                    accuracy_enabled=cells in contract.accuracy_cell_counts,
                    native_output_scale=spec.output_scale,
                    timeout_seconds=spec.resources.timeout_seconds,
                    max_rss_bytes=int(spec.resources.max_rss_gib * 1024**3),
                    max_gpu_bytes=int(spec.resources.max_gpu_gib * 1024**3),
                    rss_measurement="linux_proc_process_tree_rss",
                    gpu_measurement=(
                        "nvidia_smi_process_tree_used_memory"
                        if spec.resources.gpu_required
                        else "not_applicable_cpu_only_method"
                    ),
                )
            )
    body = {
        "schema_version": 1,
        "input_hashes": input_hashes,
        "entries": [entry.to_dict() for entry in entries],
        "configurations": [configuration.to_dict() for configuration in values],
    }
    return ScalingPlan(
        schema_version=1,
        input_hashes=MappingProxyType(input_hashes),
        entries=tuple(entries),
        configurations=values,
        plan_sha256=canonical_sha256(body),
    )


def _canonical_executor_receipt(unsigned: Mapping[str, object]) -> bytes:
    payload = dict(unsigned)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return _canonical_bytes(payload) + b"\n"


def _executor_receipt_bytes(
    entry: ScalingPlanEntry,
    run_entry: RunPlanEntry,
    prepared: PreparedDataset,
    outcome: AdapterOutcome,
) -> bytes:
    """Seal the exact parent-side executor outcome before evaluator conversion."""

    snapshot = None if outcome.execution is None else outcome.execution.snapshot
    native_snapshot: dict[str, object] | None = None
    if snapshot is not None:
        native_snapshot = {
            "method_id": snapshot.method_id,
            "source_dataset_sha256": snapshot.source_dataset_sha256,
            "output_scale": snapshot.output_scale,
            "shape": list(snapshot.shape),
            "matrix_sha256": snapshot.matrix_sha256,
        }
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "run_id": entry.run_id,
        "method_id": entry.method_id,
        "dataset_id": run_entry.dataset_id,
        "source_dataset_sha256": run_entry.source_dataset_sha256,
        "model_seed": entry.model_seed,
        "method_input_sha256": method_input_sha256(prepared.method_input),
        "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
        "status": outcome.status,
        "reason": outcome.reason,
        "runtime_seconds": outcome.runtime_seconds,
        "peak_rss_bytes": outcome.peak_rss_bytes,
        "peak_gpu_bytes": outcome.peak_gpu_bytes,
        "rss_measurement": outcome.rss_measurement,
        "gpu_measurement": outcome.gpu_measurement,
        "stdout_sha256": hashlib.sha256(outcome.stdout).hexdigest(),
        "stdout_size_bytes": len(outcome.stdout),
        "stderr_sha256": hashlib.sha256(outcome.stderr).hexdigest(),
        "stderr_size_bytes": len(outcome.stderr),
        "native_snapshot": native_snapshot,
    }
    if entry.comparator_configuration is None:
        unsigned["configuration_sha256"] = entry.configuration_sha256
    else:
        unsigned["comparator_configuration"] = direct_bound_comparator_value(
            entry.comparator_configuration
        )
        unsigned["comparator_nonexecution_identity"] = None
    return _canonical_executor_receipt(unsigned)


def _executor_receipt_from_attempt(attempt: ScalingEvaluatedAttempt) -> bytes:
    """Build an equivalent receipt for direct contract fixtures."""

    run = attempt.run
    native_snapshot: dict[str, object] | None = None
    if run.native_output_sha256 is not None:
        if attempt.native_output is None or attempt.native_output_scale is None:
            raise ScalingContractError(
                "scaling native output bytes are missing from the attempt"
            )
        native = np.asarray(attempt.native_output)
        native_snapshot = {
            "method_id": run.method_id,
            "source_dataset_sha256": run.source_dataset_sha256,
            "output_scale": attempt.native_output_scale,
            "shape": list(native.shape),
            "matrix_sha256": run.native_output_sha256,
        }
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "run_id": run.run_id,
        "method_id": run.method_id,
        "dataset_id": run.dataset_id,
        "source_dataset_sha256": run.source_dataset_sha256,
        "model_seed": run.model_seed,
        "method_input_sha256": run.method_input_sha256,
        "retained_cell_ids_sha256": run.retained_cell_ids_sha256,
        "status": run.status,
        "reason": run.reason,
        "runtime_seconds": run.runtime_seconds,
        "peak_rss_bytes": run.peak_rss_bytes,
        "peak_gpu_bytes": run.peak_gpu_bytes,
        "rss_measurement": run.rss_measurement,
        "gpu_measurement": run.gpu_measurement,
        "stdout_sha256": hashlib.sha256(attempt.stdout).hexdigest(),
        "stdout_size_bytes": len(attempt.stdout),
        "stderr_sha256": hashlib.sha256(attempt.stderr).hexdigest(),
        "stderr_size_bytes": len(attempt.stderr),
        "native_snapshot": native_snapshot,
    }
    if run.comparator_configuration is None:
        unsigned["configuration_sha256"] = run.configuration_sha256
    else:
        unsigned["comparator_configuration"] = direct_bound_comparator_value(
            run.comparator_configuration
        )
        unsigned["comparator_nonexecution_identity"] = None
    return _canonical_executor_receipt(unsigned)


def _parse_executor_receipt(raw: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except ScalingContractError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ScalingContractError("scaling executor receipt is invalid") from error
    if not isinstance(payload, dict):
        raise ScalingContractError("scaling executor receipt fields are not closed")
    keys = set(payload)
    if (
        keys not in (_EXECUTOR_RECEIPT_LEGACY_KEYS, _EXECUTOR_RECEIPT_DIRECT_KEYS)
        or raw != _canonical_bytes(payload) + b"\n"
    ):
        raise ScalingContractError("scaling executor receipt fields are not closed")
    if keys == _EXECUTOR_RECEIPT_DIRECT_KEYS and (
        not isinstance(payload.get("comparator_configuration"), Mapping)
        or payload.get("comparator_nonexecution_identity") is not None
    ):
        raise ScalingContractError("scaling executor comparator identity is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise ScalingContractError("scaling executor receipt checksum mismatch")
    return MappingProxyType(payload)


def scaling_attempt_record(
    attempt: ScalingEvaluatedAttempt, *, cells: int, accuracy_enabled: bool
) -> dict[str, object]:
    """Carry native/evaluator bytes and executor authority to bounded storage."""

    if not isinstance(attempt, ScalingEvaluatedAttempt):
        raise TypeError("attempt must be a ScalingEvaluatedAttempt")
    if type(cells) is not int or cells <= 0:
        raise ValueError("cells must be a positive integer")
    if type(accuracy_enabled) is not bool:
        raise TypeError("accuracy_enabled must be bool")
    run = attempt.run.to_dict()
    run.update(
        {
            "cells": cells,
            "accuracy_enabled": accuracy_enabled,
            "native_output_retention": (
                "not_available"
                if attempt.run.native_output_sha256 is None
                else _NATIVE_OUTPUT_RETENTION
            ),
            "evaluator_output_retention": (
                "not_available"
                if attempt.run.evaluator_output_sha256 is None
                else _EVALUATOR_OUTPUT_RETENTION
            ),
        }
    )
    return {
        "run": run,
        "metrics": [metric.to_dict() for metric in attempt.metrics],
        "stdout": attempt.stdout,
        "stderr": attempt.stderr,
        "native_output": attempt.native_output,
        "evaluator_output": attempt.evaluator_output,
        "executor_receipt": (
            _executor_receipt_from_attempt(attempt)
            if attempt.executor_receipt is None
            else attempt.executor_receipt
        ),
    }


class ScalingResultStore:
    """Canonical checkpoint with replayable compressed method outputs."""

    def __init__(
        self,
        output_dir: Path,
        plan: ScalingPlan,
        *,
        simulator: Any | None = None,
    ) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        if not isinstance(plan, ScalingPlan):
            raise TypeError("plan must be a ScalingPlan")
        selected = Path(os.path.abspath(output_dir))
        selected.mkdir(parents=True, exist_ok=True)
        for component in (selected, *selected.parents):
            try:
                metadata = component.lstat()
            except OSError as error:
                raise ScalingContractError(
                    "scaling output root is unavailable"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ScalingContractError(
                    "scaling output root must not contain symlink components"
                )
        try:
            resolved = selected.resolve(strict=True)
        except OSError as error:
            raise ScalingContractError("scaling output root is unavailable") from error
        if resolved != selected or not resolved.is_dir():
            raise ScalingContractError("scaling output root is not canonical")
        self.output_dir = resolved
        self.plan = plan
        self._simulator = run_symsim_pair if simulator is None else simulator
        if not callable(self._simulator):
            raise TypeError("simulator must be callable")
        repository = Path(__file__).resolve().parents[1]
        contract_path = repository / "study/scaling_panel.json"
        protocol_path = repository / "study/protocol.json"
        self.contract = load_scaling_contract(contract_path)
        self.base_protocol = load_protocol(protocol_path)
        if plan.input_hashes.get(
            "scaling_contract_sha256"
        ) != self.contract.file_sha256 or plan.input_hashes.get(
            "protocol_file_sha256"
        ) != _file_sha256(protocol_path):
            raise ScalingContractError(
                "scaling store authority differs from the tracked design"
            )
        self._prepared_datasets: dict[int, tuple[str, str, PreparedDataset]] = {}
        self._metric_authorities: dict[int, dict[str, object]] = {}
        self._snapshot: ScalingCheckpoint | None | object = _UNLOADED
        self._checkpoint_file_sha256: tuple[int, str, str] | None | object = _UNLOADED

    @contextmanager
    def _exclusive_operation(self):
        """Serialize one cache/checkpoint transaction on the stable output inode."""

        descriptor = -1
        try:
            named_before = self.output_dir.lstat()
            descriptor = os.open(
                self.output_dir,
                os.O_RDONLY
                | os.O_DIRECTORY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened_before.st_mode)
                or stat.S_ISLNK(named_before.st_mode)
                or (opened_before.st_dev, opened_before.st_ino)
                != (named_before.st_dev, named_before.st_ino)
            ):
                raise ScalingContractError(
                    "scaling operation lock directory is invalid"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened_after = os.fstat(descriptor)
            named_after = self.output_dir.lstat()
            if (
                (opened_after.st_dev, opened_after.st_ino)
                != (opened_before.st_dev, opened_before.st_ino)
                or (named_after.st_dev, named_after.st_ino)
                != (opened_before.st_dev, opened_before.st_ino)
                or stat.S_ISLNK(named_after.st_mode)
            ):
                raise ScalingContractError("scaling operation lock directory changed")
        except ScalingContractError:
            raise
        except OSError as error:
            raise ScalingContractError(
                "scaling operation lock is unavailable"
            ) from error
        try:
            yield
        finally:
            if descriptor >= 0:
                try:
                    opened_final = os.fstat(descriptor)
                    named_final = self.output_dir.lstat()
                    if (
                        (opened_final.st_dev, opened_final.st_ino)
                        != (opened_before.st_dev, opened_before.st_ino)
                        or (named_final.st_dev, named_final.st_ino)
                        != (opened_before.st_dev, opened_before.st_ino)
                        or stat.S_ISLNK(named_final.st_mode)
                    ):
                        raise ScalingContractError(
                            "scaling operation lock directory changed"
                        )
                except OSError as error:
                    raise ScalingContractError(
                        "scaling operation lock directory changed"
                    ) from error
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)

    @property
    def checkpoint_path(self) -> Path:
        paths = self._checkpoint_paths()
        if paths:
            return paths[-1]
        return self.output_dir / "checkpoints/00000001.json"

    def _checkpoint_paths(self) -> tuple[Path, ...]:
        root = self.output_dir / "checkpoints"
        if not os.path.lexists(root):
            return ()
        try:
            metadata = root.lstat()
            paths = tuple(sorted(root.iterdir(), key=lambda path: path.name))
        except OSError as error:
            raise ScalingContractError(
                "scaling checkpoint directory cannot be read"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ScalingContractError("scaling checkpoint directory is invalid")
        expected = tuple(f"{index:08d}.json" for index in range(1, len(paths) + 1))
        if tuple(path.name for path in paths) != expected or any(
            _SCALING_CHECKPOINT_FILE.fullmatch(path.name) is None for path in paths
        ):
            raise ScalingContractError(
                "scaling checkpoint history is not a contiguous prefix"
            )
        for path in paths:
            item = path.lstat()
            if (
                stat.S_ISLNK(item.st_mode)
                or not stat.S_ISREG(item.st_mode)
                or item.st_nlink != 1
                or item.st_size > _MAX_CHECKPOINT_BYTES
            ):
                raise ScalingContractError(
                    "scaling checkpoint history contains an invalid file"
                )
        return paths

    def _checkpoint_state(self, raw: bytes | None) -> tuple[int, str, str] | None:
        paths = self._checkpoint_paths()
        if raw is None:
            if paths:
                raise ScalingContractError("scaling checkpoint state is inconsistent")
            return None
        if not paths:
            raise ScalingContractError("scaling checkpoint state is inconsistent")
        return (len(paths), paths[-1].name, hashlib.sha256(raw).hexdigest())

    def _checkpoint_raw(self) -> bytes | None:
        paths = self._checkpoint_paths()
        if not paths:
            return None
        path = paths[-1]
        _reject_symlink_components(self.output_dir, path, "scaling checkpoint")
        try:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > _MAX_CHECKPOINT_BYTES
            ):
                raise ScalingContractError("scaling checkpoint file is invalid")
            return path.read_bytes()
        except ScalingContractError:
            raise
        except OSError as error:
            raise ScalingContractError("scaling checkpoint cannot be read") from error

    def _require_checkpoint_unchanged(self) -> None:
        if self._checkpoint_file_sha256 is _UNLOADED:
            raise ScalingContractError("scaling checkpoint cache is not initialized")
        raw = self._checkpoint_raw()
        observed = self._checkpoint_state(raw)
        if observed != self._checkpoint_file_sha256:
            raise ScalingContractError(
                "scaling checkpoint changed after the cached validation"
            )

    @staticmethod
    def _detached_checkpoint(
        checkpoint: ScalingCheckpoint | None,
    ) -> ScalingCheckpoint | None:
        if checkpoint is None:
            return None
        datasets = tuple(
            json.loads(_canonical_bytes(dict(value)).decode("utf-8"))
            for value in checkpoint.datasets
        )
        records = tuple(
            json.loads(_canonical_bytes(dict(value)).decode("utf-8"))
            for value in checkpoint.records
        )
        return ScalingCheckpoint(
            schema_version=checkpoint.schema_version,
            plan_sha256=checkpoint.plan_sha256,
            input_hashes=MappingProxyType(dict(checkpoint.input_hashes)),
            planned_run_count=checkpoint.planned_run_count,
            status=checkpoint.status,
            datasets=datasets,
            records=records,
            checkpoint_sha256=checkpoint.checkpoint_sha256,
        )

    def _artifact_path(
        self,
        value: object,
        name: str,
        *,
        artifact_directory: Path | None = None,
    ) -> Path:
        relative = _safe_relative_path(value, name)
        path = (self.output_dir / relative).absolute()
        try:
            path.relative_to(self.output_dir)
        except ValueError as error:
            raise ScalingContractError(f"{name} escaped the output root") from error
        _reject_symlink_components(self.output_dir, path, name)
        if artifact_directory is None:
            return path
        selected = artifact_directory / relative.name
        _reject_symlink_components(self.output_dir, selected, name)
        return selected

    def _verify_artifact(
        self,
        relative: object,
        digest: object,
        nbytes: object,
        name: str,
        *,
        max_bytes: int = _MAX_LOG_BYTES,
        artifact_directory: Path | None = None,
    ) -> Path:
        expected = _sha256(digest, f"{name} checksum")
        if type(nbytes) is not int or nbytes < 0 or nbytes > max_bytes:
            raise ScalingContractError(f"{name} byte count is invalid")
        path = self._artifact_path(
            relative, name, artifact_directory=artifact_directory
        )
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ScalingContractError(f"{name} is missing") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != nbytes
            or _file_sha256(path) != expected
        ):
            raise ScalingContractError(f"{name} integrity check failed")
        return path

    def _verify_generator_authority(
        self, receipt: Mapping[str, object], cells: int
    ) -> None:
        """Regenerate one exact SymSim pair and compare independent identities."""

        try:
            with tempfile.TemporaryDirectory(
                prefix=f"maskimpute-scaling-authority-{cells}-"
            ) as temporary_name:
                authority_root = Path(temporary_name).resolve()
                protocol = scaling_protocol(self.base_protocol, self.contract, cells)
                requests = scaling_requests(
                    self.contract, protocol, authority_root / "generated"
                )
                artifacts = self._simulator(requests, protocol)
                expected, _moderate = _dataset_receipt_from_artifacts(
                    self.contract,
                    protocol,
                    authority_root,
                    artifacts,
                )
                if any(
                    receipt.get(name) != expected.get(name)
                    for name in _GENERATOR_RECEIPT_FIELDS
                ):
                    raise ScalingContractError(
                        "retained scaling dataset differs from deterministic SymSim "
                        "generator authority"
                    )
        except ScalingContractError:
            raise
        except Exception as error:
            raise ScalingContractError(
                "deterministic SymSim generator authority could not be reproduced"
            ) from error

    def _validate_dataset_receipt(
        self,
        value: object,
        expected_cells: int,
        *,
        verify_generator: bool,
    ) -> Mapping[str, object]:
        if not isinstance(value, dict):
            raise ScalingContractError("scaling dataset receipt must be an object")
        if set(value) != _DATASET_RECEIPT_KEYS:
            raise ScalingContractError("scaling dataset receipt fields are not closed")
        unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
        if (
            type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or value.get("cells") != expected_cells
            or value.get("genes") != 500
            or value.get("namespace") != f"scaling-{expected_cells}"
            or value.get("mechanism") != "symsim"
            or value.get("technical_view") != "moderate"
            or value.get("receipt_sha256") != canonical_sha256(unsigned)
        ):
            raise ScalingContractError("scaling dataset receipt binding is invalid")
        expected_authority = _expected_scaling_dataset_authority(
            self.contract, self.base_protocol, self.output_dir, expected_cells
        )
        if any(
            value.get(name) != expected for name, expected in expected_authority.items()
        ):
            raise ScalingContractError(
                "scaling dataset seed, design, or protocol authority differs"
            )
        for field in (
            "dataset_sha256",
            "truth_sha256",
            "moderate_output_file_sha256",
            "severe_dataset_sha256",
            "severe_output_file_sha256",
            "moderate_native_manifest_sha256",
            "severe_native_manifest_sha256",
            "native_files_sha256",
            "protocol_sha256",
            "design_sha256",
            "seed_source_sha256",
        ):
            _sha256(value.get(field), f"scaling dataset {field}")
        if (
            not isinstance(value.get("dataset_id"), str)
            or not value["dataset_id"].startswith("dataset-")
            or not isinstance(value.get("independent_unit_id"), str)
            or type(value.get("moderate_output_size_bytes")) is not int
            or value["moderate_output_size_bytes"] <= 0
            or type(value.get("severe_output_size_bytes")) is not int
            or value["severe_output_size_bytes"] <= 0
            or value.get("severe_retention") != "discarded_after_receipt"
            or value.get("native_retention") != "discarded_after_receipt"
        ):
            raise ScalingContractError("scaling dataset receipt fields are invalid")
        seeds = value.get("seeds")
        if (
            not isinstance(seeds, dict)
            or set(seeds) != {"biological", "moderate", "severe"}
            or any(
                type(item) is not int or not 0 < item < 2**63 for item in seeds.values()
            )
            or len(set(seeds.values())) != 3
        ):
            raise ScalingContractError("scaling dataset receipt seeds are invalid")
        output_path = self._artifact_path(
            value.get("moderate_output_path"), "moderate scaling dataset"
        )
        try:
            metadata = output_path.lstat()
        except OSError as error:
            raise ScalingContractError("moderate scaling dataset is missing") from error
        identity = _regular_file_identity(metadata)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != value["moderate_output_size_bytes"]
        ):
            raise ScalingContractError("moderate scaling dataset integrity failed")
        if metadata.st_size > _scaling_h5ad_size_ceiling(expected_cells, 500):
            raise ScalingContractError("moderate scaling H5AD exceeds its size bound")
        _validate_scaling_h5ad_structure(output_path, expected_cells, 500)
        if _file_sha256(output_path) != value["moderate_output_file_sha256"]:
            raise ScalingContractError("moderate scaling dataset integrity failed")
        if verify_generator:
            self._verify_generator_authority(value, expected_cells)
        receipt_sha = str(value["receipt_sha256"])
        file_sha = str(value["moderate_output_file_sha256"])
        cached = self._prepared_datasets.get(expected_cells)
        if cached is None or cached[:2] != (receipt_sha, file_sha):
            try:
                import anndata as ad

                from .datasets import _truth_sha256
                from .runner import DatasetQCPolicy
                from .schema import benchmark_dataset_sha256

                dataset = ad.read_h5ad(output_path)
                try:
                    current_identity = _regular_file_identity(output_path.lstat())
                except OSError as error:
                    raise ScalingContractError(
                        "moderate scaling dataset identity changed during read"
                    ) from error
                if current_identity != identity:
                    raise ScalingContractError(
                        "moderate scaling dataset identity changed during read"
                    )
                if (
                    benchmark_dataset_sha256(dataset) != value["dataset_sha256"]
                    or _truth_sha256(dataset) != value["truth_sha256"]
                ):
                    raise ScalingContractError(
                        "moderate scaling dataset semantic hash mismatch"
                    )
                prepared = prepare_dataset_for_execution(
                    dataset,
                    _dataset_binding(value),
                    DatasetQCPolicy.fixed(),
                )
            except ScalingContractError:
                raise
            except Exception as error:
                raise ScalingContractError(
                    "moderate scaling dataset semantic validation failed"
                ) from error
            self._prepared_datasets[expected_cells] = (
                receipt_sha,
                file_sha,
                prepared,
            )
            from .runner import _evaluator_targets

            observed, truth, truth_kind, _marker_mask = _evaluator_targets(prepared)
            if truth_kind != "exact_pre_capture" or truth is None:
                raise ScalingContractError(
                    "scaling metric validation requires exact pre-capture truth"
                )
            self._metric_authorities[expected_cells] = {
                "mse": int(observed.size),
                "mse_dropout": int(((observed == 0) & (truth > 0)).sum()),
                "mse_pre_dropout_zero": int((truth == 0).sum()),
                "mse_nonzero": int(((observed > 0) & (truth > 0)).sum()),
                "gnrmse": int(truth.shape[1]),
                "mean_distortion": int(truth.shape[1]),
                "variance_distortion": int(truth.shape[1]),
                "mean_gene_wasserstein_distance": int(truth.shape[1]),
                "n_corr_genes": int(truth.shape[1]),
                "correlation_pairs": int(truth.shape[1] * (truth.shape[1] - 1) // 2),
            }
        return MappingProxyType(dict(value))

    def _load_executor_receipt(
        self,
        run: Mapping[str, object],
        entry: ScalingPlanEntry,
        prepared: PreparedDataset,
        *,
        artifact_directory: Path | None,
    ) -> Mapping[str, object]:
        expected_path = f"runs/{entry.run_id}/run.executor-receipt.json"
        if run.get("executor_receipt_path") != expected_path:
            raise ScalingContractError("scaling executor receipt path differs")
        path = self._verify_artifact(
            run.get("executor_receipt_path"),
            run.get("executor_receipt_file_sha256"),
            run.get("executor_receipt_size_bytes"),
            "scaling executor receipt",
            max_bytes=_MAX_EXECUTOR_RECEIPT_BYTES,
            artifact_directory=artifact_directory,
        )
        receipt = _parse_executor_receipt(path.read_bytes())
        expected = {
            "schema_version": 1,
            "run_id": entry.run_id,
            "method_id": entry.method_id,
            "dataset_id": prepared.binding.dataset_id,
            "source_dataset_sha256": prepared.binding.dataset_sha256,
            "model_seed": entry.model_seed,
            "method_input_sha256": method_input_sha256(prepared.method_input),
            "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
        }
        if entry.comparator_configuration is None:
            expected["configuration_sha256"] = entry.configuration_sha256
        else:
            expected["comparator_configuration"] = direct_bound_comparator_value(
                entry.comparator_configuration
            )
            expected["comparator_nonexecution_identity"] = None
        if any(
            not direct_equal(receipt.get(name), value)
            for name, value in expected.items()
        ):
            raise ScalingContractError("scaling executor receipt identity differs")
        if run.get("executor_receipt_sha256") != receipt.get("receipt_sha256"):
            raise ScalingContractError("scaling executor receipt binding differs")
        status = receipt.get("status")
        reason = receipt.get("reason")
        runtime_seconds = _finite_real(
            receipt.get("runtime_seconds"),
            "scaling executor runtime",
        )
        if (
            status
            not in {
                "completed",
                "unavailable",
                "failed",
                "timeout",
                "resource_exceeded",
                "infrastructure_error",
                "blocked_authority",
                "budget_exhausted",
            }
            or (status == "completed" and reason is not None)
            or (status != "completed" and (not isinstance(reason, str) or not reason))
            or runtime_seconds < 0.0
            or any(
                isinstance(receipt.get(name), bool)
                or type(receipt.get(name)) is not int
                or receipt[name] < 0
                for name in (
                    "peak_rss_bytes",
                    "peak_gpu_bytes",
                    "stdout_size_bytes",
                    "stderr_size_bytes",
                )
            )
        ):
            raise ScalingContractError("scaling executor receipt values are invalid")
        for name in ("stdout_sha256", "stderr_sha256"):
            _sha256(receipt.get(name), f"scaling executor {name}")
        native = receipt.get("native_snapshot")
        if status == "completed":
            if not isinstance(native, dict) or set(native) != _NATIVE_SNAPSHOT_KEYS:
                raise ScalingContractError(
                    "completed scaling executor receipt lacks a native snapshot"
                )
        elif native is not None:
            raise ScalingContractError(
                "noncompleted scaling executor receipt has a native snapshot"
            )
        return receipt

    def _decode_native_output(
        self,
        run: Mapping[str, object],
        entry: ScalingPlanEntry,
        prepared: PreparedDataset,
        receipt: Mapping[str, object],
        *,
        artifact_directory: Path | None,
    ) -> tuple[np.ndarray, bytes] | None:
        storage_fields = (
            "native_output_path",
            "native_output_file_sha256",
            "native_output_shape",
            "native_output_dtype",
            "native_output_scale",
            "native_output_encoding",
            "native_output_compressed_nbytes",
            "native_output_uncompressed_nbytes",
            "native_output_uncompressed_sha256",
        )
        native_receipt = receipt.get("native_snapshot")
        if receipt.get("status") != "completed":
            if (
                run.get("native_output_sha256") is not None
                or run.get("native_output_retention") != "not_available"
                or any(run.get(name) is not None for name in storage_fields)
            ):
                raise ScalingContractError(
                    "noncompleted scaling executor retains native output evidence"
                )
            return None
        assert isinstance(native_receipt, Mapping)
        expected_shape = prepared.method_input.shape
        expected_nbytes = int(np.prod(expected_shape, dtype=np.int64)) * 8
        if (
            run.get("native_output_retention") != _NATIVE_OUTPUT_RETENTION
            or run.get("native_output_path")
            != f"runs/{entry.run_id}/run.native-f64.zlib"
            or run.get("native_output_shape") != list(expected_shape)
            or run.get("native_output_dtype") != "<f8"
            or run.get("native_output_scale") != entry.native_output_scale
            or run.get("native_output_encoding") != _NATIVE_OUTPUT_ENCODING
            or run.get("native_output_uncompressed_nbytes") != expected_nbytes
            or expected_nbytes > _MAX_NATIVE_OUTPUT_BYTES
            or native_receipt.get("method_id") != entry.method_id
            or native_receipt.get("source_dataset_sha256")
            != prepared.binding.dataset_sha256
            or native_receipt.get("output_scale") != entry.native_output_scale
            or native_receipt.get("shape") != list(expected_shape)
        ):
            raise ScalingContractError("scaling native output identity differs")
        raw_sha256 = _sha256(
            run.get("native_output_uncompressed_sha256"),
            "scaling native uncompressed output",
        )
        compressed_nbytes = run.get("native_output_compressed_nbytes")
        maximum_compressed = _zlib_compress_bound(expected_nbytes)
        if (
            type(compressed_nbytes) is not int
            or not 0 < compressed_nbytes <= maximum_compressed
        ):
            raise ScalingContractError(
                "scaling native compressed byte count is invalid"
            )
        path = self._verify_artifact(
            run.get("native_output_path"),
            run.get("native_output_file_sha256"),
            compressed_nbytes,
            "scaling native output",
            max_bytes=maximum_compressed,
            artifact_directory=artifact_directory,
        )
        raw = self._decompress_matrix_bytes(path, expected_nbytes, raw_sha256, "native")
        output = np.frombuffer(raw, dtype="<f8").reshape(expected_shape)
        if not np.isfinite(output).all() or bool((output < 0).any()):
            raise ScalingContractError("scaling native output values are invalid")
        from .methods import _output_digest

        identity = _output_digest(
            method_id=entry.method_id,
            source_dataset_sha256=prepared.binding.dataset_sha256,
            output_scale=entry.native_output_scale,
            obs_ids=prepared.audit.retained_cell_ids,
            var_ids=prepared.method_input.var_ids,
            shape=expected_shape,
            matrix_bytes=raw,
        )
        if (
            run.get("native_output_sha256") != identity
            or native_receipt.get("matrix_sha256") != identity
        ):
            raise ScalingContractError("scaling native snapshot hash differs")
        return output, raw

    @staticmethod
    def _decompress_matrix_bytes(
        path: Path, expected_nbytes: int, expected_sha256: str, name: str
    ) -> bytes:
        try:
            compressed = path.read_bytes()
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(compressed, expected_nbytes + 1)
            raw += decompressor.flush(max(1, expected_nbytes + 1 - len(raw)))
        except (OSError, zlib.error) as error:
            raise ScalingContractError(
                f"scaling {name} output cannot be decompressed"
            ) from error
        if (
            len(raw) != expected_nbytes
            or not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise ScalingContractError(
                f"scaling {name} output differs from its receipt"
            )
        return raw

    def _decode_evaluator_output(
        self,
        run: Mapping[str, object],
        entry: ScalingPlanEntry,
        prepared: PreparedDataset,
        *,
        artifact_directory: Path | None,
    ) -> np.ndarray | None:
        storage_fields = (
            "evaluator_output_path",
            "evaluator_output_file_sha256",
            "evaluator_output_shape",
            "evaluator_output_dtype",
            "evaluator_output_scale",
            "evaluator_output_encoding",
            "evaluator_output_compressed_nbytes",
            "evaluator_output_uncompressed_nbytes",
            "evaluator_output_uncompressed_sha256",
        )
        if run.get("status") != "completed":
            if (
                run.get("evaluator_output_sha256") is not None
                or run.get("evaluator_output_retention") != "not_available"
                or any(run.get(name) is not None for name in storage_fields)
            ):
                raise ScalingContractError(
                    "noncompleted scaling run retains evaluator output evidence"
                )
            return None
        shape = run.get("evaluator_output_shape")
        expected_shape = prepared.method_input.shape
        expected_nbytes = int(np.prod(expected_shape, dtype=np.int64)) * 8
        if (
            run.get("evaluator_output_retention") != _EVALUATOR_OUTPUT_RETENTION
            or run.get("evaluator_output_path")
            != f"runs/{entry.run_id}/run.log2-cp10k-f64.zlib"
            or shape != list(expected_shape)
            or run.get("evaluator_output_dtype") != "<f8"
            or run.get("evaluator_output_scale") != _EVALUATOR_OUTPUT_SCALE
            or run.get("evaluator_output_encoding") != _EVALUATOR_OUTPUT_ENCODING
            or run.get("evaluator_output_uncompressed_nbytes") != expected_nbytes
            or expected_nbytes > _MAX_EVALUATOR_OUTPUT_BYTES
        ):
            raise ScalingContractError("scaling evaluator output identity differs")
        raw_sha256 = _sha256(
            run.get("evaluator_output_uncompressed_sha256"),
            "scaling evaluator uncompressed output",
        )
        compressed_nbytes = run.get("evaluator_output_compressed_nbytes")
        maximum_compressed = _zlib_compress_bound(expected_nbytes)
        if (
            type(compressed_nbytes) is not int
            or not 0 < compressed_nbytes <= maximum_compressed
        ):
            raise ScalingContractError(
                "scaling evaluator compressed byte count is invalid"
            )
        path = self._verify_artifact(
            run.get("evaluator_output_path"),
            run.get("evaluator_output_file_sha256"),
            compressed_nbytes,
            "scaling evaluator output",
            max_bytes=maximum_compressed,
            artifact_directory=artifact_directory,
        )
        raw = self._decompress_matrix_bytes(
            path, expected_nbytes, raw_sha256, "evaluator"
        )
        output = np.frombuffer(raw, dtype="<f8").reshape(expected_shape).copy()
        if not np.isfinite(output).all() or bool((output < 0).any()):
            raise ScalingContractError("scaling evaluator output values are invalid")
        from .runner import _evaluator_output_sha256

        expected_identity = _evaluator_output_sha256(
            _run_plan_entry(entry, prepared.binding), prepared, output
        )
        if run.get("evaluator_output_sha256") != expected_identity:
            raise ScalingContractError("scaling evaluator output hash identity differs")
        return output

    def _validate_run_directory_closed(
        self,
        run: Mapping[str, object],
        entry: ScalingPlanEntry,
        *,
        artifact_directory: Path | None,
    ) -> None:
        directory = (
            self.output_dir / "runs" / entry.run_id
            if artifact_directory is None
            else artifact_directory
        )
        _reject_symlink_components(self.output_dir, directory, "scaling run directory")
        try:
            metadata = directory.lstat()
            entries = tuple(directory.iterdir())
        except OSError as error:
            raise ScalingContractError(
                "scaling run directory cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ScalingContractError("scaling run directory is invalid")
        expected = {
            Path(str(run[name])).name
            for name in (
                "stdout_path",
                "stderr_path",
                "executor_receipt_path",
                "native_output_path",
                "evaluator_output_path",
            )
            if run.get(name) is not None
        }
        if {child.name for child in entries} != expected or any(
            stat.S_ISLNK(child.lstat().st_mode)
            or not stat.S_ISREG(child.lstat().st_mode)
            or child.lstat().st_nlink != 1
            for child in entries
        ):
            raise ScalingContractError("scaling run directory fields are not closed")

    def _validate_record(
        self,
        value: object,
        entry: ScalingPlanEntry,
        dataset_receipt: Mapping[str, object],
        *,
        artifact_directory: Path | None = None,
    ) -> Mapping[str, object]:
        if not isinstance(value, dict) or set(value) != {
            "run",
            "metrics",
            "record_sha256",
        }:
            raise ScalingContractError("stored scaling record has wrong fields")
        unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
        if value.get("record_sha256") != canonical_sha256(unsigned):
            raise ScalingContractError("stored scaling record checksum mismatch")
        run = value.get("run")
        metrics = value.get("metrics")
        if not isinstance(run, dict) or not isinstance(metrics, list):
            raise ScalingContractError("stored scaling record is malformed")
        from .runner import LongFormMetric, RawRunResult

        direct_only_fields = {
            "comparator_configuration",
            "comparator_nonexecution_identity",
        }
        raw_run_fields = {field.name for field in fields(RawRunResult)}
        expected_run_fields = (
            raw_run_fields
            - (
                {"configuration_sha256"}
                if entry.comparator_configuration is not None
                else direct_only_fields
            )
        ) | {
            "cells",
            "accuracy_enabled",
            "native_output_retention",
            "evaluator_output_retention",
            "stdout_path",
            "stdout_file_sha256",
            "stdout_size_bytes",
            "stderr_path",
            "stderr_file_sha256",
            "stderr_size_bytes",
            "executor_receipt_path",
            "executor_receipt_file_sha256",
            "executor_receipt_size_bytes",
            "executor_receipt_sha256",
            "native_output_path",
            "native_output_file_sha256",
            "native_output_shape",
            "native_output_dtype",
            "native_output_scale",
            "native_output_encoding",
            "native_output_compressed_nbytes",
            "native_output_uncompressed_nbytes",
            "native_output_uncompressed_sha256",
            "evaluator_output_path",
            "evaluator_output_file_sha256",
            "evaluator_output_shape",
            "evaluator_output_dtype",
            "evaluator_output_scale",
            "evaluator_output_encoding",
            "evaluator_output_compressed_nbytes",
            "evaluator_output_uncompressed_nbytes",
            "evaluator_output_uncompressed_sha256",
        }
        raw_metric_fields = {field.name for field in fields(LongFormMetric)}
        metric_fields = raw_metric_fields - (
            {"configuration_sha256"}
            if entry.comparator_configuration is not None
            else direct_only_fields
        )
        if set(run) != expected_run_fields or any(
            not isinstance(metric, dict) or set(metric) != metric_fields
            for metric in metrics
        ):
            raise ScalingContractError(
                "stored scaling run/metric fields are not closed"
            )
        expected = {
            "run_id": entry.run_id,
            "method_id": entry.method_id,
            "model_seed": entry.model_seed,
            "configuration_id": entry.configuration_id,
            "configuration_kind": entry.configuration_kind,
            "requires_count_score": entry.requires_count_score,
            "requires_calibration": entry.requires_calibration,
            "cells": entry.cells,
            "accuracy_enabled": entry.accuracy_enabled,
        }
        if entry.comparator_configuration is None:
            expected["configuration_sha256"] = entry.configuration_sha256
        else:
            expected["comparator_configuration"] = direct_bound_comparator_value(
                entry.comparator_configuration
            )
            expected["comparator_nonexecution_identity"] = None
        if any(
            not direct_equal(run.get(name), expected_value)
            for name, expected_value in expected.items()
        ):
            raise ScalingContractError("stored scaling record differs from its plan")
        cached = self._prepared_datasets.get(entry.cells)
        if cached is None:
            raise ScalingContractError(
                "stored scaling record lacks a validated dataset authority"
            )
        prepared = cached[2]
        from .runner import (
            DatasetQCPolicy,
            _OUTCOME_STATUSES,
            method_input_sha256,
        )

        metric_authority = self._metric_authorities.get(entry.cells)
        if metric_authority is None:
            raise ScalingContractError(
                "stored scaling metrics lack validated truth authority"
            )
        expected_run = {
            "dataset_id": dataset_receipt["dataset_id"],
            "source_dataset_sha256": dataset_receipt["dataset_sha256"],
            "mechanism": self.contract.mechanism,
            "biological_id": "draw-01",
            "technical_view": self.contract.technical_view,
            "method_input_sha256": method_input_sha256(prepared.method_input),
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "excluded_cell_count": prepared.audit.excluded_cell_count,
            "excluded_cell_ids_sha256": prepared.audit.excluded_cell_ids_sha256,
            "retained_cell_count": prepared.audit.retained_cell_count,
            "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
            "retained_gene_count": prepared.method_input.shape[1],
            "observed_zero_count": int((prepared.method_input.counts == 0).sum()),
        }
        if any(run.get(name) != expected for name, expected in expected_run.items()):
            raise ScalingContractError("stored scaling record dataset identity differs")
        executor_receipt = self._load_executor_receipt(
            run,
            entry,
            prepared,
            artifact_directory=artifact_directory,
        )
        status = run.get("status")
        runtime = _finite_real(
            run.get("runtime_seconds"),
            "stored scaling runtime",
        )
        if (
            status not in _OUTCOME_STATUSES
            or runtime < 0.0
            or any(
                isinstance(run.get(name), bool)
                or type(run.get(name)) is not int
                or run[name] < 0
                for name in ("peak_rss_bytes", "peak_gpu_bytes")
            )
        ):
            raise ScalingContractError(
                "stored scaling status, runtime, or resource evidence is invalid"
            )
        peak_rss = int(run["peak_rss_bytes"])
        peak_gpu = int(run["peak_gpu_bytes"])
        for name in (
            "runtime_seconds",
            "peak_rss_bytes",
            "peak_gpu_bytes",
            "rss_measurement",
            "gpu_measurement",
            "stdout_sha256",
            "stderr_sha256",
        ):
            if run.get(name) != executor_receipt.get(name):
                raise ScalingContractError(
                    "stored scaling run differs from its executor receipt"
                )
        if run.get("stdout_size_bytes") != executor_receipt.get(
            "stdout_size_bytes"
        ) or run.get("stderr_size_bytes") != executor_receipt.get("stderr_size_bytes"):
            raise ScalingContractError(
                "stored scaling logs differ from their executor receipt"
            )
        telemetry_unavailable = (
            status == "infrastructure_error"
            and run.get("reason") == "resource_telemetry_unavailable"
        )
        rss_provenance_valid = run.get("rss_measurement") == entry.rss_measurement or (
            telemetry_unavailable
            and run.get("rss_measurement") == "rss_measurement_unavailable"
        )
        gpu_provenance_valid = run.get("gpu_measurement") == entry.gpu_measurement or (
            telemetry_unavailable
            and run.get("gpu_measurement")
            in {"gpu_measurement_unavailable", "nvidia_smi_measurement_unavailable"}
        )
        resource_status_valid = (
            status == "resource_exceeded"
            and (
                (
                    run.get("reason") == "peak_rss_exceeded"
                    and peak_rss > entry.max_rss_bytes
                )
                or (
                    run.get("reason") == "peak_gpu_exceeded"
                    and peak_gpu > entry.max_gpu_bytes
                )
            )
        ) or (
            status != "resource_exceeded"
            and peak_rss <= entry.max_rss_bytes
            and peak_gpu <= entry.max_gpu_bytes
        )
        runtime_valid = (
            runtime <= entry.timeout_seconds
            if status != "timeout"
            else entry.timeout_seconds <= runtime <= entry.timeout_seconds + 5.0
        )
        if not (
            rss_provenance_valid
            and gpu_provenance_valid
            and resource_status_valid
            and runtime_valid
        ):
            raise ScalingContractError(
                "stored scaling runtime or resource measurement authority differs"
            )
        if run["status"] == "completed" and run.get("reason") is not None:
            raise ScalingContractError(
                "completed scaling run cannot have a terminal reason"
            )
        if run["status"] != "completed" and (
            not isinstance(run.get("reason"), str) or not run["reason"]
        ):
            raise ScalingContractError(
                "noncompleted scaling run lacks its terminal reason"
            )
        if status == "timeout" and run.get("reason") != "timeout":
            raise ScalingContractError("timeout scaling run has a noncanonical reason")
        if (
            run.get("calibration_artifact_sha256") is not None
            or run.get("calibration_context_sha256") is not None
            or run.get("calibration_fold_calibrator_sha256") is not None
            or run.get("calibration_training_manifest_sha256s") not in ((), [])
            or run.get("calibration_held_out_manifest_sha256s") not in ((), [])
        ):
            raise ScalingContractError(
                "scaling run contains an unauthorized fold-calibration receipt"
            )
        for name in ("stdout", "stderr"):
            if run.get(f"{name}_sha256") != run.get(f"{name}_file_sha256"):
                raise ScalingContractError(f"scaling {name} content receipt differs")
        for name in ("native_output_sha256", "evaluator_output_sha256"):
            if run.get(name) is not None:
                _sha256(run[name], f"scaling {name}")
        if executor_receipt.get("status") == "completed" and (
            run.get("native_output_sha256") is None
        ):
            raise ScalingContractError(
                "completed scaling executor lacks native output hash evidence"
            )
        if run["status"] == "completed" and run.get("evaluator_output_sha256") is None:
            raise ScalingContractError(
                "completed scaling run lacks evaluator output hash evidence"
            )
        if run.get("native_output_retention") not in {
            _NATIVE_OUTPUT_RETENTION,
            "not_available",
        }:
            raise ScalingContractError("native scaling output retention is invalid")
        if run.get("evaluator_output_retention") not in {
            _EVALUATOR_OUTPUT_RETENTION,
            "not_available",
        }:
            raise ScalingContractError("evaluator scaling output retention is invalid")
        if run["native_output_retention"] != (
            "not_available"
            if run.get("native_output_sha256") is None
            else _NATIVE_OUTPUT_RETENTION
        ) or run["evaluator_output_retention"] != (
            "not_available"
            if run.get("evaluator_output_sha256") is None
            else _EVALUATOR_OUTPUT_RETENTION
        ):
            raise ScalingContractError(
                "scaling output retention differs from its hashes"
            )
        if (
            run.get("stdout_path") != f"runs/{entry.run_id}/run.stdout"
            or run.get("stderr_path") != f"runs/{entry.run_id}/run.stderr"
        ):
            raise ScalingContractError("scaling log paths differ from the plan")
        stdout_path = self._verify_artifact(
            run.get("stdout_path"),
            run.get("stdout_file_sha256"),
            run.get("stdout_size_bytes"),
            "scaling stdout",
            artifact_directory=artifact_directory,
        )
        stderr_path = self._verify_artifact(
            run.get("stderr_path"),
            run.get("stderr_file_sha256"),
            run.get("stderr_size_bytes"),
            "scaling stderr",
            artifact_directory=artifact_directory,
        )
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes()
        native_decoded = self._decode_native_output(
            run,
            entry,
            prepared,
            executor_receipt,
            artifact_directory=artifact_directory,
        )
        expected_status = str(executor_receipt["status"])
        expected_reason = executor_receipt.get("reason")
        converted_output: np.ndarray | None = None
        if native_decoded is not None:
            native_output, native_raw = native_decoded
            from .methods import (
                AdapterExecution,
                AdapterUnavailableError,
                MethodOutputSnapshot,
            )
            from .runner import (
                _default_output_converter,
                _evaluator_conversion_failure_reason,
            )

            snapshot = MethodOutputSnapshot(
                method_id=entry.method_id,
                source_dataset_sha256=prepared.binding.dataset_sha256,
                output_scale=entry.native_output_scale,
                obs_ids=prepared.audit.retained_cell_ids,
                var_ids=prepared.method_input.var_ids,
                shape=prepared.method_input.shape,
                matrix_sha256=str(run["native_output_sha256"]),
                _matrix_bytes=native_raw,
            )
            execution = AdapterExecution(
                snapshot=snapshot,
                compatibility_log=(),
                environment_receipt=(),
                stdout=stdout,
                stderr=stderr,
                command=None,
            )
            try:
                converted_output = np.asarray(
                    _default_output_converter(prepared.method_input, execution),
                    dtype=np.float64,
                )
                expected_status = "completed"
                expected_reason = None
            except (
                AdapterUnavailableError,
                TypeError,
                ValueError,
                OverflowError,
            ) as error:
                expected_status = "unavailable"
                expected_reason = _evaluator_conversion_failure_reason(error)
            del native_output
        if run.get("status") != expected_status or run.get("reason") != expected_reason:
            raise ScalingContractError(
                "stored scaling status differs from executor replay"
            )
        evaluator_output = self._decode_evaluator_output(
            run,
            entry,
            prepared,
            artifact_directory=artifact_directory,
        )
        if converted_output is not None and (
            evaluator_output is None
            or not np.array_equal(converted_output, evaluator_output)
        ):
            raise ScalingContractError(
                "scaling evaluator output differs from native conversion replay"
            )
        if (
            len(metrics) != len(_SCALING_ACCURACY_METRICS)
            or tuple(
                metric.get("metric") if isinstance(metric, dict) else None
                for metric in metrics
            )
            != _SCALING_ACCURACY_METRICS
            or any(
                not isinstance(metric, dict)
                or metric.get("method") != entry.method_id
                or metric.get("model_seed") != entry.model_seed
                or (
                    metric.get("configuration_sha256") != entry.configuration_sha256
                    if entry.comparator_configuration is None
                    else not direct_equal(
                        metric.get("comparator_configuration"),
                        direct_bound_comparator_value(entry.comparator_configuration),
                    )
                    or metric.get("comparator_nonexecution_identity") is not None
                )
                for metric in metrics
            )
        ):
            raise ScalingContractError("scaling metric denominator is incomplete")
        expected_metric_identity = {
            "mechanism": self.contract.mechanism,
            "biological_id": "draw-01",
            "technical_view": self.contract.technical_view,
            "dataset_id": dataset_receipt["dataset_id"],
            "method": entry.method_id,
            "model_seed": entry.model_seed,
            "configuration_id": entry.configuration_id,
        }
        if entry.comparator_configuration is None:
            expected_metric_identity["configuration_sha256"] = (
                entry.configuration_sha256
            )
        else:
            expected_metric_identity["comparator_configuration"] = (
                direct_bound_comparator_value(entry.comparator_configuration)
            )
            expected_metric_identity["comparator_nonexecution_identity"] = None
        expected_n = metric_authority
        correlation_pairs = int(metric_authority["correlation_pairs"])
        gene_count = int(metric_authority["n_corr_genes"])
        for metric in metrics:
            if any(
                not direct_equal(metric.get(name), expected)
                for name, expected in expected_metric_identity.items()
            ):
                raise ScalingContractError(
                    "scaling metric dataset or configuration identity differs"
                )
            metric_name = str(metric["metric"])
            value_number = metric.get("value")
            metric_n = metric.get("n")
            metric_status = metric.get("status")
            metric_reason = metric.get("reason")
            if run["status"] != "completed":
                if (
                    value_number is not None
                    or metric_n != 0
                    or metric_status != "unavailable"
                    or metric_reason != run["reason"]
                ):
                    raise ScalingContractError(
                        "noncompleted scaling metric does not preserve its reason"
                    )
                continue
            if value_number is None:
                if (
                    metric_status != "unavailable"
                    or not isinstance(metric_reason, str)
                    or not metric_reason
                    or type(metric_n) is not int
                    or metric_n < 0
                ):
                    raise ScalingContractError(
                        "unavailable scaling metric evidence is invalid"
                    )
                if metric_name != "corr_err" or metric_n not in {
                    gene_count,
                    correlation_pairs,
                }:
                    raise ScalingContractError(
                        "unavailable scaling metric denominator differs"
                    )
                continue
            metric_value = _finite_real(
                value_number,
                "completed scaling metric value",
            )
            if (
                metric_value < 0.0
                or metric_status != "completed"
                or metric_reason is not None
                or type(metric_n) is not int
                or metric_n < 0
            ):
                raise ScalingContractError("completed scaling metric is invalid")
            required_n = (
                correlation_pairs
                if metric_name == "corr_err"
                else expected_n[metric_name]
            )
            if metric_n != required_n:
                raise ScalingContractError(
                    "completed scaling metric denominator differs"
                )
            if metric_name == "corr_err" and metric_value > 2.0:
                raise ScalingContractError(
                    "scaling correlation distortion exceeds its range"
                )
            if metric_name == "n_corr_genes" and metric_value != gene_count:
                raise ScalingContractError("scaling correlation gene count differs")
        if evaluator_output is not None:
            from .runner import _evaluator_targets

            observed, truth, truth_kind, _marker_mask = _evaluator_targets(prepared)
            if truth_kind != "exact_pre_capture" or truth is None:
                raise ScalingContractError(
                    "scaling metric replay lacks exact pre-capture truth"
                )
            replayed = [
                metric.to_dict()
                for metric in _scaling_metric_rows(
                    entry,
                    _run_plan_entry(entry, prepared.binding),
                    _bounded_scaling_metric_values(evaluator_output, observed, truth),
                )
            ]
            if metrics != replayed:
                raise ScalingContractError(
                    "stored scaling metrics differ from evaluator output replay"
                )
        self._validate_run_directory_closed(
            run, entry, artifact_directory=artifact_directory
        )
        return MappingProxyType(dict(value))

    def load(self, *, force_validate: bool = False) -> ScalingCheckpoint | None:
        with self._exclusive_operation():
            return self._load_unlocked(force_validate=force_validate)

    def _load_unlocked(
        self, *, force_validate: bool = False
    ) -> ScalingCheckpoint | None:
        if type(force_validate) is not bool:
            raise TypeError("force_validate must be bool")
        if not force_validate and self._snapshot is not _UNLOADED:
            if self._snapshot is None:
                return None
            assert isinstance(self._snapshot, ScalingCheckpoint)
            return self._detached_checkpoint(self._snapshot)
        self._snapshot = _UNLOADED
        self._checkpoint_file_sha256 = _UNLOADED
        self._prepared_datasets.clear()
        self._metric_authorities.clear()
        raw = self._checkpoint_raw()
        if raw is None:
            self._snapshot = None
            self._checkpoint_file_sha256 = None
            return None
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except ScalingContractError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScalingContractError(
                f"scaling checkpoint cannot be read: {error}"
            ) from error
        if (
            not isinstance(payload, dict)
            or set(payload) != _CHECKPOINT_KEYS
            or raw != _canonical_bytes(payload) + b"\n"
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
            or payload.get("plan_sha256") != self.plan.plan_sha256
            or payload.get("input_hashes") != dict(self.plan.input_hashes)
            or payload.get("planned_run_count") != len(self.plan.entries)
        ):
            raise ScalingContractError("scaling checkpoint authority mismatch")
        expected_digest = canonical_sha256(
            {key: item for key, item in payload.items() if key != "checkpoint_sha256"}
        )
        if payload.get("checkpoint_sha256") != expected_digest:
            raise ScalingContractError("scaling checkpoint checksum mismatch")
        datasets = payload.get("datasets")
        records = payload.get("records")
        if not isinstance(datasets, list) or not isinstance(records, list):
            raise ScalingContractError("scaling checkpoint arrays are invalid")
        if len(self._checkpoint_paths()) != len(datasets) + len(records):
            raise ScalingContractError(
                "scaling checkpoint history differs from its prefix length"
            )
        expected_cells = tuple(
            dict.fromkeys(entry.cells for entry in self.plan.entries)
        )
        if len(datasets) > len(expected_cells) or len(records) > len(self.plan.entries):
            raise ScalingContractError("scaling checkpoint exceeds its denominator")
        dataset_values = tuple(
            self._validate_dataset_receipt(value, cells, verify_generator=True)
            for value, cells in zip(datasets, expected_cells, strict=False)
        )
        receipts_by_cells = {int(value["cells"]): value for value in dataset_values}
        record_values = tuple(
            self._validate_record(value, entry, receipts_by_cells[entry.cells])
            for value, entry in zip(records, self.plan.entries, strict=False)
        )
        if record_values:
            maximum_size_index = expected_cells.index(
                self.plan.entries[len(record_values) - 1].cells
            )
            if len(dataset_values) <= maximum_size_index:
                raise ScalingContractError(
                    "scaling records exist without their dataset receipt"
                )
        expected_status = (
            "completed"
            if len(record_values) == len(self.plan.entries)
            and len(dataset_values) == len(expected_cells)
            else "running"
        )
        if payload.get("status") != expected_status:
            raise ScalingContractError("scaling checkpoint status is inconsistent")
        checkpoint = ScalingCheckpoint(
            schema_version=1,
            plan_sha256=self.plan.plan_sha256,
            input_hashes=MappingProxyType(dict(self.plan.input_hashes)),
            planned_run_count=len(self.plan.entries),
            status=expected_status,
            datasets=dataset_values,
            records=record_values,
            checkpoint_sha256=expected_digest,
        )
        self._snapshot = checkpoint
        self._checkpoint_file_sha256 = self._checkpoint_state(raw)
        return self._detached_checkpoint(checkpoint)

    def _write(
        self,
        datasets: Sequence[Mapping[str, object]],
        records: Sequence[Mapping[str, object]],
    ) -> ScalingCheckpoint:
        self._require_checkpoint_unchanged()
        expected_cells = tuple(
            dict.fromkeys(entry.cells for entry in self.plan.entries)
        )
        status = (
            "completed"
            if len(records) == len(self.plan.entries)
            and len(datasets) == len(expected_cells)
            else "running"
        )
        body: dict[str, object] = {
            "schema_version": 1,
            "plan_sha256": self.plan.plan_sha256,
            "input_hashes": dict(self.plan.input_hashes),
            "planned_run_count": len(self.plan.entries),
            "status": status,
            "datasets": [dict(value) for value in datasets],
            "records": [dict(value) for value in records],
        }
        body["checkpoint_sha256"] = canonical_sha256(body)
        raw = _canonical_bytes(body) + b"\n"
        sequence = len(datasets) + len(records)
        paths = self._checkpoint_paths()
        if sequence != len(paths) + 1:
            raise ScalingContractError(
                "scaling checkpoint history does not match the next prefix"
            )
        checkpoint_root = _ensure_directory(
            self.output_dir,
            Path("checkpoints"),
            "scaling checkpoint root",
        )
        path = checkpoint_root / f"{sequence:08d}.json"
        _atomic_replace(path, raw)
        isolated_datasets = tuple(
            MappingProxyType(json.loads(_canonical_bytes(dict(value)).decode("utf-8")))
            for value in datasets
        )
        isolated_records = tuple(
            MappingProxyType(json.loads(_canonical_bytes(dict(value)).decode("utf-8")))
            for value in records
        )
        checkpoint = ScalingCheckpoint(
            schema_version=1,
            plan_sha256=self.plan.plan_sha256,
            input_hashes=MappingProxyType(dict(self.plan.input_hashes)),
            planned_run_count=len(self.plan.entries),
            status=status,
            datasets=isolated_datasets,
            records=isolated_records,
            checkpoint_sha256=str(body["checkpoint_sha256"]),
        )
        self._snapshot = checkpoint
        self._checkpoint_file_sha256 = self._checkpoint_state(raw)
        detached = self._detached_checkpoint(checkpoint)
        assert detached is not None
        return detached

    def append_dataset(self, receipt: Mapping[str, object]) -> ScalingCheckpoint:
        with self._exclusive_operation():
            return self._append_dataset_unlocked(receipt)

    def _append_dataset_unlocked(
        self, receipt: Mapping[str, object]
    ) -> ScalingCheckpoint:
        report = self._load_unlocked()
        datasets = [] if report is None else list(report.datasets)
        records = [] if report is None else list(report.records)
        expected_cells = tuple(
            dict.fromkeys(entry.cells for entry in self.plan.entries)
        )
        if len(datasets) >= len(expected_cells):
            raise ScalingContractError("all scaling dataset receipts already exist")
        validated = self._validate_dataset_receipt(
            dict(receipt),
            expected_cells[len(datasets)],
            verify_generator=False,
        )
        datasets.append(validated)
        return self._write(datasets, records)

    def _runs_directory(self) -> Path:
        return _ensure_directory(self.output_dir, Path("runs"), "scaling runs root")

    @staticmethod
    def _remove_closed_run_directory(path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ScalingContractError(
                "orphan scaling run directory cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ScalingContractError("orphan scaling run path is invalid")
        allowed = {
            "run.stdout",
            "run.stderr",
            "run.executor-receipt.json",
            "run.native-f64.zlib",
            "run.log2-cp10k-f64.zlib",
        }
        entries = tuple(path.iterdir())
        if any(
            child.name not in allowed
            or stat.S_ISLNK(child.lstat().st_mode)
            or not stat.S_ISREG(child.lstat().st_mode)
            or child.lstat().st_nlink != 1
            for child in entries
        ):
            raise ScalingContractError("orphan scaling run directory is not closed")
        shutil.rmtree(path)

    def recover_unreferenced_transactions(self) -> tuple[str, ...]:
        """Remove only closed run directories absent from the cached checkpoint."""

        with self._exclusive_operation():
            report = self._load_unlocked(force_validate=True)
            self._require_checkpoint_unchanged()
            for receipt in () if report is None else report.datasets:
                _cleanup_discarded_scaling_inputs(
                    self.output_dir,
                    int(receipt["cells"]),
                    receipt,
                )
            referenced = {
                str(record["run"]["run_id"])
                for record in (() if report is None else report.records)
                if isinstance(record, Mapping)
                and isinstance(record.get("run"), Mapping)
            }
            runs = self.output_dir / "runs"
            if not os.path.lexists(runs):
                return ()
            try:
                metadata = runs.lstat()
                children = tuple(sorted(runs.iterdir(), key=lambda path: path.name))
            except OSError as error:
                raise ScalingContractError(
                    "scaling run recovery directory is unavailable"
                ) from error
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ScalingContractError("scaling run recovery path is invalid")
            by_run_id = {entry.run_id: entry for entry in self.plan.entries}
            removed: list[str] = []
            for path in children:
                if path.name in referenced:
                    continue
                run_id: str | None = None
                if path.name in by_run_id:
                    run_id = path.name
                else:
                    for candidate in by_run_id:
                        if path.name.startswith(
                            f".{candidate}."
                        ) and path.name.endswith(".tmp"):
                            run_id = candidate
                            break
                if run_id is None or run_id in referenced:
                    raise ScalingContractError(
                        "unreferenced scaling run directory is not plan-owned"
                    )
                self._require_checkpoint_unchanged()
                self._remove_closed_run_directory(path)
                if run_id not in removed:
                    removed.append(run_id)
            if removed:
                descriptor = os.open(runs, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return tuple(removed)

    def _prepare_run_transaction(self, entry: ScalingPlanEntry) -> tuple[Path, Path]:
        self._require_checkpoint_unchanged()
        if self._snapshot is _UNLOADED:
            raise ScalingContractError(
                "scaling checkpoint cache is not initialized for recovery"
            )
        snapshot = self._snapshot
        if isinstance(snapshot, ScalingCheckpoint) and any(
            record.get("run", {}).get("run_id") == entry.run_id
            for record in snapshot.records
            if isinstance(record, Mapping)
        ):
            raise ScalingContractError(
                "referenced scaling run directory cannot be recovered"
            )
        runs = self._runs_directory()
        final = runs / entry.run_id
        _reject_symlink_components(self.output_dir, final, "scaling run transaction")
        if os.path.lexists(final):
            self._remove_closed_run_directory(final)
        prefix = f".{entry.run_id}."
        for child in tuple(runs.iterdir()):
            if child.name.startswith(prefix) and child.name.endswith(".tmp"):
                self._remove_closed_run_directory(child)
        try:
            stage = Path(tempfile.mkdtemp(prefix=prefix, suffix=".tmp", dir=runs))
        except OSError as error:
            raise ScalingContractError(
                "scaling run transaction cannot be staged"
            ) from error
        _reject_symlink_components(self.output_dir, stage, "scaling run transaction")
        return stage, final

    @staticmethod
    def _write_run_file(directory: Path, name: str, raw: bytes) -> None:
        path = directory / name
        try:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
            try:
                remaining = memoryview(raw)
                while remaining:
                    written = os.write(descriptor, remaining)
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError as error:
            raise ScalingContractError(
                f"scaling run artifact {name} cannot be staged"
            ) from error

    def _publish_run_transaction(self, stage: Path, final: Path) -> None:
        try:
            descriptor = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if os.path.lexists(final):
                raise ScalingContractError("scaling run transaction already exists")
            os.rename(stage, final)
            parent = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except ScalingContractError:
            raise
        except OSError as error:
            raise ScalingContractError(
                "scaling run transaction could not be published"
            ) from error

    def append_attempt(
        self, entry: ScalingPlanEntry, record: Mapping[str, object]
    ) -> ScalingCheckpoint:
        with self._exclusive_operation():
            return self._append_attempt_unlocked(entry, record)

    def _append_attempt_unlocked(
        self, entry: ScalingPlanEntry, record: Mapping[str, object]
    ) -> ScalingCheckpoint:
        if not isinstance(entry, ScalingPlanEntry):
            raise TypeError("entry must be a ScalingPlanEntry")
        report = self._load_unlocked()
        datasets = [] if report is None else list(report.datasets)
        records = [] if report is None else list(report.records)
        if (
            len(records) >= len(self.plan.entries)
            or self.plan.entries[len(records)] != entry
        ):
            raise ScalingContractError(
                "scaling attempts must follow the exact plan prefix"
            )
        self._require_checkpoint_unchanged()
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        value = dict(record)
        stdout = value.pop("stdout", None)
        stderr = value.pop("stderr", None)
        native_output = value.pop("native_output", None)
        evaluator_output = value.pop("evaluator_output", None)
        executor_receipt_raw = value.pop("executor_receipt", None)
        if set(value) != {"run", "metrics"}:
            raise ScalingContractError("scaling attempt fields are not closed")
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise ScalingContractError("scaling logs must be exact bytes")
        if len(stdout) > _MAX_LOG_BYTES or len(stderr) > _MAX_LOG_BYTES:
            raise ScalingContractError("scaling log exceeds its 64 MiB bound")
        run = value.get("run")
        if not isinstance(run, dict):
            raise ScalingContractError("scaling attempt run is invalid")
        if (
            run.get("stdout_sha256") != hashlib.sha256(stdout).hexdigest()
            or run.get("stderr_sha256") != hashlib.sha256(stderr).hexdigest()
        ):
            raise ScalingContractError("scaling log content hash mismatch")
        if type(executor_receipt_raw) is not bytes:
            raise ScalingContractError("scaling executor receipt must be exact bytes")
        executor_receipt = _parse_executor_receipt(executor_receipt_raw)
        native_storage: dict[str, object] = {
            "native_output_path": None,
            "native_output_file_sha256": None,
            "native_output_shape": None,
            "native_output_dtype": None,
            "native_output_scale": None,
            "native_output_encoding": None,
            "native_output_compressed_nbytes": None,
            "native_output_uncompressed_nbytes": None,
            "native_output_uncompressed_sha256": None,
        }
        compressed_native: bytes | None = None
        if native_output is not None:
            try:
                native = np.asarray(native_output, dtype="<f8", order="C")
            except (TypeError, ValueError, OverflowError) as error:
                raise ScalingContractError(
                    "scaling native output is invalid"
                ) from error
            expected_attempt_shape = (
                run.get("retained_cell_count"),
                run.get("retained_gene_count"),
            )
            if (
                executor_receipt.get("status") != "completed"
                or not all(
                    type(item) is int and item > 0 for item in expected_attempt_shape
                )
                or native.shape != expected_attempt_shape
                or not np.isfinite(native).all()
                or bool((native < 0).any())
            ):
                raise ScalingContractError("scaling native output is invalid")
            raw_native = native.tobytes(order="C")
            if len(raw_native) > _MAX_NATIVE_OUTPUT_BYTES:
                raise ScalingContractError("scaling native output exceeds its bound")
            compressed_native = zlib.compress(
                raw_native, level=_EVALUATOR_OUTPUT_COMPRESSION_LEVEL
            )
            native_storage.update(
                {
                    "native_output_shape": list(native.shape),
                    "native_output_dtype": "<f8",
                    "native_output_scale": entry.native_output_scale,
                    "native_output_encoding": _NATIVE_OUTPUT_ENCODING,
                    "native_output_compressed_nbytes": len(compressed_native),
                    "native_output_uncompressed_nbytes": len(raw_native),
                    "native_output_uncompressed_sha256": hashlib.sha256(
                        raw_native
                    ).hexdigest(),
                }
            )
        elif run.get("native_output_sha256") is not None:
            raise ScalingContractError(
                "scaling native output bytes are missing from the attempt"
            )
        evaluator_storage: dict[str, object] = {
            "evaluator_output_path": None,
            "evaluator_output_file_sha256": None,
            "evaluator_output_shape": None,
            "evaluator_output_dtype": None,
            "evaluator_output_scale": None,
            "evaluator_output_encoding": None,
            "evaluator_output_compressed_nbytes": None,
            "evaluator_output_uncompressed_nbytes": None,
            "evaluator_output_uncompressed_sha256": None,
        }
        compressed_evaluator: bytes | None = None
        if evaluator_output is not None:
            try:
                evaluator = np.asarray(evaluator_output, dtype="<f8", order="C")
            except (TypeError, ValueError, OverflowError) as error:
                raise ScalingContractError(
                    "scaling evaluator output is invalid"
                ) from error
            expected_shape = (
                run.get("retained_cell_count"),
                run.get("retained_gene_count"),
            )
            if (
                run.get("status") != "completed"
                or not all(type(item) is int and item > 0 for item in expected_shape)
                or evaluator.shape != expected_shape
                or not np.isfinite(evaluator).all()
                or bool((evaluator < 0).any())
            ):
                raise ScalingContractError("scaling evaluator output is invalid")
            raw_evaluator = evaluator.tobytes(order="C")
            if len(raw_evaluator) > _MAX_EVALUATOR_OUTPUT_BYTES:
                raise ScalingContractError("scaling evaluator output exceeds its bound")
            compressed_evaluator = zlib.compress(
                raw_evaluator, level=_EVALUATOR_OUTPUT_COMPRESSION_LEVEL
            )
            evaluator_storage.update(
                {
                    "evaluator_output_shape": list(evaluator.shape),
                    "evaluator_output_dtype": "<f8",
                    "evaluator_output_scale": _EVALUATOR_OUTPUT_SCALE,
                    "evaluator_output_encoding": _EVALUATOR_OUTPUT_ENCODING,
                    "evaluator_output_compressed_nbytes": len(compressed_evaluator),
                    "evaluator_output_uncompressed_nbytes": len(raw_evaluator),
                    "evaluator_output_uncompressed_sha256": hashlib.sha256(
                        raw_evaluator
                    ).hexdigest(),
                }
            )
        elif run.get("evaluator_output_sha256") is not None:
            raise ScalingContractError(
                "scaling evaluator output bytes are missing from the attempt"
            )
        base = f"runs/{entry.run_id}"
        stdout_relative = f"{base}/run.stdout"
        stderr_relative = f"{base}/run.stderr"
        executor_relative = f"{base}/run.executor-receipt.json"
        if compressed_native is not None:
            native_relative = f"{base}/run.native-f64.zlib"
            native_storage.update(
                {
                    "native_output_path": native_relative,
                    "native_output_file_sha256": hashlib.sha256(
                        compressed_native
                    ).hexdigest(),
                }
            )
        if compressed_evaluator is not None:
            evaluator_relative = f"{base}/run.log2-cp10k-f64.zlib"
            evaluator_storage.update(
                {
                    "evaluator_output_path": evaluator_relative,
                    "evaluator_output_file_sha256": hashlib.sha256(
                        compressed_evaluator
                    ).hexdigest(),
                }
            )
        stored_run = dict(run)
        stored_run.update(
            {
                "stdout_path": stdout_relative,
                "stdout_file_sha256": hashlib.sha256(stdout).hexdigest(),
                "stdout_size_bytes": len(stdout),
                "stderr_path": stderr_relative,
                "stderr_file_sha256": hashlib.sha256(stderr).hexdigest(),
                "stderr_size_bytes": len(stderr),
                "executor_receipt_path": executor_relative,
                "executor_receipt_file_sha256": hashlib.sha256(
                    executor_receipt_raw
                ).hexdigest(),
                "executor_receipt_size_bytes": len(executor_receipt_raw),
                "executor_receipt_sha256": executor_receipt["receipt_sha256"],
                **native_storage,
                **evaluator_storage,
            }
        )
        unsigned = {"run": stored_run, "metrics": value.get("metrics")}
        stored = {**unsigned, "record_sha256": canonical_sha256(unsigned)}
        receipt = next(
            (value for value in datasets if value.get("cells") == entry.cells),
            None,
        )
        if receipt is None:
            raise ScalingContractError("scaling attempt lacks its dataset receipt")
        stage, final = self._prepare_run_transaction(entry)
        published = False
        try:
            self._write_run_file(stage, "run.stdout", stdout)
            self._write_run_file(stage, "run.stderr", stderr)
            self._write_run_file(
                stage, "run.executor-receipt.json", executor_receipt_raw
            )
            if compressed_native is not None:
                self._write_run_file(stage, "run.native-f64.zlib", compressed_native)
            if compressed_evaluator is not None:
                self._write_run_file(
                    stage, "run.log2-cp10k-f64.zlib", compressed_evaluator
                )
            self._validate_record(
                stored,
                entry,
                receipt,
                artifact_directory=stage,
            )
            self._publish_run_transaction(stage, final)
            published = True
            records.append(stored)
            return self._write(datasets, records)
        finally:
            if not published and stage.exists():
                shutil.rmtree(stage)


def _dataset_receipt_from_artifacts(
    contract: ScalingContract,
    protocol: Protocol,
    output_dir: Path,
    artifacts: Sequence[SimulationArtifact],
) -> tuple[dict[str, object], Any]:
    values = tuple(artifacts)
    if len(values) != 2 or not all(
        isinstance(value, SimulationArtifact) for value in values
    ):
        raise ScalingContractError(
            "SymSim scaling did not return exactly two artifacts"
        )
    by_view = {value.request.technical_view: value for value in values}
    if set(by_view) != {"moderate", "severe"}:
        raise ScalingContractError("SymSim scaling artifact views are incomplete")
    moderate_artifact = by_view["moderate"]
    severe_artifact = by_view["severe"]
    moderate = moderate_artifact.adata
    severe = severe_artifact.adata
    from .datasets import _truth_sha256
    from .schema import benchmark_dataset_sha256

    if (
        benchmark_dataset_sha256(moderate) != moderate_artifact.dataset_sha256
        or benchmark_dataset_sha256(severe) != severe_artifact.dataset_sha256
    ):
        raise ScalingContractError("scaling dataset semantics changed after simulation")
    moderate_truth = _truth_sha256(moderate)
    severe_truth = _truth_sha256(severe)
    if moderate_truth != severe_truth:
        raise ScalingContractError("paired scaling views do not share exact truth")
    moderate_manifest = moderate_artifact.native_manifest
    severe_manifest = severe_artifact.native_manifest
    moderate_native = moderate_manifest.as_dict()
    severe_native = severe_manifest.as_dict()
    if moderate_native["files"] != severe_native["files"]:
        raise ScalingContractError("paired scaling native file inventories differ")
    moderate_path = moderate_artifact.request.output_path.absolute()
    severe_path = severe_artifact.request.output_path.absolute()
    try:
        moderate_relative = moderate_path.relative_to(output_dir.absolute()).as_posix()
        severe_path.relative_to(output_dir.absolute())
    except ValueError as error:
        raise ScalingContractError("scaling output escaped its result root") from error
    seeds = derive_scaling_seeds(contract, protocol.development.cells)
    request = moderate_artifact.request
    design_sha256 = _scaling_dataset_design_sha256(contract, protocol, request, seeds)
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "cells": request.cells,
        "genes": request.genes,
        "namespace": request.namespace,
        "mechanism": request.mechanism,
        "technical_view": "moderate",
        "dataset_id": request.dataset_id,
        "independent_unit_id": request.independent_unit_id,
        "dataset_sha256": moderate_artifact.dataset_sha256,
        "truth_sha256": moderate_truth,
        "moderate_output_path": moderate_relative,
        "moderate_output_file_sha256": _file_sha256(moderate_path),
        "moderate_output_size_bytes": moderate_path.stat().st_size,
        "severe_dataset_sha256": severe_artifact.dataset_sha256,
        "severe_output_file_sha256": _file_sha256(severe_path),
        "severe_output_size_bytes": severe_path.stat().st_size,
        "moderate_native_manifest_sha256": moderate_manifest.manifest_sha256,
        "severe_native_manifest_sha256": severe_manifest.manifest_sha256,
        "native_files_sha256": canonical_sha256(moderate_native["files"]),
        "protocol_sha256": canonical_sha256(asdict(protocol)),
        "design_sha256": design_sha256,
        "seed_source_sha256": contract.file_sha256,
        "seeds": asdict(seeds),
        "severe_retention": "discarded_after_receipt",
        "native_retention": "discarded_after_receipt",
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}, moderate


def _dataset_binding(receipt: Mapping[str, object]) -> DatasetBinding:
    return DatasetBinding(
        mechanism=str(receipt["mechanism"]),
        biological_id="draw-01",
        technical_view=str(receipt["technical_view"]),
        dataset_id=str(receipt["dataset_id"]),
        dataset_sha256=str(receipt["dataset_sha256"]),
        output_file_sha256=str(receipt["moderate_output_file_sha256"]),
        truth_sha256=str(receipt["truth_sha256"]),
        output_path=str(receipt["moderate_output_path"]),
        independent_unit_id=str(receipt["independent_unit_id"]),
        cells=int(receipt["cells"]),
        genes=int(receipt["genes"]),
        manifest_sha256=str(receipt["receipt_sha256"]),
        protocol_sha256=str(receipt["protocol_sha256"]),
        design_sha256=str(receipt["design_sha256"]),
        seed_source_sha256=str(receipt["seed_source_sha256"]),
    )


def _load_scaling_dataset(
    store: ScalingResultStore,
    receipt: Mapping[str, object],
    authority: RunnerAuthority,
) -> PreparedDataset:
    cached = store._prepared_datasets.get(int(receipt["cells"]))
    if cached is not None and cached[:2] == (
        receipt["receipt_sha256"],
        receipt["moderate_output_file_sha256"],
    ):
        return cached[2]
    import anndata as ad

    path = store._artifact_path(
        receipt["moderate_output_path"], "moderate scaling dataset"
    )
    try:
        dataset = ad.read_h5ad(path)
    except (OSError, TypeError, ValueError) as error:
        raise ScalingContractError(
            "moderate scaling dataset cannot be loaded"
        ) from error
    from .datasets import _truth_sha256
    from .schema import benchmark_dataset_sha256

    if (
        benchmark_dataset_sha256(dataset) != receipt["dataset_sha256"]
        or _truth_sha256(dataset) != receipt["truth_sha256"]
    ):
        raise ScalingContractError("moderate scaling dataset semantic hash mismatch")
    try:
        return prepare_dataset_for_execution(
            dataset, _dataset_binding(receipt), authority.dataset_qc_policy
        )
    except Exception as error:
        raise ScalingContractError("moderate scaling dataset QC failed") from error


def _require_scaling_disk_capacity(output_dir: Path, cells: int, genes: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Three native matrices, two H5AD views, staging copies, and a 2 GiB reserve.
    required = cells * genes * 96 + 2 * 1024**3
    available = shutil.disk_usage(output_dir).free
    if available < required:
        raise ScalingContractError(
            "insufficient disk for bounded scaling materialization: "
            f"required={required}, available={available}"
        )


def _cleanup_discarded_scaling_inputs(
    output_dir: Path, cells: int, receipt: Mapping[str, object]
) -> None:
    namespace_root = (
        output_dir / "generated" / f"scaling-{cells}" / "dataset"
    ).absolute()
    severe = namespace_root / "severe.h5ad"
    if severe.exists():
        if (
            severe.is_symlink()
            or not severe.is_file()
            or severe.stat().st_size != receipt["severe_output_size_bytes"]
            or _file_sha256(severe) != receipt["severe_output_file_sha256"]
        ):
            raise ScalingContractError("discarded severe scaling input changed")
        severe.unlink()
    native = namespace_root / "native"
    if native.exists():
        if native.is_symlink() or not native.is_dir():
            raise ScalingContractError("scaling native output root is invalid")
        content_directories = tuple(native.iterdir())
        if (
            len(content_directories) != 1
            or content_directories[0].is_symlink()
            or not content_directories[0].is_dir()
        ):
            raise ScalingContractError("scaling native content directory is invalid")
        inventory: list[dict[str, object]] = []
        for path in sorted(
            content_directories[0].iterdir(), key=lambda item: item.name
        ):
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ScalingContractError("scaling native inventory is invalid")
            inventory.append(
                {
                    "path": path.name,
                    "sha256": _file_sha256(path),
                    "size_bytes": metadata.st_size,
                }
            )
        if canonical_sha256(inventory) != receipt["native_files_sha256"]:
            raise ScalingContractError(
                "scaling native inventory changed before discard"
            )
        shutil.rmtree(native)
    if namespace_root.exists():
        descriptor = os.open(namespace_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _materialize_scaling_dataset(
    contract: ScalingContract,
    base_protocol: Protocol,
    output_dir: Path,
    cells: int,
    *,
    simulator: Any = run_symsim_pair,
) -> tuple[dict[str, object], Any]:
    _require_scaling_disk_capacity(output_dir, cells, contract.genes)
    protocol = scaling_protocol(base_protocol, contract, cells)
    requests = scaling_requests(contract, protocol, output_dir / "generated")
    try:
        artifacts = simulator(requests, protocol)
    except Exception as error:
        raise ScalingContractError(
            f"SymSim scaling materialization failed at {cells} cells"
        ) from error
    return _dataset_receipt_from_artifacts(contract, protocol, output_dir, artifacts)


def _load_scaling_execution_environment_registry(
    repository: Path,
    registry: MethodRegistry,
) -> ExecutionEnvironmentRegistry:
    """Rebuild the same-input registry while binding cross-scope lock entries."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    selected = repository.resolve(strict=True)
    return ExecutionEnvironmentRegistry.fixed(
        selected,
        runtime_lock_path=selected / "environments/development-runtime.lock.json",
        benchmark_python=Path(sys.executable),
        r_library_paths={"saver": (selected / "artifacts/envs/saver-r/library",)},
        lock_only_environment_ids=derive_lock_only_environment_ids(registry),
    )


def load_scaling_execution_authority(
    repository: Path,
) -> ScalingExecutionAuthority:
    """Validate the frozen candidate, scaling contract, runtime, and denominator."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    canonical_repository = Path(__file__).resolve().parents[1]
    try:
        selected = repository.resolve(strict=True)
    except OSError as error:
        raise ScalingContractError("repository is unavailable") from error
    if selected != canonical_repository:
        raise ScalingContractError("scaling must run from the canonical repository")
    contract_path = selected / "study/scaling_panel.json"
    contract = load_scaling_contract(contract_path)
    protocol = load_protocol(selected / "study/protocol.json")
    from .publication_freeze import validate_frozen_method

    try:
        frozen = validate_frozen_method(selected)
    except Exception as error:
        raise ScalingContractError("frozen method authority is unavailable") from error
    artifact_bindings = frozen.get("artifact_bindings")
    if not isinstance(artifact_bindings, Mapping):
        raise ScalingContractError("frozen method lacks artifact bindings")
    scaling_binding = artifact_bindings.get("scaling_panel")
    if (
        not isinstance(scaling_binding, Mapping)
        or scaling_binding.get("path") != "study/scaling_panel.json"
        or scaling_binding.get("sha256") != contract.file_sha256
    ):
        raise ScalingContractError("frozen method does not bind the scaling contract")
    authority = load_runner_authority()
    if not authority.maskimpute_ready:
        raise ScalingContractError("score/calibration authority is not ready")
    registry_path = selected / "study/methods.json"
    registry = load_method_registry(registry_path)
    from .final_runner import _frozen_method_plan_authority

    try:
        _frozen_rows, all_configurations = _frozen_method_plan_authority(
            frozen, registry
        )
    except Exception as error:
        raise ScalingContractError(
            "frozen scaling method authority is invalid"
        ) from error
    configuration_by_method = {value.method_id: value for value in all_configurations}
    configurations = tuple(
        configuration_by_method[method_id] for method_id in contract.method_ids
    )
    environments = _load_scaling_execution_environment_registry(selected, registry)
    plan = build_scaling_plan(
        contract,
        registry,
        configurations,
        frozen_method_sha256=_sha256(
            frozen.get("payload_sha256"), "frozen method payload checksum"
        ),
        method_registry_file_sha256=_file_sha256(registry_path),
        protocol_file_sha256=_file_sha256(selected / "study/protocol.json"),
        execution_authority_sha256=authority.authority_sha256,
        execution_environment_sha256=environments.registry_sha256,
        implementation_source_sha256=implementation_source_sha256(),
    )
    return ScalingExecutionAuthority(
        repository=selected,
        contract=contract,
        protocol=protocol,
        frozen_method=MappingProxyType(dict(frozen)),
        registry=registry,
        runner_authority=authority,
        environments=environments,
        plan=plan,
    )


def _run_plan_entry(entry: ScalingPlanEntry, binding: DatasetBinding) -> RunPlanEntry:
    return RunPlanEntry(
        ordinal=entry.ordinal,
        run_id=entry.run_id,
        method_id=entry.method_id,
        dataset_id=binding.dataset_id,
        source_dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        model_seed=entry.model_seed,
        configuration_id=entry.configuration_id,
        configuration_sha256=entry.configuration_sha256,
        preflight_status="planned",
        preflight_reason=None,
        configuration_kind=entry.configuration_kind,
        requires_count_score=entry.requires_count_score,
        requires_calibration=entry.requires_calibration,
        comparator_configuration=entry.comparator_configuration,
        comparator_nonexecution_identity=entry.comparator_nonexecution_identity,
    )


def _bounded_scaling_metric_values(
    imputed: Any, observed: Any, truth: Any
) -> dict[str, tuple[float | None, int, str | None]]:
    """Compute only cell-linear/gene-correlation accuracy at scaling sizes."""

    try:
        imputed_array = np.asarray(imputed, dtype=np.float64)
        observed_array = np.asarray(observed, dtype=np.float64)
        truth_array = np.asarray(truth, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScalingContractError("scaling accuracy matrices are invalid") from error
    if (
        imputed_array.ndim != 2
        or imputed_array.shape != observed_array.shape
        or imputed_array.shape != truth_array.shape
        or imputed_array.size == 0
        or not np.isfinite(imputed_array).all()
        or not np.isfinite(observed_array).all()
        or not np.isfinite(truth_array).all()
    ):
        raise ScalingContractError("scaling accuracy matrices are invalid")
    difference = imputed_array - truth_array
    masks = {
        "overall": np.ones(truth_array.shape, dtype=bool),
        "dropout": (observed_array == 0) & (truth_array > 0),
        "pre_dropout_zero": truth_array == 0,
        "nonzero": (observed_array > 0) & (truth_array > 0),
    }

    def squared(mask: np.ndarray) -> tuple[float | None, int, str | None]:
        n = int(mask.sum())
        if n == 0:
            return None, 0, "no_entries"
        selected = difference[mask]
        return float(np.mean(selected * selected)), n, None

    result = {
        "mse": squared(masks["overall"]),
        "mse_dropout": squared(masks["dropout"]),
        "mse_pre_dropout_zero": squared(masks["pre_dropout_zero"]),
        "mse_nonzero": squared(masks["nonzero"]),
    }
    truth_sd = np.maximum(np.std(truth_array, axis=0, ddof=0), 1e-8)
    rmse = np.sqrt(np.mean(difference * difference, axis=0))
    result["gnrmse"] = (float(np.mean(rmse / truth_sd)), truth_array.shape[1], None)
    result["mean_distortion"] = (
        float(
            np.mean(
                np.abs(np.mean(imputed_array, axis=0) - np.mean(truth_array, axis=0))
            )
        ),
        truth_array.shape[1],
        None,
    )
    result["variance_distortion"] = (
        float(
            np.mean(
                np.abs(
                    np.var(imputed_array, axis=0, ddof=0)
                    - np.var(truth_array, axis=0, ddof=0)
                )
            )
        ),
        truth_array.shape[1],
        None,
    )
    sorted_imputed = np.sort(imputed_array, axis=0)
    sorted_truth = np.sort(truth_array, axis=0)
    result["mean_gene_wasserstein_distance"] = (
        float(np.mean(np.mean(np.abs(sorted_imputed - sorted_truth), axis=0))),
        truth_array.shape[1],
        None,
    )
    del sorted_imputed, sorted_truth
    n_genes = truth_array.shape[1]
    constant = (np.std(imputed_array, axis=0, ddof=0) == 0) | (
        np.std(truth_array, axis=0, ddof=0) == 0
    )
    if n_genes < 2:
        result["corr_err"] = (None, n_genes, "fewer_than_two_variable_genes")
    elif np.any(constant):
        result["corr_err"] = (None, n_genes, "constant_gene_profile")
    else:
        imputed_correlation = np.corrcoef(imputed_array, rowvar=False)
        truth_correlation = np.corrcoef(truth_array, rowvar=False)
        upper = np.triu_indices(n_genes, k=1)
        distortion = np.abs(imputed_correlation[upper] - truth_correlation[upper])
        if np.isfinite(distortion).all():
            result["corr_err"] = (float(np.mean(distortion)), len(distortion), None)
        else:
            result["corr_err"] = (
                None,
                len(distortion),
                "nonfinite_correlation",
            )
    result["n_corr_genes"] = (float(n_genes), n_genes, None)
    if tuple(result) != _SCALING_ACCURACY_METRICS:
        raise ScalingContractError("bounded scaling metric order drifted")
    return result


def _scaling_metric_rows(
    entry: ScalingPlanEntry,
    run_entry: RunPlanEntry,
    values: Mapping[str, tuple[float | None, int, str | None]],
) -> tuple[LongFormMetric, ...]:
    return tuple(
        LongFormMetric(
            mechanism=run_entry.mechanism,
            biological_id=run_entry.biological_id,
            technical_view=run_entry.technical_view,
            dataset_id=run_entry.dataset_id,
            method=entry.method_id,
            model_seed=entry.model_seed,
            configuration_id=entry.configuration_id,
            configuration_sha256=entry.configuration_sha256,
            metric=name,
            value=value,
            n=n,
            status="completed" if value is not None else "unavailable",
            reason=metric_reason,
            comparator_configuration=entry.comparator_configuration,
            comparator_nonexecution_identity=entry.comparator_nonexecution_identity,
        )
        for name, (value, n, metric_reason) in values.items()
    )


def _evaluate_scaling_outcome(
    entry: ScalingPlanEntry,
    run_entry: RunPlanEntry,
    prepared: PreparedDataset,
    outcome: AdapterOutcome,
) -> ScalingEvaluatedAttempt:
    """Evaluate with bounded metrics and retain both matrices for replay."""

    from .runner import (
        DatasetQCPolicy,
        RawRunResult,
        _default_output_converter,
        _evaluator_conversion_failure_reason,
        _evaluator_output_sha256,
        _evaluator_targets,
        method_input_sha256,
    )
    from .methods import AdapterUnavailableError

    executor_receipt = _executor_receipt_bytes(entry, run_entry, prepared, outcome)
    status = outcome.status
    reason = outcome.reason
    native_output_sha256: str | None = None
    evaluator_output_sha256: str | None = None
    evaluator_output: np.ndarray | None = None
    values: dict[str, tuple[float | None, int, str | None]]
    if outcome.status == "completed":
        assert outcome.execution is not None
        snapshot = outcome.execution.snapshot
        if (
            snapshot.method_id != entry.method_id
            or snapshot.source_dataset_sha256 != prepared.binding.dataset_sha256
            or snapshot.obs_ids != prepared.audit.retained_cell_ids
            or snapshot.var_ids != prepared.method_input.var_ids
        ):
            raise ScalingContractError(
                "completed scaling snapshot differs from its dataset/method"
            )
        native_output_sha256 = snapshot.matrix_sha256
        try:
            evaluator_output = _default_output_converter(
                prepared.method_input, outcome.execution
            )
            evaluator_output = np.asarray(evaluator_output, dtype=np.float64)
            observed, truth, truth_kind, _marker_mask = _evaluator_targets(prepared)
            if truth_kind != "exact_pre_capture" or truth is None:
                raise ScalingContractError(
                    "scaling accuracy requires exact pre-capture truth"
                )
            values = _bounded_scaling_metric_values(evaluator_output, observed, truth)
            evaluator_output_sha256 = _evaluator_output_sha256(
                run_entry, prepared, evaluator_output
            )
            del observed, truth
        except ScalingContractError:
            raise
        except (AdapterUnavailableError, TypeError, ValueError, OverflowError) as error:
            status = "unavailable"
            reason = _evaluator_conversion_failure_reason(error)
            evaluator_output = None
            values = {name: (None, 0, reason) for name in _SCALING_ACCURACY_METRICS}
    else:
        assert reason is not None
        values = {name: (None, 0, reason) for name in _SCALING_ACCURACY_METRICS}
    metrics = _scaling_metric_rows(entry, run_entry, values)
    calibration = outcome.calibration_fold_receipt
    run = RawRunResult(
        run_id=entry.run_id,
        method_id=entry.method_id,
        dataset_id=run_entry.dataset_id,
        source_dataset_sha256=run_entry.source_dataset_sha256,
        mechanism=run_entry.mechanism,
        biological_id=run_entry.biological_id,
        technical_view=run_entry.technical_view,
        model_seed=entry.model_seed,
        configuration_id=entry.configuration_id,
        configuration_sha256=entry.configuration_sha256,
        configuration_kind=entry.configuration_kind,
        requires_count_score=entry.requires_count_score,
        requires_calibration=entry.requires_calibration,
        method_input_sha256=method_input_sha256(prepared.method_input),
        dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
        excluded_cell_count=prepared.audit.excluded_cell_count,
        excluded_cell_ids_sha256=prepared.audit.excluded_cell_ids_sha256,
        retained_cell_count=prepared.audit.retained_cell_count,
        retained_cell_ids_sha256=prepared.audit.retained_cell_ids_sha256,
        retained_gene_count=prepared.method_input.shape[1],
        observed_zero_count=(
            prepared.method_input.counts.size
            - int(np.count_nonzero(prepared.method_input.counts))
        ),
        status=status,
        reason=reason,
        runtime_seconds=outcome.runtime_seconds,
        peak_rss_bytes=outcome.peak_rss_bytes,
        peak_gpu_bytes=outcome.peak_gpu_bytes,
        rss_measurement=outcome.rss_measurement,
        gpu_measurement=outcome.gpu_measurement,
        calibration_artifact_sha256=(
            None if calibration is None else calibration.calibration_artifact_sha256
        ),
        calibration_context_sha256=(
            None if calibration is None else calibration.calibration_context_sha256
        ),
        calibration_training_manifest_sha256s=(
            () if calibration is None else calibration.training_manifest_sha256s
        ),
        calibration_held_out_manifest_sha256s=(
            () if calibration is None else calibration.held_out_manifest_sha256s
        ),
        calibration_fold_calibrator_sha256=(
            None if calibration is None else calibration.fold_calibrator_sha256
        ),
        stdout_sha256=hashlib.sha256(outcome.stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(outcome.stderr).hexdigest(),
        native_output_sha256=native_output_sha256,
        evaluator_output_sha256=evaluator_output_sha256,
        comparator_configuration=entry.comparator_configuration,
        comparator_nonexecution_identity=entry.comparator_nonexecution_identity,
    )
    return ScalingEvaluatedAttempt(
        run=run,
        metrics=metrics,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        native_output=(
            None if outcome.execution is None else outcome.execution.snapshot.matrix
        ),
        native_output_scale=(
            None
            if outcome.execution is None
            else outcome.execution.snapshot.output_scale
        ),
        evaluator_output=evaluator_output,
        executor_receipt=executor_receipt,
    )


def execute_scaling_plan(
    authority: ScalingExecutionAuthority,
    output_dir: Path,
    *,
    simulator: Any = run_symsim_pair,
    executor: Any | None = None,
    on_checkpoint_published: Callable[[], object] | None = None,
) -> ScalingCheckpoint:
    """Execute/resume every size and method while preserving the full denominator."""

    if not isinstance(authority, ScalingExecutionAuthority):
        raise TypeError("authority must be a ScalingExecutionAuthority")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if on_checkpoint_published is not None and not callable(on_checkpoint_published):
        raise TypeError("on_checkpoint_published must be callable")

    def publish_checkpoint() -> None:
        if on_checkpoint_published is not None:
            on_checkpoint_published()

    store = ScalingResultStore(output_dir, authority.plan, simulator=simulator)
    report = store.load()
    configuration_by_method = {
        value.method_id: value for value in authority.plan.configurations
    }
    owned_executor = executor is None
    selected_executor = (
        SpawnedRepositoryExecutor(
            RepositoryAdapterDispatcher(authority.repository, authority.environments)
        )
        if executor is None
        else executor
    )
    if not callable(selected_executor):
        raise TypeError("executor must be callable")
    expected_cells = tuple(
        dict.fromkeys(entry.cells for entry in authority.plan.entries)
    )
    try:
        for size_index, cells in enumerate(expected_cells):
            assert report is not None or size_index == 0
            records_count = 0 if report is None else len(report.records)
            size_entries = tuple(
                entry for entry in authority.plan.entries if entry.cells == cells
            )
            if records_count >= size_entries[-1].ordinal:
                if report is not None and len(report.datasets) > size_index:
                    _cleanup_discarded_scaling_inputs(
                        store.output_dir, cells, report.datasets[size_index]
                    )
                    publish_checkpoint()
                continue
            if report is None or len(report.datasets) <= size_index:
                receipt, _generated = _materialize_scaling_dataset(
                    authority.contract,
                    authority.protocol,
                    store.output_dir,
                    cells,
                    simulator=simulator,
                )
                report = store.append_dataset(receipt)
                del _generated
                _cleanup_discarded_scaling_inputs(
                    store.output_dir, cells, report.datasets[size_index]
                )
                publish_checkpoint()
            else:
                _cleanup_discarded_scaling_inputs(
                    store.output_dir, cells, report.datasets[size_index]
                )
                publish_checkpoint()
            receipt = report.datasets[size_index]
            prepared = _load_scaling_dataset(store, receipt, authority.runner_authority)
            binding = prepared.binding
            for entry in size_entries:
                assert report is not None
                if len(report.records) >= entry.ordinal:
                    continue
                spec = authority.registry.by_id(entry.method_id)
                configuration = configuration_by_method[entry.method_id]
                if configuration.comparator_configuration is not None:
                    request = FinalComparatorExecutionRequest.create(
                        spec,
                        prepared.method_input,
                        model_seed=entry.model_seed,
                        configuration=configuration.comparator_configuration,
                        authority=authority.runner_authority.execution_context,
                        mechanism=binding.mechanism,
                        biological_id=binding.biological_id,
                        technical_view=binding.technical_view,
                        dataset_id=binding.dataset_id,
                        timeout_seconds=spec.resources.timeout_seconds,
                    )
                else:
                    legacy_configuration = configuration.legacy_configuration
                    if (
                        legacy_configuration is None
                    ):  # pragma: no cover - plan invariant
                        raise AssertionError("scaling legacy authority is missing")
                    request = ExecutionRequest.create(
                        spec,
                        prepared.method_input,
                        model_seed=entry.model_seed,
                        configuration=legacy_configuration,
                        authority=authority.runner_authority.execution_context,
                        mechanism=binding.mechanism,
                        biological_id=binding.biological_id,
                        technical_view=binding.technical_view,
                        dataset_id=binding.dataset_id,
                        timeout_seconds=spec.resources.timeout_seconds,
                        calibration_usage="retained_all_development",
                    )
                outcome = selected_executor(request)
                if not isinstance(outcome, AdapterOutcome):
                    raise ScalingContractError(
                        "scaling executor returned a noncanonical outcome"
                    )
                if isinstance(request, ExecutionRequest):
                    outcome = enforce_calibration_fold_receipt(request, outcome)
                attempt = _evaluate_scaling_outcome(
                    entry, _run_plan_entry(entry, binding), prepared, outcome
                )
                record = scaling_attempt_record(
                    attempt,
                    cells=cells,
                    accuracy_enabled=entry.accuracy_enabled,
                )
                report = store.append_attempt(entry, record)
                publish_checkpoint()
                del attempt, record, outcome
            del prepared
        final = store.load(force_validate=True)
        if final is None or final.status != "completed":
            raise ScalingContractError(
                "scaling execution did not complete its denominator"
            )
        return final
    finally:
        if owned_executor and isinstance(selected_executor, SpawnedRepositoryExecutor):
            selected_executor.close()


def run_scaling_panel(repository: Path, round_dir: Path) -> ScalingCheckpoint:
    """Execute only inside the sole claimed canonical final round."""

    if not isinstance(repository, Path) or not isinstance(round_dir, Path):
        raise TypeError("repository and round_dir must be pathlib.Path values")
    from .simulators.base import load_final_manifest_claim

    try:
        claim = load_final_manifest_claim(repository, round_dir)
    except Exception as error:
        raise ScalingContractError(
            "scaling requires a claimed canonical final round"
        ) from error
    output_dir = claim.round_dir / "results/scaling"
    from .final_runner import _record_incremental_results_if_changed
    from .study import record_incremental_results

    def publish_checkpoint() -> object | None:
        return _record_incremental_results_if_changed(
            repository,
            claim.round_dir,
            record_incremental_results,
        )

    return execute_scaling_plan(
        load_scaling_execution_authority(repository),
        output_dir,
        on_checkpoint_published=publish_checkpoint,
    )


def load_publication_scaling_evidence(
    repository: Path,
    round_dir: Path,
) -> ScalingCheckpoint:
    """Load scaling evidence only through its evaluated-round receipt."""

    if not isinstance(repository, Path) or not isinstance(round_dir, Path):
        raise TypeError("repository and round_dir must be pathlib.Path values")
    from .final_runner import (
        FinalRunnerContractError,
        _canonical_round,
        _scaling_checkpoint_file_bindings,
    )
    from .study import (
        StudyStateError,
        _validate_freeze,
        _validate_registry,
        _validate_result_files,
        _validate_state_record_chain,
        _verify_frozen_repository,
    )

    try:
        selected, destination = _canonical_round(repository, round_dir)
        freeze = _validate_freeze(destination, selected)
        _validate_registry(
            selected,
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
            raise ScalingContractError("evaluated scaling lifecycle lacks its receipt")
        evaluation = receipt.get("result_manifest")
        required_evaluation_fields = {
            "schema_version",
            "status",
            "final_plan_sha256",
            "final_execution_manifest_path",
            "final_execution_manifest_sha256",
            "final_execution_payload_sha256",
            "execution_validation",
            "storage_preflight",
            "scaling_evidence",
            "result_files",
        }
        if (
            not isinstance(evaluation, Mapping)
            or set(evaluation)
            not in {
                frozenset(required_evaluation_fields),
                frozenset(required_evaluation_fields | {"trajectory_evidence"}),
            }
            or type(evaluation.get("schema_version")) is not int
            or evaluation.get("schema_version") != 1
            or evaluation.get("status") != "completed"
            or receipt.get("result_manifest_sha256")
            != canonical_sha256(dict(evaluation))
        ):
            raise ScalingContractError("evaluated scaling receipt manifest is invalid")
        allowed_paths = _validate_result_files(selected, destination, evaluation)
        _verify_frozen_repository(
            selected,
            destination,
            allowed_result_paths=allowed_paths,
        )
    except ScalingContractError:
        raise
    except (FinalRunnerContractError, StudyStateError) as error:
        raise ScalingContractError(
            "evaluated scaling receipt result inventory is invalid"
        ) from error

    evidence = evaluation.get("scaling_evidence")
    evidence_fields = {
        "schema_version",
        "status",
        "plan",
        "checkpoint_path",
        "checkpoint_file_sha256",
        "checkpoint_payload",
        "result_files",
        "evidence_sha256",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != evidence_fields:
        raise ScalingContractError("evaluated scaling evidence schema is invalid")
    evidence_body = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    checkpoint_relative = evidence.get("checkpoint_path")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != 1
        or evidence.get("status") != "completed"
        or not isinstance(checkpoint_relative, str)
        or re.fullmatch(
            r"results/scaling/checkpoints/[0-9]{8}\.json",
            checkpoint_relative,
        )
        is None
        or evidence.get("evidence_sha256") != canonical_sha256(evidence_body)
    ):
        raise ScalingContractError("evaluated scaling evidence binding is invalid")

    authority = load_scaling_execution_authority(selected)
    if evidence.get("plan") != scaling_plan_payload(authority.plan):
        raise ScalingContractError(
            "evaluated scaling plan differs from frozen authority"
        )
    checkpoint_path = destination / checkpoint_relative
    try:
        raw = checkpoint_path.read_bytes()
    except OSError as error:
        raise ScalingContractError(
            "evaluated scaling checkpoint is unavailable"
        ) from error
    if (
        len(raw) > _MAX_CHECKPOINT_BYTES
        or evidence.get("checkpoint_file_sha256") != hashlib.sha256(raw).hexdigest()
        or not isinstance(evidence.get("checkpoint_payload"), Mapping)
        or raw != _canonical_bytes(evidence["checkpoint_payload"]) + b"\n"
    ):
        raise ScalingContractError("evaluated scaling checkpoint binding is invalid")

    global_files = evaluation.get("result_files")
    evidence_files = evidence.get("result_files")
    if not isinstance(global_files, list) or not isinstance(evidence_files, list):
        raise ScalingContractError("evaluated scaling result inventory is invalid")

    def bindings(rows: Sequence[object], *, scaling_only: bool) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
                raise ScalingContractError(
                    "evaluated scaling result inventory is invalid"
                )
            path = row.get("path")
            digest = row.get("sha256")
            if not isinstance(path, str):
                raise ScalingContractError(
                    "evaluated scaling result inventory is invalid"
                )
            if scaling_only and not path.startswith("results/scaling/"):
                continue
            if path in result:
                raise ScalingContractError(
                    "evaluated scaling result inventory is invalid"
                )
            result[path] = _sha256(digest, f"evaluated scaling result {path}")
        return result

    global_scaling = bindings(global_files, scaling_only=True)
    nested_scaling = bindings(evidence_files, scaling_only=False)
    declared_scaling = _scaling_checkpoint_file_bindings(destination)
    if (
        nested_scaling != global_scaling
        or global_scaling != declared_scaling
        or evidence_files
        != [
            {"path": path, "sha256": nested_scaling[path]}
            for path in sorted(nested_scaling)
        ]
    ):
        raise ScalingContractError(
            "evaluated scaling result inventory differs from its receipt"
        )

    checkpoint = ScalingResultStore(
        destination / "results/scaling",
        authority.plan,
    ).load(force_validate=True)
    if (
        checkpoint is None
        or checkpoint.status != "completed"
        or len(checkpoint.records) != checkpoint.planned_run_count
        or scaling_checkpoint_payload(checkpoint)
        != dict(evidence["checkpoint_payload"])
        or checkpoint_relative
        != (
            "results/scaling/checkpoints/"
            f"{len(checkpoint.datasets) + len(checkpoint.records):08d}.json"
        )
    ):
        raise ScalingContractError(
            "evaluated scaling denominator is incomplete or changed"
        )
    try:
        allowed_after = _validate_result_files(selected, destination, evaluation)
        _verify_frozen_repository(
            selected,
            destination,
            allowed_result_paths=allowed_after,
        )
    except StudyStateError as error:
        raise ScalingContractError(
            "evaluated scaling result inventory changed during validation"
        ) from error
    if allowed_after != allowed_paths:
        raise ScalingContractError(
            "evaluated scaling result inventory changed during validation"
        )
    return checkpoint


__all__ = [
    "ScalingCheckpoint",
    "ScalingContract",
    "ScalingContractError",
    "ScalingExecutionAuthority",
    "ScalingEvaluatedAttempt",
    "ScalingPlan",
    "ScalingPlanEntry",
    "ScalingResultStore",
    "ScalingSeeds",
    "build_scaling_plan",
    "derive_scaling_seeds",
    "execute_scaling_plan",
    "load_scaling_execution_authority",
    "load_publication_scaling_evidence",
    "load_scaling_contract",
    "run_scaling_panel",
    "scaling_checkpoint_payload",
    "scaling_plan_payload",
    "scaling_storage_preflight",
    "scaling_attempt_record",
    "scaling_protocol",
    "scaling_requests",
]
