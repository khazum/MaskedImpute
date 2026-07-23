"""Evaluator-owned, truth-isolated development competition runner.

This module provides a data-boundary isolation contract: method adapters receive
only immutable :class:`~maskimpute_benchmark.methods.MethodInput` snapshots.  A
spawned process does not inherit the evaluator's AnnData object.  This is not an
operating-system security sandbox; adapters can still access paths available to
their operating-system account.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field as dataclass_field, replace
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol
import zlib

import numpy as np

if TYPE_CHECKING:
    from .comparator_tuning import BoundComparatorConfiguration
    from .fair_comparator_checkpoint import (
        DirectCheckpointReport,
        DirectCheckpointStore,
    )
    from .fair_comparator_execution import (
        DirectEvaluatedAttempt,
        DirectExecutionRequest,
    )
    from .fair_comparator_plan import DirectCompetitionPlan, DirectPlanEntry
    from .trajectory_dataset import RegisteredTrajectoryBinding

from .comparator_tuning import (
    COMPARATOR_METHOD_IDS,
    DEVELOPMENT_MAX_CHECKPOINT_BYTES,
    DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
    DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
    DEVELOPMENT_MAX_RECORD_BYTES,
    DEVELOPMENT_STORAGE_RESERVE_BYTES,
    ComparatorAdapterConfig,
    ComparatorAuthorityReference,
    ComparatorMethodBinding,
    ComparatorTuningAuthority,
    comparator_method_binding,
    encode_comparator_configuration,
    load_comparator_tuning_authority,
)
from .methods import AdapterExecution, DirectAdapterExecution, MethodInput, MethodSpec
from .methods.registry import MethodRegistry, load_method_registry
from .prezero_evidence import (
    PreZeroEvidence,
    PreZeroEvidenceError,
    encode_prezero_evidence,
    evaluate_prezero_evidence,
    policy_from_score_diagnostics,
    validate_stored_prezero_evidence,
    zlib_compress_bound,
)
from .protocol import canonical_sha256
from .runtime_environments import (
    RuntimeChangeMonitor,
    RuntimeEnvironmentError,
    RuntimeEnvironmentSnapshot,
    load_runtime_environment_lock,
    merge_runtime_environment_snapshots,
    nvidia_smi_executable,
    process_environment_sha256,
    publication_python_spawn_search_path,
    publication_runtime_working_directory,
    runtime_environment_snapshot,
    validate_runtime_environment_lock,
    verify_runtime_environment_control_files,
    verify_runtime_environment_snapshot,
)


DEVELOPMENT_MODEL_SEEDS = (42, 43, 44)
DEVELOPMENT_MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
DEVELOPMENT_VIEWS = ("moderate", "severe")
MAX_DEVELOPMENT_CONFIGURATIONS = 20
MAX_GPU_BUDGET_SECONDS = 8 * 60 * 60
MAX_CPU_BUDGET_SECONDS = 24 * 60 * 60
SELECTION_COMPLETENESS_BLOCKERS = (
    "downstream_safety_not_evaluated",
    "null_de_fpr_not_evaluated",
    "orthogonal_endpoints_not_evaluated",
)
INTRINSIC_TERMINAL_STATUSES = frozenset(
    {"failed", "timeout", "resource_exceeded", "unavailable"}
)
COMPARATOR_SELECTION_BLOCKING_STATUSES = frozenset(
    {"budget_exhausted", "blocked_authority", "infrastructure_error"}
)
_IMPLEMENTATION_SOURCE_DIRECTORIES = ("maskimpute", "maskimpute_benchmark")
_IMPLEMENTATION_SOURCE_FILES = ("scripts/run_development_competition.py",)
_DEVELOPMENT_RUNTIME_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / "environments/development-runtime.lock.json"
)
_TRACKED_V28_REVISION_PATH = (
    Path(__file__).resolve().parents[1] / "study/v28_revision.json"
)
_TRACKED_V28_REVISION_SHA256 = (
    "04fbd61a7ab83e3f1b4b1c8a8d4d5b40b8c6bee39a7f57b50c28b68f12c36705"
)
_TRACKED_V29_REVISION_SHA256 = (
    "8d3f71f5a923b07b6fa489ee9856e0e4598084fdbf7cda77ecaf510068081ba5"
)
_V28_SELECTION_INPUT_PATH = (
    Path(__file__).resolve().parents[1] / "artifacts/study/development/evaluation/"
    "development_selection_input-downstream.json"
)
_V28_SELECTION_REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/study/development/evaluation/development_selection_report.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_EVALUATOR_CONVERSION_REASON = re.compile(
    r"evaluator_conversion_[a-z][a-z0-9_]*_detail_[0-9a-f]{64}\Z"
)
_DEVELOPMENT_TRANSACTION_FILE = re.compile(r"[0-9]{8}\.json\Z")
_MKSTEMP_TOKEN = re.compile(r"[a-z0-9_]{8}\Z")
_OUTCOME_STATUSES = frozenset(
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


class RunnerContractError(RuntimeError):
    """Raised when runner authority, planning, or execution fails closed."""


@dataclass(frozen=True, slots=True)
class DevelopmentStoragePreflight:
    """Pure direct-path worst-case retained-storage receipt."""

    schema: str
    identity_mode: Literal["direct-v1"]
    authority_revision: str
    plan_snapshot: Mapping[str, object]
    prepared_input_descriptors: tuple[Mapping[str, object], ...]
    retained_dimensions: tuple[tuple[str, tuple[int, int]], ...]
    policy: Mapping[str, object]
    planned_run_count: int
    completed_record_count: int
    remaining_executable_count: int
    matrix_bytes: int
    prezero_zlib_bound_bytes: int
    log_receipt_bytes: int
    executor_receipt_bytes: int
    record_bytes: int
    checkpoint_bytes: int
    reserve_bytes: int
    required_free_bytes: int

    def to_dict(self) -> dict[str, object]:
        from .direct_values import direct_json_value

        encoded = direct_json_value(self)
        if not isinstance(encoded, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError(
                "development storage preflight must encode as an object"
            )
        return encoded


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _unlink_owned_staging_temporary(
    path: Path,
    expected_canonical: Path,
    name: str,
) -> None:
    """Remove one exact mkstemp sibling without following or blessing aliases."""

    descriptor = -1
    try:
        parent = path.parent
        if (
            expected_canonical.parent != parent
            or parent.resolve(strict=True) != parent.absolute()
        ):
            raise RunnerContractError(f"{name} parent is not canonical")
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink not in {1, 2}
        ):
            raise RunnerContractError(
                f"{name} must be an owned regular file with one expected sibling"
            )
        sibling_identity: tuple[int, ...] | None = None
        if before.st_nlink == 2:
            try:
                sibling = expected_canonical.lstat()
            except OSError as error:
                raise RunnerContractError(
                    f"{name} hardlink lacks its exact canonical sibling"
                ) from error
            sibling_identity = _stable_stat_identity(sibling)
            if (
                not stat.S_ISREG(sibling.st_mode)
                or stat.S_ISLNK(sibling.st_mode)
                or sibling.st_uid != os.geteuid()
                or sibling_identity != _stable_stat_identity(before)
            ):
                raise RunnerContractError(
                    f"{name} hardlink differs from its exact canonical sibling"
                )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if _stable_stat_identity(before) != _stable_stat_identity(opened):
            raise RunnerContractError(f"{name} changed while opening")
        if sibling_identity is not None:
            if _stable_stat_identity(expected_canonical.lstat()) != sibling_identity:
                raise RunnerContractError(f"{name} canonical sibling changed")
        if _stable_stat_identity(path.lstat()) != _stable_stat_identity(opened):
            raise RunnerContractError(f"{name} changed before removal")
        path.unlink()
        after = os.fstat(descriptor)
        expected_links = before.st_nlink - 1
        if after.st_nlink != expected_links or os.path.lexists(path):
            raise RunnerContractError(f"{name} survived safe removal")
        if sibling_identity is not None:
            sibling_after = expected_canonical.lstat()
            if (
                (sibling_after.st_dev, sibling_after.st_ino)
                != (before.st_dev, before.st_ino)
                or sibling_after.st_nlink != 1
                or sibling_after.st_uid != os.geteuid()
                or not stat.S_ISREG(sibling_after.st_mode)
                or stat.S_ISLNK(sibling_after.st_mode)
            ):
                raise RunnerContractError(
                    f"{name} canonical sibling changed during removal"
                )
        directory = os.open(
            parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except RunnerContractError:
        raise
    except OSError as error:
        raise RunnerContractError(f"{name} could not be removed safely") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class DatasetQCPolicy:
    """The sole evaluator-wide cell exclusion allowed before dispatch."""

    cell_exclusion_rule: str
    minimum_retained_cells: int
    application: str
    additional_cell_filtering: str
    gene_filtering: str
    required_audit_fields: tuple[str, ...]

    @classmethod
    def fixed(cls) -> DatasetQCPolicy:
        return cls(
            cell_exclusion_rule="observed_library_size_equals_zero",
            minimum_retained_cells=2,
            application=(
                "pre_dispatch_pair_union_zero_library_identical_cell_subset_all_methods"
            ),
            additional_cell_filtering="forbidden",
            gene_filtering="forbidden",
            required_audit_fields=(
                "excluded_cell_count",
                "excluded_cell_ids_sha256",
                "retained_cell_count",
                "retained_cell_ids_sha256",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_exclusion_rule": self.cell_exclusion_rule,
            "minimum_retained_cells": self.minimum_retained_cells,
            "application": self.application,
            "additional_cell_filtering": self.additional_cell_filtering,
            "gene_filtering": self.gene_filtering,
            "required_audit_fields": list(self.required_audit_fields),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def require_fixed_publication_rule(self) -> None:
        if self != self.fixed():
            raise RunnerContractError(
                "dataset QC policy differs from the fixed publication rule"
            )


def _require_sha256(value: object, name: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RunnerContractError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonnegative_number(value: object, name: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RunnerContractError(f"{name} must be finite and nonnegative")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_implementation_source(path: Path, relative: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise RunnerContractError(
            f"implementation source is unavailable: {relative}"
        ) from error
    if stat.S_ISLNK(before.st_mode):
        raise RunnerContractError(
            f"implementation source must not be a symlink: {relative}"
        )
    if not stat.S_ISREG(before.st_mode):
        raise RunnerContractError(
            f"implementation source must be a regular file: {relative}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RunnerContractError(
            f"implementation source cannot be opened safely: {relative}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise RunnerContractError(
            f"implementation source changed while hashing: {relative}"
        ) from error

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(opened) or identity(opened) != identity(after):
        raise RunnerContractError(
            f"implementation source changed while hashing: {relative}"
        )
    return b"".join(chunks)


def implementation_source_sha256(repository_root: Path | None = None) -> str:
    """Hash sorted raw execution-source paths and bytes without following links."""

    if repository_root is None:
        repository_root = Path(__file__).resolve().parents[1]
    if not isinstance(repository_root, Path):
        raise TypeError("repository_root must be a pathlib.Path")
    root = repository_root.absolute()
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise RunnerContractError(
            "implementation source root is unavailable"
        ) from error
    if stat.S_ISLNK(root_metadata.st_mode):
        raise RunnerContractError("implementation source root must not be a symlink")
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise RunnerContractError("implementation source root must be a directory")

    paths: list[tuple[str, Path]] = []
    for directory_name in _IMPLEMENTATION_SOURCE_DIRECTORIES:
        directory = root / directory_name
        try:
            directory_metadata = directory.lstat()
        except OSError as error:
            raise RunnerContractError(
                f"implementation source directory is unavailable: {directory_name}"
            ) from error
        if stat.S_ISLNK(directory_metadata.st_mode):
            raise RunnerContractError(
                f"implementation source directory must not be a symlink: {directory_name}"
            )
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise RunnerContractError(
                f"implementation source path must be a directory: {directory_name}"
            )
        for current, directory_names, file_names in os.walk(
            directory, followlinks=False
        ):
            directory_names.sort()
            file_names.sort()
            current_path = Path(current)
            for child_name in directory_names:
                child = current_path / child_name
                try:
                    child_metadata = child.lstat()
                except OSError as error:
                    raise RunnerContractError(
                        "implementation source directory changed while enumerating"
                    ) from error
                if stat.S_ISLNK(child_metadata.st_mode):
                    relative = child.relative_to(root).as_posix()
                    raise RunnerContractError(
                        "implementation source directory must not be a symlink: "
                        f"{relative}"
                    )
            for file_name in file_names:
                if not file_name.endswith(".py"):
                    continue
                path = current_path / file_name
                paths.append((path.relative_to(root).as_posix(), path))
    for relative in _IMPLEMENTATION_SOURCE_FILES:
        path = root / relative
        parent = path.parent
        while parent != root:
            parent_relative = parent.relative_to(root).as_posix()
            try:
                parent_metadata = parent.lstat()
            except OSError as error:
                raise RunnerContractError(
                    f"implementation source directory is unavailable: {parent_relative}"
                ) from error
            if stat.S_ISLNK(parent_metadata.st_mode):
                raise RunnerContractError(
                    "implementation source directory must not be a symlink: "
                    f"{parent_relative}"
                )
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise RunnerContractError(
                    f"implementation source path must be a directory: {parent_relative}"
                )
            parent = parent.parent
        paths.append((relative, path))
    paths.sort(key=lambda item: os.fsencode(item[0]))
    if len({relative for relative, _ in paths}) != len(paths):
        raise RunnerContractError("implementation source paths are not unique")

    digest = hashlib.sha256()
    digest.update(b"maskimpute-implementation-source-v1\0")
    for relative, path in paths:
        relative_bytes = os.fsencode(relative)
        payload = _read_implementation_source(path, relative)
        digest.update(struct.pack("<Q", len(relative_bytes)))
        digest.update(relative_bytes)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _thaw_frozen_json(value: object) -> object:
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {str(item[0]): _thaw_frozen_json(item[1]) for item in value}
        return [_thaw_frozen_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_frozen_json(nested) for key, nested in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class DatasetBinding:
    """One exact completed development dataset bound to its status manifest."""

    mechanism: str
    biological_id: str
    technical_view: str
    dataset_id: str
    dataset_sha256: str
    output_file_sha256: str
    truth_sha256: str
    output_path: str
    independent_unit_id: str
    cells: int
    genes: int
    manifest_sha256: str
    protocol_sha256: str
    design_sha256: str
    seed_source_sha256: str


def validate_development_manifest_payload(
    payload: Mapping[str, object],
) -> tuple[DatasetBinding, ...]:
    """Require the exact validated 4 x 2 x 2 development panel.

    The caller must first obtain ``payload`` from
    :func:`maskimpute_benchmark.datasets.validate_dataset_status`; this helper
    closes the stricter competition cardinality and ordering contract.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("development manifest must be a mapping")
    if (
        payload.get("schema_version") != 1
        or type(payload.get("schema_version")) is not int
    ):
        raise RunnerContractError("development manifest schema_version must be 1")
    if payload.get("namespace") != "dev":
        raise RunnerContractError("development namespace must be exactly dev")
    if (
        payload.get("status") != "completed"
        or payload.get("completed_count") != 16
        or payload.get("failed_count") != 0
    ):
        raise RunnerContractError(
            "development manifest must be complete with no failures"
        )
    if payload.get("independent_unit_count") != 8:
        raise RunnerContractError(
            "development manifest must contain eight independent draws"
        )
    manifest_sha256 = _require_sha256(
        payload.get("manifest_sha256"), "dataset manifest checksum"
    )
    protocol_sha256 = _require_sha256(
        payload.get("protocol_sha256"), "dataset protocol checksum"
    )
    design_sha256 = _require_sha256(
        payload.get("design_sha256"), "dataset design checksum"
    )
    seed_source_sha256 = _require_sha256(
        payload.get("seed_source_sha256"), "dataset seed source checksum"
    )
    assert manifest_sha256 is not None
    assert protocol_sha256 is not None
    assert design_sha256 is not None
    assert seed_source_sha256 is not None
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise RunnerContractError("development manifest must contain exactly 16 rows")
    expected = [
        (mechanism, f"draw-{draw:02d}", view)
        for mechanism in DEVELOPMENT_MECHANISMS
        for draw in (1, 2)
        for view in DEVELOPMENT_VIEWS
    ]
    observed_order: list[tuple[object, object, object]] = []
    bindings: list[DatasetBinding] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RunnerContractError(f"development row {index} must be a mapping")
        observed_order.append(
            (row.get("mechanism"), row.get("biological_id"), row.get("technical_view"))
        )
        if row.get("status") != "completed":
            raise RunnerContractError(f"development row {index} is not completed")
        if row.get("cells") != 900 or row.get("genes") != 500:
            raise RunnerContractError(
                f"development row {index} dimensions differ from 900 x 500"
            )
        dataset_id = row.get("dataset_id")
        independent_unit_id = row.get("independent_unit_id")
        output_path = row.get("output_path")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise RunnerContractError(f"development row {index} dataset_id is invalid")
        if not isinstance(independent_unit_id, str) or not independent_unit_id:
            raise RunnerContractError(
                f"development row {index} independent_unit_id is invalid"
            )
        if not isinstance(output_path, str):
            raise RunnerContractError(f"development row {index} output_path is invalid")
        relative = PurePosixPath(output_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RunnerContractError(f"development row {index} output_path is unsafe")
        if dataset_id in seen_ids or output_path in seen_paths:
            raise RunnerContractError("development rows contain duplicate IDs or paths")
        seen_ids.add(dataset_id)
        seen_paths.add(output_path)
        dataset_sha256 = _require_sha256(
            row.get("dataset_sha256"), f"development row {index} dataset checksum"
        )
        output_file_sha256 = _require_sha256(
            row.get("output_file_sha256"),
            f"development row {index} output checksum",
        )
        truth_sha256 = _require_sha256(
            row.get("truth_sha256"), f"development row {index} truth checksum"
        )
        assert (
            dataset_sha256 is not None
            and output_file_sha256 is not None
            and truth_sha256 is not None
        )
        bindings.append(
            DatasetBinding(
                mechanism=str(row.get("mechanism")),
                biological_id=str(row.get("biological_id")),
                technical_view=str(row.get("technical_view")),
                dataset_id=dataset_id,
                dataset_sha256=dataset_sha256,
                output_file_sha256=output_file_sha256,
                truth_sha256=truth_sha256,
                output_path=output_path,
                independent_unit_id=independent_unit_id,
                cells=900,
                genes=500,
                manifest_sha256=manifest_sha256,
                protocol_sha256=protocol_sha256,
                design_sha256=design_sha256,
                seed_source_sha256=seed_source_sha256,
            )
        )
    if observed_order != expected:
        raise RunnerContractError("development rows are not in canonical order")
    independent_ids = [binding.independent_unit_id for binding in bindings[::2]]
    if len(set(independent_ids)) != 8 or any(
        first.independent_unit_id != second.independent_unit_id
        for first, second in zip(bindings[::2], bindings[1::2], strict=True)
    ):
        raise RunnerContractError("paired views do not form eight independent draws")
    if any(
        first.truth_sha256 != second.truth_sha256
        for first, second in zip(bindings[::2], bindings[1::2], strict=True)
    ):
        raise RunnerContractError("paired views do not share one truth checksum")
    return tuple(bindings)


@dataclass(frozen=True, slots=True)
class AuthorizedConfiguration:
    """One exact tracked registry, search, or ablation execution configuration."""

    method_id: str
    configuration_id: str
    kind: str
    configuration_sha256: str
    payload_json: str
    requires_count_score: bool
    requires_calibration: bool
    registry_method_sha256: str | None = None
    tuning_authority_file_sha256: str | None = None
    tuning_authority_payload_sha256: str | None = None
    source_authority_sha256: str | None = None
    runtime_lock_sha256: str | None = None
    environment_registry_sha256: str | None = None
    configuration_method_identity_sha256: str | None = None
    nonexecution_identity_sha256: str | None = None

    @classmethod
    def create(
        cls,
        *,
        method_id: str,
        configuration_id: str,
        kind: str,
        payload: Mapping[str, object],
        requires_count_score: bool,
        requires_calibration: bool,
        configuration_sha256: str | None = None,
        registry_method_sha256: str | None = None,
        tuning_authority_file_sha256: str | None = None,
        tuning_authority_payload_sha256: str | None = None,
        source_authority_sha256: str | None = None,
        runtime_lock_sha256: str | None = None,
        environment_registry_sha256: str | None = None,
        configuration_method_identity_sha256: str | None = None,
        nonexecution_identity_sha256: str | None = None,
    ) -> AuthorizedConfiguration:
        if kind == "comparator_tuning":
            raise RunnerContractError(
                "legacy comparator configuration is disabled; use the direct fair-comparator path"
            )
        if not isinstance(payload, Mapping):
            raise TypeError("configuration payload must be a mapping")
        payload_bytes = _canonical_bytes(dict(payload))
        parsed = json.loads(payload_bytes.decode("utf-8"))
        digest = canonical_sha256(parsed)
        return cls(
            method_id=method_id,
            configuration_id=configuration_id,
            kind=kind,
            configuration_sha256=(
                digest if configuration_sha256 is None else configuration_sha256
            ),
            payload_json=payload_bytes.decode("utf-8"),
            requires_count_score=requires_count_score,
            requires_calibration=requires_calibration,
            registry_method_sha256=registry_method_sha256,
            tuning_authority_file_sha256=tuning_authority_file_sha256,
            tuning_authority_payload_sha256=tuning_authority_payload_sha256,
            source_authority_sha256=source_authority_sha256,
            runtime_lock_sha256=runtime_lock_sha256,
            environment_registry_sha256=environment_registry_sha256,
            configuration_method_identity_sha256=(configuration_method_identity_sha256),
            nonexecution_identity_sha256=nonexecution_identity_sha256,
        )

    @classmethod
    def from_bound_comparator(
        cls, bound: BoundComparatorConfiguration
    ) -> AuthorizedConfiguration:
        raise RunnerContractError(
            "legacy comparator configuration is disabled; use the direct fair-comparator path"
        )

    @classmethod
    def registry_default(cls, spec: MethodSpec) -> AuthorizedConfiguration:
        if not isinstance(spec, MethodSpec):
            raise TypeError("spec must be a MethodSpec")
        payload = {
            "schema": "maskimpute-registry-default-configuration-v1",
            "method": asdict(spec),
        }
        return cls.create(
            method_id=spec.id,
            configuration_id="registry-default",
            kind="registry",
            payload=payload,
            requires_count_score=False,
            requires_calibration=False,
        )

    def __post_init__(self) -> None:
        if self.kind == "comparator_tuning":
            raise RunnerContractError(
                "legacy comparator configuration is disabled; use the direct fair-comparator path"
            )
        if not isinstance(self.method_id, str) or not _SAFE_ID.fullmatch(
            self.method_id
        ):
            raise RunnerContractError("configuration method_id must be safe")
        if not isinstance(self.configuration_id, str) or not _SAFE_ID.fullmatch(
            self.configuration_id
        ):
            raise RunnerContractError("configuration_id must be safe")
        if self.kind not in {
            "registry",
            "candidate_search",
            "ablation",
            "comparator_tuning",
            "comparator_nonexecution",
        }:
            raise RunnerContractError("configuration kind is invalid")
        _require_sha256(self.configuration_sha256, "configuration checksum")
        try:
            payload = json.loads(
                self.payload_json,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RunnerContractError(
                "configuration payload is invalid JSON"
            ) from error
        if not isinstance(
            payload, dict
        ) or self.payload_json.encode() != _canonical_bytes(payload):
            raise RunnerContractError("configuration payload must be canonical JSON")
        if canonical_sha256(payload) != self.configuration_sha256:
            raise RunnerContractError(
                "configuration checksum does not match its canonical payload"
            )
        if (
            type(self.requires_count_score) is not bool
            or type(self.requires_calibration) is not bool
        ):
            raise RunnerContractError(
                "configuration requirement flags must be booleans"
            )
        if self.requires_calibration and not self.requires_count_score:
            raise RunnerContractError("calibration configurations require count scores")
        component_fields = (
            "registry_method_sha256",
            "tuning_authority_file_sha256",
            "tuning_authority_payload_sha256",
            "source_authority_sha256",
            "runtime_lock_sha256",
            "environment_registry_sha256",
        )
        if self.kind == "comparator_nonexecution":
            if (
                any(
                    getattr(self, field_name) is not None
                    for field_name in component_fields
                )
                or self.configuration_method_identity_sha256 is not None
            ):
                raise RunnerContractError(
                    "comparator nonexecution carries only its nonexecution identity"
                )
            _require_sha256(
                self.nonexecution_identity_sha256,
                "comparator nonexecution identity",
            )
            if self.requires_count_score or self.requires_calibration:
                raise RunnerContractError(
                    "comparator nonexecution configuration is not executable"
                )
        elif (
            any(
                getattr(self, field_name) is not None for field_name in component_fields
            )
            or self.configuration_method_identity_sha256 is not None
            or self.nonexecution_identity_sha256 is not None
        ):
            raise RunnerContractError(
                "legacy configuration forbids comparator identity fields"
            )

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.payload_json))

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "configuration_id": self.configuration_id,
            "kind": self.kind,
            "configuration_sha256": self.configuration_sha256,
            "configuration_payload_sha256": self.configuration_sha256,
            "registry_method_sha256": self.registry_method_sha256,
            "tuning_authority_file_sha256": self.tuning_authority_file_sha256,
            "tuning_authority_payload_sha256": (self.tuning_authority_payload_sha256),
            "source_authority_sha256": self.source_authority_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "environment_registry_sha256": self.environment_registry_sha256,
            "configuration_method_identity_sha256": (
                self.configuration_method_identity_sha256
            ),
            "nonexecution_identity_sha256": self.nonexecution_identity_sha256,
            "payload": dict(self.payload),
            "requires_count_score": self.requires_count_score,
            "requires_calibration": self.requires_calibration,
        }


def derive_authorized_configurations(
    configuration_rows: Sequence[Mapping[str, object]],
    ablation_specs: Sequence[Mapping[str, object]],
    method_bindings: Mapping[str, str],
) -> tuple[AuthorizedConfiguration, ...]:
    """Derive the exact 20-search-config grid plus nonduplicated ablations."""

    if isinstance(configuration_rows, (str, bytes)) or not isinstance(
        configuration_rows, Sequence
    ):
        raise TypeError("configuration_rows must be a sequence")
    search: list[AuthorizedConfiguration] = []
    overrun_seen = False
    for index, row in enumerate(configuration_rows):
        if not isinstance(row, Mapping):
            raise RunnerContractError(f"configuration row {index} must be a mapping")
        disposition = row.get("disposition")
        if disposition == "exploratory_budget_overrun_not_selection_eligible":
            overrun_seen = True
            continue
        if disposition != "authorized":
            raise RunnerContractError(
                f"configuration row {index} disposition is not authorized"
            )
        if overrun_seen:
            raise RunnerContractError(
                "authorized search configuration appears after a budget overrun"
            )
        configuration_id = row.get("configuration_id")
        payload = row.get("configuration")
        configuration_sha256 = row.get("configuration_sha256")
        if not isinstance(configuration_id, str) or not isinstance(payload, Mapping):
            raise RunnerContractError(f"configuration row {index} is malformed")
        _require_sha256(configuration_sha256, f"configuration row {index} checksum")
        score_policy = str(payload.get("score_policy", "")).casefold()
        output_policy = str(payload.get("output_policy", "")).casefold()
        requires_count_score = (
            output_policy == "selective"
            or "score" in score_policy
            or "calibrat" in score_policy
        )
        requires_calibration = "calibrat" in score_policy or "retained" in score_policy
        search.append(
            AuthorizedConfiguration.create(
                method_id="maskimpute",
                configuration_id=configuration_id,
                kind="candidate_search",
                payload=dict(payload),
                requires_count_score=requires_count_score,
                requires_calibration=requires_calibration,
                configuration_sha256=str(configuration_sha256),
            )
        )
    if len(search) != MAX_DEVELOPMENT_CONFIGURATIONS:
        raise RunnerContractError(
            "tracked development search must contain exactly the first 20 authorized configurations"
        )
    search_ids = {value.configuration_id for value in search}
    ablations: list[AuthorizedConfiguration] = []
    for index, raw_spec in enumerate(ablation_specs):
        if not isinstance(raw_spec, Mapping):
            raise RunnerContractError(f"ablation spec {index} must be a mapping")
        spec = dict(raw_spec)
        spec_id = spec.get("id")
        if not isinstance(spec_id, str):
            raise RunnerContractError(f"ablation spec {index} id is invalid")
        if spec_id == "maskimpute-reference" or spec_id in search_ids:
            continue
        try:
            binding = method_bindings[spec_id]
        except KeyError as error:
            raise RunnerContractError(
                f"ablation spec {spec_id} lacks an authority binding"
            ) from error
        _require_sha256(binding, f"ablation spec {spec_id} checksum")
        if canonical_sha256(spec) != binding:
            raise RunnerContractError(
                f"ablation spec {spec_id} checksum mismatches its payload"
            )
        score_source = str(spec.get("score_source", "")).casefold()
        requires_count_score = score_source not in {"", "not_applied"}
        requires_calibration = score_source == "retained_calibrator"
        ablations.append(
            AuthorizedConfiguration.create(
                method_id=(
                    "capacity-matched-ae"
                    if spec_id == "capacity-matched-ae"
                    else "maskimpute"
                ),
                configuration_id=spec_id,
                kind="ablation",
                payload=spec,
                requires_count_score=requires_count_score,
                requires_calibration=requires_calibration,
                configuration_sha256=binding,
            )
        )
    if sum(value.method_id == "capacity-matched-ae" for value in ablations) != 1:
        raise RunnerContractError(
            "ablation authority must contain capacity-matched-ae exactly once"
        )
    return tuple((*search, *ablations))


def _freeze_json_mapping(value: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    def freeze(nested: object) -> object:
        if isinstance(nested, Mapping):
            return tuple((key, freeze(item)) for key, item in sorted(nested.items()))
        if isinstance(nested, (list, tuple)):
            return tuple(freeze(item) for item in nested)
        if nested is None or type(nested) in {str, bool, int}:
            return nested
        if type(nested) is float and math.isfinite(nested):
            return nested
        raise RunnerContractError("authority configuration is not canonical JSON")

    return tuple((key, freeze(item)) for key, item in sorted(value.items()))


def _load_strict_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except RunnerContractError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerContractError(f"invalid {name}: {error}") from error
    if not isinstance(payload, dict):
        raise RunnerContractError(f"{name} must be a JSON object")
    return payload


def load_runner_authority() -> RunnerAuthority:
    """Load only clean tracked selection authority and its exact 20-config ledger."""

    from .selection import load_publication_execution_authority

    repository = Path(__file__).resolve().parents[1]
    try:
        registry = load_method_registry(repository / "study/methods.json")
        comparator_tuning = load_comparator_tuning_authority(
            repository,
            registry=registry,
            require_clean=False,
        )
        selection = load_publication_execution_authority()
    except Exception as error:
        raise RunnerContractError(
            f"publication execution authority is unavailable: {error}"
        ) from error
    comparator_reference = selection.comparator_tuning
    if (
        comparator_reference.path != "study/comparator_tuning.json"
        or comparator_reference.schema_version != comparator_tuning.schema_version
        or comparator_reference.authority_revision
        != comparator_tuning.authority_revision
    ):
        raise RunnerContractError(
            "comparator tuning reference differs from selection authority"
        )
    comparator_method_bindings = MappingProxyType(
        {
            method_id: comparator_method_binding(registry.by_id(method_id))
            for method_id in comparator_tuning.method_order
        }
    )
    if comparator_method_bindings != selection.comparator_method_bindings:
        raise RunnerContractError(
            "comparator method bindings differ from selection authority"
        )
    ledger = _load_strict_json(
        repository / "study/development_search.json", "development search ledger"
    )
    rows = ledger.get("configurations")
    if not isinstance(rows, list):
        raise RunnerContractError("development search configurations are invalid")
    configurations = derive_authorized_configurations(
        rows,
        selection.ablation_specs,
        selection.method_bindings,
    )
    file_hashes = dict(selection.file_sha256)
    required_files = {
        "study/protocol.json",
        "study/development_panel.json",
        "study/methods.json",
        "study/ablations.json",
        "study/selection_contract.json",
        "study/development_search.json",
        "study/calibration_contract.json",
    }
    if set(file_hashes) != required_files:
        raise RunnerContractError("selection authority file hash set is incomplete")
    for relative, expected in file_hashes.items():
        _require_sha256(expected, f"authority file {relative} checksum")
        if _file_sha256(repository / relative) != expected:
            raise RunnerContractError(
                f"authority file changed after clean validation: {relative}"
            )
    qc_mapping = dict(selection.dataset_qc_policy)
    try:
        qc_policy = DatasetQCPolicy(
            cell_exclusion_rule=qc_mapping["cell_exclusion_rule"],
            minimum_retained_cells=qc_mapping["minimum_retained_cells"],
            application=qc_mapping["application"],
            additional_cell_filtering=qc_mapping["additional_cell_filtering"],
            gene_filtering=qc_mapping["gene_filtering"],
            required_audit_fields=tuple(qc_mapping["required_audit_fields"]),
        )
    except (KeyError, TypeError) as error:
        raise RunnerContractError("selection QC policy cannot be adapted") from error
    search = [value for value in configurations if value.kind == "candidate_search"]
    authority_body = {
        "schema": "maskimpute-development-runner-authority-direct-v1",
        "plan_scope": "base_full_panel",
        "file_sha256": file_hashes,
        "configurations": [value.to_dict() for value in configurations],
        "comparator_tuning": asdict(comparator_reference),
        "comparator_method_bindings": [
            asdict(binding) for binding in comparator_method_bindings.values()
        ],
        "comparator_tuning_configurations": [
            {
                "method_id": row.method_id,
                "configuration_id": row.configuration_id,
                "payload": dict(row.payload),
                "is_upstream_default": row.is_upstream_default,
            }
            for row in comparator_tuning.configurations
        ],
        "base_maskimpute_config_sha256": selection.base_maskimpute_config_sha256,
        "count_model_config_sha256": selection.count_model_config_sha256,
        "dataset_qc_policy_sha256": selection.dataset_qc_policy_sha256,
        "count_score_manifest": asdict(selection.count_score_manifest),
        "retained_calibration": asdict(selection.retained_calibration),
        "calibration_effect_status": selection.calibration_effect_status,
        "calibration_equivalence_reason": selection.calibration_equivalence_reason,
    }
    return RunnerAuthority(
        schema_version=1,
        authority_sha256=canonical_sha256(authority_body),
        method_registry_sha256=file_hashes["study/methods.json"],
        selection_contract_sha256=file_hashes["study/selection_contract.json"],
        development_search_sha256=file_hashes["study/development_search.json"],
        ablation_registry_sha256=file_hashes["study/ablations.json"],
        base_configuration_id=search[0].configuration_id,
        base_configuration_sha256=selection.base_maskimpute_config_sha256,
        base_configuration=_freeze_json_mapping(selection.base_maskimpute_config),
        count_model_config=_freeze_json_mapping(selection.count_model_config),
        count_model_config_sha256=selection.count_model_config_sha256,
        count_score_manifest_status=selection.count_score_manifest.status,
        count_score_manifest_sha256=selection.count_score_manifest.sha256,
        retained_calibration_status=selection.retained_calibration.status,
        retained_calibration_sha256=selection.retained_calibration.sha256,
        dataset_qc_policy=qc_policy,
        dataset_qc_policy_sha256=selection.dataset_qc_policy_sha256,
        count_score_manifest_path=selection.count_score_manifest.path,
        retained_calibration_path=selection.retained_calibration.path,
        configurations=configurations,
        comparator_tuning_reference=comparator_reference,
        comparator_method_bindings=comparator_method_bindings,
        comparator_tuning=comparator_tuning,
        plan_scope="base_full_panel",
    )


@dataclass(frozen=True, slots=True)
class RunnerAuthority:
    """Hash-bound tracked authority required to plan development execution."""

    schema_version: int
    authority_sha256: str
    method_registry_sha256: str
    selection_contract_sha256: str
    development_search_sha256: str
    ablation_registry_sha256: str
    base_configuration_id: str
    base_configuration_sha256: str
    base_configuration: tuple[tuple[str, object], ...]
    count_model_config: tuple[tuple[str, object], ...]
    count_model_config_sha256: str
    count_score_manifest_status: str
    count_score_manifest_sha256: str | None
    retained_calibration_status: str
    retained_calibration_sha256: str | None
    dataset_qc_policy: DatasetQCPolicy
    dataset_qc_policy_sha256: str
    count_score_manifest_path: str
    retained_calibration_path: str
    configurations: tuple[AuthorizedConfiguration, ...]
    comparator_tuning_reference: ComparatorAuthorityReference
    comparator_method_bindings: Mapping[str, ComparatorMethodBinding]
    comparator_tuning: ComparatorTuningAuthority
    plan_scope: Literal["base_full_panel", "revision_candidate_only"] = (
        "base_full_panel"
    )
    base_comparator_selection: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise RunnerContractError("runner authority schema_version must be 1")
        for name in (
            "authority_sha256",
            "method_registry_sha256",
            "selection_contract_sha256",
            "development_search_sha256",
            "ablation_registry_sha256",
            "base_configuration_sha256",
            "count_model_config_sha256",
            "dataset_qc_policy_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.comparator_tuning, ComparatorTuningAuthority):
            raise RunnerContractError(
                "comparator_tuning must be a ComparatorTuningAuthority"
            )
        if not isinstance(
            self.comparator_tuning_reference, ComparatorAuthorityReference
        ):
            raise RunnerContractError(
                "comparator_tuning_reference must be a ComparatorAuthorityReference"
            )
        if (
            self.comparator_tuning_reference.path != "study/comparator_tuning.json"
            or self.comparator_tuning_reference.schema_version
            != self.comparator_tuning.schema_version
            or self.comparator_tuning_reference.authority_revision
            != self.comparator_tuning.authority_revision
        ):
            raise RunnerContractError("runner comparator tuning reference mismatch")
        if type(self.comparator_method_bindings) is not MappingProxyType:
            raise RunnerContractError(
                "runner comparator method bindings must be immutable"
            )
        if (
            tuple(self.comparator_method_bindings)
            != self.comparator_tuning.method_order
        ):
            raise RunnerContractError(
                "runner comparator method bindings are incomplete or reordered"
            )
        if any(
            not isinstance(binding, ComparatorMethodBinding)
            or binding.method_id != method_id
            for method_id, binding in self.comparator_method_bindings.items()
        ):
            raise RunnerContractError("runner comparator method binding is invalid")
        if self.plan_scope not in {"base_full_panel", "revision_candidate_only"}:
            raise RunnerContractError("runner authority plan scope is invalid")
        if self.base_comparator_selection is not None:
            from .direct_values import direct_json_value, freeze_direct_mapping

            if not isinstance(self.base_comparator_selection, Mapping) or set(
                self.base_comparator_selection
            ) != {
                "path",
                "receipt",
                "selected_by_method",
                "nonexecution_identity_by_method",
                "ready_comparison_population_ids",
            }:
                raise RunnerContractError(
                    "runner base comparator selection is incomplete"
                )
            try:
                normalized_selection = direct_json_value(
                    self.base_comparator_selection,
                    payload=True,
                )
                if not isinstance(normalized_selection, Mapping):
                    raise ValueError(
                        "runner base comparator selection is not an object"
                    )
                frozen_selection = MappingProxyType(
                    dict(freeze_direct_mapping(normalized_selection))
                )
            except ValueError as error:
                raise RunnerContractError(
                    "runner base comparator selection is invalid"
                ) from error
            object.__setattr__(
                self,
                "base_comparator_selection",
                frozen_selection,
            )
        if not isinstance(self.base_configuration_id, str) or not _SAFE_ID.fullmatch(
            self.base_configuration_id
        ):
            raise RunnerContractError("base_configuration_id must be a safe identifier")
        if self.count_score_manifest_status not in {"pending", "ready"}:
            raise RunnerContractError("count score status must be pending or ready")
        if self.retained_calibration_status not in {"pending", "ready"}:
            raise RunnerContractError("calibration status must be pending or ready")
        _require_sha256(
            self.count_score_manifest_sha256,
            "count score manifest checksum",
            nullable=self.count_score_manifest_status == "pending",
        )
        _require_sha256(
            self.retained_calibration_sha256,
            "retained calibration checksum",
            nullable=self.retained_calibration_status == "pending",
        )
        if (
            self.count_score_manifest_status == "ready"
            and self.count_score_manifest_sha256 is None
        ):
            raise RunnerContractError("ready count score manifest requires a checksum")
        if (
            self.retained_calibration_status == "ready"
            and self.retained_calibration_sha256 is None
        ):
            raise RunnerContractError("ready calibration requires a checksum")
        if not isinstance(self.dataset_qc_policy, DatasetQCPolicy):
            raise RunnerContractError("dataset_qc_policy must be a DatasetQCPolicy")
        self.dataset_qc_policy.require_fixed_publication_rule()
        if self.dataset_qc_policy_sha256 != self.dataset_qc_policy.sha256:
            raise RunnerContractError("dataset QC policy checksum mismatch")
        for name in ("count_score_manifest_path", "retained_calibration_path"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise RunnerContractError(f"{name} must be a repository-relative path")
            relative = PurePosixPath(value)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise RunnerContractError(
                    f"{name} must be a safe repository-relative path"
                )
        if not self.configurations or not all(
            isinstance(value, AuthorizedConfiguration) for value in self.configurations
        ):
            raise RunnerContractError("runner authority requires configurations")
        identities = [
            (value.method_id, value.configuration_id) for value in self.configurations
        ]
        if len(identities) != len(set(identities)):
            raise RunnerContractError(
                "runner authority configuration identities repeat"
            )
        if self.plan_scope == "base_full_panel":
            if self.base_comparator_selection is not None:
                raise RunnerContractError(
                    "base runner authority cannot carry a later comparator selection"
                )
            search_count = sum(
                value.method_id == "maskimpute" and value.kind == "candidate_search"
                for value in self.configurations
            )
            capacity_count = sum(
                value.method_id == "capacity-matched-ae" and value.kind == "ablation"
                for value in self.configurations
            )
            if (
                len(self.configurations) != 26
                or search_count != MAX_DEVELOPMENT_CONFIGURATIONS
                or capacity_count != 1
                or sum(value.method_id == "maskimpute" for value in self.configurations)
                != 25
                or any(
                    value.method_id not in {"maskimpute", "capacity-matched-ae"}
                    for value in self.configurations
                )
            ):
                raise RunnerContractError(
                    "base runner authority requires the exact 26 configurations"
                )
        else:
            if (
                len(self.configurations) != 1
                or self.configurations[0].method_id != "maskimpute"
                or self.configurations[0].kind != "candidate_search"
            ):
                raise RunnerContractError(
                    "revision runner authority requires exactly one MaskImpute candidate"
                )

    @property
    def maskimpute_ready(self) -> bool:
        return (
            self.count_score_manifest_status == "ready"
            and self.count_score_manifest_sha256 is not None
            and self.retained_calibration_status == "ready"
            and self.retained_calibration_sha256 is not None
        )

    @property
    def execution_context(self) -> ExecutionAuthorityContext:
        return ExecutionAuthorityContext(
            authority_sha256=self.authority_sha256,
            base_configuration_json=_canonical_bytes(
                _thaw_frozen_json(self.base_configuration)
            ).decode("utf-8"),
            base_configuration_sha256=self.base_configuration_sha256,
            count_model_config_json=_canonical_bytes(
                _thaw_frozen_json(self.count_model_config)
            ).decode("utf-8"),
            count_model_config_sha256=self.count_model_config_sha256,
            count_score_manifest_path=self.count_score_manifest_path,
            count_score_manifest_sha256=self.count_score_manifest_sha256,
            retained_calibration_path=self.retained_calibration_path,
            retained_calibration_sha256=self.retained_calibration_sha256,
        )


def load_v28_revision_authority() -> RunnerAuthority:
    """Load the fixed decoder-only v28 authority without altering v27 search."""

    base = load_runner_authority()
    try:
        revision_bytes = _TRACKED_V28_REVISION_PATH.read_bytes()
    except OSError as error:
        raise RunnerContractError(
            "tracked v28 revision authority is unavailable"
        ) from error
    revision_sha256 = hashlib.sha256(revision_bytes).hexdigest()
    if revision_sha256 != _TRACKED_V28_REVISION_SHA256:
        raise RunnerContractError("tracked v28 revision authority checksum differs")
    try:
        revision = json.loads(
            revision_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except RunnerContractError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerContractError(f"invalid v28 revision authority: {error}") from error
    if not isinstance(revision, dict):
        raise RunnerContractError("v28 revision authority must be a JSON object")
    if revision_bytes != json.dumps(revision, indent=2).encode("utf-8") + b"\n":
        raise RunnerContractError(
            "v28 revision authority is not canonical tracked JSON"
        )
    expected_fields = {
        "schema_version",
        "status",
        "trigger",
        "parent_configuration_id",
        "parent_configuration_sha256",
        "configuration_id",
        "configuration",
        "configuration_sha256",
        "reason_code",
    }
    if set(revision) != expected_fields:
        raise RunnerContractError("v28 revision authority fields differ")
    if (
        revision["schema_version"] != 1
        or type(revision["schema_version"]) is not int
        or revision["status"] != "conditional_on_v28_trigger"
        or revision["trigger"] != "v28"
        or revision["reason_code"] != "prespecified_decoder_only_revision_of_v27_c03"
    ):
        raise RunnerContractError("v28 revision activation contract differs")
    parent_id = revision["parent_configuration_id"]
    parent_sha256 = revision["parent_configuration_sha256"]
    try:
        parent = next(
            value
            for value in base.configurations
            if value.method_id == "maskimpute" and value.configuration_id == parent_id
        )
    except StopIteration as error:
        raise RunnerContractError(
            "v28 parent is absent from frozen v27 authority"
        ) from error
    if (
        parent.kind != "candidate_search"
        or parent.configuration_sha256 != parent_sha256
    ):
        raise RunnerContractError("v28 parent configuration binding differs")
    payload = revision["configuration"]
    configuration_sha256 = revision["configuration_sha256"]
    if not isinstance(payload, Mapping):
        raise RunnerContractError("v28 revision configuration must be a mapping")
    _require_sha256(configuration_sha256, "v28 configuration checksum")
    if canonical_sha256(payload) != configuration_sha256:
        raise RunnerContractError("v28 revision configuration checksum differs")
    if payload.get("hyperparameters") != parent.payload.get("hyperparameters"):
        raise RunnerContractError("v28 revision changes parent hyperparameters")
    for field in ("encoder_mode", "output_policy", "score_policy"):
        if payload.get(field) != parent.payload.get(field):
            raise RunnerContractError(f"v28 revision changes parent {field}")
    candidate = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id=revision["configuration_id"],
        kind="candidate_search",
        payload=payload,
        requires_count_score=True,
        requires_calibration=True,
        configuration_sha256=configuration_sha256,
    )
    if maskimpute_decoder_for_configuration(candidate)[0] != "negative_binomial":
        raise RunnerContractError("v28 revision does not resolve to NB dispatch")
    authority_body = {
        "schema": "maskimpute-v28-revision-runner-authority-v1",
        "plan_scope": "revision_candidate_only",
        "base_runner_authority_sha256": base.authority_sha256,
        "revision_file_sha256": revision_sha256,
        "parent_configuration_id": parent.configuration_id,
        "parent_configuration_sha256": parent.configuration_sha256,
        "configuration": candidate.to_dict(),
    }
    return RunnerAuthority(
        schema_version=base.schema_version,
        authority_sha256=canonical_sha256(authority_body),
        method_registry_sha256=base.method_registry_sha256,
        selection_contract_sha256=base.selection_contract_sha256,
        development_search_sha256=base.development_search_sha256,
        ablation_registry_sha256=base.ablation_registry_sha256,
        base_configuration_id=parent.configuration_id,
        base_configuration_sha256=base.base_configuration_sha256,
        base_configuration=base.base_configuration,
        count_model_config=base.count_model_config,
        count_model_config_sha256=base.count_model_config_sha256,
        count_score_manifest_status=base.count_score_manifest_status,
        count_score_manifest_sha256=base.count_score_manifest_sha256,
        retained_calibration_status=base.retained_calibration_status,
        retained_calibration_sha256=base.retained_calibration_sha256,
        dataset_qc_policy=base.dataset_qc_policy,
        dataset_qc_policy_sha256=base.dataset_qc_policy_sha256,
        count_score_manifest_path=base.count_score_manifest_path,
        retained_calibration_path=base.retained_calibration_path,
        configurations=(candidate,),
        comparator_tuning_reference=base.comparator_tuning_reference,
        comparator_method_bindings=base.comparator_method_bindings,
        comparator_tuning=base.comparator_tuning,
        plan_scope="revision_candidate_only",
    )


def load_v29_revision_authority() -> RunnerAuthority:
    """Load the prespecified structure-only v29 authority without activating it."""

    from .revisions import load_revision_spec, thaw_revision_configuration

    repository = Path(__file__).resolve().parents[1]
    base = load_runner_authority()
    v28 = load_v28_revision_authority()
    try:
        revision = load_revision_spec(repository, "v29", require_clean=False)
    except Exception as error:
        raise RunnerContractError(
            f"tracked v29 revision authority is unavailable: {error}"
        ) from error
    if revision.file_sha256 != _TRACKED_V29_REVISION_SHA256:
        raise RunnerContractError("tracked v29 revision authority checksum differs")
    try:
        parent = next(
            value
            for value in v28.configurations
            if value.method_id == "maskimpute"
            and value.configuration_id == revision.parent_configuration_id
        )
    except StopIteration as error:
        raise RunnerContractError(
            "v29 parent is absent from prespecified v28 authority"
        ) from error
    if parent.configuration_sha256 != revision.parent_configuration_sha256:
        raise RunnerContractError("v29 parent configuration binding differs")
    payload = thaw_revision_configuration(revision)
    parent_payload = dict(parent.payload)
    if payload.get("hyperparameters") != parent_payload.get(
        "hyperparameters"
    ) or payload.get("decoder_hyperparameters") != parent_payload.get(
        "decoder_hyperparameters"
    ):
        raise RunnerContractError("v29 revision changes its parent model budget")
    for field in ("decoder", "encoder_mode", "output_policy", "score_policy"):
        if payload.get(field) != parent_payload.get(field):
            raise RunnerContractError(f"v29 revision changes parent {field}")
    structure = payload.get("structure_hyperparameters")
    if structure != {
        "variable_gene_count": 200,
        "neighborhood_k": 15,
        "covariance_penalty_weight": 0.1,
        "neighborhood_penalty_weight": 0.1,
        "variance_floor": 1e-8,
    }:
        raise RunnerContractError("v29 structure penalty authority differs")
    candidate = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id=revision.configuration_id,
        kind="candidate_search",
        payload=payload,
        requires_count_score=True,
        requires_calibration=True,
        configuration_sha256=revision.configuration_sha256,
    )
    authority_body = {
        "schema": "maskimpute-v29-revision-runner-authority-v1",
        "plan_scope": "revision_candidate_only",
        "base_runner_authority_sha256": base.authority_sha256,
        "v28_revision_authority_sha256": v28.authority_sha256,
        "revision_file_sha256": revision.file_sha256,
        "parent_configuration_id": parent.configuration_id,
        "parent_configuration_sha256": parent.configuration_sha256,
        "configuration": candidate.to_dict(),
    }
    return RunnerAuthority(
        schema_version=base.schema_version,
        authority_sha256=canonical_sha256(authority_body),
        method_registry_sha256=base.method_registry_sha256,
        selection_contract_sha256=base.selection_contract_sha256,
        development_search_sha256=base.development_search_sha256,
        ablation_registry_sha256=base.ablation_registry_sha256,
        base_configuration_id=parent.configuration_id,
        base_configuration_sha256=base.base_configuration_sha256,
        base_configuration=base.base_configuration,
        count_model_config=base.count_model_config,
        count_model_config_sha256=base.count_model_config_sha256,
        count_score_manifest_status=base.count_score_manifest_status,
        count_score_manifest_sha256=base.count_score_manifest_sha256,
        retained_calibration_status=base.retained_calibration_status,
        retained_calibration_sha256=base.retained_calibration_sha256,
        dataset_qc_policy=base.dataset_qc_policy,
        dataset_qc_policy_sha256=base.dataset_qc_policy_sha256,
        count_score_manifest_path=base.count_score_manifest_path,
        retained_calibration_path=base.retained_calibration_path,
        configurations=(candidate,),
        comparator_tuning_reference=base.comparator_tuning_reference,
        comparator_method_bindings=base.comparator_method_bindings,
        comparator_tuning=base.comparator_tuning,
        plan_scope="revision_candidate_only",
    )


def _secure_canonical_artifact(path: Path, name: str) -> tuple[dict[str, Any], str]:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o002
        ):
            raise RunnerContractError(f"{name} is not a secure unique regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    except RunnerContractError:
        raise
    except OSError as error:
        raise RunnerContractError(f"{name} is unavailable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except RunnerContractError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerContractError(f"invalid {name}: {error}") from error
    if not isinstance(payload, dict):
        raise RunnerContractError(f"{name} must be a JSON object")
    expected = _canonical_bytes(payload) + b"\n"
    if raw != expected:
        raise RunnerContractError(f"{name} is not canonical JSON")
    return payload, hashlib.sha256(raw).hexdigest()


def _validate_v28_activation(
    repository: Path | None = None,
):
    """Revalidate the complete fixed base evidence and require its v28 trigger."""

    from .revisions import validate_revision_activation

    from .revisions import RevisionActivation

    root = Path(__file__).resolve().parents[1] if repository is None else repository
    try:
        activation = validate_revision_activation(root, "v28")
    except Exception as error:
        raise RunnerContractError(f"v28 activation failed: {error}") from error
    if not isinstance(activation, RevisionActivation):  # pragma: no cover
        raise RunnerContractError("v28 activation returned an invalid authority")
    return activation


def _bind_v28_activation(
    authority: RunnerAuthority,
    activation: object,
) -> RunnerAuthority:
    from .revisions import RevisionActivation

    if not isinstance(activation, RevisionActivation):
        raise TypeError("activation must be a RevisionActivation")
    input_sha256 = activation.selection_input_file_sha256
    result_sha256 = activation.selection_result_sha256
    report_sha256 = activation.selection_report_file_sha256
    _require_sha256(input_sha256, "v28 selection input checksum")
    _require_sha256(result_sha256, "v28 selection result checksum")
    _require_sha256(report_sha256, "v28 selection report checksum")
    return replace(
        authority,
        authority_sha256=canonical_sha256(
            {
                "schema": "maskimpute-v28-activated-runner-authority-v1",
                "revision_authority_sha256": authority.authority_sha256,
                "selection_input_sha256": input_sha256,
                "selection_result_sha256": result_sha256,
                "selection_report_sha256": report_sha256,
            }
        ),
        base_comparator_selection=activation.base_comparator_selection,
    )


def direct_bound_comparator_value(
    value: BoundComparatorConfiguration,
) -> dict[str, object]:
    """Encode one complete selected comparator with its readable typed payload."""

    from .comparator_tuning import BoundComparatorConfiguration
    from .direct_values import direct_json_value

    if not isinstance(value, BoundComparatorConfiguration):
        raise TypeError("value must be a BoundComparatorConfiguration")
    payload = value.configuration.payload
    return {
        "configuration": {
            "method_id": value.configuration.method_id,
            "configuration_id": value.configuration.configuration_id,
            "is_upstream_default": value.configuration.is_upstream_default,
            "payload": direct_json_value(payload, payload=True),
        },
        "authority_reference": direct_json_value(value.authority_reference),
        "method": direct_json_value(value.method),
    }


def decode_direct_bound_comparator_value(
    value: object,
) -> BoundComparatorConfiguration:
    """Decode the readable complete comparator value without content summaries."""

    from .comparator_tuning import (
        ComparatorAuthorityReference,
        BoundComparatorConfiguration,
        ComparatorConfiguration,
        ComparatorMethodBinding,
        ComparatorTuningError,
        _validate_bound_selection_configuration,
    )
    from .direct_values import direct_equal

    if not isinstance(value, Mapping) or set(value) != {
        "configuration",
        "authority_reference",
        "method",
    }:
        raise RunnerContractError("direct comparator value schema is invalid")
    raw_configuration = value.get("configuration")
    raw_reference = value.get("authority_reference")
    raw_method = value.get("method")
    if (
        not isinstance(raw_configuration, Mapping)
        or set(raw_configuration)
        != {"method_id", "configuration_id", "is_upstream_default", "payload"}
        or not isinstance(raw_configuration.get("payload"), Mapping)
        or not isinstance(raw_reference, Mapping)
        or not isinstance(raw_method, Mapping)
    ):
        raise RunnerContractError("direct comparator value schema is invalid")
    try:
        configuration = ComparatorConfiguration(
            method_id=raw_configuration["method_id"],
            configuration_id=raw_configuration["configuration_id"],
            payload_json=_canonical_bytes(dict(raw_configuration["payload"])).decode(
                "utf-8"
            ),
            is_upstream_default=raw_configuration["is_upstream_default"],
        )
        reference = ComparatorAuthorityReference(**dict(raw_reference))
        method = ComparatorMethodBinding(**dict(raw_method))
        result = BoundComparatorConfiguration(configuration, reference, method)
        _validate_bound_selection_configuration(result)
    except (ComparatorTuningError, TypeError, ValueError) as error:
        raise RunnerContractError("direct comparator value is invalid") from error
    if not direct_equal(direct_bound_comparator_value(result), value):
        raise RunnerContractError("direct comparator value differs after decoding")
    return result


@dataclass(frozen=True, slots=True)
class RunPlanEntry:
    """One denominator entry, including preflight-blocked attempts."""

    ordinal: int
    run_id: str
    method_id: str
    dataset_id: str
    source_dataset_sha256: str
    mechanism: str
    biological_id: str
    technical_view: str
    model_seed: int | None
    configuration_id: str
    configuration_sha256: str | None
    preflight_status: Literal["planned", "blocked_authority"]
    preflight_reason: str | None
    configuration_kind: str = "registry"
    requires_count_score: bool = False
    requires_calibration: bool = False
    configuration_payload_sha256: str | None = None
    configuration_method_identity_sha256: str | None = None
    nonexecution_identity_sha256: str | None = None
    comparator_configuration: BoundComparatorConfiguration | None = None
    comparator_nonexecution_identity: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.configuration_kind not in {
            "registry",
            "candidate_search",
            "ablation",
            "comparator_tuning",
            "comparator_nonexecution",
        }:
            raise RunnerContractError("plan configuration kind is invalid")
        if self.configuration_kind in {
            "comparator_tuning",
            "comparator_nonexecution",
        }:
            self._validate_direct_comparator_identity()
            return
        _require_sha256(self.configuration_sha256, "plan configuration checksum")
        if self.configuration_payload_sha256 is None:
            object.__setattr__(
                self,
                "configuration_payload_sha256",
                self.configuration_sha256,
            )
        _require_sha256(
            self.configuration_payload_sha256,
            "plan configuration payload checksum",
        )
        if self.configuration_payload_sha256 != self.configuration_sha256:
            raise RunnerContractError("plan configuration payload checksum differs")
        if (
            self.configuration_method_identity_sha256 is not None
            or self.nonexecution_identity_sha256 is not None
            or self.comparator_configuration is not None
            or self.comparator_nonexecution_identity is not None
        ):
            raise RunnerContractError(
                "legacy plan entry forbids comparator identity fields"
            )

    def _validate_direct_comparator_identity(self) -> None:
        from .comparator_tuning import (
            BoundComparatorConfiguration,
            ComparatorTuningError,
            _validate_bound_selection_configuration,
        )
        from .direct_values import direct_json_value, freeze_direct_mapping

        if any(
            value is not None
            for value in (
                self.configuration_sha256,
                self.configuration_payload_sha256,
                self.configuration_method_identity_sha256,
                self.nonexecution_identity_sha256,
            )
        ):
            raise RunnerContractError(
                "direct comparator plan entry forbids content summaries"
            )
        if self.requires_count_score or self.requires_calibration:
            raise RunnerContractError(
                "direct comparator plan entry forbids score authority"
            )
        if self.configuration_kind == "comparator_tuning":
            if (
                not isinstance(
                    self.comparator_configuration,
                    BoundComparatorConfiguration,
                )
                or self.comparator_nonexecution_identity is not None
                or self.configuration_id == "registry-default"
                or self.comparator_configuration.method.method_id != self.method_id
                or self.comparator_configuration.configuration.method_id
                != self.method_id
                or self.comparator_configuration.configuration.configuration_id
                != self.configuration_id
            ):
                raise RunnerContractError(
                    "selected comparator plan identity is invalid"
                )
            try:
                _validate_bound_selection_configuration(self.comparator_configuration)
                decoded = self.comparator_configuration.configuration.decode()
                if encode_comparator_configuration(decoded) != dict(
                    self.comparator_configuration.configuration.payload
                ):
                    raise RunnerContractError(
                        "selected comparator payload changed during typed decoding"
                    )
            except (ComparatorTuningError, TypeError, ValueError) as error:
                raise RunnerContractError(
                    "selected comparator plan identity is invalid"
                ) from error
            return
        if (
            self.comparator_configuration is not None
            or not isinstance(self.comparator_nonexecution_identity, Mapping)
            or self.configuration_id != f"nonexecution-{self.method_id}"
        ):
            raise RunnerContractError(
                "comparator nonexecution plan identity is invalid"
            )
        try:
            encoded = direct_json_value(
                self.comparator_nonexecution_identity,
                payload=True,
            )
            if not isinstance(encoded, Mapping):
                raise ValueError("nonexecution identity is not an object")
            frozen = MappingProxyType(dict(freeze_direct_mapping(encoded)))
        except (TypeError, ValueError) as error:
            raise RunnerContractError(
                "comparator nonexecution plan identity is invalid"
            ) from error
        object.__setattr__(self, "comparator_nonexecution_identity", frozen)

    @property
    def nonexecution_identity(self) -> Mapping[str, object] | None:
        """Compatibility spelling for the complete direct nonexecution value."""

        return self.comparator_nonexecution_identity

    def to_dict(self) -> dict[str, object]:
        if self.configuration_kind not in {
            "comparator_tuning",
            "comparator_nonexecution",
        }:
            value = asdict(self)
            value.pop("comparator_configuration")
            value.pop("comparator_nonexecution_identity")
            return value
        from .direct_values import direct_json_value

        return {
            "ordinal": self.ordinal,
            "run_id": self.run_id,
            "method_id": self.method_id,
            "dataset_id": self.dataset_id,
            "source_dataset_sha256": self.source_dataset_sha256,
            "mechanism": self.mechanism,
            "biological_id": self.biological_id,
            "technical_view": self.technical_view,
            "model_seed": self.model_seed,
            "configuration_id": self.configuration_id,
            "preflight_status": self.preflight_status,
            "preflight_reason": self.preflight_reason,
            "configuration_kind": self.configuration_kind,
            "requires_count_score": self.requires_count_score,
            "requires_calibration": self.requires_calibration,
            "comparator_configuration": (
                None
                if self.comparator_configuration is None
                else direct_bound_comparator_value(self.comparator_configuration)
            ),
            "comparator_nonexecution_identity": (
                None
                if self.comparator_nonexecution_identity is None
                else direct_json_value(
                    self.comparator_nonexecution_identity,
                    payload=True,
                )
            ),
        }


def _counts_toward_configuration_limit(entry: RunPlanEntry) -> bool:
    return entry.configuration_kind in {"candidate_search", "comparator_tuning"}


def _budget_scope(entry: RunPlanEntry) -> str:
    return (
        f"{entry.method_id}:{entry.configuration_kind}"
        if entry.method_id == "maskimpute"
        else entry.method_id
    )


def _competition_plan_body(
    *,
    schema_version: int,
    input_hashes: Mapping[str, str],
    entries: Sequence[RunPlanEntry],
    configurations: Sequence[AuthorizedConfiguration],
    execution_context: ExecutionAuthorityContext | None,
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "input_hashes": dict(input_hashes),
        "entries": [entry.to_dict() for entry in entries],
        "configurations": [value.to_dict() for value in configurations],
        "execution_context": (
            None if execution_context is None else asdict(execution_context)
        ),
        "budgets": {
            "maximum_configurations": MAX_DEVELOPMENT_CONFIGURATIONS,
            "gpu_seconds": MAX_GPU_BUDGET_SECONDS,
            "cpu_seconds": MAX_CPU_BUDGET_SECONDS,
            "failures_consume_budget_except": "infrastructure_error",
        },
    }


@dataclass(frozen=True, slots=True)
class CompetitionPlan:
    """Immutable full development denominator and all authority hashes."""

    schema_version: int
    input_hashes: Mapping[str, str]
    entries: tuple[RunPlanEntry, ...]
    plan_sha256: str
    configurations: tuple[AuthorizedConfiguration, ...] = ()
    execution_context: ExecutionAuthorityContext | None = None

    @property
    def recomputed_plan_sha256(self) -> str:
        return canonical_sha256(
            _competition_plan_body(
                schema_version=self.schema_version,
                input_hashes=self.input_hashes,
                entries=self.entries,
                configurations=self.configurations,
                execution_context=self.execution_context,
            )
        )

    def validate_integrity(self) -> None:
        _require_sha256(self.plan_sha256, "competition plan checksum")
        if self.plan_sha256 != self.recomputed_plan_sha256:
            raise RunnerContractError("competition plan checksum mismatch")


def _run_id(
    spec: MethodSpec,
    binding: DatasetBinding,
    seed: int | None,
    configuration_sha256: str,
) -> str:
    seed_token = "deterministic" if seed is None else f"seed-{seed}"
    return (
        f"run-{spec.id}-{binding.dataset_id.removeprefix('dataset-')}-"
        f"{seed_token}-{configuration_sha256[:12]}"
    )


def build_fair_comparator_plan(
    registry: MethodRegistry,
    datasets: Sequence[DatasetBinding],
    authority: RunnerAuthority,
    prepared_datasets: Sequence[PreparedDataset],
    *,
    _comparator_smoke_receipt: Mapping[str, object],
    _comparator_smoke_receipt_bytes: bytes,
) -> DirectCompetitionPlan:
    """Build the direct-identity fair-comparator development plan."""

    from .fair_comparator_plan import (
        build_direct_competition_plan,
    )

    if (
        not isinstance(_comparator_smoke_receipt, Mapping)
        or type(_comparator_smoke_receipt_bytes) is not bytes
    ):
        raise RunnerContractError("comparator smoke receipt evidence is incomplete")
    return build_direct_competition_plan(
        registry,
        datasets,
        authority,
        prepared_datasets,
        comparator_smoke_receipt=_comparator_smoke_receipt,
        comparator_smoke_receipt_bytes=_comparator_smoke_receipt_bytes,
    )


def development_storage_preflight(
    plan: DirectCompetitionPlan,
    prepared_datasets: Mapping[str, PreparedDataset],
    *,
    completed_records: int,
) -> DevelopmentStoragePreflight:
    """Calculate the exact direct retained-storage ceiling without filesystem I/O."""

    from .direct_values import direct_equal, direct_json_value
    from .fair_comparator_plan import (
        DirectCompetitionPlan,
        PreparedInputDescriptor,
        describe_prepared_input,
    )

    if not isinstance(plan, DirectCompetitionPlan):
        raise TypeError("plan must be a DirectCompetitionPlan")
    if not isinstance(prepared_datasets, Mapping):
        raise TypeError("prepared_datasets must be a mapping")
    if (
        isinstance(completed_records, bool)
        or type(completed_records) is not int
        or not 0 <= completed_records <= len(plan.entries)
    ):
        raise RunnerContractError("completed storage record count is invalid")
    expected_ids = tuple(descriptor.dataset_id for descriptor in plan.inputs)
    if len(expected_ids) != len(set(expected_ids)) or set(prepared_datasets) != set(
        expected_ids
    ):
        raise RunnerContractError(
            "prepared dataset authority does not exactly cover the direct plan"
        )
    descriptor_values: list[Mapping[str, object]] = []
    for descriptor in plan.inputs:
        if not isinstance(descriptor, PreparedInputDescriptor):
            raise RunnerContractError("direct prepared input descriptor is invalid")
        prepared = prepared_datasets.get(descriptor.dataset_id)
        if not isinstance(prepared, PreparedDataset):
            raise RunnerContractError("direct prepared dataset authority is invalid")
        observed = describe_prepared_input(prepared)
        if not direct_equal(observed, descriptor):
            raise RunnerContractError("direct prepared input descriptor differs")
        encoded = direct_json_value(observed)
        if not isinstance(encoded, Mapping):  # pragma: no cover - dataclass invariant
            raise AssertionError("prepared input descriptor must encode as an object")
        descriptor_values.append(encoded)
    remaining = plan.entries[completed_records:]
    executable = tuple(
        entry for entry in remaining if entry.preflight_status == "planned"
    )
    try:
        matrix_bytes = sum(
            2 * prepared_datasets[entry.identity.dataset_id].method_input.counts.nbytes
            for entry in executable
        )
        prezero_bytes = sum(
            zlib_compress_bound(
                prepared_datasets[entry.identity.dataset_id].method_input.counts.nbytes
            )
            for entry in executable
            if entry.identity.method_id == "maskimpute" and entry.requires_count_score
        )
    except KeyError as error:
        raise RunnerContractError(
            "direct storage entry references an unknown prepared input"
        ) from error
    log_bytes = 2 * len(executable) * DEVELOPMENT_MAX_LOG_RECEIPT_BYTES
    executor_bytes = len(executable) * DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES
    record_bytes = len(remaining) * DEVELOPMENT_MAX_RECORD_BYTES
    required = (
        matrix_bytes
        + prezero_bytes
        + log_bytes
        + executor_bytes
        + record_bytes
        + DEVELOPMENT_MAX_CHECKPOINT_BYTES
        + DEVELOPMENT_STORAGE_RESERVE_BYTES
    )
    return DevelopmentStoragePreflight(
        schema="maskimpute-development-storage-preflight-v1",
        identity_mode="direct-v1",
        authority_revision=plan.authority_revision,
        plan_snapshot=plan.to_dict(),
        prepared_input_descriptors=tuple(descriptor_values),
        retained_dimensions=tuple(
            (descriptor.dataset_id, descriptor.shape) for descriptor in plan.inputs
        ),
        policy={
            "matrix_copies_per_executable": 2,
            "stream_receipts_per_executable": 2,
            "max_log_receipt_bytes": DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
            "max_executor_receipt_bytes": DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
            "max_record_bytes": DEVELOPMENT_MAX_RECORD_BYTES,
            "max_checkpoint_bytes": DEVELOPMENT_MAX_CHECKPOINT_BYTES,
            "reserve_bytes": DEVELOPMENT_STORAGE_RESERVE_BYTES,
        },
        planned_run_count=len(plan.entries),
        completed_record_count=completed_records,
        remaining_executable_count=len(executable),
        matrix_bytes=matrix_bytes,
        prezero_zlib_bound_bytes=prezero_bytes,
        log_receipt_bytes=log_bytes,
        executor_receipt_bytes=executor_bytes,
        record_bytes=record_bytes,
        checkpoint_bytes=DEVELOPMENT_MAX_CHECKPOINT_BYTES,
        reserve_bytes=DEVELOPMENT_STORAGE_RESERVE_BYTES,
        required_free_bytes=required,
    )


def require_development_storage_capacity(
    output_dir: Path,
    plan: DirectCompetitionPlan,
    prepared_datasets: Mapping[str, PreparedDataset],
    *,
    completed_records: int,
    available_bytes: int | None = None,
) -> DevelopmentStoragePreflight:
    """Require the direct retained bound before creating or changing output."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if available_bytes is not None and (
        isinstance(available_bytes, bool)
        or type(available_bytes) is not int
        or available_bytes < 0
    ):
        raise RunnerContractError("available development storage is invalid")
    receipt = development_storage_preflight(
        plan,
        prepared_datasets,
        completed_records=completed_records,
    )
    probe = output_dir.absolute()
    while not probe.exists():
        if probe.parent == probe:
            raise RunnerContractError("development storage filesystem is unavailable")
        probe = probe.parent
    if available_bytes is None:
        try:
            filesystem = os.statvfs(probe)
        except OSError as error:
            raise RunnerContractError(
                "development storage filesystem is unavailable"
            ) from error
        observed = int(filesystem.f_bavail * filesystem.f_frsize)
    else:
        observed = available_bytes
    if observed < receipt.required_free_bytes:
        raise RunnerContractError(
            "insufficient development storage before scientific write"
        )
    return receipt


def build_competition_plan(
    registry: MethodRegistry,
    datasets: Sequence[DatasetBinding],
    authority: RunnerAuthority,
    *,
    execution_environment_sha256: str | None = None,
    runtime_lock_sha256: str | None = None,
) -> CompetitionPlan:
    """Build the exhaustive method x dataset x seed denominator."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(authority, RunnerAuthority):
        raise TypeError("authority must be a RunnerAuthority")
    if execution_environment_sha256 is None or runtime_lock_sha256 is None:
        raise RunnerContractError("plan runtime identity is absent")
    _require_sha256(
        execution_environment_sha256, "execution environment registry checksum"
    )
    _require_sha256(runtime_lock_sha256, "runtime lock checksum")
    if authority.plan_scope == "base_full_panel":
        raise RunnerContractError(
            "direct fair-comparator planning requires build_fair_comparator_plan"
        )
    dataset_values = tuple(datasets)
    if len(dataset_values) != 16 or not all(
        isinstance(binding, DatasetBinding) for binding in dataset_values
    ):
        raise RunnerContractError(
            "competition planning requires exactly 16 dataset bindings"
        )
    manifest_hashes = {binding.manifest_sha256 for binding in dataset_values}
    protocol_hashes = {binding.protocol_sha256 for binding in dataset_values}
    design_hashes = {binding.design_sha256 for binding in dataset_values}
    seed_hashes = {binding.seed_source_sha256 for binding in dataset_values}
    if any(
        len(values) != 1
        for values in (manifest_hashes, protocol_hashes, design_hashes, seed_hashes)
    ):
        raise RunnerContractError("dataset bindings do not share one panel authority")
    input_hashes = {
        "dataset_manifest_sha256": next(iter(manifest_hashes)),
        "dataset_design_sha256": next(iter(design_hashes)),
        "dataset_seed_source_sha256": next(iter(seed_hashes)),
        "protocol_sha256": next(iter(protocol_hashes)),
        "method_registry_sha256": authority.method_registry_sha256,
        "selection_contract_sha256": authority.selection_contract_sha256,
        "development_search_sha256": authority.development_search_sha256,
        "ablation_registry_sha256": authority.ablation_registry_sha256,
        "runner_authority_sha256": authority.authority_sha256,
        "execution_environment_sha256": execution_environment_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "base_configuration_sha256": authority.base_configuration_sha256,
        "count_model_config_sha256": authority.count_model_config_sha256,
        "dataset_qc_policy_sha256": authority.dataset_qc_policy_sha256,
        "count_score_manifest_sha256": (
            authority.count_score_manifest_sha256 or "0" * 64
        ),
        "retained_calibration_sha256": (
            authority.retained_calibration_sha256 or "0" * 64
        ),
        "implementation_source_sha256": implementation_source_sha256(),
    }
    entries: list[RunPlanEntry] = []
    authority_by_method: dict[str, tuple[AuthorizedConfiguration, ...]] = {
        method_id: tuple(
            value for value in authority.configurations if value.method_id == method_id
        )
        for method_id in {value.method_id for value in authority.configurations}
    }
    planned_specs = (
        tuple(spec for spec in registry.methods if spec.id == "maskimpute")
        if authority.plan_scope == "revision_candidate_only"
        else tuple(
            spec
            for spec in registry.methods
            if spec.execution_scope == "same_input_required"
        )
    )
    plan_configurations: list[AuthorizedConfiguration] = []
    configurations_by_method: dict[str, tuple[AuthorizedConfiguration, ...]] = {}
    for spec in planned_specs:
        if spec.id == "observed":
            configurations = (AuthorizedConfiguration.registry_default(spec),)
        elif spec.id in {"maskimpute", "capacity-matched-ae"}:
            configurations = authority_by_method.get(spec.id, ())
            if not configurations:
                raise RunnerContractError(
                    f"tracked authority has no configuration for {spec.id}"
                )
        else:
            raise RunnerContractError(
                "legacy competition planning does not accept comparator methods"
            )
        configurations_by_method[spec.id] = configurations
        plan_configurations.extend(configurations)
    ordinal = 0
    for spec in planned_specs:
        seeds: tuple[int | None, ...] = (
            DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
        )
        for configuration in configurations_by_method[spec.id]:
            if configuration.requires_calibration and not authority.maskimpute_ready:
                blocked_reason = "count_score_or_calibration_authority_pending"
            elif (
                configuration.requires_count_score
                and authority.count_score_manifest_status != "ready"
            ):
                blocked_reason = "count_score_authority_pending"
            else:
                blocked_reason = None
            for binding in dataset_values:
                for seed in seeds:
                    ordinal += 1
                    entries.append(
                        RunPlanEntry(
                            ordinal=ordinal,
                            run_id=_run_id(
                                spec,
                                binding,
                                seed,
                                configuration.configuration_sha256,
                            ),
                            method_id=spec.id,
                            dataset_id=binding.dataset_id,
                            source_dataset_sha256=binding.dataset_sha256,
                            mechanism=binding.mechanism,
                            biological_id=binding.biological_id,
                            technical_view=binding.technical_view,
                            model_seed=seed,
                            configuration_id=configuration.configuration_id,
                            configuration_sha256=configuration.configuration_sha256,
                            preflight_status=(
                                "blocked_authority"
                                if blocked_reason is not None
                                else "planned"
                            ),
                            preflight_reason=blocked_reason,
                            configuration_kind=configuration.kind,
                            requires_count_score=configuration.requires_count_score,
                            requires_calibration=configuration.requires_calibration,
                            configuration_payload_sha256=(
                                configuration.configuration_sha256
                            ),
                            configuration_method_identity_sha256=(
                                configuration.configuration_method_identity_sha256
                            ),
                            nonexecution_identity_sha256=(
                                configuration.nonexecution_identity_sha256
                            ),
                        )
                    )
    if authority.plan_scope == "base_full_panel":
        component_counts = {
            "observed": sum(entry.method_id == "observed" for entry in entries),
            "capacity": sum(
                entry.method_id == "capacity-matched-ae" for entry in entries
            ),
            "maskimpute": sum(entry.method_id == "maskimpute" for entry in entries),
            "comparators": sum(
                entry.configuration_kind == "comparator_tuning" for entry in entries
            ),
        }
        if component_counts != {
            "observed": 16,
            "capacity": 48,
            "maskimpute": 1_200,
            "comparators": 1_632,
        }:
            raise RunnerContractError("development plan component denominator differs")
    elif (
        len(entries) != 48
        or len(plan_configurations) != 1
        or any(
            entry.method_id != "maskimpute"
            or entry.configuration_kind != "candidate_search"
            for entry in entries
        )
    ):
        raise RunnerContractError("revision candidate plan denominator differs")
    plan_body = _competition_plan_body(
        schema_version=1,
        input_hashes=input_hashes,
        entries=entries,
        configurations=plan_configurations,
        execution_context=authority.execution_context,
    )
    return CompetitionPlan(
        schema_version=1,
        input_hashes=MappingProxyType(input_hashes),
        entries=tuple(entries),
        plan_sha256=canonical_sha256(plan_body),
        configurations=tuple(plan_configurations),
        execution_context=authority.execution_context,
    )


def _cell_id_sha256(cell_ids: Sequence[str]) -> str:
    payload = bytearray(b"maskimpute-external-cell-ids-v1\0")
    payload.extend(struct.pack("<Q", len(cell_ids)))
    for cell_id in cell_ids:
        encoded = cell_id.encode("utf-8")
        payload.extend(struct.pack("<Q", len(encoded)))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetQCAudit:
    """Exact retained/excluded cell identity binding for one source dataset."""

    excluded_cell_count: int
    excluded_cell_ids_sha256: str
    retained_cell_count: int
    retained_cell_ids_sha256: str
    excluded_cell_ids: tuple[str, ...]
    retained_cell_ids: tuple[str, ...]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "excluded_cell_count": self.excluded_cell_count,
            "excluded_cell_ids_sha256": self.excluded_cell_ids_sha256,
            "retained_cell_count": self.retained_cell_count,
            "retained_cell_ids_sha256": self.retained_cell_ids_sha256,
        }


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Evaluator-private truth plus the sole reusable truth-free method input."""

    binding: DatasetBinding | RegisteredTrajectoryBinding
    audit: DatasetQCAudit
    method_input: MethodInput
    evaluator_dataset: Any


def _prepare_dataset_with_exclusions(
    dataset: Any,
    binding: DatasetBinding | RegisteredTrajectoryBinding,
    policy: DatasetQCPolicy,
    excluded_cell_ids: tuple[str, ...] | None,
) -> PreparedDataset:
    import anndata as ad

    from .methods import prepare_method_input
    from .schema import (
        benchmark_dataset_sha256,
        make_inference_view,
        validate_benchmark_dataset,
    )

    if not isinstance(policy, DatasetQCPolicy):
        raise TypeError("policy must be a DatasetQCPolicy")
    policy.require_fixed_publication_rule()
    from .trajectory_dataset import RegisteredTrajectoryBinding

    if not isinstance(binding, (DatasetBinding, RegisteredTrajectoryBinding)):
        raise TypeError(
            "binding must be a DatasetBinding or RegisteredTrajectoryBinding"
        )
    if not isinstance(dataset, ad.AnnData):
        raise TypeError("dataset must be an AnnData object")
    try:
        validate_benchmark_dataset(dataset)
    except (TypeError, ValueError) as error:
        raise RunnerContractError(
            f"benchmark dataset failed revalidation: {error}"
        ) from error
    if dataset.shape != (binding.cells, binding.genes):
        raise RunnerContractError("benchmark dataset dimensions mismatch its binding")
    if benchmark_dataset_sha256(dataset) != binding.dataset_sha256:
        raise RunnerContractError("benchmark dataset semantic checksum mismatch")
    dataset_ids = dataset.obs["dataset_id"].tolist()
    if not dataset_ids or set(dataset_ids) != {binding.dataset_id}:
        raise RunnerContractError("benchmark dataset ID mismatches its binding")

    cell_ids = tuple(dataset.obs_names.tolist())
    libraries = dataset.obs["library_size"].to_numpy(copy=True)
    own_zero_ids = tuple(
        cell_id
        for cell_id, library in zip(cell_ids, libraries, strict=True)
        if int(library) == 0
    )
    excluded_ids = own_zero_ids if excluded_cell_ids is None else excluded_cell_ids
    if len(excluded_ids) != len(set(excluded_ids)) or any(
        cell_id not in cell_ids for cell_id in excluded_ids
    ):
        raise RunnerContractError("pair-level excluded cell IDs are invalid")
    if not set(own_zero_ids).issubset(excluded_ids):
        raise RunnerContractError("pair-level QC omitted a zero-library cell")
    excluded_set = set(excluded_ids)
    retained_mask = np.asarray(
        [cell_id not in excluded_set for cell_id in cell_ids], dtype=bool
    )
    retained_ids = tuple(
        cell_id
        for cell_id, retained in zip(cell_ids, retained_mask, strict=True)
        if retained
    )
    if len(retained_ids) < policy.minimum_retained_cells:
        raise RunnerContractError(
            "dataset QC requires at least two retained cells after zero-library exclusion"
        )
    audit = DatasetQCAudit(
        excluded_cell_count=len(excluded_ids),
        excluded_cell_ids_sha256=_cell_id_sha256(excluded_ids),
        retained_cell_count=len(retained_ids),
        retained_cell_ids_sha256=_cell_id_sha256(retained_ids),
        excluded_cell_ids=excluded_ids,
        retained_cell_ids=retained_ids,
    )

    # Create the inference view before subsetting so its source hash remains the
    # exact validated source-dataset hash.  Only then apply the same row mask to
    # the truth-free and evaluator-private views.
    full_inference_view = make_inference_view(dataset)
    inference_view = full_inference_view[retained_mask, :].copy()
    evaluator_dataset = dataset[retained_mask, :].copy()
    if (
        inference_view.shape[1] != dataset.shape[1]
        or evaluator_dataset.shape[1] != dataset.shape[1]
    ):
        raise RunnerContractError("dataset QC must not filter genes")
    method_input = prepare_method_input(inference_view)
    if method_input.source_dataset_sha256 != binding.dataset_sha256:
        raise RunnerContractError("QC method input lost its source dataset binding")
    if method_input.obs_ids != audit.retained_cell_ids:
        raise RunnerContractError("QC retained cells differ from the method input")
    if method_input.var_ids != tuple(dataset.var_names.tolist()):
        raise RunnerContractError("QC method input changed the gene set")
    if evaluator_dataset.obs_names.tolist() != list(audit.retained_cell_ids):
        raise RunnerContractError("evaluator and method QC cell subsets differ")
    return PreparedDataset(
        binding=binding,
        audit=audit,
        method_input=method_input,
        evaluator_dataset=evaluator_dataset,
    )


def prepare_dataset_for_execution(
    dataset: Any,
    binding: DatasetBinding,
    policy: DatasetQCPolicy,
) -> PreparedDataset:
    """Prepare an unpaired dataset using only its own zero-library cells.

    Publication-panel execution must use :func:`prepare_dataset_pair_for_execution`
    so paired technical views share one union exclusion.  This single-dataset
    helper remains useful for schema tests and non-paired diagnostic data.
    """

    if not isinstance(binding, DatasetBinding):
        raise TypeError("binding must be a DatasetBinding")
    return _prepare_dataset_with_exclusions(dataset, binding, policy, None)


def prepare_dataset_pair_for_execution(
    first_dataset: Any,
    second_dataset: Any,
    first_binding: DatasetBinding,
    second_binding: DatasetBinding,
    policy: DatasetQCPolicy,
) -> tuple[PreparedDataset, PreparedDataset]:
    """Apply the union of paired-view zero-library IDs identically to both views."""

    import anndata as ad

    if not isinstance(policy, DatasetQCPolicy):
        raise TypeError("policy must be a DatasetQCPolicy")
    policy.require_fixed_publication_rule()
    if not isinstance(first_dataset, ad.AnnData) or not isinstance(
        second_dataset, ad.AnnData
    ):
        raise TypeError("paired datasets must be AnnData objects")
    if not isinstance(first_binding, DatasetBinding) or not isinstance(
        second_binding, DatasetBinding
    ):
        raise TypeError("paired bindings must be DatasetBinding values")
    if (
        first_binding.mechanism != second_binding.mechanism
        or first_binding.biological_id != second_binding.biological_id
        or (first_binding.technical_view, second_binding.technical_view)
        != DEVELOPMENT_VIEWS
        or first_binding.independent_unit_id != second_binding.independent_unit_id
        or first_binding.truth_sha256 != second_binding.truth_sha256
    ):
        raise RunnerContractError(
            "paired dataset bindings do not share one draw and truth"
        )
    first_ids = tuple(first_dataset.obs_names.tolist())
    second_ids = tuple(second_dataset.obs_names.tolist())
    if first_ids != second_ids:
        raise RunnerContractError("paired technical views have different cell ID order")
    if tuple(first_dataset.var_names.tolist()) != tuple(
        second_dataset.var_names.tolist()
    ):
        raise RunnerContractError("paired technical views have different gene ID order")
    first_libraries = first_dataset.obs["library_size"].to_numpy(copy=True)
    second_libraries = second_dataset.obs["library_size"].to_numpy(copy=True)
    union_ids = tuple(
        cell_id
        for cell_id, first_library, second_library in zip(
            first_ids, first_libraries, second_libraries, strict=True
        )
        if int(first_library) == 0 or int(second_library) == 0
    )
    first = _prepare_dataset_with_exclusions(
        first_dataset, first_binding, policy, union_ids
    )
    second = _prepare_dataset_with_exclusions(
        second_dataset, second_binding, policy, union_ids
    )
    if first.audit != second.audit:
        raise RunnerContractError(
            "paired technical views did not retain identical cells"
        )
    return first, second


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One independent parent-side process-tree resource observation."""

    peak_rss_bytes: int | None
    peak_gpu_bytes: int | None
    rss_provenance: str
    gpu_provenance: str

    def __post_init__(self) -> None:
        for name in ("peak_rss_bytes", "peak_gpu_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise RunnerContractError(f"resource sample {name} is invalid")
        if not self.rss_provenance or not self.gpu_provenance:
            raise RunnerContractError("resource sample provenance must be nonempty")


class ResourceSampler(Protocol):
    """Independent parent-side process-tree resource telemetry."""

    def sample(self, process_id: int, *, gpu_required: bool) -> ResourceSample: ...


def _linux_process_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            content = (entry / "stat").read_text(encoding="utf-8")
            suffix = content[content.rindex(")") + 2 :].split()
            parents[int(entry.name)] = int(suffix[1])
        except (OSError, ValueError, IndexError):
            continue
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in result and pid not in result:
                result.add(pid)
                changed = True
    return result


@dataclass(frozen=True, slots=True)
class LinuxProcessTreeResourceSampler:
    """Sample aggregate Linux RSS and nvidia-smi compute-process memory."""

    nvidia_smi_path: Path | None = dataclass_field(
        default_factory=nvidia_smi_executable
    )

    def sample(self, process_id: int, *, gpu_required: bool) -> ResourceSample:
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id <= 0
        ):
            raise ValueError("process_id must be a positive integer")
        process_ids = _linux_process_tree(process_id)
        page_size = os.sysconf("SC_PAGE_SIZE")
        rss_values: list[int] = []
        for pid in process_ids:
            try:
                fields = Path(f"/proc/{pid}/statm").read_text(encoding="utf-8").split()
                rss_values.append(int(fields[1]) * int(page_size))
            except (OSError, ValueError, IndexError):
                continue
        rss = sum(rss_values) if rss_values else None
        if not gpu_required:
            return ResourceSample(
                peak_rss_bytes=rss,
                peak_gpu_bytes=0,
                rss_provenance="linux_proc_process_tree_rss",
                gpu_provenance="not_applicable_cpu_only_method",
            )
        if self.nvidia_smi_path is None:
            return ResourceSample(
                peak_rss_bytes=rss,
                peak_gpu_bytes=None,
                rss_provenance="linux_proc_process_tree_rss",
                gpu_provenance="nvidia_smi_measurement_unavailable",
            )
        try:
            completed = subprocess.run(
                [
                    str(self.nvidia_smi_path),
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                cwd=publication_runtime_working_directory(),
                text=True,
                timeout=5,
            )
            gpu_bytes = 0
            for line in completed.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 2:
                    continue
                if int(fields[0]) in process_ids:
                    gpu_bytes += int(fields[1]) * 1024**2
            gpu: int | None = gpu_bytes
            gpu_provenance = "nvidia_smi_process_tree_used_memory"
        except (OSError, subprocess.SubprocessError, ValueError):
            gpu = None
            gpu_provenance = "nvidia_smi_measurement_unavailable"
        return ResourceSample(
            peak_rss_bytes=rss,
            peak_gpu_bytes=gpu,
            rss_provenance="linux_proc_process_tree_rss",
            gpu_provenance=gpu_provenance,
        )


def method_input_sha256(method_input: MethodInput) -> str:
    """Hash every immutable truth-free input field and its exact count bytes."""

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    digest = hashlib.sha256()
    digest.update(b"maskimpute-method-input-v1\0")
    binding = {
        "source_dataset_sha256": method_input.source_dataset_sha256,
        "obs_ids": method_input.obs_ids,
        "var_ids": method_input.var_ids,
        "shape": method_input.shape,
        "obs_covariates": [asdict(value) for value in method_input.obs_covariates],
        "var_covariates": [asdict(value) for value in method_input.var_covariates],
        "normalization": method_input.normalization,
        "dtype": "<f8",
    }
    digest.update(_canonical_bytes(binding))
    digest.update(np.asarray(method_input.counts, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExecutionAuthorityContext:
    """Tracked truth-free artifact/configuration bindings available to adapters."""

    authority_sha256: str
    base_configuration_json: str
    base_configuration_sha256: str
    count_model_config_json: str
    count_model_config_sha256: str
    count_score_manifest_path: str
    count_score_manifest_sha256: str | None
    retained_calibration_path: str
    retained_calibration_sha256: str | None

    def __post_init__(self) -> None:
        _require_sha256(self.authority_sha256, "execution authority checksum")
        _require_sha256(self.base_configuration_sha256, "base configuration checksum")
        _require_sha256(
            self.count_model_config_sha256, "count-model configuration checksum"
        )
        try:
            base_config = json.loads(
                self.base_configuration_json,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
            count_config = json.loads(
                self.count_model_config_json,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RunnerContractError(
                "count-model configuration JSON is invalid"
            ) from error
        if (
            not isinstance(base_config, dict)
            or self.base_configuration_json.encode() != _canonical_bytes(base_config)
            or canonical_sha256(base_config) != self.base_configuration_sha256
        ):
            raise RunnerContractError(
                "base configuration does not match its tracked checksum"
            )
        if (
            not isinstance(count_config, dict)
            or self.count_model_config_json.encode() != _canonical_bytes(count_config)
            or canonical_sha256(count_config) != self.count_model_config_sha256
        ):
            raise RunnerContractError(
                "count-model configuration does not match its tracked checksum"
            )
        for name in ("count_score_manifest_path", "retained_calibration_path"):
            relative = PurePosixPath(getattr(self, name))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise RunnerContractError(f"{name} is not a safe relative path")
        _require_sha256(
            self.count_score_manifest_sha256,
            "count-score manifest checksum",
            nullable=True,
        )
        _require_sha256(
            self.retained_calibration_sha256,
            "retained calibration checksum",
            nullable=True,
        )


@dataclass(frozen=True, slots=True)
class CalibrationFoldContext:
    """LODO calibration fold required for one development dataset."""

    mechanism: str
    biological_id: str
    technical_view: str
    fold_policy: str = "leave_one_biological_draw_out"

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class CalibrationFoldReceipt:
    """Proof that a calibrated development run used its held-out LODO fold."""

    calibration_artifact_sha256: str
    calibration_context_sha256: str
    mechanism: str
    biological_id: str
    training_manifest_sha256s: tuple[str, ...]
    held_out_manifest_sha256s: tuple[str, ...]
    fold_calibrator_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "calibration_artifact_sha256",
            "calibration_context_sha256",
            "fold_calibrator_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("mechanism", "biological_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise RunnerContractError(f"calibration receipt {name} is invalid")
        for name in (
            "training_manifest_sha256s",
            "held_out_manifest_sha256s",
        ):
            values = getattr(self, name)
            if (
                not values
                or tuple(sorted(values)) != values
                or len(set(values)) != len(values)
            ):
                raise RunnerContractError(
                    f"calibration receipt {name} must be nonempty, unique, and sorted"
                )
            for value in values:
                _require_sha256(value, f"calibration receipt {name}")
        if set(self.training_manifest_sha256s) & set(self.held_out_manifest_sha256s):
            raise RunnerContractError(
                "LODO training and held-out calibration manifests must be disjoint"
            )


def _execution_request_binding(value: Mapping[str, object]) -> str:
    return canonical_sha256(dict(value))


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Closed truth-free request with all configuration/artifact authority in-band."""

    method_spec: MethodSpec
    method_input: MethodInput
    method_input_sha256: str
    model_seed: int | None
    dataset_id: str
    mechanism: str
    biological_id: str
    technical_view: str
    configuration_id: str
    configuration_kind: str
    configuration_sha256: str
    configuration_payload_sha256: str
    configuration_payload_json: str
    registry_method_sha256: str | None
    tuning_authority_file_sha256: str | None
    tuning_authority_payload_sha256: str | None
    source_authority_sha256: str | None
    runtime_lock_sha256: str | None
    environment_registry_sha256: str | None
    configuration_method_identity_sha256: str | None
    nonexecution_identity_sha256: str | None
    execution_authority_sha256: str
    base_configuration_json: str | None
    base_configuration_sha256: str | None
    count_model_config_json: str | None
    count_model_config_sha256: str | None
    count_score_manifest_path: str | None
    count_score_manifest_sha256: str | None
    retained_calibration_path: str | None
    retained_calibration_sha256: str | None
    calibration_usage: str
    calibration_context: CalibrationFoldContext | None
    timeout_seconds: float
    max_rss_bytes: int
    max_gpu_bytes: int
    request_sha256: str

    @classmethod
    def create(
        cls,
        method_spec: MethodSpec,
        method_input: MethodInput,
        *,
        model_seed: int | None,
        configuration: AuthorizedConfiguration,
        authority: RunnerAuthority | ExecutionAuthorityContext,
        mechanism: str,
        biological_id: str,
        technical_view: str,
        dataset_id: str,
        timeout_seconds: int | float,
        calibration_usage: str = "development_holdout",
    ) -> ExecutionRequest:
        if not isinstance(method_spec, MethodSpec):
            raise TypeError("method_spec must be a MethodSpec")
        if not isinstance(method_input, MethodInput):
            raise TypeError("method_input must be a MethodInput")
        if not isinstance(configuration, AuthorizedConfiguration):
            raise TypeError("configuration must be an AuthorizedConfiguration")
        if configuration.kind == "comparator_tuning":
            raise RunnerContractError(
                "legacy comparator request is disabled; use the direct fair-comparator path"
            )
        if configuration.method_id != method_spec.id:
            raise RunnerContractError("configuration method does not match MethodSpec")
        if configuration.kind == "comparator_nonexecution":
            raise RunnerContractError(
                "comparator nonexecution configuration is never executable"
            )
        if calibration_usage not in {
            "development_holdout",
            "retained_all_development",
        }:
            raise RunnerContractError("calibration usage is invalid")
        context = (
            authority.execution_context
            if isinstance(authority, RunnerAuthority)
            else authority
        )
        if not isinstance(context, ExecutionAuthorityContext):
            raise TypeError("authority must provide an ExecutionAuthorityContext")
        if method_spec.stochastic:
            if (
                isinstance(model_seed, bool)
                or not isinstance(model_seed, int)
                or model_seed not in DEVELOPMENT_MODEL_SEEDS
            ):
                raise RunnerContractError(
                    "stochastic development methods require seed 42, 43, or 44"
                )
        elif model_seed is not None:
            raise RunnerContractError("deterministic methods require a null seed")
        for value, name in (
            (mechanism, "mechanism"),
            (biological_id, "biological_id"),
            (technical_view, "technical_view"),
            (dataset_id, "dataset_id"),
        ):
            if not isinstance(value, str) or not value:
                raise RunnerContractError(f"{name} must be a nonempty string")
        timeout = _require_nonnegative_number(timeout_seconds, "timeout_seconds")
        if timeout == 0:
            raise RunnerContractError("timeout_seconds must be positive")
        if configuration.requires_count_score:
            if context.count_score_manifest_sha256 is None:
                raise RunnerContractError(
                    "count-score configuration cannot execute with pending authority"
                )
            score_path: str | None = context.count_score_manifest_path
            score_sha: str | None = context.count_score_manifest_sha256
        else:
            score_path = None
            score_sha = None
        if method_spec.id in {"maskimpute", "capacity-matched-ae"}:
            base_config_json: str | None = context.base_configuration_json
            base_config_sha: str | None = context.base_configuration_sha256
            count_config_json: str | None = context.count_model_config_json
            count_config_sha: str | None = context.count_model_config_sha256
        else:
            base_config_json = None
            base_config_sha = None
            count_config_json = None
            count_config_sha = None
        if (
            configuration.requires_calibration
            and context.retained_calibration_sha256 is None
        ):
            raise RunnerContractError(
                "calibrated development configuration has pending authority"
            )
        if configuration.requires_calibration or (
            method_spec.id in {"maskimpute", "capacity-matched-ae"}
            and context.retained_calibration_sha256 is not None
        ):
            if context.retained_calibration_sha256 is None:  # pragma: no cover
                raise RunnerContractError(
                    "calibrated development configuration has pending authority"
                )
            calibration_path: str | None = context.retained_calibration_path
            calibration_sha: str | None = context.retained_calibration_sha256
            calibration_context: CalibrationFoldContext | None = (
                CalibrationFoldContext(
                    mechanism=mechanism,
                    biological_id=biological_id,
                    technical_view=technical_view,
                )
                if configuration.requires_calibration
                and mechanism == "symsim"
                and calibration_usage == "development_holdout"
                else None
            )
        else:
            calibration_path = None
            calibration_sha = None
            calibration_context = None
        values: dict[str, object] = {
            "method_spec_sha256": canonical_sha256(asdict(method_spec)),
            "method_input_sha256": method_input_sha256(method_input),
            "model_seed": model_seed,
            "dataset_id": dataset_id,
            "mechanism": mechanism,
            "biological_id": biological_id,
            "technical_view": technical_view,
            "configuration_id": configuration.configuration_id,
            "configuration_kind": configuration.kind,
            "configuration_sha256": configuration.configuration_sha256,
            "configuration_payload_sha256": configuration.configuration_sha256,
            "configuration_payload": dict(configuration.payload),
            "registry_method_sha256": configuration.registry_method_sha256,
            "tuning_authority_file_sha256": (
                configuration.tuning_authority_file_sha256
            ),
            "tuning_authority_payload_sha256": (
                configuration.tuning_authority_payload_sha256
            ),
            "source_authority_sha256": configuration.source_authority_sha256,
            "runtime_lock_sha256": configuration.runtime_lock_sha256,
            "environment_registry_sha256": (configuration.environment_registry_sha256),
            "configuration_method_identity_sha256": (
                configuration.configuration_method_identity_sha256
            ),
            "nonexecution_identity_sha256": (
                configuration.nonexecution_identity_sha256
            ),
            "execution_authority_sha256": context.authority_sha256,
            "base_configuration_sha256": base_config_sha,
            "count_model_config_sha256": count_config_sha,
            "count_score_manifest_path": score_path,
            "count_score_manifest_sha256": score_sha,
            "retained_calibration_path": calibration_path,
            "retained_calibration_sha256": calibration_sha,
            "calibration_usage": calibration_usage,
            "calibration_context": (
                None if calibration_context is None else asdict(calibration_context)
            ),
            "timeout_seconds": float(timeout),
            "max_rss_bytes": int(method_spec.resources.max_rss_gib * 1024**3),
            "max_gpu_bytes": int(method_spec.resources.max_gpu_gib * 1024**3),
        }
        return cls(
            method_spec=method_spec,
            method_input=method_input,
            method_input_sha256=str(values["method_input_sha256"]),
            model_seed=model_seed,
            dataset_id=dataset_id,
            mechanism=mechanism,
            biological_id=biological_id,
            technical_view=technical_view,
            configuration_id=configuration.configuration_id,
            configuration_kind=configuration.kind,
            configuration_sha256=configuration.configuration_sha256,
            configuration_payload_sha256=configuration.configuration_sha256,
            configuration_payload_json=configuration.payload_json,
            registry_method_sha256=configuration.registry_method_sha256,
            tuning_authority_file_sha256=(configuration.tuning_authority_file_sha256),
            tuning_authority_payload_sha256=(
                configuration.tuning_authority_payload_sha256
            ),
            source_authority_sha256=configuration.source_authority_sha256,
            runtime_lock_sha256=configuration.runtime_lock_sha256,
            environment_registry_sha256=(configuration.environment_registry_sha256),
            configuration_method_identity_sha256=(
                configuration.configuration_method_identity_sha256
            ),
            nonexecution_identity_sha256=(configuration.nonexecution_identity_sha256),
            execution_authority_sha256=context.authority_sha256,
            base_configuration_json=base_config_json,
            base_configuration_sha256=base_config_sha,
            count_model_config_json=count_config_json,
            count_model_config_sha256=count_config_sha,
            count_score_manifest_path=score_path,
            count_score_manifest_sha256=score_sha,
            retained_calibration_path=calibration_path,
            retained_calibration_sha256=calibration_sha,
            calibration_usage=calibration_usage,
            calibration_context=calibration_context,
            timeout_seconds=float(timeout),
            max_rss_bytes=int(values["max_rss_bytes"]),
            max_gpu_bytes=int(values["max_gpu_bytes"]),
            request_sha256=_execution_request_binding(values),
        )

    def validate_integrity(self) -> None:
        if self.configuration_kind == "comparator_tuning":
            raise RunnerContractError(
                "legacy comparator request is disabled; use the direct fair-comparator path"
            )
        if self.configuration_kind == "comparator_nonexecution":
            raise RunnerContractError(
                "comparator nonexecution configuration is never executable"
            )
        if self.configuration_kind not in {
            "registry",
            "candidate_search",
            "ablation",
        }:
            raise RunnerContractError("execution request configuration kind is invalid")
        _require_sha256(
            self.configuration_payload_sha256,
            "execution request configuration payload checksum",
        )
        component_fields = (
            "registry_method_sha256",
            "tuning_authority_file_sha256",
            "tuning_authority_payload_sha256",
            "source_authority_sha256",
            "runtime_lock_sha256",
            "environment_registry_sha256",
        )
        if (
            any(
                getattr(self, field_name) is not None for field_name in component_fields
            )
            or self.configuration_method_identity_sha256 is not None
            or self.nonexecution_identity_sha256 is not None
        ):
            raise RunnerContractError(
                "legacy execution request forbids comparator identity fields"
            )
        if self.calibration_usage not in {
            "development_holdout",
            "retained_all_development",
        }:
            raise RunnerContractError("calibration usage is invalid")
        configuration = json.loads(self.configuration_payload_json)
        base_configuration = (
            None
            if self.base_configuration_json is None
            else json.loads(self.base_configuration_json)
        )
        count_model_configuration = (
            None
            if self.count_model_config_json is None
            else json.loads(self.count_model_config_json)
        )
        values: dict[str, object] = {
            "method_spec_sha256": canonical_sha256(asdict(self.method_spec)),
            "method_input_sha256": method_input_sha256(self.method_input),
            "model_seed": self.model_seed,
            "dataset_id": self.dataset_id,
            "mechanism": self.mechanism,
            "biological_id": self.biological_id,
            "technical_view": self.technical_view,
            "configuration_id": self.configuration_id,
            "configuration_kind": self.configuration_kind,
            "configuration_sha256": self.configuration_sha256,
            "configuration_payload_sha256": self.configuration_payload_sha256,
            "configuration_payload": configuration,
            "registry_method_sha256": self.registry_method_sha256,
            "tuning_authority_file_sha256": self.tuning_authority_file_sha256,
            "tuning_authority_payload_sha256": (self.tuning_authority_payload_sha256),
            "source_authority_sha256": self.source_authority_sha256,
            "runtime_lock_sha256": self.runtime_lock_sha256,
            "environment_registry_sha256": self.environment_registry_sha256,
            "configuration_method_identity_sha256": (
                self.configuration_method_identity_sha256
            ),
            "nonexecution_identity_sha256": self.nonexecution_identity_sha256,
            "execution_authority_sha256": self.execution_authority_sha256,
            "base_configuration_sha256": self.base_configuration_sha256,
            "count_model_config_sha256": self.count_model_config_sha256,
            "count_score_manifest_path": self.count_score_manifest_path,
            "count_score_manifest_sha256": self.count_score_manifest_sha256,
            "retained_calibration_path": self.retained_calibration_path,
            "retained_calibration_sha256": self.retained_calibration_sha256,
            "calibration_usage": self.calibration_usage,
            "calibration_context": (
                None
                if self.calibration_context is None
                else asdict(self.calibration_context)
            ),
            "timeout_seconds": self.timeout_seconds,
            "max_rss_bytes": self.max_rss_bytes,
            "max_gpu_bytes": self.max_gpu_bytes,
        }
        if (
            self.configuration_payload_json.encode() != _canonical_bytes(configuration)
            or canonical_sha256(configuration) != self.configuration_sha256
            or self.configuration_payload_sha256 != self.configuration_sha256
            or (
                base_configuration is None
                and self.base_configuration_sha256 is not None
            )
            or (
                base_configuration is not None
                and (
                    self.base_configuration_json.encode()
                    != _canonical_bytes(base_configuration)
                    or canonical_sha256(base_configuration)
                    != self.base_configuration_sha256
                )
            )
            or (
                count_model_configuration is None
                and self.count_model_config_sha256 is not None
            )
            or (
                count_model_configuration is not None
                and (
                    self.count_model_config_json.encode()
                    != _canonical_bytes(count_model_configuration)
                    or canonical_sha256(count_model_configuration)
                    != self.count_model_config_sha256
                )
            )
            or self.method_input_sha256 != values["method_input_sha256"]
            or self.request_sha256 != _execution_request_binding(values)
        ):
            raise RunnerContractError("execution request authority checksum mismatch")


@dataclass(frozen=True, slots=True)
class FinalComparatorExecutionRequest:
    """Narrow direct request for one frozen final/trajectory comparator row."""

    method_spec: MethodSpec
    method_input: MethodInput
    model_seed: int | None
    dataset_id: str
    mechanism: str
    biological_id: str
    technical_view: str
    configuration: BoundComparatorConfiguration
    execution_authority: ExecutionAuthorityContext
    timeout_seconds: float
    max_rss_bytes: int
    max_gpu_bytes: int

    @classmethod
    def create(
        cls,
        method_spec: MethodSpec,
        method_input: MethodInput,
        *,
        model_seed: int | None,
        configuration: BoundComparatorConfiguration,
        authority: ExecutionAuthorityContext,
        mechanism: str,
        biological_id: str,
        technical_view: str,
        dataset_id: str,
        timeout_seconds: int | float,
    ) -> FinalComparatorExecutionRequest:
        timeout = _require_nonnegative_number(timeout_seconds, "timeout_seconds")
        if timeout == 0:
            raise RunnerContractError("timeout_seconds must be positive")
        request = cls(
            method_spec=method_spec,
            method_input=method_input,
            model_seed=model_seed,
            dataset_id=dataset_id,
            mechanism=mechanism,
            biological_id=biological_id,
            technical_view=technical_view,
            configuration=configuration,
            execution_authority=authority,
            timeout_seconds=float(timeout),
            max_rss_bytes=int(method_spec.resources.max_rss_gib * 1024**3),
            max_gpu_bytes=int(method_spec.resources.max_gpu_gib * 1024**3),
        )
        request.validate_integrity()
        return request

    def validate_integrity(self) -> None:
        from .comparator_tuning import (
            BoundComparatorConfiguration,
            ComparatorTuningError,
            _validate_bound_selection_configuration,
        )
        from .direct_values import direct_equal

        if not isinstance(self.method_spec, MethodSpec):
            raise TypeError("method_spec must be a MethodSpec")
        if not isinstance(self.method_input, MethodInput):
            raise TypeError("method_input must be a MethodInput")
        if not isinstance(self.configuration, BoundComparatorConfiguration):
            raise TypeError("configuration must be a BoundComparatorConfiguration")
        if not isinstance(self.execution_authority, ExecutionAuthorityContext):
            raise TypeError("authority must be an ExecutionAuthorityContext")
        for value, name in (
            (self.dataset_id, "dataset_id"),
            (self.mechanism, "mechanism"),
            (self.biological_id, "biological_id"),
            (self.technical_view, "technical_view"),
        ):
            if not isinstance(value, str) or not value:
                raise RunnerContractError(f"{name} must be a nonempty string")
        if self.method_spec.stochastic:
            if type(self.model_seed) is not int or self.model_seed not in (
                DEVELOPMENT_MODEL_SEEDS
            ):
                raise RunnerContractError(
                    "stochastic final comparators require seed 42, 43, or 44"
                )
        elif self.model_seed is not None:
            raise RunnerContractError(
                "deterministic final comparators require a null seed"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
            or type(self.max_rss_bytes) is not int
            or self.max_rss_bytes < 0
            or type(self.max_gpu_bytes) is not int
            or self.max_gpu_bytes < 0
            or self.timeout_seconds != float(self.method_spec.resources.timeout_seconds)
            or self.max_rss_bytes
            != int(self.method_spec.resources.max_rss_gib * 1024**3)
            or self.max_gpu_bytes
            != int(self.method_spec.resources.max_gpu_gib * 1024**3)
        ):
            raise RunnerContractError("final comparator resource authority is invalid")
        try:
            _validate_bound_selection_configuration(self.configuration)
            decoded = self.configuration.configuration.decode()
        except (ComparatorTuningError, TypeError, ValueError) as error:
            raise RunnerContractError(
                "final comparator configuration is invalid"
            ) from error
        if (
            not direct_equal(
                self.configuration.method,
                comparator_method_binding(self.method_spec),
            )
            or self.configuration.configuration.method_id != self.method_spec.id
            or self.configuration.configuration.configuration_id == "registry-default"
            or encode_comparator_configuration(decoded)
            != dict(self.configuration.configuration.payload)
        ):
            raise RunnerContractError(
                "final comparator configuration differs from method authority"
            )


@dataclass(frozen=True, slots=True)
class AdapterOutcome:
    """Measured adapter result before evaluator-side conversion and metrics."""

    status: str
    execution: AdapterExecution | DirectAdapterExecution | None
    reason: str | None
    runtime_seconds: int | float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    stdout: bytes
    stderr: bytes
    rss_measurement: str = "executor_reported_unverified"
    gpu_measurement: str = "executor_reported_unverified"
    calibration_fold_receipt: CalibrationFoldReceipt | None = None

    def __post_init__(self) -> None:
        if self.status not in _OUTCOME_STATUSES:
            raise RunnerContractError(f"unknown adapter outcome status: {self.status}")
        _require_nonnegative_number(self.runtime_seconds, "adapter runtime")
        for name in ("peak_rss_bytes", "peak_gpu_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RunnerContractError(f"{name} must be a nonnegative integer")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise RunnerContractError("adapter stdout and stderr must be exact bytes")
        if not self.rss_measurement or not self.gpu_measurement:
            raise RunnerContractError("adapter measurement provenance must be nonempty")
        if self.calibration_fold_receipt is not None and not isinstance(
            self.calibration_fold_receipt, CalibrationFoldReceipt
        ):
            raise RunnerContractError("calibration fold receipt is noncanonical")
        if self.status == "completed":
            if (
                not isinstance(
                    self.execution, (AdapterExecution, DirectAdapterExecution)
                )
                or self.reason is not None
            ):
                raise RunnerContractError(
                    "completed adapter outcome requires execution and no reason"
                )
        elif (
            self.execution is not None
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise RunnerContractError(
                "non-completed adapter outcome requires a reason and no execution"
            )

    @classmethod
    def completed(
        cls,
        execution: AdapterExecution | DirectAdapterExecution,
        *,
        runtime_seconds: int | float,
        peak_rss_bytes: int,
        peak_gpu_bytes: int,
        rss_measurement: str = "executor_reported_unverified",
        gpu_measurement: str = "executor_reported_unverified",
        calibration_fold_receipt: CalibrationFoldReceipt | None = None,
    ) -> AdapterOutcome:
        return cls(
            status="completed",
            execution=execution,
            reason=None,
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss_bytes,
            peak_gpu_bytes=peak_gpu_bytes,
            stdout=execution.stdout,
            stderr=execution.stderr,
            rss_measurement=rss_measurement,
            gpu_measurement=gpu_measurement,
            calibration_fold_receipt=calibration_fold_receipt,
        )

    @classmethod
    def _noncompleted(
        cls,
        status: str,
        reason: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        runtime_seconds: int | float = 0,
        peak_rss_bytes: int = 0,
        peak_gpu_bytes: int = 0,
        rss_measurement: str = "executor_reported_unverified",
        gpu_measurement: str = "executor_reported_unverified",
    ) -> AdapterOutcome:
        return cls(
            status=status,
            execution=None,
            reason=reason,
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss_bytes,
            peak_gpu_bytes=peak_gpu_bytes,
            stdout=stdout,
            stderr=stderr,
            rss_measurement=rss_measurement,
            gpu_measurement=gpu_measurement,
        )

    @classmethod
    def unavailable(cls, reason: str, **measurements: object) -> AdapterOutcome:
        return cls._noncompleted("unavailable", reason, **measurements)

    @classmethod
    def failed(cls, reason: str, **measurements: object) -> AdapterOutcome:
        return cls._noncompleted("failed", reason, **measurements)

    @classmethod
    def timeout(cls, reason: str = "timeout", **measurements: object) -> AdapterOutcome:
        return cls._noncompleted("timeout", reason, **measurements)

    @classmethod
    def resource_exceeded(cls, reason: str, **measurements: object) -> AdapterOutcome:
        return cls._noncompleted("resource_exceeded", reason, **measurements)

    @classmethod
    def infrastructure_error(
        cls, reason: str, **measurements: object
    ) -> AdapterOutcome:
        return cls._noncompleted("infrastructure_error", reason, **measurements)

    @classmethod
    def blocked_authority(cls, reason: str, **measurements: object) -> AdapterOutcome:
        return cls._noncompleted("blocked_authority", reason, **measurements)

    @classmethod
    def budget_exhausted(cls, reason: str, **measurements: object) -> AdapterOutcome:
        return cls._noncompleted("budget_exhausted", reason, **measurements)


def enforce_calibration_fold_receipt(
    request: ExecutionRequest,
    outcome: AdapterOutcome,
) -> AdapterOutcome:
    """Fail a calibrated completion unless its LODO fold proof matches exactly."""

    if not isinstance(request, ExecutionRequest):
        raise TypeError("request must be an ExecutionRequest")
    if not isinstance(outcome, AdapterOutcome):
        raise TypeError("outcome must be an AdapterOutcome")
    context = request.calibration_context
    receipt = outcome.calibration_fold_receipt
    if context is None:
        if receipt is not None:
            return AdapterOutcome.failed(
                "unexpected_calibration_fold_receipt",
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                runtime_seconds=outcome.runtime_seconds,
                peak_rss_bytes=outcome.peak_rss_bytes,
                peak_gpu_bytes=outcome.peak_gpu_bytes,
                rss_measurement=outcome.rss_measurement,
                gpu_measurement=outcome.gpu_measurement,
            )
        return outcome
    if outcome.status != "completed":
        return outcome
    if receipt is None:
        reason = "lodo_calibration_fold_receipt_missing"
    elif (
        receipt.calibration_artifact_sha256 != request.retained_calibration_sha256
        or receipt.calibration_context_sha256 != context.sha256
        or receipt.mechanism != context.mechanism
        or receipt.biological_id != context.biological_id
    ):
        reason = "lodo_calibration_fold_receipt_mismatch"
    else:
        return outcome
    return AdapterOutcome.failed(
        reason,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        runtime_seconds=outcome.runtime_seconds,
        peak_rss_bytes=outcome.peak_rss_bytes,
        peak_gpu_bytes=outcome.peak_gpu_bytes,
        rss_measurement=outcome.rss_measurement,
        gpu_measurement=outcome.gpu_measurement,
    )


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Evaluator decision for one configuration attempt."""

    authorized: bool
    reason: str | None
    remaining_seconds: float
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class DirectRevisionExecutionRequest:
    """Complete direct values and enforced limits for one revision attempt."""

    identity: object
    descriptor: object
    configuration: object
    method_spec: MethodSpec
    method_input: MethodInput
    timeout_seconds: float
    max_rss_bytes: int
    max_gpu_memory_bytes: int

    def __post_init__(self) -> None:
        _validate_direct_revision_adapter_values(
            self.identity,
            self.descriptor,
            self.configuration,
            self.method_spec,
            self.method_input,
            self.timeout_seconds,
            self.max_rss_bytes,
            self.max_gpu_memory_bytes,
        )

    @property
    def max_gpu_bytes(self) -> int:
        """Expose the accepted monitor's generic GPU-ceiling interface."""

        return self.max_gpu_memory_bytes


@dataclass(frozen=True, slots=True)
class DirectRevisionMaskImputeOutcome:
    """One direct MaskImpute adapter outcome before evaluator-only metrics."""

    status: str
    execution: object | None
    reason: str | None
    runtime_seconds: int | float | None
    peak_rss_bytes: int | None
    peak_gpu_bytes: int | None
    stdout: bytes
    stderr: bytes
    rss_measurement: str = "executor_reported_unverified"
    gpu_measurement: str = "executor_reported_unverified"

    def __post_init__(self) -> None:
        from .methods import DirectMaskImputeExecution

        if self.status not in _OUTCOME_STATUSES:
            raise RunnerContractError("direct revision outcome status is invalid")
        measurements = (
            self.runtime_seconds,
            self.peak_rss_bytes,
            self.peak_gpu_bytes,
        )
        if all(value is None for value in measurements):
            if self.status != "completed":
                raise RunnerContractError(
                    "terminal direct revision outcome lacks measurements"
                )
        elif any(value is None for value in measurements):
            raise RunnerContractError("direct revision measurements are incomplete")
        else:
            _require_nonnegative_number(
                self.runtime_seconds,
                "direct revision runtime",
            )
            if any(
                type(value) is not int or value < 0
                for value in (self.peak_rss_bytes, self.peak_gpu_bytes)
            ):
                raise RunnerContractError("direct revision resource value is invalid")
        if type(self.stdout) is not bytes or type(self.stderr) is not bytes:
            raise RunnerContractError("direct revision streams must be exact bytes")
        if not self.rss_measurement or not self.gpu_measurement:
            raise RunnerContractError("direct revision measurement code is absent")
        if self.status == "completed":
            if (
                not isinstance(self.execution, DirectMaskImputeExecution)
                or self.reason is not None
                or self.stdout != self.execution.stdout
                or self.stderr != self.execution.stderr
            ):
                raise RunnerContractError(
                    "completed direct revision outcome is invalid"
                )
        elif (
            self.execution is not None
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise RunnerContractError("terminal direct revision outcome is invalid")

    @classmethod
    def completed(
        cls,
        execution: object,
        *,
        runtime_seconds: int | float | None = None,
        peak_rss_bytes: int | None = None,
        peak_gpu_bytes: int | None = None,
        rss_measurement: str = "executor_reported_unverified",
        gpu_measurement: str = "executor_reported_unverified",
    ) -> DirectRevisionMaskImputeOutcome:
        from .methods import DirectMaskImputeExecution

        if not isinstance(execution, DirectMaskImputeExecution):
            raise TypeError("execution must be a DirectMaskImputeExecution")
        return cls(
            status="completed",
            execution=execution,
            reason=None,
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss_bytes,
            peak_gpu_bytes=peak_gpu_bytes,
            stdout=execution.stdout,
            stderr=execution.stderr,
            rss_measurement=rss_measurement,
            gpu_measurement=gpu_measurement,
        )

    @classmethod
    def terminal(
        cls,
        status: str,
        reason: str,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        runtime_seconds: int | float = 0.0,
        peak_rss_bytes: int = 0,
        peak_gpu_bytes: int = 0,
        rss_measurement: str = "executor_reported_unverified",
        gpu_measurement: str = "executor_reported_unverified",
    ) -> DirectRevisionMaskImputeOutcome:
        if status == "completed":
            raise ValueError("terminal direct revision status cannot be completed")
        return cls(
            status=status,
            execution=None,
            reason=reason,
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss_bytes,
            peak_gpu_bytes=peak_gpu_bytes,
            stdout=stdout,
            stderr=stderr,
            rss_measurement=rss_measurement,
            gpu_measurement=gpu_measurement,
        )

    @classmethod
    def unavailable(
        cls, reason: str, **measurements: object
    ) -> DirectRevisionMaskImputeOutcome:
        return cls.terminal("unavailable", reason, **measurements)

    @classmethod
    def failed(
        cls, reason: str, **measurements: object
    ) -> DirectRevisionMaskImputeOutcome:
        return cls.terminal("failed", reason, **measurements)

    @classmethod
    def timeout(
        cls, reason: str = "timeout", **measurements: object
    ) -> DirectRevisionMaskImputeOutcome:
        return cls.terminal("timeout", reason, **measurements)

    @classmethod
    def resource_exceeded(
        cls, reason: str, **measurements: object
    ) -> DirectRevisionMaskImputeOutcome:
        return cls.terminal("resource_exceeded", reason, **measurements)

    @classmethod
    def infrastructure_error(
        cls, reason: str, **measurements: object
    ) -> DirectRevisionMaskImputeOutcome:
        return cls.terminal("infrastructure_error", reason, **measurements)


def _validate_direct_revision_adapter_values(
    identity: object,
    descriptor: object,
    configuration: object,
    spec: MethodSpec,
    method_input: MethodInput,
    timeout_seconds: int | float,
    max_rss_bytes: int,
    max_gpu_memory_bytes: int,
) -> dict[str, object]:
    """Revalidate complete direct revision values and return the plain payload."""

    from .direct_values import direct_equal, direct_json_value
    from .fair_comparator_plan import (
        ComparatorRunIdentity,
        DirectAuthorizedConfiguration,
        PreparedInputDescriptor,
    )

    if not isinstance(identity, ComparatorRunIdentity):
        raise TypeError("identity must be a ComparatorRunIdentity")
    if not isinstance(descriptor, PreparedInputDescriptor):
        raise TypeError("descriptor must be a PreparedInputDescriptor")
    if not isinstance(configuration, DirectAuthorizedConfiguration):
        raise TypeError("configuration must be a DirectAuthorizedConfiguration")
    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    payload = direct_json_value(configuration.payload, payload=True)
    if type(payload) is not dict:
        raise RunnerContractError("direct revision payload is not an object")
    counts = method_input.counts
    if (
        not direct_equal(identity.method, configuration.method)
        or not direct_equal(identity.method, comparator_method_binding(spec))
        or identity.method.method_id != "maskimpute"
        or configuration.configuration_kind != "candidate_search"
        or identity.configuration_kind != configuration.configuration_kind
        or identity.configuration_id != configuration.configuration_id
        or not direct_equal(identity.configuration_payload, configuration.payload)
        or identity.dataset_id != descriptor.dataset_id
        or identity.mechanism != descriptor.mechanism
        or identity.technical_view != descriptor.technical_view
        or identity.mask_seed != descriptor.mask_seed
        or identity.model_seed not in DEVELOPMENT_MODEL_SEEDS
        or descriptor.shape != method_input.shape
        or descriptor.cell_ids != method_input.obs_ids
        or descriptor.gene_ids != method_input.var_ids
        or descriptor.dtype != "<f8"
        or descriptor.total_count != float(counts.sum(dtype=np.float64))
        or descriptor.nonzero_count != int(np.count_nonzero(counts))
        or descriptor.minimum != float(counts.min())
        or descriptor.maximum != float(counts.max())
        or not configuration.requires_count_score
        or not configuration.requires_calibration
    ):
        raise RunnerContractError("direct revision adapter values differ")
    timeout = _require_nonnegative_number(timeout_seconds, "direct revision timeout")
    if timeout <= 0 or timeout > spec.resources.timeout_seconds:
        raise RunnerContractError(
            "direct revision timeout differs from method authority"
        )
    expected_rss = int(spec.resources.max_rss_gib * 1024**3)
    expected_gpu = int(spec.resources.max_gpu_gib * 1024**3)
    if (
        type(max_rss_bytes) is not int
        or max_rss_bytes != expected_rss
        or type(max_gpu_memory_bytes) is not int
        or max_gpu_memory_bytes != expected_gpu
    ):
        raise RunnerContractError(
            "direct revision resource ceilings differ from method authority"
        )
    return payload


def _direct_revision_components(
    payload: Mapping[str, object],
    model_seed: int,
) -> tuple[object, object, object | None]:
    """Decode the exact activated v28/v29 payload without legacy summaries."""

    from maskimpute import MaskImputeConfig
    from maskimpute.nb_model import NegativeBinomialDecoderConfig
    from maskimpute.structure import StructurePenaltyConfig

    version = payload.get("method_version")
    expected_fields = {
        "method_version",
        "decoder",
        "encoder_mode",
        "output_policy",
        "score_policy",
        "hyperparameters",
        "decoder_hyperparameters",
    }
    if version == "v29":
        expected_fields.add("structure_hyperparameters")
    if (
        version not in {"v28", "v29"}
        or set(payload) != expected_fields
        or payload.get("decoder") != "negative_binomial"
        or payload.get("encoder_mode") != "explicit_mask"
        or payload.get("output_policy") != "selective"
        or payload.get("score_policy") != "retained_development_calibrator"
    ):
        raise RunnerContractError("direct revision candidate payload differs")
    hyperparameters = payload.get("hyperparameters")
    decoder_values = payload.get("decoder_hyperparameters")
    if not isinstance(hyperparameters, Mapping) or not isinstance(
        decoder_values, Mapping
    ):
        raise RunnerContractError("direct revision hyperparameters are invalid")
    if set(decoder_values) != set(NegativeBinomialDecoderConfig().to_dict()):
        raise RunnerContractError("direct revision decoder fields differ")
    try:
        config = MaskImputeConfig(**dict(hyperparameters), seed=model_seed)
        decoder = NegativeBinomialDecoderConfig(**dict(decoder_values))
        structure = None
        if version == "v29":
            structure_values = payload.get("structure_hyperparameters")
            if not isinstance(structure_values, Mapping) or set(
                structure_values
            ) != set(StructurePenaltyConfig().to_dict()):
                raise RunnerContractError("direct revision structure fields differ")
            structure = StructurePenaltyConfig(**dict(structure_values))
    except (TypeError, ValueError) as error:
        raise RunnerContractError(
            "direct revision payload values are invalid"
        ) from error
    return config, decoder, structure


@dataclass(frozen=True, slots=True)
class DirectRevisionMaskImputeChildExecutor:
    """Run only the numerical fit inside the measured direct child process."""

    calibration_payload: object
    count_model_config: object
    device: str = "cuda"
    numerical_adapter: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        from maskimpute import PreZeroCountModelConfig
        from maskimpute.calibration import CalibrationArtifact

        try:
            CalibrationArtifact(self.calibration_payload)
        except (TypeError, ValueError) as error:
            raise TypeError("calibration_payload must define a calibration") from error
        if type(self.count_model_config) is not PreZeroCountModelConfig:
            raise TypeError("count_model_config must be a PreZeroCountModelConfig")
        if not isinstance(self.device, str) or not self.device:
            raise TypeError("device must be a nonempty string")
        if self.numerical_adapter is not None and not callable(self.numerical_adapter):
            raise TypeError("numerical_adapter must be callable or None")

    def __call__(
        self, request: DirectRevisionExecutionRequest
    ) -> DirectRevisionMaskImputeOutcome:
        from .methods import AdapterUnavailableError, run_revision_maskimpute_direct

        if not isinstance(request, DirectRevisionExecutionRequest):
            raise TypeError("request must be a DirectRevisionExecutionRequest")
        payload = _validate_direct_revision_adapter_values(
            request.identity,
            request.descriptor,
            request.configuration,
            request.method_spec,
            request.method_input,
            request.timeout_seconds,
            request.max_rss_bytes,
            request.max_gpu_memory_bytes,
        )
        identity = request.identity
        assert isinstance(identity.model_seed, int)
        config, decoder, structure = _direct_revision_components(
            payload,
            identity.model_seed,
        )
        numerical_adapter = (
            run_revision_maskimpute_direct
            if self.numerical_adapter is None
            else self.numerical_adapter
        )
        try:
            from maskimpute.calibration import CalibrationArtifact

            execution = numerical_adapter(
                request.method_spec,
                request.method_input,
                calibration_artifact=CalibrationArtifact(self.calibration_payload),
                seed=identity.model_seed,
                config=config,
                count_model_config=self.count_model_config,
                decoder_config=decoder,
                structure_config=structure,
                device=self.device,
                development_mechanism=identity.mechanism,
                development_biological_id=identity.biological_id,
            )
        except AdapterUnavailableError as error:
            return DirectRevisionMaskImputeOutcome.terminal(
                "unavailable",
                error.reason_code,
                stdout=error.stdout,
                stderr=error.stderr,
            )
        except TimeoutError:
            return DirectRevisionMaskImputeOutcome.terminal("timeout", "timeout")
        except MemoryError:
            return DirectRevisionMaskImputeOutcome.terminal(
                "resource_exceeded",
                "memory_limit_exceeded",
            )
        except Exception:
            return DirectRevisionMaskImputeOutcome.terminal(
                "failed",
                "adapter_exception",
            )
        return DirectRevisionMaskImputeOutcome.completed(
            execution,
        )


@dataclass(frozen=True, slots=True)
class DirectRevisionMaskImputeAdapter:
    """Apply direct identity, deadline, and resource limits around MaskImpute."""

    calibration_artifact: object
    count_model_config: object
    device: str = "cuda"
    numerical_adapter: Callable[..., object] | None = None
    resource_sampler: object = dataclass_field(
        default_factory=LinuxProcessTreeResourceSampler
    )
    poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        from maskimpute import PreZeroCountModelConfig
        from maskimpute.calibration import CalibrationArtifact

        if type(self.calibration_artifact) is not CalibrationArtifact:
            raise TypeError("calibration_artifact must be a CalibrationArtifact")
        if type(self.count_model_config) is not PreZeroCountModelConfig:
            raise TypeError("count_model_config must be a PreZeroCountModelConfig")
        if not isinstance(self.device, str) or not self.device:
            raise TypeError("device must be a nonempty string")
        if self.numerical_adapter is not None and not callable(self.numerical_adapter):
            raise TypeError("numerical_adapter must be callable or None")
        if not hasattr(self.resource_sampler, "sample"):
            raise TypeError("resource_sampler must implement sample")
        if (
            isinstance(self.poll_interval_seconds, bool)
            or not isinstance(self.poll_interval_seconds, (int, float))
            or not math.isfinite(self.poll_interval_seconds)
            or self.poll_interval_seconds <= 0
        ):
            raise ValueError("poll_interval_seconds must be positive and finite")

    def __call__(
        self,
        identity: object,
        descriptor: object,
        configuration: object,
        spec: MethodSpec,
        method_input: MethodInput,
        timeout_seconds: int | float,
        max_rss_bytes: int,
        max_gpu_memory_bytes: int,
    ) -> DirectRevisionMaskImputeOutcome:
        request = DirectRevisionExecutionRequest(
            identity=identity,
            descriptor=descriptor,
            configuration=configuration,
            method_spec=spec,
            method_input=method_input,
            timeout_seconds=float(timeout_seconds),
            max_rss_bytes=max_rss_bytes,
            max_gpu_memory_bytes=max_gpu_memory_bytes,
        )
        child_executor = DirectRevisionMaskImputeChildExecutor(
            calibration_payload=self.calibration_artifact.to_dict(),
            count_model_config=self.count_model_config,
            device=self.device,
            numerical_adapter=self.numerical_adapter,
        )
        return execute_direct_revision_adapter_in_spawned_process(
            request,
            child_executor,
            poll_interval_seconds=self.poll_interval_seconds,
            resource_sampler=self.resource_sampler,
        )


def _store_direct_revision_p_pre_zero(
    checkpoint_directory: Path,
    run_id: str,
    probability: np.ndarray,
) -> object:
    """Store one probability matrix create-only and compare complete bytes on replay."""

    from .fair_comparator_execution import DirectPreZeroEvidence

    raw = np.asarray(probability, dtype="<f8", order="C").tobytes(order="C")
    compressed = zlib.compress(raw)
    relative_path = Path("runs") / f"{run_id}.p-pre-zero-f64.zlib"
    evidence_path = checkpoint_directory / relative_path
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            evidence_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError:
        try:
            existing = evidence_path.read_bytes()
        except OSError as error:
            raise RunnerContractError(
                "revision p_pre_zero evidence is unreadable"
            ) from error
        if existing != compressed:
            raise RunnerContractError(
                "revision p_pre_zero evidence differs from storage"
            )
    else:
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(compressed)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                evidence_path.unlink()
            except OSError:
                pass
            raise
    return DirectPreZeroEvidence(
        applicable=True,
        status="completed",
        reason=None,
        shape=probability.shape,
        dtype="<f8",
        encoding="zlib",
        path=relative_path.as_posix(),
        compressed_byte_count=len(compressed),
    )


def _evaluate_direct_revision_outcome(
    entry: object,
    prepared: PreparedDataset,
    outcome: DirectRevisionMaskImputeOutcome,
    checkpoint_directory: Path,
) -> DirectEvaluatedAttempt:
    """Evaluate exact direct MaskImpute output without legacy result helpers."""

    from .fair_comparator_execution import (
        DIRECT_RECONSTRUCTION_METRICS,
        DirectEvaluatedAttempt,
        DirectLogReceipt,
        DirectMetricRow,
        DirectPreZeroEvidence,
        DirectRunResult,
        _canonical_measurement_code,
        _canonical_metric_reason,
        _canonical_terminal_reason,
    )
    from .fair_comparator_plan import DirectPlanEntry
    from .methods import (
        AdapterUnavailableError,
        DirectMaskImputeExecution,
        count_equivalent_to_log2_cp10k,
        maskimpute_to_evaluator_counts,
    )
    from .metrics import reconstruction_metrics

    if not isinstance(entry, DirectPlanEntry):
        raise TypeError("entry must be a DirectPlanEntry")
    if not isinstance(outcome, DirectRevisionMaskImputeOutcome):
        raise TypeError("outcome must be a DirectRevisionMaskImputeOutcome")
    status = outcome.status
    reason = outcome.reason
    native: np.ndarray | None = None
    evaluator: np.ndarray | None = None
    if status == "completed":
        assert isinstance(outcome.execution, DirectMaskImputeExecution)
        output = outcome.execution.output
        if (
            output.method_id != "maskimpute"
            or output.output_scale != entry.identity.method.output_scale
            or output.obs_ids != prepared.method_input.obs_ids
            or output.var_ids != prepared.method_input.var_ids
            or output.shape != prepared.method_input.shape
        ):
            raise RunnerContractError("direct revision output differs from request")
        native = np.array(output.matrix, dtype=np.float64, copy=True, order="C")
        try:
            evaluator = np.array(
                count_equivalent_to_log2_cp10k(
                    maskimpute_to_evaluator_counts(prepared.method_input, native)
                ),
                dtype=np.float64,
                copy=True,
                order="C",
            )
        except AdapterUnavailableError as error:
            status = "unavailable"
            reason = error.reason_code
        except (TypeError, ValueError, OverflowError):
            status = "unavailable"
            reason = "evaluator_conversion_invalid"
    reason = _canonical_terminal_reason(status, reason)
    if evaluator is None:
        if reason is None:  # pragma: no cover - outcome contract
            raise AssertionError("terminal direct revision lacks a reason")
        metrics = tuple(
            DirectMetricRow(
                identity=entry.identity,
                metric=name,
                value=None,
                n=0,
                status=status,
                reason=reason,
            )
            for name in DIRECT_RECONSTRUCTION_METRICS
        )
        native = None
        p_pre_zero = DirectPreZeroEvidence(
            applicable=True,
            status=status,
            reason=reason,
            shape=None,
            dtype=None,
            encoding=None,
            path=None,
            compressed_byte_count=0,
        )
    else:
        observed, truth, truth_kind, marker_mask = _evaluator_targets(prepared)
        values = reconstruction_metrics(
            evaluator,
            observed,
            truth,
            marker_genes=marker_mask,
            truth_kind=truth_kind,
        )
        metrics = tuple(
            DirectMetricRow(
                identity=entry.identity,
                metric=name,
                value=(
                    None if values[name].value is None else float(values[name].value)
                ),
                n=int(values[name].n),
                status=("unavailable" if values[name].value is None else "completed"),
                reason=_canonical_metric_reason(
                    "unavailable" if values[name].value is None else "completed",
                    values[name].reason,
                ),
            )
            for name in DIRECT_RECONSTRUCTION_METRICS
        )
        assert isinstance(outcome.execution, DirectMaskImputeExecution)
        p_pre_zero = _store_direct_revision_p_pre_zero(
            checkpoint_directory,
            entry.run_id,
            outcome.execution.p_pre_zero,
        )
    run = DirectRunResult(
        run_id=entry.run_id,
        identity=entry.identity,
        status=status,
        reason=reason,
        runtime_seconds=outcome.runtime_seconds,
        peak_rss_bytes=outcome.peak_rss_bytes,
        peak_gpu_bytes=outcome.peak_gpu_bytes,
        rss_measurement=_canonical_measurement_code(outcome.rss_measurement),
        gpu_measurement=_canonical_measurement_code(outcome.gpu_measurement),
        excluded_cell_count=prepared.audit.excluded_cell_count,
        excluded_cell_ids=prepared.audit.excluded_cell_ids,
        retained_cell_count=prepared.audit.retained_cell_count,
        retained_cell_ids=prepared.audit.retained_cell_ids,
        retained_gene_count=prepared.method_input.shape[1],
        observed_zero_count=int((prepared.method_input.counts == 0).sum()),
        stdout=DirectLogReceipt(
            "stdout", len(outcome.stdout), "discard_content", reason
        ),
        stderr=DirectLogReceipt(
            "stderr", len(outcome.stderr), "discard_content", reason
        ),
    )
    return DirectEvaluatedAttempt(
        run=run,
        metrics=metrics,
        native_output=native,
        native_output_scale=(
            entry.identity.method.output_scale if status == "completed" else None
        ),
        evaluator_output=evaluator,
        p_pre_zero_evidence=p_pre_zero,
    )


@dataclass(frozen=True, slots=True)
class RevisionMaskImputeExecutor:
    """Execute one activated revision using only complete direct values."""

    authority: RunnerAuthority
    direct_adapter: Callable[..., DirectRevisionMaskImputeOutcome]
    authorized_configuration: object
    input_descriptors: Mapping[str, object]
    checkpoint_directory: Path
    registry: MethodRegistry

    def __post_init__(self) -> None:
        from .direct_values import direct_equal, freeze_direct_mapping
        from .fair_comparator_plan import (
            DirectAuthorizedConfiguration,
            PreparedInputDescriptor,
        )

        if not isinstance(self.authority, RunnerAuthority):
            raise TypeError("authority must be a RunnerAuthority")
        if (
            self.authority.plan_scope != "revision_candidate_only"
            or self.authority.base_comparator_selection is None
        ):
            raise RunnerContractError(
                "revision executor requires an activated candidate-only authority"
            )
        if not callable(self.direct_adapter):
            raise TypeError("direct_adapter must be callable")
        if not isinstance(self.authorized_configuration, DirectAuthorizedConfiguration):
            raise TypeError("authorized_configuration must be direct")
        if not isinstance(self.input_descriptors, Mapping) or any(
            not isinstance(value, PreparedInputDescriptor)
            or dataset_id != value.dataset_id
            for dataset_id, value in self.input_descriptors.items()
        ):
            raise TypeError("input_descriptors must contain direct descriptors")
        if not isinstance(self.checkpoint_directory, Path):
            raise TypeError("checkpoint_directory must be a pathlib.Path")
        if not isinstance(self.registry, MethodRegistry):
            raise TypeError("registry must be a MethodRegistry")
        configuration = self.authority.configurations[0]
        spec = self.registry.by_id("maskimpute")
        if (
            self.authorized_configuration.method != comparator_method_binding(spec)
            or self.authorized_configuration.configuration_id
            != configuration.configuration_id
            or self.authorized_configuration.configuration_kind != configuration.kind
            or not direct_equal(
                self.authorized_configuration.payload,
                freeze_direct_mapping(configuration.payload),
            )
            or self.authorized_configuration.requires_count_score
            != configuration.requires_count_score
            or self.authorized_configuration.requires_calibration
            != configuration.requires_calibration
        ):
            raise RunnerContractError(
                "direct revision configuration differs from activated authority"
            )

    def __call__(
        self,
        entry: DirectPlanEntry,
        prepared: PreparedDataset,
        decision: BudgetDecision,
    ) -> DirectEvaluatedAttempt:
        from .direct_values import direct_equal
        from .fair_comparator_plan import DirectPlanEntry, describe_prepared_input

        if not isinstance(entry, DirectPlanEntry):
            raise TypeError("entry must be a DirectPlanEntry")
        if not isinstance(prepared, PreparedDataset):
            raise TypeError("prepared must be a PreparedDataset")
        if not isinstance(decision, BudgetDecision):
            raise TypeError("decision must be a BudgetDecision")
        descriptor = self.input_descriptors.get(entry.identity.dataset_id)
        if descriptor is None or not direct_equal(
            descriptor,
            describe_prepared_input(prepared),
        ):
            raise RunnerContractError("revision prepared descriptor differs")
        spec = self.registry.by_id("maskimpute")
        max_rss_bytes = int(spec.resources.max_rss_gib * 1024**3)
        max_gpu_memory_bytes = int(spec.resources.max_gpu_gib * 1024**3)
        _validate_direct_revision_adapter_values(
            entry.identity,
            descriptor,
            self.authorized_configuration,
            spec,
            prepared.method_input,
            max(decision.timeout_seconds, 1.0),
            max_rss_bytes,
            max_gpu_memory_bytes,
        )
        if entry.preflight_status == "blocked_authority":
            outcome = DirectRevisionMaskImputeOutcome.terminal(
                "blocked_authority",
                str(entry.preflight_reason),
            )
        elif not decision.authorized:
            outcome = DirectRevisionMaskImputeOutcome.terminal(
                "budget_exhausted",
                str(decision.reason),
            )
        else:
            outcome = self.direct_adapter(
                entry.identity,
                descriptor,
                self.authorized_configuration,
                spec,
                prepared.method_input,
                decision.timeout_seconds,
                max_rss_bytes,
                max_gpu_memory_bytes,
            )
            if not isinstance(outcome, DirectRevisionMaskImputeOutcome):
                raise RunnerContractError(
                    "direct revision adapter returned a noncanonical outcome"
                )
            _validate_direct_revision_adapter_values(
                entry.identity,
                descriptor,
                self.authorized_configuration,
                spec,
                prepared.method_input,
                decision.timeout_seconds,
                max_rss_bytes,
                max_gpu_memory_bytes,
            )
        return _evaluate_direct_revision_outcome(
            entry,
            prepared,
            outcome,
            self.checkpoint_directory,
        )


class DevelopmentBudget:
    """Per-method matched search/runtime budget with explicit failure accounting."""

    def __init__(self) -> None:
        self._configurations: dict[str, set[str]] = {}
        self._consumed_seconds: dict[str, float] = {}

    def authorize(
        self,
        spec: MethodSpec,
        configuration_sha256: str,
        *,
        counts_toward_configuration_limit: bool = True,
        budget_scope: str | None = None,
    ) -> BudgetDecision:
        if not isinstance(spec, MethodSpec):
            raise TypeError("spec must be a MethodSpec")
        _require_sha256(configuration_sha256, "budget configuration checksum")
        scope = spec.id if budget_scope is None else budget_scope
        if not isinstance(scope, str) or not scope:
            raise RunnerContractError("budget scope must be a nonempty string")
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
            and configuration_sha256 not in configurations
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
        configuration_sha256: str,
        outcome: AdapterOutcome,
        *,
        counts_toward_configuration_limit: bool = True,
        budget_scope: str | None = None,
    ) -> None:
        if not isinstance(spec, MethodSpec):
            raise TypeError("spec must be a MethodSpec")
        _require_sha256(configuration_sha256, "budget configuration checksum")
        if not isinstance(outcome, AdapterOutcome):
            raise TypeError("outcome must be an AdapterOutcome")
        if outcome.status == "infrastructure_error":
            return
        scope = spec.id if budget_scope is None else budget_scope
        if not isinstance(scope, str) or not scope:
            raise RunnerContractError("budget scope must be a nonempty string")
        if counts_toward_configuration_limit:
            configurations = self._configurations.setdefault(scope, set())
            configurations.add(configuration_sha256)
            if len(configurations) > MAX_DEVELOPMENT_CONFIGURATIONS:
                raise RunnerContractError(
                    "configuration budget was exceeded before recording"
                )
        self._consumed_seconds[scope] = self._consumed_seconds.get(scope, 0.0) + float(
            outcome.runtime_seconds
        )

    def restore(
        self,
        spec: MethodSpec,
        configuration_sha256: str,
        status: str,
        runtime_seconds: int | float,
        *,
        counts_toward_configuration_limit: bool = True,
        budget_scope: str | None = None,
    ) -> None:
        """Replay one already validated checkpoint record into the ledger."""

        if status in {"infrastructure_error", "blocked_authority", "budget_exhausted"}:
            return
        _require_sha256(configuration_sha256, "restored configuration checksum")
        runtime = _require_nonnegative_number(runtime_seconds, "restored runtime")
        scope = spec.id if budget_scope is None else budget_scope
        if not isinstance(scope, str) or not scope:
            raise RunnerContractError("budget scope must be a nonempty string")
        if counts_toward_configuration_limit:
            configurations = self._configurations.setdefault(scope, set())
            configurations.add(configuration_sha256)
            if len(configurations) > MAX_DEVELOPMENT_CONFIGURATIONS:
                raise RunnerContractError("checkpoint exceeds the configuration budget")
        self._consumed_seconds[scope] = self._consumed_seconds.get(scope, 0.0) + float(
            runtime
        )

    def to_dict(self) -> dict[str, object]:
        return {
            method_id: {
                "configuration_sha256": sorted(
                    self._configurations.get(method_id, set())
                ),
                "consumed_seconds": self._consumed_seconds.get(method_id, 0.0),
            }
            for method_id in sorted(
                set(self._configurations) | set(self._consumed_seconds)
            )
        }


def replay_development_budget(
    registry: MethodRegistry,
    entries: Sequence[RunPlanEntry],
    records: Sequence[Mapping[str, object]],
) -> DevelopmentBudget:
    if len(records) > len(entries):
        raise RunnerContractError("checkpoint records are not a plan prefix")
    budget = DevelopmentBudget()
    for entry, stored in zip(entries, records, strict=False):
        run = stored.get("run")
        if not isinstance(run, Mapping):
            raise RunnerContractError("checkpoint budget replay record is invalid")
        budget.restore(
            registry.by_id(entry.method_id),
            entry.configuration_sha256,
            str(run.get("status")),
            run.get("runtime_seconds"),
            counts_toward_configuration_limit=(
                _counts_toward_configuration_limit(entry)
            ),
            budget_scope=_budget_scope(entry),
        )
    return budget


def _comparator_selection_status(
    entries: Sequence[RunPlanEntry],
    records: Sequence[Mapping[str, object]],
) -> Literal[
    "complete_terminal_denominator",
    "blocked_incomplete_denominator",
]:
    for position, entry in enumerate(entries):
        if entry.configuration_kind != "comparator_tuning":
            continue
        if position >= len(records):
            return "blocked_incomplete_denominator"
        run = records[position].get("run")
        if not isinstance(run, Mapping):
            raise RunnerContractError(
                "checkpoint comparator-selection record is invalid"
            )
        status = str(run.get("status"))
        if status in COMPARATOR_SELECTION_BLOCKING_STATUSES:
            return "blocked_incomplete_denominator"
        if status != "completed" and status not in INTRINSIC_TERMINAL_STATUSES:
            raise RunnerContractError(
                "checkpoint comparator-selection status is invalid"
            )
    return "complete_terminal_denominator"


@dataclass(frozen=True, slots=True)
class RawRunResult:
    """Status-complete measured attempt before artifact paths are assigned."""

    run_id: str
    method_id: str
    dataset_id: str
    source_dataset_sha256: str
    mechanism: str
    biological_id: str
    technical_view: str
    model_seed: int | None
    configuration_id: str
    configuration_sha256: str | None
    configuration_kind: str
    requires_count_score: bool
    requires_calibration: bool
    method_input_sha256: str
    dataset_qc_policy_sha256: str
    excluded_cell_count: int
    excluded_cell_ids_sha256: str
    retained_cell_count: int
    retained_cell_ids_sha256: str
    retained_gene_count: int
    observed_zero_count: int
    status: str
    reason: str | None
    runtime_seconds: int | float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    rss_measurement: str
    gpu_measurement: str
    calibration_artifact_sha256: str | None
    calibration_context_sha256: str | None
    calibration_training_manifest_sha256s: tuple[str, ...]
    calibration_held_out_manifest_sha256s: tuple[str, ...]
    calibration_fold_calibrator_sha256: str | None
    stdout_sha256: str
    stderr_sha256: str
    native_output_sha256: str | None
    evaluator_output_sha256: str | None
    comparator_configuration: BoundComparatorConfiguration | None = None
    comparator_nonexecution_identity: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        from .direct_values import direct_json_value

        value = direct_json_value(self)
        if not isinstance(value, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError("raw run result must encode as an object")
        if self.configuration_kind in {
            "comparator_tuning",
            "comparator_nonexecution",
        }:
            value.pop("configuration_sha256")
            value["comparator_configuration"] = (
                None
                if self.comparator_configuration is None
                else direct_bound_comparator_value(self.comparator_configuration)
            )
            value["comparator_nonexecution_identity"] = (
                None
                if self.comparator_nonexecution_identity is None
                else direct_json_value(
                    self.comparator_nonexecution_identity,
                    payload=True,
                )
            )
        else:
            value.pop("comparator_configuration")
            value.pop("comparator_nonexecution_identity")
        return value


@dataclass(frozen=True, slots=True)
class LongFormMetric:
    """One complete publication metric denominator row."""

    mechanism: str
    biological_id: str
    technical_view: str
    dataset_id: str
    method: str
    model_seed: int | None
    configuration_id: str
    configuration_sha256: str | None
    metric: str
    value: float | None
    n: int
    status: str
    reason: str | None
    comparator_configuration: BoundComparatorConfiguration | None = None
    comparator_nonexecution_identity: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        from .direct_values import direct_json_value

        value = direct_json_value(self)
        if not isinstance(value, dict):  # pragma: no cover - dataclass invariant
            raise AssertionError("long-form metric must encode as an object")
        if (
            self.comparator_configuration is not None
            or self.comparator_nonexecution_identity is not None
        ):
            value.pop("configuration_sha256")
            value["comparator_configuration"] = (
                None
                if self.comparator_configuration is None
                else direct_bound_comparator_value(self.comparator_configuration)
            )
            value["comparator_nonexecution_identity"] = (
                None
                if self.comparator_nonexecution_identity is None
                else direct_json_value(
                    self.comparator_nonexecution_identity,
                    payload=True,
                )
            )
        else:
            value.pop("comparator_configuration")
            value.pop("comparator_nonexecution_identity")
        return value


@dataclass(frozen=True, slots=True)
class EvaluatedAttempt:
    """One raw attempt, its complete metrics, logs, and optional common output."""

    run: RawRunResult
    metrics: tuple[LongFormMetric, ...]
    stdout: bytes
    stderr: bytes
    native_output: np.ndarray | None
    native_output_scale: str | None
    evaluator_output: np.ndarray | None
    p_pre_zero_evidence: PreZeroEvidence


def _metric_names() -> tuple[str, ...]:
    from .metrics import reconstruction_metrics

    values = reconstruction_metrics(
        np.zeros((1, 1), dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        None,
        truth_kind="orthogonal_only",
    )
    return tuple(values)


_RECONSTRUCTION_METRIC_NAMES = _metric_names()


def _default_output_converter(
    method_input: MethodInput,
    execution: AdapterExecution,
) -> np.ndarray:
    from .methods import (
        AdapterUnavailableError,
        core_output_to_evaluator_log2_cp10k,
        recent_output_to_evaluator_log2_cp10k,
    )

    snapshot = execution.snapshot
    from .methods import _output_digest

    matrix = snapshot.matrix
    expected_snapshot_sha256 = _output_digest(
        method_id=snapshot.method_id,
        source_dataset_sha256=snapshot.source_dataset_sha256,
        output_scale=snapshot.output_scale,
        obs_ids=snapshot.obs_ids,
        var_ids=snapshot.var_ids,
        shape=snapshot.shape,
        matrix_bytes=np.asarray(matrix, dtype="<f8", order="C").tobytes(order="C"),
    )
    if snapshot.matrix_sha256 != expected_snapshot_sha256:
        raise ValueError("native output snapshot checksum mismatch")
    if snapshot.method_id in {
        "observed",
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "maskimpute",
        "capacity-matched-ae",
    }:
        return core_output_to_evaluator_log2_cp10k(method_input, snapshot)
    if snapshot.method_id in {"scziva", "afmf", "biaeimpute", "d3impute"}:
        return recent_output_to_evaluator_log2_cp10k(method_input, snapshot)
    if snapshot.method_id in {"sccr", "scsdae"}:
        from .methods import count_equivalent_to_log2_cp10k
        from .methods.sccr import sccr_to_evaluator_counts
        from .methods.scsdae import scsdae_to_evaluator_counts

        converter = (
            sccr_to_evaluator_counts
            if snapshot.method_id == "sccr"
            else scsdae_to_evaluator_counts
        )
        return count_equivalent_to_log2_cp10k(converter(method_input, matrix))
    raise AdapterUnavailableError(
        "evaluator_converter_unavailable",
        f"no common-scale converter is registered for {snapshot.method_id}",
    )


def _dense_evaluator_matrix(value: object, name: str) -> np.ndarray:
    from scipy import sparse

    from maskimpute.sparse_input import _unmasked_array

    try:
        array = (
            value.toarray() if sparse.issparse(value) else _unmasked_array(value, name)
        )
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
    from .methods import count_equivalent_to_log2_cp10k

    observed = count_equivalent_to_log2_cp10k(
        _dense_evaluator_matrix(prepared.evaluator_dataset.X, "observed counts")
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
            _dense_evaluator_matrix(
                prepared.evaluator_dataset.layers[truth_layer],
                "primary evaluator truth",
            )
        )
    marker_columns = [
        name
        for name in prepared.evaluator_dataset.var.columns
        if isinstance(name, str) and name.casefold().startswith("marker")
    ]
    marker_mask: np.ndarray | None = None
    if marker_columns:
        masks = []
        for name in marker_columns:
            values = prepared.evaluator_dataset.var[name].to_numpy(copy=True)
            if values.dtype.kind != "b":
                continue
            masks.append(np.asarray(values, dtype=bool))
        if masks:
            marker_mask = np.logical_or.reduce(masks)
    return observed, truth, truth_kind, marker_mask


def _prezero_evaluator_targets(
    prepared: PreparedDataset,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """Return evaluator-private raw-count targets for realized score validation."""

    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    if prepared.evaluator_dataset is None:
        raise RunnerContractError(
            "evaluator-private score authority is unavailable for p_pre_zero"
        )
    observed = _dense_evaluator_matrix(
        prepared.evaluator_dataset.X, "observed counts for p_pre_zero"
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
            raise RunnerContractError("evaluator score truth layer is unavailable")
        truth = _dense_evaluator_matrix(
            prepared.evaluator_dataset.layers[truth_layer],
            "p_pre_zero evaluator truth",
        )
    return observed, truth, truth_kind


def _count_score_input_sha256(prepared: PreparedDataset) -> str:
    """Recompute the count-model input digest from authoritative method counts."""

    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    from maskimpute.count_model import _counts_sha256

    return _counts_sha256(prepared.method_input.counts)


def _derive_prezero_execution_authority(
    prepared: PreparedDataset,
    entry: RunPlanEntry,
    context: ExecutionAuthorityContext,
    *,
    calibration_usage: str,
    repository: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Independently derive one exact realized-score matrix and policy."""

    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    if not isinstance(entry, RunPlanEntry):
        raise TypeError("entry must be a RunPlanEntry")
    if not isinstance(context, ExecutionAuthorityContext):
        raise TypeError("context must be an ExecutionAuthorityContext")
    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if entry.method_id != "maskimpute" or not entry.requires_count_score:
        raise RunnerContractError(
            "realized score authority requires a count-score MaskImpute entry"
        )
    if entry.dataset_id != prepared.binding.dataset_id:
        raise RunnerContractError("score authority dataset differs from its plan entry")
    repository = repository.resolve(strict=True)

    def authority_path(relative_value: str, name: str) -> Path:
        if not isinstance(relative_value, str):
            raise RunnerContractError(f"{name} path must be a string")
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RunnerContractError(f"{name} path is unsafe")
        path = repository.joinpath(*relative.parts)
        for component in (path, *path.parents):
            if component == repository.parent:
                break
            if os.path.lexists(component) and stat.S_ISLNK(component.lstat().st_mode):
                raise RunnerContractError(f"{name} path contains a symlink")
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RunnerContractError(f"{name} file is unavailable") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise RunnerContractError(f"{name} must be a unique regular file")
        return path

    calibration_file_sha256 = _require_sha256(
        context.retained_calibration_sha256,
        "retained calibration file checksum",
    )
    assert calibration_file_sha256 is not None
    calibration_path = authority_path(
        context.retained_calibration_path,
        "retained calibration authority",
    )
    if _file_sha256(calibration_path) != calibration_file_sha256:
        raise RunnerContractError("retained calibration file checksum mismatch")
    try:
        from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model
        from maskimpute.ablations import (
            _derive_prezero_execution_policy,
            load_ablation_registry,
        )
        from maskimpute.calibration import load_calibration_artifact

        count_model_payload = json.loads(
            context.count_model_config_json,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        if (
            not isinstance(count_model_payload, dict)
            or canonical_sha256(count_model_payload)
            != context.count_model_config_sha256
        ):
            raise RunnerContractError(
                "count-model configuration differs from execution authority"
            )
        count_model_config = PreZeroCountModelConfig(**count_model_payload)
        calibration = load_calibration_artifact(calibration_path)
    except Exception as error:
        raise RunnerContractError(
            "retained score/calibration authority failed semantic validation"
        ) from error

    if calibration_usage == "development_holdout":
        manifest_file_sha256 = _require_sha256(
            context.count_score_manifest_sha256,
            "development count-score manifest file checksum",
        )
        assert manifest_file_sha256 is not None
        manifest_path = authority_path(
            context.count_score_manifest_path,
            "development count-score manifest authority",
        )
        if _file_sha256(manifest_path) != manifest_file_sha256:
            raise RunnerContractError(
                "development count-score manifest file checksum mismatch"
            )
        manifest = _load_strict_json(manifest_path, "development count-score manifest")
        expected_manifest_fields = {
            "schema_version",
            "artifact_type",
            "dataset_manifest_sha256",
            "count_model_config_sha256",
            "dataset_qc_policy_sha256",
            "entries",
            "manifest_sha256",
        }
        unsigned_manifest = {
            name: value for name, value in manifest.items() if name != "manifest_sha256"
        }
        if (
            set(manifest) != expected_manifest_fields
            or manifest.get("schema_version") != 1
            or manifest.get("artifact_type")
            != "maskimpute_development_count_score_manifest"
            or manifest.get("dataset_manifest_sha256")
            != prepared.binding.manifest_sha256
            or manifest.get("count_model_config_sha256")
            != context.count_model_config_sha256
            or manifest.get("dataset_qc_policy_sha256")
            != DatasetQCPolicy.fixed().sha256
            or manifest.get("manifest_sha256") != canonical_sha256(unsigned_manifest)
        ):
            raise RunnerContractError(
                "development count-score manifest authority is invalid"
            )
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise RunnerContractError(
                "development count-score manifest entries are invalid"
            )
        matches = [
            value
            for value in entries
            if isinstance(value, Mapping)
            and value.get("mechanism") == entry.mechanism
            and value.get("biological_id") == entry.biological_id
            and value.get("technical_view") == entry.technical_view
            and value.get("dataset_id") == entry.dataset_id
        ]
        if len(matches) != 1:
            raise RunnerContractError(
                "development count-score manifest lacks one exact dataset entry"
            )
        try:
            from .development_scores import (
                _score_entry,
                _score_filename,
                load_count_score_artifact,
            )

            score = load_count_score_artifact(
                manifest_path.parent / _score_filename(prepared)
            )
            score.score_for_counts(
                prepared.method_input.counts,
                prepared.method_input.obs_ids,
            )
            expected_entry = _score_entry(prepared, score)
        except Exception as error:
            raise RunnerContractError(
                "development count-score artifact failed deterministic validation"
            ) from error
        if dict(matches[0]) != expected_entry:
            raise RunnerContractError(
                "development count-score entry differs from its deterministic artifact"
            )
    elif calibration_usage == "retained_all_development":
        authority_file_sha256 = _require_sha256(
            context.count_score_manifest_sha256,
            "final count-score authority file checksum",
        )
        assert authority_file_sha256 is not None
        authority_file = authority_path(
            context.count_score_manifest_path,
            "final count-score authority",
        )
        if _file_sha256(authority_file) != authority_file_sha256:
            raise RunnerContractError(
                "final count-score authority file checksum mismatch"
            )
        authority_payload = _load_strict_json(
            authority_file, "final count-score authority"
        )
        unsigned_authority = {
            name: value
            for name, value in authority_payload.items()
            if name != "payload_sha256"
        }
        from .trajectory_dataset import RegisteredTrajectoryBinding

        if isinstance(prepared.binding, RegisteredTrajectoryBinding):
            binding = prepared.binding
            expected_authority_fields = {
                "schema_version",
                "artifact_type",
                "status",
                "scope",
                "frozen_method_sha256",
                "primary_final_plan_sha256",
                "primary_execution_authority_sha256",
                "primary_count_score_authority_file_sha256",
                "execution_claim_sha256",
                "execution_environment_sha256",
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
                "count_model_config",
                "count_model_config_sha256",
                "retained_calibration_file_sha256",
                "payload_sha256",
            }
            if (
                set(authority_payload) != expected_authority_fields
                or authority_payload.get("schema_version") != 1
                or authority_payload.get("artifact_type")
                != "maskimpute_trajectory_count_score_authority"
                or authority_payload.get("status") != "ready"
                or authority_payload.get("scope")
                != "truth_free_registered_trajectory_inference"
                or entry.mechanism != binding.mechanism
                or entry.biological_id != binding.biological_id
                or entry.technical_view != binding.technical_view
                or entry.source_dataset_sha256 != binding.dataset_sha256
                or authority_payload.get("trajectory_authority_file_sha256")
                != binding.authority_file_sha256
                or authority_payload.get("trajectory_authority_sha256")
                != binding.authority_sha256
                or authority_payload.get("trajectory_binding_sha256")
                != binding.registered_binding_sha256
                or authority_payload.get("trajectory_dataset_sha256")
                != binding.dataset_sha256
                or authority_payload.get("trajectory_dataset_file_sha256")
                != binding.dataset_file_sha256
                or authority_payload.get("trajectory_method_input_sha256")
                != method_input_sha256(prepared.method_input)
                or authority_payload.get("trajectory_retained_cell_ids_sha256")
                != prepared.audit.retained_cell_ids_sha256
                or authority_payload.get("dataset_qc_policy_sha256")
                != DatasetQCPolicy.fixed().sha256
                or authority_payload.get("count_model_config") != count_model_payload
                or authority_payload.get("count_model_config_sha256")
                != context.count_model_config_sha256
                or authority_payload.get("retained_calibration_file_sha256")
                != calibration_file_sha256
                or authority_payload.get("payload_sha256")
                != canonical_sha256(unsigned_authority)
            ):
                raise RunnerContractError(
                    "registered trajectory count-score authority is invalid"
                )
            for name in (
                "frozen_method_sha256",
                "primary_final_plan_sha256",
                "primary_execution_authority_sha256",
                "primary_count_score_authority_file_sha256",
                "execution_claim_sha256",
                "execution_environment_sha256",
                "trajectory_dataset_receipt_sha256",
                "trajectory_dataset_receipt_file_sha256",
            ):
                _require_sha256(
                    authority_payload.get(name),
                    f"trajectory count-score authority {name}",
                )
        else:
            expected_authority_fields = {
                "schema_version",
                "artifact_type",
                "status",
                "scope",
                "frozen_method_sha256",
                "execution_claim_sha256",
                "execution_environment_sha256",
                "dataset_manifest_sha256",
                "selection_contract_file_sha256",
                "count_model_config",
                "count_model_config_sha256",
                "payload_sha256",
            }
            if (
                set(authority_payload) != expected_authority_fields
                or authority_payload.get("schema_version") != 1
                or authority_payload.get("artifact_type")
                != "maskimpute_final_count_score_authority"
                or authority_payload.get("status") != "ready"
                or authority_payload.get("scope") != "truth_free_final_inference"
                or authority_payload.get("dataset_manifest_sha256")
                != prepared.binding.manifest_sha256
                or authority_payload.get("count_model_config") != count_model_payload
                or authority_payload.get("count_model_config_sha256")
                != context.count_model_config_sha256
                or authority_payload.get("payload_sha256")
                != canonical_sha256(unsigned_authority)
            ):
                raise RunnerContractError("final count-score authority is invalid")
            for name in (
                "frozen_method_sha256",
                "execution_claim_sha256",
                "execution_environment_sha256",
                "selection_contract_file_sha256",
            ):
                _require_sha256(
                    authority_payload.get(name),
                    f"final count-score authority {name}",
                )
        score = fit_p_pre_zero_count_model(
            prepared.method_input.counts,
            prepared.method_input.obs_ids,
            count_model_config,
        )
    else:
        raise RunnerContractError("score authority calibration usage is invalid")

    if score.config_sha256 != context.count_model_config_sha256:
        raise RunnerContractError(
            "deterministic score configuration differs from execution authority"
        )
    registry = load_ablation_registry(repository / "study/ablations.json")
    score_spec = (
        registry.reference
        if entry.requires_calibration
        else registry.by_id["direct-score"]
    )
    try:
        probability, diagnostics = _derive_prezero_execution_policy(
            prepared.method_input.counts,
            prepared.method_input.obs_ids,
            score,
            calibration,
            score_spec,
            calibration_usage=calibration_usage,
            development_mechanism=entry.mechanism,
            development_biological_id=entry.biological_id,
        )
        policy = policy_from_score_diagnostics(diagnostics)
    except Exception as error:
        raise RunnerContractError(
            "realized p_pre_zero derivation failed execution-policy validation"
        ) from error
    if policy["calibration_file_sha256"] != calibration_file_sha256:
        raise RunnerContractError(
            "calibration file and payload digest domains differ from authority"
        )
    return probability, policy


def _evaluator_output_sha256(
    entry: RunPlanEntry,
    prepared: PreparedDataset,
    output: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"maskimpute-evaluator-log2-cp10k-output-v1\0")
    digest.update(
        _canonical_bytes(
            {
                "run_id": entry.run_id,
                "method_input_sha256": method_input_sha256(prepared.method_input),
                "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
                "shape": output.shape,
                "dtype": "<f8",
                "scale": "log2_cp10k_plus_1",
            }
        )
    )
    digest.update(np.asarray(output, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _long_form_unavailable(
    entry: RunPlanEntry,
    status: str,
    reason: str,
) -> tuple[LongFormMetric, ...]:
    return tuple(
        LongFormMetric(
            mechanism=entry.mechanism,
            biological_id=entry.biological_id,
            technical_view=entry.technical_view,
            dataset_id=entry.dataset_id,
            method=entry.method_id,
            model_seed=entry.model_seed,
            configuration_id=entry.configuration_id,
            configuration_sha256=entry.configuration_sha256,
            comparator_configuration=entry.comparator_configuration,
            comparator_nonexecution_identity=(entry.comparator_nonexecution_identity),
            metric=name,
            value=None,
            n=0,
            status=status,
            reason=reason,
        )
        for name in _RECONSTRUCTION_METRIC_NAMES
    )


def _stable_exception_detail_sha256(error: Exception, domain: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    if len(error.args) == 1 and type(error.args[0]) is str:
        digest.update(error.args[0].encode("utf-8", errors="surrogatepass"))
    else:
        for argument in error.args:
            if type(argument) is str:
                payload = b"str\0" + argument.encode("utf-8", errors="surrogatepass")
            elif type(argument) is bytes:
                payload = b"bytes\0" + argument
            elif argument is None:
                payload = b"none\0"
            elif type(argument) is bool:
                payload = b"bool\0" + (b"true" if argument else b"false")
            elif type(argument) is int:
                payload = b"int\0" + str(argument).encode("ascii")
            elif type(argument) is float:
                payload = b"float\0" + argument.hex().encode("ascii")
            else:
                payload = (
                    b"object-type\0"
                    + type(argument).__module__.encode("utf-8", errors="surrogatepass")
                    + b"."
                    + type(argument).__qualname__.encode(
                        "utf-8", errors="surrogatepass"
                    )
                )
            digest.update(struct.pack("<Q", len(payload)))
            digest.update(payload)
    return digest.hexdigest()


def _terminal_reason_component(value: str, fallback: str) -> str:
    normalized = value.replace("-", "_").lower()
    if re.fullmatch(r"[a-z][a-z0-9_]*", normalized) and not any(
        marker in normalized for marker in ("pending", "not_yet", "unverified")
    ):
        return normalized
    return fallback


def _evaluator_conversion_failure_reason(error: Exception) -> str:
    from .methods import AdapterUnavailableError

    if isinstance(error, AdapterUnavailableError):
        error_name = _terminal_reason_component(
            error.reason_code, "adapter_unavailable"
        )
    else:
        error_name = next(
            name
            for error_type, name in (
                (TypeError, "typeerror"),
                (ValueError, "valueerror"),
                (OverflowError, "overflowerror"),
            )
            if isinstance(error, error_type)
        )
    detail_sha256 = _stable_exception_detail_sha256(
        error, b"maskimpute-evaluator-conversion-detail-v1\0"
    )
    return f"evaluator_conversion_{error_name}_detail_{detail_sha256}"


def _adapter_failure_reason(
    error: Exception,
    method_id: str,
    *,
    unavailable: bool,
) -> str:
    from .methods import AdapterUnavailableError

    method = _terminal_reason_component(method_id, "method")
    if unavailable and isinstance(error, AdapterUnavailableError):
        category = _terminal_reason_component(error.reason_code, "adapter_unavailable")
    else:
        error_type = f"{type(error).__module__}.{type(error).__qualname__}"
        category = "adapter_exception_" + _terminal_reason_component(
            type(error).__name__, "exception"
        )
        # Bind the qualified type even when two exception classes share a name.
        error = RuntimeError(error_type, *error.args)
    detail_sha256 = _stable_exception_detail_sha256(
        error, b"maskimpute-adapter-failure-detail-v1\0"
    )
    return f"{category}_{method}_detail_{detail_sha256}"


def _declared_failure_reason(category: str, method_id: str) -> str:
    safe_category = _terminal_reason_component(category, "adapter_unavailable")
    safe_method = _terminal_reason_component(method_id, "method")
    digest = hashlib.sha256()
    digest.update(b"maskimpute-declared-adapter-disposition-v1\0")
    digest.update(safe_category.encode("ascii"))
    digest.update(b"\0")
    digest.update(safe_method.encode("ascii"))
    return f"{safe_category}_{safe_method}_detail_{digest.hexdigest()}"


def evaluate_adapter_outcome(
    entry: RunPlanEntry,
    prepared: PreparedDataset,
    outcome: AdapterOutcome,
    *,
    output_converter: Callable[
        [MethodInput, AdapterExecution], np.ndarray
    ] = _default_output_converter,
    dataset_qc_policy_sha256: str | None = None,
) -> EvaluatedAttempt:
    """Convert one attempt and compute complete evaluator-only metric rows."""

    from maskimpute.sparse_input import _unmasked_array

    from .methods import AdapterUnavailableError
    from .metrics import reconstruction_metrics

    if not isinstance(entry, RunPlanEntry):
        raise TypeError("entry must be a RunPlanEntry")
    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    if not isinstance(outcome, AdapterOutcome):
        raise TypeError("outcome must be an AdapterOutcome")
    if (
        entry.dataset_id != prepared.binding.dataset_id
        or entry.source_dataset_sha256 != prepared.binding.dataset_sha256
    ):
        raise RunnerContractError("plan entry does not match the prepared dataset")
    if dataset_qc_policy_sha256 is None:
        dataset_qc_policy_sha256 = DatasetQCPolicy.fixed().sha256
    _require_sha256(dataset_qc_policy_sha256, "dataset QC policy checksum")
    run_status = outcome.status
    reason = outcome.reason
    native_output_sha256: str | None = None
    native_output: np.ndarray | None = None
    native_output_scale: str | None = None
    evaluator_output: np.ndarray | None = None
    metric_rows: tuple[LongFormMetric, ...]
    if outcome.status == "completed":
        assert outcome.execution is not None
        snapshot = outcome.execution.snapshot
        if (
            snapshot.method_id != entry.method_id
            or snapshot.source_dataset_sha256 != entry.source_dataset_sha256
            or snapshot.obs_ids != prepared.audit.retained_cell_ids
            or snapshot.var_ids != prepared.method_input.var_ids
        ):
            raise RunnerContractError(
                "completed adapter snapshot mismatches its run plan"
            )
        native_output_sha256 = snapshot.matrix_sha256
        native_output = np.array(
            snapshot.matrix, dtype=np.float64, copy=True, order="C", subok=False
        )
        native_output_scale = snapshot.output_scale
        try:
            converted = output_converter(prepared.method_input, outcome.execution)
            evaluator_output = np.array(
                _unmasked_array(converted, "common-scale output"),
                dtype=np.float64,
                copy=True,
                order="C",
                subok=False,
            )
            if (
                evaluator_output.shape != prepared.method_input.shape
                or not np.isfinite(evaluator_output).all()
                or bool((evaluator_output < 0).any())
            ):
                raise ValueError("common-scale output is invalid")
        except AdapterUnavailableError as error:
            run_status = "unavailable"
            reason = _evaluator_conversion_failure_reason(error)
            evaluator_output = None
        except (TypeError, ValueError, OverflowError) as error:
            run_status = "unavailable"
            reason = _evaluator_conversion_failure_reason(error)
            evaluator_output = None
        if evaluator_output is None:
            assert reason is not None
            metric_rows = _long_form_unavailable(entry, run_status, reason)
        else:
            observed, truth, truth_kind, marker_mask = _evaluator_targets(prepared)
            metrics = reconstruction_metrics(
                evaluator_output,
                observed,
                truth,
                marker_genes=marker_mask,
                truth_kind=truth_kind,
            )
            metric_rows = tuple(
                LongFormMetric(
                    mechanism=entry.mechanism,
                    biological_id=entry.biological_id,
                    technical_view=entry.technical_view,
                    dataset_id=entry.dataset_id,
                    method=entry.method_id,
                    model_seed=entry.model_seed,
                    configuration_id=entry.configuration_id,
                    configuration_sha256=entry.configuration_sha256,
                    comparator_configuration=entry.comparator_configuration,
                    comparator_nonexecution_identity=(
                        entry.comparator_nonexecution_identity
                    ),
                    metric=name,
                    value=None if metric.value is None else float(metric.value),
                    n=int(metric.n),
                    status="unavailable" if metric.value is None else "completed",
                    reason=metric.reason,
                )
                for name, metric in metrics.items()
            )
    else:
        assert reason is not None
        metric_rows = _long_form_unavailable(entry, run_status, reason)
    stdout_sha256 = hashlib.sha256(outcome.stdout).hexdigest()
    stderr_sha256 = hashlib.sha256(outcome.stderr).hexdigest()
    evaluator_digest = (
        None
        if evaluator_output is None
        else _evaluator_output_sha256(entry, prepared, evaluator_output)
    )
    calibration_receipt = outcome.calibration_fold_receipt
    method_input_digest = method_input_sha256(prepared.method_input)
    score_observed, score_truth, score_truth_kind = _prezero_evaluator_targets(prepared)
    try:
        p_pre_zero_evidence = evaluate_prezero_evidence(
            identity={
                "run_id": entry.run_id,
                "method_id": entry.method_id,
                "dataset_id": entry.dataset_id,
                "source_dataset_sha256": entry.source_dataset_sha256,
                "mechanism": entry.mechanism,
                "biological_id": entry.biological_id,
                "technical_view": entry.technical_view,
                "model_seed": entry.model_seed,
                "configuration_id": entry.configuration_id,
                "configuration_sha256": entry.configuration_sha256,
                "method_input_sha256": method_input_digest,
                "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
            },
            method_shape=prepared.method_input.shape,
            method_id=entry.method_id,
            execution=outcome.execution,
            run_status=run_status,
            run_reason=reason,
            observed=score_observed,
            truth=score_truth,
            truth_kind=score_truth_kind,
        )
    except PreZeroEvidenceError as error:
        raise RunnerContractError(str(error)) from error
    run = RawRunResult(
        run_id=entry.run_id,
        method_id=entry.method_id,
        dataset_id=entry.dataset_id,
        source_dataset_sha256=entry.source_dataset_sha256,
        mechanism=entry.mechanism,
        biological_id=entry.biological_id,
        technical_view=entry.technical_view,
        model_seed=entry.model_seed,
        configuration_id=entry.configuration_id,
        configuration_sha256=entry.configuration_sha256,
        configuration_kind=entry.configuration_kind,
        requires_count_score=entry.requires_count_score,
        requires_calibration=entry.requires_calibration,
        method_input_sha256=method_input_digest,
        dataset_qc_policy_sha256=dataset_qc_policy_sha256,
        excluded_cell_count=prepared.audit.excluded_cell_count,
        excluded_cell_ids_sha256=prepared.audit.excluded_cell_ids_sha256,
        retained_cell_count=prepared.audit.retained_cell_count,
        retained_cell_ids_sha256=prepared.audit.retained_cell_ids_sha256,
        retained_gene_count=prepared.method_input.shape[1],
        observed_zero_count=int((score_observed == 0).sum()),
        status=run_status,
        reason=reason,
        runtime_seconds=outcome.runtime_seconds,
        peak_rss_bytes=outcome.peak_rss_bytes,
        peak_gpu_bytes=outcome.peak_gpu_bytes,
        rss_measurement=outcome.rss_measurement,
        gpu_measurement=outcome.gpu_measurement,
        calibration_artifact_sha256=(
            None
            if calibration_receipt is None
            else calibration_receipt.calibration_artifact_sha256
        ),
        calibration_context_sha256=(
            None
            if calibration_receipt is None
            else calibration_receipt.calibration_context_sha256
        ),
        calibration_training_manifest_sha256s=(
            ()
            if calibration_receipt is None
            else calibration_receipt.training_manifest_sha256s
        ),
        calibration_held_out_manifest_sha256s=(
            ()
            if calibration_receipt is None
            else calibration_receipt.held_out_manifest_sha256s
        ),
        calibration_fold_calibrator_sha256=(
            None
            if calibration_receipt is None
            else calibration_receipt.fold_calibrator_sha256
        ),
        stdout_sha256=stdout_sha256,
        stderr_sha256=stderr_sha256,
        native_output_sha256=native_output_sha256,
        evaluator_output_sha256=evaluator_digest,
        comparator_configuration=entry.comparator_configuration,
        comparator_nonexecution_identity=entry.comparator_nonexecution_identity,
    )
    return EvaluatedAttempt(
        run=run,
        metrics=metric_rows,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        native_output=native_output,
        native_output_scale=native_output_scale,
        evaluator_output=evaluator_output,
        p_pre_zero_evidence=p_pre_zero_evidence,
    )


@dataclass(frozen=True, slots=True)
class CheckpointReport:
    """Canonical resumable prefix of an immutable competition plan."""

    schema_version: int
    plan_sha256: str
    input_hashes: Mapping[str, str]
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
    checkpoint_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise RunnerContractError(f"nonfinite checkpoint JSON constant: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunnerContractError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _validate_prepared_dataset_authority(
    plan: CompetitionPlan,
    prepared_datasets: Mapping[str, PreparedDataset],
) -> dict[str, PreparedDataset]:
    """Require one exact evaluator-private authority for every planned dataset."""

    if not isinstance(prepared_datasets, Mapping):
        raise TypeError("prepared_datasets must be a mapping")
    planned_dataset_ids = {entry.dataset_id for entry in plan.entries}
    if set(prepared_datasets) != planned_dataset_ids:
        raise RunnerContractError(
            "prepared dataset authority does not exactly cover the competition plan"
        )
    prepared_by_dataset: dict[str, PreparedDataset] = {}
    for dataset_id in sorted(planned_dataset_ids):
        prepared = prepared_datasets.get(dataset_id)
        if not isinstance(prepared, PreparedDataset):
            raise RunnerContractError(
                f"prepared dataset authority is invalid for {dataset_id}"
            )
        entries = tuple(
            entry for entry in plan.entries if entry.dataset_id == dataset_id
        )
        expected_binding = {
            (
                entry.source_dataset_sha256,
                entry.mechanism,
                entry.biological_id,
                entry.technical_view,
            )
            for entry in entries
        }
        actual_binding = (
            prepared.binding.dataset_sha256,
            prepared.binding.mechanism,
            prepared.binding.biological_id,
            prepared.binding.technical_view,
        )
        if (
            prepared.binding.dataset_id != dataset_id
            or expected_binding != {actual_binding}
            or prepared.method_input.source_dataset_sha256
            != prepared.binding.dataset_sha256
            or prepared.method_input.shape[0] != prepared.audit.retained_cell_count
            or prepared.method_input.shape[1] != prepared.binding.genes
            or prepared.method_input.obs_ids != prepared.audit.retained_cell_ids
        ):
            raise RunnerContractError(
                f"prepared dataset authority differs from plan binding {dataset_id}"
            )
        observed, truth, truth_kind = _prezero_evaluator_targets(prepared)
        if observed.shape != prepared.method_input.shape or not np.array_equal(
            observed, prepared.method_input.counts
        ):
            raise RunnerContractError(
                f"prepared evaluator observations differ from method input {dataset_id}"
            )
        if truth_kind != "orthogonal_only" and (
            truth is None or truth.shape != prepared.method_input.shape
        ):
            raise RunnerContractError(
                f"prepared evaluator truth differs from method input {dataset_id}"
            )
        prepared_by_dataset[dataset_id] = prepared
    return prepared_by_dataset


class CheckpointStore:
    """Atomic canonical checkpoint plus immutable raw run artifacts."""

    _CHECKPOINT_KEYS = frozenset(
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

    def __init__(
        self,
        output_dir: Path,
        *,
        authority_repository: Path | None = None,
    ) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        if authority_repository is not None and not isinstance(
            authority_repository, Path
        ):
            raise TypeError("authority_repository must be a pathlib.Path")
        self.output_dir = output_dir.absolute()
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.authority_repository = (
            Path(__file__).resolve().parents[1]
            if authority_repository is None
            else authority_repository.resolve(strict=True)
        )
        self._prezero_authority_cache: dict[
            tuple[str, str, str, str, str, str, bool, str],
            tuple[tuple[int, int], bytes, bytes],
        ] = {}

    def _expected_prezero_authority(
        self,
        entry: RunPlanEntry,
        prepared: PreparedDataset,
        execution_authority: ExecutionAuthorityContext | None,
        *,
        calibration_usage: str,
        expected_matrix_present: bool,
    ) -> tuple[np.ndarray | None, dict[str, object] | None]:
        """Derive and cache authority whenever realized score bytes are present."""

        if (
            not expected_matrix_present
            or entry.method_id != "maskimpute"
            or not entry.requires_count_score
        ):
            return None, None
        if execution_authority is None:
            raise RunnerContractError(
                "realized MaskImpute score lacks frozen execution authority"
            )
        key = (
            execution_authority.authority_sha256,
            execution_authority.count_model_config_sha256,
            execution_authority.count_score_manifest_sha256 or "",
            execution_authority.retained_calibration_sha256 or "",
            entry.dataset_id,
            _count_score_input_sha256(prepared),
            entry.requires_calibration,
            calibration_usage,
        )
        cached = self._prezero_authority_cache.get(key)
        if cached is None:
            probability, policy = _derive_prezero_execution_authority(
                prepared,
                entry,
                execution_authority,
                calibration_usage=calibration_usage,
                repository=self.authority_repository,
            )
            canonical_probability = np.array(
                probability,
                dtype="<f8",
                copy=True,
                order="C",
                subok=False,
            )
            cached = (
                canonical_probability.shape,
                canonical_probability.tobytes(order="C"),
                _canonical_bytes(policy),
            )
            self._prezero_authority_cache[key] = cached
        shape, probability_bytes, policy_bytes = cached
        detached_probability = (
            np.frombuffer(probability_bytes, dtype="<f8").reshape(shape).copy(order="C")
        )
        detached_probability.setflags(write=False)
        detached_policy = json.loads(
            policy_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        if not isinstance(detached_policy, dict):  # pragma: no cover - internal bytes
            raise RunnerContractError("cached p_pre_zero policy is invalid")
        return detached_probability, detached_policy

    @staticmethod
    def _expects_realized_prezero(
        entry: RunPlanEntry,
        run: Mapping[str, object],
    ) -> bool:
        """Derive score presence from plan and post-execution provenance."""

        if entry.method_id != "maskimpute" or not entry.requires_count_score:
            return False
        status = run.get("status")
        reason = run.get("reason")
        native_provenance = any(
            run.get(name) is not None
            for name in (
                "native_output_sha256",
                "native_output_path",
                "native_output_file_sha256",
                "native_output_shape",
                "native_output_dtype",
                "native_output_scale",
            )
        )
        conversion_terminal = (
            status == "unavailable"
            and isinstance(reason, str)
            and _EVALUATOR_CONVERSION_REASON.fullmatch(reason) is not None
        )
        return status == "completed" or native_provenance or conversion_terminal

    def _ensure_root(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self.output_dir.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise RunnerContractError(
                "checkpoint output directory must not be a symlink"
            )

    def _safe_artifact_path(self, relative_value: object, name: str) -> Path:
        if not isinstance(relative_value, str):
            raise RunnerContractError(f"{name} path must be a string")
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RunnerContractError(f"{name} path is unsafe")
        path = self.output_dir.joinpath(*relative.parts)
        for component in (path, *path.parents):
            if component == self.output_dir.parent:
                break
            if os.path.lexists(component) and stat.S_ISLNK(component.lstat().st_mode):
                raise RunnerContractError(f"{name} path contains a symlink")
        return path

    @staticmethod
    def _transaction_artifact_paths(
        attempt: EvaluatedAttempt,
    ) -> tuple[str, ...]:
        base = f"runs/{attempt.run.run_id}"
        paths = {
            f"{base}.stdout",
            f"{base}.stderr",
            *(() if attempt.native_output is None else (f"{base}.native-f64",)),
            *(() if attempt.evaluator_output is None else (f"{base}.log2-cp10k-f64",)),
            *(
                ()
                if attempt.p_pre_zero_evidence.raw_matrix_bytes is None
                else (f"{base}.p-pre-zero-f64.zlib",)
            ),
        }
        return tuple(sorted(paths))

    @staticmethod
    def _allowed_transaction_artifact_paths(entry: RunPlanEntry) -> frozenset[str]:
        base = f"runs/{entry.run_id}"
        return frozenset(
            {
                f"{base}.stdout",
                f"{base}.stderr",
                f"{base}.native-f64",
                f"{base}.log2-cp10k-f64",
                f"{base}.p-pre-zero-f64.zlib",
            }
        )

    @staticmethod
    def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
        return _stable_stat_identity(value)

    @staticmethod
    def _staging_canonical_name(
        name: str,
        prefixes: Mapping[str, str],
    ) -> str | None:
        for canonical_name, prefix in prefixes.items():
            if not name.startswith(prefix) or not name.endswith(".tmp"):
                continue
            token = name[len(prefix) : -len(".tmp")]
            if _MKSTEMP_TOKEN.fullmatch(token) is not None:
                return canonical_name
        return None

    @staticmethod
    def _owned_directory_entries(
        path: Path,
        name: str,
        *,
        missing_ok: bool = False,
    ) -> tuple[Path, ...] | None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise RunnerContractError(f"{name} is unavailable")
        except OSError as error:
            raise RunnerContractError(f"{name} is unavailable") from error
        try:
            canonical = path.resolve(strict=True) == path.absolute()
        except OSError as error:
            raise RunnerContractError(f"{name} is not canonical") from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or not canonical
        ):
            raise RunnerContractError(
                f"{name} must be an owned canonical nonsymlink directory"
            )
        try:
            return tuple(sorted(path.iterdir(), key=lambda item: item.name))
        except OSError as error:
            raise RunnerContractError(f"{name} cannot be enumerated") from error

    @staticmethod
    def _require_owned_unique_file(path: Path, name: str) -> None:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RunnerContractError(f"{name} is unavailable") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise RunnerContractError(f"{name} must be an owned unique regular file")

    def _remove_stale_publication_temporaries(
        self,
        plan: CompetitionPlan,
    ) -> tuple[str, ...]:
        """Remove only exact owned mkstemp files from development-owned paths."""

        root_entries = self._owned_directory_entries(
            self.output_dir,
            "checkpoint output directory",
            missing_ok=True,
        )
        if root_entries is None:
            return ()
        root_files = {"checkpoint.json": self.checkpoint_path}
        root_directories = {"runs", "transactions"}
        root_prefixes = {"checkpoint.json": ".checkpoint."}

        transaction_files = {
            f"{position:08d}.json": self.output_dir
            / "transactions"
            / f"{position:08d}.json"
            for position in range(1, len(plan.entries) + 1)
        }
        transaction_prefixes = {name: f".{name}." for name in transaction_files}
        run_files = {
            PurePosixPath(relative).name: self.output_dir / relative
            for entry in plan.entries
            for relative in self._allowed_transaction_artifact_paths(entry)
        }
        run_prefixes = {name: f".{name}." for name in run_files}
        temporaries: list[tuple[Path, Path, str]] = []

        def classify(
            entries: tuple[Path, ...],
            *,
            files: Mapping[str, Path],
            directories: frozenset[str] = frozenset(),
            prefixes: Mapping[str, str],
            location: str,
        ) -> None:
            for entry in entries:
                if entry.name in files or entry.name in directories:
                    continue
                canonical_name = self._staging_canonical_name(entry.name, prefixes)
                if canonical_name is None:
                    raise RunnerContractError(
                        f"{location} contains an unexpected path: {entry.name}"
                    )
                temporaries.append(
                    (
                        entry,
                        files[canonical_name],
                        f"{location} staging temporary",
                    )
                )

        classify(
            root_entries,
            files=root_files,
            directories=frozenset(root_directories),
            prefixes=root_prefixes,
            location="checkpoint output",
        )
        transactions = self.output_dir / "transactions"
        transaction_entries = self._owned_directory_entries(
            transactions,
            "development transaction directory",
            missing_ok=True,
        )
        if transaction_entries is not None:
            classify(
                transaction_entries,
                files=transaction_files,
                prefixes=transaction_prefixes,
                location="development transaction directory",
            )
        runs = self.output_dir / "runs"
        run_entries = self._owned_directory_entries(
            runs,
            "development run directory",
            missing_ok=True,
        )
        if run_entries is not None:
            classify(
                run_entries,
                files=run_files,
                prefixes=run_prefixes,
                location="development run directory",
            )

        removed: list[str] = []
        for temporary, canonical, name in sorted(
            temporaries, key=lambda item: item[0].as_posix()
        ):
            _unlink_owned_staging_temporary(temporary, canonical, name)
            removed.append(temporary.relative_to(self.output_dir).as_posix())

        root_entries = self._owned_directory_entries(
            self.output_dir,
            "checkpoint output directory",
        )
        assert root_entries is not None
        classify(
            root_entries,
            files=root_files,
            directories=frozenset(root_directories),
            prefixes={},
            location="checkpoint output",
        )
        if os.path.lexists(self.checkpoint_path):
            self._require_owned_unique_file(
                self.checkpoint_path,
                "development checkpoint",
            )
        for directory, files, name in (
            (transactions, transaction_files, "development transaction"),
            (runs, run_files, "development run artifact"),
        ):
            entries = self._owned_directory_entries(
                directory,
                f"{name} directory",
                missing_ok=True,
            )
            if entries is None:
                continue
            classify(
                entries,
                files=files,
                prefixes={},
                location=f"{name} directory",
            )
            for entry in entries:
                self._require_owned_unique_file(entry, name)
        return tuple(removed)

    def _read_owned_transaction_file(
        self,
        path: Path,
        name: str,
    ) -> tuple[bytes, tuple[int, ...]]:
        descriptor = -1
        try:
            before = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or before.st_size > 1024 * 1024
            ):
                raise RunnerContractError(
                    f"{name} must be an owned bounded unique regular file"
                )
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if self._stat_identity(before) != self._stat_identity(opened):
                raise RunnerContractError(f"{name} changed while opening")
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024 + 1 - consumed, 65536))
                if not chunk:
                    break
                chunks.append(chunk)
                consumed += len(chunk)
                if consumed > 1024 * 1024:
                    raise RunnerContractError(f"{name} exceeds its byte bound")
            after = os.fstat(descriptor)
            after_path = path.lstat()
            if self._stat_identity(opened) != self._stat_identity(
                after
            ) or self._stat_identity(opened) != self._stat_identity(after_path):
                raise RunnerContractError(f"{name} changed while reading")
            return b"".join(chunks), self._stat_identity(opened)
        except RunnerContractError:
            raise
        except OSError as error:
            raise RunnerContractError(f"{name} is unavailable") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _checkpoint_transaction_snapshot(
        self,
        plan: CompetitionPlan,
    ) -> tuple[bytes | None, tuple[frozenset[str], ...]]:
        if not os.path.lexists(self.checkpoint_path):
            return None, ()
        raw = self._read_checkpoint_bytes()
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise RunnerContractError(
                "checkpoint transaction prefix is invalid"
            ) from error
        if (
            not isinstance(payload, dict)
            or set(payload) != self._CHECKPOINT_KEYS
            or raw != _canonical_bytes(payload) + b"\n"
            or payload.get("schema_version") != 1
            or payload.get("plan_sha256") != plan.plan_sha256
            or payload.get("input_hashes") != dict(plan.input_hashes)
            or payload.get("planned_run_count") != len(plan.entries)
        ):
            raise RunnerContractError(
                "checkpoint transaction prefix differs from its plan"
            )
        checksum_body = {
            key: nested for key, nested in payload.items() if key != "checkpoint_sha256"
        }
        if payload.get("checkpoint_sha256") != canonical_sha256(checksum_body):
            raise RunnerContractError("checkpoint transaction prefix checksum differs")
        records = payload.get("records")
        if not isinstance(records, list) or len(records) > len(plan.entries):
            raise RunnerContractError(
                "checkpoint transaction records are not a plan prefix"
            )
        expected_status = (
            "completed" if len(records) == len(plan.entries) else "running"
        )
        if payload.get("status") != expected_status:
            raise RunnerContractError(
                "checkpoint transaction status contradicts its prefix"
            )
        references: list[frozenset[str]] = []
        for record, entry in zip(records, plan.entries, strict=False):
            if not isinstance(record, Mapping) or set(record) != {
                "run",
                "metrics",
                "p_pre_zero_evidence",
            }:
                raise RunnerContractError(
                    "checkpoint transaction record has wrong schema"
                )
            run = record.get("run")
            evidence = record.get("p_pre_zero_evidence")
            storage = evidence.get("storage") if isinstance(evidence, Mapping) else None
            if (
                not isinstance(run, Mapping)
                or run.get("run_id") != entry.run_id
                or not isinstance(storage, Mapping)
            ):
                raise RunnerContractError(
                    "checkpoint transaction record differs from its plan"
                )
            observed: set[str] = set()
            for prefix in ("stdout", "stderr", "native_output", "evaluator_output"):
                relative = run.get(f"{prefix}_path")
                if relative is not None:
                    if not isinstance(relative, str):
                        raise RunnerContractError(
                            "checkpoint transaction artifact path is invalid"
                        )
                    observed.add(relative)
            score_relative = storage.get("path")
            if score_relative is not None:
                if not isinstance(score_relative, str):
                    raise RunnerContractError(
                        "checkpoint transaction score path is invalid"
                    )
                observed.add(score_relative)
            allowed = self._allowed_transaction_artifact_paths(entry)
            required = {
                f"runs/{entry.run_id}.stdout",
                f"runs/{entry.run_id}.stderr",
            }
            if not required.issubset(observed) or not observed.issubset(allowed):
                raise RunnerContractError(
                    "checkpoint transaction artifact paths differ from their run"
                )
            references.append(frozenset(observed))
        return raw, tuple(references)

    def _assert_checkpoint_transaction_snapshot(
        self,
        snapshot: bytes | None,
    ) -> None:
        if snapshot is None:
            if os.path.lexists(self.checkpoint_path):
                raise RunnerContractError(
                    "checkpoint prefix appeared during transaction recovery"
                )
            return
        if not os.path.lexists(self.checkpoint_path):
            raise RunnerContractError(
                "checkpoint prefix disappeared during transaction recovery"
            )
        if self._read_checkpoint_bytes() != snapshot:
            raise RunnerContractError(
                "checkpoint prefix changed during transaction recovery"
            )

    def _unlink_closed_transaction_file(
        self,
        path: Path,
        name: str,
        *,
        checkpoint_snapshot: bytes | None,
        expected_identity: tuple[int, ...] | None = None,
        missing_ok: bool = False,
    ) -> bool:
        descriptor = -1
        try:
            try:
                before = path.lstat()
            except FileNotFoundError:
                if missing_ok:
                    return False
                raise
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or (
                    expected_identity is not None
                    and self._stat_identity(before) != expected_identity
                )
            ):
                raise RunnerContractError(f"{name} is not an owned unique closed file")
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if self._stat_identity(before) != self._stat_identity(opened):
                raise RunnerContractError(f"{name} changed while opening")
            self._assert_checkpoint_transaction_snapshot(checkpoint_snapshot)
            after_path = path.lstat()
            if self._stat_identity(opened) != self._stat_identity(after_path):
                raise RunnerContractError(f"{name} changed before removal")
            path.unlink()
            if os.fstat(descriptor).st_nlink != 0 or os.path.lexists(path):
                raise RunnerContractError(f"{name} survived removal")
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return True
        except RunnerContractError:
            raise
        except OSError as error:
            raise RunnerContractError(f"{name} could not be removed safely") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _recover_interrupted_transactions(
        self,
        plan: CompetitionPlan,
    ) -> tuple[int, ...]:
        """Close committed intents and roll back only uncommitted artifacts."""

        self._remove_stale_publication_temporaries(plan)
        transactions = self.output_dir / "transactions"
        if not os.path.lexists(transactions):
            return ()
        try:
            metadata = transactions.lstat()
            names = tuple(sorted(path.name for path in transactions.iterdir()))
        except OSError as error:
            raise RunnerContractError(
                "development transaction directory cannot be enumerated"
            ) from error
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            raise RunnerContractError("development transaction directory is invalid")
        checkpoint_snapshot, committed = self._checkpoint_transaction_snapshot(plan)
        committed_all = frozenset().union(*committed)
        recovered: list[int] = []
        for name in names:
            if _DEVELOPMENT_TRANSACTION_FILE.fullmatch(name) is None:
                raise RunnerContractError("development transaction name is invalid")
            position = int(name.removesuffix(".json")) - 1
            if position < 0 or position >= len(plan.entries):
                raise RunnerContractError("development transaction position is invalid")
            entry = plan.entries[position]
            intent_path = transactions / name
            raw, identity = self._read_owned_transaction_file(
                intent_path,
                "development transaction intent",
            )
            try:
                intent = json.loads(
                    raw.decode("utf-8"),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_json_object,
                )
            except (UnicodeError, ValueError, json.JSONDecodeError) as error:
                raise RunnerContractError(
                    "development transaction intent is invalid"
                ) from error
            body = (
                {
                    key: nested
                    for key, nested in intent.items()
                    if key != "intent_sha256"
                }
                if isinstance(intent, Mapping)
                else {}
            )
            artifact_paths = (
                intent.get("artifact_paths") if isinstance(intent, Mapping) else None
            )
            allowed = self._allowed_transaction_artifact_paths(entry)
            required = {
                f"runs/{entry.run_id}.stdout",
                f"runs/{entry.run_id}.stderr",
            }
            if (
                not isinstance(intent, dict)
                or set(intent)
                != {
                    "schema_version",
                    "plan_sha256",
                    "position",
                    "ordinal",
                    "run_id",
                    "artifact_paths",
                    "intent_sha256",
                }
                or intent.get("schema_version") != 1
                or intent.get("plan_sha256") != plan.plan_sha256
                or intent.get("position") != position
                or intent.get("ordinal") != entry.ordinal
                or intent.get("run_id") != entry.run_id
                or not isinstance(artifact_paths, list)
                or not all(isinstance(item, str) for item in artifact_paths)
                or artifact_paths != sorted(artifact_paths)
                or len(set(artifact_paths)) != len(artifact_paths)
                or not required.issubset(set(artifact_paths))
                or not set(artifact_paths).issubset(allowed)
                or intent.get("intent_sha256") != canonical_sha256(body)
                or raw != _canonical_bytes(intent) + b"\n"
            ):
                raise RunnerContractError("development transaction intent is invalid")
            if position > len(committed):
                raise RunnerContractError(
                    "development transaction is beyond the checkpoint prefix"
                )
            if position < len(committed):
                if set(artifact_paths) != set(committed[position]):
                    raise RunnerContractError(
                        "committed development transaction differs from its record"
                    )
            else:
                if set(artifact_paths) & committed_all:
                    raise RunnerContractError(
                        "interrupted development artifacts are checkpoint-referenced"
                    )
                for relative in artifact_paths:
                    path = self._safe_artifact_path(
                        relative,
                        "interrupted development artifact",
                    )
                    self._unlink_closed_transaction_file(
                        path,
                        "interrupted development artifact",
                        checkpoint_snapshot=checkpoint_snapshot,
                        missing_ok=True,
                    )
            self._unlink_closed_transaction_file(
                intent_path,
                "development transaction intent",
                checkpoint_snapshot=checkpoint_snapshot,
                expected_identity=identity,
            )
            recovered.append(entry.ordinal)
        try:
            transactions.rmdir()
        except OSError as error:
            raise RunnerContractError(
                "development transaction directory did not become empty"
            ) from error
        directory = os.open(self.output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        runs = self.output_dir / "runs"
        try:
            runs.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        return tuple(recovered)

    def _publish_transaction_intent(
        self,
        plan: CompetitionPlan,
        position: int,
        entry: RunPlanEntry,
        attempt: EvaluatedAttempt,
    ) -> Path:
        body: dict[str, object] = {
            "schema_version": 1,
            "plan_sha256": plan.plan_sha256,
            "position": position,
            "ordinal": entry.ordinal,
            "run_id": entry.run_id,
            "artifact_paths": list(self._transaction_artifact_paths(attempt)),
        }
        intent = {**body, "intent_sha256": canonical_sha256(body)}
        relative, _digest = self._publish_immutable(
            f"transactions/{position + 1:08d}.json",
            _canonical_bytes(intent) + b"\n",
        )
        return self.output_dir / relative

    def _publish_immutable(self, relative: str, data: bytes) -> tuple[str, str]:
        self._ensure_root()
        path = self._safe_artifact_path(relative, "run artifact")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise RunnerContractError(
                    f"refusing to replace run artifact {relative}"
                )
            return relative, hashlib.sha256(data).hexdigest()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise RunnerContractError(
                        f"conflicting run artifact appeared at {relative}"
                    )
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return relative, hashlib.sha256(data).hexdigest()

    def _publish_checkpoint(self, payload: Mapping[str, object]) -> None:
        self._ensure_root()
        encoded = _canonical_bytes(payload) + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".checkpoint.", suffix=".tmp", dir=self.output_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.checkpoint_path)
            directory = os.open(self.output_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _stored_attempt(self, attempt: EvaluatedAttempt) -> dict[str, object]:
        run = attempt.run.to_dict()
        base = f"runs/{attempt.run.run_id}"
        stdout_path, stdout_file_sha256 = self._publish_immutable(
            f"{base}.stdout", attempt.stdout
        )
        stderr_path, stderr_file_sha256 = self._publish_immutable(
            f"{base}.stderr", attempt.stderr
        )
        run.update(
            {
                "stdout_path": stdout_path,
                "stdout_file_sha256": stdout_file_sha256,
                "stderr_path": stderr_path,
                "stderr_file_sha256": stderr_file_sha256,
                "native_output_path": None,
                "native_output_file_sha256": None,
                "native_output_shape": None,
                "native_output_dtype": None,
                "native_output_scale": attempt.native_output_scale,
                "evaluator_output_path": None,
                "evaluator_output_file_sha256": None,
                "evaluator_output_shape": None,
                "evaluator_output_dtype": None,
                "evaluator_scale": None,
            }
        )
        if attempt.native_output is not None:
            native = np.asarray(attempt.native_output, dtype="<f8", order="C")
            path, digest = self._publish_immutable(
                f"{base}.native-f64", native.tobytes(order="C")
            )
            run.update(
                {
                    "native_output_path": path,
                    "native_output_file_sha256": digest,
                    "native_output_shape": list(native.shape),
                    "native_output_dtype": "<f8",
                }
            )
        if attempt.evaluator_output is not None:
            evaluator = np.asarray(attempt.evaluator_output, dtype="<f8", order="C")
            path, digest = self._publish_immutable(
                f"{base}.log2-cp10k-f64", evaluator.tobytes(order="C")
            )
            run.update(
                {
                    "evaluator_output_path": path,
                    "evaluator_output_file_sha256": digest,
                    "evaluator_output_shape": list(evaluator.shape),
                    "evaluator_output_dtype": "<f8",
                    "evaluator_scale": "log2_cp10k_plus_1",
                }
            )
        p_pre_zero_evidence, compressed_p_pre_zero = encode_prezero_evidence(
            attempt.p_pre_zero_evidence
        )
        if compressed_p_pre_zero is not None:
            path, digest = self._publish_immutable(
                f"{base}.p-pre-zero-f64.zlib", compressed_p_pre_zero
            )
            storage = p_pre_zero_evidence["storage"]
            assert isinstance(storage, dict)
            if digest != storage["compressed_sha256"]:
                raise RunnerContractError(
                    "published p_pre_zero checksum differs from its evidence"
                )
            storage["path"] = path
        return {
            "run": run,
            "metrics": [metric.to_dict() for metric in attempt.metrics],
            "p_pre_zero_evidence": p_pre_zero_evidence,
        }

    def write(
        self,
        plan: CompetitionPlan,
        records: Sequence[Mapping[str, object]],
        budget: DevelopmentBudget,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> CheckpointReport:
        _validate_prepared_dataset_authority(plan, prepared_datasets)
        record_values = tuple(dict(record) for record in records)
        status = "completed" if len(record_values) == len(plan.entries) else "running"
        comparator_selection_status = _comparator_selection_status(
            plan.entries,
            record_values,
        )
        body: dict[str, object] = {
            "schema_version": 1,
            "plan_sha256": plan.plan_sha256,
            "input_hashes": dict(plan.input_hashes),
            "planned_run_count": len(plan.entries),
            "status": status,
            "evaluation_scope": "reconstruction_only",
            "comparator_selection_status": comparator_selection_status,
            "selection_complete": False,
            "selection_blockers": list(SELECTION_COMPLETENESS_BLOCKERS),
            "records": list(record_values),
            "budget": budget.to_dict(),
        }
        body["checkpoint_sha256"] = canonical_sha256(body)
        self._publish_checkpoint(body)
        return self.load(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def append(
        self,
        plan: CompetitionPlan,
        report: CheckpointReport | None,
        attempt: EvaluatedAttempt,
        budget: DevelopmentBudget,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> CheckpointReport:
        prepared_by_dataset = _validate_prepared_dataset_authority(
            plan, prepared_datasets
        )
        self._recover_interrupted_transactions(plan)
        records = [] if report is None else list(report.records)
        if len(records) >= len(plan.entries):
            raise RunnerContractError("checkpoint already contains its full plan")
        entry = plan.entries[len(records)]
        prepared = prepared_by_dataset[entry.dataset_id]
        with tempfile.TemporaryDirectory(
            prefix="maskimpute-checkpoint-stage-"
        ) as staging_name:
            staging = CheckpointStore(
                Path(staging_name),
                authority_repository=self.authority_repository,
            )
            staged_record = json.loads(
                _canonical_bytes(staging._stored_attempt(attempt)).decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
            staging._validate_stored_record(
                staged_record,
                entry,
                prepared=prepared,
                expected_dataset_qc_policy_sha256=plan.input_hashes.get(
                    "dataset_qc_policy_sha256"
                ),
                expected_score_config_sha256=plan.input_hashes.get(
                    "count_model_config_sha256"
                ),
                expected_calibration_file_sha256=plan.input_hashes.get(
                    "retained_calibration_sha256"
                ),
                execution_authority=plan.execution_context,
                calibration_usage="development_holdout",
            )
            self._prezero_authority_cache.update(staging._prezero_authority_cache)
        self._publish_transaction_intent(
            plan,
            len(records),
            entry,
            attempt,
        )
        records.append(self._stored_attempt(attempt))
        return self.write(
            plan,
            records,
            budget,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )

    def _verify_artifact(self, relative: object, digest: object, name: str) -> Path:
        expected = _require_sha256(digest, f"{name} file checksum")
        assert expected is not None
        path = self._safe_artifact_path(relative, name)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RunnerContractError(f"{name} artifact is missing") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise RunnerContractError(f"{name} artifact must be a unique regular file")
        if _file_sha256(path) != expected:
            raise RunnerContractError(f"{name} artifact checksum mismatch")
        return path

    def _read_bounded_artifact(
        self,
        relative: object,
        digest: object,
        name: str,
        *,
        max_bytes: int,
    ) -> bytes:
        """Read one unique immutable artifact without exceeding a byte bound."""

        expected = _require_sha256(digest, f"{name} file checksum")
        assert expected is not None
        if isinstance(max_bytes, bool) or type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        path = self._safe_artifact_path(relative, name)
        descriptor = -1
        try:
            before = path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > max_bytes
            ):
                raise RunnerContractError(
                    f"{name} artifact is not a bounded unique regular file"
                )
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            identity = lambda item: (  # noqa: E731 - compact stat identity
                item.st_dev,
                item.st_ino,
                item.st_mode,
                item.st_nlink,
                item.st_uid,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            if identity(before) != identity(opened):
                raise RunnerContractError(f"{name} artifact changed while opening")
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - consumed))
                if not chunk:
                    break
                chunks.append(chunk)
                consumed += len(chunk)
                if consumed > max_bytes:
                    raise RunnerContractError(f"{name} artifact exceeds its byte bound")
            after = os.fstat(descriptor)
            after_path = path.lstat()
            if identity(opened) != identity(after) or identity(opened) != identity(
                after_path
            ):
                raise RunnerContractError(f"{name} artifact changed while reading")
            raw = b"".join(chunks)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise RunnerContractError(f"{name} artifact checksum mismatch")
            return raw
        except RunnerContractError:
            raise
        except OSError as error:
            raise RunnerContractError(f"{name} artifact is unavailable") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _read_checkpoint_bytes(self) -> bytes:
        descriptor = -1
        try:
            before = self.checkpoint_path.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
            ):
                raise RunnerContractError(
                    "checkpoint must be an owned unique regular file, not a symlink"
                )
            descriptor = os.open(
                self.checkpoint_path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_uid != os.geteuid()
                or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise RunnerContractError(
                    "checkpoint changed while opening or is not a regular file"
                )
            chunks = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            after_path = self.checkpoint_path.lstat()
            identity = lambda value: (  # noqa: E731 - compact stable-stat tuple
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_nlink,
                value.st_uid,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if identity(opened) != identity(after) or identity(opened) != identity(
                after_path
            ):
                raise RunnerContractError("checkpoint changed while being read")
            return b"".join(chunks)
        except RunnerContractError:
            raise
        except OSError as error:
            raise RunnerContractError(
                "checkpoint is unavailable or cannot be read without following links"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _validate_stored_record(
        self,
        value: object,
        entry: RunPlanEntry,
        *,
        prepared: PreparedDataset,
        expected_dataset_qc_policy_sha256: str | None,
        expected_score_config_sha256: str | None,
        expected_calibration_file_sha256: str | None,
        execution_authority: ExecutionAuthorityContext | None = None,
        calibration_usage: str = "development_holdout",
    ) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {
            "run",
            "metrics",
            "p_pre_zero_evidence",
        }:
            raise RunnerContractError("checkpoint record has wrong schema")
        run = value["run"]
        metrics = value["metrics"]
        if not isinstance(run, dict) or not isinstance(metrics, list):
            raise RunnerContractError("checkpoint record payload is malformed")
        direct_comparator = entry.configuration_kind in {
            "comparator_tuning",
            "comparator_nonexecution",
        }
        expected_run_fields = (
            ("run_id", entry.run_id),
            ("method_id", entry.method_id),
            ("dataset_id", entry.dataset_id),
            ("source_dataset_sha256", entry.source_dataset_sha256),
            ("model_seed", entry.model_seed),
            ("configuration_id", entry.configuration_id),
            ("configuration_kind", entry.configuration_kind),
            ("requires_count_score", entry.requires_count_score),
            ("requires_calibration", entry.requires_calibration),
            ("mechanism", entry.mechanism),
            ("biological_id", entry.biological_id),
            ("technical_view", entry.technical_view),
        )
        for name, expected in (
            expected_run_fields
            if direct_comparator
            else (
                *expected_run_fields,
                ("configuration_sha256", entry.configuration_sha256),
            )
        ):
            if run.get(name) != expected:
                raise RunnerContractError(
                    f"checkpoint run {name} mismatches plan prefix"
                )
        from .direct_values import direct_equal

        if direct_comparator:
            if "configuration_sha256" in run:
                raise RunnerContractError(
                    "direct comparator record contains a configuration summary"
                )
            expected_configuration = (
                None
                if entry.comparator_configuration is None
                else direct_bound_comparator_value(entry.comparator_configuration)
            )
            if not direct_equal(
                run.get("comparator_configuration"),
                expected_configuration,
            ) or not direct_equal(
                run.get("comparator_nonexecution_identity"),
                entry.comparator_nonexecution_identity,
            ):
                raise RunnerContractError(
                    "direct comparator record differs from complete plan identity"
                )
        elif (
            "comparator_configuration" in run
            or "comparator_nonexecution_identity" in run
        ):
            raise RunnerContractError(
                "legacy checkpoint record contains comparator identity fields"
            )
        if run.get("status") not in _OUTCOME_STATUSES:
            raise RunnerContractError("checkpoint run status is invalid")
        authoritative_method_input_sha256 = method_input_sha256(prepared.method_input)
        for name, expected in (
            ("method_input_sha256", authoritative_method_input_sha256),
            ("dataset_qc_policy_sha256", expected_dataset_qc_policy_sha256),
            ("excluded_cell_count", prepared.audit.excluded_cell_count),
            ("excluded_cell_ids_sha256", prepared.audit.excluded_cell_ids_sha256),
            ("retained_cell_count", prepared.audit.retained_cell_count),
            ("retained_cell_ids_sha256", prepared.audit.retained_cell_ids_sha256),
            ("retained_gene_count", prepared.method_input.shape[1]),
            (
                "observed_zero_count",
                int((prepared.method_input.counts == 0.0).sum()),
            ),
        ):
            if run.get(name) != expected:
                raise RunnerContractError(
                    f"checkpoint run {name} differs from prepared-data authority"
                )
        _require_nonnegative_number(run.get("runtime_seconds"), "checkpoint runtime")
        for name in (
            "retained_cell_count",
            "retained_gene_count",
            "observed_zero_count",
        ):
            nested = run.get(name)
            if isinstance(nested, bool) or type(nested) is not int or nested < 0:
                raise RunnerContractError(f"checkpoint {name} is invalid")
        if run.get("retained_cell_count") <= 0 or run.get("retained_gene_count") <= 0:
            raise RunnerContractError(
                "checkpoint retained matrix dimensions are invalid"
            )
        if run.get("observed_zero_count") > (
            run["retained_cell_count"] * run["retained_gene_count"]
        ):
            raise RunnerContractError(
                "checkpoint observed_zero_count exceeds retained matrix entries"
            )
        calibration_artifact = run.get("calibration_artifact_sha256")
        if calibration_artifact is None:
            if (
                run.get("calibration_context_sha256") is not None
                or run.get("calibration_fold_calibrator_sha256") is not None
                or run.get("calibration_training_manifest_sha256s") != []
                or run.get("calibration_held_out_manifest_sha256s") != []
            ):
                raise RunnerContractError("checkpoint calibration receipt is partial")
            if (
                entry.requires_calibration
                and run.get("status") == "completed"
                and expected_calibration_file_sha256 is None
            ):
                raise RunnerContractError(
                    "calibrated completed run lacks its LODO receipt"
                )
            if expected_calibration_file_sha256 is not None:
                _require_sha256(
                    expected_calibration_file_sha256,
                    "checkpoint final calibration artifact",
                )
        else:
            for name in (
                "calibration_artifact_sha256",
                "calibration_context_sha256",
                "calibration_fold_calibrator_sha256",
            ):
                _require_sha256(run.get(name), f"checkpoint {name}")
            for name in (
                "calibration_training_manifest_sha256s",
                "calibration_held_out_manifest_sha256s",
            ):
                values = run.get(name)
                if not isinstance(values, list) or not values:
                    raise RunnerContractError(f"checkpoint {name} is invalid")
                for digest in values:
                    _require_sha256(digest, f"checkpoint {name}")
            if (
                expected_calibration_file_sha256 is not None
                and calibration_artifact != expected_calibration_file_sha256
            ):
                raise RunnerContractError(
                    "checkpoint calibration artifact differs from execution authority"
                )
        for stream in ("stdout", "stderr"):
            path = self._verify_artifact(
                run.get(f"{stream}_path"),
                run.get(f"{stream}_file_sha256"),
                stream,
            )
            expected_content = _require_sha256(
                run.get(f"{stream}_sha256"), f"{stream} content checksum"
            )
            assert expected_content is not None
            if _file_sha256(path) != expected_content:
                raise RunnerContractError(f"{stream} content checksum mismatch")
        for prefix in ("native_output", "evaluator_output"):
            path_value = run.get(f"{prefix}_path")
            if path_value is None:
                if any(
                    run.get(f"{prefix}_{suffix}") is not None
                    for suffix in ("file_sha256", "shape", "dtype")
                ):
                    raise RunnerContractError(f"checkpoint {prefix} binding is partial")
                continue
            path = self._verify_artifact(
                path_value, run.get(f"{prefix}_file_sha256"), prefix
            )
            shape = run.get(f"{prefix}_shape")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(item) is not int or item <= 0 for item in shape)
                or run.get(f"{prefix}_dtype") != "<f8"
                or path.stat().st_size != shape[0] * shape[1] * 8
            ):
                raise RunnerContractError(
                    f"checkpoint {prefix} shape or dtype is invalid"
                )
        evidence_value = value.get("p_pre_zero_evidence")
        compressed_p_pre_zero: bytes | None = None
        if isinstance(evidence_value, Mapping):
            storage_value = evidence_value.get("storage")
            if (
                isinstance(storage_value, Mapping)
                and storage_value.get("path") is not None
            ):
                expected_uncompressed = (
                    run["retained_cell_count"] * run["retained_gene_count"] * 8
                )
                compressed_p_pre_zero = self._read_bounded_artifact(
                    storage_value.get("path"),
                    storage_value.get("compressed_sha256"),
                    "p_pre_zero",
                    max_bytes=zlib_compress_bound(expected_uncompressed),
                )
        expected_identity = {
            "run_id": entry.run_id,
            "method_id": entry.method_id,
            "dataset_id": entry.dataset_id,
            "source_dataset_sha256": entry.source_dataset_sha256,
            "mechanism": entry.mechanism,
            "biological_id": entry.biological_id,
            "technical_view": entry.technical_view,
            "model_seed": entry.model_seed,
            "configuration_id": entry.configuration_id,
            "configuration_sha256": entry.configuration_sha256,
            "method_input_sha256": authoritative_method_input_sha256,
            "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
        }
        expected_matrix_present = self._expects_realized_prezero(entry, run)
        expected_probability, expected_policy = self._expected_prezero_authority(
            entry,
            prepared,
            execution_authority,
            calibration_usage=calibration_usage,
            expected_matrix_present=expected_matrix_present,
        )
        observed, truth, truth_kind = _prezero_evaluator_targets(prepared)
        try:
            validate_stored_prezero_evidence(
                evidence_value,
                expected_identity=expected_identity,
                run_status=str(run.get("status")),
                run_reason=(
                    None if run.get("reason") is None else str(run.get("reason"))
                ),
                observed_zero_count=run["observed_zero_count"],
                expected_shape=(
                    run["retained_cell_count"],
                    run["retained_gene_count"],
                ),
                requires_count_score=entry.requires_count_score,
                requires_calibration=entry.requires_calibration,
                expected_calibration_file_sha256=(expected_calibration_file_sha256),
                compressed=compressed_p_pre_zero,
                observed=observed,
                truth=truth,
                truth_kind=truth_kind,
                expected_score_input_sha256=_count_score_input_sha256(prepared),
                expected_score_config_sha256=expected_score_config_sha256,
                expected_matrix_present=expected_matrix_present,
                expected_probability=expected_probability,
                expected_policy=expected_policy,
            )
        except PreZeroEvidenceError as error:
            raise RunnerContractError(str(error)) from error
        if len(metrics) != len(_RECONSTRUCTION_METRIC_NAMES):
            raise RunnerContractError("checkpoint metric denominator is incomplete")
        for metric, expected_name in zip(
            metrics, _RECONSTRUCTION_METRIC_NAMES, strict=True
        ):
            if not isinstance(metric, dict) or metric.get("metric") != expected_name:
                raise RunnerContractError(
                    "checkpoint metrics are not canonically ordered"
                )
            metric_expected_fields = (
                ("method", entry.method_id),
                ("dataset_id", entry.dataset_id),
                ("model_seed", entry.model_seed),
            )
            for name, expected in (
                metric_expected_fields
                if direct_comparator
                else (
                    *metric_expected_fields,
                    ("configuration_sha256", entry.configuration_sha256),
                )
            ):
                if metric.get(name) != expected:
                    raise RunnerContractError(
                        f"checkpoint metric {name} mismatches plan"
                    )
            if direct_comparator:
                expected_configuration = (
                    None
                    if entry.comparator_configuration is None
                    else direct_bound_comparator_value(entry.comparator_configuration)
                )
                if (
                    "configuration_sha256" in metric
                    or not direct_equal(
                        metric.get("comparator_configuration"),
                        expected_configuration,
                    )
                    or not direct_equal(
                        metric.get("comparator_nonexecution_identity"),
                        entry.comparator_nonexecution_identity,
                    )
                ):
                    raise RunnerContractError(
                        "checkpoint metric direct comparator identity differs"
                    )
            elif (
                "comparator_configuration" in metric
                or "comparator_nonexecution_identity" in metric
            ):
                raise RunnerContractError(
                    "legacy checkpoint metric contains comparator identity fields"
                )
        return value

    def load(
        self,
        plan: CompetitionPlan,
        *,
        registry: MethodRegistry,
        prepared_datasets: Mapping[str, PreparedDataset],
    ) -> CheckpointReport:
        if not isinstance(plan, CompetitionPlan):
            raise TypeError("plan must be a CompetitionPlan")
        if not isinstance(registry, MethodRegistry):
            raise TypeError("registry must be a MethodRegistry")
        self._recover_interrupted_transactions(plan)
        prepared_by_dataset = _validate_prepared_dataset_authority(
            plan, prepared_datasets
        )
        expected_source_sha256 = _require_sha256(
            plan.input_hashes.get("implementation_source_sha256"),
            "plan implementation source checksum",
        )
        current_source_sha256 = implementation_source_sha256()
        if expected_source_sha256 != current_source_sha256:
            raise RunnerContractError(
                "checkpoint implementation source differs from the current code bytes"
            )
        try:
            raw = self._read_checkpoint_bytes()
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
        except RunnerContractError:
            raise
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise RunnerContractError(
                f"checkpoint cannot be loaded: {error}"
            ) from error
        if not isinstance(payload, dict) or set(payload) != self._CHECKPOINT_KEYS:
            raise RunnerContractError("checkpoint has wrong schema")
        if raw != _canonical_bytes(payload) + b"\n":
            raise RunnerContractError("checkpoint is not canonical JSON")
        expected_checksum = canonical_sha256(
            {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
        )
        if payload.get("checkpoint_sha256") != expected_checksum:
            raise RunnerContractError("checkpoint checksum mismatch")
        if payload.get("schema_version") != 1:
            raise RunnerContractError("checkpoint schema_version must be 1")
        if payload.get("input_hashes") != dict(plan.input_hashes):
            raise RunnerContractError(
                "checkpoint input hashes mismatch current authority"
            )
        if payload.get("plan_sha256") != plan.plan_sha256:
            raise RunnerContractError(
                "checkpoint plan checksum mismatches current plan"
            )
        if payload.get("planned_run_count") != len(plan.entries):
            raise RunnerContractError("checkpoint planned denominator changed")
        records_value = payload.get("records")
        if not isinstance(records_value, list) or len(records_value) > len(
            plan.entries
        ):
            raise RunnerContractError("checkpoint records are not a plan prefix")
        records = tuple(
            self._validate_stored_record(
                value,
                entry,
                prepared=prepared_by_dataset[entry.dataset_id],
                expected_dataset_qc_policy_sha256=plan.input_hashes.get(
                    "dataset_qc_policy_sha256"
                ),
                expected_score_config_sha256=plan.input_hashes.get(
                    "count_model_config_sha256"
                ),
                expected_calibration_file_sha256=plan.input_hashes.get(
                    "retained_calibration_sha256"
                ),
                execution_authority=plan.execution_context,
                calibration_usage="development_holdout",
            )
            for value, entry in zip(records_value, plan.entries, strict=False)
        )
        expected_status = (
            "completed" if len(records) == len(plan.entries) else "running"
        )
        if payload.get("status") != expected_status:
            raise RunnerContractError("checkpoint status contradicts its plan prefix")
        comparator_selection_status = _comparator_selection_status(
            plan.entries,
            records,
        )
        if payload.get("comparator_selection_status") != comparator_selection_status:
            raise RunnerContractError(
                "checkpoint comparator-selection status differs from its records"
            )
        if (
            payload.get("evaluation_scope") != "reconstruction_only"
            or payload.get("selection_complete") is not False
            or payload.get("selection_blockers")
            != list(SELECTION_COMPLETENESS_BLOCKERS)
        ):
            raise RunnerContractError(
                "reconstruction checkpoint overstates selection completeness"
            )
        budget = payload.get("budget")
        if not isinstance(budget, dict):
            raise RunnerContractError("checkpoint budget ledger is malformed")
        replayed_budget = replay_development_budget(
            registry,
            plan.entries,
            records,
        )
        if budget != replayed_budget.to_dict():
            raise RunnerContractError("checkpoint budget ledger differs from replay")
        return CheckpointReport(
            schema_version=1,
            plan_sha256=plan.plan_sha256,
            input_hashes=dict(plan.input_hashes),
            planned_run_count=len(plan.entries),
            status=expected_status,
            evaluation_scope="reconstruction_only",
            comparator_selection_status=comparator_selection_status,
            selection_complete=False,
            selection_blockers=SELECTION_COMPLETENESS_BLOCKERS,
            records=records,
            budget=budget,
            checkpoint_sha256=expected_checksum,
        )


def execute_competition_plan(
    plan: CompetitionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    executor: Callable[[ExecutionRequest], AdapterOutcome],
    checkpoint_store: CheckpointStore,
) -> CheckpointReport:
    """Execute or resume an immutable plan, checkpointing every denominator row."""

    if not isinstance(plan, CompetitionPlan):
        raise TypeError("plan must be a CompetitionPlan")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(prepared_datasets, Mapping):
        raise TypeError("prepared_datasets must be a mapping")
    if not callable(executor):
        raise TypeError("executor must be callable")
    if not isinstance(checkpoint_store, CheckpointStore):
        raise TypeError("checkpoint_store must be a CheckpointStore")
    plan.validate_integrity()
    report = (
        checkpoint_store.load(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        if checkpoint_store.checkpoint_path.exists()
        else None
    )
    budget = (
        DevelopmentBudget()
        if report is None
        else replay_development_budget(registry, plan.entries, report.records)
    )
    start = 0 if report is None else len(report.records)
    qc_policy_sha256 = plan.input_hashes.get(
        "dataset_qc_policy_sha256", DatasetQCPolicy.fixed().sha256
    )
    execution_context = plan.execution_context
    if execution_context is None:
        empty_config = "{}"
        execution_context = ExecutionAuthorityContext(
            authority_sha256=plan.plan_sha256,
            base_configuration_json=empty_config,
            base_configuration_sha256=canonical_sha256({}),
            count_model_config_json=empty_config,
            count_model_config_sha256=canonical_sha256({}),
            count_score_manifest_path=(
                "artifacts/study/development/count_scores/manifest.json"
            ),
            count_score_manifest_sha256=None,
            retained_calibration_path=(
                "artifacts/study/development/calibration/retained_calibration.json"
            ),
            retained_calibration_sha256=None,
        )
    for entry in plan.entries[start:]:
        try:
            spec = registry.by_id(entry.method_id)
        except KeyError as error:
            raise RunnerContractError(
                f"plan references unknown method {entry.method_id}"
            ) from error
        prepared = prepared_datasets.get(entry.dataset_id)
        if not isinstance(prepared, PreparedDataset):
            raise RunnerContractError(
                f"prepared dataset is missing for {entry.dataset_id}"
            )
        if (
            entry.method_id not in {"observed"}
            and entry.configuration_id == "registry-default"
        ):
            raise RunnerContractError(
                "publication comparator cannot use registry-default"
            )
        matching_configurations = tuple(
            value
            for value in plan.configurations
            if value.method_id == entry.method_id
            and value.configuration_id == entry.configuration_id
            and value.configuration_sha256 == entry.configuration_sha256
        )
        if len(matching_configurations) != 1:
            raise RunnerContractError(
                f"plan lacks one exact execution configuration for {entry.run_id}"
            )
        configuration = matching_configurations[0]
        if (
            configuration.kind != entry.configuration_kind
            or configuration.requires_count_score != entry.requires_count_score
            or configuration.requires_calibration != entry.requires_calibration
        ):
            raise RunnerContractError(
                f"plan entry execution flags mismatch {entry.run_id} configuration"
            )
        if (
            entry.configuration_payload_sha256 != configuration.configuration_sha256
            or entry.configuration_method_identity_sha256
            != configuration.configuration_method_identity_sha256
            or entry.nonexecution_identity_sha256
            != configuration.nonexecution_identity_sha256
        ):
            raise RunnerContractError(
                f"plan entry configuration identity mismatch {entry.run_id}"
            )
        if entry.preflight_status == "blocked_authority":
            assert entry.preflight_reason is not None
            outcome = AdapterOutcome.blocked_authority(entry.preflight_reason)
        else:
            decision = budget.authorize(
                spec,
                entry.configuration_sha256,
                counts_toward_configuration_limit=(
                    _counts_toward_configuration_limit(entry)
                ),
                budget_scope=_budget_scope(entry),
            )
            if not decision.authorized:
                assert decision.reason is not None
                outcome = AdapterOutcome.budget_exhausted(decision.reason)
            else:
                request = ExecutionRequest.create(
                    spec,
                    prepared.method_input,
                    model_seed=entry.model_seed,
                    configuration=configuration,
                    authority=execution_context,
                    mechanism=entry.mechanism,
                    biological_id=entry.biological_id,
                    technical_view=entry.technical_view,
                    dataset_id=entry.dataset_id,
                    timeout_seconds=decision.timeout_seconds,
                )
                request.validate_integrity()
                outcome = executor(request)
                if not isinstance(outcome, AdapterOutcome):
                    raise RunnerContractError(
                        "adapter executor returned a noncanonical outcome"
                    )
                outcome = enforce_calibration_fold_receipt(request, outcome)
                if (
                    outcome.status != "infrastructure_error"
                    and float(outcome.runtime_seconds) > decision.remaining_seconds
                ):
                    outcome = AdapterOutcome.resource_exceeded(
                        "development_time_budget_exceeded",
                        stdout=outcome.stdout,
                        stderr=outcome.stderr,
                        runtime_seconds=outcome.runtime_seconds,
                        peak_rss_bytes=outcome.peak_rss_bytes,
                        peak_gpu_bytes=outcome.peak_gpu_bytes,
                    )
                budget.record(
                    spec,
                    entry.configuration_sha256,
                    outcome,
                    counts_toward_configuration_limit=(
                        _counts_toward_configuration_limit(entry)
                    ),
                    budget_scope=_budget_scope(entry),
                )
        evaluated = evaluate_adapter_outcome(
            entry,
            prepared,
            outcome,
            dataset_qc_policy_sha256=qc_policy_sha256,
        )
        report = checkpoint_store.append(
            plan,
            report,
            evaluated,
            budget,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
    if report is None:
        report = checkpoint_store.write(
            plan,
            (),
            budget,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
    return report


def execute_fair_comparator_plan(
    plan: DirectCompetitionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    executor: Callable[
        [DirectPlanEntry, PreparedDataset, BudgetDecision],
        DirectEvaluatedAttempt,
    ],
    checkpoint_store: DirectCheckpointStore,
    *,
    authority: RunnerAuthority,
    datasets: Sequence[DatasetBinding],
) -> DirectCheckpointReport:
    """Execute/resume the production direct plan through evaluated attempts."""

    from .fair_comparator_plan import validate_direct_competition_plan

    validate_direct_competition_plan(
        plan,
        registry=registry,
        prepared_datasets=prepared_datasets,
        authority=authority,
        datasets=datasets,
    )
    return _execute_fair_comparator_plan_validated(
        plan,
        registry,
        prepared_datasets,
        executor,
        checkpoint_store,
        authority=authority,
        datasets=datasets,
        structural=False,
    )


def _execute_fair_comparator_plan_structural(
    plan: DirectCompetitionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    executor: Callable[
        [DirectPlanEntry, PreparedDataset, BudgetDecision],
        DirectEvaluatedAttempt,
    ],
    checkpoint_store: DirectCheckpointStore,
) -> DirectCheckpointReport:
    """Execute a private, in-memory structural fixture."""

    from .fair_comparator_plan import _validate_direct_competition_plan_structure

    _validate_direct_competition_plan_structure(
        plan,
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    return _execute_fair_comparator_plan_validated(
        plan,
        registry,
        prepared_datasets,
        executor,
        checkpoint_store,
        authority=None,
        datasets=None,
        structural=True,
    )


def _execute_fair_comparator_plan_validated(
    plan: DirectCompetitionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    executor: Callable[
        [DirectPlanEntry, PreparedDataset, BudgetDecision],
        DirectEvaluatedAttempt,
    ],
    checkpoint_store: DirectCheckpointStore,
    *,
    authority: RunnerAuthority | None,
    datasets: Sequence[DatasetBinding] | None,
    structural: bool,
) -> DirectCheckpointReport:
    """Execute a direct plan whose public or private gate already validated it."""

    from .fair_comparator_checkpoint import (
        DirectCheckpointStore,
        DirectDevelopmentBudget,
        budget_scope,
        configuration_budget_key,
        counts_toward_configuration_limit,
        replay_direct_development_budget,
    )
    from .fair_comparator_execution import DirectEvaluatedAttempt
    from .fair_comparator_plan import DirectCompetitionPlan

    if not isinstance(plan, DirectCompetitionPlan):
        raise TypeError("plan must be a DirectCompetitionPlan")
    if plan.identity_mode != "direct-v1":
        raise RunnerContractError("direct fair-comparator plan mode differs")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(prepared_datasets, Mapping):
        raise TypeError("prepared_datasets must be a mapping")
    if not callable(executor):
        raise TypeError("executor must be callable")
    if not isinstance(checkpoint_store, DirectCheckpointStore):
        raise TypeError("checkpoint_store must be a DirectCheckpointStore")
    completed_records = (
        checkpoint_store._inspect_prefix_structural(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        if structural
        else checkpoint_store.inspect_prefix(
            plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
            authority=authority,
            datasets=datasets,
        )
    )
    require_development_storage_capacity(
        checkpoint_store.path.parent,
        plan,
        prepared_datasets,
        completed_records=completed_records,
    )
    has_checkpoint_state = os.path.lexists(checkpoint_store.path) or os.path.lexists(
        checkpoint_store.intent_path
    )
    report = (
        (
            checkpoint_store._load_structural(
                plan,
                registry=registry,
                prepared_datasets=prepared_datasets,
            )
            if structural
            else checkpoint_store.load(
                plan,
                registry=registry,
                prepared_datasets=prepared_datasets,
                authority=authority,
                datasets=datasets,
            )
        )
        if has_checkpoint_state
        else None
    )
    budget = (
        DirectDevelopmentBudget()
        if report is None
        else replay_direct_development_budget(
            registry,
            plan.entries,
            report.records,
        )
    )
    start = 0 if report is None else len(report.records)
    for entry in plan.entries[start:]:
        try:
            spec = registry.by_id(entry.identity.method.method_id)
        except KeyError as error:
            raise RunnerContractError(
                f"direct plan references unknown method {entry.identity.method.method_id}"
            ) from error
        prepared = prepared_datasets.get(entry.identity.dataset_id)
        if not isinstance(prepared, PreparedDataset):
            raise RunnerContractError(
                f"direct prepared dataset is missing for {entry.identity.dataset_id}"
            )
        if entry.preflight_status == "blocked_authority":
            if entry.preflight_reason is None:
                raise RunnerContractError("direct preflight blocker lacks a reason")
            decision = BudgetDecision(
                authorized=False,
                reason=entry.preflight_reason,
                remaining_seconds=0.0,
                timeout_seconds=0.0,
            )
        else:
            decision = budget.authorize(
                spec,
                configuration_budget_key(entry),
                counts_toward_configuration_limit=(
                    counts_toward_configuration_limit(entry)
                ),
                budget_scope=budget_scope(entry),
            )
        attempt = executor(entry, prepared, decision)
        if not isinstance(attempt, DirectEvaluatedAttempt):
            raise RunnerContractError(
                "direct attempt executor returned a noncanonical value"
            )
        if entry.preflight_status == "blocked_authority":
            if (
                attempt.run.status != "blocked_authority"
                or attempt.run.reason != entry.preflight_reason
            ):
                raise RunnerContractError(
                    "direct preflight attempt differs from its blocker"
                )
        elif not decision.authorized:
            if (
                attempt.run.status != "budget_exhausted"
                or attempt.run.reason != decision.reason
            ):
                raise RunnerContractError(
                    "direct budget attempt differs from its decision"
                )
        elif attempt.run.status in {"blocked_authority", "budget_exhausted"}:
            raise RunnerContractError(
                "authorized direct attempt returned a caller selection blocker"
            )
        if (
            decision.authorized
            and attempt.run.status != "infrastructure_error"
            and float(attempt.run.runtime_seconds) > decision.remaining_seconds
        ):
            raise RunnerContractError("direct attempt exceeded its method budget")
        if attempt.run.status not in COMPARATOR_SELECTION_BLOCKING_STATUSES:
            budget.record(
                spec,
                configuration_budget_key(entry),
                attempt,
                counts_toward_configuration_limit=(
                    counts_toward_configuration_limit(entry)
                ),
                budget_scope=budget_scope(entry),
            )
        if structural:
            report = checkpoint_store._append_structural(
                plan,
                report,
                attempt,
                registry=registry,
                prepared_datasets=prepared_datasets,
            )
        else:
            report = checkpoint_store.append(
                plan,
                report,
                attempt,
                registry=registry,
                prepared_datasets=prepared_datasets,
                authority=authority,
                datasets=datasets,
            )
    if report is None:
        if structural:
            report = checkpoint_store._write_structural(
                plan,
                (),
                registry=registry,
                prepared_datasets=prepared_datasets,
            )
        else:
            report = checkpoint_store.write(
                plan,
                (),
                registry=registry,
                prepared_datasets=prepared_datasets,
                authority=authority,
                datasets=datasets,
            )
    return report


class AdapterExecutor(Protocol):
    """Dependency-injected adapter dispatch boundary."""

    def __call__(self, request: ExecutionRequest) -> AdapterOutcome: ...


def _execution_environment_snapshot() -> str:
    return process_environment_sha256()


def derive_lock_only_environment_ids(registry: MethodRegistry) -> tuple[str, ...]:
    """Derive ready external-reference runtime IDs from one validated registry."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    scopes_by_id: dict[str, str] = {}
    lock_only_ids: list[str] = []
    for spec in registry.methods:
        if not isinstance(spec.id, str) or not _SAFE_ID.fullmatch(spec.id):
            raise RunnerContractError("method registry environment ID is invalid")
        previous_scope = scopes_by_id.get(spec.id)
        if previous_scope is not None:
            if (
                "external_reference_only" in {previous_scope, spec.execution_scope}
                and previous_scope != spec.execution_scope
            ):
                raise RunnerContractError("method registry environment scopes overlap")
            raise RunnerContractError("method registry environment IDs are duplicated")
        scopes_by_id[spec.id] = spec.execution_scope
        if (
            spec.execution_scope == "external_reference_only"
            and spec.environment.status == "ready"
        ):
            lock_only_ids.append(spec.id)
    return tuple(sorted(lock_only_ids))


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentRegistry:
    """Explicit executable paths for every adapter environment available locally."""

    repository_root: Path
    executable_paths: tuple[tuple[str, str | None], ...]
    lock_only_environment_ids: tuple[str, ...]
    registry_sha256: str
    runtime_lock_sha256: str | None
    runtime_lock_path: Path | None
    benchmark_python: Path | None
    r_library_paths: tuple[tuple[str, tuple[str, ...]], ...]
    execution_environment_sha256: str
    python_spawn_search_path: tuple[str, ...]
    runtime_identity_snapshots: tuple[tuple[str, str], ...]
    runtime_closure_paths_sha256s: tuple[tuple[str, str], ...]
    runtime_snapshot: RuntimeEnvironmentSnapshot | None

    @classmethod
    def fixed(
        cls,
        repository_root: Path,
        overrides: Mapping[str, Path] | None = None,
        *,
        runtime_lock_path: Path | None = None,
        benchmark_python: Path | None = None,
        r_library_paths: Mapping[str, Sequence[Path]] | None = None,
        lock_only_environment_ids: Sequence[str] = (),
    ) -> ExecutionEnvironmentRegistry:
        if not isinstance(repository_root, Path):
            raise TypeError("repository_root must be a pathlib.Path")
        repository = repository_root.resolve(strict=True)
        if isinstance(lock_only_environment_ids, (str, bytes)) or not isinstance(
            lock_only_environment_ids, Sequence
        ):
            raise TypeError("lock_only_environment_ids must be a sequence")
        raw_lock_only_ids = tuple(lock_only_environment_ids)
        if any(
            not isinstance(environment_id, str)
            or not _SAFE_ID.fullmatch(environment_id)
            for environment_id in raw_lock_only_ids
        ):
            raise RunnerContractError("lock-only environment ID is invalid")
        if len(set(raw_lock_only_ids)) != len(raw_lock_only_ids):
            raise RunnerContractError("lock-only environment IDs are duplicated")
        normalized_lock_only_ids = tuple(sorted(raw_lock_only_ids))
        if normalized_lock_only_ids and runtime_lock_path is None:
            raise RunnerContractError(
                "lock-only environment IDs require a runtime lock"
            )
        declared: dict[str, Path] = {
            "afmf": repository / "artifacts/envs/afmf-python/bin/python",
            "alra": repository / "artifacts/envs/alra-r/bin/Rscript",
            "biaeimpute": repository / "artifacts/envs/biaeimpute-python/bin/python",
            "dca": repository / "artifacts/envs/dca-legacy-python/bin/python",
            "magic": repository / "artifacts/envs/magic-python/bin/python",
            "saver": repository / "artifacts/envs/saver-r/bin/Rscript",
            "sccr": repository / "artifacts/envs/sccr-python/bin/python",
            "scsdae": repository / "artifacts/envs/scsdae-legacy-python/bin/python",
            "scvi": repository / "artifacts/envs/scvi-py312/bin/python",
            "scziva": repository / "artifacts/envs/scziva-python/bin/python",
        }
        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise TypeError("environment overrides must be a mapping")
            for method_id, path in overrides.items():
                if not isinstance(method_id, str) or not _SAFE_ID.fullmatch(method_id):
                    raise RunnerContractError(
                        "environment override method ID is invalid"
                    )
                if not isinstance(path, Path):
                    raise TypeError("environment override paths must be pathlib.Path")
                declared[method_id] = path
        implemented = (
            "afmf",
            "alra",
            "biaeimpute",
            "dca",
            "magic",
            "saver",
            "sccr",
            "scsdae",
            "scvi",
            "scziva",
        )
        unknown_overrides = set(declared) - set(implemented)
        if unknown_overrides:
            raise RunnerContractError(
                "environment override names a method without a concrete adapter: "
                + ", ".join(sorted(unknown_overrides))
            )
        entries: list[tuple[str, str | None]] = []
        receipt: dict[str, object] = {}

        def display_path(path: Path) -> str:
            absolute = path.absolute()
            try:
                return absolute.relative_to(repository).as_posix()
            except ValueError:
                return str(absolute)

        for method_id in implemented:
            declared_path = declared.get(method_id)
            if declared_path is None or not declared_path.exists():
                entries.append((method_id, None))
                receipt[method_id] = {
                    "status": "unavailable",
                    "declared_path": (
                        None if declared_path is None else display_path(declared_path)
                    ),
                }
                continue
            invocation = declared_path.absolute()
            target = invocation.resolve(strict=True)
            metadata = target.stat()
            if not stat.S_ISREG(metadata.st_mode) or not os.access(invocation, os.X_OK):
                raise RunnerContractError(
                    f"environment executable is not an executable file: {declared_path}"
                )
            entries.append((method_id, str(invocation)))
            receipt[method_id] = {
                "status": "ready",
                "declared_path": display_path(declared_path),
                "executable_sha256": _file_sha256(target),
            }
        runtime_lock_sha256: str | None = None
        runtime_receipt: dict[str, object] | None = None
        runtime_identity_snapshots: tuple[tuple[str, str], ...] = ()
        runtime_closure_paths_sha256s: tuple[tuple[str, str], ...] = ()
        runtime_snapshot: RuntimeEnvironmentSnapshot | None = None
        normalized_r_libraries = tuple(
            (
                environment_id,
                tuple(path.absolute().as_posix() for path in paths),
            )
            for environment_id, paths in sorted(
                ({} if r_library_paths is None else r_library_paths).items()
            )
        )
        execution_environment_sha256 = _execution_environment_snapshot()
        python_spawn_search_path = publication_python_spawn_search_path()
        if runtime_lock_path is not None:
            if not isinstance(runtime_lock_path, Path):
                raise TypeError("runtime_lock_path must be a pathlib.Path")
            if not isinstance(benchmark_python, Path):
                raise TypeError(
                    "benchmark_python must be a pathlib.Path with a runtime lock"
                )
            declarations: dict[str, tuple[Literal["python", "r"], Path]] = {
                "benchmark": ("python", benchmark_python)
            }
            r_methods = {"alra", "saver"}
            for method_id, executable_path in entries:
                if executable_path is None:
                    continue
                declarations[method_id] = (
                    "r" if method_id in r_methods else "python",
                    Path(executable_path),
                )
            try:
                initial_lock = load_runtime_environment_lock(runtime_lock_path)
                libraries = {} if r_library_paths is None else r_library_paths
                identity_cache: dict[
                    tuple[str, str, tuple[str, ...]], RuntimeEnvironmentSnapshot
                ] = {}
                identities: list[tuple[str, str]] = []
                closure_paths: dict[str, str] = {}
                snapshots: list[RuntimeEnvironmentSnapshot] = []
                for environment_id, (kind, executable) in sorted(declarations.items()):
                    selected_libraries = tuple(libraries.get(environment_id, ()))
                    cache_key = (
                        kind,
                        executable.absolute().as_posix(),
                        tuple(
                            path.absolute().as_posix() for path in selected_libraries
                        ),
                    )
                    if cache_key not in identity_cache:
                        identity_cache[cache_key] = runtime_environment_snapshot(
                            kind,
                            executable,
                            r_library_paths=selected_libraries,
                        )
                        snapshots.append(identity_cache[cache_key])
                    identities.append(
                        (environment_id, identity_cache[cache_key].identity_sha256)
                    )
                    closure_paths[environment_id] = identity_cache[
                        cache_key
                    ].closure_paths_sha256
                runtime_identity_snapshots = tuple(identities)
                runtime_closure_paths_sha256s = tuple(sorted(closure_paths.items()))
                runtime_snapshot = merge_runtime_environment_snapshots(
                    snapshots,
                    additional_files=(initial_lock.path,),
                )
                with RuntimeChangeMonitor(runtime_snapshot.watch_specs) as monitor:
                    verify_runtime_environment_snapshot(runtime_snapshot)
                    monitor.assert_unchanged()
                    runtime_lock = load_runtime_environment_lock(initial_lock.path)
                    runtime_receipt = validate_runtime_environment_lock(
                        runtime_lock,
                        declarations,
                        r_library_paths=r_library_paths,
                        expected_closure_paths_sha256s=closure_paths,
                        lock_only_environment_ids=normalized_lock_only_ids,
                    )
                    verify_runtime_environment_snapshot(runtime_snapshot)
                    monitor.assert_unchanged()
                if _execution_environment_snapshot() != execution_environment_sha256:
                    raise RuntimeEnvironmentError(
                        "execution-affecting environment changed during planning"
                    )
            except RuntimeEnvironmentError as error:
                raise RunnerContractError(str(error)) from error
            runtime_lock_sha256 = runtime_lock.file_sha256
        body = {
            "schema": (
                "maskimpute-execution-environment-registry-v2"
                if runtime_receipt is not None
                else "maskimpute-execution-environment-registry-v1"
            ),
            "methods": receipt,
            "runtime_lock": runtime_receipt,
            "execution_environment_sha256": execution_environment_sha256,
            "python_spawn_search_path": python_spawn_search_path,
            "runtime_identity_snapshots": runtime_identity_snapshots,
            "runtime_closure_paths_sha256s": runtime_closure_paths_sha256s,
            "runtime_integrity_snapshot_sha256": (
                None if runtime_snapshot is None else runtime_snapshot.identity_sha256
            ),
        }
        if normalized_lock_only_ids:
            body["lock_only_environment_ids"] = normalized_lock_only_ids
        return cls(
            repository_root=repository,
            executable_paths=tuple(entries),
            lock_only_environment_ids=normalized_lock_only_ids,
            registry_sha256=canonical_sha256(body),
            runtime_lock_sha256=runtime_lock_sha256,
            runtime_lock_path=(
                None if runtime_lock_path is None else runtime_lock.path
            ),
            benchmark_python=(
                None if benchmark_python is None else benchmark_python.absolute()
            ),
            r_library_paths=normalized_r_libraries,
            execution_environment_sha256=execution_environment_sha256,
            python_spawn_search_path=python_spawn_search_path,
            runtime_identity_snapshots=runtime_identity_snapshots,
            runtime_closure_paths_sha256s=runtime_closure_paths_sha256s,
            runtime_snapshot=runtime_snapshot,
        )

    def executable_for(self, method_id: str) -> Path | None:
        for observed_method, path in self.executable_paths:
            if observed_method == method_id:
                return None if path is None else Path(path)
        return None

    def revalidate_control_state_for(self, method_id: str) -> None:
        """Recheck exact libc environment and race-safe runtime-lock bytes."""
        if _execution_environment_snapshot() != self.execution_environment_sha256:
            raise RunnerContractError(
                "execution-affecting environment changed after plan construction"
            )
        if publication_python_spawn_search_path() != self.python_spawn_search_path:
            raise RunnerContractError(
                "Python spawn search path changed after plan construction"
            )
        if self.runtime_snapshot is not None:
            try:
                verify_runtime_environment_control_files(self.runtime_snapshot)
            except RuntimeEnvironmentError as error:
                raise RunnerContractError(str(error)) from error
        if self.runtime_lock_path is None:
            return
        try:
            lock = load_runtime_environment_lock(self.runtime_lock_path)
            if lock.file_sha256 != self.runtime_lock_sha256:
                raise RuntimeEnvironmentError("runtime lock changed after planning")
            if method_id in {"observed", "maskimpute", "capacity-matched-ae"}:
                environment_id = "benchmark"
                executable = self.benchmark_python
            else:
                environment_id = method_id
                executable = self.executable_for(method_id)
            if executable is None:
                return
            lock.by_id(environment_id)
        except RuntimeEnvironmentError as error:
            raise RunnerContractError(str(error)) from error

    def revalidate_for(self, method_id: str) -> None:
        """Recheck control state and all prevalidated path identities."""

        self.revalidate_control_state_for(method_id)
        if self.runtime_lock_path is None:
            return
        if self.runtime_snapshot is None:
            raise RunnerContractError("runtime integrity snapshot is unavailable")
        try:
            verify_runtime_environment_snapshot(self.runtime_snapshot)
        except RuntimeEnvironmentError as error:
            raise RunnerContractError(str(error)) from error

    def full_revalidate(self) -> None:
        """Rebuild all frozen inventories while a replacement monitor is active."""

        self.revalidate_for("observed")
        if self.runtime_lock_path is None:
            return
        if self.benchmark_python is None:
            raise RunnerContractError("benchmark Python runtime is unavailable")
        declarations: dict[str, tuple[Literal["python", "r"], Path]] = {
            "benchmark": ("python", self.benchmark_python)
        }
        r_methods = {"alra", "saver"}
        for method_id, raw_path in self.executable_paths:
            if raw_path is None:
                continue
            declarations[method_id] = (
                "r" if method_id in r_methods else "python",
                Path(raw_path),
            )
        libraries = {
            environment_id: tuple(Path(path) for path in paths)
            for environment_id, paths in self.r_library_paths
        }
        try:
            lock = load_runtime_environment_lock(self.runtime_lock_path)
            if lock.file_sha256 != self.runtime_lock_sha256:
                raise RuntimeEnvironmentError(
                    "runtime lock changed after plan construction"
                )
            validate_runtime_environment_lock(
                lock,
                declarations,
                r_library_paths=libraries,
                expected_closure_paths_sha256s=dict(self.runtime_closure_paths_sha256s),
                lock_only_environment_ids=self.lock_only_environment_ids,
            )
            if self.runtime_snapshot is not None:
                verify_runtime_environment_snapshot(self.runtime_snapshot)
        except RuntimeEnvironmentError as error:
            raise RunnerContractError(str(error)) from error

    def change_monitor(self) -> RuntimeChangeMonitor:
        specs = (
            () if self.runtime_snapshot is None else self.runtime_snapshot.watch_specs
        )
        try:
            return RuntimeChangeMonitor(specs)
        except RuntimeEnvironmentError as error:
            raise RunnerContractError(str(error)) from error


def _score_manifest_entry(
    request: ExecutionRequest,
    score_diagnostics: Mapping[str, object],
    repository: Path,
) -> None:
    if (
        request.count_score_manifest_path is None
        or request.count_score_manifest_sha256 is None
    ):
        raise RunnerContractError("count-score execution lacks a manifest binding")
    path = repository / request.count_score_manifest_path
    if _file_sha256(path) != request.count_score_manifest_sha256:
        raise RunnerContractError("count-score manifest file checksum mismatch")
    payload = _load_strict_json(path, "count-score manifest")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RunnerContractError("count-score manifest entries are unavailable")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("dataset_id") == request.dataset_id
        and entry.get("mechanism") == request.mechanism
        and entry.get("biological_id") == request.biological_id
        and entry.get("technical_view") == request.technical_view
    ]
    if len(matches) != 1:
        raise RunnerContractError("count-score manifest lacks one exact dataset entry")
    entry = matches[0]
    expected = {
        "dataset_sha256": request.method_input.source_dataset_sha256,
        "cell_ids_sha256": _cell_id_sha256(request.method_input.obs_ids),
        "retained_cell_ids_sha256": _cell_id_sha256(request.method_input.obs_ids),
        "input_sha256": score_diagnostics.get("score_input_sha256"),
        "score_sha256": score_diagnostics.get("score_artifact_sha256"),
        "config_sha256": request.count_model_config_sha256,
    }
    for field, value in expected.items():
        if entry.get(field) != value:
            raise RunnerContractError(
                f"count-score manifest entry {field} mismatches adapter diagnostics"
            )


def maskimpute_variant_for_configuration(
    configuration: AuthorizedConfiguration,
) -> str:
    """Map one exact authority config to the tracked ablation implementation."""

    if not isinstance(configuration, AuthorizedConfiguration):
        raise TypeError("configuration must be an AuthorizedConfiguration")
    if configuration.method_id == "capacity-matched-ae":
        return "capacity-matched-ae"
    if configuration.method_id != "maskimpute":
        raise RunnerContractError("configuration is not an in-tree learned method")
    if configuration.kind == "ablation":
        return configuration.configuration_id
    if configuration.kind != "candidate_search":
        raise RunnerContractError("MaskImpute configuration kind is unsupported")
    score_policy = configuration.payload.get("score_policy")
    if score_policy == "direct_cross_fitted_count_score":
        return "direct-score"
    if score_policy in {
        "retained_calibrator",
        "retained_calibrated_count_score",
        "retained_development_calibrator",
    }:
        return "maskimpute-reference"
    raise RunnerContractError(
        "candidate-search score_policy is not a tracked direct or calibrated policy"
    )


def maskimpute_decoder_for_configuration(
    configuration: AuthorizedConfiguration,
) -> tuple[str, object | None]:
    """Resolve an exact v27/v28 decoder without widening development authority."""

    if not isinstance(configuration, AuthorizedConfiguration):
        raise TypeError("configuration must be an AuthorizedConfiguration")
    payload = configuration.payload
    if configuration.kind == "ablation":
        if "method_version" in payload or "decoder" in payload:
            raise RunnerContractError(
                "v28 negative-binomial execution must be a development candidate"
            )
        return "scaled_gaussian", None
    if configuration.kind != "candidate_search":
        raise RunnerContractError("MaskImpute decoder configuration kind is invalid")
    if configuration.method_id != "maskimpute":
        raise RunnerContractError("candidate decoder is not bound to MaskImpute")
    method_version = payload.get("method_version")
    decoder = payload.get("decoder")
    if method_version == "v27":
        expected_fields = {
            "method_version",
            "decoder",
            "encoder_mode",
            "output_policy",
            "score_policy",
            "hyperparameters",
        }
        if set(payload) != expected_fields:
            raise RunnerContractError("v27 candidate decoder payload fields differ")
        if (
            decoder != "scaled_gaussian"
            or payload.get("encoder_mode") != "explicit_mask"
            or payload.get("output_policy") != "selective"
        ):
            raise RunnerContractError("v27 candidate decoder contract differs")
        return "scaled_gaussian", None
    if method_version not in {"v28", "v29"} or decoder != "negative_binomial":
        raise RunnerContractError("candidate version and decoder pair is unsupported")
    expected_fields = {
        "method_version",
        "decoder",
        "encoder_mode",
        "output_policy",
        "score_policy",
        "hyperparameters",
        "decoder_hyperparameters",
    }
    if method_version == "v29":
        expected_fields.add("structure_hyperparameters")
    if set(payload) != expected_fields:
        raise RunnerContractError(
            f"{method_version} candidate decoder payload fields differ"
        )
    if (
        payload.get("encoder_mode") != "explicit_mask"
        or payload.get("output_policy") != "selective"
        or payload.get("score_policy") != "retained_development_calibrator"
        or not configuration.requires_count_score
        or not configuration.requires_calibration
    ):
        raise RunnerContractError("v28 candidate score or output contract differs")
    decoder_payload = payload.get("decoder_hyperparameters")
    if not isinstance(decoder_payload, Mapping):
        raise RunnerContractError("v28 decoder_hyperparameters must be a mapping")
    from maskimpute.nb_model import NegativeBinomialDecoderConfig

    if set(decoder_payload) != set(NegativeBinomialDecoderConfig().to_dict()):
        raise RunnerContractError("v28 decoder_hyperparameters fields differ")
    try:
        decoder_config = NegativeBinomialDecoderConfig(**dict(decoder_payload))
    except (TypeError, ValueError) as error:
        raise RunnerContractError("v28 decoder_hyperparameters are invalid") from error
    return "negative_binomial", decoder_config


def maskimpute_structure_for_configuration(
    configuration: AuthorizedConfiguration,
) -> object | None:
    """Resolve the fixed v29 structure objective, absent from v27/v28/ablations."""

    if not isinstance(configuration, AuthorizedConfiguration):
        raise TypeError("configuration must be an AuthorizedConfiguration")
    if configuration.kind == "ablation":
        return None
    if (
        configuration.method_id != "maskimpute"
        or configuration.kind != "candidate_search"
    ):
        raise RunnerContractError(
            "structure configuration is not a MaskImpute candidate"
        )
    version = configuration.payload.get("method_version")
    if version in {"v27", "v28"}:
        if "structure_hyperparameters" in configuration.payload:
            raise RunnerContractError("v27/v28 cannot carry structure penalties")
        return None
    if version != "v29":
        raise RunnerContractError("structure configuration version is unsupported")
    raw = configuration.payload.get("structure_hyperparameters")
    from maskimpute.structure import StructurePenaltyConfig

    if not isinstance(raw, Mapping) or set(raw) != set(
        StructurePenaltyConfig().to_dict()
    ):
        raise RunnerContractError("v29 structure_hyperparameters fields differ")
    try:
        return StructurePenaltyConfig(**dict(raw))
    except (TypeError, ValueError) as error:
        raise RunnerContractError(
            "v29 structure_hyperparameters are invalid"
        ) from error


@dataclass(frozen=True, slots=True)
class RepositoryAdapterDispatcher:
    """Concrete dispatch for every implemented adapter; unbuilt envs stay visible."""

    repository_root: Path
    environments: ExecutionEnvironmentRegistry
    monitor_runtime_changes: bool = True
    comparator_tuning_authority: ComparatorTuningAuthority | None = dataclass_field(
        default=None,
        repr=False,
    )

    @property
    def supported_method_ids(self) -> tuple[str, ...]:
        return (
            "observed",
            "capacity-matched-ae",
            "maskimpute",
            "alra",
            "magic",
            "dca",
            "scvi",
            "saver",
            "scziva",
            "afmf",
            "biaeimpute",
            "d3impute",
            "sccr",
            "scsdae",
        )

    def __post_init__(self) -> None:
        repository = self.repository_root.resolve(strict=True)
        if repository != self.environments.repository_root:
            raise RunnerContractError(
                "dispatcher repository differs from environment registry"
            )
        if self.comparator_tuning_authority is not None and not isinstance(
            self.comparator_tuning_authority,
            ComparatorTuningAuthority,
        ):
            raise TypeError(
                "comparator_tuning_authority must be a ComparatorTuningAuthority"
            )
        object.__setattr__(self, "repository_root", repository)

    def direct_comparator_adapters(
        self,
    ) -> Mapping[str, Callable[..., AdapterOutcome]]:
        """Return the closed production mapping for direct comparator dispatch."""

        method_ids = COMPARATOR_METHOD_IDS

        def adapter_for(method_id: str) -> Callable[..., AdapterOutcome]:
            def execute(
                spec: MethodSpec,
                method_input: MethodInput,
                *,
                seed: int | None,
                config: ComparatorAdapterConfig,
            ) -> AdapterOutcome:
                return self._execute_direct_comparator(
                    method_id,
                    spec,
                    method_input,
                    seed=seed,
                    config=config,
                )

            return execute

        return MappingProxyType(
            {method_id: adapter_for(method_id) for method_id in method_ids}
        )

    def _execute_direct_comparator(
        self,
        method_id: str,
        spec: MethodSpec,
        method_input: MethodInput,
        *,
        seed: int | None,
        config: ComparatorAdapterConfig,
    ) -> AdapterOutcome:
        """Invoke one repository adapter from the closed direct mapping."""

        if spec.id != method_id:
            raise RunnerContractError("direct repository adapter method differs")
        monitor = (
            self.environments.change_monitor() if self.monitor_runtime_changes else None
        )
        try:
            if self.monitor_runtime_changes:
                self.environments.revalidate_for(method_id)
            else:
                self.environments.revalidate_control_state_for(method_id)
            if monitor is not None:
                monitor.assert_unchanged()
            try:
                executable = self.environments.executable_for(method_id)
                if executable is None:
                    return AdapterOutcome.unavailable(
                        f"environment_executable_unavailable_{method_id}"
                    )
                source = spec.source.cache_path
                if source is None:
                    return AdapterOutcome.unavailable(
                        f"pinned_source_path_unavailable_{method_id}"
                    )
                if isinstance(seed, bool) or not isinstance(seed, int):
                    return AdapterOutcome.failed(f"stochastic_seed_missing_{method_id}")
                source_dir = self.repository_root / source
                if method_id == "alra":
                    from .methods import run_alra_direct

                    execution = run_alra_direct(
                        spec,
                        method_input,
                        source_dir=source_dir,
                        rscript=executable,
                        seed=seed,
                        config=config,
                    )
                elif method_id == "saver":
                    from .methods import run_saver_direct

                    execution = run_saver_direct(
                        spec,
                        method_input,
                        source_dir=source_dir,
                        rscript=executable,
                        seed=seed,
                        library_dir=(
                            self.repository_root / "artifacts/envs/saver-r/library"
                        ),
                        lock_manifest=(
                            self.repository_root / "environments/saver-r.lock.json"
                        ),
                        build_receipt=(
                            self.repository_root
                            / "environments/saver-r.build-receipt.json"
                        ),
                        config=config,
                    )
                else:
                    from .methods import (
                        run_afmf_direct,
                        run_biaeimpute_direct,
                        run_dca_direct,
                        run_magic_direct,
                        run_sccr_direct,
                        run_scsdae_direct,
                        run_scvi_direct,
                        run_scziva_direct,
                    )

                    function = {
                        "afmf": run_afmf_direct,
                        "biaeimpute": run_biaeimpute_direct,
                        "dca": run_dca_direct,
                        "magic": run_magic_direct,
                        "sccr": run_sccr_direct,
                        "scsdae": run_scsdae_direct,
                        "scvi": run_scvi_direct,
                        "scziva": run_scziva_direct,
                    }[method_id]
                    execution = function(
                        spec,
                        method_input,
                        source_dir=source_dir,
                        python_executable=executable,
                        seed=seed,
                        config=config,
                    )
                return AdapterOutcome.completed(
                    execution,
                    runtime_seconds=0,
                    peak_rss_bytes=0,
                    peak_gpu_bytes=0,
                )
            finally:
                if monitor is not None:
                    monitor.assert_unchanged()
                if self.monitor_runtime_changes:
                    self.environments.revalidate_for(method_id)
                else:
                    self.environments.revalidate_control_state_for(method_id)
        except RuntimeEnvironmentError as error:
            return AdapterOutcome.infrastructure_error(str(error))
        finally:
            if monitor is not None:
                monitor.close()

    def _comparator_config(self, request: ExecutionRequest) -> ComparatorAdapterConfig:
        raise RunnerContractError(
            "legacy comparator dispatcher is disabled; use the direct fair-comparator path"
        )

    def _in_tree(self, request: ExecutionRequest) -> AdapterOutcome:
        if (
            request.base_configuration_json is None
            or request.count_model_config_json is None
            or request.retained_calibration_path is None
            or request.retained_calibration_sha256 is None
            or request.model_seed is None
        ):
            return AdapterOutcome.unavailable("in_tree_artifact_authority_pending")
        calibration_path = self.repository_root / request.retained_calibration_path
        if _file_sha256(calibration_path) != request.retained_calibration_sha256:
            return AdapterOutcome.failed("retained_calibration_file_checksum_mismatch")
        trajectory_identity = (
            request.mechanism == "synthetic_trajectory"
            or request.biological_id == "trajectory-draw-01"
        )
        if trajectory_identity:
            if (
                request.mechanism != "synthetic_trajectory"
                or request.biological_id != "trajectory-draw-01"
                or request.technical_view != "deterministic-count-allocation"
                or request.dataset_id != "trajectory-exact-latent-01"
                or request.count_score_manifest_path is None
                or request.count_score_manifest_sha256 is None
            ):
                return AdapterOutcome.failed(
                    "registered_trajectory_identity_outside_authority"
                )
            score_authority_path = (
                self.repository_root / request.count_score_manifest_path
            )
            if (
                _file_sha256(score_authority_path)
                != request.count_score_manifest_sha256
            ):
                return AdapterOutcome.failed(
                    "trajectory_count_score_authority_checksum_mismatch"
                )
            try:
                score_authority = _load_strict_json(
                    score_authority_path,
                    "trajectory count-score authority",
                )
                score_body = {
                    key: value
                    for key, value in score_authority.items()
                    if key != "payload_sha256"
                }
                if (
                    score_authority.get("artifact_type")
                    != "maskimpute_trajectory_count_score_authority"
                    or score_authority.get("scope")
                    != "truth_free_registered_trajectory_inference"
                    or score_authority.get("trajectory_dataset_sha256")
                    != request.method_input.source_dataset_sha256
                    or score_authority.get("trajectory_method_input_sha256")
                    != request.method_input_sha256
                    or score_authority.get("count_model_config_sha256")
                    != request.count_model_config_sha256
                    or score_authority.get("payload_sha256")
                    != canonical_sha256(score_body)
                ):
                    raise RunnerContractError(
                        "trajectory count-score authority differs from request"
                    )
            except (OSError, RunnerContractError, ValueError):
                return AdapterOutcome.failed("trajectory_count_score_authority_invalid")
        from maskimpute import MaskImputeConfig, PreZeroCountModelConfig
        from maskimpute.calibration import load_calibration_artifact
        from maskimpute_benchmark.methods.maskimpute import (
            _run_in_tree,
            run_frozen_final_in_tree,
        )

        base = json.loads(request.base_configuration_json)
        configuration = json.loads(request.configuration_payload_json)
        hyperparameters = configuration.get("hyperparameters", base)
        if not isinstance(hyperparameters, dict):
            return AdapterOutcome.failed("maskimpute_hyperparameters_invalid")
        config = MaskImputeConfig(**hyperparameters, seed=request.model_seed)
        count_config = PreZeroCountModelConfig(
            **json.loads(request.count_model_config_json)
        )
        calibration = load_calibration_artifact(calibration_path)
        score_policy = str(configuration.get("score_policy", "")).casefold()
        requires_calibration = "calibrat" in score_policy or "retained" in score_policy
        authorized_configuration = AuthorizedConfiguration.create(
            method_id=request.method_spec.id,
            configuration_id=request.configuration_id,
            kind=request.configuration_kind,
            payload=configuration,
            requires_count_score=request.count_score_manifest_sha256 is not None,
            requires_calibration=requires_calibration,
            configuration_sha256=request.configuration_sha256,
        )
        variant_id = maskimpute_variant_for_configuration(authorized_configuration)
        decoder, decoder_config = maskimpute_decoder_for_configuration(
            authorized_configuration
        )
        structure_config = maskimpute_structure_for_configuration(
            authorized_configuration
        )
        if request.calibration_usage == "retained_all_development":
            execution = run_frozen_final_in_tree(
                request.method_spec,
                request.method_input,
                variant_id=variant_id,
                calibration_artifact=calibration,
                seed=request.model_seed,
                config=config,
                count_model_config=count_config,
                device="cuda",
                mechanism=request.mechanism,
                biological_id=request.biological_id,
                decoder=decoder,
                decoder_config=decoder_config,
                structure_config=structure_config,
            )
        else:
            execution = _run_in_tree(
                request.method_spec,
                request.method_input,
                variant_id=variant_id,
                calibration_artifact=calibration,
                seed=request.model_seed,
                config=config,
                count_model_config=count_config,
                device="cuda",
                development_mechanism=request.mechanism,
                development_biological_id=request.biological_id,
                decoder=decoder,
                decoder_config=decoder_config,
                structure_config=structure_config,
            )
        diagnostics = execution.ablation_result.diagnostics
        score_diagnostics = diagnostics.get("score")
        if (
            request.count_score_manifest_sha256 is not None
            and request.calibration_usage == "development_holdout"
        ):
            if not isinstance(score_diagnostics, Mapping):
                raise RunnerContractError(
                    "MaskImpute score diagnostics are unavailable"
                )
            _score_manifest_entry(request, score_diagnostics, self.repository_root)
        receipt: CalibrationFoldReceipt | None = None
        if request.calibration_context is not None:
            if not isinstance(score_diagnostics, Mapping):
                raise RunnerContractError(
                    "calibration score diagnostics are unavailable"
                )
            fold = score_diagnostics.get("calibration_fold_receipt")
            if not isinstance(fold, Mapping):
                raise RunnerContractError(
                    "LODO calibration diagnostics are unavailable"
                )
            receipt = CalibrationFoldReceipt(
                calibration_artifact_sha256=request.retained_calibration_sha256,
                calibration_context_sha256=request.calibration_context.sha256,
                mechanism=request.mechanism,
                biological_id=request.biological_id,
                training_manifest_sha256s=tuple(
                    sorted(fold["training_manifest_sha256s"])
                ),
                held_out_manifest_sha256s=tuple(
                    sorted(fold["held_out_manifest_sha256s"])
                ),
                fold_calibrator_sha256=fold["calibrator_sha256"],
            )
        return AdapterOutcome.completed(
            execution,
            runtime_seconds=0,
            peak_rss_bytes=0,
            peak_gpu_bytes=0,
            calibration_fold_receipt=receipt,
        )

    def __call__(
        self,
        request: ExecutionRequest | FinalComparatorExecutionRequest,
    ) -> AdapterOutcome:
        method_id = request.method_spec.id
        monitor = (
            self.environments.change_monitor() if self.monitor_runtime_changes else None
        )
        try:
            if self.monitor_runtime_changes:
                self.environments.revalidate_for(method_id)
            else:
                self.environments.revalidate_control_state_for(method_id)
            if monitor is not None:
                try:
                    monitor.assert_unchanged()
                except RuntimeEnvironmentError as error:
                    raise RunnerContractError(str(error)) from error
            try:
                return self._execute_validated(request)
            finally:
                if monitor is not None:
                    try:
                        monitor.assert_unchanged()
                    except RuntimeEnvironmentError as error:
                        raise RunnerContractError(str(error)) from error
                if self.monitor_runtime_changes:
                    self.environments.revalidate_for(method_id)
                else:
                    self.environments.revalidate_control_state_for(method_id)
        finally:
            if monitor is not None:
                monitor.close()

    def _execute_validated(
        self,
        request: ExecutionRequest | FinalComparatorExecutionRequest,
    ) -> AdapterOutcome:
        request.validate_integrity()
        method_id = request.method_spec.id
        if isinstance(request, FinalComparatorExecutionRequest):
            configuration = request.configuration.configuration.decode()
            outcome = self._execute_direct_comparator(
                method_id,
                request.method_spec,
                request.method_input,
                seed=request.model_seed,
                config=configuration,
            )
            request.validate_integrity()
            if encode_comparator_configuration(configuration) != dict(
                request.configuration.configuration.payload
            ):
                raise RunnerContractError(
                    "final comparator effective configuration changed"
                )
            return outcome
        try:
            if method_id == "observed":
                from .methods import run_observed

                execution = run_observed(request.method_spec, request.method_input)
            elif method_id in {"maskimpute", "capacity-matched-ae"}:
                return self._in_tree(request)
            elif method_id == "d3impute":
                return AdapterOutcome.unavailable(
                    _declared_failure_reason(
                        "external_reference_input_not_prepared", method_id
                    )
                )
            elif method_id not in self.supported_method_ids:
                return AdapterOutcome.unavailable(
                    _declared_failure_reason("adapter_not_implemented", method_id)
                )
            else:
                config = self._comparator_config(request)
                try:
                    executable = self.environments.executable_for(method_id)
                    if executable is None:
                        return AdapterOutcome.unavailable(
                            _declared_failure_reason(
                                "environment_executable_unavailable", method_id
                            )
                        )
                    source = request.method_spec.source.cache_path
                    if source is None:
                        return AdapterOutcome.unavailable(
                            _declared_failure_reason(
                                "pinned_source_path_unavailable", method_id
                            )
                        )
                    source_dir = self.repository_root / source
                    seed = request.model_seed
                    if seed is None:
                        return AdapterOutcome.failed(
                            _declared_failure_reason(
                                "stochastic_seed_missing", method_id
                            )
                        )
                    if method_id == "alra":
                        from .methods import run_alra

                        execution = run_alra(
                            request.method_spec,
                            request.method_input,
                            source_dir=source_dir,
                            rscript=executable,
                            seed=seed,
                            config=config,
                        )
                    elif method_id == "saver":
                        from .methods import run_saver

                        execution = run_saver(
                            request.method_spec,
                            request.method_input,
                            source_dir=source_dir,
                            rscript=executable,
                            seed=seed,
                            library_dir=(
                                self.repository_root / "artifacts/envs/saver-r/library"
                            ),
                            lock_manifest=(
                                self.repository_root / "environments/saver-r.lock.json"
                            ),
                            build_receipt=(
                                self.repository_root
                                / "environments/saver-r.build-receipt.json"
                            ),
                            config=config,
                        )
                    else:
                        from .methods import (
                            run_afmf,
                            run_biaeimpute,
                            run_dca,
                            run_magic,
                            run_sccr,
                            run_scsdae,
                            run_scvi,
                            run_scziva,
                        )

                        functions = {
                            "afmf": run_afmf,
                            "biaeimpute": run_biaeimpute,
                            "dca": run_dca,
                            "magic": run_magic,
                            "sccr": run_sccr,
                            "scsdae": run_scsdae,
                            "scvi": run_scvi,
                            "scziva": run_scziva,
                        }
                        execution = functions[method_id](
                            request.method_spec,
                            request.method_input,
                            source_dir=source_dir,
                            python_executable=executable,
                            seed=seed,
                            config=config,
                        )
                finally:
                    encoded = encode_comparator_configuration(config)
                    if (
                        _canonical_bytes(encoded)
                        != request.configuration_payload_json.encode("utf-8")
                        or canonical_sha256(encoded)
                        != request.configuration_payload_sha256
                    ):
                        raise RunnerContractError(
                            "comparator payload changed during adapter attempt"
                        )
            return AdapterOutcome.completed(
                execution,
                runtime_seconds=0,
                peak_rss_bytes=0,
                peak_gpu_bytes=0,
            )
        except Exception as error:
            from .methods import AdapterUnavailableError

            if isinstance(error, AdapterUnavailableError):
                return AdapterOutcome.unavailable(
                    _adapter_failure_reason(error, method_id, unavailable=True),
                    stdout=error.stdout,
                    stderr=error.stderr,
                )
            return AdapterOutcome.failed(
                _adapter_failure_reason(error, method_id, unavailable=False),
                stderr=str(error).encode("utf-8", errors="replace"),
            )


@dataclass(frozen=True, slots=True)
class DirectRepositoryComparatorExecutor:
    """Picklable child executor accepting only the direct request type."""

    dispatcher: RepositoryAdapterDispatcher
    authority: ComparatorTuningAuthority

    def __call__(self, request: DirectExecutionRequest) -> AdapterOutcome:
        from .fair_comparator_execution import (
            DirectExecutionRequest,
            dispatch_direct_request,
        )

        if not isinstance(request, DirectExecutionRequest):
            raise TypeError("request must be a DirectExecutionRequest")
        return dispatch_direct_request(
            request,
            self.authority,
            self.dispatcher.direct_comparator_adapters(),
        )


def execute_fair_comparator_request(
    request: DirectExecutionRequest,
    prepared: PreparedDataset,
    authority: ComparatorTuningAuthority,
    dispatcher: RepositoryAdapterDispatcher,
) -> DirectEvaluatedAttempt:
    """Route one direct request through the closed repository adapter mapping."""

    if not isinstance(dispatcher, RepositoryAdapterDispatcher):
        raise TypeError("dispatcher must be a RepositoryAdapterDispatcher")
    from .fair_comparator_execution import (
        evaluate_direct_outcome,
        validate_direct_request,
    )

    validate_direct_request(request, prepared, authority)
    child_dispatcher = replace(
        dispatcher,
        environments=replace(dispatcher.environments, runtime_snapshot=None),
        monitor_runtime_changes=False,
    )
    snapshot = dispatcher.environments.runtime_snapshot
    sampler = (
        LinuxProcessTreeResourceSampler()
        if snapshot is None
        else LinuxProcessTreeResourceSampler(
            None if snapshot.nvidia_smi_path is None else Path(snapshot.nvidia_smi_path)
        )
    )
    outcome = execute_direct_adapter_in_spawned_process(
        request,
        DirectRepositoryComparatorExecutor(child_dispatcher, authority),
        resource_sampler=sampler,
        expected_spawn_executable=dispatcher.environments.benchmark_python,
        spawn_search_path=dispatcher.environments.python_spawn_search_path,
    )
    return evaluate_direct_outcome(request, prepared, authority, outcome)


@dataclass(frozen=True, slots=True)
class SpawnedRepositoryExecutor:
    """Publication executor using spawn plus independent parent telemetry."""

    dispatcher: RepositoryAdapterDispatcher
    _runtime_monitor: RuntimeChangeMonitor = dataclass_field(
        init=False, repr=False, compare=False
    )
    _child_dispatcher: RepositoryAdapterDispatcher = dataclass_field(
        init=False, repr=False, compare=False
    )
    _resource_sampler: ResourceSampler = dataclass_field(
        init=False, repr=False, compare=False
    )
    _closed: bool = dataclass_field(
        init=False, repr=False, compare=False, default=False
    )

    def __post_init__(self) -> None:
        monitor = self.dispatcher.environments.change_monitor()
        try:
            self.dispatcher.environments.full_revalidate()
            monitor.assert_unchanged()
            snapshot = self.dispatcher.environments.runtime_snapshot
            resource_sampler = (
                LinuxProcessTreeResourceSampler()
                if snapshot is None
                else LinuxProcessTreeResourceSampler(
                    None
                    if snapshot.nvidia_smi_path is None
                    else Path(snapshot.nvidia_smi_path)
                )
            )
            child_dispatcher = replace(
                self.dispatcher,
                environments=replace(
                    self.dispatcher.environments,
                    runtime_snapshot=None,
                ),
                monitor_runtime_changes=False,
            )
        except RuntimeEnvironmentError as error:
            monitor.close()
            raise RunnerContractError(str(error)) from error
        except BaseException:
            monitor.close()
            raise
        object.__setattr__(self, "_runtime_monitor", monitor)
        object.__setattr__(self, "_resource_sampler", resource_sampler)
        object.__setattr__(self, "_child_dispatcher", child_dispatcher)

    def __call__(
        self,
        request: ExecutionRequest | FinalComparatorExecutionRequest,
    ) -> AdapterOutcome:
        if self._closed:
            raise RunnerContractError("spawned repository executor is closed")
        method_id = request.method_spec.id
        self.dispatcher.environments.revalidate_control_state_for(method_id)
        try:
            self._runtime_monitor.assert_unchanged()
        except RuntimeEnvironmentError as error:
            raise RunnerContractError(str(error)) from error
        try:
            if isinstance(request, FinalComparatorExecutionRequest):
                return execute_final_comparator_adapter_in_spawned_process(
                    request,
                    self._child_dispatcher,
                    resource_sampler=self._resource_sampler,
                    expected_spawn_executable=(
                        self.dispatcher.environments.benchmark_python
                    ),
                    spawn_search_path=(
                        self.dispatcher.environments.python_spawn_search_path
                    ),
                )
            return execute_adapter_in_spawned_process(
                request,
                self._child_dispatcher,
                resource_sampler=self._resource_sampler,
                expected_spawn_executable=self.dispatcher.environments.benchmark_python,
                spawn_search_path=self.dispatcher.environments.python_spawn_search_path,
            )
        finally:
            try:
                self._runtime_monitor.assert_unchanged()
            except RuntimeEnvironmentError as error:
                raise RunnerContractError(str(error)) from error
            self.dispatcher.environments.revalidate_control_state_for(method_id)

    def close(self) -> None:
        """Release the long-lived runtime-change monitor exactly once."""

        if self._closed:
            return
        self._runtime_monitor.close()
        object.__setattr__(self, "_closed", True)

    def __enter__(self) -> SpawnedRepositoryExecutor:
        if self._closed:
            raise RunnerContractError("spawned repository executor is closed")
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def load_prepared_development_panel(
    authority: RunnerAuthority,
) -> tuple[tuple[DatasetBinding, ...], Mapping[str, PreparedDataset]]:
    """Byte-revalidate and pair-union-QC the fixed real 16-dataset panel."""

    import anndata as ad

    from .datasets import validate_dataset_status

    if not isinstance(authority, RunnerAuthority):
        raise TypeError("authority must be a RunnerAuthority")
    repository = Path(__file__).resolve().parents[1]
    status_path = repository / "artifacts/study/development/results/dataset_status.json"
    try:
        status = validate_dataset_status(status_path, repo=repository)
        bindings = validate_development_manifest_payload(status)
    except Exception as error:
        raise RunnerContractError(
            "development dataset status failed byte-level revalidation"
        ) from error
    prepared: dict[str, PreparedDataset] = {}
    result_root = repository / "artifacts/study/development/results"
    for first_binding, second_binding in zip(
        bindings[::2], bindings[1::2], strict=True
    ):
        try:
            first_dataset = ad.read_h5ad(result_root / first_binding.output_path)
            second_dataset = ad.read_h5ad(result_root / second_binding.output_path)
            first, second = prepare_dataset_pair_for_execution(
                first_dataset,
                second_dataset,
                first_binding,
                second_binding,
                authority.dataset_qc_policy,
            )
        except Exception as error:
            raise RunnerContractError(
                f"development pair preparation failed for {first_binding.mechanism}/"
                f"{first_binding.biological_id}"
            ) from error
        prepared[first.binding.dataset_id] = first
        prepared[second.binding.dataset_id] = second
    if tuple(prepared) != tuple(binding.dataset_id for binding in bindings):
        raise RunnerContractError(
            "prepared development panel order or cardinality drifted"
        )
    if (
        authority.count_score_manifest_status == "ready"
        or authority.retained_calibration_status == "ready"
    ):
        if not (
            authority.count_score_manifest_status == "ready"
            and authority.retained_calibration_status == "ready"
            and authority.count_score_manifest_sha256 is not None
            and authority.retained_calibration_sha256 is not None
        ):
            raise RunnerContractError(
                "score and calibration authority must become ready atomically"
            )
        score_path = repository / authority.count_score_manifest_path
        calibration_path = repository / authority.retained_calibration_path
        if (
            not score_path.is_file()
            or not calibration_path.is_file()
            or _file_sha256(score_path) != authority.count_score_manifest_sha256
            or _file_sha256(calibration_path) != authority.retained_calibration_sha256
        ):
            raise RunnerContractError(
                "ready score/calibration file checksum binding failed"
            )
        from maskimpute import PreZeroCountModelConfig
        from .development_scores import prepare_validated_development_scores

        count_config = PreZeroCountModelConfig(
            **dict(_thaw_frozen_json(authority.count_model_config))
        )
        validation = prepare_validated_development_scores(
            repository,
            prepared_datasets=tuple(prepared.values()),
            dataset_manifest_sha256=bindings[0].manifest_sha256,
            count_model_config=count_config,
            count_model_config_sha256=authority.count_model_config_sha256,
            dataset_qc_policy_sha256=authority.dataset_qc_policy_sha256,
        )
        if (
            validation.get("status") != "reused"
            or validation.get("count_score_manifest_file_sha256")
            != authority.count_score_manifest_sha256
            or validation.get("calibration_file_sha256")
            != authority.retained_calibration_sha256
        ):
            raise RunnerContractError(
                "ready score/calibration semantic revalidation failed"
            )
    return bindings, MappingProxyType(prepared)


def run_development_competition(
    output_dir: Path,
    *,
    environment_overrides: Mapping[str, Path] | None = None,
) -> DirectCheckpointReport:
    """Run/resume the fixed tracked development competition with no design overrides."""

    authority = load_runner_authority()
    smoke_receipt, smoke_receipt_bytes = _load_required_comparator_smoke_evidence(
        authority
    )
    datasets, prepared_datasets = load_prepared_development_panel(authority)
    repository = Path(__file__).resolve().parents[1]
    registry = load_method_registry(repository / "study/methods.json")
    return _run_fair_comparator_base_with_authority(
        output_dir,
        authority,
        environment_overrides=environment_overrides,
        _comparator_smoke_receipt=smoke_receipt,
        _comparator_smoke_receipt_bytes=smoke_receipt_bytes,
        _datasets=datasets,
        _prepared_datasets=prepared_datasets,
        _registry=registry,
    )


def _load_required_comparator_smoke_evidence(
    authority: RunnerAuthority,
) -> tuple[Mapping[str, object], bytes]:
    """Load complete fixed smoke evidence before any production write boundary."""

    if not isinstance(authority, RunnerAuthority):
        raise TypeError("authority must be a RunnerAuthority")
    from .comparator_tuning import (
        ComparatorTuningError,
        _load_comparator_smoke_receipt_evidence,
    )

    repository = Path(__file__).resolve().parents[1]
    registry = load_method_registry(repository / "study/methods.json")
    try:
        return _load_comparator_smoke_receipt_evidence(
            repository,
            authority.comparator_tuning,
            registry,
        )
    except (ComparatorTuningError, OSError, TypeError, ValueError) as error:
        raise RunnerContractError(
            "required comparator smoke receipt is absent or invalid"
        ) from error


def _run_fair_comparator_base_with_authority(
    output_dir: Path,
    authority: RunnerAuthority,
    *,
    environment_overrides: Mapping[str, Path] | None,
    _comparator_smoke_receipt: Mapping[str, object],
    _comparator_smoke_receipt_bytes: bytes,
    _datasets: Sequence[DatasetBinding],
    _prepared_datasets: Mapping[str, PreparedDataset],
    _registry: MethodRegistry,
    _direct_executor: Callable[
        [DirectPlanEntry, PreparedDataset, BudgetDecision],
        DirectEvaluatedAttempt,
    ]
    | None = None,
) -> DirectCheckpointReport:
    """Bind smoke evidence through the direct production checkpoint boundary."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(authority, RunnerAuthority):
        raise TypeError("authority must be a RunnerAuthority")
    if authority.plan_scope not in {"base_full_panel", "revision_candidate_only"}:
        raise RunnerContractError("fair-comparator authority scope differs")
    if environment_overrides is not None and not isinstance(
        environment_overrides, Mapping
    ):
        raise TypeError("environment_overrides must be a mapping or None")
    if (
        not isinstance(_comparator_smoke_receipt, Mapping)
        or type(_comparator_smoke_receipt_bytes) is not bytes
    ):
        raise TypeError("comparator smoke receipt evidence has invalid types")
    bindings = tuple(_datasets)
    if not bindings or not all(
        isinstance(binding, DatasetBinding) for binding in bindings
    ):
        raise TypeError("_datasets must contain DatasetBinding values")
    if not isinstance(_prepared_datasets, Mapping) or not all(
        isinstance(value, PreparedDataset) for value in _prepared_datasets.values()
    ):
        raise TypeError("_prepared_datasets must map to PreparedDataset values")
    if not isinstance(_registry, MethodRegistry):
        raise TypeError("_registry must be a MethodRegistry")
    ordered_prepared = tuple(
        _prepared_datasets[binding.dataset_id] for binding in bindings
    )
    plan = build_fair_comparator_plan(
        _registry,
        bindings,
        authority,
        ordered_prepared,
        _comparator_smoke_receipt=_comparator_smoke_receipt,
        _comparator_smoke_receipt_bytes=_comparator_smoke_receipt_bytes,
    )
    from .fair_comparator_checkpoint import DirectCheckpointStore

    store = DirectCheckpointStore(output_dir / "checkpoint.json")
    if authority.plan_scope == "revision_candidate_only":
        if authority.base_comparator_selection is None:
            raise RunnerContractError(
                "activated revision authority lacks base comparator selection"
            )
        if _direct_executor is not None:
            return execute_fair_comparator_plan(
                plan,
                _registry,
                _prepared_datasets,
                _direct_executor,
                store,
                authority=authority,
                datasets=bindings,
            )
        repository = Path(__file__).resolve().parents[1]
        from maskimpute import PreZeroCountModelConfig
        from maskimpute.calibration import load_calibration_artifact

        from .direct_values import direct_json_value

        count_model_payload = direct_json_value(
            authority.count_model_config,
            payload=True,
        )
        if type(count_model_payload) is not dict:
            raise RunnerContractError("direct revision count model is not an object")
        revision_adapter = DirectRevisionMaskImputeAdapter(
            calibration_artifact=load_calibration_artifact(
                repository / authority.retained_calibration_path
            ),
            count_model_config=PreZeroCountModelConfig(**count_model_payload),
        )
        revision_executor = RevisionMaskImputeExecutor(
            authority=authority,
            direct_adapter=revision_adapter,
            authorized_configuration=plan.configurations[0],
            input_descriptors={value.dataset_id: value for value in plan.inputs},
            checkpoint_directory=output_dir,
            registry=_registry,
        )
        return execute_fair_comparator_plan(
            plan,
            _registry,
            _prepared_datasets,
            revision_executor,
            store,
            authority=authority,
            datasets=bindings,
        )
    store.inspect_prefix(
        plan,
        registry=_registry,
        prepared_datasets=_prepared_datasets,
        authority=authority,
        datasets=bindings,
    )
    raise RunnerContractError(
        "direct fair-comparator execution awaits production adapter composition"
    )


def run_v28_revision_competition(
    *,
    environment_overrides: Mapping[str, Path] | None = None,
) -> DirectCheckpointReport:
    """Run/resume v28 only at its fixed path after the base selection trigger."""

    repository = Path(__file__).resolve().parents[1]
    authority = load_activated_v28_revision_authority()
    smoke_receipt, smoke_receipt_bytes = _load_required_comparator_smoke_evidence(
        authority
    )
    datasets, prepared_datasets = load_prepared_development_panel(authority)
    registry = load_method_registry(repository / "study/methods.json")
    return _run_fair_comparator_base_with_authority(
        repository / "artifacts/study/development/competition-v28-revision",
        authority,
        environment_overrides=environment_overrides,
        _comparator_smoke_receipt=smoke_receipt,
        _comparator_smoke_receipt_bytes=smoke_receipt_bytes,
        _datasets=datasets,
        _prepared_datasets=prepared_datasets,
        _registry=registry,
    )


def load_activated_v28_revision_authority(
    repository: Path | None = None,
) -> RunnerAuthority:
    """Return v28 authority bound to the exact recomputed base selection trigger."""

    activation = _validate_v28_activation(repository)
    return _bind_v28_activation(
        load_v28_revision_authority(),
        activation,
    )


def load_activated_v29_revision_authority(
    repository: Path | None = None,
) -> RunnerAuthority:
    """Bind the v29 plan to the independently recomputed combined v28 report."""

    from .revisions import validate_revision_activation

    root = Path(__file__).resolve().parents[1] if repository is None else repository
    try:
        activation = validate_revision_activation(root, "v29")
    except Exception as error:
        raise RunnerContractError(f"v29 activation failed: {error}") from error
    authority = load_v29_revision_authority()
    return replace(
        authority,
        authority_sha256=canonical_sha256(
            {
                "schema": "maskimpute-v29-activated-runner-authority-v1",
                "revision_authority_sha256": authority.authority_sha256,
                "selection_input_sha256": activation.selection_input_file_sha256,
                "selection_result_sha256": activation.selection_result_sha256,
                "selection_report_sha256": activation.selection_report_file_sha256,
            }
        ),
        base_comparator_selection=activation.base_comparator_selection,
    )


def run_v29_revision_competition(
    *,
    environment_overrides: Mapping[str, Path] | None = None,
) -> DirectCheckpointReport:
    """Run/resume v29 only at its fixed path after the combined v28 trigger."""

    repository = Path(__file__).resolve().parents[1]
    authority = load_activated_v29_revision_authority()
    smoke_receipt, smoke_receipt_bytes = _load_required_comparator_smoke_evidence(
        authority
    )
    datasets, prepared_datasets = load_prepared_development_panel(authority)
    registry = load_method_registry(repository / "study/methods.json")
    return _run_fair_comparator_base_with_authority(
        repository / "artifacts/study/development/competition-v29-revision",
        authority,
        environment_overrides=environment_overrides,
        _comparator_smoke_receipt=smoke_receipt,
        _comparator_smoke_receipt_bytes=smoke_receipt_bytes,
        _datasets=datasets,
        _prepared_datasets=prepared_datasets,
        _registry=registry,
    )


def _run_competition_with_authority(
    output_dir: Path,
    authority: RunnerAuthority,
    *,
    environment_overrides: Mapping[str, Path] | None,
) -> CheckpointReport:
    """Execute one fully derived authority through the common runner."""

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if not isinstance(authority, RunnerAuthority):
        raise TypeError("authority must be a RunnerAuthority")
    repository = Path(__file__).resolve().parents[1]
    bindings, prepared = load_prepared_development_panel(authority)
    from .methods import load_method_registry

    registry = load_method_registry(repository / "study/methods.json")
    environments = ExecutionEnvironmentRegistry.fixed(
        repository,
        environment_overrides,
        runtime_lock_path=_DEVELOPMENT_RUNTIME_LOCK_PATH,
        benchmark_python=Path(sys.executable),
        r_library_paths={"saver": (repository / "artifacts/envs/saver-r/library",)},
        lock_only_environment_ids=derive_lock_only_environment_ids(registry),
    )
    if environments.runtime_lock_sha256 is None:
        raise RunnerContractError("development runtime lock checksum is absent")
    plan = build_competition_plan(
        registry,
        bindings,
        authority,
        execution_environment_sha256=environments.registry_sha256,
        runtime_lock_sha256=environments.runtime_lock_sha256,
    )
    dispatcher = RepositoryAdapterDispatcher(repository, environments)
    return execute_competition_plan(
        plan,
        registry,
        prepared,
        SpawnedRepositoryExecutor(dispatcher),
        CheckpointStore(output_dir),
    )


def _adapter_process_target(
    connection: Any,
    request: object,
    executor: Callable[[object], object],
    outcome_type: type,
) -> None:
    try:
        os.chdir(publication_runtime_working_directory())
        outcome = executor(request)
        if not isinstance(outcome, outcome_type):
            raise TypeError("adapter executor returned a noncanonical outcome")
        connection.send(("outcome", outcome))
    except BaseException as error:
        connection.send(
            (
                "error",
                type(error).__name__,
                str(error),
            )
        )
    finally:
        connection.close()


def _execute_measured_spawn(
    request: object,
    executor: Callable[[object], object],
    *,
    poll_interval_seconds: float = 0.05,
    resource_sampler: ResourceSampler | None = None,
    require_gpu_measurement: bool = False,
    expected_spawn_executable: Path | None = None,
    spawn_search_path: tuple[str, ...] | None = None,
    outcome_type: type = AdapterOutcome,
) -> object:
    """Execute one already-validated request with parent-owned telemetry."""

    if not callable(executor):
        raise TypeError("executor must be callable")
    if outcome_type not in {AdapterOutcome, DirectRevisionMaskImputeOutcome}:
        raise TypeError("outcome_type is not supported by the measured executor")
    from multiprocessing import spawn as multiprocessing_spawn

    raw_spawn_executable = multiprocessing_spawn.get_executable()
    spawn_executable = Path(os.fsdecode(raw_spawn_executable)).absolute()
    expected_executable = (
        Path(sys.executable).absolute()
        if expected_spawn_executable is None
        else expected_spawn_executable.absolute()
    )
    if spawn_executable != expected_executable:
        raise RunnerContractError(
            "multiprocessing spawn executable differs lexically from benchmark Python"
        )
    try:
        if spawn_executable.resolve(strict=True) != Path(sys.executable).resolve(
            strict=True
        ):
            raise RunnerContractError(
                "multiprocessing spawn executable differs from benchmark Python"
            )
    except OSError as error:
        raise RunnerContractError(
            "multiprocessing spawn executable is unavailable"
        ) from error
    incompatible_flags = (
        "debug",
        "inspect",
        "interactive",
        "optimize",
        "dont_write_bytecode",
        "no_user_site",
        "no_site",
        "ignore_environment",
        "verbose",
        "bytes_warning",
        "quiet",
        "isolated",
        "dev_mode",
        "utf8_mode",
        "warn_default_encoding",
        "safe_path",
    )
    if (
        any(getattr(sys.flags, name) for name in incompatible_flags)
        or sys._xoptions
        or sys.warnoptions
    ):
        raise RunnerContractError(
            "nondefault Python flags are unsupported for publication spawning"
        )
    selected_spawn_search_path = (
        publication_python_spawn_search_path()
        if spawn_search_path is None
        else spawn_search_path
    )
    if (
        not isinstance(selected_spawn_search_path, tuple)
        or not selected_spawn_search_path
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in selected_spawn_search_path
        )
    ):
        raise TypeError("spawn_search_path must contain absolute path strings")
    if (
        isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, (int, float))
        or not math.isfinite(poll_interval_seconds)
        or poll_interval_seconds <= 0
    ):
        raise ValueError("poll_interval_seconds must be positive and finite")
    if type(require_gpu_measurement) is not bool:
        raise TypeError("require_gpu_measurement must be boolean")
    sampler = (
        LinuxProcessTreeResourceSampler()
        if resource_sampler is None
        else resource_sampler
    )
    if not hasattr(sampler, "sample"):
        raise TypeError("resource_sampler must implement sample")
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_adapter_process_target,
        args=(child_connection, request, executor, outcome_type),
        daemon=False,
    )
    started = time.monotonic()
    original_directory = os.open(
        ".",
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    original_search_path = list(sys.path)
    try:
        os.chdir(publication_runtime_working_directory())
        sys.path[:] = selected_spawn_search_path
        process.start()
    finally:
        sys.path[:] = original_search_path
        os.fchdir(original_directory)
        os.close(original_directory)
    child_connection.close()
    if process.pid is None:  # pragma: no cover - multiprocessing invariant
        raise RunnerContractError("spawned adapter process has no process ID")
    message: tuple[object, ...] | None = None
    deadline = started + request.timeout_seconds
    peak_rss: int | None = None
    gpu_measurement_required = (
        request.method_spec.resources.gpu_required or require_gpu_measurement
    )
    peak_gpu: int | None = None if gpu_measurement_required else 0
    rss_provenance = "rss_measurement_unavailable"
    gpu_provenance = (
        "gpu_measurement_unavailable"
        if gpu_measurement_required
        else "not_applicable_cpu_only_method"
    )
    required_telemetry_gap = False
    next_resource_sample = started

    def sample_resources() -> str | None:
        nonlocal peak_rss, peak_gpu, rss_provenance, gpu_provenance
        nonlocal required_telemetry_gap
        try:
            sample = sampler.sample(
                process.pid,
                gpu_required=gpu_measurement_required,
            )
        except Exception:
            required_telemetry_gap = True
            return None
        if not isinstance(sample, ResourceSample):
            required_telemetry_gap = True
            return None
        rss_provenance = sample.rss_provenance
        gpu_provenance = sample.gpu_provenance
        if (
            sample.peak_rss_bytes is None
            or sample.rss_provenance != "linux_proc_process_tree_rss"
            or (
                gpu_measurement_required
                and (
                    sample.peak_gpu_bytes is None
                    or sample.gpu_provenance != "nvidia_smi_process_tree_used_memory"
                )
            )
        ):
            required_telemetry_gap = True
        if sample.peak_rss_bytes is not None:
            peak_rss = max(peak_rss or 0, sample.peak_rss_bytes)
        if sample.peak_gpu_bytes is not None:
            peak_gpu = max(peak_gpu or 0, sample.peak_gpu_bytes)
        if peak_rss is not None and peak_rss > request.max_rss_bytes:
            return "peak_rss_exceeded"
        if peak_gpu is not None and peak_gpu > request.max_gpu_bytes:
            return "peak_gpu_exceeded"
        return None

    def telemetry_is_unavailable() -> bool:
        return (
            required_telemetry_gap
            or peak_rss is None
            or (gpu_measurement_required and peak_gpu is None)
        )

    live_resource_reason = sample_resources()
    try:
        while time.monotonic() < deadline:
            if live_resource_reason is not None and process.is_alive():
                break
            if parent_connection.poll(float(poll_interval_seconds)):
                message = parent_connection.recv()
                break
            now = time.monotonic()
            if now >= next_resource_sample:
                sampled_reason = sample_resources()
                if sampled_reason is not None and process.is_alive():
                    live_resource_reason = sampled_reason
                next_resource_sample = now + (
                    1.0 if request.method_spec.resources.gpu_required else 0.05
                )
            if not process.is_alive():
                break
        if live_resource_reason is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            elapsed = max(0.0, time.monotonic() - started)
            if telemetry_is_unavailable():
                return outcome_type.infrastructure_error(
                    "resource_telemetry_unavailable",
                    runtime_seconds=elapsed,
                    peak_rss_bytes=peak_rss or 0,
                    peak_gpu_bytes=peak_gpu or 0,
                    rss_measurement=rss_provenance,
                    gpu_measurement=gpu_provenance,
                )
            assert peak_gpu is not None
            return outcome_type.resource_exceeded(
                live_resource_reason,
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        sample_resources()
        if message is None and parent_connection.poll(0):
            message = parent_connection.recv()
        if message is None and process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
            elapsed = max(0.0, time.monotonic() - started)
            if telemetry_is_unavailable():
                return outcome_type.infrastructure_error(
                    "resource_telemetry_unavailable",
                    runtime_seconds=elapsed,
                    peak_rss_bytes=peak_rss or 0,
                    peak_gpu_bytes=peak_gpu or 0,
                    rss_measurement=rss_provenance,
                    gpu_measurement=gpu_provenance,
                )
            return outcome_type.timeout(
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        process.join(timeout=2)
        elapsed = max(0.0, time.monotonic() - started)
        if telemetry_is_unavailable():
            return outcome_type.infrastructure_error(
                "resource_telemetry_unavailable",
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss or 0,
                peak_gpu_bytes=peak_gpu or 0,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        assert peak_gpu is not None
        if message is None:
            return outcome_type.infrastructure_error(
                f"adapter_process_exit_{process.exitcode}",
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        if message[0] == "error":
            return outcome_type.failed(
                f"executor_exception:{message[1]}",
                stderr=str(message[2]).encode("utf-8", errors="replace"),
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        if message[0] != "outcome" or len(message) != 2:
            return outcome_type.infrastructure_error(
                "malformed_adapter_process_message",
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        outcome = message[1]
        if not isinstance(outcome, outcome_type):
            return outcome_type.infrastructure_error(
                "noncanonical_adapter_process_outcome",
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        outcome = replace(
            outcome,
            runtime_seconds=elapsed,
            peak_rss_bytes=peak_rss,
            peak_gpu_bytes=peak_gpu,
            rss_measurement=rss_provenance,
            gpu_measurement=gpu_provenance,
        )
        if outcome.peak_rss_bytes > request.max_rss_bytes:
            return outcome_type.resource_exceeded(
                "peak_rss_exceeded",
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                runtime_seconds=outcome.runtime_seconds,
                peak_rss_bytes=outcome.peak_rss_bytes,
                peak_gpu_bytes=outcome.peak_gpu_bytes,
                rss_measurement=outcome.rss_measurement,
                gpu_measurement=outcome.gpu_measurement,
            )
        if outcome.peak_gpu_bytes > request.max_gpu_bytes:
            return outcome_type.resource_exceeded(
                "peak_gpu_exceeded",
                stdout=outcome.stdout,
                stderr=outcome.stderr,
                runtime_seconds=outcome.runtime_seconds,
                peak_rss_bytes=outcome.peak_rss_bytes,
                peak_gpu_bytes=outcome.peak_gpu_bytes,
                rss_measurement=outcome.rss_measurement,
                gpu_measurement=outcome.gpu_measurement,
            )
        return outcome
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


def execute_adapter_in_spawned_process(
    request: ExecutionRequest,
    executor: Callable[[ExecutionRequest], AdapterOutcome],
    **options: object,
) -> AdapterOutcome:
    """Execute a legacy non-comparator request in a measured spawned process."""

    if not isinstance(request, ExecutionRequest):
        raise TypeError("request must be an ExecutionRequest")
    request.validate_integrity()
    outcome = _execute_measured_spawn(request, executor, **options)
    if not isinstance(outcome, AdapterOutcome):  # pragma: no cover - fixed type
        raise AssertionError("legacy measured executor returned the wrong type")
    return outcome


def execute_direct_adapter_in_spawned_process(
    request: DirectExecutionRequest,
    executor: Callable[[DirectExecutionRequest], AdapterOutcome],
    **options: object,
) -> AdapterOutcome:
    """Execute only a direct comparator request with parent-owned telemetry."""

    from .fair_comparator_execution import DirectExecutionRequest

    if not isinstance(request, DirectExecutionRequest):
        raise TypeError("request must be a DirectExecutionRequest")
    outcome = _execute_measured_spawn(request, executor, **options)
    if not isinstance(outcome, AdapterOutcome):  # pragma: no cover - fixed type
        raise AssertionError("direct measured executor returned the wrong type")
    return outcome


def execute_final_comparator_adapter_in_spawned_process(
    request: FinalComparatorExecutionRequest,
    executor: Callable[[FinalComparatorExecutionRequest], AdapterOutcome],
    **options: object,
) -> AdapterOutcome:
    """Execute one frozen final comparator request with parent telemetry."""

    if not isinstance(request, FinalComparatorExecutionRequest):
        raise TypeError("request must be a FinalComparatorExecutionRequest")
    request.validate_integrity()
    outcome = _execute_measured_spawn(request, executor, **options)
    if not isinstance(outcome, AdapterOutcome):  # pragma: no cover - fixed type
        raise AssertionError("final comparator executor returned the wrong type")
    return outcome


def execute_direct_revision_adapter_in_spawned_process(
    request: DirectRevisionExecutionRequest,
    executor: Callable[
        [DirectRevisionExecutionRequest], DirectRevisionMaskImputeOutcome
    ],
    **options: object,
) -> DirectRevisionMaskImputeOutcome:
    """Execute one revision request with direct parent-owned telemetry."""

    if not isinstance(request, DirectRevisionExecutionRequest):
        raise TypeError("request must be a DirectRevisionExecutionRequest")
    outcome = _execute_measured_spawn(
        request,
        executor,
        outcome_type=DirectRevisionMaskImputeOutcome,
        **options,
    )
    if not isinstance(
        outcome,
        DirectRevisionMaskImputeOutcome,
    ):  # pragma: no cover - fixed type
        raise AssertionError("revision measured executor returned the wrong type")
    return outcome


__all__ = [
    "AdapterExecutor",
    "AdapterOutcome",
    "AuthorizedConfiguration",
    "BudgetDecision",
    "CalibrationFoldContext",
    "CalibrationFoldReceipt",
    "CheckpointReport",
    "CheckpointStore",
    "CompetitionPlan",
    "DEVELOPMENT_MODEL_SEEDS",
    "DatasetQCAudit",
    "DatasetQCPolicy",
    "DatasetBinding",
    "DevelopmentBudget",
    "DevelopmentStoragePreflight",
    "DirectRepositoryComparatorExecutor",
    "DirectRevisionExecutionRequest",
    "DirectRevisionMaskImputeAdapter",
    "DirectRevisionMaskImputeOutcome",
    "EvaluatedAttempt",
    "ExecutionEnvironmentRegistry",
    "ExecutionRequest",
    "FinalComparatorExecutionRequest",
    "LongFormMetric",
    "LinuxProcessTreeResourceSampler",
    "MAX_CPU_BUDGET_SECONDS",
    "MAX_DEVELOPMENT_CONFIGURATIONS",
    "MAX_GPU_BUDGET_SECONDS",
    "ResourceSample",
    "ResourceSampler",
    "RepositoryAdapterDispatcher",
    "RunPlanEntry",
    "RunnerAuthority",
    "RunnerContractError",
    "SpawnedRepositoryExecutor",
    "build_competition_plan",
    "build_fair_comparator_plan",
    "derive_lock_only_environment_ids",
    "derive_authorized_configurations",
    "development_storage_preflight",
    "decode_direct_bound_comparator_value",
    "execute_adapter_in_spawned_process",
    "execute_direct_adapter_in_spawned_process",
    "execute_final_comparator_adapter_in_spawned_process",
    "execute_direct_revision_adapter_in_spawned_process",
    "execute_competition_plan",
    "execute_fair_comparator_plan",
    "execute_fair_comparator_request",
    "evaluate_adapter_outcome",
    "enforce_calibration_fold_receipt",
    "method_input_sha256",
    "maskimpute_decoder_for_configuration",
    "maskimpute_structure_for_configuration",
    "maskimpute_variant_for_configuration",
    "load_prepared_development_panel",
    "load_runner_authority",
    "load_v28_revision_authority",
    "load_activated_v28_revision_authority",
    "load_activated_v29_revision_authority",
    "load_v29_revision_authority",
    "prepare_dataset_for_execution",
    "prepare_dataset_pair_for_execution",
    "require_development_storage_capacity",
    "run_development_competition",
    "run_v28_revision_competition",
    "run_v29_revision_competition",
    "validate_development_manifest_payload",
]
