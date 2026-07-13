"""Evaluator-owned, truth-isolated development competition runner.

This module provides a data-boundary isolation contract: method adapters receive
only immutable :class:`~maskimpute_benchmark.methods.MethodInput` snapshots.  A
spawned process does not inherit the evaluator's AnnData object.  This is not an
operating-system security sandbox; adapters can still access paths available to
their operating-system account.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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
from typing import Any, Literal, Protocol

import numpy as np

from .methods import AdapterExecution, MethodInput, MethodSpec
from .methods.registry import MethodRegistry
from .protocol import canonical_sha256
from .runtime_environments import (
    RuntimeEnvironmentError,
    load_runtime_environment_lock,
    validate_runtime_environment_lock,
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
_IMPLEMENTATION_SOURCE_DIRECTORIES = ("maskimpute", "maskimpute_benchmark")
_IMPLEMENTATION_SOURCE_FILES = ("scripts/run_development_competition.py",)
_DEVELOPMENT_RUNTIME_LOCK_PATH = (
    Path(__file__).resolve().parents[1]
    / "environments/development-runtime.lock.json"
)
_TRACKED_V28_REVISION_PATH = (
    Path(__file__).resolve().parents[1] / "study/v28_revision.json"
)
_TRACKED_V28_REVISION_SHA256 = (
    "04fbd61a7ab83e3f1b4b1c8a8d4d5b40b8c6bee39a7f57b50c28b68f12c36705"
)
_V28_SELECTION_INPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/study/development/evaluation/development_selection_input.json"
)
_V28_SELECTION_REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/study/development/evaluation/development_selection_report.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
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
                    "implementation source directory is unavailable: "
                    f"{parent_relative}"
                ) from error
            if stat.S_ISLNK(parent_metadata.st_mode):
                raise RunnerContractError(
                    "implementation source directory must not be a symlink: "
                    f"{parent_relative}"
                )
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise RunnerContractError(
                    "implementation source path must be a directory: "
                    f"{parent_relative}"
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
    ) -> AuthorizedConfiguration:
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
        if not isinstance(self.method_id, str) or not _SAFE_ID.fullmatch(
            self.method_id
        ):
            raise RunnerContractError("configuration method_id must be safe")
        if not isinstance(self.configuration_id, str) or not _SAFE_ID.fullmatch(
            self.configuration_id
        ):
            raise RunnerContractError("configuration_id must be safe")
        if self.kind not in {"registry", "candidate_search", "ablation"}:
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

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.payload_json))

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "configuration_id": self.configuration_id,
            "kind": self.kind,
            "configuration_sha256": self.configuration_sha256,
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
        selection = load_publication_execution_authority()
    except Exception as error:
        raise RunnerContractError(
            f"publication execution authority is unavailable: {error}"
        ) from error
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
        "schema": "maskimpute-development-runner-authority-v1",
        "file_sha256": file_hashes,
        "configurations": [value.to_dict() for value in configurations],
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
        search_count = sum(
            value.method_id == "maskimpute" and value.kind == "candidate_search"
            for value in self.configurations
        )
        if not 1 <= search_count <= MAX_DEVELOPMENT_CONFIGURATIONS:
            raise RunnerContractError(
                "MaskImpute candidate-search configuration count must lie in [1, 20]"
            )
        capacity = [
            value
            for value in self.configurations
            if value.method_id == "capacity-matched-ae"
        ]
        if len(capacity) != 1:
            raise RunnerContractError(
                "capacity-matched-ae must have exactly one authority configuration"
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
        raise RunnerContractError("tracked v28 revision authority is unavailable") from error
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
        raise RunnerContractError("v28 revision authority is not canonical tracked JSON")
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
        or revision["reason_code"]
        != "prespecified_decoder_only_revision_of_v27_c03"
    ):
        raise RunnerContractError("v28 revision activation contract differs")
    parent_id = revision["parent_configuration_id"]
    parent_sha256 = revision["parent_configuration_sha256"]
    try:
        parent = next(
            value
            for value in base.configurations
            if value.method_id == "maskimpute"
            and value.configuration_id == parent_id
        )
    except StopIteration as error:
        raise RunnerContractError("v28 parent is absent from frozen v27 authority") from error
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
    capacity = tuple(
        value
        for value in base.configurations
        if value.method_id == "capacity-matched-ae"
    )
    if len(capacity) != 1:
        raise RunnerContractError("v28 authority lacks one fixed capacity control")
    authority_body = {
        "schema": "maskimpute-v28-revision-runner-authority-v1",
        "base_runner_authority_sha256": base.authority_sha256,
        "revision_file_sha256": revision_sha256,
        "parent_configuration_id": parent.configuration_id,
        "parent_configuration_sha256": parent.configuration_sha256,
        "configuration": candidate.to_dict(),
        "capacity_control": capacity[0].to_dict(),
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
        configurations=(candidate, capacity[0]),
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


def _validate_v28_activation() -> tuple[str, str]:
    """Recompute the fixed selection report and require its v28 trigger."""

    selection_input, input_sha256 = _secure_canonical_artifact(
        _V28_SELECTION_INPUT_PATH,
        "v28 activation selection input",
    )
    selection_report, report_sha256 = _secure_canonical_artifact(
        _V28_SELECTION_REPORT_PATH,
        "v28 activation selection report",
    )
    from .selection import select_development_candidate

    try:
        recomputed = select_development_candidate(selection_input).to_dict()
    except Exception as error:
        raise RunnerContractError(
            "v28 activation selection input failed semantic revalidation"
        ) from error
    if selection_report != recomputed:
        raise RunnerContractError(
            "v28 activation report differs from recomputed selection"
        )
    if (
        selection_report.get("trigger") != "v28"
        or selection_report.get("selected_configuration") is not None
    ):
        raise RunnerContractError("v28 activation report does not authorize v28")
    assessments = selection_report.get("assessments")
    if not isinstance(assessments, list) or not any(
        isinstance(value, dict)
        and value.get("configuration_id") == "v27-c03-calibrated-r1-g1"
        and value.get("version") == "v27"
        for value in assessments
    ):
        raise RunnerContractError(
            "v28 activation report lacks the prespecified assessed parent"
        )
    bindings = selection_report.get("authority_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise RunnerContractError("v28 activation report lacks authority bindings")
    return input_sha256, report_sha256


def _bind_v28_activation(
    authority: RunnerAuthority,
    input_sha256: str,
    report_sha256: str,
) -> RunnerAuthority:
    _require_sha256(input_sha256, "v28 selection input checksum")
    _require_sha256(report_sha256, "v28 selection report checksum")
    return replace(
        authority,
        authority_sha256=canonical_sha256(
            {
                "schema": "maskimpute-v28-activated-runner-authority-v1",
                "revision_authority_sha256": authority.authority_sha256,
                "selection_input_sha256": input_sha256,
                "selection_report_sha256": report_sha256,
            }
        ),
    )


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
    configuration_sha256: str
    preflight_status: Literal["planned", "blocked_authority"]
    preflight_reason: str | None
    configuration_kind: str = "registry"
    requires_count_score: bool = False
    requires_calibration: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompetitionPlan:
    """Immutable full development denominator and all authority hashes."""

    schema_version: int
    input_hashes: Mapping[str, str]
    entries: tuple[RunPlanEntry, ...]
    plan_sha256: str
    configurations: tuple[AuthorizedConfiguration, ...] = ()
    execution_context: ExecutionAuthorityContext | None = None


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


def build_competition_plan(
    registry: MethodRegistry,
    datasets: Sequence[DatasetBinding],
    authority: RunnerAuthority,
    *,
    execution_environment_sha256: str | None = None,
) -> CompetitionPlan:
    """Build the exhaustive method x dataset x seed denominator."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(authority, RunnerAuthority):
        raise TypeError("authority must be a RunnerAuthority")
    if execution_environment_sha256 is None:
        execution_environment_sha256 = "0" * 64
    _require_sha256(
        execution_environment_sha256, "execution environment registry checksum"
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
    plan_configurations: list[AuthorizedConfiguration] = []
    for spec in registry.methods:
        if spec.id in {"maskimpute", "capacity-matched-ae"}:
            configurations = authority_by_method.get(spec.id, ())
            if not configurations:
                raise RunnerContractError(
                    f"tracked authority has no configuration for {spec.id}"
                )
        else:
            configurations = (AuthorizedConfiguration.registry_default(spec),)
        plan_configurations.extend(configurations)
    ordinal = 0
    for binding in dataset_values:
        for spec in registry.methods:
            configurations = tuple(
                value for value in plan_configurations if value.method_id == spec.id
            )
            seeds: tuple[int | None, ...] = (
                DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
            )
            for configuration in configurations:
                if (
                    configuration.requires_calibration
                    and not authority.maskimpute_ready
                ):
                    blocked_reason = "count_score_or_calibration_authority_pending"
                elif (
                    configuration.requires_count_score
                    and authority.count_score_manifest_status != "ready"
                ):
                    blocked_reason = "count_score_authority_pending"
                else:
                    blocked_reason = None
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
                        )
                    )
    plan_body = {
        "schema_version": 1,
        "input_hashes": input_hashes,
        "entries": [entry.to_dict() for entry in entries],
        "configurations": [value.to_dict() for value in plan_configurations],
        "execution_context": asdict(authority.execution_context),
        "budgets": {
            "maximum_configurations": MAX_DEVELOPMENT_CONFIGURATIONS,
            "gpu_seconds": MAX_GPU_BUDGET_SECONDS,
            "cpu_seconds": MAX_CPU_BUDGET_SECONDS,
            "failures_consume_budget_except": "infrastructure_error",
        },
    }
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

    binding: DatasetBinding
    audit: DatasetQCAudit
    method_input: MethodInput
    evaluator_dataset: Any


def _prepare_dataset_with_exclusions(
    dataset: Any,
    binding: DatasetBinding,
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
    if not isinstance(binding, DatasetBinding):
        raise TypeError("binding must be a DatasetBinding")
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


class LinuxProcessTreeResourceSampler:
    """Sample aggregate Linux RSS and nvidia-smi compute-process memory."""

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
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
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
    configuration_payload_json: str
    execution_authority_sha256: str
    base_configuration_json: str | None
    base_configuration_sha256: str | None
    count_model_config_json: str | None
    count_model_config_sha256: str | None
    count_score_manifest_path: str | None
    count_score_manifest_sha256: str | None
    retained_calibration_path: str | None
    retained_calibration_sha256: str | None
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
    ) -> ExecutionRequest:
        if not isinstance(method_spec, MethodSpec):
            raise TypeError("method_spec must be a MethodSpec")
        if not isinstance(method_input, MethodInput):
            raise TypeError("method_input must be a MethodInput")
        if not isinstance(configuration, AuthorizedConfiguration):
            raise TypeError("configuration must be an AuthorizedConfiguration")
        if configuration.method_id != method_spec.id:
            raise RunnerContractError("configuration method does not match MethodSpec")
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
                if configuration.requires_calibration and mechanism == "symsim"
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
            "configuration_payload": dict(configuration.payload),
            "execution_authority_sha256": context.authority_sha256,
            "base_configuration_sha256": base_config_sha,
            "count_model_config_sha256": count_config_sha,
            "count_score_manifest_path": score_path,
            "count_score_manifest_sha256": score_sha,
            "retained_calibration_path": calibration_path,
            "retained_calibration_sha256": calibration_sha,
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
            configuration_payload_json=configuration.payload_json,
            execution_authority_sha256=context.authority_sha256,
            base_configuration_json=base_config_json,
            base_configuration_sha256=base_config_sha,
            count_model_config_json=count_config_json,
            count_model_config_sha256=count_config_sha,
            count_score_manifest_path=score_path,
            count_score_manifest_sha256=score_sha,
            retained_calibration_path=calibration_path,
            retained_calibration_sha256=calibration_sha,
            calibration_context=calibration_context,
            timeout_seconds=float(timeout),
            max_rss_bytes=int(values["max_rss_bytes"]),
            max_gpu_bytes=int(values["max_gpu_bytes"]),
            request_sha256=_execution_request_binding(values),
        )

    def validate_integrity(self) -> None:
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
            "configuration_payload": configuration,
            "execution_authority_sha256": self.execution_authority_sha256,
            "base_configuration_sha256": self.base_configuration_sha256,
            "count_model_config_sha256": self.count_model_config_sha256,
            "count_score_manifest_path": self.count_score_manifest_path,
            "count_score_manifest_sha256": self.count_score_manifest_sha256,
            "retained_calibration_path": self.retained_calibration_path,
            "retained_calibration_sha256": self.retained_calibration_sha256,
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
class AdapterOutcome:
    """Measured adapter result before evaluator-side conversion and metrics."""

    status: str
    execution: AdapterExecution | None
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
                not isinstance(self.execution, AdapterExecution)
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
        execution: AdapterExecution,
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
    configuration_sha256: str
    configuration_kind: str
    requires_count_score: bool
    requires_calibration: bool
    method_input_sha256: str
    dataset_qc_policy_sha256: str
    excluded_cell_count: int
    excluded_cell_ids_sha256: str
    retained_cell_count: int
    retained_cell_ids_sha256: str
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
    configuration_sha256: str
    metric: str
    value: float | None
    n: int
    status: str
    reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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
            metric=name,
            value=None,
            n=0,
            status=status,
            reason=reason,
        )
        for name in _RECONSTRUCTION_METRIC_NAMES
    )


def _evaluator_conversion_failure_reason(error: Exception) -> str:
    from .methods import AdapterUnavailableError

    if isinstance(error, AdapterUnavailableError) and re.fullmatch(
        r"[a-z][a-z0-9_]*", error.reason_code
    ):
        return f"evaluator_conversion:{error.reason_code}"
    error_name = next(
        name
        for error_type, name in (
            (TypeError, "TypeError"),
            (ValueError, "ValueError"),
            (OverflowError, "OverflowError"),
            (AdapterUnavailableError, "AdapterUnavailableError"),
        )
        if isinstance(error, error_type)
    )
    digest = hashlib.sha256()
    digest.update(b"maskimpute-evaluator-conversion-detail-v1\0")
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
    return f"evaluator_conversion:{error_name}:detail_sha256={digest.hexdigest()}"


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
                converted, dtype=np.float64, copy=True, order="C", subok=False
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
        method_input_sha256=method_input_sha256(prepared.method_input),
        dataset_qc_policy_sha256=dataset_qc_policy_sha256,
        excluded_cell_count=prepared.audit.excluded_cell_count,
        excluded_cell_ids_sha256=prepared.audit.excluded_cell_ids_sha256,
        retained_cell_count=prepared.audit.retained_cell_count,
        retained_cell_ids_sha256=prepared.audit.retained_cell_ids_sha256,
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
    )
    return EvaluatedAttempt(
        run=run,
        metrics=metric_rows,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        native_output=native_output,
        native_output_scale=native_output_scale,
        evaluator_output=evaluator_output,
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
            "selection_complete",
            "selection_blockers",
            "records",
            "budget",
            "checkpoint_sha256",
        }
    )

    def __init__(self, output_dir: Path) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        self.output_dir = output_dir.absolute()
        self.checkpoint_path = self.output_dir / "checkpoint.json"

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
        run = asdict(attempt.run)
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
        return {
            "run": run,
            "metrics": [metric.to_dict() for metric in attempt.metrics],
        }

    def write(
        self,
        plan: CompetitionPlan,
        records: Sequence[Mapping[str, object]],
        budget: DevelopmentBudget,
    ) -> CheckpointReport:
        record_values = tuple(dict(record) for record in records)
        status = "completed" if len(record_values) == len(plan.entries) else "running"
        body: dict[str, object] = {
            "schema_version": 1,
            "plan_sha256": plan.plan_sha256,
            "input_hashes": dict(plan.input_hashes),
            "planned_run_count": len(plan.entries),
            "status": status,
            "evaluation_scope": "reconstruction_only",
            "selection_complete": False,
            "selection_blockers": list(SELECTION_COMPLETENESS_BLOCKERS),
            "records": list(record_values),
            "budget": budget.to_dict(),
        }
        body["checkpoint_sha256"] = canonical_sha256(body)
        self._publish_checkpoint(body)
        return self.load(plan)

    def append(
        self,
        plan: CompetitionPlan,
        report: CheckpointReport | None,
        attempt: EvaluatedAttempt,
        budget: DevelopmentBudget,
    ) -> CheckpointReport:
        records = [] if report is None else list(report.records)
        records.append(self._stored_attempt(attempt))
        return self.write(plan, records, budget)

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
    ) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != {"run", "metrics"}:
            raise RunnerContractError("checkpoint record has wrong schema")
        run = value["run"]
        metrics = value["metrics"]
        if not isinstance(run, dict) or not isinstance(metrics, list):
            raise RunnerContractError("checkpoint record payload is malformed")
        for name, expected in (
            ("run_id", entry.run_id),
            ("method_id", entry.method_id),
            ("dataset_id", entry.dataset_id),
            ("source_dataset_sha256", entry.source_dataset_sha256),
            ("model_seed", entry.model_seed),
            ("configuration_id", entry.configuration_id),
            ("configuration_sha256", entry.configuration_sha256),
            ("configuration_kind", entry.configuration_kind),
            ("requires_count_score", entry.requires_count_score),
            ("requires_calibration", entry.requires_calibration),
        ):
            if run.get(name) != expected:
                raise RunnerContractError(
                    f"checkpoint run {name} mismatches plan prefix"
                )
        if run.get("status") not in _OUTCOME_STATUSES:
            raise RunnerContractError("checkpoint run status is invalid")
        _require_nonnegative_number(run.get("runtime_seconds"), "checkpoint runtime")
        calibration_artifact = run.get("calibration_artifact_sha256")
        if calibration_artifact is None:
            if (
                run.get("calibration_context_sha256") is not None
                or run.get("calibration_fold_calibrator_sha256") is not None
                or run.get("calibration_training_manifest_sha256s") != []
                or run.get("calibration_held_out_manifest_sha256s") != []
            ):
                raise RunnerContractError("checkpoint calibration receipt is partial")
            if entry.requires_calibration and run.get("status") == "completed":
                raise RunnerContractError(
                    "calibrated completed run lacks its LODO receipt"
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
        if len(metrics) != len(_RECONSTRUCTION_METRIC_NAMES):
            raise RunnerContractError("checkpoint metric denominator is incomplete")
        for metric, expected_name in zip(
            metrics, _RECONSTRUCTION_METRIC_NAMES, strict=True
        ):
            if not isinstance(metric, dict) or metric.get("metric") != expected_name:
                raise RunnerContractError(
                    "checkpoint metrics are not canonically ordered"
                )
            for name, expected in (
                ("method", entry.method_id),
                ("dataset_id", entry.dataset_id),
                ("model_seed", entry.model_seed),
                ("configuration_sha256", entry.configuration_sha256),
            ):
                if metric.get(name) != expected:
                    raise RunnerContractError(
                        f"checkpoint metric {name} mismatches plan"
                    )
        return value

    def load(self, plan: CompetitionPlan) -> CheckpointReport:
        if not isinstance(plan, CompetitionPlan):
            raise TypeError("plan must be a CompetitionPlan")
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
            self._validate_stored_record(value, entry)
            for value, entry in zip(records_value, plan.entries, strict=False)
        )
        expected_status = (
            "completed" if len(records) == len(plan.entries) else "running"
        )
        if payload.get("status") != expected_status:
            raise RunnerContractError("checkpoint status contradicts its plan prefix")
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
        return CheckpointReport(
            schema_version=1,
            plan_sha256=plan.plan_sha256,
            input_hashes=dict(plan.input_hashes),
            planned_run_count=len(plan.entries),
            status=expected_status,
            evaluation_scope="reconstruction_only",
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
    _require_sha256(plan.plan_sha256, "competition plan checksum")
    report = (
        checkpoint_store.load(plan)
        if checkpoint_store.checkpoint_path.exists()
        else None
    )
    budget = DevelopmentBudget()
    if report is not None:
        for stored, entry in zip(report.records, plan.entries, strict=False):
            run = stored["run"]
            assert isinstance(run, Mapping)
            spec = registry.by_id(str(run["method_id"]))
            budget.restore(
                spec,
                str(run["configuration_sha256"]),
                str(run["status"]),
                run["runtime_seconds"],
                counts_toward_configuration_limit=(
                    entry.configuration_kind == "candidate_search"
                ),
                budget_scope=(
                    f"{spec.id}:{entry.configuration_kind}"
                    if spec.id == "maskimpute"
                    else spec.id
                ),
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
        matching_configurations = tuple(
            value
            for value in plan.configurations
            if value.method_id == entry.method_id
            and value.configuration_id == entry.configuration_id
            and value.configuration_sha256 == entry.configuration_sha256
        )
        if not matching_configurations and entry.configuration_id == "registry-default":
            matching_configurations = (AuthorizedConfiguration.registry_default(spec),)
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
        if entry.preflight_status == "blocked_authority":
            assert entry.preflight_reason is not None
            outcome = AdapterOutcome.blocked_authority(entry.preflight_reason)
        else:
            decision = budget.authorize(
                spec,
                entry.configuration_sha256,
                counts_toward_configuration_limit=(
                    entry.configuration_kind == "candidate_search"
                ),
                budget_scope=(
                    f"{spec.id}:{entry.configuration_kind}"
                    if spec.id == "maskimpute"
                    else spec.id
                ),
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
                        entry.configuration_kind == "candidate_search"
                    ),
                    budget_scope=(
                        f"{spec.id}:{entry.configuration_kind}"
                        if spec.id == "maskimpute"
                        else spec.id
                    ),
                )
        evaluated = evaluate_adapter_outcome(
            entry,
            prepared,
            outcome,
            dataset_qc_policy_sha256=qc_policy_sha256,
        )
        report = checkpoint_store.append(plan, report, evaluated, budget)
    if report is None:
        report = checkpoint_store.write(plan, (), budget)
    return report


class AdapterExecutor(Protocol):
    """Dependency-injected adapter dispatch boundary."""

    def __call__(self, request: ExecutionRequest) -> AdapterOutcome: ...


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentRegistry:
    """Explicit executable paths for every adapter environment available locally."""

    repository_root: Path
    executable_paths: tuple[tuple[str, str | None], ...]
    registry_sha256: str
    runtime_lock_sha256: str | None

    @classmethod
    def fixed(
        cls,
        repository_root: Path,
        overrides: Mapping[str, Path] | None = None,
        *,
        runtime_lock_path: Path | None = None,
        benchmark_python: Path | None = None,
        r_library_paths: Mapping[str, Sequence[Path]] | None = None,
    ) -> ExecutionEnvironmentRegistry:
        if not isinstance(repository_root, Path):
            raise TypeError("repository_root must be a pathlib.Path")
        repository = repository_root.resolve(strict=True)
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
                runtime_lock = load_runtime_environment_lock(runtime_lock_path)
                runtime_receipt = validate_runtime_environment_lock(
                    runtime_lock,
                    declarations,
                    r_library_paths=r_library_paths,
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
        }
        return cls(
            repository_root=repository,
            executable_paths=tuple(entries),
            registry_sha256=canonical_sha256(body),
            runtime_lock_sha256=runtime_lock_sha256,
        )

    def executable_for(self, method_id: str) -> Path | None:
        for observed_method, path in self.executable_paths:
            if observed_method == method_id:
                return None if path is None else Path(path)
        return None


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
    if method_version != "v28" or decoder != "negative_binomial":
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
    if set(payload) != expected_fields:
        raise RunnerContractError("v28 candidate decoder payload fields differ")
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


@dataclass(frozen=True, slots=True)
class RepositoryAdapterDispatcher:
    """Concrete dispatch for every implemented adapter; unbuilt envs stay visible."""

    repository_root: Path
    environments: ExecutionEnvironmentRegistry

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
        object.__setattr__(self, "repository_root", repository)

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
        from maskimpute import MaskImputeConfig, PreZeroCountModelConfig
        from maskimpute.calibration import load_calibration_artifact
        from maskimpute_benchmark.methods.maskimpute import _run_in_tree

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
        )
        diagnostics = execution.ablation_result.diagnostics
        score_diagnostics = diagnostics.get("score")
        if request.count_score_manifest_sha256 is not None:
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

    def __call__(self, request: ExecutionRequest) -> AdapterOutcome:
        request.validate_integrity()
        method_id = request.method_spec.id
        try:
            if method_id == "observed":
                from .methods import run_observed

                execution = run_observed(request.method_spec, request.method_input)
            elif method_id in {"maskimpute", "capacity-matched-ae"}:
                return self._in_tree(request)
            elif method_id == "d3impute":
                return AdapterOutcome.unavailable(
                    "external_reference_input_not_prepared:d3impute"
                )
            elif method_id not in self.supported_method_ids:
                return AdapterOutcome.unavailable(
                    f"adapter_not_implemented:{method_id}"
                )
            else:
                executable = self.environments.executable_for(method_id)
                if executable is None:
                    return AdapterOutcome.unavailable(
                        f"environment_executable_unavailable:{method_id}"
                    )
                source = request.method_spec.source.cache_path
                if source is None:
                    return AdapterOutcome.unavailable(
                        f"pinned_source_path_unavailable:{method_id}"
                    )
                source_dir = self.repository_root / source
                seed = request.model_seed
                if seed is None:
                    return AdapterOutcome.failed(f"stochastic_seed_missing:{method_id}")
                if method_id in {"alra", "saver"}:
                    from .methods import run_alra, run_saver

                    function = run_alra if method_id == "alra" else run_saver
                    execution = function(
                        request.method_spec,
                        request.method_input,
                        source_dir=source_dir,
                        rscript=executable,
                        seed=seed,
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
                    f"{error.reason_code}:{method_id}",
                    stdout=error.stdout,
                    stderr=error.stderr,
                )
            return AdapterOutcome.failed(
                f"adapter_exception:{method_id}:{type(error).__name__}",
                stderr=str(error).encode("utf-8", errors="replace"),
            )


@dataclass(frozen=True, slots=True)
class SpawnedRepositoryExecutor:
    """Publication executor using spawn plus independent parent telemetry."""

    dispatcher: RepositoryAdapterDispatcher

    def __call__(self, request: ExecutionRequest) -> AdapterOutcome:
        return execute_adapter_in_spawned_process(request, self.dispatcher)


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
) -> CheckpointReport:
    """Run/resume the fixed tracked development competition with no design overrides."""

    return _run_competition_with_authority(
        output_dir,
        load_runner_authority(),
        environment_overrides=environment_overrides,
    )


def run_v28_revision_competition(
    output_dir: Path,
    *,
    environment_overrides: Mapping[str, Path] | None = None,
) -> CheckpointReport:
    """Run/resume the separately tracked conditional v28 revision panel."""

    input_sha256, report_sha256 = _validate_v28_activation()
    authority = _bind_v28_activation(
        load_v28_revision_authority(),
        input_sha256,
        report_sha256,
    )
    return _run_competition_with_authority(
        output_dir,
        authority,
        environment_overrides=environment_overrides,
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
        r_library_paths={
            "saver": (repository / "artifacts/envs/saver-r/library",)
        },
    )
    plan = build_competition_plan(
        registry,
        bindings,
        authority,
        execution_environment_sha256=environments.registry_sha256,
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
    request: ExecutionRequest,
    executor: Callable[[ExecutionRequest], AdapterOutcome],
) -> None:
    try:
        outcome = executor(request)
        if not isinstance(outcome, AdapterOutcome):
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


def execute_adapter_in_spawned_process(
    request: ExecutionRequest,
    executor: Callable[[ExecutionRequest], AdapterOutcome],
    *,
    poll_interval_seconds: float = 0.05,
    resource_sampler: ResourceSampler | None = None,
) -> AdapterOutcome:
    """Execute from a fresh interpreter that never receives evaluator AnnData.

    This enforces a process-level data boundary, not a filesystem or hostile-code
    sandbox.  Only ``ExecutionRequest`` is serialized into the child.
    """

    if not isinstance(request, ExecutionRequest):
        raise TypeError("request must be an ExecutionRequest")
    if not callable(executor):
        raise TypeError("executor must be callable")
    request.validate_integrity()
    if (
        isinstance(poll_interval_seconds, bool)
        or not isinstance(poll_interval_seconds, (int, float))
        or not math.isfinite(poll_interval_seconds)
        or poll_interval_seconds <= 0
    ):
        raise ValueError("poll_interval_seconds must be positive and finite")
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
        args=(child_connection, request, executor),
        daemon=False,
    )
    started = time.monotonic()
    process.start()
    child_connection.close()
    if process.pid is None:  # pragma: no cover - multiprocessing invariant
        raise RunnerContractError("spawned adapter process has no process ID")
    message: tuple[object, ...] | None = None
    deadline = started + request.timeout_seconds
    peak_rss: int | None = None
    peak_gpu: int | None = 0 if not request.method_spec.resources.gpu_required else None
    rss_provenance = "rss_measurement_unavailable"
    gpu_provenance = (
        "not_applicable_cpu_only_method"
        if not request.method_spec.resources.gpu_required
        else "gpu_measurement_unavailable"
    )
    next_resource_sample = started

    def sample_resources() -> None:
        nonlocal peak_rss, peak_gpu, rss_provenance, gpu_provenance
        try:
            sample = sampler.sample(
                process.pid,
                gpu_required=request.method_spec.resources.gpu_required,
            )
        except Exception:
            return
        if not isinstance(sample, ResourceSample):
            return
        rss_provenance = sample.rss_provenance
        gpu_provenance = sample.gpu_provenance
        if sample.peak_rss_bytes is not None:
            peak_rss = max(peak_rss or 0, sample.peak_rss_bytes)
        if sample.peak_gpu_bytes is not None:
            peak_gpu = max(peak_gpu or 0, sample.peak_gpu_bytes)

    sample_resources()
    try:
        while time.monotonic() < deadline:
            if parent_connection.poll(float(poll_interval_seconds)):
                message = parent_connection.recv()
                break
            now = time.monotonic()
            if now >= next_resource_sample:
                sample_resources()
                next_resource_sample = now + (
                    1.0 if request.method_spec.resources.gpu_required else 0.05
                )
            if not process.is_alive():
                break
        sample_resources()
        if message is None and parent_connection.poll(0):
            message = parent_connection.recv()
        if message is None and process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
            elapsed = max(0.0, time.monotonic() - started)
            if peak_rss is None or (
                request.method_spec.resources.gpu_required and peak_gpu is None
            ):
                return AdapterOutcome.infrastructure_error(
                    "resource_telemetry_unavailable",
                    runtime_seconds=elapsed,
                    peak_rss_bytes=peak_rss or 0,
                    peak_gpu_bytes=peak_gpu or 0,
                    rss_measurement=rss_provenance,
                    gpu_measurement=gpu_provenance,
                )
            return AdapterOutcome.timeout(
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        process.join(timeout=2)
        elapsed = max(0.0, time.monotonic() - started)
        if peak_rss is None or (
            request.method_spec.resources.gpu_required and peak_gpu is None
        ):
            return AdapterOutcome.infrastructure_error(
                "resource_telemetry_unavailable",
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss or 0,
                peak_gpu_bytes=peak_gpu or 0,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        assert peak_gpu is not None
        if message is None:
            return AdapterOutcome.infrastructure_error(
                f"adapter_process_exit_{process.exitcode}",
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        if message[0] == "error":
            return AdapterOutcome.failed(
                f"executor_exception:{message[1]}",
                stderr=str(message[2]).encode("utf-8", errors="replace"),
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        if message[0] != "outcome" or len(message) != 2:
            return AdapterOutcome.infrastructure_error(
                "malformed_adapter_process_message",
                runtime_seconds=elapsed,
                peak_rss_bytes=peak_rss,
                peak_gpu_bytes=peak_gpu,
                rss_measurement=rss_provenance,
                gpu_measurement=gpu_provenance,
            )
        outcome = message[1]
        if not isinstance(outcome, AdapterOutcome):
            return AdapterOutcome.infrastructure_error(
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
            return AdapterOutcome.resource_exceeded(
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
            return AdapterOutcome.resource_exceeded(
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
    "EvaluatedAttempt",
    "ExecutionEnvironmentRegistry",
    "ExecutionRequest",
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
    "derive_authorized_configurations",
    "execute_adapter_in_spawned_process",
    "execute_competition_plan",
    "evaluate_adapter_outcome",
    "enforce_calibration_fold_receipt",
    "method_input_sha256",
    "maskimpute_decoder_for_configuration",
    "maskimpute_variant_for_configuration",
    "load_prepared_development_panel",
    "load_runner_authority",
    "load_v28_revision_authority",
    "prepare_dataset_for_execution",
    "prepare_dataset_pair_for_execution",
    "run_development_competition",
    "run_v28_revision_competition",
    "validate_development_manifest_payload",
]
