"""No-override execution of the once-only frozen final publication panel."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Literal
import zlib

import numpy as np

from .comparator_tuning import (
    BoundComparatorConfiguration,
    ComparatorSelectionProjection,
    ComparatorTuningError,
    _canonical_comparator_tuning_authority,
    comparator_method_binding,
    comparator_selection_projection,
    comparator_selection_projection_value,
)
from .direct_values import direct_equal, direct_json_value
from .methods.registry import MethodRegistry
from .prezero_evidence import (
    PREZERO_STORAGE_COMPRESSION_LEVEL,
    PREZERO_STORAGE_ENCODING,
    zlib_compress_bound as _prezero_zlib_compress_bound,
)
from .protocol import canonical_sha256
from .runner import (
    DEVELOPMENT_MECHANISMS,
    DEVELOPMENT_MODEL_SEEDS,
    DEVELOPMENT_VIEWS,
    AuthorizedConfiguration,
    AdapterOutcome,
    CheckpointStore,
    DatasetBinding,
    DatasetQCPolicy,
    ExecutionEnvironmentRegistry,
    EvaluatedAttempt,
    ExecutionAuthorityContext,
    ExecutionRequest,
    FinalComparatorExecutionRequest,
    PreparedDataset,
    RepositoryAdapterDispatcher,
    RunnerContractError,
    RunPlanEntry,
    SpawnedRepositoryExecutor,
    _prezero_evaluator_targets,
    _prepare_dataset_with_exclusions,
    _unlink_owned_staging_temporary,
    derive_lock_only_environment_ids,
    direct_bound_comparator_value,
    enforce_calibration_fold_receipt,
    evaluate_adapter_outcome,
    method_input_sha256,
    prepare_dataset_pair_for_execution,
)
from .trajectory_dataset import (
    REGISTERED_TRAJECTORY_DATASET_ID,
    RegisteredTrajectoryBinding,
    TrajectoryPreparedDataset,
    generate_registered_trajectory_dataset,
    load_trajectory_authority,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_STAGING_FILE = re.compile(r"\.(?P<canonical>.+)\.(?P<token>[a-z0-9_]{8})\.tmp\Z")
_TRANSACTION_FILE = re.compile(r"[0-9]{8}\.json\Z")
_FINAL_RUN_ID = re.compile(r"final-[a-z0-9-]+\Z")
_TRAJECTORY_RUN_ID = re.compile(r"trajectory-[a-z0-9-]+\Z")
_FINAL_DRAWS = tuple(f"draw-{draw:02d}" for draw in range(1, 6))
_FINAL_OUTPUT_ENCODING = "zlib_raw_f64_v1"
_FINAL_OUTPUT_COMPRESSION_LEVEL = 6
_FINAL_NATIVE_RETENTION = "omitted_redundant_final_output"
_FINAL_STORAGE_POLICY = {
    "evaluator_output_encoding": _FINAL_OUTPUT_ENCODING,
    "evaluator_output_compression_level": _FINAL_OUTPUT_COMPRESSION_LEVEL,
    "native_output_retention": _FINAL_NATIVE_RETENTION,
    "p_pre_zero_encoding": PREZERO_STORAGE_ENCODING,
    "p_pre_zero_compression_level": PREZERO_STORAGE_COMPRESSION_LEVEL,
}
_FINAL_MATRIX_UNCOMPRESSED_NBYTES = 2_700 * 1_200 * 8
_FINAL_PREZERO_MATRIX_UNCOMPRESSED_NBYTES = 2_700 * 1_200 * 8
_FINAL_RECORD_OVERHEAD_BYTES = 1024 * 1024
_FINAL_STORAGE_RESERVE_BYTES = 1024**3


class FinalRunnerContractError(ValueError):
    """Raised when frozen final execution authority is incomplete or changed."""


def _zlib_compress_bound(uncompressed_nbytes: int) -> int:
    """Return zlib's documented single-call compression upper bound."""

    return (
        uncompressed_nbytes
        + (uncompressed_nbytes >> 12)
        + (uncompressed_nbytes >> 14)
        + (uncompressed_nbytes >> 25)
        + 13
    )


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FinalRunnerContractError(f"{name} is not a SHA-256 digest")
    return value


def validate_final_manifest_payload(
    payload: Mapping[str, object],
) -> tuple[DatasetBinding, ...]:
    """Require the exact validated 4 x 5 x 2 unseen final simulation panel."""

    if not isinstance(payload, Mapping):
        raise TypeError("final manifest must be a mapping")
    rows = payload.get("rows")
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("namespace") != "final"
        or payload.get("status") != "completed"
        or payload.get("independent_unit_count") != 20
        or payload.get("completed_count") != 40
        or payload.get("failed_count") != 0
        or not isinstance(payload.get("execution_claim_id"), str)
        or not payload.get("execution_claim_id")
        or not isinstance(payload.get("round_id"), str)
        or not payload.get("round_id")
        or type(rows) is not list
        or len(rows) != 40
        or payload.get("manifest_sha256") != canonical_sha256(unsigned)
    ):
        raise FinalRunnerContractError(
            "final manifest must be the complete canonical 40-dataset panel"
        )
    manifest_sha256 = _sha256(payload.get("manifest_sha256"), "final manifest")
    protocol_sha256 = _sha256(payload.get("protocol_sha256"), "final protocol")
    design_sha256 = _sha256(payload.get("design_sha256"), "final design")
    seed_source_sha256 = _sha256(payload.get("seed_source_sha256"), "final seeds")
    expected = [
        (mechanism, draw, view)
        for mechanism in DEVELOPMENT_MECHANISMS
        for draw in _FINAL_DRAWS
        for view in DEVELOPMENT_VIEWS
    ]
    observed: list[tuple[object, object, object]] = []
    bindings: list[DatasetBinding] = []
    dataset_ids: set[str] = set()
    output_paths: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FinalRunnerContractError(f"final row {index} is invalid")
        observed.append(
            (row.get("mechanism"), row.get("biological_id"), row.get("technical_view"))
        )
        dataset_id = row.get("dataset_id")
        output_path = row.get("output_path")
        independent_unit_id = row.get("independent_unit_id")
        if (
            row.get("status") != "completed"
            or row.get("cells") != 2700
            or row.get("genes") != 1200
            or not isinstance(dataset_id, str)
            or not dataset_id
            or not isinstance(output_path, str)
            or not output_path
            or output_path.startswith("/")
            or ".." in output_path.split("/")
            or not isinstance(independent_unit_id, str)
            or not independent_unit_id
            or dataset_id in dataset_ids
            or output_path in output_paths
        ):
            raise FinalRunnerContractError(f"final row {index} is invalid")
        dataset_ids.add(dataset_id)
        output_paths.add(output_path)
        bindings.append(
            DatasetBinding(
                mechanism=str(row.get("mechanism")),
                biological_id=str(row.get("biological_id")),
                technical_view=str(row.get("technical_view")),
                dataset_id=dataset_id,
                dataset_sha256=_sha256(
                    row.get("dataset_sha256"), f"final row {index} dataset"
                ),
                output_file_sha256=_sha256(
                    row.get("output_file_sha256"), f"final row {index} output"
                ),
                truth_sha256=_sha256(
                    row.get("truth_sha256"), f"final row {index} truth"
                ),
                output_path=output_path,
                independent_unit_id=independent_unit_id,
                cells=2700,
                genes=1200,
                manifest_sha256=manifest_sha256,
                protocol_sha256=protocol_sha256,
                design_sha256=design_sha256,
                seed_source_sha256=seed_source_sha256,
            )
        )
    if observed != expected:
        raise FinalRunnerContractError("final 40-dataset panel order differs")
    for first, second in zip(bindings[::2], bindings[1::2], strict=True):
        if (
            first.independent_unit_id != second.independent_unit_id
            or first.truth_sha256 != second.truth_sha256
        ):
            raise FinalRunnerContractError(
                "final paired views do not share one biological unit and truth"
            )
    return tuple(bindings)


def _stable_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


@contextmanager
def _pinned_parent_directory(path: Path, name: str) -> Iterator[int]:
    """Yield a parent dirfd reached through one revalidated no-symlink walk."""

    absolute = path.absolute()
    if not absolute.name:
        raise FinalRunnerContractError(f"{name} path is invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    edges: list[tuple[int, str, int, tuple[int, ...]]] = []
    try:
        current = os.open(absolute.anchor, flags)
        descriptors.append(current)
        for component in absolute.parent.relative_to(absolute.anchor).parts:
            named = os.stat(component, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(named.st_mode):
                raise FinalRunnerContractError(f"{name} parent path is not a directory")
            child = os.open(component, flags, dir_fd=current)
            opened = os.fstat(child)
            expected = _directory_identity(named)
            if _directory_identity(opened) != expected:
                os.close(child)
                raise FinalRunnerContractError(
                    f"{name} parent path changed while being opened"
                )
            descriptors.append(child)
            edges.append((current, component, child, expected))
            current = child
        yield current
        for parent, component, child, expected in edges:
            named_after = os.stat(
                component,
                dir_fd=parent,
                follow_symlinks=False,
            )
            opened_after = os.fstat(child)
            if (
                _directory_identity(named_after) != expected
                or _directory_identity(opened_after) != expected
            ):
                raise FinalRunnerContractError(
                    f"{name} parent path changed during validation"
                )
    except FinalRunnerContractError:
        raise
    except OSError as error:
        raise FinalRunnerContractError(f"cannot validate {name} parent path") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_bound_h5ad(
    path: Path,
    binding: DatasetBinding | RegisteredTrajectoryBinding,
    *,
    max_bytes: int | None = None,
    structure_validator: Callable[[Path], None] | None = None,
    parent_descriptor: int | None = None,
):
    """Open one exact H5AD inode and recheck its byte and semantic bindings."""

    if not isinstance(path, Path) or not isinstance(
        binding, (DatasetBinding, RegisteredTrajectoryBinding)
    ):
        raise TypeError("path and binding must be canonical values")
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer or None")
    if structure_validator is not None and not callable(structure_validator):
        raise TypeError("structure_validator must be callable or None")
    if parent_descriptor is not None and (
        type(parent_descriptor) is not int or parent_descriptor < 0
    ):
        raise ValueError("parent_descriptor must be an open descriptor or None")
    if parent_descriptor is None:
        with _pinned_parent_directory(path, "final dataset") as pinned_parent:
            return _read_bound_h5ad_from_parent(
                path,
                binding,
                parent_descriptor=pinned_parent,
                max_bytes=max_bytes,
                structure_validator=structure_validator,
            )
    return _read_bound_h5ad_from_parent(
        path,
        binding,
        parent_descriptor=parent_descriptor,
        max_bytes=max_bytes,
        structure_validator=structure_validator,
    )


def _read_bound_h5ad_from_parent(
    path: Path,
    binding: DatasetBinding | RegisteredTrajectoryBinding,
    *,
    parent_descriptor: int,
    max_bytes: int | None,
    structure_validator: Callable[[Path], None] | None,
):
    import anndata as ad

    from .schema import benchmark_dataset_sha256

    descriptor = -1
    try:
        if not stat.S_ISDIR(os.fstat(parent_descriptor).st_mode):
            raise FinalRunnerContractError("final dataset parent is not a directory")
        named_before = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or _stable_file_identity(opened_before)
            != _stable_file_identity(named_before)
        ):
            raise FinalRunnerContractError("final dataset is not a unique regular file")
        if max_bytes is not None and opened_before.st_size > max_bytes:
            raise FinalRunnerContractError("final dataset exceeds its size bound")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        expected_file_sha256 = (
            binding.output_file_sha256
            if isinstance(binding, DatasetBinding)
            else binding.dataset_file_sha256
        )
        if digest.hexdigest() != expected_file_sha256:
            raise FinalRunnerContractError("final dataset file checksum differs")
        opened_path = Path(f"/proc/self/fd/{descriptor}")
        if structure_validator is not None:
            structure_validator(opened_path)
        dataset = ad.read_h5ad(opened_path)
        opened_after = os.fstat(descriptor)
        named_after = os.stat(
            path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _stable_file_identity(opened_before) != _stable_file_identity(
            opened_after
        ) or _stable_file_identity(opened_before) != _stable_file_identity(named_after):
            raise FinalRunnerContractError("final dataset changed while being loaded")
        if benchmark_dataset_sha256(dataset) != binding.dataset_sha256:
            raise FinalRunnerContractError("final dataset semantic checksum differs")
        return dataset
    except FinalRunnerContractError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise FinalRunnerContractError(
            f"cannot load final dataset {binding.dataset_id}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256_bound_regular_file(
    parent_descriptor: int,
    leaf_name: str,
    name: str,
    *,
    max_bytes: int | None = None,
) -> str:
    """Hash one unique regular leaf through an already pinned parent dirfd."""

    descriptor = -1
    try:
        named_before = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            leaf_name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or _stable_file_identity(opened_before)
            != _stable_file_identity(named_before)
        ):
            raise FinalRunnerContractError(f"{name} is not a unique regular file")
        if max_bytes is not None and opened_before.st_size > max_bytes:
            raise FinalRunnerContractError(f"{name} exceeds its size bound")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        named_after = os.stat(
            leaf_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _stable_file_identity(opened_before) != _stable_file_identity(
            opened_after
        ) or _stable_file_identity(opened_before) != _stable_file_identity(named_after):
            raise FinalRunnerContractError(f"{name} changed while being hashed")
        return digest.hexdigest()
    except FinalRunnerContractError:
        raise
    except OSError as error:
        raise FinalRunnerContractError(f"cannot hash {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validated_evaluated_final_round_snapshot(
    repository: Path,
    round_dir: Path,
) -> str:
    """Return one read-only commitment to exact evaluated lifecycle inputs."""

    from .datasets import validate_evaluated_final_dataset_status
    from .study import (
        _validate_freeze,
        _validate_registry,
        _validate_result_files,
        _validate_result_journal,
        _validate_seed_manifest,
        _validate_state_record_chain,
        _verify_frozen_repository,
    )

    selected_repository, destination = _canonical_round(repository, round_dir)
    try:
        freeze = _validate_freeze(destination, selected_repository)
        registry = _validate_registry(
            selected_repository,
            destination,
            freeze,
            expected_state="evaluated",
        )
        materialization, claim, receipt = _validate_state_record_chain(
            destination,
            freeze,
            expected_state="evaluated",
        )
        repeated_materialization, seed_manifest = _validate_seed_manifest(
            destination,
            freeze,
        )
        if (
            not isinstance(materialization, Mapping)
            or not isinstance(claim, Mapping)
            or not isinstance(receipt, Mapping)
            or repeated_materialization != materialization
        ):
            raise FinalRunnerContractError(
                "evaluated final lifecycle records are unavailable"
            )
        evaluation = receipt.get("result_manifest")
        if not isinstance(evaluation, Mapping):
            raise FinalRunnerContractError(
                "evaluated final result manifest is unavailable"
            )
        allowed_paths = _validate_result_files(
            selected_repository,
            destination,
            evaluation,
        )
        journal = _validate_result_journal(
            selected_repository,
            destination,
            freeze,
            materialization,
            claim,
        )
        evaluation_files = evaluation.get("result_files", [])
        if (
            not isinstance(evaluation_files, list)
            or journal.get("cumulative_result_files") != evaluation_files
            or journal.get("allowed_result_paths") != allowed_paths
        ):
            raise FinalRunnerContractError(
                "evaluated final result journal differs from its receipt"
            )
        status = validate_evaluated_final_dataset_status(
            destination / "results/dataset_status.json",
            repo=selected_repository,
            round_dir=destination,
            protocol_path=selected_repository / str(freeze["protocol_path"]),
            execution_claim=claim,
            seed_manifest=seed_manifest,
        )
        verified_freeze = _verify_frozen_repository(
            selected_repository,
            destination,
            allowed_result_paths=allowed_paths,
        )
    except FinalRunnerContractError:
        raise
    except Exception as error:
        raise FinalRunnerContractError(
            "evaluated final round failed read-only validation"
        ) from error
    if dict(verified_freeze) != dict(freeze):
        raise FinalRunnerContractError(
            "evaluated final freeze changed during validation"
        )
    return canonical_sha256(
        {
            "freeze": dict(freeze),
            "registry": dict(registry),
            "materialization": dict(materialization),
            "seed_manifest": dict(seed_manifest),
            "execution_claim": dict(claim),
            "evaluation_receipt": dict(receipt),
            "allowed_result_paths": sorted(allowed_paths),
            "result_journal": {
                "sequence": journal.get("sequence"),
                "head_sha256": journal.get("head_sha256"),
                "cumulative_result_files": journal.get("cumulative_result_files"),
            },
            "dataset_status_sha256": status.get("manifest_sha256"),
        }
    )


def load_prepared_final_panel(
    repository: Path,
    round_dir: Path,
    *,
    allow_evaluated: bool = False,
    simulator_assets_root: Path | None = None,
    simulator_r_environment: Path | None = None,
) -> tuple[tuple[DatasetBinding, ...], Mapping[str, PreparedDataset]]:
    """Byte-revalidate and pair-union-QC the exact unseen final panel."""

    from .datasets import validate_dataset_status

    if not isinstance(repository, Path) or not isinstance(round_dir, Path):
        raise TypeError("repository and round_dir must be pathlib.Path values")
    if type(allow_evaluated) is not bool:
        raise TypeError("allow_evaluated must be a boolean")
    try:
        selected_repository = repository.resolve(strict=True)
        destination = round_dir.resolve(strict=True)
        destination.relative_to(selected_repository)
    except ValueError as error:
        raise FinalRunnerContractError(
            "final round must be inside repository"
        ) from error
    except OSError as error:
        raise FinalRunnerContractError(
            "final repository or round is unavailable"
        ) from error
    if destination != round_dir.absolute() or not destination.is_dir():
        raise FinalRunnerContractError("final round path is not canonical")
    runtime_paths = (simulator_assets_root, simulator_r_environment)
    if any(
        value is not None and not isinstance(value, Path) for value in runtime_paths
    ):
        raise TypeError("simulator runtime paths must be pathlib.Path values")
    if any(value is None for value in runtime_paths):
        if any(value is not None for value in runtime_paths) or not allow_evaluated:
            raise FinalRunnerContractError(
                "running final panel validation requires both runtime asset paths"
            )
        runtime_status_kwargs: dict[str, Path] = {}
    else:
        assert simulator_assets_root is not None
        assert simulator_r_environment is not None
        runtime_status_kwargs = {
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
        }
    status_path = destination / "results/dataset_status.json"
    try:
        status = validate_dataset_status(
            status_path,
            repo=selected_repository,
            round_dir=destination,
            **runtime_status_kwargs,
        )
        bindings = validate_final_manifest_payload(status)
    except Exception as error:
        if allow_evaluated:
            try:
                snapshot = _validated_evaluated_final_round_snapshot(
                    selected_repository,
                    destination,
                )
                direct = _load_unclaimed_prepared_final_panel(
                    selected_repository,
                    destination,
                )
                if (
                    _validated_evaluated_final_round_snapshot(
                        selected_repository,
                        destination,
                    )
                    != snapshot
                ):
                    raise FinalRunnerContractError(
                        "evaluated final round changed during panel preparation"
                    )
                return direct
            except Exception as evaluated_error:
                raise FinalRunnerContractError(
                    "final dataset status failed running and evaluated revalidation"
                ) from evaluated_error
        raise FinalRunnerContractError(
            "final dataset status failed byte-level revalidation"
        ) from error
    prepared = _prepare_final_panel_bindings(destination, bindings)
    try:
        status_after = validate_dataset_status(
            status_path,
            repo=selected_repository,
            round_dir=destination,
            **runtime_status_kwargs,
        )
    except Exception as error:
        raise FinalRunnerContractError(
            "final dataset status changed during panel preparation"
        ) from error
    if status_after != status:
        raise FinalRunnerContractError(
            "final dataset status changed during panel preparation"
        )
    return bindings, prepared


def _prepare_final_panel_bindings(
    destination: Path,
    bindings: tuple[DatasetBinding, ...],
) -> Mapping[str, PreparedDataset]:
    """Prepare already-validated bindings without consulting lifecycle state."""

    policy = DatasetQCPolicy.fixed()
    prepared: dict[str, PreparedDataset] = {}
    results_root = destination / "results"
    for first_binding, second_binding in zip(
        bindings[::2], bindings[1::2], strict=True
    ):
        try:
            first_dataset = _read_bound_h5ad(
                results_root / first_binding.output_path, first_binding
            )
            second_dataset = _read_bound_h5ad(
                results_root / second_binding.output_path, second_binding
            )
            first, second = prepare_dataset_pair_for_execution(
                first_dataset,
                second_dataset,
                first_binding,
                second_binding,
                policy,
            )
        except Exception as error:
            raise FinalRunnerContractError(
                "final pair preparation failed for "
                f"{first_binding.mechanism}/{first_binding.biological_id}"
            ) from error
        prepared[first.binding.dataset_id] = first
        prepared[second.binding.dataset_id] = second
    expected_ids = tuple(binding.dataset_id for binding in bindings)
    if tuple(prepared) != expected_ids:
        raise FinalRunnerContractError(
            "prepared final panel order or cardinality drifted"
        )
    return MappingProxyType(prepared)


def _load_unclaimed_prepared_final_panel(
    repository: Path,
    round_dir: Path,
) -> tuple[tuple[DatasetBinding, ...], Mapping[str, PreparedDataset]]:
    """Load a final panel before claim validation during exact resume recovery.

    Lifecycle validation cannot run while runner-owned bytes are not yet present in
    the result journal.  This path therefore validates the canonical status and
    every bound H5AD directly; the caller must validate all other observed result
    scopes before adding the cumulative inventory to the journal.
    """

    selected_repository, destination = _canonical_round(repository, round_dir)
    status_path = destination / "results/dataset_status.json"
    raw = _read_unique_file(status_path, "final dataset status")
    status = _strict_json(raw, "final dataset status")
    try:
        canonical_status = (
            json.dumps(status, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FinalRunnerContractError(
            "final dataset status is not canonical"
        ) from error
    if raw != canonical_status:
        raise FinalRunnerContractError("final dataset status is not canonical")
    try:
        bindings = validate_final_manifest_payload(status)
    except (TypeError, ValueError) as error:
        raise FinalRunnerContractError(
            "final dataset status failed direct revalidation"
        ) from error
    prepared = _prepare_final_panel_bindings(destination, bindings)
    if _read_unique_file(
        status_path, "final dataset status"
    ) != raw or selected_repository != repository.resolve(strict=True):
        raise FinalRunnerContractError(
            "final dataset status changed during direct preparation"
        )
    return bindings, prepared


def _validate_trajectory_h5ad_structure(path: Path) -> None:
    """Reject non-self-contained or structurally unbounded trajectory H5ADs."""

    from .scaling import ScalingContractError, _validate_scaling_h5ad_structure

    try:
        _validate_scaling_h5ad_structure(
            path,
            2_700,
            120,
            profile="trajectory",
        )
    except ScalingContractError as error:
        raise FinalRunnerContractError(str(error)) from error


def _trajectory_binding(
    authority: object,
    *,
    authority_file_sha256: str,
    dataset_file_sha256: str,
) -> RegisteredTrajectoryBinding:
    from .trajectory_dataset import TrajectoryAuthority

    if not isinstance(authority, TrajectoryAuthority):
        raise TypeError("authority must be a TrajectoryAuthority")
    return RegisteredTrajectoryBinding(
        schema_version="trajectory-execution-dataset-binding-v1",
        dataset_id=REGISTERED_TRAJECTORY_DATASET_ID,
        mechanism=authority.mechanism,
        biological_id=authority.biological_id,
        technical_view=authority.technical_view,
        condition=authority.condition,
        draw=authority.draw,
        cells=authority.cells,
        genes=authority.genes,
        source_id=authority.source_id,
        root_cell_id=authority.root_cell_id,
        seed=authority.seed,
        dataset_sha256=authority.expected_dataset_sha256,
        dataset_file_path="results/trajectory/dataset/evaluator.h5ad",
        dataset_file_sha256=dataset_file_sha256,
        authority_path="study/trajectory_panel.json",
        authority_file_sha256=authority_file_sha256,
        authority_sha256=authority.authority_sha256,
        registered_binding_sha256=authority.binding_sha256,
    )


def _prepare_registered_trajectory_dataset(
    dataset: object,
    binding: RegisteredTrajectoryBinding,
) -> PreparedDataset:
    prepared = _prepare_dataset_with_exclusions(
        dataset,
        binding,
        DatasetQCPolicy.fixed(),
        None,
    )
    evaluator_columns = set(prepared.evaluator_dataset.obs.columns)
    provenance = prepared.evaluator_dataset.uns.get("provenance")
    parameters = (
        provenance.get("parameters") if isinstance(provenance, Mapping) else None
    )
    if (
        prepared.method_input.obs_covariates
        or prepared.method_input.var_covariates
        or not {"pseudotime", "group"}.issubset(evaluator_columns)
        or not isinstance(parameters, Mapping)
        or parameters.get("root_cell_id") != binding.root_cell_id
        or parameters.get("source_id") != binding.source_id
    ):
        raise FinalRunnerContractError(
            "trajectory evaluator targets crossed the method-input boundary"
        )
    return prepared


def _trajectory_dataset_receipt(
    binding: RegisteredTrajectoryBinding,
    prepared: PreparedDataset,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "scope": "supplementary_trajectory",
        "binding": asdict(binding),
        "method_input_sha256": method_input_sha256(prepared.method_input),
        "excluded_cell_count": prepared.audit.excluded_cell_count,
        "excluded_cell_ids_sha256": prepared.audit.excluded_cell_ids_sha256,
        "retained_cell_count": prepared.audit.retained_cell_count,
        "retained_cell_ids_sha256": prepared.audit.retained_cell_ids_sha256,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


def _remove_unreceipted_trajectory_dataset(round_dir: Path) -> bool:
    """Remove only the owned dataset half of an interrupted two-file publish."""

    if not isinstance(round_dir, Path):
        raise TypeError("round_dir must be a pathlib.Path")
    try:
        destination = round_dir.resolve(strict=True)
    except OSError as error:
        raise FinalRunnerContractError("final round is unavailable") from error
    if destination != round_dir.absolute():
        raise FinalRunnerContractError("final round path is not canonical")
    dataset = destination / "results/trajectory/dataset/evaluator.h5ad"
    receipt = destination / "results/trajectory/dataset/dataset_receipt.json"
    dataset_exists = os.path.lexists(dataset)
    receipt_exists = os.path.lexists(receipt)
    if receipt_exists and not dataset_exists:
        raise FinalRunnerContractError(
            "registered trajectory receipt exists without its dataset"
        )
    if not dataset_exists or receipt_exists:
        return False
    parent_descriptor = -1
    dataset_descriptor = -1
    try:
        parent_named = dataset.parent.lstat()
        if dataset.parent.resolve(strict=True) != dataset.parent.absolute():
            raise FinalRunnerContractError(
                "unreceipted trajectory dataset parent is unsafe"
            )
        parent_descriptor = os.open(
            dataset.parent,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        parent_opened = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_opened.st_mode)
            or (parent_opened.st_dev, parent_opened.st_ino)
            != (parent_named.st_dev, parent_named.st_ino)
            or parent_opened.st_uid != os.geteuid()
        ):
            raise FinalRunnerContractError(
                "unreceipted trajectory dataset parent is unsafe"
            )
        named_before = os.stat(
            dataset.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        dataset_descriptor = os.open(
            dataset.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(dataset_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or _stable_file_identity(opened) != _stable_file_identity(named_before)
        ):
            raise FinalRunnerContractError(
                "unreceipted trajectory evaluator dataset is unsafe"
            )
        named_immediately_before = os.stat(
            dataset.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if _stable_file_identity(named_immediately_before) != _stable_file_identity(
            opened
        ):
            raise FinalRunnerContractError(
                "unreceipted trajectory evaluator dataset changed"
            )
        try:
            os.stat(
                receipt.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FinalRunnerContractError(
                "registered trajectory receipt appeared before dataset recovery"
            )
        os.unlink(dataset.name, dir_fd=parent_descriptor)
        if os.fstat(dataset_descriptor).st_nlink != 0:
            raise FinalRunnerContractError(
                "unreceipted trajectory evaluator dataset unlink raced"
            )
        try:
            os.stat(
                dataset.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FinalRunnerContractError(
                "unreceipted trajectory evaluator dataset was replaced"
            )
        os.fsync(parent_descriptor)
    except FinalRunnerContractError:
        raise
    except OSError as error:
        raise FinalRunnerContractError(
            "unreceipted trajectory evaluator dataset cannot be removed"
        ) from error
    finally:
        if dataset_descriptor >= 0:
            os.close(dataset_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    return True


def load_prepared_trajectory_dataset(
    repository: Path,
    round_dir: Path,
) -> TrajectoryPreparedDataset:
    """Load an existing registered trajectory dataset without mutating its tree."""

    from .scaling import _scaling_h5ad_size_ceiling

    selected_repository, destination = _canonical_round(repository, round_dir)
    authority_path = selected_repository / "study/trajectory_panel.json"
    dataset_relative = "results/trajectory/dataset/evaluator.h5ad"
    receipt_relative = "results/trajectory/dataset/dataset_receipt.json"
    dataset_path = destination / dataset_relative
    receipt_path = destination / receipt_relative
    if not os.path.lexists(dataset_path) or not os.path.lexists(receipt_path):
        raise FinalRunnerContractError(
            "registered trajectory dataset pair is incomplete and unavailable"
        )
    try:
        initial_identities = {
            authority_path: _stable_file_identity(authority_path.lstat()),
            dataset_path: _stable_file_identity(dataset_path.lstat()),
            receipt_path: _stable_file_identity(receipt_path.lstat()),
        }
    except OSError as error:
        raise FinalRunnerContractError(
            "registered trajectory dataset pair is unavailable"
        ) from error

    authority_raw = _read_unique_file(
        authority_path,
        "registered trajectory authority",
        max_bytes=1024 * 1024,
    )
    authority_payload = _strict_json(authority_raw, "registered trajectory authority")
    exact_authority_raw = (
        json.dumps(
            authority_payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if authority_raw != exact_authority_raw:
        raise FinalRunnerContractError(
            "registered trajectory authority is not canonical"
        )
    authority_file_sha256 = hashlib.sha256(authority_raw).hexdigest()
    try:
        authority = load_trajectory_authority(authority_path)
    except Exception as error:
        raise FinalRunnerContractError(
            "registered trajectory dataset authority is invalid"
        ) from error
    validated_authority_payload: dict[str, object] = {}
    for definition in fields(authority):
        value = getattr(authority, definition.name)
        validated_authority_payload[definition.name] = (
            dict(value) if isinstance(value, Mapping) else value
        )
    validated_authority_raw = (
        json.dumps(
            validated_authority_payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if validated_authority_raw != authority_raw:
        raise FinalRunnerContractError(
            "registered trajectory dataset authority differs from captured bytes"
        )

    receipt_raw = _read_unique_file(
        receipt_path,
        "registered trajectory dataset receipt",
        max_bytes=1024 * 1024,
    )
    receipt = _strict_json(receipt_raw, "registered trajectory dataset receipt")
    if receipt_raw != _canonical_bytes(receipt) + b"\n":
        raise FinalRunnerContractError(
            "registered trajectory dataset receipt is not canonical"
        )
    binding_payload = receipt.get("binding")
    try:
        if not isinstance(binding_payload, Mapping):
            raise TypeError("binding is not a mapping")
        binding = RegisteredTrajectoryBinding(**dict(binding_payload))
    except (TypeError, ValueError) as error:
        raise FinalRunnerContractError(
            "registered trajectory dataset receipt binding is invalid"
        ) from error
    expected_binding = _trajectory_binding(
        authority,
        authority_file_sha256=authority_file_sha256,
        dataset_file_sha256=binding.dataset_file_sha256,
    )
    if binding != expected_binding:
        raise FinalRunnerContractError(
            "registered trajectory dataset receipt identity differs"
        )

    size_ceiling = _scaling_h5ad_size_ceiling(authority.cells, authority.genes)
    with _pinned_parent_directory(
        dataset_path,
        "registered trajectory evaluator dataset",
    ) as dataset_parent_descriptor:
        try:
            evaluator_dataset = _read_bound_h5ad(
                dataset_path,
                binding,
                max_bytes=size_ceiling,
                structure_validator=_validate_trajectory_h5ad_structure,
                parent_descriptor=dataset_parent_descriptor,
            )
            prepared = _prepare_registered_trajectory_dataset(
                evaluator_dataset,
                binding,
            )
        except FinalRunnerContractError:
            raise
        except Exception as error:
            raise FinalRunnerContractError(
                "registered trajectory evaluator dataset failed preparation"
            ) from error
        if _trajectory_dataset_receipt(binding, prepared) != receipt:
            raise FinalRunnerContractError(
                "registered trajectory prepared input differs from its receipt"
            )
        if (
            _sha256_bound_regular_file(
                dataset_parent_descriptor,
                dataset_path.name,
                "registered trajectory evaluator dataset",
                max_bytes=size_ceiling,
            )
            != binding.dataset_file_sha256
        ):
            raise FinalRunnerContractError(
                "registered trajectory evaluator dataset checksum changed "
                "during preparation"
            )
        try:
            final_identities = {
                path: _stable_file_identity(path.lstat()) for path in initial_identities
            }
        except OSError as error:
            raise FinalRunnerContractError(
                "registered trajectory dataset pair changed during preparation"
            ) from error
        if (
            final_identities != initial_identities
            or _read_unique_file(
                authority_path,
                "registered trajectory authority",
                max_bytes=1024 * 1024,
            )
            != authority_raw
            or _read_unique_file(
                receipt_path,
                "registered trajectory dataset receipt",
                max_bytes=1024 * 1024,
            )
            != receipt_raw
        ):
            raise FinalRunnerContractError(
                "registered trajectory dataset pair changed during preparation"
            )
    return TrajectoryPreparedDataset(
        authority=authority,
        binding=binding,
        prepared=prepared,
        receipt=MappingProxyType(receipt),
        receipt_file_path=receipt_relative,
        receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
    )


def materialize_prepared_trajectory_dataset(
    repository: Path,
    round_dir: Path,
) -> TrajectoryPreparedDataset:
    """Persist and prepare the exact registered evaluator-only trajectory dataset."""

    from .scaling import _scaling_h5ad_size_ceiling

    selected_repository, destination = _canonical_round(repository, round_dir)
    authority_path = selected_repository / "study/trajectory_panel.json"
    dataset_relative = "results/trajectory/dataset/evaluator.h5ad"
    receipt_relative = "results/trajectory/dataset/dataset_receipt.json"
    receipt_path = destination / receipt_relative
    if os.path.lexists(receipt_path):
        return load_prepared_trajectory_dataset(selected_repository, destination)
    authority_raw = _read_unique_file(
        authority_path,
        "registered trajectory authority",
        max_bytes=1024 * 1024,
    )
    authority_payload = _strict_json(authority_raw, "registered trajectory authority")
    exact_authority_raw = (
        json.dumps(
            authority_payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if authority_raw != exact_authority_raw:
        raise FinalRunnerContractError(
            "registered trajectory authority is not canonical"
        )
    authority_file_sha256 = hashlib.sha256(authority_raw).hexdigest()
    try:
        authority = load_trajectory_authority(authority_path)
        generated = generate_registered_trajectory_dataset(authority=authority)
    except Exception as error:
        raise FinalRunnerContractError(
            "registered trajectory dataset authority is invalid"
        ) from error
    if (
        _read_unique_file(
            authority_path,
            "registered trajectory authority",
            max_bytes=1024 * 1024,
        )
        != authority_raw
    ):
        raise FinalRunnerContractError(
            "registered trajectory authority changed during preparation"
        )

    artifacts = CheckpointStore(
        destination,
        authority_repository=selected_repository,
    )
    size_ceiling = _scaling_h5ad_size_ceiling(authority.cells, authority.genes)
    try:
        with tempfile.TemporaryDirectory(
            prefix="maskimpute-trajectory-dataset-stage-"
        ) as staging_name:
            staged_path = Path(staging_name) / "evaluator.h5ad"
            generated.write_h5ad(staged_path)
            _validate_trajectory_h5ad_structure(staged_path)
            raw = _read_unique_file(
                staged_path,
                "staged registered trajectory dataset",
                max_bytes=size_ceiling,
            )
        _remove_unreceipted_trajectory_dataset(destination)
        artifacts._publish_immutable(dataset_relative, raw)
    except FinalRunnerContractError:
        raise
    except (OSError, RunnerContractError, ValueError) as error:
        raise FinalRunnerContractError(
            "registered trajectory evaluator dataset cannot be persisted"
        ) from error
    binding = _trajectory_binding(
        authority,
        authority_file_sha256=authority_file_sha256,
        dataset_file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    expected_prepared = _prepare_registered_trajectory_dataset(generated, binding)
    receipt = _trajectory_dataset_receipt(binding, expected_prepared)
    try:
        artifacts._publish_immutable(
            receipt_relative,
            _canonical_bytes(receipt) + b"\n",
        )
    except (OSError, RunnerContractError) as error:
        raise FinalRunnerContractError(
            "registered trajectory dataset receipt cannot be persisted"
        ) from error
    return load_prepared_trajectory_dataset(selected_repository, destination)


@dataclass(frozen=True, slots=True)
class FinalPlanEntry:
    """One executable attempt or one explicit non-run denominator row."""

    run: RunPlanEntry
    action: Literal["execute", "not_applicable"]
    reason: str | None

    def __post_init__(self) -> None:
        if self.action == "execute" and self.reason is not None:
            raise FinalRunnerContractError(
                "executable final entry has a non-run reason"
            )
        if self.action == "not_applicable" and (
            not isinstance(self.reason, str) or not self.reason
        ):
            raise FinalRunnerContractError("non-run final entry lacks an exact reason")

    def to_dict(self) -> dict[str, object]:
        return {"run": self.run.to_dict(), "action": self.action, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class FrozenPlanMethodAuthority:
    """One exact legacy, selected-comparator, or nonexecution method authority."""

    method_id: str
    legacy_configuration: AuthorizedConfiguration | None
    comparator_configuration: BoundComparatorConfiguration | None
    comparator_nonexecution_identity: Mapping[str, object] | None
    action: Literal["execute", "not_applicable"]
    reason: str | None
    seeds: tuple[int | None, ...]

    def __post_init__(self) -> None:
        choices = (
            self.legacy_configuration,
            self.comparator_configuration,
            self.comparator_nonexecution_identity,
        )
        if sum(value is not None for value in choices) != 1:
            raise FinalRunnerContractError(
                "frozen method configuration authority is not exclusive"
            )
        if not isinstance(self.method_id, str) or not self.method_id:
            raise FinalRunnerContractError("frozen method authority ID is invalid")
        if self.legacy_configuration is not None and (
            self.legacy_configuration.method_id != self.method_id
        ):
            raise FinalRunnerContractError("legacy frozen configuration method differs")
        if self.comparator_configuration is not None and (
            self.comparator_configuration.method.method_id != self.method_id
            or self.comparator_configuration.configuration.method_id != self.method_id
        ):
            raise FinalRunnerContractError("selected frozen comparator method differs")
        if self.comparator_configuration is not None and self.action != "execute":
            raise FinalRunnerContractError(
                f"selected comparator {self.method_id} must be executable"
            )
        if (
            self.comparator_nonexecution_identity is not None
            and self.action != "not_applicable"
        ):
            raise FinalRunnerContractError(
                f"comparator {self.method_id} nonexecution authority must not execute"
            )
        if self.action == "execute" and self.reason is not None:
            raise FinalRunnerContractError(
                "executable frozen method authority has a reason"
            )
        if self.action == "not_applicable" and (
            not isinstance(self.reason, str) or not self.reason
        ):
            raise FinalRunnerContractError(
                "nonexecution frozen method authority lacks a reason"
            )
        if not self.seeds or any(
            seed is not None and (type(seed) is not int or seed < 0)
            for seed in self.seeds
        ):
            raise FinalRunnerContractError("frozen method seed denominator is invalid")

    @property
    def configuration_id(self) -> str:
        if self.legacy_configuration is not None:
            return self.legacy_configuration.configuration_id
        if self.comparator_configuration is not None:
            return self.comparator_configuration.configuration.configuration_id
        return f"nonexecution-{self.method_id}"

    @property
    def kind(self) -> str:
        if self.legacy_configuration is not None:
            return self.legacy_configuration.kind
        if self.comparator_configuration is not None:
            return "comparator_tuning"
        return "comparator_nonexecution"

    @property
    def requires_count_score(self) -> bool:
        return bool(
            self.legacy_configuration is not None
            and self.legacy_configuration.requires_count_score
        )

    @property
    def requires_calibration(self) -> bool:
        return bool(
            self.legacy_configuration is not None
            and self.legacy_configuration.requires_calibration
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method_id": self.method_id,
            "legacy_configuration": (
                None
                if self.legacy_configuration is None
                else self.legacy_configuration.to_dict()
            ),
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
            "action": self.action,
            "reason": self.reason,
            "seeds": list(self.seeds),
        }


@dataclass(frozen=True, slots=True)
class FinalExecutionPlan:
    """Hash-bound frozen method x final dataset x nested seed denominator."""

    schema_version: int
    input_hashes: Mapping[str, str]
    entries: tuple[FinalPlanEntry, ...]
    configurations: tuple[FrozenPlanMethodAuthority, ...]
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryExecutionPlan:
    """Exact one-dataset supplementary trajectory execution denominator."""

    schema_version: int
    scope: Literal["supplementary_trajectory"]
    input_hashes: Mapping[str, str]
    entries: tuple[FinalPlanEntry, ...]
    configurations: tuple[FrozenPlanMethodAuthority, ...]
    plan_sha256: str


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
        raise FinalRunnerContractError("final result is not canonical JSON") from error


def _read_unique_file(
    path: Path,
    name: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError("max_bytes must be a nonnegative integer or None")
    descriptor = -1
    try:
        metadata = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stable_file_identity(opened) != _stable_file_identity(metadata)
        ):
            raise FinalRunnerContractError(f"{name} is not a unique regular file")
        if max_bytes is not None and opened.st_size > max_bytes:
            raise FinalRunnerContractError(f"{name} exceeds its size bound")
        chunks: list[bytes] = []
        read_bytes = 0
        while True:
            read_size = 1024 * 1024
            if max_bytes is not None:
                read_size = min(read_size, max_bytes + 1 - read_bytes)
                read_size = max(1, read_size)
            chunk = os.read(descriptor, read_size)
            if not chunk:
                break
            chunks.append(chunk)
            read_bytes += len(chunk)
            if max_bytes is not None and read_bytes > max_bytes:
                raise FinalRunnerContractError(f"{name} exceeds its size bound")
        after = os.fstat(descriptor)
        named_after = path.lstat()
        identity = _stable_file_identity(opened)
        if identity != _stable_file_identity(after) or identity != (
            _stable_file_identity(named_after)
        ):
            raise FinalRunnerContractError(f"{name} changed while being read")
        return b"".join(chunks)
    except FinalRunnerContractError:
        raise
    except OSError as error:
        raise FinalRunnerContractError(f"cannot read {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _strict_json(raw: bytes, name: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FinalRunnerContractError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise FinalRunnerContractError(f"{name} contains non-finite {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except FinalRunnerContractError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise FinalRunnerContractError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise FinalRunnerContractError(f"{name} must be a JSON object")
    return value


def _canonical_round(repository: Path, round_dir: Path) -> tuple[Path, Path]:
    if not isinstance(repository, Path) or not isinstance(round_dir, Path):
        raise TypeError("repository and round_dir must be pathlib.Path values")
    try:
        selected_repository = repository.resolve(strict=True)
        destination = round_dir.resolve(strict=True)
        destination.relative_to(selected_repository)
    except ValueError as error:
        raise FinalRunnerContractError(
            "final round must be inside repository"
        ) from error
    except OSError as error:
        raise FinalRunnerContractError(
            "final repository or round is unavailable"
        ) from error
    if destination != round_dir.absolute() or not destination.is_dir():
        raise FinalRunnerContractError("final round path is not canonical")
    return selected_repository, destination


def _validate_final_runtime_lock(
    frozen_method: Mapping[str, object],
    environments: ExecutionEnvironmentRegistry,
) -> str:
    """Bind the observed executable closure to the exact frozen runtime lock."""

    if not isinstance(frozen_method, Mapping):
        raise TypeError("frozen_method must be a mapping")
    if not isinstance(environments, ExecutionEnvironmentRegistry):
        raise TypeError("environments must be an ExecutionEnvironmentRegistry")
    expected = _sha256(frozen_method.get("runtime_lock_sha256"), "frozen runtime lock")
    if environments.runtime_lock_sha256 != expected:
        raise FinalRunnerContractError(
            "final execution runtime lock differs from frozen method authority"
        )
    return expected


def _load_final_execution_environment_registry(
    repository: Path,
    registry: MethodRegistry,
) -> ExecutionEnvironmentRegistry:
    """Rebuild the exact executable/runtime registry used by final execution."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    selected_repository = repository.resolve(strict=True)
    return ExecutionEnvironmentRegistry.fixed(
        selected_repository,
        runtime_lock_path=(
            selected_repository / "environments/development-runtime.lock.json"
        ),
        benchmark_python=Path(sys.executable),
        r_library_paths={
            "saver": (selected_repository / "artifacts/envs/saver-r/library",)
        },
        lock_only_environment_ids=derive_lock_only_environment_ids(registry),
    )


def _validate_final_storage_capacity(
    plan: FinalExecutionPlan,
    round_dir: Path,
    *,
    completed_records: int = 0,
) -> dict[str, int | str]:
    """Require enough space for one bounded compressed common matrix per run."""

    if not isinstance(plan, FinalExecutionPlan):
        raise TypeError("plan must be a FinalExecutionPlan")
    if not isinstance(round_dir, Path):
        raise TypeError("round_dir must be a pathlib.Path")
    if (
        isinstance(completed_records, bool)
        or type(completed_records) is not int
        or not 0 <= completed_records <= len(plan.entries)
    ):
        raise ValueError("completed_records must be a valid plan prefix length")
    remaining_entries = plan.entries[completed_records:]
    remaining_executions = sum(entry.action == "execute" for entry in remaining_entries)
    remaining_p_pre_zero_executions = sum(
        entry.action == "execute" and entry.run.method_id == "maskimpute"
        for entry in remaining_entries
    )
    compress_bound = _zlib_compress_bound(_FINAL_MATRIX_UNCOMPRESSED_NBYTES)
    p_pre_zero_compress_bound = _prezero_zlib_compress_bound(
        _FINAL_PREZERO_MATRIX_UNCOMPRESSED_NBYTES
    )
    required = (
        remaining_executions * compress_bound
        + remaining_p_pre_zero_executions * p_pre_zero_compress_bound
        + len(remaining_entries) * _FINAL_RECORD_OVERHEAD_BYTES
        + _FINAL_STORAGE_RESERVE_BYTES
    )
    try:
        free = shutil.disk_usage(round_dir).free
    except OSError as error:
        raise FinalRunnerContractError(
            "final free storage cannot be measured"
        ) from error
    if free < required:
        raise FinalRunnerContractError(
            "final free storage is below the fail-closed compressed-output bound"
        )
    return {
        "schema": "maskimpute-final-storage-preflight-v1",
        "completed_record_count": completed_records,
        "remaining_entry_count": len(remaining_entries),
        "remaining_execution_count": remaining_executions,
        "remaining_p_pre_zero_execution_count": remaining_p_pre_zero_executions,
        "per_execution_compressed_bound_bytes": compress_bound,
        "per_p_pre_zero_compressed_bound_bytes": p_pre_zero_compress_bound,
        "required_free_bytes": required,
        "observed_free_bytes": free,
    }


def _execution_storage_component(
    plan: FinalExecutionPlan | TrajectoryExecutionPlan,
    *,
    completed_records: int,
    cells: int,
    genes: int,
) -> dict[str, int]:
    """Derive a reserve-free compressed-output bound for one execution scope."""

    if not isinstance(plan, (FinalExecutionPlan, TrajectoryExecutionPlan)):
        raise TypeError("plan must be a frozen execution plan")
    if (
        type(completed_records) is not int
        or not 0 <= completed_records <= len(plan.entries)
        or type(cells) is not int
        or cells <= 0
        or type(genes) is not int
        or genes <= 0
    ):
        raise ValueError("execution storage component inputs are invalid")
    remaining = plan.entries[completed_records:]
    execution_count = sum(entry.action == "execute" for entry in remaining)
    score_count = sum(
        entry.action == "execute" and entry.run.method_id == "maskimpute"
        for entry in remaining
    )
    matrix_nbytes = cells * genes * 8
    matrix_bound = _zlib_compress_bound(matrix_nbytes)
    score_bound = _prezero_zlib_compress_bound(matrix_nbytes)
    required = (
        execution_count * matrix_bound
        + score_count * score_bound
        + len(remaining) * _FINAL_RECORD_OVERHEAD_BYTES
    )
    return {
        "completed_record_count": completed_records,
        "remaining_entry_count": len(remaining),
        "remaining_execution_count": execution_count,
        "remaining_p_pre_zero_execution_count": score_count,
        "cells": cells,
        "genes": genes,
        "per_execution_compressed_bound_bytes": matrix_bound,
        "per_p_pre_zero_compressed_bound_bytes": score_bound,
        "required_free_bytes": required,
    }


def _validate_combined_storage_capacity(
    primary_plan: FinalExecutionPlan,
    trajectory_plan: TrajectoryExecutionPlan,
    scaling_authority: object,
    round_dir: Path,
    *,
    primary_completed_records: int = 0,
    trajectory_completed_records: int = 0,
) -> dict[str, object]:
    """Preflight all retained round outputs with one shared safety reserve."""

    if not isinstance(primary_plan, FinalExecutionPlan):
        raise TypeError("primary_plan must be a FinalExecutionPlan")
    if not isinstance(trajectory_plan, TrajectoryExecutionPlan):
        raise TypeError("trajectory_plan must be a TrajectoryExecutionPlan")
    if not isinstance(round_dir, Path):
        raise TypeError("round_dir must be a pathlib.Path")
    from .scaling import scaling_storage_preflight

    primary = _execution_storage_component(
        primary_plan,
        completed_records=primary_completed_records,
        cells=2_700,
        genes=1_200,
    )
    trajectory = _execution_storage_component(
        trajectory_plan,
        completed_records=trajectory_completed_records,
        cells=2_700,
        genes=120,
    )
    try:
        scaling = scaling_storage_preflight(scaling_authority)
    except Exception as error:
        raise FinalRunnerContractError(
            "supplementary scaling storage authority is invalid"
        ) from error
    scaling_required = scaling.get("required_free_bytes")
    if type(scaling_required) is not int or scaling_required < 0:
        raise FinalRunnerContractError("supplementary scaling storage bound is invalid")
    required = (
        primary["required_free_bytes"]
        + trajectory["required_free_bytes"]
        + scaling_required
        + _FINAL_STORAGE_RESERVE_BYTES
    )
    try:
        free = shutil.disk_usage(round_dir).free
    except OSError as error:
        raise FinalRunnerContractError(
            "combined final free storage cannot be measured"
        ) from error
    if free < required:
        raise FinalRunnerContractError(
            "final free storage is below the combined fail-closed output bound"
        )
    return {
        "schema": "maskimpute-combined-final-storage-preflight-v1",
        "primary": primary,
        "trajectory": trajectory,
        "scaling": dict(scaling),
        "reserve_bytes": _FINAL_STORAGE_RESERVE_BYTES,
        "required_free_bytes": required,
        "observed_free_bytes": free,
    }


@dataclass(frozen=True, slots=True)
class _DerivedFinalExecutionAuthority:
    """Deterministic primary authority bytes before immutable publication."""

    context: ExecutionAuthorityContext
    calibration_relative_path: str
    calibration_raw: bytes
    score_relative_path: str
    score_raw: bytes
    authority_relative_path: str
    authority_raw: bytes


def _derive_final_execution_authority(
    repository: Path,
    round_dir: Path,
    frozen_method: Mapping[str, object],
    *,
    execution_claim_sha256: str,
    execution_environment_sha256: str,
    dataset_manifest_sha256: str,
) -> _DerivedFinalExecutionAuthority:
    """Purely derive the exact primary execution authority and immutable bytes."""

    from maskimpute.calibration import CalibrationArtifact

    selected_repository, destination = _canonical_round(repository, round_dir)
    if not isinstance(frozen_method, Mapping):
        raise TypeError("frozen_method must be a mapping")
    unsigned_frozen = {
        key: value for key, value in frozen_method.items() if key != "payload_sha256"
    }
    frozen_sha256 = frozen_method.get("payload_sha256")
    if frozen_sha256 != canonical_sha256(unsigned_frozen):
        raise FinalRunnerContractError("frozen method receipt is invalid")
    _sha256(frozen_sha256, "frozen method")
    claim_sha256 = _sha256(execution_claim_sha256, "execution claim")
    environment_sha256 = _sha256(execution_environment_sha256, "execution environment")
    manifest_sha256 = _sha256(dataset_manifest_sha256, "dataset manifest")
    runtime_lock_sha256 = _sha256(
        frozen_method.get("runtime_lock_sha256"), "frozen runtime lock"
    )

    artifact_bindings = frozen_method.get("artifact_bindings")
    if not isinstance(artifact_bindings, Mapping):
        raise FinalRunnerContractError("frozen artifact bindings are unavailable")
    selection_binding = artifact_bindings.get("selection_contract")
    if (
        not isinstance(selection_binding, Mapping)
        or selection_binding.get("path") != "study/selection_contract.json"
    ):
        raise FinalRunnerContractError("selection contract binding is invalid")
    selection_path = selected_repository / "study/selection_contract.json"
    selection_raw = _read_unique_file(selection_path, "selection contract")
    selection_file_sha256 = hashlib.sha256(selection_raw).hexdigest()
    if selection_file_sha256 != _sha256(
        selection_binding.get("sha256"), "selection contract file"
    ):
        raise FinalRunnerContractError("selection contract file checksum differs")
    selection = _strict_json(selection_raw, "selection contract")
    count_config = selection.get("count_model_config")
    count_config_sha256 = selection.get("count_model_config_sha256")
    if (
        not isinstance(count_config, Mapping)
        or canonical_sha256(count_config) != count_config_sha256
    ):
        raise FinalRunnerContractError("count-model configuration binding differs")
    _sha256(count_config_sha256, "count-model configuration")

    selected_configuration = frozen_method.get("selected_configuration")
    if (
        not isinstance(selected_configuration, Mapping)
        or canonical_sha256(selected_configuration)
        != _sha256(
            frozen_method.get("selected_configuration_sha256"),
            "selected final configuration",
        )
        or selected_configuration.get("method_version")
        != frozen_method.get("selected_version")
    ):
        raise FinalRunnerContractError("selected final configuration is invalid")
    base_configuration = selected_configuration.get("hyperparameters")
    if not isinstance(base_configuration, Mapping):
        raise FinalRunnerContractError("selected final hyperparameters are invalid")
    base_configuration_sha256 = canonical_sha256(base_configuration)

    calibrator_summary = frozen_method.get("selected_calibrator")
    if not isinstance(calibrator_summary, Mapping):
        raise FinalRunnerContractError("selected final calibrator is invalid")
    score_policy = selected_configuration.get("score_policy")
    expected_usage = (
        "retained_all_development_calibrator"
        if score_policy == "retained_development_calibrator"
        else "direct_count_score"
    )
    if (
        score_policy
        not in {"retained_development_calibrator", "direct_cross_fitted_count_score"}
        or calibrator_summary.get("score_policy") != score_policy
        or calibrator_summary.get("final_usage") != expected_usage
    ):
        raise FinalRunnerContractError("selected final score policy is invalid")
    calibration_payload = calibrator_summary.get("artifact")
    if not isinstance(calibration_payload, Mapping):
        raise FinalRunnerContractError("embedded calibration artifact is unavailable")
    try:
        calibration_artifact = CalibrationArtifact(dict(calibration_payload))
    except (TypeError, ValueError) as error:
        raise FinalRunnerContractError(
            "embedded calibration artifact is not executable"
        ) from error
    if calibration_artifact.to_dict() != calibration_payload:
        raise FinalRunnerContractError("embedded calibration artifact differs")

    calibration_raw = _canonical_bytes(calibration_payload) + b"\n"
    expected_calibration_file_sha256 = _sha256(
        calibrator_summary.get("artifact_file_sha256"),
        "frozen calibration artifact file",
    )
    if hashlib.sha256(calibration_raw).hexdigest() != expected_calibration_file_sha256:
        raise FinalRunnerContractError(
            "materialized calibration bytes differ from frozen evidence"
        )

    calibration_relative = "final/execution_authority/retained_calibration.json"
    calibration_file_sha256 = hashlib.sha256(calibration_raw).hexdigest()

    score_body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "maskimpute_final_count_score_authority",
        "status": "ready",
        "scope": "truth_free_final_inference",
        "frozen_method_sha256": frozen_sha256,
        "execution_claim_sha256": claim_sha256,
        "execution_environment_sha256": environment_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "selection_contract_file_sha256": selection_file_sha256,
        "count_model_config": dict(count_config),
        "count_model_config_sha256": count_config_sha256,
    }
    score_payload = {
        **score_body,
        "payload_sha256": canonical_sha256(score_body),
    }
    score_relative = "final/execution_authority/count_score_authority.json"
    score_raw = _canonical_bytes(score_payload) + b"\n"
    score_file_sha256 = hashlib.sha256(score_raw).hexdigest()
    calibration_repo_path = (
        (destination / "results" / calibration_relative)
        .relative_to(selected_repository)
        .as_posix()
    )
    score_repo_path = (
        (destination / "results" / score_relative)
        .relative_to(selected_repository)
        .as_posix()
    )
    authority_body: dict[str, object] = {
        "schema_version": 1,
        "authority_type": "maskimpute_frozen_final_execution",
        "frozen_method_sha256": frozen_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "execution_claim_sha256": claim_sha256,
        "execution_environment_sha256": environment_sha256,
        "dataset_manifest_sha256": manifest_sha256,
        "base_configuration": dict(base_configuration),
        "base_configuration_sha256": base_configuration_sha256,
        "count_model_config": dict(count_config),
        "count_model_config_sha256": count_config_sha256,
        "count_score_authority_path": score_repo_path,
        "count_score_authority_sha256": score_file_sha256,
        "retained_calibration_path": calibration_repo_path,
        "retained_calibration_sha256": calibration_file_sha256,
        "calibration_usage": expected_usage,
    }
    authority_sha256 = canonical_sha256(authority_body)
    authority_payload = {
        **authority_body,
        "authority_sha256": authority_sha256,
    }
    authority_relative = "final/execution_authority/authority.json"
    return _DerivedFinalExecutionAuthority(
        context=ExecutionAuthorityContext(
            authority_sha256=authority_sha256,
            base_configuration_json=_canonical_bytes(dict(base_configuration)).decode(),
            base_configuration_sha256=base_configuration_sha256,
            count_model_config_json=_canonical_bytes(dict(count_config)).decode(),
            count_model_config_sha256=str(count_config_sha256),
            count_score_manifest_path=score_repo_path,
            count_score_manifest_sha256=score_file_sha256,
            retained_calibration_path=calibration_repo_path,
            retained_calibration_sha256=calibration_file_sha256,
        ),
        calibration_relative_path=calibration_relative,
        calibration_raw=calibration_raw,
        score_relative_path=score_relative,
        score_raw=score_raw,
        authority_relative_path=authority_relative,
        authority_raw=_canonical_bytes(authority_payload) + b"\n",
    )


def materialize_final_execution_authority(
    repository: Path,
    round_dir: Path,
    frozen_method: Mapping[str, object],
    *,
    execution_claim_sha256: str,
    execution_environment_sha256: str,
    dataset_manifest_sha256: str,
) -> ExecutionAuthorityContext:
    """Publish the exact embedded calibrator and truth-free final score authority."""

    selected_repository, destination = _canonical_round(repository, round_dir)
    derived = _derive_final_execution_authority(
        selected_repository,
        destination,
        frozen_method,
        execution_claim_sha256=execution_claim_sha256,
        execution_environment_sha256=execution_environment_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    results_store = CheckpointStore(destination / "results")
    for relative_path, raw, expected_sha256 in (
        (
            derived.calibration_relative_path,
            derived.calibration_raw,
            derived.context.retained_calibration_sha256,
        ),
        (
            derived.score_relative_path,
            derived.score_raw,
            derived.context.count_score_manifest_sha256,
        ),
        (
            derived.authority_relative_path,
            derived.authority_raw,
            hashlib.sha256(derived.authority_raw).hexdigest(),
        ),
    ):
        observed_relative, observed_sha256 = results_store._publish_immutable(
            relative_path,
            raw,
        )
        if observed_relative != relative_path or observed_sha256 != expected_sha256:
            raise FinalRunnerContractError(
                "materialized final execution authority bytes differ"
            )
    return derived.context


def _read_repository_authority_file(
    repository: Path,
    relative_value: str,
    expected_sha256: str | None,
    name: str,
) -> bytes:
    """Read one context-bound authority file without following path aliases."""

    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise FinalRunnerContractError(f"{name} path is unsafe")
    path = repository.joinpath(*relative.parts)
    try:
        if path.resolve(strict=True) != path.absolute():
            raise FinalRunnerContractError(f"{name} path contains a symlink")
    except OSError as error:
        raise FinalRunnerContractError(f"{name} is unavailable") from error
    raw = _read_unique_file(path, name, max_bytes=32 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != _sha256(expected_sha256, f"{name} file"):
        raise FinalRunnerContractError(f"{name} file checksum differs")
    return raw


def materialize_trajectory_execution_authority(
    repository: Path,
    round_dir: Path,
    frozen_method: Mapping[str, object],
    registered: TrajectoryPreparedDataset,
    primary_authority: ExecutionAuthorityContext,
    *,
    execution_claim_sha256: str,
    execution_environment_sha256: str,
    primary_final_plan_sha256: str,
) -> ExecutionAuthorityContext:
    """Publish a distinct authority for the one registered trajectory input."""

    selected_repository, destination = _canonical_round(repository, round_dir)
    if not isinstance(frozen_method, Mapping):
        raise TypeError("frozen_method must be a mapping")
    if not isinstance(registered, TrajectoryPreparedDataset):
        raise TypeError("registered must be a TrajectoryPreparedDataset")
    if not isinstance(primary_authority, ExecutionAuthorityContext):
        raise TypeError("primary_authority must be an ExecutionAuthorityContext")
    unsigned_frozen = {
        key: value for key, value in frozen_method.items() if key != "payload_sha256"
    }
    frozen_sha256 = frozen_method.get("payload_sha256")
    if frozen_sha256 != canonical_sha256(unsigned_frozen):
        raise FinalRunnerContractError("frozen method receipt is invalid")
    _sha256(frozen_sha256, "frozen method")
    runtime_lock_sha256 = _sha256(
        frozen_method.get("runtime_lock_sha256"), "frozen runtime lock"
    )
    claim_sha256 = _sha256(execution_claim_sha256, "execution claim")
    environment_sha256 = _sha256(execution_environment_sha256, "execution environment")
    primary_plan_sha256 = _sha256(primary_final_plan_sha256, "primary final plan")

    binding = registered.binding
    prepared = registered.prepared
    if not isinstance(binding, RegisteredTrajectoryBinding) or not isinstance(
        prepared, PreparedDataset
    ):
        raise FinalRunnerContractError("registered trajectory dataset is invalid")
    expected_binding = _trajectory_binding(
        registered.authority,
        authority_file_sha256=binding.authority_file_sha256,
        dataset_file_sha256=binding.dataset_file_sha256,
    )
    expected_receipt = _trajectory_dataset_receipt(binding, prepared)
    if (
        binding != expected_binding
        or prepared.binding != binding
        or dict(registered.receipt) != expected_receipt
        or registered.receipt_file_sha256
        != hashlib.sha256(_canonical_bytes(expected_receipt) + b"\n").hexdigest()
        or registered.receipt_file_path
        != "results/trajectory/dataset/dataset_receipt.json"
    ):
        raise FinalRunnerContractError("registered trajectory identity differs")

    selected_configuration = frozen_method.get("selected_configuration")
    if not isinstance(selected_configuration, Mapping):
        raise FinalRunnerContractError("selected final configuration is invalid")
    base_configuration = selected_configuration.get("hyperparameters")
    try:
        primary_base = json.loads(primary_authority.base_configuration_json)
        count_config = json.loads(primary_authority.count_model_config_json)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FinalRunnerContractError(
            "primary execution authority is invalid"
        ) from error
    if (
        not isinstance(base_configuration, Mapping)
        or dict(base_configuration) != primary_base
        or canonical_sha256(primary_base) != primary_authority.base_configuration_sha256
        or not isinstance(count_config, dict)
        or canonical_sha256(count_config) != primary_authority.count_model_config_sha256
    ):
        raise FinalRunnerContractError(
            "trajectory configuration differs from primary frozen authority"
        )

    calibration_raw = _read_repository_authority_file(
        selected_repository,
        primary_authority.retained_calibration_path,
        primary_authority.retained_calibration_sha256,
        "primary retained calibration authority",
    )
    _read_repository_authority_file(
        selected_repository,
        primary_authority.count_score_manifest_path,
        primary_authority.count_score_manifest_sha256,
        "primary count-score authority",
    )
    results_store = CheckpointStore(destination / "results")
    calibration_relative, calibration_file_sha256 = results_store._publish_immutable(
        "trajectory/execution_authority/retained_calibration.json",
        calibration_raw,
    )
    calibration_repo_path = (
        (destination / "results" / calibration_relative)
        .relative_to(selected_repository)
        .as_posix()
    )
    method_input_digest = method_input_sha256(prepared.method_input)
    score_body: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "maskimpute_trajectory_count_score_authority",
        "status": "ready",
        "scope": "truth_free_registered_trajectory_inference",
        "frozen_method_sha256": frozen_sha256,
        "primary_final_plan_sha256": primary_plan_sha256,
        "primary_execution_authority_sha256": primary_authority.authority_sha256,
        "primary_count_score_authority_file_sha256": (
            primary_authority.count_score_manifest_sha256
        ),
        "execution_claim_sha256": claim_sha256,
        "execution_environment_sha256": environment_sha256,
        "trajectory_authority_file_sha256": binding.authority_file_sha256,
        "trajectory_authority_sha256": binding.authority_sha256,
        "trajectory_binding_sha256": binding.registered_binding_sha256,
        "trajectory_dataset_sha256": binding.dataset_sha256,
        "trajectory_dataset_file_sha256": binding.dataset_file_sha256,
        "trajectory_dataset_receipt_sha256": registered.receipt["receipt_sha256"],
        "trajectory_dataset_receipt_file_sha256": registered.receipt_file_sha256,
        "trajectory_method_input_sha256": method_input_digest,
        "trajectory_retained_cell_ids_sha256": (
            prepared.audit.retained_cell_ids_sha256
        ),
        "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
        "count_model_config": count_config,
        "count_model_config_sha256": primary_authority.count_model_config_sha256,
        "retained_calibration_file_sha256": calibration_file_sha256,
    }
    score_payload = {
        **score_body,
        "payload_sha256": canonical_sha256(score_body),
    }
    score_relative, score_file_sha256 = results_store._publish_immutable(
        "trajectory/execution_authority/count_score_authority.json",
        _canonical_bytes(score_payload) + b"\n",
    )
    score_repo_path = (
        (destination / "results" / score_relative)
        .relative_to(selected_repository)
        .as_posix()
    )
    authority_body: dict[str, object] = {
        "schema_version": 1,
        "authority_type": "maskimpute_frozen_trajectory_execution",
        "scope": "supplementary_trajectory",
        "frozen_method_sha256": frozen_sha256,
        "runtime_lock_sha256": runtime_lock_sha256,
        "primary_final_plan_sha256": primary_plan_sha256,
        "primary_execution_authority_sha256": primary_authority.authority_sha256,
        "execution_claim_sha256": claim_sha256,
        "execution_environment_sha256": environment_sha256,
        "trajectory_authority_sha256": binding.authority_sha256,
        "trajectory_binding_sha256": binding.registered_binding_sha256,
        "trajectory_dataset_sha256": binding.dataset_sha256,
        "trajectory_method_input_sha256": method_input_digest,
        "base_configuration": primary_base,
        "base_configuration_sha256": primary_authority.base_configuration_sha256,
        "count_model_config": count_config,
        "count_model_config_sha256": primary_authority.count_model_config_sha256,
        "count_score_authority_path": score_repo_path,
        "count_score_authority_sha256": score_file_sha256,
        "retained_calibration_path": calibration_repo_path,
        "retained_calibration_sha256": calibration_file_sha256,
        "calibration_usage": "retained_all_development_calibrator",
    }
    authority_sha256 = canonical_sha256(authority_body)
    results_store._publish_immutable(
        "trajectory/execution_authority/authority.json",
        _canonical_bytes({**authority_body, "authority_sha256": authority_sha256})
        + b"\n",
    )
    return ExecutionAuthorityContext(
        authority_sha256=authority_sha256,
        base_configuration_json=_canonical_bytes(primary_base).decode(),
        base_configuration_sha256=primary_authority.base_configuration_sha256,
        count_model_config_json=_canonical_bytes(count_config).decode(),
        count_model_config_sha256=primary_authority.count_model_config_sha256,
        count_score_manifest_path=score_repo_path,
        count_score_manifest_sha256=score_file_sha256,
        retained_calibration_path=calibration_repo_path,
        retained_calibration_sha256=calibration_file_sha256,
    )


def _hash_unique_file(path: Path, name: str) -> str:
    descriptor = -1
    try:
        named_before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or _stable_file_identity(opened_before)
            != _stable_file_identity(named_before)
        ):
            raise FinalRunnerContractError(f"{name} is not a unique regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if _stable_file_identity(opened_before) != _stable_file_identity(
            os.fstat(descriptor)
        ) or _stable_file_identity(opened_before) != _stable_file_identity(
            path.lstat()
        ):
            raise FinalRunnerContractError(f"{name} changed while being hashed")
        return digest.hexdigest()
    except FinalRunnerContractError:
        raise
    except OSError as error:
        raise FinalRunnerContractError(f"cannot hash {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def final_result_file_manifest(round_dir: Path) -> dict[str, object]:
    """Hash every unique regular result without accepting aliases or symlinks."""

    if not isinstance(round_dir, Path):
        raise TypeError("round_dir must be a pathlib.Path")
    try:
        destination = round_dir.resolve(strict=True)
    except OSError as error:
        raise FinalRunnerContractError("final round is unavailable") from error
    if destination != round_dir.absolute() or not destination.is_dir():
        raise FinalRunnerContractError("final round path is not canonical")
    results = destination / "results"
    try:
        if results.resolve(strict=True) != results.absolute():
            raise FinalRunnerContractError("final result path contains a symlink")
        entries = sorted(results.rglob("*"))
    except OSError as error:
        raise FinalRunnerContractError("final result path is unavailable") from error
    files: list[dict[str, str]] = []
    for path in entries:
        try:
            metadata = path.lstat()
        except OSError as error:
            raise FinalRunnerContractError("final result path changed") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise FinalRunnerContractError("final result path contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise FinalRunnerContractError("final result contains a special file")
        files.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "sha256": _hash_unique_file(path, "final result file"),
            }
        )
    return {"result_files": files}


def _scaling_checkpoint_file_bindings(round_dir: Path) -> dict[str, str]:
    """Derive every scaling result path and digest from its closed checkpoint."""

    checkpoint_root = round_dir / "results/scaling/checkpoints"
    if not os.path.lexists(checkpoint_root):
        return {}
    try:
        root_metadata = checkpoint_root.lstat()
        checkpoint_paths = tuple(
            sorted(checkpoint_root.iterdir(), key=lambda path: path.name)
        )
    except OSError as error:
        raise FinalRunnerContractError(
            "supplementary scaling checkpoint history is unavailable"
        ) from error
    expected_names = tuple(
        f"{index:08d}.json" for index in range(1, len(checkpoint_paths) + 1)
    )
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or not checkpoint_paths
        or tuple(path.name for path in checkpoint_paths) != expected_names
    ):
        raise FinalRunnerContractError(
            "supplementary scaling checkpoint history is not canonical"
        )
    bindings: dict[str, str] = {}
    previous_datasets: list[object] = []
    previous_records: list[object] = []
    expected_plan: object = None
    expected_inputs: object = None
    expected_planned: object = None
    checkpoint: dict[str, object] | None = None
    raw = b""
    for sequence, checkpoint_path in enumerate(checkpoint_paths, start=1):
        raw = _read_unique_file(
            checkpoint_path,
            f"supplementary scaling checkpoint {sequence}",
            max_bytes=32 * 1024 * 1024,
        )
        checkpoint = _strict_json(raw, "supplementary scaling checkpoint")
        if (
            set(checkpoint)
            != {
                "schema_version",
                "plan_sha256",
                "input_hashes",
                "planned_run_count",
                "status",
                "datasets",
                "records",
                "checkpoint_sha256",
            }
            or raw != _canonical_bytes(checkpoint) + b"\n"
        ):
            raise FinalRunnerContractError(
                "supplementary scaling checkpoint is not canonical"
            )
        body = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_sha256"
        }
        input_hashes = checkpoint.get("input_hashes")
        datasets = checkpoint.get("datasets")
        records = checkpoint.get("records")
        planned = checkpoint.get("planned_run_count")
        if (
            checkpoint.get("schema_version") != 1
            or _sha256(checkpoint.get("plan_sha256"), "scaling plan")
            != checkpoint.get("plan_sha256")
            or not isinstance(input_hashes, Mapping)
            or not input_hashes
            or any(
                not isinstance(name, str)
                or _sha256(value, f"scaling input {name}") != value
                for name, value in input_hashes.items()
            )
            or type(planned) is not int
            or planned <= 0
            or not isinstance(datasets, list)
            or len(datasets) > 4
            or not isinstance(records, list)
            or len(records) > planned
            or len(datasets) + len(records) != sequence
            or datasets[: len(previous_datasets)] != previous_datasets
            or records[: len(previous_records)] != previous_records
            or checkpoint.get("checkpoint_sha256") != canonical_sha256(body)
        ):
            raise FinalRunnerContractError(
                "supplementary scaling checkpoint binding is invalid"
            )
        if sequence == 1:
            expected_plan = checkpoint.get("plan_sha256")
            expected_inputs = input_hashes
            expected_planned = planned
        elif (
            checkpoint.get("plan_sha256") != expected_plan
            or input_hashes != expected_inputs
            or planned != expected_planned
        ):
            raise FinalRunnerContractError(
                "supplementary scaling checkpoint authority changed"
            )
        expected_status = (
            "completed" if datasets and len(records) == planned else "running"
        )
        if checkpoint.get("status") != expected_status:
            raise FinalRunnerContractError(
                "supplementary scaling checkpoint status is invalid"
            )
        previous_datasets = datasets
        previous_records = records
        bindings[f"results/scaling/checkpoints/{checkpoint_path.name}"] = (
            hashlib.sha256(raw).hexdigest()
        )
    assert checkpoint is not None
    datasets = previous_datasets
    records = previous_records

    def add(
        relative_value: object,
        digest_value: object,
        name: str,
        *,
        expected: str | None = None,
    ) -> None:
        if not isinstance(relative_value, str):
            raise FinalRunnerContractError(f"{name} path is invalid")
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or (expected is not None and relative.as_posix() != expected)
        ):
            raise FinalRunnerContractError(f"{name} path is unsafe")
        path = f"results/scaling/{relative.as_posix()}"
        digest = _sha256(digest_value, f"{name} file")
        if path in bindings:
            raise FinalRunnerContractError(f"{name} path is duplicated")
        bindings[path] = digest

    observed_cells: list[int] = []
    for index, receipt in enumerate(datasets):
        if not isinstance(receipt, Mapping):
            raise FinalRunnerContractError(
                "supplementary scaling dataset receipt is invalid"
            )
        cells = receipt.get("cells")
        if (
            type(cells) is not int
            or cells not in {10_000, 25_000, 50_000, 100_000}
            or cells in observed_cells
            or type(receipt.get("moderate_output_size_bytes")) is not int
            or receipt["moderate_output_size_bytes"] <= 0
        ):
            raise FinalRunnerContractError(
                "supplementary scaling dataset receipt is invalid"
            )
        observed_cells.append(cells)
        add(
            receipt.get("moderate_output_path"),
            receipt.get("moderate_output_file_sha256"),
            f"supplementary scaling dataset {index}",
            expected=f"generated/scaling-{cells}/dataset/moderate.h5ad",
        )

    seen_runs: set[str] = set()
    for index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != {"run", "metrics", "record_sha256"}
            or not isinstance(record.get("run"), Mapping)
            or not isinstance(record.get("metrics"), list)
        ):
            raise FinalRunnerContractError("supplementary scaling record is invalid")
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        if record.get("record_sha256") != canonical_sha256(unsigned):
            raise FinalRunnerContractError(
                "supplementary scaling record binding is invalid"
            )
        run = record["run"]
        run_id = run.get("run_id")
        if (
            not isinstance(run_id, str)
            or _SAFE_ID.fullmatch(run_id) is None
            or not run_id.startswith("scaling-")
            or run_id in seen_runs
        ):
            raise FinalRunnerContractError(
                "supplementary scaling run identity is invalid"
            )
        seen_runs.add(run_id)
        base = f"runs/{run_id}"
        for prefix, filename in (
            ("stdout", "run.stdout"),
            ("stderr", "run.stderr"),
            ("executor_receipt", "run.executor-receipt.json"),
        ):
            add(
                run.get(f"{prefix}_path"),
                run.get(f"{prefix}_file_sha256"),
                f"supplementary scaling record {index} {prefix}",
                expected=f"{base}/{filename}",
            )
        for prefix, filename in (
            ("native_output", "run.native-f64.zlib"),
            ("evaluator_output", "run.log2-cp10k-f64.zlib"),
        ):
            relative = run.get(f"{prefix}_path")
            digest = run.get(f"{prefix}_file_sha256")
            if relative is None and digest is None:
                continue
            add(
                relative,
                digest,
                f"supplementary scaling record {index} {prefix}",
                expected=f"{base}/{filename}",
            )
    return bindings


def _owned_final_result_paths(round_dir: Path) -> frozenset[str]:
    """Derive exact dataset/final-runner paths; never bless arbitrary files."""

    destination = round_dir.resolve(strict=True)
    results = destination / "results"
    owned: set[str] = set()

    def add_result_relative(value: object, name: str) -> None:
        if not isinstance(value, str):
            raise FinalRunnerContractError(f"{name} path is invalid")
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise FinalRunnerContractError(f"{name} path is unsafe")
        owned.add(f"results/{relative.as_posix()}")

    status_path = results / "dataset_status.json"
    if os.path.lexists(status_path):
        raw = _read_unique_file(status_path, "final dataset status")
        status = _strict_json(raw, "final dataset status")
        if raw != _canonical_bytes(status) + b"\n":
            raise FinalRunnerContractError("final dataset status is not canonical")
        bindings = validate_final_manifest_payload(status)
        rows = status["rows"]
        assert isinstance(rows, list)
        owned.add("results/dataset_status.json")
        for binding, row in zip(bindings, rows, strict=True):
            assert isinstance(row, Mapping)
            add_result_relative(row.get("output_path"), "final dataset output")
            expected_log = f"logs/{binding.mechanism}/{binding.biological_id}.json"
            if row.get("log_path") != expected_log:
                raise FinalRunnerContractError("final dataset log path differs")
            add_result_relative(expected_log, "final dataset log")
            native_files = row.get("native_files")
            if not isinstance(native_files, list):
                raise FinalRunnerContractError(
                    "final dataset native file receipt is invalid"
                )
            for item in native_files:
                if not isinstance(item, Mapping):
                    raise FinalRunnerContractError(
                        "final dataset native file receipt is invalid"
                    )
                _sha256(item.get("sha256"), "final dataset native file")
                add_result_relative(item.get("path"), "final dataset native file")
            add_result_relative(
                f"receipts/{binding.mechanism}/{binding.biological_id}.json",
                "final dataset pair receipt",
            )

    authority_root = results / "final/execution_authority"
    for name in (
        "retained_calibration.json",
        "count_score_authority.json",
        "authority.json",
    ):
        path = authority_root / name
        if os.path.lexists(path):
            owned.add(f"results/final/execution_authority/{name}")

    execution_root = results / "final/execution"
    records_root = execution_root / "records"
    if os.path.lexists(records_root):
        try:
            metadata = records_root.lstat()
            names = tuple(sorted(path.name for path in records_root.iterdir()))
        except OSError as error:
            raise FinalRunnerContractError(
                "final execution record paths cannot be enumerated"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise FinalRunnerContractError(
                "final execution record path is not a directory"
            )
        if list(names) != [
            f"{ordinal:08d}.json" for ordinal in range(1, len(names) + 1)
        ]:
            raise FinalRunnerContractError(
                "final execution records are not a canonical prefix"
            )
        for name in names:
            record_path = records_root / name
            raw = _read_unique_file(record_path, "final execution record")
            record = _strict_json(raw, "final execution record")
            if raw != _canonical_bytes(record) + b"\n":
                raise FinalRunnerContractError(
                    "final execution record is not canonical"
                )
            run = record.get("run")
            if set(record) != {
                "run",
                "metrics",
                "p_pre_zero_evidence",
                "execution_request",
            } or not isinstance(run, Mapping):
                raise FinalRunnerContractError("final execution record is invalid")
            owned.add(f"results/final/execution/records/{name}")
            for prefix in ("stdout", "stderr", "native_output", "evaluator_output"):
                relative = run.get(f"{prefix}_path")
                if relative is None:
                    continue
                _sha256(
                    run.get(f"{prefix}_file_sha256"),
                    f"final execution {prefix} file",
                )
                safe = PurePosixPath(str(relative))
                if (
                    not isinstance(relative, str)
                    or safe.is_absolute()
                    or ".." in safe.parts
                    or not safe.parts
                    or safe.parts[0] != "runs"
                ):
                    raise FinalRunnerContractError(
                        f"final execution {prefix} path is unsafe"
                    )
                owned.add(f"results/final/execution/{safe.as_posix()}")
            score_evidence = record.get("p_pre_zero_evidence")
            if not isinstance(score_evidence, Mapping):
                raise FinalRunnerContractError(
                    "final p_pre_zero evidence record is invalid"
                )
            score_storage = score_evidence.get("storage")
            if not isinstance(score_storage, Mapping):
                raise FinalRunnerContractError(
                    "final p_pre_zero storage receipt is invalid"
                )
            score_relative = score_storage.get("path")
            if score_relative is not None:
                _sha256(
                    score_storage.get("compressed_sha256"),
                    "final p_pre_zero compressed file",
                )
                safe_score = PurePosixPath(str(score_relative))
                if (
                    not isinstance(score_relative, str)
                    or safe_score.is_absolute()
                    or ".." in safe_score.parts
                    or not safe_score.parts
                    or safe_score.parts[0] != "runs"
                ):
                    raise FinalRunnerContractError("final p_pre_zero path is unsafe")
                owned.add(f"results/final/execution/{safe_score.as_posix()}")
    execution_manifest = execution_root / "execution_manifest.json"
    if os.path.lexists(execution_manifest):
        owned.add("results/final/execution/execution_manifest.json")

    trajectory_dataset_root = results / "trajectory/dataset"
    trajectory_dataset = trajectory_dataset_root / "evaluator.h5ad"
    trajectory_receipt_path = trajectory_dataset_root / "dataset_receipt.json"
    if os.path.lexists(trajectory_dataset) or os.path.lexists(trajectory_receipt_path):
        if not (
            os.path.lexists(trajectory_dataset)
            and os.path.lexists(trajectory_receipt_path)
        ):
            raise FinalRunnerContractError(
                "registered trajectory dataset receipt is incomplete"
            )
        receipt_raw = _read_unique_file(
            trajectory_receipt_path,
            "registered trajectory dataset receipt",
            max_bytes=1024 * 1024,
        )
        receipt = _strict_json(
            receipt_raw,
            "registered trajectory dataset receipt",
        )
        receipt_body = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        receipt_binding = receipt.get("binding")
        if (
            receipt_raw != _canonical_bytes(receipt) + b"\n"
            or receipt.get("schema_version") != 1
            or receipt.get("scope") != "supplementary_trajectory"
            or receipt.get("receipt_sha256") != canonical_sha256(receipt_body)
            or not isinstance(receipt_binding, Mapping)
            or receipt_binding.get("dataset_file_path")
            != "results/trajectory/dataset/evaluator.h5ad"
            or receipt_binding.get("dataset_file_sha256")
            != _hash_unique_file(
                trajectory_dataset,
                "registered trajectory evaluator dataset",
            )
        ):
            raise FinalRunnerContractError(
                "registered trajectory dataset receipt differs"
            )
        owned.update(
            {
                "results/trajectory/dataset/evaluator.h5ad",
                "results/trajectory/dataset/dataset_receipt.json",
            }
        )

    trajectory_authority_root = results / "trajectory/execution_authority"
    for name in (
        "retained_calibration.json",
        "count_score_authority.json",
        "authority.json",
    ):
        path = trajectory_authority_root / name
        if os.path.lexists(path):
            owned.add(f"results/trajectory/execution_authority/{name}")

    trajectory_execution_root = results / "trajectory/execution"
    trajectory_records_root = trajectory_execution_root / "records"
    if os.path.lexists(trajectory_records_root):
        try:
            metadata = trajectory_records_root.lstat()
            names = tuple(
                sorted(path.name for path in trajectory_records_root.iterdir())
            )
        except OSError as error:
            raise FinalRunnerContractError(
                "trajectory execution record paths cannot be enumerated"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise FinalRunnerContractError(
                "trajectory execution record path is not a directory"
            )
        if list(names) != [
            f"{ordinal:08d}.json" for ordinal in range(1, len(names) + 1)
        ]:
            raise FinalRunnerContractError(
                "trajectory execution records are not a canonical prefix"
            )
        for name in names:
            record_path = trajectory_records_root / name
            raw = _read_unique_file(record_path, "trajectory execution record")
            record = _strict_json(raw, "trajectory execution record")
            run = record.get("run")
            if (
                raw != _canonical_bytes(record) + b"\n"
                or set(record)
                != {
                    "run",
                    "metrics",
                    "p_pre_zero_evidence",
                    "execution_request",
                }
                or not isinstance(run, Mapping)
                or not isinstance(run.get("run_id"), str)
                or _TRAJECTORY_RUN_ID.fullmatch(str(run.get("run_id"))) is None
            ):
                raise FinalRunnerContractError("trajectory execution record is invalid")
            owned.add(f"results/trajectory/execution/records/{name}")
            for prefix in (
                "stdout",
                "stderr",
                "native_output",
                "evaluator_output",
            ):
                relative = run.get(f"{prefix}_path")
                if relative is None:
                    continue
                _sha256(
                    run.get(f"{prefix}_file_sha256"),
                    f"trajectory execution {prefix} file",
                )
                safe = PurePosixPath(str(relative))
                if (
                    not isinstance(relative, str)
                    or safe.is_absolute()
                    or ".." in safe.parts
                    or not safe.parts
                    or safe.parts[0] != "runs"
                ):
                    raise FinalRunnerContractError(
                        f"trajectory execution {prefix} path is unsafe"
                    )
                owned.add(f"results/trajectory/execution/{safe.as_posix()}")
            score_evidence = record.get("p_pre_zero_evidence")
            score_storage = (
                score_evidence.get("storage")
                if isinstance(score_evidence, Mapping)
                else None
            )
            if not isinstance(score_storage, Mapping):
                raise FinalRunnerContractError(
                    "trajectory p_pre_zero storage receipt is invalid"
                )
            score_relative = score_storage.get("path")
            if score_relative is not None:
                _sha256(
                    score_storage.get("compressed_sha256"),
                    "trajectory p_pre_zero compressed file",
                )
                safe_score = PurePosixPath(str(score_relative))
                if (
                    not isinstance(score_relative, str)
                    or safe_score.is_absolute()
                    or ".." in safe_score.parts
                    or not safe_score.parts
                    or safe_score.parts[0] != "runs"
                ):
                    raise FinalRunnerContractError(
                        "trajectory p_pre_zero path is unsafe"
                    )
                owned.add(f"results/trajectory/execution/{safe_score.as_posix()}")
    trajectory_execution_manifest = (
        trajectory_execution_root / "execution_manifest.json"
    )
    if os.path.lexists(trajectory_execution_manifest):
        owned.add("results/trajectory/execution/execution_manifest.json")
    owned.update(_scaling_checkpoint_file_bindings(destination))
    return frozenset(owned)


def _owned_final_result_file_manifest(round_dir: Path) -> dict[str, object]:
    manifest = final_result_file_manifest(round_dir)
    observed = {
        item["path"]
        for item in manifest["result_files"]
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    owned = _owned_final_result_paths(round_dir)
    if observed != owned:
        unexpected = sorted(observed - owned)
        missing = sorted(owned - observed)
        detail = unexpected[0] if unexpected else missing[0]
        raise FinalRunnerContractError(
            f"final results contain an unowned or missing path: {detail}"
        )
    return manifest


def _remove_stale_result_temporaries(round_dir: Path) -> tuple[str, ...]:
    """Remove only implementation-defined staging names left by interruption."""

    if not isinstance(round_dir, Path):
        raise TypeError("round_dir must be a pathlib.Path")
    try:
        destination = round_dir.resolve(strict=True)
        results = destination / "results"
        if not os.path.lexists(results):
            return ()
        if (
            destination != round_dir.absolute()
            or results.resolve(strict=True) != results.absolute()
        ):
            raise FinalRunnerContractError(
                "final result temporary path contains a symlink"
            )
        entries = sorted(results.rglob("*"))
    except FinalRunnerContractError:
        raise
    except OSError as error:
        raise FinalRunnerContractError(
            "final result temporaries cannot be enumerated"
        ) from error
    removed: list[str] = []
    for path in entries:
        match = _STAGING_FILE.fullmatch(path.name)
        if match is None:
            continue
        try:
            result_relative = path.relative_to(results)
        except ValueError as error:  # pragma: no cover - rglob containment
            raise FinalRunnerContractError(
                "stale final result temporary escaped results"
            ) from error
        if result_relative.parts[:2] not in {
            ("final", "execution"),
            ("final", "execution_authority"),
            ("trajectory", "dataset"),
            ("trajectory", "execution"),
            ("trajectory", "execution_authority"),
        }:
            continue
        try:
            relative = path.relative_to(destination).as_posix()
            _unlink_owned_staging_temporary(
                path,
                path.with_name(match.group("canonical")),
                "stale final result temporary",
            )
            removed.append(relative)
        except FinalRunnerContractError:
            raise
        except RunnerContractError as error:
            raise FinalRunnerContractError(str(error)) from error
        except OSError as error:
            raise FinalRunnerContractError(
                "stale final result temporary could not be removed"
            ) from error
    return tuple(removed)


def _recover_interrupted_execution_transactions(
    round_dir: Path,
    scope: Literal["final", "trajectory"],
) -> tuple[int, ...]:
    """Roll back artifacts lacking a committed record; retain committed attempts."""

    destination = round_dir.resolve(strict=True)
    execution = destination / f"results/{scope}/execution"
    run_pattern = _FINAL_RUN_ID if scope == "final" else _TRAJECTORY_RUN_ID
    transactions = execution / "transactions"
    if not os.path.lexists(transactions):
        return ()
    try:
        metadata = transactions.lstat()
        names = tuple(sorted(path.name for path in transactions.iterdir()))
    except OSError as error:
        raise FinalRunnerContractError(
            "final transaction directory cannot be enumerated"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FinalRunnerContractError("final transaction directory is invalid")
    recovered: list[int] = []
    for name in names:
        if _TRANSACTION_FILE.fullmatch(name) is None:
            raise FinalRunnerContractError("final transaction name is invalid")
        intent_path = transactions / name
        raw = _read_unique_file(intent_path, "final transaction intent")
        intent = _strict_json(raw, "final transaction intent")
        body = {key: value for key, value in intent.items() if key != "intent_sha256"}
        ordinal = int(name.removesuffix(".json"))
        run_id = intent.get("run_id")
        record_relative = f"records/{ordinal:08d}.json"
        required = {
            f"runs/{run_id}.stdout",
            f"runs/{run_id}.stderr",
        }
        allowed = required | {
            f"runs/{run_id}.log2-cp10k-f64.zlib",
            f"runs/{run_id}.p-pre-zero-f64.zlib",
        }
        artifact_paths = intent.get("artifact_paths")
        if (
            set(intent)
            != {
                "schema_version",
                "ordinal",
                "run_id",
                "record_path",
                "artifact_paths",
                "intent_sha256",
            }
            or intent.get("schema_version") != 1
            or intent.get("ordinal") != ordinal
            or not isinstance(run_id, str)
            or run_pattern.fullmatch(run_id) is None
            or intent.get("record_path") != record_relative
            or not isinstance(artifact_paths, list)
            or not all(isinstance(path, str) for path in artifact_paths)
            or artifact_paths != sorted(artifact_paths)
            or len(set(artifact_paths)) != len(artifact_paths)
            or not required.issubset(set(artifact_paths))
            or not set(artifact_paths).issubset(allowed)
            or intent.get("intent_sha256") != canonical_sha256(body)
            or raw != _canonical_bytes(intent) + b"\n"
        ):
            raise FinalRunnerContractError("final transaction intent is invalid")
        record_path = execution / record_relative
        if os.path.lexists(record_path):
            record = _strict_json(
                _read_unique_file(record_path, "final transaction record"),
                "final transaction record",
            )
            run = record.get("run")
            if not isinstance(run, Mapping) or run.get("run_id") != run_id:
                raise FinalRunnerContractError(
                    "final transaction record differs from its intent"
                )
        else:
            for relative in artifact_paths:
                path = execution.joinpath(*PurePosixPath(relative).parts)
                if not os.path.lexists(path):
                    continue
                item = path.lstat()
                if (
                    not stat.S_ISREG(item.st_mode)
                    or stat.S_ISLNK(item.st_mode)
                    or item.st_uid != os.geteuid()
                    or item.st_nlink != 1
                ):
                    raise FinalRunnerContractError(
                        "interrupted final artifact is not an owned unique file"
                    )
                path.unlink()
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        intent_path.unlink()
        directory = os.open(transactions, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        recovered.append(ordinal)
    try:
        transactions.rmdir()
    except OSError as error:
        raise FinalRunnerContractError(
            "final transaction directory did not become empty"
        ) from error
    directory = os.open(transactions.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    for empty_candidate in (execution / "runs", execution):
        try:
            empty_candidate.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            break
        parent = os.open(empty_candidate.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    return tuple(recovered)


def _recover_interrupted_final_transactions(round_dir: Path) -> tuple[int, ...]:
    """Recover only primary final execution transactions."""

    return _recover_interrupted_execution_transactions(round_dir, "final")


def _recover_interrupted_trajectory_transactions(round_dir: Path) -> tuple[int, ...]:
    """Recover only supplementary trajectory execution transactions."""

    return _recover_interrupted_execution_transactions(round_dir, "trajectory")


def _record_incremental_results_if_changed(
    repository: Path,
    round_dir: Path,
    recorder: Callable[..., object],
) -> object | None:
    """Reconcile a cumulative result tree while treating exact no-change as resume."""

    from .study import StudyStateError

    if not isinstance(repository, Path) or not isinstance(round_dir, Path):
        raise TypeError("repository and round_dir must be pathlib.Path values")
    if not callable(recorder):
        raise TypeError("recorder must be callable")
    manifest = _owned_final_result_file_manifest(round_dir)
    try:
        return recorder(round_dir, manifest, repo=repository)
    except StudyStateError as error:
        if str(error) == "incremental result manifest adds no files":
            return None
        raise


def _recover_scaling_transactions_for_resume(
    repository: Path,
    round_dir: Path,
) -> tuple[str, ...]:
    """Recover checkpoint-unreferenced scaling run publications before journaling."""

    scaling_root = round_dir / "results/scaling"
    if not os.path.lexists(scaling_root):
        return ()
    from .scaling import ScalingResultStore, load_scaling_execution_authority

    try:
        authority = load_scaling_execution_authority(repository)
        return ScalingResultStore(
            scaling_root,
            authority.plan,
        ).recover_unreferenced_transactions()
    except Exception as error:
        raise FinalRunnerContractError(
            "supplementary scaling transactions are not resumable"
        ) from error


def _validate_scaling_publications_for_reconciliation(
    repository: Path,
    round_dir: Path,
) -> object | None:
    """Replay the exact frozen scaling checkpoint prefix before journaling it."""

    scaling_root = round_dir / "results/scaling"
    if not os.path.lexists(scaling_root):
        return None
    from .scaling import ScalingResultStore, load_scaling_execution_authority

    try:
        authority = load_scaling_execution_authority(repository)
        return ScalingResultStore(scaling_root, authority.plan).load(
            force_validate=True
        )
    except Exception as error:
        raise FinalRunnerContractError(
            "supplementary scaling publications are not resumable"
        ) from error


def _reconcile_interrupted_final_publications(
    repository: Path,
    round_dir: Path,
    frozen_method: Mapping[str, object],
    recorder: Callable[..., object],
) -> object | None:
    """Validate every observed owned publication, then journal them atomically.

    This runs before lifecycle claim validation because the claim verifier must
    reject unjournaled files.  No observed known-path byte is added to the journal
    until its exact frozen authority, plan, record, and manifest bindings have all
    been replayed successfully.
    """

    from . import study
    from .methods import load_method_registry

    selected_repository, destination = _canonical_round(repository, round_dir)
    if not callable(recorder):
        raise TypeError("recorder must be callable")
    results = destination / "results"
    status_path = results / "dataset_status.json"
    if not os.path.lexists(status_path):
        if not os.path.lexists(results):
            return None
        inventory = final_result_file_manifest(destination)
        if inventory["result_files"]:
            raise FinalRunnerContractError(
                "interrupted final publications lack their dataset status authority"
            )
        return None

    try:
        freeze = study._validate_freeze(destination, selected_repository)
        materialization, _seed_manifest = study._validate_seed_manifest(
            destination,
            freeze,
        )
        execution = study._validate_execution_claim_record(
            destination,
            freeze,
            materialization,
        )
        journal = study._validate_result_journal(
            selected_repository,
            destination,
            freeze,
            materialization,
            execution,
        )
    except Exception as error:
        raise FinalRunnerContractError(
            "existing final result journal is not resumable"
        ) from error
    observed_inventory = final_result_file_manifest(destination)
    if observed_inventory["result_files"] == journal["cumulative_result_files"]:
        return None

    _remove_unreceipted_trajectory_dataset(destination)
    claim_path = destination / "execution_claim.json"
    claim = _strict_json(
        _read_unique_file(claim_path, "final execution claim"),
        "final execution claim",
    )
    claim_sha256 = canonical_sha256(claim)
    bindings, prepared = _load_unclaimed_prepared_final_panel(
        selected_repository,
        destination,
    )
    registry = load_method_registry(selected_repository / "study/methods.json")
    environments = _load_final_execution_environment_registry(
        selected_repository,
        registry,
    )
    _validate_final_runtime_lock(frozen_method, environments)
    (
        registered,
        authority,
        plan,
        trajectory_authority,
        trajectory_plan,
    ) = _materialize_final_execution_inputs(
        selected_repository,
        destination,
        frozen_method,
        registry,
        bindings,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environments.registry_sha256,
        publish_results=lambda: None,
    )
    store = FinalResultStore(
        destination / "results/final/execution",
        plan,
        prepared,
        authority,
        authority_repository=selected_repository,
    )
    records = store.load_records()
    store._records_cache = records
    if os.path.lexists(store.manifest_path):
        store.load_manifest()
    trajectory_store = FinalResultStore(
        destination / "results/trajectory/execution",
        trajectory_plan,
        {registered.binding.dataset_id: registered.prepared},
        trajectory_authority,
        authority_repository=selected_repository,
    )
    trajectory_records = trajectory_store.load_records()
    trajectory_store._records_cache = trajectory_records
    if os.path.lexists(trajectory_store.manifest_path):
        trajectory_store.load_manifest()
    _validate_scaling_publications_for_reconciliation(
        selected_repository,
        destination,
    )
    return _record_incremental_results_if_changed(
        selected_repository,
        destination,
        recorder,
    )


class FinalResultStore:
    """Append-only per-attempt artifacts plus one immutable completion manifest."""

    def __init__(
        self,
        output_dir: Path,
        plan: FinalExecutionPlan | TrajectoryExecutionPlan,
        prepared_datasets: Mapping[str, PreparedDataset],
        execution_authority: ExecutionAuthorityContext,
        *,
        authority_repository: Path | None = None,
    ) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        if not isinstance(plan, (FinalExecutionPlan, TrajectoryExecutionPlan)):
            raise TypeError(
                "plan must be a FinalExecutionPlan or TrajectoryExecutionPlan"
            )
        if not isinstance(prepared_datasets, Mapping):
            raise TypeError("prepared_datasets must be a mapping")
        if not isinstance(execution_authority, ExecutionAuthorityContext):
            raise TypeError("execution_authority must be an ExecutionAuthorityContext")
        if (
            plan.input_hashes.get("execution_authority_sha256")
            != execution_authority.authority_sha256
        ):
            raise FinalRunnerContractError(
                "final result authority differs from the execution plan"
            )
        self.output_dir = output_dir.absolute()
        self.plan = plan
        self.scope = (
            "primary_final" if isinstance(plan, FinalExecutionPlan) else plan.scope
        )
        self.prepared_datasets = MappingProxyType(
            self._validate_prepared_datasets(prepared_datasets)
        )
        self.execution_authority = execution_authority
        self._artifacts = CheckpointStore(
            self.output_dir,
            authority_repository=authority_repository,
        )
        self._records_cache: tuple[dict[str, object], ...] | None = None

    def _validate_prepared_datasets(
        self, prepared_datasets: Mapping[str, PreparedDataset]
    ) -> dict[str, PreparedDataset]:
        planned_dataset_ids = {entry.run.dataset_id for entry in self.plan.entries}
        if set(prepared_datasets) != planned_dataset_ids:
            raise FinalRunnerContractError(
                "prepared dataset authority does not exactly cover the final plan"
            )
        result: dict[str, PreparedDataset] = {}
        for dataset_id in sorted(planned_dataset_ids):
            prepared = prepared_datasets.get(dataset_id)
            if not isinstance(prepared, PreparedDataset):
                raise FinalRunnerContractError(
                    f"prepared final dataset is invalid for {dataset_id}"
                )
            entries = tuple(
                entry.run
                for entry in self.plan.entries
                if entry.run.dataset_id == dataset_id
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
                or prepared.method_input.obs_ids != prepared.audit.retained_cell_ids
            ):
                raise FinalRunnerContractError(
                    f"prepared final dataset differs from plan binding {dataset_id}"
                )
            if isinstance(self.plan, FinalExecutionPlan):
                if type(prepared.binding) is not DatasetBinding:
                    raise FinalRunnerContractError(
                        "primary final store requires exact final dataset bindings"
                    )
            else:
                binding = prepared.binding
                if (
                    len(planned_dataset_ids) != 1
                    or not isinstance(binding, RegisteredTrajectoryBinding)
                    or self.plan.scope != "supplementary_trajectory"
                    or self.plan.input_hashes.get("trajectory_binding_sha256")
                    != binding.registered_binding_sha256
                    or self.plan.input_hashes.get("trajectory_authority_sha256")
                    != binding.authority_sha256
                    or self.plan.input_hashes.get("trajectory_authority_file_sha256")
                    != binding.authority_file_sha256
                    or self.plan.input_hashes.get("trajectory_dataset_sha256")
                    != binding.dataset_sha256
                    or self.plan.input_hashes.get("trajectory_dataset_file_sha256")
                    != binding.dataset_file_sha256
                    or self.plan.input_hashes.get("trajectory_method_input_sha256")
                    != method_input_sha256(prepared.method_input)
                    or self.plan.input_hashes.get("trajectory_retained_cell_ids_sha256")
                    != prepared.audit.retained_cell_ids_sha256
                    or self.plan.input_hashes.get("dataset_qc_policy_sha256")
                    != DatasetQCPolicy.fixed().sha256
                ):
                    raise FinalRunnerContractError(
                        "registered trajectory prepared dataset differs from plan"
                    )
            observed, truth, truth_kind = _prezero_evaluator_targets(prepared)
            if observed.shape != prepared.method_input.shape or not np.array_equal(
                observed, prepared.method_input.counts
            ):
                raise FinalRunnerContractError(
                    f"prepared final observations differ from method input {dataset_id}"
                )
            if truth_kind != "orthogonal_only" and (
                truth is None or truth.shape != prepared.method_input.shape
            ):
                raise FinalRunnerContractError(
                    f"prepared final truth differs from method input {dataset_id}"
                )
            result[dataset_id] = prepared
        return result

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "execution_manifest.json"

    def _record_path(self, ordinal: int) -> Path:
        return self.output_dir / "records" / f"{ordinal:08d}.json"

    def _publish_transaction_intent(
        self, plan_entry: FinalPlanEntry, attempt: EvaluatedAttempt
    ) -> Path:
        artifacts = sorted(
            {
                f"runs/{plan_entry.run.run_id}.stdout",
                f"runs/{plan_entry.run.run_id}.stderr",
                *(
                    ()
                    if attempt.evaluator_output is None
                    else (f"runs/{plan_entry.run.run_id}.log2-cp10k-f64.zlib",)
                ),
                *(
                    ()
                    if attempt.p_pre_zero_evidence.raw_matrix_bytes is None
                    else (f"runs/{plan_entry.run.run_id}.p-pre-zero-f64.zlib",)
                ),
            }
        )
        body: dict[str, object] = {
            "schema_version": 1,
            "ordinal": plan_entry.run.ordinal,
            "run_id": plan_entry.run.run_id,
            "record_path": f"records/{plan_entry.run.ordinal:08d}.json",
            "artifact_paths": artifacts,
        }
        intent = {**body, "intent_sha256": canonical_sha256(body)}
        relative, _digest = self._artifacts._publish_immutable(
            f"transactions/{plan_entry.run.ordinal:08d}.json",
            _canonical_bytes(intent) + b"\n",
        )
        return self.output_dir / relative

    def _complete_transaction(self, intent_path: Path) -> None:
        intent_path.unlink()
        directory = os.open(intent_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        intent_path.parent.rmdir()
        directory = os.open(self.output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _stored_final_attempt(
        self,
        attempt: EvaluatedAttempt,
        *,
        artifacts: CheckpointStore | None = None,
    ) -> dict[str, object]:
        """Store one compressed evaluator matrix and omit its redundant native form."""

        artifact_store = self._artifacts if artifacts is None else artifacts
        without_dense_outputs = replace(
            attempt,
            native_output=None,
            evaluator_output=None,
        )
        stored = artifact_store._stored_attempt(without_dense_outputs)
        run = stored["run"]
        assert isinstance(run, dict)
        run["native_output_retention"] = (
            "not_available"
            if attempt.native_output is None
            else _FINAL_NATIVE_RETENTION
        )
        run.update(
            {
                "evaluator_output_encoding": None,
                "evaluator_output_uncompressed_nbytes": None,
                "evaluator_output_uncompressed_sha256": None,
            }
        )
        if attempt.evaluator_output is None:
            return stored
        evaluator = np.asarray(attempt.evaluator_output, dtype="<f8", order="C")
        raw = evaluator.tobytes(order="C")
        compressed = zlib.compress(raw, level=_FINAL_OUTPUT_COMPRESSION_LEVEL)
        relative, digest = artifact_store._publish_immutable(
            f"runs/{attempt.run.run_id}.log2-cp10k-f64.zlib",
            compressed,
        )
        run.update(
            {
                "evaluator_output_path": relative,
                "evaluator_output_file_sha256": digest,
                "evaluator_output_shape": list(evaluator.shape),
                "evaluator_output_dtype": "<f8",
                "evaluator_scale": "log2_cp10k_plus_1",
                "evaluator_output_encoding": _FINAL_OUTPUT_ENCODING,
                "evaluator_output_uncompressed_nbytes": len(raw),
                "evaluator_output_uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        return stored

    def _validate_final_output_storage(
        self,
        run: Mapping[str, object],
        *,
        artifacts: CheckpointStore | None = None,
    ) -> dict[str, object]:
        """Validate bounded decompression and return a raw-store validation view."""

        artifact_store = self._artifacts if artifacts is None else artifacts
        validation_run = dict(run)
        native_present = run.get("native_output_sha256") is not None
        if run.get("native_output_retention") != (
            _FINAL_NATIVE_RETENTION if native_present else "not_available"
        ) or any(
            run.get(f"native_output_{suffix}") is not None
            for suffix in ("path", "file_sha256", "shape", "dtype")
        ):
            raise FinalRunnerContractError(
                "final native output retention policy differs"
            )

        path_value = run.get("evaluator_output_path")
        compression_fields = (
            "evaluator_output_encoding",
            "evaluator_output_uncompressed_nbytes",
            "evaluator_output_uncompressed_sha256",
        )
        if path_value is None:
            if run.get("evaluator_output_sha256") is not None or any(
                run.get(name) is not None for name in compression_fields
            ):
                raise FinalRunnerContractError(
                    "final evaluator output compression binding is partial"
                )
            return validation_run
        if (
            not isinstance(path_value, str)
            or not path_value.endswith(".log2-cp10k-f64.zlib")
            or run.get("evaluator_output_encoding") != _FINAL_OUTPUT_ENCODING
            or run.get("evaluator_output_dtype") != "<f8"
            or run.get("evaluator_scale") != "log2_cp10k_plus_1"
        ):
            raise FinalRunnerContractError("final evaluator output encoding differs")
        shape = run.get("evaluator_output_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value <= 0 for value in shape)
        ):
            raise FinalRunnerContractError("final evaluator output shape is invalid")
        expected_nbytes = shape[0] * shape[1] * 8
        if expected_nbytes > _FINAL_MATRIX_UNCOMPRESSED_NBYTES:
            raise FinalRunnerContractError(
                "compressed evaluator output exceeds the fixed final matrix bound"
            )
        if run.get("evaluator_output_uncompressed_nbytes") != expected_nbytes:
            raise FinalRunnerContractError(
                "final evaluator uncompressed byte count differs"
            )
        expected_raw_sha256 = _sha256(
            run.get("evaluator_output_uncompressed_sha256"),
            "final evaluator uncompressed output",
        )
        try:
            expected_file_sha256 = _sha256(
                run.get("evaluator_output_file_sha256"),
                "compressed evaluator output file",
            )
            path = artifact_store._safe_artifact_path(
                path_value,
                "compressed evaluator output",
            )
            maximum_compressed = _zlib_compress_bound(expected_nbytes)
            compressed = _read_unique_file(
                path,
                "compressed evaluator output",
                max_bytes=maximum_compressed,
            )
            if hashlib.sha256(compressed).hexdigest() != expected_file_sha256:
                raise FinalRunnerContractError(
                    "compressed evaluator output checksum differs"
                )
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(compressed, expected_nbytes + 1)
            raw += decompressor.flush(max(1, expected_nbytes + 1 - len(raw)))
        except (OSError, RunnerContractError, zlib.error) as error:
            raise FinalRunnerContractError(
                "compressed evaluator output is invalid"
            ) from error
        if (
            len(raw) != expected_nbytes
            or not decompressor.eof
            or decompressor.unconsumed_tail
            or decompressor.unused_data
            or hashlib.sha256(raw).hexdigest() != expected_raw_sha256
        ):
            raise FinalRunnerContractError(
                "compressed evaluator output differs from its receipt"
            )
        values = np.frombuffer(raw, dtype="<f8")
        if values.size != shape[0] * shape[1] or not np.isfinite(values).all():
            raise FinalRunnerContractError(
                "compressed evaluator output contains invalid values"
            )
        for suffix in ("path", "file_sha256", "shape", "dtype"):
            validation_run[f"evaluator_output_{suffix}"] = None
        return validation_run

    def _validate_execution_request_receipt(
        self,
        value: object,
        plan_entry: FinalPlanEntry,
        run: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        direct_comparator = plan_entry.run.configuration_kind == "comparator_tuning"
        if value is None:
            if plan_entry.action == "not_applicable":
                return None
            if direct_comparator:
                raise FinalRunnerContractError(
                    "executed final comparator lacks its complete request receipt"
                )
            if run.get("status") == "completed" and plan_entry.run.requires_calibration:
                raise FinalRunnerContractError(
                    "completed final calibration lacks its execution request receipt"
                )
            return None
        if direct_comparator:
            if not isinstance(value, Mapping) or set(value) != {
                "request_kind",
                "configuration",
                "dataset_id",
                "execution_authority_sha256",
                "method_input_sha256",
                "model_seed",
            }:
                raise FinalRunnerContractError(
                    "final comparator request receipt has an invalid schema"
                )
            for name in (
                "execution_authority_sha256",
                "method_input_sha256",
            ):
                _sha256(value.get(name), f"final comparator request {name}")
            expected_configuration = plan_entry.run.comparator_configuration
            if (
                expected_configuration is None
                or value.get("request_kind") != "frozen_comparator_direct"
                or not direct_equal(
                    value.get("configuration"),
                    direct_bound_comparator_value(expected_configuration),
                )
                or value.get("dataset_id") != plan_entry.run.dataset_id
                or value.get("execution_authority_sha256")
                != self.plan.input_hashes.get("execution_authority_sha256")
                or value.get("method_input_sha256") != run.get("method_input_sha256")
                or value.get("model_seed") != plan_entry.run.model_seed
            ):
                raise FinalRunnerContractError(
                    "final comparator request receipt differs from its plan"
                )
            return value
        if not isinstance(value, Mapping) or set(value) != {
            "calibration_usage",
            "configuration_sha256",
            "count_score_manifest_sha256",
            "dataset_id",
            "execution_authority_sha256",
            "method_input_sha256",
            "model_seed",
            "request_sha256",
            "retained_calibration_sha256",
        }:
            raise FinalRunnerContractError(
                "final execution request receipt has an invalid schema"
            )
        if plan_entry.action != "execute":
            raise FinalRunnerContractError(
                "non-run final entry has an execution request receipt"
            )
        for name in (
            "configuration_sha256",
            "execution_authority_sha256",
            "method_input_sha256",
            "request_sha256",
        ):
            _sha256(value.get(name), f"final execution request {name}")
        for name in (
            "count_score_manifest_sha256",
            "retained_calibration_sha256",
        ):
            nested = value.get(name)
            if nested is not None:
                _sha256(nested, f"final execution request {name}")
        if (
            value.get("calibration_usage") != "retained_all_development"
            or value.get("configuration_sha256") != plan_entry.run.configuration_sha256
            or value.get("dataset_id") != plan_entry.run.dataset_id
            or value.get("execution_authority_sha256")
            != self.plan.input_hashes.get("execution_authority_sha256")
            or value.get("method_input_sha256") != run.get("method_input_sha256")
            or value.get("model_seed") != plan_entry.run.model_seed
            or (
                plan_entry.run.requires_count_score
                and value.get("count_score_manifest_sha256") is None
            )
            or (
                plan_entry.run.requires_calibration
                and value.get("retained_calibration_sha256") is None
            )
            or value.get("count_score_manifest_sha256")
            != (
                self.execution_authority.count_score_manifest_sha256
                if plan_entry.run.requires_count_score
                else None
            )
            or value.get("retained_calibration_sha256")
            != (
                self.execution_authority.retained_calibration_sha256
                if plan_entry.run.method_id in {"maskimpute", "capacity-matched-ae"}
                else None
            )
        ):
            raise FinalRunnerContractError(
                "final execution request receipt differs from its plan"
            )
        return value

    def _read_record(self, path: Path, plan_entry: FinalPlanEntry) -> dict[str, object]:
        raw = _read_unique_file(path, "final execution record")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise FinalRunnerContractError(
                "final execution record is invalid"
            ) from error
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "run",
                "metrics",
                "p_pre_zero_evidence",
                "execution_request",
            }
            or raw != _canonical_bytes(value) + b"\n"
        ):
            raise FinalRunnerContractError("final execution record is not canonical")
        try:
            run = value.get("run")
            if not isinstance(run, Mapping):
                raise FinalRunnerContractError("final execution run is invalid")
            self._validate_execution_request_receipt(
                value.get("execution_request"), plan_entry, run
            )
            validation_run = self._validate_final_output_storage(run)
            self._artifacts._validate_stored_record(
                {
                    "run": validation_run,
                    "metrics": value["metrics"],
                    "p_pre_zero_evidence": value["p_pre_zero_evidence"],
                },
                plan_entry.run,
                prepared=self.prepared_datasets[plan_entry.run.dataset_id],
                expected_dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
                expected_score_config_sha256=(
                    self.execution_authority.count_model_config_sha256
                ),
                expected_calibration_file_sha256=(
                    self.execution_authority.retained_calibration_sha256
                ),
                execution_authority=self.execution_authority,
                calibration_usage="retained_all_development",
            )
        except (RunnerContractError, OSError, ValueError) as error:
            raise FinalRunnerContractError(
                f"final execution record or artifact is invalid: {error}"
            ) from error
        return value

    def _record_names(self) -> tuple[str, ...]:
        records_dir = self.output_dir / "records"
        if not os.path.lexists(records_dir):
            return ()
        try:
            metadata = records_dir.lstat()
            names = tuple(sorted(path.name for path in records_dir.iterdir()))
        except OSError as error:
            raise FinalRunnerContractError(
                "final record directory is unavailable"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise FinalRunnerContractError("final record directory is invalid")
        expected_names = [f"{ordinal:08d}.json" for ordinal in range(1, len(names) + 1)]
        if list(names) != expected_names or len(names) > len(self.plan.entries):
            raise FinalRunnerContractError("final execution records are not a prefix")
        return names

    def load_records(self) -> tuple[dict[str, object], ...]:
        records_dir = self.output_dir / "records"
        names = self._record_names()
        return tuple(
            self._read_record(records_dir / name, self.plan.entries[index])
            for index, name in enumerate(names)
        )

    def _cached_records(self) -> tuple[dict[str, object], ...]:
        if self._records_cache is None:
            self._records_cache = self.load_records()
        names = self._record_names()
        if len(names) != len(self._records_cache):
            raise FinalRunnerContractError(
                "final execution record prefix changed during this process"
            )
        return self._records_cache

    def append(
        self,
        plan_entry: FinalPlanEntry,
        attempt: EvaluatedAttempt,
        *,
        execution_request: (
            ExecutionRequest | FinalComparatorExecutionRequest | None
        ) = None,
    ) -> dict[str, object]:
        if not isinstance(plan_entry, FinalPlanEntry):
            raise TypeError("plan_entry must be a FinalPlanEntry")
        if not isinstance(attempt, EvaluatedAttempt):
            raise TypeError("attempt must be an EvaluatedAttempt")
        if os.path.lexists(self.manifest_path):
            raise FinalRunnerContractError("final execution is already complete")
        records = self._cached_records()
        next_index = len(records)
        if (
            next_index >= len(self.plan.entries)
            or plan_entry != self.plan.entries[next_index]
        ):
            raise FinalRunnerContractError(
                "final attempts must follow exact plan order"
            )
        if attempt.run.run_id != plan_entry.run.run_id:
            raise FinalRunnerContractError("final attempt differs from its plan entry")
        request_receipt: dict[str, object] | None = None
        if execution_request is not None:
            if isinstance(execution_request, FinalComparatorExecutionRequest):
                execution_request.validate_integrity()
                configuration = plan_entry.run.comparator_configuration
                input_identity = method_input_sha256(execution_request.method_input)
                if (
                    plan_entry.action != "execute"
                    or plan_entry.run.configuration_kind != "comparator_tuning"
                    or configuration is None
                    or execution_request.method_spec.id != plan_entry.run.method_id
                    or input_identity != attempt.run.method_input_sha256
                    or execution_request.model_seed != plan_entry.run.model_seed
                    or execution_request.dataset_id != plan_entry.run.dataset_id
                    or execution_request.mechanism != plan_entry.run.mechanism
                    or execution_request.biological_id != plan_entry.run.biological_id
                    or execution_request.technical_view != plan_entry.run.technical_view
                    or not direct_equal(
                        execution_request.configuration,
                        configuration,
                    )
                    or execution_request.execution_authority != self.execution_authority
                ):
                    raise FinalRunnerContractError(
                        "final comparator request differs from its plan entry"
                    )
                request_receipt = {
                    "request_kind": "frozen_comparator_direct",
                    "configuration": direct_bound_comparator_value(
                        execution_request.configuration
                    ),
                    "dataset_id": execution_request.dataset_id,
                    "execution_authority_sha256": (
                        execution_request.execution_authority.authority_sha256
                    ),
                    "method_input_sha256": input_identity,
                    "model_seed": execution_request.model_seed,
                }
            elif not isinstance(execution_request, ExecutionRequest):
                raise TypeError("execution_request must be a supported final request")
            else:
                execution_request.validate_integrity()
                if (
                    plan_entry.action != "execute"
                    or execution_request.method_spec.id != plan_entry.run.method_id
                    or execution_request.method_input_sha256
                    != attempt.run.method_input_sha256
                    or execution_request.model_seed != plan_entry.run.model_seed
                    or execution_request.dataset_id != plan_entry.run.dataset_id
                    or execution_request.mechanism != plan_entry.run.mechanism
                    or execution_request.biological_id != plan_entry.run.biological_id
                    or execution_request.technical_view != plan_entry.run.technical_view
                    or execution_request.configuration_id
                    != plan_entry.run.configuration_id
                    or execution_request.configuration_sha256
                    != plan_entry.run.configuration_sha256
                    or execution_request.execution_authority_sha256
                    != self.plan.input_hashes.get("execution_authority_sha256")
                    or execution_request.calibration_usage != "retained_all_development"
                    or execution_request.calibration_context is not None
                ):
                    raise FinalRunnerContractError(
                        "final execution request differs from its plan entry"
                    )
                request_receipt = {
                    "calibration_usage": execution_request.calibration_usage,
                    "configuration_sha256": execution_request.configuration_sha256,
                    "count_score_manifest_sha256": (
                        execution_request.count_score_manifest_sha256
                    ),
                    "dataset_id": execution_request.dataset_id,
                    "execution_authority_sha256": (
                        execution_request.execution_authority_sha256
                    ),
                    "method_input_sha256": execution_request.method_input_sha256,
                    "model_seed": execution_request.model_seed,
                    "request_sha256": execution_request.request_sha256,
                    "retained_calibration_sha256": (
                        execution_request.retained_calibration_sha256
                    ),
                }
        try:
            with tempfile.TemporaryDirectory(
                prefix="maskimpute-final-attempt-stage-"
            ) as staging_name:
                staging = CheckpointStore(
                    Path(staging_name),
                    authority_repository=self._artifacts.authority_repository,
                )
                staged_base = json.loads(
                    _canonical_bytes(
                        self._stored_final_attempt(attempt, artifacts=staging)
                    ).decode("utf-8")
                )
                staged = {
                    "run": staged_base["run"],
                    "metrics": staged_base["metrics"],
                    "p_pre_zero_evidence": staged_base["p_pre_zero_evidence"],
                    "execution_request": request_receipt,
                }
                self._validate_execution_request_receipt(
                    request_receipt, plan_entry, staged["run"]
                )
                staged_validation_run = self._validate_final_output_storage(
                    staged["run"], artifacts=staging
                )
                staging._validate_stored_record(
                    {
                        "run": staged_validation_run,
                        "metrics": staged["metrics"],
                        "p_pre_zero_evidence": staged["p_pre_zero_evidence"],
                    },
                    plan_entry.run,
                    prepared=self.prepared_datasets[plan_entry.run.dataset_id],
                    expected_dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
                    expected_score_config_sha256=(
                        self.execution_authority.count_model_config_sha256
                    ),
                    expected_calibration_file_sha256=(
                        self.execution_authority.retained_calibration_sha256
                    ),
                    execution_authority=self.execution_authority,
                    calibration_usage="retained_all_development",
                )
                self._artifacts._prezero_authority_cache.update(
                    staging._prezero_authority_cache
                )
        except (RunnerContractError, OSError, ValueError) as error:
            raise FinalRunnerContractError(
                "cannot publish final execution record"
            ) from error
        intent_path = self._publish_transaction_intent(plan_entry, attempt)
        try:
            base_stored = json.loads(
                _canonical_bytes(self._stored_final_attempt(attempt)).decode("utf-8")
            )
            stored = {
                "run": base_stored["run"],
                "metrics": base_stored["metrics"],
                "p_pre_zero_evidence": base_stored["p_pre_zero_evidence"],
                "execution_request": request_receipt,
            }
            self._validate_execution_request_receipt(
                request_receipt, plan_entry, stored["run"]
            )
            validation_run = self._validate_final_output_storage(stored["run"])
            self._artifacts._validate_stored_record(
                {
                    "run": validation_run,
                    "metrics": stored["metrics"],
                    "p_pre_zero_evidence": stored["p_pre_zero_evidence"],
                },
                plan_entry.run,
                prepared=self.prepared_datasets[plan_entry.run.dataset_id],
                expected_dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
                expected_score_config_sha256=(
                    self.execution_authority.count_model_config_sha256
                ),
                expected_calibration_file_sha256=(
                    self.execution_authority.retained_calibration_sha256
                ),
                execution_authority=self.execution_authority,
                calibration_usage="retained_all_development",
            )
            self._artifacts._publish_immutable(
                f"records/{plan_entry.run.ordinal:08d}.json",
                _canonical_bytes(stored) + b"\n",
            )
        except (RunnerContractError, OSError, ValueError) as error:
            raise FinalRunnerContractError(
                "cannot publish final execution record"
            ) from error
        names = self._record_names()
        if len(names) != next_index + 1:
            raise FinalRunnerContractError("final execution record publication failed")
        observed = self._read_record(
            self.output_dir / "records" / names[-1], plan_entry
        )
        self._complete_transaction(intent_path)
        self._records_cache = (*records, observed)
        return observed

    def finalize(self) -> dict[str, object]:
        if os.path.lexists(self.manifest_path):
            raise FinalRunnerContractError("final execution is already complete")
        records = self.load_records()
        self._records_cache = records
        if len(records) != len(self.plan.entries):
            raise FinalRunnerContractError("final execution records are incomplete")
        references = []
        for entry in self.plan.entries:
            path = self._record_path(entry.run.ordinal)
            raw = _read_unique_file(path, "final execution record")
            references.append(
                {
                    "ordinal": entry.run.ordinal,
                    "run_id": entry.run.run_id,
                    "path": path.relative_to(self.output_dir).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        body: dict[str, object] = {
            "schema_version": 1,
            "status": "completed",
            "plan_sha256": self.plan.plan_sha256,
            "input_hashes": dict(self.plan.input_hashes),
            "planned_run_count": len(self.plan.entries),
            "recorded_run_count": len(records),
            "records": references,
            "artifact_storage": dict(_FINAL_STORAGE_POLICY),
        }
        if isinstance(self.plan, TrajectoryExecutionPlan):
            body.update(
                {
                    "scope": self.plan.scope,
                    "plan_entries": [entry.to_dict() for entry in self.plan.entries],
                    "configurations": [
                        value.to_dict() for value in self.plan.configurations
                    ],
                    "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
                }
            )
        manifest = {**body, "manifest_sha256": canonical_sha256(body)}
        try:
            self._artifacts._publish_immutable(
                "execution_manifest.json", _canonical_bytes(manifest) + b"\n"
            )
        except (RunnerContractError, OSError) as error:
            raise FinalRunnerContractError(
                "cannot publish final execution manifest"
            ) from error
        return self.load_manifest()

    def load_manifest(self) -> dict[str, object]:
        raw = _read_unique_file(self.manifest_path, "final execution manifest")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise FinalRunnerContractError(
                "final execution manifest is invalid"
            ) from error
        expected_fields = {
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
        if isinstance(self.plan, TrajectoryExecutionPlan):
            expected_fields.update(
                {
                    "scope",
                    "plan_entries",
                    "configurations",
                    "model_seed_policy",
                }
            )
        if (
            not isinstance(value, dict)
            or raw != _canonical_bytes(value) + b"\n"
            or set(value) != expected_fields
        ):
            raise FinalRunnerContractError("final execution manifest is not canonical")
        body = {
            key: nested for key, nested in value.items() if key != "manifest_sha256"
        }
        records = self._cached_records()
        references = value.get("records")
        if (
            value.get("schema_version") != 1
            or value.get("status") != "completed"
            or value.get("plan_sha256") != self.plan.plan_sha256
            or value.get("input_hashes") != dict(self.plan.input_hashes)
            or value.get("planned_run_count") != len(self.plan.entries)
            or value.get("recorded_run_count") != len(records)
            or len(records) != len(self.plan.entries)
            or value.get("artifact_storage") != _FINAL_STORAGE_POLICY
            or (
                isinstance(self.plan, TrajectoryExecutionPlan)
                and (
                    value.get("scope") != self.plan.scope
                    or value.get("plan_entries")
                    != [entry.to_dict() for entry in self.plan.entries]
                    or value.get("configurations")
                    != [value.to_dict() for value in self.plan.configurations]
                    or value.get("model_seed_policy") != list(DEVELOPMENT_MODEL_SEEDS)
                )
            )
            or value.get("manifest_sha256") != canonical_sha256(body)
            or not isinstance(references, list)
            or len(references) != len(records)
        ):
            raise FinalRunnerContractError("final execution manifest differs from plan")
        for plan_entry, reference in zip(self.plan.entries, references, strict=True):
            path = self._record_path(plan_entry.run.ordinal)
            if not isinstance(reference, Mapping) or reference != {
                "ordinal": plan_entry.run.ordinal,
                "run_id": plan_entry.run.run_id,
                "path": path.relative_to(self.output_dir).as_posix(),
                "sha256": hashlib.sha256(
                    _read_unique_file(path, "final execution record")
                ).hexdigest(),
            }:
                raise FinalRunnerContractError(
                    "final execution manifest record binding differs"
                )
        return value


def _final_run_id(
    ordinal: int,
    method_id: str,
    dataset_id: str,
    model_seed: int | None,
    configuration: FrozenPlanMethodAuthority,
) -> str:
    seed = "deterministic" if model_seed is None else f"seed-{model_seed}"
    if configuration.kind in {"comparator_tuning", "comparator_nonexecution"}:
        return (
            f"final-{ordinal:04d}-{method_id}-"
            f"{dataset_id.removeprefix('dataset-')}-{seed}-"
            f"{configuration.configuration_id}"
        )
    assert configuration.legacy_configuration is not None
    return (
        f"final-{method_id}-{dataset_id.removeprefix('dataset-')}-{seed}-"
        f"{configuration.legacy_configuration.configuration_sha256[:12]}"
    )


def _configuration_for_method(
    method_id: str,
    spec: object,
    frozen_method: Mapping[str, object],
) -> AuthorizedConfiguration:
    if method_id == "maskimpute":
        payload = frozen_method.get("selected_configuration")
        configuration_id = frozen_method.get("selected_configuration_id")
        digest = frozen_method.get("selected_configuration_sha256")
        selected_version = frozen_method.get("selected_version")
        if (
            not isinstance(payload, Mapping)
            or not isinstance(configuration_id, str)
            or _SAFE_ID.fullmatch(configuration_id) is None
            or canonical_sha256(payload) != digest
            or payload.get("method_version") != selected_version
        ):
            raise FinalRunnerContractError("frozen candidate configuration is invalid")
        score_policy = payload.get("score_policy")
        if score_policy == "retained_development_calibrator":
            requires_calibration = True
        elif score_policy == "direct_cross_fitted_count_score":
            requires_calibration = False
        else:
            raise FinalRunnerContractError("frozen candidate score policy is invalid")
        try:
            configuration = AuthorizedConfiguration.create(
                method_id=method_id,
                configuration_id=configuration_id,
                kind="candidate_search",
                payload=payload,
                requires_count_score=True,
                requires_calibration=requires_calibration,
                configuration_sha256=str(digest),
            )
            from .runner import (
                maskimpute_decoder_for_configuration,
                maskimpute_structure_for_configuration,
                maskimpute_variant_for_configuration,
            )

            maskimpute_variant_for_configuration(configuration)
            maskimpute_decoder_for_configuration(configuration)
            maskimpute_structure_for_configuration(configuration)
        except (RunnerContractError, TypeError, ValueError) as error:
            raise FinalRunnerContractError(
                "frozen candidate configuration is not executable"
            ) from error
        return configuration
    if method_id == "capacity-matched-ae":
        control = frozen_method.get("selected_ablation_control")
        if not isinstance(control, Mapping):
            raise FinalRunnerContractError("frozen capacity control is invalid")
        payload = control.get("capacity_matched_definition")
        digest = control.get("capacity_matched_definition_sha256")
        if (
            control.get("capacity_matched_control_id") != "capacity-matched-ae"
            or not isinstance(payload, Mapping)
            or canonical_sha256(payload) != digest
        ):
            raise FinalRunnerContractError("frozen capacity control is invalid")
        return AuthorizedConfiguration.create(
            method_id=method_id,
            configuration_id="capacity-matched-ae",
            kind="ablation",
            payload=payload,
            requires_count_score=True,
            requires_calibration=True,
            configuration_sha256=str(digest),
        )
    from .methods.base import MethodSpec

    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if method_id in _canonical_comparator_tuning_authority().method_order:
        raise FinalRunnerContractError(
            "selected comparator configuration must come from the frozen receipt"
        )
    return AuthorizedConfiguration.registry_default(spec)


def _frozen_comparator_projection(
    frozen_method: Mapping[str, object],
) -> tuple[object, ComparatorSelectionProjection]:
    """Recompute and close the complete direct selection handoff."""

    authority = _canonical_comparator_tuning_authority()
    frozen_authority = frozen_method.get("comparator_tuning_authority")
    selection = frozen_method.get("comparator_selection")
    if not isinstance(selection, Mapping) or not isinstance(
        selection.get("receipt"), Mapping
    ):
        raise FinalRunnerContractError("frozen comparator selection is incomplete")
    try:
        projection = comparator_selection_projection(selection["receipt"])
    except (ComparatorTuningError, TypeError, ValueError) as error:
        raise FinalRunnerContractError(
            "frozen comparator selection is invalid"
        ) from error
    if (
        not direct_equal(frozen_authority, direct_json_value(authority))
        or not direct_equal(
            selection,
            comparator_selection_projection_value(projection),
        )
        or frozen_method.get("scheduled_same_input_ids")
        != list(authority.scheduled_same_input_ids)
        or frozen_method.get("required_control_ids")
        != list(authority.required_control_ids)
        or frozen_method.get("established_comparator_ids")
        != list(authority.established_comparator_ids)
        or frozen_method.get("modern_core_ids") != list(authority.modern_core_ids)
        or frozen_method.get("ready_comparison_population_ids")
        != list(projection.ready_comparison_population_ids)
    ):
        raise FinalRunnerContractError(
            "frozen comparator selection differs from its complete receipt"
        )
    selected = frozen_method.get("selected_comparator_configurations")
    unavailable = frozen_method.get("unavailable_comparator_nonexecution_identities")
    if not isinstance(selected, Mapping) or not isinstance(unavailable, Mapping):
        raise FinalRunnerContractError("frozen comparator maps are incomplete")
    if (
        set(selected) & set(unavailable)
        or set(selected) | set(unavailable) != set(authority.method_order)
        or set(selected) != set(projection.selected_by_method)
        or set(unavailable) != set(projection.nonexecution_identity_by_method)
    ):
        raise FinalRunnerContractError(
            "frozen comparator selected/nonexecution maps are incomplete"
        )
    for method_id, value in projection.selected_by_method.items():
        if not direct_equal(
            selected.get(method_id),
            direct_bound_comparator_value(value.configuration),
        ):
            raise FinalRunnerContractError(
                f"frozen selected comparator {method_id} differs"
            )
    for method_id, value in projection.nonexecution_identity_by_method.items():
        if not direct_equal(
            unavailable.get(method_id),
            direct_json_value(value, payload=True),
        ):
            raise FinalRunnerContractError(
                f"frozen comparator {method_id} nonexecution identity differs"
            )
    statuses = frozen_method.get("scheduled_same_input_statuses")
    if (
        not isinstance(statuses, list)
        or tuple(
            row.get("method_id") if isinstance(row, Mapping) else None
            for row in statuses
        )
        != authority.scheduled_same_input_ids
    ):
        raise FinalRunnerContractError(
            "frozen scheduled same-input status denominator differs"
        )
    status_by_id = {
        str(row["method_id"]): row for row in statuses if isinstance(row, Mapping)
    }
    for method_id in authority.method_order:
        row = status_by_id.get(method_id)
        if not isinstance(row, Mapping):
            raise FinalRunnerContractError(
                f"frozen comparator {method_id} status is absent"
            )
        selected_value = projection.selected_by_method.get(method_id)
        unavailable_value = projection.nonexecution_identity_by_method.get(method_id)
        if not direct_equal(
            row.get("selected_comparator_configuration"),
            (
                None
                if selected_value is None
                else direct_bound_comparator_value(selected_value.configuration)
            ),
        ) or not direct_equal(
            row.get("nonexecution_identity"),
            (
                None
                if unavailable_value is None
                else direct_json_value(unavailable_value, payload=True)
            ),
        ):
            raise FinalRunnerContractError(
                f"frozen comparator {method_id} status identity differs"
            )
    return authority, projection


def _frozen_method_plan_authority(
    frozen_method: Mapping[str, object],
    registry: MethodRegistry,
) -> tuple[
    dict[str, Mapping[str, object]],
    tuple[FrozenPlanMethodAuthority, ...],
]:
    """Validate the one frozen method/configuration/applicability authority."""

    if not isinstance(frozen_method, Mapping):
        raise TypeError("frozen_method must be a mapping")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    unsigned = {
        key: value for key, value in frozen_method.items() if key != "payload_sha256"
    }
    if (
        frozen_method.get("schema_version") != 1
        or frozen_method.get("candidate_method_id") != "maskimpute"
        or frozen_method.get("payload_sha256") != canonical_sha256(unsigned)
    ):
        raise FinalRunnerContractError("frozen method receipt is invalid")
    denominator = frozen_method.get("method_denominator")
    if not isinstance(denominator, list) or len(denominator) != len(registry.methods):
        raise FinalRunnerContractError("frozen method denominator is incomplete")
    frozen_by_id: dict[str, Mapping[str, object]] = {}
    for spec, row in zip(registry.methods, denominator, strict=True):
        if not isinstance(row, Mapping) or row.get("id") != spec.id:
            raise FinalRunnerContractError("frozen method order differs from registry")
        if row.get("method_sha256") != canonical_sha256(asdict(spec)):
            raise FinalRunnerContractError(f"method {spec.id} differs from receipt")
        frozen_by_id[spec.id] = row
    registry_payload = {
        "schema_version": registry.schema_version,
        "methods": [asdict(spec) for spec in registry.methods],
    }
    if frozen_method.get("method_registry_sha256") != canonical_sha256(
        registry_payload
    ):
        raise FinalRunnerContractError("method registry differs from frozen receipt")
    comparator_authority, projection = _frozen_comparator_projection(frozen_method)
    configurations: list[FrozenPlanMethodAuthority] = []
    for spec in registry.methods:
        row = frozen_by_id[spec.id]
        legacy_configuration: AuthorizedConfiguration | None = None
        comparator_configuration: BoundComparatorConfiguration | None = None
        comparator_nonexecution_identity: Mapping[str, object] | None = None
        if spec.id in comparator_authority.method_order:
            selected = projection.selected_by_method.get(spec.id)
            nonexecution = projection.nonexecution_identity_by_method.get(spec.id)
            if selected is not None:
                comparator_configuration = selected.configuration
                if (
                    not direct_equal(
                        comparator_configuration.method,
                        comparator_method_binding(spec),
                    )
                    or not direct_equal(
                        row.get("selected_comparator_configuration"),
                        direct_bound_comparator_value(comparator_configuration),
                    )
                    or row.get("nonexecution_identity") is not None
                ):
                    raise FinalRunnerContractError(
                        f"frozen selected comparator {spec.id} differs from registry"
                    )
            elif nonexecution is not None:
                comparator_nonexecution_identity = nonexecution
                if (
                    not direct_equal(
                        row.get("nonexecution_identity"),
                        direct_json_value(nonexecution, payload=True),
                    )
                    or row.get("selected_comparator_configuration") is not None
                ):
                    raise FinalRunnerContractError(
                        f"frozen comparator {spec.id} nonexecution differs"
                    )
            else:  # pragma: no cover - complete projection checked above
                raise AssertionError("comparator method projection is absent")
        else:
            legacy_configuration = _configuration_for_method(
                spec.id,
                spec,
                frozen_method,
            )
        kind = (
            legacy_configuration.kind
            if legacy_configuration is not None
            else (
                "comparator_tuning"
                if comparator_configuration is not None
                else "comparator_nonexecution"
            )
        )
        action, reason, seeds = _frozen_final_applicability(spec, row, kind)
        if comparator_configuration is not None and (
            action != "execute" or reason is not None
        ):
            raise FinalRunnerContractError(
                f"selected comparator {spec.id} must retain its executable disposition"
            )
        if comparator_nonexecution_identity is not None and (
            action != "not_applicable" or reason is None
        ):
            raise FinalRunnerContractError(
                f"comparator {spec.id} nonexecution disposition is invalid"
            )
        configurations.append(
            FrozenPlanMethodAuthority(
                method_id=spec.id,
                legacy_configuration=legacy_configuration,
                comparator_configuration=comparator_configuration,
                comparator_nonexecution_identity=(comparator_nonexecution_identity),
                action=action,
                reason=reason,
                seeds=seeds,
            )
        )
    return frozen_by_id, tuple(configurations)


def _frozen_final_applicability(
    spec: object,
    row: Mapping[str, object],
    configuration_kind: str,
) -> tuple[
    Literal["execute", "not_applicable"],
    str | None,
    tuple[int | None, ...],
]:
    """Derive one frozen applicability disposition with no caller override."""

    from .methods.base import MethodSpec

    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    applicability = row.get("final_applicability")
    if not isinstance(applicability, Mapping) or set(applicability) != {
        "rule",
        "non_run_reason",
        "required_reference",
    }:
        raise FinalRunnerContractError(
            f"method {spec.id} final applicability is invalid"
        )
    rule = applicability.get("rule")
    integration_status = row.get("integration_status")
    if rule == "all_final_datasets":
        if (
            integration_status != "implemented"
            or applicability.get("non_run_reason") is not None
            or applicability.get("required_reference") is not None
        ):
            raise FinalRunnerContractError(
                f"method {spec.id} executable disposition is not implemented"
            )
        return (
            "execute",
            None,
            DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,),
        )
    if rule not in {"never", "matched_bulk_reference_present"}:
        raise FinalRunnerContractError(
            f"method {spec.id} has an unknown final applicability rule"
        )
    raw_reason = applicability.get("non_run_reason")
    if not isinstance(raw_reason, str) or not raw_reason:
        raise FinalRunnerContractError(f"method {spec.id} lacks a final non-run reason")
    if rule == "never":
        if applicability.get("required_reference") is not None:
            raise FinalRunnerContractError(
                f"method {spec.id} non-run reference binding is invalid"
            )
        if raw_reason == "historical_method_not_rerun":
            if integration_status != "historical":
                raise FinalRunnerContractError(
                    f"method {spec.id} historical integration status differs"
                )
        elif integration_status != "unavailable":
            raise FinalRunnerContractError(
                f"method {spec.id} unavailable integration status differs"
            )
    else:
        required_reference = applicability.get("required_reference")
        if required_reference != {
            "kind": "prespecified_matched_bulk_expression",
            "binding": "final_dataset_manifest_external_reference",
            "evaluator_truth_as_reference": "forbidden",
        }:
            raise FinalRunnerContractError(
                f"method {spec.id} matched-bulk reference binding is invalid"
            )
        if (
            integration_status != "implemented"
            or raw_reason != "matched_bulk_reference_absent"
        ):
            raise FinalRunnerContractError(
                f"method {spec.id} matched-bulk disposition is invalid"
            )
    if (
        rule == "never"
        and spec.execution_scope == "same_input_required"
        and configuration_kind == "comparator_nonexecution"
    ):
        return (
            "not_applicable",
            raw_reason,
            DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,),
        )
    return "not_applicable", raw_reason, (None,)


def _production_registry_matches(registry: MethodRegistry) -> bool:
    """Identify the frozen production denominator by complete execution fields."""

    from .methods.registry import load_method_registry

    canonical = load_method_registry(
        Path(__file__).resolve().parents[1] / "study/methods.json"
    )
    return (
        registry.ids == canonical.ids
        and len(registry.methods) == len(canonical.methods)
        and all(
            direct_equal(
                comparator_method_binding(observed),
                comparator_method_binding(expected),
            )
            for observed, expected in zip(
                registry.methods,
                canonical.methods,
                strict=True,
            )
        )
    )


def build_final_execution_plan(
    frozen_method: Mapping[str, object],
    registry: MethodRegistry,
    datasets: Sequence[DatasetBinding],
    *,
    execution_claim_sha256: str,
    execution_environment_sha256: str,
    execution_authority_sha256: str,
) -> FinalExecutionPlan:
    """Derive the final denominator without accepting design or tuning overrides."""

    frozen_by_id, configurations = _frozen_method_plan_authority(
        frozen_method,
        registry,
    )
    dataset_values = tuple(datasets)
    if len(dataset_values) != 40 or not all(
        isinstance(value, DatasetBinding) for value in dataset_values
    ):
        raise FinalRunnerContractError("final plan requires exactly 40 datasets")
    for name, values in (
        ("manifest", {item.manifest_sha256 for item in dataset_values}),
        ("protocol", {item.protocol_sha256 for item in dataset_values}),
        ("design", {item.design_sha256 for item in dataset_values}),
        ("seed", {item.seed_source_sha256 for item in dataset_values}),
    ):
        if len(values) != 1:
            raise FinalRunnerContractError(f"final dataset {name} authority differs")
    claim_sha256 = _sha256(execution_claim_sha256, "execution claim")
    environment_sha256 = _sha256(execution_environment_sha256, "execution environment")
    authority_sha256 = _sha256(execution_authority_sha256, "execution authority")
    configuration_by_id = {
        configuration.method_id: configuration for configuration in configurations
    }
    entries: list[FinalPlanEntry] = []
    ordinal = 0
    for binding in dataset_values:
        for spec in registry.methods:
            configuration = configuration_by_id[spec.id]
            for seed in configuration.seeds:
                ordinal += 1
                legacy = configuration.legacy_configuration
                run = RunPlanEntry(
                    ordinal=ordinal,
                    run_id=_final_run_id(
                        ordinal,
                        spec.id,
                        binding.dataset_id,
                        seed,
                        configuration,
                    ),
                    method_id=spec.id,
                    dataset_id=binding.dataset_id,
                    source_dataset_sha256=binding.dataset_sha256,
                    mechanism=binding.mechanism,
                    biological_id=binding.biological_id,
                    technical_view=binding.technical_view,
                    model_seed=seed,
                    configuration_id=configuration.configuration_id,
                    configuration_sha256=(
                        None if legacy is None else legacy.configuration_sha256
                    ),
                    preflight_status="planned",
                    preflight_reason=None,
                    configuration_kind=configuration.kind,
                    requires_count_score=configuration.requires_count_score,
                    requires_calibration=configuration.requires_calibration,
                    comparator_configuration=(configuration.comparator_configuration),
                    comparator_nonexecution_identity=(
                        configuration.comparator_nonexecution_identity
                    ),
                )
                entries.append(
                    FinalPlanEntry(
                        run=run,
                        action=configuration.action,
                        reason=configuration.reason,
                    )
                )
    if _production_registry_matches(registry):
        if len(entries) != 1_760:
            raise FinalRunnerContractError(
                "final structural denominator must equal 1760"
            )
        all_comparators_selected = not any(
            configuration.comparator_nonexecution_identity is not None
            for configuration in configurations
        )
        execute_count = sum(entry.action == "execute" for entry in entries)
        not_applicable_count = sum(
            entry.action == "not_applicable" for entry in entries
        )
        if all_comparators_selected and (
            execute_count != 1_480 or not_applicable_count != 280
        ):
            raise FinalRunnerContractError(
                "all-selected production final denominator must contain "
                "1480 executable and 280 not-applicable rows"
            )
    input_hashes = {
        "frozen_method_sha256": str(frozen_method["payload_sha256"]),
        "method_registry_sha256": str(frozen_method["method_registry_sha256"]),
        "runtime_lock_sha256": _sha256(
            frozen_method.get("runtime_lock_sha256"), "frozen runtime lock"
        ),
        "dataset_manifest_sha256": dataset_values[0].manifest_sha256,
        "dataset_design_sha256": dataset_values[0].design_sha256,
        "dataset_seed_source_sha256": dataset_values[0].seed_source_sha256,
        "protocol_sha256": dataset_values[0].protocol_sha256,
        "execution_claim_sha256": claim_sha256,
        "execution_environment_sha256": environment_sha256,
        "execution_authority_sha256": authority_sha256,
    }
    body = {
        "schema_version": 1,
        "input_hashes": input_hashes,
        "entries": [entry.to_dict() for entry in entries],
        "configurations": [value.to_dict() for value in configurations],
        "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
    }
    return FinalExecutionPlan(
        schema_version=1,
        input_hashes=MappingProxyType(input_hashes),
        entries=tuple(entries),
        configurations=configurations,
        plan_sha256=canonical_sha256(body),
    )


def _trajectory_run_id(
    ordinal: int,
    method_id: str,
    dataset_id: str,
    model_seed: int | None,
    configuration: FrozenPlanMethodAuthority,
) -> str:
    seed = "deterministic" if model_seed is None else f"seed-{model_seed}"
    if configuration.kind in {"comparator_tuning", "comparator_nonexecution"}:
        return (
            f"trajectory-{ordinal:02d}-{method_id}-{dataset_id}-{seed}-"
            f"{configuration.configuration_id}"
        )
    assert configuration.legacy_configuration is not None
    return (
        f"trajectory-{method_id}-{dataset_id}-{seed}-"
        f"{configuration.legacy_configuration.configuration_sha256[:12]}"
    )


def build_trajectory_execution_plan(
    frozen_method: Mapping[str, object],
    registry: MethodRegistry,
    registered: TrajectoryPreparedDataset,
    *,
    execution_claim_sha256: str,
    execution_environment_sha256: str,
    execution_authority_sha256: str,
    primary_final_plan_sha256: str,
) -> TrajectoryExecutionPlan:
    """Derive the exact supplementary denominator from frozen authorities only."""

    frozen_by_id, configurations = _frozen_method_plan_authority(
        frozen_method,
        registry,
    )
    if not isinstance(registered, TrajectoryPreparedDataset):
        raise TypeError("registered must be a TrajectoryPreparedDataset")
    binding = registered.binding
    prepared = registered.prepared
    if not isinstance(binding, RegisteredTrajectoryBinding) or not isinstance(
        prepared, PreparedDataset
    ):
        raise FinalRunnerContractError("registered trajectory dataset is invalid")
    expected_binding = _trajectory_binding(
        registered.authority,
        authority_file_sha256=binding.authority_file_sha256,
        dataset_file_sha256=binding.dataset_file_sha256,
    )
    expected_receipt = _trajectory_dataset_receipt(binding, prepared)
    expected_receipt_raw = _canonical_bytes(expected_receipt) + b"\n"
    if (
        binding != expected_binding
        or prepared.binding != binding
        or prepared.method_input.source_dataset_sha256 != binding.dataset_sha256
        or prepared.method_input.shape != (binding.cells, binding.genes)
        or prepared.method_input.obs_ids != prepared.audit.retained_cell_ids
        or dict(registered.receipt) != expected_receipt
        or registered.receipt_file_path
        != "results/trajectory/dataset/dataset_receipt.json"
        or registered.receipt_file_sha256
        != hashlib.sha256(expected_receipt_raw).hexdigest()
    ):
        raise FinalRunnerContractError("registered trajectory identity differs")
    claim_sha256 = _sha256(execution_claim_sha256, "execution claim")
    environment_sha256 = _sha256(
        execution_environment_sha256,
        "execution environment",
    )
    authority_sha256 = _sha256(
        execution_authority_sha256,
        "trajectory execution authority",
    )
    primary_plan_sha256 = _sha256(
        primary_final_plan_sha256,
        "primary final plan",
    )
    configuration_by_id = {
        configuration.method_id: configuration for configuration in configurations
    }
    entries: list[FinalPlanEntry] = []
    ordinal = 0
    for spec in registry.methods:
        configuration = configuration_by_id[spec.id]
        for seed in configuration.seeds:
            ordinal += 1
            legacy = configuration.legacy_configuration
            run = RunPlanEntry(
                ordinal=ordinal,
                run_id=_trajectory_run_id(
                    ordinal,
                    spec.id,
                    binding.dataset_id,
                    seed,
                    configuration,
                ),
                method_id=spec.id,
                dataset_id=binding.dataset_id,
                source_dataset_sha256=binding.dataset_sha256,
                mechanism=binding.mechanism,
                biological_id=binding.biological_id,
                technical_view=binding.technical_view,
                model_seed=seed,
                configuration_id=configuration.configuration_id,
                configuration_sha256=(
                    None if legacy is None else legacy.configuration_sha256
                ),
                preflight_status="planned",
                preflight_reason=None,
                configuration_kind=configuration.kind,
                requires_count_score=configuration.requires_count_score,
                requires_calibration=configuration.requires_calibration,
                comparator_configuration=configuration.comparator_configuration,
                comparator_nonexecution_identity=(
                    configuration.comparator_nonexecution_identity
                ),
            )
            entries.append(
                FinalPlanEntry(
                    run=run,
                    action=configuration.action,
                    reason=configuration.reason,
                )
            )
    if _production_registry_matches(registry) and len(entries) != 44:
        raise FinalRunnerContractError(
            "trajectory structural denominator must equal 44"
        )
    input_hashes = {
        "frozen_method_sha256": str(frozen_method["payload_sha256"]),
        "method_registry_sha256": str(frozen_method["method_registry_sha256"]),
        "runtime_lock_sha256": _sha256(
            frozen_method.get("runtime_lock_sha256"),
            "frozen runtime lock",
        ),
        "primary_final_plan_sha256": primary_plan_sha256,
        "trajectory_authority_file_sha256": binding.authority_file_sha256,
        "trajectory_authority_sha256": binding.authority_sha256,
        "trajectory_binding_sha256": binding.registered_binding_sha256,
        "trajectory_dataset_sha256": binding.dataset_sha256,
        "trajectory_dataset_file_sha256": binding.dataset_file_sha256,
        "trajectory_dataset_receipt_sha256": str(registered.receipt["receipt_sha256"]),
        "trajectory_dataset_receipt_file_sha256": (registered.receipt_file_sha256),
        "trajectory_method_input_sha256": method_input_sha256(prepared.method_input),
        "trajectory_retained_cell_ids_sha256": (
            prepared.audit.retained_cell_ids_sha256
        ),
        "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
        "execution_claim_sha256": claim_sha256,
        "execution_environment_sha256": environment_sha256,
        "execution_authority_sha256": authority_sha256,
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "scope": "supplementary_trajectory",
        "input_hashes": input_hashes,
        "entries": [entry.to_dict() for entry in entries],
        "configurations": [value.to_dict() for value in configurations],
        "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
    }
    return TrajectoryExecutionPlan(
        schema_version=1,
        scope="supplementary_trajectory",
        input_hashes=MappingProxyType(input_hashes),
        entries=tuple(entries),
        configurations=configurations,
        plan_sha256=canonical_sha256(body),
    )


def _materialize_final_execution_inputs(
    repository: Path,
    round_dir: Path,
    frozen_method: Mapping[str, object],
    registry: MethodRegistry,
    bindings: tuple[DatasetBinding, ...],
    *,
    execution_claim_sha256: str,
    execution_environment_sha256: str,
    publish_results: Callable[[], object],
) -> tuple[
    TrajectoryPreparedDataset,
    ExecutionAuthorityContext,
    FinalExecutionPlan,
    ExecutionAuthorityContext,
    TrajectoryExecutionPlan,
]:
    """Materialize and durably journal each immutable final input stage."""

    if not callable(publish_results):
        raise TypeError("publish_results must be callable")
    provisional_plan = build_final_execution_plan(
        frozen_method,
        registry,
        bindings,
        execution_claim_sha256=execution_claim_sha256,
        execution_environment_sha256=execution_environment_sha256,
        execution_authority_sha256="0" * 64,
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    publish_results()
    authority = materialize_final_execution_authority(
        repository,
        round_dir,
        frozen_method,
        execution_claim_sha256=execution_claim_sha256,
        execution_environment_sha256=execution_environment_sha256,
        dataset_manifest_sha256=bindings[0].manifest_sha256,
    )
    publish_results()
    plan = build_final_execution_plan(
        frozen_method,
        registry,
        bindings,
        execution_claim_sha256=execution_claim_sha256,
        execution_environment_sha256=execution_environment_sha256,
        execution_authority_sha256=authority.authority_sha256,
    )
    if getattr(plan, "entries", None) != getattr(
        provisional_plan, "entries", None
    ) or getattr(plan, "configurations", None) != getattr(
        provisional_plan, "configurations", None
    ):
        raise FinalRunnerContractError(
            "final plan changed while materializing execution authority"
        )
    trajectory_authority = materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen_method,
        registered,
        authority,
        execution_claim_sha256=execution_claim_sha256,
        execution_environment_sha256=execution_environment_sha256,
        primary_final_plan_sha256=plan.plan_sha256,
    )
    publish_results()
    trajectory_plan = build_trajectory_execution_plan(
        frozen_method,
        registry,
        registered,
        execution_claim_sha256=execution_claim_sha256,
        execution_environment_sha256=execution_environment_sha256,
        execution_authority_sha256=trajectory_authority.authority_sha256,
        primary_final_plan_sha256=plan.plan_sha256,
    )
    return registered, authority, plan, trajectory_authority, trajectory_plan


def trajectory_execution_plan_payload(
    plan: TrajectoryExecutionPlan,
) -> dict[str, object]:
    """Return and verify the complete registered-trajectory plan payload."""

    if not isinstance(plan, TrajectoryExecutionPlan):
        raise TypeError("plan must be a TrajectoryExecutionPlan")
    body: dict[str, object] = {
        "schema_version": plan.schema_version,
        "scope": plan.scope,
        "input_hashes": dict(plan.input_hashes),
        "entries": [entry.to_dict() for entry in plan.entries],
        "configurations": [value.to_dict() for value in plan.configurations],
        "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
    }
    if plan.schema_version != 1 or canonical_sha256(body) != plan.plan_sha256:
        raise FinalRunnerContractError("trajectory plan payload binding is invalid")
    return {**body, "plan_sha256": plan.plan_sha256}


def _execute_frozen_plan(
    plan: FinalExecutionPlan | TrajectoryExecutionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    authority: ExecutionAuthorityContext,
    executor: Callable[
        [ExecutionRequest | FinalComparatorExecutionRequest],
        AdapterOutcome,
    ],
    store: FinalResultStore,
    *,
    on_record_published: Callable[[], object],
) -> dict[str, object]:
    """Execute/resume an exact plan and journal its complete immutable manifest."""

    if not isinstance(plan, (FinalExecutionPlan, TrajectoryExecutionPlan)):
        raise TypeError("plan must be a FinalExecutionPlan or TrajectoryExecutionPlan")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(prepared_datasets, Mapping):
        raise TypeError("prepared_datasets must be a mapping")
    if not isinstance(authority, ExecutionAuthorityContext):
        raise TypeError("authority must be an ExecutionAuthorityContext")
    if not callable(executor) or not callable(on_record_published):
        raise TypeError("executor and publication callback must be callable")
    if not isinstance(store, FinalResultStore) or store.plan != plan:
        raise FinalRunnerContractError("final result store uses a different plan")
    if (
        plan.input_hashes.get("execution_authority_sha256")
        != authority.authority_sha256
        or store.execution_authority != authority
    ):
        raise FinalRunnerContractError("final execution authority differs from plan")
    if os.path.lexists(store.manifest_path):
        return store.load_manifest()
    existing = store._cached_records()
    configuration_by_method = {value.method_id: value for value in plan.configurations}
    if set(configuration_by_method) != set(registry.ids):
        raise FinalRunnerContractError("final configurations differ from denominator")
    for plan_entry in plan.entries[len(existing) :]:
        try:
            spec = registry.by_id(plan_entry.run.method_id)
            prepared = prepared_datasets[plan_entry.run.dataset_id]
            configuration = configuration_by_method[plan_entry.run.method_id]
        except KeyError as error:
            raise FinalRunnerContractError(
                "final plan method or dataset is unavailable"
            ) from error
        if plan_entry.action == "not_applicable":
            assert plan_entry.reason is not None
            outcome = AdapterOutcome.unavailable(plan_entry.reason)
        else:
            if configuration.comparator_configuration is not None:
                request: ExecutionRequest | FinalComparatorExecutionRequest = (
                    FinalComparatorExecutionRequest.create(
                        spec,
                        prepared.method_input,
                        model_seed=plan_entry.run.model_seed,
                        configuration=configuration.comparator_configuration,
                        authority=authority,
                        mechanism=plan_entry.run.mechanism,
                        biological_id=plan_entry.run.biological_id,
                        technical_view=plan_entry.run.technical_view,
                        dataset_id=plan_entry.run.dataset_id,
                        timeout_seconds=spec.resources.timeout_seconds,
                    )
                )
            else:
                legacy = configuration.legacy_configuration
                if legacy is None:  # pragma: no cover - action/config invariant
                    raise AssertionError("executable final configuration is absent")
                request = ExecutionRequest.create(
                    spec,
                    prepared.method_input,
                    model_seed=plan_entry.run.model_seed,
                    configuration=legacy,
                    authority=authority,
                    mechanism=plan_entry.run.mechanism,
                    biological_id=plan_entry.run.biological_id,
                    technical_view=plan_entry.run.technical_view,
                    dataset_id=plan_entry.run.dataset_id,
                    timeout_seconds=spec.resources.timeout_seconds,
                    calibration_usage="retained_all_development",
                )
            outcome = executor(request)
            if not isinstance(outcome, AdapterOutcome):
                raise FinalRunnerContractError(
                    "final adapter returned a noncanonical outcome"
                )
            if isinstance(request, ExecutionRequest):
                outcome = enforce_calibration_fold_receipt(request, outcome)
            if outcome.status in {
                "infrastructure_error",
                "blocked_authority",
                "budget_exhausted",
            }:
                raise FinalRunnerContractError(
                    "retryable infrastructure or authority failure was not sealed"
                )
        attempt = evaluate_adapter_outcome(
            plan_entry.run,
            prepared,
            outcome,
        )
        store.append(
            plan_entry,
            attempt,
            execution_request=(
                None if plan_entry.action == "not_applicable" else request
            ),
        )
    manifest = store.finalize()
    on_record_published()
    return manifest


def execute_final_plan(
    plan: FinalExecutionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    authority: ExecutionAuthorityContext,
    executor: Callable[
        [ExecutionRequest | FinalComparatorExecutionRequest],
        AdapterOutcome,
    ],
    store: FinalResultStore,
    *,
    on_record_published: Callable[[], object],
) -> dict[str, object]:
    """Execute/resume the strict 40-dataset primary final plan."""

    if not isinstance(plan, FinalExecutionPlan):
        raise TypeError("plan must be a FinalExecutionPlan")
    return _execute_frozen_plan(
        plan,
        registry,
        prepared_datasets,
        authority,
        executor,
        store,
        on_record_published=on_record_published,
    )


def execute_trajectory_plan(
    plan: TrajectoryExecutionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    authority: ExecutionAuthorityContext,
    executor: Callable[
        [ExecutionRequest | FinalComparatorExecutionRequest],
        AdapterOutcome,
    ],
    store: FinalResultStore,
    *,
    on_record_published: Callable[[], object],
) -> dict[str, object]:
    """Execute/resume the separately scoped registered trajectory plan."""

    if not isinstance(plan, TrajectoryExecutionPlan):
        raise TypeError("plan must be a TrajectoryExecutionPlan")
    return _execute_frozen_plan(
        plan,
        registry,
        prepared_datasets,
        authority,
        executor,
        store,
        on_record_published=on_record_published,
    )


def _validate_frozen_execution_for_evaluation(
    plan: FinalExecutionPlan | TrajectoryExecutionPlan,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require a complete terminal denominator without success-conditioned exclusion."""

    from .runner import _RECONSTRUCTION_METRIC_NAMES

    if not isinstance(plan, (FinalExecutionPlan, TrajectoryExecutionPlan)):
        raise TypeError("plan must be a FinalExecutionPlan or TrajectoryExecutionPlan")
    values = tuple(records)
    if len(values) != len(plan.entries):
        raise FinalRunnerContractError(
            "final execution record denominator is incomplete"
        )
    completed = 0
    algorithmic_failures = 0
    nonruns = 0
    executed_status_counts: dict[str, int] = {}
    record_sha256s: list[str] = []
    for plan_entry, record in zip(plan.entries, values, strict=True):
        if not isinstance(record, Mapping) or not isinstance(
            record.get("run"), Mapping
        ):
            raise FinalRunnerContractError("final execution record is invalid")
        run = record["run"]
        metrics = record.get("metrics")
        if (
            run.get("run_id") != plan_entry.run.run_id
            or run.get("method_id") != plan_entry.run.method_id
            or run.get("dataset_id") != plan_entry.run.dataset_id
            or run.get("model_seed") != plan_entry.run.model_seed
            or run.get("configuration_sha256") != plan_entry.run.configuration_sha256
        ):
            raise FinalRunnerContractError(
                "final execution record differs from its evaluation plan"
            )
        if (
            not isinstance(metrics, list)
            or len(metrics) != len(_RECONSTRUCTION_METRIC_NAMES)
            or any(
                not isinstance(metric, Mapping) or metric.get("metric") != expected_name
                for metric, expected_name in zip(
                    metrics, _RECONSTRUCTION_METRIC_NAMES, strict=True
                )
            )
        ):
            raise FinalRunnerContractError(
                "final execution metric denominator is incomplete"
            )

        def require_reason_coded_metrics(status: str, reason: object) -> None:
            if not isinstance(reason, str) or not reason:
                raise FinalRunnerContractError(
                    "terminal final failure lacks a reason code"
                )
            if any(
                metric.get("status") != status
                or metric.get("reason") != reason
                or metric.get("value") is not None
                or metric.get("n") != 0
                for metric in metrics
            ):
                raise FinalRunnerContractError(
                    "terminal final failure lacks complete reason-coded metric rows"
                )

        if plan_entry.action == "execute":
            status = run.get("status")
            if status == "completed":
                if run.get("reason") is not None:
                    raise FinalRunnerContractError(
                        "completed final execution has a failure reason"
                    )
                completed += 1
            elif status in {"failed", "timeout", "resource_exceeded", "unavailable"}:
                require_reason_coded_metrics(str(status), run.get("reason"))
                algorithmic_failures += 1
            elif status in {
                "infrastructure_error",
                "blocked_authority",
                "budget_exhausted",
            }:
                raise FinalRunnerContractError(
                    "final infrastructure or authority incompleteness blocks evaluation"
                )
            else:
                raise FinalRunnerContractError(
                    "final executable record has a nonterminal status"
                )
            assert isinstance(status, str)
            executed_status_counts[status] = executed_status_counts.get(status, 0) + 1
        else:
            if (
                run.get("status") != "unavailable"
                or run.get("reason") != plan_entry.reason
            ):
                raise FinalRunnerContractError(
                    "final non-run record differs from its frozen applicability reason"
                )
            require_reason_coded_metrics("unavailable", plan_entry.reason)
            nonruns += 1
        record_sha256s.append(canonical_sha256(record))
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "eligible_for_final_evaluation_complete_terminal_denominator",
        "planned_run_count": len(plan.entries),
        "executed_completed_count": completed,
        "executed_algorithmic_failure_count": algorithmic_failures,
        "executed_status_counts": {
            status: executed_status_counts[status]
            for status in sorted(executed_status_counts)
        },
        "not_applicable_count": nonruns,
        "record_payload_sha256s": record_sha256s,
    }
    if isinstance(plan, FinalExecutionPlan):
        body["final_plan_sha256"] = plan.plan_sha256
    else:
        body["scope"] = plan.scope
        body["trajectory_plan_sha256"] = plan.plan_sha256
    return {**body, "validation_sha256": canonical_sha256(body)}


def validate_final_execution_for_evaluation(
    plan: FinalExecutionPlan,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require the complete terminal primary final denominator."""

    if not isinstance(plan, FinalExecutionPlan):
        raise TypeError("plan must be a FinalExecutionPlan")
    return _validate_frozen_execution_for_evaluation(plan, records)


def validate_trajectory_execution_for_evaluation(
    plan: TrajectoryExecutionPlan,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require the complete terminal supplementary trajectory denominator."""

    if not isinstance(plan, TrajectoryExecutionPlan):
        raise TypeError("plan must be a TrajectoryExecutionPlan")
    return _validate_frozen_execution_for_evaluation(plan, records)


def _trajectory_evaluation_evidence(
    round_dir: Path,
    plan: TrajectoryExecutionPlan,
    registered: TrajectoryPreparedDataset,
    authority: ExecutionAuthorityContext,
    store: FinalResultStore,
    validation: Mapping[str, object],
    result_files: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind every trajectory-only byte and independently derived denominator."""

    if not isinstance(round_dir, Path):
        raise TypeError("round_dir must be a pathlib.Path")
    if not isinstance(plan, TrajectoryExecutionPlan):
        raise TypeError("plan must be a TrajectoryExecutionPlan")
    if not isinstance(registered, TrajectoryPreparedDataset):
        raise TypeError("registered must be a TrajectoryPreparedDataset")
    if not isinstance(authority, ExecutionAuthorityContext):
        raise TypeError("authority must be an ExecutionAuthorityContext")
    if not isinstance(store, FinalResultStore) or store.plan != plan:
        raise FinalRunnerContractError("trajectory evidence store differs from plan")
    expected_validation = validate_trajectory_execution_for_evaluation(
        plan,
        store.load_records(),
    )
    if dict(validation) != expected_validation:
        raise FinalRunnerContractError(
            "trajectory execution validation differs from stored records"
        )
    destination = round_dir.resolve(strict=True)
    observed: dict[str, str] = {}
    for row in result_files:
        if not isinstance(row, Mapping):
            raise FinalRunnerContractError("final result inventory is invalid")
        path = row.get("path")
        digest = row.get("sha256")
        if isinstance(path, str) and path.startswith("results/trajectory/"):
            if path in observed:
                raise FinalRunnerContractError(
                    "trajectory result inventory path is duplicated"
                )
            observed[path] = _sha256(digest, f"trajectory result {path}")
    expected_paths = {
        path
        for path in _owned_final_result_paths(destination)
        if path.startswith("results/trajectory/")
    }
    if set(observed) != expected_paths:
        raise FinalRunnerContractError(
            "trajectory result inventory differs from its owned files"
        )
    trajectory_files = [
        {"path": path, "sha256": observed[path]} for path in sorted(observed)
    ]

    binding = registered.binding
    receipt_path = destination / registered.receipt_file_path
    receipt_raw = _read_unique_file(
        receipt_path,
        "registered trajectory dataset receipt",
        max_bytes=1024 * 1024,
    )
    h5ad_path = destination / binding.dataset_file_path
    h5ad_file_sha256 = _hash_unique_file(
        h5ad_path,
        "registered trajectory evaluator dataset",
    )
    if (
        receipt_raw != _canonical_bytes(dict(registered.receipt)) + b"\n"
        or hashlib.sha256(receipt_raw).hexdigest() != registered.receipt_file_sha256
        or h5ad_file_sha256 != binding.dataset_file_sha256
    ):
        raise FinalRunnerContractError(
            "registered trajectory dataset bytes changed before receipt"
        )
    dataset_evidence: dict[str, object] = {
        "binding": asdict(binding),
        "dataset_path": binding.dataset_file_path,
        "dataset_file_sha256": h5ad_file_sha256,
        "dataset_sha256": binding.dataset_sha256,
        "receipt_path": registered.receipt_file_path,
        "receipt_file_sha256": registered.receipt_file_sha256,
        "receipt_payload_sha256": registered.receipt["receipt_sha256"],
    }

    authority_path = (
        destination / "results/trajectory/execution_authority/authority.json"
    )
    authority_raw = _read_unique_file(
        authority_path,
        "trajectory execution authority",
        max_bytes=1024 * 1024,
    )
    authority_payload = _strict_json(
        authority_raw,
        "trajectory execution authority",
    )
    authority_body = {
        key: value
        for key, value in authority_payload.items()
        if key != "authority_sha256"
    }
    if (
        authority_raw != _canonical_bytes(authority_payload) + b"\n"
        or authority_payload.get("authority_sha256") != authority.authority_sha256
        or canonical_sha256(authority_body) != authority.authority_sha256
    ):
        raise FinalRunnerContractError(
            "trajectory execution authority bytes changed before receipt"
        )
    authority_files = [
        row
        for row in trajectory_files
        if str(row["path"]).startswith("results/trajectory/execution_authority/")
    ]
    if len(authority_files) != 3:
        raise FinalRunnerContractError(
            "trajectory execution authority inventory is incomplete"
        )
    authority_evidence: dict[str, object] = {
        "authority_path": "results/trajectory/execution_authority/authority.json",
        "authority_file_sha256": hashlib.sha256(authority_raw).hexdigest(),
        "authority_sha256": authority.authority_sha256,
        "count_score_authority_path": authority.count_score_manifest_path,
        "count_score_authority_file_sha256": (authority.count_score_manifest_sha256),
        "retained_calibration_path": authority.retained_calibration_path,
        "retained_calibration_file_sha256": (authority.retained_calibration_sha256),
        "files": authority_files,
    }

    manifest = store.load_manifest()
    manifest_raw = _read_unique_file(
        store.manifest_path,
        "trajectory execution manifest",
        max_bytes=32 * 1024 * 1024,
    )
    manifest_relative = store.manifest_path.relative_to(destination).as_posix()
    manifest_evidence: dict[str, object] = {
        "path": manifest_relative,
        "file_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "payload_sha256": manifest["manifest_sha256"],
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "scope": plan.scope,
        "plan": trajectory_execution_plan_payload(plan),
        "dataset": dataset_evidence,
        "execution_authority": authority_evidence,
        "execution_manifest": manifest_evidence,
        "execution_validation": dict(validation),
        "result_files": trajectory_files,
    }
    return {**body, "evidence_sha256": canonical_sha256(body)}


def _validate_trajectory_primary_authority_chain(
    repository: Path,
    round_dir: Path,
    *,
    primary_final_plan_sha256: str,
    simulator_assets_root: Path | None = None,
    simulator_r_environment: Path | None = None,
) -> Mapping[str, str]:
    """Purely validate trajectory provenance back to the claimed primary run."""

    from .methods import load_method_registry
    from .publication_freeze import validate_frozen_method

    selected_repository, destination = _canonical_round(repository, round_dir)
    primary_plan_sha256 = _sha256(
        primary_final_plan_sha256,
        "primary final plan",
    )
    runtime_paths = (simulator_assets_root, simulator_r_environment)
    if any(value is None for value in runtime_paths):
        if any(value is not None for value in runtime_paths):
            raise FinalRunnerContractError(
                "primary revalidation requires both simulator runtime paths"
            )
        runtime_panel_kwargs: dict[str, Path] = {}
    else:
        assert simulator_assets_root is not None
        assert simulator_r_environment is not None
        runtime_panel_kwargs = {
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
        }
    claim_path = destination / "execution_claim.json"
    claim_raw = _read_unique_file(
        claim_path,
        "final execution claim",
        max_bytes=1024 * 1024,
    )
    claim = _strict_json(claim_raw, "final execution claim")
    claim_sha256 = canonical_sha256(claim)
    try:
        frozen_method = validate_frozen_method(selected_repository)
        registry = load_method_registry(selected_repository / "study/methods.json")
        environments = _load_final_execution_environment_registry(
            selected_repository,
            registry,
        )
        _validate_final_runtime_lock(frozen_method, environments)
        environment_sha256 = _sha256(
            environments.registry_sha256,
            "primary execution environment",
        )
        bindings, prepared = load_prepared_final_panel(
            selected_repository,
            destination,
            allow_evaluated=True,
            **runtime_panel_kwargs,
        )
        derived_primary = _derive_final_execution_authority(
            selected_repository,
            destination,
            frozen_method,
            execution_claim_sha256=claim_sha256,
            execution_environment_sha256=environment_sha256,
            dataset_manifest_sha256=bindings[0].manifest_sha256,
        )
    except FinalRunnerContractError:
        raise
    except Exception as error:
        raise FinalRunnerContractError(
            "primary frozen execution inputs cannot be rederived"
        ) from error

    primary_context = derived_primary.context
    primary_authority_sha256 = primary_context.authority_sha256
    primary_path = destination / "results" / derived_primary.authority_relative_path
    primary_raw = _read_unique_file(
        primary_path,
        "primary execution authority",
        max_bytes=1024 * 1024,
    )
    primary_score_raw = _read_repository_authority_file(
        selected_repository,
        primary_context.count_score_manifest_path,
        primary_context.count_score_manifest_sha256,
        "primary count-score authority",
    )
    primary_calibration_raw = _read_repository_authority_file(
        selected_repository,
        primary_context.retained_calibration_path,
        primary_context.retained_calibration_sha256,
        "primary retained calibration authority",
    )
    if (
        primary_raw != derived_primary.authority_raw
        or primary_score_raw != derived_primary.score_raw
        or primary_calibration_raw != derived_primary.calibration_raw
    ):
        raise FinalRunnerContractError(
            "primary execution authority differs from frozen rederivation"
        )
    primary = _strict_json(primary_raw, "primary execution authority")

    try:
        primary_plan = build_final_execution_plan(
            frozen_method,
            registry,
            bindings,
            execution_claim_sha256=claim_sha256,
            execution_environment_sha256=environment_sha256,
            execution_authority_sha256=primary_authority_sha256,
        )
    except Exception as error:
        raise FinalRunnerContractError(
            "primary final execution plan cannot be rederived"
        ) from error
    if not primary_plan.entries or primary_plan.plan_sha256 != primary_plan_sha256:
        raise FinalRunnerContractError(
            "primary final plan digest or denominator differs from frozen rederivation"
        )
    primary_store = FinalResultStore(
        destination / "results/final/execution",
        primary_plan,
        prepared,
        primary_context,
        authority_repository=selected_repository,
    )
    primary_records = primary_store.load_records()
    primary_store._records_cache = primary_records
    primary_manifest = primary_store.load_manifest()
    primary_validation = validate_final_execution_for_evaluation(
        primary_plan,
        primary_records,
    )
    if (
        not primary_records
        or primary_manifest.get("plan_sha256") != primary_plan_sha256
        or primary_manifest.get("input_hashes") != dict(primary_plan.input_hashes)
        or primary_manifest.get("planned_run_count") != len(primary_plan.entries)
        or primary_manifest.get("recorded_run_count") != len(primary_records)
        or primary_validation.get("final_plan_sha256") != primary_plan_sha256
        or primary_validation.get("planned_run_count") != len(primary_records)
    ):
        raise FinalRunnerContractError(
            "primary execution manifest or terminal denominator differs"
        )
    stable_records = primary_store.load_records()
    primary_store._records_cache = stable_records
    stable_manifest = primary_store.load_manifest()
    stable_validation = validate_final_execution_for_evaluation(
        primary_plan,
        stable_records,
    )
    try:
        environments.full_revalidate()
        stable_frozen_method = validate_frozen_method(selected_repository)
        stable_registry = load_method_registry(
            selected_repository / "study/methods.json"
        )
        stable_environments = _load_final_execution_environment_registry(
            selected_repository,
            stable_registry,
        )
        _validate_final_runtime_lock(stable_frozen_method, stable_environments)
        stable_environments.full_revalidate()
        stable_environment_sha256 = _sha256(
            stable_environments.registry_sha256,
            "stable primary execution environment",
        )
        stable_bindings, _stable_prepared = load_prepared_final_panel(
            selected_repository,
            destination,
            allow_evaluated=True,
            **runtime_panel_kwargs,
        )
        stable_derived_primary = _derive_final_execution_authority(
            selected_repository,
            destination,
            stable_frozen_method,
            execution_claim_sha256=claim_sha256,
            execution_environment_sha256=stable_environment_sha256,
            dataset_manifest_sha256=stable_bindings[0].manifest_sha256,
        )
        stable_primary_plan = build_final_execution_plan(
            stable_frozen_method,
            stable_registry,
            stable_bindings,
            execution_claim_sha256=claim_sha256,
            execution_environment_sha256=stable_environment_sha256,
            execution_authority_sha256=(
                stable_derived_primary.context.authority_sha256
            ),
        )
    except FinalRunnerContractError:
        raise
    except Exception as error:
        raise FinalRunnerContractError(
            "primary frozen execution cannot be revalidated after store replay"
        ) from error
    if (
        stable_records != primary_records
        or stable_manifest != primary_manifest
        or stable_validation != primary_validation
        or dict(stable_frozen_method) != dict(frozen_method)
        or stable_registry != registry
        or stable_environments.runtime_lock_sha256 != environments.runtime_lock_sha256
        or stable_environment_sha256 != environment_sha256
        or stable_bindings != bindings
        or stable_derived_primary != derived_primary
        or stable_primary_plan != primary_plan
        or stable_primary_plan.plan_sha256 != primary_plan_sha256
        or _read_unique_file(
            claim_path,
            "final execution claim",
            max_bytes=1024 * 1024,
        )
        != claim_raw
        or _read_unique_file(
            primary_path,
            "primary execution authority",
            max_bytes=1024 * 1024,
        )
        != primary_raw
        or _read_repository_authority_file(
            selected_repository,
            primary_context.count_score_manifest_path,
            primary_context.count_score_manifest_sha256,
            "primary count-score authority",
        )
        != primary_score_raw
        or _read_repository_authority_file(
            selected_repository,
            primary_context.retained_calibration_path,
            primary_context.retained_calibration_sha256,
            "primary retained calibration authority",
        )
        != primary_calibration_raw
    ):
        raise FinalRunnerContractError(
            "primary frozen execution changed during rederivation"
        )

    trajectory_path = (
        destination / "results/trajectory/execution_authority/authority.json"
    )
    trajectory_raw = _read_unique_file(
        trajectory_path,
        "trajectory execution authority",
        max_bytes=1024 * 1024,
    )
    trajectory = _strict_json(trajectory_raw, "trajectory execution authority")
    trajectory_fields = {
        "schema_version",
        "authority_type",
        "scope",
        "frozen_method_sha256",
        "runtime_lock_sha256",
        "primary_final_plan_sha256",
        "primary_execution_authority_sha256",
        "execution_claim_sha256",
        "execution_environment_sha256",
        "trajectory_authority_sha256",
        "trajectory_binding_sha256",
        "trajectory_dataset_sha256",
        "trajectory_method_input_sha256",
        "base_configuration",
        "base_configuration_sha256",
        "count_model_config",
        "count_model_config_sha256",
        "count_score_authority_path",
        "count_score_authority_sha256",
        "retained_calibration_path",
        "retained_calibration_sha256",
        "calibration_usage",
        "authority_sha256",
    }
    trajectory_body = {
        key: value for key, value in trajectory.items() if key != "authority_sha256"
    }
    if (
        set(trajectory) != trajectory_fields
        or trajectory_raw != _canonical_bytes(trajectory) + b"\n"
        or type(trajectory.get("schema_version")) is not int
        or trajectory.get("schema_version") != 1
        or trajectory.get("authority_type") != "maskimpute_frozen_trajectory_execution"
        or trajectory.get("scope") != "supplementary_trajectory"
        or trajectory.get("primary_final_plan_sha256") != primary_plan_sha256
        or trajectory.get("primary_execution_authority_sha256")
        != primary_authority_sha256
        or trajectory.get("execution_claim_sha256") != claim_sha256
        or trajectory.get("execution_environment_sha256") != environment_sha256
        or trajectory.get("frozen_method_sha256") != primary.get("frozen_method_sha256")
        or trajectory.get("runtime_lock_sha256") != primary.get("runtime_lock_sha256")
        or _canonical_bytes(trajectory.get("base_configuration"))
        != _canonical_bytes(primary.get("base_configuration"))
        or trajectory.get("base_configuration_sha256")
        != primary.get("base_configuration_sha256")
        or _canonical_bytes(trajectory.get("count_model_config"))
        != _canonical_bytes(primary.get("count_model_config"))
        or trajectory.get("count_model_config_sha256")
        != primary.get("count_model_config_sha256")
        or trajectory.get("calibration_usage") != "retained_all_development_calibrator"
        or trajectory.get("authority_sha256") != canonical_sha256(trajectory_body)
    ):
        raise FinalRunnerContractError(
            "trajectory authority is not derived from the primary authority"
        )
    trajectory_authority_sha256 = _sha256(
        trajectory.get("authority_sha256"),
        "trajectory execution authority",
    )
    trajectory_score_sha256 = _sha256(
        trajectory.get("count_score_authority_sha256"),
        "trajectory count-score authority file",
    )
    trajectory_calibration_sha256 = _sha256(
        trajectory.get("retained_calibration_sha256"),
        "trajectory retained calibration file",
    )
    trajectory_score_raw = _read_repository_authority_file(
        selected_repository,
        str(trajectory.get("count_score_authority_path")),
        trajectory_score_sha256,
        "trajectory count-score authority",
    )
    trajectory_calibration_raw = _read_repository_authority_file(
        selected_repository,
        str(trajectory.get("retained_calibration_path")),
        trajectory_calibration_sha256,
        "trajectory retained calibration authority",
    )
    trajectory_score = _strict_json(
        trajectory_score_raw,
        "trajectory count-score authority",
    )
    trajectory_score_body = {
        key: value for key, value in trajectory_score.items() if key != "payload_sha256"
    }
    trajectory_score_fields = {
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
        set(trajectory_score) != trajectory_score_fields
        or trajectory_score_raw != _canonical_bytes(trajectory_score) + b"\n"
        or type(trajectory_score.get("schema_version")) is not int
        or trajectory_score.get("schema_version") != 1
        or trajectory_score.get("artifact_type")
        != "maskimpute_trajectory_count_score_authority"
        or trajectory_score.get("status") != "ready"
        or trajectory_score.get("scope") != "truth_free_registered_trajectory_inference"
        or trajectory_score.get("frozen_method_sha256")
        != primary.get("frozen_method_sha256")
        or trajectory_score.get("primary_final_plan_sha256") != primary_plan_sha256
        or trajectory_score.get("primary_execution_authority_sha256")
        != primary_authority_sha256
        or trajectory_score.get("primary_count_score_authority_file_sha256")
        != hashlib.sha256(primary_score_raw).hexdigest()
        or trajectory_score.get("execution_claim_sha256") != claim_sha256
        or trajectory_score.get("execution_environment_sha256") != environment_sha256
        or _canonical_bytes(trajectory_score.get("count_model_config"))
        != _canonical_bytes(primary.get("count_model_config"))
        or trajectory_score.get("count_model_config_sha256")
        != primary.get("count_model_config_sha256")
        or trajectory_score.get("retained_calibration_file_sha256")
        != hashlib.sha256(primary_calibration_raw).hexdigest()
        or trajectory_score.get("payload_sha256")
        != canonical_sha256(trajectory_score_body)
        or trajectory_calibration_raw != primary_calibration_raw
        or trajectory_calibration_sha256
        != hashlib.sha256(primary_calibration_raw).hexdigest()
    ):
        raise FinalRunnerContractError(
            "trajectory authority copies differ from the primary authority"
        )
    return MappingProxyType(
        {
            "execution_claim_sha256": claim_sha256,
            "execution_environment_sha256": environment_sha256,
            "primary_execution_authority_sha256": primary_authority_sha256,
            "trajectory_execution_authority_sha256": (trajectory_authority_sha256),
        }
    )


def _rederive_trajectory_evidence_before_receipt(
    repository: Path,
    round_dir: Path,
    evidence: Mapping[str, object],
    result_files: Sequence[Mapping[str, object]],
    *,
    primary_final_plan_sha256: str,
    simulator_assets_root: Path | None = None,
    simulator_r_environment: Path | None = None,
) -> dict[str, object]:
    """Rebuild trajectory authority, plan, records, and inventory from fresh bytes."""

    from .methods import load_method_registry
    from .publication_freeze import validate_frozen_method

    selected_repository, destination = _canonical_round(repository, round_dir)
    primary_plan_sha256 = _sha256(
        primary_final_plan_sha256,
        "primary final plan",
    )
    if not isinstance(evidence, Mapping):
        raise FinalRunnerContractError("trajectory evidence is unavailable")
    evidence_plan = evidence.get("plan")
    if not isinstance(evidence_plan, Mapping):
        raise FinalRunnerContractError("trajectory evidence plan is unavailable")
    evidence_inputs = evidence_plan.get("input_hashes")
    if (
        not isinstance(evidence_inputs, Mapping)
        or evidence_inputs.get("primary_final_plan_sha256") != primary_plan_sha256
    ):
        raise FinalRunnerContractError(
            "trajectory evidence primary-plan binding differs"
        )
    authority_chain = _validate_trajectory_primary_authority_chain(
        selected_repository,
        destination,
        primary_final_plan_sha256=primary_plan_sha256,
        simulator_assets_root=simulator_assets_root,
        simulator_r_environment=simulator_r_environment,
    )
    if (
        evidence_inputs.get("execution_claim_sha256")
        != authority_chain["execution_claim_sha256"]
        or evidence_inputs.get("execution_environment_sha256")
        != authority_chain["execution_environment_sha256"]
        or evidence_inputs.get("execution_authority_sha256")
        != authority_chain["trajectory_execution_authority_sha256"]
    ):
        raise FinalRunnerContractError(
            "trajectory evidence differs from the primary execution authority"
        )

    try:
        frozen_method = validate_frozen_method(selected_repository)
        registry = load_method_registry(selected_repository / "study/methods.json")
        registered = load_prepared_trajectory_dataset(
            selected_repository,
            destination,
        )
    except Exception as error:
        raise FinalRunnerContractError(
            "registered trajectory authority cannot be rederived before receipt"
        ) from error

    authority_path = (
        destination / "results/trajectory/execution_authority/authority.json"
    )
    authority_raw = _read_unique_file(
        authority_path,
        "trajectory execution authority",
        max_bytes=1024 * 1024,
    )
    authority_payload = _strict_json(
        authority_raw,
        "trajectory execution authority",
    )
    authority_body = {
        key: value
        for key, value in authority_payload.items()
        if key != "authority_sha256"
    }
    expected_authority_fields = {
        "schema_version",
        "authority_type",
        "scope",
        "frozen_method_sha256",
        "runtime_lock_sha256",
        "primary_final_plan_sha256",
        "primary_execution_authority_sha256",
        "execution_claim_sha256",
        "execution_environment_sha256",
        "trajectory_authority_sha256",
        "trajectory_binding_sha256",
        "trajectory_dataset_sha256",
        "trajectory_method_input_sha256",
        "base_configuration",
        "base_configuration_sha256",
        "count_model_config",
        "count_model_config_sha256",
        "count_score_authority_path",
        "count_score_authority_sha256",
        "retained_calibration_path",
        "retained_calibration_sha256",
        "calibration_usage",
        "authority_sha256",
    }
    if (
        set(authority_payload) != expected_authority_fields
        or authority_raw != _canonical_bytes(authority_payload) + b"\n"
        or authority_payload.get("schema_version") != 1
        or authority_payload.get("authority_type")
        != "maskimpute_frozen_trajectory_execution"
        or authority_payload.get("scope") != "supplementary_trajectory"
        or authority_payload.get("primary_final_plan_sha256") != primary_plan_sha256
        or authority_payload.get("frozen_method_sha256")
        != frozen_method.get("payload_sha256")
        or authority_payload.get("trajectory_authority_sha256")
        != registered.binding.authority_sha256
        or authority_payload.get("trajectory_binding_sha256")
        != registered.binding.registered_binding_sha256
        or authority_payload.get("trajectory_dataset_sha256")
        != registered.binding.dataset_sha256
        or authority_payload.get("trajectory_method_input_sha256")
        != method_input_sha256(registered.prepared.method_input)
        or authority_payload.get("authority_sha256") != canonical_sha256(authority_body)
    ):
        raise FinalRunnerContractError(
            "trajectory execution authority cannot be rederived before receipt"
        )
    try:
        execution_authority = ExecutionAuthorityContext(
            authority_sha256=str(authority_payload["authority_sha256"]),
            base_configuration_json=_canonical_bytes(
                authority_payload["base_configuration"]
            ).decode(),
            base_configuration_sha256=str(
                authority_payload["base_configuration_sha256"]
            ),
            count_model_config_json=_canonical_bytes(
                authority_payload["count_model_config"]
            ).decode(),
            count_model_config_sha256=str(
                authority_payload["count_model_config_sha256"]
            ),
            count_score_manifest_path=str(
                authority_payload["count_score_authority_path"]
            ),
            count_score_manifest_sha256=str(
                authority_payload["count_score_authority_sha256"]
            ),
            retained_calibration_path=str(
                authority_payload["retained_calibration_path"]
            ),
            retained_calibration_sha256=str(
                authority_payload["retained_calibration_sha256"]
            ),
        )
        plan = build_trajectory_execution_plan(
            frozen_method,
            registry,
            registered,
            execution_claim_sha256=authority_chain["execution_claim_sha256"],
            execution_environment_sha256=(
                authority_chain["execution_environment_sha256"]
            ),
            execution_authority_sha256=execution_authority.authority_sha256,
            primary_final_plan_sha256=primary_plan_sha256,
        )
    except (TypeError, ValueError, RunnerContractError) as error:
        raise FinalRunnerContractError(
            "trajectory execution plan cannot be rederived before receipt"
        ) from error
    if trajectory_execution_plan_payload(plan) != dict(evidence_plan):
        raise FinalRunnerContractError(
            "trajectory execution plan changed before receipt"
        )
    store = FinalResultStore(
        destination / "results/trajectory/execution",
        plan,
        {registered.binding.dataset_id: registered.prepared},
        execution_authority,
        authority_repository=selected_repository,
    )
    validation = validate_trajectory_execution_for_evaluation(
        plan,
        store.load_records(),
    )
    rederived = _trajectory_evaluation_evidence(
        destination,
        plan,
        registered,
        execution_authority,
        store,
        validation,
        result_files,
    )
    if rederived != dict(evidence):
        raise FinalRunnerContractError(
            "trajectory evidence changed before the final evaluation receipt"
        )
    return rederived


def _scaling_evaluation_evidence(
    repository: Path,
    round_dir: Path,
    checkpoint: object,
    result_files: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Bind the validated scaling plan, checkpoint, and exact result bytes."""

    from .scaling import (
        ScalingCheckpoint,
        load_scaling_execution_authority,
        scaling_checkpoint_payload,
        scaling_plan_payload,
    )

    if not isinstance(checkpoint, ScalingCheckpoint):
        raise FinalRunnerContractError(
            "supplementary scaling checkpoint is not canonical"
        )
    authority = load_scaling_execution_authority(repository)
    plan_payload = scaling_plan_payload(authority.plan)
    checkpoint_payload = scaling_checkpoint_payload(checkpoint)
    if (
        checkpoint.plan_sha256 != authority.plan.plan_sha256
        or dict(checkpoint.input_hashes) != dict(authority.plan.input_hashes)
        or checkpoint.status != "completed"
        or len(checkpoint.records) != checkpoint.planned_run_count
    ):
        raise FinalRunnerContractError(
            "supplementary scaling denominator is incomplete or changed"
        )
    checkpoint_relative = (
        "results/scaling/checkpoints/"
        f"{len(checkpoint.datasets) + len(checkpoint.records):08d}.json"
    )
    checkpoint_path = round_dir / checkpoint_relative
    raw = _read_unique_file(
        checkpoint_path,
        "supplementary scaling checkpoint",
        max_bytes=32 * 1024 * 1024,
    )
    if raw != _canonical_bytes(checkpoint_payload) + b"\n":
        raise FinalRunnerContractError(
            "supplementary scaling checkpoint differs from its validated payload"
        )
    declared = _scaling_checkpoint_file_bindings(round_dir)
    observed: dict[str, str] = {}
    for row in result_files:
        if not isinstance(row, Mapping):
            raise FinalRunnerContractError("final result inventory is invalid")
        path = row.get("path")
        digest = row.get("sha256")
        if isinstance(path, str) and path.startswith("results/scaling/"):
            if path in observed:
                raise FinalRunnerContractError(
                    "supplementary scaling result path is duplicated"
                )
            observed[path] = _sha256(digest, f"supplementary scaling result {path}")
    if observed != declared:
        raise FinalRunnerContractError(
            "supplementary scaling result inventory differs from its checkpoint"
        )
    scaling_files = [
        {"path": path, "sha256": observed[path]} for path in sorted(observed)
    ]
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "plan": plan_payload,
        "checkpoint_path": checkpoint_relative,
        "checkpoint_file_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_payload": checkpoint_payload,
        "result_files": scaling_files,
    }
    return {**body, "evidence_sha256": canonical_sha256(body)}


def _run_pre_receipt_supplementary_phases(
    repository: Path,
    round_dir: Path,
) -> Mapping[str, object]:
    """Run frozen supplementary phases at the explicit pre-receipt seam."""

    from .scaling import run_scaling_panel

    # The planned trajectory phase is inserted in this closed seam before the
    # sole receipt without changing the primary final execution denominator.
    return {"scaling": run_scaling_panel(repository, round_dir)}


def _record_final_evaluation_after_scaling(
    repository: Path,
    round_dir: Path,
    evaluation_manifest: Mapping[str, object],
    *,
    simulator_assets_root: Path,
    simulator_r_environment: Path,
) -> dict[str, object]:
    """Require the complete supplementary denominator before issuing the receipt."""

    from .study import record_final_evaluation

    trajectory_evidence = evaluation_manifest.get("trajectory_evidence")
    trajectory_body = (
        {
            key: value
            for key, value in trajectory_evidence.items()
            if key != "evidence_sha256"
        }
        if isinstance(trajectory_evidence, Mapping)
        else {}
    )
    if (
        not isinstance(trajectory_evidence, Mapping)
        or set(trajectory_evidence)
        != {
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
        or trajectory_evidence.get("schema_version") != 1
        or trajectory_evidence.get("status") != "completed"
        or trajectory_evidence.get("scope") != "supplementary_trajectory"
        or trajectory_evidence.get("evidence_sha256")
        != canonical_sha256(trajectory_body)
    ):
        raise FinalRunnerContractError(
            "supplementary trajectory evidence is incomplete"
        )
    supplementary = _run_pre_receipt_supplementary_phases(repository, round_dir)
    checkpoint = supplementary.get("scaling")
    if (
        checkpoint is None
        or getattr(checkpoint, "status", None) != "completed"
        or len(getattr(checkpoint, "records", ()))
        != getattr(checkpoint, "planned_run_count", None)
    ):
        raise FinalRunnerContractError(
            "supplementary scaling denominator is incomplete"
        )
    cumulative = _owned_final_result_file_manifest(round_dir)
    result_files = cumulative["result_files"]
    assert isinstance(result_files, list)
    final_plan_sha256 = _sha256(
        evaluation_manifest.get("final_plan_sha256"),
        "primary final plan",
    )
    rederived_trajectory_evidence = _rederive_trajectory_evidence_before_receipt(
        repository,
        round_dir,
        trajectory_evidence,
        result_files,
        primary_final_plan_sha256=final_plan_sha256,
        simulator_assets_root=simulator_assets_root,
        simulator_r_environment=simulator_r_environment,
    )
    if rederived_trajectory_evidence != dict(trajectory_evidence):
        raise FinalRunnerContractError(
            "supplementary trajectory evidence changed before receipt"
        )
    evidence = _scaling_evaluation_evidence(
        repository,
        round_dir,
        checkpoint,
        result_files,
    )
    sealed_manifest = dict(evaluation_manifest)
    if "scaling_evidence" in sealed_manifest:
        raise FinalRunnerContractError(
            "evaluation manifest already contains supplementary scaling evidence"
        )
    sealed_manifest["scaling_evidence"] = evidence
    sealed_manifest["result_files"] = result_files
    return record_final_evaluation(
        round_dir,
        sealed_manifest,
        repo=repository,
    )


def run_frozen_final_round(
    repository: Path,
    round_dir: Path,
    *,
    simulator_assets_root: Path,
    simulator_r_environment: Path,
) -> dict[str, object]:
    """Claim and execute the frozen final round without scientific overrides."""

    from .datasets import generate_dataset_panel
    from .methods import load_method_registry
    from .publication_freeze import validate_frozen_method
    from .simulators.base import load_final_manifest_claim
    from .simulators.runtime_assets import load_simulator_runtime_assets
    from .study import (
        assert_final_runnable,
        record_incremental_results,
    )

    selected_repository, destination = _canonical_round(repository, round_dir)
    try:
        frozen_method = validate_frozen_method(selected_repository)
    except Exception as error:
        raise FinalRunnerContractError(
            "frozen publication method failed validation"
        ) from error
    try:
        runtime_assets = load_simulator_runtime_assets(
            selected_repository,
            external_root=simulator_assets_root,
            r_environment=simulator_r_environment,
            require_outside_repository=True,
        )
        try:
            preclaim_sha256 = runtime_assets.semantic_sha256
            preclaim_receipt = runtime_assets.semantic_receipt
        finally:
            runtime_assets.close()
    except Exception as error:
        raise FinalRunnerContractError(
            "frozen final simulator runtime preclaim failed"
        ) from error
    claim_path = destination / "execution_claim.json"
    resuming = os.path.lexists(claim_path)
    if resuming:
        _remove_stale_result_temporaries(destination)
        _recover_interrupted_final_transactions(destination)
        _recover_interrupted_trajectory_transactions(destination)
        _recover_scaling_transactions_for_resume(
            selected_repository,
            destination,
        )
        try:
            _reconcile_interrupted_final_publications(
                selected_repository,
                destination,
                frozen_method,
                record_incremental_results,
            )
            load_final_manifest_claim(selected_repository, destination)
        except Exception as error:
            raise FinalRunnerContractError(
                "existing final execution claim is not resumable"
            ) from error
        claim = _strict_json(
            _read_unique_file(claim_path, "final execution claim"),
            "final execution claim",
        )
    else:
        try:
            issued_claim = assert_final_runnable(selected_repository, destination)
        except Exception as error:
            raise FinalRunnerContractError(
                "final round cannot be claimed for execution"
            ) from error
        claim = _strict_json(
            _read_unique_file(claim_path, "final execution claim"),
            "final execution claim",
        )
        if claim != issued_claim:
            raise FinalRunnerContractError(
                "issued final execution claim differs from its record"
            )
    claim_sha256 = canonical_sha256(claim)
    try:
        status = generate_dataset_panel(
            repo=selected_repository,
            namespace="final",
            round_dir=destination,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )
        bindings, prepared = load_prepared_final_panel(
            selected_repository,
            destination,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )
    except Exception as error:
        raise FinalRunnerContractError(
            "frozen final dataset panel is unavailable"
        ) from error
    if (
        status.get("runtime_assets_sha256") != preclaim_sha256
        or status.get("runtime_assets_receipt") != preclaim_receipt
    ):
        raise FinalRunnerContractError(
            "generated final simulator runtime differs from its preclaim"
        )
    if status.get("manifest_sha256") != bindings[0].manifest_sha256:
        raise FinalRunnerContractError(
            "generated final manifest differs from prepared panel"
        )

    registry = load_method_registry(selected_repository / "study/methods.json")
    environments = _load_final_execution_environment_registry(
        selected_repository,
        registry,
    )
    _validate_final_runtime_lock(frozen_method, environments)

    def publish_results() -> object:
        return _record_incremental_results_if_changed(
            selected_repository,
            destination,
            record_incremental_results,
        )

    (
        registered_trajectory,
        authority,
        plan,
        trajectory_authority,
        trajectory_plan,
    ) = _materialize_final_execution_inputs(
        selected_repository,
        destination,
        frozen_method,
        registry,
        bindings,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environments.registry_sha256,
        publish_results=publish_results,
    )
    executor: SpawnedRepositoryExecutor | None = None
    try:
        store = FinalResultStore(
            destination / "results/final/execution",
            plan,
            prepared,
            authority,
            authority_repository=selected_repository,
        )
        trajectory_prepared = {
            registered_trajectory.binding.dataset_id: (registered_trajectory.prepared)
        }
        trajectory_store = FinalResultStore(
            destination / "results/trajectory/execution",
            trajectory_plan,
            trajectory_prepared,
            trajectory_authority,
            authority_repository=selected_repository,
        )
        existing_records = store.load_records()
        store._records_cache = existing_records
        existing_trajectory_records = trajectory_store.load_records()
        trajectory_store._records_cache = existing_trajectory_records
        from .scaling import load_scaling_execution_authority

        scaling_authority = load_scaling_execution_authority(selected_repository)
        storage_preflight = _validate_combined_storage_capacity(
            plan,
            trajectory_plan,
            scaling_authority,
            destination,
            primary_completed_records=len(existing_records),
            trajectory_completed_records=len(existing_trajectory_records),
        )
        dispatcher = RepositoryAdapterDispatcher(selected_repository, environments)

        executor = SpawnedRepositoryExecutor(dispatcher)
        execution_manifest = execute_final_plan(
            plan,
            registry,
            prepared,
            authority,
            executor,
            store,
            on_record_published=publish_results,
        )
        evaluation_validation = validate_final_execution_for_evaluation(
            plan, store._cached_records()
        )
        trajectory_execution_manifest = execute_trajectory_plan(
            trajectory_plan,
            registry,
            trajectory_prepared,
            trajectory_authority,
            executor,
            trajectory_store,
            on_record_published=publish_results,
        )
        trajectory_evaluation_validation = validate_trajectory_execution_for_evaluation(
            trajectory_plan,
            trajectory_store._cached_records(),
        )
        pre_scaling_inventory = _owned_final_result_file_manifest(destination)
        pre_scaling_result_files = pre_scaling_inventory["result_files"]
        assert isinstance(pre_scaling_result_files, list)
        trajectory_evidence = _trajectory_evaluation_evidence(
            destination,
            trajectory_plan,
            registered_trajectory,
            trajectory_authority,
            trajectory_store,
            trajectory_evaluation_validation,
            pre_scaling_result_files,
        )
        evaluation_manifest: dict[str, object] = {
            "schema_version": 1,
            "status": "completed",
            "final_plan_sha256": plan.plan_sha256,
            "final_execution_manifest_path": store.manifest_path.relative_to(
                destination
            ).as_posix(),
            "final_execution_manifest_sha256": _hash_unique_file(
                store.manifest_path, "final execution manifest"
            ),
            "final_execution_payload_sha256": execution_manifest["manifest_sha256"],
            "execution_validation": evaluation_validation,
            "trajectory_evidence": trajectory_evidence,
            "storage_preflight": storage_preflight,
        }
        evaluation_receipt = _record_final_evaluation_after_scaling(
            selected_repository,
            destination,
            evaluation_manifest,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )
    finally:
        if executor is not None:
            executor.close()
    return {
        "execution_manifest": execution_manifest,
        "trajectory_execution_manifest": trajectory_execution_manifest,
        "evaluation_receipt": evaluation_receipt,
    }


__all__ = [
    "FinalExecutionPlan",
    "FinalPlanEntry",
    "FinalRunnerContractError",
    "TrajectoryExecutionPlan",
    "build_final_execution_plan",
    "build_trajectory_execution_plan",
    "execute_final_plan",
    "execute_trajectory_plan",
    "final_result_file_manifest",
    "load_prepared_final_panel",
    "load_prepared_trajectory_dataset",
    "materialize_final_execution_authority",
    "materialize_prepared_trajectory_dataset",
    "materialize_trajectory_execution_authority",
    "run_frozen_final_round",
    "trajectory_execution_plan_payload",
    "validate_final_execution_for_evaluation",
    "validate_final_manifest_payload",
    "validate_trajectory_execution_for_evaluation",
]
