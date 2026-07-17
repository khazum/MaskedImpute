"""No-override execution of the once-only frozen final publication panel."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
from types import MappingProxyType
from typing import Literal
import zlib

import numpy as np

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
    PreparedDataset,
    RepositoryAdapterDispatcher,
    RunnerContractError,
    RunPlanEntry,
    SpawnedRepositoryExecutor,
    enforce_calibration_fold_receipt,
    evaluate_adapter_outcome,
    prepare_dataset_pair_for_execution,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_STAGING_FILE = re.compile(r"\..+\.[A-Za-z0-9_-]{6,}\.tmp\Z")
_TRANSACTION_FILE = re.compile(r"[0-9]{8}\.json\Z")
_FINAL_RUN_ID = re.compile(r"final-[a-z0-9-]+\Z")
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


def _read_bound_h5ad(path: Path, binding: DatasetBinding):
    """Open one exact H5AD inode and recheck its byte and semantic bindings."""

    import anndata as ad

    from .schema import benchmark_dataset_sha256

    if not isinstance(path, Path) or not isinstance(binding, DatasetBinding):
        raise TypeError("path and binding must be canonical values")
    descriptor = -1
    try:
        if path.resolve(strict=True) != path.absolute():
            raise FinalRunnerContractError("final dataset path contains a symlink")
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
            raise FinalRunnerContractError("final dataset is not a unique regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != binding.output_file_sha256:
            raise FinalRunnerContractError("final dataset file checksum differs")
        dataset = ad.read_h5ad(Path(f"/proc/self/fd/{descriptor}"))
        opened_after = os.fstat(descriptor)
        named_after = path.lstat()
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


def load_prepared_final_panel(
    repository: Path, round_dir: Path
) -> tuple[tuple[DatasetBinding, ...], Mapping[str, PreparedDataset]]:
    """Byte-revalidate and pair-union-QC the exact unseen final panel."""

    from .datasets import validate_dataset_status

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
    status_path = destination / "results/dataset_status.json"
    try:
        status = validate_dataset_status(
            status_path, repo=selected_repository, round_dir=destination
        )
        bindings = validate_final_manifest_payload(status)
    except Exception as error:
        raise FinalRunnerContractError(
            "final dataset status failed byte-level revalidation"
        ) from error
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
    try:
        status_after = validate_dataset_status(
            status_path, repo=selected_repository, round_dir=destination
        )
    except Exception as error:
        raise FinalRunnerContractError(
            "final dataset status changed during panel preparation"
        ) from error
    if status_after != status:
        raise FinalRunnerContractError(
            "final dataset status changed during panel preparation"
        )
    return bindings, MappingProxyType(prepared)


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
class FinalExecutionPlan:
    """Hash-bound frozen method x final dataset x nested seed denominator."""

    schema_version: int
    input_hashes: Mapping[str, str]
    entries: tuple[FinalPlanEntry, ...]
    configurations: tuple[AuthorizedConfiguration, ...]
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

    results_store = CheckpointStore(destination / "results")
    calibration_relative, calibration_file_sha256 = results_store._publish_immutable(
        "final/execution_authority/retained_calibration.json",
        calibration_raw,
    )
    if calibration_file_sha256 != expected_calibration_file_sha256:
        raise FinalRunnerContractError(
            "materialized calibration bytes differ from frozen evidence"
        )

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
    score_relative, score_file_sha256 = results_store._publish_immutable(
        "final/execution_authority/count_score_authority.json",
        _canonical_bytes(score_payload) + b"\n",
    )
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
    results_store._publish_immutable(
        "final/execution_authority/authority.json",
        _canonical_bytes(authority_payload) + b"\n",
    )
    return ExecutionAuthorityContext(
        authority_sha256=authority_sha256,
        base_configuration_json=_canonical_bytes(dict(base_configuration)).decode(),
        base_configuration_sha256=base_configuration_sha256,
        count_model_config_json=_canonical_bytes(dict(count_config)).decode(),
        count_model_config_sha256=str(count_config_sha256),
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
        if _STAGING_FILE.fullmatch(path.name) is None:
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
        }:
            continue
        try:
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink not in {1, 2}
                or path.parent.resolve(strict=True) != path.parent.absolute()
            ):
                raise FinalRunnerContractError(
                    "stale final result temporary is not an owned regular file"
                )
            relative = path.relative_to(destination).as_posix()
            path.unlink()
            directory = os.open(
                path.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if os.path.lexists(path):
                raise FinalRunnerContractError(
                    "stale final result temporary survived removal"
                )
            removed.append(relative)
        except FinalRunnerContractError:
            raise
        except OSError as error:
            raise FinalRunnerContractError(
                "stale final result temporary could not be removed"
            ) from error
    return tuple(removed)


def _recover_interrupted_final_transactions(round_dir: Path) -> tuple[int, ...]:
    """Roll back artifacts lacking a committed record; retain committed attempts."""

    destination = round_dir.resolve(strict=True)
    execution = destination / "results/final/execution"
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
            or _FINAL_RUN_ID.fullmatch(run_id) is None
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


class FinalResultStore:
    """Append-only per-attempt artifacts plus one immutable completion manifest."""

    def __init__(self, output_dir: Path, plan: FinalExecutionPlan) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        if not isinstance(plan, FinalExecutionPlan):
            raise TypeError("plan must be a FinalExecutionPlan")
        self.output_dir = output_dir.absolute()
        self.plan = plan
        self._artifacts = CheckpointStore(self.output_dir)
        self._records_cache: tuple[dict[str, object], ...] | None = None

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

    def _stored_final_attempt(self, attempt: EvaluatedAttempt) -> dict[str, object]:
        """Store one compressed evaluator matrix and omit its redundant native form."""

        without_dense_outputs = replace(
            attempt,
            native_output=None,
            evaluator_output=None,
        )
        stored = self._artifacts._stored_attempt(without_dense_outputs)
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
        relative, digest = self._artifacts._publish_immutable(
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
        self, run: Mapping[str, object]
    ) -> dict[str, object]:
        """Validate bounded decompression and return a raw-store validation view."""

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
            path = self._artifacts._safe_artifact_path(
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
        if value is None:
            if plan_entry.action == "not_applicable":
                return None
            if run.get("status") == "completed" and plan_entry.run.requires_calibration:
                raise FinalRunnerContractError(
                    "completed final calibration lacks its execution request receipt"
                )
            return None
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
            receipt = self._validate_execution_request_receipt(
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
                final_calibration_artifact_sha256=(
                    None
                    if receipt is None
                    else receipt.get("retained_calibration_sha256")
                ),
            )
        except (RunnerContractError, OSError, ValueError) as error:
            raise FinalRunnerContractError(
                "final execution record or artifact is invalid"
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
        execution_request: ExecutionRequest | None = None,
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
        intent_path = self._publish_transaction_intent(plan_entry, attempt)
        request_receipt: dict[str, object] | None = None
        if execution_request is not None:
            if not isinstance(execution_request, ExecutionRequest):
                raise TypeError("execution_request must be an ExecutionRequest")
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
                or execution_request.configuration_id != plan_entry.run.configuration_id
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
                final_calibration_artifact_sha256=(
                    None
                    if request_receipt is None
                    else request_receipt["retained_calibration_sha256"]
                ),
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
        if (
            not isinstance(value, dict)
            or raw != _canonical_bytes(value) + b"\n"
            or set(value)
            != {
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
    method_id: str,
    dataset_id: str,
    model_seed: int | None,
    configuration_sha256: str,
) -> str:
    seed = "deterministic" if model_seed is None else f"seed-{model_seed}"
    return (
        f"final-{method_id}-{dataset_id.removeprefix('dataset-')}-{seed}-"
        f"{configuration_sha256[:12]}"
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
    return AuthorizedConfiguration.registry_default(spec)


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
    claim_sha256 = _sha256(execution_claim_sha256, "execution claim")
    environment_sha256 = _sha256(execution_environment_sha256, "execution environment")
    authority_sha256 = _sha256(execution_authority_sha256, "execution authority")
    configurations = tuple(
        _configuration_for_method(spec.id, spec, frozen_method)
        for spec in registry.methods
    )
    configuration_by_id = {
        configuration.method_id: configuration for configuration in configurations
    }
    entries: list[FinalPlanEntry] = []
    ordinal = 0
    for binding in dataset_values:
        for spec in registry.methods:
            row = frozen_by_id[spec.id]
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
                action: Literal["execute", "not_applicable"] = "execute"
                reason: str | None = None
                seeds: tuple[int | None, ...] = (
                    DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
                )
            elif rule in {"never", "matched_bulk_reference_present"}:
                action = "not_applicable"
                raw_reason = applicability.get("non_run_reason")
                if not isinstance(raw_reason, str) or not raw_reason:
                    raise FinalRunnerContractError(
                        f"method {spec.id} lacks a final non-run reason"
                    )
                reason = raw_reason
                if rule == "never":
                    if applicability.get("required_reference") is not None:
                        raise FinalRunnerContractError(
                            f"method {spec.id} non-run reference binding is invalid"
                        )
                    if reason == "historical_method_not_rerun":
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
                        or reason != "matched_bulk_reference_absent"
                    ):
                        raise FinalRunnerContractError(
                            f"method {spec.id} matched-bulk disposition is invalid"
                        )
                seeds = (None,)
            else:
                raise FinalRunnerContractError(
                    f"method {spec.id} has an unknown final applicability rule"
                )
            configuration = configuration_by_id[spec.id]
            for seed in seeds:
                ordinal += 1
                run = RunPlanEntry(
                    ordinal=ordinal,
                    run_id=_final_run_id(
                        spec.id,
                        binding.dataset_id,
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
                    preflight_status="planned",
                    preflight_reason=None,
                    configuration_kind=configuration.kind,
                    requires_count_score=configuration.requires_count_score,
                    requires_calibration=configuration.requires_calibration,
                )
                entries.append(FinalPlanEntry(run=run, action=action, reason=reason))
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


def execute_final_plan(
    plan: FinalExecutionPlan,
    registry: MethodRegistry,
    prepared_datasets: Mapping[str, PreparedDataset],
    authority: ExecutionAuthorityContext,
    executor: Callable[[ExecutionRequest], AdapterOutcome],
    store: FinalResultStore,
    *,
    on_record_published: Callable[[], object],
) -> dict[str, object]:
    """Execute/resume an exact plan and journal its complete immutable manifest."""

    if not isinstance(plan, FinalExecutionPlan):
        raise TypeError("plan must be a FinalExecutionPlan")
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
            request = ExecutionRequest.create(
                spec,
                prepared.method_input,
                model_seed=plan_entry.run.model_seed,
                configuration=configuration,
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


def validate_final_execution_for_evaluation(
    plan: FinalExecutionPlan,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Require a complete terminal denominator without success-conditioned exclusion."""

    from .runner import _RECONSTRUCTION_METRIC_NAMES

    if not isinstance(plan, FinalExecutionPlan):
        raise TypeError("plan must be a FinalExecutionPlan")
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
        "final_plan_sha256": plan.plan_sha256,
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
    return {**body, "validation_sha256": canonical_sha256(body)}


def run_frozen_final_round(repository: Path, round_dir: Path) -> dict[str, object]:
    """Claim and execute the frozen final round without scientific overrides."""

    from .datasets import generate_dataset_panel
    from .methods import load_method_registry
    from .publication_freeze import validate_frozen_method
    from .simulators.base import load_final_manifest_claim
    from .study import (
        assert_final_runnable,
        record_final_evaluation,
        record_incremental_results,
    )

    selected_repository, destination = _canonical_round(repository, round_dir)
    try:
        frozen_method = validate_frozen_method(selected_repository)
    except Exception as error:
        raise FinalRunnerContractError(
            "frozen publication method failed validation"
        ) from error
    claim_path = destination / "execution_claim.json"
    resuming = os.path.lexists(claim_path)
    if resuming:
        _remove_stale_result_temporaries(destination)
        _recover_interrupted_final_transactions(destination)
        results = destination / "results/final"
        if os.path.lexists(results / "execution_authority") or os.path.lexists(
            results / "execution"
        ):
            _record_incremental_results_if_changed(
                selected_repository,
                destination,
                record_incremental_results,
            )
        try:
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
        )
        bindings, prepared = load_prepared_final_panel(selected_repository, destination)
    except Exception as error:
        raise FinalRunnerContractError(
            "frozen final dataset panel is unavailable"
        ) from error
    if status.get("manifest_sha256") != bindings[0].manifest_sha256:
        raise FinalRunnerContractError(
            "generated final manifest differs from prepared panel"
        )

    registry = load_method_registry(selected_repository / "study/methods.json")
    environments = ExecutionEnvironmentRegistry.fixed(
        selected_repository,
        runtime_lock_path=(
            selected_repository / "environments/development-runtime.lock.json"
        ),
        benchmark_python=Path(sys.executable),
        r_library_paths={
            "saver": (selected_repository / "artifacts/envs/saver-r/library",)
        },
    )
    _validate_final_runtime_lock(frozen_method, environments)
    provisional_plan = build_final_execution_plan(
        frozen_method,
        registry,
        bindings,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environments.registry_sha256,
        execution_authority_sha256="0" * 64,
    )
    storage_preflight: dict[str, int | str] | None = None
    if not resuming:
        storage_preflight = _validate_final_storage_capacity(
            provisional_plan, destination
        )
    authority = materialize_final_execution_authority(
        selected_repository,
        destination,
        frozen_method,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environments.registry_sha256,
        dataset_manifest_sha256=bindings[0].manifest_sha256,
    )
    executor: SpawnedRepositoryExecutor | None = None
    try:
        _record_incremental_results_if_changed(
            selected_repository,
            destination,
            record_incremental_results,
        )
        plan = build_final_execution_plan(
            frozen_method,
            registry,
            bindings,
            execution_claim_sha256=claim_sha256,
            execution_environment_sha256=environments.registry_sha256,
            execution_authority_sha256=authority.authority_sha256,
        )
        if (
            plan.entries != provisional_plan.entries
            or plan.configurations != provisional_plan.configurations
        ):
            raise FinalRunnerContractError(
                "final plan changed while materializing execution authority"
            )
        store = FinalResultStore(destination / "results/final/execution", plan)
        if resuming:
            existing_records = store.load_records()
            store._records_cache = existing_records
            storage_preflight = _validate_final_storage_capacity(
                plan,
                destination,
                completed_records=len(existing_records),
            )
        assert storage_preflight is not None
        dispatcher = RepositoryAdapterDispatcher(selected_repository, environments)

        def publish_results() -> object:
            return _record_incremental_results_if_changed(
                selected_repository,
                destination,
                record_incremental_results,
            )

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
        cumulative = _owned_final_result_file_manifest(destination)
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
            "storage_preflight": storage_preflight,
            "result_files": cumulative["result_files"],
        }
        evaluation_receipt = record_final_evaluation(
            destination,
            evaluation_manifest,
            repo=selected_repository,
        )
    finally:
        if executor is not None:
            executor.close()
    return {
        "execution_manifest": execution_manifest,
        "evaluation_receipt": evaluation_receipt,
    }


__all__ = [
    "FinalExecutionPlan",
    "FinalPlanEntry",
    "FinalRunnerContractError",
    "build_final_execution_plan",
    "execute_final_plan",
    "final_result_file_manifest",
    "load_prepared_final_panel",
    "materialize_final_execution_authority",
    "run_frozen_final_round",
    "validate_final_execution_for_evaluation",
    "validate_final_manifest_payload",
]
