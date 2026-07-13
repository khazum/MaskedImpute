"""No-override execution of the once-only frozen final publication panel."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType
from typing import Literal

from .methods.registry import MethodRegistry
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
_FINAL_DRAWS = tuple(f"draw-{draw:02d}" for draw in range(1, 6))


class FinalRunnerContractError(ValueError):
    """Raised when frozen final execution authority is incomplete or changed."""


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
            raise FinalRunnerContractError(
                "final dataset is not a unique regular file"
            )
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
        if (
            _stable_file_identity(opened_before)
            != _stable_file_identity(opened_after)
            or _stable_file_identity(opened_before)
            != _stable_file_identity(named_after)
        ):
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
        raise FinalRunnerContractError("final repository or round is unavailable") from error
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


def _read_unique_file(path: Path, name: str) -> bytes:
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
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
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
        raise FinalRunnerContractError("final repository or round is unavailable") from error
    if destination != round_dir.absolute() or not destination.is_dir():
        raise FinalRunnerContractError("final round path is not canonical")
    return selected_repository, destination


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
    environment_sha256 = _sha256(
        execution_environment_sha256, "execution environment"
    )
    manifest_sha256 = _sha256(dataset_manifest_sha256, "dataset manifest")

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
    if not isinstance(selected_configuration, Mapping):
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

    results_store = CheckpointStore(destination / "results")
    calibration_relative, calibration_file_sha256 = results_store._publish_immutable(
        "final/execution_authority/retained_calibration.json",
        _canonical_bytes(calibration_payload) + b"\n",
    )
    if calibration_file_sha256 != _sha256(
        calibrator_summary.get("artifact_file_sha256"),
        "frozen calibration artifact file",
    ):
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
        destination / "results" / calibration_relative
    ).relative_to(selected_repository).as_posix()
    score_repo_path = (
        destination / "results" / score_relative
    ).relative_to(selected_repository).as_posix()
    authority_body: dict[str, object] = {
        "schema_version": 1,
        "authority_type": "maskimpute_frozen_final_execution",
        "frozen_method_sha256": frozen_sha256,
        "runtime_lock_sha256": _sha256(
            frozen_method.get("runtime_lock_sha256"), "frozen runtime lock"
        ),
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
        if (
            _stable_file_identity(opened_before)
            != _stable_file_identity(os.fstat(descriptor))
            or _stable_file_identity(opened_before)
            != _stable_file_identity(path.lstat())
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

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "execution_manifest.json"

    def _record_path(self, ordinal: int) -> Path:
        return self.output_dir / "records" / f"{ordinal:08d}.json"

    def _read_record(self, path: Path, plan_entry: FinalPlanEntry) -> dict[str, object]:
        raw = _read_unique_file(path, "final execution record")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise FinalRunnerContractError(
                "final execution record is invalid"
            ) from error
        if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
            raise FinalRunnerContractError("final execution record is not canonical")
        try:
            validated = self._artifacts._validate_stored_record(value, plan_entry.run)
        except (RunnerContractError, OSError, ValueError) as error:
            raise FinalRunnerContractError(
                "final execution record or artifact is invalid"
            ) from error
        return dict(validated)

    def load_records(self) -> tuple[dict[str, object], ...]:
        records_dir = self.output_dir / "records"
        if not records_dir.exists():
            return ()
        try:
            metadata = records_dir.lstat()
            names = sorted(path.name for path in records_dir.iterdir())
        except OSError as error:
            raise FinalRunnerContractError(
                "final record directory is unavailable"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise FinalRunnerContractError("final record directory is invalid")
        expected_names = [f"{ordinal:08d}.json" for ordinal in range(1, len(names) + 1)]
        if names != expected_names or len(names) > len(self.plan.entries):
            raise FinalRunnerContractError("final execution records are not a prefix")
        return tuple(
            self._read_record(records_dir / name, self.plan.entries[index])
            for index, name in enumerate(names)
        )

    def append(
        self, plan_entry: FinalPlanEntry, attempt: EvaluatedAttempt
    ) -> dict[str, object]:
        if not isinstance(plan_entry, FinalPlanEntry):
            raise TypeError("plan_entry must be a FinalPlanEntry")
        if not isinstance(attempt, EvaluatedAttempt):
            raise TypeError("attempt must be an EvaluatedAttempt")
        if self.manifest_path.exists():
            raise FinalRunnerContractError("final execution is already complete")
        records = self.load_records()
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
        try:
            stored = json.loads(
                _canonical_bytes(self._artifacts._stored_attempt(attempt)).decode(
                    "utf-8"
                )
            )
            self._artifacts._validate_stored_record(stored, plan_entry.run)
            self._artifacts._publish_immutable(
                f"records/{plan_entry.run.ordinal:08d}.json",
                _canonical_bytes(stored) + b"\n",
            )
        except (RunnerContractError, OSError, ValueError) as error:
            raise FinalRunnerContractError(
                "cannot publish final execution record"
            ) from error
        observed = self.load_records()
        if len(observed) != next_index + 1:
            raise FinalRunnerContractError("final execution record publication failed")
        return observed[-1]

    def finalize(self) -> dict[str, object]:
        if self.manifest_path.exists():
            raise FinalRunnerContractError("final execution is already complete")
        records = self.load_records()
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
                "manifest_sha256",
            }
        ):
            raise FinalRunnerContractError("final execution manifest is not canonical")
        body = {
            key: nested for key, nested in value.items() if key != "manifest_sha256"
        }
        records = self.load_records()
        references = value.get("records")
        if (
            value.get("schema_version") != 1
            or value.get("status") != "completed"
            or value.get("plan_sha256") != self.plan.plan_sha256
            or value.get("input_hashes") != dict(self.plan.input_hashes)
            or value.get("planned_run_count") != len(self.plan.entries)
            or value.get("recorded_run_count") != len(records)
            or len(records) != len(self.plan.entries)
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
        if (
            not isinstance(payload, Mapping)
            or not isinstance(configuration_id, str)
            or _SAFE_ID.fullmatch(configuration_id) is None
            or canonical_sha256(payload) != digest
        ):
            raise FinalRunnerContractError("frozen candidate configuration is invalid")
        return AuthorizedConfiguration.create(
            method_id=method_id,
            configuration_id=configuration_id,
            kind="candidate_search",
            payload=payload,
            requires_count_score=True,
            requires_calibration=True,
            configuration_sha256=str(digest),
        )
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
            if not isinstance(applicability, Mapping):
                raise FinalRunnerContractError(
                    f"method {spec.id} final applicability is invalid"
                )
            rule = applicability.get("rule")
            integration_status = row.get("integration_status")
            if rule == "all_final_datasets":
                if integration_status != "implemented":
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
    """Execute/resume an exact plan and publish after every immutable attempt."""

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
    if plan.input_hashes.get("execution_authority_sha256") != authority.authority_sha256:
        raise FinalRunnerContractError("final execution authority differs from plan")
    if store.manifest_path.exists():
        return store.load_manifest()
    existing = store.load_records()
    configuration_by_method = {
        value.method_id: value for value in plan.configurations
    }
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
        attempt = evaluate_adapter_outcome(
            plan_entry.run,
            prepared,
            outcome,
        )
        store.append(plan_entry, attempt)
        on_record_published()
    manifest = store.finalize()
    on_record_published()
    return manifest


def run_frozen_final_round(
    repository: Path, round_dir: Path
) -> dict[str, object]:
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
    if os.path.lexists(claim_path):
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
            issued_claim = assert_final_runnable(
                selected_repository, destination
            )
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
        bindings, prepared = load_prepared_final_panel(
            selected_repository, destination
        )
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
    authority = materialize_final_execution_authority(
        selected_repository,
        destination,
        frozen_method,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environments.registry_sha256,
        dataset_manifest_sha256=bindings[0].manifest_sha256,
    )
    try:
        record_incremental_results(
            destination,
            final_result_file_manifest(destination),
            repo=selected_repository,
        )
        plan = build_final_execution_plan(
            frozen_method,
            registry,
            bindings,
            execution_claim_sha256=claim_sha256,
            execution_environment_sha256=environments.registry_sha256,
            execution_authority_sha256=authority.authority_sha256,
        )
        store = FinalResultStore(
            destination / "results/final/execution", plan
        )
        dispatcher = RepositoryAdapterDispatcher(
            selected_repository, environments
        )

        def publish_results() -> object:
            return record_incremental_results(
                destination,
                final_result_file_manifest(destination),
                repo=selected_repository,
            )

        execution_manifest = execute_final_plan(
            plan,
            registry,
            prepared,
            authority,
            SpawnedRepositoryExecutor(dispatcher),
            store,
            on_record_published=publish_results,
        )
        cumulative = final_result_file_manifest(destination)
        record_incremental_results(
            destination, cumulative, repo=selected_repository
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
            "final_execution_payload_sha256": execution_manifest[
                "manifest_sha256"
            ],
            "result_files": cumulative["result_files"],
        }
        evaluation_receipt = record_final_evaluation(
            destination,
            evaluation_manifest,
            repo=selected_repository,
        )
    finally:
        close = getattr(environments, "close", None)
        if callable(close):
            close()
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
    "validate_final_manifest_payload",
]
