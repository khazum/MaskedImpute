"""Produce and revalidate the fixed Tung matched-bulk development evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from types import MappingProxyType
from typing import Any

import numpy as np

from .development_evaluation import (
    EndpointUnit,
    PreparedRealOrthogonalPanel,
    TungSource,
    prepare_real_orthogonal_panel,
    tung_concordance_units,
)
from .methods import (
    AdapterExecution,
    AdapterUnavailableError,
    MethodInput,
    MethodSpec,
    SourceReceipt,
    count_equivalent_to_log2_cp10k,
    d3impute_to_evaluator_counts,
    finalize_d3impute_output,
    finalize_sctsi_output,
    prepare_matched_bulk_reference,
    prepare_sctsi_matched_bulk_reference,
    run_d3impute,
    run_sctsi,
    sctsi_to_evaluator_counts,
    validate_matched_bulk_reference,
    validate_sctsi_matched_bulk_reference,
    verify_pinned_source,
)
from .methods.d3impute import (
    D3ImputeConfig,
    MatchedBulkReference,
    _D3IMPUTE_DRIVER,
)
from .methods.registry import MethodRegistry, parse_method_registry
from .methods.sctsi import SCTSIConfig, SCTSIMatchedBulkReference, _SCTSI_DRIVER
from .protocol import canonical_sha256
from .runtime_environments import (
    RuntimeEnvironmentError,
    RuntimeEnvironmentLock,
    load_runtime_environment_lock,
    validate_runtime_environment_entry,
)


OUTPUT_RELATIVE_PATH = Path(
    "artifacts/study/development/competition-external-reference"
)
CHECKPOINT_RELATIVE_PATH = OUTPUT_RELATIVE_PATH / "checkpoint.json"
RUNTIME_LOCK_RELATIVE_PATH = Path("environments/development-runtime.lock.json")
METHOD_REGISTRY_RELATIVE_PATH = Path("study/methods.json")

_PRODUCER = "maskimpute-external-reference-development-v1"
_DATASET_ID = "tung-ipsc-ercc-bulk-replicates"
_REFERENCE_ID = "tung-ipsc-matched-bulk"
_METHOD_IDS = ("d3impute", "sctsi")
_ENDPOINT_IDS = (
    "bulk_pseudobulk_concordance",
    "ercc_recovery",
    "technical_replicate_concordance",
)
_ENDPOINT_REFERENCE_OVERLAP = MappingProxyType(
    {
        "bulk_pseudobulk_concordance": "adapter_input_matched_bulk",
        "ercc_recovery": "adapter_input_matched_bulk",
        "technical_replicate_concordance": "same_experiment_technical_lane",
    }
)
_COMPATIBILITY_CODES = MappingProxyType(
    {
        "d3impute": (
            "external_reference_binding",
            "source_archive_execution",
            "upstream_parameters",
            "fixed_rng_compatibility",
            "evaluation_label_exclusion",
            "evaluator_scale_conversion",
        ),
        "sctsi": (
            "input_scale_conversion",
            "input_orientation",
            "bulk_average_contract",
            "published_demo_truth_exclusion",
            "upstream_defaults",
            "deterministic_execution",
            "upstream_selective_policy",
            "evaluator_scale_conversion",
            "source_policy",
        ),
    }
)
_TOP_ARTIFACT_PATHS = {
    "input_metadata": "inputs/tung-method-input.json",
    "input_counts": "inputs/tung-counts.f64",
    "bulk_metadata": "references/tung-bulk-reference.json",
    "bulk_reference": "references/tung-bulk-counts.f64",
    "plan": "plan.json",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REASON = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*\Z")
_NONFINAL_MARKERS = ("pending", "not_yet", "unverified")


class ExternalReferenceDevelopmentError(RuntimeError):
    """Raised when fixed external-reference evidence is incomplete or altered."""


@dataclass(frozen=True, slots=True)
class ValidatedExternalReferenceEvidence:
    """Freshly reopened production evidence ready for publication freeze."""

    output_directory: Path
    checkpoint_path: Path
    checkpoint_file_sha256: str
    checkpoint: Mapping[str, object]
    dataset_id: str
    method_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Authority:
    repository: Path
    registry: MethodRegistry
    registry_file_sha256: str
    specs: Mapping[str, MethodSpec]
    runtime_lock: RuntimeEnvironmentLock
    runtime_lock_file_sha256: str
    panel: PreparedRealOrthogonalPanel
    method_input: MethodInput
    tung: TungSource
    bulk_matrix: np.ndarray
    sample_ids: tuple[str, ...]
    d3_reference: MatchedBulkReference
    sctsi_reference: SCTSIMatchedBulkReference
    source_evidence: Mapping[str, object]
    source_receipts: Mapping[str, SourceReceipt]
    locators: Mapping[str, Path]
    sctsi_library: Path
    locator_identities: Mapping[str, object]
    runtime_inventory_sha256s: Mapping[str, str]


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
        raise ExternalReferenceDevelopmentError(
            "external-reference payload is not canonical JSON"
        ) from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalReferenceDevelopmentError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ExternalReferenceDevelopmentError(f"non-finite JSON constant {value}")


def _json_value(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except ExternalReferenceDevelopmentError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ExternalReferenceDevelopmentError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
        raise ExternalReferenceDevelopmentError(f"{label} is not canonical JSON")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExternalReferenceDevelopmentError(f"{label} is not a SHA-256")
    return value


def _stable_read(path: Path, label: str) -> bytes:
    descriptor: int | None = None
    try:
        if not path.is_absolute():
            raise ExternalReferenceDevelopmentError(f"{label} path is not absolute")
        parent = path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise ExternalReferenceDevelopmentError(f"{label} parent is unsafe")
        named_before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or stat.S_ISLNK(named_before.st_mode)
            or (
                named_before.st_dev,
                named_before.st_ino,
                named_before.st_mode,
                named_before.st_size,
                named_before.st_mtime_ns,
                named_before.st_ctime_ns,
            )
            != (
                opened_before.st_dev,
                opened_before.st_ino,
                opened_before.st_mode,
                opened_before.st_size,
                opened_before.st_mtime_ns,
                opened_before.st_ctime_ns,
            )
        ):
            raise ExternalReferenceDevelopmentError(
                f"{label} is not a unique regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        named_after = path.lstat()
        identity_after = (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_mode,
            opened_after.st_size,
            opened_after.st_mtime_ns,
            opened_after.st_ctime_ns,
        )
        named_identity_after = (
            named_after.st_dev,
            named_after.st_ino,
            named_after.st_mode,
            named_after.st_size,
            named_after.st_mtime_ns,
            named_after.st_ctime_ns,
        )
        identity_before = (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_mode,
            opened_before.st_size,
            opened_before.st_mtime_ns,
            opened_before.st_ctime_ns,
        )
        if identity_after != identity_before or named_identity_after != identity_before:
            raise ExternalReferenceDevelopmentError(f"{label} changed during access")
        return b"".join(chunks)
    except ExternalReferenceDevelopmentError:
        raise
    except OSError as error:
        raise ExternalReferenceDevelopmentError(
            f"cannot read {label}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _repository_root(repository: Path) -> Path:
    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if not repository.is_absolute():
        repository = repository.absolute()
    try:
        metadata = repository.lstat()
        resolved = repository.resolve(strict=True)
    except OSError as error:
        raise ExternalReferenceDevelopmentError("repository is unavailable") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or resolved != repository
    ):
        raise ExternalReferenceDevelopmentError(
            "repository must be an exact non-symlink directory path"
        )
    return repository


def _safe_directory_chain(
    root: Path,
    relative: Path,
    *,
    create: bool,
) -> Path:
    """Reach one repository-relative directory without following symlinks."""

    if relative.is_absolute() or ".." in relative.parts:
        raise ExternalReferenceDevelopmentError(
            "external-reference output directory is unsafe"
        )
    current = root
    for component in relative.parts:
        candidate = current / component
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            if not create:
                raise ExternalReferenceDevelopmentError(
                    "external-reference output directory is unavailable"
                ) from None
            try:
                os.mkdir(candidate, mode=0o700)
            except FileExistsError:
                pass
            except OSError as error:
                raise ExternalReferenceDevelopmentError(
                    "cannot create external-reference output directory"
                ) from error
            try:
                metadata = candidate.lstat()
            except OSError as error:
                raise ExternalReferenceDevelopmentError(
                    "external-reference output directory changed during creation"
                ) from error
        except OSError as error:
            raise ExternalReferenceDevelopmentError(
                "external-reference output directory is unavailable"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ExternalReferenceDevelopmentError(
                "external-reference output path contains an unsafe directory"
            )
        try:
            if candidate.resolve(strict=True) != candidate:
                raise ExternalReferenceDevelopmentError(
                    "external-reference output path contains a symlink"
                )
        except OSError as error:
            raise ExternalReferenceDevelopmentError(
                "external-reference output directory changed during validation"
            ) from error
        current = candidate
    return current


def _relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ExternalReferenceDevelopmentError(f"{label} path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ExternalReferenceDevelopmentError(f"{label} path is unsafe")
    return relative


def _artifact_path(output: Path, value: object, label: str) -> Path:
    relative = _relative_path(value, label)
    path = output.joinpath(*relative.parts)
    current = output
    try:
        if output.is_symlink() or not output.is_dir():
            raise ExternalReferenceDevelopmentError(
                "external-reference output is unsafe"
            )
        for component in relative.parts[:-1]:
            current = current / component
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ExternalReferenceDevelopmentError(
                    f"{label} path contains an unsafe directory"
                )
    except ExternalReferenceDevelopmentError:
        raise
    except OSError as error:
        raise ExternalReferenceDevelopmentError(
            f"{label} path is unavailable"
        ) from error
    return path


def _binding(relative: str, raw: bytes) -> dict[str, object]:
    _relative_path(relative, "artifact")
    return {
        "path": relative,
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
    }


def _read_binding(
    output: Path,
    value: object,
    label: str,
) -> tuple[bytes, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ExternalReferenceDevelopmentError(f"{label} binding is invalid")
    expected_sha = _sha256(value.get("sha256"), f"{label} checksum")
    expected_size = value.get("size_bytes")
    if type(expected_size) is not int or expected_size < 0:
        raise ExternalReferenceDevelopmentError(f"{label} byte size is invalid")
    path = _artifact_path(output, value.get("path"), label)
    raw = _stable_read(path, label)
    if len(raw) != expected_size or _sha256_bytes(raw) != expected_sha:
        raise ExternalReferenceDevelopmentError(f"{label} checksum or bytes differ")
    return raw, str(value["path"])


def _write_new(output: Path, relative: str, raw: bytes) -> dict[str, object]:
    path_value = _relative_path(relative, "new artifact")
    path = output.joinpath(*path_value.parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    except FileExistsError as error:
        raise ExternalReferenceDevelopmentError(
            f"refusing to overwrite external-reference artifact {relative}"
        ) from error
    except OSError as error:
        raise ExternalReferenceDevelopmentError(
            f"cannot publish external-reference artifact {relative}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return _binding(relative, raw)


def _write_json(output: Path, relative: str, value: object) -> dict[str, object]:
    return _write_new(output, relative, _canonical_bytes(value) + b"\n")


def _locator_identity(path: Path, label: str, *, directory: bool) -> dict[str, object]:
    if not isinstance(path, Path):
        raise ExternalReferenceDevelopmentError(
            f"{label} locator must be a pathlib.Path"
        )
    if not path.is_absolute() or ".." in path.parts:
        raise ExternalReferenceDevelopmentError(f"{label} locator must be absolute")
    current = Path(path.anchor)
    try:
        for part in path.relative_to(path.anchor).parts:
            current = current / part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ExternalReferenceDevelopmentError(
                    f"{label} locator must not contain a symlink"
                )
        metadata = path.lstat()
        if directory:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ExternalReferenceDevelopmentError(
                    f"{label} locator must be a directory"
                )
        elif not stat.S_ISREG(metadata.st_mode) or not os.access(path, os.X_OK):
            raise ExternalReferenceDevelopmentError(
                f"{label} locator must be an executable regular file"
            )
        resolved = path.resolve(strict=True)
        if resolved != path:
            raise ExternalReferenceDevelopmentError(
                f"{label} locator must use its exact non-symlink path"
            )
        result: dict[str, object] = {
            "path": str(path),
            "kind": "directory" if directory else "executable",
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": stat.S_IMODE(metadata.st_mode),
        }
        if not directory:
            raw = _stable_read(path, f"{label} executable")
            result.update({"size_bytes": len(raw), "sha256": _sha256_bytes(raw)})
        return result
    except ExternalReferenceDevelopmentError:
        raise
    except OSError as error:
        raise ExternalReferenceDevelopmentError(
            f"{label} locator is unavailable"
        ) from error


def _source_evidence(panel: PreparedRealOrthogonalPanel) -> dict[str, object]:
    evidence = panel.source_evidence
    return {
        "ledger_path": evidence.ledger_path,
        "ledger_file_sha256": evidence.ledger_file_sha256,
        "ledger_sha256": evidence.ledger_sha256,
        "receipts": [asdict(value) for value in evidence.receipts],
        "artifacts": [asdict(value) for value in evidence.artifacts],
    }


def _load_registry(repository: Path) -> tuple[MethodRegistry, str]:
    raw = _stable_read(repository / METHOD_REGISTRY_RELATIVE_PATH, "method registry")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        registry = parse_method_registry(value)
    except ExternalReferenceDevelopmentError:
        raise
    except (UnicodeError, ValueError, TypeError) as error:
        raise ExternalReferenceDevelopmentError("method registry is invalid") from error
    external_ids = tuple(
        sorted(
            spec.id
            for spec in registry.methods
            if spec.execution_scope == "external_reference_only"
        )
    )
    if external_ids != _METHOD_IDS:
        raise ExternalReferenceDevelopmentError(
            "external-reference registry denominator is not fixed to D3Impute and scTsI"
        )
    for method_id in _METHOD_IDS:
        spec = registry.by_id(method_id)
        if spec.track != "external_reference" or spec.source.kind != "git":
            raise ExternalReferenceDevelopmentError(
                f"method {method_id} is not a pinned external-reference adapter"
            )
    return registry, _sha256_bytes(raw)


def _load_runtime_lock(repository: Path) -> tuple[RuntimeEnvironmentLock, str]:
    path = repository / RUNTIME_LOCK_RELATIVE_PATH
    raw = _stable_read(path, "development runtime lock")
    expected = _sha256_bytes(raw)
    try:
        lock = load_runtime_environment_lock(path)
    except (RuntimeEnvironmentError, OSError, ValueError) as error:
        raise ExternalReferenceDevelopmentError(
            f"development runtime lock is invalid: {error}"
        ) from error
    if lock.file_sha256 != expected:
        raise ExternalReferenceDevelopmentError(
            "development runtime lock checksum changed during access"
        )
    for method_id in _METHOD_IDS:
        try:
            lock.by_id(method_id)
        except RuntimeEnvironmentError as error:
            raise ExternalReferenceDevelopmentError(
                f"runtime lock lacks fixed external method {method_id}"
            ) from error
    return lock, expected


def _prepared_tung(
    repository: Path,
) -> tuple[
    PreparedRealOrthogonalPanel, MethodInput, TungSource, np.ndarray, tuple[str, ...]
]:
    try:
        panel = prepare_real_orthogonal_panel(repository)
    except Exception as error:
        raise ExternalReferenceDevelopmentError(
            f"canonical Tung source preparation failed: {error}"
        ) from error
    if not isinstance(panel, PreparedRealOrthogonalPanel):
        raise ExternalReferenceDevelopmentError(
            "real orthogonal preparation did not return its fixed panel"
        )
    matches = [value for value in panel.method_inputs if value.source_id == _DATASET_ID]
    if len(matches) != 1 or not isinstance(matches[0].method_input, MethodInput):
        raise ExternalReferenceDevelopmentError(
            "canonical Tung method input is not uniquely prepared"
        )
    method_input = matches[0].method_input
    tung = panel.tung
    if (
        not isinstance(tung, TungSource)
        or method_input.obs_ids != tung.cell_ids
        or method_input.var_ids != tung.gene_ids
        or method_input.shape != tung.counts.shape
        or not np.array_equal(method_input.counts, tung.counts)
    ):
        raise ExternalReferenceDevelopmentError(
            "canonical Tung method input differs from freshly prepared source bytes"
        )
    sample_ids = tuple(sorted(tung.bulk_profiles))
    if len(sample_ids) < 2 or set(sample_ids) != set(tung.sample_ids):
        raise ExternalReferenceDevelopmentError(
            "canonical Tung measured bulk samples do not match single-cell samples"
        )
    try:
        bulk = np.column_stack(
            [np.asarray(tung.bulk_profiles[value]) for value in sample_ids]
        )
    except (TypeError, ValueError) as error:
        raise ExternalReferenceDevelopmentError(
            "canonical Tung bulk profiles cannot be aligned"
        ) from error
    return panel, method_input, tung, bulk, sample_ids


def _runtime_inventory(
    lock: RuntimeEnvironmentLock,
    method_id: str,
    executable: Path,
    r_library: Path,
) -> str:
    try:
        return validate_runtime_environment_entry(
            lock,
            method_id,
            "python" if method_id == "d3impute" else "r",
            executable,
            r_library_paths=() if method_id == "d3impute" else (r_library,),
        )
    except (RuntimeEnvironmentError, OSError, ValueError) as error:
        raise ExternalReferenceDevelopmentError(
            f"runtime validation failed for {method_id}: {error}"
        ) from error


def _derive_authority(
    repository: Path,
    *,
    environments: Mapping[str, Path],
    sctsi_library: Path,
) -> _Authority:
    if not isinstance(environments, Mapping) or set(environments) != set(_METHOD_IDS):
        raise ExternalReferenceDevelopmentError(
            "environment locators must name exactly d3impute and sctsi"
        )
    locators: dict[str, Path] = {}
    locator_identities: dict[str, object] = {}
    for method_id in _METHOD_IDS:
        locator = environments.get(method_id)
        if not isinstance(locator, Path):
            raise ExternalReferenceDevelopmentError(
                f"environment locator for {method_id} must be a pathlib.Path"
            )
        locators[method_id] = locator
        locator_identities[method_id] = _locator_identity(
            locator, f"{method_id} environment", directory=False
        )
    library_identity = _locator_identity(sctsi_library, "scTsI library", directory=True)
    locator_identities["sctsi_library"] = library_identity

    registry, registry_sha = _load_registry(repository)
    specs = {method_id: registry.by_id(method_id) for method_id in _METHOD_IDS}
    lock, lock_sha = _load_runtime_lock(repository)
    for method_id, spec in specs.items():
        if (
            spec.environment.status != "ready"
            or spec.environment.lock_sha256 != lock_sha
        ):
            raise ExternalReferenceDevelopmentError(
                f"registry environment for {method_id} is not ready and bound to the runtime lock"
            )
    panel, method_input, tung, bulk, sample_ids = _prepared_tung(repository)
    try:
        d3_reference = prepare_matched_bulk_reference(
            reference_id=_REFERENCE_ID,
            source_sha256=tung.bulk_sample_file_sha256,
            matrix=bulk,
            var_ids=method_input.var_ids,
            sample_ids=sample_ids,
        )
        sctsi_reference = prepare_sctsi_matched_bulk_reference(
            reference_id=_REFERENCE_ID,
            source_sha256=tung.bulk_sample_file_sha256,
            matrix=bulk,
            var_ids=method_input.var_ids,
            sample_ids=sample_ids,
            expression_scale="raw_counts",
        )
        if not np.array_equal(
            validate_matched_bulk_reference(method_input, d3_reference), bulk
        ) or not np.array_equal(
            validate_sctsi_matched_bulk_reference(method_input, sctsi_reference),
            bulk,
        ):
            raise ExternalReferenceDevelopmentError(
                "adapter matched-bulk references changed their measured input"
            )
    except ExternalReferenceDevelopmentError:
        raise
    except (TypeError, ValueError) as error:
        raise ExternalReferenceDevelopmentError(
            f"canonical Tung matched-bulk reference is invalid: {error}"
        ) from error

    receipts: dict[str, SourceReceipt] = {}
    inventories: dict[str, str] = {}
    for method_id in _METHOD_IDS:
        spec = specs[method_id]
        assert spec.source.cache_path is not None
        source_dir = repository / spec.source.cache_path
        try:
            receipts[method_id] = verify_pinned_source(spec, source_dir)
        except Exception as error:
            raise ExternalReferenceDevelopmentError(
                f"pinned source validation failed for {method_id}: {error}"
            ) from error
        inventories[method_id] = _runtime_inventory(
            lock, method_id, locators[method_id], sctsi_library
        )
        if inventories[method_id] != lock.by_id(method_id).inventory_sha256:
            raise ExternalReferenceDevelopmentError(
                f"runtime receipt differs from lock for {method_id}"
            )
    return _Authority(
        repository=repository,
        registry=registry,
        registry_file_sha256=registry_sha,
        specs=MappingProxyType(specs),
        runtime_lock=lock,
        runtime_lock_file_sha256=lock_sha,
        panel=panel,
        method_input=method_input,
        tung=tung,
        bulk_matrix=np.asarray(bulk, dtype="<f8", order="C"),
        sample_ids=sample_ids,
        d3_reference=d3_reference,
        sctsi_reference=sctsi_reference,
        source_evidence=MappingProxyType(_source_evidence(panel)),
        source_receipts=MappingProxyType(receipts),
        locators=MappingProxyType(locators),
        sctsi_library=sctsi_library,
        locator_identities=MappingProxyType(locator_identities),
        runtime_inventory_sha256s=MappingProxyType(inventories),
    )


def _input_metadata(authority: _Authority) -> dict[str, object]:
    value = authority.method_input
    return {
        "schema_version": 1,
        "dataset_id": _DATASET_ID,
        "source_dataset_sha256": value.source_dataset_sha256,
        "obs_ids": list(value.obs_ids),
        "var_ids": list(value.var_ids),
        "shape": list(value.shape),
        "dtype": "<f8",
        "normalization": value.normalization,
        "truth_free": True,
    }


def _bulk_metadata(authority: _Authority) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": _DATASET_ID,
        "reference_id": _REFERENCE_ID,
        "source_kind": "prespecified_measured_bulk_expression",
        "source_sha256": authority.tung.bulk_sample_file_sha256,
        "var_ids": list(authority.method_input.var_ids),
        "sample_ids": list(authority.sample_ids),
        "shape": list(authority.bulk_matrix.shape),
        "dtype": "<f8",
        "expression_scale": "raw_counts",
        "evaluator_truth_used": False,
        "access_scope": "external_reference_adapters_only",
    }


def _reference_bindings(authority: _Authority) -> list[dict[str, object]]:
    return [
        {
            "method_id": method_id,
            "dataset_id": _DATASET_ID,
            "reference_id": _REFERENCE_ID,
            "source_kind": "prespecified_measured_bulk_expression",
            "source_sha256": authority.tung.bulk_sample_file_sha256,
            "matrix_sha256": (
                authority.d3_reference.matrix_sha256
                if method_id == "d3impute"
                else authority.sctsi_reference.matrix_sha256
            ),
            "evaluator_truth_used": False,
        }
        for method_id in _METHOD_IDS
    ]


def _plan_payload(
    authority: _Authority,
    artifact_bindings: Mapping[str, object],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "producer": _PRODUCER,
        "track": "external_reference",
        "dataset_id": _DATASET_ID,
        "method_ids": list(_METHOD_IDS),
        "method_registry_file_sha256": authority.registry_file_sha256,
        "runtime_lock_file_sha256": authority.runtime_lock_file_sha256,
        "method_specs_sha256": {
            method_id: canonical_sha256(asdict(authority.specs[method_id]))
            for method_id in _METHOD_IDS
        },
        "source_evidence": dict(authority.source_evidence),
        "source_evidence_sha256": canonical_sha256(dict(authority.source_evidence)),
        "source_receipts": {
            method_id: asdict(authority.source_receipts[method_id])
            for method_id in _METHOD_IDS
        },
        "runtime_inventory_sha256s": dict(authority.runtime_inventory_sha256s),
        "operational_locators": dict(authority.locator_identities),
        "reference_bindings": _reference_bindings(authority),
        "artifacts": dict(artifact_bindings),
        "scientific_design": {
            "adapter_configuration": "pinned_defaults",
            "bulk_access_scope": "external_reference_adapters_only",
            "evaluator_truth_used": False,
            "endpoint_ids": list(_ENDPOINT_IDS),
            "endpoint_reference_overlap": dict(_ENDPOINT_REFERENCE_OVERLAP),
            "independent_endpoint_ids": [],
            "non_matched_bulk_endpoint_ids": [
                "technical_replicate_concordance"
            ],
            "independence_disclosure": (
                "no_endpoint_is_an_independent_validation_cohort"
            ),
            "matched_bulk_input_reused_by_endpoint_ids": [
                "bulk_pseudobulk_concordance",
                "ercc_recovery",
            ],
            "seed_overrides": False,
        },
    }
    return body | {"plan_sha256": canonical_sha256(body)}


def _endpoint_payload(
    method_id: str,
    output: np.ndarray,
    source: TungSource,
) -> dict[str, object]:
    units = tung_concordance_units(output, source)
    if tuple(sorted(units)) != _ENDPOINT_IDS:
        raise ExternalReferenceDevelopmentError(
            "Tung endpoint implementation differs from the fixed denominator"
        )
    endpoints: list[dict[str, object]] = []
    for endpoint in _ENDPOINT_IDS:
        values = units[endpoint]
        if not values or any(not isinstance(value, EndpointUnit) for value in values):
            raise ExternalReferenceDevelopmentError(
                f"completed {method_id} output has empty Tung endpoint {endpoint}"
            )
        endpoints.append(
            {
                "endpoint": endpoint,
                "reference_overlap": _ENDPOINT_REFERENCE_OVERLAP[endpoint],
                "status": "completed",
                "units": [asdict(value) for value in values],
            }
        )
    return {
        "schema_version": 1,
        "dataset_id": _DATASET_ID,
        "method_id": method_id,
        "status": "completed",
        "endpoints": endpoints,
    }


def _unavailable_endpoint_payload(
    method_id: str,
    reason: str,
    detail_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": _DATASET_ID,
        "method_id": method_id,
        "status": "unavailable",
        "endpoints": [
            {
                "endpoint": endpoint,
                "reference_overlap": _ENDPOINT_REFERENCE_OVERLAP[endpoint],
                "status": "unavailable",
                "reason": reason,
                "reason_detail_sha256": detail_sha256,
            }
            for endpoint in _ENDPOINT_IDS
        ],
    }


def _terminal_reason(error: AdapterUnavailableError) -> tuple[str, dict[str, object]]:
    original = error.reason_code
    if (
        isinstance(original, str)
        and _REASON.fullmatch(original) is not None
        and not any(marker in original for marker in _NONFINAL_MARKERS)
    ):
        reason = original
    else:
        reason = "adapter_unavailable"
    detail = {
        "schema_version": 1,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "original_reason_code": original,
        "detail": error.detail,
    }
    return reason, detail


def _environment_payload(
    authority: _Authority,
    method_id: str,
    *,
    before_inventory: str,
    after_inventory: str,
    source_before: SourceReceipt,
    source_after: SourceReceipt,
    execution: AdapterExecution | None,
    error: AdapterUnavailableError | None,
) -> dict[str, object]:
    command = (
        execution.command if execution is not None else error.command if error else None
    )
    attempt_receipt = None
    if error is not None and hasattr(error, "attempt_receipt"):
        attempt_receipt = asdict(error.attempt_receipt)
    return {
        "schema_version": 1,
        "method_id": method_id,
        "dataset_id": _DATASET_ID,
        "runtime_lock_file_sha256": authority.runtime_lock_file_sha256,
        "runtime_environment_id": method_id,
        "runtime_inventory_sha256_before": before_inventory,
        "runtime_inventory_sha256_after": after_inventory,
        "operational_locator": authority.locator_identities[method_id],
        "sctsi_library_locator": (
            authority.locator_identities["sctsi_library"]
            if method_id == "sctsi"
            else None
        ),
        "source_receipt_before": asdict(source_before),
        "source_receipt_after": asdict(source_after),
        "command": None if command is None else list(command),
        "adapter_environment_receipt": (
            []
            if execution is None
            else [list(value) for value in execution.environment_receipt]
        ),
        "compatibility_log": (
            []
            if execution is None
            else [asdict(value) for value in execution.compatibility_log]
        ),
        "adapter_attempt_receipt": attempt_receipt,
    }


def _post_attempt_authority(
    authority: _Authority,
    method_id: str,
) -> tuple[str, SourceReceipt]:
    if (
        _locator_identity(
            authority.locators[method_id],
            f"{method_id} environment",
            directory=False,
        )
        != authority.locator_identities[method_id]
    ):
        raise ExternalReferenceDevelopmentError(
            f"operational locator changed during {method_id} attempt"
        )
    if (
        method_id == "sctsi"
        and _locator_identity(authority.sctsi_library, "scTsI library", directory=True)
        != authority.locator_identities["sctsi_library"]
    ):
        raise ExternalReferenceDevelopmentError(
            "scTsI library locator changed during attempt"
        )
    inventory = _runtime_inventory(
        authority.runtime_lock,
        method_id,
        authority.locators[method_id],
        authority.sctsi_library,
    )
    spec = authority.specs[method_id]
    assert spec.source.cache_path is not None
    try:
        receipt = verify_pinned_source(
            spec, authority.repository / spec.source.cache_path
        )
    except Exception as error:
        raise ExternalReferenceDevelopmentError(
            f"pinned source changed during {method_id} attempt: {error}"
        ) from error
    return inventory, receipt


def _attempt(
    authority: _Authority,
    output: Path,
    method_id: str,
) -> dict[str, object]:
    before_inventory, source_before = _post_attempt_authority(authority, method_id)
    execution: AdapterExecution | None = None
    adapter_error: AdapterUnavailableError | None = None
    unexpected: Exception | None = None
    work_root = output / "adapter-work"
    work_root.mkdir(exist_ok=True)
    spec = authority.specs[method_id]
    assert spec.source.cache_path is not None
    source_dir = authority.repository / spec.source.cache_path
    try:
        if method_id == "d3impute":
            execution = run_d3impute(
                spec,
                authority.method_input,
                bulk_reference=authority.d3_reference,
                source_dir=source_dir,
                python_executable=authority.locators[method_id],
                work_root=work_root,
            )
        else:
            execution = run_sctsi(
                spec,
                authority.method_input,
                bulk_reference=authority.sctsi_reference,
                source_dir=source_dir,
                rscript=authority.locators[method_id],
                r_library=authority.sctsi_library,
                work_root=work_root,
            )
    except AdapterUnavailableError as error:
        adapter_error = error
    except Exception as error:  # unexpected producer faults must not become evidence
        unexpected = error
    after_inventory, source_after = _post_attempt_authority(authority, method_id)
    try:
        work_root.rmdir()
    except OSError as error:
        raise ExternalReferenceDevelopmentError(
            f"adapter {method_id} left unbound work artifacts"
        ) from error
    if before_inventory != after_inventory or source_before != source_after:
        raise ExternalReferenceDevelopmentError(
            f"runtime or source authority changed during {method_id} attempt"
        )
    if unexpected is not None:
        raise ExternalReferenceDevelopmentError(
            f"unexpected {method_id} adapter failure: {type(unexpected).__name__}"
        ) from unexpected

    base = f"records/{method_id}"
    stdout = execution.stdout if execution is not None else adapter_error.stdout
    stderr = execution.stderr if execution is not None else adapter_error.stderr
    artifacts: dict[str, object] = {
        "stdout": _write_new(output, f"{base}/stdout.bin", stdout),
        "stderr": _write_new(output, f"{base}/stderr.bin", stderr),
    }
    environment = _environment_payload(
        authority,
        method_id,
        before_inventory=before_inventory,
        after_inventory=after_inventory,
        source_before=source_before,
        source_after=source_after,
        execution=execution,
        error=adapter_error,
    )
    artifacts["environment"] = _write_json(
        output, f"{base}/environment.json", environment
    )
    reference = (
        authority.d3_reference if method_id == "d3impute" else authority.sctsi_reference
    )
    run: dict[str, object] = {
        "method_id": method_id,
        "dataset_id": _DATASET_ID,
        "status": "completed" if execution is not None else "unavailable",
        "reason": None,
        "reason_detail_sha256": None,
        "reference_id": _REFERENCE_ID,
        "reference_source_sha256": authority.tung.bulk_sample_file_sha256,
        "reference_matrix_sha256": reference.matrix_sha256,
        "runtime_environment_id": method_id,
        "runtime_inventory_sha256": after_inventory,
        "stdout_sha256": _sha256_bytes(stdout),
        "stderr_sha256": _sha256_bytes(stderr),
        "native_output_sha256": None,
        "evaluator_output_sha256": None,
    }
    if execution is not None:
        native = np.array(
            execution.snapshot.matrix,
            dtype=np.float64,
            copy=True,
            order="C",
            subok=False,
        )
        expected_snapshot = (
            finalize_d3impute_output(spec, authority.method_input, native)
            if method_id == "d3impute"
            else finalize_sctsi_output(spec, authority.method_input, native)
        )
        if expected_snapshot != execution.snapshot:
            raise ExternalReferenceDevelopmentError(
                f"{method_id} adapter snapshot is not reproducible from its native bytes"
            )
        evaluator_counts = (
            d3impute_to_evaluator_counts(authority.method_input, native)
            if method_id == "d3impute"
            else sctsi_to_evaluator_counts(authority.method_input, native)
        )
        evaluator = count_equivalent_to_log2_cp10k(evaluator_counts)
        native_raw = np.asarray(native, dtype="<f8", order="C").tobytes(order="C")
        evaluator_raw = np.asarray(evaluator, dtype="<f8", order="C").tobytes(order="C")
        artifacts["native_output"] = _write_new(
            output, f"{base}/native-output.f64", native_raw
        )
        artifacts["evaluator_output"] = _write_new(
            output, f"{base}/evaluator-output.f64", evaluator_raw
        )
        metrics = _endpoint_payload(method_id, evaluator, authority.tung)
        run["native_output_sha256"] = expected_snapshot.matrix_sha256
        run["evaluator_output_sha256"] = _sha256_bytes(evaluator_raw)
    else:
        assert adapter_error is not None
        reason, detail = _terminal_reason(adapter_error)
        detail_sha = canonical_sha256(detail)
        artifacts["reason_detail"] = _write_json(
            output, f"{base}/reason-detail.json", detail
        )
        run["reason"] = reason
        run["reason_detail_sha256"] = detail_sha
        metrics = _unavailable_endpoint_payload(method_id, reason, detail_sha)
    artifacts["metrics"] = _write_json(output, f"{base}/metrics.json", metrics)
    return {"run": run, "artifacts": artifacts}


def _make_read_only(root: Path) -> None:
    for path in sorted(
        root.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        if path.is_symlink():
            raise ExternalReferenceDevelopmentError(
                "generated external-reference evidence contains a symlink"
            )
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any destination entry."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise ExternalReferenceDevelopmentError(
            "atomic no-overwrite publication is unavailable on this platform"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ExternalReferenceDevelopmentError(
            "external-reference development output appeared during production; "
            "refusing overwrite"
        )
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
        raise ExternalReferenceDevelopmentError(
            "atomic no-overwrite publication is unavailable on this filesystem"
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _validate_immutable_tree(root: Path) -> None:
    try:
        if stat.S_IMODE(root.lstat().st_mode) != 0o555:
            raise ExternalReferenceDevelopmentError(
                "external-reference evidence root is not immutable"
            )
        for path in root.rglob("*"):
            metadata = path.lstat()
            expected = 0o555 if stat.S_ISDIR(metadata.st_mode) else 0o444
            if stat.S_IMODE(metadata.st_mode) != expected:
                raise ExternalReferenceDevelopmentError(
                    "external-reference evidence artifact is not immutable"
                )
    except ExternalReferenceDevelopmentError:
        raise
    except OSError as error:
        raise ExternalReferenceDevelopmentError(
            "external-reference immutable modes cannot be verified"
        ) from error


def _expected_file_set(checkpoint: Mapping[str, object]) -> set[str]:
    paths = {"checkpoint.json"}
    top = checkpoint.get("artifacts")
    records = checkpoint.get("records")
    if not isinstance(top, Mapping) or not isinstance(records, list):
        raise ExternalReferenceDevelopmentError(
            "external-reference checkpoint artifact index is invalid"
        )
    groups: list[Mapping[str, object]] = [top]
    for record in records:
        artifacts = record.get("artifacts") if isinstance(record, Mapping) else None
        if not isinstance(artifacts, Mapping):
            raise ExternalReferenceDevelopmentError(
                "external-reference record artifact index is invalid"
            )
        groups.append(artifacts)
    for group in groups:
        for value in group.values():
            if not isinstance(value, Mapping):
                raise ExternalReferenceDevelopmentError(
                    "external-reference artifact binding is invalid"
                )
            relative = _relative_path(value.get("path"), "artifact")
            rendered = relative.as_posix()
            if rendered in paths:
                raise ExternalReferenceDevelopmentError(
                    "external-reference artifact path is duplicated"
                )
            paths.add(rendered)
    return paths


def _validate_closed_tree(output: Path, expected_files: set[str]) -> None:
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in output.rglob("*"):
        relative = path.relative_to(output).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ExternalReferenceDevelopmentError(
                "external-reference evidence tree contains a symlink"
            )
        if stat.S_ISREG(metadata.st_mode):
            observed_files.add(relative)
        elif stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(relative)
        else:
            raise ExternalReferenceDevelopmentError(
                "external-reference evidence tree contains a special file"
            )
    expected_directories = {
        parent.as_posix()
        for value in expected_files
        for parent in PurePosixPath(value).parents
        if parent.as_posix() != "."
    }
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ExternalReferenceDevelopmentError(
            "external-reference evidence tree has missing or extra artifacts"
        )


def _matrix_from_raw(
    raw: bytes,
    shape: Sequence[object],
    label: str,
) -> np.ndarray:
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or len(raw) != int(shape[0]) * int(shape[1]) * 8
    ):
        raise ExternalReferenceDevelopmentError(f"{label} shape or bytes are invalid")
    return np.frombuffer(raw, dtype="<f8").reshape((int(shape[0]), int(shape[1])))


def _fixed_work_paths(
    values: Sequence[str],
    output: Path,
    *,
    prefix: str,
    names: tuple[str, ...],
) -> tuple[str, ...]:
    if len(values) != len(names):
        raise ExternalReferenceDevelopmentError("adapter command work paths are invalid")
    paths = tuple(Path(value) for value in values)
    parents = {path.parent for path in paths}
    if (
        len(parents) != 1
        or any(not path.is_absolute() or ".." in path.parts for path in paths)
        or tuple(path.name for path in paths) != names
    ):
        raise ExternalReferenceDevelopmentError("adapter command work paths are invalid")
    parent = next(iter(parents))
    work_root = parent.parent
    current_staging = work_root == output / "adapter-work"
    published_staging = (
        work_root.name == "adapter-work"
        and work_root.parent.parent == output.parent
        and work_root.parent.name.startswith(
            ".competition-external-reference-staging-"
        )
    )
    if (
        not (current_staging or published_staging)
        or not parent.name.startswith(prefix)
    ):
        raise ExternalReferenceDevelopmentError("adapter command work paths are invalid")
    return tuple(str(path) for path in paths)


def _validate_fixed_command(
    command: Sequence[str],
    output: Path,
    authority: _Authority,
    method_id: str,
) -> None:
    spec = authority.specs[method_id]
    assert spec.source.cache_path is not None
    source = authority.repository / spec.source.cache_path
    if method_id == "d3impute":
        if len(command) != 19:
            raise ExternalReferenceDevelopmentError(
                "adapter command is not the fixed d3impute invocation"
            )
        work = _fixed_work_paths(
            command[6:10],
            output,
            prefix="maskimpute-d3impute-",
            names=("input.npy", "bulk.npy", "output.npy", "receipt.tsv"),
        )
        config = D3ImputeConfig()
        expected = (
            str(authority.locators[method_id]),
            "-B",
            "-I",
            "-c",
            _D3IMPUTE_DRIVER,
            str((source / "PYTHON.zip").resolve()),
            *work,
            str(config.fixed_seed),
            str(config.neighbors),
            str(config.latent_dimension),
            str(config.iterations),
            repr(float(config.sparsity)),
            repr(float(config.cell_regularization)),
            repr(float(config.gene_regularization)),
            authority.d3_reference.reference_id,
            authority.d3_reference.source_sha256,
        )
    else:
        if len(command) != 18:
            raise ExternalReferenceDevelopmentError(
                "adapter command is not the fixed sctsi invocation"
            )
        work = _fixed_work_paths(
            command[5:9],
            output,
            prefix="maskimpute-sctsi-",
            names=("input.bin", "bulk-average.bin", "output.bin", "receipt.tsv"),
        )
        config = SCTSIConfig()
        cells, genes = authority.method_input.shape
        expected = (
            str(authority.locators[method_id]),
            "--vanilla",
            "-e",
            _SCTSI_DRIVER,
            str((source / "code/scTsI.R").resolve()),
            *work,
            str(authority.sctsi_library),
            str(genes),
            str(cells),
            repr(float(config.threshold)),
            str(config.cell_neighbors),
            str(config.gene_neighbors),
            authority.sctsi_reference.reference_id,
            authority.sctsi_reference.source_sha256,
            authority.sctsi_reference.matrix_sha256,
        )
    if tuple(command) != expected:
        raise ExternalReferenceDevelopmentError(
            f"adapter command is not the fixed {method_id} invocation"
        )


def _validate_completed_environment_receipt(
    rows: Sequence[Sequence[str]],
    authority: _Authority,
    method_id: str,
) -> None:
    receipt = {row[0]: row[1] for row in rows}
    spec = authority.specs[method_id]
    assert spec.source.cache_path is not None
    source = authority.repository / spec.source.cache_path
    if method_id == "d3impute":
        expected_keys = {
            "bulk_reference_id",
            "bulk_reference_sha256",
            "inference_module",
            "numpy_version",
            "pandas_version",
            "python_version",
            "scipy_version",
            "sklearn_version",
            "source_archive",
        }
        archive = str((source / "PYTHON.zip").resolve())
        valid = (
            set(receipt) == expected_keys
            and receipt.get("bulk_reference_id")
            == authority.d3_reference.reference_id
            and receipt.get("bulk_reference_sha256")
            == authority.d3_reference.source_sha256
            and receipt.get("source_archive") == archive
            and isinstance(receipt.get("inference_module"), str)
            and receipt["inference_module"].startswith(archive + "/PYTHON/")
        )
    else:
        expected_keys = {
            "bulk_constraint_scale",
            "bulk_reference_id",
            "bulk_reference_input_scale",
            "bulk_reference_matrix_sha256",
            "bulk_reference_source_sha256",
            "cpm_target",
            "devtools_version",
            "fnn_version",
            "fpc_version",
            "glmnet_version",
            "matrix_version",
            "mclust_version",
            "metrics_version",
            "ngram_version",
            "r_library",
            "r_library_paths",
            "r_version",
            "sctsi_native_output_scale",
            "single_cell_input_scale",
            "sctsi_source_file",
        }
        library_paths = receipt.get("r_library_paths", "").split(";")
        valid = (
            set(receipt) == expected_keys
            and receipt.get("bulk_constraint_scale") == "cpm"
            and receipt.get("bulk_reference_id")
            == authority.sctsi_reference.reference_id
            and receipt.get("bulk_reference_input_scale") == "raw_counts"
            and receipt.get("bulk_reference_matrix_sha256")
            == authority.sctsi_reference.matrix_sha256
            and receipt.get("bulk_reference_source_sha256")
            == authority.sctsi_reference.source_sha256
            and receipt.get("cpm_target") == "1000000"
            and receipt.get("r_library") == str(authority.sctsi_library)
            and len(library_paths) == 2
            and library_paths[0] == str(authority.sctsi_library)
            and all(library_paths)
            and receipt.get("sctsi_native_output_scale") == "cpm"
            and receipt.get("single_cell_input_scale") == "cpm"
            and receipt.get("sctsi_source_file")
            == str((source / "code/scTsI.R").resolve())
        )
    if not valid:
        raise ExternalReferenceDevelopmentError(
            f"environment receipt is invalid for {method_id}"
        )


def _validate_environment_payload(
    value: Mapping[str, object],
    output: Path,
    authority: _Authority,
    method_id: str,
    status: str,
    *,
    stdout_sha256: str,
    stderr_sha256: str,
) -> None:
    expected_receipt = asdict(authority.source_receipts[method_id])
    expected_inventory = authority.runtime_inventory_sha256s[method_id]
    command = value.get("command")
    adapter_receipt = value.get("adapter_environment_receipt")
    compatibility = value.get("compatibility_log")
    expected_keys = {
        "schema_version",
        "method_id",
        "dataset_id",
        "runtime_lock_file_sha256",
        "runtime_environment_id",
        "runtime_inventory_sha256_before",
        "runtime_inventory_sha256_after",
        "operational_locator",
        "sctsi_library_locator",
        "source_receipt_before",
        "source_receipt_after",
        "command",
        "adapter_environment_receipt",
        "compatibility_log",
        "adapter_attempt_receipt",
    }
    if command is None:
        command_valid = status == "unavailable"
    elif isinstance(command, list) and command:
        command_valid = command[0] == str(authority.locators[method_id]) and all(
            isinstance(part, str) for part in command
        )
    else:
        command_valid = False
    receipt_valid = (
        isinstance(adapter_receipt, list)
        and all(
            isinstance(row, list)
            and len(row) == 2
            and all(isinstance(part, str) and part for part in row)
            for row in adapter_receipt
        )
        and len({row[0] for row in adapter_receipt}) == len(adapter_receipt)
    )
    compatibility_valid = isinstance(compatibility, list) and all(
        isinstance(row, Mapping)
        and set(row) == {"code", "detail"}
        and isinstance(row.get("code"), str)
        and bool(row.get("code"))
        and isinstance(row.get("detail"), str)
        and bool(row.get("detail"))
        for row in compatibility
    )
    if (
        set(value) != expected_keys
        or value.get("schema_version") != 1
        or value.get("method_id") != method_id
        or value.get("dataset_id") != _DATASET_ID
        or value.get("runtime_lock_file_sha256") != authority.runtime_lock_file_sha256
        or value.get("runtime_environment_id") != method_id
        or value.get("runtime_inventory_sha256_before") != expected_inventory
        or value.get("runtime_inventory_sha256_after") != expected_inventory
        or value.get("operational_locator") != authority.locator_identities[method_id]
        or value.get("source_receipt_before") != expected_receipt
        or value.get("source_receipt_after") != expected_receipt
        or not command_valid
        or not receipt_valid
        or not compatibility_valid
        or (status == "completed" and (not adapter_receipt or not compatibility))
        or (
            status == "unavailable"
            and (adapter_receipt != [] or compatibility != [])
        )
        or (
            value.get("adapter_attempt_receipt") is not None
            and not isinstance(value.get("adapter_attempt_receipt"), Mapping)
        )
    ):
        raise ExternalReferenceDevelopmentError(
            f"environment receipt is invalid for {method_id}"
        )
    expected_library = (
        authority.locator_identities["sctsi_library"] if method_id == "sctsi" else None
    )
    if value.get("sctsi_library_locator") != expected_library:
        raise ExternalReferenceDevelopmentError(
            f"environment library receipt is invalid for {method_id}"
        )
    if status == "completed" and tuple(
        row["code"] for row in compatibility
    ) != _COMPATIBILITY_CODES[method_id]:
        raise ExternalReferenceDevelopmentError(
            f"compatibility disclosure is incomplete for {method_id}"
        )
    if command is not None:
        _validate_fixed_command(command, output, authority, method_id)
    if status == "completed":
        assert isinstance(adapter_receipt, list)
        _validate_completed_environment_receipt(
            adapter_receipt, authority, method_id
        )
    attempt = value.get("adapter_attempt_receipt")
    if status == "completed" or method_id == "d3impute":
        if attempt is not None:
            raise ExternalReferenceDevelopmentError(
                f"adapter attempt receipt is invalid for {method_id}"
            )
        return
    if not isinstance(attempt, Mapping) or set(attempt) != {
        "source_revision",
        "source_tree",
        "source_url",
        "environment_id",
        "environment_registry_status",
        "executable",
        "r_library",
        "reference_id",
        "reference_source_sha256",
        "reference_matrix_sha256",
        "outcome",
        "reason_code",
        "command",
        "stdout_sha256",
        "stderr_sha256",
    }:
        raise ExternalReferenceDevelopmentError(
            "adapter attempt receipt is invalid for sctsi"
        )
    reason_code = attempt.get("reason_code")
    expected_source = authority.source_receipts[method_id]
    expected_reference = authority.sctsi_reference
    if (
        attempt.get("source_revision") != expected_source.revision
        or attempt.get("source_tree") != expected_source.tree
        or attempt.get("source_url") != expected_source.url
        or attempt.get("environment_id") != authority.specs[method_id].environment.id
        or attempt.get("environment_registry_status") != "ready"
        or attempt.get("executable") != str(authority.locators[method_id])
        or attempt.get("r_library") != str(authority.sctsi_library)
        or attempt.get("reference_id") != expected_reference.reference_id
        or attempt.get("reference_source_sha256")
        != expected_reference.source_sha256
        or attempt.get("reference_matrix_sha256")
        != expected_reference.matrix_sha256
        or attempt.get("outcome") != "unavailable"
        or not isinstance(reason_code, str)
        or _REASON.fullmatch(reason_code) is None
        or any(marker in reason_code for marker in _NONFINAL_MARKERS)
        or attempt.get("command") != command
        or attempt.get("stdout_sha256") != stdout_sha256
        or attempt.get("stderr_sha256") != stderr_sha256
    ):
        raise ExternalReferenceDevelopmentError(
            "adapter attempt receipt is invalid for sctsi"
        )


def _validate_record(
    output: Path,
    record: object,
    authority: _Authority,
    method_id: str,
) -> None:
    if not isinstance(record, Mapping) or set(record) != {"run", "artifacts"}:
        raise ExternalReferenceDevelopmentError(
            f"external-reference record schema is invalid for {method_id}"
        )
    run = record.get("run")
    artifacts = record.get("artifacts")
    if not isinstance(run, Mapping) or not isinstance(artifacts, Mapping):
        raise ExternalReferenceDevelopmentError(
            f"external-reference record is malformed for {method_id}"
        )
    expected_run_keys = {
        "method_id",
        "dataset_id",
        "status",
        "reason",
        "reason_detail_sha256",
        "reference_id",
        "reference_source_sha256",
        "reference_matrix_sha256",
        "runtime_environment_id",
        "runtime_inventory_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "native_output_sha256",
        "evaluator_output_sha256",
    }
    reference = (
        authority.d3_reference if method_id == "d3impute" else authority.sctsi_reference
    )
    status = run.get("status")
    if (
        set(run) != expected_run_keys
        or run.get("method_id") != method_id
        or run.get("dataset_id") != _DATASET_ID
        or status not in {"completed", "unavailable"}
        or run.get("reference_id") != _REFERENCE_ID
        or run.get("reference_source_sha256") != authority.tung.bulk_sample_file_sha256
        or run.get("reference_matrix_sha256") != reference.matrix_sha256
        or run.get("runtime_environment_id") != method_id
        or run.get("runtime_inventory_sha256")
        != authority.runtime_inventory_sha256s[method_id]
    ):
        raise ExternalReferenceDevelopmentError(
            f"external-reference run binding is invalid for {method_id}"
        )
    expected_artifact_keys = {
        "stdout",
        "stderr",
        "environment",
        "metrics",
    } | (
        {"native_output", "evaluator_output"}
        if status == "completed"
        else {"reason_detail"}
    )
    if set(artifacts) != expected_artifact_keys:
        raise ExternalReferenceDevelopmentError(
            f"external-reference artifacts are incomplete for {method_id}"
        )
    base = f"records/{method_id}"
    expected_paths = {
        "stdout": f"{base}/stdout.bin",
        "stderr": f"{base}/stderr.bin",
        "environment": f"{base}/environment.json",
        "metrics": f"{base}/metrics.json",
    } | (
        {
            "native_output": f"{base}/native-output.f64",
            "evaluator_output": f"{base}/evaluator-output.f64",
        }
        if status == "completed"
        else {"reason_detail": f"{base}/reason-detail.json"}
    )
    if any(
        not isinstance(artifacts.get(name), Mapping)
        or artifacts[name].get("path") != expected
        for name, expected in expected_paths.items()
    ):
        raise ExternalReferenceDevelopmentError(
            f"external-reference artifact path is not fixed for {method_id}"
        )
    stdout, _ = _read_binding(output, artifacts["stdout"], f"{method_id} stdout")
    stderr, _ = _read_binding(output, artifacts["stderr"], f"{method_id} stderr")
    if _sha256_bytes(stdout) != run.get("stdout_sha256") or _sha256_bytes(
        stderr
    ) != run.get("stderr_sha256"):
        raise ExternalReferenceDevelopmentError(
            f"adapter logs differ from run receipt for {method_id}"
        )
    environment_raw, _ = _read_binding(
        output, artifacts["environment"], f"{method_id} environment"
    )
    environment = _json_value(environment_raw, f"{method_id} environment")
    _validate_environment_payload(
        environment,
        output,
        authority,
        method_id,
        str(status),
        stdout_sha256=_sha256_bytes(stdout),
        stderr_sha256=_sha256_bytes(stderr),
    )
    metrics_raw, _ = _read_binding(output, artifacts["metrics"], f"{method_id} metrics")
    metrics = _json_value(metrics_raw, f"{method_id} metrics")
    if status == "completed":
        if (
            run.get("reason") is not None
            or run.get("reason_detail_sha256") is not None
            or not environment.get("compatibility_log")
        ):
            raise ExternalReferenceDevelopmentError(
                f"completed {method_id} record has a failure disposition"
            )
        native_raw, _ = _read_binding(
            output, artifacts["native_output"], f"{method_id} native output"
        )
        native = _matrix_from_raw(
            native_raw, list(authority.method_input.shape), f"{method_id} native output"
        )
        spec = authority.specs[method_id]
        snapshot = (
            finalize_d3impute_output(spec, authority.method_input, native)
            if method_id == "d3impute"
            else finalize_sctsi_output(spec, authority.method_input, native)
        )
        if snapshot.matrix_sha256 != run.get("native_output_sha256"):
            raise ExternalReferenceDevelopmentError(
                f"native output semantic checksum differs for {method_id}"
            )
        counts = (
            d3impute_to_evaluator_counts(authority.method_input, native)
            if method_id == "d3impute"
            else sctsi_to_evaluator_counts(authority.method_input, native)
        )
        expected_evaluator = count_equivalent_to_log2_cp10k(counts)
        evaluator_raw, _ = _read_binding(
            output,
            artifacts["evaluator_output"],
            f"{method_id} evaluator output",
        )
        evaluator = _matrix_from_raw(
            evaluator_raw,
            list(authority.method_input.shape),
            f"{method_id} evaluator output",
        )
        if evaluator_raw != np.asarray(
            expected_evaluator, dtype="<f8", order="C"
        ).tobytes(order="C") or _sha256_bytes(evaluator_raw) != run.get(
            "evaluator_output_sha256"
        ):
            raise ExternalReferenceDevelopmentError(
                f"evaluator conversion differs for {method_id}"
            )
        expected_metrics = _endpoint_payload(method_id, evaluator, authority.tung)
        if metrics != expected_metrics:
            raise ExternalReferenceDevelopmentError(
                f"Tung endpoint metrics differ for {method_id}"
            )
    else:
        reason = run.get("reason")
        detail_sha = run.get("reason_detail_sha256")
        if (
            not isinstance(reason, str)
            or _REASON.fullmatch(reason) is None
            or any(marker in reason for marker in _NONFINAL_MARKERS)
        ):
            raise ExternalReferenceDevelopmentError(
                f"unavailable {method_id} record has a nonterminal reason"
            )
        _sha256(detail_sha, f"{method_id} reason detail")
        if (
            run.get("native_output_sha256") is not None
            or run.get("evaluator_output_sha256") is not None
            or environment.get("compatibility_log")
        ):
            raise ExternalReferenceDevelopmentError(
                f"unavailable {method_id} record contains completed output evidence"
            )
        detail_raw, _ = _read_binding(
            output, artifacts["reason_detail"], f"{method_id} reason detail"
        )
        detail = _json_value(detail_raw, f"{method_id} reason detail")
        if canonical_sha256(detail) != detail_sha:
            raise ExternalReferenceDevelopmentError(
                f"unavailable {method_id} reason detail differs"
            )
        if method_id == "sctsi" and (
            not isinstance(environment.get("adapter_attempt_receipt"), Mapping)
            or environment["adapter_attempt_receipt"].get("reason_code")
            != detail.get("original_reason_code")
        ):
            raise ExternalReferenceDevelopmentError(
                "adapter attempt receipt differs from sctsi failure detail"
            )
        expected_metrics = _unavailable_endpoint_payload(
            method_id, reason, str(detail_sha)
        )
        if metrics != expected_metrics:
            raise ExternalReferenceDevelopmentError(
                f"unavailable endpoint metrics differ for {method_id}"
            )


def _load_external_reference_evidence(
    repository: Path,
    output: Path,
) -> ValidatedExternalReferenceEvidence:
    checkpoint_path = output / "checkpoint.json"
    checkpoint_raw = _stable_read(checkpoint_path, "external-reference checkpoint")
    checkpoint = _json_value(checkpoint_raw, "external-reference checkpoint")
    expected_checkpoint_keys = {
        "schema_version",
        "producer",
        "track",
        "status",
        "dataset_source_id",
        "method_ids",
        "eligible_dataset_ids",
        "method_registry_file_sha256",
        "runtime_lock_file_sha256",
        "source_evidence_sha256",
        "locator_bindings_sha256",
        "plan_sha256",
        "planned_run_count",
        "reference_bindings",
        "artifacts",
        "records",
        "checkpoint_sha256",
    }
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    if (
        set(checkpoint) != expected_checkpoint_keys
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("producer") != _PRODUCER
        or checkpoint.get("track") != "external_reference"
        or checkpoint.get("status") != "completed"
        or checkpoint.get("dataset_source_id") != _DATASET_ID
        or checkpoint.get("method_ids") != list(_METHOD_IDS)
        or checkpoint.get("eligible_dataset_ids") != [_DATASET_ID]
        or checkpoint.get("planned_run_count") != len(_METHOD_IDS)
        or checkpoint.get("checkpoint_sha256") != canonical_sha256(unsigned)
    ):
        raise ExternalReferenceDevelopmentError(
            "external-reference checkpoint is not a completed production artifact"
        )
    artifacts = checkpoint.get("artifacts")
    records = checkpoint.get("records")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {
            "input_metadata",
            "input_counts",
            "bulk_metadata",
            "bulk_reference",
            "plan",
        }
        or not isinstance(records, list)
        or len(records) != len(_METHOD_IDS)
    ):
        raise ExternalReferenceDevelopmentError(
            "external-reference checkpoint has incomplete production artifacts"
        )
    if any(
        not isinstance(artifacts.get(name), Mapping)
        or artifacts[name].get("path") != expected
        for name, expected in _TOP_ARTIFACT_PATHS.items()
    ):
        raise ExternalReferenceDevelopmentError(
            "external-reference top-level artifact path is not fixed"
        )
    expected_files = _expected_file_set(checkpoint)
    _validate_closed_tree(output, expected_files)

    plan_raw, _ = _read_binding(output, artifacts["plan"], "execution plan")
    plan = _json_value(plan_raw, "execution plan")
    locator_values = plan.get("operational_locators")
    if not isinstance(locator_values, Mapping) or set(locator_values) != {
        "d3impute",
        "sctsi",
        "sctsi_library",
    }:
        raise ExternalReferenceDevelopmentError(
            "execution plan operational locators are invalid"
        )
    try:
        environments = {
            method_id: Path(str(locator_values[method_id]["path"]))
            for method_id in _METHOD_IDS
        }
        sctsi_library = Path(str(locator_values["sctsi_library"]["path"]))
    except (KeyError, TypeError) as error:
        raise ExternalReferenceDevelopmentError(
            "execution plan operational locator paths are invalid"
        ) from error
    authority = _derive_authority(
        repository,
        environments=environments,
        sctsi_library=sctsi_library,
    )
    if (
        checkpoint.get("method_registry_file_sha256") != authority.registry_file_sha256
        or checkpoint.get("runtime_lock_file_sha256")
        != authority.runtime_lock_file_sha256
        or checkpoint.get("source_evidence_sha256")
        != canonical_sha256(dict(authority.source_evidence))
        or checkpoint.get("locator_bindings_sha256")
        != canonical_sha256(dict(authority.locator_identities))
    ):
        raise ExternalReferenceDevelopmentError(
            "external-reference checkpoint authority changed"
        )

    input_metadata_raw, _ = _read_binding(
        output, artifacts["input_metadata"], "Tung input metadata"
    )
    input_metadata = _json_value(input_metadata_raw, "Tung input metadata")
    input_counts_raw, _ = _read_binding(
        output, artifacts["input_counts"], "Tung input counts"
    )
    bulk_metadata_raw, _ = _read_binding(
        output, artifacts["bulk_metadata"], "Tung bulk metadata"
    )
    bulk_metadata = _json_value(bulk_metadata_raw, "Tung bulk metadata")
    bulk_raw, _ = _read_binding(
        output, artifacts["bulk_reference"], "Tung bulk reference"
    )
    if (
        input_metadata != _input_metadata(authority)
        or input_counts_raw
        != np.asarray(authority.method_input.counts, dtype="<f8", order="C").tobytes(
            order="C"
        )
        or bulk_metadata != _bulk_metadata(authority)
        or bulk_raw
        != np.asarray(authority.bulk_matrix, dtype="<f8", order="C").tobytes(order="C")
    ):
        raise ExternalReferenceDevelopmentError(
            "persisted Tung input or measured reference bytes differ from source"
        )
    plan_artifacts = {key: value for key, value in artifacts.items() if key != "plan"}
    expected_plan = _plan_payload(authority, plan_artifacts)
    if (
        plan != expected_plan
        or plan.get("plan_sha256") != checkpoint.get("plan_sha256")
        or checkpoint.get("reference_bindings") != _reference_bindings(authority)
    ):
        raise ExternalReferenceDevelopmentError(
            "external-reference execution plan or reference bindings changed"
        )
    observed_method_ids: list[str] = []
    for expected_method_id, record in zip(_METHOD_IDS, records, strict=True):
        run = record.get("run") if isinstance(record, Mapping) else None
        if not isinstance(run, Mapping):
            raise ExternalReferenceDevelopmentError(
                "external-reference record lacks its run disposition"
            )
        observed_method_ids.append(str(run.get("method_id")))
        _validate_record(output, record, authority, expected_method_id)
    if tuple(observed_method_ids) != _METHOD_IDS:
        raise ExternalReferenceDevelopmentError(
            "external-reference records are not in their fixed method order"
        )
    return ValidatedExternalReferenceEvidence(
        output_directory=output,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=_sha256_bytes(checkpoint_raw),
        checkpoint=MappingProxyType(checkpoint),
        dataset_id=_DATASET_ID,
        method_ids=_METHOD_IDS,
    )


def load_external_reference_evidence(
    repository: Path,
) -> ValidatedExternalReferenceEvidence:
    """Freshly validate the production checkpoint, every byte, and all endpoints."""

    root = _repository_root(repository)
    _safe_directory_chain(root, OUTPUT_RELATIVE_PATH.parent, create=False)
    output = root / OUTPUT_RELATIVE_PATH
    try:
        metadata = output.lstat()
    except OSError as error:
        raise ExternalReferenceDevelopmentError(
            "external-reference development output is unavailable"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ExternalReferenceDevelopmentError(
            "external-reference development output is not a non-symlink directory"
        )
    evidence = _load_external_reference_evidence(root, output)
    _validate_immutable_tree(output)
    return evidence


def run_external_reference_development(
    repository: Path,
    *,
    environments: Mapping[str, Path],
    sctsi_library: Path,
) -> ValidatedExternalReferenceEvidence:
    """Run both pinned adapters once on the fixed Tung measured-bulk track."""

    root = _repository_root(repository)
    final_output = root / OUTPUT_RELATIVE_PATH
    _safe_directory_chain(root, OUTPUT_RELATIVE_PATH.parent, create=True)
    if os.path.lexists(final_output):
        raise ExternalReferenceDevelopmentError(
            "external-reference development output already exists; refusing overwrite"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=".competition-external-reference-staging-",
            dir=final_output.parent,
        )
    )
    published = False
    try:
        authority = _derive_authority(
            root,
            environments=environments,
            sctsi_library=sctsi_library,
        )
        input_metadata_binding = _write_json(
            staging, "inputs/tung-method-input.json", _input_metadata(authority)
        )
        input_counts_binding = _write_new(
            staging,
            "inputs/tung-counts.f64",
            np.asarray(authority.method_input.counts, dtype="<f8", order="C").tobytes(
                order="C"
            ),
        )
        bulk_metadata_binding = _write_json(
            staging, "references/tung-bulk-reference.json", _bulk_metadata(authority)
        )
        bulk_reference_binding = _write_new(
            staging,
            "references/tung-bulk-counts.f64",
            np.asarray(authority.bulk_matrix, dtype="<f8", order="C").tobytes(
                order="C"
            ),
        )
        plan_artifacts: dict[str, object] = {
            "input_metadata": input_metadata_binding,
            "input_counts": input_counts_binding,
            "bulk_metadata": bulk_metadata_binding,
            "bulk_reference": bulk_reference_binding,
        }
        plan = _plan_payload(authority, plan_artifacts)
        plan_binding = _write_json(staging, "plan.json", plan)
        records = [_attempt(authority, staging, method_id) for method_id in _METHOD_IDS]
        artifacts = plan_artifacts | {"plan": plan_binding}
        checkpoint_body: dict[str, object] = {
            "schema_version": 1,
            "producer": _PRODUCER,
            "track": "external_reference",
            "status": "completed",
            "dataset_source_id": _DATASET_ID,
            "method_ids": list(_METHOD_IDS),
            "eligible_dataset_ids": [_DATASET_ID],
            "method_registry_file_sha256": authority.registry_file_sha256,
            "runtime_lock_file_sha256": authority.runtime_lock_file_sha256,
            "source_evidence_sha256": canonical_sha256(dict(authority.source_evidence)),
            "locator_bindings_sha256": canonical_sha256(
                dict(authority.locator_identities)
            ),
            "plan_sha256": plan["plan_sha256"],
            "planned_run_count": len(_METHOD_IDS),
            "reference_bindings": _reference_bindings(authority),
            "artifacts": artifacts,
            "records": records,
        }
        checkpoint = checkpoint_body | {
            "checkpoint_sha256": canonical_sha256(checkpoint_body)
        }
        _write_json(staging, "checkpoint.json", checkpoint)
        _load_external_reference_evidence(root, staging)
        _make_read_only(staging)
        _rename_directory_noreplace(staging, final_output)
        published = True
        return load_external_reference_evidence(root)
    except ExternalReferenceDevelopmentError:
        raise
    except Exception as error:
        raise ExternalReferenceDevelopmentError(
            f"external-reference development production failed: {error}"
        ) from error
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "CHECKPOINT_RELATIVE_PATH",
    "ExternalReferenceDevelopmentError",
    "OUTPUT_RELATIVE_PATH",
    "ValidatedExternalReferenceEvidence",
    "load_external_reference_evidence",
    "run_external_reference_development",
]
