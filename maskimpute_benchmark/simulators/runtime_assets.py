"""Path-independent authority for external publication simulator assets."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any

from ..protocol import canonical_sha256
from ..runtime_environments import (
    RuntimeEnvironmentError,
    load_runtime_environment_lock,
    validate_runtime_environment_lock,
)
from ..sources import SourceLedgerError, load_source_ledger, verify_fetched_sources


_AUTHORITY_NAME = "study/simulator_runtime_assets.json"
_AUTHORITY_SCHEMA = "maskimpute-simulator-runtime-assets-authority-v1"
_RECEIPT_SCHEMA = "maskimpute-simulator-runtime-assets-receipt-v1"
_R_RECEIPT_SCHEMA = "maskimpute-simulator-r-runtime-receipt-v1"
_R_ENVIRONMENT_ID = "simulator-r"
_R_LOCK_PATH = "study/simulator_r_environment.lock.json"
_SOURCE_IDS = ("baron-pancreas-umi", "sergio", "sparsim", "symsim")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = object()


class SimulatorRuntimeAssetsError(ValueError):
    """Raised when simulator assets or their tracked authority drift."""


@dataclass(frozen=True, slots=True, init=False)
class SimulatorRuntimeAssets:
    """Validated immutable paths plus a path-independent semantic receipt."""

    external_root: Path
    r_environment: Path
    semantic_sha256: str
    _semantic_json: bytes = field(repr=False)
    _filesystem_identity_sha256: str = field(repr=False)
    _authority_external_root: Path = field(repr=False)
    _authority_r_environment: Path = field(repr=False)
    _snapshot_owner: tempfile.TemporaryDirectory[str] = field(repr=False)
    _repository: Path = field(repr=False)
    _require_outside_repository: bool = field(repr=False)
    _token: object = field(repr=False)

    @property
    def semantic_receipt(self) -> dict[str, object]:
        """Return a defensive copy of the path-independent receipt."""

        value = json.loads(self._semantic_json.decode("utf-8"))
        assert isinstance(value, dict)
        return value


def _reject_constant(value: str) -> None:
    raise SimulatorRuntimeAssetsError(f"invalid JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulatorRuntimeAssetsError(
                f"duplicate runtime authority field {key!r}"
            )
        result[key] = value
    return result


def _canonical_bytes(value: object, *, pretty: bool) -> bytes:
    try:
        if pretty:
            text = json.dumps(value, allow_nan=False, indent=2, sort_keys=True)
        else:
            text = json.dumps(
                value, allow_nan=False, separators=(",", ":"), sort_keys=True
            )
    except (TypeError, ValueError) as error:
        raise SimulatorRuntimeAssetsError(
            "runtime asset receipt is not canonical JSON"
        ) from error
    return (text + "\n").encode("utf-8")


def _read_authority(path: Path) -> tuple[dict[str, object], str]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o002
        ):
            raise SimulatorRuntimeAssetsError(
                "simulator runtime authority is not a secure unique regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    except SimulatorRuntimeAssetsError:
        raise
    except OSError as error:
        raise SimulatorRuntimeAssetsError(
            "simulator runtime authority is unavailable"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except SimulatorRuntimeAssetsError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SimulatorRuntimeAssetsError(
            "simulator runtime authority is invalid JSON"
        ) from error
    if not isinstance(value, dict) or raw != _canonical_bytes(value, pretty=True):
        raise SimulatorRuntimeAssetsError(
            "simulator runtime authority must be a canonical JSON object"
        )
    if set(value) != {
        "schema",
        "source_ledger_sha256",
        "source_snapshot",
        "r_environment",
    }:
        raise SimulatorRuntimeAssetsError(
            "simulator runtime authority has wrong fields"
        )
    if value.get("schema") != _AUTHORITY_SCHEMA:
        raise SimulatorRuntimeAssetsError("simulator runtime authority schema mismatch")
    source_digest = value.get("source_ledger_sha256")
    environment = value.get("r_environment")
    if not isinstance(source_digest, str) or _SHA256.fullmatch(source_digest) is None:
        raise SimulatorRuntimeAssetsError(
            "simulator source-ledger authority checksum is invalid"
        )
    _validate_tree_authority(value.get("source_snapshot"), "simulator source snapshot")
    _validate_environment_authority(environment)
    return value, hashlib.sha256(raw).hexdigest()


def _secure_directory(path: Path, name: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    candidate = path.expanduser().absolute()
    for component in (candidate, *candidate.parents):
        try:
            metadata = component.lstat()
        except OSError as error:
            raise SimulatorRuntimeAssetsError(f"{name} is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SimulatorRuntimeAssetsError(f"{name} path must not contain symlinks")
    try:
        resolved = candidate.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError as error:
        raise SimulatorRuntimeAssetsError(f"{name} is unavailable") from error
    if resolved != candidate or not stat.S_ISDIR(metadata.st_mode):
        raise SimulatorRuntimeAssetsError(
            f"{name} must be a canonical non-symlink directory"
        )
    if metadata.st_mode & 0o002:
        raise SimulatorRuntimeAssetsError(f"{name} must not be world-writable")
    return resolved


def _require_outside(path: Path, repository: Path, name: str) -> None:
    try:
        path.relative_to(repository)
    except ValueError:
        try:
            repository.relative_to(path)
        except ValueError:
            return
    raise SimulatorRuntimeAssetsError(f"{name} must be outside the final repository")


def _file_sha256(path: Path, name: str) -> str:
    descriptor: int | None = None
    try:
        before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise SimulatorRuntimeAssetsError(
                f"{name} must be a unique regular non-symlink file"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        named_after = path.lstat()
    except SimulatorRuntimeAssetsError:
        raise
    except OSError as error:
        raise SimulatorRuntimeAssetsError(f"{name} cannot be hashed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named_after):
        raise SimulatorRuntimeAssetsError(f"{name} changed while hashing")
    return digest.hexdigest()


def _validate_environment_authority(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "environment_id",
        "lock_file_sha256",
        "lock_path",
        "tree_entry_count",
        "tree_sha256",
    }:
        raise SimulatorRuntimeAssetsError(
            "simulator R environment authority has wrong fields"
        )
    lock_path = value.get("lock_path")
    if (
        value.get("environment_id") != _R_ENVIRONMENT_ID
        or lock_path != _R_LOCK_PATH
        or not isinstance(lock_path, str)
        or PurePosixPath(lock_path).as_posix() != lock_path
        or not isinstance(value.get("lock_file_sha256"), str)
        or _SHA256.fullmatch(value["lock_file_sha256"]) is None
        or not isinstance(value.get("tree_sha256"), str)
        or _SHA256.fullmatch(value["tree_sha256"]) is None
        or type(value.get("tree_entry_count")) is not int
        or value["tree_entry_count"] <= 0
    ):
        raise SimulatorRuntimeAssetsError(
            "simulator R environment authority is invalid"
        )
    return dict(value)


def _validate_tree_authority(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "tree_entry_count",
        "tree_sha256",
    }:
        raise SimulatorRuntimeAssetsError(f"{name} authority has wrong fields")
    entry_count = value.get("tree_entry_count")
    digest = value.get("tree_sha256")
    if (
        type(entry_count) is not int
        or entry_count <= 0
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        raise SimulatorRuntimeAssetsError(f"{name} authority is invalid")
    return dict(value)


def _secure_lock_file_sha256(path: Path) -> str:
    return _file_sha256(path, "simulator R runtime lock")


def _r_environment_receipt(
    repository: Path,
    r_environment: Path,
    authority_value: object,
) -> dict[str, object]:
    authority = _validate_environment_authority(authority_value)
    binary_directory = _secure_directory(
        r_environment / "bin", "simulator R binary directory"
    )
    rscript = binary_directory / "Rscript"
    if not os.access(rscript, os.X_OK):
        raise SimulatorRuntimeAssetsError(
            "simulator R executable is unavailable or not executable"
        )
    lock_path = repository / _R_LOCK_PATH
    observed_lock_sha256 = _secure_lock_file_sha256(lock_path)
    if observed_lock_sha256 != authority["lock_file_sha256"]:
        raise SimulatorRuntimeAssetsError(
            "simulator R runtime lock differs from tracked authority"
        )
    try:
        lock = load_runtime_environment_lock(lock_path)
        receipt = validate_runtime_environment_lock(
            lock,
            {_R_ENVIRONMENT_ID: ("r", rscript)},
        )
    except (OSError, RuntimeEnvironmentError, TypeError, ValueError) as error:
        raise SimulatorRuntimeAssetsError(
            "simulator R installed package content differs from its runtime lock"
        ) from error
    inventory_values = receipt.get("environment_inventory_sha256s")
    if (
        receipt.get("lock_file_sha256") != observed_lock_sha256
        or not isinstance(inventory_values, tuple)
        or len(inventory_values) != 1
        or not isinstance(inventory_values[0], tuple)
        or len(inventory_values[0]) != 2
        or inventory_values[0][0] != _R_ENVIRONMENT_ID
        or not isinstance(inventory_values[0][1], str)
        or _SHA256.fullmatch(inventory_values[0][1]) is None
    ):
        raise SimulatorRuntimeAssetsError(
            "simulator R runtime validation returned a noncanonical receipt"
        )
    return {
        "schema": _R_RECEIPT_SCHEMA,
        "environment_id": _R_ENVIRONMENT_ID,
        "lock_file_sha256": observed_lock_sha256,
        "inventory_sha256": inventory_values[0][1],
    }


def _tree_file_sha256(path: Path, name: str) -> str:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise SimulatorRuntimeAssetsError(f"{name} is not a regular file")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SimulatorRuntimeAssetsError(f"{name} changed while opening")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        named_after = path.lstat()
    except SimulatorRuntimeAssetsError:
        raise
    except OSError as error:
        raise SimulatorRuntimeAssetsError(f"{name} cannot be hashed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after) or identity(before) != identity(named_after):
        raise SimulatorRuntimeAssetsError(f"{name} changed while hashing")
    return digest.hexdigest()


def _content_tree_receipt(root: Path, name: str) -> dict[str, object]:
    selected = _secure_directory(root, f"{name} tree")
    entries: list[dict[str, object]] = []
    errors: list[OSError] = []
    for current, directory_names, file_names in os.walk(
        selected, topdown=True, followlinks=False, onerror=errors.append
    ):
        directory_names.sort(key=os.fsencode)
        file_names.sort(key=os.fsencode)
        current_path = Path(current)
        for entry_name in tuple(directory_names):
            path = current_path / entry_name
            relative = path.relative_to(selected).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                target_path = Path(target)
                if target_path.is_absolute():
                    raise SimulatorRuntimeAssetsError(
                        f"{name} contains an absolute symlink"
                    )
                try:
                    (path.parent / target_path).resolve(strict=True).relative_to(
                        selected
                    )
                except (OSError, ValueError) as error:
                    raise SimulatorRuntimeAssetsError(
                        f"{name} symlink escapes its tree"
                    ) from error
                entries.append({"kind": "symlink", "path": relative, "target": target})
                directory_names.remove(entry_name)
            elif not stat.S_ISDIR(metadata.st_mode):
                raise SimulatorRuntimeAssetsError(f"{name} contains a special entry")
            else:
                entries.append({"kind": "directory", "path": relative})
        for entry_name in file_names:
            path = current_path / entry_name
            relative = path.relative_to(selected).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path)
                target_path = Path(target)
                if target_path.is_absolute():
                    raise SimulatorRuntimeAssetsError(
                        f"{name} contains an absolute symlink"
                    )
                try:
                    (path.parent / target_path).resolve(strict=True).relative_to(
                        selected
                    )
                except (OSError, ValueError) as error:
                    raise SimulatorRuntimeAssetsError(
                        f"{name} symlink escapes its tree"
                    ) from error
                entries.append({"kind": "symlink", "path": relative, "target": target})
            elif stat.S_ISREG(metadata.st_mode):
                entries.append(
                    {
                        "executable": bool(metadata.st_mode & 0o111),
                        "kind": "file",
                        "path": relative,
                        "sha256": _tree_file_sha256(path, f"{name} file"),
                        "size_bytes": metadata.st_size,
                    }
                )
            else:
                raise SimulatorRuntimeAssetsError(f"{name} contains a special entry")
    if errors:
        raise SimulatorRuntimeAssetsError(
            f"{name} tree cannot be traversed"
        ) from errors[0]
    entries.sort(key=lambda value: os.fsencode(str(value["path"])))
    payload = {"schema": "maskimpute-runtime-tree-v1", "entries": entries}
    return {
        "entry_count": len(entries),
        "sha256": canonical_sha256(payload),
    }


def _directory_content_receipt(root: Path) -> dict[str, object]:
    return _content_tree_receipt(root, "simulator R environment")


def _source_snapshot_content_receipt(root: Path) -> dict[str, object]:
    return _content_tree_receipt(root, "simulator source execution snapshot")


def _copy_snapshot_file(source: str, destination: str) -> str:
    source_path = Path(source)
    destination_path = Path(destination)
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        before = source_path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise SimulatorRuntimeAssetsError(
                "runtime snapshot source is not a regular file"
            )
        source_descriptor = os.open(
            source_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        destination_descriptor = os.open(
            destination_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            stat.S_IMODE(before.st_mode),
        )
        try:
            fcntl.ioctl(destination_descriptor, 0x40049409, source_descriptor)
        except OSError as error:
            if error.errno not in {
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            while True:
                chunk = os.read(source_descriptor, 1024 * 1024)
                if not chunk:
                    break
                offset = 0
                while offset < len(chunk):
                    offset += os.write(destination_descriptor, chunk[offset:])
        os.fsync(destination_descriptor)
        after = source_path.lstat()
    except SimulatorRuntimeAssetsError:
        raise
    except OSError as error:
        raise SimulatorRuntimeAssetsError(
            "runtime asset changed while copying its immutable snapshot"
        ) from error
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)
    identity = lambda value: (  # noqa: E731
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise SimulatorRuntimeAssetsError(
            "runtime asset changed while copying its immutable snapshot"
        )
    shutil.copystat(source_path, destination_path, follow_symlinks=False)
    return destination


def _copy_if_present(source: Path, destination: Path) -> None:
    if not os.path.lexists(source):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = source.lstat()
    if stat.S_ISDIR(metadata.st_mode):
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            copy_function=_copy_snapshot_file,
            ignore=shutil.ignore_patterns(".git"),
        )
    elif stat.S_ISREG(metadata.st_mode):
        _copy_snapshot_file(str(source), str(destination))
    else:
        raise SimulatorRuntimeAssetsError(
            "runtime source snapshot contains an unsupported entry"
        )


def _copy_source_snapshot(external_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    _copy_if_present(
        external_root / "checkouts/symsim",
        destination / "checkouts/symsim",
    )
    _copy_if_present(
        external_root / "checkouts/sparsim",
        destination / "checkouts/sparsim",
    )
    sergio_root = external_root / "checkouts/sergio"
    for relative in (
        "SERGIO/__init__.py",
        "SERGIO/gene.py",
        "SERGIO/sergio.py",
        "data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt",
        "data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt",
        "data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt",
        "data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt",
    ):
        _copy_if_present(
            sergio_root / relative,
            destination / "checkouts/sergio" / relative,
        )
    _copy_if_present(
        external_root / "data/baron-pancreas-umi/GSE84133_RAW.tar",
        destination / "data/baron-pancreas-umi/GSE84133_RAW.tar",
    )


def _make_read_only(root: Path) -> None:
    paths = sorted(
        (path for path in root.rglob("*") if not path.is_symlink()),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for path in paths:
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o555)
        elif stat.S_ISREG(metadata.st_mode):
            path.chmod(0o555 if metadata.st_mode & 0o111 else 0o444)
        else:
            raise SimulatorRuntimeAssetsError(
                "runtime snapshot contains a special filesystem entry"
            )
    root.chmod(0o555)


def _snapshot_identity(root: Path) -> str:
    entries: list[dict[str, object]] = []
    paths = [root, *sorted(root.rglob("*"), key=lambda value: os.fsencode(str(value)))]
    for path in paths:
        metadata = path.lstat()
        relative = "." if path == root else path.relative_to(root).as_posix()
        entry: dict[str, object] = {
            "path": relative,
            "state": [
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ],
        }
        if stat.S_ISLNK(metadata.st_mode):
            entry["target"] = os.readlink(path)
        elif not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
            raise SimulatorRuntimeAssetsError(
                "runtime snapshot contains a special filesystem entry"
            )
        entries.append(entry)
    return canonical_sha256(
        {"schema": "maskimpute-runtime-snapshot-identity-v1", "entries": entries}
    )


def _create_runtime_snapshot(
    external_root: Path,
    r_environment: Path,
    environment_authority: object,
    source_snapshot_authority: object,
) -> tuple[tempfile.TemporaryDirectory[str], Path, Path, str]:
    authority = _validate_environment_authority(environment_authority)
    source_authority = _validate_tree_authority(
        source_snapshot_authority, "simulator source snapshot"
    )
    owner = tempfile.TemporaryDirectory(prefix="maskimpute-runtime-snapshot-")
    snapshot_root = Path(owner.name)
    snapshot_external = snapshot_root / "external"
    snapshot_environment = snapshot_root / "r-environment"
    try:
        _copy_source_snapshot(external_root, snapshot_external)
        snapshot_sources = _source_snapshot_content_receipt(snapshot_external)
        if (
            snapshot_sources.get("sha256") != source_authority["tree_sha256"]
            or snapshot_sources.get("entry_count")
            != source_authority["tree_entry_count"]
        ):
            raise SimulatorRuntimeAssetsError(
                "copied simulator source snapshot differs from tracked authority"
            )
        shutil.copytree(
            r_environment,
            snapshot_environment,
            symlinks=True,
            copy_function=_copy_snapshot_file,
        )
        snapshot_tree = _directory_content_receipt(snapshot_environment)
        if (
            snapshot_tree.get("sha256") != authority["tree_sha256"]
            or snapshot_tree.get("entry_count") != authority["tree_entry_count"]
        ):
            raise SimulatorRuntimeAssetsError(
                "copied simulator R environment differs from tracked authority"
            )
        _make_read_only(snapshot_external)
        _make_read_only(snapshot_environment)
        identity = _snapshot_identity(snapshot_root)
    except BaseException:
        owner.cleanup()
        raise
    return owner, snapshot_external, snapshot_environment, identity


def _collect_source_receipts(
    ledger_path: Path, external_root: Path
) -> tuple[str, tuple[dict[str, object], ...]]:
    try:
        ledger = load_source_ledger(ledger_path)
        receipts = verify_fetched_sources(ledger, external_root, source_ids=_SOURCE_IDS)
    except (OSError, SourceLedgerError, TypeError, ValueError) as error:
        raise SimulatorRuntimeAssetsError(
            "publication simulator sources do not match tracked authority"
        ) from error
    ordered = tuple(
        sorted((dict(value) for value in receipts), key=lambda v: v["source_id"])
    )
    if tuple(value.get("source_id") for value in ordered) != _SOURCE_IDS:
        raise SimulatorRuntimeAssetsError(
            "publication simulator source receipt set is incomplete"
        )
    return ledger.sha256, ordered


def load_simulator_runtime_assets(
    repo: Path,
    *,
    external_root: Path,
    r_environment: Path,
    require_outside_repository: bool,
) -> SimulatorRuntimeAssets:
    """Validate explicit runtime paths against tracked path-independent authority."""

    if not isinstance(repo, Path):
        raise TypeError("repo must be a pathlib.Path")
    if type(require_outside_repository) is not bool:
        raise TypeError("require_outside_repository must be bool")
    try:
        repository = repo.resolve(strict=True)
    except OSError as error:
        raise SimulatorRuntimeAssetsError("repository is unavailable") from error
    if not repository.is_dir():
        raise SimulatorRuntimeAssetsError("repository must be a directory")
    external = _secure_directory(external_root, "simulator external asset root")
    environment = _secure_directory(r_environment, "simulator R environment")
    if require_outside_repository:
        _require_outside(external, repository, "simulator external asset root")
        _require_outside(environment, repository, "simulator R environment")
    authority, authority_sha256 = _read_authority(repository / _AUTHORITY_NAME)
    ledger_sha256, source_receipts = _collect_source_receipts(
        repository / "study/sources.json", external
    )
    if ledger_sha256 != authority["source_ledger_sha256"]:
        raise SimulatorRuntimeAssetsError(
            "simulator source ledger differs from tracked runtime authority"
        )
    environment_authority = authority["r_environment"]
    assert isinstance(environment_authority, dict)
    source_snapshot_authority = authority["source_snapshot"]
    assert isinstance(source_snapshot_authority, dict)
    environment_tree = _directory_content_receipt(environment)
    if (
        environment_tree.get("sha256") != environment_authority["tree_sha256"]
        or environment_tree.get("entry_count")
        != environment_authority["tree_entry_count"]
    ):
        raise SimulatorRuntimeAssetsError(
            "simulator R environment native bytes differ from tracked authority"
        )
    environment_receipt = _r_environment_receipt(
        repository, environment, environment_authority
    )
    semantic: dict[str, object] = {
        "schema": _RECEIPT_SCHEMA,
        "authority_sha256": authority_sha256,
        "source_ledger_sha256": ledger_sha256,
        "source_receipts": list(source_receipts),
        "source_snapshot": dict(source_snapshot_authority),
        "r_environment": environment_receipt,
    }
    semantic_sha256 = canonical_sha256(semantic)
    snapshot_owner, snapshot_external, snapshot_environment, snapshot_identity = (
        _create_runtime_snapshot(
            external,
            environment,
            environment_authority,
            source_snapshot_authority,
        )
    )
    try:
        current_ledger_sha256, current_source_receipts = _collect_source_receipts(
            repository / "study/sources.json", external
        )
        current_environment_receipt = _r_environment_receipt(
            repository, environment, environment_authority
        )
        current_environment_tree = _directory_content_receipt(environment)
        if (
            current_ledger_sha256 != ledger_sha256
            or current_source_receipts != source_receipts
            or current_environment_receipt != environment_receipt
            or current_environment_tree != environment_tree
        ):
            raise SimulatorRuntimeAssetsError(
                "simulator authority assets changed while snapshotting"
            )
    except BaseException:
        snapshot_owner.cleanup()
        raise
    value = object.__new__(SimulatorRuntimeAssets)
    object.__setattr__(value, "external_root", snapshot_external)
    object.__setattr__(value, "r_environment", snapshot_environment)
    object.__setattr__(value, "semantic_sha256", semantic_sha256)
    object.__setattr__(
        value, "_semantic_json", _canonical_bytes(semantic, pretty=False)
    )
    object.__setattr__(value, "_filesystem_identity_sha256", snapshot_identity)
    object.__setattr__(value, "_authority_external_root", external)
    object.__setattr__(value, "_authority_r_environment", environment)
    object.__setattr__(value, "_snapshot_owner", snapshot_owner)
    object.__setattr__(value, "_repository", repository)
    object.__setattr__(value, "_require_outside_repository", require_outside_repository)
    object.__setattr__(value, "_token", _TOKEN)
    return value


def revalidate_simulator_runtime_assets(
    assets: SimulatorRuntimeAssets,
) -> SimulatorRuntimeAssets:
    """Recompute all semantics and reject path or byte drift."""

    if (
        not isinstance(assets, SimulatorRuntimeAssets)
        or getattr(assets, "_token", None) is not _TOKEN
    ):
        raise SimulatorRuntimeAssetsError(
            "runtime assets must come from the authoritative loader"
        )
    current = load_simulator_runtime_assets(
        assets._repository,
        external_root=assets._authority_external_root,
        r_environment=assets._authority_r_environment,
        require_outside_repository=assets._require_outside_repository,
    )
    if (
        current.semantic_sha256 != assets.semantic_sha256
        or current._semantic_json != assets._semantic_json
    ):
        raise SimulatorRuntimeAssetsError(
            "simulator runtime asset paths or semantics drifted"
        )
    return current


def simulator_runtime_asset_values(
    assets: SimulatorRuntimeAssets,
) -> tuple[Path, Path, str]:
    """Return validated immutable adapter inputs without exposing private state."""

    if (
        not isinstance(assets, SimulatorRuntimeAssets)
        or getattr(assets, "_token", None) is not _TOKEN
    ):
        raise SimulatorRuntimeAssetsError(
            "runtime assets must come from the authoritative loader"
        )
    external = _secure_directory(assets.external_root, "simulator external asset root")
    environment = _secure_directory(assets.r_environment, "simulator R environment")
    if external != assets.external_root or environment != assets.r_environment:
        raise SimulatorRuntimeAssetsError("simulator runtime asset path drift")
    if _snapshot_identity(Path(assets._snapshot_owner.name)) != (
        assets._filesystem_identity_sha256
    ):
        raise SimulatorRuntimeAssetsError("simulator runtime snapshot identity drift")
    if (
        not isinstance(assets.semantic_sha256, str)
        or _SHA256.fullmatch(assets.semantic_sha256) is None
    ):
        raise SimulatorRuntimeAssetsError("simulator runtime receipt is invalid")
    return external, environment, assets.semantic_sha256


def revalidate_simulator_runtime_asset_identity(
    assets: SimulatorRuntimeAssets,
) -> None:
    """Reject any mutation of the private execution snapshot."""

    simulator_runtime_asset_values(assets)


def simulator_runtime_source_receipt(
    assets: SimulatorRuntimeAssets, source_id: str
) -> dict[str, object]:
    """Return one defensive path-free source receipt from the sealed contract."""

    simulator_runtime_asset_values(assets)
    receipts = assets.semantic_receipt.get("source_receipts")
    if not isinstance(receipts, list):
        raise SimulatorRuntimeAssetsError("runtime source receipts are invalid")
    matches = [
        value
        for value in receipts
        if isinstance(value, dict) and value.get("source_id") == source_id
    ]
    if len(matches) != 1:
        raise SimulatorRuntimeAssetsError(
            "runtime source receipt is missing or duplicated"
        )
    return json.loads(json.dumps(matches[0]))


__all__ = [
    "SimulatorRuntimeAssets",
    "SimulatorRuntimeAssetsError",
    "load_simulator_runtime_assets",
    "revalidate_simulator_runtime_assets",
    "revalidate_simulator_runtime_asset_identity",
    "simulator_runtime_asset_values",
    "simulator_runtime_source_receipt",
]
