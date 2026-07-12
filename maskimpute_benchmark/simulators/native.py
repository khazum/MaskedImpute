"""Canonical, race-resistant sealing of pristine simulator outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from ..protocol import canonical_sha256
from .base import SimulationContractError


_CHUNK_SIZE = 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NATIVE_MANIFEST_TOKEN = object()


@dataclass(frozen=True, slots=True)
class NativeFile:
    """One logical native output bound to its exact regular-file bytes."""

    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class NativeManifest:
    """Immutable canonical manifest; metadata is retained as canonical JSON."""

    schema_version: int
    files: tuple[NativeFile, ...]
    _metadata_json: str
    manifest_sha256: str
    _sealed_files: tuple[_FileSnapshot, ...] = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self._metadata_json)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "files": [entry.as_dict() for entry in self.files],
            "metadata": self.metadata,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    logical_path: str
    physical_path: Path
    identity: tuple[int, int]
    state: tuple[int, int, int, int, int, int]
    size_bytes: int
    sha256: str


def _validate_json_value(value: object, name: str) -> None:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SimulationContractError(
                f"{name} metadata object keys must all be strings"
            )
        for nested in value.values():
            _validate_json_value(nested, name)
        return
    if isinstance(value, list):
        for nested in value:
            _validate_json_value(nested, name)
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise SimulationContractError(f"{name} metadata must be finite canonical JSON")


def _snapshot_metadata(metadata: Mapping[str, object]) -> str:
    if not isinstance(metadata, Mapping):
        raise SimulationContractError("native metadata must be a JSON object")
    _validate_json_value(metadata, "native")
    try:
        encoded = json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SimulationContractError(
            "native metadata must be finite canonical JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise SimulationContractError("native metadata must be a JSON object")
    return encoded


def _logical_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        qualifier = "POSIX " if isinstance(value, str) and "\\" in value else ""
        raise SimulationContractError(
            f"native logical path must be a nonempty relative {qualifier}path"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SimulationContractError(
            "native logical path must be a canonical relative POSIX path"
        )
    return value


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    components = [absolute, *absolute.parents]
    for component in components:
        try:
            metadata = component.lstat()
        except OSError as error:
            raise SimulationContractError(
                f"native file cannot be inspected: {path}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SimulationContractError(
                f"native file path must not contain a symlink: {path}"
            )


def _state(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_uid,
    )


def _hash_regular_file(logical_path: str, path: Path) -> _FileSnapshot:
    _reject_symlink_components(path)
    try:
        before_path = path.lstat()
    except OSError as error:
        raise SimulationContractError(f"native file does not exist: {path}") from error
    if not stat.S_ISREG(before_path.st_mode):
        raise SimulationContractError(f"native output must be a regular file: {path}")
    if before_path.st_nlink != 1:
        raise SimulationContractError(
            f"native output must not be a hard link or duplicate inode: {path}"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SimulationContractError(
            f"native file cannot be opened safely: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SimulationContractError(
                f"native output must be a unique regular file: {path}"
            )
        if (before.st_dev, before.st_ino) != (
            before_path.st_dev,
            before_path.st_ino,
        ):
            raise SimulationContractError(f"native file changed while opening: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise SimulationContractError(
            f"native file changed while hashing: {path}"
        ) from error
    finally:
        os.close(descriptor)

    try:
        after_path = path.lstat()
    except OSError as error:
        raise SimulationContractError(
            f"native file changed while hashing: {path}"
        ) from error
    identity = (before.st_dev, before.st_ino)
    if (
        identity != (after.st_dev, after.st_ino)
        or identity != (after_path.st_dev, after_path.st_ino)
        or _state(before) != _state(after)
        or _state(before) != _state(after_path)
    ):
        raise SimulationContractError(f"native file changed while hashing: {path}")
    return _FileSnapshot(
        logical_path=logical_path,
        physical_path=path.absolute(),
        identity=identity,
        state=_state(before),
        size_bytes=before.st_size,
        sha256=digest.hexdigest(),
    )


def _manifest_payload(
    files: tuple[NativeFile, ...], metadata_json: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "files": [entry.as_dict() for entry in files],
        "metadata": json.loads(metadata_json),
    }


def validate_native_manifest(manifest: NativeManifest) -> None:
    """Validate an in-memory manifest before trusting its embedded commitment."""

    if not isinstance(manifest, NativeManifest):
        raise SimulationContractError("native manifest must be a NativeManifest")
    if manifest._token is not _NATIVE_MANIFEST_TOKEN:
        raise SimulationContractError("native manifest was not produced by sealing")
    if not isinstance(manifest._sealed_files, tuple) or len(
        manifest._sealed_files
    ) != len(manifest.files):
        raise SimulationContractError("native manifest sealed-file state is invalid")
    if type(manifest.schema_version) is not int or manifest.schema_version != 1:
        raise SimulationContractError("native manifest schema version is invalid")
    if not isinstance(manifest.files, tuple) or not manifest.files:
        raise SimulationContractError("native manifest files must be nonempty")
    paths: list[str] = []
    for entry in manifest.files:
        if not isinstance(entry, NativeFile):
            raise SimulationContractError("native manifest file entry is invalid")
        path = _logical_path(entry.path)
        if type(entry.size_bytes) is not int or entry.size_bytes < 0:
            raise SimulationContractError("native manifest file size is invalid")
        if not isinstance(entry.sha256, str) or not _SHA256.fullmatch(entry.sha256):
            raise SimulationContractError("native manifest file hash is invalid")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SimulationContractError("native manifest paths must be sorted and unique")
    if not isinstance(manifest._metadata_json, str):
        raise SimulationContractError("native manifest metadata is invalid")
    try:
        metadata = json.loads(manifest._metadata_json)
        canonical_metadata = _snapshot_metadata(metadata)
    except (SimulationContractError, json.JSONDecodeError) as error:
        raise SimulationContractError("native manifest metadata is invalid") from error
    if canonical_metadata != manifest._metadata_json:
        raise SimulationContractError("native manifest metadata is not canonical")
    expected = canonical_sha256(
        _manifest_payload(manifest.files, manifest._metadata_json)
    )
    if (
        not isinstance(manifest.manifest_sha256, str)
        or not _SHA256.fullmatch(manifest.manifest_sha256)
        or manifest.manifest_sha256 != expected
    ):
        raise SimulationContractError("native manifest hash is invalid")


def revalidate_native_outputs(manifest: NativeManifest) -> None:
    """Prove native bytes still match the seal after adapter translation."""

    validate_native_manifest(manifest)
    for entry, sealed in zip(manifest.files, manifest._sealed_files, strict=True):
        if (
            not isinstance(sealed, _FileSnapshot)
            or sealed.logical_path != entry.path
            or sealed.size_bytes != entry.size_bytes
            or sealed.sha256 != entry.sha256
        ):
            raise SimulationContractError(
                "native manifest sealed-file state is invalid"
            )
        current = _hash_regular_file(sealed.logical_path, sealed.physical_path)
        if (
            current.identity != sealed.identity
            or current.state != sealed.state
            or current.size_bytes != sealed.size_bytes
            or current.sha256 != sealed.sha256
        ):
            raise SimulationContractError(
                f"native file changed after sealing: {sealed.physical_path}"
            )


def seal_native_outputs(
    files: Mapping[str, Path], metadata: Mapping[str, object]
) -> NativeManifest:
    """Hash exact native bytes twice and return a canonical immutable manifest."""

    if not isinstance(files, Mapping) or not files:
        raise SimulationContractError("native files must be a nonempty path mapping")
    metadata_json = _snapshot_metadata(metadata)
    prepared: list[tuple[str, Path]] = []
    for logical_value, physical_value in files.items():
        logical = _logical_path(logical_value)
        if not isinstance(physical_value, Path):
            raise SimulationContractError(
                "native physical paths must be pathlib.Path values"
            )
        prepared.append((logical, physical_value))
    prepared.sort(key=lambda item: item[0])
    if len({logical for logical, _ in prepared}) != len(prepared):
        raise SimulationContractError("native logical paths must be unique")

    first_pass = tuple(_hash_regular_file(logical, path) for logical, path in prepared)
    identities = [snapshot.identity for snapshot in first_pass]
    if len(set(identities)) != len(identities):
        raise SimulationContractError("native files contain a duplicate inode")

    second_pass = tuple(
        _hash_regular_file(snapshot.logical_path, snapshot.physical_path)
        for snapshot in first_pass
    )
    for first, second in zip(first_pass, second_pass, strict=True):
        if (
            first.identity != second.identity
            or first.state != second.state
            or first.sha256 != second.sha256
        ):
            raise SimulationContractError(
                f"native file changed while hashing: {first.physical_path}"
            )

    entries = tuple(
        NativeFile(
            path=snapshot.logical_path,
            size_bytes=snapshot.size_bytes,
            sha256=snapshot.sha256,
        )
        for snapshot in first_pass
    )
    manifest_hash = canonical_sha256(_manifest_payload(entries, metadata_json))
    return NativeManifest(
        schema_version=1,
        files=entries,
        _metadata_json=metadata_json,
        manifest_sha256=manifest_hash,
        _sealed_files=first_pass,
        _token=_NATIVE_MANIFEST_TOKEN,
    )


__all__ = [
    "NativeFile",
    "NativeManifest",
    "seal_native_outputs",
    "revalidate_native_outputs",
    "validate_native_manifest",
]
