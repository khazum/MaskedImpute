"""Immutable promotion of development selection evidence to schema 4."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any

from .protocol import canonical_sha256
from .revisions import (
    development_selection_stage_paths,
)
from .selection import (
    SelectionAuthorityError,
    _validate_schema_four_source_projection,
    _validate_selection_source_payload,
    attach_downstream_evidence_to_selection_result,
)


class SelectionPromotionError(RuntimeError):
    """Raised when fixed development evidence cannot be promoted immutably."""


@dataclass(frozen=True, slots=True)
class SelectionPromotionReceipt:
    """Exact byte and payload bindings for one completed promotion."""

    schema_version: int
    through_version: str
    source_selection_input_path: str
    source_selection_input_file_sha256: str
    source_selection_result_sha256: str
    downstream_manifest_path: str
    downstream_manifest_file_sha256: str
    downstream_manifest_sha256: str
    selection_complete_input_path: str
    selection_complete_input_file_sha256: str
    selection_complete_result_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelectionPromotionError(
            "promoted selection input is not canonical JSON"
        ) from error


def _fixed_repository_path(repository: Path, relative_value: str, name: str) -> Path:
    relative = PurePosixPath(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise SelectionPromotionError(f"fixed {name} path is unsafe")
    return repository.joinpath(*relative.parts)


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
    )


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
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


@contextmanager
def _pinned_parent(path: Path, label: str):
    """Yield a parent dirfd reached and rechecked without following symlinks."""

    if not path.is_absolute() or not path.name:
        raise SelectionPromotionError(f"{label} path must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    edges: list[tuple[int, str, int, tuple[int, ...]]] = []
    try:
        current = os.open(path.anchor, flags)
        descriptors.append(current)
        for component in path.parent.relative_to(path.anchor).parts:
            named = os.stat(component, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(named.st_mode):
                raise SelectionPromotionError(f"{label} parent path is not a directory")
            child = os.open(component, flags, dir_fd=current)
            opened = os.fstat(child)
            expected = _directory_identity(named)
            if _directory_identity(opened) != expected:
                os.close(child)
                raise SelectionPromotionError(
                    f"{label} parent path changed while being opened"
                )
            descriptors.append(child)
            edges.append((current, component, child, expected))
            current = child
        yield current
        for parent, component, child, expected in edges:
            try:
                named_after = os.stat(
                    component,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
                opened_after = os.fstat(child)
            except OSError as error:
                raise SelectionPromotionError(
                    f"{label} parent path changed during access"
                ) from error
            if (
                _directory_identity(named_after) != expected
                or _directory_identity(opened_after) != expected
            ):
                raise SelectionPromotionError(
                    f"{label} parent path changed during access"
                )
    except SelectionPromotionError:
        raise
    except OSError as error:
        raise SelectionPromotionError(
            f"cannot open {label} parent path: {error}"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _read_unique_regular_at(parent: int, name: str, label: str) -> bytes:
    descriptor = -1
    try:
        named_before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(named_before.st_mode) or named_before.st_nlink != 1:
            raise SelectionPromotionError(f"{label} is not a unique regular file")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or opened_before.st_mode & 0o002
            or _file_identity(opened_before) != _file_identity(named_before)
            or opened_before.st_size > 128 * 1024 * 1024
        ):
            raise SelectionPromotionError(f"{label} is not a unique regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _file_identity(opened_before) != _file_identity(
            opened_after
        ) or _file_identity(opened_before) != _file_identity(named_after):
            raise SelectionPromotionError(f"{label} changed while being read")
        raw = b"".join(chunks)
        if len(raw) != opened_before.st_size:
            raise SelectionPromotionError(f"{label} changed while being read")
        return raw
    except SelectionPromotionError:
        raise
    except OSError as error:
        raise SelectionPromotionError(f"cannot read {label}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectionPromotionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SelectionPromotionError(f"nonfinite JSON constant {value}")


def _secure_canonical_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        with _pinned_parent(path, label) as parent:
            raw = _read_unique_regular_at(parent, path.name, label)
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except SelectionPromotionError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise SelectionPromotionError(f"cannot parse {label}: {error}") from error
    if type(payload) is not dict or raw != _canonical_json_bytes(payload):
        raise SelectionPromotionError(f"{label} is not canonical JSON")
    return payload, hashlib.sha256(raw).hexdigest()


def _immutable_publish(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    with _pinned_parent(path, "selection-complete input") as parent:
        descriptor = -1
        temporary_exists = False
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent,
            )
            temporary_exists = True
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError:
                existing = _read_unique_regular_at(
                    parent,
                    path.name,
                    "existing selection-complete input",
                )
                if existing != data:
                    raise SelectionPromotionError(
                        "existing selection-complete input conflicts with promotion"
                    )
            except OSError as error:
                raise SelectionPromotionError(
                    "selection-complete input could not be published"
                ) from error
            os.unlink(temporary_name, dir_fd=parent)
            temporary_exists = False
            published = _read_unique_regular_at(
                parent,
                path.name,
                "published selection-complete input",
            )
            if published != data:
                raise SelectionPromotionError(
                    "published selection-complete input differs"
                )
            os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_exists:
                try:
                    os.unlink(temporary_name, dir_fd=parent)
                except FileNotFoundError:
                    pass
    return digest


def _read_source(
    repository: Path,
    through_version: str | None,
) -> tuple[dict[str, Any], str]:
    paths = development_selection_stage_paths(through_version)
    source_path = _fixed_repository_path(
        repository,
        paths.source_selection_input,
        "source selection input",
    )
    try:
        source, file_sha256 = _secure_canonical_json(
            source_path,
            "source development selection input",
        )
        observed_stage = _validate_selection_source_payload(source)
    except (SelectionPromotionError, SelectionAuthorityError) as error:
        raise SelectionPromotionError(
            f"source selection input is invalid: {error}"
        ) from error
    if observed_stage != through_version:
        raise SelectionPromotionError("source selection input stage differs")
    return source, file_sha256


def _read_embedded_comparator_receipt(
    repository: Path,
    source: dict[str, Any],
) -> bytes:
    """Reread and byte-compare the complete embedded direct receipt."""

    selection = source.get("comparator_selection")
    if not isinstance(selection, dict):
        raise SelectionPromotionError("source comparator selection is invalid")
    path_value = selection.get("path")
    receipt = selection.get("receipt")
    if not isinstance(path_value, str) or type(receipt) is not dict:
        raise SelectionPromotionError("source comparator selection is invalid")
    path = _fixed_repository_path(
        repository,
        path_value,
        "comparator selection receipt",
    )
    try:
        with _pinned_parent(path, "comparator selection receipt") as parent:
            raw = _read_unique_regular_at(
                parent,
                path.name,
                "comparator selection receipt",
            )
    except SelectionPromotionError as error:
        raise SelectionPromotionError(
            "comparator selection receipt is invalid"
        ) from error
    if raw != _canonical_json_bytes(receipt):
        raise SelectionPromotionError("comparator selection receipt differs")
    return raw


def _validate_promoted_payload(
    repository: Path,
    through_version: str | None,
    source: dict[str, Any],
    source_file_sha256: str,
    promoted: object,
) -> tuple[dict[str, Any], str, str]:
    paths = development_selection_stage_paths(through_version)
    if type(promoted) is not dict or promoted.get("schema_version") != 4:
        raise SelectionPromotionError("attachment did not produce schema 4")
    expected_revisions = source.get("revision_versions", [])
    if promoted.get("revision_versions") != expected_revisions:
        raise SelectionPromotionError("promoted selection revision chain differs")
    binding = promoted.get("downstream_evidence")
    if not isinstance(binding, dict):
        raise SelectionPromotionError("promoted downstream binding is invalid")
    try:
        _validate_schema_four_source_projection(repository, promoted, binding)
    except SelectionAuthorityError as error:
        raise SelectionPromotionError(str(error)) from error
    if (
        binding.get("path") != paths.downstream_directory
        or binding.get("source_selection_input_path") != paths.source_selection_input
        or binding.get("source_selection_input_file_sha256") != source_file_sha256
        or binding.get("source_selection_result_sha256") != source.get("result_sha256")
    ):
        raise SelectionPromotionError("promoted source or downstream path differs")
    promoted_result_sha = promoted.get("result_sha256")
    if not isinstance(promoted_result_sha, str):
        raise SelectionPromotionError("promoted selection checksum is invalid")
    unsigned = {key: value for key, value in promoted.items() if key != "result_sha256"}
    if canonical_sha256(unsigned) != promoted_result_sha:
        raise SelectionPromotionError("promoted selection checksum differs")
    manifest_path_value = str(
        PurePosixPath(paths.downstream_directory) / "downstream_manifest.json"
    )
    manifest_path = _fixed_repository_path(
        repository,
        manifest_path_value,
        "downstream manifest",
    )
    try:
        manifest, manifest_file_sha = _secure_canonical_json(
            manifest_path,
            "downstream manifest",
        )
    except SelectionPromotionError as error:
        raise SelectionPromotionError("downstream manifest is invalid") from error
    manifest_payload_sha = manifest.get("manifest_sha256")
    if (
        not isinstance(manifest_payload_sha, str)
        or binding.get("manifest_file_sha256") != manifest_file_sha
        or binding.get("manifest_sha256") != manifest_payload_sha
    ):
        raise SelectionPromotionError("promoted downstream manifest binding differs")
    return promoted, manifest_file_sha, manifest_payload_sha


def promote_development_selection_input(
    repository: Path,
    through_version: str | None,
) -> SelectionPromotionReceipt:
    """Validate and immutably publish the fixed schema-4 input for one stage."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if through_version not in {None, "v28", "v29"}:
        raise ValueError("through_version must be null, v28, or v29")
    root = repository.absolute()
    if not root.is_dir() or root.is_symlink():
        raise SelectionPromotionError(
            "selection promotion repository is absent or unsafe"
        )
    paths = development_selection_stage_paths(through_version)
    source, source_file_sha = _read_source(root, through_version)
    comparator_receipt_bytes = _read_embedded_comparator_receipt(root, source)
    try:
        attached = attach_downstream_evidence_to_selection_result(
            source,
            root,
            paths.downstream_directory,
        )
    except SelectionAuthorityError as error:
        raise SelectionPromotionError(str(error)) from error
    promoted, manifest_file_sha, manifest_payload_sha = _validate_promoted_payload(
        root,
        through_version,
        source,
        source_file_sha,
        attached,
    )
    encoded = _canonical_json_bytes(promoted)
    destination = _fixed_repository_path(
        root,
        paths.selection_complete_input,
        "selection-complete input",
    )
    published_file_sha = _immutable_publish(destination, encoded)
    try:
        published, reread_file_sha = _secure_canonical_json(
            destination,
            "selection-complete input",
        )
    except SelectionPromotionError as error:  # pragma: no cover - post-publish guard
        raise SelectionPromotionError(
            "published selection-complete input failed revalidation"
        ) from error
    if published != promoted or reread_file_sha != published_file_sha:
        raise SelectionPromotionError(
            "published selection-complete input differs after publication"
        )
    if (
        published.get("comparator_selection") != source["comparator_selection"]
        or _read_embedded_comparator_receipt(root, published)
        != comparator_receipt_bytes
    ):
        raise SelectionPromotionError(
            "published comparator selection differs after publication"
        )
    result_sha = promoted["result_sha256"]
    assert isinstance(result_sha, str)
    source_result_sha = source["result_sha256"]
    assert isinstance(source_result_sha, str)
    return SelectionPromotionReceipt(
        schema_version=1,
        through_version="base" if through_version is None else through_version,
        source_selection_input_path=paths.source_selection_input,
        source_selection_input_file_sha256=source_file_sha,
        source_selection_result_sha256=source_result_sha,
        downstream_manifest_path=str(
            PurePosixPath(paths.downstream_directory) / "downstream_manifest.json"
        ),
        downstream_manifest_file_sha256=manifest_file_sha,
        downstream_manifest_sha256=manifest_payload_sha,
        selection_complete_input_path=paths.selection_complete_input,
        selection_complete_input_file_sha256=published_file_sha,
        selection_complete_result_sha256=result_sha,
    )


def promote_latest_development_selection_input(
    repository: Path,
) -> SelectionPromotionReceipt:
    """Promote the latest present exact source stage without fallback."""

    from .downstream_evidence import (
        DownstreamEvidenceError,
        development_downstream_revision_version,
    )

    try:
        through_version = development_downstream_revision_version(repository)
    except DownstreamEvidenceError as error:
        raise SelectionPromotionError(str(error)) from error
    return promote_development_selection_input(repository, through_version)


__all__ = [
    "SelectionPromotionError",
    "SelectionPromotionReceipt",
    "promote_development_selection_input",
    "promote_latest_development_selection_input",
]
