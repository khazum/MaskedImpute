"""Immutable promotion of development selection evidence to schema 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from .protocol import canonical_sha256
from .revisions import (
    RevisionAuthorityError,
    _read_canonical_json,
    _read_stable_bytes,
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


def _safe_repository_path(repository: Path, relative_value: str, name: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SelectionPromotionError(f"fixed {name} path is unsafe")
    path = repository.joinpath(*relative.parts)
    current = repository
    if current.is_symlink():
        raise SelectionPromotionError("selection promotion repository is a symlink")
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise SelectionPromotionError(f"fixed {name} path contains a symlink")
    return path


def _immutable_publish(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
            try:
                existing = _read_stable_bytes(
                    path,
                    "existing selection-complete input",
                )
            except RevisionAuthorityError as error:
                raise SelectionPromotionError(
                    "existing selection-complete input is unsafe"
                ) from error
            if existing != data:
                raise SelectionPromotionError(
                    "existing selection-complete input conflicts with promotion"
                )
        except OSError as error:
            raise SelectionPromotionError(
                "selection-complete input could not be published"
            ) from error
        try:
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            raise SelectionPromotionError(
                "selection-complete input directory could not be synchronized"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _read_source(
    repository: Path,
    through_version: str | None,
) -> tuple[dict[str, Any], str]:
    paths = development_selection_stage_paths(through_version)
    source_path = _safe_repository_path(
        repository,
        paths.source_selection_input,
        "source selection input",
    )
    try:
        source, file_sha256 = _read_canonical_json(
            source_path,
            "source development selection input",
            indented=False,
        )
        observed_stage = _validate_selection_source_payload(source)
    except (RevisionAuthorityError, SelectionAuthorityError) as error:
        raise SelectionPromotionError(f"source selection input is invalid: {error}") from error
    if observed_stage != through_version:
        raise SelectionPromotionError("source selection input stage differs")
    return source, file_sha256


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
        or binding.get("source_selection_input_path")
        != paths.source_selection_input
        or binding.get("source_selection_input_file_sha256") != source_file_sha256
        or binding.get("source_selection_result_sha256")
        != source.get("result_sha256")
    ):
        raise SelectionPromotionError("promoted source or downstream path differs")
    promoted_result_sha = promoted.get("result_sha256")
    if not isinstance(promoted_result_sha, str):
        raise SelectionPromotionError("promoted selection checksum is invalid")
    unsigned = {
        key: value for key, value in promoted.items() if key != "result_sha256"
    }
    if canonical_sha256(unsigned) != promoted_result_sha:
        raise SelectionPromotionError("promoted selection checksum differs")
    manifest_path_value = str(
        PurePosixPath(paths.downstream_directory) / "downstream_manifest.json"
    )
    manifest_path = _safe_repository_path(
        repository,
        manifest_path_value,
        "downstream manifest",
    )
    try:
        manifest, manifest_file_sha = _read_canonical_json(
            manifest_path,
            "downstream manifest",
            indented=False,
        )
    except RevisionAuthorityError as error:
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
    destination = _safe_repository_path(
        root,
        paths.selection_complete_input,
        "selection-complete input",
    )
    published_file_sha = _immutable_publish(destination, encoded)
    try:
        published, reread_file_sha = _read_canonical_json(
            destination,
            "selection-complete input",
            indented=False,
        )
    except RevisionAuthorityError as error:  # pragma: no cover - post-publish guard
        raise SelectionPromotionError(
            "published selection-complete input failed revalidation"
        ) from error
    if published != promoted or reread_file_sha != published_file_sha:
        raise SelectionPromotionError(
            "published selection-complete input differs after publication"
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
