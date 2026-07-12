"""Validated immutable source pins and credential-safe external fetching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SPDX_LIKE = re.compile(
    r"^(?:LicenseRef-[A-Za-z0-9][A-Za-z0-9.-]*|"
    r"[A-Za-z0-9][A-Za-z0-9.+-]*)(?: WITH [A-Za-z0-9][A-Za-z0-9.+-]*)?$"
)
_DOI = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_DATA_REVISION = re.compile(
    r"^[A-Za-z][A-Za-z0-9._-]*\d+:"
    r"(?:\d{4}-\d{2}-\d{2}|v?\d+(?:\.\d+)*)$"
)
_ROLES = {"mechanism", "semisynthetic_source", "orthogonal_validation"}
_SOURCE_TYPES = {"git", "data"}
_ELIGIBILITY = {"eligible", "pending", "ineligible"}
_SOURCE_KEYS = {
    "id",
    "role",
    "mechanism",
    "source_type",
    "url",
    "revision",
    "license",
    "license_url",
    "citation_doi",
    "expected_checksum",
    "eligibility",
    "endpoints",
    "artifacts",
    "ineligibility_reason",
}


class SourceLedgerError(ValueError):
    """Raised when a source pin or fetched source violates the ledger contract."""


@dataclass(frozen=True, slots=True)
class Checksum:
    algorithm: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}


@dataclass(frozen=True, slots=True)
class DataArtifact:
    name: str
    url: str
    expected_checksum: Checksum


@dataclass(frozen=True, slots=True)
class SourcePin:
    id: str
    role: str
    mechanism: str | None
    source_type: str
    url: str
    revision: str
    license: str
    license_url: str
    citation_doi: str
    expected_checksum: Checksum | None
    eligibility: str
    endpoints: tuple[str, ...]
    artifacts: tuple[DataArtifact, ...]
    ineligibility_reason: str | None


@dataclass(frozen=True, slots=True)
class SourceLedger:
    schema_version: int
    sources: tuple[SourcePin, ...]
    sha256: str


def _reject_json_constant(value: str) -> None:
    raise SourceLedgerError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceLedgerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SourceLedgerError(f"{name} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, object], required: set[str], optional: set[str], name: str
) -> None:
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing:
        raise SourceLedgerError(f"{name} missing fields: {sorted(missing)!r}")
    if extra:
        raise SourceLedgerError(f"{name} has unknown fields: {sorted(extra)!r}")


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SourceLedgerError(f"{name} must be a nonempty trimmed string")
    return value


def _portable_path_key(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _reject_portable_collisions(values: Sequence[str], name: str) -> None:
    keys = [_portable_path_key(value) for value in values]
    if len(set(keys)) != len(keys):
        raise SourceLedgerError(f"{name} contains a nonportable path collision")


def _validate_url(value: object, name: str, *, allow_local_urls: bool) -> str:
    url = _nonempty_string(value, name)
    parsed = urlsplit(url)
    if parsed.username is not None or parsed.password is not None:
        raise SourceLedgerError(f"{name} must not embed credentials")
    if parsed.fragment:
        raise SourceLedgerError(f"{name} must not contain a URL fragment")
    if parsed.scheme == "file" and allow_local_urls:
        if parsed.netloc not in {"", "localhost"} or not parsed.path.startswith("/"):
            raise SourceLedgerError(f"{name} local file URL must be absolute")
        return url
    if parsed.scheme != "https" or not parsed.netloc:
        raise SourceLedgerError(f"{name} must be an absolute HTTPS URL")
    return url


def _validate_license(value: object, name: str) -> str:
    license_id = _nonempty_string(value, name)
    if not _SPDX_LIKE.fullmatch(license_id) or license_id.casefold() in {
        "tbd",
        "unknown",
        "none",
    }:
        raise SourceLedgerError(f"{name} must be an SPDX-like license identifier")
    return license_id


def _validate_doi(value: object, name: str) -> str:
    doi = _nonempty_string(value, name)
    if not _DOI.fullmatch(doi):
        raise SourceLedgerError(f"{name} must be a bare citation DOI")
    return doi


def _validate_checksum(
    value: object, name: str, *, required_algorithm: str
) -> Checksum:
    checksum = _mapping(value, name)
    _exact_keys(checksum, {"algorithm", "value"}, set(), name)
    algorithm = _nonempty_string(checksum.get("algorithm"), f"{name}.algorithm")
    if algorithm != required_algorithm:
        raise SourceLedgerError(
            f"{name}.algorithm must be {required_algorithm}, not {algorithm}"
        )
    digest = _nonempty_string(checksum.get("value"), f"{name}.value")
    pattern = _HEX40 if algorithm == "git-tree-sha1" else _HEX64
    if not pattern.fullmatch(digest):
        length = 40 if algorithm == "git-tree-sha1" else 64
        raise SourceLedgerError(
            f"{name}.value must be a lowercase {length}-character checksum"
        )
    return Checksum(algorithm=algorithm, value=digest)


def _validate_artifact(
    value: object, source_name: str, *, allow_local_urls: bool
) -> DataArtifact:
    artifact = _mapping(value, source_name)
    _exact_keys(
        artifact, {"name", "url", "expected_checksum"}, set(), source_name
    )
    name = _nonempty_string(artifact.get("name"), f"{source_name}.name")
    if Path(name).name != name or name in {".", ".."}:
        raise SourceLedgerError(f"{source_name}.name must be a safe basename")
    return DataArtifact(
        name=name,
        url=_validate_url(
            artifact.get("url"),
            f"{source_name}.url",
            allow_local_urls=allow_local_urls,
        ),
        expected_checksum=_validate_checksum(
            artifact.get("expected_checksum"),
            f"{source_name}.expected_checksum",
            required_algorithm="sha256",
        ),
    )


def _validate_source(
    value: object, index: int, *, allow_local_urls: bool
) -> SourcePin:
    name = f"sources[{index}]"
    source = _mapping(value, name)
    required = _SOURCE_KEYS - {"artifacts", "ineligibility_reason"}
    _exact_keys(source, required, {"artifacts", "ineligibility_reason"}, name)

    source_id = _nonempty_string(source.get("id"), f"{name}.id")
    if not _SAFE_ID.fullmatch(source_id):
        raise SourceLedgerError(f"{name}.id must be a safe lowercase source identifier")
    role = _nonempty_string(source.get("role"), f"{name}.role")
    if role not in _ROLES:
        raise SourceLedgerError(f"{name}.role must be one of {sorted(_ROLES)!r}")
    source_type = _nonempty_string(
        source.get("source_type"), f"{name}.source_type"
    )
    if source_type not in _SOURCE_TYPES:
        raise SourceLedgerError(
            f"{name}.source_type must be one of {sorted(_SOURCE_TYPES)!r}"
        )
    mechanism_value = source.get("mechanism")
    if mechanism_value is not None and (
        not isinstance(mechanism_value, str)
        or not _SAFE_ID.fullmatch(mechanism_value)
    ):
        raise SourceLedgerError(f"{name}.mechanism must be null or a safe identifier")
    if role in {"mechanism", "semisynthetic_source"} and mechanism_value is None:
        raise SourceLedgerError(f"{name}.mechanism is required for role {role}")
    if role == "orthogonal_validation" and mechanism_value is not None:
        raise SourceLedgerError(f"{name}.mechanism must be null for orthogonal data")

    eligibility = _nonempty_string(
        source.get("eligibility"), f"{name}.eligibility"
    )
    if eligibility not in _ELIGIBILITY:
        raise SourceLedgerError(
            f"{name}.eligibility must be one of {sorted(_ELIGIBILITY)!r}"
        )
    reason_value = source.get("ineligibility_reason")
    if eligibility == "eligible":
        if reason_value is not None:
            raise SourceLedgerError(
                f"{name}.ineligibility_reason must be absent for eligible sources"
            )
        reason = None
    else:
        reason = _nonempty_string(reason_value, f"{name}.ineligibility_reason")

    endpoints_value = source.get("endpoints")
    if not isinstance(endpoints_value, list) or not endpoints_value:
        raise SourceLedgerError(f"{name}.endpoints must be a nonempty list")
    endpoints: list[str] = []
    for endpoint in endpoints_value:
        endpoint_name = _nonempty_string(endpoint, f"{name}.endpoints[]")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", endpoint_name):
            raise SourceLedgerError(
                f"{name}.endpoints entries must be lowercase identifiers"
            )
        endpoints.append(endpoint_name)
    if len(set(endpoints)) != len(endpoints):
        raise SourceLedgerError(f"{name}.endpoints must be unique")

    revision = _nonempty_string(source.get("revision"), f"{name}.revision")
    if source_type == "git":
        if not _HEX40.fullmatch(revision):
            qualifier = "lowercase " if len(revision) == 40 else ""
            raise SourceLedgerError(
                f"{name}.revision must be an exact {qualifier}40-character Git commit"
            )
        expected = _validate_checksum(
            source.get("expected_checksum"),
            f"{name}.expected_checksum",
            required_algorithm="git-tree-sha1",
        )
        if "artifacts" in source:
            raise SourceLedgerError(f"{name}.artifacts is only valid for data sources")
        artifacts: tuple[DataArtifact, ...] = ()
    else:
        if not _DATA_REVISION.fullmatch(revision):
            raise SourceLedgerError(
                f"{name}.revision must be an exact accession:version value"
            )
        if source.get("expected_checksum") is not None:
            raise SourceLedgerError(
                f"{name}.expected_checksum must be null; pin each data artifact"
            )
        expected = None
        artifacts_value = source.get("artifacts")
        if not isinstance(artifacts_value, list) or not artifacts_value:
            raise SourceLedgerError(f"{name}.artifacts must be a nonempty list")
        artifacts = tuple(
            _validate_artifact(
                artifact,
                f"{name}.artifacts[{artifact_index}]",
                allow_local_urls=allow_local_urls,
            )
            for artifact_index, artifact in enumerate(artifacts_value)
        )
        artifact_names = [artifact.name for artifact in artifacts]
        if len(set(artifact_names)) != len(artifact_names):
            raise SourceLedgerError(f"{name} has duplicate artifact names")
        _reject_portable_collisions(artifact_names, f"{name}.artifacts")

    return SourcePin(
        id=source_id,
        role=role,
        mechanism=mechanism_value,
        source_type=source_type,
        url=_validate_url(
            source.get("url"), f"{name}.url", allow_local_urls=allow_local_urls
        ),
        revision=revision,
        license=_validate_license(source.get("license"), f"{name}.license"),
        license_url=_validate_url(
            source.get("license_url"),
            f"{name}.license_url",
            allow_local_urls=False,
        ),
        citation_doi=_validate_doi(
            source.get("citation_doi"), f"{name}.citation_doi"
        ),
        expected_checksum=expected,
        eligibility=eligibility,
        endpoints=tuple(endpoints),
        artifacts=artifacts,
        ineligibility_reason=reason,
    )


def load_source_ledger(
    path: Path, *, allow_local_urls: bool = False
) -> SourceLedger:
    """Load a strict version-1 source ledger.

    ``allow_local_urls`` exists solely for isolated unit fixtures; publication
    commands intentionally leave it disabled.
    """

    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(
            raw,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except SourceLedgerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceLedgerError(f"cannot read source ledger: {exc}") from exc
    ledger = _mapping(value, "ledger")
    _exact_keys(ledger, {"schema_version", "sources"}, set(), "ledger")
    if type(ledger.get("schema_version")) is not int or ledger["schema_version"] != 1:
        raise SourceLedgerError("schema_version must be 1")
    values = ledger.get("sources")
    if not isinstance(values, list) or not values:
        raise SourceLedgerError("sources must be a nonempty list")
    sources = tuple(
        _validate_source(source, index, allow_local_urls=allow_local_urls)
        for index, source in enumerate(values)
    )
    identifiers = [source.id for source in sources]
    if len(set(identifiers)) != len(identifiers):
        raise SourceLedgerError("duplicate source id in ledger")
    digest = hashlib.sha256(_canonical_json_bytes(ledger)).hexdigest()
    return SourceLedger(schema_version=1, sources=sources, sha256=digest)


def _source_pin_payload(source: SourcePin) -> dict[str, object]:
    if type(source) is not SourcePin:
        raise SourceLedgerError("ledger sources must be SourcePin values")
    expected = source.expected_checksum
    if expected is not None and type(expected) is not Checksum:
        raise SourceLedgerError("source checksum must be a Checksum value")
    if not isinstance(source.artifacts, tuple) or not all(
        type(artifact) is DataArtifact for artifact in source.artifacts
    ):
        raise SourceLedgerError("source artifacts must be DataArtifact values")
    payload: dict[str, object] = {
        "id": source.id,
        "role": source.role,
        "mechanism": source.mechanism,
        "source_type": source.source_type,
        "url": source.url,
        "revision": source.revision,
        "license": source.license,
        "license_url": source.license_url,
        "citation_doi": source.citation_doi,
        "expected_checksum": expected.as_dict() if expected is not None else None,
        "eligibility": source.eligibility,
        "endpoints": list(source.endpoints),
    }
    if source.artifacts:
        payload["artifacts"] = [
            {
                "name": artifact.name,
                "url": artifact.url,
                "expected_checksum": artifact.expected_checksum.as_dict(),
            }
            for artifact in source.artifacts
        ]
    if source.ineligibility_reason is not None:
        payload["ineligibility_reason"] = source.ineligibility_reason
    return payload


def _revalidate_ledger_object(
    ledger: SourceLedger, *, allow_local_urls: bool
) -> SourceLedger:
    if type(ledger) is not SourceLedger:
        raise SourceLedgerError("ledger must be a SourceLedger")
    if type(ledger.schema_version) is not int or ledger.schema_version != 1:
        raise SourceLedgerError("ledger schema_version must be 1")
    if not isinstance(ledger.sources, tuple) or not ledger.sources:
        raise SourceLedgerError("ledger sources must be a nonempty tuple")
    try:
        payload = {
            "schema_version": 1,
            "sources": [_source_pin_payload(source) for source in ledger.sources],
        }
        sources = tuple(
            _validate_source(source, index, allow_local_urls=allow_local_urls)
            for index, source in enumerate(payload["sources"])
        )
        identifiers = [source.id for source in sources]
        if len(set(identifiers)) != len(identifiers):
            raise SourceLedgerError("duplicate source id in ledger")
        digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    except (AttributeError, TypeError, ValueError) as exc:
        raise SourceLedgerError("ledger object is invalid") from exc
    if ledger.sha256 != digest or ledger.sources != sources:
        raise SourceLedgerError("ledger object does not match its canonical digest")
    return SourceLedger(schema_version=1, sources=sources, sha256=digest)


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": os.devnull,
            "SSH_ASKPASS": os.devnull,
            "GCM_INTERACTIVE": "Never",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_SSH_COMMAND": "ssh -oBatchMode=yes -oStrictHostKeyChecking=yes",
        }
    )
    return environment


def _git_arguments(*args: str) -> list[str]:
    return [
        "git",
        "-c",
        "credential.helper=",
        "-c",
        f"core.askPass={os.devnull}",
        "-c",
        "credential.interactive=never",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.ignoreCase=false",
        *args,
    ]


def _git(
    *args: str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            _git_arguments(*args),
            cwd=cwd,
            env=_git_environment(),
            capture_output=True,
            text=True,
            check=check,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise SourceLedgerError(f"Git command failed{suffix}") from exc


def _containing_git_root(path: Path) -> Path | None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir():
        probe = probe.parent
    result = _git("rev-parse", "--show-toplevel", cwd=probe, check=False)
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_git_administrative_root(root: Path) -> None:
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir():
        probe = probe.parent
    for candidate in (probe, *probe.parents):
        result = _git(
            "rev-parse",
            "--absolute-git-dir",
            "--git-common-dir",
            cwd=candidate,
            check=False,
        )
        if result.returncode != 0:
            continue
        values = [line for line in result.stdout.splitlines() if line]
        if len(values) != 2:
            raise SourceLedgerError("Git administrative directory is ambiguous")
        git_directory = Path(values[0])
        common_directory = Path(values[1])
        if not git_directory.is_absolute():
            git_directory = candidate / git_directory
        if not common_directory.is_absolute():
            common_directory = candidate / common_directory
        git_directory = git_directory.resolve()
        common_directory = common_directory.resolve()
        if _is_within(root, git_directory) or _is_within(root, common_directory):
            raise SourceLedgerError(
                "fetch root must not be inside a Git administrative directory"
            )


def _validate_fetch_root(root: Path) -> Path:
    if root.exists() and root.is_symlink():
        raise SourceLedgerError("fetch root must not be a symlink")
    resolved = root.expanduser().resolve()
    _reject_git_administrative_root(resolved)
    if resolved.exists() and not resolved.is_dir():
        raise SourceLedgerError("fetch root must be a directory")
    for reserved_name in ("checkouts", "data", "receipts"):
        reserved = resolved / reserved_name
        if reserved.is_symlink():
            raise SourceLedgerError(
                f"reserved fetch directory {reserved_name} must not be a symlink"
            )
        if reserved.exists() and not reserved.is_dir():
            raise SourceLedgerError(
                f"reserved fetch path {reserved_name} must be a directory"
            )
    repository = _containing_git_root(resolved)
    if repository is not None:
        try:
            relative = resolved.relative_to(repository)
        except ValueError:
            return resolved
        if relative == Path("."):
            raise SourceLedgerError("fetch root cannot be a Git worktree root")
        ignored = _git(
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            relative.as_posix().rstrip("/") + "/",
            cwd=repository,
            check=False,
        )
        if ignored.returncode != 0:
            raise SourceLedgerError(
                "fetch root inside a Git worktree must be ignored"
            )
    return resolved


def _contained_path(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts)
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError) as exc:
        raise SourceLedgerError(
            "source destination path escapes fetch root through a symlink or invalid path"
        ) from exc
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(receipt)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _base_receipt(source: SourcePin, ledger: SourceLedger) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_id": source.id,
        "role": source.role,
        "source_type": source.source_type,
        "source_url": source.url,
        "revision": source.revision,
        "license": source.license,
        "citation_doi": source.citation_doi,
        "ledger_sha256": ledger.sha256,
    }


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def _assert_git_admin_in_place(git_directory: Path, source_id: str) -> None:
    walk_errors: list[OSError] = []
    for current_root, directory_names, file_names in os.walk(
        git_directory, topdown=True, followlinks=False, onerror=walk_errors.append
    ):
        current = Path(current_root)
        for name in [*directory_names, *file_names]:
            candidate = current / name
            try:
                if candidate.is_symlink():
                    raise SourceLedgerError(
                        f"existing checkout {source_id} Git administrative "
                        "state contains a symlink"
                    )
            except OSError as exc:
                raise SourceLedgerError(
                    f"cannot inspect checkout {source_id} Git administrative state"
                ) from exc
    if walk_errors:
        raise SourceLedgerError(
            f"cannot inspect checkout {source_id} Git administrative state"
        ) from walk_errors[0]
    for alternate_name in ("alternates", "http-alternates"):
        alternate = git_directory / "objects" / "info" / alternate_name
        if alternate.exists() and alternate.read_bytes().strip():
            raise SourceLedgerError(
                f"existing checkout {source_id} contains an object alternate"
            )


def _assert_no_extra_worktree_entries(checkout: Path, source_id: str) -> None:
    indexed = frozenset(
        name
        for name in _git("ls-files", "-z", cwd=checkout).stdout.split("\0")
        if name
    )
    _reject_portable_collisions(
        tuple(indexed), f"existing checkout {source_id} tracked paths"
    )
    actual: set[str] = set()
    walk_errors: list[OSError] = []
    for current_root, directory_names, file_names in os.walk(
        checkout, topdown=True, followlinks=False, onerror=walk_errors.append
    ):
        current = Path(current_root)
        if current == checkout:
            directory_names[:] = [
                name for name in directory_names if name != ".git"
            ]
        for directory_name in list(directory_names):
            candidate = current / directory_name
            if candidate.is_symlink():
                actual.add(candidate.relative_to(checkout).as_posix())
                directory_names.remove(directory_name)
        for file_name in file_names:
            candidate = current / file_name
            actual.add(candidate.relative_to(checkout).as_posix())
    if walk_errors:
        raise SourceLedgerError(
            f"cannot inspect checkout {source_id} worktree entries"
        ) from walk_errors[0]
    _reject_portable_collisions(
        tuple(actual), f"existing checkout {source_id} local changes; filesystem paths"
    )
    if actual - indexed:
        raise SourceLedgerError(f"existing checkout {source_id} has local changes")


def _assert_tracked_bytes(checkout: Path, expected_tree: str, source_id: str) -> None:
    index_tree = _git("write-tree", cwd=checkout).stdout.strip()
    if index_tree != expected_tree:
        raise SourceLedgerError(
            f"existing checkout {source_id} index differs from the pinned tree"
        )
    attributes = checkout / ".git" / "info" / "attributes"
    if attributes.exists() and attributes.read_bytes().strip():
        raise SourceLedgerError(
            f"existing checkout {source_id} has local attribute rules"
        )

    records = _git("ls-files", "--stage", "-z", cwd=checkout).stdout.split("\0")
    for record in records:
        if not record:
            continue
        metadata, separator, relative_name = record.partition("\t")
        if not separator:
            raise SourceLedgerError(
                f"existing checkout {source_id} has an invalid index record"
            )
        fields = metadata.split()
        if len(fields) != 3 or fields[2] != "0":
            raise SourceLedgerError(
                f"existing checkout {source_id} has non-stage-zero index entries"
            )
        mode, expected_blob, _stage = fields
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise SourceLedgerError(
                f"existing checkout {source_id} has an unsafe tracked path"
            )
        current = checkout
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise SourceLedgerError(
                    f"existing checkout {source_id} has a symlink path escape"
                )
        tracked = checkout / relative
        try:
            file_stat = tracked.lstat()
        except OSError as exc:
            raise SourceLedgerError(
                f"existing checkout {source_id} tracked bytes are missing"
            ) from exc

        if mode in {"100644", "100755"}:
            if not stat.S_ISREG(file_stat.st_mode):
                raise SourceLedgerError(
                    f"existing checkout {source_id} tracked bytes changed type"
                )
            executable = bool(file_stat.st_mode & 0o111)
            if executable != (mode == "100755"):
                raise SourceLedgerError(
                    f"existing checkout {source_id} tracked bytes changed mode"
                )
            actual_blob = _git(
                "hash-object", "--no-filters", "--", relative_name,
                cwd=checkout,
            ).stdout.strip()
        elif mode == "120000":
            if not stat.S_ISLNK(file_stat.st_mode):
                raise SourceLedgerError(
                    f"existing checkout {source_id} tracked bytes changed type"
                )
            link_target = os.readlink(tracked)
            target_path = Path(link_target)
            resolved_target = (
                target_path.resolve(strict=False)
                if target_path.is_absolute()
                else (tracked.parent / target_path).resolve(strict=False)
            )
            checkout_root = checkout.resolve(strict=True)
            if (
                target_path.is_absolute()
                or not _is_within(resolved_target, checkout_root)
                or _is_within(resolved_target, (checkout_root / ".git").resolve())
            ):
                raise SourceLedgerError(
                    f"existing checkout {source_id} tracked symlink escapes checkout"
                )
            actual_blob = _git_blob_sha1(os.fsencode(link_target))
        elif mode == "160000":
            raise SourceLedgerError(
                f"existing checkout {source_id} contains an unsupported submodule"
            )
        else:
            raise SourceLedgerError(
                f"existing checkout {source_id} has unsupported mode {mode}"
            )
        if actual_blob != expected_blob:
            raise SourceLedgerError(
                f"existing checkout {source_id} tracked bytes differ from the pin"
            )


def _assert_pristine_checkout(checkout: Path, source: SourcePin) -> None:
    if checkout.is_symlink():
        raise SourceLedgerError(
            f"existing checkout {source.id} must not be a symlink"
        )
    git_directory = checkout / ".git"
    if not checkout.is_dir() or git_directory.is_symlink() or not git_directory.is_dir():
        raise SourceLedgerError(
            f"existing checkout {source.id} must have an in-place Git directory"
        )
    _assert_git_admin_in_place(git_directory, source.id)
    top_level = Path(
        _git("rev-parse", "--show-toplevel", cwd=checkout).stdout.strip()
    ).resolve()
    resolved_git_directory = Path(
        _git("rev-parse", "--absolute-git-dir", cwd=checkout).stdout.strip()
    ).resolve()
    common_directory = Path(
        _git("rev-parse", "--git-common-dir", cwd=checkout).stdout.strip()
    )
    if not common_directory.is_absolute():
        common_directory = checkout / common_directory
    if (
        top_level != checkout.resolve()
        or resolved_git_directory != git_directory.resolve()
        or common_directory.resolve() != git_directory.resolve()
    ):
        raise SourceLedgerError(
            f"existing checkout {source.id} Git directory/worktree escapes its path"
        )
    replacements = _git(
        "for-each-ref", "--format=%(refname)", "refs/replace", cwd=checkout
    ).stdout.strip()
    if replacements:
        raise SourceLedgerError(
            f"existing checkout {source.id} contains replacement refs"
        )
    grafts = git_directory / "info" / "grafts"
    if grafts.exists() and grafts.read_bytes().strip():
        raise SourceLedgerError(
            f"existing checkout {source.id} contains graft metadata"
        )
    unsafe_configuration = _git(
        "config",
        "--local",
        "--get-regexp",
        r"^(credential\.|filter\.|http\.|url\.|include\.|includeif\.|"
        r"submodule\.|core\.(askpass|attributesfile|gitproxy|sshcommand|worktree)$|"
        r"remote\..*\.(proxy|pushurl|receivepack|uploadpack)$)",
        cwd=checkout,
        check=False,
    )
    if unsafe_configuration.returncode == 0:
        raise SourceLedgerError(
            f"existing checkout {source.id} contains unsafe credential, filter, "
            "or transport configuration"
        )
    _assert_no_extra_worktree_entries(checkout, source.id)
    status = _git(
        "status", "--porcelain=v1", "--untracked-files=all", cwd=checkout
    ).stdout
    if status:
        raise SourceLedgerError(f"existing checkout {source.id} has local changes")
    other_files = _git("ls-files", "--others", "-z", cwd=checkout).stdout
    if other_files:
        raise SourceLedgerError(f"existing checkout {source.id} has local changes")
    revision = _git("rev-parse", "--verify", "HEAD^{commit}", cwd=checkout).stdout.strip()
    if revision != source.revision:
        raise SourceLedgerError(
            f"existing checkout {source.id} is at the wrong commit"
        )
    branch = _git("symbolic-ref", "-q", "HEAD", cwd=checkout, check=False)
    if branch.returncode == 0:
        raise SourceLedgerError(f"existing checkout {source.id} is not detached")
    remotes = _git("remote", cwd=checkout).stdout.splitlines()
    if remotes != ["origin"]:
        raise SourceLedgerError(
            f"existing checkout {source.id} has unexpected remote configuration"
        )
    origin_urls = _git(
        "remote", "get-url", "--all", "origin", cwd=checkout
    ).stdout.splitlines()
    if origin_urls != [source.url]:
        raise SourceLedgerError(f"existing checkout {source.id} has the wrong origin")
    flags = _git("ls-files", "-v", cwd=checkout).stdout.splitlines()
    if any(line and (line[0].islower() or line[0] == "S") for line in flags):
        raise SourceLedgerError(f"existing checkout {source.id} has hidden index flags")
    tree = _git("rev-parse", "HEAD^{tree}", cwd=checkout).stdout.strip()
    assert source.expected_checksum is not None
    if tree != source.expected_checksum.value:
        raise SourceLedgerError(f"existing checkout {source.id} tree checksum mismatch")
    _assert_tracked_bytes(checkout, tree, source.id)


def _clone_git_source(source: SourcePin, checkout: Path) -> None:
    checkout.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkout.parent / f".{source.id}.clone-{os.getpid()}"
    if temporary.exists():
        raise SourceLedgerError(f"temporary clone path already exists for {source.id}")
    try:
        _git(
            "clone",
            "--no-checkout",
            "--no-tags",
            "--config",
            "core.autocrlf=false",
            source.url,
            temporary.as_posix(),
        )
        _git("checkout", "--detach", source.revision, cwd=temporary)
        _assert_pristine_checkout(temporary, source)
        os.rename(temporary, checkout)
        _fsync_directory(checkout.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _fetch_git_source(
    source: SourcePin, root: Path, ledger: SourceLedger
) -> dict[str, object]:
    checkout = _contained_path(root, "checkouts", source.id)
    if checkout.exists():
        _assert_pristine_checkout(checkout, source)
    else:
        _clone_git_source(source, checkout)
    _assert_pristine_checkout(checkout, source)
    assert source.expected_checksum is not None
    receipt = _base_receipt(source, ledger)
    receipt.update(
        {
            "resolved_revision": source.revision,
            "verified_checksum": source.expected_checksum.as_dict(),
        }
    )
    return receipt


def _download_artifact(artifact: DataArtifact, destination: Path) -> tuple[str, int]:
    expected = artifact.expected_checksum.value
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise SourceLedgerError(
                f"existing data artifact {artifact.name} is not a regular file"
            )
        actual = _file_sha256(destination)
        if actual != expected:
            raise SourceLedgerError(
                f"existing data artifact {artifact.name} checksum mismatch"
            )
        return actual, destination.stat().st_size

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            request = Request(
                artifact.url,
                headers={"User-Agent": "MaskImpute-source-fetcher/1"},
            )
            with urlopen(request, timeout=120) as response:  # noqa: S310
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise SourceLedgerError(
                f"downloaded data artifact {artifact.name} checksum mismatch"
            )
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return actual, size
    except SourceLedgerError:
        raise
    except (OSError, ValueError) as exc:
        raise SourceLedgerError(
            f"failed to fetch data artifact {artifact.name}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _fetch_data_source(
    source: SourcePin, root: Path, ledger: SourceLedger
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    destination_root = _contained_path(root, "data", source.id)
    if destination_root.is_symlink():
        raise SourceLedgerError(
            f"data destination for {source.id} must not be a symlink"
        )
    if destination_root.exists() and not destination_root.is_dir():
        raise SourceLedgerError(
            f"data destination for {source.id} must be a directory"
        )
    for artifact in source.artifacts:
        sha256, size = _download_artifact(
            artifact, _contained_path(root, "data", source.id, artifact.name)
        )
        artifacts.append(
            {"name": artifact.name, "sha256": sha256, "size_bytes": size}
        )
    receipt = _base_receipt(source, ledger)
    receipt.update(
        {
            "resolved_revision": source.revision,
            "verified_checksum": None,
            "artifacts": artifacts,
        }
    )
    return receipt


def fetch_sources(
    ledger: SourceLedger,
    root: Path,
    *,
    source_ids: Sequence[str] | None = None,
    allow_local_urls: bool = False,
) -> tuple[dict[str, object], ...]:
    """Fetch immutable pins beneath an explicit external/ignored root.

    Existing Git checkouts are inspected but never fetched or advanced. Existing
    data artifacts are reused only after matching their pinned SHA-256.
    """

    if not isinstance(root, Path):
        raise SourceLedgerError("fetch root must be a pathlib.Path")
    ledger = _revalidate_ledger_object(
        ledger, allow_local_urls=allow_local_urls
    )
    root = _validate_fetch_root(root)
    by_id = {source.id: source for source in ledger.sources}
    if source_ids is None:
        selected = list(ledger.sources)
    else:
        if isinstance(source_ids, (str, bytes)):
            raise SourceLedgerError("source_ids must be a sequence of identifiers")
        if len(set(source_ids)) != len(source_ids):
            raise SourceLedgerError("source_ids must not contain duplicates")
        unknown = [source_id for source_id in source_ids if source_id not in by_id]
        if unknown:
            raise SourceLedgerError(f"unknown source ids: {unknown!r}")
        selected = [by_id[source_id] for source_id in source_ids]
    if not selected:
        raise SourceLedgerError("at least one source must be selected")

    receipts: list[dict[str, object]] = []
    for source in selected:
        if source.eligibility != "eligible":
            raise SourceLedgerError(
                f"source {source.id} is {source.eligibility}: "
                f"{source.ineligibility_reason}"
            )
        if not allow_local_urls:
            _validate_url(source.url, f"source {source.id} URL", allow_local_urls=False)
            for artifact in source.artifacts:
                _validate_url(
                    artifact.url,
                    f"source {source.id} artifact URL",
                    allow_local_urls=False,
                )
        receipt = (
            _fetch_git_source(source, root, ledger)
            if source.source_type == "git"
            else _fetch_data_source(source, root, ledger)
        )
        _write_receipt(
            _contained_path(root, "receipts", f"{source.id}.json"), receipt
        )
        receipts.append(receipt)
    return tuple(receipts)


__all__ = [
    "Checksum",
    "DataArtifact",
    "SourceLedger",
    "SourceLedgerError",
    "SourcePin",
    "fetch_sources",
    "load_source_ledger",
]
