"""Fail-closed state records for sealed publication benchmark rounds.

The final seed manifest is revealed only after a clean code/config/environment
freeze.  Verifying a final round atomically claims its sole execution, and a
receipt can be written only from that claim while every frozen binding still
matches.  A failed claimed run must be superseded rather than silently retried.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat as stat_module
import subprocess
from typing import Any

from .protocol import canonical_sha256, file_sha256, load_protocol


FREEZE_NAME = "freeze.json"
MATERIALIZATION_CLAIM_NAME = "materialization_claim.json"
FINAL_MANIFEST_NAME = "final_manifest.json"
MATERIALIZATION_NAME = "materialization.json"
EXECUTION_CLAIM_NAME = "execution_claim.json"
EVALUATION_RECEIPT_NAME = "evaluation_receipt.json"
SUPERSESSION_NAME = "supersession.json"

ROUNDS_ROOT = Path("artifacts/study")
REGISTRY_DIR_NAME = ".registry"
LOCKS_DIR_NAME = ".locks"
GIT_STATE_DIR_NAME = "maskimpute-study"
RESULTS_DIR_NAME = "results"
RESULT_JOURNALS_DIR_NAME = ".result-journals"

_ROUND_STATE_NAMES = frozenset(
    {
        FREEZE_NAME,
        MATERIALIZATION_CLAIM_NAME,
        FINAL_MANIFEST_NAME,
        MATERIALIZATION_NAME,
        EXECUTION_CLAIM_NAME,
        EVALUATION_RECEIPT_NAME,
        SUPERSESSION_NAME,
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_ROUND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_JOURNAL_ENTRY_BYTES = 16 * 1024 * 1024


class StudyStateError(RuntimeError):
    """Raised when a benchmark transition or integrity check is invalid."""


@dataclass(frozen=True)
class _RoundLockIdentity:
    common: tuple[int, int]
    state_root: tuple[int, int]
    locks_dir: tuple[int, int]
    registry_dir: tuple[int, int]
    lock_file: tuple[int, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_text(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(payload), allow_nan=False, indent=2, sort_keys=True
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise StudyStateError(f"record is not valid JSON: {exc}") from exc


def _fsync_directory(directory: Path) -> None:
    """Durably publish directory-entry changes on supported POSIX filesystems."""

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_parents_durable(directory: Path) -> None:
    missing: list[Path] = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created)
        _fsync_directory(created.parent)


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    """Publish complete JSON atomically, optionally with create-once semantics."""

    _mkdir_parents_durable(path.parent)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            os.fchmod(destination.fileno(), 0o600)
            destination.write(_json_text(payload))
            destination.flush()
            os.fsync(destination.fileno())
        if exclusive:
            # A hard link provides O_EXCL-like publication without ever exposing
            # the empty reservation file used by the previous implementation.
            os.link(temporary, path)
        else:
            temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)


def _read_record(path: Path) -> dict[str, Any]:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            raise StudyStateError(f"invalid round record {path.name}: insecure file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as source:
            descriptor = -1
            payload = json.load(
                source,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
    except FileNotFoundError as exc:
        raise StudyStateError(f"missing round record: {path.name}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StudyStateError(f"invalid round record {path.name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise StudyStateError(f"invalid round record {path.name}: expected an object")
    return payload


def _file_state(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_private_canonical_record(path: Path, label: str) -> dict[str, Any]:
    """Read one exact 0600 record from a stable O_NOFOLLOW descriptor."""

    descriptor = -1
    try:
        named_before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or stat_module.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or _file_state(opened) != _file_state(named_before)
        ):
            raise StudyStateError(f"{label} is not a private unique regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_JOURNAL_ENTRY_BYTES + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_JOURNAL_ENTRY_BYTES:
                raise StudyStateError(f"{label} exceeds its maximum size")
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        named_after = path.lstat()
        if (
            _file_state(opened) != _file_state(opened_after)
            or _file_state(opened) != _file_state(named_after)
        ):
            raise StudyStateError(f"{label} changed while reading")
        raw = b"".join(chunks)
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except StudyStateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StudyStateError(f"invalid {label}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise StudyStateError(f"{label} must be a JSON object")
    if raw != _json_text(payload).encode("utf-8"):
        raise StudyStateError(f"{label} is not canonical JSON")
    return payload


def _git_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            env=_git_environment(),
        )
    except (FileNotFoundError, OSError) as exc:
        raise StudyStateError("Git command failed") from exc


def _run_git_input(
    repo: Path, arguments: list[str], input_text: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            input=input_text,
            env=_git_environment(),
        )
    except (FileNotFoundError, OSError) as exc:
        raise StudyStateError("Git command failed") from exc


def _git(repo: Path, *arguments: str) -> str:
    completed = _run_git(repo, *arguments)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise StudyStateError(f"Git command failed{suffix}")
    return completed.stdout.strip()


def _repo_path(repo: Path) -> Path:
    try:
        resolved = Path(repo).resolve(strict=True)
    except OSError as exc:
        raise StudyStateError(f"repository does not exist: {repo}") from exc
    if not resolved.is_dir():
        raise StudyStateError(f"repository is not a directory: {repo}")
    if _git(resolved, "rev-parse", "--show-toplevel") != str(resolved):
        raise StudyStateError(f"repository is not a Git worktree root: {repo}")
    return resolved


def _round_path(repo: Path, round_dir: Path) -> Path:
    candidate = Path(round_dir)
    if not candidate.is_absolute():
        candidate = repo / candidate
    destination = candidate.resolve()
    try:
        destination.relative_to(repo)
    except ValueError as exc:
        raise StudyStateError("round directory must be inside the repository") from exc
    canonical_root = (repo / ROUNDS_ROOT).resolve()
    if destination.parent != canonical_root:
        raise StudyStateError(
            f"round directory must be a direct child of canonical rounds root {ROUNDS_ROOT}"
        )
    if not _ROUND_ID_RE.fullmatch(destination.name):
        raise StudyStateError("round ID contains unsupported characters")
    return destination


def _repository_for_round(round_dir: Path, repo: Path | None) -> tuple[Path, Path]:
    if repo is not None:
        repository = _repo_path(repo)
        return repository, _round_path(repository, Path(round_dir))

    destination = Path(round_dir).resolve()
    if repo is None:
        try:
            root = _git(destination, "rev-parse", "--show-toplevel")
        except StudyStateError as exc:
            raise StudyStateError(
                "repository could not be derived from round directory; pass repo"
            ) from exc
        repository = _repo_path(Path(root))
    return repository, _round_path(repository, destination)


def _round_relative_path(repo: Path, round_dir: Path) -> str:
    return round_dir.relative_to(repo).as_posix()


def _git_common_dir(repo: Path) -> Path:
    value = Path(_git(repo, "rev-parse", "--git-common-dir"))
    if not value.is_absolute():
        value = repo / value
    try:
        common = value.resolve(strict=True)
    except OSError as exc:
        raise StudyStateError("Git common directory is unavailable") from exc
    if not common.is_dir():
        raise StudyStateError("Git common directory is unavailable")
    return common


def _git_common_dir_identity(repo: Path) -> tuple[int, int]:
    try:
        metadata = _git_common_dir(repo).stat()
    except OSError as exc:
        raise StudyStateError("Git common directory identity is unavailable") from exc
    return metadata.st_dev, metadata.st_ino


def _secure_authority_directory(
    path: Path, *, parent: Path, create: bool
) -> tuple[int, int]:
    created = False
    if create:
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise StudyStateError("study authority directory is unavailable") from exc
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StudyStateError("study authority directory is invalid") from exc
    try:
        opened = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
        valid = (
            stat_module.S_ISDIR(opened.st_mode)
            and stat_module.S_ISDIR(named.st_mode)
            and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
            and opened.st_uid == os.geteuid()
            and not (opened.st_mode & 0o022)
        )
        if not valid:
            raise StudyStateError("study authority directory is invalid")
        if created:
            os.fsync(descriptor)
            _fsync_directory(parent)
        return opened.st_dev, opened.st_ino
    except OSError as exc:
        raise StudyStateError("study authority directory is invalid") from exc
    finally:
        os.close(descriptor)


def _authority_directories(
    repo: Path, *, create: bool
) -> tuple[Path, Path, Path, tuple[int, int], tuple[int, int], tuple[int, int]]:
    common = _git_common_dir(repo)
    state_root = common / GIT_STATE_DIR_NAME
    state_identity = _secure_authority_directory(
        state_root, parent=common, create=create
    )
    locks_dir = state_root / LOCKS_DIR_NAME
    locks_identity = _secure_authority_directory(
        locks_dir, parent=state_root, create=create
    )
    registry_dir = state_root / REGISTRY_DIR_NAME
    registry_identity = _secure_authority_directory(
        registry_dir, parent=state_root, create=create
    )
    return (
        state_root,
        locks_dir,
        registry_dir,
        state_identity,
        locks_identity,
        registry_identity,
    )


def _study_state_root(repo: Path) -> Path:
    return _authority_directories(repo, create=False)[0]


def _registry_path(repo: Path, round_id: str) -> Path:
    registry_dir = _authority_directories(repo, create=False)[2]
    return registry_dir / f"{round_id}.json"


def _result_journal_directories(
    repo: Path, round_id: str, *, create: bool
) -> tuple[Path, Path, tuple[int, int], tuple[int, int]]:
    """Return secure authority directories for one append-only result journal."""

    if not _ROUND_ID_RE.fullmatch(round_id):
        raise StudyStateError("round ID contains unsupported characters")
    state_root = _study_state_root(repo)
    journals_root = state_root / RESULT_JOURNALS_DIR_NAME
    root_identity = _secure_authority_directory(
        journals_root, parent=state_root, create=create
    )
    journal = journals_root / round_id
    journal_identity = _secure_authority_directory(
        journal, parent=journals_root, create=create
    )
    for label, path, identity in (
        ("result journals root", journals_root, root_identity),
        ("result journal", journal, journal_identity),
    ):
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise StudyStateError(f"{label} is unavailable") from exc
        if (
            not stat_module.S_ISDIR(metadata.st_mode)
            or stat_module.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.geteuid()
            or (metadata.st_dev, metadata.st_ino) != identity
        ):
            raise StudyStateError(f"{label} must remain an owner-private directory")
    return journals_root, journal, root_identity, journal_identity


def _result_journal_anchor(
    execution_claim_id: str,
    seed_manifest_sha256: str,
    freeze: Mapping[str, Any],
    root_identity: tuple[int, int],
    journal_identity: tuple[int, int],
) -> str:
    return canonical_sha256(
        {
            "schema": "maskimpute-result-journal-anchor-v1",
            "execution_claim_id": execution_claim_id,
            "seed_manifest_sha256": seed_manifest_sha256,
            "bindings": _binding_fields(freeze),
            "result_journals_root_device": root_identity[0],
            "result_journals_root_inode": root_identity[1],
            "result_journal_device": journal_identity[0],
            "result_journal_inode": journal_identity[1],
        }
    )


def _worktree_path_sha256(repo: Path) -> str:
    return hashlib.sha256(str(repo.resolve()).encode("utf-8")).hexdigest()


def _repository_instance_id(repo: Path, *, create: bool) -> str:
    path = _study_state_root(repo) / "instance.json"
    if create and not path.exists():
        record = {
            "schema_version": 1,
            "repository_instance_id": secrets.token_hex(16),
            "created_at": _utc_now(),
        }
        try:
            _atomic_write_json(path, record, exclusive=True)
        except FileExistsError:
            pass
    record = _read_record(path)
    instance_id = record.get("repository_instance_id")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or not isinstance(instance_id, str)
        or not _TOKEN_RE.fullmatch(instance_id)
    ):
        raise StudyStateError("repository study instance is invalid")
    return instance_id


@contextmanager
def _round_lock(repo: Path, round_id: str):
    """Serialize all transitions for one repository-level round identity."""

    if not _ROUND_ID_RE.fullmatch(round_id):
        raise StudyStateError("round ID contains unsupported characters")
    (
        _,
        lock_dir,
        _,
        state_identity,
        locks_identity,
        registry_identity,
    ) = _authority_directories(repo, create=True)
    lock_path = lock_dir / f"{round_id}.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        _fsync_directory(lock_dir)
    except OSError as exc:
        raise StudyStateError(f"could not lock study round {round_id}: {exc}") from exc
    with os.fdopen(descriptor, "a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            metadata = os.fstat(lock_file.fileno())
            named = lock_path.stat(follow_symlinks=False)
            if (
                not stat_module.S_ISREG(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino)
                != (named.st_dev, named.st_ino)
            ):
                raise StudyStateError("study round lock identity changed")
            identity = _RoundLockIdentity(
                common=_git_common_dir_identity(repo),
                state_root=state_identity,
                locks_dir=locks_identity,
                registry_dir=registry_identity,
                lock_file=(metadata.st_dev, metadata.st_ino),
            )
        except OSError as exc:
            raise StudyStateError(
                f"could not lock study round {round_id}: {exc}"
            ) from exc
        try:
            yield identity
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _assert_round_lock_identity(
    repo: Path, round_id: str, identity: _RoundLockIdentity
) -> None:
    (
        _,
        locks_dir,
        _,
        state_identity,
        locks_identity,
        registry_identity,
    ) = _authority_directories(repo, create=False)
    lock_path = locks_dir / f"{round_id}.lock"
    try:
        stat = lock_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise StudyStateError("study round lock identity changed") from exc
    observed = _RoundLockIdentity(
        common=_git_common_dir_identity(repo),
        state_root=state_identity,
        locks_dir=locks_identity,
        registry_dir=registry_identity,
        lock_file=(stat.st_dev, stat.st_ino),
    )
    if observed != identity:
        raise StudyStateError("study round lock identity changed")


def _input_path(repo: Path, path: Path, label: str) -> tuple[Path, str]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(repo).as_posix()
    except (OSError, ValueError) as exc:
        raise StudyStateError(f"{label} must be a file inside the repository") from exc
    if not resolved.is_file():
        raise StudyStateError(f"{label} must be a file inside the repository")
    _git(repo, "ls-files", "--error-unmatch", "--", relative)
    return resolved, relative


def _recorded_path(repo: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise StudyStateError(f"frozen {label} path is invalid")
    resolved = (repo / value).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise StudyStateError(f"frozen {label} path is invalid") from exc
    if not resolved.is_file():
        raise StudyStateError(f"frozen {label} file is missing")
    return resolved


def _current_state(round_dir: Path) -> str | None:
    checks = (
        (SUPERSESSION_NAME, "superseded"),
        (EVALUATION_RECEIPT_NAME, "evaluated"),
        (EXECUTION_CLAIM_NAME, "running"),
        (MATERIALIZATION_NAME, "materialized"),
        (MATERIALIZATION_CLAIM_NAME, "materializing"),
        (FREEZE_NAME, "frozen"),
    )
    for filename, state in checks:
        if (round_dir / filename).exists():
            return state
    return None


def _require_round_record(
    round_dir: Path, filename: str, expected_state: str
) -> dict[str, Any]:
    record = _read_record(round_dir / filename)
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise StudyStateError(f"invalid round record {filename}: schema_version")
    if record.get("state") != expected_state:
        raise StudyStateError(f"invalid round record {filename}: state")
    if record.get("round_id") != round_dir.name:
        raise StudyStateError(
            f"round identity does not match record {filename}: round_id"
        )
    return record


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StudyStateError(f"invalid {label}")
    return value


def _binding_fields(freeze: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "round_path": freeze.get("round_path"),
        "round_token": freeze.get("round_token"),
        "repository_instance_id": freeze.get("repository_instance_id"),
        "worktree_path_sha256": freeze.get("worktree_path_sha256"),
        "git_common_dir_device": freeze.get("git_common_dir_device"),
        "git_common_dir_inode": freeze.get("git_common_dir_inode"),
        "study_state_root_device": freeze.get("study_state_root_device"),
        "study_state_root_inode": freeze.get("study_state_root_inode"),
        "registry_dir_device": freeze.get("registry_dir_device"),
        "registry_dir_inode": freeze.get("registry_dir_inode"),
        "method_commit": freeze.get("method_commit"),
        "config_sha256": freeze.get("config_sha256"),
        "protocol_sha256": freeze.get("protocol_sha256"),
        "environment_sha256": freeze.get("environment_sha256"),
        "operational_artifact_roots_sha256": freeze.get(
            "operational_artifact_roots_sha256"
        ),
    }


def _validate_freeze(round_dir: Path, repo: Path | None = None) -> dict[str, Any]:
    freeze = _require_round_record(round_dir, FREEZE_NAME, "frozen")
    if not isinstance(freeze.get("round_token"), str) or not _TOKEN_RE.fullmatch(
        freeze["round_token"]
    ):
        raise StudyStateError("invalid frozen round token")
    expected_path = (
        _round_relative_path(repo, round_dir)
        if repo is not None
        else f"{ROUNDS_ROOT.as_posix()}/{round_dir.name}"
    )
    if freeze.get("round_path") != expected_path:
        raise StudyStateError("round identity path does not match frozen record")
    if not isinstance(
        freeze.get("repository_instance_id"), str
    ) or not _TOKEN_RE.fullmatch(freeze["repository_instance_id"]):
        raise StudyStateError("invalid frozen repository instance")
    _require_sha256(
        freeze.get("worktree_path_sha256"), "frozen worktree path hash"
    )
    if (
        type(freeze.get("git_common_dir_device")) is not int
        or freeze["git_common_dir_device"] < 0
        or type(freeze.get("git_common_dir_inode")) is not int
        or freeze["git_common_dir_inode"] <= 0
    ):
        raise StudyStateError("invalid frozen Git common directory identity")
    for label in ("study_state_root", "registry_dir"):
        if (
            type(freeze.get(f"{label}_device")) is not int
            or freeze[f"{label}_device"] < 0
            or type(freeze.get(f"{label}_inode")) is not int
            or freeze[f"{label}_inode"] <= 0
        ):
            raise StudyStateError(f"invalid frozen {label} identity")
    if not isinstance(freeze.get("method_commit"), str) or not _GIT_OID_RE.fullmatch(
        freeze["method_commit"]
    ):
        raise StudyStateError("invalid frozen method commit")
    for label in ("config", "protocol", "environment"):
        path_value = freeze.get(f"{label}_path")
        if (
            not isinstance(path_value, str)
            or not path_value
            or Path(path_value).is_absolute()
        ):
            raise StudyStateError(f"invalid frozen {label} path")
        _require_sha256(freeze.get(f"{label}_sha256"), f"frozen {label} hash")
    _validated_operational_root_receipts(
        freeze.get("operational_artifact_roots"),
        freeze.get("operational_artifact_roots_sha256"),
    )
    return freeze


def _validate_bindings(record: Mapping[str, Any], freeze: Mapping[str, Any]) -> None:
    if any(record.get(key) != value for key, value in _binding_fields(freeze).items()):
        raise StudyStateError("round record bindings do not match freeze")


def _registry_entry(
    *,
    state: str,
    record_name: str,
    record: Mapping[str, Any],
    previous_entry_sha256: str | None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "state": state,
        "at": _utc_now(),
        "record_name": record_name,
        "record_sha256": canonical_sha256(dict(record)),
        "previous_entry_sha256": previous_entry_sha256,
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    return entry


def _validate_registry(
    repo: Path,
    round_dir: Path,
    freeze: Mapping[str, Any],
    *,
    expected_state: str | None = None,
) -> dict[str, Any]:
    registry = _read_record(_registry_path(repo, round_dir.name))
    if type(registry.get("schema_version")) is not int or registry["schema_version"] != 1:
        raise StudyStateError("round registry has invalid schema_version")
    valid_identity = (
        registry.get("round_id") == round_dir.name
        and registry.get("round_path") == _round_relative_path(repo, round_dir)
        and registry.get("round_token") == freeze.get("round_token")
        and registry.get("repository_instance_id")
        == freeze.get("repository_instance_id")
        and registry.get("worktree_path_sha256")
        == freeze.get("worktree_path_sha256")
        and registry.get("git_common_dir_device")
        == freeze.get("git_common_dir_device")
        and registry.get("git_common_dir_inode")
        == freeze.get("git_common_dir_inode")
        and registry.get("study_state_root_device")
        == freeze.get("study_state_root_device")
        and registry.get("study_state_root_inode")
        == freeze.get("study_state_root_inode")
        and registry.get("registry_dir_device")
        == freeze.get("registry_dir_device")
        and registry.get("registry_dir_inode")
        == freeze.get("registry_dir_inode")
    )
    if not valid_identity:
        raise StudyStateError("round identity does not match repository registry")
    history = registry.get("history")
    if (
        not isinstance(history, list)
        or not history
        or not all(isinstance(entry, Mapping) for entry in history)
    ):
        raise StudyStateError("round registry history is invalid")
    if (
        history[0].get("state") != "frozen"
        or history[0].get("record_name") != FREEZE_NAME
    ):
        raise StudyStateError("round registry history lacks frozen anchor")
    state = registry.get("state")
    if not isinstance(state, str) or history[-1].get("state") != state:
        raise StudyStateError("round registry state is invalid")
    previous_entry_sha256: str | None = None
    expected_entry_keys = {
        "state",
        "at",
        "record_name",
        "record_sha256",
        "previous_entry_sha256",
        "entry_sha256",
    }
    previous_state: str | None = None
    lifecycle = {
        "frozen": ("materialized", MATERIALIZATION_NAME),
        "materialized": ("running", EXECUTION_CLAIM_NAME),
        "running": ("evaluated", EVALUATION_RECEIPT_NAME),
    }
    for index, entry in enumerate(history):
        if (
            set(entry) != expected_entry_keys
            or not isinstance(entry.get("state"), str)
            or not isinstance(entry.get("at"), str)
            or not entry["at"]
            or not isinstance(entry.get("record_name"), str)
            or not isinstance(entry.get("record_sha256"), str)
            or not _SHA256_RE.fullmatch(entry["record_sha256"])
            or entry.get("previous_entry_sha256") != previous_entry_sha256
            or not isinstance(entry.get("entry_sha256"), str)
            or not _SHA256_RE.fullmatch(entry["entry_sha256"])
        ):
            raise StudyStateError("round registry history is invalid")
        if index > 0:
            expected_transition = lifecycle.get(previous_state or "")
            is_normal_transition = expected_transition == (
                entry["state"],
                entry["record_name"],
            )
            is_terminal_supersession = (
                previous_state != "superseded"
                and entry["state"] == "superseded"
                and entry["record_name"] == SUPERSESSION_NAME
            )
            if not (is_normal_transition or is_terminal_supersession):
                raise StudyStateError("round registry history transition is invalid")
        entry_payload = dict(entry)
        observed_entry_sha256 = entry_payload.pop("entry_sha256")
        if canonical_sha256(entry_payload) != observed_entry_sha256:
            raise StudyStateError("round registry history digest is invalid")
        previous_entry_sha256 = observed_entry_sha256
        record_name = entry["record_name"]
        if record_name not in {
            FREEZE_NAME,
            MATERIALIZATION_NAME,
            EXECUTION_CLAIM_NAME,
            EVALUATION_RECEIPT_NAME,
            SUPERSESSION_NAME,
        }:
            raise StudyStateError("round registry history record is invalid")
        actual_record = _read_record(round_dir / record_name)
        if canonical_sha256(actual_record) != entry["record_sha256"]:
            raise StudyStateError(
                f"round registry history hash does not match {record_name}"
            )
        previous_state = entry["state"]
    if registry.get("history_head_sha256") != previous_entry_sha256:
        raise StudyStateError("round registry history head is invalid")
    if expected_state is not None and state != expected_state:
        raise StudyStateError(
            f"round registry is {state}; expected {expected_state}"
        )
    return registry


def _advance_registry(
    repo: Path,
    round_dir: Path,
    freeze: Mapping[str, Any],
    *,
    expected_state: str,
    new_state: str,
    record_name: str,
    record: Mapping[str, Any],
    lock_identity: _RoundLockIdentity,
) -> dict[str, Any]:
    registry = _validate_registry(
        repo, round_dir, freeze, expected_state=expected_state
    )
    advanced = dict(registry)
    advanced["state"] = new_state
    entry = _registry_entry(
        state=new_state,
        record_name=record_name,
        record=record,
        previous_entry_sha256=registry["history_head_sha256"],
    )
    advanced["history"] = [
        *registry["history"],
        entry,
    ]
    advanced["history_head_sha256"] = entry["entry_sha256"]
    actual_record = _read_record(round_dir / record_name)
    if canonical_sha256(actual_record) != canonical_sha256(dict(record)):
        raise StudyStateError("transition record changed before registry publication")
    _assert_round_lock_identity(repo, round_dir.name, lock_identity)
    _atomic_write_json(_registry_path(repo, round_dir.name), advanced)
    _assert_round_lock_identity(repo, round_dir.name, lock_identity)
    actual_record = _read_record(round_dir / record_name)
    if canonical_sha256(actual_record) != entry["record_sha256"]:
        raise StudyStateError("transition record changed during registry publication")
    return advanced


def _has_hidden_index_paths(repo: Path) -> bool:
    output = _git(repo, "ls-files", "-v", "-z")
    if not output:
        return False
    for entry in output.split("\0"):
        if not entry:
            continue
        tag = entry[0]
        if tag == "S" or tag.islower():
            return True
    return False


def _index_entries(repo: Path) -> list[tuple[str, str, str]]:
    entries = _git(repo, "ls-files", "--stage", "-z")
    result: list[tuple[str, str, str]] = []
    for entry in entries.split("\0"):
        if not entry:
            continue
        try:
            metadata, relative = entry.split("\t", 1)
            mode, object_id, stage = metadata.split()
        except ValueError:
            raise StudyStateError("Git index contains an invalid entry")
        if stage != "0":
            raise StudyStateError("Git index contains unresolved stages")
        result.append((mode, object_id, relative))
    return result


def _gitlinks(repo: Path) -> list[tuple[Path, str]]:
    return [
        (repo / relative, object_id)
        for mode, object_id, relative in _index_entries(repo)
        if mode == "160000"
    ]


def _raw_tracked_files_match_index(repo: Path) -> bool:
    """Compare raw checkout bytes with stage-0 blobs, bypassing clean filters."""

    try:
        entries = _index_entries(repo)
    except StudyStateError:
        return False
    regular: list[tuple[str, str]] = []
    for mode, object_id, relative in entries:
        path = repo / relative
        if mode in {"100644", "100755"}:
            if not path.is_file() or path.is_symlink() or "\n" in relative:
                return False
            is_executable = bool(path.stat().st_mode & 0o111)
            if is_executable != (mode == "100755"):
                return False
            regular.append((relative, object_id))
        elif mode == "120000":
            if not path.is_symlink():
                return False
            target = os.readlink(path)
            hashed = _run_git_input(repo, ["hash-object", "--stdin"], target)
            if hashed.returncode != 0 or hashed.stdout.strip() != object_id:
                return False
        elif mode == "160000":
            if not path.is_dir():
                return False
        else:
            return False
    if regular:
        paths = "\n".join(relative for relative, _ in regular) + "\n"
        hashed = _run_git_input(
            repo,
            ["hash-object", "--no-filters", "--stdin-paths"],
            paths,
        )
        observed = hashed.stdout.splitlines()
        if hashed.returncode != 0 or len(observed) != len(regular):
            return False
        if any(
            actual != expected
            for actual, (_, expected) in zip(observed, regular, strict=True)
        ):
            return False
    return True


def _path_is_within_roots(relative: str, roots: frozenset[str]) -> bool:
    return any(relative == root or relative.startswith(root + "/") for root in roots)


def _operational_file_entry(path: Path, relative: str) -> dict[str, Any]:
    descriptor = -1
    try:
        named_before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened_before = os.fstat(descriptor)
        identity = lambda value: (  # noqa: E731 - compact stable-stat projection
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            not stat_module.S_ISREG(opened_before.st_mode)
            or identity(named_before) != identity(opened_before)
        ):
            raise StudyStateError("operational artifact is not a stable regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        named_after = path.lstat()
        if (
            identity(opened_before) != identity(opened_after)
            or identity(opened_before) != identity(named_after)
        ):
            raise StudyStateError("operational artifact changed while being hashed")
        return {
            "path": relative,
            "kind": "file",
            "mode": stat_module.S_IMODE(opened_before.st_mode),
            "size_bytes": opened_before.st_size,
            "sha256": digest.hexdigest(),
        }
    except StudyStateError:
        raise
    except OSError as error:
        raise StudyStateError("operational artifact cannot be hashed") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _operational_tree_snapshot(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        closed_root = root.resolve(strict=True)
    except OSError as error:
        raise StudyStateError("operational artifact root is unavailable") from error

    def visit(directory: Path, relative: str) -> None:
        try:
            metadata = directory.lstat()
            if not stat_module.S_ISDIR(metadata.st_mode):
                raise StudyStateError(
                    "operational artifact root contains an unsafe directory"
                )
            entries.append(
                {
                    "path": relative or ".",
                    "kind": "directory",
                    "mode": stat_module.S_IMODE(metadata.st_mode),
                }
            )
            children = sorted(os.scandir(directory), key=lambda value: value.name)
        except StudyStateError:
            raise
        except OSError as error:
            raise StudyStateError(
                "operational artifact directory cannot be enumerated"
            ) from error
        for child in children:
            child_relative = f"{relative}/{child.name}" if relative else child.name
            if "\n" in child_relative or "\x00" in child_relative:
                raise StudyStateError("operational artifact path is unsupported")
            path = Path(child.path)
            try:
                child_metadata = path.lstat()
            except OSError as error:
                raise StudyStateError("operational artifact path changed") from error
            if stat_module.S_ISDIR(child_metadata.st_mode):
                visit(path, child_relative)
            elif stat_module.S_ISREG(child_metadata.st_mode):
                entries.append(_operational_file_entry(path, child_relative))
            elif stat_module.S_ISLNK(child_metadata.st_mode):
                try:
                    target_before = os.readlink(path)
                    resolved_before = path.resolve(strict=True)
                    resolved_before.relative_to(closed_root)
                    metadata_after = path.lstat()
                    target_after = os.readlink(path)
                    resolved_after = path.resolve(strict=True)
                    resolved_after.relative_to(closed_root)
                except ValueError as error:
                    raise StudyStateError(
                        "operational artifact symlink resolves outside its closed root"
                    ) from error
                except OSError as error:
                    raise StudyStateError(
                        "operational artifact symlink target is unavailable"
                    ) from error
                if (
                    target_before != target_after
                    or resolved_before != resolved_after
                    or (
                        child_metadata.st_dev,
                        child_metadata.st_ino,
                        child_metadata.st_mode,
                        child_metadata.st_mtime_ns,
                        child_metadata.st_ctime_ns,
                    )
                    != (
                        metadata_after.st_dev,
                        metadata_after.st_ino,
                        metadata_after.st_mode,
                        metadata_after.st_mtime_ns,
                        metadata_after.st_ctime_ns,
                    )
                ):
                    raise StudyStateError("operational artifact symlink changed")
                entries.append(
                    {
                        "path": child_relative,
                        "kind": "symlink",
                        "mode": stat_module.S_IMODE(child_metadata.st_mode),
                        "target": target_before,
                    }
                )
            else:
                raise StudyStateError(
                    "operational artifact root contains a special file"
                )

    visit(root, "")
    return entries


def _operational_tree_receipt(repo: Path, relative: str) -> dict[str, Any]:
    path_value = PurePosixPath(relative)
    if (
        path_value.is_absolute()
        or not path_value.parts
        or any(part in {"", ".", ".."} for part in path_value.parts)
        or path_value.parts[0] == ".git"
    ):
        raise StudyStateError("operational artifact root path is invalid")
    root = repo.joinpath(*path_value.parts)
    try:
        metadata = root.lstat()
        if stat_module.S_ISLNK(metadata.st_mode) or not stat_module.S_ISDIR(
            metadata.st_mode
        ):
            raise StudyStateError(
                "operational artifact root must be a non-symlink directory"
            )
        if root.resolve(strict=True) != root:
            raise StudyStateError(
                "operational artifact root contains a symlinked ancestor"
            )
    except StudyStateError:
        raise
    except OSError as error:
        raise StudyStateError("operational artifact root is unavailable") from error
    first = _operational_tree_snapshot(root)
    second = _operational_tree_snapshot(root)
    if first != second:
        raise StudyStateError("operational artifact root changed while being hashed")
    return {
        "path": path_value.as_posix(),
        "entry_count": len(first),
        "tree_sha256": canonical_sha256(first),
    }


def _operational_root_receipts(
    repo: Path,
    roots: Sequence[Path],
) -> list[dict[str, Any]]:
    if isinstance(roots, (str, bytes)) or not isinstance(roots, Sequence):
        raise StudyStateError("operational artifact roots must be a path sequence")
    relatives: list[str] = []
    for value in roots:
        if not isinstance(value, Path):
            raise StudyStateError("operational artifact roots must be pathlib paths")
        candidate = value if value.is_absolute() else repo / value
        absolute = candidate.absolute()
        try:
            relative = absolute.relative_to(repo).as_posix()
        except ValueError as error:
            raise StudyStateError(
                "operational artifact root must be inside the repository"
            ) from error
        relatives.append(relative)
    if relatives != sorted(set(relatives)):
        raise StudyStateError("operational artifact roots must be unique and sorted")
    for index, first in enumerate(relatives):
        if any(
            second.startswith(first + "/") or first.startswith(second + "/")
            for second in relatives[index + 1 :]
        ):
            raise StudyStateError("operational artifact roots must not overlap")
    return [_operational_tree_receipt(repo, relative) for relative in relatives]


def _validated_operational_root_receipts(
    value: object,
    expected_sha256: object,
) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise StudyStateError("frozen operational artifact roots are invalid")
    observed_paths: list[str] = []
    for row in value:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "entry_count", "tree_sha256"}
            or not isinstance(row.get("path"), str)
            or type(row.get("entry_count")) is not int
            or row["entry_count"] < 1
        ):
            raise StudyStateError("frozen operational artifact receipt is invalid")
        path = PurePosixPath(row["path"])
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.parts[0] == ".git"
        ):
            raise StudyStateError("frozen operational artifact path is invalid")
        _require_sha256(row.get("tree_sha256"), "operational artifact tree hash")
        observed_paths.append(path.as_posix())
    if observed_paths != sorted(set(observed_paths)):
        raise StudyStateError("frozen operational artifact paths are invalid")
    for index, first in enumerate(observed_paths):
        if any(
            second.startswith(first + "/") or first.startswith(second + "/")
            for second in observed_paths[index + 1 :]
        ):
            raise StudyStateError("frozen operational artifact roots overlap")
    digest = _require_sha256(
        expected_sha256, "frozen operational artifact roots hash"
    )
    if canonical_sha256(value) != digest:
        raise StudyStateError("frozen operational artifact roots hash differs")
    return [dict(row) for row in value]


def _untracked_files_are_allowed(
    repo: Path,
    allowed_untracked: frozenset[str],
    allowed_untracked_roots: frozenset[str],
) -> bool:
    """Account for both ordinary and ignored untracked files without quoting."""

    observed: set[str] = set()
    for ignored in (False, True):
        arguments = ["ls-files", "--others", "--exclude-standard", "-z"]
        if ignored:
            arguments.insert(2, "--ignored")
        completed = _run_git(repo, *arguments)
        if completed.returncode != 0:
            return False
        observed.update(path for path in completed.stdout.split("\0") if path)
    return all(
        path in allowed_untracked
        or _path_is_within_roots(path, allowed_untracked_roots)
        for path in observed
    )


def _worktree_paths_are_allowed(
    repo: Path,
    allowed_untracked: frozenset[str],
    allowed_untracked_roots: frozenset[str],
) -> bool:
    """Walk raw directory entries so Git-invisible special paths cannot escape."""

    try:
        entries = _index_entries(repo)
    except StudyStateError:
        return False
    tracked = {relative for _, _, relative in entries}
    gitlinks = {relative for mode, _, relative in entries if mode == "160000"}
    permitted = tracked.union(allowed_untracked)
    directory_prefixes: set[str] = set()
    for relative in permitted.union(allowed_untracked_roots):
        for parent in Path(relative).parents:
            value = parent.as_posix()
            if value == ".":
                break
            directory_prefixes.add(value)

    def visit(directory: Path, relative_parent: str = "") -> bool:
        try:
            children = list(os.scandir(directory))
        except OSError:
            return False
        for child in children:
            if not relative_parent and child.name == ".git":
                continue
            relative = (
                f"{relative_parent}/{child.name}"
                if relative_parent
                else child.name
            )
            try:
                is_directory = child.is_dir(follow_symlinks=False)
            except OSError:
                return False
            if relative in gitlinks:
                if not is_directory:
                    return False
                continue
            if is_directory:
                if relative not in directory_prefixes and not _path_is_within_roots(
                    relative, allowed_untracked_roots
                ):
                    return False
                if not visit(Path(child.path), relative):
                    return False
            elif relative not in permitted and not _path_is_within_roots(
                relative, allowed_untracked_roots
            ):
                return False
        return True

    return visit(repo)


def _round_state_untracked_paths(repo: Path, round_dir: Path) -> frozenset[str]:
    allowed: set[str] = set()
    for name in _ROUND_STATE_NAMES:
        path = round_dir / name
        if not os.path.lexists(path):
            continue
        if path.is_symlink() or not path.is_file():
            raise StudyStateError(f"round state path is not a regular file: {name}")
        allowed.add(path.relative_to(repo).as_posix())
    return frozenset(allowed)


def _repository_is_clean_at(
    repo: Path,
    commit: str,
    *,
    allowed_untracked: frozenset[str] = frozenset(),
    allowed_untracked_roots: frozenset[str] = frozenset(),
    _visited: set[Path] | None = None,
) -> bool:
    visited = set() if _visited is None else _visited
    resolved_repo = repo.resolve()
    if resolved_repo in visited:
        return False
    visited.add(resolved_repo)
    try:
        common_dir = _git_common_dir(repo)
        grafts = common_dir / "info/grafts"
        if grafts.exists() and grafts.stat().st_size > 0:
            return False
        if _git(repo, "for-each-ref", "--format=%(refname)", "refs/replace"):
            return False
    except (OSError, StudyStateError):
        return False
    if _git(repo, "rev-parse", "HEAD") != commit:
        return False
    # Reject index flags that deliberately hide working-tree content.  With
    # those flags prohibited, both porcelain and the direct HEAD diff cover all
    # tracked paths, while porcelain also rejects untracked source additions.
    if _has_hidden_index_paths(repo):
        return False
    if not _raw_tracked_files_match_index(repo):
        return False
    if not _untracked_files_are_allowed(
        repo, allowed_untracked, allowed_untracked_roots
    ):
        return False
    if not _worktree_paths_are_allowed(
        repo, allowed_untracked, allowed_untracked_roots
    ):
        return False
    if _git(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ):
        return False
    diff = _run_git(
        repo,
        "diff",
        "--quiet",
        "--no-ext-diff",
        "--ignore-submodules=none",
        "HEAD",
        "--",
    )
    if diff.returncode != 0:
        return False
    for submodule, expected_object in _gitlinks(repo):
        if not submodule.is_dir():
            return False
        try:
            if (
                Path(_git(submodule, "rev-parse", "--show-toplevel")).resolve()
                != submodule.resolve()
            ):
                return False
            if not _repository_is_clean_at(
                submodule,
                expected_object,
                allowed_untracked=frozenset(),
                allowed_untracked_roots=frozenset(),
                _visited=visited,
            ):
                return False
        except StudyStateError:
            return False
    return True


def _verify_frozen_repository(
    repo: Path,
    round_dir: Path,
    *,
    allowed_result_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    failure = "final evaluation requires a clean frozen commit and unchanged inputs"
    # Record identity/schema failures are reported precisely; only mutable
    # repository/input mismatches collapse to the common integrity error.
    freeze = _validate_freeze(round_dir, repo)
    try:
        config = _recorded_path(repo, freeze.get("config_path"), "config")
        protocol = _recorded_path(repo, freeze.get("protocol_path"), "protocol")
        environment = _recorded_path(
            repo, freeze.get("environment_path"), "environment"
        )
        operational_receipts = _validated_operational_root_receipts(
            freeze.get("operational_artifact_roots"),
            freeze.get("operational_artifact_roots_sha256"),
        )
        operational_roots = frozenset(
            str(row["path"]) for row in operational_receipts
        )
        allowed_untracked = _round_state_untracked_paths(repo, round_dir).union(
            allowed_result_paths
        )
        valid = (
            _repository_instance_id(repo, create=False)
            == freeze["repository_instance_id"]
            and _worktree_path_sha256(repo) == freeze["worktree_path_sha256"]
            and _git_common_dir_identity(repo)
            == (
                freeze["git_common_dir_device"],
                freeze["git_common_dir_inode"],
            )
            and _authority_directories(repo, create=False)[3]
            == (
                freeze["study_state_root_device"],
                freeze["study_state_root_inode"],
            )
            and _authority_directories(repo, create=False)[5]
            == (
                freeze["registry_dir_device"],
                freeze["registry_dir_inode"],
            )
            and _repository_is_clean_at(
                repo,
                freeze["method_commit"],
                allowed_untracked=frozenset(allowed_untracked),
                allowed_untracked_roots=operational_roots,
            )
            and _operational_root_receipts(
                repo, tuple(repo / root for root in sorted(operational_roots))
            )
            == operational_receipts
            and file_sha256(config) == freeze["config_sha256"]
            and file_sha256(protocol) == freeze["protocol_sha256"]
            and file_sha256(environment) == freeze["environment_sha256"]
        )
    except (OSError, StudyStateError, TypeError, ValueError):
        valid = False
    if not valid:
        raise StudyStateError(failure)
    return freeze


def _hash_unique_result_file(path: Path) -> str:
    """Hash one stable regular result through its O_NOFOLLOW descriptor."""

    descriptor = -1
    try:
        named_before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _file_state(opened) != _file_state(named_before)
        ):
            raise StudyStateError("result file must be a unique regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        named_after = path.lstat()
        if (
            _file_state(opened) != _file_state(opened_after)
            or _file_state(opened) != _file_state(named_after)
        ):
            raise StudyStateError("result file changed while hashing")
        return digest.hexdigest()
    except StudyStateError:
        raise
    except OSError as exc:
        raise StudyStateError("result file could not be hashed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_result_files(
    repo: Path, round_dir: Path, manifest: Mapping[str, Any]
) -> frozenset[str]:
    """Validate the exact ignored output allowlist bound into a receipt."""

    if "result_files" not in manifest:
        return frozenset()
    entries = manifest.get("result_files")
    if type(entries) is not list:
        raise StudyStateError("result_files must be a JSON array")
    allowed: set[str] = set()
    declared: set[str] = set()
    results_root = round_dir / RESULTS_DIR_NAME
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise StudyStateError(
                "each result file must contain exactly path and sha256"
            )
        value = entry.get("path")
        if not isinstance(value, str) or not value or "\\" in value:
            raise StudyStateError("result file path is invalid")
        relative = Path(value)
        if (
            relative.is_absolute()
            or value != relative.as_posix()
            or len(relative.parts) < 2
            or relative.parts[0] != RESULTS_DIR_NAME
            or any(part in {"", ".", ".."} for part in relative.parts)
            or value in declared
        ):
            raise StudyStateError("result file path is invalid")
        declared.add(value)
        expected_hash = _require_sha256(entry.get("sha256"), "result file hash")
        candidate = round_dir.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(results_root)
        except (OSError, ValueError) as exc:
            raise StudyStateError("result file path is invalid") from exc
        if (
            candidate.is_symlink()
            or resolved != candidate.absolute()
        ):
            raise StudyStateError("result file must be a regular file")
        observed_hash = _hash_unique_result_file(candidate)
        try:
            if candidate.resolve(strict=True) != candidate.absolute():
                raise StudyStateError("result file path changed while hashing")
        except OSError as exc:
            raise StudyStateError("result file path changed while hashing") from exc
        if observed_hash != expected_hash:
            raise StudyStateError("result file hash does not match manifest")
        allowed.add(candidate.relative_to(repo).as_posix())
    return frozenset(allowed)


def _validate_result_journal(
    repo: Path,
    round_dir: Path,
    freeze: Mapping[str, Any],
    materialization: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the authority-rooted append-only chain of published results."""

    (
        _journals_root,
        journal,
        root_identity,
        journal_identity,
    ) = _result_journal_directories(repo, round_dir.name, create=False)
    if root_identity != (
        execution.get("result_journals_root_device"),
        execution.get("result_journals_root_inode"),
    ) or journal_identity != (
        execution.get("result_journal_device"),
        execution.get("result_journal_inode"),
    ):
        raise StudyStateError("result journal directory identity changed")
    expected_anchor = _result_journal_anchor(
        execution["execution_claim_id"],
        materialization["seed_manifest_sha256"],
        freeze,
        root_identity,
        journal_identity,
    )
    if execution.get("result_journal_anchor_sha256") != expected_anchor:
        raise StudyStateError("result journal anchor does not match execution claim")
    try:
        entries = sorted(journal.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise StudyStateError("result journal cannot be inspected") from exc
    expected_names = [f"{index:08d}.json" for index in range(1, len(entries) + 1)]
    if [path.name for path in entries] != expected_names:
        raise StudyStateError("result journal contains a gap or extra entry")

    execution_sha256 = canonical_sha256(dict(execution))
    previous_sha256 = expected_anchor
    cumulative: dict[str, str] = {}
    last_entry: dict[str, Any] | None = None
    for sequence, path in enumerate(entries, start=1):
        entry = _read_private_canonical_record(path, "result journal entry")
        expected_keys = {
            "schema_version",
            "round_id",
            "state",
            "execution_claim_id",
            "execution_claim_sha256",
            "seed_manifest_sha256",
            "sequence",
            "previous_entry_sha256",
            "new_result_files",
            "cumulative_result_files_sha256",
            "entry_sha256",
        }
        if set(entry) != expected_keys:
            raise StudyStateError("result journal entry has invalid schema")
        observed_entry_sha256 = entry.get("entry_sha256")
        entry_payload = dict(entry)
        entry_payload.pop("entry_sha256", None)
        if (
            entry.get("schema_version") != 1
            or entry.get("round_id") != round_dir.name
            or entry.get("state") != "running"
            or entry.get("execution_claim_id") != execution.get("execution_claim_id")
            or entry.get("execution_claim_sha256") != execution_sha256
            or entry.get("seed_manifest_sha256")
            != materialization.get("seed_manifest_sha256")
            or entry.get("sequence") != sequence
            or entry.get("previous_entry_sha256") != previous_sha256
            or not isinstance(observed_entry_sha256, str)
            or not _SHA256_RE.fullmatch(observed_entry_sha256)
            or canonical_sha256(entry_payload) != observed_entry_sha256
        ):
            raise StudyStateError("result journal entry digest or binding is invalid")
        new_files = entry.get("new_result_files")
        if not isinstance(new_files, list) or not new_files:
            raise StudyStateError("result journal entry must add result files")
        paths: list[str] = []
        for item in new_files:
            if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
                raise StudyStateError("result journal file entry is invalid")
            value = item.get("path")
            digest = item.get("sha256")
            if not isinstance(value, str):
                raise StudyStateError("result journal file path is invalid")
            _require_sha256(digest, "result journal file hash")
            if value in cumulative:
                raise StudyStateError("result journal contains a duplicate result path")
            cumulative[value] = digest
            paths.append(value)
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise StudyStateError("result journal file paths are not sorted and unique")
        cumulative_files = [
            {"path": path, "sha256": cumulative[path]} for path in sorted(cumulative)
        ]
        if entry.get("cumulative_result_files_sha256") != canonical_sha256(
            cumulative_files
        ):
            raise StudyStateError("result journal cumulative digest is invalid")
        previous_sha256 = observed_entry_sha256
        last_entry = entry

    cumulative_files = [
        {"path": path, "sha256": cumulative[path]} for path in sorted(cumulative)
    ]
    _root_after, _journal_after, root_identity_after, journal_identity_after = (
        _result_journal_directories(repo, round_dir.name, create=False)
    )
    if (
        root_identity_after != root_identity
        or journal_identity_after != journal_identity
    ):
        raise StudyStateError("result journal directory changed during validation")
    try:
        names_after = sorted(path.name for path in journal.iterdir())
    except OSError as exc:
        raise StudyStateError("result journal cannot be re-inspected") from exc
    if names_after != expected_names:
        raise StudyStateError("result journal entries changed during validation")
    allowed = _validate_result_files(
        repo, round_dir, {"result_files": cumulative_files}
    )
    return {
        "allowed_result_paths": allowed,
        "cumulative_result_files": cumulative_files,
        "head_sha256": previous_sha256,
        "last_entry": last_entry,
        "sequence": len(entries),
    }


def _canonical_incremental_manifest(
    result_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result_manifest, Mapping) or set(result_manifest) != {
        "result_files"
    }:
        raise StudyStateError(
            "incremental result manifest must contain exactly result_files"
        )
    try:
        manifest = json.loads(
            _json_text(dict(result_manifest)),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StudyStateError(f"record is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise StudyStateError("incremental result manifest must be a JSON object")
    return manifest


def _record_incremental_results_locked(
    repo: Path,
    round_dir: Path,
    freeze: Mapping[str, Any],
    result_manifest: Mapping[str, Any],
    lock_identity: _RoundLockIdentity,
) -> dict[str, Any] | None:
    manifest = _canonical_incremental_manifest(result_manifest)
    _validate_registry(repo, round_dir, freeze, expected_state="running")
    materialization, _seed_manifest = _validate_seed_manifest(round_dir, freeze)
    execution = _validate_execution_claim_record(round_dir, freeze, materialization)
    journal = _validate_result_journal(
        repo, round_dir, freeze, materialization, execution
    )
    proposed_allowed = _validate_result_files(repo, round_dir, manifest)
    proposed_entries = manifest.get("result_files")
    assert isinstance(proposed_entries, list)
    proposed: dict[str, str] = {
        str(item["path"]): str(item["sha256"])
        for item in proposed_entries
        if isinstance(item, Mapping)
    }
    if len(proposed) != len(proposed_entries):
        raise StudyStateError("incremental result manifest contains duplicate paths")
    existing_entries = journal["cumulative_result_files"]
    assert isinstance(existing_entries, list)
    existing = {
        str(item["path"]): str(item["sha256"])
        for item in existing_entries
        if isinstance(item, Mapping)
    }
    if any(proposed.get(path) != digest for path, digest in existing.items()):
        raise StudyStateError(
            "incremental result manifest must preserve every previous path and hash"
        )
    new_paths = sorted(set(proposed) - set(existing))
    _verify_frozen_repository(repo, round_dir, allowed_result_paths=proposed_allowed)
    if not new_paths:
        if set(proposed) != set(existing):
            raise StudyStateError("incremental result manifest is not monotonic")
        last_entry = journal["last_entry"]
        if last_entry is None:
            return None
        assert isinstance(last_entry, dict)
        return last_entry

    _fsync_declared_result_files(repo, round_dir, manifest)
    sequence = int(journal["sequence"]) + 1
    new_files = [{"path": path, "sha256": proposed[path]} for path in new_paths]
    cumulative_files = [
        {"path": path, "sha256": proposed[path]} for path in sorted(proposed)
    ]
    entry: dict[str, Any] = {
        "schema_version": 1,
        "round_id": round_dir.name,
        "state": "running",
        "execution_claim_id": execution["execution_claim_id"],
        "execution_claim_sha256": canonical_sha256(execution),
        "seed_manifest_sha256": materialization["seed_manifest_sha256"],
        "sequence": sequence,
        "previous_entry_sha256": journal["head_sha256"],
        "new_result_files": new_files,
        "cumulative_result_files_sha256": canonical_sha256(cumulative_files),
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    _assert_round_lock_identity(repo, round_dir.name, lock_identity)
    _journal_root, journal_dir, _root_identity, _journal_identity = (
        _result_journal_directories(repo, round_dir.name, create=False)
    )
    try:
        _atomic_write_json(journal_dir / f"{sequence:08d}.json", entry, exclusive=True)
    except FileExistsError as exc:
        raise StudyStateError("result journal sequence was already published") from exc
    try:
        _assert_round_lock_identity(repo, round_dir.name, lock_identity)
        observed = _validate_result_journal(
            repo, round_dir, freeze, materialization, execution
        )
        if observed["head_sha256"] != entry["entry_sha256"]:
            raise StudyStateError("result journal head changed during publication")
        _verify_frozen_repository(
            repo, round_dir, allowed_result_paths=proposed_allowed
        )
    except StudyStateError as integrity_error:
        try:
            _supersede_integrity_failure_locked(
                repo,
                round_dir,
                freeze,
                reason="incremental result journal failed post-publication validation",
                lock_identity=lock_identity,
            )
        except StudyStateError as supersession_error:
            raise supersession_error from integrity_error
        raise
    return entry


def record_incremental_results(
    round_dir: Path,
    result_manifest: Mapping[str, Any],
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Append exact immutable result files to a claimed round's hash chain."""

    repository, destination = _repository_for_round(round_dir, repo)
    with _round_lock(repository, destination.name) as lock_identity:
        freeze = _validate_freeze(destination, repository)
        entry = _record_incremental_results_locked(
            repository,
            destination,
            freeze,
            result_manifest,
            lock_identity,
        )
        if entry is None:
            raise StudyStateError("incremental result manifest adds no files")
        return entry


def _fsync_declared_result_files(
    repo: Path, round_dir: Path, manifest: Mapping[str, Any]
) -> frozenset[str]:
    """Persist declared result bytes and every containing directory before receipt."""

    allowed = _validate_result_files(repo, round_dir, manifest)
    directories: set[Path] = set()
    for entry in manifest.get("result_files", []):
        candidate = round_dir / entry["path"]
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(candidate, flags)
        except OSError as exc:
            raise StudyStateError("result file could not be synced") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat_module.S_ISREG(metadata.st_mode):
                raise StudyStateError("result file must be a regular file")
            os.fsync(descriptor)
        except OSError as exc:
            raise StudyStateError("result file could not be synced") from exc
        finally:
            os.close(descriptor)
        cursor = candidate.parent
        while True:
            directories.add(cursor)
            if cursor == round_dir:
                break
            cursor = cursor.parent
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)
    if _validate_result_files(repo, round_dir, manifest) != allowed:
        raise StudyStateError("result file set changed while syncing")
    return allowed


def _validate_seed_manifest(
    round_dir: Path, freeze: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    materialization_claim = _require_round_record(
        round_dir, MATERIALIZATION_CLAIM_NAME, "materializing"
    )
    _validate_bindings(materialization_claim, freeze)
    claim_id = materialization_claim.get("claim_id")
    if not isinstance(claim_id, str) or not _TOKEN_RE.fullmatch(claim_id):
        raise StudyStateError("materialization claim ID is invalid")
    materialization = _require_round_record(
        round_dir, MATERIALIZATION_NAME, "materialized"
    )
    _validate_bindings(materialization, freeze)
    if materialization.get("materialization_claim_id") != claim_id:
        raise StudyStateError("materialization claim does not match final manifest")
    if materialization.get("seed_manifest_path") != FINAL_MANIFEST_NAME:
        raise StudyStateError("seed manifest path is invalid")
    expected_hash = _require_sha256(
        materialization.get("seed_manifest_sha256"), "seed manifest hash"
    )
    manifest = _read_record(round_dir / FINAL_MANIFEST_NAME)
    seeds = manifest.get("generator_seeds")
    valid = (
        type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 1
        and manifest.get("round_id") == round_dir.name
        and isinstance(seeds, list)
        and len(seeds) > 0
        and type(manifest.get("seed_count")) is int
        and manifest["seed_count"] == len(seeds)
        and all(type(seed) is int and 0 <= seed < 2**63 for seed in seeds)
        and len(set(seeds)) == len(seeds)
    )
    if not valid:
        raise StudyStateError("final seed manifest is invalid")
    try:
        observed_hash = canonical_sha256(manifest)
    except (TypeError, ValueError) as exc:
        raise StudyStateError("final seed manifest is invalid") from exc
    if observed_hash != expected_hash:
        raise StudyStateError("final seed manifest hash does not match materialization")
    recorded_seeds = materialization.get("generator_seeds")
    if (
        not isinstance(recorded_seeds, list)
        or len(recorded_seeds) != len(seeds)
        or any(
            type(recorded) is not int or recorded != expected
            for recorded, expected in zip(recorded_seeds, seeds, strict=True)
        )
    ):
        raise StudyStateError("materialization generator seeds do not match manifest")
    return materialization, manifest


def _validate_execution_claim_record(
    round_dir: Path,
    freeze: Mapping[str, Any],
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    claim = _require_round_record(round_dir, EXECUTION_CLAIM_NAME, "running")
    _validate_bindings(claim, freeze)
    claim_id = claim.get("execution_claim_id")
    if not isinstance(claim_id, str) or not _TOKEN_RE.fullmatch(claim_id):
        raise StudyStateError("execution claim is invalid")
    if claim.get("seed_manifest_sha256") != materialization.get("seed_manifest_sha256"):
        raise StudyStateError("execution claim seed manifest does not match")
    for label in ("result_journals_root", "result_journal"):
        if (
            type(claim.get(f"{label}_device")) is not int
            or claim[f"{label}_device"] < 0
            or type(claim.get(f"{label}_inode")) is not int
            or claim[f"{label}_inode"] <= 0
        ):
            raise StudyStateError("execution claim result journal identity is invalid")
    _require_sha256(
        claim.get("result_journal_anchor_sha256"),
        "execution claim result journal anchor",
    )
    return claim


def _validate_evaluation_receipt_record(
    round_dir: Path,
    freeze: Mapping[str, Any],
    materialization: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _require_round_record(
        round_dir, EVALUATION_RECEIPT_NAME, "evaluated"
    )
    _validate_bindings(receipt, freeze)
    if receipt.get("execution_claim_id") != claim.get("execution_claim_id"):
        raise StudyStateError("evaluation receipt claim does not match")
    if receipt.get("seed_manifest_sha256") != materialization.get(
        "seed_manifest_sha256"
    ):
        raise StudyStateError("evaluation receipt seed manifest does not match")
    manifest = receipt.get("result_manifest")
    if not isinstance(manifest, Mapping):
        raise StudyStateError("evaluation receipt result manifest is invalid")
    expected = _require_sha256(
        receipt.get("result_manifest_sha256"), "result manifest hash"
    )
    if canonical_sha256(dict(manifest)) != expected:
        raise StudyStateError("evaluation receipt result manifest hash does not match")
    return receipt


def _validate_state_record_chain(
    round_dir: Path,
    freeze: Mapping[str, Any],
    *,
    expected_state: str,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Validate all immutable records required by a lifecycle state."""

    order = {"frozen": 0, "materialized": 1, "running": 2, "evaluated": 3}
    if expected_state not in order:
        raise StudyStateError("state record chain target is invalid")
    materialization: dict[str, Any] | None = None
    claim: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    if order[expected_state] >= 1:
        materialization, _ = _validate_seed_manifest(round_dir, freeze)
    if order[expected_state] >= 2:
        if materialization is None:
            raise StudyStateError("materialization record chain is incomplete")
        claim = _validate_execution_claim_record(
            round_dir, freeze, materialization
        )
    if order[expected_state] >= 3:
        if materialization is None or claim is None:
            raise StudyStateError("execution record chain is incomplete")
        receipt = _validate_evaluation_receipt_record(
            round_dir, freeze, materialization, claim
        )
    return materialization, claim, receipt


def _reconcile_registry(
    repo: Path,
    round_dir: Path,
    freeze: Mapping[str, Any],
    lock_identity: _RoundLockIdentity,
) -> dict[str, Any]:
    """Finish a record-before-registry transition after an interrupted write."""

    registry = _validate_registry(repo, round_dir, freeze)
    while True:
        state = registry["state"]
        if state == "evaluated":
            materialization, _ = _validate_seed_manifest(round_dir, freeze)
            claim = _validate_execution_claim_record(
                round_dir, freeze, materialization
            )
            receipt = _validate_evaluation_receipt_record(
                round_dir, freeze, materialization, claim
            )
            try:
                allowed_results = _validate_result_files(
                    repo, round_dir, receipt["result_manifest"]
                )
                _verify_frozen_repository(
                    repo,
                    round_dir,
                    allowed_result_paths=allowed_results,
                )
            except StudyStateError:
                _supersede_integrity_failure_locked(
                    repo,
                    round_dir,
                    freeze,
                    reason="evaluated result artifacts failed integrity validation",
                    lock_identity=lock_identity,
                )
                raise
            return registry
        supersession_path = round_dir / SUPERSESSION_NAME
        if state != "superseded" and supersession_path.exists():
            supersession = _require_round_record(
                round_dir, SUPERSESSION_NAME, "superseded"
            )
            _validate_bindings(supersession, freeze)
            if supersession.get("previous_state") != state:
                raise StudyStateError("supersession previous state does not match registry")
            registry = _advance_registry(
                repo,
                round_dir,
                freeze,
                expected_state=state,
                new_state="superseded",
                record_name=SUPERSESSION_NAME,
                record=supersession,
                lock_identity=lock_identity,
            )
            continue
        if state == "frozen" and (round_dir / MATERIALIZATION_NAME).exists():
            materialization, _ = _validate_seed_manifest(round_dir, freeze)
            registry = _advance_registry(
                repo,
                round_dir,
                freeze,
                expected_state="frozen",
                new_state="materialized",
                record_name=MATERIALIZATION_NAME,
                record=materialization,
                lock_identity=lock_identity,
            )
            continue
        if state == "materialized" and (round_dir / EXECUTION_CLAIM_NAME).exists():
            materialization, _ = _validate_seed_manifest(round_dir, freeze)
            claim = _validate_execution_claim_record(
                round_dir, freeze, materialization
            )
            registry = _advance_registry(
                repo,
                round_dir,
                freeze,
                expected_state="materialized",
                new_state="running",
                record_name=EXECUTION_CLAIM_NAME,
                record=claim,
                lock_identity=lock_identity,
            )
            continue
        if state == "running" and (round_dir / EVALUATION_RECEIPT_NAME).exists():
            materialization, _ = _validate_seed_manifest(round_dir, freeze)
            claim = _validate_execution_claim_record(
                round_dir, freeze, materialization
            )
            receipt = _validate_evaluation_receipt_record(
                round_dir, freeze, materialization, claim
            )
            try:
                allowed_results = _fsync_declared_result_files(
                    repo, round_dir, receipt["result_manifest"]
                )
                _verify_frozen_repository(
                    repo,
                    round_dir,
                    allowed_result_paths=allowed_results,
                )
                registry = _advance_registry(
                    repo,
                    round_dir,
                    freeze,
                    expected_state="running",
                    new_state="evaluated",
                    record_name=EVALUATION_RECEIPT_NAME,
                    record=receipt,
                    lock_identity=lock_identity,
                )
                _validate_result_files(repo, round_dir, receipt["result_manifest"])
                _verify_frozen_repository(
                    repo,
                    round_dir,
                    allowed_result_paths=allowed_results,
                )
            except StudyStateError:
                _supersede_integrity_failure_locked(
                    repo,
                    round_dir,
                    freeze,
                    reason="interrupted evaluation result artifacts failed integrity validation",
                    lock_identity=lock_identity,
                )
                raise
            continue
        return registry


def _supersede_integrity_failure_locked(
    repo: Path,
    round_dir: Path,
    freeze: Mapping[str, Any],
    *,
    reason: str,
    lock_identity: _RoundLockIdentity,
) -> None:
    registry = _validate_registry(repo, round_dir, freeze)
    if registry["state"] == "superseded":
        return
    record: dict[str, Any] = {
        "schema_version": 1,
        "round_id": round_dir.name,
        "state": "superseded",
        "previous_state": registry["state"],
        "observed_record_state": _current_state(round_dir),
        "reason": reason,
        "superseded_at": _utc_now(),
        **_binding_fields(freeze),
    }
    try:
        _assert_round_lock_identity(repo, round_dir.name, lock_identity)
        _atomic_write_json(round_dir / SUPERSESSION_NAME, record, exclusive=True)
    except FileExistsError:
        existing = _require_round_record(
            round_dir, SUPERSESSION_NAME, "superseded"
        )
        _validate_bindings(existing, freeze)
        record = existing
    _assert_round_lock_identity(repo, round_dir.name, lock_identity)
    _advance_registry(
        repo,
        round_dir,
        freeze,
        expected_state=registry["state"],
        new_state="superseded",
        record_name=SUPERSESSION_NAME,
        record=record,
        lock_identity=lock_identity,
    )
    _validate_registry(
        repo, round_dir, freeze, expected_state="superseded"
    )
    _assert_round_lock_identity(repo, round_dir.name, lock_identity)


def freeze_round(
    repo: Path,
    round_dir: Path,
    config_path: Path,
    protocol_path: Path,
    *,
    environment_path: Path,
    expected_config_sha256: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_environment_sha256: str | None = None,
    expected_method_commit: str | None = None,
    operational_artifact_roots: Sequence[Path] = (),
    expected_operational_artifact_roots_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze a clean commit and tracked config, protocol, and environment lock."""

    expected_hashes = (
        expected_config_sha256,
        expected_protocol_sha256,
        expected_environment_sha256,
    )
    if any(value is not None for value in expected_hashes) and not all(
        value is not None for value in expected_hashes
    ):
        raise StudyStateError("validated input checksums must be supplied together")
    if expected_config_sha256 is not None:
        _require_sha256(expected_config_sha256, "validated config checksum")
        _require_sha256(expected_protocol_sha256, "validated protocol checksum")
        _require_sha256(expected_environment_sha256, "validated environment checksum")
    if expected_method_commit is not None and (
        not isinstance(expected_method_commit, str)
        or not _GIT_OID_RE.fullmatch(expected_method_commit)
    ):
        raise StudyStateError("validated method commit is not a Git object ID")
    if expected_operational_artifact_roots_sha256 is not None:
        _require_sha256(
            expected_operational_artifact_roots_sha256,
            "validated operational artifact roots checksum",
        )

    repository = _repo_path(repo)
    destination = _round_path(repository, round_dir)
    operational_receipts = _operational_root_receipts(
        repository, operational_artifact_roots
    )
    if (
        expected_operational_artifact_roots_sha256 is not None
        and canonical_sha256(operational_receipts)
        != expected_operational_artifact_roots_sha256
    ):
        raise StudyStateError(
            "validated operational artifact roots changed before freeze"
        )
    operational_roots = frozenset(str(row["path"]) for row in operational_receipts)
    destination_relative = _round_relative_path(repository, destination)
    if any(
        destination_relative == root
        or destination_relative.startswith(root + "/")
        or root.startswith(destination_relative + "/")
        for root in operational_roots
    ):
        raise StudyStateError("operational artifact roots overlap the final round")
    with _round_lock(repository, destination.name) as lock_identity:
        repository_instance_id = _repository_instance_id(repository, create=True)
        commit = _git(repository, "rev-parse", "HEAD")
        if expected_method_commit is not None and commit != expected_method_commit:
            raise StudyStateError("validated method commit changed before freeze")
        if not _repository_is_clean_at(
            repository,
            commit,
            allowed_untracked_roots=operational_roots,
        ):
            raise StudyStateError("repository must be clean before freezing a round")
        if _current_state(destination) is not None or any(destination.glob("*.json")):
            raise StudyStateError("round already has a state record")
        if _registry_path(repository, destination.name).exists():
            raise StudyStateError("round registry already reserves this round ID")

        config, config_relative = _input_path(repository, config_path, "config")
        protocol, protocol_relative = _input_path(
            repository, protocol_path, "protocol"
        )
        environment, environment_relative = _input_path(
            repository, environment_path, "environment"
        )
        try:
            load_protocol(protocol)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise StudyStateError(f"protocol is invalid: {exc}") from exc
        config_sha256 = file_sha256(config)
        protocol_sha256 = file_sha256(protocol)
        environment_sha256 = file_sha256(environment)
        if (
            expected_config_sha256 is not None
            and config_sha256 != expected_config_sha256
        ):
            raise StudyStateError("validated config checksum changed before freeze")
        if (
            expected_protocol_sha256 is not None
            and protocol_sha256 != expected_protocol_sha256
        ):
            raise StudyStateError("validated protocol checksum changed before freeze")
        if (
            expected_environment_sha256 is not None
            and environment_sha256 != expected_environment_sha256
        ):
            raise StudyStateError("validated environment checksum changed before freeze")
        final_operational_receipts = _operational_root_receipts(
            repository, operational_artifact_roots
        )
        if final_operational_receipts != operational_receipts:
            raise StudyStateError("operational artifact roots changed during freeze")

        common_device, common_inode = _git_common_dir_identity(repository)
        record: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "round_path": _round_relative_path(repository, destination),
            "round_token": secrets.token_hex(16),
            "repository_instance_id": repository_instance_id,
            "worktree_path_sha256": _worktree_path_sha256(repository),
            "git_common_dir_device": common_device,
            "git_common_dir_inode": common_inode,
            "study_state_root_device": lock_identity.state_root[0],
            "study_state_root_inode": lock_identity.state_root[1],
            "registry_dir_device": lock_identity.registry_dir[0],
            "registry_dir_inode": lock_identity.registry_dir[1],
            "state": "frozen",
            "frozen_at": _utc_now(),
            "method_commit": commit,
            "config_path": config_relative,
            "config_sha256": config_sha256,
            "protocol_path": protocol_relative,
            "protocol_sha256": protocol_sha256,
            "environment_path": environment_relative,
            "environment_sha256": environment_sha256,
            "operational_artifact_roots": operational_receipts,
            "operational_artifact_roots_sha256": canonical_sha256(
                operational_receipts
            ),
        }
        first_entry = _registry_entry(
            state="frozen",
            record_name=FREEZE_NAME,
            record=record,
            previous_entry_sha256=None,
        )
        registry: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "round_path": record["round_path"],
            "round_token": record["round_token"],
            "repository_instance_id": record["repository_instance_id"],
            "worktree_path_sha256": record["worktree_path_sha256"],
            "git_common_dir_device": record["git_common_dir_device"],
            "git_common_dir_inode": record["git_common_dir_inode"],
            "study_state_root_device": record["study_state_root_device"],
            "study_state_root_inode": record["study_state_root_inode"],
            "registry_dir_device": record["registry_dir_device"],
            "registry_dir_inode": record["registry_dir_inode"],
            "state": "frozen",
            "history": [first_entry],
            "history_head_sha256": first_entry["entry_sha256"],
        }
        # The registry is the authoritative reservation.  Publishing it first
        # makes any partial freeze fail closed rather than reusable.
        if _git(repository, "rev-parse", "HEAD") != commit:
            raise StudyStateError("method commit changed during freeze validation")
        if (
            _operational_root_receipts(
                repository,
                tuple(repository / root for root in sorted(operational_roots)),
            )
            != operational_receipts
        ):
            raise StudyStateError(
                "operational artifact roots changed during freeze validation"
            )
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _atomic_write_json(
            _registry_path(repository, destination.name), registry, exclusive=True
        )
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _atomic_write_json(destination / FREEZE_NAME, record, exclusive=True)
        _validate_freeze(destination, repository)
        _validate_registry(
            repository, destination, record, expected_state="frozen"
        )
        _verify_frozen_repository(repository, destination)
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


def materialize_final(
    round_dir: Path, seed_count: int, *, repo: Path | None = None
) -> dict[str, Any]:
    """Reveal a unique final seed manifest after rechecking every frozen binding."""

    if type(seed_count) is not int or seed_count <= 0:
        raise StudyStateError("seed_count must be a positive integer")
    if repo is not None:
        repository, destination = _repository_for_round(round_dir, repo)
    else:
        raw_destination = Path(round_dir).resolve()
        if _current_state(raw_destination) is None:
            raise StudyStateError("round must be frozen before final materialization")
        repository, destination = _repository_for_round(raw_destination, None)
    if _current_state(destination) is None:
        raise StudyStateError("round must be frozen before final materialization")
    with _round_lock(repository, destination.name) as lock_identity:
        freeze = _verify_frozen_repository(repository, destination)
        registry = _reconcile_registry(
            repository, destination, freeze, lock_identity
        )
        state = registry["state"]
        observed_state = _current_state(destination)
        if state != "frozen" or observed_state != "frozen":
            if state == "evaluated":
                raise StudyStateError("round was already evaluated")
            if state == "superseded":
                raise StudyStateError("round is superseded")
            if state in {"materialized", "running"} or observed_state in {
                "materializing",
                "materialized",
                "running",
            }:
                raise StudyStateError("final seed manifest already exists or was claimed")
            raise StudyStateError("round must be frozen before final materialization")
        claim: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "state": "materializing",
            "claimed_at": _utc_now(),
            "claim_id": secrets.token_hex(16),
            **_binding_fields(freeze),
        }
        try:
            _assert_round_lock_identity(
                repository, destination.name, lock_identity
            )
            _atomic_write_json(
                destination / MATERIALIZATION_CLAIM_NAME, claim, exclusive=True
            )
        except FileExistsError as exc:
            raise StudyStateError("final materialization was already claimed") from exc

        seeds: list[int] = []
        used: set[int] = set()
        while len(seeds) < seed_count:
            seed = secrets.randbits(63)
            if seed not in used:
                used.add(seed)
                seeds.append(seed)
        seed_manifest: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "seed_count": seed_count,
            "generator_seeds": seeds,
        }
        record: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "state": "materialized",
            "materialized_at": _utc_now(),
            "materialization_claim_id": claim["claim_id"],
            "seed_manifest_path": FINAL_MANIFEST_NAME,
            "seed_manifest_sha256": canonical_sha256(seed_manifest),
            "generator_seeds": seeds,
            **_binding_fields(freeze),
        }
        # Close the validation-to-publication window before revealing seeds.
        _verify_frozen_repository(repository, destination)
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _atomic_write_json(
                destination / FINAL_MANIFEST_NAME, seed_manifest, exclusive=True
            )
            _atomic_write_json(
                destination / MATERIALIZATION_NAME, record, exclusive=True
            )
        except FileExistsError as exc:
            raise StudyStateError("final materialization records already exist") from exc
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _verify_frozen_repository(repository, destination)
            _validate_state_record_chain(
                destination, freeze, expected_state="materialized"
            )
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="repository integrity changed during final materialization",
                lock_identity=lock_identity,
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _advance_registry(
                repository,
                destination,
                freeze,
                expected_state="frozen",
                new_state="materialized",
                record_name=MATERIALIZATION_NAME,
                record=record,
                lock_identity=lock_identity,
            )
            _validate_state_record_chain(
                destination, freeze, expected_state="materialized"
            )
            _validate_registry(
                repository, destination, freeze, expected_state="materialized"
            )
            _verify_frozen_repository(repository, destination)
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="seed manifest changed during registry publication",
                lock_identity=lock_identity,
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


def assert_final_runnable(repo: Path, round_dir: Path) -> dict[str, Any]:
    """Verify and atomically claim the single permitted final execution."""

    repository, destination = _repository_for_round(round_dir, repo)
    with _round_lock(repository, destination.name) as lock_identity:
        freeze = _verify_frozen_repository(repository, destination)
        registry = _reconcile_registry(
            repository, destination, freeze, lock_identity
        )
        state = registry["state"]
        if state == "evaluated":
            raise StudyStateError("final round was already evaluated")
        if state == "superseded":
            raise StudyStateError("final round is superseded")
        if state == "running":
            raise StudyStateError("final execution was already claimed")
        if state != "materialized" or _current_state(destination) != "materialized":
            raise StudyStateError(
                "final round must be frozen and materialized before running"
            )
        materialization, _ = _validate_seed_manifest(destination, freeze)
        _validate_registry(
            repository, destination, freeze, expected_state="materialized"
        )
        execution_claim_id = secrets.token_hex(16)
        (
            _journals_root,
            result_journal,
            journals_root_identity,
            result_journal_identity,
        ) = _result_journal_directories(repository, destination.name, create=True)
        try:
            if any(result_journal.iterdir()):
                raise StudyStateError("result journal is not empty before final claim")
        except OSError as exc:
            raise StudyStateError("result journal cannot be inspected") from exc
        journal_anchor = _result_journal_anchor(
            execution_claim_id,
            materialization["seed_manifest_sha256"],
            freeze,
            journals_root_identity,
            result_journal_identity,
        )
        record: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "state": "running",
            "claimed_at": _utc_now(),
            "execution_claim_id": execution_claim_id,
            "seed_manifest_sha256": materialization["seed_manifest_sha256"],
            "result_journals_root_device": journals_root_identity[0],
            "result_journals_root_inode": journals_root_identity[1],
            "result_journal_device": result_journal_identity[0],
            "result_journal_inode": result_journal_identity[1],
            "result_journal_anchor_sha256": journal_anchor,
            **_binding_fields(freeze),
        }
        _verify_frozen_repository(repository, destination)
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _atomic_write_json(
                destination / EXECUTION_CLAIM_NAME, record, exclusive=True
            )
        except FileExistsError as exc:
            raise StudyStateError("final execution was already claimed") from exc
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _verify_frozen_repository(repository, destination)
            _validate_state_record_chain(
                destination, freeze, expected_state="running"
            )
            current_materialization, _manifest = _validate_seed_manifest(
                destination, freeze
            )
            current_execution = _validate_execution_claim_record(
                destination, freeze, current_materialization
            )
            if _validate_result_journal(
                repository,
                destination,
                freeze,
                current_materialization,
                current_execution,
            )["sequence"] != 0:
                raise StudyStateError("new execution claim journal is not empty")
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="repository integrity changed while claiming final execution",
                lock_identity=lock_identity,
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _advance_registry(
                repository,
                destination,
                freeze,
                expected_state="materialized",
                new_state="running",
                record_name=EXECUTION_CLAIM_NAME,
                record=record,
                lock_identity=lock_identity,
            )
            _validate_state_record_chain(
                destination, freeze, expected_state="running"
            )
            _validate_registry(
                repository, destination, freeze, expected_state="running"
            )
            current_materialization, _manifest = _validate_seed_manifest(
                destination, freeze
            )
            current_execution = _validate_execution_claim_record(
                destination, freeze, current_materialization
            )
            if _validate_result_journal(
                repository,
                destination,
                freeze,
                current_materialization,
                current_execution,
            )["sequence"] != 0:
                raise StudyStateError("new execution claim journal is not empty")
            _verify_frozen_repository(repository, destination)
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="execution claim changed during registry publication",
                lock_identity=lock_identity,
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


def record_final_evaluation(
    round_dir: Path,
    result_manifest: Mapping[str, Any],
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Record results from the sole claimed run after rechecking all bindings."""

    repository, destination = _repository_for_round(round_dir, repo)
    if not isinstance(result_manifest, Mapping):
        raise StudyStateError("result_manifest must be a JSON object")
    try:
        manifest_text = _json_text(dict(result_manifest))
        manifest = json.loads(
            manifest_text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        result_hash = canonical_sha256(manifest)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StudyStateError(f"record is not valid JSON: {exc}") from exc
    with _round_lock(repository, destination.name) as lock_identity:
        freeze = _validate_freeze(destination, repository)
        if (destination / EVALUATION_RECEIPT_NAME).exists():
            try:
                existing_materialization, _ = _validate_seed_manifest(
                    destination, freeze
                )
                existing_claim = _validate_execution_claim_record(
                    destination, freeze, existing_materialization
                )
                existing_receipt = _validate_evaluation_receipt_record(
                    destination,
                    freeze,
                    existing_materialization,
                    existing_claim,
                )
                allowed_result_paths = _validate_result_files(
                    repository,
                    destination,
                    existing_receipt["result_manifest"],
                )
                freeze = _verify_frozen_repository(
                    repository,
                    destination,
                    allowed_result_paths=allowed_result_paths,
                )
            except StudyStateError:
                _supersede_integrity_failure_locked(
                    repository,
                    destination,
                    freeze,
                    reason="published evaluation receipt failed artifact validation",
                    lock_identity=lock_identity,
                )
                raise
        else:
            freeze = _validate_freeze(destination, repository)
            pre_journal_registry = _validate_registry(repository, destination, freeze)
            if pre_journal_registry.get("state") != "running":
                raise StudyStateError(
                    "final execution must be claimed before evaluation"
                )
            _record_incremental_results_locked(
                repository,
                destination,
                freeze,
                {"result_files": manifest.get("result_files", [])},
                lock_identity,
            )
            materialization, _seed_manifest = _validate_seed_manifest(
                destination, freeze
            )
            execution = _validate_execution_claim_record(
                destination, freeze, materialization
            )
            journal = _validate_result_journal(
                repository,
                destination,
                freeze,
                materialization,
                execution,
            )
            allowed_result_paths = _validate_result_files(
                repository, destination, manifest
            )
            if allowed_result_paths != journal["allowed_result_paths"]:
                raise StudyStateError(
                    "final result manifest does not reconcile with result journal"
                )
            freeze = _verify_frozen_repository(
                repository,
                destination,
                allowed_result_paths=allowed_result_paths,
            )
        registry = _reconcile_registry(
            repository, destination, freeze, lock_identity
        )
        state = registry["state"]
        if state == "evaluated":
            raise StudyStateError("final round was already evaluated")
        if state == "superseded":
            raise StudyStateError("final round is superseded")
        if state != "running" or _current_state(destination) != "running":
            raise StudyStateError("final execution must be claimed before evaluation")
        materialization, _ = _validate_seed_manifest(destination, freeze)
        claim = _validate_execution_claim_record(
            destination, freeze, materialization
        )
        claim_id = claim["execution_claim_id"]
        _validate_registry(
            repository, destination, freeze, expected_state="running"
        )

        record: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "state": "evaluated",
            "evaluated_at": _utc_now(),
            "execution_claim_id": claim_id,
            "result_manifest": manifest,
            "result_manifest_sha256": result_hash,
            "seed_manifest_sha256": materialization["seed_manifest_sha256"],
            **_binding_fields(freeze),
        }
        # Recheck after result hashing and immediately before receipt publication.
        allowed_result_paths = _fsync_declared_result_files(
            repository, destination, manifest
        )
        _validate_state_record_chain(
            destination, freeze, expected_state="running"
        )
        _verify_frozen_repository(
            repository,
            destination,
            allowed_result_paths=allowed_result_paths,
        )
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _atomic_write_json(
                destination / EVALUATION_RECEIPT_NAME, record, exclusive=True
            )
        except FileExistsError as exc:
            raise StudyStateError("final round was already evaluated") from exc
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _validate_state_record_chain(
                destination, freeze, expected_state="evaluated"
            )
            _validate_result_files(repository, destination, manifest)
            _verify_frozen_repository(
                repository,
                destination,
                allowed_result_paths=allowed_result_paths,
            )
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="repository integrity changed while publishing evaluation receipt",
                lock_identity=lock_identity,
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _advance_registry(
                repository,
                destination,
                freeze,
                expected_state="running",
                new_state="evaluated",
                record_name=EVALUATION_RECEIPT_NAME,
                record=record,
                lock_identity=lock_identity,
            )
            _validate_state_record_chain(
                destination, freeze, expected_state="evaluated"
            )
            _validate_result_files(repository, destination, manifest)
            _validate_registry(
                repository, destination, freeze, expected_state="evaluated"
            )
            _verify_frozen_repository(
                repository,
                destination,
                allowed_result_paths=allowed_result_paths,
            )
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="result artifacts changed during registry publication",
                lock_identity=lock_identity,
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


def supersede_round(round_dir: Path, reason: str) -> dict[str, Any]:
    """Append a terminal supersession record without deleting prior evidence."""

    if not isinstance(reason, str) or not reason.strip():
        raise StudyStateError("supersession requires a nonempty reason")
    raw_destination = Path(round_dir).resolve()
    if _current_state(raw_destination) is None:
        raise StudyStateError("round must be frozen before it can be superseded")
    repository, destination = _repository_for_round(raw_destination, None)
    with _round_lock(repository, destination.name) as lock_identity:
        state = _current_state(destination)
        if state is None:
            raise StudyStateError("round must be frozen before it can be superseded")
        freeze = _validate_freeze(destination, repository)
        registry = _reconcile_registry(
            repository, destination, freeze, lock_identity
        )
        if registry["state"] == "superseded":
            return _require_round_record(
                destination, SUPERSESSION_NAME, "superseded"
            )
        authoritative_state = registry["state"]
        record: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "state": "superseded",
            "previous_state": authoritative_state,
            "observed_record_state": state,
            "reason": reason.strip(),
            "superseded_at": _utc_now(),
            **_binding_fields(freeze),
        }
        materialization_path = destination / MATERIALIZATION_NAME
        if materialization_path.exists():
            record["seed_manifest_sha256"] = _read_record(
                materialization_path
            ).get("seed_manifest_sha256")
        receipt_path = destination / EVALUATION_RECEIPT_NAME
        if receipt_path.exists():
            record["result_manifest_sha256"] = _read_record(receipt_path).get(
                "result_manifest_sha256"
            )
        try:
            _assert_round_lock_identity(
                repository, destination.name, lock_identity
            )
            _atomic_write_json(
                destination / SUPERSESSION_NAME, record, exclusive=True
            )
        except FileExistsError as exc:
            raise StudyStateError("round is already superseded") from exc
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _advance_registry(
            repository,
            destination,
            freeze,
            expected_state=authoritative_state,
            new_state="superseded",
            record_name=SUPERSESSION_NAME,
            record=record,
            lock_identity=lock_identity,
        )
        _validate_registry(
            repository, destination, freeze, expected_state="superseded"
        )
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


__all__ = [
    "StudyStateError",
    "assert_final_runnable",
    "freeze_round",
    "materialize_final",
    "record_incremental_results",
    "record_final_evaluation",
    "supersede_round",
]
