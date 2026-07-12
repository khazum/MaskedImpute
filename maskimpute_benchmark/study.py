"""Fail-closed state records for sealed publication benchmark rounds.

The final seed manifest is revealed only after a clean code/config/environment
freeze.  Verifying a final round atomically claims its sole execution, and a
receipt can be written only from that claim while every frozen binding still
matches.  A failed claimed run must be superseded rather than silently retried.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{32}$")
_ROUND_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class StudyStateError(RuntimeError):
    """Raised when a benchmark transition or integrity check is invalid."""


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


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    """Publish complete JSON atomically, optionally with create-once semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            destination.write(_json_text(payload))
            destination.flush()
            os.fsync(destination.fileno())
        if exclusive:
            # A hard link provides O_EXCL-like publication without ever exposing
            # the empty reservation file used by the previous implementation.
            os.link(temporary, path)
        else:
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except FileNotFoundError as exc:
        raise StudyStateError(f"missing round record: {path.name}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StudyStateError(f"invalid round record {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyStateError(f"invalid round record {path.name}: expected an object")
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
    destination = Path(round_dir).resolve()
    if repo is None:
        try:
            root = _git(destination, "rev-parse", "--show-toplevel")
        except StudyStateError as exc:
            raise StudyStateError(
                "repository could not be derived from round directory; pass repo"
            ) from exc
        repository = _repo_path(Path(root))
    else:
        repository = _repo_path(repo)
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


def _study_state_root(repo: Path) -> Path:
    return _git_common_dir(repo) / GIT_STATE_DIR_NAME


def _registry_path(repo: Path, round_id: str) -> Path:
    return _study_state_root(repo) / REGISTRY_DIR_NAME / f"{round_id}.json"


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
    lock_dir = _study_state_root(repo) / LOCKS_DIR_NAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{round_id}.lock"
    try:
        descriptor = os.open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
    except OSError as exc:
        raise StudyStateError(f"could not lock study round {round_id}: {exc}") from exc
    with os.fdopen(descriptor, "a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            stat = os.fstat(lock_file.fileno())
            identity = (stat.st_dev, stat.st_ino)
        except OSError as exc:
            raise StudyStateError(
                f"could not lock study round {round_id}: {exc}"
            ) from exc
        try:
            yield identity
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _assert_round_lock_identity(
    repo: Path, round_id: str, identity: tuple[int, int]
) -> None:
    lock_path = _study_state_root(repo) / LOCKS_DIR_NAME / f"{round_id}.lock"
    try:
        stat = lock_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise StudyStateError("study round lock identity changed") from exc
    if (stat.st_dev, stat.st_ino) != identity:
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
        "method_commit": freeze.get("method_commit"),
        "config_sha256": freeze.get("config_sha256"),
        "protocol_sha256": freeze.get("protocol_sha256"),
        "environment_sha256": freeze.get("environment_sha256"),
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
    return freeze


def _validate_bindings(record: Mapping[str, Any], freeze: Mapping[str, Any]) -> None:
    if any(record.get(key) != value for key, value in _binding_fields(freeze).items()):
        raise StudyStateError("round record bindings do not match freeze")


def _registry_entry(
    *,
    state: str,
    record_name: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "state": state,
        "at": _utc_now(),
        "record_name": record_name,
        "record_sha256": canonical_sha256(dict(record)),
    }


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
    state = registry.get("state")
    if not isinstance(state, str) or history[-1].get("state") != state:
        raise StudyStateError("round registry state is invalid")
    for entry in history:
        if (
            not isinstance(entry.get("state"), str)
            or not isinstance(entry.get("record_name"), str)
            or not isinstance(entry.get("record_sha256"), str)
            or not _SHA256_RE.fullmatch(entry["record_sha256"])
        ):
            raise StudyStateError("round registry history is invalid")
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
) -> dict[str, Any]:
    registry = _validate_registry(
        repo, round_dir, freeze, expected_state=expected_state
    )
    advanced = dict(registry)
    advanced["state"] = new_state
    advanced["history"] = [
        *registry["history"],
        _registry_entry(state=new_state, record_name=record_name, record=record),
    ]
    _atomic_write_json(_registry_path(repo, round_dir.name), advanced)
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


def _repository_is_clean_at(
    repo: Path, commit: str, *, _visited: set[Path] | None = None
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
                _visited=visited,
            ):
                return False
        except StudyStateError:
            return False
    return True


def _verify_frozen_repository(repo: Path, round_dir: Path) -> dict[str, Any]:
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
        valid = (
            _repository_instance_id(repo, create=False)
            == freeze["repository_instance_id"]
            and _worktree_path_sha256(repo) == freeze["worktree_path_sha256"]
            and _repository_is_clean_at(repo, freeze["method_commit"])
            and file_sha256(config) == freeze["config_sha256"]
            and file_sha256(protocol) == freeze["protocol_sha256"]
            and file_sha256(environment) == freeze["environment_sha256"]
        )
    except (OSError, StudyStateError, TypeError, ValueError):
        valid = False
    if not valid:
        raise StudyStateError(failure)
    return freeze


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
    if claim.get("seed_manifest_sha256") != materialization.get(
        "seed_manifest_sha256"
    ):
        raise StudyStateError("execution claim seed manifest does not match")
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


def _reconcile_registry(
    repo: Path, round_dir: Path, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    """Finish a record-before-registry transition after an interrupted write."""

    registry = _validate_registry(repo, round_dir, freeze)
    while True:
        state = registry["state"]
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
            registry = _advance_registry(
                repo,
                round_dir,
                freeze,
                expected_state="running",
                new_state="evaluated",
                record_name=EVALUATION_RECEIPT_NAME,
                record=receipt,
            )
            continue
        return registry


def _supersede_integrity_failure_locked(
    repo: Path,
    round_dir: Path,
    freeze: Mapping[str, Any],
    *,
    reason: str,
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
        _atomic_write_json(round_dir / SUPERSESSION_NAME, record, exclusive=True)
    except FileExistsError:
        existing = _require_round_record(
            round_dir, SUPERSESSION_NAME, "superseded"
        )
        _validate_bindings(existing, freeze)
        record = existing
    _advance_registry(
        repo,
        round_dir,
        freeze,
        expected_state=registry["state"],
        new_state="superseded",
        record_name=SUPERSESSION_NAME,
        record=record,
    )


def freeze_round(
    repo: Path,
    round_dir: Path,
    config_path: Path,
    protocol_path: Path,
    *,
    environment_path: Path,
) -> dict[str, Any]:
    """Freeze a clean commit and tracked config, protocol, and environment lock."""

    repository = _repo_path(repo)
    destination = _round_path(repository, round_dir)
    with _round_lock(repository, destination.name) as lock_identity:
        repository_instance_id = _repository_instance_id(repository, create=True)
        commit = _git(repository, "rev-parse", "HEAD")
        if not _repository_is_clean_at(repository, commit):
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

        record: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "round_path": _round_relative_path(repository, destination),
            "round_token": secrets.token_hex(16),
            "repository_instance_id": repository_instance_id,
            "worktree_path_sha256": _worktree_path_sha256(repository),
            "state": "frozen",
            "frozen_at": _utc_now(),
            "method_commit": commit,
            "config_path": config_relative,
            "config_sha256": file_sha256(config),
            "protocol_path": protocol_relative,
            "protocol_sha256": file_sha256(protocol),
            "environment_path": environment_relative,
            "environment_sha256": file_sha256(environment),
        }
        registry: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "round_path": record["round_path"],
            "round_token": record["round_token"],
            "repository_instance_id": record["repository_instance_id"],
            "worktree_path_sha256": record["worktree_path_sha256"],
            "state": "frozen",
            "history": [
                _registry_entry(
                    state="frozen", record_name=FREEZE_NAME, record=record
                )
            ],
        }
        # The registry is the authoritative reservation.  Publishing it first
        # makes any partial freeze fail closed rather than reusable.
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _atomic_write_json(
            _registry_path(repository, destination.name), registry, exclusive=True
        )
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _atomic_write_json(destination / FREEZE_NAME, record, exclusive=True)
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


def materialize_final(
    round_dir: Path, seed_count: int, *, repo: Path | None = None
) -> dict[str, Any]:
    """Reveal a unique final seed manifest after rechecking every frozen binding."""

    raw_destination = Path(round_dir).resolve()
    if _current_state(raw_destination) is None:
        raise StudyStateError("round must be frozen before final materialization")
    if type(seed_count) is not int or seed_count <= 0:
        raise StudyStateError("seed_count must be a positive integer")

    repository, destination = _repository_for_round(raw_destination, repo)
    with _round_lock(repository, destination.name) as lock_identity:
        freeze = _verify_frozen_repository(repository, destination)
        registry = _reconcile_registry(repository, destination, freeze)
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
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="repository integrity changed during final materialization",
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _advance_registry(
            repository,
            destination,
            freeze,
            expected_state="frozen",
            new_state="materialized",
            record_name=MATERIALIZATION_NAME,
            record=record,
        )
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


def assert_final_runnable(repo: Path, round_dir: Path) -> dict[str, Any]:
    """Verify and atomically claim the single permitted final execution."""

    repository, destination = _repository_for_round(round_dir, repo)
    with _round_lock(repository, destination.name) as lock_identity:
        freeze = _verify_frozen_repository(repository, destination)
        registry = _reconcile_registry(repository, destination, freeze)
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
        record: dict[str, Any] = {
            "schema_version": 1,
            "round_id": destination.name,
            "state": "running",
            "claimed_at": _utc_now(),
            "execution_claim_id": secrets.token_hex(16),
            "seed_manifest_sha256": materialization["seed_manifest_sha256"],
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
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="repository integrity changed while claiming final execution",
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _advance_registry(
            repository,
            destination,
            freeze,
            expected_state="materialized",
            new_state="running",
            record_name=EXECUTION_CLAIM_NAME,
            record=record,
        )
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
    with _round_lock(repository, destination.name) as lock_identity:
        freeze = _verify_frozen_repository(repository, destination)
        registry = _reconcile_registry(repository, destination, freeze)
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

        manifest = dict(result_manifest)
        try:
            result_hash = canonical_sha256(manifest)
            # Validate the indented representation too, so direct API callers and
            # CLI callers have the same finite-JSON contract.
            _json_text(manifest)
        except (TypeError, ValueError) as exc:
            raise StudyStateError(f"record is not valid JSON: {exc}") from exc
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
        _verify_frozen_repository(repository, destination)
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _atomic_write_json(
                destination / EVALUATION_RECEIPT_NAME, record, exclusive=True
            )
        except FileExistsError as exc:
            raise StudyStateError("final round was already evaluated") from exc
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        try:
            _verify_frozen_repository(repository, destination)
        except StudyStateError:
            _supersede_integrity_failure_locked(
                repository,
                destination,
                freeze,
                reason="repository integrity changed while publishing evaluation receipt",
            )
            raise
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        _advance_registry(
            repository,
            destination,
            freeze,
            expected_state="running",
            new_state="evaluated",
            record_name=EVALUATION_RECEIPT_NAME,
            record=record,
        )
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
        registry = _reconcile_registry(repository, destination, freeze)
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
        )
        _assert_round_lock_identity(repository, destination.name, lock_identity)
        return record


__all__ = [
    "StudyStateError",
    "assert_final_runnable",
    "freeze_round",
    "materialize_final",
    "record_final_evaluation",
    "supersede_round",
]
