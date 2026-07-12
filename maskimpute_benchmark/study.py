"""Fail-closed state records for sealed publication benchmark rounds.

The final seed manifest is revealed only after a clean code/config/environment
freeze.  Verifying a final round atomically claims its sole execution, and a
receipt can be written only from that claim while every frozen binding still
matches.  A failed claimed run must be superseded rather than silently retried.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
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

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class StudyStateError(RuntimeError):
    """Raised when a benchmark transition or integrity check is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


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
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except FileNotFoundError as exc:
        raise StudyStateError(f"missing round record: {path.name}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StudyStateError(f"invalid round record {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyStateError(f"invalid round record {path.name}: expected an object")
    return payload


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
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
    try:
        destination.relative_to(repository)
    except ValueError as exc:
        raise StudyStateError("round directory must be inside the repository") from exc
    return repository, destination


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
    if record.get("schema_version") != 1:
        raise StudyStateError(f"invalid round record {filename}: schema_version")
    if record.get("state") != expected_state:
        raise StudyStateError(f"invalid round record {filename}: state")
    if record.get("round_id") != round_dir.name:
        raise StudyStateError(f"invalid round record {filename}: round_id")
    return record


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise StudyStateError(f"invalid {label}")
    return value


def _binding_fields(freeze: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method_commit": freeze.get("method_commit"),
        "config_sha256": freeze.get("config_sha256"),
        "protocol_sha256": freeze.get("protocol_sha256"),
        "environment_sha256": freeze.get("environment_sha256"),
    }


def _validate_freeze(round_dir: Path) -> dict[str, Any]:
    freeze = _require_round_record(round_dir, FREEZE_NAME, "frozen")
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


def _repository_is_clean_at(repo: Path, commit: str) -> bool:
    if _git(repo, "rev-parse", "HEAD") != commit:
        return False
    # Reject index flags that deliberately hide working-tree content.  With
    # those flags prohibited, both porcelain and the direct HEAD diff cover all
    # tracked paths, while porcelain also rejects untracked source additions.
    if _has_hidden_index_paths(repo):
        return False
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        return False
    diff = _run_git(repo, "diff", "--quiet", "--no-ext-diff", "HEAD", "--")
    return diff.returncode == 0


def _verify_frozen_repository(repo: Path, round_dir: Path) -> dict[str, Any]:
    failure = "final evaluation requires a clean frozen commit and unchanged inputs"
    try:
        freeze = _validate_freeze(round_dir)
        config = _recorded_path(repo, freeze.get("config_path"), "config")
        protocol = _recorded_path(repo, freeze.get("protocol_path"), "protocol")
        environment = _recorded_path(
            repo, freeze.get("environment_path"), "environment"
        )
        valid = (
            _repository_is_clean_at(repo, freeze["method_commit"])
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
    materialization = _require_round_record(
        round_dir, MATERIALIZATION_NAME, "materialized"
    )
    _validate_bindings(materialization, freeze)
    if materialization.get("materialization_claim_id") != materialization_claim.get(
        "claim_id"
    ):
        raise StudyStateError("materialization claim does not match final manifest")
    expected_hash = _require_sha256(
        materialization.get("seed_manifest_sha256"), "seed manifest hash"
    )
    manifest = _read_record(round_dir / FINAL_MANIFEST_NAME)
    seeds = manifest.get("generator_seeds")
    valid = (
        manifest.get("schema_version") == 1
        and manifest.get("round_id") == round_dir.name
        and isinstance(seeds, list)
        and len(seeds) > 0
        and manifest.get("seed_count") == len(seeds)
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
    return materialization, manifest


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
    if not _repository_is_clean_at(repository, _git(repository, "rev-parse", "HEAD")):
        raise StudyStateError("repository must be clean before freezing a round")
    if _current_state(destination) is not None or any(destination.glob("*.json")):
        raise StudyStateError("round already has a state record")

    config, config_relative = _input_path(repository, config_path, "config")
    protocol, protocol_relative = _input_path(repository, protocol_path, "protocol")
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
        "state": "frozen",
        "frozen_at": _utc_now(),
        "method_commit": _git(repository, "rev-parse", "HEAD"),
        "config_path": config_relative,
        "config_sha256": file_sha256(config),
        "protocol_path": protocol_relative,
        "protocol_sha256": file_sha256(protocol),
        "environment_path": environment_relative,
        "environment_sha256": file_sha256(environment),
    }
    _atomic_write_json(destination / FREEZE_NAME, record, exclusive=True)
    return record


def materialize_final(
    round_dir: Path, seed_count: int, *, repo: Path | None = None
) -> dict[str, Any]:
    """Reveal a unique final seed manifest after rechecking every frozen binding."""

    raw_destination = Path(round_dir).resolve()
    state = _current_state(raw_destination)
    if state != "frozen":
        if state == "evaluated":
            raise StudyStateError("round was already evaluated")
        if state == "superseded":
            raise StudyStateError("round is superseded")
        if state in {"materializing", "materialized", "running"}:
            raise StudyStateError("final seed manifest already exists or was claimed")
        raise StudyStateError("round must be frozen before final materialization")
    if type(seed_count) is not int or seed_count <= 0:
        raise StudyStateError("seed_count must be a positive integer")

    repository, destination = _repository_for_round(raw_destination, repo)
    freeze = _verify_frozen_repository(repository, destination)
    claim: dict[str, Any] = {
        "schema_version": 1,
        "round_id": destination.name,
        "state": "materializing",
        "claimed_at": _utc_now(),
        "claim_id": secrets.token_hex(16),
        **_binding_fields(freeze),
    }
    try:
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
    try:
        _atomic_write_json(
            destination / FINAL_MANIFEST_NAME, seed_manifest, exclusive=True
        )
        _atomic_write_json(
            destination / MATERIALIZATION_NAME, record, exclusive=True
        )
    except FileExistsError as exc:
        raise StudyStateError("final materialization records already exist") from exc
    return record


def assert_final_runnable(repo: Path, round_dir: Path) -> dict[str, Any]:
    """Verify and atomically claim the single permitted final execution."""

    repository, destination = _repository_for_round(round_dir, repo)
    state = _current_state(destination)
    if state == "evaluated":
        raise StudyStateError("final round was already evaluated")
    if state == "superseded":
        raise StudyStateError("final round is superseded")
    if state == "running":
        raise StudyStateError("final execution was already claimed")
    if state != "materialized":
        raise StudyStateError("final round must be frozen and materialized before running")

    freeze = _verify_frozen_repository(repository, destination)
    materialization, _ = _validate_seed_manifest(destination, freeze)
    record: dict[str, Any] = {
        "schema_version": 1,
        "round_id": destination.name,
        "state": "running",
        "claimed_at": _utc_now(),
        "execution_claim_id": secrets.token_hex(16),
        "seed_manifest_sha256": materialization["seed_manifest_sha256"],
        **_binding_fields(freeze),
    }
    try:
        _atomic_write_json(destination / EXECUTION_CLAIM_NAME, record, exclusive=True)
    except FileExistsError as exc:
        raise StudyStateError("final execution was already claimed") from exc
    return record


def record_final_evaluation(
    round_dir: Path,
    result_manifest: Mapping[str, Any],
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Record results from the sole claimed run after rechecking all bindings."""

    repository, destination = _repository_for_round(round_dir, repo)
    state = _current_state(destination)
    if state == "evaluated":
        raise StudyStateError("final round was already evaluated")
    if state == "superseded":
        raise StudyStateError("final round is superseded")
    if state != "running":
        raise StudyStateError("final execution must be claimed before evaluation")
    if not isinstance(result_manifest, Mapping):
        raise StudyStateError("result_manifest must be a JSON object")

    freeze = _verify_frozen_repository(repository, destination)
    materialization, _ = _validate_seed_manifest(destination, freeze)
    claim = _require_round_record(destination, EXECUTION_CLAIM_NAME, "running")
    _validate_bindings(claim, freeze)
    if not isinstance(claim.get("execution_claim_id"), str) or not claim.get(
        "execution_claim_id"
    ):
        raise StudyStateError("execution claim is invalid")
    if claim.get("seed_manifest_sha256") != materialization.get(
        "seed_manifest_sha256"
    ):
        raise StudyStateError("execution claim seed manifest does not match")

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
        "execution_claim_id": claim.get("execution_claim_id"),
        "result_manifest": manifest,
        "result_manifest_sha256": result_hash,
        "seed_manifest_sha256": materialization["seed_manifest_sha256"],
        **_binding_fields(freeze),
    }
    try:
        _atomic_write_json(
            destination / EVALUATION_RECEIPT_NAME, record, exclusive=True
        )
    except FileExistsError as exc:
        raise StudyStateError("final round was already evaluated") from exc
    return record


def supersede_round(round_dir: Path, reason: str) -> dict[str, Any]:
    """Append a terminal supersession record without deleting prior evidence."""

    if not isinstance(reason, str) or not reason.strip():
        raise StudyStateError("supersession requires a nonempty reason")
    destination = Path(round_dir).resolve()
    state = _current_state(destination)
    if state is None:
        raise StudyStateError("round must be frozen before it can be superseded")
    if state == "superseded":
        raise StudyStateError("round is already superseded")

    freeze = _validate_freeze(destination)
    record: dict[str, Any] = {
        "schema_version": 1,
        "round_id": destination.name,
        "state": "superseded",
        "previous_state": state,
        "reason": reason.strip(),
        "superseded_at": _utc_now(),
        **_binding_fields(freeze),
    }
    materialization_path = destination / MATERIALIZATION_NAME
    if materialization_path.exists():
        record["seed_manifest_sha256"] = _read_record(materialization_path).get(
            "seed_manifest_sha256"
        )
    receipt_path = destination / EVALUATION_RECEIPT_NAME
    if receipt_path.exists():
        record["result_manifest_sha256"] = _read_record(receipt_path).get(
            "result_manifest_sha256"
        )
    try:
        _atomic_write_json(destination / SUPERSESSION_NAME, record, exclusive=True)
    except FileExistsError as exc:
        raise StudyStateError("round is already superseded") from exc
    return record


__all__ = [
    "StudyStateError",
    "assert_final_runnable",
    "freeze_round",
    "materialize_final",
    "record_final_evaluation",
    "supersede_round",
]
