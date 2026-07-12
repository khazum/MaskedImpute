"""One-use state records for frozen publication benchmark rounds."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any

from .protocol import canonical_sha256, file_sha256, load_protocol


FREEZE_NAME = "freeze.json"
FINAL_MANIFEST_NAME = "final_manifest.json"
MATERIALIZATION_NAME = "materialization.json"
EVALUATION_RECEIPT_NAME = "evaluation_receipt.json"
SUPERSESSION_NAME = "supersession.json"


class StudyStateError(RuntimeError):
    """Raised when a benchmark round transition or integrity check is invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_text(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(payload),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise StudyStateError(f"record is not valid JSON: {exc}") from exc


def _atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, exclusive: bool = False
) -> None:
    """Write JSON through a temporary sibling and an atomic ``Path.replace``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    text = _json_text(payload)
    reserved = False
    try:
        with temporary.open("x", encoding="utf-8") as destination:
            destination.write(text)
            destination.flush()
            os.fsync(destination.fileno())
        if exclusive:
            with path.open("x", encoding="utf-8"):
                pass
            reserved = True
        temporary.replace(path)
        reserved = False
    finally:
        temporary.unlink(missing_ok=True)
        # A failed replace leaves the exclusive empty marker in place. Keeping it
        # is fail-closed: the one-use evaluation cannot silently run again.
        if reserved:
            pass


def _read_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StudyStateError(f"missing round record: {path.name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyStateError(f"invalid round record {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyStateError(f"invalid round record {path.name}: expected an object")
    return payload


def _git(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise StudyStateError(f"Git command failed{suffix}") from exc
    return completed.stdout.strip()


def _repo_path(repo: Path) -> Path:
    try:
        resolved = Path(repo).resolve(strict=True)
    except OSError as exc:
        raise StudyStateError(f"repository does not exist: {repo}") from exc
    if not resolved.is_dir():
        raise StudyStateError(f"repository is not a directory: {repo}")
    return resolved


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
    return resolved


def _current_state(round_dir: Path) -> str | None:
    if (round_dir / SUPERSESSION_NAME).exists():
        return "superseded"
    if (round_dir / EVALUATION_RECEIPT_NAME).exists():
        return "evaluated"
    if (round_dir / MATERIALIZATION_NAME).exists():
        return "materialized"
    if (round_dir / FREEZE_NAME).exists():
        return "frozen"
    return None


def _binding_fields(freeze: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method_commit": freeze.get("method_commit"),
        "config_sha256": freeze.get("config_sha256"),
        "protocol_sha256": freeze.get("protocol_sha256"),
    }


def freeze_round(
    repo: Path, round_dir: Path, config_path: Path, protocol_path: Path
) -> dict[str, Any]:
    """Freeze a clean repository commit and its tracked config/protocol inputs."""

    repository = _repo_path(repo)
    destination = Path(round_dir)
    if _git(repository, "status", "--porcelain"):
        raise StudyStateError("repository must be clean before freezing a round")
    if _current_state(destination) is not None or any(
        (destination / name).exists()
        for name in (
            FREEZE_NAME,
            FINAL_MANIFEST_NAME,
            MATERIALIZATION_NAME,
            EVALUATION_RECEIPT_NAME,
            SUPERSESSION_NAME,
        )
    ):
        raise StudyStateError("round already has a state record")

    config, config_relative = _input_path(repository, config_path, "config")
    protocol, protocol_relative = _input_path(repository, protocol_path, "protocol")
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
    }
    _atomic_write_json(destination / FREEZE_NAME, record)
    return record


def materialize_final(round_dir: Path, seed_count: int) -> dict[str, Any]:
    """Create the unseen final seed manifest after a round has been frozen."""

    destination = Path(round_dir)
    state = _current_state(destination)
    if state != "frozen":
        if state == "evaluated":
            raise StudyStateError("round was already evaluated")
        if state == "superseded":
            raise StudyStateError("round is superseded")
        raise StudyStateError("round must be frozen before final materialization")
    if type(seed_count) is not int or seed_count <= 0:
        raise StudyStateError("seed_count must be a positive integer")
    if (destination / FINAL_MANIFEST_NAME).exists():
        raise StudyStateError("final seed manifest already exists")

    freeze = _read_record(destination / FREEZE_NAME)
    seeds: list[int] = []
    used: set[int] = set()
    while len(seeds) < seed_count:
        seed = secrets.randbits(63)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)

    seed_manifest: dict[str, Any] = {
        "schema_version": 1,
        "round_id": freeze.get("round_id"),
        "generator_seeds": seeds,
    }
    _atomic_write_json(destination / FINAL_MANIFEST_NAME, seed_manifest)
    record: dict[str, Any] = {
        "schema_version": 1,
        "round_id": freeze.get("round_id"),
        "state": "materialized",
        "materialized_at": _utc_now(),
        "seed_manifest_path": FINAL_MANIFEST_NAME,
        "seed_manifest_sha256": canonical_sha256(seed_manifest),
        "generator_seeds": seeds,
        **_binding_fields(freeze),
    }
    _atomic_write_json(destination / MATERIALIZATION_NAME, record)
    return record


def assert_final_runnable(repo: Path, round_dir: Path) -> dict[str, Any]:
    """Verify that a materialized final round is still bound to its clean freeze."""

    destination = Path(round_dir)
    state = _current_state(destination)
    if state == "evaluated":
        raise StudyStateError("final round was already evaluated")
    if state == "superseded":
        raise StudyStateError("final round is superseded")
    if state != "materialized":
        raise StudyStateError("final round must be frozen and materialized before running")

    failure = "final evaluation requires a clean frozen commit and unchanged inputs"
    repository = _repo_path(repo)
    try:
        freeze = _read_record(destination / FREEZE_NAME)
        materialization = _read_record(destination / MATERIALIZATION_NAME)
        manifest = _read_record(destination / FINAL_MANIFEST_NAME)
        config = _recorded_path(repository, freeze.get("config_path"), "config")
        protocol = _recorded_path(repository, freeze.get("protocol_path"), "protocol")
        valid = (
            not _git(repository, "status", "--porcelain")
            and _git(repository, "rev-parse", "HEAD") == freeze.get("method_commit")
            and file_sha256(config) == freeze.get("config_sha256")
            and file_sha256(protocol) == freeze.get("protocol_sha256")
            and canonical_sha256(manifest)
            == materialization.get("seed_manifest_sha256")
            and manifest.get("round_id") == freeze.get("round_id")
        )
    except (OSError, StudyStateError, TypeError, ValueError):
        valid = False
    if not valid:
        raise StudyStateError(failure)

    return {
        "round_id": freeze.get("round_id"),
        "state": "materialized",
        **_binding_fields(freeze),
        "seed_manifest_sha256": materialization.get("seed_manifest_sha256"),
    }


def record_final_evaluation(
    round_dir: Path, result_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Write the exclusive receipt that permanently consumes a final round."""

    destination = Path(round_dir)
    state = _current_state(destination)
    if state == "evaluated":
        raise StudyStateError("final round was already evaluated")
    if state == "superseded":
        raise StudyStateError("final round is superseded")
    if state != "materialized":
        raise StudyStateError("final round must be materialized before evaluation")
    if not isinstance(result_manifest, Mapping):
        raise StudyStateError("result_manifest must be a JSON object")

    freeze = _read_record(destination / FREEZE_NAME)
    materialization = _read_record(destination / MATERIALIZATION_NAME)
    manifest = dict(result_manifest)
    result_hash = canonical_sha256(manifest)
    record: dict[str, Any] = {
        "schema_version": 1,
        "round_id": freeze.get("round_id"),
        "state": "evaluated",
        "evaluated_at": _utc_now(),
        "result_manifest": manifest,
        "result_manifest_sha256": result_hash,
        "seed_manifest_sha256": materialization.get("seed_manifest_sha256"),
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
    """Append a supersession record without removing any prior round evidence."""

    if not isinstance(reason, str) or not reason.strip():
        raise StudyStateError("supersession requires a nonempty reason")
    destination = Path(round_dir)
    state = _current_state(destination)
    if state is None:
        raise StudyStateError("round must be frozen before it can be superseded")
    if state == "superseded":
        raise StudyStateError("round is already superseded")

    freeze = _read_record(destination / FREEZE_NAME)
    record: dict[str, Any] = {
        "schema_version": 1,
        "round_id": freeze.get("round_id"),
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
    _atomic_write_json(destination / SUPERSESSION_NAME, record)
    return record


__all__ = [
    "StudyStateError",
    "assert_final_runnable",
    "freeze_round",
    "materialize_final",
    "record_final_evaluation",
    "supersede_round",
]
