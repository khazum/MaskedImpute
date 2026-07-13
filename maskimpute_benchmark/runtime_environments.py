"""Exact runtime-package inventories for publication competition environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Literal

from .protocol import canonical_sha256


_SAFE_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_PACKAGE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LOCK_SCHEMA = "maskimpute-runtime-environment-lock-v1"
_PYTHON_SCHEMA = "maskimpute-python-runtime-inventory-v1"
_R_SCHEMA = "maskimpute-r-runtime-inventory-v1"
_MAX_PROBE_BYTES = 16 * 1024 * 1024


class RuntimeEnvironmentError(ValueError):
    """Raised when a runtime environment does not match its frozen inventory."""


_PYTHON_PROBE = r"""
import importlib.metadata
import json
import re
import sys

def normalized(value):
    return re.sub(r"[-_.]+", "-", value).lower()

packages = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if not name or not version:
        raise RuntimeError("installed distribution lacks name or version")
    packages.append({"name": normalized(name), "version": str(version)})
packages.sort(key=lambda value: (value["name"], value["version"]))
if len({value["name"] for value in packages}) != len(packages):
    raise RuntimeError("installed distributions contain duplicate normalized names")
payload = {
    "schema": "maskimpute-python-runtime-inventory-v1",
    "interpreter": {
        "cache_tag": sys.implementation.cache_tag,
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:3]),
    },
    "packages": packages,
}
print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


_R_PROBE = r"""
packages <- installed.packages(noCache = TRUE)
rows <- data.frame(
  name = tolower(gsub("[._]+", "-", packages[, "Package"])),
  version = packages[, "Version"],
  stringsAsFactors = FALSE
)
rows <- rows[order(rows$name, rows$version), , drop = FALSE]
if (anyDuplicated(rows$name)) stop("installed packages contain duplicate normalized names")
cat("MASKIMPUTE-R-RUNTIME-INVENTORY-V1\n")
cat(paste(R.version$major, R.version$minor, R.version$platform, sep = "\t"), "\n", sep = "")
for (index in seq_len(nrow(rows))) {
  cat(rows$name[[index]], rows$version[[index]], sep = "\t")
  cat("\n")
}
"""


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
        raise RuntimeEnvironmentError(
            "runtime environment value is not canonical JSON"
        ) from error


def _reject_constant(value: str) -> None:
    raise RuntimeEnvironmentError(f"invalid JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeEnvironmentError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("runtime executable must be a pathlib.Path")
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except OSError as error:
        raise RuntimeEnvironmentError("runtime executable is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise RuntimeEnvironmentError("runtime executable is not an executable file")
    return resolved


def _probe_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    return environment


def _run_probe(command: list[str], name: str) -> bytes:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=Path("/"),
            env=_probe_environment(),
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeEnvironmentError(f"{name} runtime probe failed") from error
    if completed.returncode != 0:
        detail = hashlib.sha256(completed.stderr).hexdigest()
        raise RuntimeEnvironmentError(
            f"{name} runtime probe failed with stderr_sha256={detail}"
        )
    if len(completed.stdout) > _MAX_PROBE_BYTES:
        raise RuntimeEnvironmentError(f"{name} runtime probe output is too large")
    return completed.stdout


def _validate_packages(value: object, name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise RuntimeEnvironmentError(f"{name} packages must be a list")
    packages: list[dict[str, str]] = []
    names: set[str] = set()
    for package in value:
        if not isinstance(package, dict) or set(package) != {"name", "version"}:
            raise RuntimeEnvironmentError(f"{name} package entry is invalid")
        package_name = package.get("name")
        version = package.get("version")
        if (
            not isinstance(package_name, str)
            or not _PACKAGE_NAME.fullmatch(package_name)
            or not isinstance(version, str)
            or not version
        ):
            raise RuntimeEnvironmentError(f"{name} package identity is invalid")
        if package_name in names:
            raise RuntimeEnvironmentError(f"{name} package names are duplicated")
        names.add(package_name)
        packages.append({"name": package_name, "version": version})
    expected = sorted(packages, key=lambda item: (item["name"], item["version"]))
    if packages != expected:
        raise RuntimeEnvironmentError(f"{name} packages are not sorted")
    return packages


def _validate_inventory(
    value: object, kind: Literal["python", "r"]
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeEnvironmentError("runtime inventory must be an object")
    schema = _PYTHON_SCHEMA if kind == "python" else _R_SCHEMA
    if value.get("schema") != schema:
        raise RuntimeEnvironmentError("runtime inventory schema mismatch")
    expected_keys = {"schema", "interpreter", "packages", "executable_sha256"}
    if set(value) != expected_keys:
        raise RuntimeEnvironmentError("runtime inventory fields are invalid")
    executable_sha256 = value.get("executable_sha256")
    if not isinstance(executable_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", executable_sha256
    ):
        raise RuntimeEnvironmentError("runtime executable checksum is invalid")
    interpreter = value.get("interpreter")
    if not isinstance(interpreter, dict):
        raise RuntimeEnvironmentError("runtime interpreter identity is invalid")
    if kind == "python":
        if set(interpreter) != {"implementation", "version", "cache_tag"}:
            raise RuntimeEnvironmentError("Python interpreter fields are invalid")
        if (
            not isinstance(interpreter.get("implementation"), str)
            or not isinstance(interpreter.get("cache_tag"), str)
            or not isinstance(interpreter.get("version"), list)
            or len(interpreter["version"]) != 3
            or any(type(item) is not int or item < 0 for item in interpreter["version"])
        ):
            raise RuntimeEnvironmentError("Python interpreter identity is invalid")
    else:
        if set(interpreter) != {"major", "minor", "platform"} or any(
            not isinstance(interpreter.get(field), str) or not interpreter[field]
            for field in ("major", "minor", "platform")
        ):
            raise RuntimeEnvironmentError("R interpreter identity is invalid")
    packages = _validate_packages(value.get("packages"), kind)
    return {
        "schema": schema,
        "interpreter": dict(interpreter),
        "packages": packages,
        "executable_sha256": executable_sha256,
    }


def probe_python_environment(executable: Path) -> dict[str, object]:
    """Return a canonical inventory from one selected Python executable."""

    resolved = _executable(executable)
    raw = _run_probe([str(resolved), "-I", "-c", _PYTHON_PROBE], "Python")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except RuntimeEnvironmentError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeEnvironmentError("Python runtime probe returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeEnvironmentError("Python runtime probe did not return an object")
    value["executable_sha256"] = _file_sha256(resolved)
    return _validate_inventory(value, "python")


def probe_r_environment(executable: Path) -> dict[str, object]:
    """Return a canonical inventory from one selected Rscript executable."""

    resolved = _executable(executable)
    raw = _run_probe([str(resolved), "--vanilla", "-e", _R_PROBE], "R")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise RuntimeEnvironmentError("R runtime probe returned invalid text") from error
    if len(lines) < 2 or lines[0] != "MASKIMPUTE-R-RUNTIME-INVENTORY-V1":
        raise RuntimeEnvironmentError("R runtime probe header is invalid")
    identity = lines[1].split("\t")
    if len(identity) != 3:
        raise RuntimeEnvironmentError("R runtime interpreter identity is invalid")
    packages: list[dict[str, str]] = []
    for line in lines[2:]:
        fields = line.split("\t")
        if len(fields) != 2:
            raise RuntimeEnvironmentError("R runtime package row is invalid")
        packages.append({"name": fields[0], "version": fields[1]})
    return _validate_inventory(
        {
            "schema": _R_SCHEMA,
            "interpreter": {
                "major": identity[0],
                "minor": identity[1],
                "platform": identity[2],
            },
            "packages": packages,
            "executable_sha256": _file_sha256(resolved),
        },
        "r",
    )


def probe_runtime_environment(
    kind: Literal["python", "r"], executable: Path
) -> dict[str, object]:
    if kind == "python":
        return probe_python_environment(executable)
    if kind == "r":
        return probe_r_environment(executable)
    raise RuntimeEnvironmentError("runtime kind must be python or r")


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentEntry:
    environment_id: str
    kind: Literal["python", "r"]
    inventory_json: bytes
    inventory_sha256: str

    @property
    def inventory(self) -> dict[str, object]:
        value = json.loads(self.inventory_json.decode("utf-8"))
        assert isinstance(value, dict)
        return value


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentLock:
    path: Path
    file_sha256: str
    entries: tuple[RuntimeEnvironmentEntry, ...]

    def by_id(self, environment_id: str) -> RuntimeEnvironmentEntry:
        matches = [
            entry for entry in self.entries if entry.environment_id == environment_id
        ]
        if len(matches) != 1:
            raise RuntimeEnvironmentError(
                f"runtime environment {environment_id!r} is not uniquely locked"
            )
        return matches[0]


def build_runtime_environment_lock(
    environments: Mapping[str, tuple[Literal["python", "r"], Path]],
) -> dict[str, object]:
    """Probe selected executables and build a deterministic lock payload."""

    if not isinstance(environments, Mapping) or not environments:
        raise RuntimeEnvironmentError("at least one runtime environment is required")
    entries: list[dict[str, object]] = []
    for environment_id in sorted(environments):
        if not isinstance(environment_id, str) or not _SAFE_ID.fullmatch(environment_id):
            raise RuntimeEnvironmentError("runtime environment ID is invalid")
        declaration = environments[environment_id]
        if (
            not isinstance(declaration, tuple)
            or len(declaration) != 2
            or declaration[0] not in {"python", "r"}
            or not isinstance(declaration[1], Path)
        ):
            raise RuntimeEnvironmentError("runtime environment declaration is invalid")
        kind, executable = declaration
        inventory = probe_runtime_environment(kind, executable)
        entries.append(
            {
                "id": environment_id,
                "kind": kind,
                "inventory": inventory,
                "inventory_sha256": canonical_sha256(inventory),
            }
        )
    return {"schema": _LOCK_SCHEMA, "environments": entries}


def load_runtime_environment_lock(path: Path) -> RuntimeEnvironmentLock:
    """Load a secure canonical runtime lock without trusting its hashes."""

    if not isinstance(path, Path):
        raise TypeError("runtime lock path must be a pathlib.Path")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o002
        ):
            raise RuntimeEnvironmentError(
                "runtime lock must be a secure unique regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError("runtime lock is unavailable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except RuntimeEnvironmentError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeEnvironmentError("runtime lock is invalid JSON") from error
    if raw != _canonical_bytes(value) + b"\n":
        raise RuntimeEnvironmentError("runtime lock must be canonical JSON")
    if not isinstance(value, dict) or set(value) != {"schema", "environments"}:
        raise RuntimeEnvironmentError("runtime lock fields are invalid")
    if value.get("schema") != _LOCK_SCHEMA:
        raise RuntimeEnvironmentError("runtime lock schema is invalid")
    raw_entries = value.get("environments")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise RuntimeEnvironmentError("runtime lock environments are invalid")
    entries: list[RuntimeEnvironmentEntry] = []
    identifiers: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "id",
            "kind",
            "inventory",
            "inventory_sha256",
        }:
            raise RuntimeEnvironmentError("runtime lock entry fields are invalid")
        environment_id = raw_entry.get("id")
        kind = raw_entry.get("kind")
        if (
            not isinstance(environment_id, str)
            or not _SAFE_ID.fullmatch(environment_id)
            or kind not in {"python", "r"}
        ):
            raise RuntimeEnvironmentError("runtime lock entry identity is invalid")
        if environment_id in identifiers:
            raise RuntimeEnvironmentError("duplicate environment ID in runtime lock")
        identifiers.add(environment_id)
        inventory = _validate_inventory(raw_entry.get("inventory"), kind)
        inventory_sha256 = raw_entry.get("inventory_sha256")
        if (
            not isinstance(inventory_sha256, str)
            or inventory_sha256 != canonical_sha256(inventory)
        ):
            raise RuntimeEnvironmentError("runtime inventory checksum mismatch")
        entries.append(
            RuntimeEnvironmentEntry(
                environment_id=environment_id,
                kind=kind,
                inventory_json=_canonical_bytes(inventory),
                inventory_sha256=inventory_sha256,
            )
        )
    if [entry.environment_id for entry in entries] != sorted(identifiers):
        raise RuntimeEnvironmentError("runtime lock environments are not sorted")
    return RuntimeEnvironmentLock(
        path=path.resolve(strict=True),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        entries=tuple(entries),
    )


def validate_runtime_environment_lock(
    lock: RuntimeEnvironmentLock,
    environments: Mapping[str, tuple[Literal["python", "r"], Path]],
) -> dict[str, object]:
    """Independently probe every runtime and compare it with a loaded lock."""

    if not isinstance(lock, RuntimeEnvironmentLock):
        raise TypeError("lock must be a RuntimeEnvironmentLock")
    if not isinstance(environments, Mapping):
        raise TypeError("environments must be a mapping")
    expected_ids = {entry.environment_id for entry in lock.entries}
    observed_ids = set(environments)
    if observed_ids != expected_ids:
        raise RuntimeEnvironmentError("runtime IDs mismatch frozen lock")
    receipts: list[tuple[str, str]] = []
    for environment_id in sorted(observed_ids):
        declaration = environments[environment_id]
        if (
            not isinstance(declaration, tuple)
            or len(declaration) != 2
            or declaration[0] not in {"python", "r"}
            or not isinstance(declaration[1], Path)
        ):
            raise RuntimeEnvironmentError("runtime environment declaration is invalid")
        entry = lock.by_id(environment_id)
        kind, executable = declaration
        if kind != entry.kind:
            raise RuntimeEnvironmentError(
                f"runtime kind mismatch for {environment_id}"
            )
        inventory = probe_runtime_environment(kind, executable)
        if inventory != entry.inventory:
            raise RuntimeEnvironmentError(
                f"runtime inventory mismatch for {environment_id}"
            )
        receipts.append((environment_id, entry.inventory_sha256))
    return {
        "lock_file_sha256": lock.file_sha256,
        "environment_inventory_sha256s": tuple(receipts),
    }


__all__ = [
    "RuntimeEnvironmentEntry",
    "RuntimeEnvironmentError",
    "RuntimeEnvironmentLock",
    "build_runtime_environment_lock",
    "load_runtime_environment_lock",
    "probe_python_environment",
    "probe_r_environment",
    "probe_runtime_environment",
    "validate_runtime_environment_lock",
]
