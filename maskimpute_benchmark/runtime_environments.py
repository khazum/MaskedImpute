"""Exact runtime-package inventories for publication competition environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
import hashlib
try:
    import importlib.metadata as metadata
except ImportError:
    import importlib_metadata as metadata
import json
import os
from pathlib import Path
import re
import struct
import sys

def normalized(value):
    return re.sub(r"[-_.]+", "-", value).lower()

def is_beneath(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

packages = []
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    version = distribution.version
    if not name or not version:
        raise RuntimeError("installed distribution lacks name or version")
    files = sorted(
        (str(value), value) for value in (distribution.files or ())
    )
    digest = hashlib.sha256()
    digest.update(b"maskimpute-python-distribution-content-v1\0")
    file_count = 0
    for logical, relative in files:
        encoded = logical.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        path = distribution.locate_file(relative)
        try:
            if path.is_symlink():
                target = os.readlink(path).encode("utf-8")
                digest.update(b"L")
                digest.update(struct.pack("<Q", len(target)))
                digest.update(target)
                resolved = path.resolve(strict=True)
                if resolved.is_file():
                    payload = resolved.read_bytes()
                    digest.update(struct.pack("<Q", len(payload)))
                    digest.update(payload)
            elif path.is_file():
                payload = path.read_bytes()
                digest.update(b"F")
                digest.update(struct.pack("<Q", len(payload)))
                digest.update(payload)
            else:
                digest.update(b"M")
        except OSError as error:
            raise RuntimeError(f"cannot hash installed distribution file {logical}") from error
        file_count += 1
    distribution_path = Path(distribution._path).resolve(strict=True)
    candidates = []
    for index, entry in enumerate(sys.path):
        try:
            distribution_path.relative_to(Path(entry).resolve(strict=True))
        except (OSError, ValueError):
            continue
        candidates.append(index)
    if not candidates:
        raise RuntimeError("installed distribution is outside interpreter search path")
    packages.append({
        "content_sha256": digest.hexdigest(),
        "file_count": file_count,
        "name": normalized(name),
        "precedence": min(candidates),
        "version": str(version),
    })
packages.sort(key=lambda value: (value["name"], value["precedence"], value["version"]))
if len({(value["name"], value["precedence"], value["version"], value["content_sha256"]) for value in packages}) != len(packages):
    raise RuntimeError("installed distributions contain duplicate identities")
root_candidates = [
    ("base-prefix-lib", Path(sys.base_prefix) / "lib"),
    ("prefix-lib", Path(sys.prefix) / "lib"),
]
covered = [path.resolve(strict=True) for _role, path in root_candidates if path.exists()]
for index, value in enumerate(sys.path):
    if not value or not Path(value).exists():
        continue
    selected = Path(value).resolve(strict=True)
    if any(selected == root or is_beneath(selected, root) for root in covered):
        continue
    root_candidates.append((f"search-path-{index:03d}", Path(value)))
observed_roots = set()
runtime_roots = []
for role, path in root_candidates:
    if not path.exists():
        continue
    resolved = path.resolve(strict=True)
    if resolved in observed_roots:
        continue
    observed_roots.add(resolved)
    runtime_roots.append({
        "kind": "directory" if path.is_dir() else "file",
        "path": str(path.absolute()),
        "role": role,
    })
payload = {
    "schema": "maskimpute-python-runtime-inventory-v1",
    "interpreter": {
        "cache_tag": sys.implementation.cache_tag,
        "implementation": sys.implementation.name,
        "is_virtual_environment": sys.prefix != sys.base_prefix,
        "version": list(sys.version_info[:3]),
    },
    "packages": packages,
    "_runtime_root_paths": sorted(runtime_roots, key=lambda value: value["role"]),
}
print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


_PYTHON_ROOT_PROBE = r"""
import json
from pathlib import Path
import sys

def is_beneath(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

root_candidates = [
    ("base-prefix-lib", Path(sys.base_prefix) / "lib"),
    ("prefix-lib", Path(sys.prefix) / "lib"),
]
covered = [path.resolve(strict=True) for _role, path in root_candidates if path.exists()]
for index, value in enumerate(sys.path):
    if not value or not Path(value).exists():
        continue
    selected = Path(value).resolve(strict=True)
    if any(selected == root or is_beneath(selected, root) for root in covered):
        continue
    root_candidates.append((f"search-path-{index:03d}", Path(value)))
observed = set()
roots = []
for role, path in root_candidates:
    if not path.exists():
        continue
    resolved = path.resolve(strict=True)
    if resolved in observed:
        continue
    observed.add(resolved)
    roots.append({
        "kind": "directory" if path.is_dir() else "file",
        "path": str(path.absolute()),
        "role": role,
    })
print(json.dumps(sorted(roots, key=lambda value: value["role"]), separators=(",", ":"), sort_keys=True))
"""


_R_PROBE = r"""
packages <- installed.packages(noCache = TRUE)
library_paths <- normalizePath(.libPaths(), mustWork = TRUE)
rows <- data.frame(
  package = packages[, "Package"],
  name = tolower(gsub("[._]+", "-", packages[, "Package"])),
  version = packages[, "Version"],
  precedence = match(normalizePath(packages[, "LibPath"], mustWork = TRUE),
                     library_paths) - 1L,
  path = file.path(packages[, "LibPath"], packages[, "Package"]),
  stringsAsFactors = FALSE
)
if (any(is.na(rows$precedence))) stop("installed package escaped R library paths")
rows <- rows[order(rows$name, rows$precedence, rows$version), , drop = FALSE]
if (anyDuplicated(rows[, c("name", "version", "precedence", "path")])) {
  stop("installed packages contain duplicate identities")
}
cat("MASKIMPUTE-R-RUNTIME-INVENTORY-V1\n")
cat(paste(R.version$major, R.version$minor, R.version$platform,
          length(.libPaths()), sep = "\t"), "\n", sep = "")
for (index in seq_along(library_paths)) {
  cat("LIB", index - 1L, library_paths[[index]], sep = "\t")
  cat("\n")
}
for (index in seq_len(nrow(rows))) {
  package_path <- normalizePath(rows$path[[index]], mustWork = TRUE)
  cat("PKG", rows$name[[index]], rows$version[[index]], rows$precedence[[index]],
      package_path, sep = "\t")
  cat("\n")
}
"""


_R_ROOT_PROBE = r"""
paths <- normalizePath(.libPaths(), mustWork = TRUE)
for (index in seq_along(paths)) {
  cat(index - 1L, paths[[index]], sep = "\t")
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


def _directory_content_sha256(path: Path) -> tuple[str, int]:
    cache: dict[Path, tuple[str, int]] = {}
    observed_paths: dict[Path, tuple[int, int, int, int, int, int, int]] = {}

    def remember(item: Path, metadata: os.stat_result) -> None:
        identity = _stat_identity(metadata)
        previous = observed_paths.setdefault(item, identity)
        if previous != identity:
            raise RuntimeEnvironmentError(
                "runtime content path changed during traversal"
            )

    def read_regular(item: Path) -> tuple[bytes, os.stat_result]:
        try:
            before = item.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeEnvironmentError(
                    "runtime root contains a non-regular file"
                )
            remember(item, before)
            descriptor = os.open(
                item,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            after_path = item.lstat()
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime root changed while hashing a file"
            ) from error
        identity = _stat_identity(before)
        if any(
            _stat_identity(value) != identity
            for value in (opened, after, after_path)
        ):
            raise RuntimeEnvironmentError("runtime root file changed while hashing")
        return b"".join(chunks), before

    def hash_directory(directory: Path, ancestry: frozenset[Path]) -> tuple[str, int]:
        resolved = directory.resolve(strict=True)
        if resolved in ancestry:
            raise RuntimeEnvironmentError("runtime root contains a directory cycle")
        if resolved in cache:
            return cache[resolved]
        try:
            before = directory.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise RuntimeEnvironmentError(
                    "runtime package path is not a secure directory"
                )
            remember(directory, before)
            children = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime package directory is unavailable"
            ) from error
        digest = hashlib.sha256()
        digest.update(b"maskimpute-runtime-directory-v2\0")
        digest.update(stat.S_IMODE(before.st_mode).to_bytes(4, "little"))
        count = 1
        for child_entry in children:
            child = Path(child_entry.path)
            encoded = os.fsencode(child_entry.name)
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            try:
                child_before = child.lstat()
                remember(child, child_before)
                if stat.S_ISLNK(child_before.st_mode):
                    target_text = os.fsencode(os.readlink(child))
                    target = child.resolve(strict=True)
                    digest.update(b"L")
                    digest.update(len(target_text).to_bytes(8, "little"))
                    digest.update(target_text)
                    if target.is_file():
                        remember(target, target.stat())
                        payload, _target_metadata = read_regular(target)
                        digest.update(b"F")
                        digest.update(len(payload).to_bytes(8, "little"))
                        digest.update(payload)
                        nested_count = 1
                    elif target.is_dir():
                        nested_sha, nested_count = hash_directory(
                            target, ancestry.union({resolved})
                        )
                        digest.update(b"D")
                        digest.update(bytes.fromhex(nested_sha))
                    else:
                        raise RuntimeEnvironmentError(
                            "runtime root symlink target is not regular"
                        )
                    child_after = child.lstat()
                    if _stat_identity(child_before) != _stat_identity(child_after):
                        raise RuntimeEnvironmentError(
                            "runtime root symlink changed while hashing"
                        )
                    count += 1 + nested_count
                elif stat.S_ISDIR(child_before.st_mode):
                    nested_sha, nested_count = hash_directory(
                        child, ancestry.union({resolved})
                    )
                    digest.update(b"D")
                    digest.update(bytes.fromhex(nested_sha))
                    count += nested_count
                elif stat.S_ISREG(child_before.st_mode):
                    payload, metadata = read_regular(child)
                    digest.update(b"F")
                    digest.update(stat.S_IMODE(metadata.st_mode).to_bytes(4, "little"))
                    digest.update(len(payload).to_bytes(8, "little"))
                    digest.update(payload)
                    count += 1
                else:
                    raise RuntimeEnvironmentError(
                        "runtime root contains a special filesystem entry"
                    )
            except RuntimeEnvironmentError:
                raise
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "runtime root changed while enumerating"
                ) from error
        try:
            after = directory.lstat()
            after_children = sorted(
                (entry.name for entry in os.scandir(directory)), key=os.fsencode
            )
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime root changed after enumeration"
            ) from error
        if (
            _stat_identity(before) != _stat_identity(after)
            or [entry.name for entry in children] != after_children
        ):
            raise RuntimeEnvironmentError("runtime root changed while hashing")
        result = (digest.hexdigest(), count)
        cache[resolved] = result
        return result

    result = hash_directory(path, frozenset())
    try:
        for observed_path, expected in observed_paths.items():
            if _stat_identity(observed_path.lstat()) != expected:
                raise RuntimeEnvironmentError(
                    "runtime content path changed after it was visited"
                )
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError(
            "runtime content path disappeared after it was visited"
        ) from error
    return result


def _runtime_file_content_sha256(path: Path) -> tuple[str, int]:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            target_text = os.fsencode(os.readlink(path))
            target = path.resolve(strict=True)
            digest = hashlib.sha256(
                b"maskimpute-runtime-root-symlink-v1\0" + target_text
            )
            if target.is_file():
                digest.update(target.read_bytes())
            elif target.is_dir():
                nested_sha, nested_count = _directory_content_sha256(target)
                digest.update(bytes.fromhex(nested_sha))
                return digest.hexdigest(), nested_count + 1
            else:
                raise RuntimeEnvironmentError("runtime root symlink target is invalid")
        elif stat.S_ISREG(before.st_mode):
            digest = hashlib.sha256(
                b"maskimpute-runtime-root-file-v1\0" + path.read_bytes()
            )
        else:
            raise RuntimeEnvironmentError("runtime root file is invalid")
        after = path.lstat()
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError("runtime root file changed while hashing") from error
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeEnvironmentError("runtime root file changed while hashing")
    return digest.hexdigest(), 1


def _validated_runtime_root_paths(
    raw_roots: object,
) -> tuple[tuple[str, Literal["directory", "file"], Path], ...]:
    if not isinstance(raw_roots, list) or not raw_roots:
        raise RuntimeEnvironmentError("runtime probe did not expose its roots")
    result: list[tuple[str, Literal["directory", "file"], Path]] = []
    roles: set[str] = set()
    for raw_root in raw_roots:
        if not isinstance(raw_root, dict) or set(raw_root) != {"role", "kind", "path"}:
            raise RuntimeEnvironmentError("runtime probe root declaration is invalid")
        role = raw_root.get("role")
        kind = raw_root.get("kind")
        raw_path = raw_root.get("path")
        if (
            not isinstance(role, str)
            or re.fullmatch(r"[a-z][a-z0-9-]*", role) is None
            or role in roles
            or kind not in {"directory", "file"}
            or not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
        ):
            raise RuntimeEnvironmentError("runtime probe root identity is invalid")
        roles.add(role)
        path = Path(raw_path)
        try:
            for parent in path.parents:
                if stat.S_ISLNK(parent.lstat().st_mode):
                    raise RuntimeEnvironmentError(
                        "runtime root parent directory must not be a symlink"
                    )
            path.resolve(strict=True)
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError("runtime root is unavailable") from error
        if kind == "directory" and not path.is_dir():
            raise RuntimeEnvironmentError("runtime root kind is inconsistent")
        if kind == "file" and not path.is_file():
            raise RuntimeEnvironmentError("runtime root kind is inconsistent")
        result.append((role, kind, path))
    return tuple(sorted(result, key=lambda item: item[0]))


def _with_native_dependency_roots(
    raw_roots: object, executable_target: Path
) -> tuple[list[dict[str, str]], str]:
    validated = _validated_runtime_root_paths(raw_roots)
    covered = tuple(path.resolve(strict=True) for _role, _kind, path in validated)
    candidates: set[Path] = {executable_target.resolve(strict=True)}
    nvidia_smi = Path("/usr/bin/nvidia-smi")
    if nvidia_smi.is_file():
        candidates.add(nvidia_smi.resolve(strict=True))
    for _role, kind, root in validated:
        if kind == "file":
            candidates.add(root.resolve(strict=True))
            continue
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            directory_names.sort()
            file_names.sort()
            current_path = Path(current)
            for name in file_names:
                if ".so" not in name:
                    continue
                path = current_path / name
                try:
                    target = path.resolve(strict=True)
                    if target.is_file():
                        with target.open("rb") as stream:
                            if stream.read(4) == b"\x7fELF":
                                candidates.add(target)
                except OSError as error:
                    raise RuntimeEnvironmentError(
                        "native runtime candidate changed during discovery"
                    ) from error
    dependencies: set[Path] = set()
    linkage_digest = hashlib.sha256(
        b"maskimpute-native-linkage-resolution-v1\0"
    )
    candidate_list = sorted(candidates, key=lambda path: os.fsencode(path.as_posix()))
    for offset in range(0, len(candidate_list), 64):
        batch = candidate_list[offset : offset + 64]
        try:
            completed = subprocess.run(
                ["/usr/bin/ldd", *(path.as_posix() for path in batch)],
                check=False,
                capture_output=True,
                env=_probe_environment(),
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeEnvironmentError(
                "native runtime dependency discovery failed"
            ) from error
        try:
            text = completed.stdout.decode("utf-8", errors="strict")
            stderr = completed.stderr.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise RuntimeEnvironmentError(
                "native runtime dependency output is not UTF-8"
            ) from error
        normalized = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
        normalized_stderr = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", stderr)
        linkage_digest.update(completed.returncode.to_bytes(4, "little", signed=True))
        linkage_digest.update(normalized.encode("utf-8"))
        linkage_digest.update(normalized_stderr.encode("utf-8"))
        for match in re.finditer(r"(?:=>\s*)?(/[^\s()]+)", text):
            if match.group(1).endswith(":"):
                continue
            try:
                dependency = Path(match.group(1)).resolve(strict=True)
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "native runtime dependency disappeared"
                ) from error
            if not dependency.is_file():
                continue
            inside = False
            for root in covered:
                try:
                    dependency.relative_to(root)
                    inside = True
                    break
                except ValueError:
                    continue
            if not inside:
                dependencies.add(dependency)
    result = [
        {"role": role, "kind": kind, "path": path.absolute().as_posix()}
        for role, kind, path in validated
    ]
    driver_version = Path("/proc/driver/nvidia/version")
    if driver_version.is_file():
        result.append(
            {
                "role": "gpu-driver-version",
                "kind": "file",
                "path": driver_version.as_posix(),
            }
        )
    if nvidia_smi.is_file():
        result.append(
            {
                "role": "nvidia-smi-executable",
                "kind": "file",
                "path": nvidia_smi.resolve(strict=True).as_posix(),
            }
        )
    result.extend(
        {
            "role": f"native-dependency-{index:03d}",
            "kind": "file",
            "path": path.as_posix(),
        }
        for index, path in enumerate(
            sorted(dependencies, key=lambda value: os.fsencode(value.as_posix()))
        )
    )
    return sorted(result, key=lambda item: item["role"]), linkage_digest.hexdigest()


def _runtime_root_inventory(raw_roots: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    cache: dict[tuple[str, Path], tuple[str, int]] = {}
    for role, kind, path in _validated_runtime_root_paths(raw_roots):
        resolved = path.resolve(strict=True)
        cache_key = (kind, resolved)
        if cache_key not in cache:
            cache[cache_key] = (
                _directory_content_sha256(path)
                if kind == "directory"
                else _runtime_file_content_sha256(path)
            )
        content_sha256, entry_count = cache[cache_key]
        result.append(
            {
                "role": role,
                "kind": kind,
                "content_sha256": content_sha256,
                "entry_count": entry_count,
            }
        )
    return result


@dataclass(frozen=True, slots=True)
class _ExecutableIdentity:
    invocation: Path
    target: Path
    launcher_kind: Literal["regular", "symlink"]
    launcher_sha256: str
    invocation_state: tuple[int, int, int, int, int, int, int]
    target_state: tuple[int, int, int, int, int, int, int]


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _executable(path: Path) -> _ExecutableIdentity:
    if not isinstance(path, Path):
        raise TypeError("runtime executable must be a pathlib.Path")
    try:
        invocation = path.absolute()
        for parent in invocation.parents:
            metadata = parent.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeEnvironmentError(
                    "runtime executable parent directory must not be a symlink"
                )
        invocation_metadata = invocation.lstat()
        target = invocation.resolve(strict=True)
        target_metadata = target.stat()
    except OSError as error:
        raise RuntimeEnvironmentError("runtime executable is unavailable") from error
    if not stat.S_ISREG(target_metadata.st_mode) or not os.access(invocation, os.X_OK):
        raise RuntimeEnvironmentError("runtime executable is not an executable file")
    if stat.S_ISLNK(invocation_metadata.st_mode):
        try:
            target_text = os.readlink(invocation).encode("utf-8")
        except (OSError, UnicodeError) as error:
            raise RuntimeEnvironmentError(
                "runtime executable launcher cannot be read"
            ) from error
        launcher_kind: Literal["regular", "symlink"] = "symlink"
        launcher_sha256 = hashlib.sha256(
            b"maskimpute-runtime-launcher-symlink-v1\0" + target_text
        ).hexdigest()
    elif stat.S_ISREG(invocation_metadata.st_mode):
        launcher_kind = "regular"
        launcher_sha256 = _file_sha256(invocation)
    else:
        raise RuntimeEnvironmentError("runtime executable launcher is invalid")
    return _ExecutableIdentity(
        invocation=invocation,
        target=target,
        launcher_kind=launcher_kind,
        launcher_sha256=launcher_sha256,
        invocation_state=_stat_identity(invocation_metadata),
        target_state=_stat_identity(target_metadata),
    )


def _revalidate_executable(executable: _ExecutableIdentity) -> None:
    try:
        invocation = executable.invocation.lstat()
        target = executable.target.stat()
    except OSError as error:
        raise RuntimeEnvironmentError(
            "runtime executable changed during inventory"
        ) from error
    if (
        _stat_identity(invocation) != executable.invocation_state
        or _stat_identity(target) != executable.target_state
        or executable.invocation.resolve(strict=True) != executable.target
    ):
        raise RuntimeEnvironmentError("runtime executable changed during inventory")


def _runtime_root_identity_sha256(path: Path) -> str:
    cache: dict[Path, str] = {}
    observed_paths: dict[Path, tuple[int, int, int, int, int, int, int]] = {}

    def remember(item: Path, metadata: os.stat_result) -> None:
        identity = _stat_identity(metadata)
        previous = observed_paths.setdefault(item, identity)
        if previous != identity:
            raise RuntimeEnvironmentError(
                "runtime identity path changed during traversal"
            )

    def hash_directory(directory: Path, ancestry: frozenset[Path]) -> str:
        resolved = directory.resolve(strict=True)
        if resolved in ancestry:
            raise RuntimeEnvironmentError("runtime identity contains a directory cycle")
        if resolved in cache:
            return cache[resolved]
        try:
            before = directory.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise RuntimeEnvironmentError("runtime identity root is not a directory")
            remember(directory, before)
            children = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime identity directory is unavailable"
            ) from error
        digest = hashlib.sha256()
        digest.update(b"maskimpute-runtime-metadata-identity-v1\0")
        digest.update(repr(_stat_identity(before)).encode("ascii"))
        for entry in children:
            child = Path(entry.path)
            encoded = os.fsencode(entry.name)
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            try:
                child_before = child.lstat()
                remember(child, child_before)
                digest.update(repr(_stat_identity(child_before)).encode("ascii"))
                if stat.S_ISLNK(child_before.st_mode):
                    target_text = os.fsencode(os.readlink(child))
                    target = child.resolve(strict=True)
                    digest.update(b"L" + len(target_text).to_bytes(8, "little"))
                    digest.update(target_text)
                    if target.is_dir():
                        nested = hash_directory(target, ancestry.union({resolved}))
                        digest.update(bytes.fromhex(nested))
                    elif target.is_file():
                        target_metadata = target.stat()
                        remember(target, target_metadata)
                        digest.update(repr(_stat_identity(target_metadata)).encode("ascii"))
                    else:
                        raise RuntimeEnvironmentError(
                            "runtime identity symlink target is invalid"
                        )
                    if _stat_identity(child.lstat()) != _stat_identity(child_before):
                        raise RuntimeEnvironmentError(
                            "runtime identity symlink changed during traversal"
                        )
                elif stat.S_ISDIR(child_before.st_mode):
                    nested = hash_directory(child, ancestry.union({resolved}))
                    digest.update(bytes.fromhex(nested))
                elif not stat.S_ISREG(child_before.st_mode):
                    raise RuntimeEnvironmentError(
                        "runtime identity contains a special filesystem entry"
                    )
            except RuntimeEnvironmentError:
                raise
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "runtime identity changed during traversal"
                ) from error
        try:
            after = directory.lstat()
            names_after = sorted(
                (entry.name for entry in os.scandir(directory)), key=os.fsencode
            )
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime identity changed after traversal"
            ) from error
        if (
            _stat_identity(before) != _stat_identity(after)
            or [entry.name for entry in children] != names_after
        ):
            raise RuntimeEnvironmentError("runtime identity changed during traversal")
        value = digest.hexdigest()
        cache[resolved] = value
        return value

    if path.is_dir() and not path.is_symlink():
        result = hash_directory(path, frozenset())
    else:
        try:
            metadata = path.lstat()
            remember(path, metadata)
            digest = hashlib.sha256(b"maskimpute-runtime-metadata-file-v1\0")
            digest.update(repr(_stat_identity(metadata)).encode("ascii"))
            if stat.S_ISLNK(metadata.st_mode):
                target = path.resolve(strict=True)
                target_metadata = target.stat()
                remember(target, target_metadata)
                digest.update(os.fsencode(os.readlink(path)))
                digest.update(repr(_stat_identity(target_metadata)).encode("ascii"))
            elif not stat.S_ISREG(metadata.st_mode):
                raise RuntimeEnvironmentError("runtime identity file is invalid")
            after = path.lstat()
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError("runtime identity file is unavailable") from error
        if _stat_identity(metadata) != _stat_identity(after):
            raise RuntimeEnvironmentError("runtime identity file changed")
        result = digest.hexdigest()
    try:
        for observed_path, expected in observed_paths.items():
            if _stat_identity(observed_path.lstat()) != expected:
                raise RuntimeEnvironmentError(
                    "runtime identity path changed after it was visited"
                )
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError(
            "runtime identity path disappeared after it was visited"
        ) from error
    return result


def _python_runtime_root_paths(executable: _ExecutableIdentity) -> object:
    raw = _run_probe(
        [str(executable.invocation), "-I", "-c", _PYTHON_ROOT_PROBE], "Python"
    )
    try:
        roots = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        return _with_native_dependency_roots(roots, executable.target)
    except RuntimeEnvironmentError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeEnvironmentError("Python runtime roots are invalid") from error


def _r_runtime_root_paths(
    executable: _ExecutableIdentity, library_paths: tuple[Path, ...]
) -> object:
    raw = _run_probe(
        [str(executable.invocation), "--vanilla", "-e", _R_ROOT_PROBE],
        "R",
        r_library_paths=library_paths,
    )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise RuntimeEnvironmentError("R runtime roots are invalid") from error
    roots: list[dict[str, str]] = []
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0].isdigit():
            raise RuntimeEnvironmentError("R runtime root row is invalid")
        roots.append(
            {
                "role": f"library-path-{int(fields[0]):03d}",
                "kind": "directory",
                "path": fields[1],
            }
        )
    return _with_native_dependency_roots(roots, executable.target)


def runtime_environment_identity_sha256(
    kind: Literal["python", "r"],
    executable: Path,
    *,
    r_library_paths: tuple[Path, ...] = (),
) -> str:
    """Return a fast ctime/inode tree identity for pre/post row revalidation."""

    selected = _executable(executable)
    libraries = _r_library_paths(r_library_paths) if kind == "r" else ()
    if kind == "python":
        raw_roots, native_linkage_sha256 = _python_runtime_root_paths(selected)
    elif kind == "r":
        raw_roots, native_linkage_sha256 = _r_runtime_root_paths(selected, libraries)
    else:
        raise RuntimeEnvironmentError("runtime kind must be python or r")
    roots = [
        {
            "role": role,
            "kind": root_kind,
            "identity_sha256": _runtime_root_identity_sha256(path),
        }
        for role, root_kind, path in _validated_runtime_root_paths(raw_roots)
    ]
    _revalidate_executable(selected)
    return canonical_sha256(
        {
            "schema": "maskimpute-runtime-metadata-snapshot-v1",
            "kind": kind,
            "launcher_kind": selected.launcher_kind,
            "launcher_sha256": selected.launcher_sha256,
            "invocation_state": selected.invocation_state,
            "target_state": selected.target_state,
            "native_linkage_sha256": native_linkage_sha256,
            "roots": roots,
        }
    )


def _probe_environment(*, r_library_paths: tuple[Path, ...] = ()) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    if r_library_paths:
        joined = os.pathsep.join(path.as_posix() for path in r_library_paths)
        environment.update(
            {"R_LIBS": joined, "R_LIBS_SITE": joined, "R_LIBS_USER": joined}
        )
    return environment


def _run_probe(
    command: list[str], name: str, *, r_library_paths: tuple[Path, ...] = ()
) -> bytes:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=Path("/"),
            env=_probe_environment(r_library_paths=r_library_paths),
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


def _validate_packages(value: object, name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise RuntimeEnvironmentError(f"{name} packages must be a list")
    packages: list[dict[str, object]] = []
    identities: set[tuple[str, int, str, str]] = set()
    for package in value:
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "content_sha256",
            "file_count",
            "precedence",
        }:
            raise RuntimeEnvironmentError(f"{name} package entry is invalid")
        package_name = package.get("name")
        version = package.get("version")
        content_sha256 = package.get("content_sha256")
        file_count = package.get("file_count")
        precedence = package.get("precedence")
        if (
            not isinstance(package_name, str)
            or not _PACKAGE_NAME.fullmatch(package_name)
            or not isinstance(version, str)
            or not version
            or not isinstance(content_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
            or type(file_count) is not int
            or file_count < 0
            or type(precedence) is not int
            or precedence < 0
        ):
            raise RuntimeEnvironmentError(f"{name} package identity is invalid")
        identity = (package_name, precedence, version, content_sha256)
        if identity in identities:
            raise RuntimeEnvironmentError(f"{name} package identities are duplicated")
        identities.add(identity)
        packages.append(
            {
                "name": package_name,
                "version": version,
                "content_sha256": content_sha256,
                "file_count": file_count,
                "precedence": precedence,
            }
        )
    expected = sorted(
        packages,
        key=lambda item: (item["name"], item["precedence"], item["version"]),
    )
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
    expected_keys = {
        "schema",
        "interpreter",
        "packages",
        "executable_sha256",
        "launcher",
        "runtime_roots",
        "native_linkage_sha256",
    }
    if set(value) != expected_keys:
        raise RuntimeEnvironmentError("runtime inventory fields are invalid")
    executable_sha256 = value.get("executable_sha256")
    if not isinstance(executable_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", executable_sha256
    ):
        raise RuntimeEnvironmentError("runtime executable checksum is invalid")
    native_linkage_sha256 = value.get("native_linkage_sha256")
    if not isinstance(native_linkage_sha256, str) or re.fullmatch(
        r"[0-9a-f]{64}", native_linkage_sha256
    ) is None:
        raise RuntimeEnvironmentError("native runtime linkage checksum is invalid")
    launcher = value.get("launcher")
    if (
        not isinstance(launcher, dict)
        or set(launcher) != {"kind", "sha256"}
        or launcher.get("kind") not in {"regular", "symlink"}
        or not isinstance(launcher.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", launcher["sha256"]) is None
    ):
        raise RuntimeEnvironmentError("runtime launcher identity is invalid")
    interpreter = value.get("interpreter")
    if not isinstance(interpreter, dict):
        raise RuntimeEnvironmentError("runtime interpreter identity is invalid")
    if kind == "python":
        if set(interpreter) != {
            "implementation",
            "version",
            "cache_tag",
            "is_virtual_environment",
        }:
            raise RuntimeEnvironmentError("Python interpreter fields are invalid")
        if (
            not isinstance(interpreter.get("implementation"), str)
            or not isinstance(interpreter.get("cache_tag"), str)
            or type(interpreter.get("is_virtual_environment")) is not bool
            or not isinstance(interpreter.get("version"), list)
            or len(interpreter["version"]) != 3
            or any(type(item) is not int or item < 0 for item in interpreter["version"])
        ):
            raise RuntimeEnvironmentError("Python interpreter identity is invalid")
    else:
        if set(interpreter) != {
            "major",
            "minor",
            "platform",
            "library_path_count",
        } or any(
            not isinstance(interpreter.get(field), str) or not interpreter[field]
            for field in ("major", "minor", "platform")
        ) or (
            type(interpreter.get("library_path_count")) is not int
            or interpreter["library_path_count"] < 1
        ):
            raise RuntimeEnvironmentError("R interpreter identity is invalid")
    packages = _validate_packages(value.get("packages"), kind)
    raw_roots = value.get("runtime_roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise RuntimeEnvironmentError("runtime root inventory is invalid")
    runtime_roots: list[dict[str, object]] = []
    roles: set[str] = set()
    for raw_root in raw_roots:
        if not isinstance(raw_root, dict) or set(raw_root) != {
            "role",
            "kind",
            "content_sha256",
            "entry_count",
        }:
            raise RuntimeEnvironmentError("runtime root entry is invalid")
        role = raw_root.get("role")
        root_kind = raw_root.get("kind")
        content_sha256 = raw_root.get("content_sha256")
        entry_count = raw_root.get("entry_count")
        if (
            not isinstance(role, str)
            or re.fullmatch(r"[a-z][a-z0-9-]*", role) is None
            or role in roles
            or root_kind not in {"directory", "file"}
            or not isinstance(content_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
            or type(entry_count) is not int
            or entry_count < 1
        ):
            raise RuntimeEnvironmentError("runtime root identity is invalid")
        roles.add(role)
        runtime_roots.append(dict(raw_root))
    if runtime_roots != sorted(runtime_roots, key=lambda item: item["role"]):
        raise RuntimeEnvironmentError("runtime roots are not sorted")
    return {
        "schema": schema,
        "interpreter": dict(interpreter),
        "packages": packages,
        "executable_sha256": executable_sha256,
        "launcher": dict(launcher),
        "runtime_roots": runtime_roots,
        "native_linkage_sha256": native_linkage_sha256,
    }


def probe_python_environment(executable: Path) -> dict[str, object]:
    """Return a canonical inventory from one selected Python executable."""

    selected = _executable(executable)
    raw = _run_probe(
        [str(selected.invocation), "-I", "-c", _PYTHON_PROBE], "Python"
    )
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
    raw_roots = value.pop("_runtime_root_paths", None)
    raw_roots, native_linkage_sha256 = _with_native_dependency_roots(
        raw_roots, selected.target
    )
    value["runtime_roots"] = _runtime_root_inventory(raw_roots)
    value["native_linkage_sha256"] = native_linkage_sha256
    value["executable_sha256"] = _file_sha256(selected.target)
    value["launcher"] = {
        "kind": selected.launcher_kind,
        "sha256": selected.launcher_sha256,
    }
    _revalidate_executable(selected)
    return _validate_inventory(value, "python")


def _r_library_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    selected: list[Path] = []
    for path in paths:
        if not isinstance(path, Path):
            raise TypeError("R library paths must be pathlib.Path values")
        absolute = path.absolute()
        try:
            metadata = absolute.lstat()
        except OSError as error:
            raise RuntimeEnvironmentError("R library path is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeEnvironmentError("R library path must be a non-symlink directory")
        selected.append(absolute)
    if len(selected) != len(set(selected)):
        raise RuntimeEnvironmentError("R library paths are duplicated")
    return tuple(selected)


def probe_r_environment(
    executable: Path, *, library_paths: tuple[Path, ...] = ()
) -> dict[str, object]:
    """Return a canonical inventory from one selected Rscript executable."""

    selected = _executable(executable)
    selected_libraries = _r_library_paths(library_paths)
    raw = _run_probe(
        [str(selected.invocation), "--vanilla", "-e", _R_PROBE],
        "R",
        r_library_paths=selected_libraries,
    )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise RuntimeEnvironmentError("R runtime probe returned invalid text") from error
    if len(lines) < 2 or lines[0] != "MASKIMPUTE-R-RUNTIME-INVENTORY-V1":
        raise RuntimeEnvironmentError("R runtime probe header is invalid")
    identity = lines[1].split("\t")
    if len(identity) != 4 or not identity[3].isdigit():
        raise RuntimeEnvironmentError("R runtime interpreter identity is invalid")
    packages: list[dict[str, object]] = []
    raw_roots: list[dict[str, str]] = []
    for line in lines[2:]:
        fields = line.split("\t")
        if fields[0] == "LIB" and len(fields) == 3 and fields[1].isdigit():
            raw_roots.append(
                {
                    "role": f"library-path-{int(fields[1]):03d}",
                    "kind": "directory",
                    "path": fields[2],
                }
            )
            continue
        if (
            fields[0] != "PKG"
            or len(fields) != 5
            or not fields[3].isdigit()
        ):
            raise RuntimeEnvironmentError("R runtime package row is invalid")
        content_sha256, file_count = _directory_content_sha256(Path(fields[4]))
        packages.append(
            {
                "name": fields[1],
                "version": fields[2],
                "content_sha256": content_sha256,
                "file_count": file_count,
                "precedence": int(fields[3]),
            }
        )
    if len(raw_roots) != int(identity[3]):
        raise RuntimeEnvironmentError("R runtime library roots are incomplete")
    runtime_roots, native_linkage_sha256 = _with_native_dependency_roots(
        raw_roots, selected.target
    )
    inventory = _validate_inventory(
        {
            "schema": _R_SCHEMA,
            "interpreter": {
                "major": identity[0],
                "minor": identity[1],
                "platform": identity[2],
                "library_path_count": int(identity[3]),
            },
            "packages": packages,
            "executable_sha256": _file_sha256(selected.target),
            "launcher": {
                "kind": selected.launcher_kind,
                "sha256": selected.launcher_sha256,
            },
            "runtime_roots": _runtime_root_inventory(runtime_roots),
            "native_linkage_sha256": native_linkage_sha256,
        },
        "r",
    )
    _revalidate_executable(selected)
    return inventory


def probe_runtime_environment(
    kind: Literal["python", "r"],
    executable: Path,
    *,
    r_library_paths: tuple[Path, ...] = (),
) -> dict[str, object]:
    if kind == "python":
        return probe_python_environment(executable)
    if kind == "r":
        return probe_r_environment(executable, library_paths=r_library_paths)
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
    *,
    r_library_paths: Mapping[str, Sequence[Path]] | None = None,
) -> dict[str, object]:
    """Probe selected executables and build a deterministic lock payload."""

    if not isinstance(environments, Mapping) or not environments:
        raise RuntimeEnvironmentError("at least one runtime environment is required")
    libraries = {} if r_library_paths is None else dict(r_library_paths)
    if set(libraries) - set(environments):
        raise RuntimeEnvironmentError("R library paths name an unknown environment")
    entries: list[dict[str, object]] = []
    inventory_cache: dict[tuple[str, str, tuple[str, ...]], dict[str, object]] = {}
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
        selected_libraries = tuple(libraries.get(environment_id, ()))
        if kind != "r" and selected_libraries:
            raise RuntimeEnvironmentError(
                "R library paths cannot be assigned to a Python environment"
            )
        cache_key = (
            kind,
            executable.absolute().as_posix(),
            tuple(path.absolute().as_posix() for path in selected_libraries),
        )
        if cache_key not in inventory_cache:
            inventory_cache[cache_key] = probe_runtime_environment(
                kind, executable, r_library_paths=selected_libraries
            )
        inventory = inventory_cache[cache_key]
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
    *,
    r_library_paths: Mapping[str, Sequence[Path]] | None = None,
) -> dict[str, object]:
    """Independently probe every runtime and compare it with a loaded lock."""

    if not isinstance(lock, RuntimeEnvironmentLock):
        raise TypeError("lock must be a RuntimeEnvironmentLock")
    if not isinstance(environments, Mapping):
        raise TypeError("environments must be a mapping")
    libraries = {} if r_library_paths is None else dict(r_library_paths)
    if set(libraries) - set(environments):
        raise RuntimeEnvironmentError("R library paths name an unknown environment")
    expected_ids = {entry.environment_id for entry in lock.entries}
    observed_ids = set(environments)
    if observed_ids != expected_ids:
        raise RuntimeEnvironmentError("runtime IDs mismatch frozen lock")
    receipts: list[tuple[str, str]] = []
    inventory_cache: dict[tuple[str, str, tuple[str, ...]], dict[str, object]] = {}
    for environment_id in sorted(observed_ids):
        declaration = environments[environment_id]
        if (
            not isinstance(declaration, tuple)
            or len(declaration) != 2
            or declaration[0] not in {"python", "r"}
            or not isinstance(declaration[1], Path)
        ):
            raise RuntimeEnvironmentError("runtime environment declaration is invalid")
        kind, executable = declaration
        selected_libraries = tuple(libraries.get(environment_id, ()))
        if kind != "r" and selected_libraries:
            raise RuntimeEnvironmentError(
                "R library paths cannot be assigned to a Python environment"
            )
        entry = lock.by_id(environment_id)
        if kind != entry.kind:
            raise RuntimeEnvironmentError(
                f"runtime kind mismatch for {environment_id}"
            )
        cache_key = (
            kind,
            executable.absolute().as_posix(),
            tuple(path.absolute().as_posix() for path in selected_libraries),
        )
        if cache_key not in inventory_cache:
            inventory_cache[cache_key] = probe_runtime_environment(
                kind, executable, r_library_paths=selected_libraries
            )
        if inventory_cache[cache_key] != entry.inventory:
            raise RuntimeEnvironmentError(
                f"runtime inventory mismatch for {environment_id}"
            )
        inventory_sha256 = entry.inventory_sha256
        receipts.append((environment_id, inventory_sha256))
    return {
        "lock_file_sha256": lock.file_sha256,
        "environment_inventory_sha256s": tuple(receipts),
    }


def validate_runtime_environment_entry(
    lock: RuntimeEnvironmentLock,
    environment_id: str,
    kind: Literal["python", "r"],
    executable: Path,
    *,
    r_library_paths: tuple[Path, ...] = (),
) -> str:
    """Rehash one executable runtime immediately around one method execution."""

    if not isinstance(lock, RuntimeEnvironmentLock):
        raise TypeError("lock must be a RuntimeEnvironmentLock")
    entry = lock.by_id(environment_id)
    if kind != entry.kind:
        raise RuntimeEnvironmentError(f"runtime kind mismatch for {environment_id}")
    inventory = probe_runtime_environment(
        kind, executable, r_library_paths=r_library_paths
    )
    if inventory != entry.inventory:
        raise RuntimeEnvironmentError(
            f"runtime inventory mismatch for {environment_id}"
        )
    return entry.inventory_sha256


__all__ = [
    "RuntimeEnvironmentEntry",
    "RuntimeEnvironmentError",
    "RuntimeEnvironmentLock",
    "build_runtime_environment_lock",
    "load_runtime_environment_lock",
    "probe_python_environment",
    "probe_r_environment",
    "probe_runtime_environment",
    "runtime_environment_identity_sha256",
    "validate_runtime_environment_lock",
    "validate_runtime_environment_entry",
]
