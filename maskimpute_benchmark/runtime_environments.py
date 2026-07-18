"""Exact runtime-package inventories for publication competition environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import site
import stat
import struct
import subprocess
import sys
from typing import Any, Literal

from .protocol import canonical_sha256


_SAFE_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_PACKAGE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_LOCK_SCHEMA = "maskimpute-runtime-environment-lock-v1"
_PYTHON_SCHEMA = "maskimpute-python-runtime-inventory-v1"
_R_SCHEMA = "maskimpute-r-runtime-inventory-v1"
_MAX_PROBE_BYTES = 16 * 1024 * 1024
_MAX_ENVIRONMENT_ENTRIES = 1_048_576
_MAX_ENVIRONMENT_BYTES = 64 * 1024 * 1024
_MAX_LOCK_BYTES = 64 * 1024 * 1024


class RuntimeEnvironmentError(ValueError):
    """Raised when a runtime environment does not match its frozen inventory."""


_PYTHON_PROBE = r"""
import hashlib
import json
from pathlib import Path
import site
import sys

def stat_state(value):
    return [
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    ]

def executable_receipt(value):
    invocation = Path(value).absolute()
    target = invocation.resolve(strict=True)
    return {
        "invocation": str(invocation),
        "invocation_state": stat_state(invocation.lstat()),
        "target": str(target),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "target_state": stat_state(target.stat()),
    }

def is_beneath(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def search_path(value):
    return Path(value) if value else Path.cwd()

root_candidates = [
    ("base-prefix-configuration", Path(sys.base_prefix) / "pyvenv.cfg"),
    ("base-prefix-lib", Path(sys.base_prefix) / "lib"),
    ("prefix-configuration", Path(sys.prefix) / "pyvenv.cfg"),
    ("prefix-lib", Path(sys.prefix) / "lib"),
]
for index, value in enumerate(dict.fromkeys(
    [sys.executable, getattr(sys, "_base_executable", sys.executable)]
)):
    executable_path = Path(value).absolute()
    root_candidates.extend((
        (f"executable-configuration-{index:03d}", executable_path.parent / "pyvenv.cfg"),
        (f"executable-parent-configuration-{index:03d}", executable_path.parent.parent / "pyvenv.cfg"),
    ))
user_sites = site.getusersitepackages()
if isinstance(user_sites, str):
    user_sites = [user_sites]
if site.ENABLE_USER_SITE:
    for index, value in enumerate(user_sites):
        root_candidates.append((f"user-site-{index:03d}", Path(value)))
covered = [path.resolve(strict=True) for _role, path in root_candidates if path.exists()]
missing_search_paths = []
for role, path in root_candidates:
    if ("configuration" in role or role.startswith("user-site-")) and not path.exists():
        missing_search_paths.append(str(path.absolute()))
for index, value in enumerate(sys.path):
    candidate = search_path(value)
    if not candidate.exists():
        missing_search_paths.append(str(candidate.absolute()))
        continue
    selected = candidate.resolve(strict=True)
    if any(selected == root or is_beneath(selected, root) for root in covered):
        continue
    root_candidates.append((f"search-path-{index:03d}", candidate))
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
    "_distribution_search_paths": [
        {"path": str(search_path(value).resolve(strict=True)), "precedence": index}
        for index, value in enumerate(sys.path)
        if search_path(value).exists()
    ],
    "_missing_search_paths": missing_search_paths,
    "_runtime_executables": [
        executable_receipt(value)
        for value in dict.fromkeys(
            [sys.executable, getattr(sys, "_base_executable", sys.executable)]
        )
        if value
    ],
    "_runtime_prefixes": list(dict.fromkeys([
        sys.prefix,
        sys.base_prefix,
        *([site.getuserbase()] if site.ENABLE_USER_SITE else []),
    ])),
    "_runtime_root_paths": sorted(runtime_roots, key=lambda value: value["role"]),
}
print(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True))
"""


_PYTHON_ROOT_PROBE = r"""
import hashlib
import json
from pathlib import Path
import site
import sys

def stat_state(value):
    return [
        value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
        value.st_size, value.st_mtime_ns, value.st_ctime_ns,
    ]

def executable_receipt(value):
    invocation = Path(value).absolute()
    target = invocation.resolve(strict=True)
    return {
        "invocation": str(invocation),
        "invocation_state": stat_state(invocation.lstat()),
        "target": str(target),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "target_state": stat_state(target.stat()),
    }

def is_beneath(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False

def search_path(value):
    return Path(value) if value else Path.cwd()

root_candidates = [
    ("base-prefix-configuration", Path(sys.base_prefix) / "pyvenv.cfg"),
    ("base-prefix-lib", Path(sys.base_prefix) / "lib"),
    ("prefix-configuration", Path(sys.prefix) / "pyvenv.cfg"),
    ("prefix-lib", Path(sys.prefix) / "lib"),
]
for index, value in enumerate(dict.fromkeys(
    [sys.executable, getattr(sys, "_base_executable", sys.executable)]
)):
    executable_path = Path(value).absolute()
    root_candidates.extend((
        (f"executable-configuration-{index:03d}", executable_path.parent / "pyvenv.cfg"),
        (f"executable-parent-configuration-{index:03d}", executable_path.parent.parent / "pyvenv.cfg"),
    ))
user_sites = site.getusersitepackages()
if isinstance(user_sites, str):
    user_sites = [user_sites]
if site.ENABLE_USER_SITE:
    for index, value in enumerate(user_sites):
        root_candidates.append((f"user-site-{index:03d}", Path(value)))
covered = [path.resolve(strict=True) for _role, path in root_candidates if path.exists()]
missing_search_paths = []
for role, path in root_candidates:
    if ("configuration" in role or role.startswith("user-site-")) and not path.exists():
        missing_search_paths.append(str(path.absolute()))
for index, value in enumerate(sys.path):
    candidate = search_path(value)
    if not candidate.exists():
        missing_search_paths.append(str(candidate.absolute()))
        continue
    selected = candidate.resolve(strict=True)
    if any(selected == root or is_beneath(selected, root) for root in covered):
        continue
    root_candidates.append((f"search-path-{index:03d}", candidate))
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
payload = {
    "distribution_search_paths": [
        {"path": str(search_path(value).resolve(strict=True)), "precedence": index}
        for index, value in enumerate(sys.path)
        if search_path(value).exists()
    ],
    "missing_search_paths": missing_search_paths,
    "runtime_executables": [
        executable_receipt(value)
        for value in dict.fromkeys(
            [sys.executable, getattr(sys, "_base_executable", sys.executable)]
        )
        if value
    ],
    "runtime_prefixes": list(dict.fromkeys([
        sys.prefix,
        sys.base_prefix,
        *([site.getuserbase()] if site.ENABLE_USER_SITE else []),
    ])),
    "roots": sorted(roots, key=lambda value: value["role"]),
}
print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
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


def _libc_environment_entries() -> tuple[bytes, ...]:
    """Read the process's actual ``char **environ`` without ``os.environ`` caching."""

    try:
        libc = ctypes.CDLL(None)
        environ = ctypes.POINTER(ctypes.c_char_p).in_dll(libc, "environ")
    except (OSError, ValueError) as error:
        raise RuntimeEnvironmentError(
            "libc process environment is unavailable"
        ) from error
    entries: list[bytes] = []
    total = 0
    for index in range(_MAX_ENVIRONMENT_ENTRIES):
        raw = environ[index]
        if raw is None:
            return tuple(entries)
        value = bytes(raw)
        if b"=" not in value or value.startswith(b"="):
            raise RuntimeEnvironmentError("libc process environment entry is invalid")
        total += len(value)
        if total > _MAX_ENVIRONMENT_BYTES:
            raise RuntimeEnvironmentError("libc process environment is too large")
        entries.append(value)
    raise RuntimeEnvironmentError("libc process environment has too many entries")


def process_environment_sha256() -> str:
    """Hash the ordered byte representation inherited by child processes."""

    digest = hashlib.sha256(b"maskimpute-libc-process-environment-v1\0")
    entries = _libc_environment_entries()
    digest.update(len(entries).to_bytes(8, "little"))
    for entry in entries:
        digest.update(len(entry).to_bytes(8, "little"))
        digest.update(entry)
    return digest.hexdigest()


def _libc_environment_mapping() -> dict[str, str]:
    environment: dict[str, str] = {}
    for entry in _libc_environment_entries():
        raw_name, raw_value = entry.split(b"=", 1)
        name = os.fsdecode(raw_name)
        if name in environment:
            raise RuntimeEnvironmentError(
                "libc process environment contains duplicate variable names"
            )
        environment[name] = os.fsdecode(raw_value)
    return environment


def _reject_loader_injection(environment: dict[str, str]) -> None:
    for variable in ("LD_AUDIT", "LD_PRELOAD"):
        if environment.get(variable):
            raise RuntimeEnvironmentError(
                f"{variable} is unsupported for publication runtime discovery"
            )


def publication_runtime_working_directory() -> Path:
    """Return the bound CWD used whenever publication runtime code can load."""

    selected = Path(__file__).resolve().parents[1] / "environments"
    try:
        metadata = selected.lstat()
    except OSError as error:
        raise RuntimeEnvironmentError(
            "publication runtime working directory is unavailable"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeEnvironmentError(
            "publication runtime working directory must be a non-symlink directory"
        )
    return selected


def publication_python_spawn_search_path() -> tuple[str, ...]:
    """Return the existing, bound search path restored in spawned children."""

    repository = Path(__file__).resolve().parents[1]
    fixed_cwd = publication_runtime_working_directory()
    allowed_roots = [
        Path(sys.prefix).resolve(strict=True),
        Path(sys.base_prefix).resolve(strict=True),
        repository,
    ]
    if site.ENABLE_USER_SITE:
        user_base = Path(site.getuserbase())
        if user_base.is_dir():
            allowed_roots.append(user_base.resolve(strict=True))
    environment = _libc_environment_mapping()
    for raw_path in environment.get("PYTHONPATH", "").split(os.pathsep):
        candidate = fixed_cwd if not raw_path else Path(raw_path)
        if not candidate.is_absolute():
            candidate = fixed_cwd / candidate
        if candidate.exists():
            allowed_roots.append(candidate.resolve(strict=True))

    def beneath(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    selected: list[str] = []
    for value in sys.path:
        if not value:
            candidate = fixed_cwd
        else:
            candidate = Path(value)
            if not candidate.is_absolute():
                continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if not any(beneath(resolved, root) for root in allowed_roots):
            continue
        text = resolved.as_posix()
        if text not in selected:
            selected.append(text)
    repository_text = repository.as_posix()
    if repository_text not in selected:
        selected.insert(0, repository_text)
    return tuple(selected)


def _missing_python_search_path_roots(raw_paths: object) -> list[dict[str, str]]:
    if not isinstance(raw_paths, list) or any(
        not isinstance(value, str) or not Path(value).is_absolute()
        for value in raw_paths
    ):
        raise RuntimeEnvironmentError("Python missing search paths are invalid")
    ancestors: set[Path] = set()
    for value in raw_paths:
        selected = Path(value)
        if selected.exists() or selected.is_symlink():
            raise RuntimeEnvironmentError(
                "Python search path changed after target probing"
            )
        ancestor = selected.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        try:
            resolved = ancestor.resolve(strict=True)
        except OSError as error:
            raise RuntimeEnvironmentError(
                "Python missing search path has no stable ancestor"
            ) from error
        if not resolved.is_dir():
            raise RuntimeEnvironmentError(
                "Python missing search path ancestor is not a directory"
            )
        ancestors.add(resolved)
    return [
        {
            "role": f"python-missing-search-ancestor-{index:03d}",
            "kind": "search-directory",
            "path": path.as_posix(),
        }
        for index, path in enumerate(sorted(ancestors, key=os.fsencode))
    ]


def _python_distribution_inventory(
    raw_search_paths: object,
    *,
    runtime_prefixes: object,
    runtime_roots: object,
    include_content: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    if not isinstance(raw_search_paths, list):
        raise RuntimeEnvironmentError("Python distribution search paths are invalid")
    selected: dict[Path, int] = {}
    for entry in raw_search_paths:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "precedence"}
            or not isinstance(entry.get("path"), str)
            or not Path(entry["path"]).is_absolute()
            or type(entry.get("precedence")) is not int
            or entry["precedence"] < 0
        ):
            raise RuntimeEnvironmentError(
                "Python distribution search path entry is invalid"
            )
        try:
            path = Path(entry["path"]).resolve(strict=True)
        except OSError as error:
            raise RuntimeEnvironmentError(
                "Python distribution search path is unavailable"
            ) from error
        precedence = entry["precedence"]
        selected[path] = min(precedence, selected.get(path, precedence))

    if (
        not isinstance(runtime_prefixes, list)
        or not runtime_prefixes
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in runtime_prefixes
        )
    ):
        raise RuntimeEnvironmentError("Python runtime prefixes are invalid")
    try:
        allowed_prefixes = tuple(
            dict.fromkeys(
                [
                    *(Path(value).resolve(strict=True) for value in runtime_prefixes),
                    *selected,
                ]
            )
        )
    except OSError as error:
        raise RuntimeEnvironmentError("Python runtime prefix is unavailable") from error
    if any(not path.is_dir() for path in allowed_prefixes):
        raise RuntimeEnvironmentError("Python runtime prefix is not a directory")
    covered_roots = _validated_runtime_root_paths(runtime_roots)

    def beneath(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    packages: list[dict[str, object]] = []
    external_directories: set[Path] = set()
    identities: set[tuple[str, int, str, str]] = set()
    for search_path, precedence in sorted(
        selected.items(), key=lambda item: (item[1], os.fsencode(item[0]))
    ):
        try:
            distributions = importlib.metadata.distributions(path=[str(search_path)])
            for distribution in distributions:
                name = distribution.metadata.get("Name")
                version = distribution.version
                if not name or not version:
                    raise RuntimeEnvironmentError(
                        "installed distribution lacks name or version"
                    )
                files = sorted(
                    (str(value), value) for value in (distribution.files or ())
                )
                digest = hashlib.sha256(b"maskimpute-python-distribution-content-v1\0")
                for logical, relative in files:
                    encoded = logical.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "little"))
                    digest.update(encoded)
                    path = Path(os.path.abspath(distribution.locate_file(relative)))
                    matching_prefixes = tuple(
                        prefix for prefix in allowed_prefixes if beneath(path, prefix)
                    )
                    if not matching_prefixes:
                        raise RuntimeEnvironmentError(
                            "installed distribution file escaped runtime prefixes"
                        )
                    if not path.exists() and not path.is_symlink():
                        covered_missing = any(
                            kind == "directory"
                            and beneath(path, root.resolve(strict=True))
                            for _role, kind, root in covered_roots
                        )
                        parent = path.parent
                        while not parent.exists() and parent != parent.parent:
                            parent = parent.parent
                        parent = parent.resolve(strict=True)
                        if not any(
                            beneath(parent, prefix) for prefix in matching_prefixes
                        ):
                            raise RuntimeEnvironmentError(
                                "installed distribution file escaped runtime prefixes"
                            )
                        if not covered_missing:
                            external_directories.add(parent)
                        digest.update(b"M")
                        continue
                    resolved = path.resolve(strict=True)
                    if not any(
                        beneath(resolved, prefix) for prefix in matching_prefixes
                    ):
                        raise RuntimeEnvironmentError(
                            "installed distribution file escaped runtime prefixes"
                        )
                    covered = any(
                        (
                            kind == "directory"
                            and beneath(resolved, root.resolve(strict=True))
                        )
                        or (kind == "file" and resolved == root.resolve(strict=True))
                        for _role, kind, root in covered_roots
                    )
                    if not covered:
                        external_directories.add(path.parent)
                    if path.is_symlink():
                        target_text = os.fsencode(os.readlink(path))
                        digest.update(b"L")
                        digest.update(len(target_text).to_bytes(8, "little"))
                        digest.update(target_text)
                        if resolved.is_file() and include_content:
                            payload = _secure_regular_file_bytes(resolved)
                            digest.update(len(payload).to_bytes(8, "little"))
                            digest.update(payload)
                    elif path.is_file():
                        digest.update(b"F")
                        if include_content:
                            payload = _secure_regular_file_bytes(path)
                            digest.update(len(payload).to_bytes(8, "little"))
                            digest.update(payload)
                    else:
                        raise RuntimeEnvironmentError(
                            "installed distribution file is not a regular file"
                        )
                normalized = re.sub(r"[-_.]+", "-", name).lower()
                identity = (normalized, precedence, str(version), digest.hexdigest())
                if identity in identities:
                    raise RuntimeEnvironmentError(
                        "installed distributions contain duplicate identities"
                    )
                identities.add(identity)
                packages.append(
                    {
                        "content_sha256": identity[3],
                        "file_count": len(files),
                        "name": normalized,
                        "precedence": precedence,
                        "version": str(version),
                    }
                )
        except RuntimeEnvironmentError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise RuntimeEnvironmentError(
                "cannot inventory installed Python distributions"
            ) from error
    return (
        sorted(
            packages,
            key=lambda value: (value["name"], value["precedence"], value["version"]),
        ),
        [
            {
                "role": f"distribution-artifact-directory-{index:05d}",
                "kind": "directory",
                "path": path.as_posix(),
            }
            for index, path in enumerate(sorted(external_directories, key=os.fsencode))
        ],
    )


def _secure_regular_file_bytes(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeEnvironmentError("runtime file is not regular")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = path.lstat()
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError("runtime file changed while reading") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    expected = _stat_identity(before)
    if any(
        _stat_identity(metadata) != expected for metadata in (opened, after, path_after)
    ):
        raise RuntimeEnvironmentError("runtime file changed while reading")
    return b"".join(chunks)


def _stable_kernel_modules_bytes(path: Path = Path("/proc/modules")) -> bytes:
    try:
        lines = (
            _secure_regular_file_bytes(path)
            .decode("utf-8", errors="strict")
            .splitlines()
        )
    except UnicodeError as error:
        raise RuntimeEnvironmentError("kernel module state is not UTF-8") from error
    modules: list[dict[str, object]] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 6:
            raise RuntimeEnvironmentError("kernel module state row is invalid")
        if not fields[0].startswith("nvidia"):
            continue
        if not fields[1].isdigit() or not fields[2].isdigit():
            raise RuntimeEnvironmentError("kernel module state row is invalid")
        dependencies = (
            []
            if fields[3] == "-"
            else sorted(value for value in fields[3].split(",") if value)
        )
        modules.append(
            {
                "dependencies": dependencies,
                "name": fields[0],
                "size": int(fields[1]),
                "state": fields[4],
            }
        )
    return json.dumps(
        sorted(modules, key=lambda value: value["name"]),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _control_file_sha256(path: Path) -> str:
    if path == Path("/proc/modules"):
        return hashlib.sha256(
            b"maskimpute-nvidia-kernel-modules-v1\0"
            + _stable_kernel_modules_bytes(path)
        ).hexdigest()
    return _file_sha256(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_secure_regular_file_bytes(path)).hexdigest()


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
            _stat_identity(value) != identity for value in (opened, after, after_path)
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
            children = sorted(
                os.scandir(directory), key=lambda item: os.fsencode(item.name)
            )
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
    if path == Path("/proc/modules"):
        digest = hashlib.sha256(
            b"maskimpute-runtime-root-nvidia-modules-v1\0"
            + _stable_kernel_modules_bytes(path)
        )
        return digest.hexdigest(), 1

    def read_regular(item: Path) -> bytes:
        descriptor: int | None = None
        try:
            before = item.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeEnvironmentError("runtime root file is not regular")
            descriptor = os.open(
                item,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            path_after = item.lstat()
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime root file changed while hashing"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        expected = _stat_identity(before)
        if any(
            _stat_identity(metadata) != expected
            for metadata in (opened, after, path_after)
        ):
            raise RuntimeEnvironmentError("runtime root file changed while hashing")
        return b"".join(chunks)

    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            target_text = os.fsencode(os.readlink(path))
            target = path.resolve(strict=True)
            digest = hashlib.sha256(
                b"maskimpute-runtime-root-symlink-v1\0" + target_text
            )
            if target.is_file():
                digest.update(read_regular(target))
            elif target.is_dir():
                nested_sha, nested_count = _directory_content_sha256(target)
                digest.update(bytes.fromhex(nested_sha))
                if _stat_identity(path.lstat()) != _stat_identity(before):
                    raise RuntimeEnvironmentError(
                        "runtime root file changed while hashing"
                    )
                return digest.hexdigest(), nested_count + 1
            else:
                raise RuntimeEnvironmentError("runtime root symlink target is invalid")
        elif stat.S_ISREG(before.st_mode):
            digest = hashlib.sha256(
                b"maskimpute-runtime-root-file-v1\0" + read_regular(path)
            )
        else:
            raise RuntimeEnvironmentError("runtime root file is invalid")
        after = path.lstat()
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError(
            "runtime root file changed while hashing"
        ) from error
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeEnvironmentError("runtime root file changed while hashing")
    return digest.hexdigest(), 1


def _search_directory_content_sha256(path: Path) -> tuple[str, int]:
    """Hash loader search-state names/identities without unrelated file bytes."""

    try:
        alias_before = path.lstat()
        target = path.resolve(strict=True)
        before = target.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeEnvironmentError("runtime search path is not a directory")
        children = sorted(os.scandir(target), key=lambda item: os.fsencode(item.name))
        digest = hashlib.sha256(b"maskimpute-runtime-search-directory-v1\0")
        if stat.S_ISLNK(alias_before.st_mode):
            target_text = os.fsencode(os.readlink(path))
            digest.update(b"L" + len(target_text).to_bytes(8, "little"))
            digest.update(target_text)
        digest.update(repr(_stat_identity(before)).encode("ascii"))
        for child_entry in children:
            encoded = os.fsencode(child_entry.name)
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        after = target.lstat()
        alias_after = path.lstat()
        names_after = sorted(
            (entry.name for entry in os.scandir(target)), key=os.fsencode
        )
        if (
            _stat_identity(alias_before) != _stat_identity(alias_after)
            or _stat_identity(before) != _stat_identity(after)
            or [entry.name for entry in children] != names_after
        ):
            raise RuntimeEnvironmentError(
                "runtime search directory changed while hashing"
            )
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError(
            "runtime search directory changed while hashing"
        ) from error
    return digest.hexdigest(), 1 + len(children)


def _validated_runtime_root_paths(
    raw_roots: object,
) -> tuple[tuple[str, Literal["directory", "file", "search-directory"], Path], ...]:
    if not isinstance(raw_roots, list) or not raw_roots:
        raise RuntimeEnvironmentError("runtime probe did not expose its roots")
    result: list[
        tuple[str, Literal["directory", "file", "search-directory"], Path]
    ] = []
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
            or kind not in {"directory", "file", "search-directory"}
            or not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
        ):
            raise RuntimeEnvironmentError("runtime probe root identity is invalid")
        roles.add(role)
        path = Path(raw_path)
        try:
            path.resolve(strict=True)
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError("runtime root is unavailable") from error
        if kind in {"directory", "search-directory"} and not path.is_dir():
            raise RuntimeEnvironmentError("runtime root kind is inconsistent")
        if kind == "file" and not path.is_file():
            raise RuntimeEnvironmentError("runtime root kind is inconsistent")
        result.append((role, kind, path))
    return tuple(sorted(result, key=lambda item: item[0]))


def _dynamic_elf(path: Path) -> bool:
    try:
        before = path.stat()
        with path.open("rb") as stream:
            header = stream.read(18)
        after = path.stat()
    except OSError as error:
        raise RuntimeEnvironmentError(
            "native runtime candidate changed during discovery"
        ) from error
    if _stat_identity(before) != _stat_identity(after):
        raise RuntimeEnvironmentError(
            "native runtime candidate changed during discovery"
        )
    if len(header) < 18 or header[:4] != b"\x7fELF":
        return False
    if header[5] == 1:
        byteorder = "little"
    elif header[5] == 2:
        byteorder = "big"
    else:
        raise RuntimeEnvironmentError("native runtime ELF byte order is invalid")
    return int.from_bytes(header[16:18], byteorder) in {2, 3}


def _runtime_elf_candidates(
    roots: tuple[
        tuple[str, Literal["directory", "file", "search-directory"], Path], ...
    ],
) -> set[Path]:
    candidates: set[Path] = set()
    visited_directories: set[Path] = set()

    def visit_file(path: Path) -> None:
        try:
            with path.open("rb") as stream:
                header = stream.read(18)
            if len(header) >= 18 and header[:4] == b"\x7fELF":
                resolved = path.resolve(strict=True)
                if _dynamic_elf(resolved):
                    candidates.add(resolved)
        except OSError as error:
            raise RuntimeEnvironmentError(
                "native runtime file changed during discovery"
            ) from error

    def visit_directory(path: Path, ancestry: frozenset[Path]) -> None:
        try:
            resolved = path.resolve(strict=True)
            if resolved in ancestry:
                raise RuntimeEnvironmentError(
                    "native runtime tree contains a directory cycle"
                )
            if resolved in visited_directories:
                return
            visited_directories.add(resolved)
            children = sorted(
                os.scandir(path), key=lambda entry: os.fsencode(entry.name)
            )
            for child in children:
                if child.is_dir(follow_symlinks=True):
                    visit_directory(Path(child.path), ancestry.union({resolved}))
                elif child.is_file(follow_symlinks=True):
                    visit_file(Path(child.path))
                else:
                    raise RuntimeEnvironmentError(
                        "native runtime tree contains a special filesystem entry"
                    )
            names_after = sorted(
                (entry.name for entry in os.scandir(path)), key=os.fsencode
            )
            if [entry.name for entry in children] != names_after:
                raise RuntimeEnvironmentError(
                    "native runtime tree changed during discovery"
                )
        except RuntimeEnvironmentError:
            raise
        except OSError as error:
            raise RuntimeEnvironmentError(
                "native runtime tree changed during discovery"
            ) from error

    for _role, kind, root in roots:
        if kind == "directory":
            visit_directory(root, frozenset())
        elif kind == "file":
            visit_file(root)
    return candidates


def _tool_execution_roots(role: str, path: Path) -> list[tuple[str, str, Path]]:
    """Bind a publication helper plus each shebang interpreter it executes."""

    pending = [path.absolute()]
    seen: set[Path] = set()
    selected: list[Path] = []
    for _depth in range(8):
        if not pending:
            break
        lexical = pending.pop(0)
        try:
            target = lexical.resolve(strict=True)
        except OSError as error:
            raise RuntimeEnvironmentError(
                f"{role} execution helper is unavailable"
            ) from error
        if target in seen:
            continue
        seen.add(target)
        for candidate in (lexical, target):
            if candidate not in selected:
                selected.append(candidate)
        payload = _secure_regular_file_bytes(target)
        if payload.startswith(b"\x7fELF"):
            continue
        first_line = payload.split(b"\n", 1)[0]
        if not first_line.startswith(b"#!") or len(first_line) > 255:
            raise RuntimeEnvironmentError(
                f"{role} execution helper has an invalid shebang"
            )
        try:
            declaration = first_line[2:].decode("utf-8", errors="strict").strip()
        except UnicodeError as error:
            raise RuntimeEnvironmentError(
                f"{role} execution helper shebang is not UTF-8"
            ) from error
        fields = declaration.split(maxsplit=1)
        if not fields or not Path(fields[0]).is_absolute():
            raise RuntimeEnvironmentError(
                f"{role} execution helper shebang is unsupported"
            )
        if Path(fields[0]).name == "env":
            raise RuntimeEnvironmentError(
                f"{role} execution helper must not use env in its shebang"
            )
        pending.append(Path(fields[0]))
    else:
        raise RuntimeEnvironmentError(f"{role} execution helper chain is too deep")
    return [
        (
            role if index == 0 else f"{role}-bootstrap-{index:02d}",
            "file",
            candidate,
        )
        for index, candidate in enumerate(selected)
    ]


def _path_selected_executable(
    command: str,
) -> tuple[Path | None, tuple[Path, ...]]:
    environment = _libc_environment_mapping()
    raw_path = environment.get("PATH")
    if raw_path is None:
        raise RuntimeEnvironmentError("publication PATH is unavailable")
    selected_text = shutil.which(command, path=raw_path)
    if selected_text is None:
        return None, ()
    selected = Path(selected_text).absolute()
    search_directories: list[Path] = []
    for raw_directory in raw_path.split(os.pathsep):
        if not raw_directory or not Path(raw_directory).is_absolute():
            raise RuntimeEnvironmentError(
                "publication PATH entries must be nonempty absolute directories"
            )
        directory = Path(os.path.abspath(raw_directory))
        if not directory.is_dir():
            raise RuntimeEnvironmentError("publication PATH directory is unavailable")
        if directory not in search_directories:
            search_directories.append(directory)
        if directory == selected.parent:
            return selected, tuple(search_directories)
    raise RuntimeEnvironmentError("publication PATH resolution is inconsistent")


def publication_git_executable() -> Path:
    """Return the exact monitored git selected through libc PATH."""

    selected, _directories = _path_selected_executable("git")
    if selected is None:
        raise RuntimeEnvironmentError("git is unavailable for source verification")
    return selected


def _host_loader_roots() -> tuple[list[tuple[str, str, Path]], bytes, set[str]]:
    environment = _libc_environment_mapping()
    _reject_loader_injection(environment)
    cache = Path("/etc/ld.so.cache")
    if not cache.is_file():
        raise RuntimeEnvironmentError("dynamic loader cache is unavailable")
    ldconfig = next(
        (
            path
            for path in (Path("/usr/sbin/ldconfig"), Path("/sbin/ldconfig"))
            if path.is_file()
        ),
        None,
    )
    if ldconfig is None:
        raise RuntimeEnvironmentError("ldconfig is unavailable")
    try:
        completed = subprocess.run(
            [ldconfig.as_posix(), "-p"],
            check=False,
            capture_output=True,
            env=_probe_environment(),
            cwd=publication_runtime_working_directory(),
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeEnvironmentError("ldconfig discovery failed") from error
    if completed.returncode != 0 or len(completed.stdout) > _MAX_PROBE_BYTES:
        raise RuntimeEnvironmentError("ldconfig discovery failed")
    try:
        output = completed.stdout.decode("utf-8", errors="strict")
        completed.stderr.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise RuntimeEnvironmentError("ldconfig output is not UTF-8") from error
    libraries: dict[Path, Path] = {}
    logical_names: set[str] = set()
    for line in output.splitlines():
        if "=>" not in line:
            continue
        match = re.fullmatch(r"\s*(\S+)\s+.*=>\s+(/\S+)\s*", line)
        if match is None:
            raise RuntimeEnvironmentError("ldconfig output row is invalid")
        raw_path = Path(match.group(2))
        try:
            resolved = raw_path.resolve(strict=True)
        except OSError as error:
            raise RuntimeEnvironmentError(
                "ldconfig library path is unavailable"
            ) from error
        if not resolved.is_file():
            raise RuntimeEnvironmentError("ldconfig library path is not a file")
        logical_names.add(match.group(1))
        libraries.setdefault(raw_path, resolved)
    if not libraries:
        raise RuntimeEnvironmentError("ldconfig exposed no library paths")

    roots: list[tuple[str, str, Path]] = [
        ("dynamic-loader-cache", "file", cache),
    ]
    roots.extend(_tool_execution_roots("ldconfig-executable", ldconfig))
    git, git_search_directories = _path_selected_executable("git")
    if git is None:
        raise RuntimeEnvironmentError("git is unavailable for source verification")
    roots.extend(_tool_execution_roots("git-executable", git))
    roots.extend(
        (
            f"git-search-directory-{index:03d}",
            "search-directory",
            directory,
        )
        for index, directory in enumerate(git_search_directories)
    )
    configuration = Path("/etc/ld.so.conf")
    if configuration.is_file():
        roots.append(("dynamic-loader-configuration", "file", configuration))
    configuration_directory = Path("/etc/ld.so.conf.d")
    if configuration_directory.is_dir():
        roots.append(
            (
                "dynamic-loader-configuration-directory",
                "directory",
                configuration_directory,
            )
        )
    preload = Path("/etc/ld.so.preload")
    if preload.exists() or preload.is_symlink():
        roots.extend(_loader_preload_roots(preload))
    else:
        roots.append(
            (
                "dynamic-loader-preload-parent",
                "search-directory",
                preload.parent,
            )
        )
    for index, raw_path in enumerate(sorted(libraries, key=os.fsencode)):
        roots.append((f"loader-cache-library-{index:04d}", "file", raw_path))
        resolved = libraries[raw_path]
        if resolved != raw_path:
            roots.append((f"loader-cache-library-target-{index:04d}", "file", resolved))
    loader_directories = {
        directory
        for raw_path, resolved in libraries.items()
        for directory in (raw_path.parent, resolved.parent)
    }
    roots.extend(
        (
            f"loader-default-search-directory-{index:03d}",
            "search-directory",
            directory,
        )
        for index, directory in enumerate(sorted(loader_directories, key=os.fsencode))
    )
    loader_path = environment.get("LD_LIBRARY_PATH")
    if loader_path is not None:
        for index, raw_path in enumerate(loader_path.split(os.pathsep)):
            if not raw_path or not Path(raw_path).is_absolute():
                raise RuntimeEnvironmentError(
                    "LD_LIBRARY_PATH entries must be nonempty absolute paths"
                )
            try:
                selected = Path(raw_path)
                for component in (selected, *selected.parents):
                    if stat.S_ISLNK(component.lstat().st_mode):
                        raise RuntimeEnvironmentError(
                            "LD_LIBRARY_PATH entries must not contain symlinks"
                        )
                selected.resolve(strict=True)
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "LD_LIBRARY_PATH entry is unavailable"
                ) from error
            if not selected.is_dir():
                raise RuntimeEnvironmentError(
                    "LD_LIBRARY_PATH entry is not a directory"
                )
            roots.append((f"loader-search-root-{index:03d}", "directory", selected))
    return roots, completed.stdout, logical_names


def _loader_preload_roots(path: Path) -> list[tuple[str, str, Path]]:
    """Bind ld.so.preload bytes plus every lexical alias and resolved target."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise RuntimeEnvironmentError("dynamic loader preload path is invalid")
    try:
        preload_before = path.lstat()
        preload_target = path.resolve(strict=True)
        payload = _secure_regular_file_bytes(preload_target).decode(
            "utf-8", errors="strict"
        )
        if (
            _stat_identity(path.lstat()) != _stat_identity(preload_before)
            or path.resolve(strict=True) != preload_target
        ):
            raise RuntimeEnvironmentError(
                "dynamic loader preload configuration changed"
            )
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError(
            "dynamic loader preload configuration is unavailable"
        ) from error
    except UnicodeError as error:
        raise RuntimeEnvironmentError(
            "dynamic loader preload configuration is not UTF-8"
        ) from error
    entries: list[Path] = []
    for line in payload.splitlines():
        content = line.split("#", 1)[0]
        for raw_value in re.split(r"[:\s]+", content.strip()):
            if not raw_value:
                continue
            selected = Path(raw_value)
            if not selected.is_absolute():
                raise RuntimeEnvironmentError(
                    "dynamic loader preload entry is not absolute"
                )
            try:
                target = selected.resolve(strict=True)
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "dynamic loader preload entry is unavailable"
                ) from error
            if not target.is_file():
                raise RuntimeEnvironmentError(
                    "dynamic loader preload entry is not a file"
                )
            entries.append(selected)
    roots: list[tuple[str, str, Path]] = [("dynamic-loader-preload", "file", path)]
    if preload_target != path:
        roots.append(("dynamic-loader-preload-target", "file", preload_target))
    for index, selected in enumerate(dict.fromkeys(entries)):
        roots.append((f"dynamic-loader-preload-library-{index:03d}", "file", selected))
        target = selected.resolve(strict=True)
        if target != selected:
            roots.append(
                (
                    f"dynamic-loader-preload-library-target-{index:03d}",
                    "file",
                    target,
                )
            )
    return roots


def nvidia_smi_executable() -> Path | None:
    """Resolve the exact nvidia-smi invocation inherited through libc PATH."""

    selected, _directories = _path_selected_executable("nvidia-smi")
    if selected is None:
        return None
    return _executable(selected).invocation


def _gpu_driver_roots(
    logical_libraries: set[str],
    nvidia_smi: _ExecutableIdentity | None,
) -> list[tuple[str, str, Path]]:
    roots: list[tuple[str, str, Path]] = []
    for role, path in (
        ("kernel-modules-state", Path("/proc/modules")),
        ("kernel-version", Path("/proc/version")),
        ("kernel-release", Path("/proc/sys/kernel/osrelease")),
        ("gpu-driver-version", Path("/proc/driver/nvidia/version")),
    ):
        if path.is_file():
            roots.append((role, "file", path))
    version_files = sorted(
        (
            path
            for pattern in (
                "/sys/module/nvidia*/version",
                "/sys/module/nvidia*/srcversion",
            )
            for path in Path("/").glob(pattern.removeprefix("/"))
            if path.is_file()
        ),
        key=os.fsencode,
    )
    roots.extend(
        (f"gpu-module-version-{index:03d}", "file", path)
        for index, path in enumerate(version_files)
    )
    gpu_present = (
        nvidia_smi is not None or Path("/proc/driver/nvidia/version").is_file()
    )
    if nvidia_smi is not None:
        selected_nvidia_smi, search_directories = _path_selected_executable(
            "nvidia-smi"
        )
        if selected_nvidia_smi != nvidia_smi.invocation:
            raise RuntimeEnvironmentError(
                "nvidia-smi PATH selection changed during discovery"
            )
        roots.extend(
            (
                f"nvidia-smi-search-directory-{index:03d}",
                "search-directory",
                directory,
            )
            for index, directory in enumerate(search_directories)
        )
        roots.append(("nvidia-smi-executable", "file", nvidia_smi.invocation))
        if nvidia_smi.target != nvidia_smi.invocation:
            roots.append(("nvidia-smi-executable-target", "file", nvidia_smi.target))
    if gpu_present and not {"libcuda.so.1", "libnvidia-ml.so.1"} <= logical_libraries:
        raise RuntimeEnvironmentError(
            "GPU driver libraries are absent from the dynamic loader cache"
        )
    modules_path = Path("/proc/modules")
    if modules_path.is_file():
        try:
            module_names = sorted(
                {
                    line.split()[0]
                    for line in modules_path.read_text(encoding="utf-8").splitlines()
                    if line.split() and line.split()[0].startswith("nvidia")
                }
            )
        except (OSError, UnicodeError) as error:
            raise RuntimeEnvironmentError(
                "GPU kernel module state is unavailable"
            ) from error
        modinfo = next(
            (
                path
                for path in (Path("/usr/sbin/modinfo"), Path("/sbin/modinfo"))
                if path.is_file()
            ),
            None,
        )
        if module_names and modinfo is None:
            raise RuntimeEnvironmentError("modinfo is unavailable for GPU modules")
        if modinfo is not None:
            roots.extend(_tool_execution_roots("modinfo-executable", modinfo))
        for index, module_name in enumerate(module_names):
            assert modinfo is not None
            try:
                completed = subprocess.run(
                    [modinfo.as_posix(), "-n", module_name],
                    check=False,
                    capture_output=True,
                    env=_probe_environment(),
                    cwd=publication_runtime_working_directory(),
                    timeout=30,
                )
                raw_path = completed.stdout.decode("utf-8", errors="strict").strip()
            except (OSError, UnicodeError, subprocess.TimeoutExpired) as error:
                raise RuntimeEnvironmentError(
                    "GPU kernel module artifact discovery failed"
                ) from error
            if (
                completed.returncode != 0
                or not raw_path
                or not Path(raw_path).is_absolute()
            ):
                raise RuntimeEnvironmentError(
                    "GPU kernel module artifact discovery failed"
                )
            try:
                module_path = Path(raw_path).resolve(strict=True)
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "GPU kernel module artifact is unavailable"
                ) from error
            roots.append((f"gpu-kernel-module-{index:03d}", "file", module_path))
    return roots


def _elf_runtime_search_roots(
    candidates: Sequence[Path],
    covered_roots: tuple[
        tuple[str, Literal["directory", "file", "search-directory"], Path], ...
    ],
) -> tuple[list[tuple[str, str, Path]], bytes]:
    """Bind external DT_RPATH/DT_RUNPATH directories for optional DSOs."""

    readelf = Path("/usr/bin/readelf")
    if not readelf.is_file():
        raise RuntimeEnvironmentError("readelf is unavailable")
    covered_directories = tuple(
        path.resolve(strict=True)
        for _role, kind, path in covered_roots
        if kind == "directory"
    )

    def beneath(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    search_paths: set[tuple[Path, Path]] = set()
    transcript = bytearray()
    for offset in range(0, len(candidates), 64):
        batch = tuple(candidates[offset : offset + 64])
        if not batch:
            continue
        try:
            completed = subprocess.run(
                [readelf.as_posix(), "-d", *(path.as_posix() for path in batch)],
                check=False,
                capture_output=True,
                env=_probe_environment(),
                cwd=publication_runtime_working_directory(),
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeEnvironmentError(
                "native runtime search-path discovery failed"
            ) from error
        if (
            completed.returncode != 0
            or len(completed.stdout) > _MAX_PROBE_BYTES
            or len(completed.stderr) > _MAX_PROBE_BYTES
        ):
            raise RuntimeEnvironmentError("native runtime search-path discovery failed")
        try:
            output = completed.stdout.decode("utf-8", errors="strict")
            completed.stderr.decode("utf-8", errors="strict")
        except UnicodeError as error:
            raise RuntimeEnvironmentError(
                "native runtime search-path output is not UTF-8"
            ) from error
        transcript.extend(len(completed.stdout).to_bytes(8, "little"))
        transcript.extend(completed.stdout)
        current = batch[0] if len(batch) == 1 else None
        for line in output.splitlines():
            header = re.fullmatch(r"File: (.+)", line)
            if header is not None:
                selected = Path(header.group(1))
                current = selected if selected in batch else None
                continue
            match = re.search(
                r"\((?:RPATH|RUNPATH)\).*Library (?:rpath|runpath): \[(.*)\]",
                line,
            )
            if match is None:
                continue
            if current is None:
                raise RuntimeEnvironmentError(
                    "native runtime search-path owner is ambiguous"
                )
            for raw_entry in match.group(1).split(os.pathsep):
                expanded = (
                    publication_runtime_working_directory().as_posix()
                    if not raw_entry
                    else raw_entry.replace(
                        "${ORIGIN}", current.parent.as_posix()
                    ).replace("$ORIGIN", current.parent.as_posix())
                )
                if "$" in expanded:
                    raise RuntimeEnvironmentError(
                        "native runtime search path uses an unsupported token"
                    )
                selected = Path(expanded)
                if not selected.is_absolute():
                    raise RuntimeEnvironmentError(
                        "native runtime search path is not absolute"
                    )
                lexical = Path(os.path.abspath(selected))
                if not lexical.exists():
                    if any(beneath(lexical, root) for root in covered_directories):
                        continue
                    ancestor = lexical.parent
                    while not ancestor.exists() and ancestor != ancestor.parent:
                        ancestor = ancestor.parent
                    try:
                        resolved_ancestor = ancestor.resolve(strict=True)
                    except OSError as error:
                        raise RuntimeEnvironmentError(
                            "native runtime search path has no stable ancestor"
                        ) from error
                    if not resolved_ancestor.is_dir():
                        raise RuntimeEnvironmentError(
                            "native runtime search path ancestor is not a directory"
                        )
                    search_paths.add((ancestor, resolved_ancestor))
                    continue
                try:
                    resolved = lexical.resolve(strict=True)
                except OSError as error:
                    raise RuntimeEnvironmentError(
                        "native runtime search path is unavailable"
                    ) from error
                if not resolved.is_dir():
                    raise RuntimeEnvironmentError(
                        "native runtime search path is not a directory"
                    )
                if not any(beneath(resolved, root) for root in covered_directories):
                    search_paths.add((lexical, resolved))
    roots = _tool_execution_roots("readelf-executable", readelf)
    for index, (lexical, resolved) in enumerate(
        sorted(
            search_paths,
            key=lambda value: (os.fsencode(value[0]), os.fsencode(value[1])),
        )
    ):
        roots.append(
            (
                f"native-search-directory-{index:04d}",
                "search-directory",
                lexical,
            )
        )
        if resolved != lexical:
            roots.append(
                (
                    f"native-search-directory-target-{index:04d}",
                    "search-directory",
                    resolved,
                )
            )
    return roots, bytes(transcript)


def _with_native_dependency_roots(
    raw_roots: object,
    executable_target: Path,
    *,
    additional_entrypoints: Sequence[Path] = (),
) -> tuple[list[dict[str, str]], str]:
    validated = _validated_runtime_root_paths(raw_roots)
    candidates = _runtime_elf_candidates(validated)
    required_candidates: set[Path] = set()
    for raw_entrypoint in (executable_target, *additional_entrypoints):
        if not isinstance(raw_entrypoint, Path):
            raise TypeError("native runtime entrypoints must be pathlib.Path values")
        entrypoint = raw_entrypoint.resolve(strict=True)
        if _dynamic_elf(entrypoint):
            candidates.add(entrypoint)
            required_candidates.add(entrypoint)
    host_roots, ldconfig_output, logical_libraries = _host_loader_roots()
    nvidia_smi_path = nvidia_smi_executable()
    nvidia_smi = None if nvidia_smi_path is None else _executable(nvidia_smi_path)
    gpu_roots = _gpu_driver_roots(logical_libraries, nvidia_smi)
    if nvidia_smi is not None and _dynamic_elf(nvidia_smi.target):
        candidates.add(nvidia_smi.target)
        required_candidates.add(nvidia_smi.target)
    ldd = Path("/usr/bin/ldd")
    ldd_roots = _tool_execution_roots("ldd-executable", ldd)
    tool_roots = [*host_roots, *gpu_roots, *ldd_roots]
    for role, kind, path in tool_roots:
        if kind != "file" or not ("executable" in role or "bootstrap" in role):
            continue
        target = path.resolve(strict=True)
        if _dynamic_elf(target):
            candidates.add(target)
            required_candidates.add(target)
    readelf_target = Path("/usr/bin/readelf").resolve(strict=True)
    if _dynamic_elf(readelf_target):
        candidates.add(readelf_target)
        required_candidates.add(readelf_target)
    linkage_digest = hashlib.sha256(b"maskimpute-native-linkage-resolution-v2\0")
    linkage_digest.update(len(ldconfig_output).to_bytes(8, "little"))
    linkage_digest.update(ldconfig_output)
    dependencies: dict[Path, Path] = {}
    candidate_list = sorted(candidates, key=os.fsencode)
    optional_candidates = [
        path for path in candidate_list if path not in required_candidates
    ]
    pending_batches: list[tuple[bool, list[Path]]] = [
        (True, sorted(required_candidates, key=os.fsencode)),
        *(
            (False, optional_candidates[offset : offset + 64])
            for offset in range(0, len(optional_candidates), 64)
        ),
    ]
    processed_candidates: set[Path] = set()
    while pending_batches:
        linkage_required, raw_batch = pending_batches.pop(0)
        batch = [path for path in raw_batch if path not in processed_candidates]
        if not batch:
            continue
        processed_candidates.update(batch)
        try:
            completed = subprocess.run(
                [ldd.as_posix(), *(path.as_posix() for path in batch)],
                check=False,
                capture_output=True,
                env=_probe_environment(),
                cwd=publication_runtime_working_directory(),
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
        unresolved = set(re.findall(r"^\s*(\S+)\s+=>\s+not found\s*$", text, re.M))
        if linkage_required and unresolved:
            raise RuntimeEnvironmentError("unresolved native linkage")
        combined = f"{text}\n{stderr}".casefold()
        if completed.returncode != 0 and "not a dynamic executable" not in combined:
            raise RuntimeEnvironmentError("native runtime dependency discovery failed")
        normalized = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
        normalized_stderr = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", stderr)
        linkage_digest.update(completed.returncode.to_bytes(4, "little", signed=True))
        linkage_digest.update(normalized.encode("utf-8"))
        linkage_digest.update(normalized_stderr.encode("utf-8"))
        for match in re.finditer(r"(?:=>\s*)?(/[^\s()]+)", text):
            if match.group(1).endswith(":"):
                continue
            raw_dependency = Path(match.group(1))
            try:
                dependency = raw_dependency.resolve(strict=True)
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "native runtime dependency disappeared"
                ) from error
            if not dependency.is_file():
                raise RuntimeEnvironmentError("native runtime dependency is not a file")
            dependencies[raw_dependency] = dependency
        newly_discovered = sorted(
            set(dependencies.values()) - processed_candidates,
            key=os.fsencode,
        )
        if len(processed_candidates) + len(newly_discovered) > _MAX_ENVIRONMENT_ENTRIES:
            raise RuntimeEnvironmentError(
                "native runtime dependency closure is too large"
            )
        pending_batches.extend(
            (False, newly_discovered[offset : offset + 64])
            for offset in range(0, len(newly_discovered), 64)
        )

    all_candidates = sorted(
        processed_candidates.union(dependencies.values()), key=os.fsencode
    )
    search_roots, readelf_output = _elf_runtime_search_roots(all_candidates, validated)
    linkage_digest.update(len(readelf_output).to_bytes(8, "little"))
    linkage_digest.update(readelf_output)

    result = [
        {"role": role, "kind": kind, "path": path.absolute().as_posix()}
        for role, kind, path in validated
    ]
    existing_paths = {Path(entry["path"]).absolute() for entry in result}
    for role, kind, path in [
        *host_roots,
        *gpu_roots,
        *ldd_roots,
        *search_roots,
    ]:
        result.append({"role": role, "kind": kind, "path": path.absolute().as_posix()})
        existing_paths.add(path.absolute())
    dependency_index = 0
    for raw_path, target in sorted(
        dependencies.items(),
        key=lambda item: (os.fsencode(item[0]), os.fsencode(item[1])),
    ):
        for suffix, path in (("", raw_path), ("-target", target)):
            absolute = path.absolute()
            if absolute in existing_paths:
                continue
            result.append(
                {
                    "role": f"native-dependency{suffix}-{dependency_index:04d}",
                    "kind": "file",
                    "path": absolute.as_posix(),
                }
            )
            existing_paths.add(absolute)
        dependency_index += 1
    symlink_additions: list[dict[str, str]] = []
    for entry in tuple(result):
        chain, final_target = _lexical_symlink_chain(Path(entry["path"]))
        candidates = [alias for alias, _target_text, _identity in chain]
        candidates.append(final_target)
        hop_index = 0
        for candidate in candidates:
            absolute = candidate.absolute()
            if absolute in existing_paths:
                continue
            symlink_additions.append(
                {
                    "role": f"{entry['role']}-symlink-hop-{hop_index:02d}",
                    "kind": entry["kind"],
                    "path": absolute.as_posix(),
                }
            )
            existing_paths.add(absolute)
            hop_index += 1
    result.extend(symlink_additions)
    if len({entry["role"] for entry in result}) != len(result):
        raise RuntimeEnvironmentError("native runtime root roles are duplicated")
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
                else (
                    _search_directory_content_sha256(path)
                    if kind == "search-directory"
                    else _runtime_file_content_sha256(path)
                )
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


def _runtime_closure_paths_sha256(
    executable: _ExecutableIdentity,
    raw_roots: object,
) -> str:
    roots = [
        {
            "role": role,
            "kind": kind,
            "path": path.absolute().as_posix(),
            "target": path.resolve(strict=True).as_posix(),
        }
        for role, kind, path in _validated_runtime_root_paths(raw_roots)
    ]
    return canonical_sha256(
        {
            "schema": "maskimpute-runtime-closure-paths-v1",
            "executable_invocation": executable.invocation.as_posix(),
            "executable_target": executable.target.as_posix(),
            "roots": roots,
        }
    )


@dataclass(frozen=True, slots=True)
class _ExecutableIdentity:
    invocation: Path
    target: Path
    launcher_kind: Literal["regular", "symlink"]
    launcher_sha256: str
    invocation_state: tuple[int, int, int, int, int, int, int]
    target_state: tuple[int, int, int, int, int, int, int]
    symlink_chain: tuple[
        tuple[
            Path,
            bytes,
            tuple[int, int, int, int, int, int, int],
        ],
        ...,
    ]


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


def _lexical_symlink_chain(
    path: Path,
) -> tuple[
    tuple[
        tuple[
            Path,
            bytes,
            tuple[int, int, int, int, int, int, int],
        ],
        ...,
    ],
    Path,
]:
    current = path.absolute()
    chain: list[
        tuple[
            Path,
            bytes,
            tuple[int, int, int, int, int, int, int],
        ]
    ] = []
    seen: set[Path] = set()
    for _depth in range(40):
        if current in seen:
            raise RuntimeEnvironmentError("runtime symlink chain contains a cycle")
        seen.add(current)
        try:
            before = current.lstat()
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime symlink chain is unavailable"
            ) from error
        if not stat.S_ISLNK(before.st_mode):
            return tuple(chain), current.resolve(strict=True)
        try:
            raw_target = os.readlink(current)
            target_bytes = os.fsencode(raw_target)
            after = current.lstat()
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime symlink chain changed while reading"
            ) from error
        identity = _stat_identity(before)
        if _stat_identity(after) != identity:
            raise RuntimeEnvironmentError("runtime symlink chain changed while reading")
        chain.append((current, target_bytes, identity))
        target_path = Path(raw_target)
        current = Path(
            os.path.abspath(
                target_path
                if target_path.is_absolute()
                else current.parent / target_path
            )
        )
    raise RuntimeEnvironmentError("runtime symlink chain is too deep")


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
        symlink_chain, target = _lexical_symlink_chain(invocation)
        invocation_metadata = invocation.lstat()
        target_metadata = target.stat()
    except OSError as error:
        raise RuntimeEnvironmentError("runtime executable is unavailable") from error
    if not stat.S_ISREG(target_metadata.st_mode) or not os.access(invocation, os.X_OK):
        raise RuntimeEnvironmentError("runtime executable is not an executable file")
    if stat.S_ISLNK(invocation_metadata.st_mode):
        launcher_kind: Literal["regular", "symlink"] = "symlink"
        launcher_digest = hashlib.sha256(b"maskimpute-runtime-launcher-symlink-v2\0")
        for alias, target_text, _identity in symlink_chain:
            encoded = os.fsencode(alias)
            launcher_digest.update(len(encoded).to_bytes(8, "little"))
            launcher_digest.update(encoded)
            launcher_digest.update(len(target_text).to_bytes(8, "little"))
            launcher_digest.update(target_text)
        launcher_sha256 = launcher_digest.hexdigest()
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
        symlink_chain=symlink_chain,
    )


def _python_shebang_bootstrap(
    executable: _ExecutableIdentity,
) -> tuple[
    tuple[_ExecutableIdentity, ...],
    list[dict[str, str]],
    tuple[_ExecutableIdentity, ...],
]:
    """Bind every interpreter and search directory used before Python starts."""

    discovered: list[_ExecutableIdentity] = []
    python_interpreters: list[_ExecutableIdentity] = []
    search_directories: list[Path] = []
    script_directories: list[Path] = []
    pending = [executable]
    visited_targets: set[Path] = set()

    def append_executable(value: _ExecutableIdentity) -> None:
        if value.invocation not in {item.invocation for item in discovered}:
            discovered.append(value)

    def append_python(value: _ExecutableIdentity) -> bool:
        names = {value.invocation.name.casefold(), value.target.name.casefold()}
        if not any(
            re.fullmatch(r"(?:python|pypy)\d*(?:\.\d+)*", name) for name in names
        ):
            return False
        if value.invocation not in {item.invocation for item in python_interpreters}:
            python_interpreters.append(value)
        return True

    for _depth in range(8):
        if not pending:
            break
        current = pending.pop(0)
        if current.target in visited_targets:
            continue
        visited_targets.add(current.target)
        payload = _secure_regular_file_bytes(current.target)
        if payload.startswith(b"\x7fELF"):
            continue
        first_line = payload.split(b"\n", 1)[0]
        if not first_line.startswith(b"#!") or len(first_line) > 255:
            raise RuntimeEnvironmentError(
                "Python runtime wrapper has an invalid shebang"
            )
        try:
            declaration = first_line[2:].decode("utf-8", errors="strict").strip()
        except UnicodeError as error:
            raise RuntimeEnvironmentError(
                "Python runtime wrapper shebang is not UTF-8"
            ) from error
        fields = declaration.split(maxsplit=1)
        if not fields or not Path(fields[0]).is_absolute():
            raise RuntimeEnvironmentError(
                "Python runtime wrapper shebang interpreter must be absolute"
            )
        for directory in (current.invocation.parent, current.target.parent):
            if directory not in script_directories:
                script_directories.append(directory)
        interpreter = _executable(Path(fields[0]))
        append_executable(interpreter)
        if interpreter.invocation.name != "env" and interpreter.target.name != "env":
            if not append_python(interpreter):
                raise RuntimeEnvironmentError(
                    "Python runtime wrapper must use a Python shebang interpreter"
                )
            pending.append(interpreter)
            continue

        if len(fields) != 2:
            raise RuntimeEnvironmentError("Python runtime env shebang lacks a command")
        raw_command = fields[1]
        if raw_command.startswith("-S"):
            if raw_command == "-S":
                raise RuntimeEnvironmentError(
                    "Python runtime env shebang lacks a command"
                )
            try:
                command_fields = shlex.split(raw_command[2:].lstrip())
            except ValueError as error:
                raise RuntimeEnvironmentError(
                    "Python runtime env shebang command is invalid"
                ) from error
        else:
            if any(character.isspace() for character in raw_command):
                raise RuntimeEnvironmentError(
                    "Python runtime env shebang requires -S for command arguments"
                )
            command_fields = [raw_command]
        command = command_fields[0] if command_fields else ""
        if not command or command.startswith("-") or "=" in command:
            raise RuntimeEnvironmentError(
                "Python runtime env shebang command is unsupported"
            )
        if os.path.sep in command:
            selected_path = Path(command)
            if not selected_path.is_absolute():
                raise RuntimeEnvironmentError(
                    "Python runtime env shebang command must be absolute"
                )
        else:
            environment = _libc_environment_mapping()
            raw_path = environment.get("PATH")
            if raw_path is None:
                raise RuntimeEnvironmentError(
                    "Python runtime env shebang requires an explicit PATH"
                )
            selected_text = shutil.which(command, path=raw_path)
            if selected_text is None:
                raise RuntimeEnvironmentError(
                    "Python runtime env shebang command is unavailable"
                )
            selected_path = Path(selected_text).absolute()
            selected_directory = selected_path.parent
            matched = False
            for raw_directory in raw_path.split(os.pathsep):
                if not raw_directory or not Path(raw_directory).is_absolute():
                    raise RuntimeEnvironmentError(
                        "PATH entries used by Python wrappers must be nonempty absolute paths"
                    )
                directory = Path(os.path.abspath(raw_directory))
                try:
                    if not directory.is_dir():
                        raise RuntimeEnvironmentError(
                            "PATH directory used by Python wrapper is unavailable"
                        )
                except OSError as error:
                    raise RuntimeEnvironmentError(
                        "PATH directory used by Python wrapper is unavailable"
                    ) from error
                if directory not in search_directories:
                    search_directories.append(directory)
                if directory == selected_directory:
                    matched = True
                    break
            if not matched:
                raise RuntimeEnvironmentError(
                    "Python runtime env shebang resolution is inconsistent"
                )
        selected_command = _executable(selected_path)
        append_executable(selected_command)
        if not append_python(selected_command):
            raise RuntimeEnvironmentError(
                "Python runtime env shebang must select a Python interpreter"
            )
        pending.append(selected_command)
    else:
        raise RuntimeEnvironmentError(
            "Python runtime wrapper shebang chain is too deep"
        )

    roots = _python_executable_roots(
        discovered, role_prefix="python-bootstrap-entrypoint"
    )
    roots.extend(
        {
            "role": f"python-bootstrap-script-directory-{index:03d}",
            "kind": "directory",
            "path": path.absolute().as_posix(),
        }
        for index, path in enumerate(script_directories)
    )
    roots.extend(
        {
            "role": f"python-bootstrap-search-directory-{index:03d}",
            "kind": "search-directory",
            "path": path.absolute().as_posix(),
        }
        for index, path in enumerate(search_directories)
    )
    return tuple(discovered), roots, tuple(python_interpreters)


def _validated_python_runtime_executables(
    raw_receipts: object,
) -> tuple[_ExecutableIdentity, ...]:
    if not isinstance(raw_receipts, list) or not raw_receipts:
        raise RuntimeEnvironmentError("Python runtime executables are unavailable")
    result: list[_ExecutableIdentity] = []
    seen: set[Path] = set()
    expected_fields = {
        "invocation",
        "invocation_state",
        "target",
        "target_sha256",
        "target_state",
    }
    for receipt in raw_receipts:
        if not isinstance(receipt, dict) or set(receipt) != expected_fields:
            raise RuntimeEnvironmentError(
                "Python runtime executable receipt is invalid"
            )
        invocation = receipt.get("invocation")
        raw_target = receipt.get("target")
        target_sha256 = receipt.get("target_sha256")
        invocation_state = receipt.get("invocation_state")
        target_state = receipt.get("target_state")
        if (
            not isinstance(invocation, str)
            or not Path(invocation).is_absolute()
            or not isinstance(raw_target, str)
            or not Path(raw_target).is_absolute()
            or not isinstance(target_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", target_sha256) is None
            or not isinstance(invocation_state, list)
            or len(invocation_state) != 7
            or any(type(value) is not int for value in invocation_state)
            or not isinstance(target_state, list)
            or len(target_state) != 7
            or any(type(value) is not int for value in target_state)
        ):
            raise RuntimeEnvironmentError(
                "Python runtime executable receipt is invalid"
            )
        selected = _executable(Path(invocation))
        if (
            selected.invocation in seen
            or selected.target != Path(raw_target)
            or selected.invocation_state != tuple(invocation_state)
            or selected.target_state != tuple(target_state)
            or _file_sha256(selected.target) != target_sha256
        ):
            raise RuntimeEnvironmentError(
                "Python runtime executable changed after target probing"
            )
        seen.add(selected.invocation)
        result.append(selected)
    return tuple(result)


def _python_executable_roots(
    executables: Sequence[_ExecutableIdentity],
    *,
    role_prefix: str = "python-entrypoint",
) -> list[dict[str, str]]:
    roots: list[dict[str, str]] = []
    for index, executable in enumerate(executables):
        roots.append(
            {
                "role": f"{role_prefix}-{index:03d}",
                "kind": "file",
                "path": executable.invocation.as_posix(),
            }
        )
        if executable.target != executable.invocation:
            roots.append(
                {
                    "role": f"{role_prefix}-target-{index:03d}",
                    "kind": "file",
                    "path": executable.target.as_posix(),
                }
            )
        for hop_index, (alias, _target_text, _identity) in enumerate(
            executable.symlink_chain[1:], start=1
        ):
            if alias in {executable.invocation, executable.target}:
                continue
            roots.append(
                {
                    "role": (f"{role_prefix}-symlink-hop-{index:03d}-{hop_index:02d}"),
                    "kind": "file",
                    "path": alias.as_posix(),
                }
            )
    return roots


def _publication_controller_python_roots(
    executable: _ExecutableIdentity,
    covered_runtime_roots: object,
) -> list[dict[str, str]]:
    try:
        current_target = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise RuntimeEnvironmentError(
            "controller Python executable is unavailable"
        ) from error
    if executable.target != current_target:
        return []
    repository = Path(__file__).resolve().parents[1]
    selected: dict[Path, str] = {repository: "search-directory"}
    covered = _validated_runtime_root_paths(covered_runtime_roots)

    def beneath(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    for raw_path in publication_python_spawn_search_path():
        path = Path(raw_path)
        resolved = path.resolve(strict=True)
        if any(
            (kind == "directory" and beneath(resolved, root.resolve(strict=True)))
            or (kind == "file" and resolved == root.resolve(strict=True))
            for _role, kind, root in covered
        ):
            continue
        if path != repository:
            selected[path] = "directory" if path.is_dir() else "file"
    for relative in ("maskimpute", "maskimpute_benchmark"):
        path = repository / relative
        if path.is_dir():
            selected[path] = "directory"
    fixed_cwd = publication_runtime_working_directory()
    selected[fixed_cwd] = "directory"
    return [
        {
            "role": f"python-controller-search-root-{index:03d}",
            "kind": kind,
            "path": path.as_posix(),
        }
        for index, (path, kind) in enumerate(
            sorted(selected.items(), key=lambda item: os.fsencode(item[0]))
        )
    ]


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
    for alias, target_text, identity in executable.symlink_chain:
        try:
            if (
                _stat_identity(alias.lstat()) != identity
                or os.fsencode(os.readlink(alias)) != target_text
            ):
                raise RuntimeEnvironmentError(
                    "runtime executable symlink chain changed during inventory"
                )
        except OSError as error:
            raise RuntimeEnvironmentError(
                "runtime executable symlink chain changed during inventory"
            ) from error


def _runtime_root_identity_sha256(
    path: Path,
    *,
    observed_paths: dict[Path, tuple[int, int, int, int, int, int, int]] | None = None,
) -> str:
    cache: dict[Path, str] = {}
    observed: dict[Path, tuple[int, int, int, int, int, int, int]] = {}

    def remember(item: Path, metadata: os.stat_result) -> None:
        identity = _stat_identity(metadata)
        previous = observed.setdefault(item, identity)
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
                raise RuntimeEnvironmentError(
                    "runtime identity root is not a directory"
                )
            remember(directory, before)
            children = sorted(
                os.scandir(directory), key=lambda item: os.fsencode(item.name)
            )
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
                        digest.update(
                            repr(_stat_identity(target_metadata)).encode("ascii")
                        )
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
            raise RuntimeEnvironmentError(
                "runtime identity file is unavailable"
            ) from error
        if _stat_identity(metadata) != _stat_identity(after):
            raise RuntimeEnvironmentError("runtime identity file changed")
        result = digest.hexdigest()
    try:
        for observed_path, expected in observed.items():
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
    if observed_paths is not None:
        for observed_path, identity in observed.items():
            previous = observed_paths.setdefault(observed_path, identity)
            if previous != identity:
                raise RuntimeEnvironmentError(
                    "runtime identity snapshots disagree on a path"
                )
    return result


def _runtime_search_directory_identity_sha256(
    path: Path,
    *,
    observed_paths: dict[Path, tuple[int, int, int, int, int, int, int]],
) -> str:
    try:
        alias_before = path.lstat()
        target = path.resolve(strict=True)
        before = target.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeEnvironmentError("runtime search identity is not a directory")
        local: dict[Path, tuple[int, int, int, int, int, int, int]] = {
            path: _stat_identity(alias_before),
            target: _stat_identity(before),
        }
        children = sorted(os.scandir(target), key=lambda item: os.fsencode(item.name))
        digest = hashlib.sha256(b"maskimpute-runtime-search-identity-v1\0")
        if stat.S_ISLNK(alias_before.st_mode):
            digest.update(os.fsencode(os.readlink(path)))
        for child_entry in children:
            encoded = os.fsencode(child_entry.name)
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        after = target.lstat()
        alias_after = path.lstat()
        names_after = sorted(
            (entry.name for entry in os.scandir(target)), key=os.fsencode
        )
        if (
            _stat_identity(alias_before) != _stat_identity(alias_after)
            or _stat_identity(before) != _stat_identity(after)
            or [entry.name for entry in children] != names_after
            or any(
                _stat_identity(item.lstat()) != identity
                for item, identity in local.items()
            )
        ):
            raise RuntimeEnvironmentError(
                "runtime search identity changed during traversal"
            )
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError(
            "runtime search identity changed during traversal"
        ) from error
    for item, identity in local.items():
        previous = observed_paths.setdefault(item, identity)
        if previous != identity:
            raise RuntimeEnvironmentError(
                "runtime identity snapshots disagree on a path"
            )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentSnapshot:
    identity_sha256: str
    closure_paths_sha256: str
    nvidia_smi_path: str | None
    path_identities: tuple[tuple[str, tuple[int, int, int, int, int, int, int]], ...]
    watch_specs: tuple[tuple[str, bool, tuple[str, ...]], ...]
    control_file_sha256s: tuple[tuple[str, str], ...]


def _snapshot_components(
    observed: dict[Path, tuple[int, int, int, int, int, int, int]],
    content_directories: set[Path],
) -> tuple[
    tuple[tuple[str, tuple[int, int, int, int, int, int, int]], ...],
    tuple[tuple[str, bool, tuple[str, ...]], ...],
    tuple[tuple[str, str], ...],
]:
    for path in tuple(observed):
        for parent in path.parents:
            if parent in observed:
                continue
            try:
                metadata = parent.lstat()
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "runtime path component is unavailable"
                ) from error
            identity = _stat_identity(metadata)
            observed[parent] = (*identity[:3], -1, -1, -1, -1)

    rules: dict[Path, tuple[bool, set[str]]] = {}

    def rule(path: Path) -> tuple[bool, set[str]]:
        return rules.setdefault(path, (False, set()))

    for path, identity in observed.items():
        if stat.S_ISDIR(identity[2]) or path.parent not in content_directories:
            all_children, names = rule(path)
            if path in content_directories and stat.S_ISDIR(identity[2]):
                rules[path] = (True, names)
        parent = path.parent
        if parent != path:
            parent_all, parent_names = rule(parent)
            parent_names.add(path.name)
            rules[parent] = (parent_all, parent_names)

    identities = tuple(
        sorted(
            ((path.as_posix(), identity) for path, identity in observed.items()),
            key=lambda item: os.fsencode(item[0]),
        )
    )
    watch_specs = tuple(
        (
            path.as_posix(),
            all_children,
            tuple(sorted(names, key=os.fsencode)),
        )
        for path, (all_children, names) in sorted(
            rules.items(), key=lambda item: os.fsencode(item[0])
        )
    )
    control_files: list[tuple[str, str]] = []
    for path, identity in observed.items():
        if stat.S_ISREG(identity[2]) and (
            path == Path("/proc")
            or Path("/proc") in path.parents
            or path == Path("/sys")
            or Path("/sys") in path.parents
        ):
            control_files.append((path.as_posix(), _control_file_sha256(path)))
    return (
        identities,
        watch_specs,
        tuple(sorted(control_files, key=lambda item: os.fsencode(item[0]))),
    )


def runtime_environment_snapshot(
    kind: Literal["python", "r"],
    executable: Path,
    *,
    r_library_paths: tuple[Path, ...] = (),
) -> RuntimeEnvironmentSnapshot:
    """Discover one complete closure once and retain cheap monitored identities."""

    selected = _executable(executable)
    libraries = _r_library_paths(r_library_paths) if kind == "r" else ()
    if kind == "python":
        raw_roots, native_linkage_sha256 = _python_runtime_root_paths(selected)
    elif kind == "r":
        raw_roots, native_linkage_sha256 = _r_runtime_root_paths(selected, libraries)
    else:
        raise RuntimeEnvironmentError("runtime kind must be python or r")
    closure_paths_sha256 = _runtime_closure_paths_sha256(selected, raw_roots)
    observed: dict[Path, tuple[int, int, int, int, int, int, int]] = {
        selected.invocation: selected.invocation_state,
        selected.target: selected.target_state,
    }
    observed.update(
        (alias, identity) for alias, _target_text, identity in selected.symlink_chain
    )
    roots: list[dict[str, str]] = []
    content_directories: set[Path] = set()
    nvidia_smi_path: str | None = None
    for role, root_kind, path in _validated_runtime_root_paths(raw_roots):
        if role == "nvidia-smi-executable":
            nvidia_smi_path = path.absolute().as_posix()
        before_paths = set(observed)
        identity_sha256 = (
            _runtime_search_directory_identity_sha256(path, observed_paths=observed)
            if root_kind == "search-directory"
            else _runtime_root_identity_sha256(path, observed_paths=observed)
        )
        if root_kind == "search-directory":
            content_directories.add(path.resolve(strict=True))
            if path.is_dir() and not path.is_symlink():
                content_directories.add(path)
        else:
            content_directories.update(
                item
                for item in set(observed) - before_paths
                if stat.S_ISDIR(observed[item][2])
                and not stat.S_ISLNK(observed[item][2])
            )
        roots.append(
            {
                "role": role,
                "kind": root_kind,
                "identity_sha256": identity_sha256,
            }
        )
    _revalidate_executable(selected)
    identities, watch_specs, control_files = _snapshot_components(
        observed, content_directories
    )
    identity_sha256 = canonical_sha256(
        {
            "schema": "maskimpute-runtime-metadata-snapshot-v2",
            "kind": kind,
            "launcher_kind": selected.launcher_kind,
            "launcher_sha256": selected.launcher_sha256,
            "invocation_state": selected.invocation_state,
            "target_state": selected.target_state,
            "native_linkage_sha256": native_linkage_sha256,
            "roots": roots,
            "path_identity_sha256": canonical_sha256(identities),
            "control_file_sha256s": control_files,
        }
    )
    return RuntimeEnvironmentSnapshot(
        identity_sha256=identity_sha256,
        closure_paths_sha256=closure_paths_sha256,
        nvidia_smi_path=nvidia_smi_path,
        path_identities=identities,
        watch_specs=watch_specs,
        control_file_sha256s=control_files,
    )


def merge_runtime_environment_snapshots(
    snapshots: Sequence[RuntimeEnvironmentSnapshot],
    *,
    additional_files: Sequence[Path] = (),
) -> RuntimeEnvironmentSnapshot:
    observed: dict[Path, tuple[int, int, int, int, int, int, int]] = {}
    content_directories: set[Path] = set()
    control_files: dict[str, str] = {}
    identity_sha256s: list[str] = []
    closure_paths_sha256s: list[str] = []
    nvidia_smi_paths: set[str | None] = set()
    for snapshot in snapshots:
        if not isinstance(snapshot, RuntimeEnvironmentSnapshot):
            raise TypeError(
                "runtime snapshots must be RuntimeEnvironmentSnapshot values"
            )
        identity_sha256s.append(snapshot.identity_sha256)
        closure_paths_sha256s.append(snapshot.closure_paths_sha256)
        nvidia_smi_paths.add(snapshot.nvidia_smi_path)
        for raw_path, identity in snapshot.path_identities:
            path = Path(raw_path)
            previous = observed.get(path)
            if previous is None:
                observed[path] = identity
            elif previous[:3] != identity[:3]:
                raise RuntimeEnvironmentError(
                    "runtime snapshots disagree on path identity"
                )
            elif previous[3:] == (-1, -1, -1, -1) and identity[3:] != (
                -1,
                -1,
                -1,
                -1,
            ):
                observed[path] = identity
        for raw_path, all_children, _names in snapshot.watch_specs:
            if all_children:
                content_directories.add(Path(raw_path))
        for raw_path, digest in snapshot.control_file_sha256s:
            previous_digest = control_files.setdefault(raw_path, digest)
            if previous_digest != digest:
                raise RuntimeEnvironmentError(
                    "runtime snapshots disagree on control file content"
                )
    for path in additional_files:
        if not isinstance(path, Path):
            raise TypeError("additional runtime snapshot files must be pathlib.Path")
        absolute = path.absolute()
        try:
            metadata = absolute.lstat()
        except OSError as error:
            raise RuntimeEnvironmentError(
                "additional runtime snapshot file is unavailable"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeEnvironmentError(
                "additional runtime snapshot path is not a regular file"
            )
        observed[absolute] = _stat_identity(metadata)
    identities, watch_specs, discovered_controls = _snapshot_components(
        observed, content_directories
    )
    for raw_path, digest in discovered_controls:
        control_files.setdefault(raw_path, digest)
    merged_controls = tuple(
        sorted(control_files.items(), key=lambda item: os.fsencode(item[0]))
    )
    if len(nvidia_smi_paths) != 1:
        raise RuntimeEnvironmentError(
            "runtime snapshots disagree on nvidia-smi executable"
        )
    return RuntimeEnvironmentSnapshot(
        identity_sha256=canonical_sha256(
            {
                "schema": "maskimpute-merged-runtime-snapshot-v1",
                "runtime_identity_sha256s": sorted(identity_sha256s),
                "path_identities": identities,
                "control_file_sha256s": merged_controls,
            }
        ),
        closure_paths_sha256=canonical_sha256(
            {
                "schema": "maskimpute-merged-runtime-closure-paths-v1",
                "closure_paths_sha256s": sorted(closure_paths_sha256s),
            }
        ),
        nvidia_smi_path=next(iter(nvidia_smi_paths)),
        path_identities=identities,
        watch_specs=watch_specs,
        control_file_sha256s=merged_controls,
    )


def verify_runtime_environment_snapshot(snapshot: RuntimeEnvironmentSnapshot) -> None:
    if not isinstance(snapshot, RuntimeEnvironmentSnapshot):
        raise TypeError("snapshot must be a RuntimeEnvironmentSnapshot")
    try:
        for raw_path, expected in snapshot.path_identities:
            observed = _stat_identity(Path(raw_path).lstat())
            if observed[:3] != expected[:3] or (
                expected[3:] != (-1, -1, -1, -1) and observed != expected
            ):
                raise RuntimeEnvironmentError(
                    f"runtime identity mismatch for {raw_path}"
                )
        verify_runtime_environment_control_files(snapshot)
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError("runtime identity path is unavailable") from error


def verify_runtime_environment_control_files(
    snapshot: RuntimeEnvironmentSnapshot,
) -> None:
    """Rehash synthetic procfs/sysfs bytes that inotify cannot report changing."""

    if not isinstance(snapshot, RuntimeEnvironmentSnapshot):
        raise TypeError("snapshot must be a RuntimeEnvironmentSnapshot")
    for raw_path, expected_sha256 in snapshot.control_file_sha256s:
        if _control_file_sha256(Path(raw_path)) != expected_sha256:
            raise RuntimeEnvironmentError("runtime control file content mismatch")


class RuntimeChangeMonitor:
    """Fail-closed Linux inotify monitor for a prevalidated runtime closure."""

    _CHANGE_MASK = (
        0x00000002  # IN_MODIFY
        | 0x00000004  # IN_ATTRIB
        | 0x00000008  # IN_CLOSE_WRITE
        | 0x00000040  # IN_MOVED_FROM
        | 0x00000080  # IN_MOVED_TO
        | 0x00000100  # IN_CREATE
        | 0x00000200  # IN_DELETE
        | 0x00000400  # IN_DELETE_SELF
        | 0x00000800  # IN_MOVE_SELF
        | 0x00002000  # IN_UNMOUNT
        | 0x02000000  # IN_DONT_FOLLOW
    )
    _SELF_OR_FATAL_MASK = 0x00000400 | 0x00000800 | 0x00002000 | 0x00004000 | 0x00008000

    def __init__(self, watch_specs: Sequence[tuple[str, bool, tuple[str, ...]]]):
        self._descriptor = -1
        self._rules: dict[int, tuple[bool, set[bytes], str]] = {}
        if not watch_specs:
            return
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        add = libc.inotify_add_watch
        add.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add.restype = ctypes.c_int
        descriptor = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
        if descriptor < 0:
            error = ctypes.get_errno()
            raise RuntimeEnvironmentError(
                f"runtime change monitor initialization failed with errno={error}"
            )
        self._descriptor = descriptor
        try:
            for raw_path, all_children, names in watch_specs:
                encoded_path = os.fsencode(raw_path)
                watch = add(descriptor, encoded_path, self._CHANGE_MASK)
                if watch < 0:
                    error = ctypes.get_errno()
                    raise RuntimeEnvironmentError(
                        "runtime change monitor could not watch "
                        f"{raw_path!r} with errno={error}"
                    )
                previous_all, previous_names, previous_path = self._rules.get(
                    watch, (False, set(), raw_path)
                )
                self._rules[watch] = (
                    previous_all or all_children,
                    previous_names.union(os.fsencode(name) for name in names),
                    previous_path,
                )
        except BaseException:
            self.close()
            raise

    def assert_unchanged(self) -> None:
        if self._descriptor < 0:
            return
        while True:
            try:
                payload = os.read(self._descriptor, 1024 * 1024)
            except BlockingIOError:
                return
            except OSError as error:
                raise RuntimeEnvironmentError(
                    "runtime change monitor failed"
                ) from error
            if not payload:
                raise RuntimeEnvironmentError(
                    "runtime change monitor closed unexpectedly"
                )
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < 16:
                    raise RuntimeEnvironmentError(
                        "runtime change monitor event is torn"
                    )
                watch, mask, _cookie, name_length = struct.unpack_from(
                    "iIII", payload, offset
                )
                offset += 16
                if name_length > len(payload) - offset:
                    raise RuntimeEnvironmentError(
                        "runtime change monitor event is torn"
                    )
                raw_name = payload[offset : offset + name_length].split(b"\0", 1)[0]
                offset += name_length
                if mask & 0x00004000:  # IN_Q_OVERFLOW
                    raise RuntimeEnvironmentError("runtime change monitor overflowed")
                rule = self._rules.get(watch)
                if rule is None:
                    raise RuntimeEnvironmentError("runtime change monitor lost a watch")
                all_children, names, raw_path = rule
                if (
                    mask & self._SELF_OR_FATAL_MASK
                    or not raw_name
                    or all_children
                    or raw_name in names
                ):
                    raise RuntimeEnvironmentError(
                        f"runtime changed during execution: {raw_path}"
                    )

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1
            self._rules.clear()

    def __enter__(self) -> RuntimeChangeMonitor:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort descriptor cleanup
        self.close()


def _python_root_probe_components(
    executable: _ExecutableIdentity,
) -> tuple[list[dict[str, str]], tuple[_ExecutableIdentity, ...]]:
    raw = _run_probe(
        [str(executable.invocation), "-c", _PYTHON_ROOT_PROBE],
        "Python",
        exact_environment=True,
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
        if not isinstance(payload, dict) or set(payload) != {
            "distribution_search_paths",
            "missing_search_paths",
            "roots",
            "runtime_executables",
            "runtime_prefixes",
        }:
            raise RuntimeEnvironmentError("Python runtime roots are invalid")
        roots = payload["roots"]
        executables = _validated_python_runtime_executables(
            payload["runtime_executables"]
        )
        _packages, external_roots = _python_distribution_inventory(
            payload["distribution_search_paths"],
            runtime_prefixes=payload["runtime_prefixes"],
            runtime_roots=roots,
            include_content=False,
        )
        if not isinstance(roots, list):
            raise RuntimeEnvironmentError("Python runtime roots are invalid")
        roots.extend(_python_executable_roots(executables))
        roots.extend(_missing_python_search_path_roots(payload["missing_search_paths"]))
        roots.extend(external_roots)
        return roots, executables
    except RuntimeEnvironmentError:
        raise
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeEnvironmentError("Python runtime roots are invalid") from error


def _python_runtime_root_paths(executable: _ExecutableIdentity) -> object:
    bootstrap, bootstrap_roots, bootstrap_pythons = _python_shebang_bootstrap(
        executable
    )
    roots, executables = _python_root_probe_components(executable)
    roots.extend(
        _python_executable_roots((executable,), role_prefix="python-selected-launcher")
    )
    roots.extend(_publication_controller_python_roots(executable, roots))
    roots.extend(bootstrap_roots)
    bootstrap_runtime_executables: list[_ExecutableIdentity] = []
    for index, bootstrap_python in enumerate(bootstrap_pythons):
        runtime_roots, runtime_executables = _python_root_probe_components(
            bootstrap_python
        )
        for root in runtime_roots:
            roots.append(
                {
                    **root,
                    "role": (f"python-bootstrap-runtime-{index:03d}-{root['role']}"),
                }
            )
        bootstrap_runtime_executables.extend(runtime_executables)
    all_executables = (
        *executables,
        *bootstrap,
        *bootstrap_runtime_executables,
    )
    result = _with_native_dependency_roots(
        roots,
        executable.target,
        additional_entrypoints=tuple(value.target for value in all_executables),
    )
    for executable_identity in (executable, *all_executables):
        _revalidate_executable(executable_identity)
    return result


def _r_runtime_root_paths(
    executable: _ExecutableIdentity, library_paths: tuple[Path, ...]
) -> object:
    if not _dynamic_elf(executable.target):
        raise RuntimeEnvironmentError(
            "R runtime executable must resolve to a dynamic ELF"
        )
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
    roots.extend(
        _python_executable_roots((executable,), role_prefix="r-selected-launcher")
    )
    return _with_native_dependency_roots(roots, executable.target)


def runtime_environment_identity_sha256(
    kind: Literal["python", "r"],
    executable: Path,
    *,
    r_library_paths: tuple[Path, ...] = (),
) -> str:
    """Return the complete startup metadata identity for one runtime closure."""
    return runtime_environment_snapshot(
        kind,
        executable,
        r_library_paths=r_library_paths,
    ).identity_sha256


def _probe_environment(
    *,
    r_library_paths: tuple[Path, ...] = (),
    exact_environment: bool = False,
) -> dict[str, str]:
    environment = _libc_environment_mapping()
    _reject_loader_injection(environment)
    if not exact_environment:
        environment.update({"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    if r_library_paths:
        joined = os.pathsep.join(path.as_posix() for path in r_library_paths)
        environment.update(
            {"R_LIBS": joined, "R_LIBS_SITE": joined, "R_LIBS_USER": joined}
        )
    return environment


def _run_probe(
    command: list[str],
    name: str,
    *,
    r_library_paths: tuple[Path, ...] = (),
    exact_environment: bool = False,
) -> bytes:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=publication_runtime_working_directory(),
            env=_probe_environment(
                r_library_paths=r_library_paths,
                exact_environment=exact_environment,
            ),
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
    if (
        not isinstance(native_linkage_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", native_linkage_sha256) is None
    ):
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
        if (
            set(interpreter)
            != {
                "major",
                "minor",
                "platform",
                "library_path_count",
            }
            or any(
                not isinstance(interpreter.get(field), str) or not interpreter[field]
                for field in ("major", "minor", "platform")
            )
            or (
                type(interpreter.get("library_path_count")) is not int
                or interpreter["library_path_count"] < 1
            )
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
            or root_kind not in {"directory", "file", "search-directory"}
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


def _probe_python_environment_with_closure(
    executable: Path,
) -> tuple[dict[str, object], str]:
    """Return a canonical inventory from one selected Python executable."""

    selected = _executable(executable)
    bootstrap, bootstrap_roots, bootstrap_pythons = _python_shebang_bootstrap(selected)
    raw = _run_probe(
        [str(selected.invocation), "-c", _PYTHON_PROBE],
        "Python",
        exact_environment=True,
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
        raise RuntimeEnvironmentError(
            "Python runtime probe returned invalid JSON"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeEnvironmentError("Python runtime probe did not return an object")
    raw_roots = value.pop("_runtime_root_paths", None)
    raw_search_paths = value.pop("_distribution_search_paths", None)
    raw_missing_search_paths = value.pop("_missing_search_paths", None)
    raw_prefixes = value.pop("_runtime_prefixes", None)
    executables = _validated_python_runtime_executables(
        value.pop("_runtime_executables", None)
    )
    packages, external_roots = _python_distribution_inventory(
        raw_search_paths,
        runtime_prefixes=raw_prefixes,
        runtime_roots=raw_roots,
    )
    value["packages"] = packages
    if not isinstance(raw_roots, list):
        raise RuntimeEnvironmentError("Python runtime roots are invalid")
    raw_roots.extend(_python_executable_roots(executables))
    raw_roots.extend(
        _python_executable_roots((selected,), role_prefix="python-selected-launcher")
    )
    raw_roots.extend(_missing_python_search_path_roots(raw_missing_search_paths))
    raw_roots.extend(_publication_controller_python_roots(selected, raw_roots))
    raw_roots.extend(external_roots)
    raw_roots.extend(bootstrap_roots)
    bootstrap_runtime_executables: list[_ExecutableIdentity] = []
    for index, bootstrap_python in enumerate(bootstrap_pythons):
        runtime_roots, runtime_executables = _python_root_probe_components(
            bootstrap_python
        )
        for root in runtime_roots:
            raw_roots.append(
                {
                    **root,
                    "role": (f"python-bootstrap-runtime-{index:03d}-{root['role']}"),
                }
            )
        bootstrap_runtime_executables.extend(runtime_executables)
    all_executables = (
        *executables,
        *bootstrap,
        *bootstrap_runtime_executables,
    )
    raw_roots, native_linkage_sha256 = _with_native_dependency_roots(
        raw_roots,
        selected.target,
        additional_entrypoints=tuple(value.target for value in all_executables),
    )
    closure_paths_sha256 = _runtime_closure_paths_sha256(selected, raw_roots)
    value["runtime_roots"] = _runtime_root_inventory(raw_roots)
    value["native_linkage_sha256"] = native_linkage_sha256
    value["executable_sha256"] = _file_sha256(selected.target)
    value["launcher"] = {
        "kind": selected.launcher_kind,
        "sha256": selected.launcher_sha256,
    }
    for executable_identity in (selected, *all_executables):
        _revalidate_executable(executable_identity)
    return _validate_inventory(value, "python"), closure_paths_sha256


def probe_python_environment(executable: Path) -> dict[str, object]:
    """Return a canonical inventory from one selected Python executable."""

    return _probe_python_environment_with_closure(executable)[0]


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
            raise RuntimeEnvironmentError(
                "R library path must be a non-symlink directory"
            )
        selected.append(absolute)
    if len(selected) != len(set(selected)):
        raise RuntimeEnvironmentError("R library paths are duplicated")
    return tuple(selected)


def _probe_r_environment_with_closure(
    executable: Path, *, library_paths: tuple[Path, ...] = ()
) -> tuple[dict[str, object], str]:
    """Return a canonical inventory from one selected Rscript executable."""

    selected = _executable(executable)
    if not _dynamic_elf(selected.target):
        raise RuntimeEnvironmentError(
            "R runtime executable must resolve to a dynamic ELF"
        )
    selected_libraries = _r_library_paths(library_paths)
    raw = _run_probe(
        [str(selected.invocation), "--vanilla", "-e", _R_PROBE],
        "R",
        r_library_paths=selected_libraries,
    )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise RuntimeEnvironmentError(
            "R runtime probe returned invalid text"
        ) from error
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
        if fields[0] != "PKG" or len(fields) != 5 or not fields[3].isdigit():
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
    raw_roots.extend(
        _python_executable_roots((selected,), role_prefix="r-selected-launcher")
    )
    runtime_roots, native_linkage_sha256 = _with_native_dependency_roots(
        raw_roots, selected.target
    )
    closure_paths_sha256 = _runtime_closure_paths_sha256(selected, runtime_roots)
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
    return inventory, closure_paths_sha256


def probe_r_environment(
    executable: Path, *, library_paths: tuple[Path, ...] = ()
) -> dict[str, object]:
    """Return a canonical inventory from one selected Rscript executable."""

    return _probe_r_environment_with_closure(executable, library_paths=library_paths)[0]


def _probe_runtime_environment_with_closure(
    kind: Literal["python", "r"],
    executable: Path,
    *,
    r_library_paths: tuple[Path, ...] = (),
) -> tuple[dict[str, object], str]:
    if kind == "python":
        return _probe_python_environment_with_closure(executable)
    if kind == "r":
        return _probe_r_environment_with_closure(
            executable, library_paths=r_library_paths
        )
    raise RuntimeEnvironmentError("runtime kind must be python or r")


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
        if not isinstance(environment_id, str) or not _SAFE_ID.fullmatch(
            environment_id
        ):
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


def _read_secure_runtime_lock(
    path: Path,
) -> tuple[
    Path,
    bytes,
    tuple[tuple[Path, tuple[int, int, int, int, int, int, int]], ...],
]:
    absolute = path.absolute()
    if absolute.name in {"", ".", ".."}:
        raise RuntimeEnvironmentError("runtime lock path is invalid")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory_descriptors: list[int] = []
    directory_paths: list[Path] = []
    descriptor: int | None = None
    try:
        current = os.open(os.path.sep, directory_flags)
        directory_descriptors.append(current)
        current_path = Path(os.path.sep)
        directory_paths.append(current_path)
        directory_states = [_stat_identity(os.fstat(current))]
        for component in absolute.parts[1:-1]:
            try:
                current = os.open(component, directory_flags, dir_fd=current)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise RuntimeEnvironmentError(
                        "runtime lock parent directory must not be a symlink"
                    ) from error
                raise
            directory_descriptors.append(current)
            current_path /= component
            directory_paths.append(current_path)
            metadata = os.fstat(current)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeEnvironmentError(
                    "runtime lock parent directory must be a directory"
                )
            directory_states.append(_stat_identity(metadata))
        try:
            descriptor = os.open(absolute.name, file_flags, dir_fd=current)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise RuntimeEnvironmentError(
                    "runtime lock must not be a symlink"
                ) from error
            raise
        before = os.fstat(descriptor)
        path_before = os.stat(
            absolute.name,
            dir_fd=current,
            follow_symlinks=False,
        )
        if _stat_identity(before) != _stat_identity(path_before):
            raise RuntimeEnvironmentError("runtime lock changed while opening")
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o002
        ):
            raise RuntimeEnvironmentError(
                "runtime lock must be a secure unique regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_LOCK_BYTES:
                raise RuntimeEnvironmentError("runtime lock is too large")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(
            absolute.name,
            dir_fd=current,
            follow_symlinks=False,
        )
        directories_after = [
            _stat_identity(os.fstat(value)) for value in directory_descriptors
        ]
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(path_after)
            or directory_states != directories_after
        ):
            raise RuntimeEnvironmentError("runtime lock changed while reading")
        identity_snapshot = tuple(
            (
                selected_path,
                (
                    selected_identity
                    if index == len(directory_paths) - 1
                    else (*selected_identity[:3], -1, -1, -1, -1)
                ),
            )
            for index, (selected_path, selected_identity) in enumerate(
                zip(directory_paths, directory_states, strict=True)
            )
        ) + ((absolute, _stat_identity(before)),)
        return absolute, b"".join(chunks), identity_snapshot
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError("runtime lock is unavailable") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        for directory_descriptor in reversed(directory_descriptors):
            os.close(directory_descriptor)


def load_runtime_environment_lock(path: Path) -> RuntimeEnvironmentLock:
    """Load a secure canonical runtime lock without trusting its hashes."""

    if not isinstance(path, Path):
        raise TypeError("runtime lock path must be a pathlib.Path")
    absolute, raw, identity_snapshot = _read_secure_runtime_lock(path)
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
        if not isinstance(
            inventory_sha256, str
        ) or inventory_sha256 != canonical_sha256(inventory):
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
    try:
        for selected_path, expected in identity_snapshot:
            observed = _stat_identity(selected_path.lstat())
            if observed[:3] != expected[:3] or (
                expected[3:] != (-1, -1, -1, -1) and observed != expected
            ):
                raise RuntimeEnvironmentError("runtime lock changed while being parsed")
    except RuntimeEnvironmentError:
        raise
    except OSError as error:
        raise RuntimeEnvironmentError(
            "runtime lock changed while being parsed"
        ) from error
    return RuntimeEnvironmentLock(
        path=absolute,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        entries=tuple(entries),
    )


def validate_runtime_environment_lock(
    lock: RuntimeEnvironmentLock,
    environments: Mapping[str, tuple[Literal["python", "r"], Path]],
    *,
    r_library_paths: Mapping[str, Sequence[Path]] | None = None,
    expected_closure_paths_sha256s: Mapping[str, str] | None = None,
    lock_only_environment_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Independently probe every runtime and compare it with a loaded lock."""

    if not isinstance(lock, RuntimeEnvironmentLock):
        raise TypeError("lock must be a RuntimeEnvironmentLock")
    if not isinstance(environments, Mapping):
        raise TypeError("environments must be a mapping")
    if isinstance(lock_only_environment_ids, (str, bytes)) or not isinstance(
        lock_only_environment_ids, Sequence
    ):
        raise TypeError("lock_only_environment_ids must be a sequence")
    lock_only_ids = tuple(lock_only_environment_ids)
    if any(
        not isinstance(environment_id, str) or not _SAFE_ID.fullmatch(environment_id)
        for environment_id in lock_only_ids
    ):
        raise RuntimeEnvironmentError("runtime lock-only environment ID is invalid")
    if len(set(lock_only_ids)) != len(lock_only_ids):
        raise RuntimeEnvironmentError(
            "runtime lock-only environment IDs are duplicated"
        )
    libraries = {} if r_library_paths is None else dict(r_library_paths)
    expected_closures = (
        None
        if expected_closure_paths_sha256s is None
        else dict(expected_closure_paths_sha256s)
    )
    if expected_closures is not None and set(expected_closures) != set(environments):
        raise RuntimeEnvironmentError(
            "runtime closure path expectations do not match environments"
        )
    if set(libraries) - set(environments):
        raise RuntimeEnvironmentError("R library paths name an unknown environment")
    expected_ids = {entry.environment_id for entry in lock.entries}
    observed_ids = set(environments)
    lock_only_id_set = set(lock_only_ids)
    if observed_ids & lock_only_id_set:
        raise RuntimeEnvironmentError(
            "runtime lock-only environment IDs overlap live declarations"
        )
    if observed_ids | lock_only_id_set != expected_ids:
        raise RuntimeEnvironmentError("runtime IDs mismatch frozen lock")
    receipts: list[tuple[str, str]] = []
    inventory_cache: dict[
        tuple[str, str, tuple[str, ...]], tuple[dict[str, object], str]
    ] = {}
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
            raise RuntimeEnvironmentError(f"runtime kind mismatch for {environment_id}")
        cache_key = (
            kind,
            executable.absolute().as_posix(),
            tuple(path.absolute().as_posix() for path in selected_libraries),
        )
        if cache_key not in inventory_cache:
            inventory_cache[cache_key] = _probe_runtime_environment_with_closure(
                kind, executable, r_library_paths=selected_libraries
            )
        inventory, closure_paths_sha256 = inventory_cache[cache_key]
        if inventory != entry.inventory:
            raise RuntimeEnvironmentError(
                f"runtime inventory mismatch for {environment_id}"
            )
        if (
            expected_closures is not None
            and closure_paths_sha256 != expected_closures[environment_id]
        ):
            raise RuntimeEnvironmentError(
                f"runtime closure paths mismatch for {environment_id}"
            )
        inventory_sha256 = entry.inventory_sha256
        receipts.append((environment_id, inventory_sha256))
    receipt: dict[str, object] = {
        "lock_file_sha256": lock.file_sha256,
        "environment_inventory_sha256s": tuple(receipts),
    }
    if lock_only_ids:
        receipt["lock_only_environment_inventory_sha256s"] = tuple(
            (environment_id, lock.by_id(environment_id).inventory_sha256)
            for environment_id in sorted(lock_only_ids)
        )
    return receipt


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
    "RuntimeChangeMonitor",
    "RuntimeEnvironmentEntry",
    "RuntimeEnvironmentError",
    "RuntimeEnvironmentLock",
    "RuntimeEnvironmentSnapshot",
    "build_runtime_environment_lock",
    "load_runtime_environment_lock",
    "merge_runtime_environment_snapshots",
    "nvidia_smi_executable",
    "process_environment_sha256",
    "publication_git_executable",
    "publication_runtime_working_directory",
    "probe_python_environment",
    "probe_r_environment",
    "probe_runtime_environment",
    "runtime_environment_identity_sha256",
    "runtime_environment_snapshot",
    "validate_runtime_environment_lock",
    "validate_runtime_environment_entry",
    "verify_runtime_environment_control_files",
    "verify_runtime_environment_snapshot",
]
