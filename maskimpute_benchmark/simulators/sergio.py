"""Pinned, paired-view SERGIO validation adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import errno
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scipy

from ..protocol import Protocol, canonical_sha256, file_sha256
from ..schema import benchmark_dataset_sha256
from ..sources import (
    SourceLedgerError,
    fetch_sources,
    load_source_ledger,
    verify_fetched_sources,
)
from .base import (
    FinalManifestClaim,
    SimulationArtifact,
    SimulationContractError,
    SimulationRequest,
    simulation_scientific_identity,
    validate_paired_simulation_requests,
)
from .native import seal_native_outputs
from .runtime_assets import (
    SimulatorRuntimeAssets,
    revalidate_simulator_runtime_asset_identity,
    simulator_runtime_asset_values,
    simulator_runtime_source_receipt,
)
from .symsim import _revalidate_published_final_claim


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER_PATH = _REPO_ROOT / "study/sources.json"
_EXTERNAL_ROOT = _REPO_ROOT / "artifacts/external"
_CHECKOUT = _EXTERNAL_ROOT / "checkouts/sergio"
_RUNNER = _REPO_ROOT / "scripts/simulators/run_sergio.py"
_SERGIO_COMMIT = "a6190b74425112834c8fa9b4b6157d9cb3d1ab88"
_SERGIO_TREE = "15558fe60f62683c6fa46bcde01d9f3d3382e34a"
_MODULE_PATH = _CHECKOUT / "SERGIO/sergio.py"
_COMPATIBILITY_SHIM = {
    "numpy_removed_aliases": {
        "np.float": "builtins.float",
        "np.int": "builtins.int",
    }
}
_EXPECTED_NATIVE_FILES = frozenset(
    {
        "clean.npy",
        "config.json",
        "dropout_indicator_moderate.npy",
        "dropout_indicator_severe.npy",
        "observed_moderate.npy",
        "observed_severe.npy",
        "pre_dropout_moderate.npy",
        "pre_dropout_severe.npy",
        "run_metadata.json",
    }
)
_ARRAY_FILES = tuple(
    sorted(name for name in _EXPECTED_NATIVE_FILES if name.endswith(".npy"))
)
_VIEW_PARAMETERS: dict[str, dict[str, int | float]] = {
    "moderate": {
        "outlier_prob": 0.01,
        "outlier_mean": 0.8,
        "outlier_scale": 1.0,
        "library_log_mean": 5.2,
        "library_log_sd": 0.3,
        "dropout_shape": 6.5,
        "dropout_percentile": 65,
    },
    "severe": {
        "outlier_prob": 0.01,
        "outlier_mean": 0.8,
        "outlier_scale": 1.0,
        "library_log_mean": 4.6,
        "library_log_sd": 0.4,
        "dropout_shape": 6.5,
        "dropout_percentile": 82,
    },
}
_PROFILES: tuple[dict[str, object], ...] = (
    {
        "maximum_requested_genes": 100,
        "name": "De-noised_100G_9T_300cPerT_4_DS1",
        "simulated_genes": 100,
        "interaction_path": (
            "data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt"
        ),
        "regulator_path": ("data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt"),
    },
    {
        "maximum_requested_genes": 1200,
        "name": "De-noised_1200G_9T_300cPerT_6_DS3",
        "simulated_genes": 1200,
        "interaction_path": (
            "data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Interaction_cID_6.txt"
        ),
        "regulator_path": (
            "data_sets/De-noised_1200G_9T_300cPerT_6_DS3/Regs_cID_6.txt"
        ),
    },
)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulationContractError(f"duplicate JSON key in SERGIO output: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SimulationContractError(f"non-finite JSON value in SERGIO output: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json_bytes(data: bytes, name: str) -> object:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimulationContractError(f"{name} is not strict UTF-8 JSON") from error
    try:
        canonical = _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise SimulationContractError(f"{name} is not canonical JSON") from error
    if canonical != data:
        raise SimulationContractError(f"{name} is not canonical JSON")
    return value


def _mapped_numpy_seed(seed: int, domain: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise SimulationContractError(
            "SERGIO seeds must be 63-bit nonnegative integers"
        )
    digest = canonical_sha256(
        {"schema": "maskimpute-sergio-numpy-seed-v1", "domain": domain, "seed": seed}
    )
    return int(digest[:16], 16) % (2**32 - 1) + 1


def map_sergio_numpy_seeds(
    biological_seed: int, moderate_seed: int, severe_seed: int
) -> dict[str, int]:
    """Map 63-bit study seeds into distinct deterministic NumPy seeds."""

    originals = {
        "biological": biological_seed,
        "moderate": moderate_seed,
        "severe": severe_seed,
    }
    mapped: dict[str, int] = {}
    used: set[int] = set()
    for role, seed in originals.items():
        candidate = _mapped_numpy_seed(seed, role)
        while candidate in used:
            candidate = candidate % (2**32 - 1) + 1
        mapped[role] = candidate
        used.add(candidate)
    return mapped


def _profile_for_genes(genes: int) -> dict[str, object]:
    if type(genes) is not int or genes <= 0:
        raise SimulationContractError(
            "SERGIO requested genes must be a positive integer"
        )
    for profile in _PROFILES:
        maximum = profile["maximum_requested_genes"]
        if isinstance(maximum, int) and genes <= maximum:
            return {
                key: value
                for key, value in profile.items()
                if key != "maximum_requested_genes"
            }
    raise SimulationContractError("SERGIO supports at most 1200 requested genes")


def _verify_sergio_source(
    *, external_root: Path | None = None, immutable: bool = False
) -> dict[str, object]:
    selected_root = _EXTERNAL_ROOT if external_root is None else external_root
    module_path = (
        _MODULE_PATH
        if external_root is None
        else selected_root / "checkouts/sergio/SERGIO/sergio.py"
    )
    try:
        ledger = load_source_ledger(_LEDGER_PATH)
        verifier = verify_fetched_sources if immutable else fetch_sources
        receipt = verifier(ledger, selected_root, source_ids=("sergio",))[0]
    except (OSError, SourceLedgerError, TypeError, ValueError) as error:
        raise SimulationContractError(
            f"pinned SERGIO source is not pristine: {error}"
        ) from error
    checksum = receipt.get("verified_checksum")
    if (
        receipt.get("resolved_revision") != _SERGIO_COMMIT
        or not isinstance(checksum, Mapping)
        or checksum.get("algorithm") != "git-tree-sha1"
        or checksum.get("value") != _SERGIO_TREE
    ):
        raise SimulationContractError(
            "SERGIO source receipt does not match the exact pin"
        )
    if module_path.is_symlink() or not module_path.is_file():
        raise SimulationContractError("pinned SERGIO module path is unavailable")
    return receipt


def _environment_receipt() -> dict[str, object]:
    executable = Path(sys.executable).resolve(strict=True)
    if executable.is_symlink() or not executable.is_file():
        raise SimulationContractError("SERGIO Python executable is unavailable")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
    }
    payload = {
        "schema": "maskimpute-python-environment-v1",
        "python_executable_sha256": file_sha256(executable),
        "versions": versions,
    }
    return {**payload, "sha256": canonical_sha256(payload)}


def _execute_sergio(
    config_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: int,
    external_root: Path | None = None,
) -> None:
    selected_root = _EXTERNAL_ROOT if external_root is None else external_root
    checkout = (
        _CHECKOUT if external_root is None else selected_root / "checkouts/sergio"
    )
    if _RUNNER.is_symlink() or not _RUNNER.is_file():
        raise SimulationContractError("tracked SERGIO Python runner is unavailable")
    environment = {
        "HOME": output_dir.as_posix(),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{Path(sys.executable).resolve().parent.as_posix()}:/usr/bin:/bin",
        "TZ": "UTC",
    }
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                _RUNNER.as_posix(),
                config_path.as_posix(),
                checkout.as_posix(),
                output_dir.as_posix(),
            ],
            cwd=_REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SimulationContractError(
            f"SERGIO native runner failed: {error}"
        ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-4000:]
        raise SimulationContractError(
            f"SERGIO native runner exited {completed.returncode}: {stderr}"
        )


def _expected_ids(prefix: str, count: int) -> list[str]:
    width = max(4, len(str(count)))
    return [f"{prefix}-{index:0{width}d}" for index in range(1, count + 1)]


def _read_regular_bytes(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.lstat()
        if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
            raise SimulationContractError(
                f"SERGIO native output must be a unique regular file: {path.name}"
            )
        descriptor = os.open(path, flags)
    except SimulationContractError:
        raise
    except OSError as error:
        raise SimulationContractError(
            f"SERGIO native output cannot be opened safely: {path.name}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise SimulationContractError(
                f"SERGIO native output changed while opening: {path.name}"
            )
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise SimulationContractError(
                f"SERGIO native output is too large: {path.name}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if maximum_bytes is not None and total > maximum_bytes:
                raise SimulationContractError(
                    f"SERGIO native output grew beyond its size limit: {path.name}"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
        after_path = path.lstat()
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_state = (
            after_path.st_dev,
            after_path.st_ino,
            after_path.st_mode,
            after_path.st_nlink,
            after_path.st_size,
            after_path.st_mtime_ns,
            after_path.st_ctime_ns,
        )
        if before_state != after_state or before_state != path_state:
            raise SimulationContractError(
                f"SERGIO native output changed while reading: {path.name}"
            )
        return b"".join(chunks)
    except OSError as error:
        raise SimulationContractError(
            f"SERGIO native output changed while reading: {path.name}"
        ) from error
    finally:
        os.close(descriptor)


def _validate_stage_entries(stage: Path) -> dict[str, Path]:
    try:
        entries = list(os.scandir(stage))
    except OSError as error:
        raise SimulationContractError(
            "SERGIO native output directory is unavailable"
        ) from error
    names = {entry.name for entry in entries}
    if names != _EXPECTED_NATIVE_FILES or len(entries) != len(_EXPECTED_NATIVE_FILES):
        raise SimulationContractError(
            "SERGIO native outputs do not match the closed file set"
        )
    result: dict[str, Path] = {}
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise SimulationContractError(
                "SERGIO native output cannot be inspected"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SimulationContractError(
                "SERGIO native outputs must be unique regular files without symlinks"
            )
        result[entry.name] = Path(entry.path)
    return result


def _read_npy(
    path: Path, *, dtype: np.dtype[Any], shape: tuple[int, int]
) -> np.ndarray:
    expected_dtype = np.dtype(dtype)
    payload_size = math.prod(shape) * expected_dtype.itemsize
    try:
        size_bytes = path.lstat().st_size
    except OSError as error:
        raise SimulationContractError(f"{path.name} cannot be inspected") from error
    if size_bytes > payload_size + 65_536:
        raise SimulationContractError(
            f"{path.name} is too large for its declared numeric shape"
        )
    data = _read_regular_bytes(path, maximum_bytes=payload_size + 65_536)
    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            observed_shape, fortran_order, observed_dtype = (
                np.lib.format.read_array_header_1_0(stream)
            )
        elif version == (2, 0):
            observed_shape, fortran_order, observed_dtype = (
                np.lib.format.read_array_header_2_0(stream)
            )
        else:
            raise SimulationContractError(
                f"{path.name} uses an unsupported NPY version"
            )
    except (EOFError, ValueError) as error:
        raise SimulationContractError(f"{path.name} is malformed NPY") from error
    expected_size = stream.tell() + math.prod(shape) * expected_dtype.itemsize
    if (
        observed_shape != shape
        or fortran_order
        or observed_dtype.hasobject
        or observed_dtype.str != expected_dtype.str
        or len(data) != expected_size
    ):
        raise SimulationContractError(
            f"{path.name} has wrong shape, orientation, dtype, or byte length"
        )
    stream.seek(0)
    try:
        values = np.load(stream, allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError) as error:
        raise SimulationContractError(
            f"{path.name} is not a safe numeric NPY"
        ) from error
    if (
        values.shape != shape
        or values.dtype.str != expected_dtype.str
        or not values.flags.c_contiguous
        or stream.tell() != len(data)
    ):
        raise SimulationContractError(f"{path.name} violates the canonical NPY layout")
    return values


def _validate_run_metadata(
    path: Path,
    files: Mapping[str, Path],
    config: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    value = _load_json_bytes(
        _read_regular_bytes(path, maximum_bytes=1024 * 1024), path.name
    )
    if not isinstance(value, dict):
        raise SimulationContractError("run_metadata.json must be an object")
    expected_keys = {
        "schema_version",
        "array_sha256",
        "biological_seed_numpy",
        "call_counts",
        "cells",
        "cell_types",
        "compatibility_shim",
        "measurement_seeds_numpy",
        "module_path",
        "requested_genes",
        "simulated_genes",
        "versions",
        "views",
    }
    simulation = config.get("simulation")
    seeds = config.get("seeds")
    views = config.get("views")
    source = config.get("source")
    adapter = config.get("adapter")
    if not all(
        isinstance(item, Mapping) for item in (simulation, seeds, source, adapter)
    ) or not isinstance(views, list):
        raise SimulationContractError("SERGIO config binding is invalid")
    assert isinstance(simulation, Mapping)
    assert isinstance(seeds, Mapping)
    assert isinstance(source, Mapping)
    assert isinstance(adapter, Mapping)
    biological = seeds.get("biological")
    if not isinstance(biological, Mapping):
        raise SimulationContractError("SERGIO biological seed binding is invalid")
    expected_measurement = {
        view["technical_view"]: view["measurement_seed_numpy"]
        for view in views
        if isinstance(view, Mapping)
    }
    per_view = {
        name: {
            "outlier_effect": 1,
            "lib_size_effect": 1,
            "dropout_indicator": 1,
            "convert_to_umi_counts": 1,
        }
        for name in ("moderate", "severe")
    }
    expected_call_counts = {
        "sergio_constructor": 1,
        "build_graph": 1,
        "simulate": 1,
        "get_expressions": 1,
        "outlier_effect": 2,
        "lib_size_effect": 2,
        "dropout_indicator": 2,
        "convert_to_umi_counts": 2,
        "per_view": per_view,
    }
    expected_binding = {
        "schema_version": 1,
        "biological_seed_numpy": biological.get("mapped_numpy"),
        "call_counts": expected_call_counts,
        "cells": simulation.get("cells"),
        "cell_types": 9,
        "compatibility_shim": adapter.get("compatibility_shim"),
        "measurement_seeds_numpy": expected_measurement,
        "module_path": source.get("module_path"),
        "requested_genes": simulation.get("requested_genes"),
        "simulated_genes": simulation.get("simulated_genes"),
        "views": [
            view.get("technical_view") for view in views if isinstance(view, Mapping)
        ],
    }
    observed_binding = {key: value.get(key) for key in expected_binding}
    versions = value.get("versions")
    environment_versions = environment.get("versions")
    hashes = value.get("array_sha256")
    if (
        set(value) != expected_keys
        or canonical_sha256(observed_binding) != canonical_sha256(expected_binding)
        or not isinstance(versions, Mapping)
        or set(versions) != {"python", "numpy", "scipy", "networkx", "sergio"}
        or not all(
            isinstance(version, str) and version for version in versions.values()
        )
        or not isinstance(environment_versions, Mapping)
        or dict(versions) != {**dict(environment_versions), "sergio": "1.0.0"}
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(_ARRAY_FILES)
    ):
        raise SimulationContractError(
            "run_metadata.json does not bind the exact paired run"
        )
    for name in _ARRAY_FILES:
        digest = hashes.get(name)
        if not isinstance(digest, str) or digest != file_sha256(files[name]):
            raise SimulationContractError(
                f"run_metadata.json does not bind exact bytes for {name}"
            )
    return value


def _native_descriptor(files: Mapping[str, Path]) -> list[dict[str, object]]:
    return [
        {
            "path": name,
            "sha256": file_sha256(files[name]),
            "size_bytes": files[name].stat().st_size,
        }
        for name in sorted(files)
    ]


def _reject_output_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for component in [absolute, *absolute.parents]:
        if not os.path.lexists(component):
            continue
        try:
            metadata = component.lstat()
        except OSError as error:
            raise SimulationContractError(
                "SERGIO output path cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SimulationContractError(
                "SERGIO output path must not contain symlinks"
            )


def _path_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SimulationContractError(
            f"SERGIO path identity is unavailable: {path}"
        ) from error
    return metadata.st_dev, metadata.st_ino


def _remove_owned_file(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        return
    if (
        stat.S_ISREG(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == identity
    ):
        try:
            path.unlink()
        except OSError:
            pass


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        return
    if (
        stat.S_ISDIR(metadata.st_mode)
        and not stat.S_ISLNK(metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino) == identity
    ):
        shutil.rmtree(path, ignore_errors=True)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing path."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), destination.as_posix()
        )
    raise OSError(error_number, os.strerror(error_number), destination.as_posix())


def _publish_native_directory(
    files: Mapping[str, Path], parent: Path
) -> tuple[Path, bool, tuple[int, int]]:
    descriptor = _native_descriptor(files)
    sealed_sizes = {
        str(entry["path"]): int(entry["size_bytes"]) for entry in descriptor
    }
    content_id = canonical_sha256(
        {"schema": "maskimpute-sergio-native-v1", "files": descriptor}
    )[:24]
    _reject_output_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_output_symlink_components(parent)
    native_root = parent / "native"
    if native_root.is_symlink() or (native_root.exists() and not native_root.is_dir()):
        raise SimulationContractError("SERGIO native output root is invalid")
    native_root.mkdir(mode=0o755, exist_ok=True)
    destination = native_root / f"sergio-{content_id}"
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise SimulationContractError("existing SERGIO native directory is invalid")
        existing = _validate_stage_entries(destination)
        if _native_descriptor(existing) != descriptor:
            raise SimulationContractError("existing SERGIO native bytes changed")
        return destination, False, _path_identity(destination)
    publication = Path(tempfile.mkdtemp(prefix=".sergio-publish-", dir=native_root))
    publication_identity = _path_identity(publication)
    renamed = False
    try:
        for name in sorted(files):
            data = _read_regular_bytes(files[name], maximum_bytes=sealed_sizes[name])
            output = publication / name
            output_fd = os.open(
                output,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o644,
            )
            try:
                remaining = memoryview(data)
                while remaining:
                    written = os.write(output_fd, remaining)
                    remaining = remaining[written:]
                os.fsync(output_fd)
            finally:
                os.close(output_fd)
        copied = _validate_stage_entries(publication)
        if _native_descriptor(copied) != descriptor:
            raise SimulationContractError(
                "SERGIO native bytes changed while publishing"
            )
        _rename_directory_no_replace(publication, destination)
        renamed = True
        directory_fd = os.open(native_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except SimulationContractError:
        raise
    except OSError as error:
        if renamed:
            _remove_owned_directory(destination, publication_identity)
        raise SimulationContractError(
            "SERGIO native outputs could not be published"
        ) from error
    finally:
        if publication.exists():
            shutil.rmtree(publication)
    return destination, True, publication_identity


def _h5ad_staging_root(destination: Path) -> Path:
    absolute = destination.absolute()
    current = absolute.parent
    while not os.path.lexists(current):
        if current == current.parent:
            raise SimulationContractError("SERGIO output has no existing ancestor")
        current = current.parent
    try:
        current_metadata = current.lstat()
    except OSError as error:
        raise SimulationContractError(
            "SERGIO output ancestor cannot be inspected"
        ) from error
    if not stat.S_ISDIR(current_metadata.st_mode) or stat.S_ISLNK(
        current_metadata.st_mode
    ):
        raise SimulationContractError(
            "SERGIO output ancestor must be a non-symlink directory"
        )
    try:
        inside_repository = absolute.is_relative_to(_REPO_ROOT)
    except ValueError:
        inside_repository = False
    root = _REPO_ROOT.parent if inside_repository else current
    try:
        root_metadata = root.lstat()
    except OSError as error:
        raise SimulationContractError(
            "SERGIO h5ad staging root cannot be inspected"
        ) from error
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or root_metadata.st_dev != current_metadata.st_dev
    ):
        raise SimulationContractError(
            "SERGIO h5ad staging root must be a same-device non-symlink directory"
        )
    return root


def _stage_h5ad(adata: ad.AnnData, destination: Path) -> tuple[Path, ad.AnnData]:
    _reject_output_symlink_components(destination)
    staging_root = _h5ad_staging_root(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="maskimpute-sergio-h5ad-", suffix=".h5ad", dir=staging_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        adata.write_h5ad(temporary)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return temporary, ad.read_h5ad(temporary)
    except BaseException as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if not isinstance(error, (OSError, RuntimeError, TypeError, ValueError)):
            raise
        raise SimulationContractError(
            f"SERGIO dataset could not be serialized for atomic persistence: {destination}"
        ) from error


def _publish_staged_h5ad(
    temporary: Path, destination: Path
) -> tuple[ad.AnnData, tuple[int, int]]:
    _reject_output_symlink_components(destination)
    if os.path.lexists(destination):
        raise SimulationContractError(
            f"SERGIO refuses to overwrite an existing result: {destination}"
        )
    linked = False
    identity = _path_identity(temporary)
    try:
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        temporary.unlink()
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return ad.read_h5ad(destination), identity
    except BaseException as error:
        if linked:
            _remove_owned_file(destination, identity)
        if not isinstance(error, (OSError, RuntimeError, TypeError, ValueError)):
            raise
        raise SimulationContractError(
            f"SERGIO dataset could not be atomically persisted: {destination}"
        ) from error


def _marker_truth(clean_gene_cell: np.ndarray, cells_per_type: int) -> pd.DataFrame:
    gene_ids = _expected_ids("gene", clean_gene_cell.shape[0])
    values: dict[str, np.ndarray] = {}
    pseudocount = 1e-8
    for group in range(1, 10):
        start = (group - 1) * cells_per_type
        stop = group * cells_per_type
        mask = np.zeros(clean_gene_cell.shape[1], dtype=bool)
        mask[start:stop] = True
        group_mean = clean_gene_cell[:, mask].mean(axis=1, dtype=np.float64)
        other_mean = clean_gene_cell[:, ~mask].mean(axis=1, dtype=np.float64)
        log2fc = np.log2(np.divide(group_mean + pseudocount, other_mean + pseudocount))
        values[f"clean_log2fc_cell_type_{group}"] = log2fc.astype(
            np.float64, copy=False
        )
        values[f"marker_cell_type_{group}"] = log2fc > 1.0
    return pd.DataFrame(values, index=gene_ids)


def _build_dataset(
    request: SimulationRequest,
    observed_gene_cell: np.ndarray,
    clean_gene_cell: np.ndarray,
    pre_dropout_gene_cell: np.ndarray,
    marker_truth: pd.DataFrame,
    native_manifest_sha256: str,
    source_receipt: Mapping[str, object],
    environment: Mapping[str, object],
    config: Mapping[str, object],
    run_metadata: Mapping[str, object],
    pair_request_sha256: str,
) -> ad.AnnData:
    observed = observed_gene_cell.T.copy()
    clean = clean_gene_cell.T.copy()
    pre_dropout = pre_dropout_gene_cell.T.copy()
    library_sizes = [sum(int(value) for value in row) for row in observed]
    if any(total > np.iinfo(np.int64).max for total in library_sizes):
        raise SimulationContractError("SERGIO observed library size exceeds int64")
    draw = int(request.biological_id.removeprefix("draw-"))
    cells_per_type = request.cells // 9
    groups = [
        f"cell-type-{group}" for group in range(1, 10) for _ in range(cells_per_type)
    ]
    obs = pd.DataFrame(
        {
            "dataset_id": [request.dataset_id] * request.cells,
            "mechanism": [request.mechanism] * request.cells,
            "condition": [request.technical_view] * request.cells,
            "biological_id": [request.biological_id] * request.cells,
            "technical_view": [request.technical_view] * request.cells,
            "draw": np.full(request.cells, draw, dtype=np.int64),
            "library_size": np.asarray(library_sizes, dtype=np.int64),
            "group": groups,
        },
        index=_expected_ids("cell", request.cells),
    )
    views = config["views"]
    seeds = config["seeds"]
    assert isinstance(views, list)
    assert isinstance(seeds, Mapping)
    biological_config = seeds["biological"]
    assert isinstance(biological_config, Mapping)
    view_config = next(
        view
        for view in views
        if isinstance(view, Mapping)
        and view.get("technical_view") == request.technical_view
    )
    dataset = ad.AnnData(
        X=observed,
        obs=obs,
        var=marker_truth.copy(deep=True),
        layers={
            "latent_expression": clean,
            "pre_dropout_expression": pre_dropout,
        },
    )
    dataset.uns.update(
        {
            "truth_kind": "exact_continuous",
            "primary_truth_layer": "latent_expression",
            "allowed_covariates": {"obs": [], "var": []},
            "provenance": {
                "source": source_receipt["source_url"],
                "source_sha256": canonical_sha256(source_receipt),
                "software": "SERGIO",
                "software_version": source_receipt["resolved_revision"],
                "parameters": {
                    "adapter": config["adapter"],
                    "adapter_schema": "maskimpute-sergio-adapter-v1",
                    "environment": environment,
                    "marker_definition": {
                        "estimand": "clean_mean_cell_type_vs_all_other_cells",
                        "log2fc_threshold": 1.0,
                        "pseudocount": 1e-8,
                    },
                    "measurement": {
                        key: value
                        for key, value in view_config.items()
                        if key
                        not in {
                            "measurement_seed_original",
                            "measurement_seed_numpy",
                            "technical_view",
                        }
                    },
                    "native_manifest_sha256": native_manifest_sha256,
                    "native_run_metadata": run_metadata,
                    "pair_request_sha256": pair_request_sha256,
                    "profile": config["profile"],
                    "score_truth": "undefined_for_continuous_truth",
                    "simulation": config["simulation"],
                    "simulation_request": simulation_scientific_identity(request),
                    "source_receipt": source_receipt,
                },
                "seeds": {
                    "biological": request.biological_seed,
                    "measurement": request.measurement_seed,
                    "numpy_biological": biological_config["mapped_numpy"],
                    "numpy_measurement": view_config["measurement_seed_numpy"],
                },
            },
        }
    )
    return dataset


def _pair_config(requests: Mapping[str, SimulationRequest]) -> dict[str, object]:
    moderate = requests["moderate"]
    severe = requests["severe"]
    mapped = map_sergio_numpy_seeds(
        moderate.biological_seed,
        moderate.measurement_seed,
        severe.measurement_seed,
    )
    selected_profile = _profile_for_genes(moderate.genes)
    simulated_genes = selected_profile["simulated_genes"]
    if not isinstance(simulated_genes, int):
        raise SimulationContractError("SERGIO profile gene count is invalid")
    profile = {
        key: value
        for key, value in selected_profile.items()
        if key != "simulated_genes"
    }
    views: list[dict[str, object]] = []
    for name in ("moderate", "severe"):
        request = requests[name]
        views.append(
            {
                "technical_view": name,
                "measurement_seed_original": request.measurement_seed,
                "measurement_seed_numpy": mapped[name],
                **_VIEW_PARAMETERS[name],
            }
        )
    return {
        "adapter": {
            "compatibility_shim": _COMPATIBILITY_SHIM,
            "python_adapter_sha256": file_sha256(Path(__file__)),
            "python_runner_sha256": file_sha256(_RUNNER),
        },
        "profile": profile,
        "schema_version": 1,
        "seeds": {
            "biological": {
                "original": moderate.biological_seed,
                "mapped_numpy": mapped["biological"],
            }
        },
        "simulation": {
            "cells": moderate.cells,
            "cell_types": 9,
            "cells_per_type": moderate.cells // 9,
            "decays": 0.8,
            "noise_params": 1.0,
            "noise_type": "dpd",
            "requested_genes": moderate.genes,
            "sampling_state": 15,
            "shared_coop_state": 2.0,
            "simulated_genes": simulated_genes,
        },
        "source": {
            "commit": _SERGIO_COMMIT,
            "module_path": "SERGIO/sergio.py",
            "tree": _SERGIO_TREE,
        },
        "views": views,
    }


def run_sergio_pair(
    requests: Sequence[SimulationRequest],
    protocol: Protocol,
    final_manifest: FinalManifestClaim | None = None,
    *,
    runtime_assets: SimulatorRuntimeAssets | None = None,
) -> tuple[SimulationArtifact, SimulationArtifact]:
    """Generate paired moderate/severe views from one clean SERGIO draw."""

    try:
        ordered_requests = tuple(requests)
    except TypeError as error:
        raise SimulationContractError(
            "SERGIO requests must be a finite pair"
        ) from error
    validate_paired_simulation_requests(ordered_requests, protocol, final_manifest)
    if any(request.mechanism != "sergio" for request in ordered_requests):
        raise SimulationContractError("SERGIO adapter accepts only sergio requests")
    by_view = {request.technical_view: request for request in ordered_requests}
    if set(by_view) != {"moderate", "severe"} or len(by_view) != 2:
        raise SimulationContractError(
            "SERGIO adapter requires exactly moderate and severe technical views"
        )
    moderate = by_view["moderate"]
    if moderate.cells < 18 or moderate.cells % 9 != 0:
        raise SimulationContractError(
            "SERGIO requires at least two cells per type and cells divisible by 9"
        )
    _profile_for_genes(moderate.genes)
    parents = {request.output_path.parent.absolute() for request in ordered_requests}
    if len(parents) != 1:
        raise SimulationContractError("paired SERGIO outputs must share one directory")
    output_parent = next(iter(parents))
    _reject_output_symlink_components(output_parent)
    for request in ordered_requests:
        if os.path.lexists(request.output_path):
            raise SimulationContractError(
                f"SERGIO refuses to overwrite an existing result: {request.output_path}"
            )

    config = _pair_config(by_view)
    config_bytes = _canonical_json_bytes(config)
    runtime_assets_sha256: str | None = None
    if runtime_assets is None:
        verify_source = _verify_sergio_source
        execute = _execute_sergio
    else:
        external_root, _r_environment, runtime_assets_sha256 = (
            simulator_runtime_asset_values(runtime_assets)
        )
        verify_source = lambda: simulator_runtime_source_receipt(  # noqa: E731
            runtime_assets, "sergio"
        )

        def execute(
            config_path: Path, output_dir: Path, *, timeout_seconds: int
        ) -> None:
            _execute_sergio(
                config_path,
                output_dir,
                timeout_seconds=timeout_seconds,
                external_root=external_root,
            )

    before_source = verify_source()
    before_environment = _environment_receipt()
    stage = Path(tempfile.mkdtemp(prefix="maskimpute-sergio-native-"))
    native_directory: Path | None = None
    native_identity: tuple[int, int] | None = None
    native_created = False
    publication_complete = False
    runner_error: BaseException | None = None
    try:
        config_path = stage / "config.json"
        config_path.write_bytes(config_bytes)
        try:
            execute(
                config_path,
                stage,
                timeout_seconds=protocol.final_timeout_seconds,
            )
        except BaseException as error:  # source/environment must still be rechecked
            runner_error = error
        try:
            after_source = verify_source()
            after_environment = _environment_receipt()
        except Exception as error:
            raise SimulationContractError(
                "SERGIO source or environment was not pristine after execution"
            ) from error
        if canonical_sha256(before_source) != canonical_sha256(after_source):
            raise SimulationContractError(
                "SERGIO source receipt changed during execution"
            )
        if canonical_sha256(before_environment) != canonical_sha256(after_environment):
            raise SimulationContractError("SERGIO environment changed during execution")
        if runner_error is not None:
            if isinstance(runner_error, SimulationContractError):
                raise runner_error
            if not isinstance(runner_error, Exception):
                raise runner_error
            raise SimulationContractError(
                "SERGIO native runner failed"
            ) from runner_error

        files = _validate_stage_entries(stage)
        observed_config_bytes = _read_regular_bytes(
            files["config.json"], maximum_bytes=len(config_bytes)
        )
        observed_config = _load_json_bytes(observed_config_bytes, "config.json")
        if observed_config_bytes != config_bytes or canonical_sha256(
            observed_config
        ) != canonical_sha256(config):
            raise SimulationContractError(
                "SERGIO native runner changed its sealed config"
            )
        shape = (moderate.genes, moderate.cells)
        clean = _read_npy(files["clean.npy"], dtype=np.dtype("<f8"), shape=shape)
        pre_dropout = {
            name: _read_npy(
                files[f"pre_dropout_{name}.npy"],
                dtype=np.dtype("<f8"),
                shape=shape,
            )
            for name in ("moderate", "severe")
        }
        indicators = {
            name: _read_npy(
                files[f"dropout_indicator_{name}.npy"],
                dtype=np.dtype("u1"),
                shape=shape,
            )
            for name in ("moderate", "severe")
        }
        observed = {
            name: _read_npy(
                files[f"observed_{name}.npy"],
                dtype=np.dtype("<i8"),
                shape=shape,
            )
            for name in ("moderate", "severe")
        }
        if not np.isfinite(clean).all() or bool((clean < 0).any()):
            raise SimulationContractError(
                "SERGIO clean expression must be finite and nonnegative"
            )
        if any(
            not np.isfinite(matrix).all() or bool((matrix < 0).any())
            for matrix in pre_dropout.values()
        ):
            raise SimulationContractError(
                "SERGIO pre-dropout expression must be finite and nonnegative"
            )
        if any(
            not bool(np.isin(matrix, (0, 1)).all()) for matrix in indicators.values()
        ):
            raise SimulationContractError(
                "SERGIO dropout indicators must contain only zero and one"
            )
        if any(bool((matrix < 0).any()) for matrix in observed.values()):
            raise SimulationContractError(
                "SERGIO observed UMI counts must be nonnegative"
            )
        if any(
            bool((observed[name][indicators[name] == 0] != 0).any())
            for name in ("moderate", "severe")
        ):
            raise SimulationContractError(
                "SERGIO observed UMI counts contradict the dropout indicators"
            )
        if np.array_equal(observed["moderate"], observed["severe"]):
            raise SimulationContractError(
                "paired SERGIO technical views must have different observed counts"
            )
        run_metadata = _validate_run_metadata(
            files["run_metadata.json"], files, config, before_environment
        )
        marker_truth = _marker_truth(clean, moderate.cells // 9)
        pair_identity = {
            name: simulation_scientific_identity(by_view[name])
            for name in ("moderate", "severe")
        }
        pair_request_sha256 = canonical_sha256(pair_identity)
        manifest_metadata: dict[str, dict[str, object]] = {}
        staging_manifests = {}
        for name in ("moderate", "severe"):
            request = by_view[name]
            metadata = {
                "adapter": config["adapter"],
                "adapter_schema": "maskimpute-sergio-native-v1",
                "config_sha256": canonical_sha256(config),
                "environment": before_environment,
                "pair_request_sha256": pair_request_sha256,
                "run_metadata": run_metadata,
                "simulation_request": simulation_scientific_identity(request),
                "source_receipt": before_source,
            }
            if runtime_assets_sha256 is not None:
                metadata["runtime_assets_sha256"] = runtime_assets_sha256
            manifest_metadata[name] = metadata
            staging_manifests[name] = seal_native_outputs(files, metadata)

        staged_datasets: dict[str, tuple[Path, ad.AnnData]] = {}
        staged_hashes: dict[str, str] = {}
        published_results: list[tuple[Path, tuple[int, int]]] = []
        try:
            for name in ("moderate", "severe"):
                request = by_view[name]
                manifest = staging_manifests[name]
                dataset = _build_dataset(
                    request,
                    observed[name],
                    clean,
                    pre_dropout[name],
                    marker_truth,
                    manifest.manifest_sha256,
                    before_source,
                    before_environment,
                    config,
                    run_metadata,
                    pair_request_sha256,
                )
                staged_datasets[name] = _stage_h5ad(dataset, request.output_path)
                _temporary, staged_semantics = staged_datasets[name]
                staged_sha256 = benchmark_dataset_sha256(staged_semantics)
                staged_hashes[name] = staged_sha256
                SimulationArtifact(request, staged_semantics, manifest, staged_sha256)

            if not np.array_equal(
                staged_datasets["moderate"][1].layers["latent_expression"],
                staged_datasets["severe"][1].layers["latent_expression"],
            ):
                raise SimulationContractError(
                    "staged paired SERGIO datasets do not preserve identical latent truth"
                )
            validate_paired_simulation_requests(
                ordered_requests, protocol, final_manifest
            )
            if runtime_assets is not None:
                revalidate_simulator_runtime_asset_identity(runtime_assets)
            native_directory, native_created, native_identity = (
                _publish_native_directory(files, output_parent)
            )
            persistent_files = {
                name: native_directory / name for name in sorted(_EXPECTED_NATIVE_FILES)
            }
            manifests = {
                name: seal_native_outputs(persistent_files, manifest_metadata[name])
                for name in ("moderate", "severe")
            }
            for name in ("moderate", "severe"):
                if (
                    manifests[name].manifest_sha256
                    != staging_manifests[name].manifest_sha256
                ):
                    raise SimulationContractError(
                        "published SERGIO native manifest differs from staged bytes"
                    )

            artifacts: dict[str, SimulationArtifact] = {}
            for name in ("moderate", "severe"):
                request = by_view[name]
                temporary, _staged_semantics = staged_datasets[name]
                persisted, result_identity = _publish_staged_h5ad(
                    temporary, request.output_path
                )
                published_results.append((request.output_path, result_identity))
                dataset_sha256 = benchmark_dataset_sha256(persisted)
                if dataset_sha256 != staged_hashes[name]:
                    raise SimulationContractError(
                        "published SERGIO semantics differ from the staged roundtrip"
                    )
                artifacts[name] = SimulationArtifact(
                    request, persisted, manifests[name], dataset_sha256
                )
            if ordered_requests[0].namespace == protocol.final.namespace:
                _revalidate_published_final_claim(final_manifest)
            else:
                validate_paired_simulation_requests(
                    ordered_requests, protocol, final_manifest
                )
            publication_complete = True
        except BaseException:
            if not publication_complete:
                for path, identity in published_results:
                    _remove_owned_file(path, identity)
                if (
                    native_created
                    and native_directory is not None
                    and native_identity is not None
                ):
                    _remove_owned_directory(native_directory, native_identity)
                    try:
                        native_directory.parent.rmdir()
                    except OSError:
                        pass
            raise
        finally:
            for temporary, _dataset in staged_datasets.values():
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass

        return (
            artifacts[ordered_requests[0].technical_view],
            artifacts[ordered_requests[1].technical_view],
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


__all__ = ["map_sergio_numpy_seeds", "run_sergio_pair"]
