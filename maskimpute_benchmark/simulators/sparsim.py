"""Pinned, paired-view SPARSim validation adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

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


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER_PATH = _REPO_ROOT / "study/sources.json"
_EXTERNAL_ROOT = _REPO_ROOT / "artifacts/external"
_CHECKOUT = _EXTERNAL_ROOT / "checkouts/sparsim"
_R_ENVIRONMENT = _REPO_ROOT / "artifacts/envs/symsim-r44"
_RSCRIPT = _R_ENVIRONMENT / "bin/Rscript"
_COMPILER = Path("/usr/bin/g++")
_RUNNER = _REPO_ROOT / "scripts/simulators/run_sparsim.R"
_SPARSIM_COMMIT = "4e7712fb236a92ce7c173da169c8a29cc2a9f0ef"
_SPARSIM_TREE = "5d66b28cc6afd8d68364f4205cc983c7f681e2fe"
_EXPECTED_ENVIRONMENT_RECEIPT = {
    "schema": "maskimpute-sparsim-r-environment-v1",
    "sha256": "8441fdcee075818054991c2c487288811f810ec391f52aa77b09c98ded269292",
    "r_executable_sha256": (
        "6be9bb0438a23c06ff964b6f0fc01c0bfa657d33164bb409185f1d1a009763a3"
    ),
    "compiler": {
        "command": "/usr/bin/g++",
        "executable_sha256": (
            "1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769"
        ),
        "version_sha256": (
            "6665a44f75e5a8bfb50207f2ea30b25540e7528d70405ff29c9533bc8d36b468"
        ),
    },
    "package_count": 217,
}
_SOURCE_PATHS = {
    "cpp": "src/Random_number.cpp",
    "preset": "data/Chu_param_preset.RData",
    "simulate": "R/SPARSim_simulate.R",
    "utilities": "R/SPARSim_utilities.R",
}
_GROUP_PRESETS = {
    "chu-c1": "Chu_C1",
    "chu-c3": "Chu_C3",
    "chu-c6": "Chu_C6",
}
_SOURCE_GROUP_SIZES = {"chu-c1": 92, "chu-c3": 66, "chu-c6": 188}
_VIEW_PARAMETERS = {
    "moderate": {
        "library_size_divisor": 100,
        "library_size_rounding": "nearest_half_up_minimum_1",
    },
    "severe": {
        "library_size_divisor": 400,
        "library_size_rounding": "nearest_half_up_minimum_1",
    },
}
_EXPECTED_NATIVE_FILES = frozenset(
    {
        "cell_metadata.tsv",
        "config.json",
        "gene_metadata.tsv",
        "latent_expression.tsv",
        "observed_moderate.tsv",
        "observed_severe.tsv",
        "run_metadata.json",
    }
)
_MATRIX_FILES = (
    "latent_expression.tsv",
    "observed_moderate.tsv",
    "observed_severe.tsv",
)
_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
_FLOAT = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulationContractError(
                f"duplicate JSON key in SPARSim output: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SimulationContractError(f"non-finite JSON value in SPARSim output: {value}")


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


def _mapped_r_seed(seed: int, domain: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise SimulationContractError(
            "SPARSim seeds must be 63-bit nonnegative integers"
        )
    digest = canonical_sha256(
        {"schema": "maskimpute-sparsim-r-seed-v1", "domain": domain, "seed": seed}
    )
    return int(digest[:16], 16) % (2**31 - 1) + 1


def map_sparsim_r_seeds(
    biological_seed: int, moderate_seed: int, severe_seed: int
) -> dict[str, int]:
    """Map study seeds into distinct deterministic native-R seed values."""

    originals = {
        "biological": biological_seed,
        "moderate": moderate_seed,
        "severe": severe_seed,
    }
    mapped: dict[str, int] = {}
    used: set[int] = set()
    for role, seed in originals.items():
        candidate = _mapped_r_seed(seed, role)
        while candidate in used:
            candidate = candidate % (2**31 - 1) + 1
        mapped[role] = candidate
        used.add(candidate)
    return mapped


def _proportional_allocations(cells: int) -> dict[str, int]:
    if type(cells) is not int or cells < len(_SOURCE_GROUP_SIZES):
        raise SimulationContractError(
            "SPARSim requires at least one cell in each of its three Chu groups"
        )
    total = sum(_SOURCE_GROUP_SIZES.values())
    allocations = {
        name: cells * size // total for name, size in _SOURCE_GROUP_SIZES.items()
    }
    remainders = {
        name: (cells * size) % total for name, size in _SOURCE_GROUP_SIZES.items()
    }
    missing = cells - sum(allocations.values())
    order = sorted(
        _SOURCE_GROUP_SIZES,
        key=lambda name: (-remainders[name], tuple(_SOURCE_GROUP_SIZES).index(name)),
    )
    for name in order[:missing]:
        allocations[name] += 1
    if any(value <= 0 for value in allocations.values()):
        raise SimulationContractError(
            "SPARSim proportional allocation leaves an empty Chu group"
        )
    return allocations


def _verify_sparsim_source(
    *, external_root: Path | None = None, immutable: bool = False
) -> dict[str, object]:
    selected_root = _EXTERNAL_ROOT if external_root is None else external_root
    checkout = (
        _CHECKOUT if external_root is None else selected_root / "checkouts/sparsim"
    )
    try:
        ledger = load_source_ledger(_LEDGER_PATH)
        verifier = verify_fetched_sources if immutable else fetch_sources
        receipt = verifier(ledger, selected_root, source_ids=("sparsim",))[0]
    except (OSError, SourceLedgerError, TypeError, ValueError) as error:
        raise SimulationContractError(
            f"pinned SPARSim source is not pristine: {error}"
        ) from error
    checksum = receipt.get("verified_checksum")
    if (
        receipt.get("resolved_revision") != _SPARSIM_COMMIT
        or not isinstance(checksum, Mapping)
        or checksum.get("algorithm") != "git-tree-sha1"
        or checksum.get("value") != _SPARSIM_TREE
    ):
        raise SimulationContractError(
            "SPARSim source receipt does not match the exact pin"
        )
    for logical_path in _SOURCE_PATHS.values():
        path = checkout / logical_path
        if path.is_symlink() or not path.is_file():
            raise SimulationContractError(
                f"pinned SPARSim source file is unavailable: {logical_path}"
            )
    return receipt


def _source_file_receipt(
    *, external_root: Path | None = None
) -> dict[str, dict[str, str]]:
    selected_root = _EXTERNAL_ROOT if external_root is None else external_root
    checkout = (
        _CHECKOUT if external_root is None else selected_root / "checkouts/sparsim"
    )
    receipt: dict[str, dict[str, str]] = {}
    for role, logical_path in _SOURCE_PATHS.items():
        path = checkout / logical_path
        if path.is_symlink() or not path.is_file():
            raise SimulationContractError(
                f"pinned SPARSim source file is unavailable: {logical_path}"
            )
        try:
            digest = file_sha256(path)
        except OSError as error:
            raise SimulationContractError(
                f"pinned SPARSim source file cannot be hashed: {logical_path}"
            ) from error
        receipt[role] = {"path": logical_path, "sha256": digest}
    return receipt


def _compiler_version_sha256() -> str:
    try:
        completed = subprocess.run(
            [_COMPILER.as_posix(), "--version"],
            check=True,
            capture_output=True,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "TZ": "UTC",
            },
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SimulationContractError(
            "SPARSim compiler version cannot be fingerprinted"
        ) from error
    return canonical_sha256(
        {
            "schema": "maskimpute-compiler-version-v1",
            "stdout_hex": completed.stdout.hex(),
            "stderr_hex": completed.stderr.hex(),
        }
    )


def _environment_receipt(*, r_environment: Path | None = None) -> dict[str, object]:
    selected_environment = _R_ENVIRONMENT if r_environment is None else r_environment
    rscript = (
        _RSCRIPT if r_environment is None else selected_environment / "bin/Rscript"
    )
    if (
        selected_environment.is_symlink()
        or not selected_environment.is_dir()
        or rscript.is_symlink()
        or not rscript.is_file()
    ):
        raise SimulationContractError("pinned SPARSim R environment is unavailable")
    try:
        compiler = _COMPILER.resolve(strict=True)
    except OSError as error:
        raise SimulationContractError("SPARSim compiler is unavailable") from error
    if not compiler.is_file():
        raise SimulationContractError("SPARSim compiler is not a regular file")
    records: list[dict[str, object]] = []
    try:
        for path in sorted((selected_environment / "conda-meta").glob("*.json")):
            value = _load_json_bytes(
                _canonical_json_bytes(json.loads(path.read_text(encoding="utf-8"))),
                f"environment record {path.name}",
            )
            if not isinstance(value, Mapping):
                raise SimulationContractError(
                    "Conda package record must be a JSON object"
                )
            record = {
                key: value.get(key)
                for key in (
                    "name",
                    "version",
                    "build",
                    "build_number",
                    "channel",
                    "subdir",
                    "sha256",
                    "md5",
                )
            }
            if not all(
                isinstance(record[key], str) and record[key]
                for key in ("name", "version", "build")
            ):
                raise SimulationContractError("Conda package identity is incomplete")
            records.append(record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimulationContractError(
            "pinned SPARSim R environment could not be fingerprinted"
        ) from error
    if not records:
        raise SimulationContractError(
            "pinned SPARSim R environment has no package lock"
        )
    r_sha256 = file_sha256(rscript)
    compiler_sha256 = file_sha256(compiler)
    compiler_version_sha256 = _compiler_version_sha256()
    payload = {
        "schema": "maskimpute-sparsim-r-environment-v1",
        "r_executable_sha256": r_sha256,
        "compiler": {
            "command": _COMPILER.as_posix(),
            "executable_sha256": compiler_sha256,
            "version_sha256": compiler_version_sha256,
        },
        "packages": records,
    }
    receipt = {
        "schema": payload["schema"],
        "sha256": canonical_sha256(payload),
        "r_executable_sha256": r_sha256,
        "compiler": payload["compiler"],
        "package_count": len(records),
    }
    if receipt != _EXPECTED_ENVIRONMENT_RECEIPT:
        raise SimulationContractError(
            "SPARSim R environment or /usr/bin/g++ differs from the exact pin"
        )
    return receipt


def _execute_sparsim(
    config_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: int,
    external_root: Path | None = None,
    r_environment: Path | None = None,
    direct_r: bool = False,
) -> None:
    selected_root = _EXTERNAL_ROOT if external_root is None else external_root
    selected_environment = _R_ENVIRONMENT if r_environment is None else r_environment
    rscript = (
        _RSCRIPT if r_environment is None else selected_environment / "bin/Rscript"
    )
    checkout = (
        _CHECKOUT if external_root is None else selected_root / "checkouts/sparsim"
    )
    if _RUNNER.is_symlink() or not _RUNNER.is_file():
        raise SimulationContractError("tracked SPARSim R runner is unavailable")
    with tempfile.TemporaryDirectory(prefix="maskimpute-sparsim-build-") as build:
        build_path = Path(build)
        makevars = build_path / "Makevars"
        makevars.write_text(
            "CXX11=/usr/bin/g++\nCXX14=/usr/bin/g++\nCXX17=/usr/bin/g++\n",
            encoding="utf-8",
        )
        environment = {
            "HOME": build_path.as_posix(),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": f"{(selected_environment / 'bin').as_posix()}:/usr/bin:/bin",
            "R_ENVIRON_USER": "/dev/null",
            "R_MAKEVARS_USER": makevars.as_posix(),
            "R_PROFILE_USER": "/dev/null",
            "TZ": "UTC",
        }
        if direct_r:
            r_home = selected_environment / "lib/R"
            r_library = r_home / "library"
            executable = r_home / "bin/exec/R"
            environment.update(
                {
                    "R_HOME": r_home.as_posix(),
                    "R_LIBS": r_library.as_posix(),
                    "R_LIBS_SITE": r_library.as_posix(),
                    "R_LIBS_USER": r_library.as_posix(),
                }
            )
            command = [
                executable.as_posix(),
                "--vanilla",
                "--slave",
                f"--file={_RUNNER.as_posix()}",
                "--args",
                config_path.as_posix(),
                checkout.as_posix(),
                output_dir.as_posix(),
                build_path.as_posix(),
            ]
        else:
            command = [
                rscript.as_posix(),
                "--vanilla",
                _RUNNER.as_posix(),
                config_path.as_posix(),
                checkout.as_posix(),
                output_dir.as_posix(),
                build_path.as_posix(),
            ]
        try:
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SimulationContractError(
                f"SPARSim native runner failed: {error}"
            ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-4000:]
        raise SimulationContractError(
            f"SPARSim native runner exited {completed.returncode}: {stderr}"
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
                f"SPARSim native output must be a unique regular file: {path.name}"
            )
        descriptor = os.open(path, flags)
    except SimulationContractError:
        raise
    except OSError as error:
        raise SimulationContractError(
            f"SPARSim native output cannot be opened safely: {path.name}"
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
                f"SPARSim native output changed while opening: {path.name}"
            )
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise SimulationContractError(
                f"SPARSim native output is too large: {path.name}"
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
                    f"SPARSim native output grew beyond its limit: {path.name}"
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
                f"SPARSim native output changed while reading: {path.name}"
            )
        return b"".join(chunks)
    except OSError as error:
        raise SimulationContractError(
            f"SPARSim native output changed while reading: {path.name}"
        ) from error
    finally:
        os.close(descriptor)


def _read_tsv(path: Path, *, maximum_bytes: int) -> list[list[str]]:
    data = _read_regular_bytes(path, maximum_bytes=maximum_bytes)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SimulationContractError(f"{path.name} is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text or "\x00" in text or '"' in text:
        raise SimulationContractError(f"{path.name} is not canonical TSV")
    try:
        rows = list(csv.reader(text.splitlines(), delimiter="\t", strict=True))
    except csv.Error as error:
        raise SimulationContractError(f"{path.name} is malformed TSV") from error
    if not rows or any(not row for row in rows):
        raise SimulationContractError(f"{path.name} is empty or malformed")
    return rows


def _read_matrix(
    path: Path,
    gene_ids: list[str],
    cell_ids: list[str],
    *,
    integer: bool,
) -> np.ndarray:
    maximum_bytes = (len(gene_ids) + 1) * (len(cell_ids) + 1) * 40 + 1024
    rows = _read_tsv(path, maximum_bytes=maximum_bytes)
    if rows[0] != ["gene_id", *cell_ids] or len(rows) != len(gene_ids) + 1:
        raise SimulationContractError(f"{path.name} has wrong orientation or IDs")
    dtype = np.int64 if integer else np.float64
    matrix = np.empty((len(gene_ids), len(cell_ids)), dtype=dtype)
    maximum = np.iinfo(np.int64).max
    for row_index, (expected_gene, row) in enumerate(
        zip(gene_ids, rows[1:], strict=True)
    ):
        if len(row) != len(cell_ids) + 1 or row[0] != expected_gene:
            raise SimulationContractError(f"{path.name} has wrong shape or gene IDs")
        for column_index, text in enumerate(row[1:]):
            if integer:
                if not _INTEGER.fullmatch(text):
                    raise SimulationContractError(
                        f"{path.name} must contain nonnegative integer counts"
                    )
                value = int(text)
                if value > maximum:
                    raise SimulationContractError(f"{path.name} count exceeds int64")
            else:
                if not _FLOAT.fullmatch(text):
                    raise SimulationContractError(
                        f"{path.name} must contain canonical nonnegative numbers"
                    )
                value = float(text)
                if (
                    not math.isfinite(value)
                    or value < 0
                    or text != format(value, ".17g")
                ):
                    raise SimulationContractError(
                        f"{path.name} must contain finite nonnegative expression"
                    )
            matrix[row_index, column_index] = value
    return matrix


def _read_groups(
    path: Path, cell_ids: list[str], allocations: Mapping[str, object]
) -> np.ndarray:
    rows = _read_tsv(path, maximum_bytes=(len(cell_ids) + 1) * 80)
    if rows[0] != ["cell_id", "group"] or len(rows) != len(cell_ids) + 1:
        raise SimulationContractError("cell_metadata.tsv has wrong shape")
    expected: list[str] = []
    for name in _GROUP_PRESETS:
        count = allocations.get(name)
        if type(count) is not int or count <= 0:
            raise SimulationContractError("SPARSim group allocation is invalid")
        expected.extend([name] * count)
    groups: list[str] = []
    for cell_id, row in zip(cell_ids, rows[1:], strict=True):
        if len(row) != 2 or row[0] != cell_id or row[1] not in _GROUP_PRESETS:
            raise SimulationContractError(
                "cell_metadata.tsv has invalid cell IDs or groups"
            )
        groups.append(row[1])
    if groups != expected:
        raise SimulationContractError(
            "SPARSim groups do not match the proportional Chu allocation"
        )
    return np.asarray(groups, dtype=object)


def _read_gene_metadata(path: Path, gene_ids: list[str]) -> list[str]:
    rows = _read_tsv(path, maximum_bytes=(len(gene_ids) + 1) * 256)
    if rows[0] != ["gene_id", "source_gene_id"] or len(rows) != len(gene_ids) + 1:
        raise SimulationContractError("gene_metadata.tsv has wrong shape")
    source_ids: list[str] = []
    for gene_id, row in zip(gene_ids, rows[1:], strict=True):
        if len(row) != 2 or row[0] != gene_id or not row[1].strip():
            raise SimulationContractError(
                "gene_metadata.tsv has invalid generic or source gene IDs"
            )
        source_ids.append(row[1])
    if len(source_ids) != len(set(source_ids)):
        raise SimulationContractError(
            "gene_metadata.tsv source gene IDs are not unique"
        )
    return source_ids


def _validate_stage_entries(stage: Path) -> dict[str, Path]:
    try:
        entries = list(os.scandir(stage))
    except OSError as error:
        raise SimulationContractError(
            "SPARSim native output directory is unavailable"
        ) from error
    names = {entry.name for entry in entries}
    if names != _EXPECTED_NATIVE_FILES or len(entries) != len(_EXPECTED_NATIVE_FILES):
        raise SimulationContractError(
            "SPARSim native outputs do not match the closed file set"
        )
    result: dict[str, Path] = {}
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise SimulationContractError(
                "SPARSim native output cannot be inspected"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SimulationContractError(
                "SPARSim native outputs must be unique regular files without symlinks"
            )
        result[entry.name] = Path(entry.path)
    return result


def _validate_run_metadata(
    path: Path,
    files: Mapping[str, Path],
    config: Mapping[str, object],
    environment: Mapping[str, object],
    config_sha256: str,
) -> dict[str, object]:
    value = _load_json_bytes(
        _read_regular_bytes(path, maximum_bytes=1024 * 1024), path.name
    )
    if not isinstance(value, dict):
        raise SimulationContractError("run_metadata.json must be an object")
    expected_keys = {
        "array_sha256",
        "biological_seed_r",
        "cells",
        "compiler_sha256",
        "config_sha256",
        "gene_matrix_equal",
        "genes",
        "group_allocations",
        "measurement_seeds_r",
        "r_version",
        "rcpp_version",
        "schema_version",
        "source_cpp_calls",
        "sparsim_simulation_calls",
        "views",
    }
    simulation = config.get("simulation")
    seeds = config.get("seeds")
    views = config.get("views")
    compiler = environment.get("compiler")
    if (
        not isinstance(simulation, Mapping)
        or not isinstance(seeds, Mapping)
        or not isinstance(views, list)
        or not isinstance(compiler, Mapping)
    ):
        raise SimulationContractError("SPARSim config binding is invalid")
    biological = seeds.get("biological")
    if not isinstance(biological, Mapping):
        raise SimulationContractError("SPARSim biological seed binding is invalid")
    expected_measurement = {
        view["technical_view"]: view["measurement_seed_r"]
        for view in views
        if isinstance(view, Mapping)
    }
    expected_binding = {
        "biological_seed_r": biological.get("mapped_r"),
        "cells": simulation.get("cells"),
        "compiler_sha256": compiler.get("executable_sha256"),
        "config_sha256": config_sha256,
        "gene_matrix_equal": True,
        "genes": simulation.get("genes"),
        "group_allocations": simulation.get("group_allocations"),
        "measurement_seeds_r": expected_measurement,
        "schema_version": 1,
        "source_cpp_calls": 1,
        "sparsim_simulation_calls": 2,
        "views": [
            view.get("technical_view") for view in views if isinstance(view, Mapping)
        ],
    }
    observed_binding = {key: value.get(key) for key in expected_binding}
    hashes = value.get("array_sha256")
    integer_fields = {
        "biological_seed_r",
        "cells",
        "genes",
        "schema_version",
        "source_cpp_calls",
        "sparsim_simulation_calls",
    }
    observed_allocations = value.get("group_allocations")
    observed_measurement = value.get("measurement_seeds_r")
    observed_views = value.get("views")
    if (
        set(value) != expected_keys
        or observed_binding != expected_binding
        or any(type(value.get(name)) is not int for name in integer_fields)
        or type(value.get("gene_matrix_equal")) is not bool
        or not isinstance(observed_allocations, dict)
        or any(type(item) is not int for item in observed_allocations.values())
        or not isinstance(observed_measurement, dict)
        or any(type(item) is not int for item in observed_measurement.values())
        or not isinstance(observed_views, list)
        or any(type(item) is not str for item in observed_views)
        or not isinstance(value.get("r_version"), str)
        or not value["r_version"]
        or not isinstance(value.get("rcpp_version"), str)
        or not value["rcpp_version"]
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(_MATRIX_FILES)
    ):
        raise SimulationContractError(
            "run_metadata.json does not bind the exact paired run"
        )
    for name in _MATRIX_FILES:
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
                "SPARSim output path cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SimulationContractError(
                "SPARSim output path must not contain symlinks"
            )


def _path_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SimulationContractError(
            f"SPARSim path identity is unavailable: {path}"
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
        {"schema": "maskimpute-sparsim-native-v1", "files": descriptor}
    )[:24]
    _reject_output_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_output_symlink_components(parent)
    native_root = parent / "native"
    if native_root.is_symlink() or (native_root.exists() and not native_root.is_dir()):
        raise SimulationContractError("SPARSim native output root is invalid")
    native_root.mkdir(mode=0o755, exist_ok=True)
    destination = native_root / f"sparsim-{content_id}"
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise SimulationContractError(
                "existing SPARSim native directory is invalid"
            )
        existing = _validate_stage_entries(destination)
        if _native_descriptor(existing) != descriptor:
            raise SimulationContractError("existing SPARSim native bytes changed")
        return destination, False, _path_identity(destination)
    publication = Path(tempfile.mkdtemp(prefix=".sparsim-publish-", dir=native_root))
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
                "SPARSim native bytes changed while publishing"
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
            "SPARSim native outputs could not be published"
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
            raise SimulationContractError("SPARSim output has no existing ancestor")
        current = current.parent
    try:
        current_metadata = current.lstat()
    except OSError as error:
        raise SimulationContractError(
            "SPARSim output ancestor cannot be inspected"
        ) from error
    if not stat.S_ISDIR(current_metadata.st_mode) or stat.S_ISLNK(
        current_metadata.st_mode
    ):
        raise SimulationContractError(
            "SPARSim output ancestor must be a non-symlink directory"
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
            "SPARSim h5ad staging root cannot be inspected"
        ) from error
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or root_metadata.st_dev != current_metadata.st_dev
    ):
        raise SimulationContractError(
            "SPARSim h5ad staging root must be a same-device non-symlink directory"
        )
    return root


def _stage_h5ad(adata: ad.AnnData, destination: Path) -> tuple[Path, ad.AnnData]:
    _reject_output_symlink_components(destination)
    staging_root = _h5ad_staging_root(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="maskimpute-sparsim-h5ad-", suffix=".h5ad", dir=staging_root
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
            "SPARSim dataset could not be serialized for atomic persistence: "
            f"{destination}"
        ) from error


def _publish_staged_h5ad(
    temporary: Path, destination: Path
) -> tuple[ad.AnnData, tuple[int, int]]:
    _reject_output_symlink_components(destination)
    if os.path.lexists(destination):
        raise SimulationContractError(
            f"SPARSim refuses to overwrite an existing result: {destination}"
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
            f"SPARSim dataset could not be atomically persisted: {destination}"
        ) from error


def _marker_truth(latent_gene_cell: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    gene_ids = _expected_ids("gene", latent_gene_cell.shape[0])
    values: dict[str, np.ndarray] = {}
    pseudocount = 1e-8
    for group in _GROUP_PRESETS:
        mask = groups == group
        group_mean = latent_gene_cell[:, mask].mean(axis=1, dtype=np.float64)
        other_mean = latent_gene_cell[:, ~mask].mean(axis=1, dtype=np.float64)
        log2fc = np.log2((group_mean + pseudocount) / (other_mean + pseudocount))
        suffix = group.replace("-", "_")
        values[f"latent_log2fc_{suffix}"] = log2fc.astype(np.float64, copy=False)
        values[f"marker_{suffix}"] = log2fc > 1.0
    return pd.DataFrame(values, index=gene_ids)


def _build_dataset(
    request: SimulationRequest,
    observed_gene_cell: np.ndarray,
    latent_gene_cell: np.ndarray,
    groups: np.ndarray,
    marker_truth: pd.DataFrame,
    source_gene_ids: list[str],
    native_manifest_sha256: str,
    source_receipt: Mapping[str, object],
    environment: Mapping[str, object],
    config: Mapping[str, object],
    run_metadata: Mapping[str, object],
    pair_request_sha256: str,
) -> ad.AnnData:
    observed = observed_gene_cell.T.copy()
    latent = latent_gene_cell.T.copy()
    library_sizes = [sum(int(value) for value in row) for row in observed]
    if any(total > np.iinfo(np.int64).max for total in library_sizes):
        raise SimulationContractError("SPARSim observed library size exceeds int64")
    draw = int(request.biological_id.removeprefix("draw-"))
    obs = pd.DataFrame(
        {
            "dataset_id": [request.dataset_id] * request.cells,
            "mechanism": [request.mechanism] * request.cells,
            "condition": [request.technical_view] * request.cells,
            "biological_id": [request.biological_id] * request.cells,
            "technical_view": [request.technical_view] * request.cells,
            "draw": np.full(request.cells, draw, dtype=np.int64),
            "library_size": np.asarray(library_sizes, dtype=np.int64),
            "group": groups.tolist(),
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
    var = marker_truth.copy(deep=True)
    var["source_gene_id"] = source_gene_ids
    dataset = ad.AnnData(
        X=observed,
        obs=obs,
        var=var,
        layers={"latent_expression": latent},
    )
    dataset.uns.update(
        {
            "truth_kind": "exact_continuous",
            "primary_truth_layer": "latent_expression",
            "allowed_covariates": {"obs": [], "var": []},
            "provenance": {
                "source": source_receipt["source_url"],
                "source_sha256": canonical_sha256(source_receipt),
                "software": "SPARSim",
                "software_version": source_receipt["resolved_revision"],
                "parameters": {
                    "adapter": config["adapter"],
                    "adapter_schema": "maskimpute-sparsim-adapter-v1",
                    "environment": environment,
                    "marker_definition": {
                        "estimand": "latent_mean_chu_group_vs_all_other_cells",
                        "log2fc_threshold": 1.0,
                        "pseudocount": 1e-8,
                    },
                    "measurement": {
                        key: value
                        for key, value in view_config.items()
                        if key
                        not in {
                            "measurement_seed_original",
                            "measurement_seed_r",
                            "technical_view",
                        }
                    },
                    "native_manifest_sha256": native_manifest_sha256,
                    "native_run_metadata": run_metadata,
                    "pair_request_sha256": pair_request_sha256,
                    "score_truth": "undefined_for_continuous_truth",
                    "simulation": config["simulation"],
                    "source": config["source"],
                    "source_receipt": source_receipt,
                },
                "seeds": {
                    "biological": request.biological_seed,
                    "measurement": request.measurement_seed,
                    "r_biological": biological_config["mapped_r"],
                    "r_measurement": view_config["measurement_seed_r"],
                },
            },
        }
    )
    return dataset


def _pair_config(
    requests: Mapping[str, SimulationRequest],
    environment: Mapping[str, object],
    source_files: Mapping[str, object],
) -> dict[str, object]:
    moderate = requests["moderate"]
    severe = requests["severe"]
    mapped = map_sparsim_r_seeds(
        moderate.biological_seed,
        moderate.measurement_seed,
        severe.measurement_seed,
    )
    compiler = environment.get("compiler")
    environment_sha256 = environment.get("sha256")
    if (
        not isinstance(compiler, Mapping)
        or not isinstance(compiler.get("executable_sha256"), str)
        or not isinstance(environment_sha256, str)
    ):
        raise SimulationContractError("SPARSim environment receipt is incomplete")
    views: list[dict[str, object]] = []
    for name in ("moderate", "severe"):
        request = requests[name]
        views.append(
            {
                "technical_view": name,
                "measurement_seed_original": request.measurement_seed,
                "measurement_seed_r": mapped[name],
                **_VIEW_PARAMETERS[name],
            }
        )
    if set(source_files) != set(_SOURCE_PATHS):
        raise SimulationContractError("SPARSim source-file receipt is incomplete")
    return {
        "adapter": {
            "python_adapter_sha256": file_sha256(Path(__file__)),
            "r_runner_sha256": file_sha256(_RUNNER),
        },
        "environment": {
            "compiler_executable_sha256": compiler["executable_sha256"],
            "environment_sha256": environment_sha256,
        },
        "schema_version": 1,
        "seeds": {
            "biological": {
                "original": moderate.biological_seed,
                "mapped_r": mapped["biological"],
            }
        },
        "simulation": {
            "cells": moderate.cells,
            "gene_selection": "sha256_ranked_source_gene_id_v1",
            "gene_selection_domain": "maskimpute-sparsim-gene-v1",
            "genes": moderate.genes,
            "group_allocations": _proportional_allocations(moderate.cells),
            "group_presets": _GROUP_PRESETS,
            "library_template_selection": "midpoint_quantile_with_replacement",
            "source_group_sizes": _SOURCE_GROUP_SIZES,
        },
        "source": {
            "commit": _SPARSIM_COMMIT,
            "files": source_files,
            "tree": _SPARSIM_TREE,
        },
        "views": views,
    }


def _revalidate_published_final_claim(claim: FinalManifestClaim | None) -> None:
    """Recheck lifecycle records after unreceipted SPARSim publication."""

    from .. import study

    if not isinstance(claim, FinalManifestClaim):
        raise SimulationContractError(
            "published final SPARSim pair requires its original execution claim"
        )
    try:
        repository = claim._repository
        destination = claim.round_dir
        canonical_repository, canonical_destination = study._repository_for_round(
            destination, repository
        )
        if canonical_repository != repository or canonical_destination != destination:
            raise SimulationContractError(
                "final SPARSim claim changed repository identity"
            )
        with study._round_lock(repository, destination.name) as lock_identity:
            freeze = study._validate_freeze(destination, repository)
            if freeze.get("protocol_sha256") != claim._protocol_sha256:
                raise SimulationContractError(
                    "final SPARSim claim changed frozen protocol"
                )
            study._validate_registry(
                repository, destination, freeze, expected_state="running"
            )
            materialization, manifest = study._validate_seed_manifest(
                destination, freeze
            )
            execution = study._validate_execution_claim_record(
                destination, freeze, materialization
            )
            study._assert_round_lock_identity(
                repository, destination.name, lock_identity
            )
            current_binding = (
                manifest.get("round_id"),
                tuple(manifest.get("generator_seeds", ())),
                materialization.get("seed_manifest_sha256"),
                execution.get("execution_claim_id"),
                destination,
                repository,
                freeze.get("protocol_sha256"),
            )
            expected_binding = (
                claim.round_id,
                claim.generator_seeds,
                claim.seed_manifest_sha256,
                claim.execution_claim_id,
                claim.round_dir,
                claim._repository,
                claim._protocol_sha256,
            )
            if current_binding != expected_binding:
                raise SimulationContractError(
                    "final SPARSim execution claim changed during publication"
                )
    except SimulationContractError:
        raise
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        study.StudyStateError,
    ) as error:
        raise SimulationContractError(
            f"final SPARSim execution is no longer the claimed running round: {error}"
        ) from error


def run_sparsim_pair(
    requests: Sequence[SimulationRequest],
    protocol: Protocol,
    final_manifest: FinalManifestClaim | None = None,
    *,
    runtime_assets: SimulatorRuntimeAssets | None = None,
) -> tuple[SimulationArtifact, SimulationArtifact]:
    """Generate paired count-depth views from one exact SPARSim gene matrix."""

    try:
        ordered_requests = tuple(requests)
    except TypeError as error:
        raise SimulationContractError(
            "SPARSim requests must be a finite pair"
        ) from error
    validate_paired_simulation_requests(ordered_requests, protocol, final_manifest)
    if any(request.mechanism != "sparsim" for request in ordered_requests):
        raise SimulationContractError("SPARSim adapter accepts only sparsim requests")
    by_view = {request.technical_view: request for request in ordered_requests}
    if set(by_view) != {"moderate", "severe"} or len(by_view) != 2:
        raise SimulationContractError(
            "SPARSim adapter requires exactly moderate and severe technical views"
        )
    moderate = by_view["moderate"]
    _proportional_allocations(moderate.cells)
    if moderate.genes > 17_782:
        raise SimulationContractError("SPARSim Chu presets contain at most 17782 genes")
    parents = {request.output_path.parent.absolute() for request in ordered_requests}
    if len(parents) != 1:
        raise SimulationContractError("paired SPARSim outputs must share one directory")
    output_parent = next(iter(parents))
    _reject_output_symlink_components(output_parent)
    for request in ordered_requests:
        if os.path.lexists(request.output_path):
            raise SimulationContractError(
                f"SPARSim refuses to overwrite an existing result: {request.output_path}"
            )

    runtime_assets_sha256: str | None = None
    if runtime_assets is None:
        verify_source = _verify_sparsim_source
        source_file_receipt = _source_file_receipt
        environment_receipt = _environment_receipt
        execute = _execute_sparsim
    else:
        external_root, r_environment, runtime_assets_sha256 = (
            simulator_runtime_asset_values(runtime_assets)
        )
        verify_source = lambda: simulator_runtime_source_receipt(  # noqa: E731
            runtime_assets, "sparsim"
        )
        source_file_receipt = lambda: _source_file_receipt(  # noqa: E731
            external_root=external_root
        )
        environment_receipt = lambda: _environment_receipt(  # noqa: E731
            r_environment=r_environment
        )

        def execute(
            config_path: Path, output_dir: Path, *, timeout_seconds: int
        ) -> None:
            _execute_sparsim(
                config_path,
                output_dir,
                timeout_seconds=timeout_seconds,
                external_root=external_root,
                r_environment=r_environment,
                direct_r=True,
            )

    before_source = verify_source()
    before_source_files = source_file_receipt()
    before_environment = environment_receipt()
    config = _pair_config(by_view, before_environment, before_source_files)
    config_bytes = _canonical_json_bytes(config)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    stage = Path(tempfile.mkdtemp(prefix="maskimpute-sparsim-native-"))
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
        except BaseException as error:
            runner_error = error
        try:
            after_source = verify_source()
            after_source_files = source_file_receipt()
            after_environment = environment_receipt()
        except Exception as error:
            raise SimulationContractError(
                "SPARSim source or environment was not pristine after execution"
            ) from error
        if canonical_sha256(before_source) != canonical_sha256(after_source):
            raise SimulationContractError(
                "SPARSim source receipt changed during execution"
            )
        if canonical_sha256(before_source_files) != canonical_sha256(
            after_source_files
        ):
            raise SimulationContractError(
                "SPARSim source file bytes changed during execution"
            )
        if canonical_sha256(before_environment) != canonical_sha256(after_environment):
            raise SimulationContractError(
                "SPARSim environment changed during execution"
            )
        if runner_error is not None:
            if isinstance(runner_error, SimulationContractError):
                raise runner_error
            if not isinstance(runner_error, Exception):
                raise runner_error
            raise SimulationContractError(
                "SPARSim native runner failed"
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
                "SPARSim native runner changed its sealed config"
            )
        gene_ids = _expected_ids("gene", moderate.genes)
        cell_ids = _expected_ids("cell", moderate.cells)
        latent = _read_matrix(
            files["latent_expression.tsv"],
            gene_ids,
            cell_ids,
            integer=False,
        )
        observed = {
            name: _read_matrix(
                files[f"observed_{name}.tsv"],
                gene_ids,
                cell_ids,
                integer=True,
            )
            for name in ("moderate", "severe")
        }
        if np.array_equal(observed["moderate"], observed["severe"]):
            raise SimulationContractError(
                "paired SPARSim technical views must have different observed counts"
            )
        allocations = config["simulation"]["group_allocations"]
        assert isinstance(allocations, Mapping)
        groups = _read_groups(files["cell_metadata.tsv"], cell_ids, allocations)
        source_gene_ids = _read_gene_metadata(files["gene_metadata.tsv"], gene_ids)
        run_metadata = _validate_run_metadata(
            files["run_metadata.json"],
            files,
            config,
            before_environment,
            config_sha256,
        )
        marker_truth = _marker_truth(latent, groups)
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
                "adapter_schema": "maskimpute-sparsim-native-v1",
                "config_sha256": config_sha256,
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
                    latent,
                    groups,
                    marker_truth,
                    source_gene_ids,
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
                    "staged SPARSim pair does not preserve identical latent truth"
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
                        "published SPARSim native manifest differs from staged bytes"
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
                        "published SPARSim semantics differ from staged roundtrip"
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
            try:
                shutil.rmtree(stage)
            except OSError:
                # The pair is already atomically committed when
                # publication_complete is true. Temporary cleanup must not
                # turn that success into an unreceipted apparent failure; on
                # earlier failures it must likewise not mask the root cause.
                pass


__all__ = ["map_sparsim_r_seeds", "run_sparsim_pair"]
