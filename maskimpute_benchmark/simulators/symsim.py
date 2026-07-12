"""Pinned, paired-view SymSim validation adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
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
from ..sources import SourceLedgerError, fetch_sources, load_source_ledger
from .base import (
    FinalManifestClaim,
    SimulationArtifact,
    SimulationContractError,
    SimulationRequest,
    simulation_scientific_identity,
    validate_paired_simulation_requests,
)
from .native import seal_native_outputs


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER_PATH = _REPO_ROOT / "study/sources.json"
_EXTERNAL_ROOT = _REPO_ROOT / "artifacts/external"
_CHECKOUT = _EXTERNAL_ROOT / "checkouts/symsim"
_R_ENVIRONMENT = _REPO_ROOT / "artifacts/envs/symsim-r44"
_RSCRIPT = _R_ENVIRONMENT / "bin/Rscript"
_RUNNER = _REPO_ROOT / "scripts/simulators/run_symsim.R"
_SYMSIM_COMMIT = "76a674b407ce44bf2690a9161cf28b905598d0a5"
_SYMSIM_TREE = "12d9c7e9e8c22bb0bae917aec7860627dcb8489b"
_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
_EXPECTED_NATIVE_FILES = frozenset(
    {
        "config.json",
        "true_counts.tsv",
        "observed_moderate.tsv",
        "observed_severe.tsv",
        "cell_metadata.tsv",
        "marker_truth.tsv",
        "run_metadata.json",
    }
)
_VIEW_PARAMETERS: dict[str, dict[str, int | float | str]] = {
    "moderate": {
        "protocol": "UMI",
        "alpha_mean": 0.10,
        "alpha_sd": 0.002,
        "depth_mean": 50_000,
        "depth_sd": 3_000,
    },
    "severe": {
        "protocol": "UMI",
        "alpha_mean": 0.05,
        "alpha_sd": 0.002,
        "depth_mean": 25_000,
        "depth_sd": 1_500,
    },
}


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulationContractError(f"duplicate JSON key in SymSim output: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SimulationContractError(f"non-finite JSON value in SymSim output: {value}")


def _load_json_bytes(data: bytes, name: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SimulationContractError(f"{name} is not strict UTF-8 JSON") from error


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _mapped_r_seed(seed: int, domain: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise SimulationContractError(
            "SymSim seeds must be 63-bit nonnegative integers"
        )
    digest = canonical_sha256(
        {"schema": "maskimpute-symsim-r-seed-v1", "domain": domain, "seed": seed}
    )
    return int(digest[:16], 16) % (2**31 - 1) + 1


def map_symsim_r_seeds(
    biological_seed: int, moderate_seed: int, severe_seed: int
) -> dict[str, int]:
    """Map 63-bit study seeds into distinct deterministic native-R seeds."""

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


def _verify_symsim_source() -> dict[str, object]:
    try:
        ledger = load_source_ledger(_LEDGER_PATH)
        receipt = fetch_sources(ledger, _EXTERNAL_ROOT, source_ids=("symsim",))[0]
    except (OSError, SourceLedgerError, TypeError, ValueError) as error:
        raise SimulationContractError(
            f"pinned SymSim source is not pristine: {error}"
        ) from error
    checksum = receipt.get("verified_checksum")
    if (
        receipt.get("resolved_revision") != _SYMSIM_COMMIT
        or not isinstance(checksum, Mapping)
        or checksum.get("algorithm") != "git-tree-sha1"
        or checksum.get("value") != _SYMSIM_TREE
    ):
        raise SimulationContractError(
            "SymSim source receipt does not match the exact pin"
        )
    return receipt


def _environment_receipt() -> dict[str, object]:
    if (
        _R_ENVIRONMENT.is_symlink()
        or not _R_ENVIRONMENT.is_dir()
        or _RSCRIPT.is_symlink()
        or not _RSCRIPT.is_file()
    ):
        raise SimulationContractError("pinned SymSim R environment is unavailable")
    records: list[dict[str, object]] = []
    try:
        for path in sorted((_R_ENVIRONMENT / "conda-meta").glob("*.json")):
            value = _load_json_bytes(
                path.read_bytes(), f"environment record {path.name}"
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
    except OSError as error:
        raise SimulationContractError(
            "pinned SymSim R environment could not be fingerprinted"
        ) from error
    if not records:
        raise SimulationContractError("pinned SymSim R environment has no package lock")
    executable_sha256 = file_sha256(_RSCRIPT)
    payload = {
        "schema": "maskimpute-conda-environment-v1",
        "r_executable_sha256": executable_sha256,
        "packages": records,
    }
    return {
        "schema": payload["schema"],
        "sha256": canonical_sha256(payload),
        "r_executable_sha256": executable_sha256,
        "package_count": len(records),
    }


def _execute_symsim(
    config_path: Path, output_dir: Path, *, timeout_seconds: int
) -> None:
    if not _RUNNER.is_file() or _RUNNER.is_symlink():
        raise SimulationContractError("tracked SymSim R runner is unavailable")
    environment = {
        "HOME": output_dir.as_posix(),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{(_R_ENVIRONMENT / 'bin').as_posix()}:/usr/bin:/bin",
        "R_ENVIRON_USER": "/dev/null",
        "R_PROFILE_USER": "/dev/null",
        "TZ": "UTC",
    }
    try:
        completed = subprocess.run(
            [
                _RSCRIPT.as_posix(),
                "--vanilla",
                _RUNNER.as_posix(),
                config_path.as_posix(),
                _CHECKOUT.as_posix(),
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
            f"SymSim native runner failed: {error}"
        ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-4000:]
        raise SimulationContractError(
            f"SymSim native runner exited {completed.returncode}: {stderr}"
        )


def _expected_ids(prefix: str, count: int) -> list[str]:
    width = max(4, len(str(count)))
    return [f"{prefix}-{index:0{width}d}" for index in range(1, count + 1)]


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.lstat()
        if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
            raise SimulationContractError(
                f"SymSim native output must be a unique regular file: {path.name}"
            )
        descriptor = os.open(path, flags)
    except SimulationContractError:
        raise
    except OSError as error:
        raise SimulationContractError(
            f"SymSim native output cannot be opened safely: {path.name}"
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
                f"SymSim native output changed while opening: {path.name}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
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
                f"SymSim native output changed while reading: {path.name}"
            )
        return b"".join(chunks)
    except OSError as error:
        raise SimulationContractError(
            f"SymSim native output changed while reading: {path.name}"
        ) from error
    finally:
        os.close(descriptor)


def _read_tsv(path: Path) -> list[list[str]]:
    data = _read_regular_bytes(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SimulationContractError(f"{path.name} is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text or "\x00" in text:
        raise SimulationContractError(f"{path.name} is not canonical TSV")
    try:
        rows = list(csv.reader(text.splitlines(), delimiter="\t", strict=True))
    except csv.Error as error:
        raise SimulationContractError(f"{path.name} is malformed TSV") from error
    if not rows or any(not row for row in rows):
        raise SimulationContractError(f"{path.name} is empty or malformed")
    return rows


def _read_count_matrix(
    path: Path, gene_ids: list[str], cell_ids: list[str]
) -> np.ndarray:
    rows = _read_tsv(path)
    if rows[0] != ["gene_id", *cell_ids] or len(rows) != len(gene_ids) + 1:
        raise SimulationContractError(f"{path.name} has wrong orientation or IDs")
    matrix = np.empty((len(gene_ids), len(cell_ids)), dtype=np.int64)
    maximum = np.iinfo(np.int64).max
    for row_index, (expected_gene, row) in enumerate(
        zip(gene_ids, rows[1:], strict=True)
    ):
        if len(row) != len(cell_ids) + 1 or row[0] != expected_gene:
            raise SimulationContractError(f"{path.name} has wrong shape or gene IDs")
        for column_index, value in enumerate(row[1:]):
            if not _INTEGER.fullmatch(value):
                raise SimulationContractError(
                    f"{path.name} must contain nonnegative integer counts"
                )
            integer = int(value)
            if integer > maximum:
                raise SimulationContractError(f"{path.name} count exceeds int64")
            matrix[row_index, column_index] = integer
    return matrix


def _read_groups(path: Path, cell_ids: list[str], rare_count: int) -> np.ndarray:
    rows = _read_tsv(path)
    if rows[0] != ["cell_id", "group"] or len(rows) != len(cell_ids) + 1:
        raise SimulationContractError("cell_metadata.tsv has wrong shape")
    groups = np.empty(len(cell_ids), dtype=np.int64)
    for index, (cell_id, row) in enumerate(zip(cell_ids, rows[1:], strict=True)):
        if (
            len(row) != 2
            or row[0] != cell_id
            or row[1] not in {"1", "2", "3", "4", "5"}
        ):
            raise SimulationContractError(
                "cell_metadata.tsv has invalid cell IDs or groups"
            )
        groups[index] = int(row[1])
    counts = {group: int((groups == group).sum()) for group in range(1, 6)}
    if counts[1] != rare_count or any(counts[group] <= 0 for group in range(1, 6)):
        raise SimulationContractError(
            "SymSim must contain five groups with population 1 exactly 5%"
        )
    return groups


def _read_marker_truth(
    path: Path, gene_ids: list[str], threshold: float
) -> pd.DataFrame:
    expected_header = ["gene_id"]
    for group in range(1, 6):
        expected_header.extend(
            [f"theoretical_log2fc_group_{group}", f"marker_group_{group}"]
        )
    rows = _read_tsv(path)
    if rows[0] != expected_header or len(rows) != len(gene_ids) + 1:
        raise SimulationContractError("marker_truth.tsv has wrong schema or shape")
    values: dict[str, list[float | bool]] = {
        column: [] for column in expected_header[1:]
    }
    for expected_gene, row in zip(gene_ids, rows[1:], strict=True):
        if len(row) != len(expected_header) or row[0] != expected_gene:
            raise SimulationContractError("marker_truth.tsv has invalid gene IDs")
        for group in range(1, 6):
            score_text = row[1 + 2 * (group - 1)]
            marker_text = row[2 + 2 * (group - 1)]
            try:
                score = float(score_text)
            except ValueError as error:
                raise SimulationContractError(
                    "marker_truth.tsv has a nonnumeric theoretical score"
                ) from error
            if not math.isfinite(score) or marker_text not in {"0", "1"}:
                raise SimulationContractError(
                    "marker_truth.tsv has invalid marker truth"
                )
            marker = marker_text == "1"
            if marker != (score > threshold):
                raise SimulationContractError(
                    "marker_truth.tsv marker flag contradicts its theoretical score"
                )
            values[f"theoretical_log2fc_group_{group}"].append(score)
            values[f"marker_group_{group}"].append(marker)
    return pd.DataFrame(values, index=gene_ids)


def _validate_stage_entries(stage: Path) -> dict[str, Path]:
    try:
        entries = list(os.scandir(stage))
    except OSError as error:
        raise SimulationContractError(
            "SymSim native output directory is unavailable"
        ) from error
    names = {entry.name for entry in entries}
    if names != _EXPECTED_NATIVE_FILES or len(entries) != len(_EXPECTED_NATIVE_FILES):
        raise SimulationContractError(
            "SymSim native outputs do not match the closed file set"
        )
    result: dict[str, Path] = {}
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise SimulationContractError(
                "SymSim native output cannot be inspected"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SimulationContractError(
                "SymSim native outputs must be unique regular files without symlinks"
            )
        result[entry.name] = Path(entry.path)
    return result


def _validate_run_metadata(
    path: Path, config: Mapping[str, object]
) -> dict[str, object]:
    value = _load_json_bytes(_read_regular_bytes(path), path.name)
    if not isinstance(value, dict):
        raise SimulationContractError("run_metadata.json must be an object")
    expected_keys = {
        "schema_version",
        "simulate_true_counts_calls",
        "true2observed_counts_calls",
        "cells",
        "genes",
        "views",
        "biological_seed_r",
        "measurement_seeds_r",
        "r_version",
        "symsim_version",
    }
    simulation = config["simulation"]
    seeds = config["seeds"]
    views = config["views"]
    assert isinstance(simulation, Mapping)
    assert isinstance(seeds, Mapping)
    assert isinstance(views, list)
    biological = seeds["biological"]
    assert isinstance(biological, Mapping)
    expected_measurement = {
        view["technical_view"]: view["measurement_seed_r"]
        for view in views
        if isinstance(view, Mapping)
    }
    expected_binding = {
        "schema_version": 1,
        "simulate_true_counts_calls": 1,
        "true2observed_counts_calls": 2,
        "cells": simulation["cells"],
        "genes": simulation["genes"],
        "views": [view["technical_view"] for view in views],
        "biological_seed_r": biological["mapped_r"],
        "measurement_seeds_r": expected_measurement,
    }
    observed_binding = {key: value.get(key) for key in expected_binding}
    if (
        set(value) != expected_keys
        or canonical_sha256(observed_binding) != canonical_sha256(expected_binding)
        or not isinstance(value.get("r_version"), str)
        or not value["r_version"]
        or not isinstance(value.get("symsim_version"), str)
        or not value["symsim_version"]
    ):
        raise SimulationContractError(
            "run_metadata.json does not bind the exact paired run"
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


def _publish_native_directory(
    stage: Path, files: Mapping[str, Path], parent: Path
) -> Path:
    descriptor = _native_descriptor(files)
    content_id = canonical_sha256(
        {"schema": "maskimpute-symsim-native-v1", "files": descriptor}
    )[:24]
    _reject_output_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_output_symlink_components(parent)
    native_root = parent / "native"
    if native_root.is_symlink() or (native_root.exists() and not native_root.is_dir()):
        raise SimulationContractError("SymSim native output root is invalid")
    native_root.mkdir(mode=0o755, exist_ok=True)
    destination = native_root / f"symsim-{content_id}"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise SimulationContractError("existing SymSim native directory is invalid")
        existing = _validate_stage_entries(destination)
        if _native_descriptor(existing) != descriptor:
            raise SimulationContractError("existing SymSim native bytes changed")
        shutil.rmtree(stage)
        return destination
    publication = Path(tempfile.mkdtemp(prefix=".symsim-publish-", dir=native_root))
    try:
        for name in sorted(files):
            data = _read_regular_bytes(files[name])
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
                "SymSim native bytes changed while publishing"
            )
        os.rename(publication, destination)
        descriptor_fd = os.open(
            native_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor_fd)
        finally:
            os.close(descriptor_fd)
    except OSError as error:
        raise SimulationContractError(
            "SymSim native outputs could not be published"
        ) from error
    finally:
        if publication.exists():
            shutil.rmtree(publication)
    shutil.rmtree(stage)
    return destination


def _reject_output_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    for component in [absolute, *absolute.parents]:
        if not os.path.lexists(component):
            continue
        try:
            metadata = component.lstat()
        except OSError as error:
            raise SimulationContractError(
                "SymSim output path cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SimulationContractError(
                "SymSim output path must not contain symlinks"
            )


def _h5ad_staging_root(destination: Path) -> Path:
    absolute = destination.absolute()
    current = absolute.parent
    while not os.path.lexists(current):
        if current == current.parent:
            raise SimulationContractError("SymSim output has no existing ancestor")
        current = current.parent
    try:
        current_metadata = current.lstat()
    except OSError as error:
        raise SimulationContractError(
            "SymSim output ancestor cannot be inspected"
        ) from error
    if not stat.S_ISDIR(current_metadata.st_mode) or stat.S_ISLNK(
        current_metadata.st_mode
    ):
        raise SimulationContractError(
            "SymSim output ancestor must be a non-symlink directory"
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
            "SymSim h5ad staging root cannot be inspected"
        ) from error
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or root_metadata.st_dev != current_metadata.st_dev
    ):
        raise SimulationContractError(
            "SymSim h5ad staging root must be a same-device non-symlink directory"
        )
    return root


def _stage_h5ad(adata: ad.AnnData, destination: Path) -> tuple[Path, ad.AnnData]:
    _reject_output_symlink_components(destination)
    staging_root = _h5ad_staging_root(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="maskimpute-symsim-h5ad-", suffix=".h5ad", dir=staging_root
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
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise SimulationContractError(
            f"SymSim dataset could not be serialized for atomic persistence: {destination}"
        ) from error


def _publish_staged_h5ad(temporary: Path, destination: Path) -> ad.AnnData:
    _reject_output_symlink_components(destination)
    try:
        os.replace(temporary, destination)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return ad.read_h5ad(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SimulationContractError(
            f"SymSim dataset could not be atomically persisted: {destination}"
        ) from error


def _build_dataset(
    request: SimulationRequest,
    observed_gene_cell: np.ndarray,
    true_gene_cell: np.ndarray,
    groups: np.ndarray,
    marker_truth: pd.DataFrame,
    native_manifest_sha256: str,
    source_receipt: Mapping[str, object],
    environment: Mapping[str, object],
    config: Mapping[str, object],
    run_metadata: Mapping[str, object],
) -> ad.AnnData:
    cell_ids = _expected_ids("cell", request.cells)
    observed = observed_gene_cell.T.copy()
    truth = true_gene_cell.T.copy()
    draw = int(request.biological_id.removeprefix("draw-"))
    library_sizes = [sum(int(value) for value in row) for row in observed]
    if any(total > np.iinfo(np.int64).max for total in library_sizes):
        raise SimulationContractError("SymSim observed library size exceeds int64")
    obs = pd.DataFrame(
        {
            "dataset_id": [request.dataset_id] * request.cells,
            "mechanism": [request.mechanism] * request.cells,
            "condition": [request.technical_view] * request.cells,
            "biological_id": [request.biological_id] * request.cells,
            "technical_view": [request.technical_view] * request.cells,
            "draw": np.full(request.cells, draw, dtype=np.int64),
            "library_size": np.asarray(library_sizes, dtype=np.int64),
            "group": [f"pop-{int(group)}" for group in groups],
        },
        index=cell_ids,
    )
    view_config = next(
        view
        for view in config["views"]
        if isinstance(view, Mapping)
        and view.get("technical_view") == request.technical_view
    )
    biological_config = config["seeds"]["biological"]
    adata = ad.AnnData(
        X=observed,
        obs=obs,
        var=marker_truth.copy(deep=True),
        layers={"pre_capture_counts": truth},
    )
    adata.uns.update(
        {
            "truth_kind": "exact_pre_capture",
            "primary_truth_layer": "pre_capture_counts",
            "provenance": {
                "source": source_receipt["source_url"],
                "source_sha256": canonical_sha256(source_receipt),
                "software": "SymSim",
                "software_version": source_receipt["resolved_revision"],
                "parameters": {
                    "adapter": config["adapter"],
                    "adapter_schema": "maskimpute-symsim-adapter-v1",
                    "environment": environment,
                    "marker_definition": {
                        "estimand": "theoretical_mean_group_vs_all_other_cells",
                        "log2fc_threshold": config["simulation"][
                            "marker_log2fc_threshold"
                        ],
                    },
                    "measurement": {
                        key: value
                        for key, value in view_config.items()
                        if key
                        not in {"measurement_seed_original", "measurement_seed_r"}
                    },
                    "native_manifest_sha256": native_manifest_sha256,
                    "native_runtime": {
                        "r_version": run_metadata["r_version"],
                        "symsim_version": run_metadata["symsim_version"],
                    },
                    "simulation": config["simulation"],
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
    return adata


def _pair_config(requests: Mapping[str, SimulationRequest]) -> dict[str, object]:
    moderate = requests["moderate"]
    severe = requests["severe"]
    mapped = map_symsim_r_seeds(
        moderate.biological_seed,
        moderate.measurement_seed,
        severe.measurement_seed,
    )
    simulation = {
        "cells": moderate.cells,
        "genes": moderate.genes,
        "gene_length": 1000,
        "gene_module_prop": 0,
        "i_minpop": 1,
        "marker_log2fc_threshold": 1.0,
        "min_popsize": moderate.cells // 20,
        "n_de_evf": 9,
        "nevf": 10,
        "prop_hge": 0,
        "vary": "s",
    }
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
    return {
        "adapter": {
            "python_adapter_sha256": file_sha256(Path(__file__)),
            "r_runner_sha256": file_sha256(_RUNNER),
        },
        "schema_version": 1,
        "simulation": simulation,
        "seeds": {
            "biological": {
                "original": moderate.biological_seed,
                "mapped_r": mapped["biological"],
            }
        },
        "views": views,
    }


def _revalidate_published_final_claim(claim: FinalManifestClaim | None) -> None:
    """Recheck lifecycle records after unreceipted result publication."""

    from .. import study

    if not isinstance(claim, FinalManifestClaim):
        raise SimulationContractError(
            "published final SymSim pair requires its original execution claim"
        )
    try:
        repository = claim._repository
        destination = claim.round_dir
        canonical_repository, canonical_destination = study._repository_for_round(
            destination, repository
        )
        if canonical_repository != repository or canonical_destination != destination:
            raise SimulationContractError(
                "final SymSim claim changed repository identity"
            )
        with study._round_lock(repository, destination.name) as lock_identity:
            freeze = study._validate_freeze(destination, repository)
            if freeze.get("protocol_sha256") != claim._protocol_sha256:
                raise SimulationContractError(
                    "final SymSim claim changed frozen protocol"
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
                    "final SymSim execution claim changed during publication"
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
            f"final SymSim execution is no longer the claimed running round: {error}"
        ) from error


def run_symsim_pair(
    requests: Sequence[SimulationRequest],
    protocol: Protocol,
    final_manifest: FinalManifestClaim | None = None,
) -> tuple[SimulationArtifact, SimulationArtifact]:
    """Generate paired moderate/severe views from one exact SymSim truth draw."""

    try:
        ordered_requests = tuple(requests)
    except TypeError as error:
        raise SimulationContractError(
            "SymSim requests must be a finite pair"
        ) from error
    validate_paired_simulation_requests(ordered_requests, protocol, final_manifest)
    if any(request.mechanism != "symsim" for request in ordered_requests):
        raise SimulationContractError("SymSim adapter accepts only symsim requests")
    by_view = {request.technical_view: request for request in ordered_requests}
    if set(by_view) != {"moderate", "severe"} or len(by_view) != 2:
        raise SimulationContractError(
            "SymSim adapter requires exactly moderate and severe technical views"
        )
    moderate = by_view["moderate"]
    if moderate.cells < 20 or moderate.cells % 20 != 0:
        raise SimulationContractError(
            "SymSim cell count must make the 5% rare population an exact integer"
        )
    if moderate.genes < 20:
        raise SimulationContractError("SymSim requires at least 20 genes")
    parents = {request.output_path.parent.absolute() for request in ordered_requests}
    if len(parents) != 1:
        raise SimulationContractError("paired SymSim outputs must share one directory")
    output_parent = next(iter(parents))
    _reject_output_symlink_components(output_parent)

    config = _pair_config(by_view)
    before_source = _verify_symsim_source()
    before_environment = _environment_receipt()
    stage = Path(tempfile.mkdtemp(prefix="maskimpute-symsim-native-"))
    published = False
    runner_error: Exception | None = None
    try:
        config_path = stage / "config.json"
        config_path.write_bytes(_canonical_json_bytes(config))
        try:
            _execute_symsim(
                config_path,
                stage,
                timeout_seconds=protocol.final_timeout_seconds,
            )
        except Exception as error:  # source/environment must still be rechecked
            runner_error = error
        try:
            after_source = _verify_symsim_source()
            after_environment = _environment_receipt()
        except Exception as error:
            raise SimulationContractError(
                "SymSim source or environment was not pristine after execution"
            ) from error
        if canonical_sha256(before_source) != canonical_sha256(after_source):
            raise SimulationContractError(
                "SymSim source receipt changed during execution"
            )
        if canonical_sha256(before_environment) != canonical_sha256(after_environment):
            raise SimulationContractError("SymSim environment changed during execution")
        if runner_error is not None:
            if isinstance(runner_error, SimulationContractError):
                raise runner_error
            raise SimulationContractError(
                "SymSim native runner failed"
            ) from runner_error

        files = _validate_stage_entries(stage)
        observed_config = _load_json_bytes(
            _read_regular_bytes(files["config.json"]), "config.json"
        )
        if canonical_sha256(observed_config) != canonical_sha256(config):
            raise SimulationContractError(
                "SymSim native runner changed its sealed config"
            )
        gene_ids = _expected_ids("gene", moderate.genes)
        cell_ids = _expected_ids("cell", moderate.cells)
        true_counts = _read_count_matrix(files["true_counts.tsv"], gene_ids, cell_ids)
        observed = {
            view: _read_count_matrix(files[f"observed_{view}.tsv"], gene_ids, cell_ids)
            for view in ("moderate", "severe")
        }
        if any(bool((matrix > true_counts).any()) for matrix in observed.values()):
            raise SimulationContractError(
                "SymSim UMI counts cannot exceed true molecules"
            )
        groups = _read_groups(
            files["cell_metadata.tsv"], cell_ids, moderate.cells // 20
        )
        marker_truth = _read_marker_truth(
            files["marker_truth.tsv"],
            gene_ids,
            float(config["simulation"]["marker_log2fc_threshold"]),
        )
        run_metadata = _validate_run_metadata(files["run_metadata.json"], config)
        pair_identity = {
            name: simulation_scientific_identity(by_view[name])
            for name in ("moderate", "severe")
        }
        manifest_metadata: dict[str, dict[str, object]] = {}
        staging_manifests = {}
        for name in ("moderate", "severe"):
            request = by_view[name]
            metadata = {
                "adapter": config["adapter"],
                "adapter_schema": "maskimpute-symsim-native-v1",
                "environment": before_environment,
                "pair_request_sha256": canonical_sha256(pair_identity),
                "run_metadata": run_metadata,
                "simulation_request": simulation_scientific_identity(request),
                "source_receipt": before_source,
            }
            manifest_metadata[name] = metadata
            staging_manifests[name] = seal_native_outputs(files, metadata)

        staged_datasets: dict[str, tuple[Path, ad.AnnData]] = {}
        try:
            for name in ("moderate", "severe"):
                request = by_view[name]
                manifest = staging_manifests[name]
                dataset = _build_dataset(
                    request,
                    observed[name],
                    true_counts,
                    groups,
                    marker_truth,
                    manifest.manifest_sha256,
                    before_source,
                    before_environment,
                    config,
                    run_metadata,
                )
                staged_datasets[name] = _stage_h5ad(dataset, request.output_path)
                _temporary, staged_semantics = staged_datasets[name]
                staged_sha256 = benchmark_dataset_sha256(staged_semantics)
                SimulationArtifact(request, staged_semantics, manifest, staged_sha256)

            # R execution and both serialization round trips are complete, but
            # no result path exists yet.  This is the terminal authoritative
            # check before the atomic publication sequence.
            validate_paired_simulation_requests(
                ordered_requests, protocol, final_manifest
            )
            native_directory = _publish_native_directory(stage, files, output_parent)
            published = True
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
                        "published SymSim native manifest differs from staged bytes"
                    )

            artifacts: dict[str, SimulationArtifact] = {}
            for name in ("moderate", "severe"):
                request = by_view[name]
                temporary, _staged_semantics = staged_datasets[name]
                persisted = _publish_staged_h5ad(temporary, request.output_path)
                dataset_sha256 = benchmark_dataset_sha256(persisted)
                artifacts[name] = SimulationArtifact(
                    request, persisted, manifests[name], dataset_sha256
                )
        finally:
            for temporary, _dataset in staged_datasets.values():
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        first = artifacts[ordered_requests[0].technical_view]
        second = artifacts[ordered_requests[1].technical_view]
        if not np.array_equal(
            first.adata.layers["pre_capture_counts"],
            second.adata.layers["pre_capture_counts"],
        ):
            raise SimulationContractError(
                "paired SymSim artifacts do not preserve identical truth"
            )
        if ordered_requests[0].namespace == protocol.final.namespace:
            _revalidate_published_final_claim(final_manifest)
        else:
            validate_paired_simulation_requests(
                ordered_requests, protocol, final_manifest
            )
        return first, second
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


__all__ = ["map_symsim_r_seeds", "run_symsim_pair"]
