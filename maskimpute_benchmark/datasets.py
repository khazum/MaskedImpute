"""Deterministic, fail-closed orchestration of the publication dataset panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from types import MappingProxyType
from typing import Any, Callable

import anndata as ad
import numpy as np
from scipy import sparse

from .protocol import Protocol, canonical_sha256, file_sha256, load_protocol
from .schema import benchmark_dataset_sha256, validate_benchmark_dataset
from .study import record_incremental_results
from .simulators import (
    FinalManifestClaim,
    SimulationArtifact,
    SimulationRequest,
    load_final_manifest_claim,
    run_semisynthetic_pair,
    run_sergio_pair,
    run_sparsim_pair,
    run_symsim_pair,
)


_MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
_VIEWS = ("moderate", "severe")
_EXPECTED_FINAL_SEEDS = len(_MECHANISMS) * 5 * (1 + len(_VIEWS))
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FAILED_REASON = re.compile(r"^adapter_failed:[A-Za-z_][A-Za-z0-9_]*$")
_ROW_KEYS = frozenset(
    {
        "biological_id",
        "biological_seed_commitment",
        "cells",
        "dataset_id",
        "dataset_sha256",
        "genes",
        "independent_unit_id",
        "log_path",
        "log_sha256",
        "measurement_seed_commitment",
        "mechanism",
        "native_files",
        "native_manifest_sha256",
        "output_file_sha256",
        "output_path",
        "reason",
        "source_sha256",
        "status",
        "technical_view",
        "truth_sha256",
    }
)
_STATUS_KEYS = frozenset(
    {
        "completed_count",
        "design_sha256",
        "execution_claim_id",
        "failed_count",
        "independent_unit_count",
        "manifest_sha256",
        "namespace",
        "protocol_sha256",
        "round_id",
        "rows",
        "schema_version",
        "seed_source_sha256",
        "status",
    }
)

Adapter = Callable[
    [Sequence[SimulationRequest], Protocol, FinalManifestClaim | None],
    tuple[SimulationArtifact, SimulationArtifact],
]

_ADAPTERS: Mapping[str, Adapter] = MappingProxyType(
    {
        "symsim": run_symsim_pair,
        "sergio": run_sergio_pair,
        "sparsim": run_sparsim_pair,
        "semisynthetic": run_semisynthetic_pair,
    }
)


class DatasetRegistryError(RuntimeError):
    """Raised when a panel design, adapter result, or receipt fails closed."""


@dataclass(frozen=True, slots=True)
class DevelopmentPanel:
    """Validated development-only design and its tracked seed derivation."""

    schema_version: int
    role: str
    namespace: str
    mechanisms: tuple[str, ...]
    technical_views: tuple[str, ...]
    draws_per_mechanism: int
    cells: int
    genes: int
    seed_algorithm: str
    master_seed: int
    file_sha256: str


@dataclass(frozen=True, slots=True)
class _PairDesign:
    mechanism: str
    biological_id: str
    requests: tuple[SimulationRequest, SimulationRequest]
    biological_seed_commitment: str
    measurement_seed_commitments: tuple[str, str]


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(dict(payload), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DatasetRegistryError(
            f"record is not finite canonical JSON: {error}"
        ) from error


def _read_canonical_json(path: Path, name: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_mode & 0o002
        ):
            raise DatasetRegistryError(f"{name} is not a secure unique regular file")
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except DatasetRegistryError:
        raise
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetRegistryError(f"invalid {name}: {error}") from error
    if not isinstance(payload, dict):
        raise DatasetRegistryError(f"{name} must be a JSON object")
    if raw != _canonical_json_bytes(payload):
        raise DatasetRegistryError(f"{name} is not canonical JSON")
    return payload


def _reject_symlink_components(path: Path, name: str) -> None:
    for component in (path.absolute(), *path.absolute().parents):
        if not os.path.lexists(component):
            continue
        try:
            metadata = component.lstat()
        except OSError as error:
            raise DatasetRegistryError(f"{name} path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise DatasetRegistryError(f"{name} path must not contain symlinks")


def _publish_new_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical_json_bytes(payload)
    _reject_symlink_components(path.parent, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, path.name)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _read_canonical_json(path, path.name)
            if _canonical_json_bytes(existing) != encoded:
                raise DatasetRegistryError(f"refusing to replace existing {path.name}")
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_development_panel(path: Path, protocol: Protocol) -> DevelopmentPanel:
    """Load the exact tracked development-only design."""

    if not isinstance(path, Path):
        raise TypeError("development panel path must be a pathlib.Path")
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    payload = _read_canonical_json(path, "development panel")
    expected_keys = {
        "schema_version",
        "role",
        "namespace",
        "mechanisms",
        "technical_views",
        "draws_per_mechanism",
        "cells",
        "genes",
        "seed_derivation",
    }
    if set(payload) != expected_keys:
        raise DatasetRegistryError("development panel has missing or extra fields")
    derivation = payload.get("seed_derivation")
    if not isinstance(derivation, dict) or set(derivation) != {
        "algorithm",
        "master_seed",
    }:
        raise DatasetRegistryError("development seed derivation has wrong schema")
    if payload.get("schema_version") != 1:
        raise DatasetRegistryError("development panel schema_version must be 1")
    if payload.get("role") != "development_only":
        raise DatasetRegistryError("development panel role must be development_only")
    if payload.get("namespace") != protocol.development.namespace:
        raise DatasetRegistryError("development panel namespace mismatches protocol")
    if tuple(payload.get("mechanisms", ())) != _MECHANISMS:
        raise DatasetRegistryError("development mechanisms must be the closed panel")
    if tuple(payload.get("technical_views", ())) != _VIEWS:
        raise DatasetRegistryError("technical views must be moderate and severe")
    if payload.get("draws_per_mechanism") != 2 or (
        protocol.development.draws_per_condition != 2
    ):
        raise DatasetRegistryError("development panel requires exactly two draws")
    if (
        payload.get("cells") != protocol.development.cells
        or payload.get("genes") != protocol.development.genes
    ):
        raise DatasetRegistryError("development dimensions mismatch protocol")
    if derivation.get("algorithm") != "sha256-domain-separated-63bit-v1":
        raise DatasetRegistryError("unsupported development seed derivation")
    master_seed = derivation.get("master_seed")
    if type(master_seed) is not int or not 0 <= master_seed < 2**63:
        raise DatasetRegistryError("development master seed must be a 63-bit integer")
    return DevelopmentPanel(
        schema_version=1,
        role="development_only",
        namespace=protocol.development.namespace,
        mechanisms=_MECHANISMS,
        technical_views=_VIEWS,
        draws_per_mechanism=2,
        cells=protocol.development.cells,
        genes=protocol.development.genes,
        seed_algorithm="sha256-domain-separated-63bit-v1",
        master_seed=master_seed,
        file_sha256=file_sha256(path),
    )


def _derived_development_seed(
    panel: DevelopmentPanel,
    *,
    mechanism: str,
    biological_id: str,
    role: str,
    technical_view: str | None,
) -> int:
    payload = {
        "schema": "maskimpute-development-seed-v1",
        "namespace": panel.namespace,
        "development_role": panel.role,
        "master_seed": panel.master_seed,
        "mechanism": mechanism,
        "biological_id": biological_id,
        "seed_role": role,
        "technical_view": technical_view,
    }
    return int(canonical_sha256(payload)[:16], 16) & (2**63 - 1)


def _seed_commitment(
    seed: int,
    *,
    namespace: str,
    mechanism: str,
    biological_id: str,
    role: str,
    technical_view: str | None,
    seed_source_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "maskimpute-seed-commitment-v1",
            "namespace": namespace,
            "mechanism": mechanism,
            "biological_id": biological_id,
            "seed_role": role,
            "technical_view": technical_view,
            "seed_source_sha256": seed_source_sha256,
            "seed": seed,
        }
    )


def _development_seed_set(panel: DevelopmentPanel) -> set[int]:
    seeds: list[int] = []
    for mechanism in panel.mechanisms:
        for draw in range(1, panel.draws_per_mechanism + 1):
            biological_id = f"draw-{draw:02d}"
            seeds.append(
                _derived_development_seed(
                    panel,
                    mechanism=mechanism,
                    biological_id=biological_id,
                    role="biological",
                    technical_view=None,
                )
            )
            for view in panel.technical_views:
                seeds.append(
                    _derived_development_seed(
                        panel,
                        mechanism=mechanism,
                        biological_id=biological_id,
                        role="measurement",
                        technical_view=view,
                    )
                )
    if len(seeds) != 24 or len(set(seeds)) != len(seeds):
        raise DatasetRegistryError("development seed derivation contains a collision")
    return set(seeds)


def _safe_relative_path(value: str, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DatasetRegistryError(f"{name} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DatasetRegistryError(f"{name} must be a canonical relative POSIX path")
    return path


def _ensure_beneath(root: Path, path: Path, name: str) -> Path:
    root_absolute = root.absolute()
    path_absolute = path.absolute()
    try:
        relative = path_absolute.relative_to(root_absolute)
    except ValueError as error:
        raise DatasetRegistryError(
            f"{name} must be beneath the results root"
        ) from error
    component = root_absolute
    components = [component]
    for part in relative.parts:
        component = component / part
        components.append(component)
    for index, component in enumerate(components):
        if not os.path.lexists(component):
            continue
        try:
            metadata = component.lstat()
        except OSError as error:
            raise DatasetRegistryError(f"{name} path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise DatasetRegistryError(f"{name} path must not contain symlinks")
        if index < len(components) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise DatasetRegistryError(f"{name} parent must be a directory")
    return path_absolute


def _file_sha256_secure(
    path: Path, root: Path, name: str
) -> tuple[str, tuple[int, int]]:
    _ensure_beneath(root, path, name)
    try:
        before = path.lstat()
    except OSError as error:
        raise DatasetRegistryError(f"{name} does not exist") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink != 1
    ):
        raise DatasetRegistryError(f"{name} must be a unique regular file")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise DatasetRegistryError(f"{name} changed while opening")
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            named_after = path.lstat()
        finally:
            os.close(descriptor)
    except DatasetRegistryError:
        raise
    except OSError as error:
        raise DatasetRegistryError(f"{name} could not be hashed") from error
    if (
        (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or (named_after.st_dev, named_after.st_ino)
        != (before.st_dev, before.st_ino)
        or named_after.st_mode != before.st_mode
        or named_after.st_nlink != before.st_nlink
        or named_after.st_uid != before.st_uid
        or after.st_mode != before.st_mode
        or after.st_nlink != before.st_nlink
        or after.st_uid != before.st_uid
        or after.st_size != before.st_size
        or named_after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or named_after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or named_after.st_ctime_ns != before.st_ctime_ns
    ):
        raise DatasetRegistryError(f"{name} changed while hashing")
    _ensure_beneath(root, path, name)
    return digest.hexdigest(), (before.st_dev, before.st_ino)


def _matrix_sha256(matrix: object, *, discrete: bool) -> str:
    shape = getattr(matrix, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise DatasetRegistryError("truth matrix is not two-dimensional")
    digest = hashlib.sha256()
    digest.update(np.asarray(shape, dtype="<u8").tobytes())
    dtype = np.dtype("<u8") if discrete else np.dtype("<f8")
    digest.update(b"discrete-uint64" if discrete else b"continuous-float64")
    if sparse.issparse(matrix):
        canonical = matrix.tocsr(copy=True)
        canonical.sum_duplicates()
        canonical.sort_indices()
        canonical.eliminate_zeros()
        for row in range(shape[0]):
            start, stop = canonical.indptr[row : row + 2]
            columns = np.asarray(canonical.indices[start:stop], dtype="<u8")
            values = np.asarray(canonical.data[start:stop], dtype=dtype)
            digest.update(np.asarray([len(columns)], dtype="<u8").tobytes())
            digest.update(columns.tobytes())
            digest.update(values.tobytes())
    else:
        dense = np.asarray(matrix)
        for row in range(shape[0]):
            values = np.asarray(dense[row]).reshape(-1)
            columns = np.flatnonzero(values != 0).astype("<u8", copy=False)
            digest.update(np.asarray([len(columns)], dtype="<u8").tobytes())
            digest.update(columns.tobytes())
            digest.update(np.asarray(values[columns], dtype=dtype).tobytes())
    return digest.hexdigest()


def _truth_sha256(dataset: ad.AnnData) -> str:
    validate_benchmark_dataset(dataset)
    layer_name = dataset.uns["primary_truth_layer"]
    if not isinstance(layer_name, str) or layer_name not in dataset.layers:
        raise DatasetRegistryError("dataset primary truth layer is invalid")
    discrete = layer_name in {
        "pre_capture_counts",
        "reference_counts",
        "heldout_counts",
    }
    return canonical_sha256(
        {
            "schema": "maskimpute-primary-truth-v1",
            "truth_kind": dataset.uns["truth_kind"],
            "primary_truth_layer": layer_name,
            "shape": list(dataset.shape),
            "obs_ids": [str(value) for value in dataset.obs_names],
            "var_ids": [str(value) for value in dataset.var_names],
            "matrix_sha256": _matrix_sha256(
                dataset.layers[layer_name], discrete=discrete
            ),
        }
    )


def _designs(
    *,
    namespace: str,
    protocol: Protocol,
    panel: DevelopmentPanel,
    results_root: Path,
    claim: FinalManifestClaim | None,
) -> tuple[_PairDesign, ...]:
    if namespace == protocol.development.namespace:
        draws = panel.draws_per_mechanism
        cells, genes = panel.cells, panel.genes
        seed_source = panel.file_sha256
        final_seeds: tuple[int, ...] | None = None
    elif namespace == protocol.final.namespace:
        if claim is None:
            raise DatasetRegistryError(
                "final namespace requires a running execution claim"
            )
        draws = protocol.final.draws_per_condition
        if draws != 5:
            raise DatasetRegistryError("final panel requires exactly five draws")
        cells, genes = protocol.final.cells, protocol.final.genes
        seed_source = claim.seed_manifest_sha256
        final_seeds = claim.generator_seeds
        if len(final_seeds) != _EXPECTED_FINAL_SEEDS:
            raise DatasetRegistryError(
                "final seed manifest must contain exactly 60 seeds"
            )
        if len(set(final_seeds)) != len(final_seeds):
            raise DatasetRegistryError("final seed manifest contains reused seeds")
        if set(final_seeds) & _development_seed_set(panel):
            raise DatasetRegistryError(
                "final seed manifest collides with development seeds"
            )
    else:
        raise DatasetRegistryError("namespace must be exactly dev or final")

    designs: list[_PairDesign] = []
    cursor = 0
    used: list[int] = []
    for mechanism in _MECHANISMS:
        for draw in range(1, draws + 1):
            biological_id = f"draw-{draw:02d}"
            if final_seeds is None:
                biological_seed = _derived_development_seed(
                    panel,
                    mechanism=mechanism,
                    biological_id=biological_id,
                    role="biological",
                    technical_view=None,
                )
                measurement_seeds = tuple(
                    _derived_development_seed(
                        panel,
                        mechanism=mechanism,
                        biological_id=biological_id,
                        role="measurement",
                        technical_view=view,
                    )
                    for view in _VIEWS
                )
            else:
                biological_seed = final_seeds[cursor]
                measurement_seeds = (final_seeds[cursor + 1], final_seeds[cursor + 2])
                cursor += 3
            used.extend((biological_seed, *measurement_seeds))
            requests = tuple(
                SimulationRequest(
                    mechanism=mechanism,
                    namespace=namespace,
                    biological_id=biological_id,
                    biological_seed=biological_seed,
                    measurement_seed=measurement_seed,
                    technical_view=view,
                    cells=cells,
                    genes=genes,
                    output_path=(
                        results_root
                        / namespace
                        / "datasets"
                        / mechanism
                        / biological_id
                        / f"{view}.h5ad"
                    ),
                )
                for view, measurement_seed in zip(
                    _VIEWS, measurement_seeds, strict=True
                )
            )
            assert len(requests) == 2
            designs.append(
                _PairDesign(
                    mechanism=mechanism,
                    biological_id=biological_id,
                    requests=(requests[0], requests[1]),
                    biological_seed_commitment=_seed_commitment(
                        biological_seed,
                        namespace=namespace,
                        mechanism=mechanism,
                        biological_id=biological_id,
                        role="biological",
                        technical_view=None,
                        seed_source_sha256=seed_source,
                    ),
                    measurement_seed_commitments=tuple(
                        _seed_commitment(
                            measurement_seed,
                            namespace=namespace,
                            mechanism=mechanism,
                            biological_id=biological_id,
                            role="measurement",
                            technical_view=view,
                            seed_source_sha256=seed_source,
                        )
                        for view, measurement_seed in zip(
                            _VIEWS, measurement_seeds, strict=True
                        )
                    ),
                )
            )
    if len(used) != len(set(used)):
        raise DatasetRegistryError("panel seed allocation contains reuse or collision")
    if final_seeds is not None and (
        cursor != len(final_seeds) or tuple(used) != final_seeds
    ):
        raise DatasetRegistryError(
            "final panel did not consume the exact claimed seed order"
        )
    return tuple(designs)


def _design_sha256(
    namespace: str,
    protocol_sha256: str,
    panel: DevelopmentPanel,
    claim: FinalManifestClaim | None,
) -> str:
    return canonical_sha256(
        {
            "schema": "maskimpute-dataset-panel-design-v1",
            "namespace": namespace,
            "protocol_sha256": protocol_sha256,
            "development_panel_sha256": panel.file_sha256,
            "seed_source_sha256": (
                panel.file_sha256 if claim is None else claim.seed_manifest_sha256
            ),
            "execution_claim_id": None if claim is None else claim.execution_claim_id,
            "round_id": None if claim is None else claim.round_id,
        }
    )


def _relative(root: Path, path: Path, name: str) -> str:
    absolute = _ensure_beneath(root, path, name)
    return absolute.relative_to(root.absolute()).as_posix()


def _native_file_rows(
    artifact: SimulationArtifact, results_root: Path
) -> list[dict[str, object]]:
    manifest = artifact.native_manifest
    rows: list[dict[str, object]] = []
    sealed_files = getattr(manifest, "_sealed_files", ())
    if len(sealed_files) != len(manifest.files):
        raise DatasetRegistryError("native manifest lacks its sealed file inventory")
    for entry, sealed in zip(manifest.files, sealed_files, strict=True):
        physical = getattr(sealed, "physical_path", None)
        if not isinstance(physical, Path):
            raise DatasetRegistryError("native manifest physical path is invalid")
        observed, _identity = _file_sha256_secure(
            physical, results_root, "native output"
        )
        if observed != entry.sha256 or physical.stat().st_size != entry.size_bytes:
            raise DatasetRegistryError("native output checksum or size mismatch")
        rows.append(
            {
                "logical_path": entry.path,
                "path": _relative(results_root, physical, "native output"),
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
        )
    return rows


def _log_record(
    design: _PairDesign,
    namespace: str,
    *,
    status: str,
    rows: Sequence[Mapping[str, object]] | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "namespace": namespace,
        "mechanism": design.mechanism,
        "biological_id": design.biological_id,
        "status": status,
        "error_type": None if error is None else type(error).__name__,
        "message": None if error is None else str(error),
        "dataset_ids": [] if rows is None else [row["dataset_id"] for row in rows],
    }


def _log_path(results_root: Path, design: _PairDesign) -> Path:
    return results_root / "logs" / design.mechanism / f"{design.biological_id}.json"


def _failed_rows(
    design: _PairDesign,
    results_root: Path,
    log_relative: str,
    log_sha256: str,
    error: BaseException,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, request in enumerate(design.requests):
        output_hash: str | None = None
        if os.path.lexists(request.output_path):
            output_hash, _identity = _file_sha256_secure(
                request.output_path, results_root, "failed adapter output"
            )
        rows.append(
            {
                "biological_id": request.biological_id,
                "biological_seed_commitment": design.biological_seed_commitment,
                "cells": request.cells,
                "dataset_id": request.dataset_id,
                "dataset_sha256": None,
                "genes": request.genes,
                "independent_unit_id": request.independent_unit_id,
                "log_path": log_relative,
                "log_sha256": log_sha256,
                "measurement_seed_commitment": design.measurement_seed_commitments[
                    index
                ],
                "mechanism": request.mechanism,
                "native_files": [],
                "native_manifest_sha256": None,
                "output_file_sha256": output_hash,
                "output_path": _relative(
                    results_root, request.output_path, "dataset output"
                ),
                "reason": f"adapter_failed:{type(error).__name__}",
                "source_sha256": None,
                "status": "failed",
                "technical_view": request.technical_view,
                "truth_sha256": None,
            }
        )
    return rows


def _completed_rows(
    design: _PairDesign,
    artifacts_value: object,
    results_root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if (
        not isinstance(artifacts_value, Sequence)
        or isinstance(artifacts_value, (str, bytes))
        or len(artifacts_value) != 2
    ):
        raise DatasetRegistryError("adapter must return exactly two artifacts")
    artifacts = tuple(artifacts_value)
    if not all(isinstance(artifact, SimulationArtifact) for artifact in artifacts):
        raise DatasetRegistryError("adapter returned a noncanonical artifact")
    by_view = {artifact.request.technical_view: artifact for artifact in artifacts}
    if set(by_view) != set(_VIEWS) or len(by_view) != 2:
        raise DatasetRegistryError("adapter result has missing or extra views")
    rows: list[dict[str, object]] = []
    truth_hashes: list[str] = []
    for index, request in enumerate(design.requests):
        artifact = by_view[request.technical_view]
        if artifact.request != request:
            raise DatasetRegistryError(
                "adapter artifact request or provenance mismatch"
            )
        dataset = artifact.adata
        if dataset.shape != (request.cells, request.genes):
            raise DatasetRegistryError("adapter artifact dimension drift")
        observed_dataset_hash = benchmark_dataset_sha256(dataset)
        if observed_dataset_hash != artifact.dataset_sha256:
            raise DatasetRegistryError("adapter dataset checksum mismatch")
        truth_hash = _truth_sha256(dataset)
        truth_hashes.append(truth_hash)
        provenance = dataset.uns.get("provenance")
        if not isinstance(provenance, Mapping):
            raise DatasetRegistryError("adapter provenance is missing")
        parameters = provenance.get("parameters")
        seeds = provenance.get("seeds")
        manifest = artifact.native_manifest
        if (
            not isinstance(parameters, Mapping)
            or parameters.get("native_manifest_sha256") != manifest.manifest_sha256
            or not isinstance(seeds, Mapping)
            or seeds.get("biological") != request.biological_seed
            or seeds.get("measurement") != request.measurement_seed
        ):
            raise DatasetRegistryError(
                "adapter source, seed, or native provenance mismatch"
            )
        source_sha256 = provenance.get("source_sha256")
        if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
            raise DatasetRegistryError("adapter source provenance checksum is invalid")
        output_hash, _identity = _file_sha256_secure(
            request.output_path, results_root, "dataset output"
        )
        rows.append(
            {
                "biological_id": request.biological_id,
                "biological_seed_commitment": design.biological_seed_commitment,
                "cells": request.cells,
                "dataset_id": request.dataset_id,
                "dataset_sha256": artifact.dataset_sha256,
                "genes": request.genes,
                "independent_unit_id": request.independent_unit_id,
                "log_path": "",
                "log_sha256": "",
                "measurement_seed_commitment": design.measurement_seed_commitments[
                    index
                ],
                "mechanism": request.mechanism,
                "native_files": _native_file_rows(artifact, results_root),
                "native_manifest_sha256": manifest.manifest_sha256,
                "output_file_sha256": output_hash,
                "output_path": _relative(
                    results_root, request.output_path, "dataset output"
                ),
                "reason": "completed",
                "source_sha256": source_sha256,
                "status": "completed",
                "technical_view": request.technical_view,
                "truth_sha256": truth_hash,
            }
        )
    if len(set(truth_hashes)) != 1:
        raise DatasetRegistryError("paired technical views do not share exact truth")
    log = _log_record(
        design, design.requests[0].namespace, status="completed", rows=rows
    )
    return rows, log


def _receipt_path(results_root: Path, design: _PairDesign) -> Path:
    return results_root / "receipts" / design.mechanism / f"{design.biological_id}.json"


def _receipt_payload(
    design_sha256: str,
    design: _PairDesign,
    namespace: str,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "design_sha256": design_sha256,
        "namespace": namespace,
        "mechanism": design.mechanism,
        "biological_id": design.biological_id,
        "rows": [dict(row) for row in rows],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _execute_pair(
    *,
    design: _PairDesign,
    design_sha256: str,
    namespace: str,
    protocol: Protocol,
    claim: FinalManifestClaim | None,
    results_root: Path,
) -> list[dict[str, object]]:
    receipt_path = _receipt_path(results_root, design)
    if receipt_path.exists():
        receipt = _read_canonical_json(receipt_path, "dataset pair receipt")
        return _validate_receipt(
            receipt,
            design=design,
            design_sha256=design_sha256,
            namespace=namespace,
            results_root=results_root,
        )
    if any(os.path.lexists(request.output_path) for request in design.requests):
        raise DatasetRegistryError(
            "existing dataset output has no canonical receipt; refusing overwrite"
        )
    try:
        adapter = _ADAPTERS[design.mechanism]
    except (KeyError, TypeError) as error:
        raise DatasetRegistryError(
            "closed simulator adapter mapping is incomplete"
        ) from error
    try:
        artifacts = adapter(design.requests, protocol, claim)
        rows, log = _completed_rows(design, artifacts, results_root)
    except Exception as error:
        log = _log_record(design, namespace, status="failed", error=error)
        log_path = _log_path(results_root, design)
        _publish_new_json(log_path, log)
        log_hash = file_sha256(log_path)
        rows = _failed_rows(
            design,
            results_root,
            _relative(results_root, log_path, "adapter log"),
            log_hash,
            error,
        )
    else:
        log_path = _log_path(results_root, design)
        _publish_new_json(log_path, log)
        log_hash = file_sha256(log_path)
        log_relative = _relative(results_root, log_path, "adapter log")
        for row in rows:
            row["log_path"] = log_relative
            row["log_sha256"] = log_hash
    receipt = _receipt_payload(design_sha256, design, namespace, rows)
    _publish_new_json(receipt_path, receipt)
    return rows


def _validate_digest(value: object, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise DatasetRegistryError(f"{name} must be a lowercase SHA-256 digest")


def _expected_row_fields(
    design: _PairDesign, request: SimulationRequest, index: int
) -> dict[str, object]:
    return {
        "biological_id": request.biological_id,
        "biological_seed_commitment": design.biological_seed_commitment,
        "cells": request.cells,
        "dataset_id": request.dataset_id,
        "genes": request.genes,
        "independent_unit_id": request.independent_unit_id,
        "measurement_seed_commitment": design.measurement_seed_commitments[index],
        "mechanism": request.mechanism,
        "technical_view": request.technical_view,
    }


def _validate_native_rows(value: object, results_root: Path) -> None:
    if not isinstance(value, list) or not value:
        raise DatasetRegistryError("completed row native file inventory is empty")
    logical_paths: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "logical_path",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise DatasetRegistryError("native file inventory has wrong schema")
        logical = item["logical_path"]
        if not isinstance(logical, str) or not logical:
            raise DatasetRegistryError("native logical path is invalid")
        logical_paths.append(logical)
        relative = _safe_relative_path(item["path"], "native output path")
        _validate_digest(item["sha256"], "native output checksum")
        size = item["size_bytes"]
        if type(size) is not int or size < 0:
            raise DatasetRegistryError("native output size is invalid")
        path = results_root.joinpath(*relative.parts)
        observed, _identity = _file_sha256_secure(path, results_root, "native output")
        if observed != item["sha256"] or path.stat().st_size != size:
            raise DatasetRegistryError("native output checksum or size mismatch")
    if logical_paths != sorted(logical_paths) or len(logical_paths) != len(
        set(logical_paths)
    ):
        raise DatasetRegistryError("native logical paths must be sorted and unique")


def _validate_row(
    row: object,
    *,
    design: _PairDesign,
    request: SimulationRequest,
    index: int,
    results_root: Path,
) -> tuple[str | None, str | None, str | None, tuple[int, int] | None]:
    if not isinstance(row, dict) or set(row) != _ROW_KEYS:
        raise DatasetRegistryError("dataset status row is noncanonical")
    for key, expected in _expected_row_fields(design, request, index).items():
        if row.get(key) != expected:
            raise DatasetRegistryError(f"dataset status row {key} mismatches design")
    _validate_digest(row["biological_seed_commitment"], "biological seed commitment")
    _validate_digest(row["measurement_seed_commitment"], "measurement seed commitment")
    relative_output = _safe_relative_path(row["output_path"], "dataset output path")
    expected_relative = _relative(results_root, request.output_path, "dataset output")
    if relative_output.as_posix() != expected_relative:
        raise DatasetRegistryError("dataset output path mismatches the closed design")
    log_relative = _safe_relative_path(row["log_path"], "adapter log path")
    expected_log = _relative(
        results_root, _log_path(results_root, design), "adapter log"
    )
    if log_relative.as_posix() != expected_log:
        raise DatasetRegistryError("adapter log path mismatches design")
    _validate_digest(row["log_sha256"], "adapter log checksum")
    observed_log, _log_identity = _file_sha256_secure(
        results_root.joinpath(*log_relative.parts), results_root, "adapter log"
    )
    if observed_log != row["log_sha256"]:
        raise DatasetRegistryError("adapter log checksum mismatch")
    output_identity: tuple[int, int] | None = None
    if row["status"] == "completed":
        if row["reason"] != "completed":
            raise DatasetRegistryError("completed row reason is noncanonical")
        for field in (
            "dataset_sha256",
            "truth_sha256",
            "source_sha256",
            "native_manifest_sha256",
            "output_file_sha256",
        ):
            _validate_digest(row[field], field)
        _validate_native_rows(row["native_files"], results_root)
        output_path = results_root.joinpath(*relative_output.parts)
        observed_output, output_identity = _file_sha256_secure(
            output_path, results_root, "dataset output"
        )
        if observed_output != row["output_file_sha256"]:
            raise DatasetRegistryError("dataset output checksum mismatch")
        try:
            dataset = ad.read_h5ad(output_path)
            validate_benchmark_dataset(dataset)
            dataset_hash = benchmark_dataset_sha256(dataset)
            truth_hash = _truth_sha256(dataset)
        except (OSError, TypeError, ValueError) as error:
            raise DatasetRegistryError(
                "dataset output cannot be revalidated"
            ) from error
        if dataset_hash != row["dataset_sha256"]:
            raise DatasetRegistryError("dataset semantic checksum mismatch")
        if truth_hash != row["truth_sha256"]:
            raise DatasetRegistryError("dataset truth checksum mismatch")
        if dataset.shape != (request.cells, request.genes):
            raise DatasetRegistryError("dataset dimension drift")
        provenance = dataset.uns.get("provenance")
        parameters = (
            provenance.get("parameters") if isinstance(provenance, Mapping) else None
        )
        seeds = provenance.get("seeds") if isinstance(provenance, Mapping) else None
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("source_sha256") != row["source_sha256"]
            or not isinstance(parameters, Mapping)
            or parameters.get("native_manifest_sha256") != row["native_manifest_sha256"]
            or not isinstance(seeds, Mapping)
            or seeds.get("biological") != request.biological_seed
            or seeds.get("measurement") != request.measurement_seed
        ):
            raise DatasetRegistryError("dataset source or provenance mismatch")
        return (
            row["dataset_sha256"],
            row["output_file_sha256"],
            row["native_manifest_sha256"],
            output_identity,
        )
    if (
        row["status"] != "failed"
        or not isinstance(row["reason"], str)
        or not _FAILED_REASON.fullmatch(row["reason"])
    ):
        raise DatasetRegistryError("failed row status or reason is noncanonical")
    if (
        any(
            row[field] is not None
            for field in (
                "dataset_sha256",
                "truth_sha256",
                "source_sha256",
                "native_manifest_sha256",
            )
        )
        or row["native_files"] != []
    ):
        raise DatasetRegistryError("failed row contains unsupported artifact claims")
    if row["output_file_sha256"] is None:
        if os.path.lexists(request.output_path):
            raise DatasetRegistryError("failed row output appeared after recording")
    else:
        _validate_digest(row["output_file_sha256"], "failed output checksum")
        observed, output_identity = _file_sha256_secure(
            request.output_path, results_root, "failed adapter output"
        )
        if observed != row["output_file_sha256"]:
            raise DatasetRegistryError("failed adapter output checksum mismatch")
    return None, row["output_file_sha256"], None, output_identity


def _validate_rows(
    rows: object,
    *,
    designs: Sequence[_PairDesign],
    results_root: Path,
) -> list[dict[str, object]]:
    expected = [
        (design, request, index)
        for design in designs
        for index, request in enumerate(design.requests)
    ]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise DatasetRegistryError("dataset status row cardinality is incomplete")
    claimed_outputs = [
        row.get("output_path") for row in rows if isinstance(row, Mapping)
    ]
    if len(claimed_outputs) != len(set(claimed_outputs)):
        raise DatasetRegistryError("duplicate output alias in dataset status rows")
    claimed_dataset_ids = [
        row.get("dataset_id") for row in rows if isinstance(row, Mapping)
    ]
    if len(claimed_dataset_ids) != len(set(claimed_dataset_ids)):
        raise DatasetRegistryError("duplicate dataset ID in dataset status rows")
    for field, label in (
        ("dataset_sha256", "dataset checksum"),
        ("output_file_sha256", "output checksum"),
        ("native_manifest_sha256", "native manifest checksum"),
    ):
        claims = [
            row.get(field)
            for row in rows
            if isinstance(row, Mapping) and row.get(field) is not None
        ]
        if len(claims) != len(set(claims)):
            raise DatasetRegistryError(f"duplicate {label} in dataset status rows")
    dataset_hashes: list[str] = []
    output_hashes: list[str] = []
    manifest_hashes: list[str] = []
    output_identities: list[tuple[int, int]] = []
    validated: list[dict[str, object]] = []
    for row, (design, request, index) in zip(rows, expected, strict=True):
        dataset_hash, output_hash, manifest_hash, output_identity = _validate_row(
            row,
            design=design,
            request=request,
            index=index,
            results_root=results_root,
        )
        if dataset_hash is not None:
            dataset_hashes.append(dataset_hash)
        if output_hash is not None:
            output_hashes.append(output_hash)
        if manifest_hash is not None:
            manifest_hashes.append(manifest_hash)
        if output_identity is not None:
            output_identities.append(output_identity)
        assert isinstance(row, dict)
        validated.append(row)
    for values, label in (
        (dataset_hashes, "dataset checksum"),
        (output_hashes, "output checksum"),
        (manifest_hashes, "native manifest checksum"),
    ):
        if len(values) != len(set(values)):
            raise DatasetRegistryError(f"duplicate {label} in dataset status rows")
    if len(output_identities) != len(set(output_identities)):
        raise DatasetRegistryError("duplicate output alias in dataset status rows")
    independent_ids = [design.requests[0].independent_unit_id for design in designs]
    if len(independent_ids) != len(set(independent_ids)):
        raise DatasetRegistryError("biological draws do not form independent units")
    for offset in range(0, len(validated), 2):
        first, second = validated[offset : offset + 2]
        if first["status"] != second["status"]:
            raise DatasetRegistryError(
                "paired views cannot have mixed completion status"
            )
        if (
            first["status"] == "completed"
            and first["truth_sha256"] != second["truth_sha256"]
        ):
            raise DatasetRegistryError(
                "paired technical views do not share exact truth"
            )
    return validated


def _validate_receipt(
    receipt: Mapping[str, object],
    *,
    design: _PairDesign,
    design_sha256: str,
    namespace: str,
    results_root: Path,
) -> list[dict[str, object]]:
    if set(receipt) != {
        "schema_version",
        "design_sha256",
        "namespace",
        "mechanism",
        "biological_id",
        "rows",
        "receipt_sha256",
    }:
        raise DatasetRegistryError("dataset pair receipt has wrong schema")
    expected_hash = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    if receipt.get("receipt_sha256") != expected_hash:
        raise DatasetRegistryError("dataset pair receipt checksum mismatch")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("design_sha256") != design_sha256
        or receipt.get("namespace") != namespace
        or receipt.get("mechanism") != design.mechanism
        or receipt.get("biological_id") != design.biological_id
    ):
        raise DatasetRegistryError("dataset pair receipt mismatches design")
    return _validate_rows(
        receipt.get("rows"), designs=(design,), results_root=results_root
    )


def _status_payload(
    *,
    namespace: str,
    protocol_sha256: str,
    design_sha256: str,
    panel: DevelopmentPanel,
    claim: FinalManifestClaim | None,
    designs: Sequence[_PairDesign],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    completed = sum(row["status"] == "completed" for row in rows)
    failed = len(rows) - completed
    payload: dict[str, object] = {
        "schema_version": 1,
        "namespace": namespace,
        "status": "completed" if failed == 0 else "failed",
        "protocol_sha256": protocol_sha256,
        "design_sha256": design_sha256,
        "seed_source_sha256": (
            panel.file_sha256 if claim is None else claim.seed_manifest_sha256
        ),
        "execution_claim_id": None if claim is None else claim.execution_claim_id,
        "round_id": None if claim is None else claim.round_id,
        "independent_unit_count": len(designs),
        "completed_count": completed,
        "failed_count": failed,
        "rows": [dict(row) for row in rows],
    }
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def _incremental_result_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    results_root: Path,
    designs: Sequence[_PairDesign],
    include_status: bool,
) -> dict[str, object]:
    """Build the exact cumulative immutable-file list for the state journal."""

    round_dir = results_root.parent
    files: dict[str, str] = {}

    def add(relative_to_results: object, digest: object) -> None:
        relative = _safe_relative_path(relative_to_results, "journal result path")
        _validate_digest(digest, "journal result checksum")
        assert isinstance(digest, str)
        path = f"results/{relative.as_posix()}"
        previous = files.setdefault(path, digest)
        if previous != digest:
            raise DatasetRegistryError("journal result path changed checksum")

    for row in rows:
        add(row["log_path"], row["log_sha256"])
        if row["output_file_sha256"] is not None:
            add(row["output_path"], row["output_file_sha256"])
        native_files = row["native_files"]
        assert isinstance(native_files, list)
        for item in native_files:
            assert isinstance(item, Mapping)
            add(item["path"], item["sha256"])
    completed_pairs = len(rows) // 2
    for design in designs[:completed_pairs]:
        receipt = _receipt_path(results_root, design)
        digest, _identity = _file_sha256_secure(
            receipt, results_root, "dataset pair receipt"
        )
        add(_relative(results_root, receipt, "dataset pair receipt"), digest)
    if include_status:
        status_path = results_root / "dataset_status.json"
        digest, _identity = _file_sha256_secure(
            status_path, results_root, "dataset status manifest"
        )
        add("dataset_status.json", digest)
    # Every journal path is relative to the round and has already been proven
    # beneath round/results.  The state controller repeats the path/hash checks.
    if not all((round_dir / path).is_relative_to(results_root) for path in files):
        raise DatasetRegistryError("journal result path escapes the results root")
    return {
        "result_files": [
            {"path": path, "sha256": files[path]} for path in sorted(files)
        ]
    }


def _paths(
    repo: Path,
    namespace: str,
    round_dir: Path | None,
) -> tuple[Path, Path]:
    if namespace == "dev":
        if round_dir is not None:
            raise DatasetRegistryError(
                "development generation cannot receive a round directory"
            )
        results_root = repo / "artifacts/study/development/results"
    elif namespace == "final":
        if round_dir is None:
            raise DatasetRegistryError(
                "final generation requires a claimed round directory"
            )
        try:
            canonical_round = round_dir.absolute()
            canonical_round.relative_to(repo.absolute())
        except ValueError as error:
            raise DatasetRegistryError(
                "final round directory must be inside repository"
            ) from error
        results_root = canonical_round / "results"
    else:
        raise DatasetRegistryError("namespace must be exactly dev or final")
    return results_root, results_root / "dataset_status.json"


def _load_inputs(
    repo: Path,
    protocol_path: Path | None,
    development_panel_path: Path | None,
) -> tuple[Protocol, Path, DevelopmentPanel]:
    if not isinstance(repo, Path):
        raise TypeError("repo must be a pathlib.Path")
    repository = repo.resolve(strict=True)
    if not repository.is_dir():
        raise DatasetRegistryError("repo must be a directory")
    protocol_file = (
        repository / "study/protocol.json" if protocol_path is None else protocol_path
    )
    panel_file = (
        repository / "study/development_panel.json"
        if development_panel_path is None
        else development_panel_path
    )
    try:
        protocol = load_protocol(protocol_file)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetRegistryError(f"invalid study protocol: {error}") from error
    panel = load_development_panel(panel_file, protocol)
    return protocol, protocol_file, panel


def generate_dataset_panel(
    *,
    repo: Path,
    namespace: str,
    round_dir: Path | None = None,
    protocol_path: Path | None = None,
    development_panel_path: Path | None = None,
) -> dict[str, object]:
    """Generate or strictly resume the complete development or final panel."""

    protocol, protocol_file, panel = _load_inputs(
        repo, protocol_path, development_panel_path
    )
    repository = repo.resolve(strict=True)
    if namespace not in {protocol.development.namespace, protocol.final.namespace}:
        raise DatasetRegistryError("namespace must match the protocol exactly")
    results_root, status_path = _paths(repository, namespace, round_dir)
    claim: FinalManifestClaim | None = None
    if namespace == protocol.final.namespace:
        assert round_dir is not None
        try:
            claim = load_final_manifest_claim(repository, round_dir)
        except Exception as error:
            raise DatasetRegistryError(
                "final namespace requires the state controller's running execution claim"
            ) from error
    protocol_hash = file_sha256(protocol_file)
    if claim is not None and protocol_hash != claim._protocol_sha256:
        raise DatasetRegistryError(
            "final dataset panel must use the exact frozen protocol bytes"
        )
    design_hash = _design_sha256(namespace, protocol_hash, panel, claim)
    designs = _designs(
        namespace=namespace,
        protocol=protocol,
        panel=panel,
        results_root=results_root,
        claim=claim,
    )
    if status_path.exists():
        return validate_dataset_status(
            status_path,
            repo=repository,
            round_dir=round_dir,
            protocol_path=protocol_file,
            development_panel_path=(
                repository / "study/development_panel.json"
                if development_panel_path is None
                else development_panel_path
            ),
        )
    rows: list[dict[str, object]] = []
    for design in designs:
        rows.extend(
            _execute_pair(
                design=design,
                design_sha256=design_hash,
                namespace=namespace,
                protocol=protocol,
                claim=claim,
                results_root=results_root,
            )
        )
        if claim is not None:
            record_incremental_results(
                claim.round_dir,
                _incremental_result_manifest(
                    rows,
                    results_root=results_root,
                    designs=designs,
                    include_status=False,
                ),
                repo=repository,
            )
    validated_rows = _validate_rows(rows, designs=designs, results_root=results_root)
    status = _status_payload(
        namespace=namespace,
        protocol_sha256=protocol_hash,
        design_sha256=design_hash,
        panel=panel,
        claim=claim,
        designs=designs,
        rows=validated_rows,
    )
    _publish_new_json(status_path, status)
    if claim is not None:
        record_incremental_results(
            claim.round_dir,
            _incremental_result_manifest(
                validated_rows,
                results_root=results_root,
                designs=designs,
                include_status=True,
            ),
            repo=repository,
        )
    return status


def validate_dataset_status(
    path: Path,
    *,
    repo: Path,
    round_dir: Path | None = None,
    protocol_path: Path | None = None,
    development_panel_path: Path | None = None,
) -> dict[str, object]:
    """Revalidate a canonical status manifest and every bound output byte."""

    protocol, protocol_file, panel = _load_inputs(
        repo, protocol_path, development_panel_path
    )
    repository = repo.resolve(strict=True)
    payload = _read_canonical_json(path, "dataset status manifest")
    if set(payload) != _STATUS_KEYS:
        raise DatasetRegistryError("dataset status manifest has wrong schema")
    namespace = payload.get("namespace")
    if namespace not in {protocol.development.namespace, protocol.final.namespace}:
        raise DatasetRegistryError("dataset status namespace is invalid")
    results_root, expected_path = _paths(repository, namespace, round_dir)
    if path.absolute() != expected_path.absolute():
        raise DatasetRegistryError("dataset status path is not canonical")
    claim: FinalManifestClaim | None = None
    if namespace == protocol.final.namespace:
        assert round_dir is not None
        try:
            claim = load_final_manifest_claim(repository, round_dir)
        except Exception as error:
            raise DatasetRegistryError(
                "final status requires the journaled running execution claim"
            ) from error
    protocol_hash = file_sha256(protocol_file)
    if claim is not None and protocol_hash != claim._protocol_sha256:
        raise DatasetRegistryError(
            "final dataset status does not use the exact frozen protocol bytes"
        )
    design_hash = _design_sha256(namespace, protocol_hash, panel, claim)
    designs = _designs(
        namespace=namespace,
        protocol=protocol,
        panel=panel,
        results_root=results_root,
        claim=claim,
    )
    expected_manifest_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    if payload.get("manifest_sha256") != expected_manifest_hash:
        raise DatasetRegistryError("dataset status manifest checksum mismatch")
    expected_top = {
        "schema_version": 1,
        "namespace": namespace,
        "protocol_sha256": protocol_hash,
        "design_sha256": design_hash,
        "seed_source_sha256": (
            panel.file_sha256 if claim is None else claim.seed_manifest_sha256
        ),
        "execution_claim_id": None if claim is None else claim.execution_claim_id,
        "round_id": None if claim is None else claim.round_id,
        "independent_unit_count": len(designs),
    }
    for key, expected in expected_top.items():
        if payload.get(key) != expected:
            raise DatasetRegistryError(f"dataset status {key} mismatches design")
    rows = _validate_rows(
        payload.get("rows"), designs=designs, results_root=results_root
    )
    completed = sum(row["status"] == "completed" for row in rows)
    failed = len(rows) - completed
    if (
        payload.get("completed_count") != completed
        or payload.get("failed_count") != failed
        or payload.get("status") != ("completed" if failed == 0 else "failed")
    ):
        raise DatasetRegistryError("dataset status counts or state are inconsistent")
    return payload


__all__ = [
    "DatasetRegistryError",
    "DevelopmentPanel",
    "generate_dataset_panel",
    "load_development_panel",
    "validate_dataset_status",
]
