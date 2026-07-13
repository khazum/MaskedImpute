"""Pinned, donor-disjoint semisynthetic pancreas validation adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import ctypes
from dataclasses import dataclass
import errno
import gzip
import hashlib
import io
from importlib.metadata import version as distribution_version
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
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
_ARCHIVE = _EXTERNAL_ROOT / "data/baron-pancreas-umi/GSE84133_RAW.tar"
_ARCHIVE_NAME = "GSE84133_RAW.tar"
_ARCHIVE_SHA256 = "aed2d208d47a36658aa0e63629afe5d4144ef465a8e3d9a0f377422b1f1073dc"
_PARTITION_RULE = "human1+human2_development__human3+human4_final"
_PARTITIONS = {
    "dev": (
        "GSM2230757_human1_umifm_counts.csv.gz",
        "GSM2230758_human2_umifm_counts.csv.gz",
    ),
    "final": (
        "GSM2230759_human3_umifm_counts.csv.gz",
        "GSM2230760_human4_umifm_counts.csv.gz",
    ),
}
_ALL_ARCHIVE_MEMBERS = frozenset(
    {
        *_PARTITIONS["dev"],
        *_PARTITIONS["final"],
        "GSM2230761_mouse1_umifm_counts.csv.gz",
        "GSM2230762_mouse2_umifm_counts.csv.gz",
    }
)
_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_MEMBER_BYTES = 32 * 1024 * 1024
_MAX_HEADER_BYTES = 4 * 1024 * 1024
_MIN_CELLS_PER_TYPE = 20
_MAX_DISPERSION_ALPHA = 5.0
_VIEW_PARAMETERS: dict[str, dict[str, float | str]] = {
    "moderate": {
        "observed_probability": 0.50,
        "heldout_probability": 0.10,
        "split_model": "disjoint_multinomial_via_sequential_binomial",
    },
    "severe": {
        "observed_probability": 0.25,
        "heldout_probability": 0.10,
        "split_model": "disjoint_multinomial_via_sequential_binomial",
    },
}
_GENE_SELECTION_RULE = "pooled_total_umi_descending_then_gene_id_ascending"
_EXPECTED_NATIVE_FILES = frozenset(
    {
        "cell_type_index.npy",
        "cell_type_probabilities.npy",
        "cell_types.json",
        "config.json",
        "dispersion_alpha.npy",
        "fit_metadata.json",
        "gene_ids.json",
        "heldout_moderate.npy",
        "heldout_severe.npy",
        "library_log_parameters.npy",
        "mean_fraction.npy",
        "observed_moderate.npy",
        "observed_severe.npy",
        "reference_counts.npy",
        "run_metadata.json",
    }
)
_CELL_TYPE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class _DonorSummary:
    member: str
    row_ids: tuple[str, ...]
    barcodes: tuple[str, ...]
    groups: tuple[str, ...]
    library_sizes: np.ndarray
    gene_totals: np.ndarray


@dataclass(frozen=True, slots=True)
class _SourceFit:
    namespace: str
    donors: tuple[str, ...]
    gene_ids: tuple[str, ...]
    selected_gene_indices: np.ndarray
    cell_types: tuple[str, ...]
    cell_type_counts: np.ndarray
    cell_type_probabilities: np.ndarray
    mean_fraction: np.ndarray
    dispersion_alpha: np.ndarray
    library_log_parameters: np.ndarray
    donor_row_counts: tuple[int, ...]
    available_positive_genes: int
    fit_sha256: str


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SimulationContractError(
                f"duplicate JSON key in semisynthetic native output: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SimulationContractError(
        f"non-finite JSON value in semisynthetic native output: {value}"
    )


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


def _environment_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "anndata": distribution_version("anndata"),
    }


def _read_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before_path = path.lstat()
        if (
            not stat.S_ISREG(before_path.st_mode)
            or before_path.st_nlink != 1
            or before_path.st_size > maximum_bytes
        ):
            raise SimulationContractError(
                f"semisynthetic input must be a bounded unique regular file: {path.name}"
            )
        descriptor = os.open(path, flags)
    except SimulationContractError:
        raise
    except OSError as error:
        raise SimulationContractError(
            f"semisynthetic input cannot be opened safely: {path.name}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum_bytes
            or (before.st_dev, before.st_ino)
            != (before_path.st_dev, before_path.st_ino)
        ):
            raise SimulationContractError(
                f"semisynthetic input changed while opening: {path.name}"
            )
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining == 0 and os.read(descriptor, 1):
            raise SimulationContractError(
                f"semisynthetic input exceeds its byte limit: {path.name}"
            )
        after = os.fstat(descriptor)
        after_path = path.lstat()
        state = lambda item: (  # noqa: E731
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if state(before) != state(after) or state(before) != state(after_path):
            raise SimulationContractError(
                f"semisynthetic input changed while reading: {path.name}"
            )
        return b"".join(chunks)
    except OSError as error:
        raise SimulationContractError(
            f"semisynthetic input changed while reading: {path.name}"
        ) from error
    finally:
        os.close(descriptor)


def _verify_semisynthetic_source(
    *, external_root: Path | None = None, immutable: bool = False
) -> tuple[Path, dict[str, object]]:
    selected_root = _EXTERNAL_ROOT if external_root is None else external_root
    archive_path = (
        _ARCHIVE
        if external_root is None
        else selected_root / "data/baron-pancreas-umi/GSE84133_RAW.tar"
    )
    try:
        ledger = load_source_ledger(_LEDGER_PATH)
        verifier = verify_fetched_sources if immutable else fetch_sources
        receipt = verifier(ledger, selected_root, source_ids=("baron-pancreas-umi",))[0]
    except (OSError, SourceLedgerError, TypeError, ValueError) as error:
        raise SimulationContractError(
            f"pinned semisynthetic source is unavailable: {error}"
        ) from error
    artifacts = receipt.get("artifacts")
    expected_artifact = {
        "name": _ARCHIVE_NAME,
        "sha256": _ARCHIVE_SHA256,
        "size_bytes": archive_path.stat().st_size if archive_path.is_file() else -1,
    }
    if (
        receipt.get("source_id") != "baron-pancreas-umi"
        or receipt.get("resolved_revision") != "GSE84133:2019-05-15"
        or not isinstance(artifacts, list)
        or len(artifacts) != 1
        or artifacts[0] != expected_artifact
        or file_sha256(archive_path) != _ARCHIVE_SHA256
    ):
        raise SimulationContractError(
            "semisynthetic source receipt does not match the exact Baron archive pin"
        )
    return archive_path, receipt


def _source_archive_sha256(source_receipt: Mapping[str, object]) -> str:
    artifacts = source_receipt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise SimulationContractError(
            "semisynthetic source receipt must bind exactly one archive"
        )
    artifact = artifacts[0]
    if not isinstance(artifact, Mapping):
        raise SimulationContractError(
            "semisynthetic source receipt artifact must be an object"
        )
    digest = artifact.get("sha256")
    if (
        artifact.get("name") != _ARCHIVE_NAME
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise SimulationContractError(
            "semisynthetic source receipt artifact checksum is invalid"
        )
    return digest


def _archive_payloads(archive_bytes: bytes) -> dict[str, bytes]:
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if (
                len(members) != len(_ALL_ARCHIVE_MEMBERS)
                or set(names) != _ALL_ARCHIVE_MEMBERS
                or len(names) != len(set(names))
            ):
                raise SimulationContractError(
                    "Baron archive does not contain the exact closed donor-member set"
                )
            payloads: dict[str, bytes] = {}
            for member in members:
                if (
                    not member.isreg()
                    or member.name != Path(member.name).name
                    or member.size <= 0
                    or member.size > _MAX_MEMBER_BYTES
                ):
                    raise SimulationContractError(
                        "Baron archive members must be bounded regular basenames"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise SimulationContractError(
                        f"Baron archive member cannot be read: {member.name}"
                    )
                payload = handle.read(_MAX_MEMBER_BYTES + 1)
                if len(payload) != member.size or len(payload) > _MAX_MEMBER_BYTES:
                    raise SimulationContractError(
                        f"Baron archive member size changed: {member.name}"
                    )
                payloads[member.name] = payload
            return payloads
    except SimulationContractError:
        raise
    except (OSError, EOFError, tarfile.TarError) as error:
        raise SimulationContractError(
            "Baron source is not a valid tar archive"
        ) from error


def _csv_header(payload: bytes, member: str) -> list[str]:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
            line = compressed.readline(_MAX_HEADER_BYTES + 1)
    except (OSError, EOFError) as error:
        raise SimulationContractError(f"{member} is not valid gzip data") from error
    if (
        not line.endswith(b"\n")
        or len(line) > _MAX_HEADER_BYTES
        or b"\r" in line
        or b"\x00" in line
    ):
        raise SimulationContractError(f"{member} has a noncanonical CSV header")
    try:
        decoded = line.decode("utf-8")
        rows = list(csv.reader([decoded.removesuffix("\n")], strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise SimulationContractError(f"{member} has a malformed CSV header") from error
    if len(rows) != 1:
        raise SimulationContractError(f"{member} has a malformed CSV header")
    header = rows[0]
    if (
        len(header) < 4
        or header[:3] != ["", "barcode", "assigned_cluster"]
        or any(not value or value != value.strip() for value in header[3:])
        or len(header[3:]) != len(set(header[3:]))
    ):
        raise SimulationContractError(
            f"{member} must contain index, barcode, assigned_cluster, then unique genes"
        )
    return header


def _read_donor_frame(
    payload: bytes,
    member: str,
    *,
    selected_genes: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    header = _csv_header(payload, member)
    gene_ids = tuple(header[3:])
    usecols = (
        None
        if selected_genes is None
        else ["barcode", "assigned_cluster", *selected_genes]
    )
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
            frame = pd.read_csv(
                compressed,
                index_col=0 if usecols is None else None,
                usecols=usecols,
                low_memory=False,
            )
    except (
        OSError,
        EOFError,
        UnicodeDecodeError,
        ValueError,
        pd.errors.ParserError,
    ) as error:
        raise SimulationContractError(f"{member} is malformed CSV") from error
    expected_columns = (
        ["barcode", "assigned_cluster", *gene_ids]
        if selected_genes is None
        else ["barcode", "assigned_cluster", *selected_genes]
    )
    if frame.columns.tolist() != expected_columns or frame.empty:
        raise SimulationContractError(f"{member} has wrong columns or no cells")
    if selected_genes is None:
        row_ids = frame.index.tolist()
        if not frame.index.is_unique or any(
            not isinstance(value, str) or not value.strip() for value in row_ids
        ):
            raise SimulationContractError(f"{member} has invalid source cell IDs")
    barcodes = frame["barcode"].tolist()
    groups = frame["assigned_cluster"].tolist()
    if any(not isinstance(value, str) or not value.strip() for value in barcodes):
        raise SimulationContractError(f"{member} has invalid barcodes")
    if any(
        not isinstance(value, str) or not _CELL_TYPE.fullmatch(value)
        for value in groups
    ):
        raise SimulationContractError(f"{member} has invalid assigned_cluster labels")
    count_columns = gene_ids if selected_genes is None else selected_genes
    counts = frame.loc[:, list(count_columns)]
    if not all(np.issubdtype(dtype, np.integer) for dtype in counts.dtypes):
        raise SimulationContractError(f"{member} gene columns must be integer counts")
    matrix = counts.to_numpy(copy=False)
    if matrix.dtype == np.dtype("O") or bool((matrix < 0).any()):
        raise SimulationContractError(f"{member} counts must be nonnegative integers")
    maximum = int(matrix.max(initial=0))
    if maximum > np.iinfo(np.int64).max // max(1, matrix.shape[1], matrix.shape[0]):
        raise SimulationContractError(f"{member} counts cannot be summed in int64")
    return frame, gene_ids


def _summarize_donor(
    payload: bytes, member: str
) -> tuple[_DonorSummary, tuple[str, ...]]:
    frame, gene_ids = _read_donor_frame(payload, member)
    matrix = frame.loc[:, list(gene_ids)].to_numpy(dtype=np.int64, copy=False)
    return (
        _DonorSummary(
            member=member,
            row_ids=tuple(str(value) for value in frame.index.tolist()),
            barcodes=tuple(str(value) for value in frame["barcode"].tolist()),
            groups=tuple(str(value) for value in frame["assigned_cluster"].tolist()),
            library_sizes=matrix.sum(axis=1, dtype=np.int64),
            gene_totals=matrix.sum(axis=0, dtype=np.int64),
        ),
        gene_ids,
    )


def _fit_source(
    archive_path: Path,
    namespace: str,
    requested_genes: int,
    *,
    expected_sha256: str | None = None,
) -> _SourceFit:
    archive_bytes = _read_regular_bytes(archive_path, maximum_bytes=_MAX_ARCHIVE_BYTES)
    if expected_sha256 is not None and (
        re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or hashlib.sha256(archive_bytes).hexdigest() != expected_sha256
    ):
        raise SimulationContractError(
            "semisynthetic source snapshot checksum does not match its receipt"
        )
    payloads = _archive_payloads(archive_bytes)
    try:
        donors = _PARTITIONS[namespace]
    except KeyError as error:
        raise SimulationContractError(
            "semisynthetic namespace has no donor partition"
        ) from error
    summaries: list[_DonorSummary] = []
    common_genes: tuple[str, ...] | None = None
    for member in donors:
        summary, gene_ids = _summarize_donor(payloads[member], member)
        if common_genes is None:
            common_genes = gene_ids
        elif gene_ids != common_genes:
            raise SimulationContractError(
                "Baron donor files do not have identical gene columns and orientation"
            )
        summaries.append(summary)
    assert common_genes is not None
    pooled_totals = np.zeros(len(common_genes), dtype=np.int64)
    for summary in summaries:
        if bool((pooled_totals > np.iinfo(np.int64).max - summary.gene_totals).any()):
            raise SimulationContractError("pooled Baron gene totals exceed int64")
        pooled_totals += summary.gene_totals
    positive = [index for index, value in enumerate(pooled_totals) if int(value) > 0]
    if type(requested_genes) is not int or requested_genes <= 0:
        raise SimulationContractError("requested semisynthetic genes must be positive")
    if requested_genes > len(positive):
        raise SimulationContractError(
            "semisynthetic source has fewer expressed genes than requested"
        )
    ranked = sorted(
        positive,
        key=lambda index: (-int(pooled_totals[index]), common_genes[index]),
    )[:requested_genes]
    selected_indices = np.asarray(sorted(ranked), dtype=np.int64)
    selected_genes = tuple(common_genes[index] for index in selected_indices)

    selected_counts: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    all_libraries: list[np.ndarray] = []
    for summary in summaries:
        frame, observed_genes = _read_donor_frame(
            payloads[summary.member],
            summary.member,
            selected_genes=selected_genes,
        )
        if observed_genes != common_genes:
            raise SimulationContractError(
                "Baron donor gene header changed between reads"
            )
        if (
            tuple(str(value) for value in frame["barcode"].tolist()) != summary.barcodes
            or tuple(str(value) for value in frame["assigned_cluster"].tolist())
            != summary.groups
        ):
            raise SimulationContractError("Baron donor rows changed between fit passes")
        selected_counts.append(
            frame.loc[:, list(selected_genes)].to_numpy(dtype=np.int64, copy=True)
        )
        all_groups.append(np.asarray(summary.groups, dtype=str))
        all_libraries.append(summary.library_sizes.copy())
    counts = np.concatenate(selected_counts, axis=0)
    groups = np.concatenate(all_groups)
    libraries = np.concatenate(all_libraries)
    if bool((libraries <= 0).any()):
        raise SimulationContractError("Baron source cells must have positive libraries")
    group_counts = {
        group: int((groups == group).sum()) for group in sorted(set(groups.tolist()))
    }
    cell_types = tuple(
        group for group, count in group_counts.items() if count >= _MIN_CELLS_PER_TYPE
    )
    if len(cell_types) < 2:
        raise SimulationContractError(
            "semisynthetic fit requires at least two adequately represented cell types"
        )
    eligible = np.isin(groups, cell_types)
    eligible_counts = counts[eligible]
    eligible_groups = groups[eligible]
    eligible_libraries = libraries[eligible].astype(np.float64)
    cell_type_counts = np.asarray(
        [int((eligible_groups == group).sum()) for group in cell_types],
        dtype=np.int64,
    )
    probabilities = cell_type_counts.astype(np.float64)
    probabilities /= probabilities.sum(dtype=np.float64)
    log_libraries = np.log(eligible_libraries)
    log_parameters = np.asarray(
        [
            float(log_libraries.mean(dtype=np.float64)),
            float(log_libraries.std(ddof=1, dtype=np.float64)),
        ],
        dtype="<f8",
    )
    median_library = float(np.median(eligible_libraries))
    mean_fraction = np.zeros((len(cell_types), requested_genes), dtype="<f8")
    dispersion = np.zeros_like(mean_fraction)
    for type_index, group in enumerate(cell_types):
        selected = eligible_groups == group
        group_counts_matrix = eligible_counts[selected].astype(np.float64)
        group_libraries = eligible_libraries[selected]
        total_library = float(group_libraries.sum(dtype=np.float64))
        mean_fraction[type_index] = (
            group_counts_matrix.sum(axis=0, dtype=np.float64) / total_library
        )
        normalized = (
            group_counts_matrix
            * np.divide(
                median_library,
                group_libraries,
            )[:, None]
        )
        mean_at_median = mean_fraction[type_index] * median_library
        variance = normalized.var(axis=0, ddof=1, dtype=np.float64)
        positive_mean = mean_at_median > 0
        raw_alpha = np.zeros(requested_genes, dtype=np.float64)
        raw_alpha[positive_mean] = np.maximum(
            variance[positive_mean] - mean_at_median[positive_mean], 0.0
        ) / np.square(mean_at_median[positive_mean])
        dispersion[type_index] = np.clip(raw_alpha, 0.0, _MAX_DISPERSION_ALPHA)
    fit_payload = {
        "schema": "maskimpute-semisynthetic-fit-v1",
        "namespace": namespace,
        "donors": list(donors),
        "gene_ids": list(selected_genes),
        "cell_types": list(cell_types),
        "cell_type_counts": cell_type_counts.tolist(),
        "cell_type_probabilities": probabilities.tolist(),
        "mean_fraction": mean_fraction.tolist(),
        "dispersion_alpha": dispersion.tolist(),
        "library_log_parameters": log_parameters.tolist(),
    }
    return _SourceFit(
        namespace=namespace,
        donors=donors,
        gene_ids=selected_genes,
        selected_gene_indices=selected_indices,
        cell_types=cell_types,
        cell_type_counts=cell_type_counts,
        cell_type_probabilities=probabilities.astype("<f8", copy=False),
        mean_fraction=mean_fraction,
        dispersion_alpha=dispersion,
        library_log_parameters=log_parameters,
        donor_row_counts=tuple(len(summary.groups) for summary in summaries),
        available_positive_genes=len(positive),
        fit_sha256=canonical_sha256(fit_payload),
    )


def _mapped_numpy_seed(seed: int, domain: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise SimulationContractError(
            "semisynthetic seeds must be 63-bit nonnegative integers"
        )
    return int(
        canonical_sha256(
            {
                "schema": "maskimpute-semisynthetic-numpy-seed-v1",
                "domain": domain,
                "seed": seed,
            }
        )[:16],
        16,
    )


def _mapped_seeds(requests: Mapping[str, SimulationRequest]) -> dict[str, int]:
    originals = {
        "biological": requests["moderate"].biological_seed,
        "moderate": requests["moderate"].measurement_seed,
        "severe": requests["severe"].measurement_seed,
    }
    mapped: dict[str, int] = {}
    used: set[int] = set()
    for role, seed in originals.items():
        candidate = _mapped_numpy_seed(seed, role)
        while candidate in used:
            candidate = (candidate + 1) % 2**64
        mapped[role] = candidate
        used.add(candidate)
    return mapped


def _fit_metadata(fit: _SourceFit) -> dict[str, object]:
    return {
        "schema_version": 1,
        "fit_sha256": fit.fit_sha256,
        "source_partition": {
            "namespace": fit.namespace,
            "donors": list(fit.donors),
            "partition_rule": _PARTITION_RULE,
            "donor_row_counts": list(fit.donor_row_counts),
        },
        "gene_selection": {
            "rule": _GENE_SELECTION_RULE,
            "output_order": "source_column_order",
            "available_positive_genes": fit.available_positive_genes,
            "selected_gene_ids": list(fit.gene_ids),
            "selected_gene_ids_sha256": canonical_sha256(list(fit.gene_ids)),
        },
        "cell_type_model": {
            "minimum_source_cells_per_type": _MIN_CELLS_PER_TYPE,
            "cell_types": list(fit.cell_types),
            "source_cell_counts": fit.cell_type_counts.tolist(),
            "proportion_rule": "pooled_eligible_source_cells_largest_remainder",
        },
        "gamma_poisson_model": {
            "mean": "pooled_umi_over_pooled_full_library_by_cell_type_gene",
            "dispersion": "method_of_moments_on_median_library_normalized_counts",
            "dispersion_parameterization": "variance=mean+alpha*mean_squared",
            "maximum_alpha": _MAX_DISPERSION_ALPHA,
        },
        "library_size_model": {
            "distribution": "lognormal",
            "fit": "mean_and_sample_sd_of_log_full_library_in_eligible_cells",
            "high_depth_multiplier": 1.0,
        },
    }


def _pair_config(
    requests: Mapping[str, SimulationRequest],
    fit: _SourceFit,
    source_receipt: Mapping[str, object],
) -> dict[str, object]:
    moderate = requests["moderate"]
    mapped = _mapped_seeds(requests)
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
            "python_adapter_sha256": file_sha256(Path(__file__)),
            "schema": "maskimpute-semisynthetic-adapter-v1",
        },
        "fit": _fit_metadata(fit),
        "schema_version": 1,
        "seeds": {
            "biological": {
                "original": moderate.biological_seed,
                "mapped_numpy": mapped["biological"],
            }
        },
        "simulation": {
            "cells": moderate.cells,
            "genes": moderate.genes,
            "native_orientation": "cells_by_genes",
            "reference_kind": "gamma_poisson_high_depth_proxy",
        },
        "source_receipt_sha256": canonical_sha256(source_receipt),
        "views": views,
    }


def _largest_remainder_counts(probabilities: np.ndarray, cells: int) -> np.ndarray:
    expected = probabilities * cells
    allocation = np.floor(expected).astype(np.int64)
    remaining = cells - int(allocation.sum(dtype=np.int64))
    order = sorted(
        range(len(probabilities)),
        key=lambda index: (-(expected[index] - allocation[index]), index),
    )
    for index in order[:remaining]:
        allocation[index] += 1
    if int(allocation.sum(dtype=np.int64)) != cells:
        raise SimulationContractError(
            "cell-type allocation did not preserve cell count"
        )
    return allocation


def _generate_reference(
    fit: _SourceFit, config: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    simulation = config["simulation"]
    seeds = config["seeds"]
    assert isinstance(simulation, Mapping)
    assert isinstance(seeds, Mapping)
    biological = seeds["biological"]
    assert isinstance(biological, Mapping)
    cells = int(simulation["cells"])
    rng = np.random.Generator(np.random.PCG64(int(biological["mapped_numpy"])))
    allocation = _largest_remainder_counts(fit.cell_type_probabilities, cells)
    type_index = np.concatenate(
        [
            np.full(int(count), index, dtype=np.int64)
            for index, count in enumerate(allocation)
            if count > 0
        ]
    )
    type_index = type_index[rng.permutation(cells)]
    log_mean, log_sd = fit.library_log_parameters.tolist()
    maximum_library = np.iinfo(np.int64).max // max(1, fit.mean_fraction.shape[1])
    maximum_log_library = float(np.log(maximum_library))
    if (
        not np.isfinite(log_mean)
        or not np.isfinite(log_sd)
        or log_sd < 0
        or log_mean + 12.0 * log_sd > maximum_log_library
    ):
        raise SimulationContractError(
            "fitted library-size distribution exceeds int64 safety"
        )
    library_draws = rng.lognormal(log_mean, log_sd, size=cells)
    if not np.isfinite(library_draws).all() or bool(
        (library_draws > maximum_library).any()
    ):
        raise SimulationContractError(
            "sampled library-size distribution exceeds int64 safety"
        )
    libraries = np.rint(library_draws).astype(np.int64)
    libraries = np.maximum(libraries, 1)
    expected = fit.mean_fraction[type_index] * libraries[:, None]
    alpha = fit.dispersion_alpha[type_index]
    latent = expected.copy()
    dispersed = (expected > 0) & (alpha > 0)
    if bool(dispersed.any()):
        shape = np.divide(
            1.0,
            alpha,
            out=np.ones_like(alpha),
            where=alpha > 0,
        )
        scale = expected * alpha
        latent[dispersed] = rng.gamma(shape[dispersed], scale[dispersed])
    if not np.isfinite(latent).all() or bool((latent < 0).any()):
        raise SimulationContractError("Gamma-Poisson latent rates are invalid")
    if float(latent.max(initial=0.0)) > np.iinfo(np.int64).max / 16:
        raise SimulationContractError("Gamma-Poisson latent rate exceeds int64 safety")
    reference = rng.poisson(latent).astype("<i8", copy=False)
    if int(reference.max(initial=0)) > np.iinfo(np.int64).max // max(
        1, reference.shape[1]
    ):
        raise SimulationContractError(
            "Gamma-Poisson reference cannot be summed safely in int64"
        )
    return reference, type_index.astype("<i8", copy=False)


def _split_view(
    reference: np.ndarray,
    *,
    observed_probability: float,
    heldout_probability: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < observed_probability < 1 or not 0 < heldout_probability < 1:
        raise SimulationContractError("semisynthetic split probabilities are invalid")
    if observed_probability + heldout_probability >= 1:
        raise SimulationContractError(
            "semisynthetic split probabilities must sum below one"
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    heldout = rng.binomial(reference, heldout_probability).astype("<i8", copy=False)
    remaining = reference - heldout
    conditional = observed_probability / (1.0 - heldout_probability)
    observed = rng.binomial(remaining, conditional).astype("<i8", copy=False)
    return observed, heldout


def _write_new_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _save_npy(path: Path, values: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    _write_new_bytes(path, buffer.getvalue())


def _generate_native(
    fit: _SourceFit,
    config: Mapping[str, object],
    output_dir: Path,
) -> None:
    config_bytes = _canonical_json_bytes(config)
    _write_new_bytes(output_dir / "config.json", config_bytes)
    reference, type_index = _generate_reference(fit, config)
    arrays: dict[str, np.ndarray] = {
        "cell_type_index.npy": type_index,
        "cell_type_probabilities.npy": fit.cell_type_probabilities,
        "dispersion_alpha.npy": fit.dispersion_alpha,
        "library_log_parameters.npy": fit.library_log_parameters,
        "mean_fraction.npy": fit.mean_fraction,
        "reference_counts.npy": reference,
    }
    views = config["views"]
    assert isinstance(views, list)
    for view in views:
        assert isinstance(view, Mapping)
        name = str(view["technical_view"])
        observed, heldout = _split_view(
            reference,
            observed_probability=float(view["observed_probability"]),
            heldout_probability=float(view["heldout_probability"]),
            seed=int(view["measurement_seed_numpy"]),
        )
        arrays[f"observed_{name}.npy"] = observed
        arrays[f"heldout_{name}.npy"] = heldout
    for name in sorted(arrays):
        _save_npy(output_dir / name, arrays[name])
    _write_new_bytes(
        output_dir / "cell_types.json",
        _canonical_json_bytes(
            {"cell_types": list(fit.cell_types), "schema_version": 1}
        ),
    )
    _write_new_bytes(
        output_dir / "gene_ids.json",
        _canonical_json_bytes({"gene_ids": list(fit.gene_ids), "schema_version": 1}),
    )
    _write_new_bytes(
        output_dir / "fit_metadata.json",
        _canonical_json_bytes(_fit_metadata(fit)),
    )
    bound_files = sorted(_EXPECTED_NATIVE_FILES - {"run_metadata.json"})
    hashes = {name: file_sha256(output_dir / name) for name in bound_files}
    run_metadata = {
        "schema_version": 1,
        "fit_calls": 1,
        "reference_draw_calls": 1,
        "technical_split_calls": 2,
        "native_file_sha256": hashes,
        "versions": _environment_versions(),
    }
    _write_new_bytes(
        output_dir / "run_metadata.json", _canonical_json_bytes(run_metadata)
    )


def _validate_stage_entries(stage: Path) -> dict[str, Path]:
    try:
        entries = list(os.scandir(stage))
    except OSError as error:
        raise SimulationContractError(
            "semisynthetic native output directory is unavailable"
        ) from error
    names = {entry.name for entry in entries}
    if names != _EXPECTED_NATIVE_FILES or len(entries) != len(_EXPECTED_NATIVE_FILES):
        raise SimulationContractError(
            "semisynthetic native outputs do not match the closed file set"
        )
    files: dict[str, Path] = {}
    for entry in entries:
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as error:
            raise SimulationContractError(
                "semisynthetic native output cannot be inspected"
            ) from error
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise SimulationContractError(
                "semisynthetic native outputs must be unique regular files"
            )
        files[entry.name] = Path(entry.path)
    return files


def _read_npy(
    path: Path, *, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> np.ndarray:
    itemsize = int(dtype.itemsize)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * itemsize
    data = _read_regular_bytes(path, maximum_bytes=expected_bytes + 8192)
    stream = io.BytesIO(data)
    try:
        values = np.load(stream, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SimulationContractError(
            f"{path.name} is not a valid NPY array"
        ) from error
    if (
        stream.tell() != len(data)
        or not isinstance(values, np.ndarray)
        or values.shape != shape
        or values.dtype != dtype
    ):
        raise SimulationContractError(
            f"{path.name} has wrong orientation, shape, dtype, or trailing bytes"
        )
    return values.copy()


def _read_canonical_json(path: Path, *, maximum_bytes: int) -> object:
    data = _read_regular_bytes(path, maximum_bytes=maximum_bytes)
    value = _load_json_bytes(data, path.name)
    if data != _canonical_json_bytes(value):
        raise SimulationContractError(f"{path.name} must be canonical JSON")
    return value


def _validate_run_metadata(path: Path, files: Mapping[str, Path]) -> dict[str, object]:
    value = _read_canonical_json(path, maximum_bytes=1024 * 1024)
    if not isinstance(value, dict):
        raise SimulationContractError("run_metadata.json must be an object")
    expected_keys = {
        "schema_version",
        "fit_calls",
        "reference_draw_calls",
        "technical_split_calls",
        "native_file_sha256",
        "versions",
    }
    hashes = value.get("native_file_sha256")
    versions = value.get("versions")
    bound_files = sorted(_EXPECTED_NATIVE_FILES - {"run_metadata.json"})
    if (
        set(value) != expected_keys
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(bound_files)
        or not isinstance(versions, Mapping)
        or set(versions) != {"python", "numpy", "pandas", "anndata"}
        or not all(isinstance(item, str) and item for item in versions.values())
    ):
        raise SimulationContractError(
            "run_metadata.json does not bind the exact semisynthetic run"
        )
    if any(
        type(value.get(name)) is not int or value.get(name) != expected
        for name, expected in (
            ("fit_calls", 1),
            ("reference_draw_calls", 1),
            ("technical_split_calls", 2),
        )
    ):
        raise SimulationContractError(
            "run_metadata.json call counts must be exact integers"
        )
    if dict(versions) != _environment_versions():
        raise SimulationContractError(
            "run_metadata.json environment versions do not match execution"
        )
    for name in bound_files:
        if hashes.get(name) != file_sha256(files[name]):
            raise SimulationContractError(
                f"run_metadata.json does not bind exact bytes for {name}"
            )
    return value


def _load_native(
    files: Mapping[str, Path],
    config: Mapping[str, object],
    fit: _SourceFit,
    cells: int,
    genes: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
]:
    config_bytes = _read_regular_bytes(files["config.json"], maximum_bytes=1024 * 1024)
    if config_bytes != _canonical_json_bytes(config) or canonical_sha256(
        _load_json_bytes(config_bytes, "config.json")
    ) != canonical_sha256(config):
        raise SimulationContractError("native runner changed its sealed config")
    cell_types = _read_canonical_json(
        files["cell_types.json"], maximum_bytes=1024 * 1024
    )
    gene_ids = _read_canonical_json(files["gene_ids.json"], maximum_bytes=1024 * 1024)
    fit_metadata = _read_canonical_json(
        files["fit_metadata.json"], maximum_bytes=4 * 1024 * 1024
    )
    expected_cell_types = {"cell_types": list(fit.cell_types), "schema_version": 1}
    expected_gene_ids = {"gene_ids": list(fit.gene_ids), "schema_version": 1}
    if (
        canonical_sha256(cell_types) != canonical_sha256(expected_cell_types)
        or canonical_sha256(gene_ids) != canonical_sha256(expected_gene_ids)
        or canonical_sha256(fit_metadata) != canonical_sha256(_fit_metadata(fit))
    ):
        raise SimulationContractError("native fit labels or metadata changed")
    mean_fraction = _read_npy(
        files["mean_fraction.npy"],
        dtype=np.dtype("<f8"),
        shape=(len(fit.cell_types), genes),
    )
    dispersion = _read_npy(
        files["dispersion_alpha.npy"],
        dtype=np.dtype("<f8"),
        shape=(len(fit.cell_types), genes),
    )
    probabilities = _read_npy(
        files["cell_type_probabilities.npy"],
        dtype=np.dtype("<f8"),
        shape=(len(fit.cell_types),),
    )
    library_parameters = _read_npy(
        files["library_log_parameters.npy"],
        dtype=np.dtype("<f8"),
        shape=(2,),
    )
    for observed, expected, name in (
        (mean_fraction, fit.mean_fraction, "mean_fraction.npy"),
        (dispersion, fit.dispersion_alpha, "dispersion_alpha.npy"),
        (probabilities, fit.cell_type_probabilities, "cell_type_probabilities.npy"),
        (library_parameters, fit.library_log_parameters, "library_log_parameters.npy"),
    ):
        if not np.array_equal(observed, expected):
            raise SimulationContractError(f"native fit array changed: {name}")
    reference = _read_npy(
        files["reference_counts.npy"],
        dtype=np.dtype("<i8"),
        shape=(cells, genes),
    )
    type_index = _read_npy(
        files["cell_type_index.npy"],
        dtype=np.dtype("<i8"),
        shape=(cells,),
    )
    observed_views = {
        name: _read_npy(
            files[f"observed_{name}.npy"],
            dtype=np.dtype("<i8"),
            shape=(cells, genes),
        )
        for name in ("moderate", "severe")
    }
    heldout_views = {
        name: _read_npy(
            files[f"heldout_{name}.npy"],
            dtype=np.dtype("<i8"),
            shape=(cells, genes),
        )
        for name in ("moderate", "severe")
    }
    if (
        bool((reference < 0).any())
        or bool((type_index < 0).any())
        or bool((type_index >= len(fit.cell_types)).any())
        or any(bool((matrix < 0).any()) for matrix in observed_views.values())
        or any(bool((matrix < 0).any()) for matrix in heldout_views.values())
        or any(
            bool((observed_views[name] > reference - heldout_views[name]).any())
            for name in ("moderate", "severe")
        )
    ):
        raise SimulationContractError(
            "semisynthetic native counts contradict the reference/split contract"
        )
    metadata = _validate_run_metadata(files["run_metadata.json"], files)
    expected_reference, expected_type_index = _generate_reference(fit, config)
    if not np.array_equal(reference, expected_reference) or not np.array_equal(
        type_index, expected_type_index
    ):
        raise SimulationContractError(
            "semisynthetic native truth does not match deterministic derivation"
        )
    views = config.get("views")
    if not isinstance(views, list):
        raise SimulationContractError("semisynthetic config views are invalid")
    for view in views:
        if not isinstance(view, Mapping):
            raise SimulationContractError("semisynthetic config view is invalid")
        name = view.get("technical_view")
        if not isinstance(name, str) or name not in {"moderate", "severe"}:
            raise SimulationContractError("semisynthetic config view name is invalid")
        expected_observed, expected_heldout = _split_view(
            expected_reference,
            observed_probability=float(view["observed_probability"]),
            heldout_probability=float(view["heldout_probability"]),
            seed=int(view["measurement_seed_numpy"]),
        )
        if not np.array_equal(
            observed_views[name], expected_observed
        ) or not np.array_equal(heldout_views[name], expected_heldout):
            raise SimulationContractError(
                "semisynthetic native split does not match deterministic derivation"
            )
    return reference, type_index, observed_views, heldout_views, metadata


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
                "semisynthetic output path cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SimulationContractError(
                "semisynthetic output path must not contain symlinks"
            )


def _path_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SimulationContractError(
            f"semisynthetic path identity is unavailable: {path}"
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
    size_by_name = {str(item["path"]): int(item["size_bytes"]) for item in descriptor}
    content_id = canonical_sha256(
        {"schema": "maskimpute-semisynthetic-native-v1", "files": descriptor}
    )[:24]
    _reject_output_symlink_components(parent)
    parent.mkdir(parents=True, exist_ok=True)
    _reject_output_symlink_components(parent)
    native_root = parent / "native"
    if native_root.is_symlink() or (native_root.exists() and not native_root.is_dir()):
        raise SimulationContractError("semisynthetic native output root is invalid")
    native_root.mkdir(mode=0o755, exist_ok=True)
    destination = native_root / f"semisynthetic-{content_id}"
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise SimulationContractError(
                "existing semisynthetic native directory is invalid"
            )
        existing = _validate_stage_entries(destination)
        if _native_descriptor(existing) != descriptor:
            raise SimulationContractError("existing semisynthetic native bytes changed")
        return destination, False, _path_identity(destination)
    publication = Path(
        tempfile.mkdtemp(prefix=".semisynthetic-publish-", dir=native_root)
    )
    publication_identity = _path_identity(publication)
    renamed = False
    try:
        for name in sorted(files):
            payload = _read_regular_bytes(files[name], maximum_bytes=size_by_name[name])
            _write_new_bytes(publication / name, payload)
        copied = _validate_stage_entries(publication)
        if _native_descriptor(copied) != descriptor:
            raise SimulationContractError(
                "semisynthetic native bytes changed while publishing"
            )
        _rename_directory_no_replace(publication, destination)
        renamed = True
        descriptor_fd = os.open(
            native_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(descriptor_fd)
        finally:
            os.close(descriptor_fd)
    except SimulationContractError:
        raise
    except OSError as error:
        if renamed:
            _remove_owned_directory(destination, publication_identity)
        raise SimulationContractError(
            "semisynthetic native outputs could not be published"
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
            raise SimulationContractError(
                "semisynthetic output has no existing ancestor"
            )
        current = current.parent
    try:
        current_metadata = current.lstat()
    except OSError as error:
        raise SimulationContractError(
            "semisynthetic output ancestor cannot be inspected"
        ) from error
    if not stat.S_ISDIR(current_metadata.st_mode) or stat.S_ISLNK(
        current_metadata.st_mode
    ):
        raise SimulationContractError(
            "semisynthetic output ancestor must be a non-symlink directory"
        )
    try:
        inside_repository = absolute.is_relative_to(_REPO_ROOT)
    except ValueError:
        inside_repository = False
    root = _REPO_ROOT.parent if inside_repository else current
    root_metadata = root.lstat()
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_ISLNK(root_metadata.st_mode)
        or root_metadata.st_dev != current_metadata.st_dev
    ):
        raise SimulationContractError(
            "semisynthetic h5ad staging root must be a same-device directory"
        )
    return root


def _stage_h5ad(adata: ad.AnnData, destination: Path) -> tuple[Path, ad.AnnData]:
    _reject_output_symlink_components(destination)
    staging_root = _h5ad_staging_root(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="maskimpute-semisynthetic-h5ad-", suffix=".h5ad", dir=staging_root
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
        temporary.unlink(missing_ok=True)
        if not isinstance(error, (OSError, RuntimeError, TypeError, ValueError)):
            raise
        raise SimulationContractError(
            f"semisynthetic dataset could not be serialized: {destination}"
        ) from error


def _publish_staged_h5ad(
    temporary: Path, destination: Path
) -> tuple[ad.AnnData, tuple[int, int]]:
    _reject_output_symlink_components(destination)
    if os.path.lexists(destination):
        raise SimulationContractError(
            f"semisynthetic adapter refuses to overwrite a result: {destination}"
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
            f"semisynthetic dataset could not be atomically persisted: {destination}"
        ) from error


def _build_dataset(
    request: SimulationRequest,
    observed: np.ndarray,
    heldout: np.ndarray,
    reference: np.ndarray,
    type_index: np.ndarray,
    fit: _SourceFit,
    native_manifest_sha256: str,
    source_receipt: Mapping[str, object],
    config: Mapping[str, object],
    run_metadata: Mapping[str, object],
    pair_request_sha256: str,
) -> ad.AnnData:
    views = config["views"]
    seeds = config["seeds"]
    assert isinstance(views, list)
    assert isinstance(seeds, Mapping)
    biological = seeds["biological"]
    assert isinstance(biological, Mapping)
    view = next(
        item
        for item in views
        if isinstance(item, Mapping)
        and item.get("technical_view") == request.technical_view
    )
    library_sizes = observed.sum(axis=1, dtype=np.int64)
    draw = int(request.biological_id.removeprefix("draw-"))
    obs = pd.DataFrame(
        {
            "dataset_id": [request.dataset_id] * request.cells,
            "mechanism": [request.mechanism] * request.cells,
            "condition": [request.technical_view] * request.cells,
            "biological_id": [request.biological_id] * request.cells,
            "technical_view": [request.technical_view] * request.cells,
            "draw": np.full(request.cells, draw, dtype=np.int64),
            "library_size": library_sizes,
            "group": [fit.cell_types[int(index)] for index in type_index],
        },
        index=[f"cell-{index:04d}" for index in range(1, request.cells + 1)],
    )
    dataset = ad.AnnData(
        X=observed.copy(),
        obs=obs,
        var=pd.DataFrame(index=list(fit.gene_ids)),
        layers={
            "reference_counts": reference.copy(),
            "heldout_counts": heldout.copy(),
        },
    )
    fit_metadata = _fit_metadata(fit)
    source_partition = fit_metadata["source_partition"]
    assert isinstance(source_partition, Mapping)
    dataset.uns.update(
        {
            "truth_kind": "proxy_high_depth",
            "primary_truth_layer": "reference_counts",
            "allowed_covariates": {"obs": [], "var": []},
            "provenance": {
                "source": source_receipt["source_url"],
                "source_sha256": _source_archive_sha256(source_receipt),
                "software": "MaskImpute-semisynthetic",
                "software_version": "1",
                "parameters": {
                    "adapter": config["adapter"],
                    "adapter_schema": "maskimpute-semisynthetic-adapter-v1",
                    "source_partition": {
                        key: source_partition[key]
                        for key in (
                            "donors",
                            "namespace",
                            "partition_rule",
                            "donor_row_counts",
                        )
                    },
                    "gene_selection": fit_metadata["gene_selection"],
                    "cell_type_model": fit_metadata["cell_type_model"],
                    "gamma_poisson_model": fit_metadata["gamma_poisson_model"],
                    "library_size_model": fit_metadata["library_size_model"],
                    "fit_sha256": fit.fit_sha256,
                    "measurement": {
                        key: value
                        for key, value in view.items()
                        if key
                        not in {
                            "technical_view",
                            "measurement_seed_original",
                            "measurement_seed_numpy",
                        }
                    },
                    "metric_availability": {
                        "mse_pre_dropout_zero": "proxy_truth_not_exact",
                        "p_pre_zero_calibration": "proxy_truth_not_exact",
                    },
                    "native_manifest_sha256": native_manifest_sha256,
                    "native_run_metadata": run_metadata,
                    "pair_request_sha256": pair_request_sha256,
                    "simulation": config["simulation"],
                    "source_receipt_json": json.dumps(
                        source_receipt,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "source_receipt_sha256": canonical_sha256(source_receipt),
                },
                "seeds": {
                    "biological": request.biological_seed,
                    "measurement": request.measurement_seed,
                    "numpy_biological": biological["mapped_numpy"],
                    "numpy_measurement": view["measurement_seed_numpy"],
                },
            },
        }
    )
    return dataset


def prepare_source_summary(
    archive_path: Path,
    namespace: str,
    genes: int,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    """Return a canonical, seed-free fit summary for a pinned source partition."""

    fit = _fit_source(
        archive_path,
        namespace,
        genes,
        expected_sha256=expected_sha256,
    )
    return _fit_metadata(fit)


def _revalidate_published_final_claim(claim: FinalManifestClaim | None) -> None:
    """Recheck lifecycle records after unreceipted result publication."""

    from .. import study

    if not isinstance(claim, FinalManifestClaim):
        raise SimulationContractError(
            "published final semisynthetic pair requires its execution claim"
        )
    try:
        repository = claim._repository
        destination = claim.round_dir
        canonical_repository, canonical_destination = study._repository_for_round(
            destination, repository
        )
        if canonical_repository != repository or canonical_destination != destination:
            raise SimulationContractError(
                "final semisynthetic claim changed repository identity"
            )
        with study._round_lock(repository, destination.name) as lock_identity:
            freeze = study._validate_freeze(destination, repository)
            if freeze.get("protocol_sha256") != claim._protocol_sha256:
                raise SimulationContractError(
                    "final semisynthetic claim changed frozen protocol"
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
                    "final semisynthetic execution claim changed during publication"
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
            "final semisynthetic execution is no longer the claimed running round: "
            f"{error}"
        ) from error


def run_semisynthetic_pair(
    requests: Sequence[SimulationRequest],
    protocol: Protocol,
    final_manifest: FinalManifestClaim | None = None,
    *,
    runtime_assets: SimulatorRuntimeAssets | None = None,
) -> tuple[SimulationArtifact, SimulationArtifact]:
    """Generate paired thinnings from one fitted Gamma-Poisson reference proxy."""

    try:
        ordered_requests = tuple(requests)
    except TypeError as error:
        raise SimulationContractError(
            "semisynthetic requests must be a finite pair"
        ) from error
    validate_paired_simulation_requests(ordered_requests, protocol, final_manifest)
    if any(request.mechanism != "semisynthetic" for request in ordered_requests):
        raise SimulationContractError(
            "semisynthetic adapter accepts only semisynthetic requests"
        )
    by_view = {request.technical_view: request for request in ordered_requests}
    if set(by_view) != {"moderate", "severe"} or len(by_view) != 2:
        raise SimulationContractError(
            "semisynthetic adapter requires moderate and severe technical views"
        )
    parents = {request.output_path.parent.absolute() for request in ordered_requests}
    if len(parents) != 1:
        raise SimulationContractError(
            "paired semisynthetic outputs must share one directory"
        )
    output_parent = next(iter(parents))
    _reject_output_symlink_components(output_parent)
    for request in ordered_requests:
        if os.path.lexists(request.output_path):
            raise SimulationContractError(
                f"semisynthetic adapter refuses to overwrite a result: {request.output_path}"
            )

    runtime_assets_sha256: str | None = None
    if runtime_assets is None:
        verify_source = _verify_semisynthetic_source
    else:
        external_root, _r_environment, runtime_assets_sha256 = (
            simulator_runtime_asset_values(runtime_assets)
        )
        verify_source = lambda: (  # noqa: E731
            external_root / "data/baron-pancreas-umi/GSE84133_RAW.tar",
            simulator_runtime_source_receipt(runtime_assets, "baron-pancreas-umi"),
        )
    archive_path, before_source = verify_source()
    stage = Path(tempfile.mkdtemp(prefix="maskimpute-semisynthetic-native-"))
    native_directory: Path | None = None
    native_identity: tuple[int, int] | None = None
    native_created = False
    publication_complete = False
    generation_error: BaseException | None = None
    fit: _SourceFit | None = None
    config: dict[str, object] | None = None
    try:
        try:
            fit = _fit_source(
                archive_path,
                by_view["moderate"].namespace,
                by_view["moderate"].genes,
                expected_sha256=_source_archive_sha256(before_source),
            )
            config = _pair_config(by_view, fit, before_source)
            _generate_native(fit, config, stage)
        except BaseException as error:
            generation_error = error
        try:
            after_path, after_source = verify_source()
        except Exception as error:
            raise SimulationContractError(
                "semisynthetic source was not pristine after generation"
            ) from error
        if after_path != archive_path or canonical_sha256(
            after_source
        ) != canonical_sha256(before_source):
            raise SimulationContractError(
                "semisynthetic source receipt changed during generation"
            )
        if generation_error is not None:
            if isinstance(generation_error, SimulationContractError):
                raise generation_error
            if not isinstance(generation_error, Exception):
                raise generation_error
            raise SimulationContractError(
                "semisynthetic native generation failed"
            ) from generation_error
        assert fit is not None and config is not None

        files = _validate_stage_entries(stage)
        reference, type_index, observed, heldout, run_metadata = _load_native(
            files,
            config,
            fit,
            by_view["moderate"].cells,
            by_view["moderate"].genes,
        )
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
                "adapter_schema": "maskimpute-semisynthetic-native-v1",
                "config_sha256": canonical_sha256(config),
                "fit_sha256": fit.fit_sha256,
                "pair_request_sha256": pair_request_sha256,
                "simulation_request": simulation_scientific_identity(request),
                "source_partition": {
                    "namespace": fit.namespace,
                    "donors": list(fit.donors),
                    "partition_rule": _PARTITION_RULE,
                },
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
                    heldout[name],
                    reference,
                    type_index,
                    fit,
                    manifest.manifest_sha256,
                    before_source,
                    config,
                    run_metadata,
                    pair_request_sha256,
                )
                staged_datasets[name] = _stage_h5ad(dataset, request.output_path)
                _temporary, staged_semantics = staged_datasets[name]
                staged_hash = benchmark_dataset_sha256(staged_semantics)
                staged_hashes[name] = staged_hash
                SimulationArtifact(request, staged_semantics, manifest, staged_hash)
            if not np.array_equal(
                staged_datasets["moderate"][1].layers["reference_counts"],
                staged_datasets["severe"][1].layers["reference_counts"],
            ):
                raise SimulationContractError(
                    "paired semisynthetic datasets do not share reference truth"
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
                        "published semisynthetic manifest differs from staged bytes"
                    )
            artifacts: dict[str, SimulationArtifact] = {}
            for name in ("moderate", "severe"):
                request = by_view[name]
                temporary, _staged = staged_datasets[name]
                persisted, identity = _publish_staged_h5ad(
                    temporary, request.output_path
                )
                published_results.append((request.output_path, identity))
                dataset_sha256 = benchmark_dataset_sha256(persisted)
                if dataset_sha256 != staged_hashes[name]:
                    raise SimulationContractError(
                        "published semisynthetic semantics differ from staging"
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
            raise
        finally:
            for temporary, _dataset in staged_datasets.values():
                temporary.unlink(missing_ok=True)
        return (
            artifacts[ordered_requests[0].technical_view],
            artifacts[ordered_requests[1].technical_view],
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


__all__ = ["prepare_source_summary", "run_semisynthetic_pair"]
