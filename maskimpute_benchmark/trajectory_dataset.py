"""Registered evaluator-only synthetic trajectory authority and dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd

from .protocol import canonical_sha256
from .schema import benchmark_dataset_sha256, validate_benchmark_dataset


FOUR_RECONSTRUCTION_MECHANISMS = frozenset(
    {"symsim", "sergio", "sparsim", "semisynthetic"}
)
REGISTERED_TRAJECTORY_DATASET_ID = "trajectory-exact-latent-01"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_AUTHORITY_FIELDS = frozenset(
    {
        "authority_sha256",
        "binding_sha256",
        "biological_id",
        "cells",
        "condition",
        "draw",
        "expected_dataset_sha256",
        "generator",
        "genes",
        "mechanism",
        "root_cell_id",
        "schema_version",
        "seed",
        "source_id",
        "technical_view",
    }
)
_GENERATOR_FIELDS = frozenset(
    {
        "algorithm",
        "baseline_activity",
        "count_allocation",
        "early_genes",
        "late_genes",
        "library_size_minimum",
        "library_size_span",
        "transient_genes",
        "transient_width",
    }
)


class TrajectoryAuthorityError(ValueError):
    """Raised when the tracked trajectory authority or binding is invalid."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrajectoryAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class TrajectoryAuthority:
    authority_sha256: str
    binding_sha256: str
    biological_id: str
    cells: int
    condition: str
    draw: int
    expected_dataset_sha256: str
    generator: Mapping[str, object]
    genes: int
    mechanism: str
    root_cell_id: str
    schema_version: str
    seed: int
    source_id: str
    technical_view: str


@dataclass(frozen=True, slots=True)
class RegisteredTrajectoryBinding:
    """Truthful file and registered-authority binding for trajectory execution."""

    schema_version: str
    dataset_id: str
    mechanism: str
    biological_id: str
    technical_view: str
    condition: str
    draw: int
    cells: int
    genes: int
    source_id: str
    root_cell_id: str
    seed: int
    dataset_sha256: str
    dataset_file_path: str
    dataset_file_sha256: str
    authority_path: str
    authority_file_sha256: str
    authority_sha256: str
    registered_binding_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "trajectory-execution-dataset-binding-v1":
            raise TrajectoryAuthorityError(
                "trajectory execution binding schema differs"
            )
        for name in (
            "dataset_sha256",
            "dataset_file_sha256",
            "authority_file_sha256",
            "authority_sha256",
            "registered_binding_sha256",
        ):
            _digest(getattr(self, name), f"trajectory binding {name}")
        for name in (
            "dataset_id",
            "mechanism",
            "biological_id",
            "technical_view",
            "condition",
            "source_id",
            "root_cell_id",
        ):
            _text(getattr(self, name), f"trajectory binding {name}")
        for name in ("draw", "cells", "genes", "seed"):
            _positive_integer(getattr(self, name), f"trajectory binding {name}")
        if (
            self.dataset_id != REGISTERED_TRAJECTORY_DATASET_ID
            or self.mechanism != "synthetic_trajectory"
            or self.cells != 2_700
            or self.genes != 120
        ):
            raise TrajectoryAuthorityError(
                "trajectory execution binding identity differs"
            )
        for value, expected, name in (
            (
                self.dataset_file_path,
                "results/trajectory/dataset/evaluator.h5ad",
                "dataset file",
            ),
            (
                self.authority_path,
                "study/trajectory_panel.json",
                "authority file",
            ),
        ):
            relative = PurePosixPath(value)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.as_posix() != expected
            ):
                raise TrajectoryAuthorityError(
                    f"trajectory execution {name} path differs"
                )


@dataclass(frozen=True, slots=True)
class TrajectoryPreparedDataset:
    """Registered authority, persisted evaluator data, and truth-free input."""

    authority: TrajectoryAuthority
    binding: RegisteredTrajectoryBinding
    prepared: Any
    receipt: Mapping[str, object]
    receipt_file_path: str
    receipt_file_sha256: str


def default_trajectory_authority_path() -> Path:
    return Path(__file__).resolve().parents[1] / "study" / "trajectory_panel.json"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrajectoryAuthorityError(f"{name} must be a nonempty string")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise TrajectoryAuthorityError(f"{name} must be a positive integer")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TrajectoryAuthorityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_trajectory_authority(
    path: str | Path | None = None,
) -> TrajectoryAuthority:
    """Load and cryptographically validate the tracked trajectory authority."""

    authority_path = default_trajectory_authority_path() if path is None else Path(path)
    try:
        payload = json.loads(
            authority_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except TrajectoryAuthorityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrajectoryAuthorityError("trajectory authority cannot be read") from error
    if not isinstance(payload, dict) or set(payload) != _AUTHORITY_FIELDS:
        raise TrajectoryAuthorityError("trajectory authority fields differ")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"authority_sha256", "binding_sha256", "expected_dataset_sha256"}
    }
    authority_sha256 = _digest(payload["authority_sha256"], "authority_sha256")
    if authority_sha256 != canonical_sha256(unsigned):
        raise TrajectoryAuthorityError("trajectory authority_sha256 differs")
    expected_dataset_sha256 = _digest(
        payload["expected_dataset_sha256"], "expected_dataset_sha256"
    )
    binding_sha256 = _digest(payload["binding_sha256"], "binding_sha256")
    expected_binding = canonical_sha256(
        {
            "authority_sha256": authority_sha256,
            "expected_dataset_sha256": expected_dataset_sha256,
        }
    )
    if binding_sha256 != expected_binding:
        raise TrajectoryAuthorityError("trajectory binding_sha256 differs")

    generator = payload["generator"]
    if not isinstance(generator, dict) or set(generator) != _GENERATOR_FIELDS:
        raise TrajectoryAuthorityError("trajectory generator fields differ")
    if generator.get("algorithm") != "deterministic-smooth-count-allocation-v1":
        raise TrajectoryAuthorityError("trajectory generator algorithm differs")
    if generator.get("count_allocation") != "largest_remainder_gene_id_tiebreak":
        raise TrajectoryAuthorityError("trajectory count allocation differs")
    for field in ("early_genes", "late_genes", "transient_genes"):
        _positive_integer(generator.get(field), f"generator {field}")
    for field in ("library_size_minimum", "library_size_span"):
        _positive_integer(generator.get(field), f"generator {field}")
    for field in ("baseline_activity", "transient_width"):
        value = generator.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise TrajectoryAuthorityError(f"generator {field} must be positive")

    cells = _positive_integer(payload["cells"], "trajectory cells")
    genes = _positive_integer(payload["genes"], "trajectory genes")
    if cells != 2_700 or genes != 120:
        raise TrajectoryAuthorityError("trajectory panel dimensions differ")
    if (
        sum(
            int(generator[field])
            for field in ("early_genes", "late_genes", "transient_genes")
        )
        != genes
    ):
        raise TrajectoryAuthorityError("trajectory gene programs do not cover genes")
    mechanism = _text(payload["mechanism"], "trajectory mechanism")
    if (
        mechanism != "synthetic_trajectory"
        or mechanism in FOUR_RECONSTRUCTION_MECHANISMS
    ):
        raise TrajectoryAuthorityError("trajectory mechanism is not separate")
    root_cell_id = _text(payload["root_cell_id"], "trajectory root_cell_id")
    if root_cell_id != "cell-000001":
        raise TrajectoryAuthorityError("trajectory root_cell_id differs")
    if payload["schema_version"] != "trajectory-panel-v1":
        raise TrajectoryAuthorityError("trajectory schema version differs")

    return TrajectoryAuthority(
        authority_sha256=authority_sha256,
        binding_sha256=binding_sha256,
        biological_id=_text(payload["biological_id"], "trajectory biological_id"),
        cells=cells,
        condition=_text(payload["condition"], "trajectory condition"),
        draw=_positive_integer(payload["draw"], "trajectory draw"),
        expected_dataset_sha256=expected_dataset_sha256,
        generator=MappingProxyType(dict(generator)),
        genes=genes,
        mechanism=mechanism,
        root_cell_id=root_cell_id,
        schema_version=str(payload["schema_version"]),
        seed=_positive_integer(payload["seed"], "trajectory seed"),
        source_id=_text(payload["source_id"], "trajectory source_id"),
        technical_view=_text(payload["technical_view"], "trajectory technical_view"),
    )


def _trajectory_activity(
    pseudotime: np.ndarray, authority: TrajectoryAuthority
) -> np.ndarray:
    generator = authority.generator
    early = int(generator["early_genes"])
    late = int(generator["late_genes"])
    transient = int(generator["transient_genes"])
    baseline = float(generator["baseline_activity"])
    width = float(generator["transient_width"])
    result = np.empty((authority.cells, authority.genes), dtype=np.float64)

    early_index = np.arange(early, dtype=np.float64)
    early_amplitude = 1.0 + ((early_index * 17.0) % 23.0) / 10.0
    early_decay = 2.0 + ((early_index * 7.0) % 9.0) / 4.0
    result[:, :early] = baseline + early_amplitude[None, :] * np.exp(
        -pseudotime[:, None] * early_decay[None, :]
    )

    late_index = np.arange(late, dtype=np.float64)
    late_amplitude = 1.0 + ((late_index * 19.0) % 29.0) / 10.0
    late_decay = 2.0 + ((late_index * 5.0) % 11.0) / 4.0
    result[:, early : early + late] = baseline + late_amplitude[None, :] * np.exp(
        -(1.0 - pseudotime[:, None]) * late_decay[None, :]
    )

    transient_index = np.arange(transient, dtype=np.float64)
    centers = (transient_index + 0.5) / transient
    transient_amplitude = 1.5 + ((transient_index * 13.0) % 31.0) / 10.0
    result[:, early + late :] = baseline + transient_amplitude[None, :] * np.exp(
        -0.5 * ((pseudotime[:, None] - centers[None, :]) / width) ** 2
    )
    return result


def _allocate_counts(
    activity: np.ndarray, authority: TrajectoryAuthority
) -> np.ndarray:
    generator = authority.generator
    minimum = int(generator["library_size_minimum"])
    span = int(generator["library_size_span"])
    counts = np.empty(activity.shape, dtype=np.int64)
    for cell in range(authority.cells):
        library_size = minimum + ((cell * 104_729 + authority.seed) % span)
        expected = activity[cell] * (library_size / float(np.sum(activity[cell])))
        allocated = np.floor(expected).astype(np.int64)
        remainder = library_size - int(np.sum(allocated, dtype=np.int64))
        order = np.argsort(-(expected - allocated), kind="mergesort")
        allocated[order[:remainder]] += 1
        counts[cell] = allocated
    return counts


def _build_registered_dataset(authority: TrajectoryAuthority) -> ad.AnnData:
    pseudotime = np.linspace(0.0, 1.0, authority.cells, dtype=np.float64)
    counts = _allocate_counts(_trajectory_activity(pseudotime, authority), authority)
    cell_ids = [f"cell-{index:06d}" for index in range(1, authority.cells + 1)]
    gene_ids = [f"gene-{index:04d}" for index in range(1, authority.genes + 1)]
    group = np.where(
        pseudotime < 1.0 / 3.0,
        "early",
        np.where(pseudotime < 2.0 / 3.0, "middle", "late"),
    )
    obs = pd.DataFrame(
        {
            "dataset_id": [REGISTERED_TRAJECTORY_DATASET_ID] * authority.cells,
            "mechanism": [authority.mechanism] * authority.cells,
            "condition": [authority.condition] * authority.cells,
            "biological_id": [authority.biological_id] * authority.cells,
            "technical_view": [authority.technical_view] * authority.cells,
            "draw": np.full(authority.cells, authority.draw, dtype=np.int64),
            "library_size": np.sum(counts, axis=1, dtype=np.int64),
            "group": group,
            "pseudotime": pseudotime,
        },
        index=cell_ids,
    )
    for field in (
        "dataset_id",
        "mechanism",
        "condition",
        "biological_id",
        "technical_view",
        "group",
    ):
        obs[field] = pd.Categorical(obs[field])
    dataset = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=gene_ids),
    )
    dataset.uns.update(
        {
            "truth_kind": "orthogonal_only",
            "provenance": {
                "source": "repository:study/trajectory_panel.json",
                "source_sha256": authority.authority_sha256,
                "software": "maskimpute_benchmark.trajectory_dataset",
                "software_version": "1",
                "parameters": {
                    "generator": dict(authority.generator),
                    "panel_schema": authority.schema_version,
                    "root_cell_id": authority.root_cell_id,
                    "source_id": authority.source_id,
                },
                "seeds": {"count_allocation_seed": authority.seed},
            },
            "normalization": {"input": "raw_umi_counts", "size_factor": "none"},
        }
    )
    validate_benchmark_dataset(dataset)
    return dataset


def generate_registered_trajectory_dataset(
    *, authority: TrajectoryAuthority | None = None
) -> ad.AnnData:
    """Generate and verify the registered 2,700-cell exact-latent panel."""

    selected = load_trajectory_authority() if authority is None else authority
    if not isinstance(selected, TrajectoryAuthority):
        raise TypeError("authority must be a validated TrajectoryAuthority")
    dataset = _build_registered_dataset(selected)
    observed = benchmark_dataset_sha256(dataset)
    if observed != selected.expected_dataset_sha256:
        raise TrajectoryAuthorityError("trajectory expected_dataset_sha256 differs")
    return dataset


__all__ = [
    "FOUR_RECONSTRUCTION_MECHANISMS",
    "REGISTERED_TRAJECTORY_DATASET_ID",
    "RegisteredTrajectoryBinding",
    "TrajectoryAuthority",
    "TrajectoryAuthorityError",
    "TrajectoryPreparedDataset",
    "default_trajectory_authority_path",
    "generate_registered_trajectory_dataset",
    "load_trajectory_authority",
]
