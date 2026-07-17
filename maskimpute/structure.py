"""Observed-only structure authority and differentiable v29 penalties."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real

import numpy as np
import torch

from maskimpute.train import normalize_observed_counts, validate_observed_counts


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class StructurePenaltyConfig:
    """Prespecified capacity-neutral v29 structure objective."""

    variable_gene_count: int = 200
    neighborhood_k: int = 15
    covariance_penalty_weight: float = 0.1
    neighborhood_penalty_weight: float = 0.1
    variance_floor: float = 1e-8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variable_gene_count",
            _positive_integer(self.variable_gene_count, "variable_gene_count"),
        )
        object.__setattr__(
            self,
            "neighborhood_k",
            _positive_integer(self.neighborhood_k, "neighborhood_k"),
        )
        for name in (
            "covariance_penalty_weight",
            "neighborhood_penalty_weight",
            "variance_floor",
        ):
            value = _finite_nonnegative(getattr(self, name), name)
            if name == "variance_floor" and value == 0:
                raise ValueError("variance_floor must be positive")
            object.__setattr__(self, name, value)
        if (
            self.covariance_penalty_weight == 0
            and self.neighborhood_penalty_weight == 0
        ):
            raise ValueError("at least one structure penalty weight must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "variable_gene_count": self.variable_gene_count,
            "neighborhood_k": self.neighborhood_k,
            "covariance_penalty_weight": self.covariance_penalty_weight,
            "neighborhood_penalty_weight": self.neighborhood_penalty_weight,
            "variance_floor": self.variance_floor,
        }


@dataclass(frozen=True, slots=True)
class StructureAuthority:
    """Frozen variable-gene and observed-neighborhood identities."""

    variable_gene_indices: tuple[int, ...]
    neighbor_indices: np.ndarray
    variable_gene_sha256: str
    neighborhood_sha256: str

    def __post_init__(self) -> None:
        genes = tuple(self.variable_gene_indices)
        neighbors = np.asarray(self.neighbor_indices)
        if (
            not genes
            or len(genes) != len(set(genes))
            or any(type(value) is not int or value < 0 for value in genes)
        ):
            raise ValueError("variable gene indices are invalid")
        if (
            neighbors.ndim != 2
            or neighbors.dtype.kind not in "iu"
            or neighbors.shape[0] < 2
            or neighbors.shape[1] < 1
            or np.any(neighbors < 0)
            or np.any(neighbors >= neighbors.shape[0])
        ):
            raise ValueError("neighbor indices are invalid")
        if any(row in neighbors[row] for row in range(neighbors.shape[0])):
            raise ValueError("a cell cannot be its own observed neighbor")
        frozen = np.asarray(neighbors, dtype=np.int64, order="C").copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "variable_gene_indices", genes)
        object.__setattr__(self, "neighbor_indices", frozen)
        for name in ("variable_gene_sha256", "neighborhood_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} is invalid")
        expected_gene_sha256 = _array_sha256(
            b"maskimpute-v29-variable-genes-v1\0",
            np.asarray(genes, dtype=np.int64),
        )
        if self.variable_gene_sha256 != expected_gene_sha256:
            raise ValueError("variable gene checksum differs from its indices")
        expected_neighborhood_sha256 = _array_sha256(
            b"maskimpute-v29-observed-neighbors-v1\0",
            frozen,
        )
        if self.neighborhood_sha256 != expected_neighborhood_sha256:
            raise ValueError("neighborhood checksum differs from its indices")


def _array_sha256(domain: bytes, value: np.ndarray) -> str:
    canonical = np.asarray(value, dtype="<i8", order="C")
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def build_structure_authority(
    observed_counts: object,
    config: StructurePenaltyConfig = StructurePenaltyConfig(),
) -> StructureAuthority:
    """Derive variable genes and neighbors solely from the observed input."""

    if type(config) is not StructurePenaltyConfig:
        raise TypeError("config must be an exact StructurePenaltyConfig")
    counts = validate_observed_counts(observed_counts)
    n_cells, n_genes = counts.shape
    if config.variable_gene_count > n_genes:
        raise ValueError("variable_gene_count exceeds the observed gene count")
    if config.neighborhood_k >= n_cells:
        raise ValueError("neighborhood_k must be smaller than the observed cell count")
    normalized, _libraries = normalize_observed_counts(counts, target=10_000.0)
    variances = np.var(normalized, axis=0, ddof=0)
    gene_order = np.lexsort((np.arange(n_genes, dtype=np.int64), -variances))
    selected = tuple(
        int(value) for value in gene_order[: config.variable_gene_count]
    )
    panel = normalized[:, selected]
    squared_norm = np.sum(panel * panel, axis=1, keepdims=True)
    squared_distances = squared_norm + squared_norm.T - 2.0 * (panel @ panel.T)
    squared_distances = np.maximum(squared_distances, 0.0)
    np.fill_diagonal(squared_distances, np.inf)
    neighbors = np.argsort(squared_distances, axis=1, kind="stable")[
        :, : config.neighborhood_k
    ]
    selected_array = np.asarray(selected, dtype=np.int64)
    return StructureAuthority(
        variable_gene_indices=selected,
        neighbor_indices=neighbors,
        variable_gene_sha256=_array_sha256(
            b"maskimpute-v29-variable-genes-v1\0", selected_array
        ),
        neighborhood_sha256=_array_sha256(
            b"maskimpute-v29-observed-neighbors-v1\0", neighbors
        ),
    )


def structure_preservation_loss(
    prediction: torch.Tensor,
    observed_target: torch.Tensor,
    global_rows: object,
    authority: StructureAuthority,
    config: StructurePenaltyConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match observed covariance and neighbor geometry within one minibatch."""

    if not isinstance(prediction, torch.Tensor) or not isinstance(
        observed_target, torch.Tensor
    ):
        raise TypeError("prediction and observed_target must be torch tensors")
    if (
        prediction.ndim != 2
        or observed_target.shape != prediction.shape
        or prediction.device != observed_target.device
    ):
        raise ValueError("prediction and target must be aligned matrices")
    if type(authority) is not StructureAuthority:
        raise TypeError("authority must be an exact StructureAuthority")
    if type(config) is not StructurePenaltyConfig:
        raise TypeError("config must be an exact StructurePenaltyConfig")
    rows = np.asarray(global_rows)
    if (
        rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or rows.shape[0] != prediction.shape[0]
        or len(set(int(value) for value in rows)) != len(rows)
        or np.any(rows < 0)
        or np.any(rows >= authority.neighbor_indices.shape[0])
    ):
        raise ValueError("global_rows must identify distinct authority rows")
    genes = torch.as_tensor(
        authority.variable_gene_indices,
        dtype=torch.long,
        device=prediction.device,
    )
    predicted_panel = prediction.index_select(1, genes)
    target_panel = observed_target.index_select(1, genes).to(prediction.dtype)
    if prediction.shape[0] < 2:
        covariance_loss = prediction.sum() * 0.0
    else:
        predicted_centered = predicted_panel - predicted_panel.mean(dim=0)
        target_centered = target_panel - target_panel.mean(dim=0)
        denominator = float(prediction.shape[0] - 1)
        predicted_covariance = predicted_centered.T @ predicted_centered / denominator
        target_covariance = target_centered.T @ target_centered / denominator
        covariance_loss = torch.mean(
            (predicted_covariance - target_covariance) ** 2
        )
    local_by_global = {int(global_row): local for local, global_row in enumerate(rows)}
    source_indices: list[int] = []
    neighbor_indices: list[int] = []
    for local, global_row in enumerate(rows):
        for neighbor in authority.neighbor_indices[int(global_row)]:
            neighbor_local = local_by_global.get(int(neighbor))
            if neighbor_local is not None:
                source_indices.append(local)
                neighbor_indices.append(neighbor_local)
    if not source_indices:
        neighborhood_loss = prediction.sum() * 0.0
    else:
        source = torch.as_tensor(
            source_indices, dtype=torch.long, device=prediction.device
        )
        neighbor = torch.as_tensor(
            neighbor_indices, dtype=torch.long, device=prediction.device
        )
        predicted_distance = torch.mean(
            (predicted_panel[source] - predicted_panel[neighbor]) ** 2,
            dim=1,
        )
        target_distance = torch.mean(
            (target_panel[source] - target_panel[neighbor]) ** 2,
            dim=1,
        )
        scale = torch.clamp(target_distance.abs(), min=config.variance_floor)
        neighborhood_loss = torch.mean(
            ((predicted_distance - target_distance) / scale) ** 2
        )
    total = (
        config.covariance_penalty_weight * covariance_loss
        + config.neighborhood_penalty_weight * neighborhood_loss
    )
    if not torch.isfinite(total):
        raise FloatingPointError("v29 structure preservation loss is nonfinite")
    return total, {
        "covariance": float(covariance_loss.detach().cpu()),
        "neighborhood": float(neighborhood_loss.detach().cpu()),
    }


__all__ = [
    "StructureAuthority",
    "StructurePenaltyConfig",
    "build_structure_authority",
    "structure_preservation_loss",
]
