"""Evaluator-only downstream and molecular endpoints.

Method execution is deliberately absent from this module.  A method output
contains only its numeric matrix and stable identifiers; all biological truth
is supplied later through a separate evaluator-owned object.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import sparse


def _stable_ids(values: object, name: str) -> tuple[str, ...]:
    try:
        identifiers = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of strings") from error
    if not identifiers:
        raise ValueError(f"{name} must be nonempty")
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} must be unique")
    return identifiers


def _dense_float_matrix(value: object, name: str) -> np.ndarray:
    if sparse.issparse(value):
        array = value.toarray()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError(f"{name} must be a real numeric matrix")
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class MethodOutput:
    """Truth-free output returned by a method before evaluator access."""

    values: np.ndarray
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = _dense_float_matrix(self.values, "values")
        cell_ids = _stable_ids(self.cell_ids, "cell_ids")
        gene_ids = _stable_ids(self.gene_ids, "gene_ids")
        if values.shape[0] != len(cell_ids):
            raise ValueError("values row count must match cell_ids")
        if values.shape[1] != len(gene_ids):
            raise ValueError("values column count must match gene_ids")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "gene_ids", gene_ids)


@dataclass(frozen=True, slots=True)
class TrajectoryTruth:
    """Genuine evaluator pseudotime with a prespecified orientation root."""

    pseudotime: np.ndarray
    cell_ids: tuple[str, ...]
    root_cell_id: str
    source_id: str

    def __post_init__(self) -> None:
        cell_ids = _stable_ids(self.cell_ids, "trajectory cell_ids")
        pseudotime = np.asarray(self.pseudotime)
        if pseudotime.ndim != 1 or pseudotime.size != len(cell_ids):
            raise ValueError("pseudotime must have one value per trajectory cell")
        if pseudotime.dtype.kind not in {"i", "u", "f"}:
            raise TypeError("pseudotime must be real numeric")
        values = np.array(pseudotime, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("pseudotime must contain only finite values")
        if not isinstance(self.root_cell_id, str) or self.root_cell_id not in cell_ids:
            raise ValueError("trajectory root_cell_id must identify one trajectory cell")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("trajectory source_id must be nonempty")
        minimum = float(np.min(values))
        root_index = cell_ids.index(self.root_cell_id)
        if values[root_index] != minimum or int(np.sum(values == minimum)) != 1:
            raise ValueError("trajectory root must be the unique minimum of pseudotime")
        values.setflags(write=False)
        object.__setattr__(self, "pseudotime", values)
        object.__setattr__(self, "cell_ids", cell_ids)


@dataclass(frozen=True, slots=True)
class EvaluatorTargets:
    """Biological truth retained exclusively on the evaluator side."""

    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    group_labels: tuple[str, ...] | None
    group_labels_reason: str | None
    group_markers: Mapping[str, np.ndarray] | None
    group_markers_reason: str | None
    heldout_counts: np.ndarray | None
    heldout_reason: str | None
    trajectory: TrajectoryTruth | None
    trajectory_reason: str | None

    def __post_init__(self) -> None:
        cell_ids = _stable_ids(self.cell_ids, "evaluator cell_ids")
        gene_ids = _stable_ids(self.gene_ids, "evaluator gene_ids")
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "gene_ids", gene_ids)

        if self.group_labels is None:
            if not self.group_labels_reason:
                raise ValueError("missing group labels require a reason")
            labels = None
        else:
            labels = tuple(self.group_labels)
            if len(labels) != len(cell_ids):
                raise ValueError("group_labels must have one value per evaluator cell")
            if any(not isinstance(value, str) or not value.strip() for value in labels):
                raise ValueError("group_labels must contain nonempty strings")
            if self.group_labels_reason is not None:
                raise ValueError("available group labels cannot have a reason")
        object.__setattr__(self, "group_labels", labels)

        if self.group_markers is None:
            if not self.group_markers_reason:
                raise ValueError("missing group markers require a reason")
        else:
            if labels is None:
                raise ValueError("group markers require group labels")
            if self.group_markers_reason is not None:
                raise ValueError("available group markers cannot have a reason")
            expected_groups = set(labels)
            if set(self.group_markers) != expected_groups:
                raise ValueError("group markers must match the evaluator groups exactly")
            markers: dict[str, np.ndarray] = {}
            for group in sorted(expected_groups):
                mask = np.asarray(self.group_markers[group])
                if mask.ndim != 1 or mask.size != len(gene_ids):
                    raise ValueError("each group marker mask must have one value per gene")
                if mask.dtype.kind != "b":
                    raise TypeError("group marker masks must be boolean")
                copied = np.array(mask, dtype=bool, copy=True)
                copied.setflags(write=False)
                markers[group] = copied
            object.__setattr__(self, "group_markers", MappingProxyType(markers))

        if self.heldout_counts is None:
            if not self.heldout_reason:
                raise ValueError("missing heldout counts require a reason")
        else:
            if self.heldout_reason is not None:
                raise ValueError("available heldout counts cannot have a reason")
            heldout = _dense_float_matrix(self.heldout_counts, "heldout_counts")
            if heldout.shape != (len(cell_ids), len(gene_ids)):
                raise ValueError("heldout_counts shape must match evaluator IDs")
            object.__setattr__(self, "heldout_counts", heldout)

        if self.trajectory is None:
            if not self.trajectory_reason:
                raise ValueError("missing trajectory truth requires a reason")
        else:
            if self.trajectory_reason is not None:
                raise ValueError("available trajectory truth cannot have a reason")
            if set(self.trajectory.cell_ids) != set(cell_ids):
                raise ValueError("trajectory cells must match evaluator cells exactly")


def _marker_column(mechanism: str, group: str) -> str | None:
    if mechanism == "symsim" and group.startswith("pop-"):
        return f"marker_group_{group.removeprefix('pop-')}"
    if mechanism == "sergio" and group.startswith("cell-type-"):
        return f"marker_cell_type_{group.removeprefix('cell-type-')}"
    if mechanism == "sparsim" and group.startswith("chu-"):
        return f"marker_{group.replace('-', '_')}"
    return None


def evaluator_targets_from_dataset(
    dataset: object,
    *,
    trajectory_root_cell_id: str | None = None,
    trajectory_source_id: str | None = None,
) -> EvaluatorTargets:
    """Extract truth only after method execution from a benchmark dataset.

    Marker-column mappings are explicit for the three simulator schemas.  No
    group label is ever promoted to pseudotime; a genuine pseudotime column
    and an independently prespecified root are both required.
    """

    obs = getattr(dataset, "obs", None)
    var = getattr(dataset, "var", None)
    layers = getattr(dataset, "layers", None)
    if obs is None or var is None or layers is None:
        raise TypeError("dataset must expose AnnData-compatible obs, var, and layers")
    cell_ids = _stable_ids(tuple(str(value) for value in dataset.obs_names), "cell_ids")
    gene_ids = _stable_ids(tuple(str(value) for value in dataset.var_names), "gene_ids")

    if "group" not in obs:
        group_labels: tuple[str, ...] | None = None
        group_labels_reason = "group_labels_unavailable"
    else:
        group_labels = tuple(str(value) for value in obs["group"].tolist())
        group_labels_reason = None

    mechanism_values = (
        {str(value) for value in obs["mechanism"].tolist()}
        if "mechanism" in obs
        else set()
    )
    mechanism = next(iter(mechanism_values)) if len(mechanism_values) == 1 else ""
    marker_mapping: dict[str, np.ndarray] | None = None
    marker_reason = "group_specific_marker_truth_unavailable"
    if group_labels is not None:
        expected_columns = {
            group: _marker_column(mechanism, group)
            for group in sorted(set(group_labels))
        }
        if expected_columns and all(
            column is not None and column in var
            for column in expected_columns.values()
        ):
            marker_mapping = {}
            for group, column in expected_columns.items():
                assert column is not None
                mask = np.asarray(var[column].to_numpy())
                if mask.dtype.kind != "b":
                    raise TypeError(f"{column} must be a Boolean marker mask")
                marker_mapping[group] = mask
            marker_reason = None

    if "heldout_counts" in layers:
        heldout = _dense_float_matrix(layers["heldout_counts"], "heldout_counts")
        heldout_reason = None
    else:
        heldout = None
        heldout_reason = "independent_heldout_counts_unavailable"

    trajectory: TrajectoryTruth | None = None
    if "pseudotime" not in obs:
        trajectory_reason = "genuine_pseudotime_not_available_in_simulator_output"
    elif trajectory_root_cell_id is None:
        trajectory_reason = "trajectory_root_not_prespecified"
    else:
        if trajectory_source_id is None:
            raise ValueError("trajectory_source_id is required with a trajectory root")
        trajectory = TrajectoryTruth(
            pseudotime=np.asarray(obs["pseudotime"].to_numpy()),
            cell_ids=cell_ids,
            root_cell_id=trajectory_root_cell_id,
            source_id=trajectory_source_id,
        )
        trajectory_reason = None

    return EvaluatorTargets(
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        group_labels=group_labels,
        group_labels_reason=group_labels_reason,
        group_markers=marker_mapping,
        group_markers_reason=marker_reason,
        heldout_counts=heldout,
        heldout_reason=heldout_reason,
        trajectory=trajectory,
        trajectory_reason=trajectory_reason,
    )
