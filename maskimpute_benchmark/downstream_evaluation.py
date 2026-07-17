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
from scipy import stats


CLUSTERING_SEED = 20_260_716
POSITIVE_DE_ALPHA = 0.05
POSITIVE_DE_FAMILY_ID = "one_vs_rest_all_groups_all_genes"


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


@dataclass(frozen=True, slots=True)
class EndpointRecord:
    """One endpoint from one independent biological draw.

    ``descriptive_n`` reports the number of groups, markers, discoveries,
    genes, or cells entering the within-draw calculation.  It is never an
    independent sample size.
    """

    endpoint: str
    value: float | None
    status: str
    reason: str | None
    direction: str
    independent_unit: str
    independent_n: int
    descriptive_n: int
    descriptive_unit: str
    procedure: str
    family_id: str | None = None
    family_size: int | None = None
    alpha: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint:
            raise ValueError("endpoint must be nonempty")
        if self.status not in {"completed", "unavailable"}:
            raise ValueError("endpoint status must be completed or unavailable")
        if self.direction not in {"lower_is_better", "higher_is_better"}:
            raise ValueError("endpoint direction is invalid")
        if self.independent_unit != "biological_draw" or self.independent_n != 1:
            raise ValueError("each endpoint record must represent one biological draw")
        if type(self.descriptive_n) is not int or self.descriptive_n < 0:
            raise ValueError("descriptive_n must be a nonnegative integer")
        if not isinstance(self.descriptive_unit, str) or not self.descriptive_unit:
            raise ValueError("descriptive_unit must be nonempty")
        if not isinstance(self.procedure, str) or not self.procedure:
            raise ValueError("procedure must be nonempty")
        if self.status == "completed":
            if self.value is None or not np.isfinite(float(self.value)):
                raise ValueError("completed endpoint requires a finite value")
            if self.reason is not None:
                raise ValueError("completed endpoint cannot have a reason")
        elif self.value is not None or not self.reason:
            raise ValueError("unavailable endpoint requires only a reason")
        family_fields = (self.family_id, self.family_size, self.alpha)
        if any(value is not None for value in family_fields):
            if (
                not isinstance(self.family_id, str)
                or not self.family_id
                or type(self.family_size) is not int
                or self.family_size <= 0
                or not isinstance(self.alpha, (int, float))
                or not 0.0 < float(self.alpha) < 1.0
            ):
                raise ValueError("multiple-testing family metadata is incomplete")


def _completed_record(
    endpoint: str,
    value: float,
    *,
    direction: str,
    descriptive_n: int,
    descriptive_unit: str,
    procedure: str,
    family_id: str | None = None,
    family_size: int | None = None,
    alpha: float | None = None,
) -> EndpointRecord:
    return EndpointRecord(
        endpoint=endpoint,
        value=float(value),
        status="completed",
        reason=None,
        direction=direction,
        independent_unit="biological_draw",
        independent_n=1,
        descriptive_n=int(descriptive_n),
        descriptive_unit=descriptive_unit,
        procedure=procedure,
        family_id=family_id,
        family_size=family_size,
        alpha=alpha,
    )


def _unavailable_record(
    endpoint: str,
    reason: str,
    *,
    direction: str,
    descriptive_n: int,
    descriptive_unit: str,
    procedure: str,
    family_id: str | None = None,
    family_size: int | None = None,
    alpha: float | None = None,
) -> EndpointRecord:
    return EndpointRecord(
        endpoint=endpoint,
        value=None,
        status="unavailable",
        reason=reason,
        direction=direction,
        independent_unit="biological_draw",
        independent_n=1,
        descriptive_n=int(descriptive_n),
        descriptive_unit=descriptive_unit,
        procedure=procedure,
        family_id=family_id,
        family_size=family_size,
        alpha=alpha,
    )


def _aligned_evaluator_arrays(
    output: MethodOutput, targets: EvaluatorTargets
) -> tuple[
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
]:
    if set(output.cell_ids) != set(targets.cell_ids):
        raise ValueError("method and evaluator cell IDs must match exactly")
    if set(output.gene_ids) != set(targets.gene_ids):
        raise ValueError("method and evaluator gene IDs must match exactly")
    cell_ids = tuple(sorted(targets.cell_ids))
    gene_ids = tuple(sorted(targets.gene_ids))
    output_cells = {value: index for index, value in enumerate(output.cell_ids)}
    output_genes = {value: index for index, value in enumerate(output.gene_ids)}
    target_cells = {value: index for index, value in enumerate(targets.cell_ids)}
    target_genes = {value: index for index, value in enumerate(targets.gene_ids)}
    output_cell_order = np.asarray([output_cells[value] for value in cell_ids])
    output_gene_order = np.asarray([output_genes[value] for value in gene_ids])
    target_cell_order = np.asarray([target_cells[value] for value in cell_ids])
    target_gene_order = np.asarray([target_genes[value] for value in gene_ids])
    values = output.values[output_cell_order][:, output_gene_order]
    return values, cell_ids, gene_ids, target_cell_order, target_gene_order


def _log_cp10k(values: np.ndarray) -> np.ndarray:
    library_size = np.sum(values, axis=1, dtype=np.float64)
    scale = np.divide(
        10_000.0,
        library_size,
        out=np.zeros_like(library_size),
        where=library_size > 0.0,
    )
    return np.log1p(values * scale[:, None])


def _descending_average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    return ranks


def _one_vs_rest_scores(
    values: np.ndarray, labels: np.ndarray, groups: tuple[str, ...]
) -> np.ndarray:
    scores = np.empty((len(groups), values.shape[1]), dtype=np.float64)
    for group_index, group in enumerate(groups):
        selected = labels == group
        scores[group_index] = np.mean(values[selected], axis=0) - np.mean(
            values[~selected], axis=0
        )
    return scores


def _one_sided_welch_p_values(
    values: np.ndarray, labels: np.ndarray, groups: tuple[str, ...]
) -> np.ndarray:
    result = np.empty((len(groups), values.shape[1]), dtype=np.float64)
    for group_index, group in enumerate(groups):
        selected = labels == group
        inside = values[selected]
        outside = values[~selected]
        n_inside = inside.shape[0]
        n_outside = outside.shape[0]
        mean_difference = np.mean(inside, axis=0) - np.mean(outside, axis=0)
        inside_variance = np.var(inside, axis=0, ddof=1)
        outside_variance = np.var(outside, axis=0, ddof=1)
        inside_term = inside_variance / n_inside
        outside_term = outside_variance / n_outside
        standard_error_squared = inside_term + outside_term
        p_values = np.ones(values.shape[1], dtype=np.float64)
        separated_constants = (standard_error_squared == 0.0) & (
            mean_difference > 0.0
        )
        p_values[separated_constants] = 0.0
        variable = standard_error_squared > 0.0
        if np.any(variable):
            denominator = (
                inside_term[variable] ** 2 / (n_inside - 1)
                + outside_term[variable] ** 2 / (n_outside - 1)
            )
            degrees_of_freedom = np.divide(
                standard_error_squared[variable] ** 2,
                denominator,
                out=np.full(np.sum(variable), np.inf, dtype=np.float64),
                where=denominator > 0.0,
            )
            statistic = mean_difference[variable] / np.sqrt(
                standard_error_squared[variable]
            )
            p_values[variable] = stats.t.sf(statistic, degrees_of_freedom)
        result[group_index] = p_values
    return result


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    flat = np.asarray(p_values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ordered = flat[order]
    adjusted_ordered = ordered * flat.size / np.arange(1, flat.size + 1)
    adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
    adjusted = np.empty(flat.size, dtype=np.float64)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return adjusted.reshape(p_values.shape)


def evaluate_marker_and_de_endpoints(
    output: MethodOutput, targets: EvaluatorTargets
) -> tuple[EndpointRecord, EndpointRecord, EndpointRecord]:
    """Evaluate group-specific marker ranking and positive-control DE.

    Benjamini-Hochberg correction uses one family containing every
    one-vs-rest group-by-gene hypothesis in this biological draw.
    """

    values, _cell_ids, _gene_ids, target_cells, target_genes = (
        _aligned_evaluator_arrays(output, targets)
    )
    marker_procedure = "group_macro_mean_normalized_true_marker_rank_log1p_cp10k"
    de_procedure = "one_sided_welch_log1p_cp10k_global_bh"
    if targets.group_labels is None:
        reason = targets.group_labels_reason or "group_labels_unavailable"
        family_size = None
    else:
        labels = np.asarray(targets.group_labels, dtype=object)[target_cells]
        family_size = len(set(labels.tolist())) * values.shape[1]
        reason = None
    if reason is None and targets.group_markers is None:
        reason = targets.group_markers_reason or "group_specific_marker_truth_unavailable"
    if reason is None and len(set(labels.tolist())) < 2:
        reason = "fewer_than_two_groups"
    if reason is not None:
        family_kwargs = (
            {
                "family_id": POSITIVE_DE_FAMILY_ID,
                "family_size": family_size,
                "alpha": POSITIVE_DE_ALPHA,
            }
            if family_size is not None and family_size > 0
            else {}
        )
        return (
            _unavailable_record(
                "marker_rank_loss",
                reason,
                direction="lower_is_better",
                descriptive_n=0,
                descriptive_unit="truth_markers",
                procedure=marker_procedure,
            ),
            _unavailable_record(
                "positive_de_marker_recall",
                reason,
                direction="higher_is_better",
                descriptive_n=0,
                descriptive_unit="truth_markers",
                procedure=de_procedure,
                **family_kwargs,
            ),
            _unavailable_record(
                "positive_de_false_discovery_rate",
                reason,
                direction="lower_is_better",
                descriptive_n=0,
                descriptive_unit="discoveries",
                procedure=de_procedure,
                **family_kwargs,
            ),
        )

    assert targets.group_markers is not None
    groups = tuple(sorted(set(labels.tolist())))
    marker_truth = np.stack(
        [targets.group_markers[group][target_genes] for group in groups], axis=0
    )
    truth_marker_count = int(np.sum(marker_truth))
    log_values = _log_cp10k(values)
    scores = _one_vs_rest_scores(log_values, labels, groups)

    if values.shape[1] < 2:
        marker_record = _unavailable_record(
            "marker_rank_loss",
            "fewer_than_two_genes",
            direction="lower_is_better",
            descriptive_n=truth_marker_count,
            descriptive_unit="truth_markers",
            procedure=marker_procedure,
        )
    elif any(not np.any(marker_truth[index]) for index in range(len(groups))):
        marker_record = _unavailable_record(
            "marker_rank_loss",
            "group_has_no_truth_markers",
            direction="lower_is_better",
            descriptive_n=truth_marker_count,
            descriptive_unit="truth_markers",
            procedure=marker_procedure,
        )
    else:
        group_losses = []
        for group_index in range(len(groups)):
            ranks = _descending_average_ranks(scores[group_index])
            group_losses.append(
                float(
                    np.mean(
                        (ranks[marker_truth[group_index]] - 1.0)
                        / (values.shape[1] - 1.0)
                    )
                )
            )
        marker_record = _completed_record(
            "marker_rank_loss",
            float(np.mean(group_losses)),
            direction="lower_is_better",
            descriptive_n=truth_marker_count,
            descriptive_unit="truth_markers",
            procedure=marker_procedure,
        )

    family_kwargs = {
        "family_id": POSITIVE_DE_FAMILY_ID,
        "family_size": len(groups) * values.shape[1],
        "alpha": POSITIVE_DE_ALPHA,
    }
    if any(int(np.sum(labels == group)) < 2 for group in groups) or any(
        int(np.sum(labels != group)) < 2 for group in groups
    ):
        de_reason = "fewer_than_two_cells_in_one_vs_rest_arm"
        recall_record = _unavailable_record(
            "positive_de_marker_recall",
            de_reason,
            direction="higher_is_better",
            descriptive_n=truth_marker_count,
            descriptive_unit="truth_markers",
            procedure=de_procedure,
            **family_kwargs,
        )
        fdr_record = _unavailable_record(
            "positive_de_false_discovery_rate",
            de_reason,
            direction="lower_is_better",
            descriptive_n=0,
            descriptive_unit="discoveries",
            procedure=de_procedure,
            **family_kwargs,
        )
    elif truth_marker_count == 0:
        recall_record = _unavailable_record(
            "positive_de_marker_recall",
            "no_truth_markers",
            direction="higher_is_better",
            descriptive_n=0,
            descriptive_unit="truth_markers",
            procedure=de_procedure,
            **family_kwargs,
        )
        fdr_record = _unavailable_record(
            "positive_de_false_discovery_rate",
            "no_truth_markers",
            direction="lower_is_better",
            descriptive_n=0,
            descriptive_unit="discoveries",
            procedure=de_procedure,
            **family_kwargs,
        )
    else:
        adjusted = _benjamini_hochberg(
            _one_sided_welch_p_values(log_values, labels, groups)
        )
        discoveries = adjusted <= POSITIVE_DE_ALPHA
        discovery_count = int(np.sum(discoveries))
        true_positive_count = int(np.sum(discoveries & marker_truth))
        false_positive_count = discovery_count - true_positive_count
        recall_record = _completed_record(
            "positive_de_marker_recall",
            true_positive_count / truth_marker_count,
            direction="higher_is_better",
            descriptive_n=truth_marker_count,
            descriptive_unit="truth_markers",
            procedure=de_procedure,
            **family_kwargs,
        )
        fdr_record = _completed_record(
            "positive_de_false_discovery_rate",
            false_positive_count / discovery_count if discovery_count else 0.0,
            direction="lower_is_better",
            descriptive_n=discovery_count,
            descriptive_unit="discoveries",
            procedure=de_procedure,
            **family_kwargs,
        )
    return marker_record, recall_record, fdr_record


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
