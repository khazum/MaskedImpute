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
CLUSTERING_N_INIT = 20
CLUSTERING_MAX_COMPONENTS = 30
CLUSTERING_MIN_K = 2
CLUSTERING_MAX_K = 10
POSITIVE_DE_ALPHA = 0.05
POSITIVE_DE_FAMILY_ID = "one_vs_rest_all_groups_all_genes"
TRAJECTORY_MAX_NEIGHBORS = 15
TRAJECTORY_MAX_COMPONENTS = 30
TRAJECTORY_DIFFUSION_MODES = 15
TRAJECTORY_DISTANCE_BLOCK_SIZE = 128

DOWNSTREAM_ENDPOINT_NAMES = (
    "marker_rank_loss",
    "clustering_ari_loss",
    "clustering_nmi_loss",
    "positive_de_marker_recall",
    "positive_de_false_discovery_rate",
    "heldout_gene_profile_rank_loss",
    "heldout_cell_profile_rank_loss",
    "trajectory_pseudotime_rank_loss",
)

_ENDPOINT_CONTRACT = MappingProxyType(
    {
        "marker_rank_loss": ("lower_is_better", "truth_markers"),
        "clustering_ari_loss": ("lower_is_better", "cells"),
        "clustering_nmi_loss": ("lower_is_better", "cells"),
        "positive_de_marker_recall": ("higher_is_better", "truth_markers"),
        "positive_de_false_discovery_rate": (
            "lower_is_better",
            "discoveries",
        ),
        "heldout_gene_profile_rank_loss": (
            "lower_is_better",
            "heldout_variable_genes",
        ),
        "heldout_cell_profile_rank_loss": (
            "lower_is_better",
            "heldout_variable_cells",
        ),
        "trajectory_pseudotime_rank_loss": (
            "lower_is_better",
            "trajectory_cells",
        ),
    }
)

ENDPOINT_REASON_CODES = frozenset(
    {
        "constant_diffusion_pseudotime",
        "constant_method_representation",
        "deterministic_kmeans_failed",
        "fewer_distinct_method_profiles_than_groups",
        "fewer_than_three_trajectory_cells",
        "fewer_than_two_cells_in_group",
        "fewer_than_two_cells_in_one_vs_rest_arm",
        "fewer_than_two_distinct_method_profiles",
        "fewer_than_two_genes",
        "fewer_than_two_groups",
        "genuine_pseudotime_not_available_in_simulator_output",
        "group_has_no_truth_markers",
        "group_labels_unavailable",
        "group_specific_marker_truth_unavailable",
        "heldout_has_no_variable_cell_profiles",
        "heldout_has_no_variable_gene_profiles",
        "independent_heldout_counts_unavailable",
        "no_truth_markers",
        "nonfinite_diffusion_pseudotime",
        "numeric_evaluation_failed",
        "trajectory_affinity_scale_unavailable",
        "trajectory_diffusion_eigensolver_failed",
        "trajectory_diffusion_modes_unavailable",
        "trajectory_graph_disconnected",
        "trajectory_graph_has_zero_degree",
        "trajectory_root_not_prespecified",
    }
)


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
    """Truth-free method matrix on the evaluator's common output scale.

    Production records persist values as ``log2(CP10k + 1)``.  This object
    intentionally carries no labels, markers, heldout split, or trajectory
    truth and downstream procedures never renormalize the persisted values.
    """

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
            raise ValueError(
                "trajectory root_cell_id must identify one trajectory cell"
            )
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
                raise ValueError(
                    "group markers must match the evaluator groups exactly"
                )
            markers: dict[str, np.ndarray] = {}
            for group in sorted(expected_groups):
                mask = np.asarray(self.group_markers[group])
                if mask.ndim != 1 or mask.size != len(gene_ids):
                    raise ValueError(
                        "each group marker mask must have one value per gene"
                    )
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
        if self.endpoint not in _ENDPOINT_CONTRACT:
            raise ValueError("endpoint is not in the fixed downstream schema")
        if self.status not in {"completed", "unavailable"}:
            raise ValueError("endpoint status must be completed or unavailable")
        expected_direction, expected_descriptive_unit = _ENDPOINT_CONTRACT[
            self.endpoint
        ]
        if self.direction != expected_direction:
            raise ValueError("endpoint direction contradicts the fixed schema")
        if self.independent_unit != "biological_draw" or self.independent_n != 1:
            raise ValueError("each endpoint record must represent one biological draw")
        if type(self.descriptive_n) is not int or self.descriptive_n < 0:
            raise ValueError("descriptive_n must be a nonnegative integer")
        if self.descriptive_unit != expected_descriptive_unit:
            raise ValueError("endpoint descriptive unit contradicts the fixed schema")
        if not isinstance(self.procedure, str) or not self.procedure:
            raise ValueError("procedure must be nonempty")
        if self.status == "completed":
            if self.value is None or not np.isfinite(float(self.value)):
                raise ValueError("completed endpoint requires a finite value")
            if self.reason is not None:
                raise ValueError("completed endpoint cannot have a reason")
        elif self.value is not None or self.reason not in ENDPOINT_REASON_CODES:
            raise ValueError("unavailable endpoint requires a fixed reason code")
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


def _counts_to_log2_cp10k(values: np.ndarray) -> np.ndarray:
    library_size = np.sum(values, axis=1, dtype=np.float64)
    scale = np.divide(
        10_000.0,
        library_size,
        out=np.zeros_like(library_size),
        where=library_size > 0.0,
    )
    return np.log2(values * scale[:, None] + 1.0)


@dataclass(frozen=True, slots=True)
class _EvaluationWorkspace:
    """Canonical alignment and one reusable truth-free PCA representation."""

    values: np.ndarray
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    target_cell_order: np.ndarray
    target_gene_order: np.ndarray
    representation: np.ndarray | None
    representation_reason: str | None


def _pca_representation(values: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    centered = np.array(values, dtype=np.float64, copy=True)
    centered -= np.mean(centered, axis=0)
    left, singular, _right = np.linalg.svd(centered, full_matrices=False)
    if not singular.size:
        return None, "constant_method_representation"
    input_scale = max(float(singular[0]), 1.0)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * input_scale
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        return None, "constant_method_representation"
    components = min(max(CLUSTERING_MAX_COMPONENTS, TRAJECTORY_MAX_COMPONENTS), rank)
    representation = left[:, :components] * singular[:components]
    representation.setflags(write=False)
    return representation, None


def _evaluation_workspace(
    output: MethodOutput,
    targets: EvaluatorTargets,
    *,
    include_representation: bool,
) -> _EvaluationWorkspace:
    values, cell_ids, gene_ids, target_cells, target_genes = _aligned_evaluator_arrays(
        output, targets
    )
    representation: np.ndarray | None = None
    representation_reason: str | None = None
    if include_representation:
        representation, representation_reason = _pca_representation(values)
    return _EvaluationWorkspace(
        values=values,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        target_cell_order=target_cells,
        target_gene_order=target_genes,
        representation=representation,
        representation_reason=representation_reason,
    )


def _truth_free_workspace(output: MethodOutput) -> _EvaluationWorkspace:
    cell_ids = tuple(sorted(output.cell_ids))
    gene_ids = tuple(sorted(output.gene_ids))
    cells = {value: index for index, value in enumerate(output.cell_ids)}
    genes = {value: index for index, value in enumerate(output.gene_ids)}
    cell_order = np.asarray([cells[value] for value in cell_ids])
    gene_order = np.asarray([genes[value] for value in gene_ids])
    values = output.values[cell_order][:, gene_order]
    representation, reason = _pca_representation(values)
    return _EvaluationWorkspace(
        values=values,
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        target_cell_order=np.arange(len(cell_ids)),
        target_gene_order=np.arange(len(gene_ids)),
        representation=representation,
        representation_reason=reason,
    )


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
        separated_constants = (standard_error_squared == 0.0) & (mean_difference > 0.0)
        p_values[separated_constants] = 0.0
        variable = standard_error_squared > 0.0
        if np.any(variable):
            denominator = inside_term[variable] ** 2 / (n_inside - 1) + outside_term[
                variable
            ] ** 2 / (n_outside - 1)
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
    output: MethodOutput,
    targets: EvaluatorTargets,
    *,
    _workspace: _EvaluationWorkspace | None = None,
) -> tuple[EndpointRecord, EndpointRecord, EndpointRecord]:
    """Evaluate group-specific marker ranking and positive-control DE.

    Benjamini-Hochberg correction uses one family containing every
    one-vs-rest group-by-gene hypothesis in this biological draw.
    """

    workspace = _workspace or _evaluation_workspace(
        output, targets, include_representation=False
    )
    values = workspace.values
    target_cells = workspace.target_cell_order
    target_genes = workspace.target_gene_order
    marker_procedure = "group_macro_mean_normalized_true_marker_rank_log2_cp10k_plus_1"
    de_procedure = "one_sided_welch_log2_cp10k_plus_1_global_bh"
    if targets.group_labels is None:
        reason = targets.group_labels_reason or "group_labels_unavailable"
        family_size = None
    else:
        labels = np.asarray(targets.group_labels, dtype=object)[target_cells]
        family_size = len(set(labels.tolist())) * values.shape[1]
        reason = None
    if reason is None and targets.group_markers is None:
        reason = (
            targets.group_markers_reason or "group_specific_marker_truth_unavailable"
        )
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
    scores = _one_vs_rest_scores(values, labels, groups)

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
            _one_sided_welch_p_values(values, labels, groups)
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


def _squared_euclidean(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    differences = values[:, None, :] - centers[None, :, :]
    return np.sum(differences * differences, axis=2)


def _kmeans_once(
    values: np.ndarray, n_clusters: int, random: np.random.Generator
) -> tuple[np.ndarray, float] | None:
    n_cells = values.shape[0]
    centers = [values[int(random.integers(0, n_cells))].copy()]
    for _ in range(1, n_clusters):
        distances = np.min(_squared_euclidean(values, np.asarray(centers)), axis=1)
        total = float(np.sum(distances))
        if total <= 0.0:
            return None
        centers.append(values[int(random.choice(n_cells, p=distances / total))].copy())
    center_matrix = np.asarray(centers)
    previous: np.ndarray | None = None
    for _ in range(300):
        distances = _squared_euclidean(values, center_matrix)
        labels = np.argmin(distances, axis=1)
        counts = np.bincount(labels, minlength=n_clusters)
        if np.any(counts == 0):
            occupied_distance = distances[np.arange(n_cells), labels]
            candidates = np.argsort(-occupied_distance, kind="mergesort")
            used: set[int] = set()
            for empty in np.flatnonzero(counts == 0):
                replacement = next(
                    int(index) for index in candidates if int(index) not in used
                )
                used.add(replacement)
                center_matrix[empty] = values[replacement]
            continue
        new_centers = np.vstack(
            [np.mean(values[labels == index], axis=0) for index in range(n_clusters)]
        )
        if previous is not None and np.array_equal(labels, previous):
            center_matrix = new_centers
            break
        previous = labels.copy()
        center_matrix = new_centers
    final_distances = _squared_euclidean(values, center_matrix)
    final_labels = np.argmin(final_distances, axis=1)
    if np.any(np.bincount(final_labels, minlength=n_clusters) == 0):
        return None
    inertia = float(np.sum(final_distances[np.arange(n_cells), final_labels]))
    return final_labels, inertia


def _canonical_cluster_labels(labels: np.ndarray) -> tuple[int, ...]:
    mapping: dict[int, int] = {}
    result: list[int] = []
    for value in labels.tolist():
        integer = int(value)
        if integer not in mapping:
            mapping[integer] = len(mapping)
        result.append(mapping[integer])
    return tuple(result)


def _deterministic_kmeans(values: np.ndarray, n_clusters: int) -> np.ndarray | None:
    master = np.random.default_rng(CLUSTERING_SEED)
    best_labels: np.ndarray | None = None
    best_inertia = np.inf
    best_canonical: tuple[int, ...] | None = None
    for _ in range(CLUSTERING_N_INIT):
        run_seed = int(master.integers(0, 2**63, dtype=np.int64))
        result = _kmeans_once(values, n_clusters, np.random.default_rng(run_seed))
        if result is None:
            continue
        labels, inertia = result
        canonical = _canonical_cluster_labels(labels)
        if inertia < best_inertia or (
            inertia == best_inertia
            and (best_canonical is None or canonical < best_canonical)
        ):
            best_labels = labels
            best_inertia = inertia
            best_canonical = canonical
    return best_labels


def _davies_bouldin_index(
    values: np.ndarray, labels: np.ndarray, n_clusters: int
) -> float:
    centers = np.vstack(
        [np.mean(values[labels == index], axis=0) for index in range(n_clusters)]
    )
    point_distances = np.sqrt(
        np.maximum(
            _squared_euclidean(values, centers)[np.arange(values.shape[0]), labels],
            0.0,
        )
    )
    scatter = np.asarray(
        [np.mean(point_distances[labels == index]) for index in range(n_clusters)],
        dtype=np.float64,
    )
    center_distances = np.sqrt(np.maximum(_squared_euclidean(centers, centers), 0.0))
    np.fill_diagonal(center_distances, np.inf)
    ratios = np.divide(
        scatter[:, None] + scatter[None, :],
        center_distances,
        out=np.full_like(center_distances, np.inf),
        where=center_distances > 0.0,
    )
    np.fill_diagonal(ratios, -np.inf)
    return float(np.mean(np.max(ratios, axis=1)))


@dataclass(frozen=True, slots=True)
class ClusteringPartition:
    """Auditable truth-free cluster fit in canonical cell-ID order."""

    cell_ids: tuple[str, ...]
    assignments: tuple[int, ...]
    n_clusters: int
    model_selection: str
    criterion_value: float

    def __post_init__(self) -> None:
        if len(self.cell_ids) != len(self.assignments):
            raise ValueError("clustering assignments must match cell IDs")
        if self.n_clusters < CLUSTERING_MIN_K:
            raise ValueError("clustering partition requires at least two clusters")
        if len(set(self.assignments)) != self.n_clusters:
            raise ValueError("clustering partition has the wrong cluster count")
        if self.model_selection != "minimum_davies_bouldin_fixed_k_grid_2_to_10":
            raise ValueError("clustering model-selection rule is not prespecified")
        if not np.isfinite(self.criterion_value) or self.criterion_value < 0.0:
            raise ValueError("clustering criterion must be finite and nonnegative")


def _select_clustering_partition(
    representation: np.ndarray, cell_ids: tuple[str, ...]
) -> tuple[ClusteringPartition | None, str | None]:
    distinct_profiles = int(np.unique(representation, axis=0).shape[0])
    maximum_k = min(
        CLUSTERING_MAX_K,
        representation.shape[0] // 2,
        distinct_profiles,
    )
    if maximum_k < CLUSTERING_MIN_K:
        return None, "fewer_than_two_distinct_method_profiles"

    best: ClusteringPartition | None = None
    for n_clusters in range(CLUSTERING_MIN_K, maximum_k + 1):
        labels = _deterministic_kmeans(representation, n_clusters)
        if labels is None:
            continue
        counts = np.bincount(labels, minlength=n_clusters)
        if np.any(counts < 2):
            continue
        criterion = _davies_bouldin_index(representation, labels, n_clusters)
        if not np.isfinite(criterion):
            continue
        canonical = _canonical_cluster_labels(labels)
        candidate = ClusteringPartition(
            cell_ids=cell_ids,
            assignments=canonical,
            n_clusters=n_clusters,
            model_selection="minimum_davies_bouldin_fixed_k_grid_2_to_10",
            criterion_value=criterion,
        )
        if best is None or (candidate.criterion_value, candidate.n_clusters) < (
            best.criterion_value,
            best.n_clusters,
        ):
            best = candidate
    if best is None:
        return None, "deterministic_kmeans_failed"
    return best, None


def infer_clustering_partition(output: MethodOutput) -> ClusteringPartition:
    """Fit clusters without accepting or consulting evaluator truth.

    Candidate ``k`` values are the fixed grid 2--10, restricted only by the
    number of cells and distinct method profiles.  Davies--Bouldin index is
    minimized with smaller ``k`` as the deterministic tie-break.
    """

    workspace = _truth_free_workspace(output)
    if workspace.representation is None:
        raise ValueError(
            workspace.representation_reason or "constant_method_representation"
        )
    partition, reason = _select_clustering_partition(
        workspace.representation, workspace.cell_ids
    )
    if partition is None:
        raise ValueError(reason or "deterministic_kmeans_failed")
    return partition


def _contingency(
    truth: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_levels = {value: index for index, value in enumerate(sorted(set(truth)))}
    predicted_levels = {
        int(value): index for index, value in enumerate(sorted(set(predicted.tolist())))
    }
    table = np.zeros((len(truth_levels), len(predicted_levels)), dtype=np.int64)
    for truth_value, predicted_value in zip(truth, predicted, strict=True):
        table[
            truth_levels[str(truth_value)], predicted_levels[int(predicted_value)]
        ] += 1
    return table, np.sum(table, axis=1), np.sum(table, axis=0)


def _adjusted_rand_index(truth: np.ndarray, predicted: np.ndarray) -> float:
    table, truth_counts, predicted_counts = _contingency(truth, predicted)

    def pair(value: np.ndarray | float) -> np.ndarray | float:
        return value * (value - 1.0) / 2.0

    observed = float(np.sum(pair(table.astype(np.float64))))
    truth_pairs = float(np.sum(pair(truth_counts.astype(np.float64))))
    predicted_pairs = float(np.sum(pair(predicted_counts.astype(np.float64))))
    total_pairs = pair(float(truth.size))
    expected = truth_pairs * predicted_pairs / total_pairs
    maximum = 0.5 * (truth_pairs + predicted_pairs)
    denominator = maximum - expected
    return (
        1.0
        if denominator == 0.0 and observed == maximum
        else ((observed - expected) / denominator)
    )


def _normalized_mutual_information(truth: np.ndarray, predicted: np.ndarray) -> float:
    table, truth_counts, predicted_counts = _contingency(truth, predicted)
    total = float(truth.size)
    joint = table.astype(np.float64) / total
    truth_probability = truth_counts.astype(np.float64) / total
    predicted_probability = predicted_counts.astype(np.float64) / total
    mutual_information = 0.0
    for truth_index, predicted_index in zip(*np.nonzero(table), strict=True):
        probability = joint[truth_index, predicted_index]
        mutual_information += probability * np.log(
            probability
            / (truth_probability[truth_index] * predicted_probability[predicted_index])
        )
    truth_entropy = float(-np.sum(truth_probability * np.log(truth_probability)))
    predicted_entropy = float(
        -np.sum(predicted_probability * np.log(predicted_probability))
    )
    denominator = truth_entropy + predicted_entropy
    return 1.0 if denominator == 0.0 else 2.0 * mutual_information / denominator


def evaluate_clustering_endpoints(
    output: MethodOutput,
    targets: EvaluatorTargets,
    *,
    _workspace: _EvaluationWorkspace | None = None,
) -> tuple[EndpointRecord, EndpointRecord]:
    """Return deterministic clustering recovery as one minus ARI and NMI."""

    workspace = _workspace or _evaluation_workspace(
        output, targets, include_representation=True
    )
    values = workspace.values
    target_cells = workspace.target_cell_order
    procedure = (
        "log2_cp10k_plus_1_full_svd_pca_kmeans_fixed_k_grid=2:10_"
        "minimum_davies_bouldin_"
        f"seed={CLUSTERING_SEED}_n_init={CLUSTERING_N_INIT}"
    )
    labels: np.ndarray | None
    if targets.group_labels is None:
        reason = targets.group_labels_reason or "group_labels_unavailable"
        labels = None
    else:
        labels = np.asarray(targets.group_labels, dtype=str)[target_cells]
        groups, counts = np.unique(labels, return_counts=True)
        if groups.size < 2:
            reason = "fewer_than_two_groups"
        elif np.any(counts < 2):
            reason = "fewer_than_two_cells_in_group"
        else:
            reason = None
    if reason is None:
        assert labels is not None
        representation = workspace.representation
        if representation is None:
            reason = workspace.representation_reason or "constant_method_representation"
        else:
            components = min(CLUSTERING_MAX_COMPONENTS, representation.shape[1])
            cluster_representation = representation[:, :components]
            partition, reason = _select_clustering_partition(
                cluster_representation, workspace.cell_ids
            )
            if partition is None:
                assert reason is not None
            else:
                predicted = np.asarray(partition.assignments, dtype=np.int64)
                ari_loss = 1.0 - _adjusted_rand_index(labels, predicted)
                nmi_loss = 1.0 - _normalized_mutual_information(labels, predicted)
                fitted_procedure = (
                    f"{procedure}_selected_k={partition.n_clusters}_"
                    f"davies_bouldin={partition.criterion_value:.17g}"
                )
                return (
                    _completed_record(
                        "clustering_ari_loss",
                        ari_loss,
                        direction="lower_is_better",
                        descriptive_n=values.shape[0],
                        descriptive_unit="cells",
                        procedure=fitted_procedure,
                    ),
                    _completed_record(
                        "clustering_nmi_loss",
                        nmi_loss,
                        direction="lower_is_better",
                        descriptive_n=values.shape[0],
                        descriptive_unit="cells",
                        procedure=fitted_procedure,
                    ),
                )
    return (
        _unavailable_record(
            "clustering_ari_loss",
            reason,
            direction="lower_is_better",
            descriptive_n=values.shape[0],
            descriptive_unit="cells",
            procedure=procedure,
        ),
        _unavailable_record(
            "clustering_nmi_loss",
            reason,
            direction="lower_is_better",
            descriptive_n=values.shape[0],
            descriptive_unit="cells",
            procedure=procedure,
        ),
    )


def _profile_spearman(method_profile: np.ndarray, heldout_profile: np.ndarray) -> float:
    heldout_ranks = stats.rankdata(heldout_profile, method="average")
    method_ranks = stats.rankdata(method_profile, method="average")
    heldout_centered = heldout_ranks - np.mean(heldout_ranks)
    method_centered = method_ranks - np.mean(method_ranks)
    heldout_norm = float(np.linalg.norm(heldout_centered))
    method_norm = float(np.linalg.norm(method_centered))
    if heldout_norm == 0.0:  # caller fixes eligibility from heldout only
        raise AssertionError("heldout profile must be variable")
    if method_norm == 0.0:
        return 0.0
    value = float(
        np.dot(method_centered, heldout_centered) / (method_norm * heldout_norm)
    )
    return float(np.clip(value, -1.0, 1.0))


def evaluate_heldout_endpoints(
    output: MethodOutput,
    targets: EvaluatorTargets,
    *,
    _workspace: _EvaluationWorkspace | None = None,
) -> tuple[EndpointRecord, EndpointRecord]:
    """Compare method profiles with an independent evaluator-only count split."""

    workspace = _workspace or _evaluation_workspace(
        output, targets, include_representation=False
    )
    values = workspace.values
    target_cells = workspace.target_cell_order
    target_genes = workspace.target_gene_order
    procedure = "mean_profile_spearman_log2_cp10k_plus_1_independent_count_split"
    if targets.heldout_counts is None:
        reason = targets.heldout_reason or "independent_heldout_counts_unavailable"
        return (
            _unavailable_record(
                "heldout_gene_profile_rank_loss",
                reason,
                direction="lower_is_better",
                descriptive_n=0,
                descriptive_unit="heldout_variable_genes",
                procedure=procedure,
            ),
            _unavailable_record(
                "heldout_cell_profile_rank_loss",
                reason,
                direction="lower_is_better",
                descriptive_n=0,
                descriptive_unit="heldout_variable_cells",
                procedure=procedure,
            ),
        )

    heldout = targets.heldout_counts[target_cells][:, target_genes]
    method_log = values
    heldout_log = _counts_to_log2_cp10k(heldout)
    variable_genes = np.ptp(heldout_log, axis=0) > 0.0
    variable_cells = np.ptp(heldout_log, axis=1) > 0.0

    if np.any(variable_genes):
        gene_correlations = [
            _profile_spearman(method_log[:, gene], heldout_log[:, gene])
            for gene in np.flatnonzero(variable_genes)
        ]
        gene_record = _completed_record(
            "heldout_gene_profile_rank_loss",
            1.0 - float(np.mean(gene_correlations)),
            direction="lower_is_better",
            descriptive_n=len(gene_correlations),
            descriptive_unit="heldout_variable_genes",
            procedure=procedure,
        )
    else:
        gene_record = _unavailable_record(
            "heldout_gene_profile_rank_loss",
            "heldout_has_no_variable_gene_profiles",
            direction="lower_is_better",
            descriptive_n=0,
            descriptive_unit="heldout_variable_genes",
            procedure=procedure,
        )

    if np.any(variable_cells):
        cell_correlations = [
            _profile_spearman(method_log[cell], heldout_log[cell])
            for cell in np.flatnonzero(variable_cells)
        ]
        cell_record = _completed_record(
            "heldout_cell_profile_rank_loss",
            1.0 - float(np.mean(cell_correlations)),
            direction="lower_is_better",
            descriptive_n=len(cell_correlations),
            descriptive_unit="heldout_variable_cells",
            procedure=procedure,
        )
    else:
        cell_record = _unavailable_record(
            "heldout_cell_profile_rank_loss",
            "heldout_has_no_variable_cell_profiles",
            direction="lower_is_better",
            descriptive_n=0,
            descriptive_unit="heldout_variable_cells",
            procedure=procedure,
        )
    return gene_record, cell_record


def _squared_distance_block(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norms = np.sum(left * left, axis=1, dtype=np.float64)
    right_norms = np.sum(right * right, axis=1, dtype=np.float64)
    distances = left_norms[:, None] + right_norms[None, :] - 2.0 * (left @ right.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def _sparse_knn_squared_distances(
    representation: np.ndarray, n_neighbors: int
) -> sparse.csr_matrix:
    """Return a symmetric exact-kNN graph without allocating an N-by-N array."""

    n_cells = representation.shape[0]
    if not 1 <= n_neighbors < n_cells:
        raise ValueError("n_neighbors must be between one and n_cells minus one")
    edges: dict[tuple[int, int], float] = {}
    for start in range(0, n_cells, TRAJECTORY_DISTANCE_BLOCK_SIZE):
        stop = min(start + TRAJECTORY_DISTANCE_BLOCK_SIZE, n_cells)
        distances = _squared_distance_block(representation[start:stop], representation)
        local = np.arange(stop - start)
        distances[local, np.arange(start, stop)] = np.inf
        neighbors = np.argsort(distances, axis=1, kind="mergesort")[:, :n_neighbors]
        for local_index, columns in enumerate(neighbors):
            row = start + local_index
            for column in columns.tolist():
                edge = (min(row, int(column)), max(row, int(column)))
                distance = float(distances[local_index, column])
                previous = edges.get(edge)
                if previous is None or distance < previous:
                    edges[edge] = distance
    ordered = tuple(sorted(edges.items()))
    rows = np.asarray(
        [index for (left, right), _value in ordered for index in (left, right)],
        dtype=np.int64,
    )
    columns = np.asarray(
        [index for (left, right), _value in ordered for index in (right, left)],
        dtype=np.int64,
    )
    data = np.asarray(
        [value for _edge, value in ordered for _ in range(2)], dtype=np.float64
    )
    result = sparse.csr_matrix((data, (rows, columns)), shape=(n_cells, n_cells))
    result.sort_indices()
    return result


def _diffusion_distance_from_root(
    representation: np.ndarray, root_index: int
) -> tuple[np.ndarray | None, str | None]:
    n_cells = representation.shape[0]
    n_neighbors = min(
        TRAJECTORY_MAX_NEIGHBORS,
        max(2, int(np.sqrt(n_cells))),
        n_cells - 1,
    )
    distances = _sparse_knn_squared_distances(representation, n_neighbors)
    positive = distances.data[distances.data > 0.0]
    if positive.size == 0:
        return None, "trajectory_affinity_scale_unavailable"
    scale_squared = float(np.median(positive))
    if not np.isfinite(scale_squared) or scale_squared <= 0.0:
        return None, "trajectory_affinity_scale_unavailable"
    graph = distances.copy()
    graph.data = np.ones(graph.nnz, dtype=np.float64)
    component_count = sparse.csgraph.connected_components(
        graph, directed=False, return_labels=False
    )
    if component_count != 1:
        return None, "trajectory_graph_disconnected"
    affinity = distances.copy()
    affinity.data = np.exp(-affinity.data / (2.0 * scale_squared))
    affinity = affinity + sparse.identity(n_cells, format="csr", dtype=np.float64)
    degree = np.asarray(affinity.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0.0):
        return None, "trajectory_graph_has_zero_degree"
    inverse_sqrt_degree = 1.0 / np.sqrt(degree)
    degree_scaling = sparse.diags(inverse_sqrt_degree)
    symmetric_transition = degree_scaling @ affinity @ degree_scaling
    n_eigenvectors = min(TRAJECTORY_DIFFUSION_MODES + 1, n_cells - 1)
    initial = np.linspace(1.0, 2.0, n_cells, dtype=np.float64)
    initial /= np.linalg.norm(initial)
    try:
        eigenvalues, eigenvectors = sparse.linalg.eigsh(
            symmetric_transition,
            k=n_eigenvectors,
            which="LA",
            v0=initial,
            tol=1e-10,
        )
    except sparse.linalg.ArpackNoConvergence:
        return None, "trajectory_diffusion_eigensolver_failed"
    order = np.argsort(-eigenvalues, kind="mergesort")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    mode_mask = (
        (np.arange(eigenvalues.size) > 0)
        & (np.abs(eigenvalues) > 1e-10)
        & (eigenvalues < 1.0 - 1e-10)
    )
    if not np.any(mode_mask):
        return None, "trajectory_diffusion_modes_unavailable"
    selected_values = eigenvalues[mode_mask]
    right_eigenvectors = inverse_sqrt_degree[:, None] * eigenvectors[:, mode_mask]
    multiscale = (
        right_eigenvectors * (selected_values / (1.0 - selected_values))[None, :]
    )
    difference = multiscale - multiscale[root_index]
    pseudotime = np.sqrt(np.sum(difference * difference, axis=1))
    if not np.all(np.isfinite(pseudotime)):
        return None, "nonfinite_diffusion_pseudotime"
    if np.ptp(pseudotime) == 0.0:
        return None, "constant_diffusion_pseudotime"
    return pseudotime, None


def evaluate_trajectory_endpoint(
    output: MethodOutput,
    targets: EvaluatorTargets,
    *,
    _workspace: _EvaluationWorkspace | None = None,
) -> EndpointRecord:
    """Evaluate genuine root-oriented pseudotime with multiscale diffusion."""

    workspace = _workspace or _evaluation_workspace(
        output, targets, include_representation=True
    )
    values = workspace.values
    cell_ids = workspace.cell_ids
    procedure = (
        "root_oriented_multiscale_diffusion_log2_cp10k_plus_1_full_svd_"
        "blockwise_exact_knn=floor_sqrt_n_capped_15_sparse_eigsh_modes=15"
    )
    if targets.trajectory is None:
        return _unavailable_record(
            "trajectory_pseudotime_rank_loss",
            targets.trajectory_reason
            or "genuine_pseudotime_not_available_in_simulator_output",
            direction="lower_is_better",
            descriptive_n=0,
            descriptive_unit="trajectory_cells",
            procedure=procedure,
        )
    if values.shape[0] < 3:
        return _unavailable_record(
            "trajectory_pseudotime_rank_loss",
            "fewer_than_three_trajectory_cells",
            direction="lower_is_better",
            descriptive_n=values.shape[0],
            descriptive_unit="trajectory_cells",
            procedure=procedure,
        )
    representation = workspace.representation
    if representation is None:
        return _unavailable_record(
            "trajectory_pseudotime_rank_loss",
            "constant_method_representation",
            direction="lower_is_better",
            descriptive_n=values.shape[0],
            descriptive_unit="trajectory_cells",
            procedure=procedure,
        )
    representation = representation[:, :TRAJECTORY_MAX_COMPONENTS]
    trajectory = targets.trajectory
    trajectory_cells = {value: index for index, value in enumerate(trajectory.cell_ids)}
    pseudotime = trajectory.pseudotime[
        np.asarray([trajectory_cells[value] for value in cell_ids])
    ]
    root_index = cell_ids.index(trajectory.root_cell_id)
    estimated, reason = _diffusion_distance_from_root(representation, root_index)
    if estimated is None:
        assert reason is not None
        return _unavailable_record(
            "trajectory_pseudotime_rank_loss",
            reason,
            direction="lower_is_better",
            descriptive_n=values.shape[0],
            descriptive_unit="trajectory_cells",
            procedure=procedure,
        )
    correlation = _profile_spearman(estimated, pseudotime)
    return _completed_record(
        "trajectory_pseudotime_rank_loss",
        1.0 - correlation,
        direction="lower_is_better",
        descriptive_n=values.shape[0],
        descriptive_unit="trajectory_cells",
        procedure=procedure,
    )


def evaluate_downstream_endpoints(
    output: MethodOutput, targets: EvaluatorTargets
) -> tuple[EndpointRecord, ...]:
    """Return the fixed complete evaluator-only endpoint record."""

    try:
        workspace = _evaluation_workspace(
            output,
            targets,
            include_representation=(
                targets.group_labels is not None or targets.trajectory is not None
            ),
        )
        marker, recall, false_discovery_rate = evaluate_marker_and_de_endpoints(
            output, targets, _workspace=workspace
        )
        clustering_ari, clustering_nmi = evaluate_clustering_endpoints(
            output, targets, _workspace=workspace
        )
        heldout_gene, heldout_cell = evaluate_heldout_endpoints(
            output, targets, _workspace=workspace
        )
        trajectory = evaluate_trajectory_endpoint(output, targets, _workspace=workspace)
    except (
        np.linalg.LinAlgError,
        sparse.linalg.ArpackError,
        FloatingPointError,
        OverflowError,
    ):
        records = tuple(
            _unavailable_record(
                endpoint,
                "numeric_evaluation_failed",
                direction=_ENDPOINT_CONTRACT[endpoint][0],
                descriptive_n=0,
                descriptive_unit=_ENDPOINT_CONTRACT[endpoint][1],
                procedure="terminal_expected_numeric_failure",
            )
            for endpoint in DOWNSTREAM_ENDPOINT_NAMES
        )
        return records
    records = (
        marker,
        clustering_ari,
        clustering_nmi,
        recall,
        false_discovery_rate,
        heldout_gene,
        heldout_cell,
        trajectory,
    )
    if tuple(record.endpoint for record in records) != DOWNSTREAM_ENDPOINT_NAMES:
        raise AssertionError("downstream endpoint implementation is incomplete")
    return records


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
            column is not None and column in var for column in expected_columns.values()
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
