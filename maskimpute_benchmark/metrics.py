"""Truth-isolated metrics for the publication benchmark.

All matrices use the AnnData convention (cells by genes) and must already be
on the evaluator's common scale.  This module deliberately does no
normalization, clipping, missing-value replacement, or other sanitization.
Only probabilities entering logarithms are clipped, and that clipping is
local to log-loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from maskimpute.sparse_input import _unmasked_array


TRUTH_KINDS = {
    "exact_pre_capture",
    "exact_continuous",
    "proxy_high_depth",
    "orthogonal_only",
}

_SCORE_NAMES = (
    "auroc",
    "average_precision",
    "brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "ece",
)

_SUBSETS = (
    "overall",
    "induced_dropout",
    "pre_dropout_zero",
    "non_dropout_nonzero",
    "truth_nonzero",
    "observed_positive",
    "marker",
)


@dataclass(frozen=True)
class MetricValue:
    """A scalar estimate together with its denominator and availability state."""

    value: float | None
    n: int
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.n, (int, np.integer)) or isinstance(self.n, bool):
            raise TypeError("n must be an integer")
        if self.n < 0:
            raise ValueError("n must be non-negative")
        if self.value is None:
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("an unavailable metric requires a reason")
            return
        if not isinstance(self.value, (int, float, np.integer, np.floating)):
            raise TypeError("value must be numeric or None")
        if not np.isfinite(float(self.value)):
            raise ValueError("metric value must be finite")
        if self.reason is not None:
            raise ValueError("reason must be None when a metric has a value")


def _metric(value: float, n: int) -> MetricValue:
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return _unavailable(n, "nonfinite_metric")
    if not np.isfinite(numeric):
        return _unavailable(n, "nonfinite_metric")
    return MetricValue(numeric, int(n), None)


def _unavailable(n: int, reason: str) -> MetricValue:
    return MetricValue(None, int(n), reason)


def _validate_truth_kind(truth_kind: str) -> None:
    if truth_kind not in TRUTH_KINDS:
        allowed = ", ".join(sorted(TRUTH_KINDS))
        raise ValueError(f"truth_kind must be one of: {allowed}")


def _as_matrix(
    name: str,
    value: Any,
    *,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required")
    array = _unmasked_array(value, name)
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must be a real numeric matrix")
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one cell and one gene")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(float, copy=False)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must remain finite when represented as float64")
    return converted


def _as_probability_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray:
    probability = _as_matrix("p_pre_zero", value, shape=shape)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("p_pre_zero probabilities must lie in [0, 1]")
    return probability


def _gene_selector(
    name: str,
    selector: Any,
    n_genes: int,
    *,
    default_all: bool,
) -> np.ndarray:
    if selector is None:
        return np.full(n_genes, default_all, dtype=bool)
    array = _unmasked_array(selector, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(array.dtype, np.bool_):
        if array.size != n_genes:
            raise ValueError(f"{name} must have one value per gene")
        return array.astype(bool, copy=True)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a boolean mask or integer indices")
    if np.any((array < 0) | (array >= n_genes)):
        raise ValueError(f"{name} contains an out-of-range gene index")
    mask = np.zeros(n_genes, dtype=bool)
    mask[array.astype(int)] = True
    return mask


def entry_masks(observed: Any, truth: Any) -> dict[str, np.ndarray]:
    """Return the prespecified entry sets without modifying either matrix."""

    observed_array = _as_matrix("observed", observed)
    truth_array = _as_matrix("truth", truth, shape=observed_array.shape)
    return {
        "overall": np.ones(observed_array.shape, dtype=bool),
        "induced_dropout": (observed_array == 0) & (truth_array > 0),
        "pre_dropout_zero": truth_array == 0,
        "non_dropout_nonzero": (observed_array > 0) & (truth_array > 0),
        "truth_nonzero": truth_array > 0,
        "observed_positive": observed_array > 0,
    }


def _error_metric(
    difference: np.ndarray,
    mask: np.ndarray,
    *,
    squared: bool,
) -> MetricValue:
    n = int(mask.sum())
    if n == 0:
        return _unavailable(0, "no_entries")
    values = difference[mask]
    with np.errstate(over="ignore", invalid="ignore"):
        if squared:
            return _metric(np.mean(values * values), n)
        return _metric(np.mean(np.abs(values)), n)


def _gnrmse(
    difference: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> MetricValue:
    if not np.any(mask):
        return _unavailable(0, "no_entries")
    truth_sd = np.maximum(np.std(truth, axis=0, ddof=0), 1e-8)
    values: list[float] = []
    for gene in range(truth.shape[1]):
        selected = mask[:, gene]
        if np.any(selected):
            with np.errstate(over="ignore", invalid="ignore"):
                rmse = np.sqrt(np.mean(difference[selected, gene] ** 2))
                values.append(float(rmse / truth_sd[gene]))
    return _metric(np.mean(values), len(values))


def _correlation_matrix_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
    selector: np.ndarray,
    *,
    variables: str,
) -> tuple[MetricValue, int]:
    if variables == "genes":
        selected_imputed = imputed[:, selector]
        selected_truth = truth[:, selector]
        n_variables = int(selector.sum())
        reason = "fewer_than_two_variable_genes"
        constant_reason = "constant_gene_profile"
        rowvar = False
        standard_deviation_axis = 0
    elif variables == "cells":
        selected_imputed = imputed
        selected_truth = truth
        n_variables = truth.shape[0]
        reason = "fewer_than_two_variable_cells"
        constant_reason = "constant_cell_profile"
        rowvar = True
        standard_deviation_axis = 1
    else:  # pragma: no cover - internal programming error
        raise AssertionError(variables)

    if n_variables < 2:
        return _unavailable(n_variables, reason), n_variables
    imputed_reference = np.take(selected_imputed, [0], axis=standard_deviation_axis)
    truth_reference = np.take(selected_truth, [0], axis=standard_deviation_axis)
    constant = np.all(
        selected_imputed == imputed_reference, axis=standard_deviation_axis
    ) | np.all(selected_truth == truth_reference, axis=standard_deviation_axis)
    if np.any(constant):
        return _unavailable(n_variables, constant_reason), n_variables
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        corr_imputed = np.corrcoef(selected_imputed, rowvar=rowvar)
        corr_truth = np.corrcoef(selected_truth, rowvar=rowvar)
    upper = np.triu_indices(n_variables, k=1)
    differences = np.abs(corr_imputed[upper] - corr_truth[upper])
    if not np.all(np.isfinite(differences)):
        return _unavailable(len(differences), "nonfinite_correlation"), n_variables
    return _metric(np.mean(differences), len(differences)), n_variables


def _pairwise_distance_distortion(
    imputed: np.ndarray, truth: np.ndarray
) -> MetricValue:
    n_cells = truth.shape[0]
    n_pairs = n_cells * (n_cells - 1) // 2
    if n_pairs == 0:
        return _unavailable(0, "fewer_than_two_cells")
    values = np.empty(n_pairs, dtype=float)
    offset = 0
    for first in range(n_cells - 1):
        count = n_cells - first - 1
        with np.errstate(over="ignore", invalid="ignore"):
            truth_distance = np.linalg.norm(truth[first + 1 :] - truth[first], axis=1)
            imputed_distance = np.linalg.norm(
                imputed[first + 1 :] - imputed[first], axis=1
            )
            values[offset : offset + count] = np.abs(imputed_distance - truth_distance)
        offset += count
    return _metric(np.mean(values), n_pairs)


def _mean_gene_wasserstein_distance(
    imputed: np.ndarray, truth: np.ndarray
) -> MetricValue:
    """Average exact empirical 1-Wasserstein distance across genes.

    Both matrices contain the same number of equally weighted cells.  In one
    dimension, pairing their sorted values therefore gives the exact empirical
    1-Wasserstein distance for a gene without a fitted bandwidth or a random
    projection.
    """

    sorted_imputed = np.sort(imputed, axis=0)
    sorted_truth = np.sort(truth, axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        per_gene = np.mean(np.abs(sorted_imputed - sorted_truth), axis=0)
        return _metric(np.mean(per_gene), truth.shape[1])


def _reconstruction_metric_names() -> list[str]:
    names: list[str] = []
    for subset in _SUBSETS:
        suffix = "" if subset == "overall" else f"_{subset}"
        names.extend(f"{metric}{suffix}" for metric in ("mse", "mae", "gnrmse"))
    names.extend(
        [
            "mean_distortion",
            "variance_distortion",
            "mean_gene_wasserstein_distance",
            "false_positive_expression",
            "corr_err",
            "n_corr_genes",
            "cell_correlation_distortion",
            "cell_distance_distortion",
        ]
    )
    names.extend(
        f"{metric}_{subset}"
        for metric in ("mse", "mae", "gnrmse")
        for subset in ("dropout", "nonzero")
    )
    names.append("pairwise_cell_distance_distortion")
    return names


def reconstruction_metrics(
    imputed: Any,
    observed: Any,
    truth: Any,
    marker_genes: Any = None,
    corr_gene_mask: Any = None,
    *,
    truth_kind: str = "exact_pre_capture",
) -> dict[str, MetricValue]:
    """Compute a complete, reason-coded reconstruction metric record."""

    _validate_truth_kind(truth_kind)
    observed_array = _as_matrix("observed", observed)
    imputed_array = _as_matrix("imputed", imputed, shape=observed_array.shape)

    if truth_kind == "orthogonal_only":
        if truth is not None:
            _as_matrix("truth", truth, shape=observed_array.shape)
        return {
            name: _unavailable(0, "truth_unavailable")
            for name in _reconstruction_metric_names()
        }

    truth_array = _as_matrix("truth", truth, shape=observed_array.shape)
    masks = entry_masks(observed_array, truth_array)
    marker_mask = _gene_selector(
        "marker_genes", marker_genes, truth_array.shape[1], default_all=False
    )
    masks["marker"] = np.broadcast_to(marker_mask, truth_array.shape)
    corr_mask = _gene_selector(
        "corr_gene_mask", corr_gene_mask, truth_array.shape[1], default_all=True
    )

    difference = imputed_array - truth_array
    result: dict[str, MetricValue] = {}
    for subset in _SUBSETS:
        suffix = "" if subset == "overall" else f"_{subset}"
        mask = masks[subset]
        if subset == "marker" and marker_genes is None:
            for metric in ("mse", "mae", "gnrmse"):
                result[f"{metric}{suffix}"] = _unavailable(
                    0, "marker_genes_not_provided"
                )
            continue
        if subset == "pre_dropout_zero" and truth_kind != "exact_pre_capture":
            reason = (
                "undefined_for_continuous_truth"
                if truth_kind == "exact_continuous"
                else "proxy_truth_not_exact"
            )
            n = int(mask.sum())
            for metric in ("mse", "mae", "gnrmse"):
                result[f"{metric}{suffix}"] = _unavailable(n, reason)
            continue
        result[f"mse{suffix}"] = _error_metric(difference, mask, squared=True)
        result[f"mae{suffix}"] = _error_metric(difference, mask, squared=False)
        result[f"gnrmse{suffix}"] = _gnrmse(difference, truth_array, mask)

    with np.errstate(over="ignore", invalid="ignore"):
        mean_difference = np.abs(
            np.mean(imputed_array, axis=0) - np.mean(truth_array, axis=0)
        )
        variance_difference = np.abs(
            np.var(imputed_array, axis=0, ddof=0) - np.var(truth_array, axis=0, ddof=0)
        )
    result["mean_distortion"] = _metric(np.mean(mean_difference), truth_array.shape[1])
    result["variance_distortion"] = _metric(
        np.mean(variance_difference), truth_array.shape[1]
    )
    result["mean_gene_wasserstein_distance"] = _mean_gene_wasserstein_distance(
        imputed_array, truth_array
    )

    pre_zero_n = int(masks["pre_dropout_zero"].sum())
    if truth_kind == "exact_pre_capture":
        if pre_zero_n == 0:
            result["false_positive_expression"] = _unavailable(0, "no_entries")
        else:
            expressed = imputed_array[masks["pre_dropout_zero"]] > 0
            result["false_positive_expression"] = _metric(
                np.mean(expressed), pre_zero_n
            )
    else:
        reason = (
            "undefined_for_continuous_truth"
            if truth_kind == "exact_continuous"
            else "proxy_truth_not_exact"
        )
        result["false_positive_expression"] = _unavailable(pre_zero_n, reason)

    corr_err, n_corr_genes = _correlation_matrix_distortion(
        imputed_array, truth_array, corr_mask, variables="genes"
    )
    result["corr_err"] = corr_err
    result["n_corr_genes"] = _metric(float(n_corr_genes), n_corr_genes)
    cell_corr, _ = _correlation_matrix_distortion(
        imputed_array,
        truth_array,
        np.ones(truth_array.shape[1], dtype=bool),
        variables="cells",
    )
    result["cell_correlation_distortion"] = cell_corr
    result["cell_distance_distortion"] = _pairwise_distance_distortion(
        imputed_array, truth_array
    )
    for metric in ("mse", "mae", "gnrmse"):
        result[f"{metric}_dropout"] = result[f"{metric}_induced_dropout"]
        result[f"{metric}_nonzero"] = result[f"{metric}_non_dropout_nonzero"]
    result["pairwise_cell_distance_distortion"] = result["cell_distance_distortion"]
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _auroc(probability: np.ndarray, outcome: np.ndarray) -> float:
    n_positive = int(outcome.sum())
    n_negative = outcome.size - n_positive
    positive_rank_sum = float(_average_ranks(probability)[outcome == 1].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )


def _average_precision(probability: np.ndarray, outcome: np.ndarray) -> float:
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_outcome = outcome[order]
    total_positive = int(outcome.sum())
    true_positive = 0
    seen = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < outcome.size:
        stop = start + 1
        while (
            stop < outcome.size
            and sorted_probability[stop] == sorted_probability[start]
        ):
            stop += 1
        true_positive += int(sorted_outcome[start:stop].sum())
        seen += stop - start
        recall = true_positive / total_positive
        precision = true_positive / seen
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return average_precision


def _expit(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=float)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _calibration_fit(
    probability: np.ndarray, outcome: np.ndarray
) -> tuple[MetricValue, MetricValue]:
    n = outcome.size
    if np.unique(outcome).size < 2:
        unavailable = _unavailable(n, "single_class")
        return unavailable, unavailable
    if np.unique(probability).size < 2:
        unavailable = _unavailable(n, "constant_predictions")
        return unavailable, unavailable
    if np.any((probability == 0.0) | (probability == 1.0)):
        unavailable = _unavailable(n, "boundary_predictions")
        return unavailable, unavailable

    logit = np.log(probability / (1.0 - probability))
    design = np.column_stack((np.ones(n), logit))
    coefficient = np.array([0.0, 1.0])
    converged = False
    for _ in range(100):
        fitted = _expit(design @ coefficient)
        gradient = design.T @ (outcome - fitted)
        weights = fitted * (1.0 - fitted)
        information = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            unavailable = _unavailable(n, "calibration_fit_failed")
            return unavailable, unavailable
        coefficient += step
        if not np.all(np.isfinite(coefficient)) or np.max(np.abs(coefficient)) > 1e6:
            unavailable = _unavailable(n, "calibration_fit_failed")
            return unavailable, unavailable
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    if not converged:
        unavailable = _unavailable(n, "calibration_fit_failed")
        return unavailable, unavailable
    return _metric(coefficient[0], n), _metric(coefficient[1], n)


def _wilson_interval(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return float(centre - radius), float(centre + radius)


def _reliability(
    probability: np.ndarray, outcome: np.ndarray, n_bins: int
) -> tuple[MetricValue, list[dict[str, float | int]]]:
    chunks = tie_aware_groups(probability, n_bins)
    bins: list[dict[str, float | int]] = []
    weighted_error = 0.0
    for index, chunk in enumerate(chunks, start=1):
        mean_prediction = float(np.mean(probability[chunk]))
        successes = int(outcome[chunk].sum())
        observed_fraction = successes / len(chunk)
        # Wilson intervals describe each reliability bin; they are not
        # inferential confidence intervals for comparing methods.
        lower, upper = _wilson_interval(successes, len(chunk))
        weighted_error += len(chunk) * abs(mean_prediction - observed_fraction)
        bins.append(
            {
                "bin": index,
                "n": int(len(chunk)),
                "mean_prediction": mean_prediction,
                "observed_fraction": float(observed_fraction),
                "wilson_lower": lower,
                "wilson_upper": upper,
            }
        )
    return _metric(weighted_error / outcome.size, outcome.size), bins


def tie_aware_groups(values: np.ndarray, maximum_groups: int) -> list[np.ndarray]:
    """Target equal-frequency groups without splitting identical values."""

    if not isinstance(maximum_groups, (int, np.integer)) or isinstance(
        maximum_groups, (bool, np.bool_)
    ):
        raise TypeError("maximum_groups must be an integer")
    if maximum_groups <= 0:
        raise ValueError("maximum_groups must be positive")
    array = _unmasked_array(values, "values")
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size == 0:
        raise ValueError("values must be non-empty")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError("values must be real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite values")

    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    value_groups: list[np.ndarray] = []
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        value_groups.append(order[start:stop])
        start = stop

    n_groups = min(int(maximum_groups), len(value_groups))
    if n_groups == 1:
        return [np.concatenate(value_groups)]

    cumulative = np.cumsum([len(group) for group in value_groups])
    result: list[np.ndarray] = []
    previous_end = 0
    for split in range(1, n_groups):
        target = split * array.size / n_groups
        remaining_groups = n_groups - split
        latest_end = len(value_groups) - remaining_groups
        candidates = range(previous_end + 1, latest_end + 1)
        end = min(
            candidates,
            key=lambda candidate: (
                abs(cumulative[candidate - 1] - target),
                cumulative[candidate - 1],
            ),
        )
        result.append(np.concatenate(value_groups[previous_end:end]))
        previous_end = end
    result.append(np.concatenate(value_groups[previous_end:]))
    return result


def _undefined_score_result(n: int, reason: str) -> dict[str, Any]:
    result: dict[str, Any] = {"n": int(n)}
    result.update({name: _unavailable(n, reason) for name in _SCORE_NAMES})
    result["reliability_bins"] = []
    return result


def _score_selected(
    probability_matrix: np.ndarray,
    observed: np.ndarray,
    truth: np.ndarray | None,
    evaluation_mask: np.ndarray,
    n_bins: int,
    truth_kind: str,
) -> dict[str, Any]:
    n = int(evaluation_mask.sum())
    if truth_kind == "orthogonal_only":
        return _undefined_score_result(n, "truth_unavailable")
    if truth_kind == "exact_continuous":
        return _undefined_score_result(n, "undefined_for_continuous_truth")
    if truth_kind == "proxy_high_depth":
        return _undefined_score_result(n, "proxy_truth_not_exact")
    if truth is None:  # pragma: no cover - guarded by public validation
        raise AssertionError("exact truth is required")
    if n == 0:
        return _undefined_score_result(0, "no_observed_zeros")

    probability = probability_matrix[evaluation_mask]
    outcome = (truth[evaluation_mask] == 0).astype(int)
    canonical_order = np.lexsort((outcome, probability))
    probability = probability[canonical_order]
    outcome = outcome[canonical_order]
    result: dict[str, Any] = {"n": n}
    if np.unique(outcome).size < 2:
        result["auroc"] = _unavailable(n, "single_class")
        result["average_precision"] = _unavailable(n, "single_class")
    else:
        result["auroc"] = _metric(_auroc(probability, outcome), n)
        result["average_precision"] = _metric(
            _average_precision(probability, outcome), n
        )
    result["brier"] = _metric(np.mean((probability - outcome) ** 2), n)
    epsilon = 1e-15
    log_probability = np.clip(probability, epsilon, 1.0 - epsilon)
    log_loss = -np.mean(
        outcome * np.log(log_probability)
        + (1 - outcome) * np.log(1.0 - log_probability)
    )
    result["log_loss"] = _metric(log_loss, n)
    intercept, slope = _calibration_fit(probability, outcome)
    result["calibration_intercept"] = intercept
    result["calibration_slope"] = slope
    ece, bins = _reliability(probability, outcome, n_bins)
    result["ece"] = ece
    result["reliability_bins"] = bins
    return result


def _validate_n_bins(n_bins: int) -> None:
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool):
        raise TypeError("n_bins must be an integer")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")


def zero_score_metrics(
    p_pre_zero: Any,
    observed: Any,
    truth: Any,
    n_bins: int = 10,
    *,
    truth_kind: str = "exact_pre_capture",
) -> dict[str, Any]:
    """Evaluate a pre-capture-zero probability only at observed zero entries."""

    _validate_truth_kind(truth_kind)
    _validate_n_bins(n_bins)
    observed_array = _as_matrix("observed", observed)
    probability = _as_probability_matrix(p_pre_zero, observed_array.shape)
    truth_array: np.ndarray | None
    if truth_kind == "orthogonal_only" and truth is None:
        truth_array = None
    else:
        truth_array = _as_matrix("truth", truth, shape=observed_array.shape)
    evaluation_mask = observed_array == 0
    return _score_selected(
        probability,
        observed_array,
        truth_array,
        evaluation_mask,
        int(n_bins),
        truth_kind,
    )


def _stratum_record(
    *,
    stratum_type: str,
    label: str,
    lower: float | None,
    upper: float | None,
    mask: np.ndarray,
    probability: np.ndarray,
    observed: np.ndarray,
    truth: np.ndarray | None,
    n_bins: int,
    truth_kind: str,
) -> dict[str, Any]:
    metrics = _score_selected(probability, observed, truth, mask, n_bins, truth_kind)
    return {
        "stratum_type": stratum_type,
        "label": label,
        "lower": lower,
        "upper": upper,
        "n": int(mask.sum()),
        "metrics": metrics,
    }


def stratified_zero_score_metrics(
    p_pre_zero: Any,
    observed: Any,
    truth: Any,
    n_bins: int = 10,
    *,
    truth_kind: str = "exact_pre_capture",
) -> dict[str, list[dict[str, Any]]]:
    """Return zero-score metrics by library quartile and truth-expression bin."""

    _validate_truth_kind(truth_kind)
    _validate_n_bins(n_bins)
    observed_array = _as_matrix("observed", observed)
    probability = _as_probability_matrix(p_pre_zero, observed_array.shape)
    if truth_kind == "orthogonal_only" and truth is None:
        truth_array = None
    else:
        truth_array = _as_matrix("truth", truth, shape=observed_array.shape)

    observed_zero = observed_array == 0
    library_size = np.sum(observed_array, axis=1)
    cell_chunks = tie_aware_groups(library_size, 4)
    cell_chunks.extend(np.array([], dtype=int) for _ in range(4 - len(cell_chunks)))
    library_records: list[dict[str, Any]] = []
    for quartile, cells in enumerate(cell_chunks, start=1):
        cell_mask = np.zeros(observed_array.shape[0], dtype=bool)
        cell_mask[cells] = True
        mask = observed_zero & cell_mask[:, None]
        lower = float(np.min(library_size[cells])) if len(cells) else None
        upper = float(np.max(library_size[cells])) if len(cells) else None
        library_records.append(
            _stratum_record(
                stratum_type="library_size_quartiles",
                label=f"Q{quartile}",
                lower=lower,
                upper=upper,
                mask=mask,
                probability=probability,
                observed=observed_array,
                truth=truth_array,
                n_bins=int(n_bins),
                truth_kind=truth_kind,
            )
        )

    bounds = ((0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, np.inf))
    labels = ("[0,1)", "[1,2)", "[2,4)", "[4,inf)")
    truth_records: list[dict[str, Any]] = []
    for (lower, upper), label in zip(bounds, labels, strict=True):
        if truth_array is None:
            mask = np.zeros(observed_array.shape, dtype=bool)
        else:
            mask = observed_zero & (truth_array >= lower) & (truth_array < upper)
        truth_records.append(
            _stratum_record(
                stratum_type="truth_expression_bins",
                label=label,
                lower=lower,
                upper=None if np.isinf(upper) else float(upper),
                mask=mask,
                probability=probability,
                observed=observed_array,
                truth=truth_array,
                n_bins=int(n_bins),
                truth_kind=truth_kind,
            )
        )
    return {
        "library_size_quartiles": library_records,
        "truth_expression_bins": truth_records,
    }


__all__ = [
    "MetricValue",
    "entry_masks",
    "reconstruction_metrics",
    "stratified_zero_score_metrics",
    "tie_aware_groups",
    "zero_score_metrics",
]
