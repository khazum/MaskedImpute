"""Truth-isolated metrics for the publication benchmark.

All matrices use the AnnData convention (cells by genes) and must already be
on the evaluator's common scale.  This module deliberately does no
normalization, clipping, missing-value replacement, or other sanitization.
Only probabilities entering logarithms are clipped, and that clipping is
local to log-loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
import math
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


_ScaledTerm = tuple[float, int]
_ZERO_TERM: _ScaledTerm = (0.0, 0)
_FLOAT64_MAX = np.finfo(np.float64).max


def _normalize_term(mantissa: float, exponent: int) -> _ScaledTerm:
    if mantissa == 0.0:
        return _ZERO_TERM
    normalized_mantissa, adjustment = math.frexp(float(mantissa))
    return normalized_mantissa, int(exponent) + adjustment


def _term_from_float(value: float) -> _ScaledTerm:
    if value < 0.0 or not math.isfinite(value):
        raise ValueError("scaled terms require finite nonnegative values")
    return _normalize_term(value, 0)


def _term_from_scaled(
    scale: float,
    coefficient: float,
    *,
    power: int = 1,
) -> _ScaledTerm:
    if (
        scale < 0.0
        or coefficient < 0.0
        or not math.isfinite(scale)
        or not math.isfinite(coefficient)
    ):
        raise ValueError("scaled terms require finite nonnegative factors")
    if scale == 0.0 or coefficient == 0.0:
        return _ZERO_TERM
    scale_mantissa, scale_exponent = math.frexp(scale)
    coefficient_mantissa, coefficient_exponent = math.frexp(coefficient)
    return _normalize_term(
        coefficient_mantissa * scale_mantissa**power,
        coefficient_exponent + power * scale_exponent,
    )


def _term_ratio(numerator: _ScaledTerm, denominator: _ScaledTerm) -> _ScaledTerm:
    numerator_mantissa, numerator_exponent = numerator
    denominator_mantissa, denominator_exponent = denominator
    if denominator_mantissa == 0.0:
        raise ZeroDivisionError("scaled-term denominator must be positive")
    if numerator_mantissa == 0.0:
        return _ZERO_TERM
    return _normalize_term(
        numerator_mantissa / denominator_mantissa,
        numerator_exponent - denominator_exponent,
    )


def _term_mean(terms: list[_ScaledTerm]) -> _ScaledTerm:
    nonzero = [term for term in terms if term[0] != 0.0]
    if not nonzero:
        return _ZERO_TERM
    maximum_exponent = max(exponent for _, exponent in nonzero)
    aligned = math.fsum(
        math.ldexp(mantissa, exponent - maximum_exponent)
        for mantissa, exponent in nonzero
    )
    return _normalize_term(aligned / len(terms), maximum_exponent)


def _term_is_less(left: _ScaledTerm, right: _ScaledTerm) -> bool:
    if left[0] == 0.0:
        return right[0] != 0.0
    if right[0] == 0.0:
        return False
    return left[1] < right[1] or (left[1] == right[1] and left[0] < right[0])


def _term_to_float(term: _ScaledTerm) -> float | None:
    mantissa, exponent = term
    if mantissa == 0.0:
        return 0.0
    try:
        value = math.ldexp(mantissa, exponent)
    except OverflowError:
        return None
    if not math.isfinite(value) or value == 0.0:
        return None
    return value


def _metric_from_term(term: _ScaledTerm, n: int) -> MetricValue:
    value = _term_to_float(term)
    if value is None:
        return _unavailable(n, "nonfinite_metric")
    return MetricValue(value, int(n), None)


def _term_from_fraction(value: Fraction) -> _ScaledTerm:
    """Retain one positive exact rational without materializing its magnitude."""

    if value < 0:
        raise ValueError("scaled terms require nonnegative values")
    if value == 0:
        return _ZERO_TERM
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if exponent >= 0:
        coefficient = value / Fraction(1 << exponent)
    else:
        coefficient = value * Fraction(1 << -exponent)
    return _normalize_term(float(coefficient), exponent)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _variance_difference_fraction(
    left: np.ndarray,
    right: np.ndarray,
) -> Fraction:
    """Return an exact population-variance difference in linear time."""

    left_values = [Fraction.from_float(float(value)) for value in left]
    right_values = [Fraction.from_float(float(value)) for value in right]
    common_scale = max(
        (abs(value) for value in (*left_values, *right_values)),
        default=Fraction(),
    )
    if common_scale == 0:
        return Fraction()

    left_scaled = [value / common_scale for value in left_values]
    right_scaled = [value / common_scale for value in right_values]
    count = len(left_scaled)
    left_mean = sum(left_scaled, start=Fraction()) / count
    right_mean = sum(right_scaled, start=Fraction()) / count
    centered_left = [value - left_mean for value in left_scaled]
    centered_right = [value - right_mean for value in right_scaled]
    scaled_difference = sum(
        (
            (left_value - right_value) * (left_value + right_value)
            for left_value, right_value in zip(
                centered_left,
                centered_right,
                strict=True,
            )
        ),
        start=Fraction(),
    )
    return abs(scaled_difference) * common_scale * common_scale / count


def _euclidean_norm_difference_term(
    imputed_left: np.ndarray,
    imputed_right: np.ndarray,
    truth_left: np.ndarray,
    truth_right: np.ndarray,
) -> _ScaledTerm:
    """Compute ``abs(||u|| - ||v||)`` before either norm is rounded."""

    imputed_difference = [
        Fraction.from_float(float(left)) - Fraction.from_float(float(right))
        for left, right in zip(imputed_left, imputed_right, strict=True)
    ]
    truth_difference = [
        Fraction.from_float(float(left)) - Fraction.from_float(float(right))
        for left, right in zip(truth_left, truth_right, strict=True)
    ]
    common_scale = max(
        (abs(value) for value in (*imputed_difference, *truth_difference)),
        default=Fraction(),
    )
    if common_scale == 0:
        return _ZERO_TERM

    imputed_scaled = [value / common_scale for value in imputed_difference]
    truth_scaled = [value / common_scale for value in truth_difference]
    squared_norm_difference = abs(
        sum(
            (
                (imputed_value - truth_value) * (imputed_value + truth_value)
                for imputed_value, truth_value in zip(
                    imputed_scaled,
                    truth_scaled,
                    strict=True,
                )
            ),
            start=Fraction(),
        )
    )
    if squared_norm_difference == 0:
        return _ZERO_TERM
    imputed_squared_norm = sum(
        (value * value for value in imputed_scaled),
        start=Fraction(),
    )
    truth_squared_norm = sum(
        (value * value for value in truth_scaled),
        start=Fraction(),
    )
    with localcontext() as context:
        context.prec = 120
        denominator = _fraction_to_decimal(imputed_squared_norm).sqrt()
        denominator += _fraction_to_decimal(truth_squared_norm).sqrt()
        coefficient = _fraction_to_decimal(squared_norm_difference) / denominator
    return _term_from_fraction(common_scale * Fraction(coefficient))


def _scaled_signed_differences(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return ``left - right`` divided by a safe common magnitude."""

    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape:
        raise ValueError("difference operands must have the same shape")
    if left_values.size == 0:
        return 0.0, np.empty(0, dtype=np.float64)

    left_magnitude = np.abs(left_values)
    right_magnitude = np.abs(right_values)
    same_sign = np.signbit(left_values) == np.signbit(right_values)
    safe_opposite = left_magnitude <= (_FLOAT64_MAX - right_magnitude)
    safe = same_sign | safe_opposite
    safe_differences = np.empty(left_values.size, dtype=np.float64)
    safe_differences[safe] = left_values[safe] - right_values[safe]

    maximum_safe = (
        float(np.max(np.abs(safe_differences[safe]))) if np.any(safe) else 0.0
    )
    unsafe = ~safe
    maximum_unsafe = (
        float(
            max(
                np.max(left_magnitude[unsafe]),
                np.max(right_magnitude[unsafe]),
            )
        )
        if np.any(unsafe)
        else 0.0
    )
    scale = max(maximum_safe, maximum_unsafe)
    if scale == 0.0:
        return 0.0, np.zeros(left_values.size, dtype=np.float64)

    normalized = np.empty(left_values.size, dtype=np.float64)
    # Terms too small relative to ``scale`` cannot affect a float64 reduction,
    # but NumPy still signals when their correctly rounded quotient is
    # subnormal. Permit only that expected scaling underflow.
    with np.errstate(under="ignore"):
        normalized[safe] = safe_differences[safe] / scale
        normalized[unsafe] = left_values[unsafe] / scale - right_values[unsafe] / scale
    return scale, normalized


def _root_mean_square_term(left: np.ndarray, right: np.ndarray) -> _ScaledTerm:
    scale, normalized = _scaled_signed_differences(left, right)
    if scale == 0.0:
        return _ZERO_TERM
    with np.errstate(under="ignore"):
        coefficient = math.sqrt(float(np.sum(normalized * normalized))) / math.sqrt(
            normalized.size
        )
    return _term_from_scaled(scale, coefficient)


def _standard_deviation_term(values: np.ndarray) -> _ScaledTerm:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return _ZERO_TERM
    with np.errstate(under="ignore"):
        normalized = vector / scale
    mean = math.fsum(float(value) for value in normalized) / normalized.size
    with np.errstate(under="ignore"):
        centered = normalized - mean
        coefficient = math.sqrt(float(np.mean(centered * centered)))
    return _term_from_scaled(scale, coefficient)


def _ordinary_gnrmse_value(
    imputed: np.ndarray,
    truth: np.ndarray,
    selected: np.ndarray,
) -> float | None:
    """Return the exact legacy NumPy formula unless it signals or is nonfinite."""

    try:
        with np.errstate(all="raise"):
            difference = imputed[selected] - truth[selected]
            rmse = np.sqrt(np.mean(difference**2))
            truth_sd = max(float(np.std(truth, ddof=0)), 1e-8)
            value = float(rmse / truth_sd)
    except (FloatingPointError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _stable_library_sizes(observed: np.ndarray) -> np.ndarray:
    """Sum each row without overflowing a representable library size."""

    sizes = np.empty(observed.shape[0], dtype=np.float64)
    for index, row in enumerate(observed):
        try:
            with np.errstate(all="raise"):
                legacy = np.sum(row)
        except FloatingPointError:
            legacy = None
        if legacy is not None and np.isfinite(legacy):
            sizes[index] = legacy
            continue

        values = [float(value) for value in row]
        try:
            value = math.fsum(values)
        except OverflowError:
            from fractions import Fraction

            exact = sum(
                (Fraction.from_float(value) for value in values),
                start=Fraction(),
            )
            try:
                value = float(exact)
            except OverflowError as error:
                raise ValueError(
                    "library size is not representable as float64"
                ) from error
        if not math.isfinite(value):
            raise ValueError("library size is not representable as float64")
        sizes[index] = value
    return sizes


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
    try:
        with np.errstate(over="raise", invalid="raise"):
            converted = array.astype(float, copy=False)
    except FloatingPointError as error:
        raise ValueError(
            f"{name} must remain finite when represented as float64"
        ) from error
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
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    squared: bool,
) -> MetricValue:
    n = int(mask.sum())
    if n == 0:
        return _unavailable(0, "no_entries")
    try:
        with np.errstate(all="raise"):
            difference = imputed[mask] - truth[mask]
            value = (
                np.mean(difference * difference)
                if squared
                else np.mean(np.abs(difference))
            )
    except FloatingPointError:
        value = None
    if value is not None and np.isfinite(value):
        return MetricValue(float(value), n, None)

    scale, normalized = _scaled_signed_differences(imputed[mask], truth[mask])
    power = 2 if squared else 1
    with np.errstate(under="ignore"):
        coefficient = float(np.mean(np.abs(normalized) ** power))
    return _metric_from_term(
        _term_from_scaled(scale, coefficient, power=power),
        n,
    )


def _gnrmse(
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> MetricValue:
    if not np.any(mask):
        return _unavailable(0, "no_entries")
    floor = _term_from_float(1e-8)
    values: list[_ScaledTerm] = []
    ordinary_values: list[float] = []
    all_ordinary = True
    for gene in range(truth.shape[1]):
        selected = mask[:, gene]
        if np.any(selected):
            ordinary = _ordinary_gnrmse_value(
                imputed[:, gene],
                truth[:, gene],
                selected,
            )
            if ordinary is not None:
                ordinary_values.append(ordinary)
                values.append(_term_from_float(ordinary))
                continue
            all_ordinary = False
            rmse = _root_mean_square_term(
                imputed[selected, gene],
                truth[selected, gene],
            )
            truth_sd = _standard_deviation_term(truth[:, gene])
            denominator = floor if _term_is_less(truth_sd, floor) else truth_sd
            values.append(_term_ratio(rmse, denominator))
    if all_ordinary:
        return _metric(np.mean(ordinary_values), len(ordinary_values))
    return _metric_from_term(_term_mean(values), len(values))


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

    def stable_correlation(value: np.ndarray) -> np.ndarray | None:
        variables_by_observation = value if rowvar else value.T
        try:
            with np.errstate(all="raise"):
                correlation = np.corrcoef(value, rowvar=rowvar)
        except FloatingPointError:
            correlation = None
        if correlation is not None and np.all(np.isfinite(correlation)):
            return correlation

        scales = np.max(np.abs(variables_by_observation), axis=1)
        try:
            with np.errstate(under="ignore"):
                normalized = variables_by_observation / scales[:, None]
            with np.errstate(all="raise"):
                correlation = np.corrcoef(normalized, rowvar=True)
        except FloatingPointError:
            return None
        return correlation if np.all(np.isfinite(correlation)) else None

    corr_imputed = stable_correlation(selected_imputed)
    corr_truth = stable_correlation(selected_truth)
    if corr_imputed is None or corr_truth is None:
        pair_count = n_variables * (n_variables - 1) // 2
        return _unavailable(pair_count, "nonfinite_correlation"), n_variables
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
    try:
        with np.errstate(all="raise"):
            ordinary = np.empty(n_pairs, dtype=np.float64)
            offset = 0
            for first in range(n_cells - 1):
                count = n_cells - first - 1
                truth_distance = np.linalg.norm(
                    truth[first + 1 :] - truth[first],
                    axis=1,
                )
                imputed_distance = np.linalg.norm(
                    imputed[first + 1 :] - imputed[first],
                    axis=1,
                )
                ordinary[offset : offset + count] = np.abs(
                    imputed_distance - truth_distance
                )
                offset += count
            ordinary_value = np.mean(ordinary)
    except FloatingPointError:
        ordinary_value = None
    if ordinary_value is not None and np.isfinite(ordinary_value):
        return MetricValue(float(ordinary_value), n_pairs, None)

    values: list[_ScaledTerm] = []
    for first in range(n_cells - 1):
        values.extend(
            _euclidean_norm_difference_term(
                imputed[second],
                imputed[first],
                truth[second],
                truth[first],
            )
            for second in range(first + 1, n_cells)
        )
    return _metric_from_term(_term_mean(values), n_pairs)


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
    try:
        with np.errstate(all="raise"):
            per_gene = np.mean(
                np.abs(sorted_imputed - sorted_truth),
                axis=0,
            )
            ordinary_value = np.mean(per_gene)
    except FloatingPointError:
        ordinary_value = None
    if ordinary_value is not None and np.isfinite(ordinary_value):
        return MetricValue(float(ordinary_value), truth.shape[1], None)

    scale, normalized = _scaled_signed_differences(
        sorted_imputed,
        sorted_truth,
    )
    return _metric_from_term(
        _term_from_scaled(scale, float(np.mean(np.abs(normalized)))),
        truth.shape[1],
    )


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
        result[f"mse{suffix}"] = _error_metric(
            imputed_array,
            truth_array,
            mask,
            squared=True,
        )
        result[f"mae{suffix}"] = _error_metric(
            imputed_array,
            truth_array,
            mask,
            squared=False,
        )
        result[f"gnrmse{suffix}"] = _gnrmse(imputed_array, truth_array, mask)

    try:
        with np.errstate(all="raise"):
            ordinary_mean = np.mean(
                np.abs(np.mean(imputed_array, axis=0) - np.mean(truth_array, axis=0))
            )
    except FloatingPointError:
        ordinary_mean = None
    if ordinary_mean is not None and np.isfinite(ordinary_mean):
        result["mean_distortion"] = MetricValue(
            float(ordinary_mean),
            truth_array.shape[1],
            None,
        )
    else:
        mean_differences: list[_ScaledTerm] = []
        for gene in range(truth_array.shape[1]):
            mean_scale, normalized_difference = _scaled_signed_differences(
                imputed_array[:, gene],
                truth_array[:, gene],
            )
            mean_coefficient = abs(
                math.fsum(float(value) for value in normalized_difference)
                / normalized_difference.size
            )
            mean_differences.append(_term_from_scaled(mean_scale, mean_coefficient))
        result["mean_distortion"] = _metric_from_term(
            _term_mean(mean_differences),
            truth_array.shape[1],
        )

    try:
        with np.errstate(all="raise"):
            ordinary_variance = np.mean(
                np.abs(
                    np.var(imputed_array, axis=0, ddof=0)
                    - np.var(truth_array, axis=0, ddof=0)
                )
            )
    except FloatingPointError:
        ordinary_variance = None
    if ordinary_variance is not None and np.isfinite(ordinary_variance):
        result["variance_distortion"] = MetricValue(
            float(ordinary_variance),
            truth_array.shape[1],
            None,
        )
    else:
        variance_differences = [
            _variance_difference_fraction(
                imputed_array[:, gene],
                truth_array[:, gene],
            )
            for gene in range(truth_array.shape[1])
        ]
        variance_mean = sum(
            variance_differences,
            start=Fraction(),
        ) / len(variance_differences)
        result["variance_distortion"] = _metric_from_term(
            _term_from_fraction(variance_mean),
            truth_array.shape[1],
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
    library_size = _stable_library_sizes(observed_array)
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
