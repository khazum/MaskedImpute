"""Truth-isolated metrics for the publication benchmark.

All matrices use the AnnData convention (cells by genes) and must already be
on the evaluator's common scale.  This module deliberately does no
normalization, clipping, missing-value replacement, or other sanitization.
Only probabilities entering logarithms are clipped, and that clipping is
local to log-loss computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import heapq
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
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("an unavailable metric requires a reason")
            return
        if isinstance(self.value, (bool, np.bool_)):
            raise TypeError("value must be numeric or None")
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


_FLOAT64_INFO = np.finfo(np.float64)
_LONGDOUBLE_INFO = np.finfo(np.longdouble)
_PAIRWISE_DISTANCE_BLOCK_SIZE = 128
_PAIRWISE_REFINEMENT_BATCH_SIZE = 64
_PAIRWISE_MAX_REFINEMENTS = 1_024
_VARIANCE_GENE_BLOCK_SIZE = 256


def _longdouble_supports_float64_products() -> bool:
    """Whether long double safely spans every product of two binary64 values."""

    float64_minimum_exponent = _FLOAT64_INFO.minexp - _FLOAT64_INFO.nmant
    longdouble_minimum_exponent = _LONGDOUBLE_INFO.minexp - _LONGDOUBLE_INFO.nmant
    return bool(
        _LONGDOUBLE_INFO.nmant >= _FLOAT64_INFO.nmant + 8
        and _LONGDOUBLE_INFO.maxexp > 2 * _FLOAT64_INFO.maxexp
        and longdouble_minimum_exponent < 2 * float64_minimum_exponent
    )


_WIDE_LONGDOUBLE = _longdouble_supports_float64_products()


def _metric_from_high_precision(value: Fraction | Decimal, n: int) -> MetricValue:
    """Round one completed nonnegative estimand directly to float64."""

    if value < 0:
        raise ValueError("metric values must be nonnegative")
    if value == 0:
        return MetricValue(0.0, int(n), None)
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return _unavailable(n, "nonfinite_metric")
    if not math.isfinite(numeric) or numeric == 0.0:
        return _unavailable(n, "nonfinite_metric")
    return MetricValue(numeric, int(n), None)


def _metric_if_interval_rounds_together(
    lower: Fraction | Decimal,
    upper: Fraction | Decimal,
    n: int,
) -> MetricValue | None:
    """Return a metric only when both conservative endpoints round alike."""

    try:
        lower_float = float(lower)
    except (OverflowError, ValueError):
        lower_float = math.inf
    try:
        upper_float = float(upper)
    except (OverflowError, ValueError):
        upper_float = math.inf
    if lower_float != upper_float:
        return None
    if not math.isfinite(lower_float):
        return _unavailable(n, "nonfinite_metric")
    if lower_float == 0.0 and upper > 0:
        return _unavailable(n, "nonfinite_metric")
    return MetricValue(lower_float, int(n), None)


def _exact_difference_power_mean(
    left: np.ndarray,
    right: np.ndarray,
    *,
    power: int,
) -> Fraction:
    """Return an exact mean of absolute binary64 differences."""

    if power not in {1, 2}:  # pragma: no cover - private programming error
        raise AssertionError(power)
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape or left_values.size == 0:
        raise ValueError("exact difference operands must be nonempty and aligned")
    total = Fraction()
    for left_value, right_value in zip(left_values, right_values):
        difference = abs(
            Fraction.from_float(float(left_value))
            - Fraction.from_float(float(right_value))
        )
        total += difference if power == 1 else difference * difference
    return total / left_values.size


def _fraction_population_variance(values: np.ndarray) -> Fraction:
    """Return an exact population variance of finite binary64 values."""

    exact = [
        Fraction.from_float(float(value))
        for value in np.asarray(values, dtype=np.float64).reshape(-1)
    ]
    mean = sum(exact, start=Fraction()) / len(exact)
    return sum(((value - mean) ** 2 for value in exact), start=Fraction()) / len(exact)


def _decimal_sqrt_mean(
    squared_values: list[Fraction],
    *,
    precision: int,
    rounding: str,
) -> Decimal:
    """Evaluate a nonnegative square-root mean with directed rounding."""

    if rounding not in {ROUND_FLOOR, ROUND_CEILING}:  # pragma: no cover
        raise AssertionError(rounding)
    with localcontext() as context:
        context.prec = precision
        context.rounding = rounding
        total = Decimal()
        for value in squared_values:
            if value < 0:  # pragma: no cover - private programming invariant
                raise AssertionError(value)
            radicand = Decimal(value.numerator) / Decimal(value.denominator)
            if radicand == 0:
                root = Decimal()
            else:
                # Decimal.sqrt always rounds half-even, regardless of the
                # context's rounding mode. Step past that rounded result so
                # each endpoint is genuinely outward before accumulation.
                root = radicand.sqrt()
                if rounding == ROUND_FLOOR:
                    root = context.next_minus(root)
                else:
                    root = context.next_plus(root)
            total += root
        return total / Decimal(len(squared_values))


def _exact_gnrmse(
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> MetricValue:
    """Evaluate the completed gene-wise normalized RMSE with one final cast."""

    floor_squared = Fraction.from_float(1e-8) ** 2
    squared_ratios: list[Fraction] = []
    for gene in range(truth.shape[1]):
        selected = mask[:, gene]
        if not np.any(selected):
            continue
        mse = _exact_difference_power_mean(
            imputed[selected, gene],
            truth[selected, gene],
            power=2,
        )
        variance = _fraction_population_variance(truth[:, gene])
        squared_ratios.append(
            mse / (variance if variance >= floor_squared else floor_squared)
        )

    n_genes = len(squared_ratios)
    if not squared_ratios:
        return _unavailable(0, "no_entries")
    if all(value == 0 for value in squared_ratios):
        return MetricValue(0.0, n_genes, None)
    precision = 120
    while precision <= 15_360:
        lower = _decimal_sqrt_mean(
            squared_ratios,
            precision=precision,
            rounding=ROUND_FLOOR,
        )
        upper = _decimal_sqrt_mean(
            squared_ratios,
            precision=precision,
            rounding=ROUND_CEILING,
        )
        metric = _metric_if_interval_rounds_together(lower, upper, n_genes)
        if metric is not None:
            return metric
        precision *= 2
    raise ArithmeticError("gNRMSE square-root mean could not certify float rounding")


def _decimal_add_exact(left: Decimal, right: Decimal) -> Decimal:
    """Add two finite nonnegative Decimal values without context re-rounding."""

    if left == 0:
        return right
    if right == 0:
        return left
    minimum_exponent = min(left.as_tuple().exponent, right.as_tuple().exponent)
    precision = max(left.adjusted(), right.adjusted()) - minimum_exponent + 2
    with localcontext() as context:
        context.prec = precision
        return left + right


def _decimal_multiply_exact(left: Decimal, right: Decimal) -> Decimal:
    """Multiply two finite Decimals without context re-rounding."""

    if left == 0 or right == 0:
        return Decimal()
    precision = len(left.as_tuple().digits) + len(right.as_tuple().digits) + 2
    with localcontext() as context:
        context.prec = precision
        return left * right


def _metric_from_directed_decimal_totals(
    lower_total: Decimal,
    upper_total: Decimal,
    denominator: int,
) -> MetricValue | None:
    """Certify the mean of outward Decimal totals without midpoint rounding."""

    if denominator <= 0:  # pragma: no cover - private programming invariant
        raise AssertionError(denominator)
    precision = 120
    while precision <= 15_360:
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_FLOOR
            lower_mean = lower_total / Decimal(denominator)
            context.rounding = ROUND_CEILING
            upper_mean = upper_total / Decimal(denominator)
        metric = _metric_if_interval_rounds_together(
            lower_mean,
            upper_mean,
            denominator,
        )
        if metric is not None:
            return metric
        # Additional division precision cannot close an interval whose exact
        # endpoints already straddle two float cells.
        if (
            precision
            >= max(
                len(lower_total.as_tuple().digits),
                len(upper_total.as_tuple().digits),
            )
            + len(str(denominator))
            + 8
        ):
            return None
        precision *= 2
    return None


def _longdouble_to_decimal(value: np.longdouble) -> Decimal:
    numerator, denominator = value.as_integer_ratio()
    return Decimal(numerator) / Decimal(denominator)


def _compensated_add(
    total: np.longdouble,
    correction: np.longdouble,
    value: np.longdouble,
) -> tuple[np.longdouble, np.longdouble]:
    """Neumaier-add one nonnegative block total without retaining its entries."""

    updated = total + value
    if abs(total) >= abs(value):
        correction += (total - updated) + value
    else:
        correction += (value - updated) + total
    return updated, correction


def _interval_ambiguity(
    estimate: np.ndarray,
    error: np.ndarray,
    nonzero_scale: np.ndarray,
) -> np.ndarray:
    """Conservatively flag cancellation too close to the arithmetic error."""

    return nonzero_scale & (error > 0) & (estimate <= np.longdouble(128.0) * error)


def _widen_accumulated_interval(
    lower: np.longdouble,
    upper: np.longdouble,
) -> tuple[np.longdouble, np.longdouble]:
    """Cover the residual rounding error after compensated block summation."""

    if lower == 0 and upper == 0:
        return np.longdouble(0.0), np.longdouble(0.0)
    with np.errstate(under="ignore"):
        error = (
            np.longdouble(8.0)
            * np.longdouble(_LONGDOUBLE_INFO.eps)
            * max(abs(lower), abs(upper))
        )
        return (
            max(
                np.nextafter(lower - error, -np.longdouble(np.inf)),
                np.longdouble(0.0),
            ),
            np.nextafter(upper + error, np.longdouble(np.inf)),
        )


def _longdouble_norm_difference_intervals(
    imputed_difference: np.ndarray,
    truth_difference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound a block of norm differences in a wider floating-point format."""

    imputed_values = np.array(
        imputed_difference,
        dtype=np.longdouble,
        copy=True,
        order="C",
        subok=False,
    )
    truth_values = np.array(
        truth_difference,
        dtype=np.longdouble,
        copy=True,
        order="C",
        subok=False,
    )
    if imputed_values.ndim != 2 or truth_values.shape != imputed_values.shape:
        raise ValueError("norm-difference blocks must be aligned matrices")

    scale = np.maximum(
        np.max(np.abs(imputed_values), axis=1),
        np.max(np.abs(truth_values), axis=1),
    )
    nonzero_scale = scale != 0
    divisor = scale[:, None]
    with np.errstate(under="ignore"):
        np.divide(
            imputed_values,
            divisor,
            out=imputed_values,
            where=nonzero_scale[:, None],
        )
        np.divide(
            truth_values,
            divisor,
            out=truth_values,
            where=nonzero_scale[:, None],
        )
        np.square(imputed_values, out=imputed_values)
        np.square(truth_values, out=truth_values)
        imputed_squared_norm = np.sum(
            imputed_values,
            axis=1,
            dtype=np.longdouble,
        )
        truth_squared_norm = np.sum(
            truth_values,
            axis=1,
            dtype=np.longdouble,
        )
        imputed_norm = np.sqrt(imputed_squared_norm)
        truth_norm = np.sqrt(truth_squared_norm)
        denominator = imputed_norm + truth_norm
        estimate = np.zeros(scale.shape, dtype=np.longdouble)
        np.divide(
            np.abs(imputed_squared_norm - truth_squared_norm) * scale,
            denominator,
            out=estimate,
            where=denominator != 0,
        )

    # The fast-axis NumPy reduction is pairwise. This factor covers conversion,
    # division, squaring, pairwise summation, roots, and the final quotient,
    # while remaining narrow enough to distinguish binary64 rounding cells.
    reduction_depth = max(
        1,
        math.ceil(math.log2(max(1, imputed_values.shape[1]))),
    )
    error_factor = np.longdouble(16 + 4 * reduction_depth)
    with np.errstate(under="ignore"):
        error = (
            error_factor
            * np.longdouble(_LONGDOUBLE_INFO.eps)
            * scale
            * (imputed_norm + truth_norm)
        )
        lower = np.maximum(estimate - error, np.longdouble(0.0))
        upper = estimate + error
        lower = np.nextafter(lower, -np.longdouble(np.inf))
        upper = np.nextafter(upper, np.longdouble(np.inf))
        lower = np.maximum(lower, np.longdouble(0.0))
        lower[~nonzero_scale] = np.longdouble(0.0)
        upper[~nonzero_scale] = np.longdouble(0.0)
    ambiguous = _interval_ambiguity(estimate, error, nonzero_scale)
    return lower, upper, ambiguous


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


def _longdouble_variance_difference_intervals(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound per-column population-variance differences in long double."""

    left_values = np.array(
        left,
        dtype=np.longdouble,
        copy=True,
        order="C",
        subok=False,
    )
    right_values = np.array(
        right,
        dtype=np.longdouble,
        copy=True,
        order="C",
        subok=False,
    )
    if left_values.ndim != 2 or right_values.shape != left_values.shape:
        raise ValueError("variance-difference blocks must be aligned matrices")

    scale = np.maximum(
        np.max(np.abs(left_values), axis=0),
        np.max(np.abs(right_values), axis=0),
    )
    nonzero_scale = scale != 0
    with np.errstate(under="ignore"):
        np.divide(
            left_values,
            scale[None, :],
            out=left_values,
            where=nonzero_scale[None, :],
        )
        np.divide(
            right_values,
            scale[None, :],
            out=right_values,
            where=nonzero_scale[None, :],
        )
        left_values -= np.mean(left_values, axis=0, dtype=np.longdouble)
        right_values -= np.mean(right_values, axis=0, dtype=np.longdouble)
        np.square(left_values, out=left_values)
        np.square(right_values, out=right_values)
        left_variance = np.mean(
            left_values,
            axis=0,
            dtype=np.longdouble,
        )
        right_variance = np.mean(
            right_values,
            axis=0,
            dtype=np.longdouble,
        )
        squared_scale = scale * scale
        estimate = np.abs(left_variance - right_variance) * squared_scale

    reduction_depth = max(
        1,
        math.ceil(math.log2(max(1, left_values.shape[0]))),
    )
    # Unlike a relative bound on the two computed variances, this absolute
    # bound also covers normalization, mean formation, centering, squaring,
    # and reduction.  Those steps can dominate when a profile is almost
    # constant and both computed variances are themselves inaccurate.
    error_factor = np.longdouble(32 + 12 * reduction_depth)
    with np.errstate(under="ignore"):
        error = (
            error_factor
            * np.longdouble(_LONGDOUBLE_INFO.eps)
            * squared_scale
            * (np.longdouble(1.0) + left_variance + right_variance)
        )
        lower = np.maximum(estimate - error, np.longdouble(0.0))
        upper = estimate + error
        lower = np.nextafter(lower, -np.longdouble(np.inf))
        upper = np.nextafter(upper, np.longdouble(np.inf))
        lower = np.maximum(lower, np.longdouble(0.0))
        lower[~nonzero_scale] = np.longdouble(0.0)
        upper[~nonzero_scale] = np.longdouble(0.0)
    ambiguous = _interval_ambiguity(estimate, error, nonzero_scale)
    return lower, upper, ambiguous


def _euclidean_norm_difference_decimal(
    imputed_left: np.ndarray,
    imputed_right: np.ndarray,
    truth_left: np.ndarray,
    truth_right: np.ndarray,
    directed_precision: int | None = None,
) -> Decimal | tuple[Decimal, Decimal]:
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
        if directed_precision is not None:
            return Decimal(), Decimal()
        return Decimal()

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
        if directed_precision is not None:
            return Decimal(), Decimal()
        return Decimal()
    imputed_squared_norm = sum(
        (value * value for value in imputed_scaled),
        start=Fraction(),
    )
    truth_squared_norm = sum(
        (value * value for value in truth_scaled),
        start=Fraction(),
    )
    precision = 120 if directed_precision is None else directed_precision
    while precision <= 7_680:
        with localcontext() as context:
            context.prec = precision

            context.rounding = ROUND_FLOOR
            numerator_lower = Decimal(squared_norm_difference.numerator) / Decimal(
                squared_norm_difference.denominator
            )
            scale_lower = Decimal(common_scale.numerator) / Decimal(
                common_scale.denominator
            )
            imputed_radicand_lower = Decimal(imputed_squared_norm.numerator) / Decimal(
                imputed_squared_norm.denominator
            )
            truth_radicand_lower = Decimal(truth_squared_norm.numerator) / Decimal(
                truth_squared_norm.denominator
            )

            context.rounding = ROUND_CEILING
            numerator_upper = Decimal(squared_norm_difference.numerator) / Decimal(
                squared_norm_difference.denominator
            )
            scale_upper = Decimal(common_scale.numerator) / Decimal(
                common_scale.denominator
            )
            imputed_radicand_upper = Decimal(imputed_squared_norm.numerator) / Decimal(
                imputed_squared_norm.denominator
            )
            truth_radicand_upper = Decimal(truth_squared_norm.numerator) / Decimal(
                truth_squared_norm.denominator
            )

            imputed_root_lower = context.next_minus(imputed_radicand_lower.sqrt())
            truth_root_lower = context.next_minus(truth_radicand_lower.sqrt())
            imputed_root_upper = context.next_plus(imputed_radicand_upper.sqrt())
            truth_root_upper = context.next_plus(truth_radicand_upper.sqrt())

            context.rounding = ROUND_FLOOR
            denominator_lower = imputed_root_lower + truth_root_lower
            context.rounding = ROUND_CEILING
            denominator_upper = imputed_root_upper + truth_root_upper

            context.rounding = ROUND_FLOOR
            lower = scale_lower * numerator_lower / denominator_upper
            context.rounding = ROUND_CEILING
            upper = scale_upper * numerator_upper / denominator_lower

            if directed_precision is not None:
                return lower, upper
            lower_float = float(lower)
            upper_float = float(upper)
            if lower_float == upper_float:
                context.rounding = ROUND_FLOOR
                return (lower + upper) / Decimal(2)
        precision *= 2
    raise ArithmeticError("pairwise norm difference could not certify float rounding")


def _exact_pairwise_distance_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
    n_pairs: int,
) -> MetricValue:
    """Portable bounded-memory evaluation when no wider NumPy type exists."""

    precision = 120 + len(str(n_pairs))
    while precision <= 7_680:
        lower_total = Decimal()
        upper_total = Decimal()
        for first in range(truth.shape[0] - 1):
            for second in range(first + 1, truth.shape[0]):
                interval = _euclidean_norm_difference_decimal(
                    imputed[second],
                    imputed[first],
                    truth[second],
                    truth[first],
                    precision,
                )
                if not isinstance(interval, tuple):  # pragma: no cover
                    raise AssertionError("directed pair interval was not returned")
                lower, upper = interval
                lower_total = _decimal_add_exact(lower_total, lower)
                upper_total = _decimal_add_exact(upper_total, upper)
        metric = _metric_from_directed_decimal_totals(
            lower_total,
            upper_total,
            n_pairs,
        )
        if metric is not None:
            return metric
        precision *= 2
    raise ArithmeticError("pairwise mean could not certify float rounding")


def _float64_norm_difference_intervals(
    imputed_difference: np.ndarray,
    truth_difference: np.ndarray,
    *,
    active: np.ndarray,
    exact_zero: np.ndarray,
    exact_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bound normalized norm differences using portable float64 arithmetic."""

    imputed_values = np.array(
        imputed_difference,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    truth_values = np.array(
        truth_difference,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    if (
        imputed_values.ndim != 2
        or truth_values.shape != imputed_values.shape
        or active.shape != imputed_values.shape[:1]
        or exact_zero.shape != active.shape
        or exact_unit.shape != active.shape
    ):
        raise ValueError("portable norm-difference blocks must be aligned")

    with np.errstate(under="ignore"):
        np.square(imputed_values, out=imputed_values)
        np.square(truth_values, out=truth_values)
        imputed_squared_norm = np.sum(
            imputed_values,
            axis=1,
            dtype=np.float64,
        )
        truth_squared_norm = np.sum(
            truth_values,
            axis=1,
            dtype=np.float64,
        )
        imputed_norm = np.sqrt(imputed_squared_norm)
        truth_norm = np.sqrt(truth_squared_norm)
        denominator = imputed_norm + truth_norm
        estimate = np.zeros(active.shape, dtype=np.float64)
        np.divide(
            np.abs(imputed_squared_norm - truth_squared_norm),
            denominator,
            out=estimate,
            where=denominator != 0,
        )

    reduction_depth = max(
        1,
        math.ceil(math.log2(max(1, imputed_values.shape[1]))),
    )
    # Inputs have already been divided by one exact float64 common scale.
    # The absolute term covers that division, subtraction, squaring, pairwise
    # reduction, roots, and the cancellation-preserving quotient.
    error_factor = float(40 + 12 * reduction_depth)
    with np.errstate(under="ignore"):
        error = (
            error_factor * float(_FLOAT64_INFO.eps) * (1.0 + imputed_norm + truth_norm)
        )
        error[exact_zero] = 0.0
        lower = np.maximum(estimate - error, 0.0)
        upper = estimate + error
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
        lower = np.maximum(lower, 0.0)
        lower[exact_zero] = 0.0
        upper[exact_zero] = 0.0
        estimate[exact_unit] = 1.0
        lower[exact_unit] = 1.0
        upper[exact_unit] = 1.0
    ambiguous = active & ~exact_zero & ((error > 0.0) & (estimate <= 128.0 * error))
    ambiguous[exact_unit] = False
    return estimate, lower, upper, ambiguous


def _float64_compensated_add(
    total: float,
    correction: float,
    value: float,
) -> tuple[float, float]:
    """Neumaier-add one finite nonnegative block total."""

    updated = total + value
    if abs(total) >= abs(value):
        correction += (total - updated) + value
    else:
        correction += (value - updated) + total
    return updated, correction


def _float64_accumulated_interval(
    lower: float,
    lower_correction: float,
    upper: float,
    upper_correction: float,
) -> tuple[Decimal, Decimal]:
    """Convert a compensated binary64 interval to exact outward bounds."""

    lower_estimate = max(lower + lower_correction, 0.0)
    upper_estimate = max(upper + upper_correction, lower_estimate)
    magnitude = max(abs(lower_estimate), abs(upper_estimate))
    rounding_error = 8.0 * float(_FLOAT64_INFO.eps) * magnitude
    lower_bound = max(
        math.nextafter(lower_estimate - rounding_error, -math.inf),
        0.0,
    )
    upper_bound = math.nextafter(upper_estimate + rounding_error, math.inf)
    return Decimal.from_float(lower_bound), Decimal.from_float(upper_bound)


def _portable_pairwise_blocks(
    imputed: np.ndarray,
    truth: np.ndarray,
    imputed_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    common_scale: float,
):
    """Yield portable interval blocks together with their original row indices."""

    for first in range(truth.shape[0] - 1):
        for start in range(
            first + 1,
            truth.shape[0],
            _PAIRWISE_DISTANCE_BLOCK_SIZE,
        ):
            stop = min(
                start + _PAIRWISE_DISTANCE_BLOCK_SIZE,
                truth.shape[0],
            )
            with np.errstate(under="ignore"):
                imputed_difference = imputed_scaled[start:stop] - imputed_scaled[first]
                truth_difference = truth_scaled[start:stop] - truth_scaled[first]
            active = np.any(
                (imputed[start:stop] != imputed[first])
                | (truth[start:stop] != truth[first]),
                axis=1,
            )
            exact_zero = np.all(
                (imputed[start:stop] == truth[start:stop])
                & (imputed[first] == truth[first]),
                axis=1,
            )
            imputed_changed = imputed[start:stop] != imputed[first]
            truth_changed = truth[start:stop] != truth[first]
            imputed_scale_axis = (
                (imputed[start:stop] == 0.0) & (np.abs(imputed[first]) == common_scale)
            ) | (
                (imputed[first] == 0.0) & (np.abs(imputed[start:stop]) == common_scale)
            )
            truth_scale_axis = (
                (truth[start:stop] == 0.0) & (np.abs(truth[first]) == common_scale)
            ) | ((truth[first] == 0.0) & (np.abs(truth[start:stop]) == common_scale))
            exact_unit = (
                (
                    (np.count_nonzero(imputed_changed, axis=1) == 1)
                    & np.all(~imputed_changed | imputed_scale_axis, axis=1)
                    & ~np.any(truth_changed, axis=1)
                )
                | (
                    (np.count_nonzero(truth_changed, axis=1) == 1)
                    & np.all(~truth_changed | truth_scale_axis, axis=1)
                    & ~np.any(imputed_changed, axis=1)
                )
            ) & active
            _, lower, upper, ambiguous = _float64_norm_difference_intervals(
                imputed_difference,
                truth_difference,
                active=active,
                exact_zero=exact_zero,
                exact_unit=exact_unit,
            )
            yield first, start, lower, upper, ambiguous


def _portable_pairwise_interval(
    imputed: np.ndarray,
    truth: np.ndarray,
    imputed_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    common_scale: float,
) -> tuple[Decimal, Decimal, Decimal, Decimal, int, Decimal]:
    """Stream aggregate pair bounds and exact cancellation-only contributions."""

    lower_total = 0.0
    lower_correction = 0.0
    upper_total = 0.0
    upper_correction = 0.0
    fixed_total = Decimal()
    refinable_count = 0
    exact_lower_total = Decimal()
    exact_upper_total = Decimal()
    directed_precision = 120 + len(str(truth.shape[0]))
    for first, start, lower, upper, ambiguous in _portable_pairwise_blocks(
        imputed,
        truth,
        imputed_scaled,
        truth_scaled,
        common_scale,
    ):
        safe = ~ambiguous
        exact_unit = safe & (lower == 1.0) & (upper == 1.0)
        fixed_total += Decimal(int(np.count_nonzero(exact_unit)))
        approximate = safe & ~exact_unit
        if np.any(approximate):
            fixed = approximate & (upper == lower)
            fixed_total += sum(
                (Decimal.from_float(float(value)) for value in lower[fixed]),
                start=Decimal(),
            )
            ranged = approximate & ~fixed
            refinable_count += int(np.count_nonzero(ranged))
            lower_block = math.fsum(float(value) for value in lower[ranged])
            upper_block = math.fsum(float(value) for value in upper[ranged])
            lower_total, lower_correction = _float64_compensated_add(
                lower_total,
                lower_correction,
                lower_block,
            )
            upper_total, upper_correction = _float64_compensated_add(
                upper_total,
                upper_correction,
                upper_block,
            )
        for offset in np.flatnonzero(ambiguous):
            second = start + int(offset)
            interval = _euclidean_norm_difference_decimal(
                imputed[second],
                imputed[first],
                truth[second],
                truth[first],
                directed_precision,
            )
            if not isinstance(interval, tuple):  # pragma: no cover
                raise AssertionError("directed pair interval was not returned")
            exact_lower, exact_upper = interval
            exact_lower_total = _decimal_add_exact(
                exact_lower_total,
                exact_lower,
            )
            exact_upper_total = _decimal_add_exact(
                exact_upper_total,
                exact_upper,
            )

    if lower_total == 0.0 and upper_total == 0.0:
        lower_bound = upper_bound = fixed_total
    else:
        lower_bound, upper_bound = _float64_accumulated_interval(
            lower_total,
            lower_correction,
            upper_total,
            upper_correction,
        )
        lower_bound += fixed_total
        upper_bound += fixed_total
    return (
        lower_bound,
        upper_bound,
        exact_lower_total,
        exact_upper_total,
        refinable_count,
        fixed_total,
    )


def _portable_pairwise_refinement_candidates(
    imputed: np.ndarray,
    truth: np.ndarray,
    imputed_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    common_scale: float,
) -> list[tuple[float, float, int, int, float]]:
    """Retain only the most influential unresolved intervals in a second pass."""

    candidates: list[tuple[float, float, int, int, float]] = []
    for first, start, lower, upper, ambiguous in _portable_pairwise_blocks(
        imputed,
        truth,
        imputed_scaled,
        truth_scaled,
        common_scale,
    ):
        for offset in np.flatnonzero(~ambiguous & ~((lower == 1.0) & (upper == 1.0))):
            lower_value = float(lower[offset])
            upper_value = float(upper[offset])
            width = upper_value - lower_value
            if width <= 0.0:
                continue
            candidate = (
                width,
                upper_value,
                first,
                start + int(offset),
                lower_value,
            )
            if len(candidates) < _PAIRWISE_MAX_REFINEMENTS:
                heapq.heappush(candidates, candidate)
            elif candidate > candidates[0]:
                heapq.heapreplace(candidates, candidate)
    return sorted(candidates, reverse=True)


def _portable_pairwise_certified_metric(
    lower_dimensionless: Decimal,
    upper_dimensionless: Decimal,
    exact_lower_total: Decimal,
    exact_upper_total: Decimal,
    common_scale: float,
    n_pairs: int,
) -> MetricValue | None:
    """Certify one binary64 rounding cell from exact Decimal interval endpoints."""

    scale = Decimal.from_float(common_scale)
    lower_total = _decimal_add_exact(
        exact_lower_total,
        _decimal_multiply_exact(lower_dimensionless, scale),
    )
    upper_total = _decimal_add_exact(
        exact_upper_total,
        _decimal_multiply_exact(upper_dimensionless, scale),
    )
    return _metric_from_directed_decimal_totals(
        lower_total,
        upper_total,
        n_pairs,
    )


def _portable_pairwise_distance_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
    n_pairs: int,
) -> MetricValue:
    """Use bounded float64 blocks and exact arithmetic only for ambiguity."""

    common_scale = float(
        max(
            np.max(np.abs(imputed)),
            np.max(np.abs(truth)),
        )
    )
    if common_scale == 0.0:
        return MetricValue(0.0, n_pairs, None)
    with np.errstate(under="ignore"):
        imputed_scaled = imputed / common_scale
        truth_scaled = truth / common_scale

    with localcontext() as context:
        context.prec = 120 + len(str(n_pairs))
        (
            lower_total,
            upper_total,
            exact_lower_total,
            exact_upper_total,
            refinable_count,
            fixed_total,
        ) = _portable_pairwise_interval(
            imputed,
            truth,
            imputed_scaled,
            truth_scaled,
            common_scale,
        )
        metric = _portable_pairwise_certified_metric(
            lower_total,
            upper_total,
            exact_lower_total,
            exact_upper_total,
            common_scale,
            n_pairs,
        )
        if metric is not None:
            return metric

        # A bounded second pass keeps only the widest or largest unresolved
        # intervals.  Refine those pairs exactly in small batches and certify
        # the completed mean after every batch.
        candidates = _portable_pairwise_refinement_candidates(
            imputed,
            truth,
            imputed_scaled,
            truth_scaled,
            common_scale,
        )
        refined_count = 0
        for start in range(0, len(candidates), _PAIRWISE_REFINEMENT_BATCH_SIZE):
            for _, upper, first, second, lower in candidates[
                start : start + _PAIRWISE_REFINEMENT_BATCH_SIZE
            ]:
                interval = _euclidean_norm_difference_decimal(
                    imputed[second],
                    imputed[first],
                    truth[second],
                    truth[first],
                    context.prec,
                )
                if not isinstance(interval, tuple):  # pragma: no cover
                    raise AssertionError("directed pair interval was not returned")
                exact_lower, exact_upper = interval
                exact_lower_total = _decimal_add_exact(
                    exact_lower_total,
                    exact_lower,
                )
                exact_upper_total = _decimal_add_exact(
                    exact_upper_total,
                    exact_upper,
                )
                lower_total -= Decimal.from_float(lower)
                upper_total -= Decimal.from_float(upper)
                refined_count += 1
            if refined_count == refinable_count:
                lower_total = fixed_total
                upper_total = fixed_total
            else:
                lower_total = max(lower_total, Decimal())
                upper_total = max(upper_total, lower_total)
            metric = _portable_pairwise_certified_metric(
                lower_total,
                upper_total,
                exact_lower_total,
                exact_upper_total,
                common_scale,
                n_pairs,
            )
            if metric is not None:
                return metric

    # Last bounded-memory correctness path. It is reached only after aggregate
    # interval certification and the capped ambiguity refinement are exhausted.
    return _exact_pairwise_distance_distortion(imputed, truth, n_pairs)


def _wide_pairwise_distance_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
    n_pairs: int,
) -> MetricValue:
    """Evaluate unsafe pairs in blocks, escalating only ambiguous pairs."""

    imputed_wide = np.asarray(imputed, dtype=np.longdouble)
    truth_wide = np.asarray(truth, dtype=np.longdouble)
    lower_total = np.longdouble(0.0)
    lower_correction = np.longdouble(0.0)
    upper_total = np.longdouble(0.0)
    upper_correction = np.longdouble(0.0)
    with localcontext() as context:
        context.prec = 120 + len(str(n_pairs))
        exact_lower_total = Decimal()
        exact_upper_total = Decimal()
        for first in range(truth.shape[0] - 1):
            for start in range(
                first + 1,
                truth.shape[0],
                _PAIRWISE_DISTANCE_BLOCK_SIZE,
            ):
                stop = min(
                    start + _PAIRWISE_DISTANCE_BLOCK_SIZE,
                    truth.shape[0],
                )
                imputed_difference = imputed_wide[start:stop] - imputed_wide[first]
                truth_difference = truth_wide[start:stop] - truth_wide[first]
                lower, upper, ambiguous = _longdouble_norm_difference_intervals(
                    imputed_difference,
                    truth_difference,
                )
                safe = ~ambiguous
                if np.any(safe):
                    with np.errstate(under="ignore"):
                        lower_block = np.sum(
                            lower[safe],
                            dtype=np.longdouble,
                        )
                        upper_block = np.sum(
                            upper[safe],
                            dtype=np.longdouble,
                        )
                        lower_total, lower_correction = _compensated_add(
                            lower_total,
                            lower_correction,
                            lower_block,
                        )
                        upper_total, upper_correction = _compensated_add(
                            upper_total,
                            upper_correction,
                            upper_block,
                        )
                for offset in np.flatnonzero(ambiguous):
                    second = start + int(offset)
                    interval = _euclidean_norm_difference_decimal(
                        imputed[second],
                        imputed[first],
                        truth[second],
                        truth[first],
                        context.prec,
                    )
                    if not isinstance(interval, tuple):  # pragma: no cover
                        raise AssertionError("directed pair interval was not returned")
                    exact_lower, exact_upper = interval
                    exact_lower_total = _decimal_add_exact(
                        exact_lower_total,
                        exact_lower,
                    )
                    exact_upper_total = _decimal_add_exact(
                        exact_upper_total,
                        exact_upper,
                    )

        lower_sum, upper_sum = _widen_accumulated_interval(
            max(
                lower_total + lower_correction,
                np.longdouble(0.0),
            ),
            max(
                upper_total + upper_correction,
                lower_total + lower_correction,
            ),
        )
        combined_lower = _decimal_add_exact(
            exact_lower_total,
            _longdouble_to_decimal(lower_sum),
        )
        combined_upper = _decimal_add_exact(
            exact_upper_total,
            _longdouble_to_decimal(upper_sum),
        )
        metric = _metric_from_directed_decimal_totals(
            combined_lower,
            combined_upper,
            n_pairs,
        )
        if metric is not None:
            return metric

    # The completed mean lies on a binary64 rounding boundary. Re-evaluate
    # incrementally rather than retaining an all-pair exact-value collection.
    return _exact_pairwise_distance_distortion(imputed, truth, n_pairs)


def _exact_variance_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
) -> MetricValue:
    total = sum(
        (
            _variance_difference_fraction(
                imputed[:, gene],
                truth[:, gene],
            )
            for gene in range(truth.shape[1])
        ),
        start=Fraction(),
    )
    return _metric_from_high_precision(
        total / truth.shape[1],
        truth.shape[1],
    )


def _exact_mean_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
) -> MetricValue:
    """Round the joint mean-expression distortion from exact input values."""

    total = Fraction()
    n_cells = truth.shape[0]
    n_genes = truth.shape[1]
    for gene in range(n_genes):
        difference = sum(
            (
                Fraction.from_float(float(imputed[cell, gene]))
                - Fraction.from_float(float(truth[cell, gene]))
                for cell in range(n_cells)
            ),
            start=Fraction(),
        )
        total += abs(difference) / n_cells
    return _metric_from_high_precision(total / n_genes, n_genes)


def _certified_mean_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
    ordinary: float,
) -> MetricValue:
    """Retain the ordinary value only when a joint interval certifies it."""

    n_cells, n_genes = truth.shape
    lower_total = np.longdouble(0.0)
    lower_correction = np.longdouble(0.0)
    upper_total = np.longdouble(0.0)
    upper_correction = np.longdouble(0.0)
    epsilon = np.longdouble(_LONGDOUBLE_INFO.eps)
    reduction_depth = max(1, math.ceil(math.log2(max(1, n_cells))))
    error_factor = np.longdouble(16 + 4 * reduction_depth)
    for gene in range(n_genes):
        left = np.asarray(imputed[:, gene], dtype=np.longdouble)
        right = np.asarray(truth[:, gene], dtype=np.longdouble)
        scale = max(
            np.max(np.abs(left)),
            np.max(np.abs(right)),
        )
        if scale == 0:
            continue
        with np.errstate(under="ignore"):
            left_scaled = left / scale
            right_scaled = right / scale
            difference = left_scaled - right_scaled
            signed_total = np.sum(difference, dtype=np.longdouble)
            estimate = abs(signed_total) * scale / np.longdouble(n_cells)
            absolute_work = np.sum(
                np.abs(left_scaled) + np.abs(right_scaled),
                dtype=np.longdouble,
            )
            error = (
                error_factor
                * epsilon
                * (absolute_work + abs(signed_total))
                * scale
                / np.longdouble(n_cells)
            )
            lower = max(estimate - error, np.longdouble(0.0))
            upper = estimate + error
            lower = max(
                np.nextafter(lower, -np.longdouble(np.inf)),
                np.longdouble(0.0),
            )
            upper = np.nextafter(upper, np.longdouble(np.inf))
        lower_total, lower_correction = _compensated_add(
            lower_total,
            lower_correction,
            lower,
        )
        upper_total, upper_correction = _compensated_add(
            upper_total,
            upper_correction,
            upper,
        )

    lower_sum, upper_sum = _widen_accumulated_interval(
        max(lower_total + lower_correction, np.longdouble(0.0)),
        max(
            upper_total + upper_correction,
            lower_total + lower_correction,
        ),
    )
    with localcontext() as context:
        context.prec = 120 + len(str(n_genes))
        lower_mean = _longdouble_to_decimal(lower_sum) / Decimal(n_genes)
        upper_mean = _longdouble_to_decimal(upper_sum) / Decimal(n_genes)
        certified = _metric_if_interval_rounds_together(
            lower_mean,
            upper_mean,
            n_genes,
        )
    if (
        certified is not None
        and certified.reason is None
        and certified.value == ordinary
    ):
        return MetricValue(float(ordinary), n_genes, None)
    if certified is not None:
        return certified
    return _exact_mean_distortion(imputed, truth)


def _float64_variance_difference_intervals(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bound normalized per-gene variance differences in float64 blocks."""

    left_values = np.array(
        left,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    right_values = np.array(
        right,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    if left_values.ndim != 2 or right_values.shape != left_values.shape:
        raise ValueError("portable variance-difference blocks must be aligned")

    scale = np.maximum(
        np.max(np.abs(left_values), axis=0),
        np.max(np.abs(right_values), axis=0),
    )
    nonzero_scale = scale != 0.0
    with np.errstate(under="ignore"):
        np.divide(
            left_values,
            scale[None, :],
            out=left_values,
            where=nonzero_scale[None, :],
        )
        np.divide(
            right_values,
            scale[None, :],
            out=right_values,
            where=nonzero_scale[None, :],
        )
        left_values -= np.mean(left_values, axis=0, dtype=np.float64)
        right_values -= np.mean(right_values, axis=0, dtype=np.float64)
        np.square(left_values, out=left_values)
        np.square(right_values, out=right_values)
        left_variance = np.mean(left_values, axis=0, dtype=np.float64)
        right_variance = np.mean(right_values, axis=0, dtype=np.float64)
        estimate = np.abs(left_variance - right_variance)

    reduction_depth = max(
        1,
        math.ceil(math.log2(max(1, left_values.shape[0]))),
    )
    error_factor = float(32 + 12 * reduction_depth)
    with np.errstate(under="ignore"):
        error = (
            error_factor
            * float(_FLOAT64_INFO.eps)
            * (1.0 + left_variance + right_variance)
        )
        lower = np.maximum(estimate - error, 0.0)
        upper = estimate + error
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
        lower = np.maximum(lower, 0.0)
        lower[~nonzero_scale] = 0.0
        upper[~nonzero_scale] = 0.0
    ambiguous = nonzero_scale & (error > 0.0) & (estimate <= 128.0 * error)
    return estimate, lower, upper, ambiguous, scale


def _portable_variance_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
) -> MetricValue:
    """Vectorize portable variance estimates and exactly resolve ambiguity."""

    n_genes = truth.shape[1]
    exact_total = Fraction()
    uncertain: list[tuple[int, Fraction, Fraction]] = []
    for start in range(0, n_genes, _VARIANCE_GENE_BLOCK_SIZE):
        stop = min(start + _VARIANCE_GENE_BLOCK_SIZE, n_genes)
        _, lower, upper, ambiguous, scale = _float64_variance_difference_intervals(
            imputed[:, start:stop],
            truth[:, start:stop],
        )
        for offset in np.flatnonzero(~ambiguous):
            if scale[offset] == 0.0 or upper[offset] == 0.0:
                continue
            squared_scale = Fraction.from_float(float(scale[offset])) ** 2
            uncertain.append(
                (
                    start + int(offset),
                    squared_scale * Fraction.from_float(float(lower[offset])),
                    squared_scale * Fraction.from_float(float(upper[offset])),
                )
            )
        for offset in np.flatnonzero(ambiguous):
            gene = start + int(offset)
            exact_total += _variance_difference_fraction(
                imputed[:, gene],
                truth[:, gene],
            )

    lower_total = exact_total + sum(
        (lower for _, lower, _ in uncertain),
        start=Fraction(),
    )
    upper_total = exact_total + sum(
        (upper for _, _, upper in uncertain),
        start=Fraction(),
    )
    metric = _metric_if_interval_rounds_together(
        lower_total / n_genes,
        upper_total / n_genes,
        n_genes,
    )
    if metric is not None:
        return metric

    # Refine only the widest unresolved gene intervals.  A safe approximate
    # gene is not exact-recomputed unless its uncertainty prevents the final
    # binary64 result from being certified.
    for gene, lower, upper in sorted(
        uncertain,
        key=lambda item: item[2] - item[1],
        reverse=True,
    ):
        exact = _variance_difference_fraction(
            imputed[:, gene],
            truth[:, gene],
        )
        lower_total += exact - lower
        upper_total += exact - upper
        metric = _metric_if_interval_rounds_together(
            lower_total / n_genes,
            upper_total / n_genes,
            n_genes,
        )
        if metric is not None:
            return metric
    return _metric_from_high_precision(lower_total / n_genes, n_genes)


def _wide_variance_distortion(
    imputed: np.ndarray,
    truth: np.ndarray,
) -> MetricValue:
    """Vectorize unsafe genes and retain exact arithmetic for ambiguity only."""

    n_genes = truth.shape[1]
    imputed_wide = np.asarray(imputed, dtype=np.longdouble)
    truth_wide = np.asarray(truth, dtype=np.longdouble)
    exact_total = Fraction()
    uncertain: list[tuple[int, Fraction, Fraction]] = []
    for start in range(0, n_genes, _VARIANCE_GENE_BLOCK_SIZE):
        stop = min(start + _VARIANCE_GENE_BLOCK_SIZE, n_genes)
        lower, upper, ambiguous = _longdouble_variance_difference_intervals(
            imputed_wide[:, start:stop],
            truth_wide[:, start:stop],
        )
        safe = ~ambiguous
        for offset in np.flatnonzero(safe):
            if upper[offset] == 0:
                continue
            uncertain.append(
                (
                    start + int(offset),
                    Fraction(*lower[offset].as_integer_ratio()),
                    Fraction(*upper[offset].as_integer_ratio()),
                )
            )
        for offset in np.flatnonzero(ambiguous):
            gene = start + int(offset)
            exact_total += _variance_difference_fraction(
                imputed[:, gene],
                truth[:, gene],
            )

    lower_total = exact_total + sum(
        (lower for _, lower, _ in uncertain),
        start=Fraction(),
    )
    upper_total = exact_total + sum(
        (upper for _, _, upper in uncertain),
        start=Fraction(),
    )
    metric = _metric_if_interval_rounds_together(
        lower_total / n_genes,
        upper_total / n_genes,
        n_genes,
    )
    if metric is not None:
        return metric
    for gene, lower, upper in sorted(
        uncertain,
        key=lambda item: item[2] - item[1],
        reverse=True,
    ):
        exact = _variance_difference_fraction(
            imputed[:, gene],
            truth[:, gene],
        )
        lower_total += exact - lower
        upper_total += exact - upper
        metric = _metric_if_interval_rounds_together(
            lower_total / n_genes,
            upper_total / n_genes,
            n_genes,
        )
        if metric is not None:
            return metric
    return _metric_from_high_precision(lower_total / n_genes, n_genes)


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
    power = 2 if squared else 1
    return _metric_from_high_precision(
        _exact_difference_power_mean(
            imputed[mask],
            truth[mask],
            power=power,
        ),
        n,
    )


def _gnrmse(
    imputed: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
) -> MetricValue:
    if not np.any(mask):
        return _unavailable(0, "no_entries")
    ordinary_values: list[float] = []
    for gene in range(truth.shape[1]):
        selected = mask[:, gene]
        if np.any(selected):
            ordinary = _ordinary_gnrmse_value(
                imputed[:, gene],
                truth[:, gene],
                selected,
            )
            if ordinary is None:
                return _exact_gnrmse(imputed, truth, mask)
            ordinary_values.append(ordinary)
    try:
        with np.errstate(all="raise"):
            ordinary_mean = np.mean(ordinary_values)
    except FloatingPointError:
        return _exact_gnrmse(imputed, truth, mask)
    if np.isfinite(ordinary_mean):
        return MetricValue(
            float(ordinary_mean),
            len(ordinary_values),
            None,
        )
    return _exact_gnrmse(imputed, truth, mask)


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
    if np.array_equal(imputed, truth):
        return MetricValue(0.0, n_pairs, None)
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
        certified = (
            _wide_pairwise_distance_distortion(imputed, truth, n_pairs)
            if _WIDE_LONGDOUBLE
            else _portable_pairwise_distance_distortion(imputed, truth, n_pairs)
        )
        if certified.reason is None and certified.value == float(ordinary_value):
            return MetricValue(float(ordinary_value), n_pairs, None)
        return certified

    if _WIDE_LONGDOUBLE:
        return _wide_pairwise_distance_distortion(imputed, truth, n_pairs)
    return _portable_pairwise_distance_distortion(imputed, truth, n_pairs)


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
    exact = _exact_difference_power_mean(
        sorted_imputed,
        sorted_truth,
        power=1,
    )
    return _metric_from_high_precision(
        exact,
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

    identical_reconstruction = np.array_equal(imputed_array, truth_array)
    ordinary_mean: float | None = None
    if identical_reconstruction:
        result["mean_distortion"] = MetricValue(
            0.0,
            truth_array.shape[1],
            None,
        )
    else:
        try:
            with np.errstate(all="raise"):
                ordinary_mean = np.mean(
                    np.abs(
                        np.mean(imputed_array, axis=0) - np.mean(truth_array, axis=0)
                    )
                )
        except FloatingPointError:
            ordinary_mean = None
    if (
        not identical_reconstruction
        and ordinary_mean is not None
        and np.isfinite(ordinary_mean)
    ):
        result["mean_distortion"] = _certified_mean_distortion(
            imputed_array,
            truth_array,
            float(ordinary_mean),
        )
    elif not identical_reconstruction:
        result["mean_distortion"] = _exact_mean_distortion(
            imputed_array,
            truth_array,
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
        certified_variance = (
            _wide_variance_distortion(imputed_array, truth_array)
            if _WIDE_LONGDOUBLE
            else _portable_variance_distortion(imputed_array, truth_array)
        )
        if certified_variance.reason is None and certified_variance.value == float(
            ordinary_variance
        ):
            result["variance_distortion"] = MetricValue(
                float(ordinary_variance),
                truth_array.shape[1],
                None,
            )
        else:
            result["variance_distortion"] = certified_variance
    else:
        if _WIDE_LONGDOUBLE:
            result["variance_distortion"] = _wide_variance_distortion(
                imputed_array,
                truth_array,
            )
        else:
            result["variance_distortion"] = _portable_variance_distortion(
                imputed_array,
                truth_array,
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
    with np.errstate(under="ignore"):
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
        with np.errstate(under="ignore"):
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
    difference = probability - outcome
    with np.errstate(under="ignore"):
        squared_difference = difference**2
        brier = np.mean(squared_difference)
    if np.any((difference != 0.0) & (squared_difference == 0.0)):
        result["brier"] = _metric_from_high_precision(
            _exact_difference_power_mean(
                probability,
                outcome.astype(np.float64, copy=False),
                power=2,
            ),
            n,
        )
    else:
        result["brier"] = _metric(brier, n)
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
