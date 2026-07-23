from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from fractions import Fraction
import time
import tracemalloc
import warnings

import numpy as np
import pytest

import maskimpute_benchmark.metrics as metrics_module
from maskimpute_benchmark.metrics import (
    MetricValue,
    entry_masks,
    reconstruction_metrics,
    stratified_zero_score_metrics,
    tie_aware_groups,
    zero_score_metrics,
)


class _MaskedArrayProtocol:
    def __init__(self, values, mask):
        self._values = values
        self._mask = mask

    def __array__(self, dtype=None, copy=None):
        return np.ma.array(
            self._values,
            mask=self._mask,
            dtype=dtype,
            copy=False if copy is None else copy,
        )


@pytest.fixture
def reconstruction_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.array(
        [
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 4.0],
            [0.0, 3.0, 2.0],
        ]
    )
    observed = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    imputed = np.array(
        [
            [0.2, 1.0, 1.0],
            [1.0, 0.5, 2.0],
            [0.1, 3.0, 3.0],
        ]
    )
    return imputed, observed, truth


def test_metric_value_is_frozen_and_rejects_invalid_states() -> None:
    metric = MetricValue(1.5, 3, None)
    with pytest.raises(FrozenInstanceError):
        metric.value = 2.0  # type: ignore[misc]

    with pytest.raises(ValueError, match="finite"):
        MetricValue(float("nan"), 1, None)
    with pytest.raises(ValueError, match="non-negative"):
        MetricValue(1.0, -1, None)
    with pytest.raises(ValueError, match="reason"):
        MetricValue(None, 1, None)
    with pytest.raises(ValueError, match="reason must be None"):
        MetricValue(1.0, 1, "not_applicable")


def test_entry_masks_partition_requested_entry_types(
    reconstruction_fixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, observed, truth = reconstruction_fixture
    masks = entry_masks(observed, truth)

    assert set(masks) == {
        "overall",
        "induced_dropout",
        "pre_dropout_zero",
        "non_dropout_nonzero",
        "truth_nonzero",
        "observed_positive",
    }
    assert int(masks["overall"].sum()) == 9
    assert int(masks["induced_dropout"].sum()) == 3
    assert int(masks["pre_dropout_zero"].sum()) == 3
    assert int(masks["non_dropout_nonzero"].sum()) == 3
    assert int(masks["truth_nonzero"].sum()) == 6
    assert int(masks["observed_positive"].sum()) == 3
    assert not np.any(masks["induced_dropout"] & masks["pre_dropout_zero"])


@pytest.mark.parametrize(
    "masked",
    [
        np.ma.array([[0.0, 1.0]], mask=[[False, True]]),
        [_MaskedArrayProtocol([0.0, 1.0], [False, True])],
    ],
)
def test_metric_matrix_boundary_rejects_direct_and_nested_protocol_masks(
    masked: object,
) -> None:
    with pytest.raises(TypeError, match="masked arrays"):
        entry_masks(masked, np.zeros((1, 2)))


def test_metric_matrix_boundary_rejects_values_not_representable_as_float64() -> None:
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("longdouble has no wider finite range on this platform")
    outside_float64 = np.longdouble(np.finfo(np.float64).max) * np.longdouble(2)

    with pytest.raises(ValueError, match="finite"):
        entry_masks(np.array([[outside_float64]], dtype=np.longdouble), [[0.0]])


def test_finite_metric_inputs_keep_overflow_reason_coded() -> None:
    maximum = np.finfo(np.float64).max
    result = reconstruction_metrics(
        np.full((2, 2), maximum),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )

    assert result["mse"] == MetricValue(None, 4, "nonfinite_metric")
    assert all(
        metric.value is None or np.isfinite(metric.value) for metric in result.values()
    )


def test_gnrmse_preserves_representable_subnormal_ratio() -> None:
    maximum = np.finfo(np.float64).max
    truth = np.array([[0.0], [maximum]])
    imputed = np.array([[1.0], [maximum]])

    result = reconstruction_metrics(imputed, truth, truth)

    assert result["gnrmse"].value == pytest.approx(
        7.866824069956793e-309,
        rel=1e-15,
        abs=0.0,
    )
    assert result["gnrmse"].n == 1
    assert result["gnrmse"].reason is None


def test_gnrmse_preserves_tiny_representable_error_with_strict_fp_errors() -> None:
    tiny = np.finfo(np.float64).tiny
    truth = np.zeros((2, 1))
    imputed = np.array([[tiny], [0.0]])

    with np.errstate(all="raise"):
        result = reconstruction_metrics(imputed, truth, truth)

    assert result["gnrmse"] == MetricValue(
        tiny / np.sqrt(2.0) / 1e-8,
        1,
        None,
    )


def test_tiny_nonconstant_correlations_use_a_scaled_route_without_fp_errors() -> None:
    tiny = np.finfo(np.float64).tiny
    truth = tiny * np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0]])
    imputed = tiny * np.array([[1.0, 3.0], [3.0, 2.0], [4.0, 1.0]])

    with np.errstate(all="raise"):
        result = reconstruction_metrics(imputed, truth, truth)

    expected_gene = np.abs(
        np.corrcoef(imputed / tiny, rowvar=False)[0, 1]
        - np.corrcoef(truth / tiny, rowvar=False)[0, 1]
    )
    assert result["corr_err"].value == pytest.approx(
        expected_gene,
        rel=0.0,
        abs=2e-16,
    )
    assert result["corr_err"].n == 1
    assert result["corr_err"].reason is None


def test_mixed_scale_fallbacks_complete_with_strict_fp_errors() -> None:
    maximum = np.finfo(np.float64).max
    tiny = np.finfo(np.float64).tiny
    truth = np.zeros((3, 2))
    imputed = np.array([[maximum, tiny], [0.0, 0.0], [maximum, 2.0 * tiny]])

    with np.errstate(all="raise"):
        result = reconstruction_metrics(imputed, truth, truth)

    assert result["mse"] == MetricValue(None, 6, "nonfinite_metric")
    assert result["mae"].value == pytest.approx(maximum / 3.0)
    assert result["mean_distortion"].value == pytest.approx(maximum / 3.0)
    assert result["mean_gene_wasserstein_distance"].value == pytest.approx(
        maximum / 3.0
    )
    assert result["cell_distance_distortion"].reason is None


def test_safe_mean_keeps_legacy_value_when_variance_requires_scaled_route() -> None:
    value = 1.1 * np.sqrt(np.finfo(np.float64).max)
    truth = np.zeros((2, 1))
    imputed = np.array([[value], [0.0]])
    expected_mean = np.mean(np.abs(np.mean(imputed, axis=0) - np.mean(truth, axis=0)))

    with np.errstate(all="raise"):
        result = reconstruction_metrics(imputed, truth, truth)

    assert result["mean_distortion"] == MetricValue(expected_mean, 1, None)
    assert result["variance_distortion"].reason is None
    assert result["variance_distortion"].value is not None


def test_random_ordinary_reconstruction_values_are_exactly_legacy_numpy() -> None:
    rng = np.random.default_rng(20260723)
    for _ in range(100):
        truth = rng.uniform(0.25, 20.0, size=(5, 4))
        imputed = rng.uniform(0.25, 20.0, size=(5, 4))
        difference = imputed - truth
        result = reconstruction_metrics(imputed, truth, truth)

        expected_mse = np.mean(difference * difference)
        expected_mae = np.mean(np.abs(difference))
        truth_sd = np.maximum(np.std(truth, axis=0, ddof=0), 1e-8)
        expected_gnrmse = np.mean(
            [
                np.sqrt(np.mean(difference[:, gene] ** 2)) / truth_sd[gene]
                for gene in range(truth.shape[1])
            ]
        )
        expected_mean = np.mean(
            np.abs(np.mean(imputed, axis=0) - np.mean(truth, axis=0))
        )
        expected_variance = np.mean(
            np.abs(np.var(imputed, axis=0, ddof=0) - np.var(truth, axis=0, ddof=0))
        )
        expected_wasserstein = np.mean(
            np.mean(
                np.abs(np.sort(imputed, axis=0) - np.sort(truth, axis=0)),
                axis=0,
            )
        )
        pairwise = []
        for first in range(truth.shape[0] - 1):
            truth_distance = np.linalg.norm(
                truth[first + 1 :] - truth[first],
                axis=1,
            )
            imputed_distance = np.linalg.norm(
                imputed[first + 1 :] - imputed[first],
                axis=1,
            )
            pairwise.extend(np.abs(imputed_distance - truth_distance))
        expected_pairwise = np.mean(np.asarray(pairwise))
        upper = np.triu_indices(truth.shape[1], k=1)
        expected_correlation = np.mean(
            np.abs(
                np.corrcoef(imputed, rowvar=False)[upper]
                - np.corrcoef(truth, rowvar=False)[upper]
            )
        )

        assert result["mse"].value == expected_mse
        assert result["mae"].value == expected_mae
        assert result["gnrmse"].value == expected_gnrmse
        assert result["mean_distortion"].value == expected_mean
        assert result["variance_distortion"].value == expected_variance
        assert result["mean_gene_wasserstein_distance"].value == expected_wasserstein
        assert result["cell_distance_distortion"].value == expected_pairwise
        assert result["corr_err"].value == expected_correlation


def test_reconstruction_extremes_are_stable_with_warnings_as_errors() -> None:
    maximum = np.finfo(np.float64).max
    truth = np.array([[-maximum, -maximum], [maximum, maximum]])
    imputed = -truth

    with np.errstate(all="raise"):
        result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    assert result["mae"] == MetricValue(None, 4, "nonfinite_metric")
    assert result["mse"] == MetricValue(None, 4, "nonfinite_metric")
    assert result["gnrmse"].value == pytest.approx(2.0)
    assert result["gnrmse"].n == 2
    assert result["gnrmse"].reason is None
    assert result["mean_distortion"] == MetricValue(0.0, 2, None)
    assert result["variance_distortion"] == MetricValue(0.0, 2, None)
    assert result["mean_gene_wasserstein_distance"] == MetricValue(0.0, 2, None)
    assert result["corr_err"] == MetricValue(0.0, 1, None)
    assert result["cell_distance_distortion"] == MetricValue(0.0, 1, None)


@pytest.mark.parametrize("reverse", [False, True])
def test_variance_distortion_preserves_adjacent_extreme_variance_cancellation(
    reverse: bool,
) -> None:
    x = float.fromhex("0x1.8p+537")
    y = np.nextafter(x, np.inf)
    truth = np.array([[-x], [x]])
    imputed = np.array([[-y], [y]])
    if reverse:
        truth, imputed = imputed, truth

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    assert result["variance_distortion"] == MetricValue(
        float.fromhex("0x1.8p+1023"),
        1,
        None,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_variance_distortion_rounds_exact_subnormal_difference_once(
    reverse: bool,
) -> None:
    x = float.fromhex("0x1p-512")
    y = np.nextafter(x, np.inf)
    truth = np.array([[-x], [x]])
    imputed = np.array([[-y], [y]])
    if reverse:
        truth, imputed = imputed, truth

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    assert result["variance_distortion"] == MetricValue(
        float.fromhex("0x0.0000000000001p-1022"),
        1,
        None,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_variance_interval_covers_centering_error_before_accepting_gene(
    reverse: bool,
) -> None:
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    imputed = np.array(
        [
            [float.fromhex("0x1.ffffffffffffap+0"), 0.0],
            [float.fromhex("0x1.0000000000003p+1"), minimum_subnormal],
            [float.fromhex("0x1.ffffffffffffap+0"), 0.0],
        ]
    )
    truth = np.array(
        [
            [float.fromhex("0x1.0000000000000p+1"), 0.0],
            [float.fromhex("0x1.0000000000002p+1"), minimum_subnormal],
            [float.fromhex("0x1.0000000000000p+1"), 0.0],
        ]
    )
    if reverse:
        imputed, truth = truth, imputed

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    assert result["variance_distortion"] == MetricValue(
        float.fromhex("0x1.c71c71c71c71cp-101"),
        2,
        None,
    )


def test_variance_intervals_contain_fraction_oracle_for_near_constant_genes() -> None:
    rng = np.random.default_rng(20260723)
    left = np.empty((17, 48), dtype=np.float64)
    right = np.empty_like(left)
    for gene in range(left.shape[1]):
        base = float(rng.uniform(0.5, 2.0))
        spacing = np.spacing(base)
        left[:, gene] = base + rng.integers(-4, 5, left.shape[0]) * spacing
        right[:, gene] = base + rng.integers(-4, 5, right.shape[0]) * spacing

    lower, upper, _ = metrics_module._longdouble_variance_difference_intervals(  # type: ignore[attr-defined]
        left,
        right,
    )

    for gene in range(left.shape[1]):
        oracle = metrics_module._variance_difference_fraction(  # type: ignore[attr-defined]
            left[:, gene],
            right[:, gene],
        )
        lower_fraction = Fraction(*lower[gene].as_integer_ratio())
        upper_fraction = Fraction(*upper[gene].as_integer_ratio())
        assert lower_fraction <= oracle <= upper_fraction
        with localcontext() as context:
            context.prec = 320
            oracle_decimal = Decimal(oracle.numerator) / Decimal(oracle.denominator)
            lower_decimal = Decimal(lower_fraction.numerator) / Decimal(
                lower_fraction.denominator
            )
            upper_decimal = Decimal(upper_fraction.numerator) / Decimal(
                upper_fraction.denominator
            )
        assert lower_decimal <= oracle_decimal <= upper_decimal


@pytest.mark.parametrize("reverse", [False, True])
def test_cell_distance_distortion_preserves_adjacent_extreme_norm_cancellation(
    reverse: bool,
) -> None:
    x = float.fromhex("0x1.1999999999999p+1023")
    y = np.nextafter(x, np.inf)
    truth = np.array([[0.0, 0.0], [x, x]])
    imputed = np.array([[0.0, 0.0], [y, y]])
    if reverse:
        truth, imputed = imputed, truth

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    expected = MetricValue(
        float.fromhex("0x1.6a09e667f3bcdp+971"),
        1,
        None,
    )
    assert result["cell_distance_distortion"] == expected
    assert result["pairwise_cell_distance_distortion"] == expected


@pytest.mark.parametrize("reverse", [False, True])
def test_cell_distance_distortion_rounds_exact_subnormal_difference_once(
    reverse: bool,
) -> None:
    t = np.nextafter(0.0, np.inf)
    m = 94_906_265
    n = m * m - 1
    truth = np.array([[0.0, 0.0], [n * t, 0.0]])
    imputed = np.array([[0.0, 0.0], [n * t, m * t]])
    if reverse:
        truth, imputed = imputed, truth

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    expected = MetricValue(t, 1, None)
    assert result["cell_distance_distortion"] == expected
    assert result["pairwise_cell_distance_distortion"] == expected


def _independent_decimal_norm_difference(
    imputed_difference: np.ndarray,
    truth_difference: np.ndarray,
) -> Decimal:
    with localcontext() as context:
        context.prec = 180
        imputed_squared = sum(
            (
                Decimal.from_float(float(value)) * Decimal.from_float(float(value))
                for value in imputed_difference
            ),
            start=Decimal(),
        )
        truth_squared = sum(
            (
                Decimal.from_float(float(value)) * Decimal.from_float(float(value))
                for value in truth_difference
            ),
            start=Decimal(),
        )
        return abs(imputed_squared.sqrt() - truth_squared.sqrt())


def test_extended_pair_intervals_escalate_only_rounding_ambiguous_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not metrics_module._WIDE_LONGDOUBLE:
        pytest.skip("longdouble cannot represent all float64 products")
    maximum = np.finfo(np.float64).max
    x = float.fromhex("0x1.1999999999999p+1023")
    y = np.nextafter(x, np.inf)
    imputed_difference = np.zeros((2, 128), dtype=np.longdouble)
    truth_difference = np.zeros_like(imputed_difference)
    imputed_difference[0, :2] = y
    truth_difference[0, :2] = x
    imputed_difference[1, 0] = maximum

    lower, upper, ambiguous = metrics_module._longdouble_norm_difference_intervals(  # type: ignore[attr-defined]
        imputed_difference,
        truth_difference,
    )

    oracle = _independent_decimal_norm_difference(
        np.asarray(imputed_difference[0], dtype=np.float64),
        np.asarray(truth_difference[0], dtype=np.float64),
    )
    oracle_longdouble = np.longdouble(str(oracle))
    assert lower[0] <= oracle_longdouble <= upper[0]
    assert ambiguous.tolist() == [True, False]
    assert float(lower[1]) == float(upper[1]) == maximum

    truth = np.vstack(
        (
            np.zeros(128),
            np.asarray(truth_difference[0], dtype=np.float64),
        )
    )
    imputed = np.vstack(
        (
            np.zeros(128),
            np.asarray(imputed_difference[0], dtype=np.float64),
        )
    )
    original = metrics_module._euclidean_norm_difference_decimal
    exact_calls = 0

    def counted_exact(*args: np.ndarray) -> Decimal:
        nonlocal exact_calls
        exact_calls += 1
        return original(*args)

    monkeypatch.setattr(
        metrics_module,
        "_euclidean_norm_difference_decimal",
        counted_exact,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = metrics_module._pairwise_distance_distortion(imputed, truth)
    assert result == MetricValue(float(oracle), 1, None)
    assert exact_calls == 1


@pytest.mark.parametrize(
    ("n_cells", "n_genes"),
    [(20, 128), (60, 16), (100, 16)],
)
def test_pairwise_fallback_does_not_exactly_recompute_every_safe_pair(
    monkeypatch: pytest.MonkeyPatch,
    n_cells: int,
    n_genes: int,
) -> None:
    if not metrics_module._WIDE_LONGDOUBLE:
        pytest.skip("longdouble cannot represent all float64 products")
    maximum = np.finfo(np.float64).max
    truth = np.zeros((n_cells, n_genes))
    imputed = truth.copy()
    imputed[0, 0] = maximum
    original = metrics_module._euclidean_norm_difference_decimal
    exact_calls = 0

    def counted_exact(*args: np.ndarray) -> Decimal:
        nonlocal exact_calls
        exact_calls += 1
        return original(*args)

    monkeypatch.setattr(
        metrics_module,
        "_euclidean_norm_difference_decimal",
        counted_exact,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = metrics_module._pairwise_distance_distortion(imputed, truth)

    expected = float(Fraction.from_float(maximum) * 2 / n_cells)
    assert result == MetricValue(
        expected,
        n_cells * (n_cells - 1) // 2,
        None,
    )
    assert exact_calls == 0


def test_large_unsafe_pairwise_fallback_is_bounded_and_vectorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not metrics_module._WIDE_LONGDOUBLE:
        pytest.skip("longdouble cannot represent all float64 products")
    n_cells = 300
    n_genes = 96
    maximum = np.finfo(np.float64).max
    truth = np.zeros((n_cells, n_genes))
    imputed = truth.copy()
    imputed[0, 0] = maximum
    original = metrics_module._euclidean_norm_difference_decimal
    exact_calls = 0

    def counted_exact(*args: np.ndarray) -> Decimal:
        nonlocal exact_calls
        exact_calls += 1
        return original(*args)

    monkeypatch.setattr(
        metrics_module,
        "_euclidean_norm_difference_decimal",
        counted_exact,
    )
    started = time.perf_counter()
    result = metrics_module._pairwise_distance_distortion(imputed, truth)
    elapsed = time.perf_counter() - started

    assert result.value == float(Fraction.from_float(maximum) * 2 / n_cells)
    assert result.reason is None
    assert exact_calls == 0
    assert elapsed < 10.0


def test_portable_exact_pairwise_fallback_does_not_retain_all_pair_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_cells = 160
    n_genes = 16
    maximum = np.finfo(np.float64).max
    truth = np.zeros((n_cells, n_genes))
    imputed = truth.copy()
    imputed[0, 0] = maximum
    monkeypatch.setattr(metrics_module, "_WIDE_LONGDOUBLE", False)

    tracemalloc.start()
    try:
        result = metrics_module._pairwise_distance_distortion(imputed, truth)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.value == float(Fraction.from_float(maximum) * 2 / n_cells)
    assert result.reason is None
    assert peak_bytes < 256_000


@pytest.mark.parametrize(
    ("n_cells", "n_genes", "maximum_seconds"),
    [(300, 96, 10.0), (900, 8, 15.0)],
)
def test_portable_pairwise_fallback_is_vectorized_for_unambiguous_pairs(
    monkeypatch: pytest.MonkeyPatch,
    n_cells: int,
    n_genes: int,
    maximum_seconds: float,
) -> None:
    maximum = np.finfo(np.float64).max
    truth = np.zeros((n_cells, n_genes))
    imputed = truth.copy()
    imputed[0, 0] = maximum
    exact_calls = 0

    def counted_exact(*args: np.ndarray) -> Decimal:
        nonlocal exact_calls
        exact_calls += 1
        raise AssertionError("unambiguous portable pair used exact arithmetic")

    monkeypatch.setattr(metrics_module, "_WIDE_LONGDOUBLE", False)
    monkeypatch.setattr(
        metrics_module,
        "_euclidean_norm_difference_decimal",
        counted_exact,
    )
    started = time.perf_counter()
    result = metrics_module._pairwise_distance_distortion(imputed, truth)
    elapsed = time.perf_counter() - started

    assert result == MetricValue(
        float(Fraction.from_float(maximum) * 2 / n_cells),
        n_cells * (n_cells - 1) // 2,
        None,
    )
    assert exact_calls == 0
    assert elapsed < maximum_seconds


def test_portable_pairwise_fallback_escalates_only_ambiguous_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = float.fromhex("0x1.1999999999999p+1023")
    y = np.nextafter(x, np.inf)
    truth = np.array([[0.0, 0.0], [x, x]])
    imputed = np.array([[0.0, 0.0], [y, y]])
    original = metrics_module._euclidean_norm_difference_decimal
    exact_calls = 0

    def counted_exact(*args: np.ndarray) -> Decimal:
        nonlocal exact_calls
        exact_calls += 1
        return original(*args)

    monkeypatch.setattr(metrics_module, "_WIDE_LONGDOUBLE", False)
    monkeypatch.setattr(
        metrics_module,
        "_euclidean_norm_difference_decimal",
        counted_exact,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = metrics_module._pairwise_distance_distortion(imputed, truth)

    assert result == MetricValue(
        float.fromhex("0x1.6a09e667f3bcdp+971"),
        1,
        None,
    )
    assert exact_calls == 1


def test_portable_pairwise_fallback_preserves_subnormal_rounding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimum_subnormal = np.nextafter(0.0, np.inf)
    m = 94_906_265
    n = m * m - 1
    truth = np.array([[0.0, 0.0], [n * minimum_subnormal, 0.0]])
    imputed = np.array([[0.0, 0.0], [n * minimum_subnormal, m * minimum_subnormal]])
    monkeypatch.setattr(metrics_module, "_WIDE_LONGDOUBLE", False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = metrics_module._pairwise_distance_distortion(imputed, truth)

    assert result == MetricValue(minimum_subnormal, 1, None)


@pytest.mark.parametrize("reverse", [False, True])
def test_portable_variance_fallback_preserves_reviewed_centering_edge(
    monkeypatch: pytest.MonkeyPatch,
    reverse: bool,
) -> None:
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    imputed = np.array(
        [
            [float.fromhex("0x1.ffffffffffffap+0"), 0.0],
            [float.fromhex("0x1.0000000000003p+1"), minimum_subnormal],
            [float.fromhex("0x1.ffffffffffffap+0"), 0.0],
        ]
    )
    truth = np.array(
        [
            [float.fromhex("0x1.0000000000000p+1"), 0.0],
            [float.fromhex("0x1.0000000000002p+1"), minimum_subnormal],
            [float.fromhex("0x1.0000000000000p+1"), 0.0],
        ]
    )
    if reverse:
        imputed, truth = truth, imputed
    monkeypatch.setattr(metrics_module, "_WIDE_LONGDOUBLE", False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="raise"):
            result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    assert result["variance_distortion"] == MetricValue(
        float.fromhex("0x1.c71c71c71c71cp-101"),
        2,
        None,
    )


def test_portable_variance_fallback_escalates_only_ambiguous_genes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_cells = 100
    n_genes = 128
    value = 2.0 * np.sqrt(np.finfo(np.float64).max)
    truth = np.zeros((n_cells, n_genes))
    imputed = truth.copy()
    imputed[0, 0] = value
    original = metrics_module._variance_difference_fraction
    exact_calls = 0

    def counted_exact(*args: np.ndarray) -> Fraction:
        nonlocal exact_calls
        exact_calls += 1
        return original(*args)

    monkeypatch.setattr(metrics_module, "_WIDE_LONGDOUBLE", False)
    monkeypatch.setattr(
        metrics_module,
        "_variance_difference_fraction",
        counted_exact,
    )

    result = reconstruction_metrics(imputed, truth, truth)

    expected = float(
        Fraction.from_float(value) ** 2 * (n_cells - 1) / (n_cells * n_cells * n_genes)
    )
    assert result["variance_distortion"] == MetricValue(expected, n_genes, None)
    assert exact_calls <= 1


def test_variance_fallback_escalates_only_ambiguous_genes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not metrics_module._WIDE_LONGDOUBLE:
        pytest.skip("longdouble cannot represent all float64 products")
    n_cells = 100
    n_genes = 128
    value = 2.0 * np.sqrt(np.finfo(np.float64).max)
    truth = np.zeros((n_cells, n_genes))
    imputed = truth.copy()
    imputed[0, 0] = value
    original = metrics_module._variance_difference_fraction
    exact_calls = 0

    def counted_exact(*args: np.ndarray) -> Fraction:
        nonlocal exact_calls
        exact_calls += 1
        return original(*args)

    monkeypatch.setattr(
        metrics_module,
        "_variance_difference_fraction",
        counted_exact,
    )

    result = reconstruction_metrics(imputed, truth, truth)

    expected = float(
        Fraction.from_float(value) ** 2 * (n_cells - 1) / (n_cells * n_cells * n_genes)
    )
    assert result["variance_distortion"] == MetricValue(expected, n_genes, None)
    assert exact_calls <= 1


def test_reconstruction_returns_representable_mean_after_raw_difference_overflow() -> (
    None
):
    maximum = np.finfo(np.float64).max
    truth = np.array([[-maximum], [0.0], [0.0]])
    imputed = np.array([[maximum], [0.0], [0.0]])

    with np.errstate(all="raise"):
        result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    assert result["mae"].value == pytest.approx(maximum * (2.0 / 3.0))
    assert result["mae"].n == 3
    assert result["mae"].reason is None

    square_root = np.sqrt(maximum)
    squared_truth = np.zeros((2, 1))
    squared_imputed = np.array([[1.1 * square_root], [0.0]])
    with np.errstate(all="raise"):
        squared_result = reconstruction_metrics(
            squared_imputed,
            squared_truth,
            squared_truth,
        )

    assert squared_result["mse"].value == pytest.approx(maximum * (1.1**2 / 2.0))
    assert squared_result["mse"].n == 2
    assert squared_result["mse"].reason is None


def test_gnrmse_uses_representable_rmse_after_one_difference_exceeds_float64() -> None:
    maximum = np.finfo(np.float64).max
    truth = np.array([[-maximum], [0.0], [0.0], [0.0]])
    imputed = np.array([[maximum], [0.0], [0.0], [0.0]])

    with np.errstate(all="raise"):
        result = reconstruction_metrics(imputed, np.zeros_like(truth), truth)

    assert result["gnrmse"].value == pytest.approx(4.0 / np.sqrt(3.0))
    assert result["gnrmse"].n == 1
    assert result["gnrmse"].reason is None


def test_stratified_scores_stabilize_representable_extreme_library_sums() -> None:
    maximum = np.finfo(np.float64).max
    observed = np.array(
        [
            [maximum, maximum, -maximum, 0.0],
            [maximum, -maximum, 1.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ]
    )
    truth = np.zeros_like(observed)
    probability = np.full_like(observed, 0.5)

    with np.errstate(all="raise"):
        result = stratified_zero_score_metrics(probability, observed, truth)

    records = result["library_size_quartiles"]
    assert [(record["lower"], record["upper"]) for record in records] == [
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (maximum, maximum),
    ]


def test_library_strata_preserve_exact_legacy_sum_tie_membership() -> None:
    magnitude = float(2**53)
    observed = np.array(
        [
            [magnitude, 1.0, -magnitude],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )

    with np.errstate(all="raise"):
        library_sizes = metrics_module._stable_library_sizes(observed)
        groups = tie_aware_groups(library_sizes, 4)

    np.testing.assert_array_equal(library_sizes, np.sum(observed, axis=1))
    assert [group.tolist() for group in groups] == [[0, 2], [1], [3]]


def test_tie_aware_groups_requires_nonempty_finite_one_dimensional_values() -> None:
    for values in (
        np.array([]),
        np.zeros((1, 1)),
        np.array([0.0, np.nan]),
    ):
        with pytest.raises(ValueError):
            tie_aware_groups(values, 2)


@pytest.mark.parametrize("maximum_groups", [True, 0, -1, 1.5])
def test_tie_aware_groups_requires_a_positive_integer_maximum(
    maximum_groups: object,
) -> None:
    expected = TypeError if isinstance(maximum_groups, (bool, float)) else ValueError
    with pytest.raises(expected):
        tie_aware_groups(np.array([0.0, 1.0]), maximum_groups)


def test_reconstruction_subset_mse_and_mae_are_hand_calculated(
    reconstruction_fixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    imputed, observed, truth = reconstruction_fixture
    result = reconstruction_metrics(
        imputed,
        observed,
        truth,
        marker_genes=np.array([True, False, True]),
    )

    assert result["mse"] == MetricValue(0.7, 9, None)
    assert result["mae"].value == pytest.approx(4.8 / 9)
    assert result["mse_induced_dropout"] == MetricValue(2.0, 3, None)
    assert result["mae_induced_dropout"].value == pytest.approx(4.0 / 3)
    assert result["mse_pre_dropout_zero"].value == pytest.approx(0.1)
    assert result["mse_pre_dropout_zero"].n == 3
    assert result["mae_pre_dropout_zero"].value == pytest.approx(0.8 / 3)
    assert result["mse_truth_nonzero"] == MetricValue(1.0, 6, None)
    assert result["mae_truth_nonzero"].value == pytest.approx(4.0 / 6)
    assert result["mse_observed_positive"] == MetricValue(0.0, 3, None)
    assert result["mae_observed_positive"] == MetricValue(0.0, 3, None)
    assert result["mse_marker"].value == pytest.approx(5.05 / 6)
    assert result["mae_marker"].value == pytest.approx(3.3 / 6)


def test_reconstruction_result_exposes_protocol_metric_names(
    reconstruction_fixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    imputed, observed, truth = reconstruction_fixture
    result = reconstruction_metrics(imputed, observed, truth)

    assert result["mse_dropout"] == result["mse_induced_dropout"]
    assert result["mae_dropout"] == result["mae_induced_dropout"]
    assert result["gnrmse_dropout"] == result["gnrmse_induced_dropout"]
    assert result["mse_nonzero"] == result["mse_non_dropout_nonzero"]
    assert (
        result["pairwise_cell_distance_distortion"]
        == result["cell_distance_distortion"]
    )


def test_gnrmse_averages_gene_rmse_over_population_truth_sd(
    reconstruction_fixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    imputed, observed, truth = reconstruction_fixture
    result = reconstruction_metrics(imputed, observed, truth)
    expected = np.mean(
        np.sqrt(np.mean((imputed - truth) ** 2, axis=0)) / np.std(truth, axis=0, ddof=0)
    )

    assert result["gnrmse"].value == pytest.approx(expected)
    assert result["gnrmse"].n == 3
    assert result["gnrmse_observed_positive"] == MetricValue(0.0, 3, None)
    assert result["gnrmse_marker"].reason == "marker_genes_not_provided"


def test_gnrmse_includes_constant_truth_genes_with_epsilon_denominator() -> None:
    truth = np.array([[1.0, 0.0], [1.0, 1.0]])
    observed = truth.copy()
    imputed = np.array([[2.0, 0.0], [2.0, 1.0]])

    result = reconstruction_metrics(imputed, observed, truth)

    # Gene 1 contributes 1 / 1e-8 and gene 2 contributes zero.
    assert result["gnrmse"] == MetricValue(50_000_000.0, 2, None)


def test_subset_gnrmse_includes_each_gene_with_a_selected_entry() -> None:
    truth = np.array([[1.0, 0.0], [1.0, 2.0]])
    observed = np.zeros_like(truth)
    imputed = np.array([[2.0, 3.0], [2.0, 2.0]])

    result = reconstruction_metrics(imputed, observed, truth)

    # Induced dropouts are both constant gene-1 entries and gene-2/cell-2.
    # The first gene contributes 1 / 1e-8; the second contributes zero.
    assert result["gnrmse_induced_dropout"] == MetricValue(50_000_000.0, 2, None)
    # The sole truth-zero entry is gene 2/cell 1; full-dataset SD is 1.
    assert result["gnrmse_pre_dropout_zero"] == MetricValue(3.0, 1, None)


def test_reconstruction_distortion_and_false_positive_rate_are_complete(
    reconstruction_fixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    imputed, observed, truth = reconstruction_fixture
    result = reconstruction_metrics(imputed, observed, truth)

    expected_variance = np.mean(
        np.abs(np.var(imputed, axis=0, ddof=0) - np.var(truth, axis=0, ddof=0))
    )
    assert result["mean_distortion"].value == pytest.approx(0.2)
    assert result["mean_distortion"].n == 3
    assert result["variance_distortion"].value == pytest.approx(expected_variance)
    assert result["false_positive_expression"] == MetricValue(1.0, 3, None)


def test_mean_gene_wasserstein_distance_is_hand_calculated() -> None:
    truth = np.array([[0.0, 0.0], [2.0, 4.0]])
    observed = truth.copy()
    imputed = np.array([[1.0, 1.0], [1.0, 3.0]])

    result = reconstruction_metrics(imputed, observed, truth)

    # Each gene has empirical 1-Wasserstein distance 1 on the common scale.
    assert result["mean_gene_wasserstein_distance"] == MetricValue(1.0, 2, None)


def test_mean_gene_wasserstein_distance_ignores_cell_order() -> None:
    truth = np.array([[0.0, 4.0], [2.0, 0.0], [1.0, 3.0]])
    observed = truth.copy()
    imputed = truth[[2, 0, 1]]

    result = reconstruction_metrics(imputed, observed, truth)

    assert result["mean_gene_wasserstein_distance"] == MetricValue(0.0, 2, None)


def test_truth_zero_and_nonzero_estimands_include_ambient_observed_positive() -> None:
    truth = np.array([[0.0, 2.0], [0.0, 3.0]])
    observed = np.array([[5.0, 2.0], [0.0, 0.0]])
    imputed = np.array([[4.0, 2.0], [1.0, 1.0]])

    masks = entry_masks(observed, truth)
    assert masks["pre_dropout_zero"].tolist() == [[True, False], [True, False]]
    assert masks["non_dropout_nonzero"].tolist() == [
        [False, True],
        [False, False],
    ]
    assert masks["observed_positive"].tolist() == [[True, True], [False, False]]

    result = reconstruction_metrics(imputed, observed, truth)
    assert result["mse_pre_dropout_zero"] == MetricValue(8.5, 2, None)
    assert result["false_positive_expression"] == MetricValue(1.0, 2, None)
    assert result["mse_non_dropout_nonzero"] == MetricValue(0.0, 1, None)
    assert result["mse_truth_nonzero"] == MetricValue(2.0, 2, None)
    assert result["mse_observed_positive"] == MetricValue(8.0, 2, None)
    assert result["mse_nonzero"] == result["mse_non_dropout_nonzero"]


def test_corr_err_uses_prespecified_common_genes() -> None:
    truth = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [2.0, 2.0, 1.0]])
    imputed = np.array([[0.0, 2.0, 0.0], [1.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    observed = truth.copy()

    result = reconstruction_metrics(
        imputed,
        observed,
        truth,
        corr_gene_mask=np.array([True, True, False]),
    )

    assert result["n_corr_genes"] == MetricValue(2.0, 2, None)
    assert result["corr_err"] == MetricValue(2.0, 1, None)


def test_corr_err_does_not_drop_a_selected_gene_collapsed_by_a_method() -> None:
    truth = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    observed = truth.copy()
    imputed = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])

    result = reconstruction_metrics(
        imputed,
        observed,
        truth,
        corr_gene_mask=np.array([True, True]),
    )

    assert result["n_corr_genes"] == MetricValue(2.0, 2, None)
    assert result["corr_err"] == MetricValue(None, 2, "constant_gene_profile")


def test_pairwise_cell_distance_distortion_is_hand_calculated() -> None:
    truth = np.array([[0.0, 0.0], [3.0, 4.0]])
    observed = truth.copy()
    imputed = np.zeros_like(truth)

    result = reconstruction_metrics(imputed, observed, truth)

    assert result["cell_distance_distortion"] == MetricValue(5.0, 1, None)
    assert result["cell_correlation_distortion"].reason == "constant_cell_profile"


def test_cell_correlation_distortion_is_hand_calculated() -> None:
    truth = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [2.0, 1.0, 0.0]])
    observed = truth.copy()
    imputed = np.array([[0.0, 1.0, 2.0], [4.0, 2.0, 0.0], [2.0, 1.0, 0.0]])

    result = reconstruction_metrics(imputed, observed, truth)

    # Of three cell-cell correlations, two change from -/+1 to +/-1 by 2.
    assert result["cell_correlation_distortion"] == MetricValue(4.0 / 3.0, 3, None)


def test_constant_and_empty_reconstruction_endpoints_are_reason_coded() -> None:
    truth = np.ones((2, 2))
    observed = truth.copy()
    imputed = truth.copy()
    result = reconstruction_metrics(imputed, observed, truth)

    assert result["mse_induced_dropout"] == MetricValue(None, 0, "no_entries")
    assert result["gnrmse"] == MetricValue(0.0, 2, None)
    assert result["corr_err"].reason == "constant_gene_profile"
    assert result["cell_correlation_distortion"].reason == "constant_cell_profile"
    assert all(
        not isinstance(value, MetricValue)
        or value.value is None
        or np.isfinite(value.value)
        for value in result.values()
    )


@pytest.mark.parametrize(
    ("truth_kind", "reason"),
    [
        ("exact_continuous", "undefined_for_continuous_truth"),
        ("proxy_high_depth", "proxy_truth_not_exact"),
    ],
)
def test_discrete_zero_reconstruction_endpoints_respect_truth_kind(
    reconstruction_fixture: tuple[np.ndarray, np.ndarray, np.ndarray],
    truth_kind: str,
    reason: str,
) -> None:
    imputed, observed, truth = reconstruction_fixture
    result = reconstruction_metrics(imputed, observed, truth, truth_kind=truth_kind)

    assert result["mse"].reason is None
    assert result["mse_pre_dropout_zero"].reason == reason
    assert result["mae_pre_dropout_zero"].reason == reason
    assert result["gnrmse_pre_dropout_zero"].reason == reason
    assert result["false_positive_expression"].reason == reason


def test_orthogonal_reconstruction_truth_is_explicitly_unavailable() -> None:
    observed = np.array([[0.0, 1.0], [2.0, 0.0]])
    result = reconstruction_metrics(
        observed.copy(), observed, None, truth_kind="orthogonal_only"
    )

    for name, metric in result.items():
        if isinstance(metric, MetricValue) and name != "n_corr_genes":
            assert metric.value is None
            assert metric.reason == "truth_unavailable"


def test_score_metrics_with_ties_are_hand_calculated_and_calibrated() -> None:
    # Each score group exactly matches its empirical event frequency.  The
    # standard logistic calibration model therefore has intercept 0, slope 1.
    observed = np.zeros((2, 4))
    truth = np.array([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    probability = np.array([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.75, 0.75]])
    result = zero_score_metrics(probability, observed, truth, n_bins=2)

    assert result["n"] == 8
    assert result["auroc"] == MetricValue(0.75, 8, None)
    assert result["average_precision"] == MetricValue(0.6875, 8, None)
    assert result["brier"] == MetricValue(0.1875, 8, None)
    expected_log_loss = -(6 * np.log(0.75) + 2 * np.log(0.25)) / 8
    assert result["log_loss"].value == pytest.approx(expected_log_loss)
    assert result["calibration_intercept"].value == pytest.approx(0.0, abs=1e-8)
    assert result["calibration_slope"].value == pytest.approx(1.0, abs=1e-8)
    assert result["ece"] == MetricValue(0.0, 8, None)

    assert len(result["reliability_bins"]) == 2
    low, high = result["reliability_bins"]
    assert (low["n"], low["mean_prediction"], low["observed_fraction"]) == (
        4,
        0.25,
        0.25,
    )
    assert (high["n"], high["mean_prediction"], high["observed_fraction"]) == (
        4,
        0.75,
        0.75,
    )
    assert low["wilson_lower"] == pytest.approx(0.0455872608)
    assert low["wilson_upper"] == pytest.approx(0.6993581574)


def test_average_rank_auroc_and_average_precision_handle_mixed_tie() -> None:
    observed = np.zeros((2, 3))
    truth = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    probability = np.array([[0.8, 0.8, 0.4], [0.3, 0.2, 0.1]])

    result = zero_score_metrics(probability, observed, truth, n_bins=2)

    assert result["auroc"].value == pytest.approx(5.5 / 9)
    assert result["average_precision"].value == pytest.approx(53 / 90)


def test_score_metrics_ignore_probabilities_at_observed_positive_entries() -> None:
    observed = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    truth = np.array([[0.0, 2.0, 2.0], [0.0, 1.0, 3.0]])
    p_one = np.array([[0.8, 0.3, 0.01], [0.6, 0.2, 0.01]])
    p_two = p_one.copy()
    p_two[:, 2] = 0.99

    one = zero_score_metrics(p_one, observed, truth, n_bins=2)
    two = zero_score_metrics(p_two, observed, truth, n_bins=2)

    assert one == two
    assert one["n"] == int((observed == 0).sum()) == 4


def test_empty_and_single_class_score_cases_have_explicit_reasons() -> None:
    observed_positive = np.ones((2, 2))
    empty = zero_score_metrics(
        np.full((2, 2), 0.5), observed_positive, observed_positive
    )
    assert empty["n"] == 0
    assert empty["brier"] == MetricValue(None, 0, "no_observed_zeros")
    assert empty["reliability_bins"] == []

    observed_zero = np.zeros((2, 2))
    one_class = zero_score_metrics(
        np.full((2, 2), 0.8), observed_zero, observed_zero, n_bins=2
    )
    assert one_class["auroc"].reason == "single_class"
    assert one_class["average_precision"].reason == "single_class"
    assert one_class["calibration_intercept"].reason == "single_class"
    assert one_class["brier"].value == pytest.approx(0.04)
    assert one_class["brier"].n == 4
    assert one_class["log_loss"].value == pytest.approx(-np.log(0.8))
    assert one_class["ece"].value == pytest.approx(0.2)


def test_constant_scores_keep_proper_scores_but_reason_code_calibration() -> None:
    observed = np.zeros((2, 2))
    truth = np.array([[0.0, 1.0], [0.0, 1.0]])
    result = zero_score_metrics(np.full((2, 2), 0.5), observed, truth)

    assert result["auroc"] == MetricValue(0.5, 4, None)
    assert result["average_precision"] == MetricValue(0.5, 4, None)
    assert result["brier"] == MetricValue(0.25, 4, None)
    assert result["calibration_intercept"].reason == "constant_predictions"
    assert result["calibration_slope"].reason == "constant_predictions"
    assert len(result["reliability_bins"]) == 1
    assert result["reliability_bins"][0]["n"] == 4


def test_reliability_bins_do_not_split_ties_and_are_permutation_invariant() -> None:
    observed = np.zeros((2, 4))
    truth = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    probability = np.array([[0.1, 0.2, 0.2, 0.2], [0.2, 0.8, 0.8, 0.9]])

    original = zero_score_metrics(probability, observed, truth, n_bins=3)
    permutation = np.array([3, 0, 2, 1])
    permuted = zero_score_metrics(
        probability[:, permutation],
        observed[:, permutation],
        truth[:, permutation],
        n_bins=3,
    )

    assert original == permuted
    assert [record["n"] for record in original["reliability_bins"]] == [1, 4, 3]
    score_to_bin: dict[float, set[int]] = {}
    for record in original["reliability_bins"]:
        score_to_bin.setdefault(record["mean_prediction"], set()).add(record["bin"])
    assert all(len(bin_ids) == 1 for bin_ids in score_to_bin.values())


@pytest.mark.parametrize(
    ("truth_kind", "reason"),
    [
        ("exact_continuous", "undefined_for_continuous_truth"),
        ("proxy_high_depth", "proxy_truth_not_exact"),
        ("orthogonal_only", "truth_unavailable"),
    ],
)
def test_zero_score_truth_kind_reason_codes(truth_kind: str, reason: str) -> None:
    observed = np.zeros((2, 2))
    truth = None if truth_kind == "orthogonal_only" else np.eye(2)
    result = zero_score_metrics(
        np.full((2, 2), 0.5), observed, truth, truth_kind=truth_kind
    )

    assert result["n"] == 4
    for name in (
        "auroc",
        "average_precision",
        "brier",
        "log_loss",
        "calibration_intercept",
        "calibration_slope",
        "ece",
    ):
        assert result[name] == MetricValue(None, 4, reason)
    assert result["reliability_bins"] == []


@pytest.mark.parametrize(
    "bad_probability",
    [-0.01, 1.01, np.nan, np.inf],
)
def test_probability_validation_rejects_invalid_values(bad_probability: float) -> None:
    observed = np.zeros((1, 2))
    truth = np.array([[0.0, 1.0]])
    p = np.array([[0.5, bad_probability]])
    with pytest.raises(ValueError):
        zero_score_metrics(p, observed, truth)


@pytest.mark.parametrize(
    "function_name",
    ["entry_masks", "reconstruction_metrics", "zero_score_metrics"],
)
def test_matrix_inputs_must_be_two_dimensional_shape_matched_and_finite(
    function_name: str,
) -> None:
    observed = np.zeros((2, 2))
    truth = np.zeros((2, 2))
    imputed = np.zeros((2, 2))
    function = {
        "entry_masks": lambda: entry_masks(observed, truth[:, :1]),
        "reconstruction_metrics": lambda: reconstruction_metrics(
            imputed.ravel(), observed, truth
        ),
        "zero_score_metrics": lambda: zero_score_metrics(
            np.array([[0.5, np.nan], [0.5, 0.5]]), observed, truth
        ),
    }[function_name]
    with pytest.raises(ValueError):
        function()


def test_stratified_scores_return_explicit_disjoint_strata() -> None:
    observed = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 2.0, 0.0],
        ]
    )
    truth = np.array(
        [
            [0.0, 1.0, 2.0, 5.0],
            [1.0, 0.0, 3.0, 5.0],
            [1.0, 1.0, 0.0, 4.0],
            [1.0, 1.0, 2.0, 0.0],
        ]
    )
    p = np.full((4, 4), 0.4)
    result = stratified_zero_score_metrics(p, observed, truth, n_bins=2)

    assert set(result) == {"library_size_quartiles", "truth_expression_bins"}
    assert [record["label"] for record in result["library_size_quartiles"]] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]
    assert [record["label"] for record in result["truth_expression_bins"]] == [
        "[0,1)",
        "[1,2)",
        "[2,4)",
        "[4,inf)",
    ]
    n_zeros = int((observed == 0).sum())
    assert sum(r["n"] for r in result["library_size_quartiles"]) == n_zeros
    assert sum(r["n"] for r in result["truth_expression_bins"]) == n_zeros
    for stratum_type, records in result.items():
        for record in records:
            assert record["stratum_type"] == stratum_type
            assert record["metrics"]["n"] == record["n"]
            for boundary in (record["lower"], record["upper"]):
                assert boundary is None or np.isfinite(boundary)
    assert result["truth_expression_bins"][-1]["upper"] is None


def test_library_quartiles_never_split_ties_and_are_permutation_invariant() -> None:
    observed = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
        ]
    )
    truth = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [4.0, 1.0],
        ]
    )
    probability = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.6, 0.4],
        ]
    )
    original = stratified_zero_score_metrics(probability, observed, truth, n_bins=2)
    permutation = np.array([5, 2, 0, 4, 1, 3])
    permuted = stratified_zero_score_metrics(
        probability[permutation],
        observed[permutation],
        truth[permutation],
        n_bins=2,
    )

    assert original == permuted
    records = original["library_size_quartiles"]
    nonempty = [record for record in records if record["n"] > 0]
    for left, right in zip(nonempty, nonempty[1:]):
        assert left["upper"] < right["lower"]


def test_equal_library_sizes_form_one_effective_quartile() -> None:
    observed = np.zeros((4, 2))
    truth = np.tile(np.array([[0.0, 1.0]]), (4, 1))
    probability = np.full((4, 2), 0.5)

    result = stratified_zero_score_metrics(probability, observed, truth)
    records = result["library_size_quartiles"]

    assert [record["n"] for record in records] == [8, 0, 0, 0]
    assert records[0]["lower"] == records[0]["upper"] == 0.0
    assert all(record["lower"] is None for record in records[1:])


def test_stratified_orthogonal_scores_emit_truth_unavailable_records() -> None:
    observed = np.zeros((2, 2))
    result = stratified_zero_score_metrics(
        np.full((2, 2), 0.5),
        observed,
        None,
        truth_kind="orthogonal_only",
    )

    assert len(result["library_size_quartiles"]) == 4
    assert len(result["truth_expression_bins"]) == 4
    assert all(
        record["metrics"]["brier"].reason == "truth_unavailable"
        for records in result.values()
        for record in records
    )
