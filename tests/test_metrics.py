from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from maskimpute_benchmark.metrics import (
    MetricValue,
    entry_masks,
    reconstruction_metrics,
    stratified_zero_score_metrics,
    zero_score_metrics,
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
    assert result["pairwise_cell_distance_distortion"] == result[
        "cell_distance_distortion"
    ]


def test_gnrmse_averages_gene_rmse_over_population_truth_sd(
    reconstruction_fixture: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    imputed, observed, truth = reconstruction_fixture
    result = reconstruction_metrics(imputed, observed, truth)
    expected = np.mean(
        np.sqrt(np.mean((imputed - truth) ** 2, axis=0))
        / np.std(truth, axis=0, ddof=0)
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
    result = reconstruction_metrics(
        imputed, observed, truth, truth_kind=truth_kind
    )

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
    probability = np.array(
        [[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.75, 0.75]]
    )
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
