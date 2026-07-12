from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import pytest

from maskimpute_benchmark.statistics import (
    BootstrapResult,
    hierarchical_paired_bootstrap,
    holm_adjust,
    summarize_seed_variance,
)


def _record(
    mechanism: str,
    biological_id: str,
    technical_view: str,
    dataset_id: str,
    method: str,
    model_seed: int,
    value: float | None,
    *,
    metric: str = "mse",
    status: str = "ok",
) -> dict[str, object]:
    return {
        "mechanism": mechanism,
        "biological_id": biological_id,
        "technical_view": technical_view,
        "dataset_id": dataset_id,
        "method": method,
        "model_seed": model_seed,
        "metric": metric,
        "value": value,
        "status": status,
    }


@pytest.fixture
def paired_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    moderate_effects = {
        "mechanism-a": (-0.2, -0.1, 0.0),
        "mechanism-b": (-0.3, -0.2, -0.1),
    }
    for mechanism, biological_effects in moderate_effects.items():
        for biological_index, moderate_effect in enumerate(biological_effects):
            biological_id = f"draw-{biological_index}"
            for view, effect in (
                ("moderate", moderate_effect),
                ("severe", moderate_effect - 0.2),
            ):
                dataset_id = f"{mechanism}-{biological_id}-{view}"
                for seed, offset in enumerate((-0.3, 0.0, 0.3), start=42):
                    records.append(
                        _record(
                            mechanism,
                            biological_id,
                            view,
                            dataset_id,
                            "dca",
                            seed,
                            10.0 + offset,
                        )
                    )
                    records.append(
                        _record(
                            mechanism,
                            biological_id,
                            view,
                            dataset_id,
                            "maskimpute",
                            seed,
                            10.0 * (1.0 + effect) + offset,
                        )
                    )
    return records


def test_model_seeds_and_views_do_not_inflate_independent_n(
    paired_records: list[dict[str, object]],
) -> None:
    result = hierarchical_paired_bootstrap(
        paired_records,
        "maskimpute",
        "dca",
        "mse",
        n_boot=200,
        seed=7,
    )

    assert isinstance(result, BootstrapResult)
    assert result.n_independent_draws == 6
    assert result.n_raw_rows == 72
    assert result.n_paired_views == 12
    assert result.median_effect == pytest.approx(-0.25)
    assert result.n_wins == 6
    assert result.n_ties == 0
    assert result.n_losses == 0
    assert len(result.bootstrap_distribution) == 200
    assert len(result.bootstrap_checksum) == 64

    with pytest.raises(FrozenInstanceError):
        result.n_independent_draws = 12  # type: ignore[misc]


def test_exact_duplicate_rows_do_not_reweight_seeds_or_inflate_draw_n(
    paired_records: list[dict[str, object]],
) -> None:
    original = hierarchical_paired_bootstrap(
        paired_records, "maskimpute", "dca", "mse", n_boot=100, seed=19
    )
    duplicated = hierarchical_paired_bootstrap(
        paired_records + [dict(row) for row in paired_records],
        "maskimpute",
        "dca",
        "mse",
        n_boot=100,
        seed=19,
    )

    assert duplicated.n_raw_rows == 144
    assert duplicated.n_independent_draws == original.n_independent_draws == 6
    assert duplicated.n_paired_views == original.n_paired_views == 12
    assert duplicated.median_effect == original.median_effect
    assert duplicated.bootstrap_distribution == original.bootstrap_distribution


def test_view_effects_are_averaged_after_pairwise_percent_change() -> None:
    records: list[dict[str, object]] = []
    for view, method_value, comparator_value in (
        ("moderate", 5.0, 10.0),
        ("severe", 90.0, 100.0),
    ):
        for method, value in (
            ("maskimpute", method_value),
            ("dca", comparator_value),
        ):
            records.append(
                _record("mechanism-a", "draw-0", view, view, method, 42, value)
            )

    result = hierarchical_paired_bootstrap(
        records, "maskimpute", "dca", "mse", n_boot=20, seed=1
    )

    assert result.n_independent_draws == 1
    assert result.n_paired_views == 2
    assert result.median_effect == pytest.approx((-0.5 - 0.1) / 2.0)


def test_view_stratification_retains_biological_draw_as_the_unit(
    paired_records: list[dict[str, object]],
) -> None:
    result = hierarchical_paired_bootstrap(
        paired_records,
        "maskimpute",
        "dca",
        "mse",
        n_boot=100,
        seed=11,
        technical_view="moderate",
    )

    assert result.n_raw_rows == 36
    assert result.n_paired_views == 6
    assert result.n_independent_draws == 6
    assert result.median_effect == pytest.approx(-0.15)
    assert result.n_wins == 5
    assert result.n_ties == 1


def test_missing_failed_nonfinite_and_zero_comparator_pairs_are_explicitly_excluded(
    paired_records: list[dict[str, object]],
) -> None:
    records = [dict(row) for row in paired_records]
    failed_key = ("mechanism-a", "draw-0", "moderate")
    nonfinite_key = ("mechanism-a", "draw-1", "moderate")
    zero_key = ("mechanism-a", "draw-2", "moderate")
    missing_key = ("mechanism-b", "draw-0", "moderate")
    retained: list[dict[str, object]] = []
    for row in records:
        key = (row["mechanism"], row["biological_id"], row["technical_view"])
        if key == missing_key and row["method"] == "dca":
            continue
        if key == failed_key and row["method"] == "maskimpute":
            row["status"] = "failed"
        if key == nonfinite_key and row["method"] == "dca":
            row["value"] = float("nan")
        if key == zero_key and row["method"] == "dca":
            row["value"] = 0.0
        retained.append(row)

    result = hierarchical_paired_bootstrap(
        retained, "maskimpute", "dca", "mse", n_boot=50, seed=5
    )

    assert result.n_raw_rows == 69
    assert result.n_independent_draws == 6
    assert result.n_paired_views == 8
    assert result.exclusions["failed_rows"] == 3
    assert result.exclusions["nonfinite_rows"] == 3
    assert result.exclusions["missing_method_pairs"] == 1
    assert result.exclusions["missing_comparator_pairs"] == 2
    assert result.exclusions["zero_comparator_pairs"] == 1


def test_zero_denominator_can_remove_an_entire_biological_draw() -> None:
    records = []
    for biological_id, comparator in (("usable", 2.0), ("zero", 0.0)):
        for view in ("moderate", "severe"):
            for method, value in (("maskimpute", 1.0), ("dca", comparator)):
                records.append(
                    _record(
                        "mechanism-a",
                        biological_id,
                        view,
                        f"{biological_id}-{view}",
                        method,
                        42,
                        value,
                    )
                )

    result = hierarchical_paired_bootstrap(
        records, "maskimpute", "dca", "mse", n_boot=20, seed=2
    )

    assert result.n_independent_draws == 1
    assert result.n_paired_views == 2
    assert result.exclusions["zero_comparator_pairs"] == 2
    assert result.exclusions["biological_draws_without_pairs"] == 1


def test_bootstrap_is_reproducible_and_resamples_nested_model_seeds() -> None:
    variable = []
    fixed = []
    for seed, method_value in enumerate((5.0, 10.0, 15.0), start=42):
        variable.extend(
            (
                _record("m", "b", "v", "d", "maskimpute", seed, method_value),
                _record("m", "b", "v", "d", "dca", seed, 10.0),
            )
        )
        fixed.extend(
            (
                _record("m", "b", "v", "d", "maskimpute", seed, 10.0),
                _record("m", "b", "v", "d", "dca", seed, 10.0),
            )
        )

    first = hierarchical_paired_bootstrap(
        variable, "maskimpute", "dca", "mse", n_boot=300, seed=31
    )
    second = hierarchical_paired_bootstrap(
        variable, "maskimpute", "dca", "mse", n_boot=300, seed=31
    )
    no_seed_variation = hierarchical_paired_bootstrap(
        fixed, "maskimpute", "dca", "mse", n_boot=300, seed=31
    )

    assert first.bootstrap_distribution == second.bootstrap_distribution
    assert first.bootstrap_checksum == second.bootstrap_checksum
    assert len(set(first.bootstrap_distribution)) > 1
    assert set(no_seed_variation.bootstrap_distribution) == {0.0}


def test_hierarchical_bootstrap_preserves_cluster_resampling_structure() -> None:
    records = []
    for biological_id, effect in (("left", -0.5), ("right", 0.5)):
        for method, value in (("maskimpute", 10.0 * (1 + effect)), ("dca", 10.0)):
            records.append(_record("m", biological_id, "v", biological_id, method, 1, value))

    result = hierarchical_paired_bootstrap(
        records, "maskimpute", "dca", "mse", n_boot=500, seed=9
    )

    # Resampling two biological clusters with replacement produces duplicate-left,
    # mixed, and duplicate-right replicates.  Duplicate clusters must not collapse.
    assert {round(value, 12) for value in result.bootstrap_distribution} == {
        -0.5,
        0.0,
        0.5,
    }


def test_empty_pairing_returns_an_explicit_unavailable_result() -> None:
    records = [
        _record("m", "b", "v", "d", "maskimpute", 1, 1.0),
    ]

    result = hierarchical_paired_bootstrap(
        records, "maskimpute", "dca", "mse", n_boot=10, seed=1
    )

    assert result.n_independent_draws == 0
    assert result.n_paired_views == 0
    assert result.median_effect is None
    assert result.ci_95 == (None, None)
    assert result.probability_effect_lt_zero is None
    assert result.bootstrap_distribution == ()
    assert result.exclusions["missing_comparator_pairs"] == 1


def test_records_require_the_complete_publication_identity() -> None:
    incomplete = _record("m", "b", "v", "d", "maskimpute", 1, 1.0)
    del incomplete["biological_id"]

    with pytest.raises(ValueError, match="biological_id"):
        hierarchical_paired_bootstrap(
            [incomplete], "maskimpute", "dca", "mse", n_boot=10
        )


def test_holm_adjustment_is_step_down_bounded_and_restores_order() -> None:
    p_values = [0.04, 0.01, None, 0.03, float("nan"), 0.8]

    adjusted = holm_adjust(p_values)

    assert adjusted[:4] == pytest.approx([0.09, 0.04, None, 0.09])
    assert adjusted[5] == 0.8
    assert adjusted[2] is None
    assert math.isnan(adjusted[4])
    assert all(
        value is None or not math.isfinite(value) or 0.0 <= value <= 1.0
        for value in adjusted
    )


def test_holm_rejects_finite_values_outside_probability_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        holm_adjust([0.2, 1.1])


def test_seed_variance_separates_nested_sources_without_row_weighting() -> None:
    records = []
    values = {
        ("draw-0", "moderate"): (1.0, 3.0),
        ("draw-0", "severe"): (3.0, 5.0),
        ("draw-1", "moderate"): (5.0, 7.0),
        ("draw-1", "severe"): (7.0, 9.0),
    }
    for (biological_id, view), seed_values in values.items():
        for seed, value in enumerate(seed_values, start=42):
            records.append(
                _record(
                    "mechanism-a",
                    biological_id,
                    view,
                    f"{biological_id}-{view}",
                    "maskimpute",
                    seed,
                    value,
                )
            )

    report = summarize_seed_variance(records)
    summary = report[("maskimpute", "mse")]

    assert summary.within_draw_seed_variance == pytest.approx(2.0)
    assert summary.between_view_variance == pytest.approx(2.0)
    assert summary.between_biological_draw_variance == pytest.approx(8.0)
    assert summary.n_seed_groups == 4
    assert summary.n_biological_draws == 2
    assert summary.n_mechanisms == 1

    duplicated = summarize_seed_variance(records + [dict(row) for row in records])
    assert duplicated[("maskimpute", "mse")] == summary
