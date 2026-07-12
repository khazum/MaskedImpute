"""Hierarchical inference for the publication benchmark.

Model seeds and technical views are repeated measurements, not independent
experimental units.  This module therefore pairs methods at the dataset-view
level and keeps the biological simulation draw as the unit of inference.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any

import numpy as np


_REQUIRED_RECORD_FIELDS = (
    "mechanism",
    "biological_id",
    "technical_view",
    "dataset_id",
    "method",
    "model_seed",
    "metric",
    "value",
    "status",
)
_IDENTITY_FIELDS = (
    "mechanism",
    "biological_id",
    "technical_view",
    "dataset_id",
    "method",
    "metric",
    "status",
)


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """A paired effect estimate with its hierarchical bootstrap distribution."""

    median_effect: float | None
    ci_lower: float | None
    ci_upper: float | None
    probability_effect_lt_zero: float | None
    two_sided_sign_probability: float | None
    n_wins: int
    n_ties: int
    n_losses: int
    n_independent_draws: int
    n_raw_rows: int
    n_paired_views: int
    exclusions: Mapping[str, int]
    bootstrap_distribution: tuple[float, ...]
    bootstrap_checksum: str

    @property
    def median_paired_effect(self) -> float | None:
        """Alias spelling out the estimand represented by ``median_effect``."""

        return self.median_effect

    @property
    def ci_95(self) -> tuple[float | None, float | None]:
        """The percentile 95% confidence interval."""

        return self.ci_lower, self.ci_upper

    @property
    def percentile_95_ci(self) -> tuple[float | None, float | None]:
        return self.ci_95

    @property
    def probability_improvement(self) -> float | None:
        """Probability of a lower metric value in the bootstrap distribution."""

        return self.probability_effect_lt_zero

    @property
    def biological_draw_wins(self) -> int:
        return self.n_wins

    @property
    def biological_draw_ties(self) -> int:
        return self.n_ties


@dataclass(frozen=True, slots=True)
class SeedVarianceSummary:
    """Variance components for one method and metric."""

    method: str
    metric: str
    within_draw_seed_variance: float | None
    between_biological_draw_variance: float | None
    between_view_variance: float | None
    n_seed_groups: int
    n_seed_variance_groups: int
    n_biological_draws: int
    n_view_variance_draws: int
    n_mechanisms: int
    n_between_draw_mechanisms: int

    @property
    def within_draw(self) -> float | None:
        return self.within_draw_seed_variance

    @property
    def between_draw(self) -> float | None:
        return self.between_biological_draw_variance

    @property
    def between_view(self) -> float | None:
        return self.between_view_variance


@dataclass(frozen=True, slots=True)
class SeedVarianceReport(Mapping[tuple[str, str], SeedVarianceSummary]):
    """Immutable mapping from ``(method, metric)`` to variance components."""

    summaries: tuple[SeedVarianceSummary, ...]

    def __getitem__(self, key: tuple[str, str]) -> SeedVarianceSummary:
        for summary in self.summaries:
            if (summary.method, summary.metric) == key:
                return summary
        raise KeyError(key)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter((summary.method, summary.metric) for summary in self.summaries)

    def __len__(self) -> int:
        return len(self.summaries)


@dataclass(frozen=True, slots=True)
class _PairedView:
    mechanism: str
    biological_id: str
    technical_view: str
    dataset_id: str
    method_seeds: tuple[float, ...]
    comparator_seeds: tuple[float, ...]
    observed_effect: float


def _validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"record {index} is missing required field(s): {', '.join(missing)}"
            )
        for field in _IDENTITY_FIELDS:
            if not isinstance(record[field], str) or not record[field]:
                raise ValueError(f"record {index} field {field} must be a non-empty string")
        try:
            hash(record["model_seed"])
        except TypeError as error:
            raise ValueError(f"record {index} model_seed must be hashable") from error
        if record["model_seed"] is None:
            raise ValueError(f"record {index} model_seed must not be None")
        value = record["value"]
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
        ):
            raise TypeError(f"record {index} value must be numeric or None")


def _seed_values(
    grouped: Mapping[object, Sequence[float]],
) -> tuple[float, ...]:
    """Collapse duplicate result rows without giving a seed extra weight."""

    ordered_seeds = sorted(grouped, key=lambda value: (type(value).__name__, repr(value)))
    return tuple(float(np.mean(grouped[seed])) for seed in ordered_seeds)


def _checksum(distribution: Sequence[float]) -> str:
    values = np.asarray(distribution, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _empty_result(n_raw_rows: int, exclusions: Mapping[str, int]) -> BootstrapResult:
    distribution: tuple[float, ...] = ()
    return BootstrapResult(
        median_effect=None,
        ci_lower=None,
        ci_upper=None,
        probability_effect_lt_zero=None,
        two_sided_sign_probability=None,
        n_wins=0,
        n_ties=0,
        n_losses=0,
        n_independent_draws=0,
        n_raw_rows=n_raw_rows,
        n_paired_views=0,
        exclusions=MappingProxyType(dict(exclusions)),
        bootstrap_distribution=distribution,
        bootstrap_checksum=_checksum(distribution),
    )


def hierarchical_paired_bootstrap(
    records: Sequence[Mapping[str, Any]],
    method: str,
    comparator: str,
    metric: str,
    n_boot: int = 10_000,
    seed: int = 20_260_712,
    *,
    technical_view: str | None = None,
) -> BootstrapResult:
    """Estimate paired percent change with a nested biological bootstrap.

    Negative values favor ``method`` because every pair is calculated as
    ``(method - comparator) / abs(comparator)``.  Duplicate records for an
    identical model seed are collapsed before seed averaging and resampling.
    """

    materialized = list(records)
    _validate_records(materialized)
    for name, value in (("method", method), ("comparator", comparator), ("metric", metric)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if method == comparator:
        raise ValueError("method and comparator must be different")
    if type(n_boot) is not int or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer")
    if technical_view is not None and (
        not isinstance(technical_view, str) or not technical_view
    ):
        raise ValueError("technical_view must be None or a non-empty string")

    exclusions = {
        "failed_rows": 0,
        "nonfinite_rows": 0,
        "missing_method_pairs": 0,
        "missing_comparator_pairs": 0,
        "zero_comparator_pairs": 0,
        "biological_draws_without_pairs": 0,
        "bootstrap_zero_comparator_pairs": 0,
        "bootstrap_empty_replicates": 0,
    }
    selected: list[Mapping[str, Any]] = []
    for record in materialized:
        if record["metric"] != metric or record["method"] not in {method, comparator}:
            continue
        if technical_view is not None and record["technical_view"] != technical_view:
            continue
        selected.append(record)

    pair_keys: set[tuple[str, str, str, str]] = set()
    values_by_pair: dict[
        tuple[str, str, str, str],
        dict[str, dict[object, list[float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in selected:
        pair_key = (
            record["mechanism"],
            record["biological_id"],
            record["technical_view"],
            record["dataset_id"],
        )
        pair_keys.add(pair_key)
        if record["status"].strip().lower() != "ok":
            exclusions["failed_rows"] += 1
            continue
        value = record["value"]
        if value is None or not math.isfinite(float(value)):
            exclusions["nonfinite_rows"] += 1
            continue
        values_by_pair[pair_key][record["method"]][record["model_seed"]].append(
            float(value)
        )

    paired_views: list[_PairedView] = []
    for pair_key in sorted(pair_keys):
        methods = values_by_pair.get(pair_key, {})
        method_rows = methods.get(method)
        comparator_rows = methods.get(comparator)
        if not method_rows:
            exclusions["missing_method_pairs"] += 1
        if not comparator_rows:
            exclusions["missing_comparator_pairs"] += 1
        if not method_rows or not comparator_rows:
            continue
        method_seeds = _seed_values(method_rows)
        comparator_seeds = _seed_values(comparator_rows)
        comparator_mean = float(np.mean(comparator_seeds))
        if comparator_mean == 0.0:
            exclusions["zero_comparator_pairs"] += 1
            continue
        effect = (float(np.mean(method_seeds)) - comparator_mean) / abs(comparator_mean)
        paired_views.append(
            _PairedView(
                mechanism=pair_key[0],
                biological_id=pair_key[1],
                technical_view=pair_key[2],
                dataset_id=pair_key[3],
                method_seeds=method_seeds,
                comparator_seeds=comparator_seeds,
                observed_effect=float(effect),
            )
        )

    all_units = {(key[0], key[1]) for key in pair_keys}
    unit_views: dict[tuple[str, str], list[_PairedView]] = defaultdict(list)
    for paired_view in paired_views:
        unit_views[(paired_view.mechanism, paired_view.biological_id)].append(
            paired_view
        )
    exclusions["biological_draws_without_pairs"] = len(all_units - set(unit_views))

    if not unit_views:
        return _empty_result(len(selected), exclusions)

    for views in unit_views.values():
        views.sort(key=lambda view: (view.technical_view, view.dataset_id))
    unit_effects = {
        unit: float(np.mean([view.observed_effect for view in views]))
        for unit, views in unit_views.items()
    }
    observed_values = np.asarray(list(unit_effects.values()), dtype=float)
    ties = np.isclose(observed_values, 0.0, rtol=0.0, atol=1e-12)
    n_wins = int(np.sum((observed_values < 0.0) & ~ties))
    n_ties = int(np.sum(ties))
    n_losses = int(np.sum((observed_values > 0.0) & ~ties))

    units_by_mechanism: dict[str, list[str]] = defaultdict(list)
    for mechanism, biological_id in unit_views:
        units_by_mechanism[mechanism].append(biological_id)
    mechanisms = sorted(units_by_mechanism)
    for mechanism in mechanisms:
        units_by_mechanism[mechanism].sort()

    rng = np.random.default_rng(int(seed))
    bootstrap: list[float] = []
    for _ in range(n_boot):
        replicate_effects: list[float] = []
        sampled_mechanisms = rng.integers(0, len(mechanisms), size=len(mechanisms))
        for mechanism_index in sampled_mechanisms:
            mechanism_name = mechanisms[int(mechanism_index)]
            biological_ids = units_by_mechanism[mechanism_name]
            sampled_biological_ids = rng.integers(
                0, len(biological_ids), size=len(biological_ids)
            )
            for biological_index in sampled_biological_ids:
                biological_id = biological_ids[int(biological_index)]
                view_effects: list[float] = []
                for view in unit_views[(mechanism_name, biological_id)]:
                    method_values = np.asarray(view.method_seeds, dtype=float)
                    comparator_values = np.asarray(view.comparator_seeds, dtype=float)
                    method_mean = float(
                        np.mean(
                            method_values[
                                rng.integers(
                                    0, len(method_values), size=len(method_values)
                                )
                            ]
                        )
                    )
                    comparator_mean = float(
                        np.mean(
                            comparator_values[
                                rng.integers(
                                    0,
                                    len(comparator_values),
                                    size=len(comparator_values),
                                )
                            ]
                        )
                    )
                    if comparator_mean == 0.0:
                        exclusions["bootstrap_zero_comparator_pairs"] += 1
                        continue
                    view_effects.append(
                        (method_mean - comparator_mean) / abs(comparator_mean)
                    )
                if view_effects:
                    # Append every sampled occurrence.  Do not collapse duplicate
                    # mechanisms or biological IDs selected by the bootstrap.
                    replicate_effects.append(float(np.mean(view_effects)))
        if replicate_effects:
            bootstrap.append(float(np.median(replicate_effects)))
        else:
            exclusions["bootstrap_empty_replicates"] += 1

    distribution = tuple(bootstrap)
    if not distribution:
        # Observed pairs existed but all bootstrap denominators vanished.  Keep
        # the observed diagnostics while marking interval quantities unavailable.
        return BootstrapResult(
            median_effect=float(np.median(observed_values)),
            ci_lower=None,
            ci_upper=None,
            probability_effect_lt_zero=None,
            two_sided_sign_probability=None,
            n_wins=n_wins,
            n_ties=n_ties,
            n_losses=n_losses,
            n_independent_draws=len(unit_views),
            n_raw_rows=len(selected),
            n_paired_views=len(paired_views),
            exclusions=MappingProxyType(dict(exclusions)),
            bootstrap_distribution=distribution,
            bootstrap_checksum=_checksum(distribution),
        )

    bootstrap_array = np.asarray(distribution, dtype=float)
    probability_less = float(np.mean(bootstrap_array < 0.0))
    probability_nonpositive = float(np.mean(bootstrap_array <= 0.0))
    probability_nonnegative = float(np.mean(bootstrap_array >= 0.0))
    sign_probability = min(
        1.0,
        2.0 * min(probability_nonpositive, probability_nonnegative),
    )
    ci_lower, ci_upper = np.percentile(bootstrap_array, [2.5, 97.5])
    return BootstrapResult(
        median_effect=float(np.median(observed_values)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        probability_effect_lt_zero=probability_less,
        two_sided_sign_probability=float(sign_probability),
        n_wins=n_wins,
        n_ties=n_ties,
        n_losses=n_losses,
        n_independent_draws=len(unit_views),
        n_raw_rows=len(selected),
        n_paired_views=len(paired_views),
        exclusions=MappingProxyType(dict(exclusions)),
        bootstrap_distribution=distribution,
        bootstrap_checksum=_checksum(distribution),
    )


def holm_adjust(
    p_values: Sequence[float | None],
) -> list[float | None]:
    """Adjust finite p-values with Holm's step-down procedure.

    ``None`` and non-finite entries remain in their original positions and do
    not contribute to the number of available hypotheses.
    """

    adjusted: list[float | None] = list(p_values)
    finite: list[tuple[float, int]] = []
    for index, value in enumerate(p_values):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("p-values must be numeric or None")
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("finite p-values must lie in [0, 1]")
        finite.append((numeric, index))

    finite.sort(key=lambda item: (item[0], item[1]))
    running_maximum = 0.0
    n_hypotheses = len(finite)
    for rank, (p_value, original_index) in enumerate(finite):
        candidate = min(1.0, (n_hypotheses - rank) * p_value)
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = min(1.0, running_maximum)
    return adjusted


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def summarize_seed_variance(
    records: Sequence[Mapping[str, Any]],
) -> SeedVarianceReport:
    """Separate seed, biological-draw, and technical-view variance components.

    Each component is first estimated at its own nesting level and those
    estimates are then averaged with equal weight.  Consequently, extra seed
    rows cannot turn a repeated measurement into an independent draw.
    """

    materialized = list(records)
    _validate_records(materialized)
    grouped: dict[
        tuple[str, str],
        dict[tuple[str, str, str, str], dict[object, list[float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for record in materialized:
        if record["status"].strip().lower() != "ok":
            continue
        value = record["value"]
        if value is None or not math.isfinite(float(value)):
            continue
        method_metric = (record["method"], record["metric"])
        seed_group = (
            record["mechanism"],
            record["biological_id"],
            record["technical_view"],
            record["dataset_id"],
        )
        grouped[method_metric][seed_group][record["model_seed"]].append(float(value))

    summaries: list[SeedVarianceSummary] = []
    for (method, metric), seed_groups in sorted(grouped.items()):
        within_seed_variances: list[float] = []
        dataset_means: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for (mechanism, biological_id, view, _dataset_id), seed_rows in sorted(
            seed_groups.items()
        ):
            seed_values = _seed_values(seed_rows)
            if len(seed_values) >= 2:
                within_seed_variances.append(float(np.var(seed_values, ddof=1)))
            dataset_means[(mechanism, biological_id, view)].append(
                float(np.mean(seed_values))
            )

        view_means: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (mechanism, biological_id, _view), means in sorted(dataset_means.items()):
            view_means[(mechanism, biological_id)].append(float(np.mean(means)))

        between_view_variances: list[float] = []
        draw_means: dict[str, list[float]] = defaultdict(list)
        for (mechanism, _biological_id), means in sorted(view_means.items()):
            if len(means) >= 2:
                between_view_variances.append(float(np.var(means, ddof=1)))
            draw_means[mechanism].append(float(np.mean(means)))

        between_draw_variances: list[float] = []
        for means in draw_means.values():
            if len(means) >= 2:
                between_draw_variances.append(float(np.var(means, ddof=1)))

        summaries.append(
            SeedVarianceSummary(
                method=method,
                metric=metric,
                within_draw_seed_variance=_mean_or_none(within_seed_variances),
                between_biological_draw_variance=_mean_or_none(
                    between_draw_variances
                ),
                between_view_variance=_mean_or_none(between_view_variances),
                n_seed_groups=len(seed_groups),
                n_seed_variance_groups=len(within_seed_variances),
                n_biological_draws=len(view_means),
                n_view_variance_draws=len(between_view_variances),
                n_mechanisms=len(draw_means),
                n_between_draw_mechanisms=len(between_draw_variances),
            )
        )
    return SeedVarianceReport(tuple(summaries))
