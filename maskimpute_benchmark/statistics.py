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
_RESULT_IDENTITY_FIELDS = (
    "mechanism",
    "biological_id",
    "technical_view",
    "dataset_id",
    "method",
    "model_seed",
    "metric",
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
    exclusions: Mapping[str, int]

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


@dataclass(frozen=True, slots=True)
class _ValidatedRecords:
    records: tuple[Mapping[str, Any], ...]
    multiplicities: Mapping[tuple[object, ...], int]


def _record_identity(record: Mapping[str, Any]) -> tuple[object, ...]:
    return tuple(record[field] for field in _RESULT_IDENTITY_FIELDS)


def _value_identity(value: float | None) -> tuple[str, object] | None:
    if value is None:
        return None
    if math.isnan(value):
        return ("nan", 0)
    if math.isinf(value):
        return ("infinity", math.copysign(1.0, value))
    if value == 0.0:
        return ("finite", 0.0)
    return ("finite", value.hex())


def _validate_records(
    records: Sequence[Mapping[str, Any]],
) -> _ValidatedRecords:
    unique: dict[tuple[object, ...], Mapping[str, Any]] = {}
    outcomes: dict[tuple[object, ...], tuple[str, tuple[str, object] | None]] = {}
    multiplicities: dict[tuple[object, ...], int] = defaultdict(int)
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
                raise ValueError(
                    f"record {index} field {field} must be a non-empty string"
                )
        raw_seed = record["model_seed"]
        if isinstance(raw_seed, (bool, np.bool_)) or not isinstance(
            raw_seed, (int, np.integer)
        ):
            raise TypeError(f"record {index} model_seed must be an integer")
        seed = int(raw_seed)
        if not 0 <= seed < 2**63:
            raise ValueError(f"record {index} model_seed must lie in [0, 2**63)")
        value = record["value"]
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
        ):
            raise TypeError(f"record {index} value must be numeric or None")
        try:
            canonical_value = None if value is None else float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"record {index} value must be representable as a float"
            ) from error

        canonical = dict(record)
        canonical["model_seed"] = seed
        canonical["status"] = record["status"].strip().lower()
        canonical["value"] = canonical_value
        identity = _record_identity(canonical)
        outcome = (canonical["status"], _value_identity(canonical_value))
        if identity in unique and outcomes[identity] != outcome:
            raise ValueError(
                "conflicting duplicate result identity "
                f"{identity!r}: status or value differs"
            )
        if identity not in unique:
            unique[identity] = canonical
            outcomes[identity] = outcome
        multiplicities[identity] += 1
    return _ValidatedRecords(
        records=tuple(unique.values()),
        multiplicities=MappingProxyType(dict(multiplicities)),
    )


def _seed_values(
    grouped: Mapping[object, Sequence[float]],
) -> tuple[float, ...]:
    """Collapse duplicate result rows without giving a seed extra weight."""

    ordered_seeds = sorted(
        grouped, key=lambda value: (type(value).__name__, repr(value))
    )
    return tuple(_finite_mean(grouped[seed]) for seed in ordered_seeds)


def _finite_mean(values: Sequence[float]) -> float:
    """Return a finite arithmetic mean without overflowing its finite inputs."""

    if not values:
        raise ValueError("cannot average an empty sequence")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    scaled = tuple(value / scale for value in values)
    normalized = math.fsum(scaled) / len(scaled)
    # A mean lies in its input range.  Clamping only corrects a possible final
    # rounding ulp that could otherwise overflow when scale is DBL_MAX.
    normalized = min(max(normalized, min(scaled)), max(scaled))
    result = scale * normalized
    if not math.isfinite(result):  # pragma: no cover - defensive invariant
        raise ArithmeticError("finite mean became non-finite")
    return result


def _finite_relative_effect(method_mean: float, comparator_mean: float) -> float | None:
    """Return ``(method-comparator)/abs(comparator)`` when representable."""

    if comparator_mean == 0.0:
        return None
    same_sign = (method_mean >= 0.0 and comparator_mean > 0.0) or (
        method_mean <= 0.0 and comparator_mean < 0.0
    )
    if same_sign:
        # Same-sign subtraction cannot overflow and retains cancellation
        # precision for nearly equal metric values.
        effect = (method_mean - comparator_mean) / abs(comparator_mean)
    else:
        ratio = method_mean / abs(comparator_mean)
        if not math.isfinite(ratio):
            return None
        effect = math.fsum((ratio, -math.copysign(1.0, comparator_mean)))
    if not math.isfinite(effect):
        return None
    return effect


def _finite_median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take the median of an empty sequence")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return _finite_mean((ordered[midpoint - 1], ordered[midpoint]))


def _finite_quantile(values: Sequence[float], probability: float) -> float:
    """NumPy-compatible linear quantile with overflow-safe interpolation."""

    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    if lower_index == upper_index or lower == upper:
        return lower
    upper_weight = position - lower_index
    scale = max(abs(lower), abs(upper))
    if scale == 0.0:
        return 0.0
    lower_scaled = lower / scale
    upper_scaled = upper / scale
    normalized = math.fsum(
        ((1.0 - upper_weight) * lower_scaled, upper_weight * upper_scaled)
    )
    normalized = min(max(normalized, lower_scaled), upper_scaled)
    result = scale * normalized
    if not math.isfinite(result):  # pragma: no cover - defensive invariant
        raise ArithmeticError("finite quantile became non-finite")
    return result


def _finite_sample_variance(values: Sequence[float]) -> float | None:
    """Return sample variance, or ``None`` when it is not float-representable."""

    if len(values) < 2:
        raise ValueError("sample variance requires at least two values")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = tuple(value / scale for value in values)
    normalized_mean = _finite_mean(normalized)
    variance_normalized = math.fsum(
        (value - normalized_mean) ** 2 for value in normalized
    ) / (len(normalized) - 1)
    if variance_normalized == 0.0:
        return 0.0
    variance_mantissa, variance_exponent = math.frexp(variance_normalized)
    scale_mantissa, scale_exponent = math.frexp(scale)
    mantissa = variance_mantissa * scale_mantissa * scale_mantissa
    try:
        result = math.ldexp(mantissa, variance_exponent + 2 * scale_exponent)
    except OverflowError:
        return None
    if not math.isfinite(result) or result == 0.0:
        return None
    return result


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

    raw_records = list(records)
    validated = _validate_records(raw_records)
    materialized = list(validated.records)
    for name, value in (
        ("method", method),
        ("comparator", comparator),
        ("metric", metric),
    ):
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
        "duplicate_rows": 0,
        "missing_method_pairs": 0,
        "missing_comparator_pairs": 0,
        "zero_comparator_pairs": 0,
        "nonrepresentable_effect_pairs": 0,
        "biological_draws_without_pairs": 0,
        "bootstrap_zero_comparator_pairs": 0,
        "bootstrap_nonrepresentable_effect_pairs": 0,
        "bootstrap_empty_replicates": 0,
    }
    selected: list[Mapping[str, Any]] = []
    selected_raw_rows = 0
    for record in materialized:
        if record["metric"] != metric or record["method"] not in {method, comparator}:
            continue
        if technical_view is not None and record["technical_view"] != technical_view:
            continue
        selected.append(record)
        multiplicity = validated.multiplicities[_record_identity(record)]
        selected_raw_rows += multiplicity
        exclusions["duplicate_rows"] += multiplicity - 1

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
        method_mean = _finite_mean(method_seeds)
        comparator_mean = _finite_mean(comparator_seeds)
        if comparator_mean == 0.0:
            exclusions["zero_comparator_pairs"] += 1
            continue
        effect = _finite_relative_effect(method_mean, comparator_mean)
        if effect is None:
            exclusions["nonrepresentable_effect_pairs"] += 1
            continue
        paired_views.append(
            _PairedView(
                mechanism=pair_key[0],
                biological_id=pair_key[1],
                technical_view=pair_key[2],
                dataset_id=pair_key[3],
                method_seeds=method_seeds,
                comparator_seeds=comparator_seeds,
                observed_effect=effect,
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
        return _empty_result(selected_raw_rows, exclusions)

    for views in unit_views.values():
        views.sort(key=lambda view: (view.technical_view, view.dataset_id))
    unit_effects = {
        unit: _finite_mean(tuple(view.observed_effect for view in views))
        for unit, views in unit_views.items()
    }
    observed_values = tuple(unit_effects.values())
    observed_array = np.asarray(observed_values, dtype=float)
    ties = np.isclose(observed_array, 0.0, rtol=0.0, atol=1e-12)
    n_wins = int(np.sum((observed_array < 0.0) & ~ties))
    n_ties = int(np.sum(ties))
    n_losses = int(np.sum((observed_array > 0.0) & ~ties))
    observed_median = _finite_median(observed_values)

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
                    method_indices = rng.integers(
                        0, len(view.method_seeds), size=len(view.method_seeds)
                    )
                    comparator_indices = rng.integers(
                        0,
                        len(view.comparator_seeds),
                        size=len(view.comparator_seeds),
                    )
                    method_mean = _finite_mean(
                        tuple(view.method_seeds[int(index)] for index in method_indices)
                    )
                    comparator_mean = _finite_mean(
                        tuple(
                            view.comparator_seeds[int(index)]
                            for index in comparator_indices
                        )
                    )
                    if comparator_mean == 0.0:
                        exclusions["bootstrap_zero_comparator_pairs"] += 1
                        continue
                    effect = _finite_relative_effect(method_mean, comparator_mean)
                    if effect is None:
                        exclusions["bootstrap_nonrepresentable_effect_pairs"] += 1
                        continue
                    view_effects.append(effect)
                if view_effects:
                    # Append every sampled occurrence.  Do not collapse duplicate
                    # mechanisms or biological IDs selected by the bootstrap.
                    replicate_effects.append(_finite_mean(view_effects))
        if replicate_effects:
            bootstrap.append(_finite_median(replicate_effects))
        else:
            exclusions["bootstrap_empty_replicates"] += 1

    distribution = tuple(bootstrap)
    if not distribution:
        # Observed pairs existed but all bootstrap denominators vanished.  Keep
        # the observed diagnostics while marking interval quantities unavailable.
        return BootstrapResult(
            median_effect=observed_median,
            ci_lower=None,
            ci_upper=None,
            probability_effect_lt_zero=None,
            two_sided_sign_probability=None,
            n_wins=n_wins,
            n_ties=n_ties,
            n_losses=n_losses,
            n_independent_draws=len(unit_views),
            n_raw_rows=selected_raw_rows,
            n_paired_views=len(paired_views),
            exclusions=MappingProxyType(dict(exclusions)),
            bootstrap_distribution=distribution,
            bootstrap_checksum=_checksum(distribution),
        )

    probability_less = sum(value < 0.0 for value in distribution) / len(distribution)
    probability_nonpositive = sum(value <= 0.0 for value in distribution) / len(
        distribution
    )
    probability_nonnegative = sum(value >= 0.0 for value in distribution) / len(
        distribution
    )
    sign_probability = min(
        1.0,
        2.0 * min(probability_nonpositive, probability_nonnegative),
    )
    ci_lower = _finite_quantile(distribution, 0.025)
    ci_upper = _finite_quantile(distribution, 0.975)
    return BootstrapResult(
        median_effect=observed_median,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        probability_effect_lt_zero=probability_less,
        two_sided_sign_probability=float(sign_probability),
        n_wins=n_wins,
        n_ties=n_ties,
        n_losses=n_losses,
        n_independent_draws=len(unit_views),
        n_raw_rows=selected_raw_rows,
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
    return _finite_mean(values)


def summarize_seed_variance(
    records: Sequence[Mapping[str, Any]],
) -> SeedVarianceReport:
    """Separate seed, biological-draw, and technical-view variance components.

    Each component is first estimated at its own nesting level and those
    estimates are then averaged with equal weight.  Consequently, extra seed
    rows cannot turn a repeated measurement into an independent draw.
    """

    raw_records = list(records)
    validated = _validate_records(raw_records)
    materialized = list(validated.records)
    grouped: dict[
        tuple[str, str],
        dict[tuple[str, str, str, str], dict[object, list[float]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    exclusion_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "failed_rows": 0,
            "nonfinite_rows": 0,
            "nonrepresentable_within_seed_variances": 0,
            "nonrepresentable_between_view_variances": 0,
            "nonrepresentable_between_draw_variances": 0,
        }
    )
    for record in materialized:
        method_metric = (record["method"], record["metric"])
        exclusions = exclusion_counts[method_metric]
        if record["status"].strip().lower() != "ok":
            exclusions["failed_rows"] += 1
            continue
        value = record["value"]
        if value is None or not math.isfinite(float(value)):
            exclusions["nonfinite_rows"] += 1
            continue
        seed_group = (
            record["mechanism"],
            record["biological_id"],
            record["technical_view"],
            record["dataset_id"],
        )
        grouped[method_metric][seed_group][record["model_seed"]].append(float(value))

    summaries: list[SeedVarianceSummary] = []
    for method, metric in sorted(exclusion_counts):
        seed_groups = grouped.get((method, metric), {})
        exclusions = exclusion_counts[(method, metric)]
        within_seed_variances: list[float] = []
        dataset_means: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for (mechanism, biological_id, view, _dataset_id), seed_rows in sorted(
            seed_groups.items()
        ):
            seed_values = _seed_values(seed_rows)
            if len(seed_values) >= 2:
                variance = _finite_sample_variance(seed_values)
                if variance is None:
                    exclusions["nonrepresentable_within_seed_variances"] += 1
                else:
                    within_seed_variances.append(variance)
            dataset_means[(mechanism, biological_id, view)].append(
                _finite_mean(seed_values)
            )

        view_means: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (mechanism, biological_id, _view), means in sorted(dataset_means.items()):
            view_means[(mechanism, biological_id)].append(_finite_mean(means))

        between_view_variances: list[float] = []
        draw_means: dict[str, list[float]] = defaultdict(list)
        for (mechanism, _biological_id), means in sorted(view_means.items()):
            if len(means) >= 2:
                variance = _finite_sample_variance(means)
                if variance is None:
                    exclusions["nonrepresentable_between_view_variances"] += 1
                else:
                    between_view_variances.append(variance)
            draw_means[mechanism].append(_finite_mean(means))

        between_draw_variances: list[float] = []
        for means in draw_means.values():
            if len(means) >= 2:
                variance = _finite_sample_variance(means)
                if variance is None:
                    exclusions["nonrepresentable_between_draw_variances"] += 1
                else:
                    between_draw_variances.append(variance)

        summaries.append(
            SeedVarianceSummary(
                method=method,
                metric=metric,
                within_draw_seed_variance=_mean_or_none(within_seed_variances),
                between_biological_draw_variance=_mean_or_none(between_draw_variances),
                between_view_variance=_mean_or_none(between_view_variances),
                n_seed_groups=len(seed_groups),
                n_seed_variance_groups=len(within_seed_variances),
                n_biological_draws=len(view_means),
                n_view_variance_draws=len(between_view_variances),
                n_mechanisms=len(draw_means),
                n_between_draw_mechanisms=len(between_draw_variances),
                exclusions=MappingProxyType(dict(exclusions)),
            )
        )
    return SeedVarianceReport(tuple(summaries))
