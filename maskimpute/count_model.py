"""Leakage-safe count-only pre-capture-zero score estimation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import struct
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

from maskimpute.prezero import p_pre_zero_from_counts
from maskimpute.sparse_input import (
    SUPPORTED_SPARSE_TYPES as _SUPPORTED_SPARSE_TYPES,
    contains_masked_array as _contains_masked_array,
    sparse_coordinate_snapshot,
)


_MAX_EXACT_FLOAT64_INTEGER = 2**53
_BUILD_TOKEN = object()


def _integer_at_least(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(
    value: object,
    name: str,
    *,
    strictly_positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    invalid_sign = result <= 0 if strictly_positive else result < 0
    if not math.isfinite(result) or invalid_sign:
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class PreZeroCountModelConfig:
    """Immutable controls for the cross-fitted count-only score model."""

    n_folds: int = 5
    use_library_size_exposure: bool = True
    mean_prior_strength: float = 1.0
    mean_floor: float = 1e-8
    dispersion_prior_strength: float = 10.0
    link_bins: int = 64
    link_max_iter: int = 200
    link_tolerance: float = 1e-10
    link_bound: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "n_folds", _integer_at_least(self.n_folds, "n_folds", 2)
        )
        if type(self.use_library_size_exposure) is not bool:
            raise TypeError("use_library_size_exposure must be a bool")
        object.__setattr__(
            self,
            "mean_prior_strength",
            _finite_float(
                self.mean_prior_strength,
                "mean_prior_strength",
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "mean_floor",
            _finite_float(self.mean_floor, "mean_floor", strictly_positive=True),
        )
        if self.mean_floor > 1:
            raise ValueError("mean_floor must not exceed 1")
        object.__setattr__(
            self,
            "dispersion_prior_strength",
            _finite_float(
                self.dispersion_prior_strength,
                "dispersion_prior_strength",
                strictly_positive=False,
            ),
        )
        object.__setattr__(
            self,
            "link_bins",
            _integer_at_least(self.link_bins, "link_bins", 2),
        )
        object.__setattr__(
            self,
            "link_max_iter",
            _integer_at_least(self.link_max_iter, "link_max_iter", 1),
        )
        object.__setattr__(
            self,
            "link_tolerance",
            _finite_float(
                self.link_tolerance,
                "link_tolerance",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self,
            "link_bound",
            _finite_float(self.link_bound, "link_bound", strictly_positive=True),
        )
        if self.link_bound > 30:
            raise ValueError("link_bound must not exceed 30")


@dataclass(frozen=True, slots=True)
class _ArraySnapshot:
    payload: bytes
    dtype: str
    shape: tuple[int, ...]


def _snapshot_array(value: object, dtype: str) -> _ArraySnapshot:
    array = np.asarray(value, dtype=np.dtype(dtype), order="C")
    contiguous = np.array(array, dtype=np.dtype(dtype), copy=True, order="C")
    return _ArraySnapshot(
        payload=contiguous.tobytes(order="C"),
        dtype=contiguous.dtype.str,
        shape=tuple(contiguous.shape),
    )


def _materialize_array(snapshot: _ArraySnapshot) -> np.ndarray:
    return np.ndarray(
        snapshot.shape,
        dtype=np.dtype(snapshot.dtype),
        buffer=snapshot.payload,
        order="C",
    )


def _validate_entry_values(entries: np.ndarray) -> None:
    if entries.dtype.metadata is not None:
        raise TypeError("observed_counts dtype metadata is not supported")
    if entries.dtype.kind not in "iuf" or entries.dtype.kind == "b":
        raise TypeError("observed_counts must contain real numeric values")
    if not np.all(np.isfinite(entries)):
        raise ValueError("observed_counts must contain only finite values")
    if np.any(entries < 0):
        raise ValueError("observed_counts must be nonnegative")
    if np.any(entries != np.floor(entries)):
        raise ValueError("observed_counts must contain integral counts")
    if np.any(entries > _MAX_EXACT_FLOAT64_INTEGER):
        raise ValueError("observed_counts exceed exact float64 count range")


def _validated_counts(observed_counts: object) -> np.ndarray:
    is_sparse = sparse.issparse(observed_counts)
    if is_sparse and type(observed_counts) not in _SUPPORTED_SPARSE_TYPES:
        raise TypeError("observed_counts must use an exact supported SciPy sparse type")

    if is_sparse:
        entries, rows, columns, shape = sparse_coordinate_snapshot(
            observed_counts,
            "observed_counts",
        )
        if 0 in shape:
            raise ValueError("observed_counts must have at least one cell and one gene")
        _validate_entry_values(entries)
        counts = np.zeros(shape, dtype=np.float64, order="C")
        counts[rows, columns] = entries
    else:
        if _contains_masked_array(observed_counts):
            raise TypeError("observed_counts must not contain masked arrays")
        coerced = np.asanyarray(observed_counts)
        if np.ma.isMaskedArray(coerced):
            raise TypeError("observed_counts must not contain masked arrays")
        if coerced.ndim != 2:
            raise ValueError("observed_counts must be a two-dimensional matrix")
        if 0 in coerced.shape:
            raise ValueError("observed_counts must have at least one cell and one gene")
        if coerced.dtype.metadata is not None:
            raise TypeError("observed_counts dtype metadata is not supported")
        entries = np.array(coerced, copy=True, order="C", subok=False)
        _validate_entry_values(entries)
        counts = np.asarray(entries, dtype=np.float64, order="C")

    if counts.shape[0] < 2:
        raise ValueError("observed_counts must contain at least two cells")
    return counts


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_counts_bytes(counts: np.ndarray) -> bytes:
    canonical = np.asarray(counts, dtype="<u8", order="C")
    shape = struct.pack("<QQ", counts.shape[0], counts.shape[1])
    return b"maskimpute-count-matrix-v1\0" + shape + canonical.tobytes(order="C")


def _counts_sha256(counts: np.ndarray) -> str:
    return _sha256_bytes(_canonical_counts_bytes(counts))


def _config_payload(config: PreZeroCountModelConfig) -> dict[str, object]:
    return {
        "dispersion_prior_strength": config.dispersion_prior_strength,
        "link_bins": config.link_bins,
        "link_bound": config.link_bound,
        "link_max_iter": config.link_max_iter,
        "link_tolerance": config.link_tolerance,
        "mean_floor": config.mean_floor,
        "mean_prior_strength": config.mean_prior_strength,
        "n_folds": config.n_folds,
        "use_library_size_exposure": config.use_library_size_exposure,
    }


def _balanced_fold_ids(
    counts: np.ndarray,
    effective_folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    canonical_rows = np.asarray(counts, dtype="<u8", order="C")
    domain = b"maskimpute-count-model-row-fold-v2\0" + struct.pack(
        "<Q", counts.shape[1]
    )
    keys = []
    for index in range(counts.shape[0]):
        row_bytes = canonical_rows[index].tobytes(order="C")
        keys.append(
            (
                hashlib.sha256(domain + row_bytes).digest(),
                row_bytes,
                index,
            )
        )
    order = np.asarray([index for _, _, index in sorted(keys)], dtype=np.int64)
    fold_ids = np.empty(counts.shape[0], dtype=np.int64)
    for position, index in enumerate(order):
        fold_ids[index] = position % effective_folds
    return fold_ids, order


def _library_exposures(
    training: np.ndarray,
    held_out: np.ndarray,
    use_library_size_exposure: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not use_library_size_exposure:
        return (
            np.ones(training.shape[0], dtype=np.float64),
            np.ones(held_out.shape[0], dtype=np.float64),
            1.0,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        training_libraries = training.sum(axis=1, dtype=np.float64)
        held_out_libraries = held_out.sum(axis=1, dtype=np.float64)
    if not np.all(np.isfinite(training_libraries)) or not np.all(
        np.isfinite(held_out_libraries)
    ):
        raise FloatingPointError("count-model library totals are not finite")
    positive_libraries = training_libraries[training_libraries > 0]
    if positive_libraries.size:
        maximum_library = float(np.max(positive_libraries))
        reference = maximum_library * float(
            np.mean(positive_libraries / maximum_library, dtype=np.float64)
        )
    else:
        reference = 1.0
    training_exposure = training_libraries / reference
    held_out_exposure = held_out_libraries / reference
    if (
        not math.isfinite(reference)
        or reference <= 0
        or not np.all(np.isfinite(training_exposure))
        or not np.all(np.isfinite(held_out_exposure))
    ):
        raise FloatingPointError("count-model library exposure is not finite")
    return training_exposure, held_out_exposure, reference


def _fit_gene_parameters(
    training: np.ndarray,
    exposures: np.ndarray,
    config: PreZeroCountModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        exposure_total = float(np.sum(exposures, dtype=np.float64))
        gene_totals = training.sum(axis=0, dtype=np.float64)
    if not math.isfinite(exposure_total) or not np.all(np.isfinite(gene_totals)):
        raise FloatingPointError("count-model gene or exposure totals are not finite")
    if exposure_total > 0:
        raw_rates = gene_totals / exposure_total
        maximum_rate = float(np.max(raw_rates))
        global_rate = (
            maximum_rate * float(np.mean(raw_rates / maximum_rate, dtype=np.float64))
            if maximum_rate > 0
            else 0.0
        )
    else:
        global_rate = 0.0
        raw_rates = np.zeros(training.shape[1], dtype=np.float64)
    mean_scale = max(exposure_total, config.mean_prior_strength)
    mean_weight = (
        (exposure_total / mean_scale)
        / (exposure_total / mean_scale + config.mean_prior_strength / mean_scale)
        if mean_scale > 0
        else 0.0
    )
    rates = mean_weight * raw_rates + (1.0 - mean_weight) * global_rate
    gene_means = np.maximum(rates, config.mean_floor)
    with np.errstate(over="ignore", invalid="ignore"):
        fitted_mu = np.maximum(
            exposures[:, None] * gene_means[None, :],
            config.mean_floor,
        )
    if not np.all(np.isfinite(fitted_mu)):
        raise FloatingPointError("count-model fitted gene means are not finite")

    with np.errstate(over="ignore", invalid="ignore"):
        moment_numerator = np.sum(
            np.square(training - fitted_mu) - fitted_mu,
            axis=0,
            dtype=np.float64,
        )
        moment_denominator = np.sum(
            np.square(fitted_mu),
            axis=0,
            dtype=np.float64,
        )
    if not np.all(np.isfinite(moment_numerator)) or not np.all(
        np.isfinite(moment_denominator)
    ):
        raise FloatingPointError("count-model dispersion moments are not finite")
    raw_dispersion = np.zeros(training.shape[1], dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.divide(
            moment_numerator,
            moment_denominator,
            out=raw_dispersion,
            where=moment_denominator > 0,
        )
    raw_dispersion = np.maximum(raw_dispersion, 0.0)
    if not np.all(np.isfinite(raw_dispersion)):
        raise FloatingPointError("count-model dispersion moments are not finite")
    global_dispersion = float(np.median(raw_dispersion))
    dispersion_scale = max(
        float(training.shape[0]),
        config.dispersion_prior_strength,
    )
    dispersion_weight = (training.shape[0] / dispersion_scale) / (
        training.shape[0] / dispersion_scale
        + config.dispersion_prior_strength / dispersion_scale
    )
    gene_dispersion = (
        dispersion_weight * raw_dispersion
        + (1.0 - dispersion_weight) * global_dispersion
    )
    if not np.all(np.isfinite(gene_dispersion)):
        raise FloatingPointError("count-model shrunk dispersion is not finite")
    return gene_means, gene_dispersion, fitted_mu


@dataclass(frozen=True, slots=True)
class _LinkFit:
    intercept: float
    slope: float
    converged: bool
    fallback: str | None
    iterations: int
    aggregated_bin_count: int


def _aggregate_binary_link(
    predictor: np.ndarray,
    target: np.ndarray,
    bin_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(predictor, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    minimum = float(np.min(x))
    maximum = float(np.max(x))
    if minimum == maximum or x.size <= bin_limit:
        unique, inverse = np.unique(x, return_inverse=True)
        totals = np.bincount(inverse, minlength=unique.size).astype(np.float64)
        successes = np.bincount(
            inverse,
            weights=y,
            minlength=unique.size,
        ).astype(np.float64)
        return unique, totals, successes

    scaled = (x - minimum) / (maximum - minimum)
    indices = np.minimum((scaled * bin_limit).astype(np.int64), bin_limit - 1)
    totals = np.bincount(indices, minlength=bin_limit).astype(np.float64)
    successes = np.bincount(
        indices,
        weights=y,
        minlength=bin_limit,
    ).astype(np.float64)
    predictor_sums = np.bincount(
        indices,
        weights=x,
        minlength=bin_limit,
    ).astype(np.float64)
    occupied = totals > 0
    return (
        predictor_sums[occupied] / totals[occupied],
        totals[occupied],
        successes[occupied],
    )


def _fit_monotone_zero_link(
    fitted_mu: np.ndarray,
    training: np.ndarray,
    config: PreZeroCountModelConfig,
) -> _LinkFit:
    predictor = np.clip(
        np.log(np.maximum(fitted_mu, config.mean_floor)),
        -config.link_bound,
        config.link_bound,
    ).reshape(-1)
    target = (training == 0).astype(np.float64, copy=False).reshape(-1)
    zero_count = int(np.count_nonzero(target))
    total = target.size
    if zero_count == 0:
        return _LinkFit(
            -config.link_bound,
            0.0,
            False,
            "no_observed_zeros",
            0,
            1,
        )
    if zero_count == total:
        return _LinkFit(
            config.link_bound,
            0.0,
            False,
            "all_observed_zeros",
            0,
            1,
        )

    x, totals, successes = _aggregate_binary_link(
        predictor,
        target,
        config.link_bins,
    )
    zero_fraction = zero_count / total
    initial_intercept = float(
        np.clip(
            math.log(zero_fraction) - math.log1p(-zero_fraction),
            -config.link_bound,
            config.link_bound,
        )
    )
    if float(np.ptp(x)) == 0.0:
        return _LinkFit(
            initial_intercept,
            0.0,
            True,
            "constant_log_mean",
            0,
            int(x.size),
        )

    normalization = float(np.sum(totals))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = parameters
        linear = intercept + slope * x
        loss = float(np.sum(totals * np.logaddexp(0.0, linear) - successes * linear))
        fitted = expit(linear)
        residual = totals * fitted - successes
        gradient = np.array(
            [np.sum(residual), np.sum(residual * x)],
            dtype=np.float64,
        )
        return loss / normalization, gradient / normalization

    result = minimize(
        objective,
        x0=np.array([initial_intercept, -0.1], dtype=np.float64),
        jac=True,
        bounds=(
            (-config.link_bound, config.link_bound),
            (-config.link_bound, 0.0),
        ),
        method="L-BFGS-B",
        options={
            "maxiter": config.link_max_iter,
            "ftol": config.link_tolerance,
            "gtol": config.link_tolerance,
            "maxls": 40,
        },
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    if (
        result.success
        and parameters.shape == (2,)
        and np.all(np.isfinite(parameters))
        and parameters[1] <= 0
    ):
        return _LinkFit(
            float(parameters[0]),
            float(parameters[1]),
            True,
            None,
            int(result.nit),
            int(x.size),
        )
    return _LinkFit(
        initial_intercept,
        0.0,
        False,
        f"optimizer_failed_status_{int(result.status)}",
        int(result.nit),
        int(x.size),
    )


def _nb_zero_probability(mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    negative_log = np.array(mu, dtype=np.float64, copy=True, order="C")
    overdispersed = alpha > 0
    if np.any(overdispersed):
        local_mu = mu[overdispersed]
        local_alpha = alpha[overdispersed]
        with np.errstate(
            over="ignore",
            under="ignore",
            divide="ignore",
            invalid="ignore",
        ):
            product = local_alpha * local_mu
            local_negative_log = np.empty_like(product)
            finite_positive = np.isfinite(product) & (product > 0)
            local_negative_log[finite_positive] = (
                np.log1p(product[finite_positive]) / product[finite_positive]
            ) * local_mu[finite_positive]
            local_negative_log[product == 0] = local_mu[product == 0]
            infinite = np.isinf(product)
            log_product = np.log(local_alpha[infinite]) + np.log(local_mu[infinite])
            local_negative_log[infinite] = (
                np.logaddexp(0.0, log_product) / local_alpha[infinite]
            )
        negative_log[overdispersed] = local_negative_log
    with np.errstate(under="ignore", invalid="ignore"):
        probability = np.exp(-negative_log)
    if not np.all(np.isfinite(probability)):
        raise FloatingPointError("count-model NB zero probability is not finite")
    return probability


def _derive_loss_probability(
    mu: np.ndarray,
    alpha: np.ndarray,
    link: _LinkFit,
    config: PreZeroCountModelConfig,
) -> tuple[np.ndarray, float]:
    predictor = np.clip(
        np.log(np.maximum(mu, config.mean_floor)),
        -config.link_bound,
        config.link_bound,
    )
    total_zero = expit(link.intercept + link.slope * predictor)
    count_zero = _nb_zero_probability(mu, alpha)
    clamped = total_zero < count_zero
    total_zero = np.maximum(total_zero, count_zero)
    denominator = 1.0 - count_zero
    loss = np.zeros_like(mu, dtype=np.float64)
    np.divide(
        total_zero - count_zero,
        denominator,
        out=loss,
        where=denominator > 0,
    )
    np.clip(loss, 0.0, 1.0, out=loss)
    return loss, float(np.count_nonzero(clamped) / clamped.size)


class _FoldModelRecord:
    __slots__ = (
        "_aggregated_bin_count",
        "_clamp_fraction",
        "_exposure_reference",
        "_fold_id",
        "_gene_dispersion",
        "_gene_means",
        "_held_out_indices",
        "_link_converged",
        "_link_fallback",
        "_link_intercept",
        "_link_iterations",
        "_link_slope",
        "_training_cell_count",
        "_training_input_sha256",
    )

    def __init__(
        self,
        token: object,
        *,
        fold_id: int,
        held_out_indices: tuple[int, ...],
        training_cell_count: int,
        training_input_sha256: str,
        gene_means: np.ndarray,
        gene_dispersion: np.ndarray,
        exposure_reference: float,
        link: _LinkFit,
        clamp_fraction: float,
    ) -> None:
        if token is not _BUILD_TOKEN:
            raise TypeError("fold model records are created by the estimator")
        object.__setattr__(self, "_fold_id", fold_id)
        object.__setattr__(self, "_held_out_indices", held_out_indices)
        object.__setattr__(self, "_training_cell_count", training_cell_count)
        object.__setattr__(self, "_training_input_sha256", training_input_sha256)
        object.__setattr__(self, "_gene_means", _snapshot_array(gene_means, "<f8"))
        object.__setattr__(
            self,
            "_gene_dispersion",
            _snapshot_array(gene_dispersion, "<f8"),
        )
        object.__setattr__(self, "_exposure_reference", exposure_reference)
        object.__setattr__(self, "_link_intercept", link.intercept)
        object.__setattr__(self, "_link_slope", link.slope)
        object.__setattr__(self, "_link_converged", link.converged)
        object.__setattr__(self, "_link_fallback", link.fallback)
        object.__setattr__(self, "_link_iterations", link.iterations)
        object.__setattr__(self, "_aggregated_bin_count", link.aggregated_bin_count)
        object.__setattr__(self, "_clamp_fraction", clamp_fraction)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot assign to immutable fold-model field {name!r}")

    @property
    def fold_id(self) -> int:
        return self._fold_id

    @property
    def held_out_indices(self) -> tuple[int, ...]:
        return self._held_out_indices

    @property
    def training_cell_count(self) -> int:
        return self._training_cell_count

    @property
    def training_input_sha256(self) -> str:
        return self._training_input_sha256

    @property
    def gene_means(self) -> np.ndarray:
        return _materialize_array(self._gene_means)

    @property
    def gene_dispersion(self) -> np.ndarray:
        return _materialize_array(self._gene_dispersion)

    @property
    def exposure_reference(self) -> float:
        return self._exposure_reference

    @property
    def link_intercept(self) -> float:
        return self._link_intercept

    @property
    def link_slope(self) -> float:
        return self._link_slope

    @property
    def link_converged(self) -> bool:
        return self._link_converged

    @property
    def link_fallback(self) -> str | None:
        return self._link_fallback

    @property
    def link_iterations(self) -> int:
        return self._link_iterations

    @property
    def aggregated_bin_count(self) -> int:
        return self._aggregated_bin_count

    @property
    def clamp_fraction(self) -> float:
        return self._clamp_fraction


def _array_binding(snapshot: _ArraySnapshot) -> dict[str, object]:
    return {
        "dtype": snapshot.dtype,
        "shape": list(snapshot.shape),
        "sha256": _sha256_bytes(snapshot.payload),
    }


def _fold_payload(record: _FoldModelRecord) -> dict[str, object]:
    return {
        "aggregated_bin_count": record.aggregated_bin_count,
        "clamp_fraction": record.clamp_fraction,
        "exposure_reference": record.exposure_reference,
        "fold_id": record.fold_id,
        "gene_dispersion": _array_binding(record._gene_dispersion),
        "gene_means": _array_binding(record._gene_means),
        "held_out_indices": list(record.held_out_indices),
        "link": {
            "converged": record.link_converged,
            "fallback": record.link_fallback,
            "intercept": record.link_intercept,
            "iterations": record.link_iterations,
            "slope": record.link_slope,
        },
        "training_cell_count": record.training_cell_count,
        "training_input_sha256": record.training_input_sha256,
    }


class PreZeroCountModelScore:
    """Immutable score artifact returned by the count-only model."""

    __slots__ = (
        "_alpha",
        "_config_bytes",
        "_config_sha256",
        "_fold_ids",
        "_fold_models",
        "_input_sha256",
        "_manifest_bytes",
        "_mu",
        "_p_pre_zero",
        "_pi",
        "_score_sha256",
        "_shape",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("score artifacts are created by fit_p_pre_zero_count_model")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot assign to immutable score field {name!r}")

    @property
    def shape(self) -> tuple[int, int]:
        self._verify_integrity()
        return self._shape

    @property
    def p_pre_zero(self) -> np.ndarray:
        self._verify_integrity()
        return _materialize_array(self._p_pre_zero)

    @property
    def mu(self) -> np.ndarray:
        self._verify_integrity()
        return _materialize_array(self._mu)

    @property
    def alpha(self) -> np.ndarray:
        self._verify_integrity()
        return _materialize_array(self._alpha)

    @property
    def pi(self) -> np.ndarray:
        self._verify_integrity()
        return _materialize_array(self._pi)

    @property
    def fold_ids(self) -> np.ndarray:
        self._verify_integrity()
        return _materialize_array(self._fold_ids)

    @property
    def fold_models(self) -> tuple[_FoldModelRecord, ...]:
        self._verify_integrity()
        return self._fold_models

    @property
    def input_sha256(self) -> str:
        self._verify_integrity()
        return self._input_sha256

    @property
    def config_sha256(self) -> str:
        self._verify_integrity()
        return self._config_sha256

    @property
    def score_sha256(self) -> str:
        self._verify_integrity()
        return self._score_sha256

    @property
    def manifest(self) -> dict[str, Any]:
        self._verify_integrity()
        return json.loads(self._manifest_bytes)

    def _unsigned_manifest(self) -> dict[str, object]:
        config = json.loads(self._config_bytes)
        return {
            "arrays": {
                "alpha": _array_binding(self._alpha),
                "fold_ids": _array_binding(self._fold_ids),
                "mu": _array_binding(self._mu),
                "p_pre_zero": _array_binding(self._p_pre_zero),
                "pi": _array_binding(self._pi),
            },
            "artifact_type": "maskimpute_count_model_score",
            "config": config,
            "config_sha256": self._config_sha256,
            "cross_fitting": {
                "assignment": (
                    "balanced_sha256_row_content_order_round_robin_index_ties"
                ),
                "duplicate_row_tie_breaker": (
                    "input_row_index_for_identical_canonical_rows"
                ),
                "effective_folds": len(self._fold_models),
                "fold_models": [_fold_payload(record) for record in self._fold_models],
            },
            "estimand": "pre_capture_zero_given_observed_zero",
            "input_sha256": self._input_sha256,
            "model": {
                "count_family": "negative_binomial_2_with_poisson_limit",
                "dispersion": "gene_moment_estimate_with_global_shrinkage",
                "exposure": (
                    "training_reference_library_size"
                    if config["use_library_size_exposure"]
                    else "unit_cell_exposure"
                ),
                "mean": "exposure_times_shrunk_gene_mean_with_absolute_floor",
                "score": "bayes_pre_capture_zero_given_observed_zero",
                "total_zero_link": ("bounded_nonincreasing_logistic_on_log_mean"),
            },
            "schema_version": 1,
            "shape": list(self._shape),
        }

    def _verify_integrity(self) -> None:
        if type(self) is not PreZeroCountModelScore:
            raise TypeError("verified scores require the exact PreZeroCountModelScore")
        try:
            if _sha256_bytes(self._config_bytes) != self._config_sha256:
                raise ValueError("configuration digest mismatch")
            unsigned = self._unsigned_manifest()
            expected_score_sha256 = _sha256_bytes(_canonical_json_bytes(unsigned))
            expected_manifest = dict(unsigned)
            expected_manifest["score_sha256"] = expected_score_sha256
            if (
                expected_score_sha256 != self._score_sha256
                or _canonical_json_bytes(expected_manifest) != self._manifest_bytes
            ):
                raise ValueError("score digest mismatch")
        except Exception as error:
            raise ValueError("count-model score integrity check failed") from error

    def score_for_counts(self, observed_counts: object) -> np.ndarray:
        """Return the score only when freshly validated counts match its binding."""

        self._verify_integrity()
        counts = _validated_counts(observed_counts)
        if _counts_sha256(counts) != self._input_sha256:
            raise ValueError(
                "observed_counts does not match the bound count-model input"
            )
        try:
            config_payload = json.loads(self._config_bytes)
            config = PreZeroCountModelConfig(**config_payload)
            expected = _fit_validated_count_model(counts, config)
        except Exception as error:
            raise ValueError(
                "count-model score derivation verification failed"
            ) from error
        if not _same_score_derivation(self, expected):
            raise ValueError("count-model score derivation verification failed")
        return _materialize_array(self._p_pre_zero)


def _build_score(
    *,
    counts: np.ndarray,
    config: PreZeroCountModelConfig,
    input_sha256: str,
    fold_ids: np.ndarray,
    fold_models: tuple[_FoldModelRecord, ...],
    p_pre_zero: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    pi: np.ndarray,
) -> PreZeroCountModelScore:
    score = object.__new__(PreZeroCountModelScore)
    config_bytes = _canonical_json_bytes(_config_payload(config))
    object.__setattr__(score, "_shape", tuple(counts.shape))
    object.__setattr__(score, "_input_sha256", input_sha256)
    object.__setattr__(score, "_config_bytes", config_bytes)
    object.__setattr__(score, "_config_sha256", _sha256_bytes(config_bytes))
    object.__setattr__(score, "_p_pre_zero", _snapshot_array(p_pre_zero, "<f8"))
    object.__setattr__(score, "_mu", _snapshot_array(mu, "<f8"))
    object.__setattr__(score, "_alpha", _snapshot_array(alpha, "<f8"))
    object.__setattr__(score, "_pi", _snapshot_array(pi, "<f8"))
    object.__setattr__(score, "_fold_ids", _snapshot_array(fold_ids, "<i8"))
    object.__setattr__(score, "_fold_models", fold_models)
    unsigned = score._unsigned_manifest()
    score_sha256 = _sha256_bytes(_canonical_json_bytes(unsigned))
    manifest = dict(unsigned)
    manifest["score_sha256"] = score_sha256
    object.__setattr__(score, "_score_sha256", score_sha256)
    object.__setattr__(score, "_manifest_bytes", _canonical_json_bytes(manifest))
    score._verify_integrity()
    return score


def _same_score_derivation(
    actual: PreZeroCountModelScore,
    expected: PreZeroCountModelScore,
) -> bool:
    if (
        actual._shape != expected._shape
        or actual._input_sha256 != expected._input_sha256
        or actual._config_bytes != expected._config_bytes
        or actual._config_sha256 != expected._config_sha256
        or actual._score_sha256 != expected._score_sha256
        or actual._manifest_bytes != expected._manifest_bytes
        or actual._p_pre_zero != expected._p_pre_zero
        or actual._mu != expected._mu
        or actual._alpha != expected._alpha
        or actual._pi != expected._pi
        or actual._fold_ids != expected._fold_ids
        or len(actual._fold_models) != len(expected._fold_models)
    ):
        return False
    for actual_fold, expected_fold in zip(
        actual._fold_models,
        expected._fold_models,
        strict=True,
    ):
        if (
            actual_fold._gene_means != expected_fold._gene_means
            or actual_fold._gene_dispersion != expected_fold._gene_dispersion
            or _canonical_json_bytes(_fold_payload(actual_fold))
            != _canonical_json_bytes(_fold_payload(expected_fold))
        ):
            return False
    return True


def _fit_validated_count_model(
    counts: np.ndarray,
    config: PreZeroCountModelConfig,
) -> PreZeroCountModelScore:
    input_sha256 = _counts_sha256(counts)
    effective_folds = min(config.n_folds, counts.shape[0])
    fold_ids, canonical_row_order = _balanced_fold_ids(counts, effective_folds)

    p_pre_zero = np.empty(counts.shape, dtype=np.float64)
    mu = np.empty(counts.shape, dtype=np.float64)
    alpha = np.empty(counts.shape, dtype=np.float64)
    pi = np.empty(counts.shape, dtype=np.float64)
    fold_models = []
    for fold_id in range(effective_folds):
        held_out_indices_array = canonical_row_order[
            fold_ids[canonical_row_order] == fold_id
        ]
        training_indices = canonical_row_order[fold_ids[canonical_row_order] != fold_id]
        training = np.array(counts[training_indices], copy=True, order="C")
        held_out = np.array(counts[held_out_indices_array], copy=True, order="C")
        training_exposure, held_out_exposure, reference = _library_exposures(
            training,
            held_out,
            config.use_library_size_exposure,
        )
        gene_means, gene_dispersion, training_mu = _fit_gene_parameters(
            training,
            training_exposure,
            config,
        )
        link = _fit_monotone_zero_link(training_mu, training, config)
        held_out_mu = np.maximum(
            held_out_exposure[:, None] * gene_means[None, :],
            config.mean_floor,
        )
        held_out_alpha = np.broadcast_to(
            gene_dispersion[None, :],
            held_out.shape,
        ).copy()
        held_out_pi, clamp_fraction = _derive_loss_probability(
            held_out_mu,
            held_out_alpha,
            link,
            config,
        )
        held_out_score = p_pre_zero_from_counts(
            held_out,
            held_out_mu,
            held_out_alpha,
            held_out_pi,
        )
        p_pre_zero[held_out_indices_array] = held_out_score
        mu[held_out_indices_array] = held_out_mu
        alpha[held_out_indices_array] = held_out_alpha
        pi[held_out_indices_array] = held_out_pi
        fold_models.append(
            _FoldModelRecord(
                _BUILD_TOKEN,
                fold_id=fold_id,
                held_out_indices=tuple(int(index) for index in held_out_indices_array),
                training_cell_count=int(training.shape[0]),
                training_input_sha256=_counts_sha256(training),
                gene_means=gene_means,
                gene_dispersion=gene_dispersion,
                exposure_reference=reference,
                link=link,
                clamp_fraction=clamp_fraction,
            )
        )

    for value, name in (
        (p_pre_zero, "p_pre_zero"),
        (mu, "mu"),
        (alpha, "alpha"),
        (pi, "pi"),
    ):
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"count-model {name} is not finite")
    if np.any((p_pre_zero < 0) | (p_pre_zero > 1)):
        raise FloatingPointError("count-model p_pre_zero lies outside [0, 1]")
    if np.any(p_pre_zero[counts > 0] != 0):
        raise FloatingPointError("count-model scored an observed positive entry")
    if np.any(mu <= 0) or np.any(alpha < 0) or np.any((pi < 0) | (pi > 1)):
        raise FloatingPointError("count-model parameter support check failed")
    return _build_score(
        counts=counts,
        config=config,
        input_sha256=input_sha256,
        fold_ids=fold_ids,
        fold_models=tuple(fold_models),
        p_pre_zero=p_pre_zero,
        mu=mu,
        alpha=alpha,
        pi=pi,
    )


def fit_p_pre_zero_count_model(
    observed_counts: object,
    config: PreZeroCountModelConfig = PreZeroCountModelConfig(),
) -> PreZeroCountModelScore:
    """Fit cross-fitted pre-capture-zero probabilities from counts alone.

    Fold ordering hashes canonical row content, so distinct-cell results are
    equivariant to row permutations.  Exact duplicate rows cannot be assigned
    persistent identities without external cell IDs and therefore use current
    input row index as their deterministic tie breaker.  Cross-fitting prevents
    cell self-fit but does not identify count-zero versus loss components from
    counts alone; this score is the posterior under the disclosed fitted
    NB2-plus-loss model.
    """

    if type(config) is not PreZeroCountModelConfig:
        raise TypeError("config must be an exact PreZeroCountModelConfig")
    try:
        config = PreZeroCountModelConfig(**_config_payload(config))
    except (TypeError, ValueError) as error:
        raise type(error)(f"config revalidation failed: {error}") from error
    counts = _validated_counts(observed_counts)
    return _fit_validated_count_model(counts, config)
