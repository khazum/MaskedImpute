"""Count-model probability that a pre-capture count was zero."""

import numpy as np
from scipy import sparse


def _reject_sparse_duplicate_coordinates(value, name):
    coordinates = value.tocoo(copy=True)
    if coordinates.nnz < 2:
        return
    order = np.lexsort((coordinates.col, coordinates.row))
    rows = coordinates.row[order]
    columns = coordinates.col[order]
    if np.any((rows[1:] == rows[:-1]) & (columns[1:] == columns[:-1])):
        raise ValueError(f"{name} must not contain duplicate sparse coordinates")


def _observed_array(value):
    if np.ma.isMaskedArray(value):
        raise TypeError("observed_counts must not be a masked array")
    if sparse.issparse(value):
        if np.ma.isMaskedArray(getattr(value, "data", None)):
            raise TypeError("observed_counts must not contain masked sparse data")
        matrix = value.copy()
        _reject_sparse_duplicate_coordinates(matrix, "observed_counts")
        matrix = matrix.tocsr(copy=True)
        entries = matrix.data
        dtype = matrix.dtype
        observed = matrix.toarray()
    else:
        observed = np.asarray(value)
        entries = observed
        dtype = observed.dtype

    if dtype.kind not in "iuf":
        raise TypeError("observed_counts must contain real, non-boolean numbers")
    if not np.all(np.isfinite(entries)):
        raise ValueError("observed_counts must be finite")
    if np.any(entries < 0):
        raise ValueError("observed_counts must be nonnegative")
    if dtype.kind == "f" and np.any(entries != np.floor(entries)):
        raise ValueError("observed_counts must be integral")
    return observed


def _model_parameter(value, name, shape, *, upper=None):
    if np.ma.isMaskedArray(value):
        raise TypeError(f"{name} must not be a masked array")
    if sparse.issparse(value):
        raise TypeError(f"{name} must be a dense real number or array")
    original = np.asarray(value)
    if original.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real, non-boolean numbers")
    if not np.all(np.isfinite(original)):
        raise ValueError(f"{name} must be finite")
    if np.any(original < 0):
        raise ValueError(f"{name} must be nonnegative")
    if upper is not None and np.any(original > upper):
        raise ValueError(f"{name} must lie in [0, {upper}]")
    try:
        computation_dtype = np.result_type(original.dtype, np.float64)
        return np.broadcast_to(original.astype(computation_dtype, copy=False), shape)
    except ValueError as error:
        raise ValueError(f"{name} must broadcast into observed_counts.shape") from error


def _negative_log_zero_probability(mean, dispersion):
    result = mean.copy()
    overdispersed = dispersion > 0
    local_mean = mean[overdispersed]
    local_dispersion = dispersion[overdispersed]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        product = local_dispersion * local_mean
        negative_log = np.empty_like(product)

        finite_positive = np.isfinite(product) & (product > 0)
        negative_log[finite_positive] = (
            np.log1p(product[finite_positive]) / product[finite_positive]
        ) * local_mean[finite_positive]
        negative_log[product == 0] = local_mean[product == 0]

        infinite = np.isinf(product)
        log_product = np.log(local_dispersion[infinite]) + np.log(local_mean[infinite])
        negative_log[infinite] = (
            np.logaddexp(0.0, log_product) / local_dispersion[infinite]
        )

    result[overdispersed] = negative_log
    return result


def p_pre_zero_from_counts(observed_counts, mu, alpha, pi):
    """Return the fitted probability that a pre-capture count was zero.

    The observation model retains a positive latent count with probability
    ``1 - pi`` and maps it to zero otherwise.  ``mu`` and ``alpha`` parameterize
    a negative binomial count with variance ``mu + alpha * mu**2``; ``alpha=0``
    is its Poisson limit.  For an observed zero, Bayes' rule gives
    ``p0 / (pi + (1 - pi) * p0)``.  Observed positive counts receive probability
    zero.

    This probability is model-dependent, and its count and loss components are
    not separately identifiable from one observed count matrix without added
    assumptions or information.  It describes the discrete pre-capture count,
    not the underlying expression state.
    """

    observed = _observed_array(observed_counts)
    mean = _model_parameter(mu, "mu", observed.shape)
    dispersion = _model_parameter(alpha, "alpha", observed.shape)
    loss = _model_parameter(pi, "pi", observed.shape, upper=1.0)
    positive = observed > 0
    if np.any(positive & ((mean == 0) | (loss == 1))):
        raise ValueError("positive observations are impossible when mu=0 or pi=1")

    log_p0 = -_negative_log_zero_probability(mean, dispersion)
    result = np.zeros(observed.shape, dtype=np.float64)
    zero = observed == 0
    zero_log_p0 = log_p0[zero]
    zero_loss = loss[zero]
    zero_result = np.empty_like(zero_log_p0)
    no_loss = zero_loss == 0
    zero_result[no_loss] = 1.0
    with_loss = ~no_loss
    with np.errstate(divide="ignore", invalid="ignore"):
        log_denominator = np.logaddexp(
            np.log(zero_loss[with_loss]),
            np.log1p(-zero_loss[with_loss]) + zero_log_p0[with_loss],
        )
    zero_result[with_loss] = np.exp(zero_log_p0[with_loss] - log_denominator)
    result[zero] = zero_result
    return result
