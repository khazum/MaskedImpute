from fnmatch import fnmatchcase
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
from types import MethodType

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import nbinom, poisson


def test_public_api_is_importable_and_discovered_by_setuptools():
    from maskimpute import ImputationResult, MaskImputeConfig, p_pre_zero_from_counts

    assert MaskImputeConfig.__module__ == "maskimpute.config"
    assert ImputationResult.__module__ == "maskimpute.result"
    assert p_pre_zero_from_counts.__module__ == "maskimpute.prezero"

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    patterns = project["tool"]["setuptools"]["packages"]["find"]["include"]
    assert any(fnmatchcase("maskimpute", pattern) for pattern in patterns)


def test_config_has_immutable_explicit_training_defaults():
    from maskimpute import MaskImputeConfig

    config = MaskImputeConfig()

    assert config.hidden_dims == (128, 64)
    assert config.latent_dim == 24
    assert config.learning_rate == pytest.approx(2e-4)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.batch_size == 64
    assert config.max_epochs == 300
    assert config.patience == 30
    assert config.artificial_mask_fraction == pytest.approx(0.20)
    assert config.pre_zero_regularization == pytest.approx(1.0)
    assert config.gate_gamma == pytest.approx(1.0)
    assert config.normalization_target == pytest.approx(10_000.0)
    assert config.seed == 42
    assert not hasattr(config, "__dict__")

    with pytest.raises(FrozenInstanceError):
        config.seed = 7


def test_config_freezes_a_caller_owned_hidden_dimension_sequence():
    from maskimpute import MaskImputeConfig

    hidden_dims = [32, 16]
    config = MaskImputeConfig(hidden_dims=hidden_dims)
    hidden_dims[0] = 999

    assert config.hidden_dims == (32, 16)


@pytest.mark.parametrize(
    "hidden_dims",
    [
        {32, 16},
        {"first": 32, "second": 16},
        (dimension for dimension in (32, 16)),
        iter((32, 16)),
    ],
)
def test_config_rejects_unordered_or_one_shot_hidden_dimension_iterables(hidden_dims):
    from maskimpute import MaskImputeConfig

    with pytest.raises(TypeError, match="sequence"):
        MaskImputeConfig(hidden_dims=hidden_dims)


@pytest.mark.parametrize(
    "override",
    [
        {"hidden_dims": ()},
        {"hidden_dims": (32, 0)},
        {"hidden_dims": (True, 16)},
        {"latent_dim": 0},
        {"latent_dim": True},
        {"learning_rate": 0.0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1.0},
        {"weight_decay": float("inf")},
        {"batch_size": 0},
        {"max_epochs": 1.5},
        {"patience": 0},
        {"artificial_mask_fraction": 0.0},
        {"artificial_mask_fraction": 1.0},
        {"pre_zero_regularization": -1.0},
        {"gate_gamma": -0.1},
        {"normalization_target": 0.0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_config_rejects_invalid_values(override):
    from maskimpute import MaskImputeConfig

    with pytest.raises((TypeError, ValueError)):
        MaskImputeConfig(**override)


def test_result_defensively_freezes_dense_outputs_and_diagnostics():
    from maskimpute import ImputationResult

    selective = np.array([[2, 0], [1, 3]], dtype=np.int64)
    denoised = np.array([[2.0, 0.5], [1.1, 3.0]])
    probability = np.array([[0.0, 0.8], [0.0, 0.0]])
    latent = np.array([[1.0, -1.0], [0.5, 0.25]])
    trace = np.array([2.0, 1.0])
    diagnostics = {"status": "ok", "trace": trace, "stages": ["fit", "infer"]}

    result = ImputationResult(selective, denoised, probability, latent, diagnostics)

    selective[0, 0] = 99
    denoised[0, 1] = 99
    probability[0, 1] = 0
    latent[0, 0] = 99
    trace[0] = 99
    diagnostics["status"] = "changed"
    diagnostics["stages"].append("changed")

    assert result.selective_counts[0, 0] == 2
    assert result.denoised_counts[0, 1] == pytest.approx(0.5)
    assert result.p_pre_zero[0, 1] == pytest.approx(0.8)
    assert result.latent[0, 0] == pytest.approx(1.0)
    np.testing.assert_array_equal(result.diagnostics["trace"], [2.0, 1.0])
    assert result.diagnostics["status"] == "ok"
    assert result.diagnostics["stages"] == ("fit", "infer")
    assert not hasattr(result, "__dict__")

    for array in (
        result.selective_counts,
        result.denoised_counts,
        result.p_pre_zero,
        result.latent,
        result.diagnostics["trace"],
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0

    with pytest.raises(TypeError):
        result.diagnostics["new"] = 1
    with pytest.raises(FrozenInstanceError):
        result.latent = np.zeros((2, 2))
    with pytest.raises(FrozenInstanceError):
        del result.latent


def test_result_dense_views_have_immutable_backing_and_are_freshly_materialized():
    from maskimpute import ImputationResult

    result = ImputationResult(**_valid_result_arguments())

    expected = {
        "selective_counts": result.selective_counts.copy(),
        "denoised_counts": result.denoised_counts.copy(),
        "p_pre_zero": result.p_pre_zero.copy(),
        "latent": result.latent.copy(),
    }
    for field, original in expected.items():
        exposed = getattr(result, field)
        with pytest.raises(ValueError):
            exposed.flags.writeable = True
        exposed.shape = (exposed.size,)

        fresh = getattr(result, field)
        np.testing.assert_array_equal(fresh, original)
        assert fresh.shape == original.shape


def test_result_nested_diagnostic_arrays_have_immutable_private_snapshots():
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    diagnostic_array = np.array([[1.0, 2.0]])
    arguments["diagnostics"] = {"outer": {"trace": diagnostic_array}}
    result = ImputationResult(**arguments)
    diagnostic_array[:] = 99

    exposed = result.diagnostics["outer"]["trace"]
    with pytest.raises(ValueError):
        exposed.flags.writeable = True
    exposed.shape = (2,)

    fresh = result.diagnostics["outer"]["trace"]
    np.testing.assert_array_equal(fresh, [[1.0, 2.0]])
    assert fresh.shape == (1, 2)


def test_result_rejects_void_diagnostic_arrays_even_when_zero_width():
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    arguments["diagnostics"] = {"zero_width": np.empty(2, dtype="V0")}

    with pytest.raises(TypeError, match="dtype"):
        ImputationResult(**arguments)


def test_result_copies_sparse_counts_into_read_only_csr_storage():
    from maskimpute import ImputationResult

    selective = sparse.csr_matrix([[2, 0], [0, 3]], dtype=np.int64)
    denoised = sparse.csc_matrix([[2.0, 0.5], [0.25, 3.0]])

    result = ImputationResult(
        selective,
        denoised,
        np.array([[0.0, 0.8], [0.6, 0.0]]),
        np.ones((2, 1)),
        {},
    )
    selective.data[0] = 99
    denoised.data[0] = 99

    assert sparse.isspmatrix_csr(result.selective_counts)
    assert sparse.isspmatrix_csr(result.denoised_counts)
    np.testing.assert_array_equal(result.selective_counts.toarray(), [[2, 0], [0, 3]])
    np.testing.assert_allclose(
        result.denoised_counts.toarray(), [[2.0, 0.5], [0.25, 3.0]]
    )
    for matrix in (result.selective_counts, result.denoised_counts):
        assert not matrix.data.flags.writeable
        assert not matrix.indices.flags.writeable
        assert not matrix.indptr.flags.writeable


def test_result_accepts_dok_counts_without_requiring_array_backing_attributes():
    from maskimpute import ImputationResult

    selective = sparse.dok_matrix((2, 2), dtype=np.int64)
    selective[0, 0] = 2
    selective[1, 1] = 3
    denoised = sparse.dok_array((2, 2), dtype=np.float64)
    denoised[0, 0] = 2.0
    denoised[0, 1] = 0.5
    denoised[1, 0] = 0.25
    denoised[1, 1] = 3.0

    result = ImputationResult(
        selective,
        denoised,
        np.array([[0.0, 0.8], [0.6, 0.0]]),
        np.ones((2, 1)),
        {},
    )

    np.testing.assert_array_equal(result.selective_counts.toarray(), [[2, 0], [0, 3]])
    np.testing.assert_allclose(
        result.denoised_counts.toarray(), [[2.0, 0.5], [0.25, 3.0]]
    )
    assert sparse.isspmatrix_csr(result.selective_counts)
    assert isinstance(result.denoised_counts, sparse.csr_array)


def test_result_sparse_views_cannot_mutate_private_storage():
    from maskimpute import ImputationResult

    result = ImputationResult(
        sparse.csr_matrix([[2, 0], [0, 3]], dtype=np.int64),
        sparse.csc_matrix([[2.0, 0.5], [0.25, 3.0]]),
        np.array([[0.0, 0.8], [0.6, 0.0]]),
        np.ones((2, 1)),
        {},
    )
    expected_selective = result.selective_counts.toarray()
    expected_denoised = result.denoised_counts.toarray()

    selective = result.selective_counts
    for backing in (selective.data, selective.indices, selective.indptr):
        with pytest.raises(ValueError):
            backing.flags.writeable = True
    selective.data = np.array([99, 98], dtype=np.int64)
    selective.indices = np.array([1, 0], dtype=np.int32)
    selective.indptr = np.array([0, 1, 2], dtype=np.int32)
    selective._shape = (1, 2)

    denoised = result.denoised_counts
    denoised.data = np.full_like(denoised.data, 77)
    denoised.indices = np.zeros_like(denoised.indices)
    denoised._shape = (9, 9)

    fresh_selective = result.selective_counts
    fresh_denoised = result.denoised_counts
    np.testing.assert_array_equal(fresh_selective.toarray(), expected_selective)
    np.testing.assert_array_equal(fresh_denoised.toarray(), expected_denoised)
    assert fresh_selective.shape == (2, 2)
    assert fresh_denoised.shape == (2, 2)


def _valid_result_arguments():
    return {
        "selective_counts": np.array([[2, 0], [0, 3]]),
        "denoised_counts": np.array([[2.0, 0.5], [0.25, 3.0]]),
        "p_pre_zero": np.array([[0.0, 0.8], [0.6, 0.0]]),
        "latent": np.ones((2, 1)),
        "diagnostics": {"loss": 0.5},
    }


def _copy_with_dtype(value, dtype):
    copied = np.empty(np.shape(value), dtype=dtype)
    copied[...] = value
    return copied


class _MaskedArrayLike:
    def __init__(self, values, mask):
        self._values = values
        self._mask = mask

    def __array__(self, dtype=None, copy=None):
        return np.ma.array(self._values, mask=self._mask, dtype=dtype, copy=copy)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("selective_counts", np.array([1, 2])),
        ("selective_counts", np.array([[1, -1], [0, 2]])),
        ("selective_counts", np.array([[True, False], [False, True]])),
        ("selective_counts", np.array([[1 + 0j, 0], [0, 2]])),
        ("selective_counts", sparse.csr_matrix([[1.0, np.nan], [0, 2]])),
        ("denoised_counts", np.ones((2, 2, 1))),
        ("denoised_counts", np.ones((3, 2))),
        ("denoised_counts", np.array([[1.0, np.inf], [0, 2]])),
        ("denoised_counts", sparse.csr_matrix([[1.0, -0.1], [0, 2]])),
        ("p_pre_zero", np.ones((2, 1))),
        ("p_pre_zero", sparse.csr_matrix(np.zeros((2, 2)))),
        ("p_pre_zero", np.array([[0.0, np.nan], [0.5, 1.0]])),
        ("p_pre_zero", np.array([[0.0, -0.01], [0.5, 1.0]])),
        ("p_pre_zero", np.array([[0.0, 1.01], [0.5, 1.0]])),
        ("latent", np.ones(2)),
        ("latent", np.ones((3, 1))),
        ("latent", np.array([[0.0], [np.inf]])),
        ("diagnostics", []),
        ("diagnostics", {1: "not a string key"}),
        ("diagnostics", {"loss": float("nan")}),
    ],
)
def test_result_rejects_invalid_shapes_domains_and_diagnostics(field, invalid):
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    arguments[field] = invalid

    with pytest.raises((TypeError, ValueError)):
        ImputationResult(**arguments)


@pytest.mark.parametrize(
    "field",
    ["selective_counts", "denoised_counts", "p_pre_zero", "latent"],
)
def test_result_rejects_masked_arrays_for_every_dense_matrix_field(field):
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    arguments[field] = np.ma.array(arguments[field], mask=False)

    with pytest.raises(TypeError, match="masked"):
        ImputationResult(**arguments)


def test_result_rejects_masked_arrays_nested_in_diagnostics():
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    arguments["diagnostics"] = {
        "fit": {"trace": np.ma.array([1.0, 2.0], mask=[False, True])}
    }

    with pytest.raises(TypeError, match="masked"):
        ImputationResult(**arguments)


def _overflowing_duplicate_sparse_matrix(dtype):
    if np.issubdtype(dtype, np.signedinteger):
        values = np.array([np.iinfo(dtype).max, 1], dtype=dtype)
    elif np.issubdtype(dtype, np.unsignedinteger):
        values = np.array([np.iinfo(dtype).max, 1], dtype=dtype)
    else:
        values = np.array([np.finfo(dtype).max, np.finfo(dtype).max], dtype=dtype)
    return sparse.coo_matrix((values, ([0, 0], [0, 0])), shape=(1, 1))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
def test_result_rejects_duplicate_sparse_coordinates_before_overflow(dtype):
    from maskimpute import ImputationResult

    duplicate = _overflowing_duplicate_sparse_matrix(dtype)

    with pytest.raises(ValueError, match="duplicate sparse coordinates"):
        ImputationResult(
            duplicate,
            sparse.csr_matrix([[1.0]]),
            np.zeros((1, 1)),
            np.ones((1, 1)),
            {},
        )


def test_result_accepts_empty_cell_axis_without_relaxing_shape_contract():
    from maskimpute import ImputationResult

    result = ImputationResult(
        np.empty((0, 3)),
        sparse.csr_matrix((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 2)),
        {"status": "empty"},
    )

    assert result.selective_counts.shape == (0, 3)
    assert result.denoised_counts.shape == (0, 3)
    assert result.p_pre_zero.shape == (0, 3)
    assert result.latent.shape == (0, 2)


def test_pre_zero_posterior_uses_p0_as_the_numerator_and_zeros_positives():
    from maskimpute import p_pre_zero_from_counts

    observed = np.array([[0, 2]], dtype=np.int64)
    posterior = p_pre_zero_from_counts(
        observed_counts=observed,
        mu=np.full((1, 2), np.log(4.0)),
        alpha=0.0,
        pi=0.4,
    )

    assert posterior.shape == observed.shape
    assert posterior.dtype == np.float64
    assert posterior[0, 0] == pytest.approx(5.0 / 11.0)
    assert posterior[0, 0] != pytest.approx(3.0 / 11.0)
    assert posterior[0, 1] == 0.0


def test_pre_zero_posterior_matches_scipy_nb_and_poisson_oracles():
    from maskimpute import p_pre_zero_from_counts

    observed = np.zeros((2, 3), dtype=np.int64)
    mu = np.array([[0.0, 0.5, 6.0], [1.0, 4.0, 20.0]])
    alpha = np.array([[0.0, 0.0, 0.2], [0.1, 1.0, 2.0]])
    pi = np.array([[0.1], [0.65]])

    p0 = np.empty_like(mu)
    poisson_entries = alpha == 0
    p0[poisson_entries] = poisson.pmf(0, mu[poisson_entries])
    nb_entries = ~poisson_entries
    size = 1.0 / alpha[nb_entries]
    success_probability = size / (size + mu[nb_entries])
    p0[nb_entries] = nbinom.pmf(0, size, success_probability)
    expected = p0 / (pi + (1.0 - pi) * p0)

    np.testing.assert_allclose(
        p_pre_zero_from_counts(observed, mu, alpha, pi),
        expected,
        rtol=2e-14,
        atol=0.0,
    )


def test_pre_zero_posterior_for_mu6_alpha_point2_pi_point3():
    from maskimpute import p_pre_zero_from_counts

    p0 = (1.0 + 0.2 * 6.0) ** (-1.0 / 0.2)
    expected = p0 / (0.3 + 0.7 * p0)

    result = p_pre_zero_from_counts(np.array([0]), 6.0, 0.2, 0.3)

    assert result[0] == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize(
    "observed",
    [
        np.array([True, False]),
        np.array([-1, 0]),
        np.array([0.5, 0.0]),
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.array([0 + 0j, 1 + 0j]),
        np.array(["0", "1"]),
        np.array([0, 1], dtype=object),
    ],
)
def test_pre_zero_rejects_non_raw_count_observations(observed):
    from maskimpute import p_pre_zero_from_counts

    with pytest.raises((TypeError, ValueError)):
        p_pre_zero_from_counts(observed, 1.0, 0.0, 0.2)


def test_pre_zero_rejects_masked_observed_counts_before_discarding_the_mask():
    from maskimpute import p_pre_zero_from_counts

    observed = np.ma.array([0, 4], mask=[False, True])

    with pytest.raises(TypeError, match="masked"):
        p_pre_zero_from_counts(observed, 1.0, 0.0, 0.2)


@pytest.mark.parametrize("field", ["mu", "alpha", "pi"])
def test_pre_zero_rejects_masked_model_parameters_before_array_coercion(field):
    from maskimpute import p_pre_zero_from_counts

    parameters = {"mu": 2.0, "alpha": 0.2, "pi": 0.3}
    parameters[field] = np.ma.array([parameters[field]], mask=[True])

    with pytest.raises(TypeError, match="masked"):
        p_pre_zero_from_counts(np.array([0]), **parameters)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
def test_pre_zero_rejects_duplicate_sparse_coordinates_before_overflow(dtype):
    from maskimpute import p_pre_zero_from_counts

    duplicate = _overflowing_duplicate_sparse_matrix(dtype)

    with pytest.raises(ValueError, match="duplicate sparse coordinates"):
        p_pre_zero_from_counts(duplicate, 1.0, 0.0, 0.2)


def test_pre_zero_accepts_integral_float_unsigned_and_sparse_raw_counts():
    from maskimpute import p_pre_zero_from_counts

    expected = p_pre_zero_from_counts(np.array([[0, 2]], dtype=np.int64), 2, 0, 0.2)

    np.testing.assert_array_equal(
        p_pre_zero_from_counts(np.array([[0.0, 2.0]]), 2, 0, 0.2), expected
    )
    np.testing.assert_array_equal(
        p_pre_zero_from_counts(np.array([[0, 2]], dtype=np.uint64), 2, 0, 0.2),
        expected,
    )


def test_pre_zero_accepts_every_supported_exact_sparse_storage_type():
    from maskimpute import p_pre_zero_from_counts

    expected = p_pre_zero_from_counts(np.array([[0, 2]]), 2, 0, 0.2)
    for name in (
        "bsr_matrix",
        "coo_matrix",
        "csc_matrix",
        "csr_matrix",
        "dia_matrix",
        "dok_matrix",
        "lil_matrix",
        "bsr_array",
        "coo_array",
        "csc_array",
        "csr_array",
        "dia_array",
        "dok_array",
        "lil_array",
    ):
        constructor = getattr(sparse, name, None)
        if constructor is not None:
            observed = constructor(np.array([[0, 2]], dtype=np.int64))
            np.testing.assert_array_equal(
                p_pre_zero_from_counts(observed, 2, 0, 0.2),
                expected,
            )


def test_pre_zero_rejects_sparse_conversion_shadows_without_execution():
    from maskimpute import p_pre_zero_from_counts

    observed = sparse.coo_matrix([[0, 2]])
    calls = {"copy": 0, "tocoo": 0, "tocsr": 0, "toarray": 0}

    def hostile_for(name):
        def hostile(self, *args, **kwargs):
            del self, args, kwargs
            calls[name] += 1
            raise AssertionError("sparse conversion hook must never execute")

        return hostile

    for current_name in tuple(calls):
        setattr(observed, current_name, MethodType(hostile_for(current_name), observed))

    with pytest.raises(TypeError, match="callable sparse instance shadow"):
        p_pre_zero_from_counts(observed, 2, 0, 0.2)
    assert calls == {"copy": 0, "tocoo": 0, "tocsr": 0, "toarray": 0}


def test_pre_zero_rejects_malformed_csr_before_scipy_conversion():
    script = """\
import numpy as np
from scipy import sparse
from maskimpute import p_pre_zero_from_counts

observed = sparse.csr_matrix(np.array([[0, 2], [1, 0]], dtype=np.int64))
observed.indptr[-1] = 1_000_000
try:
    p_pre_zero_from_counts(observed, 2.0, 0.0, 0.2)
except ValueError as error:
    assert "invalid sparse structure" in str(error), error
else:
    raise AssertionError("malformed CSR was accepted")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, (
        f"returncode={completed.returncode}\nstdout={completed.stdout}\n"
        f"stderr={completed.stderr}"
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("mu", -0.1),
        ("mu", float("nan")),
        ("mu", float("inf")),
        ("mu", True),
        ("mu", 1 + 0j),
        ("alpha", -0.1),
        ("alpha", float("nan")),
        ("alpha", float("inf")),
        ("alpha", False),
        ("alpha", "0.2"),
        ("pi", -0.01),
        ("pi", 1.01),
        ("pi", float("nan")),
        ("pi", float("inf")),
        ("pi", True),
        ("pi", 0.2 + 0j),
    ],
)
def test_pre_zero_rejects_invalid_model_parameters(field, invalid):
    from maskimpute import p_pre_zero_from_counts

    parameters = {"mu": 2.0, "alpha": 0.2, "pi": 0.3}
    parameters[field] = invalid
    with pytest.raises((TypeError, ValueError)):
        p_pre_zero_from_counts(np.array([0]), **parameters)


@pytest.mark.parametrize(
    ("mu", "pi"),
    [(np.array([0.0, 1.0]), 0.2), (1.0, np.array([1.0, 0.2]))],
)
def test_pre_zero_rejects_model_impossible_positive_observations(mu, pi):
    from maskimpute import p_pre_zero_from_counts

    with pytest.raises(ValueError, match="positive"):
        p_pre_zero_from_counts(np.array([2, 0]), mu, 0.2, pi)


def test_pre_zero_boundaries_and_extreme_parameters_are_log_space_stable():
    from maskimpute import p_pre_zero_from_counts

    observed = np.zeros(6, dtype=np.int64)
    mu = np.array([0.0, 1_000.0, 1_000.0, 2.0, 1e308, 2.0])
    alpha = np.array([0.0, 0.0, 0.0, 0.0, 1e308, 1e-300])
    pi = np.array([0.7, 0.0, 0.3, 1.0, 0.5, 0.4])

    result = p_pre_zero_from_counts(observed, mu, alpha, pi)

    assert np.all(np.isfinite(result))
    assert result[0] == 1.0
    assert result[1] == 1.0
    assert result[2] == 0.0
    assert result[3] == pytest.approx(np.exp(-2.0), rel=1e-15)
    assert 0.0 < result[4] <= 1.0
    poisson_limit_p0 = np.exp(-2.0)
    poisson_limit = poisson_limit_p0 / (0.4 + 0.6 * poisson_limit_p0)
    assert result[5] == pytest.approx(poisson_limit, rel=1e-14)


def test_pre_zero_broadcasts_parameters_into_but_never_beyond_observed_shape():
    from maskimpute import p_pre_zero_from_counts

    observed = np.array([[0, 2, 0], [1, 0, 0]], dtype=np.int64)
    mu = np.array([0.5, 2.0, 6.0])
    alpha = np.array([[0.0], [0.2]])
    pi = np.array([0.1, 0.3, 0.6])
    observed_before = observed.copy()
    mu_before = mu.copy()
    alpha_before = alpha.copy()
    pi_before = pi.copy()

    result = p_pre_zero_from_counts(observed, mu, alpha, pi)

    expected = np.zeros_like(result)
    for row, column in zip(*np.nonzero(observed == 0)):
        local_mu = mu[column]
        local_alpha = alpha[row, 0]
        p0 = (
            np.exp(-local_mu)
            if local_alpha == 0
            else (1.0 + local_alpha * local_mu) ** (-1.0 / local_alpha)
        )
        expected[row, column] = p0 / (pi[column] + (1 - pi[column]) * p0)

    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=0.0)
    assert result.dtype == np.float64
    assert result.shape == observed.shape
    np.testing.assert_array_equal(observed, observed_before)
    np.testing.assert_array_equal(mu, mu_before)
    np.testing.assert_array_equal(alpha, alpha_before)
    np.testing.assert_array_equal(pi, pi_before)

    with pytest.raises(ValueError, match="broadcast"):
        p_pre_zero_from_counts(np.zeros((2, 1), dtype=int), np.ones((2, 3)), 0, 0.2)


def test_pre_zero_supports_empty_and_scalar_observations_without_shape_expansion():
    from maskimpute import p_pre_zero_from_counts

    empty = p_pre_zero_from_counts(np.empty((0, 3), dtype=int), np.ones(3), 0, 0.2)
    scalar = p_pre_zero_from_counts(np.asarray(0), 2.0, 0.0, 0.2)

    assert empty.shape == (0, 3)
    assert empty.dtype == np.float64
    assert scalar.shape == ()
    assert scalar.dtype == np.float64


def test_pre_zero_is_monotone_in_mean_dispersion_and_loss_probability():
    from maskimpute import p_pre_zero_from_counts

    by_mean = p_pre_zero_from_counts(np.zeros(4, dtype=int), [0.5, 1, 2, 4], 0.2, 0.3)
    by_dispersion = p_pre_zero_from_counts(
        np.zeros(4, dtype=int), 4.0, [0.0, 0.1, 0.5, 1.0], 0.3
    )
    by_loss = p_pre_zero_from_counts(
        np.zeros(4, dtype=int), 2.0, 0.2, [0.05, 0.2, 0.5, 0.9]
    )

    assert np.all(np.diff(by_mean) < 0)
    assert np.all(np.diff(by_dispersion) > 0)
    assert np.all(np.diff(by_loss) < 0)


def test_pre_zero_does_not_clip_a_representable_loss_probability():
    from maskimpute import p_pre_zero_from_counts

    smallest_positive = np.nextafter(np.float64(0), np.float64(1))
    result = p_pre_zero_from_counts(np.array([0]), 1_000.0, 0.0, smallest_positive)
    log_expected = -1_000.0 - np.logaddexp(np.log(smallest_positive), -1_000.0)

    assert 0.0 < result[0] < 1.0
    assert result[0] == pytest.approx(np.exp(log_expected), rel=2e-13)


def test_pre_zero_accepts_finite_extended_precision_parameters_without_cast_overflow():
    from maskimpute import p_pre_zero_from_counts

    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("long double has no wider finite range on this platform")
    huge_finite_mean = np.longdouble("1e4000")
    assert np.isfinite(huge_finite_mean)

    result = p_pre_zero_from_counts(np.array([0]), huge_finite_mean, 0.0, 0.2)

    assert result.dtype == np.float64
    assert result[0] == 0.0


@pytest.mark.parametrize("field", ["observed_counts", "mu", "alpha", "pi"])
def test_pre_zero_rejects_masked_values_nested_in_python_sequences(field):
    from maskimpute import p_pre_zero_from_counts

    arguments = {
        "observed_counts": [[0]],
        "mu": [[2.0]],
        "alpha": [[0.2]],
        "pi": [[0.3]],
    }
    arguments[field] = [[np.ma.array(0.0, mask=True)]]

    with pytest.raises(TypeError, match="masked"):
        p_pre_zero_from_counts(**arguments)


@pytest.mark.parametrize("field", ["observed_counts", "mu", "alpha", "pi"])
def test_pre_zero_rejects_masked_values_nested_in_object_arrays(field):
    from maskimpute import p_pre_zero_from_counts

    nested = np.empty((1,), dtype=object)
    nested[0] = (np.ma.array(0.0, mask=True),)
    arguments = {
        "observed_counts": np.array([0]),
        "mu": np.array([2.0]),
        "alpha": np.array([0.2]),
        "pi": np.array([0.3]),
    }
    arguments[field] = nested

    with pytest.raises(TypeError, match="masked"):
        p_pre_zero_from_counts(**arguments)


@pytest.mark.parametrize("sparse_type", [sparse.lil_matrix, sparse.lil_array])
def test_pre_zero_rejects_masked_values_in_nested_lil_data_rows(sparse_type):
    from maskimpute import p_pre_zero_from_counts

    observed = sparse_type((1, 1), dtype=np.float64)
    observed.rows[0] = [0]
    observed.data[0] = [np.ma.array(1.0, mask=True)]

    with pytest.raises(TypeError, match="masked"):
        p_pre_zero_from_counts(observed, 2.0, 0.2, 0.3)


def test_pre_zero_does_not_python_iterate_ordinary_numeric_ndarrays():
    from maskimpute import p_pre_zero_from_counts

    class NonIterableNumericArray(np.ndarray):
        def __iter__(self):
            raise AssertionError("numeric ndarray was recursively scanned")

    observed = np.zeros((2, 3), dtype=np.int64).view(NonIterableNumericArray)
    mean = np.ones((2, 3), dtype=np.float64).view(NonIterableNumericArray)

    result = p_pre_zero_from_counts(observed, mean, 0.2, 0.3)

    assert result.shape == (2, 3)


@pytest.mark.parametrize("field", ["observed_counts", "mu", "alpha", "pi"])
def test_pre_zero_rejects_array_protocol_results_that_are_masked(field):
    from maskimpute import p_pre_zero_from_counts

    arguments = {
        "observed_counts": np.array([0]),
        "mu": np.array([2.0]),
        "alpha": np.array([0.2]),
        "pi": np.array([0.3]),
    }
    arguments[field] = _MaskedArrayLike([0.0], [True])

    with pytest.raises(TypeError, match="masked"):
        p_pre_zero_from_counts(**arguments)


@pytest.mark.parametrize(
    "field", ["selective_counts", "denoised_counts", "p_pre_zero", "latent"]
)
def test_result_rejects_masked_values_nested_in_every_matrix(field):
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    nested = np.empty((2, 2), dtype=object)
    nested[:] = 0.0
    nested[0, 0] = [np.ma.array(0.0, mask=True)]
    arguments[field] = nested

    with pytest.raises(TypeError, match="masked"):
        ImputationResult(**arguments)


@pytest.mark.parametrize("field", ["selective_counts", "denoised_counts"])
@pytest.mark.parametrize("sparse_type", [sparse.lil_matrix, sparse.lil_array])
def test_result_rejects_masked_values_in_nested_lil_data_rows(field, sparse_type):
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    nested = sparse_type((2, 2), dtype=np.float64)
    nested.rows[0] = [0]
    nested.data[0] = [np.ma.array(1.0, mask=True)]
    arguments[field] = nested

    with pytest.raises(TypeError, match="masked"):
        ImputationResult(**arguments)


@pytest.mark.parametrize(
    "field", ["selective_counts", "denoised_counts", "p_pre_zero", "latent"]
)
def test_result_rejects_array_protocol_results_that_are_masked(field):
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    arguments[field] = _MaskedArrayLike([[0.0, 0.0], [0.0, 0.0]], [[True, False]] * 2)

    with pytest.raises(TypeError, match="masked"):
        ImputationResult(**arguments)


@pytest.mark.parametrize("field", ["observed_counts", "mu", "alpha", "pi"])
def test_pre_zero_rejects_arrays_with_dtype_metadata(field):
    from maskimpute import p_pre_zero_from_counts

    annotated_dtype = np.dtype(np.float64, metadata={"units": ["caller-owned"]})
    arguments = {
        "observed_counts": np.array([0]),
        "mu": np.array([2.0]),
        "alpha": np.array([0.2]),
        "pi": np.array([0.3]),
    }
    arguments[field] = _copy_with_dtype(arguments[field], annotated_dtype)
    assert arguments[field].dtype.metadata is not None

    with pytest.raises(TypeError, match="metadata"):
        p_pre_zero_from_counts(**arguments)


@pytest.mark.parametrize(
    "field", ["selective_counts", "denoised_counts", "p_pre_zero", "latent"]
)
def test_result_rejects_caller_owned_dtype_metadata_in_every_matrix(field):
    from maskimpute import ImputationResult

    mutable_metadata = {"labels": ["before"]}
    annotated_dtype = np.dtype(np.float64, metadata=mutable_metadata)
    arguments = _valid_result_arguments()
    arguments[field] = _copy_with_dtype(arguments[field], annotated_dtype)
    assert arguments[field].dtype.metadata is not None

    with pytest.raises(TypeError, match="metadata"):
        ImputationResult(**arguments)

    mutable_metadata["labels"].append("after")
    assert annotated_dtype.metadata["labels"] == ["before", "after"]


def test_result_rejects_caller_owned_dtype_metadata_in_diagnostics():
    from maskimpute import ImputationResult

    mutable_metadata = {"labels": ["before"]}
    annotated_dtype = np.dtype(np.float64, metadata=mutable_metadata)
    arguments = _valid_result_arguments()
    trace = _copy_with_dtype([1.0, 0.5], annotated_dtype)
    assert trace.dtype.metadata is not None
    arguments["diagnostics"] = {"trace": trace}

    with pytest.raises(TypeError, match="metadata"):
        ImputationResult(**arguments)

    mutable_metadata["labels"].append("after")
    assert annotated_dtype.metadata["labels"] == ["before", "after"]


def test_result_rejects_structured_diagnostic_with_nonfinite_numeric_field():
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    arguments["diagnostics"] = {
        "trace": np.array([(np.nan,)], dtype=[("loss", np.float64)])
    }

    with pytest.raises(TypeError, match="dtype"):
        ImputationResult(**arguments)


@pytest.mark.parametrize(
    "unsupported",
    [
        np.array([1.0 + 0.0j]),
        np.array(["2026-07-12"], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[D]"),
        np.array([b"opaque"], dtype="V6"),
    ],
)
def test_result_rejects_unsupported_diagnostic_array_dtypes(unsupported):
    from maskimpute import ImputationResult

    arguments = _valid_result_arguments()
    arguments["diagnostics"] = {"value": unsupported}

    with pytest.raises(TypeError, match="dtype"):
        ImputationResult(**arguments)
