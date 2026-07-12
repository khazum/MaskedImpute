from fnmatch import fnmatchcase
from dataclasses import FrozenInstanceError
from pathlib import Path
import tomllib

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


def _valid_result_arguments():
    return {
        "selective_counts": np.array([[2, 0], [0, 3]]),
        "denoised_counts": np.array([[2.0, 0.5], [0.25, 3.0]]),
        "p_pre_zero": np.array([[0.0, 0.8], [0.6, 0.0]]),
        "latent": np.ones((2, 1)),
        "diagnostics": {"loss": 0.5},
    }


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
    np.testing.assert_array_equal(
        p_pre_zero_from_counts(sparse.csr_matrix([[0, 2]]), 2, 0, 0.2),
        expected,
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
