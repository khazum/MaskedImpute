import dataclasses
import hashlib
import json
import subprocess
import sys
from types import MethodType

import numpy as np
import pytest
from scipy import sparse


def _counts() -> np.ndarray:
    return np.array(
        [
            [8, 0, 1, 0],
            [5, 0, 0, 1],
            [3, 1, 0, 0],
            [1, 2, 0, 0],
            [0, 4, 1, 0],
            [0, 7, 0, 1],
        ],
        dtype=np.int64,
    )


def test_count_model_public_api_imports_without_torch():
    script = r"""
import builtins

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("count-model API must not import torch")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from maskimpute import (
    PreZeroCountModelConfig,
    PreZeroCountModelScore,
    fit_p_pre_zero_count_model,
)
assert PreZeroCountModelConfig.__module__ == "maskimpute.count_model"
assert PreZeroCountModelScore.__module__ == "maskimpute.count_model"
assert fit_p_pre_zero_count_model.__module__ == "maskimpute.count_model"
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_count_model_config_is_strict_and_immutable():
    from maskimpute import PreZeroCountModelConfig

    config = PreZeroCountModelConfig()

    assert config.n_folds == 5
    assert config.use_library_size_exposure is True
    assert config.mean_prior_strength == pytest.approx(1.0)
    assert config.mean_floor == pytest.approx(1e-8)
    assert config.dispersion_prior_strength == pytest.approx(10.0)
    assert config.link_bins == 64
    assert config.link_max_iter == 200
    assert config.link_tolerance == pytest.approx(1e-10)
    assert config.link_bound == pytest.approx(30.0)
    assert not hasattr(config, "__dict__")

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.n_folds = 3


@pytest.mark.parametrize(
    "override",
    [
        {"n_folds": True},
        {"n_folds": 1},
        {"use_library_size_exposure": 1},
        {"mean_prior_strength": True},
        {"mean_prior_strength": -1.0},
        {"mean_floor": 0.0},
        {"mean_floor": 1.1},
        {"mean_floor": float("nan")},
        {"dispersion_prior_strength": -1.0},
        {"link_bins": 1},
        {"link_max_iter": 0},
        {"link_tolerance": 0.0},
        {"link_bound": 0.0},
        {"link_bound": 31.0},
    ],
)
def test_count_model_config_rejects_invalid_values(override):
    from maskimpute import PreZeroCountModelConfig

    with pytest.raises((TypeError, ValueError)):
        PreZeroCountModelConfig(**override)


def test_cross_fitted_score_has_finite_supported_audit_arrays():
    from maskimpute import fit_p_pre_zero_count_model

    counts = _counts()
    score = fit_p_pre_zero_count_model(counts)

    assert score.shape == counts.shape
    assert len(score.fold_models) == 5
    assert score.p_pre_zero.shape == counts.shape
    assert score.mu.shape == counts.shape
    assert score.alpha.shape == counts.shape
    assert score.pi.shape == counts.shape
    assert score.fold_ids.shape == (counts.shape[0],)
    assert np.all(np.isfinite(score.p_pre_zero))
    assert np.all(np.isfinite(score.mu))
    assert np.all(np.isfinite(score.alpha))
    assert np.all(np.isfinite(score.pi))
    assert np.all((score.p_pre_zero >= 0) & (score.p_pre_zero <= 1))
    assert np.all(score.p_pre_zero[counts > 0] == 0)
    assert np.all(score.mu > 0)
    assert np.all(score.alpha >= 0)
    assert np.all((score.pi >= 0) & (score.pi <= 1))
    sizes = np.bincount(score.fold_ids, minlength=5)
    assert sizes.max() - sizes.min() <= 1

    covered = []
    for record in score.fold_models:
        assert record.training_cell_count == counts.shape[0] - len(
            record.held_out_indices
        )
        assert len(record.training_input_sha256) == 64
        assert record.gene_means.shape == (counts.shape[1],)
        assert record.gene_dispersion.shape == (counts.shape[1],)
        assert np.all(record.gene_means > 0)
        assert np.all(record.gene_dispersion >= 0)
        assert record.link_slope <= 0
        assert record.link_iterations >= 0
        assert 0 <= record.clamp_fraction <= 1
        covered.extend(record.held_out_indices)
    assert sorted(covered) == list(range(counts.shape[0]))


def test_score_equals_public_bayes_formula_and_uses_effective_fold_count():
    from maskimpute import (
        PreZeroCountModelConfig,
        fit_p_pre_zero_count_model,
        p_pre_zero_from_counts,
    )

    counts = _counts()[:3]
    score = fit_p_pre_zero_count_model(
        counts,
        PreZeroCountModelConfig(n_folds=9),
    )

    expected = p_pre_zero_from_counts(counts, score.mu, score.alpha, score.pi)
    np.testing.assert_array_equal(score.p_pre_zero, expected)
    assert len(score.fold_models) == counts.shape[0]
    assert set(score.fold_ids) == set(range(counts.shape[0]))


def test_fit_is_exactly_deterministic_and_does_not_consume_numpy_rng():
    from maskimpute import fit_p_pre_zero_count_model

    np.random.seed(913)
    state = np.random.get_state()
    first = fit_p_pre_zero_count_model(_counts())
    after = np.random.get_state()
    second = fit_p_pre_zero_count_model(_counts())

    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    np.testing.assert_array_equal(first.p_pre_zero, second.p_pre_zero)
    np.testing.assert_array_equal(first.mu, second.mu)
    np.testing.assert_array_equal(first.alpha, second.alpha)
    np.testing.assert_array_equal(first.pi, second.pi)
    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    assert first.input_sha256 == second.input_sha256
    assert first.config_sha256 == second.config_sha256
    assert first.score_sha256 == second.score_sha256
    assert first.manifest == second.manifest


def test_leave_one_out_fold_parameters_exclude_the_held_out_cell():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = _counts()[:4]
    config = PreZeroCountModelConfig(n_folds=4)
    baseline = fit_p_pre_zero_count_model(counts, config)
    changed_counts = counts.copy()
    changed_counts[2] = np.array([80, 0, 0, 20])
    changed = fit_p_pre_zero_count_model(changed_counts, config)

    baseline_fold = next(
        record for record in baseline.fold_models if record.held_out_indices == (2,)
    )
    changed_fold = next(
        record for record in changed.fold_models if record.held_out_indices == (2,)
    )
    assert baseline_fold.training_input_sha256 == changed_fold.training_input_sha256
    np.testing.assert_array_equal(
        baseline_fold.gene_means,
        changed_fold.gene_means,
    )
    np.testing.assert_array_equal(
        baseline_fold.gene_dispersion,
        changed_fold.gene_dispersion,
    )
    assert baseline_fold.link_intercept == changed_fold.link_intercept
    assert baseline_fold.link_slope == changed_fold.link_slope


def test_balanced_fold_assignment_depends_on_cell_identity_not_count_values():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = _counts()
    config = PreZeroCountModelConfig(n_folds=3)
    baseline = fit_p_pre_zero_count_model(counts, config)
    changed_counts = counts.copy()
    changed_counts[2] = np.array([80, 0, 0, 20])
    changed = fit_p_pre_zero_count_model(changed_counts, config)

    np.testing.assert_array_equal(baseline.fold_ids, changed.fold_ids)
    fold_id = int(baseline.fold_ids[2])
    baseline_fold = baseline.fold_models[fold_id]
    changed_fold = changed.fold_models[fold_id]
    assert 2 in baseline_fold.held_out_indices
    assert baseline_fold.held_out_indices == changed_fold.held_out_indices
    assert baseline_fold.training_input_sha256 == changed_fold.training_input_sha256
    np.testing.assert_array_equal(
        baseline_fold.gene_means,
        changed_fold.gene_means,
    )
    np.testing.assert_array_equal(
        baseline_fold.gene_dispersion,
        changed_fold.gene_dispersion,
    )
    assert baseline_fold.link_intercept == changed_fold.link_intercept
    assert baseline_fold.link_slope == changed_fold.link_slope


def test_equivalent_dense_and_sparse_inputs_have_identical_results_and_digests():
    from maskimpute import fit_p_pre_zero_count_model

    counts = _counts()
    dense = fit_p_pre_zero_count_model(counts.astype(np.float64))

    for constructor in (sparse.coo_matrix, sparse.csc_matrix, sparse.csr_matrix):
        encoded = fit_p_pre_zero_count_model(constructor(counts))
        np.testing.assert_array_equal(encoded.p_pre_zero, dense.p_pre_zero)
        np.testing.assert_array_equal(encoded.mu, dense.mu)
        np.testing.assert_array_equal(encoded.alpha, dense.alpha)
        np.testing.assert_array_equal(encoded.pi, dense.pi)
        np.testing.assert_array_equal(encoded.fold_ids, dense.fold_ids)
        assert encoded.input_sha256 == dense.input_sha256
        assert encoded.config_sha256 == dense.config_sha256
        assert encoded.score_sha256 == dense.score_sha256
        assert encoded.manifest == dense.manifest


def test_score_arrays_and_fresh_manifest_are_immutable_snapshots():
    from maskimpute import fit_p_pre_zero_count_model

    counts = _counts()
    score = fit_p_pre_zero_count_model(counts)
    original = score.p_pre_zero.copy()
    counts[:] = 999

    for value in (
        score.p_pre_zero,
        score.mu,
        score.alpha,
        score.pi,
        score.fold_ids,
        score.fold_models[0].gene_means,
        score.fold_models[0].gene_dispersion,
    ):
        assert value.flags.writeable is False
        assert isinstance(value.base, bytes)
        with pytest.raises(ValueError):
            value.flat[0] = 0

    np.testing.assert_array_equal(score.p_pre_zero, original)
    first_manifest = score.manifest
    canonical = json.dumps(first_manifest, sort_keys=True, separators=(",", ":"))
    first_manifest["artifact_type"] = "tampered"
    assert score.manifest["artifact_type"] == "maskimpute_count_model_score"
    assert (
        json.dumps(score.manifest, sort_keys=True, separators=(",", ":")) == canonical
    )


def test_manifest_digests_bind_canonical_config_model_and_score_payload():
    from maskimpute import fit_p_pre_zero_count_model

    score = fit_p_pre_zero_count_model(_counts())
    manifest = score.manifest
    unsigned = dict(manifest)
    unsigned.pop("score_sha256")
    canonical_unsigned = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    canonical_config = json.dumps(
        manifest["config"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

    assert manifest["model"] == {
        "count_family": "negative_binomial_2_with_poisson_limit",
        "dispersion": "gene_moment_estimate_with_global_shrinkage",
        "exposure": "training_reference_library_size",
        "mean": "exposure_times_shrunk_gene_mean_with_absolute_floor",
        "score": "bayes_pre_capture_zero_given_observed_zero",
        "total_zero_link": "bounded_nonincreasing_logistic_on_log_mean",
    }
    assert hashlib.sha256(canonical_config).hexdigest() == score.config_sha256
    assert hashlib.sha256(canonical_unsigned).hexdigest() == score.score_sha256
    assert manifest["score_sha256"] == score.score_sha256


def test_score_for_counts_revalidates_exact_canonical_input():
    from maskimpute import fit_p_pre_zero_count_model

    counts = _counts()
    score = fit_p_pre_zero_count_model(counts)

    np.testing.assert_array_equal(
        score.score_for_counts(counts.copy()), score.p_pre_zero
    )
    np.testing.assert_array_equal(
        score.score_for_counts(sparse.csr_matrix(counts)), score.p_pre_zero
    )
    changed = counts.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="does not match"):
        score.score_for_counts(changed)


def test_score_rejects_subclass_and_detects_internal_digest_tampering():
    from maskimpute import PreZeroCountModelScore, fit_p_pre_zero_count_model

    score = fit_p_pre_zero_count_model(_counts())

    class ScoreSubclass(PreZeroCountModelScore):
        pass

    forged = object.__new__(ScoreSubclass)
    for slot in PreZeroCountModelScore.__slots__:
        object.__setattr__(forged, slot, getattr(score, slot))
    with pytest.raises(TypeError, match="exact PreZeroCountModelScore"):
        forged.score_for_counts(_counts())

    object.__setattr__(score, "_score_sha256", "0" * 64)
    with pytest.raises(ValueError, match="integrity"):
        score.score_for_counts(_counts())
    with pytest.raises(ValueError, match="integrity"):
        _ = score.p_pre_zero

    malformed = fit_p_pre_zero_count_model(_counts())
    object.__setattr__(malformed, "_config_bytes", None)
    with pytest.raises(ValueError, match="integrity"):
        _ = malformed.manifest


def test_fit_revalidates_config_after_low_level_mutation():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    config = PreZeroCountModelConfig()
    object.__setattr__(config, "n_folds", 1)

    with pytest.raises((TypeError, ValueError), match="config|n_folds"):
        fit_p_pre_zero_count_model(_counts(), config)


def test_extreme_finite_shrinkage_strengths_remain_warning_free_and_finite():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = np.array(
        [
            [2**53, 0],
            [2**53, 1],
            [2**53, 2],
        ],
        dtype=np.uint64,
    )
    score = fit_p_pre_zero_count_model(
        counts,
        PreZeroCountModelConfig(
            n_folds=3,
            mean_prior_strength=1e308,
            dispersion_prior_strength=1e308,
        ),
    )

    assert np.all(np.isfinite(score.mu))
    assert np.all(np.isfinite(score.alpha))
    assert np.all(np.isfinite(score.p_pre_zero))


def test_nb_zero_probability_is_stable_when_finite_product_overflows():
    from maskimpute.count_model import _nb_zero_probability

    mu = np.array([[1e200, 1e308]], dtype=np.float64)
    alpha = np.array([[1e200, 1e308]], dtype=np.float64)
    expected_negative_log = np.logaddexp(0.0, np.log(alpha) + np.log(mu)) / alpha

    probability = _nb_zero_probability(mu, alpha)

    np.testing.assert_array_equal(probability, np.exp(-expected_negative_log))
    assert np.all(np.isfinite(probability))
    assert np.all((probability >= 0) & (probability <= 1))


def test_finite_extreme_internal_totals_fail_closed_without_runtime_warning():
    from maskimpute import PreZeroCountModelConfig
    from maskimpute.count_model import _fit_gene_parameters, _library_exposures

    extreme = np.full((2, 2), 1e308, dtype=np.float64)
    with pytest.raises(FloatingPointError, match="library|exposure"):
        _library_exposures(extreme, extreme, True)
    with pytest.raises(FloatingPointError, match="gene|exposure|moment"):
        _fit_gene_parameters(
            extreme,
            np.ones(2, dtype=np.float64),
            PreZeroCountModelConfig(),
        )


@pytest.mark.parametrize(
    ("counts", "fallback"),
    [
        (np.zeros((5, 3), dtype=np.int64), "all_observed_zeros"),
        (np.ones((5, 3), dtype=np.int64), "no_observed_zeros"),
    ],
)
def test_single_class_folds_use_explicit_warning_free_fallbacks(counts, fallback):
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    first = fit_p_pre_zero_count_model(
        counts,
        PreZeroCountModelConfig(n_folds=3),
    )
    second = fit_p_pre_zero_count_model(
        counts,
        PreZeroCountModelConfig(n_folds=3),
    )

    assert all(record.link_converged is False for record in first.fold_models)
    assert all(record.link_fallback == fallback for record in first.fold_models)
    assert all(record.link_slope == 0 for record in first.fold_models)
    np.testing.assert_array_equal(first.p_pre_zero, second.p_pre_zero)
    assert first.score_sha256 == second.score_sha256


def test_constant_log_mean_fold_records_explicit_fallback_and_finite_scores():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=np.int64,
    )
    score = fit_p_pre_zero_count_model(
        counts,
        PreZeroCountModelConfig(
            n_folds=2,
            use_library_size_exposure=False,
            mean_prior_strength=0.0,
            dispersion_prior_strength=0.0,
        ),
    )

    assert all(
        record.link_fallback == "constant_log_mean" for record in score.fold_models
    )
    assert all(record.link_slope == 0 for record in score.fold_models)
    assert all(record.exposure_reference == 1 for record in score.fold_models)
    assert np.all(np.isfinite(score.p_pre_zero))


def test_fitted_zero_link_is_monotone_nonincreasing_in_log_mean():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = np.array(
        [
            [20, 8, 4, 0, 0, 0],
            [18, 7, 3, 1, 0, 0],
            [16, 6, 2, 1, 0, 0],
            [14, 5, 2, 1, 0, 0],
            [12, 4, 1, 0, 0, 0],
            [10, 3, 1, 0, 0, 0],
            [8, 2, 1, 0, 0, 0],
            [6, 1, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    score = fit_p_pre_zero_count_model(
        counts,
        PreZeroCountModelConfig(n_folds=3),
    )

    assert all(record.link_slope <= 0 for record in score.fold_models)
    assert any(record.link_slope < 0 for record in score.fold_models)


class _ChangingArrayProtocol:
    def __init__(self, first: np.ndarray, second: np.ndarray):
        self.first = first
        self.second = second
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        del copy
        self.calls += 1
        value = self.first if self.calls == 1 else self.second
        return np.asarray(value, dtype=dtype)


def test_custom_dense_array_protocol_is_materialized_exactly_once():
    from maskimpute import fit_p_pre_zero_count_model

    protocol = _ChangingArrayProtocol(
        _counts(),
        np.full(_counts().shape, -1, dtype=np.int64),
    )
    expected = fit_p_pre_zero_count_model(_counts())

    actual = fit_p_pre_zero_count_model(protocol)

    assert protocol.calls == 1
    assert actual.input_sha256 == expected.input_sha256
    np.testing.assert_array_equal(actual.p_pre_zero, expected.p_pre_zero)


@pytest.mark.parametrize(
    "invalid",
    [
        np.empty((0, 2), dtype=np.int64),
        np.empty((2, 0), dtype=np.int64),
        np.array([[1, 0]], dtype=np.int64),
        np.array([[True, False], [False, True]]),
        np.array([[1, 0], [0, 2]], dtype=object),
        np.array([[1 + 0j, 0], [0, 2]], dtype=np.complex128),
        np.array([[1.5, 0], [0, 2.0]]),
        np.array([[1, -1], [0, 2]]),
        np.array([[1.0, np.inf], [0, 2.0]]),
        np.array([[2**53 + 1, 0], [0, 2]], dtype=np.uint64),
        np.ma.array([[1, 0], [0, 2]], mask=False),
        np.array(
            [[1, 0], [0, 2]],
            dtype=np.dtype(np.int64, metadata={"semantic": "hidden"}),
        ),
        sparse.coo_matrix(
            (
                np.array([1, 2, 3]),
                (np.array([0, 0, 1]), np.array([0, 0, 1])),
            ),
            shape=(2, 2),
        ),
    ],
)
def test_count_model_rejects_ambiguous_or_lossy_count_inputs(invalid):
    from maskimpute import fit_p_pre_zero_count_model

    with pytest.raises((TypeError, ValueError)):
        fit_p_pre_zero_count_model(invalid)


def test_sparse_subclass_is_rejected_without_invoking_conversion_hook():
    from maskimpute import fit_p_pre_zero_count_model

    class HostileSparse(sparse.csr_matrix):
        calls = 0

        def tocoo(self, copy=False):
            type(self).calls += 1
            return super().tocoo(copy=copy)

    counts = HostileSparse(_counts())

    with pytest.raises(TypeError, match="exact supported SciPy sparse type"):
        fit_p_pre_zero_count_model(counts)
    assert HostileSparse.calls == 0


def test_exact_sparse_instance_conversion_hook_is_bypassed():
    from maskimpute import fit_p_pre_zero_count_model

    counts = sparse.csr_matrix(_counts())
    calls = 0

    def hostile_tocoo(self, copy=False):
        nonlocal calls
        calls += 1
        coordinates = type(self).tocoo(self, copy=copy)
        coordinates.data[:] = -1
        return coordinates

    counts.tocoo = MethodType(hostile_tocoo, counts)
    expected = fit_p_pre_zero_count_model(_counts())

    actual = fit_p_pre_zero_count_model(counts)

    assert calls == 0
    assert actual.input_sha256 == expected.input_sha256
    np.testing.assert_array_equal(actual.p_pre_zero, expected.p_pre_zero)


def test_nested_masked_sparse_storage_is_rejected_before_conversion():
    from maskimpute import fit_p_pre_zero_count_model

    counts = sparse.lil_matrix(np.array([[1, 0], [0, 2]], dtype=np.int64))
    counts.data[0][0] = np.ma.array(1, mask=True)

    with pytest.raises(TypeError, match="masked"):
        fit_p_pre_zero_count_model(counts)
