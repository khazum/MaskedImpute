import dataclasses
from collections.abc import Sequence
import hashlib
import inspect
import json
import subprocess
import sys
from types import MethodType

import numpy as np
import pytest
from scipy import sparse


_SPARSE_CONSTRUCTORS = tuple(
    constructor
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
    )
    if (constructor := getattr(sparse, name, None)) is not None
)


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


def _cell_ids(count: int = 6) -> tuple[str, ...]:
    return tuple(f"external-cell-{index:03d}" for index in range(count))


class _StatefulCellIds(Sequence):
    def __init__(self, first: tuple[str, ...], second: tuple[str, ...]):
        self._values = (first, second)
        self.calls = 0

    def __len__(self):
        return len(self._values[0])

    def __getitem__(self, index):
        return self._values[0][index]

    def __iter__(self):
        index = min(self.calls, 1)
        self.calls += 1
        return iter(self._values[index])


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


def test_count_model_public_api_requires_external_cell_ids():
    from maskimpute import fit_p_pre_zero_count_model
    from maskimpute.count_model import PreZeroCountModelScore

    assert tuple(inspect.signature(fit_p_pre_zero_count_model).parameters) == (
        "observed_counts",
        "cell_ids",
        "config",
    )
    assert tuple(
        inspect.signature(PreZeroCountModelScore.score_for_counts).parameters
    ) == (
        "self",
        "observed_counts",
        "cell_ids",
    )
    with pytest.raises(TypeError):
        fit_p_pre_zero_count_model(_counts())


@pytest.mark.parametrize(
    "cell_ids",
    [
        (),
        _cell_ids(5),
        (*_cell_ids(5), "external-cell-004"),
        (*_cell_ids(5), ""),
        (*_cell_ids(5), " \t"),
        (*_cell_ids(5), 6),
        (*_cell_ids(5), "\ud800"),
        "six-cell-identifiers",
        {"cell-0", "cell-1", "cell-2", "cell-3", "cell-4", "cell-5"},
        (value for value in _cell_ids()),
    ],
)
def test_external_cell_ids_are_strict_unique_nonempty_ordered_strings(cell_ids):
    from maskimpute import fit_p_pre_zero_count_model

    with pytest.raises((TypeError, ValueError), match="cell_ids"):
        fit_p_pre_zero_count_model(_counts(), cell_ids)


def test_cell_ids_are_snapshotted_once_and_bound_without_exposing_values():
    from maskimpute import fit_p_pre_zero_count_model

    original = _cell_ids()
    protocol = _StatefulCellIds(
        original,
        tuple(f"swapped-{index}" for index in range(len(original))),
    )

    score = fit_p_pre_zero_count_model(_counts(), protocol)

    assert protocol.calls == 1
    identity = score.manifest["cell_identity"]
    assert identity == {
        "assignment": "balanced_sha256_external_cell_id_order_round_robin",
        "canonical_training_order": "sha256_external_cell_id",
        "cell_count": len(original),
        "digest_sha256": score.cell_ids_sha256,
        "source": "caller_supplied_external_cell_ids",
    }
    assert not any(value in json.dumps(score.manifest) for value in original)
    verification_protocol = _StatefulCellIds(
        original,
        tuple(f"verification-swap-{index}" for index in range(len(original))),
    )
    np.testing.assert_array_equal(
        score.score_for_counts(_counts(), verification_protocol),
        score.p_pre_zero,
    )
    assert verification_protocol.calls == 1


def test_score_verification_requires_exact_bound_cell_ids():
    from maskimpute import fit_p_pre_zero_count_model

    score = fit_p_pre_zero_count_model(_counts(), _cell_ids())
    reordered = list(_cell_ids())
    reordered[0], reordered[1] = reordered[1], reordered[0]

    with pytest.raises(ValueError, match="cell_ids.*match"):
        score.score_for_counts(_counts(), reordered)


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
    score = fit_p_pre_zero_count_model(counts, _cell_ids(len(counts)))

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
        _cell_ids(len(counts)),
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
    first = fit_p_pre_zero_count_model(_counts(), _cell_ids())
    after = np.random.get_state()
    second = fit_p_pre_zero_count_model(_counts(), _cell_ids())

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
    cell_ids = _cell_ids(len(counts))
    config = PreZeroCountModelConfig(n_folds=4)
    baseline = fit_p_pre_zero_count_model(counts, cell_ids, config)
    changed_counts = counts.copy()
    changed_counts[2] = np.array([80, 0, 0, 20])
    changed = fit_p_pre_zero_count_model(changed_counts, cell_ids, config)

    np.testing.assert_array_equal(baseline.fold_ids, changed.fold_ids)
    assert tuple(record.held_out_indices for record in baseline.fold_models) == tuple(
        record.held_out_indices for record in changed.fold_models
    )

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


def test_held_out_score_is_exactly_invariant_to_positive_redistribution():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = _counts()[:4]
    cell_ids = _cell_ids(len(counts))
    config = PreZeroCountModelConfig(n_folds=4)
    baseline = fit_p_pre_zero_count_model(counts, cell_ids, config)
    changed_counts = counts.copy()
    changed_counts[2] = np.array([1, 3, 0, 0])
    changed = fit_p_pre_zero_count_model(changed_counts, cell_ids, config)

    assert counts[2].sum() == changed_counts[2].sum()
    np.testing.assert_array_equal(counts[2] > 0, changed_counts[2] > 0)
    np.testing.assert_array_equal(baseline.fold_ids, changed.fold_ids)
    for baseline_value, changed_value in (
        (baseline.p_pre_zero[2], changed.p_pre_zero[2]),
        (baseline.mu[2], changed.mu[2]),
        (baseline.alpha[2], changed.alpha[2]),
        (baseline.pi[2], changed.pi[2]),
    ):
        np.testing.assert_array_equal(baseline_value, changed_value)


def test_balanced_fold_assignment_is_exactly_equivariant_for_paired_permutations():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = np.vstack([_counts(), _counts()[[0, 2]]])
    cell_ids = _cell_ids(len(counts))
    config = PreZeroCountModelConfig(n_folds=3)
    baseline = fit_p_pre_zero_count_model(counts, cell_ids, config)
    permutation = np.array([6, 4, 1, 7, 5, 0, 3, 2], dtype=np.int64)
    inverse = np.argsort(permutation)
    permuted = fit_p_pre_zero_count_model(
        counts[permutation],
        tuple(cell_ids[index] for index in permutation),
        config,
    )

    np.testing.assert_array_equal(baseline.fold_ids, permuted.fold_ids[inverse])
    np.testing.assert_array_equal(baseline.p_pre_zero, permuted.p_pre_zero[inverse])
    np.testing.assert_array_equal(baseline.mu, permuted.mu[inverse])
    np.testing.assert_array_equal(baseline.alpha, permuted.alpha[inverse])
    np.testing.assert_array_equal(baseline.pi, permuted.pi[inverse])
    for baseline_fold, permuted_fold in zip(
        baseline.fold_models,
        permuted.fold_models,
        strict=True,
    ):
        assert (
            baseline_fold.training_input_sha256 == permuted_fold.training_input_sha256
        )
        np.testing.assert_array_equal(
            baseline_fold.gene_means,
            permuted_fold.gene_means,
        )
        np.testing.assert_array_equal(
            baseline_fold.gene_dispersion,
            permuted_fold.gene_dispersion,
        )
        assert baseline_fold.link_intercept == permuted_fold.link_intercept
        assert baseline_fold.link_slope == permuted_fold.link_slope


def test_duplicate_count_rows_use_unique_external_ids_without_index_ties():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = np.array(
        [
            [3, 0, 1],
            [3, 0, 1],
            [0, 4, 1],
            [0, 4, 1],
            [1, 2, 0],
            [1, 2, 0],
        ],
        dtype=np.int64,
    )
    cell_ids = _cell_ids(len(counts))
    config = PreZeroCountModelConfig(n_folds=3)

    first = fit_p_pre_zero_count_model(counts, cell_ids, config)
    second = fit_p_pre_zero_count_model(counts, cell_ids, config)

    np.testing.assert_array_equal(first.fold_ids, second.fold_ids)
    np.testing.assert_array_equal(first.p_pre_zero, second.p_pre_zero)
    assert first.manifest["cross_fitting"]["assignment"] == (
        "balanced_sha256_external_cell_id_order_round_robin"
    )
    assert "duplicate_row_tie_breaker" not in first.manifest["cross_fitting"]


def test_fixed_cell_ids_make_fold_membership_independent_of_count_content():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    counts = _counts()
    cell_ids = _cell_ids()
    config = PreZeroCountModelConfig(n_folds=3)
    baseline = fit_p_pre_zero_count_model(counts, cell_ids, config)
    changed_counts = counts.copy()
    changed_counts[[0, 1, 4]] = np.array(
        [[0, 0, 0, 0], [100, 50, 25, 12], [9, 0, 0, 0]],
        dtype=np.int64,
    )
    changed = fit_p_pre_zero_count_model(changed_counts, cell_ids, config)

    np.testing.assert_array_equal(baseline.fold_ids, changed.fold_ids)
    assert tuple(record.held_out_indices for record in baseline.fold_models) == tuple(
        record.held_out_indices for record in changed.fold_models
    )


def test_equivalent_dense_and_sparse_inputs_have_identical_results_and_digests():
    from maskimpute import fit_p_pre_zero_count_model

    counts = _counts()
    cell_ids = _cell_ids(len(counts))
    dense = fit_p_pre_zero_count_model(counts.astype(np.float64), cell_ids)

    for constructor in _SPARSE_CONSTRUCTORS:
        encoded = fit_p_pre_zero_count_model(constructor(counts), cell_ids)
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
    score = fit_p_pre_zero_count_model(counts, _cell_ids(len(counts)))
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

    score = fit_p_pre_zero_count_model(_counts(), _cell_ids())
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
    cell_ids = _cell_ids(len(counts))
    score = fit_p_pre_zero_count_model(counts, cell_ids)

    np.testing.assert_array_equal(
        score.score_for_counts(counts.copy(), cell_ids), score.p_pre_zero
    )
    np.testing.assert_array_equal(
        score.score_for_counts(sparse.csr_matrix(counts), cell_ids), score.p_pre_zero
    )
    changed = counts.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="does not match"):
        score.score_for_counts(changed, cell_ids)


def test_score_rejects_subclass_and_detects_internal_digest_tampering():
    from maskimpute import PreZeroCountModelScore, fit_p_pre_zero_count_model

    score = fit_p_pre_zero_count_model(_counts(), _cell_ids())

    class ScoreSubclass(PreZeroCountModelScore):
        pass

    forged = object.__new__(ScoreSubclass)
    for slot in PreZeroCountModelScore.__slots__:
        object.__setattr__(forged, slot, getattr(score, slot))
    with pytest.raises(TypeError, match="exact PreZeroCountModelScore"):
        forged.score_for_counts(_counts(), _cell_ids())

    object.__setattr__(score, "_score_sha256", "0" * 64)
    with pytest.raises(ValueError, match="integrity"):
        score.score_for_counts(_counts(), _cell_ids())
    with pytest.raises(ValueError, match="integrity"):
        _ = score.p_pre_zero

    malformed = fit_p_pre_zero_count_model(_counts(), _cell_ids())
    object.__setattr__(malformed, "_config_bytes", None)
    with pytest.raises(ValueError, match="integrity"):
        _ = malformed.manifest


def test_rehashed_forged_score_fails_derivation_verification():
    from maskimpute import fit_p_pre_zero_count_model
    from maskimpute.count_model import (
        _canonical_json_bytes,
        _sha256_bytes,
        _snapshot_array,
    )

    counts = _counts()
    cell_ids = _cell_ids(len(counts))
    score = fit_p_pre_zero_count_model(counts, cell_ids)
    forged_probability = score.p_pre_zero.copy()
    forged_probability[counts == 0] = 0.125
    object.__setattr__(
        score,
        "_p_pre_zero",
        _snapshot_array(forged_probability, "<f8"),
    )
    unsigned = score._unsigned_manifest()
    score_sha256 = _sha256_bytes(_canonical_json_bytes(unsigned))
    manifest = dict(unsigned)
    manifest["score_sha256"] = score_sha256
    object.__setattr__(score, "_score_sha256", score_sha256)
    object.__setattr__(score, "_manifest_bytes", _canonical_json_bytes(manifest))

    assert score.manifest["score_sha256"] == score_sha256
    with pytest.raises(ValueError, match="derivation"):
        score.score_for_counts(counts, cell_ids)


def test_fit_revalidates_config_after_low_level_mutation():
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    config = PreZeroCountModelConfig()
    object.__setattr__(config, "n_folds", 1)

    with pytest.raises((TypeError, ValueError), match="config|n_folds"):
        fit_p_pre_zero_count_model(_counts(), _cell_ids(), config)


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
        _cell_ids(len(counts)),
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
        _cell_ids(len(counts)),
        PreZeroCountModelConfig(n_folds=3),
    )
    second = fit_p_pre_zero_count_model(
        counts,
        _cell_ids(len(counts)),
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
        _cell_ids(len(counts)),
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
        _cell_ids(len(counts)),
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
    expected = fit_p_pre_zero_count_model(_counts(), _cell_ids())

    actual = fit_p_pre_zero_count_model(protocol, _cell_ids())

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
        fit_p_pre_zero_count_model(invalid, _cell_ids(2))


def test_sparse_subclass_is_rejected_without_invoking_conversion_hook():
    from maskimpute import fit_p_pre_zero_count_model

    class HostileSparse(sparse.csr_matrix):
        calls = 0

        def tocoo(self, copy=False):
            type(self).calls += 1
            return super().tocoo(copy=copy)

    counts = HostileSparse(_counts())

    with pytest.raises(TypeError, match="exact supported SciPy sparse type"):
        fit_p_pre_zero_count_model(counts, _cell_ids())
    assert HostileSparse.calls == 0


def test_exact_coo_copy_shadow_cannot_swap_the_validated_count_snapshot():
    from maskimpute import fit_p_pre_zero_count_model

    stored = _counts()
    swapped = stored.copy()
    swapped[0, 0] += 9
    counts = sparse.coo_matrix(stored)
    calls = 0

    def hostile_copy(self):
        nonlocal calls
        calls += 1
        return sparse.coo_matrix(swapped)

    counts.copy = MethodType(hostile_copy, counts)

    with pytest.raises(TypeError, match="callable sparse instance shadow"):
        fit_p_pre_zero_count_model(counts, _cell_ids())
    assert calls == 0


def test_exact_dok_stateful_dict_storage_is_rejected_before_iteration():
    from maskimpute import fit_p_pre_zero_count_model

    stored = sparse.dok_matrix(_counts())
    swapped_counts = _counts()
    swapped_counts[0, 0] += 9
    alternate = dict(sparse.dok_matrix(swapped_counts)._dict)
    calls = {"items": 0, "keys": 0, "values": 0}

    class StatefulDict(dict):
        def items(self):
            calls["items"] += 1
            return super().items()

        def keys(self):
            calls["keys"] += 1
            return alternate.keys()

        def values(self):
            calls["values"] += 1
            return alternate.values()

    stored._dict = StatefulDict(stored._dict)

    with pytest.raises(TypeError, match="trusted internal sparse storage"):
        fit_p_pre_zero_count_model(stored, _cell_ids())
    assert calls == {"items": 0, "keys": 0, "values": 0}


@pytest.mark.parametrize(
    ("constructor", "storage_name"),
    [
        (sparse.coo_matrix, "data"),
        (sparse.csr_matrix, "indices"),
        (sparse.csc_matrix, "indptr"),
        (sparse.bsr_matrix, "data"),
        (sparse.dia_matrix, "offsets"),
    ],
)
def test_sparse_internal_array_subclasses_are_rejected_before_conversion(
    constructor,
    storage_name,
):
    from maskimpute import fit_p_pre_zero_count_model

    class ArraySubclass(np.ndarray):
        pass

    counts = constructor(_counts())
    setattr(counts, storage_name, getattr(counts, storage_name).view(ArraySubclass))

    with pytest.raises(TypeError, match="trusted internal sparse storage"):
        fit_p_pre_zero_count_model(counts, _cell_ids())


@pytest.mark.parametrize("nested", [False, True])
def test_lil_internal_storage_requires_exact_object_arrays_and_lists(nested):
    from maskimpute import fit_p_pre_zero_count_model

    class ArraySubclass(np.ndarray):
        pass

    class ListSubclass(list):
        pass

    counts = sparse.lil_matrix(_counts())
    if nested:
        counts.rows[0] = ListSubclass(counts.rows[0])
    else:
        counts.data = counts.data.view(ArraySubclass)

    with pytest.raises(TypeError, match="trusted internal sparse storage"):
        fit_p_pre_zero_count_model(counts, _cell_ids())


@pytest.mark.parametrize(
    ("setup", "mutation"),
    [
        (
            "x = sparse.coo_matrix(np.array([[1, 0], [0, 2]]))",
            "x.coords = (np.array([0, 2]), x.coords[1])",
        ),
        (
            "x = sparse.csr_matrix(np.array([[1, 0], [0, 2]]))",
            "x.indptr[-1] = 1_000_000",
        ),
        (
            "x = sparse.csr_matrix(np.array([[1, 0], [0, 2]]))",
            "x.indptr[0] = 1",
        ),
        (
            "x = sparse.csc_matrix(np.array([[1, 0], [0, 2]]))",
            "x.indices[0] = x.shape[0]",
        ),
        (
            "x = sparse.bsr_matrix(np.array([[1, 0], [0, 2]]))",
            "x.indices[0] = x.shape[1] // x.blocksize[1]",
        ),
        (
            "x = sparse.dia_matrix(np.array([[1, 0], [0, 2]]))",
            "x.offsets[0] = x.shape[1]",
        ),
        (
            "x = sparse.dia_matrix((np.array([[1, 0], [0, 1]]), "
            "np.array([0, 1])), shape=(2, 2))",
            "x.offsets[1] = 0",
        ),
        (
            "x = sparse.dok_matrix(np.array([[1, 0], [0, 2]]))",
            "x._dict[(x.shape[0], 0)] = np.int64(3)",
        ),
        (
            "x = sparse.lil_matrix(np.array([[1, 2], [0, 3]]))",
            "x.rows[0] = list(reversed(x.rows[0]))",
        ),
        (
            "x = sparse.lil_matrix(np.array([[1, 2], [0, 3]]))",
            "x.rows[0] = [0, 0]",
        ),
    ],
)
def test_malformed_exact_sparse_structures_fail_before_scipy_conversion(
    setup,
    mutation,
):
    script = f"""\
import numpy as np
from scipy import sparse
from maskimpute import fit_p_pre_zero_count_model
{setup}
{mutation}
try:
    fit_p_pre_zero_count_model(x, ("external-cell-0", "external-cell-1"))
except ValueError as error:
    assert "invalid sparse structure" in str(error), error
else:
    raise AssertionError("malformed sparse structure was accepted")
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
    ("constructor", "method_name"),
    [
        (sparse.coo_matrix, "tocoo"),
        (sparse.coo_matrix, "toarray"),
        (sparse.csr_matrix, "_swap"),
        (sparse.csc_matrix, "_swap"),
        (sparse.bsr_matrix, "_get_index_dtype"),
        (sparse.dia_matrix, "_get_index_dtype"),
        (sparse.dok_matrix, "values"),
        (sparse.lil_matrix, "tocsr"),
    ],
)
def test_format_specific_callable_sparse_shadows_are_rejected_before_use(
    constructor,
    method_name,
):
    from maskimpute import fit_p_pre_zero_count_model

    counts = constructor(_counts())
    calls = 0

    def hostile_method(self, *args, **kwargs):
        nonlocal calls
        del self, args, kwargs
        calls += 1
        raise AssertionError("hostile sparse method must never execute")

    setattr(counts, method_name, MethodType(hostile_method, counts))

    with pytest.raises(TypeError, match="callable sparse instance shadow"):
        fit_p_pre_zero_count_model(counts, _cell_ids())
    assert calls == 0


def test_exact_sparse_instance_conversion_hook_is_rejected_without_execution():
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

    with pytest.raises(TypeError, match="callable sparse instance shadow"):
        fit_p_pre_zero_count_model(counts, _cell_ids())
    assert calls == 0


def test_nested_masked_sparse_storage_is_rejected_before_conversion():
    from maskimpute import fit_p_pre_zero_count_model

    counts = sparse.lil_matrix(np.array([[1, 0], [0, 2]], dtype=np.int64))
    counts.data[0][0] = np.ma.array(1, mask=True)

    with pytest.raises(TypeError, match="masked"):
        fit_p_pre_zero_count_model(counts, _cell_ids(2))


@pytest.mark.parametrize(
    "constructor_name",
    ["dok_matrix", "lil_matrix", "dok_array", "lil_array"],
)
@pytest.mark.parametrize(
    ("declared_dtype", "stored_scalar"),
    [(np.int64, 1.5), (np.float32, 0.1)],
)
def test_sparse_declared_dtype_rejects_nonlossless_internal_scalar(
    constructor_name,
    declared_dtype,
    stored_scalar,
):
    from maskimpute import fit_p_pre_zero_count_model

    constructor = getattr(sparse, constructor_name, None)
    if constructor is None:
        pytest.skip(f"SciPy does not provide {constructor_name}")
    counts = constructor(np.array([[1, 0], [0, 2]], dtype=declared_dtype))
    if constructor_name.startswith("dok"):
        counts._dict[(0, 1)] = stored_scalar
    else:
        counts.rows[0].append(1)
        counts.data[0].append(stored_scalar)

    with pytest.raises(ValueError, match="losslessly compatible.*dtype"):
        fit_p_pre_zero_count_model(counts, _cell_ids(2))


@pytest.mark.parametrize(
    "constructor_name",
    ["dok_matrix", "lil_matrix", "dok_array", "lil_array"],
)
@pytest.mark.parametrize(
    ("declared_dtype", "stored_scalar"),
    [(np.float32, 2**1000), (np.float64, 2**10000)],
)
def test_float_sparse_storage_rejects_huge_python_int_without_warning(
    constructor_name,
    declared_dtype,
    stored_scalar,
):
    from maskimpute import fit_p_pre_zero_count_model

    constructor = getattr(sparse, constructor_name, None)
    if constructor is None:
        pytest.skip(f"SciPy does not provide {constructor_name}")
    counts = constructor(np.array([[1, 0], [0, 2]], dtype=declared_dtype))
    if constructor_name.startswith("dok"):
        counts._dict[(0, 1)] = stored_scalar
    else:
        counts.rows[0].append(1)
        counts.data[0].append(stored_scalar)

    with pytest.raises(ValueError, match="invalid sparse structure.*declared dtype"):
        fit_p_pre_zero_count_model(counts, _cell_ids(2))
