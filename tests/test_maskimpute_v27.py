from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import sparse


torch = pytest.importorskip("torch")


class _MaskedArrayProtocol:
    def __init__(self, values, mask):
        self._values = values
        self._mask = mask

    def __array__(self, dtype=None, copy=None):
        return np.ma.array(
            self._values,
            mask=self._mask,
            dtype=dtype,
            copy=False if copy is None else copy,
        )


class _ChangingArrayProtocol:
    def __init__(self, first, second):
        self._arrays = (np.asarray(first), np.asarray(second))
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        index = min(self.calls, 1)
        self.calls += 1
        return np.array(
            self._arrays[index],
            dtype=dtype,
            copy=True if copy is None else copy,
        )


def test_v27_configuration_prespecifies_validation_and_early_stopping():
    from maskimpute import MaskImputeConfig

    config = MaskImputeConfig()

    assert config.validation_fraction == pytest.approx(0.10)
    assert config.log_count_bin_edges == pytest.approx(
        (np.log1p(2.0), np.log1p(8.0), np.log1p(32.0))
    )
    assert config.early_stopping_min_delta == pytest.approx(0.0)


def test_v27_configuration_bounds_seed_to_publication_int64_domain():
    from maskimpute import MaskImputeConfig

    assert MaskImputeConfig(seed=2**63 - 1).seed == 2**63 - 1
    with pytest.raises(ValueError, match="seed"):
        MaskImputeConfig(seed=2**63)


def test_explicit_mask_token_makes_unavailable_payload_irrelevant():
    from maskimpute.model import ExplicitMaskAutoencoder

    model = ExplicitMaskAutoencoder(n_genes=3, hidden_dims=(5,), latent_dim=2)
    with torch.no_grad():
        model.mask_token.copy_(torch.tensor([0.25, 0.5, 0.75]))

    availability = torch.tensor([[True, False, True], [False, True, False]])
    first = torch.tensor([[1.0, 7.0, 3.0], [4.0, 5.0, 6.0]])
    second = torch.tensor([[1.0, -90.0, 3.0], [80.0, 5.0, -70.0]])

    encoded_first = model.prepare_encoder_input(first, availability)
    encoded_second = model.prepare_encoder_input(second, availability)

    torch.testing.assert_close(encoded_first, encoded_second)
    torch.testing.assert_close(
        encoded_first[:, :3],
        torch.tensor([[1.0, 0.5, 3.0], [0.25, 5.0, 0.75]]),
    )
    torch.testing.assert_close(encoded_first[:, 3:], availability.to(torch.float32))


def test_decoder_outputs_finite_nonnegative_normalized_values():
    from maskimpute.model import ExplicitMaskAutoencoder

    model = ExplicitMaskAutoencoder(n_genes=3, hidden_dims=(5,), latent_dim=2)
    values = torch.tensor([[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]])
    availability = values > 0

    reconstruction, latent = model(values, availability)

    assert reconstruction.shape == values.shape
    assert latent.shape == (2, 2)
    assert torch.isfinite(reconstruction).all()
    assert torch.all(reconstruction >= 0)


def test_observed_library_normalization_has_an_explicit_inverse_contract():
    from maskimpute.train import (
        invert_observed_normalization,
        normalize_observed_counts,
    )

    counts = np.array([[0, 0, 0], [1, 3, 6], [5, 0, 5]], dtype=np.int64)

    normalized, library_sizes = normalize_observed_counts(counts, target=1_000.0)
    restored = invert_observed_normalization(
        normalized,
        library_sizes,
        target=1_000.0,
    )

    np.testing.assert_array_equal(library_sizes, [0.0, 10.0, 10.0])
    np.testing.assert_allclose(restored, counts, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(normalized[0], np.zeros(3))
    np.testing.assert_array_equal(restored[0], np.zeros(3))


def test_corrupted_encoder_normalization_cannot_see_unavailable_target_magnitude():
    from maskimpute.model import ExplicitMaskAutoencoder
    from maskimpute.train import normalize_available_encoder_input

    availability = np.array([[True, False, True, False]], dtype=np.bool_)
    first = np.array([[2, 5, 3, 11]], dtype=np.int64)
    changed_hidden_targets = np.array([[2, 500, 3, 900]], dtype=np.int64)

    first_input, first_library = normalize_available_encoder_input(
        first, availability, target=1_000.0
    )
    changed_input, changed_library = normalize_available_encoder_input(
        changed_hidden_targets, availability, target=1_000.0
    )

    np.testing.assert_array_equal(first_library, [5.0])
    np.testing.assert_array_equal(changed_library, [5.0])
    np.testing.assert_array_equal(first_input, changed_input)
    model = ExplicitMaskAutoencoder(n_genes=4, hidden_dims=(5,), latent_dim=2)
    encoded_first = model.prepare_encoder_input(
        torch.tensor(first_input, dtype=torch.float32),
        torch.tensor(availability),
    )
    encoded_changed = model.prepare_encoder_input(
        torch.tensor(changed_input, dtype=torch.float32),
        torch.tensor(availability),
    )
    torch.testing.assert_close(encoded_first, encoded_changed)


def test_encoder_availability_is_single_snapshot_and_rejects_hidden_masks():
    from maskimpute.train import normalize_available_encoder_input

    counts = np.array([[1, 2], [3, 4]], dtype=np.int64)
    changing = _ChangingArrayProtocol(
        np.array([[True, False], [False, True]]),
        np.array([[False, True], [True, False]]),
    )
    normalized, _ = normalize_available_encoder_input(counts, changing, target=1_000.0)
    assert changing.calls == 1
    np.testing.assert_array_equal(normalized > 0, [[True, False], [False, True]])

    masked_protocol = _MaskedArrayProtocol(
        [[True, False], [False, True]],
        [[False, True], [False, False]],
    )
    with pytest.raises(TypeError, match="masked"):
        normalize_available_encoder_input(counts, masked_protocol, target=1_000.0)

    nested = np.empty((2, 2), dtype=object)
    nested[:] = False
    nested[0, 0] = np.ma.array(True, mask=True)
    with pytest.raises(TypeError, match="masked"):
        normalize_available_encoder_input(counts, nested, target=1_000.0)


@pytest.mark.parametrize(
    "invalid_counts",
    [
        np.array([[1.25, 0], [0, 2.0]]),
        np.ma.array([[1, 0], [0, 2]], mask=False),
    ],
)
def test_normalization_enforces_the_raw_integral_unmasked_count_contract(
    invalid_counts,
):
    from maskimpute.train import normalize_observed_counts

    with pytest.raises((TypeError, ValueError)):
        normalize_observed_counts(invalid_counts, target=1_000.0)


def test_validation_holdout_is_deterministic_and_stratified_over_log_counts():
    from maskimpute.train import make_stratified_validation_mask

    counts = np.array(
        [
            [1, 1, 2, 2],
            [4, 4, 8, 8],
        ],
        dtype=np.int64,
    )
    edges = (np.log1p(1.0), np.log1p(2.0), np.log1p(4.0))

    first = make_stratified_validation_mask(
        counts,
        fraction=0.5,
        log_count_bin_edges=edges,
        rng=np.random.default_rng(91),
    )
    second = make_stratified_validation_mask(
        counts,
        fraction=0.5,
        log_count_bin_edges=edges,
        rng=np.random.default_rng(91),
    )

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.bool_
    for value in (1, 2, 4, 8):
        positions = counts == value
        assert np.count_nonzero(first & positions) == 1
        assert np.count_nonzero(~first & positions) == 1


def test_epoch_mask_uses_only_training_positives_outside_fixed_validation():
    from maskimpute.train import make_epoch_training_mask

    counts = np.array([[1, 0, 2, 3], [4, 5, 0, 6]], dtype=np.int64)
    validation = np.array([[True, False, False, False], [False, False, False, True]])

    epoch_mask = make_epoch_training_mask(
        counts,
        validation_mask=validation,
        fraction=0.5,
        log_count_bin_edges=(np.log1p(2.0), np.log1p(4.0)),
        rng=np.random.default_rng(7),
    )

    assert np.any(epoch_mask)
    assert not np.any(epoch_mask & validation)
    assert not np.any(epoch_mask & (counts == 0))
    assert np.all((~epoch_mask) | ((counts > 0) & ~validation))


def test_epoch_mask_is_stratified_across_populated_log_count_bins():
    from maskimpute.train import make_epoch_training_mask

    counts = np.array([[1, 1, 2, 2], [4, 4, 8, 8]], dtype=np.int64)
    epoch_mask = make_epoch_training_mask(
        counts,
        validation_mask=np.zeros_like(counts, dtype=np.bool_),
        fraction=0.5,
        log_count_bin_edges=(np.log1p(1.5), np.log1p(3.0), np.log1p(6.0)),
        rng=np.random.default_rng(22),
    )

    for value in (1, 2, 4, 8):
        assert np.count_nonzero(epoch_mask & (counts == value)) == 1


def test_mask_helpers_enforce_the_raw_integral_count_contract():
    from maskimpute.train import (
        make_epoch_training_mask,
        make_stratified_validation_mask,
    )

    fractional = np.array([[1.25, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="integral"):
        make_stratified_validation_mask(
            fractional,
            fraction=0.25,
            log_count_bin_edges=(np.log1p(2.0),),
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="integral"):
        make_epoch_training_mask(
            fractional,
            validation_mask=np.zeros_like(fractional, dtype=np.bool_),
            fraction=0.25,
            log_count_bin_edges=(np.log1p(2.0),),
            rng=np.random.default_rng(1),
        )


def test_natural_zero_penalty_is_softly_weighted_by_external_probability():
    from maskimpute.train import natural_zero_preservation_loss

    predictions = torch.tensor([[2.0, 3.0, 100.0]])
    natural_zeros = torch.tensor([[True, True, False]])
    probability = torch.tensor([[0.25, 1.0, 1.0]])

    loss = natural_zero_preservation_loss(
        predictions,
        natural_zeros,
        probability,
    )

    assert loss.item() == pytest.approx((0.25 * 2.0**2 + 1.0 * 3.0**2) / 2)


def test_training_signature_cannot_receive_evaluator_truth_or_labels():
    from maskimpute.train import train_v27

    parameters = set(inspect.signature(train_v27).parameters)
    forbidden = {
        "truth",
        "labels",
        "group",
        "markers",
        "pseudotime",
        "adata",
        "pre_capture_counts",
    }

    assert not parameters & forbidden


def test_training_uses_fixed_validation_holdout_and_validation_only_early_stopping():
    from maskimpute import MaskImputeConfig
    from maskimpute.train import train_v27

    counts = np.array(
        [
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [5, 0, 6, 0],
            [0, 7, 0, 8],
        ],
        dtype=np.int64,
    )
    probability = np.where(counts == 0, 0.8, 0.0)
    config = MaskImputeConfig(
        hidden_dims=(6,),
        latent_dim=2,
        batch_size=4,
        max_epochs=9,
        patience=2,
        validation_fraction=0.25,
        artificial_mask_fraction=0.5,
        early_stopping_min_delta=1_000_000.0,
        seed=17,
    )

    outcome = train_v27(counts, probability, config, device="cpu")

    assert outcome.best_epoch == 1
    assert outcome.stopped_epoch == 3
    assert len(outcome.validation_loss_history) == 3
    assert len(outcome.training_loss_history) == 3
    assert np.count_nonzero(outcome.validation_mask) == 2
    assert np.all((~outcome.validation_mask) | (counts > 0))
    assert len(set(outcome.validation_mask_hashes)) == 1
    assert len(outcome.validation_mask_hashes) == outcome.stopped_epoch
    assert all(np.isfinite(outcome.validation_loss_history))


@pytest.mark.parametrize(
    "counts",
    [
        np.zeros((2, 3), dtype=np.int64),
        np.array([[0, 1, 0]], dtype=np.int64),
    ],
)
def test_validation_mask_rejects_no_or_too_few_observed_positives(counts):
    from maskimpute.train import make_stratified_validation_mask

    with pytest.raises(ValueError, match="at least two observed positive"):
        make_stratified_validation_mask(
            counts,
            fraction=0.2,
            log_count_bin_edges=(np.log1p(1.0),),
            rng=np.random.default_rng(1),
        )


def _tiny_counts() -> np.ndarray:
    return np.array(
        [
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [5, 0, 6, 0],
            [0, 7, 0, 8],
        ],
        dtype=np.int64,
    )


def _tiny_probability(counts: np.ndarray) -> np.ndarray:
    return np.where(counts == 0, 0.75, 0.0)


def _tiny_config(**overrides):
    from maskimpute import MaskImputeConfig

    values = {
        "hidden_dims": (6,),
        "latent_dim": 2,
        "batch_size": 4,
        "max_epochs": 2,
        "patience": 2,
        "validation_fraction": 0.25,
        "artificial_mask_fraction": 0.5,
        "seed": 23,
    }
    values.update(overrides)
    return MaskImputeConfig(**values)


def test_public_imputation_signature_has_no_evaluator_side_channels():
    from maskimpute import impute_counts

    assert tuple(inspect.signature(impute_counts).parameters) == (
        "observed_counts",
        "p_pre_zero",
        "config",
        "device",
    )


def test_inference_marks_exactly_natural_observed_zeros_unavailable():
    from maskimpute.impute import inference_availability_mask

    counts = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int64)

    availability = inference_availability_mask(counts)

    np.testing.assert_array_equal(availability, counts > 0)


def test_power_complement_gate_is_monotone_in_pre_zero_probability():
    from maskimpute.impute import apply_zero_gate

    candidates = np.full((1, 5), 4.0)
    probability = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]])

    gated = apply_zero_gate(candidates, probability, gamma=2.0)

    np.testing.assert_allclose(gated, [[4.0, 2.25, 1.0, 0.25, 0.0]])
    assert np.all(np.diff(gated[0]) <= 0)


@pytest.mark.parametrize("field", ["candidates", "p_pre_zero"])
@pytest.mark.parametrize(
    "invalid",
    [
        np.ma.array([[1.0, 0.5]], mask=[[False, True]]),
        np.array([[1.0, 0.5]], dtype=object),
        np.array(
            [[1.0, 0.5]],
            dtype=np.dtype(np.float64, metadata={"semantic": "mutable-alias"}),
        ),
        _MaskedArrayProtocol([[1.0, 0.5]], [[False, True]]),
    ],
)
def test_power_complement_gate_rejects_ambiguous_direct_inputs(field, invalid):
    from maskimpute.impute import apply_zero_gate

    arguments = {
        "candidates": np.array([[1.0, 0.5]]),
        "p_pre_zero": np.array([[0.0, 0.5]]),
        "gamma": 1.0,
    }
    arguments[field] = invalid

    with pytest.raises(TypeError):
        apply_zero_gate(**arguments)


def test_selective_imputation_preserves_positives_and_audits_training_contract():
    from maskimpute import impute_counts

    counts = _tiny_counts()
    probability = _tiny_probability(counts)

    result = impute_counts(counts, probability, config=_tiny_config(), device="cpu")

    np.testing.assert_array_equal(
        result.selective_counts[counts > 0], counts[counts > 0]
    )
    assert result.selective_counts.shape == counts.shape
    assert result.denoised_counts.shape == counts.shape
    assert result.latent.shape == (counts.shape[0], 2)
    assert np.all(np.isfinite(result.selective_counts))
    assert np.all(np.isfinite(result.denoised_counts))
    assert np.all(result.selective_counts >= 0)
    assert np.all(result.denoised_counts >= 0)
    np.testing.assert_array_equal(result.p_pre_zero, probability)

    diagnostics = result.diagnostics
    assert diagnostics["method_version"] == "v27"
    assert diagnostics["score_source"] == (
        "caller_supplied_count_model_p_pre_zero_unverified"
    )
    assert diagnostics["score_provenance_verified"] is False
    assert diagnostics["normalization"]["target"] == pytest.approx(10_000.0)
    assert diagnostics["normalization"]["zero_library_policy"] == "preserve_all_zero"
    assert diagnostics["masks"]["inference_unavailable"] == "observed_count_equals_zero"
    assert diagnostics["losses"]["validation_criterion"] == (
        "fixed_artificial_positive_mse"
    )
    assert diagnostics["early_stopping"]["best_epoch"] >= 1
    assert diagnostics["randomness"]["model_seed"] == 23
    assert diagnostics["device"] == "cpu"


def test_sparse_input_and_zero_library_cells_follow_the_same_count_scale_contract():
    from maskimpute import impute_counts

    counts = _tiny_counts()
    counts[0] = 0
    probability = _tiny_probability(counts)

    result = impute_counts(
        sparse.csr_matrix(counts),
        probability,
        config=_tiny_config(),
        device="cpu",
    )

    np.testing.assert_array_equal(result.selective_counts[0], np.zeros(counts.shape[1]))
    np.testing.assert_array_equal(result.denoised_counts[0], np.zeros(counts.shape[1]))
    np.testing.assert_array_equal(
        result.selective_counts[counts > 0], counts[counts > 0]
    )


def test_sparse_validators_accept_csr_dok_and_lil_matrix_and_array_storage():
    from maskimpute.train import validate_observed_counts, validate_p_pre_zero

    counts = _tiny_counts()
    probability = _tiny_probability(counts)
    constructors = [sparse.csr_matrix, sparse.dok_matrix, sparse.lil_matrix]
    for name in ("csr_array", "dok_array", "lil_array"):
        constructor = getattr(sparse, name, None)
        if constructor is not None:
            constructors.append(constructor)

    for constructor in constructors:
        validated_counts = validate_observed_counts(constructor(counts))
        validated_probability = validate_p_pre_zero(
            constructor(probability),
            validated_counts,
        )
        np.testing.assert_array_equal(validated_counts, counts)
        np.testing.assert_array_equal(validated_probability, probability)


def test_sparse_lil_nested_masked_scalar_is_rejected_before_coercion():
    from maskimpute.train import validate_observed_counts

    counts = sparse.lil_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
    counts.data[0][0] = np.ma.array(1.0, mask=True)

    with pytest.raises(TypeError, match="masked"):
        validate_observed_counts(counts)


def test_custom_sparse_conversion_hook_cannot_swap_unvalidated_storage():
    from maskimpute.train import validate_observed_counts

    class MaskedCoordinateSwap(sparse.csr_matrix):
        conversion_called = False

        def tocoo(self, copy=False):
            type(self).conversion_called = True
            coordinates = super().tocoo(copy=copy)
            coordinates.data = np.ma.array(
                coordinates.data,
                mask=np.ones(coordinates.data.shape, dtype=np.bool_),
            )
            return coordinates

    counts = MaskedCoordinateSwap(np.array([[1, 0], [0, 2]], dtype=np.int64))

    with pytest.raises(TypeError, match="exact supported SciPy sparse"):
        validate_observed_counts(counts)
    assert MaskedCoordinateSwap.conversion_called is False


def test_count_and_score_dtype_metadata_are_rejected():
    from maskimpute.train import validate_observed_counts, validate_p_pre_zero

    count_dtype = np.dtype(np.int64, metadata={"semantic": "mutable-count-alias"})
    score_dtype = np.dtype(np.float64, metadata={"semantic": "mutable-score-alias"})
    metadata_counts = _tiny_counts().astype(count_dtype)
    plain_counts = _tiny_counts().astype(np.int64)
    metadata_score = _tiny_probability(plain_counts).astype(score_dtype)

    with pytest.raises(TypeError, match="metadata"):
        validate_observed_counts(metadata_counts)
    with pytest.raises(TypeError, match="metadata"):
        validate_p_pre_zero(metadata_score, plain_counts.astype(np.float64))


def test_stateful_array_protocol_is_materialized_exactly_once_before_validation():
    from maskimpute.train import validate_observed_counts, validate_p_pre_zero

    counts_protocol = _ChangingArrayProtocol(
        _tiny_counts(),
        np.full_like(_tiny_counts(), -1),
    )
    counts = validate_observed_counts(counts_protocol)
    np.testing.assert_array_equal(counts, _tiny_counts())
    assert counts_protocol.calls == 1

    probability_protocol = _ChangingArrayProtocol(
        _tiny_probability(_tiny_counts()),
        np.full_like(_tiny_probability(_tiny_counts()), np.nan),
    )
    probability = validate_p_pre_zero(probability_protocol, counts)
    np.testing.assert_array_equal(probability, _tiny_probability(_tiny_counts()))
    assert probability_protocol.calls == 1


def test_cpu_reruns_are_exactly_reproducible_for_the_same_seed():
    from maskimpute import impute_counts

    counts = _tiny_counts()
    probability = _tiny_probability(counts)
    config = _tiny_config(max_epochs=3, patience=3)

    first = impute_counts(counts, probability, config=config, device="cpu")
    second = impute_counts(counts, probability, config=config, device="cpu")

    np.testing.assert_array_equal(first.selective_counts, second.selective_counts)
    np.testing.assert_array_equal(first.denoised_counts, second.denoised_counts)
    np.testing.assert_array_equal(first.latent, second.latent)
    assert first.diagnostics == second.diagnostics


def test_training_scopes_deterministic_algorithms_and_restores_caller_rng_state():
    from maskimpute import impute_counts

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_state = torch.random.get_rng_state().clone()
    try:
        torch.use_deterministic_algorithms(False)
        torch.manual_seed(919)
        caller_state = torch.random.get_rng_state().clone()
        result = impute_counts(
            _tiny_counts(),
            _tiny_probability(_tiny_counts()),
            config=_tiny_config(max_epochs=1, patience=1),
            device="cpu",
        )

        assert not torch.are_deterministic_algorithms_enabled()
        torch.testing.assert_close(torch.random.get_rng_state(), caller_state)
        assert result.diagnostics["randomness"]["deterministic_algorithms"] is True
        assert result.diagnostics["randomness"]["caller_rng_state_restored"] is True
    finally:
        torch.random.set_rng_state(previous_state)
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )


def _nested_masked_matrix() -> np.ndarray:
    result = np.empty((2, 2), dtype=object)
    result[:] = 0
    result[0, 0] = np.ma.array(1, mask=True)
    result[1, 1] = 2
    return result


@pytest.mark.parametrize(
    "invalid_counts",
    [
        np.empty((0, 3), dtype=np.int64),
        np.empty((3, 0), dtype=np.int64),
        np.array([[1.5, 0], [0, 2.0]]),
        np.array([[1, -1], [0, 2]]),
        np.array([[1.0, np.inf], [0, 2.0]]),
        np.array([[True, False], [False, True]]),
        np.array([[1, 0], [0, 2]], dtype=object),
        np.ma.array([[1, 0], [0, 2]], mask=False),
        _nested_masked_matrix(),
        np.array([[2**53 + 1, 0], [0, 2]], dtype=np.uint64),
        sparse.coo_matrix(
            (
                np.array([1, 2, 3], dtype=np.int64),
                (np.array([0, 0, 1]), np.array([0, 0, 1])),
            ),
            shape=(2, 2),
        ),
    ],
)
def test_public_api_rejects_ambiguous_or_lossy_count_inputs(invalid_counts):
    from maskimpute import impute_counts

    with pytest.raises((TypeError, ValueError)):
        impute_counts(
            invalid_counts,
            np.zeros(np.shape(invalid_counts), dtype=np.float64),
            config=_tiny_config(max_epochs=1, patience=1),
            device="cpu",
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda score, counts: score[:, :-1],
        lambda score, counts: np.where(counts == 0, np.nan, 0.0),
        lambda score, counts: np.where(counts == 0, 1.1, 0.0),
        lambda score, counts: np.where(counts == 0, -0.1, 0.0),
        lambda score, counts: np.where(counts > 0, 0.1, score),
        lambda score, counts: np.ma.array(score, mask=False),
        lambda score, counts: _nested_masked_matrix(),
    ],
)
def test_public_api_rejects_invalid_or_positive_entry_scores(mutator):
    from maskimpute import impute_counts

    counts = _tiny_counts()
    score = mutator(_tiny_probability(counts), counts)

    with pytest.raises((TypeError, ValueError)):
        impute_counts(
            counts,
            score,
            config=_tiny_config(max_epochs=1, patience=1),
            device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_smoke_returns_finite_outputs_without_accuracy_assumptions(monkeypatch):
    from maskimpute import impute_counts

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    counts = _tiny_counts()
    result = impute_counts(
        counts,
        _tiny_probability(counts),
        config=_tiny_config(max_epochs=1, patience=1),
        device="cuda",
    )

    assert np.all(np.isfinite(result.selective_counts))
    assert np.all(np.isfinite(result.denoised_counts))
    assert np.all(np.isfinite(result.latent))
    assert result.diagnostics["randomness"]["cublas_workspace_config"] == ":4096:8"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fails_closed_without_deterministic_cublas_workspace(monkeypatch):
    from maskimpute import impute_counts

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        impute_counts(
            _tiny_counts(),
            _tiny_probability(_tiny_counts()),
            config=_tiny_config(max_epochs=1, patience=1),
            device="cuda",
        )
