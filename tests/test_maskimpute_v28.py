from __future__ import annotations

import hashlib
from pathlib import Path
import json

import numpy as np
import pytest
from scipy.stats import nbinom
import torch


def test_negative_binomial_nll_matches_scipy_mean_size_parameterization() -> None:
    from maskimpute.nb_model import negative_binomial_nll

    counts = torch.tensor(
        [[0.0, 1.0, 4.0], [2.0, 7.0, 12.0]], dtype=torch.float64
    )
    mean = torch.tensor(
        [[0.2, 1.5, 3.3], [1.1, 6.4, 10.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    inverse_dispersion = torch.tensor([0.7, 3.0, 12.0], dtype=torch.float64)
    mask = torch.tensor(
        [[True, False, True], [False, True, True]], dtype=torch.bool
    )

    observed = negative_binomial_nll(
        counts,
        mean,
        inverse_dispersion,
        mask=mask,
        reduction="sum",
    )
    size = inverse_dispersion.numpy()[None, :]
    probability = size / (size + mean.detach().numpy())
    expected_matrix = -nbinom.logpmf(
        counts.numpy(),
        size,
        probability,
    )

    np.testing.assert_allclose(
        observed.detach().numpy(),
        expected_matrix[mask.numpy()].sum(),
        rtol=1e-12,
        atol=1e-12,
    )
    observed.backward()
    assert mean.grad is not None
    assert torch.isfinite(mean.grad).all()


def test_gene_dispersion_is_robust_bounded_shrunk_and_deterministic() -> None:
    from maskimpute.nb_model import (
        NegativeBinomialDecoderConfig,
        estimate_shrunk_gene_dispersion,
    )

    counts = np.array(
        [
            [10, 1, 3, 86],
            [12, 0, 9, 79],
            [9, 6, 2, 83],
            [11, 0, 11, 78],
            [10, 8, 1, 81],
            [13, 0, 12, 75],
            [8, 10, 2, 80],
            [90, 0, 10, 0],
        ],
        dtype=np.float64,
    )
    libraries = counts.sum(axis=1)
    robust_config = NegativeBinomialDecoderConfig(
        dispersion_prior_strength=0.0,
        winsor_quantile=0.75,
        min_dispersion=1e-4,
        max_dispersion=100.0,
    )
    unbounded_influence_config = NegativeBinomialDecoderConfig(
        dispersion_prior_strength=0.0,
        winsor_quantile=1.0,
        min_dispersion=1e-4,
        max_dispersion=100.0,
    )

    first = estimate_shrunk_gene_dispersion(counts, libraries, robust_config)
    second = estimate_shrunk_gene_dispersion(counts, libraries, robust_config)
    unbounded = estimate_shrunk_gene_dispersion(
        counts,
        libraries,
        unbounded_influence_config,
    )
    strongly_shrunk = estimate_shrunk_gene_dispersion(
        counts,
        libraries,
        NegativeBinomialDecoderConfig(
            dispersion_prior_strength=1e9,
            winsor_quantile=0.75,
            min_dispersion=1e-4,
            max_dispersion=100.0,
        ),
    )

    np.testing.assert_array_equal(first.dispersion, second.dispersion)
    np.testing.assert_array_equal(
        first.inverse_dispersion,
        second.inverse_dispersion,
    )
    assert np.all(np.isfinite(first.dispersion))
    assert np.all((first.dispersion >= 1e-4) & (first.dispersion <= 100.0))
    assert first.dispersion[0] < unbounded.dispersion[0]
    assert np.max(
        np.abs(np.log(strongly_shrunk.dispersion / strongly_shrunk.global_dispersion))
    ) < 1e-6
    np.testing.assert_allclose(
        first.inverse_dispersion,
        1.0 / first.dispersion,
    )


def test_nb_decoder_uses_explicit_mask_and_library_size_offset() -> None:
    from maskimpute.nb_model import (
        NegativeBinomialMaskAutoencoder,
        apply_library_size_offset,
    )

    torch.manual_seed(11)
    model = NegativeBinomialMaskAutoencoder(
        n_genes=4,
        hidden_dims=(7, 5),
        latent_dim=3,
    ).double()
    availability = torch.tensor(
        [[True, False, True, False], [True, True, False, True]],
        dtype=torch.bool,
    )
    first_payload = torch.tensor(
        [[1.0, 20.0, 0.5, 90.0], [0.2, 1.4, 70.0, 2.0]],
        dtype=torch.float64,
    )
    second_payload = first_payload.clone()
    second_payload[~availability] = torch.tensor(
        [-9000.0, 8000.0, 7000.0], dtype=torch.float64
    )

    first_fraction, first_latent = model(first_payload, availability)
    second_fraction, second_latent = model(second_payload, availability)
    np.testing.assert_array_equal(
        first_fraction.detach().numpy(), second_fraction.detach().numpy()
    )
    np.testing.assert_array_equal(
        first_latent.detach().numpy(), second_latent.detach().numpy()
    )
    np.testing.assert_allclose(
        first_fraction.detach().sum(dim=1).numpy(),
        np.ones(2),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.all(first_fraction > 0)

    libraries = torch.tensor([120.0, 37.0], dtype=torch.float64)
    means = apply_library_size_offset(first_fraction, libraries)
    np.testing.assert_allclose(
        means.detach().sum(dim=1).numpy(),
        libraries.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("keyword", "invalid", "match"),
    [
        ("dispersion_prior_strength", -1.0, "prior"),
        ("winsor_quantile", 0.49, "winsor"),
        ("winsor_quantile", 1.01, "winsor"),
        ("min_dispersion", 0.0, "min_dispersion"),
        ("max_dispersion", np.inf, "max_dispersion"),
    ],
)
def test_nb_decoder_configuration_fails_closed(
    keyword: str,
    invalid: float,
    match: str,
) -> None:
    from maskimpute.nb_model import NegativeBinomialDecoderConfig

    with pytest.raises((TypeError, ValueError), match=match):
        NegativeBinomialDecoderConfig(**{keyword: invalid})


def _tiny_counts() -> np.ndarray:
    return np.array(
        [
            [8, 0, 1, 0],
            [0, 7, 2, 1],
            [5, 1, 0, 2],
            [2, 0, 8, 1],
            [0, 3, 4, 5],
            [6, 2, 0, 3],
        ],
        dtype=np.float64,
    )


def _tiny_config():
    from maskimpute import MaskImputeConfig

    return MaskImputeConfig(
        hidden_dims=(8, 5),
        latent_dim=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=3,
        max_epochs=4,
        patience=2,
        artificial_mask_fraction=0.25,
        validation_fraction=0.2,
        pre_zero_regularization=0.5,
        gate_gamma=2.0,
        seed=42,
    )


def _method_input(counts: np.ndarray):
    from maskimpute_benchmark.methods.base import MethodInput

    return MethodInput(
        source_dataset_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        obs_ids=tuple(f"cell-{index}" for index in range(counts.shape[0])),
        var_ids=tuple(f"gene-{index}" for index in range(counts.shape[1])),
        shape=counts.shape,
        obs_covariates=(),
        var_covariates=(),
        _count_bytes=np.asarray(counts, dtype="<f8", order="C").tobytes(order="C"),
        _normalization_bytes=b"{}",
    )


def _identity_calibration_artifact():
    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
        fit_development_calibration,
    )

    records = []
    for index, (draw, view) in enumerate(
        (
            ("draw-01", "moderate"),
            ("draw-01", "severe"),
            ("draw-02", "moderate"),
            ("draw-02", "severe"),
        ),
        start=1,
    ):
        dataset_sha = hashlib.sha256(f"{draw}:{view}:v28".encode()).hexdigest()
        records.append(
            CalibrationRecord(
                p_pre_zero=(0.1, 0.25, 0.7, 0.9),
                target=(0, 0, 1, 1),
                mechanism="symsim",
                biological_id=draw,
                manifest_sha256=f"{index:x}" * 64,
                truth_kind="exact_pre_capture",
                namespace="dev",
                data_role="development",
                technical_view=view,
                dataset_id=f"dataset-{dataset_sha[:24]}",
                dataset_sha256=dataset_sha,
                protocol_sha256=DEVELOPMENT_PROTOCOL_SHA256,
            )
        )
    artifact = fit_development_calibration(records)
    assert artifact.selected_algorithm == "identity"
    return artifact


def test_train_v28_is_deterministic_truth_free_and_restores_caller_rng() -> None:
    from maskimpute.nb_model import NegativeBinomialDecoderConfig
    from maskimpute.train import train_v28

    counts = _tiny_counts()
    probability = np.zeros_like(counts)
    probability[counts == 0] = 0.6
    decoder_config = NegativeBinomialDecoderConfig(
        dispersion_prior_strength=5.0,
        winsor_quantile=0.9,
    )
    caller_state = torch.random.get_rng_state().clone()

    first = train_v28(
        counts,
        probability,
        _tiny_config(),
        "cpu",
        decoder_config=decoder_config,
    )
    after_first = torch.random.get_rng_state().clone()
    second = train_v28(
        counts,
        probability,
        _tiny_config(),
        "cpu",
        decoder_config=decoder_config,
    )

    assert torch.equal(caller_state, after_first)
    assert torch.equal(caller_state, torch.random.get_rng_state())
    assert first.training.training_loss_history == second.training.training_loss_history
    assert (
        first.training.validation_loss_history
        == second.training.validation_loss_history
    )
    assert first.training.validation_mask_hashes == second.training.validation_mask_hashes
    assert (
        first.training.epoch_training_mask_hashes
        == second.training.epoch_training_mask_hashes
    )
    for name, value in first.training.model.state_dict().items():
        assert torch.equal(value, second.training.model.state_dict()[name])
    np.testing.assert_array_equal(
        first.dispersion.dispersion,
        second.dispersion.dispersion,
    )
    expected_effective = np.sum(
        ~first.training.validation_mask
        & (first.training.library_sizes[:, None] > 0),
        axis=0,
    )
    np.testing.assert_array_equal(
        first.dispersion.effective_observations,
        expected_effective,
    )
    assert first.training.deterministic_algorithms is True
    assert first.training.caller_rng_state_restored is True
    assert all(np.isfinite(first.training.training_loss_history))
    assert all(np.isfinite(first.training.validation_loss_history))

    with pytest.raises(TypeError):
        train_v28(
            counts,
            probability,
            _tiny_config(),
            "cpu",
            decoder_config=decoder_config,
            evaluator_truth=np.ones_like(counts),
        )


def test_v28_development_candidate_shares_score_gate_and_selective_output() -> None:
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model
    from maskimpute.nb_model import NegativeBinomialDecoderConfig
    from maskimpute_benchmark.methods.maskimpute import (
        run_v28_development_candidate,
    )
    from maskimpute_benchmark.methods.registry import load_method_registry

    counts = _tiny_counts()
    method_input = _method_input(counts)
    score_config = PreZeroCountModelConfig(n_folds=2, link_max_iter=25)
    calibration = _identity_calibration_artifact()
    execution = run_v28_development_candidate(
        load_method_registry(Path("study/methods.json")).by_id("maskimpute"),
        method_input,
        calibration_artifact=calibration,
        seed=42,
        config=_tiny_config(),
        count_model_config=score_config,
        decoder_config=NegativeBinomialDecoderConfig(
            dispersion_prior_strength=5.0,
            winsor_quantile=0.9,
        ),
        device="cpu",
        development_mechanism="symsim",
        development_biological_id="draw-01",
    )

    direct_score = fit_p_pre_zero_count_model(
        counts,
        method_input.obs_ids,
        score_config,
    ).score_for_counts(counts, method_input.obs_ids)
    expected_probability = np.zeros_like(direct_score)
    observed_zero = counts == 0
    expected_probability[observed_zero] = (
        calibration.transform_for_development_holdout(
            direct_score[observed_zero],
            mechanism="symsim",
            biological_id="draw-01",
        )
    )
    result = execution.ablation_result
    np.testing.assert_array_equal(result.p_pre_zero, expected_probability)
    np.testing.assert_array_equal(
        execution.snapshot.matrix[counts > 0],
        counts[counts > 0],
    )
    np.testing.assert_allclose(
        result.denoised_counts.sum(axis=1),
        counts.sum(axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    expected_output = result.denoised_counts * np.power(
        1.0 - expected_probability,
        _tiny_config().gate_gamma,
    )
    expected_output[counts > 0] = counts[counts > 0]
    np.testing.assert_allclose(result.selective_counts, expected_output)
    assert np.all(np.isfinite(result.denoised_counts))
    assert np.all(result.denoised_counts >= 0)
    diagnostics = result.diagnostics
    assert diagnostics["method_version"] == "v28-development-candidate-single-run"
    assert diagnostics["decoder"]["family"] == "negative_binomial"
    assert diagnostics["decoder"]["mean"] == (
        "observed_library_size_times_decoded_gene_fraction"
    )
    assert diagnostics["losses"]["primary"] == (
        "artificially_masked_observed_positive_negative_binomial_nll"
    )
    assert diagnostics["score"]["calibration_scope"] == (
        "leave_one_biological_draw_out"
    )
    assert diagnostics["primary_output_policy"] == "selective"


def _candidate_configuration(payload: dict[str, object], *, identifier: str):
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    return AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id=identifier,
        kind="candidate_search",
        payload=payload,
        requires_count_score=True,
        requires_calibration=True,
    )


def _v28_configuration_payload() -> dict[str, object]:
    from maskimpute.nb_model import NegativeBinomialDecoderConfig

    search = json.loads(Path("study/development_search.json").read_text())
    base = search["configurations"][2]["configuration"]
    return {
        **base,
        "method_version": "v28",
        "decoder": "negative_binomial",
        "decoder_hyperparameters": NegativeBinomialDecoderConfig().to_dict(),
    }


def test_candidate_decoder_dispatch_accepts_only_exact_v27_and_v28_pairs() -> None:
    from maskimpute.nb_model import NegativeBinomialDecoderConfig
    from maskimpute_benchmark.runner import (
        AuthorizedConfiguration,
        RunnerContractError,
        maskimpute_decoder_for_configuration,
    )

    search = json.loads(Path("study/development_search.json").read_text())
    v27_payload = search["configurations"][2]["configuration"]
    v27 = _candidate_configuration(v27_payload, identifier="v27-test")
    v28_payload = _v28_configuration_payload()
    v28 = _candidate_configuration(v28_payload, identifier="v28-test")

    assert maskimpute_decoder_for_configuration(v27) == (
        "scaled_gaussian",
        None,
    )
    decoder, decoder_config = maskimpute_decoder_for_configuration(v28)
    assert decoder == "negative_binomial"
    assert type(decoder_config) is NegativeBinomialDecoderConfig
    assert decoder_config.to_dict() == v28_payload["decoder_hyperparameters"]

    ablation_payload = json.loads(Path("study/ablations.json").read_text())["variants"][0]
    ablation = AuthorizedConfiguration.create(
        method_id="capacity-matched-ae",
        configuration_id="capacity-matched-ae",
        kind="ablation",
        payload=ablation_payload,
        requires_count_score=False,
        requires_calibration=False,
    )
    assert maskimpute_decoder_for_configuration(ablation) == (
        "scaled_gaussian",
        None,
    )

    invalid_payloads = []
    for field, value in (
        ("method_version", "v27"),
        ("decoder", "scaled_gaussian"),
        ("encoder_mode", "implicit_numeric_zero"),
        ("output_policy", "full_gated"),
        ("score_policy", "direct_cross_fitted_count_score"),
    ):
        payload = _v28_configuration_payload()
        payload[field] = value
        invalid_payloads.append(payload)
    missing_decoder_config = _v28_configuration_payload()
    missing_decoder_config.pop("decoder_hyperparameters")
    invalid_payloads.append(missing_decoder_config)
    extra_decoder_config = _v28_configuration_payload()
    extra_decoder_config["decoder_hyperparameters"]["untracked"] = 1.0
    invalid_payloads.append(extra_decoder_config)
    invalid_prior = _v28_configuration_payload()
    invalid_prior["decoder_hyperparameters"]["dispersion_prior_strength"] = -1.0
    invalid_payloads.append(invalid_prior)

    for index, payload in enumerate(invalid_payloads):
        with pytest.raises(RunnerContractError):
            maskimpute_decoder_for_configuration(
                _candidate_configuration(payload, identifier=f"v28-invalid-{index}")
            )

    v28_ablation = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="v28-ablation",
        kind="ablation",
        payload=_v28_configuration_payload(),
        requires_count_score=True,
        requires_calibration=True,
    )
    with pytest.raises(RunnerContractError, match="development candidate"):
        maskimpute_decoder_for_configuration(v28_ablation)
