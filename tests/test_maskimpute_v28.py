from __future__ import annotations

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
