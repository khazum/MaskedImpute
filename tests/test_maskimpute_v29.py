from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch


def _counts() -> np.ndarray:
    return np.asarray(
        [
            [8, 0, 1, 0, 3, 0],
            [7, 1, 0, 0, 2, 1],
            [0, 6, 2, 1, 0, 0],
            [1, 7, 1, 0, 0, 2],
            [2, 0, 7, 1, 1, 0],
            [1, 1, 8, 0, 0, 1],
            [0, 2, 1, 8, 1, 0],
            [1, 0, 0, 7, 2, 1],
        ],
        dtype=np.int64,
    )


def test_structure_authority_is_deterministic_and_observed_only() -> None:
    from maskimpute.structure import (
        StructurePenaltyConfig,
        build_structure_authority,
    )

    config = StructurePenaltyConfig(variable_gene_count=4, neighborhood_k=3)
    first = build_structure_authority(_counts(), config)
    second = build_structure_authority(_counts().copy(), config)

    assert first.variable_gene_indices == second.variable_gene_indices
    np.testing.assert_array_equal(first.neighbor_indices, second.neighbor_indices)
    assert first.variable_gene_sha256 == second.variable_gene_sha256
    assert first.neighborhood_sha256 == second.neighborhood_sha256
    assert len(first.variable_gene_indices) == 4
    assert first.neighbor_indices.shape == (8, 3)
    assert all(
        row not in first.neighbor_indices[row]
        for row in range(first.neighbor_indices.shape[0])
    )
    assert not first.neighbor_indices.flags.writeable

    with pytest.raises(ValueError, match="variable gene checksum"):
        replace(first, variable_gene_sha256="0" * 64)
    changed_neighbors = first.neighbor_indices.copy()
    changed_neighbors[0] = changed_neighbors[0][::-1]
    with pytest.raises(ValueError, match="neighborhood checksum"):
        replace(first, neighbor_indices=changed_neighbors)


def test_structure_penalty_is_zero_at_observed_geometry_and_differentiable() -> None:
    from maskimpute.structure import (
        StructurePenaltyConfig,
        build_structure_authority,
        structure_preservation_loss,
    )

    counts = _counts()
    config = StructurePenaltyConfig(
        variable_gene_count=4,
        neighborhood_k=3,
        covariance_penalty_weight=0.1,
        neighborhood_penalty_weight=0.1,
    )
    authority = build_structure_authority(counts, config)
    target = torch.as_tensor(np.log1p(counts), dtype=torch.float64)
    prediction = target.clone().requires_grad_(True)
    rows = np.arange(counts.shape[0], dtype=np.int64)

    exact, components = structure_preservation_loss(
        prediction,
        target,
        rows,
        authority,
        config,
    )
    assert exact.item() == pytest.approx(0.0, abs=1e-12)
    assert components["covariance"] == pytest.approx(0.0, abs=1e-12)
    assert components["neighborhood"] == pytest.approx(0.0, abs=1e-12)

    perturbed = (target + torch.linspace(0.0, 1.0, target.numel()).reshape_as(target))
    perturbed.requires_grad_(True)
    loss, _ = structure_preservation_loss(
        perturbed,
        target,
        rows,
        authority,
        config,
    )
    assert loss.item() > 0
    loss.backward()
    assert perturbed.grad is not None
    assert torch.isfinite(perturbed.grad).all()


def test_v29_runner_resolves_exact_structure_configuration() -> None:
    from maskimpute.structure import StructurePenaltyConfig
    from maskimpute_benchmark.runner import (
        load_v29_revision_authority,
        maskimpute_decoder_for_configuration,
        maskimpute_structure_for_configuration,
    )

    authority = load_v29_revision_authority()
    candidate = next(
        value for value in authority.configurations if value.method_id == "maskimpute"
    )
    assert maskimpute_decoder_for_configuration(candidate)[0] == "negative_binomial"
    structure = maskimpute_structure_for_configuration(candidate)
    assert type(structure) is StructurePenaltyConfig
    assert structure.variable_gene_count == 200
    assert structure.neighborhood_k == 15


def test_revision_and_final_inference_share_one_calibration_usage_api() -> None:
    from maskimpute.ablations import _fit_ablation_once
    from maskimpute_benchmark.methods.maskimpute import _run_in_tree

    for function in (_fit_ablation_once, _run_in_tree):
        parameters = inspect.signature(function).parameters
        assert parameters["calibration_usage"].default == "development_holdout"
        assert "external_inference" not in parameters


def test_v29_structure_configuration_rejects_free_form_changes() -> None:
    from maskimpute.structure import StructurePenaltyConfig

    with pytest.raises(ValueError, match="variable_gene_count"):
        StructurePenaltyConfig(variable_gene_count=0)
    with pytest.raises(ValueError, match="neighborhood_k"):
        StructurePenaltyConfig(neighborhood_k=0)
    with pytest.raises(ValueError, match="weight"):
        StructurePenaltyConfig(covariance_penalty_weight=-0.1)


def test_v29_training_uses_fixed_structure_authority_without_extra_parameters() -> None:
    from maskimpute import MaskImputeConfig
    from maskimpute.nb_model import NegativeBinomialDecoderConfig
    from maskimpute.structure import StructurePenaltyConfig
    from maskimpute.train import train_v28, train_v29

    counts = _counts()
    score = np.zeros_like(counts, dtype=np.float64)
    score[counts == 0] = 0.4
    config = MaskImputeConfig(
        hidden_dims=(8,),
        latent_dim=3,
        batch_size=4,
        max_epochs=2,
        patience=2,
        artificial_mask_fraction=0.25,
        validation_fraction=0.25,
        seed=42,
    )
    decoder = NegativeBinomialDecoderConfig()
    structure = StructurePenaltyConfig(variable_gene_count=4, neighborhood_k=3)

    baseline = train_v28(
        counts,
        score,
        config,
        "cpu",
        decoder_config=decoder,
    )
    revised = train_v29(
        counts,
        score,
        config,
        "cpu",
        decoder_config=decoder,
        structure_config=structure,
    )

    baseline_parameters = sum(
        parameter.numel() for parameter in baseline.training.model.parameters()
    )
    revised_parameters = sum(
        parameter.numel() for parameter in revised.training.model.parameters()
    )
    assert revised_parameters == baseline_parameters
    assert revised.structure.variable_gene_sha256
    assert revised.structure.neighborhood_sha256
    assert len(revised.training.training_loss_history) >= 1
    assert all(np.isfinite(revised.training.training_loss_history))


def test_v29_structure_objective_does_not_receive_hidden_positive_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute import MaskImputeConfig
    from maskimpute.nb_model import NegativeBinomialDecoderConfig
    import maskimpute.structure as structure_module
    from maskimpute.structure import StructurePenaltyConfig
    from maskimpute.train import train_v29

    counts = _counts()
    hidden_positive_seen: list[bool] = []

    def audit_structure_target(
        prediction: torch.Tensor,
        observed_target: torch.Tensor,
        global_rows: object,
        authority: object,
        config: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del authority, config
        rows = np.asarray(global_rows, dtype=np.int64)
        target = observed_target.detach().cpu().numpy()
        hidden_positive_seen.append(bool(np.any((counts[rows] > 0) & (target == 0))))
        return prediction.sum() * 0.0, {"covariance": 0.0, "neighborhood": 0.0}

    monkeypatch.setattr(
        structure_module,
        "structure_preservation_loss",
        audit_structure_target,
    )
    score = np.zeros_like(counts, dtype=np.float64)
    score[counts == 0] = 0.4
    train_v29(
        counts,
        score,
        MaskImputeConfig(
            hidden_dims=(8,),
            latent_dim=3,
            batch_size=4,
            max_epochs=1,
            patience=1,
            artificial_mask_fraction=0.25,
            validation_fraction=0.25,
            seed=42,
        ),
        "cpu",
        decoder_config=NegativeBinomialDecoderConfig(),
        structure_config=StructurePenaltyConfig(
            variable_gene_count=4,
            neighborhood_k=3,
        ),
    )

    assert hidden_positive_seen
    assert any(hidden_positive_seen)


def test_v29_structure_authority_excludes_fixed_validation_positives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute import MaskImputeConfig
    from maskimpute.nb_model import NegativeBinomialDecoderConfig
    import maskimpute.structure as structure_module
    from maskimpute.structure import StructurePenaltyConfig
    from maskimpute.train import train_v29

    counts = _counts()
    authority_inputs: list[np.ndarray] = []
    real_builder = structure_module.build_structure_authority

    def audit_builder(observed_counts: object, config: object):
        authority_inputs.append(np.asarray(observed_counts).copy())
        return real_builder(observed_counts, config)

    monkeypatch.setattr(structure_module, "build_structure_authority", audit_builder)
    score = np.zeros_like(counts, dtype=np.float64)
    score[counts == 0] = 0.4
    outcome = train_v29(
        counts,
        score,
        MaskImputeConfig(
            hidden_dims=(8,),
            latent_dim=3,
            batch_size=4,
            max_epochs=1,
            patience=1,
            artificial_mask_fraction=0.25,
            validation_fraction=0.25,
            seed=42,
        ),
        "cpu",
        decoder_config=NegativeBinomialDecoderConfig(),
        structure_config=StructurePenaltyConfig(
            variable_gene_count=4,
            neighborhood_k=3,
        ),
    )

    assert len(authority_inputs) == 1
    expected = counts.copy()
    expected[outcome.training.validation_mask] = 0
    np.testing.assert_array_equal(authority_inputs[0], expected)
    assert np.any(outcome.training.validation_mask)
