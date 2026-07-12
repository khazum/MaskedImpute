from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pytest


EXPECTED_VARIANTS = (
    "capacity-matched-ae",
    "no-gate",
    "no-pre-zero-regularizer",
    "no-explicit-mask",
    "full-denoising",
    "direct-score",
    "calibrated-score",
)


def test_tracked_ablation_registry_is_complete_and_prespecified():
    from maskimpute.ablations import load_ablation_registry

    registry = load_ablation_registry(Path("study/ablations.json"))

    assert registry.schema_version == 1
    assert registry.model_seeds == (42, 43, 44)
    assert tuple(spec.id for spec in registry.variants) == EXPECTED_VARIANTS
    assert registry.parameter_budget == "exact_nominal_match"
    assert registry.optimizer_budget == "shared_frozen_candidate_budget"
    assert registry.preprocessing_budget == "shared_except_named_component"

    by_id = registry.by_id
    assert by_id["capacity-matched-ae"].positive_masking == "uniform"
    assert by_id["capacity-matched-ae"].pre_zero_regularizer is False
    assert by_id["capacity-matched-ae"].gate == "none"
    assert by_id["capacity-matched-ae"].output_policy == "full_ungated"
    assert by_id["no-gate"].gate == "none"
    assert by_id["no-pre-zero-regularizer"].pre_zero_regularizer is False
    assert by_id["no-explicit-mask"].encoder_mode == "implicit_numeric_zero"
    assert by_id["full-denoising"].output_policy == "full_gated"
    assert by_id["direct-score"].score_source == "direct"
    assert by_id["calibrated-score"].score_source == "retained_calibrator"


def test_ablation_registry_is_immutable_and_defensively_indexed():
    from maskimpute.ablations import load_ablation_registry

    registry = load_ablation_registry(Path("study/ablations.json"))
    with pytest.raises(FrozenInstanceError):
        registry.model_seeds = (1,)  # type: ignore[misc]

    first = registry.by_id
    second = registry.by_id
    assert first is not second
    first.pop("no-gate")
    assert "no-gate" in second


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload.update(schema_version=2), "schema_version"),
        (
            lambda payload: payload["variants"].append(payload["variants"][0]),
            "duplicate",
        ),
        (
            lambda payload: payload["variants"][1].update(gate="mystery"),
            "gate",
        ),
        (
            lambda payload: payload["variants"][1].update(hidden_weight=0.2),
            "missing or extra",
        ),
        (
            lambda payload: payload.update(model_seeds=[42, 42, 44]),
            "model_seeds",
        ),
    ],
)
def test_ablation_registry_fails_closed_on_unprespecified_changes(
    tmp_path, mutator, match
):
    from maskimpute.ablations import load_ablation_registry

    payload = json.loads(Path("study/ablations.json").read_text())
    mutator(payload)
    path = tmp_path / "ablations.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=match):
        load_ablation_registry(path)


def test_explicit_and_no_explicit_mask_models_have_exact_parameter_parity():
    torch = pytest.importorskip("torch")
    from maskimpute.ablations import build_capacity_matched_model

    explicit = build_capacity_matched_model(
        n_genes=11,
        hidden_dims=(13, 7),
        latent_dim=5,
        encoder_mode="explicit_mask",
    )
    implicit = build_capacity_matched_model(
        n_genes=11,
        hidden_dims=(13, 7),
        latent_dim=5,
        encoder_mode="implicit_numeric_zero",
    )

    explicit_count = sum(parameter.numel() for parameter in explicit.parameters())
    implicit_count = sum(parameter.numel() for parameter in implicit.parameters())
    assert explicit_count == implicit_count
    assert explicit.parameter_count == explicit_count
    assert implicit.parameter_count == implicit_count
    assert not any("mask" in name for name, _ in implicit.named_parameters())
    assert isinstance(explicit, torch.nn.Module)


def test_no_explicit_mask_representation_has_no_availability_channel_or_token():
    torch = pytest.importorskip("torch")
    from maskimpute.ablations import build_capacity_matched_model

    model = build_capacity_matched_model(
        n_genes=3,
        hidden_dims=(4,),
        latent_dim=2,
        encoder_mode="implicit_numeric_zero",
    )
    expression = torch.tensor([[1.0, 0.0, 3.0]])
    first = model.prepare_encoder_input(
        expression,
        torch.tensor([[True, True, True]]),
    )
    second = model.prepare_encoder_input(
        expression,
        torch.tensor([[True, False, True]]),
    )

    # A numeric zero is indistinguishable from an unavailable zero in this ablation.
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (1, 6)


def test_uniform_masking_control_is_deterministic_and_uses_training_positives_only():
    from maskimpute.ablations import make_uniform_positive_mask

    counts = np.array([[1, 0, 2, 3], [4, 5, 0, 6]], dtype=np.int64)
    validation = np.array([[True, False, False, False], [False, False, False, True]])
    first = make_uniform_positive_mask(
        counts,
        validation_mask=validation,
        fraction=0.5,
        rng=np.random.default_rng(71),
    )
    second = make_uniform_positive_mask(
        counts,
        validation_mask=validation,
        fraction=0.5,
        rng=np.random.default_rng(71),
    )

    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first) == 2
    assert not np.any(first & validation)
    assert not np.any(first & (counts == 0))


def test_named_variants_do_not_change_optimizer_or_architecture_budget():
    from maskimpute import MaskImputeConfig
    from maskimpute.ablations import (
        load_ablation_registry,
        optimization_budget_signature,
        resolve_training_config,
    )

    base = MaskImputeConfig(
        hidden_dims=(9, 5),
        latent_dim=3,
        batch_size=7,
        max_epochs=13,
        patience=4,
        seed=43,
    )
    registry = load_ablation_registry(Path("study/ablations.json"))
    signatures = {
        optimization_budget_signature(resolve_training_config(base, spec))
        for spec in registry.variants
    }

    assert len(signatures) == 1
    for spec in registry.variants:
        resolved = resolve_training_config(base, spec)
        assert resolved.hidden_dims == base.hidden_dims
        assert resolved.latent_dim == base.latent_dim
        assert resolved.artificial_mask_fraction == base.artificial_mask_fraction
        expected_regularization = (
            base.pre_zero_regularization if spec.pre_zero_regularizer else 0.0
        )
        assert resolved.pre_zero_regularization == expected_regularization


def test_ablation_outputs_isolate_gate_and_selective_copying():
    from maskimpute.ablations import apply_ablation_output, load_ablation_registry

    observed = np.array([[2, 0], [0, 3]], dtype=np.int64)
    candidates = np.array([[10.0, 8.0], [6.0, 12.0]])
    probability = np.array([[0.0, 0.75], [0.25, 0.0]])
    by_id = load_ablation_registry(Path("study/ablations.json")).by_id

    expected_selective = np.array([[2.0, 2.0], [4.5, 3.0]])
    np.testing.assert_allclose(
        apply_ablation_output(
            candidates,
            observed,
            probability,
            by_id["direct-score"],
            gamma=1.0,
        ),
        expected_selective,
    )
    np.testing.assert_allclose(
        apply_ablation_output(
            candidates,
            observed,
            probability,
            by_id["no-gate"],
            gamma=1.0,
        ),
        np.array([[2.0, 8.0], [6.0, 3.0]]),
    )
    np.testing.assert_allclose(
        apply_ablation_output(
            candidates,
            observed,
            probability,
            by_id["capacity-matched-ae"],
            gamma=1.0,
        ),
        candidates,
    )
    np.testing.assert_allclose(
        apply_ablation_output(
            candidates,
            observed,
            probability,
            by_id["full-denoising"],
            gamma=1.0,
        ),
        np.array([[10.0, 2.0], [4.5, 12.0]]),
    )


def test_direct_and_calibrated_score_variants_use_only_the_supplied_count_score():
    from maskimpute.ablations import load_ablation_registry, resolve_score
    from maskimpute.calibration import ScoreCalibrator

    raw = np.array([[0.0, 0.2], [0.8, 1.0]])
    by_id = load_ablation_registry(Path("study/ablations.json")).by_id
    calibrator = ScoreCalibrator.logistic(intercept=-0.4, slope=0.7)

    direct = resolve_score(raw, by_id["direct-score"], calibrator=calibrator)
    calibrated = resolve_score(
        raw,
        by_id["calibrated-score"],
        calibrator=calibrator,
    )

    np.testing.assert_array_equal(direct, raw)
    np.testing.assert_allclose(calibrated, calibrator.transform(raw))
    assert not np.array_equal(calibrated, raw)
    with pytest.raises(ValueError, match="calibrator"):
        resolve_score(raw, by_id["calibrated-score"], calibrator=None)


def test_noncontrol_ablations_change_only_the_declared_component():
    from maskimpute.ablations import load_ablation_registry

    registry = load_ablation_registry(Path("study/ablations.json"))
    reference = registry.reference
    allowed_changes = {
        "no-gate": {"gate"},
        "no-pre-zero-regularizer": {"pre_zero_regularizer"},
        "no-explicit-mask": {"encoder_mode"},
        "full-denoising": {"output_policy"},
        "direct-score": {"score_source"},
        "calibrated-score": set(),
    }
    fields = {
        "positive_masking",
        "pre_zero_regularizer",
        "encoder_mode",
        "gate",
        "output_policy",
        "score_source",
    }
    for variant_id, expected in allowed_changes.items():
        spec = registry.by_id[variant_id]
        changed = {
            field
            for field in fields
            if getattr(spec, field) != getattr(reference, field)
        }
        assert changed == expected

    with pytest.raises(ValueError, match="reference"):
        replace(reference, gate="none").validate_against_reference(reference)
