from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
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
        dataset_sha = hashlib.sha256(f"{draw}:{view}".encode()).hexdigest()
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


def _binding_identity_calibration_artifact():
    from scipy.special import expit

    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
        fit_development_calibration,
    )

    levels = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.96])
    records = []
    for index, (draw, view, intercept) in enumerate(
        (
            ("draw-01", "moderate", -1.1),
            ("draw-01", "severe", -1.1),
            ("draw-02", "moderate", -0.9),
            ("draw-02", "severe", -0.9),
        ),
        start=5,
    ):
        calibrated = expit(0.5 * np.log(levels) - 1.7 * np.log1p(-levels) + intercept)
        probabilities = []
        targets = []
        for probability, calibrated_probability in zip(levels, calibrated, strict=True):
            positives = min(49, max(1, round(50 * calibrated_probability)))
            probabilities.extend([float(probability)] * 50)
            targets.extend([1] * positives + [0] * (50 - positives))
        dataset_sha = hashlib.sha256(f"{draw}:{view}:nonidentity".encode()).hexdigest()
        records.append(
            CalibrationRecord(
                p_pre_zero=probabilities,
                target=targets,
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


def test_tracked_ablation_registry_is_complete_and_prespecified():
    from maskimpute.ablations import load_ablation_registry

    registry = load_ablation_registry(Path("study/ablations.json"))

    assert registry.schema_version == 1
    assert registry.model_seeds == (42, 43, 44)
    assert tuple(spec.id for spec in registry.variants) == EXPECTED_VARIANTS
    assert registry.parameter_budget == "exact_nominal_match"
    assert registry.optimizer_budget == "shared_frozen_candidate_budget"
    assert registry.preprocessing_budget == "shared_except_named_component"
    assert registry.reference.score_source == "retained_calibrator"

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


def test_ablation_registry_rejects_duplicate_json_keys(tmp_path):
    from maskimpute.ablations import load_ablation_registry

    text = Path("study/ablations.json").read_text()
    text = text.replace(
        '"schema_version": 1,', '"schema_version": 1, "schema_version": 1,'
    )
    path = tmp_path / "duplicate.json"
    path.write_text(text)

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_ablation_registry(path)


def test_publication_ablation_authority_rejects_registry_byte_edits(
    tmp_path, monkeypatch
):
    import maskimpute.ablations as module

    payload = Path("study/ablations.json").read_text()
    changed = tmp_path / "ablations.json"
    changed.write_text(payload.replace('"gate_gamma"', '"gate_gamma"'))
    # A formatting-only edit still changes the prespecified authority bytes.
    changed.write_text(changed.read_text().replace("{\n", "{\n  \n", 1))
    monkeypatch.setattr(module, "_TRACKED_ABLATION_REGISTRY", changed)
    spec = module.load_ablation_registry(changed).reference

    with pytest.raises(ValueError, match="digest"):
        module._trusted_ablation_spec(spec)


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


def test_no_mask_compensation_is_an_active_expression_only_parameter():
    torch = pytest.importorskip("torch")
    from maskimpute.ablations import build_capacity_matched_model

    model = build_capacity_matched_model(
        n_genes=3,
        hidden_dims=(4,),
        latent_dim=2,
        encoder_mode="implicit_numeric_zero",
    )
    expression = torch.tensor([[1.0, 2.0, 3.0]])
    availability = torch.ones_like(expression, dtype=torch.bool)

    prepared = model.prepare_encoder_input(expression, availability)
    prepared.sum().backward()

    parameters = dict(model.named_parameters())
    assert "expression_curvature" in parameters
    assert parameters["expression_curvature"].grad is not None
    assert torch.all(parameters["expression_curvature"].grad != 0)


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


@pytest.mark.parametrize(
    ("override", "variant_id", "match"),
    [
        ({"gate_gamma": 0.0}, "no-gate", "gate_gamma.*positive"),
        (
            {"pre_zero_regularization": 0.0},
            "no-pre-zero-regularizer",
            "pre_zero_regularization.*positive",
        ),
    ],
)
def test_reference_configuration_cannot_make_a_named_ablation_degenerate(
    override, variant_id, match
):
    from maskimpute import MaskImputeConfig
    from maskimpute.ablations import load_ablation_registry, resolve_training_config

    spec = load_ablation_registry(Path("study/ablations.json")).by_id[variant_id]

    with pytest.raises(ValueError, match=match):
        resolve_training_config(MaskImputeConfig(**override), spec)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"gate_gamma": 0.0}, "enabled power-complement gate"),
        ({"pre_zero_regularization": 0.0}, "enabled pre-zero regularizer"),
    ],
)
def test_enabled_reference_components_cannot_be_silently_degenerate(override, match):
    from maskimpute import MaskImputeConfig
    from maskimpute.ablations import load_ablation_registry, resolve_training_config

    reference = load_ablation_registry(Path("study/ablations.json")).reference
    with pytest.raises(ValueError, match=match):
        resolve_training_config(MaskImputeConfig(**override), reference)


def test_ablation_outputs_isolate_gate_and_selective_copying():
    from maskimpute.ablations import apply_ablation_output, load_ablation_registry

    observed = np.array([[2, 0], [0, 3]], dtype=np.int64)
    candidates = np.array([[10.0, 8.0], [6.0, 12.0]])
    probability = np.array([[0.0, 0.75], [0.25, 0.0]])
    registry = load_ablation_registry(Path("study/ablations.json"))
    by_id = registry.by_id

    expected_selective = np.array([[2.0, 2.0], [4.5, 3.0]])
    np.testing.assert_allclose(
        apply_ablation_output(
            candidates,
            observed,
            probability,
            registry.reference,
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


def test_retained_reference_and_direct_score_ablation_use_only_count_score():
    from maskimpute.ablations import load_ablation_registry, resolve_score
    from maskimpute.calibration import ScoreCalibrator

    observed = np.array([[3, 0], [0, 0]], dtype=np.int64)
    raw = np.array([[0.0, 0.2], [0.8, 1.0]])
    by_id = load_ablation_registry(Path("study/ablations.json")).by_id
    calibrator = ScoreCalibrator.logistic(intercept=-0.4, slope=0.7)

    reference = load_ablation_registry(Path("study/ablations.json")).reference
    calibrated = resolve_score(raw, observed, reference, calibrator=calibrator)
    direct = resolve_score(
        raw,
        observed,
        by_id["direct-score"],
        calibrator=calibrator,
    )

    np.testing.assert_array_equal(direct, raw)
    expected = np.zeros_like(raw)
    expected[observed == 0] = calibrator.transform(raw[observed == 0])
    np.testing.assert_allclose(calibrated, expected)
    assert calibrated[0, 0] == 0.0
    assert not np.array_equal(calibrated, raw)
    with pytest.raises(ValueError, match="calibrator"):
        resolve_score(
            raw,
            observed,
            reference,
            calibrator=None,
        )


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


def test_single_ablation_rejects_unverified_score_or_calibration():
    from maskimpute import (
        MaskImputeConfig,
        PreZeroCountModelConfig,
        fit_p_pre_zero_count_model,
    )
    from maskimpute.ablations import (
        _fit_ablation_once,
        load_ablation_registry,
    )

    counts = np.array([[2, 0], [1, 3]], dtype=np.int64)
    raw = np.array([[0.0, 0.4], [0.0, 0.0]])
    reference = load_ablation_registry(Path("study/ablations.json")).reference

    with pytest.raises(TypeError, match="PreZeroCountModelScore"):
        _fit_ablation_once(
            counts,
            raw,
            _identity_calibration_artifact(),
            reference,
            MaskImputeConfig(max_epochs=1, patience=1),
            "cpu",
            cell_ids=("cell-a", "cell-b"),
            development_mechanism="symsim",
            development_biological_id="draw-01",
        )

    score = fit_p_pre_zero_count_model(
        counts,
        ("cell-a", "cell-b"),
        PreZeroCountModelConfig(n_folds=2),
    )
    with pytest.raises(TypeError, match="CalibrationArtifact"):
        _fit_ablation_once(
            counts,
            score,
            object(),
            reference,
            MaskImputeConfig(max_epochs=1, patience=1),
            "cpu",
            cell_ids=("cell-a", "cell-b"),
            development_mechanism="symsim",
            development_biological_id="draw-01",
        )


def test_development_reference_records_binding_identity_holdout_receipt():
    from maskimpute import (
        MaskImputeConfig,
        PreZeroCountModelConfig,
        fit_p_pre_zero_count_model,
    )
    from maskimpute.ablations import (
        _derive_prezero_execution_policy,
        _fit_ablation_once,
        load_ablation_registry,
    )

    counts = np.array([[5, 0, 1], [2, 3, 0], [0, 4, 2], [1, 0, 3]], dtype=np.int64)
    cell_ids = tuple(f"cell-{index}" for index in range(len(counts)))
    score = fit_p_pre_zero_count_model(
        counts,
        cell_ids,
        PreZeroCountModelConfig(n_folds=2, link_max_iter=25),
    )
    artifact = _binding_identity_calibration_artifact()
    result = _fit_ablation_once(
        counts,
        score,
        artifact,
        load_ablation_registry(Path("study/ablations.json")).reference,
        MaskImputeConfig(
            hidden_dims=(5,),
            latent_dim=2,
            batch_size=2,
            max_epochs=1,
            patience=1,
            seed=42,
        ),
        "cpu",
        cell_ids=cell_ids,
        development_mechanism="symsim",
        development_biological_id="draw-01",
    )
    direct = score.score_for_counts(counts, cell_ids)
    observed_zero = counts == 0
    expected = np.zeros_like(direct)
    expected[observed_zero] = artifact.transform_for_development_holdout(
        direct[observed_zero],
        mechanism="symsim",
        biological_id="draw-01",
    )
    full = np.zeros_like(direct)
    full[observed_zero] = artifact.transform(direct[observed_zero])

    np.testing.assert_allclose(result.p_pre_zero, expected)
    np.testing.assert_array_equal(result.p_pre_zero, full)
    assert result.diagnostics["score"]["calibration_scope"] == (
        "leave_one_biological_draw_out"
    )
    assert result.diagnostics["score"]["calibration_holdout"] == {
        "mechanism": "symsim",
        "biological_id": "draw-01",
    }
    fold = next(
        item
        for item in artifact.to_dict()["development_holdout_calibrators"]
        if item["mechanism"] == "symsim" and item["biological_id"] == "draw-01"
    )
    receipt = result.diagnostics["score"]["calibration_fold_receipt"]
    assert receipt["calibrator_algorithm"] == "identity"
    assert len(receipt["calibrator_sha256"]) == 64
    assert receipt["held_out_manifest_sha256s"] == tuple(
        fold["held_out_manifest_sha256s"]
    )
    assert receipt["training_manifest_sha256s"] == tuple(
        fold["training_manifest_sha256s"]
    )
    assert set(receipt["held_out_manifest_sha256s"]).isdisjoint(
        receipt["training_manifest_sha256s"]
    )
    assert result.diagnostics["score"]["equivalence_reason"] == (
        "retained_identity_calibrator_equals_direct_score"
    )
    calibration_payload = artifact.to_dict()
    calibration_file_sha256 = hashlib.sha256(
        (
            json.dumps(
                calibration_payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    assert result.diagnostics["score"]["calibration_file_sha256"] == (
        calibration_file_sha256
    )
    assert (
        result.diagnostics["score"]["calibration_payload_sha256"]
        == (calibration_payload["payload_sha256"])
    )
    assert calibration_file_sha256 != calibration_payload["payload_sha256"]
    derived_probability, derived_policy = _derive_prezero_execution_policy(
        counts,
        cell_ids,
        score,
        artifact,
        load_ablation_registry(Path("study/ablations.json")).reference,
        calibration_usage="development_holdout",
        development_mechanism="symsim",
        development_biological_id="draw-01",
    )
    np.testing.assert_array_equal(derived_probability, result.p_pre_zero)
    assert derived_policy == result.diagnostics["score"]


@pytest.mark.parametrize(
    ("spec_id", "mechanism", "expected_scope", "expected_equivalence"),
    (
        (
            "direct-score",
            "symsim",
            "not_applicable_direct_score",
            "direct_cross_fitted_count_score",
        ),
        (
            "maskimpute-reference",
            "sergio",
            "all_development_external_exact_truth_mechanism",
            "retained_identity_calibrator_equals_direct_score",
        ),
        (
            "maskimpute-reference",
            "sparsim",
            "all_development_external_exact_truth_mechanism",
            "retained_identity_calibrator_equals_direct_score",
        ),
        (
            "maskimpute-reference",
            "semisynthetic",
            "all_development_external_exact_truth_mechanism",
            "retained_identity_calibrator_equals_direct_score",
        ),
    ),
)
def test_development_score_derivation_uses_exact_policy_by_mechanism(
    spec_id: str,
    mechanism: str,
    expected_scope: str,
    expected_equivalence: str,
) -> None:
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model
    from maskimpute.ablations import (
        _derive_prezero_execution_policy,
        load_ablation_registry,
    )

    counts = np.array([[5, 0, 1], [2, 3, 0], [0, 4, 2], [1, 0, 3]], dtype=np.int64)
    cell_ids = tuple(f"cell-{index}" for index in range(len(counts)))
    score = fit_p_pre_zero_count_model(
        counts,
        cell_ids,
        PreZeroCountModelConfig(n_folds=2, link_max_iter=25),
    )
    artifact = _binding_identity_calibration_artifact()
    registry = load_ablation_registry(Path("study/ablations.json"))
    spec = (
        registry.reference
        if spec_id == registry.reference.id
        else registry.by_id[spec_id]
    )

    probability, diagnostics = _derive_prezero_execution_policy(
        counts,
        cell_ids,
        score,
        artifact,
        spec,
        calibration_usage="development_holdout",
        development_mechanism=mechanism,
        development_biological_id="draw-01",
    )

    np.testing.assert_array_equal(
        probability,
        score.score_for_counts(counts, cell_ids),
    )
    assert diagnostics["calibration_scope"] == expected_scope
    assert diagnostics["equivalence_reason"] == expected_equivalence
    assert diagnostics["calibration_holdout"] is None
    assert diagnostics["calibration_fold_receipt"] is None


def test_final_reference_uses_retained_all_development_calibrator_on_unseen_draw():
    from maskimpute import (
        MaskImputeConfig,
        PreZeroCountModelConfig,
        fit_p_pre_zero_count_model,
    )
    from maskimpute.ablations import _fit_ablation_once, load_ablation_registry

    counts = np.array([[5, 0, 1], [2, 3, 0], [0, 4, 2], [1, 0, 3]], dtype=np.int64)
    cell_ids = tuple(f"cell-{index}" for index in range(len(counts)))
    score = fit_p_pre_zero_count_model(
        counts,
        cell_ids,
        PreZeroCountModelConfig(n_folds=2, link_max_iter=25),
    )
    artifact = _binding_identity_calibration_artifact()
    result = _fit_ablation_once(
        counts,
        score,
        artifact,
        load_ablation_registry(Path("study/ablations.json")).reference,
        MaskImputeConfig(
            hidden_dims=(5,),
            latent_dim=2,
            batch_size=2,
            max_epochs=1,
            patience=1,
            seed=42,
        ),
        "cpu",
        cell_ids=cell_ids,
        development_mechanism="symsim",
        development_biological_id="draw-03",
        calibration_usage="retained_all_development",
    )
    direct = score.score_for_counts(counts, cell_ids)
    observed_zero = counts == 0
    expected = np.zeros_like(direct)
    expected[observed_zero] = artifact.transform(direct[observed_zero])

    np.testing.assert_allclose(result.p_pre_zero, expected)
    assert result.diagnostics["score"]["calibration_scope"] == (
        "retained_all_development_for_final_inference"
    )
    assert result.diagnostics["score"]["calibration_holdout"] is None
    assert result.diagnostics["score"]["calibration_fold_receipt"] is None


def test_single_ablation_executes_model_mask_loss_score_and_output_contracts():
    from maskimpute import (
        MaskImputeConfig,
        PreZeroCountModelConfig,
        fit_p_pre_zero_count_model,
    )
    from maskimpute.ablations import (
        _fit_ablation_once,
        load_ablation_registry,
    )

    counts = np.array(
        [
            [5, 0, 1, 0],
            [2, 3, 0, 1],
            [0, 4, 2, 1],
            [1, 0, 3, 2],
            [4, 1, 0, 2],
            [3, 2, 1, 0],
            [2, 0, 4, 1],
            [1, 3, 2, 0],
        ],
        dtype=np.int64,
    )
    cell_ids = tuple(f"cell-{index}" for index in range(len(counts)))
    score = fit_p_pre_zero_count_model(
        counts,
        cell_ids,
        PreZeroCountModelConfig(n_folds=2, link_max_iter=25),
    )
    calibration = _identity_calibration_artifact()
    registry = load_ablation_registry(Path("study/ablations.json"))
    config = MaskImputeConfig(
        hidden_dims=(7, 5),
        latent_dim=3,
        batch_size=4,
        max_epochs=2,
        patience=2,
        seed=42,
    )

    specifications = (registry.reference, *registry.variants)
    results = {
        spec.id: _fit_ablation_once(
            counts,
            score,
            calibration,
            spec,
            config,
            "cpu",
            cell_ids=cell_ids,
            development_mechanism="symsim",
            development_biological_id="draw-01",
        )
        for spec in specifications
    }

    parameter_counts = {
        result.diagnostics["ablation"]["nominal_parameter_count"]
        for result in results.values()
    }
    assert len(parameter_counts) == 1
    validation_hashes = {
        result.diagnostics["masks"]["fixed_validation_mask_sha256"]
        for result in results.values()
    }
    assert len(validation_hashes) == 1
    assert (
        results["capacity-matched-ae"].diagnostics["masks"]["epoch_positive_masking"]
        == "uniform"
    )
    assert (
        results["no-explicit-mask"].diagnostics["ablation"]["encoder_mode"]
        == "implicit_numeric_zero"
    )
    assert (
        results["no-explicit-mask"]
        .diagnostics["ablation"]["encoder_interpretation"]
        .startswith("broader_expression_only_encoder_representation")
    )
    assert (
        results["no-pre-zero-regularizer"].diagnostics["losses"][
            "natural_zero_penalty_weight"
        ]
        == 0.0
    )
    assert results["no-gate"].diagnostics["ablation"]["gate"] == "none"
    assert (
        results["full-denoising"].diagnostics["ablation"]["output_policy"]
        == "full_gated"
    )
    assert (
        results["direct-score"].diagnostics["score"]["equivalence_reason"]
        == "direct_cross_fitted_count_score"
    )
    assert (
        results["maskimpute-reference"].diagnostics["score"]["equivalence_reason"]
        == "retained_identity_calibrator_equals_direct_score"
    )
    reference_diagnostics = results["maskimpute-reference"].diagnostics
    assert reference_diagnostics["gate"] == {
        "family": "power_complement",
        "formula": "prediction * (1 - p_pre_zero) ** gamma",
        "gamma": 1.0,
    }
    assert len(reference_diagnostics["budget"]["base_config_sha256"]) == 64
    assert reference_diagnostics["budget"]["base_config"]["gate_gamma"] == 1.0
    np.testing.assert_array_equal(
        results["maskimpute-reference"].p_pre_zero,
        results["direct-score"].p_pre_zero,
    )
    np.testing.assert_allclose(
        results["maskimpute-reference"].selective_counts,
        results["direct-score"].selective_counts,
        rtol=0,
        atol=0,
    )
    noncontrol_mask_hashes = {
        results[variant_id].diagnostics["masks"]["epoch_training_mask_sha256"]
        for variant_id in (
            "maskimpute-reference",
            "no-gate",
            "no-pre-zero-regularizer",
            "no-explicit-mask",
            "full-denoising",
            "direct-score",
        )
    }
    assert len(noncontrol_mask_hashes) == 1
    for variant_id in (
        "maskimpute-reference",
        "no-gate",
        "no-pre-zero-regularizer",
        "no-explicit-mask",
        "direct-score",
    ):
        output = results[variant_id].selective_counts
        np.testing.assert_array_equal(output[counts > 0], counts[counts > 0])
        assert results[variant_id].diagnostics["score"]["artifact_integrity_verified"]
        assert not results[variant_id].diagnostics["score"][
            "source_authorized_by_panel"
        ]

    for variant_id in ("capacity-matched-ae", "full-denoising"):
        assert results[variant_id].primary_counts.shape == counts.shape
        with pytest.raises(AttributeError, match="nonselective"):
            _ = results[variant_id].selective_counts

    from maskimpute import impute_counts

    production = impute_counts(
        counts,
        score,
        config,
        "cpu",
        cell_ids=cell_ids,
        calibration_artifact=calibration,
    )
    reference = results["maskimpute-reference"]
    np.testing.assert_allclose(reference.selective_counts, production.selective_counts)
    np.testing.assert_allclose(reference.denoised_counts, production.denoised_counts)
    np.testing.assert_allclose(reference.latent, production.latent)
