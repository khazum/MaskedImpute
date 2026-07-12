from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import expit, logit


def _sha(character: str) -> str:
    return character * 64


def _record(
    mechanism: str,
    biological_id: str,
    manifest_character: str,
    probabilities=(0.1, 0.25, 0.7, 0.9),
    targets=None,
    *,
    namespace="dev",
    data_role="development",
    technical_view="moderate",
):
    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
    )

    if targets is None:
        targets = tuple(int(value >= 0.5) for value in probabilities)
    dataset_digest = hashlib.sha256(
        f"{mechanism}:{biological_id}:{technical_view}:{manifest_character}".encode()
    ).hexdigest()
    return CalibrationRecord(
        p_pre_zero=probabilities,
        target=targets,
        mechanism=mechanism,
        biological_id=biological_id,
        manifest_sha256=_sha(manifest_character),
        truth_kind="exact_pre_capture",
        namespace=namespace,
        data_role=data_role,
        technical_view=technical_view,
        dataset_id=f"dataset-{dataset_digest[:24]}",
        dataset_sha256=dataset_digest,
        protocol_sha256=DEVELOPMENT_PROTOCOL_SHA256,
    )


def test_calibration_record_is_validated_immutable_exact_truth_snapshot():
    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
    )

    probability = np.array([0.1, 0.8])
    target = np.array([0, 1], dtype=np.int64)
    record = CalibrationRecord(
        p_pre_zero=probability,
        target=target,
        mechanism="symsim",
        biological_id="draw-01",
        manifest_sha256=_sha("a"),
        truth_kind="exact_pre_capture",
        namespace="dev",
        data_role="development",
        technical_view="moderate",
        dataset_id="dataset-" + "b" * 24,
        dataset_sha256=_sha("c"),
        protocol_sha256=DEVELOPMENT_PROTOCOL_SHA256,
    )
    probability[:] = 0.5
    target[:] = 0

    assert record.p_pre_zero == (0.1, 0.8)
    assert record.target == (0, 1)
    assert not hasattr(record, "__dict__")
    with pytest.raises(AttributeError):
        record.mechanism = "sergio"


def test_calibration_scope_constant_binds_the_tracked_development_protocol():
    from maskimpute.calibration import DEVELOPMENT_PROTOCOL_SHA256

    protocol = Path("study/protocol.json").read_bytes()

    assert hashlib.sha256(protocol).hexdigest() == DEVELOPMENT_PROTOCOL_SHA256
    payload = json.loads(protocol)
    assert payload["development"] == {
        "namespace": "dev",
        "draws_per_condition": 2,
        "cells": 900,
        "genes": 500,
    }


@pytest.mark.parametrize(
    "override",
    [
        {"p_pre_zero": ()},
        {"p_pre_zero": (0.1,), "target": (0, 1)},
        {"p_pre_zero": (np.nan, 0.2)},
        {"p_pre_zero": (-0.1, 0.2)},
        {"p_pre_zero": (0.1, 1.1)},
        {"p_pre_zero": np.array([0.1, 0.2], dtype=object)},
        {"p_pre_zero": np.ma.array([0.1, 0.2], mask=[False, True])},
        {"target": (0, 2)},
        {"target": (False, True)},
        {"target": np.array([0, 256], dtype=np.uint16)},
        {"mechanism": "Sym Sim"},
        {"mechanism": "sergio"},
        {"biological_id": "draw-1"},
        {"biological_id": "draw-00"},
        {"biological_id": "draw-03"},
        {"manifest_sha256": "A" * 64},
        {"manifest_sha256": "a" * 63},
        {"truth_kind": "exact_continuous"},
        {"truth_kind": "proxy_high_depth"},
        {"namespace": "final"},
        {"data_role": "final_evaluation"},
        {"technical_view": "unknown"},
        {"dataset_id": "dataset-not-a-digest"},
        {"dataset_sha256": "a" * 63},
        {"protocol_sha256": "f" * 64},
    ],
)
def test_calibration_record_rejects_ambiguous_or_noncanonical_fields(override):
    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
    )

    arguments = {
        "p_pre_zero": (0.1, 0.8),
        "target": (0, 1),
        "mechanism": "symsim",
        "biological_id": "draw-01",
        "manifest_sha256": _sha("a"),
        "truth_kind": "exact_pre_capture",
        "namespace": "dev",
        "data_role": "development",
        "technical_view": "moderate",
        "dataset_id": "dataset-" + "b" * 24,
        "dataset_sha256": _sha("c"),
        "protocol_sha256": DEVELOPMENT_PROTOCOL_SHA256,
    }
    arguments.update(override)
    with pytest.raises((TypeError, ValueError)):
        CalibrationRecord(**arguments)


def test_calibrator_formula_oracles_are_monotone_and_finite_at_extremes():
    from maskimpute.calibration import ScoreCalibrator

    probability = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)

    identity = ScoreCalibrator.identity()
    logistic = ScoreCalibrator.logistic(intercept=-0.3, slope=1.7)
    beta = ScoreCalibrator.beta(a=1.2, b=0.8, intercept=0.15)

    np.testing.assert_array_equal(identity.transform(probability), probability)
    np.testing.assert_allclose(
        logistic.transform(probability),
        expit(-0.3 + 1.7 * logit(clipped)),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        beta.transform(probability),
        expit(0.15 + 1.2 * np.log(clipped) - 0.8 * np.log1p(-clipped)),
        rtol=1e-13,
        atol=1e-13,
    )
    for calibrator in (identity, logistic, beta):
        transformed = calibrator.transform(probability)
        assert np.all(np.isfinite(transformed))
        assert np.all((transformed >= 0) & (transformed <= 1))
        assert np.all(np.diff(transformed) >= 0)


def test_extreme_finite_calibrator_coefficients_transform_without_warnings():
    from maskimpute.calibration import ScoreCalibrator

    probability = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    calibrators = (
        ScoreCalibrator.logistic(intercept=0.0, slope=1e308),
        ScoreCalibrator.beta(a=1e308, b=1e308, intercept=0.0),
    )

    for calibrator in calibrators:
        transformed = calibrator.transform(probability)
        assert np.all(np.isfinite(transformed))
        assert np.all((0 <= transformed) & (transformed <= 1))
        assert np.all(np.diff(transformed) >= 0)


@pytest.mark.parametrize(
    "factory_arguments",
    [
        ("logistic", {"intercept": 0.0, "slope": -0.1}),
        ("logistic", {"intercept": np.nan, "slope": 1.0}),
        ("logistic", {"intercept": False, "slope": 1.0}),
        ("logistic", {"intercept": "0", "slope": 1.0}),
        ("beta", {"a": -0.1, "b": 1.0, "intercept": 0.0}),
        ("beta", {"a": 1.0, "b": -0.1, "intercept": 0.0}),
    ],
)
def test_parametric_calibrators_reject_nonmonotone_or_nonfinite_coefficients(
    factory_arguments,
):
    from maskimpute.calibration import ScoreCalibrator

    name, arguments = factory_arguments
    with pytest.raises((TypeError, ValueError)):
        getattr(ScoreCalibrator, name)(**arguments)


@pytest.mark.parametrize(
    ("knots", "values"),
    [
        ((0.1, True), (0.2, 0.8)),
        ((0.1, "0.9"), (0.2, 0.8)),
        ((0.1, 0.9), (0.2, False)),
    ],
)
def test_isotonic_calibrator_rejects_coercible_non_numeric_parameters(knots, values):
    from maskimpute.calibration import ScoreCalibrator

    with pytest.raises((TypeError, ValueError)):
        ScoreCalibrator.isotonic(knots=knots, values=values)


def test_weighted_isotonic_pav_matches_hand_computed_oracle():
    from maskimpute.calibration import fit_score_calibrator

    probability = np.array([0.1, 0.2, 0.3, 0.4])
    target = np.array([0, 1, 0, 1])
    calibrator = fit_score_calibrator(
        "isotonic",
        probability,
        target,
        np.ones(4),
    )

    np.testing.assert_allclose(
        calibrator.transform(probability),
        [0.0, 0.5, 0.5, 1.0],
    )
    dense = calibrator.transform(np.linspace(0, 1, 101))
    assert np.all(np.diff(dense) >= -1e-15)


def test_score_transform_signature_cannot_receive_reconstruction_or_truth_features():
    from maskimpute.calibration import ScoreCalibrator

    parameters = tuple(inspect.signature(ScoreCalibrator.transform).parameters)
    assert parameters == ("self", "p_pre_zero")
    calibrator = ScoreCalibrator.identity()
    with pytest.raises(TypeError):
        calibrator.transform(np.array([0.2]), reconstruction=np.array([1.0]))


def test_development_weights_balance_draws_records_and_entries():
    from maskimpute.calibration import development_weights

    records = (
        _record("symsim", "draw-01", "a", probabilities=(0.1, 0.9)),
        _record(
            "symsim",
            "draw-01",
            "b",
            probabilities=(0.1, 0.2, 0.8, 0.9),
            targets=(0, 0, 1, 1),
            technical_view="severe",
        ),
        _record("symsim", "draw-02", "c", probabilities=(0.1, 0.9)),
        _record(
            "symsim",
            "draw-02",
            "d",
            probabilities=(0.1, 0.9),
            technical_view="severe",
        ),
    )

    weights = development_weights(records)

    assert sum(sum(value) for value in weights.values()) == pytest.approx(1.0)
    assert sum(weights[_sha("a")]) + sum(weights[_sha("b")]) == pytest.approx(1 / 2)
    assert sum(weights[_sha("c")]) + sum(weights[_sha("d")]) == pytest.approx(1 / 2)
    assert sum(weights[_sha("a")]) == pytest.approx(sum(weights[_sha("b")]))
    assert sum(weights[_sha("c")]) == pytest.approx(sum(weights[_sha("d")]))
    assert weights[_sha("a")][0] == pytest.approx(weights[_sha("a")][1])
    assert weights[_sha("b")][0] == pytest.approx(sum(weights[_sha("b")]) / 4)


def test_records_reject_duplicate_manifest_and_are_order_canonical():
    from maskimpute.calibration import validate_calibration_records

    first = _record("symsim", "draw-02", "b", technical_view="severe")
    second = _record("symsim", "draw-01", "a")
    third = _record("symsim", "draw-01", "c", technical_view="severe")
    fourth = _record("symsim", "draw-02", "d")

    canonical = validate_calibration_records((first, second, third, fourth))

    assert canonical == (second, third, fourth, first)
    with pytest.raises(ValueError, match="duplicate.*manifest"):
        validate_calibration_records((first, first))


@pytest.mark.parametrize("restriction", ["missing-view", "duplicate-view"])
def test_records_require_the_exact_prespecified_draw_view_panel(restriction):
    from maskimpute.calibration import validate_calibration_records

    records = list(_development_records())
    if restriction == "missing-view":
        records.pop()
    else:
        records[-1] = _record(
            "symsim",
            "draw-02",
            "5",
            technical_view="moderate",
        )

    with pytest.raises(ValueError, match="complete.*draw-view panel"):
        validate_calibration_records(records)


def test_records_reject_reused_dataset_content_across_panel_slots():
    import dataclasses

    from maskimpute.calibration import validate_calibration_records

    records = list(_development_records())
    records[1] = dataclasses.replace(
        records[1],
        dataset_sha256=records[0].dataset_sha256,
    )

    with pytest.raises(ValueError, match="duplicate.*dataset.*sha256|dataset.*digest"):
        validate_calibration_records(records)


def test_lodo_folds_hold_out_entire_mechanism_biological_draw_without_leakage():
    from maskimpute.calibration import cross_validate_calibrator

    records = tuple(
        _record(
            "symsim",
            f"draw-0{draw}",
            manifest,
            technical_view=technical_view,
        )
        for draw, manifests in ((1, ("a", "b")), (2, ("c", "d")))
        for technical_view, manifest in zip(("moderate", "severe"), manifests)
    )

    first = cross_validate_calibrator(records, "isotonic")
    second = cross_validate_calibrator(tuple(reversed(records)), "isotonic")

    assert first == second
    assert len(first.folds) == 2
    assert not first.fit_failures
    for fold in first.folds:
        assert set(fold.held_out_manifests).isdisjoint(fold.training_manifests)
        held_records = [
            record
            for record in records
            if record.manifest_sha256 in fold.held_out_manifests
        ]
        assert {
            (record.mechanism, record.biological_id) for record in held_records
        } == {(fold.mechanism, fold.biological_id)}
    assert {manifest for manifest, _ in first.predictions} == {
        record.manifest_sha256 for record in records
    }


def test_calibration_metrics_match_weighted_brier_and_log_loss_oracles():
    from maskimpute.calibration import calibration_metrics

    probability = np.array([0.2, 0.4, 0.6, 0.8])
    target = np.array([0, 1, 0, 1])
    weights = np.array([1.0, 2.0, 1.0, 2.0])

    metrics = calibration_metrics(probability, target, weights)

    expected_brier = np.average((probability - target) ** 2, weights=weights)
    expected_log_loss = np.average(
        -(target * np.log(probability) + (1 - target) * np.log1p(-probability)),
        weights=weights,
    )
    assert metrics.brier == pytest.approx(expected_brier)
    assert metrics.log_loss == pytest.approx(expected_log_loss)
    assert metrics.n == 4
    assert metrics.calibration_intercept is not None
    assert metrics.calibration_slope is not None
    assert np.isfinite(metrics.calibration_intercept)
    assert np.isfinite(metrics.calibration_slope)


def test_calibration_is_invariant_to_huge_finite_weight_scale_without_warnings():
    from maskimpute.calibration import calibration_metrics, fit_score_calibrator

    probability = np.array([0.2, 0.2])
    target = np.array([0, 1])
    huge = np.array([1e308, 1e308])

    metrics = calibration_metrics(probability, target, huge)
    isotonic = fit_score_calibrator("isotonic", probability, target, huge)

    assert metrics.brier == pytest.approx(0.34)
    assert metrics.log_loss == pytest.approx(-0.5 * (np.log(0.8) + np.log(0.2)))
    np.testing.assert_allclose(isotonic.transform(probability), [0.5, 0.5])


def test_calibration_line_reports_complete_separation_as_undefined():
    from maskimpute.calibration import calibration_metrics

    metrics = calibration_metrics(
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0, 0, 1, 1]),
        np.ones(4),
    )

    assert metrics.calibration_intercept is None
    assert metrics.calibration_slope is None
    assert metrics.slope_reason == "complete_or_quasi_separation"


def _metrics(brier=0.2, log_loss=0.5, slope=1.0):
    from maskimpute.calibration import CalibrationMetrics

    return CalibrationMetrics(
        brier=brier,
        log_loss=log_loss,
        calibration_intercept=0.0,
        calibration_slope=slope,
        slope_reason=None,
        n=20,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"brier": np.nan},
        {"brier": -0.1},
        {"brier": 1.1},
        {"log_loss": np.inf},
        {"log_loss": -0.1},
        {"n": True},
        {"n": 0},
        {"calibration_intercept": None, "calibration_slope": 1.0},
        {"calibration_intercept": 0.0, "calibration_slope": None},
        {"calibration_intercept": 0.0, "calibration_slope": 1.0, "slope_reason": "bad"},
    ],
)
def test_calibration_metrics_reject_invalid_manual_candidate_values(override):
    from maskimpute.calibration import CalibrationMetrics

    arguments = {
        "brier": 0.2,
        "log_loss": 0.5,
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "slope_reason": None,
        "n": 20,
    }
    arguments.update(override)
    with pytest.raises((TypeError, ValueError)):
        CalibrationMetrics(**arguments)


def _evaluation(
    algorithm,
    mechanism_metrics,
    aggregate=None,
    *,
    failures=(),
    eligible=False,
    reasons=(),
):
    from maskimpute.calibration import CandidateEvaluation

    return CandidateEvaluation(
        algorithm=algorithm,
        mechanism_metrics=tuple(sorted(mechanism_metrics.items())),
        aggregate_metrics=aggregate or _metrics(),
        fit_failures=tuple(failures),
        brier_improved_mechanisms=(),
        eligible=eligible,
        eligibility_reasons=tuple(reasons),
    )


def test_retention_gate_requires_three_brier_wins_log_loss_and_slope_safety():
    from maskimpute.calibration import (
        CalibrationThresholds,
        retention_reasons,
    )

    mechanisms = ("m1", "m2", "m3", "m4")
    identity = _evaluation(
        "identity", {mechanism: _metrics() for mechanism in mechanisms}
    )
    candidate = _evaluation(
        "logistic",
        {
            "m1": _metrics(brier=0.19),
            "m2": _metrics(brier=0.19),
            "m3": _metrics(brier=0.19),
            "m4": _metrics(brier=0.21, slope=1.21),
        },
        aggregate=_metrics(brier=0.195),
    )

    reasons, improved = retention_reasons(
        candidate,
        identity,
        CalibrationThresholds(),
    )

    assert improved == ("m1", "m2", "m3")
    assert "mechanism_calibration_slope_outside_tolerance:m4" in reasons


def test_retention_gate_passes_only_when_every_prespecified_guardrail_passes():
    from maskimpute.calibration import CalibrationThresholds, retention_reasons

    mechanisms = ("m1", "m2", "m3", "m4")
    identity = _evaluation(
        "identity", {mechanism: _metrics() for mechanism in mechanisms}
    )
    candidate = _evaluation(
        "beta",
        {
            "m1": _metrics(brier=0.19, log_loss=0.5005),
            "m2": _metrics(brier=0.19),
            "m3": _metrics(brier=0.19),
            "m4": _metrics(brier=0.21),
        },
        aggregate=_metrics(brier=0.195, log_loss=0.5005),
    )

    reasons, improved = retention_reasons(
        candidate,
        identity,
        CalibrationThresholds(),
    )

    assert reasons == ()
    assert improved == ("m1", "m2", "m3")


@pytest.mark.parametrize(
    "override",
    [
        {"minimum_mechanisms_improved": 1},
        {"minimum_mechanisms_improved": 4},
        {"brier_improvement_epsilon": 0.0},
        {"log_loss_worsening_tolerance": 0.0},
        {"calibration_slope_lower": 0.7},
        {"calibration_slope_upper": 1.3},
    ],
)
def test_publication_calibration_thresholds_cannot_be_weakened_or_changed(override):
    from maskimpute.calibration import CalibrationThresholds

    with pytest.raises(ValueError, match="prespecified|publication|exactly"):
        CalibrationThresholds(**override)


def test_public_calibration_functions_revalidate_mutated_or_duck_typed_thresholds():
    from maskimpute.calibration import (
        CalibrationThresholds,
        evaluate_calibration_candidates,
        fit_development_calibration,
    )

    mutated = CalibrationThresholds()
    object.__setattr__(mutated, "minimum_mechanisms_improved", 1)

    class LooseThresholds:
        minimum_mechanisms_improved = 1
        brier_improvement_epsilon = 0.0
        log_loss_worsening_tolerance = 1.0
        calibration_slope_lower = 0.0
        calibration_slope_upper = 10.0

    with pytest.raises((TypeError, ValueError), match="threshold|prespecified"):
        evaluate_calibration_candidates(_development_records(), mutated)
    with pytest.raises((TypeError, ValueError), match="threshold|prespecified"):
        fit_development_calibration(_development_records(), LooseThresholds())


def test_selection_is_deterministic_and_retains_every_candidate_report():
    from maskimpute.calibration import select_candidate

    mechanisms = {name: _metrics(brier=0.19) for name in ("a", "b", "c", "d")}
    identity = _evaluation("identity", mechanisms, eligible=True)
    logistic = _evaluation(
        "logistic", mechanisms, aggregate=_metrics(brier=0.18), eligible=True
    )
    beta = _evaluation(
        "beta", mechanisms, aggregate=_metrics(brier=0.18), eligible=True
    )
    isotonic = _evaluation(
        "isotonic", mechanisms, aggregate=_metrics(brier=0.17), eligible=False
    )

    decision = select_candidate((isotonic, beta, identity, logistic))

    assert decision.selected_algorithm == "logistic"
    assert tuple(item.algorithm for item in decision.candidates) == (
        "identity",
        "logistic",
        "beta",
        "isotonic",
    )


def test_calibration_decision_rejects_ineligible_or_incomplete_selection():
    from maskimpute.calibration import CalibrationDecision

    mechanisms = {name: _metrics() for name in ("a", "b", "c", "d")}
    identity = _evaluation("identity", mechanisms, eligible=True)
    logistic = _evaluation("logistic", mechanisms, eligible=False)

    with pytest.raises(ValueError):
        CalibrationDecision("logistic", (identity, logistic))


def test_cross_validated_candidate_panel_keeps_identity_and_all_three_alternatives():
    from maskimpute.calibration import evaluate_calibration_candidates

    records = _development_records()

    decision = evaluate_calibration_candidates(records)

    assert tuple(candidate.algorithm for candidate in decision.candidates) == (
        "identity",
        "logistic",
        "beta",
        "isotonic",
    )
    assert decision.selected_algorithm == "identity"
    assert decision.candidates[0].eligible is True
    assert decision.candidates[0].eligibility_reasons == ("default_uncalibrated_score",)
    for candidate in decision.candidates[1:]:
        assert any(
            reason.startswith("insufficient_eligible_exact_truth_mechanisms")
            for reason in candidate.eligibility_reasons
        )


def _development_records():
    return (
        _record("symsim", "draw-01", "1", technical_view="moderate"),
        _record("symsim", "draw-01", "2", technical_view="severe"),
        _record("symsim", "draw-02", "3", technical_view="moderate"),
        _record("symsim", "draw-02", "4", technical_view="severe"),
    )


def test_fitted_artifact_is_deterministic_complete_and_score_only():
    from maskimpute.calibration import fit_development_calibration

    records = _development_records()

    first = fit_development_calibration(records)
    second = fit_development_calibration(tuple(reversed(records)))

    assert first.to_dict() == second.to_dict()
    payload = first.to_dict()
    assert payload["schema_version"] == 2
    assert payload["artifact_type"] == "maskimpute_prezero_calibration"
    assert payload["inference_features"] == ["p_pre_zero"]
    assert payload["selected_algorithm"] == payload["calibrator"]["algorithm"]
    assert payload["truth_eligibility"] == {
        "accepted_truth_kind": "exact_pre_capture",
        "eligible_mechanisms": ["symsim"],
        "eligible_mechanism_count": 1,
        "minimum_mechanisms_required": 3,
        "panel_limitations": {
            "semisynthetic": "proxy_truth_not_exact",
            "sergio": "undefined_for_continuous_truth",
            "sparsim": "undefined_for_continuous_truth",
        },
    }
    assert [item["algorithm"] for item in payload["selection"]["candidates"]] == [
        "identity",
        "logistic",
        "beta",
        "isotonic",
    ]
    assert payload["data_scope"] == {
        "allowed_biological_ids": ["draw-01", "draw-02"],
        "data_role": "development",
        "namespace": "dev",
        "protocol_sha256": payload["training"]["record_bindings"][0]["protocol_sha256"],
    }
    assert payload["training"]["record_count"] == 4
    assert payload["training"]["entry_count"] == 16
    assert len(payload["training"]["record_bindings"]) == 4
    assert payload["training"]["manifest_sha256s"] == sorted(
        record.manifest_sha256 for record in records
    )
    assert len(payload["training"]["record_digest_sha256"]) == 64
    assert len(payload["payload_sha256"]) == 64
    transformed = first.transform(np.array([0.0, 0.2, 1.0]))
    assert transformed.shape == (3,)
    assert np.all(np.isfinite(transformed))


def test_artifact_canonical_save_load_roundtrip_and_tamper_rejection(tmp_path: Path):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
        save_calibration_artifact,
    )

    artifact = fit_development_calibration(_development_records())
    output = tmp_path / "calibration.json"

    save_calibration_artifact(output, artifact)
    loaded = load_calibration_artifact(output)

    assert loaded.to_dict() == artifact.to_dict()
    assert output.read_bytes().endswith(b"\n")
    assert output.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        save_calibration_artifact(output, artifact)

    tampered = artifact.to_dict()
    tampered["selected_algorithm"] = "beta"
    output.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(ValueError, match="digest|selected|canonical"):
        load_calibration_artifact(output)


def test_artifact_loader_rejects_noncanonical_duplicate_extra_and_nonfinite_json(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
        save_calibration_artifact,
    )

    artifact = fit_development_calibration(_development_records())
    canonical = tmp_path / "canonical.json"
    save_calibration_artifact(canonical, artifact)
    canonical_text = canonical.read_text()

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        canonical_text.replace(
            '{"artifact_type"', '{"schema_version":1,"artifact_type"', 1
        )
    )
    extra = tmp_path / "extra.json"
    extra_payload = artifact.to_dict()
    extra_payload["unexpected"] = True
    extra.write_text(
        json.dumps(extra_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text(
        canonical_text.replace('"record_count":4', '"record_count":NaN')
    )

    for path in (noncanonical, duplicate, extra, nonfinite):
        with pytest.raises(ValueError):
            load_calibration_artifact(path)


def test_artifact_loader_rejects_boolean_schema_even_with_recomputed_digest(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    payload["schema_version"] = True
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    canonical_unsigned = (
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical_unsigned).hexdigest()
    path = tmp_path / "boolean-schema.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="schema"):
        load_calibration_artifact(path)


def test_artifact_loader_recomputes_retention_semantics_after_valid_rehash(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    logistic = next(
        item
        for item in payload["selection"]["candidates"]
        if item["algorithm"] == "logistic"
    )
    logistic["eligible"] = True
    logistic["eligibility_reasons"] = []
    logistic["brier_improved_mechanisms"] = ["symsim"]
    payload["selected_algorithm"] = "logistic"
    payload["calibrator"] = {
        "algorithm": "logistic",
        "parameters": {"intercept": 0.0, "slope": 1.0},
    }
    payload["selection"]["decision_reason"] = (
        "nonidentity_passed_prespecified_retention_gate"
    )
    _rehash_payload(payload)
    path = tmp_path / "forged-retention.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="retention|eligib|mechanism"):
        load_calibration_artifact(path)


def test_artifact_loader_rejects_cross_validation_mechanism_outside_truth_scope(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    for group in payload["cross_validation"]["groups"]:
        group["mechanism"] = "sergio"
    _rehash_payload(payload)
    path = tmp_path / "forged-mechanism.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="mechanism|truth|eligible"):
        load_calibration_artifact(path)


def test_artifact_loader_rejects_rehashed_final_namespace_scope(tmp_path: Path):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    payload["data_scope"]["namespace"] = "final"
    _rehash_payload(payload)
    path = tmp_path / "forged-final-scope.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="development|scope|namespace"):
        load_calibration_artifact(path)


@pytest.mark.parametrize("restriction", ["draw-01-only", "moderate-only"])
def test_artifact_loader_requires_complete_development_draw_view_panel(
    tmp_path: Path,
    restriction: str,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    bindings = payload["training"]["record_bindings"]
    if restriction == "draw-01-only":
        bindings = [item for item in bindings if item["biological_id"] == "draw-01"]
    else:
        bindings = [item for item in bindings if item["technical_view"] == "moderate"]
    manifests = sorted(item["manifest_sha256"] for item in bindings)
    groups = []
    for group in payload["cross_validation"]["groups"]:
        retained = [
            manifest for manifest in group["manifest_sha256s"] if manifest in manifests
        ]
        if retained:
            groups.append({**group, "manifest_sha256s": retained})
    payload["training"]["record_bindings"] = bindings
    payload["training"]["manifest_sha256s"] = manifests
    payload["training"]["record_count"] = len(bindings)
    payload["training"]["entry_count"] = 4 * len(bindings)
    payload["cross_validation"]["groups"] = groups
    _rehash_payload(payload)
    path = tmp_path / f"incomplete-{restriction}.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="draw|view|complete|record|scope"):
        load_calibration_artifact(path)


def test_artifact_loader_cross_checks_entry_count_against_candidate_metrics(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    payload["training"]["entry_count"] = 1
    _rehash_payload(payload)
    path = tmp_path / "false-entry-count.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="entry|metric|count"):
        load_calibration_artifact(path)


def test_artifact_loader_cross_checks_aggregate_and_mechanism_metric_counts(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    payload["selection"]["candidates"][0]["aggregate_metrics"]["n"] += 1
    payload["training"]["entry_count"] += 1
    _rehash_payload(payload)
    path = tmp_path / "contradictory-metric-counts.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="aggregate.*count|mechanism.*metric"):
        load_calibration_artifact(path)


def test_artifact_loader_rejects_reused_dataset_content_across_bindings(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    shared_digest = payload["training"]["record_bindings"][0]["dataset_sha256"]
    for binding in payload["training"]["record_bindings"]:
        binding["dataset_sha256"] = shared_digest
    _rehash_payload(payload)
    path = tmp_path / "reused-dataset-content.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="duplicate.*dataset.*sha256|dataset.*digest"):
        load_calibration_artifact(path)


def _training_input_payload():
    return {
        "schema_version": 2,
        "artifact_type": "maskimpute_prezero_calibration_training_records",
        "records": [
            {
                "p_pre_zero": list(record.p_pre_zero),
                "target": list(record.target),
                "mechanism": record.mechanism,
                "biological_id": record.biological_id,
                "manifest_sha256": record.manifest_sha256,
                "truth_kind": record.truth_kind,
                "namespace": record.namespace,
                "data_role": record.data_role,
                "technical_view": record.technical_view,
                "dataset_id": record.dataset_id,
                "dataset_sha256": record.dataset_sha256,
                "protocol_sha256": record.protocol_sha256,
            }
            for record in _development_records()
        ],
    }


def _canonical_write(path: Path, payload) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _rehash_payload(payload) -> None:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    payload["payload_sha256"] = hashlib.sha256(
        (
            json.dumps(
                unsigned,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _run_calibration_cli(input_path: Path, output_path: Path):
    return subprocess.run(
        [
            sys.executable,
            "scripts/fit_prezero_calibration.py",
            str(input_path),
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_calibration_cli_writes_atomic_canonical_artifact_and_refuses_overwrite(
    tmp_path: Path,
):
    from maskimpute.calibration import load_calibration_artifact

    input_path = tmp_path / "records.json"
    output_path = tmp_path / "calibration.json"
    _canonical_write(input_path, _training_input_payload())

    first = _run_calibration_cli(input_path, output_path)

    assert first.returncode == 0, first.stderr
    artifact = load_calibration_artifact(output_path)
    assert artifact.selected_algorithm == "identity"
    assert json.loads(first.stdout)["selected_algorithm"] == "identity"
    original = output_path.read_bytes()

    second = _run_calibration_cli(input_path, output_path)
    assert second.returncode == 2
    assert json.loads(second.stderr)["error"]
    assert output_path.read_bytes() == original
    assert not list(tmp_path.glob(".calibration.json.*.tmp"))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["records"][0].update({"reconstruction": [1, 2]}),
        lambda payload: payload["records"].append(dict(payload["records"][0])),
        lambda payload: payload["records"][0].update(
            {"truth_kind": "exact_continuous"}
        ),
    ],
)
def test_calibration_cli_rejects_extra_duplicate_or_ineligible_training_input(
    tmp_path: Path,
    mutator,
):
    input_path = tmp_path / "records.json"
    output_path = tmp_path / "calibration.json"
    payload = _training_input_payload()
    mutator(payload)
    _canonical_write(input_path, payload)

    result = _run_calibration_cli(input_path, output_path)

    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]
    assert not output_path.exists()


def test_calibration_cli_rejects_noncanonical_and_duplicate_key_json(tmp_path: Path):
    output_path = tmp_path / "calibration.json"
    payload = _training_input_payload()
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(payload, indent=2) + "\n")
    duplicate_key = tmp_path / "duplicate-key.json"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    duplicate_key.write_text(
        canonical.replace('{"artifact_type"', '{"schema_version":1,"artifact_type"', 1)
        + "\n"
    )

    for input_path in (noncanonical, duplicate_key):
        result = _run_calibration_cli(input_path, output_path)
        assert result.returncode == 2
        assert json.loads(result.stderr)["error"]
        assert not output_path.exists()


def test_atomic_artifact_failure_leaves_no_output_or_temporary_file(
    tmp_path: Path,
    monkeypatch,
):
    import maskimpute.calibration as module

    artifact = module.fit_development_calibration(_development_records())
    output = tmp_path / "calibration.json"

    def fail_link(*args, **kwargs):
        raise OSError("injected link failure")

    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        module.save_calibration_artifact(output, artifact)

    assert not output.exists()
    assert not list(tmp_path.glob(".calibration.json.*.tmp"))


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_calibration_loaders_reject_linked_input_files(tmp_path: Path, link_kind):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
        load_calibration_records,
        save_calibration_artifact,
    )

    records_source = tmp_path / "records-source.json"
    artifact_source = tmp_path / "artifact-source.json"
    _canonical_write(records_source, _training_input_payload())
    save_calibration_artifact(
        artifact_source,
        fit_development_calibration(_development_records()),
    )
    records_link = tmp_path / "records-link.json"
    artifact_link = tmp_path / "artifact-link.json"
    if link_kind == "symlink":
        records_link.symlink_to(records_source)
        artifact_link.symlink_to(artifact_source)
    else:
        records_link.hardlink_to(records_source)
        artifact_link.hardlink_to(artifact_source)

    with pytest.raises(ValueError, match="link|regular|unique"):
        load_calibration_records(records_link)
    with pytest.raises(ValueError, match="link|regular|unique"):
        load_calibration_artifact(artifact_link)


def test_calibration_save_rejects_symlinked_output_parent(tmp_path: Path):
    from maskimpute.calibration import (
        fit_development_calibration,
        save_calibration_artifact,
    )

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output = linked_parent / "calibration.json"

    with pytest.raises((OSError, ValueError), match="symlink|link"):
        save_calibration_artifact(
            output,
            fit_development_calibration(_development_records()),
        )
    assert not (real_parent / "calibration.json").exists()


@pytest.mark.parametrize("failure_call", [2, 3])
def test_directory_fsync_failure_rolls_back_published_artifact(
    tmp_path: Path,
    monkeypatch,
    failure_call: int,
):
    import maskimpute.calibration as module

    artifact = module.fit_development_calibration(_development_records())
    output = tmp_path / "calibration.json"
    real_fsync = module.os.fsync
    calls = 0

    def fail_directory_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("injected directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync"):
        module.save_calibration_artifact(output, artifact)

    assert not output.exists()
    assert not list(tmp_path.glob(".calibration.json.*.tmp"))


def test_temporary_unlink_failure_rolls_back_instead_of_reporting_success(
    tmp_path: Path,
    monkeypatch,
):
    import maskimpute.calibration as module

    artifact = module.fit_development_calibration(_development_records())
    output = tmp_path / "calibration.json"
    real_unlink = module.os.unlink
    injected = False

    def fail_first_temporary_unlink(path, *args, **kwargs):
        nonlocal injected
        if not injected and str(path).startswith(".calibration.json."):
            injected = True
            raise OSError("injected temporary unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "unlink", fail_first_temporary_unlink)

    with pytest.raises(OSError, match="temporary unlink"):
        module.save_calibration_artifact(output, artifact)

    assert injected is True
    assert not output.exists()
    assert not list(tmp_path.glob(".calibration.json.*.tmp"))
