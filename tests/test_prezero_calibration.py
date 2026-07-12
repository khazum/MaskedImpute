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


def test_calibration_amendment_contract_is_tracked_fixed_and_pre_final():
    from maskimpute.calibration import CALIBRATION_CONTRACT_SHA256

    path = Path("study/calibration_contract.json")
    contract_bytes = path.read_bytes()
    contract = json.loads(contract_bytes)

    assert hashlib.sha256(contract_bytes).hexdigest() == CALIBRATION_CONTRACT_SHA256
    assert contract["schema_version"] == 1
    assert contract["artifact_schema_version"] == 3
    assert contract["status"] == "adopted"
    assert contract["timing"] == {
        "adopted_before": "final_seed_execution",
        "data_scope": "development_only",
        "final_data_used": False,
    }
    assert contract["truth_scope"]["eligible_exact_mechanisms"] == ["symsim"]
    assert contract["truth_scope"]["proxy_truth_relabelled"] is False
    assert contract["truth_scope"]["panel_limitations"] == {
        "semisynthetic": "proxy_truth_not_exact",
        "sergio": "undefined_for_continuous_truth",
        "sparsim": "undefined_for_continuous_truth",
    }
    assert contract["cross_validation"] == {
        "development_inference": "held_out_fold_calibrator_only",
        "final_inference": "all_development_fitted_calibrator",
        "independent_unit": "biological_draw",
        "nested_technical_unit": "draw_technical_view_record",
        "scheme": "leave_one_mechanism_biological_draw_out",
    }
    assert contract["retention_rules"] == {
        "brier_improvement_epsilon": 1e-6,
        "calibration_slope_gated_levels": [
            "aggregate",
            "mechanism",
            "biological_draw",
        ],
        "calibration_slope_lower": 0.8,
        "calibration_slope_upper": 1.2,
        "log_loss_gated_levels": [
            "aggregate",
            "mechanism",
            "biological_draw",
            "technical_record",
        ],
        "log_loss_worsening_tolerance": 1e-3,
        "minimum_biological_draws_improved": 2,
        "minimum_exact_mechanisms_improved": 1,
        "minimum_technical_records_improved": 4,
        "require_all_biological_draws_improved": True,
        "require_all_eligible_exact_mechanisms_improved": True,
        "require_all_technical_records_improved": True,
        "require_no_fit_failures": True,
        "technical_record_slope_policy": (
            "reported_not_gated_nested_technical_observation"
        ),
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
    biological_draw_metrics=None,
    technical_record_metrics=None,
    failures=(),
    improved_mechanisms=(),
    improved_draws=(),
    improved_records=(),
    eligible=False,
    reasons=(),
):
    from maskimpute.calibration import CandidateEvaluation

    biological_draw_metrics = biological_draw_metrics or {
        "symsim/draw-01": _metrics(),
        "symsim/draw-02": _metrics(),
    }
    technical_record_metrics = technical_record_metrics or {
        "symsim/draw-01/moderate": _metrics(),
        "symsim/draw-01/severe": _metrics(),
        "symsim/draw-02/moderate": _metrics(),
        "symsim/draw-02/severe": _metrics(),
    }
    return CandidateEvaluation(
        algorithm=algorithm,
        mechanism_metrics=tuple(sorted(mechanism_metrics.items())),
        biological_draw_metrics=tuple(sorted(biological_draw_metrics.items())),
        technical_record_metrics=tuple(sorted(technical_record_metrics.items())),
        aggregate_metrics=aggregate or _metrics(),
        fit_failures=tuple(failures),
        brier_improved_mechanisms=tuple(improved_mechanisms),
        brier_improved_biological_draws=tuple(improved_draws),
        brier_improved_technical_records=tuple(improved_records),
        eligible=eligible,
        eligibility_reasons=tuple(reasons),
    )


def test_retention_gate_requires_every_exact_mechanism_draw_and_technical_record():
    from maskimpute.calibration import (
        CalibrationThresholds,
        retention_reasons,
    )

    mechanisms = {"symsim": _metrics()}
    draws = {
        "symsim/draw-01": _metrics(),
        "symsim/draw-02": _metrics(),
    }
    records = {
        "symsim/draw-01/moderate": _metrics(),
        "symsim/draw-01/severe": _metrics(),
        "symsim/draw-02/moderate": _metrics(),
        "symsim/draw-02/severe": _metrics(),
    }
    identity = _evaluation(
        "identity",
        mechanisms,
        biological_draw_metrics=draws,
        technical_record_metrics=records,
    )
    candidate = _evaluation(
        "logistic",
        {"symsim": _metrics(brier=0.19)},
        biological_draw_metrics={
            "symsim/draw-01": _metrics(brier=0.19),
            "symsim/draw-02": _metrics(brier=0.21),
        },
        technical_record_metrics={
            **{
                name: _metrics(brier=0.19)
                for name in tuple(records)[:-1]
            },
            "symsim/draw-02/severe": _metrics(brier=0.21, log_loss=0.5011),
        },
        aggregate=_metrics(brier=0.195, log_loss=0.5011),
        failures=("symsim/draw-02:RuntimeError:fit failed",),
    )

    reasons, improved_mechanisms, improved_draws, improved_records = retention_reasons(
        candidate,
        identity,
        CalibrationThresholds(),
    )

    assert improved_mechanisms == ("symsim",)
    assert improved_draws == ("symsim/draw-01",)
    assert improved_records == tuple(records)[:-1]
    assert "insufficient_biological_draw_brier_improvement:1<2" in reasons
    assert "not_all_biological_draws_improved:symsim/draw-02" in reasons
    assert "insufficient_technical_record_brier_improvement:3<4" in reasons
    assert "not_all_technical_records_improved:symsim/draw-02/severe" in reasons
    assert "aggregate_log_loss_worsened" in reasons
    assert (
        "fold_fit_failure:symsim/draw-02:RuntimeError:fit failed" in reasons
    )
    assert (
        "technical_record_log_loss_worsened:symsim/draw-02/severe" in reasons
    )


def test_retention_gate_passes_consistent_two_draw_evidence_and_does_not_gate_record_slope():
    from maskimpute.calibration import CalibrationThresholds, retention_reasons

    records = {
        "symsim/draw-01/moderate": _metrics(),
        "symsim/draw-01/severe": _metrics(),
        "symsim/draw-02/moderate": _metrics(),
        "symsim/draw-02/severe": _metrics(),
    }
    identity = _evaluation("identity", {"symsim": _metrics()})
    candidate = _evaluation(
        "beta",
        {"symsim": _metrics(brier=0.19, log_loss=0.5005)},
        biological_draw_metrics={
            "symsim/draw-01": _metrics(brier=0.19, log_loss=0.5005, slope=0.8),
            "symsim/draw-02": _metrics(brier=0.19, slope=1.2),
        },
        technical_record_metrics={
            name: _metrics(
                brier=0.19,
                log_loss=0.5005,
                slope=9.0 if name.endswith("severe") else 1.0,
            )
            for name in records
        },
        aggregate=_metrics(brier=0.19, log_loss=0.5005),
    )

    reasons, improved_mechanisms, improved_draws, improved_records = retention_reasons(
        candidate,
        identity,
        CalibrationThresholds(),
    )

    assert reasons == ()
    assert improved_mechanisms == ("symsim",)
    assert improved_draws == ("symsim/draw-01", "symsim/draw-02")
    assert improved_records == tuple(records)


@pytest.mark.parametrize(
    "override",
    [
        {"minimum_exact_mechanisms_improved": 2},
        {"minimum_biological_draws_improved": 1},
        {"minimum_biological_draws_improved": 3},
        {"minimum_technical_records_improved": 3},
        {"minimum_technical_records_improved": 5},
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
    object.__setattr__(mutated, "minimum_biological_draws_improved", 1)

    class LooseThresholds:
        minimum_exact_mechanisms_improved = 1
        minimum_biological_draws_improved = 1
        minimum_technical_records_improved = 1
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
        assert candidate.eligible is False
        assert candidate.eligibility_reasons
        assert not any(
            "eligible_exact_truth_mechanisms:1<3" in reason
            for reason in candidate.eligibility_reasons
        )


def _development_records():
    return (
        _record("symsim", "draw-01", "1", technical_view="moderate"),
        _record("symsim", "draw-01", "2", technical_view="severe"),
        _record("symsim", "draw-02", "3", technical_view="moderate"),
        _record("symsim", "draw-02", "4", technical_view="severe"),
    )


def _consistent_two_draw_retention_records():
    levels = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.96])

    def draw_values(intercept):
        calibrated = expit(
            0.5 * np.log(levels) - 1.7 * np.log1p(-levels) + intercept
        )
        probabilities = []
        targets = []
        for probability, calibrated_probability in zip(
            levels,
            calibrated,
            strict=True,
        ):
            positives = min(49, max(1, round(50 * calibrated_probability)))
            probabilities.extend([float(probability)] * 50)
            targets.extend([1] * positives + [0] * (50 - positives))
        return probabilities, targets

    draw_01_probability, draw_01_target = draw_values(-1.1)
    draw_02_probability, draw_02_target = draw_values(-0.9)
    return (
        _record(
            "symsim",
            "draw-01",
            "5",
            probabilities=draw_01_probability,
            targets=draw_01_target,
            technical_view="moderate",
        ),
        _record(
            "symsim",
            "draw-01",
            "6",
            probabilities=draw_01_probability,
            targets=draw_01_target,
            technical_view="severe",
        ),
        _record(
            "symsim",
            "draw-02",
            "7",
            probabilities=draw_02_probability,
            targets=draw_02_target,
            technical_view="moderate",
        ),
        _record(
            "symsim",
            "draw-02",
            "8",
            probabilities=draw_02_probability,
            targets=draw_02_target,
            technical_view="severe",
        ),
    )


def test_real_shaped_two_draw_fixture_retains_beta_and_isotonic_but_rejects_logistic():
    from maskimpute.calibration import evaluate_calibration_candidates

    decision = evaluate_calibration_candidates(_consistent_two_draw_retention_records())
    candidates = {candidate.algorithm: candidate for candidate in decision.candidates}

    assert candidates["logistic"].eligible is False
    assert any(
        "not_all_biological_draws_improved" in reason
        for reason in candidates["logistic"].eligibility_reasons
    )
    for algorithm in ("beta", "isotonic"):
        candidate = candidates[algorithm]
        assert candidate.eligible is True
        assert candidate.eligibility_reasons == ()
        assert candidate.brier_improved_mechanisms == ("symsim",)
        assert candidate.brier_improved_biological_draws == (
            "symsim/draw-01",
            "symsim/draw-02",
        )
        assert candidate.brier_improved_technical_records == (
            "symsim/draw-01/moderate",
            "symsim/draw-01/severe",
            "symsim/draw-02/moderate",
            "symsim/draw-02/severe",
        )
    assert decision.selected_algorithm == "isotonic"


def test_schema3_artifact_retains_exact_lodo_calibrators_for_development_use():
    from maskimpute.calibration import fit_development_calibration

    records = _consistent_two_draw_retention_records()
    artifact = fit_development_calibration(records)
    payload = artifact.to_dict()

    assert artifact.selected_algorithm == "isotonic"
    folds = payload["development_holdout_calibrators"]
    assert [(fold["mechanism"], fold["biological_id"]) for fold in folds] == [
        ("symsim", "draw-01"),
        ("symsim", "draw-02"),
    ]
    all_manifests = {record.manifest_sha256 for record in records}
    for fold in folds:
        held_out = set(fold["held_out_manifest_sha256s"])
        training = set(fold["training_manifest_sha256s"])
        assert held_out
        assert held_out.isdisjoint(training)
        assert held_out | training == all_manifests
        assert fold["calibrator"]["algorithm"] == artifact.selected_algorithm

    sample = np.array([0.03, 0.4, 0.91])
    transformed = artifact.transform_for_development_holdout(
        sample,
        mechanism="symsim",
        biological_id="draw-01",
    )
    fold = folds[0]["calibrator"]["parameters"]
    np.testing.assert_allclose(
        transformed,
        np.interp(sample, fold["knots"], fold["values"]),
    )
    assert not np.array_equal(transformed, artifact.transform(sample))
    assert not np.array_equal(
        transformed,
        artifact.transform_for_development_holdout(
            sample,
            mechanism="symsim",
            biological_id="draw-02",
        ),
    )
    with pytest.raises(ValueError, match="mechanism|holdout"):
        artifact.transform_for_development_holdout(
            sample,
            mechanism="sergio",
            biological_id="draw-01",
        )
    with pytest.raises(ValueError, match="biological|holdout"):
        artifact.transform_for_development_holdout(
            sample,
            mechanism="symsim",
            biological_id="draw-03",
        )


def test_fitted_artifact_is_deterministic_complete_and_score_only():
    from maskimpute.calibration import (
        CALIBRATION_CONTRACT_SHA256,
        fit_development_calibration,
    )

    records = _development_records()

    first = fit_development_calibration(records)
    second = fit_development_calibration(tuple(reversed(records)))

    assert first.to_dict() == second.to_dict()
    payload = first.to_dict()
    assert payload["schema_version"] == 3
    assert payload["artifact_type"] == "maskimpute_prezero_calibration"
    assert payload["inference_features"] == ["p_pre_zero"]
    assert payload["selected_algorithm"] == payload["calibrator"]["algorithm"]
    assert payload["retention_contract"] == {
        "contract_id": "prezero-calibration-retention-development-amendment-v1",
        "path": "study/calibration_contract.json",
        "sha256": CALIBRATION_CONTRACT_SHA256,
    }
    assert payload["truth_eligibility"] == {
        "accepted_truth_kind": "exact_pre_capture",
        "eligible_mechanisms": ["symsim"],
        "eligible_mechanism_count": 1,
        "minimum_biological_draws_improved": 2,
        "minimum_exact_mechanisms_improved": 1,
        "minimum_technical_records_improved": 4,
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
    for candidate in payload["selection"]["candidates"]:
        assert tuple(candidate["biological_draw_metrics"]) == (
            "symsim/draw-01",
            "symsim/draw-02",
        )
        assert tuple(candidate["technical_record_metrics"]) == (
            "symsim/draw-01/moderate",
            "symsim/draw-01/severe",
            "symsim/draw-02/moderate",
            "symsim/draw-02/severe",
        )
        assert "brier_improved_biological_draws" in candidate
        assert "brier_improved_technical_records" in candidate
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


def test_artifact_loader_rejects_lodo_fold_that_trains_on_held_out_truth(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(
        _consistent_two_draw_retention_records()
    ).to_dict()
    fold = payload["development_holdout_calibrators"][0]
    fold["training_manifest_sha256s"].append(fold["held_out_manifest_sha256s"][0])
    fold["training_manifest_sha256s"].sort()
    _rehash_payload(payload)
    path = tmp_path / "leaky-fold.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="held|disjoint|training|fold"):
        load_calibration_artifact(path)


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


def test_artifact_loader_rejects_obsolete_schema2_after_amendment(tmp_path: Path):
    from maskimpute.calibration import load_calibration_artifact

    payload = {
        "schema_version": 2,
        "artifact_type": "maskimpute_prezero_calibration",
    }
    path = tmp_path / "obsolete-schema2.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="schema 2.*obsolete|obsolete.*schema 2"):
        load_calibration_artifact(path)


def test_artifact_loader_rejects_rehashed_amendment_contract_substitution(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    payload["retention_contract"]["sha256"] = "0" * 64
    _rehash_payload(payload)
    path = tmp_path / "substituted-amendment.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="amendment|contract"):
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


def test_artifact_loader_rejects_rehashed_cross_level_metric_contradiction(
    tmp_path: Path,
):
    from maskimpute.calibration import (
        fit_development_calibration,
        load_calibration_artifact,
    )

    payload = fit_development_calibration(_development_records()).to_dict()
    record_id = "symsim/draw-01/moderate"
    for candidate in payload["selection"]["candidates"]:
        candidate["technical_record_metrics"][record_id]["brier"] += 0.01
    _rehash_payload(payload)
    path = tmp_path / "contradictory-level-metrics.json"
    _canonical_write(path, payload)

    with pytest.raises(ValueError, match="metric|weighted|contradict"):
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
