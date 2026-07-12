from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest


MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
DRAWS = ("draw-01", "draw-02")
VIEWS = ("moderate", "severe")
METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
    "null_de_fpr",
)
ENDPOINTS = (
    "rna_protein_concordance",
    "ercc_recovery",
    "technical_replicate_concordance",
    "bulk_pseudobulk_concordance",
)


def _declarations(*candidate_ids: str):
    from maskimpute_benchmark.selection import MethodDeclaration

    declarations = [
        MethodDeclaration(
            id="observed",
            role="observed_control",
            track="same_input",
            stochastic=False,
            required_for_claim=True,
        ),
        MethodDeclaration(
            id="strong",
            role="learned_comparator",
            track="same_input",
            stochastic=True,
            required_for_claim=True,
        ),
        MethodDeclaration(
            id="weak",
            role="learned_comparator",
            track="same_input",
            stochastic=True,
            required_for_claim=False,
        ),
    ]
    declarations.extend(
        MethodDeclaration(
            id=candidate_id,
            role="candidate",
            track="same_input",
            stochastic=True,
            required_for_claim=True,
        )
        for candidate_id in candidate_ids
    )
    return tuple(declarations)


def _attempts(*items: tuple[str, str]):
    from maskimpute_benchmark.selection import CandidateAttempt

    prior: dict[str, str] = {}
    attempts = []
    for configuration, version in items:
        parent = None
        if version == "v28":
            parent = prior.get("v27")
        elif version == "v29":
            parent = prior.get("v28")
        attempts.append(
            CandidateAttempt(
                configuration_id=configuration,
                version=version,
                parent_configuration_id=parent,
            )
        )
        prior[version] = configuration
    return tuple(attempts)


def _base_values():
    return {
        "observed": {
            "mse": 1.40,
            "mse_dropout": 1.40,
            "gnrmse": 1.40,
            "mse_pre_dropout_zero": 1.00,
            "corr_err": 1.00,
            "mse_non_dropout_nonzero": 1.00,
            "null_de_fpr": 0.045,
        },
        "strong": {
            "mse": 1.00,
            "mse_dropout": 1.00,
            "gnrmse": 1.00,
            "mse_pre_dropout_zero": 1.00,
            "corr_err": 1.00,
            "mse_non_dropout_nonzero": 1.00,
            "null_de_fpr": 0.052,
        },
        "weak": {
            "mse": 1.20,
            "mse_dropout": 1.20,
            "gnrmse": 1.20,
            "mse_pre_dropout_zero": 1.20,
            "corr_err": 1.20,
            "mse_non_dropout_nonzero": 1.20,
            "null_de_fpr": 0.055,
        },
        "v27-a": {
            "mse": 0.80,
            "mse_dropout": 0.80,
            "gnrmse": 0.80,
            "mse_pre_dropout_zero": 1.05,
            "corr_err": 1.05,
            "mse_non_dropout_nonzero": 0.90,
            "null_de_fpr": 0.050,
        },
    }


def _records(values, declarations, *, fail=(), omit=()):
    stochastic = {item.id: item.stochastic for item in declarations}
    failed = set(fail)
    omitted = set(omit)
    records = []
    for method, metric_values in values.items():
        seeds = (42, 43, 44) if stochastic[method] else (None,)
        for metric in METRICS:
            mechanisms = ("symsim",) if metric == "mse_pre_dropout_zero" else MECHANISMS
            for mechanism in mechanisms:
                for draw in DRAWS:
                    for view in VIEWS:
                        for seed in seeds:
                            identity = (method, metric, mechanism, draw, view, seed)
                            if identity in omitted:
                                continue
                            is_failed = identity in failed
                            records.append(
                                {
                                    "mechanism": mechanism,
                                    "biological_id": draw,
                                    "technical_view": view,
                                    "dataset_id": f"{mechanism}-{draw}-{view}",
                                    "dataset_sha256": hashlib.sha256(
                                        f"{mechanism}-{draw}-{view}".encode()
                                    ).hexdigest(),
                                    "method": method,
                                    "method_sha256": hashlib.sha256(
                                        method.encode()
                                    ).hexdigest(),
                                    "model_seed": seed,
                                    "metric": metric,
                                    "value": None
                                    if is_failed
                                    else metric_values[metric],
                                    "status": "failed" if is_failed else "completed",
                                }
                            )
    return records


def _intervals(*candidate_ids: str, unsafe: set[tuple[str, str]] | None = None):
    unsafe = unsafe or set()
    return [
        {
            "configuration": candidate,
            "endpoint": endpoint,
            "comparison": "observed",
            "estimate": 0.0 if (candidate, endpoint) not in unsafe else -0.08,
            "ci_lower": -0.01 if (candidate, endpoint) not in unsafe else -0.11,
            "ci_upper": 0.01 if (candidate, endpoint) not in unsafe else -0.05,
            "status": "completed",
        }
        for candidate in candidate_ids
        for endpoint in ENDPOINTS
    ]


def _select(
    values=None,
    *,
    attempts=None,
    declarations=None,
    records=None,
    intervals=None,
    exclusions=(),
):
    from maskimpute_benchmark.selection import _evaluate_development_candidates
    from maskimpute_benchmark.selection import EndpointPolicy
    from maskimpute_benchmark.selection import RevisionPolicy

    values = _base_values() if values is None else values
    attempts = _attempts(("v27-a", "v27")) if attempts is None else attempts
    declarations = (
        _declarations(*(item.configuration_id for item in attempts))
        if declarations is None
        else declarations
    )
    records = _records(values, declarations) if records is None else records
    intervals = (
        _intervals(*(item.configuration_id for item in attempts))
        if intervals is None
        else intervals
    )
    dataset_bindings = {
        (mechanism, draw, view): (
            f"{mechanism}-{draw}-{view}",
            hashlib.sha256(f"{mechanism}-{draw}-{view}".encode()).hexdigest(),
        )
        for mechanism in MECHANISMS
        for draw in DRAWS
        for view in VIEWS
    }
    return _evaluate_development_candidates(
        records,
        attempts,
        declarations,
        intervals,
        mechanisms=MECHANISMS,
        biological_ids=DRAWS,
        technical_views=VIEWS,
        model_seeds=(42, 43, 44),
        required_orthogonal_endpoints=ENDPOINTS,
        dataset_bindings=dataset_bindings,
        method_bindings={
            declaration.id: hashlib.sha256(declaration.id.encode()).hexdigest()
            for declaration in declarations
        },
        endpoint_policies=tuple(
            EndpointPolicy(
                id=endpoint,
                comparison="candidate_minus_observed",
                favorable_direction="higher",
                materiality_margin=0.02,
            )
            for endpoint in ENDPOINTS
        ),
        revision_policy=RevisionPolicy(v29_max_dropout_mse_loss=0.02),
        exclusions=exclusions,
    )


def test_candidate_passes_each_prespecified_gate_without_a_combined_score():
    report = _select()

    assert report.selected_configuration == "v27-a"
    assert report.pareto_set == ("v27-a",)
    assert report.trigger == "freeze_candidate"
    assessment = report.by_configuration["v27-a"]
    assert assessment.eligible is True
    assert assessment.efficacy_pass is True
    assert assessment.safety_pass is True
    assert assessment.ineligibility_reasons == ()
    assert set(assessment.gates) == {
        "rank_mse",
        "rank_mse_dropout",
        "rank_gnrmse",
        "pareto_non_dominated",
        "dropout_improvement",
        "prezero_degradation",
        "corr_err_degradation",
        "null_de_safety",
        "orthogonal_safety",
        "candidate_completeness",
        "required_comparator_completeness",
        "revision_retention",
    }
    assert assessment.gates["rank_mse"].value == pytest.approx(1.0)
    assert dict(assessment.gates["rank_mse"].details["mechanism_ranks"]) == {
        mechanism: pytest.approx(1.0) for mechanism in MECHANISMS
    }
    assert set(
        dict(assessment.gates["pareto_non_dominated"].details["mechanism_results"])
    ) == set(MECHANISMS)
    assert assessment.gates["dropout_improvement"].value == 4
    assert assessment.gates["prezero_degradation"].value == pytest.approx(0.05)
    assert assessment.gates["corr_err_degradation"].value == pytest.approx(0.05)
    assert assessment.gates["null_de_safety"].details["maximum_fpr"] == pytest.approx(
        0.05
    )
    assert assessment.gates["null_de_safety"].details[
        "maximum_above_observed"
    ] == pytest.approx(0.005)
    assert not hasattr(assessment, "combined_score")
    assert not hasattr(report, "combined_score")


def test_seed_and_view_rows_are_nested_before_draw_level_ranking():
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    baseline = _select(declarations=declarations, records=records)
    duplicated = [*records, *copy.deepcopy(records), *copy.deepcopy(records[:37])]
    shuffled = copy.deepcopy(duplicated)
    random.Random(917).shuffle(shuffled)

    replay = _select(declarations=declarations, records=shuffled)

    assert replay.to_dict() == baseline.to_dict()
    assert replay.by_configuration["v27-a"].independent_draws == 8


def test_missing_candidate_row_is_retained_as_an_ineligible_configuration():
    declarations = _declarations("v27-a")
    omitted = {("v27-a", "mse", "symsim", "draw-01", "moderate", 42)}
    records = _records(_base_values(), declarations, omit=omitted)

    report = _select(declarations=declarations, records=records)

    assessment = report.by_configuration["v27-a"]
    assert assessment.eligible is False
    assert assessment.gates["candidate_completeness"].passed is False
    assert "incomplete_candidate_metrics" in assessment.ineligibility_reasons
    assert report.selected_configuration is None
    assert report.trigger == "v28"


def test_required_comparator_failure_blocks_an_unqualified_competitive_claim():
    declarations = _declarations("v27-a")
    failure = {("strong", "mse", "sergio", "draw-02", "severe", 43)}
    records = _records(_base_values(), declarations, fail=failure)

    report = _select(declarations=declarations, records=records)

    assessment = report.by_configuration["v27-a"]
    gate = assessment.gates["required_comparator_completeness"]
    assert gate.passed is False
    assert "required_comparator_incomplete:strong" in assessment.ineligibility_reasons


def test_conflicting_duplicate_result_identity_fails_closed():
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    conflicting = copy.deepcopy(records[0])
    conflicting["value"] += 0.1
    records.append(conflicting)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        _select(declarations=declarations, records=records)


def test_pareto_gate_uses_all_four_declared_dimensions_and_is_order_independent():
    values = _base_values()
    values["v27-a"].update(
        mse_dropout=1.10,
        mse_pre_dropout_zero=1.10,
        corr_err=1.10,
        mse_non_dropout_nonzero=1.10,
    )
    declarations = _declarations("v27-a")
    records = _records(values, declarations)
    random.Random(3).shuffle(records)

    report = _select(values, declarations=declarations, records=records)

    gate = report.by_configuration["v27-a"].gates["pareto_non_dominated"]
    assert gate.passed is False
    assert "strong" in gate.details["dominated_by"]


def test_null_de_and_orthogonal_safety_are_separate_hard_gates():
    values = _base_values()
    values["v27-a"]["null_de_fpr"] = 0.061
    unsafe = {("v27-a", "rna_protein_concordance")}

    report = _select(values, intervals=_intervals("v27-a", unsafe=unsafe))

    assessment = report.by_configuration["v27-a"]
    assert assessment.gates["null_de_safety"].passed is False
    assert assessment.gates["orthogonal_safety"].passed is False
    assert set(assessment.gates["orthogonal_safety"].details["unsafe_endpoints"]) == {
        "rna_protein_concordance"
    }


def test_revision_trigger_requires_v28_assessment_before_v29():
    unsafe = {("v27-a", endpoint) for endpoint in ENDPOINTS}
    v27_report = _select(intervals=_intervals("v27-a", unsafe=unsafe))
    assert v27_report.by_configuration["v27-a"].efficacy_pass is True
    assert v27_report.by_configuration["v27-a"].safety_pass is False
    assert v27_report.trigger == "v28"

    values = _base_values()
    values["v28-a"] = dict(values["v27-a"])
    attempts = _attempts(("v27-a", "v27"), ("v28-a", "v28"))
    declarations = _declarations("v27-a", "v28-a")
    unsafe = {
        (candidate, endpoint)
        for candidate in ("v27-a", "v28-a")
        for endpoint in ENDPOINTS
    }
    records = _records(values, declarations)
    v28_report = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=records,
        intervals=_intervals("v27-a", "v28-a", unsafe=unsafe),
    )
    assert v28_report.trigger == "v29"


def test_failed_v29_exhausts_revisions_and_downgrades_the_claim():
    values = _base_values()
    values["v28-a"] = dict(values["v27-a"])
    values["v29-a"] = dict(values["v27-a"])
    attempts = _attempts(
        ("v27-a", "v27"),
        ("v28-a", "v28"),
        ("v29-a", "v29"),
    )
    declarations = _declarations("v27-a", "v28-a", "v29-a")
    unsafe = {
        (candidate, endpoint)
        for candidate in ("v27-a", "v28-a", "v29-a")
        for endpoint in ENDPOINTS
    }

    report = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        intervals=_intervals("v27-a", "v28-a", "v29-a", unsafe=unsafe),
    )

    assert report.selected_configuration is None
    assert report.trigger == "downgrade_claim"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(status="completed", value=float("nan")),
        lambda row: row.update(model_seed=True),
        lambda row: row.update(metric="unknown"),
        lambda row: row.update(method="undeclared"),
        lambda row: row.update(extra="hidden"),
    ],
)
def test_selection_records_reject_ambiguous_or_unprespecified_inputs(mutation):
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    mutation(records[0])

    with pytest.raises((TypeError, ValueError)):
        _select(declarations=declarations, records=records)


def test_missing_orthogonal_endpoint_is_not_silently_ignored():
    intervals = _intervals("v27-a")
    intervals = [
        row for row in intervals if row["endpoint"] != "bulk_pseudobulk_concordance"
    ]

    report = _select(intervals=intervals)

    gate = report.by_configuration["v27-a"].gates["orthogonal_safety"]
    assert gate.passed is False
    assert gate.details["missing_endpoints"] == ("bulk_pseudobulk_concordance",)


def test_zero_comparator_metrics_fail_gates_without_nonfinite_report_values():
    values = _base_values()
    for method in ("strong", "weak"):
        values[method]["mse_dropout"] = 0.0
        values[method]["mse_pre_dropout_zero"] = 0.0

    report = _select(values)

    assessment = report.by_configuration["v27-a"]
    assert assessment.gates["dropout_improvement"].passed is False
    assert assessment.gates["prezero_degradation"].passed is False
    json.dumps(report.to_dict(), allow_nan=False)


def test_selection_cli_rejects_caller_supplied_design_authority(tmp_path):
    attempts = _attempts(("v27-a", "v27"))
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    intervals = _intervals("v27-a")
    payload = {
        "schema_version": 1,
        "records": records,
        "attempts": [asdict(item) for item in attempts],
        "declarations": [asdict(item) for item in declarations],
        "orthogonal_intervals": intervals,
        "design": {
            "mechanisms": list(MECHANISMS),
            "biological_ids": list(DRAWS),
            "technical_views": list(VIEWS),
            "model_seeds": [42, 43, 44],
            "required_orthogonal_endpoints": list(ENDPOINTS),
        },
    }
    input_path = tmp_path / "development-results.json"
    output_path = tmp_path / "selection-report.json"
    input_path.write_text(json.dumps(payload, allow_nan=False))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/select_development_candidate.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not output_path.exists()
    assert Path("scripts/select_development_candidate.py").is_file()


def _set_draw_metric(records, method, metric, draw_values):
    for row in records:
        if row["method"] == method and row["metric"] == metric:
            row["value"] = draw_values[row["biological_id"]]


def test_effect_gates_use_median_paired_draw_percentages_not_ratios_of_medians():
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    _set_draw_metric(
        records,
        "strong",
        "mse_dropout",
        {"draw-01": 1.0, "draw-02": 100.0},
    )
    _set_draw_metric(
        records,
        "v27-a",
        "mse_dropout",
        {"draw-01": 1.1, "draw-02": 90.0},
    )
    _set_draw_metric(
        records,
        "weak",
        "mse_dropout",
        {"draw-01": 2.0, "draw-02": 200.0},
    )

    report = _select(declarations=declarations, records=records)

    gate = report.by_configuration["v27-a"].gates["dropout_improvement"]
    assert gate.passed is False
    assert dict(gate.details["mechanism_improvements"])["symsim"] == pytest.approx(0.0)


def test_degradation_gates_use_paired_draw_effects_before_mechanism_summary():
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    for metric in ("mse_pre_dropout_zero", "corr_err"):
        _set_draw_metric(
            records,
            "strong",
            metric,
            {"draw-01": 1.0, "draw-02": 100.0},
        )
        _set_draw_metric(
            records,
            "v27-a",
            metric,
            {"draw-01": 2.0, "draw-02": 105.0},
        )
        _set_draw_metric(
            records,
            "weak",
            metric,
            {"draw-01": 2.0, "draw-02": 200.0},
        )

    report = _select(declarations=declarations, records=records)

    assessment = report.by_configuration["v27-a"]
    assert assessment.gates["prezero_degradation"].passed is False
    assert assessment.gates["corr_err_degradation"].passed is False
    prezero = dict(
        assessment.gates["prezero_degradation"].details["mechanism_degradation"]
    )
    assert prezero["symsim"] == pytest.approx(0.525)


def test_other_candidate_configurations_do_not_artificially_worsen_competitive_ranks():
    candidate_ids = ("v27-a", "v27-b", "v27-c", "v27-d")
    attempts = _attempts(*((candidate, "v27") for candidate in candidate_ids))
    declarations = _declarations(*candidate_ids)
    values = _base_values()
    for candidate in candidate_ids[1:]:
        values[candidate] = dict(values["v27-a"])
    records = _records(values, declarations)

    report = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=records,
        intervals=_intervals(*candidate_ids),
    )

    for candidate in candidate_ids:
        assessment = report.by_configuration[candidate]
        assert assessment.gates["rank_mse"].value == pytest.approx(1.0)
        assert assessment.gates["rank_mse"].passed is True


def test_dataset_id_cannot_be_reused_to_fabricate_independent_units():
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    for row in records:
        row["dataset_id"] = "one-reused-dataset"

    with pytest.raises(ValueError, match="dataset.*manifest|reused"):
        _select(declarations=declarations, records=records)


def test_record_dataset_hash_must_match_the_validated_development_manifest():
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    records[0]["dataset_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="dataset.*checksum|dataset_sha256"):
        _select(declarations=declarations, records=records)


def test_record_method_hash_must_match_the_tracked_method_or_configuration():
    declarations = _declarations("v27-a")
    records = _records(_base_values(), declarations)
    records[0]["method_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="method.*checksum|method_sha256"):
        _select(declarations=declarations, records=records)


def test_null_de_only_failure_after_v28_downgrades_instead_of_triggering_v29():
    values = _base_values()
    values["v27-a"]["null_de_fpr"] = 0.061
    values["v28-a"] = dict(values["v27-a"])
    attempts = _attempts(("v27-a", "v27"), ("v28-a", "v28"))
    declarations = _declarations("v27-a", "v28-a")

    report = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=_records(values, declarations),
        intervals=_intervals("v27-a", "v28-a"),
    )

    assert all(item.efficacy_pass for item in report.assessments)
    assert all(not item.gates["null_de_safety"].passed for item in report.assessments)
    assert report.trigger == "downgrade_claim"


def test_v29_attempt_is_invalid_until_v28_was_assessed():
    values = _base_values()
    values["v29-a"] = dict(values["v27-a"])

    with pytest.raises(ValueError, match="v28|parent"):
        attempts = _attempts(("v27-a", "v27"), ("v29-a", "v29"))
        declarations = _declarations("v27-a", "v29-a")
        _select(
            values,
            attempts=attempts,
            declarations=declarations,
            records=_records(values, declarations),
            intervals=_intervals("v27-a", "v29-a"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("estimate", True),
        ("ci_lower", "-0.1"),
        ("ci_upper", False),
    ],
)
def test_orthogonal_intervals_reject_boolean_and_numeric_string_aliases(field, value):
    intervals = _intervals("v27-a")
    intervals[0][field] = value

    with pytest.raises((TypeError, ValueError), match="numeric|materiality"):
        _select(intervals=intervals)


def test_interval_rows_cannot_override_the_tracked_materiality_margin():
    intervals = _intervals("v27-a")
    intervals[0]["materiality_margin"] = 100.0

    with pytest.raises(ValueError, match="missing or extra"):
        _select(intervals=intervals)


def test_attempt_and_endpoint_order_do_not_change_selection_serialization():
    values = _base_values()
    values["v27-z"] = dict(values["v27-a"])
    attempts = _attempts(("v27-z", "v27"), ("v27-a", "v27"))
    declarations = _declarations("v27-z", "v27-a")
    records = _records(values, declarations)
    intervals = list(reversed(_intervals("v27-z", "v27-a")))

    first = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=records,
        intervals=intervals,
    )
    second = _select(
        values,
        attempts=tuple(reversed(attempts)),
        declarations=tuple(reversed(declarations)),
        records=list(reversed(records)),
        intervals=list(reversed(intervals)),
    )

    assert first.to_dict() == second.to_dict()


def test_v28_is_not_retained_without_a_strict_pareto_improvement():
    values = _base_values()
    values["v28-a"] = dict(values["v27-a"])
    attempts = _attempts(("v27-a", "v27"), ("v28-a", "v28"))
    declarations = _declarations("v27-a", "v28-a")
    unsafe_v27 = {("v27-a", endpoint) for endpoint in ENDPOINTS}

    report = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=_records(values, declarations),
        intervals=_intervals("v27-a", "v28-a", unsafe=unsafe_v27),
    )

    retention = report.by_configuration["v28-a"].gates["revision_retention"]
    assert retention.passed is False
    assert retention.details["reason_code"] == "v28_no_strict_pareto_improvement"
    assert report.selected_configuration is None


def test_v28_retention_requires_pareto_improvement_and_zero_de_safety():
    values = _base_values()
    values["v28-a"] = dict(values["v27-a"], mse_dropout=0.70)
    attempts = _attempts(("v27-a", "v27"), ("v28-a", "v28"))
    declarations = _declarations("v27-a", "v28-a")
    unsafe_v27 = {("v27-a", endpoint) for endpoint in ENDPOINTS}

    retained = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=_records(values, declarations),
        intervals=_intervals("v27-a", "v28-a", unsafe=unsafe_v27),
    )
    assert retained.by_configuration["v28-a"].gates["revision_retention"].passed is True
    assert retained.selected_configuration == "v28-a"

    values["v28-a"]["null_de_fpr"] = 0.061
    rejected = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=_records(values, declarations),
        intervals=_intervals("v27-a", "v28-a", unsafe=unsafe_v27),
    )
    gate = rejected.by_configuration["v28-a"].gates["revision_retention"]
    assert gate.passed is False
    assert gate.details["reason_code"] == "v28_zero_or_de_safety_violation"


def test_v29_retention_requires_structure_gain_without_material_dropout_loss():
    values = _base_values()
    values["v28-a"] = dict(values["v27-a"], mse_dropout=0.70)
    values["v29-a"] = dict(values["v28-a"], corr_err=0.95, mse_dropout=0.71)
    attempts = _attempts(
        ("v27-a", "v27"),
        ("v28-a", "v28"),
        ("v29-a", "v29"),
    )
    declarations = _declarations("v27-a", "v28-a", "v29-a")
    unsafe_parents = {
        (candidate, endpoint)
        for candidate in ("v27-a", "v28-a")
        for endpoint in ENDPOINTS
    }

    retained = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=_records(values, declarations),
        intervals=_intervals("v27-a", "v28-a", "v29-a", unsafe=unsafe_parents),
    )
    gate = retained.by_configuration["v29-a"].gates["revision_retention"]
    assert gate.passed is True
    assert gate.details["maximum_dropout_mse_loss"] == pytest.approx(1 / 70)
    assert retained.selected_configuration == "v29-a"

    values["v29-a"]["mse_dropout"] = 0.75
    rejected = _select(
        values,
        attempts=attempts,
        declarations=declarations,
        records=_records(values, declarations),
        intervals=_intervals("v27-a", "v28-a", "v29-a", unsafe=unsafe_parents),
    )
    gate = rejected.by_configuration["v29-a"].gates["revision_retention"]
    assert gate.passed is False
    assert gate.details["reason_code"] == "v29_material_dropout_mse_loss"


def test_identity_calibration_equivalence_is_retained_with_an_explicit_reason_code():
    from maskimpute_benchmark.selection import SearchExclusion

    exclusion = SearchExclusion(
        configuration_id="v27-calibrated-score",
        version="v27",
        equivalent_to="v27-a",
        reason_code="retained_identity_calibrator_equals_direct_score",
    )

    report = _select(exclusions=(exclusion,))

    assert report.to_dict()["excluded_configurations"] == [
        {
            "configuration_id": "v27-calibrated-score",
            "version": "v27",
            "equivalent_to": "v27-a",
            "reason_code": "retained_identity_calibrator_equals_direct_score",
        }
    ]
