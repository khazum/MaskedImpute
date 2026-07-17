from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from maskimpute_benchmark.protocol import canonical_sha256
from maskimpute_benchmark.publication_freeze import (
    PublicationFreezeError,
    _pinned_parent,
    build_frozen_method_payload,
    freeze_publication_round,
    prepare_frozen_method,
    validate_frozen_method,
)


def _selection_report() -> dict[str, object]:
    return {
        "assessments": [
            {
                "configuration_id": "v28-c01-nb-parent-c03",
                "version": "v28",
                "gates": {
                    "dropout_improvement": {
                        "passed": True,
                        "value": 0.08,
                        "threshold": 0.05,
                        "details": {},
                    }
                },
                "efficacy_pass": True,
                "safety_pass": True,
                "eligible": True,
                "ineligibility_reasons": [],
                "independent_draws": 8,
            }
        ],
        "pareto_set": ["v28-c01-nb-parent-c03"],
        "selected_configuration": "v28-c01-nb-parent-c03",
        "trigger": "freeze_candidate",
        "excluded_configurations": [],
        "authority_bindings": {
            "development_result_sha256": "a" * 64,
            "evaluation_manifest_file_sha256": "b" * 64,
            "retained_calibration_artifact_sha256": "c" * 64,
        },
        "selection_rule": ("all_hard_gates_then_lowest_version_then_configuration_id"),
        "combined_score": None,
    }


def _configuration() -> dict[str, object]:
    configuration = {
        "method_version": "v28",
        "decoder": "negative_binomial",
        "encoder_mode": "explicit_mask",
        "output_policy": "selective",
        "score_policy": "retained_development_calibrator",
        "hyperparameters": {"latent_dim": 24},
        "decoder_hyperparameters": {"dispersion_prior_strength": 20.0},
    }
    return {
        "configuration_id": "v28-c01-nb-parent-c03",
        "version": "v28",
        "configuration": configuration,
        "configuration_sha256": canonical_sha256(configuration),
    }


def _method(
    method_id: str,
    *,
    role: str = "competitor",
    track: str = "same_input",
    execution_scope: str = "same_input_required",
    applicability_reason: str | None = None,
    integration_status: str = "implemented",
    integration_reason: str | None = "development_execution_completed",
) -> dict[str, object]:
    in_tree = method_id in {"observed", "capacity-matched-ae", "maskimpute"}
    return {
        "id": method_id,
        "display_name": method_id,
        "role": role,
        "track": track,
        "execution_scope": execution_scope,
        "applicability_reason": applicability_reason,
        "input_scale": "raw_counts",
        "output_scale": "raw_counts",
        "stochastic": method_id != "observed",
        "seed_policy": "not_applicable" if method_id == "observed" else "required",
        "source": {
            "kind": "in_tree" if in_tree else "git",
            "url": None if in_tree else f"https://example.org/{method_id}.git",
            "revision": None if in_tree else "1" * 40,
            "tree": None if in_tree else "2" * 40,
            "cache_path": None if in_tree else f"artifacts/method-sources/{method_id}",
            "freeze_binding": "study_freeze_commit" if in_tree else None,
        },
        "license": {"status": "declared", "spdx": "MIT", "notice": None},
        "citation": {"status": "verified", "doi": "10.0000/example", "url": None},
        "environment": {
            "id": f"{method_id}-environment",
            "status": "ready",
            "lock_sha256": "4" * 64,
        },
        "resources": {
            "timeout_seconds": 21600,
            "cpu_cores": 8,
            "gpu_required": False,
            "max_rss_gib": 48,
            "max_gpu_gib": 0,
        },
        "preserves_observed_positives": method_id == "observed",
        "source_policy": (
            "study_freeze_bound_in_tree"
            if in_tree
            else "pinned_adapter_isolated_environment"
        ),
        "integration_status": integration_status,
        "integration_reason": integration_reason,
    }


def _runtime_summary(
    lock_sha256: str = "4" * 64,
    environment_ids: tuple[str, ...] = ("benchmark", "magic"),
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "lock_file_sha256": lock_sha256,
        "environment_inventory_sha256s": {
            environment_id: hashlib.sha256(environment_id.encode()).hexdigest()
            for environment_id in environment_ids
        },
    }


def _execution_evidence(
    method_ids: tuple[str, ...] = ("observed", "maskimpute", "magic"),
    *,
    unavailable: tuple[str, ...] = (),
) -> dict[str, dict[str, object]]:
    result = {}
    for method_id in method_ids:
        failed = method_id in unavailable
        failure_reasons = ["adapter_unavailable"] if failed else []
        attempted_dataset_ids = ["dataset-001"]
        completed_dataset_ids = [] if failed else attempted_dataset_ids
        external = method_id in {"d3impute", "sctsi"}
        result[method_id] = {
            "artifact": (
                "external_reference_checkpoint"
                if external
                else "reconstruction_checkpoint"
            ),
            "execution_track": "external_reference" if external else "same_input",
            "checkpoint_payload_sha256": hashlib.sha256(
                b"complete-checkpoint"
            ).hexdigest(),
            "records_sha256": hashlib.sha256(
                f"{method_id}-records".encode()
            ).hexdigest(),
            "eligible_dataset_count": 1,
            "eligible_dataset_ids_sha256": canonical_sha256(attempted_dataset_ids),
            "attempted_run_count": 1,
            "completed_run_count": 0 if failed else 1,
            "failed_run_count": 1 if failed else 0,
            "status_counts": {"unavailable" if failed else "completed": 1},
            "attempted_dataset_count": 1,
            "completed_dataset_count": 0 if failed else 1,
            "attempted_dataset_ids_sha256": canonical_sha256(attempted_dataset_ids),
            "completed_dataset_ids_sha256": canonical_sha256(completed_dataset_ids),
            "failure_reason_codes": failure_reasons,
            "failure_reasons_sha256": canonical_sha256(failure_reasons),
        }
    return result


@lru_cache(maxsize=1)
def _valid_calibration_payload() -> dict[str, object]:
    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
        fit_development_calibration,
    )

    records = []
    for draw, view, token in (
        ("draw-01", "moderate", "1"),
        ("draw-01", "severe", "2"),
        ("draw-02", "moderate", "3"),
        ("draw-02", "severe", "4"),
    ):
        dataset_sha256 = hashlib.sha256(f"{draw}:{view}".encode()).hexdigest()
        records.append(
            CalibrationRecord(
                p_pre_zero=(0.1, 0.25, 0.7, 0.9),
                target=(0, 0, 1, 1),
                mechanism="symsim",
                biological_id=draw,
                manifest_sha256=token * 64,
                truth_kind="exact_pre_capture",
                namespace="dev",
                data_role="development",
                technical_view=view,
                dataset_id=f"dataset-{dataset_sha256[:24]}",
                dataset_sha256=dataset_sha256,
                protocol_sha256=DEVELOPMENT_PROTOCOL_SHA256,
            )
        )
    return fit_development_calibration(tuple(records)).to_dict()


def _calibrator_summary(
    artifact_file_sha256: str = "7" * 64,
) -> dict[str, object]:
    artifact = deepcopy(_valid_calibration_payload())
    definition = artifact["calibrator"]
    inference_features = artifact["inference_features"]
    selected_algorithm = artifact["selected_algorithm"]
    return {
        "score_policy": "retained_development_calibrator",
        "final_usage": "retained_all_development_calibrator",
        "selected_algorithm": selected_algorithm,
        "artifact_file_sha256": artifact_file_sha256,
        "artifact_payload_sha256": artifact["payload_sha256"],
        "artifact": artifact,
        "calibrator_definition": definition,
        "calibrator_definition_sha256": canonical_sha256(definition),
        "inference_features": inference_features,
        "inference_features_sha256": canonical_sha256(inference_features),
    }


def _ablation_registry() -> dict[str, object]:
    reference = {
        "id": "maskimpute-reference",
        "changed_component": "reference",
        "positive_masking": "log_count_stratified",
        "pre_zero_regularizer": True,
        "encoder_mode": "explicit_mask",
        "gate": "power_complement",
        "output_policy": "selective",
        "score_source": "retained_calibrator",
    }
    capacity = {
        "id": "capacity-matched-ae",
        "changed_component": "control_bundle",
        "positive_masking": "uniform",
        "pre_zero_regularizer": False,
        "encoder_mode": "explicit_mask",
        "gate": "none",
        "output_policy": "full_ungated",
        "score_source": "retained_calibrator",
    }
    return {"schema_version": 1, "reference": reference, "variants": [capacity]}


def _minimum_artifact_bindings() -> dict[str, dict[str, str]]:
    return {
        "retained_calibration": {
            "path": "artifacts/study/development/calibration/retained_calibration.json",
            "sha256": "7" * 64,
        },
        "ablation_registry": {
            "path": "study/ablations.json",
            "sha256": "b" * 64,
        },
    }


def _payload(
    *, selected_calibrator_summary: dict[str, object] | None = None
) -> dict[str, object]:
    return build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry={
            "schema_version": 1,
            "methods": [
                _method("observed", role="control", integration_reason=None),
                _method("maskimpute", role="candidate", integration_reason=None),
                _method("magic"),
            ],
        },
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=_execution_evidence(),
        selected_calibrator_summary=(
            _calibrator_summary()
            if selected_calibrator_summary is None
            else selected_calibrator_summary
        ),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(),
        artifact_bindings={
            "selection_input": {
                "path": "artifacts/study/development/evaluation/development_selection_input.json",
                "sha256": "5" * 64,
            },
            "selection_report": {
                "path": "artifacts/study/development/evaluation/development_selection_report.json",
                "sha256": "6" * 64,
            },
            "runtime_lock": {
                "path": "environments/development-runtime.lock.json",
                "sha256": "4" * 64,
            },
            **_minimum_artifact_bindings(),
        },
    )


def test_frozen_method_retains_exact_selection_and_competitor_authority() -> None:
    payload = _payload()

    assert payload["selected_configuration_id"] == "v28-c01-nb-parent-c03"
    assert payload["selected_version"] == "v28"
    assert payload["selected_configuration"] == _configuration()["configuration"]
    assert payload["selection_gate_table"] == _selection_report()["assessments"]
    assert payload["required_comparator_ids"] == ["observed", "magic"]
    assert [row["id"] for row in payload["method_denominator"]] == [
        "observed",
        "maskimpute",
        "magic",
    ]
    assert payload["correlation_gene_panel_rule"] == {
        "id": "all-retained-genes-v1",
        "selection": "all_genes_after_only_zero_library_cell_exclusion",
        "gene_filtering": "forbidden",
        "shared_across_methods": True,
    }
    assert payload["runtime_lock_sha256"] == "4" * 64
    assert payload["selected_calibrator"] == _calibrator_summary()
    assert payload["selected_ablation_control"]["capacity_matched_control_id"] == (
        "capacity-matched-ae"
    )
    assert (
        payload["selected_ablation_control"]["capacity_matched_definition"]
        == (_ablation_registry()["variants"][0])
    )
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    assert payload["payload_sha256"] == canonical_sha256(unsigned)


def test_frozen_method_embeds_an_executable_complete_calibration_artifact() -> None:
    from maskimpute.calibration import CalibrationArtifact

    embedded = _payload()["selected_calibrator"]["artifact"]
    artifact = CalibrationArtifact(embedded)

    assert artifact.to_dict() == embedded
    assert artifact.transform([0.1, 0.9]).shape == (2,)


def test_in_tree_pending_citation_is_recorded_as_self_citation() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
        _method("magic"),
    ]
    for row in methods[:2]:
        row["citation"] = {"status": "pending", "doi": None, "url": None}

    payload = build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry={"schema_version": 1, "methods": methods},
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=_execution_evidence(),
        selected_calibrator_summary=_calibrator_summary(),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(),
        artifact_bindings=_minimum_artifact_bindings(),
    )

    assert [row["citation_disposition"] for row in payload["method_denominator"]] == [
        "in_tree_self_citation_no_external_doi",
        "in_tree_self_citation_no_external_doi",
        "verified_external_citation",
    ]


def test_in_tree_pending_project_license_remains_a_submission_blocker() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
        _method("magic"),
    ]
    methods[1]["license"] = {
        "status": "pending",
        "spdx": None,
        "notice": "project license requires human approval",
    }
    methods[1]["citation"] = {"status": "pending", "doi": None, "url": None}

    with pytest.raises(PublicationFreezeError, match="maskimpute.*project license"):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=_selection_report(),
            candidate_configuration=_configuration(),
            method_registry={"schema_version": 1, "methods": methods},
            required_comparator_ids=("observed", "magic"),
            method_execution_evidence=_execution_evidence(),
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(),
            artifact_bindings=_minimum_artifact_bindings(),
        )


def test_external_method_pending_citation_remains_a_freeze_blocker() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
        _method("magic"),
    ]
    methods[2]["citation"] = {"status": "pending", "doi": None, "url": None}

    with pytest.raises(PublicationFreezeError, match="magic.*external citation"):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=_selection_report(),
            candidate_configuration=_configuration(),
            method_registry={"schema_version": 1, "methods": methods},
            required_comparator_ids=("observed", "magic"),
            method_execution_evidence=_execution_evidence(),
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(),
            artifact_bindings=_minimum_artifact_bindings(),
        )


def test_frozen_method_rejects_partial_calibration_artifact_even_if_definition_runs() -> (
    None
):
    summary = _calibrator_summary()
    partial = deepcopy(summary["artifact"])
    partial.pop("artifact_type")
    summary["artifact"] = partial

    with pytest.raises(PublicationFreezeError, match="calibrator artifact"):
        _payload(selected_calibrator_summary=summary)


def test_frozen_method_materializes_complete_ordered_applicability_denominator() -> (
    None
):
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
        _method("magic"),
        _method("biaeimpute"),
        _method(
            "d3impute",
            track="external_reference",
            execution_scope="external_reference_only",
        ),
        _method(
            "wedge",
            execution_scope="historical_not_run",
            integration_status="historical",
            integration_reason="historical_adapter_not_run",
        ),
        _method(
            "scgacl",
            execution_scope="not_applicable",
            applicability_reason="no_truth_free_configuration",
            integration_status="unavailable",
            integration_reason="upstream_no_truth_free_configuration",
        ),
    ]
    method_ids = tuple(row["id"] for row in methods[:5])

    payload = build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry={"schema_version": 1, "methods": methods},
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=_execution_evidence(method_ids),
        selected_calibrator_summary=_calibrator_summary(),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(
            environment_ids=("benchmark", "biaeimpute", "d3impute", "magic")
        ),
        artifact_bindings=_minimum_artifact_bindings(),
    )

    denominator = payload["method_denominator"]
    assert [row["id"] for row in denominator] == [row["id"] for row in methods]
    assert denominator[3]["claim_required"] is False
    assert denominator[4]["final_applicability"] == {
        "rule": "matched_bulk_reference_present",
        "non_run_reason": "matched_bulk_reference_absent",
        "required_reference": {
            "kind": "prespecified_matched_bulk_expression",
            "binding": "final_dataset_manifest_external_reference",
            "evaluator_truth_as_reference": "forbidden",
        },
    }
    assert denominator[5]["disposition"] == "historical_not_run"
    assert denominator[6]["disposition"] == "not_applicable"


def test_frozen_method_rejects_pending_historical_disposition() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
        _method("magic"),
        _method(
            "wedge",
            execution_scope="historical_not_run",
            integration_status="pending",
            integration_reason="historical_adapter_not_run",
        ),
    ]

    with pytest.raises(PublicationFreezeError, match="wedge.*historical disposition"):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=_selection_report(),
            candidate_configuration=_configuration(),
            method_registry={"schema_version": 1, "methods": methods},
            required_comparator_ids=("observed", "magic"),
            method_execution_evidence=_execution_evidence(),
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(),
            artifact_bindings=_minimum_artifact_bindings(),
        )


def test_checkpoint_terminalizes_pending_registry_without_mutating_plan_authority() -> (
    None
):
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    methods.append(
        _method(
            "magic",
            integration_status="pending",
            integration_reason="environment_lock_pending",
        )
    )

    registry = {"schema_version": 1, "methods": methods}
    payload = build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry=registry,
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=_execution_evidence(),
        selected_calibrator_summary=_calibrator_summary(),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(),
        artifact_bindings=_minimum_artifact_bindings(),
    )

    magic = payload["method_denominator"][2]
    assert payload["method_registry_sha256"] == canonical_sha256(registry)
    assert magic["registry_integration_status"] == "pending"
    assert magic["registry_integration_reason"] == "environment_lock_pending"
    assert magic["integration_status"] == "implemented"
    assert magic["integration_reason"] == "development_execution_completed"


def test_frozen_method_accepts_explicit_reason_coded_unavailable_comparator() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    evidence = _execution_evidence(unavailable=("magic",))
    unavailable = _method(
        "magic",
        integration_status="unavailable",
        integration_reason=(
            "technical_unavailable_development_attempts_"
            f"{evidence['magic']['failure_reasons_sha256'][:16]}"
        ),
    )
    unavailable["environment"] = {
        "id": "magic-environment",
        "status": "failed",
        "lock_sha256": None,
    }
    methods.append(unavailable)

    payload = build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry={"schema_version": 1, "methods": methods},
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=evidence,
        selected_calibrator_summary=_calibrator_summary(),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(),
        artifact_bindings=_minimum_artifact_bindings(),
    )

    assert payload["method_denominator"][2]["disposition"] == (
        "explicit_reason_coded_unavailable"
    )
    assert payload["method_denominator"][2]["final_applicability"] == {
        "rule": "never",
        "non_run_reason": unavailable["integration_reason"],
        "required_reference": None,
    }


def test_frozen_method_rejects_unavailable_disposition_with_any_completed_run() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    evidence = _execution_evidence()
    unavailable = _method(
        "magic",
        integration_status="unavailable",
        integration_reason=(
            "technical_unavailable_development_attempts_"
            f"{evidence['magic']['failure_reasons_sha256'][:16]}"
        ),
    )
    unavailable["environment"] = {
        "id": "magic-environment",
        "status": "failed",
        "lock_sha256": None,
    }
    methods.append(unavailable)

    with pytest.raises(PublicationFreezeError, match="magic.*ready runtime"):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=_selection_report(),
            candidate_configuration=_configuration(),
            method_registry={"schema_version": 1, "methods": methods},
            required_comparator_ids=("observed", "magic"),
            method_execution_evidence=evidence,
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(),
            artifact_bindings=_minimum_artifact_bindings(),
        )


def test_frozen_method_rejects_implemented_method_without_dataset_coverage() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
        _method("magic"),
    ]
    evidence = _execution_evidence()
    evidence["magic"]["attempted_dataset_ids_sha256"] = canonical_sha256(
        ["dataset-other"]
    )

    with pytest.raises(PublicationFreezeError, match="magic.*eligibility"):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=_selection_report(),
            candidate_configuration=_configuration(),
            method_registry={"schema_version": 1, "methods": methods},
            required_comparator_ids=("observed", "magic"),
            method_execution_evidence=evidence,
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(),
            artifact_bindings=_minimum_artifact_bindings(),
        )


def test_failed_checkpoint_terminalizes_pending_registry_reason() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    unavailable = _method(
        "magic",
        integration_status="unavailable",
        integration_reason="environment_lock_pending",
    )
    unavailable["environment"] = {
        "id": "magic-environment",
        "status": "failed",
        "lock_sha256": None,
    }
    methods.append(unavailable)

    evidence = _execution_evidence(unavailable=("magic",))
    payload = build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry={"schema_version": 1, "methods": methods},
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=evidence,
        selected_calibrator_summary=_calibrator_summary(),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(),
        artifact_bindings=_minimum_artifact_bindings(),
    )

    magic = payload["method_denominator"][2]
    assert magic["registry_integration_reason"] == "environment_lock_pending"
    assert magic["integration_reason"] == (
        "technical_unavailable_development_attempts_"
        f"{evidence['magic']['failure_reasons_sha256'][:16]}"
    )


def test_frozen_method_rejects_pending_reason_inside_failed_attempt_evidence() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    evidence = _execution_evidence(unavailable=("magic",))
    pending_reasons = ["environment_lock_pending"]
    evidence["magic"]["failure_reason_codes"] = pending_reasons
    evidence["magic"]["failure_reasons_sha256"] = canonical_sha256(pending_reasons)
    unavailable = _method(
        "magic",
        integration_status="unavailable",
        integration_reason=(
            "technical_unavailable_development_attempts_"
            f"{evidence['magic']['failure_reasons_sha256'][:16]}"
        ),
    )
    unavailable["environment"] = {
        "id": "magic-environment",
        "status": "failed",
        "lock_sha256": None,
    }
    methods.append(unavailable)

    with pytest.raises(PublicationFreezeError, match="magic.*execution evidence"):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=_selection_report(),
            candidate_configuration=_configuration(),
            method_registry={"schema_version": 1, "methods": methods},
            required_comparator_ids=("observed", "magic"),
            method_execution_evidence=evidence,
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(),
            artifact_bindings=_minimum_artifact_bindings(),
        )


def test_failed_checkpoint_overrides_unbound_registry_availability_claim() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    unavailable = _method(
        "magic",
        integration_status="unavailable",
        integration_reason="technical_unavailable_arbitrary_unrelated_claim",
    )
    unavailable["environment"] = {
        "id": "magic-environment",
        "status": "failed",
        "lock_sha256": None,
    }
    methods.append(unavailable)

    evidence = _execution_evidence(unavailable=("magic",))
    payload = build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry={"schema_version": 1, "methods": methods},
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=evidence,
        selected_calibrator_summary=_calibrator_summary(),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(),
        artifact_bindings=_minimum_artifact_bindings(),
    )

    magic = payload["method_denominator"][2]
    assert magic["registry_integration_reason"] == (
        "technical_unavailable_arbitrary_unrelated_claim"
    )
    assert magic["integration_reason"] != magic["registry_integration_reason"]


def test_checkpoint_replaces_pre_execution_smoke_disposition() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    methods.append(
        _method(
            "magic",
            integration_status="implemented",
            integration_reason="real_pinned_smoke_passed_environment_lock_pending",
        )
    )

    payload = build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=_selection_report(),
        candidate_configuration=_configuration(),
        method_registry={"schema_version": 1, "methods": methods},
        required_comparator_ids=("observed", "magic"),
        method_execution_evidence=_execution_evidence(),
        selected_calibrator_summary=_calibrator_summary(),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256="4" * 64,
        runtime_environment_summary=_runtime_summary(),
        artifact_bindings=_minimum_artifact_bindings(),
    )

    magic = payload["method_denominator"][2]
    assert magic["registry_integration_reason"].endswith("lock_pending")
    assert magic["integration_reason"] == "development_execution_completed"


def test_frozen_method_requires_completed_execution_for_in_tree_comparator() -> None:
    methods = [
        _method("observed", role="control", integration_reason=None),
        _method("maskimpute", role="candidate", integration_reason=None),
    ]
    capacity_matched = _method(
        "capacity-matched-ae",
        integration_status="implemented",
        integration_reason=None,
    )
    capacity_matched["source"] = {
        "kind": "in_tree",
        "url": None,
        "revision": None,
        "tree": None,
        "cache_path": None,
        "freeze_binding": "study_freeze_commit",
    }
    capacity_matched["source_policy"] = "study_freeze_bound_in_tree"
    methods.append(capacity_matched)

    with pytest.raises(
        PublicationFreezeError, match="capacity-matched-ae.*execution evidence"
    ):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=_selection_report(),
            candidate_configuration=_configuration(),
            method_registry={"schema_version": 1, "methods": methods},
            required_comparator_ids=("observed", "capacity-matched-ae"),
            method_execution_evidence=_execution_evidence(("observed", "maskimpute")),
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(
                environment_ids=("benchmark",)
            ),
            artifact_bindings=_minimum_artifact_bindings(),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda report, config: report.update(trigger="v29"), "freeze_candidate"),
        (
            lambda report, config: report.update(selected_configuration=None),
            "selected configuration",
        ),
        (
            lambda report, config: config.update(configuration_sha256="0" * 64),
            "configuration checksum",
        ),
        (
            lambda report, config: config.update(configuration_id="v28-other"),
            "selected configuration",
        ),
    ],
)
def test_frozen_method_rejects_nonfreezable_or_mismatched_selection(
    mutate, message: str
) -> None:
    report = deepcopy(_selection_report())
    configuration = deepcopy(_configuration())
    mutate(report, configuration)

    with pytest.raises(PublicationFreezeError, match=message):
        build_frozen_method_payload(
            preparation_commit="f" * 40,
            selection_report=report,
            candidate_configuration=configuration,
            method_registry={
                "schema_version": 1,
                "methods": [
                    _method("observed", role="control", integration_reason=None),
                    _method("maskimpute", role="candidate", integration_reason=None),
                    _method("magic"),
                ],
            },
            required_comparator_ids=("observed", "magic"),
            method_execution_evidence=_execution_evidence(),
            selected_calibrator_summary=_calibrator_summary(),
            ablation_registry=_ablation_registry(),
            runtime_lock_sha256="4" * 64,
            runtime_environment_summary=_runtime_summary(),
            artifact_bindings=_minimum_artifact_bindings(),
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _runtime_lock_payload(environment_ids: tuple[str, ...]) -> dict[str, object]:
    entries = []
    for environment_id in sorted(environment_ids):
        inventory = {
            "schema": "maskimpute-python-runtime-inventory-v1",
            "interpreter": {
                "implementation": "cpython",
                "version": [3, 11, 0],
                "cache_tag": "cpython-311",
                "is_virtual_environment": True,
            },
            "packages": [],
            "executable_sha256": hashlib.sha256(
                f"{environment_id}-executable".encode()
            ).hexdigest(),
            "launcher": {
                "kind": "regular",
                "sha256": hashlib.sha256(
                    f"{environment_id}-launcher".encode()
                ).hexdigest(),
            },
            "runtime_roots": [
                {
                    "role": "runtime-root",
                    "kind": "directory",
                    "content_sha256": hashlib.sha256(
                        f"{environment_id}-root".encode()
                    ).hexdigest(),
                    "entry_count": 1,
                }
            ],
            "native_linkage_sha256": hashlib.sha256(
                f"{environment_id}-native".encode()
            ).hexdigest(),
        }
        entries.append(
            {
                "id": environment_id,
                "kind": "python",
                "inventory": inventory,
                "inventory_sha256": canonical_sha256(inventory),
            }
        )
    return {
        "schema": "maskimpute-runtime-environment-lock-v1",
        "environments": entries,
    }


def _checkpoint_payload() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "planned_run_count": 3,
        "status": "completed",
        "records": [
            {
                "run": {
                    "method_id": method_id,
                    "dataset_id": "dataset-001",
                    "status": "completed",
                    "reason": None,
                },
                "metrics": [],
            }
            for method_id in ("observed", "maskimpute", "magic")
        ],
    }
    return {**body, "checkpoint_sha256": canonical_sha256(body)}


def _dataset_status_payload() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "namespace": "dev",
        "status": "completed",
        "completed_count": 1,
        "failed_count": 0,
        "rows": [{"dataset_id": "dataset-001", "status": "completed"}],
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def _retained_calibration_payload() -> dict[str, object]:
    return deepcopy(_valid_calibration_payload())


def _external_checkpoint_payload(repository: Path) -> dict[str, object]:
    binding = {
        "method_id": "d3impute",
        "dataset_id": "dataset-001",
        "reference_id": "matched-bulk-001",
        "source_kind": "prespecified_measured_bulk_expression",
        "source_sha256": "a" * 64,
        "matrix_sha256": "b" * 64,
        "evaluator_truth_used": False,
    }
    body: dict[str, object] = {
        "schema_version": 1,
        "track": "external_reference",
        "status": "completed",
        "method_ids": ["d3impute"],
        "eligible_dataset_ids": ["dataset-001"],
        "reference_bindings": [binding],
        "method_registry_file_sha256": hashlib.sha256(
            (repository / "study/methods.json").read_bytes()
        ).hexdigest(),
        "dataset_status_file_sha256": hashlib.sha256(
            (
                repository / "artifacts/study/development/results/dataset_status.json"
            ).read_bytes()
        ).hexdigest(),
        "runtime_lock_file_sha256": hashlib.sha256(
            (repository / "environments/development-runtime.lock.json").read_bytes()
        ).hexdigest(),
        "planned_run_count": 1,
        "records": [
            {
                "run": {
                    "method_id": "d3impute",
                    "dataset_id": "dataset-001",
                    "status": "completed",
                    "reason": None,
                    "reference_id": binding["reference_id"],
                    "reference_source_sha256": binding["source_sha256"],
                    "reference_matrix_sha256": binding["matrix_sha256"],
                },
                "metrics": [],
            }
        ],
    }
    return {**body, "checkpoint_sha256": canonical_sha256(body)}


def _repository_fixture(
    tmp_path: Path, *, include_external: bool = False
) -> tuple[Path, dict[str, object]]:
    repository = tmp_path / "repository"
    selection_input = {
        "schema_version": 2,
        "records": [],
        "orthogonal_intervals": [],
        "dataset_manifest_sha256": "7" * 64,
        "count_score_manifest_sha256": "8" * 64,
        "retained_calibration_artifact_sha256": "9" * 64,
        "evaluation_manifest_sha256": "a" * 64,
        "result_sha256": "b" * 64,
    }
    report = _selection_report()
    methods = {
        "schema_version": 1,
        "methods": [
            _method("observed", role="control", integration_reason=None),
            _method("maskimpute", role="candidate", integration_reason=None),
            _method("magic"),
        ],
    }
    if include_external:
        methods["methods"].append(
            _method(
                "d3impute",
                track="external_reference",
                execution_scope="external_reference_only",
            )
        )
    runtime_ids = (
        ("benchmark", "d3impute", "magic")
        if include_external
        else ("benchmark", "magic")
    )
    runtime_lock = _runtime_lock_payload(runtime_ids)
    runtime_raw = (
        json.dumps(runtime_lock, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    runtime_sha256 = hashlib.sha256(runtime_raw).hexdigest()
    for method in methods["methods"]:
        method["environment"]["lock_sha256"] = runtime_sha256
    selection_contract = {
        "schema_version": 1,
        "required_comparator_ids": ["observed", "magic"],
    }
    configuration = _configuration()
    v28 = {
        "schema_version": 1,
        "status": "conditional_on_v28_trigger",
        "trigger": "v28",
        "parent_configuration_id": "v27-parent",
        "parent_configuration_sha256": "c" * 64,
        **configuration,
        "reason_code": "prespecified_decoder_only_revision",
    }
    paths = {
        "selection_input": "artifacts/study/development/evaluation/development_selection_input.json",
        "selection_report": "artifacts/study/development/evaluation/development_selection_report.json",
        "evaluation_manifest": "artifacts/study/development/evaluation/evaluation_manifest.json",
        "reconstruction_checkpoint": "artifacts/study/development/competition-reconstruction/checkpoint.json",
        "dataset_status": "artifacts/study/development/results/dataset_status.json",
        "count_score_manifest": "artifacts/study/development/count_scores/manifest.json",
        "retained_calibration": "artifacts/study/development/calibration/retained_calibration.json",
        "runtime_lock": "environments/development-runtime.lock.json",
        "method_registry": "study/methods.json",
        "selection_contract": "study/selection_contract.json",
        "development_search": "study/development_search.json",
        "v28_revision": "study/v28_revision.json",
        "ablation_registry": "study/ablations.json",
        "scaling_panel": "study/scaling_panel.json",
        "protocol": "study/protocol.json",
        "saver_qualification": "environments/saver-r.qualification.json",
        "saver_package_lock": "environments/saver-r.lock.json",
        "saver_build_receipt": "environments/saver-r.build-receipt.json",
    }
    for name, relative in paths.items():
        payload: object = {"artifact": name}
        if name == "selection_input":
            payload = selection_input
        elif name == "selection_report":
            payload = report
        elif name == "method_registry":
            payload = methods
        elif name == "selection_contract":
            payload = selection_contract
        elif name == "development_search":
            payload = {"schema_version": 1, "configurations": []}
        elif name == "v28_revision":
            payload = v28
        elif name == "runtime_lock":
            payload = runtime_lock
        elif name == "reconstruction_checkpoint":
            payload = _checkpoint_payload()
        elif name == "dataset_status":
            payload = _dataset_status_payload()
        elif name == "retained_calibration":
            payload = _retained_calibration_payload()
        elif name == "ablation_registry":
            payload = _ablation_registry()
        elif name == "scaling_panel":
            payload = json.loads(
                Path("study/scaling_panel.json").read_text(encoding="utf-8")
            )
        elif name == "protocol":
            payload = json.loads(
                Path("study/protocol.json").read_text(encoding="utf-8")
            )
        elif name in {
            "saver_qualification",
            "saver_package_lock",
            "saver_build_receipt",
        }:
            destination = repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(relative), destination)
            continue
        _write_json(repository / relative, payload)
    if include_external:
        _write_json(
            repository
            / "artifacts/study/development/competition-external-reference/checkpoint.json",
            _external_checkpoint_payload(repository),
        )
    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Freeze Test"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "freeze@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "publication preparation authority"],
        cwd=repository,
        check=True,
    )
    return repository, report


def test_prepare_and_validate_frozen_method_recompute_fixed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    calls: list[tuple[Path, dict[str, object]]] = []

    def recompute(selected_repository: Path, payload: dict[str, object]):
        calls.append((selected_repository, payload))
        return deepcopy(report)

    monkeypatch.setattr(freeze_module, "_recompute_selection_report", recompute)

    prepared = prepare_frozen_method(repository)
    output = repository / "study/frozen_method.json"

    assert calls == [
        (
            repository.resolve(),
            json.loads(
                (
                    repository
                    / "artifacts/study/development/evaluation/development_selection_input.json"
                ).read_text(encoding="utf-8")
            ),
        )
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == prepared
    assert output.read_bytes() == (
        json.dumps(prepared, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    assert validate_frozen_method(repository) == prepared
    assert (
        prepared["artifact_bindings"]["runtime_lock"]["sha256"]
        == hashlib.sha256(
            (repository / "environments/development-runtime.lock.json").read_bytes()
        ).hexdigest()
    )


def test_prepare_rejects_saver_package_lock_outside_qualification_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    qualification_path = repository / "environments/saver-r.qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["package_lock"]["sha256"] = "0" * 64
    _write_json(qualification_path, qualification)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "mutate SAVER qualification"],
        cwd=repository,
        check=True,
    )
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda selected_repository, payload: deepcopy(report),
    )

    with pytest.raises(PublicationFreezeError, match="SAVER qualification"):
        prepare_frozen_method(repository)


def test_prepare_binds_external_reference_execution_to_exact_measured_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, include_external=True)
    from maskimpute_benchmark.external_reference_development import (
        ValidatedExternalReferenceEvidence,
    )

    checkpoint = _external_checkpoint_payload(repository)
    checkpoint["eligible_dataset_ids"] = ["tung-ipsc-ercc-bulk-replicates"]
    checkpoint["reference_bindings"][0]["dataset_id"] = "tung-ipsc-ercc-bulk-replicates"
    checkpoint["records"][0]["run"]["dataset_id"] = "tung-ipsc-ercc-bulk-replicates"
    checkpoint["checkpoint_sha256"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"}
    )
    external_path = (
        repository
        / "artifacts/study/development/competition-external-reference/checkpoint.json"
    )
    calls: list[Path] = []

    def load_external(selected_repository: Path):
        calls.append(selected_repository)
        return ValidatedExternalReferenceEvidence(
            output_directory=external_path.parent,
            checkpoint_path=external_path,
            checkpoint_file_sha256=hashlib.sha256(
                external_path.read_bytes()
            ).hexdigest(),
            checkpoint=checkpoint,
            dataset_id="tung-ipsc-ercc-bulk-replicates",
            method_ids=("d3impute",),
        )

    monkeypatch.setattr(
        freeze_module,
        "load_external_reference_evidence",
        load_external,
        raising=False,
    )
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )

    prepared = prepare_frozen_method(repository)
    d3impute = next(
        row for row in prepared["method_denominator"] if row["id"] == "d3impute"
    )

    assert d3impute["development_execution_evidence"]["artifact"] == (
        "external_reference_checkpoint"
    )
    assert d3impute["development_execution_evidence"]["execution_track"] == (
        "external_reference"
    )
    assert calls == [repository.resolve(), repository.resolve()]
    assert d3impute["development_execution_evidence"][
        "eligible_dataset_ids_sha256"
    ] == canonical_sha256(["tung-ipsc-ercc-bulk-replicates"])
    assert prepared["artifact_bindings"]["external_reference_checkpoint"][
        "path"
    ].endswith("competition-external-reference/checkpoint.json")
    assert validate_frozen_method(repository) == prepared
    assert calls == [repository.resolve(), repository.resolve(), repository.resolve()]


def test_prepare_propagates_production_external_reference_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, include_external=True)
    from maskimpute_benchmark.external_reference_development import (
        ExternalReferenceDevelopmentError,
    )

    def reject_external(_repository: Path):
        raise ExternalReferenceDevelopmentError("evaluator truth reference rejected")

    monkeypatch.setattr(
        freeze_module,
        "load_external_reference_evidence",
        reject_external,
        raising=False,
    )
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )

    with pytest.raises(
        PublicationFreezeError, match="evaluator truth reference rejected"
    ):
        prepare_frozen_method(repository)


def test_prepare_rejects_selection_report_different_from_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    changed = deepcopy(report)
    changed["pareto_set"] = []
    monkeypatch.setattr(
        freeze_module, "_recompute_selection_report", lambda *_args: changed
    )

    with pytest.raises(PublicationFreezeError, match="selection report.*recomputed"):
        prepare_frozen_method(repository)

    assert not (repository / "study/frozen_method.json").exists()


def test_prepare_rejects_dirty_executable_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    executable = repository / "selection_code.py"
    executable.write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "selection_code.py"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "track selection code"],
        cwd=repository,
        check=True,
    )
    executable.write_text("VERSION = 2\n", encoding="utf-8")
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )

    with pytest.raises(
        PublicationFreezeError,
        match="(?:tracked bytes|clean executable repository)",
    ):
        prepare_frozen_method(repository)

    assert not (repository / "study/frozen_method.json").exists()


def test_prepare_rejects_incomplete_development_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    checkpoint_path = (
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["status"] = "running"
    checkpoint_body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    _write_json(checkpoint_path, checkpoint)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )

    with pytest.raises(PublicationFreezeError, match="checkpoint.*complete"):
        prepare_frozen_method(repository)


def test_prepare_rejects_fixed_evidence_changed_during_recomputation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    protocol = repository / "study/protocol.json"

    def recompute(*_args):
        _write_json(protocol, {"artifact": "changed-after-first-read"})
        return deepcopy(report)

    monkeypatch.setattr(freeze_module, "_recompute_selection_report", recompute)

    with pytest.raises(
        PublicationFreezeError, match="fixed publication evidence changed"
    ):
        prepare_frozen_method(repository)

    assert not (repository / "study/frozen_method.json").exists()


def test_prepare_rejects_runtime_lock_with_extra_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    runtime_path = repository / "environments/development-runtime.lock.json"
    runtime = _runtime_lock_payload(("benchmark", "extra", "magic"))
    _write_json(runtime_path, runtime)
    runtime_sha256 = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    methods_path = repository / "study/methods.json"
    methods = json.loads(methods_path.read_text(encoding="utf-8"))
    for method in methods["methods"]:
        method["environment"]["lock_sha256"] = runtime_sha256
    _write_json(methods_path, methods)
    subprocess.run(
        [
            "git",
            "add",
            "environments/development-runtime.lock.json",
            "study/methods.json",
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "install malformed runtime authority"],
        cwd=repository,
        check=True,
    )
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )

    with pytest.raises(PublicationFreezeError, match="runtime lock IDs"):
        prepare_frozen_method(repository)


def test_runtime_summary_retains_ready_environment_for_terminal_unavailability(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        _runtime_environment_summary,
    )

    runtime_path = tmp_path / "development-runtime.lock.json"
    _write_json(runtime_path, _runtime_lock_payload(("benchmark", "d3impute")))
    runtime_sha256 = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    observed = _method("observed", role="control", integration_reason=None)
    external = _method(
        "d3impute",
        track="external_reference",
        execution_scope="external_reference_only",
        integration_status="unavailable",
        integration_reason="technical_unavailable_development_attempts_deadbeefdeadbeef",
    )
    for row in (observed, external):
        row["environment"]["lock_sha256"] = runtime_sha256

    summary = _runtime_environment_summary(
        runtime_path,
        runtime_sha256,
        {"schema_version": 1, "methods": [observed, external]},
    )

    assert set(summary["environment_inventory_sha256s"]) == {
        "benchmark",
        "d3impute",
    }


def test_prepare_never_overwrites_concurrently_published_frozen_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    real_link = freeze_module.os.link

    def publish_competing_file(source, destination, **kwargs):
        competing = repository / "study/frozen_method.json"
        competing.write_text('{"competing":true}\n', encoding="utf-8")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(freeze_module.os, "link", publish_competing_file)

    with pytest.raises(PublicationFreezeError, match="concurrently"):
        prepare_frozen_method(repository)

    assert (repository / "study/frozen_method.json").read_text(
        encoding="utf-8"
    ) == '{"competing":true}\n'


def test_secure_json_rejects_parent_directory_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    root = tmp_path / "root"
    parent = root / "authority"
    replacement = root / "replacement"
    parent.mkdir(parents=True)
    replacement.mkdir()
    _write_json(parent / "data.json", {"value": "original"})
    _write_json(replacement / "data.json", {"value": "replacement"})
    original = root / "authority-original"
    real_read = freeze_module.os.read
    replaced = False

    def replace_parent(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        if not replaced:
            parent.rename(original)
            replacement.rename(parent)
            replaced = True
        return real_read(descriptor, size)

    monkeypatch.setattr(freeze_module.os, "read", replace_parent)

    with pytest.raises(PublicationFreezeError, match="parent path changed"):
        freeze_module._secure_json(parent / "data.json", "test authority")


def test_pinned_parent_ignores_unrelated_sibling_directory_churn(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "authority" / "data.json").resolve()
    target.parent.mkdir()

    with _pinned_parent(target, "test authority"):
        (tmp_path / "unrelated-sibling").mkdir()


def test_validate_frozen_method_does_not_reopen_validated_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    expected = prepare_frozen_method(repository)
    original_read_bytes = Path.read_bytes

    def reject_frozen_reopen(path: Path) -> bytes:
        if path == repository / "study/frozen_method.json":
            raise AssertionError("validated frozen config path was reopened")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_frozen_reopen)

    assert validate_frozen_method(repository) == expected


def test_clean_phase_validation_does_not_require_ignored_development_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    expected = prepare_frozen_method(repository)
    shutil.rmtree(repository / "artifacts")

    assert validate_frozen_method(repository) == expected


def test_freeze_requires_raw_development_evidence_after_preparation_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    prepare_frozen_method(repository)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-m", "freeze publication authority"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    shutil.rmtree(repository / "artifacts")
    round_dir = repository / "artifacts/study/round-001"

    with pytest.raises(PublicationFreezeError, match="development evidence"):
        freeze_publication_round(repository, round_dir)


def test_publication_freeze_receipts_realistic_ignored_operational_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.study import (
        StudyStateError,
        assert_final_runnable,
        materialize_final,
    )

    repository, report = _repository_fixture(tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    operational_files = {
        "artifacts/envs/scvi-py312/bin/python": b"runtime launcher\n",
        "artifacts/method-sources/magic/magic.py": b"pinned method source\n",
        "artifacts/external/data/tung/source.tsv": b"measured source data\n",
    }
    for relative, raw in operational_files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    prepare_frozen_method(repository)
    subprocess.run(
        ["git", "add", "study/frozen_method.json"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "commit", "-qm", "freeze selected publication method"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "environments/development-runtime.lock.json",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    round_dir = repository / "artifacts/study/round-001"

    frozen = freeze_publication_round(repository, round_dir)

    assert [row["path"] for row in frozen["operational_artifact_roots"]] == [
        "artifacts/envs",
        "artifacts/external",
        "artifacts/method-sources",
        "artifacts/study/development",
    ]
    materialize_final(round_dir, seed_count=2, repo=repository)
    source_path = repository / "artifacts/method-sources/magic/magic.py"
    source_path.write_bytes(b"mutated method source\n")
    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(repository, round_dir)


def test_freeze_rejects_commit_that_changes_more_than_frozen_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    prepare_frozen_method(repository)
    (repository / "unrelated.txt").write_text("changed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "study/frozen_method.json", "unrelated.txt"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "mix authority with unrelated change"],
        cwd=repository,
        check=True,
    )

    with pytest.raises(PublicationFreezeError, match="sole change"):
        freeze_publication_round(repository, repository / "artifacts/study/round-001")


def test_freeze_publication_round_uses_only_fixed_tracked_authorities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    prepare_frozen_method(repository)
    subprocess.run(
        ["git", "add", "study/frozen_method.json"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "commit", "-qm", "freeze selected publication method"],
        cwd=repository,
        check=True,
    )
    observed: dict[str, object] = {}

    def freeze_round(
        repo,
        round_dir,
        config_path,
        protocol_path,
        *,
        environment_path,
        expected_config_sha256,
        expected_protocol_sha256,
        expected_environment_sha256,
        expected_method_commit,
        operational_artifact_roots,
        expected_operational_artifact_roots_sha256,
    ):
        observed.update(
            repo=repo,
            round_dir=round_dir,
            config_path=config_path,
            protocol_path=protocol_path,
            environment_path=environment_path,
            expected_config_sha256=expected_config_sha256,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_environment_sha256=expected_environment_sha256,
            expected_method_commit=expected_method_commit,
            operational_artifact_roots=operational_artifact_roots,
            expected_operational_artifact_roots_sha256=(
                expected_operational_artifact_roots_sha256
            ),
        )
        return {"state": "frozen", "round_id": round_dir.name}

    monkeypatch.setattr(freeze_module, "freeze_round", freeze_round)
    round_dir = repository / "artifacts/study/round-001"

    result = freeze_publication_round(repository, round_dir)

    assert result == {"state": "frozen", "round_id": "round-001"}
    assert observed == {
        "repo": repository.resolve(),
        "round_dir": round_dir,
        "config_path": repository / "study/frozen_method.json",
        "protocol_path": repository / "study/protocol.json",
        "environment_path": repository / "environments/development-runtime.lock.json",
        "expected_config_sha256": hashlib.sha256(
            (repository / "study/frozen_method.json").read_bytes()
        ).hexdigest(),
        "expected_protocol_sha256": hashlib.sha256(
            (repository / "study/protocol.json").read_bytes()
        ).hexdigest(),
        "expected_environment_sha256": hashlib.sha256(
            (repository / "environments/development-runtime.lock.json").read_bytes()
        ).hexdigest(),
        "expected_method_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "operational_artifact_roots": (
            repository.resolve() / "artifacts/study/development",
        ),
        "expected_operational_artifact_roots_sha256": freeze_module.canonical_sha256(
            freeze_module._operational_root_receipts(
                repository.resolve(),
                (repository.resolve() / "artifacts/study/development",),
            )
        ),
    }


def test_freeze_publication_cli_exposes_only_prepare_and_fixed_round_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = Path("scripts/freeze_publication_round.py")
    spec = importlib.util.spec_from_file_location(
        "freeze_publication_round_script", path
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(script, "REPOSITORY_ROOT", repository)
    prepared = {"payload_sha256": "a" * 64}
    monkeypatch.setattr(script, "prepare_frozen_method", lambda repo: prepared)

    assert script.main(["prepare"]) == 0
    assert json.loads(capsys.readouterr().out) == prepared

    observed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        script,
        "freeze_publication_round",
        lambda repo, round_dir: observed.append((repo, round_dir))
        or {"state": "frozen"},
    )
    round_dir = repository / "artifacts/study/round-001"
    assert script.main(["freeze", str(round_dir)]) == 0
    assert observed == [(repository, round_dir)]
    assert json.loads(capsys.readouterr().out) == {"state": "frozen"}
