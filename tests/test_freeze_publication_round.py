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
            "development_result_sha256": hashlib.sha256(b"v28:complete").hexdigest(),
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
                else (
                    "v28_reconstruction_checkpoint"
                    if method_id == "maskimpute"
                    else "base_reconstruction_checkpoint"
                )
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


@lru_cache(maxsize=1)
def _task11_test_module():
    path = Path(__file__).with_name("test_comparator_tuning.py")
    spec = importlib.util.spec_from_file_location("_freeze_task11_factory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _complete_comparator_evidence():
    """Reuse Task 11's exact synthetic direct checkpoint and receipt."""

    from maskimpute_benchmark.comparator_tuning import (
        build_comparator_selection_receipt,
        comparator_selection_projection,
    )

    module = _task11_test_module()
    registry = module.smoke_registry.__wrapped__()
    authority = module.smoke_authority.__wrapped__(registry)
    bound = module.smoke_bound_rows.__wrapped__(registry, authority)
    outcomes = module.complete_smoke_outcomes.__wrapped__(bound)
    fixture = module.complete_selection_fixture.__wrapped__(
        registry,
        authority,
        bound,
        outcomes,
    )
    receipt = build_comparator_selection_receipt(**fixture)
    projection = comparator_selection_projection(receipt)
    assert len(fixture["checkpoint"]["records"]) == 2_896
    return fixture, receipt, projection


@lru_cache(maxsize=None)
def _intrinsic_unavailable_comparator_evidence(method_id: str):
    from maskimpute_benchmark.comparator_tuning import (
        build_comparator_selection_receipt,
        comparator_selection_projection,
    )

    fixture, _receipt, _projection = _complete_comparator_evidence()
    changed = _task11_test_module().selection_fixture_with_intrinsic_unavailable(
        fixture,
        method_id,
    )
    receipt = build_comparator_selection_receipt(**changed)
    return changed, receipt, comparator_selection_projection(receipt)


def _direct_method_registry() -> dict[str, object]:
    registry = json.loads(Path("study/methods.json").read_text(encoding="utf-8"))
    for row in registry["methods"]:
        if row["id"] not in {"observed", "capacity-matched-ae", "maskimpute"}:
            continue
        row["license"] = {"status": "declared", "spdx": "MIT", "notice": None}
        row["citation"] = {"status": "pending", "doi": None, "url": None}
    return registry


def _direct_runtime_summary(registry: dict[str, object]) -> dict[str, object]:
    rows = registry["methods"]
    lock_sha256 = rows[0]["environment"]["lock_sha256"]
    environment_ids = {
        "benchmark",
        *(
            row["environment"]["id"]
            for row in rows
            if row["execution_scope"]
            in {"same_input_required", "external_reference_only"}
            and row["id"] not in {"observed", "capacity-matched-ae", "maskimpute"}
            and row["environment"]["status"] == "ready"
        ),
    }
    return _runtime_summary(
        lock_sha256,
        tuple(sorted(environment_ids)),
    )


def _direct_payload() -> dict[str, object]:
    return _direct_build()


def _direct_build(
    *,
    selection_report: dict[str, object] | None = None,
    candidate_configuration: dict[str, object] | None = None,
    method_registry: dict[str, object] | None = None,
    comparator_fixture: dict[str, object] | None = None,
    comparator_selection_receipt: dict[str, object] | None = None,
    method_execution_evidence: dict[str, dict[str, object]] | None = None,
    selected_calibrator_summary: dict[str, object] | None = None,
    development_stage_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    fixture, receipt, _projection = _complete_comparator_evidence()
    if comparator_fixture is not None:
        fixture = comparator_fixture
    if comparator_selection_receipt is not None:
        receipt = comparator_selection_receipt
    registry = _direct_method_registry() if method_registry is None else method_registry
    runtime_summary = _direct_runtime_summary(registry)
    runtime_lock_sha256 = runtime_summary["lock_file_sha256"]
    bindings = _minimum_artifact_bindings(include_external=True)
    bindings["runtime_lock"]["sha256"] = runtime_lock_sha256
    return build_frozen_method_payload(
        preparation_commit="f" * 40,
        selection_report=(
            _selection_report() if selection_report is None else selection_report
        ),
        candidate_configuration=(
            _configuration()
            if candidate_configuration is None
            else candidate_configuration
        ),
        method_registry=registry,
        comparator_tuning_authority=fixture["authority"],
        comparator_selection_receipt=receipt,
        method_execution_evidence=(
            _execution_evidence(("maskimpute", "d3impute", "sctsi"))
            if method_execution_evidence is None
            else method_execution_evidence
        ),
        selected_calibrator_summary=(
            _calibrator_summary()
            if selected_calibrator_summary is None
            else selected_calibrator_summary
        ),
        ablation_registry=_ablation_registry(),
        runtime_lock_sha256=runtime_lock_sha256,
        runtime_environment_summary=runtime_summary,
        artifact_bindings=bindings,
        development_stage_receipt=(
            _minimum_stage_receipt()
            if development_stage_receipt is None
            else development_stage_receipt
        ),
    )


def test_frozen_payload_binds_selected_comparator_payloads_and_status_denominator() -> (
    None
):
    from maskimpute_benchmark.comparator_tuning import (
        comparator_selection_projection_value,
    )

    payload = _direct_payload()
    _fixture, _receipt, projection = _complete_comparator_evidence()

    assert payload["scheduled_same_input_ids"] == [
        "observed",
        "capacity-matched-ae",
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "scziva",
        "afmf",
        "biaeimpute",
        "sccr",
        "scsdae",
    ]
    assert payload["required_control_ids"] == ["observed", "capacity-matched-ae"]
    assert payload["established_comparator_ids"] == [
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
    ]
    assert payload["modern_core_ids"] == ["scziva", "afmf", "biaeimpute", "sccr"]
    selected = payload["selected_comparator_configurations"]
    assert selected["magic"]["configuration"]["configuration_id"] != (
        "registry-default"
    )
    assert selected["magic"]["configuration"]["payload"]
    assert selected["magic"]["method"]["method_id"] == "magic"
    assert tuple(
        row["method_id"] for row in payload["scheduled_same_input_statuses"]
    ) == tuple(payload["scheduled_same_input_ids"])
    assert len(payload["scheduled_same_input_statuses"]) == 12
    assert payload["comparator_selection"] == comparator_selection_projection_value(
        projection
    )


def _replace_nested(
    value: dict[str, object], path: tuple[object, ...], new: object
) -> None:
    cursor: object = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = new


@pytest.mark.parametrize(
    ("path", "new"),
    (
        (("authority_reference", "authority_revision"), "tampered-authority"),
        (
            (
                "methods",
                "magic",
                "selected_configuration",
                "method",
                "source_revision",
            ),
            "0" * 40,
        ),
        (
            (
                "methods",
                "magic",
                "selected_configuration",
                "configuration",
                "configuration_id",
            ),
            "magic-other",
        ),
        (
            (
                "methods",
                "magic",
                "selected_configuration",
                "configuration",
                "is_upstream_default",
            ),
            False,
        ),
        (
            (
                "methods",
                "magic",
                "selected_configuration",
                "configuration",
                "payload_json",
            ),
            "{}",
        ),
        (("scheduled_tuning_records", 0, "run", "reason"), "tampered_reason"),
        (("readiness", "status"), "blocked"),
    ),
)
def test_frozen_payload_rejects_complete_comparator_receipt_typed_field_tamper(
    path: tuple[object, ...], new: object
) -> None:
    fixture, receipt, _projection = _complete_comparator_evidence()
    changed = deepcopy(receipt)
    _replace_nested(changed, path, new)

    with pytest.raises(PublicationFreezeError, match="comparator selection receipt"):
        _direct_build(
            comparator_fixture=fixture,
            comparator_selection_receipt=changed,
        )


def test_frozen_payload_rejects_unavailable_nonexecution_identity_tamper() -> None:
    fixture, receipt, _projection = _intrinsic_unavailable_comparator_evidence(
        "biaeimpute"
    )
    changed = deepcopy(receipt)
    changed["methods"]["biaeimpute"]["nonexecution_identity"]["method"][
        "source_revision"
    ] = "0" * 40

    with pytest.raises(PublicationFreezeError, match="comparator selection receipt"):
        _direct_build(
            comparator_fixture=fixture,
            comparator_selection_receipt=changed,
        )


def _minimum_artifact_bindings(
    *, include_external: bool = False
) -> dict[str, dict[str, str]]:
    from maskimpute_benchmark.publication_freeze import (
        _publication_artifact_paths,
        _publication_layout_for_active_stage,
    )

    paths = _publication_artifact_paths(
        _publication_layout_for_active_stage("v28"),
        include_external_reference=include_external,
    )
    result = {
        name: {
            "path": path,
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        for name, path in paths.items()
    }
    result["runtime_lock"]["sha256"] = "4" * 64
    result["retained_calibration"]["sha256"] = "7" * 64
    result["ablation_registry"]["sha256"] = "b" * 64
    return result


def _minimum_stage_receipt() -> dict[str, object]:
    bindings = _minimum_artifact_bindings()
    rows = []
    for stage in ("base", "v28"):
        activation = None
        if stage == "v28":
            activation = {
                "version": "v28",
                "trigger": "v28",
                "revision_authority_artifact": "v28_revision",
                "preceding_complete_input_artifact": ("base_selection_complete_input"),
                "preceding_report_artifact": "base_selection_report",
                "selection_input_file_sha256": bindings[
                    "base_selection_complete_input"
                ]["sha256"],
                "selection_result_sha256": hashlib.sha256(b"base:complete").hexdigest(),
                "selection_report_file_sha256": bindings["base_selection_report"][
                    "sha256"
                ],
            }
        rows.append(
            {
                "stage": stage,
                "source_input_artifact": f"{stage}_selection_source_input",
                "complete_input_artifact": f"{stage}_selection_complete_input",
                "report_artifact": f"{stage}_selection_report",
                "evaluation_manifest_artifact": f"{stage}_evaluation_manifest",
                "reconstruction_checkpoint_artifact": (
                    f"{stage}_reconstruction_checkpoint"
                ),
                "orthogonal_manifest_artifact": f"{stage}_orthogonal_manifest",
                "downstream_plan_artifact": f"{stage}_downstream_plan",
                "downstream_manifest_artifact": f"{stage}_downstream_manifest",
                "source_result_sha256": hashlib.sha256(
                    f"{stage}:source".encode()
                ).hexdigest(),
                "complete_result_sha256": hashlib.sha256(
                    f"{stage}:complete".encode()
                ).hexdigest(),
                "downstream_plan_sha256": hashlib.sha256(
                    f"{stage}:plan".encode()
                ).hexdigest(),
                "downstream_manifest_sha256": hashlib.sha256(
                    f"{stage}:manifest".encode()
                ).hexdigest(),
                "reconstruction_tree_sha256": hashlib.sha256(
                    f"{stage}:reconstruction-tree".encode()
                ).hexdigest(),
                "orthogonal_tree_sha256": hashlib.sha256(
                    f"{stage}:orthogonal-tree".encode()
                ).hexdigest(),
                "downstream_tree_sha256": hashlib.sha256(
                    f"{stage}:downstream-tree".encode()
                ).hexdigest(),
                "activation": activation,
            }
        )
    names = sorted(
        f"{stage}_{suffix}"
        for stage in ("base", "v28")
        for suffix in (
            "selection_source_input",
            "selection_complete_input",
            "selection_report",
            "evaluation_manifest",
            "reconstruction_checkpoint",
            "orthogonal_manifest",
            "downstream_plan",
            "downstream_manifest",
        )
    )
    unsigned = {
        "schema_version": 1,
        "active_stage": "v28",
        "revision_versions": ["v28"],
        "stage_order": ["base", "v28"],
        "stages": rows,
        "artifact_names": names,
    }
    return {
        **unsigned,
        "inventory_sha256": canonical_sha256(
            {
                "receipt": unsigned,
                "artifact_bindings": {name: bindings[name] for name in names},
            }
        ),
    }


def _payload(
    *,
    selected_calibrator_summary: dict[str, object] | None = None,
    method_execution_evidence: dict[str, dict[str, object]] | None = None,
    development_stage_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    return _direct_build(
        method_execution_evidence=method_execution_evidence,
        selected_calibrator_summary=selected_calibrator_summary,
        development_stage_receipt=development_stage_receipt,
    )


def test_frozen_method_retains_exact_selection_and_competitor_authority() -> None:
    payload = _payload()

    assert payload["selected_configuration_id"] == "v28-c01-nb-parent-c03"
    assert payload["selected_version"] == "v28"
    assert payload["selected_configuration"] == _configuration()["configuration"]
    assert payload["selection_gate_table"] == _selection_report()["assessments"]
    assert "required_comparator_ids" not in payload
    assert payload["required_control_ids"] == ["observed", "capacity-matched-ae"]
    assert [row["id"] for row in payload["method_denominator"]] == [
        row["id"] for row in _direct_method_registry()["methods"]
    ]
    assert payload["correlation_gene_panel_rule"] == {
        "id": "all-retained-genes-v1",
        "selection": "all_genes_after_only_zero_library_cell_exclusion",
        "gene_filtering": "forbidden",
        "shared_across_methods": True,
    }
    assert (
        payload["runtime_lock_sha256"]
        == _direct_runtime_summary(_direct_method_registry())["lock_file_sha256"]
    )
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
    registry = _direct_method_registry()
    for row in registry["methods"]:
        if row["id"] not in {"observed", "capacity-matched-ae", "maskimpute"}:
            continue
        row["citation"] = {"status": "pending", "doi": None, "url": None}

    payload = _direct_build(method_registry=registry)

    dispositions = {
        row["id"]: row["citation_disposition"] for row in payload["method_denominator"]
    }
    assert dispositions["observed"] == "in_tree_self_citation_no_external_doi"
    assert dispositions["capacity-matched-ae"] == (
        "in_tree_self_citation_no_external_doi"
    )
    assert dispositions["maskimpute"] == "in_tree_self_citation_no_external_doi"
    assert dispositions["magic"] == "verified_external_citation"


def test_in_tree_pending_project_license_remains_a_submission_blocker() -> None:
    registry = _direct_method_registry()
    maskimpute = next(row for row in registry["methods"] if row["id"] == "maskimpute")
    maskimpute["license"] = {
        "status": "pending",
        "spdx": None,
        "notice": "project license requires human approval",
    }
    maskimpute["citation"] = {"status": "pending", "doi": None, "url": None}

    with pytest.raises(PublicationFreezeError, match="maskimpute.*project license"):
        _direct_build(method_registry=registry)


def test_external_method_pending_citation_remains_a_freeze_blocker() -> None:
    registry = _direct_method_registry()
    magic = next(row for row in registry["methods"] if row["id"] == "magic")
    magic["citation"] = {"status": "pending", "doi": None, "url": None}

    with pytest.raises(PublicationFreezeError, match="magic.*external citation"):
        _direct_build(method_registry=registry)


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
    registry = _direct_method_registry()
    payload = _direct_build(method_registry=registry)

    denominator = payload["method_denominator"]
    assert [row["id"] for row in denominator] == [
        row["id"] for row in registry["methods"]
    ]
    by_id = {row["id"]: row for row in denominator}
    assert by_id["biaeimpute"]["claim_required"] is True
    assert by_id["d3impute"]["final_applicability"] == {
        "rule": "matched_bulk_reference_present",
        "non_run_reason": "matched_bulk_reference_absent",
        "required_reference": {
            "kind": "prespecified_matched_bulk_expression",
            "binding": "final_dataset_manifest_external_reference",
            "evaluator_truth_as_reference": "forbidden",
        },
    }
    assert by_id["wedge"]["disposition"] == "historical_not_run"
    assert by_id["scgacl"]["disposition"] == "not_applicable"


def test_frozen_method_rejects_pending_historical_disposition() -> None:
    registry = _direct_method_registry()
    wedge = next(row for row in registry["methods"] if row["id"] == "wedge")
    wedge["integration_status"] = "pending"

    with pytest.raises(PublicationFreezeError, match="wedge.*historical disposition"):
        _direct_build(method_registry=registry)


def test_selected_comparator_rejects_registry_disposition_drift() -> None:
    registry = _direct_method_registry()
    magic_authority = next(row for row in registry["methods"] if row["id"] == "magic")
    magic_authority["integration_status"] = "pending"
    magic_authority["integration_reason"] = "environment_lock_pending"

    with pytest.raises(PublicationFreezeError, match="magic.*identity differs"):
        _direct_build(method_registry=registry)


def test_frozen_method_accepts_explicit_reason_coded_unavailable_comparator() -> None:
    fixture, receipt, _projection = _intrinsic_unavailable_comparator_evidence(
        "biaeimpute"
    )
    payload = _direct_build(
        comparator_fixture=fixture,
        comparator_selection_receipt=receipt,
    )
    method = next(
        row for row in payload["method_denominator"] if row["id"] == "biaeimpute"
    )

    assert method["disposition"] == "explicit_reason_coded_unavailable"
    assert method["selected_comparator_configuration"] is None
    assert (
        method["nonexecution_identity"]
        == payload["unavailable_comparator_nonexecution_identities"]["biaeimpute"]
    )
    assert method["final_applicability"] == {
        "rule": "never",
        "non_run_reason": "technical_unavailable_development_attempts",
        "required_reference": None,
    }


def test_frozen_method_rejects_unavailable_disposition_with_any_completed_run() -> None:
    registry = _direct_method_registry()
    magic = next(row for row in registry["methods"] if row["id"] == "magic")
    magic["environment"] = {
        "id": magic["environment"]["id"],
        "status": "failed",
        "lock_sha256": None,
    }

    with pytest.raises(PublicationFreezeError, match="magic.*identity differs|ready"):
        _direct_build(method_registry=registry)


def test_frozen_method_rejects_scheduled_comparator_execution_summary() -> None:
    evidence = _execution_evidence(("maskimpute", "magic", "d3impute", "sctsi"))

    with pytest.raises(PublicationFreezeError, match="magic.*receipt evidence"):
        _direct_build(method_execution_evidence=evidence)


def test_unavailable_comparator_rejects_registry_method_binding_drift() -> None:
    registry = _direct_method_registry()
    biaeimpute = next(row for row in registry["methods"] if row["id"] == "biaeimpute")
    biaeimpute["integration_status"] = "pending"
    biaeimpute["integration_reason"] = "environment_lock_pending"
    fixture, receipt, _projection = _intrinsic_unavailable_comparator_evidence(
        "biaeimpute"
    )

    with pytest.raises(PublicationFreezeError, match="biaeimpute.*identity differs"):
        _direct_build(
            method_registry=registry,
            comparator_fixture=fixture,
            comparator_selection_receipt=receipt,
        )


def test_frozen_method_rejects_blocking_comparator_terminal_evidence() -> None:
    fixture, receipt, _projection = _complete_comparator_evidence()
    changed = deepcopy(receipt)
    changed["methods"]["biaeimpute"]["terminal_status_counts"] = {
        "infrastructure_error": 192
    }

    with pytest.raises(PublicationFreezeError, match="selection receipt is invalid"):
        _direct_build(
            comparator_fixture=fixture,
            comparator_selection_receipt=changed,
        )


def test_failed_checkpoint_overrides_unbound_registry_availability_claim() -> None:
    registry = _direct_method_registry()
    biaeimpute = next(row for row in registry["methods"] if row["id"] == "biaeimpute")
    biaeimpute["integration_reason"] = "technical_unavailable_arbitrary_unrelated_claim"
    fixture, receipt, _projection = _intrinsic_unavailable_comparator_evidence(
        "biaeimpute"
    )
    payload = _direct_build(
        method_registry=registry,
        comparator_fixture=fixture,
        comparator_selection_receipt=receipt,
    )
    method = next(
        row for row in payload["method_denominator"] if row["id"] == "biaeimpute"
    )

    assert method["registry_integration_reason"] == (
        "technical_unavailable_arbitrary_unrelated_claim"
    )
    assert method["integration_reason"] != method["registry_integration_reason"]


def test_checkpoint_replaces_pre_execution_smoke_disposition() -> None:
    registry = _direct_method_registry()
    magic_authority = next(row for row in registry["methods"] if row["id"] == "magic")
    magic_authority["integration_reason"] = (
        "real_pinned_smoke_passed_environment_lock_pending"
    )

    payload = _direct_build(method_registry=registry)
    magic = next(row for row in payload["method_denominator"] if row["id"] == "magic")

    assert magic["registry_integration_reason"].endswith("lock_pending")
    assert magic["integration_reason"] == "development_selection_receipt_selected"


def test_frozen_method_controls_are_receipt_authoritative() -> None:
    payload = _direct_payload()
    rows = {row["method_id"]: row for row in payload["scheduled_same_input_statuses"]}

    assert rows["observed"]["aggregate_status"] == "completed"
    assert rows["capacity-matched-ae"]["aggregate_status"] == "completed"
    assert rows["observed"]["selected_comparator_configuration"] is None
    assert rows["capacity-matched-ae"]["nonexecution_identity"] is None


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
        _direct_build(
            selection_report=report,
            candidate_configuration=configuration,
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


def _direct_dataset_status_payload(checkpoint: dict[str, object]) -> dict[str, object]:
    descriptors = checkpoint["input_descriptors"]
    body: dict[str, object] = {
        "schema_version": 1,
        "namespace": "dev",
        "status": "completed",
        "completed_count": len(descriptors),
        "failed_count": 0,
        "rows": [
            {"dataset_id": descriptor["dataset_id"], "status": "completed"}
            for descriptor in descriptors
        ],
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_external: bool = False,
) -> tuple[Path, dict[str, object]]:
    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.comparator_tuning import (
        comparator_selection_projection_value,
    )

    monkeypatch.setattr(
        freeze_module,
        "validate_downstream_selection_completeness",
        lambda _repository, _records, binding: {
            "downstream_manifest_file_sha256": binding["manifest_file_sha256"],
            "downstream_manifest_sha256": binding["manifest_sha256"],
            "downstream_plan_sha256": binding["plan_sha256"],
        },
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    direct_fixture, comparator_receipt, comparator_projection = (
        _complete_comparator_evidence()
    )
    direct_checkpoint = deepcopy(direct_fixture["checkpoint"])
    comparator_selection = comparator_selection_projection_value(comparator_projection)
    _materialize_publication_stage_footprint(repository, "base")
    source_core: dict[str, object] = {
        "schema_version": 2,
        "records": [],
        "orthogonal_intervals": [],
        "dataset_manifest_sha256": "7" * 64,
        "count_score_manifest_sha256": "8" * 64,
        "retained_calibration_artifact_sha256": "9" * 64,
        "evaluation_manifest_sha256": "a" * 64,
        "comparator_selection": deepcopy(comparator_selection),
    }
    source_input = {
        **source_core,
        "result_sha256": canonical_sha256(source_core),
    }
    source_path = (
        repository
        / "artifacts/study/development/evaluation/development_selection_input.json"
    )
    _write_json(source_path, source_input)
    downstream_directory = "artifacts/study/development/evaluation/downstream"
    downstream_manifest_path = (
        repository / downstream_directory / "downstream_manifest.json"
    )
    downstream_binding = {
        "path": downstream_directory,
        "source_selection_input_path": source_path.relative_to(repository).as_posix(),
        "source_selection_input_file_sha256": hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest(),
        "source_selection_result_sha256": source_input["result_sha256"],
        "manifest_file_sha256": hashlib.sha256(
            downstream_manifest_path.read_bytes()
        ).hexdigest(),
        "manifest_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
    }
    complete_core = {
        **{
            key: value
            for key, value in source_input.items()
            if key not in {"schema_version", "result_sha256"}
        },
        "schema_version": 4,
        "revision_versions": [],
        "downstream_evidence": downstream_binding,
    }
    selection_input = {
        **complete_core,
        "result_sha256": canonical_sha256(complete_core),
    }
    development_search = json.loads(
        Path("study/development_search.json").read_text(encoding="utf-8")
    )
    selected_row = next(
        row
        for row in development_search["configurations"]
        if row["configuration_id"] == "v27-c03-calibrated-r1-g1"
    )
    report = deepcopy(_selection_report())
    report["selected_configuration"] = selected_row["configuration_id"]
    report["pareto_set"] = [selected_row["configuration_id"]]
    report["assessments"][0]["configuration_id"] = selected_row["configuration_id"]
    report["assessments"][0]["version"] = "v27"
    report["authority_bindings"]["development_result_sha256"] = selection_input[
        "result_sha256"
    ]
    methods = _direct_method_registry()
    methods["methods"] = [
        row
        for row in methods["methods"]
        if row["execution_scope"] != "external_reference_only"
        or (include_external and row["id"] == "d3impute")
    ]
    runtime_ids = tuple(
        sorted(
            {
                "benchmark",
                *(
                    row["id"]
                    for row in methods["methods"]
                    if row["execution_scope"]
                    in {"same_input_required", "external_reference_only"}
                    and row["id"]
                    not in {"observed", "capacity-matched-ae", "maskimpute"}
                    and row["environment"]["status"] == "ready"
                ),
            }
        )
    )
    runtime_lock = _runtime_lock_payload(runtime_ids)
    runtime_raw = (
        json.dumps(runtime_lock, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    runtime_sha256 = hashlib.sha256(runtime_raw).hexdigest()
    for method in methods["methods"]:
        if method["environment"]["status"] == "ready":
            method["environment"]["lock_sha256"] = runtime_sha256
    selection_contract = json.loads(
        Path("study/selection_contract.json").read_text(encoding="utf-8")
    )
    paths = freeze_module._publication_artifact_paths(
        freeze_module._publication_layout_for_active_stage("base"),
        include_external_reference=include_external,
    )
    for name, relative in paths.items():
        payload: object = {"artifact": name}
        if name == "base_selection_source_input":
            payload = source_input
        elif name == "base_selection_complete_input":
            payload = selection_input
        elif name == "base_selection_report":
            payload = report
        elif name == "method_registry":
            payload = methods
        elif name == "selection_contract":
            payload = selection_contract
        elif name == "development_search":
            payload = development_search
        elif name == "comparator_tuning":
            destination = repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(relative), destination)
            continue
        elif name == "comparator_selection":
            payload = comparator_receipt
        elif name == "runtime_lock":
            payload = runtime_lock
        elif name == "base_reconstruction_checkpoint":
            payload = direct_checkpoint
        elif name == "dataset_status":
            payload = _direct_dataset_status_payload(direct_checkpoint)
        elif name == "retained_calibration":
            payload = _retained_calibration_payload()
        elif name == "ablation_registry":
            payload = _ablation_registry()
        elif name == "scaling_panel":
            payload = json.loads(
                Path("study/scaling_panel.json").read_text(encoding="utf-8")
            )
        elif name == "trajectory_panel":
            payload = json.loads(
                Path("study/trajectory_panel.json").read_text(encoding="utf-8")
            )
        elif name == "protocol":
            payload = json.loads(
                Path("study/protocol.json").read_text(encoding="utf-8")
            )
        elif name in {
            "v28_revision",
            "v29_revision",
            "saver_qualification",
            "saver_package_lock",
            "saver_build_receipt",
        }:
            destination = repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(relative), destination)
            continue
        elif name == "external_reference_checkpoint":
            continue
        elif name in {"base_downstream_plan", "base_downstream_manifest"}:
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
    calls: list[tuple[Path, dict[str, object]]] = []

    def recompute(selected_repository: Path, payload: dict[str, object]):
        calls.append((selected_repository, payload))
        return deepcopy(report)

    monkeypatch.setattr(freeze_module, "_recompute_selection_report", recompute)

    prepared = prepare_frozen_method(repository)
    output = repository / "study/frozen_method.json"

    expected_call = (
        repository.resolve(),
        json.loads(
            (
                repository / "artifacts/study/development/evaluation/"
                "development_selection_input-downstream.json"
            ).read_text(encoding="utf-8")
        ),
    )
    assert calls == [expected_call, expected_call]
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
    changed = deepcopy(prepared)
    changed["schema_version"] = True
    unsigned = {key: value for key, value in changed.items() if key != "payload_sha256"}
    changed["payload_sha256"] = canonical_sha256(unsigned)
    _write_json(output, changed)

    with pytest.raises(PublicationFreezeError, match="commit-bound"):
        validate_frozen_method(repository)


def test_frozen_payload_rejects_comparator_receipt_and_selected_payload_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    prepare_frozen_method(repository)

    receipt_path = (
        repository / "artifacts/study/development/evaluation/comparator_selection.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["methods"]["magic"]["selected_configuration"]["configuration"][
        "payload_json"
    ] = "{}"
    _write_json(receipt_path, receipt)

    frozen_path = repository / "study/frozen_method.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen["selected_comparator_configurations"]["magic"]["configuration"]["payload"][
        "diffusion_time"
    ] = 99
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)
    _write_json(frozen_path, frozen)

    with pytest.raises(PublicationFreezeError, match="comparator selection"):
        validate_frozen_method(repository)


def test_publication_freeze_consumes_only_base_selection_complete_input() -> None:
    from maskimpute_benchmark.publication_freeze import (
        _publication_artifact_paths,
        _publication_layout_for_active_stage,
    )
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    paths = _publication_artifact_paths(_publication_layout_for_active_stage("base"))
    stage = development_selection_stage_paths(None)
    assert paths["base_selection_source_input"] == stage.source_selection_input
    assert paths["base_selection_complete_input"] == stage.selection_complete_input
    assert "selection_input" not in paths


def test_prepare_rejects_saver_package_lock_outside_qualification_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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

    repository, report = _repository_fixture(
        tmp_path, monkeypatch, include_external=True
    )
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
    assert calls == [repository.resolve(), repository.resolve()]


def test_prepare_propagates_production_external_reference_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(
        tmp_path, monkeypatch, include_external=True
    )
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
    checkpoint_path = (
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["status"] = "running"
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
    runtime_path = repository / "environments/development-runtime.lock.json"
    runtime = _runtime_lock_payload(("benchmark", "extra", "magic"))
    _write_json(runtime_path, runtime)
    subprocess.run(
        ["git", "add", "environments/development-runtime.lock.json"],
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

    with pytest.raises(PublicationFreezeError, match="runtime lock"):
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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


def test_prepare_accepts_identical_concurrently_published_frozen_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    real_link = freeze_module.os.link

    def publish_identical_file(source, destination, **kwargs):
        study = repository / "study"
        (study / destination).write_bytes((study / source).read_bytes())
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(freeze_module.os, "link", publish_identical_file)

    prepared = prepare_frozen_method(repository)

    assert (
        json.loads(
            (repository / "study/frozen_method.json").read_text(encoding="utf-8")
        )
        == prepared
    )
    assert not list((repository / "study").glob(".frozen_method.json.*.tmp"))


def test_atomic_write_retry_accepts_file_left_after_post_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    path = (tmp_path / "authority" / "frozen_method.json").resolve()
    path.parent.mkdir()
    raw = b'{"schema_version":1}\n'
    real_fsync = freeze_module.os.fsync
    calls = 0

    def fail_directory_fsync_once(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated post-link durability failure")
        real_fsync(descriptor)

    monkeypatch.setattr(freeze_module.os, "fsync", fail_directory_fsync_once)
    with pytest.raises(PublicationFreezeError, match="cannot open.*parent"):
        freeze_module._atomic_write(path, raw)
    assert path.read_bytes() == raw

    monkeypatch.setattr(freeze_module.os, "fsync", real_fsync)
    freeze_module._atomic_write(path, raw)

    assert path.read_bytes() == raw
    assert not list(path.parent.glob(".frozen_method.json.*.tmp"))


def test_atomic_write_accepts_sequential_identical_retry(tmp_path: Path) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    path = (tmp_path / "authority" / "frozen_method.json").resolve()
    path.parent.mkdir()
    raw = b'{"schema_version":1}\n'

    freeze_module._atomic_write(path, raw)
    freeze_module._atomic_write(path, raw)

    assert path.read_bytes() == raw
    assert not list(path.parent.glob(".frozen_method.json.*.tmp"))


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_atomic_write_rejects_unsafe_existing_target(
    tmp_path: Path,
    kind: str,
) -> None:
    import os

    import maskimpute_benchmark.publication_freeze as freeze_module

    path = (tmp_path / "authority" / "frozen_method.json").resolve()
    path.parent.mkdir()
    source = path.parent / "source.json"
    raw = b'{"schema_version":1}\n'
    source.write_bytes(raw)
    if kind == "symlink":
        path.symlink_to(source)
    else:
        os.link(source, path)

    with pytest.raises(PublicationFreezeError, match="different.*concurrently"):
        freeze_module._atomic_write(path, raw)

    assert path.exists()
    assert path.is_symlink() == (kind == "symlink")
    assert not list(path.parent.glob(".frozen_method.json.*.tmp"))


def test_atomic_write_interruption_before_link_leaves_no_target_or_residue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import maskimpute_benchmark.publication_freeze as freeze_module

    path = (tmp_path / "authority" / "frozen_method.json").resolve()
    path.parent.mkdir()

    def interrupt_link(*_args, **_kwargs):
        raise OSError("simulated pre-link interruption")

    monkeypatch.setattr(freeze_module.os, "link", interrupt_link)

    with pytest.raises(PublicationFreezeError, match="cannot open.*parent"):
        freeze_module._atomic_write(path, b'{"schema_version":1}\n')

    assert not os.path.lexists(path)
    assert not list(path.parent.glob(".frozen_method.json.*.tmp"))


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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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


def test_clean_phase_validation_requires_direct_comparator_development_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    prepare_frozen_method(repository)
    shutil.rmtree(repository / "artifacts")

    with pytest.raises(PublicationFreezeError, match="comparator selection receipt"):
        validate_frozen_method(repository)


def test_freeze_requires_raw_development_evidence_after_preparation_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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

    with pytest.raises(
        PublicationFreezeError,
        match="development evidence|comparator selection receipt",
    ):
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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

    repository, report = _repository_fixture(tmp_path, monkeypatch)
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


def _publication_stage_footprint_paths(stage: str) -> dict[str, tuple[str, str]]:
    from maskimpute_benchmark.revisions import (
        development_selection_stage_paths,
        revision_stage_paths,
    )

    through_version = None if stage == "base" else stage
    selection = development_selection_stage_paths(through_version)
    if stage == "base":
        evaluation_manifest = (
            "artifacts/study/development/evaluation/evaluation_manifest.json"
        )
        reconstruction_directory = (
            "artifacts/study/development/competition-reconstruction"
        )
        orthogonal_directory = "artifacts/study/development/evaluation/orthogonal"
    else:
        revision = revision_stage_paths(stage)
        evaluation_manifest = revision.evaluation_manifest
        reconstruction_directory = revision.reconstruction_directory
        orthogonal_directory = revision.orthogonal_directory
    return {
        "source_input": ("file", selection.source_selection_input),
        "complete_input": ("file", selection.selection_complete_input),
        "report": ("file", selection.selection_report),
        "downstream_directory": ("directory", selection.downstream_directory),
        "downstream_plan": (
            "file",
            f"{selection.downstream_directory}/plan.json",
        ),
        "downstream_manifest": (
            "file",
            f"{selection.downstream_directory}/downstream_manifest.json",
        ),
        "evaluation_manifest": ("file", evaluation_manifest),
        "reconstruction_directory": ("directory", reconstruction_directory),
        "reconstruction_checkpoint": (
            "file",
            f"{reconstruction_directory}/checkpoint.json",
        ),
        "orthogonal_directory": ("directory", orthogonal_directory),
        "orthogonal_manifest": (
            "file",
            f"{orthogonal_directory}/orthogonal_outputs.json",
        ),
    }


def _materialize_publication_stage_footprint(
    repository: Path,
    stage: str,
    *,
    only: str | None = None,
) -> None:
    paths = _publication_stage_footprint_paths(stage)
    selected = paths.items() if only is None else ((only, paths[only]),)
    for _name, (kind, relative) in selected:
        path = repository / relative
        if kind == "directory":
            path.mkdir(parents=True, exist_ok=True)
        else:
            _write_json(path, {"stage": stage, "path": relative})


@pytest.mark.parametrize(
    ("active_stage", "stage_order", "revision_versions"),
    (
        ("base", ("base",), ()),
        ("v28", ("base", "v28"), ("v28",)),
        ("v29", ("base", "v28", "v29"), ("v28", "v29")),
    ),
)
def test_publication_stage_resolver_returns_exact_complete_prefix(
    tmp_path: Path,
    active_stage: str,
    stage_order: tuple[str, ...],
    revision_versions: tuple[str, ...],
) -> None:
    from maskimpute_benchmark.publication_freeze import _resolve_publication_stage
    from maskimpute_benchmark.revisions import (
        development_selection_stage_paths,
        revision_stage_paths,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    for stage in stage_order:
        _materialize_publication_stage_footprint(repository, stage)

    layout = _resolve_publication_stage(repository)

    assert layout.active_stage == active_stage
    assert layout.revision_versions == revision_versions
    assert tuple(stage.stage for stage in layout.stages) == stage_order
    for stage in layout.stages:
        through_version = None if stage.stage == "base" else stage.stage
        selection = development_selection_stage_paths(through_version)
        assert stage.source_input == selection.source_selection_input
        assert stage.complete_input == selection.selection_complete_input
        assert stage.report == selection.selection_report
        assert stage.downstream_directory == selection.downstream_directory
        assert stage.downstream_plan == f"{selection.downstream_directory}/plan.json"
        assert stage.downstream_manifest == (
            f"{selection.downstream_directory}/downstream_manifest.json"
        )
        if stage.stage == "base":
            assert stage.revision_authority is None
            assert stage.activation_selection_input is None
            assert stage.activation_selection_report is None
        else:
            revision = revision_stage_paths(stage.stage)
            assert stage.revision_authority == revision.revision_authority
            assert stage.activation_selection_input == (
                revision.activation_selection_input
            )
            assert stage.activation_selection_report == (
                revision.activation_selection_report
            )
            assert stage.evaluation_manifest == revision.evaluation_manifest
            assert stage.reconstruction_directory == (revision.reconstruction_directory)
            assert stage.orthogonal_directory == revision.orthogonal_directory


@pytest.mark.parametrize("newest_stage", ("v28", "v29"))
@pytest.mark.parametrize(
    "component",
    (
        "source_input",
        "complete_input",
        "report",
        "downstream_directory",
        "downstream_manifest",
        "evaluation_manifest",
        "reconstruction_directory",
        "reconstruction_checkpoint",
        "orthogonal_directory",
        "orthogonal_manifest",
    ),
)
def test_newest_footprint_component_never_falls_back(
    tmp_path: Path,
    newest_stage: str,
    component: str,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    prior = ("base",) if newest_stage == "v28" else ("base", "v28")
    for stage in prior:
        _materialize_publication_stage_footprint(repository, stage)
    _materialize_publication_stage_footprint(
        repository,
        newest_stage,
        only=component,
    )

    with pytest.raises(PublicationFreezeError, match="stage.*incomplete|partial"):
        _resolve_publication_stage(repository)


def test_publication_stage_resolver_rejects_unsafe_or_reordered_footprints(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
    )

    broken = tmp_path / "broken"
    broken.mkdir()
    _materialize_publication_stage_footprint(broken, "base")
    report = broken / _publication_stage_footprint_paths("base")["report"][1]
    report.unlink()
    report.symlink_to(broken / "missing-report.json")
    with pytest.raises(PublicationFreezeError, match="symlink|unsafe"):
        _resolve_publication_stage(broken)

    partial = tmp_path / "empty-newest-directory"
    partial.mkdir()
    _materialize_publication_stage_footprint(partial, "base")
    _materialize_publication_stage_footprint(
        partial,
        "v28",
        only="reconstruction_directory",
    )
    with pytest.raises(PublicationFreezeError, match="stage.*incomplete|partial"):
        _resolve_publication_stage(partial)

    reordered = tmp_path / "v29-without-v28"
    reordered.mkdir()
    _materialize_publication_stage_footprint(reordered, "base")
    _materialize_publication_stage_footprint(reordered, "v29")
    with pytest.raises(PublicationFreezeError, match="v28.*incomplete|partial"):
        _resolve_publication_stage(reordered)


def test_unknown_stage_family_suffix_is_rejected(tmp_path: Path) -> None:
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    _materialize_publication_stage_footprint(repository, "base")
    _write_json(
        repository / "artifacts/study/development/evaluation/"
        "development_selection_report-v30.json",
        {"trigger": "freeze_candidate"},
    )

    with pytest.raises(PublicationFreezeError, match="unknown.*stage|stage.*suffix"):
        _resolve_publication_stage(repository)


def _schema_four_stage_chain(
    tmp_path: Path,
    active_stage: str,
    *,
    repository: Path | None = None,
) -> tuple[Path, dict[str, dict[str, object]]]:
    from maskimpute_benchmark.comparator_tuning import (
        comparator_selection_projection_value,
    )
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    order = {
        "base": ("base",),
        "v28": ("base", "v28"),
        "v29": ("base", "v28", "v29"),
    }[active_stage]
    repository = tmp_path / "repository" if repository is None else repository
    repository.mkdir(exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    _fixture, _receipt, projection = _complete_comparator_evidence()
    comparator_selection = comparator_selection_projection_value(projection)
    for index, stage in enumerate(order):
        _materialize_publication_stage_footprint(repository, stage)
        versions = [] if stage == "base" else list(order[1 : index + 1])
        source_core: dict[str, object] = {
            "schema_version": 2 if stage == "base" else 3,
            "dataset_manifest_sha256": "1" * 64,
            "count_score_manifest_sha256": "2" * 64,
            "retained_calibration_artifact_sha256": "3" * 64,
            "evaluation_manifest_sha256": "4" * 64,
            "comparator_selection": deepcopy(comparator_selection),
            "records": [],
            "orthogonal_intervals": [],
        }
        if stage != "base":
            source_core["revision_versions"] = versions
        source = {**source_core, "result_sha256": canonical_sha256(source_core)}
        paths = development_selection_stage_paths(None if stage == "base" else stage)
        _write_json(repository / paths.source_selection_input, source)
        source_file_sha256 = hashlib.sha256(
            (repository / paths.source_selection_input).read_bytes()
        ).hexdigest()
        binding = {
            "path": paths.downstream_directory,
            "source_selection_input_path": paths.source_selection_input,
            "source_selection_input_file_sha256": source_file_sha256,
            "source_selection_result_sha256": source["result_sha256"],
            "manifest_file_sha256": hashlib.sha256(
                (
                    repository / paths.downstream_directory / "downstream_manifest.json"
                ).read_bytes()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(f"{stage}:manifest".encode()).hexdigest(),
            "plan_sha256": hashlib.sha256(f"{stage}:plan".encode()).hexdigest(),
        }
        complete_core = {
            **{
                key: value
                for key, value in source.items()
                if key not in {"schema_version", "revision_versions", "result_sha256"}
            },
            "schema_version": 4,
            "revision_versions": versions,
            "downstream_evidence": binding,
        }
        complete = {
            **complete_core,
            "result_sha256": canonical_sha256(complete_core),
        }
        _write_json(repository / paths.selection_complete_input, complete)

        if stage != active_stage:
            next_stage = order[index + 1]
            report = deepcopy(_selection_report())
            report["trigger"] = next_stage
            report["selected_configuration"] = None
            report["pareto_set"] = []
        else:
            report = deepcopy(_selection_report())
            selected_version = "v27" if stage == "base" else stage
            selected_id = (
                "v27-c03-calibrated-r1-g1"
                if stage == "base"
                else "v28-c01-nb-parent-c03"
                if stage == "v28"
                else "v29-c01-structure-parent-v28-c01"
            )
            report["selected_configuration"] = selected_id
            report["pareto_set"] = [selected_id]
            report["assessments"][0]["configuration_id"] = selected_id
            report["assessments"][0]["version"] = selected_version
        report["authority_bindings"]["development_result_sha256"] = complete[
            "result_sha256"
        ]
        reports[stage] = report
        _write_json(repository / paths.selection_report, report)
    return repository, reports


def _patch_schema_four_stage_replay(
    monkeypatch: pytest.MonkeyPatch,
    repository: Path,
    reports: dict[str, dict[str, object]],
) -> list[str]:
    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.revisions import RevisionActivation, revision_stage_paths

    def stage_for(payload: dict[str, object]) -> str:
        versions = payload["revision_versions"]
        return "base" if versions == [] else versions[-1]

    monkeypatch.setattr(
        freeze_module,
        "validate_downstream_selection_completeness",
        lambda _repository, _records, binding: {
            "downstream_manifest_file_sha256": binding["manifest_file_sha256"],
            "downstream_manifest_sha256": binding["manifest_sha256"],
            "downstream_plan_sha256": binding["plan_sha256"],
        },
    )
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda _repository, payload: deepcopy(reports[stage_for(payload)]),
    )
    calls: list[str] = []

    def activate(_repository: Path, version: str, *, require_clean: bool = True):
        assert require_clean is True
        calls.append(version)
        paths = revision_stage_paths(version)
        selection_input = json.loads(
            (repository / paths.activation_selection_input).read_text(encoding="utf-8")
        )
        return RevisionActivation(
            version=version,
            trigger=version,
            selection_input_path=paths.activation_selection_input,
            selection_input_file_sha256=hashlib.sha256(
                (repository / paths.activation_selection_input).read_bytes()
            ).hexdigest(),
            selection_result_sha256=selection_input["result_sha256"],
            selection_report_path=paths.activation_selection_report,
            selection_report_file_sha256=hashlib.sha256(
                (repository / paths.activation_selection_report).read_bytes()
            ).hexdigest(),
            base_comparator_selection=selection_input["comparator_selection"],
        )

    monkeypatch.setattr(freeze_module, "validate_revision_activation", activate)
    return calls


@pytest.mark.parametrize(
    ("active_stage", "expected_order"),
    (
        ("base", ("base",)),
        ("v28", ("base", "v28")),
        ("v29", ("base", "v28", "v29")),
    ),
)
def test_schema_four_stage_chain_revalidates_exact_activation_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_stage: str,
    expected_order: tuple[str, ...],
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        _resolve_publication_stage,
        _validate_publication_stage_evidence,
    )

    repository, reports = _schema_four_stage_chain(tmp_path, active_stage)
    calls = _patch_schema_four_stage_replay(monkeypatch, repository, reports)

    receipt, active_report = _validate_publication_stage_evidence(
        repository,
        _resolve_publication_stage(repository),
    )

    assert tuple(stage.stage for stage in receipt.stages) == expected_order
    assert calls == list(expected_order[1:])
    assert receipt.stages[0].activation is None
    assert all(stage.activation is not None for stage in receipt.stages[1:])
    assert active_report == reports[active_stage]
    assert active_report["trigger"] == "freeze_candidate"
    for stage in receipt.stages:
        assert stage.source_result_sha256 == stage.source_input["result_sha256"]
        assert stage.complete_result_sha256 == stage.complete_input["result_sha256"]
        assert (
            stage.complete_input["comparator_selection"]
            == stage.source_input["comparator_selection"]
        )


def test_activation_chain_rejects_changed_base_comparator_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
        _validate_publication_stage_evidence,
    )
    from maskimpute_benchmark.revisions import RevisionActivation, revision_stage_paths

    repository, reports = _schema_four_stage_chain(tmp_path, "v28")
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)
    paths = revision_stage_paths("v28")
    selection_input = json.loads(
        (repository / paths.activation_selection_input).read_text(encoding="utf-8")
    )
    changed = deepcopy(selection_input["comparator_selection"])
    changed["ready_comparison_population_ids"] = ["observed"]
    monkeypatch.setattr(
        freeze_module,
        "validate_revision_activation",
        lambda *_args, **_kwargs: RevisionActivation(
            version="v28",
            trigger="v28",
            selection_input_path=paths.activation_selection_input,
            selection_input_file_sha256=hashlib.sha256(
                (repository / paths.activation_selection_input).read_bytes()
            ).hexdigest(),
            selection_result_sha256=selection_input["result_sha256"],
            selection_report_path=paths.activation_selection_report,
            selection_report_file_sha256=hashlib.sha256(
                (repository / paths.activation_selection_report).read_bytes()
            ).hexdigest(),
            base_comparator_selection=changed,
        ),
    )

    with pytest.raises(PublicationFreezeError, match="comparator"):
        _validate_publication_stage_evidence(
            repository,
            _resolve_publication_stage(repository),
        )


def test_activation_chain_rejects_wrong_retained_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
        _validate_publication_stage_evidence,
    )
    from maskimpute_benchmark.revisions import RevisionActivation, revision_stage_paths

    repository, reports = _schema_four_stage_chain(tmp_path, "v28")
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)
    paths = revision_stage_paths("v28")
    monkeypatch.setattr(
        freeze_module,
        "validate_revision_activation",
        lambda *_args, **_kwargs: RevisionActivation(
            version="v28",
            trigger="v28",
            selection_input_path=paths.activation_selection_input,
            selection_input_file_sha256="0" * 64,
            selection_result_sha256="0" * 64,
            selection_report_path=paths.activation_selection_report,
            selection_report_file_sha256="0" * 64,
            base_comparator_selection=json.loads(
                (repository / paths.activation_selection_input).read_text(
                    encoding="utf-8"
                )
            )["comparator_selection"],
        ),
    )

    with pytest.raises(PublicationFreezeError, match="activation.*retained|hash"):
        _validate_publication_stage_evidence(
            repository,
            _resolve_publication_stage(repository),
        )


def test_schema_four_stage_rejects_nonpromoted_complete_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
        _validate_publication_stage_evidence,
    )
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    repository, reports = _schema_four_stage_chain(tmp_path, "base")
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)
    complete_path = (
        repository / development_selection_stage_paths(None).selection_complete_input
    )
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["schema_version"] = 2
    complete.pop("revision_versions")
    complete.pop("downstream_evidence")
    complete["result_sha256"] = canonical_sha256(
        {key: value for key, value in complete.items() if key != "result_sha256"}
    )
    _write_json(complete_path, complete)

    with pytest.raises(PublicationFreezeError, match="schema 4|selection-complete"):
        _validate_publication_stage_evidence(
            repository,
            _resolve_publication_stage(repository),
        )


def test_no_selection_fallback_from_downgrade_v29(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
        _validate_publication_stage_evidence,
    )
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    repository, reports = _schema_four_stage_chain(tmp_path, "v29")
    reports["v29"]["trigger"] = "downgrade_claim"
    terminal = development_selection_stage_paths("v29").selection_report
    _write_json(repository / terminal, reports["v29"])
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)

    with pytest.raises(PublicationFreezeError, match="downgrade|terminal"):
        _validate_publication_stage_evidence(
            repository,
            _resolve_publication_stage(repository),
        )


def test_schema_four_stage_rejects_selected_version_older_than_active_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
        _validate_publication_stage_evidence,
    )
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    repository, reports = _schema_four_stage_chain(tmp_path, "v29")
    reports["v29"]["selected_configuration"] = "v28-c01-nb-parent-c03"
    reports["v29"]["assessments"][0]["configuration_id"] = "v28-c01-nb-parent-c03"
    reports["v29"]["assessments"][0]["version"] = "v28"
    terminal = development_selection_stage_paths("v29").selection_report
    _write_json(repository / terminal, reports["v29"])
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)

    with pytest.raises(PublicationFreezeError, match="selected version|active stage"):
        _validate_publication_stage_evidence(
            repository,
            _resolve_publication_stage(repository),
        )


def test_schema_four_stage_rejects_core_file_changed_during_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.publication_freeze import (
        PublicationFreezeError,
        _resolve_publication_stage,
        _validate_publication_stage_evidence,
    )
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    repository, reports = _schema_four_stage_chain(tmp_path, "base")
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)
    complete_path = (
        repository / development_selection_stage_paths(None).selection_complete_input
    )

    def mutate_after_recompute(_repository: Path, payload: dict[str, object]):
        changed = deepcopy(payload)
        changed["dataset_manifest_sha256"] = "f" * 64
        changed["result_sha256"] = canonical_sha256(
            {key: value for key, value in changed.items() if key != "result_sha256"}
        )
        _write_json(complete_path, changed)
        return deepcopy(reports["base"])

    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        mutate_after_recompute,
    )

    with pytest.raises(PublicationFreezeError, match="changed during.*replay"):
        _validate_publication_stage_evidence(
            repository,
            _resolve_publication_stage(repository),
        )


@pytest.mark.parametrize(
    ("active_stage", "stage_order"),
    (
        ("base", ("base",)),
        ("v28", ("base", "v28")),
        ("v29", ("base", "v28", "v29")),
    ),
)
def test_dynamic_artifact_paths_are_exact_and_stage_qualified(
    tmp_path: Path,
    active_stage: str,
    stage_order: tuple[str, ...],
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        _publication_artifact_paths,
        _resolve_publication_stage,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    for stage in stage_order:
        _materialize_publication_stage_footprint(repository, stage)
    layout = _resolve_publication_stage(repository)

    paths = _publication_artifact_paths(layout)

    common = {
        "runtime_lock",
        "method_registry",
        "selection_contract",
        "development_search",
        "comparator_tuning",
        "comparator_selection",
        "v28_revision",
        "v29_revision",
        "ablation_registry",
        "scaling_panel",
        "trajectory_panel",
        "protocol",
        "saver_qualification",
        "saver_package_lock",
        "saver_build_receipt",
        "dataset_status",
        "count_score_manifest",
        "retained_calibration",
    }
    suffixes = {
        "selection_source_input",
        "selection_complete_input",
        "selection_report",
        "evaluation_manifest",
        "reconstruction_checkpoint",
        "orthogonal_manifest",
        "downstream_plan",
        "downstream_manifest",
    }
    expected_stage_names = {
        f"{stage}_{suffix}" for stage in stage_order for suffix in suffixes
    }
    assert set(paths) == common | expected_stage_names
    assert paths["v28_revision"] == "study/v28_revision.json"
    assert paths["v29_revision"] == "study/v29_revision.json"
    assert paths["scaling_panel"] == "study/scaling_panel.json"
    assert paths["trajectory_panel"] == "study/trajectory_panel.json"
    assert not {
        "selection_input",
        "selection_report",
        "evaluation_manifest",
        "reconstruction_checkpoint",
    }.intersection(paths)

    external = _publication_artifact_paths(
        layout,
        include_external_reference=True,
    )
    assert set(external) == set(paths) | {"external_reference_checkpoint"}


def test_dynamic_artifact_trajectory_authority_is_not_stage_presence(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.publication_freeze import _resolve_publication_stage

    repository = tmp_path / "repository"
    repository.mkdir()
    _materialize_publication_stage_footprint(repository, "base")
    _write_json(repository / "study/trajectory_panel.json", {"schema_version": 1})

    assert _resolve_publication_stage(repository).active_stage == "base"


def test_dynamic_artifact_bindings_reject_duplicate_normalized_paths() -> None:
    from maskimpute_benchmark.publication_freeze import _artifact_bindings

    with pytest.raises(PublicationFreezeError, match="duplicate.*path"):
        _artifact_bindings(
            {
                "first": {"path": "study/protocol.json", "sha256": "1" * 64},
                "second": {"path": "study//protocol.json", "sha256": "2" * 64},
            }
        )


def test_tree_receipt_changes_for_unreferenced_file(tmp_path: Path) -> None:
    from maskimpute_benchmark.publication_freeze import _closed_stage_tree_sha256

    repository = tmp_path / "repository"
    tree = repository / "tree"
    tree.mkdir(parents=True)
    (tree / "retained.bin").write_bytes(b"retained")
    before = _closed_stage_tree_sha256(repository, "tree")

    (tree / "unreferenced.bin").write_bytes(b"new evidence")
    after = _closed_stage_tree_sha256(repository, "tree")

    assert before != after


def test_tree_receipt_hashes_large_files_as_bounded_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.publication_freeze import _stream_tree_file_sha256

    path = tmp_path / "large.bin"
    path.write_bytes(b"x" * (2 * 1024 * 1024 + 17))
    original_sha256 = hashlib.sha256
    update_sizes: list[int] = []

    class BoundedHasher:
        def __init__(self) -> None:
            self._delegate = original_sha256()

        def update(self, chunk: bytes) -> None:
            assert len(chunk) <= 1024 * 1024
            update_sizes.append(len(chunk))
            self._delegate.update(chunk)

        def hexdigest(self) -> str:
            return self._delegate.hexdigest()

    monkeypatch.setattr(freeze_module.hashlib, "sha256", BoundedHasher)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        digest, byte_count = _stream_tree_file_sha256(
            descriptor,
            path.stat().st_size,
        )
    finally:
        os.close(descriptor)

    assert byte_count == path.stat().st_size
    assert digest == original_sha256(path.read_bytes()).hexdigest()
    assert update_sizes == [1024 * 1024, 1024 * 1024, 17]


def test_tree_receipt_directory_swap_cannot_escape_pinned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    import maskimpute_benchmark.publication_freeze as freeze_module
    from maskimpute_benchmark.publication_freeze import (
        _snapshot_closed_stage_tree,
    )

    repository = tmp_path / "repository"
    container = repository / "container"
    root = container / "tree"
    competing_container = repository / "competing-container"
    competing_root = competing_container / "tree"
    held_container = repository / "container-held"
    root.mkdir(parents=True)
    competing_root.mkdir(parents=True)
    (root / "safe.bin").write_bytes(b"safe")
    (competing_root / "escaped.bin").write_bytes(b"escaped")
    root_inode = root.stat().st_ino
    real_scandir = freeze_module.os.scandir
    real_lstat = freeze_module.os.lstat
    swapped = False
    path_walk = False

    def swap_in() -> None:
        nonlocal swapped
        container.rename(held_container)
        competing_container.rename(container)
        swapped = True

    def swap_out() -> None:
        nonlocal swapped
        container.rename(competing_container)
        held_container.rename(container)
        swapped = False

    def racing_scandir(target):
        nonlocal path_walk
        descriptor_walk = type(target) is int and os.fstat(target).st_ino == root_inode
        pathname_walk = not isinstance(target, int) and Path(target) == root
        if not swapped and (descriptor_walk or pathname_walk):
            swap_in()
            iterator = real_scandir(target)
            if descriptor_walk:
                swap_out()
            else:
                path_walk = True
            return iterator
        return real_scandir(target)

    def restoring_lstat(path):
        if path_walk and swapped and Path(path) == root:
            swap_out()
        return real_lstat(path)

    monkeypatch.setattr(freeze_module.os, "scandir", racing_scandir)
    monkeypatch.setattr(freeze_module.os, "lstat", restoring_lstat)
    try:
        rows = _snapshot_closed_stage_tree(root)
    finally:
        if swapped:
            swap_out()

    paths = {row["path"] for row in rows}
    assert "safe.bin" in paths
    assert "escaped.bin" not in paths


@pytest.mark.parametrize("kind", ("symlink", "fifo", "socket", "hardlink"))
def test_tree_receipt_rejects_links_and_special_files(
    tmp_path: Path,
    kind: str,
) -> None:
    import os
    import socket

    from maskimpute_benchmark.publication_freeze import _closed_stage_tree_sha256

    repository = tmp_path / "repository"
    tree = repository / "tree"
    tree.mkdir(parents=True)
    retained = tree / "retained.bin"
    retained.write_bytes(b"retained")
    special = tree / "special"
    opened_socket = None
    if kind == "symlink":
        special.symlink_to(retained)
    elif kind == "fifo":
        special.parent.mkdir(parents=True, exist_ok=True)
        os.mkfifo(special)
    elif kind == "socket":
        opened_socket = socket.socket(socket.AF_UNIX)
        opened_socket.bind(str(special))
    else:
        os.link(retained, special)
    try:
        with pytest.raises(PublicationFreezeError, match="symlink|special|unique"):
            _closed_stage_tree_sha256(repository, "tree")
    finally:
        if opened_socket is not None:
            opened_socket.close()


def _dynamic_bindings_for_layout(
    repository: Path,
    layout: object,
) -> dict[str, dict[str, str]]:
    from maskimpute_benchmark.publication_freeze import _publication_artifact_paths

    paths = _publication_artifact_paths(layout)
    for name, relative in paths.items():
        path = repository / relative
        if not path.exists():
            _write_json(path, {"artifact": name})
    return {
        name: {
            "path": relative,
            "sha256": hashlib.sha256((repository / relative).read_bytes()).hexdigest(),
        }
        for name, relative in paths.items()
    }


def test_stage_receipt_closes_exact_inventory_and_activation_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        _build_development_stage_receipt,
        _resolve_publication_stage,
        _validate_development_stage_receipt,
        _validate_publication_stage_evidence,
    )

    repository, reports = _schema_four_stage_chain(tmp_path, "v29")
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)
    layout = _resolve_publication_stage(repository)
    evidence, _report = _validate_publication_stage_evidence(repository, layout)
    bindings = _dynamic_bindings_for_layout(repository, layout)

    receipt = _build_development_stage_receipt(
        repository,
        layout,
        evidence,
        bindings,
    )

    stage_names = sorted(
        f"{stage}_{suffix}"
        for stage in ("base", "v28", "v29")
        for suffix in (
            "selection_source_input",
            "selection_complete_input",
            "selection_report",
            "evaluation_manifest",
            "reconstruction_checkpoint",
            "orthogonal_manifest",
            "downstream_plan",
            "downstream_manifest",
        )
    )
    assert receipt["schema_version"] == 1
    assert receipt["active_stage"] == "v29"
    assert receipt["revision_versions"] == ["v28", "v29"]
    assert receipt["stage_order"] == ["base", "v28", "v29"]
    assert receipt["artifact_names"] == stage_names
    assert receipt["stages"][0]["activation"] is None
    assert receipt["stages"][1]["activation"] == {
        "version": "v28",
        "trigger": "v28",
        "revision_authority_artifact": "v28_revision",
        "preceding_complete_input_artifact": "base_selection_complete_input",
        "preceding_report_artifact": "base_selection_report",
        "selection_input_file_sha256": evidence.stages[
            1
        ].activation.selection_input_file_sha256,
        "selection_result_sha256": evidence.stages[
            1
        ].activation.selection_result_sha256,
        "selection_report_file_sha256": evidence.stages[
            1
        ].activation.selection_report_file_sha256,
    }
    assert all(
        isinstance(stage[key], str)
        and len(stage[key]) == 64
        and set(stage[key]) <= set("0123456789abcdef")
        for stage in receipt["stages"]
        for key in (
            "reconstruction_tree_sha256",
            "orthogonal_tree_sha256",
            "downstream_tree_sha256",
        )
    )
    assert _validate_development_stage_receipt(receipt, layout, bindings) == receipt


@pytest.mark.parametrize(
    "mutation",
    ("artifact_names", "stage_order", "activation", "inventory_sha256"),
)
def test_stage_receipt_rejects_any_inventory_or_chain_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        _build_development_stage_receipt,
        _resolve_publication_stage,
        _validate_development_stage_receipt,
        _validate_publication_stage_evidence,
    )

    repository, reports = _schema_four_stage_chain(tmp_path, "v29")
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)
    layout = _resolve_publication_stage(repository)
    evidence, _report = _validate_publication_stage_evidence(repository, layout)
    bindings = _dynamic_bindings_for_layout(repository, layout)
    receipt = _build_development_stage_receipt(
        repository,
        layout,
        evidence,
        bindings,
    )
    changed = deepcopy(receipt)
    if mutation == "artifact_names":
        changed["artifact_names"] = changed["artifact_names"][:-1]
    elif mutation == "stage_order":
        changed["stage_order"] = list(reversed(changed["stage_order"]))
    elif mutation == "activation":
        changed["stages"][1]["activation"]["preceding_report_artifact"] = (
            "v28_selection_report"
        )
    else:
        changed["inventory_sha256"] = "0" * 64

    with pytest.raises(
        PublicationFreezeError, match="stage receipt|inventory|activation"
    ):
        _validate_development_stage_receipt(changed, layout, bindings)


@pytest.mark.parametrize(
    "field",
    (
        "selection_input_file_sha256",
        "selection_result_sha256",
        "selection_report_file_sha256",
    ),
)
def test_stage_receipt_rejects_coherently_rehashed_activation_drift(
    field: str,
) -> None:
    from maskimpute_benchmark.publication_freeze import (
        _publication_layout_for_active_stage,
        _stage_receipt_inventory_sha256,
        _validate_development_stage_receipt,
    )

    layout = _publication_layout_for_active_stage("v28")
    bindings = _minimum_artifact_bindings()
    changed = deepcopy(_minimum_stage_receipt())
    activation = changed["stages"][1]["activation"]
    assert isinstance(activation, dict)
    activation[field] = "f" * 64
    unsigned = {
        key: value for key, value in changed.items() if key != "inventory_sha256"
    }
    changed["inventory_sha256"] = _stage_receipt_inventory_sha256(
        unsigned,
        changed["artifact_names"],
        bindings,
    )

    with pytest.raises(PublicationFreezeError, match="activation.*differs"):
        _validate_development_stage_receipt(changed, layout, bindings)


def test_frozen_method_rejects_coherently_rehashed_active_stage_result() -> None:
    from maskimpute_benchmark.publication_freeze import (
        _stage_receipt_inventory_sha256,
    )

    bindings = _minimum_artifact_bindings()
    changed = deepcopy(_minimum_stage_receipt())
    changed["stages"][-1]["complete_result_sha256"] = "f" * 64
    unsigned = {
        key: value for key, value in changed.items() if key != "inventory_sha256"
    }
    changed["inventory_sha256"] = _stage_receipt_inventory_sha256(
        unsigned,
        changed["artifact_names"],
        bindings,
    )

    with pytest.raises(PublicationFreezeError, match="active stage.*result"):
        _payload(development_stage_receipt=changed)


def test_v29_configuration_is_loaded_from_exact_tracked_revision_authority() -> None:
    from maskimpute_benchmark.publication_freeze import _candidate_configuration

    repository = Path.cwd()
    development_search = json.loads(
        (repository / "study/development_search.json").read_text(encoding="utf-8")
    )
    v29 = json.loads(
        (repository / "study/v29_revision.json").read_text(encoding="utf-8")
    )

    selected = _candidate_configuration(
        repository,
        v29["configuration_id"],
        development_search,
        ("v28", "v29"),
    )

    assert selected == {
        "configuration_id": v29["configuration_id"],
        "version": "v29",
        "configuration": v29["configuration"],
        "configuration_sha256": v29["configuration_sha256"],
    }


def test_v29_configuration_rejects_parent_drift_or_duplicate_selected_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module
    from dataclasses import replace

    from maskimpute_benchmark.publication_freeze import _candidate_configuration
    from maskimpute_benchmark.revisions import load_revision_spec

    repository = Path.cwd()
    development_search = json.loads(
        (repository / "study/development_search.json").read_text(encoding="utf-8")
    )
    v28 = load_revision_spec(repository, "v28")
    v29 = load_revision_spec(repository, "v29")

    monkeypatch.setattr(
        freeze_module,
        "load_revision_spec",
        lambda _repository, version, *, require_clean=True: (
            v28
            if version == "v28"
            else replace(v29, parent_configuration_sha256="0" * 64)
        ),
    )
    with pytest.raises(PublicationFreezeError, match="parent.*binding"):
        _candidate_configuration(
            repository,
            v29.configuration_id,
            development_search,
            ("v28", "v29"),
        )

    monkeypatch.setattr(freeze_module, "load_revision_spec", load_revision_spec)
    duplicated = deepcopy(development_search)
    duplicate_row = deepcopy(duplicated["configurations"][0])
    duplicate_row["configuration_id"] = v29.configuration_id
    duplicate_row["version"] = "v29"
    duplicated["configurations"].append(duplicate_row)
    with pytest.raises(PublicationFreezeError, match="uniquely present|collides"):
        _candidate_configuration(
            repository,
            v29.configuration_id,
            duplicated,
            ("v28", "v29"),
        )


def _selected_stage_checkpoint(
    rows: list[tuple[str, str, str]],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "planned_run_count": len(rows),
        "status": "completed",
        "records": [
            {
                "run": {
                    "method_id": method_id,
                    "dataset_id": "dataset-001",
                    "configuration_id": configuration_id,
                    "configuration_sha256": configuration_sha256,
                    "status": "completed",
                    "reason": None,
                },
                "metrics": [],
            }
            for method_id, configuration_id, configuration_sha256 in rows
        ],
    }
    return {**body, "checkpoint_sha256": canonical_sha256(body)}


def _direct_selected_stage_checkpoint(
    selected_configuration: dict[str, object],
) -> dict[str, object]:
    fixture, _receipt, _projection = _complete_comparator_evidence()
    base = fixture["checkpoint"]
    candidate_rows = [
        deepcopy(record)
        for record in base["records"]
        if record["run"]["identity"]["method"]["method_id"] == "maskimpute"
        and record["run"]["identity"]["configuration_kind"] == "candidate_search"
    ]
    original_id = candidate_rows[0]["run"]["identity"]["configuration_id"]
    records = [
        record
        for record in candidate_rows
        if record["run"]["identity"]["configuration_id"] == original_id
    ]
    assert len(records) == 48
    for record in records:
        identity = record["run"]["identity"]
        identity["configuration_id"] = selected_configuration["configuration_id"]
        identity["configuration_payload"] = deepcopy(
            selected_configuration["configuration"]
        )
    return {
        "schema_version": 1,
        "identity_mode": "direct-v1",
        "authority_revision": base["authority_revision"],
        "plan_snapshot": base["plan_snapshot"],
        "input_descriptors": base["input_descriptors"],
        "planned_run_count": 48,
        "status": "completed",
        "records": records,
    }


def test_direct_selected_stage_execution_reads_nested_candidate_identity() -> None:
    from maskimpute_benchmark.publication_freeze import _active_execution_evidence

    fixture, _receipt, _projection = _complete_comparator_evidence()
    selected_configuration = _configuration()
    revision = _direct_selected_stage_checkpoint(selected_configuration)

    evidence = _active_execution_evidence(
        fixture["checkpoint"],
        active_stage="v28",
        selected_stage_checkpoint=revision,
        selected_configuration=selected_configuration,
        eligible_dataset_ids=tuple(
            sorted(
                row["dataset_id"] for row in fixture["checkpoint"]["input_descriptors"]
            )
        ),
    )

    assert set(evidence) == {"maskimpute"}
    assert evidence["maskimpute"]["artifact"] == "v28_reconstruction_checkpoint"
    assert evidence["maskimpute"]["attempted_run_count"] == 48


def test_selected_stage_execution_swaps_only_maskimpute_evidence() -> None:
    from maskimpute_benchmark.publication_freeze import _active_execution_evidence

    base = _selected_stage_checkpoint(
        [
            ("observed", "registry-default", "1" * 64),
            ("capacity-matched-ae", "registry-default", "2" * 64),
            ("maskimpute", "v27-c03-calibrated-r1-g1", "3" * 64),
            ("magic", "registry-default", "4" * 64),
        ]
    )
    selected_configuration = {
        "configuration_id": "v29-c01-structure-parent-v28-c01",
        "version": "v29",
        "configuration": {"method_version": "v29"},
        "configuration_sha256": "5" * 64,
    }
    revision = _selected_stage_checkpoint(
        [
            (
                "maskimpute",
                selected_configuration["configuration_id"],
                selected_configuration["configuration_sha256"],
            )
        ]
    )

    evidence = _active_execution_evidence(
        base,
        active_stage="v29",
        selected_stage_checkpoint=revision,
        selected_configuration=selected_configuration,
        eligible_dataset_ids=("dataset-001",),
    )

    assert evidence["maskimpute"]["artifact"] == "v29_reconstruction_checkpoint"
    assert (
        evidence["maskimpute"]["checkpoint_payload_sha256"]
        == revision["checkpoint_sha256"]
    )
    for method_id in ("observed", "capacity-matched-ae", "magic"):
        assert evidence[method_id]["artifact"] == "base_reconstruction_checkpoint"
        assert (
            evidence[method_id]["checkpoint_payload_sha256"]
            == base["checkpoint_sha256"]
        )


@pytest.mark.parametrize(
    "revision_rows",
    (
        [("maskimpute", "v28-other", "5" * 64)],
        [
            ("maskimpute", "v28-selected", "5" * 64),
            ("maskimpute", "v28-other", "5" * 64),
        ],
        [("maskimpute", "v28-selected", "0" * 64)],
        [("magic", "v28-selected", "5" * 64)],
    ),
)
def test_selected_stage_execution_rejects_ambiguous_or_comparator_rows(
    revision_rows: list[tuple[str, str, str]],
) -> None:
    from maskimpute_benchmark.publication_freeze import _active_execution_evidence

    base = _selected_stage_checkpoint(
        [
            ("observed", "registry-default", "1" * 64),
            ("maskimpute", "v27-selected", "2" * 64),
            ("magic", "registry-default", "3" * 64),
        ]
    )
    selected_configuration = {
        "configuration_id": "v28-selected",
        "version": "v28",
        "configuration": {"method_version": "v28"},
        "configuration_sha256": "5" * 64,
    }
    revision = _selected_stage_checkpoint(revision_rows)

    with pytest.raises(
        PublicationFreezeError,
        match="selected.*configuration|only.*[Mm]ask[Ii]mpute|unique",
    ):
        _active_execution_evidence(
            base,
            active_stage="v28",
            selected_stage_checkpoint=revision,
            selected_configuration=selected_configuration,
            eligible_dataset_ids=("dataset-001",),
        )


def test_base_comparator_denominator_remains_unchanged_for_base_selection() -> None:
    from maskimpute_benchmark.publication_freeze import _active_execution_evidence

    base = _selected_stage_checkpoint(
        [
            ("observed", "registry-default", "1" * 64),
            ("maskimpute", "v27-selected", "2" * 64),
            ("magic", "registry-default", "3" * 64),
        ]
    )
    evidence = _active_execution_evidence(
        base,
        active_stage="base",
        selected_stage_checkpoint=None,
        selected_configuration={
            "configuration_id": "v27-selected",
            "version": "v27",
            "configuration": {"method_version": "v27"},
            "configuration_sha256": "2" * 64,
        },
        eligible_dataset_ids=("dataset-001",),
    )

    assert set(evidence) == {"observed", "maskimpute", "magic"}
    assert {row["artifact"] for row in evidence.values()} == {
        "base_reconstruction_checkpoint"
    }


def test_stage_receipt_payload_is_retained_by_outer_payload_checksum() -> None:
    payload = _payload()

    assert payload["development_stage_receipt"] == _minimum_stage_receipt()
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    assert payload["payload_sha256"] == canonical_sha256(unsigned)


def test_stage_receipt_rejects_maskimpute_evidence_from_another_stage() -> None:
    evidence = _execution_evidence(("maskimpute", "d3impute", "sctsi"))
    evidence["maskimpute"]["artifact"] = "base_reconstruction_checkpoint"

    with pytest.raises(PublicationFreezeError, match="selected stage"):
        _payload(method_execution_evidence=evidence)


@pytest.mark.parametrize("active_stage", ("v28", "v29"))
def test_revision_prepare_and_clean_validate_exact_active_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_stage: str,
) -> None:
    repository, _base_report = _repository_fixture(tmp_path, monkeypatch)
    repository, reports = _schema_four_stage_chain(
        tmp_path,
        active_stage,
        repository=repository,
    )
    direct_fixture, _receipt, _projection = _complete_comparator_evidence()
    _write_json(
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json",
        direct_fixture["checkpoint"],
    )
    versions = ("v28",) if active_stage == "v28" else ("v28", "v29")
    for version in versions:
        revision = json.loads(
            (repository / f"study/{version}_revision.json").read_text(encoding="utf-8")
        )
        _write_json(
            repository
            / f"artifacts/study/development/competition-{version}-revision/checkpoint.json",
            _direct_selected_stage_checkpoint(
                {
                    "configuration_id": revision["configuration_id"],
                    "version": version,
                    "configuration": revision["configuration"],
                    "configuration_sha256": revision["configuration_sha256"],
                }
            ),
        )
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)

    prepared = prepare_frozen_method(repository)

    assert prepared["development_stage_receipt"]["active_stage"] == active_stage
    assert prepared["development_stage_receipt"]["revision_versions"] == list(versions)
    assert prepared["selected_version"] == active_stage
    denominator = {row["id"]: row for row in prepared["method_denominator"]}
    assert (
        denominator["maskimpute"]["development_execution_evidence"]["artifact"]
        == f"{active_stage}_reconstruction_checkpoint"
    )
    assert denominator["observed"]["development_execution_evidence"] is None
    assert (
        denominator["observed"]["scheduled_same_input_status"]["aggregate_status"]
        == "completed"
    )
    assert validate_frozen_method(repository) == prepared


def test_stage_receipt_payload_rejects_coherent_outer_checksum_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, report = _repository_fixture(tmp_path, monkeypatch)
    import maskimpute_benchmark.publication_freeze as freeze_module

    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    prepare_frozen_method(repository)
    path = repository / "study/frozen_method.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["development_stage_receipt"]["inventory_sha256"] = "0" * 64
    unsigned = {key: value for key, value in changed.items() if key != "payload_sha256"}
    changed["payload_sha256"] = canonical_sha256(unsigned)
    _write_json(path, changed)

    with pytest.raises(PublicationFreezeError, match="stage receipt.*inventory"):
        validate_frozen_method(repository)


def test_raw_stage_drift_blocks_round_freeze_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, report = _repository_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        freeze_module,
        "_recompute_selection_report",
        lambda *_args: deepcopy(report),
    )
    prepare_frozen_method(repository)
    subprocess.run(
        ["git", "add", "-f", "study/frozen_method.json"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "freeze publication method"],
        cwd=repository,
        check=True,
    )
    _materialize_publication_stage_footprint(
        repository,
        "v28",
        only="report",
    )

    with pytest.raises(PublicationFreezeError, match="stage.*differs|incomplete"):
        freeze_publication_round(repository, repository / "artifacts/study/round-001")


def test_stage_race_cannot_publish_stale_v28_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.publication_freeze as freeze_module

    repository, _base_report = _repository_fixture(tmp_path, monkeypatch)
    repository, reports = _schema_four_stage_chain(
        tmp_path,
        "v28",
        repository=repository,
    )
    direct_fixture, _receipt, _projection = _complete_comparator_evidence()
    _write_json(
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json",
        direct_fixture["checkpoint"],
    )
    revision = json.loads(
        (repository / "study/v28_revision.json").read_text(encoding="utf-8")
    )
    _write_json(
        repository
        / "artifacts/study/development/competition-v28-revision/checkpoint.json",
        _direct_selected_stage_checkpoint(
            {
                "configuration_id": revision["configuration_id"],
                "version": "v28",
                "configuration": revision["configuration"],
                "configuration_sha256": revision["configuration_sha256"],
            }
        ),
    )
    _patch_schema_four_stage_replay(monkeypatch, repository, reports)
    original = freeze_module._validate_publication_stage_evidence
    calls = 0

    def introduce_v29(selected_repository: Path, layout: object):
        nonlocal calls
        result = original(selected_repository, layout)
        calls += 1
        if calls == 1:
            _materialize_publication_stage_footprint(
                repository,
                "v29",
                only="report",
            )
        return result

    monkeypatch.setattr(
        freeze_module,
        "_validate_publication_stage_evidence",
        introduce_v29,
    )

    with pytest.raises(PublicationFreezeError, match="stage.*changed|incomplete"):
        prepare_frozen_method(repository)
    assert not (repository / "study/frozen_method.json").exists()
