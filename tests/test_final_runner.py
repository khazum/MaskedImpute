from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from maskimpute_benchmark.methods.registry import MethodRegistry, load_method_registry
from maskimpute_benchmark.protocol import canonical_sha256
from maskimpute_benchmark.runner import DatasetBinding


METHODS = Path("study/methods.json")
MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
VIEWS = ("moderate", "severe")


def _bindings() -> tuple[DatasetBinding, ...]:
    values = []
    for mechanism in MECHANISMS:
        for draw in range(1, 6):
            biological_id = f"draw-{draw:02d}"
            truth = hashlib.sha256(f"{mechanism}:{biological_id}".encode()).hexdigest()
            for view in VIEWS:
                token = f"{mechanism}:{biological_id}:{view}"
                digest = hashlib.sha256(token.encode()).hexdigest()
                values.append(
                    DatasetBinding(
                        mechanism=mechanism,
                        biological_id=biological_id,
                        technical_view=view,
                        dataset_id=f"dataset-{digest[:24]}",
                        dataset_sha256=digest,
                        output_file_sha256=hashlib.sha256(
                            f"file:{token}".encode()
                        ).hexdigest(),
                        truth_sha256=truth,
                        output_path=(
                            f"final/datasets/{mechanism}/{biological_id}/{view}.h5ad"
                        ),
                        independent_unit_id=f"{mechanism}:{biological_id}",
                        cells=2700,
                        genes=1200,
                        manifest_sha256="1" * 64,
                        protocol_sha256="2" * 64,
                        design_sha256="3" * 64,
                        seed_source_sha256="4" * 64,
                    )
                )
    return tuple(values)


def _registry() -> MethodRegistry:
    source = load_method_registry(METHODS)
    return MethodRegistry(
        schema_version=1,
        methods=tuple(
            source.by_id(method_id)
            for method_id in ("observed", "maskimpute", "magic", "scimpute")
        ),
    )


def _receipt(registry: MethodRegistry) -> dict[str, object]:
    selected = {
        "method_version": "v28",
        "decoder": "negative_binomial",
        "encoder_mode": "explicit_mask",
        "output_policy": "selective",
        "score_policy": "retained_development_calibrator",
        "hyperparameters": {"latent_dim": 24},
        "decoder_hyperparameters": {
            "dispersion_prior_strength": 20.0,
            "winsor_quantile": 0.95,
            "min_dispersion": 0.0001,
            "max_dispersion": 100.0,
            "mean_floor": 1e-08,
        },
    }
    denominator = []
    for spec in registry.methods:
        if spec.id == "scimpute":
            applicability = {
                "rule": "never",
                "non_run_reason": "historical_method_not_rerun",
                "required_reference": None,
            }
            status = "historical"
        else:
            applicability = {
                "rule": "all_final_datasets",
                "non_run_reason": None,
                "required_reference": None,
            }
            status = "implemented"
        denominator.append(
            {
                "id": spec.id,
                "method_sha256": canonical_sha256(asdict(spec)),
                "integration_status": status,
                "final_applicability": applicability,
            }
        )
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "preparation_commit": "a" * 40,
        "candidate_method_id": "maskimpute",
        "selected_configuration_id": "v28-candidate",
        "selected_version": "v28",
        "selected_configuration": selected,
        "selected_configuration_sha256": canonical_sha256(selected),
        "method_denominator": denominator,
        "method_registry_sha256": canonical_sha256(
            {
                "schema_version": 1,
                "methods": [asdict(spec) for spec in registry.methods],
            }
        ),
        "runtime_lock_sha256": "5" * 64,
        "selected_calibrator": {
            "artifact": {"payload_sha256": "6" * 64},
            "artifact_payload_sha256": "6" * 64,
        },
        "selected_ablation_control": {
            "capacity_matched_control_id": "capacity-matched-ae",
            "capacity_matched_definition": {"id": "capacity-matched-ae"},
            "capacity_matched_definition_sha256": canonical_sha256(
                {"id": "capacity-matched-ae"}
            ),
        },
    }
    return {**unsigned, "payload_sha256": canonical_sha256(unsigned)}


def _status_payload(bindings: tuple[DatasetBinding, ...]) -> dict[str, object]:
    rows = [
        {
            "status": "completed",
            "mechanism": item.mechanism,
            "biological_id": item.biological_id,
            "technical_view": item.technical_view,
            "dataset_id": item.dataset_id,
            "dataset_sha256": item.dataset_sha256,
            "output_file_sha256": item.output_file_sha256,
            "truth_sha256": item.truth_sha256,
            "output_path": item.output_path,
            "independent_unit_id": item.independent_unit_id,
            "cells": item.cells,
            "genes": item.genes,
        }
        for item in bindings
    ]
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "namespace": "final",
        "status": "completed",
        "protocol_sha256": "2" * 64,
        "design_sha256": "3" * 64,
        "seed_source_sha256": "4" * 64,
        "execution_claim_id": "claim-001",
        "round_id": "round-001",
        "independent_unit_count": 20,
        "completed_count": 40,
        "failed_count": 0,
        "rows": rows,
    }
    return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def test_final_manifest_requires_exact_40_dataset_unseen_panel() -> None:
    from maskimpute_benchmark.final_runner import validate_final_manifest_payload

    bindings = _bindings()
    observed = validate_final_manifest_payload(_status_payload(bindings))

    assert [
        {key: value for key, value in asdict(item).items() if key != "manifest_sha256"}
        for item in observed
    ] == [
        {key: value for key, value in asdict(item).items() if key != "manifest_sha256"}
        for item in bindings
    ]
    assert len({item.independent_unit_id for item in observed}) == 20


def test_final_manifest_rejects_missing_view_even_with_rehashed_manifest() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        validate_final_manifest_payload,
    )

    payload = _status_payload(_bindings())
    payload["rows"].pop()
    payload["completed_count"] = 39
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    payload["manifest_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="40"):
        validate_final_manifest_payload(payload)


def test_final_plan_uses_three_fixed_seeds_and_retains_nonrun_denominator() -> None:
    from maskimpute_benchmark.final_runner import build_final_execution_plan

    registry = _registry()
    plan = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )

    by_method: dict[str, list[object]] = {}
    for item in plan.entries:
        by_method.setdefault(item.run.method_id, []).append(item)
    assert len(by_method["observed"]) == 40
    assert {item.run.model_seed for item in by_method["observed"]} == {None}
    assert len(by_method["maskimpute"]) == 120
    assert {item.run.model_seed for item in by_method["maskimpute"]} == {42, 43, 44}
    assert len(by_method["magic"]) == 120
    assert len(by_method["scimpute"]) == 40
    assert {item.action for item in by_method["scimpute"]} == {"not_applicable"}
    assert {item.reason for item in by_method["scimpute"]} == {
        "historical_method_not_rerun"
    }
    assert len(plan.plan_sha256) == 64


def test_final_plan_rejects_runtime_or_method_receipt_tampering() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    registry = _registry()
    receipt = _receipt(registry)
    receipt["method_denominator"][1]["method_sha256"] = "0" * 64
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="method.*differs"):
        build_final_execution_plan(
            receipt,
            registry,
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def test_final_plan_derives_direct_score_without_calibration_requirement() -> None:
    from maskimpute_benchmark.final_runner import build_final_execution_plan

    registry = _registry()
    receipt = _receipt(registry)
    selected = {
        "method_version": "v27",
        "decoder": "scaled_gaussian",
        "encoder_mode": "explicit_mask",
        "output_policy": "selective",
        "score_policy": "direct_cross_fitted_count_score",
        "hyperparameters": {"latent_dim": 24},
    }
    receipt["selected_version"] = "v27"
    receipt["selected_configuration_id"] = "v27-direct"
    receipt["selected_configuration"] = selected
    receipt["selected_configuration_sha256"] = canonical_sha256(selected)
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = canonical_sha256(unsigned)

    plan = build_final_execution_plan(
        receipt,
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )

    configuration = next(
        value for value in plan.configurations if value.method_id == "maskimpute"
    )
    assert configuration.requires_count_score is True
    assert configuration.requires_calibration is False
    assert {
        entry.run.requires_calibration
        for entry in plan.entries
        if entry.run.method_id == "maskimpute"
    } == {False}


def test_final_plan_rejects_nonexecutable_v29_structure_configuration() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    registry = _registry()
    receipt = _receipt(registry)
    selected = dict(receipt["selected_configuration"])
    selected["method_version"] = "v29"
    selected["structure_hyperparameters"] = {
        "variable_gene_count": 200,
        "neighborhood_k": 15,
        "covariance_penalty_weight": -1.0,
        "neighborhood_penalty_weight": 0.1,
        "variance_floor": 1e-8,
    }
    receipt["selected_version"] = "v29"
    receipt["selected_configuration_id"] = "v29-invalid"
    receipt["selected_configuration"] = selected
    receipt["selected_configuration_sha256"] = canonical_sha256(selected)
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="not executable"):
        build_final_execution_plan(
            receipt,
            registry,
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def test_final_plan_rejects_inconsistent_historical_disposition() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    registry = _registry()
    receipt = _receipt(registry)
    row = next(
        value for value in receipt["method_denominator"] if value["id"] == "scimpute"
    )
    row["integration_status"] = "implemented"
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="historical.*status"):
        build_final_execution_plan(
            receipt,
            registry,
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def test_final_plan_rejects_unbound_matched_bulk_applicability() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    registry = _registry()
    receipt = _receipt(registry)
    row = next(
        value for value in receipt["method_denominator"] if value["id"] == "magic"
    )
    row["final_applicability"] = {
        "rule": "matched_bulk_reference_present",
        "non_run_reason": "matched_bulk_reference_absent",
        "required_reference": {"kind": "truth_matrix"},
    }
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="matched-bulk.*binding"):
        build_final_execution_plan(
            receipt,
            registry,
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def _unavailable_prepared(plan_entry):
    import numpy as np

    from maskimpute_benchmark.methods.base import MethodInput
    from maskimpute_benchmark.runner import (
        DatasetQCAudit,
        PreparedDataset,
    )

    counts = np.array([[1.0, 0.0], [0.0, 2.0]], dtype="<f8")
    method_input = MethodInput(
        source_dataset_sha256=plan_entry.run.source_dataset_sha256,
        obs_ids=("cell-1", "cell-2"),
        var_ids=("gene-1", "gene-2"),
        shape=(2, 2),
        obs_covariates=(),
        var_covariates=(),
        _count_bytes=counts.tobytes(order="C"),
        _normalization_bytes=b"{}",
    )
    audit = DatasetQCAudit(
        excluded_cell_count=0,
        excluded_cell_ids_sha256=hashlib.sha256(
            b"maskimpute-external-cell-ids-v1\0\x00\x00\x00\x00\x00\x00\x00\x00"
        ).hexdigest(),
        retained_cell_count=2,
        retained_cell_ids_sha256="9" * 64,
        excluded_cell_ids=(),
        retained_cell_ids=method_input.obs_ids,
    )
    dataset_id = getattr(plan_entry.run, "dataset_id", None)
    binding = next(
        value
        for value in _bindings()
        if (
            value.dataset_id == dataset_id
            if dataset_id is not None
            else value.dataset_sha256 == plan_entry.run.source_dataset_sha256
        )
    )
    prepared = PreparedDataset(
        binding=binding,
        audit=audit,
        method_input=method_input,
        evaluator_dataset=None,
    )
    return prepared


def _unavailable_attempt(plan_entry):
    from maskimpute_benchmark.runner import AdapterOutcome, evaluate_adapter_outcome

    return evaluate_adapter_outcome(
        plan_entry.run,
        _unavailable_prepared(plan_entry),
        AdapterOutcome.unavailable(plan_entry.reason or "test_unavailable"),
    )


def _completed_attempt(plan_entry):
    from dataclasses import replace

    import anndata as ad
    import numpy as np

    from maskimpute import ImputationResult
    from maskimpute.ablations import AblationRunResult
    from maskimpute_benchmark.methods import run_observed, snapshot_method_output
    from maskimpute_benchmark.methods.maskimpute import MaskImputeAdapterExecution
    from maskimpute_benchmark.runner import AdapterOutcome, evaluate_adapter_outcome

    prepared = _unavailable_prepared(plan_entry)
    evaluator_dataset = ad.AnnData(prepared.method_input.counts)
    evaluator_dataset.obs_names = list(prepared.method_input.obs_ids)
    evaluator_dataset.var_names = list(prepared.method_input.var_ids)
    evaluator_dataset.layers["truth"] = prepared.method_input.counts
    evaluator_dataset.uns["truth_kind"] = "exact_pre_capture"
    evaluator_dataset.uns["primary_truth_layer"] = "truth"
    prepared = replace(prepared, evaluator_dataset=evaluator_dataset)
    observed = load_method_registry(METHODS).by_id("observed")
    execution = run_observed(observed, prepared.method_input)
    selected_spec = load_method_registry(METHODS).by_id(plan_entry.run.method_id)
    snapshot = snapshot_method_output(
        selected_spec,
        prepared.method_input,
        prepared.method_input.counts,
        source_dataset_sha256=prepared.method_input.source_dataset_sha256,
        output_scale=selected_spec.output_scale,
        obs_ids=prepared.method_input.obs_ids,
        var_ids=prepared.method_input.var_ids,
    )
    if plan_entry.run.method_id == "maskimpute":
        probability = np.where(prepared.method_input.counts == 0, 0.5, 0.0)
        result = ImputationResult(
            selective_counts=prepared.method_input.counts,
            denoised_counts=prepared.method_input.counts,
            p_pre_zero=probability,
            latent=np.ones((prepared.method_input.shape[0], 1)),
            diagnostics={
                "score": {
                    "source": "retained_calibrator",
                    "score_artifact_sha256": "a" * 64,
                    "score_input_sha256": "b" * 64,
                    "score_config_sha256": "c" * 64,
                    "calibration_artifact_sha256": "b" * 64,
                    "retained_calibrator": "identity",
                    "calibration_scope": "retained_all_development_for_final_inference",
                    "equivalence_reason": (
                        "retained_identity_calibrator_equals_direct_score"
                    ),
                }
            },
        )
        selected_execution = MaskImputeAdapterExecution(
            snapshot=snapshot,
            compatibility_log=(),
            environment_receipt=(),
            stdout=b"",
            stderr=b"",
            command=None,
            ablation_result=AblationRunResult(
                output_policy="selective", _result=result
            ),
        )
    else:
        selected_execution = replace(execution, snapshot=snapshot)
    outcome = AdapterOutcome.completed(
        selected_execution,
        runtime_seconds=1,
        peak_rss_bytes=1,
        peak_gpu_bytes=1,
    )
    return evaluate_adapter_outcome(plan_entry.run, prepared, outcome)


def _authority():
    from maskimpute_benchmark.runner import ExecutionAuthorityContext

    base = {"latent_dim": 2}
    count = {"n_folds": 2}
    return ExecutionAuthorityContext(
        authority_sha256="9" * 64,
        base_configuration_json=json.dumps(base, separators=(",", ":"), sort_keys=True),
        base_configuration_sha256=canonical_sha256(base),
        count_model_config_json=json.dumps(
            count, separators=(",", ":"), sort_keys=True
        ),
        count_model_config_sha256=canonical_sha256(count),
        count_score_manifest_path="artifacts/study/round-001/results/score.json",
        count_score_manifest_sha256="a" * 64,
        retained_calibration_path=(
            "artifacts/study/round-001/results/calibration.json"
        ),
        retained_calibration_sha256="b" * 64,
    )


def test_final_result_store_is_immutable_resumable_and_manifest_complete(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import FinalResultStore

    registry = _registry()
    plan = build_plan = __import__(
        "maskimpute_benchmark.final_runner", fromlist=["build_final_execution_plan"]
    ).build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    del build_plan
    store = FinalResultStore(tmp_path / "execution", plan)

    first = store.append(plan.entries[0], _unavailable_attempt(plan.entries[0]))
    resumed = FinalResultStore(tmp_path / "execution", plan)

    assert resumed.load_records() == (first,)
    with pytest.raises(Exception, match="complete"):
        resumed.finalize()

    # Exercise completion with a deliberately one-entry view of the same authority.
    tiny_plan = plan.__class__(
        schema_version=plan.schema_version,
        input_hashes=plan.input_hashes,
        entries=(plan.entries[0],),
        configurations=plan.configurations,
        plan_sha256=canonical_sha256(
            {
                "parent_plan_sha256": plan.plan_sha256,
                "entries": [plan.entries[0].to_dict()],
            }
        ),
    )
    tiny_root = tmp_path / "tiny"
    tiny = FinalResultStore(tiny_root, tiny_plan)
    tiny.append(tiny_plan.entries[0], _unavailable_attempt(tiny_plan.entries[0]))
    manifest = tiny.finalize()

    assert manifest["status"] == "completed"
    assert manifest["planned_run_count"] == 1
    assert tiny.load_manifest() == manifest


def test_final_result_store_rejects_tampered_record_or_log(tmp_path: Path) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="a" * 64,
    )
    store = FinalResultStore(tmp_path / "execution", plan)
    record = store.append(plan.entries[0], _unavailable_attempt(plan.entries[0]))
    stdout = tmp_path / "execution" / record["run"]["stdout_path"]
    stdout.write_bytes(b"tampered")

    with pytest.raises(FinalRunnerContractError, match="record|artifact"):
        FinalResultStore(tmp_path / "execution", plan).load_records()


def test_final_unique_file_reader_enforces_a_pre_read_size_bound(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _read_unique_file,
    )

    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"four")

    with pytest.raises(FinalRunnerContractError, match="size bound"):
        _read_unique_file(artifact, "bounded artifact", max_bytes=3)


def test_final_result_store_rejects_a_broken_record_directory_symlink(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    registry = _registry()
    plan = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    output = tmp_path / "execution"
    output.mkdir()
    (output / "records").symlink_to(tmp_path / "missing")

    with pytest.raises(FinalRunnerContractError, match="record directory"):
        FinalResultStore(output, plan).load_records()


def test_final_result_store_binds_successful_final_calibration_request(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        build_final_execution_plan,
    )
    from maskimpute_benchmark.runner import ExecutionRequest

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    entry = next(value for value in full.entries if value.run.method_id == "maskimpute")
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0], entry),
        configurations=full.configurations,
        plan_sha256="c" * 64,
    )
    configuration = next(
        value for value in plan.configurations if value.method_id == "maskimpute"
    )
    prepared = _unavailable_prepared(entry)
    request = ExecutionRequest.create(
        registry.by_id("maskimpute"),
        prepared.method_input,
        model_seed=entry.run.model_seed,
        configuration=configuration,
        authority=_authority(),
        mechanism=entry.run.mechanism,
        biological_id=entry.run.biological_id,
        technical_view=entry.run.technical_view,
        dataset_id=entry.run.dataset_id,
        timeout_seconds=5,
        calibration_usage="retained_all_development",
    )
    store = FinalResultStore(tmp_path / "execution", plan)
    store.append(full.entries[0], _unavailable_attempt(full.entries[0]))

    attempt = _completed_attempt(entry)
    record = store.append(
        entry,
        attempt,
        execution_request=request,
    )

    assert record["execution_request"] == {
        "calibration_usage": "retained_all_development",
        "configuration_sha256": entry.run.configuration_sha256,
        "count_score_manifest_sha256": "a" * 64,
        "dataset_id": entry.run.dataset_id,
        "execution_authority_sha256": "9" * 64,
        "method_input_sha256": request.method_input_sha256,
        "model_seed": entry.run.model_seed,
        "request_sha256": request.request_sha256,
        "retained_calibration_sha256": "b" * 64,
    }
    run = record["run"]
    assert run["native_output_path"] is None
    assert run["native_output_retention"] == "omitted_redundant_final_output"
    assert run["evaluator_output_encoding"] == "zlib_raw_f64_v1"
    import zlib

    compressed = (
        tmp_path / "execution" / str(run["evaluator_output_path"])
    ).read_bytes()
    expected = attempt.evaluator_output.astype("<f8").tobytes(order="C")
    assert zlib.decompress(compressed) == expected
    assert (
        run["evaluator_output_uncompressed_sha256"]
        == hashlib.sha256(expected).hexdigest()
    )
    score = record["p_pre_zero_evidence"]
    score_storage = score["storage"]
    score_path = tmp_path / "execution" / str(score_storage["path"])
    expected_score = attempt.p_pre_zero_evidence.matrix.astype("<f8").tobytes(order="C")
    assert score_storage["encoding"] == "zlib_raw_f64_v1"
    assert zlib.decompress(score_path.read_bytes()) == expected_score
    assert (
        score_storage["uncompressed_sha256"]
        == hashlib.sha256(expected_score).hexdigest()
    )
    assert FinalResultStore(tmp_path / "execution", plan).load_records()[-1] == record

    oversized = dict(run)
    oversized["evaluator_output_shape"] = [2701, 1200]
    oversized["evaluator_output_uncompressed_nbytes"] = 2701 * 1200 * 8
    with pytest.raises(Exception, match="matrix bound"):
        store._validate_final_output_storage(oversized)

    score_compressed = score_path.read_bytes()
    zip_bomb = zlib.compress(b"x" * (score_storage["uncompressed_nbytes"] + 1))
    score_path.write_bytes(zip_bomb)
    forged_record = json.loads(json.dumps(record))
    forged_score_storage = forged_record["p_pre_zero_evidence"]["storage"]
    forged_score_storage["compressed_sha256"] = hashlib.sha256(zip_bomb).hexdigest()
    forged_score_storage["compressed_nbytes"] = len(zip_bomb)
    record_path = tmp_path / "execution/records/00000002.json"
    record_path.write_text(
        json.dumps(
            forged_record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="record|p_pre_zero|compressed"):
        FinalResultStore(tmp_path / "execution", plan).load_records()
    score_path.write_bytes(score_compressed)
    record_path.write_text(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    compressed_path = tmp_path / "execution" / str(run["evaluator_output_path"])
    compressed_path.write_bytes(compressed + b"tamper")
    with pytest.raises(Exception, match="record|artifact|compressed"):
        FinalResultStore(tmp_path / "execution", plan).load_records()


def test_final_result_store_append_does_not_rehash_the_whole_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        build_final_execution_plan,
    )

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=full.entries[:3],
        configurations=full.configurations,
        plan_sha256="d" * 64,
    )
    store = FinalResultStore(tmp_path / "execution", plan)
    original = store._read_record
    reads: list[str] = []

    def counted(path, entry):
        reads.append(path.name)
        return original(path, entry)

    monkeypatch.setattr(store, "_read_record", counted)
    for entry in plan.entries:
        store.append(entry, _unavailable_attempt(entry))

    assert reads == ["00000001.json", "00000002.json", "00000003.json"]


def test_execute_final_plan_uses_final_calibration_request_and_resumes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        build_final_execution_plan,
        execute_final_plan,
    )
    from maskimpute_benchmark.runner import AdapterOutcome

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="a" * 64,
    )
    authority = _authority()
    prepared = {plan.entries[0].run.dataset_id: _unavailable_prepared(plan.entries[0])}
    requests = []
    publications = []

    def executor(request):
        requests.append(request)
        return AdapterOutcome.unavailable("adapter_unavailable")

    store = FinalResultStore(tmp_path / "execution", plan)
    manifest = execute_final_plan(
        plan,
        registry,
        prepared,
        authority,
        executor,
        store,
        on_record_published=lambda: publications.append("published"),
    )

    assert manifest["status"] == "completed"
    assert len(requests) == 1
    assert requests[0].calibration_usage == "retained_all_development"
    assert requests[0].calibration_context is None
    assert publications == ["published"]

    resumed = execute_final_plan(
        plan,
        registry,
        prepared,
        authority,
        executor,
        FinalResultStore(tmp_path / "execution", plan),
        on_record_published=lambda: publications.append("unexpected"),
    )
    assert resumed == manifest
    assert len(requests) == 1


def test_execute_final_plan_leaves_infrastructure_failure_retryable(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        FinalRunnerContractError,
        build_final_execution_plan,
        execute_final_plan,
    )
    from maskimpute_benchmark.runner import AdapterOutcome

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="4" * 64,
    )
    prepared = {plan.entries[0].run.dataset_id: _unavailable_prepared(plan.entries[0])}
    root = tmp_path / "execution"

    with pytest.raises(FinalRunnerContractError, match="retryable infrastructure"):
        execute_final_plan(
            plan,
            registry,
            prepared,
            _authority(),
            lambda _request: AdapterOutcome.infrastructure_error("worker_spawn_failed"),
            FinalResultStore(root, plan),
            on_record_published=lambda: None,
        )
    assert FinalResultStore(root, plan).load_records() == ()

    manifest = execute_final_plan(
        plan,
        registry,
        prepared,
        _authority(),
        lambda _request: AdapterOutcome.unavailable("algorithm_unavailable"),
        FinalResultStore(root, plan),
        on_record_published=lambda: None,
    )
    assert manifest["recorded_run_count"] == 1


def test_execute_final_plan_journals_once_after_the_complete_manifest(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        build_final_execution_plan,
        execute_final_plan,
    )
    from maskimpute_benchmark.runner import AdapterOutcome

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=full.entries[:33],
        configurations=full.configurations,
        plan_sha256="e" * 64,
    )
    prepared = {
        entry.run.dataset_id: _unavailable_prepared(entry) for entry in plan.entries
    }
    publications: list[str] = []

    manifest = execute_final_plan(
        plan,
        registry,
        prepared,
        _authority(),
        lambda _request: AdapterOutcome.unavailable("adapter_unavailable"),
        FinalResultStore(tmp_path / "execution", plan),
        on_record_published=lambda: publications.append("published"),
    )

    assert manifest["status"] == "completed"
    assert publications == ["published"]


def test_final_evaluation_retains_reason_coded_algorithmic_unavailability(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        build_final_execution_plan,
        validate_final_execution_for_evaluation,
    )

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="f" * 64,
    )
    store = FinalResultStore(tmp_path / "execution", plan)
    store.append(plan.entries[0], _unavailable_attempt(plan.entries[0]))

    validation = validate_final_execution_for_evaluation(plan, store.load_records())

    assert validation["executed_completed_count"] == 0
    assert validation["executed_algorithmic_failure_count"] == 1
    assert validation["executed_status_counts"] == {"unavailable": 1}


@pytest.mark.parametrize("status", ["failed", "timeout", "resource_exceeded"])
def test_final_evaluation_retains_each_unfavorable_algorithmic_status(
    tmp_path: Path, status: str
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        build_final_execution_plan,
        validate_final_execution_for_evaluation,
    )
    from maskimpute_benchmark.runner import AdapterOutcome, evaluate_adapter_outcome

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="2" * 64,
    )
    entry = plan.entries[0]
    outcome = getattr(AdapterOutcome, status)(f"algorithmic_{status}")
    attempt = evaluate_adapter_outcome(entry.run, _unavailable_prepared(entry), outcome)
    store = FinalResultStore(tmp_path / status / "execution", plan)
    store.append(entry, attempt)

    validation = validate_final_execution_for_evaluation(plan, store.load_records())

    assert validation["planned_run_count"] == 1
    assert validation["executed_algorithmic_failure_count"] == 1
    assert validation["executed_status_counts"] == {status: 1}


def test_final_evaluation_blocks_infrastructure_incompleteness(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        FinalRunnerContractError,
        build_final_execution_plan,
        validate_final_execution_for_evaluation,
    )
    from maskimpute_benchmark.runner import (
        AdapterOutcome,
        evaluate_adapter_outcome,
    )

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="1" * 64,
    )
    entry = plan.entries[0]
    attempt = evaluate_adapter_outcome(
        entry.run,
        _unavailable_prepared(entry),
        AdapterOutcome.infrastructure_error("worker_spawn_failed"),
    )
    store = FinalResultStore(tmp_path / "execution", plan)
    store.append(entry, attempt)

    with pytest.raises(FinalRunnerContractError, match="infrastructure|authority"):
        validate_final_execution_for_evaluation(plan, store.load_records())


def test_final_evaluation_accepts_completed_execution_and_exact_nonrun() -> None:
    from maskimpute_benchmark.final_runner import (
        build_final_execution_plan,
        validate_final_execution_for_evaluation,
    )

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    execute = full.entries[0]
    nonrun = next(value for value in full.entries if value.action == "not_applicable")
    completed = json.loads(
        json.dumps(
            {
                "run": asdict(_completed_attempt(execute).run),
                "metrics": [
                    metric.to_dict() for metric in _completed_attempt(execute).metrics
                ],
                "execution_request": None,
            }
        )
    )
    unavailable = json.loads(
        json.dumps(
            {
                "run": asdict(_unavailable_attempt(nonrun).run),
                "metrics": [
                    metric.to_dict() for metric in _unavailable_attempt(nonrun).metrics
                ],
                "execution_request": None,
            }
        )
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(execute, nonrun),
        configurations=full.configurations,
        plan_sha256="0" * 64,
    )

    validation = validate_final_execution_for_evaluation(plan, (completed, unavailable))

    assert validation["executed_completed_count"] == 1
    assert validation["not_applicable_count"] == 1


def test_load_prepared_final_panel_revalidates_and_pairs_in_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    (round_dir / "results").mkdir(parents=True)
    status_path = round_dir / "results/dataset_status.json"
    status_path.write_text("{}\n", encoding="utf-8")
    bindings = _bindings()[:2]
    status = {"validated": True}
    validations = []
    reads = []
    pair_calls = []

    def validate_status(path, *, repo, round_dir):
        validations.append((path, repo, round_dir))
        return status

    monkeypatch.setattr(
        "maskimpute_benchmark.datasets.validate_dataset_status", validate_status
    )
    monkeypatch.setattr(
        final_runner, "validate_final_manifest_payload", lambda value: bindings
    )

    datasets = {binding.dataset_id: object() for binding in bindings}

    def read_bound(path, binding):
        reads.append((path, binding.dataset_id))
        return datasets[binding.dataset_id]

    monkeypatch.setattr(final_runner, "_read_bound_h5ad", read_bound)

    prepared = {
        binding.dataset_id: _unavailable_prepared(
            type(
                "Entry",
                (),
                {
                    "run": type(
                        "Run",
                        (),
                        {
                            "source_dataset_sha256": binding.dataset_sha256,
                        },
                    )()
                },
            )()
        )
        for binding in bindings
    }
    prepared = {
        binding.dataset_id: value.__class__(
            binding=binding,
            audit=value.audit,
            method_input=value.method_input,
            evaluator_dataset=value.evaluator_dataset,
        )
        for binding, value in zip(bindings, prepared.values(), strict=True)
    }

    def prepare_pair(first, second, first_binding, second_binding, policy):
        pair_calls.append(
            (first, second, first_binding.dataset_id, second_binding.dataset_id, policy)
        )
        return prepared[first_binding.dataset_id], prepared[second_binding.dataset_id]

    monkeypatch.setattr(
        final_runner, "prepare_dataset_pair_for_execution", prepare_pair
    )

    observed_bindings, observed_prepared = final_runner.load_prepared_final_panel(
        repository, round_dir
    )

    assert observed_bindings == bindings
    assert tuple(observed_prepared) == tuple(binding.dataset_id for binding in bindings)
    assert [item[1] for item in reads] == [binding.dataset_id for binding in bindings]
    assert len(pair_calls) == 1
    assert pair_calls[0][4] == final_runner.DatasetQCPolicy.fixed()
    assert validations == [
        (status_path, repository.resolve(), round_dir.resolve()),
        (status_path, repository.resolve(), round_dir.resolve()),
    ]


def test_load_prepared_final_panel_rejects_round_outside_repository(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        load_prepared_final_panel,
    )

    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()

    with pytest.raises(FinalRunnerContractError, match="inside repository"):
        load_prepared_final_panel(repository, outside)


def test_materialize_final_execution_authority_is_frozen_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute import calibration as calibration_module
    from maskimpute_benchmark.final_runner import (
        materialize_final_execution_authority,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    (round_dir / "results").mkdir(parents=True)
    (repository / "study").mkdir()
    count_config = {"n_folds": 5, "mean_floor": 1e-8}
    selection = {
        "schema_version": 1,
        "count_model_config": count_config,
        "count_model_config_sha256": canonical_sha256(count_config),
    }
    selection_raw = (
        json.dumps(selection, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    selection_path = repository / "study/selection_contract.json"
    selection_path.write_bytes(selection_raw)
    calibration = {"schema_version": 3, "selected_algorithm": "identity"}
    calibration_raw = (
        json.dumps(calibration, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()

    class FakeCalibrationArtifact:
        def __init__(self, payload):
            assert payload == calibration

        def to_dict(self):
            return dict(calibration)

    monkeypatch.setattr(
        calibration_module, "CalibrationArtifact", FakeCalibrationArtifact
    )

    registry = _registry()
    frozen = _receipt(registry)
    frozen["artifact_bindings"] = {
        "selection_contract": {
            "path": "study/selection_contract.json",
            "sha256": hashlib.sha256(selection_raw).hexdigest(),
        }
    }
    frozen["selected_calibrator"] = {
        "score_policy": "retained_development_calibrator",
        "final_usage": "retained_all_development_calibrator",
        "artifact_file_sha256": hashlib.sha256(calibration_raw).hexdigest(),
        "artifact": calibration,
    }
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)

    correct_calibration_sha256 = frozen["selected_calibrator"]["artifact_file_sha256"]
    frozen["selected_calibrator"]["artifact_file_sha256"] = "0" * 64
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)
    calibration_path = (
        round_dir / "results/final/execution_authority/retained_calibration.json"
    )

    with pytest.raises(Exception, match="calibration.*differ"):
        materialize_final_execution_authority(
            repository,
            round_dir,
            frozen,
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            dataset_manifest_sha256="1" * 64,
        )
    assert not calibration_path.exists()

    frozen["selected_calibrator"]["artifact_file_sha256"] = correct_calibration_sha256
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)

    context = materialize_final_execution_authority(
        repository,
        round_dir,
        frozen,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        dataset_manifest_sha256="1" * 64,
    )
    resumed = materialize_final_execution_authority(
        repository,
        round_dir,
        frozen,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        dataset_manifest_sha256="1" * 64,
    )

    assert resumed == context
    assert context.base_configuration_sha256 == canonical_sha256(
        frozen["selected_configuration"]["hyperparameters"]
    )
    assert context.count_model_config_sha256 == canonical_sha256(count_config)
    assert (
        context.retained_calibration_sha256
        == hashlib.sha256(calibration_raw).hexdigest()
    )
    assert (
        repository / context.retained_calibration_path
    ).read_bytes() == calibration_raw
    authority_path = round_dir / "results/final/execution_authority/authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    assert authority["authority_sha256"] == context.authority_sha256


def test_final_result_file_manifest_rejects_symlinks(tmp_path: Path) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        final_result_file_manifest,
    )

    round_dir = tmp_path / "round-001"
    results = round_dir / "results"
    results.mkdir(parents=True)
    first = results / "first.txt"
    first.write_text("first\n", encoding="utf-8")

    manifest = final_result_file_manifest(round_dir)
    assert manifest == {
        "result_files": [
            {
                "path": "results/first.txt",
                "sha256": hashlib.sha256(b"first\n").hexdigest(),
            }
        ]
    }

    (results / "alias.txt").symlink_to(first)
    with pytest.raises(FinalRunnerContractError, match="symlink"):
        final_result_file_manifest(round_dir)


def test_incremental_result_publication_treats_only_exact_no_change_as_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.study import StudyStateError

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    (round_dir / "results").mkdir(parents=True)
    monkeypatch.setattr(
        final_runner,
        "final_result_file_manifest",
        lambda _round: {"result_files": []},
    )

    def unchanged(*_args, **_kwargs):
        raise StudyStateError("incremental result manifest adds no files")

    assert (
        final_runner._record_incremental_results_if_changed(
            repository,
            round_dir,
            unchanged,
        )
        is None
    )

    def corrupt(*_args, **_kwargs):
        raise StudyStateError("incremental result manifest hash changed")

    with pytest.raises(StudyStateError, match="hash changed"):
        final_runner._record_incremental_results_if_changed(
            repository,
            round_dir,
            corrupt,
        )


def test_incremental_reconciliation_rejects_unowned_result_files(
    tmp_path: Path,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    results = round_dir / "results/final/execution"
    results.mkdir(parents=True)
    (results / "unexpected.txt").write_text("not runner-owned\n", encoding="utf-8")
    called = False

    def recorder(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    with pytest.raises(final_runner.FinalRunnerContractError, match="unowned"):
        final_runner._record_incremental_results_if_changed(
            repository, round_dir, recorder
        )
    assert called is False


def test_owned_result_manifest_is_derived_from_final_record_references(
    tmp_path: Path,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    registry = _registry()
    full = final_runner.build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="3" * 64,
    )
    round_dir = tmp_path / "round-001"
    output = round_dir / "results/final/execution"
    store = final_runner.FinalResultStore(output, plan)
    store.append(plan.entries[0], _unavailable_attempt(plan.entries[0]))
    store.finalize()

    manifest = final_runner._owned_final_result_file_manifest(round_dir)

    assert {item["path"] for item in manifest["result_files"]} == {
        "results/final/execution/execution_manifest.json",
        "results/final/execution/records/00000001.json",
        f"results/final/execution/runs/{plan.entries[0].run.run_id}.stderr",
        f"results/final/execution/runs/{plan.entries[0].run.run_id}.stdout",
    }


def test_stale_result_temporary_recovery_repairs_interrupted_hardlink(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import _remove_stale_result_temporaries

    round_dir = tmp_path / "round-001"
    runs = round_dir / "results/final/execution/runs"
    runs.mkdir(parents=True)
    temporary = runs / ".run.stdout.0123456789.tmp"
    published = runs / "run.stdout"
    temporary.write_bytes(b"stable artifact")
    published.hardlink_to(temporary)
    untouched = runs / ".not-a-staging-file"
    untouched.write_bytes(b"keep")

    removed = _remove_stale_result_temporaries(round_dir)

    assert removed == ("results/final/execution/runs/.run.stdout.0123456789.tmp",)
    assert not temporary.exists()
    assert published.read_bytes() == b"stable artifact"
    assert published.stat().st_nlink == 1
    assert untouched.read_bytes() == b"keep"


def test_interrupted_final_attempt_transaction_removes_orphan_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        _recover_interrupted_final_transactions,
        build_final_execution_plan,
    )

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="5" * 64,
    )
    round_dir = tmp_path / "round-001"
    output = round_dir / "results/final/execution"
    store = FinalResultStore(output, plan)
    original = store._stored_final_attempt

    def interrupt_after_artifacts(attempt):
        original(attempt)
        raise RuntimeError("interrupted after artifacts")

    monkeypatch.setattr(store, "_stored_final_attempt", interrupt_after_artifacts)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.append(plan.entries[0], _completed_attempt(plan.entries[0]))

    assert any((output / "runs").iterdir())
    assert any((output / "transactions").iterdir())

    recovered = _recover_interrupted_final_transactions(round_dir)

    assert recovered == (1,)
    assert not (output / "transactions").exists()
    assert not (output / "runs").exists()
    assert FinalResultStore(output, plan).load_records() == ()


def test_interrupted_maskimpute_transaction_removes_realized_score_artifact(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        _recover_interrupted_final_transactions,
        build_final_execution_plan,
    )

    registry = _registry()
    plan = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    entry = next(item for item in plan.entries if item.run.method_id == "maskimpute")
    round_dir = tmp_path / "round-001"
    output = round_dir / "results/final/execution"
    store = FinalResultStore(output, plan)
    attempt = _completed_attempt(entry)

    intent_path = store._publish_transaction_intent(entry, attempt)
    store._stored_final_attempt(attempt)

    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    score_relative = f"runs/{entry.run.run_id}.p-pre-zero-f64.zlib"
    assert score_relative in intent["artifact_paths"]
    assert (output / score_relative).is_file()

    recovered = _recover_interrupted_final_transactions(round_dir)

    assert recovered == (entry.run.ordinal,)
    assert not (output / "transactions").exists()
    assert not (output / "runs").exists()


def test_stale_result_temporary_recovery_rejects_symlink(tmp_path: Path) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _remove_stale_result_temporaries,
    )

    round_dir = tmp_path / "round-001"
    results = round_dir / "results/final/execution"
    results.mkdir(parents=True)
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    (results / ".artifact.0123456789.tmp").symlink_to(target)

    with pytest.raises(FinalRunnerContractError, match="temporary.*regular"):
        _remove_stale_result_temporaries(round_dir)


def test_frozen_final_round_public_api_accepts_no_scientific_overrides() -> None:
    from maskimpute_benchmark.final_runner import run_frozen_final_round

    assert tuple(inspect.signature(run_frozen_final_round).parameters) == (
        "repository",
        "round_dir",
    )


def test_final_runtime_registry_must_match_the_frozen_lock(tmp_path: Path) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _validate_final_runtime_lock,
    )
    from maskimpute_benchmark.runner import ExecutionEnvironmentRegistry

    repository = tmp_path / "repository"
    repository.mkdir()
    environments = ExecutionEnvironmentRegistry.fixed(repository)

    with pytest.raises(FinalRunnerContractError, match="runtime lock differs"):
        _validate_final_runtime_lock({"runtime_lock_sha256": "5" * 64}, environments)


def test_final_storage_preflight_reserves_one_compressed_common_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    registry = _registry()
    full = final_runner.build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256="a" * 64,
    )
    monkeypatch.setattr(
        final_runner.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 10**15})(),
    )
    receipt = final_runner._validate_final_storage_capacity(plan, tmp_path)
    assert receipt["per_execution_compressed_bound_bytes"] == 25_927_923
    assert receipt["remaining_p_pre_zero_execution_count"] == 0

    monkeypatch.setattr(
        final_runner.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 1})(),
    )

    with pytest.raises(final_runner.FinalRunnerContractError, match="free storage"):
        final_runner._validate_final_storage_capacity(plan, tmp_path)


def test_final_storage_preflight_adds_realized_score_matrix_for_maskimpute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    registry = _registry()
    full = final_runner.build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    maskimpute = next(
        entry for entry in full.entries if entry.run.method_id == "maskimpute"
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0], maskimpute),
        configurations=full.configurations,
        plan_sha256="d" * 64,
    )
    monkeypatch.setattr(
        final_runner.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"free": 10**15})(),
    )

    receipt = final_runner._validate_final_storage_capacity(plan, tmp_path)

    assert receipt["remaining_execution_count"] == 2
    assert receipt["remaining_p_pre_zero_execution_count"] == 1
    assert receipt["per_p_pre_zero_compressed_bound_bytes"] == 25_927_923
    assert receipt["required_free_bytes"] == (
        3 * 25_927_923 + 2 * 1024 * 1024 + 1024**3
    )


def test_frozen_final_cli_exposes_only_round_locator() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_frozen_final.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "round_dir" in completed.stdout
    for forbidden in (
        "--repository",
        "--environment",
        "--seed",
        "--mechanism",
        "--configuration",
        "--method",
    ):
        assert forbidden not in completed.stdout
