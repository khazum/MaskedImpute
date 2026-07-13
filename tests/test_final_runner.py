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
            "dispersion_floor": 0.0001,
            "dispersion_ceiling": 10000.0,
            "dispersion_prior_strength": 20.0,
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
    prepared = PreparedDataset(
        binding=_bindings()[0],
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


def test_execute_final_plan_uses_final_calibration_request_and_resumes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        build_final_execution_plan,
        execute_final_plan,
    )
    from maskimpute_benchmark.runner import (
        AdapterOutcome,
        ExecutionAuthorityContext,
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
    base = {"latent_dim": 2}
    count = {"n_folds": 2}
    authority = ExecutionAuthorityContext(
        authority_sha256="9" * 64,
        base_configuration_json=(
            __import__("json").dumps(base, separators=(",", ":"), sort_keys=True)
        ),
        base_configuration_sha256=canonical_sha256(base),
        count_model_config_json=(
            __import__("json").dumps(count, separators=(",", ":"), sort_keys=True)
        ),
        count_model_config_sha256=canonical_sha256(count),
        count_score_manifest_path="artifacts/study/round-001/results/score.json",
        count_score_manifest_sha256="a" * 64,
        retained_calibration_path=(
            "artifacts/study/round-001/results/calibration.json"
        ),
        retained_calibration_sha256="b" * 64,
    )
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
    assert publications == ["published", "published"]

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
    monkeypatch.setattr(final_runner, "validate_final_manifest_payload", lambda value: bindings)

    datasets = {binding.dataset_id: object() for binding in bindings}

    def read_bound(path, binding):
        reads.append((path, binding.dataset_id))
        return datasets[binding.dataset_id]

    monkeypatch.setattr(final_runner, "_read_bound_h5ad", read_bound)

    prepared = {
        binding.dataset_id: _unavailable_prepared(
            type("Entry", (), {"run": type("Run", (), {
                "source_dataset_sha256": binding.dataset_sha256,
            })()})()
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

    monkeypatch.setattr(final_runner, "prepare_dataset_pair_for_execution", prepare_pair)

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
    assert context.retained_calibration_sha256 == hashlib.sha256(
        calibration_raw
    ).hexdigest()
    assert (repository / context.retained_calibration_path).read_bytes() == calibration_raw
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
            {"path": "results/first.txt", "sha256": hashlib.sha256(b"first\n").hexdigest()}
        ]
    }

    (results / "alias.txt").symlink_to(first)
    with pytest.raises(FinalRunnerContractError, match="symlink"):
        final_result_file_manifest(round_dir)


def test_frozen_final_round_public_api_accepts_no_scientific_overrides() -> None:
    from maskimpute_benchmark.final_runner import run_frozen_final_round

    assert tuple(inspect.signature(run_frozen_final_round).parameters) == (
        "repository",
        "round_dir",
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
