from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import runpy
import stat
import subprocess
import sys

import numpy as np
import pytest

from maskimpute_benchmark.methods.registry import MethodRegistry, load_method_registry
from maskimpute_benchmark.protocol import canonical_sha256
from maskimpute_benchmark.runner import DatasetBinding


METHODS = Path("study/methods.json")
MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
VIEWS = ("moderate", "severe")


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _read_only_tree_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        payload_sha256 = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if stat.S_ISREG(metadata.st_mode)
            else None
        )
        symlink_target = os.readlink(path) if stat.S_ISLNK(metadata.st_mode) else None
        rows.append(
            (
                relative,
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_nlink,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                payload_sha256,
                symlink_target,
            )
        )
    return tuple(rows)


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _claimed_lifecycle_round(
    tmp_path: Path,
    *,
    seed_count: int = 4,
    claimed: bool = True,
) -> tuple[Path, Path]:
    from maskimpute_benchmark.study import (
        assert_final_runnable,
        freeze_round,
        materialize_final,
    )

    repository = tmp_path / "lifecycle-repository"
    repository.mkdir(parents=True)
    _git(repository, "init")
    _git(repository, "config", "user.name", "Final Runner Test")
    _git(repository, "config", "user.email", "runner@example.invalid")
    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repository / "config.json").write_text('{"method":"fixture"}\n', encoding="utf-8")
    (repository / "environment.lock").write_text("python=3.11\n", encoding="utf-8")
    (repository / "protocol.json").write_bytes(Path("study/protocol.json").read_bytes())
    study = repository / "study"
    study.mkdir()
    count_config = {"n_folds": 5, "mean_floor": 1e-8}
    selection = {
        "schema_version": 1,
        "count_model_config": count_config,
        "count_model_config_sha256": canonical_sha256(count_config),
    }
    (study / "selection_contract.json").write_text(
        json.dumps(selection, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    (study / "trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    (study / "development_panel.json").write_bytes(
        Path("study/development_panel.json").read_bytes()
    )
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "freeze final runner lifecycle fixture")
    round_dir = repository / "artifacts/study/round-001"
    freeze_round(
        repository,
        round_dir,
        repository / "config.json",
        repository / "protocol.json",
        environment_path=repository / "environment.lock",
    )
    materialize_final(round_dir, seed_count=seed_count, repo=repository)
    if claimed:
        assert_final_runnable(repository, round_dir)
        (round_dir / "results").mkdir()
    return repository, round_dir


class _FakeRuntimeSnapshot:
    def __init__(
        self,
        semantic_sha256: str,
        semantic_receipt: dict[str, object],
        *,
        fail_receipt: bool = False,
    ) -> None:
        self.semantic_sha256 = semantic_sha256
        self._semantic_receipt = json.loads(json.dumps(semantic_receipt))
        self._fail_receipt = fail_receipt
        self.receipt_reads = 0
        self.enter_count = 0
        self.close_count = 0

    @property
    def semantic_receipt(self) -> dict[str, object]:
        self.receipt_reads += 1
        if self._fail_receipt:
            raise RuntimeError("semantic receipt capture failed")
        value = json.loads(json.dumps(self._semantic_receipt))
        assert isinstance(value, dict)
        return value

    def close(self) -> None:
        self.close_count += 1

    def __enter__(self) -> _FakeRuntimeSnapshot:
        self.enter_count += 1
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


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


@lru_cache(maxsize=1)
def _task14_test_module():
    path = Path(__file__).with_name("test_freeze_publication_round.py")
    spec = importlib.util.spec_from_file_location("_task15_freeze_factory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_registry() -> MethodRegistry:
    from maskimpute_benchmark.methods.registry import parse_method_registry

    return parse_method_registry(_task14_test_module()._direct_method_registry())


def _direct_payload_template() -> dict[str, object]:
    return _task14_test_module()._direct_payload()


_DIRECT_PAYLOAD_TEMPLATE = _direct_payload_template()


def _direct_frozen_method(
    *, unavailable_method: str | None = None
) -> dict[str, object]:
    module = _task14_test_module()
    if unavailable_method is None:
        frozen = deepcopy(_DIRECT_PAYLOAD_TEMPLATE)
    else:
        fixture, receipt, _projection = (
            module._intrinsic_unavailable_comparator_evidence(unavailable_method)
        )
        frozen = deepcopy(
            module._direct_build(
                comparator_fixture=fixture,
                comparator_selection_receipt=receipt,
            )
        )
    candidate = _receipt(_full_registry())
    for key in (
        "selected_configuration_id",
        "selected_version",
        "selected_configuration",
        "selected_configuration_sha256",
        "selected_calibrator",
        "selected_ablation_control",
    ):
        frozen[key] = deepcopy(candidate[key])
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)
    return frozen


def _observed_only_registry() -> MethodRegistry:
    source = load_method_registry(METHODS)
    return MethodRegistry(schema_version=1, methods=(source.by_id("observed"),))


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
    direct = deepcopy(_DIRECT_PAYLOAD_TEMPLATE)
    direct_rows = {str(row["id"]): row for row in direct["method_denominator"]}
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
        row = {
            "id": spec.id,
            "method_sha256": canonical_sha256(asdict(spec)),
            "integration_status": status,
            "final_applicability": applicability,
        }
        if spec.id in direct_rows:
            direct_row = direct_rows[spec.id]
            row["selected_comparator_configuration"] = deepcopy(
                direct_row["selected_comparator_configuration"]
            )
            row["nonexecution_identity"] = deepcopy(direct_row["nonexecution_identity"])
        denominator.append(row)
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
    for key in (
        "comparator_tuning_authority",
        "scheduled_same_input_ids",
        "required_control_ids",
        "established_comparator_ids",
        "modern_core_ids",
        "ready_comparison_population_ids",
        "selected_comparator_configurations",
        "unavailable_comparator_nonexecution_identities",
        "scheduled_same_input_statuses",
        "comparator_selection",
    ):
        unsigned[key] = deepcopy(direct[key])
    return {**unsigned, "payload_sha256": canonical_sha256(unsigned)}


def test_final_plan_is_1760_and_all_selectable_split_is_1480_280() -> None:
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.final_runner import build_final_execution_plan

    frozen = _direct_frozen_method()
    registry = _full_registry()
    plan = build_final_execution_plan(
        frozen,
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )

    assert len(plan.entries) == 1_760
    assert sum(entry.action == "execute" for entry in plan.entries) == 1_480
    assert sum(entry.action == "not_applicable" for entry in plan.entries) == 280
    magic = next(value for value in plan.configurations if value.method_id == "magic")
    assert magic.legacy_configuration is None
    assert magic.comparator_nonexecution_identity is None
    assert magic.comparator_configuration is not None
    assert (
        magic.comparator_configuration.configuration.configuration_id
        == (
            frozen["selected_comparator_configurations"]["magic"]["configuration"][
                "configuration_id"
            ]
        )
    )
    magic_rows = [entry for entry in plan.entries if entry.run.method_id == "magic"]
    assert magic_rows
    assert all(
        direct_equal(entry.run.comparator_configuration, magic.comparator_configuration)
        for entry in magic_rows
    )
    assert all(entry.run.configuration_id != "registry-default" for entry in magic_rows)


def test_final_plan_rejects_selected_comparator_nonrun_disposition_drift() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    frozen = _direct_frozen_method()
    magic = next(row for row in frozen["method_denominator"] if row["id"] == "magic")
    magic["integration_status"] = "unavailable"
    magic["final_applicability"] = {
        "rule": "never",
        "non_run_reason": "technical_unavailable_development_attempts",
        "required_reference": None,
    }
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(
        FinalRunnerContractError,
        match="selected comparator magic.*executable",
    ):
        build_final_execution_plan(
            frozen,
            _full_registry(),
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def test_full_production_final_plan_enforces_1480_280_action_split() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    frozen = _direct_frozen_method()
    observed = next(
        row for row in frozen["method_denominator"] if row["id"] == "observed"
    )
    observed["integration_status"] = "unavailable"
    observed["final_applicability"] = {
        "rule": "never",
        "non_run_reason": "technical_unavailable_development_attempts",
        "required_reference": None,
    }
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(
        FinalRunnerContractError,
        match="1480 executable and 280 not-applicable",
    ):
        build_final_execution_plan(
            frozen,
            _full_registry(),
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def test_unavailable_stochastic_comparator_reclassifies_120_seeded_rows() -> None:
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.final_runner import build_final_execution_plan

    frozen = _direct_frozen_method(unavailable_method="biaeimpute")
    plan = build_final_execution_plan(
        frozen,
        _full_registry(),
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    rows = [entry for entry in plan.entries if entry.run.method_id == "biaeimpute"]
    expected = frozen["unavailable_comparator_nonexecution_identities"]["biaeimpute"]

    assert len(plan.entries) == 1_760
    assert len(rows) == 120
    assert {entry.run.model_seed for entry in rows} == {42, 43, 44}
    assert all(entry.action == "not_applicable" for entry in rows)
    assert all(entry.run.comparator_configuration is None for entry in rows)
    assert all(
        direct_equal(entry.run.comparator_nonexecution_identity, expected)
        for entry in rows
    )
    assert all(
        entry.run.configuration_kind == "comparator_nonexecution" for entry in rows
    )
    assert all(
        entry.run.configuration_id == "nonexecution-biaeimpute" for entry in rows
    )


def test_trajectory_always_has_44_rows_and_preserves_three_unavailable_seeds(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.final_runner import (
        build_trajectory_execution_plan,
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    frozen = _direct_frozen_method(unavailable_method="biaeimpute")
    plan = build_trajectory_execution_plan(
        frozen,
        _full_registry(),
        registered,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
        primary_final_plan_sha256="a" * 64,
    )
    rows = [entry for entry in plan.entries if entry.run.method_id == "biaeimpute"]
    expected = frozen["unavailable_comparator_nonexecution_identities"]["biaeimpute"]

    assert len(plan.entries) == 44
    assert len(rows) == 3
    assert {entry.run.model_seed for entry in rows} == {42, 43, 44}
    assert all(entry.action == "not_applicable" for entry in rows)
    assert len({entry.reason for entry in rows}) == 1
    assert all(
        direct_equal(entry.run.comparator_nonexecution_identity, expected)
        for entry in rows
    )
    assert all(
        entry.run.configuration_kind == "comparator_nonexecution" for entry in rows
    )


def test_selected_final_comparator_dispatch_uses_complete_bound_configuration() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.final_runner import build_final_execution_plan
    from maskimpute_benchmark.runner import (
        AdapterOutcome,
        FinalComparatorExecutionRequest,
        RepositoryAdapterDispatcher,
        RunnerContractError,
    )

    registry = _full_registry()
    plan = build_final_execution_plan(
        _direct_frozen_method(),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    entry = next(
        value
        for value in plan.entries
        if value.run.method_id == "magic" and value.action == "execute"
    )
    prepared = _unavailable_prepared(entry)
    assert entry.run.comparator_configuration is not None
    request = FinalComparatorExecutionRequest.create(
        registry.by_id("magic"),
        prepared.method_input,
        model_seed=entry.run.model_seed,
        configuration=entry.run.comparator_configuration,
        authority=_authority(),
        mechanism=entry.run.mechanism,
        biological_id=entry.run.biological_id,
        technical_view=entry.run.technical_view,
        dataset_id=entry.run.dataset_id,
        timeout_seconds=registry.by_id("magic").resources.timeout_seconds,
    )
    calls: list[tuple[object, ...]] = []

    def execute_direct(method_id, spec, method_input, *, seed, config):
        calls.append((method_id, spec, method_input, seed, config))
        return AdapterOutcome.unavailable("synthetic_direct_terminal")

    dispatcher = SimpleNamespace(_execute_direct_comparator=execute_direct)
    outcome = RepositoryAdapterDispatcher._execute_validated(dispatcher, request)

    assert outcome.status == "unavailable"
    assert direct_equal(request.configuration, entry.run.comparator_configuration)
    assert request.configuration.configuration.configuration_id != "registry-default"
    assert calls[0][:4] == (
        "magic",
        registry.by_id("magic"),
        prepared.method_input,
        entry.run.model_seed,
    )
    assert calls[0][4] == request.configuration.configuration.decode()
    with pytest.raises(RunnerContractError, match="resource authority"):
        RepositoryAdapterDispatcher._execute_validated(
            dispatcher,
            replace(request, max_rss_bytes=request.max_rss_bytes + 1),
        )


@pytest.mark.parametrize("mutation", ("selected_payload", "overlap", "status"))
def test_final_plan_rejects_tampered_direct_comparator_handoff(
    mutation: str,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    frozen = _direct_frozen_method()
    if mutation == "selected_payload":
        frozen["selected_comparator_configurations"]["magic"]["configuration"][
            "payload"
        ]["diffusion_time"] = 5
    elif mutation == "overlap":
        frozen["unavailable_comparator_nonexecution_identities"]["magic"] = (
            deepcopy(
                next(
                    iter(
                        frozen[
                            "unavailable_comparator_nonexecution_identities"
                        ].values()
                    )
                )
            )
            if frozen["unavailable_comparator_nonexecution_identities"]
            else {"reason": "overlap"}
        )
    else:
        row = next(
            value
            for value in frozen["scheduled_same_input_statuses"]
            if value["method_id"] == "magic"
        )
        row["selected_comparator_configuration"]["configuration"]["payload"][
            "diffusion_time"
        ] = 5
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="comparator"):
        build_final_execution_plan(
            frozen,
            _full_registry(),
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def _final_authority_receipt(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: MethodRegistry | None = None,
) -> dict[str, object]:
    from maskimpute import calibration as calibration_module

    selection_raw = (repository / "study/selection_contract.json").read_bytes()
    calibration = {"schema_version": 3, "selected_algorithm": "identity"}
    calibration_raw = (
        json.dumps(calibration, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    frozen = _receipt(_registry() if registry is None else registry)

    class FakeCalibrationArtifact:
        def __init__(self, payload):
            assert payload == calibration

        def to_dict(self):
            return dict(calibration)

    monkeypatch.setattr(
        calibration_module,
        "CalibrationArtifact",
        FakeCalibrationArtifact,
    )
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
    return frozen


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


@pytest.mark.parametrize(
    "field",
    (
        "schema_version",
        "independent_unit_count",
        "completed_count",
        "failed_count",
    ),
)
def test_final_manifest_rejects_boolean_integer_fields(field: str) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        validate_final_manifest_payload,
    )

    payload = _status_payload(_bindings())
    payload[field] = bool(payload[field])
    unsigned = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    payload["manifest_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="40-dataset"):
        validate_final_manifest_payload(payload)


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


def test_final_plan_rejects_boolean_frozen_method_schema_version() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    registry = _registry()
    receipt = _receipt(registry)
    receipt["schema_version"] = True
    unsigned = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="frozen method receipt"):
        build_final_execution_plan(
            receipt,
            registry,
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def test_final_typed_authorities_reject_unknown_actions() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalPlanEntry,
        FinalRunnerContractError,
        FrozenPlanMethodAuthority,
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
    entry = plan.entries[0]
    authority = plan.configurations[0]

    with pytest.raises(FinalRunnerContractError, match="action"):
        FinalPlanEntry(run=entry.run, action="execute_anyway", reason=None)
    with pytest.raises(FinalRunnerContractError, match="action"):
        FrozenPlanMethodAuthority(
            method_id=authority.method_id,
            legacy_configuration=authority.legacy_configuration,
            comparator_configuration=authority.comparator_configuration,
            comparator_nonexecution_identity=(
                authority.comparator_nonexecution_identity
            ),
            action="execute_anyway",
            reason=None,
            seeds=authority.seeds,
        )


def test_frozen_nonexecution_authority_snapshots_nested_direct_values() -> None:
    from maskimpute_benchmark.final_runner import FrozenPlanMethodAuthority

    source = {"nested": {"reasons": ["technical_unavailable"]}}
    authority = FrozenPlanMethodAuthority(
        method_id="dca",
        legacy_configuration=None,
        comparator_configuration=None,
        comparator_nonexecution_identity=source,
        action="not_applicable",
        reason="technical_unavailable_development_attempts",
        seeds=(None,),
    )
    source["nested"]["reasons"][0] = "mutated"

    assert authority.to_dict()["comparator_nonexecution_identity"] == {
        "nested": {"reasons": ["technical_unavailable"]}
    }


def test_final_typed_authorities_reject_noncanonical_nonrun_reasons() -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    plan = build_final_execution_plan(
        _direct_frozen_method(unavailable_method="biaeimpute"),
        _full_registry(),
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    for method_id in ("scimpute", "d3impute", "scgacl", "biaeimpute"):
        entry = next(
            value for value in plan.entries if value.run.method_id == method_id
        )
        authority = next(
            value for value in plan.configurations if value.method_id == method_id
        )
        assert entry.action == authority.action == "not_applicable"
        with pytest.raises(FinalRunnerContractError, match="reason"):
            replace(entry, reason="invented_non_run_reason")
        with pytest.raises(FinalRunnerContractError, match="reason"):
            replace(authority, reason="invented_non_run_reason")


@pytest.mark.parametrize("method_id", ("scgacl", "biaeimpute"))
def test_frozen_receipt_rejects_noncanonical_nonrun_reasons(
    method_id: str,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )

    frozen = _direct_frozen_method(
        unavailable_method="biaeimpute" if method_id == "biaeimpute" else None
    )
    row = next(
        value for value in frozen["method_denominator"] if value["id"] == method_id
    )
    row["final_applicability"]["non_run_reason"] = "invented_non_run_reason"
    unsigned = {key: value for key, value in frozen.items() if key != "payload_sha256"}
    frozen["payload_sha256"] = canonical_sha256(unsigned)

    with pytest.raises(FinalRunnerContractError, match="reason|disposition"):
        build_final_execution_plan(
            frozen,
            _full_registry(),
            _bindings(),
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
        )


def test_direct_authorities_reject_nested_nonstring_mapping_keys() -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        FrozenPlanMethodAuthority,
        build_final_execution_plan,
    )
    from maskimpute_benchmark.runner import RunnerContractError

    invalid = {"nested": {1: "must-not-be-stringified"}}
    with pytest.raises(FinalRunnerContractError, match="nonexecution authority"):
        FrozenPlanMethodAuthority(
            method_id="biaeimpute",
            legacy_configuration=None,
            comparator_configuration=None,
            comparator_nonexecution_identity=invalid,
            action="not_applicable",
            reason="technical_unavailable_development_attempts",
            seeds=(42, 43, 44),
        )

    plan = build_final_execution_plan(
        _direct_frozen_method(unavailable_method="biaeimpute"),
        _full_registry(),
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    run = next(
        value.run for value in plan.entries if value.run.method_id == "biaeimpute"
    )
    serialized = run.to_dict()
    serialized["comparator_nonexecution_identity"] = invalid
    with pytest.raises(RunnerContractError, match="nonexecution plan identity"):
        replace(
            run,
            comparator_nonexecution_identity=(
                serialized["comparator_nonexecution_identity"]
            ),
        )


def test_trajectory_plan_is_exactly_one_registered_supplementary_denominator(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        build_trajectory_execution_plan,
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    registry = _registry()

    plan = build_trajectory_execution_plan(
        _receipt(registry),
        registry,
        registered,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
        primary_final_plan_sha256="a" * 64,
    )

    assert plan.scope == "supplementary_trajectory"
    assert {entry.run.dataset_id for entry in plan.entries} == {
        registered.binding.dataset_id
    }
    by_method: dict[str, list[object]] = {}
    for entry in plan.entries:
        by_method.setdefault(entry.run.method_id, []).append(entry)
        assert entry.run.run_id.startswith("trajectory-")
        spec = registry.by_id(entry.run.method_id)
        assert spec.resources.timeout_seconds > 0
    assert len(by_method["observed"]) == 1
    assert len(by_method["maskimpute"]) == 3
    assert {entry.run.model_seed for entry in by_method["maskimpute"]} == {42, 43, 44}
    assert len(by_method["magic"]) == 3
    assert len(by_method["scimpute"]) == 1
    assert by_method["scimpute"][0].action == "not_applicable"
    assert plan.input_hashes["trajectory_authority_sha256"] == (
        registered.authority.authority_sha256
    )
    assert plan.input_hashes["trajectory_binding_sha256"] == (
        registered.authority.binding_sha256
    )
    assert plan.input_hashes["trajectory_dataset_sha256"] == (
        registered.binding.dataset_sha256
    )
    assert (
        plan.input_hashes["trajectory_dataset_receipt_sha256"]
        == (registered.receipt["receipt_sha256"])
    )
    assert plan.input_hashes["primary_final_plan_sha256"] == "a" * 64


@pytest.mark.parametrize("mutation", ("schema_version", "ordinal", "empty"))
def test_trajectory_plan_payload_rejects_boolean_aliases_and_empty_plan(
    mutation: str,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        TrajectoryExecutionPlan,
        build_final_execution_plan,
        trajectory_execution_plan_payload,
    )
    from maskimpute_benchmark.runner import DEVELOPMENT_MODEL_SEEDS

    registry = _registry()
    primary = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    entry = primary.entries[0]

    def plan_with(
        *,
        schema_version: object = 1,
        entries: tuple[object, ...] = (entry,),
    ) -> TrajectoryExecutionPlan:
        body = {
            "schema_version": schema_version,
            "scope": "supplementary_trajectory",
            "input_hashes": dict(primary.input_hashes),
            "entries": [value.to_dict() for value in entries],
            "configurations": [value.to_dict() for value in primary.configurations],
            "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
        }
        return TrajectoryExecutionPlan(
            schema_version=schema_version,
            scope="supplementary_trajectory",
            input_hashes=primary.input_hashes,
            entries=entries,
            configurations=primary.configurations,
            plan_sha256=canonical_sha256(body),
        )

    assert trajectory_execution_plan_payload(plan_with())["schema_version"] == 1
    changed_entry = (
        replace(entry, run=replace(entry.run, ordinal=True))
        if mutation == "ordinal"
        else entry
    )
    changed = plan_with(
        schema_version=True if mutation == "schema_version" else 1,
        entries=() if mutation == "empty" else (changed_entry,),
    )
    with pytest.raises(FinalRunnerContractError, match="trajectory plan"):
        trajectory_execution_plan_payload(changed)


@pytest.mark.parametrize(
    ("scope", "population"),
    (
        ("final", "empty"),
        ("trajectory", "empty"),
        ("final", "incomplete"),
        ("trajectory", "incomplete"),
        ("final", "invalid_configuration"),
        ("trajectory", "invalid_configuration"),
        ("final", "invalid_entry"),
        ("trajectory", "invalid_entry"),
        ("final", "invalid_run"),
        ("trajectory", "invalid_run"),
    ),
)
def test_publication_evaluation_rejects_empty_and_incomplete_plan_populations(
    scope: str,
    population: str,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalExecutionPlan,
        FinalPlanEntry,
        FinalRunnerContractError,
        TrajectoryExecutionPlan,
        build_final_execution_plan,
        validate_final_execution_for_evaluation,
        validate_trajectory_execution_for_evaluation,
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
    if population == "empty":
        entries = ()
    elif population == "invalid_configuration":
        entries = full.entries
    elif population == "invalid_entry":
        entries = (object(),)
    elif population == "invalid_run":
        entries = (FinalPlanEntry(run=object(), action="execute", reason=None),)
    else:
        entries = (full.entries[0],)
    configurations = (
        (object(),) if population == "invalid_configuration" else full.configurations
    )
    records: tuple[object, ...] = ()
    if entries and population not in {
        "invalid_configuration",
        "invalid_entry",
        "invalid_run",
    }:
        attempt = _unavailable_attempt(entries[0])
        records = (
            {
                "run": asdict(attempt.run),
                "metrics": [metric.to_dict() for metric in attempt.metrics],
                "execution_request": None,
            },
        )
    if scope == "final":
        plan = FinalExecutionPlan(
            schema_version=1,
            input_hashes=full.input_hashes,
            entries=entries,
            configurations=configurations,
            plan_sha256="a" * 64,
        )
        validator = validate_final_execution_for_evaluation
    else:
        plan = TrajectoryExecutionPlan(
            schema_version=1,
            scope="supplementary_trajectory",
            input_hashes=full.input_hashes,
            entries=entries,
            configurations=configurations,
            plan_sha256="b" * 64,
        )
        validator = validate_trajectory_execution_for_evaluation
    with pytest.raises(FinalRunnerContractError, match="plan|population|denominator"):
        validator(plan, records)


@pytest.mark.parametrize("field", ("configuration_id", "configuration_kind"))
def test_publication_population_rejects_entry_configuration_identity_drift(
    field: str,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark import final_runner

    registry = _registry()
    full = final_runner.build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    replacement = (
        "wrong-configuration"
        if field == "configuration_id"
        else next(
            value
            for value in ("registry", "candidate_search", "ablation")
            if value != full.entries[0].run.configuration_kind
        )
    )
    changed_run = replace(
        full.entries[0].run,
        **{field: replacement},
    )
    changed_plan = replace(
        full,
        entries=(
            replace(full.entries[0], run=changed_run),
            *full.entries[1:],
        ),
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="plan|population",
    ):
        final_runner._require_complete_publication_plan_population(changed_plan)


def test_publication_population_rejects_changed_seed_denominator() -> None:
    from dataclasses import replace

    from maskimpute_benchmark import final_runner

    registry = _registry()
    full = final_runner.build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    configuration = full.configurations[0]
    changed_configuration = replace(
        configuration,
        seeds=(*configuration.seeds, 999),
    )
    templates = {
        entry.run.dataset_id: entry
        for entry in full.entries
        if entry.run.method_id == configuration.method_id
    }
    entries = list(full.entries)
    for template in templates.values():
        ordinal = len(entries) + 1
        entries.append(
            replace(
                template,
                run=replace(
                    template.run,
                    ordinal=ordinal,
                    run_id=f"expanded-denominator-{ordinal}",
                    model_seed=999,
                ),
            )
        )
    changed_plan = replace(
        full,
        entries=tuple(entries),
        configurations=(changed_configuration, *full.configurations[1:]),
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="plan|population",
    ):
        final_runner._require_complete_publication_plan_population(changed_plan)


@lru_cache(maxsize=1)
def _complete_publication_final_plan():
    from maskimpute_benchmark.final_runner import build_final_execution_plan

    registry = _full_registry()
    return build_final_execution_plan(
        _direct_frozen_method(),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )


@lru_cache(maxsize=1)
def _complete_publication_trajectory_plan():
    from dataclasses import replace

    from maskimpute_benchmark import final_runner
    from maskimpute_benchmark.trajectory_dataset import (
        REGISTERED_TRAJECTORY_DATASET_ID,
    )

    primary = _complete_publication_final_plan()
    source_dataset = primary.entries[0].run.dataset_id
    source_by_method_seed = {
        (entry.run.method_id, entry.run.model_seed): entry
        for entry in primary.entries
        if entry.run.dataset_id == source_dataset
    }
    entries = []
    ordinal = 0
    for configuration in primary.configurations:
        for seed in configuration.seeds:
            ordinal += 1
            template = source_by_method_seed[(configuration.method_id, seed)]
            run = replace(
                template.run,
                ordinal=ordinal,
                run_id=final_runner._trajectory_run_id(
                    ordinal,
                    configuration.method_id,
                    REGISTERED_TRAJECTORY_DATASET_ID,
                    seed,
                    configuration,
                ),
                dataset_id=REGISTERED_TRAJECTORY_DATASET_ID,
                mechanism="synthetic_trajectory",
                biological_id="trajectory-draw-01",
                technical_view="deterministic-count-allocation",
            )
            entries.append(replace(template, run=run))
    return final_runner.TrajectoryExecutionPlan(
        schema_version=1,
        scope="supplementary_trajectory",
        input_hashes=primary.input_hashes,
        entries=tuple(entries),
        configurations=primary.configurations,
        plan_sha256="b" * 64,
    )


@pytest.mark.parametrize("scope", ("final", "trajectory"))
def test_publication_evaluation_accepts_complete_canonical_typed_population(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    from maskimpute_benchmark import final_runner

    monkeypatch.setattr(
        final_runner,
        "_validate_frozen_execution_for_evaluation",
        lambda _plan, _records: {"status": "validated-control"},
    )
    if scope == "final":
        observed = final_runner.validate_final_execution_for_evaluation(
            _complete_publication_final_plan(),
            (),
        )
    else:
        observed = final_runner.validate_trajectory_execution_for_evaluation(
            _complete_publication_trajectory_plan(),
            (),
        )

    assert observed == {"status": "validated-control"}


@pytest.mark.parametrize(
    ("scope", "replacement"),
    (
        ("final", "renamed-complete-final-dataset"),
        ("trajectory", "forged-trajectory-dataset"),
    ),
)
def test_publication_evaluation_rejects_forged_complete_dataset_block(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    replacement: str,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark import final_runner

    monkeypatch.setattr(
        final_runner,
        "_validate_frozen_execution_for_evaluation",
        lambda _plan, _records: {"status": "must-not-be-reached"},
    )
    original = (
        _complete_publication_final_plan()
        if scope == "final"
        else _complete_publication_trajectory_plan()
    )
    target = original.entries[0].run.dataset_id
    entries = tuple(
        replace(entry, run=replace(entry.run, dataset_id=replacement))
        if entry.run.dataset_id == target
        else entry
        for entry in original.entries
    )
    changed = replace(original, entries=entries)
    validator = (
        final_runner.validate_final_execution_for_evaluation
        if scope == "final"
        else final_runner.validate_trajectory_execution_for_evaluation
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="plan|population",
    ):
        validator(changed, ())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("mechanism", "forged-mechanism"),
        ("biological_id", "draw-99"),
        ("technical_view", "forged-view"),
        ("preflight_status", "blocked_authority"),
        ("preflight_reason", "forged-preflight-reason"),
    ),
)
def test_publication_population_rejects_scientific_and_preflight_mutations(
    field: str,
    value: str,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark import final_runner

    original = _complete_publication_final_plan()
    changed_run = replace(original.entries[0].run, **{field: value})
    changed = replace(
        original,
        entries=(replace(original.entries[0], run=changed_run), *original.entries[1:]),
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="plan|population",
    ):
        final_runner._require_complete_publication_plan_population(changed)


@pytest.mark.parametrize(
    "boundary",
    ("plan", "entry", "run", "configuration", "nested_configuration"),
)
def test_publication_population_rejects_derived_authority_types(
    boundary: str,
) -> None:
    from dataclasses import fields, replace

    from maskimpute_benchmark import final_runner
    from maskimpute_benchmark.runner import AuthorizedConfiguration, RunPlanEntry

    class DerivedFinalExecutionPlan(final_runner.FinalExecutionPlan):
        pass

    class DerivedFinalPlanEntry(final_runner.FinalPlanEntry):
        pass

    class DerivedRunPlanEntry(RunPlanEntry):
        pass

    class DerivedFrozenPlanMethodAuthority(final_runner.FrozenPlanMethodAuthority):
        pass

    class DerivedAuthorizedConfiguration(AuthorizedConfiguration):
        pass

    def derived(cls, value):
        return cls(**{item.name: getattr(value, item.name) for item in fields(value)})

    original = _complete_publication_final_plan()
    changed = original
    if boundary == "plan":
        changed = derived(DerivedFinalExecutionPlan, original)
    elif boundary == "entry":
        entry = derived(DerivedFinalPlanEntry, original.entries[0])
        changed = replace(original, entries=(entry, *original.entries[1:]))
    elif boundary == "run":
        run = derived(DerivedRunPlanEntry, original.entries[0].run)
        changed = replace(
            original,
            entries=(replace(original.entries[0], run=run), *original.entries[1:]),
        )
    elif boundary == "configuration":
        configuration = derived(
            DerivedFrozenPlanMethodAuthority,
            original.configurations[0],
        )
        changed = replace(
            original,
            configurations=(configuration, *original.configurations[1:]),
        )
    else:
        configuration = original.configurations[0]
        assert configuration.legacy_configuration is not None
        nested = derived(
            DerivedAuthorizedConfiguration,
            configuration.legacy_configuration,
        )
        changed_configuration = replace(
            configuration,
            legacy_configuration=nested,
        )
        changed = replace(
            original,
            configurations=(changed_configuration, *original.configurations[1:]),
        )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="plan|population",
    ):
        final_runner._require_complete_publication_plan_population(changed)


def test_trajectory_plan_rejects_nearby_registered_identity(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_trajectory_execution_plan,
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    nearby = replace(
        registered,
        authority=replace(
            registered.authority,
            source_id="registered-synthetic-trajectory-v2",
        ),
    )
    registry = _registry()

    with pytest.raises(FinalRunnerContractError, match="trajectory.*identity"):
        build_trajectory_execution_plan(
            _receipt(registry),
            registry,
            nearby,
            execution_claim_sha256="7" * 64,
            execution_environment_sha256="8" * 64,
            execution_authority_sha256="9" * 64,
            primary_final_plan_sha256="a" * 64,
        )


def test_registered_trajectory_dataset_publication_crash_rebuilds_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.runner import CheckpointStore
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    real_publish = CheckpointStore._publish_immutable
    interrupted = False

    def interrupt_after_dataset(self, relative_path, payload):
        nonlocal interrupted
        result = real_publish(self, relative_path, payload)
        if (
            relative_path == "results/trajectory/dataset/evaluator.h5ad"
            and not interrupted
        ):
            interrupted = True
            raise RuntimeError("crash after trajectory dataset publication")
        return result

    monkeypatch.setattr(
        CheckpointStore,
        "_publish_immutable",
        interrupt_after_dataset,
    )
    with pytest.raises(RuntimeError, match="trajectory dataset publication"):
        materialize_prepared_trajectory_dataset(repository, round_dir)
    dataset = round_dir / "results/trajectory/dataset/evaluator.h5ad"
    receipt = round_dir / "results/trajectory/dataset/dataset_receipt.json"
    assert dataset.is_file()
    assert not receipt.exists()

    monkeypatch.setattr(CheckpointStore, "_publish_immutable", real_publish)
    resumed = materialize_prepared_trajectory_dataset(repository, round_dir)
    assert receipt.is_file()
    assert (
        resumed.binding.dataset_file_sha256
        == hashlib.sha256(dataset.read_bytes()).hexdigest()
    )

    dataset.unlink()
    with pytest.raises(FinalRunnerContractError, match="dataset.*unavailable"):
        materialize_prepared_trajectory_dataset(repository, round_dir)


def test_load_prepared_trajectory_dataset_is_strictly_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    expected = final_runner.materialize_prepared_trajectory_dataset(
        repository,
        round_dir,
    )
    before = _read_only_tree_snapshot(repository)

    def forbid_mutation(*_args, **_kwargs):
        raise AssertionError("read-only trajectory load attempted mutation")

    monkeypatch.setattr(
        final_runner,
        "generate_registered_trajectory_dataset",
        forbid_mutation,
    )
    monkeypatch.setattr(
        final_runner,
        "_remove_unreceipted_trajectory_dataset",
        forbid_mutation,
    )
    monkeypatch.setattr(
        final_runner.CheckpointStore,
        "_publish_immutable",
        forbid_mutation,
    )

    observed = final_runner.load_prepared_trajectory_dataset(repository, round_dir)

    assert observed.binding == expected.binding
    assert observed.receipt == expected.receipt
    assert observed.prepared.method_input == expected.prepared.method_input
    assert _read_only_tree_snapshot(repository) == before


@pytest.mark.parametrize(
    "missing_relative",
    (
        "results/trajectory/dataset/evaluator.h5ad",
        "results/trajectory/dataset/dataset_receipt.json",
    ),
)
def test_load_prepared_trajectory_dataset_never_recovers_a_missing_half(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_relative: str,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    final_runner.materialize_prepared_trajectory_dataset(repository, round_dir)
    missing = round_dir / missing_relative
    missing.unlink()
    before = _read_only_tree_snapshot(repository)

    def forbid_mutation(*_args, **_kwargs):
        raise AssertionError("missing trajectory half was regenerated")

    monkeypatch.setattr(
        final_runner,
        "generate_registered_trajectory_dataset",
        forbid_mutation,
    )
    monkeypatch.setattr(
        final_runner,
        "_remove_unreceipted_trajectory_dataset",
        forbid_mutation,
    )
    monkeypatch.setattr(
        final_runner.CheckpointStore,
        "_publish_immutable",
        forbid_mutation,
    )

    with pytest.raises(final_runner.FinalRunnerContractError, match="dataset.*pair"):
        final_runner.load_prepared_trajectory_dataset(repository, round_dir)

    assert not missing.exists()
    assert _read_only_tree_snapshot(repository) == before


def test_load_prepared_trajectory_dataset_rejects_a_receipt_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    final_runner.materialize_prepared_trajectory_dataset(repository, round_dir)
    receipt_path = round_dir / "results/trajectory/dataset/dataset_receipt.json"
    raced_raw = b'{"raced":true}\n'
    real_read = final_runner._read_unique_file
    raced = False

    def replace_after_first_receipt_read(path, name, **kwargs):
        nonlocal raced
        raw = real_read(path, name, **kwargs)
        if path == receipt_path and not raced:
            raced = True
            replacement = receipt_path.with_name("raced-receipt.json")
            replacement.write_bytes(raced_raw)
            os.replace(replacement, receipt_path)
        return raw

    monkeypatch.setattr(
        final_runner, "_read_unique_file", replace_after_first_receipt_read
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="changed|receipt",
    ):
        final_runner.load_prepared_trajectory_dataset(repository, round_dir)

    assert raced
    assert receipt_path.read_bytes() == raced_raw


def test_load_prepared_trajectory_dataset_rejects_a_transient_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    registered = final_runner.materialize_prepared_trajectory_dataset(
        repository,
        round_dir,
    )
    dataset_parent = round_dir / "results/trajectory/dataset"
    dataset_path = dataset_parent / "evaluator.h5ad"
    receipt_name = "dataset_receipt.json"
    bound_parent = dataset_parent.with_name("receipt-bound-dataset")
    decoy_parent = dataset_parent.with_name("stable-decoy-dataset")
    os.replace(dataset_parent, bound_parent)
    dataset_parent.mkdir()
    (dataset_parent / receipt_name).write_bytes(
        (bound_parent / receipt_name).read_bytes()
    )
    dataset_path.write_bytes(b"stable non-bound evaluator bytes\n")
    stable_decoy_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert stable_decoy_sha256 != registered.binding.dataset_file_sha256

    real_read = final_runner._read_bound_h5ad
    swapped = False

    def read_during_transient_parent_replacement(path, binding, **kwargs):
        nonlocal swapped
        os.replace(dataset_parent, decoy_parent)
        os.replace(bound_parent, dataset_parent)
        try:
            swapped = True
            return real_read(path, binding, **kwargs)
        finally:
            os.replace(dataset_parent, bound_parent)
            os.replace(decoy_parent, dataset_parent)

    monkeypatch.setattr(
        final_runner,
        "_read_bound_h5ad",
        read_during_transient_parent_replacement,
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="checksum",
    ):
        final_runner.load_prepared_trajectory_dataset(repository, round_dir)

    assert swapped
    assert hashlib.sha256(dataset_path.read_bytes()).hexdigest() == stable_decoy_sha256
    assert stable_decoy_sha256 != registered.binding.dataset_file_sha256


def test_registered_trajectory_dataset_before_journal_reconciles_and_resumes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        _record_incremental_results_if_changed,
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.simulators import (
        SimulationContractError,
        load_final_manifest_claim,
    )
    from maskimpute_benchmark.study import record_incremental_results

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    first = materialize_prepared_trajectory_dataset(repository, round_dir)
    with pytest.raises(SimulationContractError, match="clean frozen|valid claimed"):
        load_final_manifest_claim(repository, round_dir)

    second = materialize_prepared_trajectory_dataset(repository, round_dir)
    assert second.binding == first.binding
    _record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"


def test_trajectory_execution_authority_is_separate_and_names_registered_input(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        materialize_prepared_trajectory_dataset,
        materialize_trajectory_execution_authority,
    )
    from maskimpute_benchmark.runner import (
        ExecutionAuthorityContext,
        method_input_sha256,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    calibration_path = repository / "primary/retained_calibration.json"
    score_path = repository / "primary/count_score_authority.json"
    calibration_path.parent.mkdir()
    calibration_path.write_bytes(b"{}\n")
    score_path.write_bytes(b"{}\n")
    frozen = _receipt(_registry())
    base = frozen["selected_configuration"]["hyperparameters"]
    count = {"n_folds": 2}
    primary = ExecutionAuthorityContext(
        authority_sha256="9" * 64,
        base_configuration_json=json.dumps(base, separators=(",", ":"), sort_keys=True),
        base_configuration_sha256=canonical_sha256(base),
        count_model_config_json=json.dumps(
            count, separators=(",", ":"), sort_keys=True
        ),
        count_model_config_sha256=canonical_sha256(count),
        count_score_manifest_path="primary/count_score_authority.json",
        count_score_manifest_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest(),
        retained_calibration_path="primary/retained_calibration.json",
        retained_calibration_sha256=hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
    )
    authority = materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen,
        registered,
        primary,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        primary_final_plan_sha256="a" * 64,
    )
    resumed = materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen,
        registered,
        primary,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        primary_final_plan_sha256="a" * 64,
    )

    assert resumed == authority
    assert authority != primary
    assert authority.count_score_manifest_path.endswith(
        "results/trajectory/execution_authority/count_score_authority.json"
    )
    assert authority.retained_calibration_path.endswith(
        "results/trajectory/execution_authority/retained_calibration.json"
    )
    score = json.loads(
        (repository / authority.count_score_manifest_path).read_text(encoding="utf-8")
    )
    assert score["artifact_type"] == "maskimpute_trajectory_count_score_authority"
    assert score["scope"] == "truth_free_registered_trajectory_inference"
    assert score["trajectory_authority_sha256"] == (registered.binding.authority_sha256)
    assert score["trajectory_binding_sha256"] == (
        registered.binding.registered_binding_sha256
    )
    assert score["trajectory_method_input_sha256"] == method_input_sha256(
        registered.prepared.method_input
    )


def test_trajectory_score_policy_rederives_only_for_registered_identity(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
        build_final_execution_plan,
        build_trajectory_execution_plan,
        materialize_prepared_trajectory_dataset,
        materialize_trajectory_execution_authority,
    )
    from maskimpute_benchmark.runner import _derive_prezero_execution_authority
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    frozen = _receipt(_registry())
    primary_plan = build_final_execution_plan(
        frozen,
        _registry(),
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    primary_entry = next(
        entry for entry in primary_plan.entries if entry.run.method_id == "maskimpute"
    )
    primary_prepared = _unavailable_prepared(
        primary_entry,
        count_model_ready=True,
    )
    primary_authority, _calibration = _write_real_final_score_authority(
        repository,
        primary_prepared,
    )
    base = frozen["selected_configuration"]["hyperparameters"]
    primary_authority = replace(
        primary_authority,
        base_configuration_json=json.dumps(base, separators=(",", ":"), sort_keys=True),
        base_configuration_sha256=canonical_sha256(base),
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    trajectory_authority = materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen,
        registered,
        primary_authority,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        primary_final_plan_sha256=primary_plan.plan_sha256,
    )
    trajectory_plan = build_trajectory_execution_plan(
        frozen,
        _registry(),
        registered,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256=trajectory_authority.authority_sha256,
        primary_final_plan_sha256=primary_plan.plan_sha256,
    )
    entry = next(
        value
        for value in trajectory_plan.entries
        if value.run.method_id == "maskimpute"
    )

    probability, policy = _derive_prezero_execution_authority(
        registered.prepared,
        entry.run,
        trajectory_authority,
        calibration_usage="retained_all_development",
        repository=repository,
    )

    assert probability.shape == registered.prepared.method_input.shape
    assert policy["calibration_scope"] == (
        "retained_all_development_for_registered_trajectory_inference"
    )
    with pytest.raises(Exception, match="trajectory|authority|identity"):
        _derive_prezero_execution_authority(
            registered.prepared,
            replace(entry.run, mechanism="synthetic-trajectory"),
            trajectory_authority,
            calibration_usage="retained_all_development",
            repository=repository,
        )


def _trajectory_store_inputs(tmp_path: Path):
    from maskimpute_benchmark.final_runner import (
        build_trajectory_execution_plan,
        materialize_prepared_trajectory_dataset,
        materialize_trajectory_execution_authority,
    )
    from maskimpute_benchmark.runner import ExecutionAuthorityContext
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    frozen = _receipt(_registry())
    base = frozen["selected_configuration"]["hyperparameters"]
    count = {"n_folds": 2}
    calibration_path = repository / "primary/retained_calibration.json"
    score_path = repository / "primary/count_score_authority.json"
    calibration_path.parent.mkdir()
    calibration_path.write_bytes(b"{}\n")
    score_path.write_bytes(b"{}\n")
    claim = {"schema_version": 1, "fixture": "trajectory-store-claim"}
    (round_dir / "execution_claim.json").write_text(
        json.dumps(claim, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    claim_sha256 = canonical_sha256(claim)
    environment_sha256 = "8" * 64
    primary_authority_body = {
        "schema_version": 1,
        "authority_type": "maskimpute_frozen_final_execution",
        "frozen_method_sha256": frozen["payload_sha256"],
        "runtime_lock_sha256": frozen["runtime_lock_sha256"],
        "execution_claim_sha256": claim_sha256,
        "execution_environment_sha256": environment_sha256,
        "dataset_manifest_sha256": "1" * 64,
        "base_configuration": base,
        "base_configuration_sha256": canonical_sha256(base),
        "count_model_config": count,
        "count_model_config_sha256": canonical_sha256(count),
        "count_score_authority_path": "primary/count_score_authority.json",
        "count_score_authority_sha256": hashlib.sha256(
            score_path.read_bytes()
        ).hexdigest(),
        "retained_calibration_path": "primary/retained_calibration.json",
        "retained_calibration_sha256": hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
        "calibration_usage": "retained_all_development_calibrator",
    }
    primary_authority_sha256 = canonical_sha256(primary_authority_body)
    primary_authority_payload = {
        **primary_authority_body,
        "authority_sha256": primary_authority_sha256,
    }
    primary_authority_path = (
        round_dir / "results/final/execution_authority/authority.json"
    )
    primary_authority_path.parent.mkdir(parents=True)
    primary_authority_path.write_text(
        json.dumps(
            primary_authority_payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    primary = ExecutionAuthorityContext(
        authority_sha256=primary_authority_sha256,
        base_configuration_json=json.dumps(base, separators=(",", ":"), sort_keys=True),
        base_configuration_sha256=canonical_sha256(base),
        count_model_config_json=json.dumps(
            count, separators=(",", ":"), sort_keys=True
        ),
        count_model_config_sha256=canonical_sha256(count),
        count_score_manifest_path="primary/count_score_authority.json",
        count_score_manifest_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest(),
        retained_calibration_path="primary/retained_calibration.json",
        retained_calibration_sha256=hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
    )
    primary_manifest_body = {
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": "a" * 64,
        "input_hashes": {
            "frozen_method_sha256": frozen["payload_sha256"],
            "method_registry_sha256": frozen["method_registry_sha256"],
            "runtime_lock_sha256": frozen["runtime_lock_sha256"],
            "dataset_manifest_sha256": "1" * 64,
            "dataset_design_sha256": "2" * 64,
            "dataset_seed_source_sha256": "3" * 64,
            "protocol_sha256": "4" * 64,
            "execution_claim_sha256": claim_sha256,
            "execution_environment_sha256": environment_sha256,
            "execution_authority_sha256": primary_authority_sha256,
        },
        "planned_run_count": 0,
        "recorded_run_count": 0,
        "records": [],
        "artifact_storage": {
            "evaluator_output_compression_level": 6,
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "native_output_retention": "omitted_redundant_final_output",
            "p_pre_zero_compression_level": 6,
            "p_pre_zero_encoding": "zlib_raw_f64_v1",
        },
    }
    primary_manifest = {
        **primary_manifest_body,
        "manifest_sha256": canonical_sha256(primary_manifest_body),
    }
    primary_manifest_path = (
        round_dir / "results/final/execution/execution_manifest.json"
    )
    primary_manifest_path.parent.mkdir(parents=True)
    primary_manifest_path.write_text(
        json.dumps(primary_manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    authority = materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen,
        registered,
        primary,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environment_sha256,
        primary_final_plan_sha256="a" * 64,
    )
    plan = build_trajectory_execution_plan(
        frozen,
        _registry(),
        registered,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environment_sha256,
        execution_authority_sha256=authority.authority_sha256,
        primary_final_plan_sha256="a" * 64,
    )
    return repository, round_dir, registered, authority, plan


def test_trajectory_result_store_is_separate_resumable_and_complete(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        FinalRunnerContractError,
    )

    repository, round_dir, registered, authority, plan = _trajectory_store_inputs(
        tmp_path
    )
    output = round_dir / "results/trajectory/execution"
    prepared = {registered.binding.dataset_id: registered.prepared}
    with pytest.raises(FinalRunnerContractError, match="authority"):
        FinalResultStore(
            output,
            plan,
            prepared,
            _authority(),
            authority_repository=repository,
        )
    store = FinalResultStore(
        output,
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )

    for entry in plan.entries:
        store.append(
            entry,
            _unavailable_attempt(entry, prepared=registered.prepared),
            execution_request=_final_comparator_request(
                entry,
                _registry(),
                registered.prepared,
                authority,
            ),
        )
    manifest = store.finalize()
    resumed = FinalResultStore(
        output,
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )

    assert manifest["scope"] == "supplementary_trajectory"
    assert manifest["recorded_run_count"] == len(plan.entries)
    assert len(resumed.load_records()) == len(plan.entries)
    assert resumed.load_manifest() == manifest
    assert not (round_dir / "results/final/execution/records").exists()


def test_trajectory_manifest_publication_is_resumable_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.final_runner import FinalResultStore

    repository, round_dir, registered, authority, plan = _trajectory_store_inputs(
        tmp_path
    )
    output = round_dir / "results/trajectory/execution"
    prepared = {registered.binding.dataset_id: registered.prepared}
    store = FinalResultStore(
        output,
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )
    for entry in plan.entries:
        store.append(
            entry,
            _unavailable_attempt(entry, prepared=registered.prepared),
            execution_request=_final_comparator_request(
                entry,
                _registry(),
                registered.prepared,
                authority,
            ),
        )

    def interrupt_after_manifest() -> dict[str, object]:
        raise RuntimeError("interrupted after trajectory manifest")

    monkeypatch.setattr(store, "load_manifest", interrupt_after_manifest)
    with pytest.raises(RuntimeError, match="after trajectory manifest"):
        store.finalize()
    assert store.manifest_path.is_file()
    resumed = FinalResultStore(
        output,
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )
    assert resumed.load_manifest()["status"] == "completed"


def test_execute_trajectory_plan_reuses_executor_and_retains_terminal_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        _owned_final_result_file_manifest,
        _rederive_trajectory_evidence_before_receipt,
        _trajectory_evaluation_evidence,
        _validate_frozen_execution_for_evaluation,
        execute_trajectory_plan,
        final_result_file_manifest,
    )
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.runner import (
        AdapterOutcome,
        FinalComparatorExecutionRequest,
    )

    fixture = _exact_primary_trajectory_chain_inputs(
        tmp_path,
        monkeypatch,
        full_denominator=True,
    )
    repository = fixture["repository"]
    round_dir = fixture["round_dir"]
    registered = fixture["registered"]
    authority = fixture["trajectory"]
    plan = fixture["trajectory_plan"]
    registry = fixture["registry"]
    primary_plan = fixture["primary_plan"]
    assert isinstance(repository, Path)
    assert isinstance(round_dir, Path)
    prepared = {registered.binding.dataset_id: registered.prepared}
    store = FinalResultStore(
        round_dir / "results/trajectory/execution",
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )
    calls: list[str] = []
    requests: list[object] = []
    publications = 0

    def executor(request):
        calls.append(request.method_spec.id)
        requests.append(request)
        return AdapterOutcome.unavailable("test_terminal_unavailable")

    def published() -> None:
        nonlocal publications
        publications += 1

    manifest = execute_trajectory_plan(
        plan,
        registry,
        prepared,
        authority,
        executor,
        store,
        on_record_published=published,
    )
    records = store.load_records()
    validation = _validate_frozen_execution_for_evaluation(plan, records)
    cumulative = _owned_final_result_file_manifest(round_dir)["result_files"]
    evidence = _trajectory_evaluation_evidence(
        round_dir,
        plan,
        registered,
        authority,
        store,
        validation,
        cumulative,
    )

    assert len(calls) == sum(entry.action == "execute" for entry in plan.entries)
    magic_requests = [
        request
        for request in requests
        if isinstance(request, FinalComparatorExecutionRequest)
    ]
    assert magic_requests
    assert {request.method_spec.id for request in magic_requests} == {"magic"}
    for entry, record in zip(plan.entries, records, strict=True):
        if entry.action == "not_applicable":
            assert record["execution_request"] is None
        elif entry.run.comparator_configuration is not None:
            assert "configuration_sha256" not in record["run"]
            assert "configuration_sha256" not in record["execution_request"]
            assert all(
                "configuration_sha256" not in metric for metric in record["metrics"]
            )
            assert direct_equal(
                record["run"]["comparator_configuration"],
                entry.run.to_dict()["comparator_configuration"],
            )
            assert direct_equal(
                record["execution_request"]["configuration"],
                entry.run.to_dict()["comparator_configuration"],
            )
    assert publications == 1
    assert manifest["scope"] == "supplementary_trajectory"
    assert validation["scope"] == "supplementary_trajectory"
    assert validation["planned_run_count"] == len(plan.entries)
    assert validation["not_applicable_count"] == sum(
        entry.action == "not_applicable" for entry in plan.entries
    )
    assert evidence["execution_validation"] == validation
    assert {row["path"] for row in evidence["result_files"]} == {
        row["path"]
        for row in cumulative
        if row["path"].startswith("results/trajectory/")
    }
    before_replay = _read_only_tree_snapshot(repository)

    def forbid_materialization(*_args, **_kwargs):
        raise AssertionError("trajectory evidence replay attempted materialization")

    monkeypatch.setattr(
        final_runner,
        "materialize_prepared_trajectory_dataset",
        forbid_materialization,
    )
    binding_fields = downstream._validated_trajectory_binding_fields(
        repository,
        round_dir,
        {
            "trajectory_evidence": evidence,
            "result_files": cumulative,
            "final_plan_sha256": primary_plan.plan_sha256,
        },
    )
    assert len(plan.entries) == 8
    assert len(evidence["result_files"]) == 30
    assert binding_fields["trajectory_plan_sha256"] == plan.plan_sha256
    assert binding_fields["trajectory_planned_run_count"] == 8
    assert binding_fields["trajectory_result_file_count"] == 30
    assert _read_only_tree_snapshot(repository) == before_replay
    assert (
        _rederive_trajectory_evidence_before_receipt(
            repository,
            round_dir,
            evidence,
            cumulative,
            primary_final_plan_sha256=primary_plan.plan_sha256,
        )
        == evidence
    )
    assert _read_only_tree_snapshot(repository) == before_replay
    stdout = next(
        round_dir / row["path"]
        for row in evidence["result_files"]
        if row["path"].endswith(".stdout")
    )
    stdout.write_bytes(stdout.read_bytes() + b"changed-before-receipt")
    fresh_files = final_result_file_manifest(round_dir)["result_files"]
    with pytest.raises(Exception, match="record|artifact|changed"):
        _rederive_trajectory_evidence_before_receipt(
            repository,
            round_dir,
            evidence,
            fresh_files,
            primary_final_plan_sha256=primary_plan.plan_sha256,
        )


def test_pre_receipt_rederivation_uses_exact_running_runtime_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.study as study

    fixture = _exact_primary_trajectory_chain_inputs(
        tmp_path,
        monkeypatch,
        full_denominator=True,
    )
    repository = fixture["repository"]
    round_dir = fixture["round_dir"]
    registered = fixture["registered"]
    authority = fixture["trajectory"]
    plan = fixture["trajectory_plan"]
    primary_plan = fixture["primary_plan"]
    assert isinstance(repository, Path)
    assert isinstance(round_dir, Path)
    simulator_assets_root = tmp_path / "external-simulator-assets"
    simulator_r_environment = tmp_path / "simulator-r-environment"

    prepared = {registered.binding.dataset_id: registered.prepared}
    store = final_runner.FinalResultStore(
        round_dir / "results/trajectory/execution",
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )
    for entry in plan.entries:
        store.append(
            entry,
            _unavailable_attempt(entry, prepared=registered.prepared),
            execution_request=_final_comparator_request(
                entry,
                fixture["registry"],
                registered.prepared,
                authority,
            ),
        )
    store.finalize()
    validation = final_runner._validate_frozen_execution_for_evaluation(
        plan,
        store.load_records(),
    )
    result_files = final_runner._owned_final_result_file_manifest(round_dir)[
        "result_files"
    ]
    assert isinstance(result_files, list)
    trajectory_evidence = final_runner._trajectory_evaluation_evidence(
        round_dir,
        plan,
        registered,
        authority,
        store,
        validation,
        result_files,
    )

    original_loader = final_runner.load_prepared_final_panel
    runtime_loads: list[dict[str, object]] = []

    def load_running_panel(selected: Path, destination: Path, **kwargs):
        runtime_loads.append(dict(kwargs))
        assert kwargs == {
            "allow_evaluated": True,
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
        }
        assert kwargs["simulator_assets_root"] is simulator_assets_root
        assert kwargs["simulator_r_environment"] is simulator_r_environment
        return original_loader(selected, destination, **kwargs)

    monkeypatch.setattr(
        final_runner,
        "load_prepared_final_panel",
        load_running_panel,
    )
    checkpoint = SimpleNamespace(
        status="completed",
        planned_run_count=1,
        records=(object(),),
    )
    monkeypatch.setattr(
        final_runner,
        "_run_pre_receipt_supplementary_phases",
        lambda selected, destination: (
            {"scaling": checkpoint}
            if (selected, destination) == (repository, round_dir)
            else {}
        ),
    )
    scaling_evidence = {"evidence_sha256": "a" * 64}
    monkeypatch.setattr(
        final_runner,
        "_scaling_evaluation_evidence",
        lambda selected, destination, observed, _files: (
            scaling_evidence
            if (selected, destination, observed) == (repository, round_dir, checkpoint)
            else {}
        ),
    )
    recorded: dict[str, object] = {}

    def record_receipt(destination: Path, manifest, *, repo: Path):
        assert (repo, destination) == (repository, round_dir)
        recorded.update(manifest)
        return {"state": "evaluated"}

    monkeypatch.setattr(study, "record_final_evaluation", record_receipt)

    result = final_runner._record_final_evaluation_after_scaling(
        repository,
        round_dir,
        {
            "schema_version": 1,
            "status": "completed",
            "final_plan_sha256": primary_plan.plan_sha256,
            "trajectory_evidence": trajectory_evidence,
        },
        simulator_assets_root=simulator_assets_root,
        simulator_r_environment=simulator_r_environment,
    )

    assert result == {"state": "evaluated"}
    assert recorded["trajectory_evidence"] == trajectory_evidence
    assert recorded["scaling_evidence"] == scaling_evidence
    assert runtime_loads == [
        {
            "allow_evaluated": True,
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
        },
        {
            "allow_evaluated": True,
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
        },
    ]


def test_combined_preflight_binds_primary_trajectory_scaling_with_one_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.scaling as scaling
    from maskimpute_benchmark.final_runner import (
        _FINAL_STORAGE_RESERVE_BYTES,
        _validate_combined_storage_capacity,
        build_final_execution_plan,
    )

    _repository, round_dir, _registered, _authority, trajectory = (
        _trajectory_store_inputs(tmp_path)
    )
    primary = build_final_execution_plan(
        _receipt(_registry()),
        _registry(),
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    scaling_receipt = {
        "schema": "maskimpute-scaling-storage-preflight-v1",
        "required_free_bytes": 123_456,
        "receipt_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        scaling,
        "scaling_storage_preflight",
        lambda _authority: scaling_receipt,
    )
    observed_free = 10**15
    monkeypatch.setattr(
        "maskimpute_benchmark.final_runner.shutil.disk_usage",
        lambda _path: type("Usage", (), {"free": observed_free})(),
    )

    receipt = _validate_combined_storage_capacity(
        primary,
        trajectory,
        object(),
        round_dir,
    )

    assert receipt["scaling"] == scaling_receipt
    assert receipt["reserve_bytes"] == _FINAL_STORAGE_RESERVE_BYTES
    assert receipt["required_free_bytes"] == (
        receipt["primary"]["required_free_bytes"]
        + receipt["trajectory"]["required_free_bytes"]
        + scaling_receipt["required_free_bytes"]
        + _FINAL_STORAGE_RESERVE_BYTES
    )
    assert receipt["observed_free_bytes"] == observed_free


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
    ).legacy_configuration
    assert configuration is not None
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

    source = load_method_registry(METHODS)
    registry = MethodRegistry(
        schema_version=1,
        methods=(source.by_id("d3impute"),),
    )
    receipt = _receipt(registry)
    row = next(
        value for value in receipt["method_denominator"] if value["id"] == "d3impute"
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


def _unavailable_prepared(plan_entry, *, count_model_ready: bool = False):
    import anndata as ad
    import numpy as np

    from maskimpute_benchmark.methods.base import MethodInput
    from maskimpute_benchmark.runner import (
        DatasetQCAudit,
        PreparedDataset,
    )

    counts = (
        np.array(
            [[0.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]],
            dtype="<f8",
        )
        if count_model_ready
        else np.array([[1.0, 0.0], [0.0, 2.0]], dtype="<f8")
    )
    var_ids = tuple(f"gene-{index}" for index in range(1, counts.shape[1] + 1))
    method_input = MethodInput(
        source_dataset_sha256=plan_entry.run.source_dataset_sha256,
        obs_ids=("cell-1", "cell-2"),
        var_ids=var_ids,
        shape=counts.shape,
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
    evaluator_dataset = ad.AnnData(X=counts.copy())
    evaluator_dataset.obs_names = list(method_input.obs_ids)
    evaluator_dataset.var_names = list(method_input.var_ids)
    evaluator_dataset.layers["truth"] = counts.copy()
    evaluator_dataset.uns["truth_kind"] = "exact_pre_capture"
    evaluator_dataset.uns["primary_truth_layer"] = "truth"
    prepared = PreparedDataset(
        binding=binding,
        audit=audit,
        method_input=method_input,
        evaluator_dataset=evaluator_dataset,
    )
    return prepared


def _unavailable_attempt(plan_entry, *, prepared=None):
    from maskimpute_benchmark.runner import AdapterOutcome, evaluate_adapter_outcome

    return evaluate_adapter_outcome(
        plan_entry.run,
        _unavailable_prepared(plan_entry) if prepared is None else prepared,
        AdapterOutcome.unavailable(plan_entry.reason or "test_unavailable"),
    )


def _completed_attempt(
    plan_entry,
    *,
    probability=None,
    score_diagnostics: dict[str, object] | None = None,
    prepared=None,
    output_converter=None,
):
    from dataclasses import replace

    import anndata as ad
    import numpy as np

    from maskimpute import ImputationResult
    from maskimpute.ablations import AblationRunResult
    from maskimpute_benchmark.methods import run_observed, snapshot_method_output
    from maskimpute_benchmark.methods.maskimpute import MaskImputeAdapterExecution
    from maskimpute_benchmark.runner import AdapterOutcome, evaluate_adapter_outcome

    prepared = _unavailable_prepared(plan_entry) if prepared is None else prepared
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
        from maskimpute.count_model import _counts_sha256

        if probability is None:
            probability = np.where(prepared.method_input.counts == 0, 0.5, 0.0)
        if score_diagnostics is None:
            score_diagnostics = {
                "source": "retained_calibrator",
                "score_artifact_sha256": "a" * 64,
                "score_input_sha256": _counts_sha256(prepared.method_input.counts),
                "score_config_sha256": _authority().count_model_config_sha256,
                "calibration_file_sha256": "b" * 64,
                "calibration_payload_sha256": "c" * 64,
                "retained_calibrator": "identity",
                "calibration_scope": ("retained_all_development_for_final_inference"),
                "equivalence_reason": (
                    "retained_identity_calibrator_equals_direct_score"
                ),
            }
        result = ImputationResult(
            selective_counts=prepared.method_input.counts,
            denoised_counts=prepared.method_input.counts,
            p_pre_zero=probability,
            latent=np.ones((prepared.method_input.shape[0], 1)),
            diagnostics={"score": score_diagnostics},
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
    arguments = {}
    if output_converter is not None:
        arguments["output_converter"] = output_converter
    return evaluate_adapter_outcome(
        plan_entry.run,
        prepared,
        outcome,
        **arguments,
    )


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


def _final_comparator_request(entry, registry, prepared, authority):
    from maskimpute_benchmark.runner import FinalComparatorExecutionRequest

    configuration = entry.run.comparator_configuration
    if entry.action != "execute" or configuration is None:
        return None
    spec = registry.by_id(entry.run.method_id)
    return FinalComparatorExecutionRequest.create(
        spec,
        prepared.method_input,
        model_seed=entry.run.model_seed,
        configuration=configuration,
        authority=authority,
        mechanism=entry.run.mechanism,
        biological_id=entry.run.biological_id,
        technical_view=entry.run.technical_view,
        dataset_id=entry.run.dataset_id,
        timeout_seconds=spec.resources.timeout_seconds,
    )


def _write_real_final_score_authority(repository: Path, prepared):
    from maskimpute import PreZeroCountModelConfig
    from maskimpute.calibration import (
        DEVELOPMENT_PROTOCOL_SHA256,
        CalibrationRecord,
        fit_development_calibration,
        save_calibration_artifact,
    )
    from maskimpute_benchmark.runner import ExecutionAuthorityContext

    ablation_path = repository / "study/ablations.json"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    ablation_path.write_bytes(Path("study/ablations.json").read_bytes())
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
    calibration = fit_development_calibration(records)
    calibration_path = repository / "authority/calibration.json"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_artifact(calibration_path, calibration)
    count_config = PreZeroCountModelConfig(n_folds=2, link_max_iter=25)
    count_payload = asdict(count_config)
    score_body = {
        "schema_version": 1,
        "artifact_type": "maskimpute_final_count_score_authority",
        "status": "ready",
        "scope": "truth_free_final_inference",
        "frozen_method_sha256": "1" * 64,
        "execution_claim_sha256": "2" * 64,
        "execution_environment_sha256": "3" * 64,
        "dataset_manifest_sha256": prepared.binding.manifest_sha256,
        "selection_contract_file_sha256": "4" * 64,
        "count_model_config": count_payload,
        "count_model_config_sha256": canonical_sha256(count_payload),
    }
    score_payload = {
        **score_body,
        "payload_sha256": canonical_sha256(score_body),
    }
    score_path = repository / "authority/count_score_authority.json"
    score_path.write_text(
        json.dumps(
            score_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    base = {"latent_dim": 2}
    context = ExecutionAuthorityContext(
        authority_sha256="9" * 64,
        base_configuration_json=json.dumps(base, separators=(",", ":"), sort_keys=True),
        base_configuration_sha256=canonical_sha256(base),
        count_model_config_json=json.dumps(
            count_payload, separators=(",", ":"), sort_keys=True
        ),
        count_model_config_sha256=canonical_sha256(count_payload),
        count_score_manifest_path="authority/count_score_authority.json",
        count_score_manifest_sha256=hashlib.sha256(score_path.read_bytes()).hexdigest(),
        retained_calibration_path="authority/calibration.json",
        retained_calibration_sha256=hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest(),
    )
    return context, calibration


def _diagnostics_from_policy(policy: dict[str, object]) -> dict[str, object]:
    return {
        "source": policy["score_source"],
        "score_artifact_sha256": policy["score_artifact_sha256"],
        "score_input_sha256": policy["score_input_sha256"],
        "score_config_sha256": policy["score_config_sha256"],
        "calibration_file_sha256": policy["calibration_file_sha256"],
        "calibration_payload_sha256": policy["calibration_payload_sha256"],
        "retained_calibrator": policy["calibration_algorithm"],
        "calibration_scope": policy["calibration_scope"],
        "equivalence_reason": policy["calibration_equivalence_reason"],
    }


def _real_final_execution_inputs(repository: Path, entry, plan, registry):
    from maskimpute_benchmark.runner import (
        ExecutionRequest,
        _derive_prezero_execution_authority,
    )

    prepared = _unavailable_prepared(entry, count_model_ready=True)
    context, _calibration = _write_real_final_score_authority(repository, prepared)
    probability, policy = _derive_prezero_execution_authority(
        prepared,
        entry.run,
        context,
        calibration_usage="retained_all_development",
        repository=repository,
    )
    attempt = _completed_attempt(
        entry,
        probability=probability,
        score_diagnostics=_diagnostics_from_policy(policy),
        prepared=prepared,
    )
    configuration = next(
        value for value in plan.configurations if value.method_id == "maskimpute"
    ).legacy_configuration
    assert configuration is not None
    request = ExecutionRequest.create(
        registry.by_id("maskimpute"),
        prepared.method_input,
        model_seed=entry.run.model_seed,
        configuration=configuration,
        authority=context,
        mechanism=entry.run.mechanism,
        biological_id=entry.run.biological_id,
        technical_view=entry.run.technical_view,
        dataset_id=entry.run.dataset_id,
        timeout_seconds=5,
        calibration_usage="retained_all_development",
    )
    return prepared, context, attempt, request


def _prepared_for_plan(plan):
    prepared = {}
    for entry in plan.entries:
        prepared.setdefault(
            entry.run.dataset_id,
            _unavailable_prepared(entry),
        )
    return prepared


def _final_store(
    output: Path,
    plan,
    *,
    prepared=None,
    authority=None,
    authority_repository: Path | None = None,
):
    from maskimpute_benchmark.final_runner import FinalResultStore

    return FinalResultStore(
        output,
        plan,
        _prepared_for_plan(plan) if prepared is None else prepared,
        _authority() if authority is None else authority,
        authority_repository=authority_repository,
    )


def test_final_result_store_is_immutable_resumable_and_manifest_complete(
    tmp_path: Path,
) -> None:
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
    store = _final_store(tmp_path / "execution", plan)

    first = store.append(plan.entries[0], _unavailable_attempt(plan.entries[0]))
    resumed = _final_store(tmp_path / "execution", plan)

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
    tiny = _final_store(tiny_root, tiny_plan)
    tiny.append(tiny_plan.entries[0], _unavailable_attempt(tiny_plan.entries[0]))
    manifest = tiny.finalize()

    assert manifest["status"] == "completed"
    assert manifest["planned_run_count"] == 1
    assert tiny.load_manifest() == manifest


def test_final_result_store_rejects_tampered_record_or_log(tmp_path: Path) -> None:
    from maskimpute_benchmark.final_runner import (
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
    store = _final_store(tmp_path / "execution", plan)
    record = store.append(plan.entries[0], _unavailable_attempt(plan.entries[0]))
    stdout = tmp_path / "execution" / record["run"]["stdout_path"]
    stdout.write_bytes(b"tampered")

    with pytest.raises(FinalRunnerContractError, match="record|artifact"):
        _final_store(tmp_path / "execution", plan).load_records()


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
        _final_store(output, plan).load_records()


def test_final_result_store_binds_successful_final_calibration_request(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
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
    entry = next(value for value in full.entries if value.run.method_id == "maskimpute")
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0], entry),
        configurations=full.configurations,
        plan_sha256="c" * 64,
    )
    prepared, authority, attempt, request = _real_final_execution_inputs(
        tmp_path,
        entry,
        plan,
        registry,
    )
    prepared_datasets = _prepared_for_plan(plan)
    prepared_datasets[entry.run.dataset_id] = prepared
    store = _final_store(
        tmp_path / "execution",
        plan,
        prepared=prepared_datasets,
        authority=authority,
        authority_repository=tmp_path,
    )
    store.append(
        full.entries[0],
        _unavailable_attempt(full.entries[0], prepared=prepared),
    )

    record = store.append(
        entry,
        attempt,
        execution_request=request,
    )

    assert record["execution_request"] == {
        "calibration_usage": "retained_all_development",
        "configuration_sha256": entry.run.configuration_sha256,
        "count_score_manifest_sha256": authority.count_score_manifest_sha256,
        "dataset_id": entry.run.dataset_id,
        "execution_authority_sha256": "9" * 64,
        "method_input_sha256": request.method_input_sha256,
        "model_seed": entry.run.model_seed,
        "request_sha256": request.request_sha256,
        "retained_calibration_sha256": authority.retained_calibration_sha256,
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
    assert (
        _final_store(
            tmp_path / "execution",
            plan,
            prepared=prepared_datasets,
            authority=authority,
            authority_repository=tmp_path,
        ).load_records()[-1]
        == record
    )

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
        _final_store(
            tmp_path / "execution",
            plan,
            prepared=prepared_datasets,
            authority=authority,
            authority_repository=tmp_path,
        ).load_records()
    score_path.write_bytes(score_compressed)
    record_path.write_text(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    compressed_path = tmp_path / "execution" / str(run["evaluator_output_path"])
    compressed_path.write_bytes(compressed + b"tamper")
    with pytest.raises(Exception, match="record|artifact|compressed"):
        _final_store(
            tmp_path / "execution",
            plan,
            prepared=prepared_datasets,
            authority=authority,
            authority_repository=tmp_path,
        ).load_records()


def test_final_result_store_rejects_rehashed_prezero_metric_drift(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
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
    source_entry = next(
        value for value in full.entries if value.run.method_id == "maskimpute"
    )
    entry = replace(source_entry, run=replace(source_entry.run, ordinal=1))
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(entry,),
        configurations=full.configurations,
        plan_sha256="d" * 64,
    )
    prepared, authority, attempt, request = _real_final_execution_inputs(
        tmp_path,
        entry,
        plan,
        registry,
    )
    output = tmp_path / "execution"
    store = _final_store(
        output,
        plan,
        prepared={prepared.binding.dataset_id: prepared},
        authority=authority,
        authority_repository=tmp_path,
    )
    store.append(entry, attempt, execution_request=request)
    record_path = output / "records/00000001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    evidence = record["p_pre_zero_evidence"]
    evidence["overall"]["metrics"]["brier"]["value"] = 0.123456
    body = {
        key: value
        for key, value in evidence.items()
        if key not in {"evidence_sha256", "storage"}
    }
    evidence["evidence_sha256"] = canonical_sha256(body)
    record_path.write_text(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FinalRunnerContractError, match="report differs"):
        _final_store(
            output,
            plan,
            prepared={prepared.binding.dataset_id: prepared},
            authority=authority,
            authority_repository=tmp_path,
        ).load_records()


def test_final_result_store_refits_once_and_rejects_coordinated_score_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    import zlib

    import maskimpute
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )
    from maskimpute_benchmark.prezero_evidence import _score_report
    from maskimpute_benchmark.runner import (
        ExecutionRequest,
        _derive_prezero_execution_authority,
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
    source_entry = next(
        value for value in full.entries if value.run.method_id == "maskimpute"
    )
    entry = replace(source_entry, run=replace(source_entry.run, ordinal=1))
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(entry,),
        configurations=full.configurations,
        plan_sha256="d" * 64,
    )
    prepared = _unavailable_prepared(entry, count_model_ready=True)
    context, _calibration = _write_real_final_score_authority(tmp_path, prepared)
    probability, policy = _derive_prezero_execution_authority(
        prepared,
        entry.run,
        context,
        calibration_usage="retained_all_development",
        repository=tmp_path,
    )
    attempt = _completed_attempt(
        entry,
        probability=probability,
        score_diagnostics=_diagnostics_from_policy(policy),
        prepared=prepared,
    )
    configuration = next(
        value for value in plan.configurations if value.method_id == "maskimpute"
    ).legacy_configuration
    assert configuration is not None
    request = ExecutionRequest.create(
        registry.by_id("maskimpute"),
        prepared.method_input,
        model_seed=entry.run.model_seed,
        configuration=configuration,
        authority=context,
        mechanism=entry.run.mechanism,
        biological_id=entry.run.biological_id,
        technical_view=entry.run.technical_view,
        dataset_id=entry.run.dataset_id,
        timeout_seconds=5,
        calibration_usage="retained_all_development",
    )
    refit_calls = 0
    original_fit = maskimpute.fit_p_pre_zero_count_model

    def counted_fit(*args, **kwargs):
        nonlocal refit_calls
        refit_calls += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(maskimpute, "fit_p_pre_zero_count_model", counted_fit)
    output = tmp_path / "execution"
    store = _final_store(
        output,
        plan,
        prepared={prepared.binding.dataset_id: prepared},
        authority=context,
        authority_repository=tmp_path,
    )
    record = store.append(entry, attempt, execution_request=request)
    assert refit_calls == 1
    assert store.load_records() == (record,)
    assert refit_calls == 1
    second_seed = replace(
        entry.run,
        run_id=f"{entry.run.run_id}-seed-43",
        model_seed=43,
    )
    cached_probability, cached_policy = store._artifacts._expected_prezero_authority(
        second_seed,
        prepared,
        context,
        calibration_usage="retained_all_development",
        expected_matrix_present=True,
    )
    np.testing.assert_array_equal(cached_probability, probability)
    assert cached_policy == policy
    assert refit_calls == 1

    evidence = record["p_pre_zero_evidence"]
    replacement = np.where(
        prepared.method_input.counts == 0.0,
        0.125,
        0.0,
    ).astype("<f8")
    raw = replacement.tobytes(order="C")
    compressed = zlib.compress(raw, level=6)
    score_path = output / evidence["storage"]["path"]
    score_path.write_bytes(compressed)
    evidence["storage"].update(
        {
            "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
            "compressed_nbytes": len(compressed),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "uncompressed_nbytes": len(raw),
        }
    )
    evidence["matrix"]["content_sha256"] = hashlib.sha256(raw).hexdigest()
    evidence["overall"], evidence["strata"] = _score_report(
        replacement,
        np.asarray(prepared.evaluator_dataset.X, dtype=np.float64),
        np.asarray(prepared.evaluator_dataset.layers["truth"], dtype=np.float64),
        truth_kind="exact_pre_capture",
    )
    semantic = hashlib.sha256()
    semantic.update(b"maskimpute-realized-p-pre-zero-v1\0")
    semantic.update(
        json.dumps(
            {
                "identity": evidence["identity"],
                "shape": evidence["matrix"]["shape"],
                "dtype": evidence["matrix"]["dtype"],
                "policy_sha256": evidence["policy_sha256"],
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    semantic.update(raw)
    evidence["matrix"]["semantic_sha256"] = semantic.hexdigest()
    evidence_body = {
        name: value
        for name, value in evidence.items()
        if name not in {"evidence_sha256", "storage"}
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence_body)
    record_path = output / "records/00000001.json"
    record_path.write_text(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FinalRunnerContractError, match="matrix differs"):
        store.load_records()
    assert refit_calls == 1


def test_final_append_validates_before_publishing_and_allows_retry(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
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
    source_entry = next(
        value for value in full.entries if value.run.method_id == "maskimpute"
    )
    entry = replace(source_entry, run=replace(source_entry.run, ordinal=1))
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(entry,),
        configurations=full.configurations,
        plan_sha256="d" * 64,
    )
    prepared, context, valid_attempt, request = _real_final_execution_inputs(
        tmp_path,
        entry,
        plan,
        registry,
    )
    policy = valid_attempt.p_pre_zero_evidence.to_record()["policy"]
    assert isinstance(policy, dict)
    invalid_probability = np.where(
        prepared.method_input.counts == 0.0,
        0.125,
        0.0,
    ).astype("<f8")
    invalid_attempt = _completed_attempt(
        entry,
        probability=invalid_probability,
        score_diagnostics=_diagnostics_from_policy(policy),
        prepared=prepared,
    )
    output = tmp_path / "execution"
    store = _final_store(
        output,
        plan,
        prepared={prepared.binding.dataset_id: prepared},
        authority=context,
        authority_repository=tmp_path,
    )

    with pytest.raises(FinalRunnerContractError, match="publish final"):
        store.append(entry, invalid_attempt, execution_request=request)
    assert not output.exists()

    record = store.append(entry, valid_attempt, execution_request=request)
    assert record["run"]["status"] == "completed"
    assert store.load_records() == (record,)


@pytest.mark.parametrize("tamper", ("matrix", "policy", "report"))
def test_final_conversion_terminal_score_remains_exactly_authorized(
    tmp_path: Path,
    tamper: str,
) -> None:
    from dataclasses import replace
    import zlib

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )
    from maskimpute_benchmark.prezero_evidence import _score_report

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    source_entry = next(
        value for value in full.entries if value.run.method_id == "maskimpute"
    )
    entry = replace(source_entry, run=replace(source_entry.run, ordinal=1))
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(entry,),
        configurations=full.configurations,
        plan_sha256="d" * 64,
    )
    prepared, context, valid_attempt, request = _real_final_execution_inputs(
        tmp_path,
        entry,
        plan,
        registry,
    )
    policy = valid_attempt.p_pre_zero_evidence.to_record()["policy"]
    assert isinstance(policy, dict)

    def reject_conversion(_method_input, _execution):
        raise ValueError("deliberate evaluator conversion rejection")

    conversion_attempt = _completed_attempt(
        entry,
        probability=valid_attempt.p_pre_zero_evidence.matrix,
        score_diagnostics=_diagnostics_from_policy(policy),
        prepared=prepared,
        output_converter=reject_conversion,
    )
    assert conversion_attempt.run.status == "unavailable"
    output = tmp_path / "execution"
    store = _final_store(
        output,
        plan,
        prepared={prepared.binding.dataset_id: prepared},
        authority=context,
        authority_repository=tmp_path,
    )
    record = store.append(entry, conversion_attempt, execution_request=request)
    assert record["run"]["status"] == "unavailable"
    assert record["p_pre_zero_evidence"]["status"] == "unavailable"

    evidence = record["p_pre_zero_evidence"]
    score_path = output / evidence["storage"]["path"]
    raw = zlib.decompress(score_path.read_bytes())
    if tamper == "matrix":
        replacement = np.where(
            prepared.method_input.counts == 0.0,
            0.125,
            0.0,
        ).astype("<f8")
        raw = replacement.tobytes(order="C")
        compressed = zlib.compress(raw, level=6)
        score_path.write_bytes(compressed)
        evidence["storage"].update(
            {
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "compressed_nbytes": len(compressed),
                "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                "uncompressed_nbytes": len(raw),
            }
        )
        evidence["matrix"]["content_sha256"] = hashlib.sha256(raw).hexdigest()
        evidence["overall"], evidence["strata"] = _score_report(
            replacement,
            np.asarray(prepared.evaluator_dataset.X, dtype=np.float64),
            np.asarray(prepared.evaluator_dataset.layers["truth"], dtype=np.float64),
            truth_kind="exact_pre_capture",
            unavailable_status="unavailable",
            unavailable_reason=evidence["reason"],
        )
    elif tamper == "policy":
        evidence["policy"]["calibration_scope"] = "forged_scope"
        evidence["policy_sha256"] = canonical_sha256(evidence["policy"])
    elif tamper == "report":
        evidence["overall"]["metrics"]["brier"]["reason"] = "forged_reason"
    else:  # pragma: no cover - parametrization is fixed above
        raise AssertionError(tamper)
    semantic = hashlib.sha256()
    semantic.update(b"maskimpute-realized-p-pre-zero-v1\0")
    semantic.update(
        json.dumps(
            {
                "identity": evidence["identity"],
                "shape": evidence["matrix"]["shape"],
                "dtype": evidence["matrix"]["dtype"],
                "policy_sha256": evidence["policy_sha256"],
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    semantic.update(raw)
    evidence["matrix"]["semantic_sha256"] = semantic.hexdigest()
    evidence_body = {
        name: value
        for name, value in evidence.items()
        if name not in {"evidence_sha256", "storage"}
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence_body)
    record_path = output / "records/00000001.json"
    record_path.write_text(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FinalRunnerContractError,
        match="p_pre_zero|matrix|policy|report",
    ):
        store.load_records()


def test_final_conversion_terminal_score_cannot_be_coordinately_removed(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )
    from maskimpute_benchmark.prezero_evidence import _score_report

    registry = _registry()
    full = build_final_execution_plan(
        _receipt(registry),
        registry,
        _bindings(),
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        execution_authority_sha256="9" * 64,
    )
    source_entry = next(
        value for value in full.entries if value.run.method_id == "maskimpute"
    )
    entry = replace(source_entry, run=replace(source_entry.run, ordinal=1))
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(entry,),
        configurations=full.configurations,
        plan_sha256="d" * 64,
    )
    prepared, context, valid_attempt, request = _real_final_execution_inputs(
        tmp_path,
        entry,
        plan,
        registry,
    )
    policy = valid_attempt.p_pre_zero_evidence.to_record()["policy"]
    assert isinstance(policy, dict)

    def reject_conversion(_method_input, _execution):
        raise ValueError("deliberate evaluator conversion rejection")

    conversion_attempt = _completed_attempt(
        entry,
        probability=valid_attempt.p_pre_zero_evidence.matrix,
        score_diagnostics=_diagnostics_from_policy(policy),
        prepared=prepared,
        output_converter=reject_conversion,
    )
    output = tmp_path / "execution"
    store = _final_store(
        output,
        plan,
        prepared={prepared.binding.dataset_id: prepared},
        authority=context,
        authority_repository=tmp_path,
    )
    record = store.append(entry, conversion_attempt, execution_request=request)
    evidence = record["p_pre_zero_evidence"]
    (output / evidence["storage"]["path"]).unlink()
    evidence["matrix"] = {
        "shape": None,
        "dtype": None,
        "content_sha256": None,
        "semantic_sha256": None,
    }
    evidence["policy"] = None
    evidence["policy_sha256"] = None
    evidence["storage"] = {
        "encoding": None,
        "compression_level": None,
        "path": None,
        "compressed_sha256": None,
        "compressed_nbytes": None,
        "uncompressed_sha256": None,
        "uncompressed_nbytes": None,
    }
    evidence["overall"], evidence["strata"] = _score_report(
        None,
        np.asarray(prepared.evaluator_dataset.X, dtype=np.float64),
        np.asarray(prepared.evaluator_dataset.layers["truth"], dtype=np.float64),
        truth_kind="exact_pre_capture",
        unavailable_status="unavailable",
        unavailable_reason=evidence["reason"],
    )
    body = {
        name: value
        for name, value in evidence.items()
        if name not in {"evidence_sha256", "storage"}
    }
    evidence["evidence_sha256"] = canonical_sha256(body)
    record_path = output / "records/00000001.json"
    record_path.write_text(
        json.dumps(
            record,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        FinalRunnerContractError,
        match="p_pre_zero|matrix|authority",
    ):
        _final_store(
            output,
            plan,
            prepared={prepared.binding.dataset_id: prepared},
            authority=context,
            authority_repository=tmp_path,
        ).load_records()


def test_final_result_store_append_does_not_rehash_the_whole_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.final_runner import (
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
    store = _final_store(tmp_path / "execution", plan)
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

    store = _final_store(
        tmp_path / "execution",
        plan,
        prepared=prepared,
        authority=authority,
    )
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
        _final_store(
            tmp_path / "execution",
            plan,
            prepared=prepared,
            authority=authority,
        ),
        on_record_published=lambda: publications.append("unexpected"),
    )
    assert resumed == manifest
    assert len(requests) == 1


def test_execute_final_plan_leaves_infrastructure_failure_retryable(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
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
            _final_store(root, plan, prepared=prepared),
            on_record_published=lambda: None,
        )
    assert _final_store(root, plan, prepared=prepared).load_records() == ()

    manifest = execute_final_plan(
        plan,
        registry,
        prepared,
        _authority(),
        lambda _request: AdapterOutcome.unavailable("algorithm_unavailable"),
        _final_store(root, plan, prepared=prepared),
        on_record_published=lambda: None,
    )
    assert manifest["recorded_run_count"] == 1


def test_execute_final_plan_journals_once_after_the_complete_manifest(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
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
        _final_store(tmp_path / "execution", plan, prepared=prepared),
        on_record_published=lambda: publications.append("published"),
    )

    assert manifest["status"] == "completed"
    assert publications == ["published"]


def test_primary_execution_manifest_callback_crash_reconciles_and_resumes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        _record_incremental_results_if_changed,
        build_final_execution_plan,
        execute_final_plan,
    )
    from maskimpute_benchmark.runner import AdapterOutcome
    from maskimpute_benchmark.simulators import (
        SimulationContractError,
        load_final_manifest_claim,
    )
    from maskimpute_benchmark.study import record_incremental_results

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
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
    prepared = _prepared_for_plan(plan)
    output = round_dir / "results/final/execution"

    def crash_after_manifest() -> None:
        assert (output / "execution_manifest.json").is_file()
        raise RuntimeError("crash after primary execution manifest")

    with pytest.raises(RuntimeError, match="primary execution manifest"):
        execute_final_plan(
            plan,
            registry,
            prepared,
            _authority(),
            lambda _request: AdapterOutcome.unavailable("algorithm_unavailable"),
            _final_store(output, plan, prepared=prepared),
            on_record_published=crash_after_manifest,
        )
    with pytest.raises(SimulationContractError, match="clean frozen|valid claimed"):
        load_final_manifest_claim(repository, round_dir)

    resumed_store = _final_store(output, plan, prepared=prepared)
    assert resumed_store.load_manifest()["status"] == "completed"
    _record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"
    assert (
        execute_final_plan(
            plan,
            registry,
            prepared,
            _authority(),
            lambda _request: pytest.fail("completed plan must not execute again"),
            resumed_store,
            on_record_published=lambda: pytest.fail(
                "reconciled manifest must not be republished"
            ),
        )["status"]
        == "completed"
    )


def test_trajectory_execution_manifest_callback_crash_reconciles_and_resumes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        _record_incremental_results_if_changed,
        execute_trajectory_plan,
    )
    from maskimpute_benchmark.runner import AdapterOutcome
    from maskimpute_benchmark.simulators import (
        SimulationContractError,
        load_final_manifest_claim,
    )
    from maskimpute_benchmark.study import record_incremental_results

    repository, round_dir = _claimed_lifecycle_round(tmp_path / "lifecycle")
    (
        authority_repository,
        _trajectory_round,
        registered,
        authority,
        plan,
    ) = _trajectory_store_inputs(tmp_path / "trajectory")
    prepared = {registered.binding.dataset_id: registered.prepared}
    output = round_dir / "results/trajectory/execution"

    def store() -> FinalResultStore:
        return FinalResultStore(
            output,
            plan,
            prepared,
            authority,
            authority_repository=authority_repository,
        )

    def crash_after_manifest() -> None:
        assert (output / "execution_manifest.json").is_file()
        raise RuntimeError("crash after trajectory execution manifest")

    with pytest.raises(RuntimeError, match="trajectory execution manifest"):
        execute_trajectory_plan(
            plan,
            _registry(),
            prepared,
            authority,
            lambda _request: AdapterOutcome.unavailable("algorithm_unavailable"),
            store(),
            on_record_published=crash_after_manifest,
        )
    with pytest.raises(SimulationContractError, match="clean frozen|valid claimed"):
        load_final_manifest_claim(repository, round_dir)

    resumed_store = store()
    assert resumed_store.load_manifest()["status"] == "completed"
    _record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"
    assert (
        execute_trajectory_plan(
            plan,
            _registry(),
            prepared,
            authority,
            lambda _request: pytest.fail("completed plan must not execute again"),
            resumed_store,
            on_record_published=lambda: pytest.fail(
                "reconciled manifest must not be republished"
            ),
        )["status"]
        == "completed"
    )


def test_final_evaluation_retains_reason_coded_algorithmic_unavailability(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        _validate_frozen_execution_for_evaluation,
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
        plan_sha256="f" * 64,
    )
    store = _final_store(tmp_path / "execution", plan)
    store.append(plan.entries[0], _unavailable_attempt(plan.entries[0]))

    validation = _validate_frozen_execution_for_evaluation(plan, store.load_records())

    assert validation["executed_completed_count"] == 0
    assert validation["executed_algorithmic_failure_count"] == 1
    assert validation["executed_status_counts"] == {"unavailable": 1}


@pytest.mark.parametrize("status", ["failed", "timeout", "resource_exceeded"])
def test_final_evaluation_retains_each_unfavorable_algorithmic_status(
    tmp_path: Path, status: str
) -> None:
    from maskimpute_benchmark.final_runner import (
        _validate_frozen_execution_for_evaluation,
        build_final_execution_plan,
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
    store = _final_store(tmp_path / status / "execution", plan)
    store.append(entry, attempt)

    validation = _validate_frozen_execution_for_evaluation(plan, store.load_records())

    assert validation["planned_run_count"] == 1
    assert validation["executed_algorithmic_failure_count"] == 1
    assert validation["executed_status_counts"] == {status: 1}


def test_final_evaluation_blocks_infrastructure_incompleteness(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _validate_frozen_execution_for_evaluation,
        build_final_execution_plan,
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
    store = _final_store(tmp_path / "execution", plan)
    store.append(entry, attempt)

    with pytest.raises(FinalRunnerContractError, match="infrastructure|authority"):
        _validate_frozen_execution_for_evaluation(plan, store.load_records())


def test_final_evaluation_accepts_completed_execution_and_exact_nonrun() -> None:
    from maskimpute_benchmark.final_runner import (
        _validate_frozen_execution_for_evaluation,
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

    validation = _validate_frozen_execution_for_evaluation(
        plan, (completed, unavailable)
    )

    assert validation["executed_completed_count"] == 1
    assert validation["not_applicable_count"] == 1


def test_final_evaluation_rejects_incomplete_completed_metric_evidence() -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _validate_frozen_execution_for_evaluation,
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
    entry = full.entries[0]
    attempt = _completed_attempt(entry)
    canonical_record = json.loads(
        json.dumps(
            {
                "run": asdict(attempt.run),
                "metrics": [metric.to_dict() for metric in attempt.metrics],
                "execution_request": None,
            }
        )
    )
    plan = full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(entry,),
        configurations=full.configurations,
        plan_sha256="0" * 64,
    )
    accepted: list[str] = []
    for mutation in (
        "names_only",
        "missing_status",
        "boolean_value",
        "boolean_denominator",
        "unavailable_without_reason",
        "completed_with_reason",
    ):
        record = json.loads(json.dumps(canonical_record))
        if mutation == "names_only":
            record["metrics"] = [
                {"metric": metric["metric"]} for metric in record["metrics"]
            ]
        else:
            metric = record["metrics"][0]
            if mutation == "missing_status":
                metric.pop("status")
            elif mutation == "boolean_value":
                metric.update(value=True, status="completed", reason=None)
            elif mutation == "boolean_denominator":
                metric["n"] = True
            elif mutation == "unavailable_without_reason":
                metric.update(value=None, status="unavailable", reason=None)
            else:
                metric.update(value=0.0, status="completed", reason="no_entries")
        try:
            _validate_frozen_execution_for_evaluation(plan, (record,))
        except FinalRunnerContractError:
            continue
        accepted.append(mutation)

    assert accepted == []


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
    simulator_assets_root = tmp_path / "external-simulator-assets"
    simulator_r_environment = tmp_path / "simulator-r-environment"

    def validate_status(
        path,
        *,
        repo,
        round_dir,
        simulator_assets_root,
        simulator_r_environment,
    ):
        validations.append(
            (
                path,
                repo,
                round_dir,
                simulator_assets_root,
                simulator_r_environment,
            )
        )
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
        repository,
        round_dir,
        simulator_assets_root=simulator_assets_root,
        simulator_r_environment=simulator_r_environment,
    )

    assert observed_bindings == bindings
    assert tuple(observed_prepared) == tuple(binding.dataset_id for binding in bindings)
    assert [item[1] for item in reads] == [binding.dataset_id for binding in bindings]
    assert len(pair_calls) == 1
    assert pair_calls[0][4] == final_runner.DatasetQCPolicy.fixed()
    assert validations == [
        (
            status_path,
            repository.resolve(),
            round_dir.resolve(),
            simulator_assets_root,
            simulator_r_environment,
        ),
        (
            status_path,
            repository.resolve(),
            round_dir.resolve(),
            simulator_assets_root,
            simulator_r_environment,
        ),
    ]


@pytest.mark.parametrize(
    ("allow_evaluated", "supply_assets", "supply_environment"),
    (
        (False, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
    ),
)
def test_load_prepared_final_panel_requires_complete_running_runtime_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_evaluated: bool,
    supply_assets: bool,
    supply_environment: bool,
) -> None:
    import maskimpute_benchmark.datasets as datasets
    import maskimpute_benchmark.final_runner as final_runner

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    round_dir.mkdir(parents=True)
    simulator_assets_root = (
        tmp_path / "external-simulator-assets" if supply_assets else None
    )
    simulator_r_environment = (
        tmp_path / "simulator-r-environment" if supply_environment else None
    )

    def forbid_validation(*_args, **_kwargs):
        raise AssertionError("incomplete runtime path pair reached status validation")

    monkeypatch.setattr(datasets, "validate_dataset_status", forbid_validation)

    with pytest.raises(final_runner.FinalRunnerContractError, match="runtime.*paths"):
        final_runner.load_prepared_final_panel(
            repository,
            round_dir,
            allow_evaluated=allow_evaluated,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )


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


def test_load_prepared_final_panel_supports_validated_evaluated_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.datasets as datasets
    import maskimpute_benchmark.final_runner as final_runner

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    round_dir.mkdir(parents=True)
    bindings = _bindings()
    prepared = {binding.dataset_id: object() for binding in bindings}
    snapshots: list[str] = []
    running_calls: list[dict[str, object]] = []
    runtime_lookups: list[object] = []

    def running_only(*_args, **kwargs):
        running_calls.append(kwargs)
        raise RuntimeError("running claim is unavailable after evaluation")

    def evaluated_snapshot(_repository, _round):
        snapshots.append("stable-evaluated-snapshot")
        return "stable-evaluated-snapshot"

    monkeypatch.setattr(datasets, "validate_dataset_status", running_only)
    monkeypatch.setattr(
        datasets,
        "load_simulator_runtime_assets",
        lambda *_args, **_kwargs: runtime_lookups.append(object()),
    )
    monkeypatch.setattr(
        final_runner,
        "_validated_evaluated_final_round_snapshot",
        evaluated_snapshot,
        raising=False,
    )
    monkeypatch.setattr(
        final_runner,
        "_load_unclaimed_prepared_final_panel",
        lambda _repository, _round: (bindings, prepared),
    )

    observed = final_runner.load_prepared_final_panel(
        repository,
        round_dir,
        allow_evaluated=True,
    )

    assert observed == (bindings, prepared)
    assert snapshots == [
        "stable-evaluated-snapshot",
        "stable-evaluated-snapshot",
    ]
    assert len(running_calls) == 1
    assert running_calls[0].get("simulator_assets_root") is None
    assert running_calls[0].get("simulator_r_environment") is None
    assert runtime_lookups == []


def test_evaluated_final_panel_rejects_journaled_status_for_another_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.datasets import (
        DatasetRegistryError,
        validate_evaluated_final_dataset_status,
    )
    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.study as study
    from maskimpute_benchmark.study import record_final_evaluation

    repository, round_dir = _claimed_lifecycle_round(tmp_path, seed_count=60)
    claim = json.loads((round_dir / "execution_claim.json").read_text(encoding="utf-8"))
    bindings = _bindings()
    rows = [
        {
            **asdict(binding),
            "status": "completed",
        }
        for binding in bindings
    ]
    body = {
        "schema_version": 1,
        "namespace": "final",
        "status": "completed",
        "protocol_sha256": claim["protocol_sha256"],
        "design_sha256": "3" * 64,
        "seed_source_sha256": claim["seed_manifest_sha256"],
        "execution_claim_id": "coherently-rehashed-other-claim",
        "round_id": round_dir.name,
        "independent_unit_count": 20,
        "completed_count": 40,
        "failed_count": 0,
        "runtime_assets_sha256": "5" * 64,
        "runtime_assets_receipt": {"forged": "but-journaled"},
        "rows": rows,
    }
    status = {**body, "manifest_sha256": canonical_sha256(body)}
    status_path = round_dir / "results/dataset_status.json"
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record_final_evaluation(
        round_dir,
        {
            "result_files": [
                {
                    "path": "results/dataset_status.json",
                    "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
                }
            ]
        },
        repo=repository,
    )
    prepared = {binding.dataset_id: object() for binding in bindings}
    monkeypatch.setattr(
        final_runner,
        "_prepare_final_panel_bindings",
        lambda _round, observed: prepared
        if tuple(item.dataset_id for item in observed) == tuple(prepared)
        else {},
    )
    freeze = study._validate_freeze(round_dir, repository)
    materialization, execution_claim, _receipt = study._validate_state_record_chain(
        round_dir,
        freeze,
        expected_state="evaluated",
    )
    repeated_materialization, seed_manifest = study._validate_seed_manifest(
        round_dir,
        freeze,
    )
    assert materialization == repeated_materialization
    assert execution_claim is not None

    with pytest.raises(DatasetRegistryError, match="claim or round"):
        validate_evaluated_final_dataset_status(
            status_path,
            repo=repository,
            round_dir=round_dir,
            protocol_path=repository / str(freeze["protocol_path"]),
            execution_claim=execution_claim,
            seed_manifest=seed_manifest,
        )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="claim|design|dataset status",
    ):
        final_runner.load_prepared_final_panel(
            repository,
            round_dir,
            allow_evaluated=True,
        )


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


def test_primary_authority_before_journal_reconciles_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.final_runner import (
        _record_incremental_results_if_changed,
        materialize_final_execution_authority,
    )
    from maskimpute_benchmark.simulators import (
        SimulationContractError,
        load_final_manifest_claim,
    )
    from maskimpute_benchmark.study import record_incremental_results

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    frozen = _final_authority_receipt(repository, monkeypatch)
    arguments = {
        "execution_claim_sha256": "7" * 64,
        "execution_environment_sha256": "8" * 64,
        "dataset_manifest_sha256": "1" * 64,
    }
    first = materialize_final_execution_authority(
        repository,
        round_dir,
        frozen,
        **arguments,
    )
    with pytest.raises(SimulationContractError, match="clean frozen|valid claimed"):
        load_final_manifest_claim(repository, round_dir)

    second = materialize_final_execution_authority(
        repository,
        round_dir,
        frozen,
        **arguments,
    )
    assert second == first
    _record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"


def test_primary_authority_recovery_rejects_unexpected_known_path_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.final_runner import (
        materialize_final_execution_authority,
    )
    from maskimpute_benchmark.runner import RunnerContractError

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    frozen = _final_authority_receipt(repository, monkeypatch)
    arguments = {
        "execution_claim_sha256": "7" * 64,
        "execution_environment_sha256": "8" * 64,
        "dataset_manifest_sha256": "1" * 64,
    }
    materialize_final_execution_authority(
        repository,
        round_dir,
        frozen,
        **arguments,
    )
    authority_path = round_dir / "results/final/execution_authority/authority.json"
    authority_path.write_bytes(b'{"unexpected":"known-path-bytes"}\n')

    with pytest.raises(
        RunnerContractError,
        match="immutable|differ|existing|refusing|replace",
    ):
        materialize_final_execution_authority(
            repository,
            round_dir,
            frozen,
            **arguments,
        )


def test_trajectory_authority_before_journal_reconciles_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.final_runner import (
        _record_incremental_results_if_changed,
        materialize_final_execution_authority,
        materialize_prepared_trajectory_dataset,
        materialize_trajectory_execution_authority,
    )
    from maskimpute_benchmark.simulators import (
        SimulationContractError,
        load_final_manifest_claim,
    )
    from maskimpute_benchmark.study import record_incremental_results

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    frozen = _final_authority_receipt(repository, monkeypatch)
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    primary = materialize_final_execution_authority(
        repository,
        round_dir,
        frozen,
        execution_claim_sha256="7" * 64,
        execution_environment_sha256="8" * 64,
        dataset_manifest_sha256="1" * 64,
    )
    _record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"

    arguments = {
        "execution_claim_sha256": "7" * 64,
        "execution_environment_sha256": "8" * 64,
        "primary_final_plan_sha256": "a" * 64,
    }
    first = materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen,
        registered,
        primary,
        **arguments,
    )
    with pytest.raises(SimulationContractError, match="clean frozen|valid claimed"):
        load_final_manifest_claim(repository, round_dir)

    second = materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen,
        registered,
        primary,
        **arguments,
    )
    assert second == first
    _record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"


def _exact_primary_trajectory_chain_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    full_denominator: bool = False,
) -> dict[str, object]:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.methods as methods
    import maskimpute_benchmark.publication_freeze as publication_freeze

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    registry = _registry() if full_denominator else _observed_only_registry()
    frozen = _final_authority_receipt(repository, monkeypatch, registry)
    claim = json.loads((round_dir / "execution_claim.json").read_text(encoding="utf-8"))
    claim_sha256 = canonical_sha256(claim)
    environment_sha256 = "8" * 64
    registered = final_runner.materialize_prepared_trajectory_dataset(
        repository,
        round_dir,
    )
    primary = final_runner.materialize_final_execution_authority(
        repository,
        round_dir,
        frozen,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environment_sha256,
        dataset_manifest_sha256="1" * 64,
    )
    bindings = _bindings()
    primary_plan = final_runner.build_final_execution_plan(
        frozen,
        registry,
        bindings,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environment_sha256,
        execution_authority_sha256=primary.authority_sha256,
    )
    prepared = _prepared_for_plan(primary_plan)
    primary_store = final_runner.FinalResultStore(
        round_dir / "results/final/execution",
        primary_plan,
        prepared,
        primary,
        authority_repository=repository,
    )
    for entry in primary_plan.entries:
        prepared_entry = prepared[entry.run.dataset_id]
        execution_request = _final_comparator_request(
            entry,
            registry,
            prepared_entry,
            primary,
        )
        primary_store.append(
            entry,
            _unavailable_attempt(
                entry,
                prepared=prepared[entry.run.dataset_id],
            ),
            execution_request=execution_request,
        )
    primary_manifest = primary_store.finalize()
    trajectory = final_runner.materialize_trajectory_execution_authority(
        repository,
        round_dir,
        frozen,
        registered,
        primary,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environment_sha256,
        primary_final_plan_sha256=primary_plan.plan_sha256,
    )
    trajectory_plan = final_runner.build_trajectory_execution_plan(
        frozen,
        registry,
        registered,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environment_sha256,
        execution_authority_sha256=trajectory.authority_sha256,
        primary_final_plan_sha256=primary_plan.plan_sha256,
    )

    environment = SimpleNamespace(
        registry_sha256=environment_sha256,
        runtime_lock_sha256=frozen["runtime_lock_sha256"],
        full_revalidate=lambda: None,
    )
    monkeypatch.setattr(
        publication_freeze,
        "validate_frozen_method",
        lambda selected: frozen if selected == repository else None,
    )
    monkeypatch.setattr(
        methods,
        "load_method_registry",
        lambda path: registry if path == repository / "study/methods.json" else None,
    )
    monkeypatch.setattr(
        final_runner,
        "load_prepared_final_panel",
        lambda selected, destination, **_kwargs: (
            (bindings, prepared)
            if (selected, destination) == (repository, round_dir)
            else ((), {})
        ),
    )
    monkeypatch.setattr(
        final_runner,
        "_load_final_execution_environment_registry",
        lambda selected, selected_registry: (
            environment
            if (selected, selected_registry) == (repository, registry)
            else None
        ),
        raising=False,
    )
    monkeypatch.setattr(
        final_runner,
        "_validate_final_runtime_lock",
        lambda selected_frozen, selected_environment: (
            str(frozen["runtime_lock_sha256"])
            if (selected_frozen, selected_environment) == (frozen, environment)
            else None
        ),
    )
    return {
        "repository": repository,
        "round_dir": round_dir,
        "registry": registry,
        "frozen": frozen,
        "claim_sha256": claim_sha256,
        "environment_sha256": environment_sha256,
        "bindings": bindings,
        "prepared": prepared,
        "primary": primary,
        "primary_plan": primary_plan,
        "primary_manifest": primary_manifest,
        "registered": registered,
        "trajectory": trajectory,
        "trajectory_plan": trajectory_plan,
    }


def test_trajectory_authority_chain_requires_a_nonzero_exact_primary_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    fixture = _exact_primary_trajectory_chain_inputs(tmp_path, monkeypatch)
    repository = fixture["repository"]
    round_dir = fixture["round_dir"]
    primary = fixture["primary"]
    primary_plan = fixture["primary_plan"]
    primary_manifest = fixture["primary_manifest"]
    assert isinstance(repository, Path)
    assert isinstance(round_dir, Path)
    assert len(primary_plan.entries) == 40
    assert primary_manifest["planned_run_count"] == 40
    assert (
        final_runner._validate_trajectory_primary_authority_chain(
            repository,
            round_dir,
            primary_final_plan_sha256=primary_plan.plan_sha256,
        )["primary_execution_authority_sha256"]
        == primary.authority_sha256
    )

    manifest_path = round_dir / "results/final/execution/execution_manifest.json"
    zero = dict(primary_manifest)
    zero["planned_run_count"] = 0
    zero["recorded_run_count"] = 0
    zero["records"] = []
    zero_body = {key: value for key, value in zero.items() if key != "manifest_sha256"}
    zero["manifest_sha256"] = canonical_sha256(zero_body)
    manifest_path.write_text(
        json.dumps(zero, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="denominator|manifest|records|incomplete",
    ):
        final_runner._validate_trajectory_primary_authority_chain(
            repository,
            round_dir,
            primary_final_plan_sha256=primary_plan.plan_sha256,
        )


def test_trajectory_authority_chain_rechecks_the_exact_panel_after_store_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.final_runner as final_runner

    fixture = _exact_primary_trajectory_chain_inputs(tmp_path, monkeypatch)
    repository = fixture["repository"]
    round_dir = fixture["round_dir"]
    primary_plan = fixture["primary_plan"]
    assert isinstance(repository, Path)
    assert isinstance(round_dir, Path)
    original_loader = final_runner.load_prepared_final_panel
    calls = 0

    def drifting_panel(selected: Path, destination: Path, **kwargs):
        nonlocal calls
        calls += 1
        bindings, prepared = original_loader(selected, destination, **kwargs)
        if calls == 1:
            return bindings, prepared
        return (
            (
                replace(bindings[0], manifest_sha256="9" * 64),
                *bindings[1:],
            ),
            prepared,
        )

    monkeypatch.setattr(
        final_runner,
        "load_prepared_final_panel",
        drifting_panel,
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="changed|frozen|rederivation|authority",
    ):
        final_runner._validate_trajectory_primary_authority_chain(
            repository,
            round_dir,
            primary_final_plan_sha256=primary_plan.plan_sha256,
        )
    assert calls == 2


def test_trajectory_authority_chain_rejects_coordinated_primary_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    fixture = _exact_primary_trajectory_chain_inputs(tmp_path, monkeypatch)
    repository = fixture["repository"]
    round_dir = fixture["round_dir"]
    frozen = fixture["frozen"]
    registry = fixture["registry"]
    bindings = fixture["bindings"]
    claim_sha256 = fixture["claim_sha256"]
    environment_sha256 = fixture["environment_sha256"]
    primary_plan = fixture["primary_plan"]
    trajectory = fixture["trajectory"]
    assert isinstance(repository, Path)
    assert isinstance(round_dir, Path)
    assert (
        final_runner._validate_trajectory_primary_authority_chain(
            repository,
            round_dir,
            primary_final_plan_sha256=primary_plan.plan_sha256,
        )["primary_execution_authority_sha256"]
        == fixture["primary"].authority_sha256
    )

    primary_authority_path = (
        round_dir / "results/final/execution_authority/authority.json"
    )
    primary_authority = json.loads(primary_authority_path.read_text(encoding="utf-8"))
    replacement_base = {"latent_dim": 999}
    primary_authority["base_configuration"] = replacement_base
    primary_authority["base_configuration_sha256"] = canonical_sha256(replacement_base)
    primary_authority_body = {
        key: value
        for key, value in primary_authority.items()
        if key != "authority_sha256"
    }
    primary_authority["authority_sha256"] = canonical_sha256(primary_authority_body)
    primary_authority_path.write_text(
        json.dumps(primary_authority, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replacement_primary_sha256 = primary_authority["authority_sha256"]
    forged_plan = final_runner.build_final_execution_plan(
        frozen,
        registry,
        bindings,
        execution_claim_sha256=claim_sha256,
        execution_environment_sha256=environment_sha256,
        execution_authority_sha256=replacement_primary_sha256,
    )

    primary_manifest_path = (
        round_dir / "results/final/execution/execution_manifest.json"
    )
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    primary_manifest["plan_sha256"] = forged_plan.plan_sha256
    primary_manifest["input_hashes"] = dict(forged_plan.input_hashes)
    primary_manifest_body = {
        key: value
        for key, value in primary_manifest.items()
        if key != "manifest_sha256"
    }
    primary_manifest["manifest_sha256"] = canonical_sha256(primary_manifest_body)
    primary_manifest_path.write_text(
        json.dumps(primary_manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    score_path = repository / trajectory.count_score_manifest_path
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["primary_execution_authority_sha256"] = replacement_primary_sha256
    score["primary_final_plan_sha256"] = forged_plan.plan_sha256
    score_body = {key: value for key, value in score.items() if key != "payload_sha256"}
    score["payload_sha256"] = canonical_sha256(score_body)
    score_path.write_text(
        json.dumps(score, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    trajectory_authority_path = (
        round_dir / "results/trajectory/execution_authority/authority.json"
    )
    authority = json.loads(trajectory_authority_path.read_text(encoding="utf-8"))
    authority["primary_execution_authority_sha256"] = replacement_primary_sha256
    authority["primary_final_plan_sha256"] = forged_plan.plan_sha256
    authority["base_configuration"] = replacement_base
    authority["base_configuration_sha256"] = canonical_sha256(replacement_base)
    authority["count_score_authority_sha256"] = hashlib.sha256(
        score_path.read_bytes()
    ).hexdigest()
    authority_body = {
        key: value for key, value in authority.items() if key != "authority_sha256"
    }
    authority["authority_sha256"] = canonical_sha256(authority_body)
    trajectory_authority_path.write_text(
        json.dumps(authority, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="derived from the primary|primary authority|frozen",
    ):
        final_runner._validate_trajectory_primary_authority_chain(
            repository,
            round_dir,
            primary_final_plan_sha256=forged_plan.plan_sha256,
        )


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
    store = _final_store(output, plan)
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
    temporary = runs / ".run.stdout.01234567.tmp"
    published = runs / "run.stdout"
    temporary.write_bytes(b"stable artifact")
    published.hardlink_to(temporary)
    untouched = runs / ".not-a-staging-file"
    untouched.write_bytes(b"keep")

    removed = _remove_stale_result_temporaries(round_dir)

    assert removed == ("results/final/execution/runs/.run.stdout.01234567.tmp",)
    assert not temporary.exists()
    assert published.read_bytes() == b"stable artifact"
    assert published.stat().st_nlink == 1
    assert untouched.read_bytes() == b"keep"


def test_stale_result_temporary_recovery_rejects_unrelated_hardlink(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _remove_stale_result_temporaries,
    )

    round_dir = tmp_path / "round-001"
    runs = round_dir / "results/final/execution/runs"
    runs.mkdir(parents=True)
    temporary = runs / ".run.stdout.01234567.tmp"
    external = tmp_path / "external-hardlink"
    external.write_bytes(b"must survive")
    temporary.hardlink_to(external)

    with pytest.raises(FinalRunnerContractError, match="temporary|hardlink|sibling"):
        _remove_stale_result_temporaries(round_dir)

    assert temporary.exists()
    assert external.read_bytes() == b"must survive"
    assert external.stat().st_nlink == 2


def test_interrupted_final_attempt_transaction_removes_orphan_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.final_runner import (
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
    store = _final_store(output, plan)
    original = store._stored_final_attempt

    def interrupt_after_artifacts(attempt, *, artifacts=None):
        stored = original(attempt, artifacts=artifacts)
        if artifacts is None:
            raise RuntimeError("interrupted after artifacts")
        return stored

    monkeypatch.setattr(store, "_stored_final_attempt", interrupt_after_artifacts)
    with pytest.raises(RuntimeError, match="interrupted"):
        store.append(plan.entries[0], _completed_attempt(plan.entries[0]))

    assert any((output / "runs").iterdir())
    assert any((output / "transactions").iterdir())

    recovered = _recover_interrupted_final_transactions(round_dir)

    assert recovered == (1,)
    assert not (output / "transactions").exists()
    assert not (output / "runs").exists()
    assert _final_store(output, plan).load_records() == ()


def test_final_execution_manifest_rejects_boolean_integer_aliases(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        build_final_execution_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

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
    output = tmp_path / "round-001/results/final/execution"
    store = _final_store(output, plan)
    store.append(plan.entries[0], _completed_attempt(plan.entries[0]))
    valid = store.finalize()
    assert store.load_manifest() == valid

    for field in ("schema_version", "planned_run_count", "recorded_run_count"):
        payload = json.loads(json.dumps(valid))
        payload[field] = True
        body = {
            key: value for key, value in payload.items() if key != "manifest_sha256"
        }
        payload["manifest_sha256"] = canonical_sha256(body)
        _write_canonical_json(store.manifest_path, payload)
        with pytest.raises(FinalRunnerContractError, match="manifest"):
            store.load_manifest()

    payload = json.loads(json.dumps(valid))
    payload["records"][0]["ordinal"] = True
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    payload["manifest_sha256"] = canonical_sha256(body)
    _write_canonical_json(store.manifest_path, payload)
    with pytest.raises(FinalRunnerContractError, match="manifest"):
        store.load_manifest()


def test_final_transaction_intent_rejects_boolean_integer_aliases(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _recover_interrupted_final_transactions,
        build_final_execution_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

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
    for field in ("schema_version", "ordinal"):
        round_dir = tmp_path / field
        output = round_dir / "results/final/execution"
        store = _final_store(output, plan)
        intent_path = store._publish_transaction_intent(
            plan.entries[0],
            _completed_attempt(plan.entries[0]),
        )
        payload = json.loads(intent_path.read_text(encoding="utf-8"))
        payload[field] = True
        body = {key: value for key, value in payload.items() if key != "intent_sha256"}
        payload["intent_sha256"] = canonical_sha256(body)
        _write_canonical_json(intent_path, payload)
        with pytest.raises(FinalRunnerContractError, match="transaction intent"):
            _recover_interrupted_final_transactions(round_dir)


def test_interrupted_maskimpute_transaction_removes_realized_score_artifact(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
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
    store = _final_store(output, plan)
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


def test_interrupted_trajectory_transactions_recover_before_and_after_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.final_runner import (
        FinalResultStore,
        _recover_interrupted_trajectory_transactions,
    )

    repository, round_dir, registered, authority, plan = _trajectory_store_inputs(
        tmp_path
    )
    output = round_dir / "results/trajectory/execution"
    prepared = {registered.binding.dataset_id: registered.prepared}
    store = FinalResultStore(
        output,
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )
    entry = plan.entries[0]
    attempt = _unavailable_attempt(entry, prepared=registered.prepared)
    original = store._stored_final_attempt

    def interrupt_before_record(attempt_value, *, artifacts=None):
        stored = original(attempt_value, artifacts=artifacts)
        if artifacts is None:
            raise RuntimeError("trajectory interrupted before record")
        return stored

    monkeypatch.setattr(store, "_stored_final_attempt", interrupt_before_record)
    with pytest.raises(RuntimeError, match="before record"):
        store.append(entry, attempt)
    assert _recover_interrupted_trajectory_transactions(round_dir) == (1,)
    assert not (output / "records").exists()

    committed = FinalResultStore(
        output,
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )

    def interrupt_after_record(_intent_path):
        raise RuntimeError("trajectory interrupted after record")

    monkeypatch.setattr(committed, "_complete_transaction", interrupt_after_record)
    with pytest.raises(RuntimeError, match="after record"):
        committed.append(entry, attempt)
    assert (output / "records/00000001.json").is_file()
    assert _recover_interrupted_trajectory_transactions(round_dir) == (1,)
    resumed = FinalResultStore(
        output,
        plan,
        prepared,
        authority,
        authority_repository=repository,
    )
    assert len(resumed.load_records()) == 1


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
    (results / ".artifact.01234567.tmp").symlink_to(target)

    with pytest.raises(FinalRunnerContractError, match="temporary.*regular"):
        _remove_stale_result_temporaries(round_dir)


def test_frozen_final_round_public_api_accepts_only_runtime_locators() -> None:
    from maskimpute_benchmark.final_runner import run_frozen_final_round

    parameters = inspect.signature(run_frozen_final_round).parameters

    assert tuple(parameters) == (
        "repository",
        "round_dir",
        "simulator_assets_root",
        "simulator_r_environment",
    )
    assert parameters["repository"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["round_dir"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("simulator_assets_root", "simulator_r_environment"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


@pytest.mark.parametrize("resuming", (False, True), ids=("unclaimed", "resumable"))
def test_runtime_preclaim_failure_precedes_every_lifecycle_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resuming: bool,
) -> None:
    import maskimpute_benchmark.datasets as datasets
    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.publication_freeze as publication_freeze
    import maskimpute_benchmark.simulators.base as simulator_base
    import maskimpute_benchmark.simulators.runtime_assets as runtime_assets_module
    import maskimpute_benchmark.study as study

    repository, round_dir = _claimed_lifecycle_round(tmp_path, claimed=resuming)
    simulator_assets_root = tmp_path / "external-simulator-assets"
    simulator_r_environment = tmp_path / "simulator-r-environment"
    calls: list[str] = []

    monkeypatch.setattr(
        publication_freeze,
        "validate_frozen_method",
        lambda selected: {} if selected == repository.resolve() else None,
    )

    def reject_runtime(
        selected: Path,
        *,
        external_root: Path,
        r_environment: Path,
        require_outside_repository: bool,
    ) -> object:
        calls.append("runtime_preclaim")
        assert selected == repository.resolve()
        assert external_root is simulator_assets_root
        assert r_environment is simulator_r_environment
        assert require_outside_repository is True
        raise RuntimeError("runtime preclaim rejected")

    monkeypatch.setattr(
        runtime_assets_module,
        "load_simulator_runtime_assets",
        reject_runtime,
    )

    def forbid(name: str):
        def forbidden(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"{name} ran before runtime preclaim")

        return forbidden

    monkeypatch.setattr(study, "assert_final_runnable", forbid("claim"))
    monkeypatch.setattr(
        final_runner,
        "_remove_stale_result_temporaries",
        forbid("stale_temporary_cleanup"),
    )
    monkeypatch.setattr(
        final_runner,
        "_recover_interrupted_final_transactions",
        forbid("final_transaction_recovery"),
    )
    monkeypatch.setattr(
        final_runner,
        "_recover_interrupted_trajectory_transactions",
        forbid("trajectory_transaction_recovery"),
    )
    monkeypatch.setattr(
        final_runner,
        "_recover_scaling_transactions_for_resume",
        forbid("scaling_recovery"),
    )
    monkeypatch.setattr(
        final_runner,
        "_reconcile_interrupted_final_publications",
        forbid("reconciliation"),
    )
    monkeypatch.setattr(
        simulator_base,
        "load_final_manifest_claim",
        forbid("claim_validation"),
    )
    monkeypatch.setattr(datasets, "generate_dataset_panel", forbid("generation"))
    before = _read_only_tree_snapshot(round_dir)

    with pytest.raises(final_runner.FinalRunnerContractError, match="runtime"):
        final_runner.run_frozen_final_round(
            repository,
            round_dir,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )

    assert calls == ["runtime_preclaim"]
    assert _read_only_tree_snapshot(round_dir) == before


@pytest.mark.parametrize(
    "mismatched_field",
    ("runtime_assets_sha256", "runtime_assets_receipt"),
)
def test_runtime_preclaim_propagates_exact_paths_and_blocks_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatched_field: str,
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.datasets as datasets
    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.methods as methods
    import maskimpute_benchmark.publication_freeze as publication_freeze
    import maskimpute_benchmark.simulators.base as simulator_base
    import maskimpute_benchmark.simulators.runtime_assets as runtime_assets_module

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    simulator_assets_root = tmp_path / "external-simulator-assets"
    simulator_r_environment = tmp_path / "simulator-r-environment"
    semantic_sha256 = "a" * 64
    semantic_receipt: dict[str, object] = {
        "schema": "runtime-receipt-fixture-v1",
        "nested": {"value": "stable"},
    }
    snapshot = _FakeRuntimeSnapshot(semantic_sha256, semantic_receipt)
    runtime_calls: list[dict[str, object]] = []
    generation_calls: list[dict[str, object]] = []
    preparation_calls: list[dict[str, object]] = []
    method_registry_calls: list[Path] = []
    executor_calls: list[object] = []
    binding = SimpleNamespace(manifest_sha256="b" * 64)

    monkeypatch.setattr(
        publication_freeze,
        "validate_frozen_method",
        lambda _repository: {},
    )

    def load_runtime(
        selected: Path,
        *,
        external_root: Path,
        r_environment: Path,
        require_outside_repository: bool,
    ) -> _FakeRuntimeSnapshot:
        runtime_calls.append(
            {
                "repo": selected,
                "external_root": external_root,
                "r_environment": r_environment,
                "require_outside_repository": require_outside_repository,
            }
        )
        return snapshot

    monkeypatch.setattr(
        runtime_assets_module,
        "load_simulator_runtime_assets",
        load_runtime,
    )
    monkeypatch.setattr(
        final_runner,
        "_recover_scaling_transactions_for_resume",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        final_runner,
        "_reconcile_interrupted_final_publications",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        simulator_base,
        "load_final_manifest_claim",
        lambda *_args, **_kwargs: object(),
    )

    def generate_panel(
        *,
        repo: Path,
        namespace: str,
        round_dir: Path,
        simulator_assets_root: Path,
        simulator_r_environment: Path,
    ) -> dict[str, object]:
        generation_calls.append(
            {
                "repo": repo,
                "namespace": namespace,
                "round_dir": round_dir,
                "simulator_assets_root": simulator_assets_root,
                "simulator_r_environment": simulator_r_environment,
            }
        )
        status: dict[str, object] = {
            "manifest_sha256": binding.manifest_sha256,
            "runtime_assets_sha256": semantic_sha256,
            "runtime_assets_receipt": json.loads(json.dumps(semantic_receipt)),
        }
        if mismatched_field == "runtime_assets_sha256":
            status[mismatched_field] = "c" * 64
        else:
            status[mismatched_field] = {
                "schema": "runtime-receipt-fixture-v1",
                "nested": {"value": "drifted"},
            }
        return status

    monkeypatch.setattr(datasets, "generate_dataset_panel", generate_panel)

    def load_panel(
        selected: Path,
        destination: Path,
        *,
        simulator_assets_root: Path,
        simulator_r_environment: Path,
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        preparation_calls.append(
            {
                "repository": selected,
                "round_dir": destination,
                "simulator_assets_root": simulator_assets_root,
                "simulator_r_environment": simulator_r_environment,
            }
        )
        return (binding,), {}

    monkeypatch.setattr(final_runner, "load_prepared_final_panel", load_panel)

    def forbid_method_registry(path: Path) -> object:
        method_registry_calls.append(path)
        raise AssertionError("method registry loaded after runtime semantic drift")

    monkeypatch.setattr(methods, "load_method_registry", forbid_method_registry)
    monkeypatch.setattr(
        final_runner,
        "SpawnedRepositoryExecutor",
        lambda dispatcher: executor_calls.append(dispatcher),
    )

    with pytest.raises(final_runner.FinalRunnerContractError, match="runtime"):
        final_runner.run_frozen_final_round(
            repository,
            round_dir,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )

    assert runtime_calls == [
        {
            "repo": repository.resolve(),
            "external_root": simulator_assets_root,
            "r_environment": simulator_r_environment,
            "require_outside_repository": True,
        }
    ]
    assert generation_calls == [
        {
            "repo": repository.resolve(),
            "namespace": "final",
            "round_dir": round_dir.resolve(),
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
        }
    ]
    assert preparation_calls == [
        {
            "repository": repository.resolve(),
            "round_dir": round_dir.resolve(),
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
        }
    ]
    assert snapshot.receipt_reads == 1
    assert snapshot.close_count == 1
    assert method_registry_calls == []
    assert executor_calls == []


def test_runtime_preclaim_closes_snapshot_when_semantic_capture_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.publication_freeze as publication_freeze
    import maskimpute_benchmark.simulators.runtime_assets as runtime_assets_module
    import maskimpute_benchmark.study as study

    repository, round_dir = _claimed_lifecycle_round(tmp_path, claimed=False)
    simulator_assets_root = tmp_path / "external-simulator-assets"
    simulator_r_environment = tmp_path / "simulator-r-environment"
    snapshot = _FakeRuntimeSnapshot(
        "a" * 64,
        {"schema": "runtime-receipt-fixture-v1"},
        fail_receipt=True,
    )
    claim_calls: list[object] = []

    monkeypatch.setattr(
        publication_freeze,
        "validate_frozen_method",
        lambda _repository: {},
    )
    monkeypatch.setattr(
        runtime_assets_module,
        "load_simulator_runtime_assets",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        study,
        "assert_final_runnable",
        lambda *_args, **_kwargs: claim_calls.append(object()),
    )
    before = _read_only_tree_snapshot(round_dir)

    with pytest.raises(final_runner.FinalRunnerContractError, match="runtime"):
        final_runner.run_frozen_final_round(
            repository,
            round_dir,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )

    assert snapshot.receipt_reads == 1
    assert snapshot.close_count == 1
    assert claim_calls == []
    assert _read_only_tree_snapshot(round_dir) == before


def test_frozen_round_preflights_all_scopes_before_first_adapter() -> None:
    from maskimpute_benchmark.final_runner import run_frozen_final_round

    source = inspect.getsource(run_frozen_final_round)

    assert source.index("_validate_combined_storage_capacity(") < source.index(
        "SpawnedRepositoryExecutor("
    )
    assert source.index("execute_trajectory_plan(") < source.index(
        "_record_final_evaluation_after_scaling("
    )


@pytest.mark.parametrize(
    "crash_after",
    ("trajectory_dataset", "primary_authority", "trajectory_authority"),
)
def test_final_input_publications_are_journaled_before_the_next_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_after: str,
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner

    stages: list[str] = []
    registered = object()
    primary_authority = SimpleNamespace(authority_sha256="1" * 64)
    primary_plan = SimpleNamespace(plan_sha256="2" * 64)
    trajectory_authority = SimpleNamespace(authority_sha256="3" * 64)
    trajectory_plan = object()

    def materialize_dataset(*_args, **_kwargs):
        stages.append("trajectory_dataset")
        return registered

    def materialize_primary(*_args, **_kwargs):
        stages.append("primary_authority")
        return primary_authority

    def build_primary(*_args, **_kwargs):
        return primary_plan

    def materialize_trajectory(*_args, **_kwargs):
        stages.append("trajectory_authority")
        return trajectory_authority

    def build_trajectory(*_args, **_kwargs):
        return trajectory_plan

    monkeypatch.setattr(
        final_runner,
        "materialize_prepared_trajectory_dataset",
        materialize_dataset,
    )
    monkeypatch.setattr(
        final_runner,
        "materialize_final_execution_authority",
        materialize_primary,
    )
    monkeypatch.setattr(final_runner, "build_final_execution_plan", build_primary)
    monkeypatch.setattr(
        final_runner,
        "materialize_trajectory_execution_authority",
        materialize_trajectory,
    )
    monkeypatch.setattr(
        final_runner,
        "build_trajectory_execution_plan",
        build_trajectory,
    )

    def journal() -> None:
        stage = stages[-1]
        stages.append(f"journal:{stage}")
        if stage == crash_after:
            raise RuntimeError(f"crash after {stage}")

    with pytest.raises(RuntimeError, match=f"crash after {crash_after}"):
        final_runner._materialize_final_execution_inputs(
            tmp_path,
            tmp_path,
            {},
            object(),
            (SimpleNamespace(manifest_sha256="4" * 64),),
            execution_claim_sha256="5" * 64,
            execution_environment_sha256="6" * 64,
            publish_results=journal,
        )

    expected: list[str] = []
    for stage in (
        "trajectory_dataset",
        "primary_authority",
        "trajectory_authority",
    ):
        expected.extend((stage, f"journal:{stage}"))
        if stage == crash_after:
            break
    assert stages == expected


def test_resume_reconciles_interrupted_publications_before_claim_validation() -> None:
    from maskimpute_benchmark.final_runner import run_frozen_final_round

    source = inspect.getsource(run_frozen_final_round)
    resume = source[source.index("if resuming:") : source.index("else:")]

    assert resume.index("_reconcile_interrupted_final_publications(") < resume.index(
        "load_final_manifest_claim("
    )


@pytest.mark.parametrize(
    "crash_stage",
    (
        "trajectory_dataset",
        "primary_authority",
        "trajectory_authority",
        "primary_manifest",
        "trajectory_manifest",
        "scaling_checkpoint",
    ),
)
def test_frozen_round_second_invocation_recovers_each_publication_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_stage: str,
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.datasets as datasets
    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.methods as methods
    import maskimpute_benchmark.publication_freeze as publication_freeze
    import maskimpute_benchmark.scaling as scaling
    import maskimpute_benchmark.simulators.runtime_assets as runtime_assets_module
    import maskimpute_benchmark.study as study

    registry = load_method_registry(METHODS)
    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    simulator_assets_root = tmp_path / "external-simulator-assets"
    simulator_r_environment = tmp_path / "simulator-r-environment"
    expected_simulator_assets_root = simulator_assets_root
    expected_simulator_r_environment = simulator_r_environment
    runtime_sha256 = "8" * 64
    runtime_receipt: dict[str, object] = {"schema": "runtime-receipt-fixture-v1"}
    runtime_snapshots: list[_FakeRuntimeSnapshot] = []
    frozen = {"runtime_lock_sha256": "1" * 64}
    binding = SimpleNamespace(manifest_sha256="2" * 64, dataset_id="dataset-fixture")
    primary_plan = SimpleNamespace(
        entries=(),
        configurations=(),
        plan_sha256="3" * 64,
        input_hashes={},
    )
    trajectory_plan = SimpleNamespace(
        entries=(),
        configurations=(),
        plan_sha256="4" * 64,
        input_hashes={},
        scope="supplementary_trajectory",
    )
    primary_authority = SimpleNamespace(authority_sha256="5" * 64)
    trajectory_authority = SimpleNamespace(authority_sha256="6" * 64)
    registered = SimpleNamespace(
        binding=SimpleNamespace(dataset_id="trajectory-fixture"),
        prepared=object(),
    )
    current_stage = "setup"
    crashed = False
    real_reconcile = final_runner._record_incremental_results_if_changed

    def generic_owned_manifest(selected_round: Path) -> dict[str, object]:
        return final_runner.final_result_file_manifest(selected_round)

    monkeypatch.setattr(
        final_runner,
        "_owned_final_result_file_manifest",
        generic_owned_manifest,
    )

    def load_runtime(
        selected: Path,
        *,
        external_root: Path,
        r_environment: Path,
        require_outside_repository: bool,
    ) -> _FakeRuntimeSnapshot:
        assert selected == repository.resolve()
        assert external_root is simulator_assets_root
        assert r_environment is simulator_r_environment
        assert require_outside_repository is True
        snapshot = _FakeRuntimeSnapshot(runtime_sha256, runtime_receipt)
        runtime_snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(
        runtime_assets_module,
        "load_simulator_runtime_assets",
        load_runtime,
    )

    def generate_panel(
        *,
        repo,
        namespace,
        round_dir,
        simulator_assets_root,
        simulator_r_environment,
    ):
        assert simulator_assets_root is expected_simulator_assets_root
        assert simulator_r_environment is expected_simulator_r_environment
        status = round_dir / "results/dataset_status.json"
        if not status.exists():
            status.parent.mkdir(parents=True, exist_ok=True)
            status.write_text('{"fixture":"final-panel"}\n', encoding="utf-8")
            study.record_incremental_results(
                round_dir,
                {
                    "result_files": [
                        {
                            "path": "results/dataset_status.json",
                            "sha256": hashlib.sha256(status.read_bytes()).hexdigest(),
                        }
                    ]
                },
                repo=repo,
            )
        return {
            "manifest_sha256": binding.manifest_sha256,
            "runtime_assets_sha256": runtime_sha256,
            "runtime_assets_receipt": json.loads(json.dumps(runtime_receipt)),
        }

    monkeypatch.setattr(datasets, "generate_dataset_panel", generate_panel)
    generate_panel(
        repo=repository,
        namespace="final",
        round_dir=round_dir,
        simulator_assets_root=simulator_assets_root,
        simulator_r_environment=simulator_r_environment,
    )

    def load_panel(
        _repository,
        _round,
        *,
        simulator_assets_root,
        simulator_r_environment,
    ):
        assert simulator_assets_root is expected_simulator_assets_root
        assert simulator_r_environment is expected_simulator_r_environment
        return (binding,), {}

    monkeypatch.setattr(
        final_runner,
        "load_prepared_final_panel",
        load_panel,
    )
    monkeypatch.setattr(
        final_runner,
        "_load_unclaimed_prepared_final_panel",
        lambda _repository, _round: ((binding,), {}),
        raising=False,
    )
    monkeypatch.setattr(
        publication_freeze,
        "validate_frozen_method",
        lambda _repository: frozen,
    )
    monkeypatch.setattr(methods, "load_method_registry", lambda _path: registry)
    monkeypatch.setattr(
        final_runner,
        "ExecutionEnvironmentRegistry",
        SimpleNamespace(
            fixed=lambda *_args, **_kwargs: SimpleNamespace(
                registry_sha256="7" * 64,
                runtime_lock_sha256="1" * 64,
            )
        ),
    )
    monkeypatch.setattr(
        final_runner,
        "_validate_final_runtime_lock",
        lambda *_args, **_kwargs: "1" * 64,
    )
    monkeypatch.setattr(
        final_runner,
        "build_final_execution_plan",
        lambda *_args, **_kwargs: primary_plan,
    )
    monkeypatch.setattr(
        final_runner,
        "build_trajectory_execution_plan",
        lambda *_args, **_kwargs: trajectory_plan,
    )

    def publish(relative: str, payload: bytes) -> None:
        path = round_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == payload
        else:
            path.write_bytes(payload)

    def materialize_dataset(*_args, **_kwargs):
        nonlocal current_stage
        current_stage = "trajectory_dataset"
        publish("results/trajectory/dataset/evaluator.h5ad", b"dataset\n")
        publish("results/trajectory/dataset/dataset_receipt.json", b"{}\n")
        return registered

    def materialize_primary(*_args, **_kwargs):
        nonlocal current_stage
        current_stage = "primary_authority"
        publish("results/final/execution_authority/authority.json", b"{}\n")
        return primary_authority

    def materialize_trajectory(*_args, **_kwargs):
        nonlocal current_stage
        current_stage = "trajectory_authority"
        publish("results/trajectory/execution_authority/authority.json", b"{}\n")
        return trajectory_authority

    monkeypatch.setattr(
        final_runner,
        "materialize_prepared_trajectory_dataset",
        materialize_dataset,
    )
    monkeypatch.setattr(
        final_runner,
        "materialize_final_execution_authority",
        materialize_primary,
    )
    monkeypatch.setattr(
        final_runner,
        "materialize_trajectory_execution_authority",
        materialize_trajectory,
    )

    def journal(repository_value, round_value, recorder):
        nonlocal crashed
        if current_stage == crash_stage and not crashed:
            crashed = True
            raise RuntimeError(f"crash after {current_stage}")
        return real_reconcile(repository_value, round_value, recorder)

    monkeypatch.setattr(
        final_runner,
        "_record_incremental_results_if_changed",
        journal,
    )
    monkeypatch.setattr(
        final_runner,
        "_recover_scaling_transactions_for_resume",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        final_runner,
        "_validate_scaling_publications_for_reconciliation",
        lambda *_args, **_kwargs: None,
    )

    class FakeStore:
        def __init__(self, output_dir, *_args, **_kwargs):
            self.output_dir = output_dir
            self.manifest_path = output_dir / "execution_manifest.json"
            self._records_cache = ()

        def load_records(self):
            return ()

        def _cached_records(self):
            return ()

        def load_manifest(self):
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(final_runner, "FinalResultStore", FakeStore)
    monkeypatch.setattr(
        scaling,
        "load_scaling_execution_authority",
        lambda _repository: object(),
    )
    monkeypatch.setattr(
        final_runner,
        "_validate_combined_storage_capacity",
        lambda *_args, **_kwargs: {"schema": "fixture"},
    )
    monkeypatch.setattr(
        final_runner,
        "RepositoryAdapterDispatcher",
        lambda *_args, **_kwargs: object(),
    )

    class FakeExecutor:
        def __init__(self, _dispatcher):
            pass

        def close(self):
            pass

    monkeypatch.setattr(final_runner, "SpawnedRepositoryExecutor", FakeExecutor)

    def execute_primary(*_args, on_record_published, **_kwargs):
        nonlocal current_stage
        current_stage = "primary_manifest"
        publish(
            "results/final/execution/execution_manifest.json",
            b'{"manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n',
        )
        on_record_published()
        return {"manifest_sha256": "a" * 64}

    def execute_trajectory(*_args, on_record_published, **_kwargs):
        nonlocal current_stage
        current_stage = "trajectory_manifest"
        publish(
            "results/trajectory/execution/execution_manifest.json",
            b'{"manifest_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}\n',
        )
        on_record_published()
        return {"manifest_sha256": "b" * 64}

    monkeypatch.setattr(final_runner, "execute_final_plan", execute_primary)
    monkeypatch.setattr(final_runner, "execute_trajectory_plan", execute_trajectory)
    monkeypatch.setattr(
        final_runner,
        "validate_final_execution_for_evaluation",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        final_runner,
        "validate_trajectory_execution_for_evaluation",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        final_runner,
        "_trajectory_evaluation_evidence",
        lambda *_args, **_kwargs: {"evidence_sha256": "c" * 64},
    )

    def finalize(
        _repository,
        _round,
        _evaluation,
        *,
        simulator_assets_root,
        simulator_r_environment,
    ):
        nonlocal current_stage
        assert simulator_assets_root is expected_simulator_assets_root
        assert simulator_r_environment is expected_simulator_r_environment
        current_stage = "scaling_checkpoint"
        publish("results/scaling/checkpoints/00000001.json", b"{}\n")
        journal(repository, round_dir, study.record_incremental_results)
        return {"state": "evaluated"}

    monkeypatch.setattr(
        final_runner,
        "_record_final_evaluation_after_scaling",
        finalize,
    )

    with pytest.raises(RuntimeError, match=f"crash after {crash_stage}"):
        final_runner.run_frozen_final_round(
            repository,
            round_dir,
            simulator_assets_root=simulator_assets_root,
            simulator_r_environment=simulator_r_environment,
        )
    assert crashed is True

    result = final_runner.run_frozen_final_round(
        repository,
        round_dir,
        simulator_assets_root=simulator_assets_root,
        simulator_r_environment=simulator_r_environment,
    )
    assert result["evaluation_receipt"] == {"state": "evaluated"}
    assert len(runtime_snapshots) == 2
    assert all(snapshot.close_count == 1 for snapshot in runtime_snapshots)


def test_unreceipted_trajectory_dataset_recovery_is_closed(tmp_path: Path) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    round_dir = tmp_path / "round-001"
    dataset = round_dir / "results/trajectory/dataset/evaluator.h5ad"
    receipt = round_dir / "results/trajectory/dataset/dataset_receipt.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"interrupted owned dataset")

    assert final_runner._remove_unreceipted_trajectory_dataset(round_dir) is True
    assert not dataset.exists()

    receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="receipt.*without.*dataset|incomplete",
    ):
        final_runner._remove_unreceipted_trajectory_dataset(round_dir)


def test_unreceipted_trajectory_dataset_rechecks_receipt_through_parent_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    round_dir = tmp_path / "round-001"
    dataset = round_dir / "results/trajectory/dataset/evaluator.h5ad"
    receipt = round_dir / "results/trajectory/dataset/dataset_receipt.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"interrupted owned dataset")
    real_open = final_runner.os.open
    injected = False

    def open_with_concurrent_receipt(path, flags, *args, **kwargs):
        nonlocal injected
        descriptor = real_open(path, flags, *args, **kwargs)
        if not injected and Path(path) == dataset.parent:
            injected = True
            receipt.write_text("{}\n", encoding="utf-8")
        return descriptor

    monkeypatch.setattr(final_runner.os, "open", open_with_concurrent_receipt)

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="receipt appeared",
    ):
        final_runner._remove_unreceipted_trajectory_dataset(round_dir)
    assert dataset.read_bytes() == b"interrupted owned dataset"
    assert receipt.read_text(encoding="utf-8") == "{}\n"


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


def test_final_runtime_registry_receives_validated_method_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    registry = load_method_registry(METHODS)
    captured: dict[str, object] = {}
    expected = object()

    def fixed(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        final_runner,
        "ExecutionEnvironmentRegistry",
        type("RegistryFixture", (), {"fixed": staticmethod(fixed)}),
    )

    observed = final_runner._load_final_execution_environment_registry(
        Path.cwd(),
        registry,
    )

    assert observed is expected
    assert captured["kwargs"]["lock_only_environment_ids"] == (
        "d3impute",
        "sctsi",
    )


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


def test_frozen_final_cli_exposes_only_operational_locators() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_frozen_final.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "round_dir" in completed.stdout
    assert "--simulator-assets-root" in completed.stdout
    assert "--simulator-r-environment" in completed.stdout
    for forbidden in (
        "--repository",
        "--environment",
        "--seed",
        "--mechanism",
        "--configuration",
        "--method",
    ):
        assert forbidden not in completed.stdout


def test_frozen_final_cli_requires_and_forwards_runtime_locators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner

    calls: list[dict[str, object]] = []
    relative_round = Path("artifacts/study/final/round-001")
    simulator_assets_root = tmp_path / "external-simulator-assets"
    simulator_r_environment = tmp_path / "simulator-r-environment"

    def run_round(
        repository: Path,
        round_dir: Path,
        *,
        simulator_assets_root: Path,
        simulator_r_environment: Path,
    ) -> dict[str, object]:
        calls.append(
            {
                "repository": repository,
                "round_dir": round_dir,
                "simulator_assets_root": simulator_assets_root,
                "simulator_r_environment": simulator_r_environment,
                "path": os.environ.get("PATH"),
                "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
            }
        )
        return {
            "execution_manifest": {"manifest_sha256": "a" * 64},
            "evaluation_receipt": {"state": "evaluated"},
        }

    monkeypatch.setattr(final_runner, "run_frozen_final_round", run_round)
    namespace = runpy.run_path("scripts/run_frozen_final.py")
    main = namespace["main"]

    monkeypatch.setenv("PATH", "/tmp/ephemeral-codex-bin:/usr/bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/ephemeral-codex-libraries")
    monkeypatch.setattr(sys, "argv", ["run_frozen_final.py", str(relative_round)])
    with pytest.raises(SystemExit) as missing:
        main()
    assert missing.value.code == 2
    assert calls == []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_final.py",
            str(relative_round),
            "--simulator-assets-root",
            str(simulator_assets_root),
            "--simulator-r-environment",
            str(simulator_r_environment),
        ],
    )

    assert main() == 0
    assert calls == [
        {
            "repository": Path("scripts/run_frozen_final.py").resolve().parents[1],
            "round_dir": Path("scripts/run_frozen_final.py").resolve().parents[1]
            / relative_round,
            "simulator_assets_root": simulator_assets_root,
            "simulator_r_environment": simulator_r_environment,
            "path": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "ld_library_path": None,
        }
    ]


def _minimal_trajectory_evidence() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "scope": "supplementary_trajectory",
        "plan": {},
        "dataset": {},
        "execution_authority": {},
        "execution_manifest": {},
        "execution_validation": {},
        "result_files": [],
    }
    return {**body, "evidence_sha256": canonical_sha256(body)}


def test_incomplete_scaling_blocks_final_evaluation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.scaling as scaling
    import maskimpute_benchmark.study as study

    receipt_called = False

    monkeypatch.setattr(
        scaling,
        "run_scaling_panel",
        lambda _repository, _round_dir: SimpleNamespace(
            status="running",
            planned_run_count=20,
            records=(),
            datasets=(),
        ),
    )

    def record_receipt(*_args, **_kwargs):
        nonlocal receipt_called
        receipt_called = True
        return {"state": "evaluated"}

    monkeypatch.setattr(study, "record_final_evaluation", record_receipt)

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="scaling.*denominator.*incomplete",
    ):
        final_runner._record_final_evaluation_after_scaling(
            tmp_path,
            tmp_path,
            {
                "schema_version": 1,
                "status": "completed",
                "trajectory_evidence": _minimal_trajectory_evidence(),
            },
            simulator_assets_root=tmp_path / "external-simulator-assets",
            simulator_r_environment=tmp_path / "simulator-r-environment",
        )

    assert receipt_called is False


def test_complete_scaling_is_bound_before_the_only_evaluation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.scaling as scaling
    import maskimpute_benchmark.study as study

    checkpoint = SimpleNamespace(
        status="completed",
        planned_run_count=1,
        records=({"run": {"run_id": "scaling-observed-10000"}},),
        datasets=({"cells": 10_000},),
    )
    evidence = {
        "schema_version": 1,
        "status": "completed",
        "evidence_sha256": "a" * 64,
    }
    cumulative = [
        {"path": "results/main.txt", "sha256": "b" * 64},
        {"path": "results/scaling/checkpoint.json", "sha256": "c" * 64},
    ]
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        scaling,
        "run_scaling_panel",
        lambda _repository, _round_dir: checkpoint,
    )
    monkeypatch.setattr(
        final_runner,
        "_scaling_evaluation_evidence",
        lambda _repository, _round_dir, observed, _result_files: (
            evidence if observed is checkpoint else None
        ),
        raising=False,
    )
    monkeypatch.setattr(
        final_runner,
        "_owned_final_result_file_manifest",
        lambda _round_dir: {"result_files": cumulative},
    )
    monkeypatch.setattr(
        final_runner,
        "_rederive_trajectory_evidence_before_receipt",
        lambda _repository, _round_dir, observed, _files, **_kwargs: dict(observed),
    )

    def record_receipt(_round_dir, manifest, *, repo):
        recorded.update(manifest)
        recorded["repository"] = repo
        return {"state": "evaluated"}

    monkeypatch.setattr(study, "record_final_evaluation", record_receipt)
    base = {
        "schema_version": 1,
        "status": "completed",
        "final_plan_sha256": "d" * 64,
        "trajectory_evidence": _minimal_trajectory_evidence(),
        "result_files": [cumulative[0]],
    }

    result = final_runner._record_final_evaluation_after_scaling(
        tmp_path,
        tmp_path,
        base,
        simulator_assets_root=tmp_path / "external-simulator-assets",
        simulator_r_environment=tmp_path / "simulator-r-environment",
    )

    assert result == {"state": "evaluated"}
    assert recorded["scaling_evidence"] == evidence
    assert recorded["result_files"] == cumulative
    assert recorded["repository"] == tmp_path
    assert "scaling_evidence" not in base


def test_trajectory_change_during_scaling_blocks_the_only_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.scaling as scaling
    import maskimpute_benchmark.study as study

    trajectory_file = tmp_path / "results/trajectory/execution/run.stdout"
    trajectory_file.parent.mkdir(parents=True)
    trajectory_file.write_bytes(b"before")
    trajectory_path = "results/trajectory/execution/run.stdout"
    initial_digest = hashlib.sha256(b"before").hexdigest()
    trajectory_evidence = _minimal_trajectory_evidence()
    trajectory_body = {
        key: value
        for key, value in trajectory_evidence.items()
        if key != "evidence_sha256"
    }
    trajectory_body["result_files"] = [
        {"path": trajectory_path, "sha256": initial_digest}
    ]
    trajectory_evidence = {
        **trajectory_body,
        "evidence_sha256": canonical_sha256(trajectory_body),
    }
    receipt_called = False

    def run_scaling(_repository, _round_dir):
        trajectory_file.write_bytes(b"after")
        return SimpleNamespace(
            status="completed",
            planned_run_count=1,
            records=({},),
            datasets=({},),
        )

    def current_inventory(_round_dir):
        return {
            "result_files": [
                {
                    "path": trajectory_path,
                    "sha256": hashlib.sha256(trajectory_file.read_bytes()).hexdigest(),
                }
            ]
        }

    def rederive(_repository, _round_dir, observed, fresh, **_kwargs):
        assert fresh[0]["sha256"] != observed["result_files"][0]["sha256"]
        raise final_runner.FinalRunnerContractError(
            "trajectory evidence changed before receipt"
        )

    def record_receipt(*_args, **_kwargs):
        nonlocal receipt_called
        receipt_called = True
        return {}

    monkeypatch.setattr(scaling, "run_scaling_panel", run_scaling)
    monkeypatch.setattr(
        final_runner,
        "_owned_final_result_file_manifest",
        current_inventory,
    )
    monkeypatch.setattr(
        final_runner,
        "_rederive_trajectory_evidence_before_receipt",
        rederive,
    )
    monkeypatch.setattr(study, "record_final_evaluation", record_receipt)

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="trajectory evidence changed",
    ):
        final_runner._record_final_evaluation_after_scaling(
            tmp_path,
            tmp_path,
            {
                "schema_version": 1,
                "status": "completed",
                "final_plan_sha256": "d" * 64,
                "trajectory_evidence": trajectory_evidence,
            },
            simulator_assets_root=tmp_path / "external-simulator-assets",
            simulator_r_environment=tmp_path / "simulator-r-environment",
        )

    assert receipt_called is False
