from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.runner as runner_module
import maskimpute_benchmark.runtime_environments as runtime_module
from maskimpute_benchmark.comparator_tuning import (
    bind_comparator_configuration_identity,
    comparator_method_binding,
    load_comparator_tuning_authority,
)
from maskimpute_benchmark.fair_comparator_checkpoint import (
    DirectCheckpointStore,
    DirectDevelopmentBudget,
)
from maskimpute_benchmark.fair_comparator_execution import (
    DIRECT_RECONSTRUCTION_METRICS,
    DirectEvaluatedAttempt,
    DirectLogReceipt,
    DirectMetricRow,
    DirectPreZeroEvidence,
    DirectRunResult,
)
from maskimpute_benchmark.fair_comparator_plan import (
    ComparatorRunIdentity,
    DirectAuthorizedConfiguration,
    DirectCompetitionPlan,
    DirectPlanEntry,
    describe_prepared_input,
    direct_run_id,
)
from maskimpute_benchmark.methods import (
    AdapterExecution,
    DirectAdapterExecution,
    MethodInput,
    MethodRegistry,
    load_method_registry,
    finalize_direct_method_output,
    run_observed,
)
from maskimpute_benchmark.runner import (
    AdapterOutcome,
    AuthorizedConfiguration,
    DevelopmentBudget,
    DatasetBinding,
    DatasetQCAudit,
    ExecutionRequest,
    CheckpointStore,
    CalibrationFoldReceipt,
    CompetitionPlan,
    RunPlanEntry,
    RunnerAuthority,
    RunnerContractError,
    DatasetQCPolicy,
    PreparedDataset,
    ResourceSample,
    ExecutionEnvironmentRegistry,
    RepositoryAdapterDispatcher,
    SpawnedRepositoryExecutor,
    prepare_dataset_pair_for_execution,
    prepare_dataset_for_execution,
    build_competition_plan,
    build_fair_comparator_plan,
    execute_adapter_in_spawned_process,
    evaluate_adapter_outcome,
    execute_fair_comparator_request,
    _execute_fair_comparator_plan_structural,
    enforce_calibration_fold_receipt,
    execute_competition_plan,
    derive_authorized_configurations,
    method_input_sha256,
    implementation_source_sha256,
    maskimpute_variant_for_configuration,
    load_runner_authority,
    validate_development_manifest_payload,
)
from maskimpute_benchmark.runtime_environments import (
    RuntimeEnvironmentError,
    RuntimeEnvironmentEntry,
    RuntimeEnvironmentLock,
    RuntimeEnvironmentSnapshot,
    build_runtime_environment_lock,
)
from maskimpute_benchmark.protocol import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
METHODS_PATH = Path("study/methods.json")
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
VIEWS = ("moderate", "severe")


def _manifest_payload() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for mechanism in MECHANISMS:
        for draw in (1, 2):
            for view in VIEWS:
                ordinal += 1
                rows.append(
                    {
                        "biological_id": f"draw-{draw:02d}",
                        "cells": 900,
                        "dataset_id": f"dataset-{ordinal:024x}",
                        "dataset_sha256": f"{ordinal:064x}",
                        "genes": 500,
                        "independent_unit_id": f"biological-{(ordinal + 1) // 2:024x}",
                        "mechanism": mechanism,
                        "output_file_sha256": f"{ordinal + 100:064x}",
                        "output_path": (
                            f"dev/datasets/{mechanism}/draw-{draw:02d}/{view}.h5ad"
                        ),
                        "status": "completed",
                        "technical_view": view,
                        "truth_sha256": f"{(ordinal + 1) // 2 + 300:064x}",
                    }
                )
    return {
        "schema_version": 1,
        "namespace": "dev",
        "status": "completed",
        "completed_count": 16,
        "failed_count": 0,
        "independent_unit_count": 8,
        "manifest_sha256": SHA_A,
        "protocol_sha256": SHA_B,
        "design_sha256": SHA_C,
        "seed_source_sha256": "d" * 64,
        "rows": rows,
    }


def _authority(*, maskimpute_ready: bool = False) -> RunnerAuthority:
    from maskimpute_benchmark.protocol import canonical_sha256

    tracked = load_runner_authority()
    return RunnerAuthority(
        schema_version=1,
        authority_sha256="1" * 64,
        method_registry_sha256="2" * 64,
        selection_contract_sha256="3" * 64,
        development_search_sha256="4" * 64,
        ablation_registry_sha256="5" * 64,
        base_configuration_id=tracked.configurations[0].configuration_id,
        base_configuration_sha256=canonical_sha256({"method_version": "v27"}),
        base_configuration=(("method_version", "v27"),),
        count_model_config=(("n_folds", 5),),
        count_model_config_sha256=canonical_sha256({"n_folds": 5}),
        count_score_manifest_status="ready" if maskimpute_ready else "pending",
        count_score_manifest_sha256="8" * 64 if maskimpute_ready else None,
        retained_calibration_status="ready" if maskimpute_ready else "pending",
        retained_calibration_sha256="9" * 64 if maskimpute_ready else None,
        dataset_qc_policy=DatasetQCPolicy.fixed(),
        dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
        count_score_manifest_path=(
            "artifacts/study/development/count_scores/manifest.json"
        ),
        retained_calibration_path=(
            "artifacts/study/development/calibration/retained_calibration.json"
        ),
        configurations=tracked.configurations,
        comparator_tuning_reference=tracked.comparator_tuning_reference,
        comparator_method_bindings=tracked.comparator_method_bindings,
        comparator_tuning=tracked.comparator_tuning,
    )


def _method_input() -> MethodInput:
    counts = np.array([[2, 0, 1], [0, 3, 0]], dtype=np.int64)
    view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=["cell-1", "cell-2"]),
        var=pd.DataFrame(index=["gene-1", "gene-2", "gene-3"]),
    )
    view.uns["source_dataset_sha256"] = SHA_A
    view.uns["allowed_covariates"] = {"obs": [], "var": []}
    from maskimpute_benchmark.methods import prepare_method_input

    return prepare_method_input(view)


def _bound_magic_configuration():
    registry = load_method_registry(METHODS_PATH)
    tuning_authority = load_comparator_tuning_authority(Path.cwd(), registry=registry)
    spec = registry.by_id("magic")
    bound = bind_comparator_configuration_identity(
        tuning_authority.configurations_for("magic")[0],
        spec,
        tuning_authority,
    )
    return spec, bound


@pytest.fixture
def dispatcher_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> RepositoryAdapterDispatcher:
    registry = load_method_registry(METHODS_PATH)
    comparator_ids = (
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
    )
    for method_id in comparator_ids:
        cache_path = registry.by_id(method_id).source.cache_path
        assert cache_path is not None
        (tmp_path / cache_path).mkdir(parents=True, exist_ok=True)
    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        {method_id: Path(sys.executable) for method_id in comparator_ids},
    )
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "revalidate_for",
        lambda _self, _method_id: None,
    )
    tuning_authority = load_comparator_tuning_authority(
        Path.cwd(),
        registry=registry,
        require_clean=False,
    )
    return RepositoryAdapterDispatcher(
        tmp_path,
        environments,
        comparator_tuning_authority=tuning_authority,
    )


@pytest.fixture
def request_for_comparator(dispatcher_fixture: RepositoryAdapterDispatcher):
    registry = load_method_registry(METHODS_PATH)
    tuning_authority = dispatcher_fixture.comparator_tuning_authority
    assert tuning_authority is not None

    def create(method_id: str, configuration_id: str) -> ExecutionRequest:
        spec = registry.by_id(method_id)
        row = next(
            configuration
            for configuration in tuning_authority.configurations_for(method_id)
            if configuration.configuration_id == configuration_id
        )
        bound = bind_comparator_configuration_identity(
            row,
            spec,
            tuning_authority,
            runtime_lock_sha256="6" * 64,
            environment_registry_sha256=(
                dispatcher_fixture.environments.registry_sha256
            ),
        )
        return ExecutionRequest.create(
            spec,
            _method_input(),
            model_seed=42,
            configuration=AuthorizedConfiguration.from_bound_comparator(bound),
            authority=_authority(maskimpute_ready=True),
            mechanism="symsim",
            biological_id="draw-01",
            technical_view="moderate",
            dataset_id="dataset-test",
            timeout_seconds=5,
        )

    return create


def _configuration_with_substituted_method_component(
    field: str,
) -> tuple[object, AuthorizedConfiguration]:
    spec, bound = _bound_magic_configuration()
    configuration = AuthorizedConfiguration.from_bound_comparator(bound)
    substituted = "f" * 64
    identity_body = {
        "schema": "maskimpute-comparator-configuration-method-identity-v1",
        "registry_method_sha256": configuration.registry_method_sha256,
        "configuration_payload_sha256": configuration.configuration_sha256,
        "tuning_authority_file_sha256": configuration.tuning_authority_file_sha256,
        "tuning_authority_payload_sha256": (
            configuration.tuning_authority_payload_sha256
        ),
        "source_authority_sha256": configuration.source_authority_sha256,
        "runtime_lock_sha256": configuration.runtime_lock_sha256,
        "environment_registry_sha256": configuration.environment_registry_sha256,
    }
    identity_body[field] = substituted
    return spec, replace(
        configuration,
        **{
            field: substituted,
            "configuration_method_identity_sha256": canonical_sha256(identity_body),
        },
    )


def _request_integrity_values(request: ExecutionRequest) -> dict[str, object]:
    configuration = json.loads(request.configuration_payload_json)
    return {
        "method_spec_sha256": canonical_sha256(asdict(request.method_spec)),
        "method_input_sha256": method_input_sha256(request.method_input),
        "model_seed": request.model_seed,
        "dataset_id": request.dataset_id,
        "mechanism": request.mechanism,
        "biological_id": request.biological_id,
        "technical_view": request.technical_view,
        "configuration_id": request.configuration_id,
        "configuration_kind": request.configuration_kind,
        "configuration_sha256": request.configuration_sha256,
        "configuration_payload_sha256": request.configuration_payload_sha256,
        "configuration_payload": configuration,
        "registry_method_sha256": request.registry_method_sha256,
        "tuning_authority_file_sha256": request.tuning_authority_file_sha256,
        "tuning_authority_payload_sha256": (request.tuning_authority_payload_sha256),
        "source_authority_sha256": request.source_authority_sha256,
        "runtime_lock_sha256": request.runtime_lock_sha256,
        "environment_registry_sha256": request.environment_registry_sha256,
        "configuration_method_identity_sha256": (
            request.configuration_method_identity_sha256
        ),
        "nonexecution_identity_sha256": request.nonexecution_identity_sha256,
        "execution_authority_sha256": request.execution_authority_sha256,
        "base_configuration_sha256": request.base_configuration_sha256,
        "count_model_config_sha256": request.count_model_config_sha256,
        "count_score_manifest_path": request.count_score_manifest_path,
        "count_score_manifest_sha256": request.count_score_manifest_sha256,
        "retained_calibration_path": request.retained_calibration_path,
        "retained_calibration_sha256": request.retained_calibration_sha256,
        "calibration_usage": request.calibration_usage,
        "calibration_context": (
            None
            if request.calibration_context is None
            else asdict(request.calibration_context)
        ),
        "timeout_seconds": request.timeout_seconds,
        "max_rss_bytes": request.max_rss_bytes,
        "max_gpu_bytes": request.max_gpu_bytes,
    }


def _replace_request_with_recomputed_digest(
    request: ExecutionRequest, **changes: object
) -> ExecutionRequest:
    changed = replace(request, **changes)
    return replace(
        changed,
        request_sha256=canonical_sha256(_request_integrity_values(changed)),
    )


def _truth_dataset(counts: np.ndarray) -> ad.AnnData:
    n_cells, n_genes = counts.shape
    cell_ids = [f"cell-{index + 1}" for index in range(n_cells)]
    gene_ids = [f"gene-{index + 1}" for index in range(n_genes)]
    libraries = counts.sum(axis=1, dtype=np.int64)
    dataset = ad.AnnData(
        X=np.asarray(counts, dtype=np.int64),
        obs=pd.DataFrame(
            {
                "dataset_id": ["dataset-test"] * n_cells,
                "mechanism": ["symsim"] * n_cells,
                "condition": ["moderate"] * n_cells,
                "biological_id": ["draw-01"] * n_cells,
                "technical_view": ["moderate"] * n_cells,
                "draw": np.ones(n_cells, dtype=np.int64),
                "library_size": libraries,
            },
            index=cell_ids,
        ),
        var=pd.DataFrame(index=gene_ids),
        layers={"pre_capture_counts": np.asarray(counts + 1, dtype=np.int64)},
    )
    dataset.uns.update(
        {
            "truth_kind": "exact_pre_capture",
            "primary_truth_layer": "pre_capture_counts",
            "allowed_covariates": {"obs": [], "var": []},
            "provenance": {
                "source": "test",
                "source_sha256": SHA_B,
                "software": "test",
                "software_version": "1",
                "parameters": {},
                "seeds": {},
            },
        }
    )
    return dataset


def _prepared_plan_inputs(bindings) -> tuple[PreparedDataset, ...]:
    values = []
    for ordinal, binding in enumerate(bindings, start=1):
        method_input = replace(
            _method_input(),
            source_dataset_sha256=binding.dataset_sha256,
        )
        evaluator_dataset = _truth_dataset(
            np.asarray([[ordinal, 0, 1], [0, ordinal + 1, 0]], dtype=np.int64)
        )
        draw_index = int(binding.biological_id.removeprefix("draw-"))
        evaluator_dataset.obs["draw"] = [draw_index, draw_index]
        evaluator_dataset.uns["provenance"]["seeds"] = {
            "biological": 10_000 + ordinal,
            "measurement": 20_000 + ordinal,
        }
        values.append(
            PreparedDataset(
                binding=binding,
                audit=DatasetQCAudit(
                    excluded_cell_count=0,
                    excluded_cell_ids_sha256="e" * 64,
                    retained_cell_count=2,
                    retained_cell_ids_sha256="f" * 64,
                    excluded_cell_ids=(),
                    retained_cell_ids=method_input.obs_ids,
                ),
                method_input=method_input,
                evaluator_dataset=evaluator_dataset,
            )
        )
    return tuple(values)


def _truth_probe_executor(request: ExecutionRequest) -> AdapterOutcome:
    try:
        getattr(request.method_input, "layers")
    except AttributeError:
        return AdapterOutcome.unavailable(
            "truth_not_in_executor_boundary",
            stdout=b"truth access rejected\n",
        )
    return AdapterOutcome.failed("truth_was_visible")


def _spawn_state_executor(_request: ExecutionRequest) -> AdapterOutcome:
    return AdapterOutcome.unavailable(
        "spawn_state_receipt",
        stdout=json.dumps(
            {"cwd": str(Path.cwd()), "sys_path": sys.path},
            sort_keys=True,
        ).encode("utf-8"),
    )


def _observed_executor(request: ExecutionRequest) -> AdapterOutcome:
    execution = run_observed(request.method_spec, request.method_input)
    return AdapterOutcome.completed(
        execution,
        runtime_seconds=0.01,
        peak_rss_bytes=1024,
        peak_gpu_bytes=0,
    )


def _slow_observed_executor(request: ExecutionRequest) -> AdapterOutcome:
    time.sleep(0.4)
    return _observed_executor(request)


def _slow_marker_executor(_request: ExecutionRequest) -> AdapterOutcome:
    time.sleep(0.4)
    return AdapterOutcome.unavailable(
        "child_completed",
        stdout=b"child-finished",
    )


def _maskimpute_result_executor(request: ExecutionRequest) -> AdapterOutcome:
    from maskimpute import ImputationResult
    from maskimpute.ablations import AblationRunResult
    from maskimpute_benchmark.methods import snapshot_method_output
    from maskimpute_benchmark.methods.maskimpute import MaskImputeAdapterExecution

    counts = request.method_input.counts
    result = ImputationResult(
        selective_counts=counts,
        denoised_counts=np.asarray(counts, dtype=np.float64),
        p_pre_zero=np.zeros(counts.shape, dtype=np.float64),
        latent=np.ones((counts.shape[0], 1), dtype=np.float64),
        diagnostics={"status": "spawned"},
    )
    snapshot = snapshot_method_output(
        request.method_spec,
        request.method_input,
        result.selective_counts,
        source_dataset_sha256=request.method_input.source_dataset_sha256,
        output_scale=request.method_spec.output_scale,
        obs_ids=request.method_input.obs_ids,
        var_ids=request.method_input.var_ids,
    )
    execution = MaskImputeAdapterExecution(
        snapshot=snapshot,
        compatibility_log=(),
        environment_receipt=(),
        stdout=b"",
        stderr=b"",
        command=None,
        ablation_result=AblationRunResult(
            output_policy="selective",
            _result=result,
        ),
    )
    return AdapterOutcome.completed(
        execution,
        runtime_seconds=0.01,
        peak_rss_bytes=1024,
        peak_gpu_bytes=0,
    )


class _FixedResourceSampler:
    def __init__(self, *, rss: int | None, gpu: int | None) -> None:
        self.rss = rss
        self.gpu = gpu

    def sample(self, process_id: int, *, gpu_required: bool) -> ResourceSample:
        return ResourceSample(
            peak_rss_bytes=self.rss,
            peak_gpu_bytes=self.gpu if gpu_required else 0,
            rss_provenance="mock_process_tree_rss",
            gpu_provenance=(
                "mock_process_tree_gpu" if gpu_required else "not_applicable_cpu"
            ),
        )


def test_development_manifest_requires_exact_canonical_sixteen_dataset_panel() -> None:
    bindings = validate_development_manifest_payload(_manifest_payload())

    assert len(bindings) == 16
    assert [
        (row.mechanism, row.biological_id, row.technical_view) for row in bindings
    ] == [
        (mechanism, f"draw-{draw:02d}", view)
        for mechanism in MECHANISMS
        for draw in (1, 2)
        for view in VIEWS
    ]
    assert all(isinstance(binding, DatasetBinding) for binding in bindings)

    incomplete = _manifest_payload()
    incomplete["rows"] = incomplete["rows"][:-1]  # type: ignore[index]
    with pytest.raises(RunnerContractError, match="exactly 16"):
        validate_development_manifest_payload(incomplete)

    reordered = _manifest_payload()
    reordered["rows"][0], reordered["rows"][1] = (  # type: ignore[index]
        reordered["rows"][1],
        reordered["rows"][0],
    )
    with pytest.raises(RunnerContractError, match="canonical order"):
        validate_development_manifest_payload(reordered)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("namespace", "final", "development namespace"),
        ("status", "failed", "complete"),
        ("completed_count", 15, "complete"),
        ("failed_count", 1, "complete"),
        ("independent_unit_count", 7, "eight independent"),
    ],
)
def test_development_manifest_rejects_nonfinalized_authority(
    field: str, replacement: object, message: str
) -> None:
    payload = _manifest_payload()
    payload[field] = replacement

    with pytest.raises(RunnerContractError, match=message):
        validate_development_manifest_payload(payload)


def test_direct_plan_is_full_denominator_with_fixed_seed_policy() -> None:
    registry = load_method_registry(METHODS_PATH)
    datasets = validate_development_manifest_payload(_manifest_payload())

    plan = build_fair_comparator_plan(
        registry,
        datasets,
        load_runner_authority(),
        _prepared_plan_inputs(datasets),
    )

    deterministic = {"observed"}
    expected_per_dataset = sum(
        1 if configuration.method.method_id in deterministic else 3
        for configuration in plan.configurations
    )
    expected = len(datasets) * expected_per_dataset
    assert len(plan.entries) == expected
    assert len({entry.run_id for entry in plan.entries}) == expected
    assert not {
        method.id
        for method in registry.methods
        if method.execution_scope != "same_input_required"
    } & {entry.identity.method_id for entry in plan.entries}
    for entry in plan.entries:
        identity = entry.identity
        expected_seeds = (
            (None,) if identity.method_id in deterministic else (42, 43, 44)
        )
        assert identity.model_seed in expected_seeds
        if identity.method_id == "maskimpute":
            assert identity.configuration_kind in {"candidate_search", "ablation"}
            assert entry.preflight_status == "blocked_authority"
            assert entry.preflight_reason in {
                "count_score_authority_pending",
                "count_score_or_calibration_authority_pending",
            }
        elif identity.method_id == "capacity-matched-ae":
            assert identity.configuration_id == "capacity-matched-ae"
            assert entry.preflight_status == "blocked_authority"
            assert (
                entry.preflight_reason == "count_score_or_calibration_authority_pending"
            )
        elif identity.method_id == "observed":
            assert identity.configuration_id == "registry-default"
            assert identity.configuration_kind == "registry"
            assert entry.preflight_status == "planned"
        else:
            assert identity.configuration_id != "registry-default"
            assert identity.configuration_kind == "comparator_tuning"
            assert entry.preflight_status == "planned"
    assert (
        sum(entry.identity.method_id == "capacity-matched-ae" for entry in plan.entries)
        == len(datasets) * 3
    )


def test_tracked_plan_has_exact_2896_rows_and_complete_comparator_blocks() -> None:
    authority = load_runner_authority()
    registry = load_method_registry(ROOT / "study/methods.json")
    bindings = validate_development_manifest_payload(_manifest_payload())
    plan = build_fair_comparator_plan(
        registry,
        bindings,
        authority,
        _prepared_plan_inputs(bindings),
    )
    assert len(plan.entries) == 2_896
    assert len({entry.run_id for entry in plan.entries}) == 2_896
    comparator_rows = [
        entry
        for entry in plan.entries
        if entry.identity.configuration_kind == "comparator_tuning"
    ]
    assert len(comparator_rows) == 1_632
    assert {entry.identity.method_id for entry in comparator_rows} == {
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
    }
    assert all(
        entry.identity.configuration_id != "registry-default"
        for entry in comparator_rows
    )
    for configuration in (
        value
        for value in plan.configurations
        if value.configuration_kind == "comparator_tuning"
    ):
        block = [
            entry
            for entry in comparator_rows
            if entry.identity.method_id == configuration.method.method_id
            and entry.identity.configuration_id == configuration.configuration_id
        ]
        assert len(block) == 48
        positions = [plan.entries.index(entry) for entry in block]
        assert positions == list(range(min(positions), min(positions) + 48))
    assert len(plan.configurations) == 61
    assert not (
        {"d3impute", "sctsi"} & {entry.identity.method_id for entry in plan.entries}
    )

    cursor = 0
    for configuration in plan.configurations:
        seeds = (
            (None,) if configuration.method.method_id == "observed" else (42, 43, 44)
        )
        expected_cells = [
            (binding.dataset_id, seed) for binding in bindings for seed in seeds
        ]
        block = plan.entries[cursor : cursor + len(expected_cells)]
        assert [
            (row.identity.dataset_id, row.identity.model_seed) for row in block
        ] == expected_cells
        assert {
            (row.identity.method_id, row.identity.configuration_id) for row in block
        } == {(configuration.method.method_id, configuration.configuration_id)}
        cursor += len(expected_cells)
    assert cursor == len(plan.entries)

    tuning = authority.comparator_tuning
    assert tuple(
        (row.method.method_id, row.configuration_id)
        for row in plan.configurations
        if row.configuration_kind == "comparator_tuning"
    ) == tuple((row.method_id, row.configuration_id) for row in tuning.configurations)
    for method_id in tuning.method_order:
        configured = tuning.configurations_for(method_id)
        assert configured[0].is_upstream_default


def test_runner_direct_dispatch_route_uses_closed_comparator_mapping(
    monkeypatch: pytest.MonkeyPatch,
    dispatcher_fixture: RepositoryAdapterDispatcher,
) -> None:
    captured: dict[str, object] = {}
    request = object()
    prepared = object()
    outcome = object()
    sentinel = object()

    def fake_validate(selected_request, selected_prepared, authority):
        captured.update(
            request=selected_request,
            prepared=selected_prepared,
            authority=authority,
        )
        return object()

    monkeypatch.setattr(
        "maskimpute_benchmark.fair_comparator_execution.validate_direct_request",
        fake_validate,
    )
    monkeypatch.setattr(
        runner_module,
        "execute_direct_adapter_in_spawned_process",
        lambda selected_request, executor, **options: (
            captured.update(
                spawned_request=selected_request,
                adapter_ids=tuple(executor.dispatcher.direct_comparator_adapters()),
                spawn_options=options,
            )
            or outcome
        ),
    )
    monkeypatch.setattr(
        "maskimpute_benchmark.fair_comparator_execution.evaluate_direct_outcome",
        lambda selected_request, selected_prepared, authority, selected_outcome: (
            captured.update(
                evaluated=(
                    selected_request,
                    selected_prepared,
                    authority,
                    selected_outcome,
                )
            )
            or sentinel
        ),
    )
    authority = dispatcher_fixture.comparator_tuning_authority
    assert authority is not None

    result = execute_fair_comparator_request(
        request,
        prepared,
        authority,
        dispatcher_fixture,
    )

    assert result is sentinel
    assert captured == {
        "request": request,
        "prepared": prepared,
        "authority": authority,
        "spawned_request": request,
        "adapter_ids": authority.method_order,
        "spawn_options": {
            "resource_sampler": captured["spawn_options"]["resource_sampler"],
            "expected_spawn_executable": (
                dispatcher_fixture.environments.benchmark_python
            ),
            "spawn_search_path": (
                dispatcher_fixture.environments.python_spawn_search_path
            ),
        },
        "evaluated": (request, prepared, authority, outcome),
    }


@pytest.mark.parametrize("stage", ("before", "after"))
def test_production_direct_dispatch_preserves_runtime_environment_failure(
    monkeypatch: pytest.MonkeyPatch,
    dispatcher_fixture: RepositoryAdapterDispatcher,
    stage: str,
) -> None:
    dispatcher = replace(dispatcher_fixture, monitor_runtime_changes=False)
    calls = 0

    def revalidate(_self, _method_id):
        nonlocal calls
        calls += 1
        if stage == "before" or calls == 2:
            raise RuntimeEnvironmentError(f"synthetic_{stage}_control_state")

    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "revalidate_control_state_for",
        revalidate,
    )
    spec = load_method_registry(METHODS_PATH).by_id("magic")
    method_input = _method_input()

    def completed(*_args, **_kwargs):
        output = finalize_direct_method_output(
            spec,
            method_input,
            method_input.counts,
            output_scale=spec.output_scale,
            obs_ids=method_input.obs_ids,
            var_ids=method_input.var_ids,
        )
        return DirectAdapterExecution(output=output, stdout=b"", stderr=b"")

    monkeypatch.setattr("maskimpute_benchmark.methods.run_magic_direct", completed)
    authority = dispatcher.comparator_tuning_authority
    assert authority is not None
    config = authority.configurations_for("magic")[0].decode()

    outcome = dispatcher.direct_comparator_adapters()["magic"](
        spec,
        method_input,
        seed=42,
        config=config,
    )

    assert outcome.status == "infrastructure_error"
    assert outcome.reason == f"synthetic_{stage}_control_state"


def test_legacy_comparator_configuration_and_dispatch_entry_points_reject_directly(
    dispatcher_fixture: RepositoryAdapterDispatcher,
) -> None:
    """Migration map: legacy bind/dispatch rejects; direct tests own all behavior.

    Relabel, authority drift, post-attempt mutation, and typed-config coverage live
    in test_fair_comparator_execution.py and test_priority_method_adapters.py.
    """

    registry = load_method_registry(METHODS_PATH)
    authority = load_comparator_tuning_authority(
        Path.cwd(), registry=registry, require_clean=False
    )
    spec = registry.by_id("magic")
    row = authority.configurations_for("magic")[0]
    bound = bind_comparator_configuration_identity(row, spec, authority)

    with pytest.raises(RunnerContractError, match="direct fair-comparator"):
        AuthorizedConfiguration.from_bound_comparator(bound)

    components = {
        "registry_method_sha256": "1" * 64,
        "tuning_authority_file_sha256": "2" * 64,
        "tuning_authority_payload_sha256": "3" * 64,
        "source_authority_sha256": "4" * 64,
        "runtime_lock_sha256": "5" * 64,
        "environment_registry_sha256": "6" * 64,
    }
    method_identity = canonical_sha256(
        {
            "schema": "maskimpute-comparator-configuration-method-identity-v1",
            "configuration_payload_sha256": canonical_sha256(dict(row.payload)),
            **components,
        }
    )
    with pytest.raises(RunnerContractError, match="direct fair-comparator"):
        AuthorizedConfiguration.create(
            method_id="magic",
            configuration_id=row.configuration_id,
            kind="comparator_tuning",
            payload=dict(row.payload),
            requires_count_score=False,
            requires_calibration=False,
            configuration_method_identity_sha256=method_identity,
            **components,
        )

    stale_request = type(
        "StaleComparatorRequest",
        (),
        {"configuration_kind": "comparator_tuning"},
    )()
    with pytest.raises(RunnerContractError, match="direct fair-comparator"):
        dispatcher_fixture._comparator_config(stale_request)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"execution_environment_sha256": "2" * 64},
        {"runtime_lock_sha256": "1" * 64},
    ),
)
def test_plan_building_requires_both_runtime_identity_hashes(kwargs) -> None:
    with pytest.raises(RunnerContractError, match="plan runtime identity is absent"):
        build_competition_plan(
            load_method_registry(ROOT / "study/methods.json"),
            validate_development_manifest_payload(_manifest_payload()),
            load_runner_authority(),
            **kwargs,
        )


def test_legacy_plan_builder_rejects_the_direct_fair_comparator_scope() -> None:
    with pytest.raises(RunnerContractError, match="build_fair_comparator_plan"):
        build_competition_plan(
            load_method_registry(ROOT / "study/methods.json"),
            validate_development_manifest_payload(_manifest_payload()),
            load_runner_authority(),
            execution_environment_sha256="2" * 64,
            runtime_lock_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    "loader",
    (
        runner_module.load_v28_revision_authority,
        runner_module.load_v29_revision_authority,
    ),
)
def test_revision_plan_contains_exactly_one_48_row_maskimpute_candidate(loader) -> None:
    authority = loader()
    registry = load_method_registry(ROOT / "study/methods.json")
    bindings = validate_development_manifest_payload(_manifest_payload())

    plan = build_competition_plan(
        registry,
        bindings,
        authority,
        execution_environment_sha256="2" * 64,
        runtime_lock_sha256="1" * 64,
    )

    assert authority.plan_scope == "revision_candidate_only"
    assert len(authority.configurations) == 1
    assert len(plan.configurations) == 1
    assert len(plan.entries) == 48
    assert {entry.method_id for entry in plan.entries} == {"maskimpute"}
    assert {entry.configuration_id for entry in plan.entries} == {
        authority.configurations[0].configuration_id
    }
    assert [(entry.dataset_id, entry.model_seed) for entry in plan.entries] == [
        (binding.dataset_id, seed) for binding in bindings for seed in (42, 43, 44)
    ]


def test_authority_derives_first_twenty_search_configs_and_excludes_budget_overruns() -> (
    None
):
    from maskimpute_benchmark.protocol import canonical_sha256

    rows = []
    for index in range(22):
        payload = {
            "method_version": "v27",
            "output_policy": "selective",
            "score_policy": (
                "direct_cross_fitted_count_score"
                if index == 0
                else "retained_calibrator"
            ),
            "hyperparameters": {
                "pre_zero_regularization": 1 if index < 10 else 2,
                "gate_gamma": index + 1,
            },
        }
        rows.append(
            {
                "configuration_id": f"search-{index + 1:02d}",
                "disposition": (
                    "authorized"
                    if index < 20
                    else "exploratory_budget_overrun_not_selection_eligible"
                ),
                "configuration": payload,
                "configuration_sha256": canonical_sha256(payload),
            }
        )
    ablations = (
        {
            "id": "maskimpute-reference",
            "output_policy": "selective",
            "score_source": "retained_calibrator",
        },
        {
            "id": "capacity-matched-ae",
            "output_policy": "full_denoising",
            "score_source": "not_applied",
        },
        {
            "id": "no-gate",
            "output_policy": "selective",
            "score_source": "direct",
        },
    )
    bindings = {spec["id"]: canonical_sha256(spec) for spec in ablations}

    configurations = derive_authorized_configurations(rows, ablations, bindings)

    search = [value for value in configurations if value.kind == "candidate_search"]
    assert len(search) == 20
    assert [value.configuration_id for value in search] == [
        f"search-{index + 1:02d}" for index in range(20)
    ]
    assert not any(
        "21" in value.configuration_id or "22" in value.configuration_id
        for value in configurations
    )
    assert (
        sum(value.method_id == "capacity-matched-ae" for value in configurations) == 1
    )
    assert not any(
        value.configuration_id == "maskimpute-reference" for value in configurations
    )
    assert any(value.configuration_id == "no-gate" for value in configurations)


def test_candidate_search_maps_direct_and_calibrated_policies_to_distinct_variants() -> (
    None
):
    direct = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="direct-config",
        kind="candidate_search",
        payload={"score_policy": "direct_cross_fitted_count_score"},
        requires_count_score=True,
        requires_calibration=False,
    )
    calibrated = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="calibrated-config",
        kind="candidate_search",
        payload={"score_policy": "retained_calibrator"},
        requires_count_score=True,
        requires_calibration=True,
    )

    assert maskimpute_variant_for_configuration(direct) == "direct-score"
    assert maskimpute_variant_for_configuration(calibrated) == "maskimpute-reference"
    with pytest.raises(RunnerContractError, match="score_policy"):
        maskimpute_variant_for_configuration(
            AuthorizedConfiguration.create(
                method_id="maskimpute",
                configuration_id="unknown-config",
                kind="candidate_search",
                payload={"score_policy": "unknown"},
                requires_count_score=True,
                requires_calibration=False,
            )
        )


def test_tracked_search_and_ablation_files_expand_to_full_runner_grid() -> None:
    from maskimpute_benchmark.protocol import canonical_sha256

    search = json.loads(Path("study/development_search.json").read_text())
    ablation = json.loads(Path("study/ablations.json").read_text())
    specs = (ablation["reference"], *ablation["variants"])
    bindings = {spec["id"]: canonical_sha256(spec) for spec in specs}

    configurations = derive_authorized_configurations(
        search["configurations"], specs, bindings
    )

    assert sum(value.kind == "candidate_search" for value in configurations) == 20
    assert sum(value.kind == "ablation" for value in configurations) == 6
    direct = next(
        value
        for value in configurations
        if value.configuration_id == "v27-c01-direct-r1-g1"
    )
    calibrated = next(
        value
        for value in configurations
        if value.configuration_id == "v27-c02-calibrated-r1-g0p5"
    )
    assert maskimpute_variant_for_configuration(direct) == "direct-score"
    assert maskimpute_variant_for_configuration(calibrated) == "maskimpute-reference"


def test_clean_publication_authority_loads_pending_revalidation_bindings() -> None:
    from maskimpute_benchmark.selection import load_publication_execution_authority

    authority = load_runner_authority()
    selection_authority = load_publication_execution_authority()

    assert len(authority.configurations) == 26
    assert (
        sum(value.kind == "candidate_search" for value in authority.configurations)
        == 20
    )
    assert authority.dataset_qc_policy == DatasetQCPolicy.fixed()
    assert authority.dataset_qc_policy_sha256 == DatasetQCPolicy.fixed().sha256
    assert authority.count_score_manifest_status == "pending"
    assert authority.count_score_manifest_sha256 is None
    assert authority.retained_calibration_status == "pending"
    assert authority.retained_calibration_sha256 is None
    assert authority.plan_scope == "base_full_panel"
    assert len(authority.comparator_tuning.configurations) == 34
    assert authority.comparator_tuning_reference == (
        selection_authority.comparator_tuning
    )
    assert authority.comparator_method_bindings == (
        selection_authority.comparator_method_bindings
    )
    assert not hasattr(authority, "comparator_tuning_file_sha256")
    assert not hasattr(authority, "comparator_tuning_payload_sha256")
    assert "study/comparator_tuning.json" not in selection_authority.file_sha256


def test_direct_plan_rejects_noncanonical_ready_maskimpute_authority() -> None:
    registry = load_method_registry(METHODS_PATH)
    datasets = validate_development_manifest_payload(_manifest_payload())

    prepared = _prepared_plan_inputs(datasets)
    authority = load_runner_authority()
    blocked = build_fair_comparator_plan(
        registry,
        datasets,
        authority,
        prepared,
    )
    ready = replace(
        authority,
        count_score_manifest_status="ready",
        count_score_manifest_sha256="8" * 64,
        retained_calibration_status="ready",
        retained_calibration_sha256="9" * 64,
    )

    assert any(
        entry.preflight_status == "blocked_authority" for entry in blocked.entries
    )
    with pytest.raises(RunnerContractError, match="fixed authority"):
        build_fair_comparator_plan(registry, datasets, ready, prepared)


def test_method_input_hash_binds_only_truth_free_snapshot_and_is_stable() -> None:
    method_input = _method_input()
    first = method_input_sha256(method_input)
    second = method_input_sha256(method_input)
    changed_counts = np.array(method_input.counts, copy=True)
    changed_counts[0, 0] += 1
    with_other_counts = replace(
        method_input,
        _count_bytes=np.asarray(changed_counts, dtype="<f8").tobytes(order="C"),
    )
    with_other_source = replace(method_input, source_dataset_sha256=SHA_B)

    assert first == second
    assert len(first) == 64
    assert method_input_sha256(with_other_counts) != first
    assert method_input_sha256(with_other_source) != first
    assert not hasattr(method_input, "layers")
    assert not hasattr(method_input, "uns")


def test_execution_request_rejects_comparator_nonexecution_identity() -> None:
    spec = load_method_registry(METHODS_PATH).by_id("magic")
    configuration = AuthorizedConfiguration.create(
        method_id="magic",
        configuration_id="magic-nonexecution",
        kind="comparator_nonexecution",
        payload={"reason": "declared_nonexecution"},
        requires_count_score=False,
        requires_calibration=False,
        nonexecution_identity_sha256="e" * 64,
    )

    serialized = configuration.to_dict()
    assert serialized["configuration_method_identity_sha256"] is None
    assert serialized["nonexecution_identity_sha256"] == "e" * 64
    assert all(
        serialized[field] is None
        for field in (
            "registry_method_sha256",
            "tuning_authority_file_sha256",
            "tuning_authority_payload_sha256",
            "source_authority_sha256",
            "runtime_lock_sha256",
            "environment_registry_sha256",
        )
    )

    with pytest.raises(RunnerContractError, match="nonexecution"):
        ExecutionRequest.create(
            spec,
            _method_input(),
            model_seed=42,
            configuration=configuration,
            authority=_authority(maskimpute_ready=True),
            mechanism="symsim",
            biological_id="draw-01",
            technical_view="moderate",
            dataset_id="dataset-test",
            timeout_seconds=5,
        )


def test_run_plan_entry_propagates_closed_configuration_identity_shape() -> None:
    legacy = RunPlanEntry(
        ordinal=1,
        run_id="run-observed-test",
        method_id="observed",
        dataset_id="dataset-test",
        source_dataset_sha256="a" * 64,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        model_seed=None,
        configuration_id="registry-default",
        configuration_sha256="b" * 64,
        preflight_status="planned",
        preflight_reason=None,
    )
    assert legacy.configuration_payload_sha256 == legacy.configuration_sha256
    assert legacy.configuration_method_identity_sha256 is None
    assert legacy.nonexecution_identity_sha256 is None

    tuning = replace(
        legacy,
        configuration_kind="comparator_tuning",
        configuration_method_identity_sha256="c" * 64,
    )
    assert tuning.configuration_payload_sha256 == "b" * 64
    assert tuning.configuration_method_identity_sha256 == "c" * 64
    assert tuning.nonexecution_identity_sha256 is None

    nonexecution = replace(
        legacy,
        configuration_kind="comparator_nonexecution",
        nonexecution_identity_sha256="d" * 64,
    )
    assert nonexecution.configuration_method_identity_sha256 is None
    assert nonexecution.nonexecution_identity_sha256 == "d" * 64

    with pytest.raises(RunnerContractError, match="method identity"):
        replace(
            legacy,
            configuration_kind="comparator_tuning",
            configuration_method_identity_sha256=None,
        )
    with pytest.raises(RunnerContractError, match="nonexecution identity"):
        replace(
            legacy,
            configuration_kind="comparator_nonexecution",
            nonexecution_identity_sha256=None,
        )


@pytest.mark.parametrize(
    ("kind", "identity_fields"),
    (
        pytest.param(
            "comparator_tuning",
            {"configuration_method_identity_sha256": "c" * 64},
            id="comparator-tuning",
        ),
        pytest.param(
            "comparator_nonexecution",
            {"nonexecution_identity_sha256": "d" * 64},
            id="comparator-nonexecution",
        ),
    ),
)
def test_run_plan_entry_requires_explicit_comparator_payload_checksum(
    kind: str, identity_fields: dict[str, str]
) -> None:
    with pytest.raises(RunnerContractError, match="explicit.*payload checksum"):
        RunPlanEntry(
            ordinal=1,
            run_id="run-magic-test",
            method_id="magic",
            dataset_id="dataset-test",
            source_dataset_sha256="a" * 64,
            mechanism="symsim",
            biological_id="draw-01",
            technical_view="moderate",
            model_seed=42,
            configuration_id="magic-configuration",
            configuration_sha256="b" * 64,
            preflight_status="planned",
            preflight_reason=None,
            configuration_kind=kind,
            configuration_payload_sha256=None,
            **identity_fields,
        )


@pytest.mark.parametrize(
    "identity_field",
    (
        "registry_method_sha256",
        "tuning_authority_file_sha256",
        "tuning_authority_payload_sha256",
        "source_authority_sha256",
        "runtime_lock_sha256",
        "environment_registry_sha256",
        "configuration_method_identity_sha256",
        "nonexecution_identity_sha256",
    ),
)
@pytest.mark.parametrize("kind", ("registry", "candidate_search", "ablation"))
def test_execution_request_legacy_kinds_reject_comparator_identity_fields(
    kind: str, identity_field: str
) -> None:
    with pytest.raises(RunnerContractError, match="identity"):
        AuthorizedConfiguration.create(
            method_id="observed",
            configuration_id=f"{kind.replace('_', '-')}-configuration",
            kind=kind,
            payload={"kind": kind},
            requires_count_score=False,
            requires_calibration=False,
            **{identity_field: "a" * 64},
        )


def test_execution_request_legacy_identity_integrity_remains_strict() -> None:
    registry = load_method_registry(METHODS_PATH)
    authority = _authority(maskimpute_ready=True)
    configurations = (
        AuthorizedConfiguration.registry_default(registry.by_id("observed")),
        next(
            value
            for value in authority.configurations
            if value.kind == "candidate_search"
        ),
        next(value for value in authority.configurations if value.kind == "ablation"),
    )
    for configuration in configurations:
        spec = registry.by_id(configuration.method_id)
        request = ExecutionRequest.create(
            spec,
            _method_input(),
            model_seed=42 if spec.stochastic else None,
            configuration=configuration,
            authority=authority,
            mechanism="symsim",
            biological_id="draw-01",
            technical_view="moderate",
            dataset_id="dataset-test",
            timeout_seconds=5,
        )
        assert request.configuration_kind == configuration.kind
        assert request.configuration_payload_sha256 == request.configuration_sha256
        assert request.configuration_method_identity_sha256 is None
        assert request.nonexecution_identity_sha256 is None
        request.validate_integrity()
        for mutation in (
            {"configuration_id": "forged-configuration"},
            {"configuration_sha256": "f" * 64},
            {"configuration_payload_json": "{}"},
            {"request_sha256": "f" * 64},
        ):
            with pytest.raises(RunnerContractError):
                replace(request, **mutation).validate_integrity()


def test_spawned_executor_receives_no_anndata_or_truth_slots() -> None:
    registry = load_method_registry(METHODS_PATH)
    spec = registry.by_id("observed")
    method_input = _method_input()
    authority = _authority(maskimpute_ready=True)
    configuration = AuthorizedConfiguration.registry_default(spec)
    request = ExecutionRequest.create(
        spec,
        method_input,
        model_seed=None,
        configuration=configuration,
        authority=authority,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    outcome = execute_adapter_in_spawned_process(
        request,
        _truth_probe_executor,
        poll_interval_seconds=0.01,
    )

    assert outcome.status == "unavailable"
    assert outcome.reason == "truth_not_in_executor_boundary"
    assert outcome.stdout == b"truth access rejected\n"


def test_spawned_executor_uses_bound_cwd_and_frozen_search_path() -> None:
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )
    frozen_path = runtime_module.publication_python_spawn_search_path()

    outcome = execute_adapter_in_spawned_process(
        request,
        _spawn_state_executor,
        poll_interval_seconds=0.01,
        spawn_search_path=frozen_path,
    )
    receipt = json.loads(outcome.stdout.decode("utf-8"))

    assert receipt == {
        "cwd": str(runtime_module.publication_runtime_working_directory()),
        "sys_path": list(frozen_path),
    }


def test_spawned_executor_rejects_lexical_executable_alias(tmp_path: Path) -> None:
    from multiprocessing import spawn

    alias = tmp_path / "python-alias"
    alias.symlink_to(sys.executable)
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )
    spawn.set_executable(str(alias))
    try:
        with pytest.raises(RunnerContractError, match="lexically"):
            execute_adapter_in_spawned_process(request, _observed_executor)
    finally:
        spawn.set_executable(sys.executable)


def test_spawned_executor_rejects_warning_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )
    monkeypatch.setattr(sys, "warnoptions", ["error"])

    with pytest.raises(RunnerContractError, match="nondefault Python flags"):
        execute_adapter_in_spawned_process(request, _observed_executor)


def test_spawned_executor_returns_a_bound_observed_snapshot() -> None:
    registry = load_method_registry(METHODS_PATH)
    spec = registry.by_id("observed")
    method_input = _method_input()
    authority = _authority(maskimpute_ready=True)
    configuration = AuthorizedConfiguration.registry_default(spec)
    request = ExecutionRequest.create(
        spec,
        method_input,
        model_seed=None,
        configuration=configuration,
        authority=authority,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    outcome = execute_adapter_in_spawned_process(
        request,
        _observed_executor,
        poll_interval_seconds=0.01,
    )

    assert outcome.status == "completed"
    assert isinstance(outcome.execution, AdapterExecution)
    assert outcome.execution.snapshot.source_dataset_sha256 == SHA_A
    np.testing.assert_array_equal(
        outcome.execution.snapshot.matrix, method_input.counts
    )


def test_spawned_executor_round_trips_maskimpute_result() -> None:
    spec = load_method_registry(METHODS_PATH).by_id("maskimpute")
    authority = _authority(maskimpute_ready=True)
    configuration = next(
        value
        for value in authority.configurations
        if value.configuration_id == "v27-c01-direct-r1-g1"
    )
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=42,
        configuration=configuration,
        authority=authority,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    outcome = execute_adapter_in_spawned_process(
        request,
        _maskimpute_result_executor,
        poll_interval_seconds=0.01,
        resource_sampler=_FixedResourceSampler(rss=123_456, gpu=0),
    )

    assert outcome.status == "completed"
    assert outcome.execution is not None
    ablation_result = outcome.execution.ablation_result
    np.testing.assert_array_equal(
        ablation_result.selective_counts,
        request.method_input.counts,
    )
    assert ablation_result.diagnostics == {"status": "spawned"}


def test_execution_environment_registry_binds_exact_runtime_lock(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "python-environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    python = environment / "bin/python"
    lock = build_runtime_environment_lock(
        {
            "afmf": ("python", python),
            "benchmark": ("python", python),
        }
    )
    lock_path = tmp_path / "runtime-lock.json"
    lock_path.write_text(
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        {"afmf": python},
        runtime_lock_path=lock_path,
        benchmark_python=python,
    )

    assert environments.runtime_lock_sha256 is not None
    assert environments.executable_for("afmf") == python.absolute()


def test_method_registry_derives_ready_external_reference_lock_only_ids() -> None:
    registry = load_method_registry(METHODS_PATH)

    assert runner_module.derive_lock_only_environment_ids(registry) == (
        "d3impute",
        "sctsi",
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"id": "magic"}, "overlap"),
        ({"id": "../unsafe"}, "invalid"),
    ),
)
def test_lock_only_derivation_rejects_malformed_or_overlapping_scope(
    replacement: dict[str, str],
    message: str,
) -> None:
    registry = load_method_registry(METHODS_PATH)
    external = replace(registry.by_id("d3impute"), **replacement)
    malformed = MethodRegistry(
        schema_version=registry.schema_version,
        methods=registry.methods + (external,),
    )

    with pytest.raises(RunnerContractError, match=message):
        runner_module.derive_lock_only_environment_ids(malformed)


def test_execution_environment_registry_retains_and_revalidates_lock_only_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = RuntimeEnvironmentSnapshot(
        identity_sha256="1" * 64,
        closure_paths_sha256="2" * 64,
        nvidia_smi_path=None,
        path_identities=(),
        watch_specs=(),
        control_file_sha256s=(),
    )
    runtime_lock = RuntimeEnvironmentLock(
        path=tmp_path / "runtime-lock.json",
        file_sha256="3" * 64,
        entries=tuple(
            RuntimeEnvironmentEntry(
                environment_id=environment_id,
                kind="python",
                inventory_json=b"{}",
                inventory_sha256=digest * 64,
            )
            for environment_id, digest in (
                ("benchmark", "4"),
                ("d3impute", "5"),
                ("sctsi", "6"),
            )
        ),
    )
    validation_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def validate(
        lock,
        declarations,
        *,
        r_library_paths=None,
        expected_closure_paths_sha256s=None,
        lock_only_environment_ids=(),
    ):
        validation_calls.append(
            (tuple(sorted(declarations)), tuple(lock_only_environment_ids))
        )
        return {
            "lock_file_sha256": lock.file_sha256,
            "environment_inventory_sha256s": (("benchmark", "4" * 64),),
            "lock_only_environment_inventory_sha256s": tuple(
                (environment_id, lock.by_id(environment_id).inventory_sha256)
                for environment_id in sorted(lock_only_environment_ids)
            ),
        }

    class NoopMonitor:
        def __init__(self, _watch_specs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def assert_unchanged(self) -> None:
            pass

    monkeypatch.setattr(
        runner_module, "load_runtime_environment_lock", lambda _path: runtime_lock
    )
    monkeypatch.setattr(
        runner_module,
        "runtime_environment_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        runner_module,
        "merge_runtime_environment_snapshots",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(runner_module, "RuntimeChangeMonitor", NoopMonitor)
    monkeypatch.setattr(
        runner_module, "verify_runtime_environment_snapshot", lambda _snapshot: None
    )
    monkeypatch.setattr(
        runner_module,
        "verify_runtime_environment_control_files",
        lambda _snapshot: None,
    )
    monkeypatch.setattr(runner_module, "validate_runtime_environment_lock", validate)

    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        runtime_lock_path=runtime_lock.path,
        benchmark_python=Path(sys.executable),
        lock_only_environment_ids=("sctsi", "d3impute"),
    )

    assert environments.lock_only_environment_ids == ("d3impute", "sctsi")
    assert environments.executable_for("d3impute") is None
    assert environments.executable_for("sctsi") is None
    assert validation_calls == [
        (("benchmark",), ("d3impute", "sctsi")),
    ]

    validation_calls.clear()
    environments.full_revalidate()

    assert validation_calls == [
        (("benchmark",), ("d3impute", "sctsi")),
    ]


def test_development_competition_forwards_registry_derived_lock_only_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.methods as methods_module

    registry = load_method_registry(METHODS_PATH)
    captured: dict[str, object] = {}
    expected_report = object()

    def fixed(*args, **kwargs):
        captured["fixed_args"] = args
        captured["fixed_kwargs"] = kwargs
        return type(
            "EnvironmentFixture",
            (),
            {
                "registry_sha256": "7" * 64,
                "runtime_lock_sha256": "8" * 64,
            },
        )()

    def build(*args, **kwargs):
        captured["plan_args"] = args
        captured["plan_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        runner_module,
        "ExecutionEnvironmentRegistry",
        type("RegistryFixture", (), {"fixed": staticmethod(fixed)}),
    )
    monkeypatch.setattr(
        runner_module,
        "load_prepared_development_panel",
        lambda _authority: ((), {}),
    )
    monkeypatch.setattr(
        methods_module,
        "load_method_registry",
        lambda _path: registry,
    )
    monkeypatch.setattr(runner_module, "build_competition_plan", build)
    monkeypatch.setattr(
        runner_module, "RepositoryAdapterDispatcher", lambda *_a: object()
    )
    monkeypatch.setattr(runner_module, "SpawnedRepositoryExecutor", lambda value: value)
    monkeypatch.setattr(runner_module, "CheckpointStore", lambda _path: object())
    monkeypatch.setattr(
        runner_module,
        "execute_competition_plan",
        lambda *_args: expected_report,
    )

    report = runner_module._run_competition_with_authority(
        tmp_path / "competition",
        _authority(maskimpute_ready=True),
        environment_overrides=None,
    )

    assert report is expected_report
    assert captured["fixed_kwargs"]["lock_only_environment_ids"] == (
        "d3impute",
        "sctsi",
    )
    assert captured["plan_kwargs"] == {
        "execution_environment_sha256": "7" * 64,
        "runtime_lock_sha256": "8" * 64,
    }


def test_development_competition_requires_runtime_lock_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.methods as methods_module

    registry = load_method_registry(METHODS_PATH)
    environments = type(
        "EnvironmentFixture",
        (),
        {"registry_sha256": "7" * 64, "runtime_lock_sha256": None},
    )()
    monkeypatch.setattr(
        runner_module,
        "ExecutionEnvironmentRegistry",
        type(
            "RegistryFixture",
            (),
            {"fixed": staticmethod(lambda *_args, **_kwargs: environments)},
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "load_prepared_development_panel",
        lambda _authority: ((), {}),
    )
    monkeypatch.setattr(
        methods_module,
        "load_method_registry",
        lambda _path: registry,
    )
    monkeypatch.setattr(
        runner_module,
        "build_competition_plan",
        lambda *_args, **_kwargs: pytest.fail("plan construction must not begin"),
    )

    with pytest.raises(
        RunnerContractError, match="development runtime lock checksum is absent"
    ):
        runner_module._run_competition_with_authority(
            tmp_path / "competition",
            _authority(maskimpute_ready=True),
            environment_overrides=None,
        )


def test_execution_environment_registry_rejects_runtime_drift(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "python-environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    python = environment / "bin/python"
    lock = build_runtime_environment_lock({"benchmark": ("python", python)})
    lock_path = tmp_path / "runtime-lock.json"
    lock_path.write_text(
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RunnerContractError, match="runtime IDs mismatch"):
        ExecutionEnvironmentRegistry.fixed(
            tmp_path,
            {"afmf": python},
            runtime_lock_path=lock_path,
            benchmark_python=python,
        )


def test_execution_environment_registry_revalidates_bytes_and_environment_per_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "python-environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    python = environment / "bin/python"
    lock = build_runtime_environment_lock(
        {
            "afmf": ("python", python),
            "benchmark": ("python", python),
        }
    )
    lock_path = tmp_path / "runtime-lock.json"
    lock_path.write_text(
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        {"afmf": python},
        runtime_lock_path=lock_path,
        benchmark_python=python,
    )
    environments.revalidate_for("afmf")
    with monkeypatch.context() as context:
        context.setenv("OPENBLAS_NUM_THREADS", "publication-runtime-drift")
        with pytest.raises(RunnerContractError, match="environment changed"):
            environments.revalidate_for("afmf")
    environments.revalidate_for("afmf")
    site_packages = next(
        path for path in (environment / "lib").glob("python*/site-packages")
    )
    (site_packages / "publication_shadow.py").write_text(
        "VALUE = 'runtime-drift'\n", encoding="utf-8"
    )
    with pytest.raises(RunnerContractError, match="runtime identity mismatch"):
        environments.revalidate_for("afmf")


def test_execution_environment_snapshot_reads_libc_environ() -> None:
    name = "MASKIMPUTE_PUTENV_ONLY_REGRESSION"
    os.environ.pop(name, None)
    before = runner_module._execution_environment_snapshot()
    try:
        os.putenv(name, "bound-by-libc-environ")
        assert name not in os.environ
        assert runner_module._execution_environment_snapshot() != before
    finally:
        os.unsetenv(name)


def test_registry_rejects_parent_python_search_path_drift() -> None:
    environments = ExecutionEnvironmentRegistry.fixed(Path.cwd())
    injected = next(
        path
        for path in (Path(sys.prefix) / "include", Path(sys.prefix) / "bin")
        if path.is_dir() and str(path.resolve()) not in sys.path
    )
    sys.path.insert(0, str(injected.resolve()))
    try:
        with pytest.raises(RunnerContractError, match="spawn search path changed"):
            environments.revalidate_control_state_for("observed")
    finally:
        sys.path.remove(str(injected.resolve()))


def test_per_row_control_revalidation_rehashes_snapshot_control_files(
    tmp_path: Path,
) -> None:
    control = tmp_path / "synthetic-control"
    control.write_bytes(b"before")
    base = ExecutionEnvironmentRegistry.fixed(tmp_path)
    snapshot = RuntimeEnvironmentSnapshot(
        identity_sha256="a" * 64,
        closure_paths_sha256="b" * 64,
        nvidia_smi_path=None,
        path_identities=(),
        watch_specs=(),
        control_file_sha256s=((str(control), hashlib.sha256(b"before").hexdigest()),),
    )
    environments = replace(base, runtime_snapshot=snapshot)

    environments.revalidate_control_state_for("observed")
    control.write_bytes(b"changed")

    with pytest.raises(RunnerContractError, match="runtime control file content"):
        environments.revalidate_control_state_for("observed")


def test_per_row_revalidation_does_not_rediscover_runtime_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "python-environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    python = environment / "bin/python"
    lock = build_runtime_environment_lock({"benchmark": ("python", python)})
    lock_path = tmp_path / "runtime-lock.json"
    lock_path.write_text(
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        runtime_lock_path=lock_path,
        benchmark_python=python,
    )
    dispatcher = RepositoryAdapterDispatcher(tmp_path, environments)
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("per-row runtime closure discovery")

    monkeypatch.setattr(
        runtime_module,
        "_with_native_dependency_roots",
        forbidden,
    )

    outcome = dispatcher(request)

    assert outcome.status == "completed"


def test_spawned_repository_executor_rejects_transient_runtime_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "python-environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    python = environment / "bin/python"
    lock = build_runtime_environment_lock({"benchmark": ("python", python)})
    lock_path = tmp_path / "runtime-lock.json"
    lock_path.write_text(
        json.dumps(lock, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environments = ExecutionEnvironmentRegistry.fixed(
        tmp_path,
        runtime_lock_path=lock_path,
        benchmark_python=python,
    )
    dispatcher = RepositoryAdapterDispatcher(tmp_path, environments)
    executor = SpawnedRepositoryExecutor(dispatcher)
    real_spawn = execute_adapter_in_spawned_process
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )
    displaced = environment / "bin/python.displaced"
    failures: list[BaseException] = []

    def swap_and_restore() -> None:
        try:
            time.sleep(0.1)
            python.rename(displaced)
            python.symlink_to("/bin/false")
            time.sleep(0.05)
            python.unlink()
            displaced.rename(python)
        except BaseException as error:  # pragma: no cover - thread handoff
            failures.append(error)

    mutation = threading.Thread(target=swap_and_restore)

    def spawn_slow(request: ExecutionRequest, _executor, **_kwargs) -> AdapterOutcome:
        mutation.start()
        return real_spawn(
            request,
            _slow_observed_executor,
            poll_interval_seconds=0.01,
            resource_sampler=_FixedResourceSampler(rss=1024, gpu=0),
        )

    monkeypatch.setattr(
        runner_module,
        "execute_adapter_in_spawned_process",
        spawn_slow,
    )
    try:
        with pytest.raises(
            RunnerContractError, match="runtime changed during execution"
        ):
            executor(request)
    finally:
        if mutation.ident is not None:
            mutation.join()
    assert failures == []


def test_dispatcher_revalidates_runtime_before_and_after_each_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path.cwd()
    environments = ExecutionEnvironmentRegistry.fixed(repository)
    calls: list[str] = []
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "revalidate_for",
        lambda self, method_id: calls.append(method_id),
    )
    dispatcher = RepositoryAdapterDispatcher(repository, environments)
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    outcome = dispatcher(request)

    assert outcome.status == "completed"
    assert calls == ["observed", "observed"]


def test_spawned_executor_uses_parent_sampled_resources_not_executor_claims() -> None:
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    outcome = execute_adapter_in_spawned_process(
        request,
        _observed_executor,
        poll_interval_seconds=0.01,
        resource_sampler=_FixedResourceSampler(rss=123_456, gpu=0),
    )

    assert outcome.peak_rss_bytes == 123_456
    assert outcome.peak_gpu_bytes == 0
    assert outcome.rss_measurement == "mock_process_tree_rss"
    assert outcome.gpu_measurement == "not_applicable_cpu"


@pytest.mark.parametrize(
    ("method_id", "rss", "gpu", "expected_reason"),
    (
        ("observed", 2, 0, "peak_rss_exceeded"),
        ("scvi", 1, 2, "peak_gpu_exceeded"),
    ),
)
def test_spawned_executor_terminates_live_process_at_resource_cap(
    method_id: str,
    rss: int,
    gpu: int,
    expected_reason: str,
) -> None:
    from dataclasses import replace

    base = load_method_registry(METHODS_PATH).by_id(method_id)
    resources = replace(
        base.resources,
        max_rss_gib=(1 / 1024**3 if expected_reason == "peak_rss_exceeded" else 48),
        max_gpu_gib=(1 / 1024**3 if expected_reason == "peak_gpu_exceeded" else 0),
    )
    spec = replace(base, resources=resources)
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=42 if spec.stochastic else None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    outcome = execute_adapter_in_spawned_process(
        request,
        _slow_marker_executor,
        poll_interval_seconds=0.01,
        resource_sampler=_FixedResourceSampler(rss=rss, gpu=gpu),
    )

    assert outcome.status == "resource_exceeded"
    assert outcome.reason == expected_reason
    assert outcome.stdout == b""
    assert outcome.peak_rss_bytes == rss
    assert outcome.peak_gpu_bytes == gpu


def test_gpu_sampler_uses_bound_absolute_nvidia_smi_after_path_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected" / "nvidia-smi"
    replacement = tmp_path / "replacement" / "nvidia-smi"
    selected.parent.mkdir()
    replacement.parent.mkdir()
    for executable in (selected, replacement):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    process_id = os.getpid()
    commands: list[list[str]] = []

    def completed(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{process_id}, 7\n",
            stderr="",
        )

    monkeypatch.setattr(runner_module.subprocess, "run", completed)
    monkeypatch.setattr(
        runner_module, "_linux_process_tree", lambda _root: {process_id}
    )
    monkeypatch.setenv("PATH", str(replacement.parent))

    sample = runner_module.LinuxProcessTreeResourceSampler(selected).sample(
        process_id, gpu_required=True
    )

    assert commands[0][0] == str(selected)
    assert sample.peak_gpu_bytes == 7 * 1024**2


def test_spawned_executor_passes_snapshot_bound_gpu_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    base = ExecutionEnvironmentRegistry.fixed(tmp_path)
    snapshot = RuntimeEnvironmentSnapshot(
        identity_sha256="a" * 64,
        closure_paths_sha256="b" * 64,
        nvidia_smi_path=str(nvidia_smi),
        path_identities=(),
        watch_specs=(),
        control_file_sha256s=(),
    )
    environments = replace(base, runtime_snapshot=snapshot)
    executor = SpawnedRepositoryExecutor(
        RepositoryAdapterDispatcher(tmp_path, environments)
    )
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )
    observed: list[object] = []

    def capture(_request, _dispatcher, *, resource_sampler, **_kwargs):
        observed.append(resource_sampler)
        return _observed_executor(_request)

    monkeypatch.setattr(runner_module, "execute_adapter_in_spawned_process", capture)

    executor(request)

    assert len(observed) == 1
    assert observed[0].nvidia_smi_path == nvidia_smi


def test_gpu_execution_fails_closed_when_independent_gpu_measurement_is_missing() -> (
    None
):
    spec = load_method_registry(METHODS_PATH).by_id("dca")
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=42,
        configuration=AuthorizedConfiguration.registry_default(spec),
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    outcome = execute_adapter_in_spawned_process(
        request,
        _truth_probe_executor,
        poll_interval_seconds=0.01,
        resource_sampler=_FixedResourceSampler(rss=123_456, gpu=None),
    )

    assert outcome.status == "infrastructure_error"
    assert outcome.reason == "resource_telemetry_unavailable"
    assert outcome.gpu_measurement == "mock_process_tree_gpu"


def test_calibrated_development_completion_requires_matching_lodo_fold_receipt() -> (
    None
):
    spec = load_method_registry(METHODS_PATH).by_id("maskimpute")
    configuration = next(
        value
        for value in _authority(maskimpute_ready=True).configurations
        if value.configuration_id == "v27-c02-calibrated-r1-g0p5"
    )
    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=42,
        configuration=configuration,
        authority=_authority(maskimpute_ready=True),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )
    observed_spec = load_method_registry(METHODS_PATH).by_id("observed")
    observed_execution = run_observed(observed_spec, request.method_input)
    # Rebind only the adapter snapshot to the requested method for this receipt test.
    from maskimpute_benchmark.methods import snapshot_method_output

    snapshot = snapshot_method_output(
        spec,
        request.method_input,
        request.method_input.counts,
        source_dataset_sha256=request.method_input.source_dataset_sha256,
        output_scale="raw_counts",
        obs_ids=request.method_input.obs_ids,
        var_ids=request.method_input.var_ids,
    )
    execution = replace(observed_execution, snapshot=snapshot)
    completed = AdapterOutcome.completed(
        execution,
        runtime_seconds=1,
        peak_rss_bytes=1,
        peak_gpu_bytes=1,
    )

    missing = enforce_calibration_fold_receipt(request, completed)
    assert missing.status == "failed"
    assert missing.reason == "lodo_calibration_fold_receipt_missing"

    assert request.calibration_context is not None
    receipt = CalibrationFoldReceipt(
        calibration_artifact_sha256=request.retained_calibration_sha256,
        calibration_context_sha256=request.calibration_context.sha256,
        mechanism="symsim",
        biological_id="draw-01",
        training_manifest_sha256s=("a" * 64, "b" * 64),
        held_out_manifest_sha256s=("c" * 64, "d" * 64),
        fold_calibrator_sha256="e" * 64,
    )
    accepted = enforce_calibration_fold_receipt(
        request,
        replace(completed, calibration_fold_receipt=receipt),
    )
    assert accepted.status == "completed"
    assert accepted.calibration_fold_receipt == receipt


def test_final_calibrated_request_uses_all_development_without_lodo_receipt() -> None:
    spec = load_method_registry(METHODS_PATH).by_id("maskimpute")
    authority = _authority(maskimpute_ready=True)
    configuration = next(
        value
        for value in authority.configurations
        if value.configuration_id == "v27-c02-calibrated-r1-g0p5"
    )

    request = ExecutionRequest.create(
        spec,
        _method_input(),
        model_seed=42,
        configuration=configuration,
        authority=authority,
        mechanism="symsim",
        biological_id="draw-03",
        technical_view="moderate",
        dataset_id="dataset-final",
        timeout_seconds=5,
        calibration_usage="retained_all_development",
    )

    assert request.calibration_usage == "retained_all_development"
    assert request.calibration_context is None
    request.validate_integrity()


def test_frozen_final_in_tree_preserves_selected_direct_score_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.methods.maskimpute as adapter

    registry = load_method_registry(Path("study/methods.json"))
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(adapter, "_run_in_tree", fake_run)
    observed = adapter.run_frozen_final_in_tree(
        registry.by_id("maskimpute"),
        object(),
        variant_id="direct-score",
        calibration_artifact=object(),
        seed=42,
        config=object(),
        count_model_config=object(),
        device="cpu",
        mechanism="symsim",
        biological_id="draw-03",
    )

    assert observed is not None
    assert captured["variant_id"] == "direct-score"
    assert captured["calibration_usage"] == "retained_all_development"


def test_frozen_final_in_tree_threads_selected_v29_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.methods.maskimpute as adapter

    registry = load_method_registry(Path("study/methods.json"))
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(adapter, "_run_in_tree", fake_run)
    decoder = object()
    structure = object()
    adapter.run_frozen_final_in_tree(
        registry.by_id("maskimpute"),
        object(),
        variant_id="maskimpute-reference",
        calibration_artifact=object(),
        seed=42,
        config=object(),
        count_model_config=object(),
        device="cpu",
        mechanism="symsim",
        biological_id="draw-03",
        decoder="negative_binomial",
        decoder_config=decoder,
        structure_config=structure,
    )

    assert captured["decoder_config"] is decoder
    assert captured["structure_config"] is structure
    assert captured["calibration_usage"] == "retained_all_development"


def test_frozen_final_in_tree_calls_an_execution_path_that_accepts_final_usage() -> (
    None
):
    import inspect

    from maskimpute_benchmark.methods.maskimpute import _run_in_tree

    assert "calibration_usage" in inspect.signature(_run_in_tree).parameters


def test_spawned_repository_executor_close_releases_monitor_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Monitor:
        def __init__(self) -> None:
            self.close_count = 0

        def assert_unchanged(self) -> None:
            return None

        def close(self) -> None:
            self.close_count += 1

    monitor = Monitor()
    environments = ExecutionEnvironmentRegistry.fixed(tmp_path)
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "change_monitor",
        lambda _self: monitor,
    )
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "full_revalidate",
        lambda _self: None,
    )
    executor = SpawnedRepositoryExecutor(
        RepositoryAdapterDispatcher(tmp_path, environments)
    )

    executor.close()
    executor.close()

    assert monitor.close_count == 1
    with pytest.raises(RunnerContractError, match="closed"):
        executor(None)  # type: ignore[arg-type]


def test_spawned_repository_executor_closes_monitor_if_initialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.runner as runner

    class Monitor:
        def __init__(self) -> None:
            self.close_count = 0

        def assert_unchanged(self) -> None:
            return None

        def close(self) -> None:
            self.close_count += 1

    monitor = Monitor()
    environments = ExecutionEnvironmentRegistry.fixed(tmp_path)
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "change_monitor",
        lambda _self: monitor,
    )
    monkeypatch.setattr(
        ExecutionEnvironmentRegistry,
        "full_revalidate",
        lambda _self: None,
    )

    def fail_sampler(*_args, **_kwargs):
        raise RuntimeError("sampler failed")

    monkeypatch.setattr(runner, "LinuxProcessTreeResourceSampler", fail_sampler)
    dispatcher = RepositoryAdapterDispatcher(tmp_path.resolve(), environments)

    with pytest.raises(RuntimeError, match="sampler failed"):
        runner.SpawnedRepositoryExecutor(dispatcher)
    assert monitor.close_count == 1


def test_qc_excludes_only_zero_library_cells_before_one_shared_method_input() -> None:
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset = _truth_dataset(np.array([[0, 0], [1, 0], [0, 2]], dtype=np.int64))
    binding = replace(
        validate_development_manifest_payload(_manifest_payload())[0],
        cells=3,
        genes=2,
        dataset_id="dataset-test",
        dataset_sha256=benchmark_dataset_sha256(dataset),
    )

    prepared = prepare_dataset_for_execution(
        dataset,
        binding,
        DatasetQCPolicy.fixed(),
    )

    assert prepared.method_input.source_dataset_sha256 == binding.dataset_sha256
    assert prepared.method_input.obs_ids == ("cell-2", "cell-3")
    assert prepared.method_input.var_ids == ("gene-1", "gene-2")
    assert prepared.audit.excluded_cell_count == 1
    assert prepared.audit.retained_cell_count == 2
    assert prepared.audit.excluded_cell_ids == ("cell-1",)
    assert prepared.audit.retained_cell_ids == ("cell-2", "cell-3")
    assert prepared.evaluator_dataset.obs_names.tolist() == ["cell-2", "cell-3"]
    assert prepared.evaluator_dataset.var_names.tolist() == ["gene-1", "gene-2"]


def test_qc_refuses_nonzero_exclusion_gene_filtering_and_too_few_cells() -> None:
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset = _truth_dataset(np.array([[0, 0], [1, 0]], dtype=np.int64))
    binding = replace(
        validate_development_manifest_payload(_manifest_payload())[0],
        cells=2,
        genes=2,
        dataset_id="dataset-test",
        dataset_sha256=benchmark_dataset_sha256(dataset),
    )
    with pytest.raises(RunnerContractError, match="at least two retained cells"):
        prepare_dataset_for_execution(dataset, binding, DatasetQCPolicy.fixed())

    with pytest.raises(RunnerContractError, match="fixed publication rule"):
        prepare_dataset_for_execution(
            _truth_dataset(np.array([[1, 0], [0, 2]], dtype=np.int64)),
            replace(binding, dataset_sha256=SHA_A),
            replace(DatasetQCPolicy.fixed(), gene_filtering="allowed"),
        )


def test_pair_qc_uses_union_of_zero_library_cells_for_both_views() -> None:
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    moderate = _truth_dataset(
        np.array([[1, 0], [2, 0], [0, 3], [1, 1]], dtype=np.int64)
    )
    severe = _truth_dataset(np.array([[1, 0], [0, 0], [0, 0], [1, 1]], dtype=np.int64))
    shared_truth = np.array([[2, 1], [3, 1], [1, 4], [2, 2]], dtype=np.int64)
    moderate.layers["pre_capture_counts"] = shared_truth.copy()
    severe.layers["pre_capture_counts"] = shared_truth.copy()
    severe.obs["dataset_id"] = "dataset-test-severe"
    severe.obs["condition"] = "severe"
    severe.obs["technical_view"] = "severe"
    bindings = validate_development_manifest_payload(_manifest_payload())[:2]
    moderate_binding = replace(
        bindings[0],
        cells=4,
        genes=2,
        dataset_id="dataset-test",
        dataset_sha256=benchmark_dataset_sha256(moderate),
        truth_sha256="f" * 64,
    )
    severe_binding = replace(
        bindings[1],
        cells=4,
        genes=2,
        dataset_id="dataset-test-severe",
        dataset_sha256=benchmark_dataset_sha256(severe),
        truth_sha256="f" * 64,
    )

    prepared_moderate, prepared_severe = prepare_dataset_pair_for_execution(
        moderate,
        severe,
        moderate_binding,
        severe_binding,
        DatasetQCPolicy.fixed(),
    )

    assert prepared_moderate.audit.excluded_cell_ids == ("cell-2", "cell-3")
    assert prepared_severe.audit.excluded_cell_ids == ("cell-2", "cell-3")
    assert prepared_moderate.audit.retained_cell_ids == ("cell-1", "cell-4")
    assert prepared_severe.audit.retained_cell_ids == ("cell-1", "cell-4")
    assert (
        prepared_moderate.method_input.obs_ids == prepared_severe.method_input.obs_ids
    )
    assert (
        prepared_moderate.audit.excluded_cell_ids_sha256
        == prepared_severe.audit.excluded_cell_ids_sha256
    )


def _prepared_truth_dataset():
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset = _truth_dataset(np.array([[1, 0], [0, 2], [3, 1]], dtype=np.int64))
    binding = replace(
        validate_development_manifest_payload(_manifest_payload())[0],
        cells=3,
        genes=2,
        dataset_id="dataset-test",
        dataset_sha256=benchmark_dataset_sha256(dataset),
    )
    return prepare_dataset_for_execution(dataset, binding, DatasetQCPolicy.fixed())


def _direct_magic_checkpoint_case(prepared: PreparedDataset):
    registry = load_method_registry(METHODS_PATH)
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    prepared.evaluator_dataset.uns["provenance"]["seeds"] = {"measurement": 20_001}
    descriptor = describe_prepared_input(prepared)
    spec = registry.by_id("magic")
    method = comparator_method_binding(spec)
    row = authority.configurations_for("magic")[0]

    def freeze(value: object) -> object:
        if isinstance(value, dict):
            return tuple((key, freeze(nested)) for key, nested in sorted(value.items()))
        if isinstance(value, list):
            return tuple(freeze(nested) for nested in value)
        return value

    configuration = DirectAuthorizedConfiguration(
        method=method,
        configuration_id=row.configuration_id,
        configuration_kind="comparator_tuning",
        payload=freeze(dict(row.payload)),
        requires_count_score=False,
        requires_calibration=False,
    )
    identity = ComparatorRunIdentity(
        workflow_schema="maskimpute-fair-comparator-run-v1",
        authority_revision=authority.authority_revision,
        ordinal=1,
        method=method,
        configuration_id=configuration.configuration_id,
        configuration_kind=configuration.configuration_kind,
        configuration_payload=configuration.payload,
        dataset_id=prepared.binding.dataset_id,
        mechanism=prepared.binding.mechanism,
        biological_id=prepared.binding.biological_id,
        technical_view=prepared.binding.technical_view,
        mask_seed=descriptor.mask_seed,
        model_seed=42,
        draw_index=1,
    )
    entry = DirectPlanEntry(
        run_id=direct_run_id(identity),
        identity=identity,
        preflight_status="planned",
        preflight_reason=None,
        requires_count_score=False,
        requires_calibration=False,
    )
    plan = DirectCompetitionPlan(
        schema_version=1,
        identity_mode="direct-v1",
        authority_revision=authority.authority_revision,
        inputs=(descriptor,),
        entries=(entry,),
        configurations=(configuration,),
    )
    return plan, registry, {prepared.binding.dataset_id: prepared}


def _direct_magic_record(
    plan: DirectCompetitionPlan,
    status: str,
) -> dict[str, object]:
    entry = plan.entries[0]
    identity = plan.to_dict()["entries"][0]["identity"]
    assert isinstance(identity, dict)
    reason = None if status == "completed" else f"synthetic_{status}"
    return {
        "run": {
            "run_id": entry.run_id,
            "identity": identity,
            "status": status,
            "reason": reason,
            "runtime_seconds": 1,
            "peak_rss_bytes": 1,
            "peak_gpu_bytes": 0,
            "rss_measurement": "synthetic_parent_rss",
            "gpu_measurement": "not_applicable_cpu",
            "excluded_cell_count": 0,
            "excluded_cell_ids": [],
            "retained_cell_count": 3,
            "retained_cell_ids": ["cell-1", "cell-2", "cell-3"],
            "retained_gene_count": 2,
            "observed_zero_count": 2,
            "stdout": {
                "stream": "stdout",
                "original_byte_count": 0,
                "capture_policy": "discard_content",
                "terminal_reason": reason,
            },
            "stderr": {
                "stream": "stderr",
                "original_byte_count": 0,
                "capture_policy": "discard_content",
                "terminal_reason": reason,
            },
        },
        "metrics": [
            {
                "identity": identity,
                "metric": metric,
                "value": 0.0 if status == "completed" else None,
                "n": 1 if status == "completed" else 0,
                "status": status,
                "reason": reason,
            }
            for metric in DIRECT_RECONSTRUCTION_METRICS
        ],
        "p_pre_zero_evidence": {
            "applicable": False,
            "status": "not_applicable",
            "reason": "method_does_not_emit_p_pre_zero",
            "shape": None,
            "dtype": None,
            "encoding": None,
            "path": None,
            "compressed_byte_count": 0,
        },
    }


def _direct_magic_attempt(plan: DirectCompetitionPlan) -> DirectEvaluatedAttempt:
    entry = plan.entries[0]
    reason = "synthetic_unavailable"
    return DirectEvaluatedAttempt(
        run=DirectRunResult(
            run_id=entry.run_id,
            identity=entry.identity,
            status="unavailable",
            reason=reason,
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=0,
            rss_measurement="synthetic_parent_rss",
            gpu_measurement="not_applicable_cpu",
            excluded_cell_count=0,
            excluded_cell_ids=(),
            retained_cell_count=3,
            retained_cell_ids=("cell-1", "cell-2", "cell-3"),
            retained_gene_count=2,
            observed_zero_count=2,
            stdout=DirectLogReceipt(
                stream="stdout",
                original_byte_count=0,
                capture_policy="discard_content",
                terminal_reason=reason,
            ),
            stderr=DirectLogReceipt(
                stream="stderr",
                original_byte_count=0,
                capture_policy="discard_content",
                terminal_reason=reason,
            ),
        ),
        metrics=tuple(
            DirectMetricRow(
                identity=entry.identity,
                metric=metric,
                value=None,
                n=0,
                status="unavailable",
                reason=reason,
            )
            for metric in DIRECT_RECONSTRUCTION_METRICS
        ),
        native_output=None,
        native_output_scale=None,
        evaluator_output=None,
        p_pre_zero_evidence=DirectPreZeroEvidence(
            applicable=False,
            status="not_applicable",
            reason="method_does_not_emit_p_pre_zero",
            shape=None,
            dtype=None,
            encoding=None,
            path=None,
            compressed_byte_count=0,
        ),
    )


def test_fair_comparator_plan_execution_uses_only_direct_checkpoint_and_fake_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    calls = []

    def fake_attempt(entry, supplied_prepared, decision):
        calls.append((entry, supplied_prepared, decision))
        assert decision.authorized
        return _direct_magic_attempt(plan)

    monkeypatch.setattr(
        runner_module,
        "CheckpointStore",
        lambda *_args, **_kwargs: pytest.fail("legacy checkpoint route used"),
    )
    report = _execute_fair_comparator_plan_structural(
        plan,
        registry,
        prepared_datasets,
        fake_attempt,
        DirectCheckpointStore(tmp_path / "direct-checkpoint.json"),
    )

    assert report.identity_mode == "direct-v1"
    assert report.status == "completed"
    assert report.plan_snapshot == plan.to_dict()
    assert len(report.records) == 1
    assert len(calls) == 1


def test_development_base_entry_routes_to_fair_comparator_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    captured = {}

    def direct_base(output_dir, authority, *, environment_overrides):
        captured.update(
            output_dir=output_dir,
            authority=authority,
            environment_overrides=environment_overrides,
        )
        return sentinel

    monkeypatch.setattr(
        runner_module,
        "_run_fair_comparator_base_with_authority",
        direct_base,
    )
    monkeypatch.setattr(
        runner_module,
        "_run_competition_with_authority",
        lambda *_args, **_kwargs: pytest.fail("legacy base route used"),
    )
    result = runner_module.run_development_competition(tmp_path / "competition")

    assert result is sentinel
    assert captured["output_dir"] == tmp_path / "competition"
    assert captured["authority"].plan_scope == "base_full_panel"
    assert captured["environment_overrides"] is None


def _entry_for(prepared, method_id: str, *, seed: int | None) -> RunPlanEntry:
    spec = load_method_registry(METHODS_PATH).by_id(method_id)
    configuration = AuthorizedConfiguration.registry_default(spec)
    return RunPlanEntry(
        ordinal=1,
        run_id=f"run-{method_id}-test",
        method_id=method_id,
        dataset_id=prepared.binding.dataset_id,
        source_dataset_sha256=prepared.binding.dataset_sha256,
        mechanism=prepared.binding.mechanism,
        biological_id=prepared.binding.biological_id,
        technical_view=prepared.binding.technical_view,
        model_seed=seed,
        configuration_id="registry-default",
        configuration_sha256=configuration.configuration_sha256,
        preflight_status="planned",
        preflight_reason=None,
    )


def test_completed_output_uses_common_log2_cp10k_and_complete_long_form_metrics() -> (
    None
):
    from maskimpute_benchmark.methods import count_equivalent_to_log2_cp10k
    from maskimpute_benchmark.metrics import reconstruction_metrics

    prepared = _prepared_truth_dataset()
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    execution = run_observed(spec, prepared.method_input)
    outcome = AdapterOutcome.completed(
        execution,
        runtime_seconds=1.25,
        peak_rss_bytes=2048,
        peak_gpu_bytes=0,
    )

    evaluated = evaluate_adapter_outcome(
        _entry_for(prepared, "observed", seed=None),
        prepared,
        outcome,
    )

    expected_output = count_equivalent_to_log2_cp10k(prepared.method_input.counts)
    truth = count_equivalent_to_log2_cp10k(
        np.asarray(prepared.evaluator_dataset.layers["pre_capture_counts"])
    )
    expected = reconstruction_metrics(
        expected_output,
        expected_output,
        truth,
        truth_kind="exact_pre_capture",
    )
    assert evaluated.run.status == "completed"
    assert evaluated.run.method_input_sha256 == method_input_sha256(
        prepared.method_input
    )
    assert evaluated.run.retained_cell_ids_sha256 == (
        prepared.audit.retained_cell_ids_sha256
    )
    np.testing.assert_allclose(evaluated.evaluator_output, expected_output)
    assert len(evaluated.metrics) == len(expected) == 36
    assert [metric.metric for metric in evaluated.metrics] == list(expected)
    mse = next(metric for metric in evaluated.metrics if metric.metric == "mse")
    assert mse.status == "completed"
    assert mse.value == pytest.approx(expected["mse"].value)
    assert mse.n == expected["mse"].n


def test_evaluator_conversion_failure_retains_only_stable_hashed_detail() -> None:
    prepared = _prepared_truth_dataset()
    spec = load_method_registry(METHODS_PATH).by_id("observed")
    execution = run_observed(spec, prepared.method_input)
    unsafe_detail = "private /tmp/worker-928/token=not-for-publication"

    def invalid_converter(method_input, adapter_execution):
        raise ValueError(unsafe_detail)

    evaluated = evaluate_adapter_outcome(
        _entry_for(prepared, "observed", seed=None),
        prepared,
        AdapterOutcome.completed(
            execution,
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=0,
        ),
        output_converter=invalid_converter,
    )

    detail_sha256 = hashlib.sha256(
        b"maskimpute-evaluator-conversion-detail-v1\0" + unsafe_detail.encode()
    ).hexdigest()
    expected = f"evaluator_conversion_valueerror_detail_{detail_sha256}"
    assert evaluated.run.status == "unavailable"
    assert evaluated.run.reason == expected
    assert unsafe_detail not in evaluated.run.reason
    assert all(metric.reason == expected for metric in evaluated.metrics)


@pytest.mark.parametrize("method_id", ["maskimpute", "capacity-matched-ae"])
def test_in_tree_adapter_contracts_require_score_evidence_before_shared_conversion(
    method_id: str,
) -> None:
    from maskimpute_benchmark.methods import snapshot_method_output

    prepared = _prepared_truth_dataset()
    spec = load_method_registry(METHODS_PATH).by_id(method_id)
    snapshot = snapshot_method_output(
        spec,
        prepared.method_input,
        prepared.method_input.counts,
        source_dataset_sha256=prepared.method_input.source_dataset_sha256,
        output_scale="raw_counts",
        obs_ids=prepared.method_input.obs_ids,
        var_ids=prepared.method_input.var_ids,
    )
    execution = AdapterExecution(
        snapshot=snapshot,
        compatibility_log=(),
        environment_receipt=(),
        stdout=b"",
        stderr=b"",
        command=None,
    )

    evaluation = lambda: evaluate_adapter_outcome(  # noqa: E731 - assertion closure
        _entry_for(prepared, method_id, seed=42),
        prepared,
        AdapterOutcome.completed(
            execution,
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=1,
        ),
    )

    if method_id == "maskimpute":
        with pytest.raises(RunnerContractError, match="realized p_pre_zero"):
            evaluation()
        return
    evaluated = evaluation()

    assert evaluated.run.status == "completed"
    assert evaluated.evaluator_output is not None


@pytest.mark.parametrize(
    ("outcome", "expected_status", "reason"),
    [
        (
            AdapterOutcome.unavailable("missing_environment"),
            "unavailable",
            "missing_environment",
        ),
        (AdapterOutcome.failed("upstream_error"), "failed", "upstream_error"),
        (AdapterOutcome.timeout(), "timeout", "timeout"),
        (
            AdapterOutcome.resource_exceeded("peak_rss_exceeded"),
            "resource_exceeded",
            "peak_rss_exceeded",
        ),
    ],
)
def test_noncompleted_attempts_keep_complete_reason_coded_metric_denominator(
    outcome: AdapterOutcome, expected_status: str, reason: str
) -> None:
    prepared = _prepared_truth_dataset()

    evaluated = evaluate_adapter_outcome(
        _entry_for(prepared, "observed", seed=None),
        prepared,
        outcome,
    )

    assert evaluated.run.status == expected_status
    assert evaluated.run.reason == reason
    assert len(evaluated.metrics) == 36
    assert all(metric.value is None for metric in evaluated.metrics)
    assert all(metric.status == expected_status for metric in evaluated.metrics)
    assert all(metric.reason == reason for metric in evaluated.metrics)
    assert evaluated.evaluator_output is None


@pytest.fixture
def magic_spec():
    return load_method_registry(METHODS_PATH).by_id("magic")


@pytest.fixture
def completed_outcome() -> AdapterOutcome:
    method_input = _method_input()
    observed = load_method_registry(METHODS_PATH).by_id("observed")
    return AdapterOutcome.completed(
        run_observed(observed, method_input),
        runtime_seconds=1,
        peak_rss_bytes=1,
        peak_gpu_bytes=0,
    )


def test_comparator_configs_share_method_budget_and_restore_exactly(
    magic_spec, completed_outcome
) -> None:
    budget = DevelopmentBudget()
    hashes = tuple(f"{value:064x}" for value in range(1, 5))
    for digest in hashes:
        assert budget.authorize(magic_spec, digest).authorized
        budget.record(magic_spec, digest, completed_outcome)
    restored = DevelopmentBudget()
    for digest in hashes:
        restored.restore(
            magic_spec,
            digest,
            "completed",
            completed_outcome.runtime_seconds,
        )
    assert restored.to_dict() == budget.to_dict()


def test_comparator_tuning_configs_share_method_budget_configuration_limit(
    magic_spec, completed_outcome
) -> None:
    budget = DirectDevelopmentBudget()
    for value in range(1, runner_module.MAX_DEVELOPMENT_CONFIGURATIONS + 1):
        configuration_id = f"magic-t{value:02d}"
        assert budget.authorize(
            magic_spec,
            configuration_id,
            budget_scope="magic",
        ).authorized
        budget.restore(
            magic_spec,
            configuration_id,
            completed_outcome.status,
            completed_outcome.runtime_seconds,
            budget_scope="magic",
        )

    assert not budget.authorize(
        magic_spec,
        "magic-excess",
        budget_scope="magic",
    ).authorized


def test_budget_caps_configurations_and_runtime_and_counts_failures() -> None:
    registry = load_method_registry(METHODS_PATH)
    gpu_spec = registry.by_id("maskimpute")
    cpu_spec = registry.by_id("alra")
    budget = DevelopmentBudget()

    for index in range(20):
        configuration_sha256 = f"{index + 1:064x}"
        decision = budget.authorize(gpu_spec, configuration_sha256)
        assert decision.authorized
        budget.record(
            gpu_spec,
            configuration_sha256,
            AdapterOutcome.failed("model_failure", runtime_seconds=60),
        )
    assert not budget.authorize(gpu_spec, "f" * 64).authorized
    assert budget.authorize(gpu_spec, f"{1:064x}").authorized

    budget.record(
        gpu_spec,
        f"{1:064x}",
        AdapterOutcome.failed("model_failure", runtime_seconds=8 * 60 * 60 - 20 * 60),
    )
    assert not budget.authorize(gpu_spec, f"{1:064x}").authorized

    infrastructure_only = DevelopmentBudget()
    infrastructure_only.record(
        gpu_spec,
        SHA_A,
        AdapterOutcome.infrastructure_error(
            "scheduler_unavailable", runtime_seconds=100
        ),
    )
    decision = infrastructure_only.authorize(gpu_spec, SHA_B)
    assert decision.authorized
    assert decision.remaining_seconds == 8 * 60 * 60

    cpu_budget = DevelopmentBudget()
    cpu_budget.record(
        cpu_spec,
        SHA_A,
        AdapterOutcome.unavailable("dependency_failure", runtime_seconds=24 * 60 * 60),
    )
    assert not cpu_budget.authorize(cpu_spec, SHA_A).authorized


class _RecordingUnavailableExecutor:
    def __init__(self) -> None:
        self.input_object_ids: list[int] = []
        self.input_hashes: list[str] = []

    def __call__(self, request: ExecutionRequest) -> AdapterOutcome:
        self.input_object_ids.append(id(request.method_input))
        self.input_hashes.append(request.method_input_sha256)
        return AdapterOutcome.unavailable("adapter_not_configured", runtime_seconds=1)


class _InterruptSecondExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: ExecutionRequest) -> AdapterOutcome:
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return AdapterOutcome.unavailable("adapter_not_configured", runtime_seconds=1)


def _competition_plan_body(plan: CompetitionPlan) -> dict[str, object]:
    return {
        "schema_version": plan.schema_version,
        "input_hashes": dict(plan.input_hashes),
        "entries": [entry.to_dict() for entry in plan.entries],
        "configurations": [
            configuration.to_dict() for configuration in plan.configurations
        ],
        "execution_context": (
            None if plan.execution_context is None else asdict(plan.execution_context)
        ),
        "budgets": {
            "maximum_configurations": runner_module.MAX_DEVELOPMENT_CONFIGURATIONS,
            "gpu_seconds": runner_module.MAX_GPU_BUDGET_SECONDS,
            "cpu_seconds": runner_module.MAX_CPU_BUDGET_SECONDS,
            "failures_consume_budget_except": "infrastructure_error",
        },
    }


def _rehash_competition_plan(plan: CompetitionPlan) -> CompetitionPlan:
    return replace(plan, plan_sha256=canonical_sha256(_competition_plan_body(plan)))


def _write_implementation_source_fixture(root: Path, *, reverse: bool = False) -> None:
    files = {
        "maskimpute/a.py": b"first = 1\n",
        "maskimpute/nested/z.py": b"last = 2\n",
        "maskimpute_benchmark/runner.py": b"runner = 3\n",
        "scripts/run_development_competition.py": b"main = 4\n",
    }
    names = tuple(reversed(files)) if reverse else tuple(files)
    for relative in names:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[relative])


def _two_method_plan(prepared) -> CompetitionPlan:
    registry = load_method_registry(METHODS_PATH)
    observed = registry.by_id("observed")
    observed_configuration = AuthorizedConfiguration.registry_default(observed)
    capacity = registry.by_id("capacity-matched-ae")
    capacity_configuration = AuthorizedConfiguration.create(
        method_id=capacity.id,
        configuration_id="capacity-test",
        kind="candidate_search",
        payload={"fixture": "legacy-non-comparator"},
        requires_count_score=False,
        requires_calibration=False,
    )
    entries = (
        _entry_for(prepared, "observed", seed=None),
        replace(
            _entry_for(prepared, "capacity-matched-ae", seed=42),
            ordinal=2,
            run_id="run-capacity-matched-ae-test",
            configuration_id=capacity_configuration.configuration_id,
            configuration_sha256=capacity_configuration.configuration_sha256,
            configuration_kind=capacity_configuration.kind,
            configuration_payload_sha256=(capacity_configuration.configuration_sha256),
        ),
    )
    return _rehash_competition_plan(
        CompetitionPlan(
            schema_version=1,
            input_hashes={
                "dataset_manifest_sha256": prepared.binding.manifest_sha256,
                "method_registry_sha256": SHA_B,
                "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
                "implementation_source_sha256": implementation_source_sha256(),
            },
            entries=entries,
            plan_sha256="0" * 64,
            configurations=(observed_configuration, capacity_configuration),
        )
    )


def _single_configuration_plan(
    prepared,
    entry: RunPlanEntry,
    configuration: AuthorizedConfiguration,
) -> CompetitionPlan:
    return _rehash_competition_plan(
        CompetitionPlan(
            schema_version=1,
            input_hashes={
                "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
                "implementation_source_sha256": implementation_source_sha256(),
            },
            entries=(entry,),
            plan_sha256="0" * 64,
            configurations=(configuration,),
        )
    )


def _comparator_plan_entry(
    prepared,
    configuration: AuthorizedConfiguration,
    *,
    preflight_status: str,
    preflight_reason: str | None,
    configuration_method_identity_sha256: str | None,
    nonexecution_identity_sha256: str | None,
) -> RunPlanEntry:
    return replace(
        _entry_for(prepared, "magic", seed=42),
        configuration_id=configuration.configuration_id,
        configuration_sha256=configuration.configuration_sha256,
        configuration_kind=configuration.kind,
        configuration_payload_sha256=configuration.configuration_sha256,
        configuration_method_identity_sha256=(configuration_method_identity_sha256),
        nonexecution_identity_sha256=nonexecution_identity_sha256,
        preflight_status=preflight_status,
        preflight_reason=preflight_reason,
    )


@pytest.fixture
def completed_checkpoint_fixture(tmp_path: Path):
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store._write_structural(
        plan,
        (_direct_magic_record(plan, "unavailable"),),
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    return store.path, plan, prepared_datasets


def test_direct_checkpoint_rejects_coherent_budget_tamper(
    completed_checkpoint_fixture,
) -> None:
    checkpoint_path, plan, prepared = completed_checkpoint_fixture
    payload = json.loads(checkpoint_path.read_text())
    payload["budget"]["magic"]["consumed_seconds"] += 1
    checkpoint_path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RunnerContractError, match="budget ledger differs from replay"):
        DirectCheckpointStore(checkpoint_path)._load_structural(
            plan,
            registry=load_method_registry(METHODS_PATH),
            prepared_datasets=prepared,
        )


def test_incomplete_grid_is_unselectable_until_comparator_is_terminal(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    incomplete = store._write_structural(
        plan,
        (),
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    assert incomplete.comparator_selection_status == "blocked_incomplete_denominator"

    complete = store._write_structural(
        plan,
        (_direct_magic_record(plan, "unavailable"),),
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    assert complete.comparator_selection_status == "complete_terminal_denominator"


@pytest.mark.parametrize(
    "outcome",
    (
        AdapterOutcome.budget_exhausted("configuration_budget_exhausted"),
        AdapterOutcome.blocked_authority("comparator_authority_missing"),
        AdapterOutcome.infrastructure_error("scheduler_unavailable"),
    ),
)
def test_incomplete_grid_blocking_statuses_remain_unselectable(
    tmp_path: Path,
    outcome: AdapterOutcome,
) -> None:
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    report = DirectCheckpointStore(
        tmp_path / f"{outcome.status}.json"
    )._write_structural(
        plan,
        (_direct_magic_record(plan, outcome.status),),
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    assert report.comparator_selection_status == "blocked_incomplete_denominator"


@pytest.mark.parametrize(
    "outcome",
    (
        AdapterOutcome.failed("model_failure", runtime_seconds=1),
        AdapterOutcome.timeout(runtime_seconds=1),
        AdapterOutcome.resource_exceeded("peak_rss_exceeded", runtime_seconds=1),
        AdapterOutcome.unavailable("dependency_unavailable", runtime_seconds=1),
    ),
)
def test_comparator_selection_intrinsic_terminal_statuses_complete_grid(
    tmp_path: Path,
    outcome: AdapterOutcome,
) -> None:
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    report = DirectCheckpointStore(
        tmp_path / f"{outcome.status}.json"
    )._write_structural(
        plan,
        (_direct_magic_record(plan, outcome.status),),
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    assert report.comparator_selection_status == "complete_terminal_denominator"


def test_persisted_infrastructure_error_is_not_selectively_retried(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    first = store._write_structural(
        plan,
        (_direct_magic_record(plan, "infrastructure_error"),),
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    assert first.comparator_selection_status == "blocked_incomplete_denominator"
    resumed = store._load_structural(
        plan,
        registry=registry,
        prepared_datasets=prepared_datasets,
    )
    assert resumed == first


def test_registry_default_comparator_request_is_rejected(tmp_path: Path) -> None:
    prepared = _prepared_truth_dataset()
    spec = load_method_registry(METHODS_PATH).by_id("magic")
    configuration = AuthorizedConfiguration.registry_default(spec)
    entry = _entry_for(prepared, "magic", seed=42)
    plan = _single_configuration_plan(prepared, entry, configuration)
    executor = _RecordingUnavailableExecutor()

    with pytest.raises(
        RunnerContractError,
        match="publication comparator cannot use registry-default",
    ):
        execute_competition_plan(
            plan,
            load_method_registry(METHODS_PATH),
            {prepared.binding.dataset_id: prepared},
            executor,
            CheckpointStore(tmp_path / "competition"),
        )

    assert executor.input_hashes == []


def test_execute_fair_comparator_plan_rejects_replaced_direct_run_identity(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    stale = replace(
        plan,
        entries=(replace(plan.entries[0], run_id="run-magic-replaced"),),
    )

    def executor_must_not_run(*_args):
        raise AssertionError("executor called before direct plan validation")

    with pytest.raises(RunnerContractError, match="direct checkpoint plan run ID"):
        _execute_fair_comparator_plan_structural(
            stale,
            registry,
            prepared_datasets,
            executor_must_not_run,
            DirectCheckpointStore(tmp_path / "direct-checkpoint.json"),
        )

    assert not (tmp_path / "direct-checkpoint.json").exists()


def test_execute_fair_comparator_plan_rejects_substituted_method_projection(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    plan, registry, prepared_datasets = _direct_magic_checkpoint_case(prepared)
    changed_method = replace(
        plan.entries[0].identity.method,
        adapter_key="substituted-adapter",
    )
    changed_identity = replace(plan.entries[0].identity, method=changed_method)
    changed_entry = replace(plan.entries[0], identity=changed_identity)
    changed_plan = replace(plan, entries=(changed_entry,))

    def executor_must_not_run(*_args):
        raise AssertionError("executor called before direct plan validation")

    with pytest.raises(RunnerContractError, match="method projection differs"):
        _execute_fair_comparator_plan_structural(
            changed_plan,
            registry,
            prepared_datasets,
            executor_must_not_run,
            DirectCheckpointStore(tmp_path / "direct-checkpoint.json"),
        )

    assert not (tmp_path / "direct-checkpoint.json").exists()


def test_execute_competition_plan_rejects_substituted_blocked_nonexecution_identity(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    configuration = AuthorizedConfiguration.create(
        method_id="magic",
        configuration_id="magic-nonexecution",
        kind="comparator_nonexecution",
        payload={"reason": "declared_nonexecution"},
        requires_count_score=False,
        requires_calibration=False,
        nonexecution_identity_sha256="e" * 64,
    )
    entry = _comparator_plan_entry(
        prepared,
        configuration,
        preflight_status="blocked_authority",
        preflight_reason="declared_nonexecution",
        configuration_method_identity_sha256=None,
        nonexecution_identity_sha256="f" * 64,
    )
    plan = _single_configuration_plan(prepared, entry, configuration)
    executor = _RecordingUnavailableExecutor()

    with pytest.raises(RunnerContractError, match="configuration identity mismatch"):
        execute_competition_plan(
            plan,
            load_method_registry(METHODS_PATH),
            {prepared.binding.dataset_id: prepared},
            executor,
            CheckpointStore(tmp_path / "competition"),
        )

    assert executor.input_hashes == []


def test_execution_reuses_one_truth_free_input_and_checkpoints_full_denominator(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    plan = _two_method_plan(prepared)
    executor = _RecordingUnavailableExecutor()
    store = CheckpointStore(tmp_path / "competition")

    report = execute_competition_plan(
        plan,
        load_method_registry(METHODS_PATH),
        {prepared.binding.dataset_id: prepared},
        executor,
        store,
    )

    assert report.status == "completed"
    assert report.evaluation_scope == "reconstruction_only"
    assert report.selection_complete is False
    assert set(report.selection_blockers) == {
        "null_de_fpr_not_evaluated",
        "orthogonal_endpoints_not_evaluated",
        "downstream_safety_not_evaluated",
    }
    assert len(report.records) == 2
    assert executor.input_object_ids[0] == executor.input_object_ids[1]
    assert executor.input_hashes == [
        method_input_sha256(prepared.method_input),
        method_input_sha256(prepared.method_input),
    ]
    assert all(record["run"]["status"] == "unavailable" for record in report.records)
    checkpoint_bytes = store.checkpoint_path.read_bytes()
    assert checkpoint_bytes.endswith(b"\n")
    loaded = store.load(
        plan,
        registry=load_method_registry(METHODS_PATH),
        prepared_datasets={prepared.binding.dataset_id: prepared},
    )
    assert loaded == report
    for record in loaded.records:
        stdout = store.output_dir / record["run"]["stdout_path"]
        stderr = store.output_dir / record["run"]["stderr_path"]
        assert stdout.read_bytes() == b""
        assert stderr.read_bytes() == b""


def test_resume_requires_exact_plan_and_continues_only_after_valid_prefix(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    plan = _two_method_plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    interrupting = _InterruptSecondExecutor()

    with pytest.raises(KeyboardInterrupt):
        execute_competition_plan(
            plan,
            load_method_registry(METHODS_PATH),
            {prepared.binding.dataset_id: prepared},
            interrupting,
            store,
        )
    partial = store.load(
        plan,
        registry=load_method_registry(METHODS_PATH),
        prepared_datasets={prepared.binding.dataset_id: prepared},
    )
    assert partial.status == "running"
    assert len(partial.records) == 1

    resumed_executor = _RecordingUnavailableExecutor()
    completed = execute_competition_plan(
        plan,
        load_method_registry(METHODS_PATH),
        {prepared.binding.dataset_id: prepared},
        resumed_executor,
        store,
    )
    assert completed.status == "completed"
    assert len(completed.records) == 2
    assert len(resumed_executor.input_hashes) == 1

    changed = replace(plan, plan_sha256="d" * 64)
    with pytest.raises(RunnerContractError, match="plan checksum"):
        store.load(
            changed,
            registry=load_method_registry(METHODS_PATH),
            prepared_datasets={prepared.binding.dataset_id: prepared},
        )


def test_checkpoint_resume_rejects_changed_implementation_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    source_root = tmp_path / "source"
    _write_implementation_source_fixture(source_root)
    source_sha256 = implementation_source_sha256(source_root)
    original_plan = _two_method_plan(prepared)
    plan = _rehash_competition_plan(
        replace(
            original_plan,
            input_hashes={
                **original_plan.input_hashes,
                "implementation_source_sha256": source_sha256,
            },
        )
    )
    store = CheckpointStore(tmp_path / "competition")
    monkeypatch.setattr(
        runner_module,
        "implementation_source_sha256",
        lambda repository_root=None: implementation_source_sha256(source_root),
    )
    execute_competition_plan(
        plan,
        load_method_registry(METHODS_PATH),
        {prepared.binding.dataset_id: prepared},
        _RecordingUnavailableExecutor(),
        store,
    )

    changed = source_root / "maskimpute/a.py"
    changed.write_bytes(changed.read_bytes() + b"# changed after checkpoint\n")
    with pytest.raises(RunnerContractError, match="implementation source"):
        store.load(
            plan,
            registry=load_method_registry(METHODS_PATH),
            prepared_datasets={prepared.binding.dataset_id: prepared},
        )


def test_checkpoint_revalidates_bound_raw_logs_and_common_output(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    entry = _entry_for(prepared, "observed", seed=None)
    configuration = AuthorizedConfiguration.registry_default(
        load_method_registry(METHODS_PATH).by_id("observed")
    )
    plan = _rehash_competition_plan(
        CompetitionPlan(
            schema_version=1,
            input_hashes={
                "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
                "implementation_source_sha256": implementation_source_sha256(),
            },
            entries=(entry,),
            plan_sha256="0" * 64,
            configurations=(configuration,),
        )
    )
    store = CheckpointStore(tmp_path / "competition")

    report = execute_competition_plan(
        plan,
        load_method_registry(METHODS_PATH),
        {prepared.binding.dataset_id: prepared},
        _observed_executor,
        store,
    )
    run = report.records[0]["run"]
    assert run["evaluator_scale"] == "log2_cp10k_plus_1"
    output = store.output_dir / run["evaluator_output_path"]
    assert output.stat().st_size == prepared.method_input.counts.size * 8
    assert (
        store.load(
            plan,
            registry=load_method_registry(METHODS_PATH),
            prepared_datasets={prepared.binding.dataset_id: prepared},
        )
        == report
    )

    stdout = store.output_dir / run["stdout_path"]
    stdout.write_bytes(b"tampered")
    with pytest.raises(RunnerContractError, match="stdout.*checksum"):
        store.load(
            plan,
            registry=load_method_registry(METHODS_PATH),
            prepared_datasets={prepared.binding.dataset_id: prepared},
        )


def test_checkpoint_loader_rejects_symlink_replacement(tmp_path: Path) -> None:
    prepared = _prepared_truth_dataset()
    plan = _two_method_plan(prepared)
    store = CheckpointStore(tmp_path / "competition")
    execute_competition_plan(
        plan,
        load_method_registry(METHODS_PATH),
        {prepared.binding.dataset_id: prepared},
        _RecordingUnavailableExecutor(),
        store,
    )
    target = tmp_path / "forged.json"
    target.write_bytes(store.checkpoint_path.read_bytes())
    store.checkpoint_path.unlink()
    store.checkpoint_path.symlink_to(target)

    with pytest.raises(RunnerContractError, match="checkpoint.*regular file|symlink"):
        store.load(
            plan,
            registry=load_method_registry(METHODS_PATH),
            prepared_datasets={prepared.binding.dataset_id: prepared},
        )


def test_implementation_source_digest_is_sorted_raw_and_symlink_safe(
    tmp_path: Path,
) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    _write_implementation_source_fixture(roots[0])
    _write_implementation_source_fixture(roots[1], reverse=True)

    first = implementation_source_sha256(roots[0])
    assert implementation_source_sha256(roots[1]) == first
    ignored = roots[0] / "scripts/not-an-execution-entrypoint.py"
    ignored.write_bytes(b"ignored = True\n")
    assert implementation_source_sha256(roots[0]) == first
    changed = roots[1] / "maskimpute/nested/z.py"
    changed.write_bytes(changed.read_bytes() + b"# byte change\n")
    assert implementation_source_sha256(roots[1]) != first

    linked = roots[0] / "maskimpute/linked.py"
    linked.symlink_to(tmp_path / "outside.py")
    with pytest.raises(RunnerContractError, match="symlink"):
        implementation_source_sha256(roots[0])
    linked.unlink()
    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (roots[0] / "maskimpute/linked-directory").symlink_to(
        outside_directory, target_is_directory=True
    )
    with pytest.raises(RunnerContractError, match="directory.*symlink"):
        implementation_source_sha256(roots[0])

    entrypoint_root = tmp_path / "linked-entrypoint-root"
    _write_implementation_source_fixture(entrypoint_root)
    entrypoint = entrypoint_root / "scripts/run_development_competition.py"
    entrypoint.unlink()
    entrypoint.parent.rmdir()
    outside_scripts = tmp_path / "outside-scripts"
    outside_scripts.mkdir()
    (outside_scripts / "run_development_competition.py").write_bytes(
        b"outside = True\n"
    )
    (entrypoint_root / "scripts").symlink_to(outside_scripts, target_is_directory=True)
    with pytest.raises(RunnerContractError, match="directory.*symlink"):
        implementation_source_sha256(entrypoint_root)


def test_public_cli_exposes_only_operational_output_and_environment_paths() -> None:
    completed = subprocess.run(
        [
            str(Path(sys.executable)),
            "scripts/run_development_competition.py",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--output-dir" in completed.stdout
    assert "--environment" in completed.stdout
    for forbidden in ("--seed", "--config", "--mechanism", "--budget", "--metric"):
        assert forbidden not in completed.stdout
