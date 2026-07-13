from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.runner as runner_module
from maskimpute_benchmark.methods import (
    AdapterExecution,
    MethodInput,
    load_method_registry,
    run_observed,
)
from maskimpute_benchmark.runner import (
    AdapterOutcome,
    AuthorizedConfiguration,
    DevelopmentBudget,
    DatasetBinding,
    ExecutionRequest,
    CheckpointStore,
    CalibrationFoldReceipt,
    CompetitionPlan,
    RunPlanEntry,
    RunnerAuthority,
    RunnerContractError,
    DatasetQCPolicy,
    ResourceSample,
    ExecutionEnvironmentRegistry,
    RepositoryAdapterDispatcher,
    prepare_dataset_pair_for_execution,
    prepare_dataset_for_execution,
    build_competition_plan,
    execute_adapter_in_spawned_process,
    evaluate_adapter_outcome,
    enforce_calibration_fold_receipt,
    execute_competition_plan,
    derive_authorized_configurations,
    method_input_sha256,
    implementation_source_sha256,
    maskimpute_variant_for_configuration,
    load_runner_authority,
    validate_development_manifest_payload,
)


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

    configurations = (
        AuthorizedConfiguration.create(
            method_id="maskimpute",
            configuration_id="v27-reference",
            kind="candidate_search",
            payload={"method_version": "v27", "score_policy": "direct"},
            requires_count_score=True,
            requires_calibration=False,
        ),
        AuthorizedConfiguration.create(
            method_id="maskimpute",
            configuration_id="calibrated-score",
            kind="ablation",
            payload={
                "changed_component": "score",
                "score_source": "retained_calibrator",
            },
            requires_count_score=True,
            requires_calibration=True,
        ),
        AuthorizedConfiguration.create(
            method_id="capacity-matched-ae",
            configuration_id="capacity-matched-ae",
            kind="ablation",
            payload={"changed_component": "masking", "score_source": "not_applied"},
            requires_count_score=False,
            requires_calibration=False,
        ),
    )
    return RunnerAuthority(
        schema_version=1,
        authority_sha256="1" * 64,
        method_registry_sha256="2" * 64,
        selection_contract_sha256="3" * 64,
        development_search_sha256="4" * 64,
        ablation_registry_sha256="5" * 64,
        base_configuration_id="v27-reference",
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
        configurations=configurations,
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


def _truth_probe_executor(request: ExecutionRequest) -> AdapterOutcome:
    try:
        getattr(request.method_input, "layers")
    except AttributeError:
        return AdapterOutcome.unavailable(
            "truth_not_in_executor_boundary",
            stdout=b"truth access rejected\n",
        )
    return AdapterOutcome.failed("truth_was_visible")


def _observed_executor(request: ExecutionRequest) -> AdapterOutcome:
    execution = run_observed(request.method_spec, request.method_input)
    return AdapterOutcome.completed(
        execution,
        runtime_seconds=0.01,
        peak_rss_bytes=1024,
        peak_gpu_bytes=0,
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


def test_plan_is_full_denominator_with_fixed_seed_policy_and_bound_hashes() -> None:
    registry = load_method_registry(METHODS_PATH)
    datasets = validate_development_manifest_payload(_manifest_payload())

    plan = build_competition_plan(registry, datasets, _authority())

    deterministic = {"observed", "d3impute", "sctsi"}
    ordinary = {
        method.id
        for method in registry.methods
        if method.id not in {"maskimpute", "capacity-matched-ae"}
    }
    expected_per_dataset = (
        sum(1 if method_id in deterministic else 3 for method_id in ordinary) + 3 * 3
    )
    expected = len(datasets) * expected_per_dataset
    assert len(plan.entries) == expected
    assert len({entry.run_id for entry in plan.entries}) == expected
    assert plan.input_hashes["dataset_manifest_sha256"] == SHA_A
    assert plan.input_hashes["method_registry_sha256"] == "2" * 64
    assert plan.input_hashes["implementation_source_sha256"] == (
        implementation_source_sha256()
    )
    for entry in plan.entries:
        expected_seeds = (None,) if entry.method_id in deterministic else (42, 43, 44)
        assert entry.model_seed in expected_seeds
        if entry.method_id == "maskimpute":
            assert entry.configuration_id in {"v27-reference", "calibrated-score"}
            assert entry.preflight_status == "blocked_authority"
            assert entry.preflight_reason in {
                "count_score_authority_pending",
                "count_score_or_calibration_authority_pending",
            }
        elif entry.method_id == "capacity-matched-ae":
            assert entry.configuration_id == "capacity-matched-ae"
            assert entry.preflight_status == "planned"
        else:
            assert entry.configuration_id == "registry-default"
            assert entry.preflight_status == "planned"
    assert (
        sum(entry.method_id == "capacity-matched-ae" for entry in plan.entries)
        == len(datasets) * 3
    )


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


def test_clean_publication_authority_loads_exact_dynamic_runner_grid() -> None:
    authority = load_runner_authority()

    assert len(authority.configurations) == 26
    assert (
        sum(value.kind == "candidate_search" for value in authority.configurations)
        == 20
    )
    assert authority.dataset_qc_policy == DatasetQCPolicy.fixed()
    assert authority.dataset_qc_policy_sha256 == DatasetQCPolicy.fixed().sha256
    assert authority.retained_calibration_status == "ready"


def test_ready_maskimpute_authority_removes_only_its_preflight_block() -> None:
    registry = load_method_registry(METHODS_PATH)
    datasets = validate_development_manifest_payload(_manifest_payload())

    blocked = build_competition_plan(registry, datasets, _authority())
    ready = build_competition_plan(
        registry, datasets, _authority(maskimpute_ready=True)
    )

    assert all(
        entry.preflight_status == "planned"
        for entry in ready.entries
        if entry.method_id == "maskimpute"
    )
    assert [
        replace(entry, preflight_status="planned", preflight_reason=None)
        if entry.method_id == "maskimpute"
        else entry
        for entry in blocked.entries
    ] == list(ready.entries)
    assert blocked.plan_sha256 != ready.plan_sha256


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
        if value.configuration_id == "v27-reference"
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


def test_repository_dispatcher_runs_observed_and_reason_codes_missing_environments() -> (
    None
):
    repository = Path.cwd()
    registry = load_method_registry(METHODS_PATH)
    environments = ExecutionEnvironmentRegistry.fixed(repository)
    dispatcher = RepositoryAdapterDispatcher(repository, environments)
    authority = _authority(maskimpute_ready=True)
    observed = registry.by_id("observed")
    observed_request = ExecutionRequest.create(
        observed,
        _method_input(),
        model_seed=None,
        configuration=AuthorizedConfiguration.registry_default(observed),
        authority=authority,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )

    completed = dispatcher(observed_request)

    assert completed.status == "completed"
    assert completed.execution is not None
    assert "scvi" in dispatcher.supported_method_ids
    assert "magic" in dispatcher.supported_method_ids

    magic = registry.by_id("magic")
    missing_request = ExecutionRequest.create(
        magic,
        _method_input(),
        model_seed=42,
        configuration=AuthorizedConfiguration.registry_default(magic),
        authority=authority,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        timeout_seconds=5,
    )
    unavailable = dispatcher(missing_request)
    assert unavailable.status == "unavailable"
    assert unavailable.reason == "environment_executable_unavailable:magic"
    assert environments.executable_for("scvi") == (
        repository / "artifacts/envs/scvi-py312/bin/python"
    ).resolve(strict=True)


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
        if value.configuration_id == "calibrated-score"
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
    assert len(evaluated.metrics) == len(expected) == 35
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
    expected = f"evaluator_conversion:ValueError:detail_sha256={detail_sha256}"
    assert evaluated.run.status == "unavailable"
    assert evaluated.run.reason == expected
    assert unsafe_detail not in evaluated.run.reason
    assert all(metric.reason == expected for metric in evaluated.metrics)


@pytest.mark.parametrize("method_id", ["maskimpute", "capacity-matched-ae"])
def test_in_tree_learned_methods_use_the_shared_raw_count_converter(
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

    evaluated = evaluate_adapter_outcome(
        _entry_for(prepared, method_id, seed=42),
        prepared,
        AdapterOutcome.completed(
            execution,
            runtime_seconds=1,
            peak_rss_bytes=1,
            peak_gpu_bytes=1,
        ),
    )

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
    assert len(evaluated.metrics) == 35
    assert all(metric.value is None for metric in evaluated.metrics)
    assert all(metric.status == expected_status for metric in evaluated.metrics)
    assert all(metric.reason == reason for metric in evaluated.metrics)
    assert evaluated.evaluator_output is None


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
    entries = (
        _entry_for(prepared, "observed", seed=None),
        replace(
            _entry_for(prepared, "d3impute", seed=None),
            ordinal=2,
            run_id="run-d3impute-test",
        ),
    )
    return CompetitionPlan(
        schema_version=1,
        input_hashes={
            "dataset_manifest_sha256": prepared.binding.manifest_sha256,
            "method_registry_sha256": SHA_B,
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "implementation_source_sha256": implementation_source_sha256(),
        },
        entries=entries,
        plan_sha256="c" * 64,
    )


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
    loaded = store.load(plan)
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
    partial = store.load(plan)
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
        store.load(changed)


def test_checkpoint_resume_rejects_changed_implementation_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    source_root = tmp_path / "source"
    _write_implementation_source_fixture(source_root)
    source_sha256 = implementation_source_sha256(source_root)
    original_plan = _two_method_plan(prepared)
    plan = replace(
        original_plan,
        input_hashes={
            **original_plan.input_hashes,
            "implementation_source_sha256": source_sha256,
        },
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
        store.load(plan)


def test_checkpoint_revalidates_bound_raw_logs_and_common_output(
    tmp_path: Path,
) -> None:
    prepared = _prepared_truth_dataset()
    entry = _entry_for(prepared, "observed", seed=None)
    plan = CompetitionPlan(
        schema_version=1,
        input_hashes={
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "implementation_source_sha256": implementation_source_sha256(),
        },
        entries=(entry,),
        plan_sha256="e" * 64,
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
    assert store.load(plan) == report

    stdout = store.output_dir / run["stdout_path"]
    stdout.write_bytes(b"tampered")
    with pytest.raises(RunnerContractError, match="stdout.*checksum"):
        store.load(plan)


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
        store.load(plan)


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
