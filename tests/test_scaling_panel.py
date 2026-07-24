from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


@lru_cache(maxsize=1)
def _task15_test_module():
    path = Path(__file__).with_name("test_final_runner.py")
    spec = importlib.util.spec_from_file_location("_task16_final_factory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _direct_scaling_authorities():
    from maskimpute_benchmark.final_runner import _frozen_method_plan_authority

    module = _task15_test_module()
    frozen = module._direct_frozen_method()
    registry = module._full_registry()
    _rows, configurations = _frozen_method_plan_authority(frozen, registry)
    by_method = {value.method_id: value for value in configurations}
    return (
        frozen,
        registry,
        tuple(
            by_method[method_id]
            for method_id in ("observed", "maskimpute", "dca", "scvi", "magic")
        ),
    )


def _claimed_lifecycle_round(tmp_path: Path) -> tuple[Path, Path]:
    from maskimpute_benchmark.study import (
        assert_final_runnable,
        freeze_round,
        materialize_final,
    )

    repository = tmp_path / "lifecycle-repository"
    repository.mkdir()
    subprocess.run(("git", "init"), cwd=repository, check=True, capture_output=True)
    subprocess.run(
        ("git", "config", "user.name", "Scaling Test"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.email", "scaling@example.invalid"),
        cwd=repository,
        check=True,
    )
    (repository / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repository / "config.json").write_text('{"method":"fixture"}\n')
    (repository / "environment.lock").write_text("python=3.11\n")
    (repository / "protocol.json").write_bytes(
        (REPOSITORY / "study/protocol.json").read_bytes()
    )
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "-m", "freeze scaling lifecycle fixture"),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    round_dir = repository / "artifacts/study/round-001"
    freeze_round(
        repository,
        round_dir,
        repository / "config.json",
        repository / "protocol.json",
        environment_path=repository / "environment.lock",
    )
    materialize_final(round_dir, seed_count=4, repo=repository)
    assert_final_runnable(repository, round_dir)
    (round_dir / "results").mkdir()
    return repository, round_dir


def _configurations():
    _frozen, registry, values = _direct_scaling_authorities()
    return registry, values


def _plan():
    from maskimpute_benchmark.scaling import build_scaling_plan, load_scaling_contract

    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")
    frozen, registry, configurations = _direct_scaling_authorities()
    return build_scaling_plan(
        contract,
        registry,
        configurations,
        frozen_method_sha256=str(frozen["payload_sha256"]),
        method_registry_file_sha256=SHA_B,
        protocol_file_sha256=hashlib.sha256(
            (REPOSITORY / "study/protocol.json").read_bytes()
        ).hexdigest(),
        execution_authority_sha256=SHA_C,
        execution_environment_sha256=SHA_D,
        implementation_source_sha256="e" * 64,
    )


def _single_entry_plan():
    from maskimpute_benchmark.protocol import canonical_sha256

    full = _plan()
    body = {
        "schema_version": 1,
        "input_hashes": dict(full.input_hashes),
        "entries": [full.entries[0].to_dict()],
        "configurations": [value.to_dict() for value in full.configurations],
    }
    return full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(full.entries[0],),
        configurations=full.configurations,
        plan_sha256=canonical_sha256(body),
    )


def _single_direct_entry_plan():
    from maskimpute_benchmark.protocol import canonical_sha256

    full = _plan()
    entry = replace(full.entries[4], ordinal=1)
    body = {
        "schema_version": 1,
        "input_hashes": dict(full.input_hashes),
        "entries": [entry.to_dict()],
        "configurations": [value.to_dict() for value in full.configurations],
    }
    return full.__class__(
        schema_version=1,
        input_hashes=full.input_hashes,
        entries=(entry,),
        configurations=full.configurations,
        plan_sha256=canonical_sha256(body),
    )


def test_tracked_scaling_contract_is_exact_and_prespecified() -> None:
    from maskimpute_benchmark.scaling import load_scaling_contract

    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")

    assert contract.role == "post_freeze_resource_scaling"
    assert contract.mechanism == "symsim"
    assert contract.technical_view == "moderate"
    assert contract.cell_counts == (10_000, 25_000, 50_000, 100_000)
    assert contract.accuracy_cell_counts == contract.cell_counts
    assert contract.accuracy_metrics == (
        "mse",
        "mse_dropout",
        "mse_pre_dropout_zero",
        "mse_nonzero",
        "gnrmse",
        "mean_distortion",
        "variance_distortion",
        "mean_gene_wasserstein_distance",
        "corr_err",
        "n_corr_genes",
    )
    assert contract.genes == 500
    assert contract.method_ids == (
        "observed",
        "maskimpute",
        "dca",
        "scvi",
        "magic",
    )
    assert contract.model_seed == 42
    assert dict(contract.excluded_metric_families) == {
        "cell_cell_correlation_and_distance": (
            "quadratic_cell_pair_metric_not_scalable"
        ),
        "p_pre_zero_score_evidence": (
            "evaluated_in_main_final_panel_not_retained_in_scaling_panel"
        ),
    }
    assert (
        contract.file_sha256
        == hashlib.sha256(
            (REPOSITORY / "study/scaling_panel.json").read_bytes()
        ).hexdigest()
    )


def test_scaling_contract_rejects_design_widening(tmp_path: Path) -> None:
    from maskimpute_benchmark.scaling import ScalingContractError, load_scaling_contract

    raw = (REPOSITORY / "study/scaling_panel.json").read_text(encoding="utf-8")
    changed = tmp_path / "scaling.json"
    changed.write_text(raw.replace("100000\n  ]", "200000\n  ]", 1))

    with pytest.raises(ScalingContractError, match="cell-count grid"):
        load_scaling_contract(changed)


def test_scaling_seed_derivation_and_requests_are_exact(tmp_path: Path) -> None:
    from maskimpute_benchmark.protocol import load_protocol
    from maskimpute_benchmark.scaling import (
        derive_scaling_seeds,
        load_scaling_contract,
        scaling_protocol,
        scaling_requests,
    )

    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")
    base = load_protocol(REPOSITORY / "study/protocol.json")
    protocols = [
        scaling_protocol(base, contract, cells) for cells in contract.cell_counts
    ]
    all_seeds = []
    for cells, protocol in zip(contract.cell_counts, protocols, strict=True):
        assert protocol.development.namespace == f"scaling-{cells}"
        assert protocol.development.cells == cells
        assert protocol.development.genes == 500
        seeds = derive_scaling_seeds(contract, cells)
        assert seeds == derive_scaling_seeds(contract, cells)
        all_seeds.extend(asdict(seeds).values())
        requests = scaling_requests(contract, protocol, tmp_path)
        assert tuple(request.technical_view for request in requests) == (
            "moderate",
            "severe",
        )
        assert all(request.cells == cells for request in requests)
        assert all(
            f"scaling-{cells}" in request.output_path.parts for request in requests
        )
        assert requests[0].biological_seed == requests[1].biological_seed
        assert requests[0].measurement_seed != requests[1].measurement_seed
    assert len(all_seeds) == len(set(all_seeds))


def test_scaling_plan_closes_full_method_by_size_denominator() -> None:
    from maskimpute_benchmark.scaling import load_scaling_contract

    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")
    plan = _plan()

    assert len(plan.entries) == 4 * 5
    assert plan.input_hashes["scaling_contract_sha256"] == contract.file_sha256
    assert [entry.ordinal for entry in plan.entries] == list(range(1, 21))
    assert [entry.cells for entry in plan.entries[:5]] == [10_000] * 5
    assert [entry.method_id for entry in plan.entries[:5]] == list(contract.method_ids)
    assert plan.entries[0].model_seed is None
    assert all(entry.model_seed == 42 for entry in plan.entries[1:5])
    assert all(entry.accuracy_enabled for entry in plan.entries)
    assert plan.entries[0].native_output_scale == "raw_counts"
    assert plan.entries[0].timeout_seconds == 21_600
    assert plan.entries[0].max_rss_bytes == 48 * 1024**3
    assert plan.entries[0].max_gpu_bytes == 0
    assert plan.entries[0].rss_measurement == "linux_proc_process_tree_rss"
    assert plan.entries[0].gpu_measurement == "not_applicable_cpu_only_method"
    assert plan.entries[1].max_gpu_bytes == 14 * 1024**3
    assert plan.entries[1].gpu_measurement == "nvidia_smi_process_tree_used_memory"
    assert len({entry.run_id for entry in plan.entries}) == len(plan.entries)


def test_scaling_plan_uses_exact_frozen_comparator_payloads() -> None:
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.runner import direct_bound_comparator_value
    from maskimpute_benchmark.scaling import build_scaling_plan, load_scaling_contract

    frozen, registry, configurations = _direct_scaling_authorities()
    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")
    plan = build_scaling_plan(
        contract,
        registry,
        configurations,
        frozen_method_sha256=str(frozen["payload_sha256"]),
        method_registry_file_sha256=SHA_B,
        protocol_file_sha256=hashlib.sha256(
            (REPOSITORY / "study/protocol.json").read_bytes()
        ).hexdigest(),
        execution_authority_sha256=SHA_C,
        execution_environment_sha256=SHA_D,
        implementation_source_sha256="e" * 64,
    )

    planned = {row.method_id: row for row in plan.configurations}
    frozen_selected = frozen["selected_comparator_configurations"]
    for method_id in ("magic", "dca", "scvi"):
        expected = planned[method_id].comparator_configuration
        assert expected is not None
        assert direct_equal(
            direct_bound_comparator_value(expected),
            frozen_selected[method_id],
        )
        entries = [row for row in plan.entries if row.method_id == method_id]
        assert entries
        assert all(
            direct_equal(row.comparator_configuration, expected) for row in entries
        )
        assert all(row.comparator_nonexecution_identity is None for row in entries)


@pytest.mark.parametrize("component", ("payload", "method", "authority"))
def test_scaling_plan_rejects_complete_frozen_comparator_component_drift(
    component: str,
) -> None:
    from maskimpute_benchmark.comparator_tuning import BoundComparatorConfiguration
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        build_scaling_plan,
        load_scaling_contract,
    )

    frozen, registry, configurations = _direct_scaling_authorities()
    magic_index = next(
        index
        for index, configuration in enumerate(configurations)
        if configuration.method_id == "magic"
    )
    authority = configurations[magic_index]
    bound = authority.comparator_configuration
    assert bound is not None
    if component == "payload":
        payload = dict(bound.configuration.payload)
        payload["task16_tamper"] = True
        changed_bound = replace(
            bound,
            configuration=replace(
                bound.configuration,
                payload_json=json.dumps(payload, separators=(",", ":"), sort_keys=True),
            ),
        )
    elif component == "method":
        changed_bound = replace(
            bound,
            method=replace(
                bound.method,
                timeout_seconds=bound.method.timeout_seconds + 1,
            ),
        )
    else:
        changed_bound = replace(
            bound,
            authority_reference=replace(
                bound.authority_reference,
                authority_revision="tampered-authority-revision",
            ),
        )
    assert isinstance(changed_bound, BoundComparatorConfiguration)
    changed = list(configurations)
    changed[magic_index] = replace(
        authority,
        comparator_configuration=changed_bound,
    )

    with pytest.raises(ScalingContractError, match="comparator.*(identity|invalid)"):
        build_scaling_plan(
            load_scaling_contract(REPOSITORY / "study/scaling_panel.json"),
            registry,
            tuple(changed),
            frozen_method_sha256=str(frozen["payload_sha256"]),
            method_registry_file_sha256=SHA_B,
            protocol_file_sha256=hashlib.sha256(
                (REPOSITORY / "study/protocol.json").read_bytes()
            ).hexdigest(),
            execution_authority_sha256=SHA_C,
            execution_environment_sha256=SHA_D,
            implementation_source_sha256="e" * 64,
        )


def test_scaling_plan_rejects_registry_defaults_for_frozen_comparators() -> None:
    from maskimpute_benchmark.runner import AuthorizedConfiguration
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        build_scaling_plan,
        load_scaling_contract,
    )

    frozen, registry, configurations = _direct_scaling_authorities()
    changed = tuple(
        replace(
            value,
            legacy_configuration=AuthorizedConfiguration.registry_default(
                registry.by_id(value.method_id)
            ),
            comparator_configuration=None,
        )
        if value.method_id == "dca"
        else value
        for value in configurations
    )

    with pytest.raises(ScalingContractError, match="frozen comparator"):
        build_scaling_plan(
            load_scaling_contract(REPOSITORY / "study/scaling_panel.json"),
            registry,
            changed,
            frozen_method_sha256=str(frozen["payload_sha256"]),
            method_registry_file_sha256=SHA_B,
            protocol_file_sha256=hashlib.sha256(
                (REPOSITORY / "study/protocol.json").read_bytes()
            ).hexdigest(),
            execution_authority_sha256=SHA_C,
            execution_environment_sha256=SHA_D,
            implementation_source_sha256="e" * 64,
        )


def test_scaling_storage_preflight_is_pure_and_authority_derived() -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.scaling as scaling
    from maskimpute_benchmark.protocol import canonical_sha256

    plan = _plan()
    receipt = scaling.scaling_storage_preflight(SimpleNamespace(plan=plan))

    assert receipt["schema"] == "maskimpute-scaling-storage-preflight-v1"
    assert receipt["plan_sha256"] == plan.plan_sha256
    assert receipt["planned_run_count"] == len(plan.entries)
    assert receipt["cell_counts"] == [10_000, 25_000, 50_000, 100_000]
    assert receipt["genes"] == 500
    assert receipt["required_free_bytes"] == max(
        receipt["peak_materialization_bound_bytes"],
        receipt["retained_completion_bound_bytes"],
    )
    assert receipt["required_free_bytes"] > max(
        entry.cells * entry.genes * 96 + 2 * 1024**3 for entry in plan.entries
    )
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_scaling_plan_rejects_registry_default_for_candidate() -> None:
    from maskimpute_benchmark.runner import AuthorizedConfiguration
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        build_scaling_plan,
        load_scaling_contract,
    )

    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")
    registry, configurations = _configurations()
    changed = tuple(
        replace(
            value,
            legacy_configuration=AuthorizedConfiguration.registry_default(
                registry.by_id("maskimpute")
            ),
        )
        if value.method_id == "maskimpute"
        else value
        for value in configurations
    )

    with pytest.raises(ScalingContractError, match="frozen candidate"):
        build_scaling_plan(
            contract,
            registry,
            changed,
            frozen_method_sha256=SHA_A,
            method_registry_file_sha256=SHA_B,
            protocol_file_sha256=hashlib.sha256(
                (REPOSITORY / "study/protocol.json").read_bytes()
            ).hexdigest(),
            execution_authority_sha256=SHA_C,
            execution_environment_sha256=SHA_D,
            implementation_source_sha256="e" * 64,
        )


def test_scaling_record_keeps_native_and_evaluator_outputs_until_storage() -> None:
    import numpy as np

    from maskimpute_benchmark.runner import (
        LongFormMetric,
        RawRunResult,
    )
    from maskimpute_benchmark.scaling import (
        ScalingEvaluatedAttempt,
        scaling_attempt_record,
    )

    run = RawRunResult(
        run_id="scaling-maskimpute-10000-seed-42-aaaaaaaaaaaa",
        method_id="maskimpute",
        dataset_id="dataset-fixture",
        source_dataset_sha256=SHA_A,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        model_seed=42,
        configuration_id="v27-scaling-fixture",
        configuration_sha256=SHA_B,
        configuration_kind="candidate_search",
        requires_count_score=True,
        requires_calibration=False,
        method_input_sha256=SHA_C,
        dataset_qc_policy_sha256=SHA_D,
        excluded_cell_count=0,
        excluded_cell_ids_sha256="e" * 64,
        retained_cell_count=10_000,
        retained_cell_ids_sha256="f" * 64,
        retained_gene_count=500,
        observed_zero_count=4_000_000,
        status="completed",
        reason=None,
        runtime_seconds=1.25,
        peak_rss_bytes=1024,
        peak_gpu_bytes=0,
        rss_measurement="linux_proc_process_tree_rss",
        gpu_measurement="not_applicable_cpu_only_method",
        calibration_artifact_sha256=None,
        calibration_context_sha256=None,
        calibration_training_manifest_sha256s=(),
        calibration_held_out_manifest_sha256s=(),
        calibration_fold_calibrator_sha256=None,
        stdout_sha256=hashlib.sha256(b"out").hexdigest(),
        stderr_sha256=hashlib.sha256(b"err").hexdigest(),
        native_output_sha256=SHA_A,
        evaluator_output_sha256=SHA_B,
    )
    metric = LongFormMetric(
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-fixture",
        method="maskimpute",
        model_seed=42,
        configuration_id="v27-scaling-fixture",
        configuration_sha256=SHA_B,
        metric="mse",
        value=0.5,
        n=10_000 * 500,
        status="completed",
        reason=None,
    )
    attempt = ScalingEvaluatedAttempt(
        run=run,
        metrics=(metric,),
        stdout=b"out",
        stderr=b"err",
        native_output=np.ones((2, 2)),
        native_output_scale="raw_counts",
        evaluator_output=np.ones((2, 2)),
    )

    record = scaling_attempt_record(attempt, cells=10_000, accuracy_enabled=True)

    assert record["run"]["native_output_retention"] == "compressed_zlib_raw_f64_v1"
    assert record["run"]["evaluator_output_retention"] == "compressed_zlib_raw_f64_v1"
    np.testing.assert_array_equal(record["native_output"], np.ones((2, 2)))
    np.testing.assert_array_equal(record["evaluator_output"], np.ones((2, 2)))
    assert isinstance(record["executor_receipt"], bytes)
    assert "p_pre_zero_evidence" not in record
    assert record["metrics"] == [metric.to_dict()]
    assert record["stdout"] == b"out"
    assert record["stderr"] == b"err"


def _fixture_scaling_simulator(requests, protocol):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse

    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.schema import benchmark_dataset_sha256
    from maskimpute_benchmark.simulators import SimulationArtifact
    from maskimpute_benchmark.simulators.base import simulation_scientific_identity
    from maskimpute_benchmark.simulators.native import seal_native_outputs

    values = tuple(requests)
    assert tuple(value.technical_view for value in values) == ("moderate", "severe")
    assert all(value.cells == protocol.development.cells for value in values)
    cells = values[0].cells
    rows = np.arange(cells, dtype=np.int32)
    observed = sparse.csr_matrix(
        (np.ones(cells, dtype=np.int64), (rows, rows % 500)),
        shape=(cells, 500),
    )
    truth_columns = np.concatenate((rows % 500, (rows + 1) % 500))
    truth_rows = np.concatenate((rows, rows))
    truth = sparse.csr_matrix(
        (
            np.ones(cells * 2, dtype=np.int64),
            (truth_rows, truth_columns),
        ),
        shape=(cells, 500),
    )
    var = pd.DataFrame(index=[f"gene-{index:04d}" for index in range(500)])
    native_parent = values[0].output_path.parent / "native" / "fixture"
    native_parent.mkdir(parents=True, exist_ok=True)
    native_file = native_parent / "native.txt"
    native_file.write_bytes(b"deterministic-symsim-fixture\n")
    artifacts = []
    for request in values:
        manifest = seal_native_outputs(
            {"native.txt": native_file},
            {
                "adapter": "deterministic-symsim-test-fixture-v1",
                "simulation_request": simulation_scientific_identity(request),
            },
        )
        obs = pd.DataFrame(
            {
                "dataset_id": [request.dataset_id] * cells,
                "mechanism": ["symsim"] * cells,
                "condition": [request.technical_view] * cells,
                "biological_id": ["draw-01"] * cells,
                "technical_view": [request.technical_view] * cells,
                "draw": np.ones(cells, dtype=np.int64),
                "library_size": np.ones(cells, dtype=np.int64),
                "group": np.where(rows % 2 == 0, "group-a", "group-b"),
            },
            index=[f"cell-{index:06d}" for index in range(cells)],
        )
        dataset = ad.AnnData(X=observed.copy(), obs=obs, var=var.copy())
        dataset.layers["pre_capture_counts"] = truth.copy()
        dataset.uns["truth_kind"] = "exact_pre_capture"
        dataset.uns["primary_truth_layer"] = "pre_capture_counts"
        dataset.uns["provenance"] = {
            "source": "https://example.invalid/deterministic-symsim-fixture",
            "source_sha256": canonical_sha256(
                {"fixture": "deterministic-symsim-test-fixture-v1"}
            ),
            "software": "SymSim",
            "software_version": "fixture-1",
            "parameters": {
                "adapter": "deterministic-symsim-test-fixture-v1",
                "native_manifest_sha256": manifest.manifest_sha256,
            },
            "seeds": {
                "biological": request.biological_seed,
                "measurement": request.measurement_seed,
            },
        }
        dataset.uns["normalization"] = {
            "input": "raw_umi_counts",
            "size_factor": "none",
        }
        dataset.uns["allowed_covariates"] = {"obs": [], "var": []}
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_h5ad(request.output_path)
        persisted = ad.read_h5ad(request.output_path)
        artifacts.append(
            SimulationArtifact(
                request,
                persisted,
                manifest,
                benchmark_dataset_sha256(persisted),
            )
        )
    return tuple(artifacts)


@pytest.fixture(autouse=True)
def _use_deterministic_scaling_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    import maskimpute_benchmark.scaling as module

    monkeypatch.setattr(module, "run_symsim_pair", _fixture_scaling_simulator)


def _dataset_receipt(output_dir: Path, cells: int = 10_000) -> dict[str, object]:
    from maskimpute_benchmark.protocol import load_protocol
    from maskimpute_benchmark.scaling import (
        _dataset_receipt_from_artifacts,
        load_scaling_contract,
        scaling_protocol,
        scaling_requests,
    )

    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")
    base = load_protocol(REPOSITORY / "study/protocol.json")
    protocol = scaling_protocol(base, contract, cells)
    requests = scaling_requests(contract, protocol, output_dir / "generated")
    receipt, _dataset = _dataset_receipt_from_artifacts(
        contract,
        protocol,
        output_dir,
        _fixture_scaling_simulator(requests, protocol),
    )
    return receipt


def _first_attempt(
    plan,
    output_dir: Path,
    receipt,
    *,
    entry_index: int = 0,
    stdout: bytes = b"out",
    stderr: bytes = b"err",
):
    import anndata as ad
    import numpy as np

    from maskimpute_benchmark.runner import (
        DatasetQCPolicy,
        LongFormMetric,
        RawRunResult,
        _evaluator_output_sha256,
        _evaluator_targets,
        method_input_sha256,
        prepare_dataset_for_execution,
    )
    from maskimpute_benchmark.scaling import (
        ScalingEvaluatedAttempt,
        _bounded_scaling_metric_values,
        _dataset_binding,
        _run_plan_entry,
    )
    from maskimpute_benchmark.methods import _output_digest

    entry = plan.entries[entry_index]
    dataset = ad.read_h5ad(output_dir / receipt["moderate_output_path"])
    prepared = prepare_dataset_for_execution(
        dataset, _dataset_binding(receipt), DatasetQCPolicy.fixed()
    )
    observed, truth, truth_kind, _marker = _evaluator_targets(prepared)
    assert truth_kind == "exact_pre_capture"
    assert truth is not None
    evaluator_output = np.asarray(observed, dtype=np.float64)
    native_output = np.asarray(prepared.method_input.counts, dtype=np.float64)
    registry, _unused_configurations = _configurations()
    native_output_scale = registry.by_id(entry.method_id).output_scale
    native_output_sha256 = _output_digest(
        method_id=entry.method_id,
        source_dataset_sha256=prepared.binding.dataset_sha256,
        output_scale=native_output_scale,
        obs_ids=prepared.audit.retained_cell_ids,
        var_ids=prepared.method_input.var_ids,
        shape=prepared.method_input.shape,
        matrix_bytes=np.asarray(native_output, dtype="<f8", order="C").tobytes(
            order="C"
        ),
    )
    metric_values = _bounded_scaling_metric_values(evaluator_output, observed, truth)
    evaluator_output_sha256 = _evaluator_output_sha256(
        _run_plan_entry(entry, prepared.binding), prepared, evaluator_output
    )
    run = RawRunResult(
        run_id=entry.run_id,
        method_id=entry.method_id,
        dataset_id=str(receipt["dataset_id"]),
        source_dataset_sha256=str(receipt["dataset_sha256"]),
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        model_seed=entry.model_seed,
        configuration_id=entry.configuration_id,
        configuration_sha256=entry.configuration_sha256,
        configuration_kind=entry.configuration_kind,
        requires_count_score=entry.requires_count_score,
        requires_calibration=entry.requires_calibration,
        method_input_sha256=method_input_sha256(prepared.method_input),
        dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
        excluded_cell_count=prepared.audit.excluded_cell_count,
        excluded_cell_ids_sha256=prepared.audit.excluded_cell_ids_sha256,
        retained_cell_count=prepared.audit.retained_cell_count,
        retained_cell_ids_sha256=prepared.audit.retained_cell_ids_sha256,
        retained_gene_count=prepared.method_input.shape[1],
        observed_zero_count=int((prepared.method_input.counts == 0).sum()),
        status="completed",
        reason=None,
        runtime_seconds=1.25,
        peak_rss_bytes=1024,
        peak_gpu_bytes=0,
        rss_measurement="linux_proc_process_tree_rss",
        gpu_measurement=entry.gpu_measurement,
        calibration_artifact_sha256=None,
        calibration_context_sha256=None,
        calibration_training_manifest_sha256s=(),
        calibration_held_out_manifest_sha256s=(),
        calibration_fold_calibrator_sha256=None,
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        native_output_sha256=native_output_sha256,
        evaluator_output_sha256=evaluator_output_sha256,
        comparator_configuration=entry.comparator_configuration,
        comparator_nonexecution_identity=entry.comparator_nonexecution_identity,
    )
    metrics = tuple(
        LongFormMetric(
            mechanism="symsim",
            biological_id="draw-01",
            technical_view="moderate",
            dataset_id=str(receipt["dataset_id"]),
            method=entry.method_id,
            model_seed=entry.model_seed,
            configuration_id=entry.configuration_id,
            configuration_sha256=entry.configuration_sha256,
            metric=name,
            value=value,
            n=n,
            status="completed" if value is not None else "unavailable",
            reason=reason,
            comparator_configuration=entry.comparator_configuration,
            comparator_nonexecution_identity=(entry.comparator_nonexecution_identity),
        )
        for name, (value, n, reason) in metric_values.items()
    )
    return ScalingEvaluatedAttempt(
        run=run,
        metrics=metrics,
        stdout=stdout,
        stderr=stderr,
        native_output=native_output,
        native_output_scale=native_output_scale,
        evaluator_output=evaluator_output,
    )


def test_scaling_store_resumes_exact_prefix_with_compressed_evaluator_output(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    report = store.append_dataset(receipt)
    assert report.status == "running"
    attempt = _first_attempt(plan, tmp_path, receipt)
    report = store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(attempt, cells=10_000, accuracy_enabled=True),
    )

    assert len(report.datasets) == 1
    assert len(report.records) == 1
    run = report.records[0]["run"]
    assert run["native_output_retention"] == "compressed_zlib_raw_f64_v1"
    assert run["evaluator_output_retention"] == "compressed_zlib_raw_f64_v1"
    assert run["stdout_path"].endswith(".stdout")
    assert run["evaluator_output_path"].endswith(".log2-cp10k-f64.zlib")
    assert run["native_output_path"].endswith(".native-f64.zlib")
    assert run["executor_receipt_path"].endswith(".executor-receipt.json")
    assert store.load() == report


def _mutate_direct_scaling_value(value: dict[str, object], component: str) -> None:
    if component == "payload":
        value["configuration"]["payload"]["task16_tamper"] = True
    elif component == "method":
        value["method"]["timeout_seconds"] += 1
    else:
        value["authority_reference"]["authority_revision"] = (
            "tampered-authority-revision"
        )


@pytest.mark.parametrize(
    ("boundary", "component"),
    (
        ("executor_receipt", "payload"),
        ("checkpoint_run", "method"),
        ("stored_metric", "authority"),
    ),
)
def test_scaling_direct_evidence_rejects_complete_identity_tampering(
    tmp_path: Path,
    boundary: str,
    component: str,
) -> None:
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _single_direct_entry_plan()
    entry = plan.entries[0]
    expected = entry.to_dict()["comparator_configuration"]
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    report = store.append_attempt(
        entry,
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    run = report.records[0]["run"]
    assert "configuration_sha256" not in run
    assert direct_equal(run["comparator_configuration"], expected)
    assert all(
        "configuration_sha256" not in metric
        and direct_equal(metric["comparator_configuration"], expected)
        for metric in report.records[0]["metrics"]
    )

    checkpoint_path = store.checkpoint_path
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = payload["records"][0]
    run_payload = record["run"]
    if boundary == "executor_receipt":
        executor_path = tmp_path / run_payload["executor_receipt_path"]
        executor = json.loads(executor_path.read_text(encoding="utf-8"))
        _mutate_direct_scaling_value(executor["comparator_configuration"], component)
        executor_unsigned = {
            key: value for key, value in executor.items() if key != "receipt_sha256"
        }
        executor["receipt_sha256"] = canonical_sha256(executor_unsigned)
        executor_raw = (
            json.dumps(executor, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        executor_path.write_bytes(executor_raw)
        run_payload["executor_receipt_file_sha256"] = hashlib.sha256(
            executor_raw
        ).hexdigest()
        run_payload["executor_receipt_size_bytes"] = len(executor_raw)
        run_payload["executor_receipt_sha256"] = executor["receipt_sha256"]
    elif boundary == "checkpoint_run":
        _mutate_direct_scaling_value(run_payload["comparator_configuration"], component)
    else:
        _mutate_direct_scaling_value(
            record["metrics"][0]["comparator_configuration"], component
        )
    record_unsigned = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = canonical_sha256(record_unsigned)
    checkpoint_unsigned = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    payload["checkpoint_sha256"] = canonical_sha256(checkpoint_unsigned)
    checkpoint_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ScalingContractError,
        match="comparator|metric|identity|differs from its plan",
    ):
        ScalingResultStore(tmp_path, plan).load()


def test_scaling_checkpoints_are_immutable_append_only_snapshots(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    first_path = store.checkpoint_path
    first_raw = first_path.read_bytes()

    store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )

    assert store.checkpoint_path != first_path
    assert first_path.read_bytes() == first_raw
    assert tuple(path.name for path in sorted(first_path.parent.iterdir())) == (
        "00000001.json",
        "00000002.json",
    )


def test_scaling_store_rejects_log_tampering(tmp_path: Path) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    report = store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    stdout = tmp_path / report.records[0]["run"]["stdout_path"]
    stdout.write_bytes(b"tampered")

    with pytest.raises(ScalingContractError, match="stdout integrity"):
        ScalingResultStore(tmp_path, plan).load()


def _rewrite_scaling_checkpoint(path: Path, mutate) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256

    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    body = {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    payload["checkpoint_sha256"] = canonical_sha256(body)
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("boundary", ("checkpoint", "dataset_receipt"))
def test_scaling_store_rejects_boolean_schema_versions(
    tmp_path: Path,
    boundary: str,
) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.scaling import ScalingContractError, ScalingResultStore

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    store.append_dataset(_dataset_receipt(tmp_path))

    def mutate(payload: dict[str, object]) -> None:
        if boundary == "checkpoint":
            payload["schema_version"] = True
            return
        receipt = payload["datasets"][0]
        receipt["schema_version"] = True
        unsigned = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        receipt["receipt_sha256"] = canonical_sha256(unsigned)

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)

    with pytest.raises(ScalingContractError, match="checkpoint|dataset receipt"):
        ScalingResultStore(tmp_path, plan).load()


def test_scaling_store_rejects_rehashed_seed_authority_drift(tmp_path: Path) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)

    def mutate(payload: dict[str, object]) -> None:
        receipt = payload["datasets"][0]
        receipt["seeds"]["biological"] += 100
        unsigned = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        receipt["receipt_sha256"] = canonical_sha256(unsigned)

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)
    with pytest.raises(ScalingContractError, match="seed|design|authority"):
        ScalingResultStore(tmp_path, plan).load()


def test_scaling_store_rejects_rehashed_invalid_run_and_metric_semantics(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )

    def mutate(payload: dict[str, object]) -> None:
        record = payload["records"][0]
        record["run"]["status"] = "forged_status"
        record["run"]["runtime_seconds"] = -99
        record["run"]["peak_rss_bytes"] = -1
        record["metrics"][0]["value"] = "not-a-number"
        record["metrics"][0]["n"] = -7
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        record["record_sha256"] = canonical_sha256(unsigned)

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)
    with pytest.raises(ScalingContractError, match="status|runtime|resource|metric"):
        ScalingResultStore(tmp_path, plan).load()


def test_scaling_store_rejects_coordinated_h5ad_and_receipt_rehash(
    tmp_path: Path,
) -> None:
    import anndata as ad

    from maskimpute_benchmark.datasets import _truth_sha256
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
    )
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    retained = tmp_path / str(receipt["moderate_output_path"])
    changed = ad.read_h5ad(retained)
    original_semantic_sha256 = benchmark_dataset_sha256(changed)
    changed.obs.loc[changed.obs.index[0], "group"] = "group-b"
    changed.write_h5ad(retained)
    changed_semantic_sha256 = benchmark_dataset_sha256(changed)
    assert changed_semantic_sha256 != original_semantic_sha256

    def mutate(payload: dict[str, object]) -> None:
        dataset_receipt = payload["datasets"][0]
        dataset_receipt["dataset_sha256"] = changed_semantic_sha256
        dataset_receipt["truth_sha256"] = _truth_sha256(changed)
        dataset_receipt["moderate_output_file_sha256"] = hashlib.sha256(
            retained.read_bytes()
        ).hexdigest()
        dataset_receipt["moderate_output_size_bytes"] = retained.stat().st_size
        unsigned = {
            key: value
            for key, value in dataset_receipt.items()
            if key != "receipt_sha256"
        }
        dataset_receipt["receipt_sha256"] = canonical_sha256(unsigned)

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)

    with pytest.raises(ScalingContractError, match="generator|SymSim|authority"):
        ScalingResultStore(tmp_path, plan).load()


@pytest.mark.parametrize(
    ("rewrite", "expected_error"),
    [
        ("finite_metric", "metric.*replay|metric.*output"),
        ("runtime_below_ceiling", "executor|receipt|resource"),
        ("runtime_over_ceiling", "executor|receipt|runtime|resource"),
        ("rss_over_ceiling", "executor|receipt|resource|RSS"),
        ("gpu_over_ceiling", "executor|receipt|resource|GPU"),
        ("measurement_provenance", "executor|receipt|measurement|resource"),
    ],
)
def test_scaling_store_rejects_coordinated_finite_result_rehash(
    tmp_path: Path,
    rewrite: str,
    expected_error: str,
) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )

    def mutate(payload: dict[str, object]) -> None:
        record = payload["records"][0]
        if rewrite == "finite_metric":
            record["metrics"][0]["value"] = 12_345.0
        elif rewrite == "runtime_below_ceiling":
            record["run"]["runtime_seconds"] = 2.75
            record["run"]["peak_rss_bytes"] = 2_048
        elif rewrite == "runtime_over_ceiling":
            record["run"]["runtime_seconds"] = 21_601.0
        elif rewrite == "rss_over_ceiling":
            record["run"]["peak_rss_bytes"] = 49 * 1024**3
        elif rewrite == "gpu_over_ceiling":
            record["run"]["peak_gpu_bytes"] = 1
        else:
            assert rewrite == "measurement_provenance"
            record["run"]["rss_measurement"] = "forged_process_tree_telemetry"
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        record["record_sha256"] = canonical_sha256(unsigned)

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)

    with pytest.raises(ScalingContractError, match=expected_error):
        ScalingResultStore(tmp_path, plan).load()


def test_scaling_store_rejects_rehashed_evaluator_replacement(
    tmp_path: Path,
) -> None:
    import zlib

    import numpy as np

    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import _evaluator_output_sha256, _evaluator_targets
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        _bounded_scaling_metric_values,
        _run_plan_entry,
        _scaling_metric_rows,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    prepared = store._prepared_datasets[10_000][2]
    _observed, truth, truth_kind, _marker = _evaluator_targets(prepared)
    assert truth_kind == "exact_pre_capture"
    assert truth is not None
    replacement = np.asarray(truth, dtype="<f8", order="C")
    entry = plan.entries[0]
    run_entry = _run_plan_entry(entry, prepared.binding)
    replacement_metrics = [
        metric.to_dict()
        for metric in _scaling_metric_rows(
            entry,
            run_entry,
            _bounded_scaling_metric_values(replacement, _observed, truth),
        )
    ]
    raw = replacement.tobytes(order="C")
    compressed = zlib.compress(raw, level=6)

    def mutate(payload: dict[str, object]) -> None:
        record = payload["records"][0]
        run = record["run"]
        artifact = tmp_path / run["evaluator_output_path"]
        artifact.write_bytes(compressed)
        run["evaluator_output_file_sha256"] = hashlib.sha256(compressed).hexdigest()
        run["evaluator_output_compressed_nbytes"] = len(compressed)
        run["evaluator_output_uncompressed_sha256"] = hashlib.sha256(raw).hexdigest()
        run["evaluator_output_sha256"] = _evaluator_output_sha256(
            run_entry, prepared, replacement
        )
        record["metrics"] = replacement_metrics
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        record["record_sha256"] = canonical_sha256(unsigned)

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)

    with pytest.raises(ScalingContractError, match="native|conversion|executor"):
        ScalingResultStore(tmp_path, plan).load()


def test_scaling_store_preserves_dual_resource_exceedance(tmp_path: Path) -> None:
    from maskimpute_benchmark.runner import AdapterOutcome
    from maskimpute_benchmark.scaling import (
        ScalingResultStore,
        _evaluate_scaling_outcome,
        _run_plan_entry,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    entry = plan.entries[1]
    prepared = store._prepared_datasets[10_000][2]
    outcome = AdapterOutcome.resource_exceeded(
        "peak_rss_exceeded",
        runtime_seconds=2.0,
        peak_rss_bytes=entry.max_rss_bytes + 1,
        peak_gpu_bytes=entry.max_gpu_bytes + 1,
        rss_measurement=entry.rss_measurement,
        gpu_measurement=entry.gpu_measurement,
    )
    attempt = _evaluate_scaling_outcome(
        entry, _run_plan_entry(entry, prepared.binding), prepared, outcome
    )

    report = store.append_attempt(
        entry,
        scaling_attempt_record(attempt, cells=10_000, accuracy_enabled=True),
    )

    assert report.records[1]["run"]["status"] == "resource_exceeded"
    assert report.records[1]["run"]["reason"] == "peak_rss_exceeded"


def test_scaling_store_rejects_changed_checkpoint_before_cached_append(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )

    def mutate(payload: dict[str, object]) -> None:
        record = payload["records"][0]
        record["run"]["runtime_seconds"] = 3.5
        unsigned = {
            key: value for key, value in record.items() if key != "record_sha256"
        }
        record["record_sha256"] = canonical_sha256(unsigned)

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)

    with pytest.raises(ScalingContractError, match="checkpoint.*changed|stale"):
        store.append_attempt(
            plan.entries[1],
            scaling_attempt_record(
                _first_attempt(plan, tmp_path, receipt, entry_index=1),
                cells=10_000,
                accuracy_enabled=True,
            ),
        )


def test_scaling_store_does_not_persist_mutated_returned_snapshot(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    report = store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    report.records[0]["run"]["runtime_seconds"] = 3.5

    store.append_attempt(
        plan.entries[1],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt, entry_index=1),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    resumed = ScalingResultStore(tmp_path, plan).load()
    assert resumed is not None
    assert resumed.records[0]["run"]["runtime_seconds"] == 1.25


def test_scaling_store_validation_failure_leaves_retryable_run_path(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    invalid = scaling_attempt_record(
        _first_attempt(plan, tmp_path, receipt, stdout=b"invalid-attempt"),
        cells=10_000,
        accuracy_enabled=True,
    )
    invalid["metrics"][0]["value"] = float(invalid["metrics"][0]["value"]) + 1.0

    with pytest.raises(ScalingContractError, match="metric.*replay|metric.*output"):
        store.append_attempt(plan.entries[0], invalid)
    assert not (tmp_path / "runs" / plan.entries[0].run_id).exists()

    report = store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    assert len(report.records) == 1


def test_scaling_store_recovers_complete_orphan_run_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.scaling import ScalingResultStore, scaling_attempt_record

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    original_write = store._write

    def interrupted_write(*_args, **_kwargs):
        raise OSError("simulated checkpoint interruption")

    monkeypatch.setattr(store, "_write", interrupted_write)
    with pytest.raises(OSError, match="interruption"):
        store.append_attempt(
            plan.entries[0],
            scaling_attempt_record(
                _first_attempt(plan, tmp_path, receipt, stdout=b"first execution"),
                cells=10_000,
                accuracy_enabled=True,
            ),
        )
    monkeypatch.setattr(store, "_write", original_write)

    resumed = ScalingResultStore(tmp_path, plan)
    assert resumed.recover_unreferenced_transactions() == (plan.entries[0].run_id,)
    report = resumed.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt, stdout=b"retried execution"),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    assert len(report.records) == 1


def test_scaling_store_serializes_two_cached_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    first_store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    first_store.append_dataset(receipt)
    second_store = ScalingResultStore(tmp_path, plan)
    assert second_store.load() is not None
    first_record = scaling_attempt_record(
        _first_attempt(plan, tmp_path, receipt, stdout=b"first writer"),
        cells=10_000,
        accuracy_enabled=True,
    )
    second_record = scaling_attempt_record(
        _first_attempt(plan, tmp_path, receipt, stdout=b"second writer"),
        cells=10_000,
        accuracy_enabled=True,
    )
    first_ready_to_publish = Event()
    release_first = Event()
    second_entered_transaction = Event()
    first_publish = first_store._publish_run_transaction
    second_prepare = second_store._prepare_run_transaction

    def hold_first(stage, final):
        first_ready_to_publish.set()
        if not release_first.wait(timeout=10):
            raise AssertionError("first writer was not released")
        return first_publish(stage, final)

    def observe_second(entry):
        second_entered_transaction.set()
        return second_prepare(entry)

    monkeypatch.setattr(first_store, "_publish_run_transaction", hold_first)
    monkeypatch.setattr(second_store, "_prepare_run_transaction", observe_second)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            first_store.append_attempt, plan.entries[0], first_record
        )
        assert first_ready_to_publish.wait(timeout=120)
        second_future = pool.submit(
            second_store.append_attempt, plan.entries[0], second_record
        )
        entered_while_first_active = second_entered_transaction.wait(timeout=0.5)
        release_first.set()
        first_report = first_future.result(timeout=20)
        with pytest.raises(ScalingContractError, match="checkpoint.*changed|prefix"):
            second_future.result(timeout=20)

    assert entered_while_first_active is False
    assert len(first_report.records) == 1
    resumed = ScalingResultStore(tmp_path, plan).load()
    assert resumed is not None
    assert len(resumed.records) == 1


def test_scaling_store_rejects_symlinked_run_component_before_write(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    store.append_dataset(receipt)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ScalingContractError, match="symlink|canonical"):
        store.append_attempt(
            plan.entries[0],
            scaling_attempt_record(
                _first_attempt(plan, tmp_path, receipt),
                cells=10_000,
                accuracy_enabled=True,
            ),
        )
    assert list(outside.iterdir()) == []


def test_scaling_store_hashes_each_retained_h5ad_once_per_fresh_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.scaling as module

    plan = _plan()
    store = module.ScalingResultStore(tmp_path, plan)
    receipt = _dataset_receipt(tmp_path)
    attempt = _first_attempt(plan, tmp_path, receipt)
    retained = (tmp_path / str(receipt["moderate_output_path"])).resolve()
    original = module._file_sha256
    retained_hash_passes = 0

    def counted(path: Path) -> str:
        nonlocal retained_hash_passes
        if path.resolve() == retained:
            retained_hash_passes += 1
        return original(path)

    monkeypatch.setattr(module, "_file_sha256", counted)
    store.append_dataset(receipt)
    store.append_attempt(
        plan.entries[0],
        module.scaling_attempt_record(
            attempt,
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    store.load()
    store.load()
    assert retained_hash_passes == 1

    resumed = module.ScalingResultStore(tmp_path, plan)
    resumed.load()
    resumed.load()
    assert retained_hash_passes == 2


def test_final_result_inventory_owns_every_scaling_checkpoint_reference(
    tmp_path: Path,
) -> None:
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.scaling import (
        ScalingResultStore,
        _cleanup_discarded_scaling_inputs,
        scaling_attempt_record,
    )

    round_dir = tmp_path / "round-001"
    output = round_dir / "results/scaling"
    plan = _plan()
    store = ScalingResultStore(output, plan)
    receipt = _dataset_receipt(output)
    store.append_dataset(receipt)
    _cleanup_discarded_scaling_inputs(output, 10_000, receipt)
    store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, output, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )

    manifest = final_runner._owned_final_result_file_manifest(round_dir)
    paths = {row["path"] for row in manifest["result_files"]}

    assert paths == {
        "results/scaling/checkpoints/00000001.json",
        "results/scaling/checkpoints/00000002.json",
        f"results/scaling/{receipt['moderate_output_path']}",
        f"results/scaling/runs/{plan.entries[0].run_id}/run.stdout",
        f"results/scaling/runs/{plan.entries[0].run_id}/run.stderr",
        f"results/scaling/runs/{plan.entries[0].run_id}/run.executor-receipt.json",
        f"results/scaling/runs/{plan.entries[0].run_id}/run.native-f64.zlib",
        f"results/scaling/runs/{plan.entries[0].run_id}/run.log2-cp10k-f64.zlib",
    }


def test_public_scaling_run_requires_a_claimed_canonical_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.scaling as scaling

    execution_called = False

    monkeypatch.setattr(
        scaling,
        "load_scaling_execution_authority",
        lambda _repository: object(),
    )

    def execute(*_args, **_kwargs):
        nonlocal execution_called
        execution_called = True
        return object()

    monkeypatch.setattr(scaling, "execute_scaling_plan", execute)

    with pytest.raises(
        scaling.ScalingContractError,
        match="claimed.*round|canonical.*round",
    ):
        scaling.run_scaling_panel(REPOSITORY, tmp_path)

    assert execution_called is False


def test_scaling_runtime_registry_receives_validated_method_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.scaling as scaling
    from maskimpute_benchmark.methods import load_method_registry

    registry = load_method_registry(REPOSITORY / "study/methods.json")
    captured: dict[str, object] = {}
    expected = object()

    def fixed(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        scaling,
        "ExecutionEnvironmentRegistry",
        type("RegistryFixture", (), {"fixed": staticmethod(fixed)}),
    )

    observed = scaling._load_scaling_execution_environment_registry(
        REPOSITORY,
        registry,
    )

    assert observed is expected
    assert captured["kwargs"]["lock_only_environment_ids"] == (
        "d3impute",
        "sctsi",
    )


def test_claimed_scaling_run_journals_each_immutable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.scaling as scaling
    import maskimpute_benchmark.simulators.base as simulator_base
    import maskimpute_benchmark.study as study

    round_dir = tmp_path / "artifacts/study/round-001"
    round_dir.mkdir(parents=True)
    authority = object()
    checkpoint = object()
    publications: list[tuple[Path, Path, object]] = []
    monkeypatch.setattr(
        simulator_base,
        "load_final_manifest_claim",
        lambda _repository, _round_dir: SimpleNamespace(round_dir=round_dir),
    )
    monkeypatch.setattr(
        scaling,
        "load_scaling_execution_authority",
        lambda _repository: authority,
    )

    def record(repository, selected_round, recorder):
        publications.append((repository, selected_round, recorder))
        return None

    monkeypatch.setattr(
        final_runner,
        "_record_incremental_results_if_changed",
        record,
    )

    def execute(selected_authority, output_dir, *, on_checkpoint_published):
        assert selected_authority is authority
        assert output_dir == round_dir / "results/scaling"
        on_checkpoint_published()
        on_checkpoint_published()
        return checkpoint

    monkeypatch.setattr(scaling, "execute_scaling_plan", execute)

    assert scaling.run_scaling_panel(tmp_path, round_dir) is checkpoint
    assert publications == [
        (tmp_path, round_dir, study.record_incremental_results),
        (tmp_path, round_dir, study.record_incremental_results),
    ]


def test_mid_scaling_checkpoint_callback_crash_reconciles_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.scaling as scaling
    from maskimpute_benchmark.simulators import (
        SimulationContractError,
        load_final_manifest_claim,
    )
    from maskimpute_benchmark.study import record_incremental_results

    repository, round_dir = _claimed_lifecycle_round(tmp_path)
    output = round_dir / "results/scaling"
    plan = _single_entry_plan()
    store = scaling.ScalingResultStore(output, plan)
    receipt = _dataset_receipt(output)
    checkpoint = store.append_dataset(receipt)
    scaling._cleanup_discarded_scaling_inputs(output, 10_000, receipt)
    assert checkpoint.status == "running"
    with pytest.raises(SimulationContractError, match="clean frozen|valid claimed"):
        load_final_manifest_claim(repository, round_dir)

    monkeypatch.setattr(
        scaling,
        "load_scaling_execution_authority",
        lambda _repository: SimpleNamespace(plan=plan),
    )
    assert (
        final_runner._recover_scaling_transactions_for_resume(
            repository,
            round_dir,
        )
        == ()
    )
    final_runner._record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"

    resumed = scaling.ScalingResultStore(output, plan)
    report = resumed.append_attempt(
        plan.entries[0],
        scaling.scaling_attempt_record(
            _first_attempt(plan, output, receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    assert report.status == "completed"
    final_runner._record_incremental_results_if_changed(
        repository,
        round_dir,
        record_incremental_results,
    )
    assert load_final_manifest_claim(repository, round_dir).round_id == "round-001"


def test_prejournal_scaling_reconciliation_rejects_rehashed_checkpoint_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.scaling as scaling

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    output = round_dir / "results/scaling"
    output.mkdir(parents=True)
    plan = _single_entry_plan()
    store = scaling.ScalingResultStore(output, plan)
    store.append_dataset(_dataset_receipt(output))
    _rewrite_scaling_checkpoint(
        store.checkpoint_path,
        lambda payload: payload.__setitem__("status", "completed"),
    )
    monkeypatch.setattr(
        scaling,
        "load_scaling_execution_authority",
        lambda _repository: SimpleNamespace(plan=plan),
    )

    with pytest.raises(
        final_runner.FinalRunnerContractError,
        match="scaling publications.*not resumable",
    ):
        final_runner._validate_scaling_publications_for_reconciliation(
            repository,
            round_dir,
        )


def test_publication_scaling_loader_requires_evaluated_receipt_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.scaling as scaling
    import maskimpute_benchmark.study as study
    from maskimpute_benchmark.protocol import canonical_sha256

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    output = round_dir / "results/scaling"
    plan = _single_entry_plan()
    store = scaling.ScalingResultStore(output, plan)
    dataset_receipt = _dataset_receipt(output)
    store.append_dataset(dataset_receipt)
    scaling._cleanup_discarded_scaling_inputs(output, 10_000, dataset_receipt)
    checkpoint = store.append_attempt(
        plan.entries[0],
        scaling.scaling_attempt_record(
            _first_attempt(plan, output, dataset_receipt),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    result_files = final_runner._owned_final_result_file_manifest(round_dir)[
        "result_files"
    ]
    monkeypatch.setattr(
        scaling,
        "load_scaling_execution_authority",
        lambda _repository: SimpleNamespace(plan=plan),
    )
    evidence = final_runner._scaling_evaluation_evidence(
        repository,
        round_dir,
        checkpoint,
        result_files,
    )
    evaluation = {
        "schema_version": 1,
        "status": "completed",
        "final_plan_sha256": "a" * 64,
        "final_execution_manifest_path": (
            "results/final/execution/execution_manifest.json"
        ),
        "final_execution_manifest_sha256": "b" * 64,
        "final_execution_payload_sha256": "c" * 64,
        "execution_validation": {},
        "storage_preflight": {},
        "scaling_evidence": evidence,
        "result_files": result_files,
    }
    lifecycle_receipt = {
        "result_manifest": evaluation,
        "result_manifest_sha256": canonical_sha256(evaluation),
    }
    monkeypatch.setattr(
        final_runner,
        "_canonical_round",
        lambda _repository, _round_dir: (repository, round_dir),
    )
    monkeypatch.setattr(study, "_validate_freeze", lambda _round, _repo: {})
    monkeypatch.setattr(
        study,
        "_validate_registry",
        lambda _repo, _round, _freeze, *, expected_state: {"state": expected_state},
    )
    monkeypatch.setattr(
        study,
        "_validate_state_record_chain",
        lambda _round, _freeze, *, expected_state: (
            {"state": "materialized"},
            {"state": "running"},
            lifecycle_receipt,
        ),
    )
    monkeypatch.setattr(
        study,
        "_verify_frozen_repository",
        lambda _repo, _round, *, allowed_result_paths: {},
    )

    loaded = scaling.load_publication_scaling_evidence(repository, round_dir)
    assert loaded.status == "completed"
    assert loaded.checkpoint_sha256 == checkpoint.checkpoint_sha256

    checkpoint_path = store.checkpoint_path
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    run = payload["records"][0]["run"]
    stdout_path = output / run["stdout_path"]
    stdout_path.write_bytes(b"coherent replacement\n")
    stdout_sha256 = hashlib.sha256(stdout_path.read_bytes()).hexdigest()
    run["stdout_sha256"] = stdout_sha256
    run["stdout_file_sha256"] = stdout_sha256
    run["stdout_size_bytes"] = stdout_path.stat().st_size
    record = payload["records"][0]
    record_unsigned = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = canonical_sha256(record_unsigned)
    checkpoint_unsigned = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    payload["checkpoint_sha256"] = canonical_sha256(checkpoint_unsigned)
    checkpoint_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        scaling.ScalingContractError,
        match="evaluated.*receipt|result.*inventory|result file hash",
    ):
        scaling.load_publication_scaling_evidence(repository, round_dir)


def _rehashed_dataset_receipt(
    receipt: dict[str, object], retained: Path
) -> dict[str, object]:
    from maskimpute_benchmark.protocol import canonical_sha256

    changed = json.loads(json.dumps(receipt))
    changed["moderate_output_file_sha256"] = hashlib.sha256(
        retained.read_bytes()
    ).hexdigest()
    changed["moderate_output_size_bytes"] = retained.stat().st_size
    unsigned = {key: value for key, value in changed.items() if key != "receipt_sha256"}
    changed["receipt_sha256"] = canonical_sha256(unsigned)
    return changed


def _replace_h5ad_dataset_with_external_raw(
    retained: Path,
    dataset_path: str,
    raw_path: Path,
) -> None:
    import h5py

    with h5py.File(retained, "r+") as handle:
        original = handle[dataset_path]
        values = original[...]
        shape = original.shape
        dtype = original.dtype
        attributes = dict(original.attrs)
        parent = original.parent
        name = original.name.rsplit("/", maxsplit=1)[-1]
        del parent[name]
        replacement = parent.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            external=[(str(raw_path), 0, int(values.nbytes))],
        )
        replacement[...] = values
        for key, value in attributes.items():
            replacement.attrs[key] = value
        assert replacement.external is not None


def _replace_h5ad_dataset_with_virtual_source(
    retained: Path,
    dataset_path: str,
    source_path: Path,
) -> None:
    import h5py

    with h5py.File(retained, "r") as handle:
        original = handle[dataset_path]
        values = original[...]
        shape = original.shape
        dtype = original.dtype
        attributes = dict(original.attrs)
    with h5py.File(source_path, "w") as source:
        source.create_dataset("source", data=values)
    with h5py.File(retained, "r+") as handle:
        parent_path, name = dataset_path.rsplit("/", maxsplit=1)
        parent = handle[parent_path]
        del parent[name]
        layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
        layout[...] = h5py.VirtualSource(
            str(source_path),
            "source",
            shape=shape,
        )
        replacement = parent.create_virtual_dataset(name, layout)
        for key, value in attributes.items():
            replacement.attrs[key] = value
        assert replacement.is_virtual


def test_scaling_h5ad_preflight_rejects_external_raw_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anndata as ad

    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
    )

    plan = _plan()
    receipt = _dataset_receipt(tmp_path)
    retained = tmp_path / str(receipt["moderate_output_path"])
    _replace_h5ad_dataset_with_external_raw(
        retained,
        "obs/draw",
        tmp_path / "draw.external-raw.bin",
    )
    changed = _rehashed_dataset_receipt(receipt, retained)
    original_read = ad.read_h5ad
    read_called = False

    def observe_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        return original_read(*args, **kwargs)

    monkeypatch.setattr(ad, "read_h5ad", observe_read)

    with pytest.raises(
        ScalingContractError,
        match="H5AD HDF5.*external|H5AD HDF5.*layout",
    ):
        ScalingResultStore(tmp_path, plan).append_dataset(changed)

    assert read_called is False


def test_fresh_scaling_store_rejects_virtual_dataset_before_retained_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anndata as ad

    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
    )

    plan = _plan()
    receipt = _dataset_receipt(tmp_path)
    store = ScalingResultStore(tmp_path, plan)
    store.append_dataset(receipt)
    retained = tmp_path / str(receipt["moderate_output_path"])
    _replace_h5ad_dataset_with_virtual_source(
        retained,
        "X/data",
        tmp_path / "virtual-source.h5",
    )
    changed = _rehashed_dataset_receipt(receipt, retained)

    def mutate(payload: dict[str, object]) -> None:
        payload["datasets"][0] = changed

    _rewrite_scaling_checkpoint(store.checkpoint_path, mutate)
    original_read = ad.read_h5ad
    read_paths: list[Path] = []

    def observe_read(path, *args, **kwargs):
        read_paths.append(Path(path).resolve())
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(ad, "read_h5ad", observe_read)

    with pytest.raises(
        ScalingContractError,
        match="H5AD HDF5.*virtual|H5AD HDF5.*layout",
    ):
        ScalingResultStore(tmp_path, plan).load(force_validate=True)

    assert retained.resolve() not in read_paths


def test_fresh_scaling_store_accepts_canonical_generated_h5ad_layout(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.scaling import ScalingResultStore

    plan = _plan()
    receipt = _dataset_receipt(tmp_path)
    ScalingResultStore(tmp_path, plan).append_dataset(receipt)

    loaded = ScalingResultStore(tmp_path, plan).load(force_validate=True)

    assert loaded is not None
    assert loaded.datasets == (receipt,)


def test_scaling_h5ad_preflight_rejects_extra_rehashed_payload_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anndata as ad
    import h5py
    import numpy as np

    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
    )

    plan = _plan()
    receipt = _dataset_receipt(tmp_path)
    retained = tmp_path / str(receipt["moderate_output_path"])
    with h5py.File(retained, "a") as handle:
        handle.create_dataset("unreferenced_payload", data=np.ones(1))
    changed = _rehashed_dataset_receipt(receipt, retained)
    original_read = ad.read_h5ad
    read_called = False

    def observe_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        return original_read(*args, **kwargs)

    monkeypatch.setattr(ad, "read_h5ad", observe_read)

    with pytest.raises(ScalingContractError, match="HDF5.*structure|H5AD.*structure"):
        ScalingResultStore(tmp_path, plan).append_dataset(changed)

    assert read_called is False


def test_scaling_h5ad_preflight_rejects_oversized_file_before_hash_or_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anndata as ad
    import maskimpute_benchmark.scaling as scaling
    from maskimpute_benchmark.protocol import canonical_sha256

    plan = _plan()
    receipt = _dataset_receipt(tmp_path)
    retained = tmp_path / str(receipt["moderate_output_path"])
    ceiling = scaling._scaling_h5ad_size_ceiling(10_000, 500)
    with retained.open("r+b") as stream:
        stream.truncate(ceiling + 1)
    changed = json.loads(json.dumps(receipt))
    changed["moderate_output_size_bytes"] = ceiling + 1
    unsigned = {key: value for key, value in changed.items() if key != "receipt_sha256"}
    changed["receipt_sha256"] = canonical_sha256(unsigned)
    original_hash = scaling._file_sha256
    original_read = ad.read_h5ad
    hash_called = False
    read_called = False

    def observe_hash(path: Path) -> str:
        nonlocal hash_called
        if path == retained:
            hash_called = True
        return original_hash(path)

    def observe_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        return original_read(*args, **kwargs)

    monkeypatch.setattr(scaling, "_file_sha256", observe_hash)
    monkeypatch.setattr(ad, "read_h5ad", observe_read)

    with pytest.raises(
        scaling.ScalingContractError,
        match="H5AD.*size|dataset.*size.*bound",
    ):
        scaling.ScalingResultStore(tmp_path, plan).append_dataset(changed)

    assert hash_called is False
    assert read_called is False


def test_scaling_h5ad_preflight_rejects_malformed_matrix_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anndata as ad
    import h5py
    import numpy as np

    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
    )

    plan = _plan()
    receipt = _dataset_receipt(tmp_path)
    retained = tmp_path / str(receipt["moderate_output_path"])
    with h5py.File(retained, "r+") as handle:
        handle["X"].attrs["shape"] = np.asarray([9_999, 500], dtype=np.int64)
    changed = _rehashed_dataset_receipt(receipt, retained)
    original_read = ad.read_h5ad
    read_called = False

    def observe_read(*args, **kwargs):
        nonlocal read_called
        read_called = True
        return original_read(*args, **kwargs)

    monkeypatch.setattr(ad, "read_h5ad", observe_read)

    with pytest.raises(ScalingContractError, match="HDF5.*structure|H5AD.*structure"):
        ScalingResultStore(tmp_path, plan).append_dataset(changed)

    assert read_called is False


def test_scaling_h5ad_validation_rejects_file_identity_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anndata as ad

    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
    )

    plan = _plan()
    receipt = _dataset_receipt(tmp_path)
    retained = tmp_path / str(receipt["moderate_output_path"])
    original_read = ad.read_h5ad

    def change_identity_after_read(*args, **kwargs):
        dataset = original_read(*args, **kwargs)
        retained.touch()
        return dataset

    monkeypatch.setattr(ad, "read_h5ad", change_identity_after_read)

    with pytest.raises(ScalingContractError, match="identity.*changed|changed.*read"):
        ScalingResultStore(tmp_path, plan).append_dataset(receipt)


def test_scaling_cli_requires_only_the_canonical_round_locator() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "scripts/run_scaling_panel.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "round_dir" in completed.stdout
    assert "--output-dir" not in completed.stdout


def test_scaling_accuracy_matches_canonical_metrics_without_cell_quadratic_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from maskimpute_benchmark.metrics import reconstruction_metrics
    import maskimpute_benchmark.scaling as module

    rng = np.random.default_rng(19)
    truth = rng.poisson(2.0, size=(24, 7)).astype(float)
    observed = rng.binomial(truth.astype(int), 0.65).astype(float)
    imputed = observed + rng.uniform(0.0, 0.2, size=observed.shape)
    canonical = reconstruction_metrics(imputed, observed, truth)
    calls: list[tuple[tuple[int, ...], bool]] = []
    original = np.corrcoef

    def guarded(value, *args, **kwargs):
        calls.append((np.asarray(value).shape, kwargs.get("rowvar", True)))
        assert kwargs.get("rowvar") is False
        assert np.asarray(value).shape == truth.shape
        return original(value, *args, **kwargs)

    monkeypatch.setattr(module.np, "corrcoef", guarded)
    bounded = module._bounded_scaling_metric_values(imputed, observed, truth)
    aliases = {
        "mse": "mse",
        "mse_dropout": "mse_dropout",
        "mse_pre_dropout_zero": "mse_pre_dropout_zero",
        "mse_nonzero": "mse_nonzero",
        "gnrmse": "gnrmse",
        "mean_distortion": "mean_distortion",
        "variance_distortion": "variance_distortion",
        "mean_gene_wasserstein_distance": "mean_gene_wasserstein_distance",
        "corr_err": "corr_err",
        "n_corr_genes": "n_corr_genes",
    }
    assert tuple(bounded) == module._SCALING_ACCURACY_METRICS
    for observed_name, canonical_name in aliases.items():
        value, n, reason = bounded[observed_name]
        expected = canonical[canonical_name]
        assert value == pytest.approx(expected.value)
        assert n == expected.n
        assert reason == expected.reason
    assert calls == [(truth.shape, False), (truth.shape, False)]
