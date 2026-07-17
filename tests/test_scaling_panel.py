from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _configurations():
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    registry = load_method_registry(REPOSITORY / "study/methods.json")
    values = []
    for method_id in ("observed", "maskimpute", "dca", "scvi", "magic"):
        spec = registry.by_id(method_id)
        if method_id == "maskimpute":
            payload = {
                "method_version": "v27",
                "decoder": "scaled_gaussian",
                "encoder_mode": "explicit_mask",
                "output_policy": "selective",
                "score_policy": "direct_cross_fitted_count_score",
                "hyperparameters": {
                    "hidden_dim": 128,
                    "latent_dim": 32,
                    "epochs": 20,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "mask_rate": 0.15,
                    "natural_zero_weight": 0.5,
                    "gate_temperature": 0.1,
                },
            }
            values.append(
                AuthorizedConfiguration.create(
                    method_id="maskimpute",
                    configuration_id="v27-scaling-fixture",
                    kind="candidate_search",
                    payload=payload,
                    requires_count_score=True,
                    requires_calibration=False,
                )
            )
        else:
            values.append(AuthorizedConfiguration.registry_default(spec))
    return registry, tuple(values)


def _plan():
    from maskimpute_benchmark.scaling import build_scaling_plan, load_scaling_contract

    contract = load_scaling_contract(REPOSITORY / "study/scaling_panel.json")
    registry, configurations = _configurations()
    return build_scaling_plan(
        contract,
        registry,
        configurations,
        frozen_method_sha256=SHA_A,
        method_registry_file_sha256=SHA_B,
        protocol_file_sha256=hashlib.sha256(
            (REPOSITORY / "study/protocol.json").read_bytes()
        ).hexdigest(),
        execution_authority_sha256=SHA_C,
        execution_environment_sha256=SHA_D,
        implementation_source_sha256="e" * 64,
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
        AuthorizedConfiguration.registry_default(registry.by_id("maskimpute"))
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
    metric_values = _bounded_scaling_metric_values(
        evaluator_output, observed, truth
    )
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
    report = resumed.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan, tmp_path, receipt, stdout=b"retried execution"),
            cells=10_000,
            accuracy_enabled=True,
        ),
    )
    assert len(report.records) == 1


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
