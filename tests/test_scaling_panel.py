from __future__ import annotations

from dataclasses import asdict
import hashlib
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
            execution_authority_sha256=SHA_C,
            execution_environment_sha256=SHA_D,
            implementation_source_sha256="e" * 64,
        )


def test_scaling_record_keeps_metrics_and_hashes_but_not_dense_outputs() -> None:
    import numpy as np

    from maskimpute_benchmark.runner import (
        EvaluatedAttempt,
        LongFormMetric,
        RawRunResult,
    )
    from maskimpute_benchmark.scaling import scaling_attempt_record

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
        status="completed",
        reason=None,
        runtime_seconds=1.25,
        peak_rss_bytes=1024,
        peak_gpu_bytes=0,
        rss_measurement="linux_process_tree_peak_rss",
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
    attempt = EvaluatedAttempt(
        run=run,
        metrics=(metric,),
        stdout=b"out",
        stderr=b"err",
        native_output=np.ones((2, 2)),
        native_output_scale="raw_counts",
        evaluator_output=np.ones((2, 2)),
    )

    record = scaling_attempt_record(attempt, cells=10_000, accuracy_enabled=True)

    assert record["run"]["native_output_retention"] == "hash_only"
    assert record["run"]["evaluator_output_retention"] == "hash_only"
    assert "native_output" not in record
    assert "evaluator_output" not in record
    assert record["metrics"] == [metric.to_dict()]
    assert record["stdout"] == b"out"
    assert record["stderr"] == b"err"


def _dataset_receipt(output_dir: Path, cells: int = 10_000) -> dict[str, object]:
    from maskimpute_benchmark.protocol import canonical_sha256

    relative = f"generated/scaling-{cells}/dataset/moderate.h5ad"
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixed-moderate-h5ad-fixture")
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "cells": cells,
        "genes": 500,
        "namespace": f"scaling-{cells}",
        "mechanism": "symsim",
        "technical_view": "moderate",
        "dataset_id": f"dataset-{cells}",
        "independent_unit_id": f"scaling-{cells}-draw-01",
        "dataset_sha256": SHA_A,
        "truth_sha256": SHA_B,
        "moderate_output_path": relative,
        "moderate_output_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "moderate_output_size_bytes": path.stat().st_size,
        "severe_dataset_sha256": SHA_C,
        "severe_output_file_sha256": SHA_D,
        "severe_output_size_bytes": 123,
        "moderate_native_manifest_sha256": "e" * 64,
        "severe_native_manifest_sha256": "f" * 64,
        "native_files_sha256": "1" * 64,
        "protocol_sha256": "2" * 64,
        "design_sha256": "3" * 64,
        "seed_source_sha256": "4" * 64,
        "seeds": {"biological": 1, "moderate": 2, "severe": 3},
        "severe_retention": "discarded_after_receipt",
        "native_retention": "discarded_after_receipt",
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}


def _first_attempt(plan):
    from maskimpute_benchmark.runner import (
        EvaluatedAttempt,
        LongFormMetric,
        RawRunResult,
    )
    from maskimpute_benchmark.scaling import _SCALING_ACCURACY_METRICS

    entry = plan.entries[0]
    run = RawRunResult(
        run_id=entry.run_id,
        method_id=entry.method_id,
        dataset_id="dataset-10000",
        source_dataset_sha256=SHA_A,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        model_seed=entry.model_seed,
        configuration_id=entry.configuration_id,
        configuration_sha256=entry.configuration_sha256,
        configuration_kind=entry.configuration_kind,
        requires_count_score=entry.requires_count_score,
        requires_calibration=entry.requires_calibration,
        method_input_sha256=SHA_C,
        dataset_qc_policy_sha256=SHA_D,
        excluded_cell_count=0,
        excluded_cell_ids_sha256="e" * 64,
        retained_cell_count=10_000,
        retained_cell_ids_sha256="f" * 64,
        status="completed",
        reason=None,
        runtime_seconds=1.25,
        peak_rss_bytes=1024,
        peak_gpu_bytes=0,
        rss_measurement="linux_process_tree_peak_rss",
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
    metrics = tuple(
        LongFormMetric(
            mechanism="symsim",
            biological_id="draw-01",
            technical_view="moderate",
            dataset_id="dataset-10000",
            method=entry.method_id,
            model_seed=entry.model_seed,
            configuration_id=entry.configuration_id,
            configuration_sha256=entry.configuration_sha256,
            metric=name,
            value=0.5,
            n=5_000_000,
            status="completed",
            reason=None,
        )
        for name in _SCALING_ACCURACY_METRICS
    )
    return EvaluatedAttempt(
        run=run,
        metrics=metrics,
        stdout=b"out",
        stderr=b"err",
        native_output=None,
        native_output_scale="raw_counts",
        evaluator_output=None,
    )


def test_scaling_store_resumes_exact_prefix_without_dense_outputs(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    report = store.append_dataset(_dataset_receipt(tmp_path))
    assert report.status == "running"
    attempt = _first_attempt(plan)
    report = store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(attempt, cells=10_000, accuracy_enabled=True),
    )

    assert len(report.datasets) == 1
    assert len(report.records) == 1
    run = report.records[0]["run"]
    assert run["native_output_retention"] == "hash_only"
    assert run["evaluator_output_retention"] == "hash_only"
    assert run["stdout_path"].endswith(".stdout")
    assert "native_output_path" not in run
    assert store.load() == report


def test_scaling_store_rejects_log_tampering(tmp_path: Path) -> None:
    from maskimpute_benchmark.scaling import (
        ScalingContractError,
        ScalingResultStore,
        scaling_attempt_record,
    )

    plan = _plan()
    store = ScalingResultStore(tmp_path, plan)
    store.append_dataset(_dataset_receipt(tmp_path))
    report = store.append_attempt(
        plan.entries[0],
        scaling_attempt_record(
            _first_attempt(plan), cells=10_000, accuracy_enabled=True
        ),
    )
    stdout = tmp_path / report.records[0]["run"]["stdout_path"]
    stdout.write_bytes(b"tampered")

    with pytest.raises(ScalingContractError, match="stdout integrity"):
        store.load()


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
