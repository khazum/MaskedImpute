from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_canonical(path: Path, value: object) -> str:
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _cell_id_sha256(cell_ids: tuple[str, ...]) -> str:
    payload = bytearray(b"maskimpute-external-cell-ids-v1\0")
    payload.extend(struct.pack("<Q", len(cell_ids)))
    for cell_id in cell_ids:
        encoded = cell_id.encode("utf-8")
        payload.extend(struct.pack("<Q", len(encoded)))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def _dataset(path: Path) -> tuple[ad.AnnData, tuple[str, ...], tuple[str, ...]]:
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    counts = np.asarray(
        [
            [12, 8, 1, 0],
            [11, 7, 1, 0],
            [10, 8, 2, 0],
            [0, 1, 8, 12],
            [0, 1, 7, 11],
            [0, 2, 8, 10],
        ],
        dtype=np.int64,
    )
    cells = tuple(f"cell-{index}" for index in range(1, 7))
    genes = tuple(f"gene-{index}" for index in range(1, 5))
    dataset = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "dataset_id": ["dev-symsim-01"] * 6,
                "mechanism": ["symsim"] * 6,
                "condition": ["moderate"] * 6,
                "biological_id": ["draw-01"] * 6,
                "technical_view": ["moderate"] * 6,
                "draw": np.ones(6, dtype=np.int64),
                "library_size": np.sum(counts, axis=1, dtype=np.int64),
                "group": ["pop-1"] * 3 + ["pop-2"] * 3,
            },
            index=cells,
        ),
        var=pd.DataFrame(
            {
                "marker_group_1": [True, True, False, False],
                "marker_group_2": [False, False, True, True],
            },
            index=genes,
        ),
        layers={"pre_capture_counts": counts + 1},
    )
    dataset.uns.update(
        {
            "truth_kind": "exact_pre_capture",
            "primary_truth_layer": "pre_capture_counts",
            "provenance": {
                "source": "test-fixture",
                "source_sha256": "1" * 64,
                "software": "test",
                "software_version": "1",
                "parameters": {},
                "seeds": {},
            },
            "normalization": {"input": "raw_umi_counts", "size_factor": "none"},
        }
    )
    dataset.write_h5ad(path)
    persisted = ad.read_h5ad(path)
    assert benchmark_dataset_sha256(persisted) == benchmark_dataset_sha256(dataset)
    return persisted, cells, genes


def _common_output(dataset: ad.AnnData) -> np.ndarray:
    counts = np.asarray(dataset.X, dtype=np.float64)
    libraries = np.sum(counts, axis=1)
    return np.log2(counts * (10_000.0 / libraries)[:, None] + 1.0)


def _evaluator_output_sha256(run: dict[str, object], raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"maskimpute-evaluator-log2-cp10k-output-v1\0")
    digest.update(
        _canonical_bytes(
            {
                "run_id": run["run_id"],
                "method_input_sha256": run["method_input_sha256"],
                "retained_cell_ids_sha256": run["retained_cell_ids_sha256"],
                "shape": run["evaluator_output_shape"],
                "dtype": "<f8",
                "scale": "log2_cp10k_plus_1",
            }
        )
    )
    digest.update(raw)
    return digest.hexdigest()


def _test_configuration_authority():
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    candidate = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="default",
        kind="candidate_search",
        payload={
            "configuration_id": "default",
            "method_id": "maskimpute",
            "variant": "downstream-test",
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    registry = load_method_registry(Path("study/methods.json"))
    magic = AuthorizedConfiguration.registry_default(registry.by_id("magic"))
    return candidate, magic


def _run(
    *,
    run_id: str,
    method_id: str,
    dataset_sha256: str,
    cell_ids: tuple[str, ...],
    status: str,
    reason: str | None,
) -> dict[str, object]:
    candidate, magic = _test_configuration_authority()
    configuration = candidate if method_id == "maskimpute" else magic
    return {
        "run_id": run_id,
        "method_id": method_id,
        "dataset_id": "dev-symsim-01",
        "source_dataset_sha256": dataset_sha256,
        "mechanism": "symsim",
        "biological_id": "draw-01",
        "technical_view": "moderate",
        "model_seed": 42,
        "configuration_id": configuration.configuration_id,
        "configuration_sha256": configuration.configuration_sha256,
        "configuration_kind": configuration.kind,
        "requires_count_score": False,
        "requires_calibration": False,
        "method_input_sha256": "3" * 64,
        "dataset_qc_policy_sha256": "4" * 64,
        "excluded_cell_count": 0,
        "excluded_cell_ids_sha256": "5" * 64,
        "retained_cell_count": len(cell_ids),
        "retained_cell_ids_sha256": _cell_id_sha256(cell_ids),
        "retained_gene_count": 0,
        "observed_zero_count": 0,
        "status": status,
        "reason": reason,
        "runtime_seconds": 1.0,
        "peak_rss_bytes": 1,
        "peak_gpu_bytes": 0,
        "rss_measurement": "test_measurement",
        "gpu_measurement": "not_measured",
        "calibration_artifact_sha256": None,
        "calibration_context_sha256": None,
        "calibration_training_manifest_sha256s": [],
        "calibration_held_out_manifest_sha256s": [],
        "calibration_fold_calibrator_sha256": None,
        "stdout_sha256": "6" * 64,
        "stderr_sha256": "7" * 64,
        "native_output_sha256": None,
        "evaluator_output_sha256": None,
        "stdout_path": f"runs/{run_id}.stdout",
        "stdout_file_sha256": "8" * 64,
        "stderr_path": f"runs/{run_id}.stderr",
        "stderr_file_sha256": "9" * 64,
        "native_output_path": None,
        "native_output_file_sha256": None,
        "native_output_shape": None,
        "native_output_dtype": None,
        "native_output_scale": None,
        "evaluator_output_path": None,
        "evaluator_output_file_sha256": None,
        "evaluator_output_shape": None,
        "evaluator_output_dtype": None,
        "evaluator_scale": None,
    }


def _current_prezero_evidence(run: dict[str, object]) -> dict[str, object]:
    """Return the exact persisted score-evidence envelope used by current stores."""

    identity = {
        name: run[name]
        for name in (
            "run_id",
            "method_id",
            "dataset_id",
            "source_dataset_sha256",
            "mechanism",
            "biological_id",
            "technical_view",
            "model_seed",
            "configuration_id",
            "configuration_sha256",
            "method_input_sha256",
            "retained_cell_ids_sha256",
        )
    }
    completed_score = run["method_id"] == "maskimpute" and run["status"] == "completed"
    status = run["status"] if run["method_id"] == "maskimpute" else "not_applicable"
    reason = (
        run["reason"]
        if run["method_id"] == "maskimpute"
        else "method_does_not_emit_p_pre_zero"
    )
    policy = (
        {
            "schema_version": 2,
            "probability_semantics": (
                "pre_capture_count_is_zero_given_observed_counts"
            ),
            "evaluation_domain": "observed_zero_entries_only",
            "score_source": "direct",
            "score_artifact_sha256": "a" * 64,
            "score_input_sha256": "b" * 64,
            "score_config_sha256": "c" * 64,
            "calibration_file_sha256": "d" * 64,
            "calibration_payload_sha256": "e" * 64,
            "calibration_algorithm": "identity",
            "calibration_scope": "retained_all_development",
            "calibration_equivalence_reason": "direct_score_requires_no_calibration",
        }
        if completed_score
        else None
    )
    matrix = (
        {
            "shape": [run["retained_cell_count"], run["retained_gene_count"]],
            "dtype": "<f8",
            "content_sha256": "f" * 64,
            "semantic_sha256": "0" * 64,
        }
        if completed_score
        else {
            "shape": None,
            "dtype": None,
            "content_sha256": None,
            "semantic_sha256": None,
        }
    )
    metric_status = "completed" if completed_score else status
    metrics = {
        name: {
            "value": 0.5 if completed_score else None,
            "n": run["observed_zero_count"],
            "status": metric_status,
            "reason": None if completed_score else reason,
        }
        for name in (
            "auroc",
            "auprc",
            "brier",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
            "ece",
        )
    }
    overall = {
        "stratum_type": "overall",
        "label": "all_observed_zeros",
        "lower": None,
        "upper": None,
        "n": run["observed_zero_count"],
        "metrics": metrics,
        "reliability_bins": [],
    }
    strata = {
        "library_size_quartiles": [
            {
                **overall,
                "stratum_type": "library_size_quartiles",
                "label": f"Q{index}",
                "lower": None,
                "upper": None,
                "n": 0,
                "metrics": {name: {**value, "n": 0} for name, value in metrics.items()},
            }
            for index in range(1, 5)
        ],
        "truth_expression_bins": [
            {
                **overall,
                "stratum_type": "truth_expression_bins",
                "label": label,
                "lower": lower,
                "upper": upper,
                "n": 0,
                "metrics": {name: {**value, "n": 0} for name, value in metrics.items()},
            }
            for label, lower, upper in (
                ("[0,1)", 0.0, 1.0),
                ("[1,2)", 1.0, 2.0),
                ("[2,4)", 2.0, 4.0),
                ("[4,inf)", 4.0, None),
            )
        ],
    }
    body = {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "identity": identity,
        "truth_kind": "exact_pre_capture",
        "matrix": matrix,
        "policy": policy,
        "policy_sha256": None if policy is None else _sha256_payload(policy),
        "overall": overall,
        "strata": strata,
    }
    storage = (
        {
            "encoding": "zlib_raw_f64_v1",
            "compression_level": 6,
            "path": f"runs/{run['run_id']}.p_pre_zero.f64.zlib",
            "compressed_sha256": "1" * 64,
            "compressed_nbytes": 1,
            "uncompressed_sha256": "f" * 64,
            "uncompressed_nbytes": (
                run["retained_cell_count"] * run["retained_gene_count"] * 8
            ),
        }
        if completed_score
        else {
            "encoding": None,
            "compression_level": None,
            "path": None,
            "compressed_sha256": None,
            "compressed_nbytes": None,
            "uncompressed_sha256": None,
            "uncompressed_nbytes": None,
        }
    )
    return {**body, "evidence_sha256": _sha256_payload(body), "storage": storage}


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _development_source(tmp_path: Path):
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset_path = tmp_path / "dataset.h5ad"
    dataset, cells, _genes = _dataset(dataset_path)
    dataset_sha = benchmark_dataset_sha256(dataset)
    dataset_authority = _dataset_binding(dataset_path, cells)
    source = tmp_path / "development"
    source.mkdir()
    output = np.asarray(_common_output(dataset), dtype="<f8", order="C")
    raw = output.tobytes(order="C")
    output_path = source / "runs" / "run-completed.log2-cp10k-f64"
    output_path.parent.mkdir()
    output_path.write_bytes(raw)
    completed = _run(
        run_id="run-completed",
        method_id="maskimpute",
        dataset_sha256=dataset_sha,
        cell_ids=cells,
        status="completed",
        reason=None,
    )
    _apply_dataset_authority(completed, dataset_authority)
    completed.update(
        {
            "evaluator_output_path": "runs/run-completed.log2-cp10k-f64",
            "evaluator_output_file_sha256": hashlib.sha256(raw).hexdigest(),
            "evaluator_output_shape": list(output.shape),
            "evaluator_output_dtype": "<f8",
            "evaluator_scale": "log2_cp10k_plus_1",
        }
    )
    completed["evaluator_output_sha256"] = _evaluator_output_sha256(completed, raw)
    failed = _run(
        run_id="run-failed",
        method_id="magic",
        dataset_sha256=dataset_sha,
        cell_ids=cells,
        status="failed",
        reason="adapter_nonzero_exit",
    )
    _apply_dataset_authority(failed, dataset_authority)
    records = [
        {
            "run": completed,
            "metrics": [],
            "p_pre_zero_evidence": _current_prezero_evidence(completed),
        },
        {
            "run": failed,
            "metrics": [],
            "p_pre_zero_evidence": _current_prezero_evidence(failed),
        },
    ]
    body = {
        "schema_version": 1,
        "plan_sha256": "4" * 64,
        "input_hashes": {"dataset_manifest_sha256": "5" * 64},
        "planned_run_count": 2,
        "status": "completed",
        "evaluation_scope": "reconstruction_only",
        "selection_complete": False,
        "selection_blockers": ["downstream_evidence_pending"],
        "records": records,
        "budget": {},
    }
    checkpoint = {**body, "checkpoint_sha256": canonical_sha256(body)}
    _write_canonical(source / "checkpoint.json", checkpoint)
    return source, dataset_path, cells, output_path


def _dataset_binding(dataset_path: Path, cells: tuple[str, ...]):
    from maskimpute_benchmark.downstream_evidence import bind_evaluator_dataset

    return bind_evaluator_dataset(dataset_path, retained_cell_ids=cells)


def _apply_dataset_authority(run: dict[str, object], binding: object) -> None:
    for name in (
        "mechanism",
        "biological_id",
        "technical_view",
        "method_input_sha256",
        "dataset_qc_policy_sha256",
        "excluded_cell_count",
        "excluded_cell_ids_sha256",
        "retained_cell_count",
        "retained_cell_ids_sha256",
        "retained_gene_count",
        "observed_zero_count",
    ):
        run[name] = getattr(binding, name)


def _development_source_plan(source: Path) -> SimpleNamespace:
    checkpoint = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    return SimpleNamespace(
        plan_sha256=checkpoint["plan_sha256"],
        input_hashes=checkpoint["input_hashes"],
        entries=tuple(
            {**stored["run"], "ordinal": ordinal}
            for ordinal, stored in enumerate(checkpoint["records"], start=1)
        ),
    )


def _evaluation_manifest(
    path: Path,
    *,
    base_plan: object,
    revision_plan: object,
) -> tuple[str, str, str, str]:
    from maskimpute_benchmark.protocol import canonical_sha256

    def reconstruction(plan: object) -> dict[str, object]:
        return {
            "checkpoint_path": str(
                Path(plan.source_root).relative_to(path.parent) / "checkpoint.json"
            ),
            "checkpoint_file_sha256": plan.source_manifest_file_sha256,
            "checkpoint_sha256": plan.source_manifest_payload_sha256,
            "plan_sha256": plan.source_plan_sha256,
            "input_hashes": json.loads(
                (Path(plan.source_root) / "checkpoint.json").read_text(encoding="utf-8")
            )["input_hashes"],
            "raw_artifacts": [],
        }

    base = reconstruction(base_plan)
    revision = reconstruction(revision_plan)
    body = {
        "schema_version": 1,
        "reconstruction": base,
        "revisions": [{"version": "v28", "reconstruction": revision}],
    }
    payload = {**body, "manifest_sha256": canonical_sha256(body)}
    file_sha = _write_canonical(path, payload)
    return (
        file_sha,
        payload["manifest_sha256"],
        canonical_sha256(base),
        canonical_sha256(revision),
    )


def test_prepared_runner_panel_bridge_binds_persisted_dataset_paths(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        bind_prepared_evaluator_panel,
    )

    dataset_path = tmp_path / "dataset.h5ad"
    dataset_object, cells, _genes = _dataset(dataset_path)
    from maskimpute_benchmark.methods import prepare_method_input
    from maskimpute_benchmark.schema import make_inference_view

    authority = _dataset_binding(dataset_path, cells)
    method_input = prepare_method_input(make_inference_view(dataset_object))
    runner_binding = SimpleNamespace(
        dataset_id="dev-symsim-01", output_path="dataset.h5ad"
    )
    prepared = {
        "dev-symsim-01": SimpleNamespace(
            binding=SimpleNamespace(
                mechanism=authority.mechanism,
                biological_id=authority.biological_id,
                technical_view=authority.technical_view,
            ),
            audit=SimpleNamespace(
                retained_cell_ids=cells,
                excluded_cell_count=authority.excluded_cell_count,
                excluded_cell_ids_sha256=authority.excluded_cell_ids_sha256,
                retained_cell_count=authority.retained_cell_count,
                retained_cell_ids_sha256=authority.retained_cell_ids_sha256,
            ),
            method_input=method_input,
        )
    }

    bindings = bind_prepared_evaluator_panel(
        (runner_binding,), prepared, dataset_root=tmp_path
    )

    assert len(bindings) == 1
    assert bindings[0].dataset_id == "dev-symsim-01"
    assert bindings[0].retained_cell_ids == cells
    assert bindings[0].path == str(dataset_path.absolute())


def test_plan_rejects_rehashed_biological_identity_outside_bound_dataset_authority(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"].update(
        {
            "mechanism": "semisynthetic",
            "biological_id": "forged-draw",
            "technical_view": "forged-view",
        }
    )
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="dataset authority"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_current_runner_record_schema_is_accepted_without_field_aliases(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import build_downstream_evidence_plan
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    binding = _dataset_binding(dataset_path, cells)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for stored in checkpoint["records"]:
        run = stored["run"]
        run["retained_gene_count"] = binding.retained_gene_count
        run["observed_zero_count"] = binding.observed_zero_count
        stored["p_pre_zero_evidence"] = _current_prezero_evidence(run)
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(binding,),
        configurations=_test_configuration_authority(),
    )

    assert len(plan.entries) == 2


def test_source_adapter_rejects_retired_prezero_policy_field_alias(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    evidence = checkpoint["records"][0]["p_pre_zero_evidence"]
    policy = evidence["policy"]
    policy["calibration_artifact_sha256"] = policy.pop("calibration_file_sha256")
    policy.pop("calibration_payload_sha256")
    evidence["policy_sha256"] = canonical_sha256(policy)
    evidence_body = {
        key: value
        for key, value in evidence.items()
        if key not in {"evidence_sha256", "storage"}
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence_body)
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="score policy schema"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_selection_primary_plan_rejects_seed_drift_from_bound_source_plan(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    source_plan = _development_source_plan(source)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"]["model_seed"] = 999
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="source plan authority"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            evidence_scope="selection_primary",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
            source_plan=source_plan,
        )


def test_selection_primary_scope_excludes_only_nonselection_maskimpute_ablations(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    ablation = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="no-gate",
        kind="ablation",
        payload={
            "configuration_id": "no-gate",
            "method_id": "maskimpute",
            "variant": "downstream-supplementary-test",
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    ablation_run = dict(checkpoint["records"][1]["run"])
    ablation_run.update(
        {
            "run_id": "run-maskimpute-ablation",
            "method_id": "maskimpute",
            "configuration_id": ablation.configuration_id,
            "configuration_sha256": ablation.configuration_sha256,
            "configuration_kind": ablation.kind,
        }
    )
    checkpoint["records"].append(
        {
            "run": ablation_run,
            "metrics": [],
            "p_pre_zero_evidence": _current_prezero_evidence(ablation_run),
        }
    )
    checkpoint["planned_run_count"] = 3
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)
    configurations = (*_test_configuration_authority(), ablation)

    primary = build_downstream_evidence_plan(
        source,
        source_kind="development",
        evidence_scope="selection_primary",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=configurations,
        source_plan=_development_source_plan(source),
    )
    supplementary = build_downstream_evidence_plan(
        source,
        source_kind="development",
        evidence_scope="supplementary_nonselection",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=configurations,
        source_plan=_development_source_plan(source),
    )

    assert [entry.run_id for entry in primary.entries] == [
        "run-completed",
        "run-failed",
    ]
    assert [entry.run_id for entry in supplementary.entries] == [
        "run-maskimpute-ablation"
    ]
    destination = tmp_path / "selection-primary-downstream"
    manifest = run_downstream_evidence(primary, destination)
    loaded = load_downstream_evidence_manifest(destination)

    assert manifest["planned_denominator_count"] == 2
    assert loaded.planned_denominator_count == 2


def test_revision_downstream_bundle_covers_base_and_activated_checkpoint(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DevelopmentSourcePlan,
        build_downstream_evidence_plan,
        combine_development_downstream_evidence_plans,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.revisions import development_selection_stage_paths
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    base_root = tmp_path / "base"
    revision_root = tmp_path / "revision"
    base_root.mkdir()
    revision_root.mkdir()
    base_source, base_dataset, base_cells, _base_output = _development_source(base_root)
    revision_source, _revision_dataset, revision_cells, _revision_output = (
        _development_source(revision_root)
    )
    revision_configuration = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="v28-c01-nb-decoder",
        kind="candidate_search",
        payload={
            "configuration_id": "v28-c01-nb-decoder",
            "method_id": "maskimpute",
            "variant": "revision-downstream-test",
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    checkpoint_path = revision_source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    revised_run = checkpoint["records"][0]["run"]
    revised_run.update(
        {
            "run_id": "run-v28-completed",
            "configuration_id": revision_configuration.configuration_id,
            "configuration_sha256": revision_configuration.configuration_sha256,
            "configuration_kind": revision_configuration.kind,
        }
    )
    output_raw = (
        revision_source / str(revised_run["evaluator_output_path"])
    ).read_bytes()
    revised_run["evaluator_output_sha256"] = _evaluator_output_sha256(
        revised_run, output_raw
    )
    checkpoint["records"][0]["p_pre_zero_evidence"] = _current_prezero_evidence(
        revised_run
    )
    checkpoint_body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    _write_canonical(checkpoint_path, checkpoint)

    base_plan = build_downstream_evidence_plan(
        base_source,
        source_kind="development",
        evidence_scope="selection_primary",
        datasets=(_dataset_binding(base_dataset, base_cells),),
        configurations=_test_configuration_authority(),
        source_plan=_development_source_plan(base_source),
    )
    _base_candidate, magic = _test_configuration_authority()
    revision_plan = build_downstream_evidence_plan(
        revision_source,
        source_kind="development",
        evidence_scope="all",
        datasets=(_dataset_binding(base_dataset, revision_cells),),
        configurations=(revision_configuration, magic),
        source_plan=_development_source_plan(revision_source),
    )
    evaluation_path = tmp_path / "evaluation-v28.json"
    (
        evaluation_file_sha,
        evaluation_payload_sha,
        base_evaluation_sha,
        revision_evaluation_sha,
    ) = _evaluation_manifest(
        evaluation_path,
        base_plan=base_plan,
        revision_plan=revision_plan,
    )
    combined = combine_development_downstream_evidence_plans(
        tmp_path,
        (
            DevelopmentSourcePlan(
                source_id="base",
                plan=base_plan,
                selected_methods=("default", "magic"),
                evaluation_manifest_path="evaluation-v28.json",
                evaluation_manifest_file_sha256=evaluation_file_sha,
                evaluation_manifest_payload_sha256=evaluation_payload_sha,
                evaluation_source_pointer="/reconstruction",
                evaluation_source_sha256=base_evaluation_sha,
            ),
            DevelopmentSourcePlan(
                source_id="v28",
                plan=revision_plan,
                selected_methods=(revision_configuration.configuration_id,),
                evaluation_manifest_path="evaluation-v28.json",
                evaluation_manifest_file_sha256=evaluation_file_sha,
                evaluation_manifest_payload_sha256=evaluation_payload_sha,
                evaluation_source_pointer="/revisions/0/reconstruction",
                evaluation_source_sha256=revision_evaluation_sha,
            ),
        ),
        revision_versions=("v28",),
    )

    assert combined.source_root == str(tmp_path.absolute())
    assert combined.development_revision_versions == ("v28",)
    assert tuple(source.source_id for source in combined.development_sources) == (
        "base",
        "v28",
    )
    assert len(combined.entries) == 3
    assert [entry.configuration_id for entry in combined.entries] == [
        "default",
        "registry-default",
        "v28-c01-nb-decoder",
    ]
    assert (
        len({source.manifest_file_sha256 for source in combined.development_sources})
        == 2
    )
    assert all(
        source.evaluation_manifest_file_sha256 == evaluation_file_sha
        for source in combined.development_sources
    )

    stage_paths = development_selection_stage_paths("v28")
    destination = tmp_path / stage_paths.downstream_directory
    run_downstream_evidence(combined, destination)
    loaded = load_downstream_evidence_manifest(destination)
    assert loaded.planned_denominator_count == 3
    assert loaded.payload["development_revision_versions"] == ["v28"]
    assert len(loaded.payload["development_sources"]) == 2

    from maskimpute_benchmark.selection import (
        attach_downstream_evidence_to_selection_result,
    )

    selection_records = [
        {
            "mechanism": record["mechanism"],
            "biological_id": record["biological_id"],
            "technical_view": record["technical_view"],
            "dataset_id": record["dataset_id"],
            "dataset_sha256": record["dataset_sha256"],
            "method": record["method"],
            "method_sha256": record["method_artifact_sha256"],
            "model_seed": record["model_seed"],
            "metric": "mse",
            "value": 0.0 if record["run_status"] == "completed" else None,
            "status": record["run_status"],
        }
        for record in loaded.records
    ]
    selection_core = {
        "schema_version": 3,
        "revision_versions": ["v28"],
        "dataset_manifest_sha256": "1" * 64,
        "count_score_manifest_sha256": "2" * 64,
        "retained_calibration_artifact_sha256": "3" * 64,
        "evaluation_manifest_sha256": evaluation_file_sha,
        "records": selection_records,
        "orthogonal_intervals": [],
    }
    source_payload = {
        **selection_core,
        "result_sha256": canonical_sha256(selection_core),
    }
    source_file_sha = _write_canonical(
        tmp_path / stage_paths.source_selection_input,
        source_payload,
    )
    upgraded = attach_downstream_evidence_to_selection_result(
        source_payload,
        tmp_path,
        stage_paths.downstream_directory,
    )
    source_bindings = upgraded["downstream_evidence"]["sources"]
    assert [source["source_id"] for source in source_bindings] == ["base", "v28"]
    assert upgraded["downstream_evidence"]["revision_versions"] == ["v28"]
    assert upgraded["downstream_evidence"]["source_selection_input_path"] == (
        stage_paths.source_selection_input
    )
    assert upgraded["downstream_evidence"]["source_selection_input_file_sha256"] == (
        source_file_sha
    )
    assert upgraded["downstream_evidence"]["source_selection_result_sha256"] == (
        source_payload["result_sha256"]
    )

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["revisions"][0]["reconstruction"]["checkpoint_sha256"] = "0" * 64
    evaluation_body = {
        key: value for key, value in evaluation.items() if key != "manifest_sha256"
    }
    evaluation["manifest_sha256"] = canonical_sha256(evaluation_body)
    _write_canonical(evaluation_path, evaluation)
    from maskimpute_benchmark.downstream_evidence import DownstreamEvidenceError

    with pytest.raises(
        DownstreamEvidenceError,
        match="development checkpoint differs",
    ):
        load_downstream_evidence_manifest(destination)


def test_development_downstream_routes_to_latest_fixed_revision_without_fallback(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        development_downstream_revision_version,
    )
    from maskimpute_benchmark.revisions import revision_stage_paths

    assert development_downstream_revision_version(tmp_path) is None
    v28_path = tmp_path / revision_stage_paths("v28").selection_input
    _write_canonical(
        v28_path,
        {"schema_version": 3, "revision_versions": ["v28"]},
    )
    assert development_downstream_revision_version(tmp_path) == "v28"

    v29_path = tmp_path / revision_stage_paths("v29").selection_input
    _write_canonical(
        v29_path,
        {"schema_version": 3, "revision_versions": ["v28", "v29"]},
    )
    assert development_downstream_revision_version(tmp_path) == "v29"

    _write_canonical(
        v29_path,
        {"schema_version": 3, "revision_versions": ["v29"]},
    )
    with pytest.raises(
        DownstreamEvidenceError,
        match="v29 revision selection input identity differs",
    ):
        development_downstream_revision_version(tmp_path)


def test_production_selection_primary_keys_exactly_match_reconstruction_bridge() -> (
    None
):
    from maskimpute_benchmark.development_evaluation import (
        reconstruction_selection_method,
    )
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.runner import (
        DEVELOPMENT_MODEL_SEEDS,
        AuthorizedConfiguration,
        load_runner_authority,
    )

    authority = load_runner_authority()
    registry = load_method_registry(Path("study/methods.json"))
    configured_method_ids = {value.method_id for value in authority.configurations}
    configurations = tuple(authority.configurations) + tuple(
        AuthorizedConfiguration.registry_default(spec)
        for spec in registry.methods
        if spec.execution_scope == "same_input_required"
        and spec.id not in configured_method_ids
    )
    declared = {
        value.configuration_id
        for value in configurations
        if value.kind == "candidate_search"
    } | {
        value.method_id
        for value in configurations
        if value.kind == "registry" or value.method_id == "capacity-matched-ae"
    }
    specification = {value.id: value for value in registry.methods}
    all_keys: set[tuple[object, ...]] = set()
    selection_keys: set[tuple[object, ...]] = set()
    primary_keys: set[tuple[object, ...]] = set()
    for dataset_index in range(16):
        for configuration in configurations:
            spec = specification[configuration.method_id]
            seeds = DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
            run = {
                "configuration_kind": configuration.kind,
                "configuration_id": configuration.configuration_id,
                "method_id": configuration.method_id,
            }
            method = reconstruction_selection_method(run, declared)
            for seed in seeds:
                key = (
                    dataset_index,
                    configuration.method_id,
                    configuration.configuration_id,
                    configuration.configuration_sha256,
                    seed,
                )
                all_keys.add(key)
                if method is not None:
                    selection_keys.add(key)
                    primary_keys.add(key)

    assert primary_keys == selection_keys
    assert len(all_keys - primary_keys) == 5 * 16 * 3 == 240


def test_development_stage_resumes_and_preserves_exact_eight_row_denominators(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        load_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"

    first = run_downstream_evidence(plan, destination, max_denominators=1)
    assert first["status"] == "running"
    assert first["recorded_denominator_count"] == 1
    reloaded_plan = load_downstream_evidence_plan(destination)
    assert reloaded_plan.to_dict() == plan.to_dict()
    complete = run_downstream_evidence(reloaded_plan, destination)
    assert complete["status"] == "completed"
    assert complete["planned_denominator_count"] == 2
    assert complete["endpoint_row_count"] == 16
    assert complete["evaluator_source_sha256"] == plan.evaluator_source_sha256

    manifest = load_downstream_evidence_manifest(destination)
    records = manifest.records
    assert len(records) == 2
    assert all(len(record["endpoints"]) == 8 for record in records)
    assert records[0]["biological_id"] == "draw-01"
    assert records[0]["technical_view"] == "moderate"
    assert records[0]["model_seed"] == 42
    assert records[0]["runner_method_id"] == "maskimpute"
    assert records[0]["method"] == "default"
    assert records[0]["method_artifact_sha256"] == records[0]["configuration_sha256"]
    assert records[1]["method_artifact_sha256"] != records[1]["configuration_sha256"]
    assert {row["upstream_status"] for row in records[0]["endpoints"]} == {"completed"}
    assert {row["status"] for row in records[1]["endpoints"]} == {"failed"}
    assert {row["reason_code"] for row in records[1]["endpoints"]} == {
        "upstream_run_not_completed"
    }
    assert {row["upstream_reason"] for row in records[1]["endpoints"]} == {
        "adapter_nonzero_exit"
    }


def test_resume_revalidates_source_artifacts_and_immutable_record_prefix(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination, max_denominators=1)
    original = output_path.read_bytes()
    output_path.write_bytes(original + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="evaluator output.*checksum"):
        run_downstream_evidence(plan, destination)
    output_path.write_bytes(original)

    record_path = destination / "records" / "00000001.json"
    record_path.write_bytes(record_path.read_bytes() + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="record.*canonical"):
        run_downstream_evidence(plan, destination)


def test_resume_rejects_rehashed_finite_endpoint_value_drift(tmp_path: Path) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)

    record_path = destination / "records/00000001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    endpoint = next(row for row in record["endpoints"] if row["status"] == "completed")
    endpoint["value"] = 0.375 if endpoint["value"] != 0.375 else 0.625
    record_body = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = canonical_sha256(record_body)
    record_file_sha = _write_canonical(record_path, record)

    manifest_path = destination / "downstream_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["sha256"] = record_file_sha
    manifest["records"][0]["record_sha256"] = record["record_sha256"]
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest_body)
    _write_canonical(manifest_path, manifest)

    with pytest.raises(DownstreamEvidenceError, match="endpoint re-evaluation differs"):
        run_downstream_evidence(plan, destination)


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("direction", "endpoint contract differs"),
        ("independent_count", "independent unit differs"),
        ("descriptive_unit", "endpoint contract differs"),
        ("reason_vocabulary", "endpoint contract differs"),
        ("procedure", "endpoint procedure differs"),
        ("family", "endpoint family is unexpected"),
        ("range", "endpoint value is out of range"),
    ],
)
def test_resume_reconstructs_and_validates_endpoint_contract(
    tmp_path: Path, attack: str, message: str
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    record_path = destination / "records/00000001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    row = record["endpoints"][0]
    if attack == "direction":
        row["direction"] = "higher_is_better"
    elif attack == "independent_count":
        row["independent_n"] = 2
    elif attack == "descriptive_unit":
        row["descriptive_unit"] = "cells"
    elif attack == "reason_vocabulary":
        row["status"] = "unavailable"
        row["value"] = None
        row["reason_code"] = "forged_reason"
    elif attack == "procedure":
        row["procedure"] = "forged_procedure"
    elif attack == "family":
        row["family_id"] = "forged_family"
        row["family_size"] = 1
        row["alpha"] = 0.05
    elif attack == "range":
        row["value"] = 1.5
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(attack)
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = canonical_sha256(body)
    _write_canonical(record_path, record)

    with pytest.raises(DownstreamEvidenceError, match=message):
        run_downstream_evidence(plan, destination)


def test_plan_binds_current_evaluator_source_digest(tmp_path: Path) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    assert len(plan.evaluator_source_sha256) == 64
    provisional = replace(
        plan,
        evaluator_source_sha256="0" * 64,
        plan_sha256="0" * 64,
    )
    forged = replace(provisional, plan_sha256=canonical_sha256(provisional.body()))

    with pytest.raises(DownstreamEvidenceError, match="plan sources changed"):
        run_downstream_evidence(forged, tmp_path / "downstream")


def test_final_zlib_source_contract_is_consumed_with_bounded_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset_path = tmp_path / "dataset.h5ad"
    dataset, cells, _genes = _dataset(dataset_path)
    dataset_sha = benchmark_dataset_sha256(dataset)
    repository = tmp_path / "repository"
    round_root = repository / "round-1"
    source = round_root / "results/final/execution"
    output = np.asarray(_common_output(dataset), dtype="<f8", order="C")
    raw = output.tobytes(order="C")
    compressed = zlib.compress(raw, level=6)
    artifact = source / "runs" / "final-run.log2-cp10k-f64.zlib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(compressed)
    run = _run(
        run_id="final-run",
        method_id="maskimpute",
        dataset_sha256=dataset_sha,
        cell_ids=cells,
        status="completed",
        reason=None,
    )
    _apply_dataset_authority(run, _dataset_binding(dataset_path, cells))
    run.update(
        {
            "evaluator_output_path": "runs/final-run.log2-cp10k-f64.zlib",
            "evaluator_output_file_sha256": hashlib.sha256(compressed).hexdigest(),
            "evaluator_output_shape": list(output.shape),
            "evaluator_output_dtype": "<f8",
            "evaluator_scale": "log2_cp10k_plus_1",
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "evaluator_output_uncompressed_nbytes": len(raw),
            "evaluator_output_uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "native_output_retention": "not_available",
        }
    )
    run["evaluator_output_sha256"] = _evaluator_output_sha256(run, raw)
    record = {
        "run": run,
        "metrics": [],
        "p_pre_zero_evidence": _current_prezero_evidence(run),
        "execution_request": None,
    }
    record_path = source / "records" / "00000001.json"
    record_sha = _write_canonical(record_path, record)
    manifest_body = {
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": "6" * 64,
        "input_hashes": {"dataset_manifest_sha256": "7" * 64},
        "planned_run_count": 1,
        "recorded_run_count": 1,
        "records": [
            {
                "ordinal": 1,
                "run_id": "final-run",
                "path": "records/00000001.json",
                "sha256": record_sha,
            }
        ],
        "artifact_storage": {
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "evaluator_output_compression_level": 6,
            "native_output_retention": "omitted_redundant_final_output",
            "p_pre_zero_encoding": "zlib_raw_f64_v1",
            "p_pre_zero_compression_level": 6,
        },
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    manifest_file_sha256 = _write_canonical(
        source / "execution_manifest.json", manifest
    )

    with pytest.raises(
        DownstreamEvidenceError, match="evaluated-round binding is required"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="final",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )
    binding = downstream.EvaluatedRoundBinding(
        repository_root=str(repository.absolute()),
        round_root=str(round_root.absolute()),
        round_id="round-1",
        evaluation_receipt_path="evaluation_receipt.json",
        evaluation_receipt_file_sha256="1" * 64,
        evaluation_receipt_payload_sha256="2" * 64,
        result_manifest_sha256="3" * 64,
        final_plan_sha256=manifest["plan_sha256"],
        final_execution_manifest_path=(
            "results/final/execution/execution_manifest.json"
        ),
        final_execution_manifest_file_sha256=manifest_file_sha256,
        final_execution_manifest_payload_sha256=manifest["manifest_sha256"],
        execution_validation_sha256="4" * 64,
        storage_preflight_sha256="5" * 64,
    )
    source_plan = SimpleNamespace(
        plan_sha256=manifest["plan_sha256"],
        input_hashes=manifest["input_hashes"],
        entries=({**run, "ordinal": 1},),
    )
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    plan = build_downstream_evidence_plan(
        source,
        source_kind="final",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
        evaluated_round_binding=binding,
        source_plan=source_plan,
    )
    destination = tmp_path / "downstream-final"
    partial = run_downstream_evidence(plan, destination, max_denominators=0)
    assert partial["status"] == "running"
    result = run_downstream_evidence(plan, destination)

    assert result["status"] == "completed"
    loaded = load_downstream_evidence_manifest(destination)
    assert len(loaded.records) == 1
    assert len(loaded.records[0]["endpoints"]) == 8
    assert loaded.records[0]["source_kind"] == "final"

    manifest["artifact_storage"]["evaluator_output_compression_level"] = 9
    changed_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(changed_body)
    _write_canonical(source / "execution_manifest.json", manifest)
    with pytest.raises(
        DownstreamEvidenceError, match="final artifact storage policy differs"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="final",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
            evaluated_round_binding=binding,
            source_plan=source_plan,
        )


def test_complete_manifest_revalidates_bound_source_and_dataset_bytes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )

    source, dataset_path, cells, output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    source_raw = output_path.read_bytes()
    output_path.write_bytes(source_raw + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="evaluator output.*checksum"):
        load_downstream_evidence_manifest(destination)
    output_path.write_bytes(source_raw)

    dataset_raw = dataset_path.read_bytes()
    dataset_path.write_bytes(dataset_raw + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="dataset raw file checksum"):
        load_downstream_evidence_manifest(destination)


def test_plan_revalidation_forwards_the_evaluated_round_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = downstream.build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    binding = downstream.EvaluatedRoundBinding(
        repository_root=str((tmp_path / "repository").absolute()),
        round_root=str((tmp_path / "repository/round-1").absolute()),
        round_id="round-1",
        evaluation_receipt_path="evaluation_receipt.json",
        evaluation_receipt_file_sha256="1" * 64,
        evaluation_receipt_payload_sha256="2" * 64,
        result_manifest_sha256="3" * 64,
        final_plan_sha256="4" * 64,
        final_execution_manifest_path=(
            "results/final/execution/execution_manifest.json"
        ),
        final_execution_manifest_file_sha256="5" * 64,
        final_execution_manifest_payload_sha256="6" * 64,
        execution_validation_sha256="7" * 64,
        storage_preflight_sha256="8" * 64,
    )
    bound = replace(plan, evaluated_round_binding=binding, plan_sha256="0" * 64)
    bound = replace(bound, plan_sha256=canonical_sha256(bound.body()))
    observed: dict[str, object] = {}

    def rebuild(*args: object, evaluated_round_binding: object, **kwargs: object):
        observed["binding"] = evaluated_round_binding
        return bound

    monkeypatch.setattr(downstream, "build_downstream_evidence_plan", rebuild)

    downstream._revalidate_plan(bound)

    assert observed["binding"] is binding


def test_completed_manifest_missing_prefix_fails_without_repairing_files(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    plan_path = destination / "plan.json"
    missing_record = destination / "records/00000002.json"
    plan_path.unlink()
    missing_record.unlink()

    with pytest.raises(DownstreamEvidenceError):
        run_downstream_evidence(plan, destination)
    assert not plan_path.exists()
    assert not missing_record.exists()


def test_loader_rejects_rehashed_downstream_manifest_schema_extension(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    path = destination / "downstream_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["unknown_field"] = "forged"
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(body)
    _write_canonical(path, manifest)

    with pytest.raises(DownstreamEvidenceError, match="manifest schema differs"):
        load_downstream_evidence_manifest(destination)


def test_complete_manifest_rejects_self_consistent_sealed_source_drift(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)

    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"]["configuration_id"] = "forged-configuration"
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    checkpoint_body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(
        DownstreamEvidenceError, match="configuration authority differs"
    ):
        load_downstream_evidence_manifest(destination)


def test_plan_rejects_rehashed_registry_wrapper_with_swapped_method_spec(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    candidate, _magic = _test_configuration_authority()
    registry = load_method_registry(Path("study/methods.json"))
    swapped = AuthorizedConfiguration.create(
        method_id="magic",
        configuration_id="registry-default",
        kind="registry",
        payload={
            "schema": "maskimpute-registry-default-configuration-v1",
            "method": asdict(registry.by_id("saver")),
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["records"][1]["run"]["configuration_sha256"] = (
        swapped.configuration_sha256
    )
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(
        DownstreamEvidenceError, match="registry configuration method payload differs"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=(candidate, swapped),
        )


def test_plan_rejects_source_configuration_and_artifact_authority_mismatch(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    candidate, magic = _test_configuration_authority()
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"]["configuration_sha256"] = magic.configuration_sha256
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(
        DownstreamEvidenceError, match="configuration authority differs"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=(candidate, magic),
        )


def test_plan_rejects_rehashed_source_run_with_unknown_schema_field(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["records"][0]["run"]["unknown_field"] = "forged"
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="source run schema differs"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_development_production_wrapper_rejects_repository_symlink_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.runner as runner

    active_repository = Path(downstream.__file__).resolve().parents[1]
    alias = tmp_path / "repository-alias"
    alias.symlink_to(active_repository, target_is_directory=True)

    def unexpected_authority_load():
        raise AssertionError("symlink was resolved before validation")

    monkeypatch.setattr(runner, "load_runner_authority", unexpected_authority_load)
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="development repository path contains a symlink",
    ):
        downstream.build_development_downstream_evidence_plan(alias)


def test_final_production_wrapper_rejects_round_symlink_ancestor(
    tmp_path: Path,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    active_repository = Path(downstream.__file__).resolve().parents[1]
    round_directory = tmp_path / "round"
    round_directory.mkdir()
    alias = tmp_path / "round-alias"
    alias.symlink_to(round_directory, target_is_directory=True)

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="final round path contains a symlink",
    ):
        downstream.build_final_downstream_evidence_plan(active_repository, alias)


def test_output_symlink_ancestor_is_rejected_before_directory_creation(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    actual_parent = tmp_path / "actual-output-parent"
    actual_parent.mkdir()
    alias = tmp_path / "output-parent-alias"
    alias.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(DownstreamEvidenceError, match="path contains a symlink"):
        run_downstream_evidence(plan, alias / "downstream")
    assert not (actual_parent / "downstream").exists()


def test_generic_source_and_dataset_roots_reject_symlink_ancestors(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        bind_evaluator_dataset,
        build_downstream_evidence_plan,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    alias = tmp_path / "root-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(
        DownstreamEvidenceError, match="evaluator dataset path contains a symlink"
    ):
        bind_evaluator_dataset(
            alias / dataset_path.name,
            retained_cell_ids=cells,
        )
    with pytest.raises(
        DownstreamEvidenceError, match="source root path contains a symlink"
    ):
        build_downstream_evidence_plan(
            alias / source.name,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_registered_trajectory_binding_is_mandatory_and_exact(tmp_path: Path) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        _read_bound_dataset,
        bind_evaluator_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        generate_registered_trajectory_dataset,
        load_trajectory_authority,
    )

    authority = load_trajectory_authority()
    dataset = generate_registered_trajectory_dataset(authority=authority)
    path = tmp_path / "registered-trajectory.h5ad"
    dataset.write_h5ad(path)
    retained = tuple(dataset.obs_names.astype(str))

    with pytest.raises(
        DownstreamEvidenceError, match="registered trajectory authority is required"
    ):
        bind_evaluator_dataset(path, retained_cell_ids=retained)
    with pytest.raises(DownstreamEvidenceError, match="trajectory root differs"):
        bind_evaluator_dataset(
            path,
            retained_cell_ids=retained,
            trajectory_root_cell_id=retained[1],
            trajectory_source_id=authority.source_id,
        )
    with pytest.raises(DownstreamEvidenceError, match="trajectory source differs"):
        bind_evaluator_dataset(
            path,
            retained_cell_ids=retained,
            trajectory_root_cell_id=authority.root_cell_id,
            trajectory_source_id="ad-hoc-trajectory-source",
        )

    binding = bind_evaluator_dataset(
        path,
        retained_cell_ids=retained,
        trajectory_root_cell_id=authority.root_cell_id,
        trajectory_source_id=authority.source_id,
    )
    assert binding.dataset_sha256 == authority.expected_dataset_sha256
    assert binding.trajectory_authority_sha256 == authority.authority_sha256
    assert binding.trajectory_binding_sha256 == authority.binding_sha256
    _read_bound_dataset(binding)
    with pytest.raises(DownstreamEvidenceError, match="authority checksum differs"):
        _read_bound_dataset(replace(binding, trajectory_authority_sha256="0" * 64))
    with pytest.raises(DownstreamEvidenceError, match="binding checksum differs"):
        _read_bound_dataset(replace(binding, trajectory_binding_sha256="0" * 64))
    with pytest.raises(DownstreamEvidenceError, match="trajectory source differs"):
        _read_bound_dataset(
            replace(binding, trajectory_source_id="ad-hoc-trajectory-source")
        )


@pytest.mark.parametrize("mechanism", ["symsim", "sergio", "sparsim", "semisynthetic"])
def test_reconstruction_mechanisms_reject_trajectory_binding_fields(
    tmp_path: Path,
    mechanism: str,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        bind_evaluator_dataset,
    )

    path = tmp_path / f"{mechanism}.h5ad"
    dataset, cells, _genes = _dataset(path)
    dataset.obs["mechanism"] = mechanism
    dataset.write_h5ad(path)

    with pytest.raises(
        DownstreamEvidenceError,
        match="reconstruction mechanism cannot carry trajectory authority",
    ):
        bind_evaluator_dataset(
            path,
            retained_cell_ids=cells,
            trajectory_root_cell_id=cells[0],
            trajectory_source_id="ad-hoc-trajectory-source",
        )


def test_selection_schema_four_requires_bound_downstream_completeness(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.revisions import development_selection_stage_paths
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        attach_downstream_evidence_to_selection_result,
        validate_downstream_selection_completeness,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
        source_plan=_development_source_plan(source),
    )
    stage_paths = development_selection_stage_paths(None)
    destination = tmp_path / stage_paths.downstream_directory
    run_downstream_evidence(plan, destination)
    evidence = load_downstream_evidence_manifest(destination)
    selection_records = [
        {
            "mechanism": record["mechanism"],
            "biological_id": record["biological_id"],
            "technical_view": record["technical_view"],
            "dataset_id": record["dataset_id"],
            "dataset_sha256": record["dataset_sha256"],
            "method": record["method"],
            "method_sha256": record["method_artifact_sha256"],
            "model_seed": record["model_seed"],
            "metric": "mse",
            "value": None,
            "status": "failed",
        }
        for record in evidence.records
    ]
    core = {
        "schema_version": 2,
        "dataset_manifest_sha256": "1" * 64,
        "count_score_manifest_sha256": "2" * 64,
        "retained_calibration_artifact_sha256": "3" * 64,
        "evaluation_manifest_sha256": "4" * 64,
        "records": selection_records,
        "orthogonal_intervals": [],
    }
    payload = {**core, "result_sha256": canonical_sha256(core)}
    source_path = tmp_path / stage_paths.source_selection_input
    _write_canonical(source_path, payload)

    with pytest.raises(SelectionAuthorityError, match="source status"):
        attach_downstream_evidence_to_selection_result(
            payload,
            tmp_path,
            stage_paths.downstream_directory,
        )

    for selection_record, downstream_record in zip(
        selection_records, evidence.records, strict=True
    ):
        selection_record["status"] = downstream_record["run_status"]
    selection_records.append(
        {
            **selection_records[0],
            "metric": "null_de_fpr",
            "value": None,
            "status": "unavailable",
        }
    )
    core["records"] = selection_records
    payload = {**core, "result_sha256": canonical_sha256(core)}
    source_file_sha = _write_canonical(source_path, payload)

    forged_source = {**payload, "operator_override": True}
    forged_source_core = {
        key: value for key, value in forged_source.items() if key != "result_sha256"
    }
    forged_source["result_sha256"] = canonical_sha256(forged_source_core)
    _write_canonical(source_path, forged_source)
    with pytest.raises(SelectionAuthorityError, match="missing or extra"):
        attach_downstream_evidence_to_selection_result(
            forged_source,
            tmp_path,
            stage_paths.downstream_directory,
        )

    changed_source = dict(payload)
    changed_source["dataset_manifest_sha256"] = "9" * 64
    changed_source_core = {
        key: value for key, value in changed_source.items() if key != "result_sha256"
    }
    changed_source["result_sha256"] = canonical_sha256(changed_source_core)
    _write_canonical(source_path, changed_source)
    with pytest.raises(SelectionAuthorityError, match="source selection input differs"):
        attach_downstream_evidence_to_selection_result(
            payload,
            tmp_path,
            stage_paths.downstream_directory,
        )

    source_file_sha = _write_canonical(source_path, payload)
    upgraded = attach_downstream_evidence_to_selection_result(
        payload,
        tmp_path,
        stage_paths.downstream_directory,
    )

    assert upgraded["schema_version"] == 4
    assert upgraded["revision_versions"] == []
    binding = upgraded["downstream_evidence"]
    receipt = validate_downstream_selection_completeness(
        tmp_path, upgraded["records"], binding
    )
    assert receipt["downstream_manifest_sha256"] == evidence.manifest_sha256
    assert binding["endpoint_row_count"] == 16
    assert binding["source_selection_input_path"] == (
        stage_paths.source_selection_input
    )
    assert binding["source_selection_input_file_sha256"] == source_file_sha
    assert binding["source_selection_result_sha256"] == payload["result_sha256"]

    missing_denominator = [
        record
        for record in upgraded["records"]
        if record["method"] != upgraded["records"][0]["method"]
    ]
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(
            tmp_path, missing_denominator, binding
        )

    changed_dataset = [dict(record) for record in upgraded["records"]]
    changed_dataset[0]["dataset_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(tmp_path, changed_dataset, binding)

    changed_method = [dict(record) for record in upgraded["records"]]
    changed_method[0]["method_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(tmp_path, changed_method, binding)

    changed_checkpoint = dict(binding)
    changed_checkpoint["source_checkpoint_file_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="binding differs"):
        validate_downstream_selection_completeness(
            tmp_path, upgraded["records"], changed_checkpoint
        )


def test_final_cli_uses_external_receipt_bound_archive_without_round_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path("scripts/run_final_downstream_evidence.py").absolute()
    specification = importlib.util.spec_from_file_location(
        "run_final_downstream_evidence_test", script_path
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = script
    specification.loader.exec_module(script)

    repository = tmp_path / "repository"
    round_directory = repository / "artifacts/study/final/rounds/round-1"
    round_directory.mkdir(parents=True)
    (round_directory / "evaluation_receipt.json").write_bytes(b"sealed-receipt\n")
    before = {
        path.relative_to(round_directory).as_posix(): path.read_bytes()
        for path in round_directory.rglob("*")
        if path.is_file()
    }
    receipt_sha256 = "a" * 64
    plan = SimpleNamespace(
        evaluated_round_binding=SimpleNamespace(
            round_id="round-1",
            evaluation_receipt_payload_sha256=receipt_sha256,
        )
    )
    observed: dict[str, Path] = {}

    monkeypatch.setattr(script, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        script,
        "build_final_downstream_evidence_plan",
        lambda selected_repository, selected_round: (
            plan
            if selected_repository == repository and selected_round == round_directory
            else pytest.fail("final plan received a different repository or round")
        ),
    )

    def run_evidence(selected_plan: object, output_directory: Path):
        assert selected_plan is plan
        observed["output"] = output_directory
        output_directory.mkdir(parents=True)
        (output_directory / "proof.json").write_text("{}\n", encoding="utf-8")
        return {"status": "completed"}

    monkeypatch.setattr(script, "run_downstream_evidence", run_evidence)

    different_working_directory = tmp_path / "different-working-directory"
    different_working_directory.mkdir()
    monkeypatch.chdir(different_working_directory)
    repository_relative_round = round_directory.relative_to(repository)
    assert script.main(["--round-dir", repository_relative_round.as_posix()]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}
    expected = (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "downstream"
        / "round-1"
        / receipt_sha256
    )
    assert observed["output"] == expected
    assert not observed["output"].is_relative_to(repository)
    after = {
        path.relative_to(round_directory).as_posix(): path.read_bytes()
        for path in round_directory.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_evaluated_round_binding_revalidates_exact_receipt_and_execution_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    repository = tmp_path / "repository"
    round_directory = repository / "artifacts/study/final/rounds/round-1"
    execution_directory = round_directory / "results/final/execution"
    execution_directory.mkdir(parents=True)
    execution_body = {
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": "1" * 64,
        "input_hashes": {},
        "planned_run_count": 0,
        "recorded_run_count": 0,
        "records": [],
        "artifact_storage": {
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "evaluator_output_compression_level": 6,
            "native_output_retention": "omitted_redundant_final_output",
            "p_pre_zero_encoding": "zlib_raw_f64_v1",
            "p_pre_zero_compression_level": 6,
        },
    }
    execution_manifest = {
        **execution_body,
        "manifest_sha256": canonical_sha256(execution_body),
    }
    execution_file_sha256 = _write_canonical(
        execution_directory / "execution_manifest.json", execution_manifest
    )
    validation_body = {
        "schema_version": 1,
        "status": "eligible_for_final_evaluation_complete_terminal_denominator",
        "final_plan_sha256": "1" * 64,
        "planned_run_count": 0,
        "executed_completed_count": 0,
        "executed_algorithmic_failure_count": 0,
        "executed_status_counts": {},
        "not_applicable_count": 0,
        "record_payload_sha256s": [],
    }
    result_manifest = {
        "schema_version": 1,
        "status": "completed",
        "final_plan_sha256": "1" * 64,
        "final_execution_manifest_path": (
            "results/final/execution/execution_manifest.json"
        ),
        "final_execution_manifest_sha256": execution_file_sha256,
        "final_execution_payload_sha256": execution_manifest["manifest_sha256"],
        "execution_validation": {
            **validation_body,
            "validation_sha256": canonical_sha256(validation_body),
        },
        "storage_preflight": {"schema": "test"},
        "result_files": [],
    }
    receipt = {
        "schema_version": 1,
        "round_id": "round-1",
        "state": "evaluated",
        "evaluated_at": "2026-07-16T00:00:00Z",
        "execution_claim_id": "claim-1",
        "result_manifest": result_manifest,
        "result_manifest_sha256": canonical_sha256(result_manifest),
        "seed_manifest_sha256": "2" * 64,
        "round_path": "artifacts/study/final/rounds/round-1",
        "round_token": "round-token",
        "repository_instance_id": "repository-instance",
        "worktree_path_sha256": "3" * 64,
        "git_common_dir_device": 1,
        "git_common_dir_inode": 2,
        "study_state_root_device": 1,
        "study_state_root_inode": 3,
        "registry_dir_device": 1,
        "registry_dir_inode": 4,
        "method_commit": "4" * 40,
        "config_sha256": "5" * 64,
        "protocol_sha256": "6" * 64,
        "environment_sha256": "7" * 64,
        "operational_artifact_roots_sha256": "8" * 64,
    }
    _write_canonical(round_directory / "evaluation_receipt.json", receipt)

    def validated_receipt(_repository: Path, _round: Path):
        return json.loads(
            (round_directory / "evaluation_receipt.json").read_text(encoding="utf-8")
        )

    monkeypatch.setattr(
        downstream,
        "_validated_evaluated_round_receipt",
        validated_receipt,
        raising=False,
    )
    binding = downstream._read_verified_evaluated_round_binding(
        repository, round_directory
    )
    assert binding.result_manifest_sha256 == canonical_sha256(result_manifest)
    assert binding.final_execution_manifest_file_sha256 == execution_file_sha256
    assert (
        binding.final_execution_manifest_payload_sha256
        == execution_manifest["manifest_sha256"]
    )

    receipt["evaluated_at"] = "2026-07-16T00:00:01Z"
    _write_canonical(round_directory / "evaluation_receipt.json", receipt)
    with pytest.raises(downstream.DownstreamEvidenceError, match="binding changed"):
        downstream._validate_evaluated_round_binding(binding)

    receipt["evaluated_at"] = "2026-07-16T00:00:00Z"
    validation = result_manifest["execution_validation"]
    assert isinstance(validation, dict)
    validation["executed_completed_count"] = 1
    changed_validation_body = {
        key: value for key, value in validation.items() if key != "validation_sha256"
    }
    validation["validation_sha256"] = canonical_sha256(changed_validation_body)
    receipt["result_manifest_sha256"] = canonical_sha256(result_manifest)
    _write_canonical(round_directory / "evaluation_receipt.json", receipt)
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="validation denominator differs",
    ):
        downstream._read_verified_evaluated_round_binding(repository, round_directory)


def test_receipt_bound_final_output_rejects_round_and_repository_containment(
    tmp_path: Path,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    repository = tmp_path / "repository"
    round_directory = repository / "artifacts/study/final/rounds/round-1"
    round_directory.mkdir(parents=True)
    plan = SimpleNamespace(
        source_kind="final",
        evaluated_round_binding=SimpleNamespace(
            repository_root=str(repository.absolute()),
            round_root=str(round_directory.absolute()),
        ),
    )
    forbidden = round_directory / "results/final/downstream"
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="outside the frozen repository",
    ):
        downstream._validate_downstream_output_location(plan, forbidden)
    assert not forbidden.exists()

    external = tmp_path / "repository-final-analysis/downstream"
    downstream._validate_downstream_output_location(plan, external)
