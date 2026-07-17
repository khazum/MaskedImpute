from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
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


def _development_source(tmp_path: Path):
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset_path = tmp_path / "dataset.h5ad"
    dataset, cells, _genes = _dataset(dataset_path)
    dataset_sha = benchmark_dataset_sha256(dataset)
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
    records = [
        {"run": completed, "metrics": []},
        {"run": failed, "metrics": []},
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


def test_prepared_runner_panel_bridge_binds_persisted_dataset_paths(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        bind_prepared_evaluator_panel,
    )

    dataset_path = tmp_path / "dataset.h5ad"
    _dataset_object, cells, _genes = _dataset(dataset_path)
    runner_binding = SimpleNamespace(
        dataset_id="dev-symsim-01", output_path="dataset.h5ad"
    )
    prepared = {
        "dev-symsim-01": SimpleNamespace(audit=SimpleNamespace(retained_cell_ids=cells))
    }

    bindings = bind_prepared_evaluator_panel(
        (runner_binding,), prepared, dataset_root=tmp_path
    )

    assert len(bindings) == 1
    assert bindings[0].dataset_id == "dev-symsim-01"
    assert bindings[0].retained_cell_ids == cells
    assert bindings[0].path == str(dataset_path.absolute())


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
    assert records[0]["method_artifact_sha256"] == records[0][
        "configuration_sha256"
    ]
    assert records[1]["method_artifact_sha256"] != records[1][
        "configuration_sha256"
    ]
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
    endpoint = next(
        row for row in record["endpoints"] if row["status"] == "completed"
    )
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

    with pytest.raises(
        DownstreamEvidenceError, match="endpoint re-evaluation differs"
    ):
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
    body = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
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
) -> None:
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
    source = tmp_path / "final"
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
        },
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    _write_canonical(source / "execution_manifest.json", manifest)

    plan = build_downstream_evidence_plan(
        source,
        source_kind="final",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream-final"
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
    checkpoint["records"][0]["run"]["configuration_id"] = "forged-configuration"
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
    checkpoint["records"][0]["run"]["configuration_sha256"] = (
        magic.configuration_sha256
    )
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="configuration authority differs"):
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
        downstream.build_final_downstream_evidence_plan(
            active_repository, alias
        )


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
    )
    destination = tmp_path / "artifacts" / "downstream"
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

    upgraded = attach_downstream_evidence_to_selection_result(
        payload,
        tmp_path,
        "artifacts/downstream",
    )

    assert upgraded["schema_version"] == 4
    assert upgraded["revision_versions"] == []
    binding = upgraded["downstream_evidence"]
    receipt = validate_downstream_selection_completeness(
        tmp_path, upgraded["records"], binding
    )
    assert receipt["downstream_manifest_sha256"] == evidence.manifest_sha256
    assert binding["endpoint_row_count"] == 16

    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(
            tmp_path, upgraded["records"][:-1], binding
        )

    changed_dataset = [dict(record) for record in upgraded["records"]]
    changed_dataset[0]["dataset_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(
            tmp_path, changed_dataset, binding
        )

    changed_method = [dict(record) for record in upgraded["records"]]
    changed_method[0]["method_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(tmp_path, changed_method, binding)
