from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any

import pytest

from maskimpute_benchmark.final_analysis import (
    FinalAnalysisContractError,
    _frozen_metric_direction_contract,
    _git_object_bytes,
    build_final_analysis,
    generate_final_analysis,
)
from maskimpute_benchmark.protocol import canonical_sha256


PRIMARY_METRICS = ("mse", "corr_err")


def _metric_row(
    *,
    method: str,
    mechanism: str,
    biological_id: str,
    technical_view: str,
    dataset_id: str,
    model_seed: int | None,
    metric: str,
    value: float | None,
    status: str,
    reason: str | None,
) -> dict[str, object]:
    return {
        "mechanism": mechanism,
        "biological_id": biological_id,
        "technical_view": technical_view,
        "dataset_id": dataset_id,
        "method": method,
        "model_seed": model_seed,
        "metric": metric,
        "value": value,
        "n": 10 if value is not None else 0,
        "status": status,
        "reason": reason,
    }


def _record(
    method: str,
    run_status: str,
    *,
    ordinal: int,
    mechanism: str = "symsim",
    biological_id: str = "draw-01",
    technical_view: str = "moderate",
    model_seed: int | None = None,
    values: dict[str, float | None] | None = None,
    metric_statuses: dict[str, str] | None = None,
    metric_reasons: dict[str, str | None] | None = None,
    run_reason: str | None = None,
) -> dict[str, Any]:
    dataset_id = f"dataset-{mechanism}-{biological_id}-{technical_view}"
    values = values or {metric: None for metric in PRIMARY_METRICS}
    if metric_statuses is None:
        metric_statuses = {metric: run_status for metric in PRIMARY_METRICS}
    metric_reasons = metric_reasons or {
        metric: run_reason for metric in PRIMARY_METRICS
    }
    return {
        "run": {
            "run_id": f"run-{ordinal:03d}",
            "method_id": method,
            "mechanism": mechanism,
            "biological_id": biological_id,
            "technical_view": technical_view,
            "dataset_id": dataset_id,
            "model_seed": model_seed,
            "source_dataset_sha256": "a" * 64,
            "configuration_id": f"config-{method}",
            "configuration_sha256": "b" * 64,
            "method_input_sha256": "c" * 64,
            "retained_cell_ids_sha256": "d" * 64,
            "status": run_status,
            "reason": run_reason,
        },
        "metrics": [
            _metric_row(
                method=method,
                mechanism=mechanism,
                biological_id=biological_id,
                technical_view=technical_view,
                dataset_id=dataset_id,
                model_seed=model_seed,
                metric=metric,
                value=values[metric],
                status=metric_statuses[metric],
                reason=metric_reasons[metric],
            )
            for metric in PRIMARY_METRICS
        ],
        "execution_request": None,
    }


SCORE_METRICS = (
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "calibration_intercept",
    "calibration_slope",
    "ece",
)


def _score_group(
    *,
    stratum_type: str,
    label: str,
    n: int,
    status: str,
    reason: str | None,
    auroc: float | None,
) -> dict[str, object]:
    values = {
        "auroc": auroc,
        "auprc": None if auroc is None else auroc - 0.05,
        "brier": None if auroc is None else 1.0 - auroc,
        "log_loss": None if auroc is None else 1.5 - auroc,
        "calibration_intercept": None if auroc is None else auroc - 0.75,
        "calibration_slope": None if auroc is None else auroc + 0.25,
        "ece": None if auroc is None else (1.0 - auroc) / 2.0,
    }
    metrics = {
        metric: {
            "value": values[metric],
            "n": n,
            "status": "completed" if values[metric] is not None else status,
            "reason": None if values[metric] is not None else reason,
        }
        for metric in SCORE_METRICS
    }
    return {
        "stratum_type": stratum_type,
        "label": label,
        "lower": None,
        "upper": None,
        "n": n,
        "metrics": metrics,
        "reliability_bins": [],
    }


def _attach_score_evidence(
    record: dict[str, Any],
    *,
    matrix_present: bool | None = None,
) -> None:
    run = record["run"]
    method = run["method_id"]
    run_status = run["status"]
    if method == "maskimpute":
        evidence_status = run_status
        evidence_reason = run["reason"]
    else:
        evidence_status = "not_applicable"
        evidence_reason = "method_does_not_emit_p_pre_zero"
    numeric = method == "maskimpute" and run_status == "completed"
    if matrix_present is None:
        matrix_present = numeric
    draw_index = int(str(run["biological_id"]).split("-")[-1])
    view_offset = -0.01 if run["technical_view"] == "moderate" else 0.01
    seed_offset = {42: -0.03, 43: 0.0, 44: 0.03}.get(run["model_seed"], 0.0)
    auroc = 0.5 + draw_index / 10.0 + view_offset + seed_offset if numeric else None
    overall = _score_group(
        stratum_type="overall",
        label="all_observed_zeros",
        n=100,
        status=evidence_status,
        reason=evidence_reason,
        auroc=auroc,
    )
    strata = {
        "library_size_quartiles": [
            _score_group(
                stratum_type="library_size_quartiles",
                label=f"Q{quartile}",
                n=25,
                status=evidence_status,
                reason=evidence_reason,
                auroc=None if auroc is None else auroc - quartile / 100.0,
            )
            for quartile in range(1, 5)
        ],
        "truth_expression_bins": [
            _score_group(
                stratum_type="truth_expression_bins",
                label=label,
                n=25,
                status=evidence_status,
                reason=evidence_reason,
                auroc=None if auroc is None else auroc - index / 100.0,
            )
            for index, label in enumerate(
                ("[0,1)", "[1,2)", "[2,4)", "[4,inf)"), start=1
            )
        ],
    }
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
    if matrix_present:
        policy = {
            "schema_version": 2,
            "probability_semantics": (
                "pre_capture_count_is_zero_given_observed_counts"
            ),
            "evaluation_domain": "observed_zero_entries_only",
            "score_source": "retained_calibrator",
            "score_artifact_sha256": "1" * 64,
            "score_input_sha256": "2" * 64,
            "score_config_sha256": "3" * 64,
            "calibration_file_sha256": "4" * 64,
            "calibration_payload_sha256": "5" * 64,
            "calibration_algorithm": "identity",
            "calibration_scope": "retained_all_development",
            "calibration_equivalence_reason": (
                "retained_identity_calibrator_equals_direct_score"
            ),
        }
        matrix = {
            "shape": [10, 10],
            "dtype": "<f8",
            "content_sha256": "e" * 64,
            "semantic_sha256": "f" * 64,
        }
        storage = {
            "encoding": "zlib_raw_f64_v1",
            "compression_level": 6,
            "path": f"runs/{run['run_id']}.p_pre_zero.f64.zlib",
            "compressed_sha256": "1" * 64,
            "compressed_nbytes": 8,
            "uncompressed_sha256": "e" * 64,
            "uncompressed_nbytes": 800,
        }
    else:
        policy = None
        matrix = {
            "shape": None,
            "dtype": None,
            "content_sha256": None,
            "semantic_sha256": None,
        }
        storage = {
            "encoding": None,
            "compression_level": None,
            "path": None,
            "compressed_sha256": None,
            "compressed_nbytes": None,
            "uncompressed_sha256": None,
            "uncompressed_nbytes": None,
        }
    body = {
        "schema_version": 1,
        "status": evidence_status,
        "reason": evidence_reason,
        "identity": identity,
        "truth_kind": "exact_pre_capture",
        "matrix": matrix,
        "policy": policy,
        "policy_sha256": None if policy is None else canonical_sha256(policy),
        "overall": overall,
        "strata": strata,
    }
    record["p_pre_zero_evidence"] = {
        **body,
        "evidence_sha256": canonical_sha256(body),
        "storage": storage,
    }


def _analysis(
    records: list[dict[str, Any]],
    *,
    primary_metrics: tuple[str, ...] = PRIMARY_METRICS,
) -> dict[str, object]:
    direction_body = {
        "schema_version": 1,
        "status": "validated",
        "reason": None,
        "favorable_direction": "lower",
        "metrics": [
            "corr_err",
            "gnrmse",
            "mse",
            "mse_dropout",
            "mse_pre_dropout_zero",
        ],
        "authority": {
            "source": "synthetic_frozen_selection_gate_fixture",
            "method_commit": "f" * 40,
        },
    }
    return build_final_analysis(
        records,
        protocol={"schema_version": 1, "primary_metrics": list(primary_metrics)},
        selection_contract={
            "schema_version": 1,
            "candidate_method_id": "maskimpute",
        },
        input_bindings={
            "planned_run_count": len(records),
            "metric_direction_contract": {
                **direction_body,
                "contract_sha256": canonical_sha256(direction_body),
            },
        },
    )


def _paired_panel() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ordinal = 0
    for draw_index, candidate_mse in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        biological_id = f"draw-{draw_index:02d}"
        for technical_view, view_offset in (("moderate", -0.1), ("severe", 0.1)):
            for seed, seed_offset in zip((42, 43, 44), (-0.3, 0.0, 0.3), strict=True):
                ordinal += 1
                records.append(
                    _record(
                        "maskimpute",
                        "completed",
                        ordinal=ordinal,
                        biological_id=biological_id,
                        technical_view=technical_view,
                        model_seed=seed,
                        values={
                            "mse": candidate_mse + view_offset + seed_offset,
                            "corr_err": (
                                candidate_mse / 10.0
                                + view_offset / 10.0
                                + seed_offset / 10.0
                            ),
                        },
                        metric_statuses={
                            "mse": "completed",
                            "corr_err": "completed",
                        },
                        metric_reasons={"mse": None, "corr_err": None},
                    )
                )
            ordinal += 1
            records.append(
                _record(
                    "dca",
                    "completed",
                    ordinal=ordinal,
                    biological_id=biological_id,
                    technical_view=technical_view,
                    values={"mse": 4.0, "corr_err": 0.5},
                    metric_statuses={
                        "mse": "completed",
                        "corr_err": "completed",
                    },
                    metric_reasons={"mse": None, "corr_err": None},
                )
            )
            ordinal += 1
            records.append(
                _record(
                    "magic",
                    "failed",
                    ordinal=ordinal,
                    biological_id=biological_id,
                    technical_view=technical_view,
                    run_reason="algorithm_failure",
                )
            )
    return records


def _matching(rows: object, **identity: str) -> dict[str, object]:
    assert isinstance(rows, list)
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and all(row.get(field) == value for field, value in identity.items())
    ]
    assert len(matches) == 1
    return matches[0]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _write_json(path: Path, value: object) -> bytes:
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _evaluated_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    import maskimpute_benchmark.study as study

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/round-001"
    execution_dir = round_dir / "results/final/execution"
    records_dir = execution_dir / "records"
    records_dir.mkdir(parents=True)
    (repository / "study").mkdir()

    records = [
        _record(
            "maskimpute",
            "completed",
            ordinal=1,
            values={"mse": 1.0, "corr_err": 0.1},
            metric_statuses={"mse": "completed", "corr_err": "completed"},
            metric_reasons={"mse": None, "corr_err": None},
        ),
        _record(
            "dca",
            "completed",
            ordinal=2,
            values={"mse": 2.0, "corr_err": 0.2},
            metric_statuses={"mse": "completed", "corr_err": "completed"},
            metric_reasons={"mse": None, "corr_err": None},
        ),
    ]
    for record in records:
        _attach_score_evidence(record)
        storage = record["p_pre_zero_evidence"]["storage"]
        if storage["path"] is not None:
            score_raw = b"score123"
            score_path = execution_dir / storage["path"]
            score_path.parent.mkdir(parents=True, exist_ok=True)
            score_path.write_bytes(score_raw)
            storage["compressed_sha256"] = hashlib.sha256(score_raw).hexdigest()
            storage["compressed_nbytes"] = len(score_raw)
    record_paths = []
    for ordinal, record in enumerate(records, start=1):
        path = records_dir / f"{ordinal:08d}.json"
        _write_json(path, record)
        record_paths.append(path)

    protocol = {"schema_version": 1, "primary_metrics": list(PRIMARY_METRICS)}
    protocol_path = repository / "study/protocol.json"
    protocol_raw = _write_json(protocol_path, protocol)
    selection = {
        "schema_version": 1,
        "candidate_method_id": "maskimpute",
    }
    selection_path = repository / "study/selection_contract.json"
    selection_raw = _write_json(selection_path, selection)
    frozen_body = {
        "schema_version": 1,
        "candidate_method_id": "maskimpute",
        "artifact_bindings": {
            "selection_contract": {
                "path": "study/selection_contract.json",
                "sha256": hashlib.sha256(selection_raw).hexdigest(),
            }
        },
    }
    frozen_method = {
        **frozen_body,
        "payload_sha256": canonical_sha256(frozen_body),
    }
    config_path = repository / "study/frozen_method.json"
    config_raw = _write_json(config_path, frozen_method)

    final_plan_sha256 = "1" * 64
    fixture: dict[str, Any] = {
        "repository": repository,
        "round_dir": round_dir,
        "execution_dir": execution_dir,
        "record_paths": record_paths,
        "records": records,
        "final_plan_sha256": final_plan_sha256,
        "protocol": protocol,
        "selection": selection,
    }

    freeze = {
        "config_path": "study/frozen_method.json",
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "protocol_path": "study/protocol.json",
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
    }
    fixture["freeze"] = freeze

    def refresh_bindings() -> None:
        references = []
        payload_hashes = []
        result_files = []
        for ordinal, path in enumerate(record_paths, start=1):
            raw = path.read_bytes()
            payload = json.loads(raw)
            references.append(
                {
                    "ordinal": ordinal,
                    "path": f"records/{ordinal:08d}.json",
                    "run_id": payload["run"]["run_id"],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            payload_hashes.append(canonical_sha256(payload))
            result_files.append(
                {
                    "path": path.relative_to(round_dir).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            score_evidence = payload.get("p_pre_zero_evidence")
            score_storage = (
                score_evidence.get("storage")
                if isinstance(score_evidence, dict)
                else None
            )
            if isinstance(score_storage, dict) and score_storage["path"] is not None:
                score_path = execution_dir / score_storage["path"]
                score_raw = score_path.read_bytes()
                result_files.append(
                    {
                        "path": score_path.relative_to(round_dir).as_posix(),
                        "sha256": hashlib.sha256(score_raw).hexdigest(),
                    }
                )
        manifest_body = {
            "schema_version": 1,
            "status": "completed",
            "plan_sha256": final_plan_sha256,
            "input_hashes": {
                "dataset_design_sha256": "2" * 64,
                "dataset_manifest_sha256": "3" * 64,
                "dataset_seed_source_sha256": "4" * 64,
                "execution_authority_sha256": "5" * 64,
                "execution_claim_sha256": "6" * 64,
                "execution_environment_sha256": "7" * 64,
                "frozen_method_sha256": frozen_method["payload_sha256"],
                "method_registry_sha256": "8" * 64,
                "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
                "runtime_lock_sha256": "9" * 64,
            },
            "planned_run_count": len(record_paths),
            "recorded_run_count": len(record_paths),
            "records": references,
            "artifact_storage": {
                "evaluator_output_compression_level": 6,
                "evaluator_output_encoding": "zlib_raw_f64_v1",
                "native_output_retention": "omitted_redundant_final_output",
                "p_pre_zero_compression_level": 6,
                "p_pre_zero_encoding": "zlib_raw_f64_v1",
            },
        }
        execution_manifest = {
            **manifest_body,
            "manifest_sha256": canonical_sha256(manifest_body),
        }
        manifest_path = execution_dir / "execution_manifest.json"
        manifest_raw = _write_json(manifest_path, execution_manifest)
        result_files.insert(
            0,
            {
                "path": manifest_path.relative_to(round_dir).as_posix(),
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            },
        )
        scaling_plan_body = {
            "schema_version": 1,
            "input_hashes": {"fixture_sha256": "a" * 64},
            "entries": [{"ordinal": 1}],
            "configurations": [{"method_id": "observed"}],
        }
        scaling_plan = {
            **scaling_plan_body,
            "plan_sha256": canonical_sha256(scaling_plan_body),
        }
        scaling_checkpoint_body = {
            "schema_version": 1,
            "plan_sha256": scaling_plan["plan_sha256"],
            "input_hashes": scaling_plan["input_hashes"],
            "planned_run_count": 1,
            "status": "completed",
            "datasets": [{"cells": 10_000}],
            "records": [{"run": {"run_id": "scaling-observed-fixture"}}],
        }
        scaling_checkpoint = {
            **scaling_checkpoint_body,
            "checkpoint_sha256": canonical_sha256(scaling_checkpoint_body),
        }
        scaling_checkpoint_path = (
            round_dir / "results/scaling/checkpoints/00000002.json"
        )
        scaling_checkpoint_raw = _write_json(
            scaling_checkpoint_path,
            scaling_checkpoint,
        )
        scaling_result_files = [
            {
                "path": scaling_checkpoint_path.relative_to(round_dir).as_posix(),
                "sha256": hashlib.sha256(scaling_checkpoint_raw).hexdigest(),
            }
        ]
        result_files.extend(scaling_result_files)
        scaling_evidence_body = {
            "schema_version": 1,
            "status": "completed",
            "plan": scaling_plan,
            "checkpoint_path": "results/scaling/checkpoints/00000002.json",
            "checkpoint_file_sha256": hashlib.sha256(
                scaling_checkpoint_raw
            ).hexdigest(),
            "checkpoint_payload": scaling_checkpoint,
            "result_files": scaling_result_files,
        }
        scaling_evidence = {
            **scaling_evidence_body,
            "evidence_sha256": canonical_sha256(scaling_evidence_body),
        }
        trajectory_binding = {
            "schema_version": "trajectory-execution-dataset-binding-v1",
            "dataset_id": "trajectory-exact-latent-01",
            "mechanism": "synthetic_trajectory",
            "biological_id": "trajectory-draw-01",
            "technical_view": "deterministic-count-allocation",
            "condition": "trajectory",
            "draw": 1,
            "cells": 2700,
            "genes": 120,
            "source_id": "registered-synthetic-trajectory-v1",
            "root_cell_id": "trajectory-cell-000001",
            "seed": 20250308,
            "dataset_sha256": "b" * 64,
            "dataset_file_path": "results/trajectory/dataset/evaluator.h5ad",
            "dataset_file_sha256": "c" * 64,
            "authority_path": "study/trajectory_panel.json",
            "authority_file_sha256": "d" * 64,
            "authority_sha256": "e" * 64,
            "registered_binding_sha256": "f" * 64,
        }
        trajectory_receipt_file_sha256 = "1" * 64
        trajectory_receipt_payload_sha256 = "2" * 64
        trajectory_authority_sha256 = "3" * 64
        trajectory_plan_inputs = {
            "primary_final_plan_sha256": final_plan_sha256,
            "trajectory_binding_sha256": trajectory_binding[
                "registered_binding_sha256"
            ],
            "trajectory_dataset_sha256": trajectory_binding["dataset_sha256"],
            "trajectory_dataset_file_sha256": trajectory_binding["dataset_file_sha256"],
            "trajectory_dataset_receipt_sha256": (trajectory_receipt_payload_sha256),
            "trajectory_dataset_receipt_file_sha256": (trajectory_receipt_file_sha256),
            "execution_authority_sha256": trajectory_authority_sha256,
        }
        trajectory_plan_body = {
            "schema_version": 1,
            "scope": "supplementary_trajectory",
            "input_hashes": trajectory_plan_inputs,
            "entries": [{"run": {"run_id": "trajectory-observed-fixture"}}],
            "configurations": [{"method_id": "observed"}],
            "model_seed_policy": [42, 43, 44],
        }
        trajectory_plan = {
            **trajectory_plan_body,
            "plan_sha256": canonical_sha256(trajectory_plan_body),
        }
        trajectory_paths = {
            "results/trajectory/dataset/evaluator.h5ad": trajectory_binding[
                "dataset_file_sha256"
            ],
            "results/trajectory/dataset/dataset_receipt.json": (
                trajectory_receipt_file_sha256
            ),
            "results/trajectory/execution_authority/retained_calibration.json": (
                "4" * 64
            ),
            "results/trajectory/execution_authority/count_score_authority.json": (
                "5" * 64
            ),
            "results/trajectory/execution_authority/authority.json": "6" * 64,
            "results/trajectory/execution/execution_manifest.json": "7" * 64,
        }
        trajectory_result_files = [
            {"path": path, "sha256": digest}
            for path, digest in sorted(trajectory_paths.items())
        ]
        result_files.extend(dict(row) for row in trajectory_result_files)
        trajectory_validation_body = {
            "schema_version": 1,
            "status": ("eligible_for_final_evaluation_complete_terminal_denominator"),
            "scope": "supplementary_trajectory",
            "trajectory_plan_sha256": trajectory_plan["plan_sha256"],
            "planned_run_count": 1,
            "executed_completed_count": 1,
            "executed_algorithmic_failure_count": 0,
            "executed_status_counts": {"completed": 1},
            "not_applicable_count": 0,
            "record_payload_sha256s": ["8" * 64],
        }
        trajectory_validation = {
            **trajectory_validation_body,
            "validation_sha256": canonical_sha256(trajectory_validation_body),
        }
        trajectory_authority_files = [
            row
            for row in trajectory_result_files
            if row["path"].startswith("results/trajectory/execution_authority/")
        ]
        trajectory_evidence_body = {
            "schema_version": 1,
            "status": "completed",
            "scope": "supplementary_trajectory",
            "plan": trajectory_plan,
            "dataset": {
                "binding": trajectory_binding,
                "dataset_path": trajectory_binding["dataset_file_path"],
                "dataset_file_sha256": trajectory_binding["dataset_file_sha256"],
                "dataset_sha256": trajectory_binding["dataset_sha256"],
                "receipt_path": ("results/trajectory/dataset/dataset_receipt.json"),
                "receipt_file_sha256": trajectory_receipt_file_sha256,
                "receipt_payload_sha256": trajectory_receipt_payload_sha256,
            },
            "execution_authority": {
                "authority_path": (
                    "results/trajectory/execution_authority/authority.json"
                ),
                "authority_file_sha256": "6" * 64,
                "authority_sha256": trajectory_authority_sha256,
                "count_score_authority_path": (
                    "artifacts/study/final/round-001/results/trajectory/"
                    "execution_authority/count_score_authority.json"
                ),
                "count_score_authority_file_sha256": "5" * 64,
                "retained_calibration_path": (
                    "artifacts/study/final/round-001/results/trajectory/"
                    "execution_authority/retained_calibration.json"
                ),
                "retained_calibration_file_sha256": "4" * 64,
                "files": trajectory_authority_files,
            },
            "execution_manifest": {
                "path": "results/trajectory/execution/execution_manifest.json",
                "file_sha256": "7" * 64,
                "payload_sha256": "9" * 64,
            },
            "execution_validation": trajectory_validation,
            "result_files": trajectory_result_files,
        }
        trajectory_evidence = {
            **trajectory_evidence_body,
            "evidence_sha256": canonical_sha256(trajectory_evidence_body),
        }
        validation_body = {
            "schema_version": 1,
            "status": "eligible_for_final_evaluation_complete_terminal_denominator",
            "final_plan_sha256": final_plan_sha256,
            "planned_run_count": len(record_paths),
            "executed_completed_count": len(record_paths),
            "executed_algorithmic_failure_count": 0,
            "executed_status_counts": {"completed": len(record_paths)},
            "not_applicable_count": 0,
            "record_payload_sha256s": payload_hashes,
        }
        validation = {
            **validation_body,
            "validation_sha256": canonical_sha256(validation_body),
        }
        primary_storage = {
            "completed_record_count": 0,
            "remaining_entry_count": len(record_paths),
            "remaining_execution_count": len(record_paths),
            "remaining_p_pre_zero_execution_count": 1,
            "cells": 2700,
            "genes": 1200,
            "per_execution_compressed_bound_bytes": 1,
            "per_p_pre_zero_compressed_bound_bytes": 1,
            "required_free_bytes": 1,
        }
        trajectory_storage = {
            "completed_record_count": 0,
            "remaining_entry_count": 1,
            "remaining_execution_count": 1,
            "remaining_p_pre_zero_execution_count": 0,
            "cells": 2700,
            "genes": 120,
            "per_execution_compressed_bound_bytes": 1,
            "per_p_pre_zero_compressed_bound_bytes": 1,
            "required_free_bytes": 1,
        }
        scaling_storage_body = {
            "schema": "maskimpute-scaling-storage-preflight-v1",
            "plan_sha256": scaling_plan["plan_sha256"],
            "planned_run_count": 1,
            "required_free_bytes": 1,
        }
        scaling_storage = {
            **scaling_storage_body,
            "receipt_sha256": canonical_sha256(scaling_storage_body),
        }
        storage_required = 1024**3 + 3
        evaluation_manifest = {
            "schema_version": 1,
            "status": "completed",
            "final_plan_sha256": final_plan_sha256,
            "final_execution_manifest_path": (
                manifest_path.relative_to(round_dir).as_posix()
            ),
            "final_execution_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "final_execution_payload_sha256": execution_manifest["manifest_sha256"],
            "execution_validation": validation,
            "scaling_evidence": scaling_evidence,
            "trajectory_evidence": trajectory_evidence,
            "storage_preflight": {
                "schema": "maskimpute-combined-final-storage-preflight-v1",
                "primary": primary_storage,
                "trajectory": trajectory_storage,
                "scaling": scaling_storage,
                "reserve_bytes": 1024**3,
                "required_free_bytes": storage_required,
                "observed_free_bytes": storage_required,
            },
            "result_files": result_files,
        }
        receipt = {
            "result_manifest": evaluation_manifest,
            "result_manifest_sha256": canonical_sha256(evaluation_manifest),
        }
        fixture.update(
            {
                "execution_manifest": execution_manifest,
                "execution_manifest_path": manifest_path,
                "evaluation_manifest": evaluation_manifest,
                "receipt": receipt,
            }
        )

    fixture["refresh_bindings"] = refresh_bindings
    refresh_bindings()

    monkeypatch.setattr(study, "_validate_freeze", lambda _round, _repo: freeze)
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
            fixture["receipt"],
        ),
    )
    monkeypatch.setattr(
        study,
        "_validate_result_files",
        lambda _repo, _round, manifest: frozenset(
            (round_dir / row["path"]).relative_to(repository).as_posix()
            for row in manifest["result_files"]
        ),
    )
    monkeypatch.setattr(
        study,
        "_verify_frozen_repository",
        lambda _repo, _round, *, allowed_result_paths: freeze,
    )
    return fixture


def test_denominator_preserves_every_terminal_status_and_normalizes_only_success() -> (
    None
):
    records = [
        _record(
            "maskimpute",
            "completed",
            ordinal=1,
            values={"mse": 1.0, "corr_err": None},
            metric_statuses={"mse": "completed", "corr_err": "unavailable"},
            metric_reasons={"mse": None, "corr_err": "too_few_correlated_genes"},
        ),
        _record(
            "maskimpute",
            "failed",
            ordinal=2,
            biological_id="draw-02",
            run_reason="algorithm_failure",
        ),
        _record(
            "maskimpute",
            "timeout",
            ordinal=3,
            biological_id="draw-03",
            run_reason="wall_clock_limit",
        ),
        _record(
            "maskimpute",
            "resource_exceeded",
            ordinal=4,
            biological_id="draw-04",
            run_reason="rss_limit",
        ),
        _record(
            "maskimpute",
            "unavailable",
            ordinal=5,
            biological_id="draw-05",
            run_reason="method_not_applicable",
        ),
    ]

    report = _analysis(records)
    denominator = report["denominator"]

    assert report["analysis_policy"]["analytic_status_normalization"] == {
        "completed": "ok",
        "preserved_terminal_statuses": [
            "failed",
            "timeout",
            "resource_exceeded",
            "unavailable",
        ],
    }
    assert denominator == {
        "completeness": {
            "complete": True,
            "expected_metric_names": ["mse", "corr_err"],
            "metric_row_count": 10,
            "metric_rows_per_run": 2,
            "planned_run_count": 5,
            "recorded_run_count": 5,
        },
        "metric_analytic_status_counts": {
            "failed": 2,
            "ok": 1,
            "resource_exceeded": 2,
            "timeout": 2,
            "unavailable": 3,
        },
        "metric_unavailable_reason_counts": {
            "algorithm_failure": 2,
            "method_not_applicable": 2,
            "rss_limit": 2,
            "too_few_correlated_genes": 1,
            "wall_clock_limit": 2,
        },
        "run_terminal_status_counts": {
            "completed": 1,
            "failed": 1,
            "resource_exceeded": 1,
            "timeout": 1,
            "unavailable": 1,
        },
        "run_unavailable_reason_counts": {
            "algorithm_failure": 1,
            "method_not_applicable": 1,
            "rss_limit": 1,
            "wall_clock_limit": 1,
        },
    }


def test_duplicate_metric_result_identity_is_rejected_before_row_weighting() -> None:
    records = _paired_panel()
    duplicate = json.loads(json.dumps(records[0]))
    duplicate["run"]["run_id"] = "different-run-id"
    records.append(duplicate)

    with pytest.raises(
        FinalAnalysisContractError, match="duplicate metric result identity"
    ):
        _analysis(records)


def test_summary_collapses_seeds_and_views_before_biological_draw_iqr() -> None:
    report = _analysis(_paired_panel())

    summary = _matching(
        report["descriptive_summaries"], method="maskimpute", metric="mse"
    )
    assert {
        key: value
        for key, value in summary.items()
        if key
        not in {"first_quartile", "interquartile_range", "median", "third_quartile"}
    } == {
        "method": "maskimpute",
        "metric": "mse",
        "n_biological_draws": 4,
        "n_dataset_views": 8,
        "n_mechanisms": 1,
        "n_raw_metric_rows": 24,
        "reason": None,
        "status": "ok",
        "unit": "biological_draw",
    }
    assert summary["first_quartile"] == pytest.approx(1.75)
    assert summary["interquartile_range"] == pytest.approx(1.5)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["third_quartile"] == pytest.approx(3.25)

    unavailable = _matching(
        report["descriptive_summaries"], method="magic", metric="mse"
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "no_finite_ok_biological_draws"
    assert unavailable["median"] is None
    assert unavailable["n_biological_draws"] == 0
    assert unavailable["n_raw_metric_rows"] == 8


def test_paired_bootstrap_uses_draws_for_wins_and_emits_compact_distribution_binding() -> (
    None
):
    report = _analysis(_paired_panel())

    comparison = _matching(
        report["paired_comparisons"],
        comparator_method_id="dca",
        metric="mse",
    )
    assert comparison["status"] == "ok"
    assert comparison["reason"] is None
    assert comparison["candidate_method_id"] == "maskimpute"
    assert comparison["favorable_direction"] == "lower"
    assert comparison["direction_source"] == (
        "validated_frozen_metric_direction_contract"
    )
    assert comparison["median_relative_effect"] == pytest.approx(-0.375)
    assert comparison["biological_draw_wins"] == 3
    assert comparison["biological_draw_ties"] == 1
    assert comparison["biological_draw_losses"] == 0
    assert comparison["n_independent_biological_draws"] == 4
    assert comparison["n_paired_dataset_views"] == 8
    assert comparison["n_raw_metric_rows"] == 32
    assert 0.0 <= comparison["probability_of_improvement"] <= 1.0
    assert comparison["bootstrap"] == {
        "distribution_sha256": comparison["bootstrap"]["distribution_sha256"],
        "replicates_available": 10_000,
        "replicates_requested": 10_000,
        "seed": 20_260_712,
    }
    assert len(comparison["bootstrap"]["distribution_sha256"]) == 64
    assert "bootstrap_distribution" not in comparison


def test_holm_adjustment_is_within_each_declared_comparator_metric_family() -> None:
    report = _analysis(_paired_panel())
    dca = [
        row
        for row in report["paired_comparisons"]
        if row["comparator_method_id"] == "dca"
    ]
    magic = [
        row
        for row in report["paired_comparisons"]
        if row["comparator_method_id"] == "magic"
    ]

    assert [row["metric"] for row in dca] == list(PRIMARY_METRICS)
    assert all(row["holm_family_id"] == "protocol_primary_metrics" for row in dca)
    assert all(row["holm_hypothesis_count"] == 2 for row in dca)
    assert all(row["holm_status"] == "ok" for row in dca)
    assert all(
        row["two_sided_sign_probability"] <= row["holm_adjusted_p_value"] <= 1.0
        for row in dca
    )

    assert len(magic) == 2
    assert all(row["status"] == "unavailable" for row in magic)
    assert all(row["reason"] == "no_paired_biological_draws" for row in magic)
    assert all(row["holm_status"] == "unavailable" for row in magic)
    assert all(row["holm_reason"] == "raw_p_value_unavailable" for row in magic)
    assert all(row["holm_hypothesis_count"] == 0 for row in magic)


def test_variance_components_separate_seed_view_and_biological_draw_levels() -> None:
    report = _analysis(_paired_panel())

    candidate = _matching(
        report["variance_components"], method="maskimpute", metric="mse"
    )
    assert candidate["within_dataset_view_seed_variance"] == {
        "estimate": pytest.approx(0.09),
        "n_identifiable_groups": 8,
        "reason": None,
        "status": "ok",
    }
    assert candidate["between_technical_view_variance"] == {
        "estimate": pytest.approx(0.02),
        "n_identifiable_groups": 4,
        "reason": None,
        "status": "ok",
    }
    assert candidate["between_biological_draw_variance"] == {
        "estimate": pytest.approx(5.0 / 3.0),
        "n_identifiable_groups": 1,
        "reason": None,
        "status": "ok",
    }
    assert candidate["inference_unit"] == "biological_draw"

    deterministic = _matching(report["variance_components"], method="dca", metric="mse")
    assert deterministic["within_dataset_view_seed_variance"] == {
        "estimate": None,
        "n_identifiable_groups": 0,
        "reason": "fewer_than_two_seed_levels_per_dataset_view",
        "status": "unavailable",
    }
    assert deterministic["between_technical_view_variance"]["estimate"] == 0.0
    assert deterministic["between_biological_draw_variance"]["estimate"] == 0.0

    failed = _matching(report["variance_components"], method="magic", metric="mse")
    assert failed["within_dataset_view_seed_variance"]["status"] == "unavailable"
    assert failed["within_dataset_view_seed_variance"]["reason"] == "no_finite_ok_rows"
    assert failed["exclusions"]["failed_rows"] == 8


def test_pareto_retains_unavailable_methods_without_calling_them_dominated() -> None:
    report = _analysis(_paired_panel())
    pareto = report["pareto"]
    direction_contract = report["analysis_policy"]["metric_direction_contract"]

    direction_body = {
        key: value
        for key, value in direction_contract.items()
        if key != "contract_sha256"
    }
    assert direction_contract["status"] == "validated"
    assert direction_contract["reason"] is None
    assert direction_contract["authority"] == {
        "source": "synthetic_frozen_selection_gate_fixture",
        "method_commit": "f" * 40,
    }
    assert direction_contract["favorable_direction"] == "lower"
    assert {"mse", "corr_err"}.issubset(direction_contract["metrics"])
    assert direction_contract["contract_sha256"] == canonical_sha256(direction_body)
    assert pareto["status"] == "ok"
    assert pareto["core_metrics"] == ["mse", "corr_err"]
    assert pareto["direction_source"] == "validated_frozen_metric_direction_contract"
    assert pareto["excluded_primary_metrics"] == []
    assert pareto["complete_method_count"] == 2
    candidate = _matching(pareto["methods"], method="maskimpute")
    comparator = _matching(pareto["methods"], method="dca")
    unavailable = _matching(pareto["methods"], method="magic")
    assert candidate["non_dominated"] is True
    assert candidate["dominated_by"] == []
    assert comparator["non_dominated"] is False
    assert comparator["dominated_by"] == ["maskimpute"]
    assert unavailable == {
        "dominated_by": [],
        "method": "magic",
        "missing_metrics": ["mse", "corr_err"],
        "non_dominated": None,
        "reason": "incomplete_core_metric_denominator",
        "status": "unavailable",
    }


def test_pareto_requires_the_full_terminal_draw_and_view_denominator() -> None:
    records = _paired_panel()
    for record in records:
        run = record["run"]
        if run["method_id"] != "maskimpute" or run["biological_id"] == "draw-01":
            continue
        run["status"] = "failed"
        run["reason"] = "algorithm_failure"
        for metric in record["metrics"]:
            metric["value"] = None
            metric["n"] = 0
            metric["status"] = "failed"
            metric["reason"] = "algorithm_failure"

    report = _analysis(records)

    candidate_summary = _matching(
        report["descriptive_summaries"], method="maskimpute", metric="mse"
    )
    assert candidate_summary["status"] == "ok"
    assert candidate_summary["n_biological_draws"] == 1
    assert candidate_summary["n_dataset_views"] == 2
    candidate_pareto = _matching(report["pareto"]["methods"], method="maskimpute")
    assert candidate_pareto == {
        "dominated_by": [],
        "method": "maskimpute",
        "missing_metrics": ["mse", "corr_err"],
        "non_dominated": None,
        "reason": "incomplete_core_metric_denominator",
        "status": "unavailable",
    }
    assert report["pareto"]["status"] == "unavailable"
    assert report["pareto"]["complete_method_count"] == 1
    assert report["pareto"]["reason"] == (
        "fewer_than_two_methods_have_complete_core_metrics"
    )
    assert report["denominator"]["run_terminal_status_counts"]["failed"] == 26


def test_pareto_accepts_only_exact_truth_kind_structural_unavailability() -> None:
    truth_kinds = {
        "symsim": ("exact_pre_capture", None),
        "sergio": ("exact_continuous", "undefined_for_continuous_truth"),
        "sparsim": ("exact_continuous", "undefined_for_continuous_truth"),
        "semisynthetic": ("proxy_high_depth", "proxy_truth_not_exact"),
    }
    records: list[dict[str, Any]] = []
    ordinal = 0
    for mechanism, (truth_kind, structural_reason) in truth_kinds.items():
        for method, mse, prezero_mse in (
            ("maskimpute", 0.5, 0.1),
            ("dca", 1.0, 0.2),
        ):
            ordinal += 1
            applicable = structural_reason is None
            record = _record(
                method,
                "completed",
                ordinal=ordinal,
                mechanism=mechanism,
                values={
                    "mse": mse,
                    "corr_err": prezero_mse if applicable else None,
                },
                metric_statuses={
                    "mse": "completed",
                    "corr_err": "completed" if applicable else "unavailable",
                },
                metric_reasons={
                    "mse": None,
                    "corr_err": structural_reason,
                },
            )
            record["run"]["truth_kind"] = truth_kind
            record["metrics"][1]["metric"] = "mse_pre_dropout_zero"
            records.append(record)

    report = _analysis(
        records,
        primary_metrics=("mse", "mse_pre_dropout_zero"),
    )

    assert report["pareto"]["status"] == "ok"
    assert report["pareto"]["complete_method_count"] == 2
    candidate = _matching(report["pareto"]["methods"], method="maskimpute")
    assert candidate["status"] == "ok"
    assert candidate["non_dominated"] is True

    malformed = next(
        record
        for record in records
        if record["run"]["method_id"] == "maskimpute"
        and record["run"]["mechanism"] == "sergio"
    )
    malformed["metrics"][1]["reason"] = "algorithm_failure"
    malformed_report = _analysis(
        records,
        primary_metrics=("mse", "mse_pre_dropout_zero"),
    )
    malformed_candidate = _matching(
        malformed_report["pareto"]["methods"], method="maskimpute"
    )
    assert malformed_candidate["status"] == "unavailable"
    assert malformed_candidate["missing_metrics"] == ["mse_pre_dropout_zero"]


def test_nonerror_primary_endpoint_is_not_assumed_to_be_lower_better() -> None:
    records = _paired_panel()
    for record in records:
        for metric in record["metrics"]:
            if metric["metric"] == "corr_err":
                metric["metric"] = "auroc"
    report = build_final_analysis(
        records,
        protocol={"schema_version": 1, "primary_metrics": ["auroc"]},
        selection_contract={
            "schema_version": 1,
            "candidate_method_id": "maskimpute",
        },
        input_bindings={"planned_run_count": len(records)},
    )

    comparison = _matching(
        report["paired_comparisons"], comparator_method_id="dca", metric="auroc"
    )
    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == "metric_direction_not_declared_lower"
    assert comparison["probability_of_improvement"] is None
    assert report["pareto"] == {
        "complete_method_count": 0,
        "core_metrics": [],
        "direction_source": None,
        "excluded_primary_metrics": [
            {
                "metric": "auroc",
                "reason": "not_prespecified_lower_better_reconstruction_metric",
            }
        ],
        "methods": [],
        "reason": "no_explicit_lower_better_core_metrics",
        "status": "unavailable",
    }


def test_missing_frozen_direction_authority_disables_reconstruction_inference() -> None:
    records = _paired_panel()
    report = build_final_analysis(
        records,
        protocol={"schema_version": 1, "primary_metrics": list(PRIMARY_METRICS)},
        selection_contract={
            "schema_version": 1,
            "candidate_method_id": "maskimpute",
        },
        input_bindings={"planned_run_count": len(records)},
    )

    direction = report["analysis_policy"]["metric_direction_contract"]
    assert direction["status"] == "unavailable"
    assert direction["reason"] == "frozen_metric_direction_authority_absent"
    comparison = _matching(
        report["paired_comparisons"], comparator_method_id="dca", metric="mse"
    )
    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == "metric_direction_not_declared_lower"
    assert report["pareto"]["status"] == "unavailable"
    assert report["pareto"]["core_metrics"] == []


def test_metric_direction_contract_checksum_is_fail_closed() -> None:
    records = _paired_panel()
    direction_body = {
        "schema_version": 1,
        "status": "validated",
        "reason": None,
        "favorable_direction": "lower",
        "metrics": ["corr_err", "mse"],
        "authority": {"source": "frozen_fixture"},
    }
    direction = {
        **direction_body,
        "contract_sha256": canonical_sha256(direction_body),
    }
    direction["metrics"] = ["auroc", "corr_err", "mse"]

    with pytest.raises(FinalAnalysisContractError, match="direction contract checksum"):
        build_final_analysis(
            records,
            protocol={"schema_version": 1, "primary_metrics": list(PRIMARY_METRICS)},
            selection_contract={
                "schema_version": 1,
                "candidate_method_id": "maskimpute",
            },
            input_bindings={
                "planned_run_count": len(records),
                "metric_direction_contract": direction,
            },
        )


def test_direction_authority_binds_frozen_gates_to_method_commit_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    method_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rank_metrics = ("mse", "mse_dropout", "gnrmse")
    pareto_metrics = (
        "mse_dropout",
        "mse_pre_dropout_zero",
        "corr_err",
        "mse_non_dropout_nonzero",
    )
    gates = {
        **{
            f"rank_{metric}": {
                "threshold": "median biological-draw rank <= 2",
            }
            for metric in rank_metrics
        },
        "pareto_non_dominated": {
            "threshold": (
                "no same-input method weakly better on all four dimensions and "
                "strictly better on one"
            ),
            "details": {"dimensions": list(pareto_metrics)},
        },
    }
    assessment = {"configuration_id": "candidate-01", "gates": gates}
    frozen_method = {
        "selected_configuration_id": "candidate-01",
        "selected_assessment": assessment,
        "selection_gate_table": [assessment],
        "payload_sha256": "a" * 64,
    }
    protocol = {
        "primary_metrics": [
            "mse",
            "mse_dropout",
            "mse_pre_dropout_zero",
            "gnrmse",
            "corr_err",
        ]
    }
    monkeypatch.setenv("GIT_DIR", "/attacker-controlled/not-the-study-repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/attacker-controlled/worktree")
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")

    contract = _frozen_metric_direction_contract(
        repository,
        frozen_method,
        {"method_commit": method_commit},
        protocol,
    )

    assert contract["status"] == "validated"
    assert contract["favorable_direction"] == "lower"
    assert contract["metrics"] == sorted(set((*rank_metrics, *pareto_metrics)))
    assert contract["authority"]["method_commit"] == method_commit
    assert contract["authority"]["source"] == (
        "frozen_selection_gates_validated_against_method_commit"
    )
    assert contract["contract_sha256"] == canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )

    frozen_method["selected_assessment"]["gates"]["pareto_non_dominated"]["details"][
        "dimensions"
    ] = ["mse"]
    with pytest.raises(FinalAnalysisContractError, match="frozen Pareto gate differs"):
        _frozen_metric_direction_contract(
            repository,
            frozen_method,
            {"method_commit": method_commit},
            protocol,
        )


def test_direction_source_lookup_disables_git_replace_objects(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    def git(*arguments: str, input_text: str | None = None) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            input=input_text,
        ).stdout.strip()

    git("init", "--quiet")
    selection_path = repository / "maskimpute_benchmark/selection.py"
    selection_path.parent.mkdir()
    original_raw = b"ORIGINAL_SELECTION_SOURCE = True\n"
    selection_path.write_bytes(original_raw)
    git("add", "maskimpute_benchmark/selection.py")
    git(
        "-c",
        "user.name=Final Analysis Test",
        "-c",
        "user.email=final-analysis@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "original",
    )
    original_commit = git("rev-parse", "HEAD")
    original_blob = git(
        "rev-parse",
        f"{original_commit}:maskimpute_benchmark/selection.py",
    )

    selection_path.write_text("REPLACEMENT_SELECTION_SOURCE = True\n")
    git("add", "maskimpute_benchmark/selection.py")
    replacement_tree = git("write-tree")
    replacement_commit = git(
        "-c",
        "user.name=Final Analysis Test",
        "-c",
        "user.email=final-analysis@example.invalid",
        "commit-tree",
        replacement_tree,
        input_text="replacement\n",
    )
    git("replace", original_commit, replacement_commit)
    redirected_blob = git(
        "rev-parse",
        f"{original_commit}:maskimpute_benchmark/selection.py",
    )
    assert redirected_blob != original_blob

    observed_blob, observed_raw = _git_object_bytes(repository, original_commit)

    assert observed_blob == original_blob
    assert observed_raw == original_raw


def test_prezero_score_evidence_is_a_separate_complete_descriptive_family() -> None:
    records = _paired_panel()
    for record in records:
        _attach_score_evidence(record)

    report = _analysis(records)
    family = report["score_evidence"]

    assert family["status"] == "ok"
    assert family["family_id"] == "p_pre_zero_score_metrics"
    assert family["separate_from_metric_family"] == "protocol_primary_metrics"
    assert family["metric_roles"] == {
        "auprc": {"favorable_direction": "higher", "role": "efficacy"},
        "auroc": {"favorable_direction": "higher", "role": "efficacy"},
        "brier": {"favorable_direction": "lower", "role": "efficacy"},
        "calibration_intercept": {
            "favorable_direction": None,
            "role": "descriptive_calibration",
        },
        "calibration_slope": {
            "favorable_direction": None,
            "role": "descriptive_calibration",
        },
        "ece": {"favorable_direction": "lower", "role": "efficacy"},
        "log_loss": {"favorable_direction": "lower", "role": "efficacy"},
    }
    assert family["paired_inference"] == {
        "inference_unit": "biological_draw",
        "reason": "comparator_methods_do_not_emit_p_pre_zero",
        "status": "unavailable",
    }
    assert family["multiplicity"] == {
        "family_id": "p_pre_zero_score_metrics",
        "reason": "no_prespecified_pairwise_score_hypotheses",
        "status": "not_applicable",
    }
    assert family["expected_groups"] == {
        "library_size_quartiles": ["Q1", "Q2", "Q3", "Q4"],
        "overall": ["all_observed_zeros"],
        "truth_expression_bins": ["[0,1)", "[1,2)", "[2,4)", "[4,inf)"],
    }

    overall = _matching(
        family["group_summaries"],
        method="maskimpute",
        metric="auroc",
        stratum_type="overall",
        label="all_observed_zeros",
    )
    assert overall["status"] == "ok"
    assert overall["favorable_direction"] == "higher"
    assert overall["median"] == pytest.approx(0.75)
    assert overall["first_quartile"] == pytest.approx(0.675)
    assert overall["third_quartile"] == pytest.approx(0.825)
    assert overall["n_biological_draws"] == 4
    assert overall["n_dataset_views"] == 8
    assert overall["n_raw_metric_rows"] == 24
    assert overall["entry_denominator"] == {
        "maximum": 100,
        "median": 100.0,
        "minimum": 100,
        "unit": "observed_zero_entries_per_dataset_view",
    }

    unavailable = _matching(
        family["group_summaries"],
        method="dca",
        metric="auroc",
        stratum_type="overall",
        label="all_observed_zeros",
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "no_finite_ok_biological_draws"
    assert unavailable["unavailable_reason_counts"] == {
        "method_does_not_emit_p_pre_zero": 8
    }
    assert unavailable["n_raw_metric_rows"] == 8

    assert len(family["group_summaries"]) == 3 * 9 * 7
    assert [
        row["metric"]
        for row in report["paired_comparisons"]
        if row["comparator_method_id"] == "dca"
    ] == list(PRIMARY_METRICS)
    assert report["pareto"]["core_metrics"] == list(PRIMARY_METRICS)


def test_conversion_terminal_run_preserves_its_authorized_score_artifact() -> None:
    record = _record(
        "maskimpute",
        "unavailable",
        ordinal=1,
        model_seed=42,
        run_reason="output_conversion_failed",
    )
    _attach_score_evidence(record, matrix_present=True)

    report = _analysis([record])

    overall = _matching(
        report["score_evidence"]["group_summaries"],
        method="maskimpute",
        metric="auroc",
        stratum_type="overall",
        label="all_observed_zeros",
    )
    assert overall["status"] == "unavailable"
    assert overall["unavailable_reason_counts"] == {
        "output_conversion_failed": 1,
    }


def test_evidence_loader_binds_evaluated_receipt_manifest_and_ordered_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    before = sorted(
        path.relative_to(fixture["round_dir"]).as_posix()
        for path in fixture["round_dir"].rglob("*")
    )

    report = generate_final_analysis(fixture["repository"], fixture["round_dir"])

    bindings = report["input_bindings"]
    assert bindings["planned_run_count"] == 2
    assert bindings["final_plan_sha256"] == fixture["final_plan_sha256"]
    assert (
        bindings["result_manifest_sha256"]
        == fixture["receipt"]["result_manifest_sha256"]
    )
    assert (
        bindings["final_execution_manifest_sha256"]
        == fixture["evaluation_manifest"]["final_execution_manifest_sha256"]
    )
    assert bindings["record_bindings"] == [
        {
            "ordinal": index,
            "path": f"records/{index:08d}.json",
            "payload_sha256": canonical_sha256(record),
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run_id": record["run"]["run_id"],
        }
        for index, (record, path) in enumerate(
            zip(fixture["records"], fixture["record_paths"], strict=True),
            start=1,
        )
    ]
    assert report["denominator"]["execution_action_denominator"] == {
        "executed_algorithmic_failure_count": 0,
        "executed_completed_count": 2,
        "executed_run_count": 2,
        "executed_terminal_status_counts": {"completed": 2},
        "not_applicable_count": 0,
        "status": "validated",
    }
    body = {key: value for key, value in report.items() if key != "analysis_sha256"}
    assert report["analysis_sha256"] == canonical_sha256(body)
    assert before == sorted(
        path.relative_to(fixture["round_dir"]).as_posix()
        for path in fixture["round_dir"].rglob("*")
    )


def test_evidence_loader_rejects_record_tamper_even_when_json_remains_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    payload = json.loads(fixture["record_paths"][0].read_bytes())
    payload["metrics"][0]["value"] = 99.0
    _write_json(fixture["record_paths"][0], payload)

    with pytest.raises(FinalAnalysisContractError, match="record.*hash"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_rejects_noncanonical_record_even_if_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    path = fixture["record_paths"][0]
    path.write_text(json.dumps(fixture["records"][0], indent=2) + "\n")
    fixture["refresh_bindings"]()

    with pytest.raises(FinalAnalysisContractError, match="record.*canonical"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_rejects_prezero_artifact_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    storage = fixture["records"][0]["p_pre_zero_evidence"]["storage"]
    artifact = fixture["execution_dir"] / storage["path"]
    artifact.write_bytes(b"tamper12")

    with pytest.raises(FinalAnalysisContractError, match="p_pre_zero artifact"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_requires_prezero_artifact_in_evaluated_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    storage = fixture["records"][0]["p_pre_zero_evidence"]["storage"]
    artifact_path = (
        (fixture["execution_dir"] / storage["path"])
        .relative_to(fixture["round_dir"])
        .as_posix()
    )
    evaluation = fixture["evaluation_manifest"]
    evaluation["result_files"] = [
        row for row in evaluation["result_files"] if row["path"] != artifact_path
    ]
    fixture["receipt"]["result_manifest_sha256"] = canonical_sha256(evaluation)

    with pytest.raises(
        FinalAnalysisContractError,
        match="p_pre_zero artifact is not identically bound",
    ):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_rejects_prezero_storage_content_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    path = fixture["record_paths"][0]
    payload = json.loads(path.read_bytes())
    payload["p_pre_zero_evidence"]["storage"]["uncompressed_nbytes"] -= 1
    _write_json(path, payload)
    fixture["refresh_bindings"]()

    with pytest.raises(
        FinalAnalysisContractError,
        match="p_pre_zero storage content binding differs",
    ):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_requires_new_prezero_record_schema_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    for path in fixture["record_paths"]:
        payload = json.loads(path.read_bytes())
        del payload["p_pre_zero_evidence"]
        _write_json(path, payload)
    fixture["refresh_bindings"]()

    with pytest.raises(
        FinalAnalysisContractError, match="final execution record.*schema"
    ):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_rejects_manifest_record_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    manifest = fixture["execution_manifest"]
    manifest["records"][0]["path"] = "../outside.json"
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(body)
    raw = _write_json(fixture["execution_manifest_path"], manifest)
    evaluation = fixture["evaluation_manifest"]
    evaluation["final_execution_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    evaluation["final_execution_payload_sha256"] = manifest["manifest_sha256"]
    for row in evaluation["result_files"]:
        if row["path"] == "results/final/execution/execution_manifest.json":
            row["sha256"] = hashlib.sha256(raw).hexdigest()
    fixture["receipt"]["result_manifest_sha256"] = canonical_sha256(evaluation)

    with pytest.raises(FinalAnalysisContractError, match="record path"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_rejects_malformed_execution_input_hash_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    manifest = fixture["execution_manifest"]
    manifest["input_hashes"]["runtime_lock_sha256"] = "not-a-digest"
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(body)
    raw = _write_json(fixture["execution_manifest_path"], manifest)
    evaluation = fixture["evaluation_manifest"]
    evaluation["final_execution_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    evaluation["final_execution_payload_sha256"] = manifest["manifest_sha256"]
    for row in evaluation["result_files"]:
        if row["path"] == "results/final/execution/execution_manifest.json":
            row["sha256"] = hashlib.sha256(raw).hexdigest()
    fixture["receipt"]["result_manifest_sha256"] = canonical_sha256(evaluation)

    with pytest.raises(FinalAnalysisContractError, match="execution input hashes"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_requires_exact_evaluation_manifest_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    del fixture["evaluation_manifest"]["storage_preflight"]
    fixture["receipt"]["result_manifest_sha256"] = canonical_sha256(
        fixture["evaluation_manifest"]
    )

    with pytest.raises(FinalAnalysisContractError, match="evaluation manifest schema"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_requires_trajectory_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    del fixture["evaluation_manifest"]["trajectory_evidence"]
    fixture["receipt"]["result_manifest_sha256"] = canonical_sha256(
        fixture["evaluation_manifest"]
    )

    with pytest.raises(FinalAnalysisContractError, match="evaluation manifest schema"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_requires_exact_storage_preflight_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    del fixture["evaluation_manifest"]["storage_preflight"]["observed_free_bytes"]
    fixture["receipt"]["result_manifest_sha256"] = canonical_sha256(
        fixture["evaluation_manifest"]
    )

    with pytest.raises(FinalAnalysisContractError, match="storage preflight schema"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_combined_storage_preflight_requires_one_shared_reserve() -> None:
    from maskimpute_benchmark.final_analysis import _validate_storage_preflight

    def component(*, completed: int, remaining: int, cells: int, genes: int):
        return {
            "completed_record_count": completed,
            "remaining_entry_count": remaining,
            "remaining_execution_count": remaining,
            "remaining_p_pre_zero_execution_count": 0,
            "cells": cells,
            "genes": genes,
            "per_execution_compressed_bound_bytes": 100,
            "per_p_pre_zero_compressed_bound_bytes": 100,
            "required_free_bytes": remaining * 100,
        }

    scaling_body = {
        "schema": "maskimpute-scaling-storage-preflight-v1",
        "required_free_bytes": 300,
    }
    scaling = {
        **scaling_body,
        "receipt_sha256": canonical_sha256(scaling_body),
    }
    reserve = 1024**3
    value = {
        "schema": "maskimpute-combined-final-storage-preflight-v1",
        "primary": component(completed=2, remaining=3, cells=2700, genes=1200),
        "trajectory": component(completed=0, remaining=4, cells=2700, genes=120),
        "scaling": scaling,
        "reserve_bytes": reserve,
        "required_free_bytes": reserve + 300 + 300 + 400,
        "observed_free_bytes": reserve + 300 + 300 + 400,
    }

    assert (
        _validate_storage_preflight(
            value,
            planned_run_count=5,
            require_combined=True,
        )
        == value
    )
    value["reserve_bytes"] += 1
    with pytest.raises(FinalAnalysisContractError, match="combined.*denominator"):
        _validate_storage_preflight(
            value,
            planned_run_count=5,
            require_combined=True,
        )


def test_evidence_loader_detects_record_change_during_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_analysis as final_analysis

    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    original = final_analysis.build_final_analysis

    def mutate_after_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        report = original(*args, **kwargs)
        payload = json.loads(fixture["record_paths"][0].read_bytes())
        payload["metrics"][0]["value"] = 99.0
        _write_json(fixture["record_paths"][0], payload)
        return report

    monkeypatch.setattr(final_analysis, "build_final_analysis", mutate_after_analysis)

    with pytest.raises(FinalAnalysisContractError, match="changed during analysis"):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_detects_prezero_artifact_change_during_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_analysis as final_analysis

    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    original = final_analysis.build_final_analysis
    storage = fixture["records"][0]["p_pre_zero_evidence"]["storage"]
    artifact = fixture["execution_dir"] / storage["path"]

    def mutate_after_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        report = original(*args, **kwargs)
        artifact.write_bytes(b"changed!")
        return report

    monkeypatch.setattr(final_analysis, "build_final_analysis", mutate_after_analysis)

    with pytest.raises(
        FinalAnalysisContractError,
        match="p_pre_zero score artifact changed during analysis",
    ):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_evidence_loader_detects_protocol_change_during_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_analysis as final_analysis

    fixture = _evaluated_evidence(tmp_path, monkeypatch)
    original = final_analysis.build_final_analysis

    def mutate_after_analysis(*args: object, **kwargs: object) -> dict[str, object]:
        report = original(*args, **kwargs)
        protocol_path = fixture["repository"] / "study/protocol.json"
        protocol_path.write_text(
            json.dumps(fixture["protocol"], indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    monkeypatch.setattr(final_analysis, "build_final_analysis", mutate_after_analysis)

    with pytest.raises(
        FinalAnalysisContractError, match="protocol changed during analysis"
    ):
        generate_final_analysis(fixture["repository"], fixture["round_dir"])


def test_final_analysis_public_generator_has_no_scientific_overrides() -> None:
    assert tuple(inspect.signature(generate_final_analysis).parameters) == (
        "repository",
        "round_dir",
    )


def test_final_analysis_cli_exposes_only_the_round_locator() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_final_analysis.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "round_dir" in completed.stdout
    for forbidden in (
        "--bootstrap",
        "--seed",
        "--metric",
        "--candidate",
        "--comparator",
        "--output",
    ):
        assert forbidden not in completed.stdout


def test_final_analysis_cli_prints_one_canonical_self_hashed_json_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import maskimpute_benchmark.final_analysis as final_analysis

    body = {"schema_version": 1, "status": "completed"}
    report = {**body, "analysis_sha256": canonical_sha256(body)}
    monkeypatch.setattr(
        final_analysis,
        "generate_final_analysis",
        lambda _repository, _round_dir: report,
    )
    script = Path(__file__).resolve().parents[1] / "scripts/generate_final_analysis.py"
    monkeypatch.setattr(sys, "argv", [str(script), "artifacts/study/round-001"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == _canonical_bytes(report).decode() + "\n"
