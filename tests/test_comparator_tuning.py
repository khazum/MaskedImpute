import copy
from dataclasses import asdict, replace
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import struct
import sys
import threading

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from maskimpute_benchmark.comparator_tuning import (
    AUTHORITY_REVISION,
    COMPARATOR_SELECTION_RELATIVE_PATH,
    ComparatorSmokeOutcome,
    ComparatorTuningError,
    DEVELOPMENT_MAX_CHECKPOINT_BYTES,
    DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
    DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
    DEVELOPMENT_MAX_RECORD_BYTES,
    DEVELOPMENT_STORAGE_RESERVE_BYTES,
    bind_comparator_configuration_identity,
    build_comparator_smoke_input,
    build_comparator_smoke_receipt,
    collapse_comparator_configuration,
    decode_comparator_configuration,
    encode_comparator_configuration,
    load_comparator_smoke_receipt,
    load_comparator_tuning_authority,
    metric_rank_quarters,
    pareto_configuration_ids,
    parse_comparator_tuning_authority,
    run_comparator_tuning_smoke,
    select_one_comparator_method,
)
import maskimpute_benchmark.comparator_tuning as comparator_tuning_module
from maskimpute_benchmark.direct_values import direct_json_value, freeze_direct_mapping
from maskimpute_benchmark.fair_comparator_checkpoint import (
    replay_direct_development_budget,
)
from maskimpute_benchmark.fair_comparator_plan import (
    ComparatorRunIdentity,
    _build_structural_direct_competition_plan,
    bind_comparator_smoke_receipt_to_plan,
)
from maskimpute_benchmark.methods import load_method_registry, prepare_method_input
from maskimpute_benchmark.runner import (
    AdapterOutcome,
    DatasetQCAudit,
    ExecutionEnvironmentRegistry,
    PreparedDataset,
    RepositoryAdapterDispatcher,
    RunnerContractError,
    load_runner_authority,
    validate_development_manifest_payload,
)


ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_IDENTITY_TOKENS = (
    "hash",
    "digest",
    "checksum",
    "fingerprint",
    "sha",
)


def test_manuscript_discloses_development_only_comparator_selection() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    required = (
        "thirty-four",
        "sixteen development datasets",
        "three model seeds",
        "Pareto",
        "quarter-rank",
        "BiAEImpute",
        "Methods with no eligible development configuration will remain in the "
        "scheduled denominator",
    )
    assert all(fragment in manuscript for fragment in required)


def test_manuscript_discloses_primary_comparator_sources_and_venue_format() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    references = (ROOT / "paper/references.bib").read_text()
    expected_sources = {
        "Vo2026scZiva": "10.1186/s12859-026-06422-2",
        "Huang2025afMF": "10.1002/ctm2.70283",
        "Zhang2025BiAEImpute": "10.1186/s12864-025-11988-x",
        "Um2024scCR": "10.52202/079017-0598",
        "Chi2020scSDAE": "10.3390/genes11050532",
    }
    assert all(
        citation_key in manuscript
        and any(
            f"@{entry_type}{{{citation_key}," in references
            for entry_type in ("article", "inproceedings")
        )
        and doi in references
        for citation_key, doi in expected_sources.items()
    )
    assert (
        "publisher = {Neural Information Processing Systems Foundation, Inc. "
        "(NeurIPS)}" in references
    )
    assert (
        r"\documentclass[pdflatex,sn-vancouver-num,referee,lineno]{sn-jnl}"
        in manuscript
    )
    assert r"\pagenumbering{arabic}" in manuscript

    methods_start = manuscript.index(r"\section{Methods}")
    declarations_start = manuscript.index(r"\section*{Declarations}")
    disclosure = r"\subsection{Use of generative AI or AI-assisted technologies}"
    assert disclosure in manuscript
    disclosure_start = manuscript.index(disclosure)
    assert methods_start < disclosure_start < declarations_start
    assert disclosure not in manuscript[declarations_start:]

    log_path = ROOT / "paper/manuscript.log"
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
        assert "undefined citations" not in log_text.lower()
        assert "there were undefined references" not in log_text.lower()
        assert not any(
            "Warning: Citation " in line and " undefined" in line.lower()
            for line in log_text.splitlines()
        )


def test_manuscript_discloses_exact_execution_order_and_publication_gates() -> None:
    workflow = (ROOT / "docs/development-selection-workflow.md").read_text()
    ordered_commands = (
        "python scripts/run_comparator_tuning_smoke.py",
        "python scripts/run_development_competition.py",
        "python scripts/select_comparator_configurations.py",
        "python scripts/build_development_selection_input.py",
        "python scripts/promote_development_selection_input.py",
        "python scripts/select_development_candidate.py",
        "python scripts/run_v28_revision_competition.py "
        "[--environment METHOD=EXECUTABLE ...]",
        "python scripts/run_v29_revision_competition.py "
        "[--environment METHOD=EXECUTABLE ...]",
        "python scripts/freeze_publication_round.py prepare",
        'python scripts/freeze_publication_round.py freeze "$ROUND_DIR"',
        'python scripts/run_frozen_final.py "$ROUND_DIR"',
    )
    assert all(command in workflow for command in ordered_commands)
    positions = tuple(workflow.index(command) for command in ordered_commands)
    assert positions == tuple(sorted(positions))

    checklists = "\n".join(
        (
            (ROOT / "paper/submission_checklist.md").read_text(),
            (ROOT / "docs/genome-biology-submission-checklist.md").read_text(),
        )
    )
    required_gates = (
        "2,896",
        "34 smoke",
        "five established",
        "at least three modern",
        "BiAEImpute",
        "selected payloads",
        "method bindings",
        "1,760",
        "44 trajectory",
        "execution-status table",
        "unavailable methods",
        "100 words",
        "3--10 keywords",
        "Minimum Standards",
        "OSI-compliant",
        "static archived release",
        "editable",
        "cover letter",
        "AI-use disclosure",
    )
    assert all(fragment in checklists for fragment in required_gates)


EXPECTED_ORDER = {
    "alra": ("alra-default",),
    "magic": ("magic-t03", "magic-t01", "magic-t05", "magic-t07"),
    "dca": ("dca-h64-32-64", "dca-h32-16-32", "dca-h32-32", "dca-h64-64"),
    "scvi": ("scvi-z10", "scvi-z05", "scvi-z20", "scvi-z30"),
    "saver": ("saver-default",),
    "scziva": (
        "scziva-tau-0p001",
        "scziva-tau-0p0001",
        "scziva-tau-0p01",
        "scziva-tau-0p05",
    ),
    "afmf": ("afmf-sigma-3", "afmf-sigma-1", "afmf-sigma-2", "afmf-sigma-4"),
    "biaeimpute": (
        "biaeimpute-z128",
        "biaeimpute-z32",
        "biaeimpute-z64",
        "biaeimpute-z256",
    ),
    "sccr": ("sccr-k15", "sccr-k05", "sccr-k10", "sccr-k30"),
    "scsdae": (
        "scsdae-zero-1",
        "scsdae-zero-0p25",
        "scsdae-zero-0p5",
        "scsdae-zero-0p75",
    ),
}


def _tracked_payload() -> dict[str, object]:
    return json.loads((ROOT / "study/comparator_tuning.json").read_text())


def _all_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for child in value.values() for key in _all_keys(child)
        )
    if isinstance(value, list):
        return tuple(key for child in value for key in _all_keys(child))
    return ()


def _set_nested(
    payload: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    target: object = payload
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value


def _write_authority(repository: Path, raw: bytes) -> None:
    authority_path = repository / "study/comparator_tuning.json"
    authority_path.parent.mkdir()
    authority_path.write_bytes(raw)


@pytest.fixture(scope="module")
def smoke_registry():
    return load_method_registry(ROOT / "study/methods.json")


@pytest.fixture(scope="module")
def smoke_authority(smoke_registry):
    return load_comparator_tuning_authority(
        ROOT,
        registry=smoke_registry,
        require_clean=False,
    )


@pytest.fixture(scope="module")
def smoke_bound_rows(smoke_registry, smoke_authority):
    rows = tuple(
        bind_comparator_configuration_identity(
            row,
            smoke_registry.by_id(row.method_id),
            smoke_authority,
        )
        for row in smoke_authority.configurations
    )
    assert tuple(
        (
            row.configuration.method_id,
            row.configuration.configuration_id,
        )
        for row in rows
    ) == tuple(
        (row.method_id, row.configuration_id) for row in smoke_authority.configurations
    )
    assert tuple(row.configuration for row in rows) == smoke_authority.configurations
    return rows


@pytest.fixture(scope="module")
def complete_smoke_outcomes(smoke_bound_rows):
    return tuple(
        ComparatorSmokeOutcome(
            configuration=row,
            status="completed",
            reason=None,
            runtime_seconds=1.0,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            rss_measurement="fixed_test_sampler",
            gpu_measurement="nvidia_smi_process_tree_used_memory",
        )
        for row in smoke_bound_rows
    )


def _selection_manifest_payload() -> dict[str, object]:
    rows = []
    ordinal = 0
    for mechanism in ("symsim", "sergio", "sparsim", "semisynthetic"):
        for draw in (1, 2):
            for view in ("moderate", "severe"):
                ordinal += 1
                rows.append(
                    {
                        "biological_id": f"draw-{draw:02d}",
                        "cells": 900,
                        "dataset_id": f"dataset-{ordinal:024x}",
                        "dataset_sha256": f"{ordinal:064x}",
                        "genes": 500,
                        "independent_unit_id": (
                            f"biological-{(ordinal + 1) // 2:024x}"
                        ),
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
        "manifest_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "design_sha256": "c" * 64,
        "seed_source_sha256": "d" * 64,
        "rows": rows,
    }


def _selection_prepared_dataset(binding, ordinal: int) -> PreparedDataset:
    counts = np.asarray([[ordinal, 0, 1], [0, ordinal + 1, 0]], dtype=np.int64)
    cell_ids = [f"cell-{ordinal}-1", f"cell-{ordinal}-2"]
    gene_ids = ["gene-1", "gene-2", "gene-3"]
    method_view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=gene_ids),
    )
    method_view.uns["source_dataset_sha256"] = binding.dataset_sha256
    method_view.uns["allowed_covariates"] = {"obs": [], "var": []}
    draw_index = int(binding.biological_id.removeprefix("draw-"))
    evaluator = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"draw": [draw_index, draw_index]}, index=cell_ids),
        var=pd.DataFrame(index=gene_ids),
    )
    evaluator.uns["provenance"] = {
        "seeds": {"biological": 10_000 + ordinal, "measurement": 20_000 + ordinal}
    }
    return PreparedDataset(
        binding=binding,
        audit=DatasetQCAudit(
            excluded_cell_count=0,
            excluded_cell_ids_sha256="e" * 64,
            retained_cell_count=2,
            retained_cell_ids_sha256="f" * 64,
            excluded_cell_ids=(),
            retained_cell_ids=tuple(cell_ids),
        ),
        method_input=prepare_method_input(method_view),
        evaluator_dataset=evaluator,
    )


def _complete_selection_plan(
    registry,
    authority,
    complete_smoke_outcomes,
    smoke_bound_rows,
):
    datasets = validate_development_manifest_payload(_selection_manifest_payload())
    prepared = tuple(
        _selection_prepared_dataset(binding, ordinal)
        for ordinal, binding in enumerate(datasets, start=1)
    )
    runner_authority = replace(
        load_runner_authority(),
        count_score_manifest_status="ready",
        count_score_manifest_sha256="8" * 64,
        retained_calibration_status="ready",
        retained_calibration_sha256="9" * 64,
    )
    plan = _build_structural_direct_competition_plan(
        registry,
        datasets,
        runner_authority,
        prepared,
        _validate=False,
    )
    smoke_receipt = build_comparator_smoke_receipt(
        complete_smoke_outcomes,
        authority=authority,
        registry=registry,
        bound_configurations=smoke_bound_rows,
    )
    smoke_raw = (
        json.dumps(
            smoke_receipt,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    return bind_comparator_smoke_receipt_to_plan(
        plan,
        smoke_receipt,
        smoke_raw,
        authority=authority,
        registry=registry,
    )


def _complete_checkpoint_payload(plan, registry) -> dict[str, object]:
    comparator_positions = {
        (row.method.method_id, row.configuration_id): position
        for position, row in enumerate(
            (
                item
                for item in plan.configurations
                if item.configuration_kind == "comparator_tuning"
            ),
            start=1,
        )
    }
    candidate_metric_values: dict[str, tuple[float, ...]] = {}
    records = []
    for entry in plan.entries:
        identity = direct_json_value(entry.identity)
        assert isinstance(identity, dict)
        method_id = entry.identity.method.method_id
        if entry.identity.configuration_kind == "comparator_tuning":
            position = comparator_positions[
                (method_id, entry.identity.configuration_id)
            ]
            base_values = tuple(float(position + index / 10) for index in range(6))
        elif method_id == "maskimpute":
            base_values = tuple(
                float(100 + entry.identity.ordinal / 10_000 + index / 10)
                for index in range(6)
            )
            candidate_metric_values[entry.run_id] = base_values
        else:
            base_values = tuple(float(index) for index in range(6))
        offset = (
            0.01 * entry.identity.draw_index
            + (0.001 if entry.identity.technical_view == "severe" else 0.0)
            + (
                0.0001 * (entry.identity.model_seed - 42)
                if entry.identity.model_seed is not None
                else 0.0
            )
        )
        metrics = []
        for index, metric in enumerate(
            (
                "mse",
                "mse_dropout",
                "gnrmse",
                "mse_pre_dropout_zero",
                "corr_err",
                "mse_non_dropout_nonzero",
            )
        ):
            applicable = not (
                metric == "mse_pre_dropout_zero"
                and entry.identity.mechanism != "symsim"
            )
            metrics.append(
                {
                    "identity": copy.deepcopy(identity),
                    "metric": metric,
                    "value": float(base_values[index] + offset) if applicable else None,
                    "n": 2 if applicable else 0,
                    "status": "completed" if applicable else "unavailable",
                    "reason": None if applicable else "truth_unavailable",
                }
            )
        maskimpute = method_id == "maskimpute"
        records.append(
            {
                "run": {
                    "run_id": entry.run_id,
                    "identity": identity,
                    "status": "completed",
                    "reason": None,
                    "runtime_seconds": 0.01,
                    "peak_rss_bytes": 1024,
                    "peak_gpu_bytes": 0,
                    "rss_measurement": "linux_proc_process_tree_rss",
                    "gpu_measurement": "gpu_measurement_unavailable",
                    "excluded_cell_count": 0,
                    "excluded_cell_ids": [],
                    "retained_cell_count": 2,
                    "retained_cell_ids": ["cell-1", "cell-2"],
                    "retained_gene_count": 3,
                    "observed_zero_count": 3,
                    "stdout": {
                        "stream": "stdout",
                        "original_byte_count": 0,
                        "capture_policy": "discard_content",
                        "terminal_reason": None,
                    },
                    "stderr": {
                        "stream": "stderr",
                        "original_byte_count": 0,
                        "capture_policy": "discard_content",
                        "terminal_reason": None,
                    },
                },
                "metrics": metrics,
                "p_pre_zero_evidence": (
                    {
                        "applicable": True,
                        "status": "completed",
                        "reason": None,
                        "shape": [2, 3],
                        "dtype": "<f8",
                        "encoding": "zlib",
                        "path": f"synthetic/prezero-{entry.identity.ordinal:04d}.zlib",
                        "compressed_byte_count": 1,
                    }
                    if maskimpute
                    else {
                        "applicable": False,
                        "status": "not_applicable",
                        "reason": "method_does_not_emit_p_pre_zero",
                        "shape": None,
                        "dtype": None,
                        "encoding": None,
                        "path": None,
                        "compressed_byte_count": 0,
                    }
                ),
            }
        )
    assert len(records) == 2_896
    assert len({record["run"]["run_id"] for record in records}) == 2_896
    budget = replay_direct_development_budget(
        registry,
        plan.entries,
        records,
    ).to_dict()
    return {
        "schema_version": 1,
        "identity_mode": "direct-v1",
        "authority_revision": AUTHORITY_REVISION,
        "plan_snapshot": plan.to_dict(),
        "input_descriptors": list(plan.to_dict()["inputs"]),
        "planned_run_count": 2_896,
        "status": "completed",
        "evaluation_scope": "reconstruction_only",
        "comparator_selection_status": "complete_terminal_denominator",
        "selection_complete": False,
        "selection_blockers": [
            "downstream_safety_not_evaluated",
            "null_de_fpr_not_evaluated",
            "orthogonal_endpoints_not_evaluated",
        ],
        "records": records,
        "budget": budget,
        "storage_preflight": {},
        "remaining_storage_preflight": {},
        "_candidate_metric_values": candidate_metric_values,
    }


@pytest.fixture(scope="module")
def complete_selection_fixture(
    smoke_registry,
    smoke_authority,
    smoke_bound_rows,
    complete_smoke_outcomes,
):
    plan = _complete_selection_plan(
        smoke_registry,
        smoke_authority,
        complete_smoke_outcomes,
        smoke_bound_rows,
    )
    checkpoint = _complete_checkpoint_payload(plan, smoke_registry)
    checkpoint.pop("_candidate_metric_values")
    assert len({record["run"]["run_id"] for record in checkpoint["records"]}) == 2_896
    return {
        "checkpoint": checkpoint,
        "authority": smoke_authority,
        "registry": smoke_registry,
    }


def mutate_only_maskimpute_values(fixture):
    mutated = copy.deepcopy(fixture)
    for record in mutated["checkpoint"]["records"]:
        if record["run"]["identity"]["method"]["method_id"] != "maskimpute":
            continue
        for metric in record["metrics"]:
            if metric["value"] is not None:
                metric["value"] += 10_000.0
    return mutated


def selection_fixture_with_intrinsic_unavailable(fixture, method_id: str):
    changed = copy.deepcopy(fixture)
    matched = 0
    for record in changed["checkpoint"]["records"]:
        identity = record["run"]["identity"]
        if (
            identity["method"]["method_id"] != method_id
            or identity["configuration_kind"] != "comparator_tuning"
        ):
            continue
        matched += 1
        record["run"]["status"] = "unavailable"
        record["run"]["reason"] = "adapter_not_registered"
        record["run"]["stdout"]["terminal_reason"] = "adapter_not_registered"
        record["run"]["stderr"]["terminal_reason"] = "adapter_not_registered"
        for metric in record["metrics"]:
            metric["value"] = None
            metric["n"] = 0
            metric["status"] = "unavailable"
            metric["reason"] = "adapter_not_registered"
    assert matched == 48 * len(changed["authority"].configurations_for(method_id))
    return changed


@pytest.fixture()
def complete_checkpoint_tree(tmp_path: Path, complete_selection_fixture) -> Path:
    repository = tmp_path / "repository"
    for relative in ("study/methods.json", "study/comparator_tuning.json"):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    checkpoint = (
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            complete_selection_fixture["checkpoint"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return repository


def golden_authority():
    registry = load_method_registry(ROOT / "study/methods.json")
    return load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )


def golden_comparator_records(
    *,
    method_id: str,
    configuration_values: dict[str, tuple[float, float, float, float, float, float]],
    duplicate_each_seed_value: bool,
) -> tuple[dict[str, object], ...]:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    authority_by_id = {
        row.configuration_id: row for row in authority.configurations_for(method_id)
    }
    metrics = authority.selection_metrics
    records: list[dict[str, object]] = []
    ordinal = 0
    for configuration_id, configured_values in configuration_values.items():
        row = authority_by_id[configuration_id]
        bound = bind_comparator_configuration_identity(
            row,
            registry.by_id(method_id),
            authority,
        )
        configuration_records: list[dict[str, object]] = []
        for mechanism_index, mechanism in enumerate(
            ("symsim", "sergio", "sparsim", "semisynthetic")
        ):
            for draw_index, biological_id in enumerate(
                ("draw-01", "draw-02"),
                start=1,
            ):
                for view_index, technical_view in enumerate(("moderate", "severe")):
                    dataset_id = f"dataset-{mechanism}-{biological_id}-{technical_view}"
                    for model_seed in (42, 43, 44):
                        ordinal += 1
                        identity = ComparatorRunIdentity(
                            workflow_schema="maskimpute-fair-comparator-run-v1",
                            authority_revision=authority.authority_revision,
                            ordinal=ordinal,
                            method=bound.method,
                            configuration_id=configuration_id,
                            configuration_kind="comparator_tuning",
                            configuration_payload=freeze_direct_mapping(row.payload),
                            dataset_id=dataset_id,
                            mechanism=mechanism,
                            biological_id=biological_id,
                            technical_view=technical_view,
                            mask_seed=1_000 + 10 * mechanism_index + draw_index,
                            model_seed=model_seed,
                            draw_index=draw_index,
                        )
                        identity_json = direct_json_value(identity)
                        assert isinstance(identity_json, dict)
                        applicable_metrics = (
                            metrics
                            if mechanism == "symsim"
                            else tuple(
                                metric
                                for metric in metrics
                                if metric != "mse_pre_dropout_zero"
                            )
                        )
                        metric_rows = []
                        for metric in applicable_metrics:
                            position = metrics.index(metric)
                            seed_offset = (
                                0.0
                                if duplicate_each_seed_value
                                else 0.0001 * (model_seed - 42)
                            )
                            value = float(
                                configured_values[position]
                                + 0.01 * draw_index
                                + 0.001 * view_index
                                + seed_offset
                            )
                            metric_rows.append(
                                {
                                    "identity": copy.deepcopy(identity_json),
                                    "metric": metric,
                                    "value": value,
                                    "n": 10,
                                    "status": "completed",
                                    "reason": None,
                                }
                            )
                        configuration_records.append(
                            {
                                "run": {
                                    "run_id": f"run-{ordinal:04d}-{configuration_id}",
                                    "identity": identity_json,
                                    "status": "completed",
                                    "reason": None,
                                },
                                "metrics": metric_rows,
                                "p_pre_zero_evidence": {
                                    "applicable": False,
                                    "status": "not_applicable",
                                },
                            }
                        )
        assert len(configuration_records) == 48
        assert sum(len(record["metrics"]) for record in configuration_records) == 252
        records.extend(configuration_records)
    return tuple(records)


def _magic_golden_records() -> tuple[dict[str, object], ...]:
    return golden_comparator_records(
        method_id="magic",
        configuration_values={
            "magic-t03": (1.0, 4.0, 2.0, 3.0, 2.0, 4.0),
            "magic-t01": (2.0, 2.0, 2.0, 2.0, 3.0, 2.0),
            "magic-t05": (3.0, 1.0, 3.0, 1.0, 1.0, 3.0),
            "magic-t07": (4.0, 5.0, 4.0, 5.0, 4.0, 5.0),
        },
        duplicate_each_seed_value=True,
    )


def test_seed_view_draw_collapse_and_quarter_rank_golden() -> None:
    result = select_one_comparator_method(
        "magic",
        _magic_golden_records(),
        golden_authority(),
    )
    assert result.configuration_ids == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
        "magic-t07",
    )
    assert result.eligible_configuration_ids == result.configuration_ids
    assert result.pareto_configuration_ids == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
    )
    assert result.configuration("magic-t03").unit_counts == {
        "mse": 8,
        "mse_dropout": 8,
        "gnrmse": 8,
        "mse_pre_dropout_zero": 2,
        "corr_err": 8,
        "mse_non_dropout_nonzero": 8,
    }
    assert all(
        type(value) is int
        for row in result.pareto_rows
        for value in row.metric_rank_quarters.values()
    )
    assert (
        result.selected_configuration_id
        == min(
            result.pareto_rows,
            key=lambda row: row.selection_tuple,
        ).configuration_id
    )
    for row in (*result.collapsed_rows, *result.pareto_rows):
        assert row.configuration.configuration in golden_authority().configurations
        assert (
            row.configuration.method
            == result.configuration(row.configuration_id).configuration.method
        )


def test_average_ties_encode_exact_quarter_rank_integer() -> None:
    assert metric_rank_quarters(
        {
            "a": (1.0, 1.0),
            "b": (1.0, 2.0),
            "c": (3.0, 2.0),
        }
    ) == {"a": 5, "b": 8, "c": 11}


def test_pareto_filter_requires_weak_all_and_strict_one() -> None:
    result = select_one_comparator_method(
        "magic",
        _magic_golden_records(),
        golden_authority(),
    )
    assert pareto_configuration_ids(result.collapsed_rows) == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
    )


def test_selection_tuple_uses_default_penalty_then_configuration_id() -> None:
    records = golden_comparator_records(
        method_id="magic",
        configuration_values={
            "magic-t03": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t01": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t05": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t07": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        },
        duplicate_each_seed_value=True,
    )
    result = select_one_comparator_method("magic", records, golden_authority())
    assert result.selected_configuration_id == "magic-t03"
    assert result.pareto_rows[0].selection_tuple == (
        10,
        60,
        10,
        10,
        10,
        10,
        10,
        10,
        0,
        "magic-t03",
    )


def test_nonduplicated_seed_values_collapse_before_view_pairing() -> None:
    records = golden_comparator_records(
        method_id="magic",
        configuration_values={
            "magic-t03": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        },
        duplicate_each_seed_value=False,
    )
    authority = golden_authority()
    registry = load_method_registry(ROOT / "study/methods.json")
    bound = bind_comparator_configuration_identity(
        authority.configurations_for("magic")[0],
        registry.by_id("magic"),
        authority,
    )
    row = collapse_comparator_configuration(bound, records)

    assert row.unit_values["mse"][0] == pytest.approx(1.0106)
    assert row.unit_counts == {
        "mse": 8,
        "mse_dropout": 8,
        "gnrmse": 8,
        "mse_pre_dropout_zero": 2,
        "corr_err": 8,
        "mse_non_dropout_nonzero": 8,
    }


def test_selection_rejects_blocking_status_and_returns_none_when_all_ineligible() -> (
    None
):
    blocking = [copy.deepcopy(record) for record in _magic_golden_records()]
    blocking[0]["run"]["status"] = "budget_exhausted"
    blocking[0]["run"]["reason"] = "cpu_time_budget_exhausted"
    for metric in blocking[0]["metrics"]:
        metric["value"] = None
        metric["n"] = 0
        metric["status"] = "budget_exhausted"
        metric["reason"] = "cpu_time_budget_exhausted"
    with pytest.raises(ComparatorTuningError, match="blocking run status"):
        select_one_comparator_method("magic", blocking, golden_authority())

    unavailable = [copy.deepcopy(record) for record in _magic_golden_records()]
    for offset in range(0, len(unavailable), 48):
        record = unavailable[offset]
        record["run"]["status"] = "unavailable"
        record["run"]["reason"] = "adapter_not_registered"
        for metric in record["metrics"]:
            metric["value"] = None
            metric["n"] = 0
            metric["status"] = "unavailable"
            metric["reason"] = "adapter_not_registered"
    result = select_one_comparator_method(
        "magic",
        unavailable,
        golden_authority(),
    )
    assert result.eligible_configuration_ids == ()
    assert result.pareto_configuration_ids == ()
    assert result.selected_configuration_id is None


def test_selection_rejects_otherwise_valid_cross_configuration_unit_grid_drift() -> (
    None
):
    records = [copy.deepcopy(record) for record in _magic_golden_records()]
    for record in records[48:96]:
        identity = record["run"]["identity"]
        if identity["biological_id"] == "draw-01":
            identity["biological_id"] = "draw-alpha"
            for metric in record["metrics"]:
                metric["identity"]["biological_id"] = "draw-alpha"

    with pytest.raises(ComparatorTuningError, match="unit-ID grid differs"):
        select_one_comparator_method("magic", records, golden_authority())


def test_selection_tuple_reaches_final_configuration_id_fallback() -> None:
    records = golden_comparator_records(
        method_id="magic",
        configuration_values={
            "magic-t03": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
            "magic-t01": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t05": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t07": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
        },
        duplicate_each_seed_value=True,
    )
    result = select_one_comparator_method("magic", records, golden_authority())
    first = next(
        row for row in result.pareto_rows if row.configuration_id == "magic-t01"
    )
    second = next(
        row for row in result.pareto_rows if row.configuration_id == "magic-t05"
    )

    assert first.selection_tuple[:-1] == second.selection_tuple[:-1]
    assert first.selection_tuple[-2:] == (1, "magic-t01")
    assert second.selection_tuple[-2:] == (1, "magic-t05")
    assert result.selected_configuration_id == "magic-t01"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("identity", "identity"),
        ("missing_metric", "metric"),
        ("duplicate_metric", "metric"),
        ("wrong_applicability", "applicability"),
        ("nonfinite", "nonfinite"),
        ("unit_grid", "unit grid"),
    ),
)
def test_collapse_fails_closed_on_malformed_direct_records(
    mutation: str,
    message: str,
) -> None:
    records = [copy.deepcopy(record) for record in _magic_golden_records()[:48]]
    if mutation == "identity":
        records[0]["metrics"][0]["identity"]["configuration_id"] = "magic-t01"
    elif mutation == "missing_metric":
        records[0]["metrics"].pop()
    elif mutation == "duplicate_metric":
        records[0]["metrics"].append(copy.deepcopy(records[0]["metrics"][0]))
    elif mutation == "wrong_applicability":
        records[12]["metrics"].append(
            {
                **copy.deepcopy(records[12]["metrics"][0]),
                "metric": "mse_pre_dropout_zero",
            }
        )
    elif mutation == "nonfinite":
        records[0]["metrics"][0]["value"] = float("inf")
    else:
        for record in records:
            if record["run"]["identity"]["model_seed"] == 44:
                record["run"]["identity"]["model_seed"] = 43
                for metric in record["metrics"]:
                    metric["identity"]["model_seed"] = 43
                break
    authority = golden_authority()
    registry = load_method_registry(ROOT / "study/methods.json")
    bound = bind_comparator_configuration_identity(
        authority.configurations_for("magic")[0],
        registry.by_id("magic"),
        authority,
    )
    with pytest.raises(ComparatorTuningError, match=message):
        collapse_comparator_configuration(bound, records)


def test_collapse_intrinsic_terminal_row_is_ineligible_without_aborting_method() -> (
    None
):
    records = [copy.deepcopy(record) for record in _magic_golden_records()]
    broken = records[0]
    broken["run"]["status"] = "unavailable"
    broken["run"]["reason"] = "adapter_not_registered"
    for metric in broken["metrics"]:
        metric["value"] = None
        metric["n"] = 0
        metric["status"] = "unavailable"
        metric["reason"] = "adapter_not_registered"
    result = select_one_comparator_method("magic", records, golden_authority())
    assert result.configuration_ids == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
        "magic-t07",
    )
    assert result.eligible_configuration_ids == (
        "magic-t01",
        "magic-t05",
        "magic-t07",
    )
    assert result.configuration("magic-t03").eligible is False
    assert result.configuration("magic-t03").status_counts == {
        "completed": 47,
        "unavailable": 1,
    }


def test_collapse_rejects_bound_authority_reference_drift() -> None:
    authority = golden_authority()
    registry = load_method_registry(ROOT / "study/methods.json")
    bound = bind_comparator_configuration_identity(
        authority.configurations_for("magic")[0],
        registry.by_id("magic"),
        authority,
    )
    drifted = replace(
        bound,
        authority_reference=replace(bound.authority_reference, schema_version=1),
    )
    with pytest.raises(ComparatorTuningError, match="bound comparator identity"):
        collapse_comparator_configuration(
            drifted,
            _magic_golden_records()[:48],
        )


SELECTION_RECEIPT_KEYS = {
    "schema_version",
    "artifact_type",
    "data_scope",
    "final_data_used",
    "authority_reference",
    "plan_snapshot",
    "input_descriptors",
    "checkpoint_path",
    "scheduled_tuning_records",
    "control_records",
    "model_seeds",
    "selection_metrics",
    "methods",
    "controls",
    "scheduled_same_input_ids",
    "required_control_ids",
    "established_comparator_ids",
    "modern_core_ids",
    "readiness",
}
METHOD_RECEIPT_KEYS = {
    "method",
    "selection_status",
    "configuration_order",
    "terminal_status_counts",
    "reason_histogram",
    "configurations",
    "pareto_configuration_ids",
    "selected_configuration_id",
    "selected_configuration",
    "nonexecution_identity",
}
CONFIGURATION_RECEIPT_KEYS = {
    "configuration",
    "is_upstream_default",
    "terminal_status_counts",
    "reason_histogram",
    "eligible",
    "eligibility_reason",
    "unit_ids",
    "unit_values",
    "unit_counts",
    "metric_medians",
    "pareto_member",
    "metric_rank_quarters",
    "selection_tuple",
}
NONEXECUTION_IDENTITY_KEYS = {
    "schema_version",
    "authority_reference",
    "method",
    "selection_receipt_namespace",
    "configuration_terminal_denominator",
}


def test_readiness_requires_controls_established_and_three_modern(
    complete_selection_fixture,
) -> None:
    receipt = comparator_tuning_module.build_comparator_selection_receipt(
        **complete_selection_fixture
    )
    assert receipt["readiness"]["status"] == "ready"
    assert receipt["readiness"]["modern_selectable_count"] == 4

    three_modern = selection_fixture_with_intrinsic_unavailable(
        complete_selection_fixture,
        "biaeimpute",
    )
    receipt = comparator_tuning_module.build_comparator_selection_receipt(
        **three_modern
    )
    assert receipt["readiness"]["status"] == "ready"
    assert receipt["methods"]["biaeimpute"]["selected_configuration_id"] is None
    assert (
        receipt["methods"]["biaeimpute"]["nonexecution_identity"]["method"]["method_id"]
        == "biaeimpute"
    )

    with pytest.raises(ComparatorTuningError, match="publication readiness"):
        comparator_tuning_module.build_comparator_selection_receipt(
            **selection_fixture_with_intrinsic_unavailable(three_modern, "sccr")
        )

    incomplete_control = copy.deepcopy(complete_selection_fixture)
    control = next(
        record
        for record in incomplete_control["checkpoint"]["records"]
        if record["run"]["identity"]["method"]["method_id"] == "observed"
    )
    control["run"]["status"] = "unavailable"
    control["run"]["reason"] = "adapter_not_registered"
    control["run"]["stdout"]["terminal_reason"] = "adapter_not_registered"
    control["run"]["stderr"]["terminal_reason"] = "adapter_not_registered"
    for metric in control["metrics"]:
        metric["value"] = None
        metric["n"] = 0
        metric["status"] = "unavailable"
        metric["reason"] = "adapter_not_registered"
    with pytest.raises(ComparatorTuningError, match="required_control_incomplete"):
        comparator_tuning_module.build_comparator_selection_receipt(
            **incomplete_control
        )


def test_candidate_values_cannot_change_complete_comparator_receipt(
    complete_selection_fixture,
) -> None:
    first = comparator_tuning_module.build_comparator_selection_receipt(
        **complete_selection_fixture
    )
    mutated = mutate_only_maskimpute_values(complete_selection_fixture)
    second = comparator_tuning_module.build_comparator_selection_receipt(**mutated)

    assert first == second
    assert json.dumps(
        first,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") == json.dumps(
        second,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert all(
        record["run"]["identity"]["method"]["method_id"] != "maskimpute"
        for record in first["scheduled_tuning_records"]
    )
    assert all(
        record["run"]["identity"]["method"]["method_id"]
        in {"observed", "capacity-matched-ae"}
        for record in first["control_records"]
    )


def test_selection_builder_rejects_noncanonical_completed_run_metric_reason(
    complete_selection_fixture,
) -> None:
    malformed = copy.deepcopy(complete_selection_fixture)
    record = next(
        record
        for record in malformed["checkpoint"]["records"]
        if record["run"]["identity"]["method"]["method_id"] == "magic"
        and record["run"]["identity"]["configuration_kind"] == "comparator_tuning"
    )
    metric = record["metrics"][0]
    metric["value"] = None
    metric["n"] = 0
    metric["status"] = "unavailable"
    metric["reason"] = "invented_reason"

    with pytest.raises(ComparatorTuningError, match="accepted direct record"):
        comparator_tuning_module.build_comparator_selection_receipt(**malformed)


@pytest.mark.parametrize(
    "mutation",
    (
        "invented_intrinsic_terminal_reason",
        "invented_measurement_code",
        "overlapping_cell_audit_ids",
        "noncompleted_maskimpute_retains_completed_prezero_fields",
        "noncanonical_completed_prezero_path",
    ),
)
def test_selection_builder_rejects_record_outside_accepted_direct_contract(
    complete_selection_fixture,
    mutation: str,
) -> None:
    malformed = copy.deepcopy(complete_selection_fixture)
    records = malformed["checkpoint"]["records"]
    comparator_record = next(
        record
        for record in records
        if record["run"]["identity"]["method"]["method_id"] == "magic"
        and record["run"]["identity"]["configuration_kind"] == "comparator_tuning"
    )
    maskimpute_record = next(
        record
        for record in records
        if record["run"]["identity"]["method"]["method_id"] == "maskimpute"
    )

    if mutation == "invented_intrinsic_terminal_reason":
        run = comparator_record["run"]
        run["status"] = "unavailable"
        run["reason"] = "invented_terminal_reason"
        run["stdout"]["terminal_reason"] = run["reason"]
        run["stderr"]["terminal_reason"] = run["reason"]
        for metric in comparator_record["metrics"]:
            metric["value"] = None
            metric["n"] = 0
            metric["status"] = run["status"]
            metric["reason"] = run["reason"]
    elif mutation == "invented_measurement_code":
        comparator_record["run"]["rss_measurement"] = "invented_measurement_code"
    elif mutation == "overlapping_cell_audit_ids":
        run = comparator_record["run"]
        shared = run["retained_cell_ids"][0]
        run["excluded_cell_count"] = 1
        run["excluded_cell_ids"] = [shared]
    elif mutation == "noncompleted_maskimpute_retains_completed_prezero_fields":
        run = maskimpute_record["run"]
        run["status"] = "unavailable"
        run["reason"] = "adapter_not_registered"
        run["stdout"]["terminal_reason"] = run["reason"]
        run["stderr"]["terminal_reason"] = run["reason"]
        for metric in maskimpute_record["metrics"]:
            metric["value"] = None
            metric["n"] = 0
            metric["status"] = run["status"]
            metric["reason"] = run["reason"]
        evidence = maskimpute_record["p_pre_zero_evidence"]
        evidence["status"] = run["status"]
        evidence["reason"] = run["reason"]
    else:
        maskimpute_record["p_pre_zero_evidence"]["path"] = "."

    with pytest.raises(ComparatorTuningError, match="accepted direct record"):
        comparator_tuning_module.build_comparator_selection_receipt(**malformed)


def test_selection_builder_rejects_integer_coercion_of_input_float(
    complete_selection_fixture,
) -> None:
    malformed = copy.deepcopy(complete_selection_fixture)
    plan_input = malformed["checkpoint"]["plan_snapshot"]["inputs"][0]
    descriptor_input = malformed["checkpoint"]["input_descriptors"][0]
    assert type(plan_input["total_count"]) is float
    assert plan_input["total_count"].is_integer()
    coerced = int(plan_input["total_count"])
    plan_input["total_count"] = coerced
    descriptor_input["total_count"] = coerced

    with pytest.raises(ComparatorTuningError, match="input descriptor"):
        comparator_tuning_module.build_comparator_selection_receipt(**malformed)


def test_comparator_selection_receipt_has_exact_closed_direct_schemas(
    complete_selection_fixture,
) -> None:
    receipt = comparator_tuning_module.build_comparator_selection_receipt(
        **complete_selection_fixture
    )
    assert set(receipt) == SELECTION_RECEIPT_KEYS
    assert receipt["authority_reference"] == {
        "path": "study/comparator_tuning.json",
        "schema_version": 2,
        "authority_revision": AUTHORITY_REVISION,
    }
    assert (
        receipt["plan_snapshot"]
        == complete_selection_fixture["checkpoint"]["plan_snapshot"]
    )
    assert (
        receipt["input_descriptors"]
        == complete_selection_fixture["checkpoint"]["input_descriptors"]
    )
    assert len(receipt["scheduled_tuning_records"]) == 1_632
    assert len(receipt["control_records"]) == 64
    for method_id, method in receipt["methods"].items():
        assert set(method) == METHOD_RECEIPT_KEYS
        assert method["method"]["method_id"] == method_id
        for configuration in method["configurations"].values():
            assert set(configuration) == CONFIGURATION_RECEIPT_KEYS
            assert set(configuration["configuration"]) == {
                "configuration",
                "authority_reference",
                "method",
            }
            assert set(configuration["configuration"]["configuration"]) == {
                "method_id",
                "configuration_id",
                "payload_json",
                "is_upstream_default",
            }
    assert receipt["methods"]["magic"]["selected_configuration"]["configuration"][
        "payload_json"
    ]
    assert receipt["methods"]["magic"]["nonexecution_identity"] is None

    unavailable = selection_fixture_with_intrinsic_unavailable(
        complete_selection_fixture,
        "biaeimpute",
    )
    unavailable_receipt = comparator_tuning_module.build_comparator_selection_receipt(
        **unavailable
    )
    nonexecution = unavailable_receipt["methods"]["biaeimpute"]["nonexecution_identity"]
    assert set(nonexecution) == NONEXECUTION_IDENTITY_KEYS
    assert len(nonexecution["configuration_terminal_denominator"]) == 4


def test_comparator_receipt_publication_is_create_only_and_idempotent(
    complete_checkpoint_tree: Path,
) -> None:
    repository_copy = complete_checkpoint_tree
    first = comparator_tuning_module.publish_comparator_selection(repository_copy)
    second = comparator_tuning_module.publish_comparator_selection(repository_copy)
    assert first == second
    path = repository_copy / COMPARATOR_SELECTION_RELATIVE_PATH
    first_bytes = path.read_bytes()
    payload = json.loads(first_bytes)
    payload["readiness"]["status"] = "blocked"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ComparatorTuningError,
        match="existing comparator selection differs",
    ):
        comparator_tuning_module.publish_comparator_selection(repository_copy)
    assert path.read_bytes() != first_bytes


def test_selection_publication_rejects_symlink_and_nonunique_destination(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    destination = repository / COMPARATOR_SELECTION_RELATIVE_PATH
    destination.parent.mkdir(parents=True)
    other = repository / "other.json"
    other.write_bytes(b"{}\n")
    destination.symlink_to(other)
    with pytest.raises(ComparatorTuningError, match="not owned"):
        comparator_tuning_module._immutable_publish(
            destination,
            b"{}\n",
            repository,
        )

    destination.unlink()
    destination.hardlink_to(other)
    with pytest.raises(ComparatorTuningError, match="owned regular"):
        comparator_tuning_module._immutable_publish(
            destination,
            b"{}\n",
            repository,
        )


def test_identical_concurrent_publication_accepts_transient_two_link_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    destination = repository / COMPARATOR_SELECTION_RELATIVE_PATH
    destination.parent.mkdir(parents=True)
    data = b'{"receipt":"same"}\n'
    original_link = comparator_tuning_module.os.link
    original_secure_read = comparator_tuning_module._secure_read_regular
    existing_read_finished = threading.Event()

    def coordinated_link(*args, **kwargs):
        original_link(*args, **kwargs)
        assert existing_read_finished.wait(timeout=5)

    def observed_secure_read(*args, **kwargs):
        try:
            return original_secure_read(*args, **kwargs)
        finally:
            if len(args) >= 3 and args[2] == "existing comparator selection":
                existing_read_finished.set()

    monkeypatch.setattr(comparator_tuning_module.os, "link", coordinated_link)
    monkeypatch.setattr(
        comparator_tuning_module,
        "_secure_read_regular",
        observed_secure_read,
    )
    start = threading.Barrier(2)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            start.wait(timeout=5)
            comparator_tuning_module._immutable_publish(
                destination,
                data,
                repository,
            )
        except BaseException as error:  # noqa: BLE001 - asserted thread outcome
            errors.append(error)

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert destination.read_bytes() == data
    assert destination.stat().st_nlink == 1


def test_comparator_selection_loader_recomputes_and_rejects_tamper(
    complete_checkpoint_tree: Path,
) -> None:
    receipt = comparator_tuning_module.publish_comparator_selection(
        complete_checkpoint_tree
    )
    loaded = comparator_tuning_module.load_comparator_selection_receipt(
        complete_checkpoint_tree,
        expected_checkpoint=(
            complete_checkpoint_tree
            / "artifacts/study/development/competition-reconstruction/checkpoint.json"
        ),
    )
    assert loaded == receipt
    target = complete_checkpoint_tree / COMPARATOR_SELECTION_RELATIVE_PATH
    baseline = target.read_bytes()
    mutations = (
        (("authority_reference", "authority_revision"), "fair-comparator-direct-v2"),
        (("methods", "magic", "selected_configuration_id"), "magic-t07"),
        (
            (
                "methods",
                "magic",
                "selected_configuration",
                "configuration",
                "payload_json",
            ),
            "{}",
        ),
        (
            ("scheduled_tuning_records", 0, "run", "identity", "configuration_id"),
            "alra-forged",
        ),
        (("scheduled_tuning_records", 0, "metrics", 0, "value"), 999.0),
        (("scheduled_tuning_records", 0, "run", "status"), "unavailable"),
        (
            (
                "methods",
                "magic",
                "configurations",
                "magic-t03",
                "pareto_member",
            ),
            False,
        ),
        (
            (
                "methods",
                "magic",
                "configurations",
                "magic-t03",
                "selection_tuple",
                0,
            ),
            999,
        ),
    )
    checkpoint_path = (
        complete_checkpoint_tree
        / "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    for path, replacement in mutations:
        changed = json.loads(baseline)
        nested = changed
        for item in path[:-1]:
            nested = nested[item]
        nested[path[-1]] = replacement
        target.write_text(
            json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ComparatorTuningError):
            comparator_tuning_module.load_comparator_selection_receipt(
                complete_checkpoint_tree,
                expected_checkpoint=checkpoint_path,
            )
        target.write_bytes(baseline)

    changed = json.loads(baseline)
    changed["extra"] = True
    target.write_text(
        json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ComparatorTuningError, match="missing or extra"):
        comparator_tuning_module.load_comparator_selection_receipt(
            complete_checkpoint_tree
        )
    target.write_bytes(baseline + b" ")
    with pytest.raises(ComparatorTuningError, match="canonical"):
        comparator_tuning_module.load_comparator_selection_receipt(
            complete_checkpoint_tree
        )


def test_comparator_selection_loader_rejects_integer_coercion_of_input_float(
    complete_checkpoint_tree: Path,
) -> None:
    comparator_tuning_module.publish_comparator_selection(complete_checkpoint_tree)
    target = complete_checkpoint_tree / COMPARATOR_SELECTION_RELATIVE_PATH
    receipt = json.loads(target.read_bytes())
    plan_input = receipt["plan_snapshot"]["inputs"][0]
    descriptor_input = receipt["input_descriptors"][0]
    assert type(plan_input["total_count"]) is float
    assert plan_input["total_count"].is_integer()
    coerced = int(plan_input["total_count"])
    plan_input["total_count"] = coerced
    descriptor_input["total_count"] = coerced
    target.write_text(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ComparatorTuningError, match="input descriptor"):
        comparator_tuning_module.load_comparator_selection_receipt(
            complete_checkpoint_tree
        )


def test_selection_publication_api_and_cli_have_no_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(
        inspect.signature(
            comparator_tuning_module.publish_comparator_selection
        ).parameters
    ) == ("repository",)
    assert tuple(
        inspect.signature(
            comparator_tuning_module.load_comparator_selection_receipt
        ).parameters
    ) == ("repository", "expected_checkpoint")
    script = ROOT / "scripts/select_comparator_configurations.py"
    spec = importlib.util.spec_from_file_location("task11_selection_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "publish_comparator_selection",
        lambda repository: {
            "readiness": {"status": "ready"},
            "methods": {
                "magic": {"selection_status": "selected"},
                "biaeimpute": {
                    "selection_status": ("intrinsic_terminal_no_eligible_configuration")
                },
            },
        },
    )
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert module.main() == 0
    monkeypatch.setattr(sys, "argv", [str(script), "--method", "magic"])
    with pytest.raises(SystemExit):
        module.main()


def test_smoke_input_is_exact_truth_free_900_by_500() -> None:
    method_input = build_comparator_smoke_input()
    assert method_input.shape == (900, 500)
    assert method_input.counts[0, 0] == 0
    assert method_input.counts[17, 31] == (17 * 17 + 31 * 31 + 7 * (17 ^ 31)) % 6
    assert len(method_input.obs_covariates) == 1
    batch = method_input.obs_covariates[0]
    assert batch.name == "batch"
    assert batch.categories == ("batch-0", "batch-1")
    assert batch.values == tuple(f"batch-{index % 2}" for index in range(900))
    assert not hasattr(method_input, "truth")


def _with_negative_first_formula_zero(method_input):
    canonical_bytes = method_input._count_bytes
    positive_zero = struct.pack("<d", 0.0)
    negative_zero = struct.pack("<d", -0.0)
    assert positive_zero == b"\x00\x00\x00\x00\x00\x00\x00\x00"
    assert negative_zero == b"\x00\x00\x00\x00\x00\x00\x00\x80"
    assert canonical_bytes[:8] == positive_zero
    changed_bytes = negative_zero + canonical_bytes[8:]
    assert changed_bytes != canonical_bytes
    return replace(method_input, _count_bytes=changed_bytes)


def test_smoke_input_rejects_byte_distinct_negative_formula_zero() -> None:
    method_input = build_comparator_smoke_input()
    changed = _with_negative_first_formula_zero(method_input)

    assert changed.counts[0, 0] == method_input.counts[0, 0] == 0.0
    assert changed.counts.tobytes(order="C")[:8] == struct.pack("<d", -0.0)
    assert method_input.counts.tobytes(order="C")[:8] == struct.pack("<d", 0.0)
    with pytest.raises(ComparatorTuningError, match="fixed input"):
        comparator_tuning_module.comparator_smoke_input_descriptor(changed)


def test_smoke_receipt_requires_all_34_completed_and_projected_budget(
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    receipt = build_comparator_smoke_receipt(
        complete_smoke_outcomes,
        authority=smoke_authority,
        registry=smoke_registry,
        bound_configurations=smoke_bound_rows,
    )
    assert receipt["planned_configuration_count"] == 34
    assert receipt["completed_configuration_count"] == 34
    assert receipt["projection_multiplier"] == 48
    assert receipt["status"] == "ready"
    broken = list(complete_smoke_outcomes)
    broken[0] = replace(
        broken[0],
        status="unavailable",
        reason="smoke_unavailable",
    )
    with pytest.raises(
        ComparatorTuningError,
        match="all configurations must complete",
    ):
        build_comparator_smoke_receipt(
            broken,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
        )


@pytest.mark.parametrize("field", ("peak_rss_bytes", "peak_gpu_bytes"))
def test_smoke_receipt_rejects_each_resource_cap_with_complete_denominator(
    field: str,
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    outcomes = list(complete_smoke_outcomes)
    method_id = outcomes[0].configuration.configuration.method_id
    resources = smoke_registry.by_id(method_id).resources
    limit_gib = (
        resources.max_rss_gib if field == "peak_rss_bytes" else resources.max_gpu_gib
    )
    outcomes[0] = replace(outcomes[0], **{field: limit_gib * 1024**3 + 1})
    assert len(outcomes) == 34

    with pytest.raises(ComparatorTuningError, match="resource cap"):
        build_comparator_smoke_receipt(
            outcomes,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
        )


@pytest.mark.parametrize(
    "gpu_measurement",
    (
        "gpu_measurement_unavailable",
        "nvidia_smi_measurement_unavailable",
        "not_applicable_cpu_only_method",
        "executor_reported_unverified",
    ),
)
def test_smoke_receipt_rejects_unverified_cpu_zero_gpu_measurement(
    gpu_measurement: str,
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    outcomes = list(complete_smoke_outcomes)
    position = next(
        index
        for index, outcome in enumerate(outcomes)
        if not smoke_registry.by_id(
            outcome.configuration.configuration.method_id
        ).resources.gpu_required
    )
    outcomes[position] = replace(
        outcomes[position],
        peak_gpu_bytes=0,
        gpu_measurement=gpu_measurement,
    )

    with pytest.raises(ComparatorTuningError, match="GPU measurement"):
        build_comparator_smoke_receipt(
            outcomes,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
        )


@pytest.mark.parametrize("gpu_required", (False, True), ids=("cpu", "gpu"))
def test_smoke_receipt_rejects_projected_method_budget_with_complete_denominator(
    gpu_required: bool,
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    outcomes = list(complete_smoke_outcomes)
    position = next(
        index
        for index, outcome in enumerate(outcomes)
        if smoke_registry.by_id(
            outcome.configuration.configuration.method_id
        ).resources.gpu_required
        is gpu_required
    )
    budget_seconds = 8 * 3600 if gpu_required else 24 * 3600
    outcomes[position] = replace(
        outcomes[position],
        runtime_seconds=budget_seconds / 48.0 + 1.0,
    )
    assert len(outcomes) == 34

    with pytest.raises(ComparatorTuningError, match="method budget"):
        build_comparator_smoke_receipt(
            outcomes,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
        )


def _write_smoke_repository(repository: Path) -> None:
    (repository / "study").mkdir(parents=True)
    shutil.copy2(ROOT / "study/methods.json", repository / "study/methods.json")
    shutil.copy2(
        ROOT / "study/comparator_tuning.json",
        repository / "study/comparator_tuning.json",
    )


def test_smoke_run_uses_all_bound_rows_and_create_only_complete_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    _write_smoke_repository(repository)
    calls = []
    real_loader = load_comparator_tuning_authority

    def load_untracked_fixture(selected_repository, *, registry, require_clean=True):
        return real_loader(
            selected_repository,
            registry=registry,
            require_clean=False,
        )

    monkeypatch.setattr(
        comparator_tuning_module,
        "load_comparator_tuning_authority",
        load_untracked_fixture,
    )

    def fake_executor(request, _dispatcher, _authority):
        calls.append(request)
        assert request.model_seed == 42
        assert request.fixture.shape == (900, 500)
        assert request.method_input.shape == (900, 500)
        return ComparatorSmokeOutcome(
            configuration=request.configuration,
            status="completed",
            reason=None,
            runtime_seconds=1.0,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            rss_measurement="fixed_test_sampler",
            gpu_measurement="nvidia_smi_process_tree_used_memory",
        )

    receipt = run_comparator_tuning_smoke(
        repository,
        _executor=fake_executor,
    )
    registry = load_method_registry(repository / "study/methods.json")
    authority = load_comparator_tuning_authority(
        repository,
        registry=registry,
        require_clean=False,
    )
    assert tuple(
        (
            request.configuration.configuration.method_id,
            request.configuration.configuration.configuration_id,
        )
        for request in calls
    ) == tuple(
        (row.method_id, row.configuration_id) for row in authority.configurations
    )
    assert len(calls) == 34
    assert load_comparator_smoke_receipt(repository, authority, registry) == receipt
    path = repository / authority.smoke_receipt_path
    first_bytes = path.read_bytes()
    assert (
        run_comparator_tuning_smoke(
            repository,
            _executor=fake_executor,
        )
        == receipt
    )
    assert path.read_bytes() == first_bytes

    def changed_executor(request, _dispatcher, _authority):
        outcome = fake_executor(request, _dispatcher, _authority)
        return replace(outcome, runtime_seconds=2.0)

    with pytest.raises(ComparatorTuningError, match="conflicts"):
        run_comparator_tuning_smoke(
            repository,
            _executor=changed_executor,
        )
    assert path.read_bytes() == first_bytes

    def failing_executor(_request, _dispatcher, _authority):
        raise RuntimeError("private executor detail")

    with pytest.raises(ComparatorTuningError, match="adapter boundary failed"):
        run_comparator_tuning_smoke(
            repository,
            _executor=failing_executor,
        )
    assert path.read_bytes() == first_bytes


def test_spawned_smoke_request_retains_complete_fixed_fixture_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    smoke_authority,
    smoke_bound_rows,
    smoke_registry,
) -> None:
    method_input = build_comparator_smoke_input()
    descriptor = comparator_tuning_module.comparator_smoke_input_descriptor(
        method_input
    )
    request = comparator_tuning_module._ComparatorSmokeRequest(
        configuration=smoke_bound_rows[0],
        fixture=descriptor,
        method_input=method_input,
        method_spec=smoke_registry.by_id(smoke_bound_rows[0].configuration.method_id),
        model_seed=42,
        ordinal=1,
    )
    environments = ExecutionEnvironmentRegistry(
        repository_root=ROOT.resolve(),
        executable_paths=(),
        lock_only_environment_ids=(),
        registry_sha256="a" * 64,
        runtime_lock_sha256=None,
        runtime_lock_path=None,
        benchmark_python=Path(sys.executable),
        r_library_paths=(),
        execution_environment_sha256="b" * 64,
        python_spawn_search_path=(str(ROOT.resolve()),),
        runtime_identity_snapshots=(),
        runtime_closure_paths_sha256s=(),
        runtime_snapshot=None,
    )
    dispatcher = RepositoryAdapterDispatcher(
        ROOT,
        environments,
        comparator_tuning_authority=smoke_authority,
    )
    captured = {}

    def fake_spawn(direct_request, _executor, **options):
        captured["request"] = direct_request
        captured["options"] = options
        return AdapterOutcome.failed(
            "adapter_exception",
            runtime_seconds=1.0,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            rss_measurement="linux_proc_process_tree_rss",
            gpu_measurement="not_applicable_cpu_only_method",
        )

    monkeypatch.setattr(
        "maskimpute_benchmark.runner.execute_direct_adapter_in_spawned_process",
        fake_spawn,
    )
    comparator_tuning_module._execute_smoke_request_in_spawned_dispatcher(
        request,
        dispatcher,
        smoke_authority,
    )

    inner = captured["request"]
    assert captured["options"]["require_gpu_measurement"] is True
    assert inner.smoke_fixture == descriptor
    assert inner.timeout_seconds == float(request.method_spec.resources.timeout_seconds)
    assert inner.max_rss_bytes == request.method_spec.resources.max_rss_gib * 1024**3
    assert inner.max_gpu_bytes == request.method_spec.resources.max_gpu_gib * 1024**3
    assert inner.to_dict()["smoke_fixture"] == json.loads(
        json.dumps(asdict(descriptor))
    )
    changed = _with_negative_first_formula_zero(method_input)
    with pytest.raises(RunnerContractError, match="fixed input"):
        replace(inner, method_input=changed)


def test_spawned_smoke_request_normalizes_fractional_gib_caps_to_bytes(
    monkeypatch: pytest.MonkeyPatch,
    smoke_authority,
    smoke_bound_rows,
    smoke_registry,
) -> None:
    configuration = next(
        row
        for row in smoke_bound_rows
        if smoke_registry.by_id(row.configuration.method_id).resources.gpu_required
    )
    method_input = build_comparator_smoke_input()
    descriptor = comparator_tuning_module.comparator_smoke_input_descriptor(
        method_input
    )
    canonical = smoke_registry.by_id(configuration.configuration.method_id)
    method_spec = replace(
        canonical,
        resources=replace(
            canonical.resources,
            max_rss_gib=1.5,
            max_gpu_gib=0.5,
        ),
    )
    request = comparator_tuning_module._ComparatorSmokeRequest(
        configuration=configuration,
        fixture=descriptor,
        method_input=method_input,
        method_spec=method_spec,
        model_seed=42,
        ordinal=1,
    )
    environments = ExecutionEnvironmentRegistry(
        repository_root=ROOT.resolve(),
        executable_paths=(),
        lock_only_environment_ids=(),
        registry_sha256="a" * 64,
        runtime_lock_sha256=None,
        runtime_lock_path=None,
        benchmark_python=Path(sys.executable),
        r_library_paths=(),
        execution_environment_sha256="b" * 64,
        python_spawn_search_path=(str(ROOT.resolve()),),
        runtime_identity_snapshots=(),
        runtime_closure_paths_sha256s=(),
        runtime_snapshot=None,
    )
    dispatcher = RepositoryAdapterDispatcher(
        ROOT,
        environments,
        comparator_tuning_authority=smoke_authority,
    )
    captured: dict[str, object] = {}

    def fake_spawn(direct_request, _executor, **_options):
        captured["request"] = direct_request
        return AdapterOutcome.failed(
            "adapter_exception",
            runtime_seconds=1.0,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            rss_measurement="linux_proc_process_tree_rss",
            gpu_measurement="nvidia_smi_process_tree_used_memory",
        )

    monkeypatch.setattr(
        "maskimpute_benchmark.runner.execute_direct_adapter_in_spawned_process",
        fake_spawn,
    )

    comparator_tuning_module._execute_smoke_request_in_spawned_dispatcher(
        request,
        dispatcher,
        smoke_authority,
    )

    inner = captured["request"]
    assert inner.max_rss_bytes == int(1.5 * 1024**3)
    assert inner.max_gpu_bytes == int(0.5 * 1024**3)
    assert type(inner.max_rss_bytes) is int
    assert type(inner.max_gpu_bytes) is int


def test_smoke_loader_recomputes_complete_receipt(
    tmp_path: Path,
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    repository = tmp_path / "repository"
    _write_smoke_repository(repository)
    receipt = build_comparator_smoke_receipt(
        complete_smoke_outcomes,
        authority=smoke_authority,
        registry=smoke_registry,
        bound_configurations=smoke_bound_rows,
    )
    path = repository / smoke_authority.smoke_receipt_path
    path.parent.mkdir(parents=True)
    path.write_bytes(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    local_registry = load_method_registry(repository / "study/methods.json")
    local_authority = load_comparator_tuning_authority(
        repository,
        registry=local_registry,
        require_clean=False,
    )
    loaded = load_comparator_smoke_receipt(
        repository,
        local_authority,
        local_registry,
    )
    assert loaded == receipt
    changed = copy.deepcopy(receipt)
    changed["fixture"]["maximum"] = 3.0
    path.write_bytes(
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ComparatorTuningError, match="differs"):
        load_comparator_smoke_receipt(
            repository,
            local_authority,
            local_registry,
        )
    changed = copy.deepcopy(receipt)
    changed["outcomes"][0]["runtime_seconds"] = 1
    path.write_bytes(
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ComparatorTuningError, match="measurement is invalid"):
        load_comparator_smoke_receipt(
            repository,
            local_authority,
            local_registry,
        )
    changed = copy.deepcopy(receipt)
    changed["outcomes"][0]["runtime_seconds"] = -0.0
    path.write_bytes(
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ComparatorTuningError, match="measurement is invalid"):
        load_comparator_smoke_receipt(
            repository,
            local_authority,
            local_registry,
        )


def test_smoke_cli_has_no_override_and_never_runs_real_workload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = ROOT / "scripts/run_comparator_tuning_smoke.py"
    spec = importlib.util.spec_from_file_location("task9_smoke_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "run_comparator_tuning_smoke",
        lambda repository: {
            "status": "ready",
            "planned_configuration_count": 34,
            "completed_configuration_count": 34,
        },
    )
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "path": "artifacts/study/development/evaluation/comparator_smoke.json",
        "status": "ready",
        "planned_configuration_count": 34,
        "completed_configuration_count": 34,
    }
    monkeypatch.setattr(sys, "argv", [str(script), "--repository", str(ROOT)])
    with pytest.raises(SystemExit, match="2"):
        module.main()


def test_tracked_comparator_authority_uses_only_direct_identity() -> None:
    payload = json.loads((ROOT / "study/comparator_tuning.json").read_text())
    assert payload["authority_revision"] == "fair-comparator-direct-v1"
    assert not any(
        token in key.lower()
        for key in _all_keys(payload)
        for token in FORBIDDEN_IDENTITY_TOKENS
    )
    parameters = inspect.signature(decode_comparator_configuration).parameters
    assert tuple(parameters) == ("method_id", "payload")


def test_bound_comparator_contains_full_method_projection() -> None:
    from maskimpute_benchmark.comparator_tuning import (
        bind_comparator_configuration_identity,
        comparator_method_binding,
    )

    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )
    row = authority.configurations_for("magic")[0]
    bound = bind_comparator_configuration_identity(
        row, registry.by_id("magic"), authority
    )
    assert bound.configuration == row
    assert bound.authority_reference.authority_revision == authority.authority_revision
    assert bound.method == comparator_method_binding(registry.by_id("magic"))


def test_comparator_binding_rejects_detached_or_noncanonical_authority_rows() -> None:
    from maskimpute_benchmark.comparator_tuning import (
        bind_comparator_configuration_identity,
    )

    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )
    row = authority.configurations_for("magic")[0]

    with pytest.raises(ComparatorTuningError, match="registry method"):
        bind_comparator_configuration_identity(row, registry.by_id("dca"), authority)

    with pytest.raises(ComparatorTuningError, match="one exact authority"):
        bind_comparator_configuration_identity(
            replace(row, configuration_id="magic-detached"),
            registry.by_id("magic"),
            authority,
        )

    noncanonical = replace(row, payload_json=json.dumps(dict(row.payload), indent=2))
    noncanonical_authority = replace(
        authority,
        configurations=(
            noncanonical,
            *tuple(item for item in authority.configurations if item != row),
        ),
    )
    with pytest.raises(ComparatorTuningError, match="not canonical JSON"):
        bind_comparator_configuration_identity(
            noncanonical,
            registry.by_id("magic"),
            noncanonical_authority,
        )

    duplicate_authority = replace(
        authority,
        configurations=(*authority.configurations, row),
    )
    with pytest.raises(ComparatorTuningError, match="one exact authority"):
        bind_comparator_configuration_identity(
            row, registry.by_id("magic"), duplicate_authority
        )


def test_all_normative_configurations_round_trip_exactly() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )

    assert authority.schema_version == 2
    assert authority.authority_revision == AUTHORITY_REVISION
    assert len(authority.configurations) == 34
    for row in authority.configurations:
        decoded = row.decode()
        assert encode_comparator_configuration(decoded) == dict(row.payload)
        dataclass_payload = asdict(decoded)
        if row.method_id == "dca":
            dataclass_payload["hidden_size"] = list(dataclass_payload["hidden_size"])
        assert dataclass_payload == dict(row.payload)


def test_decode_comparator_configuration_is_closed_and_exact() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )
    for row in authority.configurations:
        missing = dict(row.payload)
        missing.pop(next(iter(missing)))
        with pytest.raises(ComparatorTuningError, match="complete field set"):
            decode_comparator_configuration(row.method_id, missing)

        extra = {**dict(row.payload), "unexpected": 1}
        with pytest.raises(ComparatorTuningError, match="complete field set"):
            decode_comparator_configuration(row.method_id, extra)

    magic = authority.configurations_for("magic")[0]
    bool_as_int = {**dict(magic.payload), "knn": True}
    with pytest.raises(ComparatorTuningError, match="primitive type"):
        decode_comparator_configuration("magic", bool_as_int)

    dca = authority.configurations_for("dca")[0]
    tuple_payload = {**dict(dca.payload), "hidden_size": (64, 32, 64)}
    with pytest.raises(ComparatorTuningError, match="JSON array"):
        decode_comparator_configuration("dca", tuple_payload)

    afmf = authority.configurations_for("afmf")[0]
    negative_zero = {**dict(afmf.payload), "lambda_p": -0.0}
    with pytest.raises(ComparatorTuningError, match="invalid float value"):
        decode_comparator_configuration("afmf", negative_zero)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        pytest.param(("schema_version",), True, id="schema-version-type"),
        pytest.param(("authority_revision",), "other-revision", id="revision"),
        pytest.param(("contract_id",), "other-contract", id="contract-id"),
        pytest.param(("scope", "data_scope"), "final", id="scope-data"),
        pytest.param(("scope", "final_data_used"), True, id="scope-final"),
        pytest.param(
            ("method_order",), list(reversed(EXPECTED_ORDER)), id="method-order"
        ),
        pytest.param(
            ("scheduled_same_input_ids",),
            ["observed", "capacity-matched-ae"],
            id="scheduled-set",
        ),
        pytest.param(("required_control_ids",), ["observed"], id="control-set"),
        pytest.param(("established_comparator_ids",), ["alra"], id="established-set"),
        pytest.param(("modern_core_ids",), ["scziva"], id="modern-set"),
        pytest.param(("model_seeds",), [42, 43, 45], id="model-seeds"),
        pytest.param(("selection", "metrics"), ["mse"], id="selection-metrics"),
        pytest.param(
            ("selection", "collapse_order"),
            ["retain_biological_draw_units"],
            id="selection-collapse",
        ),
        pytest.param(
            ("selection", "prezero_mechanism"), "sergio", id="selection-prezero"
        ),
        pytest.param(("selection", "pareto_rule"), "changed", id="selection-pareto"),
        pytest.param(("selection", "rank_rule"), "changed", id="selection-rank"),
        pytest.param(
            ("selection", "selection_tuple"),
            ["configuration_id"],
            id="selection-tuple",
        ),
        pytest.param(
            ("selection", "readiness", "minimum_required_controls_complete"),
            1,
            id="readiness-controls",
        ),
        pytest.param(
            (
                "selection",
                "readiness",
                "minimum_established_comparators_selectable",
            ),
            4,
            id="readiness-established",
        ),
        pytest.param(
            ("selection", "readiness", "minimum_modern_core_selectable"),
            2,
            id="readiness-modern",
        ),
        pytest.param(
            ("selection", "receipt_path"), "elsewhere.json", id="selection-receipt"
        ),
        pytest.param(
            ("budgets", "max_configurations_per_method"), 19, id="budget-configs"
        ),
        pytest.param(("budgets", "gpu_seconds_per_method"), 1, id="budget-gpu"),
        pytest.param(("budgets", "cpu_seconds_per_method"), 1, id="budget-cpu"),
        pytest.param(("budgets", "per_run_timeout_seconds"), 1, id="budget-timeout"),
        pytest.param(("budgets", "max_rss_bytes"), 1, id="budget-rss"),
        pytest.param(("budgets", "max_gpu_bytes"), 1, id="budget-gpu-memory"),
        pytest.param(
            ("budgets", "intrinsic_terminal_statuses"),
            ["failed"],
            id="budget-intrinsic-statuses",
        ),
        pytest.param(
            ("budgets", "blocking_statuses"),
            ["budget_exhausted"],
            id="budget-blocking-statuses",
        ),
        pytest.param(("storage", "max_log_receipt_bytes"), 1, id="storage-log"),
        pytest.param(
            ("storage", "max_executor_receipt_bytes"), 1, id="storage-executor"
        ),
        pytest.param(("storage", "max_record_bytes"), 1, id="storage-record"),
        pytest.param(("storage", "max_checkpoint_bytes"), 1, id="storage-checkpoint"),
        pytest.param(("storage", "reserve_bytes"), 1, id="storage-reserve"),
        pytest.param(("smoke", "receipt_path"), "elsewhere.json", id="smoke-path"),
        pytest.param(("smoke", "cells"), 899, id="smoke-cells"),
        pytest.param(("smoke", "genes"), 499, id="smoke-genes"),
        pytest.param(("smoke", "model_seed"), 43, id="smoke-seed"),
        pytest.param(("smoke", "batch_rule"), "changed", id="smoke-batches"),
        pytest.param(("smoke", "count_formula"), "changed", id="smoke-formula"),
        pytest.param(("smoke", "projection_multiplier"), 47, id="smoke-projection"),
        pytest.param(("smoke", "output_retention"), "retained", id="smoke-retention"),
    ),
)
def test_authority_rejects_policy_mutation(
    path: tuple[str, ...], replacement: object
) -> None:
    payload = _tracked_payload()
    _set_nested(payload, path, replacement)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


@pytest.mark.parametrize(
    "section_path",
    (
        pytest.param(("scope",), id="scope"),
        pytest.param(("selection",), id="selection"),
        pytest.param(("selection", "readiness"), id="readiness"),
        pytest.param(("budgets",), id="budgets"),
        pytest.param(("storage",), id="storage"),
        pytest.param(("smoke",), id="smoke"),
    ),
)
def test_authority_rejects_extra_nested_field(
    section_path: tuple[str, ...],
) -> None:
    payload = _tracked_payload()
    target: object = payload
    for key in section_path:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target["unexpected"] = "forged"
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError, match="missing or extra fields"):
        parse_comparator_tuning_authority(payload, registry=registry)


@pytest.mark.parametrize(
    "mutation",
    (
        "row-order",
        "row-count",
        "configuration-id",
        "duplicate-configuration-id",
        "duplicate-payload-under-another-id",
        "multiple-defaults",
        "default-payload",
        "payload-mutation",
    ),
)
def test_authority_rejects_grid_mutation(mutation: str) -> None:
    payload = _tracked_payload()
    configurations = payload["configurations"]
    assert isinstance(configurations, list)
    magic_default = configurations[1]
    magic_second = configurations[2]
    assert isinstance(magic_default, dict)
    assert isinstance(magic_second, dict)

    if mutation == "row-order":
        configurations[1], configurations[2] = configurations[2], configurations[1]
    elif mutation == "row-count":
        configurations.pop()
    elif mutation == "configuration-id":
        magic_second["configuration_id"] = "magic-t02"
    elif mutation == "duplicate-configuration-id":
        magic_second["configuration_id"] = magic_default["configuration_id"]
    elif mutation == "duplicate-payload-under-another-id":
        magic_second["payload"] = copy.deepcopy(magic_default["payload"])
    elif mutation == "multiple-defaults":
        magic_second["is_upstream_default"] = True
    elif mutation == "default-payload":
        magic_default["payload"] = copy.deepcopy(magic_second["payload"])
    elif mutation == "payload-mutation":
        second_payload = magic_second["payload"]
        assert isinstance(second_payload, dict)
        second_payload["diffusion_time"] = 2
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


def test_authority_rejects_signed_negative_zero_payload_mutation() -> None:
    payload = _tracked_payload()
    configurations = payload["configurations"]
    assert isinstance(configurations, list)
    afmf_default = configurations[18]
    assert isinstance(afmf_default, dict)
    afmf_payload = afmf_default["payload"]
    assert isinstance(afmf_payload, dict)
    afmf_payload["lambda_p"] = -0.0

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


def test_authority_rejects_unicode_payload_mutation() -> None:
    payload = _tracked_payload()
    configurations = payload["configurations"]
    assert isinstance(configurations, list)
    magic_default = configurations[1]
    assert isinstance(magic_default, dict)
    magic_payload = magic_default["payload"]
    assert isinstance(magic_payload, dict)
    magic_payload["solver"] = "\ud800"

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


@pytest.mark.parametrize("schema", ("old", "mixed"))
def test_authority_rejects_old_and_mixed_schema(schema: str) -> None:
    payload = _tracked_payload()
    if schema == "old":
        payload["schema_version"] = 1
        payload.pop("authority_revision")
        payload["payload_sha256"] = "0" * 64
    else:
        payload["payload_sha256"] = "0" * 64
        configurations = payload["configurations"]
        assert isinstance(configurations, list)
        first = configurations[0]
        assert isinstance(first, dict)
        first["payload_sha256"] = "0" * 64

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError, match="missing or extra fields"):
        parse_comparator_tuning_authority(payload, registry=registry)


@pytest.mark.parametrize(
    "malformation", ("noncanonical", "duplicate", "nonfinite", "unicode-drift")
)
def test_loader_rejects_malformed_authority_bytes(
    tmp_path: Path, malformation: str
) -> None:
    payload = _tracked_payload()
    canonical = json.dumps(payload, indent=2).encode() + b"\n"
    if malformation == "noncanonical":
        raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    elif malformation == "duplicate":
        raw = canonical.replace(
            b'  "schema_version": 2,',
            b'  "schema_version": 2,\n  "schema_version": 2,',
            1,
        )
    elif malformation == "nonfinite":
        raw = canonical.replace(b'    "cells": 900,', b'    "cells": NaN,', 1)
    elif malformation == "unicode-drift":
        raw = canonical.replace(b'"solver": "exact"', b'"solver": "\\u0065xact"', 1)
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(malformation)
    _write_authority(tmp_path, raw)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError):
        load_comparator_tuning_authority(
            tmp_path, registry=registry, require_clean=False
        )


def test_loader_rejects_non_regular_authority(tmp_path: Path) -> None:
    study = tmp_path / "study"
    study.mkdir()
    (study / "comparator_tuning.json").symlink_to(ROOT / "study/comparator_tuning.json")
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError, match="owned regular file"):
        load_comparator_tuning_authority(
            tmp_path, registry=registry, require_clean=False
        )


def test_tracked_authority_has_exact_grid_and_operational_contract() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )

    assert authority.method_order == tuple(EXPECTED_ORDER)
    assert {
        method_id: tuple(
            row.configuration_id for row in authority.configurations_for(method_id)
        )
        for method_id in authority.method_order
    } == EXPECTED_ORDER
    assert all(
        sum(row.is_upstream_default for row in authority.configurations_for(method_id))
        == 1
        and authority.configurations_for(method_id)[0].is_upstream_default
        for method_id in authority.method_order
    )
    assert authority.scheduled_same_input_ids == (
        "observed",
        "capacity-matched-ae",
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
    assert authority.required_control_ids == ("observed", "capacity-matched-ae")
    assert authority.established_comparator_ids == (
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
    )
    assert authority.modern_core_ids == ("scziva", "afmf", "biaeimpute", "sccr")
    assert authority.model_seeds == (42, 43, 44)
    assert (
        authority.receipt_path
        == "artifacts/study/development/evaluation/comparator_selection.json"
    )
    assert (
        authority.smoke_receipt_path
        == "artifacts/study/development/evaluation/comparator_smoke.json"
    )
    assert (
        DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
        DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
        DEVELOPMENT_MAX_RECORD_BYTES,
        DEVELOPMENT_MAX_CHECKPOINT_BYTES,
        DEVELOPMENT_STORAGE_RESERVE_BYTES,
    ) == (65_536, 65_536, 65_536, 67_108_864, 1_073_741_824)
