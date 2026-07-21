from __future__ import annotations

from dataclasses import replace
import hashlib
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SHA_A = "a" * 64
SHA_B = "b" * 64
ORTHOGONAL_ARTIFACT_BINDINGS = {
    "count_model_config_sha256": "c" * 64,
    "retained_calibration_artifact_sha256": "d" * 64,
    "score_fit_policy": "refit_cross_fitted_count_score_from_truth_free_input",
}


def _completed_checkpoint(tmp_path: Path):
    from maskimpute_benchmark.methods import load_method_registry, run_observed
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import (
        AdapterOutcome,
        AuthorizedConfiguration,
        CheckpointStore,
        CompetitionPlan,
        DatasetBinding,
        DatasetQCPolicy,
        DevelopmentBudget,
        RunPlanEntry,
        evaluate_adapter_outcome,
        implementation_source_sha256,
        prepare_dataset_for_execution,
    )
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    counts = np.asarray([[1, 0], [0, 2], [3, 1], [1, 2]], dtype=np.int64)
    dataset = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "dataset_id": ["dataset-test"] * 4,
                "mechanism": ["symsim"] * 4,
                "condition": ["moderate"] * 4,
                "biological_id": ["draw-01"] * 4,
                "technical_view": ["moderate"] * 4,
                "draw": np.ones(4, dtype=np.int64),
                "library_size": counts.sum(axis=1),
                "group": ["a", "a", "b", "b"],
            },
            index=[f"cell-{index}" for index in range(4)],
        ),
        var=pd.DataFrame(index=["gene-1", "gene-2"]),
        layers={"pre_capture_counts": counts + 1},
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
    binding = DatasetBinding(
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        dataset_sha256=benchmark_dataset_sha256(dataset),
        output_file_sha256=SHA_A,
        truth_sha256=SHA_B,
        output_path="dev/datasets/symsim/draw-01/moderate.h5ad",
        independent_unit_id="biological-test",
        cells=4,
        genes=2,
        manifest_sha256=SHA_A,
        protocol_sha256=SHA_B,
        design_sha256="c" * 64,
        seed_source_sha256="d" * 64,
    )
    prepared = prepare_dataset_for_execution(dataset, binding, DatasetQCPolicy.fixed())
    registry = load_method_registry(Path("study/methods.json"))
    spec = registry.by_id("observed")
    configuration = AuthorizedConfiguration.registry_default(spec)
    entry = RunPlanEntry(
        ordinal=1,
        run_id="run-observed-test",
        method_id="observed",
        dataset_id="dataset-test",
        source_dataset_sha256=binding.dataset_sha256,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        model_seed=None,
        configuration_id="registry-default",
        configuration_sha256=configuration.configuration_sha256,
        preflight_status="planned",
        preflight_reason=None,
    )
    plan_core = {
        "schema_version": 1,
        "input_hashes": {
            "dataset_manifest_sha256": SHA_A,
            "dataset_qc_policy_sha256": DatasetQCPolicy.fixed().sha256,
            "implementation_source_sha256": implementation_source_sha256(),
        },
        "entries": [entry.to_dict()],
        "configurations": [],
    }
    plan = CompetitionPlan(
        schema_version=1,
        input_hashes=plan_core["input_hashes"],
        entries=(entry,),
        plan_sha256=canonical_sha256(plan_core),
    )
    execution = run_observed(spec, prepared.method_input)
    outcome = AdapterOutcome.completed(
        execution,
        runtime_seconds=0.1,
        peak_rss_bytes=1,
        peak_gpu_bytes=0,
    )
    attempt = evaluate_adapter_outcome(
        entry,
        prepared,
        outcome,
    )
    budget = DevelopmentBudget()
    budget.record(
        spec,
        entry.configuration_sha256,
        outcome,
        counts_toward_configuration_limit=False,
        budget_scope=entry.method_id,
    )
    store = CheckpointStore(tmp_path / "competition")
    report = store.append(
        plan,
        None,
        attempt,
        budget,
        registry=registry,
        prepared_datasets={prepared.binding.dataset_id: prepared},
    )
    return plan, store, report, prepared


def _direct_projection_checkpoint(tmp_path: Path):
    from maskimpute_benchmark.comparator_tuning import (
        comparator_method_binding,
        load_comparator_tuning_authority,
    )
    from maskimpute_benchmark.fair_comparator_checkpoint import DirectCheckpointStore
    from maskimpute_benchmark.direct_values import freeze_direct_mapping
    from maskimpute_benchmark.fair_comparator_execution import (
        DIRECT_RECONSTRUCTION_METRICS,
    )
    from maskimpute_benchmark.fair_comparator_plan import (
        ComparatorRunIdentity,
        DirectAuthorizedConfiguration,
        DirectCompetitionPlan,
        DirectPlanEntry,
        describe_prepared_input,
        direct_run_id,
    )
    from maskimpute_benchmark.methods import load_method_registry

    _legacy_plan, _legacy_store, _legacy_report, prepared = _completed_checkpoint(
        tmp_path
    )
    prepared.evaluator_dataset.uns["provenance"]["seeds"] = {"measurement": 20_001}
    registry = load_method_registry(Path("study/methods.json"))
    authority = load_comparator_tuning_authority(
        Path.cwd(), registry=registry, require_clean=False
    )
    descriptor = describe_prepared_input(prepared)
    spec = registry.by_id("magic")
    method = comparator_method_binding(spec)

    selected_rows = authority.configurations_for("magic")[:1]
    plan_rows = authority.configurations_for("magic")[:3]
    configurations = tuple(
        DirectAuthorizedConfiguration(
            method=method,
            configuration_id=row.configuration_id,
            configuration_kind="comparator_tuning",
            payload=freeze_direct_mapping(row.payload),
            requires_count_score=False,
            requires_calibration=False,
        )
        for row in plan_rows
    )
    entries = []
    for ordinal, configuration in enumerate(configurations, start=1):
        identity = ComparatorRunIdentity(
            workflow_schema="maskimpute-fair-comparator-run-v1",
            authority_revision=authority.authority_revision,
            ordinal=ordinal,
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
        entries.append(
            DirectPlanEntry(
                run_id=direct_run_id(identity),
                identity=identity,
                preflight_status="planned",
                preflight_reason=None,
                requires_count_score=False,
                requires_calibration=False,
            )
        )
    plan = DirectCompetitionPlan(
        schema_version=1,
        identity_mode="direct-v1",
        authority_revision=authority.authority_revision,
        inputs=(descriptor,),
        entries=tuple(entries),
        configurations=configurations,
    )

    identity_rows = plan.to_dict()["entries"]
    assert isinstance(identity_rows, list)
    records = []
    for entry, encoded in zip(plan.entries, identity_rows, strict=True):
        assert isinstance(encoded, dict)
        reason = "synthetic_unavailable"
        records.append(
            {
                "run": {
                    "run_id": entry.run_id,
                    "identity": encoded["identity"],
                    "status": "unavailable",
                    "reason": reason,
                    "runtime_seconds": 1.0,
                    "peak_rss_bytes": 1,
                    "peak_gpu_bytes": 0,
                    "rss_measurement": "synthetic_parent_rss",
                    "gpu_measurement": "not_applicable_cpu",
                    "excluded_cell_count": prepared.audit.excluded_cell_count,
                    "excluded_cell_ids": list(prepared.audit.excluded_cell_ids),
                    "retained_cell_count": prepared.audit.retained_cell_count,
                    "retained_cell_ids": list(prepared.audit.retained_cell_ids),
                    "retained_gene_count": prepared.method_input.shape[1],
                    "observed_zero_count": int(
                        (prepared.method_input.counts == 0).sum()
                    ),
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
                        "identity": encoded["identity"],
                        "metric": metric,
                        "value": None,
                        "n": 0,
                        "status": "unavailable",
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
        )
    store = DirectCheckpointStore(tmp_path / "direct-checkpoint.json")
    store.write(
        plan,
        records,
        registry=registry,
        prepared_datasets={prepared.binding.dataset_id: prepared},
    )
    return (
        store.path,
        plan,
        registry,
        {prepared.binding.dataset_id: prepared},
        authority,
        selected_rows,
    )


def _all_direct_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for nested in value.values() for key in _all_direct_keys(nested)
        )
    if isinstance(value, list):
        return tuple(key for nested in value for key in _all_direct_keys(nested))
    return ()


def _direct_projection_checkpoint_with_selected_statuses(
    tmp_path: Path,
    statuses: tuple[str, str],
):
    from maskimpute_benchmark.fair_comparator_checkpoint import (
        replay_direct_development_budget,
    )
    from maskimpute_benchmark.fair_comparator_plan import direct_run_id

    checkpoint_path, plan, registry, prepared, authority, _selected = (
        _direct_projection_checkpoint(tmp_path)
    )
    first = plan.entries[0]
    second_identity = replace(first.identity, ordinal=2, model_seed=43)
    second = replace(
        first,
        run_id=direct_run_id(second_identity),
        identity=second_identity,
    )
    third_identity = replace(plan.entries[2].identity, ordinal=3)
    third = replace(
        plan.entries[2],
        run_id=direct_run_id(third_identity),
        identity=third_identity,
    )
    changed_plan = replace(
        plan,
        configurations=(plan.configurations[0], plan.configurations[2]),
        entries=(first, second, third),
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    encoded_entries = changed_plan.to_dict()["entries"]
    assert isinstance(encoded_entries, list)
    for record, entry, encoded, status in zip(
        checkpoint["records"],
        changed_plan.entries,
        encoded_entries,
        (*statuses, "unavailable"),
        strict=True,
    ):
        reason = None if status == "completed" else "synthetic_unavailable"
        record["run"].update(
            run_id=entry.run_id,
            identity=encoded["identity"],
            status=status,
            reason=reason,
        )
        record["run"]["stdout"]["terminal_reason"] = reason
        record["run"]["stderr"]["terminal_reason"] = reason
        for metric in record["metrics"]:
            metric["identity"] = encoded["identity"]
            if status == "completed":
                metric.update(
                    value=None,
                    n=0,
                    status="unavailable",
                    reason="synthetic_metric_unavailable",
                )
            else:
                metric.update(value=None, n=0, status=status, reason=reason)
    checkpoint["plan_snapshot"] = changed_plan.to_dict()
    checkpoint["budget"] = replay_direct_development_budget(
        registry,
        changed_plan.entries,
        checkpoint["records"],
    ).to_dict()
    checkpoint_path.write_text(
        json.dumps(
            checkpoint,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        checkpoint_path,
        changed_plan,
        registry,
        prepared,
        authority,
        authority.configurations_for("magic")[:1],
    )


def _contains_forbidden_identity_key(value: object) -> bool:
    forbidden = ("hash", "digest", "checksum", "fingerprint", "sha")
    return any(
        key.casefold() != "shape" and token in key.casefold()
        for key in _all_direct_keys(value)
        for token in forbidden
    )


def test_direct_checkpoint_projects_full_selected_comparator_payloads(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )

    (
        checkpoint_path,
        plan,
        registry,
        prepared,
        authority,
        selected_rows,
    ) = _direct_projection_checkpoint(tmp_path)
    with pytest.raises(DevelopmentEvaluationError, match="selection receipt|later"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=authority.schema_version,
                authority_revision=authority.authority_revision,
            ),
            comparator_authority=authority,
            selected_rows=selected_rows,
        )


@pytest.mark.parametrize("selection", ("empty", "subset", "arbitrary-terminal"))
def test_public_direct_handoff_requires_later_validated_selection_receipt(
    tmp_path: Path,
    selection: str,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )

    checkpoint_path, plan, registry, prepared, authority, selected = (
        _direct_projection_checkpoint(tmp_path)
    )
    if selection == "empty":
        rows = ()
    elif selection == "subset":
        rows = selected
    else:
        rows = authority.configurations_for("magic")[1:2]

    with pytest.raises(DevelopmentEvaluationError, match="selection receipt|later"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=authority,
            selected_rows=rows,
        )


@pytest.mark.parametrize(
    "statuses",
    (("completed", "completed"), ("completed", "unavailable")),
    ids=("all-completed", "mixed-completed-intrinsic-terminal"),
)
def test_direct_projection_accepts_completed_selected_configuration(
    tmp_path: Path,
    statuses: tuple[str, str],
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )

    checkpoint_path, plan, registry, prepared, authority, selected = (
        _direct_projection_checkpoint_with_selected_statuses(tmp_path, statuses)
    )

    with pytest.raises(DevelopmentEvaluationError, match="selection receipt|later"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=authority,
            selected_rows=selected,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("plan_sha256", "0" * 64),
        ("selection_complete", True),
        ("selection_receipt", {"selection_complete": True}),
    ),
)
def test_direct_projection_rejects_mixed_or_caller_selection_claims(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )

    checkpoint_path, plan, registry, prepared, authority, selected = (
        _direct_projection_checkpoint(tmp_path)
    )
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload[field] = value
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
    reference = ComparatorAuthorityReference(
        path="study/comparator_tuning.json",
        schema_version=authority.schema_version,
        authority_revision=authority.authority_revision,
    )
    with pytest.raises(DevelopmentEvaluationError, match="checkpoint validation"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=reference,
            comparator_authority=authority,
            selected_rows=selected,
        )


def test_direct_projection_rejects_incomplete_comparator_denominator(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )
    from maskimpute_benchmark.fair_comparator_checkpoint import DirectCheckpointStore

    checkpoint_path, plan, registry, prepared, authority, selected = (
        _direct_projection_checkpoint(tmp_path)
    )
    DirectCheckpointStore(checkpoint_path).write(
        plan,
        (),
        registry=registry,
        prepared_datasets=prepared,
    )
    with pytest.raises(DevelopmentEvaluationError, match="denominator is not terminal"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=authority.schema_version,
                authority_revision=authority.authority_revision,
            ),
            comparator_authority=authority,
            selected_rows=selected,
        )


def test_direct_projection_requires_selected_row_in_terminal_comparator_entries(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )
    from maskimpute_benchmark.fair_comparator_checkpoint import (
        replay_direct_development_budget,
    )

    checkpoint_path, plan, registry, prepared, authority, _selected = (
        _direct_projection_checkpoint(tmp_path)
    )
    selected = authority.configurations_for("magic")[2:3]
    changed_plan = replace(plan, entries=plan.entries[:-1])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["plan_snapshot"] = changed_plan.to_dict()
    checkpoint["planned_run_count"] = len(changed_plan.entries)
    checkpoint["records"] = checkpoint["records"][:-1]
    checkpoint["status"] = "completed"
    checkpoint["comparator_selection_status"] = "complete_terminal_denominator"
    checkpoint["budget"] = replay_direct_development_budget(
        registry,
        changed_plan.entries,
        checkpoint["records"],
    ).to_dict()
    checkpoint_path.write_text(
        json.dumps(
            checkpoint,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DevelopmentEvaluationError, match="checkpoint validation"):
        project_direct_comparator_evidence(
            checkpoint_path,
            changed_plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=authority,
            selected_rows=selected,
        )


def test_direct_projection_rejects_payload_plan_record_and_authority_drift(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )

    checkpoint_path, plan, registry, prepared, authority, selected = (
        _direct_projection_checkpoint(tmp_path)
    )
    reference = ComparatorAuthorityReference(
        path="study/comparator_tuning.json",
        schema_version=authority.schema_version,
        authority_revision=authority.authority_revision,
    )
    configuration = plan.configurations[0]
    changed_payload = tuple(
        (name, 7 if name == "knn" else value) for name, value in configuration.payload
    )
    changed_configuration = replace(configuration, payload=changed_payload)
    changed_identity = replace(
        plan.entries[0].identity,
        configuration_payload=changed_payload,
    )
    changed_entry = replace(plan.entries[0], identity=changed_identity)
    changed_plan = replace(
        plan,
        configurations=(changed_configuration, *plan.configurations[1:]),
        entries=(changed_entry, *plan.entries[1:]),
    )
    with pytest.raises(DevelopmentEvaluationError, match="configuration differs"):
        project_direct_comparator_evidence(
            checkpoint_path,
            changed_plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=reference,
            comparator_authority=authority,
            selected_rows=selected,
        )

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["records"][0]["run"]["identity"]["configuration_id"] = "drifted"
    checkpoint_path.write_text(
        json.dumps(
            checkpoint,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DevelopmentEvaluationError, match="checkpoint validation"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=reference,
            comparator_authority=authority,
            selected_rows=selected,
        )

    mismatched_reference = replace(
        reference,
        authority_revision="fair-comparator-direct-v2",
    )
    with pytest.raises(DevelopmentEvaluationError, match="authority differs"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets=prepared,
            comparator_reference=mismatched_reference,
            comparator_authority=authority,
            selected_rows=selected,
        )


def test_direct_projection_does_not_reinterpret_legacy_checkpoint_routes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.comparator_tuning import (
        ComparatorAuthorityReference,
        load_comparator_tuning_authority,
    )
    from maskimpute_benchmark.development_evaluation import (
        project_direct_comparator_evidence,
    )
    from maskimpute_benchmark.methods import load_method_registry

    legacy_plan, legacy_store, _report, prepared = _completed_checkpoint(tmp_path)
    registry = load_method_registry(Path("study/methods.json"))
    authority = load_comparator_tuning_authority(
        Path.cwd(), registry=registry, require_clean=False
    )
    with pytest.raises(TypeError, match="DirectCompetitionPlan"):
        project_direct_comparator_evidence(
            legacy_store.checkpoint_path,
            legacy_plan,
            registry=registry,
            prepared_datasets={prepared.binding.dataset_id: prepared},
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=authority.schema_version,
                authority_revision=authority.authority_revision,
            ),
            comparator_authority=authority,
            selected_rows=(),
        )


def test_null_de_uses_deterministic_balanced_stratified_split() -> None:
    from maskimpute_benchmark.development_evaluation import (
        balanced_null_split,
        evaluate_null_de_fpr,
        fixed_null_de_gene_mask,
    )

    cell_ids = tuple(f"cell-{index:03d}" for index in range(40))
    strata = tuple("type-a" if index < 20 else "type-b" for index in range(40))
    entropy = "f" * 64
    split, split_sha256 = balanced_null_split(cell_ids, strata, entropy_sha256=entropy)

    assert split.dtype == np.bool_
    assert split.shape == (40,)
    assert len(split_sha256) == 64
    for stratum in ("type-a", "type-b"):
        selected = split[np.asarray(strata) == stratum]
        assert int(selected.sum()) == 10
    repeated, repeated_sha256 = balanced_null_split(
        cell_ids, strata, entropy_sha256=entropy
    )
    np.testing.assert_array_equal(repeated, split)
    assert repeated_sha256 == split_sha256

    # A method output aligned to the evaluator-private pseudo-condition should
    # be detected as false differential expression for every testable gene.
    baseline = np.asarray(
        [[((cell + 1) * (gene + 3)) % 17 for gene in range(120)] for cell in range(40)],
        dtype=float,
    )
    fixed_mask, gene_mask_sha256 = fixed_null_de_gene_mask(
        baseline,
        cell_ids,
        strata,
        entropy_sha256=entropy,
    )
    output = baseline + split.astype(float)[:, None] * 100.0
    result = evaluate_null_de_fpr(
        output,
        cell_ids,
        strata,
        fixed_gene_mask=fixed_mask,
        entropy_sha256=entropy,
    )

    assert result.status == "completed"
    assert result.fpr == 1.0
    assert result.nominal_alpha == 0.05
    assert result.n_tested_genes == int(fixed_mask.sum())
    assert result.gene_mask_sha256 == gene_mask_sha256
    assert result.split_sha256 == split_sha256

    constantized = evaluate_null_de_fpr(
        np.zeros_like(output),
        cell_ids,
        strata,
        fixed_gene_mask=fixed_mask,
        entropy_sha256=entropy,
    )
    assert constantized.status == "unavailable"
    assert constantized.fpr is None
    assert constantized.n_tested_genes == int(fixed_mask.sum())
    assert constantized.reason == "method_non_testable_on_fixed_gene_denominator"


def test_completed_checkpoint_loader_binds_and_revalidates_every_raw_artifact(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        load_completed_reconstruction_checkpoint,
    )

    plan, store, report, prepared = _completed_checkpoint(tmp_path)
    evidence = load_completed_reconstruction_checkpoint(
        store.output_dir,
        plan,
        prepared_datasets={prepared.binding.dataset_id: prepared},
    )

    assert evidence.checkpoint_sha256 == report.checkpoint_sha256
    assert evidence.checkpoint_file_sha256 != report.checkpoint_sha256
    assert evidence.plan_sha256 == plan.plan_sha256
    assert {(item.run_id, item.kind) for item in evidence.raw_artifacts} == {
        ("run-observed-test", "stdout"),
        ("run-observed-test", "stderr"),
        ("run-observed-test", "native_output"),
        ("run-observed-test", "evaluator_output"),
    }
    assert all(len(item.file_sha256) == 64 for item in evidence.raw_artifacts)

    evaluator = next(
        item for item in evidence.raw_artifacts if item.kind == "evaluator_output"
    )
    (store.output_dir / evaluator.path).write_bytes(b"tampered")
    with pytest.raises(DevelopmentEvaluationError, match="checkpoint"):
        load_completed_reconstruction_checkpoint(
            store.output_dir,
            plan,
            prepared_datasets={prepared.binding.dataset_id: prepared},
        )


def test_completed_checkpoint_loader_rejects_cross_read_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        load_completed_reconstruction_checkpoint,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import CheckpointStore

    plan, store, _report, prepared = _completed_checkpoint(tmp_path)
    original_load = CheckpointStore.load

    def replace_after_load(
        checkpoint_store, requested_plan, *, registry, prepared_datasets
    ):
        report = original_load(
            checkpoint_store,
            requested_plan,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )
        payload = json.loads(checkpoint_store.checkpoint_path.read_bytes())
        payload["schema_version"] = 2
        core = {
            key: value for key, value in payload.items() if key != "checkpoint_sha256"
        }
        payload["checkpoint_sha256"] = canonical_sha256(core)
        checkpoint_store.checkpoint_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        return report

    monkeypatch.setattr(CheckpointStore, "load", replace_after_load)
    with pytest.raises(DevelopmentEvaluationError, match="changed"):
        load_completed_reconstruction_checkpoint(
            store.output_dir,
            plan,
            prepared_datasets={prepared.binding.dataset_id: prepared},
        )


def test_reconstruction_bridge_emits_exact_selection_rows_and_null_de_audit(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        build_reconstruction_selection_records,
        load_completed_reconstruction_checkpoint,
    )
    from maskimpute_benchmark.selection import MethodDeclaration

    plan, store, _report, prepared = _completed_checkpoint(tmp_path)
    evidence = load_completed_reconstruction_checkpoint(
        store.output_dir,
        plan,
        prepared_datasets={prepared.binding.dataset_id: prepared},
    )
    # Recreate the prepared evaluator dataset from the exact fixture used to
    # produce the checkpoint. The bridge must never send its group labels back
    # to the method process; they enter only here, after execution.
    counts = np.asarray([[1, 0], [0, 2], [3, 1], [1, 2]], dtype=np.int64)
    dataset = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "dataset_id": ["dataset-test"] * 4,
                "mechanism": ["symsim"] * 4,
                "condition": ["moderate"] * 4,
                "biological_id": ["draw-01"] * 4,
                "technical_view": ["moderate"] * 4,
                "draw": np.ones(4, dtype=np.int64),
                "library_size": counts.sum(axis=1),
                "group": ["a", "a", "b", "b"],
            },
            index=[f"cell-{index}" for index in range(4)],
        ),
        var=pd.DataFrame(index=["gene-1", "gene-2"]),
        layers={"pre_capture_counts": counts + 1},
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
    from maskimpute_benchmark.runner import (
        DatasetBinding,
        DatasetQCPolicy,
        prepare_dataset_for_execution,
    )
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    binding = DatasetBinding(
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        dataset_sha256=benchmark_dataset_sha256(dataset),
        output_file_sha256=SHA_A,
        truth_sha256=SHA_B,
        output_path="dev/datasets/symsim/draw-01/moderate.h5ad",
        independent_unit_id="biological-test",
        cells=4,
        genes=2,
        manifest_sha256=SHA_A,
        protocol_sha256=SHA_B,
        design_sha256="c" * 64,
        seed_source_sha256="d" * 64,
    )
    prepared = prepare_dataset_for_execution(dataset, binding, DatasetQCPolicy.fixed())
    binding_sha = "e" * 64
    bundle = build_reconstruction_selection_records(
        evidence,
        checkpoint_directory=store.output_dir,
        prepared_datasets={"dataset-test": prepared},
        declarations=(
            MethodDeclaration(
                id="observed",
                role="observed_control",
                track="same_input",
                stochastic=False,
                required_for_claim=True,
            ),
        ),
        method_bindings={"observed": binding_sha},
    )

    assert len(bundle.records) == 7
    assert {row["metric"] for row in bundle.records} == {
        "mse",
        "mse_dropout",
        "gnrmse",
        "mse_pre_dropout_zero",
        "corr_err",
        "mse_non_dropout_nonzero",
        "null_de_fpr",
    }
    assert all(row["method"] == "observed" for row in bundle.records)
    assert all(row["method_sha256"] == binding_sha for row in bundle.records)
    assert all(
        set(row)
        == {
            "mechanism",
            "biological_id",
            "technical_view",
            "dataset_id",
            "dataset_sha256",
            "method",
            "method_sha256",
            "model_seed",
            "metric",
            "value",
            "status",
        }
        for row in bundle.records
    )
    assert len(bundle.null_de_audits) == 1
    assert bundle.null_de_audits[0]["reason"] == "insufficient_cells_per_stratum"
    assert len(bundle.null_de_audits[0]["split_entropy_sha256"]) == 64
    assert (
        "completed_checkpoint" in bundle.null_de_audits[0]["split_entropy_derivation"]
    )
    assert len(bundle.null_de_audits[0]["gene_mask_sha256"]) == 64


def test_rna_protein_scores_are_matched_features_across_cells_in_one_specimen() -> None:
    from maskimpute_benchmark.development_evaluation import (
        rna_protein_concordance_units,
    )

    protein = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
        ],
        dtype=float,
    )
    candidate = np.column_stack((protein, np.arange(6)))
    observed = np.column_stack((protein[::-1], np.arange(6)))
    gene_ids = ("HUMAN_CD4", "HUMAN_CD8A", "HUMAN_CD14", "HUMAN_CD19", "HUMAN_OTHER")
    protein_ids = ("CD4", "CD8", "CD14", "CD19")
    cell_ids = tuple(f"cell-{index}" for index in range(6))

    candidate_units = rna_protein_concordance_units(
        candidate, gene_ids, protein, protein_ids, cell_ids=cell_ids
    )
    observed_units = rna_protein_concordance_units(
        observed, gene_ids, protein, protein_ids, cell_ids=cell_ids
    )
    assert len(candidate_units) == 4
    assert len(observed_units) == 4
    assert all(unit.value == pytest.approx(1.0) for unit in candidate_units)
    assert all(unit.value == pytest.approx(-1.0) for unit in observed_units)
    assert {unit.biological_id for unit in candidate_units} == {"cbmc-single-specimen"}


def test_real_source_validation_binds_ledger_receipts_and_every_source_byte(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        validate_real_source_artifacts,
    )
    from maskimpute_benchmark.sources import load_source_ledger

    repository = tmp_path / "repository"
    source_specs = (
        ("baron-pancreas-umi", "semisynthetic_source", "baron.dat"),
        ("cite-seq-cbmc-rna-protein", "orthogonal_validation", "cbmc.dat"),
        ("tung-ipsc-ercc-bulk-replicates", "orthogonal_validation", "tung.dat"),
    )
    sources = []
    for index, (source_id, role, name) in enumerate(source_specs, start=1):
        data = f"source-{index}\n".encode()
        digest = hashlib.sha256(data).hexdigest()
        path = repository / "artifacts/external/data" / source_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        sources.append(
            {
                "id": source_id,
                "role": role,
                "mechanism": "semisynthetic" if index == 1 else None,
                "source_type": "data",
                "url": f"https://example.org/{source_id}",
                "revision": f"GSE{index}:2026-07-12",
                "license": "CC0-1.0",
                "license_url": "https://example.org/license",
                "citation_doi": f"10.1234/source.{index}",
                "expected_checksum": None,
                "eligibility": "eligible",
                "endpoints": ["source_validation"],
                "artifacts": [
                    {
                        "name": name,
                        "url": f"https://example.org/{name}",
                        "expected_checksum": {"algorithm": "sha256", "value": digest},
                    }
                ],
            }
        )
    ledger_path = repository / "study/sources.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
    ledger = load_source_ledger(ledger_path)
    for source in sources:
        artifact = source["artifacts"][0]
        artifact_path = (
            repository / "artifacts/external/data" / source["id"] / artifact["name"]
        )
        receipt = {
            "schema_version": 1,
            "source_id": source["id"],
            "role": source["role"],
            "source_type": "data",
            "source_url": source["url"],
            "revision": source["revision"],
            "resolved_revision": source["revision"],
            "license": source["license"],
            "citation_doi": source["citation_doi"],
            "verified_checksum": None,
            "ledger_sha256": ledger.sha256,
            "artifacts": [
                {
                    "name": artifact["name"],
                    "sha256": artifact["expected_checksum"]["value"],
                    "size_bytes": artifact_path.stat().st_size,
                }
            ],
        }
        receipt_path = (
            repository / "artifacts/external/receipts" / f"{source['id']}.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )

    evidence = validate_real_source_artifacts(repository)

    assert evidence.ledger_sha256 == ledger.sha256
    assert len(evidence.receipts) == 3
    assert len(evidence.artifacts) == 3
    assert {item.source_id for item in evidence.artifacts} == {
        value[0] for value in source_specs
    }

    changed = repository / "artifacts/external/data/baron-pancreas-umi/baron.dat"
    changed.write_bytes(b"tampered")
    with pytest.raises(DevelopmentEvaluationError, match="checksum"):
        validate_real_source_artifacts(repository)


def test_real_source_preparation_uses_only_prespecified_endpoint_fields(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        prepare_baron_source,
        prepare_cite_seq_source,
        prepare_tung_source,
    )

    rna_path = tmp_path / "rna.csv.gz"
    adt_path = tmp_path / "adt.csv.gz"
    pd.DataFrame(
        {
            "cell-1": [1, 9, 0],
            "cell-2": [2, 8, 1],
            "cell-3": [3, 7, 2],
            "cell-4": [4, 6, 3],
        },
        index=["HUMAN_CD4", "HUMAN_OTHER", "HUMAN_CD8A"],
    ).to_csv(rna_path, compression="gzip")
    pd.DataFrame(
        {
            "cell-1": [1, 0],
            "cell-2": [2, 1],
            "cell-3": [3, 2],
            "cell-4": [4, 3],
        },
        index=["CD4", "CD8"],
    ).to_csv(adt_path, compression="gzip")
    cite = prepare_cite_seq_source(rna_path, adt_path)
    assert cite.cell_ids == ("cell-1", "cell-2", "cell-3", "cell-4")
    assert cite.gene_ids == ("CD4", "CD8A", "OTHER")
    assert cite.endpoint_gene_ids == ("CD4", "CD8A")
    assert cite.rna_counts.shape == (4, 3)
    assert cite.protein_counts.shape == (4, 2)

    genes = ["ENSG000001", "ERCC-00002"]
    single_sample = tmp_path / "single-sample.tsv.gz"
    bulk_sample = tmp_path / "bulk-sample.tsv.gz"
    single_lane = tmp_path / "single-lane.tsv.gz"
    pd.DataFrame(
        [
            ["person-1", "r1", "A01", 1, 2],
            ["person-1", "r1", "A02", 3, 4],
            ["person-2", "r1", "A01", 5, 6],
            ["person-2", "r1", "A02", 7, 8],
        ],
        columns=["individual", "replicate", "well", *genes],
    ).to_csv(single_sample, sep="\t", index=False, compression="gzip")
    pd.DataFrame(
        [
            ["person-1", "r1", "bulk", 10, 20],
            ["person-2", "r1", "bulk", 30, 40],
        ],
        columns=["individual", "replicate", "well", *genes],
    ).to_csv(bulk_sample, sep="\t", index=False, compression="gzip")
    lane_rows = []
    for person, well, values in (
        ("person-1", "A01", (1, 2)),
        ("person-1", "A02", (3, 4)),
        ("person-2", "A01", (5, 6)),
        ("person-2", "A02", (7, 8)),
    ):
        for flow in ("flow-a", "flow-b"):
            lane = "lane-1" if well == "A01" else "lane-2"
            lane_rows.append([person, "r1", well, "idx", lane, flow, *values])
    pd.DataFrame(
        lane_rows,
        columns=[
            "individual",
            "replicate",
            "well",
            "index",
            "lane",
            "flow_cell",
            *genes,
        ],
    ).to_csv(single_lane, sep="\t", index=False, compression="gzip")
    tung = prepare_tung_source(single_sample, bulk_sample, single_lane)
    assert tung.counts.shape == (4, 2)
    assert tung.gene_ids == tuple(genes)
    assert tung.ercc_mask.tolist() == [False, True]
    assert set(tung.bulk_profiles) == {"person-1:r1", "person-2:r1"}
    assert set(tung.lane_profiles) == {
        "person-1:r1:flow-a:lane-1",
        "person-1:r1:flow-a:lane-2",
        "person-1:r1:flow-b:lane-1",
        "person-1:r1:flow-b:lane-2",
        "person-2:r1:flow-a:lane-1",
        "person-2:r1:flow-a:lane-2",
        "person-2:r1:flow-b:lane-1",
        "person-2:r1:flow-b:lane-2",
    }

    baron_path = tmp_path / "baron.tar"
    member_names = [
        "GSM2230757_human1_umifm_counts.csv.gz",
        "GSM2230758_human2_umifm_counts.csv.gz",
        "GSM2230759_human3_umifm_counts.csv.gz",
        "GSM2230760_human4_umifm_counts.csv.gz",
        "GSM2230761_mouse1_umifm_counts.csv.gz",
        "GSM2230762_mouse2_umifm_counts.csv.gz",
    ]
    with tarfile.open(baron_path, "w") as archive:
        for member_name in member_names:
            raw = (
                b",barcode,assigned_cluster,G1\ncell-1,bc,type,1\n"
                if "mouse" in member_name
                else b",barcode,assigned_cluster,G1,G2\ncell-1,bc,type,1,2\n"
            )
            compressed = gzip.compress(raw)
            info = tarfile.TarInfo(member_name)
            info.size = len(compressed)
            archive.addfile(info, io.BytesIO(compressed))
    baron = prepare_baron_source(baron_path)
    assert baron.member_names == tuple(member_names)
    assert baron.gene_counts == (2, 2, 2, 2, 1, 1)
    assert baron.human_gene_count == 2
    assert baron.mouse_gene_count == 1
    assert baron.cell_counts == (1, 1, 1, 1, 1, 1)


def test_tung_endpoint_units_keep_replicates_nested_within_individuals() -> None:
    from types import MappingProxyType

    from maskimpute_benchmark.development_evaluation import (
        TungSource,
        tung_concordance_units,
    )

    gene_ids = tuple(f"ENSG-{index}" for index in range(4)) + tuple(
        f"ERCC-{index}" for index in range(4)
    )
    counts = np.asarray(
        [
            [1, 1000, 200, 10, 50, 5, 1, 2],
            [1, 2, 10, 100, 1, 5, 20, 2],
            [1, 500, 100, 5, 25, 2, 1, 1],
            [1, 1, 5, 50, 1, 2, 10, 1],
        ],
        dtype=np.int32,
    )
    ercc = np.asarray([False] * 4 + [True] * 4)
    bulk = {
        "person-1:r1": counts[:2].sum(axis=0).astype(float),
        "person-2:r1": counts[2:].sum(axis=0).astype(float),
    }
    # A reference-zero feature is outside the fixed evaluator denominator,
    # even when a candidate makes it large.
    for profile in bulk.values():
        profile[0] = 0.0
    lanes = {
        f"{sample}:{flow}": profile
        for sample, profile in bulk.items()
        for flow in ("flow-a", "flow-b")
    }
    source = TungSource(
        cell_ids=("p1-a", "p1-b", "p2-a", "p2-b"),
        sample_ids=("person-1:r1", "person-1:r1", "person-2:r1", "person-2:r1"),
        individual_ids=("person-1", "person-1", "person-2", "person-2"),
        replicate_ids=("r1", "r1", "r1", "r1"),
        gene_ids=gene_ids,
        counts=counts,
        ercc_mask=ercc,
        bulk_profiles=MappingProxyType(bulk),
        lane_profiles=MappingProxyType(lanes),
        single_sample_file_sha256="a" * 64,
        bulk_sample_file_sha256="b" * 64,
        single_lane_file_sha256="c" * 64,
    )
    libraries = counts.sum(axis=1, keepdims=True)
    output = np.log2(counts / libraries * 10_000.0 + 1.0)
    output[:, 0] = np.log2(10_000.0)
    old_equal_cell_mean = np.mean(np.exp2(output[:2]) - 1.0, axis=0)
    assert (
        np.argsort(old_equal_cell_mean[1:4]).tolist()
        != np.argsort(bulk["person-1:r1"][1:4]).tolist()
    )

    units = tung_concordance_units(output, source)

    assert set(units) == {
        "ercc_recovery",
        "technical_replicate_concordance",
        "bulk_pseudobulk_concordance",
    }
    assert len(units["ercc_recovery"]) == 2
    assert len(units["bulk_pseudobulk_concordance"]) == 2
    assert len(units["technical_replicate_concordance"]) == 4
    assert {unit.biological_id for values in units.values() for unit in values} == {
        "person-1",
        "person-2",
    }
    assert all(
        unit.value == pytest.approx(1.0) for values in units.values() for unit in values
    )


def test_cite_panel_preserves_distinct_nonendpoint_symbols_that_differ_by_case(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import prepare_cite_seq_source

    rna = tmp_path / "rna.csv.gz"
    adt = tmp_path / "adt.csv.gz"
    pd.DataFrame(
        np.arange(16).reshape(4, 4),
        index=["HUMAN_CD4", "HUMAN_CD8A", "HUMAN_C2ORF15", "HUMAN_C2orf15"],
        columns=[f"cell-{index}" for index in range(4)],
    ).to_csv(rna, compression="gzip")
    pd.DataFrame(
        np.arange(8).reshape(2, 4),
        index=["CD4", "CD8"],
        columns=[f"cell-{index}" for index in range(4)],
    ).to_csv(adt, compression="gzip")

    source = prepare_cite_seq_source(rna, adt)

    assert "C2ORF15" in source.gene_ids
    assert "C2orf15" in source.gene_ids
    assert source.endpoint_gene_ids == ("CD4", "CD8A")


def test_orthogonal_output_producer_exposes_only_truth_free_method_input(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        OrthogonalConfiguration,
        OrthogonalInput,
        load_orthogonal_output_evidence,
        produce_orthogonal_outputs,
    )
    from maskimpute_benchmark.methods import prepare_method_input
    from maskimpute_benchmark.protocol import canonical_sha256

    counts = np.asarray([[1, 0, 2], [0, 3, 1], [2, 1, 0], [3, 2, 1]])
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(4)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(3)]),
    )
    adata.uns["source_dataset_sha256"] = "a" * 64
    adata.uns["allowed_covariates"] = {"obs": [], "var": []}
    method_input = prepare_method_input(adata)
    payload = {"method_version": "v27", "hyperparameters": {"gate_gamma": 1.0}}
    configuration = OrthogonalConfiguration(
        configuration_id="v27-test",
        configuration_sha256=canonical_sha256(payload),
        payload=payload,
    )
    seen = []

    def executor(request):
        seen.append(request)
        assert not hasattr(request, "protein_counts")
        assert not hasattr(request, "bulk_profiles")
        assert not hasattr(request.method_input, "obs")
        return request.method_input.counts

    evidence = produce_orthogonal_outputs(
        tmp_path / "orthogonal",
        inputs=(OrthogonalInput("source-test", method_input),),
        configurations=(configuration,),
        model_seeds=(42,),
        artifact_bindings=ORTHOGONAL_ARTIFACT_BINDINGS,
        executor=executor,
    )

    assert len(seen) == 1
    assert len(evidence.records) == 2  # observed plus candidate
    assert evidence.records[0]["configuration"] == "observed"
    assert evidence.records[1]["configuration"] == "v27-test"
    assert all(record["status"] == "completed" for record in evidence.records)
    assert all(len(record["output_file_sha256"]) == 64 for record in evidence.records)
    assert all(
        record["output_encoding"] == "zlib_raw_f64_v1" for record in evidence.records
    )
    assert all(
        str(record["output_path"]).endswith(".log2-cp10k-f64.zlib")
        for record in evidence.records
    )
    completed_record = evidence.records[1]
    compressed_path = evidence.output_directory / str(completed_record["output_path"])
    compressed = compressed_path.read_bytes()
    raw = zlib.decompress(compressed)
    expected_raw = np.asarray(executor(seen[-1]), dtype="<f8", order="C")
    from maskimpute_benchmark.methods import count_equivalent_to_log2_cp10k

    expected_raw = np.asarray(
        count_equivalent_to_log2_cp10k(expected_raw), dtype="<f8", order="C"
    ).tobytes(order="C")
    assert raw == expected_raw
    assert completed_record["output_compressed_nbytes"] == len(compressed)
    assert completed_record["output_uncompressed_nbytes"] == len(raw)
    assert (
        completed_record["output_uncompressed_sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
    assert len(evidence.manifest_sha256) == 64
    assert evidence.manifest_path.read_bytes().endswith(b"\n")

    original_manifest = evidence.manifest_path.read_bytes()
    forged = json.loads(original_manifest)
    expected_authority = json.loads(json.dumps(forged["authority"]))
    forged["authority"]["artifact_bindings"]["count_model_config_sha256"] = "e" * 64
    forged_core = {
        key: value for key, value in forged.items() if key != "manifest_sha256"
    }
    forged["manifest_sha256"] = canonical_sha256(forged_core)
    evidence.manifest_path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(Exception, match="authority"):
        load_orthogonal_output_evidence(
            evidence.output_directory,
            expected_authority=expected_authority,
        )
    evidence.manifest_path.write_bytes(original_manifest)

    aliased = json.loads(original_manifest)
    aliased["records"][1]["output_path"] = aliased["records"][0]["output_path"]
    alias_core = {
        key: value for key, value in aliased.items() if key != "manifest_sha256"
    }
    aliased["manifest_sha256"] = canonical_sha256(alias_core)
    evidence.manifest_path.write_text(
        json.dumps(aliased, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(Exception, match="binding|path"):
        load_orthogonal_output_evidence(
            evidence.output_directory,
            expected_authority=expected_authority,
        )
    evidence.manifest_path.write_bytes(original_manifest)

    output_path = evidence.output_directory / evidence.records[1]["output_path"]
    original_output = output_path.read_bytes()
    trailing = original_output + b"trailing-stream-data"
    trailing_manifest = json.loads(original_manifest)
    trailing_manifest["records"][1]["output_file_sha256"] = hashlib.sha256(
        trailing
    ).hexdigest()
    trailing_manifest["records"][1]["output_compressed_nbytes"] = len(trailing)
    trailing_core = {
        key: value
        for key, value in trailing_manifest.items()
        if key != "manifest_sha256"
    }
    trailing_manifest["manifest_sha256"] = canonical_sha256(trailing_core)
    output_path.write_bytes(trailing)
    evidence.manifest_path.write_text(
        json.dumps(trailing_manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(Exception, match="compressed|encoding|receipt"):
        load_orthogonal_output_evidence(
            evidence.output_directory,
            expected_authority=expected_authority,
        )

    evidence.manifest_path.write_bytes(original_manifest)
    output_path.write_bytes(original_output)
    output_path.write_bytes(b"tampered")
    with pytest.raises(Exception, match="checksum|existing"):
        produce_orthogonal_outputs(
            tmp_path / "orthogonal",
            inputs=(OrthogonalInput("source-test", method_input),),
            configurations=(configuration,),
            model_seeds=(42,),
            artifact_bindings=ORTHOGONAL_ARTIFACT_BINDINGS,
            executor=executor,
        )


def test_orthogonal_storage_preflight_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.development_evaluation as evaluation

    output_directory = tmp_path / "orthogonal"
    output_directory.mkdir()
    monkeypatch.setattr(
        evaluation.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(Exception, match="free storage|compressed-output bound"):
        evaluation._preflight_orthogonal_output_storage(
            output_directory,
            remaining_shapes=((4, 3),),
        )


def test_orthogonal_output_producer_resumes_validated_record_prefix(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        OrthogonalConfiguration,
        OrthogonalInput,
        produce_orthogonal_outputs,
    )
    from maskimpute_benchmark.methods import prepare_method_input
    from maskimpute_benchmark.protocol import canonical_sha256

    counts = np.asarray([[1, 0, 2], [0, 3, 1], [2, 1, 0], [3, 2, 1]])
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(4)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(3)]),
    )
    adata.uns["source_dataset_sha256"] = "a" * 64
    adata.uns["allowed_covariates"] = {"obs": [], "var": []}
    method_input = prepare_method_input(adata)
    payload = {"method_version": "v27"}
    configuration = OrthogonalConfiguration(
        "v27-test", canonical_sha256(payload), payload
    )
    output_directory = tmp_path / "orthogonal"

    def interrupt_after_observed(_request):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        produce_orthogonal_outputs(
            output_directory,
            inputs=(OrthogonalInput("source-test", method_input),),
            configurations=(configuration,),
            model_seeds=(42, 43),
            artifact_bindings=ORTHOGONAL_ARTIFACT_BINDINGS,
            executor=interrupt_after_observed,
        )
    running = json.loads((output_directory / "orthogonal_outputs.json").read_bytes())
    assert running["status"] == "running"
    assert running["planned_record_count"] == 3
    assert len(running["records"]) == 1
    assert running["records"][0]["configuration"] == "observed"

    resumed_seeds = []

    def resumed_executor(request):
        resumed_seeds.append(request.model_seed)
        return request.method_input.counts

    evidence = produce_orthogonal_outputs(
        output_directory,
        inputs=(OrthogonalInput("source-test", method_input),),
        configurations=(configuration,),
        model_seeds=(42, 43),
        artifact_bindings=ORTHOGONAL_ARTIFACT_BINDINGS,
        executor=resumed_executor,
    )
    completed = json.loads(evidence.manifest_path.read_bytes())
    assert completed["status"] == "completed"
    assert len(evidence.records) == 3
    assert resumed_seeds == [42, 43]


def test_schema2_writer_binds_evaluation_manifest_without_digest_cycle(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        OrthogonalConfiguration,
        OrthogonalInput,
        RealSourceEvidence,
        SourceArtifactBinding,
        SourceReceiptBinding,
        load_completed_reconstruction_checkpoint,
        produce_orthogonal_outputs,
        write_development_selection_artifacts,
    )
    from maskimpute_benchmark.methods import prepare_method_input
    from maskimpute_benchmark.protocol import canonical_sha256

    repository = tmp_path / "repository"
    repository.mkdir()
    plan, store, _, prepared = _completed_checkpoint(repository)
    reconstruction = load_completed_reconstruction_checkpoint(
        store.output_dir,
        plan,
        prepared_datasets={prepared.binding.dataset_id: prepared},
    )
    adata = ad.AnnData(
        X=np.asarray([[1, 0], [0, 1], [2, 1], [1, 2]]),
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(4)]),
        var=pd.DataFrame(index=["gene-1", "gene-2"]),
    )
    adata.uns["source_dataset_sha256"] = "f" * 64
    adata.uns["allowed_covariates"] = {"obs": [], "var": []}
    method_input = prepare_method_input(adata)
    config_payload = {"version": "v27"}
    orthogonal = produce_orthogonal_outputs(
        repository / "artifacts/study/development/evaluation/orthogonal",
        inputs=(OrthogonalInput("source-test", method_input),),
        configurations=(
            OrthogonalConfiguration(
                "v27-test", canonical_sha256(config_payload), config_payload
            ),
        ),
        model_seeds=(42,),
        artifact_bindings=ORTHOGONAL_ARTIFACT_BINDINGS,
        executor=lambda request: request.method_input.counts,
    )
    count_path = repository / "artifacts/study/development/count_scores/manifest.json"
    calibration_path = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    count_path.parent.mkdir(parents=True)
    calibration_path.parent.mkdir(parents=True)
    count_path.write_bytes(b"count-score\n")
    calibration_path.write_bytes(b"calibration\n")
    count_sha = hashlib.sha256(count_path.read_bytes()).hexdigest()
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    ledger_file = repository / "study/sources.json"
    ledger_file.parent.mkdir(parents=True)
    ledger_file.write_bytes(b"tracked-source-ledger\n")
    receipt_bindings = []
    artifact_bindings = []
    for source_id in (
        "baron-pancreas-umi",
        "cite-seq-cbmc-rna-protein",
        "tung-ipsc-ercc-bulk-replicates",
    ):
        receipt_file = repository / f"receipts/{source_id}.json"
        receipt_file.parent.mkdir(parents=True, exist_ok=True)
        receipt_file.write_bytes(f"receipt:{source_id}\n".encode())
        artifact_file = repository / f"data/{source_id}.dat"
        artifact_file.parent.mkdir(parents=True, exist_ok=True)
        artifact_file.write_bytes(f"artifact:{source_id}\n".encode())
        receipt_bindings.append(
            SourceReceiptBinding(
                source_id,
                f"receipts/{source_id}.json",
                hashlib.sha256(receipt_file.read_bytes()).hexdigest(),
            )
        )
        artifact_bindings.append(
            SourceArtifactBinding(
                source_id,
                f"data/{source_id}.dat",
                hashlib.sha256(artifact_file.read_bytes()).hexdigest(),
                artifact_file.stat().st_size,
            )
        )
    sources = RealSourceEvidence(
        ledger_path="study/sources.json",
        ledger_file_sha256=hashlib.sha256(ledger_file.read_bytes()).hexdigest(),
        ledger_sha256="2" * 64,
        receipts=tuple(receipt_bindings),
        artifacts=tuple(artifact_bindings),
    )
    records = [
        {
            "mechanism": "symsim",
            "biological_id": "draw-01",
            "technical_view": "moderate",
            "dataset_id": "dataset-test",
            "dataset_sha256": "a" * 64,
            "method": "observed",
            "method_sha256": "b" * 64,
            "model_seed": None,
            "metric": "mse",
            "value": 0.0,
            "status": "completed",
        }
    ]
    intervals = [
        {
            "configuration": "v27-test",
            "endpoint": "rna_protein_concordance",
            "comparison": "observed",
            "estimate": 0.0,
            "ci_lower": -0.1,
            "ci_upper": 0.1,
            "status": "completed",
        }
    ]
    result_path, evaluation_path = write_development_selection_artifacts(
        repository,
        dataset_manifest_sha256="a" * 64,
        count_score_manifest_sha256=count_sha,
        retained_calibration_artifact_sha256=calibration_sha,
        reconstruction=reconstruction,
        reconstruction_relative_directory="competition",
        orthogonal=orthogonal,
        orthogonal_relative_directory="artifacts/study/development/evaluation/orthogonal",
        records=records,
        intervals=intervals,
        null_de_audits=(),
        orthogonal_audits=(),
        sources=sources,
    )

    result = json.loads(result_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    assert set(result) == {
        "schema_version",
        "dataset_manifest_sha256",
        "count_score_manifest_sha256",
        "retained_calibration_artifact_sha256",
        "evaluation_manifest_sha256",
        "records",
        "orthogonal_intervals",
        "result_sha256",
    }
    assert (
        result["evaluation_manifest_sha256"]
        == hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    )
    assert evaluation["selection_evidence_sha256"] == canonical_sha256(
        {
            "schema_version": 2,
            "dataset_manifest_sha256": "a" * 64,
            "count_score_manifest_sha256": count_sha,
            "retained_calibration_artifact_sha256": calibration_sha,
            "records": records,
            "orthogonal_intervals": intervals,
        }
    )
    assert "result_sha256" not in evaluation
    assert result["result_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )


def test_cite_orthogonal_evaluator_averages_model_seeds_before_interval(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        CiteSeqSource,
        OrthogonalConfiguration,
        OrthogonalInput,
        evaluate_cite_orthogonal_interval,
        produce_orthogonal_outputs,
    )
    from maskimpute_benchmark.methods import prepare_method_input
    from maskimpute_benchmark.protocol import canonical_sha256

    counts = np.asarray(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9],
        ],
        dtype=np.int32,
    )
    proteins = counts.copy()
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(6)]),
        var=pd.DataFrame(index=["CD4", "CD8A", "CD14", "CD19"]),
    )
    adata.uns["source_dataset_sha256"] = "a" * 64
    adata.uns["allowed_covariates"] = {"obs": [], "var": []}
    method_input = prepare_method_input(adata)
    source = CiteSeqSource(
        cell_ids=method_input.obs_ids,
        gene_ids=method_input.var_ids,
        endpoint_gene_ids=("CD4", "CD8A", "CD14", "CD19"),
        rna_counts=counts,
        protein_ids=("CD4", "CD8", "CD14", "CD19"),
        protein_counts=proteins,
        rna_file_sha256="b" * 64,
        protein_file_sha256="c" * 64,
    )
    payload = {"version": "v27"}
    config = OrthogonalConfiguration("v27-test", canonical_sha256(payload), payload)
    evidence = produce_orthogonal_outputs(
        tmp_path / "orthogonal",
        inputs=(OrthogonalInput("cite-seq-cbmc-rna-protein", method_input),),
        configurations=(config,),
        model_seeds=(42, 43, 44),
        artifact_bindings=ORTHOGONAL_ARTIFACT_BINDINGS,
        executor=lambda request: request.method_input.counts,
    )

    interval = evaluate_cite_orthogonal_interval(
        evidence, source, "v27-test", n_boot=200
    )

    assert interval.status == "completed"
    assert interval.endpoint == "rna_protein_concordance"
    assert interval.estimate == pytest.approx(0.0)
    assert interval.n_biological_units == 1
    assert interval.n_technical_units == 4


def test_selection_input_cli_has_no_scientific_design_overrides() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_development_selection_input.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--reconstruction-dir" not in completed.stdout
    assert "--orthogonal-dir" not in completed.stdout
    assert "--seed" not in completed.stdout
    assert "--endpoint" not in completed.stdout
    assert "fixed" in completed.stdout.lower()
