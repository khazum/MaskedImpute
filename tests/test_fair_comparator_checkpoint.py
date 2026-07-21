from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from maskimpute_benchmark.comparator_tuning import (
    comparator_method_binding,
    load_comparator_tuning_authority,
)
from maskimpute_benchmark.direct_values import freeze_direct_mapping
import maskimpute_benchmark.fair_comparator_plan as direct_plan_module
from maskimpute_benchmark.fair_comparator_checkpoint import (
    DirectCheckpointStore,
    DirectDevelopmentBudget,
    budget_scope,
    direct_comparator_selection_status,
    replay_direct_development_budget,
)
from maskimpute_benchmark.fair_comparator_execution import (
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
from maskimpute_benchmark.methods import load_method_registry, prepare_method_input
from maskimpute_benchmark.runner import (
    DatasetBinding,
    DatasetQCAudit,
    PreparedDataset,
    RunnerContractError,
)


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IDENTITY_TOKENS = ("hash", "digest", "checksum", "fingerprint", "sha")


def _all_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for nested in value.values() for key in _all_keys(nested)
        )
    if isinstance(value, list):
        return tuple(key for nested in value for key in _all_keys(nested))
    return ()


def _contains_forbidden_identity_key(value: object) -> bool:
    return any(
        key.casefold() != "shape" and token in key.casefold()
        for key in _all_keys(value)
        for token in FORBIDDEN_IDENTITY_TOKENS
    )


def _prepared() -> PreparedDataset:
    counts = np.asarray([[2, 0, 1], [0, 3, 0]], dtype=np.int64)
    cells = ["cell-1", "cell-2"]
    genes = ["gene-1", "gene-2", "gene-3"]
    method_view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame(index=genes),
    )
    method_view.uns["source_dataset_sha256"] = "a" * 64
    method_view.uns["allowed_covariates"] = {"obs": [], "var": []}
    evaluator = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"draw": [1, 1]}, index=cells),
        var=pd.DataFrame(index=genes),
        layers={"pre_capture_counts": counts + 1},
    )
    evaluator.uns.update(
        {
            "truth_kind": "exact_pre_capture",
            "primary_truth_layer": "pre_capture_counts",
            "provenance": {"seeds": {"measurement": 20_001}},
        }
    )
    binding = DatasetBinding(
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        dataset_sha256="a" * 64,
        output_file_sha256="b" * 64,
        truth_sha256="c" * 64,
        output_path="dev/datasets/symsim/draw-01/moderate.h5ad",
        independent_unit_id="biological-test",
        cells=2,
        genes=3,
        manifest_sha256="d" * 64,
        protocol_sha256="e" * 64,
        design_sha256="f" * 64,
        seed_source_sha256="1" * 64,
    )
    return PreparedDataset(
        binding=binding,
        audit=DatasetQCAudit(
            excluded_cell_count=0,
            excluded_cell_ids_sha256="2" * 64,
            retained_cell_count=2,
            retained_cell_ids_sha256="3" * 64,
            excluded_cell_ids=(),
            retained_cell_ids=tuple(cells),
        ),
        method_input=prepare_method_input(method_view),
        evaluator_dataset=evaluator,
    )


def _direct_checkpoint_fixture(method_id: str = "magic"):
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    prepared = _prepared()
    descriptor = describe_prepared_input(prepared)
    spec = registry.by_id(method_id)
    method = comparator_method_binding(spec)
    configurations = tuple(
        DirectAuthorizedConfiguration(
            method=method,
            configuration_id=row.configuration_id,
            configuration_kind="comparator_tuning",
            payload=freeze_direct_mapping(row.payload),
            requires_count_score=False,
            requires_calibration=False,
        )
        for row in authority.configurations_for(method_id)[:3]
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
    return plan, registry, {prepared.binding.dataset_id: prepared}


def _replace_direct_entry_identity(
    plan: DirectCompetitionPlan,
    position: int = 0,
    **changes: object,
) -> DirectCompetitionPlan:
    identity = replace(plan.entries[position].identity, **changes)
    entry = replace(
        plan.entries[position],
        run_id=direct_run_id(identity),
        identity=identity,
    )
    entries = list(plan.entries)
    entries[position] = entry
    return replace(plan, entries=tuple(entries))


@pytest.mark.parametrize(
    "mutation",
    (
        "relabel-comparator-entry",
        "configuration-without-entry",
        "duplicate-configuration",
    ),
)
def test_direct_plan_binding_validator_rejects_missing_duplicate_or_relabelled_blocks(
    mutation: str,
) -> None:
    plan, registry, prepared_datasets = _direct_checkpoint_fixture()
    if mutation == "relabel-comparator-entry":
        changed = _replace_direct_entry_identity(
            plan,
            configuration_kind="candidate_search",
        )
    elif mutation == "configuration-without-entry":
        changed = replace(plan, entries=plan.entries[:-1])
    else:
        changed = replace(
            plan,
            configurations=(*plan.configurations, plan.configurations[0]),
        )

    with pytest.raises(RunnerContractError):
        direct_plan_module.validate_direct_competition_plan(
            changed,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "schema",
        "mode",
        "revision",
        "workflow",
        "flags",
        "preflight",
        "input",
    ),
)
def test_direct_plan_binding_validator_rejects_contract_and_input_drift(
    mutation: str,
) -> None:
    plan, registry, prepared_datasets = _direct_checkpoint_fixture()
    if mutation == "schema":
        changed = replace(plan, schema_version=True)
    elif mutation == "mode":
        changed = replace(plan, identity_mode="legacy-v1")
    elif mutation == "revision":
        changed = replace(plan, authority_revision="fair-comparator-direct-v2")
    elif mutation == "workflow":
        changed = _replace_direct_entry_identity(plan, workflow_schema="forged-v1")
    elif mutation == "flags":
        changed_entry = replace(plan.entries[0], requires_count_score=True)
        changed = replace(plan, entries=(changed_entry, *plan.entries[1:]))
    elif mutation == "preflight":
        changed_entry = replace(plan.entries[0], preflight_reason="caller-claim")
        changed = replace(plan, entries=(changed_entry, *plan.entries[1:]))
    else:
        changed_descriptor = replace(plan.inputs[0], mechanism="sergio")
        changed = replace(plan, inputs=(changed_descriptor,))

    with pytest.raises(RunnerContractError):
        direct_plan_module.validate_direct_competition_plan(
            changed,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )


@pytest.mark.parametrize("replacement", (True, 5.0))
def test_direct_plan_binding_validator_rejects_numeric_payload_type_coercion(
    replacement: object,
) -> None:
    plan, registry, prepared_datasets = _direct_checkpoint_fixture()
    configuration = plan.configurations[0]
    payload = tuple(
        (name, replacement if name == "knn" else value)
        for name, value in configuration.payload
    )
    changed_configuration = replace(configuration, payload=payload)
    changed = _replace_direct_entry_identity(plan, configuration_payload=payload)
    changed = replace(
        changed,
        configurations=(changed_configuration, *plan.configurations[1:]),
    )

    with pytest.raises(RunnerContractError):
        direct_plan_module.validate_direct_competition_plan(
            changed,
            registry=registry,
            prepared_datasets=prepared_datasets,
        )


def _attempt(
    entry: DirectPlanEntry,
    *,
    status: str = "unavailable",
    runtime_seconds: float = 1.0,
) -> DirectEvaluatedAttempt:
    reason = None if status == "completed" else f"synthetic_{status}"
    run = DirectRunResult(
        run_id=entry.run_id,
        identity=entry.identity,
        status=status,
        reason=reason,
        runtime_seconds=runtime_seconds,
        peak_rss_bytes=1,
        peak_gpu_bytes=0,
        rss_measurement="synthetic_parent_rss",
        gpu_measurement="not_applicable_cpu",
        excluded_cell_count=0,
        excluded_cell_ids=(),
        retained_cell_count=2,
        retained_cell_ids=("cell-1", "cell-2"),
        retained_gene_count=3,
        observed_zero_count=3,
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
    )
    metrics = tuple(
        DirectMetricRow(
            identity=entry.identity,
            metric=metric,
            value=None if reason is not None else 0.0,
            n=0 if reason is not None else 1,
            status=status,
            reason=reason,
        )
        for metric in (
            "mse",
            "mse_dropout",
            "gnrmse",
            "mse_pre_dropout_zero",
            "corr_err",
            "mse_non_dropout_nonzero",
        )
    )
    return DirectEvaluatedAttempt(
        run=run,
        metrics=metrics,
        native_output=(
            np.zeros((2, 3), dtype=np.float64) if status == "completed" else None
        ),
        native_output_scale=(
            entry.identity.method.output_scale if status == "completed" else None
        ),
        evaluator_output=(
            np.zeros((2, 3), dtype=np.float64) if status == "completed" else None
        ),
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


@pytest.mark.parametrize(
    "mutation",
    ("empty", "duplicate", "renamed", "reordered"),
)
def test_direct_attempt_and_checkpoint_require_exact_ordered_metric_denominator(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    attempt = _attempt(plan.entries[0])
    metrics = list(attempt.metrics)
    if mutation == "empty":
        metrics = []
    elif mutation == "duplicate":
        metrics[-1] = metrics[0]
    elif mutation == "renamed":
        metrics[0] = replace(metrics[0], metric="renamed")
    else:
        metrics[0], metrics[1] = metrics[1], metrics[0]

    with pytest.raises(RunnerContractError, match="metric|denominator|order"):
        replace(attempt, metrics=tuple(metrics))
    record = attempt.to_dict()
    if mutation == "empty":
        record["metrics"] = []
    elif mutation == "duplicate":
        record["metrics"][-1] = record["metrics"][0]
    elif mutation == "renamed":
        record["metrics"][0]["metric"] = "renamed"
    else:
        record["metrics"][0], record["metrics"][1] = (
            record["metrics"][1],
            record["metrics"][0],
        )
    with pytest.raises(RunnerContractError, match="metric|denominator|order"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


@pytest.mark.parametrize(
    "mutation",
    ("metric-identity", "metric-status", "metric-reason"),
)
def test_direct_attempt_rejects_run_metric_semantic_drift(mutation: str) -> None:
    plan, _registry, _prepared = _direct_checkpoint_fixture()
    attempt = _attempt(plan.entries[0])
    metrics = list(attempt.metrics)
    if mutation == "metric-identity":
        metrics[0] = replace(
            metrics[0],
            identity=replace(metrics[0].identity, ordinal=99),
        )
    elif mutation == "metric-status":
        metrics[0] = replace(metrics[0], status="failed", reason="different_failure")
    else:
        metrics[0] = replace(metrics[0], reason="different_failure")

    with pytest.raises(RunnerContractError, match="metric|identity|status|reason"):
        replace(attempt, metrics=tuple(metrics))


@pytest.mark.parametrize(
    "mutation",
    ("cell-audit", "gene-count", "observed-zero-count"),
)
def test_direct_checkpoint_rebinds_qc_fields_to_prepared_input(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    record = _attempt(plan.entries[0]).to_dict()
    run = record["run"]
    if mutation == "cell-audit":
        run.update(
            excluded_cell_count=1,
            excluded_cell_ids=["cell-1"],
            retained_cell_count=1,
            retained_cell_ids=["cell-2"],
        )
    elif mutation == "gene-count":
        run["retained_gene_count"] = 2
    else:
        run["observed_zero_count"] = 2

    with pytest.raises(RunnerContractError, match="audit|prepared|gene|zero"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


@pytest.mark.parametrize("mutation", ("reason", "applicability"))
def test_direct_attempt_and_checkpoint_bind_non_score_prezero_receipt(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    attempt = _attempt(plan.entries[0])
    if mutation == "reason":
        evidence = replace(
            attempt.p_pre_zero_evidence,
            reason="coherently_changed_non_applicable_reason",
        )
    else:
        evidence = DirectPreZeroEvidence(
            applicable=True,
            status=attempt.run.status,
            reason=attempt.run.reason,
            shape=None,
            dtype=None,
            encoding=None,
            path=None,
            compressed_byte_count=0,
        )

    with pytest.raises(RunnerContractError, match="p_pre_zero|applicability|receipt"):
        replace(attempt, p_pre_zero_evidence=evidence)
    record = attempt.to_dict()
    record["p_pre_zero_evidence"] = evidence.to_dict()
    with pytest.raises(RunnerContractError, match="p_pre_zero|applicability|receipt"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


def test_direct_checkpoint_load_reopens_applicable_prezero_receipt(
    tmp_path: Path,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        (_attempt(plan.entries[0], status="completed").to_dict(),),
        registry=registry,
        prepared_datasets=prepared,
    )

    def mutate(payload):
        payload["records"][0]["p_pre_zero_evidence"] = {
            "applicable": True,
            "status": "completed",
            "reason": None,
            "shape": [2, 3],
            "dtype": "<f8",
            "encoding": "zlib",
            "path": "missing-prezero.zlib",
            "compressed_byte_count": 1,
        }

    _rewrite(store.path, mutate)
    with pytest.raises(RunnerContractError, match="p_pre_zero|unavailable|path"):
        store.load(plan, registry=registry, prepared_datasets=prepared)


def _terminal_prefix(
    plan: DirectCompetitionPlan,
    count: int,
    *,
    status: str = "unavailable",
) -> tuple[dict[str, object], ...]:
    return tuple(
        _attempt(entry, status=status).to_dict() for entry in plan.entries[:count]
    )


def _rewrite(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(
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


def _collision_plan() -> tuple[
    DirectCompetitionPlan,
    object,
    dict[str, PreparedDataset],
]:
    plan, registry, prepared = _direct_checkpoint_fixture()
    payload = direct_plan_module._freeze_payload_mapping(  # noqa: SLF001
        {"nested": [["a", 1]]}
    )
    configuration = replace(
        plan.configurations[0],
        configuration_id="synthetic-nested",
        configuration_kind="candidate_search",
        payload=payload,
    )
    identity = replace(
        plan.entries[0].identity,
        configuration_id=configuration.configuration_id,
        configuration_kind=configuration.configuration_kind,
        configuration_payload=payload,
    )
    entry = replace(
        plan.entries[0],
        run_id=direct_run_id(identity),
        identity=identity,
    )
    return (
        replace(
            plan,
            entries=(entry, *plan.entries[1:]),
            configurations=(configuration, *plan.configurations[1:]),
        ),
        registry,
        prepared,
    )


def _registry_with_method(registry, method_id: str, replacement):
    return replace(
        registry,
        methods=tuple(
            replacement if method.id == method_id else method
            for method in registry.methods
        ),
    )


def test_direct_checkpoint_replays_exact_prefix_and_budget(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    report = store.write(
        plan,
        _terminal_prefix(plan, 3),
        registry=registry,
        prepared_datasets=prepared,
    )

    assert report.plan_snapshot == plan.to_dict()
    assert (
        report.budget
        == replay_direct_development_budget(
            registry, plan.entries, report.records
        ).to_dict()
    )
    assert not _contains_forbidden_identity_key(report.to_dict())
    assert set(json.loads(store.path.read_text())) == {
        "schema_version",
        "identity_mode",
        "authority_revision",
        "plan_snapshot",
        "input_descriptors",
        "planned_run_count",
        "status",
        "evaluation_scope",
        "comparator_selection_status",
        "selection_complete",
        "selection_blockers",
        "records",
        "budget",
    }


@pytest.mark.parametrize("mutation", ("plan_order", "payload", "method"))
def test_direct_checkpoint_rejects_complete_plan_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    if mutation == "plan_order":
        changed = replace(
            plan, entries=(plan.entries[1], plan.entries[0], *plan.entries[2:])
        )
    elif mutation == "payload":
        identity = replace(
            plan.entries[0].identity,
            configuration_payload=(("knn", 999),),
        )
        changed = replace(
            plan,
            entries=(replace(plan.entries[0], identity=identity), *plan.entries[1:]),
        )
    else:
        method = replace(plan.entries[0].identity.method, adapter_key="wrong")
        identity = replace(plan.entries[0].identity, method=method)
        changed = replace(
            plan,
            entries=(replace(plan.entries[0], identity=identity), *plan.entries[1:]),
        )

    with pytest.raises(
        RunnerContractError,
        match="plan snapshot|ordinals|configuration|method projection",
    ):
        store.load(changed, registry=registry, prepared_datasets=prepared)


def test_direct_checkpoint_rejects_input_descriptor_drift(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    _rewrite(
        store.path,
        lambda payload: payload["input_descriptors"][0].__setitem__(
            "gene_ids", ["changed"]
        ),
    )

    with pytest.raises(RunnerContractError, match="input descriptors"):
        store.load(plan, registry=registry, prepared_datasets=prepared)


@pytest.mark.parametrize("mutation", ("extra", "skipped", "configuration_id"))
def test_direct_checkpoint_rejects_nonprefix_records(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 2),
        registry=registry,
        prepared_datasets=prepared,
    )

    def mutate(payload):
        records = payload["records"]
        if mutation == "extra":
            records.extend([records[-1], records[-1]])
        elif mutation == "skipped":
            payload["records"] = [records[1]]
        else:
            records[0]["run"]["identity"]["configuration_id"] = "wrong"

    _rewrite(store.path, mutate)
    with pytest.raises(RunnerContractError, match="prefix|identity"):
        store.load(plan, registry=registry, prepared_datasets=prepared)


def test_direct_checkpoint_rejects_budget_drift(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    _rewrite(
        store.path,
        lambda payload: payload["budget"]["magic"].__setitem__("consumed_seconds", 99),
    )
    with pytest.raises(RunnerContractError, match="budget ledger differs"):
        store.load(plan, registry=registry, prepared_datasets=prepared)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("status", "completed"),
        ("comparator_selection_status", "complete_terminal_denominator"),
        ("selection_complete", True),
        ("selection_blockers", []),
    ),
)
def test_direct_checkpoint_rejects_caller_supplied_completeness(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    _rewrite(store.path, lambda payload: payload.__setitem__(field, value))

    with pytest.raises(RunnerContractError, match="status|completeness|selection"):
        store.load(plan, registry=registry, prepared_datasets=prepared)

    assert "selection_complete" not in inspect.signature(store.write).parameters
    assert "budget" not in inspect.signature(store.write).parameters


def test_direct_budget_uses_configuration_ids_and_task7_scopes() -> None:
    plan, registry, _prepared = _direct_checkpoint_fixture()
    spec = registry.by_id("magic")
    budget = DirectDevelopmentBudget()
    for value in range(20):
        configuration_id = f"magic-{value:02d}"
        assert budget.authorize(spec, configuration_id).authorized
        budget.record(spec, configuration_id, _attempt(plan.entries[0]).run)
    assert not budget.authorize(spec, "magic-excess").authorized
    assert budget_scope(plan.entries[0]) == "magic"
    assert budget.to_dict()["magic"]["configuration_ids"] == [
        f"magic-{value:02d}" for value in range(20)
    ]

    maskimpute_method = comparator_method_binding(registry.by_id("maskimpute"))
    candidate_identity = replace(
        plan.entries[0].identity,
        method=maskimpute_method,
        configuration_kind="candidate_search",
    )
    ablation_identity = replace(
        candidate_identity,
        configuration_kind="ablation",
    )
    assert budget_scope(replace(plan.entries[0], identity=candidate_identity)) == (
        "maskimpute:candidate_search"
    )
    assert budget_scope(replace(plan.entries[0], identity=ablation_identity)) == (
        "maskimpute:ablation"
    )


@pytest.mark.parametrize(
    ("method_id", "limit"),
    (("magic", 86_400.0), ("dca", 28_800.0)),
)
def test_direct_budget_restore_and_record_enforce_exact_time_ceiling(
    method_id: str,
    limit: float,
) -> None:
    plan, registry, _prepared = _direct_checkpoint_fixture(method_id)
    spec = registry.by_id(method_id)
    entry = plan.entries[0]

    exact_restore = DirectDevelopmentBudget()
    exact_restore.restore(spec, entry.identity.configuration_id, "completed", limit)
    assert exact_restore.to_dict()[method_id]["consumed_seconds"] == limit
    exact_restore.restore(
        spec,
        entry.identity.configuration_id,
        "infrastructure_error",
        limit + 1.0,
    )
    assert exact_restore.to_dict()[method_id]["consumed_seconds"] == limit

    with pytest.raises(RunnerContractError, match="time|budget|ceiling"):
        DirectDevelopmentBudget().restore(
            spec,
            entry.identity.configuration_id,
            "completed",
            limit + 1.0,
        )
    with pytest.raises(RunnerContractError, match="time|budget|ceiling"):
        DirectDevelopmentBudget().record(
            spec,
            entry.identity.configuration_id,
            _attempt(entry, runtime_seconds=limit + 1.0),
        )


@pytest.mark.parametrize(
    ("method_id", "limit"),
    (("magic", 86_400.0), ("dca", 28_800.0)),
)
def test_direct_checkpoint_replay_rejects_coherent_time_ledger_over_ceiling(
    tmp_path: Path,
    method_id: str,
    limit: float,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture(method_id)
    record = _attempt(plan.entries[0], runtime_seconds=limit + 1.0).to_dict()

    with pytest.raises(RunnerContractError, match="time|budget|ceiling"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("completed", "complete_terminal_denominator"),
        ("failed", "complete_terminal_denominator"),
        ("timeout", "complete_terminal_denominator"),
        ("resource_exceeded", "complete_terminal_denominator"),
        ("unavailable", "complete_terminal_denominator"),
        ("budget_exhausted", "blocked_incomplete_denominator"),
        ("blocked_authority", "blocked_incomplete_denominator"),
        ("infrastructure_error", "blocked_incomplete_denominator"),
    ),
)
def test_direct_comparator_selection_status_preserves_task7_partition(
    status: str,
    expected: str,
) -> None:
    plan, _registry, _prepared = _direct_checkpoint_fixture()
    records = _terminal_prefix(plan, len(plan.entries), status=status)
    assert direct_comparator_selection_status(plan.entries, records) == expected


def test_direct_checkpoint_recovers_only_exact_next_transaction(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    report = store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    attempt = _attempt(plan.entries[1], status="infrastructure_error")
    store._publish_transaction_intent(plan, 1, plan.entries[1], attempt)

    recovered = store.load(plan, registry=registry, prepared_datasets=prepared)
    assert len(recovered.records) == 2
    assert recovered.records[1]["run"]["status"] == "infrastructure_error"
    assert not store.intent_path.exists()

    assert store.load(plan, registry=registry, prepared_datasets=prepared) == recovered
    with pytest.raises(RunnerContractError, match="identity"):
        store.append(
            plan,
            recovered,
            attempt,
            registry=registry,
            prepared_datasets=prepared,
        )
    assert report.records[0] == recovered.records[0]


def test_direct_checkpoint_append_derives_record_budget_and_completeness(
    tmp_path: Path,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    running = store.write(
        plan,
        (),
        registry=registry,
        prepared_datasets=prepared,
    )

    appended = store.append(
        plan,
        running,
        _attempt(plan.entries[0]),
        registry=registry,
        prepared_datasets=prepared,
    )

    assert len(appended.records) == 1
    assert appended.status == "running"
    assert appended.comparator_selection_status == "blocked_incomplete_denominator"
    assert (
        appended.budget
        == replay_direct_development_budget(
            registry,
            plan.entries,
            appended.records,
        ).to_dict()
    )
    assert not store.intent_path.exists()


def test_direct_checkpoint_rejects_transaction_that_skips_next_ordinal(
    tmp_path: Path,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    store._publish_transaction_intent(
        plan,
        2,
        plan.entries[2],
        _attempt(plan.entries[2]),
    )
    with pytest.raises(RunnerContractError, match="next|position"):
        store.load(plan, registry=registry, prepared_datasets=prepared)


def test_direct_checkpoint_write_rejects_symlink_replacement(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    target = tmp_path / "outside.json"
    target.write_text("outside\n", encoding="utf-8")
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.path.symlink_to(target)

    with pytest.raises(RunnerContractError, match="owned regular file|symlink"):
        store.write(
            plan,
            (),
            registry=registry,
            prepared_datasets=prepared,
        )

    assert store.path.is_symlink()
    assert target.read_text(encoding="utf-8") == "outside\n"


def test_direct_checkpoint_rejects_inconsistent_metric_record(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    records = list(_terminal_prefix(plan, 1))
    records[0]["metrics"][0]["reason"] = None

    with pytest.raises(RunnerContractError, match="metric"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            records,
            registry=registry,
            prepared_datasets=prepared,
        )


def test_direct_checkpoint_rejects_noncontiguous_plan_ordinals(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    identity = replace(plan.entries[1].identity, ordinal=99)
    changed = replace(
        plan,
        entries=(
            plan.entries[0],
            replace(plan.entries[1], identity=identity),
            *plan.entries[2:],
        ),
    )

    with pytest.raises(RunnerContractError, match="ordinal"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            changed,
            (),
            registry=registry,
            prepared_datasets=prepared,
        )


@pytest.mark.parametrize("identity_owner", ("run", "metric"))
def test_review_direct_record_identity_preserves_nested_list_of_pairs(
    identity_owner: str,
) -> None:
    plan, _registry, _prepared = _collision_plan()
    record = _attempt(plan.entries[0]).to_dict()
    expected = plan.to_dict()["entries"][0]["identity"]
    observed = (
        record["run"]["identity"]
        if identity_owner == "run"
        else record["metrics"][0]["identity"]
    )

    assert observed == expected
    assert observed["configuration_payload"]["nested"] == [["a", 1]]


def test_review_direct_checkpoint_rejects_list_object_identity_collision(
    tmp_path: Path,
) -> None:
    plan, registry, prepared = _collision_plan()
    record = _attempt(plan.entries[0]).to_dict()
    for identity in (
        record["run"]["identity"],
        record["metrics"][0]["identity"],
    ):
        identity["configuration_payload"]["nested"] = {"a": 1}

    with pytest.raises(RunnerContractError, match="identity"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


@pytest.mark.parametrize("field", ("schema_version", "planned_run_count"))
def test_review_direct_checkpoint_rejects_bool_report_integer(
    tmp_path: Path,
    field: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    _rewrite(store.path, lambda payload: payload.__setitem__(field, True))

    with pytest.raises(RunnerContractError, match="schema|denominator|integer"):
        store.load(plan, registry=registry, prepared_datasets=prepared)


def test_review_direct_checkpoint_rejects_bool_intent_schema_version(
    tmp_path: Path,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 1),
        registry=registry,
        prepared_datasets=prepared,
    )
    store._publish_transaction_intent(  # noqa: SLF001
        plan,
        1,
        plan.entries[1],
        _attempt(plan.entries[1]),
    )
    _rewrite(
        store.intent_path,
        lambda payload: payload.__setitem__("schema_version", True),
    )

    with pytest.raises(RunnerContractError, match="schema|plan snapshot"):
        store.load(plan, registry=registry, prepared_datasets=prepared)


def test_review_direct_checkpoint_rejects_signed_negative_zero_runtime(
    tmp_path: Path,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    record = _terminal_prefix(plan, 1)[0]
    record["run"]["runtime_seconds"] = -0.0

    with pytest.raises(RunnerContractError, match="runtime|negative"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


def test_review_direct_budget_rejects_signed_negative_zero() -> None:
    plan, registry, _prepared = _direct_checkpoint_fixture()

    with pytest.raises(RunnerContractError, match="runtime|negative"):
        DirectDevelopmentBudget().restore(
            registry.by_id("magic"),
            plan.entries[0].identity.configuration_id,
            "completed",
            -0.0,
        )


def test_direct_checkpoint_rejects_signed_negative_zero_metric(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    record = _attempt(plan.entries[0], status="completed").to_dict()
    record["metrics"][0]["value"] = -0.0

    with pytest.raises(RunnerContractError, match="metric value"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


def test_direct_checkpoint_accepts_nonzero_negative_metric(tmp_path: Path) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    record = _attempt(plan.entries[0], status="completed").to_dict()
    record["metrics"][0]["value"] = -0.25

    report = DirectCheckpointStore(tmp_path / "checkpoint.json").write(
        plan,
        (record,),
        registry=registry,
        prepared_datasets=prepared,
    )

    assert report.records[0]["metrics"][0]["value"] == -0.25


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.__setitem__("applicable", 1), "applicability"),
        (
            lambda value: value.__setitem__("shape", {"legacy_sha256": "a" * 64}),
            "shape|evidence",
        ),
        (lambda value: value.__setitem__("status", ["not_applicable"]), "status"),
        (
            lambda value: value.update(
                {
                    "applicable": True,
                    "status": "completed",
                    "reason": None,
                    "shape": [2, 3],
                    "dtype": "<f8",
                    "encoding": "zlib",
                    "path": "../outside.zlib",
                    "compressed_byte_count": 1,
                }
            ),
            "path|evidence",
        ),
    ),
)
def test_review_direct_checkpoint_rejects_malformed_prezero_evidence(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    record = _terminal_prefix(plan, 1)[0]
    mutation(record["p_pre_zero_evidence"])

    with pytest.raises(RunnerContractError, match=message):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            (record,),
            registry=registry,
            prepared_datasets=prepared,
        )


@pytest.mark.parametrize(
    "drift",
    (
        "gpu_class",
        "timeout",
        "source_cache",
        "scale",
        "seed_policy",
        "resource_limit",
    ),
)
def test_review_direct_checkpoint_rejects_registry_method_projection_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    spec = registry.by_id("magic")
    if drift == "gpu_class":
        changed = replace(
            spec,
            resources=replace(
                spec.resources, gpu_required=not spec.resources.gpu_required
            ),
        )
    elif drift == "timeout":
        changed = replace(
            spec,
            resources=replace(
                spec.resources,
                timeout_seconds=spec.resources.timeout_seconds + 1,
            ),
        )
    elif drift == "source_cache":
        changed = replace(spec, source=replace(spec.source, cache_path="wrong/cache"))
    elif drift == "scale":
        changed = replace(spec, output_scale=f"{spec.output_scale}-drift")
    elif drift == "seed_policy":
        changed = replace(spec, seed_policy=f"{spec.seed_policy}-drift")
    else:
        changed = replace(
            spec,
            resources=replace(
                spec.resources,
                max_rss_gib=spec.resources.max_rss_gib + 1,
            ),
        )
    changed_registry = _registry_with_method(registry, "magic", changed)

    with pytest.raises(RunnerContractError, match="method projection|registry"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            _terminal_prefix(plan, 1),
            registry=changed_registry,
            prepared_datasets=prepared,
        )


@pytest.mark.parametrize("registry_change", ("missing", "duplicate"))
def test_review_direct_checkpoint_rejects_missing_or_duplicate_registry_method(
    tmp_path: Path,
    registry_change: str,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    if registry_change == "missing":
        methods = tuple(method for method in registry.methods if method.id != "magic")
    else:
        methods = (*registry.methods, registry.by_id("magic"))
    changed_registry = replace(registry, methods=methods)

    with pytest.raises(RunnerContractError, match="method|registry"):
        DirectCheckpointStore(tmp_path / "checkpoint.json").write(
            plan,
            _terminal_prefix(plan, 1),
            registry=changed_registry,
            prepared_datasets=prepared,
        )


def test_review_direct_checkpoint_rejects_stale_historical_transaction(
    tmp_path: Path,
) -> None:
    plan, registry, prepared = _direct_checkpoint_fixture()
    store = DirectCheckpointStore(tmp_path / "checkpoint.json")
    store.write(
        plan,
        _terminal_prefix(plan, 3),
        registry=registry,
        prepared_datasets=prepared,
    )
    store._publish_transaction_intent(  # noqa: SLF001
        plan,
        0,
        plan.entries[0],
        _attempt(plan.entries[0]),
    )

    with pytest.raises(RunnerContractError, match="stale|position"):
        store.load(plan, registry=registry, prepared_datasets=prepared)
