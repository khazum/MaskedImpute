from __future__ import annotations

from dataclasses import asdict, replace
import inspect
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.fair_comparator_checkpoint as direct_checkpoint_module
import maskimpute_benchmark.fair_comparator_plan as direct_plan_module
from maskimpute_benchmark.comparator_tuning import (
    comparator_method_binding,
    load_comparator_tuning_authority,
)
from maskimpute_benchmark.fair_comparator_checkpoint import DirectCheckpointStore
from maskimpute_benchmark.fair_comparator_execution import (
    DIRECT_RECONSTRUCTION_METRICS,
    create_direct_request,
)
from maskimpute_benchmark.fair_comparator_plan import (
    DirectCompetitionPlan,
    _build_structural_direct_competition_plan,
    _validate_direct_competition_plan_structure,
    build_direct_competition_plan,
    describe_prepared_input,
    direct_run_id,
    validate_direct_competition_plan,
)
from maskimpute_benchmark.methods import load_method_registry, prepare_method_input
from maskimpute_benchmark.runner import (
    DatasetQCAudit,
    PreparedDataset,
    RunnerContractError,
    build_fair_comparator_plan,
    load_runner_authority,
    load_v28_revision_authority,
    load_v29_revision_authority,
    validate_development_manifest_payload,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS_PATH = ROOT / "study/methods.json"
MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
VIEWS = ("moderate", "severe")
FORBIDDEN_IDENTITY_TOKENS = ("hash", "digest", "checksum", "fingerprint", "sha")


def _allow_unbound_smoke_only_for_unrelated_structural_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_validate = direct_plan_module.validate_direct_competition_plan

    def validate_without_smoke(plan, **kwargs):
        kwargs["_require_smoke_receipt"] = False
        return real_validate(plan, **kwargs)

    monkeypatch.setattr(
        direct_plan_module,
        "validate_direct_competition_plan",
        validate_without_smoke,
    )
    monkeypatch.setattr(
        direct_checkpoint_module,
        "validate_direct_competition_plan",
        validate_without_smoke,
    )


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
        "manifest_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "design_sha256": "c" * 64,
        "seed_source_sha256": "d" * 64,
        "rows": rows,
    }


def _prepared_dataset(
    binding: object,
    ordinal: int,
    *,
    include_batch: bool = False,
) -> PreparedDataset:
    counts = np.asarray([[ordinal, 0, 1], [0, ordinal + 1, 0]], dtype=np.int64)
    cell_ids = [f"cell-{ordinal}-1", f"cell-{ordinal}-2"]
    gene_ids = ["gene-1", "gene-2", "gene-3"]
    obs = pd.DataFrame(index=cell_ids)
    allowed_obs: list[str] = []
    if include_batch:
        obs["batch"] = pd.Categorical(["batch-0", "batch-1"])
        allowed_obs.append("batch")
    view = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=gene_ids),
    )
    view.uns["source_dataset_sha256"] = binding.dataset_sha256
    view.uns["allowed_covariates"] = {"obs": allowed_obs, "var": []}
    method_input = prepare_method_input(view)
    draw_index = int(binding.biological_id.removeprefix("draw-"))
    evaluator_dataset = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"draw": [draw_index, draw_index]},
            index=cell_ids,
        ),
        var=pd.DataFrame(index=gene_ids),
    )
    evaluator_dataset.uns["provenance"] = {
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
        method_input=method_input,
        evaluator_dataset=evaluator_dataset,
    )


def _direct_fixture() -> tuple[
    DirectCompetitionPlan, object, tuple[object, ...], tuple[PreparedDataset, ...]
]:
    registry = load_method_registry(METHODS_PATH)
    datasets = validate_development_manifest_payload(_manifest_payload())
    prepared = tuple(
        _prepared_dataset(binding, ordinal, include_batch=ordinal == 1)
        for ordinal, binding in enumerate(datasets, start=1)
    )
    plan = _build_structural_direct_competition_plan(
        registry,
        datasets,
        load_runner_authority(),
        prepared,
    )
    return plan, registry, datasets, prepared


def _renumber_direct_entries(entries: tuple[object, ...]) -> tuple[object, ...]:
    result = []
    for ordinal, entry in enumerate(entries, start=1):
        identity = replace(entry.identity, ordinal=ordinal)
        result.append(replace(entry, run_id=direct_run_id(identity), identity=identity))
    return tuple(result)


def _truncate_direct_plan(
    plan: DirectCompetitionPlan,
    prepared: tuple[PreparedDataset, ...],
) -> tuple[DirectCompetitionPlan, dict[str, PreparedDataset]]:
    """Build one coherent non-production structural fixture."""

    removed_id = plan.inputs[-1].dataset_id
    changed = replace(
        plan,
        inputs=plan.inputs[:-1],
        entries=_renumber_direct_entries(
            tuple(
                entry
                for entry in plan.entries
                if entry.identity.dataset_id != removed_id
            )
        ),
    )
    prepared_by_id = {
        value.binding.dataset_id: value
        for value in prepared
        if value.binding.dataset_id != removed_id
    }
    assert len(changed.inputs) == 15
    assert len(changed.configurations) == 61
    assert len(changed.entries) == 2_715
    return changed, prepared_by_id


def _configuration_positions(plan, configuration) -> list[int]:
    return [
        position
        for position, entry in enumerate(plan.entries)
        if entry.identity.method.method_id == configuration.method.method_id
        and entry.identity.configuration_id == configuration.configuration_id
    ]


def _replace_entry_cell(entry, source, *, model_seed: int | None = None):
    source_identity = source.identity
    identity = replace(
        entry.identity,
        dataset_id=source_identity.dataset_id,
        mechanism=source_identity.mechanism,
        biological_id=source_identity.biological_id,
        technical_view=source_identity.technical_view,
        mask_seed=source_identity.mask_seed,
        model_seed=(source_identity.model_seed if model_seed is None else model_seed),
        draw_index=source_identity.draw_index,
    )
    return replace(entry, identity=identity)


@pytest.fixture(scope="module")
def _direct_grid_cases():
    base, registry, datasets, prepared = _direct_fixture()
    prepared_map = {value.binding.dataset_id: value for value in prepared}
    revision = _build_structural_direct_competition_plan(
        registry,
        datasets,
        load_v28_revision_authority(),
        prepared,
    )
    comparator = next(
        value
        for value in base.configurations
        if value.configuration_kind == "comparator_tuning"
    )
    ablation = next(
        value for value in base.configurations if value.configuration_kind == "ablation"
    )
    comparator_positions = _configuration_positions(base, comparator)
    synthetic_entries = tuple(
        base.entries[position] for position in comparator_positions[:6]
    )
    synthetic = replace(
        base,
        inputs=base.inputs[:2],
        configurations=(comparator,),
        entries=_renumber_direct_entries(synthetic_entries),
    )
    return (
        {
            "base": (base, comparator, prepared_map),
            "revision": (revision, revision.configurations[0], prepared_map),
            "ablation": (base, ablation, prepared_map),
            "synthetic": (
                synthetic,
                comparator,
                {
                    descriptor.dataset_id: prepared_map[descriptor.dataset_id]
                    for descriptor in synthetic.inputs
                },
            ),
        },
        registry,
        datasets,
    )


def _mutate_configuration_cells(plan, configuration, mutation: str):
    entries = list(plan.entries)
    positions = _configuration_positions(plan, configuration)
    block = [entries[position] for position in positions]
    if mutation == "reordered":
        block[0], block[1] = block[1], block[0]
    elif mutation == "missing":
        removed = block.pop(1)
        if len(plan.inputs) == 16:
            block.append(_replace_entry_cell(removed, removed, model_seed=999))
    elif mutation == "duplicated":
        block[1] = _replace_entry_cell(block[1], block[0])
    else:
        block[0] = _replace_entry_cell(block[0], block[0], model_seed=999)
    entries[positions[0] : positions[-1] + 1] = block
    return replace(plan, entries=_renumber_direct_entries(tuple(entries)))


def test_direct_base_rejects_coherent_47_49_comparator_redistribution(
    _direct_grid_cases,
) -> None:
    cases, registry, datasets = _direct_grid_cases
    plan, first, prepared = cases["base"]
    comparator_values = tuple(
        value
        for value in plan.configurations
        if value.configuration_kind == "comparator_tuning"
    )
    second = comparator_values[1]
    first_positions = _configuration_positions(plan, first)
    second_positions = _configuration_positions(plan, second)
    entries = list(plan.entries)
    entries.pop(first_positions[-1])
    shifted_second_end = second_positions[-1] - 1
    extra = _replace_entry_cell(
        entries[shifted_second_end],
        entries[shifted_second_end],
        model_seed=999,
    )
    entries.insert(shifted_second_end + 1, extra)
    changed = replace(plan, entries=_renumber_direct_entries(tuple(entries)))

    with pytest.raises(RunnerContractError, match="grid|seed|cell"):
        validate_direct_competition_plan(
            changed,
            registry=registry,
            prepared_datasets=prepared,
            authority=load_runner_authority(),
            datasets=datasets,
        )


@pytest.mark.parametrize("scope", ("base", "revision", "ablation", "synthetic"))
@pytest.mark.parametrize(
    "mutation",
    ("reordered", "missing", "duplicated", "substituted"),
)
def test_direct_plan_rejects_noncanonical_configuration_cells(
    _direct_grid_cases,
    scope: str,
    mutation: str,
) -> None:
    cases, registry, datasets = _direct_grid_cases
    plan, configuration, prepared = cases[scope]
    changed = _mutate_configuration_cells(plan, configuration, mutation)

    with pytest.raises(RunnerContractError, match="grid|seed|cell"):
        keywords = {}
        if len(plan.inputs) == 16:
            keywords = {
                "authority": (
                    load_v28_revision_authority()
                    if scope == "revision"
                    else load_runner_authority()
                ),
                "datasets": datasets,
            }
        if scope == "synthetic":
            _validate_direct_competition_plan_structure(
                changed,
                registry=registry,
                prepared_datasets=prepared,
            )
        else:
            validate_direct_competition_plan(
                changed,
                registry=registry,
                prepared_datasets=prepared,
                **keywords,
            )


def test_production_direct_plan_requires_complete_runner_and_dataset_authority() -> (
    None
):
    plan, registry, datasets, prepared = _direct_fixture()
    prepared_map = {value.binding.dataset_id: value for value in prepared}

    with pytest.raises(TypeError, match="required keyword-only argument"):
        validate_direct_competition_plan(
            plan,
            registry=registry,
            prepared_datasets=prepared_map,
        )

    authority = load_runner_authority()
    changed = _build_structural_direct_competition_plan(
        registry,
        tuple(reversed(datasets)),
        authority,
        tuple(reversed(prepared)),
    )
    with pytest.raises(RunnerContractError, match="authority|canonical|dataset"):
        validate_direct_competition_plan(
            changed,
            registry=registry,
            prepared_datasets=prepared_map,
            authority=authority,
            datasets=datasets,
        )


def test_public_direct_boundaries_require_production_authority_arguments() -> None:
    from maskimpute_benchmark.development_evaluation import (
        project_direct_comparator_evidence,
    )
    from maskimpute_benchmark.downstream_evidence import (
        validate_direct_comparator_projection,
    )
    from maskimpute_benchmark.runner import execute_fair_comparator_plan

    boundaries = (
        (validate_direct_competition_plan, ("authority", "datasets")),
        (DirectCheckpointStore.write, ("authority", "datasets")),
        (DirectCheckpointStore.load, ("authority", "datasets")),
        (DirectCheckpointStore.append, ("authority", "datasets")),
        (DirectCheckpointStore.inspect_prefix, ("authority", "datasets")),
        (execute_fair_comparator_plan, ("authority", "datasets")),
        (
            project_direct_comparator_evidence,
            ("runner_authority", "datasets"),
        ),
        (
            validate_direct_comparator_projection,
            ("runner_authority", "datasets"),
        ),
    )

    for boundary, names in boundaries:
        signature = inspect.signature(boundary)
        assert all(
            signature.parameters[name].default is inspect.Parameter.empty
            for name in names
        ), boundary.__qualname__


def test_public_direct_boundaries_reject_coherent_15_input_plan(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )
    from maskimpute_benchmark.runner import execute_fair_comparator_plan

    plan, registry, datasets, prepared = _direct_fixture()
    changed, prepared_by_id = _truncate_direct_plan(plan, prepared)
    authority = load_runner_authority()

    with pytest.raises(RunnerContractError, match="production|canonical|16"):
        validate_direct_competition_plan(
            changed,
            registry=registry,
            prepared_datasets=prepared_by_id,
            authority=authority,
            datasets=datasets,
        )
    with pytest.raises(RunnerContractError, match="production|canonical|16"):
        DirectCheckpointStore(tmp_path / "write.json").write(
            changed,
            (),
            registry=registry,
            prepared_datasets=prepared_by_id,
            authority=authority,
            datasets=datasets,
        )
    recovery_store = DirectCheckpointStore(tmp_path / "load.json")
    recovery_marker = b"must-not-be-read-or-recovered"
    recovery_store.intent_path.write_bytes(recovery_marker)
    with pytest.raises(RunnerContractError, match="production|canonical|16"):
        recovery_store.load(
            changed,
            registry=registry,
            prepared_datasets=prepared_by_id,
            authority=authority,
            datasets=datasets,
        )
    assert recovery_store.intent_path.read_bytes() == recovery_marker
    with pytest.raises(RunnerContractError, match="production|canonical|16"):
        DirectCheckpointStore(tmp_path / "append.json").append(
            changed,
            None,
            object(),  # type: ignore[arg-type]
            registry=registry,
            prepared_datasets=prepared_by_id,
            authority=authority,
            datasets=datasets,
        )

    def forbidden_executor(*_args: object) -> object:
        raise AssertionError("non-production plan reached the executor")

    with pytest.raises(RunnerContractError, match="production|canonical|16"):
        execute_fair_comparator_plan(
            changed,
            registry,
            prepared_by_id,
            forbidden_executor,
            DirectCheckpointStore(tmp_path / "runner.json"),
            authority=authority,
            datasets=datasets,
        )

    comparator_authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    with pytest.raises(
        DevelopmentEvaluationError,
        match="production|canonical|16",
    ):
        project_direct_comparator_evidence(
            tmp_path / "downstream.json",
            changed,
            registry=registry,
            prepared_datasets=prepared_by_id,
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=comparator_authority,
            selected_rows=(comparator_authority.configurations[0],),
            runner_authority=authority,
            datasets=datasets,
        )


@pytest.mark.parametrize(
    "selection",
    ((), (0,), (2, 7, 11)),
    ids=("empty", "one-row-nonwinning", "arbitrary-subset"),
)
def test_public_direct_handoff_never_accepts_caller_selected_rows(
    tmp_path: Path,
    selection: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )

    _allow_unbound_smoke_only_for_unrelated_structural_test(monkeypatch)
    plan, registry, datasets, prepared = _direct_fixture()
    authority = load_runner_authority()
    comparator_authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    checkpoint_path = tmp_path / "incomplete.json"
    DirectCheckpointStore(checkpoint_path).write(
        plan,
        (),
        registry=registry,
        prepared_datasets={value.binding.dataset_id: value for value in prepared},
        authority=authority,
        datasets=datasets,
    )

    with pytest.raises(DevelopmentEvaluationError, match="denominator is not terminal"):
        project_direct_comparator_evidence(
            checkpoint_path,
            plan,
            registry=registry,
            prepared_datasets={value.binding.dataset_id: value for value in prepared},
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=comparator_authority,
            selected_rows=tuple(
                comparator_authority.configurations[index] for index in selection
            ),
            runner_authority=authority,
            datasets=datasets,
        )


def test_public_direct_handoff_rejects_candidate_only_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.comparator_tuning import ComparatorAuthorityReference
    from maskimpute_benchmark.development_evaluation import (
        DevelopmentEvaluationError,
        project_direct_comparator_evidence,
    )

    _allow_unbound_smoke_only_for_unrelated_structural_test(monkeypatch)
    _base, registry, datasets, prepared = _direct_fixture()
    authority = load_v28_revision_authority()
    plan = _build_structural_direct_competition_plan(
        registry, datasets, authority, prepared
    )
    comparator_authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )

    with pytest.raises(DevelopmentEvaluationError, match="base.*denominator"):
        project_direct_comparator_evidence(
            tmp_path / "candidate.json",
            plan,
            registry=registry,
            prepared_datasets={value.binding.dataset_id: value for value in prepared},
            comparator_reference=ComparatorAuthorityReference(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=comparator_authority,
            selected_rows=(),
            runner_authority=authority,
            datasets=datasets,
        )


def test_production_direct_plan_rejects_coherent_noncomparator_payload_mutation() -> (
    None
):
    plan, registry, datasets, prepared = _direct_fixture()
    configuration = plan.configurations[0]
    payload = (("forged", 1),)
    changed_configuration = replace(configuration, payload=payload)
    entries = []
    for entry in plan.entries:
        if entry.identity.method.method_id == configuration.method.method_id:
            identity = replace(entry.identity, configuration_payload=payload)
            entry = replace(
                entry,
                identity=identity,
                run_id=direct_run_id(identity),
            )
        entries.append(entry)
    changed = replace(
        plan,
        configurations=(changed_configuration, *plan.configurations[1:]),
        entries=tuple(entries),
    )

    with pytest.raises(RunnerContractError, match="authority|canonical|configuration"):
        validate_direct_competition_plan(
            changed,
            registry=registry,
            prepared_datasets={value.binding.dataset_id: value for value in prepared},
            authority=load_runner_authority(),
            datasets=datasets,
        )


def _contains_forbidden_identity_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            (
                str(key).casefold() != "shape"
                and any(
                    token in str(key).casefold() for token in FORBIDDEN_IDENTITY_TOKENS
                )
            )
            or _contains_forbidden_identity_key(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_identity_key(item) for item in value)
    return False


def test_direct_plan_has_exact_denominator_and_no_summary_fields() -> None:
    plan, _registry, _datasets, _prepared = _direct_fixture()

    assert plan.identity_mode == "direct-v1"
    assert len(plan.entries) == 2_896
    assert len(plan.configurations) == 61
    assert {
        "observed": sum(row.identity.method_id == "observed" for row in plan.entries),
        "capacity": sum(
            row.identity.method_id == "capacity-matched-ae" for row in plan.entries
        ),
        "maskimpute": sum(
            row.identity.method_id == "maskimpute" for row in plan.entries
        ),
        "comparators": sum(
            row.identity.configuration_kind == "comparator_tuning"
            for row in plan.entries
        ),
    } == {"observed": 16, "capacity": 48, "maskimpute": 1_200, "comparators": 1_632}
    encoded = plan.to_dict()
    assert set(encoded) == {
        "schema_version",
        "identity_mode",
        "authority_revision",
        "inputs",
        "entries",
        "configurations",
        "comparator_smoke_receipt",
        "comparator_smoke_receipt_bytes",
    }
    assert not _contains_forbidden_identity_key(encoded)


def test_direct_plan_carries_full_frozen_methods_payloads_and_prepared_inputs() -> None:
    plan, registry, _datasets, prepared = _direct_fixture()
    authority = load_runner_authority()

    assert plan.authority_revision == authority.comparator_tuning.authority_revision
    assert plan.inputs[0] == describe_prepared_input(prepared[0])
    assert plan.inputs[0].source_reference == prepared[0].binding.output_path
    assert plan.inputs[0].preprocessing_revision == "paired-zero-library-union-v1"
    assert plan.inputs[0].shape == (2, 3)
    assert plan.inputs[0].dtype == "<f8"
    assert plan.inputs[0].cell_ids == prepared[0].method_input.obs_ids
    assert plan.inputs[0].gene_ids == prepared[0].method_input.var_ids
    assert plan.inputs[0].batch_labels == ("batch-0", "batch-1")
    assert plan.inputs[0].total_count == 4.0
    assert plan.inputs[0].nonzero_count == 3
    assert plan.inputs[0].minimum == 0.0
    assert plan.inputs[0].maximum == 2.0
    assert plan.inputs[0].mechanism == "symsim"
    assert plan.inputs[0].mask_seed == 20_001
    assert plan.inputs[0].technical_view == "moderate"
    assert plan.inputs[1].batch_labels == ()

    assert all(
        type(configuration.payload) is tuple for configuration in plan.configurations
    )
    assert all(
        configuration.method
        == comparator_method_binding(registry.by_id(configuration.method.method_id))
        for configuration in plan.configurations
    )
    encoded_configurations = plan.to_dict()["configurations"]
    assert isinstance(encoded_configurations, list)
    encoded_by_identity = {
        (row["method"]["method_id"], row["configuration_id"]): row
        for row in encoded_configurations
    }
    expected_payloads = {
        (row.method_id, row.configuration_id): dict(row.payload)
        for row in authority.configurations
    }
    expected_payloads.update(
        {
            (row.method_id, row.configuration_id): dict(row.payload)
            for row in authority.comparator_tuning.configurations
        }
    )
    expected_payloads[("observed", "registry-default")] = {}
    assert set(encoded_by_identity) == set(expected_payloads)
    assert {
        identity: row["payload"] for identity, row in encoded_by_identity.items()
    } == expected_payloads
    authority_magic = authority.comparator_tuning.configurations_for("magic")[0]
    magic = next(
        row
        for row in encoded_configurations
        if row["method"]["method_id"] == "magic"
        and row["configuration_id"] == authority_magic.configuration_id
    )
    assert magic["method"] == asdict(comparator_method_binding(registry.by_id("magic")))
    assert magic["payload"] == dict(authority_magic.payload)


def test_real_plan_all_dca_requests_and_checkpoint_json_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_unbound_smoke_only_for_unrelated_structural_test(monkeypatch)
    plan, registry, datasets, prepared = _direct_fixture()
    runner_authority = load_runner_authority()
    comparator_authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    prepared_by_id = {value.binding.dataset_id: value for value in prepared}
    descriptor_by_id = {value.dataset_id: value for value in plan.inputs}
    dca_entries = tuple(
        entry for entry in plan.entries if entry.identity.method.method_id == "dca"
    )

    requests = tuple(
        create_direct_request(
            entry,
            prepared_by_id[entry.identity.dataset_id],
            descriptor_by_id[entry.identity.dataset_id],
            registry.by_id("dca"),
            comparator_authority,
            timeout_seconds=registry.by_id("dca").resources.timeout_seconds,
        )
        for entry in dca_entries
    )
    assert len(requests) == 192
    assert all(request.identity.configuration_payload for request in requests)

    plan_snapshot = plan.to_dict()
    encoded_entries = plan_snapshot["entries"]
    assert isinstance(encoded_entries, list)
    reason = "count_score_authority_pending"
    last_dca_position = max(
        position
        for position, entry in enumerate(plan.entries)
        if entry.identity.method.method_id == "dca"
    )
    records = []
    for position, entry in enumerate(plan.entries[: last_dca_position + 1]):
        current = prepared_by_id[entry.identity.dataset_id]
        encoded_entry = encoded_entries[position]
        assert isinstance(encoded_entry, dict)
        identity = encoded_entry["identity"]
        prezero = (
            {
                "applicable": True,
                "status": "blocked_authority",
                "reason": reason,
                "shape": None,
                "dtype": None,
                "encoding": None,
                "path": None,
                "compressed_byte_count": 0,
            }
            if entry.identity.method.method_id == "maskimpute"
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
        )
        records.append(
            {
                "run": {
                    "run_id": entry.run_id,
                    "identity": identity,
                    "status": "blocked_authority",
                    "reason": reason,
                    "runtime_seconds": 0,
                    "peak_rss_bytes": 0,
                    "peak_gpu_bytes": 0,
                    "rss_measurement": "executor_reported_unverified",
                    "gpu_measurement": "executor_reported_unverified",
                    "excluded_cell_count": current.audit.excluded_cell_count,
                    "excluded_cell_ids": list(current.audit.excluded_cell_ids),
                    "retained_cell_count": current.audit.retained_cell_count,
                    "retained_cell_ids": list(current.audit.retained_cell_ids),
                    "retained_gene_count": current.method_input.shape[1],
                    "observed_zero_count": int(
                        (current.method_input.counts == 0).sum()
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
                        "identity": identity,
                        "metric": metric,
                        "value": None,
                        "n": 0,
                        "status": "blocked_authority",
                        "reason": reason,
                    }
                    for metric in DIRECT_RECONSTRUCTION_METRICS
                ],
                "p_pre_zero_evidence": prezero,
            }
        )

    checkpoint_path = tmp_path / "checkpoint.json"
    report = DirectCheckpointStore(checkpoint_path).write(
        plan,
        records,
        registry=registry,
        prepared_datasets=prepared_by_id,
        authority=runner_authority,
        datasets=datasets,
    )
    stored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert stored == report.to_dict()
    stored_dca = tuple(
        record
        for record in stored["records"]
        if record["run"]["identity"]["method"]["method_id"] == "dca"
    )
    assert len(stored_dca) == 192
    assert all(
        isinstance(
            record["run"]["identity"]["configuration_payload"]["hidden_size"], list
        )
        for record in stored_dca
    )


def test_direct_plan_preserves_configuration_dataset_seed_order_and_blocks() -> None:
    plan, _registry, datasets, _prepared = _direct_fixture()

    assert [row.identity.ordinal for row in plan.entries] == list(range(1, 2_897))
    assert len({row.run_id for row in plan.entries}) == 2_896
    cursor = 0
    for configuration in plan.configurations:
        seeds = (
            (None,) if configuration.method.method_id == "observed" else (42, 43, 44)
        )
        expected = [
            (binding.dataset_id, seed) for binding in datasets for seed in seeds
        ]
        block = plan.entries[cursor : cursor + len(expected)]
        assert [
            (row.identity.dataset_id, row.identity.model_seed) for row in block
        ] == expected
        assert {
            (row.identity.method_id, row.identity.configuration_id) for row in block
        } == {(configuration.method.method_id, configuration.configuration_id)}
        cursor += len(expected)
    assert cursor == len(plan.entries)

    comparator_configurations = [
        row
        for row in plan.configurations
        if row.configuration_kind == "comparator_tuning"
    ]
    assert len(comparator_configurations) == 34
    for configuration in comparator_configurations:
        block = [
            index
            for index, entry in enumerate(plan.entries)
            if entry.identity.method_id == configuration.method.method_id
            and entry.identity.configuration_id == configuration.configuration_id
        ]
        assert len(block) == 48
        assert block == list(range(block[0], block[0] + 48))


@pytest.mark.parametrize(
    "loader",
    (load_v28_revision_authority, load_v29_revision_authority),
)
def test_direct_revision_plan_contains_one_48_row_candidate(loader) -> None:
    _base, registry, datasets, prepared = _direct_fixture()

    plan = _build_structural_direct_competition_plan(
        registry, datasets, loader(), prepared
    )

    assert len(plan.configurations) == 1
    assert len(plan.entries) == 48
    assert {entry.identity.method_id for entry in plan.entries} == {"maskimpute"}
    assert all(
        entry.identity.configuration_kind == "candidate_search"
        for entry in plan.entries
    )


def test_runner_fair_comparator_entry_point_delegates_to_direct_plan() -> None:
    direct, registry, datasets, prepared = _direct_fixture()

    with pytest.raises(TypeError, match="smoke_receipt"):
        build_fair_comparator_plan(
            registry,
            datasets,
            load_runner_authority(),
            prepared,
        )
    assert direct == _build_structural_direct_competition_plan(
        registry,
        datasets,
        load_runner_authority(),
        prepared,
    )


def test_public_direct_builder_requires_and_validates_complete_smoke_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _direct, registry, datasets, prepared = _direct_fixture()
    authority = load_runner_authority()
    receipt = {"status": "ready"}
    receipt_bytes = b'{"status":"ready"}\n'

    plan_parameters = inspect.signature(DirectCompetitionPlan).parameters
    assert (
        plan_parameters["comparator_smoke_receipt"].default
        is inspect.Parameter.empty
    )
    assert (
        plan_parameters["comparator_smoke_receipt_bytes"].default
        is inspect.Parameter.empty
    )

    with pytest.raises(TypeError, match="comparator_smoke_receipt"):
        build_direct_competition_plan(registry, datasets, authority, prepared)

    observed = []

    def validate(payload, raw, *, authority, registry):
        observed.append((payload, raw, authority, registry))
        return dict(payload)

    monkeypatch.setattr(
        "maskimpute_benchmark.comparator_tuning.validate_comparator_smoke_receipt",
        validate,
    )
    plan = build_direct_competition_plan(
        registry,
        datasets,
        authority,
        prepared,
        comparator_smoke_receipt=receipt,
        comparator_smoke_receipt_bytes=receipt_bytes,
    )

    assert observed
    assert all(value[:2] == (receipt, receipt_bytes) for value in observed)
    assert plan.comparator_smoke_receipt == (("status", "ready"),)
    assert plan.comparator_smoke_receipt_bytes == receipt_bytes


def test_production_plan_and_checkpoint_reject_unbound_smoke_evidence(
    tmp_path: Path,
) -> None:
    plan, registry, datasets, prepared = _direct_fixture()
    prepared_by_id = {
        value.binding.dataset_id: value for value in prepared
    }
    authority = load_runner_authority()

    with pytest.raises(RunnerContractError, match="smoke receipt"):
        validate_direct_competition_plan(
            plan,
            registry=registry,
            prepared_datasets=prepared_by_id,
            authority=authority,
            datasets=datasets,
        )
    checkpoint = tmp_path / "checkpoint.json"
    with pytest.raises(RunnerContractError, match="smoke receipt"):
        DirectCheckpointStore(checkpoint).write(
            plan,
            (),
            registry=registry,
            prepared_datasets=prepared_by_id,
            authority=authority,
            datasets=datasets,
        )
    assert not checkpoint.exists()


def test_prepared_descriptor_normalizes_real_h5ad_mask_seed(tmp_path: Path) -> None:
    _plan, _registry, _datasets, prepared = _direct_fixture()
    path = tmp_path / "prepared-evaluator.h5ad"
    prepared[0].evaluator_dataset.write_h5ad(path)
    evaluator_dataset = ad.read_h5ad(path)
    stored_seed = evaluator_dataset.uns["provenance"]["seeds"]["measurement"]
    assert isinstance(stored_seed, np.integer)

    descriptor = describe_prepared_input(
        replace(prepared[0], evaluator_dataset=evaluator_dataset)
    )

    assert descriptor.mask_seed == 20_001
    assert type(descriptor.mask_seed) is int


@pytest.mark.parametrize("mask_seed", (True, -1))
def test_prepared_descriptor_rejects_invalid_evaluator_mask_seed(
    mask_seed: object,
) -> None:
    _plan, _registry, _datasets, prepared = _direct_fixture()
    bad_seed = prepared[0].evaluator_dataset.copy()
    bad_seed.uns["provenance"]["seeds"]["measurement"] = mask_seed
    with pytest.raises(RunnerContractError, match="mask seed"):
        describe_prepared_input(replace(prepared[0], evaluator_dataset=bad_seed))


def test_prepared_descriptor_requires_exact_evaluator_draw_integer() -> None:
    _plan, _registry, _datasets, prepared = _direct_fixture()
    bad_draw = prepared[0].evaluator_dataset.copy()
    bad_draw.obs["draw"] = np.asarray([1.0, 1.0], dtype=np.float64)
    with pytest.raises(RunnerContractError, match="draw index"):
        describe_prepared_input(replace(prepared[0], evaluator_dataset=bad_draw))


def test_prepared_descriptor_uses_dataset_id_when_binding_has_no_output_path() -> None:
    _plan, _registry, datasets, prepared = _direct_fixture()
    binding = replace(datasets[0], output_path="")
    value = replace(prepared[0], binding=binding)

    descriptor = describe_prepared_input(value)

    assert descriptor.source_reference == binding.dataset_id
