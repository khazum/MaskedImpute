from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from maskimpute_benchmark.comparator_tuning import comparator_method_binding
from maskimpute_benchmark.fair_comparator_plan import (
    DirectCompetitionPlan,
    build_direct_competition_plan,
    describe_prepared_input,
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


def _direct_fixture() -> tuple[DirectCompetitionPlan, object, tuple[object, ...], tuple[PreparedDataset, ...]]:
    registry = load_method_registry(METHODS_PATH)
    datasets = validate_development_manifest_payload(_manifest_payload())
    prepared = tuple(
        _prepared_dataset(binding, ordinal, include_batch=ordinal == 1)
        for ordinal, binding in enumerate(datasets, start=1)
    )
    plan = build_direct_competition_plan(
        registry,
        datasets,
        load_runner_authority(),
        prepared,
    )
    return plan, registry, datasets, prepared


def _contains_forbidden_identity_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            (
                str(key).casefold() != "shape"
                and any(
                    token in str(key).casefold()
                    for token in FORBIDDEN_IDENTITY_TOKENS
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
        "observed": sum(
            row.identity.method_id == "observed" for row in plan.entries
        ),
        "capacity": sum(
            row.identity.method_id == "capacity-matched-ae"
            for row in plan.entries
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
        type(configuration.payload) is tuple
        for configuration in plan.configurations
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


def test_direct_plan_preserves_configuration_dataset_seed_order_and_blocks() -> None:
    plan, _registry, datasets, _prepared = _direct_fixture()

    assert [row.identity.ordinal for row in plan.entries] == list(range(1, 2_897))
    assert len({row.run_id for row in plan.entries}) == 2_896
    cursor = 0
    for configuration in plan.configurations:
        seeds = (
            (None,)
            if configuration.method.method_id == "observed"
            else (42, 43, 44)
        )
        expected = [(binding.dataset_id, seed) for binding in datasets for seed in seeds]
        block = plan.entries[cursor : cursor + len(expected)]
        assert [
            (row.identity.dataset_id, row.identity.model_seed) for row in block
        ] == expected
        assert {
            (row.identity.method_id, row.identity.configuration_id) for row in block
        } == {
            (configuration.method.method_id, configuration.configuration_id)
        }
        cursor += len(expected)
    assert cursor == len(plan.entries)

    comparator_configurations = [
        row for row in plan.configurations if row.configuration_kind == "comparator_tuning"
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

    plan = build_direct_competition_plan(registry, datasets, loader(), prepared)

    assert len(plan.configurations) == 1
    assert len(plan.entries) == 48
    assert {entry.identity.method_id for entry in plan.entries} == {"maskimpute"}
    assert all(
        entry.identity.configuration_kind == "candidate_search"
        for entry in plan.entries
    )


def test_runner_fair_comparator_entry_point_delegates_to_direct_plan() -> None:
    direct, registry, datasets, prepared = _direct_fixture()

    delegated = build_fair_comparator_plan(
        registry,
        datasets,
        load_runner_authority(),
        prepared,
    )

    assert delegated == direct


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
