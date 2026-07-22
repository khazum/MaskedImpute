from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@lru_cache(maxsize=1)
def _frozen_method_authorities():
    from maskimpute_benchmark.final_runner import _frozen_method_plan_authority

    path = Path(__file__).with_name("test_final_runner.py")
    spec = importlib.util.spec_from_file_location(
        "_task18_final_null_de_final_factory",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = module._direct_frozen_method()
    registry = module._full_registry()
    _rows, configurations = _frozen_method_plan_authority(frozen, registry)
    return configurations


def _source_plan(*, include_failed: bool = True):
    from maskimpute_benchmark.downstream_evidence import (
        DatasetEvidenceBinding,
        DownstreamEvidencePlan,
        DownstreamPlanEntry,
        EvaluatedRoundBinding,
    )

    cells = tuple(f"cell-{index:03d}" for index in range(40))
    genes = tuple(f"gene-{index:03d}" for index in range(120))
    authorities = {
        authority.method_id: authority for authority in _frozen_method_authorities()
    }
    binding = DatasetEvidenceBinding(
        dataset_id="final-symsim-draw-01-moderate",
        path=str(Path("/tmp/final-null-de-fixture.h5ad").absolute()),
        file_sha256="1" * 64,
        dataset_sha256="2" * 64,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        method_input_sha256="3" * 64,
        dataset_qc_policy_sha256="4" * 64,
        excluded_cell_count=0,
        excluded_cell_ids_sha256="5" * 64,
        retained_cell_count=len(cells),
        retained_cell_ids_sha256="6" * 64,
        retained_gene_count=len(genes),
        observed_zero_count=0,
        retained_cell_ids=cells,
        gene_ids=genes,
    )

    def entry(
        ordinal: int,
        run_id: str,
        method_id: str,
        model_seed: int,
        status: str,
        reason: str | None,
    ) -> DownstreamPlanEntry:
        completed = status == "completed"
        authority = authorities[method_id]
        legacy = authority.legacy_configuration
        return DownstreamPlanEntry(
            ordinal=ordinal,
            source_record_path=f"records/{ordinal:08d}.json",
            source_record_sha256=f"{ordinal + 6:x}" * 64,
            run_id=run_id,
            method_id=method_id,
            dataset_id=binding.dataset_id,
            source_dataset_sha256=binding.dataset_sha256,
            mechanism=binding.mechanism,
            biological_id=binding.biological_id,
            technical_view=binding.technical_view,
            model_seed=model_seed,
            configuration_id=authority.configuration_id,
            configuration_sha256=(
                None if legacy is None else legacy.configuration_sha256
            ),
            configuration_kind=authority.kind,
            method_artifact_sha256=(
                None if legacy is None else f"{ordinal + 12:x}" * 64
            ),
            comparator_configuration=authority.comparator_configuration,
            comparator_nonexecution_identity=(
                authority.comparator_nonexecution_identity
            ),
            method_input_sha256=binding.method_input_sha256,
            retained_cell_ids_sha256=binding.retained_cell_ids_sha256,
            status=status,
            reason=reason,
            evaluator_output_sha256=("d" * 64 if completed else None),
            evaluator_output_path=(f"runs/{run_id}.zlib" if completed else None),
            evaluator_output_file_sha256=("e" * 64 if completed else None),
            evaluator_output_shape=((len(cells), len(genes)) if completed else None),
            evaluator_output_encoding=("zlib_raw_f64_v1" if completed else None),
            evaluator_output_uncompressed_nbytes=(
                len(cells) * len(genes) * 8 if completed else None
            ),
            evaluator_output_uncompressed_sha256=("f" * 64 if completed else None),
        )

    entries = [
        entry(1, "maskimpute-seed-42", "maskimpute", 42, "completed", None),
        entry(2, "magic-seed-42", "magic", 42, "completed", None),
    ]
    if include_failed:
        entries.append(
            entry(
                3,
                "scvi-seed-42",
                "scvi",
                42,
                "resource_exceeded",
                "peak_gpu_memory_limit_exceeded",
            )
        )
    evaluated = EvaluatedRoundBinding(
        repository_root=str(Path("/tmp/final-null-de-repository").absolute()),
        round_root=str(Path("/tmp/final-null-de-repository/round-1").absolute()),
        round_id="round-1",
        evaluation_receipt_path="evaluation_receipt.json",
        evaluation_receipt_file_sha256="a" * 64,
        evaluation_receipt_payload_sha256="b" * 64,
        result_manifest_sha256="c" * 64,
        final_plan_sha256="d" * 64,
        final_execution_manifest_path=(
            "results/final/execution/execution_manifest.json"
        ),
        final_execution_manifest_file_sha256="e" * 64,
        final_execution_manifest_payload_sha256="f" * 64,
        execution_validation_sha256="1" * 64,
        storage_preflight_sha256="2" * 64,
        scaling_evidence_sha256="3" * 64,
        scaling_plan_sha256="4" * 64,
        scaling_checkpoint_path=("results/scaling/checkpoints/00000024.json"),
        scaling_checkpoint_file_sha256="5" * 64,
        scaling_checkpoint_payload_sha256="6" * 64,
        scaling_checkpoint_history_sha256="7" * 64,
        scaling_checkpoint_history_count=24,
        scaling_result_files_sha256="8" * 64,
        scaling_result_file_count=100,
        trajectory_evidence_sha256="9" * 64,
        trajectory_plan_sha256="a" * 64,
        trajectory_execution_claim_sha256="b" * 64,
        trajectory_execution_environment_sha256="c" * 64,
        trajectory_dataset_id="trajectory-exact-latent-01",
        trajectory_dataset_sha256="d" * 64,
        trajectory_dataset_file_sha256="e" * 64,
        trajectory_dataset_receipt_file_sha256="f" * 64,
        trajectory_dataset_receipt_payload_sha256="1" * 64,
        trajectory_source_id="registered-synthetic-trajectory-v1",
        trajectory_root_cell_id="trajectory-cell-000001",
        trajectory_registered_authority_sha256="2" * 64,
        trajectory_registered_binding_sha256="3" * 64,
        trajectory_authority_sha256="4" * 64,
        trajectory_authority_file_sha256="5" * 64,
        trajectory_execution_manifest_path=(
            "results/trajectory/execution/execution_manifest.json"
        ),
        trajectory_execution_manifest_file_sha256="6" * 64,
        trajectory_execution_manifest_payload_sha256="7" * 64,
        trajectory_execution_validation_sha256="8" * 64,
        trajectory_record_payload_sha256s_sha256="9" * 64,
        trajectory_status_counts_sha256="a" * 64,
        trajectory_planned_run_count=12,
        trajectory_result_files_sha256="b" * 64,
        trajectory_result_file_count=48,
    )
    plan = DownstreamEvidencePlan(
        source_root=str(Path(evaluated.round_root) / "results/final/execution"),
        source_kind="final",
        evidence_scope="all",
        evaluator_source_sha256="3" * 64,
        source_manifest_path="execution_manifest.json",
        source_manifest_file_sha256="4" * 64,
        source_manifest_payload_sha256="5" * 64,
        source_plan_sha256="6" * 64,
        source_input_hashes_sha256="7" * 64,
        source_statuses_sha256="8" * 64,
        source_plan_authority="required_exact_execution_plan_v1",
        evaluated_round_binding=evaluated,
        development_revision_versions=(),
        development_sources=(),
        datasets=(binding,),
        configurations=tuple(
            authorities[method_id] for method_id in ("maskimpute", "magic", "scvi")
        ),
        entries=tuple(entries),
        plan_sha256="9" * 64,
    )
    return plan, binding


def _dataset(binding: object) -> ad.AnnData:
    cells = tuple(getattr(binding, "retained_cell_ids"))
    genes = tuple(getattr(binding, "gene_ids"))
    counts = np.asarray(
        [
            [((cell + 3) * (gene + 5)) % 41 + 1 for gene in range(len(genes))]
            for cell in range(len(cells))
        ],
        dtype=np.int64,
    )
    return ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"group": ["population-a"] * 20 + ["population-b"] * 20},
            index=cells,
        ),
        var=pd.DataFrame(index=genes),
    )


def test_final_null_de_entropy_is_dataset_scoped_and_receipt_bound() -> None:
    from maskimpute_benchmark.final_null_de import final_null_de_entropy_sha256

    plan, binding = _source_plan()
    receipt = plan.evaluated_round_binding.evaluation_receipt_payload_sha256
    observed = final_null_de_entropy_sha256(
        receipt_payload_sha256=receipt,
        dataset_id=binding.dataset_id,
        dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        retained_cell_ids=binding.retained_cell_ids,
    )

    assert len(observed) == 64
    assert observed == final_null_de_entropy_sha256(
        receipt_payload_sha256=receipt,
        dataset_id=binding.dataset_id,
        dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        retained_cell_ids=binding.retained_cell_ids,
    )
    assert observed != final_null_de_entropy_sha256(
        receipt_payload_sha256="0" * 64,
        dataset_id=binding.dataset_id,
        dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        retained_cell_ids=binding.retained_cell_ids,
    )
    assert observed != final_null_de_entropy_sha256(
        receipt_payload_sha256=receipt,
        dataset_id=binding.dataset_id,
        dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        retained_cell_ids=binding.retained_cell_ids[:-1],
    )


def test_final_null_de_uses_one_observed_mask_and_split_for_every_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_null_de as final_null_de
    from maskimpute_benchmark.downstream_evaluation import MethodOutput
    from maskimpute_benchmark.methods import count_equivalent_to_log2_cp10k

    plan, binding = _source_plan()
    dataset = _dataset(binding)
    observed = count_equivalent_to_log2_cp10k(np.asarray(dataset.X))
    outputs = {
        "maskimpute-seed-42": observed,
        "magic-seed-42": observed + 0.01,
    }
    monkeypatch.setattr(
        final_null_de,
        "_load_bound_dataset",
        lambda selected_binding: (
            dataset
            if selected_binding.dataset_id == binding.dataset_id
            else pytest.fail("unexpected dataset binding")
        ),
    )
    monkeypatch.setattr(
        final_null_de,
        "_decode_bound_output",
        lambda _plan, entry, selected_binding: MethodOutput(
            values=outputs[entry.run_id],
            cell_ids=selected_binding.retained_cell_ids,
            gene_ids=selected_binding.gene_ids,
        ),
    )

    records = final_null_de.evaluate_final_null_de_records(plan)

    assert len(records) == len(plan.entries)
    completed = records[:2]
    assert {record["status"] for record in completed} == {"completed"}
    assert len({record["entropy_sha256"] for record in completed}) == 1
    assert len({record["split_sha256"] for record in completed}) == 1
    assert len({record["gene_mask_sha256"] for record in completed}) == 1
    assert all(record["nominal_alpha"] == 0.05 for record in completed)
    failed = records[2]
    assert failed["status"] == "resource_exceeded"
    assert failed["fpr"] is None
    assert failed["reason_code"] == "upstream_run_not_completed"
    assert failed["upstream_reason"] == "peak_gpu_memory_limit_exceeded"
    assert failed["split_sha256"] == completed[0]["split_sha256"]
    assert failed["gene_mask_sha256"] == completed[0]["gene_mask_sha256"]


def test_final_null_de_is_invariant_to_stable_id_row_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_null_de as final_null_de
    from maskimpute_benchmark.downstream_evaluation import MethodOutput
    from maskimpute_benchmark.methods import count_equivalent_to_log2_cp10k

    plan, binding = _source_plan(include_failed=False)
    dataset = _dataset(binding)
    observed = count_equivalent_to_log2_cp10k(np.asarray(dataset.X))

    def evaluate(selected_plan, selected_binding, selected_dataset, output):
        monkeypatch.setattr(
            final_null_de,
            "_load_bound_dataset",
            lambda _binding: selected_dataset,
        )
        monkeypatch.setattr(
            final_null_de,
            "_decode_bound_output",
            lambda _plan, _entry, _binding: MethodOutput(
                values=output,
                cell_ids=selected_binding.retained_cell_ids,
                gene_ids=selected_binding.gene_ids,
            ),
        )
        return final_null_de.evaluate_final_null_de_records(selected_plan)

    baseline = evaluate(plan, binding, dataset, observed)
    order = np.asarray(list(reversed(range(dataset.n_obs))))
    permuted_dataset = dataset[order, :].copy()
    permuted_cells = tuple(permuted_dataset.obs_names.astype(str))
    permuted_binding = replace(
        binding,
        retained_cell_ids=permuted_cells,
        retained_cell_ids_sha256="a" * 64,
    )
    permuted_entries = tuple(
        replace(entry, retained_cell_ids_sha256="a" * 64) for entry in plan.entries
    )
    permuted_plan = replace(
        plan,
        datasets=(permuted_binding,),
        entries=permuted_entries,
    )
    permuted = evaluate(
        permuted_plan,
        permuted_binding,
        permuted_dataset,
        observed[order, :],
    )

    assert [row["fpr"] for row in permuted] == [row["fpr"] for row in baseline]
    assert [row["split_sha256"] for row in permuted] == [
        row["split_sha256"] for row in baseline
    ]
    assert [row["gene_mask_sha256"] for row in permuted] == [
        row["gene_mask_sha256"] for row in baseline
    ]


def test_final_null_de_archive_resumes_and_replays_every_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_null_de as final_null_de
    from maskimpute_benchmark.downstream_evaluation import MethodOutput
    from maskimpute_benchmark.methods import count_equivalent_to_log2_cp10k

    source_plan, binding = _source_plan()
    repository = tmp_path / "repository"
    round_root = repository / "round-1"
    round_root.mkdir(parents=True)
    source_plan = replace(
        source_plan,
        evaluated_round_binding=replace(
            source_plan.evaluated_round_binding,
            repository_root=str(repository),
            round_root=str(round_root),
        ),
    )
    dataset = _dataset(binding)
    observed = count_equivalent_to_log2_cp10k(np.asarray(dataset.X))
    plan = final_null_de._create_final_null_de_plan(
        source_plan,
        downstream_directory=str(tmp_path / "validated-downstream"),
        downstream_manifest_file_sha256="a" * 64,
        downstream_manifest_payload_sha256="b" * 64,
    )
    monkeypatch.setattr(final_null_de, "_rebuild_plan", lambda _plan: plan)
    monkeypatch.setattr(final_null_de, "_load_bound_dataset", lambda _binding: dataset)
    monkeypatch.setattr(
        final_null_de,
        "_decode_bound_output",
        lambda _plan, _entry, selected_binding: MethodOutput(
            values=observed,
            cell_ids=selected_binding.retained_cell_ids,
            gene_ids=selected_binding.gene_ids,
        ),
    )
    destination = final_null_de.expected_final_null_de_output_directory(plan)

    partial = final_null_de.run_final_null_de_evidence(
        plan,
        destination,
        max_denominators=1,
    )
    assert partial["status"] == "running"
    assert partial["recorded_denominator_count"] == 1
    interrupted_staging = destination / "records/.00000002.json.Interrupted123.tmp"
    interrupted_staging.write_bytes(b"interrupted publication")
    completed = final_null_de.run_final_null_de_evidence(plan, destination)
    assert completed["status"] == "completed"
    assert completed["recorded_denominator_count"] == len(source_plan.entries)
    assert not interrupted_staging.exists()

    loaded = final_null_de.load_final_null_de_manifest(destination)
    assert loaded.plan_sha256 == plan.plan_sha256
    assert len(loaded.manifest_file_sha256) == 64
    assert len(loaded.records) == len(source_plan.entries)

    record_path = destination / "records/00000001.json"
    hardlink_alias = tmp_path / "record-hardlink-alias.json"
    os.link(record_path, hardlink_alias)
    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="unique regular file",
    ):
        final_null_de.load_final_null_de_manifest(destination)
    hardlink_alias.unlink()

    unexpected = destination / "untracked-result.txt"
    unexpected.write_text("untracked\n", encoding="utf-8")
    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="archive layout",
    ):
        final_null_de.load_final_null_de_manifest(destination)
    unexpected.unlink()

    forged = json.loads(record_path.read_text(encoding="utf-8"))
    forged["fpr"] = 0.99
    body = {key: value for key, value in forged.items() if key != "record_sha256"}
    from maskimpute_benchmark.protocol import canonical_sha256

    forged["record_sha256"] = canonical_sha256(body)
    record_path.chmod(0o600)
    record_path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="re-evaluation differs",
    ):
        final_null_de._manifest_payload(destination, plan, loaded.records)
    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="re-evaluation differs",
    ):
        final_null_de.load_final_null_de_manifest(destination)


@pytest.mark.parametrize("operation", ["load", "run"])
def test_final_null_de_rejects_completed_manifest_for_partial_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import maskimpute_benchmark.final_null_de as final_null_de
    from maskimpute_benchmark.downstream_evaluation import MethodOutput
    from maskimpute_benchmark.methods import count_equivalent_to_log2_cp10k
    from maskimpute_benchmark.protocol import canonical_sha256

    source_plan, binding = _source_plan()
    repository = tmp_path / "repository"
    round_root = repository / "round-1"
    round_root.mkdir(parents=True)
    source_plan = replace(
        source_plan,
        evaluated_round_binding=replace(
            source_plan.evaluated_round_binding,
            repository_root=str(repository),
            round_root=str(round_root),
        ),
    )
    dataset = _dataset(binding)
    observed = count_equivalent_to_log2_cp10k(np.asarray(dataset.X))
    plan = final_null_de._create_final_null_de_plan(
        source_plan,
        downstream_directory=str(tmp_path / "validated-downstream"),
        downstream_manifest_file_sha256="a" * 64,
        downstream_manifest_payload_sha256="b" * 64,
    )
    monkeypatch.setattr(final_null_de, "_rebuild_plan", lambda _plan: plan)
    monkeypatch.setattr(final_null_de, "_load_bound_dataset", lambda _binding: dataset)
    monkeypatch.setattr(
        final_null_de,
        "_decode_bound_output",
        lambda _plan, _entry, selected_binding: MethodOutput(
            values=observed,
            cell_ids=selected_binding.retained_cell_ids,
            gene_ids=selected_binding.gene_ids,
        ),
    )
    destination = final_null_de.expected_final_null_de_output_directory(plan)
    partial = final_null_de.run_final_null_de_evidence(
        plan,
        destination,
        max_denominators=1,
    )
    assert partial["status"] == "running"
    expected = final_null_de._expected_records(plan)

    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="denominator is incomplete",
    ):
        final_null_de._manifest_payload(destination, plan, expected[:1])

    record = json.loads(
        (destination / "records/00000001.json").read_text(encoding="utf-8")
    )
    record_file_sha256 = hashlib.sha256(
        (destination / "records/00000001.json").read_bytes()
    ).hexdigest()
    plan_file_sha256 = hashlib.sha256(
        (destination / "plan.json").read_bytes()
    ).hexdigest()
    manifest_body = {
        "schema_version": 1,
        "algorithm": final_null_de.FINAL_NULL_DE_ALGORITHM,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "plan_file_sha256": plan_file_sha256,
        "source_plan_sha256": plan.source_plan.plan_sha256,
        "evaluated_round_binding_sha256": (
            plan.source_plan.evaluated_round_binding.binding_sha256
        ),
        "downstream_manifest_file_sha256": plan.downstream_manifest_file_sha256,
        "downstream_manifest_payload_sha256": (plan.downstream_manifest_payload_sha256),
        "evaluator_source_sha256": plan.evaluator_source_sha256,
        "planned_denominator_count": len(plan.source_plan.entries),
        "recorded_denominator_count": 1,
        "records": [
            {
                "ordinal": 1,
                "run_id": record["run_id"],
                "path": "records/00000001.json",
                "sha256": record_file_sha256,
                "record_sha256": record["record_sha256"],
            }
        ],
    }
    forged_manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    (destination / "final_null_de_manifest.json").write_bytes(
        final_null_de._canonical_bytes(forged_manifest) + b"\n"
    )

    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="denominator is incomplete",
    ):
        if operation == "load":
            final_null_de.load_final_null_de_manifest(destination)
        else:
            final_null_de.run_final_null_de_evidence(plan, destination)


def test_final_null_de_output_must_be_external_and_not_symlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_null_de as final_null_de

    source_plan, _binding = _source_plan()
    repository = tmp_path / "repository"
    round_root = repository / "round-1"
    round_root.mkdir(parents=True)
    evaluated = replace(
        source_plan.evaluated_round_binding,
        repository_root=str(repository),
        round_root=str(round_root),
    )
    source_plan = replace(source_plan, evaluated_round_binding=evaluated)
    plan = final_null_de._create_final_null_de_plan(
        source_plan,
        downstream_directory=str(tmp_path / "validated-downstream"),
        downstream_manifest_file_sha256="a" * 64,
        downstream_manifest_payload_sha256="b" * 64,
    )
    monkeypatch.setattr(final_null_de, "_rebuild_plan", lambda _plan: plan)

    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="outside the frozen repository",
    ):
        final_null_de.run_final_null_de_evidence(
            plan,
            repository / "forbidden-null-de",
            max_denominators=0,
        )

    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="receipt namespace",
    ):
        final_null_de.run_final_null_de_evidence(
            plan,
            tmp_path / "arbitrary-external",
            max_denominators=0,
        )

    external = tmp_path / "external"
    external.mkdir()
    alias = final_null_de.expected_final_null_de_output_directory(plan)
    alias.parent.mkdir(parents=True)
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="symlink",
    ):
        final_null_de.run_final_null_de_evidence(
            plan,
            alias,
            max_denominators=0,
        )


def test_mathematically_unavailable_final_null_de_retains_complete_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_null_de as final_null_de

    plan, binding = _source_plan()
    dataset = _dataset(binding)
    sparse_denominator = np.zeros(dataset.shape, dtype=np.int64)
    sparse_denominator[:, :50] = np.asarray(dataset.X)[:, :50]
    dataset.X = sparse_denominator
    monkeypatch.setattr(final_null_de, "_load_bound_dataset", lambda _binding: dataset)
    monkeypatch.setattr(
        final_null_de,
        "_decode_bound_output",
        lambda *_args: pytest.fail("unavailable fixed denominator must skip outputs"),
    )

    records = final_null_de.evaluate_final_null_de_records(plan)

    assert len(records) == len(plan.entries)
    assert [record["status"] for record in records[:2]] == [
        "unavailable",
        "unavailable",
    ]
    assert {record["reason_code"] for record in records[:2]} == {
        "fixed_observed_gene_denominator_unavailable"
    }
    assert all(record["fpr"] is None for record in records)
    assert records[2]["status"] == "resource_exceeded"
    assert records[2]["reason_code"] == "upstream_run_not_completed"


def test_missing_evaluator_group_is_reason_coded_not_coerced_to_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.final_null_de as final_null_de

    plan, binding = _source_plan(include_failed=False)
    dataset = _dataset(binding)
    dataset.obs["group"] = dataset.obs["group"].astype(object)
    dataset.obs.iloc[0, dataset.obs.columns.get_loc("group")] = np.nan
    monkeypatch.setattr(final_null_de, "_load_bound_dataset", lambda _binding: dataset)
    monkeypatch.setattr(
        final_null_de,
        "_decode_bound_output",
        lambda *_args: pytest.fail("missing evaluator group must skip outputs"),
    )

    records = final_null_de.evaluate_final_null_de_records(plan)

    assert len(records) == len(plan.entries)
    assert {record["status"] for record in records} == {"unavailable"}
    assert {record["reason_code"] for record in records} == {
        "evaluator_group_labels_unavailable"
    }


def test_final_null_de_cli_has_only_round_locator_and_uses_receipt_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path("scripts/run_final_null_de.py").absolute()
    specification = importlib.util.spec_from_file_location(
        "run_final_null_de_test", script_path
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = script
    specification.loader.exec_module(script)

    repository = tmp_path / "repository"
    round_directory = repository / "artifacts/study/final/rounds/round-1"
    round_directory.mkdir(parents=True)
    plan = SimpleNamespace(
        source_plan=SimpleNamespace(
            evaluated_round_binding=SimpleNamespace(
                round_id="round-1",
                evaluation_receipt_payload_sha256="a" * 64,
                repository_root=str(repository),
            )
        )
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(script, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        script,
        "build_final_null_de_plan",
        lambda selected_repository, selected_round: (
            plan
            if selected_repository == repository and selected_round == round_directory
            else pytest.fail("CLI changed repository or round authority")
        ),
    )
    expected_output = (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "null-de"
        / "round-1"
        / ("a" * 64)
    )
    monkeypatch.setattr(
        script,
        "expected_final_null_de_output_directory",
        lambda selected_plan: (
            expected_output
            if selected_plan is plan
            else pytest.fail("CLI changed the validated plan")
        ),
    )

    def run(selected_plan, output_directory):
        observed["plan"] = selected_plan
        observed["output"] = output_directory
        return {"status": "completed"}

    monkeypatch.setattr(script, "run_final_null_de_evidence", run)
    relative_round = round_directory.relative_to(repository)
    assert script.main(["--round-dir", relative_round.as_posix()]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}
    assert observed == {"plan": plan, "output": expected_output}

    with pytest.raises(SystemExit):
        script.main(
            [
                "--round-dir",
                relative_round.as_posix(),
                "--method",
                "maskimpute",
            ]
        )


def test_production_plan_requires_exact_completed_final_downstream_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.final_null_de as final_null_de

    source_plan, _binding = _source_plan(include_failed=False)
    repository = tmp_path / "repository"
    round_root = repository / "round-1"
    round_root.mkdir(parents=True)
    source_plan = replace(
        source_plan,
        evaluated_round_binding=replace(
            source_plan.evaluated_round_binding,
            repository_root=str(repository),
            round_root=str(round_root),
        ),
    )
    downstream_directory = (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "downstream"
        / source_plan.evaluated_round_binding.round_id
        / source_plan.evaluated_round_binding.evaluation_receipt_payload_sha256
    )
    downstream_directory.mkdir(parents=True)
    trajectory_child = downstream_directory / "trajectory"
    trajectory_child.mkdir()
    (trajectory_child / "sentinel.json").write_text("{}\n", encoding="utf-8")
    manifest_path = downstream_directory / "downstream_manifest.json"
    manifest_path.write_bytes(b"validated downstream manifest\n")
    manifest = SimpleNamespace(
        plan_sha256=source_plan.plan_sha256,
        planned_denominator_count=len(source_plan.entries),
        manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(
        downstream,
        "build_final_downstream_evidence_plan",
        lambda selected_repository, selected_round: (
            source_plan
            if Path(selected_repository) == repository
            and Path(selected_round) == round_root
            else pytest.fail("production builder changed final source authority")
        ),
    )
    monkeypatch.setattr(
        downstream,
        "build_final_trajectory_downstream_evidence_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "final null-DE opened the supplementary trajectory source"
        ),
    )
    monkeypatch.setattr(
        downstream,
        "load_downstream_evidence_plan",
        lambda selected_directory: (
            source_plan
            if Path(selected_directory) == downstream_directory
            else pytest.fail("production builder changed downstream namespace")
        ),
    )
    monkeypatch.setattr(
        downstream,
        "load_downstream_evidence_manifest",
        lambda selected_directory: (
            manifest
            if Path(selected_directory) == downstream_directory
            else pytest.fail("production builder changed downstream namespace")
        ),
    )

    plan = final_null_de.build_final_null_de_plan(repository, round_root)

    assert plan.source_plan is source_plan
    assert plan.downstream_directory == str(downstream_directory)
    assert plan.downstream_manifest_payload_sha256 == "a" * 64
    assert len(plan.source_plan.entries) == len(source_plan.entries)
    assert (trajectory_child / "sentinel.json").read_text(encoding="utf-8") == "{}\n"

    monkeypatch.setattr(
        downstream,
        "load_downstream_evidence_plan",
        lambda _directory: replace(source_plan, source_statuses_sha256="0" * 64),
    )
    with pytest.raises(
        final_null_de.FinalNullDEError,
        match="downstream plan differs",
    ):
        final_null_de.build_final_null_de_plan(repository, round_root)
