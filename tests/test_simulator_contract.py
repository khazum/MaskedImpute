from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.simulators.base as base_module
import maskimpute_benchmark.simulators.native as native_module
import maskimpute_benchmark.study as study_module
from maskimpute_benchmark.protocol import load_protocol
from maskimpute_benchmark.schema import benchmark_dataset_sha256
from maskimpute_benchmark.study import (
    assert_final_runnable,
    freeze_round,
    materialize_final,
    record_final_evaluation,
    supersede_round,
)
from maskimpute_benchmark.simulators.base import (
    FinalManifestClaim,
    SimulationArtifact,
    SimulationContractError,
    SimulationRequest,
    biological_unit_id,
    load_final_manifest_claim,
    simulation_dataset_id,
    simulation_request_identity,
    simulation_scientific_identity,
    validate_paired_simulation_requests,
    validate_simulation_request,
)
from maskimpute_benchmark.simulators.native import (
    NativeManifest,
    seal_native_outputs,
)


PROTOCOL = load_protocol(Path("study/protocol.json"))


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


@pytest.fixture
def final_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Simulator Test")
    _git(repo, "config", "user.email", "simulator@example.invalid")
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repo / "config.json").write_text('{"method":"fixture"}\n', encoding="utf-8")
    (repo / "environment.lock").write_text("python=3.11\n", encoding="utf-8")
    (repo / "protocol.json").write_text(
        Path("study/protocol.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "freeze simulator fixture")
    round_dir = repo / "artifacts/study/round-001"
    freeze_round(
        repo,
        round_dir,
        repo / "config.json",
        repo / "protocol.json",
        environment_path=repo / "environment.lock",
    )
    return repo, round_dir


def _request(tmp_path: Path, **changes: object) -> SimulationRequest:
    values: dict[str, object] = {
        "mechanism": "symsim",
        "namespace": "dev",
        "biological_id": "draw-01",
        "biological_seed": 101,
        "measurement_seed": 202,
        "technical_view": "moderate",
        "cells": PROTOCOL.development.cells,
        "genes": PROTOCOL.development.genes,
        "output_path": tmp_path / "dev" / "symsim" / "draw-01-moderate.h5ad",
    }
    values.update(changes)
    return SimulationRequest(**values)


def test_simulation_request_is_frozen_and_validates_development_contract(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    validate_simulation_request(request, PROTOCOL)

    with pytest.raises(FrozenInstanceError):
        request.cells = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("mechanism", "splatter", "mechanism"),
        ("mechanism", "SymSim", "mechanism"),
        ("namespace", "development", "namespace"),
        ("namespace", "final", "final manifest claim"),
        ("biological_id", "../draw", "biological_id"),
        ("technical_view", "moderate/view", "technical_view"),
        ("biological_seed", True, "biological_seed"),
        ("biological_seed", -1, "biological_seed"),
        ("measurement_seed", 2**63, "measurement_seed"),
        ("cells", True, "cells"),
        ("cells", 899, "dimensions"),
        ("genes", 499, "dimensions"),
        ("output_path", "result.h5ad", "output_path"),
    ],
)
def test_request_fields_are_strictly_validated(
    tmp_path: Path, field: str, invalid: object, message: str
) -> None:
    request = _request(tmp_path, **{field: invalid})

    with pytest.raises(SimulationContractError, match=message):
        validate_simulation_request(request, PROTOCOL)


def test_biological_and_measurement_seeds_must_be_distinct(tmp_path: Path) -> None:
    request = _request(tmp_path, biological_seed=101, measurement_seed=101)

    with pytest.raises(SimulationContractError, match="distinct"):
        validate_simulation_request(request, PROTOCOL)


def test_request_output_path_must_be_canonical(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        output_path=tmp_path / "dev" / ".." / "dev" / "output.h5ad",
    )

    with pytest.raises(SimulationContractError, match="canonical"):
        validate_simulation_request(request, PROTOCOL)


@pytest.mark.parametrize(
    "biological_id",
    ["draw-00", "draw-1", "draw-001", "draw-03", "sample-01"],
)
def test_development_biological_id_is_canonical_and_within_protocol_draws(
    tmp_path: Path, biological_id: str
) -> None:
    request = _request(tmp_path, biological_id=biological_id)

    with pytest.raises(SimulationContractError, match="biological_id|draw"):
        validate_simulation_request(request, PROTOCOL)


def test_deterministic_ids_bind_seeded_design_fields(
    tmp_path: Path,
) -> None:
    moderate = _request(tmp_path)
    rerun = replace(moderate, biological_seed=303, measurement_seed=404)
    remeasured = replace(moderate, measurement_seed=404)
    relabeled_draw = replace(moderate, biological_id="draw-02")
    relabeled_view = replace(moderate, technical_view="severe")
    severe = replace(moderate, technical_view="severe", measurement_seed=505)

    assert simulation_dataset_id(moderate) != simulation_dataset_id(rerun)
    assert biological_unit_id(moderate) != biological_unit_id(rerun)
    assert simulation_dataset_id(moderate) != simulation_dataset_id(remeasured)
    assert biological_unit_id(moderate) == biological_unit_id(remeasured)
    assert biological_unit_id(moderate) == biological_unit_id(relabeled_draw)
    assert simulation_dataset_id(moderate) == simulation_dataset_id(relabeled_draw)
    assert simulation_dataset_id(moderate) == simulation_dataset_id(relabeled_view)
    assert simulation_dataset_id(moderate) != simulation_dataset_id(severe)
    assert biological_unit_id(moderate) == biological_unit_id(severe)
    assert moderate.independent_unit_id == severe.independent_unit_id
    assert moderate.dataset_id == simulation_dataset_id(moderate)


def test_scientific_request_identity_is_independent_of_output_destination(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    relocated = replace(
        request,
        output_path=(
            tmp_path / "relocated" / "dev" / "symsim" / "draw-01-moderate.h5ad"
        ),
    )

    assert simulation_scientific_identity(request) == simulation_scientific_identity(
        relocated
    )
    assert simulation_request_identity(request) == simulation_scientific_identity(
        request
    )
    assert "output_path" not in simulation_scientific_identity(request)
    validate_simulation_request(relocated, PROTOCOL)

    invalid_destination = replace(
        request,
        output_path=tmp_path / "relocated" / "missing-namespace.h5ad",
    )
    with pytest.raises(SimulationContractError, match="output_path"):
        validate_simulation_request(invalid_destination, PROTOCOL)


def test_paired_views_share_one_biological_unit_and_truth_seed(tmp_path: Path) -> None:
    moderate = _request(tmp_path)
    severe = replace(
        moderate,
        technical_view="severe",
        measurement_seed=303,
        output_path=tmp_path / "dev/symsim/draw-01-severe.h5ad",
    )

    validate_paired_simulation_requests((moderate, severe), PROTOCOL)

    assert moderate.independent_unit_id == severe.independent_unit_id
    assert moderate.dataset_id != severe.dataset_id


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"biological_id": "draw-02"}, "biological draw"),
        ({"biological_seed": 404}, "biological seed"),
        ({"technical_view": "moderate"}, "technical views"),
        ({"measurement_seed": 202}, "measurement seeds"),
        ({"mechanism": "sergio"}, "biological draw"),
    ],
)
def test_paired_views_cannot_claim_independence_or_different_truth(
    tmp_path: Path, changes: dict[str, object], message: str
) -> None:
    moderate = _request(tmp_path)
    severe_changes: dict[str, object] = {
        "technical_view": "severe",
        "measurement_seed": 303,
        "output_path": tmp_path / "dev/symsim/draw-01-severe.h5ad",
    }
    severe_changes.update(changes)
    severe = replace(moderate, **severe_changes)

    with pytest.raises(SimulationContractError, match=message):
        validate_paired_simulation_requests((moderate, severe), PROTOCOL)


def test_pair_contract_requires_exactly_two_views(tmp_path: Path) -> None:
    with pytest.raises(SimulationContractError, match="exactly two"):
        validate_paired_simulation_requests((_request(tmp_path),), PROTOCOL)


def test_final_claim_cannot_be_forged_by_direct_construction() -> None:
    with pytest.raises(TypeError):
        FinalManifestClaim(  # type: ignore[call-arg]
            round_id="round-001",
            generator_seeds=(1, 2),
            seed_manifest_sha256="0" * 64,
            execution_claim_id="1" * 32,
            round_dir=Path("/tmp/round-001"),
        )
    assert not hasattr(FinalManifestClaim, "_from_validated")


def test_fabricated_token_claim_without_running_registry_is_rejected(
    final_repo: tuple[Path, Path],
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    manifest = json.loads((round_dir / "final_manifest.json").read_text())
    materialization = json.loads((round_dir / "materialization.json").read_text())
    forged = object.__new__(FinalManifestClaim)
    object.__setattr__(forged, "round_id", round_dir.name)
    object.__setattr__(forged, "generator_seeds", tuple(manifest["generator_seeds"]))
    object.__setattr__(
        forged,
        "seed_manifest_sha256",
        materialization["seed_manifest_sha256"],
    )
    object.__setattr__(forged, "execution_claim_id", "1" * 32)
    object.__setattr__(forged, "round_dir", round_dir)
    object.__setattr__(forged, "_token", base_module._CLAIM_TOKEN)
    request = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=manifest["generator_seeds"][0],
        measurement_seed=manifest["generator_seeds"][1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=round_dir / "results/final/output.h5ad",
    )

    with pytest.raises(SimulationContractError, match="claim|running|repository"):
        validate_simulation_request(request, PROTOCOL, forged)


def test_claimed_final_manifest_validates_final_request_without_consuming_claim(
    final_repo: tuple[Path, Path],
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    execution_before = (round_dir / "execution_claim.json").read_bytes()
    manifest = json.loads((round_dir / "final_manifest.json").read_text())

    first = load_final_manifest_claim(repo, round_dir)
    second = load_final_manifest_claim(repo, round_dir)
    request = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=manifest["generator_seeds"][0],
        measurement_seed=manifest["generator_seeds"][1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=(
            round_dir / "results" / "final" / "symsim" / "draw-01-moderate.h5ad"
        ),
    )

    validate_simulation_request(request, PROTOCOL, first)

    assert first == second
    assert (round_dir / "execution_claim.json").read_bytes() == execution_before


def test_final_biological_id_cannot_exceed_protocol_draw_count(
    final_repo: tuple[Path, Path],
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    claim = load_final_manifest_claim(repo, round_dir)
    request = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-06",
        biological_seed=claim.generator_seeds[0],
        measurement_seed=claim.generator_seeds[1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=round_dir / "results/final/output.h5ad",
    )

    with pytest.raises(SimulationContractError, match="biological_id|draw"):
        validate_simulation_request(request, PROTOCOL, claim)


@pytest.mark.parametrize("terminal_state", ["superseded", "evaluated"])
def test_loaded_claim_is_rejected_after_round_leaves_running_state(
    final_repo: tuple[Path, Path], terminal_state: str
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    claim = load_final_manifest_claim(repo, round_dir)
    request = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=claim.generator_seeds[0],
        measurement_seed=claim.generator_seeds[1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=round_dir / "results/final/output.h5ad",
    )
    if terminal_state == "superseded":
        supersede_round(round_dir, "simulator contract regression")
    else:
        record_final_evaluation(round_dir, {}, repo=repo)

    with pytest.raises(SimulationContractError, match="claim|running|superseded"):
        validate_simulation_request(request, PROTOCOL, claim)


def test_final_claim_binds_exact_frozen_protocol_semantics(
    final_repo: tuple[Path, Path],
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    claim = load_final_manifest_claim(repo, round_dir)
    altered = replace(
        PROTOCOL,
        final=replace(PROTOCOL.final, cells=PROTOCOL.final.cells + 1),
    )
    request = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=claim.generator_seeds[0],
        measurement_seed=claim.generator_seeds[1],
        technical_view="moderate",
        cells=altered.final.cells,
        genes=altered.final.genes,
        output_path=round_dir / "results/final/output.h5ad",
    )

    with pytest.raises(SimulationContractError, match="frozen protocol"):
        validate_simulation_request(request, altered, claim)


def test_final_manifest_loader_requires_existing_execution_claim(
    final_repo: tuple[Path, Path],
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)

    with pytest.raises(SimulationContractError, match="claimed|running"):
        load_final_manifest_claim(repo, round_dir)


def test_final_manifest_loader_rejects_tampered_seed_manifest(
    final_repo: tuple[Path, Path],
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    path = round_dir / "final_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["generator_seeds"][0] = manifest["generator_seeds"][0] + 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SimulationContractError, match="manifest|integrity"):
        load_final_manifest_claim(repo, round_dir)


def test_final_manifest_loader_rechecks_repository_after_protocol_parse(
    final_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    protocol_path = repo / "protocol.json"
    real_load_protocol = base_module.load_protocol

    def parse_then_mutate(path: Path):  # type: ignore[no-untyped-def]
        parsed = real_load_protocol(path)
        protocol_path.write_text(
            protocol_path.read_text(encoding="utf-8").replace(
                '"cells": 2700', '"cells": 2701'
            ),
            encoding="utf-8",
        )
        return parsed

    monkeypatch.setattr(base_module, "load_protocol", parse_then_mutate)

    with pytest.raises(SimulationContractError, match="integrity|unchanged"):
        load_final_manifest_claim(repo, round_dir)


def test_final_manifest_loader_binds_journal_across_outer_snapshots(
    final_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    real_validate = study_module._validate_result_journal
    validations = 0

    def change_head_between_snapshots(*args, **kwargs):
        nonlocal validations
        validations += 1
        result = dict(real_validate(*args, **kwargs))
        if validations >= 3:
            result["head_sha256"] = "f" * 64
        return result

    monkeypatch.setattr(
        study_module, "_validate_result_journal", change_head_between_snapshots
    )

    with pytest.raises(
        SimulationContractError, match="journal.*changed|changed.*journal"
    ):
        load_final_manifest_claim(repo, round_dir)


def test_final_request_rejects_unsealed_seed_and_output_outside_round(
    final_repo: tuple[Path, Path], tmp_path: Path
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    claim = load_final_manifest_claim(repo, round_dir)
    seeds = claim.generator_seeds
    base = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=seeds[0],
        measurement_seed=seeds[1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=round_dir / "results/final/output.h5ad",
    )

    with pytest.raises(SimulationContractError, match="manifest"):
        validate_simulation_request(
            replace(base, biological_seed=max(seeds) + 1), PROTOCOL, claim
        )
    with pytest.raises(SimulationContractError, match="beneath"):
        validate_simulation_request(
            replace(base, output_path=tmp_path / "final/output.h5ad"),
            PROTOCOL,
            claim,
        )
    with pytest.raises(SimulationContractError, match="development"):
        validate_simulation_request(
            _request(tmp_path),
            PROTOCOL,
            claim,
        )
    with pytest.raises(SimulationContractError, match="output_path"):
        validate_simulation_request(
            replace(base, output_path="final/output.h5ad"),  # type: ignore[arg-type]
            PROTOCOL,
            claim,
        )


@pytest.mark.parametrize("symlink_root", [True, False])
def test_final_output_rejects_symlinked_results_path_components(
    final_repo: tuple[Path, Path], tmp_path: Path, symlink_root: bool
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    claim = load_final_manifest_claim(repo, round_dir)
    external = tmp_path / "external"
    external.mkdir()
    results = round_dir / "results"
    if symlink_root:
        results.symlink_to(external, target_is_directory=True)
    else:
        results.mkdir()
        (results / "real").mkdir()
        (results / "final").symlink_to("real", target_is_directory=True)
    request = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=claim.generator_seeds[0],
        measurement_seed=claim.generator_seeds[1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=results / "final/output.h5ad",
    )

    with pytest.raises(SimulationContractError, match="symlink|integrity|claim"):
        validate_simulation_request(request, PROTOCOL, claim)


def test_final_request_rechecks_claim_after_output_path_use(
    final_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    claim = load_final_manifest_claim(repo, round_dir)
    request = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=claim.generator_seeds[0],
        measurement_seed=claim.generator_seeds[1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=round_dir / "results/final/output.h5ad",
    )
    real_validate = base_module._validate_final_output_path

    def validate_then_supersede(
        current_request: SimulationRequest, current_claim: FinalManifestClaim
    ) -> None:
        real_validate(current_request, current_claim)
        supersede_round(round_dir, "transition during simulator validation")

    monkeypatch.setattr(
        base_module,
        "_validate_final_output_path",
        validate_then_supersede,
    )

    with pytest.raises(SimulationContractError, match="claim|running|superseded"):
        validate_simulation_request(request, PROTOCOL, claim)


def test_final_pair_rechecks_claim_after_pair_output_path_use(
    final_repo: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, round_dir = final_repo
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    claim = load_final_manifest_claim(repo, round_dir)
    moderate = SimulationRequest(
        mechanism="symsim",
        namespace="final",
        biological_id="draw-01",
        biological_seed=claim.generator_seeds[0],
        measurement_seed=claim.generator_seeds[1],
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=round_dir / "results/final/moderate.h5ad",
    )
    severe = replace(
        moderate,
        measurement_seed=claim.generator_seeds[2],
        technical_view="severe",
        output_path=round_dir / "results/final/severe.h5ad",
    )
    real_revalidate = base_module._revalidate_final_manifest_claim
    real_resolve = Path.resolve
    claim_checks = 0
    pair_paths_are_active = False
    transitioned = False

    def revalidate_then_arm(
        current_claim: FinalManifestClaim | None,
    ) -> FinalManifestClaim:
        nonlocal claim_checks, pair_paths_are_active
        validated = real_revalidate(current_claim)
        claim_checks += 1
        if claim_checks == 4:
            pair_paths_are_active = True
        return validated

    def resolve_then_supersede(self: Path, *args: object, **kwargs: object) -> Path:
        nonlocal pair_paths_are_active, transitioned
        resolved = real_resolve(self, *args, **kwargs)
        if pair_paths_are_active and not transitioned and self == moderate.output_path:
            pair_paths_are_active = False
            transitioned = True
            supersede_round(round_dir, "transition during pair path validation")
        return resolved

    monkeypatch.setattr(
        base_module,
        "_revalidate_final_manifest_claim",
        revalidate_then_arm,
    )
    monkeypatch.setattr(Path, "resolve", resolve_then_supersede)

    with pytest.raises(SimulationContractError, match="claim|running|superseded"):
        validate_paired_simulation_requests((moderate, severe), PROTOCOL, claim)
    assert transitioned


def test_paired_requests_reject_output_aliases_after_resolution(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    moderate = _request(
        tmp_path,
        output_path=real / "dev/symsim/shared.h5ad",
    )
    severe = replace(
        moderate,
        technical_view="severe",
        measurement_seed=303,
        output_path=alias / "dev/symsim/shared.h5ad",
    )

    with pytest.raises(SimulationContractError, match="output paths"):
        validate_paired_simulation_requests((moderate, severe), PROTOCOL)


def test_paired_requests_reject_hard_linked_output_destinations(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "dev/symsim"
    output_root.mkdir(parents=True)
    moderate_path = output_root / "moderate.h5ad"
    severe_path = output_root / "severe.h5ad"
    moderate_path.write_bytes(b"existing output")
    os.link(moderate_path, severe_path)
    moderate = _request(tmp_path, output_path=moderate_path)
    severe = replace(
        moderate,
        technical_view="severe",
        measurement_seed=303,
        output_path=severe_path,
    )

    with pytest.raises(SimulationContractError, match="output paths|alias|inode"):
        validate_paired_simulation_requests((moderate, severe), PROTOCOL)


def test_paired_output_identity_inspection_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "dev/symsim"
    output_root.mkdir(parents=True)
    moderate_path = output_root / "moderate.h5ad"
    severe_path = output_root / "severe.h5ad"
    moderate_path.write_bytes(b"moderate")
    severe_path.write_bytes(b"severe")
    moderate = _request(tmp_path, output_path=moderate_path)
    severe = replace(
        moderate,
        technical_view="severe",
        measurement_seed=303,
        output_path=severe_path,
    )
    real_lstat = Path.lstat

    def deny_output_inspection(self: Path) -> os.stat_result:
        if self == moderate_path:
            raise PermissionError("injected output inspection failure")
        return real_lstat(self)

    monkeypatch.setattr(Path, "lstat", deny_output_inspection)

    with pytest.raises(SimulationContractError, match="output.*inspect"):
        validate_paired_simulation_requests((moderate, severe), PROTOCOL)


def test_native_outputs_are_sealed_in_a_canonical_immutable_manifest(
    tmp_path: Path,
) -> None:
    first = tmp_path / "observed.csv"
    second = tmp_path / "truth.csv"
    first.write_bytes(b"1,0\n0,2\n")
    second.write_bytes(b"1,3\n4,2\n")
    metadata = {"software": "SymSim", "parameters": {"capture": 0.2}}

    forward = seal_native_outputs(
        {"native/observed.csv": first, "native/truth.csv": second}, metadata
    )
    reverse = seal_native_outputs(
        {"native/truth.csv": second, "native/observed.csv": first}, metadata
    )
    metadata["parameters"]["capture"] = 0.9

    assert isinstance(forward, NativeManifest)
    assert forward == reverse
    assert forward.manifest_sha256 == reverse.manifest_sha256
    assert [entry.path for entry in forward.files] == [
        "native/observed.csv",
        "native/truth.csv",
    ]
    assert forward.metadata == {
        "parameters": {"capture": 0.2},
        "software": "SymSim",
    }
    assert forward.as_dict()["manifest_sha256"] == forward.manifest_sha256
    with pytest.raises(FrozenInstanceError):
        forward.manifest_sha256 = "0" * 64  # type: ignore[misc]
    with pytest.raises(TypeError):
        NativeManifest(  # type: ignore[call-arg]
            schema_version=1,
            files=forward.files,
            _metadata_json=forward._metadata_json,
            manifest_sha256=forward.manifest_sha256,
        )


@pytest.mark.parametrize(
    ("logical_path", "message"),
    [
        ("/absolute.txt", "relative"),
        ("../escape.txt", "relative"),
        ("native/../escape.txt", "relative"),
        ("native\\file.txt", "POSIX"),
        ("", "relative"),
    ],
)
def test_native_logical_paths_are_strictly_relative(
    tmp_path: Path, logical_path: str, message: str
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("stable", encoding="utf-8")

    with pytest.raises(SimulationContractError, match=message):
        seal_native_outputs({logical_path: output}, {"software": "fixture"})


@pytest.mark.parametrize(
    "logical_paths",
    [
        ("native/Result.txt", "native/result.txt"),
        (
            "native/Caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt",
            "native/Cafe\N{COMBINING ACUTE ACCENT}.txt",
        ),
    ],
)
def test_native_logical_paths_reject_portable_unicode_collisions(
    tmp_path: Path, logical_paths: tuple[str, str]
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    with pytest.raises(SimulationContractError, match="collision"):
        seal_native_outputs(
            {logical_paths[0]: first, logical_paths[1]: second},
            {"software": "fixture"},
        )


def test_native_files_reject_duplicate_inodes_and_hardlinks(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    alias = tmp_path / "alias.txt"
    output.write_text("stable", encoding="utf-8")
    os.link(output, alias)

    with pytest.raises(SimulationContractError, match="hard link|duplicate"):
        seal_native_outputs(
            {"native/output.txt": output, "native/alias.txt": alias},
            {"software": "fixture"},
        )


def test_native_files_reject_two_names_for_the_same_file(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("stable", encoding="utf-8")

    with pytest.raises(SimulationContractError, match="duplicate inode"):
        seal_native_outputs(
            {"native/a.txt": output, "native/b.txt": output},
            {"software": "fixture"},
        )


def test_native_files_must_be_nonempty_path_mapping_of_regular_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(SimulationContractError, match="nonempty"):
        seal_native_outputs({}, {"software": "fixture"})
    with pytest.raises(SimulationContractError, match="pathlib.Path"):
        seal_native_outputs(
            {"native/output.txt": str(directory)},  # type: ignore[dict-item]
            {"software": "fixture"},
        )
    with pytest.raises(SimulationContractError, match="regular file"):
        seal_native_outputs({"native/output.txt": directory}, {"software": "fixture"})


def test_native_files_reject_symlinks_and_symlinked_parents(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    output = real / "output.txt"
    output.write_text("stable", encoding="utf-8")
    file_link = tmp_path / "file-link"
    file_link.symlink_to(output)
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real, target_is_directory=True)

    with pytest.raises(SimulationContractError, match="symlink"):
        seal_native_outputs({"native/output.txt": file_link}, {"software": "fixture"})
    with pytest.raises(SimulationContractError, match="symlink"):
        seal_native_outputs(
            {"native/output.txt": parent_link / "output.txt"},
            {"software": "fixture"},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "non-string key"},
        {"value": Path("not-json")},
    ],
)
def test_native_metadata_must_be_finite_canonical_json(
    tmp_path: Path, metadata: object
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("stable", encoding="utf-8")

    with pytest.raises(SimulationContractError, match="metadata"):
        seal_native_outputs({"native/output.txt": output}, metadata)  # type: ignore[arg-type]


def test_native_sealing_rejects_mutation_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.bin"
    output.write_bytes(b"x" * (2 * 1024 * 1024))
    real_read = native_module.os.read
    mutated = False

    def mutate_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        data = real_read(descriptor, size)
        if data and not mutated:
            mutated = True
            with output.open("ab") as destination:
                destination.write(b"changed")
                destination.flush()
                os.fsync(destination.fileno())
        return data

    monkeypatch.setattr(native_module.os, "read", mutate_after_first_read)

    with pytest.raises(SimulationContractError, match="changed while hashing"):
        seal_native_outputs({"native/output.bin": output}, {"software": "fixture"})


def test_native_sealing_detects_earlier_file_mutation_while_later_file_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    earlier = tmp_path / "a.bin"
    later = tmp_path / "z.bin"
    earlier.write_bytes(b"a" * 32)
    later.write_bytes(b"z" * 32)
    real_read = native_module.os.read
    later_reads = 0
    mutated = False

    def mutate_earlier_during_later_pass(descriptor: int, size: int) -> bytes:
        nonlocal later_reads, mutated
        data = real_read(descriptor, size)
        try:
            opened_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            return data
        if data and opened_path == later:
            later_reads += 1
            if later_reads == 2 and not mutated:
                mutated = True
                with earlier.open("ab") as destination:
                    destination.write(b"changed")
                    destination.flush()
                    os.fsync(destination.fileno())
        return data

    monkeypatch.setattr(native_module.os, "read", mutate_earlier_during_later_pass)

    with pytest.raises(SimulationContractError, match="changed while hashing"):
        seal_native_outputs(
            {"native/a.bin": earlier, "native/z.bin": later},
            {"software": "fixture"},
        )


def _artifact_request(tmp_path: Path, **changes: object) -> SimulationRequest:
    values: dict[str, object] = {
        "mechanism": "symsim",
        "namespace": "dev",
        "biological_id": "draw-01",
        "biological_seed": 101,
        "measurement_seed": 202,
        "technical_view": "moderate",
        "cells": 2,
        "genes": 2,
        "output_path": tmp_path / "dev/symsim/draw-01-moderate.h5ad",
    }
    values.update(changes)
    return SimulationRequest(**values)


def _request_identity(request: SimulationRequest) -> dict[str, object]:
    return simulation_scientific_identity(request)


def _seal_for_request(
    output: Path,
    request: SimulationRequest,
    *,
    request_identity: dict[str, object] | None = None,
) -> NativeManifest:
    return seal_native_outputs(
        {"native/output.txt": output},
        {
            "software": "SymSim",
            "simulation_request": (
                _request_identity(request)
                if request_identity is None
                else request_identity
            ),
        },
    )


def _truth_dataset(
    request: SimulationRequest, native_manifest_sha256: str
) -> ad.AnnData:
    observed = np.asarray([[0, 2], [1, 0]], dtype=np.int64)
    truth = np.asarray([[1, 2], [1, 3]], dtype=np.int64)
    obs = pd.DataFrame(
        {
            "dataset_id": [request.dataset_id] * 2,
            "mechanism": [request.mechanism] * 2,
            "condition": [request.technical_view] * 2,
            "biological_id": [request.biological_id] * 2,
            "technical_view": [request.technical_view] * 2,
            "draw": [1, 1],
            "library_size": [2, 1],
        },
        index=["cell-01", "cell-02"],
    )
    adata = ad.AnnData(
        X=observed,
        obs=obs,
        var=pd.DataFrame(index=["gene-01", "gene-02"]),
        layers={"pre_capture_counts": truth},
    )
    adata.uns.update(
        {
            "truth_kind": "exact_pre_capture",
            "primary_truth_layer": "pre_capture_counts",
            "provenance": {
                "source": "SymSim",
                "source_sha256": "a" * 64,
                "software": "SymSim",
                "software_version": "76a674b",
                "parameters": {
                    "native_manifest_sha256": native_manifest_sha256,
                },
                "seeds": {
                    "biological": request.biological_seed,
                    "measurement": request.measurement_seed,
                },
            },
        }
    )
    return adata


def test_simulation_artifact_is_frozen_and_binds_validated_translation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    dataset_hash = benchmark_dataset_sha256(adata)

    artifact = SimulationArtifact(request, adata, manifest, dataset_hash)

    assert artifact.request == request
    assert artifact.dataset_sha256 == dataset_hash
    assert artifact.native_manifest.manifest_sha256 == manifest.manifest_sha256
    original_value = int(artifact.adata.X[0, 0])
    adata.X[0, 0] = 99
    assert int(artifact.adata.X[0, 0]) == original_value
    exposed = artifact.adata
    exposed.X[0, 0] = 88
    assert int(artifact.adata.X[0, 0]) == original_value
    assert benchmark_dataset_sha256(artifact.adata) == dataset_hash
    with pytest.raises(FrozenInstanceError):
        artifact.dataset_sha256 = "0" * 64  # type: ignore[misc]


def test_simulation_artifact_scientific_binding_survives_relocation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    relocated = replace(
        request,
        output_path=(
            tmp_path / "relocated" / "dev" / "symsim" / "draw-01-moderate.h5ad"
        ),
    )
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    dataset_hash = benchmark_dataset_sha256(adata)

    original = SimulationArtifact(request, adata, manifest, dataset_hash)
    moved = SimulationArtifact(relocated, adata, manifest, dataset_hash)

    assert moved.native_manifest.manifest_sha256 == (
        original.native_manifest.manifest_sha256
    )
    assert moved.dataset_sha256 == original.dataset_sha256


def test_simulation_artifact_rejects_missing_native_manifest_binding(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, "b" * 64)

    with pytest.raises(SimulationContractError, match="native manifest"):
        SimulationArtifact(request, adata, manifest, benchmark_dataset_sha256(adata))


def test_simulation_artifact_rejects_dataset_hash_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)

    with pytest.raises(SimulationContractError, match="dataset_sha256"):
        SimulationArtifact(request, adata, manifest, "0" * 64)


def test_simulation_artifact_rejects_forged_native_manifest(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    forged = replace(manifest, manifest_sha256="0" * 64)
    adata = _truth_dataset(request, "0" * 64)

    with pytest.raises(SimulationContractError, match="native manifest"):
        SimulationArtifact(request, adata, forged, benchmark_dataset_sha256(adata))


def test_simulation_artifact_rechecks_native_bytes_after_translation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native before translation", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    dataset_hash = benchmark_dataset_sha256(adata)
    output.write_text("changed during translation", encoding="utf-8")

    with pytest.raises(SimulationContractError, match="changed.*seal|native file"):
        SimulationArtifact(request, adata, manifest, dataset_hash)


def test_simulation_artifact_rejects_schema_invalid_truth_dataset(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    del adata.obs["biological_id"]

    with pytest.raises(SimulationContractError, match="translated AnnData"):
        SimulationArtifact(request, adata, manifest, "0" * 64)


def test_simulation_artifact_rejects_nonfinite_translation(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    adata.X = np.asarray([[np.nan, 2.0], [1.0, 0.0]])

    with pytest.raises(SimulationContractError, match="translated AnnData"):
        SimulationArtifact(request, adata, manifest, "0" * 64)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("shape", "shape"),
        ("dataset_id", "dataset_id"),
        ("mechanism", "mechanism"),
        ("condition", "condition"),
        ("biological_id", "biological_id"),
        ("technical_view", "technical_view"),
        ("draw", "draw"),
        ("biological_seed", "biological seed"),
        ("measurement_seed", "measurement seed"),
    ],
)
def test_simulation_artifact_binds_request_end_to_end(
    tmp_path: Path, mismatch: str, message: str
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    if mismatch == "shape":
        request = replace(request, cells=3)
        manifest = _seal_for_request(output, request)
        adata.uns["provenance"]["parameters"]["native_manifest_sha256"] = (
            manifest.manifest_sha256
        )
    elif mismatch in {
        "dataset_id",
        "mechanism",
        "condition",
        "biological_id",
        "technical_view",
    }:
        adata.obs[mismatch] = ["wrong-value"] * adata.n_obs
    elif mismatch == "draw":
        adata.obs["draw"] = [999] * adata.n_obs
    else:
        seed_name = mismatch.removesuffix("_seed")
        adata.uns["provenance"]["seeds"][seed_name] += 1
    dataset_hash = benchmark_dataset_sha256(adata)

    with pytest.raises(SimulationContractError, match=message):
        SimulationArtifact(request, adata, manifest, dataset_hash)


def test_simulation_artifact_binds_native_request_identity(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    wrong_identity = _request_identity(request)
    wrong_identity["measurement_seed"] = request.measurement_seed + 1
    manifest = _seal_for_request(
        output,
        request,
        request_identity=wrong_identity,
    )
    adata = _truth_dataset(request, manifest.manifest_sha256)

    with pytest.raises(SimulationContractError, match="request identity"):
        SimulationArtifact(
            request,
            adata,
            manifest,
            benchmark_dataset_sha256(adata),
        )


def test_simulation_artifact_rejects_boolean_seed_request_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path, biological_seed=1)
    confused_identity = _request_identity(request)
    confused_identity["biological_seed"] = True
    manifest = _seal_for_request(
        output,
        request,
        request_identity=confused_identity,
    )
    adata = _truth_dataset(request, manifest.manifest_sha256)

    with pytest.raises(SimulationContractError, match="request identity"):
        SimulationArtifact(
            request,
            adata,
            manifest,
            benchmark_dataset_sha256(adata),
        )


@pytest.mark.parametrize(
    "field",
    ["biological_seed", "measurement_seed", "cells", "genes"],
)
def test_simulation_artifact_rejects_float_integer_request_identity(
    tmp_path: Path, field: str
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    confused_identity = _request_identity(request)
    confused_identity[field] = float(confused_identity[field])
    manifest = _seal_for_request(
        output,
        request,
        request_identity=confused_identity,
    )
    adata = _truth_dataset(request, manifest.manifest_sha256)

    with pytest.raises(SimulationContractError, match="request identity"):
        SimulationArtifact(
            request,
            adata,
            manifest,
            benchmark_dataset_sha256(adata),
        )


def test_artifact_constructor_revalidates_native_after_copy_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    dataset_hash = benchmark_dataset_sha256(adata)
    real_copy = ad.AnnData.copy
    mutated = False

    def copy_and_mutate(
        self: ad.AnnData, *args: object, **kwargs: object
    ) -> ad.AnnData:
        nonlocal mutated
        snapshot = real_copy(self, *args, **kwargs)
        if not mutated:
            mutated = True
            output.write_text("mutated by copy", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(ad.AnnData, "copy", copy_and_mutate)

    with pytest.raises(SimulationContractError, match="native file|changed"):
        SimulationArtifact(request, adata, manifest, dataset_hash)


def test_artifact_access_revalidates_native_after_copy_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output.txt"
    output.write_text("native", encoding="utf-8")
    request = _artifact_request(tmp_path)
    manifest = _seal_for_request(output, request)
    adata = _truth_dataset(request, manifest.manifest_sha256)
    artifact = SimulationArtifact(
        request,
        adata,
        manifest,
        benchmark_dataset_sha256(adata),
    )
    real_copy = ad.AnnData.copy
    mutated = False

    def copy_and_mutate(
        self: ad.AnnData, *args: object, **kwargs: object
    ) -> ad.AnnData:
        nonlocal mutated
        snapshot = real_copy(self, *args, **kwargs)
        if not mutated:
            mutated = True
            output.write_text("mutated by access", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(ad.AnnData, "copy", copy_and_mutate)

    with pytest.raises(SimulationContractError, match="native file|changed"):
        _ = artifact.adata
