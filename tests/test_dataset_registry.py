from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.datasets as datasets_module
import maskimpute_benchmark.simulators.runtime_assets as runtime_assets_module
import maskimpute_benchmark.study as study_module
from maskimpute_benchmark.datasets import (
    DatasetRegistryError,
    generate_dataset_panel,
    load_development_panel,
    validate_dataset_status,
)
from maskimpute_benchmark.protocol import canonical_sha256, file_sha256, load_protocol
from maskimpute_benchmark.schema import benchmark_dataset_sha256
from maskimpute_benchmark.simulators import (
    SimulationArtifact,
    SimulationRequest,
    load_final_manifest_claim,
    run_sparsim_pair,
    run_symsim_pair,
    seal_native_outputs,
    simulation_scientific_identity,
    validate_paired_simulation_requests,
)
from maskimpute_benchmark.study import (
    assert_final_runnable,
    freeze_round,
    materialize_final,
    record_incremental_results,
)


MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
VIEWS = ("moderate", "severe")


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _protocol_payload() -> dict[str, object]:
    payload = json.loads(Path("study/protocol.json").read_text(encoding="utf-8"))
    payload["development"]["cells"] = 4
    payload["development"]["genes"] = 3
    payload["final"]["cells"] = 4
    payload["final"]["genes"] = 3
    return payload


def _panel_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "development_only",
        "namespace": "dev",
        "mechanisms": list(MECHANISMS),
        "technical_views": list(VIEWS),
        "draws_per_mechanism": 2,
        "cells": 4,
        "genes": 3,
        "seed_derivation": {
            "algorithm": "sha256-domain-separated-63bit-v1",
            "master_seed": 410184288765510201,
        },
    }


@pytest.fixture
def panel_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "study").mkdir(parents=True)
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repo / "study/protocol.json").write_text(
        json.dumps(_protocol_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (repo / "study/development_panel.json").write_text(
        json.dumps(_panel_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name in ("sources.json", "simulator_runtime_assets.json"):
        (repo / f"study/{name}").write_bytes(Path(f"study/{name}").read_bytes())
    _git(repo, "init")
    _git(repo, "config", "user.name", "Dataset Registry Test")
    _git(repo, "config", "user.email", "registry@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "dataset registry fixture")
    return repo


def _fake_dataset(request: SimulationRequest) -> ad.AnnData:
    biological_value = float(request.biological_seed % 97 + 1)
    measurement_value = int(request.measurement_seed % 13 + 1)
    truth = np.full((request.cells, request.genes), biological_value, dtype=np.float64)
    observed = np.full(
        (request.cells, request.genes), measurement_value, dtype=np.uint64
    )
    obs = pd.DataFrame(
        {
            "dataset_id": [request.dataset_id] * request.cells,
            "mechanism": [request.mechanism] * request.cells,
            "condition": [request.technical_view] * request.cells,
            "biological_id": [request.biological_id] * request.cells,
            "technical_view": [request.technical_view] * request.cells,
            "draw": [int(request.biological_id.removeprefix("draw-"))] * request.cells,
            "library_size": [measurement_value * request.genes] * request.cells,
        },
        index=[f"cell-{index:03d}" for index in range(request.cells)],
    )
    var = pd.DataFrame(index=[f"gene-{index:03d}" for index in range(request.genes)])
    dataset = ad.AnnData(
        X=observed,
        obs=obs,
        var=var,
        layers={"latent_expression": truth},
    )
    dataset.uns.update(
        {
            "truth_kind": "exact_continuous",
            "primary_truth_layer": "latent_expression",
            "allowed_covariates": {"obs": [], "var": []},
            "provenance": {
                "source": f"fixture://{request.mechanism}",
                "source_sha256": canonical_sha256(
                    {"fixture_source": request.mechanism}
                ),
                "software": "registry-fixture",
                "software_version": "1",
                "parameters": {},
                "seeds": {
                    "biological": request.biological_seed,
                    "measurement": request.measurement_seed,
                },
            },
        }
    )
    return dataset


def _fake_adapter_factory(
    calls: list[tuple[tuple[SimulationRequest, ...], object]],
    *,
    fail: tuple[str, str] | None = None,
):
    def adapter(requests, protocol, final_manifest=None, runtime_assets=None):
        ordered = tuple(requests)
        calls.append((ordered, final_manifest))
        if fail == (ordered[0].mechanism, ordered[0].biological_id):
            raise RuntimeError("deliberate adapter failure")
        artifacts: list[SimulationArtifact] = []
        for request in ordered:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            native_path = request.output_path.parent / (
                f"fixture-{request.technical_view}.native"
            )
            native_path.write_text(f"{request.dataset_id}\n", encoding="utf-8")
            manifest = seal_native_outputs(
                {"fixture.native": native_path},
                {
                    "adapter_schema": "registry-fixture-v1",
                    "simulation_request": simulation_scientific_identity(request),
                },
            )
            dataset = _fake_dataset(request)
            dataset.uns["provenance"]["parameters"] = {
                "native_manifest_sha256": manifest.manifest_sha256
            }
            dataset.write_h5ad(request.output_path)
            persisted = ad.read_h5ad(request.output_path)
            artifacts.append(
                SimulationArtifact(
                    request,
                    persisted,
                    manifest,
                    benchmark_dataset_sha256(persisted),
                )
            )
        return tuple(artifacts)

    return adapter


def _install_fake_adapters(monkeypatch: pytest.MonkeyPatch, adapter) -> None:
    monkeypatch.setattr(
        datasets_module,
        "_ADAPTERS",
        {mechanism: adapter for mechanism in MECHANISMS},
    )


def _fake_final_runtime_paths(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    external_root = tmp_path / "publication-assets"
    r_environment = tmp_path / "publication-r"
    external_root.mkdir(exist_ok=True)
    (r_environment / "bin").mkdir(parents=True, exist_ok=True)
    (r_environment / "bin/Rscript").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime_assets_module,
        "_collect_source_receipts",
        lambda _ledger, _root: (
            "5a6f60c5de980a20eb118d0b82913112650f1956562aec4c92d37d8314c9f29e",
            ({"source_id": "fixture", "sha256": "a" * 64},),
        ),
    )
    authority = json.loads(
        (repo / "study/simulator_runtime_assets.json").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        runtime_assets_module,
        "_r_environment_receipt",
        lambda _repository, _path, _authority: {
            "environment_id": "simulator-r",
            "inventory_sha256": "e" * 64,
            "lock_file_sha256": authority["r_environment"]["lock_file_sha256"],
            "schema": "maskimpute-simulator-r-runtime-receipt-v1",
        },
    )
    monkeypatch.setattr(
        runtime_assets_module,
        "_directory_content_receipt",
        lambda _path: {
            "entry_count": authority["r_environment"]["tree_entry_count"],
            "sha256": authority["r_environment"]["tree_sha256"],
        },
    )
    monkeypatch.setattr(
        runtime_assets_module,
        "_source_snapshot_content_receipt",
        lambda _path: {
            "entry_count": authority["source_snapshot"]["tree_entry_count"],
            "sha256": authority["source_snapshot"]["tree_sha256"],
        },
    )
    return {
        "simulator_assets_root": external_root,
        "simulator_r_environment": r_environment,
    }


def _generate_dev(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail: tuple[str, str] | None = None,
):
    calls: list[tuple[tuple[SimulationRequest, ...], object]] = []
    _install_fake_adapters(monkeypatch, _fake_adapter_factory(calls, fail=fail))
    status = generate_dataset_panel(repo=repo, namespace="dev")
    return status, calls


def test_tracked_development_panel_is_exact_and_development_only() -> None:
    protocol = load_protocol(Path("study/protocol.json"))

    panel = load_development_panel(Path("study/development_panel.json"), protocol)

    assert panel.namespace == protocol.development.namespace
    assert panel.role == "development_only"
    assert panel.mechanisms == MECHANISMS
    assert panel.technical_views == VIEWS
    assert panel.draws_per_mechanism == 2
    assert panel.cells == protocol.development.cells
    assert panel.genes == protocol.development.genes


def test_development_receipts_retain_the_existing_path_free_schema(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _calls = _generate_dev(panel_repo, monkeypatch)
    receipt = json.loads(
        (
            panel_repo
            / "artifacts/study/development/results/receipts/symsim/draw-01.json"
        ).read_text(encoding="utf-8")
    )

    assert "runtime_assets_sha256" not in status
    assert "runtime_assets_receipt" not in status
    assert "runtime_assets_sha256" not in receipt


def test_development_generation_has_exact_design_and_seed_cardinality(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, calls = _generate_dev(panel_repo, monkeypatch)

    assert status["namespace"] == "dev"
    assert status["status"] == "completed"
    assert status["independent_unit_count"] == 8
    assert status["completed_count"] == 16
    assert status["failed_count"] == 0
    assert len(status["rows"]) == 16
    assert len(calls) == 8
    all_requests = [request for pair, _claim in calls for request in pair]
    assert {request.mechanism for request in all_requests} == set(MECHANISMS)
    assert {request.technical_view for request in all_requests} == set(VIEWS)
    assert all(claim is None for _pair, claim in calls)
    seeds = {
        seed
        for request in all_requests
        for seed in (request.biological_seed, request.measurement_seed)
    }
    biological_seeds = {request.biological_seed for request in all_requests}
    measurement_seeds = {request.measurement_seed for request in all_requests}
    assert len(seeds) == 24
    assert len(biological_seeds) == 8
    assert len(measurement_seeds) == 16
    assert biological_seeds.isdisjoint(measurement_seeds)
    assert all(
        pair[0].biological_seed == pair[1].biological_seed
        and pair[0].measurement_seed != pair[1].measurement_seed
        and {request.technical_view for request in pair} == set(VIEWS)
        for pair, _claim in calls
    )
    assert all(
        "seed" not in key or key.endswith("commitment")
        for row in status["rows"]
        for key in row
    )
    assert all(row["status"] == "completed" for row in status["rows"])
    for offset in range(0, len(status["rows"]), 2):
        moderate, severe = status["rows"][offset : offset + 2]
        assert moderate["technical_view"] == "moderate"
        assert severe["technical_view"] == "severe"
        assert moderate["truth_sha256"] == severe["truth_sha256"]
        assert (
            moderate["biological_seed_commitment"]
            == (severe["biological_seed_commitment"])
        )
        assert (
            moderate["measurement_seed_commitment"]
            != (severe["measurement_seed_commitment"])
        )
        assert moderate["independent_unit_id"] == severe["independent_unit_id"]


def test_failed_adapter_keeps_two_explicit_rows_and_other_pairs_continue(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, calls = _generate_dev(panel_repo, monkeypatch, fail=("sparsim", "draw-02"))

    assert len(calls) == 8
    assert status["status"] == "failed"
    assert status["completed_count"] == 14
    assert status["failed_count"] == 2
    failed = [row for row in status["rows"] if row["status"] == "failed"]
    assert [
        (row["mechanism"], row["biological_id"], row["technical_view"])
        for row in failed
    ] == [
        ("sparsim", "draw-02", "moderate"),
        ("sparsim", "draw-02", "severe"),
    ]
    assert all(row["reason"] == "adapter_failed:RuntimeError" for row in failed)
    assert all(row["dataset_sha256"] is None for row in failed)
    assert all(row["truth_sha256"] is None for row in failed)
    assert all(row["log_sha256"] is not None for row in failed)


def test_resume_revalidates_receipts_and_never_calls_adapters_again(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _calls = _generate_dev(panel_repo, monkeypatch)

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("resume attempted to invoke an adapter")

    _install_fake_adapters(monkeypatch, forbidden_adapter)
    second = generate_dataset_panel(repo=panel_repo, namespace="dev")

    assert second == first

    output = (
        panel_repo
        / "artifacts/study/development/results"
        / first["rows"][0]["output_path"]
    )
    output.write_bytes(b"tampered")
    with pytest.raises(DatasetRegistryError, match="checksum|dataset|output"):
        generate_dataset_panel(repo=panel_repo, namespace="dev")


def test_resume_rejects_symlinked_output_parent(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _calls = _generate_dev(panel_repo, monkeypatch)
    results_root = panel_repo / "artifacts/study/development/results"
    output = results_root / status["rows"][0]["output_path"]
    original_parent = output.parent
    outside = panel_repo.parent / "aliased-draw"
    original_parent.rename(outside)
    original_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DatasetRegistryError, match="symlink"):
        validate_dataset_status(results_root / "dataset_status.json", repo=panel_repo)


def _prepare_final_round(repo: Path, *, seed_count: int = 60) -> Path:
    (repo / "config.json").write_text('{"method":"fixture"}\n', encoding="utf-8")
    (repo / "environment.lock").write_text("python=3.11\n", encoding="utf-8")
    _git(repo, "add", "config.json", "environment.lock")
    _git(repo, "commit", "-m", "freeze inputs")
    round_dir = repo / "artifacts/study/round-001"
    freeze_round(
        repo,
        round_dir,
        repo / "config.json",
        repo / "study/protocol.json",
        environment_path=repo / "environment.lock",
    )
    materialize_final(round_dir, seed_count=seed_count, repo=repo)
    return round_dir


def test_final_requires_claim_and_exact_seed_cardinality_before_adapter_access(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = _prepare_final_round(panel_repo)

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("adapter accessed before claim validation")

    _install_fake_adapters(monkeypatch, forbidden_adapter)
    with pytest.raises(DatasetRegistryError, match="claim|running"):
        generate_dataset_panel(repo=panel_repo, namespace="final", round_dir=round_dir)


def test_final_generation_consumes_only_claimed_seeds_and_writes_under_results(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    round_dir = _prepare_final_round(panel_repo)
    assert_final_runnable(panel_repo, round_dir)
    calls: list[tuple[tuple[SimulationRequest, ...], object]] = []
    received_assets: list[object] = []
    fake = _fake_adapter_factory(calls)

    def adapter(requests, protocol, final_manifest=None, runtime_assets=None):
        received_assets.append(runtime_assets)
        return fake(requests, protocol, final_manifest, runtime_assets)

    _install_fake_adapters(monkeypatch, adapter)
    runtime_paths = _fake_final_runtime_paths(panel_repo, tmp_path, monkeypatch)
    external_root = runtime_paths["simulator_assets_root"]
    r_environment = runtime_paths["simulator_r_environment"]

    status = generate_dataset_panel(
        repo=panel_repo,
        namespace="final",
        round_dir=round_dir,
        simulator_assets_root=external_root,
        simulator_r_environment=r_environment,
    )

    manifest = json.loads(
        (round_dir / "final_manifest.json").read_text(encoding="utf-8")
    )
    claimed_seeds = set(manifest["generator_seeds"])
    requests = [request for pair, _claim in calls for request in pair]
    used_seeds = {
        seed
        for request in requests
        for seed in (request.biological_seed, request.measurement_seed)
    }
    assert len(calls) == 20
    assert len(requests) == 40
    assert len(used_seeds) == 60
    assert used_seeds == claimed_seeds
    assert all(claim is not None for _pair, claim in calls)
    assert all(asset is received_assets[0] for asset in received_assets)
    assert status["runtime_assets_sha256"] == received_assets[0].semantic_sha256
    assert status["runtime_assets_receipt"] == received_assets[0].semantic_receipt
    assert status["independent_unit_count"] == 20
    assert status["completed_count"] == 40
    assert all(
        request.output_path.is_relative_to(round_dir / "results")
        for request in requests
    )
    encoded = json.dumps(status, sort_keys=True)
    assert not any(str(seed) in encoded for seed in claimed_seeds)
    validated = validate_dataset_status(
        round_dir / "results/dataset_status.json",
        repo=panel_repo,
        round_dir=round_dir,
        simulator_assets_root=external_root,
        simulator_r_environment=r_environment,
    )
    assert validated == status
    pair_receipt = json.loads(
        (round_dir / "results/receipts/symsim/draw-01.json").read_text(encoding="utf-8")
    )
    assert pair_receipt["runtime_assets_sha256"] == status["runtime_assets_sha256"]
    assert not (panel_repo / "artifacts/external").exists()
    assert not (panel_repo / "artifacts/envs").exists()
    load_final_manifest_claim(panel_repo, round_dir)


def test_final_generation_requires_explicit_runtime_asset_paths(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = _prepare_final_round(panel_repo)
    assert_final_runnable(panel_repo, round_dir)
    _install_fake_adapters(monkeypatch, _fake_adapter_factory([]))

    with pytest.raises(DatasetRegistryError, match="runtime asset paths"):
        generate_dataset_panel(
            repo=panel_repo,
            namespace="final",
            round_dir=round_dir,
        )


def test_final_sequential_adapters_revalidate_claim_after_prior_publication(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    round_dir = _prepare_final_round(panel_repo)
    assert_final_runnable(panel_repo, round_dir)
    calls: list[tuple[tuple[SimulationRequest, ...], object]] = []
    fake = _fake_adapter_factory(calls)

    def strict_adapter(requests, protocol, final_manifest=None, runtime_assets=None):
        validate_paired_simulation_requests(requests, protocol, final_manifest)
        return fake(requests, protocol, final_manifest, runtime_assets)

    _install_fake_adapters(monkeypatch, strict_adapter)
    runtime_paths = _fake_final_runtime_paths(panel_repo, tmp_path, monkeypatch)

    status = generate_dataset_panel(
        repo=panel_repo,
        namespace="final",
        round_dir=round_dir,
        **runtime_paths,
    )

    assert status["status"] == "completed"
    assert status["completed_count"] == 40
    assert status["failed_count"] == 0


def test_two_real_adapter_contracts_run_sequentially_in_one_final_claim(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_payload = json.loads(
        (panel_repo / "study/protocol.json").read_text(encoding="utf-8")
    )
    protocol_payload["development"]["cells"] = 20
    protocol_payload["development"]["genes"] = 20
    protocol_payload["final"]["cells"] = 20
    protocol_payload["final"]["genes"] = 20
    (panel_repo / "study/protocol.json").write_text(
        json.dumps(protocol_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    panel_payload = _panel_payload()
    panel_payload["cells"] = 20
    panel_payload["genes"] = 20
    (panel_repo / "study/development_panel.json").write_text(
        json.dumps(panel_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _git(panel_repo, "add", "study/protocol.json", "study/development_panel.json")
    _git(panel_repo, "commit", "-m", "use tiny real-adapter final panel")
    round_dir = _prepare_final_round(panel_repo, seed_count=6)
    assert_final_runnable(panel_repo, round_dir)
    claim = load_final_manifest_claim(panel_repo, round_dir)
    protocol = load_protocol(panel_repo / "study/protocol.json")

    symsim_helpers = runpy.run_path("tests/test_symsim_adapter.py")
    sparsim_helpers = runpy.run_path("tests/test_sparsim_adapter.py")
    symsim_helpers["_mock_external"](monkeypatch)
    sparsim_helpers["_mock_external"](monkeypatch)
    monkeypatch.setattr(symsim_helpers["symsim_module"], "_REPO_ROOT", panel_repo)
    monkeypatch.setattr(sparsim_helpers["sparsim_module"], "_REPO_ROOT", panel_repo)

    def requests(mechanism: str, offset: int) -> tuple[SimulationRequest, ...]:
        base = round_dir / "results/final/datasets" / mechanism / "draw-01"
        return tuple(
            SimulationRequest(
                mechanism=mechanism,
                namespace="final",
                biological_id="draw-01",
                biological_seed=claim.generator_seeds[offset],
                measurement_seed=claim.generator_seeds[offset + index + 1],
                technical_view=view,
                cells=20,
                genes=20,
                output_path=base / f"{view}.h5ad",
            )
            for index, view in enumerate(VIEWS)
        )

    symsim_artifacts = run_symsim_pair(requests("symsim", 0), protocol, claim)
    assert len(symsim_artifacts) == 2
    first_files = sorted(
        path for path in (round_dir / "results").rglob("*") if path.is_file()
    )
    record_incremental_results(
        round_dir,
        {
            "result_files": [
                {
                    "path": path.relative_to(round_dir).as_posix(),
                    "sha256": file_sha256(path),
                }
                for path in first_files
            ]
        },
        repo=panel_repo,
    )

    sparsim_artifacts = run_sparsim_pair(requests("sparsim", 3), protocol, claim)

    assert len(sparsim_artifacts) == 2
    assert all(artifact.request.output_path.is_file() for artifact in sparsim_artifacts)


def test_status_rejects_missing_rows_duplicate_hashes_and_output_aliases(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _calls = _generate_dev(panel_repo, monkeypatch)
    status_path = panel_repo / "artifacts/study/development/results/dataset_status.json"

    missing = json.loads(json.dumps(status))
    missing["rows"].pop()
    missing["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in missing.items() if key != "manifest_sha256"}
    )
    status_path.write_text(
        json.dumps(missing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(DatasetRegistryError, match="row|cardinality|complete"):
        validate_dataset_status(status_path, repo=panel_repo)

    duplicate = json.loads(json.dumps(status))
    duplicate["rows"][1]["dataset_sha256"] = duplicate["rows"][0]["dataset_sha256"]
    duplicate["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in duplicate.items() if key != "manifest_sha256"}
    )
    status_path.write_text(
        json.dumps(duplicate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(DatasetRegistryError, match="duplicate.*checksum"):
        validate_dataset_status(status_path, repo=panel_repo)

    alias = json.loads(json.dumps(status))
    alias["rows"][1]["output_path"] = alias["rows"][0]["output_path"]
    alias["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in alias.items() if key != "manifest_sha256"}
    )
    status_path.write_text(
        json.dumps(alias, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(DatasetRegistryError, match="output.*alias|duplicate.*output"):
        validate_dataset_status(status_path, repo=panel_repo)


def test_status_rejects_noncanonical_json_bytes(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status, _calls = _generate_dev(panel_repo, monkeypatch)
    status_path = panel_repo / "artifacts/study/development/results/dataset_status.json"
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")

    with pytest.raises(DatasetRegistryError, match="not canonical JSON"):
        validate_dataset_status(status_path, repo=panel_repo)


def test_final_rejects_wrong_seed_count(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    round_dir = _prepare_final_round(panel_repo, seed_count=59)
    assert_final_runnable(panel_repo, round_dir)

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("adapter accessed after seed cardinality failure")

    _install_fake_adapters(monkeypatch, forbidden_adapter)
    runtime_paths = _fake_final_runtime_paths(panel_repo, tmp_path, monkeypatch)
    with pytest.raises(DatasetRegistryError, match="exactly 60"):
        generate_dataset_panel(
            repo=panel_repo,
            namespace="final",
            round_dir=round_dir,
            **runtime_paths,
        )


def test_final_rejects_a_claimed_seed_that_collides_with_development(
    panel_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = load_protocol(panel_repo / "study/protocol.json")
    panel = load_development_panel(
        panel_repo / "study/development_panel.json", protocol
    )
    development_seeds = datasets_module._development_seed_set(panel)
    collision = min(development_seeds)
    noncolliding = [seed for seed in range(1, 10_000) if seed not in development_seeds][
        :59
    ]
    materialized = iter([collision, *noncolliding])
    monkeypatch.setattr(
        study_module.secrets, "randbits", lambda _bits: next(materialized)
    )
    round_dir = _prepare_final_round(panel_repo, seed_count=60)
    assert_final_runnable(panel_repo, round_dir)

    def forbidden_adapter(*_args, **_kwargs):
        raise AssertionError("adapter accessed after namespace collision")

    _install_fake_adapters(monkeypatch, forbidden_adapter)
    runtime_paths = _fake_final_runtime_paths(panel_repo, tmp_path, monkeypatch)
    with pytest.raises(DatasetRegistryError, match="collides with development"):
        generate_dataset_panel(
            repo=panel_repo,
            namespace="final",
            round_dir=round_dir,
            **runtime_paths,
        )


def test_cli_exposes_no_seed_or_dimension_overrides() -> None:
    script = Path("scripts/generate_study_datasets.py").read_text(encoding="utf-8")

    for forbidden in ("--seed", "--cells", "--genes", "--draws", "--mechanism"):
        assert forbidden not in script
    assert "--simulator-assets-root" in script
    assert "--simulator-r-environment" in script
