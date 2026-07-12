from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest

from maskimpute_benchmark.protocol import load_protocol
from maskimpute_benchmark.schema import benchmark_dataset_sha256
from maskimpute_benchmark.simulators.base import (
    SimulationContractError,
    SimulationRequest,
)
import maskimpute_benchmark.simulators.sparsim as sparsim_module
from maskimpute_benchmark.simulators.sparsim import (
    map_sparsim_r_seeds,
    run_sparsim_pair,
)


PROTOCOL = load_protocol(Path("study/protocol.json"))
SMOKE_PROTOCOL = replace(
    PROTOCOL,
    development=replace(PROTOCOL.development, cells=20, genes=20),
)
SOURCE_RECEIPT = {
    "citation_doi": "10.1093/bioinformatics/btz752",
    "ledger_sha256": "a" * 64,
    "license": "GPL-3.0-only",
    "resolved_revision": "4e7712fb236a92ce7c173da169c8a29cc2a9f0ef",
    "revision": "4e7712fb236a92ce7c173da169c8a29cc2a9f0ef",
    "role": "mechanism",
    "schema_version": 1,
    "source_id": "sparsim",
    "source_type": "git",
    "source_url": "https://gitlab.com/sysbiobig/sparsim.git",
    "verified_checksum": {
        "algorithm": "git-tree-sha1",
        "value": "5d66b28cc6afd8d68364f4205cc983c7f681e2fe",
    },
}
ENVIRONMENT_RECEIPT = {
    "schema": "maskimpute-sparsim-r-environment-v1",
    "sha256": "b" * 64,
    "r_executable_sha256": "c" * 64,
    "compiler": {
        "command": "/usr/bin/g++",
        "executable_sha256": "d" * 64,
        "version_sha256": "e" * 64,
    },
    "package_count": 42,
}
SOURCE_FILE_RECEIPT = {
    "cpp": {"path": "src/Random_number.cpp", "sha256": "1" * 64},
    "preset": {"path": "data/Chu_param_preset.RData", "sha256": "2" * 64},
    "simulate": {"path": "R/SPARSim_simulate.R", "sha256": "3" * 64},
    "utilities": {"path": "R/SPARSim_utilities.R", "sha256": "4" * 64},
}
NATIVE_FILES = {
    "cell_metadata.tsv",
    "config.json",
    "gene_metadata.tsv",
    "latent_expression.tsv",
    "observed_moderate.tsv",
    "observed_severe.tsv",
    "run_metadata.json",
}


with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


def _requests(tmp_path: Path) -> tuple[SimulationRequest, SimulationRequest]:
    moderate = SimulationRequest(
        mechanism="sparsim",
        namespace="dev",
        biological_id="draw-01",
        biological_seed=2**62 + 101,
        measurement_seed=2**61 + 202,
        technical_view="moderate",
        cells=20,
        genes=20,
        output_path=tmp_path / "dev/sparsim/draw-01-moderate.h5ad",
    )
    severe = replace(
        moderate,
        measurement_seed=2**61 + 303,
        technical_view="severe",
        output_path=tmp_path / "dev/sparsim/draw-01-severe.h5ad",
    )
    return moderate, severe


def _write_matrix(
    path: Path,
    row_ids: list[str],
    column_ids: list[str],
    values: np.ndarray,
    *,
    integer: bool,
) -> None:
    lines = ["\t".join(["gene_id", *column_ids])]
    for row_id, row in zip(row_ids, values, strict=True):
        formatted = (
            [str(int(value)) for value in row]
            if integer
            else [format(float(value), ".17g") for value in row]
        )
        lines.append("\t".join([row_id, *formatted]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_native(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cells = config["simulation"]["cells"]
    genes = config["simulation"]["genes"]
    cell_ids = [f"cell-{index:04d}" for index in range(1, cells + 1)]
    gene_ids = [f"gene-{index:04d}" for index in range(1, genes + 1)]
    latent = np.fromfunction(
        lambda gene, cell: (gene + 1.0) * 0.25 + (cell + 1.0) * 0.1,
        (genes, cells),
        dtype=float,
    ).astype(np.float64)
    moderate = np.fromfunction(
        lambda gene, cell: (3 * gene + cell + 2) % 7,
        (genes, cells),
        dtype=int,
    ).astype(np.int64)
    severe = np.fromfunction(
        lambda gene, cell: (gene + 2 * cell + 1) % 3,
        (genes, cells),
        dtype=int,
    ).astype(np.int64)
    _write_matrix(
        output_dir / "latent_expression.tsv",
        gene_ids,
        cell_ids,
        latent,
        integer=False,
    )
    _write_matrix(
        output_dir / "observed_moderate.tsv",
        gene_ids,
        cell_ids,
        moderate,
        integer=True,
    )
    gene_lines = ["gene_id\tsource_gene_id"]
    gene_lines.extend(
        f"{gene_id}\tSOURCE_{index}" for index, gene_id in enumerate(gene_ids, start=1)
    )
    (output_dir / "gene_metadata.tsv").write_text(
        "\n".join(gene_lines) + "\n", encoding="utf-8"
    )
    _write_matrix(
        output_dir / "observed_severe.tsv",
        gene_ids,
        cell_ids,
        severe,
        integer=True,
    )
    groups: list[str] = []
    for name, count in config["simulation"]["group_allocations"].items():
        groups.extend([name] * count)
    cell_lines = ["cell_id\tgroup"]
    cell_lines.extend(
        f"{cell_id}\t{group}" for cell_id, group in zip(cell_ids, groups, strict=True)
    )
    (output_dir / "cell_metadata.tsv").write_text(
        "\n".join(cell_lines) + "\n", encoding="utf-8"
    )
    array_sha256 = {
        name: _file_sha256(output_dir / name)
        for name in (
            "latent_expression.tsv",
            "observed_moderate.tsv",
            "observed_severe.tsv",
        )
    }
    metadata = {
        "array_sha256": array_sha256,
        "biological_seed_r": config["seeds"]["biological"]["mapped_r"],
        "cells": cells,
        "compiler_sha256": config["environment"]["compiler_executable_sha256"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "gene_matrix_equal": True,
        "genes": genes,
        "group_allocations": config["simulation"]["group_allocations"],
        "measurement_seeds_r": {
            view["technical_view"]: view["measurement_seed_r"]
            for view in config["views"]
        },
        "r_version": "R version fixture",
        "rcpp_version": "1.0.13",
        "schema_version": 1,
        "source_cpp_calls": 1,
        "sparsim_simulation_calls": 2,
        "views": [view["technical_view"] for view in config["views"]],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _mock_external(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: object | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        sparsim_module,
        "_verify_sparsim_source",
        lambda: json.loads(json.dumps(SOURCE_RECEIPT)),
    )
    monkeypatch.setattr(
        sparsim_module,
        "_environment_receipt",
        lambda: json.loads(json.dumps(ENVIRONMENT_RECEIPT)),
    )
    monkeypatch.setattr(
        sparsim_module,
        "_source_file_receipt",
        lambda: json.loads(json.dumps(SOURCE_FILE_RECEIPT)),
    )

    def fake_runner(
        config_path: Path, output_dir: Path, *, timeout_seconds: int
    ) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        calls.append({"config": config, "timeout_seconds": timeout_seconds})
        _write_valid_native(config_path, output_dir)
        if callable(mutate):
            mutate(output_dir)

    monkeypatch.setattr(sparsim_module, "_execute_sparsim", fake_runner)
    return calls


def _contains_forbidden_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_forbidden_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_path(item) for item in value)
    return isinstance(value, str) and (
        "/.worktrees/" in value or "artifacts/external/checkouts" in value
    )


def test_r_seed_mapping_is_deterministic_distinct_and_in_native_range() -> None:
    original = (2**62 + 101, 2**61 + 202, 2**61 + 303)

    first = map_sparsim_r_seeds(*original)
    second = map_sparsim_r_seeds(*original)

    assert first == second
    assert set(first) == {"biological", "moderate", "severe"}
    assert len(set(first.values())) == 3
    assert all(type(value) is int and 1 <= value < 2**31 for value in first.values())


def test_mocked_contract_does_not_read_an_unmocked_external_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    monkeypatch.setattr(sparsim_module, "_CHECKOUT", tmp_path / "missing-checkout")

    artifacts = run_sparsim_pair(_requests(tmp_path), SMOKE_PROTOCOL)

    assert len(artifacts) == 2
    assert len(calls) == 1


def test_mocked_pair_binds_chu_design_exact_truth_and_technical_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)

    artifacts = run_sparsim_pair(requests, SMOKE_PROTOCOL)

    assert len(calls) == 1
    config = calls[0]["config"]
    assert calls[0]["timeout_seconds"] == SMOKE_PROTOCOL.final_timeout_seconds
    assert config["simulation"] == {
        "cells": 20,
        "gene_selection": "sha256_ranked_source_gene_id_v1",
        "gene_selection_domain": "maskimpute-sparsim-gene-v1",
        "genes": 20,
        "group_allocations": {"chu-c1": 5, "chu-c3": 4, "chu-c6": 11},
        "group_presets": {
            "chu-c1": "Chu_C1",
            "chu-c3": "Chu_C3",
            "chu-c6": "Chu_C6",
        },
        "library_template_selection": "midpoint_quantile_with_replacement",
        "source_group_sizes": {"chu-c1": 92, "chu-c3": 66, "chu-c6": 188},
    }
    assert [view["technical_view"] for view in config["views"]] == [
        "moderate",
        "severe",
    ]
    assert config["views"][0]["library_size_divisor"] == 100
    assert config["views"][1]["library_size_divisor"] == 400
    assert config["seeds"]["biological"]["original"] == requests[0].biological_seed
    assert config["environment"] == {
        "compiler_executable_sha256": "d" * 64,
        "environment_sha256": "b" * 64,
    }
    assert not _contains_forbidden_path(config)
    for artifact, request in zip(artifacts, requests, strict=True):
        assert artifact.request == request
        assert request.output_path.is_file()
        dataset = artifact.adata
        assert dataset.shape == (20, 20)
        assert dataset.X.dtype == np.int64
        assert dataset.layers["latent_expression"].dtype == np.float64
        assert dataset.uns["truth_kind"] == "exact_continuous"
        assert dataset.uns["primary_truth_layer"] == "latent_expression"
        assert dataset.obs["group"].value_counts().to_dict() == {
            "chu-c6": 11,
            "chu-c1": 5,
            "chu-c3": 4,
        }
        assert "marker_chu_c1" in dataset.var
        assert dataset.var["source_gene_id"].tolist() == [
            f"SOURCE_{index}" for index in range(1, 21)
        ]
        provenance = dataset.uns["provenance"]
        assert provenance["software"] == "SPARSim"
        assert provenance["software_version"] == SOURCE_RECEIPT["resolved_revision"]
        assert provenance["parameters"]["source_receipt"] == SOURCE_RECEIPT
        assert provenance["parameters"]["environment"] == ENVIRONMENT_RECEIPT
        assert "runtime_seconds" not in repr(provenance)
        assert not _contains_forbidden_path(provenance)
        assert provenance["seeds"]["biological"] == request.biological_seed
        assert provenance["seeds"]["measurement"] == request.measurement_seed
        assert artifact.dataset_sha256 == benchmark_dataset_sha256(dataset)
        assert {entry.path for entry in artifact.native_manifest.files} == NATIVE_FILES
    np.testing.assert_array_equal(
        artifacts[0].adata.layers["latent_expression"],
        artifacts[1].adata.layers["latent_expression"],
    )
    assert not np.array_equal(artifacts[0].adata.X, artifacts[1].adata.X)


def test_deterministic_rerun_reuses_identical_native_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)

    first = run_sparsim_pair(requests, SMOKE_PROTOCOL)
    first_dataset_hashes = [artifact.dataset_sha256 for artifact in first]
    first_manifest_hashes = [
        artifact.native_manifest.manifest_sha256 for artifact in first
    ]
    native_directories = list((requests[0].output_path.parent / "native").iterdir())
    for request in requests:
        request.output_path.unlink()
    second = run_sparsim_pair(requests, SMOKE_PROTOCOL)

    assert len(calls) == 2
    assert first_dataset_hashes == [artifact.dataset_sha256 for artifact in second]
    assert first_manifest_hashes == [
        artifact.native_manifest.manifest_sha256 for artifact in second
    ]
    assert list((requests[0].output_path.parent / "native").iterdir()) == (
        native_directories
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "extra-file",
        "wrong-orientation",
        "fractional-count",
        "negative-latent",
        "nonfinite-latent",
        "wrong-groups",
        "same-views",
        "wrong-call-count",
        "wrong-array-hash",
        "noncanonical-json",
        "typed-cell-count",
        "typed-call-count",
        "typed-truth-flag",
        "typed-seed",
        "noncanonical-float",
        "quoted-field",
    ],
)
def test_malformed_native_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    def corrupt(output_dir: Path) -> None:
        if corruption == "extra-file":
            (output_dir / "extra.txt").write_text("extra\n", encoding="utf-8")
        elif corruption == "wrong-orientation":
            path = output_dir / "latent_expression.tsv"
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text(
                "\n".join(line.rsplit("\t", 1)[0] for line in lines) + "\n",
                encoding="utf-8",
            )
        elif corruption in {"negative-latent", "nonfinite-latent"}:
            path = output_dir / "latent_expression.tsv"
            lines = path.read_text(encoding="utf-8").splitlines()
            fields = lines[1].split("\t")
            fields[1] = "-1" if corruption == "negative-latent" else "nan"
            lines[1] = "\t".join(fields)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif corruption == "fractional-count":
            path = output_dir / "observed_moderate.tsv"
            lines = path.read_text(encoding="utf-8").splitlines()
            fields = lines[1].split("\t")
            fields[1] = "1.5"
            lines[1] = "\t".join(fields)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif corruption == "noncanonical-float":
            path = output_dir / "latent_expression.tsv"
            lines = path.read_text(encoding="utf-8").splitlines()
            fields = lines[1].split("\t")
            fields[1] += "0"
            lines[1] = "\t".join(fields)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            metadata_path = output_dir / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["array_sha256"]["latent_expression.tsv"] = _file_sha256(path)
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        elif corruption == "quoted-field":
            path = output_dir / "cell_metadata.tsv"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "cell-0001\tchu-c1", 'cell-0001\t"chu-c1"'
                ),
                encoding="utf-8",
            )
        elif corruption == "wrong-groups":
            path = output_dir / "cell_metadata.tsv"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "cell-0001\tchu-c1", "cell-0001\tchu-c6"
                ),
                encoding="utf-8",
            )
        elif corruption == "same-views":
            (output_dir / "observed_severe.tsv").write_bytes(
                (output_dir / "observed_moderate.tsv").read_bytes()
            )
        else:
            path = output_dir / "run_metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if corruption == "wrong-call-count":
                metadata["sparsim_simulation_calls"] = 1
            elif corruption == "wrong-array-hash":
                metadata["array_sha256"]["latent_expression.tsv"] = "0" * 64
            elif corruption == "typed-cell-count":
                metadata["cells"] = float(metadata["cells"])
            elif corruption == "typed-call-count":
                metadata["source_cpp_calls"] = True
            elif corruption == "typed-truth-flag":
                metadata["gene_matrix_equal"] = 1
            elif corruption == "typed-seed":
                metadata["biological_seed_r"] = float(metadata["biological_seed_r"])
            path.write_text(
                json.dumps(
                    metadata,
                    sort_keys=corruption != "noncanonical-json",
                    separators=(",", ":"),
                )
                + ("" if corruption == "noncanonical-json" else "\n"),
                encoding="utf-8",
            )

    _mock_external(monkeypatch, mutate=corrupt)
    requests = _requests(tmp_path)

    with pytest.raises(SimulationContractError):
        run_sparsim_pair(requests, SMOKE_PROTOCOL)
    assert not any(request.output_path.exists() for request in requests)
    assert not (requests[0].output_path.parent / "native").exists()


def test_source_and_environment_are_reverified_even_after_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_checks = 0
    source_file_checks = 0
    environment_checks = 0

    def verify_source() -> dict[str, object]:
        nonlocal source_checks
        source_checks += 1
        return dict(SOURCE_RECEIPT)

    def verify_environment() -> dict[str, object]:
        nonlocal environment_checks
        environment_checks += 1
        return dict(ENVIRONMENT_RECEIPT)

    def verify_source_files() -> dict[str, object]:
        nonlocal source_file_checks
        source_file_checks += 1
        return json.loads(json.dumps(SOURCE_FILE_RECEIPT))

    def fail_runner(
        config_path: Path, output_dir: Path, *, timeout_seconds: int
    ) -> None:
        raise SimulationContractError("injected SPARSim native failure")

    monkeypatch.setattr(sparsim_module, "_verify_sparsim_source", verify_source)
    monkeypatch.setattr(sparsim_module, "_source_file_receipt", verify_source_files)
    monkeypatch.setattr(sparsim_module, "_environment_receipt", verify_environment)
    monkeypatch.setattr(sparsim_module, "_execute_sparsim", fail_runner)

    with pytest.raises(SimulationContractError, match="native failure"):
        run_sparsim_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert source_checks == 2
    assert source_file_checks == 2
    assert environment_checks == 2


def test_existing_result_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    requests[0].output_path.parent.mkdir(parents=True)
    requests[0].output_path.write_bytes(b"do not overwrite")

    with pytest.raises(SimulationContractError, match="exist|overwrite"):
        run_sparsim_pair(requests, SMOKE_PROTOCOL)

    assert requests[0].output_path.read_bytes() == b"do not overwrite"
    assert calls == []


def test_both_serialized_datasets_are_validated_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_stage = sparsim_module._stage_h5ad
    stages = 0

    def fail_second(*args: object, **kwargs: object) -> object:
        nonlocal stages
        stages += 1
        if stages == 2:
            raise SimulationContractError("injected second serialization failure")
        return real_stage(*args, **kwargs)

    monkeypatch.setattr(sparsim_module, "_stage_h5ad", fail_second)

    with pytest.raises(SimulationContractError, match="serialization"):
        run_sparsim_pair(requests, SMOKE_PROTOCOL)
    assert stages == 2
    assert not any(request.output_path.exists() for request in requests)


def test_publication_failure_rolls_back_first_result_and_native_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_publish = sparsim_module._publish_staged_h5ad
    publications = 0

    def fail_second(*args: object, **kwargs: object) -> object:
        nonlocal publications
        publications += 1
        if publications == 2:
            raise SimulationContractError("injected publication failure")
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(sparsim_module, "_publish_staged_h5ad", fail_second)

    with pytest.raises(SimulationContractError, match="publication failure"):
        run_sparsim_pair(requests, SMOKE_PROTOCOL)
    assert publications == 2
    assert not any(request.output_path.exists() for request in requests)
    native_root = requests[0].output_path.parent / "native"
    assert not native_root.exists() or list(native_root.iterdir()) == []


def test_postpublication_stage_cleanup_failure_does_not_report_pair_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_rmtree = sparsim_module.shutil.rmtree
    failed_stage: Path | None = None

    def fail_final_stage_once(path: object, *args: object, **kwargs: object) -> None:
        nonlocal failed_stage
        candidate = Path(path)  # type: ignore[arg-type]
        if (
            failed_stage is None
            and candidate.name.startswith("maskimpute-sparsim-native-")
            and all(request.output_path.is_file() for request in requests)
        ):
            failed_stage = candidate
            raise OSError("injected final stage cleanup failure")
        real_rmtree(candidate, *args, **kwargs)

    monkeypatch.setattr(sparsim_module.shutil, "rmtree", fail_final_stage_once)
    try:
        artifacts = run_sparsim_pair(requests, SMOKE_PROTOCOL)
    finally:
        if failed_stage is not None and failed_stage.exists():
            real_rmtree(failed_stage)

    assert failed_stage is not None
    assert len(artifacts) == 2
    assert all(request.output_path.is_file() for request in requests)


_CHECKOUT = Path("artifacts/external/checkouts/sparsim")
_REAL_ASSETS_AVAILABLE = all(
    path.exists()
    for path in (
        _CHECKOUT / ".git",
        Path("artifacts/external/receipts/sparsim.json"),
        Path("artifacts/envs/symsim-r44/bin/Rscript"),
        Path("/usr/bin/g++"),
    )
)


@integration
@pytest.mark.skipif(
    not _REAL_ASSETS_AVAILABLE,
    reason="pinned SPARSim checkout and R 4.4 environment are unavailable",
)
def test_real_pinned_sparsim_two_rerun_smoke_is_exact_and_pristine(
    tmp_path: Path,
) -> None:
    before = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
    )
    requests = _requests(tmp_path)

    first = run_sparsim_pair(requests, SMOKE_PROTOCOL)
    dataset_hashes = [artifact.dataset_sha256 for artifact in first]
    manifest_hashes = [artifact.native_manifest.manifest_sha256 for artifact in first]
    truth = np.asarray(first[0].adata.layers["latent_expression"]).copy()
    observed = [np.asarray(artifact.adata.X).copy() for artifact in first]
    native_directories = list((requests[0].output_path.parent / "native").iterdir())
    for request in requests:
        request.output_path.unlink()
    second = run_sparsim_pair(requests, SMOKE_PROTOCOL)

    after = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=_CHECKOUT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
    )
    assert before == (
        "4e7712fb236a92ce7c173da169c8a29cc2a9f0ef",
        "5d66b28cc6afd8d68364f4205cc983c7f681e2fe",
        "",
    )
    assert after == before
    assert dataset_hashes == [artifact.dataset_sha256 for artifact in second]
    assert manifest_hashes == [
        artifact.native_manifest.manifest_sha256 for artifact in second
    ]
    assert list((requests[0].output_path.parent / "native").iterdir()) == (
        native_directories
    )
    np.testing.assert_array_equal(truth, second[1].adata.layers["latent_expression"])
    for expected, artifact in zip(observed, second, strict=True):
        np.testing.assert_array_equal(expected, artifact.adata.X)
    assert not np.array_equal(second[0].adata.X, second[1].adata.X)
    assert float(second[0].adata.obs["library_size"].mean()) > float(
        second[1].adata.obs["library_size"].mean()
    )
    assert float((np.asarray(second[0].adata.X) == 0).mean()) < float(
        (np.asarray(second[1].adata.X) == 0).mean()
    )
