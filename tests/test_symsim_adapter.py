from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import warnings

import anndata as ad
import numpy as np
import pytest

from maskimpute_benchmark.protocol import load_protocol
from maskimpute_benchmark.schema import benchmark_dataset_sha256
from maskimpute_benchmark.simulators import (
    SimulationContractError,
    SimulationRequest,
    load_final_manifest_claim,
)
from maskimpute_benchmark.study import (
    assert_final_runnable,
    freeze_round,
    materialize_final,
    supersede_round,
)
import maskimpute_benchmark.simulators.symsim as symsim_module
from maskimpute_benchmark.simulators.symsim import map_symsim_r_seeds, run_symsim_pair


PROTOCOL = load_protocol(Path("study/protocol.json"))
SMOKE_PROTOCOL = replace(
    PROTOCOL,
    development=replace(PROTOCOL.development, cells=20, genes=20),
)
SOURCE_RECEIPT = {
    "citation_doi": "10.1038/s41467-019-10500-w",
    "ledger_sha256": "5a6f60c5de980a20eb118d0b82913112650f1956562aec4c92d37d8314c9f29e",
    "license": "Artistic-2.0",
    "resolved_revision": "76a674b407ce44bf2690a9161cf28b905598d0a5",
    "revision": "76a674b407ce44bf2690a9161cf28b905598d0a5",
    "role": "mechanism",
    "schema_version": 1,
    "source_id": "symsim",
    "source_type": "git",
    "source_url": "https://github.com/YosefLab/SymSim.git",
    "verified_checksum": {
        "algorithm": "git-tree-sha1",
        "value": "12d9c7e9e8c22bb0bae917aec7860627dcb8489b",
    },
}
ENVIRONMENT_RECEIPT = {
    "schema": "maskimpute-conda-environment-v1",
    "sha256": "b" * 64,
    "r_executable_sha256": "c" * 64,
    "package_count": 42,
}


with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


def _requests(tmp_path: Path) -> tuple[SimulationRequest, SimulationRequest]:
    moderate = SimulationRequest(
        mechanism="symsim",
        namespace="dev",
        biological_id="draw-01",
        biological_seed=2**62 + 101,
        measurement_seed=2**61 + 202,
        technical_view="moderate",
        cells=20,
        genes=20,
        output_path=tmp_path / "dev/symsim/draw-01-moderate.h5ad",
    )
    severe = replace(
        moderate,
        measurement_seed=2**61 + 303,
        technical_view="severe",
        output_path=tmp_path / "dev/symsim/draw-01-severe.h5ad",
    )
    return moderate, severe


def _git(repo: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def final_claim(tmp_path: Path) -> tuple[object, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "SymSim Adapter Test")
    _git(repo, "config", "user.email", "symsim@example.invalid")
    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repo / "config.json").write_text('{"adapter":"symsim"}\n', encoding="utf-8")
    (repo / "environment.lock").write_text("fixture\n", encoding="utf-8")
    (repo / "protocol.json").write_bytes(Path("study/protocol.json").read_bytes())
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "freeze SymSim claim fixture")
    round_dir = repo / "artifacts/study/round-001"
    freeze_round(
        repo,
        round_dir,
        repo / "config.json",
        repo / "protocol.json",
        environment_path=repo / "environment.lock",
    )
    materialize_final(round_dir, seed_count=4, repo=repo)
    assert_final_runnable(repo, round_dir)
    return load_final_manifest_claim(repo, round_dir), round_dir


def _write_matrix(
    path: Path, row_ids: list[str], column_ids: list[str], values: np.ndarray
) -> None:
    lines = ["\t".join(["gene_id", *column_ids])]
    for row_id, row in zip(row_ids, values, strict=True):
        lines.append("\t".join([row_id, *(str(int(value)) for value in row)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_valid_native(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cells = config["simulation"]["cells"]
    genes = config["simulation"]["genes"]
    cell_ids = [f"cell-{index:04d}" for index in range(1, cells + 1)]
    gene_ids = [f"gene-{index:04d}" for index in range(1, genes + 1)]
    true_counts = np.fromfunction(
        lambda gene, cell: (gene + cell) % 7 + 1,
        (genes, cells),
        dtype=int,
    ).astype(np.int64)
    _write_matrix(output_dir / "true_counts.tsv", gene_ids, cell_ids, true_counts)
    for view, divisor in (("moderate", 2), ("severe", 4)):
        _write_matrix(
            output_dir / f"observed_{view}.tsv",
            gene_ids,
            cell_ids,
            true_counts // divisor,
        )

    groups = [1] + [2] * 5 + [3] * 5 + [4] * 5 + [5] * 4
    cell_lines = ["cell_id\tgroup"]
    cell_lines.extend(
        f"{cell_id}\t{group}" for cell_id, group in zip(cell_ids, groups, strict=True)
    )
    (output_dir / "cell_metadata.tsv").write_text(
        "\n".join(cell_lines) + "\n", encoding="utf-8"
    )

    marker_header = ["gene_id"]
    for group in range(1, 6):
        marker_header.extend(
            [f"theoretical_log2fc_group_{group}", f"marker_group_{group}"]
        )
    marker_lines = ["\t".join(marker_header)]
    for gene_index, gene_id in enumerate(gene_ids, start=1):
        fields = [gene_id]
        for group in range(1, 6):
            log_fc = 2.0 if (gene_index - 1) % 5 + 1 == group else -0.5
            fields.extend([format(log_fc, ".17g"), str(int(log_fc > 1.0))])
        marker_lines.append("\t".join(fields))
    (output_dir / "marker_truth.tsv").write_text(
        "\n".join(marker_lines) + "\n", encoding="utf-8"
    )

    metadata = {
        "schema_version": 1,
        "simulate_true_counts_calls": 1,
        "true2observed_counts_calls": 2,
        "cells": cells,
        "genes": genes,
        "views": [view["technical_view"] for view in config["views"]],
        "biological_seed_r": config["seeds"]["biological"]["mapped_r"],
        "measurement_seeds_r": {
            view["technical_view"]: view["measurement_seed_r"]
            for view in config["views"]
        },
        "r_version": "R version fixture",
        "symsim_version": "0.0.0.9000",
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
        symsim_module,
        "_verify_symsim_source",
        lambda: json.loads(json.dumps(SOURCE_RECEIPT)),
    )
    monkeypatch.setattr(
        symsim_module,
        "_environment_receipt",
        lambda: dict(ENVIRONMENT_RECEIPT),
    )

    def fake_runner(
        config_path: Path, output_dir: Path, *, timeout_seconds: int
    ) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        calls.append({"config": config, "timeout_seconds": timeout_seconds})
        _write_valid_native(config_path, output_dir)
        if callable(mutate):
            mutate(output_dir)

    monkeypatch.setattr(symsim_module, "_execute_symsim", fake_runner)
    return calls


def test_r_seed_mapping_is_deterministic_distinct_and_in_native_range() -> None:
    original = (2**62 + 101, 2**61 + 202, 2**61 + 303)

    first = map_symsim_r_seeds(*original)
    second = map_symsim_r_seeds(*original)

    assert first == second
    assert set(first) == {"biological", "moderate", "severe"}
    assert len(set(first.values())) == 3
    assert all(type(value) is int and 1 <= value < 2**31 for value in first.values())


def test_mocked_pair_runs_one_native_process_and_binds_exact_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)

    artifacts = run_symsim_pair(requests, SMOKE_PROTOCOL)

    assert len(calls) == 1
    assert calls[0]["timeout_seconds"] == SMOKE_PROTOCOL.final_timeout_seconds
    config = calls[0]["config"]
    assert config["simulation"] == {
        "cells": 20,
        "genes": 20,
        "gene_length": 1000,
        "gene_module_prop": 0,
        "i_minpop": 1,
        "marker_log2fc_threshold": 1.0,
        "min_popsize": 1,
        "n_de_evf": 9,
        "nevf": 10,
        "prop_hge": 0,
        "vary": "s",
    }
    assert [view["technical_view"] for view in config["views"]] == [
        "moderate",
        "severe",
    ]
    assert config["seeds"]["biological"]["original"] == requests[0].biological_seed
    assert set(config["adapter"]) == {
        "python_adapter_sha256",
        "r_runner_sha256",
    }
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in config["adapter"].values()
    )
    for artifact, request in zip(artifacts, requests, strict=True):
        assert artifact.request == request
        assert request.output_path.is_file()
        adata = artifact.adata
        assert adata.shape == (20, 20)
        assert adata.X.dtype == np.int64
        assert adata.layers["pre_capture_counts"].dtype == np.int64
        assert adata.uns["truth_kind"] == "exact_pre_capture"
        assert adata.uns["primary_truth_layer"] == "pre_capture_counts"
        assert adata.obs["group"].value_counts().to_dict()["pop-1"] == 1
        assert set(adata.obs["group"]) == {f"pop-{group}" for group in range(1, 6)}
        assert "marker_group_1" in adata.var
        provenance = adata.uns["provenance"]
        assert provenance["parameters"]["adapter"] == config["adapter"]
        assert provenance["software_version"] == SOURCE_RECEIPT["resolved_revision"]
        assert provenance["parameters"]["source_receipt"] == SOURCE_RECEIPT
        assert provenance["parameters"]["environment"] == ENVIRONMENT_RECEIPT
        assert provenance["seeds"]["biological"] == request.biological_seed
        assert provenance["seeds"]["measurement"] == request.measurement_seed
        assert artifact.dataset_sha256 == benchmark_dataset_sha256(adata)
        assert {entry.path for entry in artifact.native_manifest.files} == {
            "config.json",
            "true_counts.tsv",
            "observed_moderate.tsv",
            "observed_severe.tsv",
            "cell_metadata.tsv",
            "marker_truth.tsv",
            "run_metadata.json",
        }
    np.testing.assert_array_equal(
        artifacts[0].adata.layers["pre_capture_counts"],
        artifacts[1].adata.layers["pre_capture_counts"],
    )


def test_deterministic_rerun_has_identical_semantic_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)

    first = run_symsim_pair(requests, SMOKE_PROTOCOL)
    second = run_symsim_pair(requests, SMOKE_PROTOCOL)

    assert len(calls) == 2
    assert [artifact.dataset_sha256 for artifact in first] == [
        artifact.dataset_sha256 for artifact in second
    ]
    np.testing.assert_array_equal(
        first[0].adata.layers["pre_capture_counts"],
        second[1].adata.layers["pre_capture_counts"],
    )
    assert list(requests[0].output_path.parent.glob(".symsim-native-*")) == []


def test_relocated_rerun_has_identical_manifest_and_semantic_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)

    first = run_symsim_pair(_requests(tmp_path / "first"), SMOKE_PROTOCOL)
    relocated = run_symsim_pair(_requests(tmp_path / "relocated"), SMOKE_PROTOCOL)

    assert [artifact.native_manifest.manifest_sha256 for artifact in first] == [
        artifact.native_manifest.manifest_sha256 for artifact in relocated
    ]
    assert [artifact.dataset_sha256 for artifact in first] == [
        artifact.dataset_sha256 for artifact in relocated
    ]


def test_request_sequence_is_snapshotted_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    original = _requests(tmp_path)
    alternate = tuple(
        replace(
            request,
            biological_seed=request.biological_seed + 1000,
            measurement_seed=request.measurement_seed + 1000,
            output_path=request.output_path.with_name(
                f"alternate-{request.output_path.name}"
            ),
        )
        for request in original
    )

    class StatefulPair(Sequence[SimulationRequest]):
        def __init__(self) -> None:
            self.iterations = 0

        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> SimulationRequest:
            return original[index]

        def __iter__(self) -> Iterator[SimulationRequest]:
            self.iterations += 1
            return iter(original if self.iterations == 1 else alternate)

    stateful = StatefulPair()

    artifacts = run_symsim_pair(stateful, SMOKE_PROTOCOL)

    assert stateful.iterations == 1
    assert tuple(artifact.request for artifact in artifacts) == original
    assert all(request.output_path.is_file() for request in original)
    assert not any(request.output_path.exists() for request in alternate)


def test_pair_contract_is_revalidated_after_native_execution_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_validate = symsim_module.validate_paired_simulation_requests
    validations = 0

    def supersede_on_second_validation(*args: object, **kwargs: object) -> None:
        nonlocal validations
        validations += 1
        if validations == 2:
            assert not requests[0].output_path.parent.exists()
            raise SimulationContractError("final round became superseded")
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        symsim_module,
        "validate_paired_simulation_requests",
        supersede_on_second_validation,
    )

    with pytest.raises(SimulationContractError, match="superseded"):
        run_symsim_pair(requests, SMOKE_PROTOCOL)
    assert validations == 2
    assert not any(request.output_path.exists() for request in requests)
    assert not (requests[0].output_path.parent / "native").exists()


def test_pair_contract_is_revalidated_after_h5ad_publication_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_validate = symsim_module.validate_paired_simulation_requests
    real_publish = symsim_module._publish_staged_h5ad
    publication_started = False
    validations = 0

    def count_and_supersede(*args: object, **kwargs: object) -> None:
        nonlocal validations
        validations += 1
        if publication_started and validations == 3:
            raise SimulationContractError("final round superseded during publication")
        real_validate(*args, **kwargs)

    def publish_then_transition(temporary: Path, destination: Path) -> ad.AnnData:
        nonlocal publication_started
        persisted = real_publish(temporary, destination)
        publication_started = True
        return persisted

    monkeypatch.setattr(
        symsim_module,
        "validate_paired_simulation_requests",
        count_and_supersede,
    )
    monkeypatch.setattr(symsim_module, "_publish_staged_h5ad", publish_then_transition)

    with pytest.raises(SimulationContractError, match="superseded"):
        run_symsim_pair(requests, SMOKE_PROTOCOL)
    assert validations == 3


def test_postpublication_final_claim_check_accepts_unchanged_running_claim(
    final_claim: tuple[object, Path],
) -> None:
    claim, _round_dir = final_claim

    symsim_module._revalidate_published_final_claim(claim)


def test_postpublication_final_claim_check_rejects_supersession(
    final_claim: tuple[object, Path],
) -> None:
    claim, round_dir = final_claim
    supersede_round(round_dir, "transition during SymSim publication")

    with pytest.raises(SimulationContractError, match="running|claim|superseded"):
        symsim_module._revalidate_published_final_claim(claim)


def test_both_h5ad_serializations_are_validated_before_either_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_write = ad.AnnData.write_h5ad
    writes = 0

    def fail_second_write(
        self: ad.AnnData, filename: Path, *args: object, **kwargs: object
    ) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("injected second serialization failure")
        real_write(self, filename, *args, **kwargs)

    monkeypatch.setattr(ad.AnnData, "write_h5ad", fail_second_write)

    with pytest.raises(SimulationContractError, match="persist|serial"):
        run_symsim_pair(requests, SMOKE_PROTOCOL)
    assert writes == 2
    assert not any(request.output_path.exists() for request in requests)


def test_both_persisted_semantics_are_schema_validated_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_read = ad.read_h5ad
    reads = 0

    def invalidate_second_read(*args: object, **kwargs: object) -> ad.AnnData:
        nonlocal reads
        reads += 1
        dataset = real_read(*args, **kwargs)
        if reads == 2:
            del dataset.obs["condition"]
        return dataset

    monkeypatch.setattr(ad, "read_h5ad", invalidate_second_read)

    with pytest.raises((SimulationContractError, ValueError), match="condition|schema"):
        run_symsim_pair(requests, SMOKE_PROTOCOL)
    assert reads == 2
    assert not any(request.output_path.exists() for request in requests)


def test_h5ad_staging_uses_an_explicit_same_device_existing_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_mkstemp = symsim_module.tempfile.mkstemp
    staging_roots: list[Path] = []

    def record_staging_root(*args: object, **kwargs: object) -> tuple[int, str]:
        directory = kwargs.get("dir")
        assert directory is not None
        root = Path(directory)
        assert root.is_dir() and not root.is_symlink()
        staging_roots.append(root)
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(symsim_module.tempfile, "mkstemp", record_staging_root)

    run_symsim_pair(requests, SMOKE_PROTOCOL)

    assert len(staging_roots) == 2
    output_device = requests[0].output_path.stat().st_dev
    assert all(root.stat().st_dev == output_device for root in staging_roots)


@pytest.mark.parametrize(
    "corruption",
    [
        "extra-file",
        "symlink",
        "negative-count",
        "fractional-count",
        "wrong-orientation",
        "wrong-cell-id",
        "wrong-rare-count",
        "nonfinite-marker",
        "wrong-call-count",
        "typed-cell-count",
        "typed-mapped-seed",
        "umi-exceeds-truth",
        "library-size-overflow",
    ],
)
def test_malformed_or_malicious_native_outputs_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    outside = tmp_path / "outside.tsv"
    outside.write_text("outside\n", encoding="utf-8")

    def corrupt(output_dir: Path) -> None:
        if corruption == "extra-file":
            (output_dir / "undeclared.txt").write_text("extra", encoding="utf-8")
        elif corruption == "symlink":
            target = output_dir / "true_counts.tsv"
            target.unlink()
            target.symlink_to(outside)
        elif corruption in {"negative-count", "fractional-count", "umi-exceeds-truth"}:
            target = output_dir / "observed_moderate.tsv"
            lines = target.read_text(encoding="utf-8").splitlines()
            fields = lines[1].split("\t")
            fields[1] = {
                "negative-count": "-1",
                "fractional-count": "1.5",
                "umi-exceeds-truth": "999999",
            }[corruption]
            lines[1] = "\t".join(fields)
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif corruption == "library-size-overflow":
            maximum = str(np.iinfo(np.int64).max)
            for filename in (
                "true_counts.tsv",
                "observed_moderate.tsv",
                "observed_severe.tsv",
            ):
                target = output_dir / filename
                lines = target.read_text(encoding="utf-8").splitlines()
                for index in range(1, len(lines)):
                    fields = lines[index].split("\t")
                    fields[1] = maximum
                    lines[index] = "\t".join(fields)
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif corruption == "wrong-orientation":
            target = output_dir / "observed_moderate.tsv"
            lines = target.read_text(encoding="utf-8").splitlines()
            target.write_text(
                "\n".join(line.rsplit("\t", 1)[0] for line in lines) + "\n"
            )
        elif corruption == "wrong-cell-id":
            target = output_dir / "true_counts.tsv"
            lines = target.read_text(encoding="utf-8").splitlines()
            header = lines[0].split("\t")
            header[1] = "malicious-cell"
            lines[0] = "\t".join(header)
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif corruption == "wrong-rare-count":
            target = output_dir / "cell_metadata.tsv"
            target.write_text(
                target.read_text(encoding="utf-8").replace(
                    "cell-0001\t1", "cell-0001\t2"
                ),
                encoding="utf-8",
            )
        elif corruption == "nonfinite-marker":
            target = output_dir / "marker_truth.tsv"
            lines = target.read_text(encoding="utf-8").splitlines()
            fields = lines[1].split("\t")
            fields[1] = "nan"
            lines[1] = "\t".join(fields)
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif corruption in {
            "wrong-call-count",
            "typed-cell-count",
            "typed-mapped-seed",
        }:
            target = output_dir / "run_metadata.json"
            metadata = json.loads(target.read_text(encoding="utf-8"))
            if corruption == "wrong-call-count":
                metadata["simulate_true_counts_calls"] = 2
            elif corruption == "typed-cell-count":
                metadata["cells"] = float(metadata["cells"])
            else:
                metadata["biological_seed_r"] = float(metadata["biological_seed_r"])
            target.write_text(json.dumps(metadata), encoding="utf-8")

    _mock_external(monkeypatch, mutate=corrupt)

    with pytest.raises(SimulationContractError):
        run_symsim_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert outside.read_text(encoding="utf-8") == "outside\n"


@pytest.mark.parametrize(
    "change",
    [
        {"technical_view": "other"},
        {"cells": 21},
        {"genes": 19},
    ],
)
def test_adapter_rejects_noncanonical_design_before_native_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
) -> None:
    calls = _mock_external(monkeypatch)
    moderate, severe = _requests(tmp_path)
    severe = replace(severe, **change)
    protocol = SMOKE_PROTOCOL
    if "cells" in change or "genes" in change:
        protocol = replace(
            protocol,
            development=replace(
                protocol.development,
                cells=int(change.get("cells", 20)),
                genes=int(change.get("genes", 20)),
            ),
        )

    with pytest.raises(SimulationContractError):
        run_symsim_pair((moderate, severe), protocol)
    assert calls == []


def test_source_is_reverified_after_native_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = [dict(SOURCE_RECEIPT), {**SOURCE_RECEIPT, "ledger_sha256": "d" * 64}]
    monkeypatch.setattr(symsim_module, "_verify_symsim_source", lambda: receipts.pop(0))
    monkeypatch.setattr(
        symsim_module, "_environment_receipt", lambda: dict(ENVIRONMENT_RECEIPT)
    )
    monkeypatch.setattr(symsim_module, "_execute_symsim", _write_valid_native)

    with pytest.raises(SimulationContractError, match="source.*changed|pristine"):
        run_symsim_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert receipts == []


def test_source_is_reverified_when_native_runner_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_checks = 0

    def verify() -> dict[str, object]:
        nonlocal source_checks
        source_checks += 1
        return dict(SOURCE_RECEIPT)

    def fail_runner(
        config_path: Path, output_dir: Path, *, timeout_seconds: int
    ) -> None:
        raise SimulationContractError("injected native failure")

    monkeypatch.setattr(symsim_module, "_verify_symsim_source", verify)
    monkeypatch.setattr(
        symsim_module, "_environment_receipt", lambda: dict(ENVIRONMENT_RECEIPT)
    )
    monkeypatch.setattr(symsim_module, "_execute_symsim", fail_runner)

    with pytest.raises(SimulationContractError, match="native failure"):
        run_symsim_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert source_checks == 2


_REAL_ASSETS_AVAILABLE = all(
    path.exists()
    for path in (
        Path("artifacts/external/checkouts/symsim/.git"),
        Path("artifacts/external/receipts/symsim.json"),
        Path("artifacts/envs/symsim-r44/bin/Rscript"),
    )
)


@integration
@pytest.mark.skipif(
    not _REAL_ASSETS_AVAILABLE,
    reason="pinned SymSim checkout and R 4.4 environment are unavailable",
)
def test_real_pinned_symsim_smoke_is_deterministic(tmp_path: Path) -> None:
    requests = _requests(tmp_path)

    first = run_symsim_pair(requests, SMOKE_PROTOCOL)
    first_hashes = [artifact.dataset_sha256 for artifact in first]
    first_truth = np.asarray(first[0].adata.layers["pre_capture_counts"])
    second = run_symsim_pair(requests, SMOKE_PROTOCOL)

    assert first_hashes == [artifact.dataset_sha256 for artifact in second]
    np.testing.assert_array_equal(
        first_truth,
        np.asarray(second[1].adata.layers["pre_capture_counts"]),
    )
    assert int((first[0].adata.obs["group"] == "pop-1").sum()) == 1
    assert not np.array_equal(first[0].adata.X, first[1].adata.X)
