from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import warnings

import anndata as ad
import numpy as np
import pytest

from maskimpute_benchmark.protocol import load_protocol
from maskimpute_benchmark.schema import benchmark_dataset_sha256
from maskimpute_benchmark.simulators import SimulationContractError, SimulationRequest
import maskimpute_benchmark.simulators.sergio as sergio_module


PROTOCOL = load_protocol(Path("study/protocol.json"))
SMOKE_PROTOCOL = replace(
    PROTOCOL,
    development=replace(PROTOCOL.development, cells=18, genes=20),
)
SOURCE_RECEIPT = {
    "citation_doi": "10.1016/j.cels.2020.08.003",
    "ledger_sha256": "5a6f60c5de980a20eb118d0b82913112650f1956562aec4c92d37d8314c9f29e",
    "license": "GPL-3.0-only",
    "resolved_revision": "a6190b74425112834c8fa9b4b6157d9cb3d1ab88",
    "revision": "a6190b74425112834c8fa9b4b6157d9cb3d1ab88",
    "role": "mechanism",
    "schema_version": 1,
    "source_id": "sergio",
    "source_type": "git",
    "source_url": "https://github.com/PayamDiba/SERGIO.git",
    "verified_checksum": {
        "algorithm": "git-tree-sha1",
        "value": "15558fe60f62683c6fa46bcde01d9f3d3382e34a",
    },
}
ENVIRONMENT_RECEIPT = {
    "schema": "maskimpute-python-environment-v1",
    "sha256": "b" * 64,
    "python_executable_sha256": "c" * 64,
    "versions": {
        "python": "3.11.13",
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "networkx": "3.4.2",
    },
}
NATIVE_FILES = {
    "clean.npy",
    "config.json",
    "dropout_indicator_moderate.npy",
    "dropout_indicator_severe.npy",
    "observed_moderate.npy",
    "observed_severe.npy",
    "pre_dropout_moderate.npy",
    "pre_dropout_severe.npy",
    "run_metadata.json",
}


with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


def test_sergio_adapter_and_runner_are_present() -> None:
    assert Path("maskimpute_benchmark/simulators/sergio.py").is_file()
    assert Path("scripts/simulators/run_sergio.py").is_file()


def test_sergio_adapter_exposes_only_the_paired_public_api() -> None:
    assert callable(getattr(sergio_module, "map_sergio_numpy_seeds", None))
    assert callable(getattr(sergio_module, "run_sergio_pair", None))
    assert sergio_module.__all__ == ["map_sergio_numpy_seeds", "run_sergio_pair"]


def _requests(
    root: Path, *, namespace: str = "dev"
) -> tuple[SimulationRequest, SimulationRequest]:
    moderate = SimulationRequest(
        mechanism="sergio",
        namespace=namespace,
        biological_id="draw-01",
        biological_seed=2**62 + 101,
        measurement_seed=2**61 + 202,
        technical_view="moderate",
        cells=18,
        genes=20,
        output_path=root / namespace / "sergio/draw-01-moderate.h5ad",
    )
    severe = replace(
        moderate,
        measurement_seed=2**61 + 303,
        technical_view="severe",
        output_path=root / namespace / "sergio/draw-01-severe.h5ad",
    )
    return moderate, severe


def _save_array(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as output:
        np.save(output, values, allow_pickle=False)


def _canonical_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _array_hashes(output_dir: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in sorted(NATIVE_FILES)
        if name.endswith(".npy")
    }


def _write_valid_native(config_path: Path, output_dir: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cells = config["simulation"]["cells"]
    genes = config["simulation"]["requested_genes"]
    clean = np.fromfunction(
        lambda gene, cell: (gene + 1) * 0.125 + (cell % 9 + 1) * 0.25,
        (genes, cells),
        dtype=np.float64,
    ).astype("<f8")
    pre_dropout = {
        "moderate": np.ascontiguousarray(clean * 1.2 + 0.25, dtype="<f8"),
        "severe": np.ascontiguousarray(clean * 0.7 + 0.125, dtype="<f8"),
    }
    indicators = {
        "moderate": np.fromfunction(
            lambda gene, cell: ((gene + cell) % 4 != 0),
            (genes, cells),
            dtype=int,
        ).astype(np.uint8),
        "severe": np.fromfunction(
            lambda gene, cell: ((2 * gene + cell) % 3 == 0),
            (genes, cells),
            dtype=int,
        ).astype(np.uint8),
    }
    _save_array(output_dir / "clean.npy", clean)
    for view in ("moderate", "severe"):
        _save_array(output_dir / f"pre_dropout_{view}.npy", pre_dropout[view])
        _save_array(output_dir / f"dropout_indicator_{view}.npy", indicators[view])
        observed = np.floor(pre_dropout[view] * indicators[view]).astype("<i8")
        _save_array(output_dir / f"observed_{view}.npy", observed)

    per_view_calls = {
        view: {
            "outlier_effect": 1,
            "lib_size_effect": 1,
            "dropout_indicator": 1,
            "convert_to_umi_counts": 1,
        }
        for view in ("moderate", "severe")
    }
    metadata = {
        "schema_version": 1,
        "array_sha256": _array_hashes(output_dir),
        "biological_seed_numpy": config["seeds"]["biological"]["mapped_numpy"],
        "call_counts": {
            "sergio_constructor": 1,
            "build_graph": 1,
            "simulate": 1,
            "get_expressions": 1,
            "outlier_effect": 2,
            "lib_size_effect": 2,
            "dropout_indicator": 2,
            "convert_to_umi_counts": 2,
            "per_view": per_view_calls,
        },
        "cells": cells,
        "cell_types": 9,
        "compatibility_shim": config["adapter"]["compatibility_shim"],
        "measurement_seeds_numpy": {
            view["technical_view"]: view["measurement_seed_numpy"]
            for view in config["views"]
        },
        "module_path": config["source"]["module_path"],
        "requested_genes": genes,
        "simulated_genes": config["simulation"]["simulated_genes"],
        "versions": {
            "networkx": "3.4.2",
            "numpy": "2.2.6",
            "python": "3.11.13",
            "scipy": "1.15.3",
            "sergio": "1.0.0",
        },
        "views": [view["technical_view"] for view in config["views"]],
    }
    _canonical_json(output_dir / "run_metadata.json", metadata)


def _rehash_run_metadata(output_dir: Path) -> None:
    path = output_dir / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["array_sha256"] = _array_hashes(output_dir)
    _canonical_json(path, metadata)


def _mock_external(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: object | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        sergio_module,
        "_verify_sergio_source",
        lambda: json.loads(json.dumps(SOURCE_RECEIPT)),
    )
    monkeypatch.setattr(
        sergio_module,
        "_environment_receipt",
        lambda: json.loads(json.dumps(ENVIRONMENT_RECEIPT)),
    )

    def fake_runner(
        config_path: Path, output_dir: Path, *, timeout_seconds: int
    ) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        calls.append({"config": config, "timeout_seconds": timeout_seconds})
        _write_valid_native(config_path, output_dir)
        if callable(mutate):
            mutate(output_dir)

    monkeypatch.setattr(sergio_module, "_execute_sergio", fake_runner)
    return calls


def test_numpy_seed_mapping_is_deterministic_distinct_and_32_bit() -> None:
    original = (2**62 + 101, 2**61 + 202, 2**61 + 303)

    first = sergio_module.map_sergio_numpy_seeds(*original)
    second = sergio_module.map_sergio_numpy_seeds(*original)

    assert first == second
    assert set(first) == {"biological", "moderate", "severe"}
    assert len(set(first.values())) == 3
    assert all(
        type(value) is int and 1 <= value <= 2**32 - 1 for value in first.values()
    )


def test_seed_mapping_rejects_non_integer_and_out_of_range_inputs() -> None:
    for invalid in (-1, 2**63, 1.0, True):
        with pytest.raises(SimulationContractError, match="63-bit"):
            sergio_module.map_sergio_numpy_seeds(invalid, 2, 3)  # type: ignore[arg-type]


def test_seed_mapping_avoids_native_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sergio_module, "_mapped_numpy_seed", lambda *_args: 7)

    mapped = sergio_module.map_sergio_numpy_seeds(1, 2, 3)

    assert mapped == {"biological": 7, "moderate": 8, "severe": 9}


def test_mocked_pair_uses_fixed_profile_regimes_and_preserves_both_truths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)

    artifacts = sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)

    assert len(calls) == 1
    assert calls[0]["timeout_seconds"] == SMOKE_PROTOCOL.final_timeout_seconds
    config = calls[0]["config"]
    assert config["profile"] == {
        "interaction_path": "data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt",
        "name": "De-noised_100G_9T_300cPerT_4_DS1",
        "regulator_path": "data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt",
    }
    assert config["simulation"] == {
        "cells": 18,
        "cell_types": 9,
        "cells_per_type": 2,
        "decays": 0.8,
        "noise_params": 1.0,
        "noise_type": "dpd",
        "requested_genes": 20,
        "sampling_state": 15,
        "shared_coop_state": 2.0,
        "simulated_genes": 100,
    }
    assert config["adapter"]["compatibility_shim"] == {
        "numpy_removed_aliases": {
            "np.float": "builtins.float",
            "np.int": "builtins.int",
        }
    }
    assert set(config["adapter"]) == {
        "compatibility_shim",
        "python_adapter_sha256",
        "python_runner_sha256",
    }
    assert all(
        len(config["adapter"][key]) == 64
        for key in ("python_adapter_sha256", "python_runner_sha256")
    )
    assert config["views"] == [
        {
            "dropout_percentile": 65,
            "dropout_shape": 6.5,
            "library_log_mean": 5.2,
            "library_log_sd": 0.3,
            "measurement_seed_numpy": config["views"][0]["measurement_seed_numpy"],
            "measurement_seed_original": requests[0].measurement_seed,
            "outlier_mean": 0.8,
            "outlier_prob": 0.01,
            "outlier_scale": 1.0,
            "technical_view": "moderate",
        },
        {
            "dropout_percentile": 82,
            "dropout_shape": 6.5,
            "library_log_mean": 4.6,
            "library_log_sd": 0.4,
            "measurement_seed_numpy": config["views"][1]["measurement_seed_numpy"],
            "measurement_seed_original": requests[1].measurement_seed,
            "outlier_mean": 0.8,
            "outlier_prob": 0.01,
            "outlier_scale": 1.0,
            "technical_view": "severe",
        },
    ]

    for artifact, request in zip(artifacts, requests, strict=True):
        assert artifact.request == request
        assert request.output_path.is_file()
        adata = artifact.adata
        assert adata.shape == (18, 20)
        assert adata.X.dtype == np.int64
        assert adata.layers["latent_expression"].dtype == np.float64
        assert adata.layers["pre_dropout_expression"].dtype == np.float64
        assert adata.uns["truth_kind"] == "exact_continuous"
        assert adata.uns["primary_truth_layer"] == "latent_expression"
        assert list(adata.uns["allowed_covariates"]["obs"]) == []
        assert list(adata.uns["allowed_covariates"]["var"]) == []
        assert set(adata.obs["group"]) == {
            f"cell-type-{index}" for index in range(1, 10)
        }
        assert all(
            int(value) == int(total)
            for value, total in zip(
                adata.obs["library_size"], np.asarray(adata.X).sum(axis=1), strict=True
            )
        )
        for index in range(1, 10):
            assert f"clean_log2fc_cell_type_{index}" in adata.var
            assert f"marker_cell_type_{index}" in adata.var
        provenance = adata.uns["provenance"]
        assert provenance["software"] == "SERGIO"
        assert provenance["software_version"] == SOURCE_RECEIPT["resolved_revision"]
        assert provenance["parameters"]["source_receipt"] == SOURCE_RECEIPT
        assert provenance["parameters"]["environment"] == ENVIRONMENT_RECEIPT
        assert provenance["parameters"]["score_truth"] == (
            "undefined_for_continuous_truth"
        )
        assert provenance["seeds"]["biological"] == request.biological_seed
        assert provenance["seeds"]["measurement"] == request.measurement_seed
        assert artifact.dataset_sha256 == benchmark_dataset_sha256(adata)
        assert {entry.path for entry in artifact.native_manifest.files} == NATIVE_FILES

    np.testing.assert_array_equal(
        artifacts[0].adata.layers["latent_expression"],
        artifacts[1].adata.layers["latent_expression"],
    )
    assert not np.array_equal(
        artifacts[0].adata.layers["pre_dropout_expression"],
        artifacts[1].adata.layers["pre_dropout_expression"],
    )
    assert not np.array_equal(artifacts[0].adata.X, artifacts[1].adata.X)


def test_development_profile_simulates_full_1200_gene_network_then_subsets() -> None:
    profile = sergio_module._profile_for_genes(500)

    assert profile["name"] == "De-noised_1200G_9T_300cPerT_6_DS3"
    assert profile["simulated_genes"] == 1200
    assert profile["interaction_path"].endswith("Interaction_cID_6.txt")
    assert profile["regulator_path"].endswith("Regs_cID_6.txt")


def test_pair_config_seals_only_logical_source_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _requests(tmp_path)
    by_view = {request.technical_view: request for request in requests}

    first = sergio_module._pair_config(by_view)
    monkeypatch.setattr(sergio_module, "_CHECKOUT", tmp_path / "relocated-checkout")
    monkeypatch.setattr(
        sergio_module,
        "_MODULE_PATH",
        tmp_path / "relocated-checkout/SERGIO/sergio.py",
    )
    second = sergio_module._pair_config(by_view)

    assert first == second
    assert first["source"] == {
        "commit": "a6190b74425112834c8fa9b4b6157d9cb3d1ab88",
        "module_path": "SERGIO/sergio.py",
        "tree": "15558fe60f62683c6fa46bcde01d9f3d3382e34a",
    }
    sealed = json.dumps(first, sort_keys=True)
    assert sergio_module._REPO_ROOT.as_posix() not in sealed


def test_profile_rejects_more_than_the_pinned_network_size() -> None:
    with pytest.raises(SimulationContractError, match="1200"):
        sergio_module._profile_for_genes(1201)


def test_oversized_npy_is_rejected_before_reading_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    oversized = tmp_path / "oversized.npy"
    with oversized.open("wb") as output:
        output.truncate(10_000_000)

    def forbidden_read(path: Path) -> bytes:
        raise AssertionError(f"read attempted for {path}")

    monkeypatch.setattr(sergio_module, "_read_regular_bytes", forbidden_read)

    with pytest.raises(SimulationContractError, match="size|large"):
        sergio_module._read_npy(
            oversized,
            dtype=np.dtype("<f8"),
            shape=(20, 18),
        )


def test_request_sequence_is_snapshotted_once(
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

    artifacts = sergio_module.run_sergio_pair(stateful, SMOKE_PROTOCOL)

    assert stateful.iterations == 1
    assert tuple(artifact.request for artifact in artifacts) == original
    assert not any(request.output_path.exists() for request in alternate)


def test_existing_result_is_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    requests[0].output_path.parent.mkdir(parents=True)
    requests[0].output_path.write_bytes(b"do not overwrite")

    with pytest.raises(SimulationContractError, match="exist|overwrite"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert calls == []
    assert requests[0].output_path.read_bytes() == b"do not overwrite"


@pytest.mark.parametrize(
    "corruption",
    [
        "extra-file",
        "symlink",
        "object-array",
        "float32-clean",
        "fortran-clean",
        "wrong-orientation",
        "nonfinite-clean",
        "wrong-indicator-dtype",
        "nonbinary-indicator",
        "wrong-observed-dtype",
        "negative-observed",
        "observed-where-dropped",
        "swapped-float-arrays",
        "wrong-call-count",
        "typed-seed",
        "wrong-version",
        "changed-config",
    ],
)
def test_malformed_extra_or_swapped_native_outputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    outside = tmp_path / "outside.npy"
    _save_array(outside, np.ones((20, 18), dtype="<f8"))

    def corrupt(output_dir: Path) -> None:
        target = output_dir / "clean.npy"
        if corruption == "extra-file":
            (output_dir / "undeclared.txt").write_text("extra", encoding="utf-8")
            return
        if corruption == "symlink":
            target.unlink()
            target.symlink_to(outside)
            return
        if corruption == "object-array":
            with target.open("wb") as output:
                np.save(output, np.ones((20, 18), dtype=object), allow_pickle=True)
        elif corruption == "float32-clean":
            _save_array(target, np.ones((20, 18), dtype="<f4"))
        elif corruption == "fortran-clean":
            _save_array(target, np.asfortranarray(np.ones((20, 18), dtype="<f8")))
        elif corruption == "wrong-orientation":
            _save_array(target, np.ones((18, 20), dtype="<f8"))
        elif corruption == "nonfinite-clean":
            values = np.ones((20, 18), dtype="<f8")
            values[0, 0] = np.nan
            _save_array(target, values)
        elif corruption == "wrong-indicator-dtype":
            _save_array(
                output_dir / "dropout_indicator_moderate.npy",
                np.ones((20, 18), dtype="<i8"),
            )
        elif corruption == "nonbinary-indicator":
            values = np.ones((20, 18), dtype=np.uint8)
            values[0, 0] = 2
            _save_array(output_dir / "dropout_indicator_moderate.npy", values)
        elif corruption == "wrong-observed-dtype":
            _save_array(
                output_dir / "observed_moderate.npy",
                np.ones((20, 18), dtype="<f8"),
            )
        elif corruption == "negative-observed":
            values = np.ones((20, 18), dtype="<i8")
            values[0, 0] = -1
            _save_array(output_dir / "observed_moderate.npy", values)
        elif corruption == "observed-where-dropped":
            indicator = np.load(
                output_dir / "dropout_indicator_moderate.npy", allow_pickle=False
            )
            row, column = np.argwhere(indicator == 0)[0]
            values = np.load(output_dir / "observed_moderate.npy", allow_pickle=False)
            values[row, column] = 1
            _save_array(output_dir / "observed_moderate.npy", values)
        elif corruption == "swapped-float-arrays":
            other = output_dir / "pre_dropout_moderate.npy"
            temporary = output_dir / "swap.tmp"
            target.rename(temporary)
            other.rename(target)
            temporary.rename(other)
            return
        elif corruption in {"wrong-call-count", "typed-seed", "wrong-version"}:
            metadata_path = output_dir / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if corruption == "wrong-call-count":
                metadata["call_counts"]["simulate"] = 2
            elif corruption == "typed-seed":
                metadata["biological_seed_numpy"] = float(
                    metadata["biological_seed_numpy"]
                )
            else:
                metadata["versions"]["numpy"] = "0.0.0-forged"
            _canonical_json(metadata_path, metadata)
            return
        elif corruption == "changed-config":
            config_path = output_dir / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["simulation"]["cells"] += 9
            _canonical_json(config_path, config)
            return
        _rehash_run_metadata(output_dir)

    _mock_external(monkeypatch, mutate=corrupt)

    with pytest.raises(SimulationContractError):
        sergio_module.run_sergio_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert outside.is_file() and not outside.is_symlink()


@pytest.mark.parametrize(
    "change",
    [
        {"technical_view": "other"},
        {"cells": 19},
        {"genes": 101},
    ],
)
def test_noncanonical_design_fails_before_native_execution(
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
                cells=int(change.get("cells", 18)),
                genes=int(change.get("genes", 20)),
            ),
        )

    with pytest.raises(SimulationContractError):
        sergio_module.run_sergio_pair((moderate, severe), protocol)
    assert calls == []


def test_source_and_environment_are_rechecked_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_receipts = [
        dict(SOURCE_RECEIPT),
        {**SOURCE_RECEIPT, "ledger_sha256": "d" * 64},
    ]
    environment_checks = 0

    def environment() -> dict[str, object]:
        nonlocal environment_checks
        environment_checks += 1
        return json.loads(json.dumps(ENVIRONMENT_RECEIPT))

    monkeypatch.setattr(
        sergio_module, "_verify_sergio_source", lambda: source_receipts.pop(0)
    )
    monkeypatch.setattr(sergio_module, "_environment_receipt", environment)
    monkeypatch.setattr(sergio_module, "_execute_sergio", _write_valid_native)

    with pytest.raises(SimulationContractError, match="source.*changed|pristine"):
        sergio_module.run_sergio_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert source_receipts == []
    assert environment_checks == 2


def test_source_and_environment_are_rechecked_when_runner_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = {"source": 0, "environment": 0}

    def source() -> dict[str, object]:
        checks["source"] += 1
        return dict(SOURCE_RECEIPT)

    def environment() -> dict[str, object]:
        checks["environment"] += 1
        return json.loads(json.dumps(ENVIRONMENT_RECEIPT))

    def fail_runner(*args: object, **kwargs: object) -> None:
        raise SimulationContractError("injected SERGIO failure")

    monkeypatch.setattr(sergio_module, "_verify_sergio_source", source)
    monkeypatch.setattr(sergio_module, "_environment_receipt", environment)
    monkeypatch.setattr(sergio_module, "_execute_sergio", fail_runner)

    with pytest.raises(SimulationContractError, match="injected SERGIO failure"):
        sergio_module.run_sergio_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert checks == {"source": 2, "environment": 2}


def test_source_and_environment_are_rechecked_before_propagating_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = {"source": 0, "environment": 0}

    def source() -> dict[str, object]:
        checks["source"] += 1
        return dict(SOURCE_RECEIPT)

    def environment() -> dict[str, object]:
        checks["environment"] += 1
        return json.loads(json.dumps(ENVIRONMENT_RECEIPT))

    def interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(sergio_module, "_verify_sergio_source", source)
    monkeypatch.setattr(sergio_module, "_environment_receipt", environment)
    monkeypatch.setattr(sergio_module, "_execute_sergio", interrupt)

    with pytest.raises(KeyboardInterrupt):
        sergio_module.run_sergio_pair(_requests(tmp_path), SMOKE_PROTOCOL)
    assert checks == {"source": 2, "environment": 2}


def test_both_h5ad_roundtrips_finish_before_either_publication(
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
            raise RuntimeError("injected second persistence failure")
        real_write(self, filename, *args, **kwargs)

    monkeypatch.setattr(ad.AnnData, "write_h5ad", fail_second_write)

    with pytest.raises(SimulationContractError, match="persist|serial"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert writes == 2
    assert not any(request.output_path.exists() for request in requests)
    assert not (requests[0].output_path.parent / "native").exists()


def test_post_link_read_failure_removes_the_just_linked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_read = ad.read_h5ad
    reads = 0

    def fail_first_publication_read(*args: object, **kwargs: object) -> ad.AnnData:
        nonlocal reads
        reads += 1
        if reads == 3:
            raise RuntimeError("injected post-link read failure")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(ad, "read_h5ad", fail_first_publication_read)

    with pytest.raises(SimulationContractError, match="persist"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert reads == 3
    assert not any(request.output_path.exists() for request in requests)
    assert not (requests[0].output_path.parent / "native").exists()


def test_post_link_interrupt_removes_the_just_linked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_read = ad.read_h5ad
    reads = 0

    def interrupt_first_publication_read(*args: object, **kwargs: object) -> ad.AnnData:
        nonlocal reads
        reads += 1
        if reads == 3:
            raise KeyboardInterrupt
        return real_read(*args, **kwargs)

    monkeypatch.setattr(ad, "read_h5ad", interrupt_first_publication_read)

    with pytest.raises(KeyboardInterrupt):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert reads == 3
    assert not any(request.output_path.exists() for request in requests)
    assert not (requests[0].output_path.parent / "native").exists()


def test_staged_pair_truth_equality_is_checked_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_read = ad.read_h5ad
    reads = 0

    def alter_second_staged_truth(*args: object, **kwargs: object) -> ad.AnnData:
        nonlocal reads
        reads += 1
        dataset = real_read(*args, **kwargs)
        if reads == 2:
            dataset.layers["latent_expression"][0, 0] += 1.0
        return dataset

    monkeypatch.setattr(ad, "read_h5ad", alter_second_staged_truth)

    with pytest.raises(SimulationContractError, match="truth|latent|identical"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert reads == 2
    assert not any(request.output_path.exists() for request in requests)
    assert not (requests[0].output_path.parent / "native").exists()


def test_published_roundtrip_must_match_its_staged_semantic_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_publish = sergio_module._publish_staged_h5ad
    publications = 0

    def alter_first_published_semantics(
        temporary: Path, destination: Path
    ) -> tuple[ad.AnnData, tuple[int, int]]:
        nonlocal publications
        publications += 1
        dataset, identity = real_publish(temporary, destination)
        if publications == 1:
            dataset.layers["pre_dropout_expression"][0, 0] += 1.0
        return dataset, identity

    monkeypatch.setattr(
        sergio_module, "_publish_staged_h5ad", alter_first_published_semantics
    )

    with pytest.raises(SimulationContractError, match="staged|semantic"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert publications == 1
    assert not any(request.output_path.exists() for request in requests)
    assert not (requests[0].output_path.parent / "native").exists()


def test_full_pair_contract_is_revalidated_immediately_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_validate = sergio_module.validate_paired_simulation_requests
    validations = 0

    def invalidate_second(*args: object, **kwargs: object) -> None:
        nonlocal validations
        validations += 1
        if validations == 2:
            assert not any(request.output_path.exists() for request in requests)
            raise SimulationContractError("controller changed before publication")
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        sergio_module, "validate_paired_simulation_requests", invalidate_second
    )

    with pytest.raises(SimulationContractError, match="controller changed"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert validations == 2
    assert not any(request.output_path.exists() for request in requests)


def test_development_pair_is_revalidated_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_validate = sergio_module.validate_paired_simulation_requests
    validations = 0

    def count(*args: object, **kwargs: object) -> None:
        nonlocal validations
        validations += 1
        real_validate(*args, **kwargs)

    monkeypatch.setattr(sergio_module, "validate_paired_simulation_requests", count)

    sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)

    assert validations == 3


def test_failed_postpublication_revalidation_rolls_back_unreceipted_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_validate = sergio_module.validate_paired_simulation_requests
    validations = 0

    def fail_third(*args: object, **kwargs: object) -> None:
        nonlocal validations
        validations += 1
        if validations == 3:
            assert all(request.output_path.is_file() for request in requests)
            raise SimulationContractError("postpublication controller transition")
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        sergio_module, "validate_paired_simulation_requests", fail_third
    )

    with pytest.raises(SimulationContractError, match="controller transition"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert validations == 3
    assert not any(request.output_path.exists() for request in requests)
    native_root = requests[0].output_path.parent / "native"
    assert not native_root.exists() or list(native_root.iterdir()) == []


def test_final_postpublication_uses_locked_lifecycle_helper_not_full_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    protocol = replace(
        SMOKE_PROTOCOL,
        final=replace(SMOKE_PROTOCOL.final, cells=18, genes=20),
    )
    requests = _requests(tmp_path, namespace=protocol.final.namespace)
    validations = 0
    lifecycle_claims: list[object] = []
    claim = object()

    def validate(*args: object, **kwargs: object) -> None:
        nonlocal validations
        validations += 1

    monkeypatch.setattr(sergio_module, "validate_paired_simulation_requests", validate)
    monkeypatch.setattr(
        sergio_module,
        "_revalidate_published_final_claim",
        lambda value: lifecycle_claims.append(value),
    )

    sergio_module.run_sergio_pair(requests, protocol, claim)  # type: ignore[arg-type]

    assert validations == 2
    assert lifecycle_claims == [claim]


def test_publication_failure_rolls_back_both_result_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)
    requests = _requests(tmp_path)
    real_publish = sergio_module._publish_staged_h5ad
    publications = 0

    def fail_second(temporary: Path, destination: Path) -> ad.AnnData:
        nonlocal publications
        publications += 1
        if publications == 2:
            raise SimulationContractError("injected publication failure")
        return real_publish(temporary, destination)

    monkeypatch.setattr(sergio_module, "_publish_staged_h5ad", fail_second)

    with pytest.raises(SimulationContractError, match="publication failure"):
        sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    assert publications == 2
    assert not any(request.output_path.exists() for request in requests)


def test_native_directory_fsync_failure_removes_new_content_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    for name in NATIVE_FILES:
        (stage / name).write_bytes(name.encode("ascii"))
    files = {path.name: path for path in stage.iterdir()}
    parent = tmp_path / "results"
    real_fsync = os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected native directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(sergio_module.os, "fsync", fail_directory_fsync)

    with pytest.raises(SimulationContractError, match="published"):
        sergio_module._publish_native_directory(files, parent)
    native_root = parent / "native"
    assert not native_root.exists() or list(native_root.iterdir()) == []


def test_native_directory_rename_is_atomic_no_replace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "ours").write_text("ours\n", encoding="utf-8")
    (destination / "theirs").write_text("theirs\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        sergio_module._rename_directory_no_replace(source, destination)

    assert (source / "ours").read_text(encoding="utf-8") == "ours\n"
    assert (destination / "theirs").read_text(encoding="utf-8") == "theirs\n"


def test_deterministic_rerun_reuses_content_addressed_native_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_external(monkeypatch)
    requests = _requests(tmp_path)

    first = sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    first_hashes = [artifact.dataset_sha256 for artifact in first]
    native_dirs = list((requests[0].output_path.parent / "native").iterdir())
    for request in requests:
        request.output_path.unlink()
    second = sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)

    assert len(calls) == 2
    assert first_hashes == [artifact.dataset_sha256 for artifact in second]
    assert list((requests[0].output_path.parent / "native").iterdir()) == native_dirs


def test_relocated_rerun_has_identical_manifest_and_semantic_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_external(monkeypatch)

    first = sergio_module.run_sergio_pair(_requests(tmp_path / "first"), SMOKE_PROTOCOL)
    relocated = sergio_module.run_sergio_pair(
        _requests(tmp_path / "relocated"), SMOKE_PROTOCOL
    )

    assert [artifact.native_manifest.manifest_sha256 for artifact in first] == [
        artifact.native_manifest.manifest_sha256 for artifact in relocated
    ]
    assert [artifact.dataset_sha256 for artifact in first] == [
        artifact.dataset_sha256 for artifact in relocated
    ]


def test_sergio_pair_is_exported_from_simulators_package() -> None:
    import maskimpute_benchmark.simulators as simulators

    assert simulators.run_sergio_pair is sergio_module.run_sergio_pair


def test_runner_imports_explicit_package_and_calls_exact_paired_sequence(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    package = checkout / "SERGIO"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    call_log = tmp_path / "calls.log"
    (package / "sergio.py").write_text(
        """import os
import numpy as np
assert np.int is int
assert np.float is float

def record(name):
    with open(os.environ['FAKE_SERGIO_CALL_LOG'], 'a', encoding='utf-8') as output:
        output.write(name + '\\n')

class sergio:
    def __init__(self, number_genes, number_bins, number_sc, **kwargs):
        record('constructor')
        self.nGenes_ = number_genes
        self.nBins_ = number_bins
        self.nSC_ = number_sc

    def build_graph(self, interaction, regulators, shared_coop_state):
        record('build_graph')

    def simulate(self):
        record('simulate')

    def getExpressions(self):
        record('getExpressions')
        shape = (self.nBins_, self.nGenes_, self.nSC_)
        return np.random.random(shape) + 0.25

    def outlier_effect(self, values, outlier_prob, mean, scale):
        record('outlier_effect')
        return np.asarray(values) * np.random.lognormal(mean, scale, (self.nBins_, 1, 1))

    def lib_size_effect(self, values, mean, scale):
        record('lib_size_effect')
        factors = np.random.lognormal(mean, scale, (self.nBins_, self.nSC_))
        return factors, np.asarray(values) * factors[:, None, :]

    def dropout_indicator(self, values, shape=1, percentile=65):
        record('dropout_indicator')
        probability = 0.75 if percentile == 65 else 0.25
        return np.random.binomial(1, probability, np.asarray(values).shape)

    def convert_to_UMIcounts(self, values):
        record('convert_to_UMIcounts')
        return np.random.poisson(values)
""",
        encoding="utf-8",
    )
    profile = sergio_module._profile_for_genes(20)
    for key in ("interaction_path", "regulator_path"):
        path = checkout / str(profile[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    output_dir = tmp_path / "native"
    output_dir.mkdir()
    requests = _requests(tmp_path / "requests")
    config = sergio_module._pair_config(
        {request.technical_view: request for request in requests}
    )
    config_path = output_dir / "config.json"
    _canonical_json(config_path, config)
    environment = dict(os.environ)
    environment["FAKE_SERGIO_CALL_LOG"] = call_log.as_posix()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/simulators/run_sergio.py",
            config_path.as_posix(),
            checkout.resolve().as_posix(),
            output_dir.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in output_dir.iterdir()} == NATIVE_FILES
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "constructor",
        "build_graph",
        "simulate",
        "getExpressions",
        "outlier_effect",
        "lib_size_effect",
        "dropout_indicator",
        "convert_to_UMIcounts",
        "outlier_effect",
        "lib_size_effect",
        "dropout_indicator",
        "convert_to_UMIcounts",
    ]
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert "runtime_seconds" not in metadata
    assert metadata["module_path"] == "SERGIO/sergio.py"
    assert metadata["compatibility_shim"] == config["adapter"]["compatibility_shim"]
    assert metadata["call_counts"]["sergio_constructor"] == 1
    assert metadata["call_counts"]["simulate"] == 1
    assert metadata["call_counts"]["outlier_effect"] == 2
    clean = np.load(output_dir / "clean.npy", allow_pickle=False)
    assert clean.dtype == np.float64 and clean.shape == (20, 18)


def test_runner_removes_partial_npy_when_numpy_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_path = Path("scripts/simulators/run_sergio.py").resolve()
    specification = importlib.util.spec_from_file_location(
        "maskimpute_test_run_sergio", runner_path
    )
    assert specification is not None and specification.loader is not None
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    destination = tmp_path / "partial.npy"

    def fail_save(output: object, *args: object, **kwargs: object) -> None:
        output.write(b"partial")
        raise OSError("injected NumPy save failure")

    monkeypatch.setattr(runner.np, "save", fail_save)

    with pytest.raises(OSError, match="NumPy save"):
        runner._write_npy(destination, np.ones((2, 2), dtype=np.float64))
    assert not destination.exists()


def test_runner_removes_partial_json_when_os_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_path = Path("scripts/simulators/run_sergio.py").resolve()
    specification = importlib.util.spec_from_file_location(
        "maskimpute_test_run_sergio_json", runner_path
    )
    assert specification is not None and specification.loader is not None
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    destination = tmp_path / "partial.json"
    real_write = os.write

    def fail_write(descriptor: int, data: object) -> int:
        real_write(descriptor, memoryview(data)[:3])
        raise OSError("injected JSON write failure")

    monkeypatch.setattr(runner.os, "write", fail_write)

    with pytest.raises(OSError, match="JSON write"):
        runner._write_json(destination, {"valid": True})
    assert not destination.exists()


def test_runner_rejects_hardlinked_pinned_source_file(tmp_path: Path) -> None:
    runner_path = Path("scripts/simulators/run_sergio.py").resolve()
    specification = importlib.util.spec_from_file_location(
        "maskimpute_test_run_sergio_hardlink", runner_path
    )
    assert specification is not None and specification.loader is not None
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("source = True\n", encoding="utf-8")
    pinned = checkout / "module.py"
    os.link(outside, pinned)

    with pytest.raises(ValueError, match="hard|unique"):
        runner._regular_path(pinned, checkout, "pinned module")


_CHECKOUT = Path("artifacts/external/checkouts/sergio")
_REAL_ASSETS_AVAILABLE = all(
    path.exists()
    for path in (
        _CHECKOUT / ".git",
        Path("artifacts/external/receipts/sergio.json"),
    )
)


@integration
@pytest.mark.skipif(
    not _REAL_ASSETS_AVAILABLE,
    reason="pristine pinned SERGIO checkout is unavailable",
)
def test_real_pinned_sergio_smoke_leaves_checkout_unchanged(tmp_path: Path) -> None:
    before_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    before_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    before_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    requests = _requests(tmp_path)
    first = sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)
    first_dataset_hashes = [artifact.dataset_sha256 for artifact in first]
    first_manifest_hashes = [
        artifact.native_manifest.manifest_sha256 for artifact in first
    ]
    first_truth = np.asarray(first[0].adata.layers["latent_expression"]).copy()
    first_observed = [np.asarray(artifact.adata.X).copy() for artifact in first]
    native_directories = list((requests[0].output_path.parent / "native").iterdir())
    for request in requests:
        request.output_path.unlink()

    second = sergio_module.run_sergio_pair(requests, SMOKE_PROTOCOL)

    after_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    after_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    after_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=_CHECKOUT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert (before_head, before_tree, before_status) == (
        "a6190b74425112834c8fa9b4b6157d9cb3d1ab88",
        "15558fe60f62683c6fa46bcde01d9f3d3382e34a",
        "",
    )
    assert (after_head, after_tree, after_status) == (
        before_head,
        before_tree,
        before_status,
    )
    assert first_dataset_hashes == [artifact.dataset_sha256 for artifact in second]
    assert first_manifest_hashes == [
        artifact.native_manifest.manifest_sha256 for artifact in second
    ]
    assert list((requests[0].output_path.parent / "native").iterdir()) == (
        native_directories
    )
    np.testing.assert_array_equal(
        first_truth,
        second[1].adata.layers["latent_expression"],
    )
    for expected, artifact in zip(first_observed, second, strict=True):
        np.testing.assert_array_equal(expected, artifact.adata.X)
    assert not np.array_equal(second[0].adata.X, second[1].adata.X)
