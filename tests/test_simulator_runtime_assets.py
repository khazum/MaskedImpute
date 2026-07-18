from __future__ import annotations

import hashlib
import json
import inspect
from pathlib import Path

import pytest

import maskimpute_benchmark.simulators.runtime_assets as runtime_module
from maskimpute_benchmark.simulators.semisynthetic import run_semisynthetic_pair
from maskimpute_benchmark.simulators.sergio import run_sergio_pair
from maskimpute_benchmark.simulators.sparsim import run_sparsim_pair
from maskimpute_benchmark.simulators.symsim import run_symsim_pair
from maskimpute_benchmark.protocol import canonical_sha256
from maskimpute_benchmark.runtime_environments import load_runtime_environment_lock
from maskimpute_benchmark.simulators.runtime_assets import (
    SimulatorRuntimeAssetsError,
    load_simulator_runtime_assets,
    revalidate_simulator_runtime_assets,
)


SOURCE_LEDGER_SHA256 = (
    "5a6f60c5de980a20eb118d0b82913112650f1956562aec4c92d37d8314c9f29e"
)
R_AUTHORITY = {
    "environment_id": "simulator-r",
    "lock_file_sha256": "b" * 64,
    "lock_path": "study/simulator_r_environment.lock.json",
    "tree_entry_count": 1,
    "tree_sha256": "d" * 64,
}
SOURCE_SNAPSHOT_AUTHORITY = {
    "tree_entry_count": 1,
    "tree_sha256": "f" * 64,
}
R_RECEIPT = {
    "environment_id": "simulator-r",
    "inventory_sha256": "c" * 64,
    "lock_file_sha256": "b" * 64,
    "schema": "maskimpute-simulator-r-runtime-receipt-v1",
}
SOURCE_RECEIPTS = tuple(
    {
        "schema_version": 1,
        "source_id": source_id,
        "role": (
            "semisynthetic_source" if source_id == "baron-pancreas-umi" else "mechanism"
        ),
        "source_type": "data" if source_id == "baron-pancreas-umi" else "git",
        "source_url": f"https://example.invalid/{source_id}",
        "revision": f"fixture-{source_id}",
        "license": "MIT",
        "citation_doi": "10.0000/fixture",
        "ledger_sha256": SOURCE_LEDGER_SHA256,
        "resolved_revision": f"fixture-{source_id}",
        "verified_checksum": (
            None
            if source_id == "baron-pancreas-umi"
            else {"algorithm": "fixture", "value": "a" * 64}
        ),
        **(
            {
                "artifacts": [
                    {
                        "name": "fixture-data.tar",
                        "sha256": "a" * 64,
                        "size_bytes": 128,
                    }
                ]
            }
            if source_id == "baron-pancreas-umi"
            else {}
        ),
    }
    for source_id in ("symsim", "sergio", "sparsim", "baron-pancreas-umi")
)


def test_tracked_simulator_r_authority_uses_current_full_closure_lock() -> None:
    repository = Path(__file__).resolve().parents[1]
    authority_path = repository / "study/simulator_runtime_assets.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    r_authority = authority["r_environment"]
    lock_path = repository / r_authority["lock_path"]

    lock = load_runtime_environment_lock(lock_path)
    entry = lock.by_id("simulator-r")
    inventory = entry.inventory

    assert lock.file_sha256 == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert r_authority["environment_id"] == entry.environment_id
    assert r_authority["lock_file_sha256"] == lock.file_sha256
    assert entry.inventory_sha256 == (
        "9d32b1df5a408cc6908ff40774482df9986d4f990e770593759e177faf421b48"
    )
    assert set(inventory) == {
        "schema",
        "interpreter",
        "packages",
        "executable_sha256",
        "launcher",
        "runtime_roots",
        "native_linkage_sha256",
    }
    assert inventory["runtime_roots"]


def test_tracked_simulator_r_authority_uses_sanctioned_search_shape() -> None:
    repository = Path(__file__).resolve().parents[1]
    authority = json.loads(
        (repository / "study/simulator_runtime_assets.json").read_text(encoding="utf-8")
    )
    lock = load_runtime_environment_lock(
        repository / authority["r_environment"]["lock_path"]
    )
    roots = lock.by_id("simulator-r").inventory["runtime_roots"]

    def search_shape(prefix: str) -> tuple[tuple[str, str, int, str], ...]:
        selected = tuple(
            (
                root["role"].removeprefix(prefix),
                root["kind"],
                root["entry_count"],
                root["content_sha256"],
            )
            for root in roots
            if root["role"].startswith(prefix)
        )
        assert tuple(row[0] for row in selected) == ("000", "001", "002", "003")
        assert all(row[1] == "search-directory" for row in selected)
        assert all("symlink-hop" not in row[0] for row in selected)
        return selected

    git_shape = search_shape("git-search-directory-")
    nvidia_shape = search_shape("nvidia-smi-search-directory-")

    assert tuple(row[1:] for row in git_shape) == tuple(row[1:] for row in nvidia_shape)
    assert not any(root["role"].startswith("loader-search-root-") for root in roots)


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    (repository / "study").mkdir(parents=True)
    authority = {
        "schema": "maskimpute-simulator-runtime-assets-authority-v1",
        "source_ledger_sha256": SOURCE_LEDGER_SHA256,
        "source_snapshot": SOURCE_SNAPSHOT_AUTHORITY,
        "r_environment": R_AUTHORITY,
    }
    (repository / "study/simulator_runtime_assets.json").write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (repository / "study/sources.json").write_text(
        '{"fixture":"tracked-source-authority"}\n', encoding="utf-8"
    )
    return repository


def _external_assets(tmp_path: Path, name: str) -> tuple[Path, Path]:
    external_root = tmp_path / f"external-{name}"
    r_environment = tmp_path / f"r-environment-{name}"
    external_root.mkdir()
    (r_environment / "bin").mkdir(parents=True)
    rscript = r_environment / "bin/Rscript"
    rscript.write_bytes(b"fixture-rscript\n")
    rscript.chmod(0o755)
    return external_root, r_environment


def _mock_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "_collect_source_receipts",
        lambda _ledger_path, _external_root: (
            SOURCE_LEDGER_SHA256,
            tuple(dict(value) for value in SOURCE_RECEIPTS),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "_r_environment_receipt",
        lambda _repository, _r_environment, _authority: dict(R_RECEIPT),
    )
    monkeypatch.setattr(
        runtime_module,
        "_directory_content_receipt",
        lambda _path: {
            "entry_count": R_AUTHORITY["tree_entry_count"],
            "sha256": R_AUTHORITY["tree_sha256"],
        },
    )
    monkeypatch.setattr(
        runtime_module,
        "_source_snapshot_content_receipt",
        lambda _path: {
            "entry_count": SOURCE_SNAPSHOT_AUTHORITY["tree_entry_count"],
            "sha256": SOURCE_SNAPSHOT_AUTHORITY["tree_sha256"],
        },
        raising=False,
    )


def test_semantic_receipt_excludes_machine_specific_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    first_paths = _external_assets(tmp_path, "first")
    second_paths = _external_assets(tmp_path, "second")
    _mock_semantics(monkeypatch)

    first = load_simulator_runtime_assets(
        repository,
        external_root=first_paths[0],
        r_environment=first_paths[1],
        require_outside_repository=True,
    )
    second = load_simulator_runtime_assets(
        repository,
        external_root=second_paths[0],
        r_environment=second_paths[1],
        require_outside_repository=True,
    )

    assert first.semantic_sha256 == second.semantic_sha256
    assert first.semantic_receipt == second.semantic_receipt
    encoded = json.dumps(first.semantic_receipt, sort_keys=True)
    assert str(first_paths[0]) not in encoded
    assert str(first_paths[1]) not in encoded
    assert first.semantic_sha256 == canonical_sha256(first.semantic_receipt)
    baron = next(
        receipt
        for receipt in first.semantic_receipt["source_receipts"]
        if receipt["source_id"] == "baron-pancreas-umi"
    )
    assert baron["artifacts"] == [{"name": "fixture-data.tar", "sha256": "a" * 64}]


def test_runtime_assets_context_releases_the_private_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    external_root, r_environment = _external_assets(tmp_path, "context")
    _mock_semantics(monkeypatch)

    with load_simulator_runtime_assets(
        repository,
        external_root=external_root,
        r_environment=r_environment,
        require_outside_repository=True,
    ) as assets:
        snapshot_root = Path(assets._snapshot_owner.name)
        assert snapshot_root.is_dir()
        assert assets.semantic_sha256 == canonical_sha256(assets.semantic_receipt)

    assert not snapshot_root.exists()
    assets.close()
    with pytest.raises(SimulatorRuntimeAssetsError, match="unavailable|invalid|path"):
        revalidate_simulator_runtime_assets(assets)


def test_final_runtime_paths_reject_repository_defaults_and_symlink_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    _mock_semantics(monkeypatch)
    inside_external = repository / "artifacts/external"
    inside_r = repository / "artifacts/envs/symsim-r44"
    inside_external.mkdir(parents=True)
    (inside_r / "bin").mkdir(parents=True)
    (inside_r / "bin/Rscript").write_bytes(b"fixture\n")

    with pytest.raises(SimulatorRuntimeAssetsError, match="outside.*repository"):
        load_simulator_runtime_assets(
            repository,
            external_root=inside_external,
            r_environment=inside_r,
            require_outside_repository=True,
        )

    outside_r = _external_assets(tmp_path, "outside-r")[1]
    with pytest.raises(SimulatorRuntimeAssetsError, match="outside.*repository"):
        load_simulator_runtime_assets(
            repository,
            external_root=repository.parent,
            r_environment=outside_r,
            require_outside_repository=True,
        )

    real_external, real_r = _external_assets(tmp_path, "real")
    linked_external = tmp_path / "linked-external"
    linked_external.symlink_to(real_external, target_is_directory=True)
    with pytest.raises(SimulatorRuntimeAssetsError, match="symlink"):
        load_simulator_runtime_assets(
            repository,
            external_root=linked_external,
            r_environment=real_r,
            require_outside_repository=True,
        )


def test_r_environment_receipt_rejects_internal_symlink_components(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "r-environment"
    real_bin = tmp_path / "real-bin"
    environment.mkdir()
    real_bin.mkdir()
    rscript = real_bin / "Rscript"
    rscript.write_bytes(b"fixture\n")
    rscript.chmod(0o755)
    (environment / "bin").symlink_to(real_bin, target_is_directory=True)

    with pytest.raises(SimulatorRuntimeAssetsError, match="symlink"):
        runtime_module._secure_directory(
            environment / "bin", "simulator R binary directory"
        )


def test_r_environment_receipt_rejects_symlinked_package_records(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "r-environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "conda-meta").mkdir()
    rscript = environment / "bin/Rscript"
    rscript.write_bytes(b"fixture\n")
    rscript.chmod(0o755)
    outside = tmp_path / "outside-record.json"
    outside.write_text(
        json.dumps(
            {"name": "r-base", "version": "4.4.3", "build": "fixture"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (environment / "conda-meta/r-base.json").symlink_to(outside)

    with pytest.raises(SimulatorRuntimeAssetsError, match="absolute symlink|escape"):
        runtime_module._directory_content_receipt(environment)


def test_revalidation_rejects_runtime_semantic_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    external_root, r_environment = _external_assets(tmp_path, "stable")
    _mock_semantics(monkeypatch)
    assets = load_simulator_runtime_assets(
        repository,
        external_root=external_root,
        r_environment=r_environment,
        require_outside_repository=True,
    )

    drifted = dict(R_RECEIPT)
    drifted["inventory_sha256"] = "d" * 64
    monkeypatch.setattr(
        runtime_module,
        "_r_environment_receipt",
        lambda _repository, _path, _authority: drifted,
    )

    with pytest.raises(SimulatorRuntimeAssetsError, match="authority|drift"):
        revalidate_simulator_runtime_assets(assets)


def test_all_four_adapters_accept_the_immutable_runtime_contract() -> None:
    for adapter in (
        run_symsim_pair,
        run_sergio_pair,
        run_sparsim_pair,
        run_semisynthetic_pair,
    ):
        parameter = inspect.signature(adapter).parameters["runtime_assets"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None


def test_runtime_source_collection_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Ledger:
        sha256 = SOURCE_LEDGER_SHA256

    calls: list[tuple[Path, tuple[str, ...]]] = []

    def verify(_ledger: object, root: Path, *, source_ids: tuple[str, ...]):
        calls.append((root, source_ids))
        return tuple(dict(value) for value in SOURCE_RECEIPTS)

    def forbidden_fetch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("final runtime validation must not fetch or write")

    monkeypatch.setattr(runtime_module, "load_source_ledger", lambda _path: Ledger())
    monkeypatch.setattr(runtime_module, "verify_fetched_sources", verify, raising=False)
    monkeypatch.setattr(runtime_module, "fetch_sources", forbidden_fetch, raising=False)
    root = tmp_path / "external"
    root.mkdir()

    ledger_sha256, receipts = runtime_module._collect_source_receipts(
        tmp_path / "sources.json", root
    )

    assert ledger_sha256 == SOURCE_LEDGER_SHA256
    assert tuple(value["source_id"] for value in receipts) == (
        "baron-pancreas-umi",
        "sergio",
        "sparsim",
        "symsim",
    )
    assert calls == [
        (
            root,
            ("baron-pancreas-umi", "sergio", "sparsim", "symsim"),
        )
    ]


def test_runtime_uses_private_snapshot_when_authority_path_is_mutated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    external_root, r_environment = _external_assets(tmp_path, "identity")
    checkout_file = external_root / "checkouts/symsim/DESCRIPTION"
    checkout_file.parent.mkdir(parents=True)
    checkout_file.write_bytes(b"validated-source\n")
    _mock_semantics(monkeypatch)
    assets = load_simulator_runtime_assets(
        repository,
        external_root=external_root,
        r_environment=r_environment,
        require_outside_repository=True,
    )

    checkout_file.write_bytes(b"adversarial-replacement\n")
    snapshot_external, _snapshot_r, _digest = (
        runtime_module.simulator_runtime_asset_values(assets)
    )

    assert snapshot_external != external_root
    assert (
        snapshot_external / "checkouts/symsim/DESCRIPTION"
    ).read_bytes() == b"validated-source\n"


def test_loader_rejects_transient_source_replacement_copied_into_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    external_root, r_environment = _external_assets(tmp_path, "copy-race")
    checkout_file = external_root / "checkouts/symsim/DESCRIPTION"
    checkout_file.parent.mkdir(parents=True)
    checkout_file.write_bytes(b"validated-source\n")
    actual_receipt = runtime_module._source_snapshot_content_receipt
    baseline = tmp_path / "baseline"
    runtime_module._copy_source_snapshot(external_root, baseline)
    source_authority = actual_receipt(baseline)
    authority_path = repository / "study/simulator_runtime_assets.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["source_snapshot"] = {
        "tree_entry_count": source_authority["entry_count"],
        "tree_sha256": source_authority["sha256"],
    }
    authority_path.write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _mock_semantics(monkeypatch)
    monkeypatch.setattr(
        runtime_module, "_source_snapshot_content_receipt", actual_receipt
    )
    copy_source_snapshot = runtime_module._copy_source_snapshot

    def copy_with_transient_replacement(source: Path, destination: Path) -> None:
        copy_source_snapshot(source, destination)
        copied = destination / "checkouts/symsim/DESCRIPTION"
        copied.write_bytes(b"transient-adversarial-source\n")

    monkeypatch.setattr(
        runtime_module, "_copy_source_snapshot", copy_with_transient_replacement
    )

    with pytest.raises(
        SimulatorRuntimeAssetsError, match="copied simulator source snapshot"
    ):
        load_simulator_runtime_assets(
            repository,
            external_root=external_root,
            r_environment=r_environment,
            require_outside_repository=True,
        )


def test_runtime_snapshot_detects_mutate_and_restore_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    external_root, r_environment = _external_assets(tmp_path, "restore")
    source = external_root / "checkouts/symsim/DESCRIPTION"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"validated-source\n")
    _mock_semantics(monkeypatch)
    assets = load_simulator_runtime_assets(
        repository,
        external_root=external_root,
        r_environment=r_environment,
        require_outside_repository=True,
    )
    snapshot_source = assets.external_root / "checkouts/symsim/DESCRIPTION"
    snapshot_source.chmod(0o644)
    snapshot_source.write_bytes(b"adversarial-replacement\n")
    snapshot_source.write_bytes(b"validated-source\n")
    snapshot_source.chmod(0o444)

    with pytest.raises(SimulatorRuntimeAssetsError, match="identity drift"):
        runtime_module.revalidate_simulator_runtime_asset_identity(assets)


def test_r_receipt_uses_the_content_inventory_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    _external, r_environment = _external_assets(tmp_path, "content-lock")
    lock_path = repository / str(R_AUTHORITY["lock_path"])
    lock_path.write_bytes(b"fixture-lock\n")
    loaded = object()
    calls: list[object] = []
    monkeypatch.setattr(
        runtime_module, "load_runtime_environment_lock", lambda _path: loaded
    )

    def validate(lock: object, environments: object):
        calls.extend((lock, environments))
        return {
            "lock_file_sha256": R_AUTHORITY["lock_file_sha256"],
            "environment_inventory_sha256s": (
                ("simulator-r", R_RECEIPT["inventory_sha256"]),
            ),
        }

    monkeypatch.setattr(runtime_module, "validate_runtime_environment_lock", validate)
    monkeypatch.setattr(
        runtime_module,
        "_secure_lock_file_sha256",
        lambda _path: R_AUTHORITY["lock_file_sha256"],
        raising=False,
    )

    receipt = runtime_module._r_environment_receipt(
        repository, r_environment, R_AUTHORITY
    )

    assert receipt == R_RECEIPT
    assert calls == [
        loaded,
        {"simulator-r": ("r", r_environment / "bin/Rscript")},
    ]


@pytest.mark.parametrize(
    "relative",
    (
        "bin/Rscript",
        "lib/libR.so",
        "lib/R/library/runtimeFixture/R/function.R",
    ),
)
def test_r_tree_receipt_changes_for_executable_native_or_package_byte_drift(
    tmp_path: Path, relative: str
) -> None:
    environment = tmp_path / "r-environment"
    files = (
        "bin/Rscript",
        "lib/libR.so",
        "lib/R/library/runtimeFixture/R/function.R",
    )
    for name in files:
        path = environment / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"validated:{name}\n".encode())
    (environment / "bin/Rscript").chmod(0o755)
    before = runtime_module._directory_content_receipt(environment)

    (environment / relative).write_bytes(b"drifted-execution-byte\n")
    after = runtime_module._directory_content_receipt(environment)

    assert after["sha256"] != before["sha256"]
