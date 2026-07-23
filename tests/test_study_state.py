import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import threading

import pytest

import maskimpute_benchmark.study as study_module

from maskimpute_benchmark.study import (
    StudyStateError,
    assert_final_runnable,
    freeze_round,
    materialize_final,
    record_incremental_results,
    record_final_evaluation,
    supersede_round,
)
from maskimpute_benchmark.protocol import canonical_sha256, file_sha256


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


@pytest.fixture
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Study Test")
    _git(repo, "config", "user.email", "study@example.invalid")

    (repo / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    (repo / "tracked.py").write_text("original\n", encoding="utf-8")
    (repo / "config.json").write_text(
        json.dumps({"method": "maskimpute", "rank": 16}), encoding="utf-8"
    )
    (repo / "environment.lock").write_text(
        "python=3.11\nnumpy=2.1.2\ntorch=2.9.0\n", encoding="utf-8"
    )
    protocol = json.loads(Path("study/protocol.json").read_text(encoding="utf-8"))
    (repo / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "freeze inputs")
    return repo


def freeze_fixture(repo: Path) -> Path:
    round_dir = repo / "artifacts/study/round-001"
    freeze_round(
        repo,
        round_dir,
        repo / "config.json",
        repo / "protocol.json",
        environment_path=repo / "environment.lock",
    )
    return round_dir


def test_final_cannot_materialize_before_freeze(clean_repo: Path) -> None:
    with pytest.raises(StudyStateError, match="must be frozen"):
        materialize_final(clean_repo / "artifacts/study/round-001", seed_count=4)


def _result_manifest(round_dir: Path, *relative_paths: str) -> dict[str, object]:
    return {
        "result_files": [
            {
                "path": f"results/{relative}",
                "sha256": file_sha256(round_dir / "results" / relative),
            }
            for relative in sorted(relative_paths)
        ]
    }


def test_running_round_journals_incremental_results_for_claim_revalidation(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    claim = assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    (results / "first.txt").write_text("first\n", encoding="utf-8")

    first = record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "first.txt"),
        repo=clean_repo,
    )

    assert first["sequence"] == 1
    assert first["execution_claim_id"] == claim["execution_claim_id"]
    (results / "second.txt").write_text("second\n", encoding="utf-8")
    second = record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "first.txt", "second.txt"),
        repo=clean_repo,
    )
    assert second["sequence"] == 2

    from maskimpute_benchmark.simulators import load_final_manifest_claim

    loaded = load_final_manifest_claim(clean_repo, round_dir)
    assert loaded.execution_claim_id == claim["execution_claim_id"]


def test_incremental_result_journal_recovers_publication_before_record(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    (results / "recover.txt").write_text("recover\n", encoding="utf-8")

    from maskimpute_benchmark.simulators import load_final_manifest_claim

    with pytest.raises(Exception, match="clean frozen|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)

    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "recover.txt"),
        repo=clean_repo,
    )
    assert load_final_manifest_claim(clean_repo, round_dir).round_id == "round-001"


def test_incremental_result_journal_rejects_omission_extra_file_and_symlink(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    first = results / "first.txt"
    first.write_text("first\n", encoding="utf-8")
    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "first.txt"),
        repo=clean_repo,
    )

    with pytest.raises(study_module.StudyStateError, match="omit|monotonic|previous"):
        record_incremental_results(round_dir, {"result_files": []}, repo=clean_repo)

    extra = results / "extra.txt"
    extra.write_text("extra\n", encoding="utf-8")
    with pytest.raises(study_module.StudyStateError, match="clean frozen|unchanged"):
        record_incremental_results(
            round_dir,
            _result_manifest(round_dir, "first.txt"),
            repo=clean_repo,
        )
    extra.unlink()

    outside = clean_repo / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    symlink = results / "link.txt"
    symlink.symlink_to(outside)
    manifest = _result_manifest(round_dir, "first.txt")
    manifest["result_files"].append(
        {"path": "results/link.txt", "sha256": file_sha256(outside)}
    )
    with pytest.raises(study_module.StudyStateError, match="regular file|invalid"):
        record_incremental_results(round_dir, manifest, repo=clean_repo)


def test_final_evaluation_reconciles_incremental_journal_superset(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    (results / "first.txt").write_text("first\n", encoding="utf-8")
    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "first.txt"),
        repo=clean_repo,
    )
    (results / "last.txt").write_text("last\n", encoding="utf-8")
    manifest = _result_manifest(round_dir, "first.txt", "last.txt")

    receipt = record_final_evaluation(round_dir, manifest, repo=clean_repo)

    assert receipt["result_manifest"] == manifest


def test_result_journal_is_private_durable_and_authority_bound(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_modes: list[int] = []
    original_fsync = os.fsync

    def observed_fsync(descriptor: int) -> None:
        fsync_modes.append(os.fstat(descriptor).st_mode)
        original_fsync(descriptor)

    monkeypatch.setattr(study_module.os, "fsync", observed_fsync)
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    (results / "durable.txt").write_text("durable\n", encoding="utf-8")
    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "durable.txt"),
        repo=clean_repo,
    )

    root, journal, _root_identity, _journal_identity = (
        study_module._result_journal_directories(
            clean_repo, round_dir.name, create=False
        )
    )
    entry = journal / "00000001.json"
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(journal.stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600
    assert root.stat().st_uid == os.geteuid()
    assert journal.stat().st_uid == os.geteuid()
    assert entry.stat().st_uid == os.geteuid()
    assert entry.stat().st_nlink == 1
    assert any(stat.S_ISREG(mode) for mode in fsync_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsync_modes)


def test_result_journal_rejects_gap_extra_entry_tamper_and_directory_replacement(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    (results / "bound.txt").write_text("bound\n", encoding="utf-8")
    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "bound.txt"),
        repo=clean_repo,
    )
    from maskimpute_benchmark.simulators import load_final_manifest_claim

    _root, journal, _root_identity, _journal_identity = (
        study_module._result_journal_directories(
            clean_repo, round_dir.name, create=False
        )
    )
    entry = journal / "00000001.json"
    gap = journal / "00000002.json"
    entry.rename(gap)
    with pytest.raises(Exception, match="gap|extra|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)
    gap.rename(entry)

    extra = journal / ".entry.tmp"
    extra.write_text("{}\n", encoding="utf-8")
    extra.chmod(0o600)
    with pytest.raises(Exception, match="gap|extra|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)
    extra.unlink()

    record = json.loads(entry.read_text(encoding="utf-8"))
    record["new_result_files"][0]["sha256"] = "0" * 64
    entry.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Exception, match="digest|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)

    original = journal.with_name("round-001-original")
    journal.rename(original)
    journal.mkdir(mode=0o700)
    with pytest.raises(Exception, match="identity|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)


def test_result_journal_rejects_boolean_schema_version(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    (results / "bound.txt").write_text("bound\n", encoding="utf-8")
    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "bound.txt"),
        repo=clean_repo,
    )
    _root, journal, _root_identity, _journal_identity = (
        study_module._result_journal_directories(
            clean_repo, round_dir.name, create=False
        )
    )
    entry_path = journal / "00000001.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    entry["schema_version"] = True
    entry.pop("entry_sha256")
    entry["entry_sha256"] = canonical_sha256(entry)
    entry_path.write_text(
        json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from maskimpute_benchmark.simulators import (
        SimulationContractError,
        load_final_manifest_claim,
    )

    with pytest.raises(SimulationContractError, match="valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)


def test_incremental_journal_rejects_hardlinks_and_postrecord_mutation(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    outside = clean_repo / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    hardlink = results / "hardlink.txt"
    os.link(outside, hardlink)
    with pytest.raises(StudyStateError, match="regular file"):
        record_incremental_results(
            round_dir,
            _result_manifest(round_dir, "hardlink.txt"),
            repo=clean_repo,
        )
    hardlink.unlink()
    outside.unlink()

    result = results / "immutable.txt"
    result.write_text("before\n", encoding="utf-8")
    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "immutable.txt"),
        repo=clean_repo,
    )
    result.write_text("after\n", encoding="utf-8")
    from maskimpute_benchmark.simulators import load_final_manifest_claim

    with pytest.raises(Exception, match="hash|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)


def test_incremental_journal_postpublication_failure_supersedes_round(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    result = results / "mutated-after-entry.txt"
    result.write_text("before\n", encoding="utf-8")
    real_validate = study_module._validate_result_journal
    validations = 0

    def mutate_before_postpublication_validation(*args, **kwargs):
        nonlocal validations
        validations += 1
        if validations == 2:
            result.write_text("after\n", encoding="utf-8")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(
        study_module,
        "_validate_result_journal",
        mutate_before_postpublication_validation,
    )

    with pytest.raises(StudyStateError, match="hash"):
        record_incremental_results(
            round_dir,
            _result_manifest(round_dir, "mutated-after-entry.txt"),
            repo=clean_repo,
        )

    supersession = json.loads(
        (round_dir / "supersession.json").read_text(encoding="utf-8")
    )
    assert supersession["state"] == "superseded"
    registry = json.loads(
        study_module._registry_path(clean_repo, round_dir.name).read_text(
            encoding="utf-8"
        )
    )
    assert registry["state"] == "superseded"


def test_result_journal_rejects_relaxed_private_modes(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    results = round_dir / "results"
    results.mkdir()
    result = results / "private.txt"
    result.write_text("private\n", encoding="utf-8")
    record_incremental_results(
        round_dir,
        _result_manifest(round_dir, "private.txt"),
        repo=clean_repo,
    )
    from maskimpute_benchmark.simulators import load_final_manifest_claim

    root, journal, _root_identity, _journal_identity = (
        study_module._result_journal_directories(
            clean_repo, round_dir.name, create=False
        )
    )
    entry = journal / "00000001.json"
    entry.chmod(0o640)
    with pytest.raises(Exception, match="private|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)
    entry.chmod(0o600)

    journal.chmod(0o750)
    with pytest.raises(Exception, match="private|authority|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)
    journal.chmod(0o700)

    root.chmod(0o750)
    with pytest.raises(Exception, match="private|authority|valid claimed"):
        load_final_manifest_claim(clean_repo, round_dir)


def test_freeze_rejects_dirty_repository(clean_repo: Path) -> None:
    (clean_repo / "tracked.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(StudyStateError, match="clean"):
        freeze_fixture(clean_repo)


def test_freeze_rejects_ignored_executable_state(clean_repo: Path) -> None:
    with (clean_repo / ".gitignore").open("a", encoding="utf-8") as ignore:
        ignore.write("*.pyc\n")
    _git(clean_repo, "add", ".gitignore")
    _git(clean_repo, "commit", "-m", "ignore bytecode")
    (clean_repo / "payload.pyc").write_bytes(b"unbound executable bytes")
    assert _git(clean_repo, "status", "--porcelain", "--untracked-files=all") == ""

    with pytest.raises(StudyStateError, match="clean"):
        freeze_fixture(clean_repo)


def test_operational_root_is_receipt_bound_across_final_lifecycle(
    clean_repo: Path,
) -> None:
    environment_root = clean_repo / "artifacts/envs/magic-python"
    executable = environment_root / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"fixed runtime bytes\n")
    executable.chmod(0o755)
    source_root = clean_repo / "artifacts/method-sources/magic"
    source_root.mkdir(parents=True)
    source = source_root / "magic.py"
    source.write_bytes(b"fixed source bytes\n")
    round_dir = clean_repo / "artifacts/study/round-001"

    frozen = freeze_round(
        clean_repo,
        round_dir,
        clean_repo / "config.json",
        clean_repo / "protocol.json",
        environment_path=clean_repo / "environment.lock",
        operational_artifact_roots=(environment_root.parent, source_root.parent),
    )

    assert [row["path"] for row in frozen["operational_artifact_roots"]] == [
        "artifacts/envs",
        "artifacts/method-sources",
    ]
    assert all(
        row["entry_count"] > 0 and len(row["tree_sha256"]) == 64
        for row in frozen["operational_artifact_roots"]
    )
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    source.write_bytes(b"mutated source bytes\n")

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)


def test_operational_root_rejects_symlink_to_mutable_outside_content(
    clean_repo: Path,
) -> None:
    operational_root = clean_repo / "artifacts/envs"
    operational_root.mkdir(parents=True)
    outside = clean_repo / "config.json"
    os.symlink(outside, operational_root / "python")

    with pytest.raises(StudyStateError, match="symlink.*outside|closed"):
        freeze_round(
            clean_repo,
            clean_repo / "artifacts/study/round-001",
            clean_repo / "config.json",
            clean_repo / "protocol.json",
            environment_path=clean_repo / "environment.lock",
            operational_artifact_roots=(operational_root,),
        )


@pytest.mark.parametrize("kind", ["fifo", "empty_directory"])
def test_freeze_rejects_ignored_paths_git_does_not_list(
    clean_repo: Path, kind: str
) -> None:
    with (clean_repo / ".gitignore").open("a", encoding="utf-8") as ignore:
        ignore.write("ignored-state/\n")
    _git(clean_repo, "add", ".gitignore")
    _git(clean_repo, "commit", "-m", "ignore runtime state")
    ignored = clean_repo / "ignored-state"
    ignored.mkdir()
    if kind == "fifo":
        os.mkfifo(ignored / "control")
    assert _git(clean_repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert (
        _git(
            clean_repo,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
        )
        == ""
    )

    with pytest.raises(StudyStateError, match="clean"):
        freeze_fixture(clean_repo)


def test_materialization_rejects_ignored_state_outside_round(clean_repo: Path) -> None:
    with (clean_repo / ".gitignore").open("a", encoding="utf-8") as ignore:
        ignore.write("*.pyc\n")
    _git(clean_repo, "add", ".gitignore")
    _git(clean_repo, "commit", "-m", "ignore bytecode")
    round_dir = freeze_fixture(clean_repo)
    (clean_repo / "payload.pyc").write_bytes(b"late executable bytes")

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_materialization_rejects_unexpected_ignored_round_file(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    (round_dir / "payload.pyc").write_bytes(b"round-local executable bytes")

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_freeze_records_commit_relative_inputs_hashes_and_utc_time(
    clean_repo: Path,
) -> None:
    round_dir = clean_repo / "artifacts/study/round-001"
    record = freeze_round(
        clean_repo,
        round_dir,
        clean_repo / "config.json",
        clean_repo / "protocol.json",
        environment_path=clean_repo / "environment.lock",
    )

    assert record == json.loads((round_dir / "freeze.json").read_text(encoding="utf-8"))
    assert record["state"] == "frozen"
    assert record["round_id"] == "round-001"
    assert record["method_commit"] == _git(clean_repo, "rev-parse", "HEAD")
    assert record["config_path"] == "config.json"
    assert record["protocol_path"] == "protocol.json"
    assert record["config_sha256"] == file_sha256(clean_repo / "config.json")
    assert record["protocol_sha256"] == file_sha256(clean_repo / "protocol.json")
    assert record["environment_path"] == "environment.lock"
    assert record["environment_sha256"] == file_sha256(clean_repo / "environment.lock")
    assert record["round_path"] == "artifacts/study/round-001"
    assert len(record["round_token"]) == 32
    assert len(record["repository_instance_id"]) == 32
    assert len(record["worktree_path_sha256"]) == 64
    assert type(record["git_common_dir_device"]) is int
    assert record["git_common_dir_device"] >= 0
    assert type(record["git_common_dir_inode"]) is int
    assert record["git_common_dir_inode"] > 0
    assert type(record["study_state_root_device"]) is int
    assert type(record["study_state_root_inode"]) is int
    assert record["study_state_root_inode"] > 0
    assert type(record["registry_dir_device"]) is int
    assert type(record["registry_dir_inode"]) is int
    assert record["registry_dir_inode"] > 0
    frozen_at = datetime.fromisoformat(record["frozen_at"].replace("Z", "+00:00"))
    assert frozen_at.tzinfo == timezone.utc


def test_freeze_rejects_mismatched_validated_hash_handoff(clean_repo: Path) -> None:
    round_dir = clean_repo / "artifacts/study/round-001"

    with pytest.raises(StudyStateError, match="validated config checksum"):
        freeze_round(
            clean_repo,
            round_dir,
            clean_repo / "config.json",
            clean_repo / "protocol.json",
            environment_path=clean_repo / "environment.lock",
            expected_config_sha256="0" * 64,
            expected_protocol_sha256=file_sha256(clean_repo / "protocol.json"),
            expected_environment_sha256=file_sha256(clean_repo / "environment.lock"),
        )

    assert not (round_dir / "freeze.json").exists()


def test_freeze_rejects_mismatched_validated_commit_handoff(clean_repo: Path) -> None:
    round_dir = clean_repo / "artifacts/study/round-001"

    with pytest.raises(StudyStateError, match="validated method commit"):
        freeze_round(
            clean_repo,
            round_dir,
            clean_repo / "config.json",
            clean_repo / "protocol.json",
            environment_path=clean_repo / "environment.lock",
            expected_config_sha256=file_sha256(clean_repo / "config.json"),
            expected_protocol_sha256=file_sha256(clean_repo / "protocol.json"),
            expected_environment_sha256=file_sha256(clean_repo / "environment.lock"),
            expected_method_commit="0" * 40,
        )

    assert not (round_dir / "freeze.json").exists()


@pytest.mark.parametrize("authority", ["state_root", "locks", "registry"])
def test_freeze_rejects_symlinked_authority_directories(
    clean_repo: Path, tmp_path: Path, authority: str
) -> None:
    common = Path(_git(clean_repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = clean_repo / common
    external = tmp_path / f"external-{authority}"
    external.mkdir()
    state_root = common / "maskimpute-study"
    if authority == "state_root":
        state_root.symlink_to(external, target_is_directory=True)
    else:
        state_root.mkdir(mode=0o700)
        (state_root / f".{authority}").symlink_to(external, target_is_directory=True)

    with pytest.raises(StudyStateError, match="authority|directory|lock"):
        freeze_fixture(clean_repo)

    assert list(external.iterdir()) == []


@pytest.mark.parametrize("record_name", ["instance", "registry"])
def test_materialization_rejects_symlinked_authority_records(
    clean_repo: Path, tmp_path: Path, record_name: str
) -> None:
    round_dir = freeze_fixture(clean_repo)
    if record_name == "instance":
        record_path = study_module._study_state_root(clean_repo) / "instance.json"
    else:
        record_path = study_module._registry_path(clean_repo, "round-001")
    external = tmp_path / f"external-{record_name}.json"
    shutil.copyfile(record_path, external)
    record_path.unlink()
    record_path.symlink_to(external)

    with pytest.raises(StudyStateError, match="record|authority|clean frozen"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


@pytest.mark.parametrize("authority", ["state_root", "registry"])
def test_materialization_rejects_replaced_authority_directory(
    clean_repo: Path, tmp_path: Path, authority: str
) -> None:
    round_dir = freeze_fixture(clean_repo)
    state_root = study_module._study_state_root(clean_repo)
    target = (
        state_root
        if authority == "state_root"
        else state_root / study_module.REGISTRY_DIR_NAME
    )
    snapshot = tmp_path / f"snapshot-{authority}"
    shutil.copytree(target, snapshot)
    target.rename(tmp_path / f"old-{authority}")
    shutil.copytree(snapshot, target)

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_materialization_creates_unique_63_bit_generator_seeds(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    record = materialize_final(round_dir, seed_count=32, repo=clean_repo)
    seed_manifest = json.loads(
        (round_dir / "final_manifest.json").read_text(encoding="utf-8")
    )

    seeds = seed_manifest["generator_seeds"]
    assert record["state"] == "materialized"
    assert len(seeds) == 32
    assert len(set(seeds)) == 32
    assert all(type(seed) is int and 0 <= seed < 2**63 for seed in seeds)
    assert record["seed_manifest_sha256"] == canonical_sha256(seed_manifest)


def test_explicit_repo_makes_relative_round_path_repository_relative(
    clean_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    freeze_fixture(clean_repo)
    monkeypatch.chdir(tmp_path)

    record = materialize_final(
        Path("artifacts/study/round-001"), seed_count=4, repo=clean_repo
    )

    assert record["state"] == "materialized"


def test_dirty_or_changed_commit_cannot_run_final(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    (clean_repo / "tracked.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)


def test_changed_commit_cannot_run_final_even_when_repository_is_clean(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    (clean_repo / "tracked.py").write_text("new commit\n", encoding="utf-8")
    _git(clean_repo, "add", "tracked.py")
    _git(clean_repo, "commit", "-m", "change method")

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)


def test_changed_frozen_input_cannot_run_final_when_git_hides_worktree_change(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    _git(clean_repo, "update-index", "--assume-unchanged", "config.json")
    (clean_repo / "config.json").write_text('{"rank": 99}\n', encoding="utf-8")
    assert _git(clean_repo, "status", "--porcelain") == ""

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)


def test_hidden_tracked_code_change_cannot_materialize(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    _git(clean_repo, "update-index", "--assume-unchanged", "tracked.py")
    (clean_repo / "tracked.py").write_text("hidden change\n", encoding="utf-8")
    assert _git(clean_repo, "status", "--porcelain") == ""

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_changed_environment_cannot_materialize_when_git_hides_change(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    _git(clean_repo, "update-index", "--skip-worktree", "environment.lock")
    (clean_repo / "environment.lock").write_text("python=9.9\n", encoding="utf-8")

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_clean_materialized_round_is_runnable(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)

    record = assert_final_runnable(clean_repo, round_dir)

    assert record["state"] == "running"
    assert record["round_id"] == "round-001"
    assert (round_dir / "execution_claim.json").exists()


def test_final_execution_claim_is_atomic_and_one_use(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)

    assert_final_runnable(clean_repo, round_dir)
    with pytest.raises(StudyStateError, match="already claimed"):
        assert_final_runnable(clean_repo, round_dir)


def test_evaluation_receipt_is_one_use(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    record_final_evaluation(round_dir, {"results_sha256": "a" * 64}, repo=clean_repo)
    with pytest.raises(StudyStateError, match="already evaluated"):
        assert_final_runnable(clean_repo, round_dir)

    with pytest.raises(StudyStateError, match="already evaluated"):
        record_final_evaluation(
            round_dir, {"results_sha256": "b" * 64}, repo=clean_repo
        )


def test_evaluation_receipt_records_result_manifest_and_hash(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    result_manifest = {"results_sha256": "a" * 64, "rows": 24}

    record = record_final_evaluation(round_dir, result_manifest, repo=clean_repo)

    assert record["state"] == "evaluated"
    assert record["result_manifest"] == result_manifest
    assert record["result_manifest_sha256"] == canonical_sha256(result_manifest)
    assert record["environment_sha256"] == file_sha256(clean_repo / "environment.lock")
    assert record == json.loads(
        (round_dir / "evaluation_receipt.json").read_text(encoding="utf-8")
    )


def test_result_manifest_is_detached_from_caller_mutation(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    supplied = {"metadata": {"version": 1}}
    original = study_module._validate_result_files
    calls = 0

    def mutate_caller_after_validation(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            supplied["metadata"]["version"] = 2
        return result

    monkeypatch.setattr(
        study_module, "_validate_result_files", mutate_caller_after_validation
    )
    record = record_final_evaluation(round_dir, supplied, repo=clean_repo)

    assert record["result_manifest"] == {"metadata": {"version": 1}}
    assert record["result_manifest_sha256"] == canonical_sha256(
        record["result_manifest"]
    )


def test_evaluation_accepts_only_hash_declared_round_result_files(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    result_path = round_dir / "results" / "scores.json"
    result_path.parent.mkdir()
    result_path.write_text('{"mse": 0.5}\n', encoding="utf-8")
    manifest = {
        "result_files": [
            {
                "path": "results/scores.json",
                "sha256": file_sha256(result_path),
            }
        ]
    }

    receipt = record_final_evaluation(round_dir, manifest, repo=clean_repo)

    assert receipt["state"] == "evaluated"
    assert receipt["result_manifest"] == manifest


def test_evaluation_rejects_undeclared_ignored_result_file(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    result_path = round_dir / "results" / "scores.json"
    result_path.parent.mkdir()
    result_path.write_text('{"mse": 0.5}\n', encoding="utf-8")
    (round_dir / "results" / "payload.pyc").write_bytes(b"undeclared bytes")
    manifest = {
        "result_files": [
            {
                "path": "results/scores.json",
                "sha256": file_sha256(result_path),
            }
        ]
    }

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        record_final_evaluation(round_dir, manifest, repo=clean_repo)


def test_evaluation_rejects_declared_result_hash_mismatch(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    result_path = round_dir / "results" / "scores.json"
    result_path.parent.mkdir()
    result_path.write_text('{"mse": 0.5}\n', encoding="utf-8")

    with pytest.raises(StudyStateError, match="result file hash"):
        record_final_evaluation(
            round_dir,
            {"result_files": [{"path": "results/scores.json", "sha256": "0" * 64}]},
            repo=clean_repo,
        )


def test_evaluation_fsyncs_declared_result_and_directory(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    result_path = round_dir / "results" / "scores.json"
    result_path.parent.mkdir()
    result_path.write_text('{"mse": 0.5}\n', encoding="utf-8")
    observed: set[str] = set()
    real_fsync = os.fsync

    def observe_fsync(descriptor: int) -> None:
        try:
            observed.add(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            pass
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    record_final_evaluation(
        round_dir,
        {
            "result_files": [
                {
                    "path": "results/scores.json",
                    "sha256": file_sha256(result_path),
                }
            ]
        },
        repo=clean_repo,
    )

    assert str(result_path) in observed
    assert str(result_path.parent) in observed


def test_missing_result_after_interrupted_receipt_is_superseded(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    result_path = round_dir / "results" / "scores.json"
    result_path.parent.mkdir()
    result_path.write_text('{"mse": 0.5}\n', encoding="utf-8")
    manifest = {
        "result_files": [
            {"path": "results/scores.json", "sha256": file_sha256(result_path)}
        ]
    }
    registry_path = study_module._registry_path(clean_repo, "round-001")
    original = study_module._atomic_write_json

    def fail_evaluated_registry(path, payload, **kwargs):
        if path == registry_path and payload.get("state") == "evaluated":
            raise OSError("injected evaluated-registry crash")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(study_module, "_atomic_write_json", fail_evaluated_registry)
    with pytest.raises(OSError, match="injected evaluated-registry crash"):
        record_final_evaluation(round_dir, manifest, repo=clean_repo)
    assert (round_dir / "evaluation_receipt.json").exists()
    assert study_module._read_record(registry_path)["state"] == "running"

    monkeypatch.setattr(study_module, "_atomic_write_json", original)
    result_path.unlink()
    with pytest.raises(StudyStateError):
        record_final_evaluation(round_dir, {"summary": "incomplete"}, repo=clean_repo)

    assert study_module._read_record(registry_path)["state"] == "superseded"


def test_result_change_during_registry_advance_supersedes_round(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    result_path = round_dir / "results" / "scores.json"
    result_path.parent.mkdir()
    result_path.write_text('{"mse": 0.5}\n', encoding="utf-8")
    manifest = {
        "result_files": [
            {"path": "results/scores.json", "sha256": file_sha256(result_path)}
        ]
    }
    original = study_module._advance_registry

    def mutate_before_advance(*args, **kwargs):
        if kwargs.get("new_state") == "evaluated":
            result_path.write_text('{"mse": 999}\n', encoding="utf-8")
        return original(*args, **kwargs)

    monkeypatch.setattr(study_module, "_advance_registry", mutate_before_advance)
    with pytest.raises(StudyStateError, match="result file hash"):
        record_final_evaluation(round_dir, manifest, repo=clean_repo)

    registry = study_module._read_record(
        study_module._registry_path(clean_repo, "round-001")
    )
    assert registry["state"] == "superseded"


def test_evaluation_requires_claim_and_unchanged_repository(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    with pytest.raises(StudyStateError, match="claimed"):
        record_final_evaluation(
            round_dir, {"results_sha256": "a" * 64}, repo=clean_repo
        )

    assert_final_runnable(clean_repo, round_dir)
    _git(clean_repo, "update-index", "--assume-unchanged", "tracked.py")
    (clean_repo / "tracked.py").write_text("changed after claim\n", encoding="utf-8")
    with pytest.raises(StudyStateError, match="clean frozen commit"):
        record_final_evaluation(
            round_dir, {"results_sha256": "a" * 64}, repo=clean_repo
        )


def test_round_identity_and_manifest_are_revalidated_before_recording(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    manifest_path = round_dir / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["round_id"] = "copied-round"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StudyStateError, match="seed manifest"):
        record_final_evaluation(
            round_dir, {"results_sha256": "a" * 64}, repo=clean_repo
        )


def test_nonfinite_result_manifest_is_a_state_error(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    with pytest.raises(StudyStateError, match="valid JSON"):
        record_final_evaluation(round_dir, {"metric": float("nan")}, repo=clean_repo)


def test_supersede_preserves_prior_records_and_requires_reason(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    before = {path.name: path.read_bytes() for path in round_dir.iterdir()}

    with pytest.raises(StudyStateError, match="nonempty reason"):
        supersede_round(round_dir, "  ")

    record = supersede_round(round_dir, "new development round")

    assert record["state"] == "superseded"
    assert record["previous_state"] == "materialized"
    assert record["reason"] == "new development round"
    assert all(
        (round_dir / name).read_bytes() == contents for name, contents in before.items()
    )
    with pytest.raises(StudyStateError, match="superseded"):
        assert_final_runnable(clean_repo, round_dir)


def test_cli_lifecycle_and_json_errors(clean_repo: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/studyctl.py"
    round_dir = clean_repo / "artifacts/study/round-cli"

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args],
            cwd=clean_repo,
            capture_output=True,
            text=True,
        )

    frozen = run(
        "freeze",
        str(round_dir),
        "config.json",
        "protocol.json",
        "--environment",
        "environment.lock",
        "--repo",
        str(clean_repo),
    )
    assert frozen.returncode == 0, frozen.stderr
    assert json.loads(frozen.stdout)["state"] == "frozen"

    materialized = run(
        "materialize-final",
        str(round_dir),
        "--seed-count",
        "4",
        "--repo",
        str(clean_repo),
    )
    assert materialized.returncode == 0, materialized.stderr
    claimed = run("verify-final", str(round_dir), "--repo", str(clean_repo))
    assert claimed.returncode == 0, claimed.stderr
    assert json.loads(claimed.stdout)["state"] == "running"

    duplicate = run("verify-final", str(round_dir), "--repo", str(clean_repo))
    assert duplicate.returncode == 2
    assert "error" in json.loads(duplicate.stderr)


def test_rounds_must_use_one_canonical_repository_root(clean_repo: Path) -> None:
    with pytest.raises(StudyStateError, match="canonical rounds root"):
        freeze_round(
            clean_repo,
            clean_repo / "artifacts/alternate/round-001",
            clean_repo / "config.json",
            clean_repo / "protocol.json",
            environment_path=clean_repo / "environment.lock",
        )


def test_registry_prevents_reusing_a_frozen_snapshot_for_a_second_holdout(
    clean_repo: Path, tmp_path: Path
) -> None:
    round_dir = freeze_fixture(clean_repo)
    frozen_snapshot = tmp_path / "frozen-snapshot"
    shutil.copytree(round_dir, frozen_snapshot)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)

    shutil.rmtree(round_dir)
    shutil.copytree(frozen_snapshot, round_dir)
    with pytest.raises(StudyStateError, match="registry|missing round record"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_copied_round_records_fail_path_identity(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    copied = clean_repo / "artifacts/study/round-002"
    shutil.copytree(round_dir, copied)

    with pytest.raises(StudyStateError, match="round identity"):
        assert_final_runnable(clean_repo, copied)


def test_dirty_submodule_cannot_be_hidden_by_ignore_configuration(
    clean_repo: Path, tmp_path: Path
) -> None:
    child = tmp_path / "competitor-source"
    child.mkdir()
    _git(child, "init")
    _git(child, "config", "user.name", "Study Test")
    _git(child, "config", "user.email", "study@example.invalid")
    (child / "method.py").write_text("original\n", encoding="utf-8")
    _git(child, "add", ".")
    _git(child, "commit", "-m", "competitor source")

    _git(
        clean_repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "competitor",
    )
    _git(clean_repo, "commit", "-am", "add competitor")
    _git(clean_repo, "config", "submodule.competitor.ignore", "all")
    (clean_repo / "competitor/method.py").write_text(
        "hidden change\n", encoding="utf-8"
    )
    assert _git(clean_repo, "status", "--porcelain") == ""

    with pytest.raises(StudyStateError, match="clean"):
        freeze_fixture(clean_repo)


@pytest.mark.parametrize(
    ("mutate_claim", "mutate_materialization", "message"),
    [
        (
            lambda record: record.__setitem__("claim_id", None),
            lambda record: record.__setitem__("materialization_claim_id", None),
            "claim",
        ),
        (
            lambda record: None,
            lambda record: record.__setitem__("seed_manifest_path", "other.json"),
            "manifest path",
        ),
        (
            lambda record: None,
            lambda record: record.__setitem__("generator_seeds", [7, 8, 9]),
            "generator seeds",
        ),
    ],
)
def test_materialization_identity_fields_are_strictly_validated(
    clean_repo: Path, mutate_claim, mutate_materialization, message: str
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    claim_path = round_dir / "materialization_claim.json"
    materialization_path = round_dir / "materialization.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    mutate_claim(claim)
    mutate_materialization(materialization)
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")

    with pytest.raises(StudyStateError, match=f"{message}|registry history"):
        assert_final_runnable(clean_repo, round_dir)


def test_boolean_schema_version_is_rejected(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    freeze_path = round_dir / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["schema_version"] = True
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    with pytest.raises(StudyStateError, match="schema_version"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_concurrent_materializers_have_exactly_one_success(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)

    def attempt() -> str:
        try:
            materialize_final(round_dir, seed_count=4, repo=clean_repo)
            return "success"
        except StudyStateError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: attempt(), range(8)))
    assert outcomes.count("success") == 1
    assert outcomes.count("blocked") == 7


def test_supersede_waits_for_inflight_execution_claim(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    entered = threading.Event()
    release = threading.Event()
    superseded = threading.Event()
    original = study_module._validate_seed_manifest

    def paused_validate(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(study_module, "_validate_seed_manifest", paused_validate)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claim_future = pool.submit(assert_final_runnable, clean_repo, round_dir)
        assert entered.wait(timeout=5)

        def do_supersede():
            result = supersede_round(round_dir, "concurrent stop")
            superseded.set()
            return result

        supersede_future = pool.submit(do_supersede)
        assert not superseded.wait(timeout=0.2)
        release.set()
        assert claim_future.result(timeout=5)["state"] == "running"
        assert supersede_future.result(timeout=5)["previous_state"] == "running"


def test_record_rechecks_repository_immediately_before_receipt(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    first_check = threading.Event()
    continue_record = threading.Event()
    calls = 0
    original = study_module._verify_frozen_repository

    def controlled_verify(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            first_check.set()
            assert continue_record.wait(timeout=5)
        return result

    monkeypatch.setattr(study_module, "_verify_frozen_repository", controlled_verify)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            record_final_evaluation,
            round_dir,
            {"results_sha256": "a" * 64},
            repo=clean_repo,
        )
        assert first_check.wait(timeout=5)
        (clean_repo / "tracked.py").write_text(
            "changed during record\n", encoding="utf-8"
        )
        continue_record.set()
        with pytest.raises(StudyStateError, match="clean frozen commit"):
            future.result(timeout=5)
    assert calls == 2
    assert not (round_dir / "evaluation_receipt.json").exists()


def test_registry_hash_binds_original_freeze_record(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    (clean_repo / "tracked.py").write_text("method B\n", encoding="utf-8")
    _git(clean_repo, "add", "tracked.py")
    _git(clean_repo, "commit", "-m", "method B")
    freeze_path = round_dir / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["method_commit"] = _git(clean_repo, "rev-parse", "HEAD")
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")

    with pytest.raises(StudyStateError, match="registry history hash"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_registry_hash_binds_original_unseen_seed_manifest(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    manifest_path = round_dir / "final_manifest.json"
    materialization_path = round_dir / "materialization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator_seeds"][0] = 1
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    materialization["generator_seeds"] = manifest["generator_seeds"]
    materialization["seed_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")

    with pytest.raises(StudyStateError, match="registry history hash"):
        assert_final_runnable(clean_repo, round_dir)


def test_duplicated_seed_field_uses_type_strict_equality(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    freeze = json.loads((round_dir / "freeze.json").read_text(encoding="utf-8"))
    manifest_path = round_dir / "final_manifest.json"
    materialization_path = round_dir / "materialization.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator_seeds"][0] = 1
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    materialization["generator_seeds"] = manifest["generator_seeds"].copy()
    materialization["generator_seeds"][0] = True
    materialization["seed_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")

    with pytest.raises(StudyStateError, match="generator seeds"):
        study_module._validate_seed_manifest(round_dir, freeze)


def test_git_replacement_refs_cannot_substitute_frozen_code(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    frozen_commit = _git(clean_repo, "rev-parse", "HEAD")
    (clean_repo / "tracked.py").write_text("replacement method\n", encoding="utf-8")
    _git(clean_repo, "add", "tracked.py")
    _git(clean_repo, "commit", "-m", "replacement method")
    replacement_commit = _git(clean_repo, "rev-parse", "HEAD")
    _git(clean_repo, "replace", frozen_commit, replacement_commit)
    _git(clean_repo, "reset", "--hard", frozen_commit)
    assert _git(clean_repo, "rev-parse", "HEAD") == frozen_commit
    assert _git(clean_repo, "status", "--porcelain") == ""

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_clean_filter_cannot_hide_changed_tracked_bytes(clean_repo: Path) -> None:
    _git(
        clean_repo,
        "config",
        "filter.hide.clean",
        "sed 's/version = 2/version = 1/'",
    )
    _git(clean_repo, "config", "filter.hide.smudge", "cat")
    (clean_repo / ".gitattributes").write_text(
        "tracked.py filter=hide\n", encoding="utf-8"
    )
    (clean_repo / "tracked.py").write_text("version = 1\n", encoding="utf-8")
    _git(clean_repo, "add", ".gitattributes", "tracked.py")
    _git(clean_repo, "commit", "-m", "configure clean filter")
    round_dir = freeze_fixture(clean_repo)
    (clean_repo / "tracked.py").write_text("version = 2\n", encoding="utf-8")
    assert _git(clean_repo, "status", "--porcelain") == ""

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)


def test_complete_copied_state_cannot_run_in_a_fresh_clone(
    clean_repo: Path, tmp_path: Path
) -> None:
    round_dir = freeze_fixture(clean_repo)
    clone = tmp_path / "same-name" / "repo"
    clone.parent.mkdir()
    _git(tmp_path, "clone", "--no-local", str(clean_repo), str(clone))
    copied_round = clone / "artifacts/study/round-001"
    copied_round.parent.mkdir(parents=True)
    shutil.copytree(round_dir, copied_round)
    source_common = Path(_git(clean_repo, "rev-parse", "--git-common-dir"))
    if not source_common.is_absolute():
        source_common = clean_repo / source_common
    clone_common = Path(_git(clone, "rev-parse", "--git-common-dir"))
    if not clone_common.is_absolute():
        clone_common = clone / clone_common
    shutil.copytree(
        source_common / "maskimpute-study",
        clone_common / "maskimpute-study",
        dirs_exist_ok=True,
    )

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(copied_round, seed_count=4, repo=clone)


def test_same_path_replacement_clone_cannot_replay_frozen_round(
    clean_repo: Path, tmp_path: Path
) -> None:
    round_dir = freeze_fixture(clean_repo)
    frozen_round = tmp_path / "frozen-round"
    shutil.copytree(round_dir, frozen_round)
    common = Path(_git(clean_repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = clean_repo / common
    frozen_state = tmp_path / "frozen-study-state"
    shutil.copytree(common / "maskimpute-study", frozen_state)

    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    archived = tmp_path / "archived-repository"
    clean_repo.rename(archived)
    _git(tmp_path, "clone", "--no-local", str(archived), str(clean_repo))
    replayed_round = clean_repo / "artifacts/study/round-001"
    replayed_round.parent.mkdir(parents=True)
    shutil.copytree(frozen_round, replayed_round)
    replacement_common = Path(_git(clean_repo, "rev-parse", "--git-common-dir"))
    if not replacement_common.is_absolute():
        replacement_common = clean_repo / replacement_common
    shutil.copytree(
        frozen_state,
        replacement_common / "maskimpute-study",
        dirs_exist_ok=True,
    )

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(replayed_round, seed_count=4, repo=clean_repo)


def test_registry_history_is_digest_chained(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    registry_path = study_module._registry_path(clean_repo, "round-001")
    frozen = study_module._read_record(registry_path)
    first = frozen["history"][0]
    assert first["previous_entry_sha256"] is None
    assert frozen["history_head_sha256"] == first["entry_sha256"]

    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    materialized = study_module._read_record(registry_path)
    second = materialized["history"][1]
    assert second["previous_entry_sha256"] == first["entry_sha256"]
    assert materialized["history_head_sha256"] == second["entry_sha256"]

    second["previous_entry_sha256"] = "0" * 64
    registry_path.write_text(json.dumps(materialized), encoding="utf-8")
    freeze = study_module._read_record(round_dir / "freeze.json")
    with pytest.raises(StudyStateError, match="history"):
        study_module._validate_registry(clean_repo, round_dir, freeze)


def test_registry_history_must_start_at_freeze_and_follow_lifecycle(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    registry_path = study_module._registry_path(clean_repo, "round-001")
    registry = study_module._read_record(registry_path)
    first = registry["history"][0]

    duplicate = study_module._registry_entry(
        state="frozen",
        record_name="freeze.json",
        record=study_module._read_record(round_dir / "freeze.json"),
        previous_entry_sha256=first["entry_sha256"],
    )
    registry["history"].append(duplicate)
    registry["history_head_sha256"] = duplicate["entry_sha256"]
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    freeze = study_module._read_record(round_dir / "freeze.json")

    with pytest.raises(StudyStateError, match="history|transition"):
        study_module._validate_registry(clean_repo, round_dir, freeze)


def test_registry_history_cannot_drop_frozen_anchor(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    registry_path = study_module._registry_path(clean_repo, "round-001")
    registry = study_module._read_record(registry_path)
    surviving = dict(registry["history"][1])
    surviving["previous_entry_sha256"] = None
    payload = dict(surviving)
    payload.pop("entry_sha256")
    surviving["entry_sha256"] = canonical_sha256(payload)
    registry["history"] = [surviving]
    registry["history_head_sha256"] = surviving["entry_sha256"]
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    freeze = study_module._read_record(round_dir / "freeze.json")

    with pytest.raises(StudyStateError, match="history|frozen"):
        study_module._validate_registry(clean_repo, round_dir, freeze)


@pytest.mark.parametrize("exclusive", [False, True])
def test_atomic_json_publication_fsyncs_containing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exclusive: bool
) -> None:
    observed: list[bool] = []
    real_fsync = os.fsync

    def observe_fsync(descriptor: int) -> None:
        observed.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    study_module._atomic_write_json(
        tmp_path / f"record-{exclusive}.json",
        {"state": "test"},
        exclusive=exclusive,
    )

    assert True in observed


def test_freeze_durably_publishes_authority_root(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = Path(_git(clean_repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = clean_repo / common
    common = common.resolve()
    observed: set[str] = set()
    real_fsync = os.fsync

    def observe_fsync(descriptor: int) -> None:
        try:
            observed.add(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            pass
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    freeze_fixture(clean_repo)

    assert str(common) in observed


def test_post_publication_repository_change_supersedes_round(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    original = study_module._atomic_write_json

    def change_after_materialization(path, payload, **kwargs):
        original(path, payload, **kwargs)
        if path.name == "materialization.json":
            (clean_repo / "tracked.py").write_text(
                "changed after publication\n", encoding="utf-8"
            )

    monkeypatch.setattr(
        study_module, "_atomic_write_json", change_after_materialization
    )
    with pytest.raises(StudyStateError, match="clean frozen commit"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)

    supersession = json.loads(
        (round_dir / "supersession.json").read_text(encoding="utf-8")
    )
    assert supersession["previous_state"] == "frozen"
    registry = study_module._read_record(
        study_module._registry_path(clean_repo, "round-001")
    )
    assert registry["state"] == "superseded"


@pytest.mark.parametrize("action", ["mutate", "delete"])
def test_post_publication_seed_manifest_change_supersedes_round(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    round_dir = freeze_fixture(clean_repo)
    original = study_module._atomic_write_json

    def change_manifest_after_materialization(path, payload, **kwargs):
        original(path, payload, **kwargs)
        if path.name == "materialization.json":
            manifest_path = round_dir / "final_manifest.json"
            if action == "delete":
                manifest_path.unlink()
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["generator_seeds"][0] += 1
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        study_module, "_atomic_write_json", change_manifest_after_materialization
    )
    with pytest.raises(StudyStateError, match="manifest"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)

    assert (
        study_module._read_record(study_module._registry_path(clean_repo, "round-001"))[
            "state"
        ]
        == "superseded"
    )


def test_post_publication_execution_claim_change_supersedes_round(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    original = study_module._atomic_write_json

    def change_claim_after_publication(path, payload, **kwargs):
        original(path, payload, **kwargs)
        if path.name == "execution_claim.json":
            claim = json.loads(path.read_text(encoding="utf-8"))
            claim["execution_claim_id"] = "0" * 32
            path.write_text(json.dumps(claim), encoding="utf-8")

    monkeypatch.setattr(
        study_module, "_atomic_write_json", change_claim_after_publication
    )
    with pytest.raises(StudyStateError, match="claim|registry"):
        assert_final_runnable(clean_repo, round_dir)

    assert (
        study_module._read_record(study_module._registry_path(clean_repo, "round-001"))[
            "state"
        ]
        == "superseded"
    )


def test_seed_manifest_change_during_registry_advance_supersedes_round(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    original = study_module._advance_registry

    def mutate_before_advance(*args, **kwargs):
        if kwargs.get("new_state") == "materialized":
            manifest_path = round_dir / "final_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["generator_seeds"][0] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return original(*args, **kwargs)

    monkeypatch.setattr(study_module, "_advance_registry", mutate_before_advance)
    with pytest.raises(StudyStateError, match="manifest"):
        materialize_final(round_dir, seed_count=4, repo=clean_repo)

    assert (
        study_module._read_record(study_module._registry_path(clean_repo, "round-001"))[
            "state"
        ]
        == "superseded"
    )


def test_receipt_change_after_registry_publication_fails_closed(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    original = study_module._advance_registry

    def mutate_after_advance(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("new_state") == "evaluated":
            receipt_path = round_dir / "evaluation_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["evaluated_at"] = "changed"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return result

    monkeypatch.setattr(study_module, "_advance_registry", mutate_after_advance)
    with pytest.raises(StudyStateError, match="receipt|registry"):
        record_final_evaluation(
            round_dir, {"results_sha256": "a" * 64}, repo=clean_repo
        )

    registry_path = study_module._registry_path(clean_repo, "round-001")
    assert study_module._read_record(registry_path)["state"] == "evaluated"
    freeze = study_module._read_record(round_dir / "freeze.json")
    with pytest.raises(StudyStateError, match="registry history hash"):
        study_module._validate_registry(clean_repo, round_dir, freeze)


def test_post_receipt_repository_change_marks_receipt_superseded(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    original = study_module._atomic_write_json

    def change_after_receipt(path, payload, **kwargs):
        original(path, payload, **kwargs)
        if path.name == "evaluation_receipt.json":
            (clean_repo / "tracked.py").write_text(
                "changed after receipt\n", encoding="utf-8"
            )

    monkeypatch.setattr(study_module, "_atomic_write_json", change_after_receipt)
    with pytest.raises(StudyStateError, match="clean frozen commit"):
        record_final_evaluation(
            round_dir, {"results_sha256": "a" * 64}, repo=clean_repo
        )

    assert (round_dir / "evaluation_receipt.json").exists()
    supersession = json.loads(
        (round_dir / "supersession.json").read_text(encoding="utf-8")
    )
    assert supersession["previous_state"] == "running"
    assert (
        study_module._read_record(study_module._registry_path(clean_repo, "round-001"))[
            "state"
        ]
        == "superseded"
    )


@pytest.mark.parametrize("action", ["mutate", "delete"])
def test_post_receipt_seed_manifest_change_is_superseded(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    assert_final_runnable(clean_repo, round_dir)
    original = study_module._atomic_write_json

    def change_manifest_after_receipt(path, payload, **kwargs):
        original(path, payload, **kwargs)
        if path.name == "evaluation_receipt.json":
            manifest_path = round_dir / "final_manifest.json"
            if action == "delete":
                manifest_path.unlink()
            else:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["generator_seeds"][0] += 1
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        study_module, "_atomic_write_json", change_manifest_after_receipt
    )
    with pytest.raises(StudyStateError, match="manifest"):
        record_final_evaluation(
            round_dir, {"results_sha256": "a" * 64}, repo=clean_repo
        )

    assert (
        study_module._read_record(study_module._registry_path(clean_repo, "round-001"))[
            "state"
        ]
        == "superseded"
    )


def test_replaced_lock_path_cannot_publish_after_supersession(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    entered = threading.Event()
    release = threading.Event()
    original_validate = study_module._validate_seed_manifest

    def paused_validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=5)
        return result

    monkeypatch.setattr(study_module, "_validate_seed_manifest", paused_validate)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(assert_final_runnable, clean_repo, round_dir)
        assert entered.wait(timeout=5)
        state_root = study_module._study_state_root(clean_repo)
        (state_root / ".locks").rename(state_root / ".locks-old")
        supersede_round(round_dir, "replacement lock test")
        release.set()
        with pytest.raises(StudyStateError, match="lock identity|registry|superseded"):
            future.result(timeout=5)
    assert not (round_dir / "execution_claim.json").exists()


def test_supersession_registry_write_is_reconciled_on_retry(
    clean_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    registry_path = study_module._registry_path(clean_repo, "round-001")
    original = study_module._atomic_write_json
    failed = False

    def fail_once(path, payload, **kwargs):
        nonlocal failed
        if (
            path == registry_path
            and payload.get("state") == "superseded"
            and not failed
        ):
            failed = True
            raise OSError("injected registry publication failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(study_module, "_atomic_write_json", fail_once)
    with pytest.raises(OSError, match="injected registry") as captured:
        supersede_round(round_dir, "injected crash")
    assert (round_dir / "supersession.json").exists()
    assert "lock" not in str(captured.value)

    monkeypatch.setattr(study_module, "_atomic_write_json", original)
    record = supersede_round(round_dir, "injected crash")
    assert record["state"] == "superseded"
    assert study_module._read_record(registry_path)["state"] == "superseded"
