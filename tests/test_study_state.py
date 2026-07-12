import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import shutil
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


def test_freeze_rejects_dirty_repository(clean_repo: Path) -> None:
    (clean_repo / "tracked.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(StudyStateError, match="clean"):
        freeze_fixture(clean_repo)


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
    assert record["environment_sha256"] == file_sha256(
        clean_repo / "environment.lock"
    )
    assert record["round_path"] == "artifacts/study/round-001"
    assert len(record["round_token"]) == 32
    assert len(record["repository_instance_id"]) == 32
    assert len(record["worktree_path_sha256"]) == 64
    frozen_at = datetime.fromisoformat(record["frozen_at"].replace("Z", "+00:00"))
    assert frozen_at.tzinfo == timezone.utc


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
    record_final_evaluation(
        round_dir, {"results_sha256": "a" * 64}, repo=clean_repo
    )
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
    assert record["environment_sha256"] == file_sha256(
        clean_repo / "environment.lock"
    )
    assert record == json.loads(
        (round_dir / "evaluation_receipt.json").read_text(encoding="utf-8")
    )


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


def test_supersede_preserves_prior_records_and_requires_reason(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4, repo=clean_repo)
    before = {path.name: path.read_bytes() for path in round_dir.iterdir()}

    with pytest.raises(StudyStateError, match="nonempty reason"):
        supersede_round(round_dir, "  ")

    record = supersede_round(round_dir, "new development round")

    assert record["state"] == "superseded"
    assert record["previous_state"] == "materialized"
    assert record["reason"] == "new development round"
    assert all((round_dir / name).read_bytes() == contents for name, contents in before.items())
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
    (clean_repo / "competitor/method.py").write_text("hidden change\n", encoding="utf-8")
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
        (clean_repo / "tracked.py").write_text("changed during record\n", encoding="utf-8")
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

    monkeypatch.setattr(study_module, "_atomic_write_json", change_after_materialization)
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
    assert study_module._read_record(
        study_module._registry_path(clean_repo, "round-001")
    )["state"] == "superseded"


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
        if path == registry_path and payload.get("state") == "superseded" and not failed:
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
