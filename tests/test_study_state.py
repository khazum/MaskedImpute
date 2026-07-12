import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

import pytest

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
    protocol = json.loads(Path("study/protocol.json").read_text(encoding="utf-8"))
    (repo / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "freeze inputs")
    return repo


def freeze_fixture(repo: Path) -> Path:
    round_dir = repo / "artifacts/study/round-001"
    freeze_round(repo, round_dir, repo / "config.json", repo / "protocol.json")
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
    )

    assert record == json.loads((round_dir / "freeze.json").read_text(encoding="utf-8"))
    assert record["state"] == "frozen"
    assert record["round_id"] == "round-001"
    assert record["method_commit"] == _git(clean_repo, "rev-parse", "HEAD")
    assert record["config_path"] == "config.json"
    assert record["protocol_path"] == "protocol.json"
    assert record["config_sha256"] == file_sha256(clean_repo / "config.json")
    assert record["protocol_sha256"] == file_sha256(clean_repo / "protocol.json")
    frozen_at = datetime.fromisoformat(record["frozen_at"].replace("Z", "+00:00"))
    assert frozen_at.tzinfo == timezone.utc


def test_materialization_creates_unique_63_bit_generator_seeds(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    record = materialize_final(round_dir, seed_count=32)
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
    materialize_final(round_dir, seed_count=4)
    (clean_repo / "tracked.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)


def test_changed_commit_cannot_run_final_even_when_repository_is_clean(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)
    (clean_repo / "tracked.py").write_text("new commit\n", encoding="utf-8")
    _git(clean_repo, "add", "tracked.py")
    _git(clean_repo, "commit", "-m", "change method")

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)


def test_changed_frozen_input_cannot_run_final_when_git_hides_worktree_change(
    clean_repo: Path,
) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)
    _git(clean_repo, "update-index", "--assume-unchanged", "config.json")
    (clean_repo / "config.json").write_text('{"rank": 99}\n', encoding="utf-8")
    assert _git(clean_repo, "status", "--porcelain") == ""

    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)


def test_clean_materialized_round_is_runnable(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)

    record = assert_final_runnable(clean_repo, round_dir)

    assert record["state"] == "materialized"
    assert record["round_id"] == "round-001"


def test_evaluation_receipt_is_one_use(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)
    record_final_evaluation(round_dir, {"results_sha256": "a" * 64})
    with pytest.raises(StudyStateError, match="already evaluated"):
        assert_final_runnable(clean_repo, round_dir)

    with pytest.raises(StudyStateError, match="already evaluated"):
        record_final_evaluation(round_dir, {"results_sha256": "b" * 64})


def test_evaluation_receipt_records_result_manifest_and_hash(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)
    result_manifest = {"results_sha256": "a" * 64, "rows": 24}

    record = record_final_evaluation(round_dir, result_manifest)

    assert record["state"] == "evaluated"
    assert record["result_manifest"] == result_manifest
    assert record["result_manifest_sha256"] == canonical_sha256(result_manifest)
    assert record == json.loads(
        (round_dir / "evaluation_receipt.json").read_text(encoding="utf-8")
    )


def test_supersede_preserves_prior_records_and_requires_reason(clean_repo: Path) -> None:
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)
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
