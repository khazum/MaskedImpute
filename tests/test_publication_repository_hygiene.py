from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

_GENERATED_MATRIX_SUFFIXES = {
    ".h5",
    ".h5ad",
    ".mtx",
    ".npy",
    ".npz",
}
_DIRECT_CHECKPOINT_INTENT_SUFFIX = ".transaction.json"


def _tracked_or_unignored_paths() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return tuple(Path(name) for name in result.stdout.decode().split("\0") if name)


def _is_closed_direct_checkpoint_intent(path: Path) -> bool:
    name = path.name
    if not name.startswith(".") or not name.endswith(_DIRECT_CHECKPOINT_INTENT_SUFFIX):
        return False
    checkpoint_name = name[1 : -len(_DIRECT_CHECKPOINT_INTENT_SUFFIX)]
    return bool(checkpoint_name)


def _is_forbidden_tracked_or_unignored_path(path: Path) -> bool:
    if (
        {"__pycache__", ".pytest_cache", ".ruff_cache"} & set(path.parts)
        or path.suffix in {".pyc", ".pyo", ".tmp", ".partial"}
        or _is_closed_direct_checkpoint_intent(path)
    ):
        return True
    if path.parts and path.parts[0] in {"feedback", "historical"}:
        return False
    return bool(
        path.name == "checkpoint.json"
        or path.suffix.lower() in _GENERATED_MATRIX_SUFFIXES
        or path.parts[:2] in {("paper", "generated"), ("paper", "figures")}
    )


def _tracked_index_entries() -> tuple[tuple[str, str], ...]:
    result = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    entries: list[tuple[str, str]] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, relative = raw.decode("utf-8").split("\t", maxsplit=1)
        mode = metadata.split(" ", maxsplit=1)[0]
        entries.append((mode, relative))
    return tuple(entries)


def test_publication_tree_has_no_gitlinks_or_tracked_virtual_environment() -> None:
    entries = _tracked_index_entries()
    gitlinks = sorted(relative for mode, relative in entries if mode == "160000")
    virtual_environment = sorted(
        relative
        for _, relative in entries
        if relative == ".venv_scvi" or relative.startswith(".venv_scvi/")
    )

    assert not gitlinks, "publication tree retains gitlinks: " + ", ".join(gitlinks)
    assert not virtual_environment, (
        "publication tree retains a machine-specific virtual environment"
    )


def test_machine_specific_scvi_environment_is_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".venv_scvi/pyvenv.cfg"],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_hygiene_path_classification_is_narrow_and_cache_strict() -> None:
    assert not _is_forbidden_tracked_or_unignored_path(Path("docs/method-overview.svg"))
    assert not _is_forbidden_tracked_or_unignored_path(
        Path("feedback/review-report.pdf")
    )
    assert not _is_forbidden_tracked_or_unignored_path(
        Path("historical/v26/results.npz")
    )
    assert _is_forbidden_tracked_or_unignored_path(
        Path("historical/v26/__pycache__/module.pyc")
    )
    assert _is_forbidden_tracked_or_unignored_path(Path("feedback/.review.tmp"))
    assert _is_forbidden_tracked_or_unignored_path(Path("results/matrix.npz"))


def test_hygiene_classifies_exact_closed_direct_checkpoint_intent_globally() -> None:
    assert _is_forbidden_tracked_or_unignored_path(
        Path("historical/v26/.checkpoint.json.transaction.json")
    )
    assert not _is_forbidden_tracked_or_unignored_path(
        Path("results/checkpoint.json.transaction.json")
    )
    assert not _is_forbidden_tracked_or_unignored_path(
        Path("results/.transaction.json")
    )


def test_repository_has_no_generated_comparator_evidence_or_cache() -> None:
    forbidden_ignored_paths = (
        ROOT / "artifacts/study/development/evaluation/comparator_smoke.json",
        ROOT / "artifacts/study/development/evaluation/comparator_selection.json",
        ROOT / "artifacts/study/development/competition-reconstruction/checkpoint.json",
        ROOT / "artifacts/study/development/competition-reconstruction/"
        ".checkpoint.json.transaction.json",
        ROOT / "artifacts/study/round-001",
        ROOT / "paper/generated",
        ROOT / "paper/figures",
        ROOT / "paper/manuscript.bbl",
        ROOT / "paper/manuscript.pdf",
    )
    assert not any(os.path.lexists(path) for path in forbidden_ignored_paths)

    forbidden = [
        path.as_posix()
        for path in _tracked_or_unignored_paths()
        if _is_forbidden_tracked_or_unignored_path(path)
    ]

    assert not forbidden, "generated or cache paths remain: " + ", ".join(forbidden)


def test_repository_presence_check_includes_ignored_direct_checkpoint_intent(
    monkeypatch,
) -> None:
    checked: list[Path] = []
    actual_lexists = os.path.lexists

    def recording_lexists(path: Path) -> bool:
        checked.append(Path(path))
        return actual_lexists(path)

    monkeypatch.setattr(os.path, "lexists", recording_lexists)
    test_repository_has_no_generated_comparator_evidence_or_cache()

    expected = (
        ROOT / "artifacts/study/development/competition-reconstruction/"
        ".checkpoint.json.transaction.json"
    )
    assert expected in checked
