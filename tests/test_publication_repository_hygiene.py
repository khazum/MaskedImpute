from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

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


ACTIVE_PYTHON_CLIS = tuple(
    sorted((ROOT / "scripts").glob("*.py"))
    + sorted((ROOT / "scripts/simulators").glob("*.py"))
)


@pytest.mark.parametrize(
    "script",
    ACTIVE_PYTHON_CLIS,
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_active_python_cli_help_exits_successfully_without_running_work(
    script: Path,
) -> None:
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def _load_python_script(relative_path: str, module_name: str):
    specification = importlib.util.spec_from_file_location(
        module_name, ROOT / relative_path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_finalization_cli_returns_structured_nonzero_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from maskimpute_benchmark.selection import SelectionAuthorityError

    script = _load_python_script(
        "scripts/finalize_development_authority.py",
        "task7_finalize_development_authority",
    )

    def fail_finalization() -> None:
        raise SelectionAuthorityError("development authority is not ready")

    monkeypatch.setattr(
        script,
        "finalize_development_artifact_bindings",
        fail_finalization,
    )

    assert script.main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"error": "development authority is not ready"}


def test_environment_builder_refuses_to_replace_existing_receipt(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    source = tmp_path / "source"
    source.mkdir()
    receipt = tmp_path / "build-receipt.json"
    receipt.write_text("existing receipt\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build_saver_r_environment.sh"),
            str(library),
            str(source),
            str(receipt),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 73
    assert "refusing to replace existing build receipt" in completed.stderr
    assert receipt.read_text(encoding="utf-8") == "existing receipt\n"


def test_environment_builder_existing_receipt_precedes_source_normalization(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    missing_source = tmp_path / "missing-parent" / "source"
    receipt = tmp_path / "build-receipt.json"
    receipt.write_text("existing receipt\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build_saver_r_environment.sh"),
            str(library),
            str(missing_source),
            str(receipt),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 73
    assert "refusing to replace existing build receipt" in completed.stderr
    assert receipt.read_text(encoding="utf-8") == "existing receipt\n"
    assert not library.exists()


def test_environment_builder_rejects_receipt_equal_to_library_before_writing(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    missing_source = tmp_path / "missing-parent" / "source"

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build_saver_r_environment.sh"),
            str(library),
            str(missing_source),
            str(library),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 64
    assert "library and build receipt paths must be disjoint" in completed.stderr
    assert not library.exists()


def test_environment_builder_rejects_receipt_nested_in_library_before_writing(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    missing_source = tmp_path / "missing-parent" / "source"
    receipt = library / "nested" / ".." / "build-receipt.json"

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build_saver_r_environment.sh"),
            str(library),
            str(missing_source),
            str(receipt),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 64
    assert "library and build receipt paths must be disjoint" in completed.stderr
    assert not library.exists()


def test_environment_builder_rejects_library_nested_under_receipt_path(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt-path"
    library = receipt / "library"
    missing_source = tmp_path / "missing-parent" / "source"

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build_saver_r_environment.sh"),
            str(library),
            str(missing_source),
            str(receipt),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 64
    assert "library and build receipt paths must be disjoint" in completed.stderr
    assert not receipt.exists()


def test_environment_builder_allows_canonical_sibling_receipt(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    missing_source = tmp_path / "missing-parent" / "source"
    receipt = tmp_path / "library.build-receipt.json"

    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/build_saver_r_environment.sh"),
            str(library),
            str(missing_source),
            str(receipt),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "library and build receipt paths must be disjoint" not in completed.stderr
    assert not library.exists()
    assert not receipt.exists()


def test_conditional_revision_commands_render_inside_ordered_step() -> None:
    lines = (
        (ROOT / "docs/development-selection-workflow.md")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    start = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("13. Complete a revision stage")
    )
    end = next(
        index
        for index, line in enumerate(lines[start + 1 :], start=start + 1)
        if line.startswith("14. Complete the fixed external-reference")
    )
    continuation = lines[start + 1 : end]

    assert all(not line or line.startswith("    ") for line in continuation)
    nested = "\n".join(
        line[4:] if line.startswith("    ") else line for line in continuation
    )
    fences = tuple(line for line in nested.splitlines() if line.startswith("```"))
    assert fences == ("```text", "```", "```text", "```")
    assert nested.index("run_v28_revision_competition.py") < nested.index(
        "run_v29_revision_competition.py"
    )
