from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
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


def _latex_command_body(source: str, command: str) -> str:
    marker = f"\\{command}{{"
    start = source.index(marker) + len(marker)
    depth = 1
    cursor = start
    while depth:
        character = source[cursor]
        if character == "{" and source[cursor - 1] != "\\":
            depth += 1
        elif character == "}" and source[cursor - 1] != "\\":
            depth -= 1
        cursor += 1
    return source[start : cursor - 1]


def _latex_prose_word_count(source: str) -> int:
    without_commands = re.sub(r"\\[A-Za-z@]+\*?", "", source)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", without_commands))


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


def test_genome_biology_draft_has_complete_fail_closed_front_matter() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text(encoding="utf-8")

    ordered_front_matter = (
        r"\title[",
        r"\author*",
        r"\affil*",
        r"\email{",
        r"\abstract{",
        r"\keywords{",
        r"\maketitle",
    )
    positions = tuple(manuscript.index(fragment) for fragment in ordered_front_matter)
    assert positions == tuple(sorted(positions))
    for command in (r"\author*", r"\affil*", r"\email{"):
        line = next(line for line in manuscript.splitlines() if command in line)
        assert r"\PendingAuthor" in line

    abstract = _latex_command_body(manuscript, "abstract")
    assert _latex_prose_word_count(abstract) <= 100
    assert r"\cite" not in abstract

    keywords = tuple(
        keyword.strip()
        for keyword in _latex_command_body(manuscript, "keywords").split(",")
    )
    assert 3 <= len(keywords) <= 10
    assert all(keywords)


def test_genome_biology_draft_has_required_section_and_declaration_order() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text(encoding="utf-8")
    sections = (
        r"\section{Background}",
        r"\section{Results}",
        r"\section{Discussion}",
        r"\section{Conclusions}",
        r"\section{Methods}",
        r"\section*{Abbreviations}",
        r"\section*{Declarations}",
    )
    positions = tuple(manuscript.index(section) for section in sections)
    assert positions == tuple(sorted(positions))

    methods = positions[4]
    abbreviations = positions[5]
    disclosure = manuscript.index(
        r"\subsection{Use of generative AI or AI-assisted technologies}"
    )
    assert methods < disclosure < abbreviations
    assert methods < manuscript.index("Artificial intelligence (AI)") < disclosure

    declarations = manuscript[positions[6] :]
    required_headings = (
        "Ethics approval and consent to participate",
        "Consent for publication",
        "Availability of data and materials",
        "Competing interests",
        "Funding",
        "Authors' contributions",
        "Acknowledgements",
    )
    actual_headings = tuple(re.findall(r"\\subsection\*\{([^}]*)\}", declarations))
    assert actual_headings == required_headings


def test_genome_biology_draft_does_not_claim_unexecuted_analyses_are_complete() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text(encoding="utf-8")

    assert "All analyses are generated" not in manuscript
    assert "Their development evidence is reported" not in manuscript


def test_genome_biology_checklists_leave_evidence_dependent_criteria_open() -> None:
    compact = (ROOT / "paper/submission_checklist.md").read_text(encoding="utf-8")
    full = (ROOT / "docs/genome-biology-submission-checklist.md").read_text(
        encoding="utf-8"
    )

    checklist_items = tuple(
        match
        for match in re.finditer(
            r"(?ms)^- \[(?P<status>[ x])\] (?P<body>.*?)(?=^- \[[ x]\] |\Z)",
            compact,
        )
    )
    checked_items = tuple(
        " ".join(match.group("body").split())
        for match in checklist_items
        if match.group("status") == "x"
    )
    assert not any(
        evidence_phrase in item
        for item in checked_items
        for evidence_phrase in ("state-of-the-art", "after final rendering")
    )
    assert (
        "- [ ] Completed same-dataset side-by-side results demonstrate a clear "
        "advance over current state-of-the-art methods"
    ) in " ".join(compact.split())
    assert "**Guidance verified:** 23 July 2026" in full


def test_static_software_archive_inputs_are_conditional_on_author_choice() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text(encoding="utf-8")
    compact = (ROOT / "paper/submission_checklist.md").read_text(encoding="utf-8")
    full = (ROOT / "docs/genome-biology-submission-checklist.md").read_text(
        encoding="utf-8"
    )

    assert (
        r"\PendingAuthor{provide public source and data URLs, accessions, and "
        "reviewer links; if the authors create the recommended static software "
        "archive, provide its DOI or other persistent identifier}"
    ) in manuscript
    assert (
        "- [ ] Public source repository URL, data accessions or reviewer links, "
        "and, if the authors create the recommended static software archive, "
        "its persistent identifier."
    ) in " ".join(compact.split())
    assert (
        "if the authors create a static archived release, the availability "
        "statement gives its persistent identifier"
    ) in " ".join(full.split())
    assert (
        "if a static archived release is created, it identifies that same release"
    ) in " ".join(full.split())

    assert "provide public URLs, accessions, reviewer links, and archive DOI" not in (
        manuscript
    )
    assert "Public repository/release URL, archival DOI" not in compact
    assert "project home page, archived release, supported operating systems" not in (
        full
    )
    assert "Public source, archive, environment locks" not in full


def test_iqr_is_expanded_at_first_rendered_use_and_listed_as_an_abbreviation() -> None:
    manuscript = (ROOT / "paper/manuscript.tex").read_text(encoding="utf-8")

    first_iqr = manuscript.index("IQR")
    expansion = "interquartile range (IQR)"
    assert manuscript[first_iqr - len("interquartile range (") : first_iqr + 4] == (
        expansion
    )

    abbreviations_start = manuscript.index(r"\section*{Abbreviations}")
    declarations_start = manuscript.index(r"\section*{Declarations}")
    abbreviations = manuscript[abbreviations_start:declarations_start]
    assert "IQR, interquartile range" in abbreviations
    assert "median/IQR" not in manuscript


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


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/run_external_reference_development.py",
        "scripts/freeze_publication_round.py",
        "scripts/studyctl.py",
    ),
)
@pytest.mark.parametrize("initial_state", (False, True))
def test_cli_import_preserves_process_bytecode_state(
    relative_path: str,
    initial_state: bool,
) -> None:
    previous_state = sys.dont_write_bytecode
    sys.dont_write_bytecode = initial_state
    try:
        _load_python_script(
            relative_path,
            "task7_import_isolation_"
            + relative_path.replace("/", "_").replace(".", "_")
            + f"_{initial_state}",
        )
        assert sys.dont_write_bytecode is initial_state
    finally:
        sys.dont_write_bytecode = previous_state


def test_external_reference_cli_restores_bytecode_state_after_in_process_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_python_script(
        "scripts/run_external_reference_development.py",
        "task7_external_reference_bytecode_scope",
    )
    observed: list[bool] = []

    def stop_before_work(*_args, **_kwargs):
        observed.append(sys.dont_write_bytecode)
        raise script.ExternalReferenceDevelopmentError("expected test stop")

    monkeypatch.setattr(
        script,
        "run_external_reference_development",
        stop_before_work,
    )
    previous_state = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        status = script.main(
            [
                "--environment",
                "d3impute=/tmp/d3impute",
                "--environment",
                "sctsi=/tmp/sctsi",
                "--sctsi-library",
                "/tmp/sctsi-library",
            ]
        )
        assert observed == [True]
        assert status == 2
        assert sys.dont_write_bytecode is False
    finally:
        sys.dont_write_bytecode = previous_state


def test_publication_freeze_cli_restores_bytecode_state_after_in_process_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_python_script(
        "scripts/freeze_publication_round.py",
        "task7_publication_freeze_bytecode_scope",
    )
    observed: list[bool] = []

    def prepare(_repository: Path) -> dict[str, str]:
        observed.append(sys.dont_write_bytecode)
        return {"state": "prepared"}

    monkeypatch.setattr(script, "prepare_frozen_method", prepare)
    previous_state = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        assert script.main(["prepare"]) == 0
        assert observed == [True]
        assert sys.dont_write_bytecode is False
    finally:
        sys.dont_write_bytecode = previous_state


def test_study_cli_restores_bytecode_state_after_in_process_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_python_script(
        "scripts/studyctl.py",
        "task7_studyctl_bytecode_scope",
    )
    observed: list[bool] = []

    def supersede(_round_dir: Path, _reason: str) -> dict[str, str]:
        observed.append(sys.dont_write_bytecode)
        return {"state": "superseded"}

    monkeypatch.setattr(script, "supersede_round", supersede)
    previous_state = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        assert script.main(["supersede", "/tmp/round", "test reason"]) == 0
        assert observed == [True]
        assert sys.dont_write_bytecode is False
    finally:
        sys.dont_write_bytecode = previous_state


def test_task7_documented_shell_syntax_check_discovers_each_existing_script(
    tmp_path: Path,
) -> None:
    plan = (
        ROOT / "docs/superpowers/plans/"
        "2026-07-23-publication-integration-full-review.md"
    ).read_text(encoding="utf-8")
    task = plan.split(
        "### Task 7: Audit CLIs, study documents, and branch migration integrity",
        maxsplit=1,
    )[1]
    step = task.split("### Task 8:", maxsplit=1)[0]
    command_block = step.split("```bash\n", maxsplit=1)[1].split("\n```", maxsplit=1)[0]
    shell_command = command_block.split("done\n", maxsplit=1)[1].strip()

    scripts = tmp_path / "scripts"
    simulators = scripts / "simulators"
    simulators.mkdir(parents=True)

    empty = subprocess.run(
        ["bash", "-c", shell_command],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert empty.returncode == 0, empty.stderr

    (scripts / "valid.sh").write_text("#!/usr/bin/env bash\ntrue\n", encoding="utf-8")
    valid = subprocess.run(
        ["bash", "-c", shell_command],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr

    (simulators / "invalid.sh").write_text(
        "#!/usr/bin/env bash\nif\n",
        encoding="utf-8",
    )
    invalid = subprocess.run(
        ["bash", "-c", shell_command],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode != 0


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
