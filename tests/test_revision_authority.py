from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SHA_A = "a" * 64
SHA_B = "b" * 64


def _activation(version: str, trigger: str):
    from maskimpute_benchmark.revisions import (
        RevisionActivation,
        revision_stage_paths,
    )

    paths = revision_stage_paths(version)

    return RevisionActivation(
        version=version,
        trigger=trigger,
        selection_input_path=paths.activation_selection_input,
        selection_input_file_sha256=SHA_A,
        selection_result_sha256=SHA_B,
        selection_report_path=paths.activation_selection_report,
        selection_report_file_sha256=SHA_A,
    )


def test_v28_revision_extends_selection_authority_only_after_exact_trigger() -> None:
    from maskimpute_benchmark.revisions import (
        RevisionAuthorityError,
        derive_extended_selection_authority,
        load_revision_spec,
    )
    from maskimpute_benchmark.selection import load_publication_execution_authority

    base = load_publication_execution_authority()
    v28 = load_revision_spec(Path.cwd(), "v28", require_clean=True)

    with pytest.raises(RevisionAuthorityError, match="activation"):
        derive_extended_selection_authority(base, (v28,), ())
    with pytest.raises(ValueError, match="trigger"):
        replace(_activation("v28", "v28"), trigger="freeze_candidate")

    extended = derive_extended_selection_authority(
        base,
        (v28,),
        (_activation("v28", "v28"),),
    )
    assert extended.attempts[:-1] == base.attempts
    assert extended.attempts[-1].configuration_id == "v28-c01-nb-parent-c03"
    assert extended.attempts[-1].version == "v28"
    assert extended.attempts[-1].parent_configuration_id == (
        "v27-c03-calibrated-r1-g1"
    )
    assert extended.method_bindings["v28-c01-nb-parent-c03"] == (
        v28.configuration_sha256
    )
    assert extended.file_sha256["study/v28_revision.json"] == v28.file_sha256
    assert extended.declarations[-1].id == "v28-c01-nb-parent-c03"
    assert extended.comparator_tuning is base.comparator_tuning
    assert extended.comparator_method_bindings is base.comparator_method_bindings


def test_v29_revision_requires_the_combined_v28_report_trigger() -> None:
    from maskimpute_benchmark.revisions import (
        RevisionAuthorityError,
        derive_extended_selection_authority,
        load_revision_spec,
    )
    from maskimpute_benchmark.selection import load_publication_execution_authority

    base = load_publication_execution_authority()
    v28 = load_revision_spec(Path.cwd(), "v28", require_clean=True)
    v29 = load_revision_spec(Path.cwd(), "v29", require_clean=False)

    with pytest.raises(RevisionAuthorityError, match="v29 activation"):
        derive_extended_selection_authority(
            base,
            (v28, v29),
            (_activation("v28", "v28"),),
        )
    with pytest.raises(ValueError, match="trigger"):
        replace(_activation("v29", "v29"), trigger="downgrade_claim")

    extended = derive_extended_selection_authority(
        base,
        (v28, v29),
        (_activation("v28", "v28"), _activation("v29", "v29")),
    )
    assert [attempt.version for attempt in extended.attempts[-2:]] == ["v28", "v29"]
    assert extended.attempts[-1].parent_configuration_id == (
        "v28-c01-nb-parent-c03"
    )
    assert extended.method_bindings[v29.configuration_id] == (
        v29.configuration_sha256
    )


def test_revision_configuration_is_recursively_immutable() -> None:
    from maskimpute_benchmark.revisions import load_revision_spec

    revision = load_revision_spec(Path.cwd(), "v29", require_clean=False)

    with pytest.raises(TypeError):
        revision.configuration["hyperparameters"]["learning_rate"] = 1.0
    with pytest.raises(TypeError):
        revision.configuration["structure_hyperparameters"][
            "covariance_penalty_weight"
        ] = 0.0


def test_extended_authority_rejects_unbounded_revision_denominators() -> None:
    from maskimpute_benchmark.revisions import (
        RevisionAuthorityError,
        derive_extended_selection_authority,
        load_revision_spec,
    )
    from maskimpute_benchmark.selection import load_publication_execution_authority

    base = load_publication_execution_authority()
    v28 = load_revision_spec(Path.cwd(), "v28", require_clean=True)
    v29 = load_revision_spec(Path.cwd(), "v29", require_clean=False)

    with pytest.raises(RevisionAuthorityError, match="denominator"):
        derive_extended_selection_authority(
            base,
            (v28, v29, v29),
            (
                _activation("v28", "v28"),
                _activation("v29", "v29"),
                _activation("v29", "v29"),
            ),
        )


def test_revision_stage_paths_are_fixed_and_version_separated() -> None:
    from maskimpute_benchmark.revisions import (
        development_selection_stage_paths,
        revision_stage_paths,
    )

    base = development_selection_stage_paths(None)
    v28 = revision_stage_paths("v28")
    v29 = revision_stage_paths("v29")

    assert base.source_selection_input == (
        "artifacts/study/development/evaluation/development_selection_input.json"
    )
    assert base.selection_complete_input == (
        "artifacts/study/development/evaluation/"
        "development_selection_input-downstream.json"
    )
    assert base.downstream_directory == (
        "artifacts/study/development/evaluation/downstream"
    )
    assert v28.reconstruction_directory == (
        "artifacts/study/development/competition-v28-revision"
    )
    assert v28.orthogonal_directory == (
        "artifacts/study/development/evaluation/orthogonal-v28-revision"
    )
    assert v28.selection_input == (
        "artifacts/study/development/evaluation/development_selection_input-v28.json"
    )
    assert v28.selection_complete_input == (
        "artifacts/study/development/evaluation/"
        "development_selection_input-v28-downstream.json"
    )
    assert v29.selection_complete_input == (
        "artifacts/study/development/evaluation/"
        "development_selection_input-v29-downstream.json"
    )
    assert v28.activation_selection_input == base.selection_complete_input
    assert v29.activation_selection_input == v28.selection_complete_input
    assert v29.activation_selection_report == v28.selection_report
    assert len(
        {
            v28.reconstruction_directory,
            v28.orthogonal_directory,
            v29.reconstruction_directory,
            v29.orthogonal_directory,
        }
    ) == 4


def test_revision_activation_fails_closed_when_fixed_evidence_is_absent(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.revisions import (
        RevisionAuthorityError,
        validate_revision_activation,
    )

    with pytest.raises(RevisionAuthorityError, match="fixed.*selection input"):
        validate_revision_activation(tmp_path, "v28", require_clean=False)


def test_revision_authority_reader_rejects_writable_and_hardlinked_files(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.revisions import (
        RevisionAuthorityError,
        _read_stable_bytes,
    )

    writable = tmp_path / "writable.json"
    writable.write_bytes(b"{}\n")
    writable.chmod(0o666)
    with pytest.raises(RevisionAuthorityError, match="unsafe"):
        _read_stable_bytes(writable, "writable authority")

    original = tmp_path / "original.json"
    alias = tmp_path / "alias.json"
    original.write_bytes(b"{}\n")
    os.link(original, alias)
    with pytest.raises(RevisionAuthorityError, match="unsafe"):
        _read_stable_bytes(original, "hardlinked authority")


@pytest.mark.parametrize(
    ("gate_name", "match"),
    (
        ("required_comparator_completeness", "comparator denominator"),
        ("candidate_completeness", "candidate denominator"),
    ),
)
def test_revision_activation_rejects_incomplete_preceding_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_name: str,
    match: str,
) -> None:
    import maskimpute_benchmark.selection as selection
    from maskimpute_benchmark.revisions import (
        RevisionAuthorityError,
        revision_stage_paths,
        validate_revision_activation,
    )

    paths = revision_stage_paths("v28")
    (tmp_path / "study").mkdir()
    (tmp_path / paths.revision_authority).write_bytes(
        Path(paths.revision_authority).read_bytes()
    )
    selection_input = {
        "schema_version": 4,
        "revision_versions": [],
        "result_sha256": SHA_B,
    }
    gates = {
        "candidate_completeness": {"passed": True},
        "required_comparator_completeness": {"passed": True},
    }
    gates[gate_name]["passed"] = False
    report = {
        "assessments": [
            {
                "configuration_id": "v27-c03-calibrated-r1-g1",
                "efficacy_pass": False,
                "gates": gates,
                "version": "v27",
            }
        ],
        "selected_configuration": None,
        "trigger": "v28",
    }
    for relative, payload in (
        (paths.activation_selection_input, selection_input),
        (paths.activation_selection_report, report),
    ):
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        selection,
        "_select_for_repository",
        lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: report),
    )

    with pytest.raises(RevisionAuthorityError, match=match):
        validate_revision_activation(tmp_path, "v28", require_clean=False)

    gates[gate_name]["passed"] = True
    (tmp_path / paths.activation_selection_report).write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    activation = validate_revision_activation(
        tmp_path,
        "v28",
        require_clean=False,
    )
    assert activation.trigger == "v28"


def test_v29_activation_requires_structure_failure_in_its_exact_v28_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection as selection
    from maskimpute_benchmark.revisions import (
        RevisionAuthorityError,
        revision_stage_paths,
        validate_revision_activation,
    )

    paths = revision_stage_paths("v29")
    (tmp_path / "study").mkdir()
    (tmp_path / paths.revision_authority).write_bytes(
        Path(paths.revision_authority).read_bytes()
    )
    report = {
        "assessments": [
            {
                "configuration_id": "v28-c01-nb-parent-c03",
                "efficacy_pass": True,
                "gates": {
                    "candidate_completeness": {"passed": True},
                    "corr_err_degradation": {"passed": True},
                    "orthogonal_safety": {"passed": True},
                    "required_comparator_completeness": {"passed": True},
                },
                "version": "v28",
            }
        ],
        "selected_configuration": None,
        "trigger": "v29",
    }
    selection_input = {
        "schema_version": 4,
        "revision_versions": ["v28"],
        "result_sha256": SHA_B,
    }
    for relative, payload in (
        (paths.activation_selection_input, selection_input),
        (paths.activation_selection_report, report),
    ):
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        selection,
        "_select_for_repository",
        lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: report),
    )

    with pytest.raises(RevisionAuthorityError, match="exact v28 parent.*structure"):
        validate_revision_activation(tmp_path, "v29", require_clean=False)

    report["assessments"][0]["gates"]["corr_err_degradation"]["passed"] = False
    (tmp_path / paths.activation_selection_report).write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    activation = validate_revision_activation(
        tmp_path,
        "v29",
        require_clean=False,
    )
    assert activation.trigger == "v29"


def test_tracked_v29_runner_authority_is_conditional_and_structure_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.runner as runner

    authority = runner.load_v29_revision_authority()
    candidates = tuple(
        value
        for value in authority.configurations
        if value.method_id == "maskimpute"
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.configuration_id == "v29-c01-structure-parent-v28-c01"
    assert candidate.payload["method_version"] == "v29"
    assert candidate.payload["decoder"] == "negative_binomial"
    assert candidate.payload["structure_hyperparameters"] == {
        "variable_gene_count": 200,
        "neighborhood_k": 15,
        "covariance_penalty_weight": 0.1,
        "neighborhood_penalty_weight": 0.1,
        "variance_floor": 1e-8,
    }
    with pytest.raises(runner.RunnerContractError, match="v29.*activation"):
        runner.run_v29_revision_competition()


def test_revision_runners_do_not_accept_output_directory_overrides() -> None:
    import maskimpute_benchmark.runner as runner

    with pytest.raises(TypeError):
        runner.run_v28_revision_competition(Path("arbitrary"))
    with pytest.raises(TypeError):
        runner.run_v29_revision_competition(Path("arbitrary"))


@pytest.mark.parametrize(
    "script_name",
    (
        "run_v28_revision_competition.py",
        "run_v29_revision_competition.py",
        "build_v28_revision_selection_input.py",
        "build_v29_revision_selection_input.py",
        "select_v28_revision_candidate.py",
        "select_v29_revision_candidate.py",
    ),
)
def test_revision_entry_points_expose_no_scientific_or_path_overrides(
    script_name: str,
) -> None:
    script = Path("scripts") / script_name
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    help_text = result.stdout.lower()
    for forbidden in (
        "--output",
        "--input",
        "--configuration",
        "--version",
        "--through-version",
    ):
        assert forbidden not in help_text


def test_revision_selector_reads_only_complete_input_and_publishes_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.revision_commands as commands
    import maskimpute_benchmark.revisions as revisions
    import maskimpute_benchmark.selection as selection
    import maskimpute_benchmark.selection_promotion as promotion

    paths = revisions.revision_stage_paths("v28")
    observed: list[Path] = []
    payload = {"schema_version": 4, "revision_versions": ["v28"]}
    report = {
        "selected_configuration": None,
        "trigger": "v29",
    }

    def read(path: Path, _name: str):
        observed.append(path)
        return payload, "1" * 64

    class Selection:
        def to_dict(self):
            return report

    monkeypatch.setattr(promotion, "_secure_canonical_json", read)
    monkeypatch.setattr(
        selection,
        "_select_for_repository",
        lambda selected, repository, *, require_clean: Selection()
        if selected is payload and repository == tmp_path.absolute() and require_clean
        else pytest.fail("revision selector authority arguments changed"),
    )
    report_path = tmp_path / paths.selection_report
    report_path.parent.mkdir(parents=True, exist_ok=True)

    assert commands.select_revision_main("v28", tmp_path, []) == 0
    assert observed == [tmp_path / paths.selection_complete_input]
    published = report_path.read_bytes()
    assert commands.select_revision_main("v28", tmp_path, []) == 0
    assert report_path.read_bytes() == published

    report["trigger"] = "different"
    assert commands.select_revision_main("v28", tmp_path, []) == 2
    assert report_path.read_bytes() == published


def test_revision_selector_rejects_symlink_parent_before_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.revision_commands as commands
    import maskimpute_benchmark.revisions as revisions
    import maskimpute_benchmark.selection as selection

    paths = revisions.revision_stage_paths("v28")
    input_path = tmp_path / paths.selection_complete_input
    real_parent = input_path.parent.with_name("evaluation-real")
    real_parent.mkdir(parents=True)
    (real_parent / input_path.name).write_bytes(b'{"schema_version":4}\n')
    input_path.parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(
        selection,
        "_select_for_repository",
        lambda *_args, **_kwargs: pytest.fail(
            "symlink-parent revision input reached selection"
        ),
    )

    assert commands.select_revision_main("v28", tmp_path, []) == 2


def test_revision_selector_rejects_schema_three_at_complete_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.revision_commands as commands
    import maskimpute_benchmark.selection as selection
    import maskimpute_benchmark.selection_promotion as promotion

    monkeypatch.setattr(
        promotion,
        "_secure_canonical_json",
        lambda *_args: (
            {"schema_version": 3, "revision_versions": ["v28"]},
            "1" * 64,
        ),
    )
    monkeypatch.setattr(
        selection,
        "_select_for_repository",
        lambda *_args, **_kwargs: pytest.fail(
            "schema-three complete-path input reached selection"
        ),
    )

    assert commands.select_revision_main("v28", tmp_path, []) == 2
