from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_canonical(path: Path, value: object) -> str:
    raw = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _source_payload(through_version: str | None) -> dict[str, object]:
    from maskimpute_benchmark.protocol import canonical_sha256

    core: dict[str, object] = {
        "schema_version": 2 if through_version is None else 3,
        "dataset_manifest_sha256": "1" * 64,
        "count_score_manifest_sha256": "2" * 64,
        "retained_calibration_artifact_sha256": "3" * 64,
        "evaluation_manifest_sha256": "4" * 64,
        "records": [],
        "orthogonal_intervals": [],
    }
    if through_version is not None:
        core["revision_versions"] = (
            ["v28"] if through_version == "v28" else ["v28", "v29"]
        )
    return {**core, "result_sha256": canonical_sha256(core)}


def _prepare_fake_stage(
    repository: Path,
    through_version: str | None,
) -> tuple[dict[str, object], object, str, str]:
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    paths = development_selection_stage_paths(through_version)
    source = _source_payload(through_version)
    source_file_sha = _write_canonical(
        repository / paths.source_selection_input,
        source,
    )
    manifest = {"manifest_sha256": "5" * 64}
    manifest_file_sha = _write_canonical(
        repository / paths.downstream_directory / "downstream_manifest.json",
        manifest,
    )
    return source, paths, source_file_sha, manifest_file_sha


def _fake_attachment(
    repository: Path,
    source: dict[str, object],
    paths: object,
    source_file_sha: str,
    manifest_file_sha: str,
):
    from maskimpute_benchmark.protocol import canonical_sha256

    def attach(
        payload: object,
        selected_repository: Path,
        relative_directory: str,
    ) -> dict[str, object]:
        assert payload == source
        assert selected_repository == repository
        assert relative_directory == paths.downstream_directory
        source_core = {
            key: value for key, value in source.items() if key != "result_sha256"
        }
        revisions = source.get("revision_versions", [])
        binding = {
            "path": paths.downstream_directory,
            "source_selection_input_path": paths.source_selection_input,
            "source_selection_input_file_sha256": source_file_sha,
            "source_selection_result_sha256": source["result_sha256"],
            "manifest_file_sha256": manifest_file_sha,
            "manifest_sha256": "5" * 64,
        }
        promoted_core = {
            **source_core,
            "schema_version": 4,
            "revision_versions": revisions,
            "downstream_evidence": binding,
        }
        return {
            **promoted_core,
            "result_sha256": canonical_sha256(promoted_core),
        }

    return attach


@pytest.mark.parametrize("through_version", (None, "v28", "v29"))
def test_promotes_each_stage_to_immutable_canonical_schema_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    through_version: str | None,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    repository = tmp_path / "repository"
    repository.mkdir()
    source, paths, source_file_sha, manifest_file_sha = _prepare_fake_stage(
        repository,
        through_version,
    )
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        _fake_attachment(
            repository,
            source,
            paths,
            source_file_sha,
            manifest_file_sha,
        ),
    )

    receipt = promotion.promote_development_selection_input(
        repository,
        through_version,
    )
    published = json.loads(
        (repository / paths.selection_complete_input).read_text(encoding="utf-8")
    )

    assert published["schema_version"] == 4
    assert receipt.source_selection_input_path == paths.source_selection_input
    assert receipt.source_selection_input_file_sha256 == source_file_sha
    assert receipt.downstream_manifest_file_sha256 == manifest_file_sha
    assert receipt.selection_complete_input_path == paths.selection_complete_input
    assert receipt.selection_complete_result_sha256 == published["result_sha256"]
    assert receipt.selection_complete_input_file_sha256 == hashlib.sha256(
        _canonical_bytes(published)
    ).hexdigest()

    repeated = promotion.promote_development_selection_input(
        repository,
        through_version,
    )
    assert repeated == receipt


def test_promotion_never_overwrites_a_conflicting_complete_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    repository = tmp_path / "repository"
    repository.mkdir()
    source, paths, source_file_sha, manifest_file_sha = _prepare_fake_stage(
        repository,
        None,
    )
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        _fake_attachment(
            repository,
            source,
            paths,
            source_file_sha,
            manifest_file_sha,
        ),
    )
    destination = repository / paths.selection_complete_input
    conflicting = b'{"conflict":true}\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(conflicting)

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="conflicts",
    ):
        promotion.promote_development_selection_input(repository, None)

    assert destination.read_bytes() == conflicting


def test_interrupted_link_publication_leaves_no_partial_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    repository = tmp_path / "repository"
    repository.mkdir()
    source, paths, source_file_sha, manifest_file_sha = _prepare_fake_stage(
        repository,
        None,
    )
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        _fake_attachment(
            repository,
            source,
            paths,
            source_file_sha,
            manifest_file_sha,
        ),
    )
    monkeypatch.setattr(
        promotion.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("simulated interrupted link")
        ),
    )

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="could not be published",
    ):
        promotion.promote_development_selection_input(repository, None)

    destination = repository / paths.selection_complete_input
    assert not os.path.lexists(destination)
    assert list(destination.parent.glob(f".{destination.name}.*.tmp")) == []


def test_parent_swap_cannot_publish_into_replacement_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    repository = tmp_path / "repository"
    repository.mkdir()
    source, paths, source_file_sha, manifest_file_sha = _prepare_fake_stage(
        repository,
        None,
    )
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        _fake_attachment(
            repository,
            source,
            paths,
            source_file_sha,
            manifest_file_sha,
        ),
    )
    destination = repository / paths.selection_complete_input
    parent = destination.parent
    displaced = parent.with_name(f"{parent.name}-displaced")
    replacement = parent.with_name(f"{parent.name}-replacement")
    replacement.mkdir()
    real_link = promotion.os.link
    swapped = False

    def swap_parent_before_link(source_name, destination_name, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            parent.rename(displaced)
            replacement.rename(parent)
            swapped = True
            if not kwargs.get("src_dir_fd"):
                malicious_source = parent / Path(source_name).name
                malicious_source.write_bytes(b'{"malicious":true}\n')
        return real_link(source_name, destination_name, *args, **kwargs)

    monkeypatch.setattr(promotion.os, "link", swap_parent_before_link)

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="parent path changed",
    ):
        promotion.promote_development_selection_input(repository, None)

    assert not os.path.lexists(destination)


def test_symlink_target_is_rejected_without_touching_referent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    repository = tmp_path / "repository"
    repository.mkdir()
    source, paths, source_file_sha, manifest_file_sha = _prepare_fake_stage(
        repository,
        None,
    )
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        _fake_attachment(
            repository,
            source,
            paths,
            source_file_sha,
            manifest_file_sha,
        ),
    )
    destination = repository / paths.selection_complete_input
    referent = tmp_path / "referent.json"
    referent.write_bytes(b"unchanged\n")
    destination.symlink_to(referent)

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="unique regular|symlink|unsafe",
    ):
        promotion.promote_development_selection_input(repository, None)

    assert destination.is_symlink()
    assert referent.read_bytes() == b"unchanged\n"


def test_symlink_parent_is_rejected_before_source_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    repository = tmp_path / "repository"
    repository.mkdir()
    _source, paths, _source_file_sha, _manifest_file_sha = _prepare_fake_stage(
        repository,
        None,
    )
    parent = (repository / paths.source_selection_input).parent
    referent = parent.with_name(f"{parent.name}-referent")
    parent.rename(referent)
    parent.symlink_to(referent, target_is_directory=True)
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        lambda *_args: pytest.fail("symlinked source reached attachment"),
    )

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="parent path|not a directory|cannot open",
    ):
        promotion.promote_development_selection_input(repository, None)

    assert not (referent / Path(paths.selection_complete_input).name).exists()


def test_invalid_source_schema_is_rejected_before_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    repository = tmp_path / "repository"
    repository.mkdir()
    paths = development_selection_stage_paths(None)
    _write_canonical(
        repository / paths.source_selection_input,
        {"schema_version": 4, "result_sha256": "0" * 64},
    )
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        lambda *_args: pytest.fail("invalid source reached attachment"),
    )

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="source selection input",
    ):
        promotion.promote_development_selection_input(repository, None)


def test_manifest_tamper_after_attachment_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion

    repository = tmp_path / "repository"
    repository.mkdir()
    source, paths, source_file_sha, manifest_file_sha = _prepare_fake_stage(
        repository,
        None,
    )
    monkeypatch.setattr(
        promotion,
        "attach_downstream_evidence_to_selection_result",
        _fake_attachment(
            repository,
            source,
            paths,
            source_file_sha,
            manifest_file_sha,
        ),
    )
    _write_canonical(
        repository / paths.downstream_directory / "downstream_manifest.json",
        {"manifest_sha256": "9" * 64},
    )

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="downstream manifest binding differs",
    ):
        promotion.promote_development_selection_input(repository, None)

    assert not (repository / paths.selection_complete_input).exists()


def test_latest_stage_missing_downstream_never_falls_back(
    tmp_path: Path,
) -> None:
    import maskimpute_benchmark.selection_promotion as promotion
    from maskimpute_benchmark.revisions import development_selection_stage_paths

    repository = tmp_path / "repository"
    repository.mkdir()
    base = development_selection_stage_paths(None)
    v28 = development_selection_stage_paths("v28")
    _write_canonical(repository / v28.source_selection_input, _source_payload("v28"))
    earlier_complete = b'{"earlier":"complete"}\n'
    (repository / base.selection_complete_input).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (repository / base.selection_complete_input).write_bytes(earlier_complete)

    with pytest.raises(
        promotion.SelectionPromotionError,
        match="downstream evidence .*absent",
    ):
        promotion.promote_latest_development_selection_input(repository)

    assert not (repository / v28.selection_complete_input).exists()
    assert (repository / base.selection_complete_input).read_bytes() == earlier_complete


def test_promotion_cli_has_no_stage_or_path_overrides() -> None:
    script = Path("scripts/promote_development_selection_input.py")
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "--input",
        "--output",
        "--version",
        "--through-version",
        "--downstream",
    ):
        assert forbidden not in completed.stdout


def test_promotion_cli_uses_only_repository_owned_latest_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path("scripts/promote_development_selection_input.py").absolute()
    specification = importlib.util.spec_from_file_location(
        "promote_development_selection_input_test",
        script_path,
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    repository = tmp_path / "repository"
    repository.mkdir()
    receipt = SimpleNamespace(
        to_dict=lambda: {
            "selection_complete_input_file_sha256": "a" * 64,
            "through_version": "v28",
        }
    )
    observed: list[Path] = []
    monkeypatch.setattr(script, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        script,
        "promote_latest_development_selection_input",
        lambda selected_repository: observed.append(selected_repository) or receipt,
    )

    assert script.main([]) == 0
    assert observed == [repository]
    assert json.loads(capsys.readouterr().out) == receipt.to_dict()
