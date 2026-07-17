from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest


def _record(method: str, metric: str = "mse", *, value: float = 1.0):
    return {
        "mechanism": "symsim",
        "biological_id": "draw-01",
        "technical_view": "moderate",
        "dataset_id": "dataset-test",
        "dataset_sha256": "a" * 64,
        "method": method,
        "method_sha256": "b" * 64,
        "model_seed": 42,
        "metric": metric,
        "value": value,
        "status": "completed",
    }


def _interval(configuration: str, endpoint: str = "rna_protein_concordance"):
    return {
        "configuration": configuration,
        "endpoint": endpoint,
        "comparison": "candidate_minus_observed",
        "estimate": 0.1,
        "ci_lower": 0.0,
        "ci_upper": 0.2,
        "status": "completed",
    }


def test_combined_rows_keep_stage_provenance_and_reject_identity_collisions() -> None:
    from maskimpute_benchmark.revision_evaluation import (
        RevisionEvaluationError,
        combine_selection_rows,
    )

    base_record = _record("v27-c03-calibrated-r1-g1")
    v28_record = _record("v28-c01-nb-parent-c03")
    base_interval = _interval("v27-c03-calibrated-r1-g1")
    v28_interval = _interval("v28-c01-nb-parent-c03")

    records, intervals = combine_selection_rows(
        (base_record,),
        (base_interval,),
        (("v28", "v28-c01-nb-parent-c03", (v28_record,), (v28_interval,)),),
    )
    assert records == (base_record, v28_record)
    assert intervals == (base_interval, v28_interval)

    with pytest.raises(RevisionEvaluationError, match="only its own candidate"):
        combine_selection_rows(
            (base_record,),
            (base_interval,),
            (("v28", "v28-c01-nb-parent-c03", (base_record,), (v28_interval,)),),
        )
    with pytest.raises(RevisionEvaluationError, match="duplicate selection record"):
        combine_selection_rows(
            (base_record,),
            (base_interval,),
            (
                (
                    "v28",
                    "v27-c03-calibrated-r1-g1",
                    (replace_record(base_record),),
                    (v28_interval,),
                ),
            ),
        )


def replace_record(record: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(record))


def test_revision_manifest_binds_distinct_checkpoint_raw_and_orthogonal_bytes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        OrthogonalOutputEvidence,
        RawArtifactBinding,
        ReconstructionEvidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.revision_evaluation import (
        AssembledRevisionEvaluation,
        RevisionStageEvaluation,
        validate_revision_artifact_payloads,
        write_revision_selection_artifacts,
    )
    from maskimpute_benchmark.revisions import (
        RevisionActivation,
        load_revision_spec,
        revision_stage_paths,
    )

    repository = tmp_path / "repo"
    repository.mkdir()
    paths = revision_stage_paths("v28")
    revision_path = repository / paths.revision_authority
    revision_path.parent.mkdir(parents=True)
    revision_path.write_bytes(Path(paths.revision_authority).read_bytes())
    count_score_path = (
        repository / "artifacts/study/development/count_scores/manifest.json"
    )
    calibration_path = (
        repository
        / "artifacts/study/development/calibration/retained_calibration.json"
    )
    count_score_path.parent.mkdir(parents=True)
    calibration_path.parent.mkdir(parents=True)
    count_score_path.write_bytes(b"count-score\n")
    calibration_path.write_bytes(b"calibration\n")
    count_score_sha = hashlib.sha256(count_score_path.read_bytes()).hexdigest()
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    base_input_path = repository / paths.activation_selection_input
    base_report_path = repository / paths.activation_selection_report
    base_evaluation_path = repository / (
        "artifacts/study/development/evaluation/evaluation_manifest.json"
    )
    for path, raw in (
        (base_input_path, b'{"result_sha256":"' + b"1" * 64 + b'"}\n'),
        (base_report_path, b"{}\n"),
        (base_evaluation_path, b"{}\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    reconstruction_dir = repository / paths.reconstruction_directory
    reconstruction_dir.mkdir(parents=True)
    checkpoint = reconstruction_dir / "checkpoint.json"
    checkpoint.write_bytes(b"revision-checkpoint\n")
    raw_output = reconstruction_dir / "runs/output.bin"
    raw_output.parent.mkdir()
    raw_output.write_bytes(b"revision-output\n")
    reconstruction = ReconstructionEvidence(
        checkpoint_path="checkpoint.json",
        checkpoint_file_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_sha256="2" * 64,
        plan_sha256="3" * 64,
        input_hashes={"runner_authority_sha256": "4" * 64},
        records=(),
        raw_artifacts=(
            RawArtifactBinding(
                run_id="run-test",
                kind="evaluator_output",
                path="runs/output.bin",
                file_sha256=hashlib.sha256(raw_output.read_bytes()).hexdigest(),
            ),
        ),
    )

    orthogonal_dir = repository / paths.orthogonal_directory
    orthogonal_dir.mkdir(parents=True)
    orthogonal_manifest = orthogonal_dir / "orthogonal_outputs.json"
    orthogonal_manifest.write_bytes(b"revision-orthogonal\n")
    orthogonal = OrthogonalOutputEvidence(
        output_directory=orthogonal_dir,
        manifest_path=orthogonal_manifest,
        manifest_file_sha256=hashlib.sha256(
            orthogonal_manifest.read_bytes()
        ).hexdigest(),
        manifest_sha256="5" * 64,
        records=(),
    )
    spec = load_revision_spec(Path.cwd(), "v28", require_clean=True)
    activation = RevisionActivation(
        version="v28",
        trigger="v28",
        selection_input_path=paths.activation_selection_input,
        selection_input_file_sha256=hashlib.sha256(
            base_input_path.read_bytes()
        ).hexdigest(),
        selection_result_sha256="1" * 64,
        selection_report_path=paths.activation_selection_report,
        selection_report_file_sha256=hashlib.sha256(
            base_report_path.read_bytes()
        ).hexdigest(),
    )
    stage = RevisionStageEvaluation(
        spec=spec,
        activation=activation,
        reconstruction=reconstruction,
        records=(_record(spec.configuration_id),),
        null_de_audits=(),
        orthogonal=orthogonal,
        intervals=(_interval(spec.configuration_id),),
        orthogonal_audits=(),
    )
    result_path, evaluation_path = write_revision_selection_artifacts(
        repository,
        through_version="v28",
        dataset_manifest_sha256="6" * 64,
        count_score_manifest_sha256=count_score_sha,
        retained_calibration_artifact_sha256=calibration_sha,
        base_records=(_record("v27-c03-calibrated-r1-g1"),),
        base_intervals=(_interval("v27-c03-calibrated-r1-g1"),),
        base_evaluation_manifest_path=(
            "artifacts/study/development/evaluation/evaluation_manifest.json"
        ),
        base_evaluation_manifest_sha256=hashlib.sha256(
            base_evaluation_path.read_bytes()
        ).hexdigest(),
        stages=(stage,),
    )

    result = json.loads(result_path.read_bytes())
    evaluation = json.loads(evaluation_path.read_bytes())
    assert result["schema_version"] == 3
    assert result["revision_versions"] == ["v28"]
    assert result["evaluation_manifest_sha256"] == hashlib.sha256(
        evaluation_path.read_bytes()
    ).hexdigest()
    assert result["result_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    revision = evaluation["revisions"][0]
    assert revision["reconstruction"]["checkpoint_path"] == (
        paths.reconstruction_directory + "/checkpoint.json"
    )
    assert revision["reconstruction"]["raw_artifacts"][0]["path"] == (
        paths.reconstruction_directory + "/runs/output.bin"
    )
    assert revision["orthogonal"]["manifest_path"] == (
        paths.orthogonal_directory + "/orthogonal_outputs.json"
    )
    assert evaluation["combined_score"] is None

    assembled = AssembledRevisionEvaluation(
        through_version="v28",
        dataset_manifest_sha256="6" * 64,
        count_score_manifest_sha256=count_score_sha,
        retained_calibration_artifact_sha256=calibration_sha,
        base_records=(_record("v27-c03-calibrated-r1-g1"),),
        base_intervals=(_interval("v27-c03-calibrated-r1-g1"),),
        base_evaluation_manifest_path=(
            "artifacts/study/development/evaluation/evaluation_manifest.json"
        ),
        base_evaluation_manifest_sha256=hashlib.sha256(
            base_evaluation_path.read_bytes()
        ).hexdigest(),
        stages=(stage,),
        authority=None,
    )
    bindings = validate_revision_artifact_payloads(repository, result, assembled)
    assert bindings["v28_reconstruction_checkpoint_file_sha256"] == (
        reconstruction.checkpoint_file_sha256
    )
    assert bindings["v28_orthogonal_manifest_file_sha256"] == (
        orthogonal.manifest_file_sha256
    )

    tampered = json.loads(evaluation_path.read_bytes())
    tampered["revisions"][0]["reconstruction"]["checkpoint_sha256"] = "9" * 64
    unsigned = {key: value for key, value in tampered.items() if key != "manifest_sha256"}
    tampered["manifest_sha256"] = canonical_sha256(unsigned)
    evaluation_path.write_text(
        json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n"
    )
    result["evaluation_manifest_sha256"] = hashlib.sha256(
        evaluation_path.read_bytes()
    ).hexdigest()
    result["result_sha256"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    with pytest.raises(Exception, match="revision evidence"):
        validate_revision_artifact_payloads(repository, result, assembled)

    raw_output.write_bytes(b"tampered\n")
    with pytest.raises(Exception, match="checksum"):
        write_revision_selection_artifacts(
            repository,
            through_version="v28",
            dataset_manifest_sha256="6" * 64,
            count_score_manifest_sha256=count_score_sha,
            retained_calibration_artifact_sha256=calibration_sha,
            base_records=(_record("v27-c03-calibrated-r1-g1"),),
            base_intervals=(_interval("v27-c03-calibrated-r1-g1"),),
            base_evaluation_manifest_path=(
                "artifacts/study/development/evaluation/evaluation_manifest.json"
            ),
            base_evaluation_manifest_sha256=hashlib.sha256(
                base_evaluation_path.read_bytes()
            ).hexdigest(),
            stages=(stage,),
        )


def test_public_revision_validation_normalizes_internal_authority_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.revision_evaluation as revision_evaluation
    from maskimpute_benchmark.revisions import RevisionAuthorityError

    def fail_assembly(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RevisionAuthorityError("activation bytes differ")

    monkeypatch.setattr(
        revision_evaluation,
        "assemble_revision_evaluation",
        fail_assembly,
    )
    with pytest.raises(
        revision_evaluation.RevisionEvaluationError,
        match="revision evidence assembly failed",
    ):
        revision_evaluation.validate_revision_selection_evaluation(
            Path.cwd(),
            {},
            "v28",
            require_clean=False,
        )


def test_stage_evaluation_rejects_runner_and_revision_configuration_drift() -> None:
    from maskimpute_benchmark.revision_evaluation import (
        RevisionEvaluationError,
        _validate_stage_runner_authority,
    )
    from maskimpute_benchmark.revisions import load_revision_spec
    from maskimpute_benchmark.runner import load_v29_revision_authority

    spec = load_revision_spec(Path.cwd(), "v29", require_clean=False)
    authority = load_v29_revision_authority()
    _validate_stage_runner_authority(spec, authority)

    with pytest.raises(RevisionEvaluationError, match="runner configuration"):
        _validate_stage_runner_authority(
            replace(spec, configuration_sha256="f" * 64),
            authority,
        )
