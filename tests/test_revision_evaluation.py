from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

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
        "run_identity": None,
        "selected_configuration": None,
        "model_seed": 42,
        "metric": metric,
        "value": value,
        "status": "completed",
    }


@lru_cache(maxsize=1)
def _comparator_selection() -> dict[str, object]:
    path = Path(__file__).with_name("test_comparator_tuning.py")
    spec = importlib.util.spec_from_file_location("_revision_task11_factory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = module.smoke_registry.__wrapped__()
    authority = module.smoke_authority.__wrapped__(registry)
    bound = module.smoke_bound_rows.__wrapped__(registry, authority)
    outcomes = module.complete_smoke_outcomes.__wrapped__(bound)
    fixture = module.complete_selection_fixture.__wrapped__(
        registry,
        authority,
        bound,
        outcomes,
    )
    from maskimpute_benchmark.comparator_tuning import (
        build_comparator_selection_receipt,
        comparator_selection_projection,
        comparator_selection_projection_value,
    )

    receipt = build_comparator_selection_receipt(**fixture)
    return comparator_selection_projection_value(
        comparator_selection_projection(receipt)
    )


def _write_comparator_selection_authority(repository: Path) -> None:
    for relative in ("study/methods.json", "study/comparator_tuning.json"):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(relative), destination)
    selection = _comparator_selection()
    receipt_path = repository / str(selection["path"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(selection["receipt"], sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "field",
    (
        "selected_by_method",
        "nonexecution_identity_by_method",
        "ready_comparison_population_ids",
    ),
)
def test_schema3_writer_rejects_comparator_projection_tamper_with_unchanged_receipt(
    tmp_path: Path,
    field: str,
) -> None:
    from maskimpute_benchmark.revision_evaluation import (
        RevisionEvaluationError,
        write_revision_selection_artifacts,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    _write_comparator_selection_authority(repository)
    tampered = json.loads(json.dumps(_comparator_selection()))
    if field == "selected_by_method":
        tampered[field] = {"magic": {"forged": True}}
    elif field == "nonexecution_identity_by_method":
        tampered[field] = {"magic": {"forged": True}}
    else:
        tampered[field] = ["observed"]

    with pytest.raises(RevisionEvaluationError, match="comparator selection"):
        write_revision_selection_artifacts(
            repository,
            through_version="v28",
            dataset_manifest_sha256="1" * 64,
            count_score_manifest_sha256="2" * 64,
            retained_calibration_artifact_sha256="3" * 64,
            base_records=(),
            base_intervals=(),
            base_evaluation_manifest_path=(
                "artifacts/study/development/evaluation/evaluation_manifest.json"
            ),
            base_evaluation_manifest_sha256="4" * 64,
            comparator_selection=tampered,
            stages=(),
        )


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
        _require_inherited_rows_unchanged,
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
    _require_inherited_rows_unchanged(
        (base_record,),
        (base_interval,),
        records,
        intervals,
    )

    mutated = replace_record(base_record)
    mutated["value"] = 999.0
    with pytest.raises(RevisionEvaluationError, match="inherited base rows"):
        _require_inherited_rows_unchanged(
            (base_record,),
            (base_interval,),
            (mutated, v28_record),
            intervals,
        )

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
    _write_comparator_selection_authority(repository)
    paths = revision_stage_paths("v28")
    revision_path = repository / paths.revision_authority
    revision_path.parent.mkdir(parents=True, exist_ok=True)
    revision_path.write_bytes(Path(paths.revision_authority).read_bytes())
    count_score_path = (
        repository / "artifacts/study/development/count_scores/manifest.json"
    )
    calibration_path = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    count_score_path.parent.mkdir(parents=True)
    calibration_path.parent.mkdir(parents=True)
    count_score_path.write_bytes(b"count-score\n")
    calibration_path.write_bytes(b"calibration\n")
    count_score_sha = hashlib.sha256(count_score_path.read_bytes()).hexdigest()
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    base_input_path = repository / paths.activation_selection_input
    base_report_path = repository / paths.activation_selection_report
    for path, raw in (
        (base_input_path, b'{"result_sha256":"' + b"1" * 64 + b'"}\n'),
        (base_report_path, b"{}\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    base_checkpoint_path = (
        repository
        / "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    base_checkpoint_body = {
        "plan_sha256": "7" * 64,
        "input_hashes": {"runner_authority_sha256": "8" * 64},
        "records": [
            {
                "run": {
                    "run_id": "run-base",
                    "method_id": "maskimpute",
                    "configuration_id": "v27-c03-calibrated-r1-g1",
                    "configuration_kind": "candidate_search",
                    "status": "completed",
                    "reason": None,
                }
            }
        ],
    }
    base_checkpoint = {
        **base_checkpoint_body,
        "checkpoint_sha256": canonical_sha256(base_checkpoint_body),
    }
    base_checkpoint_path.parent.mkdir(parents=True)
    base_checkpoint_raw = (
        json.dumps(base_checkpoint, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    base_checkpoint_path.write_bytes(base_checkpoint_raw)
    base_evaluation_path = repository / (
        "artifacts/study/development/evaluation/evaluation_manifest.json"
    )
    base_reconstruction = {
        "checkpoint_path": (
            "artifacts/study/development/competition-reconstruction/checkpoint.json"
        ),
        "checkpoint_file_sha256": hashlib.sha256(base_checkpoint_raw).hexdigest(),
        "checkpoint_sha256": base_checkpoint["checkpoint_sha256"],
        "plan_sha256": base_checkpoint["plan_sha256"],
        "input_hashes": base_checkpoint["input_hashes"],
        "raw_artifacts": [],
    }
    base_evaluation_body = {
        "schema_version": 1,
        "reconstruction": base_reconstruction,
    }
    base_evaluation = {
        **base_evaluation_body,
        "manifest_sha256": canonical_sha256(base_evaluation_body),
    }
    base_evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    base_evaluation_path.write_bytes(
        json.dumps(base_evaluation, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )

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
        records=(
            {
                "run": {
                    "run_id": "run-test",
                    "configuration_id": "v28-c01-nb-parent-c03",
                    "status": "completed",
                    "reason": None,
                }
            },
        ),
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
        base_comparator_selection=_comparator_selection(),
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
        comparator_selection=_comparator_selection(),
        stages=(stage,),
    )

    result = json.loads(result_path.read_bytes())
    evaluation = json.loads(evaluation_path.read_bytes())
    assert result["schema_version"] == 3
    assert result["revision_versions"] == ["v28"]
    assert (
        result["evaluation_manifest_sha256"]
        == hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    )
    assert result["result_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    revision = evaluation["revisions"][0]
    assert revision["activation"]["base_comparator_selection"] == (
        _comparator_selection()
    )
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
        comparator_selection=_comparator_selection(),
        stages=(stage,),
        authority=SimpleNamespace(
            declarations=(
                SimpleNamespace(id="v27-c03-calibrated-r1-g1"),
                SimpleNamespace(id=spec.configuration_id),
            )
        ),
    )
    bindings = validate_revision_artifact_payloads(repository, result, assembled)
    assert bindings["v28_reconstruction_checkpoint_file_sha256"] == (
        reconstruction.checkpoint_file_sha256
    )
    assert bindings["v28_orthogonal_manifest_file_sha256"] == (
        orthogonal.manifest_file_sha256
    )
    assert (
        bindings["base_reconstruction_checkpoint_file_sha256"]
        == (base_reconstruction["checkpoint_file_sha256"])
    )
    assert bindings["base_reconstruction_statuses_sha256"] == canonical_sha256(
        [{"run_id": "run-base", "status": "completed", "reason": None}]
    )
    assert bindings["v28_reconstruction_plan_sha256"] == reconstruction.plan_sha256
    assert bindings["v28_reconstruction_input_hashes_sha256"] == canonical_sha256(
        dict(reconstruction.input_hashes)
    )
    assert bindings["v28_reconstruction_statuses_sha256"] == canonical_sha256(
        [{"run_id": "run-test", "status": "completed", "reason": None}]
    )
    assert (
        bindings["v28_evaluation_manifest_file_sha256"]
        == hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    )

    tampered = json.loads(evaluation_path.read_bytes())
    tampered["revisions"][0]["reconstruction"]["checkpoint_sha256"] = "9" * 64
    unsigned = {
        key: value for key, value in tampered.items() if key != "manifest_sha256"
    }
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
            comparator_selection=_comparator_selection(),
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
        _require_stage_comparator_selection,
        _validate_stage_runner_authority,
    )
    from maskimpute_benchmark.revisions import (
        RevisionActivation,
        load_revision_spec,
        revision_stage_paths,
    )
    from maskimpute_benchmark.runner import load_v29_revision_authority

    spec = load_revision_spec(Path.cwd(), "v29", require_clean=False)
    authority = load_v29_revision_authority()
    _validate_stage_runner_authority(spec, authority)

    with pytest.raises(RevisionEvaluationError, match="runner configuration"):
        _validate_stage_runner_authority(
            replace(spec, configuration_sha256="f" * 64),
            authority,
        )

    selection = _comparator_selection()
    paths = revision_stage_paths("v29")
    activation = RevisionActivation(
        version="v29",
        trigger="v29",
        selection_input_path=paths.activation_selection_input,
        selection_input_file_sha256="1" * 64,
        selection_result_sha256="2" * 64,
        selection_report_path=paths.activation_selection_report,
        selection_report_file_sha256="3" * 64,
        base_comparator_selection=selection,
    )
    activated_authority = replace(
        authority,
        base_comparator_selection=selection,
    )
    _require_stage_comparator_selection(
        selection,
        activation,
        activated_authority,
    )
    with pytest.raises(RevisionEvaluationError, match="comparator selection differs"):
        _require_stage_comparator_selection(
            selection,
            activation,
            replace(
                activated_authority,
                base_comparator_selection={
                    **selection,
                    "ready_comparison_population_ids": ["observed"],
                },
            ),
        )


def test_revision_stage_serializes_candidate_only_direct_reconstruction(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.development_evaluation import (
        DirectReconstructionEvidence,
        OrthogonalOutputEvidence,
    )
    from maskimpute_benchmark.revision_evaluation import (
        RevisionStageEvaluation,
        _outer_reconstruction_provenance,
        _reconstruction_dict,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.revisions import (
        RevisionActivation,
        load_revision_spec,
        revision_stage_paths,
    )

    repository = tmp_path / "repository"
    repository.mkdir()
    paths = revision_stage_paths("v28")
    checkpoint = repository / paths.reconstruction_directory / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'{"identity_mode":"direct-v1"}\n')
    evidence = DirectReconstructionEvidence(
        checkpoint_path="checkpoint.json",
        identity_mode="direct-v1",
        authority_revision="fair-comparator-direct-v1",
        plan_snapshot={"entries": [{"identity": {"ordinal": 1}}]},
        input_descriptors=({"dataset_id": "dataset-test"},),
        records=(
            {
                "run": {
                    "run_id": "run-test",
                    "identity": {
                        "configuration_id": "v28-c01-nb-parent-c03",
                    },
                    "status": "unavailable",
                    "reason": "runtime_environment_invalid",
                },
                "metrics": [],
                "p_pre_zero_evidence": {},
            },
        ),
        selected_by_method={},
        comparator_receipt_bytes=b"{}\n",
    )
    orthogonal_path = repository / paths.orthogonal_directory / "outputs.json"
    stage = RevisionStageEvaluation(
        spec=load_revision_spec(Path.cwd(), "v28", require_clean=False),
        activation=RevisionActivation(
            version="v28",
            trigger="v28",
            selection_input_path=paths.activation_selection_input,
            selection_input_file_sha256="1" * 64,
            selection_result_sha256="2" * 64,
            selection_report_path=paths.activation_selection_report,
            selection_report_file_sha256="3" * 64,
            base_comparator_selection=_comparator_selection(),
        ),
        reconstruction=evidence,
        records=(),
        null_de_audits=(),
        orthogonal=OrthogonalOutputEvidence(
            output_directory=orthogonal_path.parent,
            manifest_path=orthogonal_path,
            manifest_file_sha256="4" * 64,
            manifest_sha256="5" * 64,
            records=(),
        ),
        intervals=(),
        orthogonal_audits=(),
    )

    value = _reconstruction_dict(repository, stage)

    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert value == {
        "checkpoint_path": (paths.reconstruction_directory + "/checkpoint.json"),
        "checkpoint_file_sha256": checkpoint_sha,
        "checkpoint_sha256": checkpoint_sha,
        "identity_mode": "direct-v1",
        "authority_revision": "fair-comparator-direct-v1",
        "plan_snapshot": {"entries": [{"identity": {"ordinal": 1}}]},
        "input_descriptors": [{"dataset_id": "dataset-test"}],
    }
    assert _outer_reconstruction_provenance(value) == {
        "plan_sha256": canonical_sha256(value["plan_snapshot"]),
        "input_hashes_sha256": canonical_sha256(value["input_descriptors"]),
        "raw_artifacts_sha256": canonical_sha256([]),
    }


def test_direct_revision_loader_accepts_only_complete_48_candidate_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.revision_evaluation as revision_evaluation
    from maskimpute_benchmark.development_evaluation import (
        DirectReconstructionEvidence,
    )

    configuration_id = "v28-c01-nb-parent-c03"
    identities = tuple(
        SimpleNamespace(
            configuration_id=configuration_id,
            configuration_kind="candidate_search",
            method=SimpleNamespace(method_id="maskimpute"),
        )
        for _ in range(48)
    )
    plan = SimpleNamespace(
        identity_mode="direct-v1",
        entries=tuple(SimpleNamespace(identity=value) for value in identities),
        configurations=(SimpleNamespace(configuration_id=configuration_id),),
    )
    records = tuple(
        {
            "run": {
                "identity": {
                    "configuration_id": configuration_id,
                    "configuration_kind": "candidate_search",
                    "method": {"method_id": "maskimpute"},
                }
            }
        }
        for _ in range(48)
    )
    report = SimpleNamespace(
        status="completed",
        planned_run_count=48,
        records=records,
        identity_mode="direct-v1",
        authority_revision="fair-comparator-direct-v1",
        plan_snapshot={"entries": list(range(48))},
        input_descriptors=tuple(range(16)),
    )
    captured = {}

    class FakeStore:
        def __init__(self, path: Path) -> None:
            captured["path"] = path

        def load(self, observed_plan: object, **kwargs: object) -> object:
            captured["plan"] = observed_plan
            captured["load"] = kwargs
            return report

    monkeypatch.setattr(
        "maskimpute_benchmark.fair_comparator_checkpoint.DirectCheckpointStore",
        FakeStore,
    )
    monkeypatch.setattr(
        "maskimpute_benchmark.fair_comparator_plan.validate_direct_competition_plan",
        lambda *_args, **_kwargs: captured.setdefault("validated", True),
    )
    authority = SimpleNamespace(
        plan_scope="revision_candidate_only",
        configurations=(SimpleNamespace(configuration_id=configuration_id),),
    )
    selected_configuration = {"nested": {"values": [1, 2]}}
    projection = SimpleNamespace(
        selected_by_method={
            "magic": SimpleNamespace(configuration=selected_configuration)
        },
        receipt_bytes=b"{}\n",
    )

    evidence = revision_evaluation._load_direct_revision_reconstruction(
        tmp_path / "checkpoint.json",
        plan,
        repository=tmp_path,
        registry=object(),
        prepared_datasets={},
        runner_authority=authority,
        datasets=(),
        comparator_projection=projection,
    )

    assert isinstance(evidence, DirectReconstructionEvidence)
    assert len(evidence.records) == 48
    report.plan_snapshot["entries"][0] = 999
    selected_configuration["nested"]["values"].append(3)
    from maskimpute_benchmark.direct_values import direct_json_value

    assert direct_json_value(evidence.plan_snapshot, payload=True)["entries"][0] == 0
    assert direct_json_value(evidence.selected_by_method, payload=True) == {
        "magic": {"nested": {"values": [1, 2]}}
    }
    assert captured["validated"] is True
    assert captured["path"] == tmp_path / "checkpoint.json"


def test_revision_assembly_has_no_legacy_reconstruction_route() -> None:
    from maskimpute_benchmark.revision_evaluation import (
        assemble_revision_evaluation,
    )

    source = inspect.getsource(assemble_revision_evaluation)
    assert "build_competition_plan" not in source
    assert "load_completed_reconstruction_checkpoint" not in source
    assert "_load_direct_revision_reconstruction" in source
