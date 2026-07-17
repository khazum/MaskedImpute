"""Staged, byte-bound development evaluation for conditional revisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from .development_evaluation import (
    OrthogonalOutputEvidence,
    ReconstructionEvidence,
    _canonical_json_bytes,
    _publish_bound_file,
    _verify_bound_repository_file,
)
from .revisions import (
    RevisionActivation,
    RevisionSpec,
    revision_stage_paths,
    thaw_revision_configuration,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RevisionEvaluationError(RuntimeError):
    """Raised when staged selection evidence is incomplete or overlaps."""


@dataclass(frozen=True, slots=True)
class PublicRevisionOrthogonalExecutor:
    """Run one activated revision on truth-free real-data MethodInput values."""

    method_spec: object
    count_model_config: object
    calibration_artifact: object
    device: str = "cuda"

    def __call__(self, request: object):
        from maskimpute import MaskImputeConfig
        from maskimpute_benchmark.development_evaluation import (
            OrthogonalExecutionRequest,
        )
        from maskimpute_benchmark.methods.maskimpute import _run_in_tree
        from maskimpute_benchmark.runner import (
            AuthorizedConfiguration,
            maskimpute_decoder_for_configuration,
            maskimpute_structure_for_configuration,
        )

        if not isinstance(request, OrthogonalExecutionRequest):
            raise TypeError("request must be OrthogonalExecutionRequest")
        payload = dict(request.configuration.payload)
        hyperparameters = payload.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise RevisionEvaluationError(
                "revision orthogonal configuration lacks hyperparameters"
            )
        authorized = AuthorizedConfiguration.create(
            method_id="maskimpute",
            configuration_id=request.configuration.configuration_id,
            kind="candidate_search",
            payload=payload,
            requires_count_score=True,
            requires_calibration=True,
            configuration_sha256=request.configuration.configuration_sha256,
        )
        decoder, decoder_config = maskimpute_decoder_for_configuration(authorized)
        structure_config = maskimpute_structure_for_configuration(authorized)
        execution = _run_in_tree(
            self.method_spec,
            request.method_input,
            variant_id="maskimpute-reference",
            calibration_artifact=self.calibration_artifact,
            seed=request.model_seed,
            config=MaskImputeConfig(**hyperparameters, seed=request.model_seed),
            count_model_config=self.count_model_config,
            device=self.device,
            development_mechanism=request.source_id,
            development_biological_id="external",
            calibration_usage="retained_all_development",
            decoder=decoder,
            decoder_config=decoder_config,
            structure_config=structure_config,
        )
        return execution.snapshot.matrix


@dataclass(frozen=True, slots=True)
class RevisionStageEvaluation:
    """One independently executed revision checkpoint and orthogonal panel."""

    spec: RevisionSpec
    activation: RevisionActivation
    reconstruction: ReconstructionEvidence
    records: tuple[Mapping[str, object], ...]
    null_de_audits: tuple[Mapping[str, object], ...]
    orthogonal: OrthogonalOutputEvidence
    intervals: tuple[Mapping[str, object], ...]
    orthogonal_audits: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RevisionSpec):
            raise TypeError("spec must be a RevisionSpec")
        if not isinstance(self.activation, RevisionActivation):
            raise TypeError("activation must be a RevisionActivation")
        if self.spec.version != self.activation.version:
            raise ValueError("revision spec and activation versions differ")
        if not isinstance(self.reconstruction, ReconstructionEvidence):
            raise TypeError("reconstruction must be ReconstructionEvidence")
        if not isinstance(self.orthogonal, OrthogonalOutputEvidence):
            raise TypeError("orthogonal must be OrthogonalOutputEvidence")


@dataclass(frozen=True, slots=True)
class AssembledRevisionEvaluation:
    """Independently reconstructed base plus consecutive revision evidence."""

    through_version: str
    dataset_manifest_sha256: str
    count_score_manifest_sha256: str
    retained_calibration_artifact_sha256: str
    base_records: tuple[Mapping[str, object], ...]
    base_intervals: tuple[Mapping[str, object], ...]
    base_evaluation_manifest_path: str
    base_evaluation_manifest_sha256: str
    stages: tuple[RevisionStageEvaluation, ...]
    authority: object

    def __post_init__(self) -> None:
        expected = (
            ("v28",)
            if self.through_version == "v28"
            else ("v28", "v29")
            if self.through_version == "v29"
            else ()
        )
        if tuple(stage.spec.version for stage in self.stages) != expected:
            raise ValueError("assembled revision stages are incomplete or reordered")


def _record_identity(record: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        record.get(field)
        for field in (
            "mechanism",
            "biological_id",
            "technical_view",
            "dataset_id",
            "method",
            "model_seed",
            "metric",
        )
    )


def _interval_identity(interval: Mapping[str, object]) -> tuple[object, object]:
    return interval.get("configuration"), interval.get("endpoint")


def combine_selection_rows(
    base_records: Sequence[Mapping[str, object]],
    base_intervals: Sequence[Mapping[str, object]],
    revision_rows: Sequence[
        tuple[
            str,
            str,
            Sequence[Mapping[str, object]],
            Sequence[Mapping[str, object]],
        ]
    ],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Union base and staged rows while retaining each stage's exact identity."""

    combined_records = [dict(value) for value in base_records]
    combined_intervals = [dict(value) for value in base_intervals]
    record_identities = {_record_identity(value) for value in combined_records}
    interval_identities = {_interval_identity(value) for value in combined_intervals}
    if len(record_identities) != len(combined_records):
        raise RevisionEvaluationError("base selection records contain duplicate identity")
    if len(interval_identities) != len(combined_intervals):
        raise RevisionEvaluationError("base intervals contain duplicate identity")
    observed_versions: list[str] = []
    for version, configuration_id, records, intervals in revision_rows:
        observed_versions.append(version)
        if version not in {"v28", "v29"}:
            raise RevisionEvaluationError("revision row version is invalid")
        for raw in records:
            record = dict(raw)
            if record.get("method") != configuration_id:
                raise RevisionEvaluationError(
                    f"{version} evidence must contain only its own candidate rows"
                )
            identity = _record_identity(record)
            if identity in record_identities:
                raise RevisionEvaluationError("duplicate selection record identity")
            record_identities.add(identity)
            combined_records.append(record)
        for raw in intervals:
            interval = dict(raw)
            if interval.get("configuration") != configuration_id:
                raise RevisionEvaluationError(
                    f"{version} intervals must contain only its own candidate"
                )
            identity = _interval_identity(interval)
            if identity in interval_identities:
                raise RevisionEvaluationError("duplicate orthogonal interval identity")
            interval_identities.add(identity)
            combined_intervals.append(interval)
    if tuple(observed_versions) not in {("v28",), ("v28", "v29")}:
        raise RevisionEvaluationError("revision rows are not consecutive from v28")
    return tuple(combined_records), tuple(combined_intervals)


def _validate_stage_runner_authority(spec: RevisionSpec, authority: object) -> None:
    """Bind tracked revision semantics to the exact independently loaded runner."""

    from .runner import RunnerAuthority

    if type(spec) is not RevisionSpec:
        raise TypeError("spec must be an exact RevisionSpec")
    if type(authority) is not RunnerAuthority:
        raise TypeError("authority must be an exact RunnerAuthority")
    candidates = tuple(
        value
        for value in authority.configurations
        if value.method_id == "maskimpute"
    )
    if len(candidates) != 1:
        raise RevisionEvaluationError(
            f"{spec.version} runner must contain exactly one revision candidate"
        )
    candidate = candidates[0]
    if (
        candidate.configuration_id != spec.configuration_id
        or candidate.configuration_sha256 != spec.configuration_sha256
        or candidate.kind != "candidate_search"
        or not candidate.requires_count_score
        or not candidate.requires_calibration
        or dict(candidate.payload) != thaw_revision_configuration(spec)
    ):
        raise RevisionEvaluationError(
            f"{spec.version} runner configuration differs from tracked revision"
        )


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RevisionEvaluationError(f"{name} is not a lowercase SHA-256")


def _activation_dict(value: RevisionActivation) -> dict[str, object]:
    return {
        "version": value.version,
        "trigger": value.trigger,
        "selection_input_path": value.selection_input_path,
        "selection_input_file_sha256": value.selection_input_file_sha256,
        "selection_result_sha256": value.selection_result_sha256,
        "selection_report_path": value.selection_report_path,
        "selection_report_file_sha256": value.selection_report_file_sha256,
    }


def _spec_dict(value: RevisionSpec) -> dict[str, object]:
    return {
        "version": value.version,
        "path": value.relative_path,
        "file_sha256": value.file_sha256,
        "trigger": value.trigger,
        "parent_configuration_id": value.parent_configuration_id,
        "parent_configuration_sha256": value.parent_configuration_sha256,
        "configuration_id": value.configuration_id,
        "configuration_sha256": value.configuration_sha256,
        "reason_code": value.reason_code,
    }


def _reconstruction_dict(
    repository: Path,
    stage: RevisionStageEvaluation,
) -> dict[str, object]:
    paths = revision_stage_paths(stage.spec.version)
    checkpoint_relative = str(
        PurePosixPath(paths.reconstruction_directory)
        / stage.reconstruction.checkpoint_path
    )
    _verify_bound_repository_file(
        repository,
        checkpoint_relative,
        stage.reconstruction.checkpoint_file_sha256,
        f"{stage.spec.version} reconstruction checkpoint",
    )
    raw_artifacts = []
    for binding in stage.reconstruction.raw_artifacts:
        relative = str(
            PurePosixPath(paths.reconstruction_directory) / binding.path
        )
        _verify_bound_repository_file(
            repository,
            relative,
            binding.file_sha256,
            f"{stage.spec.version} reconstruction {binding.kind}",
        )
        raw_artifacts.append({**asdict(binding), "path": relative})
    return {
        "checkpoint_path": checkpoint_relative,
        "checkpoint_file_sha256": stage.reconstruction.checkpoint_file_sha256,
        "checkpoint_sha256": stage.reconstruction.checkpoint_sha256,
        "plan_sha256": stage.reconstruction.plan_sha256,
        "input_hashes": dict(stage.reconstruction.input_hashes),
        "raw_artifacts": raw_artifacts,
    }


def _orthogonal_dict(
    repository: Path,
    stage: RevisionStageEvaluation,
) -> dict[str, object]:
    paths = revision_stage_paths(stage.spec.version)
    manifest_relative = str(
        PurePosixPath(paths.orthogonal_directory) / stage.orthogonal.manifest_path.name
    )
    _verify_bound_repository_file(
        repository,
        manifest_relative,
        stage.orthogonal.manifest_file_sha256,
        f"{stage.spec.version} orthogonal manifest",
    )
    for index, record in enumerate(stage.orthogonal.records):
        if record.get("status") != "completed":
            continue
        output_path = record.get("output_path")
        output_sha = record.get("output_file_sha256")
        if not isinstance(output_path, str) or not isinstance(output_sha, str):
            raise RevisionEvaluationError("revision orthogonal output binding is partial")
        relative = str(PurePosixPath(paths.orthogonal_directory) / output_path)
        _verify_bound_repository_file(
            repository,
            relative,
            output_sha,
            f"{stage.spec.version} orthogonal output {index}",
        )
    return {
        "manifest_path": manifest_relative,
        "manifest_file_sha256": stage.orthogonal.manifest_file_sha256,
        "manifest_sha256": stage.orthogonal.manifest_sha256,
        "records": [dict(value) for value in stage.orthogonal.records],
    }


def _stage_evidence_dict(
    repository: Path,
    stage: RevisionStageEvaluation,
) -> dict[str, object]:
    from .protocol import canonical_sha256

    paths = revision_stage_paths(stage.spec.version)
    _verify_bound_repository_file(
        repository,
        stage.spec.relative_path,
        stage.spec.file_sha256,
        f"{stage.spec.version} tracked revision authority",
    )
    for relative, digest, name in (
        (
            stage.activation.selection_input_path,
            stage.activation.selection_input_file_sha256,
            "activation selection input",
        ),
        (
            stage.activation.selection_report_path,
            stage.activation.selection_report_file_sha256,
            "activation selection report",
        ),
    ):
        _verify_bound_repository_file(
            repository,
            relative,
            digest,
            f"{stage.spec.version} {name}",
        )
    if (
        stage.spec.relative_path != paths.revision_authority
        or stage.activation.selection_input_path != paths.activation_selection_input
        or stage.activation.selection_report_path != paths.activation_selection_report
    ):
        raise RevisionEvaluationError("revision evidence paths are not fixed")
    return {
        "version": stage.spec.version,
        "authority": _spec_dict(stage.spec),
        "activation": _activation_dict(stage.activation),
        "reconstruction": _reconstruction_dict(repository, stage),
        "selection_records_sha256": canonical_sha256(
            [dict(value) for value in stage.records]
        ),
        "null_de_audits": [dict(value) for value in stage.null_de_audits],
        "orthogonal": _orthogonal_dict(repository, stage),
        "orthogonal_intervals_sha256": canonical_sha256(
            [dict(value) for value in stage.intervals]
        ),
        "orthogonal_audits": [dict(value) for value in stage.orthogonal_audits],
    }


def write_revision_selection_artifacts(
    repository: Path,
    *,
    through_version: str,
    dataset_manifest_sha256: str,
    count_score_manifest_sha256: str,
    retained_calibration_artifact_sha256: str,
    base_records: Sequence[Mapping[str, object]],
    base_intervals: Sequence[Mapping[str, object]],
    base_evaluation_manifest_path: str,
    base_evaluation_manifest_sha256: str,
    stages: Sequence[RevisionStageEvaluation],
) -> tuple[Path, Path]:
    """Write schema 3 and an acyclic manifest binding every stage separately."""

    from .protocol import canonical_sha256

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.absolute()
    stage_values = tuple(stages)
    expected_versions = (
        ("v28",) if through_version == "v28" else ("v28", "v29")
        if through_version == "v29"
        else ()
    )
    if tuple(stage.spec.version for stage in stage_values) != expected_versions:
        raise RevisionEvaluationError("revision stage denominator is incomplete")
    for name, value in (
        ("dataset manifest checksum", dataset_manifest_sha256),
        ("count-score manifest checksum", count_score_manifest_sha256),
        ("calibration artifact checksum", retained_calibration_artifact_sha256),
        ("base evaluation manifest checksum", base_evaluation_manifest_sha256),
    ):
        _require_sha256(value, name)
    _verify_bound_repository_file(
        root,
        "artifacts/study/development/count_scores/manifest.json",
        count_score_manifest_sha256,
        "count-score manifest",
    )
    _verify_bound_repository_file(
        root,
        "artifacts/study/development/calibration/retained_calibration.json",
        retained_calibration_artifact_sha256,
        "retained calibration artifact",
    )
    _verify_bound_repository_file(
        root,
        base_evaluation_manifest_path,
        base_evaluation_manifest_sha256,
        "base evaluation manifest",
    )
    revision_rows = tuple(
        (
            stage.spec.version,
            stage.spec.configuration_id,
            stage.records,
            stage.intervals,
        )
        for stage in stage_values
    )
    records, intervals = combine_selection_rows(
        base_records,
        base_intervals,
        revision_rows,
    )
    revision_evidence = [
        _stage_evidence_dict(root, stage) for stage in stage_values
    ]
    evidence_core = {
        "schema_version": 3,
        "revision_versions": list(expected_versions),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "count_score_manifest_sha256": count_score_manifest_sha256,
        "retained_calibration_artifact_sha256": (
            retained_calibration_artifact_sha256
        ),
        "records": list(records),
        "orthogonal_intervals": list(intervals),
    }
    base_activation = stage_values[0].activation
    evaluation_core = {
        "schema_version": 2,
        "artifact_type": "maskimpute_development_revision_evaluation_manifest",
        "selection_evidence_sha256": canonical_sha256(evidence_core),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "count_score_manifest": {
            "path": "artifacts/study/development/count_scores/manifest.json",
            "file_sha256": count_score_manifest_sha256,
        },
        "retained_calibration_artifact": {
            "path": (
                "artifacts/study/development/calibration/retained_calibration.json"
            ),
            "file_sha256": retained_calibration_artifact_sha256,
        },
        "base_selection": {
            "input_path": base_activation.selection_input_path,
            "input_file_sha256": base_activation.selection_input_file_sha256,
            "result_sha256": base_activation.selection_result_sha256,
            "report_path": base_activation.selection_report_path,
            "report_file_sha256": base_activation.selection_report_file_sha256,
            "evaluation_manifest_path": base_evaluation_manifest_path,
            "evaluation_manifest_file_sha256": base_evaluation_manifest_sha256,
        },
        "revisions": revision_evidence,
        "combined_score": None,
    }
    evaluation_payload = {
        **evaluation_core,
        "manifest_sha256": canonical_sha256(evaluation_core),
    }
    output_paths = revision_stage_paths(through_version)
    evaluation_path = root / output_paths.evaluation_manifest
    evaluation_file_sha = _publish_bound_file(
        evaluation_path,
        _canonical_json_bytes(evaluation_payload) + b"\n",
    )
    result_core = {
        **evidence_core,
        "evaluation_manifest_sha256": evaluation_file_sha,
    }
    result_payload = {
        **result_core,
        "result_sha256": canonical_sha256(result_core),
    }
    result_path = root / output_paths.selection_input
    _publish_bound_file(result_path, _canonical_json_bytes(result_payload) + b"\n")
    return result_path, evaluation_path


def validate_revision_artifact_payloads(
    repository: Path,
    data: Mapping[str, object],
    assembled: AssembledRevisionEvaluation,
) -> Mapping[str, str]:
    """Revalidate schema 3 and compare every declared stage to rebuilt evidence."""

    from .development_evaluation import _strict_json
    from .protocol import canonical_sha256

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if not isinstance(assembled, AssembledRevisionEvaluation):
        raise TypeError("assembled must be AssembledRevisionEvaluation")
    root = repository.absolute()
    expected_fields = {
        "schema_version",
        "revision_versions",
        "dataset_manifest_sha256",
        "count_score_manifest_sha256",
        "retained_calibration_artifact_sha256",
        "evaluation_manifest_sha256",
        "records",
        "orthogonal_intervals",
        "result_sha256",
    }
    if type(data) is not dict or set(data) != expected_fields:
        raise RevisionEvaluationError(
            "revision selection input has missing or extra fields"
        )
    expected_versions = [stage.spec.version for stage in assembled.stages]
    unsigned_result = {key: value for key, value in data.items() if key != "result_sha256"}
    if (
        data["schema_version"] != 3
        or type(data["schema_version"]) is not int
        or data["revision_versions"] != expected_versions
        or canonical_sha256(unsigned_result) != data["result_sha256"]
    ):
        raise RevisionEvaluationError("revision selection input checksum differs")
    records, intervals = combine_selection_rows(
        assembled.base_records,
        assembled.base_intervals,
        tuple(
            (
                stage.spec.version,
                stage.spec.configuration_id,
                stage.records,
                stage.intervals,
            )
            for stage in assembled.stages
        ),
    )
    if (
        data["dataset_manifest_sha256"] != assembled.dataset_manifest_sha256
        or data["count_score_manifest_sha256"]
        != assembled.count_score_manifest_sha256
        or data["retained_calibration_artifact_sha256"]
        != assembled.retained_calibration_artifact_sha256
        or data["records"] != list(records)
        or data["orthogonal_intervals"] != list(intervals)
    ):
        raise RevisionEvaluationError(
            "revision selection rows differ from independently rebuilt evidence"
        )
    evaluation_file_sha = data["evaluation_manifest_sha256"]
    if not isinstance(evaluation_file_sha, str):
        raise RevisionEvaluationError("revision evaluation checksum is invalid")
    paths = revision_stage_paths(assembled.through_version)
    evaluation, raw = _strict_json(
        root / paths.evaluation_manifest,
        "revision evaluation manifest",
    )
    if hashlib.sha256(raw).hexdigest() != evaluation_file_sha:
        raise RevisionEvaluationError("revision evaluation file checksum differs")
    manifest_fields = {
        "schema_version",
        "artifact_type",
        "selection_evidence_sha256",
        "dataset_manifest_sha256",
        "count_score_manifest",
        "retained_calibration_artifact",
        "base_selection",
        "revisions",
        "combined_score",
        "manifest_sha256",
    }
    unsigned_manifest = {
        key: value for key, value in evaluation.items() if key != "manifest_sha256"
    }
    evidence_core = {
        key: data[key]
        for key in (
            "schema_version",
            "revision_versions",
            "dataset_manifest_sha256",
            "count_score_manifest_sha256",
            "retained_calibration_artifact_sha256",
            "records",
            "orthogonal_intervals",
        )
    }
    base = assembled.stages[0].activation
    expected_base = {
        "input_path": base.selection_input_path,
        "input_file_sha256": base.selection_input_file_sha256,
        "result_sha256": base.selection_result_sha256,
        "report_path": base.selection_report_path,
        "report_file_sha256": base.selection_report_file_sha256,
        "evaluation_manifest_path": assembled.base_evaluation_manifest_path,
        "evaluation_manifest_file_sha256": (
            assembled.base_evaluation_manifest_sha256
        ),
    }
    expected_revisions = [
        _stage_evidence_dict(root, stage) for stage in assembled.stages
    ]
    if (
        set(evaluation) != manifest_fields
        or evaluation["schema_version"] != 2
        or evaluation["artifact_type"]
        != "maskimpute_development_revision_evaluation_manifest"
        or evaluation["selection_evidence_sha256"]
        != canonical_sha256(evidence_core)
        or evaluation["dataset_manifest_sha256"]
        != assembled.dataset_manifest_sha256
        or evaluation["count_score_manifest"]
        != {
            "path": "artifacts/study/development/count_scores/manifest.json",
            "file_sha256": assembled.count_score_manifest_sha256,
        }
        or evaluation["retained_calibration_artifact"]
        != {
            "path": (
                "artifacts/study/development/calibration/retained_calibration.json"
            ),
            "file_sha256": assembled.retained_calibration_artifact_sha256,
        }
        or evaluation["base_selection"] != expected_base
        or evaluation["revisions"] != expected_revisions
        or evaluation["combined_score"] is not None
        or evaluation["manifest_sha256"] != canonical_sha256(unsigned_manifest)
    ):
        raise RevisionEvaluationError(
            "revision evidence manifest differs from independently rebuilt evidence"
        )
    bindings = {
        "revision_evaluation_manifest_file_sha256": evaluation_file_sha,
        "revision_evaluation_manifest_payload_sha256": str(
            evaluation["manifest_sha256"]
        ),
        "revision_selection_evidence_sha256": str(
            evaluation["selection_evidence_sha256"]
        ),
    }
    for stage in assembled.stages:
        prefix = stage.spec.version
        bindings.update(
            {
                f"{prefix}_revision_authority_file_sha256": stage.spec.file_sha256,
                f"{prefix}_activation_input_file_sha256": (
                    stage.activation.selection_input_file_sha256
                ),
                f"{prefix}_activation_report_file_sha256": (
                    stage.activation.selection_report_file_sha256
                ),
                f"{prefix}_reconstruction_checkpoint_file_sha256": (
                    stage.reconstruction.checkpoint_file_sha256
                ),
                f"{prefix}_reconstruction_checkpoint_payload_sha256": (
                    stage.reconstruction.checkpoint_sha256
                ),
                f"{prefix}_reconstruction_raw_artifacts_sha256": canonical_sha256(
                    expected_revisions[expected_versions.index(prefix)][
                        "reconstruction"
                    ]["raw_artifacts"]
                ),
                f"{prefix}_orthogonal_manifest_file_sha256": (
                    stage.orthogonal.manifest_file_sha256
                ),
                f"{prefix}_orthogonal_manifest_payload_sha256": (
                    stage.orthogonal.manifest_sha256
                ),
            }
        )
    return bindings


def run_revision_orthogonal_outputs(
    repository: Path,
    version: str,
) -> OrthogonalOutputEvidence:
    """Run/resume one activated revision at its fixed separate orthogonal path."""

    from maskimpute import PreZeroCountModelConfig
    from maskimpute.calibration import load_calibration_artifact

    from .development_evaluation import (
        OrthogonalConfiguration,
        _thaw_authority_value,
        prepare_real_orthogonal_panel,
        produce_orthogonal_outputs,
    )
    from .methods import load_method_registry
    from .runner import (
        load_activated_v28_revision_authority,
        load_activated_v29_revision_authority,
        load_runner_authority,
    )
    from .revisions import load_revision_spec, validate_revision_activation

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.resolve(strict=True)
    if root != Path(__file__).resolve().parents[1]:
        raise RevisionEvaluationError(
            "revision orthogonal execution must use the active repository"
        )
    validate_revision_activation(root, version)
    spec = load_revision_spec(root, version)
    stage_runner_authority = (
        load_activated_v28_revision_authority()
        if version == "v28"
        else load_activated_v29_revision_authority()
    )
    _validate_stage_runner_authority(spec, stage_runner_authority)
    base = load_runner_authority()
    calibration_sha = base.retained_calibration_sha256
    if calibration_sha is None:
        raise RevisionEvaluationError(
            "retained calibration is pending for revision orthogonal execution"
        )
    calibration_path = root / base.retained_calibration_path
    _verify_bound_repository_file(
        root,
        base.retained_calibration_path,
        calibration_sha,
        "revision retained calibration artifact",
    )
    calibration = load_calibration_artifact(calibration_path)
    count_payload = _thaw_authority_value(base.count_model_config)
    if not isinstance(count_payload, dict):
        raise RevisionEvaluationError("count-model authority is malformed")
    panel = prepare_real_orthogonal_panel(root)
    configuration = OrthogonalConfiguration(
        spec.configuration_id,
        spec.configuration_sha256,
        thaw_revision_configuration(spec),
    )
    registry = load_method_registry(root / "study/methods.json")
    executor = PublicRevisionOrthogonalExecutor(
        method_spec=registry.by_id("maskimpute"),
        count_model_config=PreZeroCountModelConfig(**count_payload),
        calibration_artifact=calibration,
    )
    return produce_orthogonal_outputs(
        root / revision_stage_paths(version).orthogonal_directory,
        inputs=panel.method_inputs,
        configurations=(configuration,),
        model_seeds=(42, 43, 44),
        artifact_bindings={
            "count_model_config_sha256": base.count_model_config_sha256,
            "retained_calibration_artifact_sha256": calibration_sha,
            "score_fit_policy": (
                "refit_cross_fitted_count_score_from_truth_free_input"
            ),
        },
        executor=executor,
    )


def assemble_revision_evaluation(
    repository: Path,
    through_version: str,
    *,
    execute_missing_orthogonal: bool = False,
    require_clean: bool = True,
) -> AssembledRevisionEvaluation:
    """Independently rebuild all rows from the base and separate stage artifacts."""

    from .development_evaluation import (
        OrthogonalConfiguration,
        _orthogonal_authority_core,
        _strict_json,
        build_reconstruction_selection_records,
        evaluate_real_orthogonal_intervals,
        load_completed_reconstruction_checkpoint,
        load_orthogonal_output_evidence,
        prepare_real_orthogonal_panel,
    )
    from .methods import load_method_registry
    from .runner import (
        build_competition_plan,
        load_activated_v28_revision_authority,
        load_activated_v29_revision_authority,
        load_prepared_development_panel,
    )
    from .selection import load_publication_execution_authority
    from .revisions import (
        derive_extended_selection_authority,
        load_revision_spec,
        validate_revision_activation,
    )

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if type(execute_missing_orthogonal) is not bool or type(require_clean) is not bool:
        raise TypeError("execution and clean flags must be boolean")
    root = repository.resolve(strict=True)
    if root != Path(__file__).resolve().parents[1]:
        raise RevisionEvaluationError(
            "revision evaluation must use the active repository"
        )
    versions = (
        ("v28",)
        if through_version == "v28"
        else ("v28", "v29")
        if through_version == "v29"
        else ()
    )
    if not versions:
        raise ValueError("through_version must be v28 or v29")
    specs = tuple(
        load_revision_spec(root, version, require_clean=require_clean)
        for version in versions
    )
    activations = tuple(
        validate_revision_activation(root, version, require_clean=require_clean)
        for version in versions
    )
    base_authority = load_publication_execution_authority()
    extended_authority = derive_extended_selection_authority(
        base_authority,
        specs,
        activations,
    )
    base_input, _base_raw = _strict_json(
        root / activations[0].selection_input_path,
        "base development selection input",
    )
    if base_input.get("result_sha256") != activations[0].selection_result_sha256:
        raise RevisionEvaluationError("base selection result binding differs")
    base_evaluation_path = (
        "artifacts/study/development/evaluation/evaluation_manifest.json"
    )
    base_evaluation_sha = base_input.get("evaluation_manifest_sha256")
    if not isinstance(base_evaluation_sha, str):
        raise RevisionEvaluationError("base evaluation manifest binding is absent")
    _verify_bound_repository_file(
        root,
        base_evaluation_path,
        base_evaluation_sha,
        "base development evaluation manifest",
    )
    base_records = base_input.get("records")
    base_intervals = base_input.get("orthogonal_intervals")
    if not isinstance(base_records, list) or not isinstance(base_intervals, list):
        raise RevisionEvaluationError("base selection rows are invalid")
    for name in (
        "dataset_manifest_sha256",
        "count_score_manifest_sha256",
        "retained_calibration_artifact_sha256",
    ):
        value = base_input.get(name)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise RevisionEvaluationError(f"base {name} binding is invalid")
    panel = prepare_real_orthogonal_panel(root)
    registry = load_method_registry(root / "study/methods.json")
    artifact_bindings = {
        "count_model_config_sha256": base_authority.count_model_config_sha256,
        "retained_calibration_artifact_sha256": (
            base_input["retained_calibration_artifact_sha256"]
        ),
        "score_fit_policy": (
            "refit_cross_fitted_count_score_from_truth_free_input"
        ),
    }
    stages: list[RevisionStageEvaluation] = []
    for version, spec, activation in zip(versions, specs, activations, strict=True):
        runner_authority = (
            load_activated_v28_revision_authority()
            if version == "v28"
            else load_activated_v29_revision_authority()
        )
        _validate_stage_runner_authority(spec, runner_authority)
        bindings, prepared = load_prepared_development_panel(runner_authority)
        paths = revision_stage_paths(version)
        reconstruction_directory = root / paths.reconstruction_directory
        checkpoint_payload, _checkpoint_raw = _strict_json(
            reconstruction_directory / "checkpoint.json",
            f"{version} reconstruction checkpoint",
        )
        input_hashes = checkpoint_payload.get("input_hashes")
        environment_sha = (
            input_hashes.get("execution_environment_sha256")
            if isinstance(input_hashes, dict)
            else None
        )
        if not isinstance(environment_sha, str) or not _SHA256.fullmatch(
            environment_sha
        ):
            raise RevisionEvaluationError(
                f"{version} execution environment binding is absent"
            )
        plan = build_competition_plan(
            registry,
            bindings,
            runner_authority,
            execution_environment_sha256=environment_sha,
        )
        reconstruction = load_completed_reconstruction_checkpoint(
            reconstruction_directory,
            plan,
            prepared_datasets=prepared,
        )
        rebuilt = build_reconstruction_selection_records(
            reconstruction,
            checkpoint_directory=reconstruction_directory,
            prepared_datasets=prepared,
            declarations=extended_authority.declarations,
            method_bindings=extended_authority.method_bindings,
        )
        records = tuple(
            value
            for value in rebuilt.records
            if value.get("method") == spec.configuration_id
        )
        null_audits = tuple(
            value
            for value in rebuilt.null_de_audits
            if value.get("method") == spec.configuration_id
        )
        if not records or not null_audits:
            raise RevisionEvaluationError(
                f"{version} checkpoint produced no candidate selection evidence"
            )
        orthogonal_configuration = OrthogonalConfiguration(
            spec.configuration_id,
            spec.configuration_sha256,
            thaw_revision_configuration(spec),
        )
        expected_orthogonal_authority = _orthogonal_authority_core(
            panel.method_inputs,
            (orthogonal_configuration,),
            (42, 43, 44),
            artifact_bindings,
        )
        orthogonal_directory = root / paths.orthogonal_directory
        if (orthogonal_directory / "orthogonal_outputs.json").is_file():
            orthogonal = load_orthogonal_output_evidence(
                orthogonal_directory,
                expected_authority=expected_orthogonal_authority,
            )
        elif execute_missing_orthogonal:
            orthogonal = run_revision_orthogonal_outputs(root, version)
        else:
            raise RevisionEvaluationError(
                f"{version} fixed orthogonal output manifest is absent"
            )
        endpoint = evaluate_real_orthogonal_intervals(
            orthogonal,
            panel.cite,
            panel.tung,
            (spec.configuration_id,),
        )
        stages.append(
            RevisionStageEvaluation(
                spec=spec,
                activation=activation,
                reconstruction=reconstruction,
                records=tuple(records),
                null_de_audits=tuple(null_audits),
                orthogonal=orthogonal,
                intervals=tuple(endpoint.intervals),
                orthogonal_audits=tuple(endpoint.audits),
            )
        )
    return AssembledRevisionEvaluation(
        through_version=through_version,
        dataset_manifest_sha256=str(base_input["dataset_manifest_sha256"]),
        count_score_manifest_sha256=str(base_input["count_score_manifest_sha256"]),
        retained_calibration_artifact_sha256=str(
            base_input["retained_calibration_artifact_sha256"]
        ),
        base_records=tuple(dict(value) for value in base_records),
        base_intervals=tuple(dict(value) for value in base_intervals),
        base_evaluation_manifest_path=base_evaluation_path,
        base_evaluation_manifest_sha256=base_evaluation_sha,
        stages=tuple(stages),
        authority=extended_authority,
    )


def build_revision_selection_input(
    repository: Path,
    through_version: str,
) -> tuple[Path, Path]:
    """Build one fixed combined selection input after all stage outputs complete."""

    assembled = assemble_revision_evaluation(
        repository,
        through_version,
        execute_missing_orthogonal=True,
        require_clean=True,
    )
    return write_revision_selection_artifacts(
        repository,
        through_version=assembled.through_version,
        dataset_manifest_sha256=assembled.dataset_manifest_sha256,
        count_score_manifest_sha256=assembled.count_score_manifest_sha256,
        retained_calibration_artifact_sha256=(
            assembled.retained_calibration_artifact_sha256
        ),
        base_records=assembled.base_records,
        base_intervals=assembled.base_intervals,
        base_evaluation_manifest_path=assembled.base_evaluation_manifest_path,
        base_evaluation_manifest_sha256=(
            assembled.base_evaluation_manifest_sha256
        ),
        stages=assembled.stages,
    )


@dataclass(frozen=True, slots=True)
class ValidatedRevisionEvaluation:
    authority: object
    bindings: Mapping[str, str]


def validate_revision_selection_evaluation(
    repository: Path,
    data: Mapping[str, object],
    through_version: str,
    *,
    require_clean: bool = True,
) -> ValidatedRevisionEvaluation:
    """Read-only schema-3 validator used by publication candidate selection."""

    try:
        assembled = assemble_revision_evaluation(
            repository,
            through_version,
            execute_missing_orthogonal=False,
            require_clean=require_clean,
        )
        bindings = validate_revision_artifact_payloads(repository, data, assembled)
    except RevisionEvaluationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RevisionEvaluationError(
            f"revision evidence assembly failed: {error}"
        ) from error
    return ValidatedRevisionEvaluation(
        authority=assembled.authority,
        bindings=bindings,
    )


__all__ = [
    "RevisionEvaluationError",
    "AssembledRevisionEvaluation",
    "RevisionStageEvaluation",
    "PublicRevisionOrthogonalExecutor",
    "combine_selection_rows",
    "assemble_revision_evaluation",
    "build_revision_selection_input",
    "run_revision_orthogonal_outputs",
    "validate_revision_artifact_payloads",
    "validate_revision_selection_evaluation",
    "ValidatedRevisionEvaluation",
    "write_revision_selection_artifacts",
]
