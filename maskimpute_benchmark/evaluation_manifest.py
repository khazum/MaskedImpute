"""Fail-closed validation of development-selection evaluation evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .protocol import canonical_sha256 as _canonical_sha256

if TYPE_CHECKING:
    from .selection import SelectionAuthority


_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_INTERVAL_FIELDS = {
    "configuration",
    "endpoint",
    "comparison",
    "estimate",
    "ci_lower",
    "ci_upper",
    "status",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvaluationManifestError(RuntimeError):
    """Raised when evaluator evidence is incomplete, altered, or unauthorized."""


# Keep the extracted validator body independent of selection.py while retaining
# its fail-closed exception sites.
SelectionAuthorityError = EvaluationManifestError


def _exact_authority_mapping(
    value: object, fields: set[str], name: str
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise EvaluationManifestError(f"{name} has missing or extra fields")
    return value


def _authority_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise EvaluationManifestError(f"{name} must be a lowercase SHA-256")
    return value


def _stable_file_bytes(path: Path, name: str) -> tuple[bytes, str]:
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise SelectionAuthorityError(f"{name} path contains a symlink")
        current = current.parent
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SelectionAuthorityError(
            f"{name} is absent or cannot be opened"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o002
        ):
            raise SelectionAuthorityError(f"{name} is not a secure unique regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    raw = b"".join(chunks)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise SelectionAuthorityError(f"{name} changed while it was read")
    return raw, hashlib.sha256(raw).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SelectionAuthorityError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise SelectionAuthorityError(f"nonfinite JSON constant {value}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SelectionAuthorityError("authority contains noncanonical JSON") from error


def _read_canonical_bound_json(
    path: Path,
    name: str,
    expected_file_sha256: str,
) -> tuple[dict[str, Any], str]:
    raw, file_sha256 = _stable_file_bytes(path, name)
    if file_sha256 != expected_file_sha256:
        raise SelectionAuthorityError(f"{name} file checksum mismatch")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SelectionAuthorityError(f"could not parse {name}") from error
    if type(payload) is not dict:
        raise SelectionAuthorityError(f"{name} must be an object")
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise SelectionAuthorityError(f"{name} is not canonical JSON")
    return payload, file_sha256


def _validate_selection_evaluation_envelope(
    repository: Path,
    data: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, str]]:
    evaluation_file_sha256 = _authority_sha(
        data["evaluation_manifest_sha256"], "evaluation manifest file checksum"
    )
    path = repository / (
        "artifacts/study/development/evaluation/evaluation_manifest.json"
    )
    evaluation, observed_file_sha256 = _read_canonical_bound_json(
        path, "fixed evaluation manifest", evaluation_file_sha256
    )
    evaluation_root = _exact_authority_mapping(
        evaluation,
        {
            "schema_version",
            "artifact_type",
            "selection_evidence_sha256",
            "dataset_manifest_sha256",
            "count_score_manifest",
            "retained_calibration_artifact",
            "reconstruction",
            "orthogonal",
            "sources",
            "null_de_audits",
            "orthogonal_audits",
            "combined_score",
            "manifest_sha256",
        },
        "evaluation manifest",
    )
    unsigned = {
        key: value for key, value in evaluation_root.items() if key != "manifest_sha256"
    }
    manifest_sha256 = _authority_sha(
        evaluation_root["manifest_sha256"], "evaluation manifest payload checksum"
    )
    if (
        evaluation_root["schema_version"] != 1
        or type(evaluation_root["schema_version"]) is not int
        or evaluation_root["artifact_type"]
        != "maskimpute_development_selection_evaluation_manifest"
        or evaluation_root["combined_score"] is not None
        or _canonical_sha256(unsigned) != manifest_sha256
    ):
        raise SelectionAuthorityError("evaluation manifest payload checksum mismatch")
    evidence_core = {
        key: data[key]
        for key in (
            "schema_version",
            "dataset_manifest_sha256",
            "count_score_manifest_sha256",
            "retained_calibration_artifact_sha256",
            "records",
            "orthogonal_intervals",
        )
    }
    if evaluation_root["selection_evidence_sha256"] != _canonical_sha256(evidence_core):
        raise SelectionAuthorityError("evaluation selection evidence checksum mismatch")
    if evaluation_root["dataset_manifest_sha256"] != data["dataset_manifest_sha256"]:
        raise SelectionAuthorityError("evaluation dataset manifest binding mismatch")
    count_score = _exact_authority_mapping(
        evaluation_root["count_score_manifest"],
        {"path", "file_sha256"},
        "evaluation count-score binding",
    )
    calibration = _exact_authority_mapping(
        evaluation_root["retained_calibration_artifact"],
        {"path", "file_sha256"},
        "evaluation calibration binding",
    )
    if count_score != {
        "path": "artifacts/study/development/count_scores/manifest.json",
        "file_sha256": data["count_score_manifest_sha256"],
    }:
        raise SelectionAuthorityError("evaluation count-score binding mismatch")
    if calibration != {
        "path": "artifacts/study/development/calibration/retained_calibration.json",
        "file_sha256": data["retained_calibration_artifact_sha256"],
    }:
        raise SelectionAuthorityError("evaluation calibration binding mismatch")
    return evaluation_root, MappingProxyType(
        {
            "evaluation_manifest_file_sha256": observed_file_sha256,
            "evaluation_manifest_payload_sha256": manifest_sha256,
            "selection_evidence_sha256": evaluation_root["selection_evidence_sha256"],
        }
    )


def _validate_evaluation_source_evidence(
    repository: Path,
    raw_sources: object,
) -> Mapping[str, str]:
    try:
        from .development_evaluation import validate_real_source_artifacts

        evidence = validate_real_source_artifacts(repository)
    except Exception as error:
        raise SelectionAuthorityError(
            "evaluation source evidence failed byte revalidation"
        ) from error
    expected = {
        "ledger_path": evidence.ledger_path,
        "ledger_file_sha256": evidence.ledger_file_sha256,
        "ledger_sha256": evidence.ledger_sha256,
        "receipts": [asdict(value) for value in evidence.receipts],
        "artifacts": [asdict(value) for value in evidence.artifacts],
    }
    if raw_sources != expected:
        raise SelectionAuthorityError(
            "evaluation source evidence differs from validated source bytes"
        )
    return MappingProxyType(
        {
            "source_ledger_file_sha256": evidence.ledger_file_sha256,
            "source_ledger_payload_sha256": evidence.ledger_sha256,
        }
    )


def _validate_reconstruction_evidence(
    repository: Path,
    raw_reconstruction: object,
    authority: SelectionAuthority,
    status: Mapping[str, Any],
    data: Mapping[str, Any],
    raw_null_de_audits: object,
) -> Mapping[str, str]:
    try:
        reconstruction = _exact_authority_mapping(
            raw_reconstruction,
            {
                "checkpoint_path",
                "checkpoint_file_sha256",
                "checkpoint_sha256",
                "plan_sha256",
                "input_hashes",
                "raw_artifacts",
            },
            "evaluation reconstruction checkpoint binding",
        )
    except SelectionAuthorityError as error:
        raise SelectionAuthorityError(
            "evaluation reconstruction checkpoint binding is invalid"
        ) from error
    expected_path = (
        "artifacts/study/development/competition-reconstruction/checkpoint.json"
    )
    if reconstruction["checkpoint_path"] != expected_path:
        raise SelectionAuthorityError(
            "evaluation reconstruction checkpoint path is not fixed"
        )
    input_hashes = reconstruction["input_hashes"]
    expected_input_fields = {
        "dataset_manifest_sha256",
        "dataset_design_sha256",
        "dataset_seed_source_sha256",
        "protocol_sha256",
        "method_registry_sha256",
        "selection_contract_sha256",
        "development_search_sha256",
        "ablation_registry_sha256",
        "runner_authority_sha256",
        "execution_environment_sha256",
        "base_configuration_sha256",
        "count_model_config_sha256",
        "dataset_qc_policy_sha256",
        "count_score_manifest_sha256",
        "retained_calibration_sha256",
    }
    if type(input_hashes) is not dict or set(input_hashes) != expected_input_fields:
        raise SelectionAuthorityError(
            "reconstruction plan/input authority field set is invalid"
        )
    for name, value in input_hashes.items():
        _authority_sha(value, f"reconstruction input {name}")
    expected_inputs = {
        "dataset_manifest_sha256": data["dataset_manifest_sha256"],
        "dataset_design_sha256": status.get("design_sha256"),
        "dataset_seed_source_sha256": status.get("seed_source_sha256"),
        "protocol_sha256": authority.file_sha256["study/protocol.json"],
        "method_registry_sha256": authority.file_sha256["study/methods.json"],
        "selection_contract_sha256": authority.file_sha256[
            "study/selection_contract.json"
        ],
        "development_search_sha256": authority.file_sha256[
            "study/development_search.json"
        ],
        "ablation_registry_sha256": authority.file_sha256["study/ablations.json"],
        "base_configuration_sha256": authority.base_maskimpute_config_sha256,
        "count_model_config_sha256": authority.count_model_config_sha256,
        "dataset_qc_policy_sha256": authority.dataset_qc_policy_sha256,
        "count_score_manifest_sha256": data["count_score_manifest_sha256"],
        "retained_calibration_sha256": data["retained_calibration_artifact_sha256"],
    }
    if any(input_hashes.get(name) != value for name, value in expected_inputs.items()):
        raise SelectionAuthorityError(
            "reconstruction plan/input authority binding differs from current authority"
        )
    try:
        expected_plan = _rebuild_reconstruction_plan(
            repository,
            authority,
            status,
            input_hashes["execution_environment_sha256"],
        )
    except Exception as error:
        raise SelectionAuthorityError(
            "current reconstruction plan authority cannot be rebuilt"
        ) from error
    if (
        dict(expected_plan.input_hashes) != input_hashes
        or expected_plan.plan_sha256 != reconstruction["plan_sha256"]
    ):
        raise SelectionAuthorityError(
            "reconstruction plan authority differs from the completed checkpoint"
        )
    try:
        from .development_evaluation import (
            load_completed_reconstruction_checkpoint,
        )

        evidence = load_completed_reconstruction_checkpoint(
            (repository / expected_path).parent, expected_plan
        )
    except Exception as error:
        raise SelectionAuthorityError(
            f"reconstruction checkpoint failed plan-bound byte revalidation: {error}"
        ) from error
    expected_raw_artifacts = [
        {
            **asdict(value),
            "path": str(
                PurePosixPath("artifacts/study/development/competition-reconstruction")
                / value.path
            ),
        }
        for value in evidence.raw_artifacts
    ]
    if (
        reconstruction["checkpoint_file_sha256"] != evidence.checkpoint_file_sha256
        or reconstruction["checkpoint_sha256"] != evidence.checkpoint_sha256
        or reconstruction["plan_sha256"] != evidence.plan_sha256
        or reconstruction["input_hashes"] != dict(evidence.input_hashes)
        or reconstruction["raw_artifacts"] != expected_raw_artifacts
    ):
        raise SelectionAuthorityError(
            "reconstruction raw artifact denominator differs from checkpoint"
        )
    try:
        from .development_evaluation import build_reconstruction_selection_records

        prepared = _prepare_reconstruction_datasets(repository, expected_plan)
        rebuilt = build_reconstruction_selection_records(
            evidence,
            checkpoint_directory=(repository / expected_path).parent,
            prepared_datasets=prepared,
            declarations=authority.declarations,
            method_bindings=authority.method_bindings,
        )
        rebuilt_records = [dict(value) for value in rebuilt.records]
        rebuilt_audits = [dict(value) for value in rebuilt.null_de_audits]
    except Exception as error:
        raise SelectionAuthorityError(
            "reconstruction selection records could not be independently rebuilt"
        ) from error
    if (
        type(data.get("records")) is not list
        or _canonical_json_bytes(data["records"])
        != _canonical_json_bytes(rebuilt_records)
    ):
        raise SelectionAuthorityError(
            "reconstructed selection records differ from validated checkpoint evidence"
        )
    if (
        type(raw_null_de_audits) is not list
        or _canonical_json_bytes(raw_null_de_audits)
        != _canonical_json_bytes(rebuilt_audits)
    ):
        raise SelectionAuthorityError(
            "null-DE audits differ from independently reconstructed evaluator evidence"
        )
    return MappingProxyType(
        {
            "reconstruction_checkpoint_file_sha256": (evidence.checkpoint_file_sha256),
            "reconstruction_checkpoint_payload_sha256": (evidence.checkpoint_sha256),
            "reconstruction_plan_sha256": evidence.plan_sha256,
            "reconstruction_raw_artifacts_sha256": _canonical_sha256(
                expected_raw_artifacts
            ),
            "reconstructed_selection_records_sha256": _canonical_sha256(
                rebuilt_records
            ),
            "reconstructed_null_de_audits_sha256": _canonical_sha256(
                rebuilt_audits
            ),
        }
    )


def _prepare_reconstruction_datasets(
    repository: Path,
    expected_plan: object,
) -> Mapping[str, object]:
    """Reprepare the bound development panel and prove it yields the same plan."""

    from .methods import load_method_registry
    from .runner import (
        build_competition_plan,
        load_prepared_development_panel,
        load_runner_authority,
    )

    module_root = Path(__file__).resolve().parents[1]
    if repository.resolve(strict=True) != module_root:
        raise SelectionAuthorityError(
            "reconstruction datasets must be prepared from the active repository"
        )
    runner_authority = load_runner_authority()
    bindings, prepared = load_prepared_development_panel(runner_authority)
    registry = load_method_registry(repository / "study/methods.json")
    independently_rebuilt_plan = build_competition_plan(
        registry,
        bindings,
        runner_authority,
        execution_environment_sha256=expected_plan.input_hashes[
            "execution_environment_sha256"
        ],
    )
    if (
        independently_rebuilt_plan.plan_sha256 != expected_plan.plan_sha256
        or dict(independently_rebuilt_plan.input_hashes)
        != dict(expected_plan.input_hashes)
    ):
        raise SelectionAuthorityError(
            "prepared development datasets differ from reconstruction plan authority"
        )
    return prepared


def _rebuild_reconstruction_plan(
    repository: Path,
    authority: SelectionAuthority,
    status: Mapping[str, Any],
    execution_environment_sha256: str,
):
    from .methods import load_method_registry
    from .protocol import canonical_sha256
    from .runner import (
        DatasetQCPolicy,
        RunnerAuthority,
        _freeze_json_mapping,
        build_competition_plan,
        derive_authorized_configurations,
        validate_development_manifest_payload,
    )

    ledger = _read_authority_json(repository, "study/development_search.json")
    rows = ledger.get("configurations")
    if not isinstance(rows, list):
        raise SelectionAuthorityError("development search configurations are invalid")
    configurations = derive_authorized_configurations(
        rows, authority.ablation_specs, authority.method_bindings
    )
    qc = dict(authority.dataset_qc_policy)
    qc_policy = DatasetQCPolicy(
        cell_exclusion_rule=qc["cell_exclusion_rule"],
        minimum_retained_cells=qc["minimum_retained_cells"],
        application=qc["application"],
        additional_cell_filtering=qc["additional_cell_filtering"],
        gene_filtering=qc["gene_filtering"],
        required_audit_fields=tuple(qc["required_audit_fields"]),
    )
    authority_body = {
        "schema": "maskimpute-development-runner-authority-v1",
        "file_sha256": dict(authority.file_sha256),
        "configurations": [value.to_dict() for value in configurations],
        "base_maskimpute_config_sha256": authority.base_maskimpute_config_sha256,
        "count_model_config_sha256": authority.count_model_config_sha256,
        "dataset_qc_policy_sha256": authority.dataset_qc_policy_sha256,
        "count_score_manifest": {
            "status": authority.count_score_manifest.status,
            "path": authority.count_score_manifest.path,
            "sha256": authority.count_score_manifest.sha256,
        },
        "retained_calibration": {
            "status": authority.retained_calibration.status,
            "path": authority.retained_calibration.path,
            "sha256": authority.retained_calibration.sha256,
        },
        "calibration_effect_status": authority.calibration_effect_status,
        "calibration_equivalence_reason": authority.calibration_equivalence_reason,
    }
    search = [value for value in configurations if value.kind == "candidate_search"]
    runner_authority = RunnerAuthority(
        schema_version=1,
        authority_sha256=canonical_sha256(authority_body),
        method_registry_sha256=authority.file_sha256["study/methods.json"],
        selection_contract_sha256=authority.file_sha256[
            "study/selection_contract.json"
        ],
        development_search_sha256=authority.file_sha256[
            "study/development_search.json"
        ],
        ablation_registry_sha256=authority.file_sha256["study/ablations.json"],
        base_configuration_id=search[0].configuration_id,
        base_configuration_sha256=authority.base_maskimpute_config_sha256,
        base_configuration=_freeze_json_mapping(authority.base_maskimpute_config),
        count_model_config=_freeze_json_mapping(authority.count_model_config),
        count_model_config_sha256=authority.count_model_config_sha256,
        count_score_manifest_status=authority.count_score_manifest.status,
        count_score_manifest_sha256=authority.count_score_manifest.sha256,
        retained_calibration_status=authority.retained_calibration.status,
        retained_calibration_sha256=authority.retained_calibration.sha256,
        dataset_qc_policy=qc_policy,
        dataset_qc_policy_sha256=authority.dataset_qc_policy_sha256,
        count_score_manifest_path=authority.count_score_manifest.path,
        retained_calibration_path=authority.retained_calibration.path,
        configurations=configurations,
    )
    bindings = validate_development_manifest_payload(status)
    registry = load_method_registry(repository / "study/methods.json")
    return build_competition_plan(
        registry,
        bindings,
        runner_authority,
        execution_environment_sha256=execution_environment_sha256,
    )


def _validate_orthogonal_evidence(
    repository: Path,
    raw_orthogonal: object,
    authority: SelectionAuthority | None = None,
    data: Mapping[str, Any] | None = None,
    raw_orthogonal_audits: object = None,
) -> Mapping[str, str]:
    try:
        orthogonal = _exact_authority_mapping(
            raw_orthogonal,
            {
                "manifest_path",
                "manifest_file_sha256",
                "manifest_sha256",
                "records",
            },
            "evaluation orthogonal binding",
        )
    except SelectionAuthorityError as error:
        raise SelectionAuthorityError(
            "evaluation orthogonal binding is invalid"
        ) from error
    expected_path = (
        "artifacts/study/development/evaluation/orthogonal/orthogonal_outputs.json"
    )
    if orthogonal["manifest_path"] != expected_path:
        raise SelectionAuthorityError("evaluation orthogonal binding path is not fixed")
    file_sha256 = _authority_sha(
        orthogonal["manifest_file_sha256"], "orthogonal manifest file checksum"
    )
    manifest, _observed_file_sha256 = _read_canonical_bound_json(
        repository / expected_path, "orthogonal output manifest", file_sha256
    )
    manifest_root = _exact_authority_mapping(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "authority",
            "status",
            "planned_record_count",
            "records",
            "manifest_sha256",
        },
        "orthogonal output manifest",
    )
    raw_authority = _exact_authority_mapping(
        manifest_root["authority"],
        {"inputs", "configurations", "model_seeds", "artifact_bindings"},
        "orthogonal authority",
    )
    inputs = raw_authority["inputs"]
    if authority is None:
        expected_authority = raw_authority
    else:
        if type(inputs) is not list or [
            value.get("source_id") if isinstance(value, Mapping) else None
            for value in inputs
        ] != [
            "cite-seq-cbmc-rna-protein",
            "tung-ipsc-ercc-bulk-replicates",
        ]:
            raise SelectionAuthorityError("orthogonal authority inputs are invalid")
        input_fields = {
            "source_id",
            "source_dataset_sha256",
            "method_input_sha256",
            "shape",
            "cell_ids_sha256",
            "gene_ids_sha256",
        }
        for index, raw_input in enumerate(inputs):
            input_row = _exact_authority_mapping(
                raw_input, input_fields, f"orthogonal authority input {index}"
            )
            for field in (
                "source_dataset_sha256",
                "method_input_sha256",
                "cell_ids_sha256",
                "gene_ids_sha256",
            ):
                _authority_sha(
                    input_row[field], f"orthogonal authority input {index} {field}"
                )
            shape = input_row["shape"]
            if (
                type(shape) is not list
                or len(shape) != 2
                or any(type(value) is not int or value <= 0 for value in shape)
            ):
                raise SelectionAuthorityError(
                    f"orthogonal authority input {index} shape is invalid"
                )
        from .development_evaluation import (
            OrthogonalConfiguration,
            _orthogonal_authority_core,
            prepare_real_orthogonal_panel,
        )
        from .runner import derive_authorized_configurations

        ledger = _read_authority_json(repository, "study/development_search.json")
        configurations = derive_authorized_configurations(
            ledger.get("configurations"),
            authority.ablation_specs,
            authority.method_bindings,
        )
        orthogonal_configurations = tuple(
            OrthogonalConfiguration(
                configuration_id=value.configuration_id,
                configuration_sha256=value.configuration_sha256,
                payload=dict(value.payload),
            )
            for value in configurations
            if value.method_id == "maskimpute" and value.kind == "candidate_search"
        )
        panel = prepare_real_orthogonal_panel(repository)
        expected_authority = _orthogonal_authority_core(
            panel.method_inputs,
            orthogonal_configurations,
            (42, 43, 44),
            {
                "count_model_config_sha256": authority.count_model_config_sha256,
                "retained_calibration_artifact_sha256": (
                    authority.retained_calibration.sha256
                ),
                "score_fit_policy": (
                    "refit_cross_fitted_count_score_from_truth_free_input"
                ),
            },
        )
        if raw_authority != expected_authority:
            raise SelectionAuthorityError(
                "orthogonal authority differs from current selection authority"
            )
    try:
        from .development_evaluation import load_orthogonal_output_evidence

        evidence = load_orthogonal_output_evidence(
            (repository / expected_path).parent,
            expected_authority=expected_authority,
        )
    except Exception as error:
        raise SelectionAuthorityError(
            f"orthogonal output evidence failed byte revalidation: {error}"
        ) from error
    if (
        evidence.manifest_file_sha256 != file_sha256
        or evidence.manifest_sha256 != orthogonal["manifest_sha256"]
        or [dict(value) for value in evidence.records] != orthogonal["records"]
    ):
        raise SelectionAuthorityError("orthogonal manifest binding is invalid")
    if authority is not None:
        try:
            from .development_evaluation import evaluate_real_orthogonal_intervals

            if data is None:
                raise TypeError("selection data is absent")
            independently_recomputed = evaluate_real_orthogonal_intervals(
                evidence,
                panel.cite,
                panel.tung,
                tuple(value.configuration_id for value in authority.attempts),
            )
            expected_intervals = [
                dict(value) for value in independently_recomputed.intervals
            ]
            expected_audits = [dict(value) for value in independently_recomputed.audits]
        except Exception as error:
            raise SelectionAuthorityError(
                "orthogonal intervals and audits could not be independently recomputed"
            ) from error
        if (
            type(data.get("orthogonal_intervals")) is not list
            or _canonical_json_bytes(data["orthogonal_intervals"])
            != _canonical_json_bytes(expected_intervals)
        ):
            raise SelectionAuthorityError(
                "orthogonal intervals differ from independently recomputed outputs"
            )
        if (
            type(raw_orthogonal_audits) is not list
            or _canonical_json_bytes(raw_orthogonal_audits)
            != _canonical_json_bytes(expected_audits)
        ):
            raise SelectionAuthorityError(
                "orthogonal audits differ from independently recomputed outputs"
            )
    return MappingProxyType(
        {
            "orthogonal_manifest_file_sha256": evidence.manifest_file_sha256,
            "orthogonal_manifest_payload_sha256": evidence.manifest_sha256,
            "orthogonal_records_sha256": _canonical_sha256(orthogonal["records"]),
            **(
                {
                    "recomputed_orthogonal_intervals_sha256": _canonical_sha256(
                        expected_intervals
                    ),
                    "recomputed_orthogonal_audits_sha256": _canonical_sha256(
                        expected_audits
                    ),
                }
                if authority is not None
                else {}
            ),
        }
    )


def _validate_evaluator_audits(
    repository: Path,
    evaluation_manifest: Mapping[str, Any],
    data: Mapping[str, Any],
    authority: SelectionAuthority,
) -> Mapping[str, str]:
    null_audits = evaluation_manifest["null_de_audits"]
    orthogonal_audits = evaluation_manifest["orthogonal_audits"]
    null_records = [
        value
        for value in data["records"]
        if isinstance(value, Mapping) and value.get("metric") == "null_de_fpr"
    ]
    if type(null_audits) is not list or len(null_audits) != len(null_records):
        raise SelectionAuthorityError(
            "evaluation null-DE audit denominator is incomplete"
        )
    if type(orthogonal_audits) is not list or len(orthogonal_audits) != len(
        data["orthogonal_intervals"]
    ):
        raise SelectionAuthorityError(
            "evaluation orthogonal audit denominator is incomplete"
        )
    reconstruction = evaluation_manifest["reconstruction"]
    checkpoint, _digest = _read_canonical_bound_json(
        repository / reconstruction["checkpoint_path"],
        "reconstruction checkpoint for audit validation",
        reconstruction["checkpoint_file_sha256"],
    )
    runs: dict[str, Mapping[str, Any]] = {}
    for stored in checkpoint["records"]:
        run = stored.get("run") if isinstance(stored, Mapping) else None
        if not isinstance(run, Mapping) or not isinstance(run.get("run_id"), str):
            raise SelectionAuthorityError(
                "reconstruction checkpoint audit run is invalid"
            )
        if run["run_id"] in runs:
            raise SelectionAuthorityError(
                "reconstruction checkpoint audit run IDs are duplicated"
            )
        runs[run["run_id"]] = run
    null_record_lookup = {
        (
            value["mechanism"],
            value["biological_id"],
            value["technical_view"],
            value["dataset_id"],
            value["method"],
            value["model_seed"],
        ): value
        for value in null_records
    }
    if len(null_record_lookup) != len(null_records):
        raise SelectionAuthorityError("selection null-DE records are duplicated")
    audit_fields = {
        "run_id",
        "dataset_id",
        "method",
        "model_seed",
        "status",
        "value",
        "nominal_alpha",
        "n_tested_genes",
        "fixed_gene_count",
        "split_entropy_sha256",
        "split_entropy_derivation",
        "split_sha256",
        "gene_mask_sha256",
        "reason",
        "evaluator_output_file_sha256",
    }
    candidate_ids = {value.configuration_id for value in authority.attempts}
    declaration_ids = {value.id for value in authority.declarations}
    seen_audit_runs: set[str] = set()
    for index, raw_audit in enumerate(null_audits):
        audit = _exact_authority_mapping(
            raw_audit, audit_fields, f"null-DE audit {index}"
        )
        run_id = audit["run_id"]
        run = runs.get(run_id)
        if run is None or run_id in seen_audit_runs:
            raise SelectionAuthorityError(
                "null-DE audit run binding is absent or duplicated"
            )
        seen_audit_runs.add(run_id)
        method = (
            run.get("configuration_id")
            if run.get("configuration_kind") == "candidate_search"
            and run.get("configuration_id") in candidate_ids
            else run.get("method_id")
        )
        if method not in declaration_ids:
            raise SelectionAuthorityError("null-DE audit method is outside authority")
        identity = (
            run.get("mechanism"),
            run.get("biological_id"),
            run.get("technical_view"),
            run.get("dataset_id"),
            method,
            run.get("model_seed"),
        )
        record = null_record_lookup.get(identity)
        digest = hashlib.sha256()
        digest.update(b"maskimpute-null-de-post-execution-entropy-v1\0")
        digest.update(reconstruction["checkpoint_sha256"].encode("ascii"))
        digest.update(b"\0")
        digest.update(str(run.get("mechanism")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(run.get("biological_id")).encode("utf-8"))
        if (
            record is None
            or audit["dataset_id"] != run.get("dataset_id")
            or audit["method"] != method
            or audit["model_seed"] != run.get("model_seed")
            or audit["status"] != record["status"]
            or audit["value"] != record["value"]
            or audit["nominal_alpha"] != 0.05
            or type(audit["n_tested_genes"]) is not int
            or audit["n_tested_genes"] < 0
            or type(audit["fixed_gene_count"]) is not int
            or audit["fixed_gene_count"] < 0
            or audit["split_entropy_sha256"] != digest.hexdigest()
            or audit["split_entropy_derivation"]
            != "sha256(completed_checkpoint_sha256,mechanism,biological_id)"
            or audit["evaluator_output_file_sha256"]
            != run.get("evaluator_output_file_sha256")
        ):
            raise SelectionAuthorityError(
                "null-DE audit differs from its selection record/run binding"
            )
        for field in (
            "split_entropy_sha256",
            "split_sha256",
            "gene_mask_sha256",
        ):
            _authority_sha(audit[field], f"null-DE audit {index} {field}")

    orthogonal_fields = {
        *_INTERVAL_FIELDS,
        "reason",
        "n_biological_units",
        "n_technical_units",
        "n_boot",
        "bootstrap_sha256",
        "aggregation",
        "inference_scope",
        "profile_scale",
    }
    for index, (raw_audit, interval) in enumerate(
        zip(orthogonal_audits, data["orthogonal_intervals"], strict=True)
    ):
        audit = _exact_authority_mapping(
            raw_audit, orthogonal_fields, f"orthogonal audit {index}"
        )
        if (
            {key: audit[key] for key in _INTERVAL_FIELDS} != interval
            or type(audit["n_biological_units"]) is not int
            or audit["n_biological_units"] < 0
            or type(audit["n_technical_units"]) is not int
            or audit["n_technical_units"] < 0
            or type(audit["n_boot"]) is not int
            or audit["n_boot"] <= 0
            or not isinstance(audit["aggregation"], str)
            or not audit["aggregation"]
            or not isinstance(audit["inference_scope"], str)
            or not audit["inference_scope"]
            or not isinstance(audit["profile_scale"], str)
            or not audit["profile_scale"]
        ):
            raise SelectionAuthorityError(
                "orthogonal audit differs from its selection interval"
            )
        _authority_sha(
            audit["bootstrap_sha256"], f"orthogonal audit {index} bootstrap checksum"
        )
    return MappingProxyType(
        {
            "null_de_audits_sha256": _canonical_sha256(null_audits),
            "orthogonal_audits_sha256": _canonical_sha256(orthogonal_audits),
        }
    )


def _read_authority_json(repository: Path, relative: str) -> dict[str, Any]:
    raw, _digest = _stable_file_bytes(
        repository / relative, f"authority file {relative}"
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationManifestError(
            f"could not parse authority file {relative}"
        ) from error
    if type(payload) is not dict:
        raise EvaluationManifestError(f"authority file {relative} must be an object")
    return payload


@dataclass(frozen=True, slots=True)
class ValidatedEvaluationEvidence:
    manifest: Mapping[str, Any]
    bindings: Mapping[str, str]


def validate_selection_evaluation_manifest(
    repository: Path,
    data: Mapping[str, Any],
    authority: object,
    status: Mapping[str, Any],
) -> ValidatedEvaluationEvidence:
    try:
        manifest, envelope = _validate_selection_evaluation_envelope(repository, data)
        sources = _validate_evaluation_source_evidence(repository, manifest["sources"])
        reconstruction = _validate_reconstruction_evidence(
            repository,
            manifest["reconstruction"],
            authority,
            status,
            data,
            manifest["null_de_audits"],
        )
        orthogonal = _validate_orthogonal_evidence(
            repository,
            manifest["orthogonal"],
            authority,
            data,
            manifest["orthogonal_audits"],
        )
        audits = _validate_evaluator_audits(repository, manifest, data, authority)
    except EvaluationManifestError:
        raise
    except Exception as error:
        raise EvaluationManifestError(
            f"evaluation evidence semantic validation failed: {type(error).__name__}"
        ) from error
    return ValidatedEvaluationEvidence(
        manifest=MappingProxyType(dict(manifest)),
        bindings=MappingProxyType(
            {
                **dict(envelope),
                **dict(sources),
                **dict(reconstruction),
                **dict(orthogonal),
                **dict(audits),
            }
        ),
    )


__all__ = [
    "EvaluationManifestError",
    "ValidatedEvaluationEvidence",
    "validate_selection_evaluation_manifest",
]
