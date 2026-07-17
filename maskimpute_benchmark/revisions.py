"""Conditional development-revision authority and fixed evidence paths.

Revision specifications are tracked before their triggering results exist.  A
specification becomes selection authority only when the preceding fixed input
and report revalidate and emit the exact trigger named by the specification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_VERSIONS = ("v28", "v29")
_EVALUATION_ROOT = "artifacts/study/development/evaluation"


class RevisionAuthorityError(RuntimeError):
    """Raised when a conditional revision is absent, altered, or untriggered."""


@dataclass(frozen=True, slots=True)
class RevisionStagePaths:
    version: str
    revision_authority: str
    activation_selection_input: str
    activation_selection_report: str
    reconstruction_directory: str
    orthogonal_directory: str
    evaluation_manifest: str
    selection_input: str
    selection_report: str


@dataclass(frozen=True, slots=True)
class RevisionSpec:
    version: str
    trigger: str
    parent_configuration_id: str
    parent_configuration_sha256: str
    configuration_id: str
    configuration: Mapping[str, Any]
    configuration_sha256: str
    reason_code: str
    relative_path: str
    file_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.configuration, Mapping):
            raise TypeError("revision configuration must be a mapping")
        object.__setattr__(self, "configuration", _freeze_json(self.configuration))


@dataclass(frozen=True, slots=True)
class RevisionActivation:
    version: str
    trigger: str
    selection_input_path: str
    selection_input_file_sha256: str
    selection_result_sha256: str
    selection_report_path: str
    selection_report_file_sha256: str

    def __post_init__(self) -> None:
        if self.version not in _VERSIONS:
            raise ValueError("revision activation version is invalid")
        if self.trigger != self.version:
            raise ValueError("revision activation trigger must equal its version")
        paths = revision_stage_paths(self.version)
        if (
            self.selection_input_path != paths.activation_selection_input
            or self.selection_report_path != paths.activation_selection_report
        ):
            raise ValueError("revision activation paths are not fixed")
        for value in (
            self.selection_input_file_sha256,
            self.selection_result_sha256,
            self.selection_report_file_sha256,
        ):
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError("revision activation checksum is invalid")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


def thaw_revision_configuration(spec: RevisionSpec) -> dict[str, object]:
    """Return a detached JSON value for execution without weakening authority."""

    if type(spec) is not RevisionSpec:
        raise TypeError("spec must be an exact RevisionSpec")
    value = _thaw_json(spec.configuration)
    assert isinstance(value, dict)
    return value


def revision_stage_paths(version: str) -> RevisionStagePaths:
    """Return the closed repository-relative paths for one revision stage."""

    if version not in _VERSIONS:
        raise ValueError("revision version must be v28 or v29")
    prior = None if version == "v28" else "v28"
    activation_suffix = "" if prior is None else f"-{prior}"
    suffix = f"-{version}"
    return RevisionStagePaths(
        version=version,
        revision_authority=f"study/{version}_revision.json",
        activation_selection_input=(
            f"{_EVALUATION_ROOT}/development_selection_input{activation_suffix}.json"
        ),
        activation_selection_report=(
            f"{_EVALUATION_ROOT}/development_selection_report{activation_suffix}.json"
        ),
        reconstruction_directory=(
            f"artifacts/study/development/competition-{version}-revision"
        ),
        orthogonal_directory=f"{_EVALUATION_ROOT}/orthogonal-{version}-revision",
        evaluation_manifest=f"{_EVALUATION_ROOT}/evaluation_manifest{suffix}.json",
        selection_input=f"{_EVALUATION_ROOT}/development_selection_input{suffix}.json",
        selection_report=f"{_EVALUATION_ROOT}/development_selection_report{suffix}.json",
    )


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RevisionAuthorityError("revision contains noncanonical JSON") from error
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RevisionAuthorityError(f"duplicate revision JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RevisionAuthorityError(f"nonfinite revision JSON constant {value}")


def _read_stable_bytes(path: Path, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RevisionAuthorityError(f"fixed {name} is absent or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o002
        ):
            raise RevisionAuthorityError(f"fixed {name} is absent or unsafe")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_nlink,
        before.st_uid,
        before.st_gid,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_nlink,
        after.st_uid,
        after.st_gid,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) != before.st_size:
        raise RevisionAuthorityError(f"fixed {name} changed while it was read")
    return raw


def _read_canonical_json(path: Path, name: str, *, indented: bool) -> tuple[dict[str, Any], str]:
    try:
        raw = _read_stable_bytes(path, name)
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except RevisionAuthorityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RevisionAuthorityError(f"fixed {name} is invalid") from error
    if type(payload) is not dict:
        raise RevisionAuthorityError(f"fixed {name} must be a JSON object")
    expected = (
        json.dumps(payload, indent=2).encode("utf-8") + b"\n"
        if indented
        else json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if raw != expected:
        raise RevisionAuthorityError(f"fixed {name} is not canonical JSON")
    return payload, hashlib.sha256(raw).hexdigest()


def _assert_tracked_clean(repository: Path, relative: str) -> None:
    try:
        subprocess.run(
            ["git", "-C", str(repository), "ls-files", "--error-unmatch", "--", relative],
            check=True,
            capture_output=True,
            timeout=15,
        )
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RevisionAuthorityError("revision authority must be tracked by Git") from error
    if dirty:
        raise RevisionAuthorityError("revision authority differs from the Git index")


def load_revision_spec(
    repository: Path,
    version: str,
    *,
    require_clean: bool = True,
) -> RevisionSpec:
    """Load and semantically validate one prespecified tracked revision."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.resolve(strict=True)
    paths = revision_stage_paths(version)
    if require_clean:
        _assert_tracked_clean(root, paths.revision_authority)
    payload, file_sha256 = _read_canonical_json(
        _safe_repository_path(
            root,
            paths.revision_authority,
            f"{version} revision authority",
        ),
        f"{version} revision authority",
        indented=True,
    )
    fields = {
        "schema_version",
        "status",
        "trigger",
        "parent_configuration_id",
        "parent_configuration_sha256",
        "configuration_id",
        "configuration",
        "configuration_sha256",
        "reason_code",
    }
    if set(payload) != fields:
        raise RevisionAuthorityError(f"{version} revision fields differ")
    expected_status = f"conditional_on_{version}_trigger"
    expected_reason = (
        "prespecified_decoder_only_revision_of_v27_c03"
        if version == "v28"
        else "prespecified_structure_revision_of_v28_c01"
    )
    configuration = payload["configuration"]
    if (
        payload["schema_version"] != 1
        or type(payload["schema_version"]) is not int
        or payload["status"] != expected_status
        or payload["trigger"] != version
        or payload["reason_code"] != expected_reason
        or type(configuration) is not dict
    ):
        raise RevisionAuthorityError(f"{version} revision contract differs")
    for field in (
        "parent_configuration_id",
        "configuration_id",
    ):
        if not isinstance(payload[field], str) or not _SAFE_ID.fullmatch(payload[field]):
            raise RevisionAuthorityError(f"{version} revision identity is invalid")
    for field in ("parent_configuration_sha256", "configuration_sha256"):
        if not isinstance(payload[field], str) or not _SHA256.fullmatch(payload[field]):
            raise RevisionAuthorityError(f"{version} revision checksum is invalid")
    if _canonical_sha256(configuration) != payload["configuration_sha256"]:
        raise RevisionAuthorityError(f"{version} configuration checksum differs")
    expected_configuration_fields = {
        "method_version",
        "decoder",
        "encoder_mode",
        "output_policy",
        "score_policy",
        "hyperparameters",
        "decoder_hyperparameters",
    }
    if version == "v29":
        expected_configuration_fields.add("structure_hyperparameters")
    if (
        set(configuration) != expected_configuration_fields
        or configuration["method_version"] != version
        or configuration["decoder"] != "negative_binomial"
        or configuration["encoder_mode"] != "explicit_mask"
        or configuration["output_policy"] != "selective"
        or configuration["score_policy"] != "retained_development_calibrator"
    ):
        raise RevisionAuthorityError(f"{version} configuration semantics differ")
    return RevisionSpec(
        version=version,
        trigger=version,
        parent_configuration_id=payload["parent_configuration_id"],
        parent_configuration_sha256=payload["parent_configuration_sha256"],
        configuration_id=payload["configuration_id"],
        configuration=configuration,
        configuration_sha256=payload["configuration_sha256"],
        reason_code=payload["reason_code"],
        relative_path=paths.revision_authority,
        file_sha256=file_sha256,
    )


def derive_extended_selection_authority(
    base: object,
    specs: Sequence[RevisionSpec],
    activations: Sequence[RevisionActivation],
):
    """Extend base selection authority only through consecutively triggered stages."""

    from .selection import (
        CandidateAttempt,
        MethodDeclaration,
        SelectionAuthority,
        _validate_attempts,
        _validate_declarations,
    )

    if type(base) is not SelectionAuthority:
        raise TypeError("base must be an exact SelectionAuthority")
    spec_values = tuple(specs)
    activation_values = tuple(activations)
    if len(spec_values) not in {1, 2}:
        raise RevisionAuthorityError(
            "revision denominator must contain v28 or consecutive v28 and v29"
        )
    if len(spec_values) != len(activation_values):
        missing = spec_values[len(activation_values)].version if spec_values else "revision"
        raise RevisionAuthorityError(f"{missing} activation is absent")
    attempts = list(base.attempts)
    declarations = list(base.declarations)
    bindings = dict(base.method_bindings)
    files = dict(base.file_sha256)
    previous_configuration = {
        attempt.configuration_id: bindings[attempt.configuration_id]
        for attempt in base.attempts
    }
    for expected_version, spec, activation in zip(
        _VERSIONS,
        spec_values,
        activation_values,
        strict=False,
    ):
        if spec.version != expected_version or activation.version != expected_version:
            raise RevisionAuthorityError("revision stages are reordered")
        if activation.trigger != spec.trigger:
            raise RevisionAuthorityError(
                f"{spec.version} activation trigger does not authorize the revision"
            )
        parent_sha = previous_configuration.get(spec.parent_configuration_id)
        if parent_sha != spec.parent_configuration_sha256:
            raise RevisionAuthorityError(
                f"{spec.version} revision parent binding differs"
            )
        if spec.configuration_id in bindings:
            raise RevisionAuthorityError("revision configuration identity collides")
        attempt = CandidateAttempt(
            configuration_id=spec.configuration_id,
            version=spec.version,
            parent_configuration_id=spec.parent_configuration_id,
        )
        attempts.append(attempt)
        declarations.append(
            MethodDeclaration(
                id=spec.configuration_id,
                role="candidate",
                track="same_input",
                stochastic=True,
                required_for_claim=True,
            )
        )
        bindings[spec.configuration_id] = spec.configuration_sha256
        previous_configuration[spec.configuration_id] = spec.configuration_sha256
        files[spec.relative_path] = spec.file_sha256
    attempt_values = _validate_attempts(tuple(attempts))
    declaration_values = _validate_declarations(tuple(declarations), attempt_values)
    return SelectionAuthority(
        mechanisms=base.mechanisms,
        biological_ids=base.biological_ids,
        technical_views=base.technical_views,
        model_seeds=base.model_seeds,
        required_comparator_ids=base.required_comparator_ids,
        attempts=attempt_values,
        declarations=declaration_values,
        endpoint_policies=base.endpoint_policies,
        revision_policy=base.revision_policy,
        exclusions=base.exclusions,
        method_bindings=MappingProxyType(bindings),
        base_maskimpute_config=base.base_maskimpute_config,
        base_maskimpute_config_sha256=base.base_maskimpute_config_sha256,
        count_model_config=base.count_model_config,
        count_model_config_sha256=base.count_model_config_sha256,
        dataset_qc_policy=base.dataset_qc_policy,
        dataset_qc_policy_sha256=base.dataset_qc_policy_sha256,
        ablation_specs=base.ablation_specs,
        ablation_spec_ids=base.ablation_spec_ids,
        ablation_run_keys=base.ablation_run_keys,
        calibration_equivalence_reason=base.calibration_equivalence_reason,
        calibration_effect_status=base.calibration_effect_status,
        retained_calibration=base.retained_calibration,
        count_score_manifest=base.count_score_manifest,
        file_sha256=MappingProxyType(files),
    )


def _safe_repository_path(repository: Path, relative_value: str, name: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RevisionAuthorityError(f"fixed {name} path is unsafe")
    path = repository.joinpath(*relative.parts)
    current = path
    while current != repository.parent:
        if current.exists() and current.is_symlink():
            raise RevisionAuthorityError(f"fixed {name} path contains a symlink")
        if current == repository:
            break
        current = current.parent
    return path


def _activation_gate_passed(
    assessment: Mapping[str, object],
    gate_name: str,
) -> bool | None:
    gates = assessment.get("gates")
    if not isinstance(gates, Mapping):
        return None
    gate = gates.get(gate_name)
    if not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool:
        return None
    return bool(gate["passed"])


def _validate_activation_denominator(
    selection_report: Mapping[str, object],
    spec: RevisionSpec,
) -> None:
    assessments = selection_report.get("assessments")
    if not isinstance(assessments, list) or not assessments:
        raise RevisionAuthorityError("activation assessment denominator is absent")
    typed: list[Mapping[str, object]] = []
    for index, assessment in enumerate(assessments):
        if not isinstance(assessment, Mapping):
            raise RevisionAuthorityError(
                f"activation assessment {index} is malformed"
            )
        typed.append(assessment)
        if _activation_gate_passed(
            assessment, "required_comparator_completeness"
        ) is not True:
            raise RevisionAuthorityError(
                "activation comparator denominator is incomplete"
            )
        if _activation_gate_passed(assessment, "candidate_completeness") is not True:
            raise RevisionAuthorityError(
                "activation candidate denominator is incomplete"
            )
    parents = [
        assessment
        for assessment in typed
        if assessment.get("configuration_id") == spec.parent_configuration_id
    ]
    expected_parent_version = "v27" if spec.version == "v28" else "v28"
    if (
        len(parents) != 1
        or parents[0].get("version") != expected_parent_version
    ):
        raise RevisionAuthorityError(
            f"{spec.version} activation lacks its exact assessed parent"
        )
    if spec.version == "v29":
        parent = parents[0]
        structure_failure = (
            parent.get("efficacy_pass") is True
            and (
                _activation_gate_passed(parent, "corr_err_degradation") is False
                or _activation_gate_passed(parent, "orthogonal_safety") is False
            )
        )
        if not structure_failure:
            raise RevisionAuthorityError(
                "v29 exact v28 parent does not establish a structure failure"
            )


def validate_revision_activation(
    repository: Path,
    version: str,
    *,
    require_clean: bool = True,
) -> RevisionActivation:
    """Recompute the preceding fixed selection and require the revision trigger."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.resolve(strict=True)
    paths = revision_stage_paths(version)
    selection_input, input_sha = _read_canonical_json(
        _safe_repository_path(
            root, paths.activation_selection_input, "revision activation selection input"
        ),
        "revision activation selection input",
        indented=False,
    )
    selection_report, report_sha = _read_canonical_json(
        _safe_repository_path(
            root, paths.activation_selection_report, "revision activation selection report"
        ),
        "revision activation selection report",
        indented=False,
    )
    from .selection import _select_for_repository

    try:
        recomputed = _select_for_repository(
            selection_input,
            root,
            require_clean=require_clean,
        ).to_dict()
    except Exception as error:
        raise RevisionAuthorityError(
            f"{version} activation selection input failed revalidation"
        ) from error
    if selection_report != recomputed:
        raise RevisionAuthorityError(
            f"{version} activation report differs from recomputed selection"
        )
    if (
        selection_report.get("trigger") != version
        or selection_report.get("selected_configuration") is not None
    ):
        raise RevisionAuthorityError(
            f"{version} activation report trigger does not authorize the revision"
        )
    spec = load_revision_spec(root, version, require_clean=require_clean)
    _validate_activation_denominator(selection_report, spec)
    result_sha = selection_input.get("result_sha256")
    if not isinstance(result_sha, str) or not _SHA256.fullmatch(result_sha):
        raise RevisionAuthorityError("activation selection result checksum is invalid")
    return RevisionActivation(
        version=version,
        trigger=version,
        selection_input_path=paths.activation_selection_input,
        selection_input_file_sha256=input_sha,
        selection_result_sha256=result_sha,
        selection_report_path=paths.activation_selection_report,
        selection_report_file_sha256=report_sha,
    )


__all__ = [
    "RevisionActivation",
    "RevisionAuthorityError",
    "RevisionSpec",
    "RevisionStagePaths",
    "derive_extended_selection_authority",
    "load_revision_spec",
    "revision_stage_paths",
    "thaw_revision_configuration",
    "validate_revision_activation",
]
