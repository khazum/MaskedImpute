"""Fail-closed assembly of the method/configuration frozen for final evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from types import MappingProxyType
from typing import Any

from .external_reference_development import (
    ExternalReferenceDevelopmentError,
    load_external_reference_evidence,
)
from .methods.base import MethodContractError
from .methods.registry import parse_method_registry
from .protocol import canonical_sha256
from .runtime_environments import (
    RuntimeEnvironmentError,
    load_runtime_environment_lock,
)
from .revisions import (
    RevisionActivation,
    RevisionAuthorityError,
    development_selection_stage_paths,
    load_revision_spec,
    revision_stage_paths,
    thaw_revision_configuration,
    validate_revision_activation,
)
from .selection import (
    SelectionAuthorityError,
    _validate_schema_four_source_projection,
    _validate_selection_source_payload,
    validate_downstream_selection_completeness,
)
from .selection_promotion import (
    SelectionPromotionError,
    _secure_canonical_json as _secure_selection_json,
)
from .study import (
    StudyStateError,
    _git,
    _operational_root_receipts,
    _raw_tracked_files_match_index,
    freeze_round,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_REASON_CODE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
_NONFINAL_REASON_MARKERS = ("pending", "not_yet", "unverified")
_RUN_STATUSES = frozenset(
    {
        "completed",
        "unavailable",
        "failed",
        "timeout",
        "resource_exceeded",
        "infrastructure_error",
        "blocked_authority",
        "budget_exhausted",
    }
)
_DISPOSITION_FAILURE_STATUSES = frozenset(
    {"unavailable", "failed", "timeout", "resource_exceeded"}
)

_COMMON_TRACKED_PATHS = {
    "runtime_lock": "environments/development-runtime.lock.json",
    "method_registry": "study/methods.json",
    "selection_contract": "study/selection_contract.json",
    "development_search": "study/development_search.json",
    "v28_revision": "study/v28_revision.json",
    "v29_revision": "study/v29_revision.json",
    "ablation_registry": "study/ablations.json",
    "scaling_panel": "study/scaling_panel.json",
    "trajectory_panel": "study/trajectory_panel.json",
    "protocol": "study/protocol.json",
    "saver_qualification": "environments/saver-r.qualification.json",
    "saver_package_lock": "environments/saver-r.lock.json",
    "saver_build_receipt": "environments/saver-r.build-receipt.json",
}
_COMMON_DEVELOPMENT_PATHS = {
    "dataset_status": "artifacts/study/development/results/dataset_status.json",
    "count_score_manifest": "artifacts/study/development/count_scores/manifest.json",
    "retained_calibration": (
        "artifacts/study/development/calibration/retained_calibration.json"
    ),
}

_FROZEN_METHOD_PATH = "study/frozen_method.json"
_EXTERNAL_REFERENCE_CHECKPOINT_PATH = (
    "artifacts/study/development/competition-external-reference/checkpoint.json"
)
_PUBLICATION_OPERATIONAL_ROOTS = (
    "artifacts/envs",
    "artifacts/external",
    "artifacts/method-sources",
    "artifacts/study/development",
)


class PublicationFreezeError(ValueError):
    """Raised when development evidence cannot authorize a publication freeze."""


@dataclass(frozen=True, slots=True)
class PublicationStagePaths:
    """Closed paths owned by one publication-development stage."""

    stage: str
    source_input: str
    complete_input: str
    report: str
    downstream_directory: str
    downstream_plan: str
    downstream_manifest: str
    evaluation_manifest: str
    reconstruction_directory: str
    reconstruction_checkpoint: str
    orthogonal_directory: str
    orthogonal_manifest: str
    revision_authority: str | None
    activation_selection_input: str | None
    activation_selection_report: str | None


@dataclass(frozen=True, slots=True)
class PublicationStageLayout:
    """The exact contiguous publication-development prefix present on disk."""

    active_stage: str
    revision_versions: tuple[str, ...]
    stages: tuple[PublicationStagePaths, ...]


@dataclass(frozen=True, slots=True)
class ValidatedPublicationStageEvidence:
    """Stable semantic and byte receipts retained for one validated stage."""

    stage: str
    source_input: Mapping[str, object]
    source_input_file_sha256: str
    source_result_sha256: str
    complete_input: Mapping[str, object]
    complete_input_file_sha256: str
    complete_result_sha256: str
    report: Mapping[str, object]
    report_file_sha256: str
    downstream_manifest_file_sha256: str
    downstream_manifest_sha256: str
    downstream_plan_sha256: str
    activation: RevisionActivation | None


@dataclass(frozen=True, slots=True)
class PublicationStageEvidenceReceipt:
    """Immutable exact evidence retained for the resolved development prefix."""

    active_stage: str
    revision_versions: tuple[str, ...]
    stages: tuple[ValidatedPublicationStageEvidence, ...]


def _publication_stage_paths(stage: str) -> PublicationStagePaths:
    if stage not in {"base", "v28", "v29"}:
        raise PublicationFreezeError(f"unknown publication stage suffix: {stage}")
    through_version = None if stage == "base" else stage
    selection = development_selection_stage_paths(through_version)
    if stage == "base":
        evaluation_manifest = (
            "artifacts/study/development/evaluation/evaluation_manifest.json"
        )
        reconstruction_directory = (
            "artifacts/study/development/competition-reconstruction"
        )
        orthogonal_directory = "artifacts/study/development/evaluation/orthogonal"
        revision_authority = None
        activation_selection_input = None
        activation_selection_report = None
    else:
        revision = revision_stage_paths(stage)
        evaluation_manifest = revision.evaluation_manifest
        reconstruction_directory = revision.reconstruction_directory
        orthogonal_directory = revision.orthogonal_directory
        revision_authority = revision.revision_authority
        activation_selection_input = revision.activation_selection_input
        activation_selection_report = revision.activation_selection_report
    return PublicationStagePaths(
        stage=stage,
        source_input=selection.source_selection_input,
        complete_input=selection.selection_complete_input,
        report=selection.selection_report,
        downstream_directory=selection.downstream_directory,
        downstream_plan=f"{selection.downstream_directory}/plan.json",
        downstream_manifest=(
            f"{selection.downstream_directory}/downstream_manifest.json"
        ),
        evaluation_manifest=evaluation_manifest,
        reconstruction_directory=reconstruction_directory,
        reconstruction_checkpoint=f"{reconstruction_directory}/checkpoint.json",
        orthogonal_directory=orthogonal_directory,
        orthogonal_manifest=f"{orthogonal_directory}/orthogonal_outputs.json",
        revision_authority=revision_authority,
        activation_selection_input=activation_selection_input,
        activation_selection_report=activation_selection_report,
    )


def _publication_stage_footprint(stage: PublicationStagePaths) -> tuple[str, ...]:
    return (
        stage.source_input,
        stage.complete_input,
        stage.report,
        stage.downstream_directory,
        stage.downstream_plan,
        stage.downstream_manifest,
        stage.evaluation_manifest,
        stage.reconstruction_directory,
        stage.reconstruction_checkpoint,
        stage.orthogonal_directory,
        stage.orthogonal_manifest,
    )


def _publication_layout_for_active_stage(active_stage: str) -> PublicationStageLayout:
    order = {
        "base": ("base",),
        "v28": ("base", "v28"),
        "v29": ("base", "v28", "v29"),
    }.get(active_stage)
    if order is None:
        raise PublicationFreezeError(
            "development stage receipt active stage is invalid"
        )
    return PublicationStageLayout(
        active_stage=active_stage,
        revision_versions=tuple(stage for stage in order if stage != "base"),
        stages=tuple(_publication_stage_paths(stage) for stage in order),
    )


def _publication_artifact_paths(
    layout: PublicationStageLayout,
    *,
    include_external_reference: bool = False,
) -> dict[str, str]:
    """Return the exact common and stage-qualified publication inventory."""

    if type(layout) is not PublicationStageLayout:
        raise TypeError("layout must be an exact PublicationStageLayout")
    if type(include_external_reference) is not bool:
        raise TypeError("include_external_reference must be a bool")
    result = {**_COMMON_TRACKED_PATHS, **_COMMON_DEVELOPMENT_PATHS}
    for stage in layout.stages:
        prefix = stage.stage
        result.update(
            {
                f"{prefix}_selection_source_input": stage.source_input,
                f"{prefix}_selection_complete_input": stage.complete_input,
                f"{prefix}_selection_report": stage.report,
                f"{prefix}_evaluation_manifest": stage.evaluation_manifest,
                f"{prefix}_reconstruction_checkpoint": (
                    stage.reconstruction_checkpoint
                ),
                f"{prefix}_orthogonal_manifest": stage.orthogonal_manifest,
                f"{prefix}_downstream_plan": stage.downstream_plan,
                f"{prefix}_downstream_manifest": stage.downstream_manifest,
            }
        )
    if include_external_reference:
        result["external_reference_checkpoint"] = _EXTERNAL_REFERENCE_CHECKPOINT_PATH
    if len(result) != len(set(result.values())):
        raise PublicationFreezeError(
            "publication artifact inventory contains duplicate fixed paths"
        )
    return result


def _safe_stage_directory_entries(repository: Path, relative: str) -> tuple[str, ...]:
    """List one generated-stage directory without following a symlink."""

    path = repository / relative
    if not os.path.lexists(path):
        return ()
    descriptor = -1
    try:
        with _pinned_parent(path, "publication stage directory") as parent:
            named_before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(named_before.st_mode):
                raise PublicationFreezeError(
                    f"publication stage directory is unsafe: {relative}"
                )
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            opened_before = os.fstat(descriptor)
            if _directory_identity(opened_before) != _directory_identity(named_before):
                raise PublicationFreezeError(
                    f"publication stage directory changed while opened: {relative}"
                )
            entries = tuple(sorted(os.listdir(descriptor)))
            opened_after = os.fstat(descriptor)
            named_after = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if _directory_identity(opened_before) != _directory_identity(
                opened_after
            ) or _directory_identity(opened_before) != _directory_identity(named_after):
                raise PublicationFreezeError(
                    f"publication stage directory changed during access: {relative}"
                )
            return entries
    except PublicationFreezeError:
        raise
    except OSError as error:
        raise PublicationFreezeError(
            f"publication stage directory is unavailable or unsafe: {relative}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _reject_unknown_publication_stages(repository: Path) -> None:
    known = {"v28", "v29"}
    evaluation_entries = _safe_stage_directory_entries(
        repository, "artifacts/study/development/evaluation"
    )
    exact_base_names = {
        "development_selection_input.json",
        "development_selection_input-downstream.json",
        "development_selection_report.json",
        "evaluation_manifest.json",
        "downstream",
        "orthogonal",
    }
    prefixes = (
        ("development_selection_input-", ".json", "-downstream"),
        ("development_selection_report-", ".json", ""),
        ("evaluation_manifest-", ".json", ""),
        ("downstream-", "", ""),
        ("orthogonal-", "-revision", ""),
    )
    for name in evaluation_entries:
        if name in exact_base_names:
            continue
        for prefix, suffix, removable_suffix in prefixes:
            if not name.startswith(prefix) or (suffix and not name.endswith(suffix)):
                continue
            version = name[len(prefix) :]
            if suffix:
                version = version[: -len(suffix)]
            if removable_suffix and version.endswith(removable_suffix):
                version = version[: -len(removable_suffix)]
            if version not in known:
                raise PublicationFreezeError(
                    f"unknown publication stage suffix in generated footprint: {name}"
                )
            break

    development_entries = _safe_stage_directory_entries(
        repository, "artifacts/study/development"
    )
    for name in development_entries:
        match = re.fullmatch(r"competition-(.+)-revision", name)
        if match is not None and match.group(1) not in known:
            raise PublicationFreezeError(
                f"unknown publication stage suffix in generated footprint: {name}"
            )


def _validate_publication_stage_path(
    repository: Path,
    stage: str,
    relative: str,
    *,
    directory: bool,
) -> None:
    current = repository
    for index, component in enumerate(PurePosixPath(relative).parts):
        current /= component
        try:
            value = os.lstat(current)
        except FileNotFoundError as error:
            raise PublicationFreezeError(
                f"{stage} publication stage is incomplete: missing {relative}"
            ) from error
        except OSError as error:
            raise PublicationFreezeError(
                f"{stage} publication stage path is unsafe: {relative}"
            ) from error
        if stat.S_ISLNK(value.st_mode):
            raise PublicationFreezeError(
                f"{stage} publication stage contains an unsafe symlink: {relative}"
            )
        final = index == len(PurePosixPath(relative).parts) - 1
        expected_directory = directory if final else True
        if expected_directory and not stat.S_ISDIR(value.st_mode):
            raise PublicationFreezeError(
                f"{stage} publication stage path is unsafe: {relative}"
            )
        if not expected_directory and (
            not stat.S_ISREG(value.st_mode) or value.st_nlink != 1
        ):
            raise PublicationFreezeError(
                f"{stage} publication stage file is unsafe: {relative}"
            )


def _resolve_publication_stage(repository: Path) -> PublicationStageLayout:
    """Resolve and validate the newest exact base/v28/v29 generated prefix."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = Path(os.path.abspath(repository))
    try:
        root_status = os.lstat(root)
    except OSError as error:
        raise PublicationFreezeError("publication repository is unavailable") from error
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise PublicationFreezeError("publication repository is unsafe")

    _reject_unknown_publication_stages(root)
    candidates = {
        stage: _publication_stage_paths(stage) for stage in ("base", "v28", "v29")
    }
    active_stage = "base"
    for stage in ("v29", "v28"):
        if any(
            os.path.lexists(root / relative)
            for relative in _publication_stage_footprint(candidates[stage])
        ):
            active_stage = stage
            break
    order = tuple(
        stage.stage
        for stage in _publication_layout_for_active_stage(active_stage).stages
    )
    directory_fields = {
        "downstream_directory",
        "reconstruction_directory",
        "orthogonal_directory",
    }
    for stage_name in order:
        paths = candidates[stage_name]
        for field in (
            "source_input",
            "complete_input",
            "report",
            "downstream_directory",
            "downstream_plan",
            "downstream_manifest",
            "evaluation_manifest",
            "reconstruction_directory",
            "reconstruction_checkpoint",
            "orthogonal_directory",
            "orthogonal_manifest",
        ):
            _validate_publication_stage_path(
                root,
                stage_name,
                getattr(paths, field),
                directory=field in directory_fields,
            )
    resolved = _publication_layout_for_active_stage(active_stage)
    return PublicationStageLayout(
        active_stage=resolved.active_stage,
        revision_versions=resolved.revision_versions,
        stages=tuple(candidates[stage] for stage in order),
    )


def _freeze_receipt_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_receipt_json(nested) for key, nested in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_receipt_json(nested) for nested in value)
    return value


def _thaw_receipt_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_receipt_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_receipt_json(nested) for nested in value]
    return value


def _read_publication_stage_core(
    repository: Path,
    paths: PublicationStagePaths,
) -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    try:
        source, source_file_sha256 = _secure_selection_json(
            repository / paths.source_input,
            f"{paths.stage} source selection input",
        )
        complete, complete_file_sha256 = _secure_selection_json(
            repository / paths.complete_input,
            f"{paths.stage} selection-complete input",
        )
        report, report_file_sha256 = _secure_selection_json(
            repository / paths.report,
            f"{paths.stage} selection report",
        )
        downstream_manifest, downstream_manifest_file_sha256 = _secure_selection_json(
            repository / paths.downstream_manifest,
            f"{paths.stage} downstream manifest",
        )
    except SelectionPromotionError as error:
        raise PublicationFreezeError(
            f"{paths.stage} publication stage core evidence is unsafe or noncanonical"
        ) from error
    return (
        source,
        source_file_sha256,
        complete,
        complete_file_sha256,
        report,
        report_file_sha256,
        downstream_manifest,
        downstream_manifest_file_sha256,
    )


def _validate_publication_stage_evidence(
    repository: Path,
    layout: PublicationStageLayout,
) -> tuple[PublicationStageEvidenceReceipt, dict[str, object]]:
    """Replay the exact schema-4 selection and activation prefix without fallback."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if type(layout) is not PublicationStageLayout:
        raise TypeError("layout must be an exact PublicationStageLayout")
    root = Path(os.path.abspath(repository))
    if _resolve_publication_stage(root) != layout:
        raise PublicationFreezeError(
            "publication stage changed before selection replay"
        )

    expected_complete_fields = {
        "schema_version",
        "revision_versions",
        "dataset_manifest_sha256",
        "count_score_manifest_sha256",
        "retained_calibration_artifact_sha256",
        "evaluation_manifest_sha256",
        "comparator_selection",
        "records",
        "orthogonal_intervals",
        "downstream_evidence",
        "result_sha256",
    }
    retained: list[ValidatedPublicationStageEvidence] = []
    snapshots: list[
        tuple[
            PublicationStagePaths,
            dict[str, Any],
            str,
            dict[str, Any],
            str,
            dict[str, Any],
            str,
            dict[str, Any],
            str,
        ]
    ] = []

    for index, paths in enumerate(layout.stages):
        (
            source,
            source_file_sha256,
            complete,
            complete_file_sha256,
            report,
            report_file_sha256,
            downstream_manifest,
            downstream_manifest_file_sha256,
        ) = _read_publication_stage_core(root, paths)
        snapshots.append(
            (
                paths,
                source,
                source_file_sha256,
                complete,
                complete_file_sha256,
                report,
                report_file_sha256,
                downstream_manifest,
                downstream_manifest_file_sha256,
            )
        )

        expected_versions = list(layout.revision_versions[:index])
        expected_source_stage = None if paths.stage == "base" else paths.stage
        try:
            source_stage = _validate_selection_source_payload(source)
        except SelectionAuthorityError as error:
            raise PublicationFreezeError(
                f"{paths.stage} source selection input is invalid: {error}"
            ) from error
        if source_stage != expected_source_stage:
            raise PublicationFreezeError(
                f"{paths.stage} source selection input stage differs"
            )
        source_result_sha256 = _sha256(
            source.get("result_sha256"),
            f"{paths.stage} source selection result checksum",
        )
        if (
            set(complete) != expected_complete_fields
            or complete.get("schema_version") != 4
            or type(complete.get("schema_version")) is not int
            or complete.get("revision_versions") != expected_versions
        ):
            raise PublicationFreezeError(
                f"{paths.stage} selection-complete input is not exact schema 4"
            )
        complete_result_sha256 = _sha256(
            complete.get("result_sha256"),
            f"{paths.stage} selection-complete result checksum",
        )
        complete_body = {
            key: value for key, value in complete.items() if key != "result_sha256"
        }
        if canonical_sha256(complete_body) != complete_result_sha256:
            raise PublicationFreezeError(
                f"{paths.stage} selection-complete result checksum mismatch"
            )
        downstream_binding = complete.get("downstream_evidence")
        if not isinstance(downstream_binding, Mapping):
            raise PublicationFreezeError(
                f"{paths.stage} selection-complete downstream binding is invalid"
            )
        if downstream_binding.get("path") != paths.downstream_directory:
            raise PublicationFreezeError(f"{paths.stage} downstream path is not fixed")
        try:
            _validate_schema_four_source_projection(
                root,
                complete,
                downstream_binding,
            )
            downstream = validate_downstream_selection_completeness(
                root,
                complete.get("records"),
                downstream_binding,
            )
        except SelectionAuthorityError as error:
            raise PublicationFreezeError(
                f"{paths.stage} schema-4 selection evidence is invalid: {error}"
            ) from error
        downstream_manifest_sha256 = _sha256(
            downstream.get("downstream_manifest_sha256"),
            f"{paths.stage} downstream manifest checksum",
        )
        downstream_plan_sha256 = _sha256(
            downstream.get("downstream_plan_sha256"),
            f"{paths.stage} downstream plan checksum",
        )
        if (
            downstream.get("downstream_manifest_file_sha256")
            != downstream_manifest_file_sha256
            or downstream_binding.get("manifest_file_sha256")
            != downstream_manifest_file_sha256
            or downstream_binding.get("manifest_sha256") != downstream_manifest_sha256
            or downstream_binding.get("plan_sha256") != downstream_plan_sha256
        ):
            raise PublicationFreezeError(
                f"{paths.stage} downstream binding differs from the fixed archive"
            )

        try:
            recomputed = _recompute_selection_report(root, complete)
        except Exception as error:
            raise PublicationFreezeError(
                f"{paths.stage} selection report could not be recomputed: {error}"
            ) from error
        if report != recomputed:
            raise PublicationFreezeError(
                f"{paths.stage} selection report differs from recomputed selection"
            )

        terminal = index == len(layout.stages) - 1
        if not terminal:
            required_trigger = layout.stages[index + 1].stage
            if (
                report.get("trigger") != required_trigger
                or report.get("selected_configuration") is not None
            ):
                raise PublicationFreezeError(
                    f"{paths.stage} preceding selection report does not authorize "
                    f"the exact {required_trigger} activation"
                )
        else:
            trigger = report.get("trigger")
            if trigger == "downgrade_claim":
                raise PublicationFreezeError(
                    f"{paths.stage} terminal selection requests claim downgrade"
                )
            if trigger != "freeze_candidate":
                raise PublicationFreezeError(
                    f"{paths.stage} terminal selection trigger is not freeze_candidate"
                )
            selected = report.get("selected_configuration")
            assessments = report.get("assessments")
            selected_rows = (
                [
                    row
                    for row in assessments
                    if isinstance(row, Mapping)
                    and row.get("configuration_id") == selected
                ]
                if isinstance(assessments, list) and isinstance(selected, str)
                else []
            )
            expected_selected_version = (
                "v27" if layout.active_stage == "base" else layout.active_stage
            )
            if (
                len(selected_rows) != 1
                or selected_rows[0].get("version") != expected_selected_version
            ):
                raise PublicationFreezeError(
                    "selected version differs from the resolved active stage"
                )

        activation: RevisionActivation | None = None
        if paths.stage != "base":
            preceding = retained[-1]
            try:
                activation = validate_revision_activation(
                    root,
                    paths.stage,
                    require_clean=True,
                )
            except RevisionAuthorityError as error:
                raise PublicationFreezeError(
                    f"{paths.stage} revision activation is invalid: {error}"
                ) from error
            if (
                activation.version != paths.stage
                or activation.trigger != paths.stage
                or activation.selection_input_path != paths.activation_selection_input
                or activation.selection_report_path != paths.activation_selection_report
                or activation.selection_input_file_sha256
                != preceding.complete_input_file_sha256
                or activation.selection_result_sha256
                != preceding.complete_result_sha256
                or activation.selection_report_file_sha256
                != preceding.report_file_sha256
            ):
                raise PublicationFreezeError(
                    f"{paths.stage} activation hashes differ from retained evidence"
                )
            from .direct_values import direct_equal, direct_json_value

            preceding_comparator_selection = _thaw_receipt_json(
                preceding.complete_input["comparator_selection"]
            )
            activated_comparator_selection = direct_json_value(
                activation.base_comparator_selection,
                payload=True,
            )
            if not direct_equal(
                source.get("comparator_selection"),
                preceding_comparator_selection,
            ) or not direct_equal(
                activated_comparator_selection,
                preceding_comparator_selection,
            ):
                raise PublicationFreezeError(
                    f"{paths.stage} base comparator selection differs from retained "
                    "evidence"
                )

        frozen_source = _freeze_receipt_json(source)
        frozen_complete = _freeze_receipt_json(complete)
        frozen_report = _freeze_receipt_json(report)
        assert isinstance(frozen_source, Mapping)
        assert isinstance(frozen_complete, Mapping)
        assert isinstance(frozen_report, Mapping)
        retained.append(
            ValidatedPublicationStageEvidence(
                stage=paths.stage,
                source_input=frozen_source,
                source_input_file_sha256=source_file_sha256,
                source_result_sha256=source_result_sha256,
                complete_input=frozen_complete,
                complete_input_file_sha256=complete_file_sha256,
                complete_result_sha256=complete_result_sha256,
                report=frozen_report,
                report_file_sha256=report_file_sha256,
                downstream_manifest_file_sha256=(downstream_manifest_file_sha256),
                downstream_manifest_sha256=downstream_manifest_sha256,
                downstream_plan_sha256=downstream_plan_sha256,
                activation=activation,
            )
        )

    if _resolve_publication_stage(root) != layout:
        raise PublicationFreezeError(
            "publication stage changed during selection replay"
        )
    for snapshot in snapshots:
        paths = snapshot[0]
        observed = _read_publication_stage_core(root, paths)
        if observed != snapshot[1:]:
            raise PublicationFreezeError(
                f"{paths.stage} core evidence changed during selection replay"
            )

    receipt = PublicationStageEvidenceReceipt(
        active_stage=layout.active_stage,
        revision_versions=layout.revision_versions,
        stages=tuple(retained),
    )
    active_report = _json_copy(
        snapshots[-1][5],
        "active selection report",
    )
    assert isinstance(active_report, dict)
    return receipt, active_report


def _json_copy(value: object, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise PublicationFreezeError(f"{label} is not canonical JSON") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationFreezeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PublicationFreezeError(f"non-finite JSON constant {value}")


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
    )


@contextmanager
def _pinned_parent(path: Path, label: str):
    """Yield a parent dirfd reached by a no-symlink openat walk."""

    if not path.is_absolute() or not path.name:
        raise PublicationFreezeError(f"{label} path must be absolute")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = []
    edges: list[tuple[int, str, int, tuple[int, ...]]] = []
    try:
        current = os.open(path.anchor, flags)
        descriptors.append(current)
        relative_parent = path.parent.relative_to(path.anchor)
        for component in relative_parent.parts:
            named = os.stat(component, dir_fd=current, follow_symlinks=False)
            if not stat.S_ISDIR(named.st_mode):
                raise PublicationFreezeError(f"{label} parent path is not a directory")
            child = os.open(component, flags, dir_fd=current)
            opened = os.fstat(child)
            expected = _directory_identity(named)
            if _directory_identity(opened) != expected:
                os.close(child)
                raise PublicationFreezeError(
                    f"{label} parent path changed while being opened"
                )
            descriptors.append(child)
            edges.append((current, component, child, expected))
            current = child
        yield current
        for parent, component, child, expected in edges:
            try:
                named_after = os.stat(component, dir_fd=parent, follow_symlinks=False)
                opened_after = os.fstat(child)
            except OSError as error:
                raise PublicationFreezeError(
                    f"{label} parent path changed during access"
                ) from error
            if (
                _directory_identity(named_after) != expected
                or _directory_identity(opened_after) != expected
            ):
                raise PublicationFreezeError(
                    f"{label} parent path changed during access"
                )
    except PublicationFreezeError:
        raise
    except OSError as error:
        raise PublicationFreezeError(
            f"cannot open {label} parent path: {error}"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _secure_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    descriptor = -1
    try:
        with _pinned_parent(path, label) as parent:
            named_before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            opened_before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened_before.st_mode)
                or opened_before.st_nlink != 1
                or _identity(opened_before) != _identity(named_before)
            ):
                raise PublicationFreezeError(f"{label} is not a unique regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            opened_after = os.fstat(descriptor)
            named_after = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if _identity(opened_before) != _identity(opened_after) or _identity(
                opened_before
            ) != _identity(named_after):
                raise PublicationFreezeError(f"{label} changed while being read")
            raw = b"".join(chunks)
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
    except PublicationFreezeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PublicationFreezeError(f"cannot read {label}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise PublicationFreezeError(f"{label} must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _tree_directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stream_tree_file_sha256(
    descriptor: int,
    expected_size: int,
) -> tuple[str, int]:
    """Hash one pinned tree file without retaining its contents in memory."""

    if isinstance(descriptor, bool) or type(descriptor) is not int or descriptor < 0:
        raise TypeError("descriptor must be a nonnegative integer")
    if (
        isinstance(expected_size, bool)
        or type(expected_size) is not int
        or expected_size < 0
    ):
        raise TypeError("expected_size must be a nonnegative integer")
    digest = hashlib.sha256()
    byte_count = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        byte_count += len(chunk)
        if byte_count > expected_size:
            raise PublicationFreezeError(
                "publication stage tree file grew while being read"
            )
        digest.update(chunk)
    if byte_count != expected_size:
        raise PublicationFreezeError(
            "publication stage tree file changed size while being read"
        )
    return digest.hexdigest(), byte_count


def _closed_tree_file_entry(
    parent: int,
    name: str,
    relative: str,
) -> dict[str, object]:
    descriptor = -1
    try:
        named_before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(named_before.st_mode):
            if stat.S_ISLNK(named_before.st_mode):
                raise PublicationFreezeError(
                    f"publication stage tree contains a symlink: {relative}"
                )
            raise PublicationFreezeError(
                f"publication stage tree contains a special file: {relative}"
            )
        if named_before.st_nlink != 1:
            raise PublicationFreezeError(
                f"publication stage tree file is not unique: {relative}"
            )
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or _identity(opened_before) != _identity(named_before)
        ):
            raise PublicationFreezeError(
                f"publication stage tree file is not unique: {relative}"
            )
        digest, byte_count = _stream_tree_file_sha256(
            descriptor,
            opened_before.st_size,
        )
        opened_after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _identity(opened_before) != _identity(opened_after) or _identity(
            opened_before
        ) != _identity(named_after):
            raise PublicationFreezeError(
                f"publication stage tree changed while reading {relative}"
            )
        return {
            "path": relative,
            "kind": "file",
            "mode": stat.S_IMODE(opened_before.st_mode),
            "size": byte_count,
            "sha256": digest,
        }
    except PublicationFreezeError:
        raise
    except OSError as error:
        raise PublicationFreezeError(
            f"publication stage tree file is unavailable or unsafe: {relative}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _snapshot_closed_stage_tree(root: Path) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def visit(descriptor: int, relative: str) -> None:
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            raise PublicationFreezeError(
                f"publication stage tree directory is unavailable: {relative}"
            ) from error
        if not stat.S_ISDIR(before.st_mode):
            raise PublicationFreezeError(
                f"publication stage tree contains a special file: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "kind": "directory",
                "mode": stat.S_IMODE(before.st_mode),
            }
        )
        try:
            with os.scandir(descriptor) as iterator:
                names = sorted(entry.name for entry in iterator)
        except OSError as error:
            raise PublicationFreezeError(
                f"publication stage tree directory is unavailable: {relative}"
            ) from error
        for name in names:
            child_relative = name if relative == "." else f"{relative}/{name}"
            try:
                value = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError as error:
                raise PublicationFreezeError(
                    f"publication stage tree changed during scan: {child_relative}"
                ) from error
            if stat.S_ISLNK(value.st_mode):
                raise PublicationFreezeError(
                    f"publication stage tree contains a symlink: {child_relative}"
                )
            if stat.S_ISDIR(value.st_mode):
                child_descriptor = -1
                try:
                    child_descriptor = os.open(
                        name,
                        directory_flags,
                        dir_fd=descriptor,
                    )
                    opened_before = os.fstat(child_descriptor)
                    if _tree_directory_identity(
                        opened_before
                    ) != _tree_directory_identity(value):
                        raise PublicationFreezeError(
                            "publication stage tree directory changed while opening: "
                            f"{child_relative}"
                        )
                    visit(child_descriptor, child_relative)
                    opened_after = os.fstat(child_descriptor)
                    named_after = os.stat(
                        name,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                    if _tree_directory_identity(
                        opened_before
                    ) != _tree_directory_identity(
                        opened_after
                    ) or _tree_directory_identity(
                        opened_before
                    ) != _tree_directory_identity(named_after):
                        raise PublicationFreezeError(
                            "publication stage tree directory changed during scan: "
                            f"{child_relative}"
                        )
                except PublicationFreezeError:
                    raise
                except OSError as error:
                    raise PublicationFreezeError(
                        "publication stage tree directory is unavailable or unsafe: "
                        f"{child_relative}"
                    ) from error
                finally:
                    if child_descriptor >= 0:
                        os.close(child_descriptor)
            elif stat.S_ISREG(value.st_mode):
                rows.append(
                    _closed_tree_file_entry(
                        descriptor,
                        name,
                        child_relative,
                    )
                )
            else:
                raise PublicationFreezeError(
                    f"publication stage tree contains a special file: {child_relative}"
                )
        try:
            after = os.fstat(descriptor)
        except OSError as error:
            raise PublicationFreezeError(
                f"publication stage tree changed during scan: {relative}"
            ) from error
        if _tree_directory_identity(before) != _tree_directory_identity(after):
            raise PublicationFreezeError(
                f"publication stage tree changed during scan: {relative}"
            )

    root_descriptor = -1
    try:
        with _pinned_parent(root, "publication stage tree") as parent:
            named_before = os.stat(
                root.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(named_before.st_mode):
                raise PublicationFreezeError(
                    "publication stage tree contains a symlink: ."
                )
            if not stat.S_ISDIR(named_before.st_mode):
                raise PublicationFreezeError(
                    "publication stage tree contains a special file: ."
                )
            root_descriptor = os.open(
                root.name,
                directory_flags,
                dir_fd=parent,
            )
            opened_before = os.fstat(root_descriptor)
            if _tree_directory_identity(opened_before) != _tree_directory_identity(
                named_before
            ):
                raise PublicationFreezeError(
                    "publication stage tree root changed while opening"
                )
            visit(root_descriptor, ".")
            opened_after = os.fstat(root_descriptor)
            named_after = os.stat(
                root.name,
                dir_fd=parent,
                follow_symlinks=False,
            )
            if _tree_directory_identity(opened_before) != _tree_directory_identity(
                opened_after
            ) or _tree_directory_identity(opened_before) != _tree_directory_identity(
                named_after
            ):
                raise PublicationFreezeError(
                    "publication stage tree root changed during scan"
                )
    except PublicationFreezeError:
        raise
    except OSError as error:
        raise PublicationFreezeError(
            "publication stage tree root is unavailable or unsafe"
        ) from error
    finally:
        if root_descriptor >= 0:
            os.close(root_descriptor)
    return tuple(sorted(rows, key=lambda row: str(row["path"])))


def _closed_stage_tree_sha256(repository: Path, relative: str) -> str:
    """Hash a stable, closed directory containing no links or special files."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    path_value = PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        path_value is None
        or path_value.is_absolute()
        or not path_value.parts
        or any(part in {"", ".", ".."} for part in path_value.parts)
        or path_value.as_posix() != relative
    ):
        raise PublicationFreezeError("publication stage tree path is unsafe")
    root = Path(os.path.abspath(repository)).joinpath(*path_value.parts)
    first = _snapshot_closed_stage_tree(root)
    second = _snapshot_closed_stage_tree(root)
    if first != second:
        raise PublicationFreezeError(
            f"publication stage tree changed during receipt: {relative}"
        )
    return canonical_sha256(
        {
            "schema_version": 1,
            "root": relative,
            "entries": list(first),
        }
    )


def _canonical_bytes(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PublicationFreezeError("frozen method is not canonical JSON") from error


def _concurrent_publication_matches(parent: int, name: str, raw: bytes) -> bool:
    """Compare a create-only publication winner through one stable descriptor."""

    descriptor = -1
    try:
        named_before = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISREG(named_before.st_mode)
            or named_before.st_nlink != 1
            or named_before.st_size != len(raw)
        ):
            return False
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or opened_before.st_nlink != 1
            or _identity(opened_before) != _identity(named_before)
        ):
            return False
        position = 0
        identical = True
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            end = position + len(chunk)
            if end > len(raw) or chunk != raw[position:end]:
                identical = False
            position = end
        opened_after = os.fstat(descriptor)
        named_after = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if _identity(opened_before) != _identity(opened_after) or _identity(
            opened_before
        ) != _identity(named_after):
            raise PublicationFreezeError(
                "concurrently published frozen method changed while being read"
            )
        if not identical or position != len(raw):
            return False
        os.fsync(descriptor)
        return True
    except PublicationFreezeError:
        raise
    except OSError as error:
        raise PublicationFreezeError(
            "concurrently published frozen method is unavailable or unsafe"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_write(path: Path, raw: bytes) -> None:
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    with _pinned_parent(path, "frozen method output") as parent:
        descriptor = -1
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent,
            )
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileExistsError as error:
                if not _concurrent_publication_matches(parent, path.name, raw):
                    raise PublicationFreezeError(
                        "a different frozen method was concurrently published"
                    ) from error
                os.unlink(temporary_name, dir_fd=parent)
                os.fsync(parent)
                return
            os.unlink(temporary_name, dir_fd=parent)
            published = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISREG(published.st_mode)
                or published.st_nlink != 1
                or published.st_size != len(raw)
            ):
                raise PublicationFreezeError(
                    "published frozen method is not a unique regular file"
                )
            os.fsync(parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent)
            except FileNotFoundError:
                pass


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PublicationFreezeError(f"{label} is not a SHA-256 digest")
    return value


def _git_oid(value: object, label: str) -> str:
    if not isinstance(value, str) or _GIT_OID.fullmatch(value) is None:
        raise PublicationFreezeError(f"{label} is not a Git object ID")
    return value


def _artifact_bindings(value: Mapping[str, Mapping[str, str]]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PublicationFreezeError("artifact bindings must be a mapping")
    result: dict[str, object] = {}
    normalized_paths: set[str] = set()
    for name in sorted(value):
        row = value[name]
        if (
            not isinstance(name, str)
            or _SAFE_ID.fullmatch(name.replace("_", "-")) is None
            or not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
        ):
            raise PublicationFreezeError("artifact binding is invalid")
        raw_path = row.get("path")
        if not isinstance(raw_path, str):
            raise PublicationFreezeError(f"artifact {name} path is invalid")
        path = PurePosixPath(raw_path)
        normalized = path.as_posix()
        if normalized in normalized_paths:
            raise PublicationFreezeError(
                f"artifact bindings contain a duplicate normalized path: {normalized}"
            )
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or normalized != raw_path
        ):
            raise PublicationFreezeError(f"artifact {name} path is invalid")
        normalized_paths.add(normalized)
        result[name] = {
            "path": normalized,
            "sha256": _sha256(row.get("sha256"), f"artifact {name} checksum"),
        }
    return result


_STAGE_RECEIPT_FIELDS = {
    "stage",
    "source_input_artifact",
    "complete_input_artifact",
    "report_artifact",
    "evaluation_manifest_artifact",
    "reconstruction_checkpoint_artifact",
    "orthogonal_manifest_artifact",
    "downstream_plan_artifact",
    "downstream_manifest_artifact",
    "source_result_sha256",
    "complete_result_sha256",
    "downstream_plan_sha256",
    "downstream_manifest_sha256",
    "reconstruction_tree_sha256",
    "orthogonal_tree_sha256",
    "downstream_tree_sha256",
    "activation",
}
_ACTIVATION_RECEIPT_FIELDS = {
    "version",
    "trigger",
    "revision_authority_artifact",
    "preceding_complete_input_artifact",
    "preceding_report_artifact",
    "selection_input_file_sha256",
    "selection_result_sha256",
    "selection_report_file_sha256",
}


def _stage_artifact_names(layout: PublicationStageLayout) -> list[str]:
    suffixes = (
        "selection_source_input",
        "selection_complete_input",
        "selection_report",
        "evaluation_manifest",
        "reconstruction_checkpoint",
        "orthogonal_manifest",
        "downstream_plan",
        "downstream_manifest",
    )
    return sorted(
        f"{stage.stage}_{suffix}" for stage in layout.stages for suffix in suffixes
    )


def _validated_publication_artifact_bindings(
    layout: PublicationStageLayout,
    value: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    artifacts = _artifact_bindings(value)
    include_external = "external_reference_checkpoint" in artifacts
    expected = _publication_artifact_paths(
        layout,
        include_external_reference=include_external,
    )
    if set(artifacts) != set(expected):
        raise PublicationFreezeError(
            "dynamic publication artifact bindings are incomplete or contain extras"
        )
    for name, relative in expected.items():
        row = artifacts[name]
        assert isinstance(row, Mapping)
        if row.get("path") != relative:
            raise PublicationFreezeError(
                f"dynamic publication artifact path is not fixed: {name}"
            )
    return artifacts


def _stage_receipt_inventory_sha256(
    unsigned: Mapping[str, object],
    artifact_names: Sequence[str],
    artifacts: Mapping[str, object],
) -> str:
    return canonical_sha256(
        {
            "receipt": unsigned,
            "artifact_bindings": {name: artifacts[name] for name in artifact_names},
        }
    )


def _validate_development_stage_receipt(
    value: object,
    layout: PublicationStageLayout,
    artifact_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    """Validate the closed stage identity carried by a frozen payload."""

    if type(layout) is not PublicationStageLayout:
        raise TypeError("layout must be an exact PublicationStageLayout")
    if type(value) is not dict or set(value) != {
        "schema_version",
        "active_stage",
        "revision_versions",
        "stage_order",
        "stages",
        "artifact_names",
        "inventory_sha256",
    }:
        raise PublicationFreezeError("development stage receipt schema differs")
    artifacts = _validated_publication_artifact_bindings(layout, artifact_bindings)
    expected_order = [stage.stage for stage in layout.stages]
    expected_versions = list(layout.revision_versions)
    expected_names = _stage_artifact_names(layout)
    if (
        value.get("schema_version") != 1
        or type(value.get("schema_version")) is not int
        or value.get("active_stage") != layout.active_stage
        or value.get("revision_versions") != expected_versions
        or value.get("stage_order") != expected_order
        or value.get("artifact_names") != expected_names
        or type(value.get("stages")) is not list
        or len(value["stages"]) != len(layout.stages)
    ):
        raise PublicationFreezeError(
            "development stage receipt inventory or stage order differs"
        )
    if any(name not in artifacts for name in expected_names):
        raise PublicationFreezeError(
            "development stage receipt artifact inventory is incomplete"
        )

    for index, (row, paths) in enumerate(zip(value["stages"], layout.stages)):
        if type(row) is not dict or set(row) != _STAGE_RECEIPT_FIELDS:
            raise PublicationFreezeError("development stage receipt row differs")
        prefix = paths.stage
        expected_artifacts = {
            "source_input_artifact": f"{prefix}_selection_source_input",
            "complete_input_artifact": f"{prefix}_selection_complete_input",
            "report_artifact": f"{prefix}_selection_report",
            "evaluation_manifest_artifact": f"{prefix}_evaluation_manifest",
            "reconstruction_checkpoint_artifact": (
                f"{prefix}_reconstruction_checkpoint"
            ),
            "orthogonal_manifest_artifact": f"{prefix}_orthogonal_manifest",
            "downstream_plan_artifact": f"{prefix}_downstream_plan",
            "downstream_manifest_artifact": f"{prefix}_downstream_manifest",
        }
        if row.get("stage") != prefix or any(
            row.get(field) != artifact for field, artifact in expected_artifacts.items()
        ):
            raise PublicationFreezeError(
                "development stage receipt artifact references differ"
            )
        for field in (
            "source_result_sha256",
            "complete_result_sha256",
            "downstream_plan_sha256",
            "downstream_manifest_sha256",
            "reconstruction_tree_sha256",
            "orthogonal_tree_sha256",
            "downstream_tree_sha256",
        ):
            _sha256(row.get(field), f"development stage receipt {field}")
        activation = row.get("activation")
        if prefix == "base":
            if activation is not None:
                raise PublicationFreezeError(
                    "base development stage receipt activation must be null"
                )
            continue
        if (
            type(activation) is not dict
            or set(activation) != _ACTIVATION_RECEIPT_FIELDS
        ):
            raise PublicationFreezeError(
                f"{prefix} development stage receipt activation differs"
            )
        preceding = layout.stages[index - 1].stage
        preceding_row = value["stages"][index - 1]
        if type(preceding_row) is not dict:  # pragma: no cover - validated in order
            raise PublicationFreezeError(
                f"{prefix} development stage receipt activation differs"
            )
        preceding_complete_artifact = f"{preceding}_selection_complete_input"
        preceding_report_artifact = f"{preceding}_selection_report"
        if (
            activation.get("version") != prefix
            or activation.get("trigger") != prefix
            or activation.get("revision_authority_artifact") != f"{prefix}_revision"
            or activation.get("preceding_complete_input_artifact")
            != preceding_complete_artifact
            or activation.get("preceding_report_artifact") != preceding_report_artifact
            or activation.get("selection_input_file_sha256")
            != artifacts[preceding_complete_artifact]["sha256"]
            or activation.get("selection_result_sha256")
            != preceding_row.get("complete_result_sha256")
            or activation.get("selection_report_file_sha256")
            != artifacts[preceding_report_artifact]["sha256"]
        ):
            raise PublicationFreezeError(
                f"{prefix} development stage receipt activation reference differs"
            )
        for field in (
            "selection_input_file_sha256",
            "selection_result_sha256",
            "selection_report_file_sha256",
        ):
            _sha256(
                activation.get(field),
                f"{prefix} development stage receipt activation {field}",
            )

    unsigned = {key: value for key, value in value.items() if key != "inventory_sha256"}
    expected_inventory = _stage_receipt_inventory_sha256(
        unsigned,
        expected_names,
        artifacts,
    )
    if value.get("inventory_sha256") != expected_inventory:
        raise PublicationFreezeError(
            "development stage receipt inventory checksum differs"
        )
    copied = _json_copy(value, "development stage receipt")
    assert isinstance(copied, dict)
    return copied


def _layout_from_development_stage_receipt(value: object) -> PublicationStageLayout:
    if not isinstance(value, Mapping):
        raise PublicationFreezeError("development stage receipt is absent")
    active_stage = value.get("active_stage")
    if not isinstance(active_stage, str):
        raise PublicationFreezeError(
            "development stage receipt active stage is invalid"
        )
    return _publication_layout_for_active_stage(active_stage)


def _build_development_stage_receipt(
    repository: Path,
    layout: PublicationStageLayout,
    evidence: PublicationStageEvidenceReceipt,
    artifact_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    """Build the exact versioned receipt for one replayed stage prefix."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if type(layout) is not PublicationStageLayout:
        raise TypeError("layout must be an exact PublicationStageLayout")
    if type(evidence) is not PublicationStageEvidenceReceipt:
        raise TypeError("evidence must be an exact PublicationStageEvidenceReceipt")
    if (
        evidence.active_stage != layout.active_stage
        or evidence.revision_versions != layout.revision_versions
        or tuple(stage.stage for stage in evidence.stages)
        != tuple(stage.stage for stage in layout.stages)
    ):
        raise PublicationFreezeError(
            "validated evidence differs from the publication stage layout"
        )
    artifacts = _validated_publication_artifact_bindings(layout, artifact_bindings)
    rows: list[dict[str, object]] = []
    for index, (paths, retained) in enumerate(zip(layout.stages, evidence.stages)):
        prefix = paths.stage
        activation = None
        if retained.activation is not None:
            preceding = layout.stages[index - 1].stage
            activation = {
                "version": retained.activation.version,
                "trigger": retained.activation.trigger,
                "revision_authority_artifact": f"{prefix}_revision",
                "preceding_complete_input_artifact": (
                    f"{preceding}_selection_complete_input"
                ),
                "preceding_report_artifact": f"{preceding}_selection_report",
                "selection_input_file_sha256": (
                    retained.activation.selection_input_file_sha256
                ),
                "selection_result_sha256": (
                    retained.activation.selection_result_sha256
                ),
                "selection_report_file_sha256": (
                    retained.activation.selection_report_file_sha256
                ),
            }
        rows.append(
            {
                "stage": prefix,
                "source_input_artifact": f"{prefix}_selection_source_input",
                "complete_input_artifact": f"{prefix}_selection_complete_input",
                "report_artifact": f"{prefix}_selection_report",
                "evaluation_manifest_artifact": f"{prefix}_evaluation_manifest",
                "reconstruction_checkpoint_artifact": (
                    f"{prefix}_reconstruction_checkpoint"
                ),
                "orthogonal_manifest_artifact": f"{prefix}_orthogonal_manifest",
                "downstream_plan_artifact": f"{prefix}_downstream_plan",
                "downstream_manifest_artifact": f"{prefix}_downstream_manifest",
                "source_result_sha256": retained.source_result_sha256,
                "complete_result_sha256": retained.complete_result_sha256,
                "downstream_plan_sha256": retained.downstream_plan_sha256,
                "downstream_manifest_sha256": retained.downstream_manifest_sha256,
                "reconstruction_tree_sha256": _closed_stage_tree_sha256(
                    repository,
                    paths.reconstruction_directory,
                ),
                "orthogonal_tree_sha256": _closed_stage_tree_sha256(
                    repository,
                    paths.orthogonal_directory,
                ),
                "downstream_tree_sha256": _closed_stage_tree_sha256(
                    repository,
                    paths.downstream_directory,
                ),
                "activation": activation,
            }
        )
    artifact_names = _stage_artifact_names(layout)
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "active_stage": layout.active_stage,
        "revision_versions": list(layout.revision_versions),
        "stage_order": [stage.stage for stage in layout.stages],
        "stages": rows,
        "artifact_names": artifact_names,
    }
    receipt = {
        **unsigned,
        "inventory_sha256": _stage_receipt_inventory_sha256(
            unsigned,
            artifact_names,
            artifacts,
        ),
    }
    return _validate_development_stage_receipt(receipt, layout, artifact_bindings)


def _validate_saver_package_authority(
    payloads: Mapping[str, Mapping[str, object]],
    artifact_bindings: Mapping[str, Mapping[str, str]],
) -> None:
    """Validate the package-specific SAVER authority independently of runtime lock."""

    qualification = payloads.get("saver_qualification")
    package_lock = payloads.get("saver_package_lock")
    build_receipt = payloads.get("saver_build_receipt")
    if not all(
        isinstance(value, Mapping)
        for value in (qualification, package_lock, build_receipt)
    ):
        raise PublicationFreezeError("SAVER qualification authority is incomplete")
    assert isinstance(qualification, Mapping)
    assert isinstance(package_lock, Mapping)
    assert isinstance(build_receipt, Mapping)
    if set(qualification) != {
        "schema_version",
        "environment_id",
        "package_lock",
        "build_receipt",
        "installed_library_sha256",
        "source",
    }:
        raise PublicationFreezeError("SAVER qualification receipt is not closed")
    package_binding = qualification.get("package_lock")
    build_binding = qualification.get("build_receipt")
    source = qualification.get("source")
    if (
        qualification.get("schema_version") != 1
        or qualification.get("environment_id") != "saver-r"
        or not isinstance(package_binding, Mapping)
        or set(package_binding) != {"path", "sha256"}
        or not isinstance(build_binding, Mapping)
        or set(build_binding) != {"path", "sha256"}
        or not isinstance(source, Mapping)
        or set(source) != {"url", "revision", "tree"}
        or not isinstance(source.get("url"), str)
        or not source.get("url")
    ):
        raise PublicationFreezeError("SAVER qualification receipt is malformed")
    _git_oid(source.get("revision"), "SAVER qualification source revision")
    _git_oid(source.get("tree"), "SAVER qualification source tree")
    installed_sha256 = _sha256(
        qualification.get("installed_library_sha256"),
        "SAVER qualification installed-library checksum",
    )
    expected_package = artifact_bindings.get("saver_package_lock")
    expected_build = artifact_bindings.get("saver_build_receipt")
    if (
        not isinstance(expected_package, Mapping)
        or not isinstance(expected_build, Mapping)
        or package_binding.get("path") != _COMMON_TRACKED_PATHS["saver_package_lock"]
        or package_binding.get("sha256") != expected_package.get("sha256")
        or build_binding.get("path") != _COMMON_TRACKED_PATHS["saver_build_receipt"]
        or build_binding.get("sha256") != expected_build.get("sha256")
    ):
        raise PublicationFreezeError(
            "SAVER qualification differs from its package/build artifacts"
        )

    if set(package_lock) != {
        "schema_version",
        "environment_id",
        "r_version",
        "packages",
        "upstream_saver",
        "installed_library_sha256",
        "build_receipt_sha256",
    }:
        raise PublicationFreezeError("SAVER package lock is not closed")
    packages = package_lock.get("packages")
    upstream = package_lock.get("upstream_saver")
    if (
        package_lock.get("schema_version") != 1
        or package_lock.get("environment_id") != "saver-r"
        or not isinstance(package_lock.get("r_version"), str)
        or not package_lock.get("r_version")
        or package_lock.get("installed_library_sha256") != installed_sha256
        or package_lock.get("build_receipt_sha256") != expected_build.get("sha256")
        or type(packages) is not list
        or not packages
        or not isinstance(upstream, Mapping)
        or set(upstream) != {"package", "version", "url", "revision", "tree"}
        or upstream.get("package") != "SAVER"
        or {key: upstream.get(key) for key in ("url", "revision", "tree")}
        != dict(source)
        or not isinstance(upstream.get("version"), str)
        or not upstream.get("version")
    ):
        raise PublicationFreezeError(
            "SAVER package lock differs from its qualification receipt"
        )
    package_versions: dict[str, str] = {}
    for row in packages:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"package", "version", "url", "sha256"}
            or not isinstance(row.get("package"), str)
            or not row.get("package")
            or row.get("package") in package_versions
            or not isinstance(row.get("version"), str)
            or not row.get("version")
            or not isinstance(row.get("url"), str)
            or not row.get("url")
        ):
            raise PublicationFreezeError("SAVER package lock entry is malformed")
        _sha256(row.get("sha256"), f"SAVER {row.get('package')} package checksum")
        package_versions[str(row["package"])] = str(row["version"])
    package_versions["SAVER"] = str(upstream["version"])

    expected_receipt_keys = {
        "schema_version",
        "status",
        "build_date",
        "environment_id",
        "build_script_sha256",
        "build_log_sha256",
        "r_version",
        "library_dir",
        "installed_library_sha256",
        "saver_source",
        "package_versions",
        "smoke_test",
    }
    saver_source = build_receipt.get("saver_source")
    smoke_test = build_receipt.get("smoke_test")
    if (
        set(build_receipt) != expected_receipt_keys
        or build_receipt.get("schema_version") != 1
        or build_receipt.get("status") != "real_tiny_smoke_passed"
        or build_receipt.get("environment_id") != "saver-r"
        or build_receipt.get("r_version") != package_lock.get("r_version")
        or build_receipt.get("installed_library_sha256") != installed_sha256
        or build_receipt.get("package_versions") != package_versions
        or not isinstance(build_receipt.get("build_date"), str)
        or not build_receipt.get("build_date")
        or not isinstance(build_receipt.get("library_dir"), str)
        or not Path(str(build_receipt.get("library_dir"))).is_absolute()
        or not isinstance(saver_source, Mapping)
        or set(saver_source) != {"revision", "tree"}
        or dict(saver_source)
        != {"revision": source.get("revision"), "tree": source.get("tree")}
        or not isinstance(smoke_test, Mapping)
        or set(smoke_test) != {"command", "result"}
        or not isinstance(smoke_test.get("command"), str)
        or not smoke_test.get("command")
        or smoke_test.get("result") != "1 passed"
    ):
        raise PublicationFreezeError(
            "SAVER build receipt differs from its qualified package lock"
        )
    _sha256(
        build_receipt.get("build_script_sha256"),
        "SAVER build script checksum",
    )
    _sha256(build_receipt.get("build_log_sha256"), "SAVER build log checksum")

    registry = payloads.get("method_registry")
    if isinstance(registry, Mapping) and type(registry.get("methods")) is list:
        saver_rows = [
            row
            for row in registry["methods"]
            if isinstance(row, Mapping) and row.get("id") == "saver"
        ]
        if len(saver_rows) > 1:
            raise PublicationFreezeError("SAVER registry authority is not unique")
        if saver_rows:
            saver = saver_rows[0]
            environment = saver.get("environment")
            registry_source = saver.get("source")
            if (
                not isinstance(registry_source, Mapping)
                or {
                    key: registry_source.get(key) for key in ("url", "revision", "tree")
                }
                != dict(source)
                or not isinstance(environment, Mapping)
                or environment.get("id") != "saver-r"
                or environment.get("status") != "ready"
            ):
                raise PublicationFreezeError(
                    "SAVER registry differs from its package qualification"
                )


def _runtime_environment_summary(
    path: Path,
    expected_sha256: str,
    method_registry: Mapping[str, object],
) -> dict[str, object]:
    try:
        lock = load_runtime_environment_lock(path)
    except (RuntimeEnvironmentError, OSError, ValueError) as error:
        raise PublicationFreezeError(f"runtime lock is invalid: {error}") from error
    if lock.file_sha256 != expected_sha256:
        raise PublicationFreezeError("runtime lock checksum changed during validation")
    methods = method_registry.get("methods")
    if type(methods) is not list:
        raise PublicationFreezeError("method registry is invalid")
    expected_ids = {"benchmark"}
    for row in methods:
        if not isinstance(row, Mapping):
            raise PublicationFreezeError("method registry row is invalid")
        method_id = row.get("id")
        environment = row.get("environment")
        if row.get("integration_status") == "implemented":
            if (
                not isinstance(environment, Mapping)
                or environment.get("status") != "ready"
                or environment.get("lock_sha256") != expected_sha256
            ):
                raise PublicationFreezeError(
                    f"implemented method {method_id} is not bound to the runtime lock"
                )
        if (
            row.get("execution_scope")
            in {"same_input_required", "external_reference_only"}
            and method_id not in {"observed", "capacity-matched-ae", "maskimpute"}
            and isinstance(environment, Mapping)
            and environment.get("status") == "ready"
        ):
            if not isinstance(method_id, str):
                raise PublicationFreezeError("method registry ID is invalid")
            if environment.get("lock_sha256") != expected_sha256:
                raise PublicationFreezeError(
                    f"ready method {method_id} is not bound to the runtime lock"
                )
            expected_ids.add(method_id)
    observed_ids = {entry.environment_id for entry in lock.entries}
    if observed_ids != expected_ids:
        raise PublicationFreezeError(
            "runtime lock IDs differ from implemented executable method environments"
        )
    return {
        "schema_version": 1,
        "lock_file_sha256": lock.file_sha256,
        "environment_inventory_sha256s": {
            entry.environment_id: entry.inventory_sha256 for entry in lock.entries
        },
    }


def _development_dataset_ids(dataset_status: Mapping[str, object]) -> tuple[str, ...]:
    rows = dataset_status.get("rows")
    unsigned = {
        key: value for key, value in dataset_status.items() if key != "manifest_sha256"
    }
    if (
        dataset_status.get("schema_version") != 1
        or dataset_status.get("namespace") != "dev"
        or dataset_status.get("status") != "completed"
        or type(rows) is not list
        or not rows
        or dataset_status.get("completed_count") != len(rows)
        or dataset_status.get("failed_count") != 0
        or dataset_status.get("manifest_sha256") != canonical_sha256(unsigned)
    ):
        raise PublicationFreezeError("development dataset status is not complete")
    dataset_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "completed":
            raise PublicationFreezeError("development dataset row is not complete")
        dataset_id = row.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise PublicationFreezeError("development dataset ID is invalid")
        dataset_ids.append(dataset_id)
    if len(dataset_ids) != len(set(dataset_ids)):
        raise PublicationFreezeError("development dataset IDs are duplicated")
    return tuple(sorted(dataset_ids))


def _development_execution_evidence(
    checkpoint: Mapping[str, object],
    *,
    artifact: str,
    execution_track: str,
    eligible_dataset_ids: Sequence[str],
) -> dict[str, dict[str, object]]:
    stage_artifacts = {
        "base_reconstruction_checkpoint",
        "v28_reconstruction_checkpoint",
        "v29_reconstruction_checkpoint",
    }
    if artifact not in stage_artifacts | {"external_reference_checkpoint"}:
        raise PublicationFreezeError("development execution artifact is invalid")
    if execution_track not in {"same_input", "external_reference"}:
        raise PublicationFreezeError("development execution track is invalid")
    if (artifact == "external_reference_checkpoint") != (
        execution_track == "external_reference"
    ):
        raise PublicationFreezeError(
            "development execution artifact and track are inconsistent"
        )
    eligible_ids = tuple(eligible_dataset_ids)
    if (
        not eligible_ids
        or eligible_ids != tuple(sorted(set(eligible_ids)))
        or any(
            not isinstance(dataset_id, str) or not dataset_id
            for dataset_id in eligible_ids
        )
    ):
        raise PublicationFreezeError("development eligible dataset panel is invalid")
    checkpoint_sha256 = checkpoint.get("checkpoint_sha256")
    checkpoint_body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    records = checkpoint.get("records")
    planned_run_count = checkpoint.get("planned_run_count")
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("status") != "completed"
        or type(records) is not list
        or type(planned_run_count) is not int
        or planned_run_count < 1
        or planned_run_count != len(records)
        or checkpoint_sha256 != canonical_sha256(checkpoint_body)
    ):
        raise PublicationFreezeError(
            "development reconstruction checkpoint is not complete and canonical"
        )
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("run"), Mapping
        ):
            raise PublicationFreezeError("development execution record is invalid")
        run = record["run"]
        method_id = run.get("method_id")
        dataset_id = run.get("dataset_id")
        status = run.get("status")
        if (
            not isinstance(method_id, str)
            or _SAFE_ID.fullmatch(method_id) is None
            or not isinstance(dataset_id, str)
            or not dataset_id
            or status not in _RUN_STATUSES
        ):
            raise PublicationFreezeError("development execution disposition is invalid")
        if status != "completed":
            reason = run.get("reason")
            if (
                not isinstance(reason, str)
                or _REASON_CODE.fullmatch(reason) is None
                or any(marker in reason for marker in _NONFINAL_REASON_MARKERS)
            ):
                raise PublicationFreezeError(
                    "failed development execution lacks a terminal reason code"
                )
        grouped.setdefault(method_id, []).append(record)
    result: dict[str, dict[str, object]] = {}
    for method_id in sorted(grouped):
        records = grouped[method_id]
        statuses = [record["run"]["status"] for record in records]
        attempted_dataset_ids = sorted(
            {str(record["run"]["dataset_id"]) for record in records}
        )
        completed_dataset_ids = sorted(
            {
                str(record["run"]["dataset_id"])
                for record in records
                if record["run"]["status"] == "completed"
            }
        )
        reasons = sorted(
            {
                reason
                for record in records
                if record["run"]["status"] != "completed"
                and isinstance((reason := record["run"].get("reason")), str)
                and reason
            }
        )
        status_counts = {
            status: statuses.count(status) for status in sorted(set(statuses))
        }
        if tuple(attempted_dataset_ids) != eligible_ids:
            raise PublicationFreezeError(
                f"method {method_id} attempts do not cover its eligible development panel"
            )
        result[method_id] = {
            "artifact": artifact,
            "execution_track": execution_track,
            "checkpoint_payload_sha256": checkpoint_sha256,
            "records_sha256": canonical_sha256(records),
            "eligible_dataset_count": len(eligible_ids),
            "eligible_dataset_ids_sha256": canonical_sha256(eligible_ids),
            "attempted_run_count": len(records),
            "completed_run_count": statuses.count("completed"),
            "failed_run_count": len(records) - statuses.count("completed"),
            "status_counts": status_counts,
            "attempted_dataset_count": len(attempted_dataset_ids),
            "completed_dataset_count": len(completed_dataset_ids),
            "attempted_dataset_ids_sha256": canonical_sha256(attempted_dataset_ids),
            "completed_dataset_ids_sha256": canonical_sha256(completed_dataset_ids),
            "failure_reason_codes": reasons,
            "failure_reasons_sha256": canonical_sha256(reasons),
        }
    return result


def _active_execution_evidence(
    base_checkpoint: Mapping[str, object],
    *,
    active_stage: str,
    selected_stage_checkpoint: Mapping[str, object] | None,
    selected_configuration: Mapping[str, object],
    eligible_dataset_ids: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Retain base comparators and replace only selected revision MaskImpute."""

    if active_stage not in {"base", "v28", "v29"}:
        raise PublicationFreezeError("active execution stage is invalid")
    if not isinstance(selected_configuration, Mapping):
        raise PublicationFreezeError("selected execution configuration is invalid")
    selected_id = selected_configuration.get("configuration_id")
    selected_version = selected_configuration.get("version")
    selected_sha256 = selected_configuration.get("configuration_sha256")
    if (
        not isinstance(selected_id, str)
        or _SAFE_ID.fullmatch(selected_id) is None
        or not isinstance(selected_sha256, str)
        or _SHA256.fullmatch(selected_sha256) is None
        or selected_version != ("v27" if active_stage == "base" else active_stage)
    ):
        raise PublicationFreezeError(
            "selected execution configuration differs from the active stage"
        )
    base = _development_execution_evidence(
        base_checkpoint,
        artifact="base_reconstruction_checkpoint",
        execution_track="same_input",
        eligible_dataset_ids=eligible_dataset_ids,
    )
    if "maskimpute" not in base:
        raise PublicationFreezeError(
            "base comparator denominator lacks MaskImpute evidence"
        )
    if active_stage == "base":
        if selected_stage_checkpoint is not None:
            raise PublicationFreezeError(
                "base selection cannot consume a revision checkpoint"
            )
        return base
    if not isinstance(selected_stage_checkpoint, Mapping):
        raise PublicationFreezeError("selected-stage execution checkpoint is absent")
    records = selected_stage_checkpoint.get("records")
    if type(records) is not list or not records:
        raise PublicationFreezeError(
            "selected-stage execution lacks a unique selected configuration"
        )
    identities: set[tuple[object, object]] = set()
    for record in records:
        run = record.get("run") if isinstance(record, Mapping) else None
        if not isinstance(run, Mapping):
            raise PublicationFreezeError("selected-stage execution record is invalid")
        if run.get("method_id") != "maskimpute":
            raise PublicationFreezeError(
                "selected-stage execution may contain only MaskImpute rows"
            )
        identities.add((run.get("configuration_id"), run.get("configuration_sha256")))
    if identities != {(selected_id, selected_sha256)}:
        raise PublicationFreezeError(
            "selected-stage execution does not contain one unique selected configuration"
        )
    revision = _development_execution_evidence(
        selected_stage_checkpoint,
        artifact=f"{active_stage}_reconstruction_checkpoint",
        execution_track="same_input",
        eligible_dataset_ids=eligible_dataset_ids,
    )
    if set(revision) != {"maskimpute"}:
        raise PublicationFreezeError(
            "selected-stage execution may replace only MaskImpute evidence"
        )
    return {**base, "maskimpute": revision["maskimpute"]}


def _validated_execution_evidence(
    value: object, method_id: str
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
        "artifact",
        "execution_track",
        "checkpoint_payload_sha256",
        "records_sha256",
        "eligible_dataset_count",
        "eligible_dataset_ids_sha256",
        "attempted_run_count",
        "completed_run_count",
        "failed_run_count",
        "status_counts",
        "attempted_dataset_count",
        "completed_dataset_count",
        "attempted_dataset_ids_sha256",
        "completed_dataset_ids_sha256",
        "failure_reason_codes",
        "failure_reasons_sha256",
    }:
        raise PublicationFreezeError(
            f"method {method_id} development execution evidence is invalid"
        )
    attempted = value.get("attempted_run_count")
    completed = value.get("completed_run_count")
    failed = value.get("failed_run_count")
    attempted_datasets = value.get("attempted_dataset_count")
    completed_datasets = value.get("completed_dataset_count")
    status_counts = value.get("status_counts")
    failure_reasons = value.get("failure_reason_codes")
    eligible_datasets = value.get("eligible_dataset_count")
    if (
        value.get("artifact")
        not in {
            "base_reconstruction_checkpoint",
            "v28_reconstruction_checkpoint",
            "v29_reconstruction_checkpoint",
            "external_reference_checkpoint",
        }
        or value.get("execution_track") not in {"same_input", "external_reference"}
        or (
            (value.get("artifact") == "external_reference_checkpoint")
            != (value.get("execution_track") == "external_reference")
        )
        or type(eligible_datasets) is not int
        or eligible_datasets < 1
        or type(attempted) is not int
        or attempted < 1
        or type(completed) is not int
        or completed < 0
        or type(failed) is not int
        or failed < 0
        or completed + failed != attempted
        or type(attempted_datasets) is not int
        or attempted_datasets < 1
        or attempted_datasets != eligible_datasets
        or type(completed_datasets) is not int
        or not 0 <= completed_datasets <= attempted_datasets
        or not isinstance(status_counts, Mapping)
        or not status_counts
        or any(
            status not in _RUN_STATUSES or type(count) is not int or count < 1
            for status, count in status_counts.items()
        )
        or sum(status_counts.values()) != attempted
        or status_counts.get("completed", 0) != completed
        or not isinstance(failure_reasons, list)
        or failure_reasons != sorted(set(failure_reasons))
        or any(
            not isinstance(reason, str)
            or _REASON_CODE.fullmatch(reason) is None
            or any(marker in reason for marker in _NONFINAL_REASON_MARKERS)
            for reason in failure_reasons
        )
    ):
        raise PublicationFreezeError(
            f"method {method_id} development execution evidence is invalid"
        )
    _sha256(
        value.get("checkpoint_payload_sha256"),
        f"method {method_id} execution checkpoint",
    )
    for name in (
        "records_sha256",
        "eligible_dataset_ids_sha256",
        "attempted_dataset_ids_sha256",
        "completed_dataset_ids_sha256",
    ):
        _sha256(value.get(name), f"method {method_id} execution {name}")
    _sha256(
        value.get("failure_reasons_sha256"),
        f"method {method_id} execution failure reasons",
    )
    if value.get("failure_reasons_sha256") != canonical_sha256(failure_reasons):
        raise PublicationFreezeError(
            f"method {method_id} execution failure reasons checksum differs"
        )
    if value.get("attempted_dataset_ids_sha256") != value.get(
        "eligible_dataset_ids_sha256"
    ):
        raise PublicationFreezeError(
            f"method {method_id} attempted dataset panel differs from eligibility"
        )
    return _json_copy(value, f"method {method_id} development execution evidence")


def _selected_calibrator_summary(
    calibration: Mapping[str, object],
    *,
    artifact_file_sha256: str,
    score_policy: object,
) -> dict[str, object]:
    selected_algorithm = calibration.get("selected_algorithm")
    definition = calibration.get("calibrator")
    payload_sha256 = calibration.get("payload_sha256")
    inference_features = calibration.get("inference_features")
    if (
        calibration.get("schema_version") != 3
        or selected_algorithm not in {"identity", "logistic", "beta", "isotonic"}
        or not isinstance(definition, Mapping)
        or definition.get("algorithm") != selected_algorithm
        or not isinstance(inference_features, (Mapping, list))
    ):
        raise PublicationFreezeError("retained calibrator identity is invalid")
    _sha256(payload_sha256, "retained calibration payload checksum")
    usage = (
        "retained_all_development_calibrator"
        if score_policy == "retained_development_calibrator"
        else "direct_count_score"
    )
    if score_policy not in {
        "retained_development_calibrator",
        "direct_cross_fitted_count_score",
    }:
        raise PublicationFreezeError("selected score policy is invalid")
    return {
        "score_policy": score_policy,
        "final_usage": usage,
        "selected_algorithm": selected_algorithm,
        "artifact_file_sha256": _sha256(
            artifact_file_sha256, "retained calibration artifact checksum"
        ),
        "artifact_payload_sha256": payload_sha256,
        "artifact": _json_copy(calibration, "retained calibration artifact"),
        "calibrator_definition": _json_copy(
            definition, "retained calibrator definition"
        ),
        "calibrator_definition_sha256": canonical_sha256(definition),
        "inference_features": _json_copy(
            inference_features, "retained calibrator inference features"
        ),
        "inference_features_sha256": canonical_sha256(inference_features),
    }


def _validate_selected_calibrator_summary(
    value: object, *, score_policy: object, artifact_file_sha256: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {
        "score_policy",
        "final_usage",
        "selected_algorithm",
        "artifact_file_sha256",
        "artifact_payload_sha256",
        "artifact",
        "calibrator_definition",
        "calibrator_definition_sha256",
        "inference_features",
        "inference_features_sha256",
    }:
        raise PublicationFreezeError("selected calibrator summary is invalid")
    if (
        value.get("score_policy") != score_policy
        or value.get("selected_algorithm")
        not in {"identity", "logistic", "beta", "isotonic"}
        or value.get("artifact_file_sha256") != artifact_file_sha256
    ):
        raise PublicationFreezeError("selected calibrator summary is inconsistent")
    expected_usage = (
        "retained_all_development_calibrator"
        if score_policy == "retained_development_calibrator"
        else "direct_count_score"
    )
    if value.get("final_usage") != expected_usage:
        raise PublicationFreezeError("selected calibrator usage is inconsistent")
    artifact = value.get("artifact")
    definition = value.get("calibrator_definition")
    inference_features = value.get("inference_features")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("schema_version") != 3
        or artifact.get("selected_algorithm") != value.get("selected_algorithm")
        or artifact.get("payload_sha256") != value.get("artifact_payload_sha256")
        or artifact.get("calibrator") != definition
        or artifact.get("inference_features") != inference_features
        or inference_features != ["p_pre_zero"]
    ):
        raise PublicationFreezeError("selected calibrator artifact is inconsistent")
    try:
        from maskimpute.calibration import CalibrationArtifact

        calibration_artifact = CalibrationArtifact(dict(artifact))
    except (TypeError, ValueError) as error:
        raise PublicationFreezeError(
            "selected calibrator artifact is not executable"
        ) from error
    if (
        calibration_artifact.to_dict() != artifact
        or calibration_artifact.selected_algorithm != value.get("selected_algorithm")
    ):
        raise PublicationFreezeError("selected calibrator artifact is inconsistent")
    for name in (
        "artifact_payload_sha256",
        "calibrator_definition_sha256",
        "inference_features_sha256",
    ):
        _sha256(value.get(name), f"selected calibrator {name}")
    if value.get("calibrator_definition_sha256") != canonical_sha256(definition):
        raise PublicationFreezeError("selected calibrator definition checksum differs")
    if value.get("inference_features_sha256") != canonical_sha256(inference_features):
        raise PublicationFreezeError(
            "selected calibrator inference-feature checksum differs"
        )
    return _json_copy(value, "selected calibrator summary")


def _ablation_summary(
    registry: Mapping[str, object], *, artifact_file_sha256: str
) -> dict[str, object]:
    if (
        registry.get("schema_version") != 1
        or not isinstance(registry.get("reference"), Mapping)
        or type(registry.get("variants")) is not list
    ):
        raise PublicationFreezeError("ablation registry is invalid")
    reference = registry["reference"]
    variants = [
        row
        for row in registry["variants"]
        if isinstance(row, Mapping) and row.get("id") == "capacity-matched-ae"
    ]
    if reference.get("id") != "maskimpute-reference" or len(variants) != 1:
        raise PublicationFreezeError("publication ablation identities are invalid")
    capacity = variants[0]
    return {
        "registry_file_sha256": _sha256(
            artifact_file_sha256, "ablation registry artifact checksum"
        ),
        "registry_payload_sha256": canonical_sha256(registry),
        "reference_id": "maskimpute-reference",
        "reference_definition_sha256": canonical_sha256(reference),
        "capacity_matched_control_id": "capacity-matched-ae",
        "capacity_matched_definition": _json_copy(
            capacity, "capacity-matched ablation definition"
        ),
        "capacity_matched_definition_sha256": canonical_sha256(capacity),
    }


def _final_applicability(
    row: Mapping[str, object], disposition: str
) -> dict[str, object]:
    scope = row.get("execution_scope")
    if disposition == "explicit_reason_coded_unavailable":
        reason = row.get("integration_reason")
        if not isinstance(reason, str) or _REASON_CODE.fullmatch(reason) is None:
            raise PublicationFreezeError(
                f"method {row.get('id')} lacks an unavailable final reason"
            )
        return {
            "rule": "never",
            "non_run_reason": reason,
            "required_reference": None,
        }
    if scope == "same_input_required":
        return {
            "rule": "all_final_datasets",
            "non_run_reason": None,
            "required_reference": None,
        }
    if scope == "external_reference_only":
        return {
            "rule": "matched_bulk_reference_present",
            "non_run_reason": "matched_bulk_reference_absent",
            "required_reference": {
                "kind": "prespecified_matched_bulk_expression",
                "binding": "final_dataset_manifest_external_reference",
                "evaluator_truth_as_reference": "forbidden",
            },
        }
    if scope == "historical_not_run":
        return {
            "rule": "never",
            "non_run_reason": "historical_method_not_rerun",
            "required_reference": None,
        }
    if scope == "not_applicable":
        reason = row.get("applicability_reason") or row.get("integration_reason")
        if not isinstance(reason, str) or _REASON_CODE.fullmatch(reason) is None:
            raise PublicationFreezeError(
                f"method {row.get('id')} lacks a final applicability reason"
            )
        return {
            "rule": "never",
            "non_run_reason": reason,
            "required_reference": None,
        }
    raise PublicationFreezeError(f"method {row.get('id')} execution scope is invalid")


def _method_panel(
    method_registry: Mapping[str, object],
    required_ids: tuple[str, ...],
    execution_evidence: Mapping[str, object],
    runtime_lock_sha256: str,
) -> list[dict[str, object]]:
    try:
        parse_method_registry(method_registry)
    except MethodContractError as error:
        raise PublicationFreezeError(f"method registry is invalid: {error}") from error
    if (
        not isinstance(method_registry, Mapping)
        or method_registry.get("schema_version") != 1
        or type(method_registry.get("methods")) is not list
    ):
        raise PublicationFreezeError("method registry is invalid")
    methods = method_registry["methods"]
    by_id: dict[str, Mapping[str, object]] = {}
    for raw in methods:
        if not isinstance(raw, Mapping):
            raise PublicationFreezeError("method registry row is invalid")
        method_id = raw.get("id")
        if (
            not isinstance(method_id, str)
            or _SAFE_ID.fullmatch(method_id) is None
            or method_id in by_id
        ):
            raise PublicationFreezeError("method registry IDs are invalid")
        by_id[method_id] = raw
    candidates = [
        method_id for method_id, row in by_id.items() if row.get("role") == "candidate"
    ]
    if candidates != ["maskimpute"]:
        raise PublicationFreezeError(
            "method registry must contain exactly the MaskImpute candidate"
        )

    for method_id in required_ids:
        if method_id not in by_id:
            raise PublicationFreezeError(
                f"required comparator {method_id} is absent from the registry"
            )

    if not isinstance(execution_evidence, Mapping):
        raise PublicationFreezeError("development execution evidence must be a mapping")
    if set(execution_evidence) - set(by_id):
        raise PublicationFreezeError("execution evidence names an unknown method")
    runtime_digest = _sha256(runtime_lock_sha256, "runtime lock checksum")
    panel: list[dict[str, object]] = []
    for raw in methods:
        assert isinstance(raw, Mapping)
        method_id = raw["id"]
        assert isinstance(method_id, str)
        row = raw
        is_required = method_id in required_ids
        if is_required and (
            row.get("track") != "same_input"
            or row.get("execution_scope") != "same_input_required"
            or row.get("role") == "candidate"
        ):
            raise PublicationFreezeError(
                f"required comparator {method_id} is not a same-input comparator"
            )
        environment = row.get("environment")
        source = row.get("source")
        if not isinstance(environment, Mapping) or not isinstance(source, Mapping):
            raise PublicationFreezeError(
                f"required comparator {method_id} authority is incomplete"
            )
        integration_status = row.get("integration_status")
        integration_reason = row.get("integration_reason")
        execution_scope = row.get("execution_scope")
        license_spec = row.get("license")
        citation = row.get("citation")
        if not isinstance(license_spec, Mapping):
            raise PublicationFreezeError(
                f"method {method_id} lacks final license authority"
            )
        in_tree = source.get("kind") == "in_tree"
        if in_tree and license_spec.get("status") != "declared":
            raise PublicationFreezeError(
                f"method {method_id} lacks a human-approved project license"
            )
        if not in_tree and license_spec.get("status") == "pending":
            raise PublicationFreezeError(
                f"method {method_id} lacks final upstream license authority"
            )
        if not isinstance(citation, Mapping):
            raise PublicationFreezeError(f"method {method_id} lacks citation authority")
        if in_tree and (
            citation.get("status") == "pending"
            and citation.get("doi") is None
            and citation.get("url") is None
        ):
            citation_disposition = "in_tree_self_citation_no_external_doi"
        elif citation.get("status") == "verified":
            citation_disposition = (
                "verified_project_citation" if in_tree else "verified_external_citation"
            )
        else:
            qualifier = "project" if in_tree else "external"
            raise PublicationFreezeError(
                f"method {method_id} lacks final {qualifier} citation authority"
            )
        if source.get("kind") == "git":
            _git_oid(
                str(source.get("revision", "")),
                f"method {method_id} source revision",
            )
            _git_oid(
                str(source.get("tree", "")),
                f"method {method_id} source tree",
            )
        evidence = _validated_execution_evidence(
            execution_evidence.get(method_id), method_id
        )
        executable_scope = execution_scope in {
            "same_input_required",
            "external_reference_only",
        }
        expected_execution_track = (
            "external_reference"
            if execution_scope == "external_reference_only"
            else "same_input"
        )
        allowed_execution_artifacts = (
            {"external_reference_checkpoint"}
            if execution_scope == "external_reference_only"
            else (
                {
                    "base_reconstruction_checkpoint",
                    "v28_reconstruction_checkpoint",
                    "v29_reconstruction_checkpoint",
                }
                if method_id == "maskimpute"
                else {"base_reconstruction_checkpoint"}
            )
        )
        if evidence is not None and (
            evidence["execution_track"] != expected_execution_track
            or evidence["artifact"] not in allowed_execution_artifacts
        ):
            raise PublicationFreezeError(
                f"method {method_id} execution evidence uses the wrong track"
            )
        terminal_integration_status = integration_status
        terminal_integration_reason = integration_reason
        if executable_scope:
            if evidence is None:
                raise PublicationFreezeError(
                    f"method {method_id} lacks development execution evidence"
                )
            status_counts = evidence["status_counts"]
            if any(
                status in status_counts
                for status in {
                    "infrastructure_error",
                    "blocked_authority",
                    "budget_exhausted",
                }
            ):
                raise PublicationFreezeError(
                    f"method {method_id} has non-scientific incomplete development attempts"
                )
            if (
                evidence["completed_dataset_count"]
                == evidence["attempted_dataset_count"]
            ):
                if (
                    environment.get("status") != "ready"
                    or environment.get("lock_sha256") != runtime_digest
                ):
                    raise PublicationFreezeError(
                        f"completed method {method_id} is not bound to a ready runtime environment"
                    )
                if (
                    method_id in {"observed", "maskimpute"}
                    and evidence["failed_run_count"] != 0
                ):
                    raise PublicationFreezeError(
                        f"method {method_id} has failed development attempts"
                    )
                terminal_integration_status = "implemented"
                terminal_integration_reason = (
                    None
                    if method_id in {"observed", "maskimpute"}
                    else (
                        "development_execution_completed"
                        if evidence["failed_run_count"] == 0
                        else "development_execution_completed_with_failures_"
                        f"{evidence['failure_reasons_sha256'][:16]}"
                    )
                )
                disposition = (
                    "selected_candidate"
                    if row.get("role") == "candidate"
                    else (
                        "verified_runnable"
                        if evidence["failed_run_count"] == 0
                        else "verified_runnable_with_recorded_failures"
                    )
                )
            elif (
                evidence["completed_run_count"] == 0
                and evidence["failed_run_count"] == evidence["attempted_run_count"]
                and set(status_counts).issubset(_DISPOSITION_FAILURE_STATUSES)
                and evidence["failure_reason_codes"]
            ):
                terminal_integration_status = "unavailable"
                terminal_integration_reason = (
                    "technical_unavailable_development_attempts_"
                    f"{evidence['failure_reasons_sha256'][:16]}"
                )
                disposition = "explicit_reason_coded_unavailable"
            else:
                if evidence["completed_run_count"] == 0:
                    raise PublicationFreezeError(
                        f"method {method_id} lacks reason-bound failed attempt evidence"
                    )
                raise PublicationFreezeError(
                    f"method {method_id} lacks completed development dataset coverage"
                )
        elif execution_scope == "historical_not_run":
            if (
                integration_status != "historical"
                or not isinstance(integration_reason, str)
                or _REASON_CODE.fullmatch(integration_reason) is None
                or any(
                    marker in integration_reason for marker in _NONFINAL_REASON_MARKERS
                )
            ):
                raise PublicationFreezeError(
                    f"method {method_id} lacks a final historical disposition"
                )
            disposition = "historical_not_run"
        elif (
            execution_scope == "not_applicable" and integration_status == "unavailable"
        ):
            if (
                not isinstance(integration_reason, str)
                or _REASON_CODE.fullmatch(integration_reason) is None
                or any(
                    marker in integration_reason for marker in _NONFINAL_REASON_MARKERS
                )
            ):
                raise PublicationFreezeError(
                    f"method {method_id} lacks a final applicability reason"
                )
            disposition = "not_applicable"
        else:
            raise PublicationFreezeError(
                f"method {method_id} lacks a final integration disposition"
            )
        terminal_row = dict(row)
        terminal_row["integration_status"] = terminal_integration_status
        terminal_row["integration_reason"] = terminal_integration_reason
        panel.append(
            {
                "id": method_id,
                "role": row.get("role"),
                "track": row.get("track"),
                "execution_scope": execution_scope,
                "claim_required": is_required,
                "method_sha256": canonical_sha256(row),
                "source": _json_copy(source, f"{method_id} source"),
                "license": _json_copy(row.get("license"), f"{method_id} license"),
                "citation": _json_copy(row.get("citation"), f"{method_id} citation"),
                "citation_disposition": citation_disposition,
                "environment": _json_copy(environment, f"{method_id} environment"),
                "registry_integration_status": integration_status,
                "registry_integration_reason": integration_reason,
                "integration_status": terminal_integration_status,
                "integration_reason": terminal_integration_reason,
                "disposition": disposition,
                "final_applicability": _final_applicability(terminal_row, disposition),
                "development_execution_evidence": evidence,
            }
        )
    return panel


def build_frozen_method_payload(
    *,
    preparation_commit: str,
    selection_report: Mapping[str, object],
    candidate_configuration: Mapping[str, object],
    method_registry: Mapping[str, object],
    required_comparator_ids: Sequence[str],
    method_execution_evidence: Mapping[str, object],
    selected_calibrator_summary: Mapping[str, object],
    ablation_registry: Mapping[str, object],
    runtime_lock_sha256: str,
    runtime_environment_summary: Mapping[str, object],
    artifact_bindings: Mapping[str, Mapping[str, str]],
    development_stage_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Build a self-authenticating final-method payload from validated evidence."""

    prepared_at = _git_oid(preparation_commit, "preparation commit")

    if not isinstance(selection_report, Mapping):
        raise PublicationFreezeError("selection report must be an object")
    if selection_report.get("trigger") != "freeze_candidate":
        raise PublicationFreezeError(
            "selection trigger must equal freeze_candidate before publication freeze"
        )
    selected = selection_report.get("selected_configuration")
    if not isinstance(selected, str) or _SAFE_ID.fullmatch(selected) is None:
        raise PublicationFreezeError("selection report has no selected configuration")
    if not isinstance(candidate_configuration, Mapping):
        raise PublicationFreezeError("candidate configuration must be an object")
    if candidate_configuration.get("configuration_id") != selected:
        raise PublicationFreezeError(
            "candidate configuration differs from the selected configuration"
        )
    configuration = candidate_configuration.get("configuration")
    if not isinstance(configuration, Mapping):
        raise PublicationFreezeError("selected configuration payload is invalid")
    observed_configuration_sha256 = canonical_sha256(configuration)
    if (
        candidate_configuration.get("configuration_sha256")
        != observed_configuration_sha256
    ):
        raise PublicationFreezeError("selected configuration checksum mismatch")
    version = candidate_configuration.get("version")
    if not isinstance(version, str) or configuration.get("method_version") != version:
        raise PublicationFreezeError("selected configuration version mismatch")

    assessments = selection_report.get("assessments")
    if type(assessments) is not list:
        raise PublicationFreezeError("selection gate table is invalid")
    selected_assessments = [
        row
        for row in assessments
        if isinstance(row, Mapping) and row.get("configuration_id") == selected
    ]
    if len(selected_assessments) != 1:
        raise PublicationFreezeError("selected configuration assessment is not unique")
    selected_assessment = selected_assessments[0]
    if (
        selected_assessment.get("version") != version
        or selected_assessment.get("eligible") is not True
        or selected_assessment.get("efficacy_pass") is not True
        or selected_assessment.get("safety_pass") is not True
    ):
        raise PublicationFreezeError(
            "selected configuration did not pass every freeze gate"
        )
    bindings = selection_report.get("authority_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise PublicationFreezeError(
            "selection report lacks development authority bindings"
        )
    for name, digest in bindings.items():
        if not isinstance(name, str):
            raise PublicationFreezeError(
                "development authority binding name is invalid"
            )
        if name.endswith("_sha256"):
            _sha256(digest, f"development authority binding {name}")

    required_ids = tuple(required_comparator_ids)
    if (
        not required_ids
        or len(required_ids) != len(set(required_ids))
        or any(
            not isinstance(item, str) or _SAFE_ID.fullmatch(item) is None
            for item in required_ids
        )
    ):
        raise PublicationFreezeError("required comparator IDs are invalid")
    runtime_digest = _sha256(runtime_lock_sha256, "runtime lock checksum")
    runtime_summary = _json_copy(
        runtime_environment_summary, "runtime environment summary"
    )
    if (
        not isinstance(runtime_summary, dict)
        or runtime_summary.get("schema_version") != 1
        or runtime_summary.get("lock_file_sha256") != runtime_digest
        or not isinstance(runtime_summary.get("environment_inventory_sha256s"), dict)
        or not runtime_summary["environment_inventory_sha256s"]
    ):
        raise PublicationFreezeError("runtime environment summary is invalid")
    for environment_id, inventory_sha256 in runtime_summary[
        "environment_inventory_sha256s"
    ].items():
        if (
            not isinstance(environment_id, str)
            or _SAFE_ID.fullmatch(environment_id) is None
        ):
            raise PublicationFreezeError("runtime environment summary ID is invalid")
        _sha256(
            inventory_sha256,
            f"runtime environment {environment_id} inventory checksum",
        )
    artifacts = _artifact_bindings(artifact_bindings)
    stage_layout = _layout_from_development_stage_receipt(development_stage_receipt)
    stage_receipt = _validate_development_stage_receipt(
        development_stage_receipt,
        stage_layout,
        artifact_bindings,
    )
    stage_rows = stage_receipt.get("stages")
    if not isinstance(stage_rows, list) or not stage_rows:
        raise PublicationFreezeError("development stage receipt is incomplete")
    active_stage_row = stage_rows[-1]
    if not isinstance(active_stage_row, Mapping) or _sha256(
        bindings.get("development_result_sha256"),
        "active selection development result",
    ) != active_stage_row.get("complete_result_sha256"):
        raise PublicationFreezeError(
            "active stage complete result differs from selection authority"
        )
    expected_selected_version = (
        "v27" if stage_layout.active_stage == "base" else stage_layout.active_stage
    )
    if version != expected_selected_version:
        raise PublicationFreezeError(
            "selected configuration version differs from development stage receipt"
        )
    for method_id, evidence in method_execution_evidence.items():
        if isinstance(evidence, Mapping):
            artifact_name = evidence.get("artifact")
            if artifact_name not in artifacts:
                raise PublicationFreezeError(
                    f"method {method_id} execution artifact is not bound"
                )
    maskimpute_evidence = method_execution_evidence.get("maskimpute")
    expected_candidate_artifact = (
        f"{stage_layout.active_stage}_reconstruction_checkpoint"
    )
    if (
        not isinstance(maskimpute_evidence, Mapping)
        or maskimpute_evidence.get("artifact") != expected_candidate_artifact
    ):
        raise PublicationFreezeError(
            "MaskImpute execution evidence differs from the selected stage"
        )
    runtime_binding = artifacts.get("runtime_lock")
    if runtime_binding is not None and runtime_binding["sha256"] != runtime_digest:
        raise PublicationFreezeError("runtime lock artifact checksum is inconsistent")
    calibration_binding = artifacts.get("retained_calibration")
    if not isinstance(calibration_binding, Mapping):
        raise PublicationFreezeError("retained calibration artifact binding is absent")
    calibrator_summary = _validate_selected_calibrator_summary(
        selected_calibrator_summary,
        score_policy=configuration.get("score_policy"),
        artifact_file_sha256=calibration_binding["sha256"],
    )
    ablation_binding = artifacts.get("ablation_registry")
    if not isinstance(ablation_binding, Mapping):
        raise PublicationFreezeError("ablation registry artifact binding is absent")
    ablations = _ablation_summary(
        ablation_registry, artifact_file_sha256=ablation_binding["sha256"]
    )

    unsigned: dict[str, object] = {
        "schema_version": 1,
        "preparation_commit": prepared_at,
        "candidate_method_id": "maskimpute",
        "selected_configuration_id": selected,
        "selected_version": version,
        "selected_configuration": _json_copy(configuration, "selected configuration"),
        "selected_configuration_sha256": observed_configuration_sha256,
        "selection_trigger": "freeze_candidate",
        "selection_rule": selection_report.get("selection_rule"),
        "pareto_set": _json_copy(selection_report.get("pareto_set"), "Pareto set"),
        "selection_gate_table": _json_copy(assessments, "selection gate table"),
        "selected_assessment": _json_copy(
            selected_assessment, "selected configuration assessment"
        ),
        "development_authority_bindings": _json_copy(
            bindings, "development authority bindings"
        ),
        "development_stage_receipt": stage_receipt,
        "required_comparator_ids": list(required_ids),
        "method_denominator": _method_panel(
            method_registry,
            required_ids,
            method_execution_evidence,
            runtime_digest,
        ),
        "method_registry_sha256": canonical_sha256(method_registry),
        "runtime_lock_sha256": runtime_digest,
        "runtime_environment_summary": runtime_summary,
        "selected_calibrator": calibrator_summary,
        "selected_ablation_control": ablations,
        "artifact_bindings": artifacts,
        "correlation_gene_panel_rule": {
            "id": "all-retained-genes-v1",
            "selection": "all_genes_after_only_zero_library_cell_exclusion",
            "gene_filtering": "forbidden",
            "shared_across_methods": True,
        },
    }
    return {**unsigned, "payload_sha256": canonical_sha256(unsigned)}


def _recompute_selection_report(
    repository: Path, selection_input: dict[str, object]
) -> dict[str, object]:
    from .selection import _select_for_repository

    return _select_for_repository(
        selection_input, repository, require_clean=True
    ).to_dict()


def _candidate_configuration(
    repository: Path,
    selected: str,
    development_search: Mapping[str, object],
    revision_versions: Sequence[str],
) -> dict[str, object]:
    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if not isinstance(selected, str) or _SAFE_ID.fullmatch(selected) is None:
        raise PublicationFreezeError("selected configuration identity is invalid")
    versions = tuple(revision_versions)
    if versions not in {(), ("v28",), ("v28", "v29")}:
        raise PublicationFreezeError(
            "selected configuration revision prefix is incomplete or reordered"
        )
    candidates: list[dict[str, object]] = []
    rows = development_search.get("configurations")
    if type(rows) is not list:
        raise PublicationFreezeError("development search configurations are invalid")
    authority_bindings: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PublicationFreezeError(
                "development search configuration row is invalid"
            )
        configuration_id = row.get("configuration_id")
        configuration_sha256 = row.get("configuration_sha256")
        if (
            not isinstance(configuration_id, str)
            or _SAFE_ID.fullmatch(configuration_id) is None
            or not isinstance(configuration_sha256, str)
            or _SHA256.fullmatch(configuration_sha256) is None
            or configuration_id in authority_bindings
        ):
            raise PublicationFreezeError(
                "development search configuration identities are invalid or duplicated"
            )
        authority_bindings[configuration_id] = configuration_sha256
        if configuration_id == selected:
            candidates.append(
                {
                    key: _json_copy(row[key], f"selected configuration {key}")
                    for key in (
                        "configuration_id",
                        "version",
                        "configuration",
                        "configuration_sha256",
                    )
                }
            )
    try:
        for version in versions:
            spec = load_revision_spec(repository, version, require_clean=True)
            if (
                authority_bindings.get(spec.parent_configuration_id)
                != spec.parent_configuration_sha256
            ):
                raise PublicationFreezeError(
                    f"{version} revision parent binding differs"
                )
            if spec.configuration_id in authority_bindings:
                raise PublicationFreezeError(
                    "revision configuration identity collides with tracked authority"
                )
            authority_bindings[spec.configuration_id] = spec.configuration_sha256
            if spec.configuration_id == selected:
                candidates.append(
                    {
                        "configuration_id": spec.configuration_id,
                        "version": spec.version,
                        "configuration": thaw_revision_configuration(spec),
                        "configuration_sha256": spec.configuration_sha256,
                    }
                )
    except RevisionAuthorityError as error:
        raise PublicationFreezeError(
            f"tracked revision configuration is invalid: {error}"
        ) from error
    if len(candidates) != 1:
        raise PublicationFreezeError(
            "selected configuration is not uniquely present in tracked authority"
        )
    return candidates[0]


def _expected_frozen_method(
    repository: Path, preparation_commit: str
) -> dict[str, object]:
    layout = _resolve_publication_stage(repository)
    payloads: dict[str, dict[str, Any]] = {}
    artifact_bindings: dict[str, dict[str, str]] = {}
    for name, relative in {
        **_COMMON_TRACKED_PATHS,
        **_COMMON_DEVELOPMENT_PATHS,
    }.items():
        payload, digest = _secure_json(repository / relative, name.replace("_", " "))
        payloads[name] = payload
        artifact_bindings[name] = {"path": relative, "sha256": digest}
    stage_evidence, recomputed = _validate_publication_stage_evidence(
        repository,
        layout,
    )
    method_rows = payloads["method_registry"].get("methods")
    if type(method_rows) is not list:
        raise PublicationFreezeError("method registry is invalid")
    external_method_ids = tuple(
        sorted(
            row["id"]
            for row in method_rows
            if isinstance(row, Mapping)
            and row.get("execution_scope") == "external_reference_only"
            and isinstance(row.get("id"), str)
        )
    )
    artifact_paths = _publication_artifact_paths(
        layout,
        include_external_reference=bool(external_method_ids),
    )
    for name, relative in artifact_paths.items():
        if name in payloads or name == "external_reference_checkpoint":
            continue
        payload, digest = _secure_json(repository / relative, name.replace("_", " "))
        payloads[name] = payload
        artifact_bindings[name] = {"path": relative, "sha256": digest}
    _validate_saver_package_authority(payloads, artifact_bindings)
    external_validation = None
    if external_method_ids:
        try:
            external_validation = load_external_reference_evidence(repository)
        except (
            ExternalReferenceDevelopmentError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise PublicationFreezeError(
                f"external-reference production evidence is invalid: {error}"
            ) from error
        if tuple(external_validation.method_ids) != external_method_ids:
            raise PublicationFreezeError(
                "external-reference production denominator differs from method registry"
            )
        try:
            external_relative = external_validation.checkpoint_path.relative_to(
                repository
            ).as_posix()
        except ValueError as error:
            raise PublicationFreezeError(
                "external-reference checkpoint escaped the repository"
            ) from error
        if external_relative != _EXTERNAL_REFERENCE_CHECKPOINT_PATH:
            raise PublicationFreezeError(
                "external-reference checkpoint path is not fixed"
            )
        payloads["external_reference_checkpoint"] = dict(external_validation.checkpoint)
        artifact_bindings["external_reference_checkpoint"] = {
            "path": external_relative,
            "sha256": external_validation.checkpoint_file_sha256,
        }
    _validated_publication_artifact_bindings(layout, artifact_bindings)
    for paths, retained in zip(layout.stages, stage_evidence.stages):
        prefix = paths.stage
        expected_core = (
            (
                f"{prefix}_selection_source_input",
                _thaw_receipt_json(retained.source_input),
                retained.source_input_file_sha256,
            ),
            (
                f"{prefix}_selection_complete_input",
                _thaw_receipt_json(retained.complete_input),
                retained.complete_input_file_sha256,
            ),
            (
                f"{prefix}_selection_report",
                _thaw_receipt_json(retained.report),
                retained.report_file_sha256,
            ),
        )
        for name, expected_payload, expected_digest in expected_core:
            if (
                payloads.get(name) != expected_payload
                or artifact_bindings.get(name, {}).get("sha256") != expected_digest
            ):
                raise PublicationFreezeError(
                    f"validated {prefix} stage differs from the artifact inventory"
                )
        if (
            artifact_bindings[f"{prefix}_downstream_manifest"]["sha256"]
            != retained.downstream_manifest_file_sha256
        ):
            raise PublicationFreezeError(
                f"validated {prefix} downstream manifest differs from inventory"
            )
    selected = recomputed.get("selected_configuration")
    if not isinstance(selected, str):
        raise PublicationFreezeError("selection report has no selected configuration")
    configuration = _candidate_configuration(
        repository,
        selected,
        payloads["development_search"],
        layout.revision_versions,
    )
    required = payloads["selection_contract"].get("required_comparator_ids")
    if type(required) is not list:
        raise PublicationFreezeError("selection contract comparators are invalid")
    runtime_summary = _runtime_environment_summary(
        repository / _COMMON_TRACKED_PATHS["runtime_lock"],
        artifact_bindings["runtime_lock"]["sha256"],
        payloads["method_registry"],
    )
    development_dataset_ids = _development_dataset_ids(payloads["dataset_status"])
    execution_evidence = _active_execution_evidence(
        payloads["base_reconstruction_checkpoint"],
        active_stage=layout.active_stage,
        selected_stage_checkpoint=(
            None
            if layout.active_stage == "base"
            else payloads[f"{layout.active_stage}_reconstruction_checkpoint"]
        ),
        selected_configuration=configuration,
        eligible_dataset_ids=development_dataset_ids,
    )
    if external_method_ids:
        if external_validation is None:  # pragma: no cover - guarded above
            raise PublicationFreezeError(
                "external-reference production validation was not retained"
            )
        external_evidence = _development_execution_evidence(
            payloads["external_reference_checkpoint"],
            artifact="external_reference_checkpoint",
            execution_track="external_reference",
            eligible_dataset_ids=(external_validation.dataset_id,),
        )
        if set(external_evidence) != set(external_method_ids):
            raise PublicationFreezeError(
                "external-reference execution checkpoint has an incomplete method denominator"
            )
        if set(execution_evidence).intersection(external_evidence):
            raise PublicationFreezeError(
                "development execution methods occur in more than one track"
            )
        execution_evidence.update(external_evidence)
    selected_calibrator = _selected_calibrator_summary(
        payloads["retained_calibration"],
        artifact_file_sha256=artifact_bindings["retained_calibration"]["sha256"],
        score_policy=configuration["configuration"].get("score_policy"),
    )
    stage_receipt = _build_development_stage_receipt(
        repository,
        layout,
        stage_evidence,
        artifact_bindings,
    )
    result = build_frozen_method_payload(
        preparation_commit=preparation_commit,
        selection_report=recomputed,
        candidate_configuration=configuration,
        method_registry=payloads["method_registry"],
        required_comparator_ids=required,
        method_execution_evidence=execution_evidence,
        selected_calibrator_summary=selected_calibrator,
        ablation_registry=payloads["ablation_registry"],
        runtime_lock_sha256=artifact_bindings["runtime_lock"]["sha256"],
        runtime_environment_summary=runtime_summary,
        artifact_bindings=artifact_bindings,
        development_stage_receipt=stage_receipt,
    )
    if _resolve_publication_stage(repository) != layout:
        raise PublicationFreezeError(
            "publication stage changed during freeze preparation"
        )
    stage_evidence_after, report_after = _validate_publication_stage_evidence(
        repository,
        layout,
    )
    if stage_evidence_after != stage_evidence or report_after != recomputed:
        raise PublicationFreezeError(
            "publication stage evidence changed during freeze preparation"
        )
    for name, relative in artifact_paths.items():
        if name == "external_reference_checkpoint":
            continue
        payload, digest = _secure_json(repository / relative, name.replace("_", " "))
        if payload != payloads[name] or digest != artifact_bindings[name]["sha256"]:
            raise PublicationFreezeError(
                f"fixed publication evidence changed during freeze preparation: {name}"
            )
    if external_method_ids:
        try:
            external_after = load_external_reference_evidence(repository)
        except (
            ExternalReferenceDevelopmentError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise PublicationFreezeError(
                f"external-reference production evidence changed: {error}"
            ) from error
        if (
            dict(external_after.checkpoint) != payloads["external_reference_checkpoint"]
            or external_after.checkpoint_file_sha256
            != artifact_bindings["external_reference_checkpoint"]["sha256"]
            or external_validation is None
            or external_after.dataset_id != external_validation.dataset_id
            or external_after.method_ids != external_validation.method_ids
        ):
            raise PublicationFreezeError(
                "fixed publication evidence changed during freeze preparation: "
                "external_reference_checkpoint"
            )
    if (
        _build_development_stage_receipt(
            repository,
            layout,
            stage_evidence_after,
            artifact_bindings,
        )
        != stage_receipt
    ):
        raise PublicationFreezeError(
            "publication stage inventory changed during freeze preparation"
        )
    return result


def _clean_preparation_commit(repository: Path) -> str:
    """Bind preparation to a recursively stable tracked tree at HEAD."""

    try:
        commit = _git(repository, "rev-parse", "HEAD")
        if not _GIT_OID.fullmatch(commit):
            raise PublicationFreezeError("preparation HEAD is not a Git object ID")
        if not _raw_tracked_files_match_index(repository):
            raise PublicationFreezeError(
                "publication preparation requires tracked bytes matching the index"
            )
        status = _git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        allowed_untracked = f"?? {_FROZEN_METHOD_PATH}"
        dirty = [line for line in status.splitlines() if line != allowed_untracked]
        if dirty:
            raise PublicationFreezeError(
                "publication preparation requires a clean executable repository"
            )
        submodules = _git(repository, "submodule", "status", "--recursive")
        if any(line and line[0] != " " for line in submodules.splitlines()):
            raise PublicationFreezeError(
                "publication preparation requires clean initialized submodules"
            )
        return commit
    except PublicationFreezeError:
        raise
    except Exception as error:
        raise PublicationFreezeError(
            "publication preparation requires the canonical Git repository"
        ) from error


def prepare_frozen_method(repository: Path) -> dict[str, object]:
    """Recompute development selection and materialize the fixed tracked config."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    try:
        selected_repository = repository.resolve(strict=True)
    except OSError as error:
        raise PublicationFreezeError("repository is unavailable") from error
    preparation_commit = _clean_preparation_commit(selected_repository)
    expected = _expected_frozen_method(selected_repository, preparation_commit)
    if _clean_preparation_commit(selected_repository) != preparation_commit:
        raise PublicationFreezeError(
            "publication repository changed during freeze preparation"
        )
    path = selected_repository / _FROZEN_METHOD_PATH
    raw = _canonical_bytes(expected)
    if path.exists():
        observed, digest = _secure_json(path, "frozen method")
        if observed != expected or digest != hashlib.sha256(raw).hexdigest():
            raise PublicationFreezeError(
                "existing frozen method differs from recomputed development evidence"
            )
        return expected
    _atomic_write(path, raw)
    return expected


def _validate_clean_frozen_method(
    repository: Path,
) -> tuple[dict[str, object], dict[str, str]]:
    observed, digest = _secure_json(repository / _FROZEN_METHOD_PATH, "frozen method")
    if digest != hashlib.sha256(_canonical_bytes(observed)).hexdigest():
        raise PublicationFreezeError("frozen method is not canonical JSON")
    raw_artifacts = observed.get("artifact_bindings")
    if not isinstance(raw_artifacts, Mapping):
        raise PublicationFreezeError("frozen method artifact bindings are invalid")
    stage_receipt_value = observed.get("development_stage_receipt")
    layout = _layout_from_development_stage_receipt(stage_receipt_value)
    artifacts = _validated_publication_artifact_bindings(layout, raw_artifacts)
    stage_receipt = _validate_development_stage_receipt(
        stage_receipt_value,
        layout,
        raw_artifacts,
    )

    tracked_payloads: dict[str, dict[str, Any]] = {}
    for name, relative in _COMMON_TRACKED_PATHS.items():
        payload, file_digest = _secure_json(
            repository / relative, name.replace("_", " ")
        )
        if file_digest != artifacts[name]["sha256"]:
            raise PublicationFreezeError(
                f"tracked publication authority differs from frozen receipt: {name}"
            )
        tracked_payloads[name] = payload
    _validate_saver_package_authority(tracked_payloads, artifacts)

    selected = observed.get("selected_configuration_id")
    if not isinstance(selected, str):
        raise PublicationFreezeError("frozen method selection is invalid")
    configuration = _candidate_configuration(
        repository,
        selected,
        tracked_payloads["development_search"],
        layout.revision_versions,
    )
    required = tracked_payloads["selection_contract"].get("required_comparator_ids")
    if type(required) is not list:
        raise PublicationFreezeError("selection contract comparators are invalid")
    method_rows = tracked_payloads["method_registry"].get("methods")
    if type(method_rows) is not list:
        raise PublicationFreezeError("method registry is invalid")
    has_external_methods = any(
        isinstance(row, Mapping)
        and row.get("execution_scope") == "external_reference_only"
        for row in method_rows
    )
    if has_external_methods != ("external_reference_checkpoint" in artifacts):
        raise PublicationFreezeError(
            "frozen method external-reference artifact membership differs"
        )
    selection_report = {
        "trigger": observed.get("selection_trigger"),
        "selected_configuration": selected,
        "assessments": observed.get("selection_gate_table"),
        "authority_bindings": observed.get("development_authority_bindings"),
        "selection_rule": observed.get("selection_rule"),
        "pareto_set": observed.get("pareto_set"),
    }
    runtime_summary = _runtime_environment_summary(
        repository / _COMMON_TRACKED_PATHS["runtime_lock"],
        artifacts["runtime_lock"]["sha256"],
        tracked_payloads["method_registry"],
    )
    frozen_denominator = observed.get("method_denominator")
    if type(frozen_denominator) is not list:
        raise PublicationFreezeError("frozen method denominator is invalid")
    execution_evidence = {
        row["id"]: row["development_execution_evidence"]
        for row in frozen_denominator
        if isinstance(row, Mapping)
        and isinstance(row.get("id"), str)
        and row.get("development_execution_evidence") is not None
    }
    rebuilt = build_frozen_method_payload(
        preparation_commit=_git_oid(
            observed.get("preparation_commit"), "frozen preparation commit"
        ),
        selection_report=selection_report,
        candidate_configuration=configuration,
        method_registry=tracked_payloads["method_registry"],
        required_comparator_ids=required,
        method_execution_evidence=execution_evidence,
        selected_calibrator_summary=observed.get("selected_calibrator"),
        ablation_registry=tracked_payloads["ablation_registry"],
        runtime_lock_sha256=artifacts["runtime_lock"]["sha256"],
        runtime_environment_summary=runtime_summary,
        artifact_bindings=artifacts,
        development_stage_receipt=stage_receipt,
    )
    if observed != rebuilt:
        raise PublicationFreezeError(
            "frozen method differs from its commit-bound tracked authorities"
        )
    for name, relative in _COMMON_TRACKED_PATHS.items():
        payload, file_digest = _secure_json(
            repository / relative, name.replace("_", " ")
        )
        if (
            payload != tracked_payloads[name]
            or file_digest != artifacts[name]["sha256"]
        ):
            raise PublicationFreezeError(
                f"tracked publication authority changed during validation: {name}"
            )
    return observed, {
        "config_sha256": digest,
        "protocol_sha256": artifacts["protocol"]["sha256"],
        "environment_sha256": artifacts["runtime_lock"]["sha256"],
    }


def validate_frozen_method(repository: Path) -> dict[str, object]:
    """Validate one committed receipt without reopening ignored development files."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    try:
        selected_repository = repository.resolve(strict=True)
    except OSError as error:
        raise PublicationFreezeError("repository is unavailable") from error
    observed, _hashes = _validate_clean_frozen_method(selected_repository)
    return observed


def _validate_development_evidence_package(
    repository: Path,
    payload: Mapping[str, object],
    operational_roots: Sequence[Path],
) -> str:
    raw_artifacts = payload.get("artifact_bindings")
    if not isinstance(raw_artifacts, Mapping):
        raise PublicationFreezeError("frozen method artifact bindings are invalid")
    stage_receipt_value = payload.get("development_stage_receipt")
    expected_layout = _layout_from_development_stage_receipt(stage_receipt_value)
    artifacts = _validated_publication_artifact_bindings(
        expected_layout,
        raw_artifacts,
    )
    expected_stage_receipt = _validate_development_stage_receipt(
        stage_receipt_value,
        expected_layout,
        raw_artifacts,
    )
    raw_names = set(_COMMON_DEVELOPMENT_PATHS) | set(
        _stage_artifact_names(expected_layout)
    )
    if "external_reference_checkpoint" in artifacts:
        raw_names.add("external_reference_checkpoint")

    def validate_files() -> None:
        for name in sorted(raw_names):
            row = artifacts[name]
            assert isinstance(row, Mapping)
            relative = row["path"]
            assert isinstance(relative, str)
            try:
                _value, observed_sha256 = _secure_json(
                    repository / relative, name.replace("_", " ")
                )
            except PublicationFreezeError as error:
                raise PublicationFreezeError(
                    f"raw development evidence is unavailable: {name}"
                ) from error
            if observed_sha256 != artifacts[name]["sha256"]:
                raise PublicationFreezeError(
                    f"raw development evidence differs from frozen receipt: {name}"
                )

    def validate_stage() -> None:
        observed_layout = _resolve_publication_stage(repository)
        if observed_layout != expected_layout:
            raise PublicationFreezeError(
                "raw publication stage differs from the frozen receipt"
            )
        evidence, active_report = _validate_publication_stage_evidence(
            repository,
            observed_layout,
        )
        if active_report.get("selected_configuration") != payload.get(
            "selected_configuration_id"
        ):
            raise PublicationFreezeError(
                "raw active selection differs from the frozen receipt"
            )
        rebuilt = _build_development_stage_receipt(
            repository,
            observed_layout,
            evidence,
            raw_artifacts,
        )
        if rebuilt != expected_stage_receipt:
            raise PublicationFreezeError(
                "raw development stage inventory differs from the frozen receipt"
            )

    def validate_external() -> None:
        if "external_reference_checkpoint" not in artifacts:
            return
        try:
            external = load_external_reference_evidence(repository)
        except (
            ExternalReferenceDevelopmentError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise PublicationFreezeError(
                f"raw external-reference evidence is invalid: {error}"
            ) from error
        external_binding = artifacts["external_reference_checkpoint"]
        assert isinstance(external_binding, Mapping)
        if (
            external.checkpoint_path != repository / _EXTERNAL_REFERENCE_CHECKPOINT_PATH
            or external.checkpoint_file_sha256 != external_binding.get("sha256")
        ):
            raise PublicationFreezeError(
                "raw external-reference evidence differs from the frozen receipt"
            )

    validate_files()
    validate_stage()
    validate_external()
    try:
        receipts = _operational_root_receipts(repository, operational_roots)
    except (OSError, StudyStateError, TypeError, ValueError) as error:
        raise PublicationFreezeError(
            "publication operational evidence package is invalid"
        ) from error
    validate_files()
    validate_stage()
    validate_external()
    return canonical_sha256(receipts)


def freeze_publication_round(repository: Path, round_dir: Path) -> dict[str, object]:
    """Validate the selected method and delegate to the sealed round controller."""

    if not isinstance(round_dir, Path):
        raise TypeError("round_dir must be a pathlib.Path")
    selected_repository = repository.resolve(strict=True)
    expected_commit = _git(selected_repository, "rev-parse", "HEAD")
    payload, hashes = _validate_clean_frozen_method(selected_repository)
    preparation_commit = _git_oid(
        payload.get("preparation_commit"), "frozen preparation commit"
    )
    parents = _git(
        selected_repository,
        "rev-list",
        "--parents",
        "-n",
        "1",
        expected_commit,
    ).split()
    changed_paths = _git(
        selected_repository,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        expected_commit,
    ).splitlines()
    if (
        len(parents) != 2
        or parents[0] != expected_commit
        or parents[1] != preparation_commit
        or changed_paths != [_FROZEN_METHOD_PATH]
    ):
        raise PublicationFreezeError(
            "frozen method must be the sole change in a direct preparation commit"
        )
    if _git(selected_repository, "rev-parse", "HEAD") != expected_commit:
        raise PublicationFreezeError("publication HEAD changed during validation")
    operational_roots = tuple(
        selected_repository / relative
        for relative in _PUBLICATION_OPERATIONAL_ROOTS
        if os.path.lexists(selected_repository / relative)
    )
    operational_roots_sha256 = _validate_development_evidence_package(
        selected_repository, payload, operational_roots
    )
    return freeze_round(
        selected_repository,
        round_dir,
        selected_repository / _FROZEN_METHOD_PATH,
        selected_repository / _COMMON_TRACKED_PATHS["protocol"],
        environment_path=(selected_repository / _COMMON_TRACKED_PATHS["runtime_lock"]),
        expected_config_sha256=hashes["config_sha256"],
        expected_protocol_sha256=hashes["protocol_sha256"],
        expected_environment_sha256=hashes["environment_sha256"],
        expected_method_commit=expected_commit,
        operational_artifact_roots=operational_roots,
        expected_operational_artifact_roots_sha256=operational_roots_sha256,
    )


__all__ = [
    "PublicationFreezeError",
    "build_frozen_method_payload",
    "freeze_publication_round",
    "prepare_frozen_method",
    "validate_frozen_method",
]
