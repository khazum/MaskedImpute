"""Fixed command entry points for conditional development revisions.

The version is supplied by a tracked wrapper script, never by a command-line
argument.  Scientific configuration, evidence paths, and output paths therefore
remain repository authority rather than operator choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


_VERSIONS = ("v28", "v29")


def _require_version(version: str) -> str:
    if version not in _VERSIONS:
        raise ValueError("revision version must be v28 or v29")
    return version


def _environment(value: str) -> tuple[str, Path]:
    method_id, separator, raw_path = value.partition("=")
    if not separator or not method_id or not raw_path:
        raise argparse.ArgumentTypeError(
            "environment must be METHOD=/path/to/executable"
        )
    return method_id, Path(raw_path)


def _parser(description: str, *, environments: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    if environments:
        parser.add_argument(
            "--environment",
            action="append",
            default=[],
            type=_environment,
            metavar="METHOD=EXECUTABLE",
            help="explicit adapter executable; tracked source pins remain authoritative",
        )
    return parser


def run_revision_main(
    version: str,
    repository: Path,
    argv: Sequence[str] | None = None,
) -> int:
    """Run one activated revision at its exact repository-owned directory."""

    version = _require_version(version)
    parser = _parser(
        f"Run/resume the fixed {version} development revision competition.",
        environments=True,
    )
    arguments = parser.parse_args(argv)
    environments: dict[str, Path] = {}
    for method_id, executable in arguments.environment:
        if method_id in environments:
            parser.error(f"duplicate --environment for {method_id}")
        environments[method_id] = executable

    from .revisions import revision_stage_paths
    from .runner import (
        RunnerContractError,
        run_v28_revision_competition,
        run_v29_revision_competition,
    )

    runner = (
        run_v28_revision_competition
        if version == "v28"
        else run_v29_revision_competition
    )
    try:
        report = runner(environment_overrides=environments)
    except (RunnerContractError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    checkpoint = (
        repository.absolute()
        / revision_stage_paths(version).reconstruction_directory
        / "checkpoint.json"
    )
    print(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": report.checkpoint_sha256,
                "evaluation_scope": report.evaluation_scope,
                "planned_run_count": report.planned_run_count,
                "recorded_run_count": len(report.records),
                "selection_blockers": list(report.selection_blockers),
                "selection_complete": report.selection_complete,
                "status": report.status,
                "version": version,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0 if report.status == "completed" else 1


def build_revision_main(
    version: str,
    repository: Path,
    argv: Sequence[str] | None = None,
) -> int:
    """Build exact combined evidence through one activated revision."""

    version = _require_version(version)
    _parser(
        f"Build the fixed combined development selection evidence through {version}."
    ).parse_args(argv)
    from .revision_evaluation import (
        RevisionEvaluationError,
        build_revision_selection_input,
    )

    try:
        result_path, evaluation_path = build_revision_selection_input(
            repository,
            version,
        )
    except (RevisionEvaluationError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "evaluation_manifest_path": str(evaluation_path),
                "evaluation_manifest_file_sha256": hashlib.sha256(
                    evaluation_path.read_bytes()
                ).hexdigest(),
                "selection_input_path": str(result_path),
                "selection_input_file_sha256": hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest(),
                "version": version,
            },
            sort_keys=True,
        )
    )
    return 0


def select_revision_main(
    version: str,
    repository: Path,
    argv: Sequence[str] | None = None,
) -> int:
    """Select from one exact combined input and publish its immutable report."""

    version = _require_version(version)
    _parser(
        f"Apply the fixed development selection gates through {version}."
    ).parse_args(argv)
    from .development_evaluation import _canonical_json_bytes
    from .revisions import RevisionAuthorityError, revision_stage_paths
    from .selection import SelectionAuthorityError, _select_for_repository
    from .selection_promotion import (
        SelectionPromotionError,
        _immutable_publish,
        _secure_canonical_json,
    )

    paths = revision_stage_paths(version)
    try:
        payload, _input_sha256 = _secure_canonical_json(
            repository.absolute() / paths.selection_complete_input,
            f"{version} revision selection input",
        )
        expected_versions = ["v28"] if version == "v28" else ["v28", "v29"]
        if (
            payload.get("schema_version") != 4
            or payload.get("revision_versions") != expected_versions
        ):
            raise SelectionAuthorityError(
                f"{version} fixed selection input is not selection-complete schema 4"
            )
        report = _select_for_repository(
            payload,
            repository.absolute(),
            require_clean=True,
        ).to_dict()
        encoded = _canonical_json_bytes(report) + b"\n"
        report_path = repository.absolute() / paths.selection_report
        report_sha256 = _immutable_publish(report_path, encoded)
    except (
        RevisionAuthorityError,
        SelectionAuthorityError,
        SelectionPromotionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "selection_report_path": str(report_path),
                "selection_report_file_sha256": report_sha256,
                "selected_configuration": report["selected_configuration"],
                "trigger": report["trigger"],
                "version": version,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "build_revision_main",
    "run_revision_main",
    "select_revision_main",
]
