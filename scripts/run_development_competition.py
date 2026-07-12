#!/usr/bin/env python3
"""Run or resume the fixed, tracked development reconstruction competition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.runner import (  # noqa: E402
    RunnerContractError,
    run_development_competition,
)


def _environment(value: str) -> tuple[str, Path]:
    method_id, separator, raw_path = value.partition("=")
    if not separator or not method_id or not raw_path:
        raise argparse.ArgumentTypeError(
            "environment must be METHOD=/path/to/executable"
        )
    return method_id, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repository-owned 16-dataset development competition; scientific "
            "design, configurations, seeds, QC, metrics, and budgets are not CLI options."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts/study/development/competition-reconstruction",
        help="checkpoint/artifact directory; an existing exact checkpoint is resumed",
    )
    parser.add_argument(
        "--environment",
        action="append",
        default=[],
        type=_environment,
        metavar="METHOD=EXECUTABLE",
        help="explicit adapter executable path; source pins remain registry-owned",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    environments: dict[str, Path] = {}
    for method_id, executable in arguments.environment:
        if method_id in environments:
            _parser().error(f"duplicate --environment for {method_id}")
        environments[method_id] = executable
    try:
        report = run_development_competition(
            arguments.output_dir,
            environment_overrides=environments,
        )
    except (RunnerContractError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    summary = {
        "checkpoint_path": str(arguments.output_dir.absolute() / "checkpoint.json"),
        "checkpoint_sha256": report.checkpoint_sha256,
        "evaluation_scope": report.evaluation_scope,
        "planned_run_count": report.planned_run_count,
        "recorded_run_count": len(report.records),
        "selection_blockers": list(report.selection_blockers),
        "selection_complete": report.selection_complete,
        "status": report.status,
    }
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
