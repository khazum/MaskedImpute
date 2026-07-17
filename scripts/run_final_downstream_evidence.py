#!/usr/bin/env python3
"""Run or resume fixed downstream evidence over one frozen final round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.downstream_evidence import (  # noqa: E402
    DownstreamEvidenceError,
    build_final_downstream_evidence_plan,
    build_final_trajectory_downstream_evidence_plan,
    expected_final_downstream_output_directory,
    run_downstream_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the fixed primary and supplementary-trajectory "
            "downstream stages over a sealed frozen final execution."
        )
    )
    parser.add_argument(
        "--round-dir",
        type=Path,
        required=True,
        help=(
            "frozen final round directory inside the active repository "
            "(absolute or repository-relative)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    round_directory = (
        arguments.round_dir
        if arguments.round_dir.is_absolute()
        else REPOSITORY_ROOT / arguments.round_dir
    )
    try:
        primary_plan = build_final_downstream_evidence_plan(
            REPOSITORY_ROOT, round_directory
        )
        trajectory_plan = build_final_trajectory_downstream_evidence_plan(
            REPOSITORY_ROOT, round_directory
        )
        primary_output = expected_final_downstream_output_directory(primary_plan)
        trajectory_output = expected_final_downstream_output_directory(trajectory_plan)
        primary_result = run_downstream_evidence(primary_plan, primary_output)
        trajectory_result = run_downstream_evidence(trajectory_plan, trajectory_output)
    except (DownstreamEvidenceError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    status = (
        "completed"
        if primary_result.get("status") == "completed"
        and trajectory_result.get("status") == "completed"
        else "running"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "primary": primary_result,
        "trajectory": trajectory_result,
    }
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
