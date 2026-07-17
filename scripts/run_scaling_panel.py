#!/usr/bin/env python3
"""Execute or resume the frozen publication resource-scaling panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from maskimpute_benchmark.scaling import (  # noqa: E402
    ScalingContractError,
    run_scaling_panel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run/resume the tracked 10k/25k/50k/100k SymSim scaling panel. "
            "Mechanism, sizes, methods, seeds, metrics, and retention are not CLI options."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY / "artifacts/study/scaling",
        help="checkpoint root; an existing exact checkpoint is resumed",
    )
    arguments = parser.parse_args()
    output_dir = (
        arguments.output_dir
        if arguments.output_dir.is_absolute()
        else REPOSITORY / arguments.output_dir
    )
    try:
        report = run_scaling_panel(REPOSITORY, output_dir)
    except (OSError, ScalingContractError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "checkpoint_path": str(output_dir.absolute() / "checkpoint.json"),
                "checkpoint_sha256": report.checkpoint_sha256,
                "dataset_count": len(report.datasets),
                "planned_run_count": report.planned_run_count,
                "recorded_run_count": len(report.records),
                "status": report.status,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0 if report.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
