#!/usr/bin/env python3
"""Run or resume frozen-final null differential-expression evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.final_null_de import (  # noqa: E402
    FinalNullDEError,
    build_final_null_de_plan,
    expected_final_null_de_output_directory,
    run_final_null_de_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the receipt-bound null-DE safety analysis over one "
            "evaluated frozen final round."
        )
    )
    parser.add_argument(
        "--round-dir",
        type=Path,
        required=True,
        help=(
            "evaluated frozen final round inside the active repository "
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
        plan = build_final_null_de_plan(REPOSITORY_ROOT, round_directory)
        output_directory = expected_final_null_de_output_directory(plan)
        result = run_final_null_de_evidence(plan, output_directory)
    except (FinalNullDEError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
