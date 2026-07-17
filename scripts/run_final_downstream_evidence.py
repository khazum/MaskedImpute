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
    run_downstream_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the fixed eight-endpoint downstream stage over a "
            "sealed frozen final execution."
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


def _external_output_directory(plan: object) -> Path:
    binding = getattr(plan, "evaluated_round_binding", None)
    round_id = getattr(binding, "round_id", None)
    receipt_sha256 = getattr(
        binding, "evaluation_receipt_payload_sha256", None
    )
    if (
        not isinstance(round_id, str)
        or not round_id
        or not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise DownstreamEvidenceError(
            "final downstream plan lacks an evaluated-round receipt binding"
        )
    return (
        REPOSITORY_ROOT.parent
        / f"{REPOSITORY_ROOT.name}-final-analysis"
        / "downstream"
        / round_id
        / receipt_sha256
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    round_directory = (
        arguments.round_dir
        if arguments.round_dir.is_absolute()
        else REPOSITORY_ROOT / arguments.round_dir
    )
    try:
        plan = build_final_downstream_evidence_plan(
            REPOSITORY_ROOT, round_directory
        )
        output_directory = _external_output_directory(plan)
        result = run_downstream_evidence(plan, output_directory)
    except (DownstreamEvidenceError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
