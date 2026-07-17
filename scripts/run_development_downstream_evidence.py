#!/usr/bin/env python3
"""Run or resume fixed downstream evidence over the development checkpoint."""

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
    build_development_downstream_evidence_plan,
    run_downstream_evidence,
)


OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "artifacts/study/development/evaluation/downstream"
)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Run or resume the fixed eight-endpoint downstream stage over the "
            "sealed development reconstruction checkpoint. Scientific paths, "
            "denominators, and procedures are repository-owned."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        plan = build_development_downstream_evidence_plan(REPOSITORY_ROOT)
        result = run_downstream_evidence(plan, OUTPUT_DIRECTORY)
    except (DownstreamEvidenceError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
