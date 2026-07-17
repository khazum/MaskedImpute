#!/usr/bin/env python3
"""Run or resume fixed downstream evidence over the active checkpoint bundle."""

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
    development_downstream_revision_version,
    run_downstream_evidence,
)


OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/study/development/evaluation"


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Run or resume the fixed eight-endpoint downstream stage over the "
            "sealed base plus any consecutive activated revision checkpoints. "
            "Scientific paths, denominators, and procedures are repository-owned."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        through_version = development_downstream_revision_version(REPOSITORY_ROOT)
        plan = build_development_downstream_evidence_plan(
            REPOSITORY_ROOT,
            through_version=through_version,
        )
        suffix = "" if through_version is None else f"-{through_version}"
        result = run_downstream_evidence(
            plan,
            OUTPUT_ROOT / f"downstream{suffix}",
        )
    except (DownstreamEvidenceError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
