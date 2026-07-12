#!/usr/bin/env python3
"""Build the fixed, evidence-bound development selection input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.development_evaluation import (  # noqa: E402
    DevelopmentEvaluationError,
    build_development_selection_input,
)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Build the fixed development selection artifact from the repository-owned "
            "reconstruction checkpoint, real-source receipts, and orthogonal outputs. "
            "Scientific design and paths are not command-line options."
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        result_path, evaluation_path = build_development_selection_input(
            REPOSITORY_ROOT
        )
    except (DevelopmentEvaluationError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "evaluation_manifest_path": str(evaluation_path),
                "evaluation_manifest_file_sha256": _sha256(evaluation_path),
                "selection_input_path": str(result_path),
                "selection_input_file_sha256": _sha256(result_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
