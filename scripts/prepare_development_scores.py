#!/usr/bin/env python3
"""Prepare and byte-validate the canonical development score authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.development_scores import (  # noqa: E402
    DevelopmentScorePreparationError,
    prepare_development_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the completed 16-view development panel and prepare its "
            "canonical cross-fitted count scores and calibration artifact."
        )
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root (default: the root containing this script)",
    )
    arguments = parser.parse_args()
    try:
        result = prepare_development_scores(arguments.repository)
    except DevelopmentScorePreparationError as error:
        parser.exit(2, f"development score preparation failed: {error}\n")
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
