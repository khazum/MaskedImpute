#!/usr/bin/env python3
"""Generate canonical analysis from one evaluated frozen-final round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from maskimpute_benchmark.final_analysis import generate_final_analysis  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the immutable, evaluated frozen-final benchmark. "
            "The evidence, methods, metrics, inference unit, and analysis policy "
            "are fixed."
        )
    )
    parser.add_argument(
        "round_dir",
        type=Path,
        help="evaluated round directory (absolute or repository-relative)",
    )
    args = parser.parse_args()
    round_dir = (
        args.round_dir
        if args.round_dir.is_absolute()
        else REPOSITORY / args.round_dir
    )
    report = generate_final_analysis(REPOSITORY, round_dir)
    print(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
