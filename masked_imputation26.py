#!/usr/bin/env python3
"""Migration-only compatibility entry point for the retired v26 script."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


ARCHIVED_IMPLEMENTATION = Path(
    "historical/v26_neurips/code/masked_imputation26.py"
)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "This retired v26 entry point is migration guidance only. "
            "Use the canonical package API: maskimpute.impute_counts together "
            "with maskimpute.p_pre_zero_from_counts."
        ),
        epilog=(
            "The preserved implementation is under historical/v26_neurips. "
            "This wrapper is not a publication runner and never imports or "
            "executes archived model code."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    parser.parse_args(argv)
    parser.print_usage(sys.stderr)
    print(
        "Execution through this retired entry point is disabled; use the "
        "canonical maskimpute package API.",
        file=sys.stderr,
    )
    print(
        f"Archived development source: {ARCHIVED_IMPLEMENTATION.as_posix()}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
