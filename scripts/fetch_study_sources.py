#!/usr/bin/env python3
"""Fetch exact publication-study source pins into an external ignored root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.sources import (  # noqa: E402
    SourceLedgerError,
    fetch_sources,
    load_source_ledger,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch immutable study sources without changing their pins."
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPOSITORY_ROOT / "study" / "sources.json",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="caller-supplied external path, or a path ignored by its Git worktree",
    )
    parser.add_argument(
        "--source",
        dest="source_ids",
        action="append",
        help="source id to fetch (repeatable; default: every ledger source)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        ledger = load_source_ledger(arguments.ledger)
        receipts = fetch_sources(
            ledger, arguments.root, source_ids=arguments.source_ids
        )
    except SourceLedgerError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(list(receipts), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
