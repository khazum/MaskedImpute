#!/usr/bin/env python3
"""Promote the latest fixed development selection stage to bound schema 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.selection_promotion import (  # noqa: E402
    SelectionPromotionError,
    promote_latest_development_selection_input,
)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Validate the latest fixed development downstream archive and "
            "immutably publish its repository-owned selection-complete input."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        receipt = promote_latest_development_selection_input(REPOSITORY_ROOT)
    except (SelectionPromotionError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(receipt.to_dict(), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
