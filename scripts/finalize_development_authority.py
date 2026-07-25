#!/usr/bin/env python3
"""Atomically bind validated development score and calibration artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.selection import (  # noqa: E402
    SelectionAuthorityError,
    finalize_development_artifact_bindings,
)


def _finalize() -> dict[str, str]:
    return dict(finalize_development_artifact_bindings())


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Validate the fixed development score and calibration artifacts, "
            "then bind them in the tracked development authority."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        result = _finalize()
    except (SelectionAuthorityError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
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
