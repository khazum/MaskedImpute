#!/usr/bin/env python3
"""Atomically bind validated development score and calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.selection import (  # noqa: E402
    finalize_development_artifact_bindings,
)


def _finalize() -> dict[str, str]:
    return dict(finalize_development_artifact_bindings())


def main() -> int:
    result = _finalize()
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
