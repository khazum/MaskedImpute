#!/usr/bin/env python3
"""Apply selection to the immutable combined v28 evidence."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.revision_commands import select_revision_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(select_revision_main("v28", REPOSITORY_ROOT))
