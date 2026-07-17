#!/usr/bin/env python3
"""Build the immutable combined selection input through v29."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.revision_commands import build_revision_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(build_revision_main("v29", REPOSITORY_ROOT))
