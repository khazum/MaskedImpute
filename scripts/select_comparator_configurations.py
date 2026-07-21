#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maskimpute_benchmark.comparator_tuning import (  # noqa: E402
    ComparatorTuningError,
    publish_comparator_selection,
)


def main() -> int:
    argparse.ArgumentParser(
        description="Select one fixed development-only configuration per comparator."
    ).parse_args()
    try:
        receipt = publish_comparator_selection(ROOT)
    except (ComparatorTuningError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "path": (
                    "artifacts/study/development/evaluation/comparator_selection.json"
                ),
                "readiness": receipt["readiness"]["status"],
                "selected_method_count": sum(
                    row["selection_status"] == "selected"
                    for row in receipt["methods"].values()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
