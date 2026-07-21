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
    run_comparator_tuning_smoke,
)


def main() -> int:
    argparse.ArgumentParser(
        description="Run the fixed truth-free 34-configuration comparator smoke gate."
    ).parse_args()
    try:
        receipt = run_comparator_tuning_smoke(ROOT)
    except (ComparatorTuningError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "path": (
                    "artifacts/study/development/evaluation/comparator_smoke.json"
                ),
                "status": receipt["status"],
                "planned_configuration_count": receipt[
                    "planned_configuration_count"
                ],
                "completed_configuration_count": receipt[
                    "completed_configuration_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
