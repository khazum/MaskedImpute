#!/usr/bin/env python3
"""Execute or resume the single frozen final publication round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from maskimpute_benchmark.final_runner import run_frozen_final_round  # noqa: E402
from maskimpute_benchmark.operational_environment import (  # noqa: E402
    establish_supported_final_runtime_environment,
)


def main() -> int:
    establish_supported_final_runtime_environment()
    parser = argparse.ArgumentParser(
        description=(
            "Execute/resume the committed frozen method on the claimed final panel. "
            "The repository, methods, environments, seeds, and design are fixed."
        )
    )
    parser.add_argument(
        "round_dir",
        type=Path,
        help="round directory inside artifacts/study (absolute or repository-relative)",
    )
    parser.add_argument(
        "--simulator-assets-root",
        type=Path,
        required=True,
        help="explicit external simulator source/data root",
    )
    parser.add_argument(
        "--simulator-r-environment",
        type=Path,
        required=True,
        help="explicit pinned simulator R environment",
    )
    args = parser.parse_args()
    round_dir = (
        args.round_dir if args.round_dir.is_absolute() else REPOSITORY / args.round_dir
    )
    result = run_frozen_final_round(
        REPOSITORY,
        round_dir,
        simulator_assets_root=args.simulator_assets_root,
        simulator_r_environment=args.simulator_r_environment,
    )
    print(
        json.dumps(
            {
                "execution_manifest_sha256": result["execution_manifest"][
                    "manifest_sha256"
                ],
                "evaluation_receipt_state": result["evaluation_receipt"]["state"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
