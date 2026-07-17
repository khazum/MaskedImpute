#!/usr/bin/env python3
"""Generate the fixed development panel or one claimed final panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.datasets import (  # noqa: E402
    DatasetRegistryError,
    generate_dataset_panel,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the immutable two-view publication dataset panel."
    )
    parser.add_argument("--namespace", choices=("dev", "final"), required=True)
    parser.add_argument("--repo", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--round-dir",
        type=Path,
        help="already-claimed canonical study round; required only for final",
    )
    parser.add_argument(
        "--simulator-assets-root",
        type=Path,
        help=(
            "explicit external source/data root outside the repository; "
            "required only for final"
        ),
    )
    parser.add_argument(
        "--simulator-r-environment",
        type=Path,
        help=(
            "explicit pinned simulator R environment outside the repository; "
            "required only for final"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.namespace == "final" and arguments.round_dir is None:
        _parser().error("--round-dir is required for --namespace final")
    if arguments.namespace == "dev" and arguments.round_dir is not None:
        _parser().error("--round-dir is not accepted for --namespace dev")
    runtime_paths = (
        arguments.simulator_assets_root,
        arguments.simulator_r_environment,
    )
    if arguments.namespace == "final" and any(value is None for value in runtime_paths):
        _parser().error(
            "--simulator-assets-root and --simulator-r-environment are required "
            "for --namespace final"
        )
    if arguments.namespace == "dev" and any(
        value is not None for value in runtime_paths
    ):
        _parser().error(
            "simulator runtime path overrides are not accepted for --namespace dev"
        )
    try:
        status = generate_dataset_panel(
            repo=arguments.repo,
            namespace=arguments.namespace,
            round_dir=arguments.round_dir,
            simulator_assets_root=arguments.simulator_assets_root,
            simulator_r_environment=arguments.simulator_r_environment,
        )
    except DatasetRegistryError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(status, allow_nan=False, sort_keys=True))
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
