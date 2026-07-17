#!/usr/bin/env python3
"""Run the fixed D3Impute/scTsI Tung external-reference development track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.external_reference_development import (  # noqa: E402
    ExternalReferenceDevelopmentError,
    run_external_reference_development,
)


_METHOD_IDS = frozenset({"d3impute", "sctsi"})


def _environment(value: str) -> tuple[str, Path]:
    method_id, separator, raw_path = value.partition("=")
    if not separator or not method_id or not raw_path:
        raise argparse.ArgumentTypeError(
            "environment must be METHOD=/absolute/path/to/executable"
        )
    return method_id, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed development-only D3Impute and scTsI adapters on the "
            "canonical Tung iPSC single-cell and measured-bulk source. Scientific "
            "design, sources, methods, parameters, seeds, endpoints, and output "
            "location are repository-owned and are not CLI options."
        )
    )
    parser.add_argument(
        "--environment",
        action="append",
        required=True,
        type=_environment,
        metavar="METHOD=EXECUTABLE",
        help=(
            "absolute non-symlink executable locator; repeat exactly once for "
            "d3impute and sctsi"
        ),
    )
    parser.add_argument(
        "--sctsi-library",
        required=True,
        type=Path,
        metavar="DIRECTORY",
        help="absolute non-symlink path to the locked isolated scTsI R library",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    environments: dict[str, Path] = {}
    for method_id, executable in arguments.environment:
        if method_id not in _METHOD_IDS:
            parser.error(f"unknown external-reference method {method_id!r}")
        if method_id in environments:
            parser.error(f"duplicate --environment for {method_id}")
        environments[method_id] = executable
    missing = sorted(_METHOD_IDS - set(environments))
    if missing:
        parser.error("missing --environment for " + ", ".join(missing))
    try:
        evidence = run_external_reference_development(
            REPOSITORY_ROOT,
            environments=environments,
            sctsi_library=arguments.sctsi_library,
        )
    except (ExternalReferenceDevelopmentError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    status_counts: dict[str, int] = {}
    for record in evidence.checkpoint["records"]:
        status = str(record["run"]["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    print(
        json.dumps(
            {
                "checkpoint_path": str(evidence.checkpoint_path),
                "checkpoint_file_sha256": evidence.checkpoint_file_sha256,
                "dataset_id": evidence.dataset_id,
                "method_ids": list(evidence.method_ids),
                "status": evidence.checkpoint["status"],
                "status_counts": status_counts,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
