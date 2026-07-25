#!/usr/bin/env python3
"""Prepare the selected method and freeze one fixed publication round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


_ENTRYPOINT_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
if __name__ == "__main__":
    sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.publication_freeze import (  # noqa: E402
    PublicationFreezeError,
    freeze_publication_round,
    prepare_frozen_method,
)
from maskimpute_benchmark.study import StudyStateError  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the fixed development selection or freeze it without "
            "caller-supplied method, hyperparameter, protocol, or environment paths."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="write study/frozen_method.json from fixed development evidence"
    )
    prepare.set_defaults(action=lambda _args: prepare_frozen_method(REPOSITORY_ROOT))

    freeze = commands.add_parser(
        "freeze", help="freeze a clean commit containing the prepared method"
    )
    freeze.add_argument("round_dir", type=Path)
    freeze.set_defaults(
        action=lambda args: freeze_publication_round(REPOSITORY_ROOT, args.round_dir)
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = arguments.action(arguments)
    except (PublicationFreezeError, StudyStateError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    previous_state = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return _main(argv)
    finally:
        sys.dont_write_bytecode = previous_state


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        sys.dont_write_bytecode = _ENTRYPOINT_DONT_WRITE_BYTECODE
