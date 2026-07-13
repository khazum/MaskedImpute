#!/usr/bin/env python3
"""Write an exclusive exact package-inventory lock for selected runtimes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Literal, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.runtime_environments import (  # noqa: E402
    RuntimeEnvironmentError,
    build_runtime_environment_lock,
)


def _environment(value: str) -> tuple[str, Literal["python", "r"], Path]:
    environment_id, separator, remainder = value.partition("=")
    kind, second_separator, raw_path = remainder.partition("=")
    if (
        not separator
        or not second_separator
        or not environment_id
        or kind not in {"python", "r"}
        or not raw_path
    ):
        raise argparse.ArgumentTypeError(
            "environment must be ID=python=/path or ID=r=/path"
        )
    return environment_id, kind, Path(raw_path)


def _r_library(value: str) -> tuple[str, Path]:
    environment_id, separator, raw_path = value.partition("=")
    if not separator or not environment_id or not raw_path:
        raise argparse.ArgumentTypeError("R library must be ID=/path")
    return environment_id, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe selected runtime executables and write a canonical lock."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--environment",
        action="append",
        required=True,
        type=_environment,
        metavar="ID=KIND=EXECUTABLE",
    )
    parser.add_argument(
        "--r-library",
        action="append",
        default=[],
        type=_r_library,
        metavar="ID=PATH",
        help="selected isolated library path for an R environment; repeat in order",
    )
    return parser


def _write_exclusive(path: Path, value: object) -> None:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    environments: dict[str, tuple[Literal["python", "r"], Path]] = {}
    for environment_id, kind, executable in arguments.environment:
        if environment_id in environments:
            _parser().error(f"duplicate environment ID {environment_id}")
        environments[environment_id] = (kind, executable)
    r_libraries: dict[str, list[Path]] = {}
    for environment_id, path in arguments.r_library:
        r_libraries.setdefault(environment_id, []).append(path)
    try:
        lock = build_runtime_environment_lock(
            environments, r_library_paths=r_libraries
        )
        _write_exclusive(arguments.output, lock)
    except (OSError, RuntimeEnvironmentError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "environment_count": len(environments),
                "output": str(arguments.output.absolute()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
