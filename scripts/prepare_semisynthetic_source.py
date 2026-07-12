#!/usr/bin/env python3
"""Prepare a canonical, seed-free Baron partition-fit receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.protocol import canonical_sha256  # noqa: E402
from maskimpute_benchmark.simulators.base import (  # noqa: E402
    SimulationContractError,
)
from maskimpute_benchmark.simulators.semisynthetic import (  # noqa: E402
    _verify_semisynthetic_source,
    prepare_source_summary,
)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0 or str(parsed) != value:
        raise argparse.ArgumentTypeError("must be a canonical positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and seal a seed-free Gamma-Poisson source summary for one "
            "donor-disjoint Baron namespace."
        )
    )
    parser.add_argument("--namespace", choices=("dev", "final"), required=True)
    parser.add_argument("--genes", type=_positive_integer, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _reject_symlink_components(path: Path) -> None:
    for component in [path.absolute(), *path.absolute().parents]:
        if not os.path.lexists(component):
            continue
        try:
            metadata = component.lstat()
        except OSError as error:
            raise SimulationContractError(
                "prepared-summary output path cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise SimulationContractError(
                "prepared-summary output path must not contain symlinks"
            )


def _publish_new_json(path: Path, value: object) -> None:
    if path.name in {"", ".", ".."}:
        raise SimulationContractError("prepared-summary output must name a file")
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path)
    if os.path.lexists(path):
        raise SimulationContractError(
            "prepared-summary output already exists; refusing to overwrite"
        )
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        if linked:
            try:
                path.unlink()
            except OSError:
                pass
        raise SimulationContractError(
            "prepared-summary output could not be atomically published"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    archive, receipt = _verify_semisynthetic_source()
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise SimulationContractError(
            "semisynthetic source receipt must bind exactly one archive"
        )
    artifact = artifacts[0]
    if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str):
        raise SimulationContractError(
            "semisynthetic source receipt archive checksum is invalid"
        )
    fit = prepare_source_summary(
        archive,
        arguments.namespace,
        arguments.genes,
        expected_sha256=artifact["sha256"],
    )
    output = {
        "schema_version": 1,
        "namespace": arguments.namespace,
        "genes": arguments.genes,
        "source_artifact": artifact,
        "source_receipt": receipt,
        "source_receipt_sha256": canonical_sha256(receipt),
        "fit": fit,
    }
    _publish_new_json(arguments.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
