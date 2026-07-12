#!/usr/bin/env python3
"""Fit a canonical development-only pre-zero calibration artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute.calibration import (  # noqa: E402
    fit_development_calibration,
    load_calibration_records,
    save_calibration_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit leakage-safe LODO calibration from canonical exact-truth "
            "development records."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        records = load_calibration_records(arguments.input)
        artifact = fit_development_calibration(records)
        save_calibration_artifact(arguments.output, artifact)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps({"error": str(exc)}, allow_nan=False, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    payload = artifact.to_dict()
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "payload_sha256": payload["payload_sha256"],
                "selected_algorithm": artifact.selected_algorithm,
                "training_record_digest_sha256": payload["training"][
                    "record_digest_sha256"
                ],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
