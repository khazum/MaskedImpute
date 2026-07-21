#!/usr/bin/env python3
"""Apply the prespecified non-compensatory development selection gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.selection import (  # noqa: E402
    SelectionAuthorityError,
    _select_for_repository,
    select_development_candidate,
)
from maskimpute_benchmark.selection_promotion import (  # noqa: E402
    SelectionPromotionError,
    _immutable_publish,
    _secure_canonical_json,
)


SELECTION_INPUT_PATH = (
    REPOSITORY_ROOT / "artifacts/study/development/evaluation/"
    "development_selection_input-downstream.json"
)
SELECTION_REPORT_PATH = (
    REPOSITORY_ROOT
    / "artifacts/study/development/evaluation/development_selection_report.json"
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _exact(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} has missing or extra fields")
    return value


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if type(payload) is not dict:
        raise ValueError("selection input must be an object")
    schema_version = payload.get("schema_version")
    base_fields = {
        "schema_version",
        "records",
        "orthogonal_intervals",
        "dataset_manifest_sha256",
        "count_score_manifest_sha256",
        "retained_calibration_artifact_sha256",
        "evaluation_manifest_sha256",
        "comparator_selection",
        "result_sha256",
    }
    expected_fields = (
        base_fields
        if schema_version == 2
        else {*base_fields, "revision_versions"}
        if schema_version == 3
        else {*base_fields, "revision_versions", "downstream_evidence"}
        if schema_version == 4
        else base_fields
    )
    root = _exact(
        payload,
        expected_fields,
        "selection input",
    )
    if (
        root["schema_version"] not in {2, 3, 4}
        or type(root["schema_version"]) is not int
    ):
        raise ValueError("selection input schema_version must equal 2, 3, or 4")
    return root


def _report(payload: dict[str, Any], repository: Path | None = None) -> dict[str, Any]:
    report = (
        select_development_candidate(payload)
        if repository is None
        else _select_for_repository(payload, repository, require_clean=True)
    )
    return report.to_dict()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Apply the fixed development gates to the immutable base "
            "selection-complete input."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    try:
        payload, _input_file_sha256 = _secure_canonical_json(
            SELECTION_INPUT_PATH,
            "base selection-complete input",
        )
        if payload.get("schema_version") != 4:
            raise ValueError(
                "fixed selection input must be selection-complete schema 4"
            )
        report = _report(payload, REPOSITORY_ROOT)
        encoded = _canonical_bytes(report)
        _immutable_publish(SELECTION_REPORT_PATH, encoded)
    except (
        OSError,
        SelectionAuthorityError,
        SelectionPromotionError,
        TypeError,
        ValueError,
    ) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(encoded.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
