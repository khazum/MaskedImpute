#!/usr/bin/env python3
"""Manage frozen one-use publication benchmark rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


_ENTRYPOINT_DONT_WRITE_BYTECODE = sys.dont_write_bytecode
if __name__ == "__main__":
    sys.dont_write_bytecode = True


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from maskimpute_benchmark.study import (  # noqa: E402
    StudyStateError,
    assert_final_runnable,
    freeze_round,
    materialize_final,
    record_final_evaluation,
    supersede_round,
)


def _load_result_manifest(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StudyStateError(f"result manifest is invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise StudyStateError("result manifest must be a JSON object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze, materialize, verify, consume, or supersede a study round."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="freeze a clean Git commit and inputs")
    freeze.add_argument("round_dir", type=Path)
    freeze.add_argument("config_path", type=Path)
    freeze.add_argument("protocol_path", type=Path)
    freeze.add_argument(
        "--environment",
        type=Path,
        required=True,
        help="tracked environment lock or container digest manifest",
    )
    freeze.add_argument("--repo", type=Path, default=Path.cwd())
    freeze.set_defaults(
        action=lambda args: freeze_round(
            args.repo,
            args.round_dir,
            args.config_path,
            args.protocol_path,
            environment_path=args.environment,
        )
    )

    materialize = commands.add_parser(
        "materialize-final", help="create a unique final generator-seed manifest"
    )
    materialize.add_argument("round_dir", type=Path)
    materialize.add_argument("--seed-count", type=int, required=True)
    materialize.add_argument("--repo", type=Path, default=Path.cwd())
    materialize.set_defaults(
        action=lambda args: materialize_final(
            args.round_dir, args.seed_count, repo=args.repo
        )
    )

    verify = commands.add_parser(
        "verify-final",
        help="verify frozen bindings and atomically claim the one permitted run",
    )
    verify.add_argument("round_dir", type=Path)
    verify.add_argument("--repo", type=Path, default=Path.cwd())
    verify.set_defaults(
        action=lambda args: assert_final_runnable(args.repo, args.round_dir)
    )

    evaluate = commands.add_parser(
        "record-evaluation", help="consume a final round with an exclusive receipt"
    )
    evaluate.add_argument("round_dir", type=Path)
    evaluate.add_argument("result_manifest", type=Path)
    evaluate.add_argument("--repo", type=Path, default=Path.cwd())
    evaluate.set_defaults(
        action=lambda args: record_final_evaluation(
            args.round_dir,
            _load_result_manifest(args.result_manifest),
            repo=args.repo,
        )
    )

    supersede = commands.add_parser(
        "supersede", help="archive a round as superseded without deleting evidence"
    )
    supersede.add_argument("round_dir", type=Path)
    supersede.add_argument("reason")
    supersede.set_defaults(
        action=lambda args: supersede_round(args.round_dir, args.reason)
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        result = arguments.action(arguments)
    except StudyStateError as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
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
