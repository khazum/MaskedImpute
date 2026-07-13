from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from maskimpute_benchmark.runtime_environments import (
    RuntimeEnvironmentError,
    build_runtime_environment_lock,
    load_runtime_environment_lock,
    probe_python_environment,
    validate_runtime_environment_lock,
)


def _write_canonical(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def test_python_probe_is_deterministic_and_package_sorted() -> None:
    first = probe_python_environment(Path(sys.executable))
    second = probe_python_environment(Path(sys.executable))

    assert first == second
    assert first["schema"] == "maskimpute-python-runtime-inventory-v1"
    assert first["interpreter"]["implementation"] == sys.implementation.name
    packages = first["packages"]
    assert packages == sorted(packages, key=lambda value: (value["name"], value["version"]))
    assert all(set(package) == {"name", "version"} for package in packages)


def test_lock_round_trip_and_exact_runtime_validation(tmp_path: Path) -> None:
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", Path(sys.executable))}
    )
    path = tmp_path / "runtime-lock.json"
    _write_canonical(path, lock)

    loaded = load_runtime_environment_lock(path)
    receipt = validate_runtime_environment_lock(
        loaded,
        {"benchmark": ("python", Path(sys.executable))},
    )

    assert receipt["lock_file_sha256"] == loaded.file_sha256
    assert receipt["environment_inventory_sha256s"] == (
        ("benchmark", loaded.by_id("benchmark").inventory_sha256),
    )


def test_rehashed_package_tamper_cannot_validate(tmp_path: Path) -> None:
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", Path(sys.executable))}
    )
    lock["environments"][0]["inventory"]["packages"][0]["version"] += ".tampered"
    # Rehash every public envelope field an attacker can rewrite.
    from maskimpute_benchmark.protocol import canonical_sha256

    inventory = lock["environments"][0]["inventory"]
    lock["environments"][0]["inventory_sha256"] = canonical_sha256(inventory)
    path = tmp_path / "runtime-lock.json"
    _write_canonical(path, lock)

    loaded = load_runtime_environment_lock(path)
    with pytest.raises(RuntimeEnvironmentError, match="inventory mismatch"):
        validate_runtime_environment_lock(
            loaded,
            {"benchmark": ("python", Path(sys.executable))},
        )


def test_lock_rejects_missing_or_extra_runtime(tmp_path: Path) -> None:
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", Path(sys.executable))}
    )
    path = tmp_path / "runtime-lock.json"
    _write_canonical(path, lock)
    loaded = load_runtime_environment_lock(path)

    with pytest.raises(RuntimeEnvironmentError, match="runtime IDs mismatch"):
        validate_runtime_environment_lock(loaded, {})
    with pytest.raises(RuntimeEnvironmentError, match="runtime IDs mismatch"):
        validate_runtime_environment_lock(
            loaded,
            {
                "benchmark": ("python", Path(sys.executable)),
                "extra": ("python", Path(sys.executable)),
            },
        )


def test_lock_loader_rejects_noncanonical_and_duplicate_ids(tmp_path: Path) -> None:
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", Path(sys.executable))}
    )
    path = tmp_path / "runtime-lock.json"
    path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    with pytest.raises(RuntimeEnvironmentError, match="canonical JSON"):
        load_runtime_environment_lock(path)

    lock["environments"].append(dict(lock["environments"][0]))
    _write_canonical(path, lock)
    with pytest.raises(RuntimeEnvironmentError, match="duplicate environment ID"):
        load_runtime_environment_lock(path)


def test_lock_cli_writes_exclusive_canonical_manifest(tmp_path: Path) -> None:
    output = tmp_path / "runtime-lock.json"
    command = [
        sys.executable,
        "scripts/lock_runtime_environments.py",
        "--output",
        str(output),
        "--environment",
        f"benchmark=python={sys.executable}",
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 2
    loaded = load_runtime_environment_lock(output)
    assert loaded.by_id("benchmark").kind == "python"
