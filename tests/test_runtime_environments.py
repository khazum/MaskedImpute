from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from maskimpute_benchmark.runtime_environments import (
    RuntimeEnvironmentError,
    build_runtime_environment_lock,
    load_runtime_environment_lock,
    probe_python_environment,
    probe_r_environment,
    validate_runtime_environment_lock,
)


def _write_canonical(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _isolated_python(tmp_path: Path) -> Path:
    environment = tmp_path / "isolated-python"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )
    return environment / "bin/python"


def test_python_probe_is_deterministic_and_package_sorted() -> None:
    first = probe_python_environment(Path(sys.executable))
    second = probe_python_environment(Path(sys.executable))

    assert first == second
    assert first["schema"] == "maskimpute-python-runtime-inventory-v1"
    assert first["interpreter"]["implementation"] == sys.implementation.name
    packages = first["packages"]
    assert packages == sorted(
        packages,
        key=lambda value: (value["name"], value["precedence"], value["version"]),
    )
    assert all(
        set(package)
        == {"name", "version", "content_sha256", "file_count", "precedence"}
        for package in packages
    )


def test_python_probe_executes_symlinked_virtual_environment_launcher(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "isolated"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        check=True,
    )

    inventory = probe_python_environment(environment / "bin/python")

    assert inventory["interpreter"]["is_virtual_environment"] is True
    assert inventory["packages"] == []
    assert inventory["launcher"]["kind"] == "symlink"


def test_r_probe_binds_selected_method_library_bytes(tmp_path: Path) -> None:
    rscript = shutil.which("Rscript")
    r = shutil.which("R")
    if rscript is None or r is None:
        pytest.skip("R is unavailable")
    source = tmp_path / "source"
    library = tmp_path / "library"
    (source / "R").mkdir(parents=True)
    library.mkdir()
    (source / "DESCRIPTION").write_text(
        "Package: runtimeLockFixture\n"
        "Version: 1.0.0\n"
        "Title: Runtime Lock Fixture\n"
        "Description: Minimal package used to test isolated library binding.\n"
        "Author: Test Fixture\n"
        "Maintainer: Test Fixture <fixture@example.org>\n"
        "License: MIT\n",
        encoding="utf-8",
    )
    (source / "NAMESPACE").write_text("export(runtime_lock_fixture)\n", encoding="utf-8")
    (source / "R/function.R").write_text(
        "runtime_lock_fixture <- function() 1L\n", encoding="utf-8"
    )
    subprocess.run(
        [r, "CMD", "INSTALL", f"--library={library}", str(source)],
        check=True,
        capture_output=True,
    )

    base = probe_r_environment(Path(rscript))
    selected = probe_r_environment(Path(rscript), library_paths=(library,))

    assert base != selected
    package = next(
        value for value in selected["packages"] if value["name"] == "runtimelockfixture"
    )
    assert package["version"] == "1.0.0"
    assert package["file_count"] > 0


def test_lock_round_trip_and_exact_runtime_validation(tmp_path: Path) -> None:
    python = _isolated_python(tmp_path)
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", python)}
    )
    path = tmp_path / "runtime-lock.json"
    _write_canonical(path, lock)

    loaded = load_runtime_environment_lock(path)
    receipt = validate_runtime_environment_lock(
        loaded,
        {"benchmark": ("python", python)},
    )

    assert receipt["lock_file_sha256"] == loaded.file_sha256
    assert receipt["environment_inventory_sha256s"] == (
        ("benchmark", loaded.by_id("benchmark").inventory_sha256),
    )


def test_rehashed_inventory_tamper_cannot_validate(tmp_path: Path) -> None:
    python = _isolated_python(tmp_path)
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", python)}
    )
    lock["environments"][0]["inventory"]["interpreter"]["version"][2] += 1
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
            {"benchmark": ("python", python)},
        )


def test_lock_rejects_missing_or_extra_runtime(tmp_path: Path) -> None:
    python = _isolated_python(tmp_path)
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", python)}
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
                "benchmark": ("python", python),
                "extra": ("python", python),
            },
        )


def test_lock_loader_rejects_noncanonical_and_duplicate_ids(tmp_path: Path) -> None:
    python = _isolated_python(tmp_path)
    lock = build_runtime_environment_lock(
        {"benchmark": ("python", python)}
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
    python = _isolated_python(tmp_path)
    output = tmp_path / "runtime-lock.json"
    command = [
        sys.executable,
        "scripts/lock_runtime_environments.py",
        "--output",
        str(output),
        "--environment",
        f"benchmark=python={python}",
    ]

    first = subprocess.run(command, check=False, capture_output=True, text=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 2
    loaded = load_runtime_environment_lock(output)
    assert loaded.by_id("benchmark").kind == "python"
