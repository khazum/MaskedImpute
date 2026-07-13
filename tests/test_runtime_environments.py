from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys

import pytest

import maskimpute_benchmark.runtime_environments as runtime_module
from maskimpute_benchmark.runtime_environments import (
    RuntimeChangeMonitor,
    RuntimeEnvironmentEntry,
    RuntimeEnvironmentError,
    RuntimeEnvironmentLock,
    _directory_content_sha256,
    _runtime_file_content_sha256,
    _runtime_root_identity_sha256,
    build_runtime_environment_lock,
    load_runtime_environment_lock,
    probe_python_environment,
    probe_r_environment,
    runtime_environment_snapshot,
    validate_runtime_environment_lock,
    verify_runtime_environment_snapshot,
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
    root_roles = {value["role"] for value in first["runtime_roots"]}
    assert any(role.startswith("native-dependency-") for role in root_roles)
    if Path("/proc/driver/nvidia/version").is_file():
        assert {"gpu-driver-version", "nvidia-smi-executable"} <= root_roles


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


def test_python_probe_does_not_import_metadata_in_target_interpreter(
    tmp_path: Path,
) -> None:
    target = _isolated_python(tmp_path)
    wrapper = tmp_path / "python-without-metadata"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "code = sys.argv[sys.argv.index('-c') + 1]\n"
        "if 'importlib.metadata' in code or 'importlib_metadata' in code:\n"
        "    raise SystemExit(97)\n"
        f"os.execv({str(target)!r}, [{str(target)!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    inventory = probe_python_environment(wrapper)

    assert inventory["interpreter"]["implementation"] == "cpython"


def test_python_snapshot_binds_wrapper_target_interpreter(
    tmp_path: Path,
) -> None:
    target = _isolated_python(tmp_path)
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        f"os.execv({str(target)!r}, [{str(target)!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    snapshot = runtime_environment_snapshot("python", wrapper)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert target.absolute() in observed
    assert target.resolve(strict=True) in observed
    displaced = target.with_name("python.displaced")
    target.rename(displaced)
    target.symlink_to("/bin/false")
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_snapshot_binds_distribution_files_outside_site_packages(
    tmp_path: Path,
) -> None:
    python = _isolated_python(tmp_path)
    site_packages = next(
        path for path in (python.parents[1] / "lib").glob("python*/site-packages")
    )
    metadata = site_packages / "publication_runtime-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: publication-runtime\nVersion: 1.0\n",
        encoding="utf-8",
    )
    (metadata / "RECORD").write_text(
        "../../../bin/publication-runtime-tool,,\n"
        "publication_runtime-1.0.dist-info/METADATA,,\n"
        "publication_runtime-1.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    tool = python.parent / "publication-runtime-tool"
    tool.write_text("before\n", encoding="utf-8")

    snapshot = runtime_environment_snapshot("python", python)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert tool in observed
    tool.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_snapshot_binds_pyvenv_configuration(tmp_path: Path) -> None:
    python = _isolated_python(tmp_path)
    configuration = python.parents[1] / "pyvenv.cfg"

    snapshot = runtime_environment_snapshot("python", python)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert configuration in observed
    configuration.write_text(
        configuration.read_text(encoding="utf-8")
        + "include-system-site-packages = true\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_snapshot_binds_nonisolated_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _isolated_python(tmp_path)
    injected = tmp_path / "injected-pythonpath"
    injected.mkdir()
    module = injected / "publication_startup.py"
    module.write_text("VALUE = 'before'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(injected))

    snapshot = runtime_environment_snapshot("python", python)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert injected in observed
    assert module in observed
    module.write_text("VALUE = 'after'\n", encoding="utf-8")
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_snapshot_binds_missing_pythonpath_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _isolated_python(tmp_path)
    parent = tmp_path / "pythonpath-parent"
    parent.mkdir()
    future = parent / "future"
    monkeypatch.setenv("PYTHONPATH", str(future))

    snapshot = runtime_environment_snapshot("python", python)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert parent in observed
    future.mkdir()
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_snapshot_binds_absent_executable_pyvenv_configuration(
    tmp_path: Path,
) -> None:
    launcher_directory = tmp_path / "direct-launcher"
    launcher_directory.mkdir()
    launcher = launcher_directory / "python"
    launcher.symlink_to(sys.executable)
    configuration = launcher_directory / "pyvenv.cfg"

    snapshot = runtime_environment_snapshot("python", launcher)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert launcher_directory in observed
    configuration.write_text(
        "include-system-site-packages = false\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_snapshot_binds_intermediate_launcher_symlink(
    tmp_path: Path,
) -> None:
    target = _isolated_python(tmp_path)
    launcher = tmp_path / "python-launcher"
    intermediate = tmp_path / "python-intermediate"
    launcher.symlink_to(intermediate.name)
    intermediate.symlink_to(target)

    snapshot = runtime_environment_snapshot("python", launcher)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert intermediate in observed
    intermediate.unlink()
    intermediate.symlink_to("/bin/false")
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_snapshot_binds_env_shebang_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _isolated_python(tmp_path)
    first = tmp_path / "first-bootstrap"
    selected = tmp_path / "selected-bootstrap"
    first.mkdir()
    selected.mkdir()
    bootstrap_python = selected / "python3"
    bootstrap_python.symlink_to(sys.executable)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join((str(first), str(selected), os.environ.get("PATH", ""))),
    )
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        f"os.execv({str(target)!r}, [{str(target)!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)

    snapshot = runtime_environment_snapshot("python", wrapper)
    observed = {Path(path) for path, _identity in snapshot.path_identities}

    assert Path("/usr/bin/env") in observed
    assert bootstrap_python in observed
    assert first in observed
    (first / "python3").symlink_to("/bin/false")
    with pytest.raises(RuntimeEnvironmentError, match="runtime identity mismatch"):
        verify_runtime_environment_snapshot(snapshot)


def test_python_probe_binds_loose_importable_runtime_bytes(tmp_path: Path) -> None:
    python = _isolated_python(tmp_path)
    before = probe_python_environment(python)
    site_packages = next(
        path
        for path in (python.parents[1] / "lib").glob("python*/site-packages")
    )
    (site_packages / "publication_shadow.py").write_text(
        "VALUE = 'changed-runtime'\n", encoding="utf-8"
    )

    after = probe_python_environment(python)

    assert before != after
    assert before["runtime_roots"] != after["runtime_roots"]


def test_native_closure_follows_symlink_trees_and_binds_host_loader_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    target = tmp_path / "plugin-target"
    loader_root = tmp_path / "loader-root"
    root.mkdir()
    target.mkdir()
    loader_root.mkdir()
    plugin = target / "plugin"
    shutil.copy2("/bin/true", plugin)
    (root / "linked").symlink_to(target, target_is_directory=True)
    commands: list[tuple[str, ...]] = []
    real_run = subprocess.run

    def completed(command, **kwargs):
        commands.append(tuple(str(value) for value in command))
        return real_run(command, **kwargs)

    monkeypatch.setattr(runtime_module.subprocess, "run", completed)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(loader_root))

    roots, _linkage = runtime_module._with_native_dependency_roots(
        [{"role": "runtime", "kind": "directory", "path": str(root)}],
        Path("/bin/true"),
    )

    flattened_commands = {value for command in commands for value in command}
    assert str(plugin.resolve()) in flattened_commands
    root_paths = {entry["path"] for entry in roots}
    assert str(loader_root) in root_paths
    assert "/etc/ld.so.cache" in root_paths
    assert "/proc/modules" in root_paths


def test_native_closure_rejects_unresolved_required_linkage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    real_run = subprocess.run
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    def unresolved(command, **kwargs):
        if command[0] == "/usr/bin/ldd":
            return subprocess.CompletedProcess(
                command,
                0,
                b"libpublication-missing.so => not found\n",
                b"",
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr(runtime_module.subprocess, "run", unresolved)

    with pytest.raises(RuntimeEnvironmentError, match="unresolved native linkage"):
        runtime_module._with_native_dependency_roots(
            [{"role": "runtime", "kind": "directory", "path": str(root)}],
            Path("/bin/true"),
        )


def test_host_loader_roots_preserve_cache_alias_and_resolved_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "libpublication.so.1"
    shutil.copy2("/bin/true", target)
    alias = tmp_path / "libpublication.so"
    alias.symlink_to(target.name)

    def ldconfig(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            (
                "1 libs found in cache `/etc/ld.so.cache'\n"
                f"\tlibpublication.so (libc6,x86-64) => {alias}\n"
            ).encode(),
            b"",
        )

    monkeypatch.setattr(runtime_module.subprocess, "run", ldconfig)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    roots, _output, _names = runtime_module._host_loader_roots()
    paths = {path for _role, _kind, path in roots}

    assert alias in paths
    assert target in paths


def test_host_loader_roots_reject_symlinked_ld_library_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "loader-target"
    target.mkdir()
    alias = tmp_path / "loader-alias"
    alias.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(alias))

    with pytest.raises(RuntimeEnvironmentError, match="LD_LIBRARY_PATH.*symlink"):
        runtime_module._host_loader_roots()


@pytest.mark.parametrize("variable", ("LD_AUDIT", "LD_PRELOAD"))
def test_host_loader_roots_reject_injected_loader_objects(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    monkeypatch.setenv(variable, "/tmp/publication-injected.so")

    with pytest.raises(RuntimeEnvironmentError, match=variable):
        runtime_module._host_loader_roots()


@pytest.mark.parametrize("variable", ("LD_AUDIT", "LD_PRELOAD"))
def test_runtime_probe_rejects_loader_injection_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    called = False

    def forbidden_spawn(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("runtime probe spawned before loader validation")

    original = runtime_module._libc_environment_mapping().get(variable)
    monkeypatch.setattr(runtime_module.subprocess, "run", forbidden_spawn)
    try:
        os.putenv(variable, "/tmp/publication-injected.so")
        with pytest.raises(RuntimeEnvironmentError, match=variable):
            runtime_module._run_probe(["/bin/true"], "test")
    finally:
        if original is None:
            os.unsetenv(variable)
        else:
            os.putenv(variable, original)

    assert not called


def test_nvidia_smi_resolver_selects_exact_first_path_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        executable = directory / "nvidia-smi"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{first}{os.pathsep}{second}")

    assert runtime_module.nvidia_smi_executable() == first / "nvidia-smi"


def test_loader_preload_roots_bind_file_aliases_and_targets(tmp_path: Path) -> None:
    target = tmp_path / "libpreload.so.1"
    shutil.copy2("/bin/true", target)
    alias = tmp_path / "libpreload.so"
    alias.symlink_to(target.name)
    preload = tmp_path / "ld.so.preload"
    preload.write_text(f"# publication fixture\n{alias}\n", encoding="utf-8")

    roots = runtime_module._loader_preload_roots(preload)
    paths = {path for _role, _kind, path in roots}

    assert {preload, alias, target} <= paths


def test_optional_elf_binds_external_runpath_and_ldd_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    plugin = runtime_root / "plugin"
    shutil.copy2("/bin/true", plugin)
    external = tmp_path / "external"
    external.mkdir()
    dependency_target = external / "libpublication.so.1"
    shutil.copy2("/bin/true", dependency_target)
    dependency_alias = external / "libpublication.so"
    dependency_alias.symlink_to(dependency_target.name)
    transitive = tmp_path / "transitive"
    transitive.mkdir()
    cache = tmp_path / "ld.so.cache"
    cache.write_bytes(b"cache")

    monkeypatch.setattr(
        runtime_module,
        "_host_loader_roots",
        lambda: ([("dynamic-loader-cache", "file", cache)], b"cache", set()),
    )
    monkeypatch.setattr(runtime_module, "_gpu_driver_roots", lambda *_args: [])
    monkeypatch.setattr(runtime_module, "nvidia_smi_executable", lambda: None)

    def completed(command, **_kwargs):
        assert _kwargs["cwd"] == runtime_module.publication_runtime_working_directory()
        if command[0] == "/usr/bin/readelf":
            sections = []
            if str(plugin) in command:
                sections.extend(
                    (
                        f"File: {plugin}\n",
                        " 0x000000000000001d (RUNPATH)            "
                        "Library runpath: [$ORIGIN/../external]\n",
                    )
                )
            if str(dependency_target) in command:
                sections.extend(
                    (
                        f"File: {dependency_target}\n",
                        " 0x000000000000001d (RUNPATH)            "
                        "Library runpath: [$ORIGIN/../transitive]\n",
                    )
                )
            return subprocess.CompletedProcess(
                command,
                0,
                "".join(sections).encode(),
                b"",
            )
        if command[0] == "/usr/bin/ldd" and str(plugin) in command:
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    f"\tlibpublication.so => {dependency_alias} (0x1234)\n"
                    "\tliboptional-missing.so => not found\n"
                ).encode(),
                b"",
            )
        if command[0] == "/usr/bin/ldd":
            return subprocess.CompletedProcess(
                command,
                0,
                f"\tlibc.so.6 => {dependency_target} (0x1234)\n".encode(),
                b"",
            )
        raise AssertionError(command)

    monkeypatch.setattr(runtime_module.subprocess, "run", completed)

    roots, _linkage = runtime_module._with_native_dependency_roots(
        [{"role": "runtime", "kind": "directory", "path": str(runtime_root)}],
        Path("/bin/true"),
    )
    paths = {Path(root["path"]) for root in roots}

    assert external in paths
    assert transitive in paths
    assert dependency_alias in paths
    assert dependency_target in paths
    assert Path("/usr/bin/ldd") in paths
    assert Path("/bin/bash") in paths


def test_elf_runtime_search_roots_bind_fixed_cwd_for_empty_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "plugin.so"
    shutil.copy2("/bin/true", plugin)

    def completed(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            0,
            (
                f"File: {plugin}\n"
                " 0x000000000000001d (RUNPATH)            "
                "Library runpath: [:/usr/lib]\n"
            ).encode(),
            b"",
        )

    monkeypatch.setattr(runtime_module.subprocess, "run", completed)

    roots, _transcript = runtime_module._elf_runtime_search_roots((plugin,), ())

    assert runtime_module.publication_runtime_working_directory() in {
        path for _role, _kind, path in roots
    }


def test_runtime_change_monitor_fails_closed_on_queue_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = RuntimeChangeMonitor(())
    monitor._descriptor = 123
    payload = struct.pack("iIII", -1, 0x00004000, 0, 0)
    monkeypatch.setattr(runtime_module.os, "read", lambda _fd, _size: payload)
    try:
        with pytest.raises(RuntimeEnvironmentError, match="overflowed"):
            monitor.assert_unchanged()
    finally:
        monitor._descriptor = -1


def test_kernel_module_control_hash_ignores_reference_counts_and_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter(
        (
            b"nvidia_uvm 123 1 nvidia-uvm-tools,nvidia Live 0x0001\n",
            b"nvidia_uvm 123 99 nvidia-uvm-tools,nvidia Live 0x9999\n",
        )
    )
    monkeypatch.setattr(
        runtime_module, "_secure_regular_file_bytes", lambda _path: next(states)
    )

    first = runtime_module._control_file_sha256(Path("/proc/modules"))
    second = runtime_module._control_file_sha256(Path("/proc/modules"))

    assert first == second


def test_lock_validation_rejects_byte_identical_closure_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {"same": "bytes"}
    lock = RuntimeEnvironmentLock(
        path=tmp_path / "runtime-lock.json",
        file_sha256="a" * 64,
        entries=(
            RuntimeEnvironmentEntry(
                environment_id="benchmark",
                kind="python",
                inventory_json=json.dumps(inventory).encode(),
                inventory_sha256="b" * 64,
            ),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "_probe_runtime_environment_with_closure",
        lambda *_args, **_kwargs: (inventory, "c" * 64),
    )

    with pytest.raises(RuntimeEnvironmentError, match="closure paths mismatch"):
        validate_runtime_environment_lock(
            lock,
            {"benchmark": ("python", Path(sys.executable))},
            expected_closure_paths_sha256s={"benchmark": "d" * 64},
        )


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


def test_r_probe_rejects_script_launcher(tmp_path: Path) -> None:
    launcher = tmp_path / "Rscript"
    launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    launcher.chmod(0o755)

    with pytest.raises(RuntimeEnvironmentError, match="dynamic ELF"):
        probe_r_environment(launcher)


def test_directory_inventory_hashes_symlinked_directory_target_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    target = tmp_path / "target"
    root.mkdir()
    target.mkdir()
    payload = target / "payload.py"
    payload.write_text("VALUE = 1\n", encoding="utf-8")
    (root / "linked").symlink_to(target, target_is_directory=True)
    before, _count = _directory_content_sha256(root)
    payload.write_text("VALUE = 2\n", encoding="utf-8")

    after, _count = _directory_content_sha256(root)

    assert before != after


def test_runtime_file_hash_rejects_symlink_target_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"before")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"change")
    alias = tmp_path / "alias"
    alias.symlink_to(target.name)
    displaced = tmp_path / "target.displaced"
    real_read = runtime_module.os.read
    changed = False

    def mutating_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        payload = real_read(descriptor, size)
        if payload and not changed:
            changed = True
            target.rename(displaced)
            replacement.rename(target)
        return payload

    monkeypatch.setattr(runtime_module.os, "read", mutating_read)

    with pytest.raises(RuntimeEnvironmentError, match="runtime root file changed"):
        _runtime_file_content_sha256(alias)

    assert changed is True


@pytest.mark.parametrize(
    "function", (_directory_content_sha256, _runtime_root_identity_sha256)
)
def test_runtime_tree_rejects_mutation_of_already_visited_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    function,
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    visited = root / "a.txt"
    visited.write_text("before\n", encoding="utf-8")
    (nested / "z.txt").write_text("nested\n", encoding="utf-8")
    real_scandir = runtime_module.os.scandir
    changed = False

    def mutating_scandir(path):
        nonlocal changed
        if Path(path) == nested and not changed:
            changed = True
            visited.write_text("after\n", encoding="utf-8")
        return real_scandir(path)

    monkeypatch.setattr(runtime_module.os, "scandir", mutating_scandir)

    with pytest.raises(RuntimeEnvironmentError, match="after it was visited"):
        function(root)


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


def test_lock_loader_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    python = _isolated_python(tmp_path)
    lock = build_runtime_environment_lock({"benchmark": ("python", python)})
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    path = real_parent / "runtime-lock.json"
    _write_canonical(path, lock)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RuntimeEnvironmentError, match="parent.*symlink"):
        load_runtime_environment_lock(linked_parent / path.name)


def test_lock_loader_opens_leaf_nonblocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "runtime-lock.json"
    path.write_text("{}\n", encoding="utf-8")
    real_open = runtime_module.os.open
    leaf_flags: list[int] = []

    def recording_open(raw_path, flags, *args, **kwargs):
        if raw_path == path.name and kwargs.get("dir_fd") is not None:
            leaf_flags.append(flags)
        return real_open(raw_path, flags, *args, **kwargs)

    monkeypatch.setattr(runtime_module.os, "open", recording_open)

    with pytest.raises(RuntimeEnvironmentError):
        load_runtime_environment_lock(path)

    assert leaf_flags
    assert leaf_flags[0] & os.O_NONBLOCK


def test_lock_loader_rejects_path_replacement_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _isolated_python(tmp_path)
    lock = build_runtime_environment_lock({"benchmark": ("python", python)})
    path = tmp_path / "runtime-lock.json"
    replacement = tmp_path / "replacement.json"
    displaced = tmp_path / "displaced.json"
    _write_canonical(path, lock)
    shutil.copy2(path, replacement)
    real_read = runtime_module.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if chunk and not replaced:
            replaced = True
            path.rename(displaced)
            replacement.rename(path)
        return chunk

    monkeypatch.setattr(runtime_module.os, "read", replacing_read)

    with pytest.raises(RuntimeEnvironmentError, match="changed while reading"):
        load_runtime_environment_lock(path)


def test_lock_loader_rejects_in_place_mutation_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _isolated_python(tmp_path)
    lock = build_runtime_environment_lock({"benchmark": ("python", python)})
    path = tmp_path / "runtime-lock.json"
    _write_canonical(path, lock)
    real_read = runtime_module.os.read
    changed = False

    def mutating_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, size)
        if not chunk and not changed:
            changed = True
            metadata = path.stat()
            os.utime(
                path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1),
            )
        return chunk

    monkeypatch.setattr(runtime_module.os, "read", mutating_read)

    with pytest.raises(RuntimeEnvironmentError, match="changed while reading"):
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
