from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _tracked_index_entries() -> tuple[tuple[str, str], ...]:
    result = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    entries: list[tuple[str, str]] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, relative = raw.decode("utf-8").split("\t", maxsplit=1)
        mode = metadata.split(" ", maxsplit=1)[0]
        entries.append((mode, relative))
    return tuple(entries)


def test_publication_tree_has_no_gitlinks_or_tracked_virtual_environment() -> None:
    entries = _tracked_index_entries()
    gitlinks = sorted(relative for mode, relative in entries if mode == "160000")
    virtual_environment = sorted(
        relative
        for _, relative in entries
        if relative == ".venv_scvi" or relative.startswith(".venv_scvi/")
    )

    assert not gitlinks, "publication tree retains gitlinks: " + ", ".join(gitlinks)
    assert not virtual_environment, (
        "publication tree retains a machine-specific virtual environment"
    )


def test_machine_specific_scvi_environment_is_ignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", ".venv_scvi/pyvenv.cfg"],
        cwd=ROOT,
        check=False,
    )

    assert result.returncode == 0
