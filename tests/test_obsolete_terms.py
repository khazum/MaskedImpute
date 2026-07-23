from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

# Historical sources remain byte-for-byte available for provenance, while
# third-party trees are outside the terminology contract for MaskImpute.
EXCLUDED_PREFIXES = (
    "historical/v26_neurips/",
    "DenseLayerPack/",
)
EXCLUDED_EXACT_PATHS = {
    # These binding policy documents must name the terminology they prohibit.
    "docs/superpowers/plans/2026-07-12-method-competition.md",
    "docs/superpowers/specs/2026-07-12-genome-biology-study-design.md",
    # Raw Git inventories must retain exact archived path names as review evidence.
    "docs/superpowers/reviews/2026-07-23-publication-integration-baseline-name-status.txt",
    "docs/superpowers/reviews/2026-07-23-publication-integration-baseline-numstat.txt",
}

OBSOLETE_PATTERNS = {
    "retired concatenated zero label": re.compile("bio" + "zero", re.IGNORECASE),
    "retired probability name": re.compile(
        r"\bp_" + "bio" + r"(?:\b|_)",
        re.IGNORECASE,
    ),
    "retired ontology phrase A": re.compile(
        r"\bstructural(?:[- ]+)zero(?:s|es)?\b",
        re.IGNORECASE,
    ),
    "retired ontology phrase B": re.compile(
        r"\bstructural(?:[- ]+)non[- ]?expression\b",
        re.IGNORECASE,
    ),
}


def _tracked_paths() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(item.decode("utf-8") for item in result.stdout.split(b"\0") if item)


def _is_excluded(relative_path: str) -> bool:
    return relative_path in EXCLUDED_EXACT_PATHS or relative_path.startswith(
        EXCLUDED_PREFIXES
    )


def _utf8_text(relative_path: str) -> str | None:
    path = ROOT / relative_path
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def test_active_tracked_tree_uses_canonical_zero_terminology() -> None:
    violations: list[str] = []
    for relative_path in _tracked_paths():
        if _is_excluded(relative_path):
            continue
        searchable = relative_path
        text = _utf8_text(relative_path)
        if text is not None:
            searchable += "\n" + text
        for description, pattern in OBSOLETE_PATTERNS.items():
            if pattern.search(searchable):
                violations.append(f"{relative_path}: {description}")

    assert not violations, (
        "obsolete terminology outside explicit archives:\n" + "\n".join(violations)
    )


def test_original_v26_entrypoint_is_migration_guidance_only() -> None:
    result = subprocess.run(
        [sys.executable, "masked_imputation26.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    guidance = " ".join(result.stdout.lower().split())
    assert "canonical package api" in guidance
    assert "historical/v26_neurips" in guidance
    assert "not a publication runner" in guidance
