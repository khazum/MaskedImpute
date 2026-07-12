from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import warnings

import pytest

from maskimpute_benchmark.sources import (
    SourceLedgerError,
    _assert_tracked_bytes,
    fetch_sources,
    load_source_ledger,
)


EXPECTED_GIT_PINS = {
    "symsim": (
        "76a674b407ce44bf2690a9161cf28b905598d0a5",
        "12d9c7e9e8c22bb0bae917aec7860627dcb8489b",
    ),
    "sergio": (
        "a6190b74425112834c8fa9b4b6157d9cb3d1ab88",
        "15558fe60f62683c6fa46bcde01d9f3d3382e34a",
    ),
    "sparsim": (
        "4e7712fb236a92ce7c173da169c8a29cc2a9f0ef",
        "5d66b28cc6afd8d68364f4205cc983c7f681e2fe",
    ),
}

# Keep the required integration-test marker local to this task's exact file set;
# the project-wide marker registry is amended with the later environment task.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    network = pytest.mark.network


def _run_git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _make_upstream(tmp_path: Path) -> tuple[Path, str, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _run_git("init", "-q", cwd=upstream)
    _run_git("config", "user.name", "Source Test", cwd=upstream)
    _run_git("config", "user.email", "source-test@example.invalid", cwd=upstream)
    (upstream / "source.txt").write_text("pinned bytes\n", encoding="utf-8")
    _run_git("add", "source.txt", cwd=upstream)
    _run_git("commit", "-qm", "pinned", cwd=upstream)
    commit = _run_git("rev-parse", "HEAD", cwd=upstream)
    tree = _run_git("rev-parse", "HEAD^{tree}", cwd=upstream)
    return upstream, commit, tree


def _git_ledger_payload(upstream: Path, commit: str, tree: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sources": [
            {
                "id": "local-source",
                "role": "mechanism",
                "mechanism": "symsim",
                "source_type": "git",
                "url": upstream.resolve().as_uri(),
                "revision": commit,
                "license": "MIT",
                "license_url": "https://example.invalid/license",
                "citation_doi": "10.1234/example.source",
                "expected_checksum": {
                    "algorithm": "git-tree-sha1",
                    "value": tree,
                },
                "eligibility": "eligible",
                "endpoints": ["simulation"],
            }
        ],
    }


def _write_payload(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load_local(tmp_path: Path, payload: object):
    return load_source_ledger(_write_payload(tmp_path, payload), allow_local_urls=True)


def test_publication_ledger_covers_pinned_panel_and_orthogonal_data() -> None:
    ledger = load_source_ledger(Path("study/sources.json"))
    by_id = {source.id: source for source in ledger.sources}

    assert set(EXPECTED_GIT_PINS) <= set(by_id)
    for source_id, (revision, tree) in EXPECTED_GIT_PINS.items():
        source = by_id[source_id]
        assert source.source_type == "git"
        assert source.revision == revision
        assert source.expected_checksum.algorithm == "git-tree-sha1"
        assert source.expected_checksum.value == tree

    semisynthetic = [
        source for source in ledger.sources if source.role == "semisynthetic_source"
    ]
    orthogonal = [
        source for source in ledger.sources if source.role == "orthogonal_validation"
    ]
    assert [(source.id, source.revision) for source in semisynthetic] == [
        ("baron-pancreas-umi", "GSE84133:2019-05-15")
    ]
    assert {source.id for source in orthogonal} == {
        "cite-seq-cbmc-rna-protein",
        "tung-ipsc-ercc-bulk-replicates",
    }
    tung = by_id["tung-ipsc-ercc-bulk-replicates"]
    assert {artifact.name for artifact in tung.artifacts} == {
        "GSE77288_molecules-raw-single-per-sample.txt.gz",
        "GSE77288_reads-raw-bulk-per-sample.txt.gz",
        "GSE77288_molecules-raw-single-per-lane.txt.gz",
        "GSE77288_reads-raw-bulk-per-lane.txt.gz",
    }
    assert {
        endpoint for source in orthogonal for endpoint in source.endpoints
    } >= {
        "rna_protein_concordance",
        "ercc_recovery",
        "technical_replicate_concordance",
        "bulk_pseudobulk_concordance",
    }
    assert all(source.eligibility == "eligible" for source in ledger.sources)
    assert all(source.license and source.citation_doi for source in ledger.sources)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("url", "http://example.org/source.git", "HTTPS"),
        ("url", "https://token@example.org/source.git", "credentials"),
        ("revision", "abc123", "40-character"),
        ("revision", "A" * 40, "lowercase"),
        ("license", "TBD", "SPDX-like"),
        ("citation_doi", "https://doi.org/not-a-doi", "DOI"),
    ],
)
def test_ledger_rejects_invalid_required_git_metadata(
    tmp_path: Path, field: str, invalid: str, message: str
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    payload = _git_ledger_payload(upstream, commit, tree)
    payload["sources"][0][field] = invalid  # type: ignore[index]

    with pytest.raises(SourceLedgerError, match=message):
        _load_local(tmp_path, payload)


@pytest.mark.parametrize(
    "algorithm,value",
    [
        ("sha256", "0" * 64),
        ("git-tree-sha1", "short"),
        ("git-tree-sha1", "A" * 40),
    ],
)
def test_git_source_rejects_non_tree_or_malformed_checksum(
    tmp_path: Path, algorithm: str, value: str
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    payload = _git_ledger_payload(upstream, commit, tree)
    payload["sources"][0]["expected_checksum"] = {  # type: ignore[index]
        "algorithm": algorithm,
        "value": value,
    }

    with pytest.raises(SourceLedgerError, match="git-tree-sha1|checksum.*lowercase"):
        _load_local(tmp_path, payload)


def test_data_source_requires_versioned_accession_and_artifact_checksums(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "sources": [
            {
                "id": "data",
                "role": "orthogonal_validation",
                "mechanism": None,
                "source_type": "data",
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1",
                "revision": "latest",
                "license": "LicenseRef-NCBI-GEO-NoRestrictions",
                "license_url": "https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html",
                "citation_doi": "10.1234/example.data",
                "expected_checksum": None,
                "eligibility": "eligible",
                "endpoints": ["orthogonal"],
                "artifacts": [],
            }
        ],
    }

    with pytest.raises(SourceLedgerError, match="accession:version"):
        _load_local(tmp_path, payload)

    payload["sources"][0]["revision"] = "GSE1:2024-01-31"  # type: ignore[index]
    with pytest.raises(SourceLedgerError, match="artifact"):
        _load_local(tmp_path, payload)


def test_ledger_rejects_duplicate_json_keys_and_source_ids(tmp_path: Path) -> None:
    duplicate_key = tmp_path / "duplicate-key.json"
    duplicate_key.write_text(
        '{"schema_version":1,"schema_version":1,"sources":[]}',
        encoding="utf-8",
    )
    with pytest.raises(SourceLedgerError, match="duplicate JSON key"):
        load_source_ledger(duplicate_key)

    upstream, commit, tree = _make_upstream(tmp_path)
    payload = _git_ledger_payload(upstream, commit, tree)
    payload["sources"].append(dict(payload["sources"][0]))  # type: ignore[union-attr,index]
    with pytest.raises(SourceLedgerError, match="duplicate source id"):
        _load_local(tmp_path, payload)


def test_fetch_git_source_detaches_exact_commit_and_writes_canonical_receipt(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"

    receipts = fetch_sources(ledger, fetch_root, allow_local_urls=True)

    checkout = fetch_root / "checkouts" / "local-source"
    assert _run_git("rev-parse", "HEAD", cwd=checkout) == commit
    assert _run_git("rev-parse", "HEAD^{tree}", cwd=checkout) == tree
    assert subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=checkout
    ).returncode == 1
    assert _run_git("status", "--porcelain=v1", cwd=checkout) == ""
    config = (checkout / ".git" / "config").read_text(encoding="utf-8")
    assert "credential" not in config.casefold()

    receipt_path = fetch_root / "receipts" / "local-source.json"
    receipt_bytes = receipt_path.read_bytes()
    assert receipt_bytes.endswith(b"\n")
    receipt = json.loads(receipt_bytes)
    assert receipts == (receipt,)
    assert receipt["resolved_revision"] == commit
    assert receipt["verified_checksum"] == {
        "algorithm": "git-tree-sha1",
        "value": tree,
    }
    assert receipt_bytes == (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def test_fetch_rejects_dirty_or_wrong_existing_checkout(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))

    dirty_root = tmp_path / "dirty-root"
    fetch_sources(ledger, dirty_root, allow_local_urls=True)
    (dirty_root / "checkouts" / "local-source" / "untracked.txt").write_text("dirty")
    with pytest.raises(SourceLedgerError, match="local changes"):
        fetch_sources(ledger, dirty_root, allow_local_urls=True)

    wrong_root = tmp_path / "wrong-root"
    wrong_checkout = wrong_root / "checkouts" / "local-source"
    wrong_checkout.parent.mkdir(parents=True)
    _run_git("clone", "-q", upstream.as_posix(), wrong_checkout.as_posix(), cwd=tmp_path)
    (upstream / "source.txt").write_text("new upstream bytes\n", encoding="utf-8")
    _run_git("commit", "-qam", "later", cwd=upstream)
    _run_git("pull", "-q", cwd=wrong_checkout)
    with pytest.raises(SourceLedgerError, match="wrong commit"):
        fetch_sources(ledger, wrong_root, allow_local_urls=True)


def test_fetch_rejects_ignored_untracked_changes_in_existing_checkout(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    (checkout / ".git" / "info" / "exclude").write_text("hidden.tmp\n")
    (checkout / "hidden.tmp").write_text("must not be ignored\n")
    assert _run_git("status", "--porcelain=v1", cwd=checkout) == ""

    with pytest.raises(SourceLedgerError, match="local changes"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_hashes_tracked_bytes_instead_of_trusting_git_stat_cache(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    tracked = checkout / "source.txt"
    original_stat = tracked.stat()
    _run_git("config", "core.trustctime", "false", cwd=checkout)
    _run_git("config", "core.checkStat", "minimal", cwd=checkout)
    old_time = original_stat.st_mtime_ns - 10_000_000_000
    os.utime(tracked, ns=(old_time, old_time))
    _run_git("update-index", "--refresh", cwd=checkout)
    original_stat = tracked.stat()
    tracked.write_text("edited bytes\n", encoding="utf-8")
    os.utime(
        tracked,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    assert _run_git("status", "--porcelain=v1", cwd=checkout) == ""

    with pytest.raises(SourceLedgerError, match="tracked bytes"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_compares_raw_bytes_without_git_attribute_filters(
    tmp_path: Path,
) -> None:
    upstream, _commit, _tree = _make_upstream(tmp_path)
    (upstream / ".gitattributes").write_text(
        "source.txt text eol=lf\n", encoding="utf-8"
    )
    _run_git("add", ".gitattributes", cwd=upstream)
    _run_git("commit", "--amend", "-qm", "pinned with attributes", cwd=upstream)
    commit = _run_git("rev-parse", "HEAD", cwd=upstream)
    tree = _run_git("rev-parse", "HEAD^{tree}", cwd=upstream)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    (checkout / "source.txt").write_bytes(b"pinned bytes\r\n")

    with pytest.raises(SourceLedgerError, match="tracked bytes"):
        _assert_tracked_bytes(checkout, tree, "local-source")


@pytest.mark.parametrize("target", ["../../outside-payload", "/tmp/outside-payload"])
def test_fetch_rejects_tracked_symlink_that_escapes_checkout(
    tmp_path: Path, target: str
) -> None:
    upstream, _commit, _tree = _make_upstream(tmp_path)
    (upstream / "escape-link").symlink_to(target)
    _run_git("add", "escape-link", cwd=upstream)
    _run_git("commit", "-qm", "add escaping symlink", cwd=upstream)
    commit = _run_git("rev-parse", "HEAD", cwd=upstream)
    tree = _run_git("rev-parse", "HEAD^{tree}", cwd=upstream)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))

    with pytest.raises(SourceLedgerError, match="symlink.*escape"):
        fetch_sources(ledger, tmp_path / "external", allow_local_urls=True)


def test_fetch_never_updates_a_correct_pin_when_upstream_advances(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)

    (upstream / "source.txt").write_text("later bytes\n", encoding="utf-8")
    _run_git("commit", "-qam", "later", cwd=upstream)
    receipts = fetch_sources(ledger, fetch_root, allow_local_urls=True)

    checkout = fetch_root / "checkouts" / "local-source"
    assert _run_git("rev-parse", "HEAD", cwd=checkout) == commit
    assert receipts[0]["resolved_revision"] == commit


def test_fetch_rejects_credential_configuration_in_existing_checkout(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    _run_git("config", "credential.helper", "store", cwd=checkout)

    with pytest.raises(SourceLedgerError, match="credential"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("http.extraHeader", "Authorization: Bearer secret"),
        ("remote.origin.pushurl", "https://token@example.invalid/repository.git"),
        ("url.https://token@example.invalid/.insteadOf", "https://example.invalid/"),
    ],
)
def test_fetch_rejects_other_persisted_transport_credentials(
    tmp_path: Path, key: str, value: str
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    _run_git("config", key, value, cwd=checkout)

    with pytest.raises(SourceLedgerError, match="credential|transport"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_rejects_credential_bearing_extra_remote(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    _run_git(
        "remote",
        "add",
        "credential-leak",
        "https://user:secret@example.invalid/repository.git",
        cwd=checkout,
    )

    with pytest.raises(SourceLedgerError, match="remote|credential"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_rejects_reserved_directory_symlink_escape(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    outside = tmp_path / "outside"
    fetch_root.mkdir()
    outside.mkdir()
    (fetch_root / "checkouts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SourceLedgerError, match="symlink"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)
    assert not (outside / "local-source").exists()


def test_fetch_rejects_checkout_symlink_escape(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    outside = tmp_path / "moved-checkout"
    checkout.rename(outside)
    checkout.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SourceLedgerError, match="symlink"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_rejects_gitfile_checkout_indirection(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    external_gitdir = tmp_path / "external-gitdir"
    (checkout / ".git").rename(external_gitdir)
    (checkout / ".git").write_text(
        f"gitdir: {external_gitdir.as_posix()}\n", encoding="utf-8"
    )

    with pytest.raises(SourceLedgerError, match="Git directory"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_rejects_symlink_inside_git_administrative_state(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    objects = checkout / ".git" / "objects"
    outside_objects = tmp_path / "outside-objects"
    objects.rename(outside_objects)
    objects.symlink_to(outside_objects, target_is_directory=True)

    with pytest.raises(SourceLedgerError, match="Git administrative.*symlink"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_rejects_git_object_alternate(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    (checkout / ".git" / "objects" / "info" / "alternates").write_text(
        f"{(upstream / '.git' / 'objects').resolve().as_posix()}\n",
        encoding="utf-8",
    )

    with pytest.raises(SourceLedgerError, match="object alternate"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_rejects_case_colliding_untracked_file_hidden_by_git_config(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    _run_git("config", "core.ignoreCase", "true", cwd=checkout)
    (checkout / "Source.txt").write_text("case-collision\n", encoding="utf-8")
    assert _run_git("status", "--porcelain=v1", cwd=checkout) == ""

    with pytest.raises(SourceLedgerError, match="local changes"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


@pytest.mark.parametrize(
    "names",
    [
        ("Case.txt", "case.txt"),
        ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.txt", "cafe\N{COMBINING ACUTE ACCENT}.txt"),
    ],
)
def test_fetch_rejects_nonportable_tracked_path_collisions(
    tmp_path: Path, names: tuple[str, str]
) -> None:
    upstream, _commit, _tree = _make_upstream(tmp_path)
    for index, name in enumerate(names):
        (upstream / name).write_text(f"payload {index}\n", encoding="utf-8")
    _run_git("add", "--", *names, cwd=upstream)
    _run_git("commit", "-qm", "add colliding paths", cwd=upstream)
    commit = _run_git("rev-parse", "HEAD", cwd=upstream)
    tree = _run_git("rev-parse", "HEAD^{tree}", cwd=upstream)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))

    with pytest.raises(SourceLedgerError, match="collision|portable"):
        fetch_sources(ledger, tmp_path / "external", allow_local_urls=True)


@pytest.mark.parametrize("metadata", ["replacement", "graft"])
def test_fetch_rejects_local_object_replacement_metadata(
    tmp_path: Path, metadata: str
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    fetch_root = tmp_path / "external"
    fetch_sources(ledger, fetch_root, allow_local_urls=True)
    checkout = fetch_root / "checkouts" / "local-source"
    if metadata == "replacement":
        replace_ref = checkout / ".git" / "refs" / "replace" / commit
        replace_ref.parent.mkdir(parents=True)
        replace_ref.write_text(f"{commit}\n", encoding="ascii")
    else:
        (checkout / ".git" / "info" / "grafts").write_text(
            f"{commit}\n", encoding="ascii"
        )

    with pytest.raises(SourceLedgerError, match="replacement|graft"):
        fetch_sources(ledger, fetch_root, allow_local_urls=True)


def test_fetch_rejects_nonignored_destination_inside_git_worktree(tmp_path: Path) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    project = tmp_path / "project"
    project.mkdir()
    _run_git("init", "-q", cwd=project)
    root = project / "external"

    with pytest.raises(SourceLedgerError, match="ignored"):
        fetch_sources(ledger, root, allow_local_urls=True)

    (project / ".gitignore").write_text("external/\n", encoding="utf-8")
    receipts = fetch_sources(ledger, root, allow_local_urls=True)
    assert receipts[0]["source_id"] == "local-source"


def test_fetch_rejects_destination_inside_git_administrative_state(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    project = tmp_path / "project"
    project.mkdir()
    _run_git("init", "-q", cwd=project)
    administrative_root = project / ".git" / "maskimpute-fetch"

    with pytest.raises(SourceLedgerError, match="administrative|Git directory"):
        fetch_sources(ledger, administrative_root, allow_local_urls=True)

    assert not administrative_root.exists()


def test_fetch_rejects_ignore_rule_that_matches_only_internal_probe(
    tmp_path: Path,
) -> None:
    upstream, commit, tree = _make_upstream(tmp_path)
    ledger = _load_local(tmp_path, _git_ledger_payload(upstream, commit, tree))
    project = tmp_path / "project"
    project.mkdir()
    _run_git("init", "-q", cwd=project)
    (project / ".gitignore").write_text(
        "external/.maskimpute-fetch-probe\n", encoding="utf-8"
    )

    with pytest.raises(SourceLedgerError, match="ignored"):
        fetch_sources(ledger, project / "external", allow_local_urls=True)


def test_fetch_revalidates_public_ledger_objects_before_path_construction(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(b"immutable public data\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "sources": [
            {
                "id": "local-data",
                "role": "orthogonal_validation",
                "mechanism": None,
                "source_type": "data",
                "url": "https://example.invalid/GSE1",
                "revision": "GSE1:2024-01-31",
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "citation_doi": "10.1234/example.data",
                "expected_checksum": None,
                "eligibility": "eligible",
                "endpoints": ["orthogonal"],
                "artifacts": [
                    {
                        "name": "source.bin",
                        "url": artifact.resolve().as_uri(),
                        "expected_checksum": {
                            "algorithm": "sha256",
                            "value": digest,
                        },
                    }
                ],
            }
        ],
    }
    ledger = _load_local(tmp_path, payload)
    malicious_source = replace(ledger.sources[0], id="../../escaped")
    malicious_ledger = replace(ledger, sources=(malicious_source,))
    root = tmp_path / "download-root"

    with pytest.raises(SourceLedgerError, match="source|identifier|ledger|path"):
        fetch_sources(malicious_ledger, root, allow_local_urls=True)

    assert not (tmp_path / "escaped" / "source.bin").exists()


def test_ledger_rejects_nonportable_data_artifact_name_collisions(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(b"immutable public data\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    base_artifact = {
        "url": artifact.resolve().as_uri(),
        "expected_checksum": {"algorithm": "sha256", "value": digest},
    }
    payload = {
        "schema_version": 1,
        "sources": [
            {
                "id": "local-data",
                "role": "orthogonal_validation",
                "mechanism": None,
                "source_type": "data",
                "url": "https://example.invalid/GSE1",
                "revision": "GSE1:2024-01-31",
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "citation_doi": "10.1234/example.data",
                "expected_checksum": None,
                "eligibility": "eligible",
                "endpoints": ["orthogonal"],
                "artifacts": [
                    {"name": "Data.bin", **base_artifact},
                    {"name": "data.bin", **base_artifact},
                ],
            }
        ],
    }

    with pytest.raises(SourceLedgerError, match="collision|unique"):
        _load_local(tmp_path, payload)


def test_data_fetch_verifies_bytes_and_rejects_corrupt_existing_file(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "source.bin"
    artifact.write_bytes(b"immutable public data\n")
    sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "sources": [
            {
                "id": "local-data",
                "role": "orthogonal_validation",
                "mechanism": None,
                "source_type": "data",
                "url": "https://example.invalid/GSE1",
                "revision": "GSE1:2024-01-31",
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "citation_doi": "10.1234/example.data",
                "expected_checksum": None,
                "eligibility": "eligible",
                "endpoints": ["orthogonal"],
                "artifacts": [
                    {
                        "name": "source.bin",
                        "url": artifact.resolve().as_uri(),
                        "expected_checksum": {
                            "algorithm": "sha256",
                            "value": sha256,
                        },
                    }
                ],
            }
        ],
    }
    ledger = _load_local(tmp_path, payload)
    root = tmp_path / "download-root"
    receipt = fetch_sources(ledger, root, allow_local_urls=True)[0]
    fetched = root / "data" / "local-data" / "source.bin"
    assert fetched.read_bytes() == artifact.read_bytes()
    assert receipt["artifacts"] == [
        {"name": "source.bin", "sha256": sha256, "size_bytes": 22}
    ]

    fetched.write_bytes(b"corrupt")
    with pytest.raises(SourceLedgerError, match="checksum"):
        fetch_sources(ledger, root, allow_local_urls=True)

    escape_root = tmp_path / "data-escape-root"
    outside = tmp_path / "data-outside"
    (escape_root / "data").mkdir(parents=True)
    outside.mkdir()
    (escape_root / "data" / "local-data").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(SourceLedgerError, match="symlink"):
        fetch_sources(ledger, escape_root, allow_local_urls=True)
    assert not (outside / "source.bin").exists()


@network
@pytest.mark.skipif(
    os.environ.get("MASKIMPUTE_RUN_NETWORK_TESTS") != "1",
    reason="set MASKIMPUTE_RUN_NETWORK_TESTS=1 to fetch a pinned upstream source",
)
def test_network_fetches_exact_symsim_pin(tmp_path: Path) -> None:
    ledger = load_source_ledger(Path("study/sources.json"))
    receipt = fetch_sources(ledger, tmp_path / "external", source_ids=["symsim"])[0]
    assert receipt["resolved_revision"] == EXPECTED_GIT_PINS["symsim"][0]
