from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import shutil
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import maskimpute_benchmark.methods as benchmark_methods
from maskimpute_benchmark.methods import load_method_registry, prepare_method_input
from maskimpute_benchmark.methods.observed import AdapterUnavailableError
from maskimpute_benchmark.methods.sctsi import (
    SCTSIConfig,
    SCTSIMatchedBulkReference,
    SCTSIUnavailableError,
    finalize_sctsi_output,
    prepare_sctsi_matched_bulk_reference,
    run_sctsi,
    validate_sctsi_matched_bulk_reference,
)


METHODS_PATH = Path("study/methods.json")
SOURCE_ROOT = Path("artifacts/method-sources")
SOURCE_SHA = "d" * 64
BULK_SHA = "e" * 64


def _method_input(*, cells: int = 10, genes: int = 7, zero_library: bool = False):
    counts = np.empty((cells, genes), dtype=np.int64)
    for cell in range(cells):
        for gene in range(genes):
            value = 1 + ((cell * 3 + gene * 5) % 11)
            counts[cell, gene] = 0 if (cell + gene * 2) % 7 == 0 else value
    if zero_library:
        counts[0, :] = 0
    view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(cells)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(genes)]),
    )
    view.uns["source_dataset_sha256"] = SOURCE_SHA
    view.uns["allowed_covariates"] = {"obs": [], "var": []}
    view.uns["normalization"] = {
        "input": "counts",
        "target_sum": None,
        "log_base": None,
        "size_factor": None,
    }
    return prepare_method_input(view)


def _bulk_reference(method_input, *, samples: int = 3):
    matrix = np.empty((method_input.shape[1], samples), dtype=np.int64)
    for gene in range(method_input.shape[1]):
        for sample in range(samples):
            matrix[gene, sample] = 2 + ((gene * 7 + sample * 3) % 19)
    return prepare_sctsi_matched_bulk_reference(
        reference_id="matched-bulk-sctsi",
        source_sha256=BULK_SHA,
        matrix=matrix,
        var_ids=method_input.var_ids,
        sample_ids=tuple(f"bulk-{index}" for index in range(samples)),
        expression_scale="raw_counts",
    )


def _registry():
    return load_method_registry(METHODS_PATH)


def _cached_source() -> Path:
    source = SOURCE_ROOT / "sctsi"
    if not source.is_dir():
        pytest.skip("ignored pinned-source cache is absent: sctsi")
    return source


def _fake_rscript(tmp_path: Path) -> Path:
    launcher = tmp_path / "fake-rscript"
    launcher.write_text(
        f"""#!{sys.executable}
import sys
import os
print("fake-sctsi-stdout")
print("R_LIBS=" + os.environ.get("R_LIBS", "<unset>"))
print("R_LIBS_SITE=" + os.environ.get("R_LIBS_SITE", "<unset>"))
print("fake-sctsi-stderr", file=sys.stderr)
raise SystemExit(7)
""",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    return launcher


def test_sctsi_config_matches_pinned_defaults_and_has_no_seed_parameter() -> None:
    assert SCTSIConfig() == SCTSIConfig(
        threshold=0.0,
        cell_neighbors=25,
        gene_neighbors=25,
    )
    assert "seed" not in inspect.signature(run_sctsi).parameters


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SCTSIConfig(threshold=-1), "threshold"),
        (lambda: SCTSIConfig(cell_neighbors=0), "cell_neighbors"),
        (lambda: SCTSIConfig(gene_neighbors=0), "gene_neighbors"),
    ],
)
def test_sctsi_config_rejects_invalid_values(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_sctsi_bulk_reference_is_immutable_scale_bound_and_gene_aligned() -> None:
    method_input = _method_input(cells=8, genes=6)
    reference = _bulk_reference(method_input)

    assert isinstance(reference, SCTSIMatchedBulkReference)
    assert reference.reference_id == "matched-bulk-sctsi"
    assert reference.source_sha256 == BULK_SHA
    assert reference.expression_scale == "raw_counts"
    assert reference.var_ids == method_input.var_ids
    assert reference.sample_ids == ("bulk-0", "bulk-1", "bulk-2")
    assert reference.shape == (6, 3)
    assert reference.matrix.flags.writeable is False
    validated = validate_sctsi_matched_bulk_reference(method_input, reference)
    assert validated.flags.writeable is False

    bad_ids = ("wrong", *reference.var_ids[1:])
    with pytest.raises(ValueError, match="gene IDs"):
        validate_sctsi_matched_bulk_reference(
            method_input, replace(reference, var_ids=bad_ids)
        )
    with pytest.raises(ValueError, match="hash"):
        validate_sctsi_matched_bulk_reference(
            method_input, replace(reference, matrix_sha256="f" * 64)
        )


def test_sctsi_bulk_reference_requires_exact_raw_integer_scale() -> None:
    method_input = _method_input(cells=8, genes=6)
    matrix = np.ones((6, 2), dtype=np.float64)
    matrix[0, 0] = 1.5

    with pytest.raises(ValueError, match="integer raw counts"):
        prepare_sctsi_matched_bulk_reference(
            reference_id="bulk",
            source_sha256=BULK_SHA,
            matrix=matrix,
            var_ids=method_input.var_ids,
            sample_ids=("a", "b"),
            expression_scale="raw_counts",
        )
    with pytest.raises(ValueError, match="expression_scale"):
        prepare_sctsi_matched_bulk_reference(
            reference_id="bulk",
            source_sha256=BULK_SHA,
            matrix=np.ones((6, 2), dtype=np.int64),
            var_ids=method_input.var_ids,
            sample_ids=("a", "b"),
            expression_scale="cpm",
        )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (
            np.full((6, 2), 2**63 + 1, dtype=np.uint64),
            "exactly representable as float64",
        ),
        (np.ones((6, 2), dtype=np.longdouble), "up to float64"),
        (np.ones((6, 2), dtype=np.bool_), "boolean"),
    ],
)
def test_sctsi_bulk_reference_rejects_lossy_or_semantic_dtypes(
    matrix: np.ndarray,
    message: str,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    with pytest.raises(ValueError, match=message):
        prepare_sctsi_matched_bulk_reference(
            reference_id="bulk",
            source_sha256=BULK_SHA,
            matrix=matrix,
            var_ids=method_input.var_ids,
            sample_ids=("a", "b"),
            expression_scale="raw_counts",
        )


def test_sctsi_native_cpm_is_inverted_with_observed_libraries_for_evaluator() -> None:
    method_input = _method_input(cells=8, genes=6)
    libraries = np.sum(method_input.counts, axis=1, dtype=np.float64)
    native = (
        np.asarray(method_input.counts, dtype=np.float64)
        / libraries[:, None]
        * 1_000_000.0
    )
    native[method_input.counts == 0] = 125.0
    snapshot = finalize_sctsi_output(_registry().by_id("sctsi"), method_input, native)

    assert snapshot.method_id == "sctsi"
    assert snapshot.output_scale == "external_reference_adjusted"
    assert snapshot.obs_ids == method_input.obs_ids
    assert snapshot.var_ids == method_input.var_ids
    converted = benchmark_methods.external_reference_output_to_evaluator_counts(
        method_input, snapshot
    )
    common = benchmark_methods.external_reference_output_to_evaluator_log2_cp10k(
        method_input, snapshot
    )
    expected_counts = native * libraries[:, None] / 1_000_000.0
    np.testing.assert_allclose(converted, expected_counts, rtol=1e-12, atol=1e-12)
    expected = np.log2(
        1.0 + expected_counts / expected_counts.sum(axis=1, keepdims=True) * 10_000.0
    )
    np.testing.assert_allclose(common, expected, rtol=1e-12, atol=1e-12)
    assert set(benchmark_methods.EXTERNAL_REFERENCE_EVALUATOR_COUNT_CONVERTERS) == {
        "d3impute",
        "sctsi",
    }


def test_sctsi_rejects_zero_library_bulk_sample_before_environment_execution(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    matrix = np.ones((6, 2), dtype=np.int64)
    matrix[:, 1] = 0
    reference = prepare_sctsi_matched_bulk_reference(
        reference_id="matched-bulk-sctsi",
        source_sha256=BULK_SHA,
        matrix=matrix,
        var_ids=method_input.var_ids,
        sample_ids=("bulk-ok", "bulk-zero"),
        expression_scale="raw_counts",
    )

    with pytest.raises(SCTSIUnavailableError) as captured:
        run_sctsi(
            _registry().by_id("sctsi"),
            method_input,
            bulk_reference=reference,
            source_dir=_cached_source(),
            rscript=tmp_path / "missing-rscript",
            r_library=tmp_path / "missing-library",
            config=SCTSIConfig(cell_neighbors=2, gene_neighbors=2),
            work_root=tmp_path,
        )

    assert captured.value.reason_code == "matched_bulk_zero_library_sample"
    assert (
        captured.value.attempt_receipt.reference_matrix_sha256
        == reference.matrix_sha256
    )


def test_sctsi_rejects_zero_library_single_cell_with_attempt_receipt(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6, zero_library=True)
    reference = _bulk_reference(method_input)

    with pytest.raises(SCTSIUnavailableError) as captured:
        run_sctsi(
            _registry().by_id("sctsi"),
            method_input,
            bulk_reference=reference,
            source_dir=_cached_source(),
            rscript=tmp_path / "missing-rscript",
            r_library=tmp_path / "missing-library",
            config=SCTSIConfig(cell_neighbors=2, gene_neighbors=2),
            work_root=tmp_path,
        )

    assert captured.value.reason_code == "zero_library_cell"
    assert (
        captured.value.attempt_receipt.reference_matrix_sha256
        == reference.matrix_sha256
    )


def test_sctsi_source_boundary_matches_frozen_pin() -> None:
    source = _cached_source()
    before = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    receipt = benchmark_methods.verify_pinned_source(_registry().by_id("sctsi"), source)

    assert receipt.revision == "402cc9723696ede77d5864e51368c3d94be3a29c"
    assert receipt.tree == "a897933047ad09985d0aa311dcae4a6e40e7db11"
    assert before == ""
    assert (
        subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_sctsi_missing_bulk_fails_with_source_bound_attempt_receipt(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)

    with pytest.raises(SCTSIUnavailableError) as captured:
        run_sctsi(
            _registry().by_id("sctsi"),
            method_input,
            bulk_reference=None,
            source_dir=_cached_source(),
            rscript=tmp_path / "missing-rscript",
            r_library=tmp_path / "missing-library",
            config=SCTSIConfig(cell_neighbors=2, gene_neighbors=2),
            work_root=tmp_path,
        )

    error = captured.value
    assert error.reason_code == "matched_bulk_reference_missing"
    assert error.command is None
    assert error.attempt_receipt.source_revision == (
        "402cc9723696ede77d5864e51368c3d94be3a29c"
    )
    assert error.attempt_receipt.reference_id is None
    assert error.attempt_receipt.environment_id == "sctsi-r"


def test_sctsi_gene_mismatch_fails_before_environment_execution(tmp_path: Path) -> None:
    method_input = _method_input(cells=8, genes=6)
    reference = replace(
        _bulk_reference(method_input),
        var_ids=("wrong", *method_input.var_ids[1:]),
    )

    with pytest.raises(SCTSIUnavailableError) as captured:
        run_sctsi(
            _registry().by_id("sctsi"),
            method_input,
            bulk_reference=reference,
            source_dir=_cached_source(),
            rscript=tmp_path / "missing-rscript",
            r_library=tmp_path / "missing-library",
            config=SCTSIConfig(cell_neighbors=2, gene_neighbors=2),
            work_root=tmp_path,
        )

    error = captured.value
    assert error.reason_code == "matched_bulk_gene_mismatch"
    assert error.command is None
    assert error.attempt_receipt.reference_id == reference.reference_id
    assert error.attempt_receipt.reference_source_sha256 == BULK_SHA


def test_sctsi_rejects_work_root_inside_pinned_source(tmp_path: Path) -> None:
    source = _cached_source()
    nested = source / "adapter-work-must-not-be-created"
    assert not nested.exists()

    with pytest.raises(AdapterUnavailableError) as captured:
        run_sctsi(
            _registry().by_id("sctsi"),
            _method_input(cells=8, genes=6),
            bulk_reference=None,
            source_dir=source,
            rscript=tmp_path / "missing-rscript",
            r_library=tmp_path / "missing-library",
            config=SCTSIConfig(cell_neighbors=2, gene_neighbors=2),
            work_root=nested,
        )

    assert captured.value.reason_code == "unsafe_work_root"
    assert not nested.exists()
    assert (
        subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_sctsi_missing_r_environment_has_complete_attempt_receipt(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    reference = _bulk_reference(method_input)

    with pytest.raises(SCTSIUnavailableError) as captured:
        run_sctsi(
            _registry().by_id("sctsi"),
            method_input,
            bulk_reference=reference,
            source_dir=_cached_source(),
            rscript=tmp_path / "missing-rscript",
            r_library=tmp_path / "missing-library",
            config=SCTSIConfig(cell_neighbors=2, gene_neighbors=2),
            work_root=tmp_path,
        )

    error = captured.value
    assert error.reason_code == "environment_executable_missing"
    assert error.command is None
    assert error.attempt_receipt.reference_matrix_sha256 == reference.matrix_sha256
    assert error.attempt_receipt.executable == str(tmp_path / "missing-rscript")
    assert error.attempt_receipt.stdout_sha256 == error.stdout_sha256
    assert error.attempt_receipt.stderr_sha256 == error.stderr_sha256


def test_sctsi_runtime_failure_retains_source_environment_and_bulk_evidence(
    tmp_path: Path,
) -> None:
    method_input = _method_input(cells=8, genes=6)
    reference = _bulk_reference(method_input)
    r_library = tmp_path / "r-library"
    r_library.mkdir()

    with pytest.raises(SCTSIUnavailableError) as captured:
        run_sctsi(
            _registry().by_id("sctsi"),
            method_input,
            bulk_reference=reference,
            source_dir=_cached_source(),
            rscript=_fake_rscript(tmp_path),
            r_library=r_library,
            config=SCTSIConfig(cell_neighbors=2, gene_neighbors=2),
            work_root=tmp_path,
        )

    error = captured.value
    attempt = error.attempt_receipt
    assert error.reason_code == "upstream_nonzero_exit"
    assert error.command is not None
    assert b"fake-sctsi-stdout" in error.stdout
    assert b"fake-sctsi-stderr" in error.stderr
    assert b"R_LIBS=\n" in error.stdout
    site_line = next(
        line for line in error.stdout.splitlines() if line.startswith(b"R_LIBS_SITE=")
    )
    assert site_line.endswith(b"/empty-site-library")
    assert attempt.reference_id == reference.reference_id
    assert attempt.reference_source_sha256 == reference.source_sha256
    assert attempt.reference_matrix_sha256 == reference.matrix_sha256
    assert attempt.r_library == str(r_library)
    assert attempt.command == error.command
    assert attempt.stdout_sha256 == error.stdout_sha256
    assert attempt.stderr_sha256 == error.stderr_sha256


def test_real_pinned_sctsi_tiny_smoke_when_isolated_r_environment_exists(
    tmp_path: Path,
) -> None:
    rscript = Path(shutil.which("Rscript") or "/missing-rscript")
    r_library = Path("/tmp/maskimpute-sctsi-r-lib")
    if not rscript.is_file() or not r_library.is_dir():
        pytest.skip("isolated scTsI R environment is absent")
    probe = subprocess.run(
        (
            str(rscript),
            "--vanilla",
            "-e",
            "lib<-commandArgs(TRUE)[1];.libPaths(c(lib,.Library));p<-c('mclust','devtools','fpc','ngram','FNN','Matrix','Metrics','glmnet');quit(status=if(all(vapply(p,requireNamespace,logical(1),quietly=TRUE)))0 else 3)",
            str(r_library),
        ),
        check=False,
        capture_output=True,
        timeout=30,
    )
    if probe.returncode != 0:
        pytest.skip("isolated scTsI R dependency installation is incomplete")
    method_input = _method_input(cells=8, genes=6)
    reference = _bulk_reference(method_input)
    config = SCTSIConfig(cell_neighbors=2, gene_neighbors=2)

    first = run_sctsi(
        _registry().by_id("sctsi"),
        method_input,
        bulk_reference=reference,
        source_dir=_cached_source(),
        rscript=rscript,
        r_library=r_library,
        config=config,
        work_root=tmp_path,
    )
    second = run_sctsi(
        _registry().by_id("sctsi"),
        method_input,
        bulk_reference=reference,
        source_dir=_cached_source(),
        rscript=rscript,
        r_library=r_library,
        config=config,
        work_root=tmp_path,
    )

    np.testing.assert_array_equal(first.snapshot.matrix, second.snapshot.matrix)
    evaluator_counts = benchmark_methods.external_reference_output_to_evaluator_counts(
        method_input, first.snapshot
    )
    observed = method_input.counts > 0
    np.testing.assert_allclose(
        evaluator_counts[observed],
        method_input.counts[observed],
        rtol=1e-12,
        atol=1e-12,
    )
    assert first.command is not None
    receipt = dict(first.environment_receipt)
    assert receipt["single_cell_input_scale"] == "cpm"
    assert receipt["bulk_reference_input_scale"] == "raw_counts"
    assert receipt["bulk_constraint_scale"] == "cpm"
    assert receipt["sctsi_native_output_scale"] == "cpm"
    assert receipt["cpm_target"] == "1000000"
    assert receipt["r_library_paths"].split(";")[0] == str(r_library)
    assert len(receipt["r_library_paths"].split(";")) == 2
    assert receipt["bulk_reference_id"] == reference.reference_id
    assert receipt["bulk_reference_source_sha256"] == reference.source_sha256
    assert receipt["bulk_reference_matrix_sha256"] == reference.matrix_sha256
    assert receipt["sctsi_source_file"].endswith("/code/scTsI.R")
    assert {event.code for event in first.compatibility_log} >= {
        "bulk_average_contract",
        "input_orientation",
        "deterministic_execution",
        "published_demo_truth_exclusion",
        "upstream_parameter_override",
    }
