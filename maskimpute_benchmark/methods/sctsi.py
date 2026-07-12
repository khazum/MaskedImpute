"""Pinned scTsI adapter with an immutable matched-bulk reference contract."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from .base import (
    MethodContractError,
    MethodInput,
    MethodOutputSnapshot,
    MethodSpec,
    snapshot_method_output,
)
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    SourceReceipt,
    execute_pinned_command,
    observed_library_sizes,
    read_environment_receipt,
    read_raw_output,
    require_executable,
    verify_pinned_source,
    write_raw_matrix,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


_SCTSI_DRIVER = r"""
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 14) stop("adapter expected fourteen arguments")
source_file <- normalizePath(args[[1]], mustWork=TRUE)
input_file <- args[[2]]
bulk_file <- args[[3]]
output_file <- args[[4]]
receipt_file <- args[[5]]
library_dir <- normalizePath(args[[6]], mustWork=TRUE)
n_genes <- as.integer(args[[7]])
n_cells <- as.integer(args[[8]])
threshold <- as.numeric(args[[9]])
cell_neighbors <- as.integer(args[[10]])
gene_neighbors <- as.integer(args[[11]])
reference_id <- args[[12]]
reference_source_sha256 <- args[[13]]
reference_matrix_sha256 <- args[[14]]

.libPaths(c(library_dir, .Library))
resolved_library_paths <- normalizePath(.libPaths(), mustWork=TRUE)
allowed_library_paths <- c(library_dir, normalizePath(.Library, mustWork=TRUE))
if (!identical(resolved_library_paths, allowed_library_paths)) {
  stop(paste(
    "isolated R library path violation",
    paste(resolved_library_paths, collapse=","),
    sep=": "
  ))
}
required_packages <- c(
  "mclust", "devtools", "fpc", "ngram", "FNN", "Matrix", "Metrics", "glmnet"
)
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1), quietly=TRUE
)]
if (length(missing_packages)) {
  stop(paste("there is no package called", paste(missing_packages, collapse=",")))
}
nonbase_packages <- setdiff(required_packages, "Matrix")
for (package in nonbase_packages) {
  package_path <- normalizePath(find.package(package), mustWork=TRUE)
  if (!startsWith(package_path, paste0(library_dir, .Platform$file.sep))) {
    stop(paste("package is outside isolated R library", package))
  }
}

input_connection <- file(input_file, open="rb")
input_values <- readBin(
  input_connection, what="double", n=n_genes*n_cells, size=8, endian="little"
)
close(input_connection)
if (length(input_values) != n_genes*n_cells) stop("input byte count differs")
gene_by_cell <- matrix(
  input_values, nrow=n_genes, ncol=n_cells, byrow=TRUE
)
if (any(!is.finite(gene_by_cell)) || any(gene_by_cell < 0)) {
  stop("single-cell input is invalid")
}

bulk_connection <- file(bulk_file, open="rb")
bulk_average <- readBin(
  bulk_connection, what="double", n=n_genes, size=8, endian="little"
)
close(bulk_connection)
if (length(bulk_average) != n_genes) stop("bulk byte count differs")
if (any(!is.finite(bulk_average)) || any(bulk_average < 0)) {
  stop("matched-bulk average is invalid")
}

source_environment <- new.env(parent=globalenv())
source(source_file, local=source_environment, chdir=FALSE)
if (!exists("scTsI", envir=source_environment, mode="function", inherits=FALSE)) {
  stop("MASKIMPUTE_SCTSI_UPSTREAM_INCOMPLETE: scTsI function is absent")
}
result <- source_environment$scTsI(
  data_sc=gene_by_cell,
  threshold=threshold,
  data_bulk=bulk_average,
  k1=cell_neighbors,
  k2=gene_neighbors
)
result <- as.matrix(result)
if (!identical(dim(result), c(n_genes, n_cells))) stop("output shape differs")
if (any(!is.finite(result)) || any(result < 0)) stop("output is invalid")

# R stores a genes-by-cells matrix column-major. That byte order is exactly
# cells-by-genes row-major for the Python evaluator.
output_connection <- file(output_file, open="wb")
writeBin(as.double(result), output_connection, size=8, endian="little")
close(output_connection)

receipt <- c(
  paste("bulk_constraint_scale", "cpm", sep="\t"),
  paste("bulk_reference_id", reference_id, sep="\t"),
  paste("bulk_reference_input_scale", "raw_counts", sep="\t"),
  paste("bulk_reference_matrix_sha256", reference_matrix_sha256, sep="\t"),
  paste("bulk_reference_source_sha256", reference_source_sha256, sep="\t"),
  paste("cpm_target", "1000000", sep="\t"),
  paste("devtools_version", as.character(utils::packageVersion("devtools")), sep="\t"),
  paste("fnn_version", as.character(utils::packageVersion("FNN")), sep="\t"),
  paste("fpc_version", as.character(utils::packageVersion("fpc")), sep="\t"),
  paste("glmnet_version", as.character(utils::packageVersion("glmnet")), sep="\t"),
  paste("matrix_version", as.character(utils::packageVersion("Matrix")), sep="\t"),
  paste("mclust_version", as.character(utils::packageVersion("mclust")), sep="\t"),
  paste("metrics_version", as.character(utils::packageVersion("Metrics")), sep="\t"),
  paste("ngram_version", as.character(utils::packageVersion("ngram")), sep="\t"),
  paste("r_library", library_dir, sep="\t"),
  paste("r_library_paths", paste(resolved_library_paths, collapse=";"), sep="\t"),
  paste("r_version", R.version.string, sep="\t"),
  paste("sctsi_native_output_scale", "cpm", sep="\t"),
  paste("single_cell_input_scale", "cpm", sep="\t"),
  paste("sctsi_source_file", source_file, sep="\t")
)
writeLines(receipt, receipt_file, useBytes=TRUE)
"""


def _identifiers(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise TypeError(f"{name} must be a nonempty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    return values


def _bulk_digest(
    reference_id: str,
    source_sha256: str,
    expression_scale: str,
    var_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    shape: tuple[int, int],
    matrix_bytes: bytes,
) -> str:
    binding = json.dumps(
        {
            "reference_id": reference_id,
            "source_sha256": source_sha256,
            "expression_scale": expression_scale,
            "var_ids": var_ids,
            "sample_ids": sample_ids,
            "shape": shape,
            "dtype": "<f8",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(binding)
    digest.update(matrix_bytes)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SCTSIMatchedBulkReference:
    """Immutable raw gene-by-sample bulk counts for the scTsI constraint."""

    reference_id: str
    source_sha256: str
    expression_scale: str
    var_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    shape: tuple[int, int]
    matrix_sha256: str
    _matrix_bytes: bytes = field(repr=False)

    @property
    def matrix(self) -> np.ndarray:
        return np.frombuffer(self._matrix_bytes, dtype="<f8").reshape(self.shape)


def prepare_sctsi_matched_bulk_reference(
    *,
    reference_id: str,
    source_sha256: str,
    matrix: object,
    var_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    expression_scale: str,
) -> SCTSIMatchedBulkReference:
    """Validate and bind prespecified raw bulk replicates without averaging yet."""

    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError("reference_id must be a nonempty string")
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256")
    if expression_scale != "raw_counts":
        raise ValueError("expression_scale must be raw_counts")
    genes = _identifiers(var_ids, "var_ids")
    samples = _identifiers(sample_ids, "sample_ids")
    if type(matrix) is not np.ndarray or matrix.ndim != 2:
        raise TypeError("matched bulk matrix must be an exact two-dimensional ndarray")
    if matrix.shape != (len(genes), len(samples)):
        raise ValueError("matched bulk matrix shape does not match its IDs")
    if matrix.dtype.metadata is not None:
        raise ValueError("matched bulk matrix must not use dtype metadata")
    if matrix.dtype.kind == "b":
        raise ValueError("matched bulk matrix must not contain boolean values")
    if matrix.dtype.kind not in {"i", "u", "f"} or matrix.dtype.itemsize > 8:
        raise ValueError(
            "matched bulk matrix must use native numeric values up to float64"
        )
    if matrix.dtype.kind in {"i", "u"}:
        for value in matrix.reshape(-1):
            integer = int(value)
            if int(float(integer)) != integer:
                raise ValueError(
                    "matched bulk matrix integer values must be exactly representable as float64"
                )
    values = np.array(matrix, dtype="<f8", copy=True, order="C", subok=False)
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("matched bulk matrix must be finite and nonnegative")
    if not bool((values == np.floor(values)).all()):
        raise ValueError("matched bulk matrix must contain integer raw counts")
    matrix_bytes = values.tobytes(order="C")
    shape = tuple(values.shape)
    return SCTSIMatchedBulkReference(
        reference_id=reference_id,
        source_sha256=source_sha256,
        expression_scale=expression_scale,
        var_ids=genes,
        sample_ids=samples,
        shape=shape,
        matrix_sha256=_bulk_digest(
            reference_id,
            source_sha256,
            expression_scale,
            genes,
            samples,
            shape,
            matrix_bytes,
        ),
        _matrix_bytes=matrix_bytes,
    )


def validate_sctsi_matched_bulk_reference(
    method_input: MethodInput,
    reference: SCTSIMatchedBulkReference,
) -> np.ndarray:
    """Validate bulk bytes and exact gene alignment to a truth-free input."""

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if not isinstance(reference, SCTSIMatchedBulkReference):
        raise TypeError("reference must be an SCTSIMatchedBulkReference")
    if reference.expression_scale != "raw_counts":
        raise ValueError("matched bulk expression_scale must be raw_counts")
    if reference.var_ids != method_input.var_ids:
        raise ValueError("matched bulk gene IDs do not match the method input")
    if reference.shape != (method_input.shape[1], len(reference.sample_ids)):
        raise ValueError("matched bulk shape does not match the method input")
    try:
        matrix = reference.matrix
    except ValueError as error:
        raise ValueError("matched bulk bytes do not match its shape") from error
    matrix_bytes = np.asarray(matrix, dtype="<f8", order="C").tobytes(order="C")
    expected_hash = _bulk_digest(
        reference.reference_id,
        reference.source_sha256,
        reference.expression_scale,
        reference.var_ids,
        reference.sample_ids,
        reference.shape,
        matrix_bytes,
    )
    if reference.matrix_sha256 != expected_hash:
        raise ValueError("matched bulk matrix hash does not match its bound content")
    return matrix


@dataclass(frozen=True, slots=True)
class SCTSIConfig:
    """Exact pinned scTsI defaults."""

    threshold: float = 0.0
    cell_neighbors: int = 25
    gene_neighbors: int = 25

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(float(self.threshold))
            or self.threshold < 0
        ):
            raise ValueError("threshold must be finite and nonnegative")
        for name, value in (
            ("cell_neighbors", self.cell_neighbors),
            ("gene_neighbors", self.gene_neighbors),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SCTSIAttemptReceipt:
    """Immutable source, environment, reference, command, and log evidence."""

    source_revision: str
    source_tree: str
    source_url: str
    environment_id: str
    environment_registry_status: str
    executable: str
    r_library: str
    reference_id: str | None
    reference_source_sha256: str | None
    reference_matrix_sha256: str | None
    outcome: str
    reason_code: str
    command: tuple[str, ...] | None
    stdout_sha256: str
    stderr_sha256: str


class SCTSIUnavailableError(AdapterUnavailableError):
    """Unavailable scTsI attempt with complete immutable evidence."""

    def __init__(
        self,
        error: AdapterUnavailableError,
        *,
        source: SourceReceipt,
        spec: MethodSpec,
        executable: Path,
        r_library: Path,
        reference: SCTSIMatchedBulkReference | None,
    ) -> None:
        super().__init__(
            error.reason_code,
            error.detail,
            command=error.command,
            stdout=error.stdout,
            stderr=error.stderr,
        )
        self.attempt_receipt = SCTSIAttemptReceipt(
            source_revision=source.revision,
            source_tree=source.tree,
            source_url=source.url,
            environment_id=spec.environment.id,
            environment_registry_status=spec.environment.status,
            executable=str(executable),
            r_library=str(r_library),
            reference_id=(None if reference is None else reference.reference_id),
            reference_source_sha256=(
                None if reference is None else reference.source_sha256
            ),
            reference_matrix_sha256=(
                None if reference is None else reference.matrix_sha256
            ),
            outcome="unavailable",
            reason_code=self.reason_code,
            command=self.command,
            stdout_sha256=self.stdout_sha256,
            stderr_sha256=self.stderr_sha256,
        )


def _require_sctsi_spec(spec: MethodSpec) -> None:
    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if spec.id != "sctsi":
        raise ValueError(f"expected method sctsi, received {spec.id}")
    if spec.track != "external_reference":
        raise ValueError("sctsi must use the external_reference track")
    if spec.input_scale != "raw_counts":
        raise ValueError("sctsi input scale must be raw_counts")
    if spec.output_scale != "external_reference_adjusted":
        raise ValueError("sctsi output scale must be external_reference_adjusted")
    if spec.stochastic or spec.seed_policy != "not_applicable":
        raise ValueError("sctsi must remain deterministic without a seed")


def _validate_work_root(work_root: Path | None, source_dir: Path) -> None:
    if work_root is None:
        return
    if not isinstance(work_root, Path):
        raise TypeError("work_root must be a pathlib.Path or None")
    source = source_dir.resolve(strict=True)
    candidate = work_root.resolve(strict=False)
    if candidate == source or source in candidate.parents:
        raise AdapterUnavailableError(
            "unsafe_work_root",
            "work_root must not be the pinned source tree or a directory inside it",
        )
    work_root.mkdir(parents=True, exist_ok=True)
    created = work_root.resolve(strict=True)
    if created == source or source in created.parents:
        raise AdapterUnavailableError(
            "unsafe_work_root",
            "resolved work_root must remain outside the pinned source tree",
        )


def _require_r_library(r_library: Path, source_dir: Path) -> Path:
    if not isinstance(r_library, Path):
        raise TypeError("r_library must be a pathlib.Path")
    if not r_library.is_absolute() or ".." in r_library.parts:
        raise AdapterUnavailableError(
            "environment_library_unsafe",
            "R library must be an absolute path without parent traversal",
        )
    try:
        library = r_library.resolve(strict=True)
    except OSError as error:
        raise AdapterUnavailableError(
            "environment_library_missing",
            f"isolated scTsI R library does not exist: {r_library}",
        ) from error
    if not library.is_dir():
        raise AdapterUnavailableError(
            "environment_library_unsafe", "isolated scTsI R library is not a directory"
        )
    source = source_dir.resolve(strict=True)
    if library == source or source in library.parents:
        raise AdapterUnavailableError(
            "environment_library_unsafe",
            "isolated R library must remain outside the pinned source tree",
        )
    return library


def _reference_error_reason(error: Exception) -> str:
    text = str(error)
    if "gene IDs" in text:
        return "matched_bulk_gene_mismatch"
    if "hash" in text or "bytes" in text:
        return "matched_bulk_reference_tampered"
    return "matched_bulk_reference_invalid"


def _classify_runtime_error(error: AdapterUnavailableError) -> AdapterUnavailableError:
    combined = error.stdout + b"\n" + error.stderr
    if b"MASKIMPUTE_SCTSI_UPSTREAM_INCOMPLETE" in combined:
        return AdapterUnavailableError(
            "upstream_incomplete",
            "pinned scTsI source does not expose the declared scTsI function",
            command=error.command,
            stdout=error.stdout,
            stderr=error.stderr,
        )
    return error


def sctsi_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Invert native CPM using each cell's observed raw-count library size."""

    libraries = observed_library_sizes(method_input)
    if type(native_output) is not np.ndarray:
        raise TypeError("native scTsI CPM output must be an exact ndarray")
    if native_output.shape != method_input.shape:
        raise ValueError("native scTsI CPM output must match the method-input shape")
    if native_output.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("native scTsI CPM output must be numeric")
    values = np.array(
        native_output,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("native scTsI CPM output must be finite and nonnegative")
    converted = values * libraries[:, None] / 1_000_000.0
    if not np.isfinite(converted).all():
        raise ValueError("native scTsI CPM output has no finite count equivalent")
    converted.setflags(write=False)
    return converted


def _single_cell_counts_to_cpm(method_input: MethodInput) -> np.ndarray:
    libraries = observed_library_sizes(method_input)
    return (
        np.asarray(method_input.counts, dtype=np.float64)
        / libraries[:, None]
        * 1_000_000.0
    )


def _bulk_counts_to_average_cpm(bulk_matrix: np.ndarray) -> np.ndarray:
    libraries = np.sum(bulk_matrix, axis=0, dtype=np.float64)
    if bool((libraries == 0).any()):
        raise AdapterUnavailableError(
            "matched_bulk_zero_library_sample",
            "scTsI bulk CPM is undefined for a zero-library bulk sample",
        )
    bulk_cpm = bulk_matrix / libraries[None, :] * 1_000_000.0
    return np.mean(bulk_cpm, axis=1, dtype=np.float64)


def finalize_sctsi_output(
    spec: MethodSpec,
    method_input: MethodInput,
    adjusted_output: object,
) -> MethodOutputSnapshot:
    """Bind untouched external-reference-adjusted output to evaluator IDs."""

    _require_sctsi_spec(spec)
    return snapshot_method_output(
        spec,
        method_input,
        adjusted_output,
        source_dataset_sha256=method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def run_sctsi(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    bulk_reference: SCTSIMatchedBulkReference | None,
    source_dir: Path,
    rscript: Path,
    r_library: Path,
    config: SCTSIConfig = SCTSIConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    """Run exact pinned scTsI with a prespecified matched-bulk constraint."""

    _require_sctsi_spec(spec)
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if not isinstance(config, SCTSIConfig):
        raise TypeError("config must be an SCTSIConfig")
    _validate_work_root(work_root, source_dir)
    source_receipt = verify_pinned_source(spec, source_dir)

    def unavailable(error: AdapterUnavailableError) -> SCTSIUnavailableError:
        return SCTSIUnavailableError(
            error,
            source=source_receipt,
            spec=spec,
            executable=rscript,
            r_library=r_library,
            reference=(
                bulk_reference
                if isinstance(bulk_reference, SCTSIMatchedBulkReference)
                else None
            ),
        )

    if bulk_reference is None:
        error = AdapterUnavailableError(
            "matched_bulk_reference_missing",
            "scTsI requires one prespecified matched-bulk reference",
        )
        raise unavailable(error)
    try:
        bulk_matrix = validate_sctsi_matched_bulk_reference(
            method_input, bulk_reference
        )
    except (TypeError, ValueError) as original:
        error = AdapterUnavailableError(
            _reference_error_reason(original),
            f"matched-bulk reference is invalid: {original}",
        )
        raise unavailable(error) from original
    try:
        single_cell_cpm = _single_cell_counts_to_cpm(method_input)
        bulk_average_cpm = _bulk_counts_to_average_cpm(bulk_matrix)
    except AdapterUnavailableError as original:
        raise unavailable(original) from original

    cells, genes = method_input.shape
    if cells <= config.cell_neighbors or genes <= config.gene_neighbors:
        error = AdapterUnavailableError(
            "upstream_minimum_dimension",
            f"scTsI requires cells>{config.cell_neighbors} and genes>{config.gene_neighbors}",
        )
        raise unavailable(error)
    if not bool((method_input.counts == 0).any()):
        error = AdapterUnavailableError(
            "upstream_no_missing_entries",
            "pinned scTsI cannot construct its zero-position regression without zeros",
        )
        raise unavailable(error)
    try:
        executable = require_executable(rscript)
        library = _require_r_library(r_library, source_dir)
    except AdapterUnavailableError as original:
        raise unavailable(original) from original

    compatibility = [
        CompatibilityEvent(
            "input_scale_conversion",
            "frozen raw single-cell counts are normalized per cell to CPM before exact upstream execution",
        ),
        CompatibilityEvent(
            "input_orientation",
            "adapter transposes truth-free cells-by-genes counts to upstream genes-by-cells and maps the returned genes-by-cells matrix back without reordering",
        ),
        CompatibilityEvent(
            "bulk_average_contract",
            "prespecified measured raw bulk samples are normalized independently to CPM, then averaged gene-wise to form the published length-m bulk constraint vector d",
        ),
        CompatibilityEvent(
            "published_demo_truth_exclusion",
            "the pinned simulation demo computes CPM from complete TrueCounts before masking and derives its bulk vector from true CPM; those truth-derived preparations are forbidden here, so the adapter uses observed-input libraries and a prespecified measured bulk reference while invoking scTsI.R unchanged",
        ),
        CompatibilityEvent(
            "upstream_defaults",
            f"threshold={config.threshold}, k1={config.cell_neighbors}, k2={config.gene_neighbors}, glmnet alpha=0, intercept=FALSE, nlambda=10, beta column 5",
        ),
        CompatibilityEvent(
            "deterministic_execution",
            "pinned FNN and glmnet path has no random sampling; no seed is accepted or set",
        ),
        CompatibilityEvent(
            "upstream_selective_policy",
            "pinned source updates zero positions, retains nonzero positions through its permutation, and clips negative final values to zero; adapter adds no clipping or copying",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native scTsI CPM is inverted with each cell's observed raw-count library before the shared log2(CP10k+1) evaluator transform",
        ),
        CompatibilityEvent(
            "source_policy",
            "adapter sources the pristine pinned scTsI.R file in an isolated R library and does not redistribute or reimplement the algorithm",
        ),
    ]
    if config != SCTSIConfig():
        compatibility.append(
            CompatibilityEvent(
                "upstream_parameter_override",
                "threshold, k1, or k2 differs from the pinned function defaults",
            )
        )

    with TemporaryDirectory(prefix="maskimpute-sctsi-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.bin"
        bulk_path = work_dir / "bulk-average.bin"
        output_path = work_dir / "output.bin"
        receipt_path = work_dir / "receipt.tsv"
        empty_site_library = work_dir / "empty-site-library"
        write_raw_matrix(input_path, single_cell_cpm.T)
        write_raw_matrix(bulk_path, bulk_average_cpm[:, None])
        source_file = source_dir / "code" / "scTsI.R"
        command = (
            str(executable),
            "--vanilla",
            "-e",
            _SCTSI_DRIVER,
            str(source_file.resolve()),
            str(input_path),
            str(bulk_path),
            str(output_path),
            str(receipt_path),
            str(library),
            str(genes),
            str(cells),
            repr(float(config.threshold)),
            str(config.cell_neighbors),
            str(config.gene_neighbors),
            bulk_reference.reference_id,
            bulk_reference.source_sha256,
            bulk_reference.matrix_sha256,
        )
        try:
            result = execute_pinned_command(
                spec,
                source_dir,
                command,
                cwd=work_dir,
                timeout_seconds=spec.resources.timeout_seconds,
                environment={
                    "MKL_NUM_THREADS": str(spec.resources.cpu_cores),
                    "OMP_NUM_THREADS": str(spec.resources.cpu_cores),
                    "R_LIBS": "",
                    "R_LIBS_SITE": str(empty_site_library),
                    "R_LIBS_USER": str(library),
                },
            )
            output = read_raw_output(output_path, method_input.shape)
            receipt = read_environment_receipt(
                receipt_path,
                expected_keys=frozenset(
                    {
                        "bulk_reference_id",
                        "bulk_constraint_scale",
                        "bulk_reference_input_scale",
                        "bulk_reference_matrix_sha256",
                        "bulk_reference_source_sha256",
                        "cpm_target",
                        "devtools_version",
                        "fnn_version",
                        "fpc_version",
                        "glmnet_version",
                        "matrix_version",
                        "mclust_version",
                        "metrics_version",
                        "ngram_version",
                        "r_library",
                        "r_library_paths",
                        "r_version",
                        "sctsi_native_output_scale",
                        "single_cell_input_scale",
                        "sctsi_source_file",
                    }
                ),
            )
            snapshot = finalize_sctsi_output(spec, method_input, output)
        except AdapterUnavailableError as original:
            raise unavailable(_classify_runtime_error(original)) from original
        except MethodContractError as original:
            error = AdapterUnavailableError(
                "malformed_upstream_output",
                f"scTsI output violates the benchmark contract: {original}",
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            raise unavailable(error) from original
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


__all__ = [
    "SCTSIAttemptReceipt",
    "SCTSIConfig",
    "SCTSIMatchedBulkReference",
    "SCTSIUnavailableError",
    "finalize_sctsi_output",
    "prepare_sctsi_matched_bulk_reference",
    "run_sctsi",
    "sctsi_to_evaluator_counts",
    "validate_sctsi_matched_bulk_reference",
]
