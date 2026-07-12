"""Pinned SAVER adapter preserving its native normalized-expression output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .observed import (
    AdapterExecution,
    CompatibilityEvent,
    _validated_native_matrix,
    execute_pinned_command,
    observed_library_sizes,
    read_environment_receipt,
    read_raw_output,
    require_executable,
    require_method_spec,
    write_raw_matrix,
)


_SAVER_DRIVER = r"""
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 10) stop("adapter expected ten arguments")
source_dir <- normalizePath(args[[1]], mustWork=TRUE)
input_file <- args[[2]]
output_file <- args[[3]]
receipt_file <- args[[4]]
library_dir <- args[[5]]
n_obs <- as.integer(args[[6]])
n_vars <- as.integer(args[[7]])
ncores <- as.integer(args[[8]])
do_fast <- identical(args[[9]], "TRUE")
seed <- as.integer(args[[10]])
required_packages <- c("Matrix", "doParallel", "foreach", "glmnet", "iterators")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                               logical(1), quietly=TRUE)]
if (length(missing_packages)) {
  stop(paste("there is no package called", paste(missing_packages, collapse=",")))
}
dir.create(library_dir, recursive=TRUE, showWarnings=FALSE)
utils::install.packages(source_dir, repos=NULL, type="source", lib=library_dir,
                        quiet=TRUE)
.libPaths(c(library_dir, .libPaths()))
suppressPackageStartupMessages(library("SAVER", character.only=TRUE,
                                       lib.loc=library_dir))
input_connection <- file(input_file, open="rb")
on.exit(close(input_connection), add=TRUE)
values <- readBin(input_connection, what="double", n=n_obs*n_vars,
                  size=8, endian="little")
if (length(values) != n_obs*n_vars) stop("input byte count differs")
cell_by_gene <- matrix(values, nrow=n_obs, ncol=n_vars, byrow=TRUE)
if (any(!is.finite(cell_by_gene)) || any(cell_by_gene < 0)) stop("input is invalid")
gene_by_cell <- t(cell_by_gene)
set.seed(seed)
result <- SAVER::saver(gene_by_cell, do.fast=do_fast, ncores=ncores,
                       size.factor=NULL, estimates.only=TRUE)
output <- t(as.matrix(result))
if (!identical(dim(output), c(n_obs, n_vars))) stop("output shape differs")
if (any(!is.finite(output)) || any(output < 0)) stop("output is invalid")
output_connection <- file(output_file, open="wb")
writeBin(as.double(t(output)), output_connection, size=8, endian="little")
close(output_connection)
receipt <- c(
  paste("r_version", R.version.string, sep="\t"),
  paste("saver_source_dir", source_dir, sep="\t"),
  paste("saver_version", as.character(utils::packageVersion("SAVER")), sep="\t"),
  paste("glmnet_version", as.character(utils::packageVersion("glmnet")), sep="\t"),
  paste("matrix_version", as.character(utils::packageVersion("Matrix")), sep="\t")
)
writeLines(receipt, receipt_file, useBytes=TRUE)
"""


@dataclass(frozen=True, slots=True)
class SAVERConfig:
    """Exact algorithmic SAVER defaults; estimates_only changes only packaging."""

    do_fast: bool = True
    ncores: int = 1
    size_factor: None = None
    estimates_only: bool = True

    def __post_init__(self) -> None:
        if self.do_fast is not True:
            raise ValueError("do_fast must remain True")
        if type(self.ncores) is not int or self.ncores <= 0:
            raise ValueError("ncores must be a positive integer")
        if self.size_factor is not None:
            raise ValueError("size_factor must remain null for upstream normalization")
        if self.estimates_only is not True:
            raise ValueError("estimates_only must remain True")


def saver_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Undo SAVER's pinned ``size.factor=NULL`` output normalization.

    Pinned ``calc.size.factor`` sets ``sf=library/mean(library)`` and
    ``scale.sf=1``. Pinned ``calc.post`` therefore returns normalized posterior
    means, so count equivalents are ``native * sf`` row-wise.
    """

    libraries = observed_library_sizes(method_input)
    normalized = _validated_native_matrix(
        method_input,
        native_output,
        name="native SAVER normalized output",
    )
    size_factors = libraries / libraries.mean()
    converted = np.array(
        normalized * size_factors[:, None],
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if not np.isfinite(converted).all() or bool((converted < 0).any()):
        raise ValueError("SAVER output does not have a finite count equivalent")
    converted.setflags(write=False)
    return converted


def finalize_saver_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> MethodOutputSnapshot:
    """Bind SAVER's native library-normalized posterior mean without alteration."""

    require_method_spec(
        spec,
        "saver",
        input_scale="raw_counts",
        output_scale="method_native_normalized",
    )
    return snapshot_method_output(
        spec,
        method_input,
        normalized_output,
        source_dataset_sha256=method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def run_saver(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    rscript: Path,
    seed: int,
    config: SAVERConfig = SAVERConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    """Run pristine pinned SAVER sources in an explicit R dependency environment."""

    require_method_spec(
        spec,
        "saver",
        input_scale="raw_counts",
        output_scale="method_native_normalized",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if not isinstance(config, SAVERConfig):
        raise TypeError("config must be a SAVERConfig")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2^31)")
    executable = require_executable(rscript)
    compatibility = (
        CompatibilityEvent("input_scale_conversion", "none; SAVER receives raw counts"),
        CompatibilityEvent(
            "upstream_defaults",
            "do.fast=TRUE, size.factor=NULL, all genes/cells; ncores is an execution resource",
        ),
        CompatibilityEvent(
            "return_container_only",
            "estimates.only=TRUE returns the same posterior mean without uncertainty fields",
        ),
        CompatibilityEvent(
            "seed_binding",
            "study seed is passed to set.seed before SAVER's randomized gene ordering",
        ),
        CompatibilityEvent(
            "output_convention",
            "returned full upstream library-size-normalized posterior mean",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native snapshot remains normalized: pinned size.factor=NULL gives sf=observed_library_size/mean_observed_library_size and scale.sf=1, so evaluator counts are native*sf before the shared log2(CP10k+1) transform; zero-library rows fail closed",
        ),
        CompatibilityEvent(
            "observed_positive_policy", "no observed entries are copied after SAVER"
        ),
        CompatibilityEvent("compatibility_shims", "none"),
    )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-saver-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.bin"
        output_path = work_dir / "output.bin"
        receipt_path = work_dir / "receipt.tsv"
        library_path = work_dir / "r-library"
        write_raw_matrix(input_path, method_input.counts)
        command = (
            str(executable),
            "--vanilla",
            "-e",
            _SAVER_DRIVER,
            str(source_dir.resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(library_path),
            str(method_input.shape[0]),
            str(method_input.shape[1]),
            str(config.ncores),
            "TRUE" if config.do_fast else "FALSE",
            str(seed),
        )
        result = execute_pinned_command(
            spec,
            source_dir,
            command,
            cwd=work_dir,
            timeout_seconds=spec.resources.timeout_seconds,
        )
        output = read_raw_output(output_path, method_input.shape)
        receipt = read_environment_receipt(
            receipt_path,
            expected_keys=frozenset(
                {
                    "r_version",
                    "saver_source_dir",
                    "saver_version",
                    "glmnet_version",
                    "matrix_version",
                }
            ),
        )
        snapshot = finalize_saver_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=compatibility,
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


__all__ = [
    "SAVERConfig",
    "finalize_saver_output",
    "run_saver",
    "saver_to_evaluator_counts",
]
