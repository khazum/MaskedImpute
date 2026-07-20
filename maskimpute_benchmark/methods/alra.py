"""Pinned ALRA adapter using the upstream base-R API without source edits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .direct import DirectAdapterExecution, DirectMethodOutput, finalize_direct_method_output
from .observed import (
    AdapterExecution,
    CompatibilityEvent,
    execute_pinned_command,
    log1p_cp10k_to_count_equivalent,
    read_environment_receipt,
    read_raw_output,
    require_executable,
    require_method_spec,
    write_raw_matrix,
)


_ALRA_DRIVER = r"""
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 11) stop("adapter expected eleven arguments")
source_file <- normalizePath(args[[1]], mustWork=TRUE)
input_file <- args[[2]]
output_file <- args[[3]]
receipt_file <- args[[4]]
n_obs <- as.integer(args[[5]])
n_vars <- as.integer(args[[6]])
seed <- as.integer(args[[7]])
k <- as.integer(args[[8]])
q <- as.integer(args[[9]])
quantile_probability <- as.numeric(args[[10]])
use_mkl <- identical(args[[11]], "TRUE")
mkl_threading_layer <- Sys.getenv("MKL_THREADING_LAYER", unset=NA_character_)
if (!identical(mkl_threading_layer, "GNU")) {
  stop("MKL_THREADING_LAYER must be GNU")
}
input_connection <- file(input_file, open="rb")
on.exit(close(input_connection), add=TRUE)
values <- readBin(input_connection, what="double", n=n_obs*n_vars,
                  size=8, endian="little")
if (length(values) != n_obs*n_vars) stop("input byte count differs")
counts <- matrix(values, nrow=n_obs, ncol=n_vars, byrow=TRUE)
if (any(!is.finite(counts)) || any(counts < 0)) stop("input is invalid")
if (any(rowSums(counts) == 0)) stop("zero-library cell is unsupported")
source(source_file, local=.GlobalEnv)
set.seed(seed)
normalized <- normalize_data(counts)
# Pinned alra.R uses `if (class(A_norm) != "matrix")`, which is length two
# under current R because an unclassed matrix has implicit classes matrix/array.
# An explicit matrix class preserves values/dimensions and makes that check scalar.
attr(normalized, "class") <- "matrix"
result <- alra(normalized, k=k, q=q, quantile.prob=quantile_probability,
               use.mkl=use_mkl, mkl.seed=-1)[[3]]
if (!identical(dim(result), c(n_obs, n_vars))) stop("output shape differs")
if (any(!is.finite(result)) || any(result < 0)) stop("output is invalid")
output_connection <- file(output_file, open="wb")
writeBin(as.double(t(result)), output_connection, size=8, endian="little")
close(output_connection)
receipt <- c(
  paste("mkl_threading_layer", mkl_threading_layer, sep="\t"),
  paste("r_version", R.version.string, sep="\t"),
  paste("rsvd_version", as.character(utils::packageVersion("rsvd")), sep="\t"),
  paste("upstream_source_file", source_file, sep="\t")
)
writeLines(receipt, receipt_file, useBytes=TRUE)
"""


@dataclass(frozen=True, slots=True)
class ALRAConfig:
    """Pinned upstream defaults; nonzero rank is an explicit disclosed override."""

    k: int = 0
    q: int = 10
    quantile_probability: float = 0.001
    use_mkl: bool = False

    def __post_init__(self) -> None:
        if type(self.k) is not int or self.k < 0:
            raise ValueError("k must be a nonnegative integer")
        if type(self.q) is not int or self.q <= 0:
            raise ValueError("q must be a positive integer")
        if (
            isinstance(self.quantile_probability, bool)
            or not isinstance(self.quantile_probability, (int, float))
            or not 0 < self.quantile_probability < 0.5
        ):
            raise ValueError("quantile_probability must be between zero and 0.5")
        if self.use_mkl is not False:
            raise ValueError("use_mkl must remain False for the pinned default adapter")


def alra_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
):
    """Invert ALRA's native log1p-CP10k on observed library sizes."""

    return log1p_cp10k_to_count_equivalent(method_input, native_output)


def finalize_alra_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> MethodOutputSnapshot:
    """Bind upstream ALRA's completed log1p-CP10k matrix to evaluator IDs."""

    require_method_spec(
        spec,
        "alra",
        input_scale="log1p_cp10k",
        output_scale="log1p_cp10k",
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


def finalize_alra_direct_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> DirectMethodOutput:
    """Validate ALRA output without deriving a content identity."""

    require_method_spec(
        spec,
        "alra",
        input_scale="log1p_cp10k",
        output_scale="log1p_cp10k",
    )
    return finalize_direct_method_output(
        spec,
        method_input,
        normalized_output,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def _run_alra_impl(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    rscript: Path,
    seed: int,
    config: ALRAConfig = ALRAConfig(),
    work_root: Path | None = None,
    _direct: bool = False,
) -> AdapterExecution | DirectAdapterExecution:
    """Run exact pinned ALRA source in a selected R environment."""

    require_method_spec(
        spec,
        "alra",
        input_scale="log1p_cp10k",
        output_scale="log1p_cp10k",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2^31)")
    if not isinstance(config, ALRAConfig):
        raise TypeError("config must be an ALRAConfig")
    executable = require_executable(rscript)
    source_file = source_dir / "alra.R"
    compatibility = [
        CompatibilityEvent(
            "input_scale_conversion",
            "upstream normalize_data applies log(1 + counts/library_size*10000)",
        ),
        CompatibilityEvent(
            "upstream_parameters",
            f"q={config.q}, quantile.prob={config.quantile_probability}, use.mkl=FALSE; study seed passed to set.seed",
        ),
        CompatibilityEvent(
            "output_convention",
            "returned upstream A_norm_rank_k_cor_sc without selective postprocessing",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native snapshot remains log1p-CP10k; evaluator counts are expm1(native)*observed_library_size/10000, followed by the shared log2(CP10k+1) transform; zero-library rows fail closed",
        ),
        CompatibilityEvent(
            "numerical_stability_policy",
            "subprocess binds MKL_THREADING_LAYER=GNU before R initializes numerical libraries and receipts the exact value",
        ),
        CompatibilityEvent(
            "compatibility_shims",
            "sets explicit class='matrix' on normalized input so pinned scalar class check works on current R; values and dimensions unchanged",
        ),
    ]
    if config.k == 0:
        compatibility.append(
            CompatibilityEvent("upstream_rank_selection", "k=0 automatic choice")
        )
    else:
        compatibility.append(
            CompatibilityEvent(
                "upstream_rank_override", f"nondefault explicit k={config.k}"
            )
        )
    if (config.q, float(config.quantile_probability)) != (10, 0.001):
        compatibility.append(
            CompatibilityEvent(
                "upstream_parameter_override",
                "q or quantile.prob differs from the pinned constructor default",
            )
        )

    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-alra-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.bin"
        output_path = work_dir / "output.bin"
        receipt_path = work_dir / "receipt.tsv"
        write_raw_matrix(input_path, method_input.counts)
        command = (
            str(executable),
            "--vanilla",
            "-e",
            _ALRA_DRIVER,
            str(source_file.resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(method_input.shape[0]),
            str(method_input.shape[1]),
            str(seed),
            str(config.k),
            str(config.q),
            repr(float(config.quantile_probability)),
            "TRUE" if config.use_mkl else "FALSE",
        )
        result = execute_pinned_command(
            spec,
            source_dir,
            command,
            cwd=work_dir,
            timeout_seconds=spec.resources.timeout_seconds,
            environment={"MKL_THREADING_LAYER": "GNU"},
        )
        output = read_raw_output(output_path, method_input.shape)
        receipt = read_environment_receipt(
            receipt_path,
            expected_keys=frozenset(
                {
                    "mkl_threading_layer",
                    "r_version",
                    "rsvd_version",
                    "upstream_source_file",
                }
            ),
        )
        if _direct:
            return DirectAdapterExecution(
                output=finalize_alra_direct_output(spec, method_input, output),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        snapshot = finalize_alra_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


def run_alra(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    rscript: Path, seed: int, config: ALRAConfig = ALRAConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    return _run_alra_impl(spec, method_input, source_dir=source_dir, rscript=rscript,
                          seed=seed, config=config, work_root=work_root)


def run_alra_direct(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    rscript: Path, seed: int, config: ALRAConfig = ALRAConfig(),
    work_root: Path | None = None,
) -> DirectAdapterExecution:
    return _run_alra_impl(spec, method_input, source_dir=source_dir, rscript=rscript,
                          seed=seed, config=config, work_root=work_root, _direct=True)


__all__ = [
    "ALRAConfig",
    "alra_to_evaluator_counts",
    "finalize_alra_direct_output",
    "finalize_alra_output",
    "run_alra",
    "run_alra_direct",
]
