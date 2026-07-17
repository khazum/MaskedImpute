"""Pinned SAVER adapter preserving its native normalized-expression output."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
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
if (length(args) != 14) stop("adapter expected fourteen arguments")
source_dir <- normalizePath(args[[1]], mustWork=TRUE)
input_file <- args[[2]]
output_file <- args[[3]]
receipt_file <- args[[4]]
library_dir <- normalizePath(args[[5]], mustWork=TRUE)
manifest_sha256 <- args[[6]]
qualification_sha256 <- args[[7]]
build_receipt_sha256 <- args[[8]]
installed_library_sha256 <- args[[9]]
n_obs <- as.integer(args[[10]])
n_vars <- as.integer(args[[11]])
ncores <- as.integer(args[[12]])
do_fast <- identical(args[[13]], "TRUE")
seed <- as.integer(args[[14]])
required_packages <- c(
  "Matrix", "Rcpp", "RcppEigen", "SAVER", "codetools", "doParallel",
  "foreach", "glmnet", "iterators", "lattice", "shape", "survival"
)
.libPaths(c(library_dir, .Library))
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                               logical(1), quietly=TRUE)]
if (length(missing_packages)) {
  stop(paste("there is no package called", paste(missing_packages, collapse=",")))
}
package_paths <- vapply(
  required_packages,
  function(package) normalizePath(find.package(package, lib.loc=library_dir),
                                   mustWork=TRUE),
  character(1)
)
if (any(dirname(package_paths) != library_dir)) {
  stop("locked SAVER dependency escaped selected library")
}
suppressPackageStartupMessages(library("SAVER", character.only=TRUE,
                                       lib.loc=library_dir))
locked_version <- function(package) {
  as.character(utils::packageDescription(
    package, fields="Version", lib.loc=library_dir
  ))
}
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
  paste("manifest_sha256", manifest_sha256, sep="\t"),
  paste("qualification_sha256", qualification_sha256, sep="\t"),
  paste("build_receipt_sha256", build_receipt_sha256, sep="\t"),
  paste("installed_library_sha256", installed_library_sha256, sep="\t"),
  paste("r_version", paste(R.version$major, R.version$minor, sep="."), sep="\t"),
  paste("saver_library_dir", library_dir, sep="\t"),
  paste("saver_source_dir", source_dir, sep="\t"),
  paste("saver_version", locked_version("SAVER"), sep="\t"),
  paste("rcpp_version", locked_version("Rcpp"), sep="\t"),
  paste("rcppeigen_version", locked_version("RcppEigen"), sep="\t"),
  paste("codetools_version", locked_version("codetools"), sep="\t"),
  paste("doparallel_version", locked_version("doParallel"), sep="\t"),
  paste("foreach_version", locked_version("foreach"), sep="\t"),
  paste("glmnet_version", locked_version("glmnet"), sep="\t"),
  paste("iterators_version", locked_version("iterators"), sep="\t"),
  paste("lattice_version", locked_version("lattice"), sep="\t"),
  paste("matrix_version", locked_version("Matrix"), sep="\t"),
  paste("shape_version", locked_version("shape"), sep="\t"),
  paste("survival_version", locked_version("survival"), sep="\t")
)
writeLines(receipt, receipt_file, useBytes=TRUE)
"""


_SAVER_PACKAGE_KEYS = {
    "Matrix": "matrix_version",
    "Rcpp": "rcpp_version",
    "RcppEigen": "rcppeigen_version",
    "codetools": "codetools_version",
    "doParallel": "doparallel_version",
    "foreach": "foreach_version",
    "glmnet": "glmnet_version",
    "iterators": "iterators_version",
    "lattice": "lattice_version",
    "shape": "shape_version",
    "survival": "survival_version",
    "SAVER": "saver_version",
}

_SAVER_QUALIFICATION_NAME = "saver-r.qualification.json"
_SAVER_LOCK_RELATIVE = "environments/saver-r.lock.json"
_SAVER_BUILD_RECEIPT_RELATIVE = "environments/saver-r.build-receipt.json"


def _load_saver_qualification(
    spec: MethodSpec,
    lock_manifest: Path,
    *,
    lock_sha256: str,
    installed_library_sha256: str,
    build_receipt_sha256: str,
) -> str:
    qualification = lock_manifest.with_name(_SAVER_QUALIFICATION_NAME)
    try:
        if qualification.is_symlink() or not qualification.is_file():
            raise OSError("qualification receipt is not a regular file")
        payload = qualification.read_bytes()
        data = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterUnavailableError(
            "environment_qualification_missing",
            f"could not read SAVER qualification receipt: {qualification}",
        ) from error
    digest = hashlib.sha256(payload).hexdigest()
    if not isinstance(data, dict) or set(data) != {
        "schema_version",
        "environment_id",
        "package_lock",
        "build_receipt",
        "installed_library_sha256",
        "source",
    }:
        raise AdapterUnavailableError(
            "environment_qualification_malformed",
            "SAVER qualification receipt root is not closed",
        )
    package_binding = data["package_lock"]
    build_binding = data["build_receipt"]
    source_binding = data["source"]
    if (
        not isinstance(package_binding, dict)
        or set(package_binding) != {"path", "sha256"}
        or not isinstance(build_binding, dict)
        or set(build_binding) != {"path", "sha256"}
        or not isinstance(source_binding, dict)
        or set(source_binding) != {"url", "revision", "tree"}
    ):
        raise AdapterUnavailableError(
            "environment_qualification_malformed",
            "SAVER qualification bindings are not closed",
        )
    repository = qualification.absolute().parent.parent
    expected_lock = repository / _SAVER_LOCK_RELATIVE
    expected_build_receipt = repository / _SAVER_BUILD_RECEIPT_RELATIVE
    try:
        if (
            expected_build_receipt.is_symlink()
            or not expected_build_receipt.is_file()
        ):
            raise OSError("build receipt is not a regular file")
        observed_build_sha256 = hashlib.sha256(
            expected_build_receipt.read_bytes()
        ).hexdigest()
    except OSError as error:
        raise AdapterUnavailableError(
            "environment_qualification_missing",
            "SAVER qualification build receipt is unavailable",
        ) from error
    expected_source = {
        "url": spec.source.url,
        "revision": spec.source.revision,
        "tree": spec.source.tree,
    }
    if (
        data["schema_version"] != 1
        or data["environment_id"] != spec.environment.id
        or spec.environment.status != "ready"
        or package_binding.get("path") != _SAVER_LOCK_RELATIVE
        or package_binding.get("sha256") != lock_sha256
        or lock_manifest.absolute() != expected_lock
        or build_binding.get("path") != _SAVER_BUILD_RECEIPT_RELATIVE
        or build_binding.get("sha256") != build_receipt_sha256
        or observed_build_sha256 != build_receipt_sha256
        or data["installed_library_sha256"] != installed_library_sha256
        or source_binding != expected_source
    ):
        raise AdapterUnavailableError(
            "environment_qualification_mismatch",
            "SAVER package authority differs from its tracked qualification receipt",
        )
    return digest


def _load_saver_environment_lock(
    spec: MethodSpec,
    lock_manifest: Path,
) -> tuple[str, str, dict[str, str], str, str, str]:
    if not isinstance(lock_manifest, Path):
        raise TypeError("lock_manifest must be a pathlib.Path")
    try:
        if lock_manifest.is_symlink() or not lock_manifest.is_file():
            raise OSError("lock manifest is not a regular file")
        payload = lock_manifest.read_bytes()
        data = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterUnavailableError(
            "environment_lock_missing",
            f"could not read SAVER environment lock: {lock_manifest}",
        ) from error
    digest = hashlib.sha256(payload).hexdigest()
    if spec.environment.status != "ready":
        raise AdapterUnavailableError(
            "environment_lock_mismatch",
            "SAVER registry environment is not ready",
        )
    if not isinstance(data, dict) or set(data) != {
        "schema_version",
        "environment_id",
        "r_version",
        "packages",
        "upstream_saver",
        "installed_library_sha256",
        "build_receipt_sha256",
    }:
        raise AdapterUnavailableError(
            "environment_lock_malformed", "SAVER lock root is not closed"
        )
    if data["schema_version"] != 1 or data["environment_id"] != spec.environment.id:
        raise AdapterUnavailableError(
            "environment_lock_mismatch", "SAVER lock identity differs from registry"
        )
    r_version = data["r_version"]
    if not isinstance(r_version, str) or not r_version:
        raise AdapterUnavailableError(
            "environment_lock_malformed", "SAVER lock R version is invalid"
        )
    packages = data["packages"]
    upstream = data["upstream_saver"]
    if not isinstance(packages, list) or not isinstance(upstream, dict):
        raise AdapterUnavailableError(
            "environment_lock_malformed", "SAVER lock packages are malformed"
        )
    versions: dict[str, str] = {}
    for item in packages:
        if not isinstance(item, dict) or set(item) != {
            "package",
            "version",
            "url",
            "sha256",
        }:
            raise AdapterUnavailableError(
                "environment_lock_malformed", "SAVER package lock entry is not closed"
            )
        package = item["package"]
        version = item["version"]
        if package in versions or package not in _SAVER_PACKAGE_KEYS:
            raise AdapterUnavailableError(
                "environment_lock_malformed", "SAVER package lock set is invalid"
            )
        if not isinstance(version, str) or not version:
            raise AdapterUnavailableError(
                "environment_lock_malformed", "SAVER package version is invalid"
            )
        versions[package] = version
    if set(upstream) != {"package", "version", "url", "revision", "tree"}:
        raise AdapterUnavailableError(
            "environment_lock_malformed", "upstream SAVER lock is not closed"
        )
    source = spec.source
    if (
        upstream["package"] != "SAVER"
        or upstream["url"] != source.url
        or upstream["revision"] != source.revision
        or upstream["tree"] != source.tree
    ):
        raise AdapterUnavailableError(
            "environment_lock_mismatch", "upstream SAVER pin differs from registry"
        )
    versions["SAVER"] = upstream["version"]
    if set(versions) != set(_SAVER_PACKAGE_KEYS):
        raise AdapterUnavailableError(
            "environment_lock_malformed",
            "SAVER lock does not cover the complete dependency closure",
        )
    installed_library_sha256 = data["installed_library_sha256"]
    build_receipt_sha256 = data["build_receipt_sha256"]
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (installed_library_sha256, build_receipt_sha256)
    ):
        raise AdapterUnavailableError(
            "environment_lock_malformed",
            "SAVER installed-library bindings are invalid",
        )
    qualification_sha256 = _load_saver_qualification(
        spec,
        lock_manifest,
        lock_sha256=digest,
        installed_library_sha256=installed_library_sha256,
        build_receipt_sha256=build_receipt_sha256,
    )
    return (
        digest,
        r_version,
        versions,
        installed_library_sha256,
        build_receipt_sha256,
        qualification_sha256,
    )


def _load_saver_build_receipt(
    spec: MethodSpec,
    build_receipt: Path,
    *,
    expected_sha256: str,
    installed_library_sha256: str,
    r_version: str,
    versions: dict[str, str],
) -> str:
    if not isinstance(build_receipt, Path):
        raise TypeError("build_receipt must be a pathlib.Path")
    try:
        if build_receipt.is_symlink() or not build_receipt.is_file():
            raise OSError("build receipt is not a regular file")
        payload = build_receipt.read_bytes()
        data = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdapterUnavailableError(
            "environment_build_receipt_missing",
            f"could not read SAVER build receipt: {build_receipt}",
        ) from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise AdapterUnavailableError(
            "environment_build_receipt_mismatch",
            "SAVER build receipt differs from the lock manifest binding",
        )
    expected_keys = {
        "schema_version",
        "status",
        "build_date",
        "environment_id",
        "build_script_sha256",
        "build_log_sha256",
        "r_version",
        "library_dir",
        "installed_library_sha256",
        "saver_source",
        "package_versions",
        "smoke_test",
    }
    if not isinstance(data, dict) or set(data) != expected_keys:
        raise AdapterUnavailableError(
            "environment_build_receipt_malformed",
            "SAVER build receipt root is not closed",
        )
    source = spec.source
    expected_source = {"revision": source.revision, "tree": source.tree}
    smoke_test = data["smoke_test"]
    if (
        data["schema_version"] != 1
        or data["status"] != "real_tiny_smoke_passed"
        or data["environment_id"] != spec.environment.id
        or data["r_version"] != r_version
        or data["installed_library_sha256"] != installed_library_sha256
        or data["saver_source"] != expected_source
        or data["package_versions"] != versions
        or not isinstance(data["build_date"], str)
        or not isinstance(data["library_dir"], str)
        or not Path(data["library_dir"]).is_absolute()
        or not isinstance(smoke_test, dict)
        or set(smoke_test) != {"command", "result"}
        or smoke_test["result"] != "1 passed"
        or not isinstance(smoke_test["command"], str)
    ):
        raise AdapterUnavailableError(
            "environment_build_receipt_mismatch",
            "SAVER build receipt differs from the locked qualification",
        )
    for key in ("build_script_sha256", "build_log_sha256"):
        value = data[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise AdapterUnavailableError(
                "environment_build_receipt_malformed",
                f"SAVER build receipt {key} is invalid",
            )
    return digest


def _saver_library_sha256(library_dir: Path) -> str:
    """Hash a closed installed R library as sorted file-content bindings."""

    digest = hashlib.sha256()
    file_count = 0
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.+@/:=-"
    )
    entries = sorted(
        library_dir.rglob("*"), key=lambda path: path.relative_to(library_dir).as_posix()
    )
    for path in entries:
        relative = path.relative_to(library_dir).as_posix()
        if path.is_symlink():
            raise AdapterUnavailableError(
                "environment_library_malformed",
                f"locked SAVER library contains a symlink: {relative}",
            )
        if path.is_dir():
            continue
        if not path.is_file() or not relative or any(
            character not in allowed for character in relative
        ):
            raise AdapterUnavailableError(
                "environment_library_malformed",
                f"locked SAVER library contains an unsupported entry: {relative}",
            )
        file_digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(chunk)
        except OSError as error:
            raise AdapterUnavailableError(
                "environment_library_malformed",
                f"could not hash locked SAVER library entry: {relative}",
            ) from error
        digest.update(f"{file_digest.hexdigest()}  {relative}\n".encode("ascii"))
        file_count += 1
    if file_count == 0:
        raise AdapterUnavailableError(
            "environment_library_incomplete", "locked SAVER library contains no files"
        )
    return digest.hexdigest()


def _validate_saver_library(
    library_dir: Path,
    versions: dict[str, str],
    expected_sha256: str,
) -> Path:
    if not isinstance(library_dir, Path):
        raise TypeError("library_dir must be a pathlib.Path")
    if library_dir.is_symlink() or not library_dir.is_dir():
        raise AdapterUnavailableError(
            "environment_library_missing",
            f"locked SAVER library is missing: {library_dir}",
        )
    selected = library_dir.resolve(strict=True)
    missing = sorted(package for package in versions if not (selected / package).is_dir())
    if missing:
        raise AdapterUnavailableError(
            "environment_library_incomplete",
            f"locked SAVER library is missing packages: {','.join(missing)}",
        )
    observed_sha256 = _saver_library_sha256(selected)
    if observed_sha256 != expected_sha256:
        raise AdapterUnavailableError(
            "environment_library_digest_mismatch",
            "installed SAVER library content differs from the lock manifest",
        )
    return selected


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
    library_dir: Path,
    lock_manifest: Path,
    build_receipt: Path,
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
    (
        manifest_sha256,
        locked_r_version,
        locked_versions,
        installed_library_sha256,
        expected_build_receipt_sha256,
        qualification_sha256,
    ) = _load_saver_environment_lock(spec, lock_manifest)
    build_receipt_sha256 = _load_saver_build_receipt(
        spec,
        build_receipt,
        expected_sha256=expected_build_receipt_sha256,
        installed_library_sha256=installed_library_sha256,
        r_version=locked_r_version,
        versions=locked_versions,
    )
    selected_library = _validate_saver_library(
        library_dir, locked_versions, installed_library_sha256
    )
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
        CompatibilityEvent(
            "environment_lock",
            f"prebuilt isolated R library content sha256={installed_library_sha256}, build receipt sha256={build_receipt_sha256}, and package manifest sha256={manifest_sha256} are independently bound by qualification receipt sha256={qualification_sha256}; no package installation occurs during method execution",
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
            str(selected_library),
            manifest_sha256,
            qualification_sha256,
            build_receipt_sha256,
            installed_library_sha256,
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
            environment={
                "R_LIBS": str(selected_library),
                "R_LIBS_SITE": str(selected_library),
                "R_LIBS_USER": str(selected_library),
            },
        )
        output = read_raw_output(output_path, method_input.shape)
        receipt = read_environment_receipt(
            receipt_path,
            expected_keys=frozenset(
                {
                    "r_version",
                    "manifest_sha256",
                    "qualification_sha256",
                    "build_receipt_sha256",
                    "installed_library_sha256",
                    "saver_library_dir",
                    "saver_source_dir",
                    "saver_version",
                    "rcpp_version",
                    "rcppeigen_version",
                    "codetools_version",
                    "doparallel_version",
                    "foreach_version",
                    "glmnet_version",
                    "iterators_version",
                    "lattice_version",
                    "matrix_version",
                    "shape_version",
                    "survival_version",
                }
            ),
        )
        receipt_values = dict(receipt)
        if (
            receipt_values["manifest_sha256"] != manifest_sha256
            or receipt_values["qualification_sha256"] != qualification_sha256
            or receipt_values["build_receipt_sha256"] != build_receipt_sha256
            or receipt_values["installed_library_sha256"]
            != installed_library_sha256
            or receipt_values["r_version"] != locked_r_version
            or any(
                receipt_values[_SAVER_PACKAGE_KEYS[package]] != version
                for package, version in locked_versions.items()
            )
        ):
            raise AdapterUnavailableError(
                "environment_lock_mismatch",
                "runtime SAVER package versions differ from the tracked lock",
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        if _saver_library_sha256(selected_library) != installed_library_sha256:
            raise AdapterUnavailableError(
                "environment_library_digest_mismatch",
                "installed SAVER library content changed during execution",
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
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
