"""Shared execution boundary and the observed-count identity control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from ..runtime_environments import (
    publication_git_executable,
    publication_runtime_working_directory,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_SAFE_RECEIPT_KEY = re.compile(r"[a-z][a-z0-9_]*\Z")
_SAME_INPUT_SEED_CONTRACTS = {
    "observed": (False, "not_applicable"),
    "capacity-matched-ae": (True, "required"),
    "maskimpute": (True, "required"),
    "alra": (True, "required"),
    "magic": (True, "required"),
    "dca": (True, "required"),
    "scvi": (True, "required"),
    "saver": (True, "required"),
    "scziva": (True, "required"),
    "afmf": (True, "required"),
    "biaeimpute": (True, "required"),
    "sccr": (True, "required"),
    "scsdae": (True, "required"),
}


@dataclass(frozen=True, slots=True)
class CompatibilityEvent:
    """One truthful, deterministic adapter behavior disclosure."""

    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class SourceReceipt:
    """Read-only proof that an adapter invoked the declared pristine pin."""

    revision: str
    tree: str
    url: str


@dataclass(frozen=True, slots=True)
class AdapterExecution:
    """Immutable output plus exact command, logs, and compatibility disclosures."""

    snapshot: MethodOutputSnapshot
    compatibility_log: tuple[CompatibilityEvent, ...]
    environment_receipt: tuple[tuple[str, str], ...]
    stdout: bytes
    stderr: bytes
    command: tuple[str, ...] | None

    @property
    def stdout_sha256(self) -> str:
        return hashlib.sha256(self.stdout).hexdigest()

    @property
    def stderr_sha256(self) -> str:
        return hashlib.sha256(self.stderr).hexdigest()

    @property
    def realized_p_pre_zero(self) -> np.ndarray | None:
        """Return no score for adapters that do not emit MaskImpute evidence."""

        return None

    @property
    def realized_p_pre_zero_policy(self) -> object | None:
        """Return no score policy for adapters without realized score evidence."""

        return None


class AdapterUnavailableError(RuntimeError):
    """A reproducible failed or unavailable upstream execution attempt."""

    def __init__(
        self,
        reason_code: str,
        detail: str,
        *,
        command: tuple[str, ...] | None = None,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail
        self.command = command
        self.stdout = bytes(stdout)
        self.stderr = bytes(stderr)

    @property
    def stdout_sha256(self) -> str:
        return hashlib.sha256(self.stdout).hexdigest()

    @property
    def stderr_sha256(self) -> str:
        return hashlib.sha256(self.stderr).hexdigest()


def require_method_spec(
    spec: MethodSpec,
    expected_id: str,
    *,
    input_scale: str,
    output_scale: str,
) -> None:
    """Reject registry drift before an adapter sees any method input."""

    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if spec.id != expected_id:
        raise ValueError(f"expected method {expected_id}, received {spec.id}")
    if spec.track != "same_input":
        raise ValueError(f"method {expected_id} must use the same_input track")
    if spec.input_scale != input_scale:
        raise ValueError(
            f"method {expected_id} input scale must be {input_scale}, got {spec.input_scale}"
        )
    if spec.output_scale != output_scale:
        raise ValueError(
            f"method {expected_id} output scale must be {output_scale}, got {spec.output_scale}"
        )
    expected_seed_contract = _SAME_INPUT_SEED_CONTRACTS.get(expected_id)
    if expected_seed_contract is None:
        raise ValueError(f"method {expected_id} has no same-input seed contract")
    expected_stochastic, expected_seed_policy = expected_seed_contract
    if (
        type(spec.stochastic) is not bool
        or spec.stochastic is not expected_stochastic
        or spec.seed_policy != expected_seed_policy
    ):
        raise ValueError(
            f"method {expected_id} stochastic/seed contract must remain "
            f"{expected_stochastic}/{expected_seed_policy}"
        )


def log1p_cp10k(counts: np.ndarray) -> np.ndarray:
    """Library-normalize exact count rows to log(1 + counts per 10,000)."""

    if type(counts) is not np.ndarray or counts.ndim != 2:
        raise TypeError("counts must be an exact two-dimensional ndarray")
    if counts.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("counts must be numeric")
    try:
        with np.errstate(over="raise", invalid="raise"):
            numeric = np.array(
                counts,
                dtype=np.float64,
                copy=True,
                order="C",
                subok=False,
            )
    except FloatingPointError as error:
        raise ValueError("counts must be representable as float64") from error
    if not np.isfinite(numeric).all() or bool((numeric < 0).any()):
        raise ValueError("counts must be finite and nonnegative")
    if bool(np.all(numeric == 0.0, axis=1).any()):
        raise AdapterUnavailableError(
            "zero_library_cell",
            "log1p CP10k is undefined for a zero-library cell",
        )
    return _cp10k_transform(numeric, log_base=1)


def _cp10k_transform(values: np.ndarray, *, log_base: int) -> np.ndarray:
    """Normalize rows exactly when safe, otherwise through scaled proportions."""

    converted = np.empty_like(values)
    for index, row in enumerate(values):
        try:
            with np.errstate(all="raise"):
                library_size = np.sum(row)
                proportions = row / library_size
                if log_base == 1:
                    converted_row = np.log1p(proportions * 10_000.0)
                elif log_base == 2:
                    converted_row = np.log2(1.0 + proportions * 10_000.0)
                else:  # pragma: no cover - private programming error
                    raise AssertionError(log_base)
        except FloatingPointError:
            scale = float(np.max(row))
            if scale == 0.0:  # guarded by each public caller
                raise AssertionError("zero row reached CP10k scaling")
            # Underflow here means a row member is below float64 precision
            # relative to its maximum; retaining the rounded subnormal or zero
            # is the correctly representable proportion.
            with np.errstate(under="ignore"):
                scaled = row / scale
                proportions = scaled / np.sum(scaled)
                if log_base == 1:
                    converted_row = np.log1p(proportions * 10_000.0)
                else:
                    converted_row = np.log2(1.0 + proportions * 10_000.0)
        if not np.isfinite(converted_row).all():
            raise ValueError("CP10k transformation did not remain finite")
        converted[index] = converted_row
    return converted


def _validated_native_matrix(
    method_input: MethodInput,
    native_output: object,
    *,
    name: str,
) -> np.ndarray:
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if type(native_output) is not np.ndarray:
        raise TypeError(f"{name} must be an exact ndarray")
    if native_output.shape != method_input.shape:
        raise ValueError(f"{name} must match the method-input shape")
    if native_output.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must be numeric")
    values = np.array(
        native_output,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")
    return values


def observed_library_sizes(method_input: MethodInput) -> np.ndarray:
    """Return observed row libraries or fail closed on an undefined zero row.

    All evaluator conversions use the observed method input as their normalization
    anchor. A zero-library cell is excluded by policy because neither inversion of
    a CP10k input nor the prespecified log2(CP10k+1) endpoint is defined for it.
    """

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    counts = method_input.counts
    if bool(np.all(counts == 0.0, axis=1).any()):
        raise AdapterUnavailableError(
            "zero_library_cell",
            "evaluator scale conversion is undefined for a zero-library cell",
        )
    try:
        with np.errstate(all="raise"):
            libraries = np.sum(counts, axis=1, dtype=np.float64)
    except FloatingPointError as error:
        raise AdapterUnavailableError(
            "unrepresentable_library_size",
            "an observed library size is not representable as float64",
        ) from error
    if not np.isfinite(libraries).all():
        raise AdapterUnavailableError(
            "unrepresentable_library_size",
            "an observed library size is not representable as float64",
        )
    libraries.setflags(write=False)
    return libraries


def raw_output_to_count_equivalent(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Copy one raw-count output onto the evaluator count-equivalent scale."""

    observed_library_sizes(method_input)
    values = _validated_native_matrix(
        method_input,
        native_output,
        name="native raw-count output",
    )
    values.setflags(write=False)
    return values


def log1p_cp10k_to_count_equivalent(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Invert native log1p-CP10k using each cell's observed library size.

    The conversion is ``expm1(native) * observed_library / 10_000``. It does
    not renormalize the method output and therefore preserves method-induced
    changes in completed total expression on an observed-library count basis.
    """

    libraries = observed_library_sizes(method_input)
    values = _validated_native_matrix(
        method_input,
        native_output,
        name="native log1p-CP10k output",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.expm1(values) * libraries[:, None] / 10_000.0
    if not np.isfinite(converted).all() or bool((converted < 0).any()):
        raise ValueError(
            "native log1p-CP10k output does not have a finite count equivalent"
        )
    converted.setflags(write=False)
    return converted


def count_equivalent_to_log2_cp10k(counts: object) -> np.ndarray:
    """Apply the single declared evaluator log2(CP10k+1) transformation.

    Count-equivalent rows are normalized by their own completed row totals.
    A zero-total row fails with ``zero_library_cell`` instead of being imputed,
    clipped, or assigned an arbitrary pseudolibrary.
    """

    if type(counts) is not np.ndarray or counts.ndim != 2:
        raise TypeError("count equivalents must be an exact two-dimensional ndarray")
    if counts.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("count equivalents must be numeric")
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = np.array(
                counts,
                dtype=np.float64,
                copy=True,
                order="C",
                subok=False,
            )
    except FloatingPointError as error:
        raise ValueError(
            "count equivalents must be representable as float64"
        ) from error
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("count equivalents must be finite and nonnegative")
    if bool(np.all(values == 0.0, axis=1).any()):
        raise AdapterUnavailableError(
            "zero_library_cell",
            "log2 CP10k is undefined for a zero-library count-equivalent row",
        )
    converted = _cp10k_transform(values, log_base=2)
    converted.setflags(write=False)
    return converted


def observed_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Declare observed raw counts as evaluator count equivalents."""

    return raw_output_to_count_equivalent(method_input, native_output)


def _git(source_dir: Path, *arguments: str) -> str:
    selected_source = source_dir.absolute()
    try:
        result = subprocess.run(
            [
                str(publication_git_executable()),
                "-C",
                str(selected_source),
                *arguments,
            ],
            check=True,
            capture_output=True,
            cwd=publication_runtime_working_directory(),
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise AdapterUnavailableError(
            "source_verification_failed",
            f"could not inspect pinned source at {source_dir}",
        ) from error
    return result.stdout.strip()


def verify_pinned_source(spec: MethodSpec, source_dir: Path) -> SourceReceipt:
    """Verify HEAD, tree, remote, and pristine state without changing checkout state."""

    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if not isinstance(source_dir, Path):
        raise TypeError("source_dir must be a pathlib.Path")
    source = spec.source
    if source.kind != "git":
        raise ValueError(f"method {spec.id} does not declare a git source")
    if not source_dir.is_dir():
        raise AdapterUnavailableError(
            "source_checkout_missing",
            f"pinned source directory does not exist: {source_dir}",
        )
    receipt = SourceReceipt(
        revision=_git(source_dir, "rev-parse", "HEAD"),
        tree=_git(source_dir, "rev-parse", "HEAD^{tree}"),
        url=_git(source_dir, "remote", "get-url", "origin"),
    )
    if receipt.revision != source.revision:
        raise AdapterUnavailableError(
            "source_revision_mismatch", f"method {spec.id} source revision differs"
        )
    if receipt.tree != source.tree:
        raise AdapterUnavailableError(
            "source_tree_mismatch", f"method {spec.id} source tree differs"
        )
    if receipt.url != source.url:
        raise AdapterUnavailableError(
            "source_url_mismatch", f"method {spec.id} source remote differs"
        )
    status_output = _git(
        source_dir, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if status_output:
        raise AdapterUnavailableError(
            "source_checkout_not_pristine",
            f"method {spec.id} source checkout contains changes",
        )
    return receipt


def require_executable(executable: Path) -> Path:
    """Validate an absolute launcher while preserving virtual-env semantics."""

    if not isinstance(executable, Path):
        raise TypeError("environment executable must be a pathlib.Path")
    if not executable.is_absolute() or ".." in executable.parts:
        raise AdapterUnavailableError(
            "environment_executable_unsafe",
            "environment executable must be an absolute path without parent traversal",
        )
    selected = executable.absolute()
    try:
        selected_before = selected.lstat()
    except OSError as error:
        raise AdapterUnavailableError(
            "environment_executable_missing",
            f"environment executable is missing or not executable: {selected}",
        ) from error
    if not (
        stat.S_ISREG(selected_before.st_mode) or stat.S_ISLNK(selected_before.st_mode)
    ):
        raise AdapterUnavailableError(
            "environment_executable_unsafe",
            f"environment executable must be a regular file or symlink: {selected}",
        )
    try:
        target = selected.stat()
        selected_after = selected.lstat()
    except OSError as error:
        raise AdapterUnavailableError(
            "environment_executable_missing",
            f"environment executable has a missing or inaccessible target: {selected}",
        ) from error
    selected_identity = (
        selected_before.st_dev,
        selected_before.st_ino,
        selected_before.st_mode,
    )
    if selected_identity != (
        selected_after.st_dev,
        selected_after.st_ino,
        selected_after.st_mode,
    ):
        raise AdapterUnavailableError(
            "environment_executable_unsafe",
            f"environment executable identity changed during validation: {selected}",
        )
    if not stat.S_ISREG(target.st_mode):
        raise AdapterUnavailableError(
            "environment_executable_unsafe",
            f"environment executable target must be a regular file: {selected}",
        )
    if not os.access(selected, os.X_OK):
        raise AdapterUnavailableError(
            "environment_executable_missing",
            f"environment executable is not executable: {selected}",
        )
    return selected


def _subprocess_environment(extra: Mapping[str, str] | None) -> dict[str, str]:
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "CUDA_VISIBLE_DEVICES",
            "HOME",
            "LD_LIBRARY_PATH",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
            "PATH",
            "R_LIBS",
            "R_LIBS_SITE",
            "R_LIBS_USER",
            "TMPDIR",
        }
    }
    allowed.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    if extra is not None:
        for key, value in extra.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "subprocess environment keys and values must be strings"
                )
            allowed[key] = value
    return allowed


def execute_pinned_command(
    spec: MethodSpec,
    source_dir: Path,
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a command from the bound loader CWD between source verifications."""

    if not command or any(not isinstance(value, str) or not value for value in command):
        raise TypeError("command must contain nonempty strings")
    if not isinstance(cwd, Path) or not cwd.is_dir():
        raise TypeError("cwd must be an existing pathlib.Path directory")
    command_tuple = tuple(command)
    before = verify_pinned_source(spec, source_dir)

    def reverify(stdout: bytes, stderr: bytes) -> None:
        try:
            after = verify_pinned_source(spec, source_dir)
        except AdapterUnavailableError as error:
            mutation_reasons = {
                "source_checkout_not_pristine",
                "source_revision_mismatch",
                "source_tree_mismatch",
                "source_url_mismatch",
            }
            reason = (
                "source_mutated_during_execution"
                if error.reason_code in mutation_reasons
                else "source_postverification_failed"
            )
            raise AdapterUnavailableError(
                reason,
                f"method {spec.id} source post-verification failed: {error.reason_code}",
                command=command_tuple,
                stdout=stdout,
                stderr=stderr,
            ) from error
        if after != before:
            raise AdapterUnavailableError(
                "source_identity_changed",
                f"method {spec.id} source identity changed during execution",
                command=command_tuple,
                stdout=stdout,
                stderr=stderr,
            )

    try:
        result = subprocess.run(
            command_tuple,
            cwd=publication_runtime_working_directory(),
            env=_subprocess_environment(environment),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        reverify(stdout, stderr)
        raise AdapterUnavailableError(
            "upstream_timeout",
            f"method {spec.id} exceeded {timeout_seconds} seconds",
            command=command_tuple,
            stdout=stdout,
            stderr=stderr,
        ) from error
    except OSError as error:
        reverify(b"", b"")
        raise AdapterUnavailableError(
            "environment_execution_failed",
            f"could not execute method {spec.id}: {error}",
            command=command_tuple,
        ) from error
    reverify(result.stdout, result.stderr)
    if result.returncode != 0:
        combined = (result.stdout + b"\n" + result.stderr).lower()
        reason = (
            "upstream_dependency_missing"
            if any(
                token in combined
                for token in (
                    b"no module named",
                    b"package or namespace load failed",
                    b"there is no package called",
                )
            )
            else "upstream_nonzero_exit"
        )
        raise AdapterUnavailableError(
            reason,
            f"method {spec.id} exited with status {result.returncode}",
            command=command_tuple,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return result


def _validate_regular_output(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise AdapterUnavailableError(
            "upstream_output_missing", f"upstream did not create output: {path.name}"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or path.is_symlink()
    ):
        raise AdapterUnavailableError(
            "unsafe_upstream_output",
            f"upstream output must be one regular non-symlink file: {path.name}",
        )


def read_npy_output(path: Path) -> np.ndarray:
    """Load one numeric upstream NumPy output after filesystem checks."""

    _validate_regular_output(path)
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, TypeError, ValueError) as error:
        raise AdapterUnavailableError(
            "malformed_upstream_output", f"could not parse {path.name}"
        ) from error
    if type(value) is not np.ndarray:
        raise AdapterUnavailableError(
            "malformed_upstream_output", f"{path.name} is not an exact ndarray"
        )
    return value


def write_raw_matrix(path: Path, matrix: np.ndarray) -> None:
    """Write a controlled little-endian row-major matrix for a base-R driver."""

    values = np.asarray(matrix, dtype="<f8", order="C")
    path.write_bytes(values.tobytes(order="C"))


def read_raw_output(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load one exact-size little-endian row-major matrix from a base-R driver."""

    _validate_regular_output(path)
    payload = path.read_bytes()
    expected_size = shape[0] * shape[1] * 8
    if len(payload) != expected_size:
        raise AdapterUnavailableError(
            "malformed_upstream_output",
            f"raw output size {len(payload)} does not match {expected_size}",
        )
    return np.frombuffer(payload, dtype="<f8").reshape(shape).copy()


def read_environment_receipt(
    path: Path,
    *,
    expected_keys: frozenset[str],
) -> tuple[tuple[str, str], ...]:
    """Read a deterministic tab-delimited environment/source receipt."""

    _validate_regular_output(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise AdapterUnavailableError(
            "malformed_environment_receipt", "environment receipt is not UTF-8"
        ) from error
    values: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise AdapterUnavailableError(
                "malformed_environment_receipt", "receipt line must have two fields"
            )
        key, value = fields
        if (
            not _SAFE_RECEIPT_KEY.fullmatch(key)
            or key in values
            or not value
            or "\x00" in value
        ):
            raise AdapterUnavailableError(
                "malformed_environment_receipt", "receipt key or value is invalid"
            )
        values[key] = value
    if set(values) != expected_keys:
        raise AdapterUnavailableError(
            "malformed_environment_receipt",
            f"receipt keys differ: {sorted(values)}",
        )
    return tuple(sorted(values.items()))


def immutable_receipt_mapping(
    receipt: tuple[tuple[str, str], ...],
) -> Mapping[str, str]:
    return MappingProxyType(dict(receipt))


def run_observed(spec: MethodSpec, method_input: MethodInput) -> AdapterExecution:
    """Return observed counts exactly as the mandatory identity control."""

    require_method_spec(
        spec,
        "observed",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    snapshot = snapshot_method_output(
        spec,
        method_input,
        method_input.counts,
        source_dataset_sha256=method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )
    return AdapterExecution(
        snapshot=snapshot,
        compatibility_log=(
            CompatibilityEvent(
                "identity_control", "returned the truth-free observed count snapshot"
            ),
            CompatibilityEvent(
                "observed_positive_policy", "all observed values are unchanged"
            ),
            CompatibilityEvent(
                "evaluator_scale_conversion",
                "raw counts are already count equivalents; all methods then use the shared log2(1 + counts/row_total*10000) evaluator transform; zero-library rows fail closed",
            ),
        ),
        environment_receipt=(),
        stdout=b"",
        stderr=b"",
        command=None,
    )


__all__ = [
    "AdapterExecution",
    "AdapterUnavailableError",
    "CompatibilityEvent",
    "SourceReceipt",
    "execute_pinned_command",
    "count_equivalent_to_log2_cp10k",
    "log1p_cp10k_to_count_equivalent",
    "log1p_cp10k",
    "observed_library_sizes",
    "observed_to_evaluator_counts",
    "raw_output_to_count_equivalent",
    "read_environment_receipt",
    "read_npy_output",
    "read_raw_output",
    "require_executable",
    "require_method_spec",
    "run_observed",
    "verify_pinned_source",
    "write_raw_matrix",
]
