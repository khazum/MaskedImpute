"""Pinned afMF adapter preserving its native log-normalized output."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    execute_pinned_command,
    log1p_cp10k_to_count_equivalent,
    read_environment_receipt,
    read_npy_output,
    require_executable,
    require_method_spec,
)


_AFMF_DRIVER = r"""
from pathlib import Path
import importlib
import sys
import numpy as np
import pandas as pd
import sklearn

if len(sys.argv) != 11:
    raise RuntimeError("adapter expected ten arguments")
source_python = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
seed = int(sys.argv[5])
iterations = int(sys.argv[6])
tolerance = float(sys.argv[7])
lambda_p = float(sys.argv[8])
lambda_q = float(sys.argv[9])
sigma = float(sys.argv[10])
sys.path.insert(0, str(source_python))
module = importlib.import_module("afMF.runafMF")
module_path = Path(module.__file__).resolve(strict=True)
if source_python not in module_path.parents:
    raise RuntimeError("imported afMF module is not from the pinned checkout")
counts = np.load(input_path, allow_pickle=False)
if type(counts) is not np.ndarray or counts.ndim != 2:
    raise RuntimeError("input count matrix is malformed")
# Upstream afMF uses genes as rows and cells as columns.
frame = pd.DataFrame(counts.T)
native = module.afMF(
    frame,
    iteration=iterations,
    tolerence=tolerance,
    lambda_P=lambda_p,
    lambda_Q=lambda_q,
    sigma=sigma,
    random_seed=seed,
)
output = np.asarray(native, dtype=np.float64).T
if output.shape != counts.shape or not np.isfinite(output).all() or (output < 0).any():
    raise RuntimeError("afMF output is invalid")
np.save(output_path, output, allow_pickle=False)
receipt = {
    "afmf_module": str(module_path),
    "numpy_version": str(np.__version__),
    "pandas_version": str(pd.__version__),
    "python_version": sys.version.split()[0],
    "sklearn_version": str(sklearn.__version__),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


@dataclass(frozen=True, slots=True)
class AFMFConfig:
    """Exact pinned afMF callable defaults, excluding the study seed."""

    iterations: int = 10_000
    tolerance: float = 1e-4
    lambda_p: float = 0.0
    lambda_q: float = 0.0
    sigma: float = 3.0

    def __post_init__(self) -> None:
        if type(self.iterations) is not int or self.iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not math.isfinite(float(self.tolerance))
            or self.tolerance <= 0
        ):
            raise ValueError("tolerance must be finite and positive")
        for name, value in (("lambda_p", self.lambda_p), ("lambda_q", self.lambda_q)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            isinstance(self.sigma, bool)
            or not isinstance(self.sigma, (int, float))
            or not math.isfinite(float(self.sigma))
            or self.sigma <= 0
        ):
            raise ValueError("sigma must be finite and positive")


def afmf_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Invert afMF's native natural-log CP10k on observed libraries."""

    return log1p_cp10k_to_count_equivalent(method_input, native_output)


def finalize_afmf_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> MethodOutputSnapshot:
    """Bind untouched upstream afMF log-normalized output to evaluator IDs."""

    require_method_spec(
        spec,
        "afmf",
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


def run_afmf(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: AFMFConfig = AFMFConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    """Run pristine pinned afMF with explicit orientation and seed binding."""

    require_method_spec(
        spec,
        "afmf",
        input_scale="raw_counts",
        output_scale="method_native_normalized",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if not isinstance(config, AFMFConfig):
        raise TypeError("config must be an AFMFConfig")
    if min(method_input.shape) < 100:
        raise AdapterUnavailableError(
            "upstream_minimum_dimension",
            "pinned afMF requests 100 randomized singular vectors and requires at least 100 cells and genes",
        )
    executable = require_executable(python_executable)
    compatibility = [
        CompatibilityEvent(
            "input_orientation",
            "adapter transposes cells-by-genes to upstream genes-by-cells",
        ),
        CompatibilityEvent(
            "upstream_defaults",
            f"iterations={config.iterations}, tolerence={config.tolerance}, lambda_P={config.lambda_p}, "
            f"lambda_Q={config.lambda_q}, sigma={config.sigma}",
        ),
        CompatibilityEvent(
            "seed_binding", "study seed replaces callable random_seed default 42"
        ),
        CompatibilityEvent(
            "output_orientation",
            "untouched upstream gene-by-cell natural-log CP10k output is transposed back to cells-by-genes",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "evaluator counts are expm1(native)*observed_library_size/10000 before the shared log2(CP10k+1) transform",
        ),
        CompatibilityEvent("compatibility_shims", "none"),
    ]
    if config.iterations != 10_000:
        compatibility.append(
            CompatibilityEvent(
                "upstream_iteration_override", f"iterations={config.iterations}"
            )
        )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-afmf-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.npy"
        output_path = work_dir / "output.npy"
        receipt_path = work_dir / "receipt.tsv"
        np.save(input_path, method_input.counts, allow_pickle=False)
        command = (
            str(executable),
            "-B",
            "-I",
            "-c",
            _AFMF_DRIVER,
            str((source_dir / "afMF").resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(seed),
            str(config.iterations),
            repr(float(config.tolerance)),
            repr(float(config.lambda_p)),
            repr(float(config.lambda_q)),
            repr(float(config.sigma)),
        )
        result = execute_pinned_command(
            spec,
            source_dir,
            command,
            cwd=work_dir,
            timeout_seconds=spec.resources.timeout_seconds,
        )
        output = read_npy_output(output_path)
        receipt = read_environment_receipt(
            receipt_path,
            expected_keys=frozenset(
                {
                    "afmf_module",
                    "numpy_version",
                    "pandas_version",
                    "python_version",
                    "sklearn_version",
                }
            ),
        )
        snapshot = finalize_afmf_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


__all__ = [
    "AFMFConfig",
    "afmf_to_evaluator_counts",
    "finalize_afmf_output",
    "run_afmf",
]
