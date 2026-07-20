"""Pinned MAGIC adapter with matched log1p-CP10k preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .direct import DirectAdapterExecution, DirectMethodOutput, finalize_direct_method_output
from .observed import (
    AdapterExecution,
    CompatibilityEvent,
    execute_pinned_command,
    log1p_cp10k,
    log1p_cp10k_to_count_equivalent,
    read_environment_receipt,
    read_npy_output,
    require_executable,
    require_method_spec,
)


_MAGIC_DRIVER = r"""
from pathlib import Path
import sys
import numpy as np

if len(sys.argv) != 14:
    raise RuntimeError("adapter expected thirteen arguments")
source_python = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
seed = int(sys.argv[5])
knn = int(sys.argv[6])
knn_max = None if sys.argv[7] == "none" else int(sys.argv[7])
decay = int(sys.argv[8])
diffusion_time = int(sys.argv[9])
n_pca = int(sys.argv[10])
solver = sys.argv[11]
distance = sys.argv[12]
n_jobs = int(sys.argv[13])
sys.path.insert(0, str(source_python))
import magic
import graphtools
import scprep

module_path = Path(magic.__file__).resolve(strict=True)
if source_python not in module_path.parents:
    raise RuntimeError("imported MAGIC module is not from the pinned checkout")
matrix = np.load(input_path, allow_pickle=False)
if type(matrix) is not np.ndarray or matrix.ndim != 2:
    raise RuntimeError("input matrix is malformed")
operator = magic.MAGIC(
    knn=knn,
    knn_max=knn_max,
    decay=decay,
    t=diffusion_time,
    n_pca=n_pca,
    solver=solver,
    knn_dist=distance,
    n_jobs=n_jobs,
    random_state=seed,
    verbose=False,
)
output = np.asarray(operator.fit_transform(matrix, genes="all_genes"), dtype=np.float64)
if output.shape != matrix.shape or not np.isfinite(output).all() or (output < 0).any():
    raise RuntimeError("MAGIC output is invalid")
np.save(output_path, output, allow_pickle=False)
receipt = {
    "graphtools_version": str(getattr(graphtools, "__version__", "unknown")),
    "magic_module": str(module_path),
    "magic_version": str(getattr(magic, "__version__", "unknown")),
    "numpy_version": str(np.__version__),
    "python_version": sys.version.split()[0],
    "scprep_version": str(getattr(scprep, "__version__", "unknown")),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


@dataclass(frozen=True, slots=True)
class MAGICConfig:
    """Exact pinned MAGIC constructor defaults used by the core comparator."""

    knn: int = 5
    knn_max: int | None = None
    decay: int = 1
    diffusion_time: int = 3
    n_pca: int = 100
    solver: str = "exact"
    distance: str = "euclidean"
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if type(self.knn) is not int or self.knn <= 0:
            raise ValueError("knn must be a positive integer")
        if self.knn_max is not None and (
            type(self.knn_max) is not int or self.knn_max < self.knn
        ):
            raise ValueError("knn_max must be null or an integer at least knn")
        for name, value in (
            ("decay", self.decay),
            ("diffusion_time", self.diffusion_time),
            ("n_pca", self.n_pca),
            ("n_jobs", self.n_jobs),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.solver != "exact":
            raise ValueError("solver must remain exact for the pinned core adapter")
        if self.distance != "euclidean":
            raise ValueError(
                "distance must remain euclidean for the pinned core adapter"
            )


def magic_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Invert MAGIC's native log1p-CP10k on observed library sizes."""

    return log1p_cp10k_to_count_equivalent(method_input, native_output)


def finalize_magic_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> MethodOutputSnapshot:
    """Bind upstream MAGIC's diffused log1p-CP10k matrix without copying entries."""

    require_method_spec(
        spec,
        "magic",
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


def finalize_magic_direct_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> DirectMethodOutput:
    """Validate MAGIC output without deriving a content identity."""

    require_method_spec(
        spec,
        "magic",
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


def _run_magic_impl(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: MAGICConfig = MAGICConfig(),
    work_root: Path | None = None,
    _direct: bool = False,
) -> AdapterExecution | DirectAdapterExecution:
    """Run pinned MAGIC code with a separately selected dependency environment."""

    require_method_spec(
        spec,
        "magic",
        input_scale="log1p_cp10k",
        output_scale="log1p_cp10k",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if not isinstance(config, MAGICConfig):
        raise TypeError("config must be a MAGICConfig")
    executable = require_executable(python_executable)
    normalized = log1p_cp10k(method_input.counts)
    compatibility = [
        CompatibilityEvent(
            "input_scale_conversion",
            "adapter applies log(1 + counts/library_size*10000) before MAGIC",
        ),
        CompatibilityEvent(
            "upstream_parameters",
            f"knn={config.knn}, knn_max={config.knn_max}, decay={config.decay}, "
            f"t={config.diffusion_time}, n_pca={config.n_pca}, solver={config.solver}, "
            f"distance={config.distance}, n_jobs={config.n_jobs}",
        ),
        CompatibilityEvent(
            "seed_binding", "study model seed is passed as MAGIC random_state"
        ),
        CompatibilityEvent(
            "output_convention",
            "returned full upstream diffused matrix on input log1p-CP10k scale",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native snapshot remains log1p-CP10k; evaluator counts are expm1(native)*observed_library_size/10000, followed by the shared log2(CP10k+1) transform; zero-library rows fail closed",
        ),
        CompatibilityEvent("compatibility_shims", "none"),
    ]
    if config != MAGICConfig():
        compatibility.append(
            CompatibilityEvent(
                "upstream_parameter_override",
                "one or more MAGIC constructor values differs from the pinned default",
            )
        )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-magic-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.npy"
        output_path = work_dir / "output.npy"
        receipt_path = work_dir / "receipt.tsv"
        np.save(input_path, normalized, allow_pickle=False)
        command = (
            str(executable),
            "-I",
            "-c",
            _MAGIC_DRIVER,
            str((source_dir / "python").resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(seed),
            str(config.knn),
            "none" if config.knn_max is None else str(config.knn_max),
            str(config.decay),
            str(config.diffusion_time),
            str(config.n_pca),
            config.solver,
            config.distance,
            str(config.n_jobs),
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
                    "graphtools_version",
                    "magic_module",
                    "magic_version",
                    "numpy_version",
                    "python_version",
                    "scprep_version",
                }
            ),
        )
        if _direct:
            return DirectAdapterExecution(
                output=finalize_magic_direct_output(spec, method_input, output),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        snapshot = finalize_magic_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


def run_magic(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    python_executable: Path, seed: int, config: MAGICConfig = MAGICConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    return _run_magic_impl(spec, method_input, source_dir=source_dir,
                           python_executable=python_executable, seed=seed,
                           config=config, work_root=work_root)


def run_magic_direct(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    python_executable: Path, seed: int, config: MAGICConfig = MAGICConfig(),
    work_root: Path | None = None,
) -> DirectAdapterExecution:
    return _run_magic_impl(spec, method_input, source_dir=source_dir,
                           python_executable=python_executable, seed=seed,
                           config=config, work_root=work_root, _direct=True)


__all__ = [
    "MAGICConfig",
    "finalize_magic_direct_output",
    "finalize_magic_output",
    "magic_to_evaluator_counts",
    "run_magic",
    "run_magic_direct",
]
