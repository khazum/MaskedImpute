"""Pinned scZiva adapter invoking the upstream ZIVAimpute callable."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .direct import (
    DirectAdapterExecution,
    DirectMethodOutput,
    finalize_direct_method_output,
)
from .observed import (
    AdapterExecution,
    CompatibilityEvent,
    execute_pinned_command,
    raw_output_to_count_equivalent,
    read_environment_receipt,
    read_npy_output,
    require_executable,
    require_method_spec,
)


_SCZIVA_DRIVER = r"""
from pathlib import Path
import importlib.util
import sys
import types
import numpy as np

if len(sys.argv) != 17:
    raise RuntimeError("adapter expected sixteen arguments")
source_python = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
seed = int(sys.argv[5])
num_epochs = int(sys.argv[6])
learning_rate = float(sys.argv[7])
hidden_dim = int(sys.argv[8])
latent_dim = int(sys.argv[9])
use_cnn = sys.argv[10] == "true"
tau = float(sys.argv[11])
weight_min = float(sys.argv[12])
weight_max = float(sys.argv[13])
regularization = float(sys.argv[14])
reorder = sys.argv[15] == "true"
device_name = None if sys.argv[16] == "none" else sys.argv[16]
sys.path.insert(0, str(source_python))
import torch
# preprocessing/__init__.py imports Scanpy/AnnData helpers that ZIVAimpute never
# calls. Load the exact pinned reorder.py and expose only its required symbol.
sys.modules.setdefault("scanpy", types.ModuleType("scanpy"))
reorder_path = source_python / "preprocessing" / "reorder.py"
reorder_spec = importlib.util.spec_from_file_location(
    "_maskimpute_scziva_reorder", reorder_path
)
if reorder_spec is None or reorder_spec.loader is None:
    raise RuntimeError("could not load pinned scZiva reorder module")
reorder_module = importlib.util.module_from_spec(reorder_spec)
reorder_spec.loader.exec_module(reorder_module)
preprocessing_module = types.ModuleType("preprocessing")
preprocessing_module.reorder_gene_cov = reorder_module.reorder_gene_cov
sys.modules["preprocessing"] = preprocessing_module
import ZIVA
from ZIVA import ZIVAimpute

module_path = Path(ZIVA.__file__).resolve(strict=True)
if source_python not in module_path.parents:
    raise RuntimeError("imported scZiva module is not from the pinned checkout")
counts = np.load(input_path, allow_pickle=False)
if type(counts) is not np.ndarray or counts.ndim != 2:
    raise RuntimeError("input count matrix is malformed")
device = None if device_name is None else torch.device(device_name)
output, _ = ZIVAimpute(
    counts,
    seed=seed,
    device=device,
    num_epochs=num_epochs,
    lr=learning_rate,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    verbose=False,
    use_cnn=use_cnn,
    tau=tau,
    w_min=weight_min,
    w_max=weight_max,
    lam_reg=regularization,
    reorder=reorder,
)
output = np.asarray(output, dtype=np.float64)
if output.shape != counts.shape or not np.isfinite(output).all() or (output < 0).any():
    raise RuntimeError("scZiva output is invalid")
np.save(output_path, output, allow_pickle=False)
actual_device = str(device) if device is not None else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
receipt = {
    "device": actual_device,
    "numpy_version": str(np.__version__),
    "python_version": sys.version.split()[0],
    "reorder_module": str(reorder_path.resolve(strict=True)),
    "torch_version": str(torch.__version__),
    "ziva_module": str(module_path),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


@dataclass(frozen=True, slots=True)
class SCZivaConfig:
    """Exact pinned ZIVAimpute defaults, excluding the study seed."""

    num_epochs: int = 200
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    latent_dim: int = 64
    use_cnn: bool = True
    tau: float = 0.001
    auxiliary_weight_min: float = 0.5
    auxiliary_weight_max: float = 1.5
    auxiliary_regularization: float = 1e-3
    reorder_genes: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("num_epochs", self.num_epochs),
            ("hidden_dim", self.hidden_dim),
            ("latent_dim", self.latent_dim),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("tau", self.tau),
            ("auxiliary_weight_min", self.auxiliary_weight_min),
            ("auxiliary_weight_max", self.auxiliary_weight_max),
            ("auxiliary_regularization", self.auxiliary_regularization),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.use_cnn is not True:
            raise ValueError("use_cnn must remain True for the pinned default")
        if self.reorder_genes is not True:
            raise ValueError("reorder_genes must remain True for the pinned default")
        if self.auxiliary_weight_max <= self.auxiliary_weight_min:
            raise ValueError("auxiliary_weight_max must exceed auxiliary_weight_min")
        if self.device not in {None, "cpu", "cuda"}:
            raise ValueError("device must be null, cpu, or cuda")


def scziva_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Declare scZiva's full raw-scale output as count equivalent."""

    return raw_output_to_count_equivalent(method_input, native_output)


def finalize_scziva_output(
    spec: MethodSpec,
    method_input: MethodInput,
    raw_output: object,
) -> MethodOutputSnapshot:
    """Bind the untouched upstream raw-scale matrix to evaluator IDs."""

    require_method_spec(
        spec,
        "scziva",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    return snapshot_method_output(
        spec,
        method_input,
        raw_output,
        source_dataset_sha256=method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def finalize_scziva_direct_output(
    spec: MethodSpec,
    method_input: MethodInput,
    raw_output: object,
) -> DirectMethodOutput:
    """Validate scZiva output without deriving a content identity."""

    require_method_spec(
        spec,
        "scziva",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    return finalize_direct_method_output(
        spec,
        method_input,
        raw_output,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def _run_scziva_impl(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCZivaConfig = SCZivaConfig(),
    work_root: Path | None = None,
    _direct: bool = False,
) -> AdapterExecution | DirectAdapterExecution:
    """Run pristine pinned scZiva in an explicit Python/Torch environment."""

    require_method_spec(
        spec,
        "scziva",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if not isinstance(config, SCZivaConfig):
        raise TypeError("config must be an SCZivaConfig")
    executable = require_executable(python_executable)
    compatibility = [
        CompatibilityEvent(
            "input_scale_conversion", "none; scZiva receives raw counts"
        ),
        CompatibilityEvent(
            "upstream_defaults",
            f"epochs={config.num_epochs}, lr={config.learning_rate}, hidden={config.hidden_dim}, "
            f"latent={config.latent_dim}, CNN=TRUE, tau={config.tau}, reorder=TRUE",
        ),
        CompatibilityEvent("seed_binding", "study seed is passed to ZIVAimpute"),
        CompatibilityEvent(
            "upstream_selective_policy",
            "pinned ZIVAimpute itself retains observed nonzeros and replaces only zeros whose learned dropout probability exceeds tau",
        ),
        CompatibilityEvent(
            "unused_preprocessing_dependency_shim",
            "pinned preprocessing package eagerly imports unused Scanpy/AnnData helpers; adapter loads exact pinned reorder.py and exposes only reorder_gene_cov without changing its code",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native raw output is a count equivalent and then uses the shared log2(CP10k+1) transform; zero-library rows fail closed",
        ),
        CompatibilityEvent("compatibility_shims", "none"),
    ]
    if config.num_epochs != 200:
        compatibility.append(
            CompatibilityEvent(
                "upstream_training_override", f"num_epochs={config.num_epochs}"
            )
        )
    if config.device is not None:
        compatibility.append(
            CompatibilityEvent(
                "execution_device_override", f"explicit device={config.device}"
            )
        )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-scziva-", dir=work_root) as temporary:
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
            _SCZIVA_DRIVER,
            str((source_dir / "src").resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(seed),
            str(config.num_epochs),
            repr(float(config.learning_rate)),
            str(config.hidden_dim),
            str(config.latent_dim),
            str(config.use_cnn).lower(),
            repr(float(config.tau)),
            repr(float(config.auxiliary_weight_min)),
            repr(float(config.auxiliary_weight_max)),
            repr(float(config.auxiliary_regularization)),
            str(config.reorder_genes).lower(),
            "none" if config.device is None else config.device,
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
                    "device",
                    "numpy_version",
                    "python_version",
                    "reorder_module",
                    "torch_version",
                    "ziva_module",
                }
            ),
        )
        if _direct:
            return DirectAdapterExecution(
                output=finalize_scziva_direct_output(spec, method_input, output),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        snapshot = finalize_scziva_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


def run_scziva(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCZivaConfig = SCZivaConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    return _run_scziva_impl(
        spec,
        method_input,
        source_dir=source_dir,
        python_executable=python_executable,
        seed=seed,
        config=config,
        work_root=work_root,
    )


def run_scziva_direct(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCZivaConfig = SCZivaConfig(),
    work_root: Path | None = None,
) -> DirectAdapterExecution:
    return _run_scziva_impl(
        spec,
        method_input,
        source_dir=source_dir,
        python_executable=python_executable,
        seed=seed,
        config=config,
        work_root=work_root,
        _direct=True,
    )


__all__ = [
    "SCZivaConfig",
    "finalize_scziva_direct_output",
    "finalize_scziva_output",
    "run_scziva",
    "run_scziva_direct",
    "scziva_to_evaluator_counts",
]
