"""Pinned DCA adapter using its public AnnData API in an isolated process."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .observed import (
    AdapterExecution,
    CompatibilityEvent,
    execute_pinned_command,
    read_environment_receipt,
    read_npy_output,
    raw_output_to_count_equivalent,
    require_executable,
    require_method_spec,
)


_DCA_DRIVER = r"""
from pathlib import Path
import sys
import numpy as np

if len(sys.argv) != 20:
    raise RuntimeError("adapter expected nineteen arguments")
source_dir = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
seed = int(sys.argv[5])
ae_type = sys.argv[6]
normalize_per_cell = sys.argv[7] == "true"
scale = sys.argv[8] == "true"
log1p = sys.argv[9] == "true"
hidden_size = tuple(int(value) for value in sys.argv[10].split(","))
hidden_dropout = float(sys.argv[11])
batchnorm = sys.argv[12] == "true"
activation = sys.argv[13]
initializer = sys.argv[14]
epochs = int(sys.argv[15])
reduce_lr = int(sys.argv[16])
early_stop = int(sys.argv[17])
batch_size = int(sys.argv[18])
optimizer = sys.argv[19]
sys.path.insert(0, str(source_dir))
import anndata
import dca
import numpy
import scanpy
import tensorflow
from dca.api import dca as dca_run

module_path = Path(dca.__file__).resolve(strict=True)
if source_dir not in module_path.parents:
    raise RuntimeError("imported DCA module is not from the pinned checkout")
counts = np.load(input_path, allow_pickle=False)
if type(counts) is not np.ndarray or counts.ndim != 2:
    raise RuntimeError("input count matrix is malformed")
adata = anndata.AnnData(X=counts)
result = dca_run(
    adata,
    mode="denoise",
    ae_type=ae_type,
    normalize_per_cell=normalize_per_cell,
    scale=scale,
    log1p=log1p,
    hidden_size=hidden_size,
    hidden_dropout=hidden_dropout,
    batchnorm=batchnorm,
    activation=activation,
    init=initializer,
    epochs=epochs,
    reduce_lr=reduce_lr,
    early_stop=early_stop,
    batch_size=batch_size,
    optimizer=optimizer,
    random_state=seed,
    verbose=False,
    return_model=False,
    return_info=False,
    copy=True,
    check_counts=True,
)
output = result.X.toarray() if hasattr(result.X, "toarray") else np.asarray(result.X)
output = np.asarray(output, dtype=np.float64)
if output.shape != counts.shape or not np.isfinite(output).all() or (output < 0).any():
    raise RuntimeError("DCA output is invalid")
np.save(output_path, output, allow_pickle=False)
receipt = {
    "anndata_version": str(getattr(anndata, "__version__", "unknown")),
    "dca_module": str(module_path),
    "dca_version": str(getattr(dca, "__version__", "unknown")),
    "numpy_version": str(numpy.__version__),
    "python_version": sys.version.split()[0],
    "scanpy_version": str(getattr(scanpy, "__version__", "unknown")),
    "tensorflow_version": str(tensorflow.__version__),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


@dataclass(frozen=True, slots=True)
class DCAConfig:
    """Pinned DCA architecture/default preprocessing with bounded training overrides."""

    ae_type: str = "zinb-conddisp"
    normalize_per_cell: bool = True
    scale: bool = True
    log1p: bool = True
    hidden_size: tuple[int, ...] = (64, 32, 64)
    hidden_dropout: float = 0.0
    batchnorm: bool = True
    activation: str = "relu"
    initializer: str = "glorot_uniform"
    epochs: int = 300
    reduce_lr: int = 10
    early_stop: int = 15
    batch_size: int = 32
    optimizer: str = "RMSprop"

    def __post_init__(self) -> None:
        if self.ae_type != "zinb-conddisp":
            raise ValueError("ae_type must remain zinb-conddisp")
        if self.normalize_per_cell is not True:
            raise ValueError("normalize_per_cell must remain True")
        if self.scale is not True:
            raise ValueError("scale must remain True")
        if self.log1p is not True:
            raise ValueError("log1p must remain True")
        if (
            type(self.hidden_size) is not tuple
            or not self.hidden_size
            or any(type(value) is not int or value <= 0 for value in self.hidden_size)
        ):
            raise ValueError("hidden_size must be a nonempty positive-integer tuple")
        if (
            isinstance(self.hidden_dropout, bool)
            or not isinstance(self.hidden_dropout, (int, float))
            or not 0 <= self.hidden_dropout < 1
        ):
            raise ValueError("hidden_dropout must be in [0, 1)")
        if self.batchnorm is not True:
            raise ValueError("batchnorm must remain True")
        if self.activation != "relu":
            raise ValueError("activation must remain relu")
        if self.initializer != "glorot_uniform":
            raise ValueError("initializer must remain glorot_uniform")
        for name, value in (
            ("epochs", self.epochs),
            ("reduce_lr", self.reduce_lr),
            ("early_stop", self.early_stop),
            ("batch_size", self.batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.optimizer != "RMSprop":
            raise ValueError("optimizer must remain RMSprop")


def dca_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Declare DCA's restored denoised mean as a count equivalent."""

    return raw_output_to_count_equivalent(method_input, native_output)


def finalize_dca_output(
    spec: MethodSpec,
    method_input: MethodInput,
    count_output: object,
) -> MethodOutputSnapshot:
    """Bind upstream DCA's count-scale denoised mean without selective copying."""

    require_method_spec(
        spec,
        "dca",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    return snapshot_method_output(
        spec,
        method_input,
        count_output,
        source_dataset_sha256=method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def run_dca(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: DCAConfig = DCAConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    """Run pinned DCA with its public API and raw-count output convention."""

    require_method_spec(
        spec,
        "dca",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2^31)")
    if not isinstance(config, DCAConfig):
        raise TypeError("config must be a DCAConfig")
    executable = require_executable(python_executable)
    compatibility = [
        CompatibilityEvent(
            "input_scale_conversion",
            "upstream DCA receives raw counts and applies its default normalize/log1p/scale training transform",
        ),
        CompatibilityEvent(
            "upstream_parameters",
            f"ae_type={config.ae_type}, hidden={config.hidden_size}, dropout={config.hidden_dropout}, "
            f"batchnorm={config.batchnorm}, activation={config.activation}, init={config.initializer}, "
            f"optimizer={config.optimizer}",
        ),
        CompatibilityEvent(
            "seed_binding", "study model seed is passed as DCA random_state"
        ),
        CompatibilityEvent(
            "determinism_limit",
            "pinned DCA sets Python/NumPy/TensorFlow seeds but legacy TensorFlow kernels are not guaranteed bitwise deterministic",
        ),
        CompatibilityEvent(
            "output_convention",
            "returned full upstream denoised count mean with library-size effects restored",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native raw-count snapshot is a count equivalent and then uses the shared log2(1 + counts/row_total*10000) transform; zero-library rows fail closed",
        ),
        CompatibilityEvent("compatibility_shims", "none"),
    ]
    if (config.hidden_size, float(config.hidden_dropout)) != ((64, 32, 64), 0.0):
        compatibility.append(
            CompatibilityEvent(
                "upstream_architecture_override",
                "hidden_size or hidden_dropout differs from the pinned default",
            )
        )
    if (
        config.epochs,
        config.reduce_lr,
        config.early_stop,
        config.batch_size,
    ) != (300, 10, 15, 32):
        compatibility.append(
            CompatibilityEvent(
                "upstream_training_override",
                "training controls differ from defaults: "
                f"epochs={config.epochs}, reduce_lr={config.reduce_lr}, "
                f"early_stop={config.early_stop}, batch_size={config.batch_size}",
            )
        )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-dca-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.npy"
        output_path = work_dir / "output.npy"
        receipt_path = work_dir / "receipt.tsv"
        np.save(input_path, method_input.counts, allow_pickle=False)
        command = (
            str(executable),
            "-I",
            "-c",
            _DCA_DRIVER,
            str(source_dir.resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(seed),
            config.ae_type,
            str(config.normalize_per_cell).lower(),
            str(config.scale).lower(),
            str(config.log1p).lower(),
            ",".join(str(value) for value in config.hidden_size),
            repr(float(config.hidden_dropout)),
            str(config.batchnorm).lower(),
            config.activation,
            config.initializer,
            str(config.epochs),
            str(config.reduce_lr),
            str(config.early_stop),
            str(config.batch_size),
            config.optimizer,
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
                    "anndata_version",
                    "dca_module",
                    "dca_version",
                    "numpy_version",
                    "python_version",
                    "scanpy_version",
                    "tensorflow_version",
                }
            ),
        )
        snapshot = finalize_dca_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


__all__ = [
    "DCAConfig",
    "dca_to_evaluator_counts",
    "finalize_dca_output",
    "run_dca",
]
