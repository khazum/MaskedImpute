"""Pinned BiAEImpute adapter around its public model and dataset classes."""

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


_BIAE_DRIVER = r"""
from pathlib import Path
import importlib
import itertools
import random
import sys
import numpy as np

if len(sys.argv) != 14:
    raise RuntimeError("adapter expected thirteen arguments")
source_python = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
seed = int(sys.argv[5])
epochs = int(sys.argv[6])
latent_size = int(sys.argv[7])
learning_rate = float(sys.argv[8])
beta1 = float(sys.argv[9])
beta2 = float(sys.argv[10])
row_batch_size = int(sys.argv[11])
column_batch_size = int(sys.argv[12])
device_name = None if sys.argv[13] == "none" else sys.argv[13]
sys.path.insert(0, str(source_python))
import torch
model_module = importlib.import_module("model")
utils_module = importlib.import_module("utils")

model_path = Path(model_module.__file__).resolve(strict=True)
utils_path = Path(utils_module.__file__).resolve(strict=True)
if source_python not in model_path.parents or source_python not in utils_path.parents:
    raise RuntimeError("imported BiAEImpute modules are not from the pinned checkout")
counts = np.load(input_path, allow_pickle=False)
if type(counts) is not np.ndarray or counts.ndim != 2:
    raise RuntimeError("input count matrix is malformed")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
device = torch.device(
    device_name if device_name is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
)
data = torch.tensor(counts, dtype=torch.float32, device=device)
row_loader = torch.utils.data.DataLoader(
    utils_module.RowDataset(data),
    batch_size=row_batch_size,
    shuffle=True,
    drop_last=False,
)
column_loader = torch.utils.data.DataLoader(
    utils_module.ColDataset(data),
    batch_size=column_batch_size,
    shuffle=True,
    drop_last=False,
)
n_cells, n_genes = counts.shape
row_encoder = model_module.Row_Encoder(n_genes, latent_size).to(device)
row_decoder = model_module.Row_Decoder(latent_size, n_genes).to(device)
column_encoder = model_module.Col_Encoder(n_cells, latent_size).to(device)
column_decoder = model_module.Col_Decoder(latent_size, n_cells).to(device)
optimizer = torch.optim.Adam(
    itertools.chain(
        row_encoder.parameters(),
        row_decoder.parameters(),
        column_encoder.parameters(),
        column_decoder.parameters(),
    ),
    lr=learning_rate,
    betas=(beta1, beta2),
)
for _ in range(epochs):
    for (row_data, row_idx), (column_data, column_idx) in zip(
        row_loader, column_loader
    ):
        row_data = row_data.to(device)
        column_data = column_data.to(device)
        row_output = row_decoder(row_encoder(row_data))
        column_output = column_decoder(column_encoder(column_data))
        row_mask = torch.where(
            row_data == 0, torch.zeros_like(row_data), torch.ones_like(row_data)
        )
        column_mask = torch.where(
            column_data == 0,
            torch.zeros_like(column_data),
            torch.ones_like(column_data),
        )
        row_loss = ((row_output * row_mask - row_data) ** 2).sum() / row_mask.sum()
        column_loss = (
            ((column_output * column_mask - column_data) ** 2).sum()
            / column_mask.sum()
        )
        row_cross = row_output[:, column_idx]
        column_cross = column_output[:, row_idx].T
        cross_loss = ((row_cross - column_cross) ** 2).mean()
        optimizer.zero_grad()
        (row_loss + column_loss + cross_loss).backward()
        optimizer.step()
for module in (row_encoder, row_decoder, column_encoder, column_decoder):
    module.eval()
row_predictions = []
column_predictions = []
with torch.no_grad():
    for cell in data:
        decoded = row_decoder(row_encoder(cell.reshape(1, -1)))
        row_predictions.append(decoded.cpu().numpy().squeeze())
    for gene in data.T:
        decoded = column_decoder(column_encoder(gene.reshape(1, -1)))
        column_predictions.append(decoded.cpu().numpy().squeeze())
row_predictions = np.asarray(row_predictions, dtype=np.float64)
column_predictions = np.asarray(column_predictions, dtype=np.float64).T
average = (row_predictions + column_predictions) / 2.0
output = np.asarray(counts, dtype=np.float64).copy()
output[output == 0] = average[output == 0]
if output.shape != counts.shape or not np.isfinite(output).all() or (output < 0).any():
    raise RuntimeError("BiAEImpute output is invalid")
np.save(output_path, output, allow_pickle=False)
receipt = {
    "device": str(device),
    "model_module": str(model_path),
    "numpy_version": str(np.__version__),
    "python_version": sys.version.split()[0],
    "torch_version": str(torch.__version__),
    "utils_module": str(utils_path),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


@dataclass(frozen=True, slots=True)
class BiAEImputeConfig:
    """Pinned training defaults with benchmark-safe no-remasking policy."""

    epochs: int = 500
    latent_size: int = 128
    learning_rate: float = 0.0002
    beta1: float = 0.9
    beta2: float = 0.999
    row_batch_size: int = 31
    column_batch_size: int = 200
    mask_ratio: float = 0.0
    device: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("latent_size", self.latent_size),
            ("row_batch_size", self.row_batch_size),
            ("column_batch_size", self.column_batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("beta1", self.beta1),
            ("beta2", self.beta2),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in (0, 1)")
        if self.mask_ratio != 0.0:
            raise ValueError(
                "mask_ratio must remain 0 so the benchmark input is not synthetically remasked"
            )
        if self.device not in {None, "cpu", "cuda"}:
            raise ValueError("device must be null, cpu, or cuda")


def biaeimpute_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Declare BiAEImpute's raw-scale output as count equivalent."""

    return raw_output_to_count_equivalent(method_input, native_output)


def finalize_biaeimpute_output(
    spec: MethodSpec,
    method_input: MethodInput,
    raw_output: object,
) -> MethodOutputSnapshot:
    """Bind untouched BiAEImpute raw output to evaluator IDs."""

    require_method_spec(
        spec,
        "biaeimpute",
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


def finalize_biaeimpute_direct_output(
    spec: MethodSpec,
    method_input: MethodInput,
    raw_output: object,
) -> DirectMethodOutput:
    """Validate BiAEImpute output without deriving a content identity."""

    require_method_spec(
        spec,
        "biaeimpute",
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


def _run_biaeimpute_impl(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: BiAEImputeConfig = BiAEImputeConfig(),
    work_root: Path | None = None,
    _direct: bool = False,
) -> AdapterExecution | DirectAdapterExecution:
    """Run pinned BiAEImpute classes with repaired adapter-only orchestration."""

    require_method_spec(
        spec,
        "biaeimpute",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if not isinstance(config, BiAEImputeConfig):
        raise TypeError("config must be a BiAEImputeConfig")
    executable = require_executable(python_executable)
    compatibility = [
        CompatibilityEvent("input_scale_conversion", "none; model receives raw counts"),
        CompatibilityEvent(
            "broken_cli_compatibility",
            "pinned train.py passes an unsupported normalization argument and hard-codes a Windows data root; adapter invokes the pinned model/dataset classes without editing source",
        ),
        CompatibilityEvent(
            "benchmark_mask_policy",
            "mask_ratio=0 prevents upstream artificial dropout because the benchmark input is already the prespecified observed matrix",
        ),
        CompatibilityEvent(
            "upstream_defaults",
            f"epochs={config.epochs}, latent={config.latent_size}, lr={config.learning_rate}, "
            f"betas=({config.beta1},{config.beta2}), row_batch={config.row_batch_size}, "
            f"column_batch={config.column_batch_size}",
        ),
        CompatibilityEvent(
            "seed_binding", "study seed initializes Python, NumPy, Torch, and CUDA RNGs"
        ),
        CompatibilityEvent(
            "upstream_selective_policy",
            "pinned inference averages row/column decoders and replaces only zero entries",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native raw output is a count equivalent and then uses the shared log2(CP10k+1) transform",
        ),
    ]
    if config.epochs != 500:
        compatibility.append(
            CompatibilityEvent("upstream_training_override", f"epochs={config.epochs}")
        )
    if config.device is not None:
        compatibility.append(
            CompatibilityEvent(
                "execution_device_override", f"explicit device={config.device}"
            )
        )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-biae-", dir=work_root) as temporary:
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
            _BIAE_DRIVER,
            str((source_dir / "BiAEImpute-main").resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(seed),
            str(config.epochs),
            str(config.latent_size),
            repr(float(config.learning_rate)),
            repr(float(config.beta1)),
            repr(float(config.beta2)),
            str(config.row_batch_size),
            str(config.column_batch_size),
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
                    "model_module",
                    "numpy_version",
                    "python_version",
                    "torch_version",
                    "utils_module",
                }
            ),
        )
        if _direct:
            return DirectAdapterExecution(
                output=finalize_biaeimpute_direct_output(spec, method_input, output),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        snapshot = finalize_biaeimpute_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


def run_biaeimpute(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: BiAEImputeConfig = BiAEImputeConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    return _run_biaeimpute_impl(
        spec,
        method_input,
        source_dir=source_dir,
        python_executable=python_executable,
        seed=seed,
        config=config,
        work_root=work_root,
    )


def run_biaeimpute_direct(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: BiAEImputeConfig = BiAEImputeConfig(),
    work_root: Path | None = None,
) -> DirectAdapterExecution:
    return _run_biaeimpute_impl(
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
    "BiAEImputeConfig",
    "finalize_biaeimpute_direct_output",
    "biaeimpute_to_evaluator_counts",
    "finalize_biaeimpute_output",
    "run_biaeimpute",
    "run_biaeimpute_direct",
]
