"""Pinned scVI adapter with explicit count-equivalent output conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .direct import DirectAdapterExecution, DirectMethodOutput, finalize_direct_method_output
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    execute_pinned_command,
    read_environment_receipt,
    read_npy_output,
    raw_output_to_count_equivalent,
    require_executable,
    require_method_spec,
)


_SCVI_DRIVER = r"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 17:
    raise RuntimeError("adapter expected sixteen arguments")
source_python = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
batch_path = None if sys.argv[3] == "none" else Path(sys.argv[3])
output_path = Path(sys.argv[4])
receipt_path = Path(sys.argv[5])
seed = int(sys.argv[6])
n_hidden = int(sys.argv[7])
n_latent = int(sys.argv[8])
n_layers = int(sys.argv[9])
dropout_rate = float(sys.argv[10])
dispersion = sys.argv[11]
gene_likelihood = sys.argv[12]
use_observed_lib_size = sys.argv[13] == "true"
latent_distribution = sys.argv[14]
max_epochs = None if sys.argv[15] == "none" else int(sys.argv[15])
batch_size = int(sys.argv[16])
sys.path.insert(0, str(source_python))
import anndata
import scvi
import torch

module_path = Path(scvi.__file__).resolve(strict=True)
if source_python not in module_path.parents:
    raise RuntimeError("imported scVI module is not from the pinned checkout")
counts = np.load(input_path, allow_pickle=False)
if type(counts) is not np.ndarray or counts.ndim != 2:
    raise RuntimeError("input count matrix is malformed")
obs = pd.DataFrame(index=[f"cell-{index}" for index in range(counts.shape[0])])
batch_key = None
if batch_path is not None:
    batch = np.load(batch_path, allow_pickle=False)
    if type(batch) is not np.ndarray or batch.shape != (counts.shape[0],):
        raise RuntimeError("batch covariate is malformed")
    obs["_batch"] = pd.Categorical(batch.astype(str))
    batch_key = "_batch"
adata = anndata.AnnData(X=counts, obs=obs)
scvi.settings.seed = seed
scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
model = scvi.model.SCVI(
    adata,
    n_hidden=n_hidden,
    n_latent=n_latent,
    n_layers=n_layers,
    dropout_rate=dropout_rate,
    dispersion=dispersion,
    gene_likelihood=gene_likelihood,
    use_observed_lib_size=use_observed_lib_size,
    latent_distribution=latent_distribution,
)
model.train(max_epochs=max_epochs, batch_size=batch_size)
frequencies = model.get_normalized_expression(
    library_size=1,
    n_samples=1,
    return_mean=True,
    return_numpy=True,
    silent=True,
)
frequencies = np.asarray(frequencies, dtype=np.float64)
if frequencies.shape != counts.shape or not np.isfinite(frequencies).all() or (frequencies < 0).any():
    raise RuntimeError("scVI output is invalid")
np.save(output_path, frequencies, allow_pickle=False)
receipt = {
    "anndata_version": str(getattr(anndata, "__version__", "unknown")),
    "python_version": sys.version.split()[0],
    "scvi_module": str(module_path),
    "scvi_version": str(getattr(scvi, "__version__", "unknown")),
    "torch_version": str(torch.__version__),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


@dataclass(frozen=True, slots=True)
class SCVIConfig:
    """Exact pinned scVI model defaults and declared batch-covariate rule."""

    n_hidden: int = 128
    n_latent: int = 10
    n_layers: int = 1
    dropout_rate: float = 0.1
    dispersion: str = "gene"
    gene_likelihood: str = "zinb"
    use_observed_lib_size: bool = True
    latent_distribution: str = "normal"
    max_epochs: int | None = None
    batch_size: int = 128
    batch_key: str | None = "batch"

    def __post_init__(self) -> None:
        for name, value in (
            ("n_hidden", self.n_hidden),
            ("n_latent", self.n_latent),
            ("n_layers", self.n_layers),
            ("batch_size", self.batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.dropout_rate, bool)
            or not isinstance(self.dropout_rate, (int, float))
            or not math.isfinite(self.dropout_rate)
            or not 0 <= self.dropout_rate < 1
        ):
            raise ValueError("dropout_rate must be finite and in [0, 1)")
        if self.dispersion != "gene":
            raise ValueError("dispersion must remain gene")
        if self.gene_likelihood != "zinb":
            raise ValueError("gene_likelihood must remain zinb")
        if self.use_observed_lib_size is not True:
            raise ValueError("use_observed_lib_size must remain True")
        if self.latent_distribution != "normal":
            raise ValueError("latent_distribution must remain normal")
        if self.max_epochs is not None and (
            type(self.max_epochs) is not int or self.max_epochs <= 0
        ):
            raise ValueError("max_epochs must be null or a positive integer")
        if self.batch_key is not None and (
            not isinstance(self.batch_key, str) or not self.batch_key
        ):
            raise ValueError("batch_key must be null or a nonempty string")


def frequencies_to_observed_library_counts(
    method_input: MethodInput,
    frequencies: object,
) -> np.ndarray:
    """Map scVI decoded frequencies to count-equivalent observed library sizes."""

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if type(frequencies) is not np.ndarray or frequencies.shape != method_input.shape:
        raise ValueError("scVI frequencies must match the method-input shape")
    if frequencies.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("scVI frequencies must be numeric")
    values = np.array(frequencies, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("scVI frequencies must be finite and nonnegative")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=1e-5, atol=1e-7):
        raise ValueError("scVI frequencies must sum to one within each cell")
    library_sizes = method_input.counts.sum(axis=1)
    if bool((library_sizes == 0).any()):
        raise AdapterUnavailableError(
            "zero_library_cell", "scVI count conversion requires nonzero libraries"
        )
    return values * library_sizes[:, None]


def scvi_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Declare the adapter's observed-library scVI output as count equivalent."""

    return raw_output_to_count_equivalent(method_input, native_output)


def finalize_scvi_output(
    spec: MethodSpec,
    method_input: MethodInput,
    decoded_frequencies: object,
) -> MethodOutputSnapshot:
    """Convert full scVI frequencies to count equivalents and bind evaluator IDs."""

    require_method_spec(
        spec,
        "scvi",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    count_output = frequencies_to_observed_library_counts(
        method_input, decoded_frequencies
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


def finalize_scvi_direct_output(
    spec: MethodSpec,
    method_input: MethodInput,
    decoded_frequencies: object,
) -> DirectMethodOutput:
    """Validate scVI output without deriving a content identity."""

    require_method_spec(
        spec,
        "scvi",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    count_output = frequencies_to_observed_library_counts(
        method_input, decoded_frequencies
    )
    return finalize_direct_method_output(
        spec,
        method_input,
        count_output,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def _run_scvi_impl(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCVIConfig = SCVIConfig(),
    work_root: Path | None = None,
    _direct: bool = False,
) -> AdapterExecution | DirectAdapterExecution:
    """Run exact pinned scVI source in an explicit dependency environment."""

    require_method_spec(
        spec,
        "scvi",
        input_scale="raw_counts",
        output_scale="raw_counts",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if not isinstance(config, SCVIConfig):
        raise TypeError("config must be an SCVIConfig")
    executable = require_executable(python_executable)
    obs = method_input.covariate_frame("obs")
    batch_values: np.ndarray | None = None
    if config.batch_key is not None and config.batch_key in obs:
        batch_values = obs[config.batch_key].astype(str).to_numpy(dtype=str)
        batch_disclosure = (
            f"allowed obs covariate {config.batch_key!r} used as batch_key"
        )
    else:
        batch_disclosure = "no declared batch covariate supplied to setup_anndata"
    compatibility = [
        CompatibilityEvent("input_scale_conversion", "none; scVI receives raw counts"),
        CompatibilityEvent(
            "upstream_parameters",
            f"n_hidden={config.n_hidden}, n_latent={config.n_latent}, n_layers={config.n_layers}, "
            f"dropout={config.dropout_rate}, dispersion={config.dispersion}, "
            f"likelihood={config.gene_likelihood}, observed_library={config.use_observed_lib_size}, "
            f"latent_distribution={config.latent_distribution}, batch_size={config.batch_size}",
        ),
        CompatibilityEvent("batch_covariate_rule", batch_disclosure),
        CompatibilityEvent("seed_binding", "study seed assigned to scvi.settings.seed"),
        CompatibilityEvent(
            "output_scale_conversion",
            "full decoded frequencies are multiplied row-wise by observed library sizes",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "observed-library decoded output is a count equivalent and then uses the shared log2(1 + counts/row_total*10000) transform; zero-library rows fail closed",
        ),
        CompatibilityEvent(
            "observed_positive_policy", "no observed entries are copied after decoding"
        ),
        CompatibilityEvent("compatibility_shims", "none"),
    ]
    if (
        config.n_hidden,
        config.n_latent,
        config.n_layers,
        float(config.dropout_rate),
    ) != (128, 10, 1, 0.1):
        compatibility.append(
            CompatibilityEvent(
                "upstream_architecture_override",
                "one or more scVI architecture values differs from the pinned default",
            )
        )
    if config.max_epochs is not None:
        compatibility.append(
            CompatibilityEvent(
                "upstream_training_override",
                f"max_epochs={config.max_epochs} replaces the upstream heuristic",
            )
        )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-scvi-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.npy"
        batch_path = work_dir / "batch.npy"
        output_path = work_dir / "output.npy"
        receipt_path = work_dir / "receipt.tsv"
        np.save(input_path, method_input.counts, allow_pickle=False)
        if batch_values is not None:
            np.save(batch_path, batch_values, allow_pickle=False)
        command = (
            str(executable),
            "-I",
            "-c",
            _SCVI_DRIVER,
            str((source_dir / "src").resolve()),
            str(input_path),
            "none" if batch_values is None else str(batch_path),
            str(output_path),
            str(receipt_path),
            str(seed),
            str(config.n_hidden),
            str(config.n_latent),
            str(config.n_layers),
            repr(float(config.dropout_rate)),
            config.dispersion,
            config.gene_likelihood,
            str(config.use_observed_lib_size).lower(),
            config.latent_distribution,
            "none" if config.max_epochs is None else str(config.max_epochs),
            str(config.batch_size),
        )
        result = execute_pinned_command(
            spec,
            source_dir,
            command,
            cwd=work_dir,
            timeout_seconds=spec.resources.timeout_seconds,
        )
        frequencies = read_npy_output(output_path)
        receipt = read_environment_receipt(
            receipt_path,
            expected_keys=frozenset(
                {
                    "anndata_version",
                    "python_version",
                    "scvi_module",
                    "scvi_version",
                    "torch_version",
                }
            ),
        )
        if _direct:
            return DirectAdapterExecution(
                output=finalize_scvi_direct_output(spec, method_input, frequencies),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        snapshot = finalize_scvi_output(spec, method_input, frequencies)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


def run_scvi(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    python_executable: Path, seed: int, config: SCVIConfig = SCVIConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    return _run_scvi_impl(spec, method_input, source_dir=source_dir,
                          python_executable=python_executable, seed=seed,
                          config=config, work_root=work_root)


def run_scvi_direct(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    python_executable: Path, seed: int, config: SCVIConfig = SCVIConfig(),
    work_root: Path | None = None,
) -> DirectAdapterExecution:
    return _run_scvi_impl(spec, method_input, source_dir=source_dir,
                          python_executable=python_executable, seed=seed,
                          config=config, work_root=work_root, _direct=True)


__all__ = [
    "SCVIConfig",
    "finalize_scvi_direct_output",
    "finalize_scvi_output",
    "frequencies_to_observed_library_counts",
    "run_scvi",
    "run_scvi_direct",
    "scvi_to_evaluator_counts",
]
