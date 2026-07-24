"""Faithful legacy TensorFlow 1.12 boundary for the pinned scSDAE source."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .base import (
    MethodContractError,
    MethodInput,
    MethodOutputSnapshot,
    MethodSpec,
    snapshot_method_output,
)
from .direct import (
    DirectAdapterExecution,
    DirectMethodOutput,
    finalize_direct_method_output,
)
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    SourceReceipt,
    _inverse_log1p_observed_library,
    execute_pinned_command,
    observed_library_sizes,
    read_environment_receipt,
    read_npy_output,
    require_executable,
    require_method_spec,
    verify_pinned_source,
)


_SCSDAE_PROBE_DRIVER = r"""
from pathlib import Path
import os
import sys

if len(sys.argv) != 3:
    raise RuntimeError("adapter probe expected two arguments")
source_script = Path(sys.argv[1]).resolve(strict=True)
gpu_index = sys.argv[2]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
print(
    "MASKIMPUTE_SCSDAE_PREFLIGHT "
    f"python={sys.version.split()[0]} executable={sys.executable} "
    f"source={source_script}",
    flush=True,
)
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib

print(
    "MASKIMPUTE_SCSDAE_PREFLIGHT "
    f"numpy={np.__version__} pandas={pd.__version__} "
    f"tensorflow={tf.__version__} keras={keras.__version__}",
    flush=True,
)
if sys.version_info[:2] not in {(3, 6), (3, 7)}:
    raise RuntimeError(
        "MASKIMPUTE_LEGACY_ENVIRONMENT_MISMATCH: scSDAE requires Python 3.6/3.7"
    )
if str(tf.__version__) != "1.12.0" or str(keras.__version__) != "2.2.4":
    raise RuntimeError(
        "MASKIMPUTE_LEGACY_ENVIRONMENT_MISMATCH: scSDAE requires "
        "TensorFlow 1.12.0 and Keras 2.2.4"
    )
probe_config = tf.ConfigProto(allow_soft_placement=False)
probe_config.gpu_options.allow_growth = True
tensorflow_memory_growth = bool(probe_config.gpu_options.allow_growth)
if not tensorflow_memory_growth:
    raise RuntimeError("TensorFlow memory-growth policy was not applied")
print(
    "MASKIMPUTE_SCSDAE_PREFLIGHT "
    f"tensorflow_memory_growth={str(tensorflow_memory_growth).lower()}",
    flush=True,
)
gpu_available = any(
    device.device_type == "GPU"
    for device in device_lib.list_local_devices(session_config=probe_config)
)
print(
    f"MASKIMPUTE_SCSDAE_PREFLIGHT gpu_available={gpu_available} "
    f"gpu_index={gpu_index}",
    flush=True,
)
if not gpu_available:
    raise RuntimeError(
        "MASKIMPUTE_LEGACY_GPU_UNAVAILABLE: frozen method resources require a GPU"
    )
try:
    probe_graph = tf.Graph()
    with probe_graph.as_default():
        with tf.device("/gpu:0"):
            probe_value = tf.matmul(
                tf.constant([[1.0]], dtype=tf.float32),
                tf.constant([[1.0]], dtype=tf.float32),
            )
        with tf.Session(graph=probe_graph, config=probe_config) as probe_session:
            observed_probe = probe_session.run(probe_value)
    if observed_probe.shape != (1, 1) or float(observed_probe[0, 0]) != 1.0:
        raise RuntimeError("logical GPU0 matrix probe returned an invalid value")
except Exception:
    print(
        "MASKIMPUTE_LEGACY_GPU_KERNEL_INCOMPATIBLE gpu=/gpu:0",
        file=sys.stderr,
        flush=True,
    )
    raise
print("MASKIMPUTE_SCSDAE_PREFLIGHT gpu0_kernel=ok", flush=True)
"""


_SCSDAE_DRIVER = r"""
from pathlib import Path
import os
import random
import runpy
import sys

if len(sys.argv) != 16:
    raise RuntimeError("adapter expected fifteen arguments")
source_script = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
output_dir = Path(sys.argv[4])
receipt_path = Path(sys.argv[5])
seed = int(sys.argv[6])
batch_size = int(sys.argv[7])
autoencoder_iterations = int(sys.argv[8])
pretrain_iterations = int(sys.argv[9])
zero_loss_weight = float(sys.argv[10])
observed_loss_weight = float(sys.argv[11])
dropout_rate = float(sys.argv[12])
l1_regularization = float(sys.argv[13])
l2_regularization = float(sys.argv[14])
gpu_index = sys.argv[15]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
print(
    "MASKIMPUTE_SCSDAE_PREFLIGHT "
    f"python={sys.version.split()[0]} executable={sys.executable} "
    f"source={source_script}",
    flush=True,
)
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

print(
    "MASKIMPUTE_SCSDAE_PREFLIGHT "
    f"numpy={np.__version__} pandas={pd.__version__} "
    f"tensorflow={tf.__version__} keras={keras.__version__}",
    flush=True,
)
if sys.version_info[:2] not in {(3, 6), (3, 7)}:
    raise RuntimeError(
        "MASKIMPUTE_LEGACY_ENVIRONMENT_MISMATCH: scSDAE requires Python 3.6/3.7"
    )
if str(tf.__version__) != "1.12.0" or str(keras.__version__) != "2.2.4":
    raise RuntimeError(
        "MASKIMPUTE_LEGACY_ENVIRONMENT_MISMATCH: scSDAE requires "
        "TensorFlow 1.12.0 and Keras 2.2.4"
    )

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
tensorflow_memory_growth = bool(run_config.gpu_options.allow_growth)
if not tensorflow_memory_growth:
    raise RuntimeError("TensorFlow memory-growth policy was not applied")
keras.backend.set_session(tf.Session(config=run_config))
print(
    "MASKIMPUTE_SCSDAE_PREFLIGHT "
    f"tensorflow_memory_growth={str(tensorflow_memory_growth).lower()}",
    flush=True,
)

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
source_arguments = [
    str(source_script),
    f"--batch_size={batch_size}",
    f"--n_iters_ae={autoencoder_iterations}",
    f"--n_iters_pretrain={pretrain_iterations}",
    f"--alpha={zero_loss_weight}",
    f"--beta={observed_loss_weight}",
    f"--dr_rate={dropout_rate}",
    f"--nu1={l1_regularization}",
    f"--nu2={l2_regularization}",
    f"--train_datapath={input_path}",
    "--data_type=count",
    f"--outDir={str(output_dir) + os.sep}",
    "--name=benchmark",
    "--no-gene_scale",
    f"--GPU_SET={gpu_index}",
]
sys.argv = source_arguments
runpy.run_path(str(source_script), run_name="__main__")

upstream_output = output_dir / "autoencoder_r.csv"
if not upstream_output.is_file():
    raise RuntimeError("pinned scSDAE did not write autoencoder_r.csv")
# Pinned pandas.to_csv writes its default integer column header even though
# upstream disables only the row index.  Consume that serialization header;
# treating it as data adds one synthetic cell to every result.
native = pd.read_csv(upstream_output, header=0).values
native = np.asarray(native, dtype=np.float64)
source_frame = pd.read_csv(input_path, index_col=0)
expected_shape = (source_frame.shape[1], source_frame.shape[0])
if native.shape != expected_shape:
    raise RuntimeError(
        f"scSDAE output shape {native.shape} differs from {expected_shape}"
    )
if not np.isfinite(native).all() or (native < 0).any():
    raise RuntimeError("scSDAE output is not finite and nonnegative")
np.save(output_path, native, allow_pickle=False)
receipt = {
    "gpu_available": "true",
    "gpu_index": gpu_index,
    "keras_version": str(keras.__version__),
    "numpy_version": str(np.__version__),
    "pandas_version": str(pd.__version__),
    "python_version": sys.version.split()[0],
    "source_script": str(source_script),
    "tensorflow_memory_growth": str(tensorflow_memory_growth).lower(),
    "tensorflow_version": str(tf.__version__),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


@dataclass(frozen=True, slots=True)
class SCSDaeConfig:
    """Exact scientific defaults plus the study host's physical GPU binding."""

    batch_size: int = 256
    autoencoder_iterations: int = 2000
    pretrain_iterations: int = 1000
    zero_loss_weight: float = 1.0
    observed_loss_weight: float = 1.0
    dropout_rate: float = 0.2
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0
    gene_scale: bool = False
    gpu_index: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("autoencoder_iterations", self.autoencoder_iterations),
            ("pretrain_iterations", self.pretrain_iterations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("zero_loss_weight", self.zero_loss_weight),
            ("observed_loss_weight", self.observed_loss_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if float(self.observed_loss_weight) != 1.0:
            raise ValueError(
                "observed_loss_weight must remain 1.0 because the pinned trainer hard-codes beta=1.0"
            )
        if (
            isinstance(self.dropout_rate, bool)
            or not isinstance(self.dropout_rate, (int, float))
            or not math.isfinite(float(self.dropout_rate))
            or not 0.0 <= float(self.dropout_rate) < 1.0
        ):
            raise ValueError("dropout_rate must be finite and in [0, 1)")
        for name, value in (
            ("l1_regularization", self.l1_regularization),
            ("l2_regularization", self.l2_regularization),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.gene_scale is not False:
            raise ValueError(
                "gene_scale must remain False because per-gene min-max scaling has no declared evaluator inverse"
            )
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError("gpu_index must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class SCSDaeAttemptReceipt:
    """Source-bound evidence for one legacy environment attempt."""

    source_revision: str
    source_tree: str
    source_url: str
    environment_id: str
    environment_registry_status: str
    executable: str
    outcome: str
    reason_code: str
    command: tuple[str, ...] | None
    stdout_sha256: str
    stderr_sha256: str
    probe_command: tuple[str, ...] | None
    run_command: tuple[str, ...] | None
    probe_stdout_sha256: str
    probe_stderr_sha256: str
    run_stdout_sha256: str
    run_stderr_sha256: str


class SCSDaeUnavailableError(AdapterUnavailableError):
    """Unavailable scSDAE attempt carrying immutable source/environment evidence."""

    def __init__(
        self,
        error: AdapterUnavailableError,
        *,
        source: SourceReceipt,
        spec: MethodSpec,
        executable: Path,
        probe_command: tuple[str, ...] | None = None,
        probe_stdout: bytes = b"",
        probe_stderr: bytes = b"",
        run_command: tuple[str, ...] | None = None,
        run_stdout: bytes = b"",
        run_stderr: bytes = b"",
    ) -> None:
        super().__init__(
            error.reason_code,
            error.detail,
            command=error.command,
            stdout=error.stdout,
            stderr=error.stderr,
        )
        self.attempt_receipt = SCSDaeAttemptReceipt(
            source_revision=source.revision,
            source_tree=source.tree,
            source_url=source.url,
            environment_id=spec.environment.id,
            environment_registry_status=spec.environment.status,
            executable=str(executable),
            outcome="unavailable",
            reason_code=self.reason_code,
            command=self.command,
            stdout_sha256=self.stdout_sha256,
            stderr_sha256=self.stderr_sha256,
            probe_command=probe_command,
            run_command=run_command,
            probe_stdout_sha256=hashlib.sha256(probe_stdout).hexdigest(),
            probe_stderr_sha256=hashlib.sha256(probe_stderr).hexdigest(),
            run_stdout_sha256=hashlib.sha256(run_stdout).hexdigest(),
            run_stderr_sha256=hashlib.sha256(run_stderr).hexdigest(),
        )


def _legacy_failure_reason(error: AdapterUnavailableError) -> AdapterUnavailableError:
    combined = error.stdout + b"\n" + error.stderr
    if b"MASKIMPUTE_LEGACY_ENVIRONMENT_MISMATCH" in combined:
        return AdapterUnavailableError(
            "legacy_environment_mismatch",
            "selected executable is not the required Python 3.6/3.7, TensorFlow 1.12.0, Keras 2.2.4 environment",
            command=error.command,
            stdout=error.stdout,
            stderr=error.stderr,
        )
    if b"MASKIMPUTE_LEGACY_GPU_UNAVAILABLE" in combined:
        return AdapterUnavailableError(
            "legacy_gpu_unavailable",
            "the exact legacy TensorFlow stack could not expose the registry-required GPU",
            command=error.command,
            stdout=error.stdout,
            stderr=error.stderr,
        )
    if b"MASKIMPUTE_LEGACY_GPU_INITIALIZATION_TIMEOUT" in combined:
        return AdapterUnavailableError(
            "legacy_gpu_initialization_timeout",
            "TensorFlow 1.12 could enumerate the modern GPU but could not initialize it within the fixed 30-second legacy probe",
            command=error.command,
            stdout=error.stdout,
            stderr=error.stderr,
        )
    if (
        b"MASKIMPUTE_LEGACY_GPU_KERNEL_INCOMPATIBLE" in combined
        or b"CUBLAS_STATUS_EXECUTION_FAILED" in combined
        or b"cuda_timer.cc" in combined
    ):
        return AdapterUnavailableError(
            "legacy_gpu_kernel_incompatible",
            "TensorFlow 1.12 exposed the registry-required logical GPU0 but failed the exact first matrix kernel probe",
            command=error.command,
            stdout=error.stdout,
            stderr=error.stderr,
        )
    return error


def _combine_probe_and_run_error(
    error: AdapterUnavailableError,
    probe_stdout: bytes,
    probe_stderr: bytes,
) -> AdapterUnavailableError:
    return AdapterUnavailableError(
        error.reason_code,
        error.detail,
        command=error.command,
        stdout=probe_stdout + b"\n" + error.stdout,
        stderr=probe_stderr + b"\n" + error.stderr,
    )


def _write_scsdae_input(path: Path, method_input: MethodInput) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("gene_id", *method_input.obs_ids))
        counts = method_input.counts
        for gene, identifier in enumerate(method_input.var_ids):
            writer.writerow(
                (
                    identifier,
                    *(format(float(value), ".17g") for value in counts[:, gene]),
                )
            )


def scsdae_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Invert scSDAE's natural-log counts-per-million native scale."""

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if type(native_output) is not np.ndarray:
        raise TypeError("native scSDAE output must be an exact ndarray")
    if native_output.shape != method_input.shape:
        raise ValueError("native scSDAE output must match the method-input shape")
    if native_output.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("native scSDAE output must be numeric")
    native = np.array(
        native_output,
        dtype=np.float64,
        copy=True,
        order="C",
        subok=False,
    )
    if not np.isfinite(native).all() or bool((native < 0).any()):
        raise ValueError("native scSDAE output must be finite and nonnegative")
    libraries = observed_library_sizes(method_input)
    converted = _inverse_log1p_observed_library(
        native,
        libraries,
        target_sum=1_000_000,
    )
    if not np.isfinite(converted).all() or bool((converted < 0).any()):
        raise ValueError("native scSDAE output does not have a finite count equivalent")
    converted.setflags(write=False)
    return converted


def finalize_scsdae_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> MethodOutputSnapshot:
    """Bind untouched scSDAE log-counts-per-million output to evaluator IDs."""

    require_method_spec(
        spec,
        "scsdae",
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


def finalize_scsdae_direct_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> DirectMethodOutput:
    """Validate scSDAE output without deriving a content identity."""

    require_method_spec(
        spec,
        "scsdae",
        input_scale="raw_counts",
        output_scale="method_native_normalized",
    )
    return finalize_direct_method_output(
        spec,
        method_input,
        normalized_output,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def _validate_work_root(work_root: Path | None, source_dir: Path) -> None:
    if work_root is None:
        return
    if not isinstance(work_root, Path):
        raise TypeError("work_root must be a pathlib.Path or None")
    source = source_dir.resolve(strict=True)
    candidate = work_root.resolve(strict=False)
    if candidate == source or source in candidate.parents:
        raise AdapterUnavailableError(
            "unsafe_work_root",
            "work_root must not be the pinned source tree or a directory inside it",
        )
    work_root.mkdir(parents=True, exist_ok=True)
    created = work_root.resolve(strict=True)
    if created == source or source in created.parents:
        raise AdapterUnavailableError(
            "unsafe_work_root",
            "resolved work_root must remain outside the pinned source tree",
        )


def _run_scsdae_impl(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCSDaeConfig = SCSDaeConfig(),
    work_root: Path | None = None,
    _direct: bool = False,
) -> AdapterExecution | DirectAdapterExecution:
    """Attempt exact legacy source; never substitute a modernized implementation."""

    require_method_spec(
        spec,
        "scsdae",
        input_scale="raw_counts",
        output_scale="method_native_normalized",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if not isinstance(config, SCSDaeConfig):
        raise TypeError("config must be an SCSDaeConfig")
    _validate_work_root(work_root, source_dir)
    source_receipt = verify_pinned_source(spec, source_dir)
    try:
        executable = require_executable(python_executable)
    except AdapterUnavailableError as error:
        if _direct:
            raise
        raise SCSDaeUnavailableError(
            error,
            source=source_receipt,
            spec=spec,
            executable=python_executable,
        ) from error

    compatibility = [
        CompatibilityEvent(
            "input_orientation",
            "adapter writes the truth-free cells-by-genes count snapshot as the pinned genes-by-cells CSV contract; upstream transposes it back",
        ),
        CompatibilityEvent(
            "legacy_environment_boundary",
            "a source-verified 30-second subprocess preflight requires Python 3.6/3.7, TensorFlow 1.12.0, Keras 2.2.4, and a TensorFlow-visible GPU; modern or hanging legacy stacks fail closed",
        ),
        CompatibilityEvent(
            "gpu_device_binding",
            f"pinned CLI default GPU_SET=3 is an operational device ordinal, not a scientific hyperparameter; the study binds physical GPU index {config.gpu_index} and the subprocess sees it as logical GPU0",
        ),
        CompatibilityEvent(
            "allocator_policy",
            "both the TensorFlow 1.12 GPU-kernel preflight and the Keras execution session set and receipt gpu_options.allow_growth=true before creating their sessions",
        ),
        CompatibilityEvent(
            "seed_binding",
            "adapter wrapper sets Python, NumPy, and TensorFlow graph seeds before executing the exact pinned script via runpy",
        ),
        CompatibilityEvent(
            "upstream_defaults",
            f"batch_size={config.batch_size}, n_iters_ae={config.autoencoder_iterations}, n_iters_pretrain={config.pretrain_iterations}, alpha={config.zero_loss_weight}, beta=1.0 (hard-coded upstream), dr_rate={config.dropout_rate}, nu1={config.l1_regularization}, nu2={config.l2_regularization}, gene_scale=False, GPU_SET={config.gpu_index}",
        ),
        CompatibilityEvent(
            "upstream_architecture",
            "pinned stacked denoising autoencoder dimensions remain [genes,500,500,2000,10] with weighted MSE/MAE training",
        ),
        CompatibilityEvent(
            "output_convention",
            "pinned save_imputation retains observed log-counts-per-million entries, fills only zeros from decoder output, and serializes four decimal places; adapter does not clip or overwrite values",
        ),
        CompatibilityEvent(
            "upstream_serialization",
            "adapter consumes the integer column header written by pinned pandas.DataFrame.to_csv(index=None); header values are metadata and never matrix data",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "evaluator counts are expm1(native)*observed_library_size/1000000 before the shared log2(CP10k+1) transform",
        ),
    ]
    if config != SCSDaeConfig():
        compatibility.append(
            CompatibilityEvent(
                "upstream_parameter_override",
                "one or more explicit scSDAE execution parameters differ from pinned CLI defaults",
            )
        )
    with TemporaryDirectory(prefix="maskimpute-scsdae-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.csv"
        output_path = work_dir / "output.npy"
        receipt_path = work_dir / "receipt.tsv"
        upstream_output_dir = work_dir / "upstream-output"
        upstream_output_dir.mkdir()
        _write_scsdae_input(input_path, method_input)
        source_script = source_dir / "pure_ae_new.py"
        environment = {
            "CUDA_VISIBLE_DEVICES": str(config.gpu_index),
            "MKL_NUM_THREADS": str(spec.resources.cpu_cores),
            "OMP_NUM_THREADS": str(spec.resources.cpu_cores),
            "PYTHONHASHSEED": str(seed),
        }
        probe_command = (
            str(executable),
            "-B",
            "-I",
            "-c",
            _SCSDAE_PROBE_DRIVER,
            str(source_script.resolve()),
            str(config.gpu_index),
        )
        try:
            probe = execute_pinned_command(
                spec,
                source_dir,
                probe_command,
                cwd=work_dir,
                timeout_seconds=30,
                environment=environment,
            )
        except AdapterUnavailableError as original:
            if original.reason_code == "upstream_timeout":
                error = AdapterUnavailableError(
                    "legacy_gpu_initialization_timeout",
                    "TensorFlow 1.12 could enumerate the modern GPU but did not finish its CUDA initialization within the fixed 30-second preflight",
                    command=original.command,
                    stdout=original.stdout,
                    stderr=original.stderr,
                )
            else:
                error = _legacy_failure_reason(original)
            if _direct:
                raise error from original
            raise SCSDaeUnavailableError(
                error,
                source=source_receipt,
                spec=spec,
                executable=executable,
                probe_command=original.command,
                probe_stdout=original.stdout,
                probe_stderr=original.stderr,
            ) from original
        command = (
            str(executable),
            "-B",
            "-I",
            "-c",
            _SCSDAE_DRIVER,
            str(source_script.resolve()),
            str(input_path),
            str(output_path),
            str(upstream_output_dir),
            str(receipt_path),
            str(seed),
            str(config.batch_size),
            str(config.autoencoder_iterations),
            str(config.pretrain_iterations),
            repr(float(config.zero_loss_weight)),
            repr(float(config.observed_loss_weight)),
            repr(float(config.dropout_rate)),
            repr(float(config.l1_regularization)),
            repr(float(config.l2_regularization)),
            str(config.gpu_index),
        )
        try:
            result = execute_pinned_command(
                spec,
                source_dir,
                command,
                cwd=work_dir,
                timeout_seconds=spec.resources.timeout_seconds,
                environment=environment,
            )
            output = read_npy_output(output_path)
            receipt = read_environment_receipt(
                receipt_path,
                expected_keys=frozenset(
                    {
                        "gpu_available",
                        "gpu_index",
                        "keras_version",
                        "numpy_version",
                        "pandas_version",
                        "python_version",
                        "source_script",
                        "tensorflow_memory_growth",
                        "tensorflow_version",
                    }
                ),
            )
        except AdapterUnavailableError as original:
            classified = _legacy_failure_reason(original)
            error = _combine_probe_and_run_error(
                classified,
                probe.stdout,
                probe.stderr,
            )
            if _direct:
                raise error from original
            raise SCSDaeUnavailableError(
                error,
                source=source_receipt,
                spec=spec,
                executable=executable,
                probe_command=probe_command,
                probe_stdout=probe.stdout,
                probe_stderr=probe.stderr,
                run_command=original.command,
                run_stdout=original.stdout,
                run_stderr=original.stderr,
            ) from original
        try:
            snapshot = (
                finalize_scsdae_direct_output(spec, method_input, output)
                if _direct
                else finalize_scsdae_output(spec, method_input, output)
            )
        except MethodContractError as original:
            classified = AdapterUnavailableError(
                "malformed_upstream_output",
                f"scSDAE output violates the benchmark contract: {original}",
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            error = _combine_probe_and_run_error(
                classified,
                probe.stdout,
                probe.stderr,
            )
            if _direct:
                raise error from original
            raise SCSDaeUnavailableError(
                error,
                source=source_receipt,
                spec=spec,
                executable=executable,
                probe_command=probe_command,
                probe_stdout=probe.stdout,
                probe_stderr=probe.stderr,
                run_command=command,
                run_stdout=result.stdout,
                run_stderr=result.stderr,
            ) from original
        if _direct:
            return DirectAdapterExecution(
                output=snapshot,
                stdout=probe.stdout + b"\n" + result.stdout,
                stderr=probe.stderr + b"\n" + result.stderr,
            )
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=probe.stdout + b"\n" + result.stdout,
            stderr=probe.stderr + b"\n" + result.stderr,
            command=command,
        )


def run_scsdae(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCSDaeConfig = SCSDaeConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    return _run_scsdae_impl(
        spec,
        method_input,
        source_dir=source_dir,
        python_executable=python_executable,
        seed=seed,
        config=config,
        work_root=work_root,
    )


def run_scsdae_direct(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCSDaeConfig = SCSDaeConfig(),
    work_root: Path | None = None,
) -> DirectAdapterExecution:
    return _run_scsdae_impl(
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
    "SCSDaeAttemptReceipt",
    "SCSDaeConfig",
    "SCSDaeUnavailableError",
    "finalize_scsdae_direct_output",
    "finalize_scsdae_output",
    "run_scsdae",
    "run_scsdae_direct",
    "scsdae_to_evaluator_counts",
]
