"""Pinned D3Impute adapter with an explicit matched-bulk contract."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import inv_boxcox
from scipy.stats import boxcox

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    _validated_native_matrix,
    execute_pinned_command,
    read_environment_receipt,
    read_npy_output,
    require_executable,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


_D3IMPUTE_DRIVER = r"""
from pathlib import Path
import importlib
import sys
import numpy as np
import pandas as pd
import scipy
import sklearn

if len(sys.argv) != 15:
    raise RuntimeError("adapter expected fourteen arguments")
archive = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
bulk_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
receipt_path = Path(sys.argv[5])
fixed_seed = int(sys.argv[6])
neighbors = int(sys.argv[7])
latent_dimension = int(sys.argv[8])
iterations = int(sys.argv[9])
sparsity = float(sys.argv[10])
cell_regularization = float(sys.argv[11])
gene_regularization = float(sys.argv[12])
reference_id = sys.argv[13]
reference_sha256 = sys.argv[14]
source_python = f"{archive}/PYTHON"
sys.path.insert(0, source_python)
transform_module = importlib.import_module("Function.transform")
manifold_module = importlib.import_module("Function.manifit")
inference_module = importlib.import_module("Function.Inference")
for module in (transform_module, manifold_module, inference_module):
    if not str(module.__file__).startswith(source_python + "/"):
        raise RuntimeError("imported D3Impute module is not from the pinned archive")
counts = np.load(input_path, allow_pickle=False)
bulk = np.load(bulk_path, allow_pickle=False)
if type(counts) is not np.ndarray or counts.ndim != 2:
    raise RuntimeError("input count matrix is malformed")
if type(bulk) is not np.ndarray or bulk.ndim != 2 or bulk.shape[0] != counts.shape[1]:
    raise RuntimeError("matched bulk matrix is malformed")
dropout = transform_module.transform(counts, "boxcox")
bulk_transformed = transform_module.transform(bulk, "boxcox")
np.random.seed(fixed_seed)
manifold = manifold_module.manfit_cosine(dropout, knn=neighbors)
options = {
    "k": neighbors,
    "p": latent_dimension,
    "iterate": iterations,
    "beta": sparsity,
    "lamda_c": cell_regularization,
    "lamda_g": gene_regularization,
}
output, _, _ = inference_module.Inference(
    dropout, bulk_transformed, manifold, options
)
output = np.asarray(output, dtype=np.float64)
if output.shape != counts.shape or not np.isfinite(output).all() or (output < 0).any():
    raise RuntimeError("D3Impute output is invalid")
np.save(output_path, output, allow_pickle=False)
receipt = {
    "bulk_reference_id": reference_id,
    "bulk_reference_sha256": reference_sha256,
    "inference_module": str(inference_module.__file__),
    "numpy_version": str(np.__version__),
    "pandas_version": str(pd.__version__),
    "python_version": sys.version.split()[0],
    "scipy_version": str(scipy.__version__),
    "sklearn_version": str(sklearn.__version__),
    "source_archive": str(archive),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""


def _identifiers(values: object, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise TypeError(f"{name} must be a nonempty tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")
    return values


def _bulk_digest(
    reference_id: str,
    source_sha256: str,
    var_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    shape: tuple[int, int],
    matrix_bytes: bytes,
) -> str:
    binding = json.dumps(
        {
            "reference_id": reference_id,
            "source_sha256": source_sha256,
            "var_ids": var_ids,
            "sample_ids": sample_ids,
            "shape": shape,
            "dtype": "<f8",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(binding)
    digest.update(matrix_bytes)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MatchedBulkReference:
    """Immutable gene-by-sample bulk expression aligned to one method input."""

    reference_id: str
    source_sha256: str
    var_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    shape: tuple[int, int]
    matrix_sha256: str
    _matrix_bytes: bytes = field(repr=False)

    @property
    def matrix(self) -> np.ndarray:
        return np.frombuffer(self._matrix_bytes, dtype="<f8").reshape(self.shape)


def prepare_matched_bulk_reference(
    *,
    reference_id: str,
    source_sha256: str,
    matrix: object,
    var_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
) -> MatchedBulkReference:
    """Validate and freeze one prespecified matched-bulk reference."""

    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError("reference_id must be a nonempty string")
    if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256")
    genes = _identifiers(var_ids, "var_ids")
    samples = _identifiers(sample_ids, "sample_ids")
    if len(samples) < 2:
        raise ValueError("matched bulk requires at least two samples")
    if type(matrix) is not np.ndarray or matrix.ndim != 2:
        raise TypeError("matched bulk matrix must be an exact two-dimensional ndarray")
    if matrix.shape != (len(genes), len(samples)):
        raise ValueError("matched bulk matrix shape does not match its IDs")
    if matrix.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("matched bulk matrix must be numeric")
    values = np.array(matrix, dtype="<f8", copy=True, order="C", subok=False)
    if not np.isfinite(values).all() or bool((values < 0).any()):
        raise ValueError("matched bulk matrix must be finite and nonnegative")
    matrix_bytes = values.tobytes(order="C")
    shape = tuple(values.shape)
    return MatchedBulkReference(
        reference_id=reference_id,
        source_sha256=source_sha256,
        var_ids=genes,
        sample_ids=samples,
        shape=shape,
        matrix_sha256=_bulk_digest(
            reference_id,
            source_sha256,
            genes,
            samples,
            shape,
            matrix_bytes,
        ),
        _matrix_bytes=matrix_bytes,
    )


def validate_matched_bulk_reference(
    method_input: MethodInput,
    reference: MatchedBulkReference,
) -> np.ndarray:
    """Validate bulk identity, bytes, and exact gene alignment to method input."""

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if not isinstance(reference, MatchedBulkReference):
        raise TypeError("reference must be a MatchedBulkReference")
    if reference.var_ids != method_input.var_ids:
        raise ValueError("matched bulk gene IDs do not match the method input")
    if reference.shape != (method_input.shape[1], len(reference.sample_ids)):
        raise ValueError("matched bulk shape does not match the method input")
    try:
        matrix = reference.matrix
    except ValueError as error:
        raise ValueError("matched bulk bytes do not match its shape") from error
    matrix_bytes = np.asarray(matrix, dtype="<f8", order="C").tobytes(order="C")
    expected_hash = _bulk_digest(
        reference.reference_id,
        reference.source_sha256,
        reference.var_ids,
        reference.sample_ids,
        reference.shape,
        matrix_bytes,
    )
    if reference.matrix_sha256 != expected_hash:
        raise ValueError("matched bulk matrix hash does not match its bound content")
    return matrix


@dataclass(frozen=True, slots=True)
class D3ImputeConfig:
    """Pinned D3Impute Python defaults plus a fixed RNG compatibility seed."""

    neighbors: int = 23
    latent_dimension: int = 10
    iterations: int = 100
    sparsity: float = 0.001
    cell_regularization: float = 0.1
    gene_regularization: float = 0.1
    fixed_seed: int = 42

    def __post_init__(self) -> None:
        for name, value in (
            ("neighbors", self.neighbors),
            ("latent_dimension", self.latent_dimension),
            ("iterations", self.iterations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("sparsity", self.sparsity),
            ("cell_regularization", self.cell_regularization),
            ("gene_regularization", self.gene_regularization),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.fixed_seed != 42:
            raise ValueError(
                "fixed_seed must remain the disclosed compatibility value 42"
            )


def _require_d3impute_spec(spec: MethodSpec) -> None:
    if not isinstance(spec, MethodSpec):
        raise TypeError("spec must be a MethodSpec")
    if spec.id != "d3impute":
        raise ValueError(f"expected method d3impute, received {spec.id}")
    if spec.track != "external_reference":
        raise ValueError("d3impute must use the external_reference track")
    if spec.input_scale != "raw_counts":
        raise ValueError("d3impute input scale must be raw_counts")
    if spec.output_scale != "external_reference_adjusted":
        raise ValueError("d3impute output scale must be external_reference_adjusted")
    if spec.stochastic or spec.seed_policy != "not_applicable":
        raise ValueError("d3impute must remain deterministic without a seed")


def d3impute_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Invert D3Impute's per-gene Box-Cox scale using observed counts."""

    native = _validated_native_matrix(
        method_input,
        native_output,
        name="native D3Impute Box-Cox output",
    )
    counts = np.asarray(method_input.counts, dtype=np.float64)
    converted = np.empty_like(native)
    for gene in range(counts.shape[1]):
        observed = counts[:, gene]
        if bool(np.all(observed == observed[0])):
            raise AdapterUnavailableError(
                "noninvertible_native_scale",
                f"D3Impute maps constant gene {method_input.var_ids[gene]} to zero, so its native scale cannot be inverted",
            )
        minimum = float(observed.min())
        shift = abs(minimum) + 1.0 if minimum <= 0 else 0.0
        _, exponent = boxcox(observed + shift)
        converted[:, gene] = inv_boxcox(native[:, gene], exponent) - shift
    converted[np.abs(converted) < 1e-12] = 0.0
    if not np.isfinite(converted).all() or bool((converted < 0).any()):
        raise AdapterUnavailableError(
            "noninvertible_native_scale",
            "D3Impute Box-Cox output has no finite nonnegative count equivalent",
        )
    converted.setflags(write=False)
    return converted


def finalize_d3impute_output(
    spec: MethodSpec,
    method_input: MethodInput,
    native_output: object,
) -> MethodOutputSnapshot:
    """Bind untouched upstream Box-Cox output to evaluator IDs."""

    _require_d3impute_spec(spec)
    return snapshot_method_output(
        spec,
        method_input,
        native_output,
        source_dataset_sha256=method_input.source_dataset_sha256,
        output_scale=spec.output_scale,
        obs_ids=method_input.obs_ids,
        var_ids=method_input.var_ids,
    )


def run_d3impute(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    bulk_reference: MatchedBulkReference,
    source_dir: Path,
    python_executable: Path,
    config: D3ImputeConfig = D3ImputeConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    """Run pinned D3Impute from its source zip with matched bulk only."""

    _require_d3impute_spec(spec)
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if not isinstance(config, D3ImputeConfig):
        raise TypeError("config must be a D3ImputeConfig")
    bulk = validate_matched_bulk_reference(method_input, bulk_reference)
    executable = require_executable(python_executable)
    archive = source_dir / "PYTHON.zip"
    compatibility = [
        CompatibilityEvent(
            "external_reference_binding",
            f"matched bulk reference {bulk_reference.reference_id} with SHA-256 {bulk_reference.source_sha256}",
        ),
        CompatibilityEvent(
            "source_archive_execution",
            "adapter imports pristine Python modules directly from pinned PYTHON.zip without extraction or source edits",
        ),
        CompatibilityEvent(
            "upstream_parameters",
            f"knn={config.neighbors}, p={config.latent_dimension}, iterations={config.iterations}, "
            f"beta={config.sparsity}, lambda_cell={config.cell_regularization}, "
            f"lambda_gene={config.gene_regularization}",
        ),
        CompatibilityEvent(
            "fixed_rng_compatibility",
            "upstream GRNMF uses unseeded NumPy initialization although the frozen registry declares deterministic; adapter fixes NumPy seed=42 for reproducibility",
        ),
        CompatibilityEvent(
            "evaluation_label_exclusion",
            "upstream clustering/ARI/NMI step is omitted because labels are evaluator-only and not part of imputation",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "native per-gene Box-Cox output is inverted with parameters recomputed from truth-free observed counts; constant genes fail closed; count equivalents then use the shared log2(CP10k+1) transform",
        ),
    ]
    if config.iterations != 100:
        compatibility.append(
            CompatibilityEvent(
                "upstream_iteration_override", f"iterations={config.iterations}"
            )
        )
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="maskimpute-d3impute-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.npy"
        bulk_path = work_dir / "bulk.npy"
        output_path = work_dir / "output.npy"
        receipt_path = work_dir / "receipt.tsv"
        np.save(input_path, method_input.counts, allow_pickle=False)
        np.save(bulk_path, bulk, allow_pickle=False)
        command = (
            str(executable),
            "-B",
            "-I",
            "-c",
            _D3IMPUTE_DRIVER,
            str(archive.resolve()),
            str(input_path),
            str(bulk_path),
            str(output_path),
            str(receipt_path),
            str(config.fixed_seed),
            str(config.neighbors),
            str(config.latent_dimension),
            str(config.iterations),
            repr(float(config.sparsity)),
            repr(float(config.cell_regularization)),
            repr(float(config.gene_regularization)),
            bulk_reference.reference_id,
            bulk_reference.source_sha256,
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
                    "bulk_reference_id",
                    "bulk_reference_sha256",
                    "inference_module",
                    "numpy_version",
                    "pandas_version",
                    "python_version",
                    "scipy_version",
                    "sklearn_version",
                    "source_archive",
                }
            ),
        )
        snapshot = finalize_d3impute_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


__all__ = [
    "D3ImputeConfig",
    "MatchedBulkReference",
    "d3impute_to_evaluator_counts",
    "finalize_d3impute_output",
    "prepare_matched_bulk_reference",
    "run_d3impute",
    "validate_matched_bulk_reference",
]
