"""Pinned scCR adapter with a source-backed reconstruction of its omitted graph utility."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np

from .base import MethodInput, MethodOutputSnapshot, MethodSpec, snapshot_method_output
from .direct import DirectAdapterExecution, DirectMethodOutput, finalize_direct_method_output
from .observed import (
    AdapterExecution,
    AdapterUnavailableError,
    CompatibilityEvent,
    execute_pinned_command,
    log1p_cp10k,
    log1p_cp10k_to_count_equivalent,
    read_environment_receipt,
    read_npy_output,
    require_executable,
    require_method_spec,
)


SCCR_GRAPH_CONTRACT_URL = "https://github.com/Junseok0207/scFP.git"
SCCR_GRAPH_CONTRACT_REVISION = "de372f99aa33a7cc4214bd99e0fa4a253652e505"
SCCR_GRAPH_CONTRACT_SHA256 = (
    "fb90fd2409337fb39247fb11ed2076f532566f946bf38103b5f4c6fe9a50cda3"
)


# The pinned scCR tree imports ``misc.graph_construction.knn_graph`` but omits
# the entire ``misc`` directory. Its trainer, embedder, and argument files retain
# scFP's import/call structure. The implementation below is a clean-room
# re-expression checked against fixed behavioral fixtures from that source
# contract. The scFP repository has no identified license: this provenance is
# not a permission claim, and the registry's NOASSERTION limitation still applies.
_SCCR_GRAPH_RECONSTRUCTION = r"""
import torch
import torch.nn.functional as F

def knn_graph(embeddings, k, gcn_norm=False, sym=True):
    unit_vectors = F.normalize(embeddings, dim=1, p=2)
    cosine = torch.mm(unit_vectors, unit_vectors.t())
    selected = torch.topk(cosine, k=int(k) + 1, dim=1).indices
    weights = torch.zeros_like(cosine)
    weights.scatter_(1, selected, cosine.gather(1, selected))
    weights = F.relu(weights) + torch.eye(
        weights.shape[0], dtype=weights.dtype, device=weights.device
    )
    if sym:
        weights = (weights + weights.t()) * 0.5
    degree = weights.sum(dim=1)
    exponent = -0.5 if gcn_norm else -1.0
    inverse_degree = degree.pow(exponent)
    inverse_degree.masked_fill_(~torch.isfinite(inverse_degree), 0.0)
    normalized = inverse_degree[:, None] * weights
    if gcn_norm:
        normalized = normalized * inverse_degree[None, :]
    sparse = normalized.to_sparse()
    return sparse.indices().detach(), sparse.values()
"""


_SCCR_DRIVER = (
    r"""
from pathlib import Path
import importlib
import random
import sys
import types
import numpy as np

if len(sys.argv) != 14:
    raise RuntimeError("adapter expected thirteen arguments")
source_python = Path(sys.argv[1]).resolve(strict=True)
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
receipt_path = Path(sys.argv[4])
seed = int(sys.argv[5])
neighbors = int(sys.argv[6])
gene_neighbors = int(sys.argv[7])
symmetric_final = sys.argv[8] == "true"
iterations = int(sys.argv[9])
complete_weight = float(sys.argv[10])
soft_weight = float(sys.argv[11])
blend_weight = float(sys.argv[12])
device_name = sys.argv[13]

sys.path.insert(0, str(source_python))
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(3)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
if device_name == "auto":
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
if device_name.startswith("cuda:") and not torch.cuda.is_available():
    raise RuntimeError("requested scCR CUDA device is unavailable")
device = torch.device(device_name)

graph_module = types.ModuleType("misc.graph_construction")
graph_module.__dict__["__builtins__"] = __builtins__
exec("""
    + repr(_SCCR_GRAPH_RECONSTRUCTION)
    + r""", graph_module.__dict__)
misc_module = types.ModuleType("misc")
misc_module.__path__ = []
misc_module.graph_construction = graph_module
sys.modules["misc"] = misc_module
sys.modules["misc.graph_construction"] = graph_module

class _TruthFreeEmbedder:
    def evaluate(self):
        return self.adata.obsm["denoised"]

embedder_module = types.ModuleType("embedder")
embedder_module.embedder = _TruthFreeEmbedder
sys.modules["embedder"] = embedder_module

sccr_module = importlib.import_module("scCR")
module_path = Path(sccr_module.__file__).resolve(strict=True)
if source_python not in module_path.parents:
    raise RuntimeError("imported scCR module is not from the pinned checkout")

normalized = np.load(input_path, allow_pickle=False)
if type(normalized) is not np.ndarray or normalized.ndim != 2:
    raise RuntimeError("input normalized matrix is malformed")
if not np.isfinite(normalized).all() or (normalized < 0).any():
    raise RuntimeError("input normalized matrix is invalid")

arguments = types.SimpleNamespace(
    device=device,
    k=neighbors,
    k_col=gene_neighbors,
    sym=symmetric_final,
    iter=iterations,
    alpha=complete_weight,
    beta=soft_weight,
    gamma=blend_weight,
)
trainer = sccr_module.scCR_Trainer.__new__(sccr_module.scCR_Trainer)
trainer.args = arguments
trainer.device = device
trainer.adata = types.SimpleNamespace(obsm={"train": normalized})
output = np.asarray(trainer.train(), dtype=np.float64)
if output.shape != normalized.shape:
    raise RuntimeError("scCR output shape differs from input")
if not np.isfinite(output).all() or (output < 0).any():
    raise RuntimeError("scCR output is not finite and nonnegative")
np.save(output_path, output, allow_pickle=False)
receipt = {
    "device": str(device),
    "graph_contract_revision": """
    + repr(SCCR_GRAPH_CONTRACT_REVISION)
    + r""",
    "graph_contract_sha256": """
    + repr(SCCR_GRAPH_CONTRACT_SHA256)
    + r""",
    "graph_contract_url": """
    + repr(SCCR_GRAPH_CONTRACT_URL)
    + r""",
    "numpy_version": str(np.__version__),
    "python_version": sys.version.split()[0],
    "sccr_module": str(module_path),
    "torch_num_threads": str(torch.get_num_threads()),
    "torch_version": str(torch.__version__),
}
receipt_path.write_text(
    "".join(f"{key}\t{receipt[key]}\n" for key in sorted(receipt)),
    encoding="utf-8",
)
"""
)

_SCCR_DIRECT_DRIVER = "".join(
    line
    for line in _SCCR_DRIVER.splitlines(keepends=True)
    if not line.lstrip().startswith('"graph_contract_')
)


@dataclass(frozen=True, slots=True)
class SCCRConfig:
    """Pinned scCR defaults, excluding the mandatory study seed."""

    neighbors: int = 15
    gene_neighbors: int = 2
    symmetric_final_graph: bool = True
    iterations: int = 40
    complete_relation_weight: float = 0.05
    soft_propagation_weight: float = 0.99
    final_blend_weight: float = 0.01
    device: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("neighbors", self.neighbors),
            ("gene_neighbors", self.gene_neighbors),
            ("iterations", self.iterations),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.symmetric_final_graph) is not bool:
            raise ValueError("symmetric_final_graph must be a boolean")
        for name, value in (
            ("complete_relation_weight", self.complete_relation_weight),
            ("soft_propagation_weight", self.soft_propagation_weight),
            ("final_blend_weight", self.final_blend_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            self.device is not None
            and self.device != "cpu"
            and re.fullmatch(r"cuda:[0-9]+", self.device) is None
        ):
            raise ValueError("device must be null, cpu, or an explicit cuda:<index>")


def reconstructed_sccr_knn_dense(
    embeddings: np.ndarray,
    neighbors: int,
    *,
    symmetric: bool,
) -> np.ndarray:
    """Evaluate the reconstructed graph contract for fixed audit fixtures.

    Runtime scCR uses the Torch reconstruction above. This dense NumPy form is
    intentionally exposed so the source-derived topology and normalization can
    be checked without making Torch a core package dependency. Tied cosine
    similarities are outside the equivalence-fixture contract.
    """

    if type(embeddings) is not np.ndarray or embeddings.ndim != 2:
        raise TypeError("embeddings must be an exact two-dimensional ndarray")
    if embeddings.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("embeddings must be numeric")
    if type(neighbors) is not int or not 0 < neighbors < embeddings.shape[0]:
        raise ValueError("neighbors must be positive and smaller than the row count")
    if type(symmetric) is not bool:
        raise ValueError("symmetric must be a boolean")
    values = np.asarray(embeddings, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1e-12)
    similarity = normalized @ normalized.T
    retained = np.zeros_like(similarity)
    for row in range(similarity.shape[0]):
        order = np.argsort(-similarity[row], kind="stable")
        selected = order[: neighbors + 1]
        retained[row, selected] = similarity[row, selected]
    retained = np.maximum(retained, 0.0)
    retained = retained + np.eye(retained.shape[0], dtype=np.float64)
    if symmetric:
        retained = (retained + retained.T) / 2.0
    degree = retained.sum(axis=1)
    inverse = np.zeros_like(degree)
    np.divide(1.0, degree, out=inverse, where=degree != 0.0)
    adjacency = inverse[:, None] * retained
    adjacency.setflags(write=False)
    return adjacency


def sccr_to_evaluator_counts(
    method_input: MethodInput,
    native_output: object,
) -> np.ndarray:
    """Invert scCR's natural-log CP10k native scale on observed libraries."""

    return log1p_cp10k_to_count_equivalent(method_input, native_output)


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


def finalize_sccr_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> MethodOutputSnapshot:
    """Bind untouched scCR normalized output to evaluator identifiers."""

    require_method_spec(
        spec,
        "sccr",
        input_scale="log1p_cp10k",
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


def finalize_sccr_direct_output(
    spec: MethodSpec,
    method_input: MethodInput,
    normalized_output: object,
) -> DirectMethodOutput:
    """Validate scCR output without deriving a content identity."""

    require_method_spec(
        spec,
        "sccr",
        input_scale="log1p_cp10k",
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


def _run_sccr_impl(
    spec: MethodSpec,
    method_input: MethodInput,
    *,
    source_dir: Path,
    python_executable: Path,
    seed: int,
    config: SCCRConfig = SCCRConfig(),
    work_root: Path | None = None,
    _direct: bool = False,
) -> AdapterExecution | DirectAdapterExecution:
    """Run the exact pinned trainer with a truth-free entrypoint and graph shim."""

    require_method_spec(
        spec,
        "sccr",
        input_scale="log1p_cp10k",
        output_scale="method_native_normalized",
    )
    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if not isinstance(config, SCCRConfig):
        raise TypeError("config must be an SCCRConfig")
    _validate_work_root(work_root, source_dir)
    cells, genes = method_input.shape
    if cells <= config.neighbors:
        raise AdapterUnavailableError(
            "upstream_minimum_dimension",
            f"scCR k={config.neighbors} requires more than {config.neighbors} cells",
        )
    if genes * 2 <= config.gene_neighbors:
        raise AdapterUnavailableError(
            "upstream_minimum_dimension",
            "scCR gene graph requires more concatenated genes than gene_neighbors",
        )
    executable = require_executable(python_executable)
    compatibility = [
        CompatibilityEvent(
            "input_scale_conversion",
            "adapter applies natural log(1 + counts/observed_library*10000), matching the frozen log1p_cp10k input contract",
        ),
        CompatibilityEvent(
            "truth_free_entrypoint",
            "adapter invokes pinned scCR_Trainer.train with the prespecified full gene panel while bypassing upstream embedder evaluation, cell-type labels, artificial dropout, dataset filtering, and HVG reselection",
        ),
        CompatibilityEvent(
            "missing_graph_utility_reconstruction",
            f"pinned scCR omits misc.graph_construction; adapter reconstructs its retained scFP call contract from {SCCR_GRAPH_CONTRACT_URL}@{SCCR_GRAPH_CONTRACT_REVISION}, file sha256={SCCR_GRAPH_CONTRACT_SHA256}",
        ),
        CompatibilityEvent(
            "reconstruction_license_limitation",
            "the graph-contract provenance is not a license grant; no scFP repository license was identified, so redistribution permission remains NOASSERTION",
        ),
        CompatibilityEvent(
            "graph_normalization_contract",
            "source call defaults use nonnegative cosine kNN with self loops, optional symmetrization, and row-degree normalization (gcn_norm=False)",
        ),
        CompatibilityEvent(
            "upstream_defaults",
            f"k={config.neighbors}, k_col={config.gene_neighbors}, sym={config.symmetric_final_graph}, iter={config.iterations}, alpha={config.complete_relation_weight}, beta={config.soft_propagation_weight}, gamma={config.final_blend_weight}, device={'auto' if config.device is None else config.device}",
        ),
        CompatibilityEvent(
            "seed_binding",
            "study seed initializes Python, NumPy, Torch, and CUDA RNGs before importing the pinned trainer",
        ),
        CompatibilityEvent(
            "resource_behavior",
            "the selected executable uses cuda:0 only when its own torch.cuda.is_available() is true and otherwise uses CPU; truth-free direct entrypoint retains pinned main.py torch.set_num_threads(3) within the registry ceiling",
        ),
        CompatibilityEvent(
            "output_convention",
            "adapter retains the full pinned denoised matrix without copying observed entries, clipping, or renormalization",
        ),
        CompatibilityEvent(
            "evaluator_scale_conversion",
            "evaluator counts are expm1(native)*observed_library_size/10000 before the shared log2(CP10k+1) transform",
        ),
    ]
    if config != SCCRConfig():
        compatibility.append(
            CompatibilityEvent(
                "upstream_parameter_override",
                "one or more explicit scCR execution parameters differ from pinned CLI defaults",
            )
        )
    with TemporaryDirectory(prefix="maskimpute-sccr-", dir=work_root) as temporary:
        work_dir = Path(temporary)
        input_path = work_dir / "input.npy"
        output_path = work_dir / "output.npy"
        receipt_path = work_dir / "receipt.tsv"
        np.save(input_path, log1p_cp10k(method_input.counts), allow_pickle=False)
        command = (
            str(executable),
            "-B",
            "-I",
            "-c",
            _SCCR_DIRECT_DRIVER if _direct else _SCCR_DRIVER,
            str(source_dir.resolve()),
            str(input_path),
            str(output_path),
            str(receipt_path),
            str(seed),
            str(config.neighbors),
            str(config.gene_neighbors),
            "true" if config.symmetric_final_graph else "false",
            str(config.iterations),
            repr(float(config.complete_relation_weight)),
            repr(float(config.soft_propagation_weight)),
            repr(float(config.final_blend_weight)),
            "auto" if config.device is None else config.device,
        )
        result = execute_pinned_command(
            spec,
            source_dir,
            command,
            cwd=work_dir,
            timeout_seconds=spec.resources.timeout_seconds,
            environment={
                "MKL_NUM_THREADS": "3",
                "OMP_NUM_THREADS": "3",
            },
        )
        output = read_npy_output(output_path)
        expected_receipt_keys = {
            "device",
            "numpy_version",
            "python_version",
            "sccr_module",
            "torch_num_threads",
            "torch_version",
        }
        if not _direct:
            expected_receipt_keys.update(
                {
                    "graph_contract_revision",
                    "graph_contract_sha256",
                    "graph_contract_url",
                }
            )
        receipt = read_environment_receipt(
            receipt_path,
            expected_keys=frozenset(expected_receipt_keys),
        )
        if _direct:
            return DirectAdapterExecution(
                output=finalize_sccr_direct_output(spec, method_input, output),
                stdout=result.stdout,
                stderr=result.stderr,
            )
        snapshot = finalize_sccr_output(spec, method_input, output)
        return AdapterExecution(
            snapshot=snapshot,
            compatibility_log=tuple(compatibility),
            environment_receipt=receipt,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


def run_sccr(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    python_executable: Path, seed: int, config: SCCRConfig = SCCRConfig(),
    work_root: Path | None = None,
) -> AdapterExecution:
    return _run_sccr_impl(spec, method_input, source_dir=source_dir,
                          python_executable=python_executable, seed=seed,
                          config=config, work_root=work_root)


def run_sccr_direct(
    spec: MethodSpec, method_input: MethodInput, *, source_dir: Path,
    python_executable: Path, seed: int, config: SCCRConfig = SCCRConfig(),
    work_root: Path | None = None,
) -> DirectAdapterExecution:
    return _run_sccr_impl(spec, method_input, source_dir=source_dir,
                          python_executable=python_executable, seed=seed,
                          config=config, work_root=work_root, _direct=True)


__all__ = [
    "SCCR_GRAPH_CONTRACT_REVISION",
    "SCCR_GRAPH_CONTRACT_SHA256",
    "SCCR_GRAPH_CONTRACT_URL",
    "SCCRConfig",
    "finalize_sccr_direct_output",
    "finalize_sccr_output",
    "reconstructed_sccr_knn_dense",
    "run_sccr",
    "run_sccr_direct",
    "sccr_to_evaluator_counts",
]
