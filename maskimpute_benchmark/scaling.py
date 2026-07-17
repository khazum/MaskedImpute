"""Frozen, bounded resource-scaling panel for the publication study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .methods.registry import MethodRegistry, load_method_registry
from .protocol import DevelopmentProtocol, Protocol, canonical_sha256, load_protocol
from .runner import (
    AdapterOutcome,
    AuthorizedConfiguration,
    DatasetBinding,
    ExecutionEnvironmentRegistry,
    ExecutionRequest,
    LongFormMetric,
    PreparedDataset,
    RawRunResult,
    RepositoryAdapterDispatcher,
    RunPlanEntry,
    RunnerAuthority,
    SpawnedRepositoryExecutor,
    enforce_calibration_fold_receipt,
    implementation_source_sha256,
    load_runner_authority,
    prepare_dataset_for_execution,
)
from .simulators import SimulationArtifact, SimulationRequest, run_symsim_pair


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CELL_COUNTS = (10_000, 25_000, 50_000, 100_000)
_METHOD_IDS = ("observed", "maskimpute", "dca", "scvi", "magic")
_ARTIFACT_POLICY = {
    "evaluator_output_retention": "metrics_only_after_hashing",
    "generated_dataset_retention": "moderate_h5ad",
    "paired_control_retention": "discard_after_semantic_and_file_hashing",
    "native_simulator_output_retention": "discard_after_manifest_hashing",
}
_CONTRACT_KEYS = {
    "schema_version",
    "role",
    "mechanism",
    "technical_view",
    "cell_counts",
    "accuracy_cell_counts",
    "accuracy_metrics",
    "excluded_metric_families",
    "genes",
    "method_ids",
    "model_seed",
    "seed_derivation",
    "artifact_policy",
}
_MAX_LOG_BYTES = 64 * 1024 * 1024
_CHECKPOINT_KEYS = {
    "schema_version",
    "plan_sha256",
    "input_hashes",
    "planned_run_count",
    "status",
    "datasets",
    "records",
    "checkpoint_sha256",
}
_SCALING_ACCURACY_METRICS = (
    "mse",
    "mse_dropout",
    "mse_pre_dropout_zero",
    "mse_nonzero",
    "gnrmse",
    "mean_distortion",
    "variance_distortion",
    "mean_gene_wasserstein_distance",
    "corr_err",
    "n_corr_genes",
)
_DATASET_RECEIPT_KEYS = {
    "schema_version",
    "cells",
    "genes",
    "namespace",
    "mechanism",
    "technical_view",
    "dataset_id",
    "independent_unit_id",
    "dataset_sha256",
    "truth_sha256",
    "moderate_output_path",
    "moderate_output_file_sha256",
    "moderate_output_size_bytes",
    "severe_dataset_sha256",
    "severe_output_file_sha256",
    "severe_output_size_bytes",
    "moderate_native_manifest_sha256",
    "severe_native_manifest_sha256",
    "native_files_sha256",
    "protocol_sha256",
    "design_sha256",
    "seed_source_sha256",
    "seeds",
    "severe_retention",
    "native_retention",
    "receipt_sha256",
}


class ScalingContractError(ValueError):
    """Raised when the scaling authority, plan, or evidence is not closed."""


@dataclass(frozen=True, slots=True)
class ScalingEvaluatedAttempt:
    """Bounded scaling result without main-study score-matrix evidence."""

    run: RawRunResult
    metrics: tuple[LongFormMetric, ...]
    stdout: bytes
    stderr: bytes
    native_output: np.ndarray | None
    native_output_scale: str | None
    evaluator_output: np.ndarray | None


@dataclass(frozen=True, slots=True)
class ScalingContract:
    """Exact tracked design of the post-freeze resource-scaling panel."""

    schema_version: int
    role: str
    mechanism: str
    technical_view: str
    cell_counts: tuple[int, ...]
    accuracy_cell_counts: tuple[int, ...]
    accuracy_metrics: tuple[str, ...]
    excluded_metric_families: Mapping[str, str]
    genes: int
    method_ids: tuple[str, ...]
    model_seed: int
    seed_algorithm: str
    seed_master: int
    artifact_policy: Mapping[str, str]
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ScalingSeeds:
    """One independent truth seed and two paired measurement seeds."""

    biological: int
    moderate: int
    severe: int


@dataclass(frozen=True, slots=True)
class ScalingPlanEntry:
    """One required method-by-size scaling attempt."""

    ordinal: int
    run_id: str
    cells: int
    genes: int
    method_id: str
    model_seed: int | None
    configuration_id: str
    configuration_sha256: str
    configuration_kind: str
    requires_count_score: bool
    requires_calibration: bool
    accuracy_enabled: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScalingPlan:
    """Hash-bound complete denominator for the resource-scaling panel."""

    schema_version: int
    input_hashes: Mapping[str, str]
    entries: tuple[ScalingPlanEntry, ...]
    configurations: tuple[AuthorizedConfiguration, ...]
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class ScalingCheckpoint:
    """Validated resumable prefix of the fixed scaling denominator."""

    schema_version: int
    plan_sha256: str
    input_hashes: Mapping[str, str]
    planned_run_count: int
    status: Literal["running", "completed"]
    datasets: tuple[Mapping[str, object], ...]
    records: tuple[Mapping[str, object], ...]
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ScalingExecutionAuthority:
    """All validated production authority needed to execute the scaling plan."""

    repository: Path
    contract: ScalingContract
    protocol: Protocol
    frozen_method: Mapping[str, object]
    registry: MethodRegistry
    runner_authority: RunnerAuthority
    environments: ExecutionEnvironmentRegistry
    plan: ScalingPlan


def _reject_constant(value: str) -> None:
    raise ScalingContractError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScalingContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ScalingContractError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ScalingContractError("scaling evidence is not canonical JSON") from error


def _safe_relative_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ScalingContractError(f"{name} is not a safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ScalingContractError(f"{name} is not a safe relative path")
    return path


def _atomic_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_immutable(path: Path, raw: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ScalingContractError(
                f"refusing to replace scaling artifact {path.name}"
            )
        return hashlib.sha256(raw).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ScalingContractError(
                    f"conflicting scaling artifact appeared at {path.name}"
                )
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def load_scaling_contract(path: Path) -> ScalingContract:
    """Load the sole tracked scaling design without accepting design overrides."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except ScalingContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScalingContractError(
            f"scaling contract cannot be read: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _CONTRACT_KEYS:
        raise ScalingContractError("scaling contract fields are not closed")
    if payload.get("schema_version") != 1 or type(payload["schema_version"]) is not int:
        raise ScalingContractError("scaling contract schema_version must be 1")
    if payload.get("role") != "post_freeze_resource_scaling":
        raise ScalingContractError("scaling role is invalid")
    if (
        payload.get("mechanism") != "symsim"
        or payload.get("technical_view") != "moderate"
    ):
        raise ScalingContractError("scaling mechanism/view is not prespecified")
    raw_cells = payload.get("cell_counts")
    raw_accuracy = payload.get("accuracy_cell_counts")
    if raw_cells != list(_CELL_COUNTS):
        raise ScalingContractError("scaling cell-count grid is invalid")
    if raw_accuracy != list(_CELL_COUNTS):
        raise ScalingContractError("scaling accuracy cell-count grid is invalid")
    if payload.get("accuracy_metrics") != list(_SCALING_ACCURACY_METRICS):
        raise ScalingContractError("scaling accuracy metric set is invalid")
    excluded = payload.get("excluded_metric_families")
    if excluded != {
        "cell_cell_correlation_and_distance": (
            "quadratic_cell_pair_metric_not_scalable"
        ),
        "p_pre_zero_score_evidence": (
            "evaluated_in_main_final_panel_not_retained_in_scaling_panel"
        ),
    }:
        raise ScalingContractError("scaling excluded metric families are invalid")
    if payload.get("genes") != 500 or type(payload["genes"]) is not int:
        raise ScalingContractError("scaling gene count must be exactly 500")
    if payload.get("method_ids") != list(_METHOD_IDS):
        raise ScalingContractError("scaling method denominator is invalid")
    if payload.get("model_seed") != 42 or type(payload["model_seed"]) is not int:
        raise ScalingContractError("scaling model seed must be 42")
    seed = payload.get("seed_derivation")
    if (
        not isinstance(seed, dict)
        or set(seed) != {"algorithm", "master_seed"}
        or seed.get("algorithm") != "sha256-domain-separated-63bit-v1"
        or type(seed.get("master_seed")) is not int
        or not 0 < seed["master_seed"] < 2**63
    ):
        raise ScalingContractError("scaling seed derivation is invalid")
    policy = payload.get("artifact_policy")
    if policy != _ARTIFACT_POLICY:
        raise ScalingContractError("scaling artifact policy is invalid")
    contract = ScalingContract(
        schema_version=1,
        role="post_freeze_resource_scaling",
        mechanism="symsim",
        technical_view="moderate",
        cell_counts=_CELL_COUNTS,
        accuracy_cell_counts=_CELL_COUNTS,
        accuracy_metrics=_SCALING_ACCURACY_METRICS,
        excluded_metric_families=MappingProxyType(dict(excluded)),
        genes=500,
        method_ids=_METHOD_IDS,
        model_seed=42,
        seed_algorithm="sha256-domain-separated-63bit-v1",
        seed_master=seed["master_seed"],
        artifact_policy=MappingProxyType(dict(_ARTIFACT_POLICY)),
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )
    derived = [
        value
        for cells in contract.cell_counts
        for value in asdict(derive_scaling_seeds(contract, cells)).values()
    ]
    if len(derived) != len(set(derived)):
        raise ScalingContractError("scaling seed derivation contains a collision")
    return contract


def _derived_seed(contract: ScalingContract, cells: int, role: str) -> int:
    payload = json.dumps(
        {
            "schema": "maskimpute-scaling-seed-v1",
            "algorithm": contract.seed_algorithm,
            "master_seed": contract.seed_master,
            "mechanism": contract.mechanism,
            "cells": cells,
            "role": role,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer % (2**63 - 1) + 1


def derive_scaling_seeds(contract: ScalingContract, cells: int) -> ScalingSeeds:
    """Derive the fixed, domain-separated seeds for one panel size."""

    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if cells not in contract.cell_counts or type(cells) is not int:
        raise ScalingContractError("cells is outside the scaling grid")
    return ScalingSeeds(
        biological=_derived_seed(contract, cells, "biological"),
        moderate=_derived_seed(contract, cells, "measurement-moderate"),
        severe=_derived_seed(contract, cells, "measurement-severe"),
    )


def scaling_protocol(base: Protocol, contract: ScalingContract, cells: int) -> Protocol:
    """Create an ephemeral development namespace accepted by the SymSim adapter."""

    if not isinstance(base, Protocol):
        raise TypeError("base must be a Protocol")
    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if cells not in contract.cell_counts or type(cells) is not int:
        raise ScalingContractError("cells is outside the scaling grid")
    development = DevelopmentProtocol(
        namespace=f"scaling-{cells}",
        draws_per_condition=1,
        cells=cells,
        genes=contract.genes,
    )
    return replace(base, development=development)


def scaling_requests(
    contract: ScalingContract, protocol: Protocol, output_root: Path
) -> tuple[SimulationRequest, SimulationRequest]:
    """Build the exact moderate/severe pair needed for one scaling truth draw."""

    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    cells = protocol.development.cells
    if (
        cells not in contract.cell_counts
        or protocol.development.namespace != f"scaling-{cells}"
        or protocol.development.draws_per_condition != 1
        or protocol.development.genes != contract.genes
    ):
        raise ScalingContractError("ephemeral scaling protocol is invalid")
    seeds = derive_scaling_seeds(contract, cells)
    parent = output_root / protocol.development.namespace / "dataset"
    values = tuple(
        SimulationRequest(
            mechanism=contract.mechanism,
            namespace=protocol.development.namespace,
            biological_id="draw-01",
            biological_seed=seeds.biological,
            measurement_seed=getattr(seeds, view),
            technical_view=view,
            cells=cells,
            genes=contract.genes,
            output_path=parent / f"{view}.h5ad",
        )
        for view in ("moderate", "severe")
    )
    return values[0], values[1]


def build_scaling_plan(
    contract: ScalingContract,
    registry: MethodRegistry,
    configurations: Sequence[AuthorizedConfiguration],
    *,
    frozen_method_sha256: str,
    method_registry_file_sha256: str,
    execution_authority_sha256: str,
    execution_environment_sha256: str,
    implementation_source_sha256: str,
) -> ScalingPlan:
    """Close the fixed five-method by four-size scaling denominator."""

    if not isinstance(contract, ScalingContract):
        raise TypeError("contract must be a ScalingContract")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    values = tuple(configurations)
    if (
        len(values) != len(contract.method_ids)
        or not all(isinstance(value, AuthorizedConfiguration) for value in values)
        or tuple(value.method_id for value in values) != contract.method_ids
    ):
        raise ScalingContractError("scaling configurations differ from method order")
    for method_id in contract.method_ids:
        try:
            spec = registry.by_id(method_id)
        except KeyError as error:
            raise ScalingContractError(
                f"scaling method {method_id} is absent from the registry"
            ) from error
        if not spec.executable:
            raise ScalingContractError(f"scaling method {method_id} is not executable")
    candidate = values[contract.method_ids.index("maskimpute")]
    if (
        candidate.kind != "candidate_search"
        or candidate.configuration_id == "registry-default"
        or not candidate.requires_count_score
    ):
        raise ScalingContractError("scaling MaskImpute is not the frozen candidate")
    input_hashes = {
        "scaling_contract_sha256": _sha256(
            contract.file_sha256, "scaling contract checksum"
        ),
        "frozen_method_sha256": _sha256(frozen_method_sha256, "frozen method checksum"),
        "method_registry_file_sha256": _sha256(
            method_registry_file_sha256, "method registry file checksum"
        ),
        "execution_authority_sha256": _sha256(
            execution_authority_sha256, "execution authority checksum"
        ),
        "execution_environment_sha256": _sha256(
            execution_environment_sha256, "execution environment checksum"
        ),
        "implementation_source_sha256": _sha256(
            implementation_source_sha256, "implementation source checksum"
        ),
    }
    entries: list[ScalingPlanEntry] = []
    ordinal = 0
    by_method = {value.method_id: value for value in values}
    for cells in contract.cell_counts:
        for method_id in contract.method_ids:
            ordinal += 1
            spec = registry.by_id(method_id)
            configuration = by_method[method_id]
            seed = contract.model_seed if spec.stochastic else None
            seed_token = "deterministic" if seed is None else f"seed-{seed}"
            entries.append(
                ScalingPlanEntry(
                    ordinal=ordinal,
                    run_id=(
                        f"scaling-{method_id}-{cells}-{seed_token}-"
                        f"{configuration.configuration_sha256[:12]}"
                    ),
                    cells=cells,
                    genes=contract.genes,
                    method_id=method_id,
                    model_seed=seed,
                    configuration_id=configuration.configuration_id,
                    configuration_sha256=configuration.configuration_sha256,
                    configuration_kind=configuration.kind,
                    requires_count_score=configuration.requires_count_score,
                    requires_calibration=configuration.requires_calibration,
                    accuracy_enabled=cells in contract.accuracy_cell_counts,
                )
            )
    body = {
        "schema_version": 1,
        "input_hashes": input_hashes,
        "entries": [entry.to_dict() for entry in entries],
        "configurations": [configuration.to_dict() for configuration in values],
    }
    return ScalingPlan(
        schema_version=1,
        input_hashes=MappingProxyType(input_hashes),
        entries=tuple(entries),
        configurations=values,
        plan_sha256=canonical_sha256(body),
    )


def scaling_attempt_record(
    attempt: ScalingEvaluatedAttempt, *, cells: int, accuracy_enabled: bool
) -> dict[str, object]:
    """Retain complete metrics/resources/logs while dropping both dense matrices."""

    if not isinstance(attempt, ScalingEvaluatedAttempt):
        raise TypeError("attempt must be a ScalingEvaluatedAttempt")
    if type(cells) is not int or cells <= 0:
        raise ValueError("cells must be a positive integer")
    if type(accuracy_enabled) is not bool:
        raise TypeError("accuracy_enabled must be bool")
    run = asdict(attempt.run)
    run.update(
        {
            "cells": cells,
            "accuracy_enabled": accuracy_enabled,
            "native_output_retention": (
                "not_available"
                if attempt.run.native_output_sha256 is None
                else "hash_only"
            ),
            "evaluator_output_retention": (
                "not_available"
                if attempt.run.evaluator_output_sha256 is None
                else "hash_only"
            ),
        }
    )
    return {
        "run": run,
        "metrics": [metric.to_dict() for metric in attempt.metrics],
        "stdout": attempt.stdout,
        "stderr": attempt.stderr,
    }


class ScalingResultStore:
    """Canonical checkpoint with exact logs and no persisted dense method outputs."""

    def __init__(self, output_dir: Path, plan: ScalingPlan) -> None:
        if not isinstance(output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        if not isinstance(plan, ScalingPlan):
            raise TypeError("plan must be a ScalingPlan")
        selected = Path(os.path.abspath(output_dir))
        selected.mkdir(parents=True, exist_ok=True)
        for component in (selected, *selected.parents):
            try:
                metadata = component.lstat()
            except OSError as error:
                raise ScalingContractError(
                    "scaling output root is unavailable"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise ScalingContractError(
                    "scaling output root must not contain symlink components"
                )
        try:
            resolved = selected.resolve(strict=True)
        except OSError as error:
            raise ScalingContractError("scaling output root is unavailable") from error
        if resolved != selected or not resolved.is_dir():
            raise ScalingContractError("scaling output root is not canonical")
        self.output_dir = resolved
        self.plan = plan

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint.json"

    def _artifact_path(self, value: object, name: str) -> Path:
        relative = _safe_relative_path(value, name)
        path = (self.output_dir / relative).absolute()
        try:
            path.relative_to(self.output_dir)
        except ValueError as error:
            raise ScalingContractError(f"{name} escaped the output root") from error
        return path

    def _verify_artifact(
        self, relative: object, digest: object, nbytes: object, name: str
    ) -> None:
        expected = _sha256(digest, f"{name} checksum")
        if type(nbytes) is not int or nbytes < 0 or nbytes > _MAX_LOG_BYTES:
            raise ScalingContractError(f"{name} byte count is invalid")
        path = self._artifact_path(relative, name)
        try:
            metadata = path.lstat()
        except OSError as error:
            raise ScalingContractError(f"{name} is missing") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != nbytes
            or _file_sha256(path) != expected
        ):
            raise ScalingContractError(f"{name} integrity check failed")

    def _validate_dataset_receipt(
        self, value: object, expected_cells: int
    ) -> Mapping[str, object]:
        if not isinstance(value, dict):
            raise ScalingContractError("scaling dataset receipt must be an object")
        if set(value) != _DATASET_RECEIPT_KEYS:
            raise ScalingContractError("scaling dataset receipt fields are not closed")
        unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
        if (
            value.get("schema_version") != 1
            or value.get("cells") != expected_cells
            or value.get("genes") != 500
            or value.get("namespace") != f"scaling-{expected_cells}"
            or value.get("mechanism") != "symsim"
            or value.get("technical_view") != "moderate"
            or value.get("receipt_sha256") != canonical_sha256(unsigned)
        ):
            raise ScalingContractError("scaling dataset receipt binding is invalid")
        for field in (
            "dataset_sha256",
            "truth_sha256",
            "moderate_output_file_sha256",
            "severe_dataset_sha256",
            "severe_output_file_sha256",
            "moderate_native_manifest_sha256",
            "severe_native_manifest_sha256",
            "native_files_sha256",
            "protocol_sha256",
            "design_sha256",
            "seed_source_sha256",
        ):
            _sha256(value.get(field), f"scaling dataset {field}")
        if (
            not isinstance(value.get("dataset_id"), str)
            or not value["dataset_id"].startswith("dataset-")
            or not isinstance(value.get("independent_unit_id"), str)
            or type(value.get("moderate_output_size_bytes")) is not int
            or value["moderate_output_size_bytes"] <= 0
            or type(value.get("severe_output_size_bytes")) is not int
            or value["severe_output_size_bytes"] <= 0
            or value.get("severe_retention") != "discarded_after_receipt"
            or value.get("native_retention") != "discarded_after_receipt"
        ):
            raise ScalingContractError("scaling dataset receipt fields are invalid")
        seeds = value.get("seeds")
        if (
            not isinstance(seeds, dict)
            or set(seeds) != {"biological", "moderate", "severe"}
            or any(
                type(item) is not int or not 0 < item < 2**63 for item in seeds.values()
            )
            or len(set(seeds.values())) != 3
        ):
            raise ScalingContractError("scaling dataset receipt seeds are invalid")
        output_path = self._artifact_path(
            value.get("moderate_output_path"), "moderate scaling dataset"
        )
        try:
            metadata = output_path.lstat()
        except OSError as error:
            raise ScalingContractError("moderate scaling dataset is missing") from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != value["moderate_output_size_bytes"]
            or _file_sha256(output_path) != value["moderate_output_file_sha256"]
        ):
            raise ScalingContractError("moderate scaling dataset integrity failed")
        return MappingProxyType(dict(value))

    def _validate_record(
        self, value: object, entry: ScalingPlanEntry
    ) -> Mapping[str, object]:
        if not isinstance(value, dict) or set(value) != {
            "run",
            "metrics",
            "record_sha256",
        }:
            raise ScalingContractError("stored scaling record has wrong fields")
        unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
        if value.get("record_sha256") != canonical_sha256(unsigned):
            raise ScalingContractError("stored scaling record checksum mismatch")
        run = value.get("run")
        metrics = value.get("metrics")
        if not isinstance(run, dict) or not isinstance(metrics, list):
            raise ScalingContractError("stored scaling record is malformed")
        from .runner import LongFormMetric, RawRunResult

        expected_run_fields = {field.name for field in fields(RawRunResult)} | {
            "cells",
            "accuracy_enabled",
            "native_output_retention",
            "evaluator_output_retention",
            "stdout_path",
            "stdout_file_sha256",
            "stdout_size_bytes",
            "stderr_path",
            "stderr_file_sha256",
            "stderr_size_bytes",
        }
        metric_fields = {field.name for field in fields(LongFormMetric)}
        if set(run) != expected_run_fields or any(
            not isinstance(metric, dict) or set(metric) != metric_fields
            for metric in metrics
        ):
            raise ScalingContractError(
                "stored scaling run/metric fields are not closed"
            )
        expected = {
            "run_id": entry.run_id,
            "method_id": entry.method_id,
            "model_seed": entry.model_seed,
            "configuration_id": entry.configuration_id,
            "configuration_sha256": entry.configuration_sha256,
            "configuration_kind": entry.configuration_kind,
            "requires_count_score": entry.requires_count_score,
            "requires_calibration": entry.requires_calibration,
            "cells": entry.cells,
            "accuracy_enabled": entry.accuracy_enabled,
        }
        if any(
            run.get(name) != expected_value for name, expected_value in expected.items()
        ):
            raise ScalingContractError("stored scaling record differs from its plan")
        if run.get("native_output_retention") not in {"hash_only", "not_available"}:
            raise ScalingContractError("native scaling output retention is invalid")
        if run.get("evaluator_output_retention") not in {
            "hash_only",
            "not_available",
        }:
            raise ScalingContractError("evaluator scaling output retention is invalid")
        self._verify_artifact(
            run.get("stdout_path"),
            run.get("stdout_file_sha256"),
            run.get("stdout_size_bytes"),
            "scaling stdout",
        )
        self._verify_artifact(
            run.get("stderr_path"),
            run.get("stderr_file_sha256"),
            run.get("stderr_size_bytes"),
            "scaling stderr",
        )
        if (
            len(metrics) != len(_SCALING_ACCURACY_METRICS)
            or tuple(
                metric.get("metric") if isinstance(metric, dict) else None
                for metric in metrics
            )
            != _SCALING_ACCURACY_METRICS
            or any(
                not isinstance(metric, dict)
                or metric.get("method") != entry.method_id
                or metric.get("model_seed") != entry.model_seed
                or metric.get("configuration_sha256") != entry.configuration_sha256
                for metric in metrics
            )
        ):
            raise ScalingContractError("scaling metric denominator is incomplete")
        return MappingProxyType(dict(value))

    def load(self) -> ScalingCheckpoint | None:
        if not self.checkpoint_path.exists():
            return None
        try:
            raw = self.checkpoint_path.read_bytes()
            payload = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except ScalingContractError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScalingContractError(
                f"scaling checkpoint cannot be read: {error}"
            ) from error
        if (
            not isinstance(payload, dict)
            or set(payload) != _CHECKPOINT_KEYS
            or raw != _canonical_bytes(payload) + b"\n"
            or payload.get("schema_version") != 1
            or payload.get("plan_sha256") != self.plan.plan_sha256
            or payload.get("input_hashes") != dict(self.plan.input_hashes)
            or payload.get("planned_run_count") != len(self.plan.entries)
        ):
            raise ScalingContractError("scaling checkpoint authority mismatch")
        expected_digest = canonical_sha256(
            {key: item for key, item in payload.items() if key != "checkpoint_sha256"}
        )
        if payload.get("checkpoint_sha256") != expected_digest:
            raise ScalingContractError("scaling checkpoint checksum mismatch")
        datasets = payload.get("datasets")
        records = payload.get("records")
        if not isinstance(datasets, list) or not isinstance(records, list):
            raise ScalingContractError("scaling checkpoint arrays are invalid")
        expected_cells = tuple(
            dict.fromkeys(entry.cells for entry in self.plan.entries)
        )
        if len(datasets) > len(expected_cells) or len(records) > len(self.plan.entries):
            raise ScalingContractError("scaling checkpoint exceeds its denominator")
        dataset_values = tuple(
            self._validate_dataset_receipt(value, cells)
            for value, cells in zip(datasets, expected_cells, strict=False)
        )
        record_values = tuple(
            self._validate_record(value, entry)
            for value, entry in zip(records, self.plan.entries, strict=False)
        )
        if record_values:
            maximum_size_index = expected_cells.index(
                self.plan.entries[len(record_values) - 1].cells
            )
            if len(dataset_values) <= maximum_size_index:
                raise ScalingContractError(
                    "scaling records exist without their dataset receipt"
                )
        expected_status = (
            "completed"
            if len(record_values) == len(self.plan.entries)
            and len(dataset_values) == len(expected_cells)
            else "running"
        )
        if payload.get("status") != expected_status:
            raise ScalingContractError("scaling checkpoint status is inconsistent")
        return ScalingCheckpoint(
            schema_version=1,
            plan_sha256=self.plan.plan_sha256,
            input_hashes=MappingProxyType(dict(self.plan.input_hashes)),
            planned_run_count=len(self.plan.entries),
            status=expected_status,
            datasets=dataset_values,
            records=record_values,
            checkpoint_sha256=expected_digest,
        )

    def _write(
        self,
        datasets: Sequence[Mapping[str, object]],
        records: Sequence[Mapping[str, object]],
    ) -> ScalingCheckpoint:
        expected_cells = tuple(
            dict.fromkeys(entry.cells for entry in self.plan.entries)
        )
        status = (
            "completed"
            if len(records) == len(self.plan.entries)
            and len(datasets) == len(expected_cells)
            else "running"
        )
        body: dict[str, object] = {
            "schema_version": 1,
            "plan_sha256": self.plan.plan_sha256,
            "input_hashes": dict(self.plan.input_hashes),
            "planned_run_count": len(self.plan.entries),
            "status": status,
            "datasets": [dict(value) for value in datasets],
            "records": [dict(value) for value in records],
        }
        body["checkpoint_sha256"] = canonical_sha256(body)
        _atomic_replace(self.checkpoint_path, _canonical_bytes(body) + b"\n")
        loaded = self.load()
        if loaded is None:  # pragma: no cover - atomic write invariant
            raise ScalingContractError("scaling checkpoint disappeared after writing")
        return loaded

    def append_dataset(self, receipt: Mapping[str, object]) -> ScalingCheckpoint:
        report = self.load()
        datasets = [] if report is None else list(report.datasets)
        records = [] if report is None else list(report.records)
        expected_cells = tuple(
            dict.fromkeys(entry.cells for entry in self.plan.entries)
        )
        if len(datasets) >= len(expected_cells):
            raise ScalingContractError("all scaling dataset receipts already exist")
        validated = self._validate_dataset_receipt(
            dict(receipt), expected_cells[len(datasets)]
        )
        datasets.append(validated)
        return self._write(datasets, records)

    def append_attempt(
        self, entry: ScalingPlanEntry, record: Mapping[str, object]
    ) -> ScalingCheckpoint:
        if not isinstance(entry, ScalingPlanEntry):
            raise TypeError("entry must be a ScalingPlanEntry")
        report = self.load()
        datasets = [] if report is None else list(report.datasets)
        records = [] if report is None else list(report.records)
        if (
            len(records) >= len(self.plan.entries)
            or self.plan.entries[len(records)] != entry
        ):
            raise ScalingContractError(
                "scaling attempts must follow the exact plan prefix"
            )
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        value = dict(record)
        stdout = value.pop("stdout", None)
        stderr = value.pop("stderr", None)
        if type(stdout) is not bytes or type(stderr) is not bytes:
            raise ScalingContractError("scaling logs must be exact bytes")
        if len(stdout) > _MAX_LOG_BYTES or len(stderr) > _MAX_LOG_BYTES:
            raise ScalingContractError("scaling log exceeds its 64 MiB bound")
        run = value.get("run")
        if not isinstance(run, dict):
            raise ScalingContractError("scaling attempt run is invalid")
        if (
            run.get("stdout_sha256") != hashlib.sha256(stdout).hexdigest()
            or run.get("stderr_sha256") != hashlib.sha256(stderr).hexdigest()
        ):
            raise ScalingContractError("scaling log content hash mismatch")
        base = f"runs/{entry.run_id}"
        stdout_relative = f"{base}.stdout"
        stderr_relative = f"{base}.stderr"
        stdout_digest = _publish_immutable(self.output_dir / stdout_relative, stdout)
        stderr_digest = _publish_immutable(self.output_dir / stderr_relative, stderr)
        stored_run = dict(run)
        stored_run.update(
            {
                "stdout_path": stdout_relative,
                "stdout_file_sha256": stdout_digest,
                "stdout_size_bytes": len(stdout),
                "stderr_path": stderr_relative,
                "stderr_file_sha256": stderr_digest,
                "stderr_size_bytes": len(stderr),
            }
        )
        unsigned = {"run": stored_run, "metrics": value.get("metrics")}
        stored = {**unsigned, "record_sha256": canonical_sha256(unsigned)}
        self._validate_record(stored, entry)
        records.append(stored)
        return self._write(datasets, records)


def _dataset_receipt_from_artifacts(
    contract: ScalingContract,
    protocol: Protocol,
    output_dir: Path,
    artifacts: Sequence[SimulationArtifact],
) -> tuple[dict[str, object], Any]:
    values = tuple(artifacts)
    if len(values) != 2 or not all(
        isinstance(value, SimulationArtifact) for value in values
    ):
        raise ScalingContractError(
            "SymSim scaling did not return exactly two artifacts"
        )
    by_view = {value.request.technical_view: value for value in values}
    if set(by_view) != {"moderate", "severe"}:
        raise ScalingContractError("SymSim scaling artifact views are incomplete")
    moderate_artifact = by_view["moderate"]
    severe_artifact = by_view["severe"]
    moderate = moderate_artifact.adata
    severe = severe_artifact.adata
    from .datasets import _truth_sha256
    from .schema import benchmark_dataset_sha256

    if (
        benchmark_dataset_sha256(moderate) != moderate_artifact.dataset_sha256
        or benchmark_dataset_sha256(severe) != severe_artifact.dataset_sha256
    ):
        raise ScalingContractError("scaling dataset semantics changed after simulation")
    moderate_truth = _truth_sha256(moderate)
    severe_truth = _truth_sha256(severe)
    if moderate_truth != severe_truth:
        raise ScalingContractError("paired scaling views do not share exact truth")
    moderate_manifest = moderate_artifact.native_manifest
    severe_manifest = severe_artifact.native_manifest
    moderate_native = moderate_manifest.as_dict()
    severe_native = severe_manifest.as_dict()
    if moderate_native["files"] != severe_native["files"]:
        raise ScalingContractError("paired scaling native file inventories differ")
    moderate_path = moderate_artifact.request.output_path.absolute()
    severe_path = severe_artifact.request.output_path.absolute()
    try:
        moderate_relative = moderate_path.relative_to(output_dir.absolute()).as_posix()
        severe_path.relative_to(output_dir.absolute())
    except ValueError as error:
        raise ScalingContractError("scaling output escaped its result root") from error
    seeds = derive_scaling_seeds(contract, protocol.development.cells)
    request = moderate_artifact.request
    design_sha256 = canonical_sha256(
        {
            "schema": "maskimpute-scaling-dataset-design-v1",
            "scaling_contract_sha256": contract.file_sha256,
            "protocol": asdict(protocol),
            "request": {
                "mechanism": request.mechanism,
                "namespace": request.namespace,
                "biological_id": request.biological_id,
                "cells": request.cells,
                "genes": request.genes,
                "biological_seed": request.biological_seed,
                "moderate_measurement_seed": seeds.moderate,
                "severe_measurement_seed": seeds.severe,
            },
        }
    )
    unsigned: dict[str, object] = {
        "schema_version": 1,
        "cells": request.cells,
        "genes": request.genes,
        "namespace": request.namespace,
        "mechanism": request.mechanism,
        "technical_view": "moderate",
        "dataset_id": request.dataset_id,
        "independent_unit_id": request.independent_unit_id,
        "dataset_sha256": moderate_artifact.dataset_sha256,
        "truth_sha256": moderate_truth,
        "moderate_output_path": moderate_relative,
        "moderate_output_file_sha256": _file_sha256(moderate_path),
        "moderate_output_size_bytes": moderate_path.stat().st_size,
        "severe_dataset_sha256": severe_artifact.dataset_sha256,
        "severe_output_file_sha256": _file_sha256(severe_path),
        "severe_output_size_bytes": severe_path.stat().st_size,
        "moderate_native_manifest_sha256": moderate_manifest.manifest_sha256,
        "severe_native_manifest_sha256": severe_manifest.manifest_sha256,
        "native_files_sha256": canonical_sha256(moderate_native["files"]),
        "protocol_sha256": canonical_sha256(asdict(protocol)),
        "design_sha256": design_sha256,
        "seed_source_sha256": contract.file_sha256,
        "seeds": asdict(seeds),
        "severe_retention": "discarded_after_receipt",
        "native_retention": "discarded_after_receipt",
    }
    return {**unsigned, "receipt_sha256": canonical_sha256(unsigned)}, moderate


def _dataset_binding(receipt: Mapping[str, object]) -> DatasetBinding:
    return DatasetBinding(
        mechanism=str(receipt["mechanism"]),
        biological_id="draw-01",
        technical_view=str(receipt["technical_view"]),
        dataset_id=str(receipt["dataset_id"]),
        dataset_sha256=str(receipt["dataset_sha256"]),
        output_file_sha256=str(receipt["moderate_output_file_sha256"]),
        truth_sha256=str(receipt["truth_sha256"]),
        output_path=str(receipt["moderate_output_path"]),
        independent_unit_id=str(receipt["independent_unit_id"]),
        cells=int(receipt["cells"]),
        genes=int(receipt["genes"]),
        manifest_sha256=str(receipt["receipt_sha256"]),
        protocol_sha256=str(receipt["protocol_sha256"]),
        design_sha256=str(receipt["design_sha256"]),
        seed_source_sha256=str(receipt["seed_source_sha256"]),
    )


def _load_scaling_dataset(
    store: ScalingResultStore,
    receipt: Mapping[str, object],
    authority: RunnerAuthority,
) -> PreparedDataset:
    import anndata as ad

    path = store._artifact_path(
        receipt["moderate_output_path"], "moderate scaling dataset"
    )
    try:
        dataset = ad.read_h5ad(path)
    except (OSError, TypeError, ValueError) as error:
        raise ScalingContractError(
            "moderate scaling dataset cannot be loaded"
        ) from error
    from .datasets import _truth_sha256
    from .schema import benchmark_dataset_sha256

    if (
        benchmark_dataset_sha256(dataset) != receipt["dataset_sha256"]
        or _truth_sha256(dataset) != receipt["truth_sha256"]
    ):
        raise ScalingContractError("moderate scaling dataset semantic hash mismatch")
    try:
        return prepare_dataset_for_execution(
            dataset, _dataset_binding(receipt), authority.dataset_qc_policy
        )
    except Exception as error:
        raise ScalingContractError("moderate scaling dataset QC failed") from error


def _require_scaling_disk_capacity(output_dir: Path, cells: int, genes: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Three native matrices, two H5AD views, staging copies, and a 2 GiB reserve.
    required = cells * genes * 96 + 2 * 1024**3
    available = shutil.disk_usage(output_dir).free
    if available < required:
        raise ScalingContractError(
            "insufficient disk for bounded scaling materialization: "
            f"required={required}, available={available}"
        )


def _cleanup_discarded_scaling_inputs(
    output_dir: Path, cells: int, receipt: Mapping[str, object]
) -> None:
    namespace_root = (
        output_dir / "generated" / f"scaling-{cells}" / "dataset"
    ).absolute()
    severe = namespace_root / "severe.h5ad"
    if severe.exists():
        if (
            severe.is_symlink()
            or not severe.is_file()
            or severe.stat().st_size != receipt["severe_output_size_bytes"]
            or _file_sha256(severe) != receipt["severe_output_file_sha256"]
        ):
            raise ScalingContractError("discarded severe scaling input changed")
        severe.unlink()
    native = namespace_root / "native"
    if native.exists():
        if native.is_symlink() or not native.is_dir():
            raise ScalingContractError("scaling native output root is invalid")
        content_directories = tuple(native.iterdir())
        if (
            len(content_directories) != 1
            or content_directories[0].is_symlink()
            or not content_directories[0].is_dir()
        ):
            raise ScalingContractError("scaling native content directory is invalid")
        inventory: list[dict[str, object]] = []
        for path in sorted(
            content_directories[0].iterdir(), key=lambda item: item.name
        ):
            metadata = path.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise ScalingContractError("scaling native inventory is invalid")
            inventory.append(
                {
                    "path": path.name,
                    "sha256": _file_sha256(path),
                    "size_bytes": metadata.st_size,
                }
            )
        if canonical_sha256(inventory) != receipt["native_files_sha256"]:
            raise ScalingContractError(
                "scaling native inventory changed before discard"
            )
        shutil.rmtree(native)
    if namespace_root.exists():
        descriptor = os.open(namespace_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _materialize_scaling_dataset(
    contract: ScalingContract,
    base_protocol: Protocol,
    output_dir: Path,
    cells: int,
    *,
    simulator: Any = run_symsim_pair,
) -> tuple[dict[str, object], Any]:
    _require_scaling_disk_capacity(output_dir, cells, contract.genes)
    protocol = scaling_protocol(base_protocol, contract, cells)
    requests = scaling_requests(contract, protocol, output_dir / "generated")
    try:
        artifacts = simulator(requests, protocol)
    except Exception as error:
        raise ScalingContractError(
            f"SymSim scaling materialization failed at {cells} cells"
        ) from error
    return _dataset_receipt_from_artifacts(contract, protocol, output_dir, artifacts)


def load_scaling_execution_authority(
    repository: Path,
) -> ScalingExecutionAuthority:
    """Validate the frozen candidate, scaling contract, runtime, and denominator."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    canonical_repository = Path(__file__).resolve().parents[1]
    try:
        selected = repository.resolve(strict=True)
    except OSError as error:
        raise ScalingContractError("repository is unavailable") from error
    if selected != canonical_repository:
        raise ScalingContractError("scaling must run from the canonical repository")
    contract_path = selected / "study/scaling_panel.json"
    contract = load_scaling_contract(contract_path)
    protocol = load_protocol(selected / "study/protocol.json")
    from .publication_freeze import validate_frozen_method

    try:
        frozen = validate_frozen_method(selected)
    except Exception as error:
        raise ScalingContractError("frozen method authority is unavailable") from error
    artifact_bindings = frozen.get("artifact_bindings")
    if not isinstance(artifact_bindings, Mapping):
        raise ScalingContractError("frozen method lacks artifact bindings")
    scaling_binding = artifact_bindings.get("scaling_panel")
    if (
        not isinstance(scaling_binding, Mapping)
        or scaling_binding.get("path") != "study/scaling_panel.json"
        or scaling_binding.get("sha256") != contract.file_sha256
    ):
        raise ScalingContractError("frozen method does not bind the scaling contract")
    authority = load_runner_authority()
    if not authority.maskimpute_ready:
        raise ScalingContractError("score/calibration authority is not ready")
    registry_path = selected / "study/methods.json"
    registry = load_method_registry(registry_path)
    from .final_runner import _configuration_for_method

    configurations = tuple(
        _configuration_for_method(method_id, registry.by_id(method_id), frozen)
        for method_id in contract.method_ids
    )
    environments = ExecutionEnvironmentRegistry.fixed(
        selected,
        runtime_lock_path=selected / "environments/development-runtime.lock.json",
        benchmark_python=Path(sys.executable),
        r_library_paths={"saver": (selected / "artifacts/envs/saver-r/library",)},
    )
    plan = build_scaling_plan(
        contract,
        registry,
        configurations,
        frozen_method_sha256=_sha256(
            frozen.get("payload_sha256"), "frozen method payload checksum"
        ),
        method_registry_file_sha256=_file_sha256(registry_path),
        execution_authority_sha256=authority.authority_sha256,
        execution_environment_sha256=environments.registry_sha256,
        implementation_source_sha256=implementation_source_sha256(),
    )
    return ScalingExecutionAuthority(
        repository=selected,
        contract=contract,
        protocol=protocol,
        frozen_method=MappingProxyType(dict(frozen)),
        registry=registry,
        runner_authority=authority,
        environments=environments,
        plan=plan,
    )


def _run_plan_entry(entry: ScalingPlanEntry, binding: DatasetBinding) -> RunPlanEntry:
    return RunPlanEntry(
        ordinal=entry.ordinal,
        run_id=entry.run_id,
        method_id=entry.method_id,
        dataset_id=binding.dataset_id,
        source_dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        model_seed=entry.model_seed,
        configuration_id=entry.configuration_id,
        configuration_sha256=entry.configuration_sha256,
        preflight_status="planned",
        preflight_reason=None,
        configuration_kind=entry.configuration_kind,
        requires_count_score=entry.requires_count_score,
        requires_calibration=entry.requires_calibration,
    )


def _bounded_scaling_metric_values(
    imputed: Any, observed: Any, truth: Any
) -> dict[str, tuple[float | None, int, str | None]]:
    """Compute only cell-linear/gene-correlation accuracy at scaling sizes."""

    try:
        imputed_array = np.asarray(imputed, dtype=np.float64)
        observed_array = np.asarray(observed, dtype=np.float64)
        truth_array = np.asarray(truth, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ScalingContractError("scaling accuracy matrices are invalid") from error
    if (
        imputed_array.ndim != 2
        or imputed_array.shape != observed_array.shape
        or imputed_array.shape != truth_array.shape
        or imputed_array.size == 0
        or not np.isfinite(imputed_array).all()
        or not np.isfinite(observed_array).all()
        or not np.isfinite(truth_array).all()
    ):
        raise ScalingContractError("scaling accuracy matrices are invalid")
    difference = imputed_array - truth_array
    masks = {
        "overall": np.ones(truth_array.shape, dtype=bool),
        "dropout": (observed_array == 0) & (truth_array > 0),
        "pre_dropout_zero": truth_array == 0,
        "nonzero": (observed_array > 0) & (truth_array > 0),
    }

    def squared(mask: np.ndarray) -> tuple[float | None, int, str | None]:
        n = int(mask.sum())
        if n == 0:
            return None, 0, "no_entries"
        selected = difference[mask]
        return float(np.mean(selected * selected)), n, None

    result = {
        "mse": squared(masks["overall"]),
        "mse_dropout": squared(masks["dropout"]),
        "mse_pre_dropout_zero": squared(masks["pre_dropout_zero"]),
        "mse_nonzero": squared(masks["nonzero"]),
    }
    truth_sd = np.maximum(np.std(truth_array, axis=0, ddof=0), 1e-8)
    rmse = np.sqrt(np.mean(difference * difference, axis=0))
    result["gnrmse"] = (float(np.mean(rmse / truth_sd)), truth_array.shape[1], None)
    result["mean_distortion"] = (
        float(
            np.mean(
                np.abs(np.mean(imputed_array, axis=0) - np.mean(truth_array, axis=0))
            )
        ),
        truth_array.shape[1],
        None,
    )
    result["variance_distortion"] = (
        float(
            np.mean(
                np.abs(
                    np.var(imputed_array, axis=0, ddof=0)
                    - np.var(truth_array, axis=0, ddof=0)
                )
            )
        ),
        truth_array.shape[1],
        None,
    )
    sorted_imputed = np.sort(imputed_array, axis=0)
    sorted_truth = np.sort(truth_array, axis=0)
    result["mean_gene_wasserstein_distance"] = (
        float(np.mean(np.mean(np.abs(sorted_imputed - sorted_truth), axis=0))),
        truth_array.shape[1],
        None,
    )
    del sorted_imputed, sorted_truth
    n_genes = truth_array.shape[1]
    constant = (np.std(imputed_array, axis=0, ddof=0) == 0) | (
        np.std(truth_array, axis=0, ddof=0) == 0
    )
    if n_genes < 2:
        result["corr_err"] = (None, n_genes, "fewer_than_two_variable_genes")
    elif np.any(constant):
        result["corr_err"] = (None, n_genes, "constant_gene_profile")
    else:
        imputed_correlation = np.corrcoef(imputed_array, rowvar=False)
        truth_correlation = np.corrcoef(truth_array, rowvar=False)
        upper = np.triu_indices(n_genes, k=1)
        distortion = np.abs(imputed_correlation[upper] - truth_correlation[upper])
        if np.isfinite(distortion).all():
            result["corr_err"] = (float(np.mean(distortion)), len(distortion), None)
        else:
            result["corr_err"] = (
                None,
                len(distortion),
                "nonfinite_correlation",
            )
    result["n_corr_genes"] = (float(n_genes), n_genes, None)
    if tuple(result) != _SCALING_ACCURACY_METRICS:
        raise ScalingContractError("bounded scaling metric order drifted")
    return result


def _evaluate_scaling_outcome(
    entry: ScalingPlanEntry,
    run_entry: RunPlanEntry,
    prepared: PreparedDataset,
    outcome: AdapterOutcome,
) -> ScalingEvaluatedAttempt:
    """Evaluate without any cell-by-cell matrix, then discard both dense outputs."""

    from .runner import (
        DatasetQCPolicy,
        LongFormMetric,
        RawRunResult,
        _default_output_converter,
        _evaluator_conversion_failure_reason,
        _evaluator_output_sha256,
        _evaluator_targets,
        method_input_sha256,
    )
    from .methods import AdapterUnavailableError

    status = outcome.status
    reason = outcome.reason
    native_output_sha256: str | None = None
    evaluator_output_sha256: str | None = None
    values: dict[str, tuple[float | None, int, str | None]]
    if outcome.status == "completed":
        assert outcome.execution is not None
        snapshot = outcome.execution.snapshot
        if (
            snapshot.method_id != entry.method_id
            or snapshot.source_dataset_sha256 != prepared.binding.dataset_sha256
            or snapshot.obs_ids != prepared.audit.retained_cell_ids
            or snapshot.var_ids != prepared.method_input.var_ids
        ):
            raise ScalingContractError(
                "completed scaling snapshot differs from its dataset/method"
            )
        native_output_sha256 = snapshot.matrix_sha256
        try:
            evaluator_output = _default_output_converter(
                prepared.method_input, outcome.execution
            )
            evaluator_output = np.asarray(evaluator_output, dtype=np.float64)
            observed, truth, truth_kind, _marker_mask = _evaluator_targets(prepared)
            if truth_kind != "exact_pre_capture" or truth is None:
                raise ScalingContractError(
                    "scaling accuracy requires exact pre-capture truth"
                )
            values = _bounded_scaling_metric_values(evaluator_output, observed, truth)
            evaluator_output_sha256 = _evaluator_output_sha256(
                run_entry, prepared, evaluator_output
            )
            del evaluator_output, observed, truth
        except ScalingContractError:
            raise
        except (AdapterUnavailableError, TypeError, ValueError, OverflowError) as error:
            status = "unavailable"
            reason = _evaluator_conversion_failure_reason(error)
            values = {name: (None, 0, reason) for name in _SCALING_ACCURACY_METRICS}
    else:
        assert reason is not None
        values = {name: (None, 0, reason) for name in _SCALING_ACCURACY_METRICS}
    metrics = tuple(
        LongFormMetric(
            mechanism=run_entry.mechanism,
            biological_id=run_entry.biological_id,
            technical_view=run_entry.technical_view,
            dataset_id=run_entry.dataset_id,
            method=entry.method_id,
            model_seed=entry.model_seed,
            configuration_id=entry.configuration_id,
            configuration_sha256=entry.configuration_sha256,
            metric=name,
            value=value,
            n=n,
            status="completed" if value is not None else "unavailable",
            reason=metric_reason,
        )
        for name, (value, n, metric_reason) in values.items()
    )
    calibration = outcome.calibration_fold_receipt
    run = RawRunResult(
        run_id=entry.run_id,
        method_id=entry.method_id,
        dataset_id=run_entry.dataset_id,
        source_dataset_sha256=run_entry.source_dataset_sha256,
        mechanism=run_entry.mechanism,
        biological_id=run_entry.biological_id,
        technical_view=run_entry.technical_view,
        model_seed=entry.model_seed,
        configuration_id=entry.configuration_id,
        configuration_sha256=entry.configuration_sha256,
        configuration_kind=entry.configuration_kind,
        requires_count_score=entry.requires_count_score,
        requires_calibration=entry.requires_calibration,
        method_input_sha256=method_input_sha256(prepared.method_input),
        dataset_qc_policy_sha256=DatasetQCPolicy.fixed().sha256,
        excluded_cell_count=prepared.audit.excluded_cell_count,
        excluded_cell_ids_sha256=prepared.audit.excluded_cell_ids_sha256,
        retained_cell_count=prepared.audit.retained_cell_count,
        retained_cell_ids_sha256=prepared.audit.retained_cell_ids_sha256,
        retained_gene_count=prepared.method_input.shape[1],
        observed_zero_count=(
            prepared.method_input.counts.size
            - int(np.count_nonzero(prepared.method_input.counts))
        ),
        status=status,
        reason=reason,
        runtime_seconds=outcome.runtime_seconds,
        peak_rss_bytes=outcome.peak_rss_bytes,
        peak_gpu_bytes=outcome.peak_gpu_bytes,
        rss_measurement=outcome.rss_measurement,
        gpu_measurement=outcome.gpu_measurement,
        calibration_artifact_sha256=(
            None if calibration is None else calibration.calibration_artifact_sha256
        ),
        calibration_context_sha256=(
            None if calibration is None else calibration.calibration_context_sha256
        ),
        calibration_training_manifest_sha256s=(
            () if calibration is None else calibration.training_manifest_sha256s
        ),
        calibration_held_out_manifest_sha256s=(
            () if calibration is None else calibration.held_out_manifest_sha256s
        ),
        calibration_fold_calibrator_sha256=(
            None if calibration is None else calibration.fold_calibrator_sha256
        ),
        stdout_sha256=hashlib.sha256(outcome.stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(outcome.stderr).hexdigest(),
        native_output_sha256=native_output_sha256,
        evaluator_output_sha256=evaluator_output_sha256,
    )
    return ScalingEvaluatedAttempt(
        run=run,
        metrics=metrics,
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        native_output=None,
        native_output_scale=(
            None
            if outcome.execution is None
            else outcome.execution.snapshot.output_scale
        ),
        evaluator_output=None,
    )


def execute_scaling_plan(
    authority: ScalingExecutionAuthority,
    output_dir: Path,
    *,
    simulator: Any = run_symsim_pair,
    executor: Any | None = None,
) -> ScalingCheckpoint:
    """Execute/resume every size and method while preserving the full denominator."""

    if not isinstance(authority, ScalingExecutionAuthority):
        raise TypeError("authority must be a ScalingExecutionAuthority")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    store = ScalingResultStore(output_dir, authority.plan)
    report = store.load()
    configuration_by_method = {
        value.method_id: value for value in authority.plan.configurations
    }
    owned_executor = executor is None
    selected_executor = (
        SpawnedRepositoryExecutor(
            RepositoryAdapterDispatcher(authority.repository, authority.environments)
        )
        if executor is None
        else executor
    )
    if not callable(selected_executor):
        raise TypeError("executor must be callable")
    expected_cells = tuple(
        dict.fromkeys(entry.cells for entry in authority.plan.entries)
    )
    try:
        for size_index, cells in enumerate(expected_cells):
            report = store.load()
            assert report is not None or size_index == 0
            records_count = 0 if report is None else len(report.records)
            size_entries = tuple(
                entry for entry in authority.plan.entries if entry.cells == cells
            )
            if records_count >= size_entries[-1].ordinal:
                if report is not None and len(report.datasets) > size_index:
                    _cleanup_discarded_scaling_inputs(
                        store.output_dir, cells, report.datasets[size_index]
                    )
                continue
            if report is None or len(report.datasets) <= size_index:
                receipt, _generated = _materialize_scaling_dataset(
                    authority.contract,
                    authority.protocol,
                    store.output_dir,
                    cells,
                    simulator=simulator,
                )
                report = store.append_dataset(receipt)
                del _generated
                _cleanup_discarded_scaling_inputs(
                    store.output_dir, cells, report.datasets[size_index]
                )
            receipt = report.datasets[size_index]
            prepared = _load_scaling_dataset(store, receipt, authority.runner_authority)
            binding = prepared.binding
            for entry in size_entries:
                current = store.load()
                assert current is not None
                if len(current.records) >= entry.ordinal:
                    continue
                spec = authority.registry.by_id(entry.method_id)
                configuration = configuration_by_method[entry.method_id]
                request = ExecutionRequest.create(
                    spec,
                    prepared.method_input,
                    model_seed=entry.model_seed,
                    configuration=configuration,
                    authority=authority.runner_authority.execution_context,
                    mechanism=binding.mechanism,
                    biological_id=binding.biological_id,
                    technical_view=binding.technical_view,
                    dataset_id=binding.dataset_id,
                    timeout_seconds=spec.resources.timeout_seconds,
                    calibration_usage="retained_all_development",
                )
                outcome = selected_executor(request)
                if not isinstance(outcome, AdapterOutcome):
                    raise ScalingContractError(
                        "scaling executor returned a noncanonical outcome"
                    )
                outcome = enforce_calibration_fold_receipt(request, outcome)
                attempt = _evaluate_scaling_outcome(
                    entry, _run_plan_entry(entry, binding), prepared, outcome
                )
                record = scaling_attempt_record(
                    attempt,
                    cells=cells,
                    accuracy_enabled=entry.accuracy_enabled,
                )
                report = store.append_attempt(entry, record)
                del attempt, record, outcome
            del prepared
        final = store.load()
        if final is None or final.status != "completed":
            raise ScalingContractError(
                "scaling execution did not complete its denominator"
            )
        return final
    finally:
        if owned_executor and isinstance(selected_executor, SpawnedRepositoryExecutor):
            selected_executor.close()


def run_scaling_panel(repository: Path, output_dir: Path) -> ScalingCheckpoint:
    """Production entry point with no scientific-design command-line overrides."""

    return execute_scaling_plan(
        load_scaling_execution_authority(repository), output_dir
    )


__all__ = [
    "ScalingCheckpoint",
    "ScalingContract",
    "ScalingContractError",
    "ScalingExecutionAuthority",
    "ScalingEvaluatedAttempt",
    "ScalingPlan",
    "ScalingPlanEntry",
    "ScalingResultStore",
    "ScalingSeeds",
    "build_scaling_plan",
    "derive_scaling_seeds",
    "execute_scaling_plan",
    "load_scaling_execution_authority",
    "load_scaling_contract",
    "run_scaling_panel",
    "scaling_attempt_record",
    "scaling_protocol",
    "scaling_requests",
]
