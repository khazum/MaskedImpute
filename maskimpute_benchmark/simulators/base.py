"""Shared immutable contract for publication-study simulator adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING

from ..protocol import Protocol, canonical_sha256

if TYPE_CHECKING:
    import anndata as ad

    from .native import NativeManifest


_SAFE_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_CLAIM_TOKEN = object()


class SimulationContractError(ValueError):
    """Raised when simulator input or output violates the study contract."""


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """All design and seed inputs for one technical-view simulation."""

    mechanism: str
    namespace: str
    biological_id: str
    biological_seed: int
    measurement_seed: int
    technical_view: str
    cells: int
    genes: int
    output_path: Path

    @property
    def dataset_id(self) -> str:
        return simulation_dataset_id(self)

    @property
    def independent_unit_id(self) -> str:
        return biological_unit_id(self)


@dataclass(frozen=True, slots=True, init=False)
class FinalManifestClaim:
    """Read-only snapshot of an already-claimed sealed final execution."""

    round_id: str
    generator_seeds: tuple[int, ...] = field(repr=False)
    seed_manifest_sha256: str
    execution_claim_id: str
    round_dir: Path
    _token: object = field(repr=False)


@dataclass(frozen=True, slots=True, init=False)
class SimulationArtifact:
    """A schema-valid translated dataset bound to sealed native bytes."""

    _adata: ad.AnnData = field(repr=False)
    native_manifest: NativeManifest
    dataset_sha256: str

    def __init__(
        self,
        adata: ad.AnnData,
        native_manifest: NativeManifest,
        dataset_sha256: str,
    ) -> None:
        from ..schema import benchmark_dataset_sha256, validate_benchmark_dataset
        from .native import revalidate_native_outputs

        revalidate_native_outputs(native_manifest)
        try:
            snapshot = adata.copy()
        except (AttributeError, TypeError, ValueError) as error:
            raise SimulationContractError(
                "translated AnnData could not be copied into the artifact"
            ) from error
        try:
            validate_benchmark_dataset(snapshot)
        except (TypeError, ValueError) as error:
            raise SimulationContractError(
                f"translated AnnData violates the truth-dataset schema: {error}"
            ) from error
        provenance = snapshot.uns.get("provenance")
        parameters = (
            provenance.get("parameters") if isinstance(provenance, Mapping) else None
        )
        embedded = (
            parameters.get("native_manifest_sha256")
            if isinstance(parameters, Mapping)
            else None
        )
        if embedded != native_manifest.manifest_sha256:
            raise SimulationContractError(
                "translated AnnData does not bind the sealed native manifest"
            )
        observed = benchmark_dataset_sha256(snapshot)
        if (
            not isinstance(dataset_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", dataset_sha256)
            or dataset_sha256 != observed
        ):
            raise SimulationContractError(
                "dataset_sha256 does not match the translated AnnData"
            )
        object.__setattr__(self, "_adata", snapshot)
        object.__setattr__(self, "native_manifest", native_manifest)
        object.__setattr__(self, "dataset_sha256", dataset_sha256)

    @property
    def adata(self) -> ad.AnnData:
        """Return a defensive copy of the artifact's validated dataset snapshot."""

        return self._adata.copy()


def _design_digest(kind: str, request: SimulationRequest, *, view: bool) -> str:
    payload = {
        "schema": f"maskimpute-{kind}-v1",
        "namespace": request.namespace,
        "mechanism": request.mechanism,
        "biological_id": request.biological_id,
    }
    if view:
        payload["technical_view"] = request.technical_view
    return canonical_sha256(payload)[:24]


def simulation_dataset_id(request: SimulationRequest) -> str:
    """Return the stable view ID, independent of seeds and biological labels."""

    return f"dataset-{_design_digest('simulation-dataset', request, view=True)}"


def biological_unit_id(request: SimulationRequest) -> str:
    """Return the unit shared by paired technical views of one biological draw."""

    return f"biological-{_design_digest('biological-unit', request, view=False)}"


def _validate_safe_id(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise SimulationContractError(
            f"{name} must be a safe lowercase hyphen-separated identifier"
        )


def _validate_seed(value: object, name: str) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise SimulationContractError(f"{name} must be a 63-bit nonnegative integer")


def _validate_output_path(request: SimulationRequest) -> None:
    path = request.output_path
    if not isinstance(path, Path):
        raise SimulationContractError("output_path must be a pathlib.Path")
    if "\\" in str(path) or any(part in {".", ".."} for part in path.parts):
        raise SimulationContractError("output_path must be a canonical native path")
    if path.name in {"", ".", ".."} or request.namespace not in path.parts:
        raise SimulationContractError(
            "output_path must contain its namespace as a complete path component"
        )


def _valid_final_claim(value: object) -> bool:
    return isinstance(value, FinalManifestClaim) and value._token is _CLAIM_TOKEN


def validate_simulation_request(
    request: SimulationRequest,
    protocol: Protocol,
    final_manifest: FinalManifestClaim | None = None,
) -> None:
    """Fail closed unless a request belongs to exactly one protocol namespace."""

    if not isinstance(request, SimulationRequest):
        raise TypeError("request must be a SimulationRequest")
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    if request.mechanism not in protocol.mechanisms:
        raise SimulationContractError("mechanism is not in the publication protocol")
    _validate_safe_id(request.mechanism, "mechanism")
    _validate_safe_id(request.biological_id, "biological_id")
    _validate_safe_id(request.technical_view, "technical_view")
    _validate_seed(request.biological_seed, "biological_seed")
    _validate_seed(request.measurement_seed, "measurement_seed")
    if request.biological_seed == request.measurement_seed:
        raise SimulationContractError(
            "biological_seed and measurement_seed must be distinct"
        )
    if type(request.cells) is not int or request.cells <= 0:
        raise SimulationContractError("cells must be a positive integer")
    if type(request.genes) is not int or request.genes <= 0:
        raise SimulationContractError("genes must be a positive integer")
    if request.namespace == protocol.development.namespace:
        if final_manifest is not None:
            raise SimulationContractError(
                "development requests cannot receive a final manifest claim"
            )
        _validate_output_path(request)
        expected = (protocol.development.cells, protocol.development.genes)
    elif request.namespace == protocol.final.namespace:
        if not _valid_final_claim(final_manifest):
            raise SimulationContractError(
                "final request requires a validated final manifest claim"
            )
        assert final_manifest is not None
        if (
            request.biological_seed not in final_manifest.generator_seeds
            or request.measurement_seed not in final_manifest.generator_seeds
        ):
            raise SimulationContractError(
                "final biological and measurement seeds must come from the manifest"
            )
        _validate_output_path(request)
        try:
            request.output_path.resolve(strict=False).relative_to(
                (final_manifest.round_dir / "results").resolve(strict=False)
            )
        except (OSError, ValueError) as error:
            raise SimulationContractError(
                "final output_path must be beneath the claimed round results directory"
            ) from error
        expected = (protocol.final.cells, protocol.final.genes)
    else:
        raise SimulationContractError(
            "namespace must be exactly the development or final protocol namespace"
        )

    if (request.cells, request.genes) != expected:
        raise SimulationContractError(
            f"request dimensions must exactly match protocol dimensions {expected}"
        )


def validate_paired_simulation_requests(
    requests: Sequence[SimulationRequest],
    protocol: Protocol,
    final_manifest: FinalManifestClaim | None = None,
) -> None:
    """Validate two technical views as one, and only one, biological unit."""

    if (
        not isinstance(requests, Sequence)
        or isinstance(requests, (str, bytes))
        or len(requests) != 2
    ):
        raise SimulationContractError(
            "paired simulation contract requires exactly two technical views"
        )
    first, second = requests
    validate_simulation_request(first, protocol, final_manifest)
    validate_simulation_request(second, protocol, final_manifest)
    first_draw = (
        first.namespace,
        first.mechanism,
        first.biological_id,
        first.cells,
        first.genes,
    )
    second_draw = (
        second.namespace,
        second.mechanism,
        second.biological_id,
        second.cells,
        second.genes,
    )
    if first_draw != second_draw:
        raise SimulationContractError(
            "paired technical views must describe the same biological draw"
        )
    if first.biological_seed != second.biological_seed:
        raise SimulationContractError(
            "paired technical views must reuse the same biological seed and truth"
        )
    if first.technical_view == second.technical_view:
        raise SimulationContractError("paired technical views must have distinct names")
    if first.measurement_seed == second.measurement_seed:
        raise SimulationContractError(
            "paired technical views must use distinct measurement seeds"
        )
    if first.independent_unit_id != second.independent_unit_id:
        raise SimulationContractError(
            "paired technical views cannot claim independent biological units"
        )


def load_final_manifest_claim(repo: Path, round_dir: Path) -> FinalManifestClaim:
    """Load, without consuming it, an already-issued final execution claim."""

    from .. import study

    try:
        repository, destination = study._repository_for_round(round_dir, repo)
        with study._round_lock(repository, destination.name) as lock_identity:
            freeze = study._verify_frozen_repository(repository, destination)
            registry = study._validate_registry(
                repository, destination, freeze, expected_state="running"
            )
            materialization, manifest = study._validate_seed_manifest(
                destination, freeze
            )
            execution = study._validate_execution_claim_record(
                destination, freeze, materialization
            )
            first_commitment = canonical_sha256(
                {
                    "registry": registry,
                    "materialization": materialization,
                    "manifest": manifest,
                    "execution": execution,
                }
            )

            # Repeat every mutable-record read before returning the snapshot.
            freeze = study._verify_frozen_repository(repository, destination)
            registry = study._validate_registry(
                repository, destination, freeze, expected_state="running"
            )
            materialization, manifest = study._validate_seed_manifest(
                destination, freeze
            )
            execution = study._validate_execution_claim_record(
                destination, freeze, materialization
            )
            second_commitment = canonical_sha256(
                {
                    "registry": registry,
                    "materialization": materialization,
                    "manifest": manifest,
                    "execution": execution,
                }
            )
            study._assert_round_lock_identity(
                repository, destination.name, lock_identity
            )
            if first_commitment != second_commitment:
                raise SimulationContractError(
                    "final manifest or execution claim changed while loading"
                )
    except SimulationContractError:
        raise
    except (OSError, TypeError, ValueError, study.StudyStateError) as error:
        raise SimulationContractError(
            f"final execution is not a valid claimed running manifest: {error}"
        ) from error

    value = object.__new__(FinalManifestClaim)
    object.__setattr__(value, "round_id", manifest["round_id"])
    object.__setattr__(value, "generator_seeds", tuple(manifest["generator_seeds"]))
    object.__setattr__(
        value, "seed_manifest_sha256", materialization["seed_manifest_sha256"]
    )
    object.__setattr__(value, "execution_claim_id", execution["execution_claim_id"])
    object.__setattr__(value, "round_dir", destination)
    object.__setattr__(value, "_token", _CLAIM_TOKEN)
    return value


__all__ = [
    "FinalManifestClaim",
    "SimulationArtifact",
    "SimulationContractError",
    "SimulationRequest",
    "biological_unit_id",
    "load_final_manifest_claim",
    "simulation_dataset_id",
    "validate_paired_simulation_requests",
    "validate_simulation_request",
]
