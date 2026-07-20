"""Direct-identity planning for the development fair-comparator panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import math
from typing import Literal

import numpy as np

from .comparator_tuning import (
    ComparatorMethodBinding,
    bind_comparator_configuration_identity,
    comparator_method_binding,
)
from .methods import MethodRegistry, MethodSpec
from .runner import (
    DEVELOPMENT_MODEL_SEEDS,
    AuthorizedConfiguration,
    DatasetBinding,
    PreparedDataset,
    RunnerAuthority,
    RunnerContractError,
)


_WORKFLOW_SCHEMA = "maskimpute-fair-comparator-run-v1"
_PREPROCESSING_REVISION = "paired-zero-library-union-v1"


class _FrozenJsonList(tuple[object, ...]):
    pass


class _FrozenJsonObject(tuple[tuple[str, object], ...]):
    pass


def _freeze_nested_payload(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RunnerContractError("direct configuration keys must be strings")
        return _FrozenJsonObject(
            (key, _freeze_payload(nested)) for key, nested in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return _FrozenJsonList(_freeze_payload(nested) for nested in value)
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise RunnerContractError("direct configuration payload is not canonical JSON")


def _freeze_payload(value: object) -> object:
    return _freeze_nested_payload(value)


def _freeze_payload_mapping(
    value: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    return tuple(
        (key, _freeze_nested_payload(nested)) for key, nested in sorted(value.items())
    )


def _thaw_payload(value: object) -> object:
    if isinstance(value, _FrozenJsonObject):
        return {item[0]: _thaw_payload(item[1]) for item in value}
    if isinstance(value, _FrozenJsonList):
        return [_thaw_payload(item) for item in value]
    if isinstance(value, tuple):
        if all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {item[0]: _thaw_payload(item[1]) for item in value}
        return [_thaw_payload(item) for item in value]
    return value


def direct_json_value(value: object, *, payload: bool = False) -> object:
    """Encode one direct dataclass/value without losing JSON container types."""

    if payload:
        return _thaw_payload(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: direct_json_value(
                getattr(value, item.name),
                payload=item.name in {"payload", "configuration_payload"},
            )
            for item in fields(value)
        }
    if isinstance(value, tuple):
        return [direct_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): direct_json_value(nested) for key, nested in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class PreparedInputDescriptor:
    dataset_id: str
    source_reference: str
    preprocessing_revision: str
    shape: tuple[int, int]
    dtype: str
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    batch_labels: tuple[str, ...]
    total_count: float
    nonzero_count: int
    minimum: float
    maximum: float
    mechanism: str
    mask_seed: int
    technical_view: str


@dataclass(frozen=True, slots=True)
class DirectAuthorizedConfiguration:
    method: ComparatorMethodBinding
    configuration_id: str
    configuration_kind: str
    payload: tuple[tuple[str, object], ...]
    requires_count_score: bool
    requires_calibration: bool


@dataclass(frozen=True, slots=True)
class ComparatorRunIdentity:
    workflow_schema: str
    authority_revision: str
    ordinal: int
    method: ComparatorMethodBinding
    configuration_id: str
    configuration_kind: str
    configuration_payload: tuple[tuple[str, object], ...]
    dataset_id: str
    mechanism: str
    biological_id: str
    technical_view: str
    mask_seed: int
    model_seed: int | None
    draw_index: int

    @property
    def method_id(self) -> str:
        return self.method.method_id


@dataclass(frozen=True, slots=True)
class DirectPlanEntry:
    run_id: str
    identity: ComparatorRunIdentity
    preflight_status: Literal["planned", "blocked_authority"]
    preflight_reason: str | None
    requires_count_score: bool
    requires_calibration: bool


@dataclass(frozen=True, slots=True)
class DirectCompetitionPlan:
    schema_version: int
    identity_mode: Literal["direct-v1"]
    authority_revision: str
    inputs: tuple[PreparedInputDescriptor, ...]
    entries: tuple[DirectPlanEntry, ...]
    configurations: tuple[DirectAuthorizedConfiguration, ...]

    def to_dict(self) -> dict[str, object]:
        return _direct_plan_to_json(self)


def _evaluator_seed_and_draw(prepared: PreparedDataset) -> tuple[int, int]:
    evaluator = prepared.evaluator_dataset
    provenance = getattr(evaluator, "uns", {}).get("provenance")
    seeds = provenance.get("seeds") if isinstance(provenance, Mapping) else None
    mask_seed = seeds.get("measurement") if isinstance(seeds, Mapping) else None
    if (
        isinstance(mask_seed, (bool, np.bool_))
        or not isinstance(mask_seed, (int, np.integer))
        or mask_seed < 0
    ):
        raise RunnerContractError(
            "prepared input mask seed must be an exact nonnegative integer"
        )
    normalized_mask_seed = int(mask_seed)
    obs = getattr(evaluator, "obs", None)
    if obs is None or "draw" not in obs:
        raise RunnerContractError("prepared input draw index metadata is absent")
    draw_values = obs["draw"].tolist()
    if (
        not draw_values
        or any(type(value) is not int or value < 1 for value in draw_values)
        or len(set(draw_values)) != 1
    ):
        raise RunnerContractError(
            "prepared input draw index must be one exact positive integer"
        )
    return normalized_mask_seed, draw_values[0]


def describe_prepared_input(prepared: PreparedDataset) -> PreparedInputDescriptor:
    """Copy one complete prepared-input description without content identity."""

    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    binding = prepared.binding
    method_input = prepared.method_input
    if not isinstance(binding, DatasetBinding):
        raise RunnerContractError(
            "direct development planning requires a development dataset binding"
        )
    mask_seed, _draw_index = _evaluator_seed_and_draw(prepared)
    batch_columns = tuple(
        column for column in method_input.obs_covariates if column.name == "batch"
    )
    if len(batch_columns) > 1:
        raise RunnerContractError("prepared input has multiple batch covariates")
    batch_labels = (
        ()
        if not batch_columns
        else tuple(str(value) for value in batch_columns[0].values)
    )
    if batch_labels and len(batch_labels) != method_input.shape[0]:
        raise RunnerContractError("prepared input batch labels differ from its rows")
    counts = method_input.counts
    if counts.size == 0:
        raise RunnerContractError("prepared input matrix must not be empty")
    source_path = getattr(binding, "output_path", None)
    source_reference = (
        source_path
        if isinstance(source_path, str) and source_path
        else binding.dataset_id
    )
    return PreparedInputDescriptor(
        dataset_id=binding.dataset_id,
        source_reference=source_reference,
        preprocessing_revision=_PREPROCESSING_REVISION,
        shape=method_input.shape,
        dtype=counts.dtype.str,
        cell_ids=method_input.obs_ids,
        gene_ids=method_input.var_ids,
        batch_labels=batch_labels,
        total_count=float(np.sum(counts, dtype=np.float64)),
        nonzero_count=int(np.count_nonzero(counts)),
        minimum=float(np.min(counts)),
        maximum=float(np.max(counts)),
        mechanism=binding.mechanism,
        mask_seed=mask_seed,
        technical_view=binding.technical_view,
    )


def direct_run_id(identity: ComparatorRunIdentity) -> str:
    seed = (
        "deterministic"
        if identity.model_seed is None
        else f"seed-{identity.model_seed}"
    )
    return (
        f"run-{identity.ordinal:04d}-{identity.method.method_id}-"
        f"{identity.dataset_id.removeprefix('dataset-')}-{seed}-"
        f"{identity.configuration_id}"
    )


def _direct_configuration(
    method: ComparatorMethodBinding,
    configuration: AuthorizedConfiguration,
) -> DirectAuthorizedConfiguration:
    return DirectAuthorizedConfiguration(
        method=method,
        configuration_id=configuration.configuration_id,
        configuration_kind=configuration.kind,
        payload=_freeze_payload_mapping(configuration.payload),
        requires_count_score=configuration.requires_count_score,
        requires_calibration=configuration.requires_calibration,
    )


def _configuration_grid(
    registry: MethodRegistry,
    authority: RunnerAuthority,
) -> tuple[
    tuple[MethodSpec, ...],
    tuple[DirectAuthorizedConfiguration, ...],
    dict[str, tuple[DirectAuthorizedConfiguration, ...]],
]:
    authority_by_method = {
        method_id: tuple(
            value for value in authority.configurations if value.method_id == method_id
        )
        for method_id in {value.method_id for value in authority.configurations}
    }
    planned_specs = (
        tuple(spec for spec in registry.methods if spec.id == "maskimpute")
        if authority.plan_scope == "revision_candidate_only"
        else tuple(
            spec
            for spec in registry.methods
            if spec.execution_scope == "same_input_required"
        )
    )
    plan_configurations: list[DirectAuthorizedConfiguration] = []
    configurations_by_method: dict[str, tuple[DirectAuthorizedConfiguration, ...]] = {}
    for spec in planned_specs:
        method = comparator_method_binding(spec)
        if spec.id == "observed":
            configurations = (
                DirectAuthorizedConfiguration(
                    method=method,
                    configuration_id="registry-default",
                    configuration_kind="registry",
                    payload=(),
                    requires_count_score=False,
                    requires_calibration=False,
                ),
            )
        elif spec.id in {"maskimpute", "capacity-matched-ae"}:
            configurations = tuple(
                _direct_configuration(method, value)
                for value in authority_by_method.get(spec.id, ())
            )
            if not configurations:
                raise RunnerContractError(
                    f"tracked authority has no configuration for {spec.id}"
                )
        else:
            authority_method = authority.comparator_method_bindings.get(spec.id)
            if authority_method != method:
                raise RunnerContractError(
                    f"runner comparator method binding differs for {spec.id}"
                )
            direct_values: list[DirectAuthorizedConfiguration] = []
            for row in authority.comparator_tuning.configurations_for(spec.id):
                bound = bind_comparator_configuration_identity(
                    row,
                    spec,
                    authority.comparator_tuning,
                )
                if (
                    bound.authority_reference != authority.comparator_tuning_reference
                    or bound.method != authority_method
                ):
                    raise RunnerContractError(
                        f"direct comparator binding differs for {spec.id}"
                    )
                direct_values.append(
                    DirectAuthorizedConfiguration(
                        method=bound.method,
                        configuration_id=bound.configuration.configuration_id,
                        configuration_kind="comparator_tuning",
                        payload=_freeze_payload_mapping(bound.configuration.payload),
                        requires_count_score=False,
                        requires_calibration=False,
                    )
                )
            configurations = tuple(direct_values)
            if not configurations:
                raise RunnerContractError(
                    f"comparator tuning has no configuration for {spec.id}"
                )
        configurations_by_method[spec.id] = configurations
        plan_configurations.extend(configurations)
    return planned_specs, tuple(plan_configurations), configurations_by_method


def _validate_direct_plan(
    plan: DirectCompetitionPlan,
    authority: RunnerAuthority,
) -> None:
    ordinals = [entry.identity.ordinal for entry in plan.entries]
    if ordinals != list(range(1, len(plan.entries) + 1)):
        raise RunnerContractError("direct plan ordinals are not contiguous")
    run_ids = [entry.run_id for entry in plan.entries]
    if len(run_ids) != len(set(run_ids)):
        raise RunnerContractError("direct plan run IDs are not unique")
    if authority.plan_scope == "base_full_panel":
        component_counts = {
            "observed": sum(
                entry.identity.method_id == "observed" for entry in plan.entries
            ),
            "capacity": sum(
                entry.identity.method_id == "capacity-matched-ae"
                for entry in plan.entries
            ),
            "maskimpute": sum(
                entry.identity.method_id == "maskimpute" for entry in plan.entries
            ),
            "comparators": sum(
                entry.identity.configuration_kind == "comparator_tuning"
                for entry in plan.entries
            ),
        }
        if component_counts != {
            "observed": 16,
            "capacity": 48,
            "maskimpute": 1_200,
            "comparators": 1_632,
        }:
            raise RunnerContractError("direct development plan denominator differs")
        if len(plan.configurations) != 61:
            raise RunnerContractError(
                "direct development configuration denominator differs"
            )
        comparator_configurations = tuple(
            configuration
            for configuration in plan.configurations
            if configuration.configuration_kind == "comparator_tuning"
        )
        expected_pairs = tuple(
            (row.method_id, row.configuration_id)
            for row in authority.comparator_tuning.configurations
        )
        if (
            tuple(
                (configuration.method.method_id, configuration.configuration_id)
                for configuration in comparator_configurations
            )
            != expected_pairs
        ):
            raise RunnerContractError(
                "direct comparator configuration order differs from authority"
            )
        for configuration in comparator_configurations:
            positions = [
                index
                for index, entry in enumerate(plan.entries)
                if entry.identity.method_id == configuration.method.method_id
                and entry.identity.configuration_id == configuration.configuration_id
            ]
            if len(positions) != 48 or positions != list(
                range(positions[0], positions[0] + 48)
            ):
                raise RunnerContractError(
                    "direct comparator plan blocks are not contiguous groups of 48"
                )
    elif (
        len(plan.entries) != 48
        or len(plan.configurations) != 1
        or any(
            entry.identity.method_id != "maskimpute"
            or entry.identity.configuration_kind != "candidate_search"
            for entry in plan.entries
        )
    ):
        raise RunnerContractError("direct revision candidate denominator differs")


def build_direct_competition_plan(
    registry: MethodRegistry,
    datasets: Sequence[DatasetBinding],
    authority: RunnerAuthority,
    prepared_datasets: Sequence[PreparedDataset],
) -> DirectCompetitionPlan:
    """Build the complete direct-identity development denominator."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if not isinstance(authority, RunnerAuthority):
        raise TypeError("authority must be a RunnerAuthority")
    dataset_values = tuple(datasets)
    prepared_values = tuple(prepared_datasets)
    if len(dataset_values) != 16 or not all(
        isinstance(binding, DatasetBinding) for binding in dataset_values
    ):
        raise RunnerContractError(
            "direct competition planning requires exactly 16 dataset bindings"
        )
    if len(prepared_values) != len(dataset_values) or not all(
        isinstance(prepared, PreparedDataset) for prepared in prepared_values
    ):
        raise RunnerContractError(
            "direct competition planning requires one prepared input per dataset"
        )
    if any(
        prepared.binding != binding
        for binding, prepared in zip(dataset_values, prepared_values, strict=True)
    ):
        raise RunnerContractError(
            "direct prepared inputs differ from their ordered dataset bindings"
        )
    descriptors = tuple(describe_prepared_input(value) for value in prepared_values)
    draw_indices = tuple(
        _evaluator_seed_and_draw(value)[1] for value in prepared_values
    )
    planned_specs, configurations, configurations_by_method = _configuration_grid(
        registry,
        authority,
    )
    entries: list[DirectPlanEntry] = []
    ordinal = 0
    for spec in planned_specs:
        seeds: tuple[int | None, ...] = (
            DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
        )
        for configuration in configurations_by_method[spec.id]:
            if configuration.requires_calibration and not authority.maskimpute_ready:
                blocked_reason = "count_score_or_calibration_authority_pending"
            elif (
                configuration.requires_count_score
                and authority.count_score_manifest_status != "ready"
            ):
                blocked_reason = "count_score_authority_pending"
            else:
                blocked_reason = None
            for binding, descriptor, draw_index in zip(
                dataset_values,
                descriptors,
                draw_indices,
                strict=True,
            ):
                for seed in seeds:
                    ordinal += 1
                    identity = ComparatorRunIdentity(
                        workflow_schema=_WORKFLOW_SCHEMA,
                        authority_revision=(
                            authority.comparator_tuning_reference.authority_revision
                        ),
                        ordinal=ordinal,
                        method=configuration.method,
                        configuration_id=configuration.configuration_id,
                        configuration_kind=configuration.configuration_kind,
                        configuration_payload=configuration.payload,
                        dataset_id=binding.dataset_id,
                        mechanism=binding.mechanism,
                        biological_id=binding.biological_id,
                        technical_view=binding.technical_view,
                        mask_seed=descriptor.mask_seed,
                        model_seed=seed,
                        draw_index=draw_index,
                    )
                    entries.append(
                        DirectPlanEntry(
                            run_id=direct_run_id(identity),
                            identity=identity,
                            preflight_status=(
                                "blocked_authority"
                                if blocked_reason is not None
                                else "planned"
                            ),
                            preflight_reason=blocked_reason,
                            requires_count_score=(configuration.requires_count_score),
                            requires_calibration=configuration.requires_calibration,
                        )
                    )
    plan = DirectCompetitionPlan(
        schema_version=1,
        identity_mode="direct-v1",
        authority_revision=authority.comparator_tuning_reference.authority_revision,
        inputs=descriptors,
        entries=tuple(entries),
        configurations=configurations,
    )
    _validate_direct_plan(plan, authority)
    return plan


def _direct_plan_to_json(plan: DirectCompetitionPlan) -> dict[str, object]:
    encoded = direct_json_value(plan)
    if not isinstance(encoded, dict):
        raise AssertionError("direct plan encoding must produce an object")
    return encoded


__all__ = [
    "ComparatorRunIdentity",
    "DirectAuthorizedConfiguration",
    "DirectCompetitionPlan",
    "DirectPlanEntry",
    "PreparedInputDescriptor",
    "build_direct_competition_plan",
    "describe_prepared_input",
    "direct_json_value",
    "direct_run_id",
]
