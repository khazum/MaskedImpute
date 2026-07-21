"""Strict authority contracts for development-only comparator tuning."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence, TypeAlias

import numpy as np

from .methods import (
    AFMFConfig,
    ALRAConfig,
    BiAEImputeConfig,
    DCAConfig,
    MAGICConfig,
    SAVERConfig,
    SCCRConfig,
    SCSDaeConfig,
    SCVIConfig,
    SCZivaConfig,
    CovariateColumn,
    MethodInput,
    load_method_registry,
)
from .methods import MethodRegistry
from .methods.base import MethodSpec
from .direct_values import direct_equal, direct_json_value, freeze_direct_mapping


DEVELOPMENT_MAX_LOG_RECEIPT_BYTES = 64 * 1024
DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES = 64 * 1024
DEVELOPMENT_MAX_RECORD_BYTES = 64 * 1024
DEVELOPMENT_MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
DEVELOPMENT_STORAGE_RESERVE_BYTES = 1024**3
COMPARATOR_SELECTION_RELATIVE_PATH = (
    "artifacts/study/development/evaluation/comparator_selection.json"
)
COMPARATOR_SMOKE_RELATIVE_PATH = (
    "artifacts/study/development/evaluation/comparator_smoke.json"
)
AUTHORITY_REVISION = "fair-comparator-direct-v1"
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


class ComparatorTuningError(RuntimeError):
    """Raised when comparator-tuning authority fails closed validation."""


ComparatorAdapterConfig: TypeAlias = (
    ALRAConfig
    | MAGICConfig
    | DCAConfig
    | SCVIConfig
    | SAVERConfig
    | SCZivaConfig
    | AFMFConfig
    | BiAEImputeConfig
    | SCCRConfig
    | SCSDaeConfig
)


_CONFIG_TYPES: Mapping[str, type[ComparatorAdapterConfig]] = MappingProxyType(
    {
        "alra": ALRAConfig,
        "magic": MAGICConfig,
        "dca": DCAConfig,
        "scvi": SCVIConfig,
        "saver": SAVERConfig,
        "scziva": SCZivaConfig,
        "afmf": AFMFConfig,
        "biaeimpute": BiAEImputeConfig,
        "sccr": SCCRConfig,
        "scsdae": SCSDaeConfig,
    }
)

EXPECTED_SCOPE = {"data_scope": "development_only", "final_data_used": False}
EXPECTED_MODEL_SEEDS = (42, 43, 44)
EXPECTED_METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
)
EXPECTED_COLLAPSE_ORDER = (
    "mean_model_seeds_within_dataset_view",
    "mean_paired_views_within_biological_draw",
    "retain_biological_draw_units",
)
EXPECTED_SELECTION_TUPLE = (
    "maximum_metric_rank_quarters",
    "sum_metric_rank_quarters",
    "mse_rank_quarters",
    "mse_dropout_rank_quarters",
    "gnrmse_rank_quarters",
    "mse_pre_dropout_zero_rank_quarters",
    "corr_err_rank_quarters",
    "mse_non_dropout_nonzero_rank_quarters",
    "upstream_default_penalty",
    "configuration_id",
)
EXPECTED_BUDGETS = {
    "max_configurations_per_method": 20,
    "gpu_seconds_per_method": 28_800,
    "cpu_seconds_per_method": 86_400,
    "per_run_timeout_seconds": 21_600,
    "max_rss_bytes": 48 * 1024**3,
    "max_gpu_bytes": 14 * 1024**3,
    "intrinsic_terminal_statuses": [
        "failed",
        "timeout",
        "resource_exceeded",
        "unavailable",
    ],
    "blocking_statuses": [
        "budget_exhausted",
        "blocked_authority",
        "infrastructure_error",
    ],
}
EXPECTED_SMOKE = {
    "receipt_path": COMPARATOR_SMOKE_RELATIVE_PATH,
    "cells": 900,
    "genes": 500,
    "model_seed": 42,
    "batch_rule": "alternating_batch-0_batch-1",
    "count_formula": "(17*cell+31*gene+7*(cell^gene))%6",
    "projection_multiplier": 48,
    "output_retention": "discarded_without_evaluator_or_metrics",
}

COMPARATOR_METHOD_IDS = tuple(_CONFIG_TYPES)
_EXPECTED_METHOD_ORDER = COMPARATOR_METHOD_IDS
_EXPECTED_SCHEDULED_SAME_INPUT_IDS = (
    "observed",
    "capacity-matched-ae",
    "alra",
    "magic",
    "dca",
    "scvi",
    "saver",
    "scziva",
    "afmf",
    "biaeimpute",
    "sccr",
    "scsdae",
)
_EXPECTED_REQUIRED_CONTROL_IDS = ("observed", "capacity-matched-ae")
_EXPECTED_ESTABLISHED_COMPARATOR_IDS = ("alra", "magic", "dca", "scvi", "saver")
_EXPECTED_MODERN_CORE_IDS = ("scziva", "afmf", "biaeimpute", "sccr")
_EXPECTED_PARETO_RULE = "all_metric_medians_no_worse_and_one_strictly_better"
_EXPECTED_RANK_RULE = "pareto_only_average_unit_ranks_median_quarters"
_EXPECTED_READINESS = {
    "minimum_required_controls_complete": 2,
    "minimum_established_comparators_selectable": 5,
    "minimum_modern_core_selectable": 3,
}
_EXPECTED_STORAGE = {
    "max_log_receipt_bytes": DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
    "max_executor_receipt_bytes": DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
    "max_record_bytes": DEVELOPMENT_MAX_RECORD_BYTES,
    "max_checkpoint_bytes": DEVELOPMENT_MAX_CHECKPOINT_BYTES,
    "reserve_bytes": DEVELOPMENT_STORAGE_RESERVE_BYTES,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_payload(config: ComparatorAdapterConfig) -> dict[str, object]:
    value = asdict(config)
    if isinstance(config, DCAConfig):
        value["hidden_size"] = list(config.hidden_size)
    return value


def _primitive_type_matches(observed: object, default: object) -> bool:
    if default is None:
        return observed is None
    if type(default) is float:
        return type(observed) is float and math.isfinite(observed)
    return type(observed) is type(default)


def encode_comparator_configuration(
    config: ComparatorAdapterConfig,
) -> dict[str, object]:
    """Encode one exact supported comparator adapter dataclass as JSON data."""

    if type(config) not in set(_CONFIG_TYPES.values()):
        raise TypeError("config must be an exact comparator adapter dataclass")
    return _json_payload(config)


def decode_comparator_configuration(
    method_id: str,
    payload: Mapping[str, object],
) -> ComparatorAdapterConfig:
    """Decode one closed JSON payload into its exact adapter dataclass."""

    config_type = _CONFIG_TYPES.get(method_id)
    if config_type is None or type(payload) not in {dict, MappingProxyType}:
        raise ComparatorTuningError("comparator method or payload is invalid")
    defaults = _json_payload(config_type())
    observed = dict(payload)
    if set(observed) != set(defaults):
        raise ComparatorTuningError(
            "comparator payload differs from its complete field set"
        )
    constructor = _decode_closed_primitive_fields(method_id, observed, defaults)
    decoded = config_type(**constructor)
    if encode_comparator_configuration(decoded) != observed:
        raise ComparatorTuningError(
            "comparator payload changed during typed normalization"
        )
    return decoded


def _decode_closed_primitive_fields(
    method_id: str,
    observed: Mapping[str, object],
    defaults: Mapping[str, object],
) -> dict[str, object]:
    constructor: dict[str, object] = {}
    for name, default in defaults.items():
        value = observed[name]
        if method_id == "dca" and name == "hidden_size":
            if (
                type(value) is not list
                or not value
                or any(type(item) is not int or item <= 0 for item in value)
            ):
                raise ComparatorTuningError(
                    "DCA hidden_size must be a positive-integer JSON array"
                )
            constructor[name] = tuple(value)
            continue
        if type(value) is float and (
            not math.isfinite(value)
            or (value == 0.0 and math.copysign(1.0, value) < 0.0)
        ):
            raise ComparatorTuningError(
                f"comparator field {name} has an invalid float value"
            )
        if not _primitive_type_matches(value, default):
            raise ComparatorTuningError(
                f"comparator field {name} has the wrong primitive type"
            )
        constructor[name] = value
    return constructor


_EXPECTED_CONFIGURATION_SPECS: tuple[
    tuple[str, str, bool, ComparatorAdapterConfig], ...
] = (
    ("alra", "alra-default", True, ALRAConfig()),
    ("magic", "magic-t03", True, MAGICConfig()),
    ("magic", "magic-t01", False, MAGICConfig(diffusion_time=1)),
    ("magic", "magic-t05", False, MAGICConfig(diffusion_time=5)),
    ("magic", "magic-t07", False, MAGICConfig(diffusion_time=7)),
    ("dca", "dca-h64-32-64", True, DCAConfig()),
    (
        "dca",
        "dca-h32-16-32",
        False,
        DCAConfig(hidden_size=(32, 16, 32)),
    ),
    ("dca", "dca-h32-32", False, DCAConfig(hidden_size=(32, 32))),
    ("dca", "dca-h64-64", False, DCAConfig(hidden_size=(64, 64))),
    ("scvi", "scvi-z10", True, SCVIConfig()),
    ("scvi", "scvi-z05", False, SCVIConfig(n_latent=5)),
    ("scvi", "scvi-z20", False, SCVIConfig(n_latent=20)),
    ("scvi", "scvi-z30", False, SCVIConfig(n_latent=30)),
    ("saver", "saver-default", True, SAVERConfig()),
    ("scziva", "scziva-tau-0p001", True, SCZivaConfig()),
    ("scziva", "scziva-tau-0p0001", False, SCZivaConfig(tau=0.0001)),
    ("scziva", "scziva-tau-0p01", False, SCZivaConfig(tau=0.01)),
    ("scziva", "scziva-tau-0p05", False, SCZivaConfig(tau=0.05)),
    ("afmf", "afmf-sigma-3", True, AFMFConfig()),
    ("afmf", "afmf-sigma-1", False, AFMFConfig(sigma=1.0)),
    ("afmf", "afmf-sigma-2", False, AFMFConfig(sigma=2.0)),
    ("afmf", "afmf-sigma-4", False, AFMFConfig(sigma=4.0)),
    ("biaeimpute", "biaeimpute-z128", True, BiAEImputeConfig()),
    (
        "biaeimpute",
        "biaeimpute-z32",
        False,
        BiAEImputeConfig(latent_size=32),
    ),
    (
        "biaeimpute",
        "biaeimpute-z64",
        False,
        BiAEImputeConfig(latent_size=64),
    ),
    (
        "biaeimpute",
        "biaeimpute-z256",
        False,
        BiAEImputeConfig(latent_size=256),
    ),
    ("sccr", "sccr-k15", True, SCCRConfig()),
    ("sccr", "sccr-k05", False, SCCRConfig(neighbors=5)),
    ("sccr", "sccr-k10", False, SCCRConfig(neighbors=10)),
    ("sccr", "sccr-k30", False, SCCRConfig(neighbors=30)),
    ("scsdae", "scsdae-zero-1", True, SCSDaeConfig()),
    (
        "scsdae",
        "scsdae-zero-0p25",
        False,
        SCSDaeConfig(zero_loss_weight=0.25),
    ),
    (
        "scsdae",
        "scsdae-zero-0p5",
        False,
        SCSDaeConfig(zero_loss_weight=0.5),
    ),
    (
        "scsdae",
        "scsdae-zero-0p75",
        False,
        SCSDaeConfig(zero_loss_weight=0.75),
    ),
)
_EXPECTED_CONFIGURATION_ORDER = tuple(
    (method_id, configuration_id)
    for method_id, configuration_id, _is_default, _config in _EXPECTED_CONFIGURATION_SPECS
)
_EXPECTED_CONFIGURATION_PAYLOADS = MappingProxyType(
    {
        (method_id, configuration_id): _canonical_bytes(_json_payload(config)).decode(
            "utf-8"
        )
        for method_id, configuration_id, _is_default, config in _EXPECTED_CONFIGURATION_SPECS
    }
)


@dataclass(frozen=True, slots=True)
class ComparatorConfiguration:
    """Immutable identity and canonical payload for one comparator setting."""

    method_id: str
    configuration_id: str
    payload_json: str
    is_upstream_default: bool

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.payload_json))

    def decode(self) -> ComparatorAdapterConfig:
        return decode_comparator_configuration(self.method_id, self.payload)


@dataclass(frozen=True, slots=True)
class ComparatorTuningAuthority:
    """Validated immutable development-only comparator-tuning authority."""

    schema_version: int
    contract_id: str
    authority_revision: str
    method_order: tuple[str, ...]
    configurations: tuple[ComparatorConfiguration, ...]
    scheduled_same_input_ids: tuple[str, ...]
    required_control_ids: tuple[str, ...]
    established_comparator_ids: tuple[str, ...]
    modern_core_ids: tuple[str, ...]
    model_seeds: tuple[int, ...]
    selection_metrics: tuple[str, ...]
    receipt_path: str
    smoke_receipt_path: str

    def configurations_for(self, method_id: str) -> tuple[ComparatorConfiguration, ...]:
        return tuple(row for row in self.configurations if row.method_id == method_id)


@dataclass(frozen=True, slots=True)
class ComparatorAuthorityReference:
    """Readable direct reference to the sole comparator-grid authority."""

    path: str
    schema_version: int
    authority_revision: str


@dataclass(frozen=True, slots=True)
class ComparatorMethodBinding:
    """Closed execution projection copied from one registry method."""

    method_id: str
    execution_scope: str
    integration_status: str
    adapter_key: str
    environment_id: str
    environment_status: str
    source_kind: str
    source_url: str | None
    source_revision: str | None
    source_tree: str | None
    source_cache_path: str | None
    source_freeze_binding: str | None
    input_scale: str
    output_scale: str
    stochastic: bool
    seed_policy: str
    gpu_mode: str
    cpu_cores: int
    timeout_seconds: int
    max_rss_gib: int | float
    max_gpu_gib: int | float
    preserves_observed_positives: bool


@dataclass(frozen=True, slots=True)
class BoundComparatorConfiguration:
    """One authoritative comparator setting and its direct method projection."""

    configuration: ComparatorConfiguration
    authority_reference: ComparatorAuthorityReference
    method: ComparatorMethodBinding


SELECTION_METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
)
_SELECTION_MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
_SELECTION_VIEWS = ("moderate", "severe")
_SELECTION_INTRINSIC_STATUSES = frozenset(
    {"failed", "timeout", "resource_exceeded", "unavailable"}
)
_SELECTION_IDENTITY_KEYS = frozenset(
    {
        "workflow_schema",
        "authority_revision",
        "ordinal",
        "method",
        "configuration_id",
        "configuration_kind",
        "configuration_payload",
        "dataset_id",
        "mechanism",
        "biological_id",
        "technical_view",
        "mask_seed",
        "model_seed",
        "draw_index",
    }
)
_SELECTION_METRIC_KEYS = frozenset(
    {"identity", "metric", "value", "n", "status", "reason"}
)


@dataclass(frozen=True, slots=True)
class CollapsedComparatorConfiguration:
    configuration: BoundComparatorConfiguration
    eligible: bool
    eligibility_reason: str | None
    status_counts: Mapping[str, int]
    reason_histogram: Mapping[str, int]
    unit_ids: Mapping[str, tuple[str, ...]]
    unit_values: Mapping[str, tuple[float, ...]]
    unit_counts: Mapping[str, int]
    metric_medians: Mapping[str, float]

    @property
    def method_id(self) -> str:
        return self.configuration.configuration.method_id

    @property
    def configuration_id(self) -> str:
        return self.configuration.configuration.configuration_id


@dataclass(frozen=True, slots=True)
class RankedComparatorConfiguration:
    configuration: BoundComparatorConfiguration
    metric_rank_quarters: Mapping[str, int]
    selection_tuple: tuple[int, int, int, int, int, int, int, int, int, str]

    @property
    def configuration_id(self) -> str:
        return self.configuration.configuration.configuration_id


@dataclass(frozen=True, slots=True)
class ComparatorMethodSelection:
    method_id: str
    collapsed_rows: tuple[CollapsedComparatorConfiguration, ...]
    pareto_rows: tuple[RankedComparatorConfiguration, ...]
    selected_configuration_id: str | None

    @property
    def configuration_ids(self) -> tuple[str, ...]:
        return tuple(row.configuration_id for row in self.collapsed_rows)

    @property
    def eligible_configuration_ids(self) -> tuple[str, ...]:
        return tuple(
            row.configuration_id for row in self.collapsed_rows if row.eligible
        )

    @property
    def pareto_configuration_ids(self) -> tuple[str, ...]:
        return tuple(row.configuration_id for row in self.pareto_rows)

    def configuration(
        self,
        configuration_id: str,
    ) -> CollapsedComparatorConfiguration:
        matches = tuple(
            row
            for row in self.collapsed_rows
            if row.configuration_id == configuration_id
        )
        if len(matches) != 1:
            raise ComparatorTuningError(
                "selected comparator configuration does not resolve exactly"
            )
        return matches[0]


def _selection_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _validate_bound_selection_configuration(
    bound: BoundComparatorConfiguration,
) -> None:
    if not isinstance(bound, BoundComparatorConfiguration):
        raise TypeError("configuration must be a BoundComparatorConfiguration")
    configuration = bound.configuration
    reference = bound.authority_reference
    method = bound.method
    canonical_authority = _canonical_comparator_tuning_authority()
    if (
        not isinstance(configuration, ComparatorConfiguration)
        or not isinstance(reference, ComparatorAuthorityReference)
        or not isinstance(method, ComparatorMethodBinding)
        or configuration.method_id != method.method_id
        or reference.path != "study/comparator_tuning.json"
        or reference.schema_version != canonical_authority.schema_version
        or reference.authority_revision != canonical_authority.authority_revision
        or sum(
            direct_equal(configuration, row)
            for row in canonical_authority.configurations
        )
        != 1
    ):
        raise ComparatorTuningError("bound comparator identity is invalid")
    try:
        payload = dict(configuration.payload)
        decoded = configuration.decode()
    except (
        ComparatorTuningError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise ComparatorTuningError("bound comparator payload is invalid") from error
    if (
        configuration.payload_json != _canonical_bytes(payload).decode("utf-8")
        or encode_comparator_configuration(decoded) != payload
    ):
        raise ComparatorTuningError("bound comparator payload differs")


def _selection_record_identity(record: object) -> Mapping[str, object]:
    if not isinstance(record, Mapping) or set(record) != {
        "run",
        "metrics",
        "p_pre_zero_evidence",
    }:
        raise ComparatorTuningError("comparator selection record schema differs")
    run = record.get("run")
    if not isinstance(run, Mapping):
        raise ComparatorTuningError("comparator selection run is invalid")
    identity = run.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != _SELECTION_IDENTITY_KEYS:
        raise ComparatorTuningError("comparator selection identity schema differs")
    method = identity.get("method")
    if not isinstance(method, Mapping) or not isinstance(method.get("method_id"), str):
        raise ComparatorTuningError("comparator selection method identity is invalid")
    return identity


def _selection_method_binding(value: object) -> ComparatorMethodBinding:
    expected_keys = {item.name for item in fields(ComparatorMethodBinding)}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ComparatorTuningError("comparator selection method identity differs")
    try:
        method = ComparatorMethodBinding(**dict(value))
    except TypeError as error:
        raise ComparatorTuningError(
            "comparator selection method identity differs"
        ) from error
    required_strings = (
        method.method_id,
        method.execution_scope,
        method.integration_status,
        method.adapter_key,
        method.environment_id,
        method.environment_status,
        method.source_kind,
        method.input_scale,
        method.output_scale,
        method.seed_policy,
        method.gpu_mode,
    )
    optional_strings = (
        method.source_url,
        method.source_revision,
        method.source_tree,
        method.source_cache_path,
        method.source_freeze_binding,
    )
    resource_numbers = (
        method.max_rss_gib,
        method.max_gpu_gib,
    )
    if (
        any(not isinstance(item, str) or not item for item in required_strings)
        or any(
            item is not None and not isinstance(item, str) for item in optional_strings
        )
        or type(method.stochastic) is not bool
        or type(method.preserves_observed_positives) is not bool
        or type(method.cpu_cores) is not int
        or method.cpu_cores <= 0
        or type(method.timeout_seconds) is not int
        or method.timeout_seconds <= 0
        or any(
            isinstance(item, bool)
            or type(item) not in {int, float}
            or not math.isfinite(float(item))
            or item < 0
            for item in resource_numbers
        )
        or not direct_equal(direct_json_value(method), value)
    ):
        raise ComparatorTuningError("comparator selection method identity differs")
    return method


def _validate_selection_identity(
    identity: Mapping[str, object],
    bound: BoundComparatorConfiguration,
) -> ComparatorMethodBinding:
    method = _selection_method_binding(identity.get("method"))
    configuration = bound.configuration
    integers = ("ordinal", "mask_seed", "model_seed", "draw_index")
    if (
        identity.get("workflow_schema") != "maskimpute-fair-comparator-run-v1"
        or identity.get("authority_revision")
        != bound.authority_reference.authority_revision
        or identity.get("configuration_id") != configuration.configuration_id
        or identity.get("configuration_kind") != "comparator_tuning"
        or not direct_equal(
            identity.get("configuration_payload"), configuration.payload
        )
        or not direct_equal(method, bound.method)
        or any(type(identity.get(name)) is not int for name in integers)
        or int(identity["ordinal"]) <= 0
        or int(identity["mask_seed"]) < 0
        or identity.get("model_seed") not in EXPECTED_MODEL_SEEDS
        or int(identity["draw_index"]) < 0
        or identity.get("mechanism") not in _SELECTION_MECHANISMS
        or identity.get("technical_view") not in _SELECTION_VIEWS
        or any(
            not isinstance(identity.get(name), str) or not identity.get(name)
            for name in ("dataset_id", "biological_id")
        )
    ):
        raise ComparatorTuningError("comparator selection identity differs")
    return method


def _selection_metric_rows(
    record: Mapping[str, object],
    identity: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    metrics = record.get("metrics")
    if not isinstance(metrics, (list, tuple)):
        raise ComparatorTuningError("comparator selection metric rows are invalid")
    rows: list[Mapping[str, object]] = []
    for value in metrics:
        if not isinstance(value, Mapping) or set(value) != _SELECTION_METRIC_KEYS:
            raise ComparatorTuningError("comparator selection metric schema differs")
        if not direct_equal(value.get("identity"), identity):
            raise ComparatorTuningError("comparator selection metric identity differs")
        metric = value.get("metric")
        status = value.get("status")
        reason = value.get("reason")
        number = value.get("value")
        denominator = value.get("n")
        if (
            metric not in SELECTION_METRICS
            or status not in {"completed", *_SELECTION_INTRINSIC_STATUSES}
            or type(denominator) is not int
            or denominator < 0
        ):
            raise ComparatorTuningError("comparator selection metric row is invalid")
        if number is not None and (
            type(number) is not float or not math.isfinite(number)
        ):
            if type(number) is float and not math.isfinite(number):
                raise ComparatorTuningError(
                    "comparator selection metric value is nonfinite"
                )
            raise ComparatorTuningError("comparator selection metric value is invalid")
        if number is None:
            if status == "completed" or not isinstance(reason, str) or not reason:
                raise ComparatorTuningError(
                    "comparator selection metric status is inconsistent"
                )
        elif status != "completed" or reason is not None:
            raise ComparatorTuningError(
                "comparator selection metric status is inconsistent"
            )
        rows.append(value)
    names = tuple(row["metric"] for row in rows)
    if len(names) != len(set(names)):
        raise ComparatorTuningError("comparator selection metric is duplicated")
    expected = (
        SELECTION_METRICS
        if identity["mechanism"] == "symsim"
        else tuple(
            metric for metric in SELECTION_METRICS if metric != "mse_pre_dropout_zero"
        )
    )
    if names != expected:
        if set(names) == set(expected):
            raise ComparatorTuningError("comparator selection metric order differs")
        extra_or_missing_prezero = ("mse_pre_dropout_zero" in names) != (
            identity["mechanism"] == "symsim"
        )
        if extra_or_missing_prezero:
            raise ComparatorTuningError(
                "comparator selection metric applicability differs"
            )
        raise ComparatorTuningError("comparator selection metric denominator differs")
    evidence = record.get("p_pre_zero_evidence")
    if (
        not isinstance(evidence, Mapping)
        or type(evidence.get("applicable")) is not bool
    ):
        raise ComparatorTuningError("comparator selection prezero evidence is invalid")
    return tuple(rows)


def _average_rank_twice(target: float, values: Sequence[float]) -> int:
    below = sum(value < target for value in values)
    tied = sum(value == target for value in values)
    return 2 * below + tied + 1


def metric_rank_quarters(
    unit_values: Mapping[str, Sequence[float]],
) -> dict[str, int]:
    if not isinstance(unit_values, Mapping):
        raise TypeError("unit_values must be a mapping")
    ids = tuple(unit_values)
    if (
        not ids
        or any(not isinstance(item, str) or not item for item in ids)
        or any(
            isinstance(unit_values[item], (str, bytes))
            or not isinstance(unit_values[item], Sequence)
            for item in ids
        )
    ):
        raise ComparatorTuningError("rank unit denominator differs")
    sequences = {item: tuple(unit_values[item]) for item in ids}
    counts = {len(values) for values in sequences.values()}
    if len(counts) != 1 or next(iter(counts)) == 0:
        raise ComparatorTuningError("rank unit denominator differs")
    ranks_twice: dict[str, list[int]] = {item: [] for item in ids}
    unit_count = len(sequences[ids[0]])
    for unit in range(unit_count):
        raw_values = [sequences[item][unit] for item in ids]
        if any(
            isinstance(value, bool) or type(value) not in {int, float}
            for value in raw_values
        ):
            raise ComparatorTuningError("rank value is invalid")
        values = [float(value) for value in raw_values]
        if not all(math.isfinite(value) for value in values):
            raise ComparatorTuningError("rank value is nonfinite")
        for item, value in zip(ids, values, strict=True):
            ranks_twice[item].append(_average_rank_twice(value, values))
    result: dict[str, int] = {}
    for item, ranks in ranks_twice.items():
        ordered = sorted(ranks)
        middle = len(ordered) // 2
        result[item] = (
            2 * ordered[middle]
            if len(ordered) % 2
            else ordered[middle - 1] + ordered[middle]
        )
    return result


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def collapse_comparator_configuration(
    configuration: BoundComparatorConfiguration,
    records: Sequence[Mapping[str, object]],
) -> CollapsedComparatorConfiguration:
    _validate_bound_selection_configuration(configuration)
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    values = tuple(records)
    if len(values) != 48:
        raise ComparatorTuningError(
            "comparator selection unit grid must contain exactly 48 records"
        )

    parsed: list[tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    ordinal_values: set[int] = set()
    cell_values: set[tuple[str, int]] = set()
    metric_row_count = 0
    for record in values:
        identity = _selection_record_identity(record)
        _validate_selection_identity(identity, configuration)
        ordinal = int(identity["ordinal"])
        cell = (str(identity["dataset_id"]), int(identity["model_seed"]))
        if ordinal in ordinal_values or cell in cell_values:
            raise ComparatorTuningError(
                "comparator selection unit grid contains a duplicate cell"
            )
        ordinal_values.add(ordinal)
        cell_values.add(cell)
        run = record["run"]
        assert isinstance(run, Mapping)
        status = run.get("status")
        reason = run.get("reason")
        if status not in {"completed", *_SELECTION_INTRINSIC_STATUSES}:
            raise ComparatorTuningError(
                "comparator selection contains a blocking run status"
            )
        if (status == "completed") != (reason is None) or (
            reason is not None and (not isinstance(reason, str) or not reason)
        ):
            raise ComparatorTuningError(
                "comparator selection run status is inconsistent"
            )
        metric_rows = _selection_metric_rows(record, identity)
        if status == "completed":
            if any(
                row["status"] not in {"completed", "unavailable"} for row in metric_rows
            ):
                raise ComparatorTuningError(
                    "comparator selection metric status differs from run"
                )
        elif any(
            row["status"] != status
            or row["reason"] != reason
            or row["value"] is not None
            for row in metric_rows
        ):
            raise ComparatorTuningError(
                "comparator selection metric status differs from run"
            )
        status_counts[str(status)] += 1
        if reason is not None:
            reason_counts[reason] += 1
        for row in metric_rows:
            metric_reason = row["reason"]
            if metric_reason is not None:
                reason_counts[str(metric_reason)] += 1
        metric_row_count += len(metric_rows)
        parsed.append((identity, metric_rows))
    if len(cell_values) != 48:
        raise ComparatorTuningError(
            "comparator selection unit grid differs from 48 unique cells"
        )
    if metric_row_count != 252:
        raise ComparatorTuningError(
            "comparator selection metric denominator differs from 252 rows"
        )

    by_dataset: dict[
        str,
        list[tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]],
    ] = {}
    for item in parsed:
        by_dataset.setdefault(str(item[0]["dataset_id"]), []).append(item)
    if len(by_dataset) != 16:
        raise ComparatorTuningError(
            "comparator selection unit grid must contain 16 datasets"
        )
    dataset_meta: dict[str, tuple[str, str, str, int, int]] = {}
    dataset_metric_values: dict[tuple[str, str], float] = {}
    for dataset_id, dataset_rows in by_dataset.items():
        if len(dataset_rows) != 3 or {
            item[0]["model_seed"] for item in dataset_rows
        } != set(EXPECTED_MODEL_SEEDS):
            raise ComparatorTuningError(
                "comparator selection unit grid must contain three model seeds"
            )
        first = dataset_rows[0][0]
        metadata = (
            str(first["mechanism"]),
            str(first["biological_id"]),
            str(first["technical_view"]),
            int(first["mask_seed"]),
            int(first["draw_index"]),
        )
        if any(
            (
                str(item[0]["mechanism"]),
                str(item[0]["biological_id"]),
                str(item[0]["technical_view"]),
                int(item[0]["mask_seed"]),
                int(item[0]["draw_index"]),
            )
            != metadata
            for item in dataset_rows
        ):
            raise ComparatorTuningError(
                "comparator selection unit grid dataset identity differs"
            )
        dataset_meta[dataset_id] = metadata
        for metric in SELECTION_METRICS:
            matching = tuple(
                row
                for _identity, metric_rows in dataset_rows
                for row in metric_rows
                if row["metric"] == metric
            )
            if not matching:
                continue
            if len(matching) != 3:
                raise ComparatorTuningError(
                    "comparator selection metric seed denominator differs"
                )
            if all(row["status"] == "completed" for row in matching):
                numeric = tuple(float(row["value"]) for row in matching)
                dataset_metric_values[(dataset_id, metric)] = math.fsum(numeric) / 3.0

    unit_views: dict[tuple[str, str], dict[str, str]] = {}
    unit_draw_indexes: dict[tuple[str, str], int] = {}
    for dataset_id, metadata in dataset_meta.items():
        mechanism, biological_id, technical_view, _mask_seed, draw_index = metadata
        unit = (mechanism, biological_id)
        views = unit_views.setdefault(unit, {})
        if technical_view in views:
            raise ComparatorTuningError(
                "comparator selection unit grid has duplicate technical views"
            )
        views[technical_view] = dataset_id
        previous_draw = unit_draw_indexes.setdefault(unit, draw_index)
        if previous_draw != draw_index:
            raise ComparatorTuningError(
                "comparator selection unit grid draw identity differs"
            )
    if (
        len(unit_views) != 8
        or {unit[0] for unit in unit_views} != set(_SELECTION_MECHANISMS)
        or any(set(views) != set(_SELECTION_VIEWS) for views in unit_views.values())
        or any(
            sum(unit[0] == mechanism for unit in unit_views) != 2
            for mechanism in _SELECTION_MECHANISMS
        )
    ):
        raise ComparatorTuningError(
            "comparator selection unit grid differs from eight paired draws"
        )
    ordered_units = tuple(
        sorted(
            unit_views,
            key=lambda unit: (
                _SELECTION_MECHANISMS.index(unit[0]),
                unit_draw_indexes[unit],
                unit[1],
            ),
        )
    )

    unit_ids: dict[str, tuple[str, ...]] = {}
    unit_values: dict[str, tuple[float, ...]] = {}
    unit_counts: dict[str, int] = {}
    metric_medians: dict[str, float] = {}
    for metric in SELECTION_METRICS:
        metric_units = (
            tuple(unit for unit in ordered_units if unit[0] == "symsim")
            if metric == "mse_pre_dropout_zero"
            else ordered_units
        )
        collapsed: list[tuple[str, float]] = []
        for unit in metric_units:
            views = unit_views[unit]
            moderate = dataset_metric_values.get((views["moderate"], metric))
            severe = dataset_metric_values.get((views["severe"], metric))
            if moderate is not None and severe is not None:
                collapsed.append((f"{unit[0]}:{unit[1]}", (moderate + severe) / 2.0))
        unit_ids[metric] = tuple(item[0] for item in collapsed)
        unit_values[metric] = tuple(item[1] for item in collapsed)
        unit_counts[metric] = len(collapsed)
        if len(collapsed) == len(metric_units):
            metric_medians[metric] = _median(tuple(item[1] for item in collapsed))
    expected_counts = {
        metric: 2 if metric == "mse_pre_dropout_zero" else 8
        for metric in SELECTION_METRICS
    }
    eligible = (
        status_counts == Counter({"completed": 48})
        and all(
            unit_counts[metric] == expected_counts[metric]
            for metric in SELECTION_METRICS
        )
        and set(metric_medians) == set(SELECTION_METRICS)
    )
    return CollapsedComparatorConfiguration(
        configuration=configuration,
        eligible=eligible,
        eligibility_reason=None if eligible else "intrinsic_terminal_evidence",
        status_counts=_selection_mapping(dict(status_counts)),
        reason_histogram=_selection_mapping(dict(sorted(reason_counts.items()))),
        unit_ids=_selection_mapping(unit_ids),
        unit_values=_selection_mapping(unit_values),
        unit_counts=_selection_mapping(unit_counts),
        metric_medians=_selection_mapping(metric_medians),
    )


def pareto_configuration_ids(
    rows: Sequence[CollapsedComparatorConfiguration],
) -> tuple[str, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    values = tuple(rows)
    if any(not isinstance(row, CollapsedComparatorConfiguration) for row in values):
        raise TypeError("rows must contain CollapsedComparatorConfiguration values")
    ids = tuple(row.configuration_id for row in values)
    if len(ids) != len(set(ids)):
        raise ComparatorTuningError("collapsed comparator configuration is duplicated")
    eligible = tuple(row for row in values if row.eligible)
    for row in eligible:
        if set(row.metric_medians) != set(SELECTION_METRICS) or any(
            type(row.metric_medians[name]) is not float
            or not math.isfinite(row.metric_medians[name])
            for name in SELECTION_METRICS
        ):
            raise ComparatorTuningError("eligible comparator metric median is invalid")
    retained: list[str] = []
    for target in eligible:
        dominated = any(
            other.configuration_id != target.configuration_id
            and all(
                other.metric_medians[name] <= target.metric_medians[name]
                for name in SELECTION_METRICS
            )
            and any(
                other.metric_medians[name] < target.metric_medians[name]
                for name in SELECTION_METRICS
            )
            for other in eligible
        )
        if not dominated:
            retained.append(target.configuration_id)
    return tuple(retained)


def _ranked_pareto_rows(
    rows: Sequence[CollapsedComparatorConfiguration],
    defaults: Mapping[str, bool],
) -> tuple[RankedComparatorConfiguration, ...]:
    pareto_ids = pareto_configuration_ids(rows)
    if set(defaults) != {row.configuration_id for row in rows} or any(
        type(value) is not bool for value in defaults.values()
    ):
        raise ComparatorTuningError("comparator default mapping differs")
    if not pareto_ids:
        return ()
    by_id = {row.configuration_id: row for row in rows}
    for metric in SELECTION_METRICS:
        expected_ids = by_id[pareto_ids[0]].unit_ids[metric]
        if any(
            by_id[item].unit_ids[metric] != expected_ids
            or len(by_id[item].unit_values[metric]) != len(expected_ids)
            for item in pareto_ids
        ):
            raise ComparatorTuningError(
                "Pareto comparator unit-ID grid differs before ranking"
            )
    metric_ranks = {
        metric: metric_rank_quarters(
            {item: by_id[item].unit_values[metric] for item in pareto_ids}
        )
        for metric in SELECTION_METRICS
    }
    result: list[RankedComparatorConfiguration] = []
    for item in pareto_ids:
        ranks = tuple(metric_ranks[metric][item] for metric in SELECTION_METRICS)
        result.append(
            RankedComparatorConfiguration(
                configuration=by_id[item].configuration,
                metric_rank_quarters=_selection_mapping(
                    dict(zip(SELECTION_METRICS, ranks, strict=True))
                ),
                selection_tuple=(
                    max(ranks),
                    sum(ranks),
                    *ranks,
                    0 if defaults[item] else 1,
                    item,
                ),
            )
        )
    return tuple(result)


def select_one_comparator_method(
    method_id: str,
    records: Sequence[Mapping[str, object]],
    authority: ComparatorTuningAuthority,
) -> ComparatorMethodSelection:
    if not isinstance(method_id, str) or not method_id:
        raise TypeError("method_id must be a nonempty string")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence")
    validate_comparator_tuning_authority(authority)
    authority_rows = authority.configurations_for(method_id)
    if not authority_rows:
        raise ComparatorTuningError("comparator method is not in the tuning authority")
    expected_ids = {row.configuration_id for row in authority_rows}
    grouped: dict[str, list[Mapping[str, object]]] = {
        row.configuration_id: [] for row in authority_rows
    }
    for record in records:
        identity = _selection_record_identity(record)
        method = identity["method"]
        assert isinstance(method, Mapping)
        if method.get("method_id") != method_id:
            continue
        if identity.get("configuration_kind") != "comparator_tuning":
            raise ComparatorTuningError("comparator selection identity kind differs")
        configuration_id = identity.get("configuration_id")
        if configuration_id not in expected_ids:
            raise ComparatorTuningError(
                "comparator selection configuration identity differs"
            )
        assert isinstance(configuration_id, str)
        grouped[configuration_id].append(record)

    collapsed: list[CollapsedComparatorConfiguration] = []
    common_method: ComparatorMethodBinding | None = None
    reference = ComparatorAuthorityReference(
        path="study/comparator_tuning.json",
        schema_version=authority.schema_version,
        authority_revision=authority.authority_revision,
    )
    for row in authority_rows:
        configuration_records = grouped[row.configuration_id]
        if not configuration_records:
            raise ComparatorTuningError(
                "comparator selection unit grid lacks an authority configuration"
            )
        identity = _selection_record_identity(configuration_records[0])
        method = _selection_method_binding(identity["method"])
        if common_method is None:
            common_method = method
        elif not direct_equal(method, common_method):
            raise ComparatorTuningError(
                "comparator selection method identity differs across configurations"
            )
        bound = BoundComparatorConfiguration(
            configuration=row,
            authority_reference=reference,
            method=method,
        )
        collapsed.append(
            collapse_comparator_configuration(bound, configuration_records)
        )
    defaults = {row.configuration_id: row.is_upstream_default for row in authority_rows}
    collapsed_rows = tuple(collapsed)
    ranked_rows = _ranked_pareto_rows(collapsed_rows, defaults)
    selected = (
        None
        if not ranked_rows
        else min(ranked_rows, key=lambda row: row.selection_tuple).configuration_id
    )
    return ComparatorMethodSelection(
        method_id=method_id,
        collapsed_rows=collapsed_rows,
        pareto_rows=ranked_rows,
        selected_configuration_id=selected,
    )


_SELECTION_CHECKPOINT_RELATIVE_PATH = (
    "artifacts/study/development/competition-reconstruction/checkpoint.json"
)
_SELECTION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "data_scope",
        "final_data_used",
        "authority_reference",
        "plan_snapshot",
        "input_descriptors",
        "checkpoint_path",
        "scheduled_tuning_records",
        "control_records",
        "model_seeds",
        "selection_metrics",
        "methods",
        "controls",
        "scheduled_same_input_ids",
        "required_control_ids",
        "established_comparator_ids",
        "modern_core_ids",
        "readiness",
    }
)
_METHOD_RECEIPT_KEYS = frozenset(
    {
        "method",
        "selection_status",
        "configuration_order",
        "terminal_status_counts",
        "reason_histogram",
        "configurations",
        "pareto_configuration_ids",
        "selected_configuration_id",
        "selected_configuration",
        "nonexecution_identity",
    }
)
_CONFIGURATION_RECEIPT_KEYS = frozenset(
    {
        "configuration",
        "is_upstream_default",
        "terminal_status_counts",
        "reason_histogram",
        "eligible",
        "eligibility_reason",
        "unit_ids",
        "unit_values",
        "unit_counts",
        "metric_medians",
        "pareto_member",
        "metric_rank_quarters",
        "selection_tuple",
    }
)
_NONEXECUTION_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "authority_reference",
        "method",
        "selection_receipt_namespace",
        "configuration_terminal_denominator",
    }
)
_DIRECT_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "identity_mode",
        "authority_revision",
        "plan_snapshot",
        "input_descriptors",
        "planned_run_count",
        "status",
        "evaluation_scope",
        "comparator_selection_status",
        "selection_complete",
        "selection_blockers",
        "records",
        "budget",
        "storage_preflight",
        "remaining_storage_preflight",
    }
)
_DIRECT_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "identity_mode",
        "authority_revision",
        "inputs",
        "entries",
        "configurations",
        "comparator_smoke_receipt",
        "comparator_smoke_receipt_bytes",
    }
)
_DIRECT_INPUT_KEYS = frozenset(
    {
        "dataset_id",
        "source_reference",
        "preprocessing_revision",
        "shape",
        "dtype",
        "cell_ids",
        "gene_ids",
        "batch_labels",
        "total_count",
        "nonzero_count",
        "minimum",
        "maximum",
        "mechanism",
        "mask_seed",
        "technical_view",
    }
)
_DIRECT_CONFIGURATION_KEYS = frozenset(
    {
        "method",
        "configuration_id",
        "configuration_kind",
        "payload",
        "requires_count_score",
        "requires_calibration",
    }
)
_DIRECT_ENTRY_KEYS = frozenset(
    {
        "run_id",
        "identity",
        "preflight_status",
        "preflight_reason",
        "requires_count_score",
        "requires_calibration",
    }
)
_DIRECT_RECORD_KEYS = frozenset({"run", "metrics", "p_pre_zero_evidence"})
_DIRECT_RUN_KEYS = frozenset(
    {
        "run_id",
        "identity",
        "status",
        "reason",
        "runtime_seconds",
        "peak_rss_bytes",
        "peak_gpu_bytes",
        "rss_measurement",
        "gpu_measurement",
        "excluded_cell_count",
        "excluded_cell_ids",
        "retained_cell_count",
        "retained_cell_ids",
        "retained_gene_count",
        "observed_zero_count",
        "stdout",
        "stderr",
    }
)
_DIRECT_LOG_KEYS = frozenset(
    {"stream", "original_byte_count", "capture_policy", "terminal_reason"}
)
_DIRECT_PREZERO_KEYS = frozenset(
    {
        "applicable",
        "status",
        "reason",
        "shape",
        "dtype",
        "encoding",
        "path",
        "compressed_byte_count",
    }
)
_BLOCKING_DIRECT_STATUSES = frozenset(
    {"budget_exhausted", "blocked_authority", "infrastructure_error"}
)


@dataclass(frozen=True, slots=True)
class ComparatorReadiness:
    """Record-derived publication readiness for the fixed comparison population."""

    status: str
    blocker_codes: tuple[str, ...]
    required_controls_complete: bool
    established_selectable_ids: tuple[str, ...]
    modern_selectable_ids: tuple[str, ...]
    modern_selectable_count: int
    ready_comparison_population_ids: tuple[str, ...]


def _authority_reference(
    authority: ComparatorTuningAuthority,
) -> ComparatorAuthorityReference:
    return ComparatorAuthorityReference(
        path="study/comparator_tuning.json",
        schema_version=authority.schema_version,
        authority_revision=authority.authority_revision,
    )


def _nonexecution_identity(
    *,
    method: ComparatorMethodBinding,
    authority_reference: ComparatorAuthorityReference,
    configuration_terminal_denominator: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "authority_reference": direct_json_value(authority_reference),
        "method": direct_json_value(method),
        "selection_receipt_namespace": "maskimpute-comparator-selection-v1",
        "configuration_terminal_denominator": [
            direct_json_value(row) for row in configuration_terminal_denominator
        ],
    }


def _readiness(
    authority: ComparatorTuningAuthority,
    control_statuses: Mapping[str, str],
    selectable_ids: set[str],
    blocking_status_count: int,
) -> ComparatorReadiness:
    controls_complete = not any(
        control_statuses.get(item) != "completed"
        for item in authority.required_control_ids
    )
    blockers: list[str] = []
    if not controls_complete:
        blockers.append("required_control_incomplete")
    if not set(authority.established_comparator_ids) <= selectable_ids:
        blockers.append("established_comparator_unselectable")
    modern = tuple(item for item in authority.modern_core_ids if item in selectable_ids)
    if len(modern) < 3:
        blockers.append("fewer_than_three_modern_core_selectable")
    if blocking_status_count:
        blockers.append("nonscientific_incomplete_outcome_present")
    return ComparatorReadiness(
        status="ready" if not blockers else "blocked",
        blocker_codes=tuple(blockers),
        required_controls_complete=controls_complete,
        established_selectable_ids=tuple(
            item
            for item in authority.established_comparator_ids
            if item in selectable_ids
        ),
        modern_selectable_ids=modern,
        modern_selectable_count=len(modern),
        ready_comparison_population_ids=tuple(
            item
            for item in authority.scheduled_same_input_ids
            if item in authority.required_control_ids or item in selectable_ids
        ),
    )


def _require_selection_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or type(value) not in {int, float}
        or not math.isfinite(float(value))
        or value < 0
        or (value == 0 and math.copysign(1.0, float(value)) < 0.0)
    ):
        raise ComparatorTuningError(f"{name} is invalid")
    return float(value)


def _validate_receipt_input_descriptor(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _DIRECT_INPUT_KEYS:
        raise ComparatorTuningError("comparator selection input descriptor differs")
    for name in ("dataset_id", "source_reference", "preprocessing_revision", "dtype"):
        if not isinstance(value.get(name), str) or not value.get(name):
            raise ComparatorTuningError("comparator selection input descriptor differs")
    if (
        value.get("mechanism") not in _SELECTION_MECHANISMS
        or value.get("technical_view") not in _SELECTION_VIEWS
    ):
        raise ComparatorTuningError("comparator selection input descriptor differs")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(item) is not int or item <= 0 for item in shape)
        or type(value.get("mask_seed")) is not int
        or int(value["mask_seed"]) < 0
        or type(value.get("nonzero_count")) is not int
        or int(value["nonzero_count"]) < 0
    ):
        raise ComparatorTuningError("comparator selection input descriptor differs")
    for name in ("cell_ids", "gene_ids", "batch_labels"):
        observed = value.get(name)
        if not isinstance(observed, list) or any(
            not isinstance(item, str) for item in observed
        ):
            raise ComparatorTuningError("comparator selection input descriptor differs")
    if (
        len(value["cell_ids"]) != shape[0]
        or len(value["gene_ids"]) != shape[1]
        or value["batch_labels"]
        and len(value["batch_labels"]) != shape[0]
    ):
        raise ComparatorTuningError("comparator selection input descriptor differs")
    for name in ("total_count", "minimum", "maximum"):
        observed = value.get(name)
        if (
            type(observed) is not float
            or not math.isfinite(observed)
            or observed < 0.0
            or (observed == 0.0 and math.copysign(1.0, observed) < 0.0)
        ):
            raise ComparatorTuningError(
                f"comparator selection input descriptor {name} differs"
            )

    from .fair_comparator_plan import PreparedInputDescriptor

    try:
        descriptor = PreparedInputDescriptor(
            dataset_id=value["dataset_id"],
            source_reference=value["source_reference"],
            preprocessing_revision=value["preprocessing_revision"],
            shape=tuple(value["shape"]),
            dtype=value["dtype"],
            cell_ids=tuple(value["cell_ids"]),
            gene_ids=tuple(value["gene_ids"]),
            batch_labels=tuple(value["batch_labels"]),
            total_count=value["total_count"],
            nonzero_count=value["nonzero_count"],
            minimum=value["minimum"],
            maximum=value["maximum"],
            mechanism=value["mechanism"],
            mask_seed=value["mask_seed"],
            technical_view=value["technical_view"],
        )
    except (TypeError, ValueError) as error:
        raise ComparatorTuningError(
            "comparator selection input descriptor differs"
        ) from error
    encoded = direct_json_value(descriptor)
    if not isinstance(encoded, Mapping) or not direct_equal(encoded, value):
        raise ComparatorTuningError("comparator selection input descriptor differs")
    return encoded


def _validate_selection_plan_snapshot(
    snapshot: object,
    *,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    if not isinstance(snapshot, Mapping) or set(snapshot) != _DIRECT_PLAN_KEYS:
        raise ComparatorTuningError("comparator selection plan snapshot differs")
    if (
        type(snapshot.get("schema_version")) is not int
        or snapshot.get("schema_version") != 1
        or snapshot.get("identity_mode") != "direct-v1"
        or snapshot.get("authority_revision") != authority.authority_revision
    ):
        raise ComparatorTuningError("comparator selection plan snapshot differs")
    inputs = snapshot.get("inputs")
    configurations = snapshot.get("configurations")
    entries = snapshot.get("entries")
    if (
        not isinstance(inputs, list)
        or len(inputs) != 16
        or not isinstance(configurations, list)
        or len(configurations) != 61
        or not isinstance(entries, list)
        or len(entries) != 2_896
    ):
        raise ComparatorTuningError("comparator selection plan denominator differs")
    input_values = tuple(_validate_receipt_input_descriptor(value) for value in inputs)
    input_ids = tuple(str(value["dataset_id"]) for value in input_values)
    if len(set(input_ids)) != 16:
        raise ComparatorTuningError("comparator selection input denominator differs")

    from .fair_comparator_plan import _configuration_grid
    from .runner import load_runner_authority

    _specs, expected_configurations, _by_method = _configuration_grid(
        registry,
        load_runner_authority(),
    )
    expected_configuration_values = direct_json_value(expected_configurations)
    if not direct_equal(configurations, expected_configuration_values):
        raise ComparatorTuningError(
            "comparator selection plan configurations differ from authority"
        )
    for value in configurations:
        if not isinstance(value, Mapping) or set(value) != _DIRECT_CONFIGURATION_KEYS:
            raise ComparatorTuningError(
                "comparator selection plan configuration schema differs"
            )

    smoke = snapshot.get("comparator_smoke_receipt")
    smoke_bytes = snapshot.get("comparator_smoke_receipt_bytes")
    if (
        not isinstance(smoke, Mapping)
        or not isinstance(smoke_bytes, list)
        or any(type(item) is not int or item < 0 or item > 255 for item in smoke_bytes)
    ):
        raise ComparatorTuningError("comparator selection plan smoke evidence differs")
    validate_comparator_smoke_receipt(
        dict(smoke),
        bytes(smoke_bytes),
        authority=authority,
        registry=registry,
    )

    method_fields = {item.name for item in fields(ComparatorMethodBinding)}
    cursor = 0
    dataset_metadata: dict[str, tuple[str, str, int, int]] = {}
    for configuration in configurations:
        assert isinstance(configuration, Mapping)
        method = configuration.get("method")
        if not isinstance(method, Mapping) or set(method) != method_fields:
            raise ComparatorTuningError(
                "comparator selection plan method binding differs"
            )
        method_id = method.get("method_id")
        if not isinstance(method_id, str):
            raise ComparatorTuningError(
                "comparator selection plan method binding differs"
            )
        try:
            spec = registry.by_id(method_id)
        except KeyError as error:
            raise ComparatorTuningError(
                "comparator selection plan method binding differs"
            ) from error
        if not direct_equal(method, direct_json_value(comparator_method_binding(spec))):
            raise ComparatorTuningError(
                "comparator selection plan method binding differs"
            )
        seeds: tuple[int | None, ...] = (
            EXPECTED_MODEL_SEEDS if spec.stochastic else (None,)
        )
        for input_value in input_values:
            for model_seed in seeds:
                if cursor >= len(entries):
                    raise ComparatorTuningError(
                        "comparator selection plan entry denominator differs"
                    )
                entry = entries[cursor]
                cursor += 1
                if not isinstance(entry, Mapping) or set(entry) != _DIRECT_ENTRY_KEYS:
                    raise ComparatorTuningError(
                        "comparator selection plan entry schema differs"
                    )
                identity = entry.get("identity")
                if (
                    not isinstance(identity, Mapping)
                    or set(identity) != _SELECTION_IDENTITY_KEYS
                    or identity.get("workflow_schema")
                    != "maskimpute-fair-comparator-run-v1"
                    or identity.get("authority_revision")
                    != authority.authority_revision
                    or identity.get("ordinal") != cursor
                    or not direct_equal(identity.get("method"), method)
                    or identity.get("configuration_id")
                    != configuration.get("configuration_id")
                    or identity.get("configuration_kind")
                    != configuration.get("configuration_kind")
                    or not direct_equal(
                        identity.get("configuration_payload"),
                        configuration.get("payload"),
                    )
                    or identity.get("dataset_id") != input_value["dataset_id"]
                    or identity.get("mechanism") != input_value["mechanism"]
                    or identity.get("technical_view") != input_value["technical_view"]
                    or identity.get("mask_seed") != input_value["mask_seed"]
                    or identity.get("model_seed") != model_seed
                    or type(identity.get("draw_index")) is not int
                    or int(identity["draw_index"]) <= 0
                    or not isinstance(identity.get("biological_id"), str)
                    or not identity.get("biological_id")
                    or entry.get("preflight_status") != "planned"
                    or entry.get("preflight_reason") is not None
                    or entry.get("requires_count_score")
                    is not configuration.get("requires_count_score")
                    or entry.get("requires_calibration")
                    is not configuration.get("requires_calibration")
                ):
                    raise ComparatorTuningError(
                        "comparator selection plan entry differs"
                    )
                seed = "deterministic" if model_seed is None else f"seed-{model_seed}"
                expected_run_id = (
                    f"run-{cursor:04d}-{method_id}-"
                    f"{str(input_value['dataset_id']).removeprefix('dataset-')}-"
                    f"{seed}-{configuration['configuration_id']}"
                )
                if entry.get("run_id") != expected_run_id:
                    raise ComparatorTuningError(
                        "comparator selection plan run ID differs"
                    )
                metadata = (
                    str(identity["mechanism"]),
                    str(identity["biological_id"]),
                    int(identity["draw_index"]),
                    int(identity["mask_seed"]),
                )
                previous = dataset_metadata.setdefault(
                    str(identity["dataset_id"]), metadata
                )
                if previous != metadata:
                    raise ComparatorTuningError(
                        "comparator selection plan dataset identity differs"
                    )
    if cursor != 2_896:
        raise ComparatorTuningError("comparator selection plan denominator differs")
    return input_values, tuple(entries)


def _validate_selection_checkpoint_record(
    value: object,
    entry: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _DIRECT_RECORD_KEYS:
        raise ComparatorTuningError(
            "comparator selection checkpoint record schema differs"
        )
    expected_identity = entry.get("identity")
    run = value.get("run")
    metrics = value.get("metrics")
    evidence_value = value.get("p_pre_zero_evidence")
    if (
        not isinstance(expected_identity, Mapping)
        or not isinstance(run, Mapping)
        or set(run) != _DIRECT_RUN_KEYS
        or not isinstance(metrics, list)
        or tuple(
            row.get("metric") if isinstance(row, Mapping) else None for row in metrics
        )
        != SELECTION_METRICS
        or any(
            not isinstance(row, Mapping) or set(row) != _SELECTION_METRIC_KEYS
            for row in metrics
        )
        or not isinstance(evidence_value, Mapping)
        or set(evidence_value) != _DIRECT_PREZERO_KEYS
    ):
        raise ComparatorTuningError(
            "comparator selection checkpoint accepted direct record differs"
        )

    from .fair_comparator_execution import (
        DirectLogReceipt,
        DirectMetricRow,
        DirectPreZeroEvidence,
        DirectRunResult,
        _executor_receipt_for_run,
        validate_direct_evidence_semantics,
    )
    from .fair_comparator_plan import ComparatorRunIdentity
    from .runner import RunnerContractError

    try:
        method_value = expected_identity.get("method")
        payload_value = expected_identity.get("configuration_payload")
        if not isinstance(method_value, Mapping) or not isinstance(
            payload_value, Mapping
        ):
            raise TypeError("direct identity projection is invalid")
        identity = ComparatorRunIdentity(
            workflow_schema=expected_identity["workflow_schema"],
            authority_revision=expected_identity["authority_revision"],
            ordinal=expected_identity["ordinal"],
            method=ComparatorMethodBinding(**dict(method_value)),
            configuration_id=expected_identity["configuration_id"],
            configuration_kind=expected_identity["configuration_kind"],
            configuration_payload=freeze_direct_mapping(payload_value),
            dataset_id=expected_identity["dataset_id"],
            mechanism=expected_identity["mechanism"],
            biological_id=expected_identity["biological_id"],
            technical_view=expected_identity["technical_view"],
            mask_seed=expected_identity["mask_seed"],
            model_seed=expected_identity["model_seed"],
            draw_index=expected_identity["draw_index"],
        )
        stdout_value = run.get("stdout")
        stderr_value = run.get("stderr")
        if (
            not isinstance(stdout_value, Mapping)
            or set(stdout_value) != _DIRECT_LOG_KEYS
            or not isinstance(stderr_value, Mapping)
            or set(stderr_value) != _DIRECT_LOG_KEYS
        ):
            raise TypeError("direct log projection is invalid")
        run_result = DirectRunResult(
            run_id=run["run_id"],
            identity=identity,
            status=run["status"],
            reason=run["reason"],
            runtime_seconds=run["runtime_seconds"],
            peak_rss_bytes=run["peak_rss_bytes"],
            peak_gpu_bytes=run["peak_gpu_bytes"],
            rss_measurement=run["rss_measurement"],
            gpu_measurement=run["gpu_measurement"],
            excluded_cell_count=run["excluded_cell_count"],
            excluded_cell_ids=tuple(run["excluded_cell_ids"]),
            retained_cell_count=run["retained_cell_count"],
            retained_cell_ids=tuple(run["retained_cell_ids"]),
            retained_gene_count=run["retained_gene_count"],
            observed_zero_count=run["observed_zero_count"],
            stdout=DirectLogReceipt(**dict(stdout_value)),
            stderr=DirectLogReceipt(**dict(stderr_value)),
        )
        metric_rows = tuple(
            DirectMetricRow(
                identity=identity,
                metric=row["metric"],
                value=row["value"],
                n=row["n"],
                status=row["status"],
                reason=row["reason"],
            )
            for row in metrics
        )
        shape_value = evidence_value.get("shape")
        if shape_value is not None and not isinstance(shape_value, list):
            raise TypeError("direct p_pre_zero shape is invalid")
        evidence = DirectPreZeroEvidence(
            applicable=evidence_value["applicable"],
            status=evidence_value["status"],
            reason=evidence_value["reason"],
            shape=(None if shape_value is None else tuple(shape_value)),
            dtype=evidence_value["dtype"],
            encoding=evidence_value["encoding"],
            path=evidence_value["path"],
            compressed_byte_count=evidence_value["compressed_byte_count"],
        )
        if evidence.path is not None:
            relative = PurePosixPath(evidence.path)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or relative.as_posix() != evidence.path
            ):
                raise RunnerContractError("direct p_pre_zero evidence path is unsafe")
        _executor_receipt_for_run(run_result)
        validate_direct_evidence_semantics(run_result, metric_rows, evidence)
        encoded = {
            "run": run_result.to_dict(),
            "metrics": [row.to_dict() for row in metric_rows],
            "p_pre_zero_evidence": evidence.to_dict(),
        }
        if not direct_equal(encoded, value):
            raise RunnerContractError("direct record re-encoding differs")
        if len(_canonical_bytes(encoded) + b"\n") > DEVELOPMENT_MAX_RECORD_BYTES:
            raise RunnerContractError("development record exceeds its byte bound")
    except (KeyError, RunnerContractError, TypeError, ValueError) as error:
        raise ComparatorTuningError(
            "comparator selection checkpoint accepted direct record differs"
        ) from error
    return encoded


def _replay_selection_budget(
    entries: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    registry: MethodRegistry,
) -> dict[str, object]:
    configurations: dict[str, set[str]] = {}
    consumed: dict[str, float] = {}
    for entry, record in zip(entries, records, strict=True):
        identity = entry["identity"]
        run = record["run"]
        assert isinstance(identity, Mapping) and isinstance(run, Mapping)
        status = str(run["status"])
        if status in _BLOCKING_DIRECT_STATUSES:
            continue
        method = identity["method"]
        assert isinstance(method, Mapping)
        method_id = str(method["method_id"])
        configuration_kind = str(identity["configuration_kind"])
        scope = (
            f"{method_id}:{configuration_kind}"
            if method_id == "maskimpute"
            else method_id
        )
        configuration_id = str(identity["configuration_id"])
        runtime = float(run["runtime_seconds"])
        consumed[scope] = consumed.get(scope, 0.0) + runtime
        if configuration_kind in {"candidate_search", "comparator_tuning"}:
            configurations.setdefault(scope, set()).add(configuration_id)
            if len(configurations[scope]) > 20:
                raise ComparatorTuningError(
                    "comparator selection checkpoint exceeds configuration budget"
                )
        try:
            spec = registry.by_id(method_id)
        except KeyError as error:
            raise ComparatorTuningError(
                "comparator selection checkpoint method differs"
            ) from error
        limit = 28_800.0 if spec.resources.gpu_required else 86_400.0
        if consumed[scope] > limit:
            raise ComparatorTuningError(
                "comparator selection checkpoint exceeds time budget"
            )
    return {
        scope: {
            "configuration_ids": sorted(configurations.get(scope, set())),
            "consumed_seconds": consumed.get(scope, 0.0),
        }
        for scope in sorted(set(configurations) | set(consumed))
    }


def _project_selection_record(
    record: Mapping[str, object],
) -> Mapping[str, object]:
    projected = direct_json_value(record)
    assert isinstance(projected, dict)
    run = projected["run"]
    assert isinstance(run, dict)
    identity = run["identity"]
    assert isinstance(identity, dict)
    if identity["mechanism"] != "symsim":
        metrics = projected["metrics"]
        assert isinstance(metrics, list)
        projected["metrics"] = [
            row for row in metrics if row["metric"] != "mse_pre_dropout_zero"
        ]
    return projected


def _validate_selection_checkpoint(
    checkpoint: object,
    *,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
) -> tuple[
    Mapping[str, object],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    int,
]:
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != _DIRECT_CHECKPOINT_KEYS
    ):
        raise ComparatorTuningError("comparator selection checkpoint schema differs")
    if (
        type(checkpoint.get("schema_version")) is not int
        or checkpoint.get("schema_version") != 1
        or checkpoint.get("identity_mode") != "direct-v1"
        or checkpoint.get("authority_revision") != authority.authority_revision
        or checkpoint.get("planned_run_count") != 2_896
        or type(checkpoint.get("planned_run_count")) is not int
        or checkpoint.get("status") != "completed"
        or checkpoint.get("evaluation_scope") != "reconstruction_only"
        or checkpoint.get("comparator_selection_status")
        != "complete_terminal_denominator"
        or checkpoint.get("selection_complete") is not False
        or checkpoint.get("selection_blockers")
        != [
            "downstream_safety_not_evaluated",
            "null_de_fpr_not_evaluated",
            "orthogonal_endpoints_not_evaluated",
        ]
        or not isinstance(checkpoint.get("storage_preflight"), Mapping)
        or not isinstance(checkpoint.get("remaining_storage_preflight"), Mapping)
    ):
        raise ComparatorTuningError(
            "comparator selection checkpoint completeness differs"
        )
    snapshot = checkpoint.get("plan_snapshot")
    inputs, entries = _validate_selection_plan_snapshot(
        snapshot,
        authority=authority,
        registry=registry,
    )
    if not direct_equal(checkpoint.get("input_descriptors"), list(inputs)):
        raise ComparatorTuningError(
            "comparator selection checkpoint input descriptors differ"
        )
    records = checkpoint.get("records")
    if not isinstance(records, list) or len(records) != 2_896:
        raise ComparatorTuningError(
            "comparator selection checkpoint record denominator differs"
        )
    validated_records = tuple(
        _validate_selection_checkpoint_record(value, entry)
        for value, entry in zip(records, entries, strict=True)
    )
    run_ids = tuple(str(record["run"]["run_id"]) for record in validated_records)
    identity_values = tuple(
        _canonical_bytes(record["run"]["identity"]) for record in validated_records
    )
    if len(set(run_ids)) != 2_896 or len(set(identity_values)) != 2_896:
        raise ComparatorTuningError(
            "comparator selection checkpoint identities are duplicated"
        )
    replayed_budget = _replay_selection_budget(entries, validated_records, registry)
    if not direct_equal(checkpoint.get("budget"), replayed_budget):
        raise ComparatorTuningError(
            "comparator selection checkpoint budget differs from replay"
        )

    tuning_records: list[Mapping[str, object]] = []
    control_records: list[Mapping[str, object]] = []
    blocking_status_count = 0
    for entry, record in zip(entries, validated_records, strict=True):
        identity = entry["identity"]
        run = record["run"]
        assert isinstance(identity, Mapping) and isinstance(run, Mapping)
        status = str(run["status"])
        if status in _BLOCKING_DIRECT_STATUSES:
            blocking_status_count += 1
        method = identity["method"]
        assert isinstance(method, Mapping)
        method_id = str(method["method_id"])
        if identity["configuration_kind"] == "comparator_tuning":
            if status in _BLOCKING_DIRECT_STATUSES:
                raise ComparatorTuningError(
                    "comparator selection checkpoint tuning denominator is blocked"
                )
            tuning_records.append(_project_selection_record(record))
        elif method_id in authority.required_control_ids:
            control_records.append(direct_json_value(record))
    if len(tuning_records) != 1_632 or len(control_records) != 64:
        raise ComparatorTuningError(
            "comparator selection checkpoint scheduled denominator differs"
        )
    assert isinstance(snapshot, Mapping)
    return (
        direct_json_value(snapshot),
        inputs,
        tuple(tuning_records),
        tuple(control_records),
        blocking_status_count,
    )


def _control_receipts(
    authority: ComparatorTuningAuthority,
    control_records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, str]]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        item: [] for item in authority.required_control_ids
    }
    for record in control_records:
        identity = _selection_record_identity(record)
        method = identity.get("method")
        method_id = method.get("method_id") if isinstance(method, Mapping) else None
        if method_id not in grouped:
            raise ComparatorTuningError("comparator selection control identity differs")
        grouped[str(method_id)].append(record)
    expected_counts = {"observed": 16, "capacity-matched-ae": 48}
    controls: dict[str, object] = {}
    statuses: dict[str, str] = {}
    for method_id in authority.required_control_ids:
        rows = grouped[method_id]
        if len(rows) != expected_counts[method_id]:
            raise ComparatorTuningError(
                "comparator selection control denominator differs"
            )
        status_counts: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        run_ids: list[str] = []
        for row in rows:
            run = row["run"]
            assert isinstance(run, Mapping)
            status_counts[str(run["status"])] += 1
            if run["reason"] is not None:
                reasons[str(run["reason"])] += 1
            run_ids.append(str(run["run_id"]))
        status = (
            "completed"
            if status_counts == Counter({"completed": expected_counts[method_id]})
            else "incomplete"
        )
        statuses[method_id] = status
        controls[method_id] = {
            "status": status,
            "terminal_status_counts": dict(sorted(status_counts.items())),
            "reason_histogram": dict(sorted(reasons.items())),
            "run_ids": run_ids,
        }
    return controls, statuses


def _method_receipt(
    selection: ComparatorMethodSelection,
    authority_reference: ComparatorAuthorityReference,
) -> dict[str, object]:
    ranked = {row.configuration_id: row for row in selection.pareto_rows}
    terminal_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    configurations: dict[str, object] = {}
    terminal_denominator: list[Mapping[str, object]] = []
    selected_configuration: BoundComparatorConfiguration | None = None
    for row in selection.collapsed_rows:
        terminal_counts.update(row.status_counts)
        reason_counts.update(row.reason_histogram)
        rank = ranked.get(row.configuration_id)
        configurations[row.configuration_id] = {
            "configuration": direct_json_value(row.configuration),
            "is_upstream_default": (
                row.configuration.configuration.is_upstream_default
            ),
            "terminal_status_counts": direct_json_value(row.status_counts),
            "reason_histogram": direct_json_value(row.reason_histogram),
            "eligible": row.eligible,
            "eligibility_reason": row.eligibility_reason,
            "unit_ids": direct_json_value(row.unit_ids),
            "unit_values": direct_json_value(row.unit_values),
            "unit_counts": direct_json_value(row.unit_counts),
            "metric_medians": direct_json_value(row.metric_medians),
            "pareto_member": rank is not None,
            "metric_rank_quarters": (
                None if rank is None else direct_json_value(rank.metric_rank_quarters)
            ),
            "selection_tuple": (
                None if rank is None else direct_json_value(rank.selection_tuple)
            ),
        }
        terminal_denominator.append(
            {
                "configuration": direct_json_value(row.configuration),
                "terminal_status_counts": direct_json_value(row.status_counts),
                "reason_histogram": direct_json_value(row.reason_histogram),
            }
        )
        if row.configuration_id == selection.selected_configuration_id:
            selected_configuration = row.configuration
    method = selection.collapsed_rows[0].configuration.method
    if selection.selected_configuration_id is None:
        selection_status = "intrinsic_terminal_no_eligible_configuration"
        nonexecution = _nonexecution_identity(
            method=method,
            authority_reference=authority_reference,
            configuration_terminal_denominator=terminal_denominator,
        )
    else:
        if selected_configuration is None:  # pragma: no cover - selector invariant
            raise AssertionError("selected comparator configuration is absent")
        selection_status = "selected"
        nonexecution = None
    return {
        "method": direct_json_value(method),
        "selection_status": selection_status,
        "configuration_order": list(selection.configuration_ids),
        "terminal_status_counts": dict(sorted(terminal_counts.items())),
        "reason_histogram": dict(sorted(reason_counts.items())),
        "configurations": configurations,
        "pareto_configuration_ids": list(selection.pareto_configuration_ids),
        "selected_configuration_id": selection.selected_configuration_id,
        "selected_configuration": (
            None
            if selected_configuration is None
            else direct_json_value(selected_configuration)
        ),
        "nonexecution_identity": nonexecution,
    }


def _build_selection_receipt_from_evidence(
    *,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
    plan_snapshot: Mapping[str, object],
    input_descriptors: Sequence[Mapping[str, object]],
    tuning_records: Sequence[Mapping[str, object]],
    control_records: Sequence[Mapping[str, object]],
    blocking_status_count: int,
    allow_blocked: bool,
) -> dict[str, object]:
    validate_comparator_tuning_authority(authority)
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    reference = _authority_reference(authority)
    methods: dict[str, object] = {}
    selectable_ids: set[str] = set()
    for method_id in authority.method_order:
        expected_method = comparator_method_binding(registry.by_id(method_id))
        selection = select_one_comparator_method(
            method_id,
            tuning_records,
            authority,
        )
        if not direct_equal(
            selection.collapsed_rows[0].configuration.method,
            expected_method,
        ):
            raise ComparatorTuningError(
                "comparator selection method differs from registry"
            )
        method = _method_receipt(selection, reference)
        methods[method_id] = method
        if selection.selected_configuration_id is not None:
            selectable_ids.add(method_id)
    controls, control_statuses = _control_receipts(authority, control_records)
    readiness = _readiness(
        authority,
        control_statuses,
        selectable_ids,
        blocking_status_count,
    )
    if readiness.status != "ready" and not allow_blocked:
        raise ComparatorTuningError(
            "comparator publication readiness is blocked: "
            + ",".join(readiness.blocker_codes)
        )
    result = {
        "schema_version": 1,
        "artifact_type": "maskimpute-comparator-selection-receipt-v1",
        "data_scope": "development_only",
        "final_data_used": False,
        "authority_reference": direct_json_value(reference),
        "plan_snapshot": direct_json_value(plan_snapshot),
        "input_descriptors": direct_json_value(tuple(input_descriptors)),
        "checkpoint_path": _SELECTION_CHECKPOINT_RELATIVE_PATH,
        "scheduled_tuning_records": direct_json_value(tuple(tuning_records)),
        "control_records": direct_json_value(tuple(control_records)),
        "model_seeds": list(authority.model_seeds),
        "selection_metrics": list(authority.selection_metrics),
        "methods": methods,
        "controls": controls,
        "scheduled_same_input_ids": list(authority.scheduled_same_input_ids),
        "required_control_ids": list(authority.required_control_ids),
        "established_comparator_ids": list(authority.established_comparator_ids),
        "modern_core_ids": list(authority.modern_core_ids),
        "readiness": direct_json_value(readiness),
    }
    if set(result) != _SELECTION_RECEIPT_KEYS:  # pragma: no cover - literal invariant
        raise AssertionError("comparator selection receipt schema is incomplete")
    return result


def build_comparator_selection_receipt(
    checkpoint: Mapping[str, object],
    *,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
) -> dict[str, object]:
    """Build the fixed ready receipt from one complete direct checkpoint."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    if not isinstance(authority, ComparatorTuningAuthority):
        raise TypeError("authority must be a ComparatorTuningAuthority")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    (
        plan_snapshot,
        input_descriptors,
        tuning_records,
        control_records,
        blocking_status_count,
    ) = _validate_selection_checkpoint(
        checkpoint,
        authority=authority,
        registry=registry,
    )
    return _build_selection_receipt_from_evidence(
        authority=authority,
        registry=registry,
        plan_snapshot=plan_snapshot,
        input_descriptors=input_descriptors,
        tuning_records=tuning_records,
        control_records=control_records,
        blocking_status_count=blocking_status_count,
        allow_blocked=False,
    )


def _validate_selection_receipt_schemas(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _SELECTION_RECEIPT_KEYS:
        raise ComparatorTuningError(
            "comparator selection receipt has missing or extra fields"
        )
    methods = value.get("methods")
    if not isinstance(methods, Mapping):
        raise ComparatorTuningError("comparator selection methods are invalid")
    for method in methods.values():
        if not isinstance(method, Mapping) or set(method) != _METHOD_RECEIPT_KEYS:
            raise ComparatorTuningError(
                "comparator selection method has missing or extra fields"
            )
        configurations = method.get("configurations")
        if not isinstance(configurations, Mapping):
            raise ComparatorTuningError(
                "comparator selection configurations are invalid"
            )
        for configuration in configurations.values():
            if (
                not isinstance(configuration, Mapping)
                or set(configuration) != _CONFIGURATION_RECEIPT_KEYS
            ):
                raise ComparatorTuningError(
                    "comparator selection configuration has missing or extra fields"
                )
        nonexecution = method.get("nonexecution_identity")
        if nonexecution is not None and (
            not isinstance(nonexecution, Mapping)
            or set(nonexecution) != _NONEXECUTION_IDENTITY_KEYS
        ):
            raise ComparatorTuningError(
                "comparator selection nonexecution identity differs"
            )
    return value


def _canonical_selection_receipt_bytes(value: object) -> bytes:
    try:
        return _canonical_bytes(value) + b"\n"
    except (TypeError, ValueError) as error:
        raise ComparatorTuningError(
            "comparator selection receipt is not canonical JSON"
        ) from error


def _selection_unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparatorTuningError(
                f"duplicate comparator selection JSON key {key!r}"
            )
        result[key] = value
    return result


def _selection_reject_constant(value: str) -> None:
    raise ComparatorTuningError(f"nonfinite comparator selection JSON constant {value}")


def _parse_canonical_selection_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_selection_reject_constant,
            object_pairs_hook=_selection_unique_object,
        )
    except ComparatorTuningError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ComparatorTuningError(f"{label} is invalid") from error
    if type(value) is not dict or raw != _canonical_selection_receipt_bytes(value):
        raise ComparatorTuningError(f"{label} is not canonical JSON")
    return value


def _selection_path(
    repository: Path,
    relative_path: str,
    label: str,
) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.as_posix() != relative_path
    ):
        raise ComparatorTuningError(f"{label} path is unsafe")
    return repository.joinpath(*relative.parts)


def _reject_selection_symlinks(path: Path, repository: Path) -> None:
    try:
        relative = path.absolute().relative_to(repository.absolute())
    except ValueError as error:
        raise ComparatorTuningError(
            "comparator selection path leaves its repository"
        ) from error
    current = repository
    for component in relative.parts:
        current = current / component
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ComparatorTuningError("comparator selection path is not owned")


def _selection_file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _secure_read_regular(
    path: Path,
    repository: Path,
    label: str,
    *,
    allowed_link_counts: frozenset[int] = frozenset({1}),
) -> bytes:
    """Read one stable owned unique regular file without following links."""

    _reject_selection_symlinks(path, repository)
    descriptor = -1
    try:
        named_before = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or stat.S_ISLNK(named_before.st_mode)
            or opened_before.st_uid != os.geteuid()
            or opened_before.st_nlink not in allowed_link_counts
            or opened_before.st_mode & 0o002
            or _selection_file_identity(opened_before)
            != _selection_file_identity(named_before)
            or opened_before.st_size > 128 * 1024 * 1024
        ):
            raise ComparatorTuningError(f"{label} must be an owned regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        named_after = path.lstat()
        raw = b"".join(chunks)
        if (
            _selection_file_identity(opened_before)
            != _selection_file_identity(opened_after)
            or _selection_file_identity(opened_before)
            != _selection_file_identity(named_after)
            or len(raw) != opened_before.st_size
        ):
            raise ComparatorTuningError(f"{label} changed while being read")
        return raw
    except ComparatorTuningError:
        raise
    except OSError as error:
        raise ComparatorTuningError(f"{label} is unavailable") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_published_selection(
    path: Path,
    repository: Path,
    label: str,
    *,
    allow_transient_link: bool,
) -> bytes:
    """Read a concurrent publication through a bounded hard-link transition."""

    allowed_link_counts = frozenset({1, 2}) if allow_transient_link else frozenset({1})
    deadline = time.monotonic() + 1.0
    while True:
        try:
            return _secure_read_regular(
                path,
                repository,
                label,
                allowed_link_counts=allowed_link_counts,
            )
        except ComparatorTuningError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.001)


def _immutable_publish(
    path: Path,
    data: bytes,
    repository: Path,
) -> None:
    """Create one receipt through an exclusive hard link, accepting equal bytes."""

    _reject_selection_symlinks(path.parent, repository)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_selection_symlinks(path.parent, repository)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            existing = _read_published_selection(
                path,
                repository,
                "existing comparator selection",
                allow_transient_link=True,
            )
            if existing != data:
                raise ComparatorTuningError(
                    "existing comparator selection differs from new evidence"
                )
        except OSError as error:
            raise ComparatorTuningError(
                "comparator selection could not be published"
            ) from error
        temporary.unlink()
        published = _read_published_selection(
            path,
            repository,
            "published comparator selection",
            allow_transient_link=False,
        )
        if published != data:
            raise ComparatorTuningError("published comparator selection differs")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_embedded_selection_records(
    *,
    plan_entries: Sequence[Mapping[str, object]],
    tuning_records: object,
    control_records: object,
    authority: ComparatorTuningAuthority,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    int,
]:
    if not isinstance(tuning_records, list) or not isinstance(control_records, list):
        raise ComparatorTuningError("comparator selection embedded records are invalid")
    expected_tuning = tuple(
        entry
        for entry in plan_entries
        if isinstance(entry.get("identity"), Mapping)
        and entry["identity"].get("configuration_kind") == "comparator_tuning"
    )
    expected_controls = tuple(
        entry
        for entry in plan_entries
        if isinstance(entry.get("identity"), Mapping)
        and isinstance(entry["identity"].get("method"), Mapping)
        and entry["identity"]["method"].get("method_id")
        in authority.required_control_ids
    )
    if (
        len(tuning_records) != len(expected_tuning) != 0
        or len(tuning_records) != 1_632
        or len(control_records) != len(expected_controls) != 0
        or len(control_records) != 64
    ):
        raise ComparatorTuningError(
            "comparator selection embedded record denominator differs"
        )
    validated_tuning: list[Mapping[str, object]] = []
    blocking_status_count = 0
    for value, entry in zip(tuning_records, expected_tuning, strict=True):
        if not isinstance(value, Mapping):
            raise ComparatorTuningError("comparator selection tuning record differs")
        expanded = json.loads(_canonical_bytes(value).decode("utf-8"))
        assert isinstance(expanded, dict)
        run = expanded.get("run")
        identity = run.get("identity") if isinstance(run, Mapping) else None
        if isinstance(identity, Mapping) and identity.get("mechanism") != "symsim":
            metrics = expanded.get("metrics")
            if not isinstance(metrics, list):
                raise ComparatorTuningError(
                    "comparator selection tuning record differs"
                )
            prezero_rows = [
                row for row in metrics if row.get("metric") == "mse_pre_dropout_zero"
            ]
            if prezero_rows:
                raise ComparatorTuningError(
                    "comparator selection metric applicability differs"
                )
            metrics.insert(
                3,
                {
                    "identity": direct_json_value(identity),
                    "metric": "mse_pre_dropout_zero",
                    "value": None,
                    "n": 0,
                    "status": (
                        "unavailable"
                        if run.get("status") == "completed"
                        else run.get("status")
                    ),
                    "reason": (
                        "truth_unavailable"
                        if run.get("status") == "completed"
                        else run.get("reason")
                    ),
                },
            )
        validated = _validate_selection_checkpoint_record(expanded, entry)
        status = str(validated["run"]["status"])
        if status in _BLOCKING_DIRECT_STATUSES:
            blocking_status_count += 1
        validated_tuning.append(_project_selection_record(validated))
    validated_controls = tuple(
        _validate_selection_checkpoint_record(value, entry)
        for value, entry in zip(control_records, expected_controls, strict=True)
    )
    blocking_status_count += sum(
        record["run"]["status"] in _BLOCKING_DIRECT_STATUSES
        for record in validated_controls
    )
    return tuple(validated_tuning), validated_controls, blocking_status_count


def _load_selection_authorities(
    repository: Path,
) -> tuple[MethodRegistry, ComparatorTuningAuthority]:
    try:
        registry = load_method_registry(repository / "study/methods.json")
        authority = load_comparator_tuning_authority(
            repository,
            registry=registry,
            require_clean=False,
        )
    except (ComparatorTuningError, OSError, TypeError, ValueError) as error:
        raise ComparatorTuningError(
            "comparator selection authority is unavailable"
        ) from error
    return registry, authority


def _expected_checkpoint_payload(
    expected_checkpoint: object,
    repository: Path,
) -> Mapping[str, object]:
    if isinstance(expected_checkpoint, Mapping):
        return expected_checkpoint
    if isinstance(expected_checkpoint, Path):
        path = (
            expected_checkpoint
            if expected_checkpoint.is_absolute()
            else repository / expected_checkpoint
        )
        raw = _secure_read_regular(path, repository, "expected direct checkpoint")
        return _parse_canonical_selection_json(raw, "expected direct checkpoint")
    raise TypeError("expected_checkpoint must be a mapping, pathlib.Path, or None")


def load_comparator_selection_receipt(
    repository: Path,
    *,
    expected_checkpoint: object = None,
) -> dict[str, object]:
    """Load canonical selection evidence and recompute every derived value."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.resolve(strict=True)
    registry, authority = _load_selection_authorities(root)
    receipt_path = _selection_path(
        root,
        COMPARATOR_SELECTION_RELATIVE_PATH,
        "comparator selection receipt",
    )
    raw = _secure_read_regular(receipt_path, root, "comparator selection receipt")
    payload = _parse_canonical_selection_json(raw, "comparator selection receipt")
    _validate_selection_receipt_schemas(payload)
    if (
        payload.get("schema_version") != 1
        or type(payload.get("schema_version")) is not int
        or payload.get("artifact_type") != "maskimpute-comparator-selection-receipt-v1"
        or payload.get("data_scope") != "development_only"
        or payload.get("final_data_used") is not False
        or payload.get("checkpoint_path") != _SELECTION_CHECKPOINT_RELATIVE_PATH
        or not direct_equal(
            payload.get("authority_reference"),
            direct_json_value(_authority_reference(authority)),
        )
    ):
        raise ComparatorTuningError(
            "comparator selection receipt authority or scope differs"
        )
    inputs, plan_entries = _validate_selection_plan_snapshot(
        payload.get("plan_snapshot"),
        authority=authority,
        registry=registry,
    )
    if not direct_equal(payload.get("input_descriptors"), list(inputs)):
        raise ComparatorTuningError(
            "comparator selection receipt input descriptors differ"
        )
    tuning, controls, blocking_status_count = _validate_embedded_selection_records(
        plan_entries=plan_entries,
        tuning_records=payload.get("scheduled_tuning_records"),
        control_records=payload.get("control_records"),
        authority=authority,
    )
    plan_snapshot = payload["plan_snapshot"]
    assert isinstance(plan_snapshot, Mapping)
    recomputed = _build_selection_receipt_from_evidence(
        authority=authority,
        registry=registry,
        plan_snapshot=plan_snapshot,
        input_descriptors=inputs,
        tuning_records=tuning,
        control_records=controls,
        blocking_status_count=blocking_status_count,
        allow_blocked=False,
    )
    if not direct_equal(
        payload, recomputed
    ) or raw != _canonical_selection_receipt_bytes(recomputed):
        raise ComparatorTuningError(
            "comparator selection receipt differs from recomputed evidence"
        )
    if expected_checkpoint is not None:
        checkpoint = _expected_checkpoint_payload(expected_checkpoint, root)
        expected = build_comparator_selection_receipt(
            checkpoint,
            authority=authority,
            registry=registry,
        )
        if raw != _canonical_selection_receipt_bytes(expected):
            raise ComparatorTuningError(
                "comparator selection receipt differs from expected checkpoint"
            )
    return recomputed


def publish_comparator_selection(repository: Path) -> dict[str, object]:
    """Publish the fixed comparator selection create-only with no overrides."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    root = repository.resolve(strict=True)
    registry, authority = _load_selection_authorities(root)
    checkpoint_path = _selection_path(
        root,
        _SELECTION_CHECKPOINT_RELATIVE_PATH,
        "direct checkpoint",
    )
    checkpoint_raw = _secure_read_regular(
        checkpoint_path,
        root,
        "direct checkpoint",
    )
    checkpoint = _parse_canonical_selection_json(
        checkpoint_raw,
        "direct checkpoint",
    )
    receipt = build_comparator_selection_receipt(
        checkpoint,
        authority=authority,
        registry=registry,
    )
    receipt_path = _selection_path(
        root,
        COMPARATOR_SELECTION_RELATIVE_PATH,
        "comparator selection receipt",
    )
    _immutable_publish(
        receipt_path,
        _canonical_selection_receipt_bytes(receipt),
        root,
    )
    return load_comparator_selection_receipt(
        root,
        expected_checkpoint=checkpoint_path,
    )


@dataclass(frozen=True, slots=True)
class ComparatorSmokeInputDescriptor:
    """Complete readable description of the fixed truth-free smoke fixture."""

    schema_version: int
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
    count_formula: str
    batch_rule: str
    normalization: object


@dataclass(frozen=True, slots=True)
class ComparatorSmokeOutcome:
    """Resource-only terminal evidence for one fixed smoke configuration."""

    configuration: BoundComparatorConfiguration
    status: str
    reason: str | None
    runtime_seconds: float
    peak_rss_bytes: int
    peak_gpu_bytes: int
    rss_measurement: str
    gpu_measurement: str


@dataclass(frozen=True, slots=True)
class _ComparatorSmokeRequest:
    configuration: BoundComparatorConfiguration
    fixture: ComparatorSmokeInputDescriptor
    method_input: MethodInput
    method_spec: MethodSpec
    model_seed: int
    ordinal: int


def comparator_method_binding(method_spec: MethodSpec) -> ComparatorMethodBinding:
    """Project exactly the registry fields required for comparator execution."""

    if not isinstance(method_spec, MethodSpec):
        raise TypeError("method_spec must be a MethodSpec")
    return ComparatorMethodBinding(
        method_id=method_spec.id,
        execution_scope=method_spec.execution_scope,
        integration_status=method_spec.integration_status,
        adapter_key=method_spec.id,
        environment_id=method_spec.environment.id,
        environment_status=method_spec.environment.status,
        source_kind=method_spec.source.kind,
        source_url=method_spec.source.url,
        source_revision=method_spec.source.revision,
        source_tree=method_spec.source.tree,
        source_cache_path=method_spec.source.cache_path,
        source_freeze_binding=method_spec.source.freeze_binding,
        input_scale=method_spec.input_scale,
        output_scale=method_spec.output_scale,
        stochastic=method_spec.stochastic,
        seed_policy=method_spec.seed_policy,
        gpu_mode=method_spec.resources.gpu_mode,
        cpu_cores=method_spec.resources.cpu_cores,
        timeout_seconds=method_spec.resources.timeout_seconds,
        max_rss_gib=method_spec.resources.max_rss_gib,
        max_gpu_gib=method_spec.resources.max_gpu_gib,
        preserves_observed_positives=method_spec.preserves_observed_positives,
    )


def bind_comparator_configuration_identity(
    configuration: ComparatorConfiguration,
    method_spec: MethodSpec,
    authority: ComparatorTuningAuthority,
) -> BoundComparatorConfiguration:
    """Resolve and bind one exact authority row without content summaries."""

    if not isinstance(configuration, ComparatorConfiguration):
        raise TypeError("configuration must be a ComparatorConfiguration")
    if not isinstance(method_spec, MethodSpec):
        raise TypeError("method_spec must be a MethodSpec")
    if not isinstance(authority, ComparatorTuningAuthority):
        raise TypeError("authority must be a ComparatorTuningAuthority")
    if configuration.method_id != method_spec.id:
        raise ComparatorTuningError("configuration method differs from registry method")
    authority_rows = tuple(
        row for row in authority.configurations if row == configuration
    )
    if len(authority_rows) != 1:
        raise ComparatorTuningError(
            "configuration does not resolve to one exact authority configuration"
        )
    authoritative_configuration = authority_rows[0]
    try:
        authoritative_payload = dict(authoritative_configuration.payload)
        canonical_payload_json = _canonical_bytes(authoritative_payload).decode("utf-8")
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparatorTuningError(
            "authority configuration payload is invalid JSON data"
        ) from error
    if authoritative_configuration.payload_json != canonical_payload_json:
        raise ComparatorTuningError(
            "authority configuration payload is not canonical JSON"
        )
    authoritative_configuration.decode()
    return BoundComparatorConfiguration(
        configuration=authoritative_configuration,
        authority_reference=ComparatorAuthorityReference(
            path="study/comparator_tuning.json",
            schema_version=authority.schema_version,
            authority_revision=authority.authority_revision,
        ),
        method=comparator_method_binding(method_spec),
    )


def build_comparator_smoke_input() -> MethodInput:
    """Build the exact fixed 900-by-500 truth-free adapter input."""

    counts = np.fromfunction(
        lambda cell, gene: (
            17 * cell.astype(np.int64)
            + 31 * gene.astype(np.int64)
            + 7
            * np.bitwise_xor(
                cell.astype(np.int64),
                gene.astype(np.int64),
            )
        )
        % 6,
        (900, 500),
        dtype=np.int64,
    )
    count_bytes = np.asarray(counts, dtype="<f8", order="C").tobytes(order="C")
    batch_values = tuple(f"batch-{index % 2}" for index in range(900))
    return MethodInput(
        "0" * 64,
        tuple(f"smoke-cell-{index:04d}" for index in range(900)),
        tuple(f"smoke-gene-{index:04d}" for index in range(500)),
        (900, 500),
        (
            CovariateColumn(
                name="batch",
                kind="categorical",
                dtype="category",
                values=batch_values,
                categories=("batch-0", "batch-1"),
                ordered=False,
                codes=tuple(index % 2 for index in range(900)),
            ),
        ),
        (),
        count_bytes,
        b'"raw_counts"',
    )


def comparator_smoke_input_descriptor(
    method_input: MethodInput,
) -> ComparatorSmokeInputDescriptor:
    """Return the full fixed fixture description without a content summary."""

    if not isinstance(method_input, MethodInput):
        raise TypeError("method_input must be a MethodInput")
    expected = build_comparator_smoke_input()
    if (
        not direct_equal(method_input.obs_ids, expected.obs_ids)
        or not direct_equal(method_input.var_ids, expected.var_ids)
        or not direct_equal(method_input.shape, expected.shape)
        or not direct_equal(method_input.obs_covariates, expected.obs_covariates)
        or not direct_equal(method_input.var_covariates, expected.var_covariates)
        or not direct_equal(method_input.normalization, expected.normalization)
        or method_input.counts.dtype.str != "<f8"
        or type(method_input._count_bytes) is not bytes
        or method_input._count_bytes != expected._count_bytes
    ):
        raise ComparatorTuningError(
            "comparator smoke fixture differs from the fixed input"
        )
    counts = method_input.counts
    batch = method_input.obs_covariates[0]
    return ComparatorSmokeInputDescriptor(
        schema_version=1,
        source_reference="tracked-fixed-truth-free-comparator-smoke",
        preprocessing_revision="raw-counts-v1",
        shape=(900, 500),
        dtype=counts.dtype.str,
        cell_ids=method_input.obs_ids,
        gene_ids=method_input.var_ids,
        batch_labels=tuple(str(value) for value in batch.values),
        total_count=float(np.sum(counts, dtype=np.float64)),
        nonzero_count=int(np.count_nonzero(counts)),
        minimum=float(np.min(counts)),
        maximum=float(np.max(counts)),
        count_formula="(17*cell+31*gene+7*(cell^gene))%6",
        batch_rule="alternating_batch-0_batch-1",
        normalization=method_input.normalization,
    )


def _bound_smoke_configurations(
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
) -> tuple[BoundComparatorConfiguration, ...]:
    validate_comparator_tuning_authority(authority)
    return tuple(
        bind_comparator_configuration_identity(
            row,
            registry.by_id(row.method_id),
            authority,
        )
        for row in authority.configurations
    )


def build_comparator_smoke_receipt(
    outcomes: Sequence[ComparatorSmokeOutcome],
    *,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
    bound_configurations: Sequence[BoundComparatorConfiguration],
) -> dict[str, object]:
    """Build the complete ready receipt from all 34 measured outcomes."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise TypeError("outcomes must be a sequence")
    if isinstance(bound_configurations, (str, bytes)) or not isinstance(
        bound_configurations, Sequence
    ):
        raise TypeError("bound_configurations must be a sequence")
    rows = tuple(outcomes)
    expected = tuple(bound_configurations)
    authoritative = _bound_smoke_configurations(authority, registry)
    if (
        len(expected) != 34
        or not direct_equal(expected, authoritative)
        or any(not isinstance(row, BoundComparatorConfiguration) for row in expected)
    ):
        raise ComparatorTuningError(
            "smoke denominator differs from bound authority order"
        )
    if (
        len(rows) != len(expected)
        or any(not isinstance(row, ComparatorSmokeOutcome) for row in rows)
        or not direct_equal(
            tuple(row.configuration for row in rows),
            expected,
        )
    ):
        raise ComparatorTuningError(
            "smoke denominator differs from bound authority order"
        )
    if any(row.status != "completed" or row.reason is not None for row in rows):
        raise ComparatorTuningError(
            "all configurations must complete before scientific execution"
        )
    if any(
        type(row.runtime_seconds) is not float
        or not math.isfinite(row.runtime_seconds)
        or row.runtime_seconds < 0
        or (
            row.runtime_seconds == 0.0 and math.copysign(1.0, row.runtime_seconds) < 0.0
        )
        or type(row.peak_rss_bytes) is not int
        or row.peak_rss_bytes < 0
        or type(row.peak_gpu_bytes) is not int
        or row.peak_gpu_bytes < 0
        or type(row.rss_measurement) is not str
        or not row.rss_measurement
        or type(row.gpu_measurement) is not str
        or not row.gpu_measurement
        for row in rows
    ):
        raise ComparatorTuningError("smoke measurement is invalid")
    projected: dict[str, float] = {}
    for row in rows:
        method_id = row.configuration.configuration.method_id
        projected[method_id] = (
            projected.get(method_id, 0.0) + 48.0 * row.runtime_seconds
        )
        if row.peak_rss_bytes > 48 * 1024**3 or row.peak_gpu_bytes > 14 * 1024**3:
            raise ComparatorTuningError("smoke resource cap is exceeded")
    if any(
        seconds
        > (8 * 3600 if registry.by_id(method).resources.gpu_required else 24 * 3600)
        for method, seconds in projected.items()
    ):
        raise ComparatorTuningError(
            "projected comparator grid exceeds its method budget"
        )
    return {
        "schema_version": 1,
        "artifact_type": "maskimpute-comparator-smoke-receipt-v1",
        "scope": "fixed_nonstudy_truth_free_operational_feasibility",
        "authority_revision": authority.authority_revision,
        "configurations": [direct_json_value(row) for row in expected],
        "fixture": direct_json_value(
            comparator_smoke_input_descriptor(build_comparator_smoke_input())
        ),
        "model_seed": 42,
        "projection_multiplier": 48,
        "planned_configuration_count": 34,
        "completed_configuration_count": 34,
        "status": "ready",
        "projected_method_runtime_seconds": dict(sorted(projected.items())),
        "outcomes": [direct_json_value(row) for row in rows],
        "output_retention": "discarded_without_evaluator_or_metrics",
    }


_SMOKE_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "scope",
        "authority_revision",
        "configurations",
        "fixture",
        "model_seed",
        "projection_multiplier",
        "planned_configuration_count",
        "completed_configuration_count",
        "status",
        "projected_method_runtime_seconds",
        "outcomes",
        "output_retention",
    }
)
_SMOKE_OUTCOME_KEYS = frozenset(
    {
        "configuration",
        "status",
        "reason",
        "runtime_seconds",
        "peak_rss_bytes",
        "peak_gpu_bytes",
        "rss_measurement",
        "gpu_measurement",
    }
)


def _canonical_smoke_receipt_bytes(value: object) -> bytes:
    try:
        return _canonical_bytes(value) + b"\n"
    except (TypeError, ValueError) as error:
        raise ComparatorTuningError(
            "comparator smoke receipt is not canonical JSON"
        ) from error


def _parse_smoke_receipt(raw: bytes) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ComparatorTuningError("comparator smoke receipt is invalid") from error
    if type(payload) is not dict or raw != _canonical_smoke_receipt_bytes(payload):
        raise ComparatorTuningError("comparator smoke receipt is not canonical JSON")
    return payload


def validate_comparator_smoke_receipt(
    payload: Mapping[str, object],
    raw: bytes,
    *,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
) -> dict[str, object]:
    """Recompute every fixed smoke identity and derived readiness value."""

    if type(payload) is not dict or type(raw) is not bytes:
        raise TypeError("smoke receipt payload and bytes have invalid types")
    if set(payload) != _SMOKE_RECEIPT_KEYS:
        raise ComparatorTuningError(
            "comparator smoke receipt has missing or extra fields"
        )
    if raw != _canonical_smoke_receipt_bytes(payload):
        raise ComparatorTuningError("comparator smoke receipt is not canonical JSON")
    expected_configurations = _bound_smoke_configurations(authority, registry)
    encoded_configurations = [direct_json_value(row) for row in expected_configurations]
    if not direct_equal(payload.get("configurations"), encoded_configurations):
        raise ComparatorTuningError(
            "comparator smoke receipt configurations differ from authority"
        )
    raw_outcomes = payload.get("outcomes")
    if type(raw_outcomes) is not list or len(raw_outcomes) != 34:
        raise ComparatorTuningError(
            "comparator smoke receipt outcome denominator differs"
        )
    outcomes: list[ComparatorSmokeOutcome] = []
    for index, (value, configuration) in enumerate(
        zip(raw_outcomes, expected_configurations, strict=True)
    ):
        if type(value) is not dict or set(value) != _SMOKE_OUTCOME_KEYS:
            raise ComparatorTuningError(
                "comparator smoke receipt outcome schema differs"
            )
        if not direct_equal(
            value["configuration"],
            encoded_configurations[index],
        ):
            raise ComparatorTuningError(
                "comparator smoke receipt outcome identity differs"
            )
        outcome = ComparatorSmokeOutcome(
            configuration=configuration,
            status=value["status"],
            reason=value["reason"],
            runtime_seconds=value["runtime_seconds"],
            peak_rss_bytes=value["peak_rss_bytes"],
            peak_gpu_bytes=value["peak_gpu_bytes"],
            rss_measurement=value["rss_measurement"],
            gpu_measurement=value["gpu_measurement"],
        )
        outcomes.append(outcome)
    recomputed = build_comparator_smoke_receipt(
        outcomes,
        authority=authority,
        registry=registry,
        bound_configurations=expected_configurations,
    )
    if not direct_equal(payload, recomputed):
        raise ComparatorTuningError(
            "comparator smoke receipt differs from recomputed fixed evidence"
        )
    return recomputed


def _smoke_receipt_path(
    repository: Path,
    authority: ComparatorTuningAuthority,
) -> Path:
    relative = PurePosixPath(authority.smoke_receipt_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ComparatorTuningError("comparator smoke receipt path is unsafe")
    return repository.joinpath(*relative.parts)


def _reject_smoke_symlinks(path: Path, repository: Path) -> None:
    current = repository
    relative = path.absolute().relative_to(repository.absolute())
    for component in relative.parts:
        current = current / component
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise ComparatorTuningError("comparator smoke receipt path is not owned")


def _read_owned_smoke_receipt(path: Path, repository: Path) -> bytes:
    _reject_smoke_symlinks(path, repository)
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != os.geteuid()
        ):
            raise ComparatorTuningError(
                "comparator smoke receipt must be an owned regular file"
            )
        raw = path.read_bytes()
        after = path.lstat()
    except ComparatorTuningError:
        raise
    except OSError as error:
        raise ComparatorTuningError(
            "comparator smoke receipt is unavailable"
        ) from error

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or before.st_size != len(raw):
        raise ComparatorTuningError("comparator smoke receipt changed while being read")
    return raw


def _load_comparator_smoke_receipt_evidence(
    repository: Path,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
) -> tuple[dict[str, object], bytes]:
    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if not isinstance(authority, ComparatorTuningAuthority):
        raise TypeError("authority must be a ComparatorTuningAuthority")
    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    root = repository.resolve(strict=True)
    reloaded = load_comparator_tuning_authority(
        root,
        registry=registry,
        require_clean=False,
    )
    if not direct_equal(reloaded, authority):
        raise ComparatorTuningError(
            "comparator smoke receipt authority differs from tracked authority"
        )
    path = _smoke_receipt_path(root, reloaded)
    raw = _read_owned_smoke_receipt(path, root)
    payload = _parse_smoke_receipt(raw)
    return (
        validate_comparator_smoke_receipt(
            payload,
            raw,
            authority=reloaded,
            registry=registry,
        ),
        raw,
    )


def load_comparator_smoke_receipt(
    repository: Path,
    authority: ComparatorTuningAuthority,
    registry: MethodRegistry,
) -> Mapping[str, object]:
    """Load and fully recompute the fixed canonical smoke receipt."""

    receipt, _raw = _load_comparator_smoke_receipt_evidence(
        repository,
        authority,
        registry,
    )
    return receipt


def _publish_comparator_smoke_receipt(
    repository: Path,
    authority: ComparatorTuningAuthority,
    receipt: Mapping[str, object],
) -> None:
    path = _smoke_receipt_path(repository, authority)
    data = _canonical_smoke_receipt_bytes(receipt)
    _reject_smoke_symlinks(path.parent, repository)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_smoke_symlinks(path.parent, repository)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read_owned_smoke_receipt(path, repository) != data:
                raise ComparatorTuningError(
                    "existing comparator smoke receipt conflicts with new evidence"
                )
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _execute_smoke_request_in_spawned_dispatcher(
    request: _ComparatorSmokeRequest,
    dispatcher: object,
    authority: ComparatorTuningAuthority,
) -> ComparatorSmokeOutcome:
    from dataclasses import replace

    from .fair_comparator_execution import DirectExecutionRequest
    from .fair_comparator_plan import ComparatorRunIdentity
    from .runner import (
        DirectRepositoryComparatorExecutor,
        LinuxProcessTreeResourceSampler,
        RepositoryAdapterDispatcher,
        execute_direct_adapter_in_spawned_process,
    )

    if not isinstance(request, _ComparatorSmokeRequest):
        raise TypeError("request must be a comparator smoke request")
    if not isinstance(dispatcher, RepositoryAdapterDispatcher):
        raise TypeError("dispatcher must be a RepositoryAdapterDispatcher")
    identity = ComparatorRunIdentity(
        workflow_schema="maskimpute-fair-comparator-run-v1",
        authority_revision=authority.authority_revision,
        ordinal=request.ordinal,
        method=request.configuration.method,
        configuration_id=request.configuration.configuration.configuration_id,
        configuration_kind="comparator_tuning",
        configuration_payload=freeze_direct_mapping(
            request.configuration.configuration.payload
        ),
        dataset_id="comparator-smoke-fixed",
        mechanism="fixed_nonstudy_truth_free_operational_feasibility",
        biological_id="fixed-smoke",
        technical_view="truth-free",
        mask_seed=0,
        model_seed=request.model_seed,
        draw_index=1,
    )
    direct_request = DirectExecutionRequest(
        identity=identity,
        method_spec=request.method_spec,
        method_input=request.method_input,
        timeout_seconds=float(EXPECTED_BUDGETS["per_run_timeout_seconds"]),
        max_rss_bytes=EXPECTED_BUDGETS["max_rss_bytes"],
        max_gpu_bytes=EXPECTED_BUDGETS["max_gpu_bytes"],
        smoke_fixture=request.fixture,
    )
    child_dispatcher = replace(
        dispatcher,
        environments=replace(dispatcher.environments, runtime_snapshot=None),
        monitor_runtime_changes=False,
    )
    snapshot = dispatcher.environments.runtime_snapshot
    sampler = (
        LinuxProcessTreeResourceSampler()
        if snapshot is None
        else LinuxProcessTreeResourceSampler(
            None if snapshot.nvidia_smi_path is None else Path(snapshot.nvidia_smi_path)
        )
    )
    outcome = execute_direct_adapter_in_spawned_process(
        direct_request,
        DirectRepositoryComparatorExecutor(child_dispatcher, authority),
        resource_sampler=sampler,
        expected_spawn_executable=dispatcher.environments.benchmark_python,
        spawn_search_path=dispatcher.environments.python_spawn_search_path,
    )
    return ComparatorSmokeOutcome(
        configuration=request.configuration,
        status=outcome.status,
        reason=outcome.reason,
        runtime_seconds=float(outcome.runtime_seconds),
        peak_rss_bytes=outcome.peak_rss_bytes,
        peak_gpu_bytes=outcome.peak_gpu_bytes,
        rss_measurement=outcome.rss_measurement,
        gpu_measurement=outcome.gpu_measurement,
    )


def run_comparator_tuning_smoke(
    repository: Path,
    *,
    _executor: Callable[
        [_ComparatorSmokeRequest, object, ComparatorTuningAuthority],
        ComparatorSmokeOutcome,
    ] = _execute_smoke_request_in_spawned_dispatcher,
) -> Mapping[str, object]:
    """Run only the fixed 34-row adapter smoke boundary and publish its receipt."""

    if not isinstance(repository, Path):
        raise TypeError("repository must be a pathlib.Path")
    if not callable(_executor):
        raise TypeError("_executor must be callable")
    root = repository.resolve(strict=True)
    registry = load_method_registry(root / "study/methods.json")
    authority = load_comparator_tuning_authority(
        root,
        registry=registry,
        require_clean=True,
    )
    configurations = _bound_smoke_configurations(authority, registry)
    method_input = build_comparator_smoke_input()
    fixture = comparator_smoke_input_descriptor(method_input)
    dispatcher: object = None
    try:
        if _executor is _execute_smoke_request_in_spawned_dispatcher:
            from .runner import (
                ExecutionEnvironmentRegistry,
                RepositoryAdapterDispatcher,
                derive_lock_only_environment_ids,
            )

            environments = ExecutionEnvironmentRegistry.fixed(
                root,
                runtime_lock_path=(root / "environments/development-runtime.lock.json"),
                benchmark_python=Path(sys.executable),
                r_library_paths={"saver": (root / "artifacts/envs/saver-r/library",)},
                lock_only_environment_ids=derive_lock_only_environment_ids(registry),
            )
            dispatcher = RepositoryAdapterDispatcher(
                root,
                environments,
                comparator_tuning_authority=authority,
            )
    except ComparatorTuningError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise ComparatorTuningError(
            "comparator smoke adapter boundary failed"
        ) from error
    outcomes = []
    for ordinal, configuration in enumerate(configurations, start=1):
        request = _ComparatorSmokeRequest(
            configuration=configuration,
            fixture=fixture,
            method_input=method_input,
            method_spec=registry.by_id(configuration.configuration.method_id),
            model_seed=42,
            ordinal=ordinal,
        )
        try:
            outcome = _executor(request, dispatcher, authority)
        except ComparatorTuningError:
            raise
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            raise ComparatorTuningError(
                "comparator smoke adapter boundary failed"
            ) from error
        if not isinstance(outcome, ComparatorSmokeOutcome) or not direct_equal(
            outcome.configuration, configuration
        ):
            raise ComparatorTuningError(
                "comparator smoke executor returned a noncanonical outcome"
            )
        outcomes.append(outcome)
    receipt = build_comparator_smoke_receipt(
        outcomes,
        authority=authority,
        registry=registry,
        bound_configurations=configurations,
    )
    _publish_comparator_smoke_receipt(root, authority, receipt)
    return load_comparator_smoke_receipt(root, authority, registry)


def _require_exact_mapping(
    value: object,
    keys: set[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ComparatorTuningError(f"{name} has missing or extra fields")
    return value


def _require_exact_literal(value: object, expected: object, name: str) -> None:
    """Require recursively identical JSON values without bool/numeric coercion."""

    if type(value) is not type(expected):
        raise ComparatorTuningError(f"{name} differs from the tracked contract")
    if type(expected) is dict:
        observed_mapping = _require_exact_mapping(
            value,
            set(expected),
            name,
        )
        for key, expected_item in expected.items():
            _require_exact_literal(
                observed_mapping[key], expected_item, f"{name}.{key}"
            )
        return
    if type(expected) in {list, tuple}:
        if len(value) != len(expected):
            raise ComparatorTuningError(f"{name} differs from the tracked contract")
        for index, (observed_item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_literal(observed_item, expected_item, f"{name}[{index}]")
        return
    if type(expected) is float and value.hex() != expected.hex():
        raise ComparatorTuningError(f"{name} differs from the tracked contract")
    if value != expected:
        raise ComparatorTuningError(f"{name} differs from the tracked contract")


def _canonical_comparator_tuning_authority() -> ComparatorTuningAuthority:
    configurations = tuple(
        ComparatorConfiguration(
            method_id=method_id,
            configuration_id=configuration_id,
            payload_json=_EXPECTED_CONFIGURATION_PAYLOADS[
                (method_id, configuration_id)
            ],
            is_upstream_default=is_upstream_default,
        )
        for method_id, configuration_id, is_upstream_default, _config in (
            _EXPECTED_CONFIGURATION_SPECS
        )
    )
    return ComparatorTuningAuthority(
        schema_version=2,
        contract_id="maskimpute-comparator-tuning-v1",
        authority_revision=AUTHORITY_REVISION,
        method_order=_EXPECTED_METHOD_ORDER,
        configurations=configurations,
        scheduled_same_input_ids=_EXPECTED_SCHEDULED_SAME_INPUT_IDS,
        required_control_ids=_EXPECTED_REQUIRED_CONTROL_IDS,
        established_comparator_ids=_EXPECTED_ESTABLISHED_COMPARATOR_IDS,
        modern_core_ids=_EXPECTED_MODERN_CORE_IDS,
        model_seeds=EXPECTED_MODEL_SEEDS,
        selection_metrics=EXPECTED_METRICS,
        receipt_path=COMPARATOR_SELECTION_RELATIVE_PATH,
        smoke_receipt_path=COMPARATOR_SMOKE_RELATIVE_PATH,
    )


def validate_comparator_tuning_authority(
    authority: ComparatorTuningAuthority,
) -> None:
    """Require the complete, fixed schema-2 comparator authority value."""

    if not isinstance(authority, ComparatorTuningAuthority):
        raise TypeError("authority must be a ComparatorTuningAuthority")
    _require_exact_literal(
        asdict(authority),
        asdict(_canonical_comparator_tuning_authority()),
        "comparator tuning authority",
    )


def parse_comparator_tuning_authority(
    payload: object,
    *,
    registry: MethodRegistry,
) -> ComparatorTuningAuthority:
    """Parse the closed schema-2 comparator-tuning authority."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    root = _require_exact_mapping(
        payload,
        {
            "schema_version",
            "contract_id",
            "authority_revision",
            "scope",
            "method_order",
            "scheduled_same_input_ids",
            "required_control_ids",
            "established_comparator_ids",
            "modern_core_ids",
            "model_seeds",
            "configurations",
            "selection",
            "budgets",
            "storage",
            "smoke",
        },
        "comparator tuning authority",
    )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != 2
        or type(root["contract_id"]) is not str
        or root["contract_id"] != "maskimpute-comparator-tuning-v1"
    ):
        raise ComparatorTuningError("comparator tuning schema or contract differs")
    _require_exact_literal(
        root["authority_revision"],
        AUTHORITY_REVISION,
        "comparator tuning authority revision",
    )

    _require_exact_literal(root["scope"], EXPECTED_SCOPE, "comparator tuning scope")
    _require_exact_literal(
        root["method_order"],
        list(_EXPECTED_METHOD_ORDER),
        "comparator tuning method order",
    )
    _require_exact_literal(
        root["required_control_ids"],
        list(_EXPECTED_REQUIRED_CONTROL_IDS),
        "required control IDs",
    )
    _require_exact_literal(
        root["established_comparator_ids"],
        list(_EXPECTED_ESTABLISHED_COMPARATOR_IDS),
        "established comparator IDs",
    )
    _require_exact_literal(
        root["modern_core_ids"],
        list(_EXPECTED_MODERN_CORE_IDS),
        "modern core IDs",
    )
    _require_exact_literal(
        root["model_seeds"],
        list(EXPECTED_MODEL_SEEDS),
        "comparator tuning model seeds",
    )

    rows = root["configurations"]
    if type(rows) is not list or len(rows) != len(_EXPECTED_CONFIGURATION_SPECS):
        raise ComparatorTuningError(
            "comparator configurations differ from the exact 34-row grid"
        )
    configurations: list[ComparatorConfiguration] = []
    seen_configuration_ids = {method_id: set() for method_id in _EXPECTED_METHOD_ORDER}
    seen_payload_json = {method_id: set() for method_id in _EXPECTED_METHOD_ORDER}
    observed_order: list[tuple[str, str]] = []
    for index, (raw, expected_spec) in enumerate(
        zip(rows, _EXPECTED_CONFIGURATION_SPECS, strict=True)
    ):
        row = _require_exact_mapping(
            raw,
            {
                "method_id",
                "configuration_id",
                "is_upstream_default",
                "payload",
            },
            f"comparator configuration {index}",
        )
        if (
            type(row["method_id"]) is not str
            or row["method_id"] not in _CONFIG_TYPES
            or type(row["configuration_id"]) is not str
        ):
            raise ComparatorTuningError("comparator configuration identity is invalid")
        if (
            _SAFE_ID.fullmatch(row["configuration_id"]) is None
            or type(row["is_upstream_default"]) is not bool
        ):
            raise ComparatorTuningError("comparator configuration identity is invalid")
        expected_method, expected_id, expected_default, _expected_config = expected_spec
        observed_identity = (row["method_id"], row["configuration_id"])
        if observed_identity != (expected_method, expected_id):
            raise ComparatorTuningError(
                "comparator configuration order or identity differs"
            )
        if row["is_upstream_default"] is not expected_default:
            raise ComparatorTuningError(
                "comparator upstream-default declaration differs"
            )
        if row["configuration_id"] in seen_configuration_ids[row["method_id"]]:
            raise ComparatorTuningError(
                "duplicate comparator configuration ID within method"
            )
        seen_configuration_ids[row["method_id"]].add(row["configuration_id"])

        expected_payload_json = _EXPECTED_CONFIGURATION_PAYLOADS[observed_identity]
        try:
            observed_payload_bytes = _canonical_bytes(row["payload"])
        except (TypeError, ValueError, UnicodeError) as error:
            raise ComparatorTuningError(
                "comparator configuration payload is not canonical JSON data"
            ) from error
        payload_json = observed_payload_bytes.decode("utf-8")
        if payload_json in seen_payload_json[row["method_id"]]:
            raise ComparatorTuningError(
                "duplicate comparator configuration payload within method"
            )
        seen_payload_json[row["method_id"]].add(payload_json)
        if observed_payload_bytes != expected_payload_json.encode("utf-8"):
            raise ComparatorTuningError(
                "comparator configuration payload representation differs"
            )
        expected_payload = json.loads(expected_payload_json)
        _require_exact_literal(
            row["payload"],
            expected_payload,
            f"comparator configuration {index} payload",
        )
        decoded = decode_comparator_configuration(
            row["method_id"],
            row["payload"],
        )
        configurations.append(
            ComparatorConfiguration(
                method_id=row["method_id"],
                configuration_id=row["configuration_id"],
                payload_json=_canonical_bytes(_json_payload(decoded)).decode("utf-8"),
                is_upstream_default=row["is_upstream_default"],
            )
        )
        observed_order.append(observed_identity)
    if tuple(observed_order) != _EXPECTED_CONFIGURATION_ORDER:
        raise ComparatorTuningError("comparator configuration order differs")
    for method_id, config_type in _CONFIG_TYPES.items():
        method_rows = tuple(row for row in configurations if row.method_id == method_id)
        defaults = tuple(row for row in method_rows if row.is_upstream_default)
        if len(defaults) != 1 or not method_rows[0].is_upstream_default:
            raise ComparatorTuningError(
                "each comparator requires exactly one first upstream default"
            )
        if dict(defaults[0].payload) != encode_comparator_configuration(config_type()):
            raise ComparatorTuningError(
                "comparator upstream default differs from its adapter dataclass"
            )

    scheduled = tuple(
        spec.id
        for spec in registry.methods
        if spec.execution_scope == "same_input_required" and spec.role != "candidate"
    )
    if scheduled != _EXPECTED_SCHEDULED_SAME_INPUT_IDS:
        raise ComparatorTuningError(
            "registry same-input denominator differs from the approved contract"
        )
    _require_exact_literal(
        root["scheduled_same_input_ids"],
        list(scheduled),
        "scheduled same-input IDs",
    )
    if tuple(root["scheduled_same_input_ids"]) != scheduled:
        raise ComparatorTuningError(
            "scheduled same-input denominator differs from registry"
        )

    selection = _require_exact_mapping(
        root["selection"],
        {
            "metrics",
            "collapse_order",
            "prezero_mechanism",
            "pareto_rule",
            "rank_rule",
            "selection_tuple",
            "readiness",
            "receipt_path",
        },
        "comparator selection policy",
    )
    _require_exact_literal(
        selection["metrics"], list(EXPECTED_METRICS), "comparator selection metrics"
    )
    _require_exact_literal(
        selection["collapse_order"],
        list(EXPECTED_COLLAPSE_ORDER),
        "comparator collapse order",
    )
    _require_exact_literal(
        selection["prezero_mechanism"],
        "symsim",
        "comparator prezero mechanism",
    )
    _require_exact_literal(
        selection["pareto_rule"],
        _EXPECTED_PARETO_RULE,
        "comparator Pareto rule",
    )
    _require_exact_literal(
        selection["rank_rule"], _EXPECTED_RANK_RULE, "comparator rank rule"
    )
    _require_exact_literal(
        selection["selection_tuple"],
        list(EXPECTED_SELECTION_TUPLE),
        "comparator selection tuple",
    )
    _require_exact_literal(
        selection["readiness"], _EXPECTED_READINESS, "comparator readiness policy"
    )
    _require_exact_literal(
        selection["receipt_path"],
        COMPARATOR_SELECTION_RELATIVE_PATH,
        "comparator selection receipt path",
    )

    budgets = _require_exact_mapping(
        root["budgets"], set(EXPECTED_BUDGETS), "comparator budget policy"
    )
    _require_exact_literal(budgets, EXPECTED_BUDGETS, "comparator budget policy")

    smoke = _require_exact_mapping(
        root["smoke"],
        {
            "receipt_path",
            "cells",
            "genes",
            "model_seed",
            "batch_rule",
            "count_formula",
            "projection_multiplier",
            "output_retention",
        },
        "comparator smoke policy",
    )
    _require_exact_literal(smoke, EXPECTED_SMOKE, "comparator smoke policy")

    storage = _require_exact_mapping(
        root["storage"],
        {
            "max_log_receipt_bytes",
            "max_executor_receipt_bytes",
            "max_record_bytes",
            "max_checkpoint_bytes",
            "reserve_bytes",
        },
        "development storage policy",
    )
    _require_exact_literal(storage, _EXPECTED_STORAGE, "development storage policy")

    return ComparatorTuningAuthority(
        schema_version=2,
        contract_id="maskimpute-comparator-tuning-v1",
        authority_revision=AUTHORITY_REVISION,
        method_order=_EXPECTED_METHOD_ORDER,
        configurations=tuple(configurations),
        scheduled_same_input_ids=_EXPECTED_SCHEDULED_SAME_INPUT_IDS,
        required_control_ids=_EXPECTED_REQUIRED_CONTROL_IDS,
        established_comparator_ids=_EXPECTED_ESTABLISHED_COMPARATOR_IDS,
        modern_core_ids=_EXPECTED_MODERN_CORE_IDS,
        model_seeds=EXPECTED_MODEL_SEEDS,
        selection_metrics=EXPECTED_METRICS,
        receipt_path=COMPARATOR_SELECTION_RELATIVE_PATH,
        smoke_receipt_path=COMPARATOR_SMOKE_RELATIVE_PATH,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparatorTuningError(f"duplicate comparator authority key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ComparatorTuningError(f"nonfinite comparator authority constant {value}")


def load_comparator_tuning_authority(
    repository: Path,
    *,
    registry: MethodRegistry,
    require_clean: bool = True,
) -> ComparatorTuningAuthority:
    """Load canonical tracked comparator-tuning authority from a repository."""

    if not isinstance(repository, Path) or not isinstance(registry, MethodRegistry):
        raise TypeError("repository and registry have invalid types")
    root = repository.resolve(strict=True)
    path = root / "study/comparator_tuning.json"
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ComparatorTuningError(
                "comparator tuning authority is not an owned regular file"
            )
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ComparatorTuningError("comparator tuning authority is invalid") from error
    if raw != json.dumps(payload, indent=2).encode("utf-8") + b"\n":
        raise ComparatorTuningError(
            "comparator tuning authority is not canonical tracked JSON"
        )
    if require_clean:
        import subprocess

        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "study/comparator_tuning.json"],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "study/comparator_tuning.json",
            ],
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if tracked.returncode != 0 or status.returncode != 0 or status.stdout:
            raise ComparatorTuningError(
                "comparator tuning authority is not tracked and clean"
            )
    return parse_comparator_tuning_authority(
        payload,
        registry=registry,
    )
