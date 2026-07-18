"""Strict authority contracts for development-only comparator tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

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
)
from .methods import MethodRegistry
from .protocol import canonical_sha256


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
_SAFE_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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

_EXPECTED_METHOD_ORDER = tuple(_CONFIG_TYPES)
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
    *,
    expected_payload_sha256: str,
) -> ComparatorAdapterConfig:
    """Bootstrap the authority loader; Task 2 replaces this with strict type checks."""

    config_type = _CONFIG_TYPES.get(method_id)
    if config_type is None or type(payload) not in {dict, MappingProxyType}:
        raise ComparatorTuningError("comparator method or payload is invalid")
    observed = dict(payload)
    defaults = _json_payload(config_type())
    if set(observed) != set(defaults):
        raise ComparatorTuningError(
            "comparator payload differs from its complete field set"
        )
    constructor = dict(observed)
    if method_id == "dca":
        hidden = observed["hidden_size"]
        if type(hidden) is not list:
            raise ComparatorTuningError("DCA hidden_size must be a JSON array")
        constructor["hidden_size"] = tuple(hidden)
    if canonical_sha256(observed) != expected_payload_sha256:
        raise ComparatorTuningError("comparator payload checksum differs")
    try:
        decoded = config_type(**constructor)
    except (TypeError, ValueError) as error:
        raise ComparatorTuningError(
            "comparator payload violates its adapter contract"
        ) from error
    if encode_comparator_configuration(decoded) != observed:
        raise ComparatorTuningError(
            "comparator payload changed during typed normalization"
        )
    return decoded


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
    payload_sha256: str
    is_upstream_default: bool

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(json.loads(self.payload_json))

    @property
    def observed_payload_sha256(self) -> str:
        return canonical_sha256(dict(self.payload))

    @property
    def payload_for_json_comparison(self) -> dict[str, object]:
        return _json_payload(self.decode())

    def decode(self) -> ComparatorAdapterConfig:
        return decode_comparator_configuration(
            self.method_id,
            self.payload,
            expected_payload_sha256=self.payload_sha256,
        )


@dataclass(frozen=True, slots=True)
class ComparatorTuningAuthority:
    """Validated immutable development-only comparator-tuning authority."""

    schema_version: int
    contract_id: str
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
    file_sha256: str
    payload_sha256: str

    def configurations_for(self, method_id: str) -> tuple[ComparatorConfiguration, ...]:
        return tuple(row for row in self.configurations if row.method_id == method_id)


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
    if type(expected) is list:
        if len(value) != len(expected):
            raise ComparatorTuningError(f"{name} differs from the tracked contract")
        for index, (observed_item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_literal(observed_item, expected_item, f"{name}[{index}]")
        return
    if value != expected:
        raise ComparatorTuningError(f"{name} differs from the tracked contract")


def parse_comparator_tuning_authority(
    payload: object,
    *,
    registry: MethodRegistry,
    file_sha256: str,
) -> ComparatorTuningAuthority:
    """Parse the closed schema-1 comparator-tuning authority."""

    if not isinstance(registry, MethodRegistry):
        raise TypeError("registry must be a MethodRegistry")
    if type(file_sha256) is not str or _SHA256.fullmatch(file_sha256) is None:
        raise ComparatorTuningError("comparator tuning file SHA-256 is invalid")
    root = _require_exact_mapping(
        payload,
        {
            "schema_version",
            "contract_id",
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
            "payload_sha256",
        },
        "comparator tuning authority",
    )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != 1
        or type(root["contract_id"]) is not str
        or root["contract_id"] != "maskimpute-comparator-tuning-v1"
    ):
        raise ComparatorTuningError("comparator tuning schema or contract differs")
    if (
        type(root["payload_sha256"]) is not str
        or _SHA256.fullmatch(root["payload_sha256"]) is None
    ):
        raise ComparatorTuningError("comparator tuning payload SHA-256 is invalid")
    unsigned = {key: value for key, value in root.items() if key != "payload_sha256"}
    try:
        observed_payload_sha256 = canonical_sha256(unsigned)
    except (TypeError, ValueError) as error:
        raise ComparatorTuningError(
            "comparator tuning payload is not canonical JSON data"
        ) from error
    if root["payload_sha256"] != observed_payload_sha256:
        raise ComparatorTuningError("comparator tuning payload checksum differs")

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
    seen_payload_sha256 = {method_id: set() for method_id in _EXPECTED_METHOD_ORDER}
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
                "payload_sha256",
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
        if (
            type(row["payload_sha256"]) is not str
            or _SHA256.fullmatch(row["payload_sha256"]) is None
        ):
            raise ComparatorTuningError(
                "comparator configuration payload SHA-256 is invalid"
            )
        if row["configuration_id"] in seen_configuration_ids[row["method_id"]]:
            raise ComparatorTuningError(
                "duplicate comparator configuration ID within method"
            )
        if row["payload_sha256"] in seen_payload_sha256[row["method_id"]]:
            raise ComparatorTuningError(
                "duplicate comparator configuration payload within method"
            )
        seen_configuration_ids[row["method_id"]].add(row["configuration_id"])
        seen_payload_sha256[row["method_id"]].add(row["payload_sha256"])

        expected_payload = json.loads(
            _EXPECTED_CONFIGURATION_PAYLOADS[observed_identity]
        )
        _require_exact_literal(
            row["payload"],
            expected_payload,
            f"comparator configuration {index} payload",
        )
        decoded = decode_comparator_configuration(
            row["method_id"],
            row["payload"],
            expected_payload_sha256=row["payload_sha256"],
        )
        configurations.append(
            ComparatorConfiguration(
                method_id=row["method_id"],
                configuration_id=row["configuration_id"],
                payload_json=_canonical_bytes(_json_payload(decoded)).decode("utf-8"),
                payload_sha256=row["payload_sha256"],
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
        schema_version=1,
        contract_id="maskimpute-comparator-tuning-v1",
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
        file_sha256=file_sha256,
        payload_sha256=root["payload_sha256"],
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
    raw = path.read_bytes()
    try:
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
        file_sha256=hashlib.sha256(raw).hexdigest(),
    )
