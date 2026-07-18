import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from maskimpute_benchmark.comparator_tuning import (
    ComparatorTuningError,
    DEVELOPMENT_MAX_CHECKPOINT_BYTES,
    DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
    DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
    DEVELOPMENT_MAX_RECORD_BYTES,
    DEVELOPMENT_STORAGE_RESERVE_BYTES,
    encode_comparator_configuration,
    load_comparator_tuning_authority,
    parse_comparator_tuning_authority,
)
from maskimpute_benchmark.methods import load_method_registry
from maskimpute_benchmark.protocol import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ORDER = {
    "alra": ("alra-default",),
    "magic": ("magic-t03", "magic-t01", "magic-t05", "magic-t07"),
    "dca": ("dca-h64-32-64", "dca-h32-16-32", "dca-h32-32", "dca-h64-64"),
    "scvi": ("scvi-z10", "scvi-z05", "scvi-z20", "scvi-z30"),
    "saver": ("saver-default",),
    "scziva": (
        "scziva-tau-0p001",
        "scziva-tau-0p0001",
        "scziva-tau-0p01",
        "scziva-tau-0p05",
    ),
    "afmf": ("afmf-sigma-3", "afmf-sigma-1", "afmf-sigma-2", "afmf-sigma-4"),
    "biaeimpute": (
        "biaeimpute-z128",
        "biaeimpute-z32",
        "biaeimpute-z64",
        "biaeimpute-z256",
    ),
    "sccr": ("sccr-k15", "sccr-k05", "sccr-k10", "sccr-k30"),
    "scsdae": (
        "scsdae-zero-1",
        "scsdae-zero-0p25",
        "scsdae-zero-0p5",
        "scsdae-zero-0p75",
    ),
}


def _tracked_payload() -> dict[str, object]:
    return json.loads((ROOT / "study/comparator_tuning.json").read_text())


def _rehash_authority(payload: dict[str, object]) -> None:
    configurations = payload["configurations"]
    assert isinstance(configurations, list)
    for row in configurations:
        assert isinstance(row, dict)
        row["payload_sha256"] = canonical_sha256(row["payload"])
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = canonical_sha256(unsigned)


def _set_nested(
    payload: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    target: object = payload
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        pytest.param(("schema_version",), True, id="schema-version-type"),
        pytest.param(("contract_id",), "other-contract", id="contract-id"),
        pytest.param(("scope", "data_scope"), "final", id="scope-data"),
        pytest.param(("scope", "final_data_used"), True, id="scope-final"),
        pytest.param(
            ("method_order",), list(reversed(EXPECTED_ORDER)), id="method-order"
        ),
        pytest.param(
            ("scheduled_same_input_ids",),
            ["observed", "capacity-matched-ae"],
            id="scheduled-set",
        ),
        pytest.param(("required_control_ids",), ["observed"], id="control-set"),
        pytest.param(("established_comparator_ids",), ["alra"], id="established-set"),
        pytest.param(("modern_core_ids",), ["scziva"], id="modern-set"),
        pytest.param(("model_seeds",), [42, 43, 45], id="model-seeds"),
        pytest.param(("selection", "metrics"), ["mse"], id="selection-metrics"),
        pytest.param(
            ("selection", "collapse_order"),
            ["retain_biological_draw_units"],
            id="selection-collapse",
        ),
        pytest.param(
            ("selection", "prezero_mechanism"), "sergio", id="selection-prezero"
        ),
        pytest.param(("selection", "pareto_rule"), "changed", id="selection-pareto"),
        pytest.param(("selection", "rank_rule"), "changed", id="selection-rank"),
        pytest.param(
            ("selection", "selection_tuple"),
            ["configuration_id"],
            id="selection-tuple",
        ),
        pytest.param(
            ("selection", "readiness", "minimum_required_controls_complete"),
            1,
            id="readiness-controls",
        ),
        pytest.param(
            (
                "selection",
                "readiness",
                "minimum_established_comparators_selectable",
            ),
            4,
            id="readiness-established",
        ),
        pytest.param(
            ("selection", "readiness", "minimum_modern_core_selectable"),
            2,
            id="readiness-modern",
        ),
        pytest.param(
            ("selection", "receipt_path"), "elsewhere.json", id="selection-receipt"
        ),
        pytest.param(
            ("budgets", "max_configurations_per_method"), 19, id="budget-configs"
        ),
        pytest.param(("budgets", "gpu_seconds_per_method"), 1, id="budget-gpu"),
        pytest.param(("budgets", "cpu_seconds_per_method"), 1, id="budget-cpu"),
        pytest.param(("budgets", "per_run_timeout_seconds"), 1, id="budget-timeout"),
        pytest.param(("budgets", "max_rss_bytes"), 1, id="budget-rss"),
        pytest.param(("budgets", "max_gpu_bytes"), 1, id="budget-gpu-memory"),
        pytest.param(
            ("budgets", "intrinsic_terminal_statuses"),
            ["failed"],
            id="budget-intrinsic-statuses",
        ),
        pytest.param(
            ("budgets", "blocking_statuses"),
            ["budget_exhausted"],
            id="budget-blocking-statuses",
        ),
        pytest.param(("storage", "max_log_receipt_bytes"), 1, id="storage-log"),
        pytest.param(
            ("storage", "max_executor_receipt_bytes"), 1, id="storage-executor"
        ),
        pytest.param(("storage", "max_record_bytes"), 1, id="storage-record"),
        pytest.param(("storage", "max_checkpoint_bytes"), 1, id="storage-checkpoint"),
        pytest.param(("storage", "reserve_bytes"), 1, id="storage-reserve"),
        pytest.param(("smoke", "receipt_path"), "elsewhere.json", id="smoke-path"),
        pytest.param(("smoke", "cells"), 899, id="smoke-cells"),
        pytest.param(("smoke", "genes"), 499, id="smoke-genes"),
        pytest.param(("smoke", "model_seed"), 43, id="smoke-seed"),
        pytest.param(("smoke", "batch_rule"), "changed", id="smoke-batches"),
        pytest.param(("smoke", "count_formula"), "changed", id="smoke-formula"),
        pytest.param(("smoke", "projection_multiplier"), 47, id="smoke-projection"),
        pytest.param(("smoke", "output_retention"), "retained", id="smoke-retention"),
    ),
)
def test_authority_rejects_rehashed_policy_mutation(
    path: tuple[str, ...], replacement: object
) -> None:
    payload = _tracked_payload()
    _set_nested(payload, path, replacement)
    _rehash_authority(payload)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(
            payload, registry=registry, file_sha256="a" * 64
        )


@pytest.mark.parametrize(
    "section_path",
    (
        pytest.param(("scope",), id="scope"),
        pytest.param(("selection",), id="selection"),
        pytest.param(("selection", "readiness"), id="readiness"),
        pytest.param(("budgets",), id="budgets"),
        pytest.param(("storage",), id="storage"),
        pytest.param(("smoke",), id="smoke"),
    ),
)
def test_authority_rejects_rehashed_extra_nested_field(
    section_path: tuple[str, ...],
) -> None:
    payload = _tracked_payload()
    target: object = payload
    for key in section_path:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target["unexpected"] = "forged"
    _rehash_authority(payload)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError, match="missing or extra fields"):
        parse_comparator_tuning_authority(
            payload, registry=registry, file_sha256="a" * 64
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "row-order",
        "row-count",
        "configuration-id",
        "duplicate-configuration-id",
        "duplicate-payload",
        "multiple-defaults",
        "default-payload",
        "nondefault-payload",
    ),
)
def test_authority_rejects_rehashed_grid_mutation(mutation: str) -> None:
    payload = _tracked_payload()
    configurations = payload["configurations"]
    assert isinstance(configurations, list)
    magic_default = configurations[1]
    magic_second = configurations[2]
    assert isinstance(magic_default, dict)
    assert isinstance(magic_second, dict)

    if mutation == "row-order":
        configurations[1], configurations[2] = configurations[2], configurations[1]
    elif mutation == "row-count":
        configurations.pop()
    elif mutation == "configuration-id":
        magic_second["configuration_id"] = "magic-t02"
    elif mutation == "duplicate-configuration-id":
        magic_second["configuration_id"] = magic_default["configuration_id"]
    elif mutation == "duplicate-payload":
        magic_second["payload"] = copy.deepcopy(magic_default["payload"])
    elif mutation == "multiple-defaults":
        magic_second["is_upstream_default"] = True
    elif mutation == "default-payload":
        magic_default["payload"] = copy.deepcopy(magic_second["payload"])
    elif mutation == "nondefault-payload":
        second_payload = magic_second["payload"]
        assert isinstance(second_payload, dict)
        second_payload["diffusion_time"] = 2
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    _rehash_authority(payload)
    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(
            payload, registry=registry, file_sha256="a" * 64
        )


def test_authority_rejects_invalid_file_sha256() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError, match="file SHA-256"):
        parse_comparator_tuning_authority(
            _tracked_payload(), registry=registry, file_sha256="not-a-sha256"
        )


@pytest.mark.parametrize("location", ("root", "configuration"))
def test_authority_rejects_rehashed_extra_structural_field(location: str) -> None:
    payload = _tracked_payload()
    if location == "root":
        payload["unexpected"] = "forged"
    else:
        configurations = payload["configurations"]
        assert isinstance(configurations, list)
        first = configurations[0]
        assert isinstance(first, dict)
        first["unexpected"] = "forged"
    _rehash_authority(payload)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError, match="missing or extra fields"):
        parse_comparator_tuning_authority(
            payload, registry=registry, file_sha256="a" * 64
        )


@pytest.mark.parametrize("malformation", ("noncanonical", "duplicate", "nonfinite"))
def test_loader_rejects_malformed_authority_bytes(
    tmp_path: Path, malformation: str
) -> None:
    payload = _tracked_payload()
    canonical = json.dumps(payload, indent=2).encode() + b"\n"
    if malformation == "noncanonical":
        raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    elif malformation == "duplicate":
        raw = canonical.replace(
            b'  "schema_version": 1,',
            b'  "schema_version": 1,\n  "schema_version": 1,',
            1,
        )
    elif malformation == "nonfinite":
        raw = canonical.replace(b'    "cells": 900,', b'    "cells": NaN,', 1)
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(malformation)
    authority_path = tmp_path / "study/comparator_tuning.json"
    authority_path.parent.mkdir()
    authority_path.write_bytes(raw)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError):
        load_comparator_tuning_authority(
            tmp_path, registry=registry, require_clean=False
        )


def test_tracked_authority_has_exact_grid_and_operational_contract() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )

    assert authority.method_order == tuple(EXPECTED_ORDER)
    assert len(authority.configurations) == 34
    assert {
        method_id: tuple(
            row.configuration_id for row in authority.configurations_for(method_id)
        )
        for method_id in authority.method_order
    } == EXPECTED_ORDER
    assert all(
        sum(row.is_upstream_default for row in authority.configurations_for(method_id))
        == 1
        and authority.configurations_for(method_id)[0].is_upstream_default
        for method_id in authority.method_order
    )
    assert authority.scheduled_same_input_ids == (
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
    assert authority.required_control_ids == ("observed", "capacity-matched-ae")
    assert authority.established_comparator_ids == (
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
    )
    assert authority.modern_core_ids == ("scziva", "afmf", "biaeimpute", "sccr")
    assert authority.model_seeds == (42, 43, 44)
    assert (
        authority.receipt_path
        == "artifacts/study/development/evaluation/comparator_selection.json"
    )
    assert (
        authority.smoke_receipt_path
        == "artifacts/study/development/evaluation/comparator_smoke.json"
    )
    assert (
        DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
        DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
        DEVELOPMENT_MAX_RECORD_BYTES,
        DEVELOPMENT_MAX_CHECKPOINT_BYTES,
        DEVELOPMENT_STORAGE_RESERVE_BYTES,
    ) == (65_536, 65_536, 65_536, 67_108_864, 1_073_741_824)
    raw_authority = (ROOT / "study/comparator_tuning.json").read_bytes()
    tracked_payload = json.loads(raw_authority)
    assert authority.file_sha256 == hashlib.sha256(raw_authority).hexdigest()
    assert authority.payload_sha256 == tracked_payload["payload_sha256"]
    for row in authority.configurations:
        assert row.payload_sha256 == row.observed_payload_sha256
        dataclass_payload = asdict(row.decode())
        if row.method_id == "dca":
            dataclass_payload["hidden_size"] = list(dataclass_payload["hidden_size"])
        assert dataclass_payload == dict(row.payload)
        assert encode_comparator_configuration(row.decode()) == dict(row.payload)
