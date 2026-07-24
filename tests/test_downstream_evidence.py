from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import sys
import zlib
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@lru_cache(maxsize=1)
def _task15_final_module():
    path = Path(__file__).with_name("test_final_runner.py")
    spec = importlib.util.spec_from_file_location("_task16_downstream_factory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _frozen_downstream_authorities(unavailable_method: str | None = None):
    from maskimpute_benchmark.final_runner import _frozen_method_plan_authority

    module = _task15_final_module()
    frozen = module._direct_frozen_method(unavailable_method=unavailable_method)
    registry = module._full_registry()
    _rows, configurations = _frozen_method_plan_authority(frozen, registry)
    return frozen, registry, configurations


def test_downstream_configuration_schema_preserves_complete_direct_authority() -> None:
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.downstream_evidence import _legacy_configuration_payload
    from maskimpute_benchmark.runner import direct_bound_comparator_value

    frozen, _registry, configurations = _frozen_downstream_authorities()
    by_method = {value.method_id: value for value in configurations}

    magic = _legacy_configuration_payload(by_method["magic"])
    assert "configuration_sha256" not in magic
    assert direct_equal(
        magic["comparator_configuration"],
        frozen["selected_comparator_configurations"]["magic"],
    )
    assert magic["comparator_nonexecution_identity"] is None
    expected_magic = by_method["magic"].comparator_configuration
    assert expected_magic is not None
    assert direct_equal(
        magic["comparator_configuration"],
        direct_bound_comparator_value(expected_magic),
    )

    unavailable_frozen, _registry, unavailable_configurations = (
        _frozen_downstream_authorities("biaeimpute")
    )
    unavailable_by_method = {
        value.method_id: value for value in unavailable_configurations
    }
    unavailable = _legacy_configuration_payload(unavailable_by_method["biaeimpute"])
    assert "configuration_sha256" not in unavailable
    assert unavailable["comparator_configuration"] is None
    assert direct_equal(
        unavailable["comparator_nonexecution_identity"],
        unavailable_frozen["unavailable_comparator_nonexecution_identities"][
            "biaeimpute"
        ],
    )


def test_direct_downstream_configuration_decoder_restores_exact_typed_authority() -> (
    None
):
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.downstream_evidence import (
        _configuration_from_payload,
        _legacy_configuration_payload,
    )
    from maskimpute_benchmark.final_runner import FrozenPlanMethodAuthority

    _frozen, _registry, selected_configurations = _frozen_downstream_authorities()
    _unavailable_frozen, _registry, unavailable_configurations = (
        _frozen_downstream_authorities("biaeimpute")
    )
    selected = {value.method_id: value for value in selected_configurations}["magic"]
    unavailable = {value.method_id: value for value in unavailable_configurations}[
        "biaeimpute"
    ]

    for authority in (selected, unavailable):
        payload = _legacy_configuration_payload(authority)
        decoded = _configuration_from_payload(payload)

        assert isinstance(decoded, FrozenPlanMethodAuthority)
        assert direct_equal(decoded.to_dict(), authority.to_dict())
        assert direct_equal(_legacy_configuration_payload(decoded), payload)


_COMPARATOR_METHOD_BINDING_FIELDS = (
    "method_id",
    "execution_scope",
    "integration_status",
    "adapter_key",
    "environment_id",
    "environment_status",
    "source_kind",
    "source_url",
    "source_revision",
    "source_tree",
    "source_cache_path",
    "source_freeze_binding",
    "input_scale",
    "output_scale",
    "stochastic",
    "seed_policy",
    "gpu_mode",
    "cpu_cores",
    "timeout_seconds",
    "max_rss_gib",
    "max_gpu_gib",
    "preserves_observed_positives",
)


def _forged_method_component(value: object) -> object:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 1.0
    if value is None:
        return "forged-optional-binding"
    assert isinstance(value, str)
    return f"{value}-forged"


@pytest.mark.parametrize("authority_kind", ("selected", "nonexecution"))
@pytest.mark.parametrize("field_name", _COMPARATOR_METHOD_BINDING_FIELDS)
def test_persisted_direct_decoder_rejects_canonical_method_binding_mutations(
    authority_kind: str,
    field_name: str,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    unavailable = authority_kind == "nonexecution"
    _frozen, _registry, configurations = _frozen_downstream_authorities(
        "biaeimpute" if unavailable else None
    )
    method_id = "biaeimpute" if unavailable else "magic"
    authority = {value.method_id: value for value in configurations}[method_id]
    payload = copy.deepcopy(downstream._legacy_configuration_payload(authority))
    if unavailable:
        identity = payload["comparator_nonexecution_identity"]
        assert isinstance(identity, dict)
        targets = [identity["method"]]
        targets.extend(
            row["configuration"]["method"]
            for row in identity["configuration_terminal_denominator"]
        )
    else:
        selected = payload["comparator_configuration"]
        assert isinstance(selected, dict)
        targets = [selected["method"]]
    for target in targets:
        target[field_name] = _forged_method_component(target[field_name])

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="persisted direct configuration authority",
    ):
        downstream._configuration_from_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    ("omitted_configuration", "reordered_configurations", "bool_schema_version"),
)
def test_persisted_direct_decoder_rejects_noncanonical_nonexecution_denominator(
    mutation: str,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    _frozen, _registry, configurations = _frozen_downstream_authorities("biaeimpute")
    authority = {value.method_id: value for value in configurations}["biaeimpute"]
    payload = copy.deepcopy(downstream._legacy_configuration_payload(authority))
    identity = payload["comparator_nonexecution_identity"]
    assert isinstance(identity, dict)
    denominator = identity["configuration_terminal_denominator"]
    assert isinstance(denominator, list) and len(denominator) > 1
    if mutation == "omitted_configuration":
        denominator.pop()
    elif mutation == "reordered_configurations":
        denominator[0], denominator[1] = denominator[1], denominator[0]
    else:
        assert mutation == "bool_schema_version"
        identity["schema_version"] = True

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="persisted direct comparator nonexecution",
    ):
        downstream._configuration_from_payload(payload)


def test_generic_downstream_builder_accepts_decoded_direct_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    _frozen, _registry, configurations = _frozen_downstream_authorities()
    authority = {value.method_id: value for value in configurations}["magic"]
    decoded = downstream._configuration_from_payload(
        downstream._legacy_configuration_payload(authority)
    )
    dataset_path = tmp_path / "dataset.h5ad"
    _dataset_value, cells, _genes = _dataset(dataset_path)
    dataset = _dataset_binding(dataset_path, cells)
    source = tmp_path / "direct-source"
    source.mkdir()
    entry = downstream.DownstreamPlanEntry(
        ordinal=1,
        source_record_path="records/00000001.json",
        source_record_sha256="1" * 64,
        run_id="direct-generic-magic",
        method_id="magic",
        dataset_id=dataset.dataset_id,
        source_dataset_sha256=dataset.dataset_sha256,
        mechanism=dataset.mechanism,
        biological_id=dataset.biological_id,
        technical_view=dataset.technical_view,
        model_seed=42,
        configuration_id=authority.configuration_id,
        configuration_sha256=None,
        configuration_kind=authority.kind,
        method_artifact_sha256=None,
        comparator_configuration=authority.comparator_configuration,
        comparator_nonexecution_identity=None,
        method_input_sha256=dataset.method_input_sha256,
        retained_cell_ids_sha256=dataset.retained_cell_ids_sha256,
        status="failed",
        reason="adapter_nonzero_exit",
        evaluator_output_sha256=None,
        evaluator_output_path=None,
        evaluator_output_file_sha256=None,
        evaluator_output_shape=None,
        evaluator_output_encoding=None,
        evaluator_output_uncompressed_nbytes=None,
        evaluator_output_uncompressed_sha256=None,
    )
    source_bundle = SimpleNamespace(
        manifest_path="checkpoint.json",
        manifest_file_sha256="2" * 64,
        manifest_payload_sha256="3" * 64,
        source_plan_sha256="4" * 64,
        source_input_hashes_sha256="5" * 64,
        records=({},),
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _value: None)
    monkeypatch.setattr(downstream, "_development_source", lambda _root: source_bundle)
    monkeypatch.setattr(
        downstream,
        "_validated_plan_entry",
        lambda *_args, **_kwargs: entry,
    )

    plan = downstream.build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(dataset,),
        configurations=(decoded,),
    )

    assert isinstance(plan.configurations[0], downstream.FrozenPlanMethodAuthority)
    assert plan.configurations[0] == decoded


@pytest.mark.parametrize(
    ("status", "reason"),
    (
        ("timeout", "adapter_timeout"),
        ("resource_exceeded", "peak_gpu_memory_limit_exceeded"),
    ),
)
def test_downstream_selected_comparator_terminal_rows_retain_complete_identity(
    status: str,
    reason: str,
) -> None:
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.downstream_evidence import DownstreamPlanEntry
    from maskimpute_benchmark.runner import direct_bound_comparator_value

    _frozen, _registry, configurations = _frozen_downstream_authorities()
    authority = {value.method_id: value for value in configurations}["magic"]
    configuration = authority.comparator_configuration
    assert configuration is not None
    entry = DownstreamPlanEntry(
        ordinal=1,
        source_record_path="records/00000001.json",
        source_record_sha256="1" * 64,
        run_id="direct-terminal-magic",
        method_id="magic",
        dataset_id="dataset-direct",
        source_dataset_sha256="2" * 64,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        model_seed=42,
        configuration_id=authority.configuration_id,
        configuration_sha256=None,
        configuration_kind=authority.kind,
        method_artifact_sha256=None,
        comparator_configuration=configuration,
        comparator_nonexecution_identity=None,
        method_input_sha256="3" * 64,
        retained_cell_ids_sha256="4" * 64,
        status=status,
        reason=reason,
        evaluator_output_sha256=None,
        evaluator_output_path=None,
        evaluator_output_file_sha256=None,
        evaluator_output_shape=None,
        evaluator_output_encoding=None,
        evaluator_output_uncompressed_nbytes=None,
        evaluator_output_uncompressed_sha256=None,
    )

    payload = entry.to_dict()
    assert "configuration_sha256" not in payload
    assert "method_artifact_sha256" not in payload
    assert payload["status"] == status
    assert payload["reason"] == reason
    assert direct_equal(
        payload["comparator_configuration"],
        direct_bound_comparator_value(configuration),
    )
    assert payload["comparator_nonexecution_identity"] is None


def test_downstream_terminal_endpoint_preserves_direct_status_reason_and_identity() -> (
    None
):
    from maskimpute_benchmark.direct_values import direct_equal
    from maskimpute_benchmark.downstream_evaluation import terminal_downstream_endpoints
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamPlanEntry,
        _endpoint_row,
    )
    from maskimpute_benchmark.runner import direct_bound_comparator_value

    _frozen, _registry, configurations = _frozen_downstream_authorities()
    authority = {value.method_id: value for value in configurations}["magic"]
    configuration = authority.comparator_configuration
    assert configuration is not None
    entry = DownstreamPlanEntry(
        ordinal=1,
        source_record_path="records/00000001.json",
        source_record_sha256="1" * 64,
        run_id="direct-timeout-magic",
        method_id="magic",
        dataset_id="dataset-direct",
        source_dataset_sha256="2" * 64,
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        model_seed=42,
        configuration_id=authority.configuration_id,
        configuration_sha256=None,
        configuration_kind=authority.kind,
        method_artifact_sha256=None,
        comparator_configuration=configuration,
        comparator_nonexecution_identity=None,
        method_input_sha256="3" * 64,
        retained_cell_ids_sha256="4" * 64,
        status="timeout",
        reason="adapter_timeout",
        evaluator_output_sha256=None,
        evaluator_output_path=None,
        evaluator_output_file_sha256=None,
        evaluator_output_shape=None,
        evaluator_output_encoding=None,
        evaluator_output_uncompressed_nbytes=None,
        evaluator_output_uncompressed_sha256=None,
    )
    endpoint = terminal_downstream_endpoints(
        "upstream_run_not_completed",
        procedure="terminal_upstream_run_not_completed",
    )[0]
    row = _endpoint_row(SimpleNamespace(source_kind="final"), entry, endpoint)

    assert "configuration_sha256" not in row
    assert "method_artifact_sha256" not in row
    assert row["status"] == "timeout"
    assert row["upstream_status"] == "timeout"
    assert row["upstream_reason"] == "adapter_timeout"
    assert row["value"] is None
    assert direct_equal(
        row["comparator_configuration"],
        direct_bound_comparator_value(configuration),
    )
    assert row["comparator_nonexecution_identity"] is None


def test_downstream_final_source_accepts_complete_direct_execution_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.runner import direct_bound_comparator_value

    _frozen, _registry, configurations = _frozen_downstream_authorities()
    authority = {value.method_id: value for value in configurations}["magic"]
    configuration = authority.comparator_configuration
    assert configuration is not None
    run = {name: None for name in downstream._DIRECT_FINAL_RUN_FIELDS}
    run.update(
        {
            "configuration_kind": "comparator_tuning",
            "native_output_sha256": None,
            "native_output_retention": "not_available",
            "native_output_path": None,
            "native_output_file_sha256": None,
            "native_output_shape": None,
            "native_output_dtype": None,
        }
    )
    request = {
        "request_kind": "frozen_comparator_direct",
        "configuration": direct_bound_comparator_value(configuration),
        "dataset_id": "dataset-direct",
        "execution_authority_sha256": "1" * 64,
        "method_input_sha256": "2" * 64,
        "model_seed": 42,
    }
    monkeypatch.setattr(
        downstream,
        "_validate_prezero_source_schema",
        lambda _value, *, run: None,
    )

    downstream._validate_source_record_schema(
        {
            "run": run,
            "metrics": [],
            "p_pre_zero_evidence": {},
            "execution_request": request,
        },
        source_kind="final",
    )


def test_direct_downstream_projection_routes_only_closed_direct_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark import development_evaluation
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        validate_direct_comparator_projection,
    )

    expected = {
        "comparator_authority": {
            "path": "study/comparator_tuning.json",
            "schema_version": 2,
            "authority_revision": "fair-comparator-direct-v1",
        },
        "selected_comparators": {
            "magic": {
                "configuration_id": "magic-t01-default",
                "payload": {"solver": "exact"},
            }
        },
    }
    calls = []

    evidence = SimpleNamespace(
        selected_by_method={
            "magic": {
                "configuration": {
                    "configuration_id": "magic-t01-default",
                    "payload": {"solver": "exact"},
                }
            }
        }
    )

    def project(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return evidence

    monkeypatch.setattr(
        development_evaluation,
        "project_direct_comparator_evidence",
        project,
    )
    plan = SimpleNamespace(identity_mode="direct-v1")
    validated = validate_direct_comparator_projection(
        expected,
        Path("synthetic-checkpoint.json"),
        plan,
        repository=Path("synthetic-repository"),
        registry=object(),
        prepared_datasets={},
        comparator_reference=SimpleNamespace(
            path="study/comparator_tuning.json",
            schema_version=2,
            authority_revision="fair-comparator-direct-v1",
        ),
        comparator_authority=object(),
        comparator_selection={"complete": True},
        runner_authority=object(),
        datasets=(),
    )
    assert validated == expected
    assert len(calls) == 1
    assert calls[0][1]["repository"] == Path("synthetic-repository")
    assert calls[0][1]["comparator_selection"] == {"complete": True}

    changed = copy.deepcopy(expected)
    changed["selected_comparators"]["magic"]["payload"]["solver"] = "drifted"
    with pytest.raises(DownstreamEvidenceError, match="projection differs"):
        validate_direct_comparator_projection(
            changed,
            Path("synthetic-checkpoint.json"),
            plan,
            repository=Path("synthetic-repository"),
            registry=object(),
            prepared_datasets={},
            comparator_reference=SimpleNamespace(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=object(),
            comparator_selection={"complete": True},
            runner_authority=object(),
            datasets=(),
        )

    mixed = copy.deepcopy(expected)
    mixed["selection_receipt"] = {"selection_complete": True}
    with pytest.raises(DownstreamEvidenceError, match="schema"):
        validate_direct_comparator_projection(
            mixed,
            Path("synthetic-checkpoint.json"),
            plan,
            repository=Path("synthetic-repository"),
            registry=object(),
            prepared_datasets={},
            comparator_reference=SimpleNamespace(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=object(),
            comparator_selection={"complete": True},
            runner_authority=object(),
            datasets=(),
        )

    with pytest.raises(DownstreamEvidenceError, match="identity mode"):
        validate_direct_comparator_projection(
            expected,
            Path("synthetic-checkpoint.json"),
            SimpleNamespace(identity_mode="legacy-v1"),
            repository=Path("synthetic-repository"),
            registry=object(),
            prepared_datasets={},
            comparator_reference=SimpleNamespace(
                path="study/comparator_tuning.json",
                schema_version=2,
                authority_revision="fair-comparator-direct-v1",
            ),
            comparator_authority=object(),
            comparator_selection={"complete": True},
            runner_authority=object(),
            datasets=(),
        )


def test_development_production_wrapper_routes_direct_base_through_real_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.comparator_tuning as comparator_tuning
    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.methods as methods
    import maskimpute_benchmark.runner as runner

    root = Path(downstream.__file__).absolute().parents[1]
    source = tmp_path / "direct-development"

    @dataclass(frozen=True)
    class DirectMethod:
        method_id: str

    magic = DirectMethod("magic")
    observed = DirectMethod("observed")
    selected_configuration = SimpleNamespace(
        method=magic,
        configuration_id="magic-t01-default",
        configuration_kind="comparator_tuning",
        payload=(("solver", "exact"),),
        requires_count_score=False,
        requires_calibration=False,
    )
    nonselected_configuration = SimpleNamespace(
        method=magic,
        configuration_id="magic-t02-nonselected",
        configuration_kind="comparator_tuning",
        payload=(("solver", "approximate"),),
        requires_count_score=False,
        requires_calibration=False,
    )
    control_configuration = SimpleNamespace(
        method=observed,
        configuration_id="registry-default",
        configuration_kind="registry",
        payload=(),
        requires_count_score=False,
        requires_calibration=False,
    )

    def record(
        run_id: str,
        method_id: str,
        configuration_id: str,
        configuration_kind: str,
    ) -> dict[str, object]:
        return {
            "run": {
                "run_id": run_id,
                "identity": {
                    "method": {"method_id": method_id},
                    "configuration_id": configuration_id,
                    "configuration_kind": configuration_kind,
                    "dataset_id": "dataset-synthetic",
                    "mechanism": "symsim",
                    "biological_id": "synthetic-draw-001",
                    "technical_view": "moderate",
                    "model_seed": None,
                },
                "status": "completed",
                "reason": None,
            }
        }

    direct_plan = SimpleNamespace(
        identity_mode="direct-v1",
        configurations=(
            selected_configuration,
            nonselected_configuration,
            control_configuration,
        ),
        entries=(
            SimpleNamespace(run_id="run-magic-selected"),
            SimpleNamespace(run_id="run-magic-nonselected"),
            SimpleNamespace(run_id="run-observed-control"),
        ),
    )
    checkpoint = {
        "schema_version": 1,
        "identity_mode": "direct-v1",
        "authority_revision": "fair-comparator-direct-v1",
        "plan_snapshot": {
            "comparator_smoke_receipt": {"status": "complete"},
            "comparator_smoke_receipt_bytes": [123, 125],
        },
        "input_descriptors": [],
        "planned_run_count": 3,
        "status": "completed",
        "records": [
            record(
                "run-magic-selected",
                "magic",
                "magic-t01-default",
                "comparator_tuning",
            ),
            record(
                "run-magic-nonselected",
                "magic",
                "magic-t02-nonselected",
                "comparator_tuning",
            ),
            record(
                "run-observed-control",
                "observed",
                "registry-default",
                "registry",
            ),
        ],
    }
    checkpoint_file_sha256 = _write_canonical(source / "checkpoint.json", checkpoint)
    assert "input_hashes" not in checkpoint

    class DirectCheckpoint(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            if key == "input_hashes":
                pytest.fail("legacy input_hashes were read")
            return super().get(key, default)

    guarded_checkpoint = DirectCheckpoint(checkpoint)
    strict_json = downstream._strict_json

    def guarded_strict_json(path: Path, name: str):
        if path == source / "checkpoint.json":
            return (
                guarded_checkpoint,
                b"synthetic-direct-checkpoint",
                (checkpoint_file_sha256),
            )
        return strict_json(path, name)

    monkeypatch.setattr(downstream, "_strict_json", guarded_strict_json)

    registry = SimpleNamespace(methods=())
    runner_bindings = (object(),)
    prepared = {"dataset-synthetic": object()}
    datasets = (
        downstream.DatasetEvidenceBinding(
            dataset_id="dataset-synthetic",
            path=str((tmp_path / "evaluator.h5ad").absolute()),
            file_sha256="1" * 64,
            dataset_sha256="2" * 64,
            mechanism="symsim",
            biological_id="synthetic-draw-001",
            technical_view="moderate",
            method_input_sha256="3" * 64,
            dataset_qc_policy_sha256="4" * 64,
            excluded_cell_count=0,
            excluded_cell_ids_sha256="5" * 64,
            retained_cell_count=1,
            retained_cell_ids_sha256="6" * 64,
            retained_gene_count=1,
            observed_zero_count=1,
            retained_cell_ids=("cell-001",),
            gene_ids=("gene-001",),
        ),
    )
    reference = SimpleNamespace(
        path="study/comparator_tuning.json",
        schema_version=2,
        authority_revision="fair-comparator-direct-v1",
    )
    authority = SimpleNamespace(
        configurations=(),
        comparator_tuning_reference=reference,
        comparator_tuning=object(),
    )
    comparator_selection = {
        "path": "artifacts/study/development/comparator-selection.json",
        "receipt": {"selection_complete": True},
        "selected_by_method": {
            "magic": {
                "configuration": {
                    "configuration_id": "magic-t01-default",
                    "payload": {"solver": "exact"},
                },
                "authority_reference": {},
                "method": {},
            }
        },
        "nonexecution_identity_by_method": {},
        "ready_comparison_population_ids": ["magic"],
    }
    expected_handoff = {
        "comparator_authority": {
            "path": "study/comparator_tuning.json",
            "schema_version": 2,
            "authority_revision": "fair-comparator-direct-v1",
        },
        "selected_comparators": {
            "magic": {
                "configuration_id": "magic-t01-default",
                "payload": {"solver": "exact"},
            }
        },
    }
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(methods, "load_method_registry", lambda _path: registry)
    monkeypatch.setattr(runner, "load_runner_authority", lambda: authority)
    monkeypatch.setattr(
        runner,
        "load_prepared_development_panel",
        lambda _authority: (runner_bindings, prepared),
    )
    monkeypatch.setattr(
        runner,
        "build_competition_plan",
        lambda *args, **kwargs: pytest.fail("legacy planner was called"),
    )
    monkeypatch.setattr(
        runner,
        "build_fair_comparator_plan",
        lambda *args, **kwargs: direct_plan,
    )
    monkeypatch.setattr(
        downstream,
        "bind_prepared_evaluator_panel",
        lambda *args, **kwargs: datasets,
    )
    monkeypatch.setattr(
        comparator_tuning,
        "load_comparator_selection_receipt",
        lambda *args, **kwargs: {"selection_complete": True},
    )
    monkeypatch.setattr(
        comparator_tuning,
        "comparator_selection_projection",
        lambda _receipt: object(),
    )
    monkeypatch.setattr(
        comparator_tuning,
        "comparator_selection_projection_value",
        lambda _projection: comparator_selection,
    )

    def validate(projection: object, *args: object, **kwargs: object) -> object:
        calls.append(("adapter", (projection, args, kwargs)))
        assert projection == expected_handoff
        assert args[0] == source / "checkpoint.json"
        assert args[1] is direct_plan
        assert kwargs["comparator_selection"] is comparator_selection
        return expected_handoff

    monkeypatch.setattr(downstream, "validate_direct_comparator_projection", validate)

    actual = downstream.build_development_downstream_evidence_plan(
        root,
        checkpoint_directory=source,
    )

    assert [name for name, _payload in calls] == ["adapter"]
    assert [(entry.method_id, entry.configuration_id) for entry in actual.entries] == [
        ("magic", "magic-t01-default"),
        ("observed", "registry-default"),
    ]
    assert [
        (configuration.method_id, configuration.kind)
        for configuration in actual.configurations
    ] == [
        ("magic", "selected_comparator"),
        ("observed", "direct_control"),
    ]
    assert all(
        isinstance(configuration, downstream.ProjectedDownstreamConfiguration)
        for configuration in actual.configurations
    )
    assert all(
        entry.status == "unavailable"
        and entry.reason == "direct_evaluator_output_not_retained"
        and entry.evaluator_output_sha256 is None
        and entry.evaluator_output_path is None
        and entry.evaluator_output_file_sha256 is None
        and entry.evaluator_output_shape is None
        and entry.evaluator_output_encoding is None
        and entry.evaluator_output_uncompressed_nbytes is None
        and entry.evaluator_output_uncompressed_sha256 is None
        for entry in actual.entries
    )
    assert "magic-t02-nonselected" not in {
        configuration.configuration_id for configuration in actual.configurations
    }

    destination = tmp_path / "persisted-downstream"
    running = downstream.run_downstream_evidence(
        actual,
        destination,
        max_denominators=0,
    )
    assert running["status"] == "running"
    assert running["recorded_denominator_count"] == 0
    reloaded = downstream.load_downstream_evidence_plan(destination)
    assert reloaded.to_dict() == actual.to_dict()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_canonical(path: Path, value: object) -> str:
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _resign_scaling_evidence(evidence: dict[str, object]) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256

    evidence_body = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence_body)


def _resign_trajectory_evidence(evidence: dict[str, object]) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256

    evidence_body = {
        key: value for key, value in evidence.items() if key != "evidence_sha256"
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence_body)


def _rebind_trajectory_primary_plan(
    evidence: dict[str, object], primary_plan_sha256: str
) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256

    plan = evidence["plan"]
    validation = evidence["execution_validation"]
    assert isinstance(plan, dict)
    assert isinstance(validation, dict)
    input_hashes = plan["input_hashes"]
    assert isinstance(input_hashes, dict)
    input_hashes["primary_final_plan_sha256"] = primary_plan_sha256
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(plan_body)
    validation["trajectory_plan_sha256"] = plan["plan_sha256"]
    validation_body = {
        key: value for key, value in validation.items() if key != "validation_sha256"
    }
    validation["validation_sha256"] = canonical_sha256(validation_body)
    _resign_trajectory_evidence(evidence)


def _resign_evaluation_receipt(
    receipt_path: Path,
    receipt: dict[str, object],
) -> None:
    from maskimpute_benchmark.protocol import canonical_sha256

    result_manifest = receipt["result_manifest"]
    assert isinstance(result_manifest, dict)
    receipt["result_manifest_sha256"] = canonical_sha256(result_manifest)
    _write_canonical(receipt_path, receipt)


def _cell_id_sha256(cell_ids: tuple[str, ...]) -> str:
    payload = bytearray(b"maskimpute-external-cell-ids-v1\0")
    payload.extend(struct.pack("<Q", len(cell_ids)))
    for cell_id in cell_ids:
        encoded = cell_id.encode("utf-8")
        payload.extend(struct.pack("<Q", len(encoded)))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def _dataset(path: Path) -> tuple[ad.AnnData, tuple[str, ...], tuple[str, ...]]:
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    counts = np.asarray(
        [
            [12, 8, 1, 0],
            [11, 7, 1, 0],
            [10, 8, 2, 0],
            [0, 1, 8, 12],
            [0, 1, 7, 11],
            [0, 2, 8, 10],
        ],
        dtype=np.int64,
    )
    cells = tuple(f"cell-{index}" for index in range(1, 7))
    genes = tuple(f"gene-{index}" for index in range(1, 5))
    dataset = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "dataset_id": ["dev-symsim-01"] * 6,
                "mechanism": ["symsim"] * 6,
                "condition": ["moderate"] * 6,
                "biological_id": ["draw-01"] * 6,
                "technical_view": ["moderate"] * 6,
                "draw": np.ones(6, dtype=np.int64),
                "library_size": np.sum(counts, axis=1, dtype=np.int64),
                "group": ["pop-1"] * 3 + ["pop-2"] * 3,
            },
            index=cells,
        ),
        var=pd.DataFrame(
            {
                "marker_group_1": [True, True, False, False],
                "marker_group_2": [False, False, True, True],
            },
            index=genes,
        ),
        layers={"pre_capture_counts": counts + 1},
    )
    dataset.uns.update(
        {
            "truth_kind": "exact_pre_capture",
            "primary_truth_layer": "pre_capture_counts",
            "provenance": {
                "source": "test-fixture",
                "source_sha256": "1" * 64,
                "software": "test",
                "software_version": "1",
                "parameters": {},
                "seeds": {},
            },
            "normalization": {"input": "raw_umi_counts", "size_factor": "none"},
        }
    )
    dataset.write_h5ad(path)
    persisted = ad.read_h5ad(path)
    assert benchmark_dataset_sha256(persisted) == benchmark_dataset_sha256(dataset)
    return persisted, cells, genes


def _common_output(dataset: ad.AnnData) -> np.ndarray:
    counts = np.asarray(dataset.X, dtype=np.float64)
    libraries = np.sum(counts, axis=1)
    return np.log2(counts * (10_000.0 / libraries)[:, None] + 1.0)


def _evaluator_output_sha256(run: dict[str, object], raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"maskimpute-evaluator-log2-cp10k-output-v1\0")
    digest.update(
        _canonical_bytes(
            {
                "run_id": run["run_id"],
                "method_input_sha256": run["method_input_sha256"],
                "retained_cell_ids_sha256": run["retained_cell_ids_sha256"],
                "shape": run["evaluator_output_shape"],
                "dtype": "<f8",
                "scale": "log2_cp10k_plus_1",
            }
        )
    )
    digest.update(raw)
    return digest.hexdigest()


def _test_configuration_authority():
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    candidate = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="default",
        kind="candidate_search",
        payload={
            "configuration_id": "default",
            "method_id": "maskimpute",
            "variant": "downstream-test",
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    registry = load_method_registry(Path("study/methods.json"))
    magic = AuthorizedConfiguration.registry_default(registry.by_id("magic"))
    return candidate, magic


def _fake_evaluated_round_binding(
    repository: Path,
    round_root: Path,
    **overrides: object,
):
    from maskimpute_benchmark.downstream_evidence import EvaluatedRoundBinding

    values: dict[str, object] = {
        "repository_root": str(repository.absolute()),
        "round_root": str(round_root.absolute()),
        "round_id": round_root.name,
        "evaluation_receipt_path": "evaluation_receipt.json",
        "evaluation_receipt_file_sha256": "1" * 64,
        "evaluation_receipt_payload_sha256": "2" * 64,
        "result_manifest_sha256": "3" * 64,
        "final_plan_sha256": "4" * 64,
        "final_execution_manifest_path": (
            "results/final/execution/execution_manifest.json"
        ),
        "final_execution_manifest_file_sha256": "5" * 64,
        "final_execution_manifest_payload_sha256": "6" * 64,
        "execution_validation_sha256": "7" * 64,
        "storage_preflight_sha256": "8" * 64,
        "scaling_evidence_sha256": "9" * 64,
        "scaling_plan_sha256": "a" * 64,
        "scaling_checkpoint_path": "results/scaling/checkpoints/00000024.json",
        "scaling_checkpoint_file_sha256": "b" * 64,
        "scaling_checkpoint_payload_sha256": "c" * 64,
        "scaling_checkpoint_history_sha256": "d" * 64,
        "scaling_checkpoint_history_count": 24,
        "scaling_result_files_sha256": "e" * 64,
        "scaling_result_file_count": 100,
        "trajectory_evidence_sha256": "f" * 64,
        "trajectory_plan_sha256": "0" * 64,
        "trajectory_execution_claim_sha256": "d" * 64,
        "trajectory_execution_environment_sha256": "e" * 64,
        "trajectory_dataset_id": "trajectory-exact-latent-01",
        "trajectory_dataset_sha256": "1" * 64,
        "trajectory_dataset_file_sha256": "2" * 64,
        "trajectory_dataset_receipt_file_sha256": "3" * 64,
        "trajectory_dataset_receipt_payload_sha256": "4" * 64,
        "trajectory_source_id": "registered-synthetic-trajectory-v1",
        "trajectory_root_cell_id": "trajectory-cell-000001",
        "trajectory_registered_authority_sha256": "f" * 64,
        "trajectory_registered_binding_sha256": "0" * 64,
        "trajectory_authority_sha256": "5" * 64,
        "trajectory_authority_file_sha256": "6" * 64,
        "trajectory_execution_manifest_path": (
            "results/trajectory/execution/execution_manifest.json"
        ),
        "trajectory_execution_manifest_file_sha256": "7" * 64,
        "trajectory_execution_manifest_payload_sha256": "8" * 64,
        "trajectory_execution_validation_sha256": "9" * 64,
        "trajectory_record_payload_sha256s_sha256": "a" * 64,
        "trajectory_status_counts_sha256": "b" * 64,
        "trajectory_planned_run_count": 1,
        "trajectory_result_files_sha256": "c" * 64,
        "trajectory_result_file_count": 7,
    }
    values.update(overrides)
    return EvaluatedRoundBinding(**values)


def _run(
    *,
    run_id: str,
    method_id: str,
    dataset_sha256: str,
    cell_ids: tuple[str, ...],
    status: str,
    reason: str | None,
) -> dict[str, object]:
    candidate, magic = _test_configuration_authority()
    configuration = candidate if method_id == "maskimpute" else magic
    return {
        "run_id": run_id,
        "method_id": method_id,
        "dataset_id": "dev-symsim-01",
        "source_dataset_sha256": dataset_sha256,
        "mechanism": "symsim",
        "biological_id": "draw-01",
        "technical_view": "moderate",
        "model_seed": 42,
        "configuration_id": configuration.configuration_id,
        "configuration_sha256": configuration.configuration_sha256,
        "configuration_kind": configuration.kind,
        "requires_count_score": False,
        "requires_calibration": False,
        "method_input_sha256": "3" * 64,
        "dataset_qc_policy_sha256": "4" * 64,
        "excluded_cell_count": 0,
        "excluded_cell_ids_sha256": "5" * 64,
        "retained_cell_count": len(cell_ids),
        "retained_cell_ids_sha256": _cell_id_sha256(cell_ids),
        "retained_gene_count": 0,
        "observed_zero_count": 0,
        "status": status,
        "reason": reason,
        "runtime_seconds": 1.0,
        "peak_rss_bytes": 1,
        "peak_gpu_bytes": 0,
        "rss_measurement": "test_measurement",
        "gpu_measurement": "not_measured",
        "calibration_artifact_sha256": None,
        "calibration_context_sha256": None,
        "calibration_training_manifest_sha256s": [],
        "calibration_held_out_manifest_sha256s": [],
        "calibration_fold_calibrator_sha256": None,
        "stdout_sha256": "6" * 64,
        "stderr_sha256": "7" * 64,
        "native_output_sha256": None,
        "evaluator_output_sha256": None,
        "stdout_path": f"runs/{run_id}.stdout",
        "stdout_file_sha256": "8" * 64,
        "stderr_path": f"runs/{run_id}.stderr",
        "stderr_file_sha256": "9" * 64,
        "native_output_path": None,
        "native_output_file_sha256": None,
        "native_output_shape": None,
        "native_output_dtype": None,
        "native_output_scale": None,
        "evaluator_output_path": None,
        "evaluator_output_file_sha256": None,
        "evaluator_output_shape": None,
        "evaluator_output_dtype": None,
        "evaluator_scale": None,
    }


def _current_prezero_evidence(run: dict[str, object]) -> dict[str, object]:
    """Return the exact persisted score-evidence envelope used by current stores."""

    identity = {
        name: run[name]
        for name in (
            "run_id",
            "method_id",
            "dataset_id",
            "source_dataset_sha256",
            "mechanism",
            "biological_id",
            "technical_view",
            "model_seed",
            "configuration_id",
            "configuration_sha256",
            "method_input_sha256",
            "retained_cell_ids_sha256",
        )
    }
    completed_score = run["method_id"] == "maskimpute" and run["status"] == "completed"
    status = run["status"] if run["method_id"] == "maskimpute" else "not_applicable"
    reason = (
        run["reason"]
        if run["method_id"] == "maskimpute"
        else "method_does_not_emit_p_pre_zero"
    )
    policy = (
        {
            "schema_version": 2,
            "probability_semantics": (
                "pre_capture_count_is_zero_given_observed_counts"
            ),
            "evaluation_domain": "observed_zero_entries_only",
            "score_source": "direct",
            "score_artifact_sha256": "a" * 64,
            "score_input_sha256": "b" * 64,
            "score_config_sha256": "c" * 64,
            "calibration_file_sha256": "d" * 64,
            "calibration_payload_sha256": "e" * 64,
            "calibration_algorithm": "identity",
            "calibration_scope": "retained_all_development",
            "calibration_equivalence_reason": "direct_score_requires_no_calibration",
        }
        if completed_score
        else None
    )
    matrix = (
        {
            "shape": [run["retained_cell_count"], run["retained_gene_count"]],
            "dtype": "<f8",
            "content_sha256": "f" * 64,
            "semantic_sha256": "0" * 64,
        }
        if completed_score
        else {
            "shape": None,
            "dtype": None,
            "content_sha256": None,
            "semantic_sha256": None,
        }
    )
    metric_status = "completed" if completed_score else status
    metrics = {
        name: {
            "value": 0.5 if completed_score else None,
            "n": run["observed_zero_count"],
            "status": metric_status,
            "reason": None if completed_score else reason,
        }
        for name in (
            "auroc",
            "auprc",
            "brier",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
            "ece",
        )
    }
    overall = {
        "stratum_type": "overall",
        "label": "all_observed_zeros",
        "lower": None,
        "upper": None,
        "n": run["observed_zero_count"],
        "metrics": metrics,
        "reliability_bins": [],
    }
    strata = {
        "library_size_quartiles": [
            {
                **overall,
                "stratum_type": "library_size_quartiles",
                "label": f"Q{index}",
                "lower": None,
                "upper": None,
                "n": 0,
                "metrics": {name: {**value, "n": 0} for name, value in metrics.items()},
            }
            for index in range(1, 5)
        ],
        "truth_expression_bins": [
            {
                **overall,
                "stratum_type": "truth_expression_bins",
                "label": label,
                "lower": lower,
                "upper": upper,
                "n": 0,
                "metrics": {name: {**value, "n": 0} for name, value in metrics.items()},
            }
            for label, lower, upper in (
                ("[0,1)", 0.0, 1.0),
                ("[1,2)", 1.0, 2.0),
                ("[2,4)", 2.0, 4.0),
                ("[4,inf)", 4.0, None),
            )
        ],
    }
    body = {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "identity": identity,
        "truth_kind": "exact_pre_capture",
        "matrix": matrix,
        "policy": policy,
        "policy_sha256": None if policy is None else _sha256_payload(policy),
        "overall": overall,
        "strata": strata,
    }
    storage = (
        {
            "encoding": "zlib_raw_f64_v1",
            "compression_level": 6,
            "path": f"runs/{run['run_id']}.p_pre_zero.f64.zlib",
            "compressed_sha256": "1" * 64,
            "compressed_nbytes": 1,
            "uncompressed_sha256": "f" * 64,
            "uncompressed_nbytes": (
                run["retained_cell_count"] * run["retained_gene_count"] * 8
            ),
        }
        if completed_score
        else {
            "encoding": None,
            "compression_level": None,
            "path": None,
            "compressed_sha256": None,
            "compressed_nbytes": None,
            "uncompressed_sha256": None,
            "uncompressed_nbytes": None,
        }
    )
    return {**body, "evidence_sha256": _sha256_payload(body), "storage": storage}


def _sha256_payload(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _development_source(tmp_path: Path):
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset_path = tmp_path / "dataset.h5ad"
    dataset, cells, _genes = _dataset(dataset_path)
    dataset_sha = benchmark_dataset_sha256(dataset)
    dataset_authority = _dataset_binding(dataset_path, cells)
    source = tmp_path / "development"
    source.mkdir()
    output = np.asarray(_common_output(dataset), dtype="<f8", order="C")
    raw = output.tobytes(order="C")
    output_path = source / "runs" / "run-completed.log2-cp10k-f64"
    output_path.parent.mkdir()
    output_path.write_bytes(raw)
    completed = _run(
        run_id="run-completed",
        method_id="maskimpute",
        dataset_sha256=dataset_sha,
        cell_ids=cells,
        status="completed",
        reason=None,
    )
    _apply_dataset_authority(completed, dataset_authority)
    completed.update(
        {
            "evaluator_output_path": "runs/run-completed.log2-cp10k-f64",
            "evaluator_output_file_sha256": hashlib.sha256(raw).hexdigest(),
            "evaluator_output_shape": list(output.shape),
            "evaluator_output_dtype": "<f8",
            "evaluator_scale": "log2_cp10k_plus_1",
        }
    )
    completed["evaluator_output_sha256"] = _evaluator_output_sha256(completed, raw)
    failed = _run(
        run_id="run-failed",
        method_id="magic",
        dataset_sha256=dataset_sha,
        cell_ids=cells,
        status="failed",
        reason="adapter_nonzero_exit",
    )
    _apply_dataset_authority(failed, dataset_authority)
    records = [
        {
            "run": completed,
            "metrics": [],
            "p_pre_zero_evidence": _current_prezero_evidence(completed),
        },
        {
            "run": failed,
            "metrics": [],
            "p_pre_zero_evidence": _current_prezero_evidence(failed),
        },
    ]
    body = {
        "schema_version": 1,
        "plan_sha256": "4" * 64,
        "input_hashes": {"dataset_manifest_sha256": "5" * 64},
        "planned_run_count": 2,
        "status": "completed",
        "evaluation_scope": "reconstruction_only",
        "comparator_selection_status": "complete_terminal_denominator",
        "selection_complete": False,
        "selection_blockers": ["downstream_evidence_pending"],
        "records": records,
        "budget": {},
    }
    checkpoint = {**body, "checkpoint_sha256": canonical_sha256(body)}
    _write_canonical(source / "checkpoint.json", checkpoint)
    return source, dataset_path, cells, output_path


def _dataset_binding(dataset_path: Path, cells: tuple[str, ...]):
    from maskimpute_benchmark.downstream_evidence import bind_evaluator_dataset

    return bind_evaluator_dataset(dataset_path, retained_cell_ids=cells)


def _apply_dataset_authority(run: dict[str, object], binding: object) -> None:
    for name in (
        "mechanism",
        "biological_id",
        "technical_view",
        "method_input_sha256",
        "dataset_qc_policy_sha256",
        "excluded_cell_count",
        "excluded_cell_ids_sha256",
        "retained_cell_count",
        "retained_cell_ids_sha256",
        "retained_gene_count",
        "observed_zero_count",
    ):
        run[name] = getattr(binding, name)


def _development_source_plan(source: Path) -> SimpleNamespace:
    checkpoint = json.loads((source / "checkpoint.json").read_text(encoding="utf-8"))
    return SimpleNamespace(
        plan_sha256=checkpoint["plan_sha256"],
        input_hashes=checkpoint["input_hashes"],
        entries=tuple(
            {**stored["run"], "ordinal": ordinal}
            for ordinal, stored in enumerate(checkpoint["records"], start=1)
        ),
    )


def _trajectory_source(
    tmp_path: Path,
) -> tuple[Path, object, object, object, dict[str, object]]:
    from dataclasses import replace

    from maskimpute_benchmark.final_runner import (
        FinalPlanEntry,
        TrajectoryExecutionPlan,
        validate_trajectory_execution_for_evaluation,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import (
        DEVELOPMENT_MODEL_SEEDS,
        RunPlanEntry,
        _RECONSTRUCTION_METRIC_NAMES,
    )

    dataset_path = tmp_path / "results/trajectory/dataset/evaluator.h5ad"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _dataset(dataset_path)
    ordinary = _dataset_binding(
        dataset_path, tuple(f"cell-{index}" for index in range(1, 7))
    )
    binding = replace(
        ordinary,
        dataset_id="trajectory-exact-latent-01",
        mechanism="synthetic_trajectory",
        biological_id="trajectory-draw-01",
        technical_view="exact_latent",
        trajectory_root_cell_id="cell-1",
        trajectory_source_id="registered-synthetic-trajectory-v1",
        trajectory_authority_sha256="7" * 64,
        trajectory_binding_sha256="8" * 64,
    )
    _candidate, configuration = _test_configuration_authority()
    reason = "technical_unavailable_development_attempts"
    run_plan = RunPlanEntry(
        ordinal=1,
        run_id="trajectory-fixture-run",
        method_id=configuration.method_id,
        dataset_id=binding.dataset_id,
        source_dataset_sha256=binding.dataset_sha256,
        mechanism=binding.mechanism,
        biological_id=binding.biological_id,
        technical_view=binding.technical_view,
        model_seed=None,
        configuration_id=configuration.configuration_id,
        configuration_sha256=configuration.configuration_sha256,
        preflight_status="planned",
        preflight_reason=None,
        configuration_kind=configuration.kind,
        requires_count_score=configuration.requires_count_score,
        requires_calibration=configuration.requires_calibration,
    )
    entry = FinalPlanEntry(run=run_plan, action="not_applicable", reason=reason)
    input_hashes = {
        "frozen_method_sha256": "1" * 64,
        "method_registry_sha256": "2" * 64,
        "runtime_lock_sha256": "3" * 64,
        "primary_final_plan_sha256": "4" * 64,
        "trajectory_authority_file_sha256": "5" * 64,
        "trajectory_authority_sha256": str(binding.trajectory_authority_sha256),
        "trajectory_binding_sha256": str(binding.trajectory_binding_sha256),
        "trajectory_dataset_sha256": binding.dataset_sha256,
        "trajectory_dataset_file_sha256": binding.file_sha256,
        "trajectory_dataset_receipt_sha256": "6" * 64,
        "trajectory_dataset_receipt_file_sha256": "7" * 64,
        "trajectory_method_input_sha256": binding.method_input_sha256,
        "trajectory_retained_cell_ids_sha256": binding.retained_cell_ids_sha256,
        "dataset_qc_policy_sha256": binding.dataset_qc_policy_sha256,
        "execution_claim_sha256": "8" * 64,
        "execution_environment_sha256": "9" * 64,
        "execution_authority_sha256": "a" * 64,
    }
    plan_body = {
        "schema_version": 1,
        "scope": "supplementary_trajectory",
        "input_hashes": input_hashes,
        "entries": [entry.to_dict()],
        "configurations": [configuration.to_dict()],
        "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
    }
    plan = TrajectoryExecutionPlan(
        schema_version=1,
        scope="supplementary_trajectory",
        input_hashes=input_hashes,
        entries=(entry,),
        configurations=(configuration,),
        plan_sha256=canonical_sha256(plan_body),
    )
    run = _run(
        run_id=run_plan.run_id,
        method_id=run_plan.method_id,
        dataset_sha256=binding.dataset_sha256,
        cell_ids=binding.retained_cell_ids,
        status="unavailable",
        reason=reason,
    )
    _apply_dataset_authority(run, binding)
    run.update(
        {
            "dataset_id": binding.dataset_id,
            "mechanism": binding.mechanism,
            "biological_id": binding.biological_id,
            "technical_view": binding.technical_view,
            "model_seed": None,
            "configuration_id": configuration.configuration_id,
            "configuration_sha256": configuration.configuration_sha256,
            "configuration_kind": configuration.kind,
            "requires_count_score": configuration.requires_count_score,
            "requires_calibration": configuration.requires_calibration,
            "native_output_retention": "not_available",
            "evaluator_output_encoding": None,
            "evaluator_output_uncompressed_nbytes": None,
            "evaluator_output_uncompressed_sha256": None,
        }
    )
    metrics = [
        {
            "mechanism": binding.mechanism,
            "biological_id": binding.biological_id,
            "technical_view": binding.technical_view,
            "dataset_id": binding.dataset_id,
            "method": run_plan.method_id,
            "model_seed": None,
            "configuration_id": configuration.configuration_id,
            "configuration_sha256": configuration.configuration_sha256,
            "metric": metric,
            "value": None,
            "n": 0,
            "status": "unavailable",
            "reason": reason,
        }
        for metric in _RECONSTRUCTION_METRIC_NAMES
    ]
    record = {
        "run": run,
        "metrics": metrics,
        "p_pre_zero_evidence": _current_prezero_evidence(run),
        "execution_request": None,
    }
    source = tmp_path / "results/trajectory/execution"
    record_file_sha256 = _write_canonical(source / "records/00000001.json", record)
    manifest_body = {
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": plan.plan_sha256,
        "input_hashes": input_hashes,
        "planned_run_count": 1,
        "recorded_run_count": 1,
        "records": [
            {
                "ordinal": 1,
                "run_id": run_plan.run_id,
                "path": "records/00000001.json",
                "sha256": record_file_sha256,
            }
        ],
        "artifact_storage": {
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "evaluator_output_compression_level": 6,
            "native_output_retention": "omitted_redundant_final_output",
            "p_pre_zero_encoding": "zlib_raw_f64_v1",
            "p_pre_zero_compression_level": 6,
        },
        "scope": "supplementary_trajectory",
        "plan_entries": [entry.to_dict()],
        "configurations": [configuration.to_dict()],
        "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
    }
    manifest = {**manifest_body, "manifest_sha256": canonical_sha256(manifest_body)}
    manifest_file_sha256 = _write_canonical(
        source / "execution_manifest.json", manifest
    )
    validation = validate_trajectory_execution_for_evaluation(plan, (record,))
    receipt_binding = _fake_evaluated_round_binding(
        tmp_path,
        tmp_path,
        trajectory_plan_sha256=plan.plan_sha256,
        final_plan_sha256=input_hashes["primary_final_plan_sha256"],
        trajectory_execution_claim_sha256=input_hashes["execution_claim_sha256"],
        trajectory_execution_environment_sha256=input_hashes[
            "execution_environment_sha256"
        ],
        trajectory_dataset_id=binding.dataset_id,
        trajectory_dataset_sha256=binding.dataset_sha256,
        trajectory_dataset_file_sha256=binding.file_sha256,
        trajectory_dataset_receipt_file_sha256=input_hashes[
            "trajectory_dataset_receipt_file_sha256"
        ],
        trajectory_dataset_receipt_payload_sha256=input_hashes[
            "trajectory_dataset_receipt_sha256"
        ],
        trajectory_source_id=str(binding.trajectory_source_id),
        trajectory_root_cell_id=str(binding.trajectory_root_cell_id),
        trajectory_registered_authority_sha256=str(binding.trajectory_authority_sha256),
        trajectory_registered_binding_sha256=str(binding.trajectory_binding_sha256),
        trajectory_authority_sha256=input_hashes["execution_authority_sha256"],
        trajectory_execution_manifest_file_sha256=manifest_file_sha256,
        trajectory_execution_manifest_payload_sha256=manifest["manifest_sha256"],
        trajectory_execution_validation_sha256=validation["validation_sha256"],
        trajectory_record_payload_sha256s_sha256=canonical_sha256(
            validation["record_payload_sha256s"]
        ),
        trajectory_status_counts_sha256=canonical_sha256(
            {
                "executed_status_counts": validation["executed_status_counts"],
                "not_applicable_count": validation["not_applicable_count"],
            }
        ),
        trajectory_planned_run_count=1,
    )
    return (
        source,
        binding,
        configuration,
        plan,
        {
            "record": record,
            "manifest": manifest,
            "evaluated_round_binding": receipt_binding,
        },
    )


def _evaluation_manifest(
    path: Path,
    *,
    base_plan: object,
    revision_plan: object,
) -> tuple[str, str, str, str]:
    from maskimpute_benchmark.protocol import canonical_sha256

    def reconstruction(plan: object) -> dict[str, object]:
        return {
            "checkpoint_path": str(
                Path(plan.source_root).relative_to(path.parent) / "checkpoint.json"
            ),
            "checkpoint_file_sha256": plan.source_manifest_file_sha256,
            "checkpoint_sha256": plan.source_manifest_payload_sha256,
            "plan_sha256": plan.source_plan_sha256,
            "input_hashes": json.loads(
                (Path(plan.source_root) / "checkpoint.json").read_text(encoding="utf-8")
            )["input_hashes"],
            "raw_artifacts": [],
        }

    base = reconstruction(base_plan)
    revision = reconstruction(revision_plan)
    body = {
        "schema_version": 1,
        "reconstruction": base,
        "revisions": [{"version": "v28", "reconstruction": revision}],
    }
    payload = {**body, "manifest_sha256": canonical_sha256(body)}
    file_sha = _write_canonical(path, payload)
    return (
        file_sha,
        payload["manifest_sha256"],
        canonical_sha256(base),
        canonical_sha256(revision),
    )


def test_prepared_runner_panel_bridge_binds_persisted_dataset_paths(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        bind_prepared_evaluator_panel,
    )

    dataset_path = tmp_path / "dataset.h5ad"
    dataset_object, cells, _genes = _dataset(dataset_path)
    from maskimpute_benchmark.methods import prepare_method_input
    from maskimpute_benchmark.schema import make_inference_view

    authority = _dataset_binding(dataset_path, cells)
    method_input = prepare_method_input(make_inference_view(dataset_object))
    runner_binding = SimpleNamespace(
        dataset_id="dev-symsim-01", output_path="dataset.h5ad"
    )
    prepared = {
        "dev-symsim-01": SimpleNamespace(
            binding=SimpleNamespace(
                mechanism=authority.mechanism,
                biological_id=authority.biological_id,
                technical_view=authority.technical_view,
            ),
            audit=SimpleNamespace(
                retained_cell_ids=cells,
                excluded_cell_count=authority.excluded_cell_count,
                excluded_cell_ids_sha256=authority.excluded_cell_ids_sha256,
                retained_cell_count=authority.retained_cell_count,
                retained_cell_ids_sha256=authority.retained_cell_ids_sha256,
            ),
            method_input=method_input,
        )
    }

    bindings = bind_prepared_evaluator_panel(
        (runner_binding,), prepared, dataset_root=tmp_path
    )

    assert len(bindings) == 1
    assert bindings[0].dataset_id == "dev-symsim-01"
    assert bindings[0].retained_cell_ids == cells
    assert bindings[0].path == str(dataset_path.absolute())


def test_plan_rejects_rehashed_biological_identity_outside_bound_dataset_authority(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"].update(
        {
            "mechanism": "semisynthetic",
            "biological_id": "forged-draw",
            "technical_view": "forged-view",
        }
    )
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="dataset authority"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_current_runner_record_schema_is_accepted_without_field_aliases(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import build_downstream_evidence_plan
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    binding = _dataset_binding(dataset_path, cells)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for stored in checkpoint["records"]:
        run = stored["run"]
        run["retained_gene_count"] = binding.retained_gene_count
        run["observed_zero_count"] = binding.observed_zero_count
        stored["p_pre_zero_evidence"] = _current_prezero_evidence(run)
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(binding,),
        configurations=_test_configuration_authority(),
    )

    assert len(plan.entries) == 2


def test_source_adapter_rejects_retired_prezero_policy_field_alias(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    evidence = checkpoint["records"][0]["p_pre_zero_evidence"]
    policy = evidence["policy"]
    policy["calibration_artifact_sha256"] = policy.pop("calibration_file_sha256")
    policy.pop("calibration_payload_sha256")
    evidence["policy_sha256"] = canonical_sha256(policy)
    evidence_body = {
        key: value
        for key, value in evidence.items()
        if key not in {"evidence_sha256", "storage"}
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence_body)
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="score policy schema"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_selection_primary_plan_rejects_seed_drift_from_bound_source_plan(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    source_plan = _development_source_plan(source)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"]["model_seed"] = 999
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="source plan authority"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            evidence_scope="selection_primary",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
            source_plan=source_plan,
        )


def test_selection_primary_scope_excludes_only_nonselection_maskimpute_ablations(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    ablation = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="no-gate",
        kind="ablation",
        payload={
            "configuration_id": "no-gate",
            "method_id": "maskimpute",
            "variant": "downstream-supplementary-test",
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    ablation_run = dict(checkpoint["records"][1]["run"])
    ablation_run.update(
        {
            "run_id": "run-maskimpute-ablation",
            "method_id": "maskimpute",
            "configuration_id": ablation.configuration_id,
            "configuration_sha256": ablation.configuration_sha256,
            "configuration_kind": ablation.kind,
        }
    )
    checkpoint["records"].append(
        {
            "run": ablation_run,
            "metrics": [],
            "p_pre_zero_evidence": _current_prezero_evidence(ablation_run),
        }
    )
    checkpoint["planned_run_count"] = 3
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)
    configurations = (*_test_configuration_authority(), ablation)

    primary = build_downstream_evidence_plan(
        source,
        source_kind="development",
        evidence_scope="selection_primary",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=configurations,
        source_plan=_development_source_plan(source),
    )
    supplementary = build_downstream_evidence_plan(
        source,
        source_kind="development",
        evidence_scope="supplementary_nonselection",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=configurations,
        source_plan=_development_source_plan(source),
    )

    assert [entry.run_id for entry in primary.entries] == [
        "run-completed",
        "run-failed",
    ]
    assert [entry.run_id for entry in supplementary.entries] == [
        "run-maskimpute-ablation"
    ]
    destination = tmp_path / "selection-primary-downstream"
    manifest = run_downstream_evidence(primary, destination)
    loaded = load_downstream_evidence_manifest(destination)

    assert manifest["planned_denominator_count"] == 2
    assert loaded.planned_denominator_count == 2


def test_revision_downstream_bundle_covers_base_and_activated_checkpoint(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DevelopmentSourcePlan,
        build_downstream_evidence_plan,
        combine_development_downstream_evidence_plans,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.revisions import development_selection_stage_paths
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    base_root = tmp_path / "base"
    revision_root = tmp_path / "revision"
    base_root.mkdir()
    revision_root.mkdir()
    base_source, base_dataset, base_cells, _base_output = _development_source(base_root)
    revision_source, _revision_dataset, revision_cells, _revision_output = (
        _development_source(revision_root)
    )
    revision_configuration = AuthorizedConfiguration.create(
        method_id="maskimpute",
        configuration_id="v28-c01-nb-decoder",
        kind="candidate_search",
        payload={
            "configuration_id": "v28-c01-nb-decoder",
            "method_id": "maskimpute",
            "variant": "revision-downstream-test",
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    checkpoint_path = revision_source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    revised_run = checkpoint["records"][0]["run"]
    revised_run.update(
        {
            "run_id": "run-v28-completed",
            "configuration_id": revision_configuration.configuration_id,
            "configuration_sha256": revision_configuration.configuration_sha256,
            "configuration_kind": revision_configuration.kind,
        }
    )
    output_raw = (
        revision_source / str(revised_run["evaluator_output_path"])
    ).read_bytes()
    revised_run["evaluator_output_sha256"] = _evaluator_output_sha256(
        revised_run, output_raw
    )
    checkpoint["records"][0]["p_pre_zero_evidence"] = _current_prezero_evidence(
        revised_run
    )
    checkpoint_body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    _write_canonical(checkpoint_path, checkpoint)

    base_plan = build_downstream_evidence_plan(
        base_source,
        source_kind="development",
        evidence_scope="selection_primary",
        datasets=(_dataset_binding(base_dataset, base_cells),),
        configurations=_test_configuration_authority(),
        source_plan=_development_source_plan(base_source),
    )
    _base_candidate, magic = _test_configuration_authority()
    revision_plan = build_downstream_evidence_plan(
        revision_source,
        source_kind="development",
        evidence_scope="all",
        datasets=(_dataset_binding(base_dataset, revision_cells),),
        configurations=(revision_configuration, magic),
        source_plan=_development_source_plan(revision_source),
    )
    evaluation_path = tmp_path / "evaluation-v28.json"
    (
        evaluation_file_sha,
        evaluation_payload_sha,
        base_evaluation_sha,
        revision_evaluation_sha,
    ) = _evaluation_manifest(
        evaluation_path,
        base_plan=base_plan,
        revision_plan=revision_plan,
    )
    combined = combine_development_downstream_evidence_plans(
        tmp_path,
        (
            DevelopmentSourcePlan(
                source_id="base",
                plan=base_plan,
                selected_methods=("default", "magic"),
                evaluation_manifest_path="evaluation-v28.json",
                evaluation_manifest_file_sha256=evaluation_file_sha,
                evaluation_manifest_payload_sha256=evaluation_payload_sha,
                evaluation_source_pointer="/reconstruction",
                evaluation_source_sha256=base_evaluation_sha,
            ),
            DevelopmentSourcePlan(
                source_id="v28",
                plan=revision_plan,
                selected_methods=(revision_configuration.configuration_id,),
                evaluation_manifest_path="evaluation-v28.json",
                evaluation_manifest_file_sha256=evaluation_file_sha,
                evaluation_manifest_payload_sha256=evaluation_payload_sha,
                evaluation_source_pointer="/revisions/0/reconstruction",
                evaluation_source_sha256=revision_evaluation_sha,
            ),
        ),
        revision_versions=("v28",),
    )

    assert combined.source_root == str(tmp_path.absolute())
    assert combined.development_revision_versions == ("v28",)
    assert tuple(source.source_id for source in combined.development_sources) == (
        "base",
        "v28",
    )
    assert len(combined.entries) == 3
    assert [entry.configuration_id for entry in combined.entries] == [
        "default",
        "registry-default",
        "v28-c01-nb-decoder",
    ]
    assert (
        len({source.manifest_file_sha256 for source in combined.development_sources})
        == 2
    )
    assert all(
        source.evaluation_manifest_file_sha256 == evaluation_file_sha
        for source in combined.development_sources
    )

    stage_paths = development_selection_stage_paths("v28")
    destination = tmp_path / stage_paths.downstream_directory
    run_downstream_evidence(combined, destination)
    loaded = load_downstream_evidence_manifest(destination)
    assert loaded.planned_denominator_count == 3
    assert loaded.payload["development_revision_versions"] == ["v28"]
    assert len(loaded.payload["development_sources"]) == 2

    from maskimpute_benchmark.selection import (
        attach_downstream_evidence_to_selection_result,
    )

    selection_records = [
        {
            "mechanism": record["mechanism"],
            "biological_id": record["biological_id"],
            "technical_view": record["technical_view"],
            "dataset_id": record["dataset_id"],
            "dataset_sha256": record["dataset_sha256"],
            "method": record["method"],
            "method_sha256": record["method_artifact_sha256"],
            "model_seed": record["model_seed"],
            "metric": "mse",
            "value": 0.0 if record["run_status"] == "completed" else None,
            "status": record["run_status"],
        }
        for record in loaded.records
    ]
    selection_core = {
        "schema_version": 3,
        "revision_versions": ["v28"],
        "dataset_manifest_sha256": "1" * 64,
        "count_score_manifest_sha256": "2" * 64,
        "retained_calibration_artifact_sha256": "3" * 64,
        "evaluation_manifest_sha256": evaluation_file_sha,
        "comparator_selection": {
            "path": (
                "artifacts/study/development/evaluation/comparator_selection.json"
            ),
            "receipt": {},
            "selected_by_method": {},
            "nonexecution_identity_by_method": {},
            "ready_comparison_population_ids": [],
        },
        "records": selection_records,
        "orthogonal_intervals": [],
    }
    source_payload = {
        **selection_core,
        "result_sha256": canonical_sha256(selection_core),
    }
    source_file_sha = _write_canonical(
        tmp_path / stage_paths.source_selection_input,
        source_payload,
    )
    upgraded = attach_downstream_evidence_to_selection_result(
        source_payload,
        tmp_path,
        stage_paths.downstream_directory,
    )
    source_bindings = upgraded["downstream_evidence"]["sources"]
    assert [source["source_id"] for source in source_bindings] == ["base", "v28"]
    assert upgraded["downstream_evidence"]["revision_versions"] == ["v28"]
    assert upgraded["downstream_evidence"]["source_selection_input_path"] == (
        stage_paths.source_selection_input
    )
    assert upgraded["downstream_evidence"]["source_selection_input_file_sha256"] == (
        source_file_sha
    )
    assert (
        upgraded["downstream_evidence"]["source_selection_result_sha256"]
        == (source_payload["result_sha256"])
    )

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["revisions"][0]["reconstruction"]["checkpoint_sha256"] = "0" * 64
    evaluation_body = {
        key: value for key, value in evaluation.items() if key != "manifest_sha256"
    }
    evaluation["manifest_sha256"] = canonical_sha256(evaluation_body)
    _write_canonical(evaluation_path, evaluation)
    from maskimpute_benchmark.downstream_evidence import DownstreamEvidenceError

    with pytest.raises(
        DownstreamEvidenceError,
        match="development checkpoint differs",
    ):
        load_downstream_evidence_manifest(destination)


def test_scaling_binding_change_preserves_valid_base_v28_v29_bundle(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DevelopmentSourcePlan,
        build_downstream_evidence_plan,
        combine_development_downstream_evidence_plans,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    base_root = tmp_path / "base"
    base_root.mkdir()
    base_source, base_dataset, base_cells, _base_output = _development_source(base_root)
    base_plan = build_downstream_evidence_plan(
        base_source,
        source_kind="development",
        evidence_scope="selection_primary",
        datasets=(_dataset_binding(base_dataset, base_cells),),
        configurations=_test_configuration_authority(),
        source_plan=_development_source_plan(base_source),
    )
    _base_candidate, magic = _test_configuration_authority()

    revision_values: list[tuple[str, object, object]] = []
    for version in ("v28", "v29"):
        revision_root = tmp_path / version
        revision_root.mkdir()
        source, _dataset_path, cells, _output = _development_source(revision_root)
        configuration = AuthorizedConfiguration.create(
            method_id="maskimpute",
            configuration_id=f"{version}-candidate",
            kind="candidate_search",
            payload={
                "configuration_id": f"{version}-candidate",
                "method_id": "maskimpute",
                "variant": f"{version}-downstream-test",
            },
            requires_count_score=False,
            requires_calibration=False,
        )
        checkpoint_path = source / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        run = checkpoint["records"][0]["run"]
        run.update(
            {
                "run_id": f"run-{version}-completed",
                "configuration_id": configuration.configuration_id,
                "configuration_sha256": configuration.configuration_sha256,
                "configuration_kind": configuration.kind,
            }
        )
        output_raw = (source / str(run["evaluator_output_path"])).read_bytes()
        run["evaluator_output_sha256"] = _evaluator_output_sha256(run, output_raw)
        checkpoint["records"][0]["p_pre_zero_evidence"] = _current_prezero_evidence(run)
        checkpoint_body = {
            key: value
            for key, value in checkpoint.items()
            if key != "checkpoint_sha256"
        }
        checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
        _write_canonical(checkpoint_path, checkpoint)
        plan = build_downstream_evidence_plan(
            source,
            source_kind="development",
            evidence_scope="all",
            datasets=(_dataset_binding(base_dataset, cells),),
            configurations=(configuration, magic),
            source_plan=_development_source_plan(source),
        )
        revision_values.append((version, configuration, plan))

    evaluation_path = tmp_path / "evaluation-v29.json"

    def reconstruction(plan: object) -> dict[str, object]:
        source_root = Path(plan.source_root)
        checkpoint = json.loads(
            (source_root / "checkpoint.json").read_text(encoding="utf-8")
        )
        return {
            "checkpoint_path": str(
                source_root.relative_to(tmp_path) / "checkpoint.json"
            ),
            "checkpoint_file_sha256": plan.source_manifest_file_sha256,
            "checkpoint_sha256": plan.source_manifest_payload_sha256,
            "plan_sha256": plan.source_plan_sha256,
            "input_hashes": checkpoint["input_hashes"],
            "raw_artifacts": [],
        }

    base_evaluation = reconstruction(base_plan)
    revision_evaluations = [
        reconstruction(plan) for _version, _configuration, plan in revision_values
    ]
    evaluation_body = {
        "schema_version": 1,
        "reconstruction": base_evaluation,
        "revisions": [
            {"version": version, "reconstruction": evaluation}
            for (version, _configuration, _plan), evaluation in zip(
                revision_values,
                revision_evaluations,
                strict=True,
            )
        ],
    }
    evaluation_payload = {
        **evaluation_body,
        "manifest_sha256": canonical_sha256(evaluation_body),
    }
    evaluation_file_sha256 = _write_canonical(evaluation_path, evaluation_payload)
    sources = [
        DevelopmentSourcePlan(
            source_id="base",
            plan=base_plan,
            selected_methods=("default", "magic"),
            evaluation_manifest_path=evaluation_path.name,
            evaluation_manifest_file_sha256=evaluation_file_sha256,
            evaluation_manifest_payload_sha256=evaluation_payload["manifest_sha256"],
            evaluation_source_pointer="/reconstruction",
            evaluation_source_sha256=canonical_sha256(base_evaluation),
        )
    ]
    sources.extend(
        DevelopmentSourcePlan(
            source_id=version,
            plan=plan,
            selected_methods=(configuration.configuration_id,),
            evaluation_manifest_path=evaluation_path.name,
            evaluation_manifest_file_sha256=evaluation_file_sha256,
            evaluation_manifest_payload_sha256=evaluation_payload["manifest_sha256"],
            evaluation_source_pointer=f"/revisions/{index}/reconstruction",
            evaluation_source_sha256=canonical_sha256(revision_evaluations[index]),
        )
        for index, (version, configuration, plan) in enumerate(revision_values)
    )
    combined = combine_development_downstream_evidence_plans(
        tmp_path,
        sources,
        revision_versions=("v28", "v29"),
    )

    assert combined.evaluated_round_binding is None
    assert combined.development_revision_versions == ("v28", "v29")
    assert [source.source_id for source in combined.development_sources] == [
        "base",
        "v28",
        "v29",
    ]
    assert [entry.configuration_id for entry in combined.entries] == [
        "default",
        "registry-default",
        "v28-candidate",
        "v29-candidate",
    ]
    destination = tmp_path / "downstream-v29"
    run_downstream_evidence(combined, destination)
    loaded = load_downstream_evidence_manifest(destination)
    assert loaded.planned_denominator_count == 4
    assert loaded.payload["development_revision_versions"] == ["v28", "v29"]


def test_development_downstream_routes_to_latest_fixed_revision_without_fallback(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        development_downstream_revision_version,
    )
    from maskimpute_benchmark.revisions import revision_stage_paths

    assert development_downstream_revision_version(tmp_path) is None
    v28_path = tmp_path / revision_stage_paths("v28").selection_input
    _write_canonical(
        v28_path,
        {"schema_version": 3, "revision_versions": ["v28"]},
    )
    assert development_downstream_revision_version(tmp_path) == "v28"

    v29_path = tmp_path / revision_stage_paths("v29").selection_input
    _write_canonical(
        v29_path,
        {"schema_version": 3, "revision_versions": ["v28", "v29"]},
    )
    assert development_downstream_revision_version(tmp_path) == "v29"

    _write_canonical(
        v29_path,
        {"schema_version": 3, "revision_versions": ["v29"]},
    )
    with pytest.raises(
        DownstreamEvidenceError,
        match="v29 revision selection input identity differs",
    ):
        development_downstream_revision_version(tmp_path)

    _write_canonical(
        v29_path,
        {"schema_version": 4, "revision_versions": ["v28", "v29"]},
    )
    with pytest.raises(
        DownstreamEvidenceError,
        match="v29 revision selection input identity differs",
    ):
        development_downstream_revision_version(tmp_path)


def test_production_selection_primary_keys_exactly_match_reconstruction_bridge() -> (
    None
):
    from maskimpute_benchmark.development_evaluation import (
        reconstruction_selection_method,
    )
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.runner import (
        DEVELOPMENT_MODEL_SEEDS,
        AuthorizedConfiguration,
        load_runner_authority,
    )

    authority = load_runner_authority()
    registry = load_method_registry(Path("study/methods.json"))
    configured_method_ids = {value.method_id for value in authority.configurations}
    configurations = tuple(authority.configurations) + tuple(
        AuthorizedConfiguration.registry_default(spec)
        for spec in registry.methods
        if spec.execution_scope == "same_input_required"
        and spec.id not in configured_method_ids
    )
    declared = {
        value.configuration_id
        for value in configurations
        if value.kind == "candidate_search"
    } | {
        value.method_id
        for value in configurations
        if value.kind == "registry" or value.method_id == "capacity-matched-ae"
    }
    specification = {value.id: value for value in registry.methods}
    all_keys: set[tuple[object, ...]] = set()
    selection_keys: set[tuple[object, ...]] = set()
    primary_keys: set[tuple[object, ...]] = set()
    for dataset_index in range(16):
        for configuration in configurations:
            spec = specification[configuration.method_id]
            seeds = DEVELOPMENT_MODEL_SEEDS if spec.stochastic else (None,)
            run = {
                "configuration_kind": configuration.kind,
                "configuration_id": configuration.configuration_id,
                "method_id": configuration.method_id,
            }
            method = reconstruction_selection_method(run, declared)
            for seed in seeds:
                key = (
                    dataset_index,
                    configuration.method_id,
                    configuration.configuration_id,
                    configuration.configuration_sha256,
                    seed,
                )
                all_keys.add(key)
                if method is not None:
                    selection_keys.add(key)
                    primary_keys.add(key)

    assert primary_keys == selection_keys
    assert len(all_keys - primary_keys) == 5 * 16 * 3 == 240


def test_development_stage_resumes_and_preserves_exact_eight_row_denominators(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        load_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"

    first = run_downstream_evidence(plan, destination, max_denominators=1)
    assert first["status"] == "running"
    assert first["recorded_denominator_count"] == 1
    reloaded_plan = load_downstream_evidence_plan(destination)
    assert reloaded_plan.to_dict() == plan.to_dict()
    complete = run_downstream_evidence(reloaded_plan, destination)
    assert complete["status"] == "completed"
    assert complete["planned_denominator_count"] == 2
    assert complete["endpoint_row_count"] == 16
    assert complete["evaluator_source_sha256"] == plan.evaluator_source_sha256

    manifest = load_downstream_evidence_manifest(destination)
    records = manifest.records
    assert len(records) == 2
    assert all(len(record["endpoints"]) == 8 for record in records)
    assert records[0]["biological_id"] == "draw-01"
    assert records[0]["technical_view"] == "moderate"
    assert records[0]["model_seed"] == 42
    assert records[0]["runner_method_id"] == "maskimpute"
    assert records[0]["method"] == "default"
    assert records[0]["method_artifact_sha256"] == records[0]["configuration_sha256"]
    assert records[1]["method_artifact_sha256"] != records[1]["configuration_sha256"]
    assert {row["upstream_status"] for row in records[0]["endpoints"]} == {"completed"}
    assert {row["status"] for row in records[1]["endpoints"]} == {"failed"}
    assert {row["reason_code"] for row in records[1]["endpoints"]} == {
        "upstream_run_not_completed"
    }
    assert {row["upstream_reason"] for row in records[1]["endpoints"]} == {
        "adapter_nonzero_exit"
    }


def test_resume_revalidates_source_artifacts_and_immutable_record_prefix(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination, max_denominators=1)
    original = output_path.read_bytes()
    output_path.write_bytes(original + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="evaluator output.*checksum"):
        run_downstream_evidence(plan, destination)
    output_path.write_bytes(original)

    record_path = destination / "records" / "00000001.json"
    record_path.write_bytes(record_path.read_bytes() + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="record.*canonical"):
        run_downstream_evidence(plan, destination)


def test_resume_rejects_rehashed_finite_endpoint_value_drift(tmp_path: Path) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)

    record_path = destination / "records/00000001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    endpoint = next(row for row in record["endpoints"] if row["status"] == "completed")
    endpoint["value"] = 0.375 if endpoint["value"] != 0.375 else 0.625
    record_body = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    record["record_sha256"] = canonical_sha256(record_body)
    record_file_sha = _write_canonical(record_path, record)

    manifest_path = destination / "downstream_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["sha256"] = record_file_sha
    manifest["records"][0]["record_sha256"] = record["record_sha256"]
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest_body)
    _write_canonical(manifest_path, manifest)

    with pytest.raises(DownstreamEvidenceError, match="endpoint re-evaluation differs"):
        run_downstream_evidence(plan, destination)


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("direction", "endpoint contract differs"),
        ("independent_count", "independent unit differs"),
        ("boolean_independent_count", "independent unit differs"),
        ("descriptive_unit", "endpoint contract differs"),
        ("reason_vocabulary", "endpoint contract differs"),
        ("procedure", "endpoint procedure differs"),
        ("family", "endpoint family is unexpected"),
        ("range", "endpoint value is out of range"),
    ],
)
def test_resume_reconstructs_and_validates_endpoint_contract(
    tmp_path: Path, attack: str, message: str
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    record_path = destination / "records/00000001.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    row = record["endpoints"][0]
    if attack == "direction":
        row["direction"] = "higher_is_better"
    elif attack == "independent_count":
        row["independent_n"] = 2
    elif attack == "boolean_independent_count":
        row["independent_n"] = True
    elif attack == "descriptive_unit":
        row["descriptive_unit"] = "cells"
    elif attack == "reason_vocabulary":
        row["status"] = "unavailable"
        row["value"] = None
        row["reason_code"] = "forged_reason"
    elif attack == "procedure":
        row["procedure"] = "forged_procedure"
    elif attack == "family":
        row["family_id"] = "forged_family"
        row["family_size"] = 1
        row["alpha"] = 0.05
    elif attack == "range":
        row["value"] = 1.5
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(attack)
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = canonical_sha256(body)
    _write_canonical(record_path, record)

    with pytest.raises(DownstreamEvidenceError, match=message):
        run_downstream_evidence(plan, destination)


def test_plan_binds_current_evaluator_source_digest(tmp_path: Path) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    assert len(plan.evaluator_source_sha256) == 64
    provisional = replace(
        plan,
        evaluator_source_sha256="0" * 64,
        plan_sha256="0" * 64,
    )
    forged = replace(provisional, plan_sha256=canonical_sha256(provisional.body()))

    with pytest.raises(DownstreamEvidenceError, match="plan sources changed"):
        run_downstream_evidence(forged, tmp_path / "downstream")


def test_final_zlib_source_contract_is_consumed_with_bounded_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        load_downstream_evidence_plan,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.schema import benchmark_dataset_sha256

    dataset_path = tmp_path / "dataset.h5ad"
    dataset, cells, _genes = _dataset(dataset_path)
    dataset_sha = benchmark_dataset_sha256(dataset)
    evaluated_fixture = _evaluated_scaling_round(tmp_path, monkeypatch)
    repository = evaluated_fixture["repository"]
    round_root = evaluated_fixture["round_directory"]
    assert isinstance(repository, Path)
    assert isinstance(round_root, Path)
    source = round_root / "results/final/execution"
    output = np.asarray(_common_output(dataset), dtype="<f8", order="C")
    raw = output.tobytes(order="C")
    compressed = zlib.compress(raw, level=6)
    artifact = source / "runs" / "final-run.log2-cp10k-f64.zlib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(compressed)
    run = _run(
        run_id="final-run",
        method_id="maskimpute",
        dataset_sha256=dataset_sha,
        cell_ids=cells,
        status="completed",
        reason=None,
    )
    _apply_dataset_authority(run, _dataset_binding(dataset_path, cells))
    run.update(
        {
            "evaluator_output_path": "runs/final-run.log2-cp10k-f64.zlib",
            "evaluator_output_file_sha256": hashlib.sha256(compressed).hexdigest(),
            "evaluator_output_shape": list(output.shape),
            "evaluator_output_dtype": "<f8",
            "evaluator_scale": "log2_cp10k_plus_1",
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "evaluator_output_uncompressed_nbytes": len(raw),
            "evaluator_output_uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "native_output_retention": "not_available",
        }
    )
    run["evaluator_output_sha256"] = _evaluator_output_sha256(run, raw)
    record = {
        "run": run,
        "metrics": [],
        "p_pre_zero_evidence": _current_prezero_evidence(run),
        "execution_request": {
            "calibration_usage": "not_required",
            "configuration_sha256": run["configuration_sha256"],
            "count_score_manifest_sha256": "a" * 64,
            "dataset_id": run["dataset_id"],
            "execution_authority_sha256": "b" * 64,
            "method_input_sha256": run["method_input_sha256"],
            "model_seed": run["model_seed"],
            "request_sha256": "c" * 64,
            "retained_calibration_sha256": "d" * 64,
        },
    }
    record_path = source / "records" / "00000001.json"
    record_sha = _write_canonical(record_path, record)
    manifest_body = {
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": "6" * 64,
        "input_hashes": {"dataset_manifest_sha256": "7" * 64},
        "planned_run_count": 1,
        "recorded_run_count": 1,
        "records": [
            {
                "ordinal": 1,
                "run_id": "final-run",
                "path": "records/00000001.json",
                "sha256": record_sha,
            }
        ],
        "artifact_storage": {
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "evaluator_output_compression_level": 6,
            "native_output_retention": "omitted_redundant_final_output",
            "p_pre_zero_encoding": "zlib_raw_f64_v1",
            "p_pre_zero_compression_level": 6,
        },
    }
    manifest = {
        **manifest_body,
        "manifest_sha256": canonical_sha256(manifest_body),
    }
    manifest_file_sha256 = _write_canonical(
        source / "execution_manifest.json", manifest
    )
    receipt = evaluated_fixture["receipt"]
    result_manifest = evaluated_fixture["result_manifest"]
    receipt_path = evaluated_fixture["receipt_path"]
    assert isinstance(receipt, dict)
    assert isinstance(result_manifest, dict)
    assert isinstance(receipt_path, Path)
    validation_body = {
        "schema_version": 1,
        "status": "eligible_for_final_evaluation_complete_terminal_denominator",
        "final_plan_sha256": manifest["plan_sha256"],
        "planned_run_count": 1,
        "executed_completed_count": 1,
        "executed_algorithmic_failure_count": 0,
        "executed_status_counts": {"completed": 1},
        "not_applicable_count": 0,
        "record_payload_sha256s": [canonical_sha256(record)],
    }
    result_manifest.update(
        {
            "final_plan_sha256": manifest["plan_sha256"],
            "final_execution_manifest_sha256": manifest_file_sha256,
            "final_execution_payload_sha256": manifest["manifest_sha256"],
            "execution_validation": {
                **validation_body,
                "validation_sha256": canonical_sha256(validation_body),
            },
        }
    )
    trajectory_evidence = result_manifest["trajectory_evidence"]
    assert isinstance(trajectory_evidence, dict)
    _rebind_trajectory_primary_plan(trajectory_evidence, str(manifest["plan_sha256"]))
    result_files = result_manifest["result_files"]
    assert isinstance(result_files, list)
    for row in result_files:
        assert isinstance(row, dict)
        if row["path"] == "results/final/execution/execution_manifest.json":
            row["sha256"] = manifest_file_sha256
    result_files.extend(
        [
            {
                "path": "results/final/execution/records/00000001.json",
                "sha256": record_sha,
            },
            {
                "path": ("results/final/execution/runs/final-run.log2-cp10k-f64.zlib"),
                "sha256": hashlib.sha256(compressed).hexdigest(),
            },
        ]
    )
    result_files.sort(key=lambda row: row["path"])
    _resign_evaluation_receipt(receipt_path, receipt)

    with pytest.raises(
        DownstreamEvidenceError, match="evaluated-round binding is required"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="final",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )
    binding = downstream._read_verified_evaluated_round_binding(
        repository,
        round_root,
    )
    source_plan = SimpleNamespace(
        plan_sha256=manifest["plan_sha256"],
        input_hashes=manifest["input_hashes"],
        entries=({**run, "ordinal": 1},),
    )
    plan = build_downstream_evidence_plan(
        source,
        source_kind="final",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
        evaluated_round_binding=binding,
        source_plan=source_plan,
    )
    monkeypatch.setattr(
        downstream,
        "build_final_downstream_evidence_plan",
        lambda _repository, _round_root: plan,
    )
    destination = downstream.expected_final_downstream_output_directory(plan)
    partial = run_downstream_evidence(plan, destination, max_denominators=0)
    assert partial["status"] == "running"
    result = run_downstream_evidence(plan, destination)

    assert result["status"] == "completed"
    loaded = load_downstream_evidence_manifest(destination)
    assert len(loaded.records) == 1
    assert len(loaded.records[0]["endpoints"]) == 8
    assert loaded.records[0]["source_kind"] == "final"
    assert loaded.payload["evaluated_round_binding_sha256"] == binding.binding_sha256
    assert load_downstream_evidence_plan(destination).evaluated_round_binding == binding

    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_evaluator_source_sha256", lambda: "f" * 64)
    manifest["artifact_storage"]["evaluator_output_compression_level"] = 9
    changed_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = canonical_sha256(changed_body)
    _write_canonical(source / "execution_manifest.json", manifest)
    with pytest.raises(
        DownstreamEvidenceError, match="final artifact storage policy differs"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="final",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
            evaluated_round_binding=binding,
            source_plan=source_plan,
        )


def test_supplementary_trajectory_plan_requires_exact_typed_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_evaluator_source_sha256", lambda: "f" * 64)
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)

    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )

    assert plan.evidence_scope == "supplementary_trajectory"
    assert plan.source_root == str(source.absolute())
    assert plan.source_plan_sha256 == source_plan.plan_sha256
    assert plan.source_plan_authority == "independent"
    assert len(plan.datasets) == 1
    assert len(plan.entries) == 1
    assert plan.entries[0].status == "unavailable"


def test_public_generic_builder_rejects_final_trajectory_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="fixed production builder",
    ):
        downstream.build_downstream_evidence_plan(
            source,
            source_kind="final",
            evidence_scope="supplementary_trajectory",
            datasets=(dataset,),
            configurations=(configuration,),
            evaluated_round_binding=fixture["evaluated_round_binding"],
            source_plan=source_plan,
        )


def test_private_trajectory_builder_rejects_byte_identical_external_dataset_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    external_path = tmp_path / "external-copy/evaluator.h5ad"
    external_path.parent.mkdir(parents=True)
    shutil.copyfile(dataset.path, external_path)
    copied = replace(dataset, path=str(external_path.absolute()))
    assert hashlib.sha256(external_path.read_bytes()).hexdigest() == dataset.file_sha256
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="trajectory dataset authority differs",
    ):
        downstream._build_downstream_evidence_plan(
            source,
            source_kind="final",
            evidence_scope="supplementary_trajectory",
            datasets=(copied,),
            configurations=(configuration,),
            evaluated_round_binding=fixture["evaluated_round_binding"],
            source_plan=source_plan,
        )


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("extra_manifest_field", "trajectory execution manifest schema differs"),
        ("changed_plan_entry", "trajectory execution manifest plan differs"),
        ("primary_manifest_schema", "trajectory execution manifest schema differs"),
        ("dataset_receipt_drift", "trajectory dataset differs from evaluated receipt"),
        (
            "dataset_receipt_payload_drift",
            "trajectory source plan authority differs",
        ),
        (
            "dataset_receipt_file_drift",
            "trajectory source plan authority differs",
        ),
    ],
)
def test_supplementary_trajectory_plan_rejects_manifest_and_dataset_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
    message: str,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    binding = fixture["evaluated_round_binding"]
    manifest_path = source / "execution_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if attack == "extra_manifest_field":
        manifest["extension"] = "forbidden"
    elif attack == "changed_plan_entry":
        manifest["plan_entries"][0]["reason"] = "coherent_local_replacement"
    elif attack == "primary_manifest_schema":
        for name in ("scope", "plan_entries", "configurations", "model_seed_policy"):
            manifest.pop(name)
    elif attack == "dataset_receipt_drift":
        binding = replace(binding, trajectory_dataset_sha256="0" * 64)
    elif attack == "dataset_receipt_payload_drift":
        binding = replace(
            binding,
            trajectory_dataset_receipt_payload_sha256="0" * 64,
        )
    elif attack == "dataset_receipt_file_drift":
        binding = replace(
            binding,
            trajectory_dataset_receipt_file_sha256="0" * 64,
        )
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(attack)
    if attack != "dataset_receipt_drift":
        body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        manifest["manifest_sha256"] = canonical_sha256(body)
        _write_canonical(manifest_path, manifest)

    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    with pytest.raises(downstream.DownstreamEvidenceError, match=message):
        downstream._build_downstream_evidence_plan(
            source,
            source_kind="final",
            evidence_scope="supplementary_trajectory",
            datasets=(dataset,),
            configurations=(configuration,),
            evaluated_round_binding=binding,
            source_plan=source_plan,
        )


def test_trajectory_plan_revalidation_uses_fixed_production_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    calls: list[tuple[Path, Path]] = []

    def fixed(repository: str, round_root: str):
        calls.append((Path(repository), Path(round_root)))
        return plan

    monkeypatch.setattr(
        downstream,
        "build_final_trajectory_downstream_evidence_plan",
        fixed,
        raising=False,
    )
    monkeypatch.setattr(
        downstream,
        "build_downstream_evidence_plan",
        lambda *_args, **_kwargs: pytest.fail(
            "persisted trajectory replay trusted a local authority label"
        ),
    )

    downstream._revalidate_plan(plan)

    binding = plan.evaluated_round_binding
    assert binding is not None
    assert calls == [(Path(binding.repository_root), Path(binding.round_root))]


def test_fixed_trajectory_builder_reloads_registered_dataset_and_receipt_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    import inspect

    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.final_runner as final_runner
    import maskimpute_benchmark.methods as methods
    import maskimpute_benchmark.publication_freeze as publication_freeze

    repository = tmp_path / "repository"
    round_root = repository / "artifacts/study/final/rounds/round-1"
    round_root.mkdir(parents=True)
    source, dataset, _configuration, source_plan, fixture = _trajectory_source(
        round_root
    )
    original_binding = fixture["evaluated_round_binding"]
    binding = replace(
        original_binding,
        repository_root=str(repository.absolute()),
        round_root=str(round_root.absolute()),
        round_id="round-1",
    )
    registered = SimpleNamespace(binding=SimpleNamespace(dataset_id=dataset.dataset_id))
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        downstream,
        "__file__",
        str(repository / "maskimpute_benchmark/downstream_evidence.py"),
    )
    monkeypatch.setattr(downstream, "_evaluator_source_sha256", lambda: "f" * 64)
    monkeypatch.setattr(
        downstream,
        "_read_verified_evaluated_round_binding",
        lambda selected_repository, selected_round: (
            binding
            if selected_repository == repository.absolute()
            and selected_round == round_root.absolute()
            else pytest.fail("fixed builder validated a different round")
        ),
    )
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    monkeypatch.setattr(
        downstream,
        "_bind_registered_trajectory_dataset",
        lambda selected_round, loaded: (
            dataset
            if selected_round == round_root.absolute() and loaded is registered
            else pytest.fail("fixed builder used a different registered dataset")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        publication_freeze,
        "validate_frozen_method",
        lambda selected_repository: {"fixture": str(selected_repository)},
    )
    registry = SimpleNamespace(methods=(SimpleNamespace(id="magic"),))
    monkeypatch.setattr(methods, "load_method_registry", lambda _path: registry)
    monkeypatch.setattr(
        final_runner,
        "load_prepared_trajectory_dataset",
        lambda selected_repository, selected_round: (
            registered
            if selected_repository == repository.absolute()
            and selected_round == round_root.absolute()
            else pytest.fail("read-only loader received a different authority")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        final_runner,
        "materialize_prepared_trajectory_dataset",
        lambda *_args, **_kwargs: pytest.fail(
            "downstream trajectory builder called a materializer"
        ),
    )

    def rebuild(
        frozen_method: object,
        selected_registry: object,
        loaded: object,
        **kwargs: object,
    ):
        calls["frozen_method"] = frozen_method
        calls["registry"] = selected_registry
        calls["registered"] = loaded
        calls["inputs"] = kwargs
        return source_plan

    monkeypatch.setattr(final_runner, "build_trajectory_execution_plan", rebuild)

    plan = downstream.build_final_trajectory_downstream_evidence_plan(
        repository, round_root
    )

    assert plan.source_root == str(source.absolute())
    assert plan.evidence_scope == "supplementary_trajectory"
    assert plan.datasets == (dataset,)
    assert plan.configurations == source_plan.configurations
    assert calls["registered"] is registered
    assert calls["inputs"] == {
        "execution_claim_sha256": binding.trajectory_execution_claim_sha256,
        "execution_environment_sha256": (
            binding.trajectory_execution_environment_sha256
        ),
        "execution_authority_sha256": binding.trajectory_authority_sha256,
        "primary_final_plan_sha256": binding.final_plan_sha256,
    }
    assert tuple(
        inspect.signature(
            downstream.build_final_trajectory_downstream_evidence_plan
        ).parameters
    ) == ("repository", "round_directory")


def test_persisted_trajectory_plan_uses_fixed_builder_not_independent_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    output = downstream.expected_final_downstream_output_directory(plan)
    _write_canonical(output / "plan.json", plan.to_dict())
    calls: list[tuple[str, str]] = []

    def fixed(repository: str, round_root: str):
        calls.append((repository, round_root))
        return plan

    monkeypatch.setattr(
        downstream,
        "build_final_trajectory_downstream_evidence_plan",
        fixed,
    )

    rebuilt, payload, _file_sha256 = downstream._load_persisted_plan(output)

    binding = plan.evaluated_round_binding
    assert binding is not None
    assert rebuilt == plan
    assert payload == plan.to_dict()
    assert calls == [(binding.repository_root, binding.round_root)]


@pytest.mark.parametrize(
    ("evidence_scope", "builder_name"),
    (
        ("all", "build_final_downstream_evidence_plan"),
        (
            "supplementary_trajectory",
            "build_final_trajectory_downstream_evidence_plan",
        ),
    ),
)
def test_persisted_direct_final_and_trajectory_plans_reload_typed_configurations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_scope: str,
    builder_name: str,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    baseline = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    _frozen, _registry, configurations = _frozen_downstream_authorities()
    direct = {value.method_id: value for value in configurations}["magic"]
    provisional = replace(
        baseline,
        evidence_scope=evidence_scope,
        configurations=(direct,),
        plan_sha256="0" * 64,
    )
    plan = replace(provisional, plan_sha256=canonical_sha256(provisional.body()))
    output = downstream.expected_final_downstream_output_directory(plan)
    _write_canonical(output / "plan.json", plan.to_dict())
    monkeypatch.setattr(
        downstream,
        builder_name,
        lambda _repository, _round_root: plan,
    )
    decoded: list[object] = []
    decode = downstream._configuration_from_payload

    def capture(value: object):
        result = decode(value)
        decoded.append(result)
        return result

    monkeypatch.setattr(downstream, "_configuration_from_payload", capture)

    rebuilt, payload, _file_sha256 = downstream._load_persisted_plan(output)

    assert rebuilt == plan
    assert payload == plan.to_dict()
    assert len(decoded) == 1
    assert isinstance(decoded[0], downstream.FrozenPlanMethodAuthority)


def test_trajectory_scope_emits_one_reason_coded_endpoint_and_exact_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    monkeypatch.setattr(
        downstream,
        "build_final_trajectory_downstream_evidence_plan",
        lambda _repository, _round_root: plan,
    )
    output = downstream.expected_final_downstream_output_directory(plan)

    running = downstream.run_downstream_evidence(plan, output, max_denominators=0)
    completed = downstream.run_downstream_evidence(plan, output)
    loaded = downstream.load_downstream_evidence_manifest(output)

    assert running["endpoint_row_count"] == 0
    assert completed["planned_denominator_count"] == 1
    assert completed["endpoint_row_count"] == 1
    assert loaded.endpoint_row_count == 1
    assert len(loaded.records) == 1
    endpoints = loaded.records[0]["endpoints"]
    assert len(endpoints) == 1
    assert endpoints[0]["endpoint"] == "trajectory_pseudotime_rank_loss"
    assert endpoints[0]["status"] == "unavailable"
    assert endpoints[0]["reason_code"] == "upstream_run_not_completed"
    assert (
        endpoints[0]["upstream_reason"] == "technical_unavailable_development_attempts"
    )


def test_completed_trajectory_scope_evaluates_only_the_direct_trajectory_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.downstream_evaluation import EndpointRecord
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    entry = replace(
        plan.entries[0],
        status="completed",
        reason=None,
        evaluator_output_path="runs/fixture.zlib",
        evaluator_output_file_sha256="1" * 64,
        evaluator_output_shape=(6, 4),
        evaluator_output_encoding="zlib_raw_f64_v1",
        evaluator_output_uncompressed_nbytes=192,
        evaluator_output_uncompressed_sha256="2" * 64,
        evaluator_output_sha256="3" * 64,
    )
    changed = replace(plan, entries=(entry,), plan_sha256="0" * 64)
    changed = replace(changed, plan_sha256=canonical_sha256(changed.body()))
    calls: list[tuple[object, object]] = []
    endpoint = EndpointRecord(
        endpoint="trajectory_pseudotime_rank_loss",
        value=0.25,
        status="completed",
        reason=None,
        direction="lower_is_better",
        independent_unit="biological_draw",
        independent_n=1,
        descriptive_n=6,
        descriptive_unit="trajectory_cells",
        procedure=(
            "root_oriented_multiscale_diffusion_log2_cp10k_plus_1_full_svd_"
            "blockwise_exact_knn=floor_sqrt_n_capped_15_sparse_eigsh_modes=15"
        ),
        family_id=None,
        family_size=None,
        alpha=None,
    )
    output = SimpleNamespace()
    targets = SimpleNamespace()
    monkeypatch.setattr(downstream, "_decode_output", lambda *_args: output)
    monkeypatch.setattr(
        downstream,
        "evaluate_downstream_endpoints",
        lambda *_args: pytest.fail("trajectory scope evaluated primary endpoints"),
    )
    monkeypatch.setattr(
        downstream,
        "evaluate_trajectory_endpoint",
        lambda observed_output, observed_targets: (
            calls.append((observed_output, observed_targets)) or endpoint
        ),
        raising=False,
    )

    record = downstream._evaluate_entry(changed, entry, dataset, targets)

    assert calls == [(output, targets)]
    assert [row["endpoint"] for row in record["endpoints"]] == [
        "trajectory_pseudotime_rank_loss"
    ]
    assert record["endpoints"][0]["value"] == 0.25


@pytest.mark.parametrize("value", (10**1000, -(10**1000)))
def test_persisted_downstream_endpoint_translates_unrepresentable_python_integers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: int,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.downstream_evaluation import EndpointRecord

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    entry = replace(plan.entries[0], status="completed", reason=None)
    endpoint = EndpointRecord(
        endpoint="trajectory_pseudotime_rank_loss",
        value=0.25,
        status="completed",
        reason=None,
        direction="lower_is_better",
        independent_unit="biological_draw",
        independent_n=1,
        descriptive_n=6,
        descriptive_unit="trajectory_cells",
        procedure=(
            "root_oriented_multiscale_diffusion_log2_cp10k_plus_1_full_svd_"
            "blockwise_exact_knn=floor_sqrt_n_capped_15_sparse_eigsh_modes=15"
        ),
    )
    row = downstream._endpoint_row(plan, entry, endpoint)
    row["value"] = value

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="endpoint value is invalid",
    ):
        downstream._validate_endpoint_rows([row], plan, entry)


def test_completed_trajectory_scope_reason_codes_expected_numeric_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    entry = replace(
        plan.entries[0],
        status="completed",
        reason=None,
        evaluator_output_path="runs/fixture.zlib",
        evaluator_output_file_sha256="1" * 64,
        evaluator_output_shape=(6, 4),
        evaluator_output_encoding="zlib_raw_f64_v1",
        evaluator_output_uncompressed_nbytes=192,
        evaluator_output_uncompressed_sha256="2" * 64,
        evaluator_output_sha256="3" * 64,
    )
    source_statuses_sha256 = canonical_sha256(
        [{"run_id": entry.run_id, "status": entry.status, "reason": entry.reason}]
    )
    changed = replace(
        plan,
        entries=(entry,),
        source_statuses_sha256=source_statuses_sha256,
        plan_sha256="0" * 64,
    )
    changed = replace(changed, plan_sha256=canonical_sha256(changed.body()))
    targets = SimpleNamespace()
    monkeypatch.setattr(downstream, "_load_targets", lambda _binding: targets)
    monkeypatch.setattr(
        downstream,
        "_decode_output",
        lambda *_args: SimpleNamespace(),
    )

    def numerical_failure(*_args: object) -> object:
        raise np.linalg.LinAlgError("expected trajectory nonconvergence")

    monkeypatch.setattr(
        downstream,
        "evaluate_trajectory_endpoint",
        numerical_failure,
    )
    monkeypatch.setattr(
        downstream,
        "build_final_trajectory_downstream_evidence_plan",
        lambda _repository, _round_root: changed,
    )

    output = downstream.expected_final_downstream_output_directory(changed)
    result = downstream.run_downstream_evidence(changed, output)
    loaded = downstream.load_downstream_evidence_manifest(output)
    record = loaded.records[0]

    assert result["status"] == "completed"
    assert loaded.endpoint_row_count == 1
    assert loaded.planned_denominator_count == 1
    assert record["run_status"] == "completed"
    assert record["run_reason"] is None
    assert len(record["endpoints"]) == 1
    assert record["endpoints"][0]["endpoint"] == ("trajectory_pseudotime_rank_loss")
    assert record["endpoints"][0]["status"] == "unavailable"
    assert record["endpoints"][0]["reason_code"] == "numeric_evaluation_failed"
    assert record["endpoints"][0]["procedure"] == ("terminal_expected_numeric_failure")


@pytest.mark.parametrize(
    "status",
    [
        "unavailable",
        "failed",
        "timeout",
        "resource_exceeded",
        "infrastructure_error",
        "blocked_authority",
        "budget_exhausted",
    ],
)
def test_trajectory_endpoint_retains_every_terminal_upstream_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    entry = replace(plan.entries[0], status=status, reason=f"fixture_{status}")
    source_statuses_sha256 = canonical_sha256(
        [{"run_id": entry.run_id, "status": entry.status, "reason": entry.reason}]
    )
    changed = replace(
        plan,
        entries=(entry,),
        source_statuses_sha256=source_statuses_sha256,
        plan_sha256="0" * 64,
    )
    changed = replace(changed, plan_sha256=canonical_sha256(changed.body()))
    monkeypatch.setattr(
        downstream,
        "build_final_trajectory_downstream_evidence_plan",
        lambda _repository, _round_root: changed,
    )

    output = downstream.expected_final_downstream_output_directory(changed)
    result = downstream.run_downstream_evidence(changed, output)
    loaded = downstream.load_downstream_evidence_manifest(output)
    record = loaded.records[0]

    assert result["status"] == "completed"
    assert loaded.endpoint_row_count == 1
    assert loaded.planned_denominator_count == 1
    assert record["run_status"] == status
    assert record["run_reason"] == f"fixture_{status}"
    assert len(record["endpoints"]) == 1
    assert record["endpoints"][0]["endpoint"] == ("trajectory_pseudotime_rank_loss")
    assert record["endpoints"][0]["status"] == status
    assert record["endpoints"][0]["reason_code"] == ("upstream_run_not_completed")
    assert record["endpoints"][0]["upstream_reason"] == f"fixture_{status}"


def test_final_downstream_output_namespace_is_exact_and_scope_separated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    trajectory = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    provisional_primary = replace(
        trajectory,
        evidence_scope="all",
        plan_sha256="0" * 64,
    )
    primary = replace(
        provisional_primary,
        plan_sha256=canonical_sha256(provisional_primary.body()),
    )
    binding = trajectory.evaluated_round_binding
    assert binding is not None
    base = (
        Path(binding.repository_root).parent
        / f"{Path(binding.repository_root).name}-final-analysis"
        / "downstream"
        / binding.round_id
        / binding.evaluation_receipt_payload_sha256
    )

    assert downstream.expected_final_downstream_output_directory(primary) == base
    assert downstream.expected_final_downstream_output_directory(trajectory) == (
        base / "trajectory"
    )
    wrong = base / "wrong-trajectory"
    monkeypatch.setattr(
        downstream,
        "build_final_trajectory_downstream_evidence_plan",
        lambda _repository, _round_root: trajectory,
    )
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="receipt-bound.*namespace",
    ):
        downstream.run_downstream_evidence(
            trajectory,
            wrong,
            max_denominators=0,
        )
    assert not wrong.exists()


def test_complete_manifest_revalidates_bound_source_and_dataset_bytes(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )

    source, dataset_path, cells, output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    source_raw = output_path.read_bytes()
    output_path.write_bytes(source_raw + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="evaluator output.*checksum"):
        load_downstream_evidence_manifest(destination)
    output_path.write_bytes(source_raw)

    dataset_raw = dataset_path.read_bytes()
    dataset_path.write_bytes(dataset_raw + b"tamper")
    with pytest.raises(DownstreamEvidenceError, match="dataset raw file checksum"):
        load_downstream_evidence_manifest(destination)


def test_plan_revalidation_forwards_the_evaluated_round_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    binding = _fake_evaluated_round_binding(
        tmp_path / "repository",
        tmp_path / "repository/round-1",
    )
    bound = replace(plan, evaluated_round_binding=binding, plan_sha256="0" * 64)
    bound = replace(bound, plan_sha256=canonical_sha256(bound.body()))
    observed: dict[str, object] = {}

    def rebuild(*args: object, evaluated_round_binding: object, **kwargs: object):
        observed["binding"] = evaluated_round_binding
        return bound

    monkeypatch.setattr(downstream, "build_downstream_evidence_plan", rebuild)

    downstream._revalidate_plan(bound)

    assert observed["binding"] is binding


def test_completed_manifest_missing_prefix_fails_without_repairing_files(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    plan_path = destination / "plan.json"
    missing_record = destination / "records/00000002.json"
    plan_path.unlink()
    missing_record.unlink()

    with pytest.raises(DownstreamEvidenceError):
        run_downstream_evidence(plan, destination)
    assert not plan_path.exists()
    assert not missing_record.exists()


def test_loader_rejects_rehashed_downstream_manifest_schema_extension(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)
    path = destination / "downstream_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["unknown_field"] = "forged"
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_sha256(body)
    _write_canonical(path, manifest)

    with pytest.raises(DownstreamEvidenceError, match="manifest schema differs"):
        load_downstream_evidence_manifest(destination)


def test_complete_manifest_rejects_self_consistent_sealed_source_drift(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    destination = tmp_path / "downstream"
    run_downstream_evidence(plan, destination)

    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"]["configuration_id"] = "forged-configuration"
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    checkpoint_body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(
        DownstreamEvidenceError, match="configuration authority differs"
    ):
        load_downstream_evidence_manifest(destination)


def test_plan_rejects_rehashed_registry_wrapper_with_swapped_method_spec(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import AuthorizedConfiguration

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    candidate, _magic = _test_configuration_authority()
    registry = load_method_registry(Path("study/methods.json"))
    swapped = AuthorizedConfiguration.create(
        method_id="magic",
        configuration_id="registry-default",
        kind="registry",
        payload={
            "schema": "maskimpute-registry-default-configuration-v1",
            "method": asdict(registry.by_id("saver")),
        },
        requires_count_score=False,
        requires_calibration=False,
    )
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["records"][1]["run"]["configuration_sha256"] = (
        swapped.configuration_sha256
    )
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(
        DownstreamEvidenceError, match="registry configuration method payload differs"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=(candidate, swapped),
        )


def test_plan_rejects_source_configuration_and_artifact_authority_mismatch(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    candidate, magic = _test_configuration_authority()
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    forged = checkpoint["records"][0]
    forged["run"]["configuration_sha256"] = magic.configuration_sha256
    forged["p_pre_zero_evidence"] = _current_prezero_evidence(forged["run"])
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(
        DownstreamEvidenceError, match="configuration authority differs"
    ):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=(candidate, magic),
        )


def test_plan_rejects_rehashed_source_run_with_unknown_schema_field(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
    )
    from maskimpute_benchmark.protocol import canonical_sha256

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    checkpoint_path = source / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["records"][0]["run"]["unknown_field"] = "forged"
    body = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = canonical_sha256(body)
    _write_canonical(checkpoint_path, checkpoint)

    with pytest.raises(DownstreamEvidenceError, match="source run schema differs"):
        build_downstream_evidence_plan(
            source,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_development_production_wrapper_rejects_repository_symlink_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.runner as runner

    active_repository = Path(downstream.__file__).resolve().parents[1]
    alias = tmp_path / "repository-alias"
    alias.symlink_to(active_repository, target_is_directory=True)

    def unexpected_authority_load():
        raise AssertionError("symlink was resolved before validation")

    monkeypatch.setattr(runner, "load_runner_authority", unexpected_authority_load)
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="development repository path contains a symlink",
    ):
        downstream.build_development_downstream_evidence_plan(alias)


def test_final_production_wrapper_rejects_round_symlink_ancestor(
    tmp_path: Path,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    active_repository = Path(downstream.__file__).resolve().parents[1]
    round_directory = tmp_path / "round"
    round_directory.mkdir()
    alias = tmp_path / "round-alias"
    alias.symlink_to(round_directory, target_is_directory=True)

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="final round path contains a symlink",
    ):
        downstream.build_final_downstream_evidence_plan(active_repository, alias)


def test_output_symlink_ancestor_is_rejected_before_directory_creation(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        build_downstream_evidence_plan,
        run_downstream_evidence,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
    )
    actual_parent = tmp_path / "actual-output-parent"
    actual_parent.mkdir()
    alias = tmp_path / "output-parent-alias"
    alias.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(DownstreamEvidenceError, match="path contains a symlink"):
        run_downstream_evidence(plan, alias / "downstream")
    assert not (actual_parent / "downstream").exists()


def test_generic_source_and_dataset_roots_reject_symlink_ancestors(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        bind_evaluator_dataset,
        build_downstream_evidence_plan,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    alias = tmp_path / "root-alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(
        DownstreamEvidenceError, match="evaluator dataset path contains a symlink"
    ):
        bind_evaluator_dataset(
            alias / dataset_path.name,
            retained_cell_ids=cells,
        )
    with pytest.raises(
        DownstreamEvidenceError, match="source root path contains a symlink"
    ):
        build_downstream_evidence_plan(
            alias / source.name,
            source_kind="development",
            datasets=(_dataset_binding(dataset_path, cells),),
            configurations=_test_configuration_authority(),
        )


def test_registered_trajectory_binding_is_mandatory_and_exact(tmp_path: Path) -> None:
    from dataclasses import replace

    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        _read_bound_dataset,
        bind_evaluator_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        generate_registered_trajectory_dataset,
        load_trajectory_authority,
    )

    authority = load_trajectory_authority()
    dataset = generate_registered_trajectory_dataset(authority=authority)
    path = tmp_path / "registered-trajectory.h5ad"
    dataset.write_h5ad(path)
    retained = tuple(dataset.obs_names.astype(str))

    with pytest.raises(
        DownstreamEvidenceError, match="registered trajectory authority is required"
    ):
        bind_evaluator_dataset(path, retained_cell_ids=retained)
    with pytest.raises(DownstreamEvidenceError, match="trajectory root differs"):
        bind_evaluator_dataset(
            path,
            retained_cell_ids=retained,
            trajectory_root_cell_id=retained[1],
            trajectory_source_id=authority.source_id,
        )
    with pytest.raises(DownstreamEvidenceError, match="trajectory source differs"):
        bind_evaluator_dataset(
            path,
            retained_cell_ids=retained,
            trajectory_root_cell_id=authority.root_cell_id,
            trajectory_source_id="ad-hoc-trajectory-source",
        )

    binding = bind_evaluator_dataset(
        path,
        retained_cell_ids=retained,
        trajectory_root_cell_id=authority.root_cell_id,
        trajectory_source_id=authority.source_id,
    )
    assert binding.dataset_sha256 == authority.expected_dataset_sha256
    assert binding.trajectory_authority_sha256 == authority.authority_sha256
    assert binding.trajectory_binding_sha256 == authority.binding_sha256
    _read_bound_dataset(binding)
    with pytest.raises(DownstreamEvidenceError, match="authority checksum differs"):
        _read_bound_dataset(replace(binding, trajectory_authority_sha256="0" * 64))
    with pytest.raises(DownstreamEvidenceError, match="binding checksum differs"):
        _read_bound_dataset(replace(binding, trajectory_binding_sha256="0" * 64))
    with pytest.raises(DownstreamEvidenceError, match="trajectory source differs"):
        _read_bound_dataset(
            replace(binding, trajectory_source_id="ad-hoc-trajectory-source")
        )


@pytest.mark.parametrize("mechanism", ["symsim", "sergio", "sparsim", "semisynthetic"])
def test_reconstruction_mechanisms_reject_trajectory_binding_fields(
    tmp_path: Path,
    mechanism: str,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        DownstreamEvidenceError,
        bind_evaluator_dataset,
    )

    path = tmp_path / f"{mechanism}.h5ad"
    dataset, cells, _genes = _dataset(path)
    dataset.obs["mechanism"] = mechanism
    dataset.write_h5ad(path)

    with pytest.raises(
        DownstreamEvidenceError,
        match="reconstruction mechanism cannot carry trajectory authority",
    ):
        bind_evaluator_dataset(
            path,
            retained_cell_ids=cells,
            trajectory_root_cell_id=cells[0],
            trajectory_source_id="ad-hoc-trajectory-source",
        )


def test_selection_schema_four_requires_bound_downstream_completeness(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.downstream_evidence import (
        build_downstream_evidence_plan,
        load_downstream_evidence_manifest,
        run_downstream_evidence,
    )
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.revisions import development_selection_stage_paths
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        attach_downstream_evidence_to_selection_result,
        validate_downstream_selection_completeness,
    )

    source, dataset_path, cells, _output_path = _development_source(tmp_path)
    plan = build_downstream_evidence_plan(
        source,
        source_kind="development",
        datasets=(_dataset_binding(dataset_path, cells),),
        configurations=_test_configuration_authority(),
        source_plan=_development_source_plan(source),
    )
    stage_paths = development_selection_stage_paths(None)
    destination = tmp_path / stage_paths.downstream_directory
    run_downstream_evidence(plan, destination)
    evidence = load_downstream_evidence_manifest(destination)
    selection_records = [
        {
            "mechanism": record["mechanism"],
            "biological_id": record["biological_id"],
            "technical_view": record["technical_view"],
            "dataset_id": record["dataset_id"],
            "dataset_sha256": record["dataset_sha256"],
            "method": record["method"],
            "method_sha256": record["method_artifact_sha256"],
            "model_seed": record["model_seed"],
            "metric": "mse",
            "value": None,
            "status": "failed",
        }
        for record in evidence.records
    ]
    core = {
        "schema_version": 2,
        "dataset_manifest_sha256": "1" * 64,
        "count_score_manifest_sha256": "2" * 64,
        "retained_calibration_artifact_sha256": "3" * 64,
        "evaluation_manifest_sha256": "4" * 64,
        "comparator_selection": {
            "path": (
                "artifacts/study/development/evaluation/comparator_selection.json"
            ),
            "receipt": {},
            "selected_by_method": {},
            "nonexecution_identity_by_method": {},
            "ready_comparison_population_ids": [],
        },
        "records": selection_records,
        "orthogonal_intervals": [],
    }
    payload = {**core, "result_sha256": canonical_sha256(core)}
    source_path = tmp_path / stage_paths.source_selection_input
    _write_canonical(source_path, payload)

    with pytest.raises(SelectionAuthorityError, match="source status"):
        attach_downstream_evidence_to_selection_result(
            payload,
            tmp_path,
            stage_paths.downstream_directory,
        )

    for selection_record, downstream_record in zip(
        selection_records, evidence.records, strict=True
    ):
        selection_record["status"] = downstream_record["run_status"]
    selection_records.append(
        {
            **selection_records[0],
            "metric": "null_de_fpr",
            "value": None,
            "status": "unavailable",
        }
    )
    core["records"] = selection_records
    payload = {**core, "result_sha256": canonical_sha256(core)}
    source_file_sha = _write_canonical(source_path, payload)

    forged_source = {**payload, "operator_override": True}
    forged_source_core = {
        key: value for key, value in forged_source.items() if key != "result_sha256"
    }
    forged_source["result_sha256"] = canonical_sha256(forged_source_core)
    _write_canonical(source_path, forged_source)
    with pytest.raises(SelectionAuthorityError, match="missing or extra"):
        attach_downstream_evidence_to_selection_result(
            forged_source,
            tmp_path,
            stage_paths.downstream_directory,
        )

    changed_source = dict(payload)
    changed_source["dataset_manifest_sha256"] = "9" * 64
    changed_source_core = {
        key: value for key, value in changed_source.items() if key != "result_sha256"
    }
    changed_source["result_sha256"] = canonical_sha256(changed_source_core)
    _write_canonical(source_path, changed_source)
    with pytest.raises(SelectionAuthorityError, match="source selection input differs"):
        attach_downstream_evidence_to_selection_result(
            payload,
            tmp_path,
            stage_paths.downstream_directory,
        )

    source_file_sha = _write_canonical(source_path, payload)
    upgraded = attach_downstream_evidence_to_selection_result(
        payload,
        tmp_path,
        stage_paths.downstream_directory,
    )

    assert upgraded["schema_version"] == 4
    assert upgraded["revision_versions"] == []
    binding = upgraded["downstream_evidence"]
    receipt = validate_downstream_selection_completeness(
        tmp_path, upgraded["records"], binding
    )
    assert receipt["downstream_manifest_sha256"] == evidence.manifest_sha256
    assert binding["endpoint_row_count"] == 16
    assert binding["source_selection_input_path"] == (
        stage_paths.source_selection_input
    )
    assert binding["source_selection_input_file_sha256"] == source_file_sha
    assert binding["source_selection_result_sha256"] == payload["result_sha256"]

    v28_paths = development_selection_stage_paths("v28")
    v28_core = {
        **core,
        "schema_version": 3,
        "revision_versions": ["v28"],
    }
    v28_payload = {
        **v28_core,
        "result_sha256": canonical_sha256(v28_core),
    }
    _write_canonical(tmp_path / v28_paths.source_selection_input, v28_payload)
    shutil.copytree(destination, tmp_path / v28_paths.downstream_directory)
    with pytest.raises(
        SelectionAuthorityError,
        match="downstream revision sources differ",
    ):
        attach_downstream_evidence_to_selection_result(
            v28_payload,
            tmp_path,
            v28_paths.downstream_directory,
        )

    missing_denominator = [
        record
        for record in upgraded["records"]
        if record["method"] != upgraded["records"][0]["method"]
    ]
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(
            tmp_path, missing_denominator, binding
        )

    changed_dataset = [dict(record) for record in upgraded["records"]]
    changed_dataset[0]["dataset_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(tmp_path, changed_dataset, binding)

    changed_method = [dict(record) for record in upgraded["records"]]
    changed_method[0]["method_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="completeness"):
        validate_downstream_selection_completeness(tmp_path, changed_method, binding)

    changed_checkpoint = dict(binding)
    changed_checkpoint["source_checkpoint_file_sha256"] = "0" * 64
    with pytest.raises(SelectionAuthorityError, match="binding differs"):
        validate_downstream_selection_completeness(
            tmp_path, upgraded["records"], changed_checkpoint
        )


def test_final_cli_uses_external_receipt_bound_archive_without_round_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path("scripts/run_final_downstream_evidence.py").absolute()
    specification = importlib.util.spec_from_file_location(
        "run_final_downstream_evidence_test", script_path
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = script
    specification.loader.exec_module(script)

    repository = tmp_path / "repository"
    round_directory = repository / "artifacts/study/final/rounds/round-1"
    round_directory.mkdir(parents=True)
    (round_directory / "evaluation_receipt.json").write_bytes(b"sealed-receipt\n")
    before = {
        path.relative_to(round_directory).as_posix(): path.read_bytes()
        for path in round_directory.rglob("*")
        if path.is_file()
    }
    receipt_sha256 = "a" * 64
    primary_plan = SimpleNamespace(
        evaluated_round_binding=SimpleNamespace(
            round_id="round-1",
            evaluation_receipt_payload_sha256=receipt_sha256,
        )
    )
    trajectory_plan = SimpleNamespace(
        evaluated_round_binding=primary_plan.evaluated_round_binding
    )
    observed: list[tuple[object, Path]] = []
    events: list[str] = []

    monkeypatch.setattr(script, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        script,
        "build_final_downstream_evidence_plan",
        lambda selected_repository, selected_round: (
            events.append("build-primary") or primary_plan
            if selected_repository == repository and selected_round == round_directory
            else pytest.fail("final plan received a different repository or round")
        ),
    )
    monkeypatch.setattr(
        script,
        "build_final_trajectory_downstream_evidence_plan",
        lambda selected_repository, selected_round: (
            events.append("build-trajectory") or trajectory_plan
            if selected_repository == repository and selected_round == round_directory
            else pytest.fail("trajectory plan received a different repository or round")
        ),
        raising=False,
    )
    expected = (
        repository.parent
        / f"{repository.name}-final-analysis"
        / "downstream"
        / "round-1"
        / receipt_sha256
    )
    monkeypatch.setattr(
        script,
        "expected_final_downstream_output_directory",
        lambda selected_plan: (
            expected if selected_plan is primary_plan else expected / "trajectory"
        ),
        raising=False,
    )

    def run_evidence(selected_plan: object, output_directory: Path):
        events.append(
            "run-primary" if selected_plan is primary_plan else "run-trajectory"
        )
        observed.append((selected_plan, output_directory))
        output_directory.mkdir(parents=True)
        (output_directory / "proof.json").write_text("{}\n", encoding="utf-8")
        return {"status": "completed"}

    monkeypatch.setattr(script, "run_downstream_evidence", run_evidence)

    different_working_directory = tmp_path / "different-working-directory"
    different_working_directory.mkdir()
    monkeypatch.chdir(different_working_directory)
    repository_relative_round = round_directory.relative_to(repository)
    assert script.main(["--round-dir", repository_relative_round.as_posix()]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "status": "completed",
        "primary": {"status": "completed"},
        "trajectory": {"status": "completed"},
    }
    assert events == [
        "build-primary",
        "build-trajectory",
        "run-primary",
        "run-trajectory",
    ]
    assert observed == [
        (primary_plan, expected),
        (trajectory_plan, expected / "trajectory"),
    ]
    assert all(not path.is_relative_to(repository) for _plan, path in observed)
    after = {
        path.relative_to(round_directory).as_posix(): path.read_bytes()
        for path in round_directory.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_final_cli_builds_both_plans_before_any_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = Path("scripts/run_final_downstream_evidence.py").absolute()
    specification = importlib.util.spec_from_file_location(
        "run_final_downstream_evidence_preflight_test", script_path
    )
    assert specification is not None and specification.loader is not None
    script = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = script
    specification.loader.exec_module(script)
    repository = tmp_path / "repository"
    round_directory = repository / "round-1"
    round_directory.mkdir(parents=True)
    primary = SimpleNamespace()
    output = tmp_path / "external/downstream"
    monkeypatch.setattr(script, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(
        script,
        "build_final_downstream_evidence_plan",
        lambda *_args: primary,
    )
    monkeypatch.setattr(
        script,
        "build_final_trajectory_downstream_evidence_plan",
        lambda *_args: (_ for _ in ()).throw(
            script.DownstreamEvidenceError("trajectory preflight failed")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "expected_final_downstream_output_directory",
        lambda _plan: output,
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "run_downstream_evidence",
        lambda *_args: pytest.fail("output write began before both plans existed"),
    )

    assert script.main(["--round-dir", str(round_directory)]) == 2
    assert not output.exists()


def _fixture_trajectory_evidence(
    repository: Path,
    round_directory: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    from maskimpute_benchmark.protocol import canonical_sha256
    from maskimpute_benchmark.runner import DEVELOPMENT_MODEL_SEEDS

    plan_body = {
        "schema_version": 1,
        "scope": "supplementary_trajectory",
        "input_hashes": {
            "frozen_method_sha256": "b" * 64,
            "method_registry_sha256": "c" * 64,
            "runtime_lock_sha256": "d" * 64,
            "primary_final_plan_sha256": "1" * 64,
            "trajectory_authority_file_sha256": "e" * 64,
            "trajectory_authority_sha256": "7" * 64,
            "trajectory_binding_sha256": "8" * 64,
            "trajectory_dataset_sha256": "6" * 64,
            "trajectory_dataset_file_sha256": "0" * 64,
            "trajectory_dataset_receipt_sha256": "9" * 64,
            "trajectory_dataset_receipt_file_sha256": "0" * 64,
            "trajectory_method_input_sha256": "f" * 64,
            "trajectory_retained_cell_ids_sha256": "a" * 64,
            "dataset_qc_policy_sha256": "b" * 64,
            "execution_claim_sha256": "2" * 64,
            "execution_environment_sha256": "3" * 64,
            "execution_authority_sha256": "4" * 64,
        },
        "entries": [{"fixture": "trajectory-entry"}],
        "configurations": [{"fixture": "trajectory-configuration"}],
        "model_seed_policy": list(DEVELOPMENT_MODEL_SEEDS),
    }
    trajectory_paths = tuple(
        sorted(
            {
                "results/trajectory/dataset/evaluator.h5ad",
                "results/trajectory/dataset/dataset_receipt.json",
                "results/trajectory/execution_authority/authority.json",
                "results/trajectory/execution_authority/count_score_authority.json",
                "results/trajectory/execution_authority/retained_calibration.json",
                "results/trajectory/execution/records/00000001.json",
                "results/trajectory/execution/execution_manifest.json",
            }
        )
    )
    result_files: list[dict[str, str]] = []
    for ordinal, relative in enumerate(trajectory_paths, start=1):
        path = round_directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"trajectory fixture {ordinal}\n".encode())
        result_files.append(
            {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    result_lookup = {row["path"]: row["sha256"] for row in result_files}
    dataset_path = "results/trajectory/dataset/evaluator.h5ad"
    receipt_path = "results/trajectory/dataset/dataset_receipt.json"
    authority_path = "results/trajectory/execution_authority/authority.json"
    score_path = "results/trajectory/execution_authority/count_score_authority.json"
    calibration_path = (
        "results/trajectory/execution_authority/retained_calibration.json"
    )
    score_repository_path = (
        (round_directory / score_path).relative_to(repository).as_posix()
    )
    calibration_repository_path = (
        (round_directory / calibration_path).relative_to(repository).as_posix()
    )
    manifest_path = "results/trajectory/execution/execution_manifest.json"
    plan_body["input_hashes"]["trajectory_dataset_file_sha256"] = result_lookup[
        dataset_path
    ]
    plan_body["input_hashes"]["trajectory_dataset_receipt_file_sha256"] = result_lookup[
        receipt_path
    ]
    plan_body["input_hashes"]["trajectory_authority_file_sha256"] = result_lookup[
        authority_path
    ]
    plan = {**plan_body, "plan_sha256": canonical_sha256(plan_body)}
    validation_body = {
        "schema_version": 1,
        "status": "eligible_for_final_evaluation_complete_terminal_denominator",
        "scope": "supplementary_trajectory",
        "trajectory_plan_sha256": plan["plan_sha256"],
        "planned_run_count": 1,
        "executed_completed_count": 0,
        "executed_algorithmic_failure_count": 1,
        "executed_status_counts": {"unavailable": 1},
        "not_applicable_count": 0,
        "record_payload_sha256s": ["5" * 64],
    }
    validation = {
        **validation_body,
        "validation_sha256": canonical_sha256(validation_body),
    }
    dataset_binding = {
        "schema_version": "trajectory-execution-dataset-binding-v1",
        "dataset_id": "trajectory-exact-latent-01",
        "mechanism": "synthetic_trajectory",
        "biological_id": "trajectory-draw-01",
        "technical_view": "exact_latent",
        "condition": "trajectory",
        "draw": 1,
        "cells": 2_700,
        "genes": 120,
        "source_id": "registered-synthetic-trajectory-v1",
        "root_cell_id": "trajectory-cell-000001",
        "seed": 20260717,
        "dataset_sha256": "6" * 64,
        "dataset_file_path": dataset_path,
        "dataset_file_sha256": result_lookup[dataset_path],
        "authority_path": "study/trajectory_panel.json",
        "authority_file_sha256": result_lookup[authority_path],
        "authority_sha256": "7" * 64,
        "registered_binding_sha256": "8" * 64,
    }
    evidence_body = {
        "schema_version": 1,
        "status": "completed",
        "scope": "supplementary_trajectory",
        "plan": plan,
        "dataset": {
            "binding": dataset_binding,
            "dataset_path": dataset_path,
            "dataset_file_sha256": result_lookup[dataset_path],
            "dataset_sha256": dataset_binding["dataset_sha256"],
            "receipt_path": receipt_path,
            "receipt_file_sha256": result_lookup[receipt_path],
            "receipt_payload_sha256": "9" * 64,
        },
        "execution_authority": {
            "authority_path": authority_path,
            "authority_file_sha256": result_lookup[authority_path],
            "authority_sha256": "4" * 64,
            "count_score_authority_path": score_repository_path,
            "count_score_authority_file_sha256": result_lookup[score_path],
            "retained_calibration_path": calibration_repository_path,
            "retained_calibration_file_sha256": result_lookup[calibration_path],
            "files": [
                row
                for row in result_files
                if row["path"].startswith("results/trajectory/execution_authority/")
            ],
        },
        "execution_manifest": {
            "path": manifest_path,
            "file_sha256": result_lookup[manifest_path],
            "payload_sha256": "a" * 64,
        },
        "execution_validation": validation,
        "result_files": result_files,
    }
    return (
        {**evidence_body, "evidence_sha256": canonical_sha256(evidence_body)},
        result_files,
    )


def _evaluated_scaling_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.scaling as scaling
    from maskimpute_benchmark.protocol import canonical_sha256

    repository = tmp_path / "repository"
    round_directory = repository / "artifacts/study/final/rounds/round-1"
    execution_directory = round_directory / "results/final/execution"
    execution_directory.mkdir(parents=True)
    execution_body = {
        "schema_version": 1,
        "status": "completed",
        "plan_sha256": "1" * 64,
        "input_hashes": {},
        "planned_run_count": 0,
        "recorded_run_count": 0,
        "records": [],
        "artifact_storage": {
            "evaluator_output_encoding": "zlib_raw_f64_v1",
            "evaluator_output_compression_level": 6,
            "native_output_retention": "omitted_redundant_final_output",
            "p_pre_zero_encoding": "zlib_raw_f64_v1",
            "p_pre_zero_compression_level": 6,
        },
    }
    execution_manifest = {
        **execution_body,
        "manifest_sha256": canonical_sha256(execution_body),
    }
    execution_file_sha256 = _write_canonical(
        execution_directory / "execution_manifest.json", execution_manifest
    )
    scaling_plan_body = {
        "schema_version": 1,
        "input_hashes": {"fixture_authority_sha256": "9" * 64},
        "entries": [{"ordinal": 1, "run_id": "scaling-fixture"}],
        "configurations": [],
    }
    scaling_plan = {
        **scaling_plan_body,
        "plan_sha256": canonical_sha256(scaling_plan_body),
    }
    checkpoint_body = {
        "schema_version": 1,
        "plan_sha256": scaling_plan["plan_sha256"],
        "input_hashes": dict(scaling_plan_body["input_hashes"]),
        "planned_run_count": 1,
        "status": "completed",
        "datasets": [],
        "records": [{"fixture_record": "original"}],
    }
    checkpoint_payload = {
        **checkpoint_body,
        "checkpoint_sha256": canonical_sha256(checkpoint_body),
    }
    checkpoint = scaling.ScalingCheckpoint(
        schema_version=1,
        plan_sha256=str(checkpoint_body["plan_sha256"]),
        input_hashes=dict(checkpoint_body["input_hashes"]),
        planned_run_count=1,
        status="completed",
        datasets=(),
        records=({"fixture_record": "original"},),
        checkpoint_sha256=checkpoint_payload["checkpoint_sha256"],
    )
    checkpoint_relative = "results/scaling/checkpoints/00000001.json"
    checkpoint_path = round_directory / checkpoint_relative
    checkpoint_file_sha256 = _write_canonical(checkpoint_path, checkpoint_payload)
    artifact_relative = "results/scaling/runs/scaling-fixture/run.stdout"
    artifact_path = round_directory / artifact_relative
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"fixture scaling stdout\n")
    artifact_file_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    scaling_result_files = [
        {"path": checkpoint_relative, "sha256": checkpoint_file_sha256},
        {"path": artifact_relative, "sha256": artifact_file_sha256},
    ]
    scaling_evidence_body = {
        "schema_version": 1,
        "status": "completed",
        "plan": scaling_plan,
        "checkpoint_path": checkpoint_relative,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "checkpoint_payload": checkpoint_payload,
        "result_files": scaling_result_files,
    }
    scaling_evidence = {
        **scaling_evidence_body,
        "evidence_sha256": canonical_sha256(scaling_evidence_body),
    }
    trajectory_evidence, trajectory_result_files = _fixture_trajectory_evidence(
        repository,
        round_directory,
    )
    validation_body = {
        "schema_version": 1,
        "status": "eligible_for_final_evaluation_complete_terminal_denominator",
        "final_plan_sha256": "1" * 64,
        "planned_run_count": 0,
        "executed_completed_count": 0,
        "executed_algorithmic_failure_count": 0,
        "executed_status_counts": {},
        "not_applicable_count": 0,
        "record_payload_sha256s": [],
    }
    result_manifest = {
        "schema_version": 1,
        "status": "completed",
        "final_plan_sha256": "1" * 64,
        "final_execution_manifest_path": (
            "results/final/execution/execution_manifest.json"
        ),
        "final_execution_manifest_sha256": execution_file_sha256,
        "final_execution_payload_sha256": execution_manifest["manifest_sha256"],
        "execution_validation": {
            **validation_body,
            "validation_sha256": canonical_sha256(validation_body),
        },
        "storage_preflight": {"schema": "test"},
        "scaling_evidence": scaling_evidence,
        "trajectory_evidence": trajectory_evidence,
        "result_files": sorted(
            [
                {
                    "path": "results/final/execution/execution_manifest.json",
                    "sha256": execution_file_sha256,
                },
                *copy.deepcopy(scaling_result_files),
                *copy.deepcopy(trajectory_result_files),
            ],
            key=lambda value: value["path"],
        ),
    }
    receipt = {
        "schema_version": 1,
        "round_id": "round-1",
        "state": "evaluated",
        "evaluated_at": "2026-07-16T00:00:00Z",
        "execution_claim_id": "claim-1",
        "result_manifest": result_manifest,
        "result_manifest_sha256": canonical_sha256(result_manifest),
        "seed_manifest_sha256": "2" * 64,
        "round_path": "artifacts/study/final/rounds/round-1",
        "round_token": "round-token",
        "repository_instance_id": "repository-instance",
        "worktree_path_sha256": "3" * 64,
        "git_common_dir_device": 1,
        "git_common_dir_inode": 2,
        "study_state_root_device": 1,
        "study_state_root_inode": 3,
        "registry_dir_device": 1,
        "registry_dir_inode": 4,
        "method_commit": "4" * 40,
        "config_sha256": "5" * 64,
        "protocol_sha256": "6" * 64,
        "environment_sha256": "7" * 64,
        "operational_artifact_roots_sha256": "8" * 64,
    }
    receipt_path = round_directory / "evaluation_receipt.json"
    _write_canonical(receipt_path, receipt)

    def validated_receipt(_repository: Path, _round: Path):
        return json.loads(
            (round_directory / "evaluation_receipt.json").read_text(encoding="utf-8")
        )

    monkeypatch.setattr(
        downstream,
        "_validated_evaluated_round_receipt",
        validated_receipt,
        raising=False,
    )

    replay_calls: list[tuple[Path, Path]] = []

    def load_publication_scaling_evidence(
        selected_repository: Path,
        selected_round: Path,
    ):
        replay_calls.append((selected_repository, selected_round))
        return checkpoint

    monkeypatch.setattr(
        scaling,
        "load_publication_scaling_evidence",
        load_publication_scaling_evidence,
    )
    trajectory_replay_calls: list[tuple[Path, Path, str]] = []

    def rederive_trajectory_evidence(
        selected_repository: Path,
        selected_round: Path,
        evidence: object,
        result_files: object,
        *,
        primary_final_plan_sha256: str,
    ):
        trajectory_replay_calls.append(
            (selected_repository, selected_round, primary_final_plan_sha256)
        )
        assert evidence == trajectory_evidence
        assert result_files == result_manifest["result_files"]
        return copy.deepcopy(trajectory_evidence)

    import maskimpute_benchmark.final_runner as final_runner

    monkeypatch.setattr(
        final_runner,
        "_rederive_trajectory_evidence_before_receipt",
        rederive_trajectory_evidence,
    )
    return {
        "repository": repository,
        "round_directory": round_directory,
        "receipt_path": receipt_path,
        "receipt": receipt,
        "result_manifest": result_manifest,
        "scaling_evidence": scaling_evidence,
        "trajectory_evidence": trajectory_evidence,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "replay_calls": replay_calls,
        "trajectory_replay_calls": trajectory_replay_calls,
    }


def test_evaluated_round_binding_requires_and_replays_exact_scaling_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    fixture = _evaluated_scaling_round(tmp_path, monkeypatch)
    repository = fixture["repository"]
    round_directory = fixture["round_directory"]
    result_manifest = fixture["result_manifest"]
    scaling_evidence = fixture["scaling_evidence"]
    trajectory_evidence = fixture["trajectory_evidence"]
    assert isinstance(repository, Path)
    assert isinstance(round_directory, Path)
    assert isinstance(result_manifest, dict)
    assert isinstance(scaling_evidence, dict)
    assert isinstance(trajectory_evidence, dict)

    binding = downstream._read_verified_evaluated_round_binding(
        repository, round_directory
    )
    assert binding.result_manifest_sha256 == canonical_sha256(result_manifest)
    assert binding.scaling_evidence_sha256 == scaling_evidence["evidence_sha256"]
    assert binding.scaling_plan_sha256 == scaling_evidence["plan"]["plan_sha256"]
    assert binding.scaling_checkpoint_path == scaling_evidence["checkpoint_path"]
    assert binding.scaling_checkpoint_history_count == 1
    assert binding.scaling_result_file_count == 2
    assert binding.trajectory_evidence_sha256 == trajectory_evidence["evidence_sha256"]
    assert binding.trajectory_plan_sha256 == trajectory_evidence["plan"]["plan_sha256"]
    assert (
        binding.trajectory_dataset_sha256
        == trajectory_evidence["dataset"]["dataset_sha256"]
    )
    assert binding.trajectory_dataset_id == "trajectory-exact-latent-01"
    assert binding.trajectory_source_id == "registered-synthetic-trajectory-v1"
    assert binding.trajectory_root_cell_id == "trajectory-cell-000001"
    assert binding.trajectory_registered_authority_sha256 == "7" * 64
    assert binding.trajectory_registered_binding_sha256 == "8" * 64
    assert binding.trajectory_execution_claim_sha256 == "2" * 64
    assert binding.trajectory_execution_environment_sha256 == "3" * 64
    assert binding.trajectory_execution_manifest_path == (
        "results/trajectory/execution/execution_manifest.json"
    )
    assert binding.trajectory_planned_run_count == 1
    assert binding.trajectory_result_file_count == 7
    assert fixture["replay_calls"] == [(repository, round_directory)]
    assert fixture["trajectory_replay_calls"] == [
        (repository, round_directory, "1" * 64)
    ]
    assert (
        downstream._evaluated_round_binding_from_payload(binding.to_dict()) == binding
    )
    legacy_binding = binding.to_dict()
    legacy_binding.pop("scaling_evidence_sha256")
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="binding schema differs",
    ):
        downstream._evaluated_round_binding_from_payload(legacy_binding)
    extended_binding = {**binding.to_dict(), "scaling_extension": "forbidden"}
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="binding schema differs",
    ):
        downstream._evaluated_round_binding_from_payload(extended_binding)

    missing_trajectory_binding = binding.to_dict()
    missing_trajectory_binding.pop("trajectory_evidence_sha256")
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="binding schema differs",
    ):
        downstream._evaluated_round_binding_from_payload(missing_trajectory_binding)

    receipt = fixture["receipt"]
    receipt_path = fixture["receipt_path"]
    assert isinstance(receipt, dict)
    assert isinstance(receipt_path, Path)
    receipt["evaluated_at"] = "2026-07-16T00:00:01Z"
    _write_canonical(receipt_path, receipt)
    with pytest.raises(downstream.DownstreamEvidenceError, match="binding changed"):
        downstream._validate_evaluated_round_binding(binding)


@pytest.mark.parametrize(
    "attack",
    [
        "missing_scaling_evidence",
        "missing_trajectory_evidence",
        "extra_result_manifest_field",
        "extra_scaling_field",
        "tampered_plan",
        "tampered_checkpoint",
        "missing_checkpoint_history",
        "tampered_result_inventory",
        "tampered_evidence_hash",
        "extra_trajectory_field",
        "tampered_trajectory_plan",
        "tampered_trajectory_seed_policy",
        "tampered_trajectory_authority_path",
        "tampered_count_score_authority_digest",
        "tampered_retained_calibration_digest",
        "tampered_trajectory_inventory",
        "tampered_trajectory_validation_count",
        "validation_denominator",
        "round_mismatch",
    ],
)
def test_evaluated_round_binding_rejects_scaling_schema_and_binding_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    fixture = _evaluated_scaling_round(tmp_path, monkeypatch)
    repository = fixture["repository"]
    round_directory = fixture["round_directory"]
    receipt_path = fixture["receipt_path"]
    receipt = fixture["receipt"]
    result_manifest = fixture["result_manifest"]
    evidence = fixture["scaling_evidence"]
    trajectory_evidence = fixture["trajectory_evidence"]
    assert isinstance(repository, Path)
    assert isinstance(round_directory, Path)
    assert isinstance(receipt_path, Path)
    assert isinstance(receipt, dict)
    assert isinstance(result_manifest, dict)
    assert isinstance(evidence, dict)
    assert isinstance(trajectory_evidence, dict)

    if attack == "missing_scaling_evidence":
        result_manifest.pop("scaling_evidence")
    elif attack == "missing_trajectory_evidence":
        result_manifest.pop("trajectory_evidence")
    elif attack == "extra_result_manifest_field":
        result_manifest["scaling_extension"] = "forbidden"
    elif attack == "extra_scaling_field":
        evidence["unexpected"] = "extension"
        _resign_scaling_evidence(evidence)
    elif attack == "tampered_plan":
        plan = evidence["plan"]
        assert isinstance(plan, dict)
        entries = plan["entries"]
        assert isinstance(entries, list) and isinstance(entries[0], dict)
        entries[0]["ordinal"] = 2
        _resign_scaling_evidence(evidence)
    elif attack == "tampered_checkpoint":
        checkpoint_payload = evidence["checkpoint_payload"]
        assert isinstance(checkpoint_payload, dict)
        records = checkpoint_payload["records"]
        assert isinstance(records, list) and isinstance(records[0], dict)
        records[0]["fixture_record"] = "tampered"
        _resign_scaling_evidence(evidence)
    elif attack == "missing_checkpoint_history":
        checkpoint_relative = evidence["checkpoint_path"]
        for inventory in (evidence["result_files"], result_manifest["result_files"]):
            assert isinstance(inventory, list)
            inventory[:] = [
                row
                for row in inventory
                if isinstance(row, dict) and row.get("path") != checkpoint_relative
            ]
        _resign_scaling_evidence(evidence)
    elif attack == "tampered_result_inventory":
        rows = evidence["result_files"]
        assert isinstance(rows, list) and isinstance(rows[-1], dict)
        rows[-1]["sha256"] = "0" * 64
        _resign_scaling_evidence(evidence)
    elif attack == "tampered_evidence_hash":
        evidence["evidence_sha256"] = "0" * 64
    elif attack == "extra_trajectory_field":
        trajectory_evidence["extension"] = "forbidden"
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "tampered_trajectory_plan":
        trajectory_plan = trajectory_evidence["plan"]
        assert isinstance(trajectory_plan, dict)
        trajectory_plan["plan_sha256"] = "0" * 64
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "tampered_trajectory_seed_policy":
        trajectory_plan = trajectory_evidence["plan"]
        assert isinstance(trajectory_plan, dict)
        trajectory_plan["model_seed_policy"] = [42, 1_729, 2_027]
        trajectory_plan_body = {
            key: value for key, value in trajectory_plan.items() if key != "plan_sha256"
        }
        from maskimpute_benchmark.protocol import canonical_sha256

        trajectory_plan["plan_sha256"] = canonical_sha256(trajectory_plan_body)
        trajectory_validation = trajectory_evidence["execution_validation"]
        assert isinstance(trajectory_validation, dict)
        trajectory_validation["trajectory_plan_sha256"] = trajectory_plan["plan_sha256"]
        validation_body = {
            key: value
            for key, value in trajectory_validation.items()
            if key != "validation_sha256"
        }
        trajectory_validation["validation_sha256"] = canonical_sha256(validation_body)
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "tampered_trajectory_authority_path":
        trajectory_authority = trajectory_evidence["execution_authority"]
        assert isinstance(trajectory_authority, dict)
        trajectory_authority["count_score_authority_path"] = (
            "artifacts/study/final/rounds/round-forged/results/trajectory/"
            "execution_authority/count_score_authority.json"
        )
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "tampered_count_score_authority_digest":
        trajectory_authority = trajectory_evidence["execution_authority"]
        assert isinstance(trajectory_authority, dict)
        trajectory_authority["count_score_authority_file_sha256"] = "0" * 64
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "tampered_retained_calibration_digest":
        trajectory_authority = trajectory_evidence["execution_authority"]
        assert isinstance(trajectory_authority, dict)
        trajectory_authority["retained_calibration_file_sha256"] = "0" * 64
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "tampered_trajectory_inventory":
        trajectory_rows = trajectory_evidence["result_files"]
        assert isinstance(trajectory_rows, list)
        trajectory_rows[0]["sha256"] = "0" * 64
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "tampered_trajectory_validation_count":
        trajectory_validation = trajectory_evidence["execution_validation"]
        assert isinstance(trajectory_validation, dict)
        trajectory_validation["planned_run_count"] = 2
        validation_body = {
            key: value
            for key, value in trajectory_validation.items()
            if key != "validation_sha256"
        }
        from maskimpute_benchmark.protocol import canonical_sha256

        trajectory_validation["validation_sha256"] = canonical_sha256(validation_body)
        _resign_trajectory_evidence(trajectory_evidence)
    elif attack == "validation_denominator":
        validation = result_manifest["execution_validation"]
        assert isinstance(validation, dict)
        validation["executed_completed_count"] = 1
        validation_body = {
            key: value
            for key, value in validation.items()
            if key != "validation_sha256"
        }
        from maskimpute_benchmark.protocol import canonical_sha256

        validation["validation_sha256"] = canonical_sha256(validation_body)
    elif attack == "round_mismatch":
        receipt["round_id"] = "round-2"
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(attack)
    _resign_evaluation_receipt(receipt_path, receipt)

    with pytest.raises(downstream.DownstreamEvidenceError):
        downstream._read_verified_evaluated_round_binding(
            repository,
            round_directory,
        )


def test_coherent_scaling_evidence_replacement_fails_independent_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    from maskimpute_benchmark.protocol import canonical_sha256

    fixture = _evaluated_scaling_round(tmp_path, monkeypatch)
    repository = fixture["repository"]
    round_directory = fixture["round_directory"]
    receipt_path = fixture["receipt_path"]
    receipt = fixture["receipt"]
    result_manifest = fixture["result_manifest"]
    evidence = fixture["scaling_evidence"]
    checkpoint_path = fixture["checkpoint_path"]
    assert isinstance(repository, Path)
    assert isinstance(round_directory, Path)
    assert isinstance(receipt_path, Path)
    assert isinstance(receipt, dict)
    assert isinstance(result_manifest, dict)
    assert isinstance(evidence, dict)
    assert isinstance(checkpoint_path, Path)

    plan = evidence["plan"]
    checkpoint_payload = evidence["checkpoint_payload"]
    assert isinstance(plan, dict)
    assert isinstance(checkpoint_payload, dict)
    entries = plan["entries"]
    records = checkpoint_payload["records"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    assert isinstance(records, list) and isinstance(records[0], dict)
    entries[0]["run_id"] = "scaling-replacement"
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(plan_body)
    checkpoint_payload["plan_sha256"] = plan["plan_sha256"]
    records[0]["fixture_record"] = "coherent-replacement"
    checkpoint_body = {
        key: value
        for key, value in checkpoint_payload.items()
        if key != "checkpoint_sha256"
    }
    checkpoint_payload["checkpoint_sha256"] = canonical_sha256(checkpoint_body)
    checkpoint_file_sha256 = _write_canonical(checkpoint_path, checkpoint_payload)
    evidence["checkpoint_file_sha256"] = checkpoint_file_sha256
    checkpoint_relative = evidence["checkpoint_path"]
    for inventory in (evidence["result_files"], result_manifest["result_files"]):
        assert isinstance(inventory, list)
        for row in inventory:
            assert isinstance(row, dict)
            if row["path"] == checkpoint_relative:
                row["sha256"] = checkpoint_file_sha256
    _resign_scaling_evidence(evidence)
    _resign_evaluation_receipt(receipt_path, receipt)

    with pytest.raises(downstream.DownstreamEvidenceError, match="scaling.*replay"):
        downstream._read_verified_evaluated_round_binding(
            repository,
            round_directory,
        )


def test_coherent_trajectory_replacement_fails_read_only_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream
    import maskimpute_benchmark.final_runner as final_runner
    from maskimpute_benchmark.protocol import canonical_sha256

    fixture = _evaluated_scaling_round(tmp_path, monkeypatch)
    repository = fixture["repository"]
    round_directory = fixture["round_directory"]
    receipt_path = fixture["receipt_path"]
    receipt = fixture["receipt"]
    evidence = fixture["trajectory_evidence"]
    assert isinstance(repository, Path)
    assert isinstance(round_directory, Path)
    assert isinstance(receipt_path, Path)
    assert isinstance(receipt, dict)
    assert isinstance(evidence, dict)
    independently_replayed = copy.deepcopy(evidence)

    plan = evidence["plan"]
    validation = evidence["execution_validation"]
    assert isinstance(plan, dict)
    assert isinstance(validation, dict)
    entries = plan["entries"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    entries[0]["fixture"] = "coherent-trajectory-replacement"
    plan_body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_sha256(plan_body)
    validation["trajectory_plan_sha256"] = plan["plan_sha256"]
    validation_body = {
        key: value for key, value in validation.items() if key != "validation_sha256"
    }
    validation["validation_sha256"] = canonical_sha256(validation_body)
    _resign_trajectory_evidence(evidence)
    _resign_evaluation_receipt(receipt_path, receipt)

    monkeypatch.setattr(
        final_runner,
        "_rederive_trajectory_evidence_before_receipt",
        lambda *_args, **_kwargs: independently_replayed,
    )

    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="trajectory.*replay",
    ):
        downstream._read_verified_evaluated_round_binding(
            repository,
            round_directory,
        )


def test_receipt_bound_final_output_rejects_round_and_repository_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maskimpute_benchmark.downstream_evidence as downstream

    source, dataset, configuration, source_plan, fixture = _trajectory_source(tmp_path)
    monkeypatch.setattr(
        downstream, "_validate_evaluated_round_binding", lambda _binding: None
    )
    monkeypatch.setattr(downstream, "_read_bound_dataset", lambda _binding: None)
    plan = downstream._build_downstream_evidence_plan(
        source,
        source_kind="final",
        evidence_scope="supplementary_trajectory",
        datasets=(dataset,),
        configurations=(configuration,),
        evaluated_round_binding=fixture["evaluated_round_binding"],
        source_plan=source_plan,
    )
    binding = plan.evaluated_round_binding
    assert binding is not None
    forbidden = Path(binding.round_root) / "results/final/downstream"
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="receipt-bound namespace",
    ):
        downstream._validate_downstream_output_location(plan, forbidden)
    assert not forbidden.exists()

    wrong_external = tmp_path.parent / "wrong-final-analysis/downstream"
    with pytest.raises(
        downstream.DownstreamEvidenceError,
        match="receipt-bound namespace",
    ):
        downstream._validate_downstream_output_location(plan, wrong_external)
    downstream._validate_downstream_output_location(
        plan,
        downstream.expected_final_downstream_output_directory(plan),
    )
