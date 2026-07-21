import copy
from dataclasses import asdict, replace
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import struct
import sys

import pytest

from maskimpute_benchmark.comparator_tuning import (
    AUTHORITY_REVISION,
    ComparatorSmokeOutcome,
    ComparatorTuningError,
    DEVELOPMENT_MAX_CHECKPOINT_BYTES,
    DEVELOPMENT_MAX_EXECUTOR_RECEIPT_BYTES,
    DEVELOPMENT_MAX_LOG_RECEIPT_BYTES,
    DEVELOPMENT_MAX_RECORD_BYTES,
    DEVELOPMENT_STORAGE_RESERVE_BYTES,
    bind_comparator_configuration_identity,
    build_comparator_smoke_input,
    build_comparator_smoke_receipt,
    collapse_comparator_configuration,
    decode_comparator_configuration,
    encode_comparator_configuration,
    load_comparator_smoke_receipt,
    load_comparator_tuning_authority,
    metric_rank_quarters,
    pareto_configuration_ids,
    parse_comparator_tuning_authority,
    run_comparator_tuning_smoke,
    select_one_comparator_method,
)
import maskimpute_benchmark.comparator_tuning as comparator_tuning_module
from maskimpute_benchmark.direct_values import direct_json_value, freeze_direct_mapping
from maskimpute_benchmark.fair_comparator_plan import ComparatorRunIdentity
from maskimpute_benchmark.methods import load_method_registry
from maskimpute_benchmark.runner import (
    AdapterOutcome,
    ExecutionEnvironmentRegistry,
    RepositoryAdapterDispatcher,
    RunnerContractError,
)


ROOT = Path(__file__).resolve().parents[1]


FORBIDDEN_IDENTITY_TOKENS = (
    "hash",
    "digest",
    "checksum",
    "fingerprint",
    "sha",
)


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


def _all_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for child in value.values() for key in _all_keys(child)
        )
    if isinstance(value, list):
        return tuple(key for child in value for key in _all_keys(child))
    return ()


def _set_nested(
    payload: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    target: object = payload
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value


def _write_authority(repository: Path, raw: bytes) -> None:
    authority_path = repository / "study/comparator_tuning.json"
    authority_path.parent.mkdir()
    authority_path.write_bytes(raw)


@pytest.fixture(scope="module")
def smoke_registry():
    return load_method_registry(ROOT / "study/methods.json")


@pytest.fixture(scope="module")
def smoke_authority(smoke_registry):
    return load_comparator_tuning_authority(
        ROOT,
        registry=smoke_registry,
        require_clean=False,
    )


@pytest.fixture(scope="module")
def smoke_bound_rows(smoke_registry, smoke_authority):
    rows = tuple(
        bind_comparator_configuration_identity(
            row,
            smoke_registry.by_id(row.method_id),
            smoke_authority,
        )
        for row in smoke_authority.configurations
    )
    assert tuple(
        (
            row.configuration.method_id,
            row.configuration.configuration_id,
        )
        for row in rows
    ) == tuple(
        (row.method_id, row.configuration_id)
        for row in smoke_authority.configurations
    )
    assert tuple(row.configuration for row in rows) == smoke_authority.configurations
    return rows


@pytest.fixture(scope="module")
def complete_smoke_outcomes(smoke_bound_rows):
    return tuple(
        ComparatorSmokeOutcome(
            configuration=row,
            status="completed",
            reason=None,
            runtime_seconds=1.0,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            rss_measurement="fixed_test_sampler",
            gpu_measurement="fixed_test_sampler",
        )
        for row in smoke_bound_rows
    )


def golden_authority():
    registry = load_method_registry(ROOT / "study/methods.json")
    return load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )


def golden_comparator_records(
    *,
    method_id: str,
    configuration_values: dict[str, tuple[float, float, float, float, float, float]],
    duplicate_each_seed_value: bool,
) -> tuple[dict[str, object], ...]:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    authority_by_id = {
        row.configuration_id: row
        for row in authority.configurations_for(method_id)
    }
    metrics = authority.selection_metrics
    records: list[dict[str, object]] = []
    ordinal = 0
    for configuration_id, configured_values in configuration_values.items():
        row = authority_by_id[configuration_id]
        bound = bind_comparator_configuration_identity(
            row,
            registry.by_id(method_id),
            authority,
        )
        configuration_records: list[dict[str, object]] = []
        for mechanism_index, mechanism in enumerate(
            ("symsim", "sergio", "sparsim", "semisynthetic")
        ):
            for draw_index, biological_id in enumerate(
                ("draw-01", "draw-02"),
                start=1,
            ):
                for view_index, technical_view in enumerate(
                    ("moderate", "severe")
                ):
                    dataset_id = (
                        f"dataset-{mechanism}-{biological_id}-{technical_view}"
                    )
                    for model_seed in (42, 43, 44):
                        ordinal += 1
                        identity = ComparatorRunIdentity(
                            workflow_schema="maskimpute-fair-comparator-run-v1",
                            authority_revision=authority.authority_revision,
                            ordinal=ordinal,
                            method=bound.method,
                            configuration_id=configuration_id,
                            configuration_kind="comparator_tuning",
                            configuration_payload=freeze_direct_mapping(row.payload),
                            dataset_id=dataset_id,
                            mechanism=mechanism,
                            biological_id=biological_id,
                            technical_view=technical_view,
                            mask_seed=1_000 + 10 * mechanism_index + draw_index,
                            model_seed=model_seed,
                            draw_index=draw_index,
                        )
                        identity_json = direct_json_value(identity)
                        assert isinstance(identity_json, dict)
                        applicable_metrics = (
                            metrics
                            if mechanism == "symsim"
                            else tuple(
                                metric
                                for metric in metrics
                                if metric != "mse_pre_dropout_zero"
                            )
                        )
                        metric_rows = []
                        for metric in applicable_metrics:
                            position = metrics.index(metric)
                            seed_offset = (
                                0.0
                                if duplicate_each_seed_value
                                else 0.0001 * (model_seed - 42)
                            )
                            value = float(
                                configured_values[position]
                                + 0.01 * draw_index
                                + 0.001 * view_index
                                + seed_offset
                            )
                            metric_rows.append(
                                {
                                    "identity": copy.deepcopy(identity_json),
                                    "metric": metric,
                                    "value": value,
                                    "n": 10,
                                    "status": "completed",
                                    "reason": None,
                                }
                            )
                        configuration_records.append(
                            {
                                "run": {
                                    "run_id": f"run-{ordinal:04d}-{configuration_id}",
                                    "identity": identity_json,
                                    "status": "completed",
                                    "reason": None,
                                },
                                "metrics": metric_rows,
                                "p_pre_zero_evidence": {
                                    "applicable": False,
                                    "status": "not_applicable",
                                },
                            }
                        )
        assert len(configuration_records) == 48
        assert sum(len(record["metrics"]) for record in configuration_records) == 252
        records.extend(configuration_records)
    return tuple(records)


def _magic_golden_records() -> tuple[dict[str, object], ...]:
    return golden_comparator_records(
        method_id="magic",
        configuration_values={
            "magic-t03": (1.0, 4.0, 2.0, 3.0, 2.0, 4.0),
            "magic-t01": (2.0, 2.0, 2.0, 2.0, 3.0, 2.0),
            "magic-t05": (3.0, 1.0, 3.0, 1.0, 1.0, 3.0),
            "magic-t07": (4.0, 5.0, 4.0, 5.0, 4.0, 5.0),
        },
        duplicate_each_seed_value=True,
    )


def test_seed_view_draw_collapse_and_quarter_rank_golden() -> None:
    result = select_one_comparator_method(
        "magic",
        _magic_golden_records(),
        golden_authority(),
    )
    assert result.configuration_ids == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
        "magic-t07",
    )
    assert result.eligible_configuration_ids == result.configuration_ids
    assert result.pareto_configuration_ids == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
    )
    assert result.configuration("magic-t03").unit_counts == {
        "mse": 8,
        "mse_dropout": 8,
        "gnrmse": 8,
        "mse_pre_dropout_zero": 2,
        "corr_err": 8,
        "mse_non_dropout_nonzero": 8,
    }
    assert all(
        type(value) is int
        for row in result.pareto_rows
        for value in row.metric_rank_quarters.values()
    )
    assert result.selected_configuration_id == min(
        result.pareto_rows,
        key=lambda row: row.selection_tuple,
    ).configuration_id
    for row in (*result.collapsed_rows, *result.pareto_rows):
        assert row.configuration.configuration in golden_authority().configurations
        assert row.configuration.method == result.configuration(
            row.configuration_id
        ).configuration.method


def test_average_ties_encode_exact_quarter_rank_integer() -> None:
    assert metric_rank_quarters(
        {
            "a": (1.0, 1.0),
            "b": (1.0, 2.0),
            "c": (3.0, 2.0),
        }
    ) == {"a": 5, "b": 8, "c": 11}


def test_pareto_filter_requires_weak_all_and_strict_one() -> None:
    result = select_one_comparator_method(
        "magic",
        _magic_golden_records(),
        golden_authority(),
    )
    assert pareto_configuration_ids(result.collapsed_rows) == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
    )


def test_selection_tuple_uses_default_penalty_then_configuration_id() -> None:
    records = golden_comparator_records(
        method_id="magic",
        configuration_values={
            "magic-t03": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t01": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t05": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "magic-t07": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        },
        duplicate_each_seed_value=True,
    )
    result = select_one_comparator_method("magic", records, golden_authority())
    assert result.selected_configuration_id == "magic-t03"
    assert result.pareto_rows[0].selection_tuple == (
        10,
        60,
        10,
        10,
        10,
        10,
        10,
        10,
        0,
        "magic-t03",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("identity", "identity"),
        ("missing_metric", "metric"),
        ("duplicate_metric", "metric"),
        ("wrong_applicability", "applicability"),
        ("nonfinite", "nonfinite"),
        ("unit_grid", "unit grid"),
    ),
)
def test_collapse_fails_closed_on_malformed_direct_records(
    mutation: str,
    message: str,
) -> None:
    records = [copy.deepcopy(record) for record in _magic_golden_records()[:48]]
    if mutation == "identity":
        records[0]["metrics"][0]["identity"]["configuration_id"] = "magic-t01"
    elif mutation == "missing_metric":
        records[0]["metrics"].pop()
    elif mutation == "duplicate_metric":
        records[0]["metrics"].append(copy.deepcopy(records[0]["metrics"][0]))
    elif mutation == "wrong_applicability":
        records[12]["metrics"].append(
            {
                **copy.deepcopy(records[12]["metrics"][0]),
                "metric": "mse_pre_dropout_zero",
            }
        )
    elif mutation == "nonfinite":
        records[0]["metrics"][0]["value"] = float("inf")
    else:
        for record in records:
            if record["run"]["identity"]["model_seed"] == 44:
                record["run"]["identity"]["model_seed"] = 43
                for metric in record["metrics"]:
                    metric["identity"]["model_seed"] = 43
                break
    authority = golden_authority()
    registry = load_method_registry(ROOT / "study/methods.json")
    bound = bind_comparator_configuration_identity(
        authority.configurations_for("magic")[0],
        registry.by_id("magic"),
        authority,
    )
    with pytest.raises(ComparatorTuningError, match=message):
        collapse_comparator_configuration(bound, records)


def test_collapse_intrinsic_terminal_row_is_ineligible_without_aborting_method() -> None:
    records = [copy.deepcopy(record) for record in _magic_golden_records()]
    broken = records[0]
    broken["run"]["status"] = "unavailable"
    broken["run"]["reason"] = "adapter_not_registered"
    for metric in broken["metrics"]:
        metric["value"] = None
        metric["n"] = 0
        metric["status"] = "unavailable"
        metric["reason"] = "adapter_not_registered"
    result = select_one_comparator_method("magic", records, golden_authority())
    assert result.configuration_ids == (
        "magic-t03",
        "magic-t01",
        "magic-t05",
        "magic-t07",
    )
    assert result.eligible_configuration_ids == (
        "magic-t01",
        "magic-t05",
        "magic-t07",
    )
    assert result.configuration("magic-t03").eligible is False
    assert result.configuration("magic-t03").status_counts == {
        "completed": 47,
        "unavailable": 1,
    }


def test_collapse_rejects_bound_authority_reference_drift() -> None:
    authority = golden_authority()
    registry = load_method_registry(ROOT / "study/methods.json")
    bound = bind_comparator_configuration_identity(
        authority.configurations_for("magic")[0],
        registry.by_id("magic"),
        authority,
    )
    drifted = replace(
        bound,
        authority_reference=replace(bound.authority_reference, schema_version=1),
    )
    with pytest.raises(ComparatorTuningError, match="bound comparator identity"):
        collapse_comparator_configuration(
            drifted,
            _magic_golden_records()[:48],
        )


def test_smoke_input_is_exact_truth_free_900_by_500() -> None:
    method_input = build_comparator_smoke_input()
    assert method_input.shape == (900, 500)
    assert method_input.counts[0, 0] == 0
    assert method_input.counts[17, 31] == (
        17 * 17 + 31 * 31 + 7 * (17 ^ 31)
    ) % 6
    assert len(method_input.obs_covariates) == 1
    batch = method_input.obs_covariates[0]
    assert batch.name == "batch"
    assert batch.categories == ("batch-0", "batch-1")
    assert batch.values == tuple(f"batch-{index % 2}" for index in range(900))
    assert not hasattr(method_input, "truth")


def _with_negative_first_formula_zero(method_input):
    canonical_bytes = method_input._count_bytes
    positive_zero = struct.pack("<d", 0.0)
    negative_zero = struct.pack("<d", -0.0)
    assert positive_zero == b"\x00\x00\x00\x00\x00\x00\x00\x00"
    assert negative_zero == b"\x00\x00\x00\x00\x00\x00\x00\x80"
    assert canonical_bytes[:8] == positive_zero
    changed_bytes = negative_zero + canonical_bytes[8:]
    assert changed_bytes != canonical_bytes
    return replace(method_input, _count_bytes=changed_bytes)


def test_smoke_input_rejects_byte_distinct_negative_formula_zero() -> None:
    method_input = build_comparator_smoke_input()
    changed = _with_negative_first_formula_zero(method_input)

    assert changed.counts[0, 0] == method_input.counts[0, 0] == 0.0
    assert changed.counts.tobytes(order="C")[:8] == struct.pack("<d", -0.0)
    assert method_input.counts.tobytes(order="C")[:8] == struct.pack("<d", 0.0)
    with pytest.raises(ComparatorTuningError, match="fixed input"):
        comparator_tuning_module.comparator_smoke_input_descriptor(changed)


def test_smoke_receipt_requires_all_34_completed_and_projected_budget(
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    receipt = build_comparator_smoke_receipt(
        complete_smoke_outcomes,
        authority=smoke_authority,
        registry=smoke_registry,
        bound_configurations=smoke_bound_rows,
    )
    assert receipt["planned_configuration_count"] == 34
    assert receipt["completed_configuration_count"] == 34
    assert receipt["projection_multiplier"] == 48
    assert receipt["status"] == "ready"
    broken = list(complete_smoke_outcomes)
    broken[0] = replace(
        broken[0],
        status="unavailable",
        reason="smoke_unavailable",
    )
    with pytest.raises(
        ComparatorTuningError,
        match="all configurations must complete",
    ):
        build_comparator_smoke_receipt(
            broken,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("peak_rss_bytes", 48 * 1024**3 + 1),
        ("peak_gpu_bytes", 14 * 1024**3 + 1),
    ),
)
def test_smoke_receipt_rejects_each_resource_cap_with_complete_denominator(
    field: str,
    value: int,
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    outcomes = list(complete_smoke_outcomes)
    outcomes[0] = replace(outcomes[0], **{field: value})
    assert len(outcomes) == 34

    with pytest.raises(ComparatorTuningError, match="resource cap"):
        build_comparator_smoke_receipt(
            outcomes,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
        )


@pytest.mark.parametrize("gpu_required", (False, True), ids=("cpu", "gpu"))
def test_smoke_receipt_rejects_projected_method_budget_with_complete_denominator(
    gpu_required: bool,
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    outcomes = list(complete_smoke_outcomes)
    position = next(
        index
        for index, outcome in enumerate(outcomes)
        if smoke_registry.by_id(
            outcome.configuration.configuration.method_id
        ).resources.gpu_required
        is gpu_required
    )
    budget_seconds = 8 * 3600 if gpu_required else 24 * 3600
    outcomes[position] = replace(
        outcomes[position],
        runtime_seconds=budget_seconds / 48.0 + 1.0,
    )
    assert len(outcomes) == 34

    with pytest.raises(ComparatorTuningError, match="method budget"):
        build_comparator_smoke_receipt(
            outcomes,
            authority=smoke_authority,
            registry=smoke_registry,
            bound_configurations=smoke_bound_rows,
        )


def _write_smoke_repository(repository: Path) -> None:
    (repository / "study").mkdir(parents=True)
    shutil.copy2(ROOT / "study/methods.json", repository / "study/methods.json")
    shutil.copy2(
        ROOT / "study/comparator_tuning.json",
        repository / "study/comparator_tuning.json",
    )


def test_smoke_run_uses_all_bound_rows_and_create_only_complete_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    _write_smoke_repository(repository)
    calls = []
    real_loader = load_comparator_tuning_authority

    def load_untracked_fixture(selected_repository, *, registry, require_clean=True):
        return real_loader(
            selected_repository,
            registry=registry,
            require_clean=False,
        )

    monkeypatch.setattr(
        comparator_tuning_module,
        "load_comparator_tuning_authority",
        load_untracked_fixture,
    )

    def fake_executor(request, _dispatcher, _authority):
        calls.append(request)
        assert request.model_seed == 42
        assert request.fixture.shape == (900, 500)
        assert request.method_input.shape == (900, 500)
        return ComparatorSmokeOutcome(
            configuration=request.configuration,
            status="completed",
            reason=None,
            runtime_seconds=1.0,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            rss_measurement="fixed_test_sampler",
            gpu_measurement="fixed_test_sampler",
        )

    receipt = run_comparator_tuning_smoke(
        repository,
        _executor=fake_executor,
    )
    registry = load_method_registry(repository / "study/methods.json")
    authority = load_comparator_tuning_authority(
        repository,
        registry=registry,
        require_clean=False,
    )
    assert tuple(
        (
            request.configuration.configuration.method_id,
            request.configuration.configuration.configuration_id,
        )
        for request in calls
    ) == tuple(
        (row.method_id, row.configuration_id) for row in authority.configurations
    )
    assert len(calls) == 34
    assert load_comparator_smoke_receipt(repository, authority, registry) == receipt
    path = repository / authority.smoke_receipt_path
    first_bytes = path.read_bytes()
    assert run_comparator_tuning_smoke(
        repository,
        _executor=fake_executor,
    ) == receipt
    assert path.read_bytes() == first_bytes

    def changed_executor(request, _dispatcher, _authority):
        outcome = fake_executor(request, _dispatcher, _authority)
        return replace(outcome, runtime_seconds=2.0)

    with pytest.raises(ComparatorTuningError, match="conflicts"):
        run_comparator_tuning_smoke(
            repository,
            _executor=changed_executor,
        )
    assert path.read_bytes() == first_bytes

    def failing_executor(_request, _dispatcher, _authority):
        raise RuntimeError("private executor detail")

    with pytest.raises(ComparatorTuningError, match="adapter boundary failed"):
        run_comparator_tuning_smoke(
            repository,
            _executor=failing_executor,
        )
    assert path.read_bytes() == first_bytes


def test_spawned_smoke_request_retains_complete_fixed_fixture_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    smoke_authority,
    smoke_bound_rows,
    smoke_registry,
) -> None:
    method_input = build_comparator_smoke_input()
    descriptor = comparator_tuning_module.comparator_smoke_input_descriptor(
        method_input
    )
    request = comparator_tuning_module._ComparatorSmokeRequest(
        configuration=smoke_bound_rows[0],
        fixture=descriptor,
        method_input=method_input,
        method_spec=smoke_registry.by_id(
            smoke_bound_rows[0].configuration.method_id
        ),
        model_seed=42,
        ordinal=1,
    )
    environments = ExecutionEnvironmentRegistry(
        repository_root=ROOT.resolve(),
        executable_paths=(),
        lock_only_environment_ids=(),
        registry_sha256="a" * 64,
        runtime_lock_sha256=None,
        runtime_lock_path=None,
        benchmark_python=Path(sys.executable),
        r_library_paths=(),
        execution_environment_sha256="b" * 64,
        python_spawn_search_path=(str(ROOT.resolve()),),
        runtime_identity_snapshots=(),
        runtime_closure_paths_sha256s=(),
        runtime_snapshot=None,
    )
    dispatcher = RepositoryAdapterDispatcher(
        ROOT,
        environments,
        comparator_tuning_authority=smoke_authority,
    )
    captured = {}

    def fake_spawn(direct_request, _executor, **_kwargs):
        captured["request"] = direct_request
        return AdapterOutcome.failed(
            "adapter_exception",
            runtime_seconds=1.0,
            peak_rss_bytes=1024,
            peak_gpu_bytes=0,
            rss_measurement="linux_proc_process_tree_rss",
            gpu_measurement="not_applicable_cpu_only_method",
        )

    monkeypatch.setattr(
        "maskimpute_benchmark.runner.execute_direct_adapter_in_spawned_process",
        fake_spawn,
    )
    comparator_tuning_module._execute_smoke_request_in_spawned_dispatcher(
        request,
        dispatcher,
        smoke_authority,
    )

    inner = captured["request"]
    assert inner.smoke_fixture == descriptor
    assert inner.to_dict()["smoke_fixture"] == json.loads(
        json.dumps(asdict(descriptor))
    )
    changed = _with_negative_first_formula_zero(method_input)
    with pytest.raises(RunnerContractError, match="fixed input"):
        replace(inner, method_input=changed)


def test_smoke_loader_recomputes_complete_receipt(
    tmp_path: Path,
    complete_smoke_outcomes,
    smoke_authority,
    smoke_registry,
    smoke_bound_rows,
) -> None:
    repository = tmp_path / "repository"
    _write_smoke_repository(repository)
    receipt = build_comparator_smoke_receipt(
        complete_smoke_outcomes,
        authority=smoke_authority,
        registry=smoke_registry,
        bound_configurations=smoke_bound_rows,
    )
    path = repository / smoke_authority.smoke_receipt_path
    path.parent.mkdir(parents=True)
    path.write_bytes(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    local_registry = load_method_registry(repository / "study/methods.json")
    local_authority = load_comparator_tuning_authority(
        repository,
        registry=local_registry,
        require_clean=False,
    )
    loaded = load_comparator_smoke_receipt(
        repository,
        local_authority,
        local_registry,
    )
    assert loaded == receipt
    changed = copy.deepcopy(receipt)
    changed["fixture"]["maximum"] = 3.0
    path.write_bytes(
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ComparatorTuningError, match="differs"):
        load_comparator_smoke_receipt(
            repository,
            local_authority,
            local_registry,
        )
    changed = copy.deepcopy(receipt)
    changed["outcomes"][0]["runtime_seconds"] = 1
    path.write_bytes(
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ComparatorTuningError, match="measurement is invalid"):
        load_comparator_smoke_receipt(
            repository,
            local_authority,
            local_registry,
        )
    changed = copy.deepcopy(receipt)
    changed["outcomes"][0]["runtime_seconds"] = -0.0
    path.write_bytes(
        json.dumps(
            changed,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ComparatorTuningError, match="measurement is invalid"):
        load_comparator_smoke_receipt(
            repository,
            local_authority,
            local_registry,
        )


def test_smoke_cli_has_no_override_and_never_runs_real_workload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = ROOT / "scripts/run_comparator_tuning_smoke.py"
    spec = importlib.util.spec_from_file_location("task9_smoke_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "run_comparator_tuning_smoke",
        lambda repository: {
            "status": "ready",
            "planned_configuration_count": 34,
            "completed_configuration_count": 34,
        },
    )
    monkeypatch.setattr(sys, "argv", [str(script)])
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "path": "artifacts/study/development/evaluation/comparator_smoke.json",
        "status": "ready",
        "planned_configuration_count": 34,
        "completed_configuration_count": 34,
    }
    monkeypatch.setattr(sys, "argv", [str(script), "--repository", str(ROOT)])
    with pytest.raises(SystemExit, match="2"):
        module.main()


def test_tracked_comparator_authority_uses_only_direct_identity() -> None:
    payload = json.loads((ROOT / "study/comparator_tuning.json").read_text())
    assert payload["authority_revision"] == "fair-comparator-direct-v1"
    assert not any(
        token in key.lower()
        for key in _all_keys(payload)
        for token in FORBIDDEN_IDENTITY_TOKENS
    )
    parameters = inspect.signature(decode_comparator_configuration).parameters
    assert tuple(parameters) == ("method_id", "payload")


def test_bound_comparator_contains_full_method_projection() -> None:
    from maskimpute_benchmark.comparator_tuning import (
        bind_comparator_configuration_identity,
        comparator_method_binding,
    )

    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )
    row = authority.configurations_for("magic")[0]
    bound = bind_comparator_configuration_identity(
        row, registry.by_id("magic"), authority
    )
    assert bound.configuration == row
    assert bound.authority_reference.authority_revision == authority.authority_revision
    assert bound.method == comparator_method_binding(registry.by_id("magic"))


def test_comparator_binding_rejects_detached_or_noncanonical_authority_rows() -> None:
    from maskimpute_benchmark.comparator_tuning import (
        bind_comparator_configuration_identity,
    )

    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )
    row = authority.configurations_for("magic")[0]

    with pytest.raises(ComparatorTuningError, match="registry method"):
        bind_comparator_configuration_identity(row, registry.by_id("dca"), authority)

    with pytest.raises(ComparatorTuningError, match="one exact authority"):
        bind_comparator_configuration_identity(
            replace(row, configuration_id="magic-detached"),
            registry.by_id("magic"),
            authority,
        )

    noncanonical = replace(row, payload_json=json.dumps(dict(row.payload), indent=2))
    noncanonical_authority = replace(
        authority,
        configurations=(
            noncanonical,
            *tuple(item for item in authority.configurations if item != row),
        ),
    )
    with pytest.raises(ComparatorTuningError, match="not canonical JSON"):
        bind_comparator_configuration_identity(
            noncanonical,
            registry.by_id("magic"),
            noncanonical_authority,
        )

    duplicate_authority = replace(
        authority,
        configurations=(*authority.configurations, row),
    )
    with pytest.raises(ComparatorTuningError, match="one exact authority"):
        bind_comparator_configuration_identity(
            row, registry.by_id("magic"), duplicate_authority
        )


def test_all_normative_configurations_round_trip_exactly() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )

    assert authority.schema_version == 2
    assert authority.authority_revision == AUTHORITY_REVISION
    assert len(authority.configurations) == 34
    for row in authority.configurations:
        decoded = row.decode()
        assert encode_comparator_configuration(decoded) == dict(row.payload)
        dataclass_payload = asdict(decoded)
        if row.method_id == "dca":
            dataclass_payload["hidden_size"] = list(dataclass_payload["hidden_size"])
        assert dataclass_payload == dict(row.payload)


def test_decode_comparator_configuration_is_closed_and_exact() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )
    for row in authority.configurations:
        missing = dict(row.payload)
        missing.pop(next(iter(missing)))
        with pytest.raises(ComparatorTuningError, match="complete field set"):
            decode_comparator_configuration(row.method_id, missing)

        extra = {**dict(row.payload), "unexpected": 1}
        with pytest.raises(ComparatorTuningError, match="complete field set"):
            decode_comparator_configuration(row.method_id, extra)

    magic = authority.configurations_for("magic")[0]
    bool_as_int = {**dict(magic.payload), "knn": True}
    with pytest.raises(ComparatorTuningError, match="primitive type"):
        decode_comparator_configuration("magic", bool_as_int)

    dca = authority.configurations_for("dca")[0]
    tuple_payload = {**dict(dca.payload), "hidden_size": (64, 32, 64)}
    with pytest.raises(ComparatorTuningError, match="JSON array"):
        decode_comparator_configuration("dca", tuple_payload)

    afmf = authority.configurations_for("afmf")[0]
    negative_zero = {**dict(afmf.payload), "lambda_p": -0.0}
    with pytest.raises(ComparatorTuningError, match="invalid float value"):
        decode_comparator_configuration("afmf", negative_zero)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        pytest.param(("schema_version",), True, id="schema-version-type"),
        pytest.param(("authority_revision",), "other-revision", id="revision"),
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
        pytest.param(
            ("established_comparator_ids",), ["alra"], id="established-set"
        ),
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
        pytest.param(
            ("selection", "pareto_rule"), "changed", id="selection-pareto"
        ),
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
        pytest.param(
            ("budgets", "per_run_timeout_seconds"), 1, id="budget-timeout"
        ),
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
        pytest.param(
            ("storage", "max_checkpoint_bytes"), 1, id="storage-checkpoint"
        ),
        pytest.param(("storage", "reserve_bytes"), 1, id="storage-reserve"),
        pytest.param(("smoke", "receipt_path"), "elsewhere.json", id="smoke-path"),
        pytest.param(("smoke", "cells"), 899, id="smoke-cells"),
        pytest.param(("smoke", "genes"), 499, id="smoke-genes"),
        pytest.param(("smoke", "model_seed"), 43, id="smoke-seed"),
        pytest.param(("smoke", "batch_rule"), "changed", id="smoke-batches"),
        pytest.param(("smoke", "count_formula"), "changed", id="smoke-formula"),
        pytest.param(
            ("smoke", "projection_multiplier"), 47, id="smoke-projection"
        ),
        pytest.param(
            ("smoke", "output_retention"), "retained", id="smoke-retention"
        ),
    ),
)
def test_authority_rejects_policy_mutation(
    path: tuple[str, ...], replacement: object
) -> None:
    payload = _tracked_payload()
    _set_nested(payload, path, replacement)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


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
def test_authority_rejects_extra_nested_field(
    section_path: tuple[str, ...],
) -> None:
    payload = _tracked_payload()
    target: object = payload
    for key in section_path:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target["unexpected"] = "forged"
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError, match="missing or extra fields"):
        parse_comparator_tuning_authority(payload, registry=registry)


@pytest.mark.parametrize(
    "mutation",
    (
        "row-order",
        "row-count",
        "configuration-id",
        "duplicate-configuration-id",
        "duplicate-payload-under-another-id",
        "multiple-defaults",
        "default-payload",
        "payload-mutation",
    ),
)
def test_authority_rejects_grid_mutation(mutation: str) -> None:
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
    elif mutation == "duplicate-payload-under-another-id":
        magic_second["payload"] = copy.deepcopy(magic_default["payload"])
    elif mutation == "multiple-defaults":
        magic_second["is_upstream_default"] = True
    elif mutation == "default-payload":
        magic_default["payload"] = copy.deepcopy(magic_second["payload"])
    elif mutation == "payload-mutation":
        second_payload = magic_second["payload"]
        assert isinstance(second_payload, dict)
        second_payload["diffusion_time"] = 2
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


def test_authority_rejects_signed_negative_zero_payload_mutation() -> None:
    payload = _tracked_payload()
    configurations = payload["configurations"]
    assert isinstance(configurations, list)
    afmf_default = configurations[18]
    assert isinstance(afmf_default, dict)
    afmf_payload = afmf_default["payload"]
    assert isinstance(afmf_payload, dict)
    afmf_payload["lambda_p"] = -0.0

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


def test_authority_rejects_unicode_payload_mutation() -> None:
    payload = _tracked_payload()
    configurations = payload["configurations"]
    assert isinstance(configurations, list)
    magic_default = configurations[1]
    assert isinstance(magic_default, dict)
    magic_payload = magic_default["payload"]
    assert isinstance(magic_payload, dict)
    magic_payload["solver"] = "\ud800"

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError):
        parse_comparator_tuning_authority(payload, registry=registry)


@pytest.mark.parametrize("schema", ("old", "mixed"))
def test_authority_rejects_old_and_mixed_schema(schema: str) -> None:
    payload = _tracked_payload()
    if schema == "old":
        payload["schema_version"] = 1
        payload.pop("authority_revision")
        payload["payload_sha256"] = "0" * 64
    else:
        payload["payload_sha256"] = "0" * 64
        configurations = payload["configurations"]
        assert isinstance(configurations, list)
        first = configurations[0]
        assert isinstance(first, dict)
        first["payload_sha256"] = "0" * 64

    registry = load_method_registry(ROOT / "study/methods.json")
    with pytest.raises(ComparatorTuningError, match="missing or extra fields"):
        parse_comparator_tuning_authority(payload, registry=registry)


@pytest.mark.parametrize(
    "malformation", ("noncanonical", "duplicate", "nonfinite", "unicode-drift")
)
def test_loader_rejects_malformed_authority_bytes(
    tmp_path: Path, malformation: str
) -> None:
    payload = _tracked_payload()
    canonical = json.dumps(payload, indent=2).encode() + b"\n"
    if malformation == "noncanonical":
        raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    elif malformation == "duplicate":
        raw = canonical.replace(
            b'  "schema_version": 2,',
            b'  "schema_version": 2,\n  "schema_version": 2,',
            1,
        )
    elif malformation == "nonfinite":
        raw = canonical.replace(b'    "cells": 900,', b'    "cells": NaN,', 1)
    elif malformation == "unicode-drift":
        raw = canonical.replace(b'"solver": "exact"', b'"solver": "\\u0065xact"', 1)
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(malformation)
    _write_authority(tmp_path, raw)
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError):
        load_comparator_tuning_authority(
            tmp_path, registry=registry, require_clean=False
        )


def test_loader_rejects_non_regular_authority(tmp_path: Path) -> None:
    study = tmp_path / "study"
    study.mkdir()
    (study / "comparator_tuning.json").symlink_to(
        ROOT / "study/comparator_tuning.json"
    )
    registry = load_method_registry(ROOT / "study/methods.json")

    with pytest.raises(ComparatorTuningError, match="owned regular file"):
        load_comparator_tuning_authority(
            tmp_path, registry=registry, require_clean=False
        )


def test_tracked_authority_has_exact_grid_and_operational_contract() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT, registry=registry, require_clean=False
    )

    assert authority.method_order == tuple(EXPECTED_ORDER)
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
