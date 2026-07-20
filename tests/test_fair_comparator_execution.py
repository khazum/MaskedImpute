from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
import zlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from maskimpute_benchmark.comparator_tuning import (
    comparator_method_binding,
    load_comparator_tuning_authority,
)
from maskimpute_benchmark.fair_comparator_execution import (
    DirectExecutionRequest,
    DirectMetricRow,
    DirectPreZeroEvidence,
    create_direct_request,
    execute_direct_request,
)
from maskimpute_benchmark.fair_comparator_plan import (
    ComparatorRunIdentity,
    DirectPlanEntry,
    describe_prepared_input,
    direct_run_id,
)
from maskimpute_benchmark.methods import (
    DirectAdapterExecution,
    count_equivalent_to_log2_cp10k,
    finalize_direct_method_output,
    load_method_registry,
    prepare_method_input,
)
from maskimpute_benchmark.metrics import reconstruction_metrics
from maskimpute_benchmark.runner import (
    AdapterOutcome,
    DatasetBinding,
    DatasetQCAudit,
    PreparedDataset,
    RunnerContractError,
)


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IDENTITY_TOKENS = ("hash", "digest", "checksum", "fingerprint", "sha")


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return tuple((key, _freeze(nested)) for key, nested in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(nested) for nested in value)
    return value


def _all_keys(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(value) + tuple(
            key for nested in value.values() for key in _all_keys(nested)
        )
    if isinstance(value, list):
        return tuple(key for nested in value for key in _all_keys(nested))
    return ()


def _prepared() -> PreparedDataset:
    counts = np.asarray([[2, 0, 1], [0, 3, 0]], dtype=np.int64)
    cell_ids = ["cell-1", "cell-2"]
    gene_ids = ["gene-1", "gene-2", "gene-3"]
    view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=gene_ids),
    )
    view.uns["source_dataset_sha256"] = "a" * 64
    view.uns["allowed_covariates"] = {"obs": [], "var": []}
    method_input = prepare_method_input(view)
    evaluator = ad.AnnData(
        X=counts,
        obs=pd.DataFrame({"draw": [1, 1]}, index=cell_ids),
        var=pd.DataFrame(index=gene_ids),
        layers={"pre_capture_counts": counts + 1},
    )
    evaluator.uns.update(
        {
            "truth_kind": "exact_pre_capture",
            "primary_truth_layer": "pre_capture_counts",
            "provenance": {"seeds": {"measurement": 20_001}},
        }
    )
    binding = DatasetBinding(
        mechanism="symsim",
        biological_id="draw-01",
        technical_view="moderate",
        dataset_id="dataset-test",
        dataset_sha256="a" * 64,
        output_file_sha256="b" * 64,
        truth_sha256="c" * 64,
        output_path="dev/datasets/symsim/draw-01/moderate.h5ad",
        independent_unit_id="biological-test",
        cells=2,
        genes=3,
        manifest_sha256="d" * 64,
        protocol_sha256="e" * 64,
        design_sha256="f" * 64,
        seed_source_sha256="1" * 64,
    )
    return PreparedDataset(
        binding=binding,
        audit=DatasetQCAudit(
            excluded_cell_count=0,
            excluded_cell_ids_sha256="2" * 64,
            retained_cell_count=2,
            retained_cell_ids_sha256="3" * 64,
            excluded_cell_ids=(),
            retained_cell_ids=tuple(cell_ids),
        ),
        method_input=method_input,
        evaluator_dataset=evaluator,
    )


def _direct_case(
    method_id: str = "magic",
    configuration_id: str | None = None,
):
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    prepared = _prepared()
    descriptor = describe_prepared_input(prepared)
    rows = authority.configurations_for(method_id)
    row = (
        rows[0]
        if configuration_id is None
        else next(value for value in rows if value.configuration_id == configuration_id)
    )
    spec = registry.by_id(method_id)
    identity = ComparatorRunIdentity(
        workflow_schema="maskimpute-fair-comparator-run-v1",
        authority_revision=authority.authority_revision,
        ordinal=1,
        method=comparator_method_binding(spec),
        configuration_id=row.configuration_id,
        configuration_kind="comparator_tuning",
        configuration_payload=_freeze(dict(row.payload)),
        dataset_id=prepared.binding.dataset_id,
        mechanism=prepared.binding.mechanism,
        biological_id=prepared.binding.biological_id,
        technical_view=prepared.binding.technical_view,
        mask_seed=descriptor.mask_seed,
        model_seed=42,
        draw_index=1,
    )
    entry = DirectPlanEntry(
        run_id=direct_run_id(identity),
        identity=identity,
        preflight_status="planned",
        preflight_reason=None,
        requires_count_score=False,
        requires_calibration=False,
    )
    request = create_direct_request(
        entry,
        prepared,
        descriptor,
        spec,
        row,
        timeout_seconds=5,
    )
    return request, entry, prepared, descriptor, spec, row, authority


def _completed_outcome(request: DirectExecutionRequest) -> AdapterOutcome:
    output = finalize_direct_method_output(
        request.method_spec,
        request.method_input,
        request.method_input.counts,
        output_scale=request.method_spec.output_scale,
        obs_ids=request.method_input.obs_ids,
        var_ids=request.method_input.var_ids,
    )
    return AdapterOutcome.completed(
        DirectAdapterExecution(
            output=output,
            stdout=b"abc",
            stderr=b"err",
        ),
        runtime_seconds=1.5,
        peak_rss_bytes=128,
        peak_gpu_bytes=0,
    )


def _slow_direct_executor(_request: DirectExecutionRequest) -> AdapterOutcome:
    time.sleep(0.4)
    return AdapterOutcome.unavailable("child_completed")


def _direct_terminal_executor(_request: DirectExecutionRequest) -> AdapterOutcome:
    return AdapterOutcome.unavailable("synthetic_terminal")


class _DirectFixedResourceSampler:
    def __init__(self, *, rss: int | None, gpu: int | None) -> None:
        self.rss = rss
        self.gpu = gpu

    def sample(self, _process_id: int, *, gpu_required: bool):
        from maskimpute_benchmark.runner import ResourceSample

        return ResourceSample(
            peak_rss_bytes=self.rss,
            peak_gpu_bytes=self.gpu if gpu_required else 0,
            rss_provenance="synthetic_parent_rss",
            gpu_provenance=(
                "synthetic_parent_gpu" if gpu_required else "not_applicable_cpu"
            ),
        )


def test_all_ten_comparator_adapters_receive_exact_authority_payloads() -> None:
    registry = load_method_registry(ROOT / "study/methods.json")
    authority = load_comparator_tuning_authority(
        ROOT,
        registry=registry,
        require_clean=False,
    )
    prepared = _prepared()
    descriptor = describe_prepared_input(prepared)
    selected = authority.configurations
    received: dict[str, object] = {}

    def adapter(method_id: str):
        def spy(spec, method_input, *, seed, config):
            received[method_id] = config
            output = finalize_direct_method_output(
                spec,
                method_input,
                method_input.counts,
                output_scale=spec.output_scale,
                obs_ids=method_input.obs_ids,
                var_ids=method_input.var_ids,
            )
            execution = DirectAdapterExecution(
                output=output,
                stdout=b"abc",
                stderr=b"",
            )
            return AdapterOutcome.completed(
                execution,
                runtime_seconds=1.5,
                peak_rss_bytes=128,
                peak_gpu_bytes=0,
            )

        return spy

    adapters = {method_id: adapter(method_id) for method_id in authority.method_order}
    for ordinal, row in enumerate(selected, start=1):
        spec = registry.by_id(row.method_id)
        identity = ComparatorRunIdentity(
            workflow_schema="maskimpute-fair-comparator-run-v1",
            authority_revision=authority.authority_revision,
            ordinal=ordinal,
            method=comparator_method_binding(spec),
            configuration_id=row.configuration_id,
            configuration_kind="comparator_tuning",
            configuration_payload=_freeze(dict(row.payload)),
            dataset_id=prepared.binding.dataset_id,
            mechanism=prepared.binding.mechanism,
            biological_id=prepared.binding.biological_id,
            technical_view=prepared.binding.technical_view,
            mask_seed=descriptor.mask_seed,
            model_seed=42,
            draw_index=1,
        )
        entry = DirectPlanEntry(
            run_id=direct_run_id(identity),
            identity=identity,
            preflight_status="planned",
            preflight_reason=None,
            requires_count_score=False,
            requires_calibration=False,
        )
        request = create_direct_request(
            entry,
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )

        result = execute_direct_request(
            request,
            prepared,
            authority,
            adapters,
        )

        assert received[row.method_id] == row.decode()
        assert asdict(result.run.stdout) == {
            "stream": "stdout",
            "original_byte_count": 3,
            "capture_policy": "discard_content",
            "terminal_reason": None,
        }
        assert not any(
            key.casefold() != "shape" and token in key.casefold()
            for value in (request.to_dict(), result.to_dict())
            for key in _all_keys(value)
            for token in FORBIDDEN_IDENTITY_TOKENS
        )


def test_create_direct_request_closes_descriptor_method_label_and_payload() -> None:
    request, entry, prepared, descriptor, spec, row, _authority = _direct_case()
    assert request.method_input is prepared.method_input

    with pytest.raises(RunnerContractError, match="descriptor"):
        create_direct_request(
            entry,
            prepared,
            replace(descriptor, total_count=descriptor.total_count + 1.0),
            spec,
            row,
            timeout_seconds=5,
        )


def test_create_direct_request_rejects_equal_numeric_type_substitutions() -> None:
    _request, entry, prepared, descriptor, spec, row, _authority = _direct_case()
    with pytest.raises(RunnerContractError, match="descriptor"):
        create_direct_request(
            entry,
            prepared,
            replace(descriptor, minimum=False),
            spec,
            row,
            timeout_seconds=5,
        )
    with pytest.raises(RunnerContractError, match="descriptor"):
        create_direct_request(
            entry,
            prepared,
            replace(descriptor, minimum=-0.0),
            spec,
            row,
            timeout_seconds=5,
        )
    with pytest.raises(RunnerContractError, match="descriptor"):
        create_direct_request(
            entry,
            prepared,
            replace(descriptor, cell_ids=list(descriptor.cell_ids)),
            spec,
            row,
            timeout_seconds=5,
        )
    changed_method = replace(entry.identity.method, max_gpu_gib=False)
    changed_identity = replace(entry.identity, method=changed_method)
    with pytest.raises(RunnerContractError, match="method projection"):
        create_direct_request(
            replace(
                entry, identity=changed_identity, run_id=direct_run_id(changed_identity)
            ),
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )
    changed_identity = replace(entry.identity, draw_index=True)
    with pytest.raises(RunnerContractError, match="draw index"):
        create_direct_request(
            replace(
                entry, identity=changed_identity, run_id=direct_run_id(changed_identity)
            ),
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )
    with pytest.raises(RunnerContractError, match="method projection"):
        changed_method = replace(entry.identity.method, integration_status="changed")
        changed_identity = replace(entry.identity, method=changed_method)
        create_direct_request(
            replace(
                entry, identity=changed_identity, run_id=direct_run_id(changed_identity)
            ),
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )
    with pytest.raises(RunnerContractError, match="authority row"):
        changed_identity = replace(entry.identity, configuration_id="magic-t99")
        create_direct_request(
            replace(
                entry, identity=changed_identity, run_id=direct_run_id(changed_identity)
            ),
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )
    payload = dict(row.payload)
    payload["diffusion_time"] = 11
    with pytest.raises(RunnerContractError, match="authority row"):
        changed_identity = replace(
            entry.identity,
            configuration_payload=_freeze(payload),
        )
        create_direct_request(
            replace(
                entry, identity=changed_identity, run_id=direct_run_id(changed_identity)
            ),
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )


@pytest.mark.parametrize(
    "changed_spec",
    (
        lambda spec: replace(
            spec,
            source=replace(spec.source, cache_path="other-cache"),
        ),
        lambda spec: replace(spec, input_scale="other-input"),
        lambda spec: replace(spec, output_scale="other-output"),
        lambda spec: replace(spec, stochastic=False),
        lambda spec: replace(spec, seed_policy="other-seed-policy"),
        lambda spec: replace(
            spec,
            resources=replace(spec.resources, cpu_cores=spec.resources.cpu_cores + 1),
        ),
        lambda spec: replace(
            spec,
            preserves_observed_positives=not spec.preserves_observed_positives,
        ),
    ),
    ids=(
        "source-cache-path",
        "input-scale",
        "output-scale",
        "stochastic",
        "seed-policy",
        "cpu-cores",
        "observed-positive-policy",
    ),
)
def test_create_direct_request_binds_every_execution_relevant_method_field(
    changed_spec,
) -> None:
    _request, entry, prepared, descriptor, spec, row, _authority = _direct_case()

    with pytest.raises(RunnerContractError, match="method projection"):
        create_direct_request(
            entry,
            prepared,
            descriptor,
            changed_spec(spec),
            row,
            timeout_seconds=5,
        )


def test_create_direct_request_normalizes_unknown_payload_and_rejects_default() -> None:
    _request, entry, prepared, descriptor, spec, row, _authority = _direct_case()
    unknown_payload = {**dict(row.payload), "unknown": 1}
    unknown_row = replace(
        row,
        payload_json=json.dumps(unknown_payload, sort_keys=True, separators=(",", ":")),
    )
    unknown_identity = replace(
        entry.identity,
        configuration_payload=_freeze(unknown_payload),
    )
    with pytest.raises(RunnerContractError, match="typed payload"):
        create_direct_request(
            replace(
                entry,
                identity=unknown_identity,
                run_id=direct_run_id(unknown_identity),
            ),
            prepared,
            descriptor,
            spec,
            unknown_row,
            timeout_seconds=5,
        )

    default_identity = replace(
        entry.identity,
        configuration_id="registry-default",
    )
    with pytest.raises(RunnerContractError, match="registry-default"):
        create_direct_request(
            replace(
                entry,
                identity=default_identity,
                run_id=direct_run_id(default_identity),
            ),
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )


def test_execute_direct_request_rejects_pre_dispatch_drift_and_duplicates() -> None:
    request, _entry, prepared, _descriptor, _spec, row, authority = _direct_case()
    attempted: list[bool] = []

    def spy(*_args, **_kwargs):
        attempted.append(True)
        return _completed_outcome(request)

    payload = dict(row.payload)
    payload["diffusion_time"] = 11
    drifted = replace(
        request,
        identity=replace(request.identity, configuration_payload=_freeze(payload)),
    )
    with pytest.raises(RunnerContractError, match="exactly one"):
        execute_direct_request(drifted, prepared, authority, {"magic": spy})
    with pytest.raises(RunnerContractError, match="exactly one"):
        execute_direct_request(
            request,
            prepared,
            replace(authority, configurations=authority.configurations + (row,)),
            {"magic": spy},
        )
    assert attempted == []


def test_execute_direct_request_revalidates_payload_after_attempt() -> None:
    request, _entry, prepared, _descriptor, _spec, _row, authority = _direct_case()

    def mutating_adapter(*_args, **kwargs):
        object.__setattr__(kwargs["config"], "diffusion_time", 11)
        return _completed_outcome(request)

    result = execute_direct_request(
        request,
        prepared,
        authority,
        {"magic": mutating_adapter},
    )

    assert result.run.status == "failed"
    assert result.run.reason == "comparator_payload_changed_during_adapter_attempt"
    assert all(metric.status == "failed" for metric in result.metrics)


def test_execute_direct_request_preserves_resources_audit_and_complete_metrics() -> (
    None
):
    request, _entry, prepared, _descriptor, _spec, _row, authority = _direct_case()

    result = execute_direct_request(
        request,
        prepared,
        authority,
        {"magic": lambda *_args, **_kwargs: _completed_outcome(request)},
    )

    assert result.run.runtime_seconds == 1.5
    assert result.run.peak_rss_bytes == 128
    assert result.run.peak_gpu_bytes == 0
    assert result.run.excluded_cell_ids == ()
    assert result.run.retained_cell_ids == ("cell-1", "cell-2")
    assert result.evaluator_output is not None
    expected = reconstruction_metrics(
        result.evaluator_output,
        count_equivalent_to_log2_cp10k(prepared.method_input.counts),
        count_equivalent_to_log2_cp10k(
            np.asarray(
                prepared.evaluator_dataset.layers["pre_capture_counts"],
                dtype=np.float64,
            )
        ),
        truth_kind="exact_pre_capture",
    )
    assert tuple(row.metric for row in result.metrics) == tuple(expected)
    assert tuple(
        (row.value, row.n, row.status, row.reason) for row in result.metrics
    ) == tuple(
        (
            None if metric.value is None else float(metric.value),
            metric.n,
            "unavailable" if metric.value is None else "completed",
            metric.reason,
        )
        for metric in expected.values()
    )


@pytest.mark.parametrize("substitute", (True, 1.0))
def test_execute_direct_request_rejects_numeric_type_coercion_before_dispatch(
    substitute: object,
) -> None:
    request, _entry, prepared, _descriptor, _spec, row, authority = _direct_case()
    payload = dict(row.payload)
    assert payload["n_jobs"] == 1 and type(payload["n_jobs"]) is int
    payload["n_jobs"] = substitute
    attempted: list[bool] = []

    def spy(*_args, **_kwargs):
        attempted.append(True)
        return _completed_outcome(request)

    with pytest.raises(RunnerContractError, match="exactly one"):
        execute_direct_request(
            replace(
                request,
                identity=replace(
                    request.identity,
                    configuration_payload=_freeze(payload),
                ),
            ),
            prepared,
            authority,
            {"magic": spy},
        )
    assert attempted == []


@pytest.mark.parametrize("substitute", (True, 1.0))
def test_execute_direct_request_rejects_numeric_type_coercion_after_dispatch(
    substitute: object,
) -> None:
    request, _entry, prepared, _descriptor, _spec, _row, authority = _direct_case()

    def mutating_adapter(*_args, **kwargs):
        object.__setattr__(kwargs["config"], "n_jobs", substitute)
        return _completed_outcome(request)

    result = execute_direct_request(
        request,
        prepared,
        authority,
        {"magic": mutating_adapter},
    )

    assert result.run.status == "failed"
    assert result.run.reason == "comparator_payload_changed_during_adapter_attempt"


@pytest.mark.parametrize(
    ("adapters", "status", "reason"),
    (
        ({}, "unavailable", "adapter_not_registered"),
        (
            {"magic": lambda *_args, **_kwargs: AdapterOutcome.timeout()},
            "timeout",
            "timeout",
        ),
        (
            {
                "magic": lambda *_args, **_kwargs: AdapterOutcome.resource_exceeded(
                    "peak_rss_exceeded"
                )
            },
            "resource_exceeded",
            "peak_rss_exceeded",
        ),
        (
            {
                "magic": lambda *_args, **_kwargs: AdapterOutcome.infrastructure_error(
                    "worker_protocol_error"
                )
            },
            "infrastructure_error",
            "worker_protocol_error",
        ),
        (
            {
                "magic": lambda *_args, **_kwargs: AdapterOutcome.blocked_authority(
                    "authority_blocked"
                )
            },
            "blocked_authority",
            "authority_blocked",
        ),
        (
            {
                "magic": lambda *_args, **_kwargs: AdapterOutcome.budget_exhausted(
                    "budget_spent"
                )
            },
            "budget_exhausted",
            "budget_spent",
        ),
    ),
)
def test_execute_direct_request_preserves_terminal_outcomes(
    adapters,
    status: str,
    reason: str,
) -> None:
    request, _entry, prepared, _descriptor, _spec, _row, authority = _direct_case()

    result = execute_direct_request(request, prepared, authority, adapters)

    assert result.run.status == status
    assert result.run.reason == reason
    assert result.run.stdout.terminal_reason == reason
    assert result.run.stderr.terminal_reason == reason
    assert all(
        metric.status == status and metric.reason == reason for metric in result.metrics
    )


def test_execute_direct_request_enforces_resource_limits_and_request_integrity() -> (
    None
):
    request, _entry, prepared, _descriptor, _spec, _row, authority = _direct_case()
    attempted: list[bool] = []

    def excessive(*_args, **_kwargs):
        attempted.append(True)
        completed = _completed_outcome(request)
        return replace(completed, peak_rss_bytes=request.max_rss_bytes + 1)

    result = execute_direct_request(request, prepared, authority, {"magic": excessive})
    assert result.run.status == "resource_exceeded"
    assert result.run.reason == "peak_rss_exceeded"

    with pytest.raises(RunnerContractError, match="resource limits"):
        execute_direct_request(
            replace(request, max_rss_bytes=request.max_rss_bytes - 1),
            prepared,
            authority,
            {"magic": excessive},
        )
    assert attempted == [True]


def test_direct_prezero_reopens_exact_regular_storage_and_rejects_drift(
    tmp_path: Path,
) -> None:
    matrix = np.asarray([[0.25, 0.75], [0.5, 1.0]], dtype="<f8")
    compressed = zlib.compress(matrix.tobytes(order="C"))
    path = tmp_path / "runs" / "p-pre-zero.zlib"
    path.parent.mkdir()
    path.write_bytes(compressed)
    evidence = DirectPreZeroEvidence(
        applicable=True,
        status="completed",
        reason=None,
        shape=matrix.shape,
        dtype="<f8",
        encoding="zlib",
        path="runs/p-pre-zero.zlib",
        compressed_byte_count=len(compressed),
    )

    np.testing.assert_array_equal(evidence.reopen(tmp_path), matrix)
    with pytest.raises(RunnerContractError, match="byte count"):
        replace(evidence, compressed_byte_count=len(compressed) + 1).reopen(tmp_path)
    path.write_bytes(b"not-zlib")
    with pytest.raises(RunnerContractError, match="invalid"):
        replace(evidence, compressed_byte_count=8).reopen(tmp_path)


def test_direct_prezero_rejects_symlinked_storage_ancestor(tmp_path: Path) -> None:
    matrix = np.asarray([[0.25]], dtype="<f8")
    compressed = zlib.compress(matrix.tobytes(order="C"))
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "value.zlib").write_bytes(compressed)
    (tmp_path / "linked").symlink_to(actual, target_is_directory=True)
    evidence = DirectPreZeroEvidence(
        applicable=True,
        status="completed",
        reason=None,
        shape=(1, 1),
        dtype="<f8",
        encoding="zlib",
        path="linked/value.zlib",
        compressed_byte_count=len(compressed),
    )

    with pytest.raises(RunnerContractError, match="owned"):
        evidence.reopen(tmp_path)


def test_direct_prezero_rejects_regular_file_owned_by_another_uid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.asarray([[0.25]], dtype="<f8")
    compressed = zlib.compress(matrix.tobytes(order="C"))
    path = tmp_path / "value.zlib"
    path.write_bytes(compressed)
    evidence = DirectPreZeroEvidence(
        applicable=True,
        status="completed",
        reason=None,
        shape=(1, 1),
        dtype="<f8",
        encoding="zlib",
        path="value.zlib",
        compressed_byte_count=len(compressed),
    )
    original_lstat = Path.lstat

    def foreign_owner(selected: Path):
        metadata = original_lstat(selected)
        if selected == path:
            return SimpleNamespace(st_mode=metadata.st_mode, st_uid=os.getuid() + 1)
        return metadata

    monkeypatch.setattr(Path, "lstat", foreign_owner)

    with pytest.raises(RunnerContractError, match="owner"):
        evidence.reopen(tmp_path)


def test_direct_spawned_executor_enforces_deadline_with_parent_telemetry() -> None:
    from maskimpute_benchmark import runner as runner_module

    request, _entry, _prepared_value, _descriptor, _spec, _row, _authority = (
        _direct_case()
    )
    execute = getattr(runner_module, "execute_direct_adapter_in_spawned_process", None)
    assert execute is not None, "direct measured executor is absent"

    outcome = execute(
        replace(request, timeout_seconds=0.05),
        _slow_direct_executor,
        poll_interval_seconds=0.01,
        resource_sampler=_DirectFixedResourceSampler(rss=123_456, gpu=0),
    )

    assert outcome.status == "timeout"
    assert outcome.runtime_seconds >= 0.05
    assert outcome.peak_rss_bytes == 123_456
    assert outcome.rss_measurement == "synthetic_parent_rss"


def test_direct_spawned_executor_fails_closed_without_required_gpu_telemetry() -> None:
    from maskimpute_benchmark import runner as runner_module

    request, _entry, _prepared_value, _descriptor, _spec, _row, _authority = (
        _direct_case("dca")
    )
    execute = getattr(runner_module, "execute_direct_adapter_in_spawned_process", None)
    assert execute is not None, "direct measured executor is absent"

    outcome = execute(
        request,
        _direct_terminal_executor,
        poll_interval_seconds=0.01,
        resource_sampler=_DirectFixedResourceSampler(rss=123_456, gpu=None),
    )

    assert outcome.status == "infrastructure_error"
    assert outcome.reason == "resource_telemetry_unavailable"


def test_direct_and_legacy_spawn_entry_points_reject_the_other_request_type() -> None:
    from maskimpute_benchmark import runner as runner_module

    request, _entry, _prepared_value, _descriptor, _spec, _row, _authority = (
        _direct_case()
    )

    with pytest.raises(TypeError, match="ExecutionRequest"):
        runner_module.execute_adapter_in_spawned_process(
            request,
            _direct_terminal_executor,
        )
    with pytest.raises(TypeError, match="DirectExecutionRequest"):
        runner_module.execute_direct_adapter_in_spawned_process(
            object(),
            _direct_terminal_executor,
        )


def test_direct_record_constructors_reject_unknown_fields() -> None:
    request, _entry, _prepared_value, _descriptor, _spec, _row, _authority = (
        _direct_case()
    )
    with pytest.raises(TypeError, match="unexpected"):
        DirectExecutionRequest(
            identity=request.identity,
            method_spec=request.method_spec,
            method_input=request.method_input,
            timeout_seconds=request.timeout_seconds,
            max_rss_bytes=request.max_rss_bytes,
            max_gpu_bytes=request.max_gpu_bytes,
            unexpected=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_seed", True, "model seed"),
        ("model_seed", 41, "model seed"),
    ),
)
def test_create_direct_request_rejects_invalid_seed_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    _request, entry, prepared, descriptor, spec, row, _authority = _direct_case()
    identity = replace(entry.identity, **{field: value})
    with pytest.raises(RunnerContractError, match=message):
        create_direct_request(
            replace(entry, identity=identity, run_id=direct_run_id(identity)),
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=5,
        )


def test_create_direct_request_rejects_boolean_timeout() -> None:
    _request, entry, prepared, descriptor, spec, row, _authority = _direct_case()
    with pytest.raises(RunnerContractError, match="timeout"):
        create_direct_request(
            entry,
            prepared,
            descriptor,
            spec,
            row,
            timeout_seconds=True,
        )


def test_direct_metric_and_prezero_statuses_are_closed() -> None:
    request, _entry, _prepared_value, _descriptor, _spec, _row, _authority = (
        _direct_case()
    )
    with pytest.raises(RunnerContractError, match="metric status"):
        DirectMetricRow(
            identity=request.identity,
            metric="mse",
            value=None,
            n=0,
            status="invented",
            reason="invented",
        )
    with pytest.raises(RunnerContractError, match="evidence status"):
        DirectPreZeroEvidence(
            applicable=True,
            status="invented",
            reason="invented",
            shape=None,
            dtype=None,
            encoding=None,
            path=None,
            compressed_byte_count=0,
        )


def test_direct_metric_rejects_signed_negative_zero() -> None:
    request, _entry, _prepared_value, _descriptor, _spec, _row, _authority = (
        _direct_case()
    )

    with pytest.raises(RunnerContractError, match="metric value"):
        DirectMetricRow(
            identity=request.identity,
            metric="signed_error",
            value=-0.0,
            n=1,
            status="completed",
            reason=None,
        )
