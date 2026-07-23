from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def test_registered_trajectory_authority_builds_bound_exact_latent_panel() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluator_targets_from_dataset,
    )
    from maskimpute_benchmark.schema import (
        benchmark_dataset_sha256,
        validate_benchmark_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        FOUR_RECONSTRUCTION_MECHANISMS,
        generate_registered_trajectory_dataset,
        load_trajectory_authority,
    )

    authority = load_trajectory_authority()
    first = generate_registered_trajectory_dataset(authority=authority)
    second = generate_registered_trajectory_dataset(authority=authority)

    validate_benchmark_dataset(first)
    assert first.shape == (2_700, 120)
    assert first.obs["mechanism"].unique().tolist() == ["synthetic_trajectory"]
    assert "synthetic_trajectory" not in FOUR_RECONSTRUCTION_MECHANISMS
    assert first.uns["truth_kind"] == "orthogonal_only"
    assert benchmark_dataset_sha256(first) == authority.expected_dataset_sha256
    assert benchmark_dataset_sha256(second) == authority.expected_dataset_sha256
    np.testing.assert_array_equal(first.X, second.X)

    pseudotime = np.asarray(first.obs["pseudotime"], dtype=np.float64)
    assert pseudotime[0] == 0.0
    assert pseudotime[-1] == 1.0
    assert np.all(np.diff(pseudotime) > 0.0)
    assert first.obs_names[0] == authority.root_cell_id
    targets = evaluator_targets_from_dataset(
        first,
        trajectory_root_cell_id=authority.root_cell_id,
        trajectory_source_id=authority.source_id,
    )
    assert targets.trajectory is not None
    assert targets.trajectory.source_id == authority.source_id
    assert targets.trajectory.root_cell_id == authority.root_cell_id


def test_registered_trajectory_truth_is_absent_from_method_view() -> None:
    from maskimpute_benchmark.schema import make_inference_view
    from maskimpute_benchmark.trajectory_dataset import (
        generate_registered_trajectory_dataset,
    )

    dataset = generate_registered_trajectory_dataset()
    method_view = make_inference_view(dataset)

    assert method_view.shape == dataset.shape
    assert tuple(method_view.obs.columns) == ()
    assert tuple(method_view.var.columns) == ()
    assert "pseudotime" not in method_view.obs
    assert "group" not in method_view.obs
    assert not method_view.layers
    assert set(method_view.uns) == {"normalization", "source_dataset_sha256"}


def test_trajectory_authority_rejects_binding_tampering(tmp_path) -> None:
    import json

    import pytest

    from maskimpute_benchmark.trajectory_dataset import (
        TrajectoryAuthorityError,
        default_trajectory_authority_path,
        load_trajectory_authority,
    )

    payload = json.loads(default_trajectory_authority_path().read_text())
    payload["root_cell_id"] = "cell-000002"
    path = tmp_path / "trajectory_panel.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(TrajectoryAuthorityError, match="authority_sha256"):
        load_trajectory_authority(path)


def test_trajectory_authority_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    import pytest

    from maskimpute_benchmark.trajectory_dataset import (
        TrajectoryAuthorityError,
        default_trajectory_authority_path,
        load_trajectory_authority,
    )

    text = default_trajectory_authority_path().read_text(encoding="utf-8")
    duplicated = text.replace(
        '"schema_version": "trajectory-panel-v1",',
        '"schema_version": "trajectory-panel-v1",\n'
        '  "schema_version": "trajectory-panel-v1",',
    )
    path = tmp_path / "trajectory_panel.json"
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(TrajectoryAuthorityError, match="duplicate JSON key"):
        load_trajectory_authority(path)


def test_registered_trajectory_preparation_persists_only_evaluator_targets(
    tmp_path: Path,
) -> None:
    from maskimpute_benchmark.final_runner import (
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.methods import load_method_registry
    from maskimpute_benchmark.runner import (
        AuthorizedConfiguration,
        ExecutionAuthorityContext,
        ExecutionRequest,
        method_input_sha256,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        REGISTERED_TRAJECTORY_DATASET_ID,
        RegisteredTrajectoryBinding,
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )

    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    resumed = materialize_prepared_trajectory_dataset(repository, round_dir)

    assert isinstance(registered.binding, RegisteredTrajectoryBinding)
    assert registered.binding.dataset_id == REGISTERED_TRAJECTORY_DATASET_ID
    assert registered.binding.dataset_sha256 == (
        registered.authority.expected_dataset_sha256
    )
    assert registered.binding.authority_sha256 == registered.authority.authority_sha256
    assert registered.binding.registered_binding_sha256 == (
        registered.authority.binding_sha256
    )
    assert registered.binding.dataset_file_path == (
        "results/trajectory/dataset/evaluator.h5ad"
    )
    assert registered.receipt["method_input_sha256"] == method_input_sha256(
        registered.prepared.method_input
    )
    assert resumed.receipt == registered.receipt
    assert resumed.binding == registered.binding

    method_input = registered.prepared.method_input
    assert method_input.obs_covariates == ()
    assert method_input.var_covariates == ()
    assert tuple(method_input.covariate_frame("obs").columns) == ()
    assert {
        "pseudotime",
        "group",
    }.issubset(registered.prepared.evaluator_dataset.obs.columns)

    observed = load_method_registry(Path("study/methods.json")).by_id("observed")
    configuration = AuthorizedConfiguration.registry_default(observed)
    context = ExecutionAuthorityContext(
        authority_sha256="1" * 64,
        base_configuration_json="{}",
        base_configuration_sha256="44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        count_model_config_json="{}",
        count_model_config_sha256="44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        count_score_manifest_path="authority/score.json",
        count_score_manifest_sha256="2" * 64,
        retained_calibration_path="authority/calibration.json",
        retained_calibration_sha256="3" * 64,
    )
    request = ExecutionRequest.create(
        observed,
        method_input,
        model_seed=None,
        configuration=configuration,
        authority=context,
        mechanism=registered.binding.mechanism,
        biological_id=registered.binding.biological_id,
        technical_view=registered.binding.technical_view,
        dataset_id=registered.binding.dataset_id,
        timeout_seconds=observed.resources.timeout_seconds,
        calibration_usage="retained_all_development",
    )
    serialized = pickle.dumps(request)
    for forbidden in (
        b"pseudotime",
        b"group",
        b"root_cell_id",
        b"trajectory_source_id",
        registered.authority.source_id.encode(),
    ):
        assert forbidden not in serialized


def test_trajectory_resume_rejects_replaced_h5ad_before_deserialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import anndata as ad
    import pytest

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    (repository / "study/trajectory_panel.json").write_bytes(
        default_trajectory_authority_path().read_bytes()
    )
    registered = materialize_prepared_trajectory_dataset(repository, round_dir)
    dataset_path = round_dir / registered.binding.dataset_file_path
    dataset_path.write_bytes(dataset_path.read_bytes() + b"coordinated replacement")
    calls = 0

    def forbidden_read(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("replaced H5AD must not be deserialized")

    monkeypatch.setattr(ad, "read_h5ad", forbidden_read)

    with pytest.raises(FinalRunnerContractError, match="file checksum"):
        materialize_prepared_trajectory_dataset(repository, round_dir)

    assert calls == 0


def test_trajectory_materialization_requires_exact_authority_bytes(
    tmp_path: Path,
) -> None:
    import json

    import pytest

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        materialize_prepared_trajectory_dataset,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        default_trajectory_authority_path,
    )

    repository = tmp_path / "repository"
    round_dir = repository / "artifacts/study/final/round-001"
    (repository / "study").mkdir(parents=True)
    round_dir.mkdir(parents=True)
    payload = json.loads(default_trajectory_authority_path().read_text())
    (repository / "study/trajectory_panel.json").write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FinalRunnerContractError, match="authority.*canonical"):
        materialize_prepared_trajectory_dataset(repository, round_dir)


def test_trajectory_h5ad_structure_rejects_external_links(tmp_path: Path) -> None:
    import h5py
    import pytest

    from maskimpute_benchmark.final_runner import (
        FinalRunnerContractError,
        _validate_trajectory_h5ad_structure,
    )
    from maskimpute_benchmark.trajectory_dataset import (
        generate_registered_trajectory_dataset,
    )

    path = tmp_path / "trajectory.h5ad"
    generate_registered_trajectory_dataset().write_h5ad(path)
    _validate_trajectory_h5ad_structure(path)

    with h5py.File(path, "r+") as handle:
        del handle["X"]
        handle["X"] = h5py.ExternalLink("outside.h5", "/X")

    with pytest.raises(FinalRunnerContractError, match="internal hard link"):
        _validate_trajectory_h5ad_structure(path)
