from __future__ import annotations

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
