from __future__ import annotations

from dataclasses import fields

import anndata as ad
import numpy as np
import pandas as pd
import pytest


def test_method_output_boundary_has_no_evaluator_truth_fields() -> None:
    from maskimpute_benchmark.downstream_evaluation import MethodOutput

    output = MethodOutput(
        values=np.asarray([[1.0, 0.0], [0.0, 2.0]]),
        cell_ids=("cell-b", "cell-a"),
        gene_ids=("gene-2", "gene-1"),
    )

    assert tuple(field.name for field in fields(MethodOutput)) == (
        "values",
        "cell_ids",
        "gene_ids",
    )
    np.testing.assert_array_equal(output.values, [[1.0, 0.0], [0.0, 2.0]])
    assert not output.values.flags.writeable


@pytest.mark.parametrize(
    ("values", "cell_ids", "gene_ids", "message"),
    [
        (np.ones(2), ("c1", "c2"), ("g1",), "two-dimensional"),
        (np.ones((2, 2)), ("c1",), ("g1", "g2"), "cell_ids"),
        (np.ones((2, 2)), ("c1", "c2"), ("g1",), "gene_ids"),
        (np.asarray([[1.0, np.nan]]), ("c1",), ("g1", "g2"), "finite"),
        (np.asarray([[1.0, -1.0]]), ("c1",), ("g1", "g2"), "nonnegative"),
        (np.ones((2, 1)), ("c1", "c1"), ("g1",), "unique"),
        (np.ones((1, 2)), ("c1",), ("g1", ""), "nonempty"),
    ],
)
def test_method_output_rejects_malformed_truth_free_values(
    values: np.ndarray,
    cell_ids: tuple[str, ...],
    gene_ids: tuple[str, ...],
    message: str,
) -> None:
    from maskimpute_benchmark.downstream_evaluation import MethodOutput

    with pytest.raises((TypeError, ValueError), match=message):
        MethodOutput(values=values, cell_ids=cell_ids, gene_ids=gene_ids)


def _simulator_dataset(
    mechanism: str,
    groups: tuple[str, ...],
    marker_columns: dict[str, tuple[bool, ...]],
    *,
    heldout: np.ndarray | None = None,
    pseudotime: np.ndarray | None = None,
) -> ad.AnnData:
    counts = np.asarray([[5, 0, 1], [4, 1, 0], [0, 5, 1], [1, 4, 0]], dtype=np.int64)
    obs = pd.DataFrame(
        {"mechanism": [mechanism] * 4, "group": list(groups)},
        index=("cell-4", "cell-2", "cell-3", "cell-1"),
    )
    if pseudotime is not None:
        obs["pseudotime"] = pseudotime
    dataset = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(marker_columns, index=("gene-c", "gene-a", "gene-b")),
    )
    if heldout is not None:
        dataset.layers["heldout_counts"] = heldout
    return dataset


@pytest.mark.parametrize(
    ("mechanism", "groups", "columns", "expected"),
    [
        (
            "symsim",
            ("pop-1", "pop-1", "pop-2", "pop-2"),
            {
                "marker_group_1": (True, False, False),
                "marker_group_2": (False, True, False),
            },
            {"pop-1": [True, False, False], "pop-2": [False, True, False]},
        ),
        (
            "sergio",
            ("cell-type-1", "cell-type-1", "cell-type-2", "cell-type-2"),
            {
                "marker_cell_type_1": (True, False, False),
                "marker_cell_type_2": (False, True, False),
            },
            {"cell-type-1": [True, False, False], "cell-type-2": [False, True, False]},
        ),
        (
            "sparsim",
            ("chu-c1", "chu-c1", "chu-c3", "chu-c3"),
            {
                "marker_chu_c1": (True, False, False),
                "marker_chu_c3": (False, True, False),
            },
            {"chu-c1": [True, False, False], "chu-c3": [False, True, False]},
        ),
    ],
)
def test_simulator_adapter_extracts_group_specific_markers_by_explicit_schema(
    mechanism: str,
    groups: tuple[str, ...],
    columns: dict[str, tuple[bool, ...]],
    expected: dict[str, list[bool]],
) -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluator_targets_from_dataset,
    )

    targets = evaluator_targets_from_dataset(
        _simulator_dataset(mechanism, groups, columns)
    )

    assert targets.group_labels == groups
    assert targets.group_labels_reason is None
    assert targets.group_markers_reason is None
    assert targets.group_markers is not None
    assert {
        group: mask.tolist() for group, mask in targets.group_markers.items()
    } == expected
    assert targets.heldout_counts is None
    assert targets.heldout_reason == "independent_heldout_counts_unavailable"
    assert targets.trajectory is None
    assert (
        targets.trajectory_reason
        == "genuine_pseudotime_not_available_in_simulator_output"
    )


def test_semisynthetic_adapter_keeps_heldout_counts_and_reports_missing_markers() -> (
    None
):
    from maskimpute_benchmark.downstream_evaluation import (
        evaluator_targets_from_dataset,
    )

    heldout = np.asarray([[2, 0, 1], [1, 1, 0], [0, 3, 0], [1, 2, 1]])
    dataset = _simulator_dataset(
        "semisynthetic",
        ("alpha", "alpha", "beta", "beta"),
        {},
        heldout=heldout,
    )
    targets = evaluator_targets_from_dataset(dataset)

    assert targets.group_markers is None
    assert targets.group_markers_reason == "group_specific_marker_truth_unavailable"
    np.testing.assert_array_equal(targets.heldout_counts, heldout)
    assert not targets.heldout_counts.flags.writeable
    assert targets.heldout_reason is None


def test_trajectory_adapter_requires_genuine_values_and_evaluator_known_root() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluator_targets_from_dataset,
    )

    dataset = _simulator_dataset(
        "semisynthetic",
        ("alpha", "alpha", "beta", "beta"),
        {},
        pseudotime=np.asarray([1.0, 0.35, 0.7, 0.0]),
    )
    missing_root = evaluator_targets_from_dataset(dataset)
    assert missing_root.trajectory is None
    assert missing_root.trajectory_reason == "trajectory_root_not_prespecified"

    targets = evaluator_targets_from_dataset(
        dataset,
        trajectory_root_cell_id="cell-1",
        trajectory_source_id="genuine-linear-fixture",
    )
    assert targets.trajectory is not None
    assert targets.trajectory.root_cell_id == "cell-1"
    assert targets.trajectory.source_id == "genuine-linear-fixture"

    with pytest.raises(ValueError, match="unique minimum"):
        evaluator_targets_from_dataset(
            dataset,
            trajectory_root_cell_id="cell-2",
            trajectory_source_id="genuine-linear-fixture",
        )


def _marker_de_fixture():
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluator_targets_from_dataset,
    )

    cell_ids = ("cell-6", "cell-2", "cell-5", "cell-1", "cell-4", "cell-3")
    gene_ids = ("gene-4", "gene-2", "gene-1", "gene-3")
    groups = ("pop-1", "pop-1", "pop-1", "pop-2", "pop-2", "pop-2")
    values = np.asarray(
        [
            [0, 8, 10, 0],
            [0, 8, 10, 0],
            [0, 8, 10, 0],
            [8, 0, 0, 10],
            [8, 0, 0, 10],
            [8, 0, 0, 10],
        ],
        dtype=float,
    )
    dataset = ad.AnnData(
        X=np.zeros_like(values, dtype=np.int64),
        obs=pd.DataFrame(
            {"mechanism": ["symsim"] * 6, "group": groups}, index=cell_ids
        ),
        var=pd.DataFrame(
            {
                "marker_group_1": [False, True, False, False],
                "marker_group_2": [True, False, False, False],
            },
            index=gene_ids,
        ),
    )
    return (
        MethodOutput(values=values, cell_ids=cell_ids, gene_ids=gene_ids),
        evaluator_targets_from_dataset(dataset),
    )


def test_marker_rank_and_positive_de_are_hand_calculated_with_one_global_bh_family() -> (
    None
):
    from maskimpute_benchmark.downstream_evaluation import (
        evaluate_marker_and_de_endpoints,
    )

    output, targets = _marker_de_fixture()
    records = {
        record.endpoint: record
        for record in evaluate_marker_and_de_endpoints(output, targets)
    }

    assert records["marker_rank_loss"].value == pytest.approx(1.0 / 3.0)
    assert records["positive_de_marker_recall"].value == 1.0
    assert records["positive_de_false_discovery_rate"].value == 0.5
    for endpoint in (
        "positive_de_marker_recall",
        "positive_de_false_discovery_rate",
    ):
        record = records[endpoint]
        assert record.family_id == "one_vs_rest_all_groups_all_genes"
        assert record.family_size == 8
        assert record.alpha == 0.05
        assert record.independent_unit == "biological_draw"
        assert record.independent_n == 1
    assert records["positive_de_marker_recall"].descriptive_n == 2
    assert records["positive_de_marker_recall"].descriptive_unit == "truth_markers"
    assert records["positive_de_false_discovery_rate"].descriptive_n == 4
    assert records["positive_de_false_discovery_rate"].descriptive_unit == "discoveries"


def test_marker_and_de_endpoints_are_invariant_to_method_row_and_gene_order() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluate_marker_and_de_endpoints,
    )

    output, targets = _marker_de_fixture()
    baseline = evaluate_marker_and_de_endpoints(output, targets)
    rows = np.asarray([3, 0, 5, 2, 1, 4])
    columns = np.asarray([2, 0, 3, 1])
    permuted = MethodOutput(
        values=output.values[rows][:, columns],
        cell_ids=tuple(output.cell_ids[index] for index in rows),
        gene_ids=tuple(output.gene_ids[index] for index in columns),
    )

    repeated = evaluate_marker_and_de_endpoints(permuted, targets)

    assert [record.endpoint for record in repeated] == [
        record.endpoint for record in baseline
    ]
    for first, second in zip(baseline, repeated, strict=True):
        assert second.status == first.status
        assert second.reason == first.reason
        assert second.value == pytest.approx(first.value)


def test_marker_and_de_emit_complete_reasons_when_marker_truth_is_unavailable() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluate_marker_and_de_endpoints,
        evaluator_targets_from_dataset,
    )

    dataset = _simulator_dataset(
        "semisynthetic", ("alpha", "alpha", "beta", "beta"), {}
    )
    output = MethodOutput(
        values=np.asarray(dataset.X, dtype=float),
        cell_ids=tuple(dataset.obs_names),
        gene_ids=tuple(dataset.var_names),
    )
    records = evaluate_marker_and_de_endpoints(
        output, evaluator_targets_from_dataset(dataset)
    )

    assert [record.endpoint for record in records] == [
        "marker_rank_loss",
        "positive_de_marker_recall",
        "positive_de_false_discovery_rate",
    ]
    assert all(record.value is None for record in records)
    assert all(record.status == "unavailable" for record in records)
    assert {record.reason for record in records} == {
        "group_specific_marker_truth_unavailable"
    }
    assert all(record.independent_n == 1 for record in records)


def _clustering_fixture(groups: tuple[str, ...], values: np.ndarray):
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluator_targets_from_dataset,
    )

    cell_ids = tuple(f"cell-{index:02d}" for index in range(len(groups), 0, -1))
    gene_ids = tuple(f"gene-{index:02d}" for index in range(values.shape[1], 0, -1))
    dataset = ad.AnnData(
        X=np.zeros_like(values, dtype=np.int64),
        obs=pd.DataFrame(
            {"mechanism": ["orthogonal"] * len(groups), "group": groups},
            index=cell_ids,
        ),
        var=pd.DataFrame(index=gene_ids),
    )
    return (
        MethodOutput(values=values, cell_ids=cell_ids, gene_ids=gene_ids),
        evaluator_targets_from_dataset(dataset),
        dataset,
    )


def test_clustering_losses_match_hand_calculated_crossed_partition() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        CLUSTERING_SEED,
        evaluate_clustering_endpoints,
    )

    output, targets, _ = _clustering_fixture(
        ("a", "a", "b", "b"),
        np.asarray([[10, 0], [0, 10], [10, 0], [0, 10]], dtype=float),
    )
    ari, nmi = evaluate_clustering_endpoints(output, targets)

    # The inferred partition is {1,3}/{2,4}; against {1,2}/{3,4},
    # ARI=-1/2 and arithmetic-normalized mutual information is zero.
    assert ari.endpoint == "clustering_ari_loss"
    assert ari.value == pytest.approx(1.5)
    assert nmi.endpoint == "clustering_nmi_loss"
    assert nmi.value == pytest.approx(1.0)
    assert CLUSTERING_SEED == 20_260_716
    assert "seed=20260716" in ari.procedure
    assert ari.independent_unit == "biological_draw"
    assert ari.independent_n == 1
    assert ari.descriptive_n == 4
    assert ari.descriptive_unit == "cells"


def test_clustering_is_deterministic_and_permutation_invariant() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluate_clustering_endpoints,
        evaluator_targets_from_dataset,
    )

    groups = ("a",) * 4 + ("b",) * 4 + ("c",) * 4
    values = np.asarray(
        [[20, 18, 0, 0, 0, 0]] * 4
        + [[0, 0, 20, 18, 0, 0]] * 4
        + [[0, 0, 0, 0, 20, 18]] * 4,
        dtype=float,
    )
    output, targets, dataset = _clustering_fixture(groups, values)
    first = evaluate_clustering_endpoints(output, targets)
    second = evaluate_clustering_endpoints(output, targets)
    assert first == second
    assert [record.value for record in first] == pytest.approx([0.0, 0.0])

    rows = np.asarray([11, 0, 7, 2, 9, 5, 1, 8, 4, 10, 3, 6])
    columns = np.asarray([5, 1, 3, 0, 4, 2])
    permuted = MethodOutput(
        values=output.values[rows][:, columns],
        cell_ids=tuple(output.cell_ids[index] for index in rows),
        gene_ids=tuple(output.gene_ids[index] for index in columns),
    )
    assert evaluate_clustering_endpoints(permuted, targets) == first

    relabeled = dataset.copy()
    relabeled.obs["group"] = relabeled.obs["group"].map(
        {"a": "third", "b": "first", "c": "second"}
    )
    assert (
        evaluate_clustering_endpoints(output, evaluator_targets_from_dataset(relabeled))
        == first
    )


@pytest.mark.parametrize(
    ("groups", "values", "reason"),
    [
        (("a", "a", "a", "a"), np.eye(4), "fewer_than_two_groups"),
        (
            ("a", "a", "b", "b"),
            np.ones((4, 3)),
            "constant_method_representation",
        ),
        (
            ("a",) * 5 + ("b",) * 5,
            np.ones((10, 3)),
            "constant_method_representation",
        ),
        (
            ("a", "b", "b"),
            np.eye(3),
            "fewer_than_two_cells_in_group",
        ),
    ],
)
def test_clustering_degenerate_inputs_have_fixed_reasons(
    groups: tuple[str, ...], values: np.ndarray, reason: str
) -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluate_clustering_endpoints,
    )

    output, targets, _ = _clustering_fixture(groups, values)
    records = evaluate_clustering_endpoints(output, targets)

    assert all(record.status == "unavailable" for record in records)
    assert {record.reason for record in records} == {reason}


def _heldout_fixture(values: np.ndarray, heldout: np.ndarray | None):
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluator_targets_from_dataset,
    )

    cell_ids = tuple(f"cell-{index}" for index in range(values.shape[0], 0, -1))
    gene_ids = tuple(f"gene-{index}" for index in range(values.shape[1], 0, -1))
    dataset = ad.AnnData(
        X=np.zeros_like(values, dtype=np.int64),
        obs=pd.DataFrame(
            {
                "mechanism": ["semisynthetic"] * values.shape[0],
                "group": [
                    "a" if index < values.shape[0] // 2 else "b"
                    for index in range(values.shape[0])
                ],
            },
            index=cell_ids,
        ),
        var=pd.DataFrame(index=gene_ids),
    )
    if heldout is not None:
        dataset.layers["heldout_counts"] = heldout
    return (
        MethodOutput(values=values, cell_ids=cell_ids, gene_ids=gene_ids),
        evaluator_targets_from_dataset(dataset),
    )


def test_heldout_profile_rank_losses_are_hand_calculated() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluate_heldout_endpoints,
    )

    heldout = np.asarray([[1, 3], [2, 2], [3, 1]], dtype=float)
    output, targets = _heldout_fixture(heldout[::-1].copy(), heldout)
    gene, cell = evaluate_heldout_endpoints(output, targets)

    # Both heldout-variable gene profiles are perfectly reversed across cells.
    assert gene.endpoint == "heldout_gene_profile_rank_loss"
    assert gene.value == pytest.approx(2.0)
    assert gene.descriptive_n == 2
    assert gene.descriptive_unit == "heldout_variable_genes"
    # The first and third cells are variable across genes and perfectly reversed;
    # the tied middle cell is excluded based only on the heldout split.
    assert cell.endpoint == "heldout_cell_profile_rank_loss"
    assert cell.value == pytest.approx(2.0)
    assert cell.descriptive_n == 2
    assert cell.descriptive_unit == "heldout_variable_cells"
    assert gene.independent_n == cell.independent_n == 1


def test_heldout_profile_losses_are_permutation_invariant_and_deterministic() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluate_heldout_endpoints,
    )

    heldout = np.asarray([[1, 4, 2], [2, 1, 4], [4, 2, 1], [3, 3, 1]], dtype=float)
    output, targets = _heldout_fixture(heldout.copy(), heldout)
    baseline = evaluate_heldout_endpoints(output, targets)
    assert [record.value for record in baseline] == pytest.approx([0.0, 0.0])
    assert evaluate_heldout_endpoints(output, targets) == baseline

    rows = np.asarray([2, 0, 3, 1])
    columns = np.asarray([1, 2, 0])
    permuted = MethodOutput(
        values=output.values[rows][:, columns],
        cell_ids=tuple(output.cell_ids[index] for index in rows),
        gene_ids=tuple(output.gene_ids[index] for index in columns),
    )
    assert evaluate_heldout_endpoints(permuted, targets) == baseline


def test_heldout_method_collapse_is_penalized_instead_of_changing_denominator() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluate_heldout_endpoints,
    )

    heldout = np.asarray([[1, 3], [2, 2], [3, 1]], dtype=float)
    output, targets = _heldout_fixture(np.ones_like(heldout), heldout)
    gene, cell = evaluate_heldout_endpoints(output, targets)

    assert gene.value == pytest.approx(1.0)
    assert gene.descriptive_n == 2
    assert cell.value == pytest.approx(1.0)
    assert cell.descriptive_n == 2


def test_heldout_endpoints_have_fixed_missing_and_constant_reasons() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluate_heldout_endpoints,
    )

    missing_output, missing_targets = _heldout_fixture(np.eye(3), None)
    missing = evaluate_heldout_endpoints(missing_output, missing_targets)
    assert {record.reason for record in missing} == {
        "independent_heldout_counts_unavailable"
    }

    constant_output, constant_targets = _heldout_fixture(np.eye(3), np.ones((3, 3)))
    gene, cell = evaluate_heldout_endpoints(constant_output, constant_targets)
    assert gene.reason == "heldout_has_no_variable_gene_profiles"
    assert cell.reason == "heldout_has_no_variable_cell_profiles"


def _trajectory_fixture(values: np.ndarray, pseudotime: np.ndarray):
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluator_targets_from_dataset,
    )

    cell_ids = tuple(f"cell-{index:02d}" for index in range(len(pseudotime), 0, -1))
    gene_ids = tuple(f"gene-{index:02d}" for index in range(values.shape[1], 0, -1))
    dataset = ad.AnnData(
        X=np.zeros_like(values, dtype=np.int64),
        obs=pd.DataFrame(
            {
                "mechanism": ["trajectory-reference"] * len(pseudotime),
                "group": ["not-used"] * len(pseudotime),
                "pseudotime": pseudotime,
            },
            index=cell_ids,
        ),
        var=pd.DataFrame(index=gene_ids),
    )
    targets = evaluator_targets_from_dataset(
        dataset,
        trajectory_root_cell_id=cell_ids[0],
        trajectory_source_id="genuine-linear-trajectory-fixture",
    )
    return (
        MethodOutput(values=values, cell_ids=cell_ids, gene_ids=gene_ids),
        targets,
    )


def test_diffusion_trajectory_rank_loss_recovers_genuine_linear_order() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluate_trajectory_endpoint,
    )

    pseudotime = np.arange(20, dtype=float)
    values = np.column_stack([20.0 - pseudotime, pseudotime + 1.0, np.full(20, 2.0)])
    output, targets = _trajectory_fixture(values, pseudotime)
    record = evaluate_trajectory_endpoint(output, targets)

    assert record.endpoint == "trajectory_pseudotime_rank_loss"
    assert record.status == "completed"
    assert record.value == pytest.approx(0.0)
    assert record.descriptive_n == 20
    assert record.descriptive_unit == "trajectory_cells"
    assert record.independent_n == 1
    assert "multiscale_diffusion" in record.procedure


def test_diffusion_trajectory_is_deterministic_and_id_permutation_invariant() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluate_trajectory_endpoint,
    )

    pseudotime = np.arange(20, dtype=float)
    values = np.column_stack([20.0 - pseudotime, pseudotime + 1.0, np.full(20, 2.0)])
    output, targets = _trajectory_fixture(values, pseudotime)
    baseline = evaluate_trajectory_endpoint(output, targets)
    assert evaluate_trajectory_endpoint(output, targets) == baseline

    rows = np.asarray(
        [19, 0, 10, 4, 15, 2, 17, 6, 12, 8, 1, 18, 3, 16, 5, 14, 7, 13, 9, 11]
    )
    columns = np.asarray([2, 0, 1])
    permuted = MethodOutput(
        values=output.values[rows][:, columns],
        cell_ids=tuple(output.cell_ids[index] for index in rows),
        gene_ids=tuple(output.gene_ids[index] for index in columns),
    )
    assert evaluate_trajectory_endpoint(permuted, targets) == baseline


def test_diffusion_trajectory_uses_bounded_sparse_eigensolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        evaluate_trajectory_endpoint,
    )

    pseudotime = np.arange(40, dtype=float)
    values = np.column_stack([40.0 - pseudotime, pseudotime + 1.0, np.full(40, 2.0)])
    output, targets = _trajectory_fixture(values, pseudotime)

    def reject_dense_eigendecomposition(*_args, **_kwargs):
        raise AssertionError("trajectory evaluation must not use dense eigh")

    monkeypatch.setattr(np.linalg, "eigh", reject_dense_eigendecomposition)
    record = evaluate_trajectory_endpoint(output, targets)

    assert record.status == "completed"
    assert record.value == pytest.approx(0.0)


def test_trajectory_endpoint_reports_missing_constant_and_disconnected_graph() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        MethodOutput,
        evaluate_trajectory_endpoint,
        evaluator_targets_from_dataset,
    )

    no_truth_dataset = _simulator_dataset(
        "symsim",
        ("pop-1", "pop-1", "pop-2", "pop-2"),
        {
            "marker_group_1": (True, False, False),
            "marker_group_2": (False, True, False),
        },
    )
    no_truth_output = MethodOutput(
        values=np.asarray(no_truth_dataset.X, dtype=float),
        cell_ids=tuple(no_truth_dataset.obs_names),
        gene_ids=tuple(no_truth_dataset.var_names),
    )
    missing = evaluate_trajectory_endpoint(
        no_truth_output, evaluator_targets_from_dataset(no_truth_dataset)
    )
    assert missing.reason == "genuine_pseudotime_not_available_in_simulator_output"

    pseudotime = np.arange(10, dtype=float)
    constant_output, constant_targets = _trajectory_fixture(
        np.ones((10, 3)), pseudotime
    )
    constant = evaluate_trajectory_endpoint(constant_output, constant_targets)
    assert constant.reason == "constant_method_representation"

    first = np.column_stack(
        [30.0 - pseudotime[:5], pseudotime[:5] + 1.0, np.zeros(5), np.ones(5)]
    )
    second_time = pseudotime[:5]
    second = np.column_stack(
        [np.zeros(5), np.ones(5), 30.0 - second_time, second_time + 1.0]
    )
    disconnected_output, disconnected_targets = _trajectory_fixture(
        np.vstack([first, second]), pseudotime
    )
    disconnected = evaluate_trajectory_endpoint(
        disconnected_output, disconnected_targets
    )
    assert disconnected.reason == "trajectory_graph_disconnected"


def test_complete_evaluator_always_returns_fixed_eight_row_schema() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        DOWNSTREAM_ENDPOINT_NAMES,
        MethodOutput,
        evaluate_downstream_endpoints,
        evaluator_targets_from_dataset,
    )

    values = np.asarray([[8, 1, 0], [7, 2, 0], [0, 8, 1], [0, 7, 2]], dtype=float)
    heldout = np.asarray([[4, 1, 0], [3, 2, 0], [0, 4, 1], [0, 3, 2]], dtype=float)
    dataset = _simulator_dataset(
        "semisynthetic",
        ("alpha", "alpha", "beta", "beta"),
        {},
        heldout=heldout,
    )
    output = MethodOutput(
        values=values,
        cell_ids=tuple(dataset.obs_names),
        gene_ids=tuple(dataset.var_names),
    )
    records = evaluate_downstream_endpoints(
        output, evaluator_targets_from_dataset(dataset)
    )

    assert tuple(record.endpoint for record in records) == DOWNSTREAM_ENDPOINT_NAMES
    assert DOWNSTREAM_ENDPOINT_NAMES == (
        "marker_rank_loss",
        "clustering_ari_loss",
        "clustering_nmi_loss",
        "positive_de_marker_recall",
        "positive_de_false_discovery_rate",
        "heldout_gene_profile_rank_loss",
        "heldout_cell_profile_rank_loss",
        "trajectory_pseudotime_rank_loss",
    )
    assert len(records) == 8
    assert all(record.independent_unit == "biological_draw" for record in records)
    assert all(record.independent_n == 1 for record in records)
    assert all(
        record.descriptive_unit != "independent_replicates" for record in records
    )
    assert all(
        (record.value is not None and record.reason is None)
        or (record.value is None and record.reason is not None)
        for record in records
    )
    by_name = {record.endpoint: record for record in records}
    assert (
        by_name["marker_rank_loss"].reason == "group_specific_marker_truth_unavailable"
    )
    assert (
        by_name["positive_de_marker_recall"].reason
        == "group_specific_marker_truth_unavailable"
    )
    assert by_name["trajectory_pseudotime_rank_loss"].reason == (
        "genuine_pseudotime_not_available_in_simulator_output"
    )
    assert by_name["heldout_gene_profile_rank_loss"].status == "completed"
    assert by_name["heldout_cell_profile_rank_loss"].status == "completed"


def test_endpoint_schema_rejects_ad_hoc_names_directions_units_and_reasons() -> None:
    from maskimpute_benchmark.downstream_evaluation import (
        ENDPOINT_REASON_CODES,
        EndpointRecord,
    )

    assert "genuine_pseudotime_not_available_in_simulator_output" in (
        ENDPOINT_REASON_CODES
    )
    valid = {
        "endpoint": "marker_rank_loss",
        "value": None,
        "status": "unavailable",
        "reason": "group_specific_marker_truth_unavailable",
        "direction": "lower_is_better",
        "independent_unit": "biological_draw",
        "independent_n": 1,
        "descriptive_n": 0,
        "descriptive_unit": "truth_markers",
        "procedure": "fixed-test-procedure",
    }
    for field, invalid, message in (
        ("endpoint", "invented_endpoint", "fixed downstream schema"),
        ("direction", "higher_is_better", "direction"),
        ("descriptive_unit", "cells", "descriptive unit"),
        ("reason", "invented_reason", "reason code"),
    ):
        with pytest.raises(ValueError, match=message):
            EndpointRecord(**{**valid, field: invalid})
