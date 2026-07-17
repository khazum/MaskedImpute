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
    counts = np.asarray(
        [[5, 0, 1], [4, 1, 0], [0, 5, 1], [1, 4, 0]], dtype=np.int64
    )
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
            {"marker_group_1": (True, False, False), "marker_group_2": (False, True, False)},
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
            {"marker_chu_c1": (True, False, False), "marker_chu_c3": (False, True, False)},
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


def test_semisynthetic_adapter_keeps_heldout_counts_and_reports_missing_markers() -> None:
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
