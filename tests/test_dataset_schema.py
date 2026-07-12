from __future__ import annotations

from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from maskimpute_benchmark.schema import (
    TruthKind,
    benchmark_dataset_sha256,
    make_inference_view,
    validate_benchmark_dataset,
)


OBSERVED = np.array(
    [
        [2, 0, 1, 0],
        [0, 3, 0, 1],
        [4, 0, 2, 0],
        [0, 1, 0, 5],
        [1, 0, 0, 2],
        [0, 2, 3, 0],
    ],
    dtype=np.int64,
)


def _provenance() -> dict[str, object]:
    return {
        "source": "https://example.invalid/source-v1.tar.gz",
        "source_sha256": "a" * 64,
        "software": "example-simulator",
        "software_version": "1.2.3+abc123",
        "parameters": {"capture_efficiency": 0.25, "genes": 4},
        "seeds": {"biological": 101, "measurement": 202},
    }


def _dataset(
    truth_kind: TruthKind = TruthKind.EXACT_PRE_CAPTURE,
    *,
    primary_truth_layer: str | None = None,
) -> ad.AnnData:
    obs = pd.DataFrame(
        {
            "dataset_id": ["dataset-001"] * 6,
            "mechanism": ["symsim"] * 6,
            "condition": ["moderate"] * 6,
            "biological_id": ["bio-1"] * 6,
            "technical_view": ["capture-0.25"] * 6,
            "draw": [1] * 6,
            "library_size": OBSERVED.sum(axis=1),
            "group": ["A", "A", "A", "B", "B", "B"],
            "label": ["type-1", "type-1", "type-2", "type-2", "rare", "rare"],
            "pseudotime": np.linspace(0, 1, 6),
            "batch": ["b1", "b1", "b2", "b2", "b3", "b3"],
            "donor_age": [30, 30, 44, 44, 52, 52],
        },
        index=[f"cell-{i}" for i in range(6)],
    )
    var = pd.DataFrame(
        {
            "is_marker": [True, False, True, False],
            "marker_score": [1.0, 0.1, 0.8, 0.0],
            "gc_content": [0.41, 0.55, 0.49, 0.62],
        },
        index=[f"gene-{i}" for i in range(4)],
    )
    dataset = ad.AnnData(X=OBSERVED.copy(), obs=obs, var=var)

    if truth_kind is TruthKind.EXACT_PRE_CAPTURE:
        layer = primary_truth_layer or "pre_capture_counts"
        dataset.layers[layer] = OBSERVED + np.array(
            [[0, 2, 0, 0], [1, 0, 2, 0], [0, 1, 0, 3], [2, 0, 1, 0], [0, 1, 4, 0], [1, 0, 0, 2]]
        )
    elif truth_kind is TruthKind.EXACT_CONTINUOUS:
        layer = primary_truth_layer or "latent_expression"
        dataset.layers[layer] = OBSERVED.astype(float) + 0.125
    elif truth_kind is TruthKind.PROXY_HIGH_DEPTH:
        layer = primary_truth_layer or "reference_counts"
        dataset.layers[layer] = OBSERVED * 4 + 1
    else:
        layer = None

    dataset.uns["truth_kind"] = truth_kind.value
    if layer is not None:
        dataset.uns["primary_truth_layer"] = layer
    dataset.uns["provenance"] = _provenance()
    dataset.uns["normalization"] = {"input": "raw_umi_counts", "size_factor": "none"}
    dataset.uns["allowed_covariates"] = {
        "obs": ["batch", "donor_age"],
        "var": ["gc_content"],
    }
    return dataset


@pytest.mark.parametrize(
    ("truth_kind", "layer"),
    [
        (TruthKind.EXACT_PRE_CAPTURE, "pre_capture_counts"),
        (TruthKind.EXACT_CONTINUOUS, "latent_expression"),
        (TruthKind.EXACT_CONTINUOUS, "pre_dropout_expression"),
        (TruthKind.PROXY_HIGH_DEPTH, "reference_counts"),
        (TruthKind.ORTHOGONAL_ONLY, None),
    ],
)
def test_all_truth_contracts_validate(truth_kind: TruthKind, layer: str | None) -> None:
    dataset = _dataset(truth_kind, primary_truth_layer=layer)

    assert validate_benchmark_dataset(dataset) is None


def test_truth_kind_values_are_publication_contract_terms() -> None:
    assert {kind.value for kind in TruthKind} == {
        "exact_pre_capture",
        "exact_continuous",
        "proxy_high_depth",
        "orthogonal_only",
    }


@pytest.mark.parametrize("value", [-1.0, 1.5, np.nan, np.inf])
def test_observed_counts_must_be_finite_nonnegative_integers(value: float) -> None:
    dataset = _dataset()
    observed = dataset.X.astype(float)
    observed[0, 0] = value
    dataset.X = observed

    with pytest.raises(ValueError, match="observed counts.*finite nonnegative integers"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize("truth_kind", [TruthKind.EXACT_PRE_CAPTURE, TruthKind.PROXY_HIGH_DEPTH])
def test_discrete_truth_must_be_integer(truth_kind: TruthKind) -> None:
    dataset = _dataset(truth_kind)
    layer = dataset.uns["primary_truth_layer"]
    truth = dataset.layers[layer].astype(float)
    truth[0, 0] = 1.25
    dataset.layers[layer] = truth

    with pytest.raises(ValueError, match="finite nonnegative integers"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize("value", [-0.01, np.nan, np.inf])
def test_continuous_truth_must_be_finite_and_nonnegative(value: float) -> None:
    dataset = _dataset(TruthKind.EXACT_CONTINUOUS)
    truth = dataset.layers["latent_expression"].copy()
    truth[0, 0] = value
    dataset.layers["latent_expression"] = truth

    with pytest.raises(ValueError, match="finite and nonnegative"):
        validate_benchmark_dataset(dataset)


def test_continuous_truth_may_be_fractional() -> None:
    dataset = _dataset(TruthKind.EXACT_CONTINUOUS)
    dataset.layers["latent_expression"][0, 0] = 0.123456

    validate_benchmark_dataset(dataset)


@pytest.mark.parametrize(
    ("truth_kind", "wrong_primary", "message"),
    [
        (TruthKind.EXACT_PRE_CAPTURE, "expected_counts", "pre_capture_counts"),
        (TruthKind.EXACT_CONTINUOUS, "expected_counts", "continuous truth"),
        (TruthKind.PROXY_HIGH_DEPTH, "heldout_counts", "reference_counts"),
    ],
)
def test_primary_truth_layer_is_fixed_by_truth_kind(
    truth_kind: TruthKind, wrong_primary: str, message: str
) -> None:
    dataset = _dataset(truth_kind)
    dataset.layers[wrong_primary] = np.ones(dataset.shape)
    dataset.uns["primary_truth_layer"] = wrong_primary

    with pytest.raises(ValueError, match=message):
        validate_benchmark_dataset(dataset)


def test_orthogonal_data_has_no_primary_truth() -> None:
    dataset = _dataset(TruthKind.ORTHOGONAL_ONLY)
    dataset.layers["reference_counts"] = np.ones(dataset.shape, dtype=int)
    dataset.uns["primary_truth_layer"] = "reference_counts"

    with pytest.raises(ValueError, match="must not declare primary truth"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("software_version"), "software_version"),
        (lambda p: p.__setitem__("source_sha256", "not-a-sha256"), "source_sha256"),
        (lambda p: p.__setitem__("parameters", {"bad": np.nan}), "canonical JSON"),
        (lambda p: p.__setitem__("seeds", [1, 2]), "seeds"),
    ],
)
def test_provenance_is_complete_and_canonical(mutate, message: str) -> None:
    dataset = _dataset()
    mutate(dataset.uns["provenance"])

    with pytest.raises(ValueError, match=message):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize("axis", ["obs", "var"])
def test_stable_ids_must_be_unique_and_nonempty(axis: str) -> None:
    dataset = _dataset()
    names = getattr(dataset, f"{axis}_names").tolist()
    names[1] = names[0]
    setattr(dataset, f"{axis}_names", names)

    with pytest.raises(ValueError, match=f"{axis} IDs must be unique"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize(
    "column",
    [
        "dataset_id",
        "mechanism",
        "condition",
        "biological_id",
        "technical_view",
        "draw",
        "library_size",
    ],
)
def test_required_cell_metadata_cannot_be_missing(column: str) -> None:
    dataset = _dataset()
    dataset.obs.drop(columns=column, inplace=True)

    with pytest.raises(ValueError, match=f"missing required obs column.*{column}"):
        validate_benchmark_dataset(dataset)


def test_evaluator_layers_are_shape_checked_even_if_not_primary() -> None:
    dataset = _dataset()
    dataset._layers["expected_counts"] = np.ones((1, 1))

    with pytest.raises(ValueError, match="expected_counts.*shape"):
        validate_benchmark_dataset(dataset)


def test_dense_and_sparse_storage_have_the_same_dataset_hash() -> None:
    dense = _dataset()
    dense.layers["heldout_counts"] = (OBSERVED > 0).astype(np.int32)
    dense.layers["expected_counts"] = OBSERVED.astype(np.float32) + 0.5
    sparse_dataset = dense.copy()
    sparse_dataset.X = sparse.csr_matrix(sparse_dataset.X)
    for layer_name in list(sparse_dataset.layers):
        sparse_dataset.layers[layer_name] = sparse.csc_matrix(
            sparse_dataset.layers[layer_name]
        )

    assert benchmark_dataset_sha256(dense) == benchmark_dataset_sha256(sparse_dataset)


def test_dataset_hash_binds_truth_metadata_ids_and_provenance() -> None:
    dataset = _dataset()
    original = benchmark_dataset_sha256(dataset)

    changed_truth = dataset.copy()
    changed_truth.layers["pre_capture_counts"][0, 1] += 1
    assert benchmark_dataset_sha256(changed_truth) != original

    changed_obs = dataset.copy()
    changed_obs.obs.loc["cell-0", "group"] = "changed"
    assert benchmark_dataset_sha256(changed_obs) != original

    changed_var = dataset.copy()
    changed_var.var.loc["gene-0", "is_marker"] = False
    assert benchmark_dataset_sha256(changed_var) != original

    changed_provenance = dataset.copy()
    changed_provenance.uns["provenance"]["seeds"]["measurement"] += 1
    assert benchmark_dataset_sha256(changed_provenance) != original

    changed_id = dataset.copy()
    changed_id.obs_names = ["replacement", *changed_id.obs_names[1:]]
    assert benchmark_dataset_sha256(changed_id) != original


def test_inference_view_contains_only_declared_non_evaluative_inputs() -> None:
    dataset = _dataset()
    dataset.layers["heldout_counts"] = np.ones(dataset.shape, dtype=int)
    dataset.layers["expected_counts"] = np.ones(dataset.shape, dtype=float)
    dataset.obsm["truth_embedding"] = np.arange(12).reshape(6, 2)
    dataset.varm["marker_loadings"] = np.arange(8).reshape(4, 2)
    dataset.obsp["truth_neighbors"] = sparse.eye(6, format="csr")
    dataset.varp["gene_network"] = sparse.eye(4, format="csr")
    dataset.raw = dataset
    dataset.uns["evaluator_settings"] = {"label_key": "group"}
    source_hash = benchmark_dataset_sha256(dataset)

    view = make_inference_view(dataset)

    np.testing.assert_array_equal(view.X, dataset.X)
    assert view.obs_names.tolist() == dataset.obs_names.tolist()
    assert view.var_names.tolist() == dataset.var_names.tolist()
    assert view.obs.columns.tolist() == ["batch", "donor_age"]
    assert view.var.columns.tolist() == ["gc_content"]
    assert not view.layers
    assert not view.obsm
    assert not view.varm
    assert not view.obsp
    assert not view.varp
    assert view.raw is None
    assert set(view.uns) == {
        "normalization",
        "allowed_covariates",
        "source_dataset_sha256",
    }
    assert view.uns["source_dataset_sha256"] == source_hash
    assert "group" not in view.obs
    assert "label" not in view.obs
    assert "pseudotime" not in view.obs
    assert "is_marker" not in view.var
    assert "marker_score" not in view.var


def test_inference_view_and_source_do_not_share_mutable_state() -> None:
    dataset = _dataset()
    original_x = np.asarray(dataset.X).copy()
    original_normalization = deepcopy(dataset.uns["normalization"])

    view = make_inference_view(dataset)
    view.X[0, 0] = 999
    view.obs.loc["cell-0", "batch"] = "mutated"
    view.var.loc["gene-0", "gc_content"] = 0.0
    view.uns["normalization"]["input"] = "mutated"

    np.testing.assert_array_equal(dataset.X, original_x)
    assert dataset.obs.loc["cell-0", "batch"] == "b1"
    assert dataset.var.loc["gene-0", "gc_content"] == 0.41
    assert dataset.uns["normalization"] == original_normalization

    dataset.X[1, 1] = 777
    dataset.obs.loc["cell-1", "batch"] = "source-mutated"
    assert view.X[1, 1] == OBSERVED[1, 1]
    assert view.obs.loc["cell-1", "batch"] == "b1"


@pytest.mark.parametrize("forbidden", ["group", "label", "pseudotime"])
def test_evaluator_metadata_cannot_be_declared_as_allowed_covariate(
    forbidden: str,
) -> None:
    dataset = _dataset()
    dataset.uns["allowed_covariates"]["obs"].append(forbidden)

    with pytest.raises(ValueError, match="evaluator metadata"):
        make_inference_view(dataset)
