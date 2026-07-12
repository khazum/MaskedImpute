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


@pytest.mark.parametrize("dtype", [object, "U8", "S8", np.complex128, np.longdouble])
@pytest.mark.parametrize("matrix_role", ["observed", "discrete_truth", "continuous_truth"])
def test_matrices_reject_non_native_or_wider_than_float64_dtypes(
    dtype, matrix_role: str
) -> None:
    truth_kind = (
        TruthKind.EXACT_CONTINUOUS
        if matrix_role == "continuous_truth"
        else TruthKind.EXACT_PRE_CAPTURE
    )
    dataset = _dataset(truth_kind)
    if matrix_role == "observed":
        dataset.X = np.asarray(dataset.X).astype(dtype)
    else:
        layer = dataset.uns["primary_truth_layer"]
        dataset.layers[layer] = np.asarray(dataset.layers[layer]).astype(dtype)

    with pytest.raises(ValueError, match="native bool/integer/float.*float64"):
        validate_benchmark_dataset(dataset)


def test_longdouble_fraction_is_not_rounded_into_a_discrete_integer() -> None:
    dataset = _dataset()
    truth = np.asarray(dataset.layers["pre_capture_counts"], dtype=np.longdouble)
    truth[0, 0] = np.longdouble(2**53) + np.longdouble("0.5")
    dataset.layers["pre_capture_counts"] = truth

    with pytest.raises(ValueError, match="float64"):
        validate_benchmark_dataset(dataset)


def test_discrete_float_counts_must_fit_uint64() -> None:
    dataset = _dataset()
    observed = np.zeros(dataset.shape, dtype=np.float64)
    observed[0, 0] = np.float64(2**64)
    dataset.X = observed
    dataset.obs["library_size"] = [2**64, 0, 0, 0, 0, 0]

    with pytest.raises(ValueError, match="uint64"):
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


def test_integer_continuous_truth_must_be_exactly_float64_representable() -> None:
    dataset = _dataset(TruthKind.EXACT_CONTINUOUS)
    truth = np.zeros(dataset.shape, dtype=np.uint64)
    truth[0, 0] = 2**53 + 1
    dataset.layers["latent_expression"] = truth

    with pytest.raises(ValueError, match="exactly representable.*float64"):
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
    "layer",
    [
        "pre_capture_counts",
        "latent_expression",
        "pre_dropout_expression",
        "reference_counts",
        "heldout_counts",
        "expected_counts",
    ],
)
def test_orthogonal_data_rejects_every_evaluator_layer(layer: str) -> None:
    dataset = _dataset(TruthKind.ORTHOGONAL_ONLY)
    dataset.layers[layer] = np.ones(dataset.shape, dtype=int)

    with pytest.raises(ValueError, match="orthogonal_only.*evaluator truth"):
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


@pytest.mark.parametrize(
    "column",
    ["dataset_id", "mechanism", "condition", "biological_id", "technical_view"],
)
@pytest.mark.parametrize("invalid", ["", "   ", 42])
def test_design_identifiers_must_be_nonempty_strings(column: str, invalid) -> None:
    dataset = _dataset()
    dataset.obs[column] = [invalid] * dataset.n_obs

    with pytest.raises(ValueError, match=f"{column}.*nonempty strings"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize(
    "column",
    ["dataset_id", "mechanism", "condition", "biological_id", "technical_view"],
)
def test_design_identifiers_are_constant_within_dataset(column: str) -> None:
    dataset = _dataset()
    dataset.obs.loc["cell-5", column] = "different"

    with pytest.raises(ValueError, match=f"{column}.*constant"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, "1", True])
def test_draw_is_a_constant_positive_integer(invalid) -> None:
    dataset = _dataset()
    dataset.obs["draw"] = [invalid] * dataset.n_obs

    with pytest.raises(ValueError, match="draw.*positive integer"):
        validate_benchmark_dataset(dataset)


def test_draw_is_constant_within_dataset() -> None:
    dataset = _dataset()
    dataset.obs.loc["cell-5", "draw"] = 2

    with pytest.raises(ValueError, match="draw.*constant"):
        validate_benchmark_dataset(dataset)


@pytest.mark.parametrize("invalid", [-1, 1.5, np.nan, np.inf, "3"])
def test_library_size_is_a_finite_nonnegative_integer(invalid) -> None:
    dataset = _dataset()
    library_size = dataset.obs["library_size"].astype(object)
    library_size.loc["cell-0"] = invalid
    dataset.obs["library_size"] = library_size

    with pytest.raises(ValueError, match="library_size.*finite nonnegative integer"):
        validate_benchmark_dataset(dataset)


def test_library_size_must_exactly_equal_observed_row_sum() -> None:
    dataset = _dataset()
    dataset.obs.loc["cell-0", "library_size"] += 1

    with pytest.raises(ValueError, match="library_size.*row sums"):
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


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int8,
        np.int32,
        np.int64,
        np.uint8,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize(
    "truth_kind",
    [TruthKind.EXACT_PRE_CAPTURE, TruthKind.EXACT_CONTINUOUS],
)
def test_every_accepted_native_matrix_dtype_validates_hashes_and_makes_view(
    dtype, truth_kind: TruthKind
) -> None:
    dataset = _dataset(truth_kind)
    dataset.X = np.asarray(dataset.X).astype(dtype)
    dataset.obs["library_size"] = [
        int(sum(int(value) for value in row)) for row in np.asarray(dataset.X)
    ]
    primary = dataset.uns["primary_truth_layer"]
    dataset.layers[primary] = np.asarray(dataset.layers[primary]).astype(dtype)

    validate_benchmark_dataset(dataset)
    digest = benchmark_dataset_sha256(dataset)
    view = make_inference_view(dataset)

    assert len(digest) == 64
    np.testing.assert_array_equal(view.X, dataset.X)


def test_discrete_hash_is_canonical_across_native_numeric_dtypes() -> None:
    digests = set()
    for dtype in (np.int64, np.uint32, np.float64):
        dataset = _dataset()
        dataset.X = np.asarray(dataset.X).astype(dtype)
        dataset.layers["pre_capture_counts"] = np.asarray(
            dataset.layers["pre_capture_counts"]
        ).astype(dtype)
        digests.add(benchmark_dataset_sha256(dataset))

    assert len(digests) == 1


def test_continuous_hash_is_canonical_for_exactly_representable_values() -> None:
    digests = set()
    for dtype in (np.int32, np.uint32, np.float32, np.float64):
        dataset = _dataset(TruthKind.EXACT_CONTINUOUS)
        dataset.layers["latent_expression"] = OBSERVED.astype(dtype)
        digests.add(benchmark_dataset_sha256(dataset))

    assert len(digests) == 1


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


def test_dataset_hash_is_lossless_for_integers_above_float_precision() -> None:
    first = _dataset()
    second = _dataset()
    first.layers["pre_capture_counts"][0, 0] = 2**53
    second.layers["pre_capture_counts"][0, 0] = 2**53 + 1

    assert benchmark_dataset_sha256(first) != benchmark_dataset_sha256(second)

    sparse_first = first.copy()
    sparse_first.X = sparse.csr_matrix(first.X)
    for layer_name in list(sparse_first.layers):
        sparse_first.layers[layer_name] = sparse.csr_matrix(
            sparse_first.layers[layer_name]
        )
    assert benchmark_dataset_sha256(first) == benchmark_dataset_sha256(sparse_first)


def test_dataset_hash_binds_categorical_levels_order_and_ordered_flag() -> None:
    baseline = _dataset()
    baseline.obs["group"] = pd.Categorical(
        baseline.obs["group"], categories=["A", "B", "unused"], ordered=True
    )
    changed_level_order = baseline.copy()
    changed_level_order.obs["group"] = pd.Categorical(
        changed_level_order.obs["group"],
        categories=["B", "A", "unused"],
        ordered=True,
    )
    changed_ordered = baseline.copy()
    changed_ordered.obs["group"] = changed_ordered.obs["group"].cat.as_unordered()

    baseline_hash = benchmark_dataset_sha256(baseline)
    assert benchmark_dataset_sha256(changed_level_order) != baseline_hash
    assert benchmark_dataset_sha256(changed_ordered) != baseline_hash


def test_inference_view_contains_only_declared_non_evaluative_inputs() -> None:
    dataset = _dataset()
    dataset.layers["heldout_counts"] = np.ones(dataset.shape, dtype=int)
    dataset.layers["expected_counts"] = np.ones(dataset.shape, dtype=float)
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


@pytest.mark.parametrize("slot", ["obsm", "varm", "obsp", "varp", "raw"])
def test_unsupported_ann_data_slots_are_rejected_fail_closed(slot: str) -> None:
    dataset = _dataset()
    if slot == "obsm":
        dataset.obsm["embedding"] = np.arange(12).reshape(6, 2)
    elif slot == "varm":
        dataset.varm["loadings"] = np.arange(8).reshape(4, 2)
    elif slot == "obsp":
        dataset.obsp["neighbors"] = sparse.eye(6, format="csr")
    elif slot == "varp":
        dataset.varp["network"] = sparse.eye(4, format="csr")
    else:
        dataset.raw = dataset

    with pytest.raises(ValueError, match=f"unsupported AnnData slot.*{slot}"):
        validate_benchmark_dataset(dataset)


def test_unsupported_uns_evaluator_settings_are_rejected_fail_closed() -> None:
    dataset = _dataset()
    dataset.uns["evaluator_settings"] = {"label_key": "group"}

    with pytest.raises(ValueError, match="unsupported uns key.*evaluator_settings"):
        validate_benchmark_dataset(dataset)


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


def test_categorical_covariates_are_copied_with_independent_schema() -> None:
    dataset = _dataset()
    dataset.obs["batch"] = pd.Categorical(
        dataset.obs["batch"], categories=["b1", "b2", "b3"], ordered=True
    )

    view = make_inference_view(dataset)

    assert view.obs["batch"].cat.ordered
    assert view.obs["batch"].cat.categories.tolist() == ["b1", "b2", "b3"]
    assert view.obs["batch"].cat.categories is not dataset.obs["batch"].cat.categories


def test_nested_object_covariates_are_rejected() -> None:
    dataset = _dataset()
    dataset.obs["unsafe"] = [["hidden-label"] for _ in range(dataset.n_obs)]
    dataset.uns["allowed_covariates"]["obs"].append("unsafe")

    with pytest.raises(ValueError, match="immutable scalar"):
        validate_benchmark_dataset(dataset)


def test_safe_string_object_covariates_are_supported() -> None:
    dataset = _dataset()
    assert dataset.obs["batch"].dtype == object

    validate_benchmark_dataset(dataset)
    view = make_inference_view(dataset)
    assert view.obs["batch"].tolist() == dataset.obs["batch"].tolist()


@pytest.mark.parametrize("innocuous", ["estate_id", "real_estate_value"])
def test_evaluator_name_matching_respects_token_boundaries(innocuous: str) -> None:
    dataset = _dataset()
    dataset.obs[innocuous] = ["north"] * dataset.n_obs
    dataset.uns["allowed_covariates"]["obs"].append(innocuous)

    view = make_inference_view(dataset)

    assert innocuous in view.obs


@pytest.mark.parametrize(
    "forbidden",
    [
        "dataset_id",
        "mechanism",
        "condition",
        "biological_id",
        "technical_view",
        "draw",
        "library_size",
        "group",
        "label",
        "pseudotime",
        "cluster_assignment",
        "class_prediction",
        "cell_state",
        "cell_type_score",
        "case_control_flag",
        "disease_status",
        "treatment_arm",
    ],
)
def test_evaluator_metadata_cannot_be_declared_as_allowed_covariate(
    forbidden: str,
) -> None:
    dataset = _dataset()
    if forbidden not in dataset.obs:
        dataset.obs[forbidden] = ["hidden"] * dataset.n_obs
    dataset.uns["allowed_covariates"]["obs"].append(forbidden)

    with pytest.raises(ValueError, match="evaluator metadata"):
        make_inference_view(dataset)


def test_normalization_accepts_only_whitelisted_scalar_metadata() -> None:
    dataset = _dataset()
    dataset.uns["normalization"] = {
        "input": "raw_umi_counts",
        "target_sum": 10_000,
        "log_base": 2.0,
        "size_factor": "library_size",
    }

    validate_benchmark_dataset(dataset)
    view = make_inference_view(dataset)
    assert view.uns["normalization"] == dataset.uns["normalization"]


@pytest.mark.parametrize(
    "normalization",
    [
        {"size_factor": "none"},
        {"input": 123, "size_factor": "none"},
        {"input": "raw_umi_counts", "target_sum": "10000"},
        {"input": "raw_umi_counts", "log_base": True},
        {"input": "raw_umi_counts", "log_base": 1.0},
        {"input": "raw_umi_counts", "size_factor": False},
        {"input": "raw_umi_counts", "labels": ["A", "B"]},
        {"input": "raw_umi_counts", "size_factor": ["A", "B"]},
        {"input": {"labels": ["A", "B"]}, "size_factor": "none"},
    ],
)
def test_normalization_cannot_carry_nested_evaluator_metadata(
    normalization: dict[str, object],
) -> None:
    dataset = _dataset()
    dataset.uns["normalization"] = normalization

    with pytest.raises(ValueError, match="normalization"):
        validate_benchmark_dataset(dataset)
