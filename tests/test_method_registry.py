from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from maskimpute_benchmark.methods import (
    MethodContractError,
    MethodPlanEntry,
    build_method_status_table,
    canonical_run_record_bytes,
    load_method_registry,
    prepare_method_input,
    snapshot_method_output,
    validate_run_record,
    verify_cached_method_sources,
)


METHODS_PATH = Path("study/methods.json")
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

MANDATORY_METHODS = {
    "observed",
    "capacity-matched-ae",
    "maskimpute",
    "alra",
    "magic",
    "dca",
    "scvi",
    "saver",
    "scimpute",
    "wedge",
    "scziva",
    "afmf",
    "biaeimpute",
    "sccr",
    "scgacl",
    "sctacl",
    "sczn",
    "scsdae",
    "d3impute",
    "sctsi",
}

REQUIRED_SAME_INPUT_METHODS = MANDATORY_METHODS - {
    "scimpute",
    "wedge",
    "scgacl",
    "sctacl",
    "sczn",
    "d3impute",
    "sctsi",
}


def _registry_payload() -> dict[str, object]:
    return json.loads(METHODS_PATH.read_text(encoding="utf-8"))


def _write_registry(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "methods.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inference_view(*, sparse_counts: bool = False) -> ad.AnnData:
    counts = np.array([[2, 0, 1], [0, 3, 0]], dtype=np.int64)
    matrix = sparse.csr_matrix(counts) if sparse_counts else counts
    view = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"batch": pd.Categorical(["a", "b"])},
            index=pd.Index(["cell-1", "cell-2"], dtype=object),
        ),
        var=pd.DataFrame(
            {"feature_class": ["gene", "gene", "gene"]},
            index=pd.Index(["gene-1", "gene-2", "gene-3"], dtype=object),
        ),
    )
    view.uns["source_dataset_sha256"] = SHA_A
    view.uns["normalization"] = {
        "input": "counts",
        "target_sum": None,
        "log_base": None,
        "size_factor": None,
    }
    view.uns["allowed_covariates"] = {
        "obs": ["batch"],
        "var": ["feature_class"],
    }
    return view


def _completed_record(method_id: str, *, seed: int | None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": f"run-{method_id}",
        "method_id": method_id,
        "source_dataset_sha256": SHA_A,
        "status": "completed",
        "seed": seed,
        "runtime_seconds": 1.25,
        "peak_rss_bytes": 1024,
        "peak_gpu_bytes": 0,
        "stdout_sha256": SHA_A,
        "stderr_sha256": SHA_B,
        "output_sha256": SHA_C,
        "reason": None,
    }


def test_tracked_registry_declares_complete_prespecified_denominator() -> None:
    registry = load_method_registry(METHODS_PATH)

    assert registry.schema_version == 1
    assert set(registry.ids) == MANDATORY_METHODS
    assert len(registry.ids) == len(set(registry.ids))
    assert registry.by_id("observed").role == "control"
    assert registry.by_id("capacity-matched-ae").role == "control"
    assert registry.by_id("maskimpute").role == "candidate"
    assert registry.by_id("d3impute").track == "external_reference"
    assert registry.by_id("sctsi").track == "external_reference"
    assert registry.by_id("scsdae").integration_status == "implemented"
    assert registry.by_id("scimpute").integration_status == "historical"
    assert registry.by_id("wedge").integration_status == "historical"


def test_execution_plan_separates_required_external_historical_and_inapplicable() -> (
    None
):
    registry = load_method_registry(METHODS_PATH)
    plan = {entry.method_id: entry for entry in registry.execution_plan()}

    assert set(plan) == MANDATORY_METHODS
    assert all(isinstance(entry, MethodPlanEntry) for entry in plan.values())
    assert {
        method_id
        for method_id, entry in plan.items()
        if entry.execution_scope == "same_input_required"
    } == REQUIRED_SAME_INPUT_METHODS
    assert all(plan[method_id].executable for method_id in REQUIRED_SAME_INPUT_METHODS)
    assert {
        method_id
        for method_id, entry in plan.items()
        if entry.execution_scope == "external_reference_only"
    } == {"d3impute", "sctsi"}
    assert plan["d3impute"].executable
    assert plan["sctsi"].executable
    assert plan["scimpute"].execution_scope == "historical_not_run"
    assert plan["wedge"].execution_scope == "historical_not_run"
    assert not plan["scimpute"].executable
    assert not plan["wedge"].executable
    assert plan["scgacl"].execution_scope == "not_applicable"
    assert plan["scgacl"].applicability_reason == (
        "upstream_no_dataset_general_truth_free_configuration"
    )
    assert plan["sctacl"].execution_scope == "not_applicable"
    assert plan["sctacl"].applicability_reason == (
        "upstream_incomplete_no_full_count_imputation_output"
    )
    assert plan["sczn"].execution_scope == "not_applicable"
    assert plan["sczn"].applicability_reason == (
        "upstream_not_packaged_as_callable_method"
    )
    assert not plan["scgacl"].executable
    assert not plan["sctacl"].executable
    assert not plan["sczn"].executable


def test_registry_derives_exact_selection_same_input_denominator() -> None:
    registry = load_method_registry(METHODS_PATH)

    assert tuple(
        spec.id
        for spec in registry.methods
        if spec.execution_scope == "same_input_required" and spec.role != "candidate"
    ) == (
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


def test_sczn_source_attempt_receipt_binds_packaging_license_labels_and_output() -> (
    None
):
    registry = load_method_registry(METHODS_PATH)
    spec = registry.by_id("sczn")
    receipt = json.loads(
        Path("study/method-attempts/sczn.json").read_text(encoding="utf-8")
    )

    assert spec.source.revision == "ab3dfd01497809e8a24b638539c0680f8aa80580"
    assert spec.source.tree == "240b09c753794a04c06a98753482faf4062d8e02"
    assert spec.license.status == "NOASSERTION"
    assert receipt["source"] == {
        "url": spec.source.url,
        "revision": spec.source.revision,
        "tree": spec.source.tree,
        "checkout_status": "pristine",
    }
    assert receipt["outcome"] == "unavailable"
    assert receipt["reason_code"] == "upstream_not_packaged_as_callable_method"
    assert receipt["license_evidence"]["status"] == "NOASSERTION"
    assert receipt["packaging_evidence"]["callable_entrypoint"] is None
    assert receipt["packaging_evidence"]["implementation_container"] == (
        "dataset_specific_notebook"
    )
    assert receipt["truth_free_evidence"]["cell_type_labels_required"] is True
    assert receipt["truth_free_evidence"]["supervised_classification_loss"] is True
    assert receipt["output_evidence"]["writes_labeled_csv"] is True
    assert receipt["output_evidence"]["runtime_identifier_validation"] is False


def test_scgimpute_discovery_receipt_records_dated_public_source_search() -> None:
    receipt = json.loads(
        Path("study/method-attempts/scgimpute.json").read_text(encoding="utf-8")
    )

    assert receipt["method_id"] == "scgimpute"
    assert receipt["paper"]["doi"] == "10.1016/j.compbiolchem.2025.108856"
    assert receipt["paper"]["publication"] == {
        "journal": "Computational Biology and Chemistry",
        "volume": "121",
        "article": "108856",
        "date": "2026-04",
    }
    assert receipt["source_search"]["search_date"] == "2026-07-12"
    assert receipt["source_search"]["public_repository"] is None
    assert receipt["registry_disposition"] == "not_added_to_execution_registry"
    assert receipt["outcome"] == "unavailable"
    assert receipt["reason_code"] == "public_source_not_located"


def test_execution_scope_corrects_sctacl_scale_and_sccr_resource_mode() -> None:
    registry = load_method_registry(METHODS_PATH)

    assert registry.by_id("sctacl").output_scale == "method_native_normalized"
    assert registry.by_id("sccr").resources.gpu_required is True
    assert registry.by_id("sccr").resources.max_gpu_gib == 14
    assert registry.by_id("sccr").resources.gpu_mode == "required"
    assert registry.by_id("saver").resources.gpu_mode == "forbidden"
    assert registry.by_id("scsdae").resources.gpu_mode == "required"


@pytest.mark.parametrize(
    ("scope", "reason", "message"),
    [
        ("external_reference_only", None, "external_reference"),
        ("not_applicable", None, "applicability_reason"),
        ("same_input_required", "not_really", "applicability_reason"),
        ("invented_scope", None, "execution_scope"),
    ],
)
def test_registry_rejects_inconsistent_execution_applicability(
    tmp_path: Path,
    scope: str,
    reason: str | None,
    message: str,
) -> None:
    payload = _registry_payload()
    methods = payload["methods"]
    assert isinstance(methods, list)
    observed = next(item for item in methods if item["id"] == "observed")
    observed["execution_scope"] = scope
    observed["applicability_reason"] = reason

    with pytest.raises(MethodContractError, match=message):
        load_method_registry(_write_registry(tmp_path, payload))


def test_status_table_uses_nonexecution_scope_before_integration_readiness() -> None:
    registry = load_method_registry(METHODS_PATH)
    rows = {row.method_id: row for row in build_method_status_table(registry, ())}

    assert rows["scimpute"].status == "historical_not_run"
    assert rows["scimpute"].reason is None
    assert rows["scgacl"].status == "not_applicable"
    assert rows["scgacl"].reason == (
        "upstream_no_dataset_general_truth_free_configuration"
    )


def test_every_method_declares_source_license_citation_environment_and_budget() -> None:
    registry = load_method_registry(METHODS_PATH)

    for spec in registry.methods:
        assert spec.role in {"control", "candidate", "competitor"}
        assert spec.track in {"same_input", "external_reference"}
        assert spec.input_scale
        assert spec.output_scale
        assert spec.seed_policy in {"required", "not_applicable"}
        assert spec.environment.id
        assert spec.environment.status in {"pending", "ready", "failed"}
        assert spec.resources.timeout_seconds > 0
        assert spec.resources.cpu_cores > 0
        assert spec.resources.max_rss_gib > 0
        assert spec.resources.max_gpu_gib >= 0
        assert isinstance(spec.preserves_observed_positives, bool)
        assert spec.citation.status in {"verified", "pending"}
        if spec.citation.status == "verified":
            assert spec.citation.doi is not None
        if spec.source.kind == "git":
            assert spec.source.url.startswith("https://")
            assert len(spec.source.revision or "") == 40
            assert len(spec.source.tree or "") == 40
            assert spec.source.freeze_binding is None
        else:
            assert spec.source.kind == "in_tree"
            assert spec.source.freeze_binding == "study_freeze_commit"
            assert spec.source.revision is None
            assert spec.source.tree is None


def test_noassertion_sources_use_pristine_nonredistribution_policy() -> None:
    registry = load_method_registry(METHODS_PATH)

    noassertion = [
        spec for spec in registry.methods if spec.license.status == "NOASSERTION"
    ]
    assert noassertion
    assert all(
        spec.source_policy == "invoke_pristine_source_no_redistribution"
        for spec in noassertion
    )
    scziva = registry.by_id("scziva")
    assert scziva.license.status == "NOASSERTION"
    assert "permission" in (scziva.license.notice or "").casefold()


def test_registry_rejects_duplicate_json_keys_and_method_ids(tmp_path: Path) -> None:
    raw = METHODS_PATH.read_text(encoding="utf-8")
    duplicate_key = raw.replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    path = tmp_path / "duplicate-key.json"
    path.write_text(duplicate_key, encoding="utf-8")
    with pytest.raises(MethodContractError, match="duplicate JSON key"):
        load_method_registry(path)

    payload = _registry_payload()
    methods = payload["methods"]
    assert isinstance(methods, list)
    methods.append(deepcopy(methods[0]))
    with pytest.raises(MethodContractError, match="duplicate method id"):
        load_method_registry(_write_registry(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("source", "revision"), "main", "revision"),
        (("source", "tree"), "0" * 39, "tree"),
        (("license", "status"), "probably-free", "license"),
        (("license", "spdx"), "LicenseRef-Handwave", "SPDX"),
        (("input_scale",), "mystery", "input_scale"),
        (("resources", "timeout_seconds"), 0, "timeout_seconds"),
        (("resources", "cpu_cores"), True, "cpu_cores"),
        (("resources", "max_rss_gib"), float("inf"), "non-finite"),
        (("resources", "max_gpu_gib"), -1, "max_gpu_gib"),
    ],
)
def test_registry_rejects_malformed_pins_licenses_scales_and_resources(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _registry_payload()
    methods = payload["methods"]
    assert isinstance(methods, list)
    method = next(item for item in methods if item["id"] == "alra")
    target = method
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(MethodContractError, match=message):
        load_method_registry(_write_registry(tmp_path, payload))


def test_registry_rejects_ready_environment_without_real_lock_hash(
    tmp_path: Path,
) -> None:
    payload = _registry_payload()
    methods = payload["methods"]
    assert isinstance(methods, list)
    methods[0]["environment"]["status"] = "ready"
    methods[0]["environment"]["lock_sha256"] = None

    with pytest.raises(MethodContractError, match="ready environment.*lock_sha256"):
        load_method_registry(_write_registry(tmp_path, payload))


def test_available_cached_git_sources_match_declared_commit_tree_and_remote() -> None:
    registry = load_method_registry(METHODS_PATH)
    cache_root = Path("artifacts/method-sources")
    if not cache_root.exists():
        pytest.skip("ignored method-source cache is absent")

    verified = verify_cached_method_sources(
        registry,
        repository_root=Path("."),
        require_all=True,
    )
    assert set(verified) == {
        spec.id for spec in registry.methods if spec.source.kind == "git"
    }
    assert all(status == "verified" for status in verified.values())


def test_cache_verifier_rejects_wrong_tree_without_mutating_checkout(
    tmp_path: Path,
) -> None:
    payload = _registry_payload()
    methods = payload["methods"]
    assert isinstance(methods, list)
    alra = next(item for item in methods if item["id"] == "alra")
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"], check=True
    )
    (source / "method.py").write_text("pass\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "method.py"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "pin"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(source), "remote", "add", "origin", alra["source"]["url"]],
        check=True,
    )
    alra["source"]["revision"] = revision
    alra["source"]["tree"] = "0" * 40
    alra["source"]["cache_path"] = source.relative_to(tmp_path).as_posix()
    narrowed = {"schema_version": 1, "methods": [alra]}
    registry = load_method_registry(_write_registry(tmp_path, narrowed))

    with pytest.raises(MethodContractError, match="tree"):
        verify_cached_method_sources(
            registry,
            repository_root=tmp_path,
            require_all=True,
        )
    assert (source / "method.py").read_text(encoding="utf-8") == "pass\n"


@pytest.mark.parametrize("sparse_counts", [False, True])
def test_truth_free_input_is_validated_and_snapshotted(sparse_counts: bool) -> None:
    method_input = prepare_method_input(_inference_view(sparse_counts=sparse_counts))

    assert method_input.source_dataset_sha256 == SHA_A
    assert method_input.shape == (2, 3)
    assert method_input.obs_ids == ("cell-1", "cell-2")
    assert method_input.var_ids == ("gene-1", "gene-2", "gene-3")
    np.testing.assert_array_equal(
        method_input.counts,
        np.array([[2, 0, 1], [0, 3, 0]], dtype=np.float64),
    )
    assert method_input.counts.flags.writeable is False
    with pytest.raises(ValueError):
        method_input.counts.flags.writeable = True


def test_method_input_preserves_immutable_covariate_values_and_categorical_schema() -> (
    None
):
    view = _inference_view()
    view.obs["batch"] = pd.Categorical(
        ["a", "b"], categories=["b", "a", "unused"], ordered=True
    )

    method_input = prepare_method_input(view)

    assert len(method_input.obs_covariates) == 1
    batch = method_input.obs_covariates[0]
    assert batch.name == "batch"
    assert batch.kind == "categorical"
    assert batch.categories == ("b", "a", "unused")
    assert batch.ordered is True
    assert batch.codes == (1, 0)
    assert batch.values == ("a", "b")
    assert method_input.var_covariates[0].values == ("gene", "gene", "gene")
    reconstructed = method_input.covariate_frame("obs")
    assert reconstructed["batch"].cat.categories.tolist() == ["b", "a", "unused"]
    assert reconstructed["batch"].cat.ordered


def test_input_normalizes_h5ad_style_numpy_json_metadata() -> None:
    view = _inference_view()
    view.uns["allowed_covariates"] = {
        "obs": np.array(["batch"], dtype=object),
        "var": np.array(["feature_class"], dtype=object),
    }
    view.uns["normalization"] = {
        "input": np.str_("counts"),
        "target_sum": None,
        "log_base": None,
        "size_factor": None,
    }

    method_input = prepare_method_input(view)

    assert method_input.normalization == {
        "input": "counts",
        "target_sum": None,
        "log_base": None,
        "size_factor": None,
    }


def test_input_rejects_sparse_subclasses_and_callable_instance_hooks() -> None:
    class HostileCSR(sparse.csr_matrix):
        def toarray(self, *args, **kwargs):
            return np.full(self.shape, 9)

    view = _inference_view()
    view.X = HostileCSR(view.X)
    with pytest.raises(MethodContractError, match="exact supported SciPy sparse type"):
        prepare_method_input(view)

    view = _inference_view(sparse_counts=True)
    view.X.toarray = lambda *args, **kwargs: np.full(view.shape, 9)
    with pytest.raises(MethodContractError, match="callable sparse instance shadow"):
        prepare_method_input(view)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.array([[True, False], [False, True]]), "boolean"),
        (np.array([[2**53 + 1, 0], [0, 1]], dtype=np.uint64), "exactly representable"),
        (np.ma.array([[1, 0], [0, 1]], mask=[[False, True], [False, False]]), "masked"),
    ],
)
def test_input_rejects_boolean_inexact_and_masked_counts(
    counts: np.ndarray,
    message: str,
) -> None:
    view = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=["cell-1", "cell-2"]),
        var=pd.DataFrame(index=["gene-1", "gene-2"]),
    )
    view.uns["source_dataset_sha256"] = SHA_A

    with pytest.raises(MethodContractError, match=message):
        prepare_method_input(view)


@pytest.mark.parametrize("truth_slot", ["layer", "obsm", "unknown_uns"])
def test_truth_bearing_or_open_slot_input_is_rejected(truth_slot: str) -> None:
    view = _inference_view()
    if truth_slot == "layer":
        view.layers["pre_capture_counts"] = np.array(view.X, copy=True)
    elif truth_slot == "obsm":
        view.obsm["latent_truth"] = np.ones((2, 1))
    else:
        view.uns["truth_kind"] = "exact_pre_capture"

    with pytest.raises(MethodContractError, match="truth-free|closed slot"):
        prepare_method_input(view)


def test_input_rejects_duplicate_ids_and_undeclared_covariates() -> None:
    view = _inference_view()
    view.obs_names = ["cell-1", "cell-1"]
    with pytest.raises(MethodContractError, match="obs IDs must be unique"):
        prepare_method_input(view)

    view = _inference_view()
    view.obs["hidden_label"] = ["x", "y"]
    with pytest.raises(MethodContractError, match="undeclared obs columns"):
        prepare_method_input(view)


def test_output_snapshot_is_bound_finite_nonnegative_and_immutable() -> None:
    registry = load_method_registry(METHODS_PATH)
    spec = registry.by_id("maskimpute")
    method_input = prepare_method_input(_inference_view())
    output = np.array([[2.0, 0.5, 1.0], [0.25, 3.0, 0.75]])

    snapshot = snapshot_method_output(
        spec,
        method_input,
        output,
        source_dataset_sha256=SHA_A,
        output_scale="raw_counts",
        obs_ids=("cell-1", "cell-2"),
        var_ids=("gene-1", "gene-2", "gene-3"),
    )

    np.testing.assert_array_equal(snapshot.matrix, output)
    assert snapshot.matrix_sha256
    assert snapshot.shape == (2, 3)
    assert snapshot.matrix.flags.writeable is False
    with pytest.raises(ValueError):
        snapshot.matrix.flags.writeable = True
    output[0, 1] = 99.0
    assert snapshot.matrix[0, 1] == 0.5


def test_output_rejects_masked_arrays_sparse_subclasses_and_dtype_metadata() -> None:
    class HostileCSR(sparse.csr_matrix):
        def toarray(self, *args, **kwargs):
            return np.full(self.shape, 1.0)

    spec = load_method_registry(METHODS_PATH).by_id("alra")
    method_input = prepare_method_input(_inference_view())
    kwargs = {
        "source_dataset_sha256": SHA_A,
        "output_scale": "log1p_cp10k",
        "obs_ids": method_input.obs_ids,
        "var_ids": method_input.var_ids,
    }
    with pytest.raises(MethodContractError, match="masked"):
        snapshot_method_output(
            spec,
            method_input,
            np.ma.array(np.ones((2, 3)), mask=[[False, True, False], [False] * 3]),
            **kwargs,
        )
    with pytest.raises(MethodContractError, match="exact supported SciPy sparse type"):
        snapshot_method_output(
            spec,
            method_input,
            HostileCSR(np.ones((2, 3))),
            **kwargs,
        )
    metadata_dtype = np.dtype(np.float64, metadata={"unit": "counts"})
    with pytest.raises(MethodContractError, match="dtype metadata"):
        snapshot_method_output(
            spec,
            method_input,
            np.ones((2, 3), dtype=metadata_dtype),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"source_dataset_sha256": SHA_B}, "source dataset"),
        ({"output_scale": "log1p_cp10k"}, "output scale"),
        ({"obs_ids": ("cell-x", "cell-2")}, "obs IDs"),
        ({"var_ids": ("gene-1", "gene-2", "gene-x")}, "var IDs"),
    ],
)
def test_output_rejects_binding_mismatches(
    override: dict[str, object],
    message: str,
) -> None:
    spec = load_method_registry(METHODS_PATH).by_id("maskimpute")
    method_input = prepare_method_input(_inference_view())
    kwargs = {
        "source_dataset_sha256": SHA_A,
        "output_scale": "raw_counts",
        "obs_ids": ("cell-1", "cell-2"),
        "var_ids": ("gene-1", "gene-2", "gene-3"),
    }
    kwargs.update(override)

    with pytest.raises(MethodContractError, match=message):
        snapshot_method_output(
            spec,
            method_input,
            np.array([[2.0, 0.5, 1.0], [0.25, 3.0, 0.75]]),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.ones((3, 2)), "shape"),
        (np.array([[2.0, np.nan, 1.0], [0.0, 3.0, 0.0]]), "finite"),
        (np.array([[2.0, -0.1, 1.0], [0.0, 3.0, 0.0]]), "nonnegative"),
        (np.array([[1.9, 0.1, 1.0], [0.1, 3.0, 0.1]]), "observed positives"),
    ],
)
def test_output_rejects_shape_numeric_and_positive_preservation_violations(
    output: np.ndarray,
    message: str,
) -> None:
    spec = load_method_registry(METHODS_PATH).by_id("maskimpute")
    method_input = prepare_method_input(_inference_view())

    with pytest.raises(MethodContractError, match=message):
        snapshot_method_output(
            spec,
            method_input,
            output,
            source_dataset_sha256=SHA_A,
            output_scale="raw_counts",
            obs_ids=("cell-1", "cell-2"),
            var_ids=("gene-1", "gene-2", "gene-3"),
        )


def test_completed_and_failed_run_records_have_closed_status_specific_fields() -> None:
    registry = load_method_registry(METHODS_PATH)
    completed = validate_run_record(
        registry,
        _completed_record("maskimpute", seed=42),
    )
    failed_payload = _completed_record("maskimpute", seed=42)
    failed_payload.update(
        {
            "status": "failed",
            "output_sha256": None,
            "reason": "timeout_after_resource_limit",
        }
    )
    failed = validate_run_record(registry, failed_payload)

    assert completed.status == "completed"
    assert completed.reason is None
    assert failed.status == "failed"
    assert failed.output_sha256 is None
    assert failed.reason == "timeout_after_resource_limit"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"runtime_seconds": None}, "runtime_seconds"),
        ({"peak_rss_bytes": -1}, "peak_rss_bytes"),
        ({"stdout_sha256": None}, "stdout_sha256"),
        ({"output_sha256": None}, "completed.*output_sha256"),
        ({"reason": "not really complete"}, "completed.*reason"),
        ({"seed": None}, "seed"),
    ],
)
def test_run_record_rejects_missing_or_fabricated_completion_fields(
    mutation: dict[str, object],
    message: str,
) -> None:
    registry = load_method_registry(METHODS_PATH)
    payload = _completed_record("maskimpute", seed=42)
    payload.update(mutation)

    with pytest.raises(MethodContractError, match=message):
        validate_run_record(registry, payload)


def test_run_record_enforces_deterministic_seed_policy_and_canonical_bytes() -> None:
    registry = load_method_registry(METHODS_PATH)
    observed_payload = _completed_record("observed", seed=None)
    first = validate_run_record(registry, observed_payload)
    second = validate_run_record(
        registry,
        dict(reversed(list(observed_payload.items()))),
    )

    assert canonical_run_record_bytes(first) == canonical_run_record_bytes(second)
    assert canonical_run_record_bytes(first).endswith(b"\n")
    observed_payload["seed"] = 42
    with pytest.raises(MethodContractError, match="seed must be null"):
        validate_run_record(registry, observed_payload)


def test_failed_and_unexecuted_methods_are_retained_in_method_denominator() -> None:
    registry = load_method_registry(METHODS_PATH)
    failed_payload = _completed_record("maskimpute", seed=42)
    failed_payload.update(
        {
            "status": "failed",
            "output_sha256": None,
            "reason": "upstream_runtime_error",
        }
    )
    failed = validate_run_record(registry, failed_payload)

    table = build_method_status_table(registry, [failed])

    assert len(table) == len(registry.methods)
    assert {row.method_id for row in table} == set(registry.ids)
    by_id = {row.method_id: row for row in table}
    assert by_id["maskimpute"].status == "failed"
    assert by_id["maskimpute"].reason == "upstream_runtime_error"
    assert by_id["scziva"].status == "implemented"
    assert by_id["scgacl"].status == "not_applicable"
    assert by_id["scsdae"].status == "implemented"
