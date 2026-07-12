from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest


MECHANISMS = ("symsim", "sergio", "sparsim", "semisynthetic")
VIEWS = ("moderate", "severe")


def _canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _method_input(counts, cell_ids, gene_ids, dataset_sha):
    from maskimpute_benchmark.methods import prepare_method_input

    view = ad.AnnData(
        X=np.asarray(counts, dtype=np.int64),
        obs=pd.DataFrame(index=list(cell_ids)),
        var=pd.DataFrame(index=list(gene_ids)),
    )
    view.uns["source_dataset_sha256"] = dataset_sha
    view.uns["allowed_covariates"] = {"obs": [], "var": []}
    return prepare_method_input(view)


def _prepared_panel():
    from maskimpute_benchmark.runner import (
        DatasetBinding,
        DatasetQCAudit,
        PreparedDataset,
    )
    from maskimpute_benchmark.schema import benchmark_dataset_sha256
    from maskimpute_benchmark.development_scores import canonical_cell_ids_sha256

    counts = np.array(
        [[1, 0, 2], [0, 2, 1], [1, 1, 0], [2, 0, 1]],
        dtype=np.int64,
    )
    truth = np.array(
        [[1, 0, 2], [1, 2, 1], [1, 1, 1], [2, 0, 1]],
        dtype=np.int64,
    )
    cell_ids = tuple(f"cell-{index + 1}" for index in range(len(counts)))
    gene_ids = ("gene-1", "gene-2", "gene-3")
    audit = DatasetQCAudit(
        excluded_cell_count=0,
        excluded_cell_ids_sha256=canonical_cell_ids_sha256(()),
        retained_cell_count=len(cell_ids),
        retained_cell_ids_sha256=canonical_cell_ids_sha256(cell_ids),
        excluded_cell_ids=(),
        retained_cell_ids=cell_ids,
    )
    result = []
    ordinal = 0
    for mechanism in MECHANISMS:
        for draw_index in (1, 2):
            biological_id = f"draw-{draw_index:02d}"
            truth_sha = hashlib.sha256(
                f"truth:{mechanism}:{biological_id}".encode()
            ).hexdigest()
            independent_id = f"biological-{mechanism}-{draw_index:02d}"
            for view in VIEWS:
                ordinal += 1
                local_counts = counts.copy()
                local_counts[0, 0] += ordinal
                dataset_id = f"dataset-{ordinal:024x}"
                evaluator = ad.AnnData(
                    X=local_counts,
                    obs=pd.DataFrame(
                        {
                            "dataset_id": [dataset_id] * len(counts),
                            "mechanism": [mechanism] * len(counts),
                            "biological_id": [biological_id] * len(counts),
                            "technical_view": [view] * len(counts),
                            "library_size": local_counts.sum(axis=1),
                        },
                        index=list(cell_ids),
                    ),
                    var=pd.DataFrame(index=list(gene_ids)),
                )
                if mechanism == "symsim":
                    evaluator.layers["pre_capture_counts"] = truth.copy()
                    evaluator.uns["truth_kind"] = "exact_pre_capture"
                    evaluator.uns["primary_truth_layer"] = "pre_capture_counts"
                dataset_sha = benchmark_dataset_sha256(
                    _benchmark_dataset(evaluator, mechanism, view, truth)
                )
                binding = DatasetBinding(
                    mechanism=mechanism,
                    biological_id=biological_id,
                    technical_view=view,
                    dataset_id=dataset_id,
                    dataset_sha256=dataset_sha,
                    output_file_sha256=f"{ordinal + 100:064x}",
                    truth_sha256=truth_sha,
                    output_path=(
                        f"dev/datasets/{mechanism}/{biological_id}/{view}.h5ad"
                    ),
                    independent_unit_id=independent_id,
                    cells=len(counts),
                    genes=len(gene_ids),
                    manifest_sha256="a" * 64,
                    protocol_sha256=(
                        "7cfa1b55458b5b2bc4c22e3a155086724586d95df40aa61c4b78b1a779794249"
                    ),
                    design_sha256="b" * 64,
                    seed_source_sha256="c" * 64,
                )
                method_input = _method_input(
                    local_counts,
                    cell_ids,
                    gene_ids,
                    dataset_sha,
                )
                result.append(
                    PreparedDataset(
                        binding=binding,
                        audit=audit,
                        method_input=method_input,
                        evaluator_dataset=evaluator,
                    )
                )
    return tuple(result)


def _benchmark_dataset(evaluator, mechanism, view, truth):
    dataset = evaluator.copy()
    dataset.obs["condition"] = view
    dataset.obs["draw"] = int(dataset.obs["biological_id"].iloc[0].split("-")[1])
    dataset.uns["allowed_covariates"] = {"obs": [], "var": []}
    dataset.uns["provenance"] = {
        "source": "test",
        "source_sha256": "d" * 64,
        "software": "test",
        "software_version": "1",
        "parameters": {},
        "seeds": {},
    }
    if mechanism == "symsim":
        dataset.layers["pre_capture_counts"] = truth.copy()
        dataset.uns["truth_kind"] = "exact_pre_capture"
        dataset.uns["primary_truth_layer"] = "pre_capture_counts"
    else:
        dataset.layers["pre_dropout_expression"] = np.asarray(
            truth,
            dtype=np.float64,
        )
        dataset.uns["truth_kind"] = "exact_continuous"
        dataset.uns["primary_truth_layer"] = "pre_dropout_expression"
    return dataset


def test_complete_count_score_artifact_roundtrip_and_tamper_rejection(tmp_path):
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model
    from maskimpute_benchmark.development_scores import (
        DevelopmentScorePreparationError,
        load_count_score_artifact,
        save_count_score_artifact,
    )

    counts = np.array([[2, 0, 1], [0, 3, 0], [1, 1, 0], [0, 2, 2]])
    cell_ids = ("cell-a", "cell-b", "cell-c", "cell-d")
    score = fit_p_pre_zero_count_model(
        counts,
        cell_ids,
        PreZeroCountModelConfig(n_folds=2, link_max_iter=25),
    )
    path = tmp_path / "score.bin"

    save_count_score_artifact(path, score)
    loaded = load_count_score_artifact(path)

    assert loaded.manifest == score.manifest
    np.testing.assert_array_equal(loaded.p_pre_zero, score.p_pre_zero)
    np.testing.assert_array_equal(loaded.mu, score.mu)
    np.testing.assert_array_equal(loaded.alpha, score.alpha)
    np.testing.assert_array_equal(loaded.pi, score.pi)
    np.testing.assert_array_equal(loaded.fold_ids, score.fold_ids)
    assert [fold.fold_id for fold in loaded.fold_models] == [
        fold.fold_id for fold in score.fold_models
    ]
    for loaded_fold, expected_fold in zip(
        loaded.fold_models,
        score.fold_models,
        strict=True,
    ):
        np.testing.assert_array_equal(loaded_fold.gene_means, expected_fold.gene_means)
        np.testing.assert_array_equal(
            loaded_fold.gene_dispersion,
            expected_fold.gene_dispersion,
        )

    tampered = bytearray(path.read_bytes())
    tampered[-1] ^= 1
    path.write_bytes(tampered)
    with pytest.raises(
        DevelopmentScorePreparationError, match="checksum|artifact|array"
    ):
        load_count_score_artifact(path)


def test_pair_union_qc_excludes_a_cell_zero_in_only_one_view_without_gene_filtering():
    from maskimpute_benchmark.runner import (
        DatasetQCPolicy,
        prepare_dataset_pair_for_execution,
        validate_development_manifest_payload,
    )
    from maskimpute_benchmark.schema import benchmark_dataset_sha256
    from tests.test_benchmark_runner import _manifest_payload, _truth_dataset

    moderate = _truth_dataset(np.array([[1, 0, 1], [2, 0, 1], [0, 3, 1], [1, 1, 1]]))
    severe = _truth_dataset(np.array([[1, 0, 1], [0, 0, 0], [0, 3, 1], [1, 1, 1]]))
    severe.obs["dataset_id"] = "dataset-test-severe"
    severe.obs["condition"] = "severe"
    severe.obs["technical_view"] = "severe"
    bindings = validate_development_manifest_payload(_manifest_payload())[:2]
    first = replace(
        bindings[0],
        cells=4,
        genes=3,
        dataset_id="dataset-test",
        dataset_sha256=benchmark_dataset_sha256(moderate),
        truth_sha256="f" * 64,
    )
    second = replace(
        bindings[1],
        cells=4,
        genes=3,
        dataset_id="dataset-test-severe",
        dataset_sha256=benchmark_dataset_sha256(severe),
        truth_sha256="f" * 64,
    )

    prepared = prepare_dataset_pair_for_execution(
        moderate,
        severe,
        first,
        second,
        DatasetQCPolicy.fixed(),
    )

    assert prepared[0].audit.excluded_cell_ids == ("cell-2",)
    assert prepared[0].audit == prepared[1].audit
    assert prepared[0].method_input.obs_ids == ("cell-1", "cell-3", "cell-4")
    assert prepared[0].method_input.var_ids == ("gene-1", "gene-2", "gene-3")
    assert prepared[1].method_input.var_ids == prepared[0].method_input.var_ids


def test_count_score_fit_boundary_receives_truth_free_counts_only(monkeypatch):
    import maskimpute_benchmark.development_scores as module
    from maskimpute import PreZeroCountModelConfig, fit_p_pre_zero_count_model

    prepared = _prepared_panel()[0]
    seen = {}

    def capturing_fitter(counts, cell_ids, config):
        seen["counts"] = np.array(counts, copy=True)
        seen["cell_ids"] = tuple(cell_ids)
        seen["config"] = config
        assert not hasattr(counts, "layers")
        return fit_p_pre_zero_count_model(counts, cell_ids, config)

    monkeypatch.setattr(module, "fit_p_pre_zero_count_model", capturing_fitter)
    score = module.fit_prepared_count_score(
        prepared,
        PreZeroCountModelConfig(n_folds=2, link_max_iter=25),
    )

    np.testing.assert_array_equal(seen["counts"], prepared.method_input.counts)
    assert seen["cell_ids"] == prepared.method_input.obs_ids
    assert score.shape == prepared.method_input.shape
    assert not hasattr(prepared.method_input, "layers")
    assert np.max(prepared.evaluator_dataset.layers["pre_capture_counts"]) > 0


def test_prevalidated_pipeline_is_canonical_idempotent_and_fails_on_tamper(tmp_path):
    from maskimpute import PreZeroCountModelConfig
    from maskimpute_benchmark.development_scores import (
        DevelopmentScorePreparationError,
        load_count_score_artifact,
        prepare_validated_development_scores,
    )

    panel = _prepared_panel()
    config = PreZeroCountModelConfig(n_folds=2, link_max_iter=25)
    config_payload = {
        "dispersion_prior_strength": config.dispersion_prior_strength,
        "link_bins": config.link_bins,
        "link_bound": config.link_bound,
        "link_max_iter": config.link_max_iter,
        "link_tolerance": config.link_tolerance,
        "mean_floor": config.mean_floor,
        "mean_prior_strength": config.mean_prior_strength,
        "n_folds": config.n_folds,
        "use_library_size_exposure": config.use_library_size_exposure,
    }
    config_sha = _canonical_sha(config_payload)

    first = prepare_validated_development_scores(
        tmp_path,
        prepared_datasets=panel,
        dataset_manifest_sha256="a" * 64,
        count_model_config=config,
        count_model_config_sha256=config_sha,
        dataset_qc_policy_sha256="e" * 64,
    )
    second = prepare_validated_development_scores(
        tmp_path,
        prepared_datasets=panel,
        dataset_manifest_sha256="a" * 64,
        count_model_config=config,
        count_model_config_sha256=config_sha,
        dataset_qc_policy_sha256="e" * 64,
    )

    assert first["status"] == "created"
    assert second["status"] == "reused"
    assert (
        first["count_score_manifest_file_sha256"]
        == second["count_score_manifest_file_sha256"]
    )
    manifest_path = tmp_path / "artifacts/study/development/count_scores/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) == {
        "schema_version",
        "artifact_type",
        "dataset_manifest_sha256",
        "count_model_config_sha256",
        "dataset_qc_policy_sha256",
        "entries",
        "manifest_sha256",
    }
    assert len(manifest["entries"]) == 16
    assert [
        (row["mechanism"], row["biological_id"], row["technical_view"])
        for row in manifest["entries"]
    ] == [
        (mechanism, f"draw-{draw:02d}", view)
        for mechanism in MECHANISMS
        for draw in (1, 2)
        for view in VIEWS
    ]
    assert manifest["manifest_sha256"] == _canonical_sha(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    entry = manifest["entries"][0]
    assert set(entry) == {
        "mechanism",
        "biological_id",
        "technical_view",
        "dataset_id",
        "dataset_sha256",
        "input_sha256",
        "cell_ids_sha256",
        "excluded_cell_count",
        "excluded_cell_ids_sha256",
        "retained_cell_count",
        "retained_cell_ids_sha256",
        "score_sha256",
        "config_sha256",
    }
    assert entry["cell_ids_sha256"] == entry["retained_cell_ids_sha256"]
    score_files = sorted(
        (tmp_path / "artifacts/study/development/count_scores").glob("*.score")
    )
    assert len(score_files) == 16
    first_score = (
        tmp_path / "artifacts/study/development/count_scores/"
        "symsim--draw-01--moderate.score"
    )
    assert load_count_score_artifact(first_score).score_sha256 == entry["score_sha256"]
    calibration_path = (
        tmp_path / "artifacts/study/development/calibration/retained_calibration.json"
    )
    calibration = json.loads(calibration_path.read_text())
    assert calibration["schema_version"] == 3
    assert calibration["training"]["record_count"] == 4
    assert len(calibration["development_holdout_calibrators"]) == 2
    assert {
        value["biological_id"]
        for value in calibration["development_holdout_calibrators"]
    } == {"draw-01", "draw-02"}
    score_entries = {
        (row["mechanism"], row["biological_id"], row["technical_view"]): row
        for row in manifest["entries"]
    }
    bindings = calibration["training"]["record_bindings"]
    assert len(bindings) == 4
    for binding in bindings:
        key = (
            binding["mechanism"],
            binding["biological_id"],
            binding["technical_view"],
        )
        score_entry = score_entries[key]
        assert binding["namespace"] == "dev"
        assert binding["data_role"] == "development"
        assert binding["dataset_id"] == score_entry["dataset_id"]
        assert binding["dataset_sha256"] == score_entry["dataset_sha256"]
        assert binding["manifest_sha256"] == score_entry["score_sha256"]
        assert binding["protocol_sha256"] == (
            "7cfa1b55458b5b2bc4c22e3a155086724586d95df40aa61c4b78b1a779794249"
        )

    tampered = bytearray(first_score.read_bytes())
    tampered[-1] ^= 1
    first_score.write_bytes(tampered)
    with pytest.raises(
        DevelopmentScorePreparationError, match="existing|tamper|artifact"
    ):
        prepare_validated_development_scores(
            tmp_path,
            prepared_datasets=panel,
            dataset_manifest_sha256="a" * 64,
            count_model_config=config,
            count_model_config_sha256=config_sha,
            dataset_qc_policy_sha256="e" * 64,
        )


def test_partial_existing_output_fails_closed(tmp_path):
    from maskimpute import PreZeroCountModelConfig
    from maskimpute_benchmark.development_scores import (
        DevelopmentScorePreparationError,
        prepare_validated_development_scores,
    )

    partial = tmp_path / "artifacts/study/development/count_scores"
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text("{}\n")
    config = PreZeroCountModelConfig(n_folds=2, link_max_iter=25)
    config_sha = _canonical_sha(
        {
            "dispersion_prior_strength": config.dispersion_prior_strength,
            "link_bins": config.link_bins,
            "link_bound": config.link_bound,
            "link_max_iter": config.link_max_iter,
            "link_tolerance": config.link_tolerance,
            "mean_floor": config.mean_floor,
            "mean_prior_strength": config.mean_prior_strength,
            "n_folds": config.n_folds,
            "use_library_size_exposure": config.use_library_size_exposure,
        }
    )

    with pytest.raises(DevelopmentScorePreparationError, match="partial|existing"):
        prepare_validated_development_scores(
            tmp_path,
            prepared_datasets=_prepared_panel(),
            dataset_manifest_sha256="a" * 64,
            count_model_config=config,
            count_model_config_sha256=config_sha,
            dataset_qc_policy_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    "attack",
    ("symlink-directory", "symlink-manifest", "hardlink-manifest", "extra-fifo"),
)
def test_existing_output_rejects_non_owned_or_special_inventory(tmp_path, attack):
    import os
    import shutil

    from maskimpute import PreZeroCountModelConfig
    from maskimpute_benchmark.development_scores import (
        DevelopmentScorePreparationError,
        prepare_validated_development_scores,
    )

    panel = _prepared_panel()
    config = PreZeroCountModelConfig(n_folds=2, link_max_iter=25)
    config_sha = _canonical_sha(
        {
            "dispersion_prior_strength": config.dispersion_prior_strength,
            "link_bins": config.link_bins,
            "link_bound": config.link_bound,
            "link_max_iter": config.link_max_iter,
            "link_tolerance": config.link_tolerance,
            "mean_floor": config.mean_floor,
            "mean_prior_strength": config.mean_prior_strength,
            "n_folds": config.n_folds,
            "use_library_size_exposure": config.use_library_size_exposure,
        }
    )
    source = tmp_path / "source"
    prepare_validated_development_scores(
        source,
        prepared_datasets=panel,
        dataset_manifest_sha256="a" * 64,
        count_model_config=config,
        count_model_config_sha256=config_sha,
        dataset_qc_policy_sha256="e" * 64,
    )
    attacked = tmp_path / "attacked"
    source_development = source / "artifacts/study/development"
    attacked_development = attacked / "artifacts/study/development"
    attacked_development.parent.mkdir(parents=True)
    shutil.copytree(source_development, attacked_development)
    count_directory = attacked_development / "count_scores"
    manifest = count_directory / "manifest.json"

    if attack == "symlink-directory":
        shutil.rmtree(count_directory)
        count_directory.symlink_to(source_development / "count_scores")
    elif attack == "symlink-manifest":
        manifest.unlink()
        manifest.symlink_to(source_development / "count_scores/manifest.json")
    elif attack == "hardlink-manifest":
        manifest.unlink()
        os.link(source_development / "count_scores/manifest.json", manifest)
    else:
        os.mkfifo(count_directory / "unexpected.fifo")

    with pytest.raises(
        DevelopmentScorePreparationError,
        match="existing|inventory|regular|directory",
    ):
        prepare_validated_development_scores(
            attacked,
            prepared_datasets=panel,
            dataset_manifest_sha256="a" * 64,
            count_model_config=config,
            count_model_config_sha256=config_sha,
            dataset_qc_policy_sha256="e" * 64,
        )
