from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
import shutil
import importlib.util
import subprocess

import pytest


METRICS = (
    "mse",
    "mse_dropout",
    "gnrmse",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
    "null_de_fpr",
)


def _dataset_rows():
    rows = []
    for mechanism in ("symsim", "sergio", "sparsim", "semisynthetic"):
        for draw in ("draw-01", "draw-02"):
            for view in ("moderate", "severe"):
                label = f"{mechanism}:{draw}:{view}"
                dataset_id = (
                    f"dataset-{hashlib.sha256(label.encode()).hexdigest()[:24]}"
                )
                rows.append(
                    {
                        "mechanism": mechanism,
                        "biological_id": draw,
                        "technical_view": view,
                        "dataset_id": dataset_id,
                        "dataset_sha256": hashlib.sha256(
                            f"dataset:{label}".encode()
                        ).hexdigest(),
                        "status": "completed",
                    }
                )
    return rows


def _ready_repository(tmp_path: Path):
    from maskimpute.calibration import CalibrationRecord, fit_development_calibration
    from maskimpute_benchmark.selection import _canonical_sha256

    repository = tmp_path / "repository"
    for relative in (
        "study/protocol.json",
        "study/development_panel.json",
        "study/methods.json",
        "study/ablations.json",
        "study/calibration_contract.json",
        "study/selection_contract.json",
        "study/development_search.json",
    ):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, destination)
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    contract = json.loads(
        (repository / "study/selection_contract.json").read_text(encoding="utf-8")
    )
    entries = []
    for row in _dataset_rows():
        label = ":".join(
            (row["mechanism"], row["biological_id"], row["technical_view"])
        )
        entries.append(
            {
                "mechanism": row["mechanism"],
                "biological_id": row["biological_id"],
                "technical_view": row["technical_view"],
                "dataset_id": row["dataset_id"],
                "dataset_sha256": row["dataset_sha256"],
                "input_sha256": hashlib.sha256(f"input:{label}".encode()).hexdigest(),
                "cell_ids_sha256": hashlib.sha256(
                    f"cells:{row['mechanism']}:{row['biological_id']}".encode()
                ).hexdigest(),
                "excluded_cell_count": 0,
                "excluded_cell_ids_sha256": hashlib.sha256(
                    b"empty-cell-set"
                ).hexdigest(),
                "retained_cell_count": 900,
                "retained_cell_ids_sha256": hashlib.sha256(
                    f"cells:{row['mechanism']}:{row['biological_id']}".encode()
                ).hexdigest(),
                "score_sha256": hashlib.sha256(f"score:{label}".encode()).hexdigest(),
                "config_sha256": contract["count_model_config_sha256"],
            }
        )
    score_manifest = (
        repository / "artifacts/study/development/count_scores/manifest.json"
    )
    score_manifest.parent.mkdir(parents=True, exist_ok=True)
    score_core = {
        "schema_version": 1,
        "artifact_type": "maskimpute_development_count_score_manifest",
        "dataset_manifest_sha256": "a" * 64,
        "count_model_config_sha256": contract["count_model_config_sha256"],
        "dataset_qc_policy_sha256": contract["dataset_qc_policy_sha256"],
        "entries": entries,
    }
    score_payload = {
        **score_core,
        "manifest_sha256": _canonical_sha256(score_core),
    }
    score_manifest.write_text(
        json.dumps(score_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ledger["count_score_manifest"] = {
        "status": "ready",
        "path": "artifacts/study/development/count_scores/manifest.json",
        "sha256": hashlib.sha256(score_manifest.read_bytes()).hexdigest(),
    }

    score_by_unit = {
        (row["mechanism"], row["biological_id"], row["technical_view"]): row
        for row in entries
    }
    calibration_records = []
    for index, row in enumerate(
        (item for item in _dataset_rows() if item["mechanism"] == "symsim"),
        start=1,
    ):
        score = score_by_unit[
            (row["mechanism"], row["biological_id"], row["technical_view"])
        ]
        calibration_records.append(
            CalibrationRecord(
                p_pre_zero=(0.1, 0.25, 0.7, 0.9),
                target=(0, 0, 1, 1),
                mechanism="symsim",
                biological_id=row["biological_id"],
                manifest_sha256=score["score_sha256"],
                truth_kind="exact_pre_capture",
                namespace="dev",
                data_role="development",
                technical_view=row["technical_view"],
                dataset_id=row["dataset_id"],
                dataset_sha256=row["dataset_sha256"],
                protocol_sha256=ledger["authority"]["protocol_sha256"],
            )
        )
    artifact = fit_development_calibration(calibration_records)
    calibration = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    calibration.parent.mkdir(parents=True, exist_ok=True)
    calibration.write_text(
        json.dumps(artifact.to_dict(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    calibration_sha = hashlib.sha256(calibration.read_bytes()).hexdigest()
    ledger["retained_calibration_artifact"] = {
        "status": "ready",
        "path": ("artifacts/study/development/calibration/retained_calibration.json"),
        "sha256": calibration_sha,
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return repository, calibration_sha


def _status_and_payload(authority):
    from maskimpute_benchmark.selection import _canonical_sha256

    datasets = _dataset_rows()
    dataset_by_unit = {
        (row["mechanism"], row["biological_id"], row["technical_view"]): row
        for row in datasets
    }
    selected_methods = [
        declaration
        for declaration in authority.declarations
        if declaration.required_for_claim or declaration.role == "candidate"
    ]
    records = []
    for declaration in selected_methods:
        if declaration.role == "observed_control":
            base = 1.4
        elif declaration.role == "candidate":
            base = 0.8
        else:
            base = 1.0
        seeds = authority.model_seeds if declaration.stochastic else (None,)
        for metric in METRICS:
            mechanisms = (
                ("symsim",)
                if metric == "mse_pre_dropout_zero"
                else authority.mechanisms
            )
            for mechanism in mechanisms:
                for draw in authority.biological_ids:
                    for view in authority.technical_views:
                        dataset = dataset_by_unit[(mechanism, draw, view)]
                        for seed in seeds:
                            value = 0.05 if metric == "null_de_fpr" else base
                            records.append(
                                {
                                    "mechanism": mechanism,
                                    "biological_id": draw,
                                    "technical_view": view,
                                    "dataset_id": dataset["dataset_id"],
                                    "dataset_sha256": dataset["dataset_sha256"],
                                    "method": declaration.id,
                                    "method_sha256": authority.method_bindings[
                                        declaration.id
                                    ],
                                    "model_seed": seed,
                                    "metric": metric,
                                    "value": value,
                                    "status": "completed",
                                }
                            )
    intervals = [
        {
            "configuration": attempt.configuration_id,
            "endpoint": endpoint.id,
            "comparison": "observed",
            "estimate": 0.0,
            "ci_lower": -0.01,
            "ci_upper": 0.01,
            "status": "completed",
        }
        for attempt in authority.attempts
        for endpoint in authority.endpoint_policies
    ]
    manifest_sha = "a" * 64
    status = {
        "namespace": "dev",
        "status": "completed",
        "manifest_sha256": manifest_sha,
        "protocol_sha256": authority.file_sha256["study/protocol.json"],
        "rows": datasets,
    }
    core = {
        "schema_version": 2,
        "dataset_manifest_sha256": manifest_sha,
        "count_score_manifest_sha256": authority.count_score_manifest.sha256,
        "retained_calibration_artifact_sha256": (authority.retained_calibration.sha256),
        "records": records,
        "orthogonal_intervals": intervals,
    }
    payload = {**core, "result_sha256": _canonical_sha256(core)}
    return status, payload


def test_public_selection_api_accepts_results_only_not_design_authority():
    from maskimpute_benchmark.selection import select_development_candidate

    signature = inspect.signature(select_development_candidate)

    assert tuple(signature.parameters) == ("payload",)


def test_repository_authority_derives_design_methods_and_pending_calibration():
    from maskimpute_benchmark.selection import (
        _load_selection_authority,
        load_publication_execution_authority,
    )

    authority = _load_selection_authority(Path.cwd(), require_clean=False)

    assert authority.mechanisms == (
        "symsim",
        "sergio",
        "sparsim",
        "semisynthetic",
    )
    assert authority.biological_ids == ("draw-01", "draw-02")
    assert authority.technical_views == ("moderate", "severe")
    assert authority.model_seeds == (42, 43, 44)
    assert authority.required_comparator_ids == (
        "observed",
        "capacity-matched-ae",
        "alra",
        "magic",
        "dca",
        "scvi",
        "saver",
        "scziva",
        "afmf",
        "sccr",
        "scsdae",
    )
    assert authority.retained_calibration.status == "pending"
    assert authority.retained_calibration.path == (
        "artifacts/study/development/calibration/retained_calibration.json"
    )
    assert authority.retained_calibration.sha256 is None
    assert authority.count_score_manifest.status == "pending"
    assert authority.count_score_manifest.path == (
        "artifacts/study/development/count_scores/manifest.json"
    )
    assert authority.count_score_manifest.sha256 is None
    assert dict(authority.base_maskimpute_config) == {
        "hidden_dims": (128, 64),
        "latent_dim": 24,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "max_epochs": 300,
        "patience": 30,
        "artificial_mask_fraction": 0.2,
        "validation_fraction": 0.1,
        "log_count_bin_edges": (
            1.0986122886681096,
            2.1972245773362196,
            3.4965075614664802,
        ),
        "early_stopping_min_delta": 0.0,
        "pre_zero_regularization": 1.0,
        "gate_gamma": 1.0,
        "normalization_target": 10000.0,
    }
    assert dict(authority.count_model_config) == {
        "n_folds": 5,
        "use_library_size_exposure": True,
        "mean_prior_strength": 1.0,
        "mean_floor": 1e-8,
        "dispersion_prior_strength": 10.0,
        "link_bins": 64,
        "link_max_iter": 200,
        "link_tolerance": 1e-10,
        "link_bound": 30.0,
    }
    assert authority.ablation_spec_ids == (
        "maskimpute-reference",
        "capacity-matched-ae",
        "no-gate",
        "no-pre-zero-regularizer",
        "no-explicit-mask",
        "full-denoising",
        "direct-score",
    )
    assert len(authority.ablation_run_keys) == 21
    assert authority.ablation_run_keys[:3] == (
        ("maskimpute-reference", 42),
        ("maskimpute-reference", 43),
        ("maskimpute-reference", 44),
    )
    assert authority.calibration_equivalence_reason is None
    assert authority.calibration_effect_status == "pending_retained_artifact"
    assert dict(authority.dataset_qc_policy) == {
        "cell_exclusion_rule": "observed_library_size_equals_zero",
        "minimum_retained_cells": 2,
        "application": (
            "pre_dispatch_pair_union_zero_library_identical_cell_subset_all_methods"
        ),
        "additional_cell_filtering": "forbidden",
        "gene_filtering": "forbidden",
        "required_audit_fields": (
            "excluded_cell_count",
            "excluded_cell_ids_sha256",
            "retained_cell_count",
            "retained_cell_ids_sha256",
        ),
    }
    assert authority.dataset_qc_policy_sha256 == (
        "81dc2ecd1749d9390e499ae21fabb8d3b08f40eec58334c860cd9a23dd4fc2d7"
    )
    assert (
        tuple(inspect.signature(load_publication_execution_authority).parameters) == ()
    )
    assert authority.file_sha256["study/methods.json"] == (
        "a8d1d1c7bc83cdc26c2c0570d34376efece52e973f3c24dda5aced28d45423f9"
    )
    assert authority.file_sha256["study/ablations.json"] == (
        "dd4da34e0ebe5e7eb349fac3ed89063781bcddf640b01601b9a3c82a2e43b26f"
    )
    assert authority.file_sha256["study/calibration_contract.json"] == (
        "c1cb47b86e1132ef080830c6b58bf7fa4aac524ca832a9e7e55b81d41fb41ef0"
    )
    assert len(authority.attempts) == 20
    assert tuple(item.configuration_id for item in authority.attempts)[:2] == (
        "v27-c01-direct-r1-g1",
        "v27-c02-calibrated-r1-g0p5",
    )
    assert tuple(item.configuration_id for item in authority.attempts)[-2:] == (
        "v27-c19-calibrated-r10-g2",
        "v27-c20-calibrated-r10-g3",
    )
    assert tuple(item.configuration_id for item in authority.exclusions) == (
        "v27-c21-calibrated-r10-g4",
        "v27-c22-calibrated-r10-g6",
    )
    assert all(
        item.reason_code == "exploratory_budget_overrun_not_selection_eligible"
        for item in authority.exclusions
    )


def test_public_selection_blocks_until_the_ledger_binds_retained_calibration():
    from maskimpute_benchmark.selection import (
        SelectionAuthorityError,
        _select_for_repository,
    )

    with pytest.raises(SelectionAuthorityError, match="calibration.*pending"):
        _select_for_repository({}, Path.cwd(), require_clean=False)


def test_ready_public_selection_binds_results_to_all_repository_authorities(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, calibration_sha = _ready_repository(tmp_path)
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    report = selection._select_for_repository(
        payload,
        repository,
        require_clean=False,
    )

    assert report.selected_configuration == "v27-c01-direct-r1-g1"
    assert tuple(item.configuration_id for item in report.excluded_configurations) == (
        "v27-c21-calibrated-r10-g4",
        "v27-c22-calibrated-r10-g6",
    )
    assert report.authority_bindings is not None
    assert report.authority_bindings["retained_calibration_artifact_sha256"] == (
        calibration_sha
    )
    assert (
        report.authority_bindings["development_result_sha256"]
        == payload["result_sha256"]
    )
    assert (
        report.authority_bindings["dataset_manifest_sha256"]
        == (status["manifest_sha256"])
    )
    calibration_payload = json.loads(
        (
            repository
            / "artifacts/study/development/calibration/retained_calibration.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        report.authority_bindings["retained_calibration_algorithm"]
        == (calibration_payload["selected_algorithm"])
    )
    if calibration_payload["selected_algorithm"] == "identity":
        assert report.authority_bindings["calibration_equivalence_reason"] == (
            "retained_identity_calibrator_equals_direct_score"
        )
    else:
        assert report.authority_bindings["calibration_equivalence_reason"] == (
            "retained_nonidentity_calibrator_transformed_score"
        )


def test_selection_blocks_if_count_score_manifest_binding_is_pending(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["count_score_manifest"] = {
        "status": "pending",
        "path": "artifacts/study/development/count_scores/manifest.json",
        "sha256": None,
    }
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(selection.SelectionAuthorityError, match="count-score.*pending"):
        selection._select_for_repository({}, repository, require_clean=False)


def test_schema_valid_count_score_manifest_cannot_change_the_frozen_config(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    manifest_path = (
        repository / "artifacts/study/development/count_scores/manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["config_sha256"] = "0" * 64
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = selection._canonical_sha256(core)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["count_score_manifest"]["sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match="configuration"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_schema_valid_calibration_cannot_invent_dataset_provenance(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    calibration_path = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["training"]["record_bindings"][0]["dataset_sha256"] = "0" * 64
    unsigned = {
        key: value for key, value in calibration.items() if key != "payload_sha256"
    }
    canonical = (
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    calibration["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["retained_calibration_artifact"]["sha256"] = hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    authority = selection._load_selection_authority(repository, require_clean=False)
    status, payload = _status_and_payload(authority)
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )

    with pytest.raises(selection.SelectionAuthorityError, match="score/dataset panel"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_cli_forwards_results_without_reconstructing_caller_design(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "select_development_candidate_script",
        Path("scripts/select_development_candidate.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sentinel = {
        "schema_version": 2,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "b" * 64,
        "retained_calibration_artifact_sha256": "c" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "result_sha256": "d" * 64,
    }
    input_path = tmp_path / "selection-input.json"
    input_path.write_text(json.dumps(sentinel), encoding="utf-8")
    loaded = script._load(input_path)

    class Report:
        def to_dict(self):
            return {"selected": sentinel}

    monkeypatch.setattr(
        script,
        "select_development_candidate",
        lambda payload: Report() if payload is loaded else None,
    )

    assert script._report(loaded) == {"selected": sentinel}


def test_result_payload_cannot_supply_attempts_declarations_or_design(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    payload = {
        "schema_version": 2,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "c" * 64,
        "retained_calibration_artifact_sha256": "d" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "attempts": [],
        "declarations": [],
        "design": {},
        "result_sha256": "b" * 64,
    }

    with pytest.raises(selection.SelectionAuthorityError, match="missing or extra"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_result_payload_checksum_is_verified_before_dataset_access(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    payload = {
        "schema_version": 2,
        "dataset_manifest_sha256": "a" * 64,
        "count_score_manifest_sha256": "c" * 64,
        "retained_calibration_artifact_sha256": "d" * 64,
        "records": [],
        "orthogonal_intervals": [],
        "result_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: pytest.fail("dataset status should not be read"),
    )

    with pytest.raises(selection.SelectionAuthorityError, match="result checksum"):
        selection._select_for_repository(payload, repository, require_clean=False)


def test_authority_loader_rejects_duplicate_json_keys(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    contract = repository / "study/selection_contract.json"
    text = contract.read_text(encoding="utf-8")
    contract.write_text(
        text.replace(
            '"schema_version": 1,',
            '"schema_version": 1,\n  "schema_version": 1,',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(selection.SelectionAuthorityError, match="duplicate JSON key"):
        selection._load_selection_authority(repository, require_clean=False)


def test_authority_must_be_tracked_and_clean_for_public_selection(tmp_path):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.org"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Selection Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "study"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "authority"], check=True
    )

    selection._load_selection_authority(repository, require_clean=True)
    contract = repository / "study/selection_contract.json"
    contract.write_text(contract.read_text(encoding="utf-8") + "\n")

    with pytest.raises(selection.SelectionAuthorityError, match="differs"):
        selection._load_selection_authority(repository, require_clean=True)


def test_finalizer_validates_both_artifacts_before_atomically_marking_ready(
    tmp_path, monkeypatch
):
    import maskimpute_benchmark.selection as selection

    repository, _calibration_sha = _ready_repository(tmp_path)
    ready_authority = selection._load_selection_authority(
        repository, require_clean=False
    )
    status, _payload = _status_and_payload(ready_authority)
    ledger_path = repository / "study/development_search.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    for field in ("count_score_manifest", "retained_calibration_artifact"):
        ledger[field]["status"] = "pending"
        ledger[field]["sha256"] = None
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(
        selection,
        "_validate_development_dataset_status",
        lambda _repository: status,
    )
    score_path = repository / "artifacts/study/development/count_scores/manifest.json"
    calibration_path = (
        repository / "artifacts/study/development/calibration/retained_calibration.json"
    )
    monkeypatch.setattr(
        selection,
        "_revalidate_development_score_preparation",
        lambda _repository: {
            "status": "reused",
            "count_score_manifest_file_sha256": hashlib.sha256(
                score_path.read_bytes()
            ).hexdigest(),
            "calibration_file_sha256": hashlib.sha256(
                calibration_path.read_bytes()
            ).hexdigest(),
        },
    )

    finalized = selection._finalize_development_artifact_bindings_for_repository(
        repository,
        require_clean=False,
    )

    updated = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert updated["count_score_manifest"]["status"] == "ready"
    assert updated["retained_calibration_artifact"]["status"] == "ready"
    assert (
        updated["count_score_manifest"]["sha256"]
        == finalized["count_score_manifest_sha256"]
    )
    assert (
        updated["retained_calibration_artifact"]["sha256"]
        == finalized["retained_calibration_artifact_sha256"]
    )
    assert finalized["next_required_action"] == "commit_development_search_ledger"


def test_public_finalizer_has_no_caller_controlled_paths_or_hashes():
    from maskimpute_benchmark.selection import (
        finalize_development_artifact_bindings,
    )

    assert (
        tuple(inspect.signature(finalize_development_artifact_bindings).parameters)
        == ()
    )


def test_finalization_cli_exposes_no_path_or_hash_arguments(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "finalize_development_authority_script",
        Path("scripts/finalize_development_authority.py"),
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(
        script,
        "finalize_development_artifact_bindings",
        lambda: {"next_required_action": "commit_development_search_ledger"},
    )

    assert script._finalize() == {
        "next_required_action": "commit_development_search_ledger"
    }
