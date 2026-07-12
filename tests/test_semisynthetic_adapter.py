from __future__ import annotations

from dataclasses import replace
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import warnings

import numpy as np
import pytest

from maskimpute_benchmark.protocol import canonical_sha256, load_protocol
from maskimpute_benchmark.schema import make_inference_view, validate_benchmark_dataset
from maskimpute_benchmark.simulators import SimulationContractError, SimulationRequest
from maskimpute_benchmark.simulators.semisynthetic import (
    prepare_source_summary,
    run_semisynthetic_pair,
)
import maskimpute_benchmark.simulators.semisynthetic as semisynthetic_module


PROTOCOL = load_protocol(Path("study/protocol.json"))
SMOKE_PROTOCOL = replace(
    PROTOCOL,
    development=replace(PROTOCOL.development, cells=20, genes=4),
)

HUMAN_MEMBERS = (
    "GSM2230757_human1_umifm_counts.csv.gz",
    "GSM2230758_human2_umifm_counts.csv.gz",
    "GSM2230759_human3_umifm_counts.csv.gz",
    "GSM2230760_human4_umifm_counts.csv.gz",
)
MOUSE_MEMBERS = (
    "GSM2230761_mouse1_umifm_counts.csv.gz",
    "GSM2230762_mouse2_umifm_counts.csv.gz",
)
REAL_ARCHIVE = Path("artifacts/external/data/baron-pancreas-umi/GSE84133_RAW.tar")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


def _gzip_csv(donor: str) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(
        ["", "barcode", "assigned_cluster", "G1", "G2", "G3", "G4", "G5", "G6"]
    )
    for index in range(20):
        cluster = "alpha" if index < 10 else "beta"
        if cluster == "alpha":
            counts = [50 + index % 3, 2, 20, 1, 10, 3]
        else:
            counts = [2, 50 + index % 3, 15, 2, 10, 3]
        writer.writerow(
            [
                f"{donor}.cell-{index + 1:03d}",
                f"{donor}-barcode-{index + 1:03d}",
                cluster,
                *counts,
            ]
        )
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as handle:
        handle.write(text.getvalue().encode("utf-8"))
    return compressed.getvalue()


def _gzip_text(value: str) -> bytes:
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0) as handle:
        handle.write(value.encode("utf-8"))
    return compressed.getvalue()


def _write_fixture_archive(
    path: Path,
    *,
    malformed_unused_final: bool = False,
    omit_member: str | None = None,
    replacements: dict[str, bytes] | None = None,
) -> None:
    payloads: dict[str, bytes] = {}
    for member in HUMAN_MEMBERS:
        donor = member.split("_")[1]
        payloads[member] = (
            b"not-gzip"
            if malformed_unused_final and donor in {"human3", "human4"}
            else _gzip_csv(donor)
        )
    payloads.update({member: b"unused-mouse-bytes" for member in MOUSE_MEMBERS})
    if replacements is not None:
        payloads.update(replacements)
    if omit_member is not None:
        payloads.pop(omit_member)
    with tarfile.open(path, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(payload))


def _receipt(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "artifacts": [
            {
                "name": "GSE84133_RAW.tar",
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        ],
        "citation_doi": "10.1016/j.cels.2016.08.011",
        "ledger_sha256": "a" * 64,
        "license": "LicenseRef-NCBI-GEO-NoRestrictions",
        "resolved_revision": "GSE84133:2019-05-15",
        "revision": "GSE84133:2019-05-15",
        "role": "semisynthetic_source",
        "schema_version": 1,
        "source_id": "baron-pancreas-umi",
        "source_type": "data",
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133",
        "verified_checksum": None,
    }


def _mock_source(monkeypatch: object, path: Path) -> None:
    receipt = _receipt(path)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_verify_semisynthetic_source",
        lambda: (path, json.loads(json.dumps(receipt))),
    )


def _rehash_run_metadata(output_dir: Path) -> None:
    path = output_dir / "run_metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["native_file_sha256"] = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in sorted(semisynthetic_module._EXPECTED_NATIVE_FILES - {path.name})
    }
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _requests(root: Path) -> tuple[SimulationRequest, SimulationRequest]:
    moderate = SimulationRequest(
        mechanism="semisynthetic",
        namespace="dev",
        biological_id="draw-01",
        biological_seed=2**62 + 411,
        measurement_seed=2**61 + 512,
        technical_view="moderate",
        cells=20,
        genes=4,
        output_path=root / "dev/semisynthetic/draw-01-moderate.h5ad",
    )
    return moderate, replace(
        moderate,
        measurement_seed=2**61 + 613,
        technical_view="severe",
        output_path=root / "dev/semisynthetic/draw-01-severe.h5ad",
    )


def test_semisynthetic_adapter_exposes_paired_public_api() -> None:
    import maskimpute_benchmark.simulators as simulators

    assert callable(run_semisynthetic_pair)
    assert simulators.run_semisynthetic_pair is run_semisynthetic_pair


def test_development_pair_uses_only_development_donors_and_preserves_proxy_truth(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive, malformed_unused_final=True)
    _mock_source(monkeypatch, archive)
    requests = _requests(tmp_path / "results")

    artifacts = run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert [artifact.request for artifact in artifacts] == list(requests)
    for artifact in artifacts:
        dataset = artifact.adata
        assert validate_benchmark_dataset(dataset) is None
        assert dataset.shape == (20, 4)
        assert dataset.X.dtype == np.int64
        assert dataset.layers["reference_counts"].dtype == np.int64
        assert dataset.layers["heldout_counts"].dtype == np.int64
        assert dataset.var_names.tolist() == ["G1", "G2", "G3", "G5"]
        assert dataset.uns["truth_kind"] == "proxy_high_depth"
        assert dataset.uns["primary_truth_layer"] == "reference_counts"
        assert (
            dataset.uns["provenance"]["source_sha256"]
            == hashlib.sha256(archive.read_bytes()).hexdigest()
        )
        assert list(dataset.uns["allowed_covariates"]["obs"]) == []
        assert list(dataset.uns["allowed_covariates"]["var"]) == []
        parameters = dataset.uns["provenance"]["parameters"]
        assert list(parameters["source_partition"]["donors"]) == [
            "GSM2230757_human1_umifm_counts.csv.gz",
            "GSM2230758_human2_umifm_counts.csv.gz",
        ]
        assert parameters["source_partition"]["namespace"] == "dev"
        assert parameters["source_partition"]["partition_rule"] == (
            "human1+human2_development__human3+human4_final"
        )
        assert list(parameters["source_partition"]["donor_row_counts"]) == [20, 20]
        assert parameters["gene_selection"]["rule"] == (
            "pooled_total_umi_descending_then_gene_id_ascending"
        )
        assert parameters["gene_selection"]["selected_gene_ids_sha256"] == (
            canonical_sha256(["G1", "G2", "G3", "G5"])
        )
        assert parameters["metric_availability"] == {
            "mse_pre_dropout_zero": "proxy_truth_not_exact",
            "p_pre_zero_calibration": "proxy_truth_not_exact",
        }
        assert "group" in dataset.obs
        assert "group" not in make_inference_view(dataset).obs

    moderate, severe = (artifact.adata for artifact in artifacts)
    np.testing.assert_array_equal(
        moderate.layers["reference_counts"],
        severe.layers["reference_counts"],
    )
    assert not np.array_equal(moderate.X, severe.X)
    assert bool(
        (
            np.asarray(moderate.X) + np.asarray(moderate.layers["heldout_counts"])
            <= np.asarray(moderate.layers["reference_counts"])
        ).all()
    )
    assert bool(
        (
            np.asarray(severe.X) + np.asarray(severe.layers["heldout_counts"])
            <= np.asarray(severe.layers["reference_counts"])
        ).all()
    )


def test_seeded_reruns_have_identical_native_bytes_and_semantics(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    _mock_source(monkeypatch, archive)

    first = run_semisynthetic_pair(_requests(tmp_path / "first"), SMOKE_PROTOCOL)
    second = run_semisynthetic_pair(_requests(tmp_path / "second"), SMOKE_PROTOCOL)

    for first_artifact, second_artifact in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_artifact.adata.X, second_artifact.adata.X)
        np.testing.assert_array_equal(
            first_artifact.adata.layers["reference_counts"],
            second_artifact.adata.layers["reference_counts"],
        )
        np.testing.assert_array_equal(
            first_artifact.adata.layers["heldout_counts"],
            second_artifact.adata.layers["heldout_counts"],
        )
        assert (
            first_artifact.adata.uns["provenance"]["parameters"]["pair_request_sha256"]
            == second_artifact.adata.uns["provenance"]["parameters"][
                "pair_request_sha256"
            ]
        )
        assert [entry.as_dict() for entry in first_artifact.native_manifest.files] == [
            entry.as_dict() for entry in second_artifact.native_manifest.files
        ]
        assert first_artifact.native_manifest.manifest_sha256 == (
            second_artifact.native_manifest.manifest_sha256
        )
        assert first_artifact.dataset_sha256 == second_artifact.dataset_sha256
        native_paths = {
            item.logical_path: item.physical_path
            for item in first_artifact.native_manifest._sealed_files
        }
        config_bytes = native_paths["config.json"].read_bytes()
        run_metadata_bytes = native_paths["run_metadata.json"].read_bytes()
        assert tmp_path.as_posix().encode() not in config_bytes
        assert tmp_path.as_posix().encode() not in run_metadata_bytes
        assert b"runtime" not in config_bytes.lower()
        assert b"runtime" not in run_metadata_bytes.lower()


def test_source_partitions_are_exact_and_donor_disjoint(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)

    development = prepare_source_summary(archive, "dev", 4)
    final = prepare_source_summary(archive, "final", 4)

    development_donors = set(development["source_partition"]["donors"])
    final_donors = set(final["source_partition"]["donors"])
    assert development_donors == set(HUMAN_MEMBERS[:2])
    assert final_donors == set(HUMAN_MEMBERS[2:])
    assert development_donors.isdisjoint(final_donors)


def test_final_namespace_is_rejected_without_claim_before_source_access(
    tmp_path: Path, monkeypatch: object
) -> None:
    moderate = SimulationRequest(
        mechanism="semisynthetic",
        namespace="final",
        biological_id="draw-01",
        biological_seed=101,
        measurement_seed=202,
        technical_view="moderate",
        cells=PROTOCOL.final.cells,
        genes=PROTOCOL.final.genes,
        output_path=tmp_path / "final/semisynthetic/moderate.h5ad",
    )
    severe = replace(
        moderate,
        measurement_seed=303,
        technical_view="severe",
        output_path=tmp_path / "final/semisynthetic/severe.h5ad",
    )

    def source_must_not_be_read() -> object:
        raise AssertionError("source accessed before final claim validation")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_verify_semisynthetic_source",
        source_must_not_be_read,
    )

    with pytest.raises(SimulationContractError, match="final manifest claim"):
        run_semisynthetic_pair((moderate, severe), PROTOCOL)


def test_final_postpublication_path_uses_lifecycle_only_revalidation(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    _mock_source(monkeypatch, archive)
    protocol = replace(
        SMOKE_PROTOCOL,
        final=replace(SMOKE_PROTOCOL.final, cells=20, genes=4),
    )
    requests = tuple(
        replace(
            request,
            namespace="final",
            output_path=tmp_path
            / f"final/semisynthetic/draw-01-{request.technical_view}.h5ad",
        )
        for request in _requests(tmp_path)
    )
    lifecycle_calls: list[object] = []
    fake_claim = object()
    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "validate_paired_simulation_requests",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_revalidate_published_final_claim",
        lifecycle_calls.append,
    )

    run_semisynthetic_pair(requests, protocol, fake_claim)  # type: ignore[arg-type]

    assert lifecycle_calls == [fake_claim]


def test_prepare_script_writes_seed_free_canonical_fit_receipt(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    script_path = Path("scripts/prepare_semisynthetic_source.py")
    spec = importlib.util.spec_from_file_location(
        "prepare_semisynthetic_source", script_path
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    receipt = _receipt(archive)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        script,
        "_verify_semisynthetic_source",
        lambda: (archive, json.loads(json.dumps(receipt))),
    )
    output = tmp_path / "prepared.json"

    assert (
        script.main(["--namespace", "dev", "--genes", "4", "--output", str(output)])
        == 0
    )

    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert b"seed" not in raw.lower()
    assert tmp_path.as_posix().encode() not in raw
    assert b"runtime" not in raw.lower()
    parsed = json.loads(raw)
    assert parsed["source_receipt_sha256"] == canonical_sha256(receipt)
    assert parsed["fit"]["source_partition"]["donors"] == list(HUMAN_MEMBERS[:2])


def test_malformed_source_member_set_fails_before_any_publication(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive, omit_member=HUMAN_MEMBERS[1])
    _mock_source(monkeypatch, archive)
    requests = _requests(tmp_path / "results")

    with pytest.raises(SimulationContractError, match="closed donor-member set"):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert not any(request.output_path.exists() for request in requests)


@pytest.mark.parametrize(
    ("csv_text", "message"),
    [
        (
            ",barcode,cluster,G1,G2\nid-1,barcode-1,alpha,1,2\n",
            "must contain index, barcode, assigned_cluster",
        ),
        (
            ",barcode,assigned_cluster,G1,G1\nid-1,barcode-1,alpha,1,2\n",
            "unique genes",
        ),
        (
            ",barcode,assigned_cluster,G1,G2\nid-1,barcode-1,alpha,-1,2\n",
            "nonnegative integers",
        ),
        (
            ",barcode,assigned_cluster,G1,G2\nid-1,barcode-1,alpha,1.5,2\n",
            "integer counts",
        ),
        (
            ",barcode,assigned_cluster,G1,G2\nid-1,,alpha,1,2\n",
            "invalid barcodes",
        ),
    ],
)
def test_selected_donor_csv_schema_and_counts_are_strictly_validated(
    tmp_path: Path, csv_text: str, message: str
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(
        archive,
        replacements={HUMAN_MEMBERS[0]: _gzip_text(csv_text)},
    )

    with pytest.raises(SimulationContractError, match=message):
        prepare_source_summary(archive, "dev", 1)


def test_extra_native_file_is_rejected_before_publication(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    _mock_source(monkeypatch, archive)
    requests = _requests(tmp_path / "results")
    real_generate = semisynthetic_module._generate_native

    def generate_extra(fit: object, config: object, output_dir: Path) -> None:
        real_generate(fit, config, output_dir)
        (output_dir / "unexpected.txt").write_text("extra\n", encoding="utf-8")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_generate_native",
        generate_extra,
    )

    with pytest.raises(SimulationContractError, match="closed file set"):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert not any(request.output_path.exists() for request in requests)


def test_unsafe_fitted_library_distribution_fails_without_numeric_warning(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    fit = semisynthetic_module._fit_source(archive, "dev", 4)
    unsafe_fit = replace(
        fit,
        library_log_parameters=np.asarray([1000.0, 0.0], dtype="<f8"),
    )
    requests = {request.technical_view: request for request in _requests(tmp_path)}
    config = semisynthetic_module._pair_config(requests, unsafe_fit, _receipt(archive))

    with pytest.raises(SimulationContractError, match="library-size distribution"):
        semisynthetic_module._generate_reference(unsafe_fit, config)


def test_transposed_native_reference_is_rejected_before_publication(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    _mock_source(monkeypatch, archive)
    requests = _requests(tmp_path / "results")
    real_generate = semisynthetic_module._generate_native

    def generate_transposed(fit: object, config: object, output_dir: Path) -> None:
        real_generate(fit, config, output_dir)
        path = output_dir / "reference_counts.npy"
        values = np.load(path, allow_pickle=False)
        with path.open("wb") as handle:
            np.save(handle, values.T, allow_pickle=False)
        _rehash_run_metadata(output_dir)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_generate_native",
        generate_transposed,
    )

    with pytest.raises(SimulationContractError, match="orientation"):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert not any(request.output_path.exists() for request in requests)


def test_rehashed_but_semantically_forged_native_counts_are_rejected(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    _mock_source(monkeypatch, archive)
    requests = _requests(tmp_path / "results")
    real_generate = semisynthetic_module._generate_native

    def generate_forged(fit: object, config: object, output_dir: Path) -> None:
        real_generate(fit, config, output_dir)
        for name in (
            "reference_counts.npy",
            "observed_moderate.npy",
            "observed_severe.npy",
            "heldout_moderate.npy",
            "heldout_severe.npy",
        ):
            path = output_dir / name
            values = np.load(path, allow_pickle=False)
            with path.open("wb") as handle:
                np.save(handle, np.zeros_like(values), allow_pickle=False)
        _rehash_run_metadata(output_dir)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_generate_native",
        generate_forged,
    )

    with pytest.raises(SimulationContractError, match="deterministic derivation"):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert not any(request.output_path.exists() for request in requests)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("forged_versions", "environment versions"),
        ("boolean_call_count", "call counts"),
        ("noncanonical_gene_json", "canonical JSON"),
        ("boolean_gene_schema", "fit labels"),
    ],
)
def test_native_provenance_requires_exact_versions_types_and_canonical_json(
    tmp_path: Path, monkeypatch: object, mutation: str, message: str
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    _mock_source(monkeypatch, archive)
    requests = _requests(tmp_path / "results")
    real_generate = semisynthetic_module._generate_native

    def generate_forged(fit: object, config: object, output_dir: Path) -> None:
        real_generate(fit, config, output_dir)
        if mutation in {"noncanonical_gene_json", "boolean_gene_schema"}:
            path = output_dir / "gene_ids.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if mutation == "noncanonical_gene_json":
                payload = json.dumps(value, indent=2) + "\n"
            else:
                value["schema_version"] = True
                payload = (
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                )
            path.write_text(payload, encoding="utf-8")
            _rehash_run_metadata(output_dir)
            return
        path = output_dir / "run_metadata.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if mutation == "forged_versions":
            value["versions"]["numpy"] = "forged-version"
        else:
            value["fit_calls"] = True
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_generate_native",
        generate_forged,
    )

    with pytest.raises(SimulationContractError, match=message):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert not any(request.output_path.exists() for request in requests)


def test_second_result_publication_failure_rolls_back_pair_and_native_bytes(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    _write_fixture_archive(archive)
    _mock_source(monkeypatch, archive)
    requests = _requests(tmp_path / "results")
    real_publish = semisynthetic_module._publish_staged_h5ad
    calls = 0

    def fail_second(temporary: Path, destination: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SimulationContractError("injected second-publication failure")
        return real_publish(temporary, destination)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_publish_staged_h5ad",
        fail_second,
    )

    with pytest.raises(SimulationContractError, match="injected"):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert calls == 2
    assert not any(request.output_path.exists() for request in requests)
    output_parent = requests[0].output_path.parent
    assert not list(output_parent.glob("native/semisynthetic-*"))


def test_existing_result_is_rejected_before_source_access(
    tmp_path: Path, monkeypatch: object
) -> None:
    requests = _requests(tmp_path / "results")
    requests[0].output_path.parent.mkdir(parents=True)
    requests[0].output_path.write_bytes(b"existing")

    def source_must_not_be_read() -> object:
        raise AssertionError("source accessed after existing-result detection")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_verify_semisynthetic_source",
        source_must_not_be_read,
    )

    with pytest.raises(SimulationContractError, match="refuses to overwrite"):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)


def test_archive_snapshot_is_bound_to_receipt_across_source_rechecks(
    tmp_path: Path, monkeypatch: object
) -> None:
    archive = tmp_path / "GSE84133_RAW.tar"
    alternate = tmp_path / "alternate.tar"
    _write_fixture_archive(archive)
    with gzip.GzipFile(fileobj=io.BytesIO(_gzip_csv("human1")), mode="rb") as handle:
        changed_csv = (
            handle.read()
            .decode("utf-8")
            .replace(",50,2,20,1,10,3\n", ",49,2,20,1,10,3\n", 1)
        )
    _write_fixture_archive(
        alternate,
        replacements={HUMAN_MEMBERS[0]: _gzip_text(changed_csv)},
    )
    original_bytes = archive.read_bytes()
    alternate_bytes = alternate.read_bytes()
    receipt = _receipt(archive)
    calls = 0

    def swap_between_rechecks() -> tuple[Path, dict[str, object]]:
        nonlocal calls
        calls += 1
        archive.write_bytes(alternate_bytes if calls == 1 else original_bytes)
        return archive, json.loads(json.dumps(receipt))

    monkeypatch.setattr(  # type: ignore[attr-defined]
        semisynthetic_module,
        "_verify_semisynthetic_source",
        swap_between_rechecks,
    )
    requests = _requests(tmp_path / "results")

    with pytest.raises(SimulationContractError, match="snapshot checksum"):
        run_semisynthetic_pair(requests, SMOKE_PROTOCOL)

    assert calls == 2
    assert not any(request.output_path.exists() for request in requests)


@integration
@pytest.mark.skipif(
    not REAL_ARCHIVE.is_file(), reason="pinned Baron archive unavailable"
)
def test_real_pinned_baron_smoke_preserves_source_and_paired_reference(
    tmp_path: Path,
) -> None:
    before = hashlib.sha256(REAL_ARCHIVE.read_bytes()).hexdigest()
    protocol = replace(
        PROTOCOL,
        development=replace(PROTOCOL.development, cells=20, genes=20),
    )
    requests = _requests(tmp_path)
    requests = tuple(replace(request, genes=20) for request in requests)

    artifacts = run_semisynthetic_pair(requests, protocol)

    first, second = (artifact.adata for artifact in artifacts)
    assert first.shape == (20, 20)
    np.testing.assert_array_equal(
        first.layers["reference_counts"], second.layers["reference_counts"]
    )
    assert not np.array_equal(first.X, second.X)
    assert artifacts[0].adata.uns["provenance"]["source_sha256"] == before
    assert hashlib.sha256(REAL_ARCHIVE.read_bytes()).hexdigest() == before
